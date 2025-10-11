use std::collections::HashSet;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use reqwest::StatusCode;
use serde_json::json;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;
use tracing::warn;

use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::error::CodexErr;
use crate::error::Result;
use crate::error::RetryLimitReachedError;
use crate::error::UnexpectedResponseError;
use crate::model_family::ModelFamily;
use crate::openai_tools::create_tools_json_for_chat_completions_api;
use crate::tool_arguments::repair_tool_arguments;
use crate::util::backoff;
use crate::ModelProviderInfo;
use codex_otel::otel_event_manager::OtelEventManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;

fn canonicalize_args(raw: &str) -> String {
    // сначала пытаемся починить аргументы
    let fixed = repair_tool_arguments(raw).unwrap_or_else(|| raw.to_string());
    // затем пробуем распарсить и сериализовать обратно, чтобы нормализовать порядок ключей
    if let Ok(v) = serde_json::from_str::<Value>(&fixed) {
        serde_json::to_string(&v).unwrap_or(fixed)
    } else {
        fixed
    }
}

/// Попытаться «отремонтировать» arguments для конкретного инструмента и гарантировать,
/// что на выходе — РОВНО один JSON-объект.
fn try_fix_arguments(name: &str, raw: &str) -> std::result::Result<String, String> {
    use serde_json::Value;

    // 0) как есть
    let mut s = raw.to_string();

    // 1) универсальный ремонт (склейки, запятые и т.п.)
    if serde_json::from_str::<Value>(&s).is_err() {
        if let Some(fixed) = repair_tool_arguments(&s) {
            s = fixed;
        }
    }

    // 2) спец-обработка по имени инструмента
    match name {
        "update_plan" => {
            // приводим к ожидаемому объекту + нормализуем `plan`
            s = fix_update_plan_args(&s).map_err(|e| format!("update_plan: {e}"))?;
        }
        "shell" => {
            if let Some(fixed) = fix_shell_args(&s) {
                s = fixed;
            }
        }
        _ => {}
    }

    // 3) финальная проверка — ровно один объект
    validate_single_json_object_str(&s).map_err(|e| format!("arguments invalid: {e}"))?;
    Ok(s)
}

/// Implementation for the classic Chat Completions API.
pub(crate) async fn stream_chat_completions(
    prompt: &Prompt,
    model_family: &ModelFamily,
    client: &reqwest::Client,
    provider: &ModelProviderInfo,
    otel_event_manager: &OtelEventManager,
) -> Result<ResponseStream> {
    if prompt.output_schema.is_some() {
        return Err(CodexErr::UnsupportedOperation(
            "output_schema is not supported for Chat Completions API".to_string(),
        ));
    }

    // Build messages array
    let mut messages = Vec::<serde_json::Value>::new();

    let full_instructions = prompt.get_full_instructions(model_family);
    messages.push(json!({"role": "system", "content": full_instructions}));

    let input = prompt.get_formatted_input();

    // Pre-scan: map Reasoning blocks to the adjacent assistant anchor after the last user.
    let mut reasoning_by_anchor_index: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();

    // Determine the last role that would be emitted to Chat Completions.
    let mut last_emitted_role: Option<&str> = None;
    for item in &input {
        match item {
            ResponseItem::Message { role, .. } => last_emitted_role = Some(role.as_str()),
            ResponseItem::FunctionCall { .. } | ResponseItem::LocalShellCall { .. } => {
                last_emitted_role = Some("assistant")
            }
            ResponseItem::FunctionCallOutput { .. } => last_emitted_role = Some("tool"),
            ResponseItem::Reasoning { .. } | ResponseItem::Other => {}
            ResponseItem::CustomToolCall { .. } => {}
            ResponseItem::CustomToolCallOutput { .. } => {}
            ResponseItem::WebSearchCall { .. } => {}
        }
    }

    // Find the last user message index in the input.
    let mut last_user_index: Option<usize> = None;
    for (idx, item) in input.iter().enumerate() {
        if let ResponseItem::Message { role, .. } = item
            && role == "user"
        {
            last_user_index = Some(idx);
        }
    }

    // Attach reasoning only if the conversation does not end with a user message.
    if !matches!(last_emitted_role, Some("user")) {
        for (idx, item) in input.iter().enumerate() {
            if let Some(u_idx) = last_user_index
                && idx <= u_idx
            {
                continue;
            }

            if let ResponseItem::Reasoning {
                content: Some(items),
                ..
            } = item
            {
                let mut text = String::new();
                for c in items {
                    match c {
                        ReasoningItemContent::ReasoningText { text: t }
                        | ReasoningItemContent::Text { text: t } => text.push_str(t),
                    }
                }
                if text.trim().is_empty() {
                    continue;
                }

                // Prefer immediate previous assistant message
                let mut attached = false;
                if idx > 0
                    && let ResponseItem::Message { role, .. } = &input[idx - 1]
                    && role == "assistant"
                {
                    reasoning_by_anchor_index
                        .entry(idx - 1)
                        .and_modify(|v| v.push_str(&text))
                        .or_insert(text.clone());
                    attached = true;
                }

                // Otherwise, attach to immediate next assistant anchor
                if !attached && idx + 1 < input.len() {
                    match &input[idx + 1] {
                        ResponseItem::FunctionCall { .. } | ResponseItem::LocalShellCall { .. } => {
                            reasoning_by_anchor_index
                                .entry(idx + 1)
                                .and_modify(|v| v.push_str(&text))
                                .or_insert(text.clone());
                        }
                        ResponseItem::Message { role, .. } if role == "assistant" => {
                            reasoning_by_anchor_index
                                .entry(idx + 1)
                                .and_modify(|v| v.push_str(&text))
                                .or_insert(text.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Track last assistant text we emitted to avoid duplicate assistant messages
    let mut last_assistant_text: Option<String> = None;

    for (idx, item) in input.iter().enumerate() {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let mut text = String::new();
                for c in content {
                    match c {
                        ContentItem::InputText { text: t }
                        | ContentItem::OutputText { text: t } => {
                            text.push_str(t);
                        }
                        _ => {}
                    }
                }
                if role == "assistant" {
                    if let Some(prev) = &last_assistant_text
                        && prev == &text
                    {
                        continue;
                    }
                    last_assistant_text = Some(text.clone());
                }

                let mut msg = json!({"role": role, "content": text});
                if role == "assistant"
                    && let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                    && let Some(obj) = msg.as_object_mut()
                {
                    obj.insert("reasoning".to_string(), json!(reasoning));
                }
                messages.push(msg);
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let mut args = arguments.clone();
                if name == "update_plan" {
                    if let Ok(new_args) = fix_update_plan_args(&args) {
                        args = new_args;
                    }
                }
                // Добавляем tool_call ТОЛЬКО если arguments — валидный JSON-объект.
                if is_valid_json_object_str(&args) {
                    let mut msg = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args,
                            }
                        }]
                    });
                    if let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        obj.insert("reasoning".to_string(), json!(reasoning));
                    }
                    messages.push(msg);
                } else {
                    messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "Tool call `{}` was omitted because `arguments` was not a single valid JSON object. \
                             Please re-emit the call with a valid JSON object only.",
                            name
                        )
                    }));
                }
            }

            ResponseItem::LocalShellCall {
                id,
                call_id: _,
                status,
                action,
            } => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id.clone().unwrap_or_else(|| "".to_string()),
                        "type": "local_shell_call",
                        "status": status,
                        "action": action,
                    }]
                });
                if let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                    && let Some(obj) = msg.as_object_mut()
                {
                    obj.insert("reasoning".to_string(), json!(reasoning));
                }
                messages.push(msg);
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output.content,
                }));
            }
            ResponseItem::CustomToolCall {
                id,
                call_id: _,
                name,
                input,
                status: _,
            } => {
                messages.push(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id,
                        "type": "custom",
                        "custom": {
                            "name": name,
                            "input": input,
                        }
                    }]
                }));
            }
            ResponseItem::CustomToolCallOutput { call_id, output } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }));
            }
            ResponseItem::Reasoning { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::Other => {
                continue;
            }
        }
    }

    let mut attempt = 0;
    let max_retries = provider.request_max_retries();
    loop {
        attempt += 1;
        // Полный список tools из промпта
        let tools_json_full = create_tools_json_for_chat_completions_api(&prompt.tools)?;

        // Финальная попытка убрать битые tool_calls и добавить поясняющий prompt
        let (removed_any, broken_tools) =
            strip_broken_tool_calls_and_add_repair_prompt(&mut messages);

        // если удаляли — оставим только нужные инструменты;
        // если после фильтрации пусто — вернём полный список
        let tools_json = if removed_any {
            let filtered: Vec<Value> = tools_json_full
                .iter()
                .filter(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|name| broken_tools.contains(name))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if filtered.is_empty() {
                Value::Array(tools_json_full.clone())
            } else {
                Value::Array(filtered)
            }
        } else {
            Value::Array(tools_json_full.clone())
        };

        let payload = json!({
            "model": model_family.slug,
            "messages": messages,
            "stream": true,
            "tools": tools_json,
        });

        debug!(
            "POST to {}: {}",
            provider.get_full_url(&None),
            serde_json::to_string_pretty(&payload).unwrap_or_default()
        );

        let req_builder = provider.create_request_builder(client, &None).await?;

        let res = otel_event_manager
            .log_request(attempt, || {
                req_builder
                    .header(reqwest::header::ACCEPT, "text/event-stream")
                    .json(&payload)
                    .send()
            })
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent>>(1600);
                let stream = resp.bytes_stream().map_err(CodexErr::Reqwest);
                tokio::spawn(process_chat_sse(
                    stream,
                    tx_event,
                    provider.stream_idle_timeout(),
                    otel_event_manager.clone(),
                ));
                return Ok(ResponseStream { rx_event });
            }
            Ok(res) => {
                let status = res.status();
                let headers = res.headers().clone();
                let body = res.text().await.unwrap_or_default();

                // Спец-кейс: 400 + "Extra data"
                if status == StatusCode::BAD_REQUEST && body.contains("Extra data") {
                    if strip_broken_tool_calls_and_add_repair_prompt(&mut messages).0 {
                        warn!("HTTP 400 'Extra data': stripped broken tool_calls and requested a re-emit; retrying once");

                        let req_builder = provider.create_request_builder(client, &None).await?;
                        let retry_resp = otel_event_manager
                            .log_request(attempt, || {
                                req_builder
                                    .header(reqwest::header::ACCEPT, "text/event-stream")
                                    .json(&serde_json::json!({
                                        "model": model_family.slug,
                                        "messages": messages,
                                        "stream": true,
                                        "tools": tools_json,
                                    }))
                                    .send()
                            })
                            .await;

                        if let Ok(retry_ok) = retry_resp {
                            if retry_ok.status().is_success() {
                                let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent>>(1600);
                                let stream = retry_ok.bytes_stream().map_err(CodexErr::Reqwest);
                                tokio::spawn(process_chat_sse(
                                    stream,
                                    tx_event,
                                    provider.stream_idle_timeout(),
                                    otel_event_manager.clone(),
                                ));
                                return Ok(ResponseStream { rx_event });
                            }
                        }
                    }

                    return Err(CodexErr::UnexpectedStatus(UnexpectedResponseError {
                        status,
                        body,
                        request_id: None,
                    }));
                }

                if !(status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()) {
                    return Err(CodexErr::UnexpectedStatus(UnexpectedResponseError {
                        status,
                        body,
                        request_id: None,
                    }));
                }

                if attempt > max_retries {
                    return Err(CodexErr::RetryLimit(RetryLimitReachedError {
                        status,
                        request_id: None,
                    }));
                }

                let retry_after_secs = headers
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());

                let delay = retry_after_secs
                    .map(|s| Duration::from_millis(s * 1_000))
                    .unwrap_or_else(|| backoff(attempt));
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                if attempt > max_retries {
                    return Err(e.into());
                }
                let delay = backoff(attempt);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

fn is_valid_json_object_str(s: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(serde_json::Value::Object(_)) => true,
        _ => false,
    }
}

fn json_kind(v: &serde_json::Value) -> &'static str {
    use serde_json::Value::*;
    match v {
        Object(_) => "object",
        Array(_) => "array",
        String(_) => "string",
        Number(_) => "number",
        Bool(_) => "boolean",
        Null => "null",
    }
}

/// Нормализует arguments для функции `shell`.
/// Чиним паттерн `find ... -exec grep ... { ;` → `{}`; также упрощаем `.*FooId.*` → `FooId`.
fn fix_shell_args(raw: &str) -> Option<String> {
    use serde_json::{json, Value};

    let mut v: Value = serde_json::from_str(raw).ok()?;
    let obj = v.as_object_mut()?;

    let cmd = obj.get_mut("command")?;
    let arr = cmd.as_array_mut()?;

    // заменим одиночный "{" на "{}" в контексте -exec
    for i in 0..arr.len() {
        if arr.get(i).and_then(Value::as_str) == Some("-exec") {
            let mut j = i + 1;
            while j < arr.len() {
                if let Some(tok) = arr.get(j).and_then(Value::as_str) {
                    if tok == ";" || tok == "+" {
                        break;
                    }
                    if tok == "{" {
                        arr[j] = Value::String("{}".to_string());
                    }
                }
                j += 1;
            }
        }
    }

    // упрощаем regex-паттерны вида ".*FooId.*" → "FooId"
    for i in 0..arr.len() {
        if arr.get(i).and_then(Value::as_str) == Some("-l") {
            if let Some(pat) = arr.get_mut(i + 1) {
                if let Some(s) = pat.as_str() {
                    if s.starts_with(".*") && s.ends_with(".*") && s.len() > 4 {
                        let inner = &s[2..s.len() - 2];
                        *pat = Value::String(inner.to_string());
                    }
                }
            }
        }
    }

    serde_json::to_string(&v).ok()
}

/// Нормализует arguments для `update_plan`.
/// - `plan` обязателен; если строка — приводим к массиву с шагом.
/// - `explanation` если присутствует — оставляем только строку.
fn fix_update_plan_args(raw: &str) -> std::result::Result<String, String> {
    use serde_json::{Map, Value};

    let mut v: Value = serde_json::from_str(raw).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj: &mut Map<String, Value> =
        v.as_object_mut().ok_or_else(|| "arguments must be a JSON object".to_string())?;

    let plan_val = obj
        .remove("plan")
        .ok_or_else(|| "missing required field `plan`".to_string())?;

    let normalized_plan = match plan_val {
        Value::Array(a) => Value::Array(a),
        Value::String(s) => {
            let step = s.trim();
            if step.is_empty() {
                return Err("`plan` is empty string; expected non-empty step text".into());
            }
            json!([{"step": step, "status": "pending"}])
        }
        other => {
            return Err(format!(
                "`plan` must be an array (or string to coerce), got {}",
                json_kind(&other)
            ))
        }
    };

    obj.insert("plan".into(), normalized_plan);

    if let Some(expl) = obj.get("explanation") {
        if !expl.is_string() {
            obj.remove("explanation");
        }
    }

    serde_json::to_string(&v).map_err(|e| e.to_string())
}

/// Возвращает Ok(()) если `s` — ровно один валидный JSON-объект без лишних токенов.
fn validate_single_json_object_str(s: &str) -> std::result::Result<(), String> {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(v) => {
            if v.is_object() {
                Ok(())
            } else {
                Err(format!(
                    "arguments must be a single JSON object, got {}",
                    json_kind(&v)
                ))
            }
        }
        Err(e) => Err(format!("invalid JSON: {e}")),
    }
}

/// Удаляет из assistant-сообщений tool_calls с битым JSON и добавляет repair-подсказку.
fn strip_broken_tool_calls_and_add_repair_prompt(
    messages: &mut Vec<serde_json::Value>,
) -> (bool, std::collections::BTreeSet<String>) {
    use serde_json::Value;
    use std::collections::{BTreeMap, BTreeSet};

    let mut removed_any = false;
    let mut problems: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut broken_tools: BTreeSet<String> = BTreeSet::new();

    let mut i = 0usize;
    while i < messages.len() {
        let role_ok = messages[i].get("role").and_then(Value::as_str) == Some("assistant");
        if !role_ok {
            i += 1;
            continue;
        }

        let Some(obj) = messages[i].as_object_mut() else {
            i += 1;
            continue;
        };

        let tcs_val = obj.remove("tool_calls");
        let Some(tcs_arr) = tcs_val.and_then(|v| v.as_array().cloned()) else {
            i += 1;
            continue;
        };

        let mut kept: Vec<Value> = Vec::with_capacity(tcs_arr.len());

        for mut tc in tcs_arr {
            let is_func = tc.get("type").and_then(Value::as_str) == Some("function");
            if !is_func {
                removed_any = true;
                continue;
            }

            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();

            let args_str_opt = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str);

            let name_ok = !name.is_empty();

            let (args_ok, reason_opt, maybe_fixed_args) = if let Some(s) = args_str_opt {
                match serde_json::from_str::<Value>(s) {
                    Ok(Value::Object(_)) => {
                        if name == "update_plan" {
                            match fix_update_plan_args(s) {
                                Ok(fixed) => {
                                    let fixed_opt = (fixed != s).then_some(fixed);
                                    (true, None, fixed_opt)
                                }
                                Err(reason) => (false, Some(reason), None),
                            }
                        } else if name == "shell" {
                            let fixed_opt = fix_shell_args(s);
                            (true, None, fixed_opt)
                        } else {
                            (true, None, None)
                        }
                    }
                    Ok(other) => (
                        false,
                        Some(format!("arguments must be object, got {}", json_kind(&other))),
                        None,
                    ),
                    Err(e) => (false, Some(format!("invalid JSON: {e}")), None),
                }
            } else {
                (false, Some("missing `arguments` string".to_string()), None)
            };

            if name_ok && args_ok {
                if let Some(fixed) = maybe_fixed_args {
                    if let Some(f) = tc.get_mut("function").and_then(Value::as_object_mut) {
                        f.insert("arguments".into(), Value::String(fixed));
                    }
                }
                kept.push(tc);
            } else {
                removed_any = true;
                broken_tools.insert(if name.is_empty() {
                    "<unknown>".into()
                } else {
                    name.clone()
                });

                let mut notes = Vec::<String>::new();
                if !name_ok {
                    notes.push("missing function `name`".to_string());
                }
                if let Some(r) = reason_opt {
                    let snippet = args_str_opt
                        .map(|s| {
                            let s = s.trim();
                            let s = if s.len() > 160 {
                                format!("{}…", &s[..160])
                            } else {
                                s.to_string()
                            };
                            format!("args snippet: `{}`", s.replace('`', "'"))
                        })
                        .unwrap_or_else(|| "args snippet: <none>".to_string());

                    if let Some(s) = args_str_opt {
                        if let Ok(v) = serde_json::from_str::<Value>(s) {
                            let kind = json_kind(&v);
                            if kind != "object" {
                                notes.push(format!("arguments must be object, got {}", kind));
                                if let Value::Array(a) = &v {
                                    notes.push(format!("array length: {}", a.len()));
                                }
                                if let Value::String(st) = &v {
                                    notes.push(format!("string length: {}", st.len()));
                                }
                            }
                        }
                    }

                    notes.push(r);
                    notes.push(snippet);
                }
                problems
                    .entry(if name.is_empty() { "<unknown>".into() } else { name })
                    .or_default()
                    .extend(notes);
            }
        }

        if !kept.is_empty() {
            obj.insert("tool_calls".to_string(), Value::Array(kept));
        }
        i += 1;
    }

    if removed_any {
        let mut lines = Vec::<String>::new();
        lines.push(
            "Some previous tool call(s) were omitted because `arguments` was not a SINGLE valid JSON object."
                .into(),
        );
        lines.push(
            "Please re-emit the SAME call(s) with exactly one JSON object in `arguments` (no arrays, no multiple top-level objects, no trailing commas, no plain text)."
                .into(),
        );
        lines.push("Example for `update_plan`:".into());
        lines.push("```json".into());
        lines.push(
            r#"{ "plan": [ { "step": "Analyze Scala project structure and main components", "status": "pending" } ], "explanation": "High-level outline" }"#
                .into(),
        );
        lines.push("```".into());
        lines.push("Details:".into());

        for (tool, notes) in problems {
            lines.push(format!("- `{}`:", tool));
            for n in notes {
                lines.push(format!("  • {}", n));
            }
        }

        let already_has_repair = messages.iter().rev().take(8).any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content")
                .and_then(|c| c.as_str())
                .map_or(false, |s| {
                    s.starts_with("Some previous tool call(s) were omitted")
                })
        });
        if !already_has_repair {
            messages.push(serde_json::json!({
                "role": "user",
                "content": lines.join("\n"),
            }));
        }

        tracing::warn!("Dropped tool_calls due to bad arguments: {:?}", broken_tools);
    }

    (removed_any, broken_tools)
}

/// Lightweight SSE processor for the Chat Completions streaming format.
async fn process_chat_sse<S>(
    stream: S,
    tx_event: mpsc::Sender<Result<ResponseEvent>>,
    idle_timeout: Duration,
    otel_event_manager: OtelEventManager,
) where
    S: Stream<Item = Result<Bytes>> + Unpin,
{
    let mut stream = stream.eventsource();

    let mut emitted_calls: HashSet<(String, String)> = HashSet::new();
    let mut saw_indexed_tool_calls = false;

    #[derive(Default, Debug, Clone)]
    struct FunctionCallState {
        name: Option<String>,
        arguments: String,
        call_id: Option<String>,
        active: bool,
    }

    let mut tool_calls: Vec<FunctionCallState> = Vec::new();
    let mut legacy_fn: FunctionCallState = FunctionCallState::default();

    fn ensure_tc_len(v: &mut Vec<FunctionCallState>, idx: usize) {
        if v.len() <= idx {
            v.resize_with(idx + 1, FunctionCallState::default);
        }
    }

    let mut assistant_text = String::new();
    let mut reasoning_text = String::new();

    loop {
        let start = std::time::Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        let duration = start.elapsed();
        otel_event_manager.log_sse_event(&response, duration);

        let sse = match response {
            Ok(Some(Ok(ev))) => ev,
            Ok(Some(Err(e))) => {
                let _ = tx_event
                    .send(Err(CodexErr::Stream(e.to_string(), None)))
                    .await;
                return;
            }
            Ok(None) => {
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id: String::new(),
                        token_usage: None,
                    }))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(CodexErr::Stream(
                        "idle timeout waiting for SSE".into(),
                        None,
                    )))
                    .await;
                return;
            }
        };

        if sse.data.trim() == "[DONE]" {
            if !assistant_text.is_empty() {
                let item = ResponseItem::Message {
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: std::mem::take(&mut assistant_text),
                    }],
                    id: None,
                };
                let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            }

            if !reasoning_text.is_empty() {
                let item = ResponseItem::Reasoning {
                    id: String::new(),
                    summary: Vec::new(),
                    content: Some(vec![ReasoningItemContent::ReasoningText {
                        text: std::mem::take(&mut reasoning_text),
                    }]),
                    encrypted_content: None,
                };
                let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            }

            let _ = tx_event
                .send(Ok(ResponseEvent::Completed {
                    response_id: String::new(),
                    token_usage: None,
                }))
                .await;
            return;
        }

        let chunk: serde_json::Value = match serde_json::from_str(&sse.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        trace!("chat_completions received SSE chunk: {chunk:?}");

        let choice_opt = chunk.get("choices").and_then(|c| c.get(0));

        if let Some(choice) = choice_opt {
            if let Some(content) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
                && !content.is_empty()
            {
                assistant_text.push_str(content);
                let _ = tx_event
                    .send(Ok(ResponseEvent::OutputTextDelta(content.to_string())))
                    .await;
            }

            if let Some(reasoning_val) = choice.get("delta").and_then(|d| d.get("reasoning")) {
                let mut maybe_text = reasoning_val
                    .as_str()
                    .map(str::to_string)
                    .filter(|s| !s.is_empty());

                if maybe_text.is_none() && reasoning_val.is_object() {
                    if let Some(s) = reasoning_val
                        .get("text")
                        .and_then(|t| t.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        maybe_text = Some(s.to_string());
                    } else if let Some(s) = reasoning_val
                        .get("content")
                        .and_then(|t| t.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        maybe_text = Some(s.to_string());
                    }
                }

                if let Some(reasoning) = maybe_text {
                    reasoning_text.push_str(&reasoning);
                    let _ = tx_event
                        .send(Ok(ResponseEvent::ReasoningContentDelta(reasoning)))
                        .await;
                }
            }

            if let Some(message_reasoning) = choice.get("message").and_then(|m| m.get("reasoning"))
            {
                if let Some(s) = message_reasoning.as_str() {
                    if !s.is_empty() {
                        reasoning_text.push_str(s);
                        let _ = tx_event
                            .send(Ok(ResponseEvent::ReasoningContentDelta(s.to_string())))
                            .await;
                    }
                } else if let Some(obj) = message_reasoning.as_object()
                    && let Some(s) = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                    && !s.is_empty()
                {
                    reasoning_text.push_str(s);
                    let _ = tx_event
                        .send(Ok(ResponseEvent::ReasoningContentDelta(s.to_string())))
                        .await;
                }
            }

            // Новый формат: delta.tool_calls[*] c index
            if let Some(tool_calls_delta) = choice
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(|tc| tc.as_array())
            {
                saw_indexed_tool_calls = true;
                for tc in tool_calls_delta {
                    let idx = tc
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize;
                    ensure_tc_len(&mut tool_calls, idx);
                    let st = &mut tool_calls[idx];
                    st.active = true;

                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        st.call_id.get_or_insert_with(|| id.to_string());
                    }

                    if let Some(function) = tc.get("function") {
                        if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                            st.name.get_or_insert_with(|| name.to_string());
                        }
                        if let Some(args_fragment) =
                            function.get("arguments").and_then(|a| a.as_str())
                        {
                            st.arguments.push_str(args_fragment);
                        }
                    }
                }
            }

            // Старый формат: delta.function_call
            if let Some(function_call) = choice
                .get("delta")
                .and_then(|d| d.get("function_call"))
                .and_then(|fc| fc.as_object())
            {
                legacy_fn.active = true;

                if let Some(name) = function_call.get("name").and_then(|n| n.as_str()) {
                    legacy_fn.name.get_or_insert_with(|| name.to_string());
                }

                if let Some(args_fragment) = function_call.get("arguments").and_then(|a| a.as_str())
                {
                    legacy_fn.arguments.push_str(args_fragment);
                }
            }

            // Полный message.tool_calls
            if let Some(full_tcs) = choice
                .get("message")
                .and_then(|m| m.get("tool_calls"))
                .and_then(|tc| tc.as_array())
            {
                for (i, tc) in full_tcs.iter().enumerate() {
                    ensure_tc_len(&mut tool_calls, i);
                    let st = &mut tool_calls[i];
                    st.active = true;

                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        st.call_id.get_or_insert_with(|| id.to_string());
                    }
                    if let Some(function) = tc.get("function") {
                        if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                            st.name.get_or_insert_with(|| name.to_string());
                        }
                        if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                            st.arguments.push_str(args);
                        }
                    }
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                match finish_reason {
                    "tool_calls" => {
                        // 1) флеш ассистентского текста
                        if !assistant_text.is_empty() {
                            let item = ResponseItem::Message {
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText {
                                    text: std::mem::take(&mut assistant_text),
                                }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        // 2) флеш reasoning
                        if !reasoning_text.is_empty() {
                            let item = ResponseItem::Reasoning {
                                id: String::new(),
                                summary: Vec::new(),
                                content: Some(vec![ReasoningItemContent::ReasoningText {
                                    text: std::mem::take(&mut reasoning_text),
                                }]),
                                encrypted_content: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        // 3) эмитим tool calls (строгая правка + дедуп)
                        let mut any_invalid = false;
                        let mut invalid_notes: Vec<String> = Vec::new();

                        for st in tool_calls.iter().filter(|s| s.active) {
                            let name = st.name.clone().unwrap_or_default();
                            let raw_args = st.arguments.clone();

                            let args = match try_fix_arguments(&name, &raw_args) {
                                Ok(fixed) => fixed,
                                Err(reason) => {
                                    any_invalid = true;
                                    invalid_notes.push(format!("- `{}`: {}", name, reason));
                                    continue;
                                }
                            };

                            let key = (name.clone(), canonicalize_args(&args));
                            if !emitted_calls.insert(key) {
                                continue;
                            }

                            let item = ResponseItem::FunctionCall {
                                id: None,
                                name,
                                arguments: args,
                                call_id: st.call_id.clone().unwrap_or_default(),
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        // 4) если что-то пропустили — эмитим repair-подсказку как ассистент
                        if any_invalid {
                            let mut lines = Vec::<String>::new();
                            lines.push("Some tool call(s) were omitted because `arguments` was not a SINGLE valid JSON object.".into());
                            lines.push("Please re-emit the SAME call(s) with exactly one JSON object in `arguments` (no arrays, no multiple top-level objects, no trailing commas, no plain text).".into());
                            lines.push("Details:".into());
                            lines.extend(invalid_notes);
                            lines.push("Example for `update_plan`:\n```json\n{ \"plan\": [ { \"step\": \"Compare TECM token caching…\", \"status\": \"pending\" } ] }\n```".into());

                            let item = ResponseItem::Message {
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText { text: lines.join("\n") }],
                                id: None,
                            };
                            let _ = tx_event
                                .send(Ok(ResponseEvent::OutputItemDone(item)))
                                .await;
                        }

                        let _ = tx_event
                            .send(Ok(ResponseEvent::Completed {
                                response_id: String::new(),
                                token_usage: None,
                            }))
                            .await;
                        return;
                    }
                    "function_call" if legacy_fn.active => {
                        if saw_indexed_tool_calls {
                            let _ = tx_event
                                .send(Ok(ResponseEvent::Completed {
                                    response_id: String::new(),
                                    token_usage: None,
                                }))
                                .await;
                            return;
                        }

                        // флеш ассистента / reasoning
                        if !assistant_text.is_empty() {
                            let item = ResponseItem::Message {
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText {
                                    text: std::mem::take(&mut assistant_text),
                                }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }
                        if !reasoning_text.is_empty() {
                            let item = ResponseItem::Reasoning {
                                id: String::new(),
                                summary: Vec::new(),
                                content: Some(vec![ReasoningItemContent::ReasoningText {
                                    text: std::mem::take(&mut reasoning_text),
                                }]),
                                encrypted_content: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        // строгая правка legacy-args
                        let name = legacy_fn.name.take().unwrap_or_default();
                        let raw_args = std::mem::take(&mut legacy_fn.arguments);

                        let args = match try_fix_arguments(&name, &raw_args) {
                            Ok(fixed) => fixed,
                            Err(reason) => {
                                let lines = vec![
                                    "A previous tool call was omitted because `arguments` was not a SINGLE valid JSON object.".to_string(),
                                    "Please re-emit the SAME call with exactly one JSON object in `arguments`.".to_string(),
                                    format!("Details:\n- `{}`: {}", name, reason),
                                    "Example for `update_plan`:\n```json\n{ \"plan\": [ { \"step\": \"Compare TECM token caching…\", \"status\": \"pending\" } ] }\n```".to_string(),
                                ];
                                let item = ResponseItem::Message {
                                    role: "assistant".to_string(),
                                    content: vec![ContentItem::OutputText { text: lines.join("\n") }],
                                    id: None,
                                };
                                let _ = tx_event
                                    .send(Ok(ResponseEvent::OutputItemDone(item)))
                                    .await;

                                let _ = tx_event
                                    .send(Ok(ResponseEvent::Completed {
                                        response_id: String::new(),
                                        token_usage: None,
                                    }))
                                    .await;
                                return;
                            }
                        };

                        let key = (name.clone(), canonicalize_args(&args));
                        if emitted_calls.insert(key) {
                            let item = ResponseItem::FunctionCall {
                                id: None,
                                name,
                                arguments: args,
                                call_id: legacy_fn.call_id.unwrap_or_default(),
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        let _ = tx_event
                            .send(Ok(ResponseEvent::Completed {
                                response_id: String::new(),
                                token_usage: None,
                            }))
                            .await;
                        return;
                    }
                    "stop" => {
                        let mut emitted = false;

                        if !assistant_text.is_empty() {
                            emitted = true;
                            let item = ResponseItem::Message {
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText {
                                    text: std::mem::take(&mut assistant_text),
                                }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }
                        if !reasoning_text.is_empty() {
                            emitted = true;
                            let item = ResponseItem::Reasoning {
                                id: String::new(),
                                summary: Vec::new(),
                                content: Some(vec![ReasoningItemContent::ReasoningText {
                                    text: std::mem::take(&mut reasoning_text),
                                }]),
                                encrypted_content: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        if !emitted {
                            let item = ResponseItem::Message {
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText {
                                    text: "Previous tool call(s) were dropped because `arguments` was not a single valid JSON object. Re-emit the same call(s) with exactly one JSON object."
                                        .to_string(),
                                }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        let _ = tx_event
                            .send(Ok(ResponseEvent::Completed {
                                response_id: String::new(),
                                token_usage: None,
                            }))
                            .await;
                        return;
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Optional client-side aggregation helper
#[derive(Copy, Clone, Eq, PartialEq)]
enum AggregateMode {
    AggregatedOnly,
    Streaming,
}
pub(crate) struct AggregatedChatStream<S> {
    inner: S,
    cumulative: String,
    cumulative_reasoning: String,
    pending: std::collections::VecDeque<ResponseEvent>,
    mode: AggregateMode,
}

impl<S> Stream for AggregatedChatStream<S>
where
    S: Stream<Item = Result<ResponseEvent>> + Unpin,
{
    type Item = Result<ResponseEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if let Some(ev) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(ev)));
        }

        loop {
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(ResponseEvent::OutputItemDone(item)))) => {
                    let is_assistant_message = matches!(
                        &item,
                        codex_protocol::models::ResponseItem::Message { role, .. } if role == "assistant"
                    );

                    if is_assistant_message {
                        match this.mode {
                            AggregateMode::AggregatedOnly => {
                                if this.cumulative.is_empty()
                                    && let codex_protocol::models::ResponseItem::Message {
                                    content,
                                    ..
                                } = &item
                                    && let Some(text) = content.iter().find_map(|c| match c {
                                    codex_protocol::models::ContentItem::OutputText {
                                        text,
                                    } => Some(text),
                                    _ => None,
                                })
                                {
                                    this.cumulative.push_str(text);
                                }
                                continue;
                            }
                            AggregateMode::Streaming => {
                                if this.cumulative.is_empty() {
                                    return Poll::Ready(Some(Ok(ResponseEvent::OutputItemDone(
                                        item,
                                    ))));
                                } else {
                                    continue;
                                }
                            }
                        }
                    }

                    return Poll::Ready(Some(Ok(ResponseEvent::OutputItemDone(item))));
                }
                Poll::Ready(Some(Ok(ResponseEvent::RateLimits(snapshot)))) => {
                    return Poll::Ready(Some(Ok(ResponseEvent::RateLimits(snapshot))));
                }
                Poll::Ready(Some(Ok(ResponseEvent::Completed {
                                        response_id,
                                        token_usage,
                                    }))) => {
                    let mut emitted_any = false;

                    if !this.cumulative_reasoning.is_empty()
                        && matches!(this.mode, AggregateMode::AggregatedOnly)
                    {
                        let aggregated_reasoning =
                            codex_protocol::models::ResponseItem::Reasoning {
                                id: String::new(),
                                summary: Vec::new(),
                                content: Some(vec![
                                    codex_protocol::models::ReasoningItemContent::ReasoningText {
                                        text: std::mem::take(&mut this.cumulative_reasoning),
                                    },
                                ]),
                                encrypted_content: None,
                            };
                        this.pending
                            .push_back(ResponseEvent::OutputItemDone(aggregated_reasoning));
                        emitted_any = true;
                    }

                    if !this.cumulative.is_empty() {
                        let aggregated_message = codex_protocol::models::ResponseItem::Message {
                            id: None,
                            role: "assistant".to_string(),
                            content: vec![codex_protocol::models::ContentItem::OutputText {
                                text: std::mem::take(&mut this.cumulative),
                            }],
                        };
                        this.pending
                            .push_back(ResponseEvent::OutputItemDone(aggregated_message));
                        emitted_any = true;
                    }

                    if emitted_any {
                        this.pending.push_back(ResponseEvent::Completed {
                            response_id: response_id.clone(),
                            token_usage: token_usage.clone(),
                        });
                        if let Some(ev) = this.pending.pop_front() {
                            return Poll::Ready(Some(Ok(ev)));
                        }
                    }

                    return Poll::Ready(Some(Ok(ResponseEvent::Completed {
                        response_id,
                        token_usage,
                    })));
                }
                Poll::Ready(Some(Ok(ResponseEvent::Created))) => {
                    continue;
                }
                Poll::Ready(Some(Ok(ResponseEvent::OutputTextDelta(delta)))) => {
                    this.cumulative.push_str(&delta);
                    if matches!(this.mode, AggregateMode::Streaming) {
                        return Poll::Ready(Some(Ok(ResponseEvent::OutputTextDelta(delta))));
                    } else {
                        continue;
                    }
                }
                Poll::Ready(Some(Ok(ResponseEvent::ReasoningContentDelta(delta)))) => {
                    this.cumulative_reasoning.push_str(&delta);
                    if matches!(this.mode, AggregateMode::Streaming) {
                        return Poll::Ready(Some(Ok(ResponseEvent::ReasoningContentDelta(
                            delta,
                        ))));
                    } else {
                        continue;
                    }
                }
                Poll::Ready(Some(Ok(ResponseEvent::ReasoningSummaryDelta(_)))) => {
                    continue;
                }
                Poll::Ready(Some(Ok(ResponseEvent::ReasoningSummaryPartAdded))) => {
                    continue;
                }
                Poll::Ready(Some(Ok(ResponseEvent::WebSearchCallBegin { call_id }))) => {
                    return Poll::Ready(Some(Ok(ResponseEvent::WebSearchCallBegin { call_id })));
                }
            }
        }
    }
}

/// Extension trait that activates aggregation on any stream of [`ResponseEvent`].
pub(crate) trait AggregateStreamExt: Stream<Item = Result<ResponseEvent>> + Sized {
    fn aggregate(self) -> AggregatedChatStream<Self> {
        AggregatedChatStream::new(self, AggregateMode::AggregatedOnly)
    }
}

impl<T> AggregateStreamExt for T where T: Stream<Item = Result<ResponseEvent>> + Sized {}

impl<S> AggregatedChatStream<S> {
    fn new(inner: S, mode: AggregateMode) -> Self {
        AggregatedChatStream {
            inner,
            cumulative: String::new(),
            cumulative_reasoning: String::new(),
            pending: std::collections::VecDeque::new(),
            mode,
        }
    }

    pub(crate) fn streaming_mode(inner: S) -> Self {
        Self::new(inner, AggregateMode::Streaming)
    }
}