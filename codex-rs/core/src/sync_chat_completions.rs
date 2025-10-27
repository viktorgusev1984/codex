use std::time::Duration;

use crate::ModelProviderInfo;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::config_types::SyncChatCompletionsConfig;
use crate::default_client::CodexHttpClient;
use crate::error::CodexErr;
use crate::error::ConnectionFailedError;
use crate::error::Result;
use crate::error::RetryLimitReachedError;
use crate::error::UnexpectedResponseError;
use crate::model_family::ModelFamily;
use crate::tools::spec::create_tools_json_for_chat_completions_api;
use crate::util::backoff;
use codex_otel::otel_event_manager::OtelEventManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use httpdate::parse_http_date;
use regex_lite::Regex;
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use serde_json::value::RawValue;
use std::collections::HashSet;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use tracing::debug;
use tracing::trace;
use tracing::warn;

// === NEW: Cycle-guard marker and utils ===
const CYCLE_GUARD_MARKER: &str = "[cycle-guard]";
const SEARCH_GUARD_MARKER: &str = "[search-guard]";

#[derive(Debug, Clone)]
struct CycleInfo {
    period: usize,
    repeats: usize,
    pattern: Vec<String>, // последовательность имён инструментов за один цикл
}

fn search_guard_recently_injected(messages: &Vec<serde_json::Value>, lookback: usize) -> bool {
    let mut seen = 0usize;
    for m in messages.iter().rev() {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "user" {
            if let Some(s) = m.get("content").and_then(|v| v.as_str()) {
                if s.contains(SEARCH_GUARD_MARKER) {
                    return true;
                }
            }
        }
        seen += 1;
        if seen >= lookback {
            break;
        }
    }
    false
}

/// Попробуем вытащить grep-паттерн вида import.*XXX из аргументов вызова shell
fn extract_grep_pattern_from_shell_args(args_raw: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_raw).ok()?;
    let cmd = v
        .get("arguments")
        .unwrap_or(&v) // поддержим оба вида: {"arguments":{...}} и просто {...}
        .get("command")?;
    let arr = cmd.as_array()?;
    // ищем "grep", затем берем следующий токен после -l (если есть), либо первый паттерновый токен
    let mut saw_grep = false;
    let mut skip_next = false;
    for (i, tok) in arr.iter().enumerate() {
        let s = tok.as_str().unwrap_or("");
        if s == "grep" {
            saw_grep = true;
            continue;
        }
        if !saw_grep {
            continue;
        }
        if skip_next {
            skip_next = false;
            continue;
        }
        if s == "-l" {
            continue;
        }
        if s.starts_with("-") {
            // другие флаги
            // если это флаг, после которого идёт значение, пропустим одно
            if s == "-e" {
                skip_next = true;
            }
            continue;
        }
        // кандидат на паттерн
        return Some(s.to_string());
    }
    None
}

#[inline]
fn prev_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[inline]
fn truncate_bytes_safe(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        s.to_string()
    } else {
        let end = prev_char_boundary(s, max_bytes);
        // Показать «полную» длину лучше в символах, а не в байтах
        let total_chars = s.chars().count();
        format!("{}… ({} chars)", &s[..end], total_chars)
    }
}

#[inline]
fn shorten_bytes_safe(s: &str, max_bytes: usize) -> String {
    // компактный вариант без приписки длины
    if s.len() <= max_bytes {
        s.to_string()
    } else {
        let end = prev_char_boundary(s, max_bytes);
        format!("{}…", &s[..end])
    }
}

/// Детектируем «рыбалку»: длинная серия find/grep по разным import.*XXX с пустыми результатами
fn detect_search_fishing_pattern(
    messages: &Vec<serde_json::Value>,
) -> Option<(usize, usize, Vec<String>)> {
    // Сканируем с конца не больше последних 20 сообщений (10 пар)
    let mut i = messages.len();
    let mut attempts = 0usize;
    let mut empty_ok = 0usize;
    let mut patterns: Vec<String> = Vec::new();

    while i >= 2 && attempts < 12 {
        // до 12 попыток (где-то ~ 24 сообщения)
        let tool_msg = &messages[i - 1];
        let asst_msg = &messages[i - 2];

        let role_tool = tool_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let role_asst = asst_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role_tool != "tool" || role_asst != "assistant" {
            break;
        }

        // достаём tool_calls[0].function.name == "shell"
        let tc = match asst_msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(0))
        {
            Some(x) => x,
            None => break,
        };
        if tc.get("type").and_then(|v| v.as_str()) != Some("function") {
            break;
        }
        let func = match tc.get("function").and_then(|v| v.as_object()) {
            Some(f) => f,
            None => break,
        };
        if func.get("name").and_then(|v| v.as_str()) != Some("shell") {
            break;
        }

        // аргументы (строка или объект)
        let args_raw = func
            .get("arguments")
            .map(|val| match val {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            })
            .unwrap_or_default();

        // эвристика: это вообще find/grep по java?
        if !args_raw.contains("find") || !args_raw.contains("grep") {
            break;
        }

        attempts += 1;

        // проверим пустоту вывода
        let tool_out = tool_msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let is_empty = is_tool_output_empty(tool_out);
        if is_empty {
            empty_ok += 1;
        }

        // вытащим паттерн grep (например "import.*cats")
        if let Some(p) = extract_grep_pattern_from_shell_args(&args_raw) {
            patterns.push(p);
        }

        if i >= 2 {
            i -= 2;
        } else {
            break;
        }
    }

    if attempts >= 5 {
        // >= 5 подряд попыток, и >= 80% пустых ответов
        let ratio = if attempts > 0 {
            (empty_ok * 100) / attempts
        } else {
            0
        };
        let distinct = {
            let mut hs = std::collections::HashSet::<String>::new();
            for p in &patterns {
                hs.insert(p.clone());
            }
            hs.len()
        };

        if ratio >= 80 && distinct >= 3 {
            // вернём (attempts, distinct, последние до 5 паттернов для сообщения)
            let mut tail = patterns;
            tail.reverse();
            tail.truncate(5);
            tail.reverse();
            return Some((attempts, distinct, tail));
        }
    }

    None
}

fn build_search_guard_feedback(
    attempts: usize,
    distinct: usize,
    sample_patterns: &[String],
    last_tool_output: Option<String>,
) -> String {
    let mut examples = String::new();
    if !sample_patterns.is_empty() {
        examples.push_str("\nПримеры последних запросов grep:\n");
        for p in sample_patterns {
            examples.push_str(&format!("- `{}`\n", p));
        }
    }
    let out_line = match last_tool_output {
        Some(s) if !s.is_empty() => format!(
            "\nПоследний вывод инструмента (фрагмент):\n```\n{}\n```\n",
            s
        ),
        _ => String::new(),
    };

    format!(
        "{marker} Похоже, идёт **бесконечный перебор поиска**: {attempts} попыток подряд (различных шаблонов: {distinct}), почти все с пустым результатом.{examples}\n\
        Продолжать такой перебор?\n\
        Ответь **Да/Нет**. Если **Да** — укажи:\n\
        - допустимые префиксы/паттерны (например: `ru.tcsbank`, `ru.tinkoff`),\n\
        - предел попыток (например: 3),\n\
        - критерий остановки (например: «нашли ≥1 файл»).\n\
        {out}\
        Пока подтверждения нет — **не делай новых вызовов поиска**.",
        marker = SEARCH_GUARD_MARKER,
        attempts = attempts,
        distinct = distinct,
        examples = examples,
        out = out_line
    )
}

/// Забираем последние assistant tool_calls (по именам инструментов)
fn recent_assistant_tool_names(messages: &Vec<serde_json::Value>, limit: usize) -> Vec<String> {
    let mut names = Vec::<String>::new();
    for m in messages.iter().rev() {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "assistant" {
            continue;
        }
        let tcs = match m.get("tool_calls").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => continue,
        };
        if tcs.len() != 1 {
            continue;
        }
        let tc = &tcs[0];
        if tc.get("type").and_then(|v| v.as_str()) != Some("function") {
            continue;
        }
        if let Some(func) = tc.get("function").and_then(|v| v.as_object()) {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                names.push(name.to_string());
                if names.len() >= limit {
                    break;
                }
            }
        }
    }
    names.reverse(); // хронологический порядок
    names
}

/// Проверяем, не подкидывали ли недавно наше служебное сообщение
fn cycle_guard_recently_injected(messages: &Vec<serde_json::Value>, lookback: usize) -> bool {
    let mut seen = 0usize;
    for m in messages.iter().rev() {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "user" {
            if let Some(s) = m.get("content").and_then(|v| v.as_str()) {
                if s.contains(CYCLE_GUARD_MARKER) {
                    return true;
                }
            }
        }
        seen += 1;
        if seen >= lookback {
            break;
        }
    }
    false
}

fn extract_shell_signature(args_raw: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_raw).ok()?;
    let args = v.get("arguments").unwrap_or(&v);
    let cmd = args.get("command")?.as_array()?;

    // Соберём полную строку (bash -lc "…")
    let full =
        if cmd.len() >= 3 && cmd[0].as_str() == Some("bash") && cmd[1].as_str() == Some("-lc") {
            cmd[2].as_str().unwrap_or("").to_string()
        } else {
            cmd.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        .to_lowercase();

    if full.contains("mvn") && full.contains("compile") {
        return Some("shell:mvn:compile".to_string());
    }
    if full.contains("grep") {
        if full.contains("spring-boot-starter-web") || full.contains("spring-boot-starter-webflux")
        {
            return Some("shell:grep:web".to_string());
        }
        return Some("shell:grep".to_string());
    }
    if full.contains("sed") {
        if full.contains("s|</dependencies>.*<dependency>|</dependencies>|") {
            return Some("shell:sed:delete-deps-block".to_string());
        }
        if full.contains("'/<dependencies>/a") || full.contains("/<dependencies>/a") {
            return Some("shell:sed:insert-webflux".to_string());
        }
        return Some("shell:sed".to_string());
    }
    if full.starts_with("cat ") && full.contains(" | grep ") {
        return Some("shell:cat-grep".to_string());
    }
    Some("shell:other".to_string())
}

fn recent_assistant_tool_signatures(
    messages: &Vec<serde_json::Value>,
    limit: usize,
) -> Vec<String> {
    let mut sigs = Vec::<String>::new();
    for m in messages.iter().rev() {
        if m.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        let tcs = match m.get("tool_calls").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => continue,
        };
        if tcs.len() != 1 {
            continue;
        }
        let tc = &tcs[0];
        if tc.get("type").and_then(|v| v.as_str()) != Some("function") {
            continue;
        }
        if tc
            .get("function")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            != Some("shell")
        {
            continue;
        }

        let args_raw = tc
            .get("function")
            .and_then(|v| v.get("arguments"))
            .map(|x| match x {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            })
            .unwrap_or_default();

        if let Some(sig) = extract_shell_signature(&args_raw) {
            sigs.push(sig);
            if sigs.len() >= limit {
                break;
            }
        }
    }
    sigs.reverse();
    sigs
}

// Поставь это рядом с другими helpers (ниже emit_sync_events_from_chat, например).

async fn guard_roundtrip(
    feedback: String,
    mut current_messages: Vec<serde_json::Value>,
    mut payload: serde_json::Value,
    client: &CodexHttpClient,
    provider: &ModelProviderInfo,
    otel: &OtelEventManager,
    attempt: usize,
    prev_body_curr: serde_json::Value,
) -> Result<(serde_json::Value, Vec<serde_json::Value>)> {
    // 1) инъекция нашего "user"-сообщения
    current_messages.push(serde_json::json!({ "role": "user", "content": feedback }));

    // 2) обновляем payload.messages
    if let Some(obj) = payload.as_object_mut() {
        obj.insert(
            "messages".into(),
            serde_json::Value::Array(current_messages.clone()),
        );
    }

    // 3) повторный запрос
    let req_retry = provider.create_request_builder(client, &None).await?;
    let resp_retry = otel
        .log_request((attempt as u64) + 1, || {
            req_retry
                .header(reqwest::header::ACCEPT, "application/json")
                .json(&payload)
                .send()
        })
        .await;

    match resp_retry {
        Ok(ok) if ok.status().is_success() => {
            let body = ok
                .json()
                .await
                .map_err(|e| CodexErr::Fatal(format!("guard roundtrip decode failed: {e}")))?;
            // подшиваем новый assistant в историю
            let mut msgs = current_messages.clone();
            append_assistant_message_from_body(&mut msgs, &body);
            Ok((body, msgs))
        }
        Ok(bad) => {
            warn!("guard roundtrip failed: status={}", bad.status());
            // не удалось — продолжаем с прежним body и уже дополненной историей
            Ok((prev_body_curr, current_messages))
        }
        Err(e) => {
            warn!("guard roundtrip error: {}", e);
            Ok((prev_body_curr, current_messages))
        }
    }
}

/// Детектируем периодический паттерн на хвосте последовательности имен инструментов.
/// Ищем период p в [min_period..=max_period], минимум min_repeats повторов.
/// Исключаем случай «все элементы одинаковые» (его и так ловит repeat-guard по args).
fn detect_cyclic_pattern(
    seq: &[String],
    min_period: usize,
    max_period: usize,
    min_repeats: usize,
) -> Option<CycleInfo> {
    if seq.len() < min_period * min_repeats {
        return None;
    }
    let n = seq.len();
    let max_p = max_period.min(n / min_repeats).max(min_period);
    for p in min_period..=max_p {
        let base = &seq[n - p..n];
        let mut repeats = 1usize;
        while repeats * p <= n {
            let start = n.saturating_sub((repeats + 1) * p);
            let end = n - repeats * p;
            if start >= end {
                break;
            }
            if &seq[start..end] == base {
                repeats += 1;
            } else {
                break;
            }
        }
        if repeats >= min_repeats {
            let uniq: HashSet<_> = base.iter().collect();
            // исключаем циклы вида [shell,shell,...] — их ловим другими guard'ами
            // if uniq.len() > 1 {
            if uniq.len() > 1 || p > 1 {
                return Some(CycleInfo {
                    period: p,
                    repeats,
                    pattern: base.to_vec(),
                });
            }
        }
    }
    None
}

fn build_cycle_guard_feedback_with_context(
    pattern: &[String],
    repeats: usize,
    last_tool_output_snippet: Option<String>,
) -> String {
    let pattern_str = if pattern.is_empty() {
        "(пусто)".to_string()
    } else {
        pattern.join(" → ")
    };
    let out_line = match last_tool_output_snippet {
        Some(s) if !s.is_empty() => format!(
            "\nПоследний вывод инструмента (фрагмент):\n```\n{}\n```\n",
            s
        ),
        _ => String::new(),
    };

    format!(
        "{marker} Похоже, ты зациклился: паттерн `{pattern}` повторился {repeats}×.\n\
         Пожалуйста, измени стратегию и продолжи — сделай **иной следующий шаг**, \
         не повторяющий элементы цикла. Если нужно, кратко отметь план (1–2 шага) и выполни первый.\n\
         Требования:\n\
         - Отправь **ровно один** новый вызов инструмента (не из цикла) с валидным JSON `arguments`.\n\
         - Избегай команд из паттерна до появления нового результата.\n\
         {out}",
        marker = CYCLE_GUARD_MARKER,
        pattern = pattern_str,
        repeats = repeats,
        out = out_line
    )
}
// === END NEW: Cycle-guard utils ===

/// Синхронный (нестриминговый) вызов Chat Completions API.
pub(crate) async fn chat_completions_sync(
    prompt: &Prompt,
    model_family: &ModelFamily,
    client: &CodexHttpClient,
    provider: &ModelProviderInfo,
    otel_event_manager: &OtelEventManager,
    sync_config: &SyncChatCompletionsConfig,
) -> Result<ResponseStream> {
    if prompt.output_schema.is_some() {
        return Err(CodexErr::UnsupportedOperation(
            "output_schema is not supported for Chat Completions API".to_string(),
        ));
    }

    // 1) Сборка messages (повторяет вашу логику из stream_*).
    let mut messages = Vec::<serde_json::Value>::new();
    let full_instructions = prompt.get_full_instructions(model_family);
    messages.push(json!({"role": "system", "content": full_instructions}));

    let input = prompt.get_formatted_input();

    // Последняя роль
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
            ResponseItem::GhostSnapshot { .. } => {}
        }
    }

    // Индекс последнего user
    let mut last_user_index: Option<usize> = None;
    for (idx, item) in input.iter().enumerate() {
        if let ResponseItem::Message { role, .. } = item
            && role == "user"
        {
            last_user_index = Some(idx);
        }
    }

    // Карман для reasoning->anchor
    let mut reasoning_by_anchor_index = std::collections::HashMap::<usize, String>::new();

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
                // привязываем к соседнему assistant/anchor
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

    // Избегаем дубликатов assistant
    let mut last_assistant_text: Option<String> = None;

    for (idx, item) in input.iter().enumerate() {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let mut text = String::new();
                for c in content {
                    match c {
                        ContentItem::InputText { text: t }
                        | ContentItem::OutputText { text: t } => text.push_str(t),
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
                let mut msg = json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments }
                    }]
                });
                if let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                    && let Some(obj) = msg.as_object_mut()
                {
                    obj.insert("reasoning".to_string(), json!(reasoning));
                }
                messages.push(msg);
            }
            ResponseItem::LocalShellCall {
                id, status, action, ..
            } => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id.clone().unwrap_or_default(),
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
                id, name, input, ..
            } => {
                messages.push(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id,
                        "type": "custom",
                        "custom": { "name": name, "input": input }
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
            | ResponseItem::GhostSnapshot { .. }
            | ResponseItem::Other => {}
        }
    }

    let tools_json = create_tools_json_for_chat_completions_api(&prompt.tools)?;
    let response_format = prompt.output_schema.as_ref().map(|schema| {
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "codex_output_schema",
                "schema": schema,
                "strict": true,
            }
        })
    });

    let mut payload = json!({
        "model": model_family.slug,
        "messages": messages,
        "stream": false,
        "tools": tools_json,
        "temperature": sync_config.temperature,
        "top_p": sync_config.top_p,
        "min_p": sync_config.min_p,
    });

    if let Some(top_k) = sync_config.top_k {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("top_k".to_string(), json!(top_k));
        }
    }

    if let Some(format) = response_format.clone() {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("response_format".to_string(), format);
        }
    }

    debug!(
        "POST (sync) to {}: {}",
        provider.get_full_url(&None),
        serde_json::to_string_pretty(&payload).unwrap_or_default()
    );

    // 2) Запрос с ретраями
    let mut attempt: usize = 0;
    let max_retries: usize = provider.request_max_retries() as usize;
    let mut next_allowed_until: Option<SystemTime> = None;

    loop {
        attempt += 1;
        let req_builder = provider.create_request_builder(client, &None).await?;
        let res = otel_event_manager
            .log_request(attempt as u64, || {
                req_builder
                    .header(reqwest::header::ACCEPT, "application/json")
                    .json(&payload)
                    .send()
            })
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                let body: Value = match resp.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(CodexErr::Fatal(format!(
                            "failed to decode chat completion JSON: {e}"
                        )));
                    }
                };
                trace!("chat_completions sync response: {body:#}");

                // рабочее состояние, которое будем мутировать в ретраях
                let mut body_curr = body.clone();
                let mut current_messages = payload
                    .get("messages")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_else(|| Vec::new());

                // === ЗАЩИТА ОТ ПОВТОРОВ ОДИНАКОВОГО ВЫЗОВА ===
                if let Some((name_now, args_now_raw)) =
                    extract_first_tool_call_from_body(&body_curr)
                {
                    if let Some(args_norm_now) = normalize_json_args(&args_now_raw) {
                        // Считаем в актуальной истории
                        let prev_same = count_recent_same_tool_calls(
                            &current_messages,
                            &name_now,
                            &args_norm_now,
                        );
                        if prev_same >= 2 {
                            // возьмём последний вывод tool для контекста
                            let last_out = extract_last_tool_output(&current_messages, 500);
                            let args_norm_display = args_norm_now.clone();

                            let feedback = build_repeat_guard_feedback_with_context(
                                &name_now,
                                &args_norm_display,
                                last_out,
                            );

                            // ⚠️ Единый путь: инжектим user + отправляем повторный запрос + возвращаем body/messages
                            let (body_new, msgs_new) = guard_roundtrip(
                                feedback,
                                current_messages,
                                payload.clone(),
                                client,
                                provider,
                                otel_event_manager,
                                attempt,
                                body_curr.clone(),
                            )
                            .await?;

                            body_curr = body_new;
                            current_messages = msgs_new;

                            // 🔧 ВАЖНО: сразу синхронизируем messages обратно в payload
                            if let Some(obj) = payload.as_object_mut() {
                                obj.insert(
                                    "messages".into(),
                                    serde_json::Value::Array(current_messages.clone()),
                                );
                            }
                        } else {
                            // обычный случай: просто добавим ассистентский месседж в историю
                            append_assistant_message_from_body(&mut current_messages, &body_curr);
                        }
                    } else {
                        append_assistant_message_from_body(&mut current_messages, &body_curr);
                    }
                } else {
                    // нет tool_call — тоже фиксируем ассистента (мог быть просто текст)
                    append_assistant_message_from_body(&mut current_messages, &body_curr);
                }
                // --- КОНЕЦ repeat-guard ---

                // === NEW: ZERO-PROGRESS GUARD — один и тот же инструмент, пустой Output подряд ===
                {
                    if let Some((repeats, tool_name)) = detect_zero_progress_loop(
                        &current_messages,
                        /*min_repeats*/ 6,
                        /*max_scan*/ 40,
                    ) {
                        let feedback =
                            build_cycle_guard_feedback_zero_progress(repeats, &tool_name);
                        let (body_new, msgs_new) = guard_roundtrip(
                            feedback,
                            current_messages,
                            payload.clone(),
                            client,
                            provider,
                            otel_event_manager,
                            attempt,
                            body_curr.clone(),
                        )
                        .await?;
                        body_curr = body_new;
                        current_messages = msgs_new;

                        // 🔧 ВАЖНО: сразу синхронизируем messages обратно в payload
                        if let Some(obj) = payload.as_object_mut() {
                            obj.insert(
                                "messages".into(),
                                serde_json::Value::Array(current_messages.clone()),
                            );
                        }
                    }
                }
                // --- END ZERO-PROGRESS GUARD ---

                // === NEW: SEARCH-GUARD — защита от бесконечного «перебора поиска» ===
                {
                    if !search_guard_recently_injected(&current_messages, 8) {
                        if let Some((attempts, distinct, sample)) =
                            detect_search_fishing_pattern(&current_messages)
                        {
                            let last_out = extract_last_tool_output(&current_messages, 500);
                            let feedback =
                                build_search_guard_feedback(attempts, distinct, &sample, last_out);

                            // ⬇️ ВМЕСТО эмита и return — делаем раунд-трип и продолжаем
                            let (body_new, msgs_new) = guard_roundtrip(
                                feedback,
                                current_messages,
                                payload.clone(),
                                client,
                                provider,
                                otel_event_manager,
                                attempt,
                                body_curr.clone(),
                            )
                            .await?;

                            body_curr = body_new;
                            current_messages = msgs_new;
                            // продолжаем обычный поток ниже, БЕЗ return
                            // 🔧 ВАЖНО: сразу синхронизируем messages обратно в payload
                            if let Some(obj) = payload.as_object_mut() {
                                obj.insert(
                                    "messages".into(),
                                    serde_json::Value::Array(current_messages.clone()),
                                );
                            }
                        }
                    }
                }
                // --- КОНЕЦ SEARCH-GUARD ---

                // === NEW: вспомогательный флаг для заметки от cycle-guard ===
                let mut cycle_guard_note: Option<String> = None;

                // === CYCLE-GUARD: обнаружили цикл — подкинем вопрос и сделаем roundtrip ===
                {
                    if !cycle_guard_recently_injected(&current_messages, 8) {
                        // 1) Попытка по role+tool: assistant:shell → tool:shell (period=2)
                        let rt_seq = recent_role_tool_seq(&current_messages, 16); // ADD
                        let mut cycle_hit = detect_cyclic_pattern(
                            &rt_seq, /*min_period*/ 2, /*max_period*/ 4,
                            /*min_repeats*/ 3,
                        );

                        // 2) Фоллбек: по "сигнатурам shell" (твой предыдущий способ)
                        if cycle_hit.is_none() {
                            let sigs = recent_assistant_tool_signatures(&current_messages, 24);
                            cycle_hit = detect_cyclic_pattern(&sigs, 1, 6, 3); // разрешим period=1
                        }

                        if let Some(ci) = cycle_hit {
                            let last_out = extract_last_tool_output(&current_messages, 500);
                            let feedback = build_cycle_guard_feedback_with_context(
                                &ci.pattern,
                                ci.repeats,
                                last_out,
                            );

                            let (body_new, msgs_new) = guard_roundtrip(
                                feedback,
                                current_messages,
                                payload.clone(),
                                client,
                                provider,
                                otel_event_manager,
                                attempt,
                                body_curr.clone(),
                            )
                            .await?;

                            body_curr = body_new;
                            current_messages = msgs_new;

                            // Сразу синхронизируем messages обратно в payload (ВАЖНО)
                            if let Some(obj) = payload.as_object_mut() {
                                obj.insert(
                                    "messages".into(),
                                    serde_json::Value::Array(current_messages.clone()),
                                );
                            }
                        }
                    }
                }
                // --- КОНЕЦ CYCLE-GUARD ---

                // === NEW: Cleanroom single-shot, если проблема именно в синтаксисе/JSON аргументов ===
                {
                    // Список «жёстких» причин синтаксической поломки
                    let reasons_now = diagnose_tool_call_issues(&body_curr);
                    let has_syntaxy_issue = reasons_now.iter().any(|r| {
                        r.contains("arguments is string but not valid JSON")
                            || r.contains("arguments exists but is not an object or string")
                            || r.contains("textual <tool_call>: 'arguments' is not a JSON object")
                            || r.contains("textual <tool_call> present but JSON failed to parse")
                    });

                    // Вытянем «битый» сырой кусок, который будем чинить
                    let mut bad_snippet = String::new();
                    if let Some(msg) = body_curr
                        .get("choices")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.get(0))
                        .and_then(|c| c.get("message"))
                    {
                        if let Some(tc) = msg
                            .get("tool_calls")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.get(0))
                        {
                            bad_snippet = serde_json::to_string_pretty(tc).unwrap_or_default();
                        } else if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                            // Фоллбек: модель вернула <tool_call> в тексте
                            bad_snippet = content.to_string();
                        }
                    }

                    if has_syntaxy_issue && !bad_snippet.is_empty() {
                        tracing::debug!(
                            target: "codex.chat_sync",
                            "Cleanroom-fix: triggering (syntax issue). reasons={}",
                            reasons_now.join(" | ")
                        );

                        // tools_json мы уже собираем выше — переиспользуем
                        // перед вызовом соберём &str
                        let model_slug: &str = model_family.slug.as_str(); // если slug = String
                        let bad_snippet_str: &str = bad_snippet.as_str(); // String -> &str
                        let reasons_str: String = reasons_now.join(" | "); // временный String сохраняем
                        let reasons_ref: &str = reasons_str.as_str(); // -> &str
                        let tools_value = serde_json::Value::Array(tools_json.clone());

                        let clean = request_cleanroom_fix(
                            client,
                            provider,
                            otel_event_manager,
                            model_slug,      // &str
                            &tools_value,    // &Value ок
                            bad_snippet_str, // &str
                            reasons_ref,     // &str
                            attempt,
                        )
                        .await;

                        match clean {
                            Ok(fixed_body) => {
                                tracing::debug!("Cleanroom-fix: got alternative body; applying.");
                                body_curr = fixed_body;
                                // Подшиваем ассистентское сообщение, чтобы последующие раунды (если будут) видели контекст
                                append_assistant_message_from_body(
                                    &mut current_messages,
                                    &body_curr,
                                );
                            }
                            Err(e) => {
                                tracing::warn!("Cleanroom-fix failed: {e}");
                                // продолжаем обычным путём, не прерывая сценарий
                                append_assistant_message_from_body(
                                    &mut current_messages,
                                    &body_curr,
                                );
                            }
                        }
                    } else {
                        // Нет синтаксической проблемы — просто подшиваем текущий ответ в историю
                        append_assistant_message_from_body(&mut current_messages, &body_curr);
                    }
                }

                // === РАУНД ПОВТОРНОГО ЗАПРОСА У МОДЕЛИ ДЛЯ «ЧИСТОГО» ВЫЗОВА ===
                let model_roundtrip_max: usize = 2;

                // если уже чисто — вообще не запускаем clean-round
                if is_clean_single_tool_call(&body_curr) {
                    let (tx_event, rx_event) =
                        tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                    // === NEW: если был цикл — сначала эмитим нашу подсказку ===
                    if let Some(note) = cycle_guard_note.take() {
                        let item = ResponseItem::Message {
                            role: "user".to_string(),
                            content: vec![ContentItem::InputText { text: note }],
                            id: None,
                        };
                        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                    }

                    emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                    return Ok(ResponseStream { rx_event });
                }

                for round in 0..=model_roundtrip_max {
                    // Если уже получили корректный единичный вызов инструмента — выходим сразу.
                    if is_clean_single_tool_call(&body_curr) {
                        debug!(
                            "Clean-round short-circuit (round {}/{}): already clean; skipping further repairs.",
                            round, model_roundtrip_max
                        );
                        let (tx_event, rx_event) =
                            tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                        // === NEW: emit cycle note first if present
                        if let Some(note) = cycle_guard_note.take() {
                            let item = ResponseItem::Message {
                                role: "user".to_string(),
                                content: vec![ContentItem::InputText { text: note }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                        return Ok(ResponseStream { rx_event });
                    }

                    // Если structured tool_calls отсутствуют И нет текстовой разметки — нечего чинить.
                    if !has_any_tool_call(&body_curr) {
                        debug!(
                            "Clean-round short-circuit (round {}/{}): no tool_calls or textual markup.",
                            round, model_roundtrip_max
                        );
                        let (tx_event, rx_event) =
                            tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                        // === NEW: emit cycle note first if present
                        if let Some(note) = cycle_guard_note.take() {
                            let item = ResponseItem::Message {
                                role: "user".to_string(),
                                content: vec![ContentItem::InputText { text: note }],
                                id: None,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }

                        emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                        return Ok(ResponseStream { rx_event });
                    }

                    // Диагностика, чтобы понимать, ПОЧЕМУ просим «чистый» вызов.
                    let reasons_now = diagnose_tool_call_issues(&body_curr);
                    if reasons_now.is_empty() {
                        // Никаких проблем не нашли — не запускаем искусственные раунды «починки».
                        debug!(
                            "Clean-round skipped (round {}/{}): no issues detected.",
                            round, model_roundtrip_max
                        );
                        break;
                    } else {
                        debug!(
                            "Clean-round (round {}/{}): reasons={}",
                            round,
                            model_roundtrip_max,
                            reasons_now.join(" | ")
                        );
                    }

                    // Возьмём кусок «плохого» ответа для подсказки модели (если есть текстовая разметка).
                    let bad_snippet = body_curr
                        .get("choices")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Сообщение пользователем: просим выслать единственный корректный вызов без обвесов.
                    let mut msgs = current_messages.clone();
                    let mut feedback = build_simple_repair_feedback();
                    if !reasons_now.is_empty() {
                        feedback.push_str("\nПричины: ");
                        feedback.push_str(&reasons_now.join(" | "));
                    }
                    if !bad_snippet.trim().is_empty() {
                        feedback.push_str("\nПроблемный фрагмент:\n");
                        feedback.push_str(&bad_snippet);
                    }
                    msgs.push(serde_json::json!({ "role": "user", "content": feedback }));

                    // Готовим повторный payload с обновлённой историей.
                    let mut payload_retry = payload.clone();
                    if let Some(obj) = payload_retry.as_object_mut() {
                        obj.insert("messages".into(), serde_json::Value::Array(msgs.clone()));
                    }

                    debug!(
                        "Requesting a clean single <tool_call> from model (round {}/{}).",
                        round, model_roundtrip_max
                    );

                    // Повторный запрос к модели.
                    let req_builder_retry = provider.create_request_builder(client, &None).await?;
                    let resp_retry = otel_event_manager
                        .log_request((attempt as u64) + 1, || {
                            req_builder_retry
                                .header(reqwest::header::ACCEPT, "application/json")
                                .json(&payload_retry)
                                .send()
                        })
                        .await;

                    match resp_retry {
                        Ok(ok) if ok.status().is_success() => {
                            body_curr = ok.json().await.map_err(|e| {
                                CodexErr::Fatal(format!(
                                    "failed to decode chat completion JSON (roundtrip): {e}"
                                ))
                            })?;
                            // Включаем ответ ассистента в историю, чтобы следующая итерация видела контекст.
                            append_assistant_message_from_body(&mut current_messages, &body_curr);
                            // цикл продолжится: либо станет "чисто" и мы выйдем в начале, либо сделаем ещё один раунд
                        }
                        Ok(bad) => {
                            warn!("roundtrip request failed with status={}", bad.status());
                            // Прерываем clean-round и эмитим то, что есть.
                            let (tx_event, rx_event) =
                                tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                            // === NEW: emit cycle note first if present
                            if let Some(note) = cycle_guard_note.take() {
                                let item = ResponseItem::Message {
                                    role: "user".to_string(),
                                    content: vec![ContentItem::InputText { text: note }],
                                    id: None,
                                };
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }

                            emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                            return Ok(ResponseStream { rx_event });
                        }
                        Err(e) => {
                            warn!("roundtrip request error: {}", e);
                            let (tx_event, rx_event) =
                                tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                            // === NEW: emit cycle note first if present
                            if let Some(note) = cycle_guard_note.take() {
                                let item = ResponseItem::Message {
                                    role: "user".to_string(),
                                    content: vec![ContentItem::InputText { text: note }],
                                    id: None,
                                };
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }

                            emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                            return Ok(ResponseStream { rx_event });
                        }
                    }
                }

                // Если вышли из цикла без раннего возврата — эмитим как есть.
                let (tx_event, rx_event) = tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                // === NEW: emit cycle note first if present
                if let Some(note) = cycle_guard_note.take() {
                    // let item = ResponseItem::Message {
                    //     role: "user".to_string(),
                    //     content: vec![ContentItem::InputText { text: note }],
                    //     id: None,
                    // };
                    // let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }

                emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                return Ok(ResponseStream { rx_event });
            }
            Ok(res) => {
                let status = res.status();

                if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    // лимит попыток — **до** логики задержек и логов
                    if attempt > max_retries {
                        let req_id_hdr = res
                            .headers()
                            .get("x-request-id")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("");
                        return Err(CodexErr::RetryLimit(RetryLimitReachedError {
                            status,
                            request_id: if req_id_hdr.is_empty() {
                                None
                            } else {
                                Some(req_id_hdr.to_string())
                            },
                        }));
                    }

                    let headers = res.headers().clone();
                    let body = res.text().await.unwrap_or_default();

                    let req_id = headers
                        .get("x-request-id")
                        .or_else(|| headers.get("x-ms-request-id"))
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");

                    // 1) считаем "сырой" кандидат задержки из заголовков/тела/бэкоффа (без порога/джиттера)
                    let (candidate_delay, candidate_reason) =
                        compute_retry_delay_sources(&headers, &body, attempt);

                    // 2) обновляем накопленный дедлайн
                    let now = SystemTime::now();
                    let candidate_until = now + candidate_delay;
                    next_allowed_until = Some(match next_allowed_until {
                        Some(existing) => existing.max(candidate_until),
                        None => candidate_until,
                    });

                    // 3) остаток до накопленного дедлайна
                    let mut remaining = next_allowed_until
                        .unwrap()
                        .duration_since(now)
                        .unwrap_or_else(|_| Duration::from_millis(0));

                    // 4) защитный минимум и "мягкое" увеличение + джиттер
                    let min_wait = Duration::from_millis(2_000);
                    if remaining < min_wait {
                        remaining = min_wait;
                    }
                    remaining = inflate_by_attempt(remaining, attempt); // +10% за попытку
                    remaining = apply_deterministic_jitter(remaining, attempt, 10); // ±10%

                    let headers_dump = dump_headers(&headers, 64, 256);
                    let next_until_str = fmt_system_time(next_allowed_until.unwrap());

                    warn!(
                        target: "codex.chat_sync",
                        "429/5xx: status={} attempt={}/{}; req_id={} \
                         ; candidate_ms={} ({}) \
                         ; accumulated_until={} remaining_ms={} \
                         ; headers: {} \
                         ; body_snippet={}",
                        status.as_u16(),
                        attempt,
                        max_retries,
                        req_id,
                        candidate_delay.as_millis(),
                        candidate_reason,
                        next_until_str,
                        remaining.as_millis(),
                        headers_dump,
                        truncate(&body, 800)
                    );

                    tokio::time::sleep(remaining).await;
                    continue;
                }

                // Неретраибельные статусы (4xx кроме 429)
                let body = res.text().await.unwrap_or_default();
                return Err(CodexErr::UnexpectedStatus(UnexpectedResponseError {
                    status,
                    body,
                    request_id: None,
                }));
            }
            Err(e) => {
                if attempt > max_retries {
                    return Err(CodexErr::ConnectionFailed(ConnectionFailedError {
                        source: e,
                    }));
                }
                let delay = backoff(attempt as u64);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Только источники задержки (заголовки/тело/бэкофф) — "сырой" кандидат.
/// НИЧЕГО не накапливает, не добавляет джиттер и минимумы.
fn compute_retry_delay_sources(
    headers: &HeaderMap,
    body: &str,
    attempt: usize,
) -> (Duration, String) {
    // 1) Тело: retry after Xs
    if let Some(secs) = parse_secs_from_body(body, r"retry after\s+([0-9]+(?:\.[0-9]+)?)s") {
        return (
            Duration::from_millis((secs * 1000.0) as u64),
            format!("body retry-after ~{secs:.3}s"),
        );
    }

    // 2) Retry-After (сек/дата)
    if let Some(v) = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
    {
        if let Ok(secs) = v.trim().parse::<u64>() {
            return (Duration::from_secs(secs), "Retry-After secs".to_string());
        }
        if let Ok(dt) = parse_http_date(v) {
            if let (Ok(now), Ok(ts)) = (
                SystemTime::now().duration_since(UNIX_EPOCH),
                dt.duration_since(UNIX_EPOCH),
            ) {
                if ts > now {
                    return (ts - now, "Retry-After date".to_string());
                }
            }
        }
    }

    // 3) retry-after-ms
    if let Some(v) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok()) {
        if let Ok(ms) = v.trim().parse::<u64>() {
            return (Duration::from_millis(ms), "retry-after-ms".to_string());
        }
    }

    // 4) x-ratelimit-reset = "секунд до окна"
    if let Some(v) = headers
        .get("x-ratelimit-reset")
        .and_then(|v| v.to_str().ok())
    {
        if let Ok(secs) = v.trim().parse::<u64>() {
            return (
                Duration::from_secs(secs),
                "x-ratelimit-reset secs".to_string(),
            );
        }
    }

    // 5) x-ratelimit-reset-* как unix timestamp (сек)
    for k in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"] {
        if let Some(v) = headers.get(k).and_then(|v| v.to_str().ok()) {
            if let Ok(ts) = v.trim().parse::<u64>() {
                if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
                    if ts > now.as_secs() {
                        return (Duration::from_secs(ts - now.as_secs()), k.to_string());
                    }
                }
            }
        }
    }

    // 6) Тело: reset after Ys
    if let Some(secs) = parse_secs_from_body(body, r"reset after\s+([0-9]+(?:\.[0-9]+)?)s") {
        return (
            Duration::from_millis((secs * 1000.0) as u64),
            format!("body reset-after ~{secs:.3}s"),
        );
    }

    // 7) fallback: ваш backoff
    (
        backoff(attempt as u64),
        format!("backoff(attempt={attempt})"),
    )
}

/// +~10% за попытку (1-я — без надбавки)
fn inflate_by_attempt(d: Duration, attempt: usize) -> Duration {
    mul_duration(d, 100 + (attempt as u64 * 10))
}

/// Детерминированный джиттер ±pct% (без rand)
fn apply_deterministic_jitter(d: Duration, attempt: usize, pct: i64) -> Duration {
    let jitter_pct = ((attempt as u64 * 73) % (2 * pct as u64 + 1)) as i64 - pct; // [-pct..+pct]
    mul_duration(d, (100i64 + jitter_pct).max(1) as u64)
}

fn mul_duration(d: Duration, pct: u64) -> Duration {
    let ms = d.as_millis() as u128;
    let new_ms = ms.saturating_mul(pct as u128) / 100u128;
    Duration::from_millis(new_ms.min(u128::from(u64::MAX)) as u64)
}

fn fmt_system_time(t: SystemTime) -> String {
    // простое ISO без внешних зависимостей
    use std::time::UNIX_EPOCH;
    let dur = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    // миллисекундная метка — читаемо в логах
    format!("epoch+{}ms", dur.as_millis())
}

fn dump_headers(h: &HeaderMap, max_pairs: usize, max_val_len: usize) -> String {
    let mut out = String::new();
    for (i, (k, v)) in h.iter().enumerate() {
        if i >= max_pairs {
            out.push('…');
            break;
        }
        let name = k.as_str();
        let val = v.to_str().unwrap_or("<bin>");
        let val_trunc = if val.len() > max_val_len {
            format!("{}…({}b)", &val[..max_val_len], val.len())
        } else {
            val.to_string()
        };
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(name);
        out.push_str(": ");
        out.push_str(&val_trunc);
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    truncate_bytes_safe(s, max)
}

fn shorten(s: &str, max: usize) -> String {
    shorten_bytes_safe(s, max)
}

struct ParsedToolCall {
    name: String,
    arguments: String,
    call_id: String,
}

fn extract_balanced_json(s: &str, start_pos: usize) -> Option<(String, usize)> {
    // ищем первую '{' начиная с start_pos
    let bytes = s.as_bytes();
    let mut i = start_pos;
    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }

    let mut out = String::new();
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut esc = false;

    for (idx, ch) in s[i..].char_indices() {
        let c = ch;
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }

        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            '{' => {
                depth += 1;
                out.push(c);
            }
            '}' => {
                depth -= 1;
                out.push(c);
                if depth == 0 {
                    // позиция после закрывающей скобки в исходной строке
                    let end_pos = i + idx + c.len_utf8();
                    return Some((out, end_pos));
                }
            }
            _ => out.push(c),
        }
    }
    None
}

fn parse_tool_calls_from_content(content: &str) -> Option<(Vec<ParsedToolCall>, String)> {
    // ТОЛЕРАНТНЫЕ теги:
    //  - открывающий: <tool_call ...> (возможны пробелы/атрибуты/переносы)
    //  - закрывающий: </tool_call>
    //  - иногда встречается «незакрытый» вариант "<tool_call" — поддержим как best-effort
    let content = {
        fn norm(s: &str) -> &str {
            let s = s.trim_start_matches(|c: char| {
                c.is_whitespace() || c == '•' || c == '-' || c == '·'
            });
            match s.find('<') {
                Some(pos) => &s[pos..],
                None => s,
            }
        }
        norm(content)
    };
    let open_re = Regex::new(r"(?is)<tool_call\b[^>]*>").ok();
    let close_re = Regex::new(r"(?is)</tool_call\s*>").ok();

    let mut calls = Vec::<ParsedToolCall>::new();
    let mut remainder = String::new();

    let mut pos = 0usize;
    while let (Some(open), Some(close)) = (open_re.as_ref(), close_re.as_ref()) {
        // ищем открывающий тег
        let open_m = match open.find_at(content, pos) {
            Some(m) => m,
            None => break,
        };
        // всё до открывающего — в остаток
        if open_m.start() > pos {
            remainder.push_str(&content[pos..open_m.start()]);
        }

        // ищем закрывающий тег после open
        let close_m = match close.find_at(content, open_m.end()) {
            Some(m) => m,
            None => {
                // нет закрывающего — попробуем всё равно вытащить JSON «вперёд»
                if let Some((json_str, end_json)) = extract_balanced_json(content, open_m.end()) {
                    if let Some(parsed) = parse_tool_call_payload(&json_str) {
                        calls.push(parsed);
                        pos = end_json; // продолжаем после JSON, т.к. тега закрытия нет
                        continue;
                    }
                }
                // не смогли — вернём хвост как есть и остановимся
                remainder.push_str(&content[open_m.start()..]);
                pos = content.len();
                break;
            }
        };

        // между тегами пытаемся достать ровно ОДИН JSON-объект по балансу
        if let Some((json_str, _end_json)) = extract_balanced_json(content, open_m.end()) {
            if let Some(parsed) = parse_tool_call_payload(&json_str) {
                calls.push(parsed);
            } else {
                // JSON не распарсился — вернём блок целиком в остаток
                remainder.push_str(&content[open_m.start()..close_m.end()]);
            }
        } else {
            // не нашли сбалансированный JSON — вернём блок как есть
            remainder.push_str(&content[open_m.start()..close_m.end()]);
        }

        pos = close_m.end();
    }

    // хвост после последнего совпадения
    if pos < content.len() {
        remainder.push_str(&content[pos..]);
    }

    // Если ничего не нашли — fallback на fenced-блоки (оставь как было)
    if calls.is_empty() {
        // 3) fenced-блоки ```tool_call/```json как запасной путь
        let re_fence = Regex::new(r"(?is)```(?:tool_call|json)\s*(\{.*?\})\s*```").ok();
        if let Some(re) = &re_fence {
            let mut last_end = 0usize;
            for m in re.captures_iter(content) {
                if let Some(mat) = m.get(0) {
                    let start = mat.start();
                    if start > last_end {
                        remainder.push_str(&content[last_end..start]);
                    }
                    last_end = mat.end();
                }
                if let Some(json_m) = m.get(1) {
                    if let Some(parsed) = parse_tool_call_payload(json_m.as_str()) {
                        calls.push(parsed);
                    } else if let Some(mat) = m.get(0) {
                        remainder.push_str(mat.as_str());
                    }
                }
            }
            if last_end < content.len() {
                remainder.push_str(&content[last_end..]);
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some((calls, remainder))
    }
}

#[derive(Deserialize)]
struct RawToolCall<'a> {
    #[serde(default)]
    id: Option<&'a str>,
    #[serde(default)]
    r#type: Option<&'a str>,
    #[serde(default)]
    function: Option<RawFunction<'a>>,
    // Альтернативная форма: {"name":..., "arguments":...} (без "function")
    #[serde(default)]
    name: Option<&'a str>,
    #[serde(default, borrow)]
    arguments: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct RawFunction<'a> {
    name: &'a str,
    #[serde(borrow)]
    arguments: &'a RawValue,
    #[serde(default)]
    id: Option<&'a str>,
    #[serde(default)]
    call_id: Option<&'a str>,
}

fn parse_tool_call_payload(raw_json: &str) -> Option<ParsedToolCall> {
    // 1) Пытаемся распарсить как OpenAI-формат с function
    if let Ok(rt) = serde_json::from_str::<RawToolCall>(raw_json) {
        if let Some(fun) = rt.function {
            // arguments как RawValue — "сырой" JSON, но мы хотим "объект" или строку-JSON без внешних кавычек?
            // Поведение — как раньше: если это объект -> to_string(); если это строка -> берём как есть содержимое строки (без внешних кавычек).
            let args_raw = fun.arguments.get();

            // Если начинается с кавычки — это JSON-строка, декодировать НЕ нужно; нам нужен её ДЕКОДИРОВАННЫЙ текст (как раньше).
            // Но важно НИЧЕГО потом не «проэкранировать» лишний раз.
            let arguments = if args_raw.starts_with('"') {
                // Декодируем РОВНО один раз как строку
                match serde_json::from_str::<String>(args_raw) {
                    Ok(s) => s, // s — содержимое строки как его задумала модель, со всеми \n и т.п.
                    Err(_) => return None,
                }
            } else {
                // Объект/массив/… — возвращаем как «сырой» JSON
                args_raw.to_string()
            };

            let call_id = fun.id.or(fun.call_id).unwrap_or_default();
            return Some(ParsedToolCall {
                name: fun.name.to_string(),
                arguments,
                call_id: call_id.to_string(),
            });
        }

        // 2) Альтернативный вид без function: {"name": "...", "arguments": ...}
        if let (Some(name), Some(arguments_raw)) = (rt.name, rt.arguments) {
            let args_raw = arguments_raw.get();
            let arguments = if args_raw.starts_with('"') {
                match serde_json::from_str::<String>(args_raw) {
                    Ok(s) => s,
                    Err(_) => return None,
                }
            } else {
                args_raw.to_string()
            };

            let call_id = rt.id.unwrap_or_default();
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments,
                call_id: call_id.to_string(),
            });
        }
    }

    None
}

fn parse_secs_from_body(body: &str, pattern: &str) -> Option<f64> {
    let re = Regex::new(pattern).ok()?;
    let caps = re.captures(body)?;
    let m = caps.get(1)?;
    m.as_str().trim().parse::<f64>().ok()
}

fn has_any_tool_call(body: &serde_json::Value) -> bool {
    let tool_calls_present = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if tool_calls_present {
        return true;
    }

    if let Some(content) = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
    {
        // более мягкая проверка
        return content.contains("<tool_call")
            || content.contains("```tool_call")
            || content.contains("```json");
    }
    false
}

fn build_simple_repair_feedback() -> String {
    r#"Предыдущий ответ содержит вызов(ы) инструмента, но формат/аргументы могут быть некорректны.
Пожалуйста, пришли заново **один** корректный вызов инструмента.

Требования:
- Отправь **ровно один** блок `<tool_call>{...}</tool_call>`.
- Формат JSON строго валиден. Поле "arguments" — это JSON-объект.
- Никакого текста до/после — только один `<tool_call>...</tool_call>`.

Пример:
<tool_call>
{"name":"TOOL_NAME","arguments":{"key":"value"}}
</tool_call>
"#
    .to_string()
}

/// Разбор нестримингового ответа и эмиссия ваших ResponseEvent’ов.
async fn emit_sync_events_from_chat(
    body: serde_json::Value,
    tx: tokio::sync::mpsc::Sender<Result<ResponseEvent>>,
) -> Result<()> {
    // id / usage (если есть)
    let response_id = body
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    // token_usage опционально: передадим как None (добавьте разбор при необходимости)

    let choices = body
        .get("choices")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Мы обрабатываем только первый choice (как и большинство интеграций).
    if let Some(choice) = choices.get(0) {
        // Reasoning может лежать в choice.message.reasoning (строка или объект).
        let message = choice.get("message").cloned().unwrap_or(json!({}));

        // 1) reasoning (если есть) → отдельным ResponseItem::Reasoning
        if let Some(reasoning_val) = message.get("reasoning") {
            if let Some(s) = reasoning_val.as_str().filter(|s| !s.is_empty()) {
                let item = ResponseItem::Reasoning {
                    id: String::new(),
                    summary: Vec::new(),
                    content: Some(vec![ReasoningItemContent::ReasoningText {
                        text: s.to_string(),
                    }]),
                    encrypted_content: None,
                };
                let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            } else if let Some(obj) = reasoning_val.as_object() {
                if let Some(s) = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                {
                    if !s.is_empty() {
                        let item = ResponseItem::Reasoning {
                            id: String::new(),
                            summary: Vec::new(),
                            content: Some(vec![ReasoningItemContent::ReasoningText {
                                text: s.to_string(),
                            }]),
                            encrypted_content: None,
                        };
                        let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                    }
                }
            }
        }

        let mut message_content_text = message
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // ===== NEW: будем помнить, извлекали ли мы хоть один tool_call =====
        let mut parsed_any_tool_call = false;

        // 2) tool_calls (если есть) → FunctionCall (одним событием на каждый вызов)
        if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                if tc.get("type").and_then(|v| v.as_str()) == Some("function") {
                    let func = tc
                        .get("function")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    let name = func
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();

                    let call_id = tc
                        .get("id")
                        .or_else(|| func.get("id"))
                        .or_else(|| func.get("call_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();

                    let arguments = func
                        .get("arguments")
                        .map(|value| match value {
                            serde_json::Value::String(s) => s.clone(),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        })
                        .unwrap_or_default();

                    let item = ResponseItem::FunctionCall {
                        id: None,
                        name: name.to_string(),
                        arguments,
                        call_id: call_id.to_string(),
                    };
                    parsed_any_tool_call = true; // <-- NEW
                    let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }
            }
        }

        // Фоллбек: парсим из текстового контента (<tool_call> / fenced-блоки)
        if !parsed_any_tool_call {
            if let Some(content) = message_content_text.as_deref() {
                if let Some((fallback_calls, remainder)) = parse_tool_calls_from_content(content) {
                    for fallback_call in fallback_calls {
                        let item = ResponseItem::FunctionCall {
                            id: None,
                            name: fallback_call.name,
                            arguments: fallback_call.arguments,
                            call_id: fallback_call.call_id,
                        };
                        parsed_any_tool_call = true; // <-- NEW
                        let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                    }
                    message_content_text = if remainder.is_empty() {
                        None
                    } else {
                        Some(remainder)
                    };
                }
            }
        }

        // 3) content (если есть и не пустой) → Message(assistant), но только после очистки артефактов tool_call
        if let Some(raw) = message_content_text.take() {
            if let Some(cleaned) = clean_display_text(&raw) {
                let item = ResponseItem::Message {
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText { text: cleaned }],
                    id: None,
                };
                let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            }
        }
    }

    // Завершаем последовательность
    let _ = tx
        .send(Ok(ResponseEvent::Completed {
            response_id,
            token_usage: None,
        }))
        .await;

    Ok(())
}

fn normalize_json_args(s: &str) -> Option<String> {
    // Парсим и детерминированно сериализуем (с отсортированными ключами) —
    // чтобы сравнивать аргументы «по смыслу», а не по пробелам.
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(canonical_json(&v))
}

fn canonical_json(v: &serde_json::Value) -> String {
    use serde_json::Map;
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut ordered = Map::new();
            for k in keys {
                ordered.insert(
                    k.clone(),
                    serde_json::from_str(&canonical_json(&map[k]))
                        .unwrap_or_else(|_| map[k].clone()),
                );
            }
            serde_json::to_string(&ordered).unwrap_or_else(|_| "{}".to_string())
        }
        serde_json::Value::Array(arr) => {
            let canon: Vec<serde_json::Value> = arr
                .iter()
                .map(|x| serde_json::from_str(&canonical_json(x)).unwrap_or_else(|_| x.clone()))
                .collect();
            serde_json::to_string(&canon).unwrap_or_else(|_| "[]".to_string())
        }
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn extract_first_tool_call_from_body(body: &serde_json::Value) -> Option<(String, String)> {
    // OpenAI-формат
    if let Some(tc) = body
        .get("choices")
        .and_then(|v| v.as_array())?
        .get(0)?
        .get("message")?
        .get("tool_calls")?
        .as_array()?
        .get(0)
    {
        if tc.get("type").and_then(|v| v.as_str()) == Some("function") {
            if let Some(func) = tc.get("function").and_then(|v| v.as_object()) {
                let name = func.get("name")?.as_str()?.to_string();
                let arguments = func
                    .get("arguments")
                    .map(|x| match x {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    })
                    .unwrap_or_default();
                return Some((name, arguments));
            }
        }
    }
    // Фоллбек: <tool_call>…</tool_call>
    if let Some(content) = body
        .get("choices")
        .and_then(|v| v.as_array())?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
    {
        if let Some((calls, _)) = parse_tool_calls_from_content(content) {
            if let Some(c) = calls.into_iter().next() {
                return Some((c.name, c.arguments));
            }
        }
    }
    None
}

fn count_recent_same_tool_calls(
    messages: &Vec<serde_json::Value>,
    tool_name: &str,
    args_norm: &str,
) -> usize {
    // Идём с конца messages (последние ассистентские tool_calls, уже записанные в историю)
    let mut count = 0usize;
    for m in messages.iter().rev() {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "assistant" {
            continue;
        }
        if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
            if tcs.len() != 1 {
                break;
            } // считаем только одиночные вызовы подряд
            let tc = &tcs[0];
            if tc.get("type").and_then(|v| v.as_str()) != Some("function") {
                break;
            }
            let fun = match tc.get("function").and_then(|v| v.as_object()) {
                Some(f) => f,
                None => break,
            };
            let name_match = fun.get("name").and_then(|v| v.as_str()) == Some(tool_name);
            let args_raw = fun
                .get("arguments")
                .map(|x| match x {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                })
                .unwrap_or_default();
            let args_norm_prev = normalize_json_args(&args_raw).unwrap_or_default();
            if name_match && args_norm_prev == args_norm {
                count += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    count
}

fn is_clean_single_tool_call(body: &serde_json::Value) -> bool {
    let msg = match body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"))
    {
        Some(m) => m,
        None => return false,
    };

    // content должен быть пустым/Null/пустая строка
    let content_ok = match msg.get("content") {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::String(s)) => s.trim().is_empty(),
        _ => false,
    };

    // ровно один tool_call типа function
    let tcs = match msg.get("tool_calls").and_then(|v| v.as_array()) {
        Some(a) if a.len() == 1 => a,
        _ => return false,
    };
    let tc0 = &tcs[0];
    if tc0.get("type").and_then(|v| v.as_str()) != Some("function") {
        return false;
    }

    // есть function.name
    let func = match tc0.get("function").and_then(|v| v.as_object()) {
        Some(f) => f,
        None => return false,
    };
    if func.get("name").and_then(|v| v.as_str()).is_none() {
        return false;
    }

    // arguments — объект ИЛИ строка, парсящаяся в объект
    match func.get("arguments") {
        Some(serde_json::Value::Object(_)) => { /* ok */ }
        Some(serde_json::Value::String(s)) => {
            match serde_json::from_str::<serde_json::Value>(s) {
                Ok(serde_json::Value::Object(_)) => { /* ok */ }
                _ => return false,
            }
        }
        _ => return false,
    }

    content_ok
}

fn build_repeat_guard_feedback(tool_name: &str) -> String {
    format!(
        r#"Ты трижды подряд попыталась вызвать один и тот же инструмент с теми же параметрами: "{tool_name}".
Пожалуйста, попробуй другой подход.

Требования к следующему ответу:
- Отправь **один** новый вызов инструмента в формате `<tool_call>{{"name":"...","arguments":{{...}}}}</tool_call>`.
- Измени стратегию/параметры (например, другой флаг/путь/шаг, либо другой инструмент).
- Никакого текста вокруг — только один `<tool_call>...</tool_call>`."#,
        tool_name = tool_name
    )
}

fn append_assistant_message_from_body(dst: &mut Vec<serde_json::Value>, body: &serde_json::Value) {
    if let Some(msg) = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"))
        .cloned()
    {
        let role = msg.get("role").cloned().unwrap_or(json!("assistant"));
        let content = msg.get("content").cloned().unwrap_or(json!(null));
        let tool_calls = msg.get("tool_calls").cloned();

        let mut out = json!({ "role": role, "content": content });
        if let Some(tc) = tool_calls {
            if let Some(obj) = out.as_object_mut() {
                obj.insert("tool_calls".to_string(), tc);
            }
        }
        dst.push(out);
    }
}

fn clean_display_text(s: &str) -> Option<String> {
    use regex_lite::Regex;

    // 1) убираем лидирующие буллеты/дефисы/пробелы
    let mut t = s
        .trim_start_matches(|c: char| c.is_whitespace() || c == '•' || c == '-' || c == '·')
        .to_string();

    // 2) вырезаем корректно размеченные <tool_call>...</tool_call>
    if let Ok(re) = Regex::new(r"(?is)<\s*tool_call\b[^>]*>.*?</\s*tool_call\s*>") {
        t = re.replace_all(&t, "").to_string();
    }

    // 3) вырезаем «висящие» одиночные теги <tool_call> и </tool_call>
    if let Ok(re) = Regex::new(r"(?is)</?\s*tool_call\b[^>]*>") {
        t = re.replace_all(&t, "").to_string();
    }

    // 4) вырезаем fenced-блоки ```tool_call ... ``` и ```json ... ```
    if let Ok(re) = Regex::new(r"(?is)```(?:tool_call|json)\s*.*?```") {
        t = re.replace_all(&t, "").to_string();
    }

    // 5) подчистим лишние пустые строки/пробелы
    let t = t.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// Возвращает список причин, почему ответ нельзя принять как корректный единичный вызов инструмента.
/// Эти причины попадут в трейс, чтобы понимать, почему мы делаем clean-round.
fn diagnose_tool_call_issues(body: &serde_json::Value) -> Vec<String> {
    let mut reasons = Vec::<String>::new();

    // Извлекаем message и удобные поля
    let msg = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"));

    if msg.is_none() {
        reasons.push("no choices[0].message".to_string());
        return reasons;
    }
    let msg = msg.unwrap();

    // 1) Кол-во tool_calls
    let tool_calls = msg.get("tool_calls").and_then(|v| v.as_array());
    match tool_calls {
        None => {
            // нет tool_calls — проверим текстовые фоллбеки
            if let Some(content_raw) = msg.get("content").and_then(|v| v.as_str()) {
                // --- FIX: нормализуем в String, чтобы не возвращать &str с некорректным лайфтаймом.
                fn normalize_for_toolcall_scan_owned(s: &str) -> String {
                    let s2 = s.trim_start_matches(|c: char| {
                        c.is_whitespace() || c == '•' || c == '-' || c == '·'
                    });
                    if let Some(pos) = s2.find('<') {
                        s2[pos..].to_string()
                    } else {
                        s2.to_string()
                    }
                }
                let content_norm = normalize_for_toolcall_scan_owned(content_raw);
                let content: &str = &content_norm;

                let contains_tag = content.contains("<tool_call")
                    || content.contains("```tool_call")
                    || content.contains("```json");

                if contains_tag {
                    reasons.push("no structured tool_calls; has textual <tool_call> markup".into());

                    match parse_tool_calls_from_content(content) {
                        Some((calls, _rem)) if calls.len() == 1 => {
                            // arguments должен парситься в JSON-объект
                            let is_obj =
                                serde_json::from_str::<serde_json::Value>(&calls[0].arguments)
                                    .map(|v| matches!(v, serde_json::Value::Object(_)))
                                    .unwrap_or(false);

                            if !is_obj {
                                reasons.push(
                                    "textual <tool_call>: 'arguments' is not a JSON object".into(),
                                );
                            }
                        }
                        Some((calls, _)) if calls.len() > 1 => {
                            reasons.push(format!(
                                "textual <tool_call>: {} calls found (expected 1)",
                                calls.len()
                            ));
                        }
                        _ => {
                            reasons.push(
                                "textual <tool_call> present but JSON failed to parse".into(),
                            );
                        }
                    }
                } else {
                    reasons.push("no tool_calls and no textual <tool_call> markup".into());
                }
            } else {
                reasons.push("no tool_calls and empty content".into());
            }
        }
        Some(arr) => {
            if arr.is_empty() {
                reasons.push("tool_calls array is empty".into());
            }
            if arr.len() > 1 {
                reasons.push(format!("multiple tool_calls: {} (expected 1)", arr.len()));
            }
            if let Some(first) = arr.get(0) {
                // 2) Тип должен быть function
                if first.get("type").and_then(|v| v.as_str()) != Some("function") {
                    reasons.push("tool_call[0].type != 'function'".into());
                }
                // 3) Есть ли имя
                let func = first.get("function").and_then(|v| v.as_object());
                if func.is_none() {
                    reasons.push("tool_call[0].function is missing".into());
                } else {
                    let func = func.unwrap();
                    if func.get("name").and_then(|v| v.as_str()).is_none() {
                        reasons.push("tool_call[0].function.name is missing".into());
                    }
                    // 4) arguments должен быть JSON-объект (или строка, распарсиваемая в объект)
                    let args_raw = func.get("arguments");
                    match args_raw {
                        None => reasons.push("tool_call[0].function.arguments is missing".into()),
                        Some(serde_json::Value::Object(_)) => { /* ок */ }
                        Some(serde_json::Value::String(s)) => {
                            match serde_json::from_str::<serde_json::Value>(s) {
                                Ok(serde_json::Value::Object(_)) => { /* ок */ }
                                Ok(_) => {
                                    reasons.push("arguments parses but is not a JSON object".into())
                                }
                                Err(_) => {
                                    reasons.push("arguments is string but not valid JSON".into())
                                }
                            }
                        }
                        Some(_) => {
                            reasons.push("arguments exists but is not an object or string".into())
                        }
                    }
                }
                // 5) Паразитный текст вокруг (не критично, но причина для «очистки»)
                if let Some(content) = msg.get("content") {
                    match content {
                        serde_json::Value::Null => {}
                        serde_json::Value::String(s) if s.trim().is_empty() => {}
                        _ => {
                            reasons.push("message.content is non-empty alongside tool_call".into())
                        }
                    }
                }
            }
        }
    }

    reasons
}

async fn request_cleanroom_fix(
    client: &CodexHttpClient,
    provider: &ModelProviderInfo,
    otel_event_manager: &OtelEventManager,
    model_slug: &str,
    tools_json: &serde_json::Value,
    bad_snippet: &str,
    reason: &str,
    attempt: usize,
) -> Result<serde_json::Value> {
    // Отдельный чистый контекст: только строгая системка + схема инструментов + пользовательская просьба
    let system = r#"Ты — транслятор вызовов инструментов. Верни РОВНО ОДИН блок:
<tool_call>
{...валидный JSON...}
</tool_call>
Требования:
- НИКАКОГО текста вне блока.
- JSON-объект со строго валидным синтаксисом.
- Поле "arguments" — ОБЯЗАТЕЛЬНО JSON-объект.
- Если нужны кавычки в bash, используй вариант с bash -lc '…' или экранируй кавычки корректно.
- Операции shell (`>`, `2>`, `|`, `&&` и т.п.) должны быть ВНУТРИ одной строки командной строки после `bash -lc`, а не отдельными элементами JSON-массива.
  Пример (верно): ["bash","-lc","echo 'hi' > out.txt"]
  Пример (неверно): ["bash","-lc","echo","'hi'",">","out.txt"]
- Не меняй намерение команды, только синтаксис/экранирование."#;

    // Передаём точную причину и «битый» фрагмент, чтобы модель сфокусировалась
    let user = format!(
        r#"Исправь синтаксис этого вызова инструмента (только кавычки/экранирование), сохранив смысл. Верни ровно один <tool_call>…</tool_call>:

Причина исправления: {reason}

Проблемный фрагмент:
{bad}

Шаблон ответа:
<tool_call>
{{"name":"TOOL_NAME","arguments":{{...}}}}
</tool_call>"#,
        bad = bad_snippet
    );

    let payload = serde_json::json!({
        "model": model_slug,
        "messages": [
            { "role": "system", "content": system },
            // Передаём инструменты как часть системки в отдельном сообщении — так многие модели лучше следуют схеме
            { "role": "system", "content": format!("Инструменты (JSON Schema):\n{}", tools_json) },
            { "role": "user", "content": user }
        ],
        "stream": false,
        // Для «синтаксического» режима — ниже температуру
        "temperature": 0.0,
        "top_p": 0.9,
        "min_p": 0.0
    });

    tracing::debug!(
        target: "codex.chat_sync",
        "Cleanroom-fix: start; reason={}; snippet={}",
        reason,
        shorten(bad_snippet, 280)
    );

    let req = provider.create_request_builder(client, &None).await?;
    let resp = otel_event_manager
        .log_request((attempt as u64) + 1, || {
            req.header(reqwest::header::ACCEPT, "application/json")
                .json(&payload)
                .send()
        })
        .await;

    match resp {
        Ok(ok) if ok.status().is_success() => {
            let body: serde_json::Value = ok
                .json()
                .await
                .map_err(|e| CodexErr::Fatal(format!("cleanroom-fix: decode json failed: {e}")))?;
            tracing::trace!("cleanroom-fix: response: {body:#}");
            Ok(body)
        }
        Ok(bad) => {
            // Снимем всё нужное ДО чтения тела (которое move-ит Response)
            let status = bad.status();
            let headers = bad.headers().clone();

            // Теперь можно "съесть" Response
            let body = bad.text().await.unwrap_or_default();

            // (необязательно) вытащим request_id из заголовков, если есть
            let request_id = headers
                .get("x-request-id")
                .or_else(|| headers.get("x-ms-request-id"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            tracing::warn!(
                "cleanroom-fix failed: status={} req_id={:?} body_snippet={}",
                status,
                request_id,
                truncate(&body, 400) // твоя утилита
            );

            Err(CodexErr::UnexpectedStatus(UnexpectedResponseError {
                status,
                body,
                request_id,
            }))
        }

        Err(e) => Err(CodexErr::ConnectionFailed(ConnectionFailedError {
            source: e,
        })),
    }
}

fn extract_last_tool_output(messages: &Vec<serde_json::Value>, max_chars: usize) -> Option<String> {
    for m in messages.iter().rev() {
        if m.get("role").and_then(|v| v.as_str()) == Some("tool") {
            if let Some(s) = m.get("content").and_then(|v| v.as_str()) {
                // вытащим только "Output:" часть, если есть
                if let Some(pos) = s.find("Output:\n") {
                    let out = &s[pos + "Output:\n".len()..];
                    let out = out.trim();
                    return Some(if out.len() > max_chars {
                        truncate_bytes_safe(out, max_chars)
                    } else {
                        out.to_string()
                    });
                }
                return Some(if s.len() > max_chars {
                    truncate_bytes_safe(s, max_chars)
                } else {
                    s.to_string()
                });
            }
        }
    }
    None
}

fn build_repeat_guard_feedback_with_context(
    tool_name: &str,
    args_json_canonical: &str,
    last_tool_output_snippet: Option<String>,
) -> String {
    let output_line = match last_tool_output_snippet {
        Some(snip) if !snip.is_empty() => format!(
            "\nПоследний вывод инструмента (фрагмент):\n```\n{}\n```\n",
            snip
        ),
        _ => String::new(),
    };

    format!(
        r#"Обнаружено зацикливание: ты **3 раза подряд** вызвала один и тот же инструмент с теми же параметрами.

Инструмент: `{tool}`
Аргументы (канонич.): `{args}`{out}

Пожалуйста, **не повторяй тот же вызов**. Попробуй **другой шаг/путь** либо **другой инструмент**.

Отправь **ровно один** новый вызов инструмента в формате:
<tool_call>
{{"name":"...","arguments":{{...}}}}
</tool_call>

Требования:
- измени стратегию или параметры (другие флаги/путь/команда);
- строго валидный JSON; поле `arguments` — JSON-объект;
- без текста до и после — только один `<tool_call>...</tool_call>`.
"#,
        tool = tool_name,
        args = args_json_canonical,
        out = output_line
    )
}

/// true если tool-ответ не дал полезного выхода (типичный "Output:\n" без данных)
fn is_tool_output_empty(content: &str) -> bool {
    // прежний вариант
    if let Some(pos) = content.find("Output:\n") {
        return content[pos + "Output:\n".len()..].trim().is_empty();
    }
    // ADD: пустой/только служебные строки без "Output:"
    let trimmed = content.trim();
    // типично в логах: "Exit code: 0\nWall time: ...\nOutput:\n"
    // либо просто "Exit code: 0\nWall time: ...\n"
    if trimmed == "Exit code: 0" || trimmed.starts_with("Exit code: 0\n") {
        return true;
    }
    false
}

// ADD: строим последовательность "assistant:shell", "tool:shell", ... с хвоста истории
fn recent_role_tool_seq(messages: &Vec<serde_json::Value>, max_pairs: usize) -> Vec<String> {
    let mut seq = Vec::<String>::new();
    // идём с конца, набираем пары [assistant(tool_call)] -> [tool(...)]
    let mut i = messages.len();
    let mut pairs = 0usize;

    while i >= 2 && pairs < max_pairs {
        let tool_msg = &messages[i - 1];
        let asst_msg = &messages[i - 2];

        let role_tool = tool_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let role_asst = asst_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");

        // требуем строго пару assistant→tool
        if role_tool != "tool" || role_asst != "assistant" {
            break;
        }

        // имя инструмента из assistant.tool_calls[0]
        let tcs = match asst_msg.get("tool_calls").and_then(|v| v.as_array()) {
            Some(a) if !a.is_empty() => a,
            _ => break,
        };
        let tc = &tcs[0];
        let tool_name = tc
            .get("function")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        seq.push(format!("assistant:{tool_name}"));
        seq.push(format!("tool:{tool_name}"));

        pairs += 1;
        i -= 2;
    }

    seq.reverse(); // хронологический порядок
    seq
}

/// Ищет в конце истории чередование пар (assistant tool_call -> tool с пустым выводом)
/// Возвращает (repeats, tool_name), если таких подряд >= min_repeats.
fn detect_zero_progress_loop(
    messages: &Vec<serde_json::Value>,
    min_repeats: usize,
    max_scan: usize,
) -> Option<(usize, String)> {
    // Идём с конца, собираем пары: [assistant(tool_call)], затем [tool(tool_call_id...)]
    let mut repeats = 0usize;
    let mut last_tool_name: Option<String> = None;

    let mut i = messages.len();
    let mut scanned = 0usize;

    while i >= 2 && scanned < max_scan {
        let tool_msg = &messages[i - 1];
        let asst_msg = &messages[i - 2];

        // проверяем, что это tool-ответ и assistant-вызов перед ним
        let role_tool = tool_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let role_asst = asst_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");

        if role_tool != "tool" || role_asst != "assistant" {
            break;
        }

        // вытащим имя инструмента из assistant.tool_calls[0]
        let tcs = match asst_msg.get("tool_calls").and_then(|v| v.as_array()) {
            Some(a) if !a.is_empty() => a,
            _ => break,
        };
        let tc = &tcs[0];
        if tc.get("type").and_then(|v| v.as_str()) != Some("function") {
            break;
        }
        let func = match tc.get("function").and_then(|v| v.as_object()) {
            Some(o) => o,
            None => break,
        };
        let tool_name = match func.get("name").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => break,
        };

        // проверяем пустой вывод tool’а
        let content = tool_msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !is_tool_output_empty(content) {
            break;
        }

        // имя должно совпадать между повторами
        if let Some(prev) = &last_tool_name {
            if prev != &tool_name {
                break;
            }
        } else {
            last_tool_name = Some(tool_name.clone());
        }

        repeats += 1;
        scanned += 2;
        if i >= 2 {
            i -= 2;
        } else {
            break;
        }
    }

    if repeats >= min_repeats {
        Some((repeats, last_tool_name.unwrap_or_default()))
    } else {
        None
    }
}

fn build_cycle_guard_feedback_zero_progress(repeats: usize, tool_name: &str) -> String {
    format!(
        "[cycle-guard] Зацикливание: инструмент `{tool}` дал пустой вывод {n}× подряд.\n\
         Пожалуйста, смени стратегию и сделай **иной следующий шаг** (не повторяй те же команды).\n\
         Требования:\n\
         - Отправь **ровно один** новый вызов инструмента с валидным JSON в `arguments`.\n\
         - Никакого текста вокруг — только `<tool_call>{{...}}</tool_call>`.\n",
        tool = tool_name,
        n = repeats
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn parses_tool_call_xml_like() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":[\"bash\",\"-lc\"],\"workdir\":\"/tmp\"}}\n</tool_call>",
                    "tool_calls": []
                }
            }]
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_sync_events_from_chat(body, tx).await.unwrap();

        let first = rx.recv().await.unwrap().unwrap();
        if let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name, arguments, ..
        }) = first
        {
            assert_eq!(name, "shell");
            assert_eq!(
                arguments,
                "{\"command\":[\"bash\",\"-lc\"],\"workdir\":\"/tmp\"}"
            );
        } else {
            panic!("unexpected");
        }
    }

    #[tokio::test]
    async fn parses_tool_call_from_fenced_block() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "some text\n```tool_call\n{\"name\":\"shell\",\"arguments\":{\"command\":[\"bash\",\"-lc\"]}}\n```\nmore text",
                    "tool_calls": []
                }
            }]
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_sync_events_from_chat(body, tx).await.unwrap();
        let first = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            first,
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { .. })
        ));
    }

    #[tokio::test]
    async fn parses_tool_call_from_content_fallback() {
        let body = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"bash\", \"-lc\"], \"workdir\": \"/tmp\"}}\n<tool_call>",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }]
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_sync_events_from_chat(body, tx).await.unwrap();

        let first = rx.recv().await.expect("first event").expect("event ok");
        match first {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }) => {
                assert_eq!(name, "shell");
                assert_eq!(call_id, "");
                assert_eq!(
                    arguments,
                    "{\"command\":[\"bash\",\"-lc\"],\"workdir\":\"/tmp\"}"
                );
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let second = rx.recv().await.expect("completed").expect("event ok");
        assert!(matches!(second, ResponseEvent::Completed { .. }));

        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn parses_tool_call_arguments_object() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_42",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": {
                                "command": [
                                    "find",
                                    "/tmp/project",
                                    "-name",
                                    "gradlew",
                                    "-o",
                                    "-name",
                                    "gradlew.bat"
                                ],
                                "workdir": "/tmp/project"
                            }
                        }
                    }]
                }
            }]
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_sync_events_from_chat(body, tx).await.unwrap();

        match rx.recv().await.expect("first event").expect("event ok") {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }) => {
                assert_eq!(name, "shell");
                assert_eq!(call_id, "call_42");
                assert_eq!(
                    arguments,
                    "{\"command\":[\"find\",\"/tmp/project\",\"-name\",\"gradlew\",\"-o\",\"-name\",\"gradlew.bat\"],\"workdir\":\"/tmp/project\"}"
                );
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let completed = rx.recv().await.expect("completed").expect("event ok");
        assert!(matches!(completed, ResponseEvent::Completed { .. }));

        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn parses_tool_call_from_nested_function_object() {
        let body = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": r#"<tool_call>
{"id": "call_123", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":[\"bash\",\"-lc\"]}"}}
</tool_call>"#,
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }]
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_sync_events_from_chat(body, tx).await.unwrap();

        let first = rx.recv().await.expect("first event").expect("event ok");
        match first {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }) => {
                assert_eq!(name, "shell");
                assert_eq!(call_id, "call_123");
                assert_eq!(arguments, "{\"command\":[\"bash\",\"-lc\"]}");
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let second = rx.recv().await.expect("completed").expect("event ok");
        assert!(matches!(second, ResponseEvent::Completed { .. }));

        assert!(rx.recv().await.is_none());
    }
}
