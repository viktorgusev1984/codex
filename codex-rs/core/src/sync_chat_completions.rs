use std::time::Duration;

use crate::ModelProviderInfo;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::error::{CodexErr, ConnectionFailedError};
use crate::error::Result;
use crate::error::RetryLimitReachedError;
use crate::error::UnexpectedResponseError;
use crate::model_family::ModelFamily;
use crate::openai_tools::create_tools_json_for_chat_completions_api;
use crate::util::backoff;
use codex_otel::otel_event_manager::OtelEventManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use httpdate::parse_http_date;
use regex_lite::Regex; // новый импорт
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH}; // было Duration — расширяем
use tracing::debug;
use tracing::trace;
use tracing::warn; // warn уже есть — оставь как есть

/// Синхронный (нестриминговый) вызов Chat Completions API.
pub(crate) async fn chat_completions_sync(
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
        "temperature": 0.2,
        // "top_p": 0.8,
        // "top_k":  20,
        // "min_p": 0
    });

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
    let max_retries: usize = provider.request_max_retries() as usize; // если возвращает u64 — кастуем
    // аккумулируем общее "разрешено не раньше чем"
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
                        )))
                    }
                };
                trace!("chat_completions sync response: {body:#}");

                // 3) Превращаем JSON в события
                let (tx_event, rx_event) = tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);

                // Без спавна фоновых задач — просто синхронно заливаем события и закрываем канал.
                emit_sync_events_from_chat(body, tx_event.clone()).await?;

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
                    return Err(CodexErr::ConnectionFailed(ConnectionFailedError { source: e }));
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
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}… ({} chars)", &s[..max], s.len())
    }
}

struct ParsedToolCall {
    name: String,
    arguments: String,
    call_id: String,
}

fn parse_tool_calls_from_content(content: &str) -> Option<(Vec<ParsedToolCall>, String)> {
    // 1) Сначала — нормальный XML-подобный вариант: <tool_call> … </tool_call>
    //   Берём все непересекающиеся пары; флаг (?s) — точка матчится на \n.
    let re_closed = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").ok();

    let mut calls = Vec::<ParsedToolCall>::new();
    let mut remainder = String::new();

    if let Some(re) = &re_closed {
        let mut last_end = 0usize;
        for m in re.captures_iter(content) {
            // Текст до тега — в остаток
            if let Some(mat) = m.get(0) {
                let start = mat.start();
                if start > last_end {
                    remainder.push_str(&content[last_end..start]);
                }
                last_end = mat.end();
            }
            // Сам JSON
            if let Some(json_m) = m.get(1) {
                if let Some(parsed) = parse_tool_call_payload(json_m.as_str()) {
                    calls.push(parsed);
                } else {
                    // Если JSON битый — вернём как есть в остатке
                    if let Some(mat) = m.get(0) {
                        remainder.push_str(mat.as_str());
                    }
                }
            }
        }
        // Хвост после последнего совпадения
        if last_end < content.len() {
            remainder.push_str(&content[last_end..]);
        }

        if !calls.is_empty() {
            return Some((calls, remainder));
        }
        // если пусто — падаем во 2-й режим
        remainder.clear();
    }

    // 2) Режим старой разметки: <tool_call> JSON <tool_call>
    const TAG: &str = "<tool_call>";
    let mut cursor = 0usize;
    let mut expecting_json = false;
    while let Some(rel) = content[cursor..].find(TAG) {
        let abs = cursor + rel;
        if !expecting_json {
            // префикс — в остаток
            remainder.push_str(&content[cursor..abs]);
            cursor = abs + TAG.len();
            expecting_json = true;
        } else {
            // между тегами — JSON
            let raw_json = &content[cursor..abs];
            if let Some(parsed) = parse_tool_call_payload(raw_json) {
                calls.push(parsed);
            } else {
                // вернуть как есть, если не распарсили
                remainder.push_str(TAG);
                remainder.push_str(raw_json);
                remainder.push_str(TAG);
            }
            cursor = abs + TAG.len();
            expecting_json = false;
        }
    }
    // Если остались “хвосты”
    if expecting_json {
        remainder.push_str(TAG);
        remainder.push_str(&content[cursor..]);
    } else {
        remainder.push_str(&content[cursor..]);
    }

    if !calls.is_empty() {
        return Some((calls, remainder));
    }

    // 3) Запасной путь: блоки ```tool_call / ```json
    let re_fence = Regex::new(r"(?s)```(?:tool_call|json)\s*(\{.*?\})\s*```").ok();
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

    if calls.is_empty() {
        None
    } else {
        Some((calls, remainder))
    }
}

fn parse_tool_call_payload(raw_json: &str) -> Option<ParsedToolCall> {
    let value: Value = serde_json::from_str(raw_json.trim()).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let call_id = value
        .get("id")
        .or_else(|| value.get("call_id"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    // arguments может быть строкой или объектом — нормализуем к строке JSON
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    };

    Some(ParsedToolCall { name, arguments, call_id })
}

fn parse_secs_from_body(body: &str, pattern: &str) -> Option<f64> {
    let re = Regex::new(pattern).ok()?;
    let caps = re.captures(body)?;
    let m = caps.get(1)?;
    m.as_str().trim().parse::<f64>().ok()
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
        if let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, arguments, .. }) = first {
            assert_eq!(name, "shell");
            assert_eq!(arguments, "{\"command\":[\"bash\",\"-lc\"],\"workdir\":\"/tmp\"}");
        } else { panic!("unexpected"); }
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
        assert!(matches!(first, ResponseEvent::OutputItemDone(ResponseItem::FunctionCall{..})));
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

        // 2) tool_calls (если есть) → FunctionCall (одним событием на каждый вызов)
        let mut emitted_tool_call = false;
        if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                if tc.get("type").and_then(|v| v.as_str()) == Some("function") {
                    let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                    let func = tc
                        .get("function")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    let name = func
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    // OpenAI возвращает arguments как строку JSON — передаём как есть.
                    let arguments = func
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    let item = ResponseItem::FunctionCall {
                        id: None,
                        name: name.to_string(),
                        arguments,
                        call_id: call_id.to_string(),
                    };
                    emitted_tool_call = true;
                    let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }
            }
        }

        if !emitted_tool_call {
            if let Some(content) = message_content_text.as_deref() {
                if let Some((fallback_calls, remainder)) = parse_tool_calls_from_content(content) {
                    for fallback_call in fallback_calls {
                        let item = ResponseItem::FunctionCall {
                            id: None,
                            name: fallback_call.name,
                            arguments: fallback_call.arguments,
                            call_id: fallback_call.call_id,
                        };
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

        // 3) content (если есть и не пустой) → Message(assistant)
        if let Some(content) = message_content_text.as_ref().filter(|s| !s.is_empty()) {
            let item = ResponseItem::Message {
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: content.to_string(),
                }],
                id: None,
            };
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
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
