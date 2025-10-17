use std::time::Duration;

use crate::ModelProviderInfo;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::config_types::SyncChatCompletionsConfig;
use crate::error::CodexErr;
use crate::error::ConnectionFailedError;
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
use regex_lite::Regex;
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use tracing::debug;
use tracing::trace;
use tracing::warn;

/// Синхронный (нестриминговый) вызов Chat Completions API.
pub(crate) async fn chat_completions_sync(
    prompt: &Prompt,
    model_family: &ModelFamily,
    client: &reqwest::Client,
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
                            let args_norm_display = args_norm_now.clone(); // уже канонизированные аргументы

                            let feedback = build_repeat_guard_feedback_with_context(
                                &name_now,
                                &args_norm_display,
                                last_out,
                            );

                            let mut msgs = current_messages.clone();
                            msgs.push(serde_json::json!({ "role": "user", "content": feedback }));

                            let mut payload_retry = payload.clone();
                            if let Some(obj) = payload_retry.as_object_mut() {
                                obj.insert(
                                    "messages".into(),
                                    serde_json::Value::Array(msgs.clone()),
                                );
                            }

                            debug!(
                                "Repeat-guard: detected 3rd identical tool call for '{}', requesting alternative with explicit loop notice.",
                                name_now
                            );

                            let mut payload_retry = payload.clone();
                            if let Some(obj) = payload_retry.as_object_mut() {
                                obj.insert(
                                    "messages".into(),
                                    serde_json::Value::Array(msgs.clone()),
                                );
                            }

                            debug!(
                                "Repeat-guard: detected 3rd identical tool call for '{}', requesting alternative.",
                                name_now
                            );

                            let req_builder_retry =
                                provider.create_request_builder(client, &None).await?;
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
                                            "failed to decode chat completion JSON (repeat-guard): {e}"
                                        ))
                                    })?;
                                    trace!("repeat-guard: got alternative response");
                                    append_assistant_message_from_body(
                                        &mut current_messages,
                                        &body_curr,
                                    );
                                }
                                Ok(bad) => {
                                    warn!(
                                        "repeat-guard roundtrip failed with status={}",
                                        bad.status()
                                    );
                                    // даже если не вышло — добавим ассистента из исходного ответа в историю,
                                    // чтобы дальнейший clean-цикл видел контекст
                                    append_assistant_message_from_body(
                                        &mut current_messages,
                                        &body_curr,
                                    );
                                }
                                Err(e) => {
                                    warn!("repeat-guard roundtrip error: {}", e);
                                    append_assistant_message_from_body(
                                        &mut current_messages,
                                        &body_curr,
                                    );
                                }
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
                            emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                            return Ok(ResponseStream { rx_event });
                        }
                        Err(e) => {
                            warn!("roundtrip request error: {}", e);
                            let (tx_event, rx_event) =
                                tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);
                            emit_sync_events_from_chat(body_curr, tx_event.clone()).await?;
                            return Ok(ResponseStream { rx_event });
                        }
                    }
                }

                // Если вышли из цикла без раннего возврата — эмитим как есть.
                let (tx_event, rx_event) = tokio::sync::mpsc::channel::<Result<ResponseEvent>>(16);
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

    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
        let call_id = value
            .get("id")
            .or_else(|| value.get("call_id"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let arguments = match value.get("arguments") {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(other) => serde_json::to_string(other).unwrap_or_default(),
            None => String::new(),
        };

        return Some(ParsedToolCall {
            name: name.to_string(),
            arguments,
            call_id,
        });
    }

    if let Some(function) = value.get("function").and_then(|v| v.as_object()) {
        let name = function.get("name").and_then(|v| v.as_str())?;
        let call_id = value
            .get("id")
            .or_else(|| value.get("call_id"))
            .or_else(|| function.get("id"))
            .or_else(|| function.get("call_id"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let arguments = match function.get("arguments") {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(other) => serde_json::to_string(other).unwrap_or_default(),
            None => String::new(),
        };

        return Some(ParsedToolCall {
            name: name.to_string(),
            arguments,
            call_id,
        });
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
    // 1) Нормальный OpenAI-формат tool_calls
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

    // 2) Фоллбек по тексту: <tool_call>…</tool_call> или ```tool_call/```json
    if let Some(content) = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
    {
        return content.contains("<tool_call>")
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

        // 2) tool_calls (если есть) → FunctionCall (одним событием на каждый вызов)
        let mut emitted_tool_call = false;
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
            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                let contains_tag = content.contains("<tool_call>")
                    || content.contains("```tool_call")
                    || content.contains("```json");
                if contains_tag {
                    reasons.push("no structured tool_calls; has textual <tool_call> markup".into());
                    // попробуем ещё быстро проверить правильность JSON внутри тега
                    if let Some((calls, _rem)) = parse_tool_calls_from_content(content) {
                        if calls.is_empty() {
                            reasons.push(
                                "textual <tool_call> present but JSON failed to parse".into(),
                            );
                        } else if calls.len() > 1 {
                            reasons.push(format!(
                                "textual <tool_call>: {} calls found (expected 1)",
                                calls.len()
                            ));
                        } else {
                            // один вызов — убедимся, что arguments — корректный JSON-объект
                            if serde_json::from_str::<serde_json::Value>(&calls[0].arguments)
                                .ok()
                                .and_then(|v| v.as_object().cloned())
                                .is_none()
                            {
                                reasons.push(
                                    "textual <tool_call>: 'arguments' is not a JSON object".into(),
                                );
                            }
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
    client: &reqwest::Client,
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
- Не меняй намерение команды, только синтаксис/кавычки/экранирование.
"#;

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

// Короткий pretty-печать в логи
fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}… ({} chars)", &s[..max], s.len())
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
                        format!("{}… ({} chars)", &out[..max_chars], out.len())
                    } else {
                        out.to_string()
                    });
                }
                return Some(if s.len() > max_chars {
                    format!("{}… ({} chars)", &s[..max_chars], s.len())
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
