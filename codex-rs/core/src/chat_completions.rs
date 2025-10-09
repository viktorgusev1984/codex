use std::time::Duration;

use crate::ModelProviderInfo;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::error::CodexErr;
use crate::error::Result;
use crate::error::RetryLimitReachedError;
use crate::error::UnexpectedResponseError;
use crate::model_family::ModelFamily;
use crate::openai_tools::create_tools_json_for_chat_completions_api;
use crate::util::backoff;
use bytes::Bytes;
use codex_otel::otel_event_manager::OtelEventManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use reqwest::StatusCode;
use serde_json::json;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

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
    // - If the last emitted message is a user message, drop all reasoning.
    // - Otherwise, for each Reasoning item after the last user message, attach it
    //   to the immediate previous assistant message (stop turns) or the immediate
    //   next assistant anchor (tool-call turns: function/local shell call, or assistant message).
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
            // Only consider reasoning that appears after the last user message.
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

                // Prefer immediate previous assistant message (stop turns)
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

                // Otherwise, attach to immediate next assistant anchor (tool-calls or assistant message)
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
    // in the outbound Chat Completions payload (can happen if a final
    // aggregated assistant message was recorded alongside an earlier partial).
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
                // Skip exact-duplicate assistant messages.
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
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
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
                id,
                call_id: _,
                status,
                action,
            } => {
                // Confirm with API team.
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
                // Omit these items from the conversation history.
                continue;
            }
        }
    }

    let tools_json = create_tools_json_for_chat_completions_api(&prompt.tools)?;
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

    let mut attempt = 0;
    let max_retries = provider.request_max_retries();
    loop {
        attempt += 1;

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
                if !(status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()) {
                    let body = (res.text().await).unwrap_or_default();
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

                let retry_after_secs = res
                    .headers()
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

/// Lightweight SSE processor for the Chat Completions streaming format. The
/// output is mapped onto Codex's internal [`ResponseEvent`] so that the rest
/// of the pipeline can stay agnostic of the underlying wire format.
async fn process_chat_sse<S>(
    stream: S,
    tx_event: mpsc::Sender<Result<ResponseEvent>>,
    idle_timeout: Duration,
    otel_event_manager: OtelEventManager,
) where
    S: Stream<Item = Result<Bytes>> + Unpin,
{
    let mut stream = stream.eventsource();

    // Состояние для нового формата (несколько tool_calls по индексу)
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
                // Stream closed gracefully – emit Completed with dummy id.
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

        // OpenAI Chat streaming sends a literal string "[DONE]" when finished.
        if sse.data.trim() == "[DONE]" {
            // Emit any finalized items before closing so downstream consumers receive
            // terminal events for both assistant content and raw reasoning.
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

        // Parse JSON chunk
        let chunk: serde_json::Value = match serde_json::from_str(&sse.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        trace!("chat_completions received SSE chunk: {chunk:?}");

        let choice_opt = chunk.get("choices").and_then(|c| c.get(0));

        if let Some(choice) = choice_opt {
            // Handle assistant content tokens as streaming deltas.
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

            // Forward any reasoning/thinking deltas if present.
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

            // Some providers only include reasoning on the final message object.
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

            // --- Новый формат: delta.tool_calls[*] c index ---
            if let Some(tool_calls_delta) = choice
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(|tc| tc.as_array())
            {
                for tc in tool_calls_delta {
                    let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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
                        if let Some(args_fragment) = function.get("arguments").and_then(|a| a.as_str())
                        {
                            st.arguments.push_str(args_fragment);
                        }
                    }
                }
            }

            // --- Старый формат: delta.function_call ---
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

            // --- Полный message.tool_calls без дельт (некоторые провайдеры) ---
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

            // --- Завершение тёрна ---
            if let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                match finish_reason {
                    "tool_calls" => {
                        // Сначала финализируем reasoning
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

                        // Эмитим все собранные tool calls в порядке индексов
                        for st in tool_calls.iter().filter(|s| s.active) {
                            let name = st.name.clone().unwrap_or_default();
                            let args = st.arguments.clone();

                            // (Опционально) можно логировать невалидный JSON,
                            // но не правим его - пусть выше по пайплайну решают.
                            if let Err(e) = serde_json::from_str::<serde_json::Value>(&args) {
                                tracing::debug!("tool_call arguments not valid JSON yet: {e}; raw kept");
                            }

                            let item = ResponseItem::FunctionCall {
                                id: None,
                                name,
                                arguments: args,
                                call_id: st.call_id.clone().unwrap_or_default(),
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
                    "function_call" if legacy_fn.active => {
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

                        let item = ResponseItem::FunctionCall {
                            id: None,
                            name: legacy_fn.name.unwrap_or_default(),
                            arguments: legacy_fn.arguments,
                            call_id: legacy_fn.call_id.unwrap_or_default(),
                        };
                        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;

                        let _ = tx_event
                            .send(Ok(ResponseEvent::Completed {
                                response_id: String::new(),
                                token_usage: None,
                            }))
                            .await;
                        return;
                    }
                    "stop" => {
                        // Обычная остановка — финализируем ассистентский текст и reasoning
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
                    _ => {}
                }
            }
        }
    }
}

/// Optional client-side aggregation helper
///
/// Stream adapter that merges the incremental `OutputItemDone` chunks coming from
/// [`process_chat_sse`] into a *running* assistant message, **suppressing the
/// per-token deltas**.  The stream stays silent while the model is thinking
/// and only emits two events per turn:
///
///   1. `ResponseEvent::OutputItemDone` with the *complete* assistant message
///      (fully concatenated).
///   2. The original `ResponseEvent::Completed` right after it.
///
/// This mirrors the behaviour the TypeScript CLI exposes to its higher layers.
///
/// The adapter is intentionally *lossless*: callers who do **not** opt in via
/// [`AggregateStreamExt::aggregate()`] keep receiving the original unmodified
/// events.
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

        // First, flush any buffered events from the previous call.
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
                                    codex_protocol::models::ContentItem::OutputText { text } => Some(text),
                                    _ => None,
                                })
                                {
                                    this.cumulative.push_str(text);
                                }
                                continue;
                            }
                            AggregateMode::Streaming => {
                                if this.cumulative.is_empty() {
                                    return Poll::Ready(Some(Ok(ResponseEvent::OutputItemDone(item))));
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
                        return Poll::Ready(Some(Ok(ResponseEvent::ReasoningContentDelta(delta))));
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
