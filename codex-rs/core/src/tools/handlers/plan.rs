use crate::client_common::tools::ResponsesApiTool;
use crate::client_common::tools::ToolSpec;
use crate::codex::Session;
use crate::function_tool::FunctionCallError;
use crate::openai_tools::JsonSchema;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;
use async_trait::async_trait;
use codex_protocol::plan_tool::UpdatePlanArgs;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use std::collections::BTreeMap;
use std::sync::LazyLock;

pub struct PlanHandler;

const ARGUMENT_PREVIEW_LIMIT: usize = 512;

pub static PLAN_TOOL: LazyLock<ToolSpec> = LazyLock::new(|| {
    let mut plan_item_props = BTreeMap::new();
    plan_item_props.insert("step".to_string(), JsonSchema::String { description: None });
    plan_item_props.insert(
        "status".to_string(),
        JsonSchema::String {
            description: Some("One of: pending, in_progress, completed".to_string()),
        },
    );

    let plan_items_schema = JsonSchema::Array {
        description: Some("The list of steps".to_string()),
        items: Box::new(JsonSchema::Object {
            properties: plan_item_props,
            required: Some(vec!["step".to_string(), "status".to_string()]),
            additional_properties: Some(false.into()),
        }),
    };

    let mut properties = BTreeMap::new();
    properties.insert(
        "explanation".to_string(),
        JsonSchema::String { description: None },
    );
    properties.insert("plan".to_string(), plan_items_schema);

    ToolSpec::Function(ResponsesApiTool {
        name: "update_plan".to_string(),
        description: r#"Updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time.
"#
        .to_string(),
        strict: false,
        parameters: JsonSchema::Object {
            properties,
            required: Some(vec!["plan".to_string()]),
            additional_properties: Some(false.into()),
        },
    })
});

#[async_trait]
impl ToolHandler for PlanHandler {
    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<ToolOutput, FunctionCallError> {
        let ToolInvocation {
            session,
            sub_id,
            call_id,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "update_plan handler received unsupported payload".to_string(),
                ));
            }
        };

        let content =
            handle_update_plan(session.as_ref(), arguments, sub_id.clone(), call_id).await?;

        Ok(ToolOutput::Function {
            content,
            success: Some(true),
        })
    }
}

/// This function doesn't do anything useful. However, it gives the model a structured way to record its plan that clients can read and render.
/// So it's the _inputs_ to this function that are useful to clients, not the outputs and neither are actually useful for the model other
/// than forcing it to come up and document a plan (TBD how that affects performance).
pub(crate) async fn handle_update_plan(
    session: &Session,
    arguments: String,
    sub_id: String,
    _call_id: String,
) -> Result<String, FunctionCallError> {
    let args = parse_update_plan_arguments(&arguments)?;
    session
        .send_event(Event {
            id: sub_id.to_string(),
            msg: EventMsg::PlanUpdate(args),
        })
        .await;
    Ok("Plan updated".to_string())
}

fn parse_update_plan_arguments(arguments: &str) -> Result<UpdatePlanArgs, FunctionCallError> {
    const EXAMPLE_PAYLOAD: &str = r#"{\"explanation\":\"Optional summary\",\"plan\":[{\"step\":\"Investigate the issue\",\"status\":\"in_progress\"},{\"step\":\"Report the findings\",\"status\":\"pending\"}]}"#;
    let received_arguments = summarize_arguments(arguments);

    let invalid_args = |message: String| {
        FunctionCallError::RespondToModel(format!(
            "failed to parse function arguments: {message}. update_plan requires a JSON object with a `plan` array. Each `plan` entry must include a `step` string and a `status` of \"pending\", \"in_progress\", or \"completed\". Example arguments: {EXAMPLE_PAYLOAD}. Received arguments: {received_arguments}"
        ))
    };

    let value = serde_json::from_str::<serde_json::Value>(arguments)
        .map_err(|err| invalid_args(format!("invalid JSON: {err}")))?;

    {
        let obj = value
            .as_object()
            .ok_or_else(|| invalid_args("expected a JSON object".to_string()))?;

        for key in obj.keys() {
            if key != "plan" && key != "explanation" {
                return Err(invalid_args(format!(
                    "unexpected property `{key}`; only `plan` and optional `explanation` are allowed"
                )));
            }
        }

        let plan_value = obj
            .get("plan")
            .ok_or_else(|| invalid_args("missing required `plan` property".to_string()))?;
        let plan_items = plan_value
            .as_array()
            .ok_or_else(|| invalid_args("`plan` must be an array of objects".to_string()))?;

        for (idx, item) in plan_items.iter().enumerate() {
            let Some(item_obj) = item.as_object() else {
                return Err(invalid_args(format!(
                    "plan[{idx}] must be an object with `step` and `status` fields"
                )));
            };

            match item_obj.get("step").and_then(|v| v.as_str()).map(str::trim) {
                Some(step) if !step.is_empty() => {}
                _ => {
                    return Err(invalid_args(format!(
                        "plan[{idx}] must include a non-empty `step` string"
                    )));
                }
            }

            let status_value =
                item_obj
                    .get("status")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        invalid_args(format!("plan[{idx}] must include a `status` value"))
                    })?;
            match status_value {
                "pending" | "in_progress" | "completed" => {}
                other => {
                    return Err(invalid_args(format!(
                        "plan[{idx}] has invalid `status` value `{other}`; use `pending`, `in_progress`, or `completed`"
                    )));
                }
            }

            for key in item_obj.keys() {
                if key != "step" && key != "status" {
                    return Err(invalid_args(format!(
                        "plan[{idx}] has unexpected property `{key}`; only `step` and `status` are allowed"
                    )));
                }
            }
        }

        if let Some(explanation) = obj.get("explanation")
            && !explanation.is_string()
            && !explanation.is_null()
        {
            return Err(invalid_args(
                "`explanation` must be a string if provided".to_string(),
            ));
        }
    }

    serde_json::from_value(value)
        .map_err(|err| invalid_args(format!("invalid plan payload: {err}")))
}

fn summarize_arguments(arguments: &str) -> String {
    let mut preview = String::new();
    let mut chars = arguments.chars();
    for _ in 0..ARGUMENT_PREVIEW_LIMIT {
        match chars.next() {
            Some(ch) => preview.push(ch),
            None => break,
        }
    }
    let truncated = chars.next().is_some();
    let escaped_preview: String = preview.chars().flat_map(char::escape_default).collect();
    if truncated {
        format!("{escaped_preview}... (truncated to {ARGUMENT_PREVIEW_LIMIT} characters)")
    } else {
        escaped_preview
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn parse_error(arguments: &str) -> FunctionCallError {
        parse_update_plan_arguments(arguments).expect_err("expected parse failure")
    }

    fn extract_message(err: FunctionCallError) -> String {
        match err {
            FunctionCallError::RespondToModel(msg) => msg,
            other => panic!("expected RespondToModel error, got {other:?}"),
        }
    }

    fn expect_message(arguments: &str) -> String {
        extract_message(parse_error(arguments))
    }

    #[test]
    fn error_message_includes_received_arguments() {
        let arguments = "{\"explanation\":\"Missing plan data\"}";
        let message = expect_message(arguments);
        assert!(
            message.contains("Received arguments: {"),
            "missing received arguments in message: {message}"
        );
        assert!(
            message.contains("Missing plan data"),
            "missing original argument details in message: {message}"
        );
    }

    #[test]
    fn error_message_truncates_long_arguments() {
        let long_value = "x".repeat(ARGUMENT_PREVIEW_LIMIT + 50);
        let arguments = format!("{{\"plan\":\"{long_value}\"}}");
        let message = expect_message(&arguments);
        assert!(
            message.contains("truncated to"),
            "expected truncation note in message: {message}"
        );
        let expected_prefix: String = arguments
            .chars()
            .take(ARGUMENT_PREVIEW_LIMIT)
            .flat_map(char::escape_default)
            .collect();
        assert!(
            message.contains(&expected_prefix),
            "missing escaped preview of arguments. expected prefix: {expected_prefix}, message: {message}"
        );
    }

    #[test]
    fn summarize_arguments_matches_non_truncated_values() {
        let arguments = "short";
        let summary = summarize_arguments(arguments);
        assert_eq!(summary, "short");
    }

    #[test]
    fn reports_missing_plan_property() {
        let message = expect_message("{\"explanation\":\"hi\"}");
        assert!(
            message.contains("missing required `plan` property"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("Example arguments"),
            "expected message to contain an example payload: {message}"
        );
    }

    #[test]
    fn reports_invalid_status_values() {
        let payload = r#"{"plan":[{"step":"Investigate","status":"waiting"}]}"#;
        let message = expect_message(payload);
        assert!(
            message.contains("invalid `status` value `waiting`"),
            "unexpected message: {message}"
        );
    }
}
