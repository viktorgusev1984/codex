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
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;

pub struct PlanHandler;

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

const PLAN_ARGUMENT_EXAMPLE: &str =
    r#"Example: {"plan": [{"step": "Outline solution", "status": "in_progress"}]}"#;

fn parse_update_plan_arguments(arguments: &str) -> Result<UpdatePlanArgs, FunctionCallError> {
    let mut deserializer = serde_json::Deserializer::from_str(arguments);
    let value: Value = Value::deserialize(&mut deserializer)
        .map_err(|error| map_invalid_json_error(arguments, error))?;

    if let Err(error) = deserializer.end() {
        return Err(map_invalid_json_error(arguments, error));
    }

    validate_plan_arguments(&value)?;
    serde_json::from_value(value).map_err(|error| map_invalid_json_error(arguments, error))
}

fn map_invalid_json_error(arguments: &str, error: serde_json::Error) -> FunctionCallError {
    use serde_json::error::Category;

    let hint = match error.classify() {
        Category::Data => {
            "Ensure `plan` is an array of objects with `step` and `status` (pending, in_progress, completed)."
                .to_string()
        }
        Category::Syntax | Category::Eof => {
            let message = error.to_string();
            if message.contains("trailing characters") {
                trailing_characters_hint(arguments)
            } else {
                "Provide the arguments as a single JSON object with a `plan` array of steps.".to_string()
            }
        }
        Category::Io => "Encountered an unexpected I/O error while reading the arguments.".to_string(),
    };

    FunctionCallError::RespondToModel(format!(
        "failed to parse function arguments: {error}. {hint} {PLAN_ARGUMENT_EXAMPLE}"
    ))
}

fn trailing_characters_hint(arguments: &str) -> String {
    if arguments.contains("},") {
        return "Wrap each step object inside the `plan` array instead of sending multiple top-level JSON objects.".to_string();
    }

    if arguments.contains("\"plan\": \"") && !arguments.contains("\"plan\": [") {
        return "Set `plan` to an array of step objects rather than a string, and include all steps inside that array.".to_string();
    }

    "Provide the arguments as a single JSON object with a `plan` array of steps.".to_string()
}

fn validate_plan_arguments(value: &Value) -> Result<(), FunctionCallError> {
    let obj = value.as_object().ok_or_else(|| {
        respond_with_hint(
            "Provide the arguments as a JSON object with `plan` and optional `explanation`.",
        )
    })?;

    if let Some(explanation) = obj.get("explanation")
        && !(explanation.is_string() || explanation.is_null())
    {
        return Err(respond_with_hint(
            "`explanation` must be a string or null when provided.",
        ));
    }

    let plan_value = obj
        .get("plan")
        .ok_or_else(|| respond_with_hint("Include a `plan` array of step objects."))?;

    let plan_items = plan_value.as_array().ok_or_else(|| {
        respond_with_hint("Wrap the plan steps in an array of objects with `step` and `status`.")
    })?;

    for (index, item) in plan_items.iter().enumerate() {
        let item_obj = item.as_object().ok_or_else(|| {
            respond_with_hint(format!(
                "Plan item {} must be an object with `step` and `status`.",
                index + 1
            ))
        })?;

        if let Some(step_value) = item_obj.get("step") {
            if !step_value.is_string() {
                return Err(respond_with_hint(format!(
                    "`step` must be a string in plan item {}.",
                    index + 1
                )));
            }
        } else {
            return Err(respond_with_hint(format!(
                "Plan item {} is missing `step`.",
                index + 1
            )));
        }

        if let Some(status_value) = item_obj.get("status") {
            let status = status_value.as_str().ok_or_else(|| {
                respond_with_hint(format!(
                    "`status` must be one of pending, in_progress, or completed in plan item {}.",
                    index + 1
                ))
            })?;

            if !matches!(status, "pending" | "in_progress" | "completed") {
                return Err(respond_with_hint(format!(
                    "`status` must be one of pending, in_progress, or completed in plan item {}.",
                    index + 1
                )));
            }
        } else {
            return Err(respond_with_hint(format!(
                "Plan item {} is missing `status`.",
                index + 1
            )));
        }

        let unexpected_fields: Vec<&str> = item_obj
            .keys()
            .filter_map(|key| {
                if key != "step" && key != "status" {
                    Some(key.as_str())
                } else {
                    None
                }
            })
            .collect();

        if !unexpected_fields.is_empty() {
            let fields = unexpected_fields.join(", ");
            return Err(respond_with_hint(format!(
                "Plan item {} has unsupported fields: {}. Only `step` and `status` are allowed.",
                index + 1,
                fields
            )));
        }
    }

    let unexpected_root_fields: Vec<&str> = obj
        .keys()
        .filter_map(|key| {
            if key != "plan" && key != "explanation" {
                Some(key.as_str())
            } else {
                None
            }
        })
        .collect();

    if !unexpected_root_fields.is_empty() {
        let fields = unexpected_root_fields.join(", ");
        return Err(respond_with_hint(format!(
            "Unsupported top-level fields: {fields}. Only `plan` and optional `explanation` are accepted."
        )));
    }

    Ok(())
}

fn respond_with_hint(message: impl Into<String>) -> FunctionCallError {
    let hint = message.into();
    FunctionCallError::RespondToModel(format!(
        "failed to parse function arguments: {hint} {PLAN_ARGUMENT_EXAMPLE}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::plan_tool::StepStatus;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn parse_update_plan_arguments_reports_trailing_content() {
        let args = r#"{"plan": "step one"}, {"plan": "step two"}"#;
        let error = parse_update_plan_arguments(args).expect_err("expected trailing content error");
        match error {
            FunctionCallError::RespondToModel(message) => {
                assert!(
                    message.contains("Wrap each step object inside the `plan` array"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn allows_null_explanation_and_maps_to_none() {
        let args = parse_update_plan_arguments(
            &json!({
                "explanation": null,
                "plan": [
                    {"step": "Do something", "status": "pending"}
                ]
            })
            .to_string(),
        )
        .expect("arguments parsed");

        assert_eq!(args.explanation, None);
        assert_eq!(args.plan.len(), 1);
        assert_eq!(args.plan[0].step, "Do something");
        assert!(matches!(args.plan[0].status, StepStatus::Pending));
    }
}
