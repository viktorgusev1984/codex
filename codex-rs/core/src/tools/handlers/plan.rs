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
use codex_protocol::plan_tool::PlanItemArg;
use codex_protocol::plan_tool::StepStatus;
use codex_protocol::plan_tool::UpdatePlanArgs;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use serde_json::Map;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;
use tracing::warn;

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

fn parse_update_plan_arguments(arguments: &str) -> Result<UpdatePlanArgs, FunctionCallError> {
    match serde_json::from_str::<UpdatePlanArgs>(arguments) {
        Ok(args) => Ok(args),
        Err(parse_err) => {
            if let Ok(raw) = serde_json::from_str::<Value>(arguments)
                && let Some(coerced) = coerce_update_plan_arguments(raw)
            {
                warn!("coerced update_plan arguments into structured plan");
                Ok(coerced)
            } else {
                Err(FunctionCallError::RespondToModel(format!(
                    "failed to parse function arguments: {parse_err}"
                )))
            }
        }
    }
}

fn coerce_update_plan_arguments(value: Value) -> Option<UpdatePlanArgs> {
    let map = value.as_object()?;
    let explanation = map
        .get("explanation")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(std::string::ToString::to_string);

    let default_status = map
        .get("status")
        .and_then(|value| value.as_str())
        .and_then(parse_status_from_str);

    let plan_value = map.get("plan").or_else(|| map.get("steps")).cloned()?;

    let plan_items = coerce_plan_items(plan_value, default_status);
    if plan_items.is_empty() {
        return None;
    }

    Some(UpdatePlanArgs {
        explanation,
        plan: plan_items,
    })
}

fn coerce_plan_items(value: Value, default_status: Option<StepStatus>) -> Vec<PlanItemArg> {
    match value {
        Value::String(step) => build_plan_item_from_string(step, 0, default_status)
            .into_iter()
            .collect(),
        Value::Array(items) => items
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| match item {
                Value::String(step) => {
                    build_plan_item_from_string(step, index, default_status.clone())
                }
                Value::Object(map) => {
                    build_plan_item_from_object(map, index, default_status.clone())
                }
                _ => None,
            })
            .collect(),
        Value::Object(map) => build_plan_item_from_object(map, 0, default_status)
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

fn build_plan_item_from_string(
    step: String,
    index: usize,
    default_status: Option<StepStatus>,
) -> Option<PlanItemArg> {
    let trimmed = step.trim();
    if trimmed.is_empty() {
        return None;
    }

    let status = if index == 0 {
        default_status.unwrap_or(StepStatus::InProgress)
    } else {
        StepStatus::Pending
    };

    Some(PlanItemArg {
        step: trimmed.to_string(),
        status,
    })
}

fn build_plan_item_from_object(
    mut map: Map<String, Value>,
    index: usize,
    default_status: Option<StepStatus>,
) -> Option<PlanItemArg> {
    let step = map
        .remove("step")
        .or_else(|| map.remove("task"))
        .or_else(|| map.remove("description"))?
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())?
        .to_string();

    let status = map
        .remove("status")
        .and_then(|value| value.as_str().map(std::string::ToString::to_string))
        .and_then(|text| parse_status_from_str(&text))
        .or_else(|| {
            if index == 0 {
                default_status.clone()
            } else {
                None
            }
        })
        .unwrap_or(StepStatus::Pending);

    Some(PlanItemArg { step, status })
}

fn parse_status_from_str(status: &str) -> Option<StepStatus> {
    let normalized = status.trim().to_lowercase().replace([' ', '-'], "_");

    match normalized.as_str() {
        "pending" => Some(StepStatus::Pending),
        "in_progress" => Some(StepStatus::InProgress),
        "inprogress" => Some(StepStatus::InProgress),
        "completed" | "complete" | "done" | "finished" => Some(StepStatus::Completed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn parse_update_plan_arguments_coerces_single_string_plan() {
        let args = parse_update_plan_arguments(
            r#"{"plan": "Create AGENTS.md with repository guidelines", "status": "in_progress"}"#,
        )
        .expect("expected coercion to succeed");

        assert_eq!(args.plan.len(), 1);
        assert_eq!(
            args.plan[0].step,
            "Create AGENTS.md with repository guidelines"
        );
        assert!(matches!(args.plan[0].status, StepStatus::InProgress));
    }

    #[test]
    fn parse_update_plan_arguments_coerces_array_of_strings() {
        let args = parse_update_plan_arguments(
            r#"{"plan": ["Audit repo", "Implement fixes", "Write summary"]}"#,
        )
        .expect("expected coercion to succeed");

        assert_eq!(args.plan.len(), 3);
        assert!(matches!(args.plan[0].status, StepStatus::InProgress));
        assert!(matches!(args.plan[1].status, StepStatus::Pending));
        assert!(matches!(args.plan[2].status, StepStatus::Pending));
    }

    #[test]
    fn parse_update_plan_arguments_coerces_objects_missing_status() {
        let args = parse_update_plan_arguments(
            r#"{"plan": [{"step": "Draft"}, {"step": "Review"}], "status": "completed"}"#,
        )
        .expect("expected coercion to succeed");

        assert!(matches!(args.plan[0].status, StepStatus::Completed));
        assert!(matches!(args.plan[1].status, StepStatus::Pending));
    }
}
