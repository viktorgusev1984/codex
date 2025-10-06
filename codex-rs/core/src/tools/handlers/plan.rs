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
    let payload = serde_json::from_str::<serde_json::Value>(arguments).map_err(|e| {
        FunctionCallError::RespondToModel(format!(
            "failed to parse function arguments as JSON object: {e}"
        ))
    })?;

    let payload_obj = payload.as_object().ok_or_else(|| {
        FunctionCallError::RespondToModel(
            "invalid plan payload: expected a JSON object with a `plan` array".to_string(),
        )
    })?;

    let plan_value = payload_obj.get("plan").ok_or_else(|| {
        FunctionCallError::RespondToModel("invalid plan payload: missing `plan` array".to_string())
    })?;

    let plan_items = plan_value.as_array().ok_or_else(|| {
        FunctionCallError::RespondToModel(
            "invalid plan payload: `plan` must be an array of objects like {\"step\": \"...\", \"status\": \"pending\"}".
                to_string(),
        )
    })?;

    for (idx, item) in plan_items.iter().enumerate() {
        let Some(item_obj) = item.as_object() else {
            return Err(FunctionCallError::RespondToModel(format!(
                "invalid plan payload: item {idx} in `plan` is not an object"
            )));
        };

        if !item_obj.contains_key("step") {
            return Err(FunctionCallError::RespondToModel(format!(
                "invalid plan payload: item {idx} in `plan` is missing the `step` field"
            )));
        }

        if !item_obj
            .get("step")
            .map(serde_json::Value::is_string)
            .unwrap_or(false)
        {
            return Err(FunctionCallError::RespondToModel(format!(
                "invalid plan payload: `step` in item {idx} must be a string"
            )));
        }

        if !item_obj.contains_key("status") {
            return Err(FunctionCallError::RespondToModel(format!(
                "invalid plan payload: item {idx} in `plan` is missing the `status` field"
            )));
        }

        if !item_obj
            .get("status")
            .map(serde_json::Value::is_string)
            .unwrap_or(false)
        {
            return Err(FunctionCallError::RespondToModel(format!(
                "invalid plan payload: `status` in item {idx} must be a string"
            )));
        }
    }

    serde_json::from_value::<UpdatePlanArgs>(payload)
        .map_err(|e| FunctionCallError::RespondToModel(format!("invalid plan payload: {e}")))
}
