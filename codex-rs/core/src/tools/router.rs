use std::collections::HashMap;
use std::sync::Arc;

use crate::client_common::tools::ToolSpec;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::function_tool::FunctionCallError;
use crate::tool_arguments::parse_tool_arguments;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::registry::ConfiguredToolSpec;
use crate::tools::registry::ToolRegistry;
use crate::tools::spec::ToolsConfig;
use crate::tools::spec::build_specs;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::ShellToolCallParams;

#[derive(Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub call_id: String,
    pub payload: ToolPayload,
}

pub struct ToolRouter {
    registry: ToolRegistry,
    specs: Vec<ConfiguredToolSpec>,
}

impl ToolRouter {
    pub fn from_config(
        config: &ToolsConfig,
        mcp_tools: Option<HashMap<String, mcp_types::Tool>>,
    ) -> Self {
        let builder = build_specs(config, mcp_tools);
        let (specs, registry) = builder.build();

        Self { registry, specs }
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.specs
            .iter()
            .map(|config| config.spec.clone())
            .collect()
    }

    pub fn tool_supports_parallel(&self, tool_name: &str) -> bool {
        self.specs
            .iter()
            .filter(|config| config.supports_parallel_tool_calls)
            .any(|config| config.spec.name() == tool_name)
    }

    pub fn build_tool_call(
        session: &Session,
        item: ResponseItem,
    ) -> Result<Option<ToolCall>, FunctionCallError> {
        match item {
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                if let Some((server, tool)) = session.parse_mcp_tool_name(&name) {
                    validate_tool_arguments(name.as_ref(), &arguments)?;
                    Ok(Some(ToolCall {
                        tool_name: name,
                        call_id,
                        payload: ToolPayload::Mcp {
                            server,
                            tool,
                            raw_arguments: arguments,
                        },
                    }))
                } else {
                    validate_tool_arguments(name.as_ref(), &arguments)?;
                    let payload = if name == "unified_exec" {
                        ToolPayload::UnifiedExec { arguments }
                    } else {
                        ToolPayload::Function { arguments }
                    };
                    Ok(Some(ToolCall {
                        tool_name: name,
                        call_id,
                        payload,
                    }))
                }
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => Ok(Some(ToolCall {
                tool_name: name,
                call_id,
                payload: ToolPayload::Custom { input },
            })),
            ResponseItem::LocalShellCall {
                id,
                call_id,
                action,
                ..
            } => {
                let call_id = call_id
                    .or(id)
                    .ok_or(FunctionCallError::MissingLocalShellCallId)?;

                match action {
                    LocalShellAction::Exec(exec) => {
                        let params = ShellToolCallParams {
                            command: exec.command,
                            workdir: exec.working_directory,
                            timeout_ms: exec.timeout_ms,
                            with_escalated_permissions: None,
                            justification: None,
                        };
                        Ok(Some(ToolCall {
                            tool_name: "local_shell".to_string(),
                            call_id,
                            payload: ToolPayload::LocalShell { params },
                        }))
                    }
                }
            }
            _ => Ok(None),
        }
    }

    pub async fn dispatch_tool_call(
        &self,
        session: Arc<Session>,
        turn: Arc<TurnContext>,
        tracker: SharedTurnDiffTracker,
        sub_id: String,
        call: ToolCall,
    ) -> Result<ResponseInputItem, FunctionCallError> {
        let ToolCall {
            tool_name,
            call_id,
            payload,
        } = call;
        let payload_outputs_custom = matches!(payload, ToolPayload::Custom { .. });
        let failure_call_id = call_id.clone();

        let invocation = ToolInvocation {
            session,
            turn,
            tracker,
            sub_id,
            call_id,
            tool_name,
            payload,
        };

        match self.registry.dispatch(invocation).await {
            Ok(response) => Ok(response),
            Err(FunctionCallError::Fatal(message)) => Err(FunctionCallError::Fatal(message)),
            Err(err) => Ok(Self::failure_response(
                failure_call_id,
                payload_outputs_custom,
                err,
            )),
        }
    }

    fn failure_response(
        call_id: String,
        payload_outputs_custom: bool,
        err: FunctionCallError,
    ) -> ResponseInputItem {
        let message = err.to_string();
        if payload_outputs_custom {
            ResponseInputItem::CustomToolCallOutput {
                call_id,
                output: message,
            }
        } else {
            ResponseInputItem::FunctionCallOutput {
                call_id,
                output: codex_protocol::models::FunctionCallOutputPayload {
                    content: message,
                    success: Some(false),
                },
            }
        }
    }
}

fn validate_tool_arguments(name: &str, arguments: &str) -> Result<(), FunctionCallError> {
    parse_tool_arguments(arguments)
        .map(|_| ())
        .map_err(|error| {
            FunctionCallError::RespondToModel(format!(
                "The {name} tool arguments must be valid JSON that matches the documented schema. Return a single JSON value with no commentary. Parser error: {error}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::validate_tool_arguments;
    use crate::function_tool::FunctionCallError;

    #[test]
    fn validate_tool_arguments_accepts_valid_json() {
        let arguments = "{\"command\": [\"echo\", \"hi\"]}";
        let result = validate_tool_arguments("shell", arguments);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_tool_arguments_accepts_empty_arguments() {
        let result = validate_tool_arguments("shell", "");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_tool_arguments_rejects_invalid_json() {
        let arguments = "{\"command\": [\"echo\" \"hi\"]}";
        let result = validate_tool_arguments("shell", arguments);
        let err = result.expect_err("expected invalid json to fail");
        match err {
            FunctionCallError::RespondToModel(message) => {
                assert!(message.contains("shell"), "unexpected message: {message}");
            }
            other => panic!("expected RespondToModel error, got {other:?}"),
        }
    }
}
