use std::collections::HashMap;
use std::sync::Arc;

use crate::client_common::tools::ToolSpec;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::function_tool::FunctionCallError;
use crate::tool_arguments::repair_tool_arguments;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::registry::ConfiguredToolSpec;
use crate::tools::registry::ToolRegistry;
use crate::tools::spec::ToolsConfig;
use crate::tools::spec::build_specs;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::ShellToolCallParams;
use tracing::warn;

#[derive(Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub call_id: String,
    pub payload: ToolPayload,
    pub arguments_repaired: bool,
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
                    Ok(Some(ToolCall {
                        tool_name: name,
                        call_id,
                        payload: ToolPayload::Mcp {
                            server,
                            tool,
                            raw_arguments: arguments,
                        },
                        arguments_repaired: false,
                    }))
                } else {
                    let mut arguments = arguments;
                    let mut arguments_repaired = false;
                    if let Some(fixed) = repair_tool_arguments(&arguments) {
                        warn!(
                            tool_name = %name,
                            "synthesized missing closing delimiters for tool arguments"
                        );
                        arguments = fixed;
                        arguments_repaired = true;
                    }

                    let payload = if name == "unified_exec" {
                        ToolPayload::UnifiedExec { arguments }
                    } else {
                        ToolPayload::Function { arguments }
                    };
                    Ok(Some(ToolCall {
                        tool_name: name,
                        call_id,
                        payload,
                        arguments_repaired,
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
                arguments_repaired: false,
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
                            arguments_repaired: false,
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
            arguments_repaired,
        } = call;
        if arguments_repaired {
            warn!(
                tool_name = %tool_name,
                call_id = %call_id,
                "rejecting repaired tool arguments"
            );
            return Ok(ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputPayload {
                    content: "Tool arguments appeared truncated. Please regenerate the full JSON payload and try again.".to_string(),
                    success: Some(false),
                },
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::Session;
    use crate::codex::TurnContext;
    use crate::codex::make_session_and_context;
    use crate::turn_diff_tracker::TurnDiffTracker;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    #[tokio::test]
    async fn repaired_arguments_are_rejected_with_failure_response() {
        let (session, turn_context) = make_session_and_context();
        let session: Arc<Session> = Arc::new(session);
        let turn: Arc<TurnContext> = Arc::new(turn_context);
        let router = ToolRouter::from_config(&turn.tools_config, None);

        let item = ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            arguments: "{\"command\": [\"bash\", \"-lc\", \"cat <<'EOF' > AGENTS.md\"".to_string(),
            call_id: "call-42".to_string(),
        };

        let call = ToolRouter::build_tool_call(session.as_ref(), item)
            .expect("should build call")
            .expect("expected call");
        assert!(call.arguments_repaired);

        let tracker = Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
        let response = router
            .dispatch_tool_call(
                Arc::clone(&session),
                Arc::clone(&turn),
                Arc::clone(&tracker),
                "sub".to_string(),
                call.clone(),
            )
            .await
            .expect("should respond to model");

        match response {
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call-42");
                assert_eq!(output.success, Some(false));
                assert!(output.content.contains("Tool arguments appeared truncated"));
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }

        // Dispatching again should continue returning the same failure response
        // without attempting to execute the underlying tool.
        let response = router
            .dispatch_tool_call(session, turn, tracker, "sub".to_string(), call)
            .await
            .expect("should respond to model");

        match response {
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call-42");
                assert_eq!(output.success, Some(false));
                assert!(output.content.contains("Tool arguments appeared truncated"));
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }
}
