pub(crate) fn parse_tool_arguments(
    arguments: &str,
) -> Result<Option<serde_json::Value>, serde_json::Error> {
    if arguments.trim().is_empty() {
        return Ok(None);
    }

    serde_json::from_str(arguments).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_arguments_returns_none_for_empty() {
        let parsed = parse_tool_arguments("").expect("should parse");
        assert!(parsed.is_none());
    }

    #[test]
    fn parse_tool_arguments_parses_valid_json() {
        let args = "{\"foo\": \"bar\"}";
        let parsed = parse_tool_arguments(args).expect("should parse");
        let value = parsed.expect("expected value");
        assert_eq!(value["foo"], "bar");
    }

    #[test]
    fn parse_tool_arguments_rejects_invalid_json() {
        let args = "{\"foo\": \"bar\"";
        let err = parse_tool_arguments(args).expect_err("should fail");
        assert!(err.to_string().contains("EOF"));
    }
}
