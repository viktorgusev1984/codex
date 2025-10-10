use tracing::warn;

pub(crate) fn parse_tool_arguments(
    arguments: &str,
) -> Result<Option<serde_json::Value>, serde_json::Error> {
    use serde_json::Value;

    let s = arguments.trim();
    if s.is_empty() {
        return Ok(None);
    }

    // 0) прямая попытка
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return Ok(Some(v));
    }

    // Пайплайн ремонтов (идём от дешёвого к дорогому)
    let mut candidates: Vec<String> = Vec::new();

    if let Some(fixed) = fix_unclosed_json_delimiters(s) {
        candidates.push(fixed);
    }

    let no_trailing = fix_trailing_commas(s);
    if no_trailing != s {
        candidates.push(no_trailing.clone());
    }

    if let Some(wrapped) = wrap_multiple_roots(s) {
        candidates.push(wrapped);
    }

    // Комбинации
    if let Some(fixed) = fix_unclosed_json_delimiters(&no_trailing) {
        candidates.push(fixed);
    }
    if let Some(wrapped) = wrap_multiple_roots(&no_trailing) {
        candidates.push(wrapped);
    }

    for cand in candidates {
        if let Ok(v) = serde_json::from_str::<Value>(&cand) {
            warn!("repaired invalid tool arguments");
            return Ok(Some(v));
        }
    }

    // Вернём исходную ошибку — для сигнатуры Result
    serde_json::from_str::<Value>(s).map(Some)
}

pub(crate) fn repair_tool_arguments(arguments: &str) -> Option<String> {
    use serde_json::Value;

    let s = arguments.trim();
    if s.is_empty() {
        return None;
    }
    if serde_json::from_str::<Value>(s).is_ok() {
        return None; // уже валидно
    }

    // Пайплайн ремонтов
    if let Some(fixed) = fix_unclosed_json_delimiters(s) {
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Some(fixed);
        }
    }

    let no_trailing = fix_trailing_commas(s);
    if no_trailing != s && serde_json::from_str::<Value>(&no_trailing).is_ok() {
        return Some(no_trailing);
    }

    if let Some(wrapped) = wrap_multiple_roots(s) {
        if serde_json::from_str::<Value>(&wrapped).is_ok() {
            return Some(wrapped);
        }
    }

    // Комбинации
    if let Some(fixed) = fix_unclosed_json_delimiters(&no_trailing) {
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Some(fixed);
        }
    }
    if let Some(wrapped) = wrap_multiple_roots(&no_trailing) {
        if serde_json::from_str::<Value>(&wrapped).is_ok() {
            return Some(wrapped);
        }
    }

    None
}

/// Чинит незакрытые {} и [] с учётом строк и экранирования.
/// ВАЖНО: парентезы () не поддерживаем — их нет в JSON.
fn fix_unclosed_json_delimiters(arguments: &str) -> Option<String> {
    let trimmed = arguments.trim_end();
    if trimmed.is_empty() {
        return None;
    }

    let trailing_whitespace = &arguments[trimmed.len()..];
    let mut result = trimmed.to_string();
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in trimmed.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"'  => in_string = false,
                _    => {}
            }
        } else {
            match ch {
                '"' => in_string = true,
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' | ']' => {
                    if let Some(expected) = stack.pop() {
                        if expected != ch {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                _ => {}
            }
        }
    }

    if stack.is_empty() {
        return None;
    }

    while let Some(ch) = stack.pop() {
        result.push(ch);
    }
    result.push_str(trailing_whitespace);
    Some(result)
}

/// Удаляет висячие запятые перед } и ] вне строк.
fn fix_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let ch = chars[i];

        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }

        match ch {
            '"' => { in_string = true; out.push(ch); }
            ',' => {
                if let Some(next) = chars.get(i + 1) {
                    if *next == '}' || *next == ']' {
                        // пропускаем запятую
                        i += 1;
                        continue;
                    }
                }
                out.push(ch);
            }
            _ => out.push(ch),
        }
        i += 1;
    }

    out
}

/// Оборачивает несколько корневых JSON-значений в массив.
/// Примеры: `{...}, {...}` или `}{` → `[{...}, {...}]`
fn wrap_multiple_roots(s: &str) -> Option<String> {
    use serde_json::Value;

    // Быстрая проверка: если одно значение — выходим
    if serde_json::from_str::<Value>(s).is_ok() {
        return None;
    }

    // Разделим на top-level значения: запятая на глубине 0 или стык `}{`, `][`, `}{`, `][`, `}{`, `]{`, etc.
    let parts = split_top_level_values(s)?;
    if parts.len() < 2 {
        return None;
    }

    // Проверим, что каждое — валидный JSON Value
    let mut values = Vec::with_capacity(parts.len());
    for p in parts {
        let p_trim = p.trim();
        if p_trim.is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<Value>(p_trim) {
            values.push(v);
        } else {
            return None;
        }
    }
    if values.len() < 2 {
        return None;
    }

    serde_json::to_string(&values).ok()
}

/// Делит строку на список top-level JSON-значений.
/// Возвращает None, если разбиения нет (одно значение).
fn split_top_level_values(s: &str) -> Option<Vec<String>> {
    let mut parts = Vec::<String>::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0i32;
    let mut start = 0usize;
    let cs: Vec<char> = s.chars().collect();

    for i in 0..cs.len() {
        let ch = cs[i];

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                // разделитель между корневыми значениями
                parts.push(cs[start..i].iter().collect());
                start = i + 1;
            }
            _ => {
                // стык объектов без запятой: `}{` или `][` и т.п.
                if depth == 0 && i > 0 {
                    let prev = cs[i - 1];
                    if (prev == '}' || prev == ']') && (ch == '{' || ch == '[' || ch == '"') {
                        parts.push(cs[start..i].iter().collect());
                        start = i;
                    }
                }
            }
        }
    }

    if start == 0 {
        return None; // не нашли разделителей
    }

    parts.push(cs[start..].iter().collect());
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn parse_tool_arguments_handles_missing_closing_brace() {
        let args = "{\"foo\": \"bar\"";
        let parsed = parse_tool_arguments(args).expect("should parse");
        let value = parsed.expect("expected some value");
        assert_eq!(value["foo"], "bar");
    }

    #[test]
    fn fix_unclosed_json_delimiters_adds_multiple_missing_braces() {
        let args = "{\"outer\": {\"inner\": 1";
        let fixed = fix_unclosed_json_delimiters(args).expect("should fix");
        assert_eq!(fixed, "{\"outer\": {\"inner\": 1}}");
    }

    #[test]
    fn fix_unclosed_json_delimiters_adds_missing_array_bracket() {
        let args = "{\"items\": [1, 2";
        let fixed = fix_unclosed_json_delimiters(args).expect("should fix");
        assert_eq!(fixed, "{\"items\": [1, 2]}");
    }

    #[test]
    fn fix_unclosed_json_delimiters_ignores_braces_in_strings() {
        let args = "{\"text\": \"use {curly}\"";
        let fixed = fix_unclosed_json_delimiters(args).expect("should fix");
        assert_eq!(fixed, "{\"text\": \"use {curly}\"}");
    }

    #[test]
    fn fix_unclosed_json_delimiters_preserves_trailing_whitespace() {
        let args = "{\"foo\": 1\n";
        let fixed = fix_unclosed_json_delimiters(args).expect("should fix");
        assert_eq!(fixed, "{\"foo\": 1}\n");
    }

    #[test]
    fn fix_unclosed_json_delimiters_returns_none_when_not_needed() {
        let args = "{\"foo\": true}";
        assert!(fix_unclosed_json_delimiters(args).is_none());
    }

    #[test]
    fn repair_tool_arguments_returns_fixed_string_when_needed() {
        let args = "{\"foo\": 1";
        let repaired = repair_tool_arguments(args).expect("should repair");
        assert_eq!(repaired, "{\"foo\": 1}");
    }

    #[test]
    fn repair_tool_arguments_returns_none_when_valid() {
        let args = "{\"foo\": 1}";
        assert!(repair_tool_arguments(args).is_none());
    }

    #[test]
    fn repair_tool_arguments_returns_none_for_empty() {
        assert!(repair_tool_arguments("").is_none());
    }

    #[test]
    fn fix_trailing_commas_is_removed() {
        let args = "{\"a\":1,}";
        let fixed = fix_trailing_commas(args);
        assert_eq!(fixed, "{\"a\":1}");
        // и массив
        let args2 = "{\"a\":[1,2,]}";
        let fixed2 = fix_trailing_commas(args2);
        assert_eq!(fixed2, "{\"a\":[1,2]}");
    }

    #[test]
    fn wrap_multiple_roots_packs_into_array() {
        let args = "{\"plan\":\"p\"}, {\"step\":\"s1\"}, {\"step\":\"s2\"}";
        let wrapped = wrap_multiple_roots(args).expect("should wrap");
        let v: serde_json::Value = serde_json::from_str(&wrapped).unwrap();
        assert_eq!(v, json!([
            {"plan":"p"},
            {"step":"s1"},
            {"step":"s2"}
        ]));
    }

    #[test]
    fn split_top_level_values_handles_adjacent_objects() {
        let args = "{\"a\":1}{\"b\":2}";
        let parts = split_top_level_values(args).expect("split");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].trim(), "{\"a\":1}");
        assert_eq!(parts[1].trim(), "{\"b\":2}");
    }
}
