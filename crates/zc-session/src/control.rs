use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::Command;

fn default_params() -> Value {
    json!({})
}

/// Deterministic JSON-Lines request envelope usable by GUI, CLI, tests, and MCP servers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlRequest {
    #[serde(default)]
    pub id: Value,
    pub command: String,
    #[serde(default = "default_params")]
    pub params: Value,
}

impl ControlRequest {
    /// Creates a new control request.
    #[must_use]
    pub fn new(id: impl Into<Value>, command: impl Into<String>, params: Value) -> Self {
        Self {
            id: id.into(),
            command: command.into(),
            params,
        }
    }
}

/// Structured error returned inside [`ControlResponse`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlError {
    pub code: String,
    pub message: String,
}

impl ControlError {
    /// Creates a new structured control error.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Deterministic JSON-Lines response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlResponse {
    pub id: Value,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ControlError>,
}

impl ControlResponse {
    /// Builds a successful control response.
    #[must_use]
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    /// Builds an error control response.
    #[must_use]
    pub fn err(id: Value, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(ControlError::new(code, message)),
        }
    }
}

/// Target capable of executing parsed [`Command`] values for the JSON-Lines control layer.
pub trait ControlTarget {
    /// Executes one command and returns a deterministic JSON result payload or structured error.
    fn dispatch_command(&mut self, command: Command) -> Result<Value, ControlError>;
}

/// Dispatches an already-parsed [`ControlRequest`] against `target`.
pub fn dispatch_request<T: ControlTarget>(target: &mut T, req: ControlRequest) -> ControlResponse {
    let ControlRequest {
        id,
        command,
        params,
    } = req;
    match Command::from_name_and_params(&command, params) {
        Ok(cmd) => match target.dispatch_command(cmd) {
            Ok(result) => ControlResponse::ok(id, result),
            Err(err) => ControlResponse {
                id,
                ok: false,
                result: None,
                error: Some(err),
            },
        },
        Err(msg) => ControlResponse::err(id, "invalid_command", msg),
    }
}

/// Parses and dispatches a single JSON-Lines request string against `target`.
pub fn dispatch_json_line<T: ControlTarget>(target: &mut T, line: &str) -> ControlResponse {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ControlResponse::err(Value::Null, "empty_request", "request line is empty");
    }

    match serde_json::from_str::<ControlRequest>(trimmed) {
        Ok(req) => dispatch_request(target, req),
        Err(err) => ControlResponse::err(
            Value::Null,
            "parse_error",
            format!("invalid JSON control request: {err}"),
        ),
    }
}

/// Parses, dispatches, and serializes a single JSON-Lines response string.
pub fn dispatch_json_line_str<T: ControlTarget>(target: &mut T, line: &str) -> String {
    let response = dispatch_json_line(target, line);
    serde_json::to_string(&response).unwrap_or_else(|err| {
        format!(
            r#"{{"id":null,"ok":false,"error":{{"code":"serialize_error","message":"{}"}}}}"#,
            err.to_string().replace('"', "'")
        )
    })
}

/// Streams newline-delimited JSON requests from `reader` and writes one JSON response per line
/// to `writer`, skipping blank lines. Returns the number of dispatched requests.
pub fn dispatch_json_lines<T, R, W>(
    target: &mut T,
    reader: R,
    mut writer: W,
) -> std::io::Result<usize>
where
    T: ControlTarget,
    R: BufRead,
    W: Write,
{
    let mut count = 0_usize;
    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let out = dispatch_json_line_str(target, &line);
        writeln!(writer, "{out}")?;
        writer.flush()?;
        count = count.saturating_add(1);
    }
    Ok(count)
}
