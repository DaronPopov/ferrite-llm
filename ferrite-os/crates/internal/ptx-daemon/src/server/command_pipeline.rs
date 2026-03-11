use std::io;
use std::sync::Arc;

use crate::commands;
use crate::state::DaemonState;
use serde::Deserialize;

#[derive(Debug)]
pub(super) struct ParsedCommand {
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CommandEnvelope {
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

pub(super) fn parse_command_line(cmdline: &str) -> Result<ParsedCommand, String> {
    let trimmed = cmdline.trim();
    if trimmed.is_empty() {
        return Err("empty command".to_string());
    }

    // New structured protocol: {"command":"run-file","args":["...","..."]}.
    // Keeps argument boundaries and quoting intact.
    if trimmed.starts_with('{') {
        let envelope: CommandEnvelope = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid command envelope: {e}"))?;
        let command = envelope.command.trim().to_lowercase();
        if command.is_empty() {
            return Err("empty command".to_string());
        }
        return Ok(ParsedCommand {
            command,
            args: envelope.args,
        });
    }

    // Legacy text protocol: command and whitespace-separated args.
    let mut parts = trimmed.split_whitespace();
    let command = parts.next().unwrap_or("").to_lowercase();
    if command.is_empty() {
        return Err("empty command".to_string());
    }
    let args = parts.map(|s| s.to_string()).collect();
    Ok(ParsedCommand { command, args })
}

pub(super) fn execute_command(
    state: &Arc<DaemonState>,
    parsed: &ParsedCommand,
) -> io::Result<String> {
    let args_refs: Vec<&str> = parsed.args.iter().map(String::as_str).collect();

    match parsed.command.as_str() {
        "ping" => commands::handle_ping(),
        "status" => commands::handle_status(state),
        "stats" => commands::handle_stats(state),
        "metrics" => commands::handle_metrics(state),
        "snapshot" => commands::handle_snapshot(state),
        "health" => commands::handle_health(state),
        "keepalive" => commands::handle_keepalive(state),
        "apps" => commands::handle_apps(state),
        "app-start" => commands::handle_app_start(state, &args_refs),
        "app-stop" => commands::handle_app_stop(state, &args_refs),
        "run-file" => commands::handle_run_file(state, &args_refs),
        "run-entry" => commands::handle_run_entry(state, &args_refs),
        "run-list" => commands::handle_run_list(state),
        "ferrite-stop" => commands::handle_ferrite_stop(state),
        "shutdown" => {
            state.shutdown();
            Ok("{\"ok\":true,\"message\":\"shutting down\"}\n".to_string())
        }
        "help" => commands::handle_help(),
        "agent" => {
            // Agent protocol: parse JSON from first arg, dispatch to TUI
            let json_str = parsed.args.join(" ");
            if json_str.is_empty() {
                return Ok("{\"error\":\"usage: agent {\\\"cmd\\\":\\\"Status\\\"}\"}\n".to_string());
            }
            match serde_json::from_str::<crate::tui::agent::protocol::AgentCommand>(&json_str) {
                Ok(_cmd) => {
                    // In the socket context we don't have direct TUI access.
                    // The agent command needs to go through the event channel.
                    // For now, return a helpful message.
                    Ok("{\"ok\":true,\"message\":\"agent command parsed — use TUI event channel for execution\"}\n".to_string())
                }
                Err(e) => {
                    Ok(format!("{{\"error\":\"invalid agent command: {}\"}}\n", e))
                }
            }
        }

        // ── scheduler / control plane commands ─────────────────
        "scheduler" => commands::handle_scheduler_command(state, &args_refs),
        "scheduler-status" => commands::handle_scheduler_status(state),
        "scheduler-queue" => commands::handle_scheduler_queue(state),
        "scheduler-policy" => commands::handle_scheduler_policy(state),
        "audit-query" => commands::handle_audit_query(state, &args_refs),
        "events-stream" => commands::handle_events_stream(state),

        // ── FerApp event ingestion ───────────────────────────────
        "app-event" => commands::handle_app_event(state, &args_refs),

        // ── durable job commands (Plan-B) ──────────────────────
        "job-submit" => commands::handle_job_submit(state, &args_refs),
        "job-stop" => commands::handle_job_stop(state, &args_refs),
        "job-status" => commands::handle_job_status(state, &args_refs),
        "job-list" => commands::handle_job_list(state),
        "job-history" => commands::handle_job_history(state, &args_refs),

        _ => {
            Ok(format!(
                "{{\"error\":\"unknown command\",\"command\":\"{}\"}}\n",
                parsed.command
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_command_line;

    #[test]
    fn parses_legacy_text_protocol() {
        let parsed = parse_command_line("run-file foo.rs -- --arg value").unwrap();
        assert_eq!(parsed.command, "run-file");
        assert_eq!(parsed.args, vec!["foo.rs", "--", "--arg", "value"]);
    }

    #[test]
    fn parses_json_envelope_and_preserves_arg_boundaries() {
        let parsed = parse_command_line(
            r#"{"command":"run-file","args":["my script.rs","--","--name","hello world"]}"#,
        )
        .unwrap();
        assert_eq!(parsed.command, "run-file");
        assert_eq!(
            parsed.args,
            vec!["my script.rs", "--", "--name", "hello world"]
        );
    }

    #[test]
    fn rejects_invalid_json_envelope() {
        let err = parse_command_line(r#"{"command":1}"#).unwrap_err();
        assert!(err.contains("invalid command envelope"));
    }
}
