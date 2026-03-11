//! Event emitter — routes events to the daemon or falls back to tracing.

use serde::Serialize;

use crate::daemon_client::DaemonClient;

/// Routes application events to the daemon event stream, or falls back
/// to `tracing` if no daemon connection is available.
pub(crate) struct Emitter {
    app_name: String,
    tenant_id: Option<u64>,
    client: Option<DaemonClient>,
}

impl Emitter {
    pub fn new(app_name: String, tenant_id: Option<u64>, client: Option<DaemonClient>) -> Self {
        Self {
            app_name,
            tenant_id,
            client,
        }
    }

    /// Emit a named event with a serializable payload.
    pub fn emit(&self, name: &str, payload: &impl Serialize) {
        let payload_json = match serde_json::to_string(payload) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(event = name, error = %e, "failed to serialize event payload");
                return;
            }
        };

        if let Some(client) = &self.client {
            let payload = serde_json::json!({
                "app_name": self.app_name,
                "event_name": name,
                "payload": payload_json,
                "tenant_id": self.tenant_id,
            })
            .to_string();
            let args = vec![payload];
            if let Err(e) = client.send_command_parts("app-event", &args) {
                tracing::warn!(
                    event = name,
                    error = %e,
                    "failed to emit event to daemon, falling back to tracing"
                );
                tracing::info!(
                    app = %self.app_name,
                    event = name,
                    payload = %payload_json,
                    "app event (daemon fallback)"
                );
            }
        } else {
            tracing::info!(
                app = %self.app_name,
                event = name,
                payload = %payload_json,
                "app event"
            );
        }
    }

    /// Emit a log message through the event stream.
    pub fn log(&self, msg: &str) {
        self.emit("log", &serde_json::json!({ "message": msg }));
    }
}
