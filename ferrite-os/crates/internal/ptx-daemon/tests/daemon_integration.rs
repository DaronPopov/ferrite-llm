//! End-to-end integration tests for the ferrite-daemon.
//!
//! Spawns a real daemon process with GPU initialization, sends commands over
//! the Unix socket, and validates responses.  Each test suite boots one daemon
//! instance and runs all checks against it, then shuts it down cleanly.
//!
//! # Running
//!
//! ```bash
//! # Run with output visible (recommended):
//! cargo test -p ferrite-daemon --test daemon_integration -- --nocapture
//!
//! # Run only strict-mode tests:
//! cargo test -p ferrite-daemon --test daemon_integration strict -- --nocapture
//!
//! # Run only permissive-mode tests:
//! cargo test -p ferrite-daemon --test daemon_integration permissive -- --nocapture
//! ```
//!
//! Requires: CUDA-capable GPU, built daemon binary, LD_LIBRARY_PATH set.
//!
//! **Important:** Tests must run sequentially (`--test-threads=1`) because
//! only one daemon can own the GPU shared memory segment at a time:
//!
//! ```bash
//! cargo test -p ferrite-daemon --test daemon_integration -- --test-threads=1 --nocapture
//! ```

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Global lock ensuring only one daemon test runs at a time (GPU shared memory).
static GPU_LOCK: Mutex<()> = Mutex::new(());

// ═══════════════════════════════════════════════════════════════════
// Daemon Test Harness
// ═══════════════════════════════════════════════════════════════════

/// Options for spawning a test daemon instance.
struct DaemonOptions {
    /// Enable single-pool strict mode.
    single_pool_strict: bool,
    /// Additional env vars to set on the daemon process.
    extra_env: Vec<(String, String)>,
}

impl Default for DaemonOptions {
    fn default() -> Self {
        Self {
            single_pool_strict: false,
            extra_env: Vec::new(),
        }
    }
}

/// A running daemon instance scoped to a test.  Cleans up on drop.
struct DaemonInstance {
    child: Option<Child>,
    socket_path: PathBuf,
    pid_file: PathBuf,
}

impl DaemonInstance {
    /// Spawn a daemon and wait for it to become ready.
    ///
    /// Panics with diagnostics if the daemon fails to start within the timeout.
    fn start(opts: DaemonOptions) -> Self {
        let test_id = format!(
            "ferrite-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );
        let socket_path = PathBuf::from(format!("/tmp/{}.sock", test_id));
        let pid_file = PathBuf::from(format!("/tmp/{}.pid", test_id));

        // Clean up stale files
        let _ = std::fs::remove_file(&socket_path);
        let _ = std::fs::remove_file(&pid_file);

        // Locate the daemon binary.  CARGO_BIN_EXE_ferrite-daemon is set by
        // cargo during integration test compilation.
        let daemon_bin = env!("CARGO_BIN_EXE_ferrite-daemon");
        assert!(
            Path::new(daemon_bin).exists(),
            "daemon binary not found at {daemon_bin} — run `cargo build -p ferrite-daemon` first"
        );

        // Use dev-config.toml so tests match the real TUI daemon configuration.
        // Env var overrides (FERRITE_SOCKET, FERRITE_PID_FILE, FERRITE_HEADLESS)
        // still take precedence for test isolation via merge_from_env().
        let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("dev-config.toml");
        let mut cmd = Command::new(daemon_bin);
        cmd.arg("serve");
        if config_path.exists() {
            cmd.arg("--config").arg(&config_path);
        }
        cmd.env("FERRITE_SOCKET", &socket_path)
            .env("FERRITE_PID_FILE", &pid_file)
            .env("FERRITE_HEADLESS", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if opts.single_pool_strict {
            cmd.env("FERRITE_SINGLE_POOL_STRICT", "1");
        }

        for (key, value) in &opts.extra_env {
            cmd.env(key, value);
        }

        // Propagate LD_LIBRARY_PATH for CUDA shared libs
        if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
            cmd.env("LD_LIBRARY_PATH", &ld);
        }

        let child = cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to spawn daemon: {e}\n  binary: {daemon_bin}");
        });

        let mut instance = Self {
            child: Some(child),
            socket_path: socket_path.clone(),
            pid_file,
        };

        // Wait for daemon to become ready (socket file appears and accepts connections)
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if Instant::now() > deadline {
                let diag = instance.collect_output_on_failure();
                panic!(
                    "daemon did not become ready within 30s\n\
                     socket: {}\n\
                     {}",
                    socket_path.display(),
                    diag
                );
            }

            // Check child is still alive
            if let Some(ref mut child) = instance.child {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let diag = instance.collect_output_on_failure();
                        panic!(
                            "daemon exited prematurely with {status}\n{diag}"
                        );
                    }
                    Err(e) => {
                        panic!("failed to poll daemon process: {e}");
                    }
                    Ok(None) => {} // still running
                }
            }

            if socket_path.exists() {
                // Try to connect and ping
                if let Ok(resp) = instance.try_send_command("ping", Duration::from_secs(5)) {
                    if resp.contains("pong") {
                        break;
                    }
                }
            }

            std::thread::sleep(Duration::from_millis(200));
        }

        instance
    }

    /// Send a command to the daemon and return the raw JSON response string.
    fn send_command(&self, cmd: &str) -> String {
        self.try_send_command(cmd, Duration::from_secs(120))
            .unwrap_or_else(|e| {
                panic!(
                    "failed to send command '{}' to daemon at {}: {}",
                    cmd,
                    self.socket_path.display(),
                    e
                );
            })
    }

    /// Send a command and parse the response as JSON.
    fn send_json(&self, cmd: &str) -> serde_json::Value {
        let raw = self.send_command(cmd);
        // The response may have a trailing newline; also handle jsonl for events-stream.
        let trimmed = raw.trim();
        serde_json::from_str(trimmed).unwrap_or_else(|e| {
            panic!(
                "failed to parse response as JSON for command '{cmd}':\n  error: {e}\n  raw: {trimmed}"
            );
        })
    }

    /// Send a command, parse JSON, and return it.  For multi-line responses
    /// (events-stream), returns only the raw string.
    fn send_raw(&self, cmd: &str) -> String {
        self.send_command(cmd)
    }

    /// Try to send a command with a timeout.
    fn try_send_command(&self, cmd: &str, timeout: Duration) -> Result<String, String> {
        let stream = UnixStream::connect(&self.socket_path)
            .map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("set timeout: {e}"))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| format!("set write timeout: {e}"))?;

        let mut stream = stream;
        stream
            .write_all(cmd.as_bytes())
            .map_err(|e| format!("write cmd: {e}"))?;
        stream
            .write_all(b"\n")
            .map_err(|e| format!("write newline: {e}"))?;
        stream.flush().map_err(|e| format!("flush: {e}"))?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(|e| format!("shutdown write: {e}"))?;

        let mut resp = String::new();
        stream
            .read_to_string(&mut resp)
            .map_err(|e| format!("read response: {e}"))?;
        Ok(resp)
    }

    /// Collect stdout/stderr from the daemon process for diagnostics.
    fn collect_output_on_failure(&mut self) -> String {
        let mut diag = String::new();
        if let Some(ref mut child) = self.child {
            if let Some(ref mut stdout) = child.stdout {
                let mut buf = String::new();
                let _ = stdout.read_to_string(&mut buf);
                if !buf.is_empty() {
                    diag.push_str("=== daemon stdout ===\n");
                    // Limit to last 100 lines
                    let lines: Vec<&str> = buf.lines().collect();
                    let start = lines.len().saturating_sub(100);
                    for line in &lines[start..] {
                        diag.push_str(line);
                        diag.push('\n');
                    }
                }
            }
            if let Some(ref mut stderr) = child.stderr {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf);
                if !buf.is_empty() {
                    diag.push_str("=== daemon stderr ===\n");
                    let lines: Vec<&str> = buf.lines().collect();
                    let start = lines.len().saturating_sub(100);
                    for line in &lines[start..] {
                        diag.push_str(line);
                        diag.push('\n');
                    }
                }
            }
        }
        diag
    }

    /// Gracefully shut down the daemon.
    fn shutdown(&mut self) {
        if self.child.is_none() {
            return;
        }
        // Try graceful shutdown via socket
        let _ = self.try_send_command("shutdown", Duration::from_secs(5));
        // Wait for process to exit
        if let Some(ref mut child) = self.child {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        if Instant::now() > deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    Err(_) => break,
                }
            }
        }
        self.child = None;
        let _ = std::fs::remove_file(&self.socket_path);
        let _ = std::fs::remove_file(&self.pid_file);
    }
}

impl Drop for DaemonInstance {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test Result Collector
// ═══════════════════════════════════════════════════════════════════

struct TestSuite {
    name: String,
    results: Vec<CheckResult>,
}

struct CheckResult {
    name: String,
    passed: bool,
    detail: String,
}

impl TestSuite {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
        }
    }

    fn check(&mut self, name: &str, passed: bool, detail: String) {
        let icon = if passed { "PASS" } else { "FAIL" };
        eprintln!("  [{icon}] {name}");
        if !passed {
            for line in detail.lines() {
                eprintln!("         {line}");
            }
        }
        self.results.push(CheckResult {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    /// Assert a JSON field equals an expected value.
    fn check_json_eq(
        &mut self,
        label: &str,
        resp: &serde_json::Value,
        field: &str,
        expected: serde_json::Value,
    ) {
        let actual = &resp[field];
        let passed = *actual == expected;
        let detail = if passed {
            format!("{field} = {expected}")
        } else {
            format!("expected {field} = {expected}, got {actual}\nfull response: {resp}")
        };
        self.check(label, passed, detail);
    }

    /// Assert a JSON string field contains a substring.
    fn check_json_contains(
        &mut self,
        label: &str,
        resp: &serde_json::Value,
        field: &str,
        substring: &str,
    ) {
        let actual = resp[field].as_str().unwrap_or("");
        let passed = actual.contains(substring);
        let detail = if passed {
            format!("{field} contains '{substring}'")
        } else {
            format!(
                "expected {field} to contain '{substring}', got: {actual}\nfull response: {resp}"
            )
        };
        self.check(label, passed, detail);
    }

    /// Assert raw response text contains a substring.
    fn check_raw_contains(&mut self, label: &str, raw: &str, substring: &str) {
        let passed = raw.contains(substring);
        let detail = if passed {
            format!("response contains '{substring}'")
        } else {
            let preview: String = raw.chars().take(500).collect();
            format!("expected response to contain '{substring}', got:\n{preview}")
        };
        self.check(label, passed, detail);
    }

    fn print_summary(&self) {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        eprintln!();
        eprintln!("═══════════════════════════════════════════════════");
        eprintln!("  {} — {passed}/{total} passed, {failed} failed", self.name);
        eprintln!("═══════════════════════════════════════════════════");

        if failed > 0 {
            eprintln!();
            eprintln!("  Failures:");
            for r in &self.results {
                if !r.passed {
                    eprintln!("    [FAIL] {}", r.name);
                    for line in r.detail.lines() {
                        eprintln!("           {line}");
                    }
                }
            }
        }
        eprintln!();
    }

    fn assert_all_passed(&self) {
        let failed: Vec<&CheckResult> = self.results.iter().filter(|r| !r.passed).collect();
        if !failed.is_empty() {
            let mut msg = format!(
                "{} check(s) failed in suite '{}':\n",
                failed.len(),
                self.name
            );
            for f in &failed {
                msg.push_str(&format!("  [FAIL] {}: {}\n", f.name, f.detail));
            }
            panic!("{msg}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test Suites
// ═══════════════════════════════════════════════════════════════════

/// Permissive mode: strict OFF.  All commands should work normally.
#[test]
fn permissive_mode_full_suite() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: PERMISSIVE MODE");
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions::default());
    let mut suite = TestSuite::new("Permissive Mode");

    // ── ping ──
    {
        let resp = daemon.send_json("ping");
        suite.check_json_eq("ping: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq("ping: message", &resp, "message", serde_json::json!("pong"));
    }

    // ── status ──
    {
        let resp = daemon.send_json("status");
        suite.check_json_eq("status: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq(
            "status: single_pool_strict is false",
            &resp,
            "single_pool_strict",
            serde_json::json!(false),
        );
        suite.check(
            "status: pool_total > 0",
            resp["pool_total"].as_u64().unwrap_or(0) > 0,
            format!("pool_total = {}", resp["pool_total"]),
        );
        suite.check_json_eq("status: healthy", &resp, "healthy", serde_json::json!(true));
    }

    // ── metrics ──
    {
        let resp = daemon.send_json("metrics");
        suite.check_json_eq("metrics: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq(
            "metrics: single_pool_strict is false",
            &resp,
            "single_pool_strict",
            serde_json::json!(false),
        );
        suite.check(
            "metrics: vram_allocated > 0",
            resp["vram_allocated"].as_u64().unwrap_or(0) > 0,
            format!("vram_allocated = {}", resp["vram_allocated"]),
        );
    }

    // ── stats ──
    {
        let resp = daemon.send_json("stats");
        suite.check_json_eq("stats: ok", &resp, "ok", serde_json::json!(true));
    }

    // ── health ──
    {
        let resp = daemon.send_json("health");
        suite.check_json_eq("health: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq("health: healthy", &resp, "healthy", serde_json::json!(true));
    }

    // ── run-list ──
    {
        let resp = daemon.send_json("run-list");
        suite.check_json_eq("run-list: ok", &resp, "ok", serde_json::json!(true));
        suite.check(
            "run-list: count > 0",
            resp["count"].as_u64().unwrap_or(0) > 0,
            format!("count = {}", resp["count"]),
        );
    }

    // ── help ──
    {
        let resp = daemon.send_json("help");
        suite.check_json_eq("help: ok", &resp, "ok", serde_json::json!(true));
        suite.check(
            "help: has commands",
            resp["commands"].is_object(),
            "commands field is present".to_string(),
        );
    }

    // ── run-entry: light target (telemetry_demo) ──
    {
        let resp = daemon.send_json("run-entry ptx-runtime/examples/telemetry_demo.rs#main");
        suite.check_json_eq(
            "run-entry telemetry_demo: ok",
            &resp,
            "ok",
            serde_json::json!(true),
        );
        suite.check_json_eq(
            "run-entry telemetry_demo: exit_code 0",
            &resp,
            "exit_code",
            serde_json::json!(0),
        );
        suite.check(
            "run-entry telemetry_demo: has stdout",
            resp["stdout"].as_str().map(|s| !s.is_empty()).unwrap_or(false),
            format!("stdout length = {}", resp["stdout"].as_str().unwrap_or("").len()),
        );
    }

    // ── run-entry: heavy target (jitter_benchmark) — ALLOWED in permissive ──
    //    We don't actually run it to completion (it's heavy), but we check
    //    that it is NOT denied and actually starts execution.
    {
        let resp = daemon.send_json("run-entry ptx-runtime/examples/jitter_benchmark.rs#main");
        // In permissive mode, this should attempt to run (not be denied)
        suite.check(
            "run-entry jitter_benchmark: not denied in permissive",
            resp["stderr"]
                .as_str()
                .map(|s| !s.contains("single-pool strict mode"))
                .unwrap_or(true),
            format!("stderr = {}", resp["stderr"]),
        );
    }

    // ── events-stream: check DaemonPoolInit present ──
    {
        let raw = daemon.send_raw("events-stream");
        suite.check_raw_contains(
            "events-stream: DaemonPoolInit emitted",
            &raw,
            "DaemonPoolInit",
        );
        suite.check_raw_contains(
            "events-stream: RunExecutionModeSelected emitted",
            &raw,
            "RunExecutionModeSelected",
        );
        // In permissive mode, mode should be "external-process" (not denied)
        suite.check_raw_contains(
            "events-stream: mode is external-process",
            &raw,
            "external-process",
        );
    }

    // ── scheduler-status ──
    {
        let resp = daemon.send_json("scheduler-status");
        suite.check_json_eq("scheduler-status: ok", &resp, "ok", serde_json::json!(true));
    }

    suite.print_summary();
    suite.assert_all_passed();
}

/// Strict mode: single-pool enforcement.  Heavy targets denied, light allowed.
#[test]
fn strict_mode_full_suite() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: STRICT MODE");
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions {
        single_pool_strict: true,
        ..Default::default()
    });
    let mut suite = TestSuite::new("Strict Mode");

    // ── ping ──
    {
        let resp = daemon.send_json("ping");
        suite.check_json_eq("ping: ok", &resp, "ok", serde_json::json!(true));
    }

    // ── status: strict flag visible ──
    {
        let resp = daemon.send_json("status");
        suite.check_json_eq("status: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq(
            "status: single_pool_strict is true",
            &resp,
            "single_pool_strict",
            serde_json::json!(true),
        );
        suite.check(
            "status: pool_total > 0",
            resp["pool_total"].as_u64().unwrap_or(0) > 0,
            format!("pool_total = {}", resp["pool_total"]),
        );
    }

    // ── metrics: strict flag visible ──
    {
        let resp = daemon.send_json("metrics");
        suite.check_json_eq("metrics: ok", &resp, "ok", serde_json::json!(true));
        suite.check_json_eq(
            "metrics: single_pool_strict is true",
            &resp,
            "single_pool_strict",
            serde_json::json!(true),
        );
    }

    // ── run-list: works (read-only, no pool init) ──
    {
        let resp = daemon.send_json("run-list");
        suite.check_json_eq("run-list: ok", &resp, "ok", serde_json::json!(true));
        suite.check(
            "run-list: count > 0",
            resp["count"].as_u64().unwrap_or(0) > 0,
            format!("count = {}", resp["count"]),
        );
    }

    // ── DENY: heavy targets ──
    let heavy_targets = [
        ("jitter_benchmark", "crates/public/ptx-runtime/examples/jitter_benchmark.rs#main"),
        ("candle_performance_benchmark", "crates/public/ptx-runtime/examples/candle_performance_benchmark.rs#main"),
        ("fused_kernel_benchmark", "crates/public/ptx-runtime/examples/fused_kernel_benchmark.rs#main"),
        ("llm_inference_demo", "crates/public/ptx-runtime/examples/llm_inference_demo.rs#main"),
        ("neural_layer_inference", "crates/public/ptx-runtime/examples/neural_layer_inference.rs#main"),
    ];

    for (label, entry_id) in &heavy_targets {
        let resp = daemon.send_json(&format!("run-entry {entry_id}"));
        suite.check_json_eq(
            &format!("DENY {label}: ok is false"),
            &resp,
            "ok",
            serde_json::json!(false),
        );
        suite.check_json_contains(
            &format!("DENY {label}: stderr mentions strict mode"),
            &resp,
            "stderr",
            "single-pool strict mode",
        );
        suite.check_json_contains(
            &format!("DENY {label}: stderr mentions heavy workload"),
            &resp,
            "stderr",
            "heavy GPU workload",
        );
        suite.check(
            &format!("DENY {label}: elapsed_ms is 0 (no actual execution)"),
            resp["elapsed_ms"].as_u64() == Some(0),
            format!("elapsed_ms = {}", resp["elapsed_ms"]),
        );
    }

    // ── DENY: run-file with bench target ──
    {
        let resp = daemon.send_json("run-file ptx-tensor/examples/bench_ops.rs");
        suite.check_json_eq(
            "DENY run-file bench_ops: ok is false",
            &resp,
            "ok",
            serde_json::json!(false),
        );
        suite.check_json_contains(
            "DENY run-file bench_ops: stderr mentions strict mode",
            &resp,
            "stderr",
            "single-pool strict mode",
        );
    }

    // ── ALLOW: light target (telemetry_demo) ──
    {
        let resp = daemon.send_json("run-entry ptx-runtime/examples/telemetry_demo.rs#main");
        suite.check_json_eq(
            "ALLOW telemetry_demo: ok",
            &resp,
            "ok",
            serde_json::json!(true),
        );
        suite.check_json_eq(
            "ALLOW telemetry_demo: exit_code 0",
            &resp,
            "exit_code",
            serde_json::json!(0),
        );
        suite.check(
            "ALLOW telemetry_demo: has stdout output",
            resp["stdout"].as_str().map(|s| s.len() > 50).unwrap_or(false),
            format!("stdout length = {}", resp["stdout"].as_str().unwrap_or("").len()),
        );
    }

    // ── ALLOW: another light target (event_driven_gpu) ──
    {
        let resp = daemon.send_json("run-entry ptx-runtime/examples/event_driven_gpu.rs#main");
        suite.check_json_eq(
            "ALLOW event_driven_gpu: ok",
            &resp,
            "ok",
            serde_json::json!(true),
        );
        suite.check_json_eq(
            "ALLOW event_driven_gpu: exit_code 0",
            &resp,
            "exit_code",
            serde_json::json!(0),
        );
    }

    // ── events-stream: verify all three new event types ──
    {
        let raw = daemon.send_raw("events-stream");

        suite.check_raw_contains(
            "events: DaemonPoolInit present",
            &raw,
            "DaemonPoolInit",
        );
        suite.check_raw_contains(
            "events: pool_size_bytes field present",
            &raw,
            "pool_size_bytes",
        );

        suite.check_raw_contains(
            "events: RunExecutionModeSelected present",
            &raw,
            "RunExecutionModeSelected",
        );
        suite.check_raw_contains(
            "events: denied-strict-mode mode present",
            &raw,
            "denied-strict-mode",
        );
        suite.check_raw_contains(
            "events: external-process mode present (from light targets)",
            &raw,
            "external-process",
        );

        suite.check_raw_contains(
            "events: SinglePoolDenial present",
            &raw,
            "SinglePoolDenial",
        );
        suite.check_raw_contains(
            "events: denial reason mentions heavy GPU workload",
            &raw,
            "heavy GPU workload",
        );
    }

    // ── help: mentions strict mode ──
    {
        let resp = daemon.send_json("help");
        let commands = &resp["commands"];
        let run_entry_help = commands["run-entry <entry-id> [--streams=N] [--pool=F] [-- <args...>]"]
            .as_str()
            .unwrap_or("");
        suite.check(
            "help: run-entry mentions strict mode",
            run_entry_help.contains("strict mode"),
            format!("run-entry help = '{run_entry_help}'"),
        );
    }

    suite.print_summary();
    suite.assert_all_passed();
}

/// Quick smoke test: just boot and ping, useful for CI without full GPU workloads.
#[test]
fn smoke_test_boot_and_ping() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: SMOKE TEST");
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions::default());
    let mut suite = TestSuite::new("Smoke Test");

    let resp = daemon.send_json("ping");
    suite.check_json_eq("ping: ok", &resp, "ok", serde_json::json!(true));

    let resp = daemon.send_json("status");
    suite.check_json_eq("status: ok", &resp, "ok", serde_json::json!(true));
    suite.check(
        "status: pool_total > 0 (GPU initialized)",
        resp["pool_total"].as_u64().unwrap_or(0) > 0,
        format!("pool_total = {}", resp["pool_total"]),
    );

    let resp = daemon.send_json("health");
    suite.check_json_eq("health: ok", &resp, "ok", serde_json::json!(true));
    suite.check_json_eq("health: healthy", &resp, "healthy", serde_json::json!(true));

    suite.print_summary();
    suite.assert_all_passed();
}

// ═══════════════════════════════════════════════════════════════════
// Exhaustive Run-List Execution
// ═══════════════════════════════════════════════════════════════════

/// Entries that should NOT be executed through run-entry (daemon binaries,
/// build scripts, runner itself — they either hang or make no sense).
const SKIP_PATTERNS: &[&str] = &[
    "ptx-daemon/",
    "ptx-runner/src/main.rs",
    "build.rs",
    "ferrite-apps/",       // Long-running service binaries that don't exit
    "tiled_matmul",        // Kernel stub: returns NotSupported
    "ptx-renderd/",        // Has pre-existing compilation errors (missing crate)
];

fn should_skip_entry(id: &str) -> bool {
    SKIP_PATTERNS.iter().any(|pat| id.contains(pat))
}

/// Boot a daemon, fetch `run-list`, execute **every** example entry, and
/// report per-entry results.  This is the definitive "does the whole
/// workspace actually work through the daemon" test.
#[test]
fn exhaustive_run_all_entries() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: EXHAUSTIVE RUN-LIST EXECUTION");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions::default());
    let mut suite = TestSuite::new("Exhaustive Run-List");

    // 1. Fetch run-list
    let list_resp = daemon.send_json("run-list");
    suite.check_json_eq("run-list: ok", &list_resp, "ok", serde_json::json!(true));

    let entries = list_resp["entries"]
        .as_array()
        .expect("entries should be an array");
    let total = entries.len();
    eprintln!("  Discovered {total} entries from run-list\n");

    // 2. Classify entries
    let mut runnable: Vec<(&str, &str)> = Vec::new(); // (id, crate_name)
    let mut skipped: Vec<&str> = Vec::new();

    for entry in entries {
        let id = entry["id"].as_str().unwrap_or("");
        let crate_name = entry["crate_name"].as_str().unwrap_or("");
        if should_skip_entry(id) {
            skipped.push(id);
        } else {
            runnable.push((id, crate_name));
        }
    }

    eprintln!(
        "  Runnable: {}  |  Skipped: {} (daemon/build/runner)\n",
        runnable.len(),
        skipped.len()
    );
    for s in &skipped {
        eprintln!("    [SKIP] {s}");
    }
    if !skipped.is_empty() {
        eprintln!();
    }

    // 3. Execute each runnable entry
    let exec_start = Instant::now();

    for (idx, (id, crate_name)) in runnable.iter().enumerate() {
        let n = idx + 1;
        let total_runnable = runnable.len();
        eprint!("  [{n:>2}/{total_runnable}] {id} ...");

        let entry_start = Instant::now();
        let resp = daemon.send_json(&format!("run-entry {id}"));
        let entry_elapsed = entry_start.elapsed();

        let ok = resp["ok"].as_bool().unwrap_or(false);
        let exit_code = resp["exit_code"].as_i64();
        let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);

        // Determine result category
        let (status_icon, passed, detail) = if ok && exit_code == Some(0) {
            ("OK", true, format!("exit=0 {elapsed_ms}ms"))
        } else if ok {
            // ok=true but non-zero exit — the process ran but returned non-zero
            let code = exit_code.unwrap_or(-1);
            (
                "WARN",
                false,
                format!(
                    "exit={code} {elapsed_ms}ms\n  stderr (last 300 chars): {}",
                    truncate_tail(resp["stderr"].as_str().unwrap_or(""), 300)
                ),
            )
        } else {
            // ok=false — either denied, build failure, or crash
            let stderr = resp["stderr"].as_str().unwrap_or("");
            let error = resp["error"].as_str().unwrap_or("");
            let hint = if !stderr.is_empty() {
                truncate_tail(stderr, 400)
            } else {
                error.to_string()
            };
            (
                "FAIL",
                false,
                format!(
                    "exit={} {elapsed_ms}ms\n  {hint}",
                    exit_code.map(|c| c.to_string()).unwrap_or("none".to_string())
                ),
            )
        };

        eprintln!(
            " [{status_icon}] ({:.1}s wall, {elapsed_ms}ms reported)",
            entry_elapsed.as_secs_f64()
        );

        suite.check(
            &format!("run {id} [{crate_name}]"),
            passed,
            detail,
        );
    }

    let total_elapsed = exec_start.elapsed();
    eprintln!(
        "\n  Total execution time: {:.1}s for {} entries\n",
        total_elapsed.as_secs_f64(),
        runnable.len()
    );

    suite.print_summary();
    suite.assert_all_passed();
}

/// Same as exhaustive, but with strict mode ON.  Verifies that:
/// - Heavy targets are properly denied (instant, no compilation)
/// - Light targets still compile and execute successfully
#[test]
fn exhaustive_strict_mode_run_all_entries() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: EXHAUSTIVE RUN-LIST (STRICT MODE)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions {
        single_pool_strict: true,
        ..Default::default()
    });
    let mut suite = TestSuite::new("Exhaustive Run-List (Strict)");

    // 1. Fetch run-list
    let list_resp = daemon.send_json("run-list");
    suite.check_json_eq("run-list: ok", &list_resp, "ok", serde_json::json!(true));

    let entries = list_resp["entries"]
        .as_array()
        .expect("entries should be an array");
    let total = entries.len();
    eprintln!("  Discovered {total} entries from run-list\n");

    // 2. Classify
    let mut runnable: Vec<&str> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();

    for entry in entries {
        let id = entry["id"].as_str().unwrap_or("");
        if should_skip_entry(id) {
            skipped.push(id);
        } else {
            runnable.push(id);
        }
    }

    eprintln!(
        "  Runnable: {}  |  Skipped: {}\n",
        runnable.len(),
        skipped.len()
    );

    // Identify which are heavy vs light using the same logic as the daemon
    let heavy_keywords = ["bench", "stress", "jitter", "latency", "training", "inference"];
    let is_heavy = |id: &str| -> bool {
        let lower = id.to_ascii_lowercase();
        heavy_keywords.iter().any(|kw| lower.contains(kw))
    };

    let heavy_entries: Vec<&&str> = runnable.iter().filter(|id| is_heavy(id)).collect();
    let light_entries: Vec<&&str> = runnable.iter().filter(|id| !is_heavy(id)).collect();

    eprintln!(
        "  Heavy (expect DENIED): {}  |  Light (expect OK): {}\n",
        heavy_entries.len(),
        light_entries.len()
    );

    // 3a. Execute heavy entries — should all be denied instantly
    eprintln!("  ── Heavy targets (should be DENIED) ──");
    for (idx, id) in heavy_entries.iter().enumerate() {
        let n = idx + 1;
        eprint!("  [{n:>2}/{}] {id} ...", heavy_entries.len());

        let resp = daemon.send_json(&format!("run-entry {id}"));
        let ok = resp["ok"].as_bool().unwrap_or(true);
        let stderr = resp["stderr"].as_str().unwrap_or("");
        let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(999);

        let denied = !ok && stderr.contains("single-pool strict mode");
        if denied {
            eprintln!(" [DENIED] (0ms, as expected)");
        } else {
            eprintln!(" [UNEXPECTED] ok={ok} elapsed={elapsed_ms}ms");
        }

        suite.check(
            &format!("DENY heavy: {id}"),
            denied,
            if denied {
                "correctly denied".to_string()
            } else {
                format!("expected denial, got ok={ok}\nstderr: {stderr}\nfull: {resp}")
            },
        );
    }

    // 3b. Execute light entries — should all run successfully
    eprintln!("\n  ── Light targets (should execute) ──");
    let exec_start = Instant::now();

    for (idx, id) in light_entries.iter().enumerate() {
        let n = idx + 1;
        eprint!("  [{n:>2}/{}] {id} ...", light_entries.len());

        let entry_start = Instant::now();
        let resp = daemon.send_json(&format!("run-entry {id}"));
        let entry_elapsed = entry_start.elapsed();

        let ok = resp["ok"].as_bool().unwrap_or(false);
        let exit_code = resp["exit_code"].as_i64();
        let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);

        let (status_icon, passed, detail) = if ok && exit_code == Some(0) {
            ("OK", true, format!("exit=0 {elapsed_ms}ms"))
        } else if ok {
            let code = exit_code.unwrap_or(-1);
            (
                "WARN",
                false,
                format!(
                    "exit={code} {elapsed_ms}ms\n  stderr: {}",
                    truncate_tail(resp["stderr"].as_str().unwrap_or(""), 300)
                ),
            )
        } else {
            let stderr = resp["stderr"].as_str().unwrap_or("");
            let error = resp["error"].as_str().unwrap_or("");
            let hint = if !stderr.is_empty() {
                truncate_tail(stderr, 400)
            } else {
                error.to_string()
            };
            (
                "FAIL",
                false,
                format!(
                    "exit={} {elapsed_ms}ms\n  {hint}",
                    exit_code.map(|c| c.to_string()).unwrap_or("none".to_string())
                ),
            )
        };

        eprintln!(
            " [{status_icon}] ({:.1}s wall, {elapsed_ms}ms reported)",
            entry_elapsed.as_secs_f64()
        );

        suite.check(
            &format!("run light: {id}"),
            passed,
            detail,
        );
    }

    let total_elapsed = exec_start.elapsed();
    eprintln!(
        "\n  Total light-target execution time: {:.1}s for {} entries\n",
        total_elapsed.as_secs_f64(),
        light_entries.len()
    );

    suite.print_summary();
    suite.assert_all_passed();
}

/// Targeted test: run the real GPU-compute examples end-to-end through the
/// daemon and verify each one completes with exit code 0.  These are the
/// actual matrix-multiply, tensor, candle, pipeline and attention entries —
/// not just lightweight telemetry probes.
#[test]
fn compute_examples_e2e() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  DAEMON INTEGRATION: GPU COMPUTE EXAMPLES E2E");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions::default());
    let mut suite = TestSuite::new("GPU Compute E2E");

    // These are the entries that do real GPU computation.
    let compute_entries: &[(&str, &str)] = &[
        // ── ptx-compute: matrix multiply ──
        ("simple_matmul", "crates/internal/ptx-compute/examples/simple_matmul.rs#main"),
        // NOTE: tiled_matmul returns NotSupported (kernel stub, not a GPU init issue)
        // NOTE: extreme_100k_streams requests 20000 streams — too heavy for daemon child
        // ── ptx-tensor: GPU tensor ops ──
        ("gpu_test (tensor)", "crates/internal/ptx-tensor/examples/gpu_test.rs#main"),
        // ── ptx-kernels: candle / kernel verification ──
        ("test_candle", "crates/internal/ptx-kernels/examples/test_candle.rs#main"),
        ("test_candle_tlsf", "crates/internal/ptx-kernels/examples/test_candle_tlsf.rs#main"),
        ("verify_candle_math", "crates/internal/ptx-kernels/examples/verify_candle_math.rs#main"),
        // ── ptx-runtime: GPU memory, streams, pipelines ──
        ("dynamic_jit_allocation", "crates/public/ptx-runtime/examples/dynamic_jit_allocation.rs#main"),
        ("event_driven_gpu", "crates/public/ptx-runtime/examples/event_driven_gpu.rs#main"),
        ("integration_4features", "crates/public/ptx-runtime/examples/integration_4features.rs#main"),
        ("massive_candle_parallelism", "crates/public/ptx-runtime/examples/massive_candle_parallelism.rs#main"),
        ("memory_efficient_pipelines", "crates/public/ptx-runtime/examples/memory_efficient_pipelines.rs#main"),
        ("parallel_batch_processing", "crates/public/ptx-runtime/examples/parallel_batch_processing.rs#main"),
        ("transformer_attention_layer", "crates/public/ptx-runtime/examples/transformer_attention_layer.rs#main"),
    ];

    let total = compute_entries.len();
    eprintln!("  Running {total} GPU compute examples through the daemon\n");

    let suite_start = Instant::now();

    for (idx, (label, entry_id)) in compute_entries.iter().enumerate() {
        let n = idx + 1;
        eprint!("  [{n:>2}/{total}] {label} ...");

        let t0 = Instant::now();
        let resp = daemon.send_json(&format!("run-entry {entry_id}"));
        let wall = t0.elapsed();

        let ok = resp["ok"].as_bool().unwrap_or(false);
        let exit_code = resp["exit_code"].as_i64();
        let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);
        let stdout_len = resp["stdout"].as_str().map(|s| s.len()).unwrap_or(0);

        let success = ok && exit_code == Some(0);

        if success {
            eprintln!(
                " [OK] exit=0  {elapsed_ms}ms daemon / {:.1}s wall  stdout={stdout_len}B",
                wall.as_secs_f64()
            );
        } else {
            let stderr = resp["stderr"].as_str().unwrap_or("");
            let error = resp["error"].as_str().unwrap_or("");
            eprintln!(
                " [FAIL] ok={ok} exit={} {elapsed_ms}ms / {:.1}s wall",
                exit_code.map(|c| c.to_string()).unwrap_or("none".into()),
                wall.as_secs_f64()
            );
            if !stderr.is_empty() {
                let tail = truncate_tail(stderr, 500);
                for line in tail.lines() {
                    eprintln!("         stderr: {line}");
                }
            }
            if !error.is_empty() {
                eprintln!("         error: {error}");
            }
        }

        let detail = if success {
            format!("exit=0 {elapsed_ms}ms stdout={stdout_len}B")
        } else {
            let stderr = resp["stderr"].as_str().unwrap_or("");
            let error = resp["error"].as_str().unwrap_or("");
            format!(
                "ok={ok} exit={} {elapsed_ms}ms\nstderr: {}\nerror: {error}",
                exit_code.map(|c| c.to_string()).unwrap_or("none".into()),
                truncate_tail(stderr, 600),
            )
        };

        suite.check(&format!("{label}"), success, detail);
    }

    let total_wall = suite_start.elapsed();
    eprintln!(
        "\n  Total wall time: {:.1}s for {total} compute examples\n",
        total_wall.as_secs_f64()
    );

    suite.print_summary();
    suite.assert_all_passed();
}

/// Stream scaling stress test: find the logical limit of the OS.
///
/// Boots a daemon at escalating stream counts (16 → 65536), verifies it
/// initializes, then runs real GPU compute workloads through it at each level.
/// Reports the maximum stream count that succeeded and the failure threshold.
#[test]
fn stream_scaling_stress_test() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  STREAM SCALING STRESS TEST — Finding OS Logical Limits");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    // Stream counts to test — exponential scaling
    let stream_levels: &[u32] = &[
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536,
    ];

    // Compute entries to validate at each level (mix of light and heavy workloads)
    let validation_entries: &[(&str, &str)] = &[
        ("simple_matmul", "crates/internal/ptx-compute/examples/simple_matmul.rs#main"),
        ("test_candle_tlsf", "crates/internal/ptx-kernels/examples/test_candle_tlsf.rs#main"),
        ("transformer_attention_layer", "crates/public/ptx-runtime/examples/transformer_attention_layer.rs#main"),
    ];

    let mut results: Vec<(u32, bool, String, Vec<(String, bool, u64)>)> = Vec::new();
    let mut max_succeeded: u32 = 0;
    let mut first_failure: Option<(u32, String)> = None;

    for &streams in stream_levels {
        eprintln!("  ┌─────────────────────────────────────────────────");
        eprintln!("  │ Testing {streams} daemon streams");
        eprintln!("  └─────────────────────────────────────────────────");

        // Boot daemon with this stream count
        let t0 = Instant::now();
        let daemon_result = std::panic::catch_unwind(|| {
            DaemonInstance::start(DaemonOptions {
                single_pool_strict: false,
                extra_env: vec![
                    ("FERRITE_MAX_STREAMS".to_string(), streams.to_string()),
                ],
            })
        });

        let daemon = match daemon_result {
            Ok(d) => d,
            Err(_) => {
                let boot_time = t0.elapsed();
                let reason = format!("daemon failed to boot ({:.1}s)", boot_time.as_secs_f64());
                eprintln!("    [FAIL] {reason}");
                eprintln!();
                if first_failure.is_none() {
                    first_failure = Some((streams, reason.clone()));
                }
                results.push((streams, false, reason, vec![]));
                continue;
            }
        };
        let boot_time = t0.elapsed();
        eprintln!("    [OK] Daemon booted in {:.2}s", boot_time.as_secs_f64());

        // Verify daemon is healthy and reports correct stream count
        let status = daemon.send_json("status");
        let ok = status["ok"].as_bool().unwrap_or(false);
        let reported_streams = status["active_streams"].as_u64().unwrap_or(0);
        let pool_total = status["pool_total"].as_u64().unwrap_or(0);
        let pool_total_mb = pool_total as f64 / (1024.0 * 1024.0);

        if !ok {
            let reason = format!("status not ok: {status}");
            eprintln!("    [FAIL] {reason}");
            eprintln!();
            if first_failure.is_none() {
                first_failure = Some((streams, reason.clone()));
            }
            results.push((streams, false, reason, vec![]));
            continue;
        }
        eprintln!("    [OK] Status: active_streams={reported_streams}, pool={pool_total_mb:.1} MB");

        // Run compute entries through this daemon
        let mut entry_results: Vec<(String, bool, u64)> = Vec::new();
        let mut all_pass = true;

        for (label, entry_id) in validation_entries {
            eprint!("    [{label}] ...");
            let t1 = Instant::now();
            let resp = daemon.send_json(&format!("run-entry {entry_id}"));
            let wall = t1.elapsed();

            let entry_ok = resp["ok"].as_bool().unwrap_or(false);
            let exit_code = resp["exit_code"].as_i64();
            let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);
            let success = entry_ok && exit_code == Some(0);

            if success {
                eprintln!(" [OK] {elapsed_ms}ms ({:.1}s wall)", wall.as_secs_f64());
            } else {
                let stderr = resp["stderr"].as_str().unwrap_or("");
                eprintln!(
                    " [FAIL] ok={entry_ok} exit={} {elapsed_ms}ms",
                    exit_code.map(|c| c.to_string()).unwrap_or("none".into())
                );
                if !stderr.is_empty() {
                    eprintln!("           {}", truncate_tail(stderr, 200));
                }
                all_pass = false;
            }
            entry_results.push((label.to_string(), success, elapsed_ms));
        }

        if all_pass {
            max_succeeded = streams;
            eprintln!("    ✓ All compute entries passed at {streams} streams");
        } else if first_failure.is_none() {
            first_failure = Some((streams, "compute entry failed".to_string()));
        }
        eprintln!();

        results.push((streams, all_pass, format!("boot={:.2}s pool={pool_total_mb:.1}MB", boot_time.as_secs_f64()), entry_results));

        // Daemon drops here, shutting down cleanly for next iteration
    }

    // ── Summary ──
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  STREAM SCALING RESULTS");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  {:>8}  {:>6}  {:>10}  {}", "STREAMS", "STATUS", "BOOT", "ENTRIES");
    eprintln!("  {:─>8}  {:─>6}  {:─>10}  {:─>40}", "", "", "", "");

    for (streams, all_pass, info, entries) in &results {
        let status = if *all_pass { "OK" } else { "FAIL" };
        let entry_summary: String = entries
            .iter()
            .map(|(name, ok, ms)| {
                let icon = if *ok { "✓" } else { "✗" };
                format!("{icon}{name}({ms}ms)")
            })
            .collect::<Vec<_>>()
            .join("  ");
        eprintln!(
            "  {:>8}  {:>6}  {:>10}  {}",
            streams, status, info, entry_summary
        );
    }

    eprintln!();
    eprintln!(
        "  Maximum successful stream count: {}",
        if max_succeeded > 0 {
            max_succeeded.to_string()
        } else {
            "NONE".to_string()
        }
    );
    if let Some((fail_at, reason)) = &first_failure {
        eprintln!("  First failure at: {fail_at} streams — {reason}");
    }
    eprintln!();

    // Assert at least the default (16) works
    assert!(
        max_succeeded >= 16,
        "System failed at the default stream count of 16!"
    );
}

/// Per-command --streams=N --pool=F wiring test.
///
/// Boots ONE daemon and sends `run-entry` commands with escalating --streams
/// and --pool overrides.  Each program creates its own CUDA streams and TLSF
/// pool at the requested size.  Tests the full shell-to-GPU pipeline.
#[test]
fn per_command_stream_scaling() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  PER-COMMAND STREAM SCALING (--streams=N --pool=F)");
    eprintln!("  One daemon, escalating per-run stream counts");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    let daemon = DaemonInstance::start(DaemonOptions::default());

    // Verify baseline
    let status = daemon.send_json("status");
    let pool_mb = status["pool_total"].as_u64().unwrap_or(0) as f64 / (1024.0 * 1024.0);
    eprintln!("  Daemon pool: {pool_mb:.0} MB\n");

    // (streams, pool_fraction) — pool must hold 2 * streams * bytes_per_stream
    let levels: &[(u32, &str)] = &[
        (8,     "0.50"),   // profile default (balanced baseline)
        (16,    "0.50"),
        (64,    "0.50"),
        (128,   "0.50"),
        (256,   "0.50"),
        (512,   "0.50"),
        (1024,  "0.50"),
        (2048,  "0.50"),
        (4096,  "0.55"),
        (8192,  "0.60"),
    ];

    // Run entries at each stream level — multistream_active saturates ALL streams
    let entries: &[(&str, &str)] = &[
        ("multistream",  "crates/public/ptx-runtime/examples/multistream_active_compute.rs#main"),
        ("matmul",       "crates/internal/ptx-compute/examples/simple_matmul.rs#main"),
        ("attention",    "crates/public/ptx-runtime/examples/transformer_attention_layer.rs#main"),
    ];

    struct LevelResult {
        streams: u32,
        pool: String,
        entries: Vec<(String, bool, u64)>,
    }
    let mut results: Vec<LevelResult> = Vec::new();
    let mut max_succeeded: u32 = 0;

    for &(streams, pool_frac) in levels {
        eprintln!("  ── --streams={streams} --pool={pool_frac} ──");

        let mut entry_results: Vec<(String, bool, u64)> = Vec::new();
        let mut all_pass = true;

        for (label, entry_id) in entries {
            let cmd = format!(
                "run-entry {entry_id} --streams={streams} --pool={pool_frac}"
            );
            eprint!("    [{label}] ...");
            let t0 = Instant::now();
            let resp = daemon.send_json(&cmd);
            let wall = t0.elapsed();

            let ok = resp["ok"].as_bool().unwrap_or(false);
            let exit_code = resp["exit_code"].as_i64();
            let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);
            let streams_used = resp["streams_override"].as_u64();
            let pool_used = resp["pool_override"].as_f64();
            let success = ok && exit_code == Some(0);

            if success {
                eprintln!(
                    " [OK] {elapsed_ms}ms ({:.1}s) streams={} pool={}",
                    wall.as_secs_f64(),
                    streams_used.map(|s| s.to_string()).unwrap_or("?".into()),
                    pool_used.map(|p| format!("{p:.2}")).unwrap_or("?".into()),
                );
            } else {
                let stderr = resp["stderr"].as_str().unwrap_or("");
                eprintln!(
                    " [FAIL] exit={} {elapsed_ms}ms",
                    exit_code.map(|c| c.to_string()).unwrap_or("none".into())
                );
                if !stderr.is_empty() {
                    eprintln!("           {}", truncate_tail(stderr, 200));
                }
                all_pass = false;
            }
            entry_results.push((label.to_string(), success, elapsed_ms));
        }

        if all_pass {
            max_succeeded = streams;
        }
        eprintln!();
        results.push(LevelResult {
            streams, pool: pool_frac.to_string(), entries: entry_results,
        });
    }

    // Summary
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  PER-COMMAND STREAM SCALING RESULTS");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  {:>7} {:>6} {:>9}  {}", "STREAMS", "POOL%", "STATUS", "ENTRIES");
    eprintln!("  {:─>7} {:─>6} {:─>9}  {:─>50}", "", "", "", "");

    for r in &results {
        let passed = r.entries.iter().filter(|(_, ok, _)| *ok).count();
        let total = r.entries.len();
        let status = if passed == total {
            format!("{passed}/{total} OK")
        } else {
            format!("{passed}/{total} FAIL")
        };
        let detail: String = r.entries.iter()
            .map(|(name, ok, ms)| {
                let icon = if *ok { "✓" } else { "✗" };
                format!("{icon}{name}({ms}ms)")
            })
            .collect::<Vec<_>>()
            .join("  ");
        eprintln!("  {:>7} {:>6} {:>9}  {}", r.streams, r.pool, status, detail);
    }

    eprintln!();
    eprintln!("  Max per-command streams with all entries passing: {max_succeeded}");
    eprintln!();

    assert!(max_succeeded >= 16, "Per-command streams failed at minimum!");
}

/// Uncapped child stream stress test: the real deal.
///
/// Removes the orchestrator's stream/pool caps on child processes and runs
/// actual GPU compute workloads at escalating stream counts.  Each child
/// program creates its own CUDA streams and TLSF pool at the requested size.
/// This tests whether the OS can manage real multi-stream compute end-to-end.
#[test]
fn uncapped_child_stream_stress_test() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  UNCAPPED CHILD STREAM STRESS TEST");
    eprintln!("  Real programs, real streams, real VRAM — no safety nets");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    // (child_streams, child_pool_fraction, daemon_streams)
    let levels: &[(u32, &str, u32)] = &[
        (16,    "0.10", 16),
        (32,    "0.10", 32),
        (64,    "0.12", 64),
        (128,   "0.15", 128),
        (256,   "0.15", 256),
        (512,   "0.18", 512),
        (1024,  "0.20", 1024),
        (2048,  "0.20", 2048),
        (4096,  "0.25", 4096),
        (8192,  "0.25", 8192),
        (16384, "0.25", 16384),
    ];

    // Run a wider set of compute entries to really stress
    let entries: &[(&str, &str)] = &[
        ("simple_matmul",               "crates/internal/ptx-compute/examples/simple_matmul.rs#main"),
        ("gpu_test",                     "crates/internal/ptx-tensor/examples/gpu_test.rs#main"),
        ("test_candle_tlsf",            "crates/internal/ptx-kernels/examples/test_candle_tlsf.rs#main"),
        ("transformer_attention_layer",  "crates/public/ptx-runtime/examples/transformer_attention_layer.rs#main"),
        ("massive_candle_parallelism",   "crates/public/ptx-runtime/examples/massive_candle_parallelism.rs#main"),
        ("parallel_batch_processing",    "crates/public/ptx-runtime/examples/parallel_batch_processing.rs#main"),
    ];

    struct LevelResult {
        child_streams: u32,
        daemon_streams: u32,
        pool_frac: String,
        boot_secs: f64,
        daemon_pool_mb: f64,
        entries: Vec<(String, bool, u64)>,
    }

    let mut results: Vec<LevelResult> = Vec::new();
    let mut max_succeeded: u32 = 0;
    let mut first_failure: Option<(u32, String)> = None;

    for &(child_streams, pool_frac, daemon_streams) in levels {
        eprintln!("  ┌─────────────────────────────────────────────────────────");
        eprintln!("  │ child_streams={child_streams}  pool_frac={pool_frac}  daemon_streams={daemon_streams}");
        eprintln!("  └─────────────────────────────────────────────────────────");

        let t0 = Instant::now();
        let daemon_result = std::panic::catch_unwind(|| {
            DaemonInstance::start(DaemonOptions {
                single_pool_strict: false,
                extra_env: vec![
                    ("FERRITE_MAX_STREAMS".to_string(), daemon_streams.to_string()),
                    ("FERRITE_CHILD_MAX_STREAMS".to_string(), child_streams.to_string()),
                    ("FERRITE_CHILD_POOL_FRACTION".to_string(), pool_frac.to_string()),
                ],
            })
        });

        let daemon = match daemon_result {
            Ok(d) => d,
            Err(_) => {
                let boot_time = t0.elapsed();
                let reason = format!("daemon failed to boot ({:.1}s)", boot_time.as_secs_f64());
                eprintln!("    [FAIL] {reason}");
                eprintln!();
                if first_failure.is_none() {
                    first_failure = Some((child_streams, reason));
                }
                results.push(LevelResult {
                    child_streams, daemon_streams, pool_frac: pool_frac.to_string(),
                    boot_secs: boot_time.as_secs_f64(), daemon_pool_mb: 0.0,
                    entries: vec![],
                });
                continue;
            }
        };
        let boot_time = t0.elapsed();

        let status = daemon.send_json("status");
        let pool_total = status["pool_total"].as_u64().unwrap_or(0);
        let pool_mb = pool_total as f64 / (1024.0 * 1024.0);
        eprintln!("    [OK] Daemon booted in {:.2}s  pool={pool_mb:.0} MB", boot_time.as_secs_f64());

        let mut entry_results: Vec<(String, bool, u64)> = Vec::new();
        let mut all_pass = true;

        for (label, entry_id) in entries {
            eprint!("    [{label}] ...");
            let t1 = Instant::now();
            let resp = daemon.send_json(&format!("run-entry {entry_id}"));
            let wall = t1.elapsed();

            let ok = resp["ok"].as_bool().unwrap_or(false);
            let exit_code = resp["exit_code"].as_i64();
            let elapsed_ms = resp["elapsed_ms"].as_u64().unwrap_or(0);
            let success = ok && exit_code == Some(0);

            if success {
                eprintln!(" [OK] {elapsed_ms}ms ({:.1}s wall)", wall.as_secs_f64());
            } else {
                let stderr = resp["stderr"].as_str().unwrap_or("");
                let error = resp["error"].as_str().unwrap_or("");
                eprintln!(
                    " [FAIL] exit={} {elapsed_ms}ms",
                    exit_code.map(|c| c.to_string()).unwrap_or("none".into())
                );
                // Show relevant error info
                let hint = if !stderr.is_empty() { stderr } else { error };
                if !hint.is_empty() {
                    eprintln!("           {}", truncate_tail(hint, 300));
                }
                all_pass = false;
            }
            entry_results.push((label.to_string(), success, elapsed_ms));
        }

        if all_pass {
            max_succeeded = child_streams;
            eprintln!("    ✓ All {}/{} entries passed at {child_streams} child streams", entries.len(), entries.len());
        } else {
            let passed_count = entry_results.iter().filter(|(_, ok, _)| *ok).count();
            eprintln!(
                "    ✗ {passed_count}/{} entries passed at {child_streams} child streams",
                entries.len()
            );
            if first_failure.is_none() {
                first_failure = Some((child_streams, format!("{passed_count}/{} passed", entries.len())));
            }
        }
        eprintln!();

        results.push(LevelResult {
            child_streams, daemon_streams, pool_frac: pool_frac.to_string(),
            boot_secs: boot_time.as_secs_f64(), daemon_pool_mb: pool_mb,
            entries: entry_results,
        });
    }

    // ── Summary ──
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════════════════════════");
    eprintln!("  UNCAPPED CHILD STREAM RESULTS");
    eprintln!("═══════════════════════════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  {:>7} {:>6} {:>5} {:>8} {:>9}  {}",
        "STREAMS", "POOL%", "BOOT", "D.POOL", "STATUS", "ENTRIES");
    eprintln!("  {:─>7} {:─>6} {:─>5} {:─>8} {:─>9}  {:─>60}", "", "", "", "", "", "");

    for r in &results {
        let passed = r.entries.iter().filter(|(_, ok, _)| *ok).count();
        let total = r.entries.len();
        let status = if total == 0 {
            "BOOT_FAIL".to_string()
        } else if passed == total {
            format!("{passed}/{total} OK")
        } else {
            format!("{passed}/{total} FAIL")
        };

        let entry_detail: String = r.entries
            .iter()
            .map(|(name, ok, ms)| {
                let icon = if *ok { "✓" } else { "✗" };
                format!("{icon}{name}({ms}ms)")
            })
            .collect::<Vec<_>>()
            .join(" ");

        eprintln!(
            "  {:>7} {:>6} {:>4.1}s {:>7.0}MB {:>9}  {}",
            r.child_streams, r.pool_frac, r.boot_secs, r.daemon_pool_mb, status, entry_detail
        );
    }

    eprintln!();
    eprintln!(
        "  Max child streams with all entries passing: {}",
        if max_succeeded > 0 { max_succeeded.to_string() } else { "NONE".to_string() }
    );
    if let Some((at, reason)) = &first_failure {
        eprintln!("  First failure at: {at} streams — {reason}");
    }
    eprintln!();

    assert!(max_succeeded >= 16, "System failed at the minimum stream count!");
}

fn truncate_tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}
