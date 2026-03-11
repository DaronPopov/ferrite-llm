use std::path::PathBuf;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::Command;
use std::sync::Arc;
use std::ffi::c_void;

use crate::events::{LogCategory, LogEntry};
use crate::state::DaemonState;
use crate::tui::state::{TuiState, UiDensity, UiMode};
use crate::tui::workspace::{self, fs_ops};
use ptx_sys;

pub(super) fn cmd_metrics(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let s = daemon.runtime.stats();
    let tlsf = daemon.runtime.tlsf_stats();
    let uptime = daemon.start_time.elapsed().as_secs();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "up {}s  clients {}  ops {}  vram {}MB  streams {}",
            uptime,
            state.active_clients,
            s.total_ops,
            s.vram_used / (1024 * 1024),
            s.active_streams,
        ),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "sm {}%  mem {}%  pool {:.1}%  hw {}  flops {}",
            s.gpu_utilization as i32,
            s.mem_utilization as i32,
            tlsf.utilization_percent,
            if s.nvml_valid { "nvml" } else { "fallback" },
            if s.cupti_valid {
                format!("{:.2} GFLOPS", s.gflops_total)
            } else {
                "na".to_string()
            },
        ),
    ));
}

pub(super) fn cmd_sysmon(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        Some("on") => state.sysmon_enabled = true,
        Some("off") => state.sysmon_enabled = false,
        Some("status") => {}
        Some("toggle") | None => state.sysmon_enabled = !state.sysmon_enabled,
        Some(_) => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "usage: /sysmon [on|off|toggle|status]",
            ));
            return;
        }
    }

    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "sysmon {}",
            if state.sysmon_enabled {
                "enabled"
            } else {
                "disabled"
            }
        ),
    ));
}

pub(super) fn cmd_detail(state: &mut TuiState) {
    state.detail_mode = !state.detail_mode;
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "detail mode {}",
            if state.detail_mode { "on" } else { "off" }
        ),
    ));
}

pub(super) fn cmd_density(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        None | Some("status") => {}
        Some("auto") => state.ui_density = UiDensity::Auto,
        Some("compact") => state.ui_density = UiDensity::Compact,
        Some("balanced") => state.ui_density = UiDensity::Balanced,
        Some("comfortable") | Some("comfy") => state.ui_density = UiDensity::Comfortable,
        Some(_) => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "usage: /density [auto|compact|balanced|comfortable|status]",
            ));
            return;
        }
    }

    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("density {}", state.ui_density.label()),
    ));
}

pub(super) fn cmd_fxscript(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        None | Some("status") => {
            let label = state
                .fx_script_label
                .as_deref()
                .unwrap_or("none");
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("fxscript {}", label),
            ));
        }
        Some("reset") | Some("builtin") => match crate::tui::fxscript::parse_script(crate::tui::fxscript::DEFAULT_SCRIPT) {
            Ok(cfg) => {
                state.fx_script = Some(cfg);
                state.fx_script_label = Some("builtin:hq".to_string());
                state.push_log(LogEntry::new(LogCategory::Sys, "fxscript builtin:hq loaded"));
            }
            Err(e) => state.push_log(LogEntry::new(LogCategory::Err, format!("fxscript error: {}", e))),
        },
        Some("off") | Some("disable") => {
            state.fx_script = None;
            state.fx_script_label = None;
            state.push_log(LogEntry::new(LogCategory::Sys, "fxscript disabled"));
        }
        Some("load") => {
            let Some(path_arg) = args.get(1) else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: /fxscript [status|off|reset|load <path.rhai>]",
                ));
                return;
            };
            let path = PathBuf::from(path_arg);
            let full = if path.is_absolute() {
                path
            } else {
                state.workspace_root.join(path)
            };
            match std::fs::read_to_string(&full) {
                Ok(src) => match crate::tui::fxscript::parse_script(&src) {
                    Ok(cfg) => {
                        state.fx_script = Some(cfg);
                        state.fx_script_label = Some(full.display().to_string());
                        state.push_log(LogEntry::new(
                            LogCategory::Sys,
                            format!("fxscript loaded {}", full.display()),
                        ));
                    }
                    Err(e) => state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("fxscript parse error: {}", e),
                    )),
                },
                Err(e) => state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("fxscript read error: {}", e),
                )),
            }
        }
        Some(_) => state.push_log(LogEntry::new(
            LogCategory::Sys,
            "usage: /fxscript [status|off|reset|load <path.rhai>]",
        )),
    }
}

pub(super) fn cmd_plot3d(state: &mut TuiState, args: &[&str]) {
    let sub = args.first().copied().unwrap_or("status");
    match sub {
        "status" => {
            refresh_plot3d_process(state);
            let proc_state = if state.plot3d_child.is_some() {
                "running"
            } else {
                "stopped"
            };
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "plot3d {}  scene={}  socket={}",
                    proc_state,
                    state.plot3d_scene,
                    state.plot3d_socket.display()
                ),
            ));
        }
        "open" | "start" => {
            refresh_plot3d_process(state);
            if state.plot3d_child.is_some() {
                state.push_log(LogEntry::new(LogCategory::Sys, "plot3d already running"));
                return;
            }
            let bin = resolve_renderd_bin(state);
            if !bin.exists() {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!(
                        "renderd binary missing: {} (build with `cargo build -p ferrite-renderd`)",
                        bin.display()
                    ),
                ));
                return;
            }
            match Command::new(&bin)
                .arg("--socket")
                .arg(state.plot3d_socket.as_os_str())
                .spawn()
            {
                Ok(child) => {
                    state.plot3d_child = Some(child);
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("plot3d started via {}", bin.display()),
                    ));
                }
                Err(e) => state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("plot3d start error: {}", e),
                )),
            }
        }
        "close" | "stop" => {
            let _ = send_plot3d_cmd(state, serde_json::json!({ "cmd": "shutdown" }));
            if let Some(mut child) = state.plot3d_child.take() {
                if let Err(e) = child.kill() {
                    state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("plot3d kill error: {}", e),
                    ));
                }
                let _ = child.wait();
            }
            free_plot3d_ipc_tensor(state);
            state.push_log(LogEntry::new(LogCategory::Sys, "plot3d stopped"));
        }
        "scene" => {
            let Some(scene) = args.get(1).copied() else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: /plot3d scene <wave|surface|tensor>",
                ));
                return;
            };
            if let Err(e) = ensure_plot3d_running(state) {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("plot3d start error: {}", e),
                ));
                return;
            }
            state.plot3d_scene = scene.to_string();
            match send_plot3d_cmd(state, serde_json::json!({ "cmd": "scene", "scene": scene })) {
                Ok(_) => state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("plot3d scene {}", scene),
                )),
                Err(e) => state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("plot3d scene error: {}", e),
                )),
            }
        }
        "tensor" => {
            if let Err(e) = ensure_plot3d_running(state) {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("plot3d start error: {}", e),
                ));
                return;
            }
            if let Some(tv) = &state.last_tensor {
                let data = downsample_f32(&tv.samples, 5000);
                let samples_len = data.len();
                match send_plot3d_tensor(state, &data) {
                    Ok(_) => state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("plot3d tensor stream pushed ({} samples)", samples_len),
                    )),
                    Err(e) => state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("plot3d tensor error: {}", e),
                    )),
                }
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "plot3d tensor: no tensor snapshot available yet",
                ));
            }
        }
        _ => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "usage: /plot3d [status|open|close|scene <wave|surface|tensor>|tensor]",
            ));
        }
    }
}

fn refresh_plot3d_process(state: &mut TuiState) {
    if let Some(child) = state.plot3d_child.as_mut() {
        if let Ok(Some(_status)) = child.try_wait() {
            state.plot3d_child = None;
            free_plot3d_ipc_tensor(state);
        }
    }
}

fn resolve_renderd_bin(state: &TuiState) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.with_file_name("ferrite-renderd");
        if sibling.exists() {
            return sibling;
        }
    }
    let debug = state.workspace_root.join("target/debug/ferrite-renderd");
    if debug.exists() {
        return debug;
    }
    state.workspace_root.join("target/release/ferrite-renderd")
}

fn ensure_plot3d_running(state: &mut TuiState) -> Result<(), String> {
    refresh_plot3d_process(state);
    if state.plot3d_child.is_some() {
        return Ok(());
    }

    let bin = resolve_renderd_bin(state);
    if !bin.exists() {
        return Err(format!(
            "renderd binary missing: {} (build with `cargo build -p ferrite-renderd`)",
            bin.display()
        ));
    }

    let child = Command::new(&bin)
        .arg("--socket")
        .arg(state.plot3d_socket.as_os_str())
        .spawn()
        .map_err(|e| format!("spawn {}: {}", bin.display(), e))?;
    state.plot3d_child = Some(child);

    // Give renderd a short moment to bind the socket before first command.
    for _ in 0..25 {
        if state.plot3d_socket.exists() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Ok(())
}

fn send_plot3d_cmd(state: &mut TuiState, value: serde_json::Value) -> Result<String, String> {
    refresh_plot3d_process(state);
    let mut stream = UnixStream::connect(&state.plot3d_socket)
        .map_err(|e| format!("connect {}: {}", state.plot3d_socket.display(), e))?;
    let line = value.to_string();
    stream
        .write_all(line.as_bytes())
        .and_then(|_| stream.write_all(b"\n"))
        .map_err(|e| format!("write: {}", e))?;
    let mut reader = BufReader::new(stream);
    let mut reply = String::new();
    reader
        .read_line(&mut reply)
        .map_err(|e| format!("read: {}", e))?;
    Ok(reply.trim().to_string())
}

fn send_plot3d_tensor(state: &mut TuiState, data: &[f32]) -> Result<String, String> {
    if data.is_empty() {
        return Ok("ok:empty".to_string());
    }

    if prepare_ipc_tensor(state, data).is_ok() {
        if let Some(handle_hex) = state.plot3d_ipc_handle_hex.clone() {
            let reply = send_plot3d_cmd(
                state,
                serde_json::json!({
                    "cmd": "tensor_ipc",
                    "ipc_handle": handle_hex,
                    "len_f32": data.len(),
                }),
            )?;
            if !reply.starts_with("err:") {
                return Ok(reply);
            }
        }
    }

    let payload = serde_json::json!({
        "cmd": "tensor",
        "shape": [1usize, data.len().max(1)],
        "data": data,
    });
    send_plot3d_cmd(state, payload)
}

fn prepare_ipc_tensor(state: &mut TuiState, data: &[f32]) -> Result<(), String> {
    let bytes = data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "tensor byte size overflow".to_string())?;
    if bytes == 0 {
        return Ok(());
    }

    if state.plot3d_ipc_dev_ptr == 0 || state.plot3d_ipc_bytes < bytes {
        free_plot3d_ipc_tensor(state);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let code = unsafe { ptx_sys::cudaMalloc(&mut ptr as *mut _, bytes) };
        cuda_check(code, "cudaMalloc(plot3d ipc tensor)")?;
        state.plot3d_ipc_dev_ptr = ptr as usize;
        state.plot3d_ipc_bytes = bytes;

        let mut handle = ptx_sys::cudaIpcMemHandle_t::default();
        let hcode = unsafe { ptx_sys::cudaIpcGetMemHandle(&mut handle as *mut _, ptr) };
        cuda_check(hcode, "cudaIpcGetMemHandle(plot3d tensor)")?;
        state.plot3d_ipc_handle_hex = Some(ipc_handle_to_hex(&handle));
    }

    let ptr = state.plot3d_ipc_dev_ptr as *mut c_void;
    let copy = unsafe {
        ptx_sys::cudaMemcpy(
            ptr,
            data.as_ptr() as *const c_void,
            bytes,
            ptx_sys::cudaMemcpyHostToDevice,
        )
    };
    cuda_check(copy, "cudaMemcpy(H2D plot3d tensor)")?;
    Ok(())
}

fn free_plot3d_ipc_tensor(state: &mut TuiState) {
    if state.plot3d_ipc_dev_ptr != 0 {
        let _ = unsafe { ptx_sys::cudaFree(state.plot3d_ipc_dev_ptr as *mut c_void) };
    }
    state.plot3d_ipc_dev_ptr = 0;
    state.plot3d_ipc_bytes = 0;
    state.plot3d_ipc_handle_hex = None;
}

fn ipc_handle_to_hex(handle: &ptx_sys::cudaIpcMemHandle_t) -> String {
    let bytes =
        unsafe { std::slice::from_raw_parts(handle.reserved.as_ptr() as *const u8, 64) };
    let mut out = String::with_capacity(128);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn cuda_check(code: ptx_sys::cudaError_t, op: &str) -> Result<(), String> {
    if code == ptx_sys::cudaSuccess {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = ptx_sys::cudaGetErrorString(code);
        if ptr.is_null() {
            format!("{} failed with code {}", op, code)
        } else {
            let c = std::ffi::CStr::from_ptr(ptr);
            format!("{}: {} ({})", op, c.to_string_lossy(), code)
        }
    };
    Err(msg)
}

pub(crate) fn plot3d_push_latest_tensor_if_running(state: &mut TuiState) {
    refresh_plot3d_process(state);
    if state.plot3d_child.is_none() {
        return;
    }
    if state.plot3d_scene != "tensor" {
        return;
    }
    let now = std::time::Instant::now();
    if let Some(last) = state.plot3d_last_push {
        if now.duration_since(last) < std::time::Duration::from_millis(140) {
            return;
        }
    }
    if let Some(tv) = &state.last_tensor {
        let data = downsample_f32(&tv.samples, 3000);
        if send_plot3d_tensor(state, &data).is_ok() {
            state.plot3d_last_push = Some(now);
        }
    }
}

fn downsample_f32(src: &[f32], max: usize) -> Vec<f32> {
    if src.len() <= max {
        return src.to_vec();
    }
    let step = (src.len() / max).max(1);
    let mut out = Vec::with_capacity(max);
    let mut i = 0usize;
    while i < src.len() && out.len() < max {
        out.push(src[i]);
        i += step;
    }
    out
}

pub(super) fn cmd_files(state: &mut TuiState) {
    state.ui_mode = UiMode::Files;
    state.reload_file_entries();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "files mode: up/down select, enter open, h/l collapse/expand, tab focus, ctrl+s save, esc exit",
    ));
}

pub(super) fn cmd_open(state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: /open <path>"));
        return;
    }
    let path = std::path::PathBuf::from(args.join(" "));
    let full = if path.is_absolute() {
        path
    } else {
        state.workspace_root.join(path)
    };
    match state.open_file_path(full) {
        Ok(()) => {
            state.ui_mode = UiMode::Files;
            state.push_log(LogEntry::new(LogCategory::Sys, "file opened"));
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_save(state: &mut TuiState) {
    match state.file_save() {
        Ok(()) => state.push_log(LogEntry::new(LogCategory::Sys, "saved file")),
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

// ── filesystem commands (Plan A) ────────────────────────────────

pub(super) fn cmd_mkdir(state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: mkdir <path>"));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_mkdir(&root, &cwd, args[0]) {
        Ok(msg) => {
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
            state.file_tree.rebuild();
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_touch(state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: touch <path>"));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_touch(&root, &cwd, args[0]) {
        Ok(msg) => {
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
            state.file_tree.rebuild();
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_mv(state: &mut TuiState, args: &[&str]) {
    if args.len() < 2 {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: mv <src> <dst>"));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_mv(&root, &cwd, args[0], args[1]) {
        Ok(msg) => {
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
            state.file_tree.rebuild();
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_cp(state: &mut TuiState, args: &[&str]) {
    if args.len() < 2 {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: cp <src> <dst>"));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_cp(&root, &cwd, args[0], args[1]) {
        Ok(msg) => {
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
            state.file_tree.rebuild();
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_rm(state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            "usage: rm <path>  (requires confirmation)",
        ));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_rm(&root, &cwd, args[0], false) {
        Ok(fs_ops::RmResult::NeedsConfirm(msg, target)) => {
            state.pending_confirm =
                Some(workspace::PendingConfirm::new("rm", target));
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("{} — type 'y' to confirm or 'n' to cancel", msg),
            ));
        }
        Ok(fs_ops::RmResult::Removed(msg)) => {
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
            state.file_tree.rebuild();
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_ls(state: &mut TuiState, args: &[&str]) {
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    let target = args.first().copied();
    match fs_ops::ws_ls(&root, &cwd, target) {
        Ok(entries) => {
            for e in entries {
                state.push_log(LogEntry::new(LogCategory::Sys, format!("  {}", e)));
            }
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_cd(state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.current_dir = state.workspace_root.clone();
        let msg = fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
        state.push_log(LogEntry::new(LogCategory::Sys, msg));
        return;
    }
    let root = state.workspace_root.clone();
    let cwd = state.current_dir.clone();
    match fs_ops::ws_cd(&root, &cwd, args[0]) {
        Ok(new_cwd) => {
            state.current_dir = new_cwd;
            let msg = fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
        }
        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
    }
}

pub(super) fn cmd_pwd(state: &mut TuiState) {
    let msg = fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
    state.push_log(LogEntry::new(LogCategory::Sys, msg));
}

pub(super) fn cmd_confirm(state: &mut TuiState) {
    if let Some(pc) = state.pending_confirm.take() {
        if pc.is_expired() {
            state.push_log(LogEntry::new(
                LogCategory::Err,
                "confirmation expired (30s) — redo the command",
            ));
            return;
        }
        let root = state.workspace_root.clone();
        let cwd = state.current_dir.clone();
        let target_str = pc.target.display().to_string();
        match fs_ops::ws_rm(&root, &cwd, &target_str, true) {
            Ok(fs_ops::RmResult::Removed(msg)) => {
                state.push_log(LogEntry::new(LogCategory::Sys, msg));
                state.file_tree.rebuild();
            }
            Ok(_) => {}
            Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
        }
    } else {
        state.push_log(LogEntry::new(LogCategory::Sys, "nothing to confirm"));
    }
}

pub(super) fn cmd_cancel(state: &mut TuiState) {
    if state.pending_confirm.take().is_some() {
        state.push_log(LogEntry::new(LogCategory::Sys, "cancelled"));
    } else {
        state.push_log(LogEntry::new(LogCategory::Sys, "nothing to cancel"));
    }
}

pub(super) fn cmd_audit(state: &mut TuiState) {
    let count = state.audit_trail.len();
    if count == 0 {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            "no agent actions recorded",
        ));
    } else {
        state.push_log(LogEntry::new(LogCategory::Jit, "── agent audit trail ──"));
        let start = count.saturating_sub(20);
        let entries: Vec<_> = state
            .audit_trail
            .iter()
            .skip(start)
            .map(|e| {
                let status = if e.success { "ok" } else { "FAIL" };
                let cat = if e.success {
                    LogCategory::Sys
                } else {
                    LogCategory::Err
                };
                (
                    cat,
                    format!(
                        "  {} [{}] {} ({}us)",
                        e.command, status, e.message, e.duration_us
                    ),
                )
            })
            .collect();
        for (cat, msg) in entries {
            state.push_log(LogEntry::new(cat, msg));
        }
    }
}
