use std::io::{self, Write as _};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use super::state::{EditorMode, TuiState, UiMode};
use super::style::ThemeVariant;
use super::{commands, files, layout, style};
use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;

// Stdio stealing and restoration are handled by ferrite_platform::tty.

pub fn run_tui(
    daemon: Arc<DaemonState>,
    runner: Arc<parking_lot::Mutex<ScriptRunner>>,
    rx: Receiver<DaemonEvent>,
    tx: Sender<DaemonEvent>,
) -> io::Result<()> {
    let stdio_guard = ferrite_platform::tty::steal_stdio()?;
    let tty = stdio_guard
        .tty_file()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "failed to capture TTY"))?;

    enable_raw_mode()?;
    let mut tty_writer = io::BufWriter::new(tty);
    execute!(
        tty_writer,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )?;
    tty_writer.flush()?;

    let render_file = stdio_guard.dup_tty_fd()?;
    let backend = CrosstermBackend::new(render_file);
    let mut terminal = Terminal::new(backend)?;

    // Initialise theme before any drawing or state creation.
    let theme = daemon
        .config
        .theme
        .as_deref()
        .map(ThemeVariant::from_str_loose)
        .unwrap_or(ThemeVariant::Default);
    style::init(theme);

    let device_name = daemon
        .config
        .gpu_name
        .clone()
        .unwrap_or_else(|| format!("GPU {}", daemon.config.device_id));

    let mut tui_state = TuiState::new(
        device_name,
        daemon.runtime.num_streams(),
        std::env::current_dir().unwrap_or_else(|_| ".".into()),
    );

    tui_state.push_log(LogEntry::new(LogCategory::Sys, "daemon started — clean view"));
    tui_state.push_log(LogEntry::new(
        LogCategory::Sys,
        "help for commands | /sysmon on for dashboard | ctrl+o for files",
    ));

    // Tick thread (~16 FPS)
    let tx_tick = tx.clone();
    std::thread::spawn(move || {
        while tx_tick.send(DaemonEvent::Tick).is_ok() {
            std::thread::sleep(Duration::from_millis(60));
        }
    });

    // Crossterm input thread
    let tx_key = tx.clone();
    std::thread::spawn(move || loop {
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(ev) = event::read() {
                let sent = match ev {
                    Event::Key(key) => tx_key.send(DaemonEvent::Key(key)),
                    Event::Mouse(mouse) => tx_key.send(DaemonEvent::Mouse(mouse)),
                    Event::Paste(text) => tx_key.send(DaemonEvent::Paste(text)),
                    _ => Ok(()),
                };
                if sent.is_err() {
                    break;
                }
            }
        }
    });

    // Main event loop
    let mut needs_draw = true;
    loop {
        while let Ok(evt) = rx.try_recv() {
            match evt {
                DaemonEvent::Tick => {
                    {
                        let r = runner.lock();
                        tui_state.refresh(&daemon, &r);
                    }
                    tui_state.tick();
                    needs_draw = true;
                }
                DaemonEvent::Key(key) => {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

                    // Ctrl+4 — cycle color theme (global, works in any mode)
                    if ctrl && matches!(key.code, KeyCode::Char('4')) {
                        let name = style::cycle_theme();
                        tui_state.push_log(LogEntry::new(
                            LogCategory::Sys,
                            format!("theme: {}", name),
                        ));
                        needs_draw = true;
                        continue;
                    }

                    if matches!(tui_state.ui_mode, UiMode::Files) {
                        if files::handle_files_key(&mut tui_state, key.code, ctrl, shift, &runner, &tx, &daemon) {
                            continue;
                        }
                    }

                    // ── scheduler-mode key handling ──────────────────
                    if matches!(tui_state.ui_mode, UiMode::Scheduler) {
                        use super::state::SchedulerViewMode;
                        match key.code {
                            KeyCode::Char('o') if ctrl => {
                                tui_state.toggle_ui_mode();
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Tab => {
                                // Cycle sub-panels: Queue -> Tenants -> Policy -> Queue
                                tui_state.scheduler_view_mode = match tui_state.scheduler_view_mode {
                                    SchedulerViewMode::Queue => SchedulerViewMode::Tenants,
                                    SchedulerViewMode::Tenants => SchedulerViewMode::Policy,
                                    SchedulerViewMode::Policy => SchedulerViewMode::Queue,
                                };
                                tui_state.scheduler_selected_index = 0;
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Up => {
                                tui_state.scheduler_selected_index =
                                    tui_state.scheduler_selected_index.saturating_sub(1);
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Down => {
                                let max_idx = match tui_state.scheduler_view_mode {
                                    SchedulerViewMode::Queue => {
                                        tui_state.scheduler_queue_snapshot.len().saturating_sub(1)
                                    }
                                    SchedulerViewMode::Tenants => {
                                        tui_state.scheduler_tenant_snapshot.len().saturating_sub(1)
                                    }
                                    SchedulerViewMode::Policy => {
                                        tui_state.scheduler_policy_decisions.len().saturating_sub(1)
                                    }
                                };
                                if tui_state.scheduler_selected_index < max_idx {
                                    tui_state.scheduler_selected_index += 1;
                                }
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('p') if !ctrl => {
                                // Toggle pause/resume
                                use std::sync::atomic::Ordering;
                                let new_paused = !tui_state.scheduler_paused;
                                daemon.scheduler_paused.store(new_paused, Ordering::Relaxed);
                                tui_state.scheduler_paused = new_paused;
                                tui_state.push_log(LogEntry::new(
                                    LogCategory::Sys,
                                    format!(
                                        "scheduler queue {}",
                                        if new_paused { "PAUSED" } else { "RESUMED" }
                                    ),
                                ));
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('k') if !ctrl => {
                                // Kill selected job (only in Queue view)
                                if matches!(tui_state.scheduler_view_mode, SchedulerViewMode::Queue) {
                                    if let Some(job) = tui_state
                                        .scheduler_queue_snapshot
                                        .get(tui_state.scheduler_selected_index)
                                    {
                                        tui_state.push_log(LogEntry::new(
                                            LogCategory::Sys,
                                            format!("[scheduler] kill job {}", job.job_id),
                                        ));
                                    }
                                }
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('r') if !ctrl => {
                                // Reprioritize selected job (only in Queue view)
                                if matches!(tui_state.scheduler_view_mode, SchedulerViewMode::Queue) {
                                    if let Some(job) = tui_state
                                        .scheduler_queue_snapshot
                                        .get(tui_state.scheduler_selected_index)
                                    {
                                        tui_state.push_log(LogEntry::new(
                                            LogCategory::Sys,
                                            format!(
                                                "[scheduler] reprioritize job {} (current priority: {})",
                                                job.job_id, job.priority
                                            ),
                                        ));
                                    }
                                }
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('c') if ctrl => {
                                daemon.shutdown();
                                continue;
                            }
                            KeyCode::Char('d') if ctrl => {
                                daemon.shutdown();
                                continue;
                            }
                            KeyCode::Esc => {
                                // Return to shell mode
                                tui_state.ui_mode = UiMode::Shell;
                                tui_state.push_log(LogEntry::new(LogCategory::Sys, "shell mode"));
                                needs_draw = true;
                                continue;
                            }
                            _ => {
                                needs_draw = true;
                                continue;
                            }
                        }
                    }

                    // ── run-output mode key handling ─────────────
                    if matches!(tui_state.ui_mode, UiMode::RunOutput) {
                        match key.code {
                            KeyCode::Esc => {
                                tui_state.exit_run_output();
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('o') if ctrl => {
                                tui_state.exit_run_output();
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('6') if ctrl => {
                                commands::run::cmd_stop_run(&mut tui_state);
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Char('c') if ctrl => {
                                daemon.shutdown();
                                continue;
                            }
                            KeyCode::Char('d') if ctrl => {
                                daemon.shutdown();
                                continue;
                            }
                            KeyCode::Up => {
                                tui_state.scroll_run_output(-1);
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::Down => {
                                tui_state.scroll_run_output(1);
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::PageUp => {
                                tui_state.scroll_run_output(-20);
                                needs_draw = true;
                                continue;
                            }
                            KeyCode::PageDown => {
                                tui_state.scroll_run_output(20);
                                needs_draw = true;
                                continue;
                            }
                            _ => {
                                needs_draw = true;
                                continue;
                            }
                        }
                    }

                    match key.code {
                        KeyCode::Char('o') if ctrl => {
                            tui_state.toggle_ui_mode();
                            continue;
                        }
                        // Ctrl+C: in Files mode handled by editor; in Scheduler mode
                        // handled above; in Shell mode quits
                        KeyCode::Char('c') if ctrl => {
                            if matches!(tui_state.ui_mode, UiMode::Files | UiMode::Scheduler | UiMode::RunOutput) {
                                // Already handled by handle_files_key / scheduler / run-output block above
                                continue;
                            }
                            daemon.shutdown();
                            break;
                        }
                        // Ctrl+D: in Files mode handled by editor (half-page); in Scheduler
                        // mode handled above; in Shell mode quits
                        KeyCode::Char('d') if ctrl => {
                            if matches!(tui_state.ui_mode, UiMode::Files | UiMode::Scheduler | UiMode::RunOutput) {
                                continue;
                            }
                            daemon.shutdown();
                            break;
                        }

                        // Ctrl+5: run-file on open file (works in any mode)
                        KeyCode::Char('5') if ctrl => {
                            commands::run::cmd_run_file(&mut tui_state, &daemon, &tx, &[]);
                        }
                        // Ctrl+6: stop run
                        KeyCode::Char('6') if ctrl => {
                            commands::run::cmd_stop_run(&mut tui_state);
                        }

                        // Submit command
                        KeyCode::Enter => {
                            if let Some(line) = tui_state.input_submit() {
                                // Echo the command so the user sees what was
                                // executed alongside the results in scrollback.
                                tui_state.push_log(LogEntry::new(
                                    LogCategory::Run,
                                    format!("> {}", line),
                                ));
                                commands::exec_command(&line, &daemon, &runner, &mut tui_state, &tx);
                            }
                        }

                        // Line editing
                        KeyCode::Backspace => tui_state.input_backspace(),
                        KeyCode::Delete => tui_state.input_delete(),
                        KeyCode::Left => tui_state.input_left(),
                        KeyCode::Right => tui_state.input_right(),
                        KeyCode::Home | KeyCode::Char('a') if ctrl => tui_state.input_home(),
                        KeyCode::End | KeyCode::Char('e') if ctrl => tui_state.input_end(),
                        KeyCode::Char('k') if ctrl => tui_state.input_kill_line(),
                        KeyCode::Char('w') if ctrl => tui_state.input_kill_word(),
                        KeyCode::Char('u') if ctrl => tui_state.input_clear(),
                        KeyCode::Esc => tui_state.input_clear(),

                        // History
                        KeyCode::Up => {
                            if tui_state.input_is_empty() {
                                if matches!(tui_state.ui_mode, UiMode::Shell)
                                    && tui_state.sysmon_enabled
                                    && tui_state.sysmon_section_max_scroll > 0
                                {
                                    tui_state.scroll_sysmon_sections(-1);
                                } else {
                                    tui_state.scroll_log(-1);
                                }
                            } else {
                                tui_state.history_up();
                            }
                        }
                        KeyCode::Down => {
                            if tui_state.input_is_empty() {
                                if matches!(tui_state.ui_mode, UiMode::Shell)
                                    && tui_state.sysmon_enabled
                                    && tui_state.sysmon_section_max_scroll > 0
                                {
                                    tui_state.scroll_sysmon_sections(1);
                                } else {
                                    tui_state.scroll_log(1);
                                }
                            } else {
                                tui_state.history_down();
                            }
                        }
                        KeyCode::PageUp => {
                            if matches!(tui_state.ui_mode, UiMode::Shell)
                                && tui_state.sysmon_enabled
                                && tui_state.sysmon_section_max_scroll > 0
                            {
                                tui_state.scroll_sysmon_sections(-3);
                            } else {
                                tui_state.scroll_log(-10);
                            }
                        }
                        KeyCode::PageDown => {
                            if matches!(tui_state.ui_mode, UiMode::Shell)
                                && tui_state.sysmon_enabled
                                && tui_state.sysmon_section_max_scroll > 0
                            {
                                tui_state.scroll_sysmon_sections(3);
                            } else {
                                tui_state.scroll_log(10);
                            }
                        }

                        // Tab — focus toggle only when input empty
                        KeyCode::Tab => {
                            if tui_state.input_is_empty() {
                                tui_state.toggle_focus();
                            }
                        }

                        // Regular character input
                        KeyCode::Char(ch) => {
                            tui_state.input_insert(ch);
                        }

                        _ => {}
                    }
                    needs_draw = true;
                }
                DaemonEvent::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            if matches!(tui_state.ui_mode, UiMode::RunOutput) {
                                tui_state.scroll_run_output(-3);
                            } else if matches!(tui_state.ui_mode, UiMode::Files) {
                                tui_state.file_scroll_lines(3);
                            } else if matches!(tui_state.ui_mode, UiMode::Scheduler) {
                                tui_state.scheduler_selected_index =
                                    tui_state.scheduler_selected_index.saturating_sub(3);
                            } else if matches!(tui_state.ui_mode, UiMode::Shell)
                                && tui_state.sysmon_enabled
                                && tui_state.sysmon_section_max_scroll > 0
                            {
                                tui_state.scroll_sysmon_sections(-1);
                            } else {
                                tui_state.scroll_log(-3);
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            if matches!(tui_state.ui_mode, UiMode::RunOutput) {
                                tui_state.scroll_run_output(3);
                            } else if matches!(tui_state.ui_mode, UiMode::Files) {
                                tui_state.file_scroll_lines(-3);
                            } else if matches!(tui_state.ui_mode, UiMode::Scheduler) {
                                use super::state::SchedulerViewMode;
                                let max_idx = match tui_state.scheduler_view_mode {
                                    SchedulerViewMode::Queue => {
                                        tui_state.scheduler_queue_snapshot.len().saturating_sub(1)
                                    }
                                    SchedulerViewMode::Tenants => {
                                        tui_state.scheduler_tenant_snapshot.len().saturating_sub(1)
                                    }
                                    SchedulerViewMode::Policy => {
                                        tui_state.scheduler_policy_decisions.len().saturating_sub(1)
                                    }
                                };
                                tui_state.scheduler_selected_index =
                                    (tui_state.scheduler_selected_index + 3).min(max_idx);
                            } else if matches!(tui_state.ui_mode, UiMode::Shell)
                                && tui_state.sysmon_enabled
                                && tui_state.sysmon_section_max_scroll > 0
                            {
                                tui_state.scroll_sysmon_sections(1);
                            } else {
                                tui_state.scroll_log(3);
                            }
                        }
                        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
                            if matches!(tui_state.ui_mode, UiMode::Files) {
                                files::handle_files_mouse_down(&mut tui_state, mouse.column, mouse.row);
                            } else if matches!(tui_state.ui_mode, UiMode::Shell) {
                                handle_log_click(&mut tui_state, mouse.column, mouse.row);
                            }
                        }
                        MouseEventKind::Drag(crossterm::event::MouseButton::Left) => {
                            if matches!(tui_state.ui_mode, UiMode::Files) {
                                files::handle_files_mouse_drag(&mut tui_state, mouse.column, mouse.row);
                            }
                        }
                        MouseEventKind::Up(crossterm::event::MouseButton::Left) => {
                            if matches!(tui_state.ui_mode, UiMode::Files) {
                                tui_state.finish_selection();
                            }
                        }
                        _ => {}
                    }
                    needs_draw = true;
                }
                DaemonEvent::Paste(text) => {
                    if matches!(tui_state.ui_mode, UiMode::Files)
                        && matches!(tui_state.editor_mode, EditorMode::Insert)
                    {
                        tui_state.push_undo();
                        tui_state.file_paste(&text);
                    } else if matches!(tui_state.ui_mode, UiMode::Shell) {
                        for ch in text.chars() {
                            if ch != '\n' && ch != '\r' {
                                tui_state.input_insert(ch);
                            }
                        }
                    }
                    // In Scheduler mode, paste is a no-op.
                    needs_draw = true;
                }

                // Channel events — all trigger a redraw
                DaemonEvent::Log(entry) => {
                    tui_state.push_log(entry);
                    needs_draw = true;
                }
                DaemonEvent::LogAction(entry) => {
                    tui_state.push_log(entry);
                    needs_draw = true;
                }
                DaemonEvent::ClientHandled { command, success } => {
                    let (cat, marker) = if success {
                        (LogCategory::Sys, "ok")
                    } else {
                        (LogCategory::Err, "FAIL")
                    };
                    tui_state.push_log(LogEntry::new(
                        cat,
                        format!("client {} — {}", command, marker),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::AppEvent { app_name, message } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::App,
                        format!("{}: {}", app_name, message),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::TensorResult { shape, data } => {
                    tui_state.set_tensor_viz(shape, data);
                    commands::push_plot3d_tensor(&mut tui_state);
                    needs_draw = true;
                }
                DaemonEvent::PipelineResult {
                    name,
                    compile_ms,
                    exec_ms,
                } => {
                    tui_state.set_pipeline(name.clone(), compile_ms, exec_ms);
                    // Also record a kernel profile for pipeline executions
                    use super::profiling::KernelProfile;
                    tui_state.kernel_profiles.push(KernelProfile {
                        timestamp: std::time::Instant::now(),
                        compile_ms: compile_ms as u64,
                        exec_ms: exec_ms as u64,
                        total_ms: (compile_ms + exec_ms) as u64,
                        input_shapes: Vec::new(),
                        output_shape: Vec::new(),
                        node_count: 0,
                        total_elements: 0,
                        success: true,
                        source_tag: name,
                        vram_before: 0,
                        vram_after: 0,
                    });
                    needs_draw = true;
                }
                DaemonEvent::Diagnostic(diag) => {
                    let cat = if diag.status == "FAIL" {
                        LogCategory::Err
                    } else {
                        LogCategory::Sys
                    };
                    // Summary on its own line so it doesn't get truncated
                    tui_state.push_log(LogEntry::new(
                        cat,
                        format!(
                            "[{}:{}] {}",
                            diag.component, diag.code, diag.summary,
                        ),
                    ));
                    // Remediation on a separate line — always visible even
                    // in compact mode where long lines get clipped.
                    if !diag.remediation.is_empty() {
                        tui_state.push_log(LogEntry::new(
                            LogCategory::Sys,
                            format!("  fix: {}", diag.remediation),
                        ));
                    }
                    needs_draw = true;
                }
                DaemonEvent::Shutdown => {
                    daemon.shutdown();
                    break;
                }

                // ── run events (Plan B) ─────────────────────
                DaemonEvent::RunStarted { file: _ } => {
                    tui_state.run_status = super::state::run_state::RunStatus::Running;
                    needs_draw = true;
                }
                DaemonEvent::RunOutput { line, is_error } => {
                    use super::state::run_state::RunOutputLine;
                    let severity = if is_error {
                        LogCategory::Err
                    } else {
                        LogCategory::Run
                    };
                    tui_state.run_output.push_back(RunOutputLine::new(severity, &line));
                    // Cap at 500 lines
                    while tui_state.run_output.len() > 500 {
                        tui_state.run_output.pop_front();
                    }
                    // Also log to main log
                    tui_state.push_log(LogEntry::new(severity, format!("[run] {}", line)));
                    // Update elapsed
                    if let Some(start) = tui_state.run_start_time {
                        tui_state.run_elapsed_ms = start.elapsed().as_millis() as u64;
                    }
                    needs_draw = true;
                }
                DaemonEvent::RunFinished { success, elapsed_ms } => {
                    tui_state.run_status = if success {
                        super::state::run_state::RunStatus::Succeeded
                    } else {
                        super::state::run_state::RunStatus::Failed
                    };
                    tui_state.run_elapsed_ms = elapsed_ms;
                    tui_state.push_log(LogEntry::new(
                        if success { LogCategory::Jit } else { LogCategory::Err },
                        format!(
                            "[run] {} in {}ms",
                            if success { "succeeded" } else { "failed" },
                            elapsed_ms
                        ),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::RunTimeout { elapsed_ms } => {
                    tui_state.run_status = super::state::run_state::RunStatus::Timeout;
                    tui_state.run_elapsed_ms = elapsed_ms;
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("[run] TIMEOUT after {}ms", elapsed_ms),
                    ));
                    needs_draw = true;
                }

                // ── profiling events ──────────────────────
                DaemonEvent::KernelProfiled {
                    compile_ms,
                    exec_ms,
                    input_shapes,
                    output_shape,
                    node_count,
                    total_elements,
                    success,
                    source_tag,
                    vram_before,
                    vram_after,
                } => {
                    use super::profiling::KernelProfile;
                    tui_state.kernel_profiles.push(KernelProfile {
                        timestamp: std::time::Instant::now(),
                        compile_ms,
                        exec_ms,
                        total_ms: compile_ms + exec_ms,
                        input_shapes,
                        output_shape,
                        node_count,
                        total_elements,
                        success,
                        source_tag,
                        vram_before,
                        vram_after,
                    });
                    needs_draw = true;
                }

                // ── agent events (Plan C) ───────────────────
                DaemonEvent::AgentCommand(cmd) => {
                    let _resp = super::agent::executor::execute_agent_command(
                        &cmd, &mut tui_state, &runner, &tx,
                    );
                    needs_draw = true;
                }
                DaemonEvent::AgentCommandSync(cmd, reply_tx) => {
                    let resp = super::agent::executor::execute_agent_command(
                        &cmd, &mut tui_state, &runner, &tx,
                    );
                    let _ = reply_tx.send(resp);
                    needs_draw = true;
                }

                // ── scheduler / control plane events ──────────
                DaemonEvent::SchedulerCommandExecuted { command, success } => {
                    let cat = if success { LogCategory::Sys } else { LogCategory::Err };
                    tui_state.push_log(LogEntry::new(
                        cat,
                        format!("[scheduler] {} — {}", command, if success { "ok" } else { "FAIL" }),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::PolicyDecisionMade { tenant_id, action, allowed, reason } => {
                    use super::state::PolicyDecisionSummary;
                    let summary = PolicyDecisionSummary {
                        elapsed_secs: daemon.start_time.elapsed().as_secs_f64(),
                        tenant_id,
                        action: action.clone(),
                        allowed,
                        reason: reason.clone(),
                    };
                    if tui_state.scheduler_policy_decisions.len() >= 200 {
                        tui_state.scheduler_policy_decisions.pop_front();
                    }
                    tui_state.scheduler_policy_decisions.push_back(summary);

                    let cat = if allowed { LogCategory::Sys } else { LogCategory::Err };
                    let marker = if allowed { "ALLOW" } else { "DENY" };
                    let reason_str = reason.as_deref().unwrap_or("");
                    tui_state.push_log(LogEntry::new(
                        cat,
                        format!("[policy] {} {} t:{} {}", marker, action, tenant_id, reason_str),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::SchedulerPauseChanged { paused } => {
                    tui_state.scheduler_paused = paused;
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("[scheduler] queue {}", if paused { "PAUSED" } else { "RESUMED" }),
                    ));
                    needs_draw = true;
                }

                // ── durable job events ────────────────────────
                DaemonEvent::JobSubmitted { job_id, name } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("[job] submitted job {} ({})", job_id, name),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::JobStarted { job_id, name, pid } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("[job] started job {} ({}) pid={}", job_id, name, pid),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::JobStopped { job_id, name, reason } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("[job] stopped job {} ({}): {}", job_id, name, reason),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::JobRestarted { job_id, name, attempt, delay_ms } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::App,
                        format!("[job] restarting job {} ({}) attempt={} delay={}ms", job_id, name, attempt, delay_ms),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::JobFailed { job_id, name, reason } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("[job] FAILED job {} ({}): {}", job_id, name, reason),
                    ));
                    needs_draw = true;
                }
                DaemonEvent::JobRecovered { job_id, name, state } => {
                    tui_state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("[job] recovered job {} ({}) state={}", job_id, name, state),
                    ));
                    needs_draw = true;
                }
            }
        }

        if !daemon.is_running() {
            break;
        }

        // Only redraw when state actually changed — avoids redundant
        // full-screen redraws that cause terminal flicker.
        if needs_draw {
            terminal.draw(|frame| layout::draw(frame, &mut tui_state))?;
            needs_draw = false;
        }

        std::thread::sleep(Duration::from_millis(5));
    }

    // Teardown
    if let Some(mut child) = tui_state.plot3d_child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    if tui_state.plot3d_ipc_dev_ptr != 0 {
        let _ = unsafe { ptx_sys::cudaFree(tui_state.plot3d_ipc_dev_ptr as *mut std::ffi::c_void) };
        tui_state.plot3d_ipc_dev_ptr = 0;
        tui_state.plot3d_ipc_bytes = 0;
        tui_state.plot3d_ipc_handle_hex = None;
    }
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    stdio_guard.restore();

    Ok(())
}

/// Handle a mouse click in the log area.  If the clicked row corresponds
/// to a log entry with an `action`, write that command to the input line.
fn handle_log_click(state: &mut TuiState, col: u16, row: u16) {
    use super::state::Panel;

    let area = state.log_body_area;
    if area.width == 0 || area.height == 0 {
        return;
    }
    // Hit-test: is the click inside the log body?
    if col < area.x || col >= area.x + area.width || row < area.y || row >= area.y + area.height {
        return;
    }

    let screen_row = (row - area.y) as usize;
    let visible = area.height as usize;

    // In sysmon compact mode, certain categories are filtered out.
    // Replicate the same filter the renderer applies so row indices match.
    let sysmon_compact = state.sysmon_enabled;
    let verbose = state.focus == Panel::Processes;

    let visible_entries: Vec<&crate::events::LogEntry> = state
        .log
        .iter()
        .rev()
        .skip(state.log_scroll)
        .filter(|entry| {
            if !sysmon_compact || verbose {
                true
            } else {
                !matches!(entry.category, LogCategory::Jit | LogCategory::Run)
            }
        })
        .take(visible)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    if let Some(entry) = visible_entries.get(screen_row) {
        if let Some(ref action) = entry.action {
            state.input = action.clone();
            state.cursor = state.input.len();
        }
    }
}
