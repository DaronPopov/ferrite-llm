use std::sync::mpsc::Sender;
use std::sync::Arc;

use crossterm::event::KeyCode;

use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::tui::commands;
use crate::tui::editor::keymap;
use crate::tui::state::{EditorMode, TuiState};
use crate::tui::workspace::fs_ops;

pub(super) fn handle_editor_command(
    state: &mut TuiState,
    code: KeyCode,
    _ctrl: bool,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) -> bool {
    match code {
        KeyCode::Esc => {
            state.cmdline_clear();
            state.editor_mode = EditorMode::Normal;
        }
        KeyCode::Enter => {
            if let Some(cmd) = state.cmdline_submit() {
                exec_editor_command(&cmd, state, runner, tx);
            }
        }
        KeyCode::Backspace => {
            state.cmdline_backspace();
            if state.editor_cmdline.is_empty() {
                state.editor_mode = EditorMode::Normal;
            }
        }
        KeyCode::Char(ch) => {
            state.cmdline_insert(ch);
        }
        _ => return false,
    }
    true
}

fn exec_editor_command(
    cmd: &str,
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return;
    }
    match parts[0] {
        "w" | "write" => match state.file_save() {
            Ok(()) => state.push_log(LogEntry::new(LogCategory::Sys, "saved file")),
            Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
        },
        "q" | "quit" => {
            if state.file_dirty {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    "unsaved changes — use :q! to discard or :wq to save",
                ));
            } else {
                state.toggle_ui_mode();
            }
        }
        "q!" => {
            state.file_dirty = false;
            state.toggle_ui_mode();
        }
        "wq" | "x" => {
            match state.file_save() {
                Ok(()) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, "saved file"));
                    state.toggle_ui_mode();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "e" | "edit" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :e <path>"));
                return;
            }
            let path = std::path::PathBuf::from(parts[1..].join(" "));
            let full = if path.is_absolute() {
                path
            } else {
                state.workspace_root.join(path)
            };
            // Save current to buffer list before opening new file
            state.save_current_to_buffer();
            match state.open_file_path(full) {
                Ok(()) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, "file opened"));
                    state.save_current_to_buffer();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }

        // ── buffer management ──────────────────────────────────
        "bnext" | "bn" => {
            state.buffer_next();
            if let Some(path) = &state.open_file {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                state.push_log(LogEntry::new(LogCategory::Sys, format!("buffer: {}", name)));
            }
        }
        "bprev" | "bp" => {
            state.buffer_prev();
            if let Some(path) = &state.open_file {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                state.push_log(LogEntry::new(LogCategory::Sys, format!("buffer: {}", name)));
            }
        }
        "bclose" | "bd" => {
            match state.buffer_close() {
                Ok(()) => state.push_log(LogEntry::new(LogCategory::Sys, "buffer closed")),
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "bclose!" | "bd!" => {
            state.file_dirty = false;
            let _ = state.buffer_close();
            state.push_log(LogEntry::new(LogCategory::Sys, "buffer discarded"));
        }
        "buffers" | "ls!" | "bufs" => {
            if state.buffers.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no buffers open"));
            } else {
                let active_buf = state.active_buffer;
                let entries: Vec<String> = state
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(i, buf)| {
                        let active = if i == active_buf { ">" } else { " " };
                        let dirty = if buf.dirty { "[+]" } else { "   " };
                        let name = buf
                            .path
                            .as_ref()
                            .and_then(|p| p.file_name())
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "[untitled]".to_string());
                        format!("  {}{} {} {}", active, i + 1, dirty, name)
                    })
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── buffers ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
            }
        }

        // ── search/replace ─────────────────────────────────────
        "noh" | "nohlsearch" => {
            state.search.matches.clear();
            state.search.current_match = None;
            state.push_log(LogEntry::new(LogCategory::Sys, "search highlights cleared"));
        }

        // ── marks ──────────────────────────────────────────────
        "marks" => {
            let marks_list = state.marks.list();
            if marks_list.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no marks set"));
            } else {
                let entries: Vec<String> = marks_list
                    .iter()
                    .map(|(name, pos)| {
                        format!("  '{}  line {} col {}", name, pos.line + 1, pos.col)
                    })
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── marks ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
            }
        }
        "delmarks" | "delm" => {
            state.marks.clear();
            state.push_log(LogEntry::new(LogCategory::Sys, "all marks cleared"));
        }

        // ── registers ─────────────────────────────────────────
        "registers" | "reg" => {
            let regs = state.registers.list();
            if regs.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "all registers empty"));
            } else {
                let entries: Vec<String> = regs
                    .iter()
                    .map(|(name, content)| format!("  {}  {}", name, content))
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── registers ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
            }
        }

        // ── macros ────────────────────────────────────────────
        "macros" => {
            let macro_list = state.macro_engine.list();
            if macro_list.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no macros recorded"));
            } else {
                let entries: Vec<String> = macro_list
                    .iter()
                    .map(|(name, len)| format!("  @{}  {} keys", name, len))
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── macros ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
            }
        }

        // ── keymap ────────────────────────────────────────────
        "map" | "nmap" => {
            if parts.len() < 3 {
                // List bindings
                let bindings = state.keymap.list_bindings(EditorMode::Normal);
                let total = bindings.len();
                let mut entries: Vec<String> = bindings
                    .iter()
                    .take(30)
                    .map(|(key, action)| format!("  {:12} {:?}", key, action))
                    .collect();
                if total > 30 {
                    entries.push(format!("  ... and {} more", total - 30));
                }
                state.push_log(LogEntry::new(LogCategory::Jit, "── normal mode bindings ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
                return;
            }
            let key_str = parts[1];
            let action_str = parts[2..].join(" ");
            let seq = match keymap::parse_key_notation(key_str) {
                Some(s) => s,
                None => {
                    state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("invalid key notation: {}", key_str),
                    ));
                    return;
                }
            };
            let action = match parse_action_name(&action_str) {
                Some(a) => a,
                None => {
                    state.push_log(LogEntry::new(
                        LogCategory::Err,
                        format!("unknown action: {} — use :actions to list", action_str),
                    ));
                    return;
                }
            };
            state.keymap.map(EditorMode::Normal, seq, action);
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("mapped {} -> {}", key_str, action_str),
            ));
        }
        "imap" => {
            if parts.len() < 3 {
                let bindings = state.keymap.list_bindings(EditorMode::Insert);
                let entries: Vec<String> = bindings
                    .iter()
                    .take(20)
                    .map(|(key, action)| format!("  {:12} {:?}", key, action))
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── insert mode bindings ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
                return;
            }
            let key_str = parts[1];
            let action_str = parts[2..].join(" ");
            if let (Some(seq), Some(action)) =
                (keymap::parse_key_notation(key_str), parse_action_name(&action_str))
            {
                state.keymap.map(EditorMode::Insert, seq, action);
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("imap {} -> {}", key_str, action_str),
                ));
            }
        }
        "vmap" => {
            if parts.len() < 3 {
                let bindings = state.keymap.list_bindings(EditorMode::Visual);
                let entries: Vec<String> = bindings
                    .iter()
                    .take(20)
                    .map(|(key, action)| format!("  {:12} {:?}", key, action))
                    .collect();
                state.push_log(LogEntry::new(LogCategory::Jit, "── visual mode bindings ──"));
                for entry in entries {
                    state.push_log(LogEntry::new(LogCategory::Sys, entry));
                }
                return;
            }
            let key_str = parts[1];
            let action_str = parts[2..].join(" ");
            if let (Some(seq), Some(action)) =
                (keymap::parse_key_notation(key_str), parse_action_name(&action_str))
            {
                state.keymap.map(EditorMode::Visual, seq, action);
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("vmap {} -> {}", key_str, action_str),
                ));
            }
        }
        "unmap" | "nunmap" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :unmap <key>"));
                return;
            }
            if let Some(seq) = keymap::parse_key_notation(parts[1]) {
                state.keymap.unmap(EditorMode::Normal, &seq);
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("unmapped {}", parts[1]),
                ));
            }
        }
        "mapclear" => {
            state.keymap = keymap::KeyMap::new();
            state.push_log(LogEntry::new(LogCategory::Sys, "keymap reset to defaults"));
        }

        // ── list available actions (for :map) ──────────────────
        "actions" => {
            state.push_log(LogEntry::new(LogCategory::Jit, "── available actions ──"));
            let actions = [
                "MoveLeft", "MoveRight", "MoveUp", "MoveDown",
                "WordForward", "WordBackward", "WordEnd",
                "BigWordForward", "BigWordBackward", "BigWordEnd",
                "LineStart", "LineEnd", "FirstNonBlank",
                "GotoFirstLine", "GotoLastLine",
                "PageDown", "PageUp", "HalfPageDown", "HalfPageUp",
                "ScreenTop", "ScreenMiddle", "ScreenBottom",
                "ParagraphForward", "ParagraphBackward", "MatchingBracket",
                "FindCharForward", "FindCharBackward",
                "TillCharForward", "TillCharBackward",
                "InsertMode", "InsertAtLineStart",
                "AppendAfterCursor", "AppendAtLineEnd",
                "OpenLineBelow", "OpenLineAbove",
                "VisualMode", "VisualLineMode", "CommandMode",
                "DeleteChar", "DeleteLine", "YankLine",
                "PutAfter", "PutBefore", "JoinLines",
                "Undo", "Redo", "Indent", "Dedent",
                "SearchForward", "SearchBackward", "SearchNext", "SearchPrev",
                "SetMark", "GotoMark", "JumpBack", "JumpForward",
                "RecordMacro", "PlayMacro",
                "BufferNext", "BufferPrev", "BufferClose",
                "ToggleFocus", "ToggleUiMode", "Run", "Stop",
            ];
            for chunk in actions.chunks(4) {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("  {}", chunk.join(", ")),
                ));
            }
        }

        // ── grep (workspace search) ──────────────────────────
        "grep" | "vimgrep" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :grep <pattern>"));
                return;
            }
            let pattern = parts[1..].join(" ");
            state.push_log(LogEntry::new(
                LogCategory::Jit,
                format!("── grep: {} ──", pattern),
            ));

            // Search all files in workspace
            let root = state.workspace_root.clone();
            let mut results = Vec::new();
            search_files_recursive(&root, &pattern, &mut results, 50);

            if results.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no matches found"));
            } else {
                for (path, line_num, line_text) in &results {
                    let rel = path
                        .strip_prefix(&state.workspace_root)
                        .unwrap_or(path);
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!(
                            "  {}:{}: {}",
                            rel.display(),
                            line_num,
                            line_text.trim()
                        ),
                    ));
                }
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("{} matches", results.len()),
                ));
            }
        }

        // ── search/replace ───────────────────────────────────
        "s" | "substitute" => {
            // :s/pattern/replacement/ or :s/pattern/replacement/g
            let rest = &cmd[parts[0].len()..].trim_start();
            if let Some(result) = parse_substitute(rest) {
                let (pattern, replacement, global) = result;
                state.push_undo();
                let line_idx = state.file_cursor_line;
                let line = &mut state.file_lines[line_idx];
                let lower_pat = pattern.to_lowercase();
                let lower_line = line.to_lowercase();

                if global {
                    // Replace all on current line
                    let mut new_line = String::new();
                    let mut search_from = 0;
                    let mut count = 0;
                    while let Some(pos) = lower_line[search_from..].find(&lower_pat) {
                        let abs = search_from + pos;
                        new_line.push_str(&line[search_from..abs]);
                        new_line.push_str(&replacement);
                        search_from = abs + pattern.len();
                        count += 1;
                    }
                    new_line.push_str(&line[search_from..]);
                    *line = new_line;
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("{} replacement(s)", count),
                    ));
                } else {
                    // Replace first on current line
                    if let Some(pos) = lower_line.find(&lower_pat) {
                        let end = pos + pattern.len();
                        let mut new_line = String::new();
                        new_line.push_str(&line[..pos]);
                        new_line.push_str(&replacement);
                        new_line.push_str(&line[end..]);
                        *line = new_line;
                        state.push_log(LogEntry::new(LogCategory::Sys, "1 replacement"));
                    } else {
                        state.push_log(LogEntry::new(LogCategory::Sys, "pattern not found"));
                    }
                }
                state.file_dirty = true;
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: :s/pattern/replacement/[g]",
                ));
            }
        }
        "%s" => {
            // :%s/pattern/replacement/g — global file replace
            let rest = &cmd[2..].trim_start();
            if let Some((pattern, replacement, _)) = parse_substitute(rest) {
                state.push_undo();
                let lower_pat = pattern.to_lowercase();
                let mut total_count = 0;
                for line in &mut state.file_lines {
                    let lower_line = line.to_lowercase();
                    let mut new_line = String::new();
                    let mut search_from = 0;
                    while let Some(pos) = lower_line[search_from..].find(&lower_pat) {
                        let abs = search_from + pos;
                        new_line.push_str(&line[search_from..abs]);
                        new_line.push_str(&replacement);
                        search_from = abs + pattern.len();
                        total_count += 1;
                    }
                    new_line.push_str(&line[search_from..]);
                    *line = new_line;
                }
                state.file_dirty = true;
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("{} replacement(s) across file", total_count),
                ));
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: :%s/pattern/replacement/[g]",
                ));
            }
        }

        // ── editor help ──────────────────────────────────────
        "help" => {
            cmd_editor_help(state);
        }
        "keymap" => {
            state.push_log(LogEntry::new(LogCategory::Jit, "── keymap commands ──"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :map                   list normal mode bindings"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :map <key> <action>    bind key in normal mode"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :imap <key> <action>   bind key in insert mode"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :vmap <key> <action>   bind key in visual mode"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :unmap <key>           remove normal mode binding"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :mapclear              reset all to defaults"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  :actions               list all available actions"));
            state.push_log(LogEntry::new(LogCategory::Sys, ""));
            state.push_log(LogEntry::new(LogCategory::Sys, "  key notation: a-z, <C-x>, <CR>, <Esc>, <F1>-<F12>"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  example: :map <C-p> SearchForward"));
            state.push_log(LogEntry::new(LogCategory::Sys, "  example: :map gn SearchNext"));
        }

        // ── run commands (Plan B) ──────────────────────────
        "run" => commands::run::cmd_run(state, runner, tx),
        "stop" => commands::run::cmd_stop_run(state),
        "rerun" => commands::run::cmd_rerun(state, runner, tx),
        "args" => commands::run::cmd_args(state, &parts[1..]),
        "profile" => commands::run::cmd_profile(state, &parts[1..]),
        "timeout" => commands::run::cmd_timeout(state, &parts[1..]),

        // ── inspect / profiling ─────────────────────────
        "ptx" | "inspect" => commands::inspect::cmd_ptx_buffer(state, runner),
        "perf" | "profiling" => commands::inspect::cmd_perf(state),
        "jit-clear" => commands::inspect::cmd_jit_clear(state, runner),
        "jit-cache" => commands::inspect::cmd_jit_cache(state, runner),
        "quota" | "quotas" => commands::inspect::cmd_quota(state, &parts[1..]),

        // ── filesystem commands (Plan A) ───────────────────
        "mkdir" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :mkdir <path>"));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_mkdir(&root, &cwd, parts[1]) {
                Ok(msg) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                    state.file_tree.rebuild();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "touch" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :touch <path>"));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_touch(&root, &cwd, parts[1]) {
                Ok(msg) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                    state.file_tree.rebuild();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "rm" => {
            if parts.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :rm <path>  (requires confirmation)"));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_rm(&root, &cwd, parts[1], false) {
                Ok(fs_ops::RmResult::NeedsConfirm(msg, target)) => {
                    state.pending_confirm = Some(crate::tui::workspace::PendingConfirm::new("rm", target));
                    state.push_log(LogEntry::new(LogCategory::Sys, format!("{} — type :y to confirm or :n to cancel", msg)));
                }
                Ok(fs_ops::RmResult::Removed(msg)) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                    state.file_tree.rebuild();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "y" | "confirm" => {
            if let Some(pc) = state.pending_confirm.take() {
                if pc.is_expired() {
                    state.push_log(LogEntry::new(LogCategory::Err, "confirmation expired (30s) — redo the command"));
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
        "n" | "cancel" => {
            if state.pending_confirm.take().is_some() {
                state.push_log(LogEntry::new(LogCategory::Sys, "cancelled"));
            } else {
                state.push_log(LogEntry::new(LogCategory::Sys, "nothing to cancel"));
            }
        }
        "mv" => {
            if parts.len() < 3 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :mv <src> <dst>"));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_mv(&root, &cwd, parts[1], parts[2]) {
                Ok(msg) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                    state.file_tree.rebuild();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "cp" => {
            if parts.len() < 3 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: :cp <src> <dst>"));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_cp(&root, &cwd, parts[1], parts[2]) {
                Ok(msg) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                    state.file_tree.rebuild();
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "ls" => {
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            let target = if parts.len() > 1 { Some(parts[1]) } else { None };
            match fs_ops::ws_ls(&root, &cwd, target) {
                Ok(entries) => {
                    for e in entries {
                        state.push_log(LogEntry::new(LogCategory::Sys, format!("  {}", e)));
                    }
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "cd" => {
            if parts.len() < 2 {
                state.current_dir = state.workspace_root.clone();
                let msg = crate::tui::workspace::fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
                state.push_log(LogEntry::new(LogCategory::Sys, msg));
                return;
            }
            let root = state.workspace_root.clone();
            let cwd = state.current_dir.clone();
            match fs_ops::ws_cd(&root, &cwd, parts[1]) {
                Ok(new_cwd) => {
                    state.current_dir = new_cwd;
                    let msg = crate::tui::workspace::fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
                    state.push_log(LogEntry::new(LogCategory::Sys, msg));
                }
                Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
            }
        }
        "pwd" => {
            let msg = crate::tui::workspace::fs_ops::ws_pwd(&state.workspace_root, &state.current_dir);
            state.push_log(LogEntry::new(LogCategory::Sys, msg));
        }

        // ── audit (Plan C) ────────────────────────────────
        "audit" => {
            let count = state.audit_trail.len();
            if count == 0 {
                state.push_log(LogEntry::new(LogCategory::Sys, "no agent actions recorded"));
            } else {
                state.push_log(LogEntry::new(LogCategory::Jit, "── agent audit trail ──"));
                let start = count.saturating_sub(20);
                let entries: Vec<_> = state
                    .audit_trail
                    .iter()
                    .skip(start)
                    .map(|e| {
                        let status = if e.success { "ok" } else { "FAIL" };
                        let cat = if e.success { LogCategory::Sys } else { LogCategory::Err };
                        (cat, format!("  {} [{}] {} ({}us)", e.command, status, e.message, e.duration_us))
                    })
                    .collect();
                for (cat, msg) in entries {
                    state.push_log(LogEntry::new(cat, msg));
                }
            }
        }

        n if n.chars().all(|c| c.is_ascii_digit()) => {
            // :<number> — jump to line
            if let Ok(line_num) = n.parse::<usize>() {
                let target = line_num.saturating_sub(1).min(state.file_lines.len().saturating_sub(1));
                state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                state.file_cursor_line = target;
                state.clamp_cursor_normal();
                state.preferred_col = state.file_cursor_col;
            }
        }
        _ => {
            // Check if it's a substitute command starting with s/
            if cmd.starts_with("s/") || cmd.starts_with("s ") {
                exec_editor_command(&format!("s {}", &cmd[2..]), state, runner, tx);
                return;
            }
            if cmd.starts_with("%s/") {
                exec_editor_command(&format!("%s {}", &cmd[3..]), state, runner, tx);
                return;
            }
            state.push_log(LogEntry::new(
                LogCategory::Err,
                format!("unknown command: :{}", cmd),
            ));
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────

fn cmd_editor_help(state: &mut TuiState) {
    state.push_log(LogEntry::new(LogCategory::Jit, "── vim editor commands ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  motions"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  h/j/k/l          cursor left/down/up/right"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  w/b/e            word forward/backward/end"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  W/B/E            WORD forward/backward/end"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  0/^/$            line start/first-nonblank/end"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  gg/G             file start/end (or Ngg for line N)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  f/F/t/T{c}      find char forward/backward (till)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ;/,              repeat/reverse last f/F/t/T"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  {/}              paragraph backward/forward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  %                matching bracket"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  H/M/L            screen top/middle/bottom"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  C-f/C-b          page down/up"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  C-d/C-u          half-page down/up"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  *                search word under cursor"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  editing"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  i/a/I/A          insert/append modes"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  o/O              open line below/above"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  r{c}             replace char"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  x/X              delete char forward/backward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  dd/cc/yy         delete/change/yank line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  d{motion}        delete with motion (dw, de, d$)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  c{motion}        change with motion (cw, ce, c$)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  y{motion}        yank with motion (yw, ye, y$)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  D/C/Y            delete/change/yank to end of line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  s/S              substitute char/line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  J                join lines"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  >/>>/</<<        indent/dedent"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  p/P              put after/before"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  u/C-r            undo/redo"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  text objects (after d/c/y)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  iw/aw            inner/around word"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  i(/a(  i)/a)     inner/around parentheses"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  i[/a[  i{/a{     inner/around brackets/braces"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  i\"/a\"  i'/a'     inner/around quotes"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  search"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /{pattern}       search forward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ?{pattern}       search backward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  n/N              next/prev match"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :noh             clear highlights"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :s/pat/rep/[g]   substitute on line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :%s/pat/rep/g    substitute in file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :grep <pattern>  workspace search"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  marks & jumps"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  m{a-z}           set mark"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  '{a-z}           jump to mark line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  `{a-z}           jump to mark exact position"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ''               jump to last position"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  C-o/C-i          jump back/forward in history"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :marks           list all marks"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  registers & macros"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  \"{reg}           select register for next d/y/p"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :registers       list all registers"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  q{a-z}           start recording macro"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  q                stop recording"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  @{a-z}           play macro"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  @@               repeat last macro"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :macros          list recorded macros"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  buffers"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :e <path>        open file in new buffer"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :bn/:bp          next/previous buffer"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :bd              close buffer"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :buffers         list open buffers"));
    state.push_log(LogEntry::new(LogCategory::Sys, ""));

    state.push_log(LogEntry::new(LogCategory::Jit, "  remapping"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :map <key> <action>     remap in normal mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :imap <key> <action>    remap in insert mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :vmap <key> <action>    remap in visual mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :unmap <key>            remove binding"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :mapclear               reset to defaults"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :actions                list available actions"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :keymap                 keymap help"));
}

/// Parse a substitute command: /pattern/replacement/[g]
fn parse_substitute(input: &str) -> Option<(String, String, bool)> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    let delim = input.chars().next()?;
    if delim != '/' {
        return None;
    }

    let rest = &input[1..];
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut escaped = false;

    for ch in rest.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == delim {
            parts.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    parts.push(current);

    if parts.len() < 2 {
        return None;
    }

    let pattern = parts[0].clone();
    let replacement = parts[1].clone();
    let global = parts.get(2).map(|s| s.contains('g')).unwrap_or(false);

    Some((pattern, replacement, global))
}

/// Parse an action name string into an EditorAction.
fn parse_action_name(name: &str) -> Option<keymap::EditorAction> {
    use keymap::EditorAction::*;
    match name {
        "MoveLeft" => Some(MoveLeft),
        "MoveRight" => Some(MoveRight),
        "MoveUp" => Some(MoveUp),
        "MoveDown" => Some(MoveDown),
        "WordForward" => Some(WordForward),
        "WordBackward" => Some(WordBackward),
        "WordEnd" => Some(WordEnd),
        "BigWordForward" => Some(BigWordForward),
        "BigWordBackward" => Some(BigWordBackward),
        "BigWordEnd" => Some(BigWordEnd),
        "LineStart" => Some(LineStart),
        "LineEnd" => Some(LineEnd),
        "FirstNonBlank" => Some(FirstNonBlank),
        "GotoFirstLine" => Some(GotoFirstLine),
        "GotoLastLine" => Some(GotoLastLine),
        "PageDown" => Some(PageDown),
        "PageUp" => Some(PageUp),
        "HalfPageDown" => Some(HalfPageDown),
        "HalfPageUp" => Some(HalfPageUp),
        "ScreenTop" => Some(ScreenTop),
        "ScreenMiddle" => Some(ScreenMiddle),
        "ScreenBottom" => Some(ScreenBottom),
        "ParagraphForward" => Some(ParagraphForward),
        "ParagraphBackward" => Some(ParagraphBackward),
        "MatchingBracket" => Some(MatchingBracket),
        "FindCharForward" => Some(FindCharForward),
        "FindCharBackward" => Some(FindCharBackward),
        "TillCharForward" => Some(TillCharForward),
        "TillCharBackward" => Some(TillCharBackward),
        "RepeatFindChar" => Some(RepeatFindChar),
        "RepeatFindCharReverse" => Some(RepeatFindCharReverse),
        "InsertMode" => Some(InsertMode),
        "InsertAtLineStart" => Some(InsertAtLineStart),
        "AppendAfterCursor" => Some(AppendAfterCursor),
        "AppendAtLineEnd" => Some(AppendAtLineEnd),
        "OpenLineBelow" => Some(OpenLineBelow),
        "OpenLineAbove" => Some(OpenLineAbove),
        "VisualMode" => Some(VisualMode),
        "VisualLineMode" => Some(VisualLineMode),
        "CommandMode" => Some(CommandMode),
        "NormalMode" => Some(NormalMode),
        "ReplaceChar" => Some(ReplaceChar),
        "DeleteChar" => Some(DeleteChar),
        "DeleteCharBefore" => Some(DeleteCharBefore),
        "DeleteLine" => Some(DeleteLine),
        "YankLine" => Some(YankLine),
        "PutAfter" => Some(PutAfter),
        "PutBefore" => Some(PutBefore),
        "JoinLines" => Some(JoinLines),
        "Undo" => Some(Undo),
        "Redo" => Some(Redo),
        "Indent" => Some(Indent),
        "Dedent" => Some(Dedent),
        "ChangeToEndOfLine" => Some(ChangeToEndOfLine),
        "DeleteToEndOfLine" => Some(DeleteToEndOfLine),
        "YankToEndOfLine" => Some(YankToEndOfLine),
        "SubstituteChar" => Some(SubstituteChar),
        "SubstituteLine" => Some(SubstituteLine),
        "RepeatLastCommand" => Some(RepeatLastCommand),
        "OperatorDelete" => Some(OperatorDelete),
        "OperatorChange" => Some(OperatorChange),
        "OperatorYank" => Some(OperatorYank),
        "SearchForward" => Some(SearchForward),
        "SearchBackward" => Some(SearchBackward),
        "SearchNext" => Some(SearchNext),
        "SearchPrev" => Some(SearchPrev),
        "SearchWordUnderCursor" => Some(SearchWordUnderCursor),
        "SetMark" => Some(SetMark),
        "GotoMark" => Some(GotoMark),
        "GotoMarkLine" => Some(GotoMarkLine),
        "JumpBack" => Some(JumpBack),
        "JumpForward" => Some(JumpForward),
        "SelectRegister" => Some(SelectRegister),
        "RecordMacro" => Some(RecordMacro),
        "PlayMacro" => Some(PlayMacro),
        "BufferNext" => Some(BufferNext),
        "BufferPrev" => Some(BufferPrev),
        "BufferClose" => Some(BufferClose),
        "BufferList" => Some(BufferList),
        "ToggleFocus" => Some(ToggleFocus),
        "ToggleUiMode" => Some(ToggleUiMode),
        "Run" => Some(Run),
        "Stop" => Some(Stop),
        "ToggleRunOutput" => Some(ToggleRunOutput),
        "ScrollLineUp" => Some(ScrollLineUp),
        "ScrollLineDown" => Some(ScrollLineDown),
        "CenterCursorLine" => Some(CenterCursorLine),
        "NoOp" => Some(NoOp),
        _ => None,
    }
}

/// Search files recursively for a pattern (simple grep).
fn search_files_recursive(
    dir: &std::path::Path,
    pattern: &str,
    results: &mut Vec<(std::path::PathBuf, usize, String)>,
    limit: usize,
) {
    let lower_pattern = pattern.to_lowercase();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if results.len() >= limit {
            return;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden dirs and common non-source dirs
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }

        if path.is_dir() {
            search_files_recursive(&path, pattern, results, limit);
        } else if path.is_file() {
            // Only search text files
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let is_text = matches!(
                ext,
                "rs" | "toml" | "cu" | "cuh" | "h" | "c" | "cpp"
                    | "py" | "sh" | "md" | "txt" | "json" | "yaml"
                    | "yml" | "cfg" | "conf" | "lock"
            );
            if !is_text {
                continue;
            }

            if let Ok(content) = std::fs::read_to_string(&path) {
                for (i, line) in content.lines().enumerate() {
                    if results.len() >= limit {
                        return;
                    }
                    if line.to_lowercase().contains(&lower_pattern) {
                        results.push((path.clone(), i + 1, line.to_string()));
                    }
                }
            }
        }
    }
}
