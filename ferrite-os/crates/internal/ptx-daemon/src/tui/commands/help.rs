use crate::events::{LogCategory, LogEntry};
use crate::tui::state::TuiState;

pub(super) fn cmd_help(state: &mut TuiState) {
    // Navigation — always useful, shown first
    state.push_log(LogEntry::new(LogCategory::Jit, "── navigation ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ctrl+o           toggle shell/files mode",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /sysmon [on|off]  visual telemetry dashboard",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /detail          toggle tensor/pipeline detail",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /density         set ui density: auto/compact/balanced/comfortable",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /fxscript        load scripted tachyonfx effects (status/off/reset/load)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /plot3d         3D render window control (status/open/close/scene/tensor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  tab              toggle log detail/summary view",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ↑↓               scroll sysmon sections (small view) or log/history",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pgup/pgdn        jump section viewport when sysmon overflows",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── processes ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ps               list running processes",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  start <app>      launch managed app",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  stop             stop all running programs (keeps daemon alive)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  kill <id|name>   stop a single process",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  uptime           daemon uptime + request counts",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── memory ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pool             vram pool status",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pool check       deep pool validation",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  defrag           defragment vram",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── gpu ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  stats            runtime stats",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  hwpoll           hardware polling status (nvml/cupti)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  streams          stream pool info",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  snapshot         system state snapshot",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  health           health check",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── demos ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  bench            gpu compute benchmark",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demos            list available demo programs",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo <name>      run a demo program",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo stop        halt running stress demo",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo gpu-logwatch GPU log stream monitor (continuous)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo dataflow-proof prove TPU-like memory determinism (continuous)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo inspect <n> open demo source in editor",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── workspace ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /files           open file explorer/editor",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /open <path>     open file directly",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /save            save current file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  clear            clear log",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quit             shutdown daemon",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── filesystem ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  mkdir <path>     create directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  touch <path>     create empty file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  mv <src> <dst>   move/rename file or directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  cp <src> <dst>   copy file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  rm <path>        remove file/dir (requires confirm)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ls [path]        list directory contents",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  cd [path]        change working directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pwd              show working directory",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── run (gpu script) ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  run              compile+execute open gpu script (ScriptRunner)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  rerun            re-run last gpu script",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── run-file (rust entry runner) ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  run-file [path] [--entry <name>] [-- <args>]",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "                   run any .rs file via ptx-runner (ctrl+5 in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  run-entry <id> [-- <args>]  run by logical entry ID",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  run-list         list entries (click to fill command)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /stop            stop running program (ctrl+6 in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /output          toggle run output panel",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  args <...>       set run arguments",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  profile [d|r]    set debug/release profile",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  timeout [secs]   set/show execution timeout (default 60s)",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── inspect / profiling ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ptx              inspect last compiled program (opens buffer in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  perf             kernel profiling history + statistics",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  jit-clear        clear JIT compilation cache",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  jit-cache        show JIT cache stats (memory + disk)",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── vram quota ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota            show per-owner VRAM usage",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota set <id> <mb>  set soft limit (MB) for owner",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota clear <id>     remove soft limit for owner",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota label <id> <n> name an owner",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── editor (vim-like) ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :help            full editor command reference (in editor mode)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :keymap          keymap/remapping help",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :map             list current keybindings",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :map <key> <act> remap key to action (e.g. :map <C-p> SearchForward)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :actions         list all remappable actions",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :marks           list marks  |  :registers  list registers",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :grep <pat>      workspace search  |  :s/pat/rep/g  substitute",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :buffers         list open buffers  |  :bn/:bp  switch buffer",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── agent ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  audit            show last 20 agent actions",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── scheduler / control plane ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler        open scheduler dashboard (ctrl+o to exit)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler pause  pause the scheduler queue",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler resume resume the scheduler queue",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler stats  scheduler statistics",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler policies  list active policy rules",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /policy          show policy status and recent decisions",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  audit-query [--tenant ID] [--last N]  query audit log",
    ));

    state.push_log(LogEntry::new(LogCategory::Sys, ""));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  type 'binds' for all keybindings",
    ));
}

pub(super) fn cmd_binds(state: &mut TuiState) {
    // ── global (all modes) ──────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── global keybindings ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+o           toggle shell / files / scheduler mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+4           cycle color theme"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+5           run-file (run open .rs file via ptx-runner)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+6           stop running process"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+c           quit (shell mode) / cancel (editor)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+d           quit (shell mode) / half-page down (editor)"));

    // ── shell mode ──────────────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── shell mode ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  enter            submit command"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  esc              clear input"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ↑ / ↓            history (when typing) or scroll log"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  pgup / pgdn      scroll log or sysmon sections"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  tab              toggle focus (log / processes)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+a / home    cursor to start of line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+e / end     cursor to end of line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+k           kill to end of line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+w           kill previous word"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+u           clear line"));

    // ── files mode: tree ────────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: tree panel ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  j / ↓            move cursor down"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  k / ↑            move cursor up"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  enter            open file / toggle directory"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  l                expand directory or focus editor"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  h                collapse directory / go to parent"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  r                refresh file index"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  tab              toggle tree / editor focus"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  esc / q          exit files mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  pgup / pgdn      page up/down"));

    // ── files mode: editor ──────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: editor (normal) ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  i                enter insert mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  v                enter visual mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :                enter command mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  h/j/k/l          cursor left/down/up/right"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  w / b / e        word forward / back / end"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  0 / $ / ^        line start / end / first non-blank"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  gg / G           go to top / bottom"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  dd / yy / p      delete line / yank line / paste"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  d{motion}        delete with motion (dw, d$, etc.)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  c{motion}        change with motion (cw, c$, etc.)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  u / ctrl+r       undo / redo"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  / / ?            search forward / backward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  n / N            next / previous search match"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  f{c} / F{c}     find char forward / backward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  t{c} / T{c}     till char forward / backward"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ; / ,            repeat / reverse last find"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  m{a-z}           set mark  |  '{a-z}  go to mark"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  \"{a-z}           use register for next yank/delete"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  q{a-z}           record macro  |  @{a-z}  play macro"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  {count}{motion}  repeat motion N times (e.g. 5j)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  > / <            indent / dedent line"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+d / ctrl+u  half-page down / up"));

    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: editor (insert) ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  esc              return to normal mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+s           save file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  type             insert text at cursor"));

    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: editor (visual) ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  esc              cancel selection"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  d                delete selection"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  y                yank selection"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  h/j/k/l          extend selection"));

    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: editor (command) ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :w               save file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :q               exit files mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :e <path>        open file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :s/pat/rep/g     substitute"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :grep <pat>      workspace search"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :bn / :bp        next / previous buffer"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :buffers         list open buffers"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :map <k> <act>   remap key to action"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :marks           list marks"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :registers       list registers"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  :actions         list remappable actions"));

    // ── files mode: shared ──────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── files mode: shared ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+s           save file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+j           toggle run output panel"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+5           run current .rs file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+6           stop running process"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+o           exit files mode"));

    // ── scheduler mode ──────────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── scheduler mode ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  tab              cycle panels (queue / tenants / policy)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ↑ / ↓            select row"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  p                toggle pause / resume queue"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  k                kill selected job (queue view)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  r                reprioritize selected job (queue view)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  esc              exit scheduler mode"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ctrl+o           exit scheduler mode"));

    // ── shell commands ──────────────────────────────────────────
    state.push_log(LogEntry::new(LogCategory::Jit, "── shell commands ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  help             full command reference"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  binds            this keybinding reference"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ping             test connectivity"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  status           pool status"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  stats            gpu runtime stats"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  metrics          comprehensive metrics"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  health           health check"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  snapshot         system snapshot"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ps / apps        list managed processes"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  start <app>      launch managed app"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  stop             stop all running programs"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  kill <id|name>   stop a single process"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  uptime           daemon uptime"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  pool             vram pool status"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  defrag           defragment vram"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  hwpoll           hardware polling status"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  streams          stream pool info"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  bench            gpu benchmark"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  demos            list demos"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  demo <name>      run demo"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  run              gpu script run (ScriptRunner)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  rerun            re-run last gpu script"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  run-file [path]  run .rs file via ptx-runner"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  run-entry <id>   run by entry ID"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  run-list         list entries (click to run)"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /stop            stop running process"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /output          toggle run output panel"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  args <...>       set run arguments"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  profile [d|r]    set debug/release"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  timeout [secs]   set execution timeout"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ptx / inspect    inspect last compiled program"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  perf             profiling history"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  jit-clear        clear JIT cache"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  jit-cache        JIT cache stats"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  quota            vram quota management"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  scheduler        scheduler dashboard / control"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /policy          policy status"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  audit-query      query audit log"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /files           open file explorer"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /open <path>     open file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /save            save file"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  /sysmon [on|off] telemetry dashboard"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  ls / cd / pwd    filesystem navigation"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  mkdir / touch / mv / cp / rm  filesystem ops"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  clear            clear log"));
    state.push_log(LogEntry::new(LogCategory::Sys, "  quit             shutdown daemon"));
}
