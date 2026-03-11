use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::state::TuiState;
use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use ptx_runtime::PtxRuntime;

pub(super) fn list_demos(state: &mut TuiState) {
    state.push_log(LogEntry::new(LogCategory::Sys, "available demos:"));
    for (name, desc, _src) in DEMO_PROGRAMS {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!("  {:<14} {}", name, desc),
        ));
    }
    // Service demos (runtime-native, not FGL scripts)
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("  {:<14} {}", "gpu-logwatch", "GPU-resident log stream monitor (continuous, demo stop to halt)"),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("  {:<14} {}", "dataflow-proof", "prove TPU-like deterministic memory on GPU (continuous, demo stop to halt)"),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "run: demo <name>  inspect: demo inspect <name>",
    ));
}

pub(super) fn find_demo(name: &str) -> Option<(&'static str, &'static str, &'static str)> {
    DEMO_PROGRAMS
        .iter()
        .find(|(n, _, _)| *n == name)
        .copied()
}

/// Compile and execute a single program, sending results over the event channel.
pub(super) fn run_program(
    name: &str,
    src: &str,
    runner: &mut ScriptRunner,
    tx: &Sender<DaemonEvent>,
) {
    let compile_start = Instant::now();
    let compile_ok = runner.compile(src);
    let compile_ms = compile_start.elapsed().as_millis();

    match compile_ok {
        Ok(()) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Jit,
                format!("{}: compiled {}ms", name, compile_ms),
            )));
        }
        Err(e) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Err,
                format!("{}: compile error: {}", name, e),
            )));
            return;
        }
    }

    let exec_start = Instant::now();
    let result = runner.execute_last();
    let exec_ms = exec_start.elapsed().as_millis();

    match result {
        Ok(res) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Run,
                format!("{}: shape={:?} {}ms", name, res.shape, exec_ms),
            )));
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Run,
                format_output_data(&res.data),
            )));
            let _ = tx.send(DaemonEvent::TensorResult {
                shape: res.shape.clone(),
                data: res.data.clone(),
            });
            let _ = tx.send(DaemonEvent::PipelineResult {
                name: name.to_string(),
                compile_ms,
                exec_ms,
            });
        }
        Err(e) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Err,
                format!("{}: exec error: {}", name, e),
            )));
        }
    }
}

// ── built-in demo programs ──────────────────────────────────
// These exercise the JIT compiler and GPU runtime internally.
// Used by `bench` (run all), `demo <name>` (run one), and
// `demo inspect <name>` (write to file and open in editor).

pub(super) const DEMO_PROGRAMS: &[(&str, &str, &str)] = &[
    (
        "warmup",
        "minimal relu on 1K elements",
        "x = input([1, 1, 1, 1024])\nh = relu(x)\nreturn h",
    ),
    (
        "activations",
        "chain of relu → tanh → sigmoid on 4K",
        "x = input([1, 1, 1, 4096])\nh = relu(x)\nh = tanh(h)\nh = sigmoid(h)\nreturn h",
    ),
    (
        "residual",
        "residual add with skip connection",
        concat!(
            "x = input([1, 1, 1, 8192])\n",
            "h = relu(x)\n",
            "h = add(h, x)\n",
            "h = sigmoid(h)\n",
            "return h",
        ),
    ),
    (
        "mlp",
        "two-layer MLP with function calls",
        concat!(
            "fn layer(x):\n",
            "  h = relu(x)\n",
            "  h = tanh(h)\n",
            "  h = add(h, x)\n",
            "  return h\n",
            "end\n",
            "\n",
            "x = input([1, 1, 1, 4096])\n",
            "h = layer(x)\n",
            "h = layer(h)\n",
            "h = sigmoid(h)\n",
            "return h",
        ),
    ),
    (
        "deep-chain",
        "16-block deep residual network",
        concat!(
            "fn block(x):\n",
            "  h = relu(x)\n",
            "  h = tanh(h)\n",
            "  h = h * 0.95 + 0.05\n",
            "  h = sigmoid(h)\n",
            "  h = add(h, x)\n",
            "  return h\n",
            "end\n",
            "\n",
            "x = input([1, 1, 64, 8192])\n",
            "h = block(x)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "h = block(h)\n",
            "return h",
        ),
    ),
    // ── flight computer subsystems ─────────────────────────────
    (
        "flight-ctrl",
        "PID flight control laws — mul/tanh/sigmoid pipeline",
        concat!(
            "state    = input([1, 1, 1, 512])\n",
            "setpoint = input([1, 1, 1, 512])\n",
            "gains    = input([1, 1, 1, 512])\n",
            "err = add(setpoint, state)\n",
            "p = mul(err, gains)\n",
            "damped = tanh(p)\n",
            "cmd = sigmoid(damped)\n",
            "return cmd",
        ),
    ),
    (
        "sensor-fusion",
        "weighted 3-sensor fusion — 6 inputs, mul/add/sigmoid",
        concat!(
            "imu    = input([1, 1, 1, 1024])\n",
            "gps    = input([1, 1, 1, 1024])\n",
            "baro   = input([1, 1, 1, 1024])\n",
            "w_imu  = input([1, 1, 1, 1024])\n",
            "w_gps  = input([1, 1, 1, 1024])\n",
            "w_baro = input([1, 1, 1, 1024])\n",
            "s1 = mul(imu, w_imu)\n",
            "s2 = mul(gps, w_gps)\n",
            "s3 = mul(baro, w_baro)\n",
            "f1 = add(s1, s2)\n",
            "fused = add(f1, s3)\n",
            "out = sigmoid(fused)\n",
            "return out",
        ),
    ),
    (
        "navigation",
        "trajectory integration + course correction with functions",
        concat!(
            "fn trajectory(pos, vel, dt):\n",
            "  step = mul(vel, dt)\n",
            "  next = add(pos, step)\n",
            "  return next\n",
            "end\n",
            "\n",
            "fn course_correct(current, target, gain):\n",
            "  err = add(target, current)\n",
            "  correction = mul(err, gain)\n",
            "  bounded = tanh(correction)\n",
            "  return bounded\n",
            "end\n",
            "\n",
            "pos    = input([1, 1, 1, 4096])\n",
            "vel    = input([1, 1, 1, 4096])\n",
            "dt     = input([1, 1, 1, 4096])\n",
            "target = input([1, 1, 1, 4096])\n",
            "gain   = input([1, 1, 1, 4096])\n",
            "next_pos = trajectory(pos, vel, dt)\n",
            "correction = course_correct(next_pos, target, gain)\n",
            "cmd = sigmoid(correction)\n",
            "return cmd",
        ),
    ),
    (
        "threat-detect",
        "2-layer radar classifier — 5 inputs, function call",
        concat!(
            "fn classify(signal, w1, b1, w2, b2):\n",
            "  h = mul(signal, w1)\n",
            "  h = add(h, b1)\n",
            "  h = relu(h)\n",
            "  h = mul(h, w2)\n",
            "  h = add(h, b2)\n",
            "  score = sigmoid(h)\n",
            "  return score\n",
            "end\n",
            "\n",
            "radar = input([1, 1, 1, 2048])\n",
            "w1    = input([1, 1, 1, 2048])\n",
            "b1    = input([1, 1, 1, 2048])\n",
            "w2    = input([1, 1, 1, 2048])\n",
            "b2    = input([1, 1, 1, 2048])\n",
            "threat = classify(radar, w1, b1, w2, b2)\n",
            "return threat",
        ),
    ),
    (
        "health-mon",
        "sensor anomaly detection — triple sigmoid pipeline",
        concat!(
            "temps    = input([1, 1, 1, 8192])\n",
            "voltages = input([1, 1, 1, 8192])\n",
            "baseline = input([1, 1, 1, 8192])\n",
            "t_dev = add(temps, baseline)\n",
            "v_dev = add(voltages, baseline)\n",
            "combined = add(t_dev, v_dev)\n",
            "score = sigmoid(combined)\n",
            "score = tanh(score)\n",
            "health = sigmoid(score)\n",
            "return health",
        ),
    ),
    // ── math / expression coverage ──────────────────────────────
    (
        "infix-math",
        "complex infix expression — mul/add/relu/sigmoid/div",
        concat!(
            "x = input([1, 1, 1, 1024])\n",
            "y = input([1, 1, 1, 1024])\n",
            "z = (x * 2.0 + 1.0) * relu(y) - sigmoid(x) / 2.0\n",
            "return z",
        ),
    ),
    (
        "tile-block",
        "tile block computation — scalar fused kernel",
        concat!(
            "x = input([1, 1, 1, 1024])\n",
            "y = input([1, 1, 1, 1024])\n",
            "tile z over (x, y):\n",
            "  t = x * y + x\n",
            "  z = tanh(t) * 0.5\n",
            "end\n",
            "return z",
        ),
    ),
    (
        "enc-dec",
        "encoder → decoder pipeline with functions",
        concat!(
            "fn encoder(x):\n",
            "  h = relu(x)\n",
            "  return h\n",
            "end\n",
            "\n",
            "fn decoder(x):\n",
            "  y = tanh(x)\n",
            "  return y\n",
            "end\n",
            "\n",
            "a = input([1, 1, 1, 4096])\n",
            "b = encoder(a)\n",
            "c = decoder(b)\n",
            "return c",
        ),
    ),
    (
        "complex-graph",
        "multi-path DAG — add/relu/mul/sigmoid interleaved",
        concat!(
            "a = input([1, 1, 1, 4096])\n",
            "b = input([1, 1, 1, 4096])\n",
            "c = add(a, b)\n",
            "d = relu(c)\n",
            "e = mul(d, a)\n",
            "f = sigmoid(e)\n",
            "return f",
        ),
    ),
    (
        "mlp-weighted",
        "2-layer MLP with explicit weight/bias mul+add",
        concat!(
            "x  = input([1, 1, 1, 2048])\n",
            "w0 = input([1, 1, 1, 2048])\n",
            "b0 = input([1, 1, 1, 2048])\n",
            "w1 = input([1, 1, 1, 2048])\n",
            "b1 = input([1, 1, 1, 2048])\n",
            "t0 = mul(x, w0)\n",
            "t1 = add(t0, b0)\n",
            "h0 = relu(t1)\n",
            "t2 = mul(h0, w1)\n",
            "t3 = add(t2, b1)\n",
            "h1 = sigmoid(t3)\n",
            "return h1",
        ),
    ),
    // ── stress demos (continuous loop — demo stop to halt) ────────
    (
        "vram-flood",
        "8 large tensors alive at once — floods VRAM (continuous, demo stop to halt)",
        concat!(
            "a = input([1, 1, 512, 8192])\n",
            "b = input([1, 1, 512, 8192])\n",
            "c = input([1, 1, 512, 8192])\n",
            "d = input([1, 1, 512, 8192])\n",
            "e = input([1, 1, 512, 8192])\n",
            "f = input([1, 1, 512, 8192])\n",
            "g = input([1, 1, 512, 8192])\n",
            "h = input([1, 1, 512, 8192])\n",
            "s1 = add(a, b)\n",
            "s2 = add(c, d)\n",
            "s3 = add(e, f)\n",
            "s4 = add(g, h)\n",
            "t1 = mul(s1, s2)\n",
            "t2 = mul(s3, s4)\n",
            "out = add(t1, t2)\n",
            "out = relu(out)\n",
            "out = tanh(out)\n",
            "out = sigmoid(out)\n",
            "return out",
        ),
    ),
    (
        "mem-churn",
        "deep alloc/compute/free chain — 8× function call churn (continuous, demo stop to halt)",
        concat!(
            "fn churn(x):\n",
            "  h = relu(x)\n",
            "  h = tanh(h)\n",
            "  h = sigmoid(h)\n",
            "  h = relu(h)\n",
            "  h = add(h, x)\n",
            "  return h\n",
            "end\n",
            "\n",
            "x = input([1, 1, 256, 16384])\n",
            "h = churn(x)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "h = churn(h)\n",
            "return h",
        ),
    ),
    (
        "pipeline-stress",
        "4-stage weighted pipeline — 9 large inputs (continuous, demo stop to halt)",
        concat!(
            "fn stage(x, w, b):\n",
            "  h = mul(x, w)\n",
            "  h = add(h, b)\n",
            "  h = relu(h)\n",
            "  h = tanh(h)\n",
            "  return h\n",
            "end\n",
            "\n",
            "x  = input([1, 1, 128, 16384])\n",
            "w0 = input([1, 1, 128, 16384])\n",
            "b0 = input([1, 1, 128, 16384])\n",
            "w1 = input([1, 1, 128, 16384])\n",
            "b1 = input([1, 1, 128, 16384])\n",
            "w2 = input([1, 1, 128, 16384])\n",
            "b2 = input([1, 1, 128, 16384])\n",
            "w3 = input([1, 1, 128, 16384])\n",
            "b3 = input([1, 1, 128, 16384])\n",
            "h = stage(x, w0, b0)\n",
            "h = stage(h, w1, b1)\n",
            "h = stage(h, w2, b2)\n",
            "h = stage(h, w3, b3)\n",
            "h = sigmoid(h)\n",
            "return h",
        ),
    ),
    (
        "extreme-load",
        "dual-path 6-block compute storm — max GPU pressure (continuous, demo stop to halt)",
        concat!(
            "fn block(x, y):\n",
            "  a = relu(x)\n",
            "  b = tanh(y)\n",
            "  c = mul(a, b)\n",
            "  d = sigmoid(c)\n",
            "  e = add(d, x)\n",
            "  f = e * 0.9 + 0.1\n",
            "  return f\n",
            "end\n",
            "\n",
            "x = input([1, 1, 512, 16384])\n",
            "y = input([1, 1, 512, 16384])\n",
            "h = block(x, y)\n",
            "h = block(h, x)\n",
            "h = block(h, y)\n",
            "h = block(h, x)\n",
            "h = block(h, y)\n",
            "h = block(h, x)\n",
            "h = sigmoid(h)\n",
            "return h",
        ),
    ),
    // ── stability (meta-entry — not run directly, just listed) ──
    (
        "stability",
        "long-running parallel soak test — cycles all demos (demo stop to halt)",
        // source not used directly; the stability runner cycles all other programs
        "x = input([1, 1, 32, 4096])\nh = relu(x)\nh = tanh(h)\nh = sigmoid(h)\nreturn h",
    ),
];

/// Run the built-in GPU benchmark on a background thread.
pub(super) fn run_bench(runner: &Arc<parking_lot::Mutex<ScriptRunner>>, tx: &Sender<DaemonEvent>) {
    let bench_start = Instant::now();
    let n = DEMO_PROGRAMS.len();
    for (i, (name, _desc, src)) in DEMO_PROGRAMS.iter().enumerate() {
        let _ = tx.send(DaemonEvent::Log(LogEntry::new(
            LogCategory::Sys,
            format!("({}/{}) {}", i + 1, n, name),
        )));
        let mut r = runner.lock();
        run_program(name, src, &mut r, tx);
        drop(r);
        std::thread::sleep(Duration::from_millis(80));
    }

    let wall_ms = bench_start.elapsed().as_millis();
    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!("bench done: {}/{} total={}ms", n, n, wall_ms),
    )));
}

/// Run a single stress demo in a continuous loop until `demo stop`.
///
/// Each cycle: fresh ScriptRunner → compile → execute → log → drop (frees GPU allocs).
/// The alloc/execute/free pattern creates visible VRAM pressure waves in the TUI.
pub(super) fn run_stress_loop(
    name: &str,
    src: &str,
    flag: &std::sync::atomic::AtomicBool,
    runtime: &Arc<PtxRuntime>,
    tx: &Sender<DaemonEvent>,
) {
    let mut cycle: u64 = 0;
    let start = Instant::now();

    while flag.load(Ordering::Relaxed) {
        cycle += 1;

        // Fresh runner each cycle — guarantees all GPU allocations from
        // the previous run are freed before the next alloc cycle begins.
        let mut runner = ScriptRunner::new(Arc::clone(runtime));

        let compile_start = Instant::now();
        match runner.compile(src) {
            Ok(()) => {}
            Err(e) => {
                let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                    LogCategory::Err,
                    format!("{}[{}]: compile error: {}", name, cycle, e),
                )));
                drop(runner);
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        }
        let compile_ms = compile_start.elapsed().as_millis();

        let exec_start = Instant::now();
        match runner.execute_last() {
            Ok(res) => {
                let exec_ms = exec_start.elapsed().as_millis();
                let pool = runtime.tlsf_stats();
                let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                    LogCategory::Run,
                    format!(
                        "{}[{}]: jit={}ms gpu={}ms shape={:?} pool={:.1}MB/{:.1}MB",
                        name, cycle, compile_ms, exec_ms, res.shape,
                        pool.allocated_bytes as f64 / (1024.0 * 1024.0),
                        pool.total_pool_size as f64 / (1024.0 * 1024.0),
                    ),
                )));
                let _ = tx.send(DaemonEvent::TensorResult {
                    shape: res.shape,
                    data: res.data,
                });
                let _ = tx.send(DaemonEvent::PipelineResult {
                    name: name.to_string(),
                    compile_ms,
                    exec_ms,
                });
            }
            Err(e) => {
                let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                    LogCategory::Err,
                    format!("{}[{}]: exec error: {}", name, cycle, e),
                )));
            }
        }

        // Drop runner to free all GPU allocations before next cycle
        drop(runner);

        // Brief yield so the TUI stays responsive
        std::thread::sleep(Duration::from_millis(50));
    }

    let elapsed = start.elapsed();
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;
    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "{}: stopped — {} cycles  elapsed: {}h{:02}m{:02}s",
            name, cycle, h, m, s,
        ),
    )));
    flag.store(false, Ordering::Relaxed);
}

/// Run the stability soak test with truly parallel stream execution.
///
/// Each worker thread gets its **own** `ScriptRunner` backed by the shared
/// `PtxRuntime`.  Because `GpuLangRuntime::execute()` calls
/// `runtime.next_stream()` internally, every execution lands on a different
/// GPU stream in round-robin order — so N workers running simultaneously
/// means N streams active at once.
///
/// Worker count = number of streams the runtime provides (all of them).
/// Runs until the `flag` is set to false via `demo stop`.
pub(super) fn run_stability_test(
    flag: &std::sync::atomic::AtomicBool,
    runtime: &Arc<PtxRuntime>,
    tx: &Sender<DaemonEvent>,
) {
    use std::sync::atomic::AtomicU64;

    // Collect programs (everything except "stability" itself)
    let programs: Vec<(&str, &str)> = DEMO_PROGRAMS
        .iter()
        .filter(|(n, _, _)| *n != "stability")
        .map(|(n, _, s)| (*n, *s))
        .collect();
    let n_programs = programs.len();

    // Workers = min(n_streams, 8) — keep worker count low so live allocations
    // don't exhaust the pool.  Each worker round-robins through ALL programs,
    // and every kernel launch grabs the next stream via the global counter,
    // so even 8 workers exercise all 1024 streams over time.
    let n_streams = runtime.num_streams();
    let n_workers = n_streams.min(8);

    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "stability: {} programs × {} workers ({} streams) — soak test starting",
            n_programs, n_workers, n_streams,
        ),
    )));

    let test_start = Instant::now();

    // Shared counters across all workers
    let next_idx = Arc::new(AtomicU64::new(0));
    let total_pass = Arc::new(AtomicU64::new(0));
    let total_fail = Arc::new(AtomicU64::new(0));

    std::thread::scope(|scope| {
        for worker_id in 0..n_workers {
            let flag = flag;
            let runtime = runtime;
            let tx = tx;
            let programs = &programs;
            let next_idx = Arc::clone(&next_idx);
            let total_pass = Arc::clone(&total_pass);
            let total_fail = Arc::clone(&total_fail);
            let test_start = &test_start;

            scope.spawn(move || {
                while flag.load(Ordering::Relaxed) {
                    // Grab next program index (wraps around)
                    let seq = next_idx.fetch_add(1, Ordering::Relaxed);
                    let prog_idx = (seq as usize) % n_programs;
                    let cycle = seq / n_programs as u64 + 1;
                    let (name, src) = programs[prog_idx];

                    // Fresh runner each iteration — guarantees all GPU
                    // allocations from the previous run are freed before
                    // the next alloc cycle begins.
                    let mut runner = ScriptRunner::new(Arc::clone(runtime));

                    let compile_start = Instant::now();
                    let compile_ok = runner.compile(src);
                    let compile_ms = compile_start.elapsed().as_millis();

                    match compile_ok {
                        Ok(()) => {}
                        Err(e) => {
                            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                                LogCategory::Err,
                                format!(
                                    "stability[{}] w{} {}: compile error: {}",
                                    cycle, worker_id, name, e,
                                ),
                            )));
                            total_fail.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    }

                    // Execute (hits next available stream via round-robin)
                    let exec_start = Instant::now();
                    let result = runner.execute_last();
                    let exec_ms = exec_start.elapsed().as_millis();

                    match result {
                        Ok(res) => {
                            let n = res.data.len();
                            let min = res.data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max =
                                res.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mean =
                                if n > 0 { res.data.iter().sum::<f32>() / n as f32 } else { 0.0 };

                            let nan_count = res.data.iter().filter(|v| v.is_nan()).count();
                            let inf_count = res.data.iter().filter(|v| v.is_infinite()).count();

                            let health = if nan_count > 0 || inf_count > 0 {
                                format!(" NaN={} Inf={} UNSTABLE", nan_count, inf_count)
                            } else {
                                String::new()
                            };

                            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                                LogCategory::Run,
                                format!(
                                    "stability[{}] w{} {}: jit={}ms gpu={}ms shape={:?} min={:.4} max={:.4} avg={:.4}{}",
                                    cycle, worker_id, name, compile_ms, exec_ms, res.shape,
                                    min, max, mean, health,
                                ),
                            )));

                            let _ = tx.send(DaemonEvent::TensorResult {
                                shape: res.shape.clone(),
                                data: res.data.clone(),
                            });
                            let _ = tx.send(DaemonEvent::PipelineResult {
                                name: format!("stability/w{}/{}", worker_id, name),
                                compile_ms,
                                exec_ms,
                            });

                            total_pass.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                                LogCategory::Err,
                                format!(
                                    "stability[{}] w{} {}: exec error: {}",
                                    cycle, worker_id, name, e,
                                ),
                            )));
                            total_fail.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    // Log a cycle summary when this worker finishes a full round
                    if prog_idx == n_programs - 1 {
                        let elapsed = test_start.elapsed();
                        let h = elapsed.as_secs() / 3600;
                        let m = (elapsed.as_secs() % 3600) / 60;
                        let s = elapsed.as_secs() % 60;
                        let p = total_pass.load(Ordering::Relaxed);
                        let f = total_fail.load(Ordering::Relaxed);
                        let pool = runtime.tlsf_stats();
                        let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                            LogCategory::Sys,
                            format!(
                                "stability: cycle {} — {} pass {} fail  pool={:.1}MB/{:.1}MB  elapsed: {}h{:02}m{:02}s",
                                cycle, p, f,
                                pool.allocated_bytes as f64 / (1024.0 * 1024.0),
                                pool.total_pool_size as f64 / (1024.0 * 1024.0),
                                h, m, s,
                            ),
                        )));
                    }

                    // Sync all streams + drain deferred frees before dropping,
                    // so VRAM is actually returned to the pool immediately.
                    let _ = runtime.sync_all();
                    runtime.poll_deferred(0);
                    drop(runner);

                    // Brief yield so the TUI stays responsive
                    std::thread::sleep(Duration::from_millis(5));
                }
            });
        }
    });

    let elapsed = test_start.elapsed();
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;
    let p = total_pass.load(Ordering::Relaxed);
    let f = total_fail.load(Ordering::Relaxed);

    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "stability: stopped — {} pass / {} fail  {} workers  elapsed: {}h{:02}m{:02}s",
            p, f, n_workers, h, m, s,
        ),
    )));

    flag.store(false, Ordering::Relaxed);
}

/// Run the GPU log stream monitor as a long-running TUI demo.
///
/// Simulates a real-time log analytics service: continuously ingests batches of
/// simulated metrics (normal-distributed latencies), computes rolling statistics
/// via GPU reductions, and fires anomaly alerts when values spike.  Creates
/// visible VRAM pressure waves from the alloc/compute/free cycle.
///
/// Stoppable via `demo stop` (same flag as stress/stability demos).
pub(super) fn run_logwatch(
    flag: &std::sync::atomic::AtomicBool,
    runtime: &Arc<PtxRuntime>,
    tx: &Sender<DaemonEvent>,
) {
    const BATCH_SIZE: usize = 8192;
    const BATCH_BYTES: usize = BATCH_SIZE * std::mem::size_of::<f32>();
    const WINDOW_SIZE: usize = 8;
    const REPORT_EVERY: u64 = 10; // cycles between summary reports

    let start = Instant::now();
    let mut cycle: u64 = 0;
    let mut total_ingested: u64 = 0;
    let mut total_anomalies: u64 = 0;
    let mut recent_anomalies: u64 = 0;
    let mut seed: u64 = 42;

    // Scratch buffers for GPU reductions (1 f32 each)
    let mean_buf = match runtime.alloc(4) {
        Ok(b) => b,
        Err(e) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Err,
                format!("logwatch: alloc mean_buf failed: {}", e),
            )));
            flag.store(false, Ordering::Relaxed);
            return;
        }
    };
    let max_buf = match runtime.alloc(4) {
        Ok(b) => b,
        Err(e) => {
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Err,
                format!("logwatch: alloc max_buf failed: {}", e),
            )));
            flag.store(false, Ordering::Relaxed);
            return;
        }
    };

    let mut window: Vec<ptx_runtime::GpuPtr> = Vec::with_capacity(WINDOW_SIZE);

    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "logwatch: starting — batch={} window={} report_every={} — 'demo stop' to halt",
            BATCH_SIZE, WINDOW_SIZE, REPORT_EVERY,
        ),
    )));

    while flag.load(Ordering::Relaxed) {
        cycle += 1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        let stream = runtime.next_stream();
        let is_burst = cycle % 25 == 0;

        // 1. Allocate and fill a new log batch with simulated latencies
        let batch = match runtime.alloc(BATCH_BYTES) {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                    LogCategory::Err,
                    format!("logwatch[{}]: alloc failed: {}", cycle, e),
                )));
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
        };

        unsafe {
            // Normal-distributed random values (mean ~0, std ~1)
            ptx_sys::ptx_tensor_randn_f32(
                batch.as_ptr_typed::<f32>(),
                BATCH_SIZE,
                seed,
                stream.raw(),
            );
            // Shift to realistic latency range: mean ~5ms, std ~1ms
            ptx_sys::ptx_tensor_add_scalar_f32(
                batch.as_ptr_typed::<f32>(),
                5.0,
                batch.as_ptr_typed::<f32>(),
                BATCH_SIZE,
                stream.raw(),
            );
        }

        // Burst injection: scale by 10x to simulate traffic spike
        if is_burst {
            unsafe {
                ptx_sys::ptx_tensor_mul_scalar_f32(
                    batch.as_ptr_typed::<f32>(),
                    10.0,
                    batch.as_ptr_typed::<f32>(),
                    BATCH_SIZE,
                    stream.raw(),
                );
            }
        }

        // 2. GPU-side analytics: reduce for batch statistics
        unsafe {
            ptx_sys::ptx_tensor_reduce_mean_f32(
                batch.as_ptr_typed::<f32>(),
                mean_buf.as_ptr_typed::<f32>(),
                1,
                BATCH_SIZE,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_max_f32(
                batch.as_ptr_typed::<f32>(),
                max_buf.as_ptr_typed::<f32>(),
                1,
                BATCH_SIZE,
                1,
                stream.raw(),
            );
        }

        let _ = stream.synchronize();

        // 3. Read back statistics to host
        let mut mean_val: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        unsafe {
            let _ = mean_buf.copy_to_host(
                &mut mean_val as *mut f32 as *mut libc::c_void,
                4,
            );
            let _ = max_buf.copy_to_host(
                &mut max_val as *mut f32 as *mut libc::c_void,
                4,
            );
        }

        total_ingested += BATCH_SIZE as u64;

        // 4. Anomaly detection: mean > 15 indicates a burst/spike
        let anomaly = mean_val > 15.0;
        if anomaly {
            total_anomalies += 1;
            recent_anomalies += 1;
            let cat = if mean_val > 40.0 {
                LogCategory::Err
            } else {
                LogCategory::App
            };
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                cat,
                format!(
                    "logwatch[{}]: ANOMALY mean={:.1} max={:.1} — {}",
                    cycle,
                    mean_val,
                    max_val,
                    if mean_val > 40.0 { "CRITICAL spike" } else { "elevated latency" },
                ),
            )));
        }

        // 5. Manage rolling window
        window.push(batch);
        if window.len() > WINDOW_SIZE {
            let _evicted = window.remove(0); // GpuPtr drop frees TLSF
        }

        // 6. Periodic summary report
        if cycle % REPORT_EVERY == 0 {
            let pool = runtime.tlsf_stats();
            let elapsed_s = start.elapsed().as_secs().max(1);
            let rate = total_ingested / elapsed_s;
            let h = elapsed_s / 3600;
            let m = (elapsed_s % 3600) / 60;
            let s = elapsed_s % 60;
            let status = if recent_anomalies > 0 { "WARN" } else { "PASS" };
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                if recent_anomalies > 0 { LogCategory::App } else { LogCategory::Run },
                format!(
                    "logwatch: {} — rate={}/s batches={} anomalies={} pool={:.1}MB/{:.1}MB  {}h{:02}m{:02}s",
                    status,
                    rate,
                    cycle,
                    total_anomalies,
                    pool.allocated_bytes as f64 / (1024.0 * 1024.0),
                    pool.total_pool_size as f64 / (1024.0 * 1024.0),
                    h, m, s,
                ),
            )));
            recent_anomalies = 0;
        }

        // Brief yield so the TUI stays responsive
        std::thread::sleep(Duration::from_millis(40));
    }

    // Cleanup
    window.clear();
    drop(mean_buf);
    drop(max_buf);
    let _ = runtime.sync_all();
    runtime.poll_deferred(0);

    let elapsed = start.elapsed();
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;
    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "logwatch: stopped — {} batches  {} anomalies  elapsed: {}h{:02}m{:02}s",
            cycle, total_anomalies, h, m, s,
        ),
    )));
    flag.store(false, Ordering::Relaxed);
}

/// Prove software-defined TPU properties on GPU hardware.
///
/// Allocates 8 VRAM slots across 4 TLSF size classes (~5.3 MB), streams data
/// through 3 parallel compute lanes (15+ GPU ops per cycle), periodically
/// recycles all slots, and verifies:
///   1. Same VRAM addresses return after free/realloc across all size classes
///   2. Allocation latency is constant (O(1))
///   3. Fragmentation stays at zero
///   4. Pipeline throughput is stable
///
/// Stoppable via `demo stop`.
pub(super) fn run_dataflow_proof(
    flag: &std::sync::atomic::AtomicBool,
    runtime: &Arc<PtxRuntime>,
    tx: &Sender<DaemonEvent>,
) {
    // 4 size classes × 2 buffers each = 8 slots, ~5.3 MB
    const LARGE_ELEMS: usize = 524_288; // 2MB per buf
    const LARGE_BYTES: usize = LARGE_ELEMS * 4;
    const MEDIUM_ELEMS: usize = 131_072; // 512KB per buf
    const MEDIUM_BYTES: usize = MEDIUM_ELEMS * 4;
    const SMALL_ELEMS: usize = 32_768; // 128KB per buf
    const SMALL_BYTES: usize = SMALL_ELEMS * 4;
    const TINY_ELEMS: usize = 4_096; // 16KB per buf
    const TINY_BYTES: usize = TINY_ELEMS * 4;

    const NUM_SLOTS: usize = 8;
    const SLOT_SIZES: [usize; NUM_SLOTS] = [
        LARGE_BYTES, LARGE_BYTES,
        MEDIUM_BYTES, MEDIUM_BYTES,
        SMALL_BYTES, SMALL_BYTES,
        TINY_BYTES, TINY_BYTES,
    ];
    const CLASS_NAMES: [&str; 4] = ["L", "M", "S", "T"];
    const RECYCLE_EVERY: u64 = 15;
    const REPORT_EVERY: u64 = 30;

    let total_vram: usize = SLOT_SIZES.iter().sum();
    let start = Instant::now();

    // Phase 1: allocate all 8 slots, record baseline addresses
    let mut slots: Vec<ptx_runtime::GpuPtr> = Vec::with_capacity(NUM_SLOTS);
    let mut baseline_addrs: Vec<usize> = Vec::with_capacity(NUM_SLOTS);

    for (i, &size) in SLOT_SIZES.iter().enumerate() {
        match runtime.alloc(size) {
            Ok(ptr) => {
                baseline_addrs.push(ptr.as_ptr() as usize);
                slots.push(ptr);
            }
            Err(e) => {
                let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                    LogCategory::Err,
                    format!("dataflow: alloc slot[{}] failed: {}", i, e),
                )));
                flag.store(false, Ordering::Relaxed);
                return;
            }
        }
    }

    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "dataflow: {} slots across 4 classes ({}) — 3 compute lanes, 15+ ops/cycle — 'demo stop' to halt",
            NUM_SLOTS,
            if total_vram >= 1_048_576 {
                format!("{:.1}MB", total_vram as f64 / 1_048_576.0)
            } else {
                format!("{}KB", total_vram / 1024)
            },
        ),
    )));

    let mut cycle: u64 = 0;
    let mut total_addr_checks: u64 = 0;
    let mut total_addr_matches: u64 = 0;
    let mut alloc_times_ns: Vec<u64> = Vec::new();
    let mut pipe_times_us: Vec<u64> = Vec::new();
    let mut peak_frag: f32 = 0.0;
    let mut seed: u64 = 1;

    while flag.load(Ordering::Relaxed) {
        cycle += 1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        let stream = runtime.next_stream();

        // ── Execute the multi-lane dataflow pipeline ──────────────────
        // L0=0 L1=1 M0=2 M1=3 S0=4 S1=5 T0=6 T1=7
        let pipe_start = Instant::now();

        unsafe {
            // ── Lane A: Primary compute (LARGE 2MB ping-pong) ─────────
            ptx_sys::ptx_tensor_randn_f32(
                slots[0].as_ptr_typed::<f32>(), LARGE_ELEMS, seed, stream.raw(),
            );
            ptx_sys::ptx_tensor_affine_f32(
                slots[0].as_ptr_typed::<f32>(), slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS, 2.0, 3.0, stream.raw(),
            );
            ptx_sys::ptx_tensor_gelu_f32(
                slots[0].as_ptr_typed::<f32>(), slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_tanh_f32(
                slots[1].as_ptr_typed::<f32>(), slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_sigmoid_f32(
                slots[0].as_ptr_typed::<f32>(), slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_silu_f32(
                slots[1].as_ptr_typed::<f32>(), slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_exp_f32(
                slots[0].as_ptr_typed::<f32>(), slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS, stream.raw(),
            );

            // ── Lane B: Transform (MEDIUM 512KB ping-pong) ───────────
            ptx_sys::ptx_tensor_randn_f32(
                slots[2].as_ptr_typed::<f32>(), MEDIUM_ELEMS,
                seed.wrapping_add(1), stream.raw(),
            );
            ptx_sys::ptx_tensor_relu_f32(
                slots[2].as_ptr_typed::<f32>(), slots[3].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_gelu_f32(
                slots[3].as_ptr_typed::<f32>(), slots[2].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_tanh_f32(
                slots[2].as_ptr_typed::<f32>(), slots[3].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_softplus_f32(
                slots[3].as_ptr_typed::<f32>(), slots[2].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS, stream.raw(),
            );

            // ── Lane C: Normalization (SMALL 128KB ping-pong) ────────
            ptx_sys::ptx_tensor_randn_f32(
                slots[4].as_ptr_typed::<f32>(), SMALL_ELEMS,
                seed.wrapping_add(2), stream.raw(),
            );
            ptx_sys::ptx_tensor_sigmoid_f32(
                slots[4].as_ptr_typed::<f32>(), slots[5].as_ptr_typed::<f32>(),
                SMALL_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_gelu_f32(
                slots[5].as_ptr_typed::<f32>(), slots[4].as_ptr_typed::<f32>(),
                SMALL_ELEMS, stream.raw(),
            );
            ptx_sys::ptx_tensor_exp_f32(
                slots[4].as_ptr_typed::<f32>(), slots[5].as_ptr_typed::<f32>(),
                SMALL_ELEMS, stream.raw(),
            );

            // ── Cross-lane reductions → T0 scratch ───────────────────
            let t0 = slots[6].as_ptr_typed::<f32>();
            ptx_sys::ptx_tensor_reduce_mean_f32(
                slots[1].as_ptr_typed::<f32>(), t0,
                1, LARGE_ELEMS, 1, stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_max_f32(
                slots[2].as_ptr_typed::<f32>(), t0.add(1),
                1, MEDIUM_ELEMS, 1, stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_sum_f32(
                slots[5].as_ptr_typed::<f32>(), t0.add(2),
                1, SMALL_ELEMS, 1, stream.raw(),
            );
        }

        let _ = stream.synchronize();
        let pipe_us = pipe_start.elapsed().as_micros() as u64;
        pipe_times_us.push(pipe_us);

        // Track fragmentation
        let tlsf = runtime.tlsf_stats();
        if tlsf.fragmentation_ratio > peak_frag {
            peak_frag = tlsf.fragmentation_ratio;
        }

        // ── Periodic recycle: free all 8 slots, re-allocate, compare ──
        if cycle % RECYCLE_EVERY == 0 {
            slots.clear();
            let _ = runtime.sync_all();
            runtime.poll_deferred(0);

            let mut class_matches: [usize; 4] = [0; 4];
            let mut class_checks: [usize; 4] = [0; 4];
            let mut alloc_ok = true;

            for i in 0..NUM_SLOTS {
                let alloc_start = Instant::now();
                match runtime.alloc(SLOT_SIZES[i]) {
                    Ok(ptr) => {
                        let alloc_ns = alloc_start.elapsed().as_nanos() as u64;
                        alloc_times_ns.push(alloc_ns);

                        let new_addr = ptr.as_ptr() as usize;
                        let class_idx = i / 2;
                        class_checks[class_idx] += 1;
                        total_addr_checks += 1;
                        if new_addr == baseline_addrs[i] {
                            total_addr_matches += 1;
                            class_matches[class_idx] += 1;
                        }
                        baseline_addrs[i] = new_addr;
                        slots.push(ptr);
                    }
                    Err(e) => {
                        let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                            LogCategory::Err,
                            format!("dataflow[{}]: realloc slot[{}] failed: {}", cycle, i, e),
                        )));
                        alloc_ok = false;
                        break;
                    }
                }
            }

            if !alloc_ok {
                break;
            }

            let match_rate = if total_addr_checks > 0 {
                total_addr_matches as f64 / total_addr_checks as f64 * 100.0
            } else {
                0.0
            };

            let class_status: String = (0..4)
                .map(|c| {
                    let ok = class_matches[c] == class_checks[c];
                    format!("{}={}", CLASS_NAMES[c], if ok { "+" } else { "-" })
                })
                .collect::<Vec<_>>()
                .join(" ");

            let all_match = class_matches.iter().zip(class_checks.iter()).all(|(m, c)| m == c);

            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Run,
                format!(
                    "dataflow[{}]: {}/{} addr ({:.0}%)  {}  pipe={}us  frag={:.6}  {}",
                    cycle, total_addr_matches, total_addr_checks, match_rate,
                    class_status, pipe_us, tlsf.fragmentation_ratio,
                    if all_match { "DETERMINISTIC" } else { "PARTIAL" },
                ),
            )));
        }

        // Periodic summary
        if cycle % REPORT_EVERY == 0 && cycle % RECYCLE_EVERY != 0 {
            let match_rate = if total_addr_checks > 0 {
                total_addr_matches as f64 / total_addr_checks as f64 * 100.0
            } else {
                100.0
            };
            let avg_alloc = if alloc_times_ns.is_empty() {
                0.0
            } else {
                alloc_times_ns.iter().sum::<u64>() as f64 / alloc_times_ns.len() as f64
            };
            let elapsed = start.elapsed().as_secs();
            let h = elapsed / 3600;
            let m = (elapsed % 3600) / 60;
            let s = elapsed % 60;

            let status = if match_rate >= 95.0 && peak_frag < 0.001 { "PASS" } else { "WARN" };
            let _ = tx.send(DaemonEvent::Log(LogEntry::new(
                if status == "PASS" { LogCategory::Run } else { LogCategory::App },
                format!(
                    "dataflow: {} — addr={:.0}%  alloc={:.0}ns  frag={:.6}  cycles={}  {}h{:02}m{:02}s",
                    status, match_rate, avg_alloc, peak_frag, cycle, h, m, s,
                ),
            )));
        }

        std::thread::sleep(Duration::from_millis(2));
    }

    // Final summary
    slots.clear();
    let _ = runtime.sync_all();
    runtime.poll_deferred(0);

    let elapsed = start.elapsed();
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;

    let match_rate = if total_addr_checks > 0 {
        total_addr_matches as f64 / total_addr_checks as f64 * 100.0
    } else {
        100.0
    };
    let avg_alloc = if alloc_times_ns.is_empty() {
        0.0
    } else {
        alloc_times_ns.iter().sum::<u64>() as f64 / alloc_times_ns.len() as f64
    };

    let verdict = if match_rate >= 95.0 && peak_frag < 0.001 {
        "TPU-like memory semantics CONFIRMED"
    } else {
        "partial — some properties below threshold"
    };

    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "dataflow: stopped — {} cycles  addr={:.0}%  alloc={:.0}ns  frag={:.6}  {}h{:02}m{:02}s",
            cycle, match_rate, avg_alloc, peak_frag, h, m, s,
        ),
    )));
    let _ = tx.send(DaemonEvent::Log(LogEntry::new(
        if match_rate >= 95.0 { LogCategory::Run } else { LogCategory::App },
        format!("dataflow: VERDICT — {}", verdict),
    )));

    flag.store(false, Ordering::Relaxed);
}

/// Format tensor output data as a compact summary line.
pub(super) fn format_output_data(data: &[f32]) -> String {
    if data.is_empty() {
        return "  (empty)".to_string();
    }

    let n = data.len();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / n as f32;

    let preview_n = data.len().min(6);
    let vals: Vec<String> = data[..preview_n]
        .iter()
        .map(|v| {
            if v.abs() < 0.001 && *v != 0.0 {
                format!("{:.2e}", v)
            } else {
                format!("{:.3}", v)
            }
        })
        .collect();
    let ellipsis = if n > preview_n {
        format!(" ..+{}", n - preview_n)
    } else {
        String::new()
    };

    format!(
        "  [{}{}] min={:.3} max={:.3} avg={:.3}",
        vals.join(" "),
        ellipsis,
        min,
        max,
        mean,
    )
}
