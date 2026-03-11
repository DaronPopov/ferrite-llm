# Hardware Polling Design (No Estimation)

## Goal
Provide live telemetry from hardware/runtime counters only, with no synthetic GFLOPS estimates.

## Non-Negotiable Behavior
- If a counter cannot be read from hardware/runtime, report `N/A`.
- Do not backfill with heuristics.
- Preserve low overhead (target <1% runtime overhead in steady state).

## What We Need to Poll
- VRAM: total/used/free (driver/hardware view, not allocator-only view)
- GPU utilization: SM %, memory controller %, clocks, power, temperature
- FLOPs: SP/HP/DP/Tensor FLOP counts over interval, converted to GFLOPS
- Poll health: data source active, last sample timestamp, sample age

## Data Sources
1. NVML (device telemetry)
- Memory used/free/total
- GPU and memory utilization
- SM/memory clocks
- Power draw and temperature

2. CUPTI (hardware performance counters)
- Metric families such as `flop_count_sp`, `flop_count_hp`, `flop_count_dp`
- Tensor metrics where supported
- Collect deltas between samples to compute interval FLOPs and GFLOPS

3. Existing runtime stats (already available)
- `total_ops`, stream counts, watchdog, etc.
- Use only as auxiliary telemetry, not as FLOP substitute

## Architecture

### A. Runtime-side Hardware Sampler (C/CUDA layer)
Add a dedicated sampler subsystem in `ferrite-os/native/core/runtime`:
- `hardware_poll.cu/.h` module
- Initialization at runtime start:
  - init NVML handle for selected device
  - init CUPTI subscriber / metric group for current CUDA context
- Sampling thread (default 200 ms, configurable):
  - sample NVML metrics
  - read CUPTI counters
  - compute interval delta for FLOP counters and GFLOPS
  - write to lock-free/latest-sample struct in runtime

### B. Public FFI Surface
Add a new C struct and API:
- `GPUHardwarePollSample`
  - timestamp_ns
  - vram_total, vram_used, vram_free
  - gpu_util_percent, mem_util_percent
  - sm_clock_mhz, mem_clock_mhz
  - power_w, temp_c
  - flop_sp_delta, flop_hp_delta, flop_dp_delta, flop_tensor_delta
  - gflops_sp, gflops_hp, gflops_dp, gflops_tensor, gflops_total
  - validity flags (`nvml_valid`, `cupti_valid`, `flops_valid`)
- `gpu_hot_get_hardware_poll(runtime, out_sample)`

Then expose in:
- `internal/ptx-sys` bindings
- `ptx-runtime` safe wrapper

### C. Daemon/TUI Integration
In `tui/state.rs` refresh path:
- read `hardware_poll_sample`
- map fields directly to UI
- if `flops_valid == false`, display `GFLOPS: N/A`

## Counter Math (Exact Polling Path)
Per sample interval `dt`:
- `delta_sp = sp_counter_now - sp_counter_prev`
- `delta_hp = hp_counter_now - hp_counter_prev`
- `delta_dp = dp_counter_now - dp_counter_prev`
- `delta_tensor = tensor_counter_now - tensor_counter_prev`
- `gflops_x = (delta_x / dt_seconds) / 1e9`
- `gflops_total = gflops_sp + gflops_hp + gflops_dp + gflops_tensor`

No transformation from `ops/s` is allowed in this mode.

## Fail/Degrade Modes
- NVML unavailable: memory/util/clock/power fields become `N/A`, CUPTI still usable.
- CUPTI unavailable: GFLOPS fields become `N/A`, NVML still usable.
- Context reset: sampler re-inits and marks samples invalid until stable.

## Configuration
Add runtime config keys:
- `hardware_poll.enabled` (default true)
- `hardware_poll.interval_ms` (default 200)
- `hardware_poll.enable_nvml` (default true)
- `hardware_poll.enable_cupti` (default true)

## Validation Plan
1. Idle baseline
- GFLOPS near 0
- memory/util stable and matches `nvidia-smi`

2. Synthetic compute kernels
- GFLOPS rises/falls with workload and stream count
- expected monotonic relation with workload size

3. Cross-check
- compare util/memory to NVML CLI snapshots
- compare CUPTI counter deltas under known kernels

4. Overhead test
- ensure polling thread overhead remains low

## Implementation Phases
Phase 1: NVML polling
- hardware memory/util/clock/power/temperature in runtime and TUI

Phase 2: CUPTI FLOP counters
- expose true GFLOPS fields, remove fallback estimates from UI

Phase 3: hardening
- reset/reinit, validity flags, robust error reporting

## Current State Gap
The current TUI still includes a GFLOPS estimate path. That should be replaced with the CUPTI-backed `flops_valid` pipeline above and report `N/A` until CUPTI is active.
