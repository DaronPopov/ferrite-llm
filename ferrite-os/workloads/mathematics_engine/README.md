# Ferrite Mathematics Engine

GPU-accelerated quantitative finance and large-scale computation control plane,
powered by the Ferrite OS TLSF allocator and wave-scheduled streaming.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         mathematics_engine/                                │
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐             │
│  │ monte_carlo/  │  │  portfolio/  │  │       risk/          │             │
│  │ path_pricer   │  │ covariance   │  │    var_engine        │             │
│  │              │  │   _stream    │  │                      │             │
│  │ 50M+ paths   │  │ 10K+ assets  │  │ 5M+ scenarios        │             │
│  │ streamed     │  │ tiled cov    │  │ streamed batches     │             │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘             │
│         │                 │                      │                         │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐             │
│  │              TLSF Pool  ·  Wave Scheduling                │             │
│  │         O(1) alloc/free · bounded VRAM streaming          │             │
│  └──────┬──────────┬──────────────────┬──────────┬───────────┘             │
│         │          │                  │          │                         │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴──────────┐  ┌──────┴─────┐ │
│  │    pde/      │  │   matrix/    │  │    greeks/      │  │calibration/│ │
│  │ black_scholes│  │  streaming   │  │ adjoint_greeks  │  │  heston    │ │
│  │    _fd       │  │ decomposition│  │                 │  │ calibrator │ │
│  │ 1B+ grid pts │  │ 20K×20K+    │  │ 100K+ instrs    │  │ 450K allocs│ │
│  │ slab-streamed│  │ tile-streamed│  │ bump-and-reval  │  │ per calib  │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘  └────────────┘ │
│                                                                            │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐        │
│  │       xva/              │  │       volatility/                │        │
│  │   exposure_engine       │  │      surface_builder             │        │
│  │                         │  │                                  │        │
│  │ 10K trades × 50K scen   │  │ 1M+ quotes → implied vol        │        │
│  │ × 100 time steps        │  │ Newton-Raphson + SABR fit        │        │
│  │ 200GB cube streamed     │  │ 100K+ iterative alloc/free       │        │
│  └─────────────────────────┘  └──────────────────────────────────┘        │
│                                                                            │
│                    ┌───────────────────────┐                               │
│                    │    ferrite-os (GPU)    │                               │
│                    │  TLSF · streams · IPC │                               │
│                    └───────────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Script | Purpose |
|--------|--------|---------|
| `monte_carlo/` | `path_pricer.rs` | Stream 50M+ GBM paths for exotic option pricing (European, Asian, Barrier, Lookback) |
| `portfolio/` | `covariance_stream.rs` | Tile-streamed covariance matrix for 10K+ asset universes with portfolio optimization |
| `risk/` | `var_engine.rs` | VaR/CVaR via millions of historical/MC/stressed scenarios streamed through TLSF |
| `pde/` | `black_scholes_fd.rs` | Finite difference PDE solver on ultra-fine grids with slab-streamed spatial updates |
| `matrix/` | `streaming_decomposition.rs` | Block-Cholesky, LU, eigenvalue decomposition for matrices too large for VRAM |
| `greeks/` | `adjoint_greeks.rs` | Full Greeks surface (Δ,Γ,V,Θ,ρ,vanna,volga) for 100K+ instrument books |
| `calibration/` | `heston_calibrator.rs` | Heston stochastic vol calibration via Levenberg-Marquardt — 450K+ alloc/free cycles |
| `xva/` | `exposure_engine.rs` | CVA/DVA/FVA for OTC portfolios — 200GB exposure cube streamed through VRAM |
| `volatility/` | `surface_builder.rs` | 1M+ quote implied vol inversion + SABR surface fit — iteration-heavy allocation |

## Usage

All scripts run via `ferrite-run`:

```bash
# Monte Carlo: 50M paths, Asian option
ferrite-run mathematics_engine/monte_carlo/path_pricer.rs \
  --paths 50000000 --steps 252 --option asian --spot 100.0 --strike 100.0

# Portfolio: 10K asset covariance matrix (400MB matrix, never in VRAM)
ferrite-run mathematics_engine/portfolio/covariance_stream.rs \
  --assets 10000 --observations 2520 --tile-size 512

# Risk: 5M scenario VaR
ferrite-run mathematics_engine/risk/var_engine.rs \
  --scenarios 5000000 --assets 2000 --confidence 0.99 --method montecarlo

# PDE: 50K×25K grid Black-Scholes (5GB grid, streamed)
ferrite-run mathematics_engine/pde/black_scholes_fd.rs \
  --spot-points 50000 --time-points 25000 --option call

# Matrix: 20K×20K Cholesky decomposition (1.6GB matrix, tiled)
ferrite-run mathematics_engine/matrix/streaming_decomposition.rs \
  --dim 20000 --tile-size 512 --method cholesky

# Greeks: 100K instrument book, full surface
ferrite-run mathematics_engine/greeks/adjoint_greeks.rs \
  --instruments 100000 --batch-size 10000

# Heston calibration: 200 strikes × 8 maturities, 50K MC paths per pricing
ferrite-run mathematics_engine/calibration/heston_calibrator.rs \
  --strikes 200 --maturities 8 --mc-paths 50000 --lm-iters 40

# XVA: 10K trades × 50K scenarios × 100 time steps (200GB exposure cube)
ferrite-run mathematics_engine/xva/exposure_engine.rs \
  --trades 10000 --scenarios 50000 --timesteps 100

# Vol surface: 1M quotes, Newton-Raphson + SABR calibration
ferrite-run mathematics_engine/volatility/surface_builder.rs \
  --quotes 1000000 --expiries 20 --newton-max 20 --sabr-iters 40
```

## Why These Problems Need TLSF

The new modules (`calibration/`, `xva/`, `volatility/`) demonstrate three distinct
allocation patterns that are infeasible with standard `cudaMalloc`:

### 1. Heston Calibrator — Extreme Allocation Count
Each Levenberg-Marquardt iteration prices the full option surface via Monte Carlo
(1600 options × MC shards × multiple tensors), then re-prices with parameter bumps
for Jacobian computation. Over 40 iterations: **~450,000 alloc/free cycles**.
With `cudaMalloc` at ~1ms each, that's **7+ minutes of pure allocation overhead**.
With TLSF at 0.24μs: **0.1 seconds**.

### 2. XVA Exposure Engine — Extreme Data Volume
The exposure cube (trades × scenarios × time steps) can be 200GB+ at full scale.
It never fits in VRAM — each time-slice is streamed shard-by-shard, valued, reduced,
and freed. The nested time × scenario loop creates **sustained allocation pressure**
across the entire computation, requiring O(1) alloc/free to avoid becoming the bottleneck.

### 3. Vol Surface Builder — Iteration-Heavy Allocation
Newton-Raphson inversion creates ~12 intermediate tensors per iteration per shard,
all allocated and freed within the convergence loop. With 1M quotes across 20 shards
and 15 Newton iterations: **~3,600 alloc/free per shard, 72,000 total** just for
inversion. SABR calibration adds more iterative allocation churn on top.

## Design Principles

1. **VRAM as throughput, not storage** — data streams through the GPU, never fully resident
2. **O(1) allocation** — TLSF gives deterministic alloc/free regardless of fragmentation
3. **Wave scheduling** — bounded concurrent shards across CUDA streams
4. **Compute > capacity** — problem sizes defined by compute budget, not memory capacity
5. **No Python in the critical path** — pure Rust from CLI to kernel dispatch
6. **Allocation pressure as a first-class concern** — iterative algorithms create extreme alloc/free churn that only O(1) allocators can sustain
