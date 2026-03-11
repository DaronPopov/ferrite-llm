# Ferrite Fine-Tune Engine

Control plane for streamed fine-tuning on the Ferrite TLSF + PTX runtime.

Streams model shards through bounded VRAM via wave-scheduled CUDA streams, training only LoRA adapters while base weights flow through as frozen data. Every module runs on the custom O(1) TLSF allocator instead of CUDA's caching allocator.

## Architecture

```
                        ┌──────────────────────────┐
                        │  Orchestration Layer      │
                        │  (scripting_finetune.rs)  │
                        └─────────┬────────────────┘
              ┌───────────┬───────┴────┬───────────┬──────────┐
              ▼           ▼            ▼           ▼          ▼
        ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
        │ loader/  │ │dataset/ │ │scheduler│ │  eval/ │ │telemetry│
        │ shard    │ │ packed  │ │ lr      │ │ valid  │ │ metrics│
        │ index    │ │pipeline │ │schedule │ │ loop   │ │collect │
        └────┬─────┘ └────┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
             │            │           │           │          │
        ┌────┴─────┐      │      ┌────┴────┐     │          │
        │checkpoint│      │      │quantize/│     │          │
        │ save/    │      │      │streaming│     │          │
        │ restore  │      │      │ quant   │     │          │
        └────┬─────┘      │      └────┬────┘     │          │
             │            │           │           │          │
        ┌────┴─────┐      │      ┌────┴────┐     │          │
        │  merge/  │      │      │distribtd│     │          │
        │  adapter │      │      │multi_gpu│     │          │
        │  export  │      │      │  wave   │     │          │
        └──────────┘      │      └─────────┘     │          │
                          │                      │          │
              ┌───────────┴──────────────────────┴──────────┘
              ▼
        ┌──────────────────────────────────┐
        │  Ferrite Runtime                 │
        │  TLSF Allocator + PTX Streams    │
        │  (ferriterc/ferrite-os)          │
        └──────────────────────────────────┘
```

## Modules

### Core Training

| File | Purpose |
|------|---------|
| `scripting_finetune.rs` | Main streamed LoRA training loop with wave scheduling |
| `pull_hf_model.rs` | Pull models from HuggingFace via git-lfs |

### loader/ — Weight Loading

| File | Purpose |
|------|---------|
| `safetensors_shard.rs` | Parse and stream safetensors files, memory-mapped tensor loading |
| `shard_index.rs` | Build shard index/manifest, group tensors into streaming chunks |

### checkpoint/ — State Persistence

| File | Purpose |
|------|---------|
| `save_adapter.rs` | Serialize/deserialize LoRA adapter checkpoints (weights + optional optimizer tensors) |

### eval/ — Validation

| File | Purpose |
|------|---------|
| `validation_loop.rs` | Forward-only eval pass over held-out split, reports loss + perplexity |

### scheduler/ — Learning Rate

| File | Purpose |
|------|---------|
| `lr_schedule.rs` | LR schedules: constant, linear warmup, cosine decay, step decay, one-cycle |

### merge/ — Adapter Export

| File | Purpose |
|------|---------|
| `merge_adapter.rs` | Fold LoRA deltas into base weights, export merged safetensors |

### quantize/ — Reduced Precision Streaming

| File | Purpose |
|------|---------|
| `streaming_quantize.rs` | Stream quantized shards (f16/bf16/int8/nf4), dequantize on-the-fly |

### dataset/ — Data Pipeline

| File | Purpose |
|------|---------|
| `packed_pipeline.rs` | Packed sequence batching with byte tokenizer, shuffling, cycling |

### telemetry/ — Observability

| File | Purpose |
|------|---------|
| `training_metrics.rs` | Structured metrics collector, divergence detection, CSV export, TLSF health |

### distributed/ — Multi-GPU

| File | Purpose |
|------|---------|
| `multi_gpu_wave.rs` | Distribute shard waves across GPUs with gradient all-reduce |

## Usage

### Streamed LoRA Training (synthetic)

```bash
./ferriterc/ferrite-run ./finetune_engine/scripting_finetune.rs --torch -- \
  --weights-source synthetic \
  --model-gb 100 --shard-mb 64 --steps 200 \
  --streams 64 --wave-streams 16 --micro-batch 8 \
  --hidden 2048 --lora-rank 16 \
  --optimizer adamw --schedule cosine_decay \
  --lr 0.001 --min-lr 0.0001 --warmup-steps 200 \
  --inner-steps 2 --grad-clip 1.0 --weight-decay 0.01 \
  --checkpoint-path /tmp/finetune.ckpt --save-every 25
```

### Real Weights (directory shape profile)

```bash
./ferriterc/ferrite-run ./finetune_engine/scripting_finetune.rs --torch -- \
  --weights-source directory --weights-dir /path/to/model \
  --dataset /path/to/train.jsonl --dataset-format jsonl \
  --steps 200 --streams 64 --wave-streams 16 \
  --optimizer sgd --schedule linear_warmup --momentum 0.9
```

### Resume from Checkpoint

```bash
./ferriterc/ferrite-run ./finetune_engine/scripting_finetune.rs --torch -- \
  --weights-source synthetic \
  --model-gb 100 --shard-mb 64 --steps 400 \
  --streams 64 --wave-streams 16 --micro-batch 8 \
  --hidden 2048 --lora-rank 16 \
  --optimizer adamw --schedule cosine_decay \
  --lr 0.001 --min-lr 0.0001 --warmup-steps 200 \
  --checkpoint-path /tmp/finetune.ckpt --resume --save-every 25
```

`scripting_finetune.rs` now writes checkpoint format `FRTA_CKP` version `2`.  
Version 2 adds per-shard optimizer tensors (`m`, `v`, `velocity`) so `--resume` can continue with optimizer history instead of cold-starting moments.  
Loaders accept both version 1 and version 2 checkpoints.

### Build Shard Index

```bash
./ferriterc/ferrite-run ./finetune_engine/loader/shard_index.rs --torch -- \
  --dir /path/to/model --chunk-mb 64 --output shard_manifest.txt
```

### Scan Safetensors

```bash
./ferriterc/ferrite-run ./finetune_engine/loader/safetensors_shard.rs --torch -- \
  --dir /path/to/model --verbose
```

### Checkpoint Save/Load

```bash
# Round-trip validation
./ferriterc/ferrite-run ./finetune_engine/checkpoint/save_adapter.rs --torch -- \
  --mode save --path /tmp/adapter_step100.ckpt

# Inspect checkpoint
./ferriterc/ferrite-run ./finetune_engine/checkpoint/save_adapter.rs --torch -- \
  --mode info --path /tmp/adapter_step100.ckpt
```

### Validation Loop

```bash
./ferriterc/ferrite-run ./finetune_engine/eval/validation_loop.rs --torch -- \
  --dataset /path/to/eval.jsonl --split-ratio 0.9 \
  --weights-source synthetic --model-gb 100 --shard-mb 64 \
  --micro-batch 8 --hidden 2048 --lora-rank 16 \
  --adapter-path /tmp/finetune.ckpt
```

Validation uses the same deterministic synthetic base-weight construction as training (`synthetic` mode), so repeated eval runs and post-resume comparisons stay stable.

### LR Schedule Preview

```bash
./ferriterc/ferrite-run ./finetune_engine/scheduler/lr_schedule.rs -- \
  --schedule cosine_decay --lr 0.001 --total-steps 200 --warmup-steps 20
```

### Merge Adapters into Base Weights

```bash
./ferriterc/ferrite-run ./finetune_engine/merge/merge_adapter.rs --torch -- \
  --base-dir /path/to/model --adapter-path /tmp/adapter.ckpt \
  --output-dir /path/to/merged --dtype f16 --verify
```

### Quantized Shard Streaming

```bash
./ferriterc/ferrite-run ./finetune_engine/quantize/streaming_quantize.rs --torch -- \
  --quant nf4 --model-gb 70 --shard-mb 64 --steps 100 \
  --streams 64 --wave-streams 16
```

### Packed Dataset Pipeline

```bash
./ferriterc/ferrite-run ./finetune_engine/dataset/packed_pipeline.rs -- \
  --dataset /path/to/train.jsonl --seq-len 2048 --batch-size 8
```

### Telemetry (demo mode)

```bash
./ferriterc/ferrite-run ./finetune_engine/telemetry/training_metrics.rs -- \
  --demo-steps 200 --metrics-csv /tmp/metrics.csv --health-csv /tmp/health.csv
```

### Multi-GPU Wave Scheduling

```bash
./ferriterc/ferrite-run ./finetune_engine/distributed/multi_gpu_wave.rs --torch -- \
  --gpus 4 --model-gb 100 --shard-mb 64 --steps 100 \
  --grad-accum 4 --streams-per-gpu 64
```

## Design Principles

- **Bounded VRAM**: Wave scheduling guarantees only `wave_streams` shards are in-flight. A 100GB model fits on an 8GB card.
- **O(1) Allocation**: Every tensor allocation goes through the TLSF allocator (0.238us vs ~1000us for cudaMalloc).
- **Composable Scripts**: Each module is a standalone `.rs` script runnable via `ferrite-run`, but also exposes `pub` types for import by the training orchestrator.
- **Machine-Readable Output**: Every script emits `RESULT key=value` lines for automated benchmarking and CI.
- **No Python**: The entire control plane is Rust. No GIL, no framework overhead, no pip.
