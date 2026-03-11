use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, set_torch_stream, sync_all_streams, reset_torch_stream};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

const DEFAULT_STREAMS: u32 = 32;
const DEFAULT_WAVE: usize = 8;
const DEFAULT_MICRO_BATCH: i64 = 8;
const DEFAULT_HIDDEN: i64 = 2048;
const DEFAULT_LORA_RANK: i64 = 16;
const DEFAULT_LORA_ALPHA: f64 = 32.0;
const DEFAULT_TEXT_KEY: &str = "text";
const DEFAULT_MODEL_GB: usize = 100;
const MAGIC: &[u8; 8] = b"FRTA_CKP";
const VERSION: u32 = 2;
const MIN_SUPPORTED_VERSION: u32 = 1;
const CHECKPOINT_HEADER_BYTES: usize = 80;

#[derive(Clone, Copy)]
enum WeightsSource {
    Synthetic,
    Directory,
}

impl WeightsSource {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "synthetic" => Ok(Self::Synthetic),
            "directory" => Ok(Self::Directory),
            _ => Err(anyhow::anyhow!("invalid --weights-source: {s} (expected synthetic|directory)")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Directory => "directory",
        }
    }
}

#[derive(Clone)]
struct Config {
    weights_source: WeightsSource,
    weights_dir: Option<PathBuf>,
    adapter_path: Option<PathBuf>,
    model_gb: usize,
    dataset: PathBuf,
    dataset_format: DatasetFormat,
    text_key: String,
    shard_mb: usize,
    streams: u32,
    wave_streams: usize,
    micro_batch: i64,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    max_samples: usize,
    split_ratio: f64,
}

#[derive(Clone, Copy)]
enum DatasetFormat {
    Auto,
    Txt,
    Jsonl,
}

impl DatasetFormat {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "txt" => Ok(Self::Txt),
            "jsonl" => Ok(Self::Jsonl),
            _ => Err(anyhow::anyhow!("invalid format: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Txt => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

fn detect_format(path: &PathBuf) -> DatasetFormat {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext == "jsonl" { DatasetFormat::Jsonl } else { DatasetFormat::Txt }
}

fn extract_json_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = line.find(&needle)?;
    let mut i = idx + needle.len();
    let bytes = line.as_bytes();
    while i < bytes.len() && bytes[i] != b':' { i += 1; }
    i += 1;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
    if i >= bytes.len() || bytes[i] != b'"' { return None; }
    i += 1;
    let mut out = String::new();
    let mut esc = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        i += 1;
        if esc { out.push(c); esc = false; continue; }
        if c == '\\' { esc = true; continue; }
        if c == '"' { return Some(out); }
        out.push(c);
    }
    None
}

fn load_samples(path: &PathBuf, fmt: DatasetFormat, key: &str) -> Result<Vec<String>> {
    let fmt = match fmt {
        DatasetFormat::Auto => detect_format(path),
        x => x,
    };
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        match fmt {
            DatasetFormat::Txt => samples.push(trimmed.to_string()),
            DatasetFormat::Jsonl => {
                if let Some(text) = extract_json_value(trimmed, key) {
                    let t = text.trim();
                    if !t.is_empty() { samples.push(t.to_string()); }
                }
            }
            DatasetFormat::Auto => {}
        }
    }
    Ok(samples)
}

fn discover_files(dir: &PathBuf) -> Result<Vec<(PathBuf, usize)>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let meta = fs::metadata(&path)?;
        let len = meta.len() as usize;
        if len > 0 {
            files.push((path, len));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    if files.is_empty() {
        return Err(anyhow::anyhow!("no non-empty files found in weights dir"));
    }
    Ok(files)
}

fn chunk_file_sizes(files: &[(PathBuf, usize)], chunk_bytes: usize) -> Vec<usize> {
    let mut shards = Vec::new();
    for (_, mut len) in files {
        while len > 0 {
            let take = len.min(chunk_bytes);
            shards.push(take);
            len -= take;
        }
    }
    shards
}

#[derive(Debug)]
struct AdapterCheckpoint {
    version: u32,
    step: usize,
    loss: f64,
    lora_rank: i64,
    lora_alpha: f64,
    lr: f64,
    hidden: i64,
    shard_count: usize,
    adapters_a: Vec<Tensor>,
    adapters_b: Vec<Tensor>,
}

#[derive(Clone, Debug)]
struct CheckpointHeader {
    version: u32,
    step: u64,
    loss_bits: u64,
    lora_rank: u32,
    hidden: u32,
    shard_count: u32,
    lora_alpha_bits: u64,
    lr_bits: u64,
    _timestamp: u64,
    tensor_count: u32,
}

impl CheckpointHeader {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < CHECKPOINT_HEADER_BYTES {
            return Err(anyhow::anyhow!("header too short: {} bytes", data.len()));
        }
        if &data[0..8] != MAGIC {
            return Err(anyhow::anyhow!("invalid checkpoint magic"));
        }
        Ok(Self {
            version: u32::from_le_bytes(data[8..12].try_into()?),
            step: u64::from_le_bytes(data[12..20].try_into()?),
            loss_bits: u64::from_le_bytes(data[20..28].try_into()?),
            lora_rank: u32::from_le_bytes(data[28..32].try_into()?),
            hidden: u32::from_le_bytes(data[32..36].try_into()?),
            shard_count: u32::from_le_bytes(data[36..40].try_into()?),
            lora_alpha_bits: u64::from_le_bytes(data[40..48].try_into()?),
            lr_bits: u64::from_le_bytes(data[48..56].try_into()?),
            _timestamp: u64::from_le_bytes(data[56..64].try_into()?),
            tensor_count: u32::from_le_bytes(data[64..68].try_into()?),
        })
    }
}

#[derive(Clone, Debug)]
struct TensorMeta {
    name: String,
    dims: Vec<i64>,
    data_bytes: u64,
}

impl TensorMeta {
    fn from_reader(data: &[u8], offset: &mut usize) -> Result<Self> {
        let name_len = u16::from_le_bytes(read_chunk(data, offset, 2)?.try_into()?);
        let name = String::from_utf8(read_chunk(data, offset, name_len as usize)?.to_vec())?;
        let ndim = read_chunk(data, offset, 1)?[0];
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(i64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?));
        }
        let _dtype = read_chunk(data, offset, 1)?[0];
        let data_bytes = u64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?);
        Ok(Self {
            name,
            dims,
            data_bytes,
        })
    }
}

fn read_chunk<'a>(data: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = (*offset).saturating_add(len);
    if end > data.len() {
        return Err(anyhow::anyhow!(
            "checkpoint truncated: need {} bytes at offset {}, total {}",
            len,
            *offset,
            data.len()
        ));
    }
    let chunk = &data[*offset..end];
    *offset = end;
    Ok(chunk)
}

fn decode_f32_bytes(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(anyhow::anyhow!(
            "invalid f32 blob size: {} bytes",
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / std::mem::size_of::<f32>());
    for bytes in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    Ok(out)
}

fn load_checkpoint(path: &PathBuf, device: Device) -> Result<AdapterCheckpoint> {
    let data = fs::read(path)?;
    let header = CheckpointHeader::from_bytes(&data)?;
    if header.version < MIN_SUPPORTED_VERSION || header.version > VERSION {
        return Err(anyhow::anyhow!(
            "checkpoint version mismatch: file={} supported={}..={}",
            header.version,
            MIN_SUPPORTED_VERSION,
            VERSION
        ));
    }

    let mut offset = CHECKPOINT_HEADER_BYTES;
    let mut adapters_a = Vec::new();
    let mut adapters_b = Vec::new();
    for _ in 0..header.tensor_count {
        let meta = TensorMeta::from_reader(&data, &mut offset)?;
        let byte_count = meta.data_bytes as usize;
        let raw = read_chunk(&data, &mut offset, byte_count)?;
        let floats = decode_f32_bytes(raw)?;

        let tensor = Tensor::from_slice(&floats)
            .reshape(&meta.dims)
            .to_device(device)
            .set_requires_grad(true);
        if meta.name.ends_with("_lora_a") {
            adapters_a.push(tensor);
        } else if meta.name.ends_with("_lora_b") {
            adapters_b.push(tensor);
        }
    }

    Ok(AdapterCheckpoint {
        version: header.version,
        step: header.step as usize,
        loss: f64::from_bits(header.loss_bits),
        lora_rank: header.lora_rank as i64,
        lora_alpha: f64::from_bits(header.lora_alpha_bits),
        lr: f64::from_bits(header.lr_bits),
        hidden: header.hidden as i64,
        shard_count: header.shard_count as usize,
        adapters_a,
        adapters_b,
    })
}

fn synthetic_base_weight(rows: i64, hidden: i64, shard_idx: usize, device: Device, inv_sqrt_d: f64) -> Tensor {
    let used = rows * hidden;
    let idx = Tensor::arange(used, (Kind::Float, device));
    let phase = (shard_idx as f64 + 1.0) * 0.0137;
    let wave = (&idx * 0.011 + phase).sin() + (&idx * 0.017 + phase * 1.7).cos();
    (wave * (0.5 * inv_sqrt_d)).view([rows, hidden])
}

fn encode_text(text: &str, len: i64, seed: u64) -> Tensor {
    let mut out = vec![0f32; len as usize];
    let bytes = text.as_bytes();
    if bytes.is_empty() { return Tensor::from_slice(&out).view([len]); }
    let mut h = 1469598103934665603u64 ^ seed;
    for (i, &b) in bytes.iter().enumerate() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
        let idx = (h as usize + i) % out.len();
        out[idx] += ((b as f32 / 255.0) * 2.0) - 1.0;
    }
    let n = (bytes.len() as f32).sqrt().max(1.0);
    for v in &mut out { *v /= n; }
    Tensor::from_slice(&out).view([len])
}

struct EvalMetrics {
    total_loss: f64,
    sample_count: usize,
    non_finite: usize,
    per_shard_loss: Vec<f64>,
    per_shard_count: Vec<usize>,
}

impl EvalMetrics {
    fn new(shard_count: usize) -> Self {
        Self {
            total_loss: 0.0,
            sample_count: 0,
            non_finite: 0,
            per_shard_loss: vec![0.0; shard_count],
            per_shard_count: vec![0; shard_count],
        }
    }

    fn record(&mut self, shard_idx: usize, loss: f64) {
        if loss.is_finite() {
            self.total_loss += loss;
            self.sample_count += 1;
            if shard_idx < self.per_shard_loss.len() {
                self.per_shard_loss[shard_idx] += loss;
                self.per_shard_count[shard_idx] += 1;
            }
        } else {
            self.non_finite += 1;
        }
    }

    fn avg_loss(&self) -> f64 {
        if self.sample_count > 0 { self.total_loss / self.sample_count as f64 } else { f64::NAN }
    }

    fn perplexity(&self) -> f64 {
        let avg = self.avg_loss();
        if avg.is_finite() { avg.exp() } else { f64::NAN }
    }
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config {
        weights_source: WeightsSource::Synthetic,
        weights_dir: None,
        adapter_path: None,
        model_gb: DEFAULT_MODEL_GB,
        dataset: PathBuf::new(),
        dataset_format: DatasetFormat::Auto,
        text_key: DEFAULT_TEXT_KEY.to_string(),
        shard_mb: 64,
        streams: DEFAULT_STREAMS,
        wave_streams: DEFAULT_WAVE,
        micro_batch: DEFAULT_MICRO_BATCH,
        hidden: DEFAULT_HIDDEN,
        lora_rank: DEFAULT_LORA_RANK,
        lora_alpha: DEFAULT_LORA_ALPHA,
        max_samples: 0,
        split_ratio: 1.0,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => cfg.dataset = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing --dataset"))?),
            "--weights-source" => cfg.weights_source = WeightsSource::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--dataset-format" => cfg.dataset_format = DatasetFormat::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--text-key" => cfg.text_key = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?,
            "--weights-dir" => cfg.weights_dir = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--adapter-path" => cfg.adapter_path = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--model-gb" => cfg.model_gb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--shard-mb" => cfg.shard_mb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--max-samples" => cfg.max_samples = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--split-ratio" => cfg.split_ratio = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: validation_loop.rs --dataset PATH [options]");
                println!("  --weights-source synthetic|directory --weights-dir PATH --model-gb N");
                println!("  Runs a forward-only eval pass over a held-out dataset split.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    if cfg.dataset.as_os_str().is_empty() {
        return Err(anyhow::anyhow!("--dataset is required"));
    }
    if cfg.model_gb == 0 || cfg.shard_mb == 0 || cfg.streams == 0 || cfg.wave_streams == 0 || cfg.micro_batch <= 0 {
        return Err(anyhow::anyhow!("invalid numeric configuration"));
    }
    if let WeightsSource::Directory = cfg.weights_source {
        let Some(dir) = &cfg.weights_dir else {
            return Err(anyhow::anyhow!("--weights-dir is required with --weights-source directory"));
        };
        if !dir.exists() {
            return Err(anyhow::anyhow!("weights directory does not exist: {}", dir.display()));
        }
    }
    Ok(cfg)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    aten_ptx::ensure_libtorch_cuda_loaded();
    if !tch::Cuda::is_available() {
        println!("CUDA not available, exiting");
        return Ok(());
    }

    // Init PTX runtime
    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: cfg.streams,
        quiet_init: 1,
        enable_leak_detection: 0,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, 0.70, cfg.streams).map_err(anyhow::Error::msg)?;

    let device = Device::Cuda(0);
    let active_streams = num_streams();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;
    let inv_sqrt_d = 1.0 / (cfg.hidden as f64).sqrt();

    // Load and split dataset
    let dataset_format_used = match cfg.dataset_format {
        DatasetFormat::Auto => detect_format(&cfg.dataset),
        x => x,
    };
    let all_samples = load_samples(&cfg.dataset, dataset_format_used, &cfg.text_key)?;
    let split_at = ((all_samples.len() as f64) * cfg.split_ratio) as usize;
    let eval_samples = if cfg.split_ratio < 1.0 {
        all_samples[split_at..].to_vec()
    } else {
        all_samples
    };

    let eval_count = if cfg.max_samples > 0 {
        eval_samples.len().min(cfg.max_samples)
    } else {
        eval_samples.len()
    };

    let default_shard_bytes = cfg.shard_mb * 1024 * 1024;
    let shard_bytes_vec: Vec<usize> = match cfg.weights_source {
        WeightsSource::Synthetic => {
            let model_bytes = cfg.model_gb * 1024 * 1024 * 1024usize;
            let shard_count = model_bytes.div_ceil(default_shard_bytes);
            vec![default_shard_bytes; shard_count]
        }
        WeightsSource::Directory => {
            let files = discover_files(cfg.weights_dir.as_ref().expect("validated in parse_args"))?;
            chunk_file_sizes(&files, default_shard_bytes)
        }
    };
    let shards_per_pass = shard_bytes_vec.len();

    let mut checkpoint = None;
    if let Some(path) = &cfg.adapter_path {
        let loaded = load_checkpoint(path, Device::Cpu)?;
        if loaded.hidden != cfg.hidden || loaded.lora_rank != cfg.lora_rank {
            return Err(anyhow::anyhow!(
                "checkpoint hidden/rank mismatch: ckpt ({}/{}) cfg ({}/{})",
                loaded.hidden,
                loaded.lora_rank,
                cfg.hidden,
                cfg.lora_rank
            ));
        }
        checkpoint = Some(loaded);
    }

    println!("=== Ferrite Validation Loop ===\n");
    println!("  weights source:  {}", cfg.weights_source.as_str());
    println!("  model gb:        {}", cfg.model_gb);
    if let Some(wd) = &cfg.weights_dir {
        println!("  weights dir:     {}", wd.display());
    }
    println!("  dataset:         {}", cfg.dataset.display());
    println!("  dataset format:  {}", dataset_format_used.as_str());
    println!("  eval samples:    {}", eval_count);
    println!("  shards/pass:     {}", shards_per_pass);
    println!("  wave streams:    {}", cfg.wave_streams);
    println!("  micro batch:     {}", cfg.micro_batch);
    println!("  hidden:          {}", cfg.hidden);
    println!("  lora rank:       {}", cfg.lora_rank);
    println!(
        "  checkpoint:      {}",
        cfg.adapter_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    let t_eval = Instant::now();
    let mut metrics = EvalMetrics::new(shards_per_pass);
    let mut sample_cursor = 0usize;
    let mut checkpoint_hits = 0usize;
    let mut checkpoint_misses = 0usize;

    tch::no_grad(|| -> Result<()> {
        let mut shard_base = 0usize;
        while shard_base < shards_per_pass {
            let wave = (shards_per_pass - shard_base).min(cfg.wave_streams);

            for local_idx in 0..wave {
                let shard_idx = shard_base + local_idx;
                let stream_id = shard_idx % active_streams;
                set_torch_stream(stream_id);

                let shard_bytes = shard_bytes_vec[shard_idx];
                let elems = (shard_bytes / std::mem::size_of::<f32>()) as i64;
                let rows = (elems / cfg.hidden).max(1);
                let base_w = synthetic_base_weight(rows, cfg.hidden, shard_idx, device, inv_sqrt_d);

                let (lora_a, lora_b) = if let Some(cp) = &checkpoint {
                    if let (Some(a), Some(b)) = (cp.adapters_a.get(shard_idx), cp.adapters_b.get(shard_idx)) {
                        if a.size() == [rows, cfg.lora_rank] && b.size() == [cfg.lora_rank, cfg.hidden] {
                            checkpoint_hits += 1;
                            (a.to_device(device), b.to_device(device))
                        } else {
                            checkpoint_misses += 1;
                            (
                                Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device)),
                                Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device)),
                            )
                        }
                    } else {
                        checkpoint_misses += 1;
                        (
                            Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device)),
                            Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device)),
                        )
                    }
                } else {
                    (
                        Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device)),
                        Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device)),
                    )
                };

                let delta = lora_a.matmul(&lora_b) * lora_scale;
                let effective_w = &base_w + &delta;

                // Build micro-batch from eval samples
                let mut rows_x = Vec::with_capacity(cfg.micro_batch as usize);
                for _ in 0..cfg.micro_batch {
                    let idx = sample_cursor % eval_count;
                    sample_cursor += 1;
                    rows_x.push(encode_text(&eval_samples[idx], cfg.hidden, 0xA11CE).view([1, cfg.hidden]));
                }
                let x = Tensor::cat(&rows_x, 0).to_device(device);

                let y = (x.matmul(&effective_w.transpose(0, 1)) * inv_sqrt_d).tanh();

                let mut rows_t = Vec::with_capacity(cfg.micro_batch as usize);
                for _ in 0..cfg.micro_batch {
                    let idx = sample_cursor % eval_count;
                    sample_cursor += 1;
                    rows_t.push(encode_text(&eval_samples[idx], rows, 0xBEEF).view([1, rows]));
                }
                let target = Tensor::cat(&rows_t, 0).to_device(device).tanh();

                let loss = (&y - &target).square().mean(Kind::Float);
                let lv = f64::try_from(&loss).unwrap_or(f64::NAN);
                metrics.record(shard_idx, lv);
            }

            sync_all_streams();
            reset_torch_stream();
            shard_base += wave;
        }
        Ok(())
    })?;

    tch::Cuda::synchronize(0);
    let wall = t_eval.elapsed().as_secs_f64();

    let s = runtime.tlsf_stats();

    println!("\n  Validation Results:");
    println!("    Avg Loss:       {:.6}", metrics.avg_loss());
    println!("    Perplexity:     {:.4}", metrics.perplexity());
    println!("    Samples:        {}", metrics.sample_count);
    println!("    Non-finite:     {}", metrics.non_finite);
    println!("    Ckpt hits:      {}", checkpoint_hits);
    println!("    Ckpt misses:    {}", checkpoint_misses);
    println!("    Wall time:      {:.2}s", wall);
    println!("    Peak VRAM:      {:.1} MB", s.peak_allocated as f64 / 1e6);
    println!("    Fragmentation:  {:.6}", s.fragmentation_ratio);

    println!("\nRESULT mode=validation");
    println!("RESULT weights_source={}", cfg.weights_source.as_str());
    println!(
        "RESULT checkpoint_path={}",
        cfg.adapter_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    if let Some(cp) = &checkpoint {
        println!("RESULT checkpoint_step={}", cp.step);
        println!("RESULT checkpoint_version={}", cp.version);
        println!("RESULT checkpoint_loss={:.9}", cp.loss);
        println!("RESULT checkpoint_shards={}", cp.shard_count);
        println!("RESULT checkpoint_lr={:.9}", cp.lr);
        println!("RESULT checkpoint_lora_alpha={:.6}", cp.lora_alpha);
    }
    println!("RESULT checkpoint_hits={}", checkpoint_hits);
    println!("RESULT checkpoint_misses={}", checkpoint_misses);
    println!("RESULT avg_loss={:.9}", metrics.avg_loss());
    println!("RESULT perplexity={:.9}", metrics.perplexity());
    println!("RESULT eval_samples={}", metrics.sample_count);
    println!("RESULT non_finite={}", metrics.non_finite);
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT peak_vram_mb={:.6}", s.peak_allocated as f64 / 1e6);
    println!("RESULT fragmentation={:.9}", s.fragmentation_ratio);

    Ok(())
}
