use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tch::{Device, Kind, Tensor};

const MAGIC: &[u8; 8] = b"FRTA_CKP";
const VERSION: u32 = 2;
const MIN_SUPPORTED_VERSION: u32 = 1;
const CHECKPOINT_HEADER_BYTES: usize = 80;

#[derive(Debug)]
pub struct AdapterCheckpoint {
    pub step: usize,
    pub loss: f64,
    pub lora_rank: i64,
    pub lora_alpha: f64,
    pub lr: f64,
    pub hidden: i64,
    pub shard_count: usize,
    pub adapters_a: Vec<Tensor>,
    pub adapters_b: Vec<Tensor>,
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
    timestamp: u64,
    tensor_count: u32,
}

impl CheckpointHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(CHECKPOINT_HEADER_BYTES);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.step.to_le_bytes());
        buf.extend_from_slice(&self.loss_bits.to_le_bytes());
        buf.extend_from_slice(&self.lora_rank.to_le_bytes());
        buf.extend_from_slice(&self.hidden.to_le_bytes());
        buf.extend_from_slice(&self.shard_count.to_le_bytes());
        buf.extend_from_slice(&self.lora_alpha_bits.to_le_bytes());
        buf.extend_from_slice(&self.lr_bits.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.tensor_count.to_le_bytes());
        // pad to 80 bytes
        buf.resize(CHECKPOINT_HEADER_BYTES, 0);
        buf
    }

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
            timestamp: u64::from_le_bytes(data[56..64].try_into()?),
            tensor_count: u32::from_le_bytes(data[64..68].try_into()?),
        })
    }
}

#[derive(Clone, Debug)]
struct TensorMeta {
    name_len: u16,
    name: String,
    ndim: u8,
    dims: Vec<i64>,
    dtype: u8, // 0=f32, 1=f16, 2=bf16
    data_bytes: u64,
}

impl TensorMeta {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.name_len.to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());
        buf.push(self.ndim);
        for &d in &self.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(self.dtype);
        buf.extend_from_slice(&self.data_bytes.to_le_bytes());
        buf
    }

    fn from_reader(data: &[u8], offset: &mut usize) -> Result<Self> {
        let name_len = u16::from_le_bytes(read_chunk(data, offset, 2)?.try_into()?);
        let name = String::from_utf8(read_chunk(data, offset, name_len as usize)?.to_vec())?;
        let ndim = read_chunk(data, offset, 1)?[0];
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(i64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?));
        }
        let dtype = read_chunk(data, offset, 1)?[0];
        let data_bytes = u64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?);
        Ok(Self { name_len, name, ndim, dims, dtype, data_bytes })
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

pub fn save_checkpoint(ckpt: &AdapterCheckpoint, path: &PathBuf) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let tensor_count = (ckpt.adapters_a.len() + ckpt.adapters_b.len()) as u32;

    let header = CheckpointHeader {
        version: VERSION,
        step: ckpt.step as u64,
        loss_bits: ckpt.loss.to_bits(),
        lora_rank: ckpt.lora_rank as u32,
        hidden: ckpt.hidden as u32,
        shard_count: ckpt.shard_count as u32,
        lora_alpha_bits: ckpt.lora_alpha.to_bits(),
        lr_bits: ckpt.lr.to_bits(),
        timestamp,
        tensor_count,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(&header.to_bytes())?;

    let cpu = Device::Cpu;

    for (i, tensor) in ckpt.adapters_a.iter().enumerate() {
        let t = tensor.to_device(cpu).to_kind(Kind::Float);
        let size = t.size();
        let numel: i64 = size.iter().product();
        let data_bytes = (numel as usize) * std::mem::size_of::<f32>();
        let name = format!("shard_{i}_lora_a");

        let meta = TensorMeta {
            name_len: name.len() as u16,
            name,
            ndim: size.len() as u8,
            dims: size,
            dtype: 0,
            data_bytes: data_bytes as u64,
        };
        writer.write_all(&meta.to_bytes())?;

        let flat = Vec::<f32>::try_from(t.flatten(0, -1))?;
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(flat.as_ptr() as *const u8, data_bytes)
        };
        writer.write_all(raw)?;
    }

    for (i, tensor) in ckpt.adapters_b.iter().enumerate() {
        let t = tensor.to_device(cpu).to_kind(Kind::Float);
        let size = t.size();
        let numel: i64 = size.iter().product();
        let data_bytes = (numel as usize) * std::mem::size_of::<f32>();
        let name = format!("shard_{i}_lora_b");

        let meta = TensorMeta {
            name_len: name.len() as u16,
            name,
            ndim: size.len() as u8,
            dims: size,
            dtype: 0,
            data_bytes: data_bytes as u64,
        };
        writer.write_all(&meta.to_bytes())?;

        let flat = Vec::<f32>::try_from(t.flatten(0, -1))?;
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(flat.as_ptr() as *const u8, data_bytes)
        };
        writer.write_all(raw)?;
    }

    writer.flush()?;

    println!("[checkpoint] saved step={} loss={:.6} tensors={} path={}",
        ckpt.step, ckpt.loss, tensor_count, path.display());
    println!("RESULT checkpoint_path={}", path.display());
    println!("RESULT checkpoint_step={}", ckpt.step);
    println!("RESULT checkpoint_tensors={}", tensor_count);
    println!("RESULT checkpoint_timestamp={}", timestamp);

    Ok(())
}

pub fn load_checkpoint(path: &PathBuf, device: Device) -> Result<AdapterCheckpoint> {
    let data = fs::read(path)?;
    let header = CheckpointHeader::from_bytes(&data)?;

    if header.version < MIN_SUPPORTED_VERSION || header.version > VERSION {
        return Err(anyhow::anyhow!(
            "checkpoint version mismatch: file={} supported={}..={}",
            header.version, MIN_SUPPORTED_VERSION, VERSION
        ));
    }

    let loss = f64::from_bits(header.loss_bits);
    let lora_alpha = f64::from_bits(header.lora_alpha_bits);
    let lr = f64::from_bits(header.lr_bits);

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

    println!("[checkpoint] loaded step={} loss={:.6} a_count={} b_count={} path={}",
        header.step, loss, adapters_a.len(), adapters_b.len(), path.display());

    Ok(AdapterCheckpoint {
        step: header.step as usize,
        loss,
        lora_rank: header.lora_rank as i64,
        lora_alpha,
        lr,
        hidden: header.hidden as i64,
        shard_count: header.shard_count as usize,
        adapters_a,
        adapters_b,
    })
}

fn parse_args() -> Result<(String, PathBuf)> {
    let mut mode = String::new();
    let mut path = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => mode = args.next().ok_or_else(|| anyhow::anyhow!("missing --mode"))?,
            "--path" => path = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing --path"))?)),
            "-h" | "--help" => {
                println!("Usage: save_adapter.rs --mode save|load|info --path CHECKPOINT_PATH");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    let path = path.ok_or_else(|| anyhow::anyhow!("--path is required"))?;
    Ok((mode, path))
}

fn main() -> Result<()> {
    let (mode, path) = parse_args()?;

    match mode.as_str() {
        "info" => {
            let data = fs::read(&path)?;
            let header = CheckpointHeader::from_bytes(&data)?;
            println!("Checkpoint: {}", path.display());
            println!("  version:     {}", header.version);
            println!("  step:        {}", header.step);
            println!("  loss:        {:.6}", f64::from_bits(header.loss_bits));
            println!("  lora_rank:   {}", header.lora_rank);
            println!("  hidden:      {}", header.hidden);
            println!("  shard_count: {}", header.shard_count);
            println!("  lora_alpha:  {:.3}", f64::from_bits(header.lora_alpha_bits));
            println!("  lr:          {:.9}", f64::from_bits(header.lr_bits));
            println!("  timestamp:   {}", header.timestamp);
            println!("  tensors:     {}", header.tensor_count);
        }
        "save" => {
            println!("[checkpoint] save mode: use save_checkpoint() from training loop");
            println!("[checkpoint] this binary validates the format by round-tripping synthetic data");

            let device = Device::Cpu;
            let rank = 16i64;
            let hidden = 2048i64;
            let shards = 4usize;
            let rows = 512i64;

            let mut a_list = Vec::new();
            let mut b_list = Vec::new();
            for _ in 0..shards {
                a_list.push(Tensor::randn([rows, rank], (Kind::Float, device)));
                b_list.push(Tensor::randn([rank, hidden], (Kind::Float, device)));
            }

            let ckpt = AdapterCheckpoint {
                step: 100,
                loss: 0.042,
                lora_rank: rank,
                lora_alpha: 32.0,
                lr: 1e-3,
                hidden,
                shard_count: shards,
                adapters_a: a_list,
                adapters_b: b_list,
            };

            save_checkpoint(&ckpt, &path)?;
            let loaded = load_checkpoint(&path, device)?;
            assert_eq!(loaded.step, 100);
            assert_eq!(loaded.adapters_a.len(), shards);
            assert_eq!(loaded.adapters_b.len(), shards);
            println!("[checkpoint] round-trip validation passed");
        }
        "load" => {
            aten_ptx::ensure_libtorch_cuda_loaded();
            let device = if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            };
            let ckpt = load_checkpoint(&path, device)?;
            println!("RESULT restored_step={}", ckpt.step);
            println!("RESULT restored_loss={:.9}", ckpt.loss);
            println!("RESULT restored_shards={}", ckpt.shard_count);
        }
        _ => return Err(anyhow::anyhow!("--mode must be save|load|info")),
    }

    Ok(())
}
