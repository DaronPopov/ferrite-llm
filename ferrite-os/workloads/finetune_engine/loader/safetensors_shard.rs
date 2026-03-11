use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Result;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub data_offset: usize,
    pub data_len: usize,
}

#[derive(Clone, Debug)]
pub struct ShardManifest {
    pub path: PathBuf,
    pub tensors: Vec<TensorInfo>,
    pub file_size: usize,
}

#[derive(Clone, Debug)]
pub struct ShardPlan {
    pub manifests: Vec<ShardManifest>,
    pub total_params: u64,
    pub total_bytes: u64,
}

fn parse_safetensors_header(data: &[u8]) -> Result<(usize, Vec<TensorInfo>)> {
    if data.len() < 8 {
        return Err(anyhow::anyhow!("file too small for safetensors header"));
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into()?) as usize;
    if data.len() < 8 + header_len {
        return Err(anyhow::anyhow!(
            "file truncated: need {} bytes for header, have {}",
            8 + header_len,
            data.len()
        ));
    }

    let header_str = std::str::from_utf8(&data[8..8 + header_len])?;
    let data_start = 8 + header_len;

    let mut tensors = Vec::new();
    let mut pos = 0;
    let bytes = header_str.as_bytes();

    // Minimal JSON parser for safetensors header format:
    // {"tensor_name": {"dtype": "F32", "shape": [dim, ...], "data_offsets": [start, end]}, ...}
    while pos < bytes.len() {
        let key_start = match find_next_string(bytes, pos) {
            Some((s, e)) => { pos = e; (s, e) }
            None => break,
        };

        let key = &header_str[key_start.0..key_start.1];

        if key == "__metadata__" {
            pos = skip_value(bytes, pos);
            continue;
        }

        let (dtype, shape, offsets) = parse_tensor_entry(header_str, bytes, &mut pos)?;

        if offsets.len() == 2 {
            let start = offsets[0] as usize;
            let end = offsets[1] as usize;
            tensors.push(TensorInfo {
                name: key.to_string(),
                dtype,
                shape,
                data_offset: data_start + start,
                data_len: end - start,
            });
        }

        pos = skip_past(bytes, pos, b',').unwrap_or(bytes.len());
    }

    tensors.sort_by(|a, b| a.data_offset.cmp(&b.data_offset));
    Ok((data_start, tensors))
}

fn find_next_string(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let open = bytes[start..].iter().position(|&b| b == b'"')? + start;
    let mut i = open + 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            return Some((open + 1, i));
        }
        i += 1;
    }
    None
}

fn skip_value(bytes: &[u8], start: usize) -> usize {
    let mut depth = 0i32;
    let mut i = start;
    let mut in_string = false;
    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_string => in_string = true,
            b'"' if in_string => {
                if i > 0 && bytes[i - 1] != b'\\' {
                    in_string = false;
                }
            }
            b'{' | b'[' if !in_string => depth += 1,
            b'}' | b']' if !in_string => {
                depth -= 1;
                if depth <= 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    bytes.len()
}

fn skip_past(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
    bytes[start..].iter().position(|&b| b == target).map(|p| start + p + 1)
}

fn parse_tensor_entry(
    header: &str,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(String, Vec<i64>, Vec<u64>)> {
    // Find the opening { of the tensor's value object
    while *pos < bytes.len() && bytes[*pos] != b'{' {
        *pos += 1;
    }
    *pos += 1;

    let mut dtype = String::new();
    let mut shape = Vec::new();
    let mut offsets = Vec::new();

    // Parse key-value pairs inside the tensor object
    let mut depth = 1i32;
    while *pos < bytes.len() && depth > 0 {
        // Find next key
        if let Some((s, e)) = find_next_string(bytes, *pos) {
            let key = &header[s..e];
            *pos = e + 1;

            // Skip to colon
            while *pos < bytes.len() && bytes[*pos] != b':' {
                *pos += 1;
            }
            *pos += 1;

            match key {
                "dtype" => {
                    if let Some((s, e)) = find_next_string(bytes, *pos) {
                        dtype = header[s..e].to_string();
                        *pos = e + 1;
                    }
                }
                "shape" => {
                    shape = parse_int_array(bytes, pos);
                }
                "data_offsets" => {
                    offsets = parse_uint_array(bytes, pos);
                }
                _ => {
                    *pos = skip_value(bytes, *pos);
                }
            }
        }

        // Check for closing brace
        while *pos < bytes.len() {
            if bytes[*pos] == b'}' {
                depth -= 1;
                *pos += 1;
                break;
            }
            if bytes[*pos] == b',' {
                *pos += 1;
                break;
            }
            *pos += 1;
        }
    }

    Ok((dtype, shape, offsets))
}

fn parse_int_array(bytes: &[u8], pos: &mut usize) -> Vec<i64> {
    let mut out = Vec::new();
    while *pos < bytes.len() && bytes[*pos] != b'[' {
        *pos += 1;
    }
    *pos += 1;
    let mut num_buf = String::new();
    while *pos < bytes.len() {
        match bytes[*pos] {
            b']' => {
                if !num_buf.is_empty() {
                    if let Ok(n) = num_buf.trim().parse::<i64>() {
                        out.push(n);
                    }
                }
                *pos += 1;
                break;
            }
            b',' => {
                if !num_buf.is_empty() {
                    if let Ok(n) = num_buf.trim().parse::<i64>() {
                        out.push(n);
                    }
                    num_buf.clear();
                }
            }
            c if (c as char).is_ascii_digit() || c == b'-' => {
                num_buf.push(c as char);
            }
            _ => {}
        }
        *pos += 1;
    }
    out
}

fn parse_uint_array(bytes: &[u8], pos: &mut usize) -> Vec<u64> {
    let ints = parse_int_array(bytes, pos);
    ints.into_iter().map(|v| v as u64).collect()
}

fn dtype_to_kind(dtype: &str) -> Kind {
    match dtype {
        "F32" | "float32" => Kind::Float,
        "F16" | "float16" => Kind::Half,
        "BF16" | "bfloat16" => Kind::BFloat16,
        "F64" | "float64" => Kind::Double,
        "I32" | "int32" => Kind::Int,
        "I64" | "int64" => Kind::Int64,
        _ => Kind::Float,
    }
}

fn dtype_element_size(dtype: &str) -> usize {
    match dtype {
        "F32" | "float32" | "I32" | "int32" => 4,
        "F16" | "float16" | "BF16" | "bfloat16" => 2,
        "F64" | "float64" | "I64" | "int64" => 8,
        "I8" | "int8" | "U8" | "uint8" => 1,
        _ => 4,
    }
}

pub fn scan_safetensors_dir(dir: &Path) -> Result<ShardPlan> {
    let mut manifests = Vec::new();
    let mut total_params = 0u64;
    let mut total_bytes = 0u64;

    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    entries.sort_by(|a, b| a.path().cmp(&b.path()));

    for entry in entries {
        let path = entry.path();
        let data = fs::read(&path)?;
        let file_size = data.len();
        let (_, tensors) = parse_safetensors_header(&data)?;

        for t in &tensors {
            let params: u64 = t.shape.iter().map(|&d| d as u64).product();
            total_params += params;
            total_bytes += t.data_len as u64;
        }

        manifests.push(ShardManifest {
            path,
            tensors,
            file_size,
        });
    }

    Ok(ShardPlan {
        manifests,
        total_params,
        total_bytes,
    })
}

pub fn load_tensor_from_file(
    file_data: &[u8],
    info: &TensorInfo,
    device: Device,
    upcast_to_f32: bool,
) -> Result<Tensor> {
    let kind = dtype_to_kind(&info.dtype);
    let raw = &file_data[info.data_offset..info.data_offset + info.data_len];
    let elem_size = dtype_element_size(&info.dtype);
    let numel = info.data_len / elem_size;

    let tensor = unsafe {
        Tensor::from_blob(
            raw.as_ptr(),
            &info.shape,
            &[], // strides auto
            kind,
            Device::Cpu,
        )
    };

    let tensor = tensor.to_device(device);

    if upcast_to_f32 && kind != Kind::Float {
        Ok(tensor.to_kind(Kind::Float))
    } else {
        Ok(tensor)
    }
}

pub fn stream_shard_tensors<F>(
    plan: &ShardPlan,
    device: Device,
    upcast: bool,
    mut callback: F,
) -> Result<()>
where
    F: FnMut(usize, usize, &str, Tensor) -> Result<()>,
{
    for (shard_idx, manifest) in plan.manifests.iter().enumerate() {
        let data = fs::read(&manifest.path)?;
        for (tensor_idx, info) in manifest.tensors.iter().enumerate() {
            let tensor = load_tensor_from_file(&data, info, device, upcast)?;
            callback(shard_idx, tensor_idx, &info.name, tensor)?;
        }
    }
    Ok(())
}

fn parse_args() -> Result<(PathBuf, bool)> {
    let mut dir = None;
    let mut verbose = false;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => dir = Some(PathBuf::from(
                args.next().ok_or_else(|| anyhow::anyhow!("missing --dir"))?,
            )),
            "--verbose" => verbose = true,
            "-h" | "--help" => {
                println!("Usage: safetensors_shard.rs --dir MODEL_DIR [--verbose]");
                println!("  Scans a directory of .safetensors files and prints the shard plan.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    let dir = dir.ok_or_else(|| anyhow::anyhow!("--dir is required"))?;
    Ok((dir, verbose))
}

fn main() -> Result<()> {
    let (dir, verbose) = parse_args()?;

    println!("[loader] scanning: {}", dir.display());
    let plan = scan_safetensors_dir(&dir)?;

    println!("[loader] found {} shard files", plan.manifests.len());
    println!("[loader] total params:  {}", plan.total_params);
    println!("[loader] total bytes:   {} ({:.2} GB)",
        plan.total_bytes, plan.total_bytes as f64 / 1e9);

    if verbose {
        for (i, m) in plan.manifests.iter().enumerate() {
            println!("\n  shard {i}: {} ({:.1} MB, {} tensors)",
                m.path.display(),
                m.file_size as f64 / 1e6,
                m.tensors.len());
            for t in &m.tensors {
                println!("    {} dtype={} shape={:?} bytes={}",
                    t.name, t.dtype, t.shape, t.data_len);
            }
        }
    }

    println!("\nRESULT shard_files={}", plan.manifests.len());
    println!("RESULT total_params={}", plan.total_params);
    println!("RESULT total_bytes={}", plan.total_bytes);

    Ok(())
}
