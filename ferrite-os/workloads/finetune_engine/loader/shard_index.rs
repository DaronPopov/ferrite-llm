use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;

#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub tensor_name: String,
    pub file: PathBuf,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub byte_offset: usize,
    pub byte_len: usize,
    pub param_count: u64,
}

#[derive(Clone, Debug)]
pub struct ShardGroup {
    pub group_id: usize,
    pub entries: Vec<IndexEntry>,
    pub total_bytes: usize,
    pub total_params: u64,
}

#[derive(Clone, Debug)]
pub struct ShardIndex {
    pub entries: Vec<IndexEntry>,
    pub groups: Vec<ShardGroup>,
    pub chunk_bytes: usize,
}

impl ShardIndex {
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn total_params(&self) -> u64 {
        self.entries.iter().map(|e| e.param_count).sum()
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.byte_len as u64).sum()
    }

    pub fn largest_group_bytes(&self) -> usize {
        self.groups.iter().map(|g| g.total_bytes).max().unwrap_or(0)
    }
}

fn dtype_element_size(dtype: &str) -> usize {
    match dtype {
        "F32" | "float32" => 4,
        "F16" | "float16" | "BF16" | "bfloat16" => 2,
        "F64" | "float64" => 8,
        "I8" | "int8" | "U8" | "uint8" => 1,
        _ => 4,
    }
}

fn scan_safetensors_entries(path: &Path) -> Result<Vec<IndexEntry>> {
    let data = fs::read(path)?;
    if data.len() < 8 {
        return Err(anyhow::anyhow!("file too small: {}", path.display()));
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into()?) as usize;
    if data.len() < 8 + header_len {
        return Err(anyhow::anyhow!("truncated header in {}", path.display()));
    }

    let header_str = std::str::from_utf8(&data[8..8 + header_len])?;
    let data_start = 8 + header_len;

    let mut entries = Vec::new();
    // Lightweight scan of the JSON header for tensor entries.
    // Each entry: "name": {"dtype": "...", "shape": [...], "data_offsets": [start, end]}
    let mut i = 0;
    let bytes = header_str.as_bytes();

    while i < bytes.len() {
        // Find key
        let key_start = match find_string(bytes, i) {
            Some((s, e)) => { i = e; (s, e) }
            None => break,
        };
        let key = &header_str[key_start.0..key_start.1];
        if key == "__metadata__" {
            i = skip_object(bytes, i);
            continue;
        }

        // Skip to value object
        while i < bytes.len() && bytes[i] != b'{' { i += 1; }
        let obj_start = i;
        let obj_text = extract_object(bytes, &mut i);

        let dtype = extract_field_string(&obj_text, "dtype").unwrap_or_default();
        let shape = extract_field_int_array(&obj_text, "shape");
        let offsets = extract_field_int_array(&obj_text, "data_offsets");

        if offsets.len() == 2 {
            let start = offsets[0] as usize;
            let end = offsets[1] as usize;
            let params: u64 = shape.iter().map(|&d| d as u64).product();

            entries.push(IndexEntry {
                tensor_name: key.to_string(),
                file: path.to_path_buf(),
                dtype,
                shape,
                byte_offset: data_start + start,
                byte_len: end - start,
                param_count: params,
            });
        }
    }

    entries.sort_by(|a, b| a.byte_offset.cmp(&b.byte_offset));
    Ok(entries)
}

fn find_string(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let open = bytes[start..].iter().position(|&b| b == b'"')? + start;
    let mut i = open + 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' { i += 2; continue; }
        if bytes[i] == b'"' { return Some((open + 1, i)); }
        i += 1;
    }
    None
}

fn skip_object(bytes: &[u8], start: usize) -> usize {
    let mut depth = 0;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth <= 0 { return i + 1; }
            }
            _ => {}
        }
        i += 1;
    }
    bytes.len()
}

fn extract_object(bytes: &[u8], pos: &mut usize) -> String {
    let start = *pos;
    *pos = skip_object(bytes, start);
    String::from_utf8_lossy(&bytes[start..*pos]).to_string()
}

fn extract_field_string(obj: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\"", field);
    let idx = obj.find(&needle)?;
    let after = &obj[idx + needle.len()..];
    let colon = after.find(':')?;
    let after = &after[colon + 1..];
    let open = after.find('"')?;
    let after = &after[open + 1..];
    let close = after.find('"')?;
    Some(after[..close].to_string())
}

fn extract_field_int_array(obj: &str, field: &str) -> Vec<i64> {
    let needle = format!("\"{}\"", field);
    let Some(idx) = obj.find(&needle) else { return Vec::new() };
    let after = &obj[idx + needle.len()..];
    let Some(open) = after.find('[') else { return Vec::new() };
    let after = &after[open + 1..];
    let Some(close) = after.find(']') else { return Vec::new() };
    let inner = &after[..close];
    inner
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect()
}

pub fn build_index(dir: &Path, chunk_bytes: usize) -> Result<ShardIndex> {
    let mut all_entries = Vec::new();

    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    paths.sort();

    for path in &paths {
        let entries = scan_safetensors_entries(path)?;
        all_entries.extend(entries);
    }

    // Group entries into shard groups by chunk_bytes budget
    let mut groups = Vec::new();
    let mut current_entries = Vec::new();
    let mut current_bytes = 0usize;
    let mut group_id = 0usize;

    for entry in &all_entries {
        if current_bytes + entry.byte_len > chunk_bytes && !current_entries.is_empty() {
            let total_params = current_entries.iter().map(|e: &IndexEntry| e.param_count).sum();
            groups.push(ShardGroup {
                group_id,
                entries: current_entries.clone(),
                total_bytes: current_bytes,
                total_params,
            });
            current_entries.clear();
            current_bytes = 0;
            group_id += 1;
        }
        current_entries.push(entry.clone());
        current_bytes += entry.byte_len;
    }

    if !current_entries.is_empty() {
        let total_params = current_entries.iter().map(|e: &IndexEntry| e.param_count).sum();
        groups.push(ShardGroup {
            group_id,
            entries: current_entries,
            total_bytes: current_bytes,
            total_params,
        });
    }

    Ok(ShardIndex {
        entries: all_entries,
        groups,
        chunk_bytes,
    })
}

pub fn write_index_manifest(index: &ShardIndex, path: &Path) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "# Ferrite Shard Index")?;
    writeln!(file, "# chunk_bytes={}", index.chunk_bytes)?;
    writeln!(file, "# groups={}", index.groups.len())?;
    writeln!(file, "# total_params={}", index.total_params())?;
    writeln!(file, "# total_bytes={}", index.total_bytes())?;
    writeln!(file)?;

    for group in &index.groups {
        writeln!(file, "[group.{}]", group.group_id)?;
        writeln!(file, "total_bytes = {}", group.total_bytes)?;
        writeln!(file, "total_params = {}", group.total_params)?;
        writeln!(file, "tensors = {}", group.entries.len())?;
        for e in &group.entries {
            writeln!(file, "  {} dtype={} shape={:?} bytes={}",
                e.tensor_name, e.dtype, e.shape, e.byte_len)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

fn parse_args() -> Result<(PathBuf, usize, Option<PathBuf>)> {
    let mut dir = None;
    let mut chunk_mb = 64usize;
    let mut output = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => dir = Some(PathBuf::from(
                args.next().ok_or_else(|| anyhow::anyhow!("missing --dir"))?,
            )),
            "--chunk-mb" => chunk_mb = args.next()
                .ok_or_else(|| anyhow::anyhow!("missing --chunk-mb"))?
                .parse()?,
            "--output" => output = Some(PathBuf::from(
                args.next().ok_or_else(|| anyhow::anyhow!("missing --output"))?,
            )),
            "-h" | "--help" => {
                println!("Usage: shard_index.rs --dir MODEL_DIR [--chunk-mb 64] [--output INDEX_FILE]");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    let dir = dir.ok_or_else(|| anyhow::anyhow!("--dir is required"))?;
    Ok((dir, chunk_mb, output))
}

fn main() -> Result<()> {
    let (dir, chunk_mb, output) = parse_args()?;
    let chunk_bytes = chunk_mb * 1024 * 1024;

    println!("[shard-index] scanning: {}", dir.display());
    println!("[shard-index] chunk size: {} MB", chunk_mb);

    let index = build_index(&dir, chunk_bytes)?;

    println!("[shard-index] total tensors:   {}", index.entries.len());
    println!("[shard-index] total params:    {}", index.total_params());
    println!("[shard-index] total bytes:     {} ({:.2} GB)",
        index.total_bytes(), index.total_bytes() as f64 / 1e9);
    println!("[shard-index] shard groups:    {}", index.group_count());
    println!("[shard-index] largest group:   {:.1} MB",
        index.largest_group_bytes() as f64 / 1e6);

    for g in &index.groups {
        println!("  group {:>3}: {:.1} MB, {} tensors, {} params",
            g.group_id,
            g.total_bytes as f64 / 1e6,
            g.entries.len(),
            g.total_params);
    }

    if let Some(out) = output {
        write_index_manifest(&index, &out)?;
        println!("[shard-index] manifest written to: {}", out.display());
    }

    println!("\nRESULT tensor_count={}", index.entries.len());
    println!("RESULT total_params={}", index.total_params());
    println!("RESULT total_bytes={}", index.total_bytes());
    println!("RESULT group_count={}", index.group_count());
    println!("RESULT largest_group_mb={:.6}", index.largest_group_bytes() as f64 / 1e6);

    Ok(())
}
