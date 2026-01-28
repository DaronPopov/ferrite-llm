//! Model checkpointing utilities
//!
//! Provides save/load functionality for:
//! - Individual Grids (weights, biases)
//! - Complete model state (all parameters)
//! - Optimizer state (momentum, Adam moments)
//! - Training state (epoch, step, loss history)

use super::{Grid, Values};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

/// Magic bytes for checkpoint file format
const CHECKPOINT_MAGIC: &[u8; 4] = b"SCPT"; // Semantic Core PyTorch
const CHECKPOINT_VERSION: u32 = 1;

/// A complete model checkpoint
#[derive(Debug)]
pub struct Checkpoint {
    /// Model parameters (weights, biases)
    pub parameters: HashMap<String, Grid>,
    /// Optimizer state (momentum, Adam m/v)
    pub optimizer_state: HashMap<String, Grid>,
    /// Training metadata
    pub metadata: CheckpointMetadata,
}

/// Training metadata stored in checkpoint
#[derive(Debug, Clone, Default)]
pub struct CheckpointMetadata {
    pub epoch: u32,
    pub global_step: u64,
    pub best_loss: f32,
    pub learning_rate: f32,
    pub extra: HashMap<String, String>,
}

/// Errors during checkpoint operations
#[derive(Debug)]
pub enum CheckpointError {
    IoError(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    CorruptedData(String),
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::IoError(e)
    }
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::IoError(e) => write!(f, "IO error: {}", e),
            CheckpointError::InvalidMagic => write!(f, "Invalid checkpoint magic bytes"),
            CheckpointError::UnsupportedVersion(v) => write!(f, "Unsupported checkpoint version: {}", v),
            CheckpointError::CorruptedData(msg) => write!(f, "Corrupted checkpoint: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl Checkpoint {
    /// Create a new empty checkpoint
    pub fn new() -> Self {
        Checkpoint {
            parameters: HashMap::new(),
            optimizer_state: HashMap::new(),
            metadata: CheckpointMetadata::default(),
        }
    }

    /// Add a parameter (weight/bias) to the checkpoint
    pub fn add_parameter(&mut self, name: &str, grid: Grid) {
        // Ensure grid is on host before saving
        let host_grid = match &grid.values {
            Values::Host(_) => grid,
            Values::Device { .. } => {
                panic!("Cannot checkpoint GPU tensor directly - move to host first")
            }
        };
        self.parameters.insert(name.to_string(), host_grid);
    }

    /// Add optimizer state (momentum, Adam m/v) to the checkpoint
    pub fn add_optimizer_state(&mut self, name: &str, grid: Grid) {
        let host_grid = match &grid.values {
            Values::Host(_) => grid,
            Values::Device { .. } => {
                panic!("Cannot checkpoint GPU tensor directly - move to host first")
            }
        };
        self.optimizer_state.insert(name.to_string(), host_grid);
    }

    /// Get a parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&Grid> {
        self.parameters.get(name)
    }

    /// Get optimizer state by name
    pub fn get_optimizer_state(&self, name: &str) -> Option<&Grid> {
        self.optimizer_state.get(name)
    }

    /// Save checkpoint to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), CheckpointError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(CHECKPOINT_MAGIC)?;
        writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;

        // Write metadata
        write_metadata(&mut writer, &self.metadata)?;

        // Write parameters
        write_u32(&mut writer, self.parameters.len() as u32)?;
        for (name, grid) in &self.parameters {
            write_string(&mut writer, name)?;
            write_grid(&mut writer, grid)?;
        }

        // Write optimizer state
        write_u32(&mut writer, self.optimizer_state.len() as u32)?;
        for (name, grid) in &self.optimizer_state {
            write_string(&mut writer, name)?;
            write_grid(&mut writer, grid)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(CheckpointError::InvalidMagic);
        }

        let version = read_u32(&mut reader)?;
        if version > CHECKPOINT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }

        // Read metadata
        let metadata = read_metadata(&mut reader)?;

        // Read parameters
        let num_params = read_u32(&mut reader)?;
        let mut parameters = HashMap::new();
        for _ in 0..num_params {
            let name = read_string(&mut reader)?;
            let grid = read_grid(&mut reader)?;
            parameters.insert(name, grid);
        }

        // Read optimizer state
        let num_opt = read_u32(&mut reader)?;
        let mut optimizer_state = HashMap::new();
        for _ in 0..num_opt {
            let name = read_string(&mut reader)?;
            let grid = read_grid(&mut reader)?;
            optimizer_state.insert(name, grid);
        }

        Ok(Checkpoint {
            parameters,
            optimizer_state,
            metadata,
        })
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Serialization helpers
// ============================================================================

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), std::io::Error> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, std::io::Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<(), std::io::Error> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, std::io::Error> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<(), std::io::Error> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, std::io::Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<(), std::io::Error> {
    let bytes = s.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, CheckpointError> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| CheckpointError::CorruptedData("Invalid UTF-8 string".to_string()))
}

fn write_metadata<W: Write>(w: &mut W, meta: &CheckpointMetadata) -> Result<(), std::io::Error> {
    write_u32(w, meta.epoch)?;
    write_u64(w, meta.global_step)?;
    write_f32(w, meta.best_loss)?;
    write_f32(w, meta.learning_rate)?;

    // Write extra metadata
    write_u32(w, meta.extra.len() as u32)?;
    for (k, v) in &meta.extra {
        write_string(w, k)?;
        write_string(w, v)?;
    }

    Ok(())
}

fn read_metadata<R: Read>(r: &mut R) -> Result<CheckpointMetadata, CheckpointError> {
    let epoch = read_u32(r)?;
    let global_step = read_u64(r)?;
    let best_loss = read_f32(r)?;
    let learning_rate = read_f32(r)?;

    let num_extra = read_u32(r)?;
    let mut extra = HashMap::new();
    for _ in 0..num_extra {
        let k = read_string(r)?;
        let v = read_string(r)?;
        extra.insert(k, v);
    }

    Ok(CheckpointMetadata {
        epoch,
        global_step,
        best_loss,
        learning_rate,
        extra,
    })
}

fn write_grid<W: Write>(w: &mut W, grid: &Grid) -> Result<(), std::io::Error> {
    // Write shape
    write_u32(w, grid.shape.len() as u32)?;
    for &dim in &grid.shape {
        write_u32(w, dim as u32)?;
    }

    // Write data
    if let Values::Host(data) = &grid.values {
        write_u32(w, data.len() as u32)?;
        for &val in data {
            write_f32(w, val)?;
        }
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Cannot serialize GPU tensor",
        ));
    }

    Ok(())
}

fn read_grid<R: Read>(r: &mut R) -> Result<Grid, CheckpointError> {
    // Read shape
    let ndim = read_u32(r)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(read_u32(r)? as usize);
    }

    // Read data
    let len = read_u32(r)? as usize;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f32(r)?);
    }

    // Validate
    let expected_len: usize = shape.iter().product();
    if data.len() != expected_len {
        return Err(CheckpointError::CorruptedData(
            format!("Shape {:?} expects {} elements, got {}", shape, expected_len, data.len())
        ));
    }

    Ok(Grid::new(data, shape))
}

/// Save a single Grid to a binary file
pub fn save_grid<P: AsRef<Path>>(grid: &Grid, path: P) -> Result<(), CheckpointError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"GRID")?;
    write_grid(&mut writer, grid)?;
    writer.flush()?;

    Ok(())
}

/// Load a single Grid from a binary file
pub fn load_grid<P: AsRef<Path>>(path: P) -> Result<Grid, CheckpointError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GRID" {
        return Err(CheckpointError::InvalidMagic);
    }

    read_grid(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_read_u32() {
        let mut buf = Cursor::new(Vec::new());
        write_u32(&mut buf, 12345).unwrap();
        buf.set_position(0);
        assert_eq!(read_u32(&mut buf).unwrap(), 12345);
    }

    #[test]
    fn test_write_read_f32() {
        let mut buf = Cursor::new(Vec::new());
        write_f32(&mut buf, 3.14159).unwrap();
        buf.set_position(0);
        let read_val = read_f32(&mut buf).unwrap();
        assert!((read_val - 3.14159).abs() < 1e-6);
    }

    #[test]
    fn test_write_read_string() {
        let mut buf = Cursor::new(Vec::new());
        write_string(&mut buf, "hello_world").unwrap();
        buf.set_position(0);
        assert_eq!(read_string(&mut buf).unwrap(), "hello_world");
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ckpt = Checkpoint::new();

        // Add some parameters
        let w1 = Grid::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b1 = Grid::new(vec![0.1, 0.2], vec![2]);
        ckpt.add_parameter("layer1.weight", w1);
        ckpt.add_parameter("layer1.bias", b1);

        // Add optimizer state
        let m1 = Grid::new(vec![0.0; 4], vec![2, 2]);
        ckpt.add_optimizer_state("layer1.weight.m", m1);

        // Set metadata
        ckpt.metadata.epoch = 5;
        ckpt.metadata.global_step = 1000;
        ckpt.metadata.best_loss = 0.05;
        ckpt.metadata.learning_rate = 0.001;
        ckpt.metadata.extra.insert("model_name".to_string(), "test_model".to_string());

        // Save to temp file
        let tmp_path = "/tmp/test_checkpoint.scpt";
        ckpt.save(tmp_path).unwrap();

        // Load back
        let loaded = Checkpoint::load(tmp_path).unwrap();

        // Verify parameters
        let loaded_w1 = loaded.get_parameter("layer1.weight").unwrap();
        if let Values::Host(data) = &loaded_w1.values {
            assert_eq!(data, &vec![1.0, 2.0, 3.0, 4.0]);
        }
        assert_eq!(loaded_w1.shape, vec![2, 2]);

        // Verify metadata
        assert_eq!(loaded.metadata.epoch, 5);
        assert_eq!(loaded.metadata.global_step, 1000);
        assert!((loaded.metadata.best_loss - 0.05).abs() < 1e-6);
        assert_eq!(loaded.metadata.extra.get("model_name"), Some(&"test_model".to_string()));

        // Cleanup
        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn test_grid_save_load() {
        let grid = Grid::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let tmp_path = "/tmp/test_grid.bin";
        save_grid(&grid, tmp_path).unwrap();

        let loaded = load_grid(tmp_path).unwrap();

        if let Values::Host(data) = &loaded.values {
            assert_eq!(data, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        assert_eq!(loaded.shape, vec![2, 3]);

        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn test_checkpoint_error_display() {
        let err = CheckpointError::InvalidMagic;
        assert!(format!("{}", err).contains("Invalid"));

        let err = CheckpointError::UnsupportedVersion(99);
        assert!(format!("{}", err).contains("99"));
    }

    #[test]
    fn test_checkpoint_metadata_default() {
        let meta = CheckpointMetadata::default();
        assert_eq!(meta.epoch, 0);
        assert_eq!(meta.global_step, 0);
        assert!(meta.extra.is_empty());
    }
}
