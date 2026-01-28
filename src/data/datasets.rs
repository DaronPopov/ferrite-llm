//! Dataset loading utilities for common ML datasets
//!
//! Provides loaders for:
//! - MNIST (handwritten digits)
//! - Future: CIFAR-10, Fashion-MNIST, etc.

use super::Grid;
use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

/// MNIST dataset loader
///
/// MNIST consists of:
/// - 60,000 training images (28x28 grayscale)
/// - 10,000 test images (28x28 grayscale)
/// - 10 classes (digits 0-9)
pub struct MnistDataset {
    pub train_images: Vec<f32>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<f32>,
    pub test_labels: Vec<u8>,
    pub num_train: usize,
    pub num_test: usize,
}

impl MnistDataset {
    /// Load MNIST from IDX files
    ///
    /// Expects files in the standard MNIST format:
    /// - train-images-idx3-ubyte (or .gz)
    /// - train-labels-idx1-ubyte (or .gz)
    /// - t10k-images-idx3-ubyte (or .gz)
    /// - t10k-labels-idx1-ubyte (or .gz)
    pub fn load<P: AsRef<Path>>(data_dir: P) -> Result<Self, MnistError> {
        let dir = data_dir.as_ref();

        let train_images_path = dir.join("train-images-idx3-ubyte");
        let train_labels_path = dir.join("train-labels-idx1-ubyte");
        let test_images_path = dir.join("t10k-images-idx3-ubyte");
        let test_labels_path = dir.join("t10k-labels-idx1-ubyte");

        let (train_images, num_train) = load_images(&train_images_path)?;
        let train_labels = load_labels(&train_labels_path)?;
        let (test_images, num_test) = load_images(&test_images_path)?;
        let test_labels = load_labels(&test_labels_path)?;

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_train,
            num_test,
        })
    }

    /// Get a batch of training data as Grids
    ///
    /// Returns (images, labels) where:
    /// - images: Grid of shape [batch_size, 784] (flattened 28x28)
    /// - labels: Grid of shape [batch_size, 10] (one-hot encoded)
    pub fn get_train_batch(&self, start: usize, batch_size: usize) -> (Grid, Grid) {
        let end = (start + batch_size).min(self.num_train);
        let actual_batch = end - start;

        // Extract image batch (normalized to [0, 1])
        let img_start = start * 784;
        let img_end = end * 784;
        let images: Vec<f32> = self.train_images[img_start..img_end].to_vec();

        // One-hot encode labels
        let mut labels = vec![0.0f32; actual_batch * 10];
        for (i, &label) in self.train_labels[start..end].iter().enumerate() {
            labels[i * 10 + label as usize] = 1.0;
        }

        (
            Grid::new(images, vec![actual_batch, 784]),
            Grid::new(labels, vec![actual_batch, 10]),
        )
    }

    /// Get a batch of test data as Grids
    pub fn get_test_batch(&self, start: usize, batch_size: usize) -> (Grid, Grid) {
        let end = (start + batch_size).min(self.num_test);
        let actual_batch = end - start;

        let img_start = start * 784;
        let img_end = end * 784;
        let images: Vec<f32> = self.test_images[img_start..img_end].to_vec();

        let mut labels = vec![0.0f32; actual_batch * 10];
        for (i, &label) in self.test_labels[start..end].iter().enumerate() {
            labels[i * 10 + label as usize] = 1.0;
        }

        (
            Grid::new(images, vec![actual_batch, 784]),
            Grid::new(labels, vec![actual_batch, 10]),
        )
    }

    /// Get training images as 2D images (for CNNs)
    ///
    /// Returns Grid of shape [batch_size, 1, 28, 28]
    pub fn get_train_batch_2d(&self, start: usize, batch_size: usize) -> (Grid, Grid) {
        let end = (start + batch_size).min(self.num_train);
        let actual_batch = end - start;

        let img_start = start * 784;
        let img_end = end * 784;
        let images: Vec<f32> = self.train_images[img_start..img_end].to_vec();

        let mut labels = vec![0.0f32; actual_batch * 10];
        for (i, &label) in self.train_labels[start..end].iter().enumerate() {
            labels[i * 10 + label as usize] = 1.0;
        }

        (
            Grid::new(images, vec![actual_batch, 1, 28, 28]),
            Grid::new(labels, vec![actual_batch, 10]),
        )
    }

    /// Get raw label (not one-hot) for a training sample
    pub fn get_train_label(&self, idx: usize) -> u8 {
        self.train_labels[idx]
    }

    /// Get raw label for a test sample
    pub fn get_test_label(&self, idx: usize) -> u8 {
        self.test_labels[idx]
    }
}

/// Errors that can occur during MNIST loading
#[derive(Debug)]
pub enum MnistError {
    IoError(std::io::Error),
    InvalidMagic { expected: u32, got: u32 },
    InvalidFormat(String),
}

impl From<std::io::Error> for MnistError {
    fn from(e: std::io::Error) -> Self {
        MnistError::IoError(e)
    }
}

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnistError::IoError(e) => write!(f, "IO error: {}", e),
            MnistError::InvalidMagic { expected, got } => {
                write!(f, "Invalid magic number: expected {}, got {}", expected, got)
            }
            MnistError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for MnistError {}

/// Load images from IDX3 file format
fn load_images<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, usize), MnistError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let magic = read_u32_be(&mut reader)?;
    if magic != 0x00000803 {
        return Err(MnistError::InvalidMagic { expected: 0x00000803, got: magic });
    }

    let num_images = read_u32_be(&mut reader)? as usize;
    let num_rows = read_u32_be(&mut reader)? as usize;
    let num_cols = read_u32_be(&mut reader)? as usize;

    if num_rows != 28 || num_cols != 28 {
        return Err(MnistError::InvalidFormat(
            format!("Expected 28x28 images, got {}x{}", num_rows, num_cols)
        ));
    }

    // Read and normalize pixel data
    let mut raw_data = vec![0u8; num_images * 784];
    reader.read_exact(&mut raw_data)?;

    // Convert to f32 and normalize to [0, 1]
    let images: Vec<f32> = raw_data.iter()
        .map(|&x| x as f32 / 255.0)
        .collect();

    Ok((images, num_images))
}

/// Load labels from IDX1 file format
fn load_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, MnistError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let magic = read_u32_be(&mut reader)?;
    if magic != 0x00000801 {
        return Err(MnistError::InvalidMagic { expected: 0x00000801, got: magic });
    }

    let num_labels = read_u32_be(&mut reader)? as usize;

    // Read labels
    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

/// Read big-endian u32
fn read_u32_be<R: Read>(reader: &mut R) -> Result<u32, std::io::Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Simple data iterator for batching
pub struct DataIterator<'a> {
    dataset: &'a MnistDataset,
    batch_size: usize,
    current: usize,
    is_train: bool,
    shuffle_indices: Option<Vec<usize>>,
}

impl<'a> DataIterator<'a> {
    /// Create iterator over training data
    pub fn train(dataset: &'a MnistDataset, batch_size: usize) -> Self {
        DataIterator {
            dataset,
            batch_size,
            current: 0,
            is_train: true,
            shuffle_indices: None,
        }
    }

    /// Create iterator over test data
    pub fn test(dataset: &'a MnistDataset, batch_size: usize) -> Self {
        DataIterator {
            dataset,
            batch_size,
            current: 0,
            is_train: false,
            shuffle_indices: None,
        }
    }

    /// Enable shuffling with a seed
    pub fn shuffle(mut self, seed: u64) -> Self {
        let n = if self.is_train { self.dataset.num_train } else { self.dataset.num_test };
        let mut indices: Vec<usize> = (0..n).collect();

        // Simple LCG shuffle
        let mut rng = seed;
        for i in (1..n).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        self.shuffle_indices = Some(indices);
        self
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        let n = if self.is_train { self.dataset.num_train } else { self.dataset.num_test };
        (n + self.batch_size - 1) / self.batch_size
    }
}

impl<'a> Iterator for DataIterator<'a> {
    type Item = (Grid, Grid);

    fn next(&mut self) -> Option<Self::Item> {
        let total = if self.is_train { self.dataset.num_train } else { self.dataset.num_test };

        if self.current >= total {
            return None;
        }

        let batch_start = self.current;
        let batch_end = (self.current + self.batch_size).min(total);
        let actual_batch = batch_end - batch_start;
        self.current = batch_end;

        // If shuffled, gather from shuffled indices
        if let Some(indices) = &self.shuffle_indices {
            let mut images = Vec::with_capacity(actual_batch * 784);
            let mut labels = vec![0.0f32; actual_batch * 10];

            for (i, &idx) in indices[batch_start..batch_end].iter().enumerate() {
                let (img_data, label) = if self.is_train {
                    let img_start = idx * 784;
                    (&self.dataset.train_images[img_start..img_start + 784], self.dataset.train_labels[idx])
                } else {
                    let img_start = idx * 784;
                    (&self.dataset.test_images[img_start..img_start + 784], self.dataset.test_labels[idx])
                };

                images.extend_from_slice(img_data);
                labels[i * 10 + label as usize] = 1.0;
            }

            Some((
                Grid::new(images, vec![actual_batch, 784]),
                Grid::new(labels, vec![actual_batch, 10]),
            ))
        } else {
            // Non-shuffled, use the optimized batch methods
            if self.is_train {
                Some(self.dataset.get_train_batch(batch_start, self.batch_size))
            } else {
                Some(self.dataset.get_test_batch(batch_start, self.batch_size))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_error_display() {
        let io_err = MnistError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(format!("{}", io_err).contains("IO error"));

        let magic_err = MnistError::InvalidMagic { expected: 0x803, got: 0x801 };
        assert!(format!("{}", magic_err).contains("Invalid magic"));

        let format_err = MnistError::InvalidFormat("bad format".to_string());
        assert!(format!("{}", format_err).contains("Invalid format"));
    }

    #[test]
    fn test_data_iterator_num_batches() {
        // Create minimal mock dataset
        let dataset = MnistDataset {
            train_images: vec![0.0; 100 * 784],
            train_labels: vec![0; 100],
            test_images: vec![0.0; 20 * 784],
            test_labels: vec![0; 20],
            num_train: 100,
            num_test: 20,
        };

        let train_iter = DataIterator::train(&dataset, 32);
        assert_eq!(train_iter.num_batches(), 4); // ceil(100/32) = 4

        let test_iter = DataIterator::test(&dataset, 10);
        assert_eq!(test_iter.num_batches(), 2); // ceil(20/10) = 2
    }

    #[test]
    fn test_get_batch() {
        let dataset = MnistDataset {
            train_images: (0..50 * 784).map(|i| (i % 256) as f32 / 255.0).collect(),
            train_labels: (0..50).map(|i| (i % 10) as u8).collect(),
            test_images: vec![0.0; 10 * 784],
            test_labels: vec![0; 10],
            num_train: 50,
            num_test: 10,
        };

        let (images, labels) = dataset.get_train_batch(0, 5);
        assert_eq!(images.shape, vec![5, 784]);
        assert_eq!(labels.shape, vec![5, 10]);
    }

    #[test]
    fn test_get_batch_2d() {
        let dataset = MnistDataset {
            train_images: vec![0.5; 10 * 784],
            train_labels: vec![5; 10],
            test_images: vec![],
            test_labels: vec![],
            num_train: 10,
            num_test: 0,
        };

        let (images, _labels) = dataset.get_train_batch_2d(0, 3);
        assert_eq!(images.shape, vec![3, 1, 28, 28]);
    }
}
