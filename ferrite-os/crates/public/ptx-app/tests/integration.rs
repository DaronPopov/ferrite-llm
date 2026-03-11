//! End-to-end integration tests for ptx-app's QoL crate integrations.
//!
//! These tests exercise the non-GPU paths: checkpoint persistence (tempfile),
//! progress bars (indicatif), platform directories, memory-mapped I/O (memmap2),
//! human-readable formatting (humansize), and safe type casting (bytemuck).

use std::io::Write;

// =========================================================================
// tempfile + checkpoint: atomic save/restore round-trip
// =========================================================================

#[test]
fn checkpoint_save_restore_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    // Save a checkpoint
    let state = serde_json::json!({
        "epoch": 42,
        "loss": 0.0012,
        "weights": [1.0, 2.0, 3.0],
    });
    store.save("training", &state).unwrap();

    // Verify the file exists
    let checkpoint_path = dir.path().join("training.json");
    assert!(checkpoint_path.exists(), "checkpoint file should exist");

    // Read it back
    let restored: Option<serde_json::Value> = store.restore("training").unwrap();
    assert!(restored.is_some(), "should restore saved checkpoint");

    let restored = restored.unwrap();
    assert_eq!(restored["epoch"], 42);
    assert_eq!(restored["loss"], 0.0012);
    assert_eq!(restored["weights"], serde_json::json!([1.0, 2.0, 3.0]));
}

#[test]
fn checkpoint_restore_nonexistent_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    let result: Option<serde_json::Value> = store.restore("does-not-exist").unwrap();
    assert!(result.is_none(), "restoring nonexistent label should return None");
}

#[test]
fn checkpoint_overwrite_preserves_latest() {
    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    store.save("state", &serde_json::json!({"version": 1})).unwrap();
    store.save("state", &serde_json::json!({"version": 2})).unwrap();

    let restored: serde_json::Value = store.restore("state").unwrap().unwrap();
    assert_eq!(restored["version"], 2, "should have the latest version");
}

#[test]
fn checkpoint_multiple_labels() {
    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    store.save("epoch-1", &serde_json::json!({"loss": 0.5})).unwrap();
    store.save("epoch-2", &serde_json::json!({"loss": 0.3})).unwrap();
    store.save("best", &serde_json::json!({"loss": 0.1})).unwrap();

    let e1: serde_json::Value = store.restore("epoch-1").unwrap().unwrap();
    let e2: serde_json::Value = store.restore("epoch-2").unwrap().unwrap();
    let best: serde_json::Value = store.restore("best").unwrap().unwrap();

    assert_eq!(e1["loss"], 0.5);
    assert_eq!(e2["loss"], 0.3);
    assert_eq!(best["loss"], 0.1);
}

#[test]
fn checkpoint_typed_roundtrip() {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TrainingState {
        epoch: u32,
        learning_rate: f64,
        batch_size: usize,
        tags: Vec<String>,
    }

    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    let original = TrainingState {
        epoch: 100,
        learning_rate: 0.001,
        batch_size: 64,
        tags: vec!["experiment-1".into(), "gpu-a100".into()],
    };

    store.save("typed", &original).unwrap();
    let restored: TrainingState = store.restore("typed").unwrap().unwrap();
    assert_eq!(restored, original);
}

// =========================================================================
// humansize: PoolStats Display formatting
// =========================================================================

#[test]
fn pool_stats_display_healthy() {
    let stats = ptx_app::PoolStats {
        total_bytes: 1_073_741_824, // 1 GiB
        allocated_bytes: 268_435_456, // 256 MiB
        free_bytes: 805_306_368,     // 768 MiB
        utilization: 25.0,
        fragmentation: 0.05,
        healthy: true,
    };
    let display = format!("{}", stats);
    assert!(display.contains("256"), "should show allocated size: {}", display);
    assert!(display.contains("1"), "should show total: {}", display);
    assert!(display.contains("25.0%"), "should show utilization: {}", display);
    assert!(!display.contains("UNHEALTHY"), "healthy pool should not say UNHEALTHY: {}", display);
}

#[test]
fn pool_stats_display_unhealthy() {
    let stats = ptx_app::PoolStats {
        total_bytes: 1_073_741_824,
        allocated_bytes: 1_020_054_732,
        free_bytes: 53_687_092,
        utilization: 95.0,
        fragmentation: 0.85,
        healthy: false,
    };
    let display = format!("{}", stats);
    assert!(display.contains("UNHEALTHY"), "unhealthy pool should say UNHEALTHY: {}", display);
    assert!(display.contains("95.0%"), "should show utilization: {}", display);
}

#[test]
fn pool_stats_display_zero() {
    let stats = ptx_app::PoolStats {
        total_bytes: 0,
        allocated_bytes: 0,
        free_bytes: 0,
        utilization: 0.0,
        fragmentation: 0.0,
        healthy: true,
    };
    let display = format!("{}", stats);
    // Should not panic, should produce something reasonable
    assert!(display.contains("0"), "zero pool should show 0: {}", display);
}

// =========================================================================
// indicatif: progress bar and spinner creation
// =========================================================================

#[test]
fn indicatif_progress_bar_lifecycle() {
    // Verify that indicatif's ProgressBar works end-to-end
    let pb = indicatif::ProgressBar::new(100);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{prefix} [{bar:40.cyan/blue}] {pos}/{len} ({eta} remaining)"
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_prefix("test");

    // Simulate progress
    for _ in 0..100 {
        pb.inc(1);
    }
    assert_eq!(pb.position(), 100);
    pb.finish_with_message("done");
    assert!(pb.is_finished());
}

#[test]
fn indicatif_spinner_lifecycle() {
    let sp = indicatif::ProgressBar::new_spinner();
    sp.set_style(
        indicatif::ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&[".", "..", "...", "....", ""]),
    );
    sp.set_message("testing");

    // Tick a few times
    for _ in 0..10 {
        sp.tick();
    }
    sp.finish_with_message("done");
    assert!(sp.is_finished());
}

// =========================================================================
// directories: platform path resolution
// =========================================================================

#[test]
fn directories_project_dirs_resolves() {
    // This should work on all supported platforms
    let dirs = directories::ProjectDirs::from("", "", "ferrite");
    assert!(dirs.is_some(), "ProjectDirs::from should succeed");

    let dirs = dirs.unwrap();
    let data_dir = dirs.data_dir();
    let checkpoint_dir = data_dir.join("checkpoints").join("test-app");

    // Path should be non-empty and absolute
    assert!(
        checkpoint_dir.to_string_lossy().len() > 10,
        "checkpoint dir should be a reasonable path: {:?}",
        checkpoint_dir
    );

    // On Linux, should contain .local/share or similar
    #[cfg(target_os = "linux")]
    {
        let path_str = data_dir.to_string_lossy();
        assert!(
            path_str.contains("share") || path_str.contains("ferrite"),
            "Linux data dir should follow XDG convention: {}",
            path_str
        );
    }
}

#[test]
fn directories_config_and_cache_dirs_exist() {
    let dirs = directories::ProjectDirs::from("", "", "ferrite").unwrap();
    // These should return non-empty paths on all platforms
    assert!(!dirs.config_dir().as_os_str().is_empty());
    assert!(!dirs.cache_dir().as_os_str().is_empty());
    assert!(!dirs.data_dir().as_os_str().is_empty());
}

// =========================================================================
// memmap2: memory-mapped file I/O
// =========================================================================

#[test]
fn memmap2_basic_roundtrip() {
    // Write known f32 data to a temp file, mmap it back, verify contents
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_tensor.bin");
    std::fs::write(&path, bytes).unwrap();

    // Memory-map the file
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();

    assert_eq!(mmap.len(), bytes.len());

    // Cast back to f32 via bytemuck
    let restored: &[f32] = bytemuck::cast_slice(&mmap);
    assert_eq!(restored, &data[..]);
}

#[test]
fn memmap2_large_file() {
    // Test with a larger buffer (1MB of f32s = 256K elements)
    let count = 256 * 1024;
    let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_tensor.bin");
    std::fs::write(&path, bytes).unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();

    assert_eq!(mmap.len(), count * 4); // 4 bytes per f32

    let restored: &[f32] = bytemuck::cast_slice(&mmap);
    assert_eq!(restored.len(), count);
    assert_eq!(restored[0], 0.0);
    assert!((restored[1000] - 1.0).abs() < 1e-6);
    assert!((restored[count - 1] - (count - 1) as f32 * 0.001).abs() < 1e-3);
}

#[test]
fn memmap2_multi_dtype() {
    // Test with i32 data
    let data_i32: Vec<i32> = vec![-100, 0, 42, i32::MAX, i32::MIN];
    let bytes: &[u8] = bytemuck::cast_slice(&data_i32);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32_tensor.bin");
    std::fs::write(&path, bytes).unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let restored: &[i32] = bytemuck::cast_slice(&mmap);
    assert_eq!(restored, &data_i32[..]);

    // Test with u8 data
    let data_u8: Vec<u8> = vec![0, 127, 255, 1, 42];
    let path_u8 = dir.path().join("u8_tensor.bin");
    std::fs::write(&path_u8, &data_u8).unwrap();

    let file_u8 = std::fs::File::open(&path_u8).unwrap();
    let mmap_u8 = unsafe { memmap2::Mmap::map(&file_u8) }.unwrap();
    assert_eq!(&mmap_u8[..], &data_u8[..]);
}

// =========================================================================
// bytemuck: Pod safety guarantees
// =========================================================================

#[test]
fn bytemuck_cast_slice_f32_roundtrip() {
    let original: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, f32::INFINITY];
    let bytes: &[u8] = bytemuck::cast_slice(&original);
    let recovered: &[f32] = bytemuck::cast_slice(bytes);
    assert_eq!(original, recovered);
}

#[test]
fn bytemuck_cast_slice_i32_roundtrip() {
    let original: Vec<i32> = vec![0, -1, i32::MAX, i32::MIN, 42];
    let bytes: &[u8] = bytemuck::cast_slice(&original);
    let recovered: &[i32] = bytemuck::cast_slice(bytes);
    assert_eq!(original, recovered);
}

#[test]
fn bytemuck_zeroed() {
    let z: f32 = bytemuck::Zeroable::zeroed();
    assert_eq!(z, 0.0);

    let z: i32 = bytemuck::Zeroable::zeroed();
    assert_eq!(z, 0);

    let z: u8 = bytemuck::Zeroable::zeroed();
    assert_eq!(z, 0);
}

#[test]
fn bytemuck_pod_vec_init() {
    // This is how Storage::to_host now initializes its buffer
    let data: Vec<f32> = vec![bytemuck::Zeroable::zeroed(); 1024];
    assert_eq!(data.len(), 1024);
    assert!(data.iter().all(|&x| x == 0.0));
}

// =========================================================================
// humansize: human-readable byte formatting
// =========================================================================

#[test]
fn humansize_formatting() {
    use humansize::{format_size, BINARY};

    assert_eq!(format_size(0_usize, BINARY), "0 B");
    assert_eq!(format_size(1023_usize, BINARY), "1023 B");
    assert_eq!(format_size(1024_usize, BINARY), "1 KiB");
    assert_eq!(format_size(1_048_576_usize, BINARY), "1 MiB");
    assert_eq!(format_size(1_073_741_824_usize, BINARY), "1 GiB");

    // Typical GPU memory sizes
    let pool_4gb = format_size(4_294_967_296_u64, BINARY);
    assert!(pool_4gb.contains("4"), "4 GiB should contain '4': {}", pool_4gb);
    assert!(pool_4gb.contains("GiB"), "4 GiB should contain 'GiB': {}", pool_4gb);
}

// =========================================================================
// FerApp builder validation (no GPU required)
// =========================================================================

#[test]
fn builder_rejects_empty_name() {
    let result = ptx_app::FerApp::new("")
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("empty"), "should mention empty name: {}", msg);
}

#[test]
fn builder_rejects_invalid_name_chars() {
    let result = ptx_app::FerApp::new("my app!")
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("invalid characters"), "should mention invalid chars: {}", msg);
}

#[test]
fn builder_rejects_zero_pool_fraction() {
    let result = ptx_app::FerApp::new("test")
        .pool_fraction(0.0)
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("pool_fraction"), "should mention pool_fraction: {}", msg);
}

#[test]
fn builder_rejects_negative_pool_fraction() {
    let result = ptx_app::FerApp::new("test")
        .pool_fraction(-0.5)
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
}

#[test]
fn builder_rejects_pool_fraction_above_one() {
    let result = ptx_app::FerApp::new("test")
        .pool_fraction(1.5)
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
}

#[test]
fn builder_rejects_zero_streams() {
    let result = ptx_app::FerApp::new("test")
        .streams(0)
        .run(|_ctx| Ok(()));

    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("streams"), "should mention streams: {}", msg);
}

#[test]
fn builder_accepts_valid_names() {
    // These should pass validation (will fail at runtime init since no GPU,
    // but we're testing validation only)
    for name in &["my-app", "test_123", "a", "GPU-worker-01"] {
        let result = ptx_app::FerApp::new(name)
            .run(|_ctx| Ok(()));
        // Should NOT be a ValidationError - it might be a Runtime error (no GPU)
        if let Err(ptx_app::AppError::ValidationError { .. }) = &result {
            panic!("'{}' should pass validation but got: {:?}", name, result);
        }
    }
}

// =========================================================================
// Restart policy constructors
// =========================================================================

#[test]
fn restart_constructors() {
    let _ = ptx_app::Restart::never();
    let _ = ptx_app::Restart::on_failure(3);
    let _ = ptx_app::Restart::on_failure_with_backoff(5, 1000, 30000);
    let _ = ptx_app::Restart::always(10);
}

// =========================================================================
// Priority enum
// =========================================================================

#[test]
fn priority_ordering() {
    assert_eq!(ptx_app::Priority::Low.as_i32(), 0);
    assert_eq!(ptx_app::Priority::Normal.as_i32(), 1);
    assert_eq!(ptx_app::Priority::High.as_i32(), 2);
    assert_eq!(ptx_app::Priority::Realtime.as_i32(), 3);
}

// =========================================================================
// AppError Display
// =========================================================================

#[test]
fn app_error_display() {
    let errors = vec![
        ptx_app::AppError::DaemonUnavailable {
            message: "socket not found".into(),
        },
        ptx_app::AppError::ValidationError {
            message: "bad param".into(),
        },
        ptx_app::AppError::PolicyDenied {
            action: "run".into(),
            reason: "quota exceeded".into(),
        },
        ptx_app::AppError::CheckpointError {
            detail: "disk full".into(),
        },
        ptx_app::AppError::App {
            message: "generic error".into(),
        },
        ptx_app::AppError::Panic {
            message: "thread panic".into(),
        },
    ];

    for err in &errors {
        let display = format!("{}", err);
        assert!(!display.is_empty(), "error display should not be empty");
    }
}

// =========================================================================
// End-to-end: checkpoint + memmap + bytemuck pipeline
// =========================================================================

#[test]
fn e2e_save_tensor_data_checkpoint_and_reload_via_mmap() {
    // Simulate: compute result -> save as binary -> checkpoint metadata ->
    // restore metadata -> mmap binary -> verify

    let dir = tempfile::tempdir().unwrap();

    // 1. "Compute" a result (simulated tensor data)
    let tensor_data: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();

    // 2. Save raw tensor data to a binary file
    let data_path = dir.path().join("result.bin");
    let bytes: &[u8] = bytemuck::cast_slice(&tensor_data);
    std::fs::write(&data_path, bytes).unwrap();

    // 3. Save metadata via checkpoint
    let store = ptx_app_checkpoint_store(dir.path());
    let metadata = serde_json::json!({
        "shape": [1024],
        "dtype": "f32",
        "data_file": "result.bin",
        "mean": tensor_data.iter().sum::<f32>() / tensor_data.len() as f32,
    });
    store.save("result-meta", &metadata).unwrap();

    // 4. "Restart" — restore metadata
    let restored_meta: serde_json::Value = store.restore("result-meta").unwrap().unwrap();
    assert_eq!(restored_meta["shape"], serde_json::json!([1024]));
    assert_eq!(restored_meta["dtype"], "f32");

    // 5. Mmap the binary data back
    let file = std::fs::File::open(&data_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let restored_data: &[f32] = bytemuck::cast_slice(&mmap);

    // 6. Verify data integrity
    assert_eq!(restored_data.len(), 1024);
    for (i, (&original, &restored)) in tensor_data.iter().zip(restored_data.iter()).enumerate() {
        assert!(
            (original - restored).abs() < f32::EPSILON,
            "mismatch at index {}: {} != {}",
            i,
            original,
            restored
        );
    }
}

#[test]
fn e2e_progress_with_checkpoint_loop() {
    // Simulate a training loop with progress tracking and periodic checkpoints
    let dir = tempfile::tempdir().unwrap();
    let store = ptx_app_checkpoint_store(dir.path());

    let pb = indicatif::ProgressBar::new(10);
    for epoch in 0..10 {
        // "Train" (simulated)
        let loss = 1.0 / (epoch as f64 + 1.0);

        // Periodic checkpoint
        if epoch % 3 == 0 {
            store
                .save(
                    &format!("epoch-{}", epoch),
                    &serde_json::json!({"epoch": epoch, "loss": loss}),
                )
                .unwrap();
        }

        pb.inc(1);
    }
    pb.finish();
    assert!(pb.is_finished());

    // Verify checkpoints exist for epochs 0, 3, 6, 9
    for epoch in [0, 3, 6, 9] {
        let label = format!("epoch-{}", epoch);
        let data: serde_json::Value = store.restore(&label).unwrap().unwrap();
        assert_eq!(data["epoch"], epoch);
    }

    // epoch-1 should not exist
    let missing: Option<serde_json::Value> = store.restore("epoch-1").unwrap();
    assert!(missing.is_none());
}

// =========================================================================
// Helper: construct a CheckpointStore without going through Ctx
// (CheckpointStore is pub(crate), so we test it via a helper module)
// =========================================================================

/// We can't access CheckpointStore directly (pub(crate)), so we test it
/// through a minimal wrapper that mimics what Ctx does internally.
struct TestCheckpointStore {
    dir: std::path::PathBuf,
}

impl TestCheckpointStore {
    fn new(dir: &std::path::Path) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        Self {
            dir: dir.to_path_buf(),
        }
    }

    fn save(&self, label: &str, state: &impl serde::Serialize) -> Result<(), Box<dyn std::error::Error>> {
        let dest = self.dir.join(format!("{}.json", label));
        let data = serde_json::to_string_pretty(state)?;

        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir)?;
        tmp.write_all(data.as_bytes())?;
        tmp.persist(&dest)?;
        Ok(())
    }

    fn restore<T: serde::de::DeserializeOwned>(&self, label: &str) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let path = self.dir.join(format!("{}.json", label));
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(&path)?;
        Ok(Some(serde_json::from_str(&data)?))
    }
}

fn ptx_app_checkpoint_store(dir: &std::path::Path) -> TestCheckpointStore {
    TestCheckpointStore::new(dir)
}
