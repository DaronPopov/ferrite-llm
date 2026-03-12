use std::sync::Arc;

use crate::capture::camera::{Camera, CameraConfig, CameraSource};
use crate::capture::frame::Frame;
use crate::runtime::context::HasAllocator;
use crate::runtime::cpu_tlsf::{global_cpu_tlsf, CpuTlsf};

use super::clock::SensorClock;
use super::stream::{SensorError, SensorInfo, SensorStream, Stamped};

/// Adapter wrapping `Camera` to implement `SensorStream<Sample = Frame>`.
///
/// Provides the allocator context internally so the capture thread can
/// produce TLSF-backed frames without needing an external `&impl HasAllocator`.
pub struct CameraAdapter {
    camera: Camera,
    info: SensorInfo,
    clock: SensorClock,
    sequence: u64,
    allocator: Arc<CpuTlsf>,
}

/// Lightweight context that holds an `Arc<CpuTlsf>` for use with `HasAllocator`.
struct AdapterCtx {
    allocator: Arc<CpuTlsf>,
}

impl HasAllocator for AdapterCtx {
    fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }
}

impl CameraAdapter {
    /// Create a new adapter from an already-opened camera.
    pub fn new(camera: Camera, name: &str) -> Self {
        let fps = camera.fps();
        Self {
            camera,
            info: SensorInfo {
                name: name.to_string(),
                nominal_hz: fps,
            },
            clock: SensorClock::new(),
            sequence: 0,
            allocator: Arc::clone(global_cpu_tlsf()),
        }
    }

    /// Open the default webcam and wrap it as a SensorStream.
    pub fn default_webcam() -> Result<Self, SensorError> {
        let camera = Camera::default_webcam().map_err(|e| {
            SensorError::ReadFailed(format!("failed to open default webcam: {}", e))
        })?;
        Ok(Self::new(camera, "webcam-0"))
    }

    /// Open a specific camera source with config.
    pub fn open(source: CameraSource, config: CameraConfig, name: &str) -> Result<Self, SensorError> {
        let camera = Camera::open(source, config).map_err(|e| {
            SensorError::ReadFailed(format!("failed to open camera: {}", e))
        })?;
        Ok(Self::new(camera, name))
    }
}

impl SensorStream for CameraAdapter {
    type Sample = Frame;

    fn info(&self) -> &SensorInfo {
        &self.info
    }

    fn read(&mut self) -> Result<Stamped<Frame>, SensorError> {
        let ctx = AdapterCtx {
            allocator: Arc::clone(&self.allocator),
        };

        let frame = self.camera.read(&ctx).map_err(|e| {
            SensorError::ReadFailed(format!("{}", e))
        })?;

        self.sequence += 1;
        let timestamp_us = self.clock.now_us();

        Ok(Stamped {
            data: frame,
            timestamp_us,
            sequence: self.sequence,
        })
    }

    fn is_open(&self) -> bool {
        self.camera.is_opened()
    }
}
