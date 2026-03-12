use std::time::Instant;

use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, VideoCaptureTraitConst};

use crate::capture::frame::{Frame, FrameMeta, PixelFormat};
use crate::runtime::context::HasAllocator;
use crate::{LangError, Result};

/// Source for camera capture.
#[derive(Clone, Debug)]
pub enum CameraSource {
    /// Webcam by device index (0 = default).
    Device(i32),
    /// Video file path.
    File(String),
}

/// Configuration for camera capture.
#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub api: i32,
    pub convert_rgb: bool,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fps: 30.0,
            api: videoio::CAP_ANY,
            convert_rgb: true,
        }
    }
}

impl CameraConfig {
    pub fn with_resolution(mut self, width: i32, height: i32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = fps;
        self
    }

    pub fn with_api(mut self, api: i32) -> Self {
        self.api = api;
        self
    }

    /// Keep BGR output (skip OpenCV cvtColor conversion).
    pub fn bgr(mut self) -> Self {
        self.convert_rgb = false;
        self
    }
}

/// OpenCV-backed video capture producing TLSF-backed `Frame`s.
pub struct Camera {
    cap: VideoCapture,
    config: CameraConfig,
    frame_counter: u64,
    start_time: Instant,
    mat_buf: Mat,
    rgb_buf: Mat,
}

impl Camera {
    /// Open a camera source with the given configuration.
    pub fn open(source: CameraSource, config: CameraConfig) -> Result<Self> {
        let mut cap = match &source {
            CameraSource::Device(idx) => {
                VideoCapture::new(*idx, config.api).map_err(|e| LangError::CameraNotOpened {
                    reason: format!("device {idx}: {e}"),
                })?
            }
            CameraSource::File(path) => VideoCapture::from_file(path, config.api).map_err(
                |e| LangError::CameraNotOpened {
                    reason: format!("file {path}: {e}"),
                },
            )?,
        };

        if !VideoCapture::is_opened(&cap).map_err(|e| LangError::CameraNotOpened {
            reason: format!("{e}"),
        })? {
            return Err(LangError::CameraNotOpened {
                reason: format!("source {:?} did not open", source),
            });
        }

        // Apply resolution/fps hints (best-effort, camera may ignore).
        if let CameraSource::Device(_) = &source {
            let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, config.width as f64);
            let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, config.height as f64);
            let _ = cap.set(videoio::CAP_PROP_FPS, config.fps);
        }

        Ok(Self {
            cap,
            config,
            frame_counter: 0,
            start_time: Instant::now(),
            mat_buf: Mat::default(),
            rgb_buf: Mat::default(),
        })
    }

    /// Shorthand: open default webcam (device 0) with default config.
    pub fn default_webcam() -> Result<Self> {
        Self::open(CameraSource::Device(0), CameraConfig::default())
    }

    /// Read the next frame into a TLSF-backed `Frame`.
    pub fn read(&mut self, ctx: &(impl HasAllocator + ?Sized)) -> Result<Frame> {
        let grabbed = self
            .cap
            .read(&mut self.mat_buf)
            .map_err(|e| LangError::Capture {
                message: format!("VideoCapture::read failed: {e}"),
            })?;

        if !grabbed || self.mat_buf.empty() {
            return Err(LangError::Capture {
                message: "no frame captured (end of stream or device error)".to_string(),
            });
        }

        let timestamp_us = self.start_time.elapsed().as_micros() as u64;
        self.frame_counter += 1;

        // Source mat for TLSF copy — either RGB-converted or raw BGR.
        let src = if self.config.convert_rgb {
            opencv::imgproc::cvt_color(&self.mat_buf, &mut self.rgb_buf, opencv::imgproc::COLOR_BGR2RGB, 0)
                .map_err(|e| LangError::Capture {
                    message: format!("cvtColor BGR→RGB failed: {e}"),
                })?;
            &self.rgb_buf
        } else {
            &self.mat_buf
        };

        let rows = src.rows() as usize;
        let cols = src.cols() as usize;
        let channels = src.channels() as usize;

        let format = if self.config.convert_rgb {
            PixelFormat::Rgb8
        } else {
            match channels {
                3 => PixelFormat::Bgr8,
                1 => PixelFormat::Gray8,
                _ => PixelFormat::Bgr8,
            }
        };

        let meta = FrameMeta {
            width: cols,
            height: rows,
            format,
            frame_index: self.frame_counter,
            timestamp_us,
        };

        // data_bytes() returns the raw pixel data as &[u8].
        let data = src.data_bytes().map_err(|e| LangError::Capture {
            message: format!("Mat::data_bytes failed: {e}"),
        })?;

        Frame::from_bytes(ctx, meta, data)
    }

    /// Actual resolution being captured (may differ from config if camera overrides).
    pub fn resolution(&self) -> (i32, i32) {
        let w = self
            .cap
            .get(videoio::CAP_PROP_FRAME_WIDTH)
            .unwrap_or(self.config.width as f64) as i32;
        let h = self
            .cap
            .get(videoio::CAP_PROP_FRAME_HEIGHT)
            .unwrap_or(self.config.height as f64) as i32;
        (w, h)
    }

    /// Actual FPS being used.
    pub fn fps(&self) -> f64 {
        self.cap
            .get(videoio::CAP_PROP_FPS)
            .unwrap_or(self.config.fps)
    }

    /// Number of frames read so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_counter
    }

    /// Whether the underlying capture device is still open.
    pub fn is_opened(&self) -> bool {
        VideoCapture::is_opened(&self.cap).unwrap_or(false)
    }
}
