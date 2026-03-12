use std::fmt;

/// A timestamped sample from any sensor.
#[derive(Clone, Debug)]
pub struct Stamped<T> {
    pub data: T,
    /// Monotonic microseconds since sensor clock epoch.
    pub timestamp_us: u64,
    /// Monotonic frame counter.
    pub sequence: u64,
}

/// Info about a sensor source.
#[derive(Clone, Debug)]
pub struct SensorInfo {
    /// Human-readable name, e.g. "webcam-0", "imu-usb1".
    pub name: String,
    /// Expected sample rate in Hz.
    pub nominal_hz: f64,
}

/// Errors that can occur during sensor operations.
#[derive(Debug)]
pub enum SensorError {
    /// The device was disconnected or lost.
    DeviceLost,
    /// A read timed out without producing a sample.
    Timeout,
    /// A read failed with a descriptive message.
    ReadFailed(String),
}

impl fmt::Display for SensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SensorError::DeviceLost => write!(f, "sensor device lost"),
            SensorError::Timeout => write!(f, "sensor read timed out"),
            SensorError::ReadFailed(msg) => write!(f, "sensor read failed: {}", msg),
        }
    }
}

impl std::error::Error for SensorError {}

/// Trait for any device that produces timestamped samples.
pub trait SensorStream {
    type Sample;

    /// Metadata about this sensor.
    fn info(&self) -> &SensorInfo;

    /// Read the next sample (blocking).
    fn read(&mut self) -> Result<Stamped<Self::Sample>, SensorError>;

    /// Whether the sensor is still open and producing data.
    fn is_open(&self) -> bool;
}
