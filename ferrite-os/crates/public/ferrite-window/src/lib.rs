//! Cross-platform window system for Ferrite-OS.
//!
//! Uses the most basic native display drivers on each platform:
//! - **Linux**: X11 protocol via `x11rb`
//! - **Windows**: Win32 API via `windows-sys`
//! - **macOS**: Cocoa/AppKit (future)
//!
//! Provides a software framebuffer — you write RGBA pixels into a `&[u32]`
//! buffer and call `present()` to blit them to the screen. No OpenGL, Vulkan,
//! or GPU driver required for basic display.
//!
//! # Example
//!
//! ```no_run
//! use ferrite_window::{Window, WindowConfig};
//!
//! let mut window = Window::new(WindowConfig {
//!     title: "Ferrite".into(),
//!     width: 800,
//!     height: 600,
//!     resizable: true,
//! }).unwrap();
//!
//! let mut pixels = vec![0xFF_00_00_FFu32; 800 * 600]; // solid red RGBA
//!
//! while window.is_open() {
//!     for event in window.poll_events() {
//!         // handle events...
//!     }
//!     window.present(&pixels, 800, 600).unwrap();
//! }
//! ```

pub mod event;
mod platform;

pub use event::{Event, Modifiers, MouseButton};

use std::fmt;

/// Window creation configuration.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    /// Request fullscreen (borderless). The window manager removes decorations
    /// and expands the window to fill the screen. `width`/`height` are still
    /// used as the initial size hint; the WM may override them.
    pub fullscreen: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Ferrite".into(),
            width: 1920,
            height: 1080,
            resizable: true,
            fullscreen: true,
        }
    }
}

/// Window error.
#[derive(Debug)]
pub enum WindowError {
    /// Failed to connect to the display server.
    ConnectionFailed(String),
    /// Failed to create the window.
    CreationFailed(String),
    /// Pixel presentation failed.
    PresentFailed(String),
    /// Platform not supported.
    Unsupported(String),
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "display connection failed: {}", msg),
            Self::CreationFailed(msg) => write!(f, "window creation failed: {}", msg),
            Self::PresentFailed(msg) => write!(f, "present failed: {}", msg),
            Self::Unsupported(msg) => write!(f, "unsupported: {}", msg),
        }
    }
}

impl std::error::Error for WindowError {}

/// A cross-platform window.
///
/// Created via [`Window::new`]. Handles event dispatch and software
/// framebuffer presentation using the native OS display driver.
pub struct Window {
    inner: platform::PlatformWindow,
}

impl Window {
    /// Create a new window with the given configuration.
    pub fn new(config: WindowConfig) -> Result<Self, WindowError> {
        let inner = platform::PlatformWindow::new(config)?;
        Ok(Self { inner })
    }

    /// Poll for pending events. Returns all events since the last call.
    /// Non-blocking — returns an empty vec if no events are queued.
    pub fn poll_events(&mut self) -> Vec<Event> {
        self.inner.poll_events()
    }

    /// Blit a pixel buffer to the window.
    ///
    /// `pixels` is packed RGBA (`0xRRGGBBAA`) in row-major order.
    /// `width` and `height` are the dimensions of the pixel buffer.
    /// The buffer is scaled to fill the window if dimensions differ.
    pub fn present(&mut self, pixels: &[u32], width: u32, height: u32) -> Result<(), WindowError> {
        self.inner.present(pixels, width, height)
    }

    /// Current window dimensions in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.inner.size()
    }

    /// Whether the window is still open (not closed by user).
    pub fn is_open(&self) -> bool {
        self.inner.is_open()
    }

    /// Set the window title.
    pub fn set_title(&mut self, title: &str) {
        self.inner.set_title(title);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = WindowConfig::default();
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert!(cfg.resizable);
        assert!(cfg.fullscreen);
        assert_eq!(cfg.title, "Ferrite");
    }

    #[test]
    fn error_display() {
        let err = WindowError::ConnectionFailed("no display".into());
        let msg = err.to_string();
        assert!(msg.contains("no display"));
        assert!(msg.contains("connection"));
    }

    #[test]
    fn event_debug() {
        let e = Event::Close;
        let _ = format!("{:?}", e);

        let e = Event::Resize { width: 100, height: 200 };
        let s = format!("{:?}", e);
        assert!(s.contains("100"));

        let e = Event::KeyDown {
            keycode: 42,
            modifiers: Modifiers { shift: true, ctrl: false, alt: false },
        };
        let s = format!("{:?}", e);
        assert!(s.contains("42"));
    }

    #[test]
    fn mouse_button_eq() {
        assert_eq!(MouseButton::Left, MouseButton::Left);
        assert_ne!(MouseButton::Left, MouseButton::Right);
    }
}
