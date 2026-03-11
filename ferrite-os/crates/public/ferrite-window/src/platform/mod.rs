//! Platform-specific window backends.
//!
//! Each backend implements the same interface; the correct one is selected
//! at compile time via `cfg` attributes.

#[cfg(all(unix, not(target_os = "macos")))]
mod x11;

#[cfg(windows)]
mod win32;

// Re-export the active platform backend as `PlatformWindow`.

#[cfg(all(unix, not(target_os = "macos")))]
pub use self::x11::X11Window as PlatformWindow;

#[cfg(windows)]
pub use self::win32::Win32Window as PlatformWindow;

#[cfg(target_os = "macos")]
compile_error!("macOS backend not yet implemented — contributions welcome");
