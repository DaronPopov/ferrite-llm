//! TTY detection and null-device path.

use std::io;

/// Check whether stdout is connected to a TTY / console.
pub fn stdout_is_tty() -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
    }
    #[cfg(windows)]
    {
        // On Windows, check via GetConsoleMode on the stdout handle.
        use windows_sys::Win32::System::Console::{GetConsoleMode, GetStdHandle, STD_OUTPUT_HANDLE};
        unsafe {
            let handle = GetStdHandle(STD_OUTPUT_HANDLE);
            let mut mode = 0u32;
            GetConsoleMode(handle, &mut mode) != 0
        }
    }
}

/// Platform path to the null device.
///
/// `/dev/null` on Unix, `NUL` on Windows.
pub fn null_device_path() -> &'static str {
    #[cfg(unix)]
    {
        "/dev/null"
    }
    #[cfg(windows)]
    {
        "NUL"
    }
}

/// Redirect C-level stdout and stderr to the null device.
/// Returns a [`StdioGuard`] that can restore the original file descriptors.
///
/// This is used by the TUI to suppress GPU runtime printf output.
pub fn steal_stdio() -> io::Result<StdioGuard> {
    #[cfg(unix)]
    {
        use std::os::unix::io::{AsRawFd, FromRawFd};

        unsafe {
            let saved_stdout = libc::dup(libc::STDOUT_FILENO);
            if saved_stdout < 0 {
                return Err(io::Error::last_os_error());
            }

            let devnull = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(null_device_path())?;

            libc::dup2(devnull.as_raw_fd(), libc::STDOUT_FILENO);
            libc::dup2(devnull.as_raw_fd(), libc::STDERR_FILENO);

            let tty_file = std::fs::File::from_raw_fd(saved_stdout);
            Ok(StdioGuard {
                tty_file: Some(tty_file),
            })
        }
    }
    #[cfg(windows)]
    {
        // On Windows the TUI can suppress console output differently.
        // For now, return a no-op guard.
        Ok(StdioGuard { tty_file: None })
    }
}

/// Guard returned by [`steal_stdio`].  Holds a dup'd copy of the original
/// stdout fd so ratatui can still render to the real terminal.
pub struct StdioGuard {
    tty_file: Option<std::fs::File>,
}

impl StdioGuard {
    /// Borrow the saved TTY file for rendering.
    pub fn tty_file(&self) -> Option<&std::fs::File> {
        self.tty_file.as_ref()
    }

    /// Duplicate the saved TTY fd for use as a separate render backend.
    #[cfg(unix)]
    pub fn dup_tty_fd(&self) -> io::Result<std::fs::File> {
        use std::os::unix::io::{AsRawFd, FromRawFd};
        match self.tty_file {
            Some(ref f) => {
                let fd = unsafe { libc::dup(f.as_raw_fd()) };
                if fd < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(unsafe { std::fs::File::from_raw_fd(fd) })
                }
            }
            None => Err(io::Error::new(io::ErrorKind::Other, "no saved TTY")),
        }
    }

    #[cfg(windows)]
    pub fn dup_tty_fd(&self) -> io::Result<std::fs::File> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dup_tty_fd not available on Windows",
        ))
    }

    /// Restore the original stdout/stderr.
    pub fn restore(&self) {
        #[cfg(unix)]
        {
            if let Some(ref tty) = self.tty_file {
                use std::os::unix::io::AsRawFd;
                unsafe {
                    libc::dup2(tty.as_raw_fd(), libc::STDOUT_FILENO);
                    libc::dup2(tty.as_raw_fd(), libc::STDERR_FILENO);
                }
            }
        }
        // Windows no-op (stdio was not stolen).
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_device_exists() {
        assert!(std::path::Path::new(null_device_path()).exists());
    }

    #[test]
    fn stdout_is_tty_returns_bool() {
        // Just ensure it doesn't panic; actual value depends on test runner.
        let _ = stdout_is_tty();
    }
}
