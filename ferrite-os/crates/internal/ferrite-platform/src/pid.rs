//! Single-instance PID file management.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Guard that creates a PID file on construction and removes it on drop.
pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    /// Create (or reclaim) a PID file at `path`.
    ///
    /// Returns `AlreadyExists` if another live process holds the file.
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        if path.exists() {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(pid) = contents.trim().parse::<u32>() {
                    if is_process_alive(pid) {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            format!("Daemon already running (PID: {})", pid),
                        ));
                    }
                }
            }
            fs::remove_file(path)?;
        }

        let pid = current_pid();
        fs::write(path, format!("{}\n", pid))?;

        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

// ── Platform helpers ────────────────────────────────────────────────────

/// Get the current process ID.
pub fn current_pid() -> u32 {
    #[cfg(unix)]
    {
        unsafe { libc::getpid() as u32 }
    }
    #[cfg(windows)]
    {
        unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessId() }
    }
}

/// Check whether a process with the given PID is alive.
pub fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle == 0 {
                false
            } else {
                CloseHandle(handle);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_pid_is_nonzero() {
        assert!(current_pid() > 0);
    }

    #[test]
    fn current_process_is_alive() {
        assert!(is_process_alive(current_pid()));
    }

    #[test]
    fn bogus_pid_is_not_alive() {
        // PID 0 is kernel on Unix and invalid on Windows — should not appear alive
        // for a normal user process.  Use a very high PID that is unlikely to exist.
        assert!(!is_process_alive(4_000_000));
    }

    #[test]
    fn pid_file_create_and_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pid");
        {
            let _pf = PidFile::create(&path).unwrap();
            assert!(path.exists());
            let contents = std::fs::read_to_string(&path).unwrap();
            assert_eq!(contents.trim(), current_pid().to_string());
        }
        // Dropped — file removed
        assert!(!path.exists());
    }
}
