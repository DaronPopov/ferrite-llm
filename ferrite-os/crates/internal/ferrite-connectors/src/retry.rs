use rand::Rng;
use std::thread;
use std::time::Duration;

/// Retry `f` up to `max_retries` times with exponential backoff + jitter.
/// Returns the first `Ok` result or the last `Err` after all retries are exhausted.
pub fn retry_with_backoff<T, E, F>(max_retries: u32, base_ms: u64, mut f: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut attempt = 0u32;
    loop {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => {
                attempt += 1;
                if attempt > max_retries {
                    return Err(e);
                }
                let backoff = base_ms.saturating_mul(1u64 << attempt.min(12));
                let jitter = rand::thread_rng().gen_range(0..=backoff / 2);
                let sleep_ms = backoff + jitter;
                tracing::warn!(attempt, max_retries, sleep_ms, "retrying after backoff");
                thread::sleep(Duration::from_millis(sleep_ms));
            }
        }
    }
}
