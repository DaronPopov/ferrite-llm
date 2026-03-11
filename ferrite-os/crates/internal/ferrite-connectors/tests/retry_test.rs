use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

// We test retry_with_backoff with small base_ms to keep tests fast.

#[test]
fn retry_succeeds_on_first_try() {
    let result = ferrite_connectors::retry::retry_with_backoff(3, 10, || Ok::<_, String>(42));
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn retry_succeeds_after_failures() {
    let counter = Arc::new(AtomicU32::new(0));
    let c = counter.clone();
    let result = ferrite_connectors::retry::retry_with_backoff(5, 10, move || {
        let n = c.fetch_add(1, Ordering::SeqCst);
        if n < 2 {
            Err("not yet")
        } else {
            Ok("done")
        }
    });
    assert_eq!(result.unwrap(), "done");
    assert_eq!(counter.load(Ordering::SeqCst), 3); // 2 failures + 1 success
}

#[test]
fn retry_exhausts_max_retries() {
    let counter = Arc::new(AtomicU32::new(0));
    let c = counter.clone();
    let result = ferrite_connectors::retry::retry_with_backoff(3, 10, move || {
        c.fetch_add(1, Ordering::SeqCst);
        Err::<(), _>("always fails")
    });
    assert!(result.is_err());
    // 1 initial attempt + 3 retries = 4 total calls
    assert_eq!(counter.load(Ordering::SeqCst), 4);
}

#[test]
fn retry_backoff_increases_delay() {
    let start = Instant::now();
    let counter = Arc::new(AtomicU32::new(0));
    let c = counter.clone();
    let _ = ferrite_connectors::retry::retry_with_backoff(2, 20, move || {
        c.fetch_add(1, Ordering::SeqCst);
        Err::<(), _>("fail")
    });
    let elapsed = start.elapsed();
    // With base 20ms and 2 retries:
    // retry 1: ~40ms + jitter, retry 2: ~80ms + jitter
    // Total should be at least ~100ms but we allow some slack
    assert!(
        elapsed.as_millis() >= 50,
        "backoff should introduce noticeable delay, got {}ms",
        elapsed.as_millis()
    );
}
