use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

pub struct BoundedQueue<T> {
    inner: Mutex<VecDeque<T>>,
    capacity: usize,
    not_empty: Condvar,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "queue capacity must be > 0");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            not_empty: Condvar::new(),
        }
    }

    /// Push an item. If at capacity, drops the oldest item and logs a warning.
    /// Returns `true` if an item was dropped.
    pub fn push(&self, item: T) -> bool {
        let mut queue = self.inner.lock().unwrap();
        let dropped = if queue.len() >= self.capacity {
            queue.pop_front();
            true
        } else {
            false
        };
        queue.push_back(item);
        self.not_empty.notify_one();
        dropped
    }

    /// Block until an item is available, then pop it.
    pub fn pop_blocking(&self) -> T {
        let mut queue = self.inner.lock().unwrap();
        loop {
            if let Some(item) = queue.pop_front() {
                return item;
            }
            queue = self.not_empty.wait(queue).unwrap();
        }
    }

    /// Non-blocking drain of up to `max` items.
    pub fn drain_batch(&self, max: usize) -> Vec<T> {
        let mut queue = self.inner.lock().unwrap();
        let n = max.min(queue.len());
        queue.drain(..n).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
