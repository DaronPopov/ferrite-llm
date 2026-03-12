use std::collections::VecDeque;
use std::sync::Mutex;

/// Single-threaded ring buffer for intra-tick stage interconnects.
pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize,
    tail: usize,
    count: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buf: (0..capacity).map(|_| None).collect(),
            head: 0,
            tail: 0,
            count: 0,
            capacity,
        }
    }

    /// Enqueue an item. Returns `false` if the buffer is full.
    pub fn push(&mut self, item: T) -> bool {
        if self.count == self.capacity {
            return false;
        }
        self.buf[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;
        true
    }

    /// Peek at the oldest item without removing it.
    pub fn peek(&self) -> Option<&T> {
        if self.count == 0 {
            return None;
        }
        self.buf[self.head].as_ref()
    }

    /// Dequeue the oldest item.
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }
        let item = self.buf[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        item
    }

    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        for slot in &mut self.buf {
            *slot = None;
        }
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
}

/// Thread-safe ring buffer for capture thread → pipeline hand-off.
///
/// Uses overwrite policy: when full, the oldest item is dropped to make room.
pub struct SharedRing<T> {
    inner: Mutex<VecDeque<T>>,
    capacity: usize,
}

impl<T> SharedRing<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SharedRing capacity must be > 0");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Push an item. If full, drops the oldest item first (overwrite policy).
    /// Returns `true` if no item was dropped, `false` if oldest was evicted.
    pub fn push(&self, item: T) -> bool {
        let mut q = self.inner.lock().unwrap();
        if q.len() >= self.capacity {
            q.pop_front();
            q.push_back(item);
            false
        } else {
            q.push_back(item);
            true
        }
    }

    /// Pop the oldest item.
    pub fn pop(&self) -> Option<T> {
        let mut q = self.inner.lock().unwrap();
        q.pop_front()
    }

    pub fn len(&self) -> usize {
        let q = self.inner.lock().unwrap();
        q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_pop() {
        let mut rb = RingBuffer::new(3);
        assert!(rb.is_empty());
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(rb.is_full());
        assert!(!rb.push(4)); // full
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
        assert!(rb.is_empty());
    }

    #[test]
    fn ring_buffer_wrap_around() {
        let mut rb = RingBuffer::new(2);
        rb.push(1);
        rb.push(2);
        assert_eq!(rb.pop(), Some(1));
        rb.push(3);
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn ring_buffer_clear() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn shared_ring_overwrite_policy() {
        let sr = SharedRing::new(2);
        assert!(sr.push(1));  // no eviction
        assert!(sr.push(2));  // no eviction
        assert!(!sr.push(3)); // evicts 1
        assert_eq!(sr.pop(), Some(2));
        assert_eq!(sr.pop(), Some(3));
        assert_eq!(sr.pop(), None);
    }

    #[test]
    fn shared_ring_len() {
        let sr = SharedRing::new(3);
        assert_eq!(sr.len(), 0);
        sr.push(10);
        sr.push(20);
        assert_eq!(sr.len(), 2);
        sr.pop();
        assert_eq!(sr.len(), 1);
    }
}
