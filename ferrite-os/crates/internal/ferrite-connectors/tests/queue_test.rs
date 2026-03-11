use ferrite_connectors::queue::BoundedQueue;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn basic_push_pop() {
    let q = BoundedQueue::new(10);
    q.push(1);
    q.push(2);
    assert_eq!(q.len(), 2);
    assert_eq!(q.pop_blocking(), 1);
    assert_eq!(q.pop_blocking(), 2);
    assert!(q.is_empty());
}

#[test]
fn drop_oldest_on_overflow() {
    let q = BoundedQueue::new(3);
    q.push(1);
    q.push(2);
    q.push(3);
    assert_eq!(q.len(), 3);

    // This should drop item 1 (oldest)
    let dropped = q.push(4);
    assert!(dropped, "should report a dropped item");
    assert_eq!(q.len(), 3);

    // Items should be 2, 3, 4
    assert_eq!(q.pop_blocking(), 2);
    assert_eq!(q.pop_blocking(), 3);
    assert_eq!(q.pop_blocking(), 4);
}

#[test]
fn drain_batch_returns_up_to_max() {
    let q = BoundedQueue::new(100);
    for i in 0..10 {
        q.push(i);
    }
    let batch = q.drain_batch(5);
    assert_eq!(batch, vec![0, 1, 2, 3, 4]);
    assert_eq!(q.len(), 5);
}

#[test]
fn drain_batch_empty_queue() {
    let q: BoundedQueue<i32> = BoundedQueue::new(10);
    let batch = q.drain_batch(5);
    assert!(batch.is_empty());
}

#[test]
fn pop_blocking_waits_for_item() {
    let q = Arc::new(BoundedQueue::new(10));
    let q2 = q.clone();

    let handle = thread::spawn(move || q2.pop_blocking());

    // Small delay to ensure the consumer is blocking
    thread::sleep(Duration::from_millis(50));
    q.push(42);

    let val = handle.join().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn concurrent_producers_single_consumer() {
    let q = Arc::new(BoundedQueue::new(1000));
    let mut producers = Vec::new();
    let items_per_producer = 100;
    let num_producers = 4;

    for _ in 0..num_producers {
        let q2 = q.clone();
        producers.push(thread::spawn(move || {
            for i in 0..items_per_producer {
                q2.push(i);
            }
        }));
    }

    for p in producers {
        p.join().unwrap();
    }

    // All items should be in the queue (capacity 1000, total 400)
    assert_eq!(q.len(), num_producers * items_per_producer);
}

#[test]
fn capacity_one_works() {
    let q = BoundedQueue::new(1);
    q.push(1);
    let dropped = q.push(2);
    assert!(dropped);
    assert_eq!(q.pop_blocking(), 2);
}
