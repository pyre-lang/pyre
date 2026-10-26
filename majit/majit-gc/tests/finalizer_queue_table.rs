//! Per-collector `framework.py get_finalizer_queue_index` table.
//! A separate integration-test process isolates the singleton handle from
//! collector-local unit tests.

use majit_gc::{
    FinalizerQueueTag, GcAllocator, TypeInfo, collector::MiniMarkGC, gc_fq_next_dead,
    gc_get_finalizer_queue_index, set_active_finalizer_queue_table,
};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

static FQ: u8 = 0;
static POPPED: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_TABLE: AtomicPtr<majit_gc::FinalizerQueueTable> =
    AtomicPtr::new(std::ptr::null_mut());

fn published_table() -> *const majit_gc::FinalizerQueueTable {
    ACTIVE_TABLE.load(Ordering::Acquire)
}

fn trigger() {
    let tag = FinalizerQueueTag {
        fq: std::ptr::addr_of!(FQ) as usize,
        trigger,
    };
    let index = gc_get_finalizer_queue_index(tag);
    if gc_fq_next_dead(index).is_some() {
        POPPED.store(1, Ordering::Relaxed);
    }
}

/// Two separately constructed collectors number the same FinalizerQueueTag
/// independently (`framework.py finalizer_queue_indexes` on each `gcdata`).
#[test]
fn two_collectors_number_finalizer_queue_indexes_independently() {
    fn shared_trigger() {}
    fn dummy() {}
    static SHARED_FQ: u8 = 0;
    let tag = FinalizerQueueTag {
        fq: std::ptr::addr_of!(SHARED_FQ) as usize,
        trigger: shared_trigger,
    };
    let mut gc_a = MiniMarkGC::new();
    let mut gc_b = MiniMarkGC::new();
    let _ = gc_b
        .finalizer_queues
        .get_finalizer_queue_index(FinalizerQueueTag {
            fq: dummy as *const () as usize,
            trigger: dummy,
        });
    let index_a = gc_a.finalizer_queues.get_finalizer_queue_index(tag);
    let index_b = gc_b.finalizer_queues.get_finalizer_queue_index(tag);
    assert_ne!(index_a, index_b);
    let tid_a = gc_a.register_type(TypeInfo::simple(16));
    let tid_b = gc_b.register_type(TypeInfo::simple(16));
    let obj_a = GcAllocator::alloc_oldgen_typed(&mut gc_a, tid_a, 16);
    let obj_b = GcAllocator::alloc_oldgen_typed(&mut gc_b, tid_b, 16);
    GcAllocator::register_finalizer(&mut gc_a, index_a, obj_a);
    GcAllocator::register_finalizer(&mut gc_b, index_b, obj_b);

    gc_b.do_collect_full();
    assert_eq!(
        gc_a.finalizer_queues
            .finalizer_deque(index_a)
            .expect("handler")
            .len(),
        0
    );
    gc_a.do_collect_full();
    assert_eq!(
        gc_a.finalizer_queues
            .finalizer_deque(index_a)
            .expect("handler")
            .popleft(),
        Some(obj_a.0)
    );
}

/// A trigger that calls `gc_get_finalizer_queue_index` and `gc_fq_next_dead`
/// while `execute_finalizer_triggers` runs drains the queue without
/// re-entering the collector.
#[test]
fn trigger_looks_up_index_and_pops_without_collector_reentry() {
    POPPED.store(0, Ordering::Relaxed);
    let mut gc = MiniMarkGC::new();
    ACTIVE_TABLE.store(
        std::sync::Arc::as_ptr(&gc.finalizer_queues) as *mut majit_gc::FinalizerQueueTable,
        Ordering::Release,
    );
    set_active_finalizer_queue_table(Some(published_table));
    let tag = FinalizerQueueTag {
        fq: std::ptr::addr_of!(FQ) as usize,
        trigger,
    };
    let tid = gc.register_type(TypeInfo::simple(16));
    let obj = GcAllocator::alloc_oldgen_typed(&mut gc, tid, 16);
    let fq_index = gc_get_finalizer_queue_index(tag);
    GcAllocator::register_finalizer(&mut gc, fq_index, obj);
    gc.do_collect_oldgen_nonmoving();
    assert_eq!(POPPED.load(Ordering::Relaxed), 1);
    assert!(gc_fq_next_dead(fq_index).is_none());
    set_active_finalizer_queue_table(None);
    ACTIVE_TABLE.store(std::ptr::null_mut(), Ordering::Release);
}

/// `boehm.py BoehmGCTransformer.get_finalizer_queue_index`: the host-heap
/// descr still numbers a tag when no framework collector is installed.
#[test]
fn host_heap_numbers_finalizer_queue_indexes_without_a_collector() {
    fn dummy() {}
    static HOST_FQ: u8 = 0;
    std::thread::spawn(|| {
        let tag = FinalizerQueueTag {
            fq: std::ptr::addr_of!(HOST_FQ) as usize,
            trigger: dummy,
        };
        let index = gc_get_finalizer_queue_index(tag);
        assert_eq!(index, gc_get_finalizer_queue_index(tag));
        assert!(gc_fq_next_dead(index).is_none());
    })
    .join()
    .expect("host-heap finalizer queue thread");
}
