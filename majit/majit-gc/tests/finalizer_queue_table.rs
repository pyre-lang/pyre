//! Process-wide `framework.py get_finalizer_queue_index` table.
//! A separate integration-test process isolates the append-only handlers
//! from collector-local unit tests.

use majit_gc::{
    FinalizerQueueTag, GcAllocator, TypeInfo, collector::MiniMarkGC, gc_fq_next_dead,
    gc_get_finalizer_queue_index,
};
use std::sync::atomic::{AtomicUsize, Ordering};

static FQ: u8 = 0;
static POPPED: AtomicUsize = AtomicUsize::new(0);

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

/// Two separately constructed collectors see the same index for the same
/// FinalizerQueueTag (`framework.py finalizer_queue_indexes`).
#[test]
fn two_collectors_share_finalizer_queue_index() {
    fn shared_trigger() {}
    static SHARED_FQ: u8 = 0;
    let tag = FinalizerQueueTag {
        fq: std::ptr::addr_of!(SHARED_FQ) as usize,
        trigger: shared_trigger,
    };
    let mut gc_a = MiniMarkGC::new();
    let mut gc_b = MiniMarkGC::new();
    let index_a = gc_get_finalizer_queue_index(tag);
    let index_b = gc_get_finalizer_queue_index(tag);
    assert_eq!(index_a, index_b);
    let tid_a = gc_a.register_type(TypeInfo::simple(16));
    let tid_b = gc_b.register_type(TypeInfo::simple(16));
    let obj_a = GcAllocator::alloc_oldgen_typed(&mut gc_a, tid_a, 16);
    let obj_b = GcAllocator::alloc_oldgen_typed(&mut gc_b, tid_b, 16);
    GcAllocator::register_finalizer(&mut gc_a, index_a, obj_a);
    GcAllocator::register_finalizer(&mut gc_b, index_b, obj_b);
}

/// A trigger that calls `gc_get_finalizer_queue_index` and `gc_fq_next_dead`
/// while `execute_finalizer_triggers` runs drains the queue without
/// re-entering the collector.
#[test]
fn trigger_looks_up_index_and_pops_without_collector_reentry() {
    POPPED.store(0, Ordering::Relaxed);
    let mut gc = MiniMarkGC::new();
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
}
