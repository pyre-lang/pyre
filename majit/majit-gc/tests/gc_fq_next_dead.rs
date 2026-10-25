//! `framework.py ll_next_dead` pops from inside `execute_finalizers`' trigger.
//! A separate integration-test process isolates the process-global STW
//! singleton from collector-local unit tests.

use majit_gc::{
    FinalizerQueueTag, GcAllocator, TypeInfo, collector::MiniMarkGC, gc_fq_next_dead, gc_sync,
    shadow_stack,
};
use std::sync::atomic::{AtomicUsize, Ordering};

static FQ: u8 = 0;
static POPPED: AtomicUsize = AtomicUsize::new(0);

fn trigger() {
    let tag = FinalizerQueueTag {
        fq: std::ptr::addr_of!(FQ) as usize,
        trigger,
    };
    let index = majit_gc::gc_get_finalizer_queue_index(tag);
    if gc_fq_next_dead(index).is_some() {
        POPPED.store(1, Ordering::Relaxed);
    }
}

fn with_singleton_mutator<R>(f: impl FnOnce() -> R) -> R {
    gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
    shadow_stack::register_mutator();
    gc_sync::register_thread();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    gc_sync::gc_thread_die();
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// A trigger that calls `gc_fq_next_dead` while `execute_finalizer_triggers`
/// runs pops the dead object; the queue is then empty.
#[test]
fn gc_fq_next_dead_pops_from_inside_execute_finalizer_triggers() {
    POPPED.store(0, Ordering::Relaxed);
    with_singleton_mutator(|| {
        let tag = FinalizerQueueTag {
            fq: std::ptr::addr_of!(FQ) as usize,
            trigger,
        };
        let fq_index = majit_gc::gc_get_finalizer_queue_index(tag);
        gc_sync::gc_op(|gc| {
            let tid = gc.register_type(TypeInfo::simple(16));
            let obj = GcAllocator::alloc_oldgen_typed(gc, tid, 16);
            GcAllocator::register_finalizer(gc, fq_index, obj);
        });
        gc_sync::gc_op(|gc| gc.do_collect_oldgen_nonmoving());
        assert_eq!(POPPED.load(Ordering::Relaxed), 1);
        assert!(gc_fq_next_dead(fq_index).is_none());
    });
}
