//! `rpython/rlib/rgc.py` `_make_sure_does_not_move` through the public hooks.
//! A separate integration-test process isolates the active-collector hooks.

use majit_gc::{
    _make_sure_does_not_move, ActiveGcGuardHooks, GcAllocator, TypeInfo, collector::MiniMarkGC,
    gc_sync, set_active_collect_generation, set_active_gc_guard_hooks, shadow_stack,
};
use majit_ir::GcRef;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static GENERATIONS: Mutex<Vec<i64>> = Mutex::new(Vec::new());
static STOP_AT: AtomicUsize = AtomicUsize::new(usize::MAX);

fn record_collection(generation: i64) -> i64 {
    GENERATIONS.lock().unwrap().push(generation);
    // A moving collector rewrites roots, not the caller's unregistered copy.
    shadow_stack::walk_roots(|p| p.0 += 16);
    0
}

fn can_still_move(p: GcRef) -> bool {
    p.0 < STOP_AT.load(Ordering::Relaxed)
}

fn root_count() -> usize {
    let mut count = 0;
    shadow_stack::walk_roots(|_| count += 1);
    count
}

#[test]
fn rgc_make_sure_does_not_move_preserves_the_collection_and_root_contract() {
    let mut null = GcRef::NULL;
    assert!(std::panic::catch_unwind(move || _make_sure_does_not_move(&mut null)).is_err());

    // No active collector is non-moving, matching can_move's public fallback.
    let mut stable = GcRef(0x1000);
    assert_eq!(_make_sure_does_not_move(&mut stable), Ok(true));
    assert_eq!(stable, GcRef(0x1000));
    assert_eq!(root_count(), 0);

    set_active_collect_generation(Some(record_collection));
    set_active_gc_guard_hooks(ActiveGcGuardHooks {
        can_move: Some(can_still_move),
        is_pinned: Some(|_| true),
        ..Default::default()
    });
    assert_eq!(_make_sure_does_not_move(&mut stable), Ok(false));
    assert!(GENERATIONS.lock().unwrap().is_empty());
    assert_eq!(root_count(), 0);

    set_active_gc_guard_hooks(ActiveGcGuardHooks {
        can_move: Some(can_still_move),
        ..Default::default()
    });
    STOP_AT.store(0x1030, Ordering::Relaxed);
    assert_eq!(_make_sure_does_not_move(&mut stable), Ok(true));
    assert_eq!(stable, GcRef(0x1030));
    assert_eq!(*GENERATIONS.lock().unwrap(), [-1, 0, 1]);
    assert_eq!(root_count(), 0);

    // A collector that always moves tries exactly -1..=6 before failing.
    GENERATIONS.lock().unwrap().clear();
    STOP_AT.store(usize::MAX, Ordering::Relaxed);
    assert_eq!(
        _make_sure_does_not_move(&mut stable),
        Err("can't make object non-movable!")
    );
    assert_eq!(*GENERATIONS.lock().unwrap(), [-1, 0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(stable, GcRef(0x10b0));
    assert_eq!(root_count(), 0);

    set_active_collect_generation(Some(|generation| {
        record_collection(generation);
        panic!("collection failed after moving the object");
    }));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            _make_sure_does_not_move(&mut stable)
        }))
        .is_err()
    );
    assert_eq!(
        stable,
        GcRef(0x10c0),
        "unwinding must reload the moved value"
    );
    assert_eq!(root_count(), 0, "unwinding must release the live root");

    // Exercise real promotion, not only mocked movement: the value is live
    // solely through the helper's root while the bare minor evacuates it.
    let mut gc = MiniMarkGC::new();
    let tid = gc.register_type(TypeInfo::simple(16));
    gc_sync::store_singleton(Box::new(gc));
    shadow_stack::register_mutator();
    gc_sync::register_thread();
    set_active_gc_guard_hooks(ActiveGcGuardHooks {
        can_move: Some(|p| gc_sync::gc_op(|gc| gc.can_move(p))),
        is_pinned: Some(|p| gc_sync::gc_op(|gc| gc.is_pinned(p))),
        ..Default::default()
    });
    set_active_collect_generation(Some(|generation| {
        gc_sync::gc_op(|gc| gc.collect_generation(generation))
    }));
    let mut p = gc_sync::gc_op(|gc| gc.alloc_with_type(tid, 16));
    unsafe { *(p.0 as *mut usize) = 0xabcdef };
    let original = p;
    let before = gc_sync::gc_op(|gc| gc.collection_counts());
    assert_eq!(_make_sure_does_not_move(&mut p), Ok(true));
    assert_ne!(p, original);
    assert!(!majit_gc::can_move(p));
    assert_eq!(unsafe { *(p.0 as *const usize) }, 0xabcdef);
    let after = gc_sync::gc_op(|gc| gc.collection_counts());
    assert_eq!(after.0, before.0 + 1);
    assert_eq!(after.1, before.1, "generation -1 must not run a major");
    assert_eq!(_make_sure_does_not_move(&mut p), Ok(true));
    assert_eq!(gc_sync::gc_op(|gc| gc.collection_counts()), after);
    assert_eq!(root_count(), 0);

    let mut pinned = gc_sync::gc_op(|gc| gc.alloc_with_type(tid, 16));
    assert!(gc_sync::gc_op(|gc| gc.pin(pinned)));
    let pinned_addr = pinned;
    assert_eq!(_make_sure_does_not_move(&mut pinned), Ok(false));
    assert_eq!(pinned, pinned_addr);
    assert_eq!(gc_sync::gc_op(|gc| gc.collection_counts()), after);
    gc_sync::gc_op(|gc| gc.unpin(pinned));
    assert_eq!(_make_sure_does_not_move(&mut pinned), Ok(true));
    assert_ne!(pinned, pinned_addr);
    assert_eq!(root_count(), 0);

    set_active_collect_generation(None);
    set_active_gc_guard_hooks(ActiveGcGuardHooks::default());
    shadow_stack::unregister_mutator();
    gc_sync::unregister_thread();
}
