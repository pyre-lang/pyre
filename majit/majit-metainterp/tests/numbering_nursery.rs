//! `resumecode.py` `NUMBERING` is a nursery `GcArray` of `UCHAR`.
//! A minor between `create_numbering` and the next read must leave
//! `numb.code` intact when the payload address is a walked slot.

use majit_backend::jitframe::jitframe_type_info;
use majit_gc::GcAllocator;
use majit_gc::collector::MiniMarkGC;
use majit_gc::gc_sync;
use majit_gc::shadow_stack::MutatorExtraAreaGuard;
use majit_ir::GcRef;
use majit_ir::resumecode::{NumberingRef, Writer, unpack_numbering};
use majit_metainterp::opencoder::register_trace_ops_gc_type;

unsafe fn walk_numbering(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
    let numb = unsafe { &*(data as *const NumberingRef) };
    numb.visit_gc(visitor);
}

fn install() {
    gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
    majit_gc::shadow_stack::register_mutator();
    gc_sync::register_thread();
    gc_sync::gc_op(|gc| {
        let jitframe_tid = gc.register_type(jitframe_type_info());
        gc.set_jitframe_type_id(jitframe_tid);
        register_trace_ops_gc_type(gc);
    });
    majit_metainterp::install_active_backend_gc_standalone();
}

fn fill(n: usize) -> (NumberingRef, Vec<i32>) {
    let items: Vec<i32> = (0..n).map(|i| (i % 50) as i32).collect();
    let mut w = Writer::new(items.len());
    for &item in &items {
        w.append_int(item as i64);
    }
    (w.create_numbering(), items)
}

fn roundtrip(n: usize, expect_move: bool) {
    let (numb, items) = fill(n);
    let before = numb.payload_addr();
    let guard = unsafe {
        MutatorExtraAreaGuard::new(
            walk_numbering,
            &numb as *const NumberingRef as *const (),
            "rd_consts",
        )
    };
    gc_sync::gc_op(|gc| gc.do_collect_nursery());
    assert_eq!(unpack_numbering(numb.as_slice()), items);
    if expect_move {
        assert_ne!(numb.payload_addr(), before, "a young NUMBERING must move");
    }
    drop(guard);
}

#[test]
fn numbering_survives_minor_and_oversized_alloc() {
    // The runner sets `PYPY_GC_NURSERY=65536` before this process starts.
    install();
    roundtrip(400, true);
    // Larger than the 64 KiB nursery: `malloc_varsize` births it old.
    roundtrip(80_000, false);
    let (minors, _) = gc_sync::gc_op(|gc| gc.collection_counts());
    assert!(minors >= 2, "minors={minors}");
}
