//! `resumecode.py` `NUMBERING` is a nursery `GcArray` of `UCHAR`.
//! A minor between `create_numbering` and the next read must leave
//! `numb.code` intact. The young list is the minor edge
//! (`incminimark.py` `collect_oldrefs_to_nursery`). A major before the
//! descr is a `compile.py` `ResumeGuardDescr.rd_numb` holder still
//! marks the payload.

use majit_backend::jitframe::jitframe_type_info;
use majit_gc::GcAllocator;
use majit_gc::collector::MiniMarkGC;
use majit_gc::gc_sync;
use majit_ir::resumecode::{NumberingRef, Writer, unpack_numbering};
use majit_metainterp::opencoder::register_trace_ops_gc_type;

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

fn survive(numb: &NumberingRef, before: usize, expect_move: bool) {
    gc_sync::gc_op(|gc| gc.do_collect_nursery());
    let after_minor = numb.payload_addr();
    if expect_move {
        assert_ne!(after_minor, before, "a young NUMBERING must move");
    } else {
        assert_eq!(
            after_minor, before,
            "a young rawmalloc NUMBERING does not move"
        );
    }
    // No holder yet. The major must still keep the payload.
    gc_sync::gc_op(|gc| gc.do_collect_full());
    assert_eq!(
        numb.payload_addr(),
        after_minor,
        "an old NUMBERING does not move"
    );
}

#[test]
fn numbering_survives_minor_and_oversized_alloc() {
    install();
    let (numb, items) = fill(400);
    let before = numb.payload_addr();
    survive(&numb, before, true);
    assert_eq!(unpack_numbering(numb.as_slice()), items);

    // Past `malloc_varsize`'s large-object cutoff. `external_malloc`
    // (`alloc_young=True`) births the array young and non-moving. The
    // young-list walk has to visit it or this minor frees it.
    let bytes = vec![0x11u8; 200_000];
    let big = NumberingRef::from_bytes(&bytes);
    let before = big.payload_addr();
    survive(&big, before, false);
    assert_eq!(big.as_slice(), bytes.as_slice());

    let (minors, _) = gc_sync::gc_op(|gc| gc.collection_counts());
    assert!(minors >= 2, "minors={minors}");
}
