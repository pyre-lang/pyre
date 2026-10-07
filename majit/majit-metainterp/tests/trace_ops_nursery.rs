//! `opencoder.py` `Trace._ops` is one nursery `GcArray(Char)`.
//! Growth allocates a new array (`Trace._double_ops`). A minor collection
//! between appends must leave every byte readable through `TraceIterator`.

use std::sync::Arc;

use majit_backend::jitframe::jitframe_type_info;
use majit_gc::GcAllocator;
use majit_gc::collector::MiniMarkGC;
use majit_gc::gc_sync;
use majit_metainterp::MetaInterpStaticData;
use majit_metainterp::opencoder::{INIT_SIZE, Trace, register_trace_ops_gc_type};

#[test]
fn trace_ops_survive_double_and_minor_collection() {
    gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
    gc_sync::register_thread();
    gc_sync::gc_op(|gc| {
        let jitframe_tid = gc.register_type(jitframe_type_info());
        gc.set_jitframe_type_id(jitframe_tid);
        register_trace_ops_gc_type(gc);
    });
    majit_metainterp::install_active_backend_gc_standalone();

    let mut trace = Trace::new(0, Arc::new(MetaInterpStaticData::new()));
    let total = INIT_SIZE + 64;
    for i in 0..total {
        if i == INIT_SIZE / 2 || i == INIT_SIZE + 8 {
            gc_sync::gc_op(|gc| gc.do_collect_nursery());
        }
        trace.append_byte((i & 0xff) as u8);
    }
    let expected: Vec<u8> = (0..total).map(|i| (i & 0xff) as u8).collect();
    assert!(trace.ops_bytes().len() > INIT_SIZE);
    assert_eq!(trace.bytes_read_by_iterator(), expected);
    let (minors, _) = gc_sync::gc_op(|gc| gc.collection_counts());
    assert!(minors >= 2, "minors={minors}");

    // Same collector. A second test would race on the process-global GC.
    let obj_tid = gc_sync::gc_op(|gc| gc.register_type(majit_gc::trace::TypeInfo::simple(8)));
    let mut pools = Trace::new(0, Arc::new(MetaInterpStaticData::new()));
    pools.fill_pools_past_reserve_for_test(obj_tid);
}
