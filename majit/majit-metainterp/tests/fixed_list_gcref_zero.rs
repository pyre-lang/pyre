//! `FixedSignedList` R banks are `GcArray(GCREF)`. `malloc_fast` writes
//! only the tid; `ll_newlist` / `gct_zero_gc_pointers_inside` null every
//! item so a collection that reaches the array does not trace leftover
//! nursery words as pointers.

use majit_backend::jitframe::jitframe_type_info;
use majit_gc::GcAllocator;
use majit_gc::collector::MiniMarkGC;
use majit_gc::gc_sync;
use majit_gc::header::GcHeader;
use majit_gc::trace::TypeInfo;
use majit_ir::GcRef;
use majit_metainterp::install_active_backend_gc_standalone;
use majit_metainterp::opencoder::{FixedSignedList, ListRootGuard, register_trace_ops_gc_type};
use std::sync::{Mutex, MutexGuard};

static TEST_LOCK: Mutex<()> = Mutex::new(());

/// One MiniMark mutator at a time. Two `register_thread` owners leave
/// `do_collect_nursery` waiting for a sibling that is itself inside
/// `gc_op` (`gc_sync.register_thread` STW census).
struct InstallGuard {
    _lock: MutexGuard<'static, ()>,
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        majit_gc::shadow_stack::unregister_mutator();
        gc_sync::unregister_thread();
    }
}

fn install() -> InstallGuard {
    let lock = TEST_LOCK.lock().unwrap();
    gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
    majit_gc::shadow_stack::register_mutator();
    gc_sync::register_thread();
    gc_sync::gc_op(|gc| {
        let jitframe_tid = gc.register_type(jitframe_type_info());
        gc.set_jitframe_type_id(jitframe_tid);
        register_trace_ops_gc_type(gc);
    });
    install_active_backend_gc_standalone();
    InstallGuard { _lock: lock }
}

#[test]
fn r_kind_gcref_items_are_null_after_nursery_reuse() {
    let _install = install();
    gc_sync::gc_op(|gc| {
        let tid = gc.register_type(TypeInfo::simple(64));
        for _ in 0..128 {
            let obj = gc.alloc_nursery_typed(tid, 64);
            if obj.0 != 0 {
                unsafe {
                    std::ptr::write_bytes(obj.0 as *mut u8, 0x11, 64);
                }
            }
        }
        gc.do_collect_nursery();
    });

    let mut list = FixedSignedList::alloc_with_roots(4, true, &mut []);
    assert_eq!(
        &list[..],
        &[0, 0, 0, 0],
        "gct_zero_gc_pointers_inside: leftover nursery words must not be live GCREF items"
    );
    let root = majit_gc::shadow_stack::OwnerRootGuard::new(list.as_gcref());
    gc_sync::gc_op(|gc| gc.do_collect_nursery());
    list.reload(root.get());
    assert_eq!(&list[..], &[0, 0, 0, 0]);
}

#[test]
fn list_root_guard_reloads_array_after_minor() {
    // `framework.py` `pop_roots`: a `FixedSignedList` copy is not the
    // shadow-stack slot. A minor at `alloc_with_roots` (blackhole
    // `read_list_from`) forwards the array; the next use must
    // `reload(guard.get(i))`.
    let _install = install();
    const MAGIC: u64 = 0xD1A6_05C1_11;
    let obj = gc_sync::gc_op(|gc| {
        let tid = gc.register_type(TypeInfo::simple(64));
        let obj = gc.alloc_nursery_typed(tid, 64);
        assert_ne!(obj.0, 0);
        unsafe {
            std::ptr::write((obj.0 as *mut u8).add(GcHeader::SIZE) as *mut u64, MAGIC);
        }
        obj
    });

    let mut list = FixedSignedList::alloc_with_roots(1, true, &mut []);
    list.fill_from_code(&[obj.0 as i64], &[0], 0);
    let stale_list = list.as_gcref();
    let stale_item = obj;
    let guard = ListRootGuard::push(&[list.as_gcref()]);
    gc_sync::gc_op(|gc| gc.do_collect_nursery());

    let forwarded_list = guard.get(0);
    assert_ne!(forwarded_list.0, 0);
    assert_ne!(
        forwarded_list.0, stale_list.0,
        "rooted nursery array must leave the nursery"
    );
    list.reload(forwarded_list);
    let item = GcRef(list[0] as usize);
    assert_ne!(item.0, 0);
    assert_ne!(
        item.0, stale_item.0,
        "GCREF item must be forwarded with the array"
    );
    let magic = unsafe { std::ptr::read((item.0 as *mut u8).add(GcHeader::SIZE) as *const u64) };
    assert_eq!(magic, MAGIC);
}

#[test]
fn list_root_guard_indexes_null_slots() {
    // Skip-null `push` shifted later slots: `get(1)` would name `c`.
    // `walk_stack_root` already skips a null `content`.
    let _install = install();
    let a = FixedSignedList::alloc_with_roots(1, true, &mut []);
    let c = FixedSignedList::alloc_with_roots(1, true, &mut []);
    let stale_a = a.as_gcref();
    let stale_c = c.as_gcref();
    let guard = ListRootGuard::push(&[stale_a, GcRef(0), stale_c]);
    assert_eq!(guard.get(1).0, 0);
    gc_sync::gc_op(|gc| gc.do_collect_nursery());
    assert_ne!(guard.get(0).0, 0);
    assert_ne!(guard.get(0).0, stale_a.0);
    assert_eq!(guard.get(1).0, 0);
    assert_ne!(guard.get(2).0, 0);
    assert_ne!(guard.get(2).0, stale_c.0);
}
