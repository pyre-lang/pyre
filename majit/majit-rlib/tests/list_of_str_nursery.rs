//! Nursery survival of `jit_ll_list_str_from_rust_vec`.
//!
//! Its own binary: installing `ACTIVE_ALLOC_NURSERY_TYPED` writes one
//! process-global cell, and the lib tests take the immortal path only
//! while that cell is empty.

use majit_gc::TypeInfo;
use majit_gc::collector::MiniMarkGC;
use majit_ir::GcRef;
use majit_rlib::lltypesystem::rlist::{
    LIST_OF_STR_GCARRAY_ITEM_SIZE, LIST_OF_STR_GCARRAY_ITEMS_OFFSET,
    LIST_OF_STR_GCARRAY_LEN_OFFSET, jit_ll_list_str_from_rust_vec, set_gc_list_of_str_gc_type_id,
};
use majit_rlib::lltypesystem::rstr::{STR_CHARS_OFFSET, STR_LEN_OFFSET};

/// Presence is the signal `gc_allocator_installed` reads. With no per-thread
/// box, `alloc_nursery_typed` takes `standalone_alloc_nursery_typed` and
/// does not call this hook.
fn unused_nursery_hook(_type_id: u32, _payload_size: usize) -> GcRef {
    GcRef(0)
}

fn list_len(payload: *const u8) -> usize {
    unsafe { *payload.add(LIST_OF_STR_GCARRAY_LEN_OFFSET).cast::<usize>() }
}

fn list_item(payload: *const u8, index: usize) -> usize {
    unsafe {
        *payload
            .add(LIST_OF_STR_GCARRAY_ITEMS_OFFSET)
            .cast::<usize>()
            .add(index)
    }
}

fn str_bytes(addr: usize) -> Vec<u8> {
    let s = addr as *const u8;
    let len = unsafe { *s.add(STR_LEN_OFFSET).cast::<usize>() };
    unsafe { std::slice::from_raw_parts(s.add(STR_CHARS_OFFSET), len).to_vec() }
}

#[test]
fn jit_ll_list_str_from_rust_vec_items_survive_a_minor_collection() {
    let mut gc = MiniMarkGC::new();
    // `mallocstr` treats str tid 0 as unpublished. A placeholder keeps both
    // live slots off that sentinel; the list slot itself may be 0.
    let _placeholder = gc.register_type(TypeInfo::simple(std::mem::size_of::<usize>()));
    let list_tid = gc.register_type(TypeInfo::varsize(
        LIST_OF_STR_GCARRAY_ITEMS_OFFSET,
        LIST_OF_STR_GCARRAY_ITEM_SIZE,
        LIST_OF_STR_GCARRAY_LEN_OFFSET,
        true,
        Vec::new(),
    ));
    let str_tid = gc.register_type(TypeInfo::varsize(
        STR_CHARS_OFFSET + 1,
        1,
        STR_LEN_OFFSET,
        false,
        Vec::new(),
    ));
    assert_ne!(str_tid, 0);
    let list_info = gc.types.get(list_tid);
    assert!(list_info.items_have_gc_ptrs);
    assert_eq!(list_info.var_gc_ptr_offsets, vec![0]);

    set_gc_list_of_str_gc_type_id(list_tid);
    majit_gc::set_lowlevel_str_type_id(str_tid);
    majit_gc::gc_sync::store_singleton(Box::new(gc));
    majit_gc::set_active_alloc_nursery_typed(Some(unused_nursery_hook));
    majit_gc::shadow_stack::register_mutator();
    majit_gc::gc_sync::register_thread();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut items = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let ptr = &mut items as *mut Vec<String>;
        let payload = jit_ll_list_str_from_rust_vec(ptr);
        std::mem::forget(items);
        assert!(!payload.is_null(), "declared tid must nursery-allocate");

        let mut root = GcRef(payload as usize);
        let expected = [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()];
        majit_gc::gc_sync::gc_op(|gc| {
            assert!(gc.is_in_nursery(root.0));
            assert_eq!(list_len(root.0 as *const u8), 3);
            for (index, text) in expected.iter().copied().enumerate() {
                let item = list_item(root.0 as *const u8, index);
                assert!(gc.is_in_nursery(item), "item {index} must be a nursery STR");
                assert_eq!(str_bytes(item), text);
            }
            unsafe { gc.roots.add(&mut root) };
        });
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        majit_gc::gc_sync::gc_op(|gc| {
            assert!(
                !gc.is_in_nursery(root.0),
                "the rooted array must leave the nursery"
            );
            assert_eq!(list_len(root.0 as *const u8), 3);
            for (index, text) in expected.iter().copied().enumerate() {
                let item = list_item(root.0 as *const u8, index);
                assert_ne!(item, 0);
                assert!(
                    !gc.is_in_nursery(item),
                    "item {index} must be forwarded with the array"
                );
                assert_eq!(str_bytes(item), text);
            }
        });
    }));
    majit_gc::gc_sync::gc_thread_die();
    match result {
        Ok(()) => {}
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
