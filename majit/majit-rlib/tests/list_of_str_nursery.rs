//! Nursery survival of `jit_ll_list_str_from_rust_vec`.
//!
//! Its own binary: installing `ACTIVE_ALLOC_NURSERY_TYPED` writes one
//! process-global cell, and the lib tests take the immortal path only
//! while that cell is empty.
//!
//! `store_singleton` keeps the first collector. Every test body holds
//! `COLLECTOR_LOCK` across `register_thread` .. `gc_thread_die`: a second
//! registered mutator makes `stw_required` true, and `register_thread`
//! waits on the GIL the collector holds until that mutator quiesces.

use std::sync::{Mutex, Once};

use majit_gc::TypeInfo;
use majit_gc::collector::MiniMarkGC;
use majit_ir::GcRef;
use majit_rlib::lltypesystem::rlist::{
    OBJECT_ITEMS_ITEM_SIZE, OBJECT_ITEMS_LEN_OFFSET, OBJECT_ITEMS_OFFSET,
    gc_object_items_array_gc_type_id, jit_ll_list_str_from_rust_vec,
    set_gc_object_items_array_gc_type_id,
};
use majit_rlib::lltypesystem::rstr::{STR_CHARS_OFFSET, STR_LEN_OFFSET};

/// Explicit `GcConfig`: `Default` reads `PYPY_GC_NURSERY` / `PYPY_GC_DEBUG`.
/// `debug: 0` keeps one nursery range so `publish_singleton_nursery` leaves
/// the `malloc_fast` bump armed.
const NURSERY_SIZE: usize = 16 * 1024;
/// `total_size >=` this is born old (`try_alloc_with_type_no_collect_body`).
const LARGE_OBJECT_THRESHOLD: usize = 1024;

static COLLECTOR_LOCK: Mutex<()> = Mutex::new(());
static INSTALL: Once = Once::new();

/// Presence is the signal `gc_allocator_installed` reads. With no per-thread
/// box, `alloc_nursery_typed` takes `standalone_alloc_nursery_typed` and
/// does not call this hook.
fn unused_nursery_hook(_type_id: u32, _payload_size: usize) -> GcRef {
    GcRef(0)
}

fn install_collector() {
    INSTALL.call_once(|| {
        let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
            nursery_size: NURSERY_SIZE,
            debug_tiny_nursery: None,
            large_object_threshold: LARGE_OBJECT_THRESHOLD,
            card_page_indices: 128,
            debug: 0,
            gc_nursery_debug: false,
            taggedpointers: false,
        });
        // The list-of-str payload is the object-items ARRAY. Register it
        // before the str row: `mallocstr` treats str tid 0 as unpublished.
        let list_tid = gc.register_type(TypeInfo::varsize(
            OBJECT_ITEMS_OFFSET,
            OBJECT_ITEMS_ITEM_SIZE,
            OBJECT_ITEMS_LEN_OFFSET,
            true,
            Vec::new(),
        ));
        set_gc_object_items_array_gc_type_id(list_tid);
        let str_tid = gc.register_type(TypeInfo::varsize(
            STR_CHARS_OFFSET + 1,
            1,
            STR_LEN_OFFSET,
            false,
            Vec::new(),
        ));
        assert_ne!(str_tid, 0);
        majit_gc::set_lowlevel_str_type_id(str_tid);
        majit_gc::gc_sync::store_singleton(Box::new(gc));
        majit_gc::set_active_alloc_nursery_typed(Some(unused_nursery_hook));
    });
}

fn with_collector(body: impl FnOnce()) {
    let _guard = COLLECTOR_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    install_collector();
    majit_gc::shadow_stack::register_mutator();
    majit_gc::gc_sync::register_thread();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    majit_gc::gc_sync::gc_thread_die();
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn list_len(payload: *const u8) -> usize {
    unsafe { *payload.add(OBJECT_ITEMS_LEN_OFFSET).cast::<usize>() }
}

fn list_item(payload: *const u8, index: usize) -> usize {
    unsafe { *payload.add(OBJECT_ITEMS_OFFSET).cast::<usize>().add(index) }
}

fn str_bytes(addr: usize) -> Vec<u8> {
    let s = addr as *const u8;
    let len = unsafe { *s.add(STR_LEN_OFFSET).cast::<usize>() };
    unsafe { std::slice::from_raw_parts(s.add(STR_CHARS_OFFSET), len).to_vec() }
}

fn header_tid(addr: usize) -> u32 {
    unsafe { (*majit_gc::header::header_of(addr)).type_id() }
}

fn minor_collections() -> usize {
    majit_gc::gc_sync::gc_op(|gc| gc.minor_collections)
}

/// Drop leftover young garbage so the next conversion starts at `nursery_free`.
fn reset_nursery() {
    majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
}

fn nursery_total(raw: usize) -> usize {
    let raw = raw.max(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE);
    (raw + majit_gc::header::MEMORY_ALIGNMENT - 1) & !(majit_gc::header::MEMORY_ALIGNMENT - 1)
}

fn array_total(len: usize) -> usize {
    nursery_total(
        majit_gc::header::GcHeader::SIZE + OBJECT_ITEMS_OFFSET + OBJECT_ITEMS_ITEM_SIZE * len,
    )
}

fn str_nursery_bytes(len: usize) -> usize {
    nursery_total(majit_gc::header::GcHeader::SIZE + STR_CHARS_OFFSET + len + 1)
}

/// `ptr::read` moves the header out. `forget` is the caller's half of that
/// contract, including when the helper drops its copy and returns null.
fn convert(items: Vec<String>) -> *mut u8 {
    let mut items = items;
    let payload = jit_ll_list_str_from_rust_vec(&mut items);
    std::mem::forget(items);
    assert!(!payload.is_null(), "object-items allocation must succeed");
    payload
}

/// A stack slot registered with `RootSet` for one collection. `Drop` removes
/// it so a later test's minor walk does not read a dead frame.
struct NurseryRoot<'a>(&'a mut GcRef);

impl<'a> NurseryRoot<'a> {
    fn track(slot: &'a mut GcRef) -> Self {
        majit_gc::gc_sync::gc_op(|gc| unsafe { gc.roots.add(slot) });
        NurseryRoot(slot)
    }

    fn addr(&self) -> usize {
        self.0.0
    }
}

impl Drop for NurseryRoot<'_> {
    fn drop(&mut self) {
        majit_gc::gc_sync::gc_op(|gc| {
            gc.roots.remove(self.0);
        });
    }
}

fn assert_slot(payload: *const u8, index: usize, text: &str) {
    let item = list_item(payload, index);
    assert_ne!(item, 0, "slot {index}");
    assert_eq!(
        header_tid(item),
        majit_gc::lowlevel_str_type_id(),
        "slot {index}"
    );
    assert_eq!(str_bytes(item), text.as_bytes(), "slot {index}");
}

#[test]
fn jit_ll_list_str_from_rust_vec_items_survive_a_minor_collection() {
    with_collector(|| {
        let list_tid = gc_object_items_array_gc_type_id();
        assert_ne!(majit_gc::lowlevel_str_type_id(), 0);
        majit_gc::gc_sync::gc_op(|gc| {
            let list_info = gc.types.get(list_tid);
            assert!(list_info.items_have_gc_ptrs);
            assert_eq!(list_info.var_gc_ptr_offsets, vec![0]);
        });
        reset_nursery();

        let payload = convert(vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
        ]);
        assert_eq!(header_tid(payload as usize), list_tid);
        let expected = ["one", "two", "three"];
        let mut root = GcRef(payload as usize);
        majit_gc::gc_sync::gc_op(|gc| {
            assert!(gc.is_in_nursery(root.0));
            assert_eq!(list_len(root.0 as *const u8), 3);
            for (index, text) in expected.iter().copied().enumerate() {
                let item = list_item(root.0 as *const u8, index);
                assert!(gc.is_in_nursery(item), "item {index} must be a nursery STR");
                assert_slot(root.0 as *const u8, index, text);
            }
        });
        let tracked = NurseryRoot::track(&mut root);
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        majit_gc::gc_sync::gc_op(|gc| {
            let addr = tracked.addr();
            assert!(
                !gc.is_in_nursery(addr),
                "the rooted array must leave the nursery"
            );
            assert_eq!(header_tid(addr), list_tid);
            assert_eq!(list_len(addr as *const u8), 3);
            for (index, text) in expected.iter().copied().enumerate() {
                let item = list_item(addr as *const u8, index);
                assert!(
                    !gc.is_in_nursery(item),
                    "item {index} must be forwarded with the array"
                );
                assert_slot(addr as *const u8, index, text);
            }
        });
    });
}

/// Strings large enough that `mallocstr_rooted` fills the nursery after the
/// array is already live. `gct_fv_gc_malloc` `push_roots` / `pop_roots` must
/// hand back the moved array; the young tail is stored into that copy.
#[test]
fn jit_ll_list_str_from_rust_vec_roots_the_array_across_minor_collections() {
    const COUNT: usize = 64;
    const NBYTES: usize = 256;
    assert!(
        array_total(COUNT) < LARGE_OBJECT_THRESHOLD,
        "the array must start in the nursery"
    );
    assert!(
        array_total(COUNT) + COUNT * str_nursery_bytes(NBYTES) > NURSERY_SIZE,
        "the strings must overflow the nursery mid-loop"
    );

    with_collector(|| {
        reset_nursery();
        let before = minor_collections();
        let items: Vec<String> = (0..COUNT).map(|index| padded(index, NBYTES)).collect();
        let payload = convert(items);
        let after = minor_collections();
        assert!(
            after > before,
            "mallocstr_rooted must collect while the array is live (before {before}, after {after})"
        );
        assert!(
            !majit_gc::gc_sync::gc_op(|gc| gc.is_in_nursery(payload as usize)),
            "a rooted young array is promoted by the mid-loop collection"
        );
        assert_eq!(
            header_tid(payload as usize),
            gc_object_items_array_gc_type_id()
        );

        let mut young = 0;
        let mut old = 0;
        majit_gc::gc_sync::gc_op(|gc| {
            assert_eq!(list_len(payload), COUNT);
            for index in 0..COUNT {
                let text = padded(index, NBYTES);
                assert_slot(payload, index, &text);
                let item = list_item(payload, index);
                if gc.is_in_nursery(item) {
                    young += 1;
                } else {
                    old += 1;
                }
            }
        });
        assert!(
            old > 0,
            "the prefix stored before the collection must be promoted"
        );
        assert!(
            young > 0,
            "the tail allocated after the collection stays young"
        );

        let mut root = GcRef(payload as usize);
        let tracked = NurseryRoot::track(&mut root);
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        majit_gc::gc_sync::gc_op(|gc| {
            let addr = tracked.addr();
            assert!(!gc.is_in_nursery(addr));
            assert_eq!(list_len(addr as *const u8), COUNT);
            for index in 0..COUNT {
                let text = padded(index, NBYTES);
                let item = list_item(addr as *const u8, index);
                assert!(
                    !gc.is_in_nursery(item),
                    "item {index} must be promoted with the remembered array"
                );
                assert_slot(addr as *const u8, index, &text);
            }
        });
    });
}

/// The array's `total_size` misses the nursery (`alloc_nursery_typed` spills
/// old). The strings stay young. `transform_generic_set` /
/// `write_barrier_from_array` has to remember that old array, or the next
/// minor leaves the slots pointing into the rewound nursery. A large object
/// is born without `GCFLAG_HAS_CARDS`; the barrier's no-card arm is
/// `remember_young_pointer`.
#[test]
fn jit_ll_list_str_from_rust_vec_write_barrier_keeps_young_strings_in_a_large_array() {
    let count = (LARGE_OBJECT_THRESHOLD - majit_gc::header::GcHeader::SIZE - OBJECT_ITEMS_OFFSET)
        / OBJECT_ITEMS_ITEM_SIZE;
    assert!(array_total(count) >= LARGE_OBJECT_THRESHOLD);
    assert!(array_total(count - 1) < LARGE_OBJECT_THRESHOLD);
    assert!(count * str_nursery_bytes(4) < NURSERY_SIZE);

    with_collector(|| {
        reset_nursery();
        let before = minor_collections();
        let items: Vec<String> = (0..count).map(short_distinct).collect();
        let payload = convert(items);
        let during = minor_collections();
        assert_eq!(
            before, during,
            "the array is born old and the young strings fit, so the loop must not collect"
        );
        assert_eq!(
            header_tid(payload as usize),
            gc_object_items_array_gc_type_id()
        );
        majit_gc::gc_sync::gc_op(|gc| {
            assert!(!gc.is_in_nursery(payload as usize), "large array is old");
            assert_eq!(list_len(payload), count);
            for index in 0..count {
                let text = short_distinct(index);
                let item = list_item(payload, index);
                assert!(gc.is_in_nursery(item), "item {index} is still young");
                assert_slot(payload, index, &text);
            }
        });

        let mut root = GcRef(payload as usize);
        let tracked = NurseryRoot::track(&mut root);
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        majit_gc::gc_sync::gc_op(|gc| {
            let addr = tracked.addr();
            assert!(!gc.is_in_nursery(addr));
            assert_eq!(list_len(addr as *const u8), count);
            for index in 0..count {
                let text = short_distinct(index);
                let item = list_item(addr as *const u8, index);
                assert!(
                    !gc.is_in_nursery(item),
                    "item {index} must leave the nursery; a missing barrier leaves a dangling nursery address"
                );
                assert_slot(addr as *const u8, index, &text);
            }
        });
    });
}

fn padded(index: usize, len: usize) -> String {
    let head = format!("{index:08x}");
    assert!(head.len() <= len);
    let mut text = String::with_capacity(len);
    text.push_str(&head);
    text.extend(std::iter::repeat('x').take(len - head.len()));
    text
}

fn short_distinct(index: usize) -> String {
    format!("{index:04}")
}
