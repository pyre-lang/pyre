//! Raw `PyObject *` mirrors and their links -- PyPy `cpyext/pyobject.py`.
//!
//! A C extension never sees a [`pyre_object::PyObject`].  It sees a mirror at a
//! fixed address carrying a reference count and an `ob_pyre_link` to the moving
//! interpreter object. Ordinary objects use rawrefcount's P-link; a
//! `W_PyCTypeObject` realised from an extension type uses its O-link and keeps
//! the mirror in `W_TypeObject._cpy_ref` (`rpython/rlib/rawrefcount.py`,
//! `W_PyCTypeObject._cpyext_attach_pyobj`).
//!
//! A mirror's block is as large as its type asks for: `sizeof(PyObject)` for an
//! ordinary object, `sizeof(PyTypeObject)` for a type, `tp_basicsize` for an
//! instance of a C-defined type.  That is upstream's `make_typedescr(basestruct
//! =...)`, and it is why the allocation is a byte block rather than a Rust
//! struct.
//!
//! # Ownership
//!
//! The link is a `rawrefcount` P-link, and the collector owns it: a mirror
//! whose count is exactly [`REFCNT_FROM_PYPY`] is referenced by nothing but the
//! link, so the collector is free to let the interpreter object die and queue
//! the mirror on its dead list.  [`drain_dead`] is what then runs the mirror's
//! deallocator.  Nothing in this module roots an interpreter object.
//!
//! The rule's other half is that a count *above* the link share does root the
//! object — `rawrefcount.rst`'s "mark 'p' as surviving, as well as all its
//! dependencies".  Read on its own that keeps a reference cycle running through
//! a C reference alive for good, which is where it stands upstream; what a
//! collection here also has is the block's own references, reported by
//! [`super::gc`] out of its `tp_traverse`, so that a reference from inside the
//! cycle is told apart from one from outside it.

use super::ForkMutex;
use super::typeobject::{CPyTypeObject, CPyVarObject};
use majit_gc::rawrefcount;
use pyre_object::{PY_NULL, PyObjectRef};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::c_char;
use std::hash::BuildHasherDefault;

/// The large ownership contribution held by the linked pyre object —
/// `rawrefcount.REFCNT_FROM_PYPY`.  Ordinary C references are added above it by
/// the public `Py_INCREF`/`Py_DECREF` macros.
///
/// The collector reads it too, and decides on it, so the constant is defined
/// where the algorithm is.
pub use rawrefcount::REFCNT_FROM_PYPY;

/// The count an immortal mirror starts at.
///
/// Type mirrors and the singletons are borrowed by every consumer and never
/// handed out with an owning reference, so their count must never fall back to
/// [`REFCNT_FROM_PYPY`] and free them.
pub use rawrefcount::REFCNT_IMMORTAL;

/// C-visible `PyObject`, matching PyPy's `parse/cpyext_object.h` shape.
#[repr(C)]
pub struct CPyObject {
    pub ob_refcnt: isize,
    pub ob_pyre_link: PyObjectRef,
    pub ob_pyre_pad: usize,
    pub ob_type: *mut CPyTypeObject,
}

/// The alignment every mirror block is allocated at.
///
/// A C-defined type's `tp_basicsize` covers fields the extension declared, so
/// the block has to satisfy the strictest alignment a C compiler would give a
/// `malloc` result — `max_align_t`, which is 16 on every target this feature
/// builds for.
pub(super) const BLOCK_ALIGN: usize = 16;

fn block_layout(size: usize) -> Layout {
    Layout::from_size_align(size, BLOCK_ALIGN).expect("a mirror block size is a small constant")
}

/// The collector reads a mirror's first two words as its own
/// [`rawrefcount::PyObjHeader`], so the two spellings have to agree.
const _: () = {
    assert!(size_of::<CPyObject>() >= size_of::<rawrefcount::PyObjHeader>());
    assert!(align_of::<CPyObject>() == align_of::<rawrefcount::PyObjHeader>());
    assert!(std::mem::offset_of!(CPyObject, ob_refcnt) == 0);
    assert!(std::mem::offset_of!(CPyObject, ob_pyre_link) == size_of::<isize>());
};

/// How each mirror block was allocated: the byte size for one this layer owns,
/// and 0 for a block it does not — a `static` in this crate or in the loaded
/// extension.
///
/// The `generation` distinguishes this block from a later one the allocator
/// hands out at the same address, which an address alone cannot: a `tp_dealloc`
/// that frees its block and allocates during the same call can be given the
/// address it just released.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Block {
    size: usize,
    generation: u64,
    /// The C mirror of `W_Root`'s weakref lifeline.  PyPy keeps the lifeline
    /// on the live `W_Root`; pyre's rawrefcount teardown can outlast that root,
    /// so the per-mirror allocation record owns it until `tp_dealloc` calls
    /// `PyObject_ClearWeakRefs`.
    weakref_lifeline: usize,
}

/// The P list itself belongs to the collector, which is where every question
/// about a link is now asked; this survives only because a mirror block's size
/// is not something the collector has any reason to know.  Its keys are mirror
/// addresses, which never move, so it needs no collection-time maintenance.
type BlockSizes = super::address_table::AddressMap<Block>;
static BLOCK_SIZES: ForkMutex<BlockSizes> =
    ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// Stamps every block with the order it was handed out in.
static BLOCK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Enter a block this layer handed out, under a generation of its own.
fn record_block(address: usize, size: usize) {
    let generation = BLOCK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BLOCK_SIZES.lock().insert(
        address,
        Block {
            size,
            generation,
            weakref_lifeline: 0,
        },
    );
}

fn block_at(address: usize) -> Option<Block> {
    BLOCK_SIZES.lock().get(&address).copied()
}

/// Every entered mirror block, for C-owned fields that are not covered by
/// `tp_traverse`. The allocation record is already the owner of this census;
/// callers filter the copied stable addresses by their concrete C type.
pub(super) fn entered_blocks() -> Vec<usize> {
    BLOCK_SIZES.lock().keys().copied().collect()
}

/// Whether the block at `address` is one this layer handed out.
///
/// Only such a block arrives with a header worth reading: [`allocate_raw`]
/// clears one.  An extension that allocates its own does not have to --
/// `_cffi_backend.c allocate_owning_object` takes its blocks from plain
/// `malloc` -- so what sits in the header words of one of those means
/// nothing until it is stamped.
pub(super) fn is_own_block(address: usize) -> bool {
    block_at(address).is_some()
}

pub(super) unsafe fn after_fork_child() {
    unsafe {
        BLOCK_SIZES.reinit_after_fork();
        BORROWED.reinit_after_fork();
        BYTE_CACHE.reinit_after_fork();
        ITEM_ARRAYS.reinit_after_fork();
    }
}

/// The mirror of an interpreter object's type.
///
/// A type a C extension defined already has one — its own `PyTypeObject`
/// static, entered by `PyType_Ready` — and every other type gets a synthesized
/// block of the same shape so that `Py_TYPE(x)->tp_name` reads something.
pub fn type_mirror(w_obj: PyObjectRef) -> *mut CPyTypeObject {
    match crate::typedef::r#type(w_obj) {
        Some(w_type) => ensure_mirror(w_type.as_ptr()) as *mut CPyTypeObject,
        None => std::ptr::null_mut(),
    }
}

/// `w_obj`'s mirror, allocated on first demand.
///
/// A type gets a `PyHeapTypeObject`-shaped block by whichever route reaches it
/// first: `Py_TYPE(x)->tp_basicsize` is read off any type mirror, so a type may
/// never receive the plain `PyObject`-sized block a non-type receives — a
/// `PyModule_AddObject` of a class would otherwise decide the shape.
pub(super) fn ensure_mirror(w_obj: PyObjectRef) -> *mut CPyObject {
    // pyobject.py `as_pyobj` / `create_ref`: the interpreter object must stay
    // live across the allocation. A nursery collection inside `type_mirror`
    // moves it; the address passed into `attach` would otherwise be the
    // pre-move word.
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    let mut w_obj = roots.pin_root(w_obj);
    if unsafe { pyre_object::is_type(w_obj) } {
        let direct = unsafe { pyre_object::w_type_get_cpy_ref(w_obj) };
        if !direct.is_null() {
            return direct as *mut CPyObject;
        }
    }
    let existing = majit_gc::gc_rawrefcount_from_obj(majit_ir::GcRef(w_obj as usize));
    if existing != 0 {
        return existing as *mut CPyObject;
    }
    if unsafe { pyre_object::is_type(w_obj) } {
        // A type is an instance of its own metatype, so resolving `ob_type`
        // before the mirror is entered would recurse forever on `type`; the
        // second call finds this mirror in the table and terminates.
        w_obj = roots.get(obj_slot);
        let mirror = attach(
            w_obj,
            REFCNT_FROM_PYPY,
            std::ptr::null_mut(),
            size_of::<super::typeobject::CPyHeapTypeObject>(),
        ) as *mut CPyTypeObject;
        // `typeobject.py type_alloc`.  The suites are the blocks the
        // `PyHeapTypeObject` above declares, so an extension that reads one
        // off `ht_type` and one off `tp_as_number` reads the same words.
        unsafe {
            let heap = mirror as *mut super::typeobject::CPyHeapTypeObject;
            (*mirror).tp_as_async = &raw mut (*heap).as_async;
            (*mirror).tp_as_number = &raw mut (*heap).as_number;
            (*mirror).tp_as_sequence = &raw mut (*heap).as_sequence;
            (*mirror).tp_as_mapping = &raw mut (*heap).as_mapping;
            (*mirror).tp_as_buffer = &raw mut (*heap).as_buffer;
        }
        // Read the link into a local before each allocating call. A
        // closure that borrows `mirror` cannot live across `set_ob_type`.
        let w_type = unsafe { (*(mirror as *mut CPyObject)).ob_pyre_link };
        super::typeobject::describe_interpreter_type(mirror, w_type);
        // `typeobject.py` `type_setup`: the metatype is referenced from
        // here only when it is itself a heap type.
        let w_type = unsafe { (*(mirror as *mut CPyObject)).ob_pyre_link };
        let of_type = type_mirror(w_type);
        unsafe { set_ob_type(&raw mut (*mirror).ob_base.ob_base, of_type) };
        // Only the startup table has a reason to defer this: every base a
        // mirror reached from here can name already has its own mirror, or
        // gets one from this same call a level down.
        let w_type = unsafe { (*(mirror as *mut CPyObject)).ob_pyre_link };
        super::typeobject::finish_interpreter_type(mirror, w_type);
        return mirror as *mut CPyObject;
    }
    w_obj = roots.get(obj_slot);
    let ob_type = type_mirror(w_obj);
    w_obj = roots.get(obj_slot);
    // `allocate`'s `size += itemcount * itemsize`: a tuple's items are part of
    // its block, which is what lets `PyTuple_GET_ITEM` read one as a field.
    let size = mirror_size(ob_type) + super::tupleobject::item_bytes(w_obj, ob_type);
    w_obj = roots.get(obj_slot);
    let raw = attach(w_obj, REFCNT_FROM_PYPY, ob_type, size);
    // Each fill allocates, so the object is read back through the mirror
    // before the next one rather than kept in a local: the block's address
    // does not move and the object's does, and the link is what the collector
    // updates.
    let linked = || unsafe { (*raw).ob_pyre_link };
    unsafe { super::typeobject::stamp_ob_size(raw, linked(), size) };
    // The types whose blocks carry fields past the header; every other block
    // this runtime hands out is exactly its header.
    super::frameobject::attach(raw, linked());
    super::sliceobject::attach(raw, linked());
    super::pyerrors::attach(raw, linked());
    super::cdatetime::attach(raw, linked());
    super::complexobject::attach(raw, linked());
    super::methodobject::attach(raw, linked());
    super::structobject::attach(raw, linked());
    super::typeobject::descriptor_attach(raw, linked());
    raw
}

/// `pyobject.py:91-93` — a block references its type's mirror only when that
/// type is a heap type.
///
/// A static type's mirror outlives every block of it, so counting the
/// references to it would be counting to a number nothing reads;
/// [`release_heap_type`] is the other half.
pub(super) fn own_heap_type(ob_type: *mut CPyTypeObject) {
    if super::typeobject::is_heap_type(ob_type) {
        unsafe { incref(&raw mut (*ob_type).ob_base.ob_base) };
    }
}

/// `object.py` — the `finally` half of [`own_heap_type`], run once the
/// block that held the reference is gone.
fn release_heap_type(ob_type: *mut CPyTypeObject) {
    if super::typeobject::is_heap_type(ob_type) {
        unsafe { decref(&raw mut (*ob_type).ob_base.ob_base) };
    }
}

/// Point a block at `ob_type`, moving the reference [`own_heap_type`] records
/// off whatever it named before.
///
/// # Safety
/// `raw` must be a live block whose `ob_type` is either null or a live mirror.
pub(super) unsafe fn set_ob_type(raw: *mut CPyObject, ob_type: *mut CPyTypeObject) {
    let previous = unsafe { (*raw).ob_type };
    unsafe { exchange_ob_type(raw, ob_type, previous) };
}

/// [`set_ob_type`] where the caller says what the block named before, which a
/// block that has never been an object does not: `Py_SET_TYPE` in
/// `object.c _PyObject_Init` stores the type rather than exchanging it.
///
/// # Safety
/// `raw` must be a live block, and `previous` either null or a live mirror.
pub(super) unsafe fn exchange_ob_type(
    raw: *mut CPyObject,
    ob_type: *mut CPyTypeObject,
    previous: *mut CPyTypeObject,
) {
    unsafe { (*raw).ob_type = ob_type };
    // The new reference is taken first: the two can name the same mirror, and
    // releasing it to the bare link share and back would trip the floor a
    // linked mirror's count has.
    own_heap_type(ob_type);
    release_heap_type(previous);
}

/// Allocate a zero-filled mirror block of `size` bytes and enter it into the
/// census and the identity table.
pub(super) fn attach(
    w_obj: PyObjectRef,
    refcnt: isize,
    ob_type: *mut CPyTypeObject,
    size: usize,
) -> *mut CPyObject {
    let size = size.max(size_of::<CPyObject>());
    let raw = unsafe { std::alloc::alloc_zeroed(block_layout(size)) } as *mut CPyObject;
    if raw.is_null() {
        std::alloc::handle_alloc_error(block_layout(size));
    }
    unsafe {
        (*raw).ob_refcnt = refcnt;
        (*raw).ob_pyre_link = w_obj;
        (*raw).ob_type = ob_type;
    }
    own_heap_type(ob_type);
    enter(w_obj, raw, size);
    raw
}

/// Enter a block this layer did not allocate — a `static` here or in the loaded
/// extension — as `w_obj`'s mirror.  Its header must already be filled.
pub(super) fn attach_foreign(w_obj: PyObjectRef, raw: *mut CPyObject) {
    unsafe { (*raw).ob_pyre_link = w_obj };
    enter(w_obj, raw, 0);
}

/// `W_PyCTypeObject._cpyext_attach_pyobj` — link an extension's existing
/// `PyTypeObject` in the O direction and retain its address directly on the
/// interpreter type.
///
/// `pyobject.py:track_reference` adds the link share before dispatching to the
/// object's attachment method. Unlike a P-link, an O-link has no
/// `rawrefcount.from_obj` entry: the direct `_cpy_ref` field is the lookup path.
pub(super) fn attach_foreign_pyobj(w_obj: PyObjectRef, raw: *mut CPyObject) {
    debug_assert!(unsafe { pyre_object::is_type(w_obj) });
    debug_assert!(
        unsafe { (*raw).ob_refcnt } < REFCNT_FROM_PYPY,
        "track_reference requires the unshared C refcount"
    );
    unsafe {
        (*raw).ob_refcnt += REFCNT_FROM_PYPY;
        (*raw).ob_pyre_link = w_obj;
        pyre_object::w_type_set_cpy_ref(w_obj, raw.cast());
    }
    record_block(raw as usize, 0);
    majit_gc::gc_rawrefcount_create_link_pyobj(majit_ir::GcRef(w_obj as usize), raw as usize);
}

/// Link a block that already exists to a fresh interpreter object.
///
/// `PyObject_Malloc` hands out an unlinked block and `PyObject_Init` is what
/// makes it an object, so the two halves of [`attach`] are separated here.  The
/// block keeps whatever [`allocate_raw`] recorded for it, which is what
/// [`free_block`] later releases it by.
pub(super) fn link_allocated(w_obj: PyObjectRef, raw: *mut CPyObject, refcnt: isize) {
    debug_assert!(
        refcnt >= REFCNT_FROM_PYPY,
        "a linked mirror carries the link share"
    );
    unsafe {
        (*raw).ob_refcnt = refcnt;
        (*raw).ob_pyre_link = w_obj;
    }
    majit_gc::gc_rawrefcount_create_link_pypy(majit_ir::GcRef(w_obj as usize), raw as usize);
}

/// `pyobject.py:track_reference` — hand the collector the P-link.
///
/// The mirror's header is already filled at this point, including the
/// [`REFCNT_FROM_PYPY`] share this link is worth, which is what
/// `track_reference`'s `c_ob_refcnt += REFCNT_FROM_PYPY` establishes.
fn enter(w_obj: PyObjectRef, raw: *mut CPyObject, size: usize) {
    debug_assert!(
        unsafe { (*raw).ob_refcnt } >= REFCNT_FROM_PYPY,
        "a linked mirror carries the link share"
    );
    if unsafe { pyre_object::is_type(w_obj) } {
        unsafe { pyre_object::w_type_set_cpy_ref(w_obj, raw.cast()) };
    }
    record_block(raw as usize, size);
    majit_gc::gc_rawrefcount_create_link_pypy(majit_ir::GcRef(w_obj as usize), raw as usize);
}

/// How large a mirror of `ob_type` has to be.
///
/// Everything pyre itself defines is exactly a `PyObject`, which is what
/// [`super::typeobject::describe_interpreter_type`] leaves `tp_basicsize` as;
/// an instance of a C-defined type carries that type's declared fields.
fn mirror_size(ob_type: *mut CPyTypeObject) -> usize {
    if ob_type.is_null() {
        return size_of::<CPyObject>();
    }
    let declared = unsafe { (*ob_type).tp_basicsize };
    if declared <= 0 {
        return size_of::<CPyObject>();
    }
    (declared as usize).max(size_of::<CPyObject>())
}

/// `pyobject.py:make_ref` — a new reference to `w_obj`'s mirror.
pub fn make_ref(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    let raw = ensure_mirror(w_obj);
    // Static builtin mirrors use the immortal sentinel.  Go through the same
    // incref operation the public API does so direct `_cpy_ref` storage does
    // not try to increment `isize::MAX`.
    unsafe { incref(raw) };
    raw
}

/// The mirror, built on demand, without taking a reference.
///
/// The block's only owner is the link, so it lives exactly as long as `w_obj`
/// does: the caller must be holding `w_obj` rooted, which every caller reaching
/// this from an interpreter-side operation is.
pub(super) fn borrow_mirror(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    ensure_mirror(w_obj)
}

/// `pyobject.py:as_pyobj` — the mirror without taking a reference.
///
/// The result is only valid while some other owner keeps the mirror alive, so
/// this is for reading a mirror the caller already owns; use [`make_ref`] for
/// anything handed out to C.
pub fn as_pyobj(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    // pyobject.py `as_pyobj`: `_cpyext_as_pyobj` first, then `create_ref`
    // when that lookup is empty. A type's `_cpy_ref` is the direct-storage
    // spelling of the same miss; returning the null field skipped the
    // create that every other object still takes.
    let existing = as_pyobj_existing(w_obj);
    if !existing.is_null() {
        return existing;
    }
    // `create_ref` needs the rawrefcount tables. Before `initialize()` they
    // do not exist, so a miss stays a miss — the same answer the type
    // `_cpy_ref` field already gave.
    if !majit_gc::gc_rawrefcount_enabled() {
        return std::ptr::null_mut();
    }
    ensure_mirror(w_obj)
}

/// pyobject.py `as_pyobj` lookup half: `_cpyext_as_pyobj` / `_cpy_ref`,
/// never `create_ref`.
pub(super) fn as_pyobj_existing(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { pyre_object::is_type(w_obj) } {
        let direct = unsafe { pyre_object::w_type_get_cpy_ref(w_obj) };
        if !direct.is_null() {
            return direct.cast();
        }
    }
    let existing = majit_gc::gc_rawrefcount_from_obj(majit_ir::GcRef(w_obj as usize));
    if existing != 0 {
        return existing as *mut CPyObject;
    }
    std::ptr::null_mut()
}

/// Attach `W_Root`'s weakref lifeline to an already-existing C mirror.
///
/// This is teardown state on the mirror itself, not a second identity table:
/// `_Py_Dealloc` replaces/clears `ob_pyre_link` before calling `tp_dealloc`,
/// while `Block` is the allocation record that deliberately survives through
/// that call.  The owned C reference keeps the lifeline reachable without
/// keeping its referent alive.
pub(super) fn remember_weakref_lifeline(w_obj: PyObjectRef, lifeline: PyObjectRef) {
    let raw = as_pyobj(w_obj);
    if raw.is_null() {
        return;
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[w_obj, lifeline]);
    let lifeline_raw = make_ref(pyre_object::gc_roots::shadow_stack_get(base + 1));
    let mut blocks = BLOCK_SIZES.lock();
    let Some(block) = blocks.get_mut(&(raw as usize)) else {
        drop(blocks);
        unsafe { decref(lifeline_raw) };
        return;
    };
    if block.weakref_lifeline == 0 {
        block.weakref_lifeline = lifeline_raw as usize;
    } else {
        drop(blocks);
        unsafe { decref(lifeline_raw) };
    }
}

/// Lifeline retained for a mirror whose deallocator is running.
pub(super) fn weakref_lifeline(raw: *mut CPyObject) -> Option<PyObjectRef> {
    let lifeline = block_at(raw as usize)?.weakref_lifeline as *mut CPyObject;
    if lifeline.is_null() {
        return None;
    }
    let object = unsafe { from_ref(lifeline) };
    (!object.is_null()).then_some(object)
}

/// The address parked in a dying mirror's link slot while its deallocator runs
/// — `pyobject.py w_marker_deallocating`, written by
/// `cpyext/src/object.c:77` before it calls `tp_dealloc`.
///
/// Upstream's marker is a real `W_Root` so that `from_ref` can compare against
/// it and report a specific fatal error.  Nothing here needs an object: the
/// address only has to be recognisable and never equal to a live one, and an
/// address inside this module's own image is both.
fn deallocating_marker() -> usize {
    static MARKER: u8 = 0;
    &raw const MARKER as usize
}

/// Whether `raw`'s deallocator is running -- its link slot holds the marker
/// [`dealloc`] parks there for the duration.
///
/// An entry point a deallocator is written to call is handed a mirror in
/// exactly this state, and has to tell it apart from a NULL: reading the
/// object out of it through [`from_ref`] answers null for both.
pub(super) fn is_deallocating(raw: *mut CPyObject) -> bool {
    !raw.is_null() && unsafe { (*raw).ob_pyre_link } as usize == deallocating_marker()
}

/// `pyobject.py:330-337` — build the interpreter object of a mirror that has
/// none yet.
///
/// Upstream reaches the `realize` of the mirror's own type descriptor; the two
/// forms pyre hands out unrealized are the `str`
/// [`super::unicodeobject::PyUnicode_New`] allocates and the `bytes`
/// [`super::bytesobject::PyBytes_FromStringAndSize`] allocates for a NULL
/// `text`, whose contents C is still writing when they are handed over.  Each
/// call below returns at once for a mirror that is not its own.
///
/// # Safety
/// `raw` must be null or a live mirror.  Building the object allocates, so no
/// unpinned interpreter object may be held across this.
pub(super) unsafe fn realize(raw: *mut CPyObject) {
    if raw.is_null() || !unsafe { (*raw).ob_pyre_link }.is_null() {
        return;
    }
    super::unicodeobject::realize_pending(raw);
    super::bytesobject::realize_pending(raw);
    super::frameobject::realize_pending(raw);
}

/// `pyobject.py:from_ref` — the interpreter object a mirror links to,
/// [`realize`]d first if it has none yet.
///
/// Null once the collector has cleared the link, and null while the mirror's
/// deallocator runs: `tp_dealloc` calling back into an API that would rebuild
/// the interpreter object is what upstream fatally errors on
/// (`pyobject.py`), because the object it would hand back is being
/// freed.  Returning null makes the caller take its own not-found path instead.
///
/// # Safety
/// `raw` must be null or a live mirror.  The result is only valid until the
/// next operation that can collect; re-read it through the mirror (or pin it
/// on the shadow stack) across anything that allocates.  Realizing allocates,
/// so converting several mirrors in a row goes through
/// [`super::object::arguments`], which realizes all of them first.
pub unsafe fn from_ref(raw: *mut CPyObject) -> PyObjectRef {
    if raw.is_null() {
        return PY_NULL;
    }
    let link = unsafe { (*raw).ob_pyre_link };
    if link.is_null() {
        unsafe { realize(raw) };
        return unsafe { (*raw).ob_pyre_link };
    }
    if link as usize == deallocating_marker() {
        return PY_NULL;
    }
    link
}

/// `pyobject.py incref`.
///
/// # Safety
/// `raw` must be null or a live mirror.
pub unsafe fn incref(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    // An immortal mirror's count carries no information, so moving it only
    // risks walking it out of its headroom.
    if unsafe { (*raw).ob_refcnt } >= REFCNT_IMMORTAL {
        return;
    }
    unsafe { (*raw).ob_refcnt += 1 };
}

/// `pyobject.py decref` — release one C reference, and deallocate at
/// exactly zero.
///
/// Zero, not the link share: while the link is in place the collector owns that
/// share, and a mirror the C side has finished with simply stops being a root.
/// It is [`drain_dead`] that later gives the share back and brings the count
/// here.
///
/// # Safety
/// `raw` must be null or a live mirror, and the caller must own the reference
/// being released.
pub unsafe fn decref(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    if unsafe { (*raw).ob_refcnt } >= REFCNT_IMMORTAL {
        return;
    }
    debug_assert!(
        unsafe { (*raw).ob_refcnt } > 0,
        "decref of a mirror with no references left"
    );
    debug_assert!(
        unsafe { (*raw).ob_pyre_link.is_null() } || unsafe { (*raw).ob_refcnt } > REFCNT_FROM_PYPY,
        "a linked mirror must not be decrefed below the link share"
    );
    unsafe { (*raw).ob_refcnt -= 1 };
    if unsafe { (*raw).ob_refcnt } == 0 {
        unsafe { dealloc(raw) };
    }
}

/// `cpyext/src/object.c:72-79 _Py_Dealloc` — mark the mirror deallocating, run
/// its type's `tp_dealloc`, then release what this layer owns for it.
///
/// # Safety
/// `raw` must be a live mirror whose count has fallen to zero.
unsafe fn dealloc(raw: *mut CPyObject) {
    let address = raw as usize;
    debug_assert_eq!(unsafe { (*raw).ob_refcnt }, 0, "dealloc below zero");
    // `object.py`, read before `tp_free` can hand the block back: the
    // reference this block owns in its type is released at the end.
    let ob_type = unsafe { (*raw).ob_type };
    // object.c:77, the same thing as `rawrefcount.mark_deallocating()`: the
    // link is dead but `tp_dealloc` may still ask for it.  The link itself is
    // kept, because a `tp_finalize` run from the deallocator may hand `self`
    // out and leave the object alive; the marker is only in force for the
    // duration of the deallocator.
    let link = unsafe { (*raw).ob_pyre_link };
    majit_gc::gc_rawrefcount_mark_deallocating(deallocating_marker(), address);
    // Everything this layer holds on the mirror's behalf goes before
    // `tp_dealloc` can free the block out from under it.  A borrow it owns may
    // be the last reference to its own container, so that recursion runs here.
    // Only a block some table keyed an entry under is walked.  `release_hold`
    // answers for all of them at once, which the tables cannot do for
    // themselves: five are never empty for a program that imported anything --
    // a module's fields, a type's name, the tracked set -- so
    // `AddressTable::is_empty` never fires for those and each costs a lock and
    // a probe that finds nothing.
    if super::address_table::release_hold(address) {
        release_borrowed(raw);
        BYTE_CACHE.take(address);
        forget_items(address);
        super::dictobject::forget_iteration(address);
        super::modsupport::forget_module_fields(address);
        super::unicodeobject::forget_block(address);
        super::frameobject::forget_pending(raw);
        super::pyerrors::forget_block(raw);
        super::typeobject::forget_descriptor_block(raw);
        unsafe { super::typeobject::forget_type_mirror(raw) };
        super::gc::forget(address);
    }
    // What a block carries in its own fields is there whether or not a table
    // keyed anything under its address.
    super::tupleobject::forget_block(raw);
    super::frameobject::forget_block(raw);
    super::sliceobject::forget_block(raw);
    super::methodobject::forget_block(raw);
    super::cdatetime::forget_block(raw);
    let block = block_at(address);
    let returned = if let Some(tp_dealloc) = unsafe { super::typeobject::tp_dealloc_of(raw) } {
        let call: unsafe extern "C" fn(*mut CPyObject) = unsafe { std::mem::transmute(tp_dealloc) };
        unsafe { call(raw) };
        // A `tp_dealloc` ends in `tp_free`, which returns the block through
        // [`free_block`]; if it did, the address is no longer ours to touch.
        // The generation and not just the address decides it: a `tp_dealloc`
        // that allocates after freeing can be handed the address back, and
        // what sits there then is somebody else's live block.
        //
        // A block that never came from here is answered yes without asking:
        // an extension allocating its instances itself frees them the same
        // way -- `_cffi_backend.c allocate_owning_object` takes a block from
        // plain `malloc` and its deallocator gives it back to `free` -- which
        // leaves the census unchanged and the address unreadable.
        block.is_none() || block_at(address) != block
    } else {
        false
    };
    // A count above zero here is a resurrection: `tp_finalize`, run from the
    // deallocator through `PyObject_CallFinalizerFromDealloc`, handed `self`
    // to something that kept it, and that deallocator answered by returning
    // without freeing.  The block is live again, so it is neither freed nor
    // left wearing the deallocating marker, and a collected type goes back
    // into the tracked set the teardown above dropped it from.
    if !returned && unsafe { (*raw).ob_refcnt } != 0 {
        unsafe { (*raw).ob_pyre_link = link };
        super::gc::track(raw);
        return;
    }
    if !returned {
        unsafe { free_block(raw) };
    } else {
        // What this layer keys by the address goes with the block, and
        // [`free_block`] is only reached for one it handed out itself.
        super::gc::forget_finalized(address);
    }
    release_heap_type(ob_type);
}

/// A block for the object allocator, entered in the same census as a mirror so
/// that [`free_block`] — which is what `PyObject_Free` calls — releases it.
///
/// Upstream reaches the same allocator from both directions: `_PyPy_Malloc`
/// hands out an instance's block and `PyObject_Malloc` hands out a plain one,
/// and `_PyPy_Free` takes either back (`object.py, 44-60`).
///
/// The size is floored at a whole [`CPyObject`] because [`free_block`] clears
/// that header before releasing the block.  A caller asking for less still gets
/// what it asked for; the floor only makes the block larger.
///
/// The header is cleared even when the caller did not ask for zeroed memory:
/// `PyObject_Init` reads `ob_pyre_link` to tell a block that is already an
/// object from one that is still bytes, so the header has to mean something
/// before anything is stamped into it.
pub(super) fn allocate_raw(size: usize, zeroed: bool) -> *mut std::ffi::c_void {
    if size > isize::MAX as usize {
        return std::ptr::null_mut();
    }
    let size = size.max(size_of::<CPyObject>());
    let layout = block_layout(size);
    let raw = unsafe {
        if zeroed {
            std::alloc::alloc_zeroed(layout)
        } else {
            std::alloc::alloc(layout)
        }
    };
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    if !zeroed {
        unsafe {
            (raw as *mut CPyObject).write(CPyObject {
                ob_refcnt: 0,
                ob_pyre_link: PY_NULL,
                ob_pyre_pad: 0,
                ob_type: std::ptr::null_mut(),
            })
        };
    }
    record_block(raw as usize, size);
    raw as *mut std::ffi::c_void
}

/// Grow or shrink a block [`allocate_raw`] handed out, keeping its contents.
///
/// # Safety
/// `raw` must be a live block from [`allocate_raw`], and must not be used
/// afterwards if this returns a different address.
pub(super) unsafe fn reallocate_raw(
    raw: *mut std::ffi::c_void,
    size: usize,
) -> *mut std::ffi::c_void {
    if size > isize::MAX as usize {
        return std::ptr::null_mut();
    }
    // A block this layer did not hand out has no recorded size, so there is no
    // layout to hand `realloc` and no way to move it.
    let Some(old) = block_at(raw as usize) else {
        return std::ptr::null_mut();
    };
    let size = size.max(size_of::<CPyObject>());
    let moved = unsafe { std::alloc::realloc(raw as *mut u8, block_layout(old.size), size) };
    if moved.is_null() {
        return std::ptr::null_mut();
    }
    let generation = BLOCK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut blocks = BLOCK_SIZES.lock();
    blocks.remove(&(raw as usize));
    blocks.insert(
        moved as usize,
        Block {
            size,
            generation,
            weakref_lifeline: old.weakref_lifeline,
        },
    );
    moved as *mut std::ffi::c_void
}

/// Return a mirror block to the allocator this layer took it from.
///
/// `cpyext/src/object.c:102-105 PyObject_Free`.  A block this layer did not
/// allocate is a `static` that outlives it, so only the bookkeeping is dropped.
///
/// # Safety
/// `raw` must be a block this module entered, and must not be used afterwards.
pub(super) unsafe fn free_block(raw: *mut CPyObject) {
    // The tracked set is keyed by address, so a block that stops existing has
    // to leave it here rather than only where a deallocator started: a
    // `tp_dealloc` may re-track the block it is tearing down -- Cython's does,
    // between its `PyObject_GC_UnTrack` and the base deallocator it ends in --
    // and the collector would then read a freed address.
    super::gc::forget(raw as usize);
    super::gc::forget_finalized(raw as usize);
    let Some(block) = BLOCK_SIZES.lock().remove(&(raw as usize)) else {
        return;
    };
    unsafe {
        (*raw).ob_pyre_link = PY_NULL;
        (*raw).ob_refcnt = 0;
    }
    if block.size != 0 {
        unsafe { std::alloc::dealloc(raw as *mut u8, block_layout(block.size)) };
    }
    if block.weakref_lifeline != 0 {
        unsafe { decref(block.weakref_lifeline as *mut CPyObject) };
    }
}

/// `state.py _rawrefcount_perform` — release every mirror the collector
/// has queued.
///
/// The collector left each one at a count of 1 (`incminimark.py:3352`), so the
/// decref here is what reaches zero and runs the deallocator.  Deallocating one
/// mirror can decref others, which is why the queue is re-read every iteration
/// rather than drained into a list.
pub fn drain_dead() {
    // `gcmodule.c:delete_garbage` is preceded by `finalize_garbage`, and for
    // the same reason: a finalizer is handed the object it belongs to, and
    // reads the fields of its block.  The collection that queued these kept
    // both alive for exactly this call.
    loop {
        let raw = majit_gc::gc_rawrefcount_next_finalize() as *mut CPyObject;
        if raw.is_null() {
            break;
        }
        super::gc::run_claimed_finalizer(raw);
    }
    // Before the deallocators, because a block the collector could not bring to
    // zero is one the queue never names: its remaining references are a
    // cycle's, and breaking them is what lets the deallocators below run at
    // all.
    super::gc::clear_garbage();
    loop {
        let raw = majit_gc::gc_rawrefcount_next_dead() as *mut CPyObject;
        if raw.is_null() {
            return;
        }
        unsafe { decref(raw) };
    }
}

/// References a container mirror owns on behalf of C.
///
/// `PyTuple_GetItem` and friends return a *borrowed* reference, which needs an
/// owner that outlives the call.  Upstream gives the container mirror an
/// `ob_item` array of owned references (`tupleobject.py:tuple_attach`); pyre
/// records the same ownership here, keyed by the container's mirror, and
/// releases it when that mirror is deallocated.
///
/// The ownership is per *item*, where upstream's is per *slot*: a mutable
/// container that has held many different items over its life keeps one
/// reference to each of them until the container itself dies, so an item
/// removed from a long-lived list is held here after the list stopped naming
/// it.  Indexing this by slot the way `ob_item` does is what removes that
/// retention, and needs the mutators to drop the slot's old entry.
///
/// A set, not a list: the membership test below runs once per item read, so a
/// C loop over `PyTuple_GetItem` for every item of an n-item container would
/// otherwise scan n/2 entries per step.
///
/// Each entry roots the item's object, and a collection is told which container
/// that root belongs to -- [`borrowed_edges`].  Without that the retention above
/// is unbounded rather than merely wide: an item referring back to its container
/// keeps the container, which is what would release the entry.
type BorrowSet = super::address_table::AddressSet;
type BorrowMap = super::address_table::HeldMap<BorrowSet>;
use super::address_table::{AddressTable, hold};

static BORROWED: AddressTable<BorrowMap> =
    AddressTable::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// A borrowed reference to `w_item`, owned by `container`'s mirror.
///
/// The reference taken here is dropped again when the item already has an
/// entry, so repeatedly reading the same slot cannot grow the list.
pub(super) fn borrow_from(container: *mut CPyObject, w_item: PyObjectRef) -> *mut CPyObject {
    if w_item.is_null() {
        return std::ptr::null_mut();
    }
    let item = make_ref(w_item);
    if container.is_null() {
        // No owner to attach to: the reference stays outstanding, which is the
        // conservative choice — the alternative hands C a freed pointer.
        return item;
    }
    let mut borrowed = BORROWED.lock();
    let fresh = borrowed
        .entry(hold(container as usize))
        .or_default()
        .insert(item as usize);
    if !fresh {
        drop(borrowed);
        unsafe { decref(item) };
    }
    item
}

/// Release everything a dying container mirror borrowed on C's behalf.
fn release_borrowed(container: *mut CPyObject) {
    let owned = BORROWED.take(container as usize);
    for item in owned.into_iter().flatten() {
        unsafe { decref(item as *mut CPyObject) };
    }
}

/// This table read as [`super::gc::c_edges`] reads a `tp_traverse`.
///
/// Each entry is a reference one mirror holds in another, released exactly when
/// the container mirror is deallocated, so it is an edge of the same kind as a C
/// field -- and one no traverse can report, the reference living here rather
/// than in the block. Without it a container that handed C a borrowed reference
/// to something referring back to the container is a cycle neither side can
/// see: the item's count roots the item, the item roots the container, and the
/// container is what would release the count.
///
/// Reported as separate entries rather than merged into a block's own, which the
/// collector sums either way.
pub(super) fn borrowed_edges(edges: &mut Vec<(usize, Vec<usize>)>) {
    let borrowed = BORROWED.lock();
    edges.reserve(borrowed.len());
    for (&container, items) in borrowed.iter() {
        if items.is_empty() {
            continue;
        }
        edges.push((container.address(), items.iter().copied().collect()));
    }
}

/// The item array a container mirror hands out — the `ob_item` field upstream's
/// `PyTupleObject` mirror carries, and the storage a list's mirror points into.
///
/// A mirror block here is exactly what its type declares, so the array is kept
/// beside the container instead, keyed by its mirror.  The address has to stay
/// good after the call returns — `PyTuple_GET_ITEM` is an lvalue and a caller
/// takes its address — which is what the box gives: it owns its slots, so a
/// rehash of the map does not move them.
///
/// Each entry is a reference the array owns, as `ob_item`'s are: a slot read
/// through `PyTuple_GET_ITEM` is borrowed from the container, and it stays
/// good for as long as the container does because the array holds it.
/// Held as addresses, which a mirror pointer is and a `Send` bound accepts.
type ItemCache = super::address_table::HeldMap<Box<[usize]>>;
static ITEM_ARRAYS: AddressTable<ItemCache> =
    AddressTable::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// Give `raw`'s array `items` and answer where they sit.
///
/// An array that already holds these items keeps the address it was handed out
/// at, and this call's references are given back instead: two reads of an
/// unchanged sequence have to answer the same address, and one of them may be
/// on another thread that is still holding the first.  Replacing the array is
/// the reallocation upstream documents, which is why a caller may only read one
/// while the sequence does not change.
pub(super) fn publish_items(raw: *mut CPyObject, items: Vec<usize>) -> *mut *mut CPyObject {
    let (address, spare) = {
        let mut cache = ITEM_ARRAYS.lock();
        let unchanged = cache
            .get(&(raw as usize))
            .is_some_and(|array| array[..] == items[..]);
        let spare = match unchanged {
            true => Some(items.into_boxed_slice()),
            false => cache.insert(hold(raw as usize), items.into_boxed_slice()),
        };
        (
            cache[&(raw as usize)].as_ptr() as *mut *mut CPyObject,
            spare,
        )
    };
    // Outside the lock: a deallocator this runs takes it again.
    release_item_array(spare);
    address
}

/// Give back the references an array owned.
fn release_item_array(array: Option<Box<[usize]>>) {
    for entry in array.into_iter().flatten() {
        unsafe { decref(entry as *mut CPyObject) };
    }
}

/// `raw`'s array, built by `items` the first time it is asked for.
///
/// For a container whose contents cannot change behind the array — a tuple —
/// so that reading it slot by slot costs what reading a field costs.
pub(super) fn items_or_build(
    raw: *mut CPyObject,
    items: impl FnOnce() -> Vec<usize>,
) -> *mut *mut CPyObject {
    let existing = ITEM_ARRAYS
        .lock()
        .get(&(raw as usize))
        .map(|array| array.as_ptr() as *mut *mut CPyObject);
    if let Some(array) = existing {
        return array;
    }
    // Built with the lock released: an entry is taken through `make_ref`,
    // which allocates and takes locks of its own.  Another thread reading the
    // same container can have built and published one meanwhile, which is why
    // publishing keeps whichever array is already there.
    let built = items();
    publish_items(raw, built)
}

/// Put `item` in one slot of an array already handed out, leaving its address
/// alone, and answer with what it displaced.
///
/// What changes a tuple after C has seen it is `PyTuple_SetItem` and
/// `PyTuple_SET_ITEM`, and a caller holding the address `PyTuple_GET_ITEM`
/// gave it has to read the new value through it.
///
/// The array takes over `item`'s reference and gives up the one it answers
/// with; a caller that gets nothing back -- no array, or an index past its end
/// -- still holds `item`'s.
pub(super) fn replace_cached_item(
    raw: *mut CPyObject,
    index: usize,
    item: *mut CPyObject,
) -> Option<*mut CPyObject> {
    let mut cache = ITEM_ARRAYS.lock();
    let slot = cache.get_mut(&(raw as usize))?.get_mut(index)?;
    Some(std::mem::replace(slot, item as usize) as *mut CPyObject)
}

/// Drop the array a dying container mirror handed out, and the references it
/// owned with it.
fn forget_items(raw: usize) {
    let array = ITEM_ARRAYS.take(raw);
    release_item_array(array);
}

/// The NUL-terminated byte view of a mirror, filled on first use.
///
/// This is the counterpart of the `c_utf8` field PyPy fills on its
/// `PyUnicodeObject` mirror: `PyUnicode_AsUTF8` and `PyBytes_AsString` must
/// hand out a stable, NUL-terminated address, and the interpreter's own storage
/// is neither NUL-terminated nor at a fixed address.  It is a side table rather
/// than a field so that a mirror block is exactly what its type declares.
///
/// The entry says whether the mirror is still only this buffer -- one
/// `PyBytes_FromStringAndSize` handed out with no `bytes` behind it, that
/// nothing has read as a value yet.  Held here rather than in a set of
/// addresses beside it, so that the lock which answers the question is the one
/// that settles the race to answer it; `unicodeobject` carries its own such
/// flag in its own entry the same way.
struct CachedBytes {
    /// The payload, with the terminator appended after it.
    bytes: Box<[u8]>,
    pending: bool,
}

type ByteCache = super::address_table::HeldMap<CachedBytes>;
static BYTE_CACHE: AddressTable<ByteCache> =
    AddressTable::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// The mirror's cached bytes and their length, without the terminator.
///
/// The `bytes` the producer returns is the payload; the terminator is appended
/// here, so the address is usable as a C string whenever the payload has no
/// interior NUL, and `PyBytes_AsStringAndSize`-style callers get the length
/// separately either way.
///
/// # Safety
/// `raw` must be a live mirror.
pub(super) unsafe fn cached_bytes(
    raw: *mut CPyObject,
    produce: impl FnOnce() -> Vec<u8>,
) -> (*const c_char, usize) {
    unsafe { cache_bytes(raw, produce, false) }
}

/// [`cached_bytes`] for a mirror that is still only the buffer it fills.
///
/// # Safety
/// `raw` must be a live mirror.
pub(super) unsafe fn cache_pending_bytes(
    raw: *mut CPyObject,
    produce: impl FnOnce() -> Vec<u8>,
) -> (*const c_char, usize) {
    unsafe { cache_bytes(raw, produce, true) }
}

/// # Safety
/// `raw` must be a live mirror.
unsafe fn cache_bytes(
    raw: *mut CPyObject,
    produce: impl FnOnce() -> Vec<u8>,
    pending: bool,
) -> (*const c_char, usize) {
    let mut cache = BYTE_CACHE.lock();
    let entry = cache.entry(hold(raw as usize)).or_insert_with(|| {
        let mut bytes = produce();
        bytes.push(0);
        CachedBytes {
            bytes: bytes.into_boxed_slice(),
            pending,
        }
    });
    // The box owns its bytes, so the address stays put as the map rehashes.
    (entry.bytes.as_ptr() as *const c_char, entry.bytes.len() - 1)
}

/// Whether `raw` is a mirror that is still only the buffer it was handed out
/// as, so C may still be writing it.
pub(super) fn bytes_pending(raw: *mut CPyObject) -> bool {
    !raw.is_null()
        && BYTE_CACHE
            .lock()
            .get(&(raw as usize))
            .is_some_and(|entry| entry.pending)
}

/// Take a copy of such a mirror's buffer, and record that it is no longer only
/// one.
///
/// The flag is cleared under the lock that answered it, so of two threads
/// reading the same mirror as a value at once exactly one is told to build the
/// `bytes`.  The copy is taken here for the same reason the caller would take
/// it: building allocates, and this lock is one a deallocator takes.
pub(super) fn claim_pending_bytes(raw: *mut CPyObject) -> Option<Vec<u8>> {
    let mut cache = BYTE_CACHE.lock();
    let entry = cache.get_mut(&(raw as usize))?;
    if !std::mem::replace(&mut entry.pending, false) {
        return None;
    }
    Some(entry.bytes[..entry.bytes.len() - 1].to_vec())
}

/// Give a mirror's cached bytes a new length, keeping what still fits and
/// zeroing what is new, and answer the address the caller reads from now.
///
/// The address moves, the box being replaced.  That is what resizing the tail
/// an object carries does upstream too, and the caller re-reads it through
/// `PyBytes_AS_STRING(*pv)` afterwards either way.
///
/// `None` is a request there was no room for; a mirror with nothing cached
/// cannot reach this, the only caller being a resize of a buffer it handed
/// out.
///
/// # Safety
/// `raw` must be a live mirror whose bytes are already cached.
pub(super) unsafe fn resize_cached_bytes(
    raw: *mut CPyObject,
    size: usize,
) -> Option<*const c_char> {
    let mut cache = BYTE_CACHE.lock();
    let entry = cache.get_mut(&(raw as usize))?;
    let mut bytes: Vec<u8> = Vec::new();
    if bytes.try_reserve_exact(size + 1).is_err() {
        return None;
    }
    bytes.extend_from_slice(&entry.bytes[..(entry.bytes.len() - 1).min(size)]);
    bytes.resize(size + 1, 0);
    entry.bytes = bytes.into_boxed_slice();
    Some(entry.bytes.as_ptr() as *const c_char)
}

/// `state.py:106-107` — give the collector the callback it fires when it has
/// queued mirrors, before anything can create the first link.
///
/// There is no root walker to register alongside it.  Upstream has none either:
/// a P-link is a root exactly when the mirror's count exceeds the link share,
/// and that is a question only the collector's own trace passes can ask at the
/// right moment.
pub(super) fn init_rawrefcount() {
    majit_gc::gc_rawrefcount_init(schedule_drain);
    majit_gc::gc_rawrefcount_set_light_free(free_light_mirror);
    majit_gc::gc_rawrefcount_set_c_edge_census(super::gc::c_edges);
    majit_gc::gc_rawrefcount_set_finalizer_claim(super::gc::claim_finalizer);
}

/// `incminimark.py _rrc_free`'s raw release for a LIGHT link, routed through
/// the same allocator and bookkeeping as every other cpyext mirror block.
fn free_light_mirror(mirror: usize) {
    unsafe { free_block(mirror as *mut CPyObject) };
}

/// Fired from inside a collection, so it may only schedule.
fn schedule_drain() {
    crate::executioncontext::fire_cpyext_dealloc_action();
}

// ── the immortal singletons ─────────────────────────────────────────────

const fn immortal() -> CPyObject {
    CPyObject {
        ob_refcnt: REFCNT_IMMORTAL,
        ob_pyre_link: PY_NULL,
        ob_pyre_pad: 0,
        ob_type: std::ptr::null_mut(),
    }
}

/// `Py_None`. C compares against this pointer, so the mirror has to be the one
/// `make_ref(w_none())` returns; [`init_singletons`] enters it in the table.
#[unsafe(no_mangle)]
pub static mut _Py_NoneStruct: CPyObject = immortal();

/// `Py_True`.
#[unsafe(no_mangle)]
pub static mut _Py_TrueStruct: CPyObject = immortal();

/// `Py_False`.
#[unsafe(no_mangle)]
pub static mut _Py_FalseStruct: CPyObject = immortal();

/// `Py_NotImplemented`.
#[unsafe(no_mangle)]
pub static mut _Py_NotImplementedStruct: CPyObject = immortal();

/// `Py_Ellipsis`.
#[unsafe(no_mangle)]
pub static mut _Py_EllipsisObject: CPyObject = immortal();

/// Bind each preallocated singleton mirror to its interpreter object.
///
/// Called once before the first `PyInit_*`, so a C function that compares a
/// result against `Py_None` sees the same pointer `make_ref` hands out.
pub fn init_singletons() {
    let bound: [(*mut CPyObject, PyObjectRef); 5] = [
        (&raw mut _Py_NoneStruct, pyre_object::w_none()),
        (
            &raw mut _Py_TrueStruct,
            pyre_object::boolobject::w_bool_from(true),
        ),
        (
            &raw mut _Py_FalseStruct,
            pyre_object::boolobject::w_bool_from(false),
        ),
        (
            &raw mut _Py_NotImplementedStruct,
            pyre_object::w_not_implemented(),
        ),
        (&raw mut _Py_EllipsisObject, pyre_object::w_ellipsis()),
    ];
    for (raw, w_obj) in bound {
        if unsafe { !(*raw).ob_pyre_link.is_null() } {
            continue;
        }
        attach_foreign(w_obj, raw);
    }
    // Deferred until the links are in the table: `type_mirror` takes the same
    // lock, and a singleton's type mirror is an ordinary synthesized mirror.
    for (raw, _) in bound {
        if unsafe { (*raw).ob_type.is_null() } {
            let of_type = type_mirror(unsafe { (*raw).ob_pyre_link });
            unsafe { set_ob_type(raw, of_type) };
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IncRef(object: *mut CPyObject) {
    unsafe { incref(object) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_DecRef(object: *mut CPyObject) {
    unsafe { decref(object) };
}

/// `Py_SET_REFCNT` — the whole count field, which is what `Py_REFCNT` reports.
///
/// A value written here has to have come from `Py_REFCNT`: the count carries
/// the share the interpreter link contributes, and an absolute number in the
/// units C code counts in would take that share away and free a block the
/// interpreter still points at.  A mirror that must never be freed keeps the
/// count it has.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Py_SetRefcnt(object: *mut CPyObject, refcnt: isize) {
    if object.is_null() || unsafe { (*object).ob_refcnt } >= REFCNT_IMMORTAL {
        return;
    }
    unsafe { (*object).ob_refcnt = refcnt };
}

/// The entry point `Py_INCREF` is a call to on a stable ABI from 3.12 on.
///
/// An extension compiled against the stable ABI expands the macro to this
/// rather than touching `ob_refcnt`, so the two names have to be two symbols
/// even though they do the same thing; `refcount.h` here spells the macro over
/// [`Py_IncRef`], and an extension that declares the prototype itself -- as
/// PyO3 does, which never reads these headers -- reaches this one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Py_IncRef(object: *mut CPyObject) {
    unsafe { incref(object) };
}

/// The `Py_DECREF` half of [`_Py_IncRef`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Py_DecRef(object: *mut CPyObject) {
    unsafe { decref(object) };
}

/// The C-visible reference count, with the interpreter's own share removed —
/// `Py_REFCNT` as a test can read it without depending on the link share.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_RefCount(object: *mut CPyObject) -> isize {
    if object.is_null() {
        return 0;
    }
    unsafe { (*object).ob_refcnt - REFCNT_FROM_PYPY }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(Py_IncRef as *const ());
    std::hint::black_box(Py_DecRef as *const ());
    std::hint::black_box(_Py_IncRef as *const ());
    std::hint::black_box(_Py_DecRef as *const ());
    std::hint::black_box(_PyPyre_RefCount as *const ());
    std::hint::black_box(_Py_SetRefcnt as *const ());
}
