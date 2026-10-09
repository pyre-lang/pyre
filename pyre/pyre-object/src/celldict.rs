//! `pypy/objspace/std/celldict.py` port — module / globals dict
//! strategy backed by a version-tagged map of `str -> value_or_cell`.
//!
//! Header docstring at upstream `celldict.py:1-4`:
//!
//! ```text
//! A very simple cell dict implementation using a version tag.
//! The dictionary maps keys to objects. If a specific key is
//! changed a lot, a level of indirection is introduced to make
//! the version tag change less often.
//! ```
//!
//! The `ModuleDictStrategy` struct itself
//! plus the supporting `VersionTag` / `ModuleDictStorage` types.
//! `MutableCell`, `ObjectMutableCell`, `IntMutableCell`,
//! `write_cell` and `unwrap_cell` (defined on `W_TypeObject`'s
//! module, `typeobject.py`) live in this file.  Module-dict stores
//! and `W_TypeObject.setdictvalue` both go through `write_cell`.
//! An in-place update returns no new object, so the caller does not
//! run `mutated()` and the type's `_version_tag` stays put.  That
//! stable tag is what keeps a traced class-attribute store's
//! quasi-immutable watcher valid across the loop.
//!
//! `W_ModuleDictObject` (`dictmultiobject.rs`) carries this strategy
//! as its `mstrategy` slot per `dictmultiobject.py`.  The
//! trait `impl crate::dictmultiobject::DictStrategy for
//! ModuleDictStrategy` lives at the bottom of this file and routes
//! every method to the existing `w_module_dict_*` / `w_dict_*`
//! free functions so callers can dispatch polymorphically via
//! `w_dict_get_strategy(obj)`.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use crate::pyobject::*;

// ── MutableCell family ──────────────────────────────────────────────
//
// `pypy/objspace/std/typeobject.py` defines the cell layer
// referenced by `celldict.py _setitem_str_cell_known` and
// `:143-145 getitem_str`.  PyPy keeps a level of indirection so that
// frequently-rewritten module / type attributes mutate the cell's
// payload without bumping `mstrategy.version`, which keeps JIT inline
// caches keyed on that version valid across the rewrite.
//
//     class MutableCell(W_Root):
//         def unwrap_cell(self, space): raise NotImplementedError
//
//     class ObjectMutableCell(MutableCell):
//         def __init__(self, w_value=None):
//             self.w_value = w_value
//         def unwrap_cell(self, space):
//             return self.w_value
//
//     class IntMutableCell(MutableCell):
//         def __init__(self, intvalue):
//             self.intvalue = intvalue
//         def unwrap_cell(self, space):
//             return space.newint(self.intvalue)
//
//     def unwrap_cell(space, w_value):
//         if isinstance(w_value, MutableCell):
//             return w_value.unwrap_cell(space)
//         return w_value
//
//     def write_cell(space, w_cell, w_value):
//         if w_cell is None:
//             return w_value
//         if isinstance(w_cell, ObjectMutableCell):
//             w_cell.w_value = w_value
//             return None
//         elif isinstance(w_cell, IntMutableCell) and is_plain_int1(w_value):
//             w_cell.intvalue = plain_int_w(space, w_value)
//             return None
//         elif space.is_w(w_cell, w_value):
//             return None
//         if is_plain_int1(w_value):
//             return IntMutableCell(plain_int_w(space, w_value))
//         else:
//             return ObjectMutableCell(w_value)

/// Internal type tag for `ObjectMutableCell`.  Never user-visible —
/// cells live inside the module dict's storage and are unwrapped
/// before any read crosses out to user code.  The PyType is allocated
/// so that `py_type_check` can disambiguate cells from real values
/// without a separate type-id field.
pub static OBJECT_MUTABLE_CELL_TYPE: PyType = new_pytype("__ObjectMutableCell");

/// Internal type tag for `IntMutableCell`.
pub static INT_MUTABLE_CELL_TYPE: PyType = new_pytype("__IntMutableCell");

/// GC type id assigned to `ObjectMutableCell` — slot 49, immediately
/// after `W_MODULE_DICT_GC_TYPE_ID=48`.
pub const W_OBJECT_MUTABLE_CELL_GC_TYPE_ID: u32 = 49;

/// GC type id assigned to `IntMutableCell`.
pub const W_INT_MUTABLE_CELL_GC_TYPE_ID: u32 = 50;

/// `typeobject.py ObjectMutableCell`.
#[repr(C)]
pub struct ObjectMutableCell {
    pub ob_header: PyObject,
    pub w_value: PyObjectRef,
}

/// `typeobject.py IntMutableCell`.
#[repr(C)]
pub struct IntMutableCell {
    pub ob_header: PyObject,
    pub intvalue: i64,
}

pub const W_OBJECT_MUTABLE_CELL_OBJECT_SIZE: usize = std::mem::size_of::<ObjectMutableCell>();
pub const W_INT_MUTABLE_CELL_OBJECT_SIZE: usize = std::mem::size_of::<IntMutableCell>();

/// Byte offset of the inline `PyObjectRef` field the GC must trace
/// during minor collection.  Mirrors `W_CELL_GC_PTR_OFFSETS` on the
/// closure-cell layer (`nestedscope.rs`).
pub const W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS: [usize; 1] =
    [std::mem::offset_of!(ObjectMutableCell, w_value)];

impl crate::lltype::GcType for ObjectMutableCell {
    fn type_id() -> u32 {
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID
    }
    const SIZE: usize = W_OBJECT_MUTABLE_CELL_OBJECT_SIZE;
}

impl crate::lltype::GcType for IntMutableCell {
    fn type_id() -> u32 {
        W_INT_MUTABLE_CELL_GC_TYPE_ID
    }
    const SIZE: usize = W_INT_MUTABLE_CELL_OBJECT_SIZE;
}

/// `typeobject.py ObjectMutableCell.__init__`.
///
/// An ordinary nursery instance (`W_Root`). A trace that promotes the cell
/// records a `ConstPtr`; `remove_constptrs_in` rewrites that to
/// `LoadFromGcTable` and `gcreftracer` updates the slot when the cell moves.
/// `w_value` is traced through `W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS`.
pub fn w_object_mutable_cell_new(w_value: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(ObjectMutableCell {
        ob_header: PyObject {
            ob_type: &OBJECT_MUTABLE_CELL_TYPE as *const PyType,
            w_class: get_instantiate(&OBJECT_MUTABLE_CELL_TYPE),
        },
        w_value,
    }) as PyObjectRef
}

/// `typeobject.py IntMutableCell.__init__`.
///
/// Same nursery placement as [`w_object_mutable_cell_new`]. The unboxed
/// `intvalue` is not a GC pointer.
pub fn w_int_mutable_cell_new(intvalue: i64) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(IntMutableCell {
        ob_header: PyObject {
            ob_type: &INT_MUTABLE_CELL_TYPE as *const PyType,
            w_class: get_instantiate(&INT_MUTABLE_CELL_TYPE),
        },
        intvalue,
    }) as PyObjectRef
}

/// `isinstance(w, ObjectMutableCell)` predicate.
///
/// # Safety
/// `obj` must be a valid non-null PyObjectRef.
#[inline]
pub unsafe fn is_object_mutable_cell(obj: PyObjectRef) -> bool {
    !obj.is_null() && py_type_check(obj, &OBJECT_MUTABLE_CELL_TYPE)
}

/// `isinstance(w, IntMutableCell)` predicate.
///
/// # Safety
/// `obj` must be a valid non-null PyObjectRef.
#[inline]
pub unsafe fn is_int_mutable_cell(obj: PyObjectRef) -> bool {
    !obj.is_null() && py_type_check(obj, &INT_MUTABLE_CELL_TYPE)
}

/// `isinstance(w, MutableCell)`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_mutable_cell(obj: PyObjectRef) -> bool {
    is_object_mutable_cell(obj) || is_int_mutable_cell(obj)
}

/// `typeobject.py unwrap_cell`:
///
/// ```python
/// def unwrap_cell(space, w_value):
///     if isinstance(w_value, MutableCell):
///         return w_value.unwrap_cell(space)
///     return w_value
/// ```
///
/// Hot path: read `ob_type` once, two pointer-equality compares
/// against the two static cell type tags.  The common case is a
/// non-cell value (raw function / int / etc.), so both compares
/// fall through to the identity return without further memory traffic.
///
/// # Safety
/// `w_value` must be a valid PyObjectRef (null tolerated).
#[inline]
pub unsafe fn unwrap_cell(w_value: PyObjectRef) -> PyObjectRef {
    if w_value.is_null() {
        return w_value;
    }
    // A tagged immediate is a plain value, never a `MutableCell`; return it
    // unchanged before the `ob_type` deref (which would fault on the
    // immediate). Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(w_value) {
        return w_value;
    }
    let tp = (*w_value).ob_type;
    if std::ptr::eq(tp, &OBJECT_MUTABLE_CELL_TYPE as *const PyType) {
        return (*(w_value as *const ObjectMutableCell)).w_value;
    }
    if std::ptr::eq(tp, &INT_MUTABLE_CELL_TYPE as *const PyType) {
        return crate::w_int_new((*(w_value as *const IntMutableCell)).intvalue);
    }
    w_value
}

/// Visit a module dict value slot during a GC root walk.
///
/// A module dict entry (and a `GlobalCache.cell`) is either a raw
/// `w_value` or a `MutableCell` wrapping it (`typeobject.py
/// unwrap_cell`).  A cell the collector owns is visited as the slot
/// itself; its `w_value` is traced through the registered offsets.  A
/// cell allocated before the allocation hook was installed is outside
/// the heap, so its inner `w_value` is visited in place instead.
///
/// # Safety
/// `slot` must point to a valid `PyObjectRef` (null tolerated).
pub unsafe fn walk_module_value_slot(
    slot: &mut PyObjectRef,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let w_value = *slot;
    if w_value.is_null() {
        return;
    }
    // A tagged immediate is a plain value, never a `MutableCell`; forward the
    // slot as-is before the `ob_type` deref (which would fault on the
    // immediate). The collector's `is_valid_gc_object` no-ops on the odd
    // address. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(w_value) {
        visitor(slot);
        return;
    }
    if !is_mutable_cell(w_value)
        || crate::gc_hook::try_gc_owns_object(w_value as crate::gc_hook::GCREF)
    {
        visitor(slot);
        return;
    }
    // IntMutableCell carries an unboxed i64; no GC reference.
    if is_object_mutable_cell(w_value) {
        let cell = &mut *(w_value as *mut ObjectMutableCell);
        visitor(&mut cell.w_value);
    }
}

/// Write barrier for an in-place `ObjectMutableCell.w_value` store.
///
/// A collector-owned cell takes the ordinary barrier. A cell outside the
/// heap is reached only by the prebuilt-family root walk. `incminimark.py`
/// `remember_young_pointer` records that walk independently of the stored
/// pointer: the incremental collector relies on the first prebuilt write
/// registering the object, and the JIT write barrier does not take `newvalue`.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn object_mutable_cell_write_barrier(cell: crate::gc_hook::GCREF) {
    if crate::gc_hook::try_gc_owns_object(cell) {
        crate::gc_hook::try_gc_write_barrier(cell);
    } else {
        crate::gc_roots::mark_prebuilt_roots_dirty();
    }
}

/// `typeobject.py write_cell`:
///
/// ```python
/// def write_cell(space, w_cell, w_value):
///     if w_cell is None:
///         return w_value
///     if isinstance(w_cell, ObjectMutableCell):
///         w_cell.w_value = w_value
///         return None
///     elif isinstance(w_cell, IntMutableCell) and is_plain_int1(w_value):
///         w_cell.intvalue = plain_int_w(space, w_value)
///         return None
///     elif space.is_w(w_cell, w_value):
///         return None
///     if is_plain_int1(w_value):
///         return IntMutableCell(plain_int_w(space, w_value))
///     else:
///         return ObjectMutableCell(w_value)
/// ```
///
/// `Option<PyObjectRef>` return: `None` => the cell mutation was
/// in-place; the storage's existing entry stays.  `Some(w)` => the
/// caller must write `w` into the storage slot (either a brand-new
/// cell or the raw value for the no-cell-yet case).
///
/// # Safety
/// `w_cell` must be either `None` or a valid PyObjectRef.  `w_value`
/// must be a valid non-null PyObjectRef.
pub unsafe fn write_cell(
    w_cell: Option<PyObjectRef>,
    mut w_value: PyObjectRef,
) -> Option<PyObjectRef> {
    debug_assert!(!w_value.is_null(), "write_cell: null value");
    // `space.is_w` of a long and a machine int builds an rbigint, so the
    // value stored below and the cell rewritten in place stay rooted
    // across the classification.
    let mut w_cell_word = w_cell.unwrap_or(std::ptr::null_mut());
    let write =
        crate::with_roots!(w_cell_word, w_value => classify_cell_write(w_cell_word, w_value));
    match write {
        CellWrite::InPlaceObject(_) => {
            let cell = w_cell_word;
            // Barrier before the store, the `remember_young_pointer` order.
            // An in-place int store writes no `PyObjectRef` and needs none.
            object_mutable_cell_write_barrier(cell as crate::gc_hook::GCREF);
            (*(cell as *mut ObjectMutableCell)).w_value = w_value;
            None
        }
        CellWrite::InPlaceInt(_, intvalue) => {
            (*(w_cell_word as *mut IntMutableCell)).intvalue = intvalue;
            None
        }
        CellWrite::Unchanged => None,
        CellWrite::StoreBare => Some(w_value),
        CellWrite::Replace => {
            if crate::listobject::is_plain_int1(w_value) {
                return Some(w_int_mutable_cell_new(crate::listobject::plain_int_w(
                    w_value,
                )));
            }
            Some(w_object_mutable_cell_new(w_value))
        }
    }
}

/// What [`write_cell`] will do with the slot.
///
/// Split out from the mutation so the tracer's bump predicate
/// ([`store_would_bump_version`]) cannot drift from the write it predicts.
enum CellWrite {
    /// Write through the existing cell; the stored pointer, and therefore
    /// `version?`, stands.
    InPlaceObject(PyObjectRef),
    InPlaceInt(PyObjectRef, i64),
    /// The slot already holds this exact value.
    Unchanged,
    /// No cell yet: store the value itself, without a level of indirection.
    /// A later write over it is what promotes the slot to a cell.
    StoreBare,
    /// A cell is there but cannot take the value in place, so the slot gets a
    /// fresh one.
    Replace,
}

/// The decision half of [`write_cell`]; performs no store.
unsafe fn classify_cell_write(w_cell: PyObjectRef, w_value: PyObjectRef) -> CellWrite {
    if w_cell.is_null() {
        // attribute does not exist at all, write it without a cell first
        return CellWrite::StoreBare;
    }
    if is_object_mutable_cell(w_cell) {
        return CellWrite::InPlaceObject(w_cell);
    }
    if is_int_mutable_cell(w_cell) && crate::listobject::is_plain_int1(w_value) {
        return CellWrite::InPlaceInt(w_cell, crate::listobject::plain_int_w(w_value));
    }
    // If the new value and the current value are the same, don't
    // create a level of indirection, or mutate the version.
    if crate::pyobject::is_w(w_cell, w_value) {
        return CellWrite::Unchanged;
    }
    CellWrite::Replace
}

/// Whether storing `w_value` over the raw slot contents `w_cell` would reach
/// `mutated()`, i.e. would assign the `version?` quasi-immutable field.
///
/// This is the tracer's stand-in for the rtyper-inserted
/// `jit_force_quasi_immutable` that upstream places on that write
/// (`rclass.py`). Pyre's store runs inside a residual helper the walker
/// never looks into, so the walker has to ask the question ahead of the call
/// instead of meeting the operation inside it. Side-effect-free by
/// construction — it only classifies.
///
/// An in-place cell write must answer `false`: it leaves the stored pointer
/// alone, so `version` stays valid and a hot module-scope loop must keep its
/// compiled trace.
///
/// # Safety
/// `w_cell` (when `Some`) and `w_value` must point at live objects.
pub unsafe fn store_would_bump_version(w_cell: Option<PyObjectRef>, w_value: PyObjectRef) -> bool {
    matches!(
        classify_cell_write(w_cell.unwrap_or(std::ptr::null_mut()), w_value),
        CellWrite::StoreBare | CellWrite::Replace
    )
}

/// `pypy/objspace/std/celldict.py VersionTag`:
///
/// ```python
/// class VersionTag(object):
///     pass
/// ```
///
/// An opaque identity tag invalidated on every mutation that affects
/// the JIT's view of the dict.  Pyre's stand-in is a monotonically
/// increasing counter — pointer-identity matches PyPy's `is` test
/// because each `Box<VersionTag>` allocates a fresh address but a
/// counter is JIT-friendlier and trivially `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionTag(pub u64);

/// The serial [`VersionTag::fresh`] hands out — a process-global counter
/// bumped once per tag.
///
/// The counter is a runtime-mutable global, so the read seam is residual
/// (`@dont_look_inside`, `rlib/jit.py`, the `importing::sys_modules_dict`
/// shape): whatever value the build process happens to observe is not a
/// constant, and folding it would hand every dict the same tag.  Upstream
/// allocates a fresh `VersionTag()` object here instead, which rtypes to a
/// `malloc` the JIT models; a `static AtomicU64` has no llop counterpart, so
/// the call itself is the last modellable point.  The `u64` serial is a single
/// word and the newtype wrap stays traced.
#[majit_macros::dont_look_inside]
pub fn next_version_tag_serial() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl VersionTag {
    /// Allocate a fresh, never-before-seen version tag.
    pub fn fresh() -> Self {
        VersionTag(next_version_tag_serial())
    }
}

/// `pypy/objspace/std/celldict.py _wrapkey`:
///
/// ```python
/// def _wrapkey(space, key):
///     return space.newtext(key)
/// ```
///
/// Wraps a Rust `&str` key as a Python `str` PyObjectRef.
///
/// The canonical object for the characters, not a fresh one.  This storage
/// half keeps names as Rust strings, so the key object user code sees is
/// rebuilt on every read, and building it is what two separate defects came
/// from: `next(iter(d)) is next(iter(d))` answered false where both CPython
/// and PyPy answer true, and `w_str_new` allocates the immortal flavour, so
/// nothing reclaimed the copies -- measured over `globals()`, 118 bytes per
/// key read, 2.1 GB across a 1.2M-iteration walk that a plain dict serves in
/// a flat 60 MB.  Interning answers both: the table hands back one object per
/// value, and `intern_wtf8_value` builds nothing on a hit.
#[inline]
pub fn _wrapkey(key: &str) -> PyObjectRef {
    crate::intern_str_value(key)
}

/// Strategy-owned storage for `ModuleDictStrategy`.
///
/// `celldict.py`:
///
/// ```python
/// erase, unerase = rerased.new_erasing_pair("modulecell")
/// ...
/// def get_empty_storage(self):
///     return self.erase({})
/// ```
///
/// PyPy erases a real Python `{}` dict — insertion-ordered (since
/// Python 3.7) and O(1) hashed.  Pyre's port wraps
/// [`ModuleDictEntries`], which provides the same
/// insertion-ordered + hashed semantics directly so the strategy
/// contract on `:188-198 getiter{keys,values,items,reversed}`
/// continues to honour insertion order while `get` / `set` / `remove`
/// stay O(1) amortised.
#[derive(Default)]
pub struct ModuleDictStorage {
    pub entries: ModuleDictEntries,
}

/// `ModuleDictStrategy` key: the unwrapped `str` (`Utf8Str`).
///
/// The block is the same payload `W_UnicodeObject.value` points at, allocated
/// by `alloc_utf8_payload` (`managed` so the collector owns the leaf).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct StrKey(pub *mut crate::unicodeobject::Utf8Str);

impl StrKey {
    /// The block's UTF-8 characters, the same slice `str` hashing and
    /// equality see.
    pub fn as_str(&self) -> &str {
        // Keys are stored from a `&str`, so the payload is UTF-8. A null
        // dummy (a cleared slot) reads as empty.
        unsafe { std::str::from_utf8_unchecked(crate::unicodeobject::utf8_payload_bytes(self.0)) }
    }
}

impl std::hash::Hash for StrKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialEq for StrKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for StrKey {}

impl std::borrow::Borrow<str> for StrKey {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl crate::rordereddict::EntryDummy for StrKey {
    fn dummy() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl std::fmt::Debug for StrKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

/// The entry table `ModuleDictStorage` holds.
///
/// Keyed by the unwrapped `str` (`ModuleDictStrategy`) and hashed with
/// [`crate::dictmultiobject::StrKeyBuildHasher`] rather than Rust's default
/// `RandomState`: upstream's storage is a plain `{}` over RPython strings,
/// whose hash is a cheap chain cached on the string object, so a probe on a
/// global's name costs one load there.
pub type ModuleDictEntries =
    crate::rordereddict::RDict<StrKey, PyObjectRef, crate::dictmultiobject::StrKeyBuildHasher>;

/// The per-name `GlobalCache` registry (`celldict.py self.caches`).
///
/// Keyed by global name, so it takes the same hasher
/// [`ModuleDictEntries`] does.
pub type GlobalCacheRegistry = std::collections::HashMap<
    String,
    std::sync::Arc<parking_lot::Mutex<GlobalCache>>,
    crate::dictmultiobject::StrKeyBuildHasher,
>;

/// Runtime-assigned GC type id for the [`ModuleDictStorage`] box.
static MODULE_DICT_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`ModuleDictStorage`] box.
pub fn set_module_dict_storage_gc_type_id(id: u32) {
    MODULE_DICT_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`ModuleDictStorage`] box.
#[majit_macros::dont_look_inside]
pub fn module_dict_storage_gc_type_id() -> u32 {
    MODULE_DICT_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// The single `IndexMap` probe [`ModuleDictStorage::get`] runs —
/// `celldict.py getitem_str`'s
/// `self.unerase(w_dict.dstorage).get(key)`.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`).  Keys are unwrapped `str` blocks, so
/// no user `__eq__` or `__hash__` runs inside the probe.
fn module_dict_entries_get_iff(entries: &ModuleDictEntries, key: &str) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(key)
}

#[majit_macros::look_inside_iff(module_dict_entries_get_iff)]
pub fn module_dict_entries_get(entries: &ModuleDictEntries, key: &str) -> Option<PyObjectRef> {
    entries.get(key).copied()
}

/// The store side of [`module_dict_entries_get`] — `celldict.py getdictvalue_no_unwrapping`'s
/// `self.unerase(w_dict.dstorage)[key] = w_value`, returning the displaced
/// value so the caller can tell an overwrite from an insert.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `_ll_dict_setitem_lookup_done` (`rordereddict.py`).  Overwrite keeps
/// the existing slot's position (`{}` assignment).
fn module_dict_entries_insert_iff(
    entries: &mut ModuleDictEntries,
    key: &str,
    _w_value: PyObjectRef,
) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(key)
}

#[majit_macros::look_inside_iff(module_dict_entries_insert_iff)]
pub fn module_dict_entries_insert(
    entries: &mut ModuleDictEntries,
    key: &str,
    w_value: PyObjectRef,
) -> Option<PyObjectRef> {
    // An existing name keeps its block. Allocating before the probe would
    // mint a `STR` the overwrite then drops.
    if let Some(slot) = entries.get_mut(key) {
        return Some(crate::rordereddict::replace_value(slot, w_value));
    }
    entries.insert(StrKey(module_dict_key_block(key)), w_value)
}

/// The `STR` a new module-dict entry is keyed by.
///
/// `celldict.py setitem_str` stores the caller's unwrapped `str` as the key
/// and allocates nothing: `STORE_NAME` / `STORE_GLOBAL` pass
/// `pycode.co_names_w[i]`'s value, which `pycode.py` interned. That object is
/// the canonical interned `str` for these characters, so its `_utf8` block is
/// the key upstream stores. A key nobody interned, which only a runtime-built
/// name reaches, gets a block of its own.
fn module_dict_key_block(key: &str) -> *mut crate::unicodeobject::Utf8Str {
    if let Some(w_name) = crate::unicodeobject::get_interned_wtf8(rustpython_wtf8::Wtf8::new(key)) {
        // `celldict.py setitem_str` stores the unwrapped STR. For an interned
        // name that is `w_name._utf8`; the entries array traces the pointer
        // (`StrKey::GC_REF_OFFSETS`) so a moving collection forwards it.
        return unsafe { crate::unicodeobject::w_str_storage(w_name) };
    }
    crate::unicodeobject::alloc_utf8_payload(key.as_bytes(), true)
}

impl ModuleDictStorage {
    pub fn new() -> Self {
        Self {
            entries: ModuleDictEntries::default(),
        }
    }

    /// `dict.__len__`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `dict.__getitem__(key)` returning the raw stored value (or
    /// cell — eventually).  None when absent.
    pub fn get(&self, key: &str) -> Option<PyObjectRef> {
        module_dict_entries_get(&self.entries, key)
    }

    /// `dict[key] = w_value` — insertion-ordered.  Returns the
    /// previous value (or None if this is a fresh slot).
    pub fn set(&mut self, key: &str, w_value: PyObjectRef) -> Option<PyObjectRef> {
        // `celldict.py` `unerase(dstorage)[key] = w_value`. Storage is
        // Box-immortal. `remember_young_pointer` records the prebuilt walk
        // on the write, independently of the stored pointer.
        crate::gc_roots::mark_prebuilt_roots_dirty();
        module_dict_entries_insert(&mut self.entries, key, w_value)
    }

    /// `del dict[key]` — returns the removed value or None.
    ///
    /// Uses `shift_remove` (not `swap_remove`) so the remaining keys
    /// keep their relative insertion order, matching Python `dict`'s
    /// `__delitem__` semantics that `celldict.py items` /
    /// `:166-171 popitem` (LIFO) depend on.
    pub fn remove(&mut self, key: &str) -> Option<PyObjectRef> {
        self.entries.remove(key)
    }

    /// `dict.clear()`.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `&mut PyObjectRef` iterator over every stored value, for the
    /// GC custom-trace hook on W_ModuleDictObject.
    pub fn iter_values_mut(&mut self) -> impl Iterator<Item = &mut PyObjectRef> + '_ {
        self.entries.values_mut_for_trace()
    }
}

/// `pypy/objspace/std/celldict.py GlobalCache`:
///
/// ```python
/// class GlobalCache(object):
///     def __init__(self, cell):
///         # works like this: self.cell is always the result of
///         # getdictvalue_no_unwrapping on the equivalent key.
///         # this means it is None if the key doesn't exist, a w_value if there is
///         # no cell, or a Cell
///         #
///         # if the module dict actually switches to a different strategy, then
///         # cell is set to None, and valid to False
///         self.cell = cell
///         self.valid = True
///         self.ref = weakref.ref(self)
///         self.builtincache = None
///
///     @objectmodel.always_inline
///     def getvalue(self, space):
///         return unwrap_cell(space, self.cell)
/// ```
///
/// Per-name fast-path cache for `LOAD_GLOBAL`.  `cell` is the result
/// of `getdictvalue_no_unwrapping` at cache install time; `valid`
/// flips to `false` when the strategy invalidates (mutated() or
/// switch_to_object_strategy).  `builtincache` chains a nested cache
/// for the builtins fallback so that `LOAD_GLOBAL` of a name that
/// lives in builtins still hits one indirection rather than
/// re-traversing through `__builtins__.w_dict` every call.
pub struct GlobalCache {
    pub cell: Option<PyObjectRef>,
    pub valid: bool,
    /// `celldict.py cache.builtincache = builtincache`: stores the
    /// _same_ `GlobalCache` object that lives inside the builtin
    /// strategy's `caches` map, so a write through the builtin's
    /// strategy that updates `cache.cell` is immediately visible
    /// here.  PyPy stores this as a direct Python attribute
    /// (ref-counted strong ref); pyre uses `Arc<Mutex<...>>` so the
    /// cache stays alive as long as either the owning strategy's
    /// `caches` map OR a chained `builtincache` holds it — matching
    /// PyPy's ref-counted lifetime. The mutex supplies object-local
    /// synchronization for free-threaded access; it does not create a
    /// parallel side table or serialize unrelated bytecode execution.
    ///
    /// As in PyPy, invalidation is tied to
    /// `ModuleDictStrategy.invalidate_caches` dropping the registry.
    pub builtincache: Option<std::sync::Arc<parking_lot::Mutex<GlobalCache>>>,
}

impl GlobalCache {
    pub fn new(cell: Option<PyObjectRef>) -> Self {
        Self {
            cell,
            valid: true,
            builtincache: None,
        }
    }

    /// `celldict.py getvalue`: return the cached cell's
    /// unwrapped value, or `None` if the cache holds `None` (key
    /// absent at install time).
    #[inline]
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn getvalue(&self) -> Option<PyObjectRef> {
        // A cached cell whose unwrapped value is null is a stale/empty binding:
        // treat it as a cache miss rather than surfacing `Some(null)`.
        self.cell.and_then(|c| {
            let v = unwrap_cell(c);
            if v.is_null() { None } else { Some(v) }
        })
    }
}

/// `celldict.py` `self.caches.get(key)`. The registry mutex is the 3.14t
/// wrapper; the lookup itself is the dict get upstream performs under the GIL.
#[allow(dead_code)]
#[majit_macros::dont_look_inside]
fn caches_get(
    caches: &parking_lot::Mutex<Option<GlobalCacheRegistry>>,
    key: &str,
) -> Option<std::sync::Arc<parking_lot::Mutex<GlobalCache>>> {
    caches.lock().as_ref()?.get(key).cloned()
}

/// `celldict.py` `cache.cell = w_value`. Residual so `MutexGuard::deref_mut`
/// is not a translation subject.
#[majit_macros::dont_look_inside]
fn cache_set_cell(
    cache: &std::sync::Arc<parking_lot::Mutex<GlobalCache>>,
    cell: Option<PyObjectRef>,
) {
    cache.lock().cell = cell;
}

/// `caches_get` plus `cache.cell =` under the registry lock so a
/// concurrent `delitem` cannot drop the entry and then have this
/// store resurrect it.
#[majit_macros::dont_look_inside]
fn caches_set_cell(
    caches: &parking_lot::Mutex<Option<GlobalCacheRegistry>>,
    key: &str,
    cell: Option<PyObjectRef>,
) {
    let guard = caches.lock();
    if let Some(cache) = guard
        .as_ref()
        .and_then(|registry| registry.get(key))
        .cloned()
    {
        cache_set_cell(&cache, cell);
    }
}

/// GC root walk over a single `GlobalCache` and its chained
/// `builtincache`, forwarding the movable value each `cell` holds.
///
/// Mirrors the `GlobalCache` object graph shape (`celldict.py`):
/// `cell` plus the nested `builtincache`, which always points at a
/// DIFFERENT `GlobalCache` instance (`celldict.py
/// builtin_strategy.get_global_cache(...)`), so the outer borrow is
/// dropped before recursing.  The cached `cell` duplicates the storage
/// entry it was installed from, so a collection that relocates that
/// value must rewrite the cache copy too or the next cached
/// `LOAD_GLOBAL` reads a stale pointer.
///
/// # Safety
/// `cache.cell`, when present, must be a valid `PyObjectRef`.
unsafe fn walk_one_global_cache(
    cache: &std::sync::Arc<parking_lot::Mutex<GlobalCache>>,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let mut c = cache.lock();
    if let Some(slot) = c.cell.as_mut() {
        walk_module_value_slot(slot, visitor);
    }
    if let Some(builtincache) = c.builtincache.clone() {
        drop(c);
        walk_one_global_cache(&builtincache, visitor);
    }
}

/// `pypy/objspace/std/celldict.py ModuleDictStrategy`.
///
/// ```python
/// class ModuleDictStrategy(DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("modulecell")
///     erase = staticmethod(erase)
///     unerase = staticmethod(unerase)
///
///     _immutable_fields_ = ["version?"]
///
///     def __init__(self, space):
///         self.space = space
///         self.version = VersionTag()
///         self.caches = None
/// ```
///
/// Pyre's port omits the `erase / unerase` static methods because
/// Rust's strong typing makes the rerasure unnecessary —
/// `ModuleDictStorage` is the concrete storage type directly.
///
/// `caches` is the per-name `GlobalCache` registry consulted by the
/// `LOAD_GLOBAL` fast path (`celldict.py get_global_cache`).
/// Allocated lazily on first cache install. The mutex protects this
/// strategy-owned registry when multiple Python threads share a module.
pub struct ModuleDictStrategy {
    pub version: VersionTag,
    pub caches: parking_lot::Mutex<Option<GlobalCacheRegistry>>,
    /// The hidden `mutate_version` field for `celldict.py:34
    /// _immutable_fields_ = ["version?"]`.  Each compiled loop whose trace
    /// promoted `self.version` (and folded a module-global lookup keyed on it)
    /// registers its `JitCellToken` invalidation flag here; `mutated()`
    /// reassigns `version`, which under the `?` declaration must invalidate
    /// every such loop.
    ///
    /// The same [`crate::quasiimmut::QuasiImmutField`] `W_TypeObject`'s
    /// `_version_tag?` uses, as upstream's one `QuasiImmut` class serves every
    /// quasi-immutable field.  Before that it was a bare `Vec` pushed to
    /// without synchronisation while `mutated()` swept it from another thread,
    /// and it had no `compress_looptokens_list`, so a module recompiled against
    /// many times and never mutated grew one entry per compile.
    version_watchers: crate::quasiimmut::QuasiImmutField,
}

/// Runtime-assigned GC type id for the [`ModuleDictStrategy`] box.
static MODULE_DICT_STRATEGY_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`ModuleDictStrategy`] box.
pub fn set_module_dict_strategy_gc_type_id(id: u32) {
    MODULE_DICT_STRATEGY_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`ModuleDictStrategy`] box.
#[majit_macros::dont_look_inside]
pub fn module_dict_strategy_gc_type_id() -> u32 {
    MODULE_DICT_STRATEGY_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl Default for ModuleDictStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDictStrategy {
    /// `celldict.py __init__`.
    pub fn new() -> Self {
        Self {
            version: VersionTag::fresh(),
            caches: parking_lot::Mutex::new(None),
            version_watchers: crate::quasiimmut::QuasiImmutField::new(),
        }
    }

    /// `quasiimmut.py get_current_qmut_instance` for the `version?`
    /// field — resolve the instance while the trace is still recording, so a
    /// `mutated()` reached later in the same trace finds the field non-null and
    /// the recording can carry the instance to `compile.py:204-207`.
    pub fn current_version_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.version_watchers.get_current_qmut_instance()
    }

    /// `pyjitpl.py MIFrame.opimpl_jit_force_quasi_immutable mutatebox.nonnull()` — whether some trace or loop is
    /// watching `version?` right now.
    pub fn version_qmut_installed(&self) -> bool {
        self.version_watchers.is_installed()
    }

    /// `quasiimmut.py do_force_quasi_immutable`, which the tracer calls
    /// itself before aborting (`pyjitpl.py:1113-1115`). Idempotent: a field
    /// already taken returns early, so the interpreter re-running the opcode
    /// after the abort forces nothing a second time.
    pub fn force_version_qmut(&self) {
        self.version_watchers.invalidate();
    }

    /// Invalidate every loop watching `version`
    /// (`quasiimmut.py _invalidate_now`).  Sets each live flag to
    /// `true`, the polarity `GuardNotInvalidated` tests.
    ///
    /// The installed check stays traced and the sweep is residual, for the
    /// reason upstream's own walk is out of line: it hangs off
    /// `jit_force_quasi_immutable`, never a trace.
    fn notify_version_watchers(&self) {
        if !self.version_watchers.is_installed() {
            return;
        }
        unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.version_watchers) };
    }

    /// `celldict.py get_global_cache`:
    ///
    /// ```python
    /// def get_global_cache(self, w_dict, key):
    ///     space = w_dict.space
    ///     if self.caches is None:
    ///         cache = None
    ///         self.caches = {}
    ///     else:
    ///         cache = self.caches.get(key, None)
    ///     if cache is None:
    ///         cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///         cache = GlobalCache(cell)
    ///         if (not space.config.objspace.honor__builtins__ and
    ///                 cell is None and
    ///                 w_dict is not space.builtin.w_dict):
    ///             w_builtin_dict = space.builtin.w_dict
    ///             assert isinstance(w_builtin_dict, W_ModuleDictObject)
    ///             builtin_strategy = w_builtin_dict.mstrategy
    ///             if isinstance(builtin_strategy, ModuleDictStrategy):
    ///                 cell = builtin_strategy.getdictvalue_no_unwrapping(
    ///                         w_builtin_dict, key)
    ///                 # logic: if the global is not defined but the builtin is,
    ///                 # cache it. otherwise don't cache the builtin ever
    ///                 if cell is not None:
    ///                     builtincache = builtin_strategy.get_global_cache(
    ///                             w_builtin_dict, key)
    ///                     cache.builtincache = builtincache
    ///         self.caches[key] = cache
    ///     return cache
    /// ```
    ///
    /// `w_builtin_dict` is `space.builtin.w_dict` when
    /// `honor__builtins__` is off, and null when it is on: `pyre-object`
    /// has no `space` to read either from, so the interpreter-side caller
    /// resolves both.
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    pub fn get_global_cache(
        &mut self,
        w_dict: PyObjectRef,
        key: &str,
        w_builtin_dict: PyObjectRef,
    ) -> std::sync::Arc<parking_lot::Mutex<GlobalCache>> {
        let mut cache_registry = self.caches.lock();
        // `celldict.py`:
        //     if self.caches is None:
        //         cache = None
        //         self.caches = {}
        //     else:
        //         cache = self.caches.get(key, None)
        //
        // One probe, not a `contains_key` followed by a `get`: `key` is the
        // global's name and this runs whenever a code object's slot for it is
        // empty, so the second hash was paid on every hit.
        let cached = match cache_registry.as_mut() {
            None => {
                *cache_registry = Some(GlobalCacheRegistry::default());
                None
            }
            Some(caches) => caches.get(key).cloned(),
        };
        if let Some(cache) = cached {
            return cache;
        }
        let cell = self.getdictvalue_no_unwrapping(w_dict, key);
        let mut cache = GlobalCache::new(cell);
        if !w_builtin_dict.is_null()
            && cell.is_none()
            && !std::ptr::eq(w_dict, w_builtin_dict)
            && unsafe { crate::dictmultiobject::is_module_dict(w_builtin_dict) }
            && !unsafe { crate::dictmultiobject::w_module_dict_is_object_strategy(w_builtin_dict) }
        {
            // A different dict's strategy, so no alias of `self`; its
            // `caches` lock nests inside this one and never the reverse.
            let builtin_strategy = unsafe {
                crate::dictmultiobject::w_module_dict_module_strategy_mut(w_builtin_dict)
            };
            if builtin_strategy
                .getdictvalue_no_unwrapping(w_builtin_dict, key)
                .is_some()
            {
                cache.builtincache = Some(builtin_strategy.get_global_cache(
                    w_builtin_dict,
                    key,
                    crate::pyobject::PY_NULL,
                ));
            }
        }
        let caches = cache_registry.as_mut().unwrap();
        // `celldict.py` `self.caches[key] = cache`. The registry is
        // Box-immortal. `remember_young_pointer` records the prebuilt walk
        // on the write, independently of the cached cell.
        crate::gc_roots::mark_prebuilt_roots_dirty();
        let cache = std::sync::Arc::new(parking_lot::Mutex::new(cache));
        caches.insert(key.to_string(), cache.clone());
        cache
    }

    /// `celldict.py switch_to_object_strategy` cache flush:
    ///
    /// ```python
    /// if self.caches is not None:
    ///     for cache in self.caches.itervalues():
    ///         cache.cell = None
    ///         cache.valid = False
    ///     self.caches = None
    /// ```
    ///
    /// Invalidate every live cache and drop the registry.  Called from
    /// the host `switch_to_object_strategy` helper on
    /// W_ModuleDictObject.
    pub fn invalidate_caches(&mut self) {
        let mut cache_registry = self.caches.lock();
        if let Some(caches) = cache_registry.as_mut() {
            for cache in caches.values() {
                let mut c = cache.lock();
                c.cell = None;
                c.valid = false;
                c.builtincache = None;
            }
        }
        *cache_registry = None;
    }

    /// `celldict.py get_empty_storage`:
    ///
    /// ```python
    /// def get_empty_storage(self):
    ///     return self.erase({})
    /// ```
    pub fn get_empty_storage(&self) -> ModuleDictStorage {
        ModuleDictStorage::new()
    }

    /// `celldict.py mutated`:
    ///
    /// ```python
    /// def mutated(self):
    ///     self.version = VersionTag()
    /// ```
    ///
    /// Reassigning the `version?` quasi-immutable field invalidates the JIT.
    /// `rclass.py hook_setfield` notifies watchers before the store;
    /// pyre flips the registered loop flags explicitly at this write site.
    #[inline]
    pub fn mutated(&mut self) {
        self.notify_version_watchers();
        self.version = VersionTag::fresh();
    }

    /// `celldict.py getdictvalue_no_unwrapping`:
    ///
    /// ```python
    /// def getdictvalue_no_unwrapping(self, w_dict, key):
    ///     self = jit.promote(self)
    ///     return self._getdictvalue_no_unwrapping_pure(
    ///         self.version, w_dict, key)
    ///
    /// @jit.elidable_promote('0,1,2')
    /// def _getdictvalue_no_unwrapping_pure(self, version, w_dict, key):
    ///     return self.unerase(w_dict.dstorage).get(key, None)
    /// ```
    ///
    /// Returns the raw stored value (a `MutableCell` or a plain
    /// `PyObjectRef`) without unwrapping a cell.
    pub fn getdictvalue_no_unwrapping(
        &self,
        w_dict: PyObjectRef,
        key: &str,
    ) -> Option<PyObjectRef> {
        self._getdictvalue_no_unwrapping_pure(self.version, w_dict, key)
    }

    /// `celldict.py _getdictvalue_no_unwrapping_pure` — keep the module dict
    /// object as the storage owner and unerase its `dstorage` here.  This is
    /// the load-bearing PyPy shape: callers never pass the concrete erased
    /// storage as a substitute for `w_dict`.
    #[majit_macros::elidable_promote(promote_args = "0,1,2")]
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the object-space boundary"
    )]
    pub fn _getdictvalue_no_unwrapping_pure(
        &self,
        _version: VersionTag,
        w_dict: PyObjectRef,
        key: &str,
    ) -> Option<PyObjectRef> {
        unsafe { crate::dictmultiobject::w_module_dict_module_storage(w_dict).get(key) }
    }

    /// `celldict.py setitem_str`:
    ///
    /// ```python
    /// def setitem_str(self, w_dict, key, w_value):
    ///     cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///     return self._setitem_str_cell_known(cell, w_dict, key, w_value)
    /// ```
    pub fn setitem_str(&mut self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        let cell = self.getdictvalue_no_unwrapping(w_dict, key);
        self._setitem_str_cell_known(cell, w_dict, key, w_value);
    }

    /// `celldict.py _setitem_str_cell_known`:
    ///
    /// ```python
    /// def _setitem_str_cell_known(self, cell, w_dict, key, w_value):
    ///     w_value = write_cell(self.space, cell, w_value)
    ///     if w_value is None:
    ///         return
    ///     self.mutated()
    ///     self.unerase(w_dict.dstorage)[key] = w_value
    ///     if self.caches is None:
    ///         return
    ///     cache = self.caches.get(key, None)
    ///     if cache:
    ///         cache.cell = w_value
    /// ```
    ///
    /// `write_cell` absorbs a rewrite in place whenever the slot already
    /// holds an `ObjectMutableCell` (or an `IntMutableCell` and the value is
    /// a plain int), returning `None` so the version is left alone.  Only a
    /// structural change — installing a key that had no cell, or replacing
    /// the cell kind — reaches `mutated()`.  That is what keeps a hot
    /// module-level counter increment from invalidating every loop that
    /// folded a module-global.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
    )]
    pub fn _setitem_str_cell_known(
        &mut self,
        cell: Option<PyObjectRef>,
        mut w_dict: PyObjectRef,
        key: &str,
        w_value: PyObjectRef,
    ) {
        let Some(w_to_store) = crate::with_roots!(w_dict => unsafe { write_cell(cell, w_value) })
        else {
            // In-place cell mutation: storage slot unchanged, version
            // stays valid (matches the JIT-cache-stable fast path).
            return;
        };
        self.mutated();
        unsafe {
            crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict).set(key, w_to_store);
        }
        // The store lands in `w_dict.dstorage`; an in-place `write_cell`
        // above barriered the cell it wrote instead.
        crate::dictmultiobject::dict_write_barrier(w_dict);
        // `celldict.py:88-90`: keep any live cache for `key` in step
        // with the new stored value so subsequent LOAD_GLOBAL through
        // the cache reads the fresh entry without an invalidation
        // round-trip.
        caches_set_cell(&self.caches, key, Some(w_to_store));
    }

    /// `celldict.py length`:
    ///
    /// ```python
    /// def length(self, w_dict):
    ///     return len(self.unerase(w_dict.dstorage))
    /// ```
    pub fn length(&self, w_dict: PyObjectRef) -> usize {
        unsafe { crate::dictmultiobject::w_module_dict_module_storage(w_dict).len() }
    }

    /// `celldict.py getitem_str`:
    ///
    /// ```python
    /// def getitem_str(self, w_dict, key):
    ///     cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///     return unwrap_cell(self.space, cell)
    /// ```
    pub fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        let raw = self.getdictvalue_no_unwrapping(w_dict, key)?;
        // `unwrap_cell` is null-tolerant and an `ObjectMutableCell` may hold a
        // null `w_value`; a null unwrap means the name has no live binding, so
        // report absence rather than `Some(null)`.
        let v = unsafe { unwrap_cell(raw) };
        if v.is_null() { None } else { Some(v) }
    }

    /// `celldict.py delitem` — minimal str-key path
    /// (`celldict.py`); the object-fallback /
    /// `_never_equal_to_string` branches belong to the full strategy
    /// dispatch once `ObjectDictStrategy` is wired.
    pub fn delitem_str(&mut self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        let removed = unsafe {
            crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict).remove(key)?
        };
        // `celldict.py:117-121`: zero out the per-key cache
        // so LOAD_GLOBAL falls through to the builtins
        // fallback (or NameError) on the next read.
        caches_set_cell(&self.caches, key, None);
        self.mutated();
        Some(removed)
    }

    /// `celldict.py clear`:
    ///
    /// ```python
    /// def clear(self, w_dict):
    ///     self.unerase(w_dict.dstorage).clear()
    ///     self.mutated()
    /// ```
    pub fn clear(&mut self, w_dict: PyObjectRef) {
        unsafe {
            crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict).clear();
        }
        self.mutated();
    }

    /// `celldict.py getiterkeys`:
    ///
    /// ```python
    /// def getiterkeys(self, w_dict):
    ///     return self.unerase(w_dict.dstorage).iterkeys()
    /// ```
    pub fn getiterkeys<'a>(
        &self,
        storage: &'a ModuleDictStorage,
    ) -> impl Iterator<Item = &'a str> + 'a {
        storage.entries.keys().map(|k| k.as_str())
    }

    /// `celldict.py getitervalues`:
    ///
    /// ```python
    /// def getitervalues(self, w_dict):
    ///     return self.unerase(w_dict.dstorage).itervalues()
    /// ```
    ///
    /// The skeleton omits the per-element `unwrap_cell` because no
    /// cells are stored yet (see `_setitem_str_cell_known`).
    pub fn getitervalues<'a>(
        &self,
        storage: &'a ModuleDictStorage,
    ) -> impl Iterator<Item = PyObjectRef> + 'a {
        // `celldict.py values`: each cell is unwrapped before
        // it crosses out of the strategy.  Without unwrapping, JIT-
        // promoted cell objects would leak into user space and break
        // identity comparisons against the previously-stored value.
        storage.entries.values().map(|v| unsafe { unwrap_cell(*v) })
    }

    /// The name in slot `index`, or `None` past the end and for a tombstone.
    ///
    /// `getiterkeys` walks the whole map; a dict view's integer cursor wants
    /// one entry, so ask the storage directly.
    pub fn nth_key<'a>(&self, storage: &'a ModuleDictStorage, index: usize) -> Option<&'a str> {
        storage.entries.get_slot(index).map(|(k, _)| k.as_str())
    }

    /// The first live entry slot at or after `from` — `_ll_dictnext`
    /// (`rordereddict.py`).  A name deleted from a module leaves a
    /// tombstone, so a walk steps over the holes rather than counting.
    pub fn next_entry_slot(&self, storage: &ModuleDictStorage, from: usize) -> Option<usize> {
        storage.entries.next_valid_slot(from)
    }

    /// [`Self::next_entry_slot`] descending, for `reversed(module.__dict__)`.
    pub fn prev_entry_slot(&self, storage: &ModuleDictStorage, before: usize) -> Option<usize> {
        storage.entries.prev_valid_slot(before)
    }

    /// [`Self::nth_key`]'s value half, unwrapped the way `getitervalues`
    /// unwraps (`celldict.py values`).
    pub fn nth_unwrapped_value(
        &self,
        storage: &ModuleDictStorage,
        index: usize,
    ) -> Option<PyObjectRef> {
        storage
            .entries
            .get_slot(index)
            .map(|(_, v)| unsafe { unwrap_cell(*v) })
    }

    /// GC root walk over every live `GlobalCache.cell` reachable through
    /// this strategy's `caches` registry (`celldict.py
    /// get_global_cache`), forwarding the movable value each cell holds.
    ///
    /// # Safety
    /// Each cached `GlobalCache.cell`, when present, must be a valid
    /// `PyObjectRef`.
    pub unsafe fn walk_cache_cells(&self, visitor: &mut dyn FnMut(&mut PyObjectRef)) {
        if let Some(caches) = self.caches.lock().as_ref() {
            for cache in caches.values() {
                walk_one_global_cache(cache, visitor);
            }
        }
    }
}

/// `pypy/objspace/std/celldict.py ModuleDictStrategy(DictStrategy)`
/// — abstract base inheritance.  Every method takes the `W_ModuleDict
/// Object` (`w_dict: PyObjectRef`) and resolves its `dstorage`
/// internally, matching PyPy's strategy contract per
/// `dictmultiobject.py DictStrategy`.
///
/// The pyre inherent methods on `ModuleDictStrategy`
/// (`setitem_str(&self, &mut ModuleDictStorage, …)` etc.) are a
/// pre-existing pyre adaptation that pairs strategy with storage
/// directly — kept for `celldict::tests` plus a handful of legacy
/// callers, but the canonical surface going forward is the
/// trait dispatch below.
impl crate::dictmultiobject::DictStrategy for ModuleDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Module
    }

    /// `celldict.py get_empty_storage` — pyre owns the
    /// `ModuleDictStorage` directly (no `rerased` indirection); return
    /// the storage as an erased `GCREF` so the trait surface stays
    /// strategy-agnostic.
    fn get_empty_storage(&self) -> crate::gc_hook::GCREF {
        // `celldict.py get_empty_storage`: `self.erase({})` is a nursery
        // `dicttable`. A `malloc_raw` table has no MiniMark header, so
        // `ll_dict_grow`'s `d.entries = newitems` write barrier cannot
        // remember the young `DICTENTRYARRAY`.
        crate::gc_storage::gc_alloc_young_storage_box(
            ModuleDictStorage::new(),
            module_dict_storage_gc_type_id(),
        ) as crate::gc_hook::GCREF
    }

    /// `celldict.py getitem` — str fast path, else
    /// `switch_to_object_strategy` then walk unified entries.
    /// Body in `w_module_dict_lookup_inner` to avoid recursing
    /// through `w_dict_lookup` (which dispatches back through
    /// the strategy slot after Phase C-3 wire-in).
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_lookup_inner(w_dict, w_key)
    }

    /// `celldict.py getitem_str` — str fast path matches
    /// `w_module_dict_getitem_str` and its cell-cache behavior.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_getitem_str(w_dict, key)
    }

    /// `celldict.py setitem` + `_setitem_str_cell_known` — str
    /// fast path; non-str keys force `switch_to_object_strategy`.
    /// Body in `w_module_dict_store_inner` to avoid recursing through
    /// `w_dict_store` (which dispatches back through the strategy
    /// slot after Phase C-3 wire-in).
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_store_inner(w_dict, w_key, w_value);
    }

    /// `celldict.py setitem_str`.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_setitem_str(w_dict, key, w_value);
    }

    /// `celldict.py delitem` — str fast path, else
    /// `switch_to_object_strategy` then walk unified entries.
    /// Body in `w_module_dict_delitem_inner` to avoid recursing
    /// through `w_dict_delitem` (which dispatches back through
    /// the strategy slot after Phase C-3 wire-in).
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        crate::dictmultiobject::w_module_dict_delitem_inner(w_dict, w_key)
    }

    /// `celldict.py length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_module_dict_length(w_dict)
    }

    /// `celldict.py w_keys` — `space.newlist(self.unerase
    /// (w_dict.dstorage).keys())`; pyre returns the wrapped key
    /// PyObjectRefs directly so callers can build whatever container
    /// they need.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        crate::dictmultiobject::w_dict_items(w_dict)
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }

    /// `celldict.py values` — reads the cells and nothing else.
    /// Routing this through `items` wrapped every name into a
    /// `W_UnicodeObject` only to drop it.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_values_inner(w_dict)
    }

    /// `celldict.py items` — branches on `is_object_strategy`
    /// and emits whichever storage half is live, wrapping str keys
    /// via `w_str_new`.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_module_dict_items_inner(w_dict)
    }

    /// `celldict.py getiterkeys`/`getitervalues` — one entry, not the
    /// whole storage.  The trait default rebuilds `items()` per step, and this
    /// strategy's `items` wraps every name into a fresh immortal string, so the
    /// default makes a single view walk quadratic in unreclaimable allocations.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_module_dict_nth_item_inner(w_dict, index)
    }

    /// A `values()` view needs no name at all, so it skips the wrap entirely
    /// (`dictmultiobject.py AbstractTypedStrategy.getitem`).
    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> Option<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_nth_value_inner(w_dict, index)
    }

    /// Both storage halves tombstone their deletes, so neither numbers its
    /// entries densely; the default would walk positions.
    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage(w_dict) {
            return entries.next_valid_slot(from);
        }
        self.next_entry_slot(
            crate::dictmultiobject::w_module_dict_module_storage(w_dict),
            from,
        )
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage(w_dict) {
            return entries.prev_valid_slot(before);
        }
        self.prev_entry_slot(
            crate::dictmultiobject::w_module_dict_module_storage(w_dict),
            before,
        )
    }

    /// `celldict.py clear`.  Branches on
    /// `is_object_strategy` and drains whichever storage half is live.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_clear_inner(w_dict);
    }

    /// `celldict.py popitem` — pop the most recently inserted
    /// (key, cell) from the IndexMap, mutated(), unwrap the cell, and
    /// return (`_wrapkey(space, key)`, `unwrap_cell(space, cell)`).
    /// O(1) via `IndexMap::pop`; after a `switch_to_object_strategy` the
    /// entries live in ObjectDictStrategy's erased storage and are popped
    /// already unwrapped.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage_mut_opt(w_dict)
        {
            let (k, v) = entries.pop()?;
            // `mutated()` on this half too: the object storage is still reached
            // through this strategy, so a compiled trace that pinned a global
            // stays valid until `version` is reassigned. `delitem` pairs the
            // two calls the same way on the object-storage arm.
            crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
            return Some((k.obj, v));
        }
        let strategy = crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict);
        let storage = crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict);
        let (key, cell) = storage.entries.pop()?;
        strategy.mutated();
        crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
        Some((_wrapkey(key.as_str()), unwrap_cell(cell)))
    }

    /// `celldict.py getiterreversed` — reverse iteration
    /// over the IndexMap's key insertion order (used by `reversed
    /// (module.__dict__)`).  Native streaming reverse via
    /// `IndexMap::iter().rev()`; the wrap_cell unwrap matches PyPy's
    /// `wrapvalue(space, value) = unwrap_cell(space, value)` per
    /// `:208 wrapvalue`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage(w_dict) {
            return entries.iter().rev().map(|(k, &v)| (k.obj, v)).collect();
        }
        let storage = crate::dictmultiobject::w_module_dict_module_storage(w_dict);
        storage
            .entries
            .iter()
            .rev()
            .map(|(k, &cell)| (_wrapkey(k.as_str()), unwrap_cell(cell)))
            .collect()
    }

    /// `celldict.py ModuleDictStrategy.copy` — a fresh `W_DictObject`
    /// on `UnicodeDictStrategy` filled from unwrapped cells.
    ///
    /// `fromcache(UnicodeDictStrategy)` / `get_empty_storage()`, then
    /// `str_dict[newtext(key)] = unwrap_cell(cell)`. Every key is a
    /// str, so the fill stays on the strategy it was born with.
    ///
    /// `switch_to_object_strategy` replaces `mstrategy` with
    /// `ObjectDictStrategy`, so this method is not reached after the
    /// switch; that state copies through `AbstractTypedStrategy.copy`.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.pin_roots(&[w_dict]);
        // Snapshot names and unwrapped values while the source is pinned and
        // nothing collects, then publish the values before minting the dest.
        // `_wrapkey` / `w_dict_store` collect; the dest is a moving nursery
        // `W_Dict` (`w_dict_new_with`), so each store reloads it.
        let mut keys = Vec::new();
        let mut values = Vec::new();
        {
            let storage =
                crate::dictmultiobject::w_module_dict_module_storage(roots.get(dict_slot));
            for (key, &cell) in storage.entries.iter() {
                let unwrapped = unwrap_cell(cell);
                // An empty cell names nothing — `getitem_str` reads a null unwrap
                // as absence — so it does not become a null-valued entry of the
                // ordinary dict this hands back.
                if unwrapped.is_null() {
                    continue;
                }
                keys.push(key.as_str().to_owned());
                values.push(unwrapped);
            }
        }
        let values_base = roots.pin_roots(&values);
        let strategy = &crate::dictmultiobject::UNICODE_DICT_STRATEGY_REF;
        let new_slot = roots.publish(&[crate::dictmultiobject::w_dict_new_with(
            strategy,
            strategy.get_empty_storage(),
        )]);
        for (i, key) in keys.iter().enumerate() {
            let key_slot = roots.pin_roots(&[_wrapkey(key)]);
            crate::dictmultiobject::w_dict_store(
                roots.get(new_slot),
                roots.get(key_slot),
                roots.get(values_base + i),
            );
        }
        roots.get(new_slot)
    }
}

/// `celldict.py remove_cell(w_dict, space, name)` — replace
/// any cell wrapper at `name` with the unwrapped value so subsequent
/// reads observe the raw value directly (used when a module-level
/// name is rebound in a context that no longer needs cell
/// indirection, e.g. function-def replacing a previously
/// cell-promoted slot).
///
/// ```python
/// def remove_cell(w_dict, space, name):
///     if isinstance(w_dict, W_DictMultiObject):
///         strategy = w_dict.get_strategy()
///         if isinstance(strategy, ModuleDictStrategy):
///             w_value = strategy.getitem_str(w_dict, name)
///             dict_w = strategy.unerase(w_dict.dstorage)
///             strategy.mutated()
///             dict_w[name] = w_value  # store without cell
/// ```
///
/// Pyre's W_ModuleDictObject path: peek the unwrapped value via
/// `getitem_str` (which already calls `unwrap_cell`), bump the
/// strategy version (cache invalidate), and write back the raw
/// PyObjectRef via `ModuleDictStorage::set` — bypassing
/// `_setitem_str_cell_known`'s `write_cell` re-wrap.
///
/// # Safety
/// `w_dict` must point at a valid PyObjectRef (W_ModuleDictObject
/// or null/other type — no-op for non-module dicts).
pub unsafe fn remove_cell(w_dict: PyObjectRef, name: &str) {
    if w_dict.is_null() {
        return;
    }
    if !std::ptr::eq(
        (*(w_dict as *const crate::pyobject::PyObject)).ob_type,
        &crate::dictmultiobject::MODULE_DICT_TYPE,
    ) {
        return;
    }
    // The body runs only under `isinstance(strategy, ModuleDictStrategy)`.
    // Past `switch_to_object_strategy` the erased slots hold the object
    // strategy and an `ObjectDictStorage`, and no cell is left to unwrap.
    if crate::dictmultiobject::w_module_dict_is_object_strategy(w_dict) {
        return;
    }
    let strategy = crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict);
    let Some(w_value) = strategy.getitem_str(w_dict, name) else {
        return;
    };
    strategy.mutated();
    let storage = crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict);
    storage.set(name, w_value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_versiontags_are_distinct() {
        let a = VersionTag::fresh();
        let b = VersionTag::fresh();
        assert_ne!(a, b);
    }

    #[test]
    fn mutated_flips_registered_version_watchers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut strategy = ModuleDictStrategy::new();
        let flag = Arc::new(AtomicBool::new(false));
        let token = crate::quasiimmut::test_loop_token(&flag);
        strategy.current_version_qmut().register_loop_token(&token);
        // Before a structural change the watching loop is still valid.
        assert!(!flag.load(Ordering::Acquire));
        // `mutated()` reassigns `version` and must invalidate the loop.
        strategy.mutated();
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn dead_version_watchers_drop_out() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let mut strategy = ModuleDictStrategy::new();
        let flag = Arc::new(AtomicBool::new(false));
        let token = crate::quasiimmut::test_loop_token(&flag);
        strategy.current_version_qmut().register_loop_token(&token);
        // Drop the only strong ref: the weak watcher can no longer upgrade.
        drop(token);
        // notify (via mutated) must not panic, and `_invalidate_now` unlinks
        // the instance whether or not any flag could still be upgraded.
        strategy.mutated();
        assert!(!strategy.version_watchers.is_installed());
    }

    #[test]
    fn module_dict_store_and_global_cache_insert_dirties_prebuilt_roots() {
        {
            // module_dict_store_dirties_prebuilt_roots
            let w_dict = crate::dictmultiobject::w_module_dict_new();
            let storage =
                unsafe { crate::dictmultiobject::w_module_dict_module_storage_mut(w_dict) };
            crate::gc_roots::clear_prebuilt_roots_dirty();
            storage.set("k", crate::w_str_new("v"));
            assert!(crate::gc_roots::prebuilt_roots_dirty());
            let cell = crate::w_str_new("cell") as crate::gc_hook::GCREF;
            crate::gc_roots::clear_prebuilt_roots_dirty();
            object_mutable_cell_write_barrier(cell);
            assert!(crate::gc_roots::prebuilt_roots_dirty());
        }
        {
            // global_cache_insert_dirties_prebuilt_roots
            let w_dict = crate::dictmultiobject::w_module_dict_new();
            let strat =
                unsafe { crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict) };
            strat.setitem_str(w_dict, "kept", crate::w_str_new("kept"));
            crate::gc_roots::clear_prebuilt_roots_dirty();
            let _cache = strat.get_global_cache(w_dict, "kept", crate::pyobject::PY_NULL);
            assert!(crate::gc_roots::prebuilt_roots_dirty());
        }
    }

    #[test]
    fn setitem_getitem_roundtrip() {
        let w_dict = crate::dictmultiobject::w_module_dict_new();
        let strat = unsafe { crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict) };
        let v = crate::w_str_new("hello");
        strat.setitem_str(w_dict, "x", v);
        assert_eq!(strat.getitem_str(w_dict, "x"), Some(v));
        assert_eq!(strat.length(w_dict), 1);
    }

    #[test]
    fn setitem_bumps_version() {
        let w_dict = crate::dictmultiobject::w_module_dict_new();
        let strat = unsafe { crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict) };
        let before = strat.version;
        strat.setitem_str(w_dict, "k", crate::w_str_new("v"));
        assert_ne!(strat.version, before);
    }

    #[test]
    fn delitem_removes_and_bumps() {
        let w_dict = crate::dictmultiobject::w_module_dict_new();
        let strat = unsafe { crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict) };
        let v = crate::w_str_new("v");
        strat.setitem_str(w_dict, "k", v);
        let v_before = strat.version;
        let removed = strat.delitem_str(w_dict, "k");
        assert_eq!(removed, Some(v));
        assert_eq!(strat.getitem_str(w_dict, "k"), None);
        assert_ne!(strat.version, v_before);
    }

    #[test]
    fn int_cell_indirection_on_rewrite() {
        // After a second write with an int value, the strategy should
        // wrap the value in `IntMutableCell` and skip the version bump
        // (typeobject.py:61-63).
        let w_dict = crate::dictmultiobject::w_module_dict_new();
        let strat = unsafe { crate::dictmultiobject::w_module_dict_module_strategy_mut(w_dict) };
        unsafe {
            let v0 = crate::intobject::w_int_new(7);
            strat.setitem_str(w_dict, "k", v0);
            let v1 = crate::intobject::w_int_new(8);
            strat.setitem_str(w_dict, "k", v1);
            // getitem_str unwraps the cell back to the int value.
            let got = strat.getitem_str(w_dict, "k").unwrap();
            assert_eq!(crate::intobject::w_int_get_value(got), 8);
        }
    }

    /// `store_would_bump_version` is the tracer's stand-in for the write it
    /// never looks into, so it has to answer for the same five shapes
    /// `classify_cell_write` distinguishes.  Pin both halves against one table
    /// so neither can drift from the other.
    #[test]
    fn store_bump_prediction_matches_the_write() {
        unsafe {
            let int7 = crate::intobject::w_int_new(7);
            let int8 = crate::intobject::w_int_new(8);
            let s = crate::w_str_new("s");
            let obj_cell = w_object_mutable_cell_new(s);
            let int_cell = w_int_mutable_cell_new(7);
            // Distinct objects, so these reach `is_w` past its pointer check.
            let f1 = crate::floatobject::w_float_new(1.5);
            let f2 = crate::floatobject::w_float_new(1.5);
            assert!(!std::ptr::eq(f1, f2));
            let ab1 = crate::w_str_new("ab");
            let ab2 = crate::w_str_new("ab");
            assert!(!std::ptr::eq(ab1, ab2));

            // (raw slot contents, value written, bumps?, what the write does)
            let cases: [(Option<PyObjectRef>, PyObjectRef, bool, &str); 8] = [
                (None, int7, true, "no cell: the value is stored bare"),
                (Some(obj_cell), s, false, "object cell: written in place"),
                (
                    Some(int_cell),
                    int8,
                    false,
                    "int cell + plain int: in place",
                ),
                (Some(int_cell), s, true, "int cell + non-int: fresh cell"),
                (Some(s), s, false, "bare value, identical: unchanged"),
                (Some(int7), int8, true, "bare value, different: fresh cell"),
                (
                    Some(f1),
                    f2,
                    false,
                    "equal exact floats: `is_w` identical, unchanged",
                ),
                (
                    Some(ab1),
                    ab2,
                    true,
                    "equal strs with separate storage: fresh cell",
                ),
            ];
            for (w_cell, w_value, bumps, what) in cases {
                assert_eq!(store_would_bump_version(w_cell, w_value), bumps, "{what}");
                // The write agrees: it returns a replacement slot value exactly
                // when it bumps, and `None` (wrote through / no-op) otherwise.
                assert_eq!(write_cell(w_cell, w_value).is_some(), bumps, "{what}");
            }
        }
    }
}
