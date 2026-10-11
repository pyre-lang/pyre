//! pypy/objspace/std/mapdict.py
//!
//! Mapdict provides per-instance dict and weakref slots for hasdict /
//! weakrefable types. PyPy stores these inside the mapdict map's "dict"
//! and "weakref" SPECIAL slots; pyre does the same for a `_getusercls`
//! layout whose base typedef is not weakrefable. `make_weakref_descr`
//! stores `_lifeline_` on the instance instead.
//!
//! The names below mirror PyPy: `MapdictDictSupport.getdict` →
//! `_obj_getdict`, `MapdictWeakrefSupport.setweakref` →
//! `_mapdict_setweakref`, etc.

use crate::PyError;
use parking_lot::ReentrantMutex;
use pyre_object::PyObjectRef;
use pyre_object::quasiimmut::QuasiImmutField;

use parking_lot::Mutex;
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, LazyLock};

// PyPy serializes mapdict map/storage transitions with the GIL.  Pyre is
// free-threaded, so use narrow address-striped reentrant locks around the same
// transition boundaries.  This is synchronization only; attribute state
// remains on `W_ObjectObject.map/storage`, exactly as upstream.
struct ForkReentrantLock(UnsafeCell<ReentrantMutex<()>>);
unsafe impl Sync for ForkReentrantLock {}

impl ForkReentrantLock {
    fn new() -> Self {
        Self(UnsafeCell::new(ReentrantMutex::new(())))
    }

    fn get(&self) -> &ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(ReentrantMutex::new(())) };
    }
}

static INSTANCE_LOCKS: LazyLock<Vec<ForkReentrantLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkReentrantLock::new()).collect());
static CODE_CACHE_LOCKS: LazyLock<Vec<ForkReentrantLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkReentrantLock::new()).collect());

type MapDictGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

/// A contended stripe must not be waited on from inside the running-mutator
/// census: the owner can allocate under the stripe and request a collection,
/// which then waits for this thread while this thread waits for the stripe.
/// Leave the census around the blocking acquire, as `w_list_lock` does.
fn lock_stripe(lock: &'static ReentrantMutex<()>) -> MapDictGuard {
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

fn instance_lock(obj: PyObjectRef) -> MapDictGuard {
    lock_stripe(INSTANCE_LOCKS[(obj as usize >> 4) & (INSTANCE_LOCKS.len() - 1)].get())
}

fn code_cache_lock(code: PyObjectRef) -> MapDictGuard {
    lock_stripe(CODE_CACHE_LOCKS[(code as usize >> 4) & (CODE_CACHE_LOCKS.len() - 1)].get())
}

pub fn after_fork_child() {
    for lock in INSTANCE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
    for lock in CODE_CACHE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

// ── attribute shapes (mapdict.py:32-42, 720-732) ──────────────────────

/// mapdict.py `NUM_DIGITS = 4`.
pub const NUM_DIGITS: u32 = 4;
/// mapdict.py `NUM_DIGITS_POW2 = 1 << NUM_DIGITS`.
///
/// Note: upstream multiplies by `NUM_DIGITS_POW2` rather than shifting by
/// `NUM_DIGITS` so the result is known non-negative (mapdict.py).
pub const NUM_DIGITS_POW2: usize = 1 << NUM_DIGITS;

/// mapdict.py:40-42 — the maximum number of attributes stored in mapdict
/// (afterwards just use a dict).
pub const LIMIT_MAP_ATTRIBUTES: usize = 80;

/// mapdict.py `ALLOW_UNBOXING_INTS = LONG_BIT == 64`.
///
/// `LONG_BIT` is the width of the interpreter's machine int. That int is `i64`
/// on every pyre target — `sys.maxsize` is `i64::MAX` and the JIT lowers
/// `LONG_BIT` to the literal 64 (`majit_metainterp::ruleopt::proof::LONG_BIT`)
/// — so the predicate holds everywhere. Pointer width is a different quantity
/// and does not decide this: reading it as one left int attributes boxed on
/// the 32-bit wasm guest alone.
pub const ALLOW_UNBOXING_INTS: bool = true;

/// mapdict.py `DICT = 0` — attrkind for instance `__dict__` entries.
pub const DICT: u16 = 0;
/// mapdict.py `SPECIAL = 1` — attrkind for the `"dict"` / `"weakref"`
/// special slots.
pub const SPECIAL: u16 = 1;
/// mapdict.py `INVALID = 2` — sentinel attrkind for empty
/// `MapAttrCache` slots.
pub const INVALID: u16 = 2;
/// mapdict.py `SLOTS_STARTING_FROM = 3` — attrkind for `__slots__`
/// slot `i` is `SLOTS_STARTING_FROM + i`.
pub const SLOTS_STARTING_FROM: u16 = 3;

/// mapdict.py `attrkind_name`.
///
/// ```python
/// def attrkind_name(attrkind):
///     if attrkind == DICT:
///         return "DICT"
///     if attrkind == SPECIAL:
///         return "SPECIAL"
///     if attrkind == INVALID:
///         return "INVALID"
///     return str(attrkind)
/// ```
pub fn attrkind_name(attrkind: u16) -> String {
    match attrkind {
        DICT => "DICT".to_string(),
        SPECIAL => "SPECIAL".to_string(),
        INVALID => "INVALID".to_string(),
        other => other.to_string(),
    }
}

// ── map nodes (mapdict.py AbstractAttribute) ─────────────────────────────────────
//
// AbstractAttribute hierarchy. PyPy uses a class hierarchy
// (AbstractAttribute → Terminator{Dict,NoDict,Devolved} / PlainAttribute,
// mapdict.py/304/420). The Rust port models all map nodes with a single
// `MapNode` enum (the enum-vs-hierarchy adaptation explicitly permitted by
// the parity rules) so the recurring `isinstance(self, PlainAttribute)`
// chain tests (mapdict.py) become a cheap `match`, and the three
// Terminator subclasses become a `TerminatorKind` field (mapdict.py).
//
// Map nodes are interned and shared per type (PyPy interns transitions so the
// same attribute added from the same map yields the same child map); they are
// never freed, so a node is referenced by a raw `*const MapNode` (`MapRef`)
// and the few mutable fields (`ever_mutated`, `allow_unboxing`) use Cell.
//
// AbstractAttribute.space (mapdict.py) is omitted: pyre's object space is
// ambient (global helpers) rather than an object threaded through nodes,
// matching the rest of pyre-interpreter.

/// `2 ** methodcachesizeexp` is the MapAttrCache size (pypyoption.py,
/// default 11).
pub const METHODCACHESIZEEXP: u32 = 11;

/// A shared, interned, immortal map node (mapdict.py AbstractAttribute).
pub type MapRef = *const MapNode;

/// mapdict.py/376/382 — the three Terminator subclasses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminatorKind {
    /// `DictTerminator` (mapdict.py).
    Dict,
    /// `NoDictTerminator` (mapdict.py).
    NoDict,
    /// `DevolvedDictTerminator` (mapdict.py).
    Devolved,
}

/// mapdict.py `Terminator(AbstractAttribute)` — the root of a map chain.
pub struct Terminator {
    /// mapdict.py __init__ `w_cls`.
    pub w_cls: PyObjectRef,
    /// mapdict.py `_immutable_fields_` declares `allow_unboxing?`;
    /// mapdict.py:308 `allow_unboxing` permits unboxed attributes until a
    /// differently-typed reassignment freezes them off. [`Self::set_allow_unboxing`]
    /// invalidates watchers before changing it.
    pub allow_unboxing: Cell<bool>,
    /// The hidden watcher field implementing mapdict.py:305
    /// `_immutable_fields_ = [..., 'allow_unboxing?']`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub allow_unboxing_watchers: QuasiImmutField,
    /// Which Terminator subclass this is.
    pub kind: TerminatorKind,
    /// mapdict.py `DictTerminator.devolved_dict_terminator` (null unless
    /// `kind == Dict`).
    pub devolved_dict_terminator: Cell<MapRef>,
    /// mapdict.py `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`. Starts as null
    /// (`cache_attrs = None`) and is interned to `{}` on first write.
    pub cache_attrs: AtomicPtr<Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>>,
    /// mapdict.py `AbstractAttribute.terminator` — a Terminator points to
    /// itself.
    pub terminator: MapRef,
}

impl Terminator {
    /// Change mapdict.py:308 `allow_unboxing`, invalidating any constant-folded
    /// reads first.
    ///
    /// The value-change return is a required pyre addition: the unguarded
    /// writes in mapdict.py `PlainAttribute._direct_write` and
    /// `CachedAttributeHolder.pick_attr` can repeatedly write `False`, and a
    /// re-armed watcher must not retrace without bound. The `is_installed()`
    /// fast path is quasiimmut.py `invalidation`'s null test and remains
    /// outside the sweep so the common case is lock-free. Notification must
    /// precede the store: rclass.py `hook_setfield` emits
    /// `jit_force_quasi_immutable` before `setfield`, as do celldict.rs
    /// `mutated` and typeobject.rs `w_type_set_version_tag`.
    pub fn set_allow_unboxing(&self, v: bool) {
        if self.allow_unboxing.get() == v {
            return;
        }
        if self.allow_unboxing_watchers.is_installed() {
            unsafe {
                pyre_object::quasiimmut::sweep_quasi_immut_field(&self.allow_unboxing_watchers)
            };
        }
        self.allow_unboxing.set(v);
    }

    pub fn current_allow_unboxing_qmut(
        &self,
    ) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.allow_unboxing_watchers.get_current_qmut_instance()
    }

    pub fn allow_unboxing_qmut_installed(&self) -> bool {
        self.allow_unboxing_watchers.is_installed()
    }

    pub fn force_allow_unboxing_qmut(&self) {
        self.allow_unboxing_watchers.invalidate();
    }
}

/// `quasiimmut.py get_current_qmut_instance` for a Terminator's `allow_unboxing?` field.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_current_allow_unboxing_qmut(
    terminator: *const Terminator,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if terminator.is_null() {
        return None;
    }
    Some(unsafe { (*terminator).current_allow_unboxing_qmut() })
}

/// Whether a Terminator's `allow_unboxing?` watcher is installed.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_allow_unboxing_qmut_installed(terminator: *const Terminator) -> bool {
    !terminator.is_null() && unsafe { (*terminator).allow_unboxing_qmut_installed() }
}

/// Force a Terminator's `allow_unboxing?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_force_allow_unboxing_qmut(terminator: *const Terminator) {
    if terminator.is_null() {
        return;
    }
    unsafe { (*terminator).force_allow_unboxing_qmut() };
}

/// The unbox type of an `UnboxedPlainAttribute` (mapdict.py/547,
/// `space.IntObjectCls` / `space.FloatObjectCls`). pyre exposes no per-type
/// impl-class object, so the int/float distinction is captured by this enum
/// and resolved through `is_int`/`is_float` (the enum-vs-class-object
/// adaptation, parallel to the `MapNode` enum-vs-hierarchy adaptation above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnboxType {
    /// `space.IntObjectCls`.
    Int,
    /// `space.FloatObjectCls`.
    Float,
}

/// mapdict.py `UnboxedPlainAttribute`'s extra fields. PyPy models the
/// unboxed attribute as a `PlainAttribute` subclass; pyre keeps it an optional
/// field on `PlainAttribute` so all the shared `PlainAttribute` machinery
/// (find_map_attr / add_attr / reorder / cache) applies unchanged
/// (enum-vs-hierarchy adaptation). When present, the value is stored unboxed in
/// a shared longlong list at `PlainAttribute.storageindex`; `listindex` is the
/// position in that list.
pub struct UnboxedExtra {
    /// mapdict.py:547 `typ`.
    pub typ: UnboxType,
    /// mapdict.py:563 `listindex` — position in the shared longlong list.
    pub listindex: usize,
    /// mapdict.py:544/561 `firstunwrapped` — this attribute is the first
    /// unboxed one to occupy its storage slot (so it allocates the list).
    pub firstunwrapped: bool,
}

/// mapdict.py `PlainAttribute(AbstractAttribute)` — one stored attribute.
pub struct PlainAttribute {
    /// mapdict.py:425 `name` (utf8-encoded).
    pub name: Wtf8Buf,
    /// mapdict.py:426 `attrkind`.
    pub attrkind: u16,
    /// mapdict.py `storageindex` (= `back.storage_needed()`).
    pub storageindex: usize,
    /// mapdict.py `_num_attributes` (= `back.num_attributes() + 1`).
    pub num_attributes: usize,
    /// mapdict.py:429 `back`.
    pub back: MapRef,
    /// mapdict.py `_immutable_fields_` declares `ever_mutated?`;
    /// mapdict.py:430 `ever_mutated` records whether this attribute has ever
    /// been overwritten or deleted. [`Self::set_ever_mutated`] invalidates
    /// watchers before changing it.
    pub ever_mutated: Cell<bool>,
    /// The hidden watcher field implementing mapdict.py:421
    /// `_immutable_fields_ = [..., 'ever_mutated?', ...]`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub ever_mutated_watchers: QuasiImmutField,
    /// mapdict.py:431 `order`.
    pub order: usize,
    /// mapdict.py `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`. Starts as null
    /// (`cache_attrs = None`) and is interned to `{}` on first write.
    pub cache_attrs: AtomicPtr<Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>>,
    /// mapdict.py `AbstractAttribute.terminator` (= `back.terminator`).
    pub terminator: MapRef,
    /// `Some` for an `UnboxedPlainAttribute` (mapdict.py); `None` for a
    /// plain boxed attribute.
    pub unboxed: Option<UnboxedExtra>,
}

impl PlainAttribute {
    /// Change mapdict.py:430 `ever_mutated` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_ever_mutated(&self, v: bool) {
        if self.ever_mutated.get() == v {
            return;
        }
        if self.ever_mutated_watchers.is_installed() {
            unsafe {
                pyre_object::quasiimmut::sweep_quasi_immut_field(&self.ever_mutated_watchers)
            };
        }
        self.ever_mutated.set(v);
    }

    pub fn current_ever_mutated_qmut(&self) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.ever_mutated_watchers.get_current_qmut_instance()
    }

    pub fn ever_mutated_qmut_installed(&self) -> bool {
        self.ever_mutated_watchers.is_installed()
    }

    pub fn force_ever_mutated_qmut(&self) {
        self.ever_mutated_watchers.invalidate();
    }
}

/// `quasiimmut.py get_current_qmut_instance` for a PlainAttribute's `ever_mutated?` field.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_current_ever_mutated_qmut(
    attribute: *const PlainAttribute,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if attribute.is_null() {
        return None;
    }
    Some(unsafe { (*attribute).current_ever_mutated_qmut() })
}

/// Whether a PlainAttribute's `ever_mutated?` watcher is installed.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_ever_mutated_qmut_installed(
    attribute: *const PlainAttribute,
) -> bool {
    !attribute.is_null() && unsafe { (*attribute).ever_mutated_qmut_installed() }
}

/// Force a PlainAttribute's `ever_mutated?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_force_ever_mutated_qmut(attribute: *const PlainAttribute) {
    if attribute.is_null() {
        return;
    }
    unsafe { (*attribute).force_ever_mutated_qmut() };
}

/// mapdict.py `AbstractAttribute` plus its two concrete subclasses.
pub enum MapNode {
    Terminator(Terminator),
    Plain(PlainAttribute),
}

fn intern_node(node: MapNode) -> MapRef {
    // Immortal raw allocation (`lltype.malloc(..., flavor='raw')`).
    pyre_object::lltype::malloc_raw(node) as MapRef
}

/// mapdict.py `Terminator.__init__`. `AbstractAttribute.__init__(space,
/// self)` makes the terminator its own `terminator`.
pub fn new_terminator(w_cls: PyObjectRef, kind: TerminatorKind) -> MapRef {
    let raw = pyre_object::lltype::malloc_raw(MapNode::Terminator(Terminator {
        w_cls,
        allow_unboxing: Cell::new(true),
        allow_unboxing_watchers: QuasiImmutField::new(),
        kind,
        devolved_dict_terminator: Cell::new(std::ptr::null()),
        cache_attrs: AtomicPtr::new(std::ptr::null_mut()),
        terminator: std::ptr::null(),
    }));
    // Patch the self-referential terminator now that the address is known
    // (still uniquely owned here, before it is shared).
    unsafe {
        if let MapNode::Terminator(t) = &mut *raw {
            t.terminator = raw as MapRef;
        }
    }
    raw as MapRef
}

/// mapdict.py `DictTerminator.__init__` — also builds the paired
/// `DevolvedDictTerminator` and links it.
pub fn new_dict_terminator(w_cls: PyObjectRef) -> MapRef {
    let devolved = new_terminator(w_cls, TerminatorKind::Devolved);
    let t = new_terminator(w_cls, TerminatorKind::Dict);
    unsafe {
        (*t).as_terminator().devolved_dict_terminator.set(devolved);
    }
    t
}

/// Build the per-type map terminator (typeobject.py:251-260 +
/// mapdict.py): a `DictTerminator` when the type has `__dict__`, else
/// a `NoDictTerminator`. `allow_unboxing` keeps its `mapdict.py` default of
/// `True`; type instability later freezes it off per-class through the reactive
/// paths (`plain_direct_write` type change / `holder_pick_attr` mismatch), and
/// the affected instances rebuild boxed storage via `convert_to_boxed`.
///
/// `W_TypeObject.__init__` builds a `DictTerminator` only when
/// `self.hasdict and not typedef.hasdict` — a type whose layout typedef
/// already manages its own dict (e.g. module) gets a `NoDictTerminator`.
/// `typedef_hasdict` is the native-owner projection of
/// `Layout.typedef.hasdict` (`TypeDef.__init__`).  A concrete RPython owner
/// manages that dictionary itself; a PyPy app-level-shaped owner collapsed
/// onto pyre's generic `INSTANCE_TYPE` keeps the same TypeDef metadata but
/// uses mapdict as its physical dictionary.
pub fn new_instance_terminator(w_cls: PyObjectRef, hasdict: bool, typedef_hasdict: bool) -> MapRef {
    // `hasdict and not typedef.hasdict`, expressed without a bare `!` so the
    // annotator can lower it on the JIT-reachable terminator path.
    let wants_dict = match typedef_hasdict {
        true => false,
        false => hasdict,
    };
    if wants_dict {
        new_dict_terminator(w_cls)
    } else {
        new_terminator(w_cls, TerminatorKind::NoDict)
    }
}

/// `_mapdict_init_empty` deferred to first attribute access (mapdict.py
/// `user_setup` calls it at construction with `w_subtype.terminator`; pyre
/// defers to first access). If the instance's `map` is null, fetch the owning
/// type's terminator — lazily creating and storing it on the type if absent,
/// covering types built before the eager install site — then set it as the
/// instance map. Must run before any `node_read`/`node_write`/`node_delete`.
///
/// # Safety
/// `obj` must be a live mapdict carrier. User-layout instances
/// (`W_IntObjectUser`, …) are nursery-born (`w_int_subclass_new`).
/// `type_terminator_or_create` allocates an immortal `Box`
/// (`new_terminator`), so the carrier address stays put across the call.
pub unsafe fn ensure_mapdict_initialized(obj: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    let mut inst = unsafe { mapdict_carrier(obj) };
    if !inst._get_mapdict_map().is_null() {
        return;
    }
    let w_type = pyre_object::w_instance_get_type(obj);
    let term = type_terminator_or_create(w_type);
    inst._set_mapdict_map(term);
}

/// [`type_terminator_or_create`] for callers outside the mapdict layer — the
/// JIT's instantiation emit bakes the terminator into the trace as the fresh
/// instance's `map`, so it needs the same one the interpreter would install
/// rather than the null a not-yet-touched type still carries.
///
/// # Safety
/// `w_type` must be a live `W_TypeObject`.
pub unsafe fn ensure_type_terminator(w_type: PyObjectRef) -> *const u8 {
    unsafe { type_terminator_or_create(w_type) as *const u8 }
}

/// Whether `obj`'s storage class is a `_getusercls` class
/// (`typedef.py _getusercls`): `user_base` names the builtin whose
/// payload the generated layout extends with `MapdictStorageMixin`.
/// A specialised tuple has its own fixed layout and no generated user
/// variant, so its typeptr leaves `user_base` null.
///
/// # Safety
/// `obj` must be a live object reference with a non-null `ob_type`.
#[inline]
unsafe fn is_generated_user_layout_family(obj: PyObjectRef) -> bool {
    unsafe { !(*(*obj).ob_type).user_base.is_null() }
}

/// Whether `obj`'s physical allocation carries the slots supplied by
/// `MapdictStorageMixin` (`mapdict.py` `import_from_mixin` /
/// `typedef._getusercls`). The storage class (`ob_type` / typeptr)
/// records the byte offset of the mixin's `map` word; `mapdict.py`
/// never names zlib / `_lsprof` / `_queue` / `_ssl`. Ordinary
/// `W_ObjectObject` instances share `INSTANCE_TYPE`, which carries that
/// offset, and a `_getusercls` class carries its own.
///
/// # Safety
/// `obj` must be null or a live object reference.
#[inline]
pub unsafe fn has_mapdict_layout(obj: PyObjectRef) -> bool {
    if obj.is_null() || unsafe { pyre_object::is_type(obj) } {
        return false;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    unsafe { pyre_object::pytype_has_mapdict_mixin(obj) }
}

/// Whether attribute access for `obj` routes through mapdict storage. This is
/// the physical [`has_mapdict_layout`] test plus, for a `_getusercls` layout
/// whose builtin payload did not already import `MapdictStorageMixin`, the
/// owning class's `hasdict` flag (`typedef.py` `_getusercls`).
///
/// `W_ObjectObject` already imports the mixin, and
/// `W_ObjectObjectUserDictWeakrefable` keeps that storage for every
/// app-level class, including a slots-only one (`hasdict` false). A
/// generated layout whose `user_base` has no mixin adds the storage on top
/// of a fixed payload and routes attributes through it only when the class
/// has a dict.
///
/// # Safety
/// `obj` must be null or a live object reference.
#[inline]
pub unsafe fn has_mapdict_storage(obj: PyObjectRef) -> bool {
    if !unsafe { has_mapdict_layout(obj) } {
        return false;
    }
    if !unsafe { is_generated_user_layout_family(obj) } {
        return true;
    }
    // `user_base` is non-null here. A base that already publishes the mixin
    // (`W_ObjectObject`) routes every carrier, slotted or not.
    let base = unsafe { (*(*obj).ob_type).user_base };
    if unsafe { (*base).mapdict_offset } != 0 {
        return true;
    }
    let w_class = unsafe { (*obj).w_class };
    !w_class.is_null()
        && unsafe { pyre_object::is_type(w_class) }
        && unsafe { pyre_object::w_type_get_hasdict(w_class) }
}

/// Whether `obj`'s class typedef owns the instance dictionary.
///
/// `typedef.py` `_getusercls` mixes `MapdictDictSupport` in only when
/// `not typedef.hasdict`. `type_terminator_or_create` reads that fact
/// with `w_type_get_typedef_hasdict` on the class from
/// `w_instance_get_type`. Ordinary attributes then go through the class's
/// own `getdict`; the map is not that dictionary.
///
/// # Safety
/// `obj` must be null or a live object. Null, a tagged int, or a
/// `w_class` that is not a type answers false.
#[inline]
pub unsafe fn typedef_owns_dict(obj: PyObjectRef) -> bool {
    if obj.is_null()
        || (pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj))
    {
        return false;
    }
    let w_type = unsafe { pyre_object::w_instance_get_type(obj) };
    !w_type.is_null()
        && unsafe { pyre_object::is_type(w_type) }
        && unsafe { pyre_object::w_type_get_typedef_hasdict(w_type) }
}

/// Fetch `w_type`'s instance terminator, lazily creating and storing it on the
/// type if absent (covering types built before the eager install site).
///
/// # Safety
/// `w_type` must be a live `W_TypeObject`.
unsafe fn type_terminator_or_create(w_type: PyObjectRef) -> MapRef {
    let mut term = pyre_object::w_type_get_terminator(w_type);
    if term.is_null() {
        let hasdict = pyre_object::w_type_get_hasdict(w_type);
        let typedef_hasdict = pyre_object::w_type_get_typedef_hasdict(w_type);
        term = new_instance_terminator(w_type, hasdict, typedef_hasdict) as *const u8;
        unsafe { pyre_object::w_type_set_terminator(w_type, term) };
    }
    term as MapRef
}

/// mapdict.py `MapdictDictSupport.setclass` — re-root `obj`'s map chain
/// onto `w_cls`'s terminator and transplant the rebuilt storage+map. Called from
/// `descr_set___class__` for a `W_ObjectObject`. pyre additionally keeps the
/// `w_class` field authoritative for `type()` (the node layer's
/// `terminator.w_cls` is never read for `getclass`), so the caller sets that
/// after this returns.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]: `node_set_terminator` rebuilds through
/// `copy_attr`/`plain_direct_read`, whose unboxed storage path
/// (`unerase_unboxed` raw `*mut Vec<i64>` array reads, mapdict.py) is
/// not annotator-lowerable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_setclass(obj: PyObjectRef, w_cls: PyObjectRef) {
    unsafe { ensure_mapdict_initialized(obj) };
    let new_term = unsafe { type_terminator_or_create(w_cls) };
    let mut inst = unsafe { mapdict_carrier(obj) };
    let map = inst._get_mapdict_map();
    let new_obj = unsafe { node_set_terminator(map, &inst, new_term) };
    let new_map = new_obj.map;
    inst._set_mapdict_storage_and_map(new_obj.storage, new_map);
}

/// `setdictvalue` routed to the mapdict node layer (mapdict.py
/// `MapdictDictSupport.setdictvalue` → `map.write(self, attrname, DICT,
/// w_value)`, dispatch at mapdict.py).
///
/// Returns `map.write`'s flag: `false` when the map is rooted at a
/// `NoDictTerminator` (`__slots__`, no instance `__dict__`), which is the
/// AttributeError signal `object_setattr` raises on.
///
/// `dont_look_inside` makes this a residual-call boundary for the JIT
/// CodeWriter: `setdictvalue` is JIT-reachable via STORE_ATTR, but the node
/// layer's unboxed storage path (`erase_unboxed`/`unerase_unboxed` raw
/// `*mut Vec<i64>` casts and `Box` allocation for the shared longlong list,
/// mapdict.py) is not annotator-lowerable, so the boundary stays.
/// Look-inside (the `map.write` JIT specialization, mapdict.py) is a
/// future convergence, once the unboxed storage shape is JIT-representable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_setdictvalue(
    obj: PyObjectRef,
    name: &Wtf8,
    value: PyObjectRef,
) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, name, DICT, value)
}

/// Whether `map` is rooted at a `DevolvedDictTerminator` (mapdict.py),
/// i.e. the instance has spilled its attributes into a real dictionary.
///
/// A devolved map no longer names the instance's attributes, and it does not
/// change when one is added, so a JIT fold that re-proves "no instance
/// attribute shadows this name" by pinning the map value is unsound here and
/// must decline instead.
///
/// # Safety
/// `map`, when non-null, must be a live map node.
pub unsafe fn map_is_devolved(map: MapRef) -> bool {
    if map.is_null() {
        return false;
    }
    let terminator = unsafe { (*map).terminator() };
    if terminator.is_null() {
        return false;
    }
    unsafe { (*terminator).as_terminator() }.kind == TerminatorKind::Devolved
}

/// Boxed `DICT` attribute, or a chain miss, when the read cannot run Python.
///
/// `None` is not a miss: there is no mapdict storage, the map is null, the
/// terminator is `DevolvedDictTerminator` (its dict probe is
/// `space.finditem_str` and can run `__eq__`), or the slot is unboxed
/// (`PlainAttribute._direct_read` would box a new object). `Some` null is
/// `_find_map_attr` missing the name. `Some` other is the boxed slot from
/// `_prim_direct_read`, with no migrate-to-boxed write.
///
/// # Safety
/// `obj` must be null or a live object. `name` must outlive the call.
pub unsafe fn mapdict_boxed_dict_attr(obj: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    if obj.is_null() || !unsafe { has_mapdict_storage(obj) } {
        return None;
    }
    // Map and storage move together under the same stripe as
    // `instance_node_getdictvalue_checked`. This probe does not convert
    // or write, so the lock only covers the read.
    let _instance_guard = instance_lock(obj);
    let inst = unsafe { mapdict_carrier(obj) };
    let map = inst._get_mapdict_map();
    if map.is_null() || unsafe { map_is_devolved(map) } {
        return None;
    }
    let Some(attr) = (unsafe { find_map_attr_chain(map, name, DICT) }) else {
        return Some(pyre_object::PY_NULL);
    };
    let plain = unsafe { (*attr).as_plain() };
    if plain.unboxed.is_some() {
        return None;
    }
    Some(unsafe { inst._mapdict_read_storage(plain.storageindex) })
}

/// `getdictvalue` routed to the mapdict node layer (mapdict.py
/// `MapdictDictSupport.getdictvalue` → `map.read(self, attrname, DICT)`).
/// Returns the value or `None` when the attribute is absent.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]: the node read path's unboxed storage branch
/// (`unerase_unboxed` raw `*mut Vec<i64>` reads + `convert_to_boxed`) is not
/// annotator-lowerable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_getdictvalue(obj: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    unsafe { instance_node_getdictvalue_checked(obj, name) }.unwrap_or(None)
}

/// Fallible [`instance_node_getdictvalue`], for the callers that have an error
/// channel to propagate a raising `__eq__` on.
///
/// Only the devolved terminator's dict probe can raise; the swallowing
/// spelling above is written in terms of this one and its `unwrap_or` consumes
/// the pending error slot, so a dropped error cannot surface on a later
/// operation.
///
/// `dont_look_inside` for the same reason as [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_getdictvalue_checked(
    obj: PyObjectRef,
    name: &Wtf8,
) -> Result<Option<PyObjectRef>, PyError> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    // mapdict.py getdictvalue → read → _direct_read (592-598): the read
    // and its lazy migrate-to-boxed tail, off one resolved node.
    let w = unsafe { node_read_converting(map, &mut inst, name, DICT) }?;
    Ok(w)
}

/// `deldictvalue` routed to the mapdict node layer (mapdict.py
/// `MapdictDictSupport.deldictvalue` → `map.delete(self, attrname, DICT)` then
/// `_set_mapdict_storage_and_map`). Returns `true` if the attribute existed and
/// was removed, `false` otherwise (the caller raises AttributeError on false).
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]; the rebuild path (`node_delete` → `node_copy`)
/// is not JIT-traced while the unboxed branches remain unported.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_deldictvalue(obj: PyObjectRef, name: &Wtf8) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    match node_delete(map, &inst, name, DICT) {
        None => false,
        Some(new_obj) => {
            inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
            true
        }
    }
}

/// Read the instance `__dict__` wrapper from the "dict" SPECIAL slot
/// (mapdict.py `w_dict = self._get_mapdict_map().read(self, "dict",
/// SPECIAL)`). Returns `None` when the wrapper has not been materialised.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_getdictvalue`]: the node read path
/// (`node_read` → `plain_direct_read` → `convert_to_boxed`) still has the
/// unported unboxed branch the annotator cannot lower.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_get_dict_slot(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_read(map, &inst, Wtf8::new("dict"), SPECIAL)
}

/// Write the instance `__dict__` wrapper into the "dict" SPECIAL slot
/// (mapdict.py/859 `flag = self._get_mapdict_map().write(self, "dict",
/// SPECIAL, w_dict)`). `node_write` grows the map+storage by the SPECIAL slot on
/// first write (the same transplant path the DICT setter takes). Returns the
/// `write` flag.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_set_dict_slot(obj: PyObjectRef, w_dict: PyObjectRef) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, Wtf8::new("dict"), SPECIAL, w_dict)
}

/// Read the weakref lifeline from the instance's `"weakref"` SPECIAL slot
/// (mapdict.py `self._get_mapdict_map().read(self, "weakref", SPECIAL)`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_get_weakref_slot(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    // A cleared lifeline (mapdict.py:802 writes `None` = null into the retained
    // node) reads back as a null slot; report it as absent rather than
    // `Some(null)`.
    node_read(map, &inst, Wtf8::new("weakref"), SPECIAL).filter(|w| !w.is_null())
}

/// Write the weakref lifeline into the instance's `"weakref"` SPECIAL slot
/// (mapdict.py `self._get_mapdict_map().write(...)`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_set_weakref_slot(obj: PyObjectRef, lifeline: PyObjectRef) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, Wtf8::new("weakref"), SPECIAL, lifeline)
}

/// Clear the weakref lifeline in place, matching mapdict.py:802
/// `self._get_mapdict_map().write(self, "weakref", SPECIAL, None)`.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_del_weakref_slot(obj: PyObjectRef) {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let _ = node_write(
        map,
        &mut inst,
        Wtf8::new("weakref"),
        SPECIAL,
        pyre_object::PY_NULL,
    );
}

// ── methods needed for slots (mapdict.py:764-780 MapdictSlotsSupport) ──

/// mapdict.py `MapdictSlotsSupport.getslotvalue` —
/// `map.read(self, "slot", SLOTS_STARTING_FROM + slotindex)`.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.getslotvalue` default — NotImplementedError
/// (baseobjspace.py) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn getslotvalue(obj: PyObjectRef, slotindex: u32) -> Option<PyObjectRef> {
    assert!(
        unsafe { has_mapdict_layout(obj) },
        "W_Root.getslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    // read → _direct_read (mapdict.py) lazily migrates an unboxed
    // attribute to boxed storage, as in `instance_node_getdictvalue`; both come
    // off the one node `node_read_converting` resolves.
    unsafe { node_read_converting(map, &mut inst, Wtf8::new("slot"), attrkind) }.unwrap_or(None)
}

/// mapdict.py `MapdictSlotsSupport.setslotvalue` —
/// `map.write(self, "slot", SLOTS_STARTING_FROM + slotindex, w_value)`.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.setslotvalue` default — NotImplementedError
/// (baseobjspace.py) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn setslotvalue(obj: PyObjectRef, slotindex: u32, w_value: PyObjectRef) {
    assert!(
        unsafe { has_mapdict_layout(obj) },
        "W_Root.setslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    let flag = node_write(map, &mut inst, Wtf8::new("slot"), attrkind, w_value);
    debug_assert!(flag, "node_write returned false for a slot attribute");
}

/// mapdict.py `MapdictSlotsSupport.delslotvalue` —
/// `map.delete(self, "slot", SLOTS_STARTING_FROM + slotindex)` then
/// `_set_mapdict_storage_and_map`. Returns `false` when the slot was
/// never written (the caller raises AttributeError).
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.delslotvalue` default — NotImplementedError
/// (baseobjspace.py) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn delslotvalue(obj: PyObjectRef, slotindex: u32) -> bool {
    assert!(
        unsafe { has_mapdict_layout(obj) },
        "W_Root.delslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    match node_delete(map, &inst, Wtf8::new("slot"), attrkind) {
        None => false,
        Some(new_obj) => {
            inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
            true
        }
    }
}

/// mapdict.py `PlainAttribute.__init__`.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_plain_attribute(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    order: usize,
) -> MapRef {
    let back_node = unsafe { &*back };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex: back_node.storage_needed(),
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        ever_mutated_watchers: QuasiImmutField::new(),
        order,
        cache_attrs: AtomicPtr::new(std::ptr::null_mut()),
        terminator: back_node.terminator(),
        unboxed: None,
    }))
}

/// mapdict.py `UnboxedPlainAttribute.__init__` +
/// `_compute_storageindex_listindex`.
///
/// Unlike `PlainAttribute.__init__`, the storage index is shared with the
/// nearest `UnboxedPlainAttribute` ancestor (all unboxed attributes pack their
/// longlong values into one shared list); only the first unboxed attribute in a
/// slot (`firstunwrapped`) allocates a fresh slot.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_unboxed_plain_attribute(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    order: usize,
    typ: UnboxType,
) -> MapRef {
    let back_node = unsafe { &*back };
    // _compute_storageindex_listindex (mapdict.py): walk up looking for
    // an existing UnboxedPlainAttribute to share a storage slot with.
    let mut attr = back;
    let mut shared = None;
    loop {
        match unsafe { &*attr } {
            MapNode::Plain(p) => {
                if let Some(u) = &p.unboxed {
                    shared = Some((p.storageindex, u.listindex + 1));
                    break;
                }
                attr = p.back;
            }
            MapNode::Terminator(_) => break,
        }
    }
    let (storageindex, listindex, firstunwrapped) = match shared {
        Some((storageindex, listindex)) => (storageindex, listindex, false),
        None => (back_node.storage_needed(), 0, true),
    };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex,
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        ever_mutated_watchers: QuasiImmutField::new(),
        order,
        cache_attrs: AtomicPtr::new(std::ptr::null_mut()),
        terminator: back_node.terminator(),
        unboxed: Some(UnboxedExtra {
            typ,
            listindex,
            firstunwrapped,
        }),
    }))
}

impl MapNode {
    /// Borrow the inner Terminator (panics on PlainAttribute).
    pub fn as_terminator(&self) -> &Terminator {
        match self {
            MapNode::Terminator(t) => t,
            MapNode::Plain(_) => panic!("as_terminator on PlainAttribute"),
        }
    }

    /// Borrow the inner PlainAttribute (panics on Terminator).
    pub fn as_plain(&self) -> &PlainAttribute {
        match self {
            MapNode::Plain(p) => p,
            MapNode::Terminator(_) => panic!("as_plain on Terminator"),
        }
    }

    /// `isinstance(self, PlainAttribute)`.
    pub fn is_plain(&self) -> bool {
        matches!(self, MapNode::Plain(_))
    }

    /// mapdict.py `AbstractAttribute.terminator` / `get_terminator`.
    pub fn terminator(&self) -> MapRef {
        match self {
            MapNode::Terminator(t) => t.terminator,
            MapNode::Plain(p) => p.terminator,
        }
    }

    /// mapdict.py (Terminator) / 478 (PlainAttribute) / 565-568
    /// (UnboxedPlainAttribute) `storage_needed`.
    pub fn storage_needed(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            // mapdict.py storage_needed: an unboxed attribute only adds a slot when it
            // is the first unboxed one in its slot (`firstunwrapped`);
            // otherwise it packs into the slot the prior unboxed attribute
            // already reserved, so its size is `back.storage_needed()`.
            MapNode::Plain(p) => match &p.unboxed {
                Some(u) => {
                    if u.firstunwrapped {
                        p.storageindex + 1
                    } else {
                        unsafe { (*p.back).storage_needed() }
                    }
                }
                None => p.storageindex + 1,
            },
        }
    }

    /// mapdict.py (Terminator) / 481 (PlainAttribute) `num_attributes`.
    pub fn num_attributes(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            MapNode::Plain(p) => p.num_attributes,
        }
    }

    /// `mapdict.py AbstractAttribute.repr` and concrete overrides.
    ///
    /// The result stays WTF-8 because attribute names may contain lone
    /// surrogates.  PyPy builds the same recursive spelling from the live map
    /// chain for `__pypy__.strategy(instance)`.
    pub unsafe fn repr_wtf8(&self) -> rustpython_wtf8::Wtf8Buf {
        match self {
            MapNode::Terminator(t) => {
                let class_name = match t.kind {
                    TerminatorKind::Dict => "DictTerminator",
                    TerminatorKind::NoDict => "NoDictTerminator",
                    TerminatorKind::Devolved => "DevolvedDictTerminator",
                };
                let mut out = rustpython_wtf8::Wtf8Buf::from_string(format!(
                    "<{class_name} w_cls=<W_TypeObject '"
                ));
                if !t.w_cls.is_null() && unsafe { pyre_object::is_type(t.w_cls) } {
                    out.push_str(unsafe { pyre_object::w_type_get_name(t.w_cls) });
                } else {
                    out.push_str("?");
                }
                out.push_str("'>>");
                out
            }
            MapNode::Plain(p) => {
                let mut out = rustpython_wtf8::Wtf8Buf::new();
                if let Some(unboxed) = &p.unboxed {
                    out.push_str("<UnboxedPlainAttribute ");
                    out.push_wtf8(&p.name);
                    out.push_str(&format!(
                        " {} {} {}{} ",
                        attrkind_name(p.attrkind),
                        p.storageindex,
                        unboxed.listindex,
                        if p.ever_mutated.get() {
                            ""
                        } else {
                            " immutable"
                        },
                    ));
                } else {
                    out.push_str("<PlainAttribute ");
                    out.push_wtf8(&p.name);
                    out.push_str(&format!(
                        " {} {}{} ",
                        attrkind_name(p.attrkind),
                        p.storageindex,
                        if p.ever_mutated.get() {
                            ""
                        } else {
                            " immutable"
                        },
                    ));
                }
                out.push_wtf8(&unsafe { (*p.back).repr_wtf8() });
                out.push_str(">");
                out
            }
        }
    }

    /// mapdict.py `AbstractAttribute.cache_attrs`.
    ///
    /// `None` is the starting `cache_attrs = None`. Allocation of `{}` is
    /// inside [`get_new_attr`] (`_get_new_attr`), not on a general getter.
    pub fn cache_attrs(
        &self,
    ) -> Option<&Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>> {
        match self {
            MapNode::Terminator(t) => load_cache_attrs(&t.cache_attrs),
            MapNode::Plain(p) => load_cache_attrs(&p.cache_attrs),
        }
    }
}

fn load_cache_attrs(
    slot: &AtomicPtr<Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>>,
) -> Option<&Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>> {
    let ptr = slot.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

/// mapdict.py `_get_new_attr`: `if cache is None: cache = self.cache_attrs = {}`.
fn intern_cache_attrs(
    slot: &AtomicPtr<Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>>,
) -> &Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>> {
    if let Some(cache) = load_cache_attrs(slot) {
        return cache;
    }
    let fresh = pyre_object::lltype::malloc_raw(Mutex::new(HashMap::new()));
    match slot.compare_exchange(
        std::ptr::null_mut(),
        fresh,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => unsafe { &*fresh },
        Err(winner) => {
            unsafe { drop(Box::from_raw(fresh)) };
            unsafe { &*winner }
        }
    }
}

/// `interp_magic.py strategy` fallback — return the live map's recursive
/// representation for an object carrying `MapdictStorageMixin`.
///
/// # Safety
/// `obj` must be null or a live Python object.
pub unsafe fn mapdict_strategy_repr(obj: PyObjectRef) -> Option<rustpython_wtf8::Wtf8Buf> {
    if !unsafe { has_mapdict_layout(obj) } {
        return None;
    }
    // `user_setup` installs the type's terminator as part of building the
    // instance, so an object carrying the mixin never holds a null map and
    // `strategy()` always has one to describe.  pyre installs it on first
    // touch instead, which would otherwise make an untouched instance report
    // nothing and raise `TypeError` for the caller.
    unsafe { ensure_mapdict_initialized(obj) };
    let map = unsafe { mapdict_carrier(obj) }._get_mapdict_map();
    if map.is_null() {
        None
    } else {
        Some(unsafe { (*map).repr_wtf8() })
    }
}

/// mapdict.py `AbstractAttribute.search`.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn node_search(node: MapRef, attrtype: u16) -> Option<MapRef> {
    match unsafe { &*node } {
        MapNode::Terminator(_) => None,
        MapNode::Plain(p) => {
            if p.attrkind == attrtype {
                Some(node)
            } else {
                unsafe { node_search(p.back, attrtype) }
            }
        }
    }
}

/// mapdict.py `AbstractAttribute._find_map_attr` — the uncached chain
/// walk.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr_chain(mut node: MapRef, name: &Wtf8, attrkind: u16) -> Option<MapRef> {
    while let MapNode::Plain(p) = unsafe { &*node } {
        if attrkind == p.attrkind && name == &*p.name {
            return Some(node);
        }
        node = p.back;
    }
    None
}

/// mapdict.py `MapAttrCache` — the per-space attribute lookup cache
/// behind `find_map_attr`. A null `MapRef` slot means "empty"/"not found".
pub struct MapAttrCache {
    attrs: Vec<MapRef>,
    names: Vec<Option<Wtf8Buf>>,
    indexes: Vec<u16>,
    cached_attrs: Vec<MapRef>,
}

// MapRef targets are process-lifetime immutable/interned nodes; the only
// mutable cache slots are protected by MAP_ATTR_CACHE's mutex.
unsafe impl Send for MapAttrCache {}

impl MapAttrCache {
    fn new() -> Self {
        let size = 1usize << METHODCACHESIZEEXP;
        MapAttrCache {
            attrs: vec![std::ptr::null(); size],
            names: vec![None; size],
            indexes: vec![INVALID; size],
            cached_attrs: vec![std::ptr::null(); size],
        }
    }

    /// mapdict.py `clear`.
    pub fn clear(&mut self) {
        for slot in self.attrs.iter_mut() {
            *slot = std::ptr::null();
        }
        for slot in self.names.iter_mut() {
            *slot = None;
        }
        for slot in self.indexes.iter_mut() {
            *slot = INVALID;
        }
        for slot in self.cached_attrs.iter_mut() {
            *slot = std::ptr::null();
        }
    }
}

/// `space.fromcache(MapAttrCache)` (mapdict.py) — one cache on the shared
/// object space.  Pyre's free-threaded execution contexts share that space, so
/// the cache remains process/interpreter-owned and is synchronized rather than
/// duplicated through TLS.
// `parking_lot::Mutex` rather than `std::sync::Mutex`: upstream's cache is
// `space.fromcache(MapAttrCache)`, reached with nothing but the GIL, so every
// cycle this lock costs is pyre's own.  The std lock adds a poison flag this
// cache has no use for — the probe below cannot panic while holding it.
static MAP_ATTR_CACHE: LazyLock<parking_lot::Mutex<MapAttrCache>> =
    LazyLock::new(|| parking_lot::Mutex::new(MapAttrCache::new()));

/// interp_gc.py — clear `space.fromcache(MapAttrCache)` before an
/// explicit full collection so cached map nodes do not retain stale entries.
#[majit_macros::dont_look_inside]
pub fn clear_map_attr_cache() {
    MAP_ATTR_CACHE.lock().clear();
}

/// `objectmodel.compute_hash(name)` for a (utf8-encoded) str (mapdict.py).
///
/// This ports the `fnv` mode of `compute_hash` — `objectmodel._hash_string`,
/// the modified-FNV string hash (`rpython/rlib/objectmodel.py`). `compute_hash`
/// is specialized through an overridable ll hash function
/// (`get_ll_hash_function`), and the production default selected by
/// `ChoiceOption("hash", ["fnv", "siphash24"], default="siphash24")`
/// (`pypy/config/pypyoption.py:186`) is siphash24, not this — so this is a
/// PRE-EXISTING-ADAPTATION choosing the deterministic `fnv` mode. It only
/// affects `MapAttrCache` bucket distribution — `find_map_attr` always rechecks
/// name+attrkind, so a divergent hash causes at most a cache miss, never a
/// wrong result.
fn compute_name_hash(name: &Wtf8) -> i64 {
    let s = name.as_bytes();
    let length = s.len();
    if length == 0 {
        return -1;
    }
    let mut x: i64 = (s[0] as i64) << 7;
    for &b in s {
        x = 1000003i64.wrapping_mul(x) ^ (b as i64);
    }
    x ^= length as i64;
    x
}

/// mapdict.py `AbstractAttribute.find_map_attr` with the method cache
/// always enabled — fuses the dispatcher (mapdict.py) and
/// `_find_map_attr_cache` (mapdict.py, `@jit.dont_look_inside`). The
/// uncached walk is `find_map_attr_chain` (`_find_map_attr`); the JIT calls
/// that directly rather than tracing this cache path.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr(self_node: MapRef, name: &Wtf8, attrkind: u16) -> Option<MapRef> {
    const SHIFT2: u32 = u64::BITS - METHODCACHESIZEEXP;
    const SHIFT1: u32 = SHIFT2 - 5;
    // current_object_addr_as_int(self) (mapdict.py:88) — the map node address.
    let attrs_as_int = self_node as usize as i64;
    // unrolled hash computation for the 2-tuple (name, attrkind) (mapdict.py:90-95)
    let c1: i64 = 0x34_5678;
    let c2: i64 = 1_000_003;
    let hash_name = compute_name_hash(name);
    let hash_selector = c2.wrapping_mul(c2.wrapping_mul(c1) ^ hash_name) ^ (attrkind as i64);
    let product = attrs_as_int.wrapping_mul(hash_selector) as u64;
    let attr_hash = ((product ^ (product << SHIFT1)) >> SHIFT2) as usize;

    let mut cache = MAP_ATTR_CACHE.lock();
    // mapdict.py:104 keeps the name by reference; pyre's slot owns its bytes,
    // so a fill that lands on a bucket already naming this attribute keeps the
    // buffer rather than reallocating it.  Which name a bucket holds is fixed
    // by the hash, so this is the ordinary case for a re-filled bucket.
    let name_matches = cache.names[attr_hash]
        .as_deref()
        .is_some_and(|cached| crate::baseobjspace::cache_name_eq(cached, name));
    if name_matches && cache.attrs[attr_hash] == self_node && cache.indexes[attr_hash] == attrkind {
        let cached = cache.cached_attrs[attr_hash];
        return if cached.is_null() { None } else { Some(cached) };
    }
    let attr = unsafe { find_map_attr_chain(self_node, name, attrkind) };
    // Populate the cache, gated on `space._side_effects_ok()`
    // (mapdict.py:110). `_side_effects_ok` returns True except under reverse
    // debugging (not ported); the JIT does not trace this write because the
    // cache path is `@jit.dont_look_inside` and the JIT calls
    // `find_map_attr_chain` directly.
    if crate::baseobjspace::side_effects_ok() {
        cache.attrs[attr_hash] = self_node;
        if !name_matches {
            cache.names[attr_hash] = Some(name.to_owned());
        }
        cache.indexes[attr_hash] = attrkind;
        cache.cached_attrs[attr_hash] = attr.unwrap_or(std::ptr::null());
    }
    attr
}

// ── LOAD_ATTR / STORE_ATTR inline cache (mapdict.py CacheEntry) ─────────
//
// The per-(pycode, nameindex) attribute cache that makes interpreter attribute
// access fast (`pycode._mapdict_caches`). It is distinct from `MapAttrCache`
// above: that one is the space-global `find_map_attr` cache, this one is the
// per-code bytecode-slot cache. Each entry remembers the instance map and the
// owning type's `version_tag` last seen at this name slot, plus the resolved
// attribute node; a read whose object still has that map and whose type still
// has that version_tag re-reads the value straight out of storage, skipping the
// type lookup + map walk.
//
// PyPy holds `map` and `attr` through weakrefs (mapdict.py/1468) because
// its `AbstractAttribute` nodes are GC-managed. pyre interns map nodes as
// immortal leaked `Box`es (`intern_node`/`new_terminator`, see comment at the
// top of this file and lines 190-213), so a raw `MapRef` is the faithful
// equivalent — the weakref could never expire. The map/attr node pointers and
// the u64 version_tag therefore need no GC walking; attribute reads re-read
// the live value through `plain_direct_read` on every hit. The one movable
// reference is the LOAD_METHOD `w_method` slot (mapdict.py), forwarded
// during collection by `pycode::walk_mapdict_method_cache_gc`. (Contingency:
// were map nodes ever made movable, the raw pointers would
// dangle and the whole entry would have to switch to that forwarded design.)

/// mapdict.py `CacheEntry`. PyPy's shared `INVALID_CACHE_ENTRY`
/// sentinel (a `CacheEntry` carrying a fake map, mapdict.py) is
/// represented here by a `None` slot in `pycode._mapdict_caches`, so this struct
/// only ever describes a *valid* (or stale-but-checked) entry. The debug-only
/// `success_counter`/`failure_counter` (mapdict.py, gated on
/// `withmethodcachecounter`) are omitted.
#[derive(Clone, Copy)]
pub struct MapdictCacheEntry {
    /// mapdict.py:1468 `entry.map_wref` target — the instance map this entry was
    /// filled for. Immortal `MapRef`: pyre interns map nodes as leaked `Box`es,
    /// so a raw pointer stands in for PyPy's weakref.
    pub cached_map: MapRef,
    /// mapdict.py:1470/1472 `entry.attr_wref` target — the resolved attribute
    /// node, re-read live via `plain_direct_read` on every hit. `null` (PyPy
    /// `dead_ref`) when the entry caches no attribute.
    pub cached_attr: MapRef,
    /// mapdict.py:1417/1473 `entry.version_tag` (0 = None).
    pub version_tag: u64,
    /// mapdict.py:1421/1475 `entry.valid_for_store`.
    pub valid_for_store: bool,
    /// mapdict.py:1418/1474 `entry.w_method` — filled only by the LOAD_METHOD
    /// cache (callmethod.py); `null` on the LOAD_ATTR / STORE_ATTR paths.
    pub w_method: PyObjectRef,
}

impl MapdictCacheEntry {
    /// mapdict.py `is_valid_for_map`.
    ///
    /// # Safety
    /// `map` and `self.cached_map` (when non-null) must point to live map nodes.
    pub unsafe fn is_valid_for_map(&self, map: MapRef, store: bool) -> bool {
        // mapdict.py:1432 `if store and not self.valid_for_store: return False`.
        if store {
            match self.valid_for_store {
                true => {}
                false => return false,
            }
        }
        unsafe { self._is_valid_for_map(map) }
    }

    /// mapdict.py `_is_valid_for_map` — same instance map (pointer
    /// identity on the immortal node) AND the owning type's current
    /// `version_tag` still equals the cached one.
    ///
    /// # Safety
    /// `map` and `self.cached_map` (when non-null) must point to live map nodes.
    unsafe fn _is_valid_for_map(&self, map: MapRef) -> bool {
        // mapdict.py:1439-1440 `mymap = self.map_wref(); if mymap is not None and
        // mymap is map`.
        if !self.cached_map.is_null() && std::ptr::eq(self.cached_map, map) {
            // mapdict.py `version_tag = map.terminator.w_cls.version_tag()`.
            let w_cls = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
            let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_cls) };
            // mapdict.py `if version_tag is self.version_tag`.
            if version_tag == self.version_tag {
                return true;
            }
        }
        false
    }
}

/// mapdict.py `W_Root._get_mapdict_map` — the instance's current map
/// (`jit.promote(self.map)`), or null for any object that does not use mapdict
/// (the base `W_Root` implementation returns None). `ensure_mapdict_initialized`
/// is a null-check + early return once the map is set, so this stays cheap on
/// the hot LOAD_ATTR path.
///
/// # Safety
/// `w_obj` must be a live object.
unsafe fn mapdict_map_or_null(w_obj: PyObjectRef) -> MapRef {
    // W_TypeObject is represented with the broad INSTANCE_TYPE family for
    // parts of the object-space dispatch, but PyPy's W_TypeObject does not
    // mix in MapdictStorageMixin.  Let type attribute stores reach the
    // metatype data descriptors (__name__, __qualname__, __module__, ...)
    // instead of treating the type object's header as mapdict storage.
    if !unsafe { has_mapdict_storage(w_obj) } {
        return std::ptr::null();
    }
    unsafe { ensure_mapdict_initialized(w_obj) };
    let inst = unsafe { mapdict_carrier(w_obj) };
    inst._get_mapdict_map()
}

/// mapdict.py `PlainAttribute._direct_read` and mapdict.py
/// `UnboxedPlainAttribute._direct_read` — the converting attribute read.
/// `_prim_direct_read` (boxed slot, or box the longlong for an unboxed
/// attribute), then for an unboxed attribute whose class has frozen unboxing
/// (`terminator.allow_unboxing == False`) migrate the instance off unboxed
/// storage so the class stops minting unboxed map variants. (Same migration
/// condition as `attr_migrates_to_boxed`, on the already-resolved node.)
///
/// # Safety
/// `attr` must point to a live `PlainAttribute`; `obj` to its live carrier.
unsafe fn direct_read<O: MapdictObject>(attr: MapRef, obj: &mut O) -> PyObjectRef {
    // mapdict.py/600-601 `_prim_direct_read`.
    let w_res = unsafe { plain_direct_read(attr, &*obj) };
    if !unsafe { attr_migrates_to_boxed(attr) } {
        return w_res;
    }
    // mapdict.py `_convert_to_boxed(obj)`.  It rebuilds the carrier
    // through `node_copy`, which allocates, and the value read above is named
    // by nothing else — publish it and take it back from the slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::pin_roots(&[w_res]);
    unsafe { convert_to_boxed(obj) };
    pyre_object::gc_roots::shadow_stack_get(value_slot)
}

/// mapdict.py — the migration test of `_direct_read`, on an
/// already-resolved node: an unboxed attribute whose class has stopped allowing
/// unboxing migrates the whole instance off unboxed storage.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute`.
unsafe fn attr_migrates_to_boxed(attr: MapRef) -> bool {
    let p = unsafe { (*attr).as_plain() };
    p.unboxed.is_some()
        && !unsafe { (*p.terminator).as_terminator() }
            .allow_unboxing
            .get()
}

/// [`direct_read`]'s migration half alone, for a caller that has already taken
/// the value through `plain_direct_read` off the same node.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute`; `obj` to its live carrier.
unsafe fn migrate_to_boxed_if_frozen<O: MapdictObject>(attr: MapRef, obj: &mut O) {
    if unsafe { attr_migrates_to_boxed(attr) } {
        unsafe { convert_to_boxed(obj) };
    }
}

/// mapdict.py `_fill_cache`. Store the resolved `(map, attr,
/// version_tag)` into slot `nameindex`. PyPy's `INVALID_CACHE_ENTRY` is a `None`
/// slot, so a fill always writes `Some(entry)`.
///
/// `dont_look_inside` — the thread-local `String`-keyed pycode slot store is not
/// annotator-lowerable, and the caller only reaches here under
/// `not we_are_jitted()`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `map`/`attr` live map nodes.
#[majit_macros::dont_look_inside]
unsafe fn fill_cache(
    pycode: PyObjectRef,
    nameindex: usize,
    map: MapRef,
    version_tag: u64,
    attr: MapRef,
    w_method: PyObjectRef,
    valid_for_store: bool,
) {
    // mapdict.py:1462 `if not pycode.space._side_effects_ok(): return`.
    if !crate::baseobjspace::side_effects_ok() {
        return;
    }
    let _code_cache_guard = code_cache_lock(pycode);
    let entry = MapdictCacheEntry {
        cached_map: map,
        cached_attr: attr,
        version_tag,
        valid_for_store,
        w_method,
    };
    unsafe { crate::pycode::w_code_mapdict_caches_set(pycode, nameindex, entry) };
}

/// mapdict.py:1507-1524 (LOAD_ATTR) / mapdict.py:1612-1626 (STORE_ATTR) —
/// classify the looked-up class descriptor into an `(attrkind, is_slot)` pair.
/// `INVALID` means "give up": the caller falls to `space.getattr`/`space.setattr`
/// without filling the cache. `is_slot` selects the `"slot"` attrname over the
/// bytecode `name`.
///
/// LOAD additionally caches a non-data descriptor whose type is immutable (the
/// instance dict wins for reads, mapdict.py); STORE has no such branch
/// — a non-data descriptor does not intercept writes, so the cache gives up and
/// the plain `setattr` re-checks each time.
///
/// # Safety
/// `w_type` and `w_descr` (when `Some`) must be live objects.
unsafe fn classify_attr(
    w_type: PyObjectRef,
    w_descr: Option<PyObjectRef>,
    for_store: bool,
) -> (u16, bool) {
    match w_descr {
        // mapdict.py:1509-1510 — no such attr in the class: the common case,
        // read/write the instance dict.
        None => (DICT, false),
        Some(d) => {
            // mapdict.py:1511-1512 — a MutableCell can change without bumping the
            // version_tag, so give up.  Reaching this arm is why every caller
            // reads the namespace entry through `type_attr_stored` rather than
            // `lookup_in_type_where`: upstream's
            // `_pure_lookup_where_with_method_cache` does not unwrap, and an
            // unwrapped snapshot cached under an unmoved tag is exactly the
            // staleness this rejects.
            if unsafe { pyre_object::celldict::is_mutable_cell(d) } {
                return (INVALID, false);
            }
            // mapdict.py:1513-1519 — a data descriptor shadows the instance dict;
            // only a `__slots__` Member that belongs to this type is cacheable.
            if unsafe { crate::baseobjspace::is_data_descr(d) } {
                if unsafe { pyre_object::is_member(d) }
                    && !unsafe { pyre_object::w_member_is_direct(d) }
                    && unsafe {
                        crate::baseobjspace::issubtype_w(w_type, pyre_object::w_member_get_cls(d))
                    }
                {
                    // mapdict.py `("slot", SLOTS_STARTING_FROM + w_descr.index)`.
                    let kind =
                        SLOTS_STARTING_FROM + unsafe { pyre_object::w_member_get_index(d) } as u16;
                    return (kind, true);
                }
                return (INVALID, false);
            }
            // mapdict.py:1520-1524 — LOAD only: a non-data descriptor whose type
            // is immutable (not a heap type) lets the instance dict win and stay
            // cacheable; a heap-type descriptor could gain `__get__`/`__set__`
            // without the cache noticing, so it is not cacheable.
            if !for_store && !unsafe { descr_type_is_heaptype(d) } {
                return (DICT, false);
            }
            (INVALID, false)
        }
    }
}

/// mapdict.py `space.type(w_descr).is_heaptype()`. Conservatively treats an
/// unresolvable type as a heap type (not cacheable).
///
/// # Safety
/// `w_descr` must be a live object.
unsafe fn descr_type_is_heaptype(w_descr: PyObjectRef) -> bool {
    match crate::typedef::r#type(w_descr) {
        Some(t) => unsafe { pyre_object::typeobject::w_type_is_heaptype(t.as_ptr()) },
        None => true,
    }
}

/// mapdict.py `LOAD_ATTR_caching`. The interpreter LOAD_ATTR fast
/// path (reached only under `not we_are_jitted()`): a monomorphic cache hit
/// re-reads the value straight out of storage; anything else drops to
/// `load_attr_slowpath`.
///
/// `dont_look_inside` — the cache machinery (thread-local pycode slot,
/// `find_map_attr`'s thread-local) is not annotator-lowerable; the JIT reaches
/// LOAD_ATTR through the `we_are_jitted()` getattr branch in the PyFrame
/// executor, never this function.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object.
#[majit_macros::dont_look_inside]
pub unsafe fn load_attr_caching(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
) -> Result<PyObjectRef, PyError> {
    // mapdict.py `map = w_obj._get_mapdict_map()` answers None for an object
    // without mapdict storage, so `entry.is_valid_for_map(map)` fails and
    // `LOAD_ATTR_slowpath` skips its map arm.  The layout test reads only
    // immutable header words, so that answer needs neither lock.
    if !unsafe { has_mapdict_layout(w_obj) } {
        return unsafe { load_attr_slowpath(pycode, w_obj, nameindex, name, std::ptr::null()) };
    }
    // Copy the immortal-node cache entry under its code-owned lock, then
    // release that lock before touching an instance. Holding both locks
    // across the slow path would extend them across arbitrary descriptor
    // calls, unlike PyPy's single GIL critical section, and permits
    // instance-A -> code -> instance-B / instance-B -> code cycles.
    let entry = {
        let _code_cache_guard = code_cache_lock(pycode);
        unsafe { crate::pycode::w_code_mapdict_caches_get(pycode, nameindex) }
    };
    let map = {
        let _instance_guard = instance_lock(w_obj);
        // mapdict.py `map = w_obj._get_mapdict_map()`.
        let map = unsafe { mapdict_map_or_null(w_obj) };
        if let Some(e) = entry {
            // mapdict.py:1483-1487 — valid cache entry with a live attr.
            if !map.is_null()
                && unsafe { e.is_valid_for_map(map, false) }
                && e.w_method.is_null()
                && !e.cached_attr.is_null()
            {
                let mut inst = unsafe { mapdict_carrier(w_obj) };
                // mapdict.py `return attr._direct_read(w_obj)`.
                return Ok(unsafe { direct_read(e.cached_attr, &mut inst) });
            }
        }
        map
    };
    unsafe { load_attr_slowpath(pycode, w_obj, nameindex, name, map) }
}

/// mapdict.py `LOAD_ATTR_slowpath`.
///
/// `dont_look_inside` — reaches the type-lookup method cache and `find_map_attr`
/// thread-locals; only called from `load_attr_caching`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object; `map` its map
/// (or null).
#[majit_macros::dont_look_inside]
unsafe fn load_attr_slowpath(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    map: MapRef,
) -> Result<PyObjectRef, PyError> {
    // mapdict.py:1495 `if map is not None:`.
    if !map.is_null() {
        // mapdict.py:1496 `w_type = map.terminator.w_cls`.
        let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
        // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access.
        // pyre has no separate `_handle_getattribute`; `space.getattr`
        // re-dispatches the custom `__getattribute__`, the same result.
        if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
            return crate::baseobjspace::getattr_str(w_obj, name);
        }
        // mapdict.py:1500 `version_tag = w_type.version_tag()`.
        let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
        // mapdict.py:1501 `if version_tag is not None:` (0 = None).
        if version_tag != 0 {
            // mapdict.py:1504-1505 `_, w_descr = _pure_lookup_where_with_method_cache`
            // -- the raw entry, not `lookup`: the comment above that line names
            // the reason ("obscure cases in which the w_descr is a MutableCell,
            // which may change without changing the version_tag"), and
            // `classify_attr` gives up on one.
            let w_descr = unsafe { crate::baseobjspace::type_attr_stored(w_type, Wtf8::new(name)) };
            // mapdict.py:1507-1524 classify.
            let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
            // mapdict.py `if attrkind != INVALID:`.
            if attrkind != INVALID {
                let attrname = if is_slot { "slot" } else { name };
                // The type/descriptor lookup above can execute Python. Re-lock
                // only for the direct map/storage observation and abandon the
                // cache path if another thread changed the instance map.
                let direct = {
                    let _instance_guard = instance_lock(w_obj);
                    let current_map = unsafe { mapdict_map_or_null(w_obj) };
                    if std::ptr::eq(current_map, map) {
                        unsafe { find_map_attr(current_map, Wtf8::new(attrname), attrkind) }.map(
                            |attr| {
                                let mut inst = unsafe { mapdict_carrier(w_obj) };
                                let value = unsafe { direct_read(attr, &mut inst) };
                                (current_map, attr, value)
                            },
                        )
                    } else {
                        None
                    }
                };
                if let Some((current_map, attr, value)) = direct {
                    // mapdict.py `_fill_cache(...,
                    // valid_for_store=w_type.setattr_if_not_from_object() is None)`.
                    let valid_for_store =
                        unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }
                            .is_none();
                    unsafe {
                        fill_cache(
                            pycode,
                            nameindex,
                            current_map,
                            version_tag,
                            attr,
                            std::ptr::null_mut(),
                            valid_for_store,
                        );
                    }
                    // mapdict.py `return attr._direct_read(w_obj)`.
                    return Ok(value);
                }
            }
        }
    }
    // mapdict.py `return space.getattr(w_obj, w_name)`.
    crate::baseobjspace::getattr_str(w_obj, name)
}

/// The JIT LOAD_ATTR fast-path resolver: the `load_attr_slowpath`
/// (mapdict.py:1492-1537) resolution steps, but instead of reading the value it
/// returns the ingredients the meta-tracer needs to fold the read to a guarded
/// inline `storage[storageindex]` — `(w_type, version_tag, map, storageindex)`.
///
/// Returns `None` (leave the access on the residual `space.getattr`) for every
/// shape the inline read cannot cover: non-instance receiver, missing map,
/// custom `__getattribute__`, uncacheable `version_tag`, a data-descriptor /
/// `INVALID` classification, an attribute not present on this instance's map,
/// or an unboxed slot (whose read boxes a longlong — not a plain slot fetch).
/// This is the mapdict analog of `baseobjspace::load_method_fast_path`, sharing
/// the same gates so the symbolic trace and the concrete frame agree.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn load_attr_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, MapRef)> {
    // mapdict.py:1495 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1496 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access;
    // not soundly foldable to a storage read.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1504-1524 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    // mapdict.py `if attrkind != INVALID:`.
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    // The inline read is a plain `storage[storageindex]` fetch; an unboxed slot
    // (mapdict.py) instead boxes a longlong out of a shared list, so
    // leave it on the residual.
    if p.unboxed.is_some() {
        return None;
    }
    Some((w_type, version_tag, map, p.storageindex, attr))
}

/// mapdict.py `AbstractAttribute.read`:
///
/// ```python
/// if (jit.isconstant(attr) and jit.isconstant(obj) and not attr.ever_mutated):
///     return attr._pure_direct_read(obj)
/// else:
///     return attr._direct_read(obj)
/// ```
///
/// The `jit.isconstant(obj)` half is the caller's to answer — it holds the
/// receiver operand.  `attr` is constant on every path that resolved it against
/// a pinned map.  This answers the remaining half and hands back the
/// `PlainAttribute` whose `ever_mutated?` quasi-immutable licenses the elidable
/// read, so the caller can record the field before baking the value in.
/// `None` means the attribute has been written or deleted since it was added
/// and only `_direct_read` is sound.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn pure_direct_read_attr(attr: MapRef) -> Option<*const PlainAttribute> {
    let p = unsafe { (*attr).as_plain() };
    (!p.ever_mutated.get()).then_some(p as *const PlainAttribute)
}

/// A class attribute that `object.__getattribute__` returns unchanged:
/// present on the type, not a descriptor, and not shadowed by this
/// instance's map.  The JIT LOAD_ATTR fold bakes the value under the same
/// `w_class` / `version_tag` / `map` pins [`load_attr_fast_path`] uses for
/// an instance slot.
///
/// `type_attr_value_fast_path` is the type-receiver twin
/// (`W_TypeObject.descr_getattribute`); this is the instance-receiver arm
/// of the same "no descriptor protocol" answer, the `space.get(w_descr,
/// w_obj)` tail of `Object.descr__getattribute__`.  A `__get__` or a
/// heap-type value declines — those bind or could gain a `__get__` without bumping the
/// type version.  A devolved map declines because `find_map_attr` cannot
/// prove the instance does not shadow the name.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn class_attr_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, PyObjectRef)> {
    if unsafe { typedef_owns_dict(w_obj) } {
        return None;
    }
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    if unsafe { map_is_devolved(map) } {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // The LOAD_ATTR fold bakes `w_descr` as a constant under this version
    // pin.  Decline when the namespace holds a `MutableCell`: the pin does
    // not move on an in-place cell write, so the baked value would be stale.
    if unsafe {
        crate::baseobjspace::type_attr_stored_is_cell(w_type, rustpython_wtf8::Wtf8::new(name))
    } {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) }?;
    if unsafe { crate::baseobjspace::is_data_descr(w_descr) } {
        return None;
    }
    let value_type = crate::typedef::r#type(w_descr)?.as_ptr();
    if unsafe { crate::baseobjspace::lookup_in_type(value_type, "__get__") }.is_some()
        || unsafe { pyre_object::w_type_is_heaptype(value_type) }
    {
        return None;
    }
    if unsafe { find_map_attr(map, Wtf8::new(name), DICT) }.is_some() {
        return None;
    }
    Some((w_type, version_tag, map, w_descr))
}

/// The `__class__` twin of [`class_attr_fast_path`].  `object.__class__` is a
/// data descriptor, so the class-attribute path declines it, but its getter
/// body is `space.type(w_obj)` — the class the returned shape already names.
///
/// PyPy has no entry of this kind: its tracer inlines
/// `Object.descr__getattribute__` → `GetSetProperty.descr_property_get` →
/// `objectobject.py descr_get_class`, and `space.type(w_obj)` const-folds
/// against the `guard_class` already in the trace.  Pyre's `LOAD_ATTR` is one
/// residual, so the same answer is spelled as a fold instead.
///
/// The predicate is the same pair `isinstance_miss_class_lookup_is_pure`
/// names: the receiver inherits `object.__getattribute__`, and its MRO
/// resolves `__class__` to `object`'s descriptor.  The trace computes that
/// lookup (`has_object_getattribute`) because the memo flag stays cold until
/// some other lookup sets it.  A `__class__` property declines.  The name is
/// a data descriptor, so no instance dict can shadow it and the map needs no
/// `find_map_attr` miss test.
///
/// Returns the shape ingredients the caller guards; the value read is
/// `w_type` itself.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn class_descr_fast_path(w_obj: PyObjectRef) -> Option<(PyObjectRef, u64, MapRef)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    if unsafe { map_is_devolved(map) } {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // Same pair as `isinstance_miss_class_lookup_is_pure`: default
    // `object.__getattribute__`, and `__class__` still `object`'s descriptor.
    // The memo flag that predicate reads stays cold until some other lookup
    // sets it, so the trace computes the lookup (`has_object_getattribute`).
    if !unsafe { crate::baseobjspace::has_object_getattribute(w_type) } {
        return None;
    }
    let Some(actual_class_descr) =
        (unsafe { crate::baseobjspace::lookup_in_type_where(w_type, "__class__") })
    else {
        return None;
    };
    let Some(object_class_descr) = (unsafe {
        crate::baseobjspace::lookup_in_type_where(crate::typedef::w_object(), "__class__")
    }) else {
        return None;
    };
    if !std::ptr::eq(actual_class_descr, object_class_descr) {
        return None;
    }
    Some((w_type, version_tag, map))
}

/// The miss twin of [`load_attr_fast_path`]: return the ingredients for
/// inlining the receiver type's `__getattr__` hook when `name` resolves
/// nowhere.
///
/// `baseobjspace::instance_getattr_hook_or_err` is the tail this stands in for
/// (`descroperation.py:242-245`): once the descriptor protocol has produced an
/// AttributeError, the type's `__getattr__` is looked up and called with the
/// receiver and the name. Reaching that tail is what the two returned pins
/// prove, and both are guards the caller owes:
///
///   * `version_tag` — the class lookup stays constant, so `name` keeps
///     resolving to nothing on the type and `__getattr__` keeps resolving to
///     the returned hook;
///   * `map` — the instance shape stays constant, so `name` keeps being absent
///     from this receiver's own storage.
///
/// Together they make the whole `object_getattr_miss` walk a compile-time
/// answer, which is the work the fold removes; the hook itself is what the
/// caller then inlines.
///
/// Returns `None` for every shape those two guards cannot cover: a non-mapdict
/// receiver, a custom `__getattribute__`, an uncacheable `version_tag`, a name
/// the type or the instance actually owns, or a type with no `__getattr__`.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn getattr_hook_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, PyObjectRef, PyObjectRef)> {
    let (w_type, version_tag, map) = unsafe { getattr_resolves_nowhere(w_obj, name) }?;
    let w_getattr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, "__getattr__") }?;
    // An in-place `ObjectMutableCell` write does not move `version_tag`.  The
    // cell is what `write_cell` updates; the caller getfields its payload.
    let cell = unsafe {
        crate::baseobjspace::type_attr_object_cell(
            w_type,
            rustpython_wtf8::Wtf8::new("__getattr__"),
        )
    };
    Some((w_type, version_tag, map, w_getattr, cell))
}

/// Return the guarded lookup ingredients for the receiver type's custom
/// `__getattribute__` slot.
///
/// `descroperation.py getattr` resolves this descriptor before it examines the
/// requested name.  PyPy traces through that lookup and
/// `get_and_call_function(w_descr, w_obj, w_name)`: the promoted receiver map
/// pins the mapdict carrier shape, while the type's `_version_tag?` pins which
/// descriptor the MRO lookup returned.  The full-body walker uses these same
/// ingredients to enter a plain Python hook instead of leaving the whole
/// access behind an opaque `load_attr_fn` residual.
///
/// A default `object.__getattribute__`, a devolved map, an uncacheable type, or
/// a type with `__getattr__` declines.  The last gate preserves
/// `_handle_getattribute`'s AttributeError fallback until the inline entry can
/// carry both Python frames as one operation.  The requested name is
/// deliberately absent from this oracle: a custom hook owns every name and
/// receives the name as its second argument.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn getattribute_hook_fast_path(
    w_obj: PyObjectRef,
) -> Option<(PyObjectRef, u64, MapRef, PyObjectRef, PyObjectRef)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() || unsafe { map_is_devolved(map) } {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    let w_getattribute = unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }?;
    if unsafe { crate::baseobjspace::lookup_in_type_where(w_type, "__getattr__") }.is_some() {
        return None;
    }
    // An in-place `ObjectMutableCell` write does not move `version_tag`.  The
    // cell is what `write_cell` updates; the caller getfields its payload.
    let cell = unsafe {
        crate::baseobjspace::type_attr_object_cell(
            w_type,
            rustpython_wtf8::Wtf8::new("__getattribute__"),
        )
    };
    Some((w_type, version_tag, map, w_getattribute, cell))
}

/// Type-receiver twin of [`getattribute_hook_fast_path`].
///
/// `W_TypeObject` has no mapdict storage, so the instance oracle returns
/// `None` before it ever sees the metaclass. `getattr_str` still selects
/// `metaclass_python_getattribute` ahead of `descr_getattribute`. The pins
/// are that metaclass — the receiver's `w_class` — and its version tag.
/// There is no instance map. A metaclass `__getattr__` declines for the
/// same AttributeError-fallback reason as the instance oracle.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn type_getattribute_hook_fast_path(
    w_obj: PyObjectRef,
) -> Option<(PyObjectRef, u64, PyObjectRef, PyObjectRef)> {
    let (metatype, w_getattribute) =
        unsafe { crate::baseobjspace::metaclass_python_getattribute(w_obj) }?;
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(metatype) };
    if version_tag == 0 {
        return None;
    }
    // `lookup_in_type_where` reaches `box_str_constant`. Both pins stay
    // live across that lookup and the cell read.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[metatype, w_getattribute]);
    if unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(base), "__getattr__") }
        .is_some()
    {
        return None;
    }
    let cell = unsafe {
        crate::baseobjspace::type_attr_object_cell(
            roots.get(base),
            rustpython_wtf8::Wtf8::new("__getattribute__"),
        )
    };
    Some((roots.get(base), version_tag, roots.get(base + 1), cell))
}

/// The `__getattr__`-less twin of [`getattr_hook_fast_path`]: `name` resolves
/// nowhere *and* the type has no hook to run afterwards, so the access ends in
/// the `AttributeError` `object_getattr_miss` raises.
///
/// That makes the whole miss a compile-time answer for the callers that only
/// need to know the access fails — `hasattr(obj, name)` is then a constant
/// False and `getattr(obj, name, default)` is `default`. The two pins the
/// caller owes are the same ones [`getattr_hook_fast_path`] documents.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn getattr_absent_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef)> {
    let pins = unsafe { getattr_resolves_nowhere(w_obj, name) }?;
    // A hook would run application code whose answer neither pin describes,
    // and which can return a value or raise something other than
    // AttributeError.
    if unsafe { crate::baseobjspace::lookup_in_type_where(pins.0, "__getattr__") }.is_some() {
        return None;
    }
    Some(pins)
}

/// The shared miss proof behind [`getattr_hook_fast_path`] and
/// [`getattr_absent_fast_path`]: `name` is on neither the type nor the
/// receiver's own storage, and the returned `version_tag` / `map` pins are what
/// keep that true.
///
/// # Safety
/// `w_obj` must be a live object.
unsafe fn getattr_resolves_nowhere(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef)> {
    // A typedef-owned `__dict__` is invisible to the map (`typedef.py`
    // `_getusercls` without `MapdictDictSupport`).
    if unsafe { typedef_owns_dict(w_obj) } {
        return None;
    }
    // mapdict.py:1495 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1496 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` runs its own lookup,
    // which neither pin describes.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // The miss itself. A type-level hit is refused before the map is consulted:
    // `classify_attr` reads a `__slots__` member under the `"slot"` name rather
    // than its own, so a descriptor found here says nothing about what
    // `find_map_attr(name)` below would answer.
    if unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) }.is_some() {
        return None;
    }
    // A devolved instance keeps its attributes in a real dictionary, and
    // mapdict.py states that `find_map_attr` "will always return
    // None if attrkind==DICT" for such a map.  The hit path reads that call
    // for a Some, so a None costs it only the fold; this path reads it for its
    // ABSENCE and would take an always-None answer as proof the name is not on
    // the instance.  `LOAD_METHOD_mapdict_fill_cache_method` — upstream's own
    // case of pinning a map to cache a negative instance lookup — refuses the
    // shape outright (mapdict.py `if map is None or
    // isinstance(map.terminator, DevolvedDictTerminator): return`), and the
    // map pin cannot stand in: the devolved terminator is a per-class
    // singleton, so the guarded map word is the same for every devolved
    // instance and unchanged by a later `obj.<name> = ...`.
    if unsafe { map_is_devolved(map) } {
        return None;
    }
    // `classify_attr(w_type, None, false)` answers `(DICT, false)` — the
    // no-descriptor arm (mapdict.py) — so this is the same
    // `find_map_attr` call the hit path makes, read for its absence.
    if unsafe { find_map_attr(map, Wtf8::new(name), DICT) }.is_some() {
        return None;
    }
    Some((w_type, version_tag, map))
}

/// The [`load_attr_fast_path`] twin for a receiver that keeps its attributes in
/// a `newdict(instance=True)` dictionary rather than in header mapdict storage
/// (`mapdict.py make_instance_dict`). It applies the same
/// `load_attr_slowpath` gates (mapdict.py) and differs only in where
/// the map comes from — the fake carrier the dictionary erases into rather than
/// the receiver's own header. Returns the type and its version tag alongside
/// the fold ingredients: the carrier, its map, the `storageindex`, and the
/// `(typ, listindex)` pair when the attribute took an unboxed transition.
///
/// The carrier's map holds only dictionary attributes, so a `__slots__` member
/// — which mapdict.py would resolve through the receiver's own map — is
/// declined here rather than looked up under the `"slot"` name.
///
/// Declines a dictionary that is not `MapDictStrategy`-backed. The caller must
/// still prove that at trace time before dereferencing `dstorage` as a carrier:
/// a devolved dictionary erases a smaller storage box, and reading a carrier
/// field off it is out of bounds.
///
/// # Safety
/// `w_obj` must be live, and `w_dict` must point at its live `W_DictObject`.
pub unsafe fn instance_dict_attr_fast_path(
    w_obj: PyObjectRef,
    w_dict: PyObjectRef,
    name: &str,
) -> Option<(
    PyObjectRef,
    u64,
    PyObjectRef,
    MapRef,
    usize,
    Option<(UnboxType, usize)>,
)> {
    // mapdict.py:1496 `w_type = map.terminator.w_cls` — the carrier's own
    // terminator is the shared fake one, so the type comes off the receiver.
    let w_type = unsafe { (*w_obj).w_class };
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1504-1526 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    if attrkind != DICT || is_slot {
        return None;
    }
    let dict = unsafe { &*(w_dict as *const pyre_object::dictmultiobject::W_DictObject) };
    if dict.dstrategy.strategy_kind() != pyre_object::dictmultiobject::StrategyKind::Map {
        return None;
    }
    let carrier = dict.dstorage as PyObjectRef;
    if carrier.is_null() {
        return None;
    }
    let map = unsafe { mapdict_carrier(carrier) }._get_mapdict_map();
    if map.is_null() || unsafe { map_is_devolved(map) } {
        return None;
    }
    // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(name), DICT) }?;
    let plain = unsafe { (*attr).as_plain() };
    // `_direct_read` migrates the instance off unboxed storage once its class
    // has frozen unboxing (mapdict.py); the fold performs
    // `_prim_direct_read` alone, so a frozen class stays on the residual that
    // still runs the migration.
    if plain.unboxed.is_some()
        && !unsafe { (*plain.terminator).as_terminator() }
            .allow_unboxing
            .get()
    {
        return None;
    }
    Some((
        w_type,
        version_tag,
        carrier,
        map,
        plain.storageindex,
        plain.unboxed.as_ref().map(|u| (u.typ, u.listindex)),
    ))
}

/// Shared resolution for [`property_get_fast_path`] / [`property_set_fast_path`]:
/// when `name` on `w_obj`'s type resolves to a `property` data descriptor,
/// return the type, its cacheable version tag, and the property object.  Applies
/// the shape gates common to both directions — a mapdict instance and a non-zero
/// `version_tag`.  The read/write hook gate differs (`__getattribute__` for the
/// getter, `__setattr__` for the setter), so each caller applies its own.  A
/// property is a data descriptor, so it takes precedence over the instance dict
/// and the version-pinned type lookup is authoritative; the instance map is not
/// read.
///
/// # Safety
/// `w_obj` must be a live object.
unsafe fn property_descr_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    unsafe { property_descr_fast_path_wtf8(w_obj, Wtf8::new(name)) }
}

unsafe fn property_descr_fast_path_wtf8(
    w_obj: PyObjectRef,
    name: &Wtf8,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, name) } {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where_wtf8(w_type, name) }?;
    // Exact type: the fold calls `fget`/`fset` directly, which stands in for
    // `type(w_descr).__get__` only where that cannot have been overridden
    // (`descroperation.py get_and_call_function`). A property subclass is
    // `PROPERTY_USER_TYPE` (`typedef.py` `_getusercls`) and fails this test.
    if !unsafe { pyre_object::descriptor::is_exact_property(w_descr) } {
        return None;
    }
    Some((w_type, version_tag, w_descr))
}

/// LOAD_ATTR `property` fast path: return the type, version tag, the property
/// object, and its Python `fget` when `obj.name` reads a property getter, so
/// the full-body walker can inline `fget(obj)` in place of the opaque `getattr`
/// residual.  Returns `None` (leave the residual) for a write-only property or
/// any shape [`property_descr_fast_path`] declines.  A custom
/// `__getattribute__` owns the read (mapdict.py), so it declines to
/// the residual.
///
/// The property object is part of the answer because `fget` alone cannot be
/// baked: `descriptor.py` declares the slot `w_fget?`, so the fold owes it
/// a `QUASIIMMUT_FIELD` marker naming the owner.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn property_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef, PyObjectRef)> {
    unsafe { property_get_fast_path_wtf8(w_obj, Wtf8::new(name)) }
}

/// WTF-8-preserving twin of [`property_get_fast_path`].
///
/// # Safety
/// Same contract as [`property_get_fast_path`].
pub unsafe fn property_get_fast_path_wtf8(
    w_obj: PyObjectRef,
    name: &Wtf8,
) -> Option<(PyObjectRef, u64, PyObjectRef, PyObjectRef)> {
    let (w_type, version_tag, w_descr) = unsafe { property_descr_fast_path_wtf8(w_obj, name) }?;
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(w_descr) };
    if fget.is_null() || unsafe { pyre_object::pyobject::is_none(fget) } {
        return None;
    }
    Some((w_type, version_tag, w_descr, fget))
}

/// `typeobject.py W_TypeObject.descr_getattribute` data-descriptor arm when
/// the metaclass entry is an exact `property`.
///
/// [`property_get_fast_path_wtf8`] asks the instance map and misses every
/// class object. `space.lookup(cls, name)` still finds the property on the
/// metaclass, and `get_and_call_function` calls `fget(cls)`. Calling `fget`
/// stands in for `type(descr).__get__` only for an exact `property`
/// (`is_exact_property`); a property subclass stays on the residual. A
/// metaclass that replaces `__getattribute__` is
/// [`type_getattribute_hook_fast_path`]'s, not this arm. A metaclass
/// `__getattr__` declines: an `AttributeError` from `fget` has to reach
/// that hook, which this direct inline does not encode.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn type_property_get_fast_path(
    w_obj: PyObjectRef,
    name: &Wtf8,
) -> Option<(PyObjectRef, u64, PyObjectRef, PyObjectRef)> {
    if w_obj.is_null() || !pyre_object::typeobject::is_type(w_obj) {
        return None;
    }
    // `metaclass_keeps_type_getattribute` and the lookups below reach
    // `box_str_constant`. Each result is pinned before the next call.
    let roots = pyre_object::gc_roots::push_roots();
    let obj_at = roots.pin_roots(&[w_obj]);
    if !unsafe { crate::baseobjspace::metaclass_keeps_type_getattribute(roots.get(obj_at)) } {
        return None;
    }
    let metatype = crate::typedef::r#type(roots.get(obj_at))?.as_ptr();
    let meta_at = roots.pin_roots(&[metatype]);
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(roots.get(meta_at)) };
    if version_tag == 0 {
        return None;
    }
    if unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(meta_at), "__getattr__") }
        .is_some()
    {
        return None;
    }
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(roots.get(meta_at), name) } {
        return None;
    }
    let w_descr =
        unsafe { crate::baseobjspace::lookup_in_type_where_wtf8(roots.get(meta_at), name) }?;
    let descr_at = roots.pin_roots(&[roots.get(meta_at), w_descr]);
    if !unsafe { crate::baseobjspace::is_data_descr(roots.get(descr_at + 1)) }
        || !unsafe { pyre_object::descriptor::is_exact_property(roots.get(descr_at + 1)) }
    {
        return None;
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(roots.get(descr_at + 1)) };
    if fget.is_null() || unsafe { pyre_object::pyobject::is_none(fget) } {
        return None;
    }
    Some((
        roots.get(descr_at),
        version_tag,
        roots.get(descr_at + 1),
        fget,
    ))
}

/// LOAD_ATTR user-data-descriptor fast path: resolve the exact Python
/// `__get__` call made by `Object.descr__getattribute__`, without executing it.
///
/// PyPy first promotes the receiver type lookup, proves that the class value
/// is a data descriptor, then looks up `__get__` on the descriptor's own live
/// type and calls it as `(w_descr, w_obj, w_type)`.  Return every object whose
/// shape that decision depends on so the full-body walker can install the
/// corresponding receiver-type, descriptor-type and descriptor-map guards
/// before entering the Python frame.
///
/// A receiver with `__getattr__` declines: an `AttributeError` raised anywhere
/// in the inlined getter would otherwise have to re-enter the outer
/// `_handle_getattribute` fallback, which this direct LOAD_ATTR sub-walk does
/// not yet encode.  Builtin descriptors and descriptor objects without
/// mapdict storage likewise stay on the complete residual protocol.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn data_descriptor_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(
    PyObjectRef,
    u64,
    PyObjectRef,
    PyObjectRef,
    u64,
    MapRef,
    PyObjectRef,
)> {
    let receiver_map = unsafe { mapdict_map_or_null(w_obj) };
    if receiver_map.is_null() {
        return None;
    }
    let w_type = unsafe { (*(*receiver_map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null()
        || unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some()
        || unsafe { crate::baseobjspace::lookup_in_type_where(w_type, "__getattr__") }.is_some()
    {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    if unsafe {
        crate::baseobjspace::type_attr_stored_is_cell(w_type, rustpython_wtf8::Wtf8::new(name))
    } {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) }?;
    if !unsafe { crate::baseobjspace::is_data_descr(w_descr) } {
        return None;
    }
    // Exact `property` has its own fold. A `_getusercls` property
    // subclass is still that builtin descriptor; this path is for a
    // user type's `__get__`, not `W_Property`.
    if unsafe { pyre_object::descriptor::is_property(w_descr) } {
        return None;
    }
    let descr_type = crate::typedef::r#type(w_descr)?.as_ptr();
    let descr_version_tag = unsafe { crate::baseobjspace::w_type_version_tag(descr_type) };
    if descr_version_tag == 0 {
        return None;
    }
    let descr_map = unsafe { mapdict_map_or_null(w_descr) };
    if descr_map.is_null() {
        return None;
    }
    // Baked by identity under `descr_version_tag`, which an in-place cell
    // write does not move.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(descr_type, Wtf8::new("__get__")) } {
        return None;
    }
    let w_get = unsafe { crate::baseobjspace::lookup_in_type_where(descr_type, "__get__") }?;
    Some((
        w_type,
        version_tag,
        w_descr,
        descr_type,
        descr_version_tag,
        descr_map,
        w_get,
    ))
}

/// Pins for inlining `type(descr).__get__(descr, obj_or_None, owner)`.
///
/// `receiver_map` is null when `class_access` is set. A type receiver has no
/// instance map; `descr_getattribute` passes `space.w_None` as the instance.
pub struct NondatadescrGetPins {
    pub w_owner: PyObjectRef,
    pub owner_version_tag: u64,
    pub receiver_map: MapRef,
    pub w_descr: PyObjectRef,
    pub descr_type: PyObjectRef,
    pub descr_version_tag: u64,
    pub descr_map: MapRef,
    pub w_get: PyObjectRef,
    pub class_access: bool,
    /// `ObjectMutableCell` holding `w_descr` when the class assignment is a
    /// cell. Null when the name is a bare value. An in-place `write_cell`
    /// does not move `owner_version_tag`; the caller getfields this payload.
    pub attr_cell: PyObjectRef,
}

/// `descroperation.py DescrOperation.get`'s general arm: `__get__` on
/// `type(w_descr)`, plus the descriptor map and version that pin it.
///
/// Exact `classmethod` / `staticmethod` / `function` decline. Their `__get__`
/// is the typedef entry, and the unwrap folds already cover the exact layout.
/// A subclass override is a Python function on a `_getusercls` instance
/// (`typedef.py _getusercls`), which is the body the caller records.
///
/// # Safety
/// `w_descr` must be a live object.
pub unsafe fn python_descr_get_pins(
    w_descr: PyObjectRef,
) -> Option<(PyObjectRef, u64, MapRef, PyObjectRef)> {
    if w_descr.is_null() {
        return None;
    }
    let ob_type = unsafe { (*w_descr).ob_type };
    if std::ptr::eq(ob_type, &crate::FUNCTION_TYPE as *const _)
        || std::ptr::eq(ob_type, &crate::METHOD_DESCRIPTOR_TYPE as *const _)
        || unsafe { pyre_object::function::is_exact_classmethod(w_descr) }
        || unsafe { pyre_object::function::is_exact_staticmethod(w_descr) }
    {
        return None;
    }
    let descr_type = crate::typedef::r#type(w_descr)?.as_ptr();
    let descr_version_tag = unsafe { crate::baseobjspace::w_type_version_tag(descr_type) };
    if descr_version_tag == 0 {
        return None;
    }
    let descr_map = unsafe { mapdict_map_or_null(w_descr) };
    if descr_map.is_null() || unsafe { map_is_devolved(descr_map) } {
        return None;
    }
    // An in-place cell write does not move `descr_version_tag`.
    // `type_attr_stored_is_cell` and `lookup_in_type_where` reach
    // `box_str_constant`, so the descriptor type stays pinned across both.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[descr_type]);
    if unsafe {
        crate::baseobjspace::type_attr_stored_is_cell(roots.get(base), Wtf8::new("__get__"))
    } {
        return None;
    }
    let w_get = unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(base), "__get__") }?;
    Some((roots.get(base), descr_version_tag, descr_map, w_get))
}

/// LOAD_ATTR of a non-data descriptor, without calling `__get__`.
///
/// `objectobject.py Object.descr__getattribute__` calls
/// `descroperation.py DescrOperation.get` only after the instance dict misses.
/// `typeobject.py W_TypeObject.descr_getattribute` calls the same `get` on the
/// class-MRO value with `space.w_None` once a metatype data descriptor has
/// lost. Both are the call this returns pins for.
///
/// A receiver with `__getattr__` declines, matching
/// [`data_descriptor_get_fast_path`]: an `AttributeError` from the inlined
/// getter has to reach `_handle_getattribute`'s fallback, which this direct
/// sub-walk does not encode. A devolved map cannot prove the instance miss
/// (`mapdict.py` `DevolvedDictTerminator`).
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn nondatadescr_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<NondatadescrGetPins> {
    if unsafe { pyre_object::typeobject::is_type(w_obj) } {
        return unsafe { nondatadescr_type_get_fast_path(w_obj, name) };
    }
    unsafe { nondatadescr_instance_get_fast_path(w_obj, name) }
}

unsafe fn nondatadescr_instance_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<NondatadescrGetPins> {
    // A typedef-owned `__dict__` is invisible to the map (`typedef.py`
    // `_getusercls` without `MapdictDictSupport`).
    if unsafe { typedef_owns_dict(w_obj) } {
        return None;
    }
    let receiver_map = unsafe { mapdict_map_or_null(w_obj) };
    if receiver_map.is_null() || unsafe { map_is_devolved(receiver_map) } {
        return None;
    }
    let w_type = unsafe { (*(*receiver_map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // The lookups below reach `box_str_constant`. The type is pinned
    // across the hook checks, then the cell and the descriptor join it
    // before the calls that keep those words live.
    let roots = pyre_object::gc_roots::push_roots();
    let type_at = roots.pin_roots(&[w_type]);
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(roots.get(type_at)) }.is_some()
        || unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(type_at), "__getattr__") }
            .is_some()
    {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(roots.get(type_at)) };
    if version_tag == 0 {
        return None;
    }
    // A class-body assignment is an `ObjectMutableCell`. `write_cell` updates
    // it in place and leaves `version_tag` alone, so the caller getfields the
    // payload the way [`getattr_hook_fast_path`] does for `__getattr__`.
    // An `IntMutableCell` is not a descriptor.
    let attr_cell = unsafe {
        crate::baseobjspace::type_attr_object_cell(
            roots.get(type_at),
            rustpython_wtf8::Wtf8::new(name),
        )
    };
    let cell_at = roots.pin_roots(&[roots.get(type_at), attr_cell]);
    if roots.get(cell_at + 1).is_null()
        && unsafe {
            crate::baseobjspace::type_attr_stored_is_cell(
                roots.get(cell_at),
                rustpython_wtf8::Wtf8::new(name),
            )
        }
    {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(cell_at), name) }?;
    let descr_at = roots.pin_roots(&[roots.get(cell_at), roots.get(cell_at + 1), w_descr]);
    // Data descriptors ignore the instance dict and have their own inline.
    if unsafe { crate::baseobjspace::is_data_descr(roots.get(descr_at + 2)) } {
        return None;
    }
    // `mapdict.py find_map_attr`: a stored instance attribute wins over a
    // non-data descriptor. The map pin is what keeps this miss true.
    if unsafe { find_map_attr(receiver_map, Wtf8::new(name), DICT) }.is_some() {
        return None;
    }
    let (descr_type, descr_version_tag, descr_map, w_get) =
        unsafe { python_descr_get_pins(roots.get(descr_at + 2)) }?;
    Some(NondatadescrGetPins {
        w_owner: roots.get(descr_at),
        owner_version_tag: version_tag,
        receiver_map,
        w_descr: roots.get(descr_at + 2),
        descr_type,
        descr_version_tag,
        descr_map,
        w_get,
        class_access: false,
        attr_cell: roots.get(descr_at + 1),
    })
}

unsafe fn nondatadescr_type_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<NondatadescrGetPins> {
    // `metaclass_keeps_type_getattribute` and the lookups below reach
    // `box_str_constant`. The class and its metaclass are pinned before
    // those calls; a metatype data descriptor is pinned on its own before
    // `is_data_descr`.
    let roots = pyre_object::gc_roots::push_roots();
    let obj_at = roots.pin_roots(&[w_obj]);
    if !unsafe { crate::baseobjspace::metaclass_keeps_type_getattribute(roots.get(obj_at)) } {
        return None;
    }
    let metatype = crate::typedef::r#type(roots.get(obj_at))?.as_ptr();
    let meta_at = roots.pin_roots(&[roots.get(obj_at), metatype]);
    // `descr_getattribute` raises into the metaclass `__getattr__`. This
    // sub-walk has no fallback frame for that hook.
    if unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(meta_at + 1), "__getattr__") }
        .is_some()
    {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(roots.get(meta_at)) };
    if version_tag == 0 {
        return None;
    }
    // typeobject.py `W_TypeObject.descr_getattribute`: a metatype data
    // descriptor preempts the class MRO. `__name__` is that case.
    if let Some(descr) =
        unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(meta_at + 1), name) }
    {
        let gate = roots.pin_roots(&[roots.get(meta_at), descr]);
        if unsafe { crate::baseobjspace::is_data_descr(roots.get(gate + 1)) } {
            return None;
        }
    }
    // Same cell as the instance arm: the name lives on this class.
    let attr_cell = unsafe {
        crate::baseobjspace::type_attr_object_cell(
            roots.get(meta_at),
            rustpython_wtf8::Wtf8::new(name),
        )
    };
    let cell_at = roots.pin_roots(&[roots.get(meta_at), attr_cell]);
    if roots.get(cell_at + 1).is_null()
        && unsafe {
            crate::baseobjspace::type_attr_stored_is_cell(
                roots.get(cell_at),
                rustpython_wtf8::Wtf8::new(name),
            )
        }
    {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(roots.get(cell_at), name) }?;
    let descr_at = roots.pin_roots(&[roots.get(cell_at), roots.get(cell_at + 1), w_descr]);
    if unsafe { crate::baseobjspace::is_data_descr(roots.get(descr_at + 2)) } {
        return None;
    }
    let (descr_type, descr_version_tag, descr_map, w_get) =
        unsafe { python_descr_get_pins(roots.get(descr_at + 2)) }?;
    Some(NondatadescrGetPins {
        w_owner: roots.get(descr_at),
        owner_version_tag: version_tag,
        receiver_map: std::ptr::null(),
        w_descr: roots.get(descr_at + 2),
        descr_type,
        descr_version_tag,
        descr_map,
        w_get,
        class_access: true,
        attr_cell: roots.get(descr_at + 1),
    })
}

/// STORE_ATTR `property` fast path: the setter twin of
/// [`property_get_fast_path`], returning the type, version tag, the property
/// object, and its Python `fset` when `obj.name = value` writes a property
/// setter.  Returns `None` (leave the residual) for a read-only property or any
/// shape [`property_descr_fast_path`] declines.  A custom `__setattr__` owns
/// the write (mapdict.py), so it declines to the residual.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn property_set_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef, PyObjectRef)> {
    let (w_type, version_tag, w_descr) = unsafe { property_descr_fast_path(w_obj, name) }?;
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    let fset = unsafe { pyre_object::descriptor::w_property_get_fset(w_descr) };
    if fset.is_null() || unsafe { pyre_object::pyobject::is_none(fset) } {
        return None;
    }
    Some((w_type, version_tag, w_descr, fset))
}

/// The unboxed counterpart of [`load_attr_fast_path`].  It applies the same
/// LOAD_ATTR resolution gates, but accepts only an `UnboxedPlainAttribute`
/// and returns the shared longlong-list coordinates and type needed to perform
/// `_prim_direct_read` (mapdict.py).
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn load_attr_unboxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, usize, UnboxType, MapRef)> {
    // mapdict.py:1495 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1496 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access;
    // not soundly foldable to a storage read.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1504-1524 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    // mapdict.py `if attrkind != INVALID:`.
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    let u = p.unboxed.as_ref()?;
    // mapdict.py — `LOAD_ATTR_caching` resolves to `attr._direct_read`,
    // which for an unboxed attribute also migrates `obj` off unboxed storage
    // once its class froze unboxing (mapdict.py). The folded read
    // performs `_prim_direct_read` alone, so leave a frozen class to the
    // residual, whose `direct_read` still runs that migration.
    if !unsafe { (*p.terminator).as_terminator() }
        .allow_unboxing
        .get()
    {
        return None;
    }
    Some((
        w_type,
        version_tag,
        map,
        p.storageindex,
        u.listindex,
        u.typ,
        attr,
    ))
}

/// The STORE_ATTR counterpart of [`load_attr_unboxed_fast_path`].  An
/// existing plain unboxed slot resolves through the same map lookup; the
/// caller separately proves that the incoming value has the slot's unbox
/// type before performing the in-place write (mapdict.py).
///
/// The resolved attribute node is returned so the caller can mark it
/// `ever_mutated` once it has committed to folding the write out of the trace.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn store_attr_unboxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, usize, UnboxType, MapRef)> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1592 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py — a custom `__setattr__` owns the write.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1630-1631 — STORE caching also requires the standard
    // `__getattribute__` so the cached map invariant remains valid.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    let u = p.unboxed.as_ref()?;
    Some((
        w_type,
        version_tag,
        map,
        p.storageindex,
        u.listindex,
        u.typ,
        attr,
    ))
}

/// Resolve an existing boxed plain slot through the STORE_ATTR caching gates.
/// Unlike an unboxed slot, the value remains a `PyObjectRef`, so the caller
/// needs only the guarded map and storage index for the direct write
/// (mapdict.py).
///
/// The resolved attribute node is returned so the caller can mark it
/// `ever_mutated` once it has committed to folding the write out of the trace.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn store_attr_boxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, MapRef)> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1592 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py — a custom `__setattr__` owns the write.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1630-1631 — STORE caching also requires the standard
    // `__getattribute__` so the cached map invariant remains valid.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    if p.unboxed.is_some() {
        return None;
    }
    Some((w_type, version_tag, map, p.storageindex, attr))
}

/// mapdict.py:1635-1636 `if not attr.ever_mutated: attr.ever_mutated = True`
/// — performed by the caller once it has committed to folding the store
/// out of the trace, so a declined resolution mutates nothing.
///
/// # Safety
/// `attr` must point at a live `PlainAttribute` map node.
pub unsafe fn mark_attr_ever_mutated(attr: MapRef) {
    unsafe { (*attr).as_plain() }.set_ever_mutated(true);
}

/// The map transition an attribute-adding STORE_ATTR takes, resolved without
/// performing it.  See [`store_attr_add_fast_path`].
pub struct StoreAttrAdd {
    /// The receiver's type, whose `version_tag` the caller guards.
    pub w_type: PyObjectRef,
    /// That type's live version tag.
    pub version_tag: u64,
    /// The instance's current map, which the caller guards.
    pub map: MapRef,
    /// The cached transition holder whose `typ` and `attr` the resolution read.
    pub holder: *const CachedAttributeHolder,
    /// The `PlainAttribute` the instance's map becomes — a direct child of
    /// `map`, so the transition is `map -> new_map` with no reordering.
    pub new_map: MapRef,
    /// `new_map.storageindex`, which is also `map.storage_needed()`: the
    /// transition appends exactly one slot, and the value lands in it.
    pub storageindex: usize,
    /// `Some` for a fresh-slot unboxed transition: the new slot holds a
    /// one-element longlong list. Only `Int` today; a float pick stays on
    /// the residual.
    pub unbox_type: Option<UnboxType>,
    /// The value returned by `pick_unbox_type`. This is distinct from
    /// `unbox_type`, which is `None` when a cached boxed transition serves an
    /// unboxable value. `picked_unbox.is_some()` means the pick read the
    /// terminator's `allow_unboxing` flag as true.
    pub picked_unbox: Option<UnboxType>,
}

/// Resolve the `map -> PlainAttribute` transition that a STORE_ATTR adding a
/// not-yet-present boxed attribute would take, *without* performing it, so the
/// JIT can emit `_set_mapdict_increase_storage1` (mapdict.py) as trace
/// operations instead of an opaque residual call.  Every value the caller needs
/// is a trace-time constant once the receiver's map and its type's version tag
/// are guarded.
///
/// Declines anything the plain grow-by-one shape does not cover, so the caller
/// can fall back to the general residual: an attribute already in the map (the
/// [`store_attr_boxed_fast_path`] / [`store_attr_unboxed_fast_path`] cases), a
/// value that would take an unboxed float slot or a shared unboxed slot, a
/// terminator that routes the write elsewhere, the `_reorder_and_add` case
/// (mapdict.py), the `LIMIT_MAP_ATTRIBUTES` devolve, and a storage
/// block the collector does not own (whose replacement the interpreter would
/// have to free).  A fresh-slot unboxed int add resolves with `unbox_type =
/// Some(Int)` and grows storage by one exactly like the boxed add.
///
/// Resolving interns the transition through `_find_branch_to_move_into`, which
/// `add_attr` would do anyway on this very store; it is idempotent and the
/// resulting node is immortal.
///
/// # Safety
/// `w_obj` must be a live object and `w_value` a live value.
pub unsafe fn store_attr_add_fast_path(
    w_obj: PyObjectRef,
    name: &str,
    w_value: PyObjectRef,
) -> Option<StoreAttrAdd> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let terminator = unsafe { (*map).terminator() };
    let term = unsafe { (*terminator).as_terminator() };
    let w_type = term.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1612-1614 / 1630-1631 — a custom `__setattr__` owns the
    // write, and STORE caching also requires the standard `__getattribute__`.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // `classify_attr` settles the attrkind from the type-level entry and the
    // fold bakes that verdict.  A `MutableCell` entry is one `write_cell`
    // rewrites in place without moving `version_tag`, so a rebind from a plain
    // value to a data descriptor would leave the classification stale.
    if unsafe { crate::baseobjspace::type_attr_stored_is_cell(w_type, Wtf8::new(name)) } {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = Wtf8::new(if is_slot { "slot" } else { name });
    // This arm is the ADD; an attribute already in the map belongs to the
    // in-place write paths (mapdict.py `map.find_map_attr`).
    if unsafe { find_map_attr(map, attrname, attrkind) }.is_some() {
        return None;
    }
    // mapdict.py `_write_terminator` — a `NoDictTerminator` rejects a
    // DICT write and a `DevolvedDictTerminator` sends it to the materialised
    // instance dict; neither reaches `add_attr`.
    if attrkind == DICT && term.kind != TerminatorKind::Dict {
        return None;
    }
    // The int unboxed shape is emittable — the trace builds the same int
    // GcArray `erase_unboxed` allocates; a float pick stays on the residual.
    let unbox = unsafe { pick_unbox_type(map, w_value) };
    if unbox == Some(UnboxType::Float) {
        return None;
    }
    // The emitted transition copies the live slots into the wider block with
    // reference loads. An unboxed attribute already in the chain is fine: its
    // slot holds the longlong list's GcArray, which is a reference like any
    // other (mapdict.py `_prim_direct_read`).
    // mapdict.py — `number_to_readd != 0` is `_reorder_and_add`, which
    // pops and re-adds intermediate attributes.
    let (number_to_readd, holder) =
        unsafe { find_branch_to_move_into(map, attrname, attrkind, unbox) };
    if number_to_readd != 0 {
        return None;
    }
    let new_map = unsafe { holder_pick_attr(holder, unbox) };
    if !unsafe { (*new_map).is_plain() } {
        return None;
    }
    let p = unsafe { (*new_map).as_plain() };
    if !std::ptr::eq(p.back, map) {
        return None;
    }
    let unbox_type = match &p.unboxed {
        // A cached boxed transition serves an unboxable value too; the write
        // stays boxed.
        None => None,
        // Only the fresh-slot shape grows storage by one; a shared-slot add
        // rewrites an existing slot's list in place.
        Some(u) => {
            if !u.firstunwrapped || u.listindex != 0 {
                return None;
            }
            debug_assert_eq!(Some(u.typ), unbox);
            Some(u.typ)
        }
    };
    // mapdict.py — a DICT add that reaches the attribute limit devolves
    // the instance's `__dict__` into a text-strategy dict.
    if attrkind == DICT && unsafe { (*new_map).num_attributes() } >= LIMIT_MAP_ATTRIBUTES {
        return None;
    }
    // The grow-by-one shape `_set_mapdict_increase_storage1` implements: the
    // new slot is appended at the current length.
    let storage_needed = unsafe { (*map).storage_needed() };
    if p.storageindex != storage_needed
        || unsafe { (*new_map).storage_needed() } != storage_needed + 1
    {
        return None;
    }
    // `grow_instance_items_block` frees the block it replaces, which is a no-op
    // only for a collector-owned block.  A `std::alloc` fallback block (no GC
    // hook) would leak once the emitted allocation replaces it without a free.
    let inst = unsafe { mapdict_carrier(w_obj) };
    let storage = unsafe { inst.storage() };
    if !storage.is_null()
        && !pyre_object::gc_hook::try_gc_owns_object(storage as pyre_object::gc_hook::GCREF)
    {
        return None;
    }
    // The emitted transition allocates a `storage_needed + 1` block, so spare
    // capacity would make it a shrink — which is exactly what
    // `_set_mapdict_storage_and_map` refuses to do. An instance that has had an
    // attribute deleted therefore keeps the interpreter's add path, which fills
    // the spare tail in place.
    if unsafe { pyre_object::object_array::items_block_capacity(inst.storage()) } != storage_needed
    {
        return None;
    }
    Some(StoreAttrAdd {
        w_type,
        version_tag,
        map,
        holder,
        new_map,
        storageindex: p.storageindex,
        unbox_type,
        picked_unbox: unbox,
    })
}

/// Perform the transition [`store_attr_add_fast_path`] resolved.  The JIT's
/// walk is the authoritative execution path, so it applies the store itself
/// after emitting the equivalent trace operations.
///
/// # Safety
/// `w_obj` must be the live instance the resolution came from, still on
/// `resolved.map`, and `w_value` a live value.
pub unsafe fn store_attr_add_commit(
    w_obj: PyObjectRef,
    resolved: &StoreAttrAdd,
    w_value: PyObjectRef,
) {
    let _instance_guard = instance_lock(w_obj);
    let mut inst = unsafe { mapdict_carrier(w_obj) };
    debug_assert!(std::ptr::eq(inst._get_mapdict_map(), resolved.map));
    // mapdict.py with `storage_needed() > _mapdict_storage_length()`,
    // which `store_attr_add_fast_path` proved by construction.
    match resolved.unbox_type {
        None => inst._set_mapdict_increase_storage1(resolved.new_map, w_value),
        // `switch_map_and_write_increase_storage1`'s `firstunwrapped` arm: a
        // fresh one-element longlong list occupies the new slot.
        Some(typ) => {
            let val = unsafe { unbox_value(typ, w_value) };
            let unboxed = erase_unboxed(&[val]);
            inst._set_mapdict_increase_storage1(resolved.new_map, unboxed);
        }
    }
}

/// mapdict.py `_mapdict_read_storage(storageindex)` for a
/// `W_ObjectObject` — the boxed-slot read the JIT LOAD_ATTR fast path folds
/// to. `load_attr_fast_path` already established that the resolved attribute
/// is a boxed plain slot (not unboxed), so this is a straight
/// `storage[storageindex]` fetch with no conversion.
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose `storage` holds at least
/// `storageindex + 1` slots — guaranteed when `storageindex` came from this
/// instance's map via `load_attr_fast_path`.
pub unsafe fn read_boxed_storage(w_obj: PyObjectRef, storageindex: usize) -> PyObjectRef {
    let inst = unsafe { mapdict_carrier(w_obj) };
    inst._mapdict_read_storage(storageindex)
}

/// Write a boxed attribute value directly to an existing storage slot,
/// matching `PlainAttribute._direct_write` (mapdict.py).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// boxed storage index.
pub unsafe fn write_boxed_storage(w_obj: PyObjectRef, storageindex: usize, value: PyObjectRef) {
    let mut inst = unsafe { mapdict_carrier(w_obj) };
    inst._mapdict_write_storage(storageindex, value);
}

/// Read the raw longlong at an unboxed attribute's `(storageindex,
/// listindex)`, matching `_prim_direct_read` before it boxes the value
/// (mapdict.py).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// unboxed storage coordinates.
pub unsafe fn read_unboxed_storage_raw(
    w_obj: PyObjectRef,
    storageindex: usize,
    listindex: usize,
) -> i64 {
    let slot = unsafe { read_boxed_storage(w_obj, storageindex) };
    unsafe { unboxed_items(slot)[listindex] }
}

/// Write a raw longlong to an unboxed attribute's `(storageindex,
/// listindex)`, matching the same-type `_direct_write` update before any
/// boxing conversion (mapdict.py).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// unboxed storage coordinates.
pub unsafe fn write_unboxed_storage_raw(
    w_obj: PyObjectRef,
    storageindex: usize,
    listindex: usize,
    raw: i64,
) {
    let slot = unsafe { read_boxed_storage(w_obj, storageindex) };
    unsafe {
        unboxed_items_mut(slot)[listindex] = raw;
    }
}

/// mapdict.py `STORE_ATTR_caching`. The interpreter STORE_ATTR fast
/// path (reached only under `not we_are_jitted()`): a monomorphic hit writes
/// straight through the cached attribute; anything else drops to
/// `store_attr_slowpath`.
///
/// `dont_look_inside` — same rationale as `load_attr_caching`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object.
#[majit_macros::dont_look_inside]
pub unsafe fn store_attr_caching(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    w_value: PyObjectRef,
) -> Result<(), PyError> {
    let entry = {
        let _code_cache_guard = code_cache_lock(pycode);
        unsafe { crate::pycode::w_code_mapdict_caches_get(pycode, nameindex) }
    };
    let map = {
        let _instance_guard = instance_lock(w_obj);
        // mapdict.py `map = w_obj._get_mapdict_map()`.
        let map = unsafe { mapdict_map_or_null(w_obj) };
        if let Some(e) = entry {
            // mapdict.py:1578-1585 — valid cache entry with a live attr.
            if !map.is_null()
                && unsafe { e.is_valid_for_map(map, true) }
                && e.w_method.is_null()
                && !e.cached_attr.is_null()
            {
                let attr = e.cached_attr;
                let p = unsafe { (*attr).as_plain() };
                // mapdict.py:1582-1583 `if not attr.ever_mutated: attr.ever_mutated = True`.
                p.set_ever_mutated(true);
                // mapdict.py `attr._direct_write(w_obj, w_value)`.
                let mut inst = unsafe { mapdict_carrier(w_obj) };
                unsafe { plain_direct_write(attr, &mut inst, w_value) };
                return Ok(());
            }
        }
        map
    };
    unsafe { store_attr_slowpath(pycode, w_obj, nameindex, name, map, w_value, entry) }
}

/// mapdict.py `STORE_ATTR_slowpath`.
///
/// `dont_look_inside` — same rationale as `load_attr_slowpath`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object; `map` its map
/// (or null).
#[majit_macros::dont_look_inside]
unsafe fn store_attr_slowpath(
    mut pycode: PyObjectRef,
    mut w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    map: MapRef,
    mut w_value: PyObjectRef,
    entry: Option<MapdictCacheEntry>,
) -> Result<(), PyError> {
    // `object.__class__` is a getset data descriptor in PyPy, so `_classify_attr`
    // (classify_attr) marks it INVALID and the store falls through to
    // `space.setattr` (mapdict.py) — the assignment re-roots the instance
    // type via `descr_set___class__`, not the instance dict.  pyre models
    // `__class__` through `object_setattr`'s special-case rather than a getset,
    // so it never surfaces as a data descriptor for classify_attr to reject;
    // exclude it from the store cache here so `obj.__class__ = NewCls` reaches
    // that special-case instead of being mis-stored as an ordinary instance-dict
    // attribute (which leaves the real type unchanged).
    if name == "__class__" {
        return crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ());
    }
    // mapdict.py:1591 `if map is not None:`.
    if !map.is_null() {
        // mapdict.py:1592 `w_type = map.terminator.w_cls`.
        let mut w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
        // mapdict.py:1593 `version_tag = w_type.version_tag()`.
        let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
        // mapdict.py:1596-1611 — fast path for stores that add a new attribute
        // that this slot has already cached the transition for.
        if let Some(e) = entry
            && e.valid_for_store
            && version_tag == e.version_tag
        {
            let entry_map = e.cached_map;
            let attr_to_add = e.cached_attr;
            // mapdict.py:1599-1602 `entry_map is not None and
            // isinstance(entry_map, PlainAttribute) and attr_to_add is entry_map
            // and entry_map.back is map`.
            if !entry_map.is_null()
                && unsafe { (*entry_map).is_plain() }
                && std::ptr::eq(attr_to_add, entry_map)
                && std::ptr::eq(unsafe { (*entry_map).as_plain() }.back, map)
            {
                // mapdict.py:1603-1606 — for an unboxed attr the new value must
                // match the unbox type, else fall through to the general path.
                let p = unsafe { (*attr_to_add).as_plain() };
                let typsafe = match &p.unboxed {
                    Some(u) => unsafe { value_has_unbox_type(u.typ, w_value) },
                    None => true,
                };
                if typsafe {
                    let _roots = pyre_object::gc_roots::push_roots();
                    let base = pyre_object::gc_roots::pin_roots(&[pycode, w_obj, w_type, w_value]);
                    let switched = {
                        let _instance_guard =
                            instance_lock(pyre_object::gc_roots::shadow_stack_get(base + 1));
                        let current_map = unsafe {
                            mapdict_map_or_null(pyre_object::gc_roots::shadow_stack_get(base + 1))
                        };
                        if std::ptr::eq(current_map, map) {
                            // mapdict.py:1610
                            // `_switch_map_and_write_increase_storage1`.
                            let mut inst = unsafe {
                                mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(base + 1))
                            };
                            unsafe {
                                switch_map_and_write_increase_storage1(
                                    attr_to_add,
                                    &mut inst,
                                    pyre_object::gc_roots::shadow_stack_get(base + 3),
                                )
                            };
                            true
                        } else {
                            false
                        }
                    };
                    pycode = pyre_object::gc_roots::shadow_stack_get(base);
                    w_obj = pyre_object::gc_roots::shadow_stack_get(base + 1);
                    w_type = pyre_object::gc_roots::shadow_stack_get(base + 2);
                    w_value = pyre_object::gc_roots::shadow_stack_get(base + 3);
                    if switched {
                        return Ok(());
                    }
                }
            }
        }
        // mapdict.py:1612-1614 — a custom `__setattr__` handles the store. pyre
        // re-dispatches through `space.setattr` (no separate helper).
        if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
            return crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ());
        }
        // mapdict.py:1616 `if version_tag is not None:` (0 = None).
        if version_tag != 0 {
            // mapdict.py:1618-1619 `_, w_descr = _pure_lookup_where_with_method_cache`
            // -- the raw entry, so `classify_attr` can give up on a MutableCell.
            let w_descr = unsafe { crate::baseobjspace::type_attr_stored(w_type, Wtf8::new(name)) };
            // mapdict.py:1620-1626 classify (no non-data heaptype branch for STORE).
            let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
            // mapdict.py `if attrkind != INVALID:`.
            if attrkind != INVALID {
                let attrname = if is_slot { "slot" } else { name };
                // mapdict.py `attr = map.find_map_attr(attrname, attrkind)`.
                match unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) } {
                    Some(attr) => {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let base =
                            pyre_object::gc_roots::pin_roots(&[pycode, w_obj, w_type, w_value]);
                        let written = {
                            let _instance_guard =
                                instance_lock(pyre_object::gc_roots::shadow_stack_get(base + 1));
                            let current_map = unsafe {
                                mapdict_map_or_null(pyre_object::gc_roots::shadow_stack_get(
                                    base + 1,
                                ))
                            };
                            if !std::ptr::eq(current_map, map)
                                || unsafe {
                                    find_map_attr(current_map, Wtf8::new(attrname), attrkind)
                                }
                                .is_none()
                            {
                                false
                            } else {
                                let p = unsafe { (*attr).as_plain() };
                                // mapdict.py:1632-1633
                                // `if not attr.ever_mutated: ...`.
                                p.set_ever_mutated(true);
                                // mapdict.py `attr._direct_write(...)`.
                                let mut inst = unsafe {
                                    mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(
                                        base + 1,
                                    ))
                                };
                                unsafe {
                                    plain_direct_write(
                                        attr,
                                        &mut inst,
                                        pyre_object::gc_roots::shadow_stack_get(base + 3),
                                    )
                                };
                                true
                            }
                        };
                        pycode = pyre_object::gc_roots::shadow_stack_get(base);
                        w_obj = pyre_object::gc_roots::shadow_stack_get(base + 1);
                        w_type = pyre_object::gc_roots::shadow_stack_get(base + 2);
                        w_value = pyre_object::gc_roots::shadow_stack_get(base + 3);
                        if !written {
                            return crate::baseobjspace::setattr_str(w_obj, name, w_value)
                                .map(|_| ());
                        }
                        // mapdict.py:1630-1631 — fill only when there is no custom
                        // `__getattribute__` to upset the cache invariant.
                        if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }
                            .is_none()
                        {
                            unsafe {
                                fill_cache(
                                    pycode,
                                    nameindex,
                                    map,
                                    version_tag,
                                    attr,
                                    std::ptr::null_mut(),
                                    true,
                                );
                            }
                        }
                        return Ok(());
                    }
                    None => {
                        // mapdict.py — add a brand-new DICT attribute via
                        // the DictTerminator, then fill the slot with the resulting
                        // transition map.
                        if attrkind == DICT
                            && unsafe { (*(*map).terminator()).as_terminator() }.kind
                                == TerminatorKind::Dict
                        {
                            let term = unsafe { (*map).terminator() };
                            let _roots = pyre_object::gc_roots::push_roots();
                            let base =
                                pyre_object::gc_roots::pin_roots(&[pycode, w_obj, w_type, w_value]);
                            let mapnew = {
                                let _instance_guard = instance_lock(
                                    pyre_object::gc_roots::shadow_stack_get(base + 1),
                                );
                                let current_map = unsafe {
                                    mapdict_map_or_null(pyre_object::gc_roots::shadow_stack_get(
                                        base + 1,
                                    ))
                                };
                                if !std::ptr::eq(current_map, map) {
                                    std::ptr::null()
                                } else {
                                    let mut inst = unsafe {
                                        mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(
                                            base + 1,
                                        ))
                                    };
                                    // mapdict.py:1639
                                    // `map.terminator._write_terminator(...)`.
                                    unsafe {
                                        write_terminator(
                                            term,
                                            &mut inst,
                                            Wtf8::new(name),
                                            attrkind,
                                            pyre_object::gc_roots::shadow_stack_get(base + 3),
                                        )
                                    };
                                    // mapdict.py:1640
                                    // `mapnew = w_obj._get_mapdict_map()`.
                                    inst._get_mapdict_map()
                                }
                            };
                            pycode = pyre_object::gc_roots::shadow_stack_get(base);
                            w_obj = pyre_object::gc_roots::shadow_stack_get(base + 1);
                            w_type = pyre_object::gc_roots::shadow_stack_get(base + 2);
                            w_value = pyre_object::gc_roots::shadow_stack_get(base + 3);
                            if mapnew.is_null() {
                                return crate::baseobjspace::setattr_str(w_obj, name, w_value)
                                    .map(|_| ());
                            }
                            // mapdict.py:1642-1648 — fill only when no attribute
                            // reordering happened (the new attr is the leaf whose
                            // `back` is the pre-write map).
                            if unsafe { (*mapnew).is_plain() }
                                && std::ptr::eq(unsafe { (*mapnew).as_plain() }.back, map)
                                && unsafe {
                                    crate::baseobjspace::getattribute_if_not_from_object(w_type)
                                }
                                .is_none()
                            {
                                unsafe {
                                    fill_cache(
                                        pycode,
                                        nameindex,
                                        mapnew,
                                        version_tag,
                                        mapnew,
                                        std::ptr::null_mut(),
                                        true,
                                    );
                                }
                            }
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
    // mapdict.py `space.setattr(w_obj, w_name, w_value)`.
    crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ())
}

// ── obj storage protocol (mapdict.py MapdictStorageMixin) ──────
//
// The map-node layer reads and writes attribute values through this trait.
// PyPy mixes `MapdictStorageMixin` into the instance class; pyre's instance
// (W_ObjectObject) implements this trait instead. Storage holds
// `PyObjectRef`, so PyPy's `erase_item`/`unerase_item` (rerased boxing of a
// W_Root into the untyped storage list) are the identity here.

pub trait MapdictObject {
    /// mapdict.py `_get_mapdict_map` (`jit.promote(self.map)`).
    fn _get_mapdict_map(&self) -> MapRef;
    /// mapdict.py `_set_mapdict_map`.
    fn _set_mapdict_map(&mut self, map: MapRef);
    /// mapdict.py `_mapdict_read_storage`.
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef;
    /// mapdict.py `_mapdict_write_storage`.
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef);
    /// mapdict.py `_mapdict_storage_length` (= `self.map.storage_needed()`).
    fn _mapdict_storage_length(&self) -> usize;
    /// mapdict.py `_mapdict_pop_attribute`.
    fn _mapdict_pop_attribute(&mut self, map: MapRef);
    /// mapdict.py `_set_mapdict_increase_storage1`.
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef);
    /// mapdict.py `_set_mapdict_storage_and_map` — install a complete
    /// replacement storage list and map (used by `delete`/`copy`).
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef);
    /// mapdict.py `MapdictDictSupport.getdict` → `_obj_getdict`
    /// (mapdict.py) — the instance's `__dict__`. Only the live instance
    /// carrier provides this; the transient `Object` (mapdict.py,
    /// copy/materialize result) lacks `MapdictDictSupport`, so its impl is
    /// unreachable. Used by `DevolvedDictTerminator`'s read/write/delete
    /// (mapdict.py), which only ever run on the live instance.
    fn getdict(&self) -> PyObjectRef;
}

/// `W_ObjectObject` (`pyre-object`) is pyre's `MapdictStorageMixin`
/// carrier (`mapdict.py`): PyPy mixes the mixin into the instance
/// class, here the instance implements the trait. `map` is the erased
/// `*const MapNode`; `storage` is a heap `Vec<PyObjectRef>` (null =
/// `None`, the `_mapdict_init_empty` empty state, mapdict.py).
/// Remember an instance that may now hold a young attribute value, mirroring
/// `dict_write_barrier` (dictmultiobject.rs). The storage block is remembered
/// separately by [`mapdict_storage_write_barrier`], which every caller runs
/// next: a compiled `SETARRAYITEM` names the array, and an old block's
/// contents are scanned only when that block is on the remembered set.
/// `setarrayitem_gc` barriers that array, so the interpreter store does
/// too — otherwise the block stays TRACK-set and a non-moving major's
/// `debug_check_not_white` sees black storage → white nursery values.
fn instance_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
}

fn mapdict_storage_write_barrier(storage: *mut pyre_object::object_array::ItemsBlock) {
    if !storage.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(storage as pyre_object::gc_hook::GCREF)
    {
        pyre_object::gc_hook::try_gc_write_barrier(storage as pyre_object::gc_hook::GCREF);
    }
}

impl MapdictObject for pyre_object::W_ObjectObject {
    fn _get_mapdict_map(&self) -> MapRef {
        // `jit.promote(self.map)` (mapdict.py _get_mapdict_map).
        self.map as MapRef
    }
    fn _set_mapdict_map(&mut self, map: MapRef) {
        self.map = map as usize;
    }
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        // mapdict.py. A read is always preceded by the
        // `_set_mapdict_increase_storage1` that made `storage` non-null; the
        // block is exact-size (capacity == map.storage_needed()), so
        // `storageindex` (from this instance's map) is always in range.
        unsafe {
            let base = pyre_object::object_array::items_block_items_base(self.storage);
            *base.add(storageindex)
        }
    }
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        // mapdict.py. The instance is the remembered-set root: the
        // collector reaches the storage block (nursery-born by
        // `alloc_mapdict_storage_block`) only through this instance's
        // `object_object_custom_trace`, which forwards the `storage` word and
        // walks the block's boxed slots in place, so remembering the instance
        // keeps a young value stored into an old-gen instance's block
        // forwarded on a minor GC.
        //
        // RPython's GC transform emits the barrier before the pointer store.
        // Publish both operands first because a foreign mutator's collection
        // can make the barrier wait; storing first would leave the new value
        // invisible until after that collection had swept it.
        let _roots = pyre_object::gc_roots::push_roots();
        let livevars = vec![self as *const Self as PyObjectRef, value];
        let self_slot = pyre_object::gc_roots::pin_roots(&livevars);
        let value_slot = self_slot + 1;
        let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
        instance_write_barrier(owner);
        let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
        let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        unsafe {
            let owner = &mut *(owner as *mut Self);
            mapdict_storage_write_barrier(owner.storage);
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
            let owner = &mut *(owner as *mut Self);
            let base = pyre_object::object_array::items_block_items_base(owner.storage);
            *base.add(storageindex) = value;
        }
    }
    fn _mapdict_storage_length(&self) -> usize {
        // mapdict.py (= self.map.storage_needed()).
        unsafe { (*(self.map as MapRef)).storage_needed() }
    }
    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        // mapdict.py _mapdict_pop_attribute; structure mirrors the MockObj test impl.
        // Both arms can collect — the unboxed one through
        // `_mapdict_write_storage`, the other through the storage shrink — so
        // the receiver is published for the whole function and every write
        // goes through the reloaded address, as in
        // `_set_mapdict_increase_storage1` and `_set_mapdict_storage_and_map`.
        // `&mut self` is a bare address that a moving collection invalidates;
        // writing `map` through it would leave the live instance on its old
        // map, claiming a storage slot the shrunk block no longer has.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self as *const Self as PyObjectRef]);
        let current_map = self.map as MapRef;
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) => match u.firstunwrapped {
                    true => None,
                    false => Some(((*current_map).as_plain().storageindex, u.listindex)),
                },
                None => None,
            }
        };
        match unboxed_slot {
            // mapdict.py:931-934: drop the last entry of the shared
            // longlong list (the slot itself stays).
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list: Vec<i64> = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                // `erase_unboxed` allocates the replacement bank, so reload the
                // receiver after it rather than before.
                let unboxed = erase_unboxed(&new_list);
                let owner = unsafe {
                    &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self)
                };
                owner._mapdict_write_storage(storageindex, unboxed);
            }
            // mapdict.py:935-938: truncate storage to the parent map's size.
            // The block itself keeps its capacity (see
            // `_set_mapdict_storage_and_map`); NULL the slots the parent map no
            // longer names so their values are released.
            None => {
                let storage_needed = unsafe { (*map).storage_needed() };
                unsafe {
                    let cap = pyre_object::object_array::items_block_capacity(self.storage);
                    if cap > storage_needed {
                        let base = pyre_object::object_array::items_block_items_base(self.storage);
                        for i in storage_needed..cap {
                            *base.add(i) = pyre_object::PY_NULL;
                        }
                    }
                }
            }
        }
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.map = map as usize;
        }
    }
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        // grow storage by one, append value (mapdict.py _set_mapdict_increase_storage1). The first
        // grow allocates the storage block (was `None`). The new attribute's
        // index comes from the map, not from the block's capacity: a delete
        // leaves the capacity above `storage_needed()` (see
        // `_set_mapdict_storage_and_map`), and that spare tail is where this
        // write lands, so no block is allocated until the map outgrows it.
        let _roots = pyre_object::gc_roots::push_roots();
        let livevars = vec![self as *const Self as PyObjectRef, value];
        let self_slot = pyre_object::gc_roots::pin_roots(&livevars);
        let value_slot = self_slot + 1;
        let needed = unsafe { (*map).storage_needed() };
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            let old = owner.storage;
            let old_slot = pyre_object::gc_roots::shadow_stack_len();
            if !old.is_null() {
                let _ = pyre_object::gc_roots::pin_root(old as PyObjectRef);
            }
            // `pin_root` is a safepoint. mapdict.py re-reads storage after an
            // allocating call; capacity and the in-place write use that block.
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            let old_cap = pyre_object::object_array::items_block_capacity(old);
            let block = if needed <= old_cap {
                old
            } else {
                let grown =
                    pyre_object::object_array::grow_instance_items_block(old, needed, old_cap);
                let _ = pyre_object::gc_roots::pin_root(grown as PyObjectRef);
                let block_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                pyre_object::gc_roots::shadow_stack_get(block_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            let base = pyre_object::object_array::items_block_items_base(block);
            *base.add(needed - 1) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.storage = block;
            owner.map = map as usize;
            instance_write_barrier(owner as *mut Self as PyObjectRef);
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            mapdict_storage_write_barrier(owner.storage);
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            if !std::ptr::eq(block, old) {
                pyre_object::object_array::dealloc_instance_items_block(old);
            }
        }
    }
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        // mapdict.py _set_mapdict_storage_and_map. The incoming `Vec` comes from a lightweight
        // `Object` carrier (delete/copy transplant); convert it to a fresh
        // storage block, freeing any prior block.
        //
        // A delete arrives here with fewer values than the instance already
        // holds, but the block never shrinks. `_mapdict_read_storage` indexes
        // without a bounds check, and the JIT folds an attribute access to that
        // bare `storage[storageindex]` load with its map guard checked *before*
        // it — so with no process GIL a concurrent shrink could leave a
        // just-validated index past the new block's capacity. A monotone
        // capacity keeps every index the guard ever accepted in range; the
        // reader's worst case becomes a stale slot value, which is what an
        // unsynchronised attribute read means anyway. The spare tail is NULL,
        // so the dropped attributes' values are still released.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self as *const Self as PyObjectRef]);
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            let old = owner.storage;
            let old_slot = pyre_object::gc_roots::shadow_stack_len();
            if !old.is_null() {
                let _ = pyre_object::gc_roots::pin_root(old as PyObjectRef);
            }
            let cap = storage
                .len()
                .max(pyre_object::object_array::items_block_capacity(old));
            let fresh = pyre_object::object_array::alloc_instance_items_block(&storage, cap);
            let _ = pyre_object::gc_roots::pin_root(fresh as PyObjectRef);
            let fresh_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.storage = pyre_object::gc_roots::shadow_stack_get(fresh_slot) as *mut _;
            owner.map = map as usize;
            instance_write_barrier(owner as *mut Self as PyObjectRef);
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            mapdict_storage_write_barrier(owner.storage);
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            pyre_object::object_array::dealloc_instance_items_block(old);
        }
    }
    fn getdict(&self) -> PyObjectRef {
        // mapdict.py MapdictDictSupport.getdict → _obj_getdict
        // (mapdict.py). The instance header is at offset 0
        // (`#[repr(C)]`), so the carrier pointer is the instance PyObjectRef.
        _obj_getdict(self as *const Self as PyObjectRef)
    }
}

/// A live view of an object's `MapdictStorageMixin` slots. `map` is the
/// word at `(*ob_type).mapdict_offset` and `storage` is the next word
/// (`mapdict.py` `import_from_mixin`, `typedef.py` `_getusercls`).
#[derive(Clone, Copy)]
struct MapdictCarrier {
    obj: PyObjectRef,
}

unsafe fn mapdict_carrier(obj: PyObjectRef) -> MapdictCarrier {
    debug_assert!(unsafe { has_mapdict_layout(obj) });
    MapdictCarrier { obj }
}

impl MapdictCarrier {
    unsafe fn slots(&self) -> (*mut usize, *mut *mut pyre_object::object_array::ItemsBlock) {
        let obj = self.obj;
        let mapdict_offset = unsafe { (*(*obj).ob_type).mapdict_offset };
        let map = unsafe { (obj as *mut u8).add(mapdict_offset) as *mut usize };
        let storage = unsafe { map.add(1) as *mut *mut pyre_object::object_array::ItemsBlock };
        (map, storage)
    }

    unsafe fn storage(&self) -> *mut pyre_object::object_array::ItemsBlock {
        unsafe { *self.slots().1 }
    }

    unsafe fn set_storage(&mut self, storage: *mut pyre_object::object_array::ItemsBlock) {
        unsafe { *self.slots().1 = storage };
    }
}

impl MapdictObject for MapdictCarrier {
    fn _get_mapdict_map(&self) -> MapRef {
        unsafe { *self.slots().0 as MapRef }
    }

    fn _set_mapdict_map(&mut self, map: MapRef) {
        unsafe { *self.slots().0 = map as usize };
    }

    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        unsafe {
            let base = pyre_object::object_array::items_block_items_base(self.storage());
            *base.add(storageindex)
        }
    }

    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj, value]);
        let value_slot = self_slot + 1;
        instance_write_barrier(pyre_object::gc_roots::shadow_stack_get(self_slot));
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let carrier = mapdict_carrier(owner);
            mapdict_storage_write_barrier(carrier.storage());
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let carrier = mapdict_carrier(owner);
            let base = pyre_object::object_array::items_block_items_base(carrier.storage());
            *base.add(storageindex) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            self.obj = owner;
        }
    }

    fn _mapdict_storage_length(&self) -> usize {
        unsafe { (*self._get_mapdict_map()).storage_needed() }
    }

    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj]);
        let current_map = self._get_mapdict_map();
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) if !u.firstunwrapped => {
                    Some(((*current_map).as_plain().storageindex, u.listindex))
                }
                _ => None,
            }
        };
        match unboxed_slot {
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                let unboxed = erase_unboxed(&new_list);
                let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
                let mut carrier = unsafe { mapdict_carrier(owner) };
                carrier._mapdict_write_storage(storageindex, unboxed);
            }
            None => unsafe {
                let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
                let carrier = mapdict_carrier(owner);
                let storage = carrier.storage();
                let storage_needed = (*map).storage_needed();
                let cap = pyre_object::object_array::items_block_capacity(storage);
                if cap > storage_needed {
                    let base = pyre_object::object_array::items_block_items_base(storage);
                    for i in storage_needed..cap {
                        *base.add(i) = pyre_object::PY_NULL;
                    }
                }
            },
        }
        self.obj = pyre_object::gc_roots::shadow_stack_get(self_slot);
        self._set_mapdict_map(map);
    }

    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj, value]);
        let value_slot = self_slot + 1;
        let needed = unsafe { (*map).storage_needed() };
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let mut carrier = mapdict_carrier(owner);
            let old = carrier.storage();
            let old_slot = pyre_object::gc_roots::shadow_stack_len();
            if !old.is_null() {
                let _ = pyre_object::gc_roots::pin_root(old as PyObjectRef);
            }
            // `pin_root` is a safepoint. mapdict.py re-reads storage after an
            // allocating call; capacity and the in-place write use that block.
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            let old_cap = pyre_object::object_array::items_block_capacity(old);
            let block = if needed <= old_cap {
                old
            } else {
                let grown =
                    pyre_object::object_array::grow_instance_items_block(old, needed, old_cap);
                let _ = pyre_object::gc_roots::pin_root(grown as PyObjectRef);
                pyre_object::gc_roots::shadow_stack_get(
                    pyre_object::gc_roots::shadow_stack_len() - 1,
                ) as *mut pyre_object::object_array::ItemsBlock
            };
            let base = pyre_object::object_array::items_block_items_base(block);
            *base.add(needed - 1) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            carrier.set_storage(block);
            carrier._set_mapdict_map(map);
            instance_write_barrier(owner);
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            mapdict_storage_write_barrier(carrier.storage());
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            if !std::ptr::eq(block, old) {
                pyre_object::object_array::dealloc_instance_items_block(old);
            }
            self.obj = owner;
        }
    }

    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj]);
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let mut carrier = mapdict_carrier(owner);
            let old = carrier.storage();
            let old_slot = pyre_object::gc_roots::shadow_stack_len();
            if !old.is_null() {
                let _ = pyre_object::gc_roots::pin_root(old as PyObjectRef);
            }
            let cap = storage
                .len()
                .max(pyre_object::object_array::items_block_capacity(old));
            let fresh = pyre_object::object_array::alloc_instance_items_block(&storage, cap);
            let _ = pyre_object::gc_roots::pin_root(fresh as PyObjectRef);
            let fresh_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            carrier.set_storage(pyre_object::gc_roots::shadow_stack_get(fresh_slot) as *mut _);
            carrier._set_mapdict_map(map);
            instance_write_barrier(owner);
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            mapdict_storage_write_barrier(carrier.storage());
            let old = if old.is_null() {
                old
            } else {
                pyre_object::gc_roots::shadow_stack_get(old_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            pyre_object::object_array::dealloc_instance_items_block(old);
            self.obj = owner;
        }
    }

    fn getdict(&self) -> PyObjectRef {
        _obj_getdict(self.obj)
    }
}

/// mapdict.py `Object` — the transient `MapdictStorageMixin` carrier
/// for the result of `delete`/`copy` (its `storage`/`map` are transplanted
/// into the real instance by `_set_mapdict_storage_and_map`). pyre keeps that
/// lightweight owned-`Vec` role here, while a real `W_ObjectObject` provides
/// the fake-instance-dict-backing role.
pub(crate) struct Object {
    map: MapRef,
    storage: Vec<PyObjectRef>,
}

impl Object {
    /// mapdict.py `_mapdict_init_empty` — fresh carrier on `map` with
    /// an empty storage list.
    fn new_empty(map: MapRef) -> Object {
        Object {
            map,
            storage: Vec::new(),
        }
    }
}

impl MapdictObject for Object {
    fn _get_mapdict_map(&self) -> MapRef {
        self.map
    }
    fn _set_mapdict_map(&mut self, map: MapRef) {
        self.map = map;
    }
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        self.storage[storageindex]
    }
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        self.storage[storageindex] = value;
    }
    fn _mapdict_storage_length(&self) -> usize {
        unsafe { (*self.map).storage_needed() }
    }
    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        // mapdict.py. `current_map` is the PlainAttribute being popped;
        // `map` is its parent. The unboxed-non-firstunwrapped slot to shrink is
        // a `match` on `firstunwrapped` (not `!firstunwrapped`) so the source
        // walker can lower it.
        let current_map = self.map;
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) => match u.firstunwrapped {
                    true => None,
                    false => Some(((*current_map).as_plain().storageindex, u.listindex)),
                },
                None => None,
            }
        };
        match unboxed_slot {
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list: Vec<i64> = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                self._mapdict_write_storage(storageindex, erase_unboxed(&new_list));
            }
            None => {
                let storage_needed = unsafe { (*map).storage_needed() };
                self.storage.truncate(storage_needed);
            }
        }
        self.map = map;
    }
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        // grow storage by one, append value (mapdict.py _set_mapdict_increase_storage1)
        self.storage.push(value);
        self.map = map;
    }
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        self.storage = storage;
        self.map = map;
    }
    fn getdict(&self) -> PyObjectRef {
        // mapdict.py gives `Object` both transient-copy and fake-dict
        // roles. pyre gives the fake-dict role to `W_ObjectObject`, whose
        // getdict is implemented, leaving this carrier transient only. The
        // terminator_read Devolved arm, write_terminator's Devolved and LIMIT
        // arms, and node_delete's Devolved arm must therefore never reach it.
        unimplemented!("transient Object carrier has no __dict__")
    }
}

/// rerased `erase_unboxed` / `unerase_unboxed` (mapdict.py:38-41) — the
/// unboxed longlong list lives in an otherwise-`PyObjectRef` storage slot.
///
/// The list is a varsize leaf GcArray (`gc_int_array_gc_type_id()`), so the slot
/// holds an ordinary GC reference like every other storage slot: the collector
/// marks it through the instance and reclaims it when the instance dies, and no
/// walker has to be told to skip the slot. Every mutation allocates a fresh
/// exact-size block, so the block's capacity header is the list's length.
///
/// The block is reachable from nothing when this returns; the caller must root
/// it until the storage store lands (see `alloc_typed_items_block`).
fn erase_unboxed(values: &[i64]) -> PyObjectRef {
    debug_assert!(
        !values.is_empty(),
        "an unboxed list holds at least the attribute that created it"
    );
    unsafe {
        let block = pyre_object::object_array::alloc_typed_items_block(
            values.len(),
            pyre_object::object_array::gc_int_array_gc_type_id(),
        );
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            pyre_object::object_array::typed_items_block_items_base(block) as *mut i64,
            values.len(),
        );
        block as PyObjectRef
    }
}

/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unerase_unboxed(slot: PyObjectRef) -> *mut pyre_object::object_array::TypedItemsBlock {
    slot as *mut pyre_object::object_array::TypedItemsBlock
}

/// The longlong list an unboxed slot holds. Its length is the block's capacity
/// header, because the block is always allocated exact-size.
///
/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unboxed_items<'a>(slot: PyObjectRef) -> &'a [i64] {
    unsafe {
        let block = unerase_unboxed(slot);
        std::slice::from_raw_parts(
            pyre_object::object_array::typed_items_block_items_base(block) as *const i64,
            pyre_object::object_array::typed_items_block_capacity(block),
        )
    }
}

/// Mutable twin of [`unboxed_items`], for the same-type in-place update.
///
/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unboxed_items_mut<'a>(slot: PyObjectRef) -> &'a mut [i64] {
    unsafe {
        let block = unerase_unboxed(slot);
        std::slice::from_raw_parts_mut(
            pyre_object::object_array::typed_items_block_items_base(block) as *mut i64,
            pyre_object::object_array::typed_items_block_capacity(block),
        )
    }
}

/// mapdict.py `UnboxedPlainAttribute._unbox`.
///
/// # Safety
/// `w_value` must be of the type named by `typ`.
unsafe fn unbox_value(typ: UnboxType, w_value: PyObjectRef) -> i64 {
    match typ {
        UnboxType::Int => unsafe { pyre_object::w_int_get_value(w_value) },
        // float2longlong (mapdict.py:577).
        UnboxType::Float => unsafe { pyre_object::w_float_get_value(w_value) }.to_bits() as i64,
    }
}

/// mapdict.py `UnboxedPlainAttribute._box`.
fn box_value(typ: UnboxType, val: i64) -> PyObjectRef {
    match typ {
        UnboxType::Int => pyre_object::w_int_new(val),
        // longlong2float (mapdict.py:584).
        UnboxType::Float => pyre_object::w_float_new(f64::from_bits(val as u64)),
    }
}

/// `type(w_value) is space.IntObjectCls` (mapdict.py).
///
/// Pyre gives an int subclass the same native storage layout as W_IntObject
/// and carries its app-level identity in `w_class`. Testing only the layout
/// (`is_int`) would unbox IntEnum/user-int instances and reconstruct them as
/// builtin ints, losing `w_class`. Compare the exact app-level type just as
/// PyPy compares the exact implementation class.
///
/// # Safety
/// `w_value` must point to a live object.
pub unsafe fn is_unboxable_int(w_value: PyObjectRef) -> bool {
    if unsafe { pyre_object::is_bool(w_value) } || !unsafe { pyre_object::is_int(w_value) } {
        return false;
    }
    let exact = crate::typedef::gettypeobject(&pyre_object::INT_TYPE);
    if exact.is_null() {
        // Interpreter-only unit tests may exercise mapdict before the app-level
        // type registry is initialized. A native int with no class stamp is
        // the bootstrap representation of that same exact type.
        if pyre_object::tagged_int::CAN_BE_TAGGED
            && unsafe { pyre_object::tagged_int::is_tagged_int(w_value) }
        {
            return true;
        }
        return unsafe { (*w_value).w_class.is_null() };
    }
    let Some(actual) = crate::typedef::r#type(w_value) else {
        return false;
    };
    std::ptr::eq(actual.as_ptr(), exact)
}

/// Float half of `_pick_unbox_type`: PyPy uses
/// `type(w_value) is space.FloatObjectCls`, so a float subclass must retain
/// its boxed object and `w_class` too.
///
/// NaNs also stay boxed: raw-f64 storage reboxes on read and would lose their
/// CPython 3.14 pointer identity. Shared with the JIT STORE_ATTR fold and used
/// for both new and existing unboxed slots.
///
/// # Safety
/// `w_value` must point to a live object.
pub unsafe fn is_unboxable_float(w_value: PyObjectRef) -> bool {
    if !unsafe { pyre_object::is_float(w_value) }
        || unsafe { pyre_object::w_float_get_value(w_value) }.is_nan()
    {
        return false;
    }
    let exact = crate::typedef::gettypeobject(&pyre_object::FLOAT_TYPE);
    if exact.is_null() {
        return unsafe { (*w_value).w_class.is_null() };
    }
    let Some(actual) = crate::typedef::r#type(w_value) else {
        return false;
    };
    std::ptr::eq(actual.as_ptr(), exact)
}

/// mapdict.py `UnboxedPlainAttribute._convert_to_boxed` — rebuild the
/// carrier with boxed storage and transplant it onto `obj`, returning the new
/// (boxed) map. `node_copy` re-adds every attribute through `add_attr`, which
/// picks no unbox type because `allow_unboxing` is already frozen off on the
/// terminator (the caller sets it before converting), so the rebuilt chain is
/// all boxed and its storage is a clean `Vec<PyObjectRef>`.
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed<O: MapdictObject>(obj: &mut O) -> MapRef {
    let map = obj._get_mapdict_map();
    let new_obj = unsafe { node_copy(map, obj) };
    let new_map = new_obj.map;
    obj._set_mapdict_storage_and_map(new_obj.storage, new_map);
    new_map
}

/// mapdict.py `UnboxedPlainAttribute._direct_write` type-change tail —
/// convert `obj` to boxed storage, then write `(name, attrkind) = w_value`
/// through the now-boxed map (no `UnboxedPlainAttribute` remains because
/// `allow_unboxing` was just frozen off).
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed_and_write<O: MapdictObject>(
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) {
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_value);
    let map = unsafe { convert_to_boxed(obj) };
    let _ = unsafe {
        node_write(
            map,
            obj,
            name,
            attrkind,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )
    };
}

/// `type(w_value) is self.typ` (mapdict.py:574,615).
///
/// # Safety
/// `w_value` must point to a live object.
unsafe fn value_has_unbox_type(typ: UnboxType, w_value: PyObjectRef) -> bool {
    match typ {
        UnboxType::Int => unsafe { is_unboxable_int(w_value) },
        UnboxType::Float => unsafe { is_unboxable_float(w_value) },
    }
}

/// mapdict.py `PlainAttribute._direct_read` / `_prim_direct_read` /
/// `_pure_direct_read` have identical bodies; the `@jit.elidable`
/// `_pure_direct_read` variant is applied when the read is JIT-wired.
/// `unerase_item` is identity. For an `UnboxedPlainAttribute`,
/// mapdict.py gives `_direct_read` the allow-unboxing conversion tail;
/// pyre places that tail in `direct_read`.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_read<O: MapdictObject>(attr: MapRef, obj: &O) -> PyObjectRef {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        // mapdict.py _pure_direct_read — boxed value straight out of the slot.
        None => obj._mapdict_read_storage(p.storageindex),
        Some(u) => {
            // _prim_direct_read (mapdict.py): box the longlong at
            // (storageindex, listindex).
            let slot = obj._mapdict_read_storage(p.storageindex);
            let raw = unsafe { unboxed_items(slot)[u.listindex] };
            let w_res = box_value(u.typ, raw);
            // This is `_prim_direct_read` (mapdict.py), the non-converting
            // read shared by node_read / copy_attr / reorder_and_add /
            // materialize. `_direct_read`'s lazy migrate-to-boxed side effect
            // (mapdict.py, when the class has frozen unboxing) lives at
            // the getattr boundary in `direct_read`, which `&mut` access permits
            // there; here `&obj` stays pure.
            w_res
        }
    }
}

/// mapdict.py `PlainAttribute._direct_write`. `erase_item` is identity.
/// For an `UnboxedPlainAttribute` this is mapdict.py.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_write<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            let storageindex = p.storageindex;
            obj._mapdict_write_storage(storageindex, w_value);
        }
        Some(u) => {
            if unsafe { value_has_unbox_type(u.typ, w_value) } {
                // mapdict.py:615-619 — same type: update the longlong in place.
                let val = unsafe { unbox_value(u.typ, w_value) };
                let slot = obj._mapdict_read_storage(p.storageindex);
                unsafe {
                    unboxed_items_mut(slot)[u.listindex] = val;
                }
            } else {
                // mapdict.py:620-627 — type change. Freeze unboxing for the
                // terminator, then convert the instance to boxed storage and
                // rewrite `(name, attrkind)` through the now-boxed map.
                unsafe { (*p.terminator).as_terminator() }.set_allow_unboxing(false);
                let name = p.name.clone();
                let attrkind = p.attrkind;
                unsafe { convert_to_boxed_and_write(obj, &name, attrkind, w_value) };
            }
        }
    }
}

/// mapdict.py `Terminator._read_terminator` and its
/// `DevolvedDictTerminator` override (mapdict.py). Returns the value or
/// `None` when the attribute is absent.
///
/// # Safety
/// `term` must point to a live Terminator map node.
#[allow(dead_code)]
unsafe fn terminator_read<O: MapdictObject>(
    term: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<PyObjectRef> {
    unsafe { terminator_read_checked(term, obj, name, attrkind) }.unwrap_or(None)
}

/// Fallible [`terminator_read`].  The devolved arm is the only one that can
/// raise, and `space.finditem_str` propagates there upstream; the swallowing
/// spelling above exists for the callers that have no error channel and is
/// written in terms of this one.  Its `unwrap_or` also consumes the pending
/// error slot, so a dropped error cannot surface on a later operation.
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn terminator_read_checked<O: MapdictObject>(
    term: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Result<Option<PyObjectRef>, PyError> {
    let t = unsafe { (*term).as_terminator() };
    match t.kind {
        TerminatorKind::Devolved if attrkind == DICT => {
            // mapdict.py: the devolved terminator reads DICT attributes
            // from the materialised instance dict (`space.finditem_str(
            // obj.getdict(space), name)`).  `finditem_str` is fallible: the
            // probe compares against whatever the bucket holds, so a stored
            // non-string key whose hash collides can reach a user `__eq__`
            // that raises, and that must not read back as a miss.
            let w_dict = obj.getdict();
            let backing = crate::type_methods::resolve_dict_backing(w_dict);
            unsafe { pyre_object::dictmultiobject::w_dict_getitem_wtf8_checked(backing, name) }
                .map_err(|_| {
                    crate::baseobjspace::take_pending_dict_key_error(
                        pyre_object::w_str_from_wtf8_managed(name.to_wtf8_buf()),
                    )
                })
        }
        // Terminator / DictTerminator / NoDictTerminator read nothing.
        _ => Ok(None),
    }
}

/// mapdict.py `AbstractAttribute.read`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<PyObjectRef> {
    unsafe { node_read_checked(self_node, obj, name, attrkind) }.unwrap_or(None)
}

/// [`node_read_checked`] plus `_direct_read`'s migration tail
/// (mapdict.py), both taken off the node this resolves once — the read
/// the getattr and slot boundaries owe.
///
/// Splitting them costs a second `find_map_attr` for the same
/// `(map, name, attrkind)`, and `find_map_attr`'s cache probe holds a
/// process-wide lock, so the re-lookup is not the free cache hit it reads as.
///
/// A raising terminator probe unwinds before the tail, which is also what
/// upstream does (`:846-847` → `:58` → `:312-313` returns on the miss path and
/// never reaches `_direct_read`).
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read_converting<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
) -> Result<Option<PyObjectRef>, PyError> {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        Some(attr) => Ok(Some(unsafe { direct_read(attr, obj) })),
        None => unsafe { terminator_read_checked((*self_node).terminator(), obj, name, attrkind) },
    }
}

/// Fallible [`node_read`], for the callers that can propagate the raising
/// `__eq__` a devolved terminator's dict probe may reach.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read_checked<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Result<Option<PyObjectRef>, PyError> {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        // The `jit.isconstant(attr) and jit.isconstant(obj) and not
        // attr.ever_mutated` guard selects `_pure_direct_read`
        // (mapdict.py). The PlainAttribute variants have the same body;
        // UnboxedPlainAttribute._direct_read's conversion tail lives in
        // `direct_read`.
        Some(attr) => Ok(Some(unsafe { plain_direct_read(attr, obj) })),
        None => unsafe { terminator_read_checked((*self_node).terminator(), obj, name, attrkind) },
    }
}

// ── copy / delete path (mapdict.py, 433-435, 461-475) ─────────

/// mapdict.py `PlainAttribute._copy_attr` — read this attribute from
/// `obj` and re-add it to the freshly built `new_obj`. The read is
/// `_prim_direct_read` (mapdict.py/600-601, the non-converting raw read);
/// pyre's `plain_direct_read` performs exactly that — boxing the longlong slot
/// when the attribute is unboxed — and defers only the read-path lazy re-box
/// (`_direct_read`'s allow-unboxing/convert distinction, see
/// `plain_direct_read`), which is invisible to the copied value.
///
/// # Safety
/// `attr_node` must point to a live `PlainAttribute`; `obj` to a live carrier.
unsafe fn copy_attr<O: MapdictObject>(attr_node: MapRef, obj: &O, new_obj: &mut Object) {
    // The carrier's storage is a plain `Vec` no collector updates, and a value
    // the read had to box is named by nothing else until `add_attr` stores it
    // — old-gen birth buys it immobility rather than survival, so the
    // re-lay-out below can sweep it. Publish both, copy the storage run back
    // once the read is done, and only then hand `add_attr` a storage it is
    // free to re-lay-out.
    let _roots = pyre_object::gc_roots::push_roots();
    let storage_slot = pyre_object::gc_roots::pin_roots(&new_obj.storage);
    let value_slot =
        pyre_object::gc_roots::pin_roots(&[unsafe { plain_direct_read(attr_node, obj) }]);
    for (i, slot) in new_obj.storage.iter_mut().enumerate() {
        *slot = pyre_object::gc_roots::shadow_stack_get(storage_slot + i);
    }
    let p = unsafe { (*attr_node).as_plain() };
    let map = new_obj._get_mapdict_map();
    unsafe {
        add_attr(
            map,
            new_obj,
            &p.name,
            p.attrkind,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )
    };
}

/// mapdict.py `Terminator.copy` / 472-475 `PlainAttribute.copy` — build
/// a fresh `Object` carrier holding the same attributes as `obj` (in canonical
/// order: the back-chain is copied bottom-up, then the node re-adds itself).
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_copy<O: MapdictObject>(self_node: MapRef, obj: &O) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let back = unsafe { (*self_node).as_plain() }.back;
        let mut new_obj = unsafe { node_copy(back, obj) };
        unsafe { copy_attr(self_node, obj, &mut new_obj) };
        new_obj
    } else {
        // Terminator.copy (mapdict.py): empty carrier on this terminator.
        Object::new_empty(self_node)
    }
}

/// mapdict.py `Terminator.set_terminator`, 483-486
/// `PlainAttribute.set_terminator`, 414-418 `DevolvedDictTerminator.set_terminator`
/// — rebuild a fresh `Object` carrier holding `obj`'s attributes (canonical
/// order, like `node_copy`) but rooted at `new_terminator`. A devolved root
/// re-roots onto `new_terminator`'s paired devolved terminator so the instance
/// stays devolved.
///
/// # Safety
/// `self_node`/its `back` chain and `new_terminator` must point to live map nodes.
unsafe fn node_set_terminator<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    new_terminator: MapRef,
) -> Object {
    if unsafe { (*self_node).is_plain() } {
        // mapdict.py set_terminator — recurse into `back` with the new terminator, then
        // re-add this attribute.
        let back = unsafe { (*self_node).as_plain() }.back;
        let mut new_obj = unsafe { node_set_terminator(back, obj, new_terminator) };
        unsafe { copy_attr(self_node, obj, &mut new_obj) };
        new_obj
    } else {
        // mapdict.py set_terminator — empty carrier on `new_terminator`; the devolved
        // override (mapdict.py set_terminator) re-targets a devolved root onto the new
        // terminator's devolved pair.
        let term = match unsafe { (*self_node).as_terminator() }.kind {
            TerminatorKind::Devolved => {
                let target = unsafe { (*new_terminator).as_terminator() };
                match target.kind {
                    TerminatorKind::Devolved => new_terminator,
                    _ => target.devolved_dict_terminator.get(),
                }
            }
            _ => new_terminator,
        };
        Object::new_empty(term)
    }
}

/// mapdict.py `AbstractAttribute.delete` (Terminator/DictTerminator,
/// returns `None`) and 461-470 `PlainAttribute.delete`. Returns the rebuilt
/// carrier with `(name, attrkind)` removed, or `None` if the attribute is
/// absent.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_delete<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<Object> {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        if attrkind == p.attrkind && &*p.name == name {
            // mapdict.py:462-466 — attribute found; drop it by rebuilding from
            // `back` (which excludes this node).
            p.set_ever_mutated(true);
            return Some(unsafe { node_copy(p.back, obj) });
        }
        // mapdict.py:467-470 — recurse, then re-add this surviving attribute.
        let back = p.back;
        match unsafe { node_delete(back, obj, name, attrkind) } {
            Some(mut new_obj) => {
                unsafe { copy_attr(self_node, obj, &mut new_obj) };
                Some(new_obj)
            }
            None => None,
        }
    } else {
        // mapdict.py Terminator.delete (DictTerminator/NoDictTerminator
        // inherit) returns None.
        let kind = unsafe { (*self_node).as_terminator() }.kind;
        match kind {
            // mapdict.py DevolvedDictTerminator.delete: drop the DICT
            // attribute from the materialised instance dict (a miss is tolerated
            // — mapdict.py swallows KeyError), then return an empty
            // carrier on this terminator (`Terminator.copy(self, obj)`).
            TerminatorKind::Devolved if attrkind == DICT => {
                // The backing dict moves and the key is allocated after it has
                // been resolved, so it is published and read back at the delete.
                let _roots = pyre_object::gc_roots::push_roots();
                let w_dict = obj.getdict();
                let backing_slot =
                    pyre_object::gc_roots::pin_roots(&[crate::type_methods::resolve_dict_backing(
                        w_dict,
                    )]);
                let w_key = pyre_object::w_str_from_wtf8_managed(name.to_owned());
                unsafe {
                    pyre_object::w_dict_delitem(
                        pyre_object::gc_roots::shadow_stack_get(backing_slot),
                        w_key,
                    )
                };
                Some(unsafe { node_copy(self_node, obj) })
            }
            _ => None,
        }
    }
}

/// mapdict.py `Terminator.remove_dict_entries` (= `self.copy(obj)`) and
/// :511-515 `PlainAttribute.remove_dict_entries` — rebuild a fresh carrier that
/// keeps every non-`DICT` attribute and drops the `DICT` ones. Used by
/// `MapDictStrategy.clear` (mapdict.py). Reuses the `node_copy` /
/// `copy_attr` machinery already built for `delete`.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_remove_dict_entries<O: MapdictObject>(self_node: MapRef, obj: &O) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        let back = p.back;
        // mapdict.py:512 — recurse into `back` first.
        let mut new_obj = unsafe { node_remove_dict_entries(back, obj) };
        // mapdict.py — re-add this attribute unless it is a DICT entry.
        if p.attrkind != DICT {
            unsafe { copy_attr(self_node, obj, &mut new_obj) };
        }
        new_obj
    } else {
        // mapdict.py Terminator.remove_dict_entries = self.copy(obj).
        unsafe { node_copy(self_node, obj) }
    }
}

/// mapdict.py `DictTerminator.materialize_r_dict`/`materialize_str_dict`
/// + 493-509 `PlainAttribute.materialize_r_dict`/`materialize_str_dict`. Drain
/// the DICT attributes into `w_dict` (already switched to its real strategy) and
/// rebuild a fresh carrier keeping only the non-DICT attributes, rooted at the
/// paired `DevolvedDictTerminator`. The walk recurses into `back` first so DICT
/// entries land in insertion (oldest-first) order.
///
/// pyre folds `materialize_r_dict` and `materialize_str_dict` into one helper:
/// PyPy's two methods differ only in the dict they fill (an `r_dict` keyed by
/// `space.eq_w`/`hash_w` vs a `str_dict` keyed by unicode), and both insert via
/// `space.newtext(name)`; here both targets are a `W_DictObject` whose strategy
/// (Object or Unicode) is already installed. Attribute names are interned at
/// the object-space boundary, so materialisation uses `box_str_constant(name)`
/// before the single `w_dict_store` shared by both strategies.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes; `w_dict` must
/// be a live `W_DictObject` on its post-switch strategy.
unsafe fn node_materialize_dict<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    w_dict: PyObjectRef,
) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        // The dict moves and both steps before this node's store allocate: the
        // recursion fills in every node closer to the terminator, and boxing
        // this node's name runs once its value has already been read. Publish
        // the dict and the value and read both back at the store.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict]);
        // mapdict.py:494/503 — recurse into `back` first.
        let mut new_obj = unsafe {
            node_materialize_dict(
                p.back,
                obj,
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
            )
        };
        if p.attrkind == DICT {
            // mapdict.py/504-506 — move the DICT attribute into the
            // materialised dict (`dict_w[space.newtext(name)] =
            // self._prim_direct_read(obj)`). `plain_direct_read` performs that
            // prim read, boxing the slot when the attribute is unboxed.
            //
            // The carrier the recursion returned holds every lower node's value
            // in a plain `Vec` the collector does not update, and nothing
            // touches it here, so the run copies back index for index.
            let storage_slot = pyre_object::gc_roots::pin_roots(&new_obj.storage);
            let value_slot =
                pyre_object::gc_roots::pin_roots(&[unsafe { plain_direct_read(self_node, obj) }]);
            let w_attr = pyre_object::unicodeobject::box_str_constant(&p.name);
            unsafe {
                pyre_object::w_dict_store(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    w_attr,
                    pyre_object::gc_roots::shadow_stack_get(value_slot),
                )
            };
            for (i, slot) in new_obj.storage.iter_mut().enumerate() {
                *slot = pyre_object::gc_roots::shadow_stack_get(storage_slot + i);
            }
        } else {
            // mapdict.py/508 — keep the non-DICT attribute on the carrier.
            unsafe { copy_attr(self_node, obj, &mut new_obj) };
        }
        new_obj
    } else {
        // mapdict.py DictTerminator.materialize_* → `_make_devolved`: an
        // empty carrier on the paired DevolvedDictTerminator.
        let t = unsafe { (*self_node).as_terminator() };
        match t.kind {
            TerminatorKind::Dict => {
                let devolved = t.devolved_dict_terminator.get();
                Object::new_empty(devolved)
            }
            // mapdict.py materialize_r_dict — the abstract base raises; materialise only
            // ever runs on a not-yet-devolved DictTerminator-rooted instance
            // dict (NoDict has no __dict__; Devolved is already materialised).
            _ => unimplemented!(
                "materialize on non-DictTerminator (mapdict.py:259-263 abstract base)"
            ),
        }
    }
}

/// mapdict.py `materialize_r_dict` / 1310-1313 `materialize_str_dict`
/// (module-level) — run the chain over `obj`'s map to fill `w_dict`, then
/// transplant the rebuilt (devolved) storage+map back onto `obj`. The backing
/// instance is always a `W_ObjectObject` (the only `MapdictDictSupport`
/// carrier whose `__dict__` adopts `MapDictStrategy`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`; `w_dict` a live `W_DictObject` on
/// its post-switch strategy.
unsafe fn materialize_dict(obj: PyObjectRef, w_dict: PyObjectRef) {
    // Filling `w_dict` allocates — boxed attribute names, the dict stores, and
    // the rebuilt carrier's own transitions — so both operands are published
    // and the transplant writes through the instance's post-collection address.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj, w_dict]);
    let dict_slot = obj_slot + 1;
    let new_obj = unsafe {
        let inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        let map = inst._get_mapdict_map();
        node_materialize_dict(
            map,
            &inst,
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        )
    };
    let mut inst = unsafe { mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot)) };
    inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
}

// ── write path (mapdict.py, 312-321, 668-691) ──────────────────

/// mapdict.py `CachedAttributeHolder` — caches the child map produced
/// by adding `(name, attrkind)` to a parent map, so transitions are shared.
/// Interned/immortal like the map nodes it holds.
pub struct CachedAttributeHolder {
    /// mapdict.py:670 `order` (= number of prior children of `back`).
    pub order: usize,
    /// mapdict.py:669 `_immutable_fields_` declares `attr?`; mapdict.py:675
    /// `attr` is the cached child map. [`Self::set_attr`] invalidates watchers
    /// before changing it.
    pub attr: Cell<MapRef>,
    /// The hidden watcher field implementing mapdict.py:669
    /// `_immutable_fields_ = ['attr?', 'typ?']` for `attr?`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub attr_watchers: QuasiImmutField,
    /// mapdict.py:669 `_immutable_fields_` declares `typ?`; mapdict.py:676
    /// `typ` is the unbox type (`None` means boxed). [`Self::set_typ`]
    /// invalidates watchers before changing it.
    pub typ: Cell<Option<UnboxType>>,
    /// The hidden watcher field implementing mapdict.py:669
    /// `_immutable_fields_ = ['attr?', 'typ?']` for `typ?`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub typ_watchers: QuasiImmutField,
}

impl CachedAttributeHolder {
    /// Change mapdict.py:675 `attr` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_attr(&self, a: MapRef) {
        if self.attr.get() == a {
            return;
        }
        if self.attr_watchers.is_installed() {
            unsafe { pyre_object::quasiimmut::sweep_quasi_immut_field(&self.attr_watchers) };
        }
        self.attr.set(a);
    }

    /// Change mapdict.py:676 `typ` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_typ(&self, t: Option<UnboxType>) {
        if self.typ.get() == t {
            return;
        }
        if self.typ_watchers.is_installed() {
            unsafe { pyre_object::quasiimmut::sweep_quasi_immut_field(&self.typ_watchers) };
        }
        self.typ.set(t);
    }

    pub fn current_attr_qmut(&self) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.attr_watchers.get_current_qmut_instance()
    }

    pub fn attr_qmut_installed(&self) -> bool {
        self.attr_watchers.is_installed()
    }

    pub fn force_attr_qmut(&self) {
        self.attr_watchers.invalidate();
    }

    pub fn current_typ_qmut(&self) -> std::sync::Arc<pyre_object::quasiimmut::QuasiImmut> {
        self.typ_watchers.get_current_qmut_instance()
    }

    pub fn typ_qmut_installed(&self) -> bool {
        self.typ_watchers.is_installed()
    }

    pub fn force_typ_qmut(&self) {
        self.typ_watchers.invalidate();
    }
}

/// `quasiimmut.py get_current_qmut_instance` for a CachedAttributeHolder's `attr?` field.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_current_attr_qmut(
    holder: *const CachedAttributeHolder,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if holder.is_null() {
        return None;
    }
    Some(unsafe { (*holder).current_attr_qmut() })
}

/// Whether a CachedAttributeHolder's `attr?` watcher is installed.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_attr_qmut_installed(holder: *const CachedAttributeHolder) -> bool {
    !holder.is_null() && unsafe { (*holder).attr_qmut_installed() }
}

/// Force a CachedAttributeHolder's `attr?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_force_attr_qmut(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).force_attr_qmut() };
}

/// `quasiimmut.py get_current_qmut_instance` for a CachedAttributeHolder's `typ?` field.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_current_typ_qmut(
    holder: *const CachedAttributeHolder,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if holder.is_null() {
        return None;
    }
    Some(unsafe { (*holder).current_typ_qmut() })
}

/// Whether a CachedAttributeHolder's `typ?` watcher is installed.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_typ_qmut_installed(holder: *const CachedAttributeHolder) -> bool {
    !holder.is_null() && unsafe { (*holder).typ_qmut_installed() }
}

/// Force a CachedAttributeHolder's `typ?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_force_typ_qmut(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).force_typ_qmut() };
}

/// mapdict.py `CachedAttributeHolder.__init__`.
///
/// # Safety
/// `back` must point to a live map node.
unsafe fn new_cached_attribute_holder(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    unbox_type: Option<UnboxType>,
    order: usize,
) -> *const CachedAttributeHolder {
    let attr = match unbox_type {
        None => unsafe { new_plain_attribute(name, attrkind, back, order) },
        Some(typ) => unsafe { new_unboxed_plain_attribute(name, attrkind, back, order, typ) },
    };
    Box::into_raw(Box::new(CachedAttributeHolder {
        order,
        attr: Cell::new(attr),
        attr_watchers: QuasiImmutField::new(),
        typ: Cell::new(unbox_type),
        typ_watchers: QuasiImmutField::new(),
    }))
}

/// mapdict.py `CachedAttributeHolder.pick_attr`.
///
/// # Safety
/// `holder` must point to a live `CachedAttributeHolder`.
unsafe fn holder_pick_attr(
    holder: *const CachedAttributeHolder,
    unbox_type: Option<UnboxType>,
) -> MapRef {
    let h = unsafe { &*holder };
    let typ = h.typ.get();
    if typ.is_none() || typ == unbox_type {
        return h.attr.get();
    }
    // The cached attribute was unboxed but the new value has a different type;
    // invalidate unboxing for this terminator and re-box (mapdict.py:682-690).
    h.set_typ(None);
    let attr = h.attr.get();
    let p = unsafe { (*attr).as_plain() };
    unsafe { (*p.terminator).as_terminator() }.set_allow_unboxing(false);
    let new_attr = unsafe { new_plain_attribute(p.name.clone(), p.attrkind, p.back, h.order) };
    h.set_attr(new_attr);
    new_attr
}

/// mapdict.py `AbstractAttribute._get_new_attr`.
///
/// # Safety
/// `self_node` must point to a live map node.
unsafe fn get_new_attr(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> *const CachedAttributeHolder {
    let key = (name.to_owned(), attrkind);
    // PyPy mutates this ordinary per-node dict while holding the GIL
    // (`_get_new_attr`). Map nodes are process-global in pyre, so keep the
    // lookup, holder construction, and insertion under one lock: parallel
    // interpreters must intern exactly one transition for a given key.
    // Intern `{}` here, the only site `_get_new_attr` allocates.
    let cache = match unsafe { &*self_node } {
        MapNode::Terminator(t) => intern_cache_attrs(&t.cache_attrs),
        MapNode::Plain(p) => intern_cache_attrs(&p.cache_attrs),
    };
    let mut cache = cache.lock();
    if let Some(&holder) = cache.get(&key) {
        return holder;
    }
    let order = cache.len();
    let holder = unsafe {
        new_cached_attribute_holder(name.to_owned(), attrkind, self_node, unbox_type, order)
    };
    cache.insert(key, holder);
    holder
}

/// mapdict.py `AbstractAttribute._find_branch_to_move_into`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
unsafe fn find_branch_to_move_into(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> (usize, *const CachedAttributeHolder) {
    let mut current_order = usize::MAX; // sys.maxint
    let mut number_to_readd = 0usize;
    let mut current = self_node;
    let key = (name.to_owned(), attrkind);
    loop {
        // `_find_branch_to_move_into`: `if current.cache_attrs is not None`.
        let holder =
            unsafe { (*current).cache_attrs() }.and_then(|cache| cache.lock().get(&key).copied());
        let reached_top = match holder {
            None => true,
            Some(h) => (unsafe { (*h).order }) > current_order,
        };
        if reached_top {
            // didn't find it anywhere yet; if we reached a non-PlainAttribute
            // (the terminator), just add it at the top attribute
            if unsafe { (*current).is_plain() } {
                // keep walking up
            } else {
                return (0, unsafe {
                    get_new_attr(self_node, name, attrkind, unbox_type)
                });
            }
        } else {
            return (number_to_readd, holder.unwrap());
        }
        // not found here, try the parent
        number_to_readd += 1;
        let p = unsafe { (*current).as_plain() };
        current_order = p.order;
        current = p.back;
    }
}

/// Side-effect-free twin of mapdict.py:170-193
/// `AbstractAttribute._find_branch_to_move_into` for the tracer's
/// `opimpl_jit_force_quasi_immutable` predicate.
///
/// The ordinary resolver interns a missing transition through `get_new_attr`.
/// This resolver instead returns `None` at exactly that point, making the
/// caller decline rather than changing later `CachedAttributeHolder.order`
/// assignments merely because a trace was attempted. Each `cache_attrs` lock
/// is released before the chain walk continues; no `QuasiImmutField` lock may
/// be taken while a `cache_attrs` mutex or an `INSTANCE_LOCKS` stripe is held.
/// In particular, moving the force check inside `add_attr` would recursively
/// take the non-reentrant `cache_attrs` mutex and deadlock.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
unsafe fn find_branch_to_move_into_readonly(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    _unbox_type: Option<UnboxType>,
) -> Option<(usize, *const CachedAttributeHolder)> {
    let mut current_order = usize::MAX; // sys.maxint
    let mut number_to_readd = 0usize;
    let mut current = self_node;
    loop {
        let holder = unsafe { (*current).cache_attrs() }.and_then(|cache| {
            cache
                .lock()
                .iter()
                .find(|((cached_name, cached_kind), _)| {
                    cached_kind == &attrkind && &**cached_name == name
                })
                .map(|(_, &holder)| holder)
        });
        let reached_top = match holder {
            None => true,
            Some(h) => (unsafe { (*h).order }) > current_order,
        };
        if reached_top {
            if !unsafe { (*current).is_plain() } {
                // `find_branch_to_move_into` would call `get_new_attr` here.
                return None;
            }
        } else {
            return Some((number_to_readd, holder.unwrap()));
        }
        number_to_readd += 1;
        let p = unsafe { (*current).as_plain() };
        current_order = p.order;
        current = p.back;
    }
}

/// The first mapdict quasi-immutable field a write would change.
#[derive(Clone, Copy)]
pub enum MapdictQmutTarget {
    PlainEverMutated(*const PlainAttribute),
    TerminatorAllowUnboxing(*const Terminator),
    HolderTyp(*const CachedAttributeHolder),
    HolderAttr(*const CachedAttributeHolder),
}

/// Resolve STORE_ATTR's mapdict.py:1618-1627 attribute classification without
/// applying the write. `bool` selects the `"slot"` attrname.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn classify_mapdict_write_attr(w_obj: PyObjectRef, name: &str) -> Option<(u16, bool)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // The raw entry, as `STORE_ATTR_slowpath` reads it, so `classify_attr`
    // gives up on a MutableCell instead of classifying what it holds.
    let w_descr = unsafe { crate::baseobjspace::type_attr_stored(w_type, Wtf8::new(name)) };
    Some(unsafe { classify_attr(w_type, w_descr, true) })
}

/// Return the first `?` field mapdict.py `AbstractAttribute.write` would
/// flip, without performing the write or interning a transition.
///
/// Watcher installation belongs to the tracer. This predicate never reads an
/// `*_qmut_installed` flag and never holds a `cache_attrs` mutex or an
/// `INSTANCE_LOCKS` stripe while a `QuasiImmutField` lock can be taken.
///
/// # Safety
/// `w_obj` and `w_value` must be live objects.
pub unsafe fn setattr_would_force_quasi_immut(
    w_obj: PyObjectRef,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> Option<MapdictQmutTarget> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    if let Some(attr) = unsafe { find_map_attr(map, name, attrkind) } {
        let p = unsafe { (*attr).as_plain() };
        if !p.ever_mutated.get() {
            return Some(MapdictQmutTarget::PlainEverMutated(p));
        }
        if let Some(unboxed) = &p.unboxed {
            let term = unsafe { (*p.terminator).as_terminator() };
            if !unsafe { value_has_unbox_type(unboxed.typ, w_value) } && term.allow_unboxing.get() {
                return Some(MapdictQmutTarget::TerminatorAllowUnboxing(term));
            }
        }
        return None;
    }

    let term = unsafe { (*(*map).terminator()).as_terminator() };
    if attrkind == DICT && term.kind != TerminatorKind::Dict {
        return None;
    }
    let unbox_type = unsafe { pick_unbox_type(map, w_value) };
    let (_, holder) =
        unsafe { find_branch_to_move_into_readonly(map, name, attrkind, unbox_type) }?;
    let typ = unsafe { (*holder).typ.get() };
    if typ.is_some() && typ != unbox_type {
        Some(MapdictQmutTarget::HolderTyp(holder))
    } else {
        None
    }
}

/// First `?` flip of `PlainAttribute._copy_attr` → `add_attr` → `pick_attr`
/// against `onto`, the map `back.copy` leaves before this node is re-added.
///
/// `pick_attr` writes `CachedAttributeHolder.typ` when the unbox type
/// disagrees; that assignment is the quasi-immut invalidation
/// (`CachedAttributeHolder.pick_attr`). `find_branch_to_move_into_readonly`
/// returning `None` is the `get_new_attr` arm, which builds a holder whose
/// `typ` already matches and so does not force.
unsafe fn copy_attr_would_force_quasi_immut<O: MapdictObject>(
    onto: MapRef,
    attr: MapRef,
    obj: &O,
) -> Option<MapdictQmutTarget> {
    if onto.is_null() {
        return None;
    }
    let p = unsafe { (*attr).as_plain() };
    let w_value = unsafe { plain_direct_read(attr, obj) };
    let unbox_type = unsafe { pick_unbox_type(onto, w_value) };
    let (_, holder) =
        unsafe { find_branch_to_move_into_readonly(onto, &p.name, p.attrkind, unbox_type) }?;
    let typ = unsafe { (*holder).typ.get() };
    if typ.is_some() && typ != unbox_type {
        Some(MapdictQmutTarget::HolderTyp(holder))
    } else {
        None
    }
}

/// Return the first `?` field `PlainAttribute.delete` would flip, without
/// applying the deletion.
///
/// Matching node: `ever_mutated` (`PlainAttribute.delete`). Newer nodes
/// that the recursive walk `_copy_attr`s onto `back.copy` run
/// `add_attr` → `pick_attr`; each is checked against `deleted.back`, which
/// is exact for the first re-add. Watcher installation belongs to the
/// tracer; this predicate reads no `*_qmut_installed` flag and takes no
/// `QuasiImmutField` lock while a `cache_attrs` mutex or an `INSTANCE_LOCKS`
/// stripe is held.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn delattr_would_force_quasi_immut(
    w_obj: PyObjectRef,
    name: &Wtf8,
    attrkind: u16,
) -> Option<MapdictQmutTarget> {
    let mut current = unsafe { mapdict_map_or_null(w_obj) };
    // Newest first; `_copy_attr` re-adds closest-to-deleted first.
    let mut newer: Vec<MapRef> = Vec::new();
    while !current.is_null() && unsafe { (*current).is_plain() } {
        let p = unsafe { (*current).as_plain() };
        if p.attrkind == attrkind && &*p.name == name {
            if !p.ever_mutated.get() {
                return Some(MapdictQmutTarget::PlainEverMutated(p));
            }
            let carrier = unsafe { mapdict_carrier(w_obj) };
            for &newer_map in newer.iter().rev() {
                if let Some(target) =
                    unsafe { copy_attr_would_force_quasi_immut(p.back, newer_map, &carrier) }
                {
                    return Some(target);
                }
            }
            return None;
        }
        newer.push(current);
        current = p.back;
    }
    None
}

/// mapdict.py `AbstractAttribute._pick_unbox_type`.
///
/// Returns the unbox type when the terminator allows unboxing and the value is
/// an unboxable int (`ALLOW_UNBOXING_INTS`) or float.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes; `w_value` to a live
/// object.
unsafe fn pick_unbox_type(self_node: MapRef, w_value: PyObjectRef) -> Option<UnboxType> {
    let term = unsafe { (*(*self_node).terminator()).as_terminator() };
    if term.allow_unboxing.get() {
        if ALLOW_UNBOXING_INTS && unsafe { is_unboxable_int(w_value) } {
            return Some(UnboxType::Int);
        } else if unsafe { is_unboxable_float(w_value) } {
            return Some(UnboxType::Float);
        }
    }
    None
}

/// mapdict.py `PlainAttribute._switch_map_and_write_increase_storage1`
/// and the `UnboxedPlainAttribute` override (mapdict.py).
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
unsafe fn switch_map_and_write_increase_storage1<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            // mapdict.py _switch_map_and_write_increase_storage1
            if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                // erase_item is identity
                obj._set_mapdict_increase_storage1(attr, w_value);
                return;
            }
            // change the map first, then the storage
            obj._set_mapdict_map(attr);
            unsafe { plain_direct_write(attr, obj, w_value) };
        }
        Some(u) => {
            // mapdict.py _switch_map_and_write_increase_storage1
            let val = unsafe { unbox_value(u.typ, w_value) };
            if u.firstunwrapped {
                // a fresh longlong list of one element occupies a new slot
                let unboxed = erase_unboxed(&[val]);
                // `erase_unboxed` hands back a block nothing references yet, so
                // it needs a root until whichever store below lands claims it.
                // The block does not move, so one pin carries it and it is
                // never read back.  The sibling arm needs none: its
                // `erase_unboxed` is the argument expression of the store.
                let _unboxed_root = pyre_object::gc_roots::push_roots();
                let unboxed = pyre_object::gc_roots::pin_root(unboxed);
                if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                    obj._set_mapdict_increase_storage1(attr, unboxed);
                    return;
                }
                obj._set_mapdict_map(attr);
                obj._mapdict_write_storage(p.storageindex, unboxed);
            } else {
                // append to the existing shared list (a fresh list, matching
                // PyPy's `unboxed + [val]`)
                let slot = obj._mapdict_read_storage(p.storageindex);
                let mut new_list = unsafe { unboxed_items(slot).to_vec() };
                obj._set_mapdict_map(attr);
                debug_assert_eq!(new_list.len(), u.listindex);
                new_list.push(val);
                obj._mapdict_write_storage(p.storageindex, erase_unboxed(&new_list));
            }
        }
    }
}

/// mapdict.py `AbstractAttribute._reorder_and_add` — the complicated
/// case where a lower-order ancestor already has the attribute, so the
/// attributes passed on the way up must be saved and re-added in order.
///
/// PyPy stores the to-be-readded `(map, value)` pairs in a flat erased array
/// indexed by `stack_index`; the Rust port uses a `Vec<(MapRef, PyObjectRef)>`
/// with push/pop (same LIFO behaviour). `erase_item`/`unerase_item` and
/// `erase_map`/`unerase_map` are the identity / the typed tuple here.
///
/// # Safety
/// `self_node`/`attr` and their chains must point to live map nodes.
unsafe fn reorder_and_add<O: MapdictObject>(
    mut self_node: MapRef,
    obj: &mut O,
    mut number_to_readd: usize,
    mut attr: MapRef,
    w_value: PyObjectRef,
) {
    // The saved values pile up in a plain `Vec` and the incoming `w_value` sits
    // in a by-value parameter, neither of which the collector walks, while the
    // loop keeps collecting: `_mapdict_pop_attribute` does so in either arm,
    // and a value the read had to box carries no heap edge until something
    // stores it, so being born old-gen buys it immobility rather than survival
    // (`alloc_typed_items_block`).  Every word here is an arbitrary attribute
    // value, so it can equally be a `list` or a `dict` and move.  All of them
    // live in root slots and are read back at every use.  Map nodes are not GC
    // objects, so the map half of each saved pair stays a plain pointer.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::pin_roots(&[w_value]);
    // mapdict.py:227 flattens the to-be-readded `(map, value)` pairs into one
    // array indexed by `stack_index`, which reuses its cells; taking a fresh
    // slot per push keeps the same LIFO order without the fixed bound.
    let mut stack: Vec<(MapRef, usize)> =
        Vec::with_capacity(unsafe { (*self_node).num_attributes() });
    loop {
        // we found the attributes further up, need to save the previous
        // values of the attributes we passed
        if number_to_readd != 0 {
            let mut current = self_node;
            for _ in 0..number_to_readd {
                // current is a PlainAttribute
                let w_self_value = unsafe { plain_direct_read(current, obj) };
                let _ = pyre_object::gc_roots::pin_root(w_self_value);
                stack.push((current, pyre_object::gc_roots::shadow_stack_len() - 1));
                current = unsafe { (*current).as_plain() }.back;
                obj._mapdict_pop_attribute(current);
            }
        }
        unsafe {
            switch_map_and_write_increase_storage1(
                attr,
                obj,
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )
        };

        // readd the current top of the stack
        match stack.pop() {
            None => return,
            Some((next_map, next_slot)) => {
                pyre_object::gc_roots::shadow_stack_set(
                    value_slot,
                    pyre_object::gc_roots::shadow_stack_get(next_slot),
                );
                let (name, attrkind) = {
                    let p = unsafe { (*next_map).as_plain() };
                    (p.name.clone(), p.attrkind)
                };
                self_node = obj._get_mapdict_map();
                let unbox_type = unsafe {
                    pick_unbox_type(
                        self_node,
                        pyre_object::gc_roots::shadow_stack_get(value_slot),
                    )
                };
                let (n, holder) =
                    unsafe { find_branch_to_move_into(self_node, &name, attrkind, unbox_type) };
                number_to_readd = n;
                attr = unsafe { holder_pick_attr(holder, unbox_type) };
            }
        }
    }
}

/// mapdict.py `AbstractAttribute.add_attr`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn add_attr<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) {
    let unbox_type = unsafe { pick_unbox_type(self_node, w_value) };
    let (number_to_readd, holder) =
        unsafe { find_branch_to_move_into(self_node, name, attrkind, unbox_type) };
    let attr = unsafe { holder_pick_attr(holder, unbox_type) };
    if number_to_readd == 0 {
        unsafe { switch_map_and_write_increase_storage1(attr, obj, w_value) };
    } else {
        // the complicated reorder case
        unsafe { reorder_and_add(self_node, obj, number_to_readd, attr, w_value) };
    }
}

/// mapdict.py `Terminator._write_terminator` plus the
/// `NoDictTerminator` override (mapdict.py).
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn write_terminator<O: MapdictObject>(
    term: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    let kind = unsafe { (*term).as_terminator() }.kind;
    match kind {
        // NoDictTerminator: object without __dict__ rejects DICT writes.
        TerminatorKind::NoDict if attrkind == DICT => return false,
        TerminatorKind::Devolved if attrkind == DICT => {
            // mapdict.py: the devolved terminator writes DICT attributes
            // into the materialised instance dict (`space.setitem_str(
            // obj.getdict(space), name, w_value)`).  `getdict` builds the
            // wrapper when the SPECIAL slot is still empty, so the incoming
            // value is published across it and read back at the store.
            let _roots = pyre_object::gc_roots::push_roots();
            let value_slot = pyre_object::gc_roots::pin_roots(&[w_value]);
            let w_dict = obj.getdict();
            let backing = crate::type_methods::resolve_dict_backing(w_dict);
            unsafe {
                pyre_object::w_dict_setitem_wtf8(
                    backing,
                    name,
                    pyre_object::gc_roots::shadow_stack_get(value_slot),
                )
            };
            return true;
        }
        _ => {}
    }
    let map = obj._get_mapdict_map();
    unsafe { add_attr(map, obj, name, attrkind, w_value) };
    if attrkind == DICT
        && unsafe { (*obj._get_mapdict_map()).num_attributes() } >= LIMIT_MAP_ATTRIBUTES
    {
        // mapdict.py:317-323: once a non-devolved instance accumulates
        // >= LIMIT_MAP_ATTRIBUTES DICT attributes, devolve its `__dict__` to a
        // UnicodeDictStrategy r_dict. `obj.getdict()` returns the MapDictStrategy
        // view installed by the `_obj_getdict` flip (asserted MapDictStrategy at
        // mapdict.py); `switch_to_text_strategy` materialises the DICT
        // attributes into the fresh strategy and rebuilds the map rooted at the
        // DevolvedDictTerminator. Only reachable for a `W_ObjectObject` carrier:
        // a devolved instance returns early through the Devolved arm above, and an
        // `Object` carrier never accrues DICT writes through a Dict terminator.
        let w_dict = obj.getdict();
        debug_assert_eq!(
            unsafe {
                (*(w_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind()
            },
            pyre_object::dictmultiobject::StrategyKind::Map,
            "LIMIT-devolve expects a MapDictStrategy __dict__ view",
        );
        unsafe { mapdict_switch_to_text_strategy(w_dict) };
    }
    true
}

/// mapdict.py `AbstractAttribute.write`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_write<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        None => unsafe {
            write_terminator((*self_node).terminator(), obj, name, attrkind, w_value)
        },
        Some(attr) => {
            let p = unsafe { (*attr).as_plain() };
            p.set_ever_mutated(true);
            unsafe { plain_direct_write(attr, obj, w_value) };
            true
        }
    }
}

// ── MapdictDictSupport ────────────────────────────────────────────────

/// `MapDictStrategy.length` (mapdict.py) — count the DICT attributes
/// by walking the `search(DICT)` chain. `dont_look_inside`: the map-node layer
/// (incl. `ensure_mapdict_initialized` → `new_instance_terminator`) is a JIT
/// residual boundary while Slice D's unboxed branches stay unported, matching
/// [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_length(obj: PyObjectRef) -> usize {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let mut res: usize = 0;
    let mut curr = node_search(inst._get_mapdict_map(), DICT);
    while let Some(node) = curr {
        // mapdict.py: advance to `back`, re-search, count.
        let back = (*node).as_plain().back;
        curr = node_search(back, DICT);
        res += 1;
    }
    res
}

/// `MapDictStrategy.clear` (mapdict.py) — rebuild the instance's
/// map+storage with every DICT entry dropped. `dont_look_inside` (same rationale
/// as [`instance_node_dict_length`]).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_clear(obj: PyObjectRef) {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let new_obj = node_remove_dict_entries(map, &inst);
    inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
}

/// Collect the instance's DICT attribute nodes in insertion order (oldest
/// first): walk `search(DICT)` newest-first (mapdict.py) then reverse
/// (mapdict.py). Shared by the keys/values/items wrappers.
///
/// # Safety
/// `inst` must be a live carrier whose map chain is live.
unsafe fn dict_nodes_in_order<O: MapdictObject>(inst: &O) -> Vec<MapRef> {
    let mut newest_first: Vec<MapRef> = Vec::new();
    let mut curr = node_search(inst._get_mapdict_map(), DICT);
    while let Some(node) = curr {
        newest_first.push(node);
        let back = (*node).as_plain().back;
        curr = node_search(back, DICT);
    }
    let mut ordered: Vec<MapRef> = Vec::new();
    let mut i = newest_first.len();
    while i > 0 {
        i -= 1;
        ordered.push(newest_first[i]);
    }
    ordered
}

/// `MapDictStrategy.iterkeys` materialised (mapdict.py / w_keys) —
/// the DICT attribute names wrapped as str keys, in insertion order.
/// `dont_look_inside` (same rationale).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_keys(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    let mut keys: Vec<PyObjectRef> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        let name = &(*node).as_plain().name;
        // baseobjspace.py:new_interned_str / mapdict.py attribute names:
        // materialising an instance dictionary must return the canonical
        // interned name object, not a fresh equal string on every iteration.
        keys.push(pyre_object::unicodeobject::box_str_constant(name));
        i += 1;
    }
    keys
}

/// `MapDictStrategy` values materialised (mapdict.py) — the DICT
/// attribute values in insertion order. `dont_look_inside` (same rationale).
///
/// Reads via `plain_direct_read` (the pure `_prim_direct_read`), intentionally
/// omitting the `_direct_read` convert-on-read tail (mapdict.py). That
/// tail is value-invisible — it returns the same box and only re-lays-out
/// storage — and upstream performs it safely only because `MapDictIterator*`
/// is name-keyed and lazy, re-resolving each attr against the rebuilt map on
/// every `next_*`. This materialiser instead snapshots raw node pointers up
/// front; converting mid-walk would `_set_mapdict_storage_and_map` in place and
/// leave the already-snapshotted nodes' `storageindex` desynced against the
/// re-laid-out storage Vec. The migrate stays at the name-keyed getattr
/// boundary (`instance_node_getdictvalue`), matching upstream's read site.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_values(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    // A value the read had to box is named by nothing the collector walks
    // until this `Vec` is handed on, and old-gen birth buys immobility rather
    // than survival, so the walk's own allocations can sweep one collected
    // earlier; pin them as they are read and copy the run back.
    let roots = pyre_object::gc_roots::push_roots();
    let values_base = roots.base();
    let mut vals: Vec<PyObjectRef> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        let w_value = plain_direct_read(node, &inst);
        let w_value = roots.pin_root(w_value);
        vals.push(w_value);
        i += 1;
    }
    for (i, slot) in vals.iter_mut().enumerate() {
        *slot = roots.get(values_base + i);
    }
    vals
}

/// `MapDictStrategy` items materialised (mapdict.py) — (str key,
/// value) pairs in insertion order. `dont_look_inside` (same rationale).
///
/// Uses the pure `plain_direct_read` and omits the `_direct_read` convert-on-
/// read tail for the same reason as [`instance_node_dict_values`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    // A value the read had to box is named by nothing the collector walks
    // until this `Vec` is handed on, and old-gen birth buys immobility rather
    // than survival, so the walk's own allocations can sweep one collected
    // earlier. Pin each value at its read and copy the run back once the walk
    // is over; the names are immortal interned constants.
    let roots = pyre_object::gc_roots::push_roots();
    let values_base = roots.base();
    let mut out: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        let name = &(*node).as_plain().name;
        let w_key = pyre_object::unicodeobject::box_str_constant(name);
        let w_value = plain_direct_read(node, &inst);
        let w_value = roots.pin_root(w_value);
        out.push((w_key, w_value));
        i += 1;
    }
    for (i, entry) in out.iter_mut().enumerate() {
        entry.1 = roots.get(values_base + i);
    }
    out
}

/// rerased unerase for [`MapDictStrategy`] (mapdict.py): the dict's
/// erased `dstorage` IS the backing instance (mapdict.py
/// `strategy.erase(self)`), so unerasing yields the `W_ObjectObject`
/// PyObjectRef directly.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
unsafe fn mapdict_strategy_unerase(w_dict: PyObjectRef) -> PyObjectRef {
    let dict = &*(w_dict as *const pyre_object::W_DictObject);
    dict.dstorage as PyObjectRef
}

/// `MapDictStrategy.switch_to_object_strategy` (mapdict.py) —
/// install a fresh ObjectDictStrategy r_dict over the dict, then materialise the
/// instance's DICT attributes into it (the map devolves to its paired
/// DevolvedDictTerminator). `dont_look_inside` keeps the residual boundary like
/// `instance_node_*`: `materialize_dict` reaches `plain_direct_read`, whose
/// unboxed branch (Slice D) the annotator cannot lower; the boundary lets the
/// callers (`getitem`/`setitem`/`delitem` non-str arms) stay lowerable.
///
/// Unlike a typed strategy's switch, the old `dstorage` here is the backing
/// `W_ObjectObject` (mapdict.py `strategy.erase(self)`), an immortal Box,
/// not an owned r_dict — so it is overwritten, never freed.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
#[majit_macros::dont_look_inside]
pub unsafe fn mapdict_switch_to_object_strategy(w_dict: PyObjectRef) {
    use pyre_object::dictmultiobject::DictStrategy;
    // A `dict` moves and `get_empty_storage` allocates, so the wrapper is
    // published before the new storage is built: a `&mut` taken beforehand
    // would write both fields to the pre-move address. The instance does not
    // move, but overwriting `dstorage` severs the dict's only edge to it, so it
    // is published for liveness across the materialisation that follows.
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict]);
    // w_obj = self.unerase(w_dict.dstorage) — the backing instance.
    let w_obj =
        unsafe { mapdict_strategy_unerase(pyre_object::gc_roots::shadow_stack_get(dict_slot)) };
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    // dict_w = strategy.unerase(strategy.get_empty_storage()); set_strategy(Object);
    // w_dict.dstorage = strategy.erase(dict_w).
    let dstorage = pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY.get_empty_storage();
    let dict = unsafe {
        &mut *(pyre_object::gc_roots::shadow_stack_get(dict_slot) as *mut pyre_object::W_DictObject)
    };
    dict.dstorage = dstorage;
    dict.dstrategy = &pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY_REF;
    // `setfield_gc` on `w_dict.dstorage`: the fresh table is born young and
    // an already-old dict has to be remembered before the next minor.
    pyre_object::gc_hook::try_gc_write_barrier(
        pyre_object::gc_roots::shadow_stack_get(dict_slot) as pyre_object::gc_hook::GCREF
    );
    // materialize_r_dict(space, w_obj, dict_w).
    unsafe { materialize_dict(w_obj, pyre_object::gc_roots::shadow_stack_get(dict_slot)) };
}

/// `MapDictStrategy.switch_to_text_strategy` (mapdict.py) — the
/// LIMIT-devolve sibling of [`mapdict_switch_to_object_strategy`]: install a
/// fresh UnicodeDictStrategy r_dict and materialise into it. Same residual
/// boundary and same overwrite-not-free `dstorage` contract.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
#[majit_macros::dont_look_inside]
pub unsafe fn mapdict_switch_to_text_strategy(w_dict: PyObjectRef) {
    use pyre_object::dictmultiobject::DictStrategy;
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict]);
    let w_obj =
        unsafe { mapdict_strategy_unerase(pyre_object::gc_roots::shadow_stack_get(dict_slot)) };
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let dstorage = pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage();
    let dict = unsafe {
        &mut *(pyre_object::gc_roots::shadow_stack_get(dict_slot) as *mut pyre_object::W_DictObject)
    };
    dict.dstorage = dstorage;
    dict.dstrategy = &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF;
    // `setfield_gc` on `w_dict.dstorage`: the fresh table is born young and
    // an already-old dict has to be remembered before the next minor.
    pyre_object::gc_hook::try_gc_write_barrier(
        pyre_object::gc_roots::shadow_stack_get(dict_slot) as pyre_object::gc_hook::GCREF
    );
    // materialize_str_dict(space, w_obj, str_dict).
    unsafe { materialize_dict(w_obj, pyre_object::gc_roots::shadow_stack_get(dict_slot)) };
}

/// mapdict.py `MapDictStrategy` — the dict strategy a user instance's
/// `__dict__` adopts. `dstorage` erases the backing `W_ObjectObject`
/// (mapdict.py), so every routed get/set/del/iter funnels into the
/// instance's mapdict map+storage. `_obj_getdict` installs this view.
pub struct MapDictStrategy;

// mapdict.py:310: allow_unboxing remains true on this shared terminator. A
// type conflict in holder_pick_attr consequently freezes unboxing for every
// instance dictionary sharing this transition tree.
static TERMINATOR_FOR_DICTS: LazyLock<usize> =
    LazyLock::new(|| new_dict_terminator(pyre_object::PY_NULL) as usize);

pub fn get_terminator_for_dicts() -> MapRef {
    *TERMINATOR_FOR_DICTS as MapRef
}

/// mapdict.py `make_instance_dict`.
#[majit_macros::dont_look_inside]
pub fn make_instance_dict() -> PyObjectRef {
    let w_fake_object = pyre_object::w_instance_new(pyre_object::PY_NULL);
    let _roots = pyre_object::gc_roots::push_roots();
    let fake_slot = pyre_object::gc_roots::pin_roots(&[w_fake_object]);
    unsafe {
        (&mut *(pyre_object::gc_roots::shadow_stack_get(fake_slot)
            as *mut pyre_object::W_ObjectObject))
            ._set_mapdict_map(get_terminator_for_dicts());
    }
    _obj_getdict(pyre_object::gc_roots::shadow_stack_get(fake_slot))
}

/// `space.fromcache(MapDictStrategy)` process-wide singleton — same `&'static`
/// ZST contract as [`pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY`].
pub static MAP_DICT_STRATEGY: MapDictStrategy = MapDictStrategy;

/// The [`pyre_object::dictmultiobject::DictStrategyRef`] holder a dict's
/// `dstrategy` slot points at.
pub static MAP_DICT_STRATEGY_REF: pyre_object::dictmultiobject::DictStrategyRef =
    pyre_object::dictmultiobject::DictStrategyRef {
        kind: pyre_object::dictmultiobject::StrategyKind::Map,
        imp: &MAP_DICT_STRATEGY,
        owner: std::ptr::null_mut(),
    };

impl pyre_object::dictmultiobject::DictStrategy for MapDictStrategy {
    fn strategy_kind(&self) -> pyre_object::dictmultiobject::StrategyKind {
        pyre_object::dictmultiobject::StrategyKind::Map
    }

    /// mapdict.py `get_empty_storage` — "mainly used for tests": a
    /// fresh fake `W_ObjectObject` carrier on the shared dict terminator,
    /// erased. Production dstorage is likewise the backing instance
    /// (mapdict.py).
    fn get_empty_storage(&self) -> pyre_object::gc_hook::GCREF {
        let w_result = pyre_object::w_instance_new(pyre_object::PY_NULL);
        unsafe {
            (&mut *(w_result as *mut pyre_object::W_ObjectObject))
                ._set_mapdict_map(get_terminator_for_dicts());
        }
        w_result as pyre_object::gc_hook::GCREF
    }

    /// mapdict.py `getitem`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py:1161 `space.text_w(w_key)` — an exact str key
            // (including a
            // lone surrogate) is looked up by its full WTF-8 name, so a
            // surrogate-named attribute is a mapdict node like any other.
            return instance_node_getdictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
            );
        }
        if pyre_object::_never_equal_to_string(w_key) {
            return None;
        }
        // A `dict` moves, and the switch allocates the object storage and
        // materialises every attribute into it, so the receiver and the key are
        // published across it and read back at the lookup.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict, w_key]);
        let key_slot = dict_slot + 1;
        self.switch_to_object_strategy(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        pyre_object::w_dict_lookup(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
        )
    }

    /// mapdict.py `getitem_str` — `w_obj.getdictvalue(space, key)`.
    /// The trait key is a UTF-8 `&str`; a UTF-8 string is valid WTF-8, so the
    /// node lookup uses its `Wtf8` view directly.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        instance_node_getdictvalue(mapdict_strategy_unerase(w_dict), Wtf8::new(key))
    }

    /// mapdict.py `setitem`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py:1180 — store under the full WTF-8 name so a
            // surrogate-named attribute becomes a mapdict node rather than
            // forcing a strategy switch.
            instance_node_setdictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
                w_value,
            );
            return;
        }
        // Non-string key: it cannot be a mapdict node, so degrade to the
        // object strategy before storing.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict, w_key, w_value]);
        let key_slot = dict_slot + 1;
        let value_slot = dict_slot + 2;
        self.switch_to_object_strategy(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        pyre_object::w_dict_store(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }

    /// mapdict.py `setitem_str` — `flag = w_obj.setdictvalue(...);
    /// assert flag`. The receiver is the instance behind a materialised
    /// `__dict__` view, so its terminator has a dict and the write cannot fail.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        let flag =
            instance_node_setdictvalue(mapdict_strategy_unerase(w_dict), Wtf8::new(key), w_value);
        debug_assert!(flag, "mapdict.py:1174 assert flag");
    }

    /// mapdict.py `setdefault`.
    unsafe fn setdefault(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_default: PyObjectRef,
    ) -> PyObjectRef {
        // The string arm allocates as well: reading an unboxed attribute boxes
        // it. `w_key`'s WTF-8 view stays valid across both arms because a `str`
        // does not move.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict, w_key, w_default]);
        let key_slot = dict_slot + 1;
        let default_slot = dict_slot + 2;
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            let key = pyre_object::w_str_get_wtf8(w_key);
            if let Some(w_result) = instance_node_getdictvalue(
                mapdict_strategy_unerase(pyre_object::gc_roots::shadow_stack_get(dict_slot)),
                key,
            ) {
                return w_result;
            }
            instance_node_setdictvalue(
                mapdict_strategy_unerase(pyre_object::gc_roots::shadow_stack_get(dict_slot)),
                key,
                pyre_object::gc_roots::shadow_stack_get(default_slot),
            );
            return pyre_object::gc_roots::shadow_stack_get(default_slot);
        }
        self.switch_to_object_strategy(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        pyre_object::dictmultiobject::w_dict_setdefault_checked(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            pyre_object::gc_roots::shadow_stack_get(default_slot),
        )
        .unwrap_or_else(|_| pyre_object::gc_roots::shadow_stack_get(default_slot))
    }

    /// mapdict.py `delitem`. pyre's trait returns `bool` (true =
    /// removed) where PyPy raises KeyError on a miss; the caller raises.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py — delete by the full WTF-8 name (a surrogate
            // name addresses a real node, no strategy switch).
            return instance_node_deldictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
            );
        }
        if pyre_object::_never_equal_to_string(w_key) {
            return false;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict, w_key]);
        let key_slot = dict_slot + 1;
        self.switch_to_object_strategy(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY.delitem(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
        )
    }

    /// mapdict.py `length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        instance_node_dict_length(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py `iterkeys` materialised.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        instance_node_dict_keys(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py `itervalues` materialised.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        instance_node_dict_values(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py `iteritems` materialised.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        instance_node_dict_items(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py `clear`.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        instance_node_dict_clear(mapdict_strategy_unerase(w_dict));
    }

    /// mapdict.py `popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        // The receiver is a `dict` and moves; it is published before the first
        // allocation below rather than alongside the carrier further down,
        // because a pin taken after a collection would freeze a pre-move word.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict]);
        let w_obj = mapdict_strategy_unerase(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        let map = inst._get_mapdict_map();
        let curr = node_search(map, DICT)?;
        let key = &(*curr).as_plain().name;
        // mapdict.py reads the value with `getitem_str(w_dict, key)`, but
        // the trait's `getitem_str` takes a `&str` and a node name is WTF-8:
        // `setitem` stores the full name (mapdict.py), so a lone surrogate
        // is a node here and has no `&str` form. The `node_search` hit makes
        // the value-present arm explicit: `node_read` would return
        // `Some(plain_direct_read(curr, inst))` (mapdict.py).
        //
        // Box the key before the read's conversion tail runs: `key` borrows the
        // map node, and the migration below replaces the instance's map.
        //
        // Boxing, the read and the migration all allocate, so each is a point
        // where a minor collection can forward a nursery address out from under
        // a Rust local. Pin the carrier and the two results and read them back
        // from the shadow stack after every such call. `map`, `curr` and `key`
        // need no pin: map nodes are not GC-allocated.
        let obj_slot = pyre_object::gc_roots::pin_roots(&[w_obj]);
        let key_slot =
            pyre_object::gc_roots::pin_roots(&[pyre_object::unicodeobject::box_str_constant(key)]);
        let inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        let value_slot = pyre_object::gc_roots::pin_roots(&[plain_direct_read(curr, &inst)]);
        // `plain_direct_read` is only `_prim_direct_read` (mapdict.py).
        // The read still owes `_direct_read`'s tail (mapdict.py): an
        // unboxed attribute whose terminator has stopped allowing unboxing
        // converts the whole instance to boxed storage. `getdictvalue` pairs
        // the two the same way.
        let mut inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        migrate_to_boxed_if_frozen(curr, &mut inst);
        self.delitem(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
        );
        Some((
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        ))
    }

    /// mapdict.py `copy`.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        use pyre_object::dictmultiobject::DictStrategy;

        // Both dicts move, and every step after the receiver arrives collects:
        // the copy and its storage are allocated, materialising the items boxes
        // each unboxed attribute, and a store hashes its key through `__hash__`.
        // Publish the receiver and the copy and read each back at its use. The
        // item values are published as one run before the first store, because
        // a store that relocates one of them would leave every entry still to
        // come naming a pre-move address; the names are interned constants,
        // which do not move.
        let _roots = pyre_object::gc_roots::push_roots();
        let dict_slot = pyre_object::gc_roots::pin_roots(&[w_dict]);
        let copy_slot = pyre_object::gc_roots::pin_roots(&[pyre_object::w_dict_new_with(
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF,
            pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage(),
        )]);
        let items = instance_node_dict_items(mapdict_strategy_unerase(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        ));
        let values: Vec<PyObjectRef> = items.iter().map(|&(_, w_value)| w_value).collect();
        let values_slot = pyre_object::gc_roots::pin_roots(&values);
        for (i, &(w_key, _)) in items.iter().enumerate() {
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(copy_slot),
                w_key,
                pyre_object::gc_roots::shadow_stack_get(values_slot + i),
            );
        }
        pyre_object::gc_roots::shadow_stack_get(copy_slot)
    }

    /// mapdict.py:1268-1276 iterator order.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        let w_obj = mapdict_strategy_unerase(w_dict);
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        let nodes = dict_nodes_in_order(&inst);
        let node = *nodes.get(index)?;
        Some((
            pyre_object::unicodeobject::box_str_constant(&(*node).as_plain().name),
            plain_direct_read(node, &inst),
        ))
    }

    /// mapdict.py `MapDictKeyIteratorReversed`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let w_obj = mapdict_strategy_unerase(w_dict);
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        // Same walk as `instance_node_dict_items`, so the same bracket: a value
        // the read had to box is named by nothing the collector walks until the
        // `Vec` is handed on, and old-gen birth buys immobility rather than
        // survival, so the walk's own allocations can sweep one collected
        // earlier.  Pin each value at its read and copy the run back once the
        // walk is over; the names are immortal interned constants.
        let roots = pyre_object::gc_roots::push_roots();
        let values_base = roots.base();
        let mut items: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
        let mut curr = node_search(inst._get_mapdict_map(), DICT);
        while let Some(node) = curr {
            let w_key = pyre_object::unicodeobject::box_str_constant(&(*node).as_plain().name);
            let w_value = plain_direct_read(node, &inst);
            let w_value = roots.pin_root(w_value);
            items.push((w_key, w_value));
            curr = node_search((*node).as_plain().back, DICT);
        }
        for (i, entry) in items.iter_mut().enumerate() {
            entry.1 = roots.get(values_base + i);
        }
        items
    }

    /// mapdict.py `switch_to_object_strategy` — Slice E (#196). The
    /// default would mis-read the instance `dstorage` as an ObjectDictStrategy
    /// `IndexMap`, so override to the materialise stub.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        mapdict_switch_to_object_strategy(w_dict);
    }

    /// The dict is a view over the backing instance (`dstorage =
    /// erase(self)`, mapdict.py). `MapDictStrategy.erase/unerase =
    /// rerased.new_erasing_pair("map")` (mapdict.py) boxes the
    /// instance as a real GC reference, so `dstorage` is a true GC edge
    /// the translated `W_DictMultiObject` tracer forwards. Visit the
    /// `dstorage` field IN PLACE (`addr_of_mut!`, never a
    /// `mapdict_strategy_unerase` stack temporary — the collector writes
    /// the relocated address back through this pointer, so it must be the
    /// real field) so a moving collector rewrites the back-pointer to the
    /// relocated instance. The instance is itself a primary GC object
    /// (forwarded from frame / shadow roots + its own custom trace), so
    /// this edge is an idempotent redundant forwarder: the cycle
    /// instance → storage[SPECIAL] → wrapper → dstorage → instance
    /// terminates on the collector's `is_forwarded` short-circuit.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        unsafe {
            let dstorage_field =
                std::ptr::addr_of_mut!((*(w_dict as *mut pyre_object::W_DictObject)).dstorage)
                    as *mut PyObjectRef;
            if (*dstorage_field).is_null() {
                return;
            }
            visitor(dstorage_field);
        }
    }
}

/// objspace/std/mapdict.py _obj_getdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_getdict(self, space):
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     w_dict = self._get_mapdict_map().read(self, "dict", SPECIAL)
///     if w_dict is not None:
///         assert isinstance(w_dict, W_DictMultiObject)
///         return w_dict
///
///     strategy = space.fromcache(MapDictStrategy)
///     storage = strategy.erase(self)
///     w_dict = W_DictObject(space, strategy, storage)
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
///     return w_dict
/// ```
///
/// `dont_look_inside` — the miss path allocates the MapDictStrategy view and
/// writes the `"dict"` SPECIAL slot, state the tracer cannot model; the call
/// residualises via the registered fnaddr (`@objectmodel.dont_inline`
/// upstream, mapdict.py).
#[majit_macros::dont_look_inside]
pub fn _obj_getdict(self_ref: PyObjectRef) -> PyObjectRef {
    debug_assert!(unsafe { has_mapdict_storage(self_ref) });
    // mapdict.py: read the "dict" SPECIAL slot; on a miss build the
    // MapDictStrategy view and write it back into that slot. `strategy.erase(self)`
    // makes the view funnel every get/set/del/iter through the instance map+storage
    // — the single `__dict__` authority.
    //
    // mapdict.py `if w_dict is not None`.  RPython's `read` answers
    // None both for an absent slot and for one holding None, and both mean
    // "build the view" — so a null must not reach the caller.  It would be
    // reported as "the receiver has no dict", which is how
    // `descr__setattr__` decides an ordinary instance store is
    // `"'%T' object attribute '%s' is read-only"` (descroperation.py).
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_ref);
    if let Some(w_dict) =
        unsafe { instance_get_dict_slot(pyre_object::gc_roots::shadow_stack_get(self_slot)) }
        && !w_dict.is_null()
    {
        return w_dict;
    }
    // Allocating the wrapper and claiming the SPECIAL slot both collect, so
    // the instance is published first and every use below reads back the
    // relocated address.
    let dict_slot = self_slot + 1;
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new_with(
        &MAP_DICT_STRATEGY_REF,
        pyre_object::gc_roots::shadow_stack_get(self_slot) as pyre_object::gc_hook::GCREF,
    ));
    unsafe {
        let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
        // `dstorage` is the view's only link to its backing instance
        // (mapdict.py) and `walk_gc_refs` forwards it — but only from
        // the collection after the wrapper became reachable.  The
        // allocation that produced the wrapper is not covered, so restate
        // the back-pointer from the instance's current address.
        (*(w_dict as *mut pyre_object::W_DictObject)).dstorage =
            pyre_object::gc_roots::shadow_stack_get(self_slot) as pyre_object::gc_hook::GCREF;
        pyre_object::gc_hook::try_gc_write_barrier(w_dict as pyre_object::gc_hook::GCREF);
        let flag = instance_set_dict_slot(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        );
        debug_assert!(flag, "write to the \"dict\" SPECIAL slot failed");
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// GC custom trace over a live instance's `storage` value slots.
///
/// `mapdict.py _set_mapdict_map` — an instance's attribute values live in the
/// `storage` block. Every slot is a GC reference: a boxed attribute's
/// `erase_item` is the identity (`mapdict.py:438/447`) and an
/// `UnboxedPlainAttribute`'s `erase_unboxed` (`:601/612`) yields a varsize
/// leaf GcArray ([`erase_unboxed`]), so the map no longer has to be
/// consulted to tell the two apart. The visitor relocates each slot's
/// contents in place, exactly as `dict_object_custom_trace` relocates dict
/// entry slots.
///
/// Registered as `W_OBJECT_OBJECT_GC_TYPE_ID`'s custom trace
/// (`object_object_custom_trace`) so a moving collector forwards an
/// instance's attributes, and so a mark-sweep old generation keeps the
/// unboxed longlong list alive for exactly as long as the instance.
///
/// # Safety
/// `obj` must be a live object for which [`has_mapdict_layout`] holds.
pub unsafe fn instance_walk_boxed_storage(obj: PyObjectRef, f: &mut dyn FnMut(*mut PyObjectRef)) {
    unsafe {
        let inst = mapdict_carrier(obj);
        let storage_slot = inst.slots().1;
        if (*storage_slot).is_null() {
            return;
        }
        if pyre_object::gc_hook::try_gc_owns_object(*storage_slot as pyre_object::gc_hook::GCREF) {
            f(storage_slot as *mut PyObjectRef);
        }
        // Both forms are needed, unlike in `list_object_custom_trace`, which
        // picks one: handing over the field slot only forwards the block
        // pointer. A young block is then scanned by its own walker; an
        // old-gen spill is not entered by a minor, so walking the items
        // here keeps an attribute value alive.
        //
        // The block's own tid registers `items_have_gc_ptrs`, which is not a
        // reason to drop this walk: a declared walker is not a guarantee that
        // the walk runs.  That flag describes what happens *when* the block is
        // traversed, and a minor does not traverse an old-gen block at all.
        //
        // Read the block back out of the slot, since the call above is free to
        // have forwarded it.
        let storage = *storage_slot;
        // The storage block is exact-size (capacity == map.storage_needed()).
        let len = pyre_object::object_array::items_block_capacity(storage);
        let base = pyre_object::object_array::items_block_items_base(storage);
        // No per-slot test and no `Vec`: this runs inside the collector's
        // marking walk, where both would be a per-object cost and a reentrancy
        // hazard, and every slot is a reference.
        for i in 0..len {
            f(base.add(i));
        }
    }
}

/// Custom trace for objects carrying the `MapdictStorageMixin` prefix
/// (`W_ObjectObject`; instance `map`+`storage`, `mapdict.py`). A
/// `#[pyre_class(..., user_layout)]` subclass carries that prefix on its
/// `_getusercls` layout (`typedef.py`), not on the builtin base.
///
/// `storage` is a GC-managed leaf block (`W_MAPDICT_STORAGE_GC_TYPE_ID`,
/// nursery-born by `alloc_mapdict_storage_block`), so the collector reaches
/// its slots only through this trace:
/// [`instance_walk_boxed_storage`] forwards the `storage` reference itself and
/// then every slot in `0..capacity` in place.  It consults no map — an
/// unboxed longlong attribute stores the erased `GC_INT_ARRAY` block
/// (`erase_unboxed`, `mapdict.py/612`), which is an ordinary GC reference
/// like every boxed slot (`mapdict.py/447` `erase_item`), so no slot has
/// to be skipped.  A minor that moves a young block rewrites the forwarded
/// `storage` word here.
///
/// `ob_header.w_class` is the instance's class reachability edge — the
/// equivalent of PyPy reaching the class through the traced
/// `terminator.w_cls` (`mapdict.py`, a strong `_immutable_field_`).
/// Pyre stores the class in the inline header word
/// (`objectobject.rs`'s `W_ObjectObject`, `typeptr` in `rclass.py`), so it
/// must be forwarded here or an instance whose class is reachable only
/// through it would have that class reclaimed: heap types are GC-managed
/// (`try_gc_alloc_stable_raw`).
///
/// # Safety
/// `obj_addr` must point to a live object with the mapdict prefix.
pub unsafe fn mapdict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let obj = obj_addr as PyObjectRef;
    f(unsafe { std::ptr::addr_of_mut!((*obj).w_class) } as *mut majit_ir::GcRef);
    unsafe {
        instance_walk_boxed_storage(obj, &mut |slot: *mut PyObjectRef| {
            f(slot as *mut majit_ir::GcRef);
        })
    };
}

/// objspace/std/mapdict.py _obj_setdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_setdict(self, space, w_dict):
///     from pypy.interpreter.error import oefmt
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     if not space.isinstance_w(w_dict, space.w_dict):
///         raise oefmt(space.w_TypeError, "setting dictionary to a non-dict")
///     assert isinstance(w_dict, W_DictMultiObject)
///     w_olddict = self.getdict(space)
///     ...
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
/// ```
///
/// `dont_look_inside` — the write reads the old `"dict"` SPECIAL view,
/// materialises it off the instance, and stores the replacement, state the
/// tracer cannot model; the JIT residualises the call
/// (`@objectmodel.dont_inline`, mapdict.py).
#[majit_macros::dont_look_inside]
pub fn _obj_setdict(self_ref: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    debug_assert!(unsafe { has_mapdict_storage(self_ref) });
    // mapdict.py `space.isinstance_w(w_dict, space.w_dict)` accepts
    // dict subclasses. Pyre's composed dict-subclass representation is
    // resolved by the getdict backing helpers at each raw dict operation,
    // while the SPECIAL slot retains the supplied object's identity.
    // A dict subclass passes the type test, so the composed representation is
    // asked for its backing too. Publish the receiver and the incoming dict
    // before that call; both are read again while the old view is materialised.
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::pin_roots(&[self_ref, w_dict]);
    let dict_slot = self_slot + 1;
    crate::baseobjspace::require_dict_for_setdict(pyre_object::gc_roots::shadow_stack_get(
        dict_slot,
    ))?;
    if crate::type_methods::resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(dict_slot))
        .is_null()
    {
        return Err(crate::baseobjspace::setdict_not_a_dict(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        ));
    }
    // mapdict.py: the old dict has `self` as its dstorage, so
    // before pointing the "dict" SPECIAL slot at the new dict, force the
    // old view to its own storage if it is still an instance-backed
    // `MapDictStrategy`. `_obj_getdict` returns (or materialises) that
    // view; switching it to an ObjectDictStrategy snapshot stops it
    // delegating to the instance once the slot is overwritten — otherwise
    // `old = obj.__dict__; obj.__dict__ = {}` leaves `old` an empty shell
    // that still mirrors the live instance.
    let w_olddict = _obj_getdict(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let old_backing = crate::type_methods::resolve_dict_backing(w_olddict);
    let is_map_view = unsafe {
        pyre_object::dictmultiobject::w_dict_get_strategy(old_backing).strategy_kind()
            == pyre_object::dictmultiobject::StrategyKind::Map
    };
    if is_map_view {
        unsafe { mapdict_switch_to_object_strategy(old_backing) };
    }
    let flag = unsafe {
        instance_set_dict_slot(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        )
    };
    debug_assert!(flag, "write to the \"dict\" SPECIAL slot failed");
    Ok(())
}

// ── MapdictWeakrefSupport ─────────────────────────────────────────────

/// Byte offset of `make_weakref_descr`'s `_lifeline_` on `obj`, or `0` when
/// the typeptr does not publish one. A tagged int has no `ob_type` word.
fn weakref_layout_offset(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return 0;
    }
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return 0;
    }
    unsafe { (*tp).weakref_offset }
}

/// `TypeDef.weakrefable` (`typedef.py`). Exact `object` shares `INSTANCE_TYPE`
/// with user instances, so the offset alone is not the flag. Both the
/// `_lifeline_` field and the `"weakref"` SPECIAL slot are gated on it.
fn class_is_weakrefable(obj: PyObjectRef) -> bool {
    crate::typedef::r#type(obj)
        .is_some_and(|w_type| unsafe { pyre_object::w_type_get_weakrefable(w_type.as_ptr()) })
}

/// `MapdictWeakrefSupport.getweakref`, and `make_weakref_descr`'s `getweakref`
/// when the typeptr publishes `_lifeline_`. `W_Root.getweakref` returns None.
/// A null lifeline is None.
pub fn getweakref(self_ref: PyObjectRef) -> Option<PyObjectRef> {
    let offset = weakref_layout_offset(self_ref);
    let mapdict = offset == 0 && unsafe { has_mapdict_layout(self_ref) };
    if (offset == 0 && !mapdict) || !class_is_weakrefable(self_ref) {
        return None;
    }
    if offset != 0 {
        let lifeline = unsafe { pyre_object::read_weakref_lifeline(self_ref) };
        (!lifeline.is_null()).then_some(lifeline)
    } else {
        unsafe { instance_get_weakref_slot(self_ref) }
    }
}

/// `MapdictWeakrefSupport.setweakref`, and `make_weakref_descr`'s `setweakref`.
/// `W_Root.setweakref` raises TypeError. A class whose `weakrefable` flag is
/// set but whose layout has neither `_lifeline_` nor mapdict storage is a
/// missing port and raises SystemError, the same shape as a missing dict.
pub fn setweakref(self_ref: PyObjectRef, weakreflifeline: PyObjectRef) -> Result<(), PyError> {
    let offset = weakref_layout_offset(self_ref);
    let mapdict = offset == 0 && unsafe { has_mapdict_layout(self_ref) };
    let w_type = crate::typedef::r#type(self_ref);
    let weakrefable =
        w_type.is_some_and(|w| unsafe { pyre_object::w_type_get_weakrefable(w.as_ptr()) });
    if weakrefable && offset != 0 {
        unsafe { pyre_object::write_weakref_lifeline(self_ref, weakreflifeline) };
        return Ok(());
    }
    if weakrefable && mapdict {
        let flag = unsafe { instance_set_weakref_slot(self_ref, weakreflifeline) };
        debug_assert!(flag, "write to the weakref SPECIAL slot failed");
        return Ok(());
    }
    let Some(w_type) = w_type else {
        return Err(PyError::type_error(
            "cannot create weak reference to object".to_string(),
        ));
    };
    let tp_name = unsafe { pyre_object::w_type_get_name(w_type.as_ptr()) };
    if weakrefable {
        Err(PyError::system_error(format!(
            "'{tp_name}' instance has no weakref storage (builtin base without a lifeline)"
        )))
    } else {
        Err(PyError::type_error(format!(
            "cannot create weak reference to '{tp_name}' object"
        )))
    }
}

/// `MapdictWeakrefSupport.delweakref`, and `make_weakref_descr`'s `delweakref`.
/// `W_Root.delweakref` is a no-op. A field clear stores null and runs the
/// write barrier.
pub fn delweakref(self_ref: PyObjectRef) {
    let offset = weakref_layout_offset(self_ref);
    let mapdict = offset == 0 && unsafe { has_mapdict_layout(self_ref) };
    if (offset == 0 && !mapdict) || !class_is_weakrefable(self_ref) {
        return;
    }
    if offset != 0 {
        unsafe { pyre_object::write_weakref_lifeline(self_ref, pyre_object::PY_NULL) };
    } else {
        unsafe { instance_del_weakref_slot(self_ref) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_subclass_instance_uses_user_typeptr() {
        let payload = majit_rlib::rbigint::RBigInt::from(10_i64)
            .pow(&majit_rlib::rbigint::RBigInt::from(30_i64), None)
            .expect("10**30");
        let exact = pyre_object::longobject::w_long_new(payload);
        let raw = unsafe { pyre_object::longobject::w_long_get_raw_value(exact) };
        let user = pyre_object::longobject::w_long_subclass_from_raw(raw);
        unsafe {
            assert!(std::ptr::eq(
                (*user).ob_type,
                &pyre_object::pyobject::LONG_USER_TYPE
            ));
            assert!(has_mapdict_layout(user));
            assert!(pyre_object::is_long(user));
            assert!(std::ptr::eq(
                (*exact).ob_type,
                &pyre_object::pyobject::LONG_TYPE
            ));
            assert!(!has_mapdict_layout(exact));
            assert!(pyre_object::is_long(exact));
        }
    }

    // Node names are WTF-8; the test fixtures spell them with plain UTF-8
    // string literals. `wn` borrows a `&Wtf8` view for the lookup/read/write
    // helpers, `wb` produces an owned `Wtf8Buf` for `new_plain_attribute`.
    fn wn(s: &str) -> &Wtf8 {
        Wtf8::new(s)
    }
    fn wb(s: &str) -> Wtf8Buf {
        Wtf8::new(s).to_owned()
    }

    // A map chain is `terminator <- "a"(DICT) <- "b"(DICT)`. The w_cls is a
    // null placeholder: the node layer never dereferences it.
    unsafe fn build_chain() -> (MapRef, MapRef, MapRef) {
        let term = new_dict_terminator(std::ptr::null_mut());
        let a = unsafe { new_plain_attribute(wb("a"), DICT, term, 0) };
        let b = unsafe { new_plain_attribute(wb("b"), DICT, a, 1) };
        (term, a, b)
    }

    #[test]
    fn terminator_is_its_own_terminator_with_zero_storage() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            assert_eq!((*term).terminator(), term);
            assert_eq!((*term).storage_needed(), 0);
            assert_eq!((*term).num_attributes(), 0);
            assert_eq!((*term).as_terminator().kind, TerminatorKind::Dict);
            assert!(
                !(*term)
                    .as_terminator()
                    .devolved_dict_terminator
                    .get()
                    .is_null()
            );
        }
    }

    #[test]
    fn plain_attribute_increments_storage_and_keeps_terminator() {
        unsafe {
            let (term, a, b) = build_chain();
            assert_eq!((*a).as_plain().storageindex, 0);
            assert_eq!((*a).storage_needed(), 1);
            assert_eq!((*a).num_attributes(), 1);
            assert_eq!((*b).as_plain().storageindex, 1);
            assert_eq!((*b).storage_needed(), 2);
            assert_eq!((*b).num_attributes(), 2);
            assert_eq!((*a).terminator(), term);
            assert_eq!((*b).terminator(), term);
            assert_eq!((*b).as_plain().back, a);
        }
    }

    #[test]
    fn map_repr_follows_recursive_pypy_strategy_spelling() {
        unsafe {
            let (_, _, b) = build_chain();
            let repr = (*b).repr_wtf8();
            assert_eq!(
                repr.as_bytes(),
                b"<PlainAttribute b DICT 1 immutable <PlainAttribute a DICT 0 immutable <DictTerminator w_cls=<W_TypeObject '?'>>>>"
            );
        }
    }

    #[test]
    fn cache_entry_validity_keys_on_map_identity_and_version_tag() {
        unsafe {
            // mapdict.py is_valid_for_map / _is_valid_for_map. Build a
            // real type so the terminator carries a w_cls with a live
            // version_tag.
            crate::typedef::init_typeobjects();
            let w_cls = crate::typedef::make_builtin_type("MapdictCacheEntryT", |_| {});
            let term = new_dict_terminator(w_cls);
            let attr = new_plain_attribute(wb("x"), DICT, term, 0);
            let version_tag = pyre_object::typeobject::w_type_get_version_tag(w_cls);

            let entry = MapdictCacheEntry {
                cached_map: attr,
                cached_attr: attr,
                version_tag,
                valid_for_store: false,
                w_method: std::ptr::null_mut(),
            };
            // matching map + version -> valid for a load
            assert!(entry.is_valid_for_map(attr, false));
            // a different map fails identity (mapdict.py `mymap is map`)
            let other = new_plain_attribute(wb("y"), DICT, term, 1);
            assert!(!entry.is_valid_for_map(other, false));
            // store gate: valid_for_store == false rejects stores (mapdict.py:1432)
            assert!(!entry.is_valid_for_map(attr, true));

            let store_entry = MapdictCacheEntry {
                valid_for_store: true,
                ..entry
            };
            assert!(store_entry.is_valid_for_map(attr, true));

            // a class mutation bumps version_tag -> the entry goes stale
            // (mapdict.py `version_tag is self.version_tag`).
            crate::baseobjspace::mutated_absent(w_cls);
            assert!(!entry.is_valid_for_map(attr, false));
            assert!(!store_entry.is_valid_for_map(attr, true));
        }
    }

    #[test]
    fn find_map_attr_chain_walks_back_pointers() {
        unsafe {
            let (_term, a, b) = build_chain();
            assert_eq!(find_map_attr_chain(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr_chain(b, wn("b"), DICT), Some(b));
            assert_eq!(find_map_attr_chain(b, wn("c"), DICT), None);
            // attrkind namespaces are distinct: "a" exists only under DICT.
            assert_eq!(find_map_attr_chain(b, wn("a"), SPECIAL), None);
        }
    }

    #[test]
    fn find_map_attr_cached_matches_uncached_on_hit_and_miss() {
        unsafe {
            MAP_ATTR_CACHE.lock().clear();
            let (_term, a, b) = build_chain();
            // first call populates the cache, second hits it
            assert_eq!(find_map_attr(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr(b, wn("b"), DICT), Some(b));
            assert_eq!(find_map_attr(b, wn("missing"), DICT), None);
            assert_eq!(find_map_attr(b, wn("missing"), DICT), None);
        }
    }

    #[test]
    fn cache_attrs_stays_none_until_get_new_attr() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            assert!(
                (*term).cache_attrs().is_none(),
                "_find_branch_to_move_into leaves cache_attrs as None"
            );
            let _ = get_new_attr(term, Wtf8::new("x"), DICT, None);
            assert!((*term).cache_attrs().is_some());
        }
    }

    #[test]
    fn new_attr_cache_interns_transitions_across_threads() {
        use std::sync::{Arc, Barrier};

        let term_addr = new_dict_terminator(std::ptr::null_mut()) as usize;
        let barrier = Arc::new(Barrier::new(16));
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        let term = term_addr as MapRef;
                        barrier.wait();
                        (0..64)
                            .map(|index| {
                                let name = format!("attr_{index}");
                                unsafe {
                                    get_new_attr(term, Wtf8::new(name.as_str()), DICT, None)
                                        as usize
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
        assert_eq!(
            unsafe { (*(term_addr as MapRef)).cache_attrs() }
                .expect("_get_new_attr interned the cache")
                .lock()
                .len(),
            64
        );
    }

    #[test]
    fn search_returns_topmost_match_for_attrkind() {
        unsafe {
            let (term, _a, b) = build_chain();
            // both "a" and "b" are DICT; search from b returns b (topmost)
            assert_eq!(node_search(b, DICT), Some(b));
            assert_eq!(node_search(b, SPECIAL), None);
            assert_eq!(node_search(term, DICT), None);
        }
    }

    // A minimal MapdictObject for read-path tests. Storage holds sentinel
    // pointers that are never dereferenced.
    // mapdict.py `Object` is pyre's production storage carrier (empty map,
    // owned `Vec` storage); the tests drive it under the historical `MockObj`
    // name.
    use super::Object as MockObj;

    fn sentinel(n: usize) -> PyObjectRef {
        n as PyObjectRef
    }

    // A DictTerminator whose `allow_unboxing` is off, so writes take the boxed
    // PlainAttribute path and never type-inspect the (sentinel) value.
    unsafe fn boxed_dict_terminator() -> MapRef {
        let term = new_dict_terminator(std::ptr::null_mut());
        unsafe { (*term).as_terminator() }.set_allow_unboxing(false);
        term
    }

    #[test]
    fn node_read_returns_stored_value_by_storageindex() {
        unsafe {
            let (_term, _a, b) = build_chain();
            // map b: "a"@storageindex 0, "b"@storageindex 1
            let obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            assert_eq!(node_read(b, &obj, wn("a"), DICT), Some(sentinel(0xa)));
            assert_eq!(node_read(b, &obj, wn("b"), DICT), Some(sentinel(0xb)));
            // absent attribute falls through to the (Dict) terminator → None
            assert_eq!(node_read(b, &obj, wn("missing"), DICT), None);
            assert_eq!(node_read(b, &obj, wn("a"), SPECIAL), None);
            assert_eq!(obj._mapdict_storage_length(), 2);
        }
    }

    #[test]
    fn set_terminator_reroots_chain_preserving_attrs() {
        unsafe {
            // chain "a","b" (DICT, boxed) under terminator t1
            let t1 = boxed_dict_terminator();
            let a = new_plain_attribute(wb("a"), DICT, t1, 0);
            let b = new_plain_attribute(wb("b"), DICT, a, 1);
            let obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            // re-root onto an unrelated terminator t2
            let t2 = boxed_dict_terminator();
            let new_obj = node_set_terminator(b, &obj, t2);
            // new chain is rooted at t2 with the same attrs/order/values
            assert_eq!((*new_obj.map).terminator(), t2);
            assert_eq!(
                node_read(new_obj.map, &new_obj, wn("a"), DICT),
                Some(sentinel(0xa))
            );
            assert_eq!(
                node_read(new_obj.map, &new_obj, wn("b"), DICT),
                Some(sentinel(0xb))
            );
            assert_eq!(new_obj.storage.len(), 2);
        }
    }

    #[test]
    fn set_terminator_from_devolved_root_targets_devolved_pair() {
        unsafe {
            // obj's map root is t1's devolved terminator (a devolved instance).
            let t1 = new_dict_terminator(std::ptr::null_mut());
            let dev1 = (*t1).as_terminator().devolved_dict_terminator.get();
            let obj = MockObj {
                map: dev1,
                storage: vec![],
            };
            // re-rooting onto t2 lands on t2's devolved pair, not t2 itself.
            let t2 = new_dict_terminator(std::ptr::null_mut());
            let dev2 = (*t2).as_terminator().devolved_dict_terminator.get();
            let new_obj = node_set_terminator(dev1, &obj, t2);
            assert_eq!(new_obj.map, dev2);
        }
    }

    #[test]
    fn plain_direct_write_then_read_roundtrips() {
        unsafe {
            let (_term, a, b) = build_chain();
            let mut obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            plain_direct_write(a, &mut obj, sentinel(0x111));
            assert_eq!(node_read(b, &obj, wn("a"), DICT), Some(sentinel(0x111)));
            assert_eq!(node_read(b, &obj, wn("b"), DICT), Some(sentinel(0xb)));
        }
    }

    #[test]
    fn add_attr_via_write_grows_map_and_storage() {
        unsafe {
            // start empty: a DictTerminator, no storage
            let term = boxed_dict_terminator();
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            // write two fresh attributes; each takes the common
            // (number_to_readd == 0) append path
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("x"), DICT, sentinel(0x1)));
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("y"), DICT, sentinel(0x2)));

            assert_eq!(obj.storage.len(), 2);
            assert_eq!(unsafe { (*obj.map).num_attributes() }, 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("x"), DICT), Some(sentinel(0x1)));
            assert_eq!(node_read(m, &obj, wn("y"), DICT), Some(sentinel(0x2)));
            assert_eq!(node_read(m, &obj, wn("z"), DICT), None);

            // overwrite an existing attribute → direct write, no growth
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("x"), DICT, sentinel(0x9)));
            assert_eq!(obj.storage.len(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("x"), DICT), Some(sentinel(0x9)));
        }
    }

    #[test]
    fn instance_object_write_grows_map_and_storage() {
        unsafe {
            // The real W_ObjectObject (pyre-object) is the
            // MapdictStorageMixin carrier; exercise its trait impl rather
            // than the MockObj double. `map`/`storage` start null
            // (_mapdict_init_empty), so the map terminator is installed
            // here as the mapdict layer would on first attribute access.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("x"), DICT, sentinel(0x1)));
            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("y"), DICT, sentinel(0x2)));

            assert_eq!(obj._mapdict_storage_length(), 2);
            assert_eq!((*obj._get_mapdict_map()).num_attributes(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, obj, wn("x"), DICT), Some(sentinel(0x1)));
            assert_eq!(node_read(m, obj, wn("y"), DICT), Some(sentinel(0x2)));
            assert_eq!(node_read(m, obj, wn("z"), DICT), None);

            // overwrite an existing attribute → direct write, no growth
            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("x"), DICT, sentinel(0x9)));
            assert_eq!(obj._mapdict_storage_length(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, obj, wn("x"), DICT), Some(sentinel(0x9)));
        }
    }

    #[test]
    fn instance_node_get_set_roundtrip() {
        unsafe {
            // Pre-install the terminator so `ensure_mapdict_initialized` is a
            // no-op (no real W_TypeObject needed) and exercise the get/set
            // wrappers the attribute read/write paths call.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("x")),
                Some(sentinel(0x11))
            );
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("y")),
                Some(sentinel(0x22))
            );
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("z")), None);

            // overwrite an existing attribute
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x99)));
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("x")),
                Some(sentinel(0x99))
            );
        }
    }

    #[test]
    fn instance_node_surrogate_name_stays_a_mapdict_node() {
        unsafe {
            // A lone-surrogate attribute name addresses a real mapdict node,
            // keyed by its full WTF-8 name, rather than degrading the instance
            // dict to the object strategy. mapdict.py getitem routes any str
            // key — surrogate-bearing included — through getdictvalue.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            // A '\udc81' lone surrogate, interleaved with a plain-named attr.
            let mut sur = Wtf8Buf::new();
            sur.push(rustpython_wtf8::CodePoint::from_u32(0xDC81).unwrap());

            assert!(instance_node_setdictvalue(obj_ref, &sur, sentinel(0x55)));
            assert!(instance_node_setdictvalue(
                obj_ref,
                wn("ascii"),
                sentinel(0x66)
            ));

            // Both round-trip; the surrogate attribute is a node, so the dict
            // length and the materialised key set include it.
            assert_eq!(
                instance_node_getdictvalue(obj_ref, &sur),
                Some(sentinel(0x55))
            );
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("ascii")),
                Some(sentinel(0x66))
            );
            assert_eq!(instance_node_dict_length(obj_ref), 2);

            // Key materialisation rebuilds the surrogate name through
            // w_str_from_wtf8 without panicking on the lone surrogate.
            let keys = instance_node_dict_keys(obj_ref);
            assert_eq!(keys.len(), 2);
            assert_eq!(pyre_object::w_str_get_wtf8(keys[0]), &*sur);

            // Deleting the surrogate node leaves the plain attribute intact.
            assert!(instance_node_deldictvalue(obj_ref, &sur));
            assert_eq!(instance_node_getdictvalue(obj_ref, &sur), None);
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("ascii")),
                Some(sentinel(0x66))
            );
            assert_eq!(instance_node_dict_length(obj_ref), 1);
        }
    }

    #[test]
    fn node_delete_rebuilds_without_target() {
        unsafe {
            let term = boxed_dict_terminator();
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            for (name, v) in [("a", 0xa), ("b", 0xb), ("c", 0xc)] {
                let m = obj._get_mapdict_map();
                assert!(node_write(m, &mut obj, wn(name), DICT, sentinel(v)));
            }
            assert_eq!(obj._mapdict_storage_length(), 3);

            // delete the middle attribute and transplant the rebuilt carrier
            // (mapdict.py deldictvalue).
            let m = obj._get_mapdict_map();
            let new_obj = node_delete(m, &obj, wn("b"), DICT).expect("b present");
            obj._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);

            assert_eq!(obj._mapdict_storage_length(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("a"), DICT), Some(sentinel(0xa)));
            assert_eq!(node_read(m, &obj, wn("b"), DICT), None);
            assert_eq!(node_read(m, &obj, wn("c"), DICT), Some(sentinel(0xc)));

            // deleting an absent attribute returns None
            let m = obj._get_mapdict_map();
            assert!(node_delete(m, &obj, wn("zzz"), DICT).is_none());
        }
    }

    #[test]
    fn instance_node_del_roundtrip() {
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x1)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x2)));
            assert!(instance_node_deldictvalue(obj_ref, wn("x")));
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("x")), None);
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("y")),
                Some(sentinel(0x2))
            );
            // deleting again reports the attribute is gone
            assert!(!instance_node_deldictvalue(obj_ref, wn("x")));
        }
    }

    #[test]
    fn map_dict_strategy_routes_through_instance() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        unsafe {
            // Back the strategy with a real instance whose terminator is
            // pre-installed (ensure_mapdict_initialized is then a no-op, no
            // W_TypeObject needed). mapdict.py:1502 erases the instance as the
            // dict's dstorage.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));

            let w_dict = pyre_object::w_dict_new_with(
                &MAP_DICT_STRATEGY_REF,
                obj_ref as pyre_object::gc_hook::GCREF,
            );

            assert_eq!(MAP_DICT_STRATEGY.strategy_kind(), StrategyKind::Map);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // getitem_str + getitem(text key) both reach the map; a missing key
            // and the never-equal short-circuit return None.
            assert_eq!(
                MAP_DICT_STRATEGY.getitem_str(w_dict, "x"),
                Some(sentinel(0x11))
            );
            let w_key_y = pyre_object::w_str_new("y");
            assert_eq!(
                MAP_DICT_STRATEGY.getitem(w_dict, w_key_y),
                Some(sentinel(0x22))
            );
            let w_key_z = pyre_object::w_str_new("z");
            assert_eq!(MAP_DICT_STRATEGY.getitem(w_dict, w_key_z), None);

            // keys / items / values in insertion order.
            let keys = MAP_DICT_STRATEGY.w_keys(w_dict);
            assert_eq!(keys.len(), 2);
            assert_eq!(pyre_object::w_str_get_wtf8(keys[0]), "x");
            assert_eq!(pyre_object::w_str_get_wtf8(keys[1]), "y");
            let items = MAP_DICT_STRATEGY.items(w_dict);
            assert_eq!(items.len(), 2);
            assert_eq!(pyre_object::w_str_get_wtf8(items[0].0), "x");
            assert_eq!(items[0].1, sentinel(0x11));
            assert_eq!(pyre_object::w_str_get_wtf8(items[1].0), "y");
            assert_eq!(items[1].1, sentinel(0x22));
            assert_eq!(
                MAP_DICT_STRATEGY.values(w_dict),
                vec![sentinel(0x11), sentinel(0x22)]
            );

            // setitem_str grows; delitem(text) shrinks the backing instance.
            MAP_DICT_STRATEGY.setitem_str(w_dict, "z", sentinel(0x33));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 3);
            let w_key_x = pyre_object::w_str_new("x");
            assert!(MAP_DICT_STRATEGY.delitem(w_dict, w_key_x));
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "x"), None);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // clear drops every DICT entry.
            MAP_DICT_STRATEGY.clear(w_dict);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 0);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "y"), None);
        }
    }

    #[test]
    fn map_dict_strategy_popitem_preserves_surrogate_node_name() {
        use pyre_object::dictmultiobject::DictStrategy;

        unsafe {
            // Install the surrogate-named attribute last so the DICT search
            // reaches it first. The mapdict node name is WTF-8; popitem must
            // preserve that name when materialising the returned key.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let mut sur = Wtf8Buf::new();
            sur.push(rustpython_wtf8::CodePoint::from_u32(0xD800).unwrap());

            assert!(instance_node_setdictvalue(
                obj_ref,
                wn("ascii"),
                sentinel(0x44)
            ));
            assert!(instance_node_setdictvalue(obj_ref, &sur, sentinel(0x55)));

            let w_dict = pyre_object::w_dict_new_with(
                &MAP_DICT_STRATEGY_REF,
                obj_ref as pyre_object::gc_hook::GCREF,
            );
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            let (w_sur_key, w_sur_value) = MAP_DICT_STRATEGY.popitem(w_dict).unwrap();
            assert_eq!(pyre_object::w_str_get_wtf8(w_sur_key), &*sur);
            assert_eq!(w_sur_value, sentinel(0x55));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 1);
            assert_eq!(MAP_DICT_STRATEGY.getitem(w_dict, w_sur_key), None);
            assert_eq!(instance_node_getdictvalue(obj_ref, &sur), None);

            let (w_ascii_key, w_ascii_value) = MAP_DICT_STRATEGY.popitem(w_dict).unwrap();
            assert_eq!(pyre_object::w_str_get_wtf8(w_ascii_key), "ascii");
            assert_eq!(w_ascii_value, sentinel(0x44));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 0);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "ascii"), None);
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("ascii")), None);
        }
    }

    #[test]
    fn make_instance_dict_uses_shared_fake_carriers() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};

        crate::test_hooks::install_hash_hook();
        unsafe {
            let first = make_instance_dict();
            let second = make_instance_dict();
            assert_eq!(
                (*(first as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Map
            );
            assert_eq!(MAP_DICT_STRATEGY.length(first), 0);

            // `Object.getdict` stores the wrapper in its SPECIAL slot, so a
            // second lookup must preserve wrapper identity.
            let first_carrier =
                (*(first as *const pyre_object::W_DictObject)).dstorage as PyObjectRef;
            assert_eq!(_obj_getdict(first_carrier), first);
            // The carrier is allocated with `w_class = PY_NULL`, so it reaches
            // `maybe_register_user_finalizer`. Its type resolves through the
            // `INSTANCE_TYPE` fallback, whose `hasuserdel` is false — but the
            // app-level type registry is not initialized in a bare lib test,
            // so `r#type` answers None here and the check only runs once a
            // registry exists.
            if let Some(carrier_type) = crate::typedef::r#type(first_carrier) {
                assert!(
                    !pyre_object::w_type_get_hasuserdel(carrier_type.as_ptr()),
                    "the fake carrier must not enter the user-finalizer queue"
                );
            }

            // The shared terminator keeps mapdict.py:310's `allow_unboxing`
            // default, so every write type-inspects its value. `sentinel` is a
            // bare integer cast to a pointer and would be dereferenced here;
            // this test uses real objects throughout for that reason.
            let first_shared = pyre_object::w_str_new("first");
            let second_shared = pyre_object::w_str_new("second");
            pyre_object::w_dict_setitem_str(first, "shared", first_shared);
            pyre_object::w_dict_setitem_str(second, "shared", second_shared);
            assert_eq!(
                pyre_object::w_dict_getitem_str(first, "shared"),
                Some(first_shared)
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(second, "shared"),
                Some(second_shared)
            );
            let first_node = node_search(
                (&*(first_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            );
            let second_carrier =
                (*(second as *const pyre_object::W_DictObject)).dstorage as PyObjectRef;
            let second_node = node_search(
                (&*(second_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            );
            assert_eq!(first_node, second_node);

            // The process-shared terminator preserves mapdict.py:310's true
            // allow_unboxing default, so an int takes the unboxed transition.
            pyre_object::w_dict_setitem_str(first, "number", pyre_object::w_int_new(42));
            let number_node = node_search(
                (&*(first_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            )
            .unwrap();
            assert!(
                (*number_node).as_plain().unboxed.is_some(),
                "int attribute did not use the shared terminator's unboxed transition"
            );

            // A non-str write devolves while retaining previous string keys.
            MAP_DICT_STRATEGY.setitem(
                first,
                pyre_object::w_int_new(7),
                pyre_object::w_str_new("v"),
            );
            assert_eq!(
                (*(first as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Object
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(first, "shared"),
                Some(first_shared)
            );

            // A separate carrier drives the LIMIT arm, which reads its wrapper
            // back through getdict before changing to a Unicode dictionary.
            let limit_dict = make_instance_dict();
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                pyre_object::w_dict_setitem_str(
                    limit_dict,
                    &format!("k{i}"),
                    pyre_object::w_str_new(&format!("v{i}")),
                );
            }
            assert_eq!(
                (*(limit_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Unicode
            );

            // A new carrier devolves on a non-str key; reads and deletes then
            // use the Devolved read and delete arms' fake-carrier getdict.
            let devolved_dict = make_instance_dict();
            let kept = pyre_object::w_str_new("kept-value");
            pyre_object::w_dict_setitem_str(devolved_dict, "kept", kept);
            MAP_DICT_STRATEGY.setitem(
                devolved_dict,
                pyre_object::w_int_new(8),
                pyre_object::w_str_new("devolve"),
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(devolved_dict, "kept"),
                Some(kept)
            );
            assert!(
                (*(devolved_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .delitem(devolved_dict, pyre_object::w_str_new("kept"))
            );
        }
    }

    #[test]
    fn map_dict_strategy_switch_to_object_materialises() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));

            let w_dict = pyre_object::w_dict_new_with(
                &MAP_DICT_STRATEGY_REF,
                obj_ref as pyre_object::gc_hook::GCREF,
            );
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // A non-str key forces switch_to_object_strategy → materialise.
            MAP_DICT_STRATEGY.switch_to_object_strategy(w_dict);

            // The dict is now ObjectDictStrategy and holds the two str attrs.
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Object);
            assert_eq!(dict.dstrategy.length(w_dict), 2);
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "x"),
                Some(sentinel(0x11))
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "y"),
                Some(sentinel(0x22))
            );

            // The backing instance devolved: its map roots at a
            // DevolvedDictTerminator and the DICT attrs left its storage.
            let inst_map = obj._get_mapdict_map();
            assert!(matches!(
                (*inst_map).as_terminator().kind,
                TerminatorKind::Devolved
            ));
            assert_eq!((*inst_map).storage_needed(), 0);
        }
    }

    #[test]
    fn map_dict_strategy_switch_to_text_materialises() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("a"), sentinel(0x55)));

            let w_dict = pyre_object::w_dict_new_with(
                &MAP_DICT_STRATEGY_REF,
                obj_ref as pyre_object::gc_hook::GCREF,
            );

            // The LIMIT-devolve path (mapdict.py:317-323) switches to text.
            mapdict_switch_to_text_strategy(w_dict);

            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Unicode);
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "a"),
                Some(sentinel(0x55))
            );
            let inst_map = obj._get_mapdict_map();
            assert!(matches!(
                (*inst_map).as_terminator().kind,
                TerminatorKind::Devolved
            ));
        }
    }

    #[test]
    fn instance_custom_trace_walks_storage_without_instance_dict() {
        // An instance's attribute values are forwarded by the per-instance
        // custom trace worker (`instance_walk_boxed_storage`).
        // `instance_node_setdictvalue` writes the attributes through
        // map+storage without calling `getdict`, and the storage walk still
        // visits the value slots.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let v1 = sentinel(0xA1);
            let v2 = sentinel(0xB2);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), v1));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), v2));

            let mut seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| seen.push(*slot));
            assert!(seen.contains(&v1), "x value not walked by custom trace");
            assert!(seen.contains(&v2), "y value not walked by custom trace");
        }
    }

    #[test]
    fn storage_class_mapdict_mixin_bit_admits_layout() {
        use pyre_object::lltype::PyreClassPyTypeOf;
        assert_eq!(
            pyre_object::INSTANCE_TYPE.mapdict_offset,
            std::mem::offset_of!(pyre_object::W_ObjectObject, map)
        );
        assert_eq!(pyre_object::INT_TYPE.mapdict_offset, 0);
        unsafe {
            let obj = pyre_object::w_instance_new(pyre_object::PY_NULL);
            assert!(has_mapdict_layout(obj));
        }
    }

    #[test]
    fn slots_only_native_subclass_has_mapdict_layout_without_routing() {
        unsafe {
            let w_class =
                pyre_object::w_type_new("SlotsOnlyStr", pyre_object::PY_NULL, std::ptr::null_mut());
            assert!(!pyre_object::w_type_get_hasdict(w_class));
            // The class flag defaults false. `setweakref` gates the SPECIAL
            // slot on `TypeDef.weakrefable`, which this slots layout sets.
            pyre_object::w_type_set_weakrefable(w_class, true);
            let obj = pyre_object::w_str_subclass_from_wtf8(wb("abc"), w_class);

            // Auto-assigned type ids are registered by the JIT driver. Unit
            // tests run without it, so stamp the declared user-layout id while
            // preserving the allocation header's flags.
            let header = &mut *majit_gc::header::header_of(obj as usize);
            header.tid_and_flags = (header.tid_and_flags & !majit_gc::header::TYPE_ID_MASK)
                | u64::from(pyre_object::unicodeobject::W_UNICODE_USER_GC_TYPE_ID);

            assert!(has_mapdict_layout(obj));
            assert!(!has_mapdict_storage(obj));
            instance_walk_boxed_storage(obj, &mut |_| {});

            let lifeline = pyre_object::w_instance_new(pyre_object::PY_NULL);
            setweakref(obj, lifeline).unwrap();
            assert_eq!(instance_get_weakref_slot(obj), Some(lifeline));
            delweakref(obj);
            assert_eq!(instance_get_weakref_slot(obj), None);
        }
    }

    #[test]
    fn object_user_layout_routes_slots_without_hasdict() {
        // `W_ObjectObjectUserDictWeakrefable` keeps `MapdictStorageMixin`
        // for a slots-only class (`hasdict` false). A slots-only native
        // subclass does not: that is
        // `slots_only_native_subclass_has_mapdict_layout_without_routing`.
        unsafe {
            let w_class =
                pyre_object::w_type_new("SlotsOnly", pyre_object::PY_NULL, std::ptr::null_mut());
            assert!(!pyre_object::w_type_get_hasdict(w_class));
            let obj = pyre_object::w_instance_new(w_class);
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &pyre_object::pyobject::INSTANCE_USER_TYPE
            ));
            assert!(has_mapdict_layout(obj));
            assert!(has_mapdict_storage(obj));
        }
    }

    #[test]
    fn instance_dict_wrapper_in_special_slot_not_instance_dict() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // An instance's `__dict__` wrapper is stored in the
        // mapdict "dict" SPECIAL slot (mapdict.py `_obj_getdict`). Repeated
        // access returns the same wrapper, and the SPECIAL slot is excluded
        // from the `__dict__` view.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w1 = _obj_getdict(obj_ref);
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w1));
            // identity stable across repeated access.
            let w2 = _obj_getdict(obj_ref);
            assert_eq!(w1, w2);
            // a fresh wrapper is a MapDictStrategy view.
            let dict = &*(w1 as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Map);

            // a DICT attribute is visible in the view; the SPECIAL "dict" slot
            // (storing the wrapper) is excluded from the view.
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x1)));
            assert_eq!(MAP_DICT_STRATEGY.length(w1), 1);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w1, "x"), Some(sentinel(0x1)));
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w1, "dict"), None);
        }
    }

    #[test]
    fn obj_getdict_rebuilds_the_view_when_the_dict_slot_reads_null() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // mapdict.py `if w_dict is not None` — RPython's `read` answers
        // None both for an absent slot and for one holding None, and both mean
        // "build the view".  Surfacing the null instead makes `getdict` report
        // that the receiver has no dict, which is the single state
        // `descr__setattr__` turns into
        // `"'%T' object attribute '%s' is read-only"` — for any name the type
        // carries, a class-variable default being enough to arm that branch.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w1 = _obj_getdict(obj_ref);
            assert!(!w1.is_null());
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w1));

            // The state a write that never landed leaves behind: the map still
            // claims the SPECIAL attribute, its storage slot holds NULL.
            assert!(instance_set_dict_slot(obj_ref, pyre_object::PY_NULL));
            assert_eq!(instance_get_dict_slot(obj_ref), Some(pyre_object::PY_NULL));

            let w2 = _obj_getdict(obj_ref);
            assert!(!w2.is_null(), "a null SPECIAL slot must rebuild the view");
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w2));
            let dict = &*(w2 as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Map);
            // The rebuilt view backs onto the instance it was built for.
            assert_eq!(dict.dstorage as PyObjectRef, obj_ref);
        }
    }

    #[test]
    fn instance_custom_trace_walks_special_wrapper_and_values() {
        // The wrapper stored in the "dict" SPECIAL slot is forwarded by the
        // instance custom trace (it is one of the storage slots), and a
        // non-devolved view's DICT values are forwarded directly from storage.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w_dict = _obj_getdict(obj_ref);
            let v1 = sentinel(0xC1);
            let v2 = sentinel(0xC2);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), v1));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), v2));

            let mut seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| seen.push(*slot));
            assert!(seen.contains(&w_dict), "SPECIAL-slot wrapper not walked");
            assert!(seen.contains(&v1), "x value not walked");
            assert!(seen.contains(&v2), "y value not walked");
        }
    }

    #[test]
    fn instance_custom_trace_and_wrapper_cover_devolved_values() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // UAF-prevention case: once an instance devolves (>= LIMIT DICT attrs,
        // mapdict.py), its materialised DICT values move into the
        // wrapper's own backing storage and leave the instance storage. The
        // wrapper stays in the "dict" SPECIAL slot. Coverage is now two-layer:
        // the instance custom trace forwards the wrapper pointer, and the
        // wrapper's own GC custom trace (`strategy.walk_gc_refs`) forwards its
        // entry values — together covering every devolved value, with no
        // INSTANCE_REGISTRY walk.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert!(instance_node_setdictvalue(
                    obj_ref,
                    wn(&name),
                    sentinel(0x2000 + i)
                ));
            }
            // The LIMIT write devolved the instance via obj.getdict(); the
            // wrapper now lives in the SPECIAL slot with a non-Map strategy.
            let w_dict = instance_get_dict_slot(obj_ref).expect("wrapper in SPECIAL slot");
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_ne!(dict.dstrategy.strategy_kind(), StrategyKind::Map);

            // (1) the instance custom trace forwards the wrapper pointer.
            let mut storage_seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| storage_seen.push(*slot));
            assert!(
                storage_seen.contains(&w_dict),
                "devolved wrapper not walked by instance custom trace"
            );

            // (2) the wrapper's own custom trace forwards every entry value.
            let mut wrapper_seen: Vec<PyObjectRef> = Vec::new();
            dict.dstrategy
                .walk_gc_refs(w_dict, &mut |slot| wrapper_seen.push(*slot));
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                assert!(
                    wrapper_seen.contains(&sentinel(0x2000 + i)),
                    "devolved value k{i} not walked by wrapper custom trace"
                );
            }
        }
    }

    #[test]
    fn write_terminator_devolves_at_limit_map_attributes() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            // mapdict.py: the (LIMIT_MAP_ATTRIBUTES)th DICT write on a
            // non-devolved instance auto-devolves its `__dict__` to text strategy.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert!(instance_node_setdictvalue(
                    obj_ref,
                    wn(&name),
                    sentinel(0x1000 + i)
                ));
            }

            // _write_terminator's LIMIT branch fetched obj.getdict() (the
            // MapDictStrategy view installed by _obj_getdict) and devolved it.
            let w_dict = _obj_getdict(obj_ref);
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Unicode);

            // every attribute survived the devolve, read through the dict view.
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert_eq!(
                    pyre_object::w_dict_getitem_str(w_dict, &name),
                    Some(sentinel(0x1000 + i))
                );
            }

            // The backing instance devolved: its map roots at a
            // DevolvedDictTerminator. The "dict" SPECIAL slot (the wrapper,
            // written by obj.getdict() during the LIMIT devolve) is kept on the
            // rebuilt carrier (mapdict.py keeps non-DICT attrs), so the
            // outermost node is that PlainAttribute and the terminator is reached
            // via `.terminator()`. Only the SPECIAL slot survives on storage.
            let inst_map = obj._get_mapdict_map();
            let inst_term = (*inst_map).terminator();
            assert!(matches!(
                (*inst_term).as_terminator().kind,
                TerminatorKind::Devolved
            ));
            assert_eq!((*inst_map).storage_needed(), 1);
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w_dict));
        }
    }

    #[test]
    fn add_attr_interns_shared_transition() {
        unsafe {
            // two independent objects adding the same attribute from the same
            // map must converge on the same child map (interned transition)
            let term = boxed_dict_terminator();
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("p"), DICT, sentinel(1));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("p"), DICT, sentinel(2));
            assert_eq!(o1.map, o2.map);
        }
    }

    #[test]
    fn out_of_order_insert_reorders_to_canonical_map() {
        unsafe {
            let term = boxed_dict_terminator();
            // o1 establishes the canonical insertion order a, b
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("a"), DICT, sentinel(0xa1));
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("b"), DICT, sentinel(0xb1));

            // o2 inserts b first, then a — adding "a" must trigger
            // _reorder_and_add (a lower-order ancestor already has "a") and
            // converge on the same canonical map as o1
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("b"), DICT, sentinel(0xb2));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("a"), DICT, sentinel(0xa2));

            // values preserved through the reorder
            let m = o2._get_mapdict_map();
            assert_eq!(node_read(m, &o2, wn("a"), DICT), Some(sentinel(0xa2)));
            assert_eq!(node_read(m, &o2, wn("b"), DICT), Some(sentinel(0xb2)));
            assert_eq!(o2.storage.len(), 2);
            // reordered to the canonical (insertion-ordered) map shared with o1
            assert_eq!(o2.map, o1.map);
        }
    }

    // ── UnboxedPlainAttribute (mapdict.py) ────────────────────

    #[test]
    fn unboxed_int_attribute_stores_and_reads() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            // an int written to an unboxing-allowed map becomes an
            // UnboxedPlainAttribute
            assert!(node_write(
                m,
                &mut obj,
                wn("x"),
                DICT,
                pyre_object::w_int_new(42)
            ));
            let p = (*obj.map).as_plain();
            assert!(p.unboxed.is_some());
            assert!(p.unboxed.as_ref().unwrap().firstunwrapped);
            assert_eq!(p.unboxed.as_ref().unwrap().typ, UnboxType::Int);
            // a single storage slot holds the erased longlong list (not a box)
            assert_eq!(obj.storage.len(), 1);
            // reading boxes the longlong back into an int of the same value
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, wn("x"), DICT).unwrap();
            assert!(pyre_object::is_int(r));
            assert_eq!(pyre_object::w_int_get_value(r), 42);
        }
    }

    #[test]
    fn two_unboxed_ints_share_one_storage_slot() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(10));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("y"), DICT, pyre_object::w_int_new(20));
            // both pack into a single shared longlong list (storageindex 0,
            // listindex 0 and 1) — storage does not grow for the second one
            assert_eq!(obj.storage.len(), 1);
            let p = (*obj.map).as_plain();
            assert_eq!(p.storageindex, 0);
            assert_eq!(p.unboxed.as_ref().unwrap().listindex, 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                10
            );
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("y"), DICT).unwrap()),
                20
            );
        }
    }

    #[test]
    fn unboxed_int_and_float_share_slot_with_correct_boxing() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("i"), DICT, pyre_object::w_int_new(7));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("f"), DICT, pyre_object::w_float_new(2.5));
            // int and float pack into one longlong list (the float as its bits)
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            let ri = node_read(m, &obj, wn("i"), DICT).unwrap();
            let rf = node_read(m, &obj, wn("f"), DICT).unwrap();
            // each is re-boxed to its own type
            assert!(pyre_object::is_int(ri));
            assert_eq!(pyre_object::w_int_get_value(ri), 7);
            assert!(pyre_object::is_float(rf));
            assert_eq!(pyre_object::w_float_get_value(rf), 2.5);
        }
    }

    #[test]
    fn unboxed_overwrite_same_type_updates_in_place() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(1));
            let map_after_first = obj.map;
            // a same-typed overwrite updates the longlong in place: no map
            // transition, no storage growth
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(2));
            assert_eq!(obj.map, map_after_first);
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                2
            );
        }
    }

    #[test]
    fn unboxed_float_preserves_bits() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let v = -3.25_f64;
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("f"), DICT, pyre_object::w_float_new(v));
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, wn("f"), DICT).unwrap();
            assert!(pyre_object::is_float(r));
            assert_eq!(pyre_object::w_float_get_value(r), v);
        }
    }

    #[test]
    fn read_migrates_to_boxed_when_unboxing_frozen() {
        unsafe {
            // "x" is stored unboxed under a (default unboxing-on) terminator.
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(10));
            assert!((*obj.map).as_plain().unboxed.is_some());
            // the class becomes type-unstable: freeze unboxing for its terminator.
            (*term).as_terminator().set_allow_unboxing(false);
            // mapdict.py — a read now lazily migrates obj to boxed storage.
            let m = obj._get_mapdict_map();
            let attr = find_map_attr(m, wn("x"), DICT).unwrap();
            migrate_to_boxed_if_frozen(attr, &mut obj);
            // the rebuilt map's attribute is boxed; the value is preserved.
            assert!((*obj.map).as_plain().unboxed.is_none());
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                10
            );
        }
    }

    /// `typedef.py` `_getusercls`: an enumerate subclass instance is
    /// `W_EnumerateUser` (`ENUMERATE_USER_TYPE`) and carries mapdict storage.
    #[test]
    fn enumerate_subclass_instance_carries_user_typeptr() {
        let w_subtype =
            pyre_object::w_type_new("EnumerateSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let obj = pyre_object::functional::w_enumerate_new(
            pyre_object::PY_NULL,
            0,
            pyre_object::PY_NULL,
            w_subtype,
        );
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &pyre_object::functional::ENUMERATE_USER_TYPE
            ));
            assert!(has_mapdict_layout(obj));
            assert!(pyre_object::functional::is_enumerate(obj));
        }
    }

    /// `typedef.py` `_getusercls`: a count subclass instance is
    /// `W_CountUser` (`COUNT_USER_TYPE`) and carries mapdict storage.
    #[test]
    fn count_subclass_instance_carries_user_typeptr() {
        let w_subtype =
            pyre_object::w_type_new("CountSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let obj = pyre_object::interp_itertools::w_count_new(
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            w_subtype,
        );
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &pyre_object::interp_itertools::COUNT_USER_TYPE
            ));
            assert!(has_mapdict_layout(obj));
            assert!(pyre_object::interp_itertools::is_count(obj));
        }
    }

    /// `typedef.py` `_getusercls`: a batched subclass instance is
    /// `W_BatchedUser` (`BATCHED_USER_TYPE`) and carries mapdict storage.
    #[test]
    fn batched_subclass_instance_carries_user_typeptr() {
        let w_subtype =
            pyre_object::w_type_new("BatchedSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let obj =
            pyre_object::interp_itertools::w_batched_new(pyre_object::PY_NULL, 0, false, w_subtype);
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &pyre_object::interp_itertools::BATCHED_USER_TYPE
            ));
            assert!(has_mapdict_layout(obj));
            assert!(pyre_object::interp_itertools::is_batched(obj));
        }
    }
}
