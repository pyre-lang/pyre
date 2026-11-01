//! Interned mapdict node types (`mapdict.py` `AbstractAttribute`).
//!
//! The map pointer is a field of the instance layout (`MapdictStorageMixin.map`);
//! `Terminator.w_cls` is the Python class (`BaseUserClassMapdict.getclass`).
//! Algorithms that intern and rewrite maps stay in the interpreter mapdict
//! layer; this crate owns the node layout so `w_instance_get_type` can read
//! the class where the data lives.

use crate::lltype::{GcType, TypeIdCell};
use crate::pyobject::PyObjectRef;
use crate::quasiimmut::{QuasiImmut, QuasiImmutLock};
use parking_lot::Mutex;
use rustpython_wtf8::Wtf8Buf;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::AtomicPtr;

/// rclass `OBJECT.typeptr` analog for mapdict classes (`rclass.py`).
/// Downcasts test it (`isinstance` + `cast_pointer`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum MapClass {
    /// `mapdict.py` `Terminator`.
    Terminator = 0,
    /// `mapdict.py` `PlainAttribute`.
    PlainAttribute = 1,
}

/// A shared, interned map node (`mapdict.py` `AbstractAttribute`).
///
/// GC pointer to the object start of a [`Terminator`] or [`PlainAttribute`].
/// Identity compare (`ptr::eq`) remains valid after a moving collection
/// because both sides are forwarded. The parent struct is first-field-inlined
/// (`lltype.Struct._first_struct` / `_note_inlined_into`), so an upcast or
/// downcast (`cast_pointer`) keeps this address.
pub type MapRef = *const AbstractAttribute;

/// mapdict.py `AbstractAttribute` — the shared prefix of every map node.
///
/// Never allocated on its own. [`Terminator`] and [`PlainAttribute`] inline
/// this as their first field (`GcStruct._first_struct`).
/// `AbstractAttribute._immutable_fields_ = ['terminator']`.
/// `AbstractAttribute._attrs_ = ['terminator', 'space', 'cache_attrs']`
/// (`space` is omitted: pyre's object space is ambient).
#[repr(C)]
#[majit_macros::jit_immutable_fields("terminator")]
pub struct AbstractAttribute {
    /// rclass `OBJECT.typeptr`. Distinguishes Terminator vs PlainAttribute.
    pub class: MapClass,
    /// mapdict.py `AbstractAttribute.terminator`.
    pub terminator: MapRef,
    /// mapdict.py `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`. Starts as null
    /// (`cache_attrs = None`) and is interned to `{}` on first write.
    pub cache_attrs: AtomicPtr<Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>>,
}

/// mapdict.py `DictTerminator` / `NoDictTerminator` / `DevolvedDictTerminator`.
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
///
/// `base` is the first-field-inlined parent (`GcStruct._first_struct`).
/// `mapdict.py Terminator._immutable_fields_ = ['w_cls', 'allow_unboxing?']`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("w_cls", "allow_unboxing?")]
pub struct Terminator {
    /// `AbstractAttribute` prefix. Same address as the Terminator object.
    pub base: AbstractAttribute,
    /// mapdict.py `Terminator.__init__` `w_cls`.
    pub w_cls: PyObjectRef,
    /// mapdict.py `_immutable_fields_` declares `allow_unboxing?`;
    /// `Terminator.allow_unboxing` permits unboxed attributes until a
    /// differently-typed reassignment freezes them off. [`Self::set_allow_unboxing`]
    /// invalidates watchers before changing it.
    pub allow_unboxing: Cell<bool>,
    /// Hidden `mutate_allow_unboxing` field (`quasiimmut.py
    /// get_mutate_field_name` of pyre `allow_unboxing`; `mapdict.py
    /// Terminator._immutable_fields_ = ['w_cls', 'allow_unboxing?']`). The
    /// owner is a GC [`Terminator`]. An `AtomicPtr` plus a lock, no GC
    /// pointers, reclaimed by [`terminator_destructor`].
    pub mutate_allow_unboxing: std::sync::atomic::AtomicPtr<QuasiImmut>,
    pub mutate_allow_unboxing_lock: std::sync::atomic::AtomicPtr<QuasiImmutLock>,
    /// Which Terminator subclass this is.
    pub kind: TerminatorKind,
    /// mapdict.py `DictTerminator.devolved_dict_terminator` (null unless
    /// `kind == Dict`).
    pub devolved_dict_terminator: Cell<MapRef>,
}

crate::quasi_immut_words!(
    Terminator,
    mutate_allow_unboxing,
    mutate_allow_unboxing_lock
);

impl Drop for Terminator {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe {
            self.mutate_allow_unboxing().reclaim();
        }
    }
}

impl Terminator {
    #[inline]
    pub fn mutate_allow_unboxing(&self) -> crate::quasiimmut::QuasiImmutField<'_> {
        crate::quasiimmut::QuasiImmutField::new(
            &self.mutate_allow_unboxing,
            &self.mutate_allow_unboxing_lock,
        )
    }

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
        if self.mutate_allow_unboxing().is_installed() {
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_allow_unboxing) };
        }
        self.allow_unboxing.set(v);
    }

    pub fn current_allow_unboxing_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.mutate_allow_unboxing().get_current_qmut_instance()
    }

    pub fn allow_unboxing_qmut_installed(&self) -> bool {
        self.mutate_allow_unboxing().is_installed()
    }

    pub fn force_allow_unboxing_qmut(&self) {
        self.mutate_allow_unboxing().invalidate();
    }
}

/// `quasiimmut.py get_current_qmut_instance` for a Terminator's `allow_unboxing?` field.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_current_allow_unboxing_qmut(
    terminator: *const Terminator,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
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

/// The unbox type of an `UnboxedPlainAttribute` (`UnboxedPlainAttribute.typ`,
/// `space.IntObjectCls` / `space.FloatObjectCls`). pyre exposes no per-type
/// impl-class object, so the int/float distinction is captured by this enum
/// and resolved through `is_int`/`is_float`.
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
///
/// `base` is the first-field-inlined parent (`GcStruct._first_struct`).
/// `mapdict.py PlainAttribute._immutable_fields_ = ['name', 'attrkind',
/// 'storageindex', '_num_attributes', 'back', 'ever_mutated?', 'order']`.
/// The `?` entry is declared here; the non-`?` entries are not yet
/// (`name`/`attrkind`/`storageindex`/`num_attributes`/`back`/`order`).
#[repr(C)]
#[majit_macros::jit_immutable_fields("ever_mutated?")]
pub struct PlainAttribute {
    /// `AbstractAttribute` prefix. Same address as the PlainAttribute object.
    pub base: AbstractAttribute,
    /// mapdict.py `PlainAttribute.name` (utf8-encoded).
    pub name: Wtf8Buf,
    /// mapdict.py `PlainAttribute.attrkind`.
    pub attrkind: u16,
    /// mapdict.py `storageindex` (= `back.storage_needed()`).
    pub storageindex: usize,
    /// mapdict.py `_num_attributes` (= `back.num_attributes() + 1`).
    pub num_attributes: usize,
    /// mapdict.py `PlainAttribute.back`.
    pub back: MapRef,
    /// mapdict.py `_immutable_fields_` declares `ever_mutated?`;
    /// `PlainAttribute.ever_mutated` records whether this attribute has ever
    /// been overwritten or deleted. [`Self::set_ever_mutated`] invalidates
    /// watchers before changing it.
    pub ever_mutated: Cell<bool>,
    /// Hidden `mutate_ever_mutated` field (`quasiimmut.py
    /// get_mutate_field_name` of pyre `ever_mutated`; `mapdict.py
    /// PlainAttribute._immutable_fields_` `ever_mutated?`). The owner is a
    /// GC [`PlainAttribute`]. An `AtomicPtr` plus a lock, no GC pointers,
    /// reclaimed by [`plain_attribute_destructor`].
    pub mutate_ever_mutated: std::sync::atomic::AtomicPtr<QuasiImmut>,
    pub mutate_ever_mutated_lock: std::sync::atomic::AtomicPtr<QuasiImmutLock>,
    /// mapdict.py `PlainAttribute.order`.
    pub order: usize,
    /// `Some` for an `UnboxedPlainAttribute` (mapdict.py); `None` for a
    /// plain boxed attribute.
    pub unboxed: Option<UnboxedExtra>,
}

crate::quasi_immut_words!(
    PlainAttribute,
    mutate_ever_mutated,
    mutate_ever_mutated_lock
);

impl Drop for PlainAttribute {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe {
            self.mutate_ever_mutated().reclaim();
        }
    }
}

impl PlainAttribute {
    #[inline]
    pub fn mutate_ever_mutated(&self) -> crate::quasiimmut::QuasiImmutField<'_> {
        crate::quasiimmut::QuasiImmutField::new(
            &self.mutate_ever_mutated,
            &self.mutate_ever_mutated_lock,
        )
    }

    /// Change mapdict.py:430 `ever_mutated` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_ever_mutated(&self, v: bool) {
        if self.ever_mutated.get() == v {
            return;
        }
        if self.mutate_ever_mutated().is_installed() {
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_ever_mutated) };
        }
        self.ever_mutated.set(v);
    }

    pub fn current_ever_mutated_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.mutate_ever_mutated().get_current_qmut_instance()
    }

    pub fn ever_mutated_qmut_installed(&self) -> bool {
        self.mutate_ever_mutated().is_installed()
    }

    pub fn force_ever_mutated_qmut(&self) {
        self.mutate_ever_mutated().invalidate();
    }
}

/// `quasiimmut.py get_current_qmut_instance` for a PlainAttribute's `ever_mutated?` field.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_current_ever_mutated_qmut(
    attribute: *const PlainAttribute,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
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

/// GC tid for [`Terminator`], assigned at `register_type`.
pub static TERMINATOR_GC_TYPE_ID: TypeIdCell = TypeIdCell::auto();

/// GC tid for [`PlainAttribute`], assigned at `register_type`.
pub static PLAIN_ATTRIBUTE_GC_TYPE_ID: TypeIdCell = TypeIdCell::auto();

impl GcType for Terminator {
    fn type_id() -> u32 {
        TERMINATOR_GC_TYPE_ID.get()
    }
    const SIZE: usize = std::mem::size_of::<Self>();
}

impl GcType for PlainAttribute {
    fn type_id() -> u32 {
        PLAIN_ATTRIBUTE_GC_TYPE_ID.get()
    }
    const SIZE: usize = std::mem::size_of::<Self>();
}

impl AbstractAttribute {
    /// `cast_pointer` to [`Terminator`] at the same address.
    ///
    /// The class test is `isinstance` ([`Self::is_terminator`]); this is
    /// the identity downcast (`lltype.cast_pointer` / `GcStruct._first_struct`).
    pub fn as_terminator(&self) -> &Terminator {
        debug_assert!(self.is_terminator());
        unsafe { &*(self as *const Self as *const Terminator) }
    }

    /// Mutable `cast_pointer` to [`Terminator`] at the same address.
    pub fn as_terminator_mut(&mut self) -> &mut Terminator {
        debug_assert!(self.is_terminator());
        unsafe { &mut *(self as *mut Self as *mut Terminator) }
    }

    /// `cast_pointer` to [`PlainAttribute`] at the same address.
    ///
    /// The class test is `isinstance` ([`Self::is_plain`]); this is the
    /// identity downcast (`lltype.cast_pointer` / `GcStruct._first_struct`).
    pub fn as_plain(&self) -> &PlainAttribute {
        debug_assert!(self.is_plain());
        unsafe { &*(self as *const Self as *const PlainAttribute) }
    }

    /// Mutable `cast_pointer` to [`PlainAttribute`] at the same address.
    pub fn as_plain_mut(&mut self) -> &mut PlainAttribute {
        debug_assert!(self.is_plain());
        unsafe { &mut *(self as *mut Self as *mut PlainAttribute) }
    }

    /// `isinstance(self, PlainAttribute)`.
    pub fn is_plain(&self) -> bool {
        self.class == MapClass::PlainAttribute
    }

    /// `isinstance(self, Terminator)`.
    pub fn is_terminator(&self) -> bool {
        self.class == MapClass::Terminator
    }

    /// mapdict.py `AbstractAttribute.storage_needed` /
    /// `PlainAttribute.storage_needed` / `UnboxedPlainAttribute.storage_needed`.
    pub fn storage_needed(&self) -> usize {
        if !self.is_plain() {
            return 0;
        }
        let p = self.as_plain();
        // `UnboxedPlainAttribute.storage_needed`: an unboxed attribute only
        // adds a slot when it is the first unboxed one in its slot
        // (`firstunwrapped`); otherwise it packs into the slot the prior
        // unboxed attribute already reserved, so its size is
        // `back.storage_needed()`.
        match &p.unboxed {
            Some(u) => {
                if u.firstunwrapped {
                    p.storageindex + 1
                } else {
                    unsafe { (*p.back).storage_needed() }
                }
            }
            None => p.storageindex + 1,
        }
    }

    /// mapdict.py `Terminator.num_attributes` / `PlainAttribute.num_attributes`.
    pub fn num_attributes(&self) -> usize {
        if self.is_plain() {
            self.as_plain().num_attributes
        } else {
            0
        }
    }
}

/// mapdict.py `CachedAttributeHolder` — caches the child map produced
/// by adding `(name, attrkind)` to a parent map, so transitions are shared.
/// A moving GC instance (`mapdict.py` `CachedAttributeHolder`).
///
/// `repr(C)` pins field offsets to declaration order so the JIT descr
/// group's `offset_of!` matches the walker arg0 (the holder itself).
/// `mapdict.py CachedAttributeHolder._immutable_fields_ = ['attr?', 'typ?']`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("attr?", "typ?")]
pub struct CachedAttributeHolder {
    /// mapdict.py:670 `order` (= number of prior children of `back`).
    pub order: usize,
    /// mapdict.py:669 `_immutable_fields_` declares `attr?`; mapdict.py:675
    /// `attr` is the cached child map. [`Self::set_attr`] invalidates watchers
    /// before changing it.
    pub attr: Cell<MapRef>,
    /// Hidden `mutate_attr` field (`quasiimmut.py get_mutate_field_name` of
    /// pyre `attr`; `mapdict.py CachedAttributeHolder._immutable_fields_ =
    /// ['attr?', 'typ?']`). The owner is a GC [`CachedAttributeHolder`].
    /// An `AtomicPtr` plus a lock, no GC pointers, reclaimed by
    /// [`cached_attribute_holder_destructor`].
    pub mutate_attr: std::sync::atomic::AtomicPtr<QuasiImmut>,
    pub mutate_attr_lock: std::sync::atomic::AtomicPtr<QuasiImmutLock>,
    /// mapdict.py:669 `_immutable_fields_` declares `typ?`; mapdict.py:676
    /// `typ` is the unbox type (`None` means boxed). [`Self::set_typ`]
    /// invalidates watchers before changing it.
    pub typ: Cell<Option<UnboxType>>,
    /// Hidden `mutate_typ` field (`quasiimmut.py get_mutate_field_name` of
    /// pyre `typ`; `mapdict.py CachedAttributeHolder._immutable_fields_ =
    /// ['attr?', 'typ?']`). The owner is a GC [`CachedAttributeHolder`].
    /// An `AtomicPtr` plus a lock, no GC pointers, reclaimed by
    /// [`cached_attribute_holder_destructor`].
    pub mutate_typ: std::sync::atomic::AtomicPtr<QuasiImmut>,
    pub mutate_typ_lock: std::sync::atomic::AtomicPtr<QuasiImmutLock>,
}

crate::quasi_immut_words!(CachedAttributeHolder, mutate_attr, mutate_attr_lock);
crate::quasi_immut_words!(CachedAttributeHolder, mutate_typ, mutate_typ_lock);

impl Drop for CachedAttributeHolder {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe {
            self.mutate_attr().reclaim();
            self.mutate_typ().reclaim();
        }
    }
}

impl CachedAttributeHolder {
    #[inline]
    pub fn mutate_attr(&self) -> crate::quasiimmut::QuasiImmutField<'_> {
        crate::quasiimmut::QuasiImmutField::new(&self.mutate_attr, &self.mutate_attr_lock)
    }

    #[inline]
    pub fn mutate_typ(&self) -> crate::quasiimmut::QuasiImmutField<'_> {
        crate::quasiimmut::QuasiImmutField::new(&self.mutate_typ, &self.mutate_typ_lock)
    }

    /// Change mapdict.py:675 `attr` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_attr(&self, a: MapRef) {
        if self.attr.get() == a {
            return;
        }
        if self.mutate_attr().is_installed() {
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_attr) };
        }
        // `rgc.ll_writebarrier` on the holder before storing `attr`
        // (`mapdict.py CachedAttributeHolder.attr`).
        let _roots = crate::gc_roots::push_roots();
        let base =
            crate::gc_roots::pin_roots(&[self as *const Self as PyObjectRef, a as PyObjectRef]);
        crate::gc_hook::try_gc_write_barrier(
            crate::gc_roots::shadow_stack_get(base) as crate::gc_hook::GCREF
        );
        let owner = crate::gc_roots::shadow_stack_get(base) as *const Self;
        let a = crate::gc_roots::shadow_stack_get(base + 1) as MapRef;
        unsafe { (*owner).attr.set(a) };
    }

    /// Change mapdict.py:676 `typ` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_typ(&self, t: Option<UnboxType>) {
        if self.typ.get() == t {
            return;
        }
        if self.mutate_typ().is_installed() {
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_typ) };
        }
        self.typ.set(t);
    }

    pub fn current_attr_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.mutate_attr().get_current_qmut_instance()
    }

    pub fn attr_qmut_installed(&self) -> bool {
        self.mutate_attr().is_installed()
    }

    pub fn force_attr_qmut(&self) {
        self.mutate_attr().invalidate();
    }

    pub fn current_typ_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.mutate_typ().get_current_qmut_instance()
    }

    pub fn typ_qmut_installed(&self) -> bool {
        self.mutate_typ().is_installed()
    }

    pub fn force_typ_qmut(&self) {
        self.mutate_typ().invalidate();
    }
}

/// GC tid for [`CachedAttributeHolder`], assigned at `register_type`.
pub static CACHED_ATTRIBUTE_HOLDER_GC_TYPE_ID: TypeIdCell = TypeIdCell::auto();

impl GcType for CachedAttributeHolder {
    fn type_id() -> u32 {
        CACHED_ATTRIBUTE_HOLDER_GC_TYPE_ID.get()
    }
    const SIZE: usize = std::mem::size_of::<Self>();
}

/// `quasiimmut.py get_current_qmut_instance` for a CachedAttributeHolder's `attr?` field.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_current_attr_qmut(
    holder: *const CachedAttributeHolder,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
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
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
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

/// `mapdict.py` `MapdictStorageMixin`: the `map` + `storage` tail mixed
/// into an instance layout (`import_from_mixin`). Not itself a `GcType`;
/// the owner object is. `map` is a GCREF (`MapdictStorageMixin.map`).
#[repr(C)]
pub struct MapdictStorageMixin {
    pub map: MapRef,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(std::mem::offset_of!(MapdictStorageMixin, storage) == std::mem::size_of::<MapRef>());
};

/// Per-class copies of `MapdictStorageMixin._get_mapdict_map` /
/// `_set_mapdict_map` / `_mapdict_read_storage` field accessors
/// (`typedef.py` `_getusercls` `import_from_mixin`). Stored on the typeptr
/// so a crate that cannot name the carrier still dispatches to that
/// class's typed `self.map` / `self.storage` (`W_Root.getclass` overridden
/// on the generated class).
#[derive(Clone, Copy)]
pub struct MapdictMixinAccess {
    pub get_map: unsafe fn(crate::PyObjectRef) -> MapRef,
    pub set_map: unsafe fn(crate::PyObjectRef, MapRef),
    pub get_storage: unsafe fn(crate::PyObjectRef) -> *mut crate::object_array::ItemsBlock,
    pub set_storage: unsafe fn(crate::PyObjectRef, *mut crate::object_array::ItemsBlock),
}

/// Typed `self.map` / `self.storage` accessors for one carrier layout.
#[macro_export]
macro_rules! mapdict_mixin_access {
    ($ty:ty) => {
        $crate::mapdict::MapdictMixinAccess {
            get_map: {
                unsafe fn get_map(obj: $crate::PyObjectRef) -> $crate::mapdict::MapRef {
                    unsafe { (*(obj as *const $ty)).map }
                }
                get_map
            },
            set_map: {
                unsafe fn set_map(obj: $crate::PyObjectRef, map: $crate::mapdict::MapRef) {
                    unsafe {
                        (*(obj as *mut $ty)).map = map;
                    }
                }
                set_map
            },
            get_storage: {
                unsafe fn get_storage(
                    obj: $crate::PyObjectRef,
                ) -> *mut $crate::object_array::ItemsBlock {
                    unsafe { (*(obj as *const $ty)).storage }
                }
                get_storage
            },
            set_storage: {
                unsafe fn set_storage(
                    obj: $crate::PyObjectRef,
                    storage: *mut $crate::object_array::ItemsBlock,
                ) {
                    unsafe {
                        (*(obj as *mut $ty)).storage = storage;
                    }
                }
                set_storage
            },
        }
    };
}

/// Select the carrier's own `map` / `storage` fields.
///
/// `typedef.py` `_getusercls` `import_from_mixin(MapdictStorageMixin)` copies
/// those fields onto each generated class; this match is how pyre names
/// the same per-class layout from a `PyObjectRef` (`mapdict_map_descr`).
macro_rules! mapdict_typed_carrier {
    ($obj:expr, $p:ident => $body:expr, else => $fallback:expr) => {{
        let obj = $obj;
        unsafe {
            if crate::py_type_check(obj, &crate::INSTANCE_USER_TYPE)
                || crate::py_type_check(obj, &crate::INSTANCE_TYPE)
            {
                let $p = obj as *mut crate::objectobject::W_ObjectObject;
                $body
            } else if crate::py_type_check(obj, &crate::INT_USER_TYPE) {
                let $p = obj as *mut crate::W_IntObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::FLOAT_USER_TYPE) {
                let $p = obj as *mut crate::W_FloatObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::BYTES_USER_TYPE) {
                let $p = obj as *mut crate::W_BytesObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::BYTEARRAY_USER_TYPE) {
                let $p = obj as *mut crate::W_BytearrayObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::LIST_USER_TYPE) {
                let $p = obj as *mut crate::W_ListObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::SET_USER_TYPE)
                || crate::py_type_check(obj, &crate::FROZENSET_USER_TYPE)
            {
                let $p = obj as *mut crate::W_SetObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::interp_array::ARRAY_USER_TYPE) {
                let $p = obj as *mut crate::interp_array::W_ArrayUser;
                $body
            } else if crate::py_type_check(obj, &crate::weakref::WEAKREF_LAYOUT_USER_TYPE) {
                let $p = obj as *mut crate::weakref::W_WeakrefUser;
                $body
            } else if crate::py_type_check(obj, &crate::COMPLEX_USER_TYPE) {
                let $p = obj as *mut crate::W_ComplexObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::STR_USER_TYPE) {
                let $p = obj as *mut crate::W_UnicodeObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::TUPLE_USER_TYPE) {
                let $p = obj as *mut crate::W_TupleObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::ENUMERATE_USER_TYPE) {
                let $p = obj as *mut crate::W_EnumerateUser;
                $body
            } else if crate::py_type_check(obj, &crate::MAP_USER_TYPE) {
                let $p = obj as *mut crate::W_MapUser;
                $body
            } else if crate::py_type_check(obj, &crate::FILTER_USER_TYPE) {
                let $p = obj as *mut crate::W_FilterUser;
                $body
            } else if crate::py_type_check(obj, &crate::ZIP_USER_TYPE) {
                let $p = obj as *mut crate::W_ZipUser;
                $body
            } else if crate::py_type_check(obj, &crate::REVERSED_USER_TYPE) {
                let $p = obj as *mut crate::W_ReversedIteratorUser;
                $body
            } else if crate::py_type_check(obj, &crate::SUPER_USER_TYPE) {
                let $p = obj as *mut crate::W_SuperUser;
                $body
            } else if crate::py_type_check(obj, &crate::PROPERTY_USER_TYPE) {
                let $p = obj as *mut crate::W_PropertyUser;
                $body
            } else if crate::py_type_check(obj, &crate::COUNT_USER_TYPE) {
                let $p = obj as *mut crate::W_CountUser;
                $body
            } else if crate::py_type_check(obj, &crate::REPEAT_USER_TYPE) {
                let $p = obj as *mut crate::W_RepeatUser;
                $body
            } else if crate::py_type_check(obj, &crate::TAKEWHILE_USER_TYPE) {
                let $p = obj as *mut crate::W_TakeWhileUser;
                $body
            } else if crate::py_type_check(obj, &crate::DROPWHILE_USER_TYPE) {
                let $p = obj as *mut crate::W_DropWhileUser;
                $body
            } else if crate::py_type_check(obj, &crate::FILTERFALSE_USER_TYPE) {
                let $p = obj as *mut crate::W_FilterFalseUser;
                $body
            } else if crate::py_type_check(obj, &crate::ISLICE_USER_TYPE) {
                let $p = obj as *mut crate::W_ISliceUser;
                $body
            } else if crate::py_type_check(obj, &crate::BATCHED_USER_TYPE) {
                let $p = obj as *mut crate::W_BatchedUser;
                $body
            } else if crate::py_type_check(obj, &crate::PRODUCT_USER_TYPE) {
                let $p = obj as *mut crate::W_ProductUser;
                $body
            } else if crate::py_type_check(obj, &crate::COMBINATIONS_USER_TYPE) {
                let $p = obj as *mut crate::W_CombinationsUser;
                $body
            } else if crate::py_type_check(obj, &crate::COMBINATIONS_WITH_REPLACEMENT_USER_TYPE) {
                let $p = obj as *mut crate::W_CombinationsWithReplacementUser;
                $body
            } else if crate::py_type_check(obj, &crate::PERMUTATIONS_USER_TYPE) {
                let $p = obj as *mut crate::W_PermutationsUser;
                $body
            } else if crate::py_type_check(obj, &crate::GROUPBY_USER_TYPE) {
                let $p = obj as *mut crate::W_GroupByUser;
                $body
            } else if crate::py_type_check(obj, &crate::COMPRESS_USER_TYPE) {
                let $p = obj as *mut crate::W_CompressUser;
                $body
            } else if crate::py_type_check(obj, &crate::STARMAP_USER_TYPE) {
                let $p = obj as *mut crate::W_StarMapUser;
                $body
            } else if crate::py_type_check(obj, &crate::ACCUMULATE_USER_TYPE) {
                let $p = obj as *mut crate::W_AccumulateUser;
                $body
            } else if crate::py_type_check(obj, &crate::ZIP_LONGEST_USER_TYPE) {
                let $p = obj as *mut crate::W_ZipLongestUser;
                $body
            } else if crate::py_type_check(obj, &crate::PAIRWISE_USER_TYPE) {
                let $p = obj as *mut crate::W_PairwiseUser;
                $body
            } else if crate::py_type_check(obj, &crate::CYCLE_USER_TYPE) {
                let $p = obj as *mut crate::W_CycleUser;
                $body
            } else if crate::py_type_check(obj, &crate::CHAIN_USER_TYPE) {
                let $p = obj as *mut crate::W_ChainUser;
                $body
            } else if crate::py_type_check(obj, &crate::GENERIC_ALIAS_USER_TYPE) {
                let $p = obj as *mut crate::GenericAliasUser;
                $body
            } else if crate::py_type_check(obj, &crate::LONG_USER_TYPE) {
                let $p = obj as *mut crate::W_LongObjectUser;
                $body
            } else if crate::py_type_check(obj, &crate::STATICMETHOD_USER_TYPE) {
                let $p = obj as *mut crate::StaticMethodUser;
                $body
            } else if crate::py_type_check(obj, &crate::CLASSMETHOD_USER_TYPE) {
                let $p = obj as *mut crate::ClassMethodUser;
                $body
            } else if crate::py_type_check(obj, &crate::MODULE_USER_TYPE) {
                let $p = obj as *mut crate::ModuleUser;
                $body
            } else if crate::py_type_check(obj, &crate::BASE_EXCEPTION_USER_TYPE) {
                let $p = obj as *mut crate::W_BaseExceptionUser;
                $body
            } else if crate::py_type_check(obj, &crate::EXCEPTION_EXTENDED_USER_TYPE) {
                let $p = obj as *mut crate::W_ExceptionExtendedUser;
                $body
            } else {
                $fallback
            }
        }
    }};
}

#[inline]
unsafe fn mapdict_access_load_map(obj: crate::PyObjectRef) -> MapRef {
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return std::ptr::null();
    }
    match unsafe { (*tp).mapdict_access } {
        Some(access) => unsafe { (access.get_map)(obj) },
        None => std::ptr::null(),
    }
}

#[inline]
unsafe fn mapdict_access_store_map(obj: crate::PyObjectRef, map: MapRef) {
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return;
    }
    if let Some(access) = unsafe { (*tp).mapdict_access } {
        unsafe { (access.set_map)(obj, map) };
    }
}

#[inline]
unsafe fn mapdict_access_load_storage(
    obj: crate::PyObjectRef,
) -> *mut crate::object_array::ItemsBlock {
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { (*tp).mapdict_access } {
        Some(access) => unsafe { (access.get_storage)(obj) },
        None => std::ptr::null_mut(),
    }
}

#[inline]
unsafe fn mapdict_access_store_storage(
    obj: crate::PyObjectRef,
    storage: *mut crate::object_array::ItemsBlock,
) {
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return;
    }
    if let Some(access) = unsafe { (*tp).mapdict_access } {
        unsafe { (access.set_storage)(obj, storage) };
    }
}

/// Load `self.map` from the carrier's own typed field
/// (`MapdictStorageMixin._get_mapdict_map`).
///
/// # Safety
/// `obj` must be a live mapdict carrier, or null.
#[inline]
pub unsafe fn mapdict_typed_load_map(obj: crate::PyObjectRef) -> MapRef {
    if obj.is_null() {
        return std::ptr::null();
    }
    mapdict_typed_carrier!(
        obj,
        p => (*p).map,
        else => mapdict_access_load_map(obj)
    )
}

/// Store `self.map` on the carrier's own typed field
/// (`MapdictStorageMixin._set_mapdict_map`).
///
/// # Safety
/// `obj` must be a live mapdict carrier. The caller has already issued
/// the write barrier.
#[inline]
pub unsafe fn mapdict_typed_store_map(obj: crate::PyObjectRef, map: MapRef) {
    if obj.is_null() {
        return;
    }
    mapdict_typed_carrier!(
        obj,
        p => (*p).map = map,
        else => mapdict_access_store_map(obj, map)
    )
}

/// Load `self.storage` from the carrier's own typed field
/// (`MapdictStorageMixin._mapdict_read_storage`).
///
/// # Safety
/// `obj` must be a live mapdict carrier, or null.
#[inline]
pub unsafe fn mapdict_typed_load_storage(
    obj: crate::PyObjectRef,
) -> *mut crate::object_array::ItemsBlock {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    mapdict_typed_carrier!(
        obj,
        p => (*p).storage,
        else => mapdict_access_load_storage(obj)
    )
}

/// Store `self.storage` on the carrier's own typed field
/// (`MapdictStorageMixin._mapdict_write_storage`).
///
/// # Safety
/// `obj` must be a live mapdict carrier. The caller has already issued
/// the write barrier.
#[inline]
pub unsafe fn mapdict_typed_store_storage(
    obj: crate::PyObjectRef,
    storage: *mut crate::object_array::ItemsBlock,
) {
    if obj.is_null() {
        return;
    }
    mapdict_typed_carrier!(
        obj,
        p => (*p).storage = storage,
        else => mapdict_access_store_storage(obj, storage)
    )
}

const _: () = {
    assert!(std::mem::offset_of!(Terminator, base) == 0);
    assert!(std::mem::offset_of!(PlainAttribute, base) == 0);
    assert!(std::mem::offset_of!(AbstractAttribute, class) == 0);
    assert!(std::mem::offset_of!(AbstractAttribute, terminator) == std::mem::size_of::<MapClass>());
    assert!(std::mem::offset_of!(Terminator, w_cls) == std::mem::size_of::<AbstractAttribute>());
    assert!(
        std::mem::offset_of!(CachedAttributeHolder, order)
            < std::mem::offset_of!(CachedAttributeHolder, attr)
    );
    assert!(
        std::mem::offset_of!(CachedAttributeHolder, attr)
            < std::mem::offset_of!(CachedAttributeHolder, mutate_attr)
    );
    assert!(
        std::mem::offset_of!(CachedAttributeHolder, mutate_attr)
            < std::mem::offset_of!(CachedAttributeHolder, typ)
    );
    assert!(
        std::mem::offset_of!(CachedAttributeHolder, typ)
            < std::mem::offset_of!(CachedAttributeHolder, mutate_typ)
    );
};

/// `AbstractAttribute.terminator` as the Terminator object
/// (`AbstractAttribute.__init__` `assert isinstance(terminator, Terminator)`).
/// Same-address `cast_pointer`; the terminator node *is* a [`Terminator`].
///
/// # Safety
/// `map` must be a live map node whose `terminator` field is a live Terminator.
#[inline]
pub unsafe fn map_terminator(map: MapRef) -> *const Terminator {
    unsafe { (*map).terminator as *const Terminator }
}

/// `mapdict.py` `AbstractAttribute.terminator` then `Terminator.w_cls`.
///
/// Two field reads: `terminator` on the shared prefix, then `w_cls` on the
/// Terminator object at that same address (`cast_pointer`).
///
/// # Safety
/// `map` must be a live map node.
#[inline]
pub unsafe fn map_terminator_w_cls(map: MapRef) -> PyObjectRef {
    unsafe { (*map_terminator(map)).w_cls }
}

/// `mapdict.py` `BaseUserClassMapdict.getclass`:
/// `return self._get_mapdict_map().terminator.w_cls`.
///
/// Only a `_getusercls` layout (`typedef.py`) is that mixin: `user_base`
/// names the builtin the generated class extends. Exact `object()` is
/// `INSTANCE_TYPE` (`user_base` null) and uses `W_Root.getclass` /
/// `gettypefor(typeptr)` — the header word. A `_getusercls` carrier with
/// a non-null map reads the terminator; a still-null map falls back to
/// the header until `user_setup` copies `w_subtype.terminator`.
///
/// # Safety
/// `obj` must be a live object reference. Tagged immediates are not
/// carriers; callers such as `typedef::r#type` handle them first.
#[inline]
pub unsafe fn instance_getclass(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { crate::typeobject::is_type(obj) } {
        return unsafe { (*obj).w_class };
    }
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return unsafe { (*obj).w_class };
    }
    // `_getusercls` (`typedef.py`): `user_base` is the builtin the
    // generated class extends. `mapdict_offset != 0` is broader — exact
    // `object()` (`INSTANCE_TYPE`) also publishes the mixin word.
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() || unsafe { (*tp).user_base.is_null() } {
        return unsafe { (*obj).w_class };
    }
    // `_get_mapdict_map` is `self.map` (`mapdict.py`). Generated
    // `_getusercls` payloads put the mixin after the builtin prefix
    // as ordinary fields of that class (`import_from_mixin`). Load
    // `map` off the carrier (`getfield_gc_r`).
    let off = unsafe { (*tp).mapdict_offset };
    if off == 0 {
        return unsafe { (*obj).w_class };
    }
    let map_word = mapdict_typed_carrier!(
        obj,
        p => (*p).map,
        else => mapdict_access_load_map(obj)
    );
    if map_word.is_null() {
        return unsafe { (*obj).w_class };
    }
    // MapdictStorageMixin._get_mapdict_map: `return jit.promote(self.map)`.
    // getclass is that word then `terminator.w_cls`; without the promote the
    // class stays a red load and every `type()` / `is_exact_type` inlines a
    // fat residual.
    let map = majit_ir::jit::promote(map_word);
    let w_cls = unsafe { map_terminator_w_cls(map) };
    if !w_cls.is_null() {
        return w_cls;
    }
    unsafe { (*obj).w_class }
}

/// `mapdict.py` `AbstractAttribute.cache_attrs` table type.
pub type CacheAttrsTable = Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>;

fn visit_map_ref_slot(slot: *mut MapRef, f: &mut dyn FnMut(*mut PyObjectRef)) {
    f(slot as *mut PyObjectRef);
}

unsafe fn walk_cache_attrs_table(
    slot: &AtomicPtr<CacheAttrsTable>,
    f: &mut dyn FnMut(*mut PyObjectRef),
) {
    let ptr = slot.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    let table = unsafe { &*ptr };
    let mut guard = table.lock();
    for holder in guard.values_mut() {
        f(holder as *mut *const CachedAttributeHolder as *mut PyObjectRef);
    }
}

/// `rgc.register_custom_trace_hook` for the interned `cache_attrs` table
/// on [`AbstractAttribute`]. Fixed GC pointer fields are registered with
/// `TypeInfo::with_gc_ptrs`; this hook only walks the holder table.
///
/// # Safety
/// `obj_addr` must point at a live [`Terminator`] or [`PlainAttribute`].
pub unsafe fn cache_attrs_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut PyObjectRef)) {
    let base = obj_addr as *mut AbstractAttribute;
    unsafe { walk_cache_attrs_table(&(*base).cache_attrs, f) };
}

unsafe fn drop_cache_attrs_table(slot: &AtomicPtr<CacheAttrsTable>) {
    let ptr = slot.swap(std::ptr::null_mut(), std::sync::atomic::Ordering::AcqRel);
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Sweep destructor for [`Terminator`]: reclaim `QuasiImmutField` and drop
/// `cache_attrs`. Map/Python refs are GC-owned.
///
/// # Safety
/// `obj_addr` must point at a swept [`Terminator`].
pub unsafe fn terminator_destructor(obj_addr: usize) {
    let t = obj_addr as *mut Terminator;
    unsafe { drop_cache_attrs_table(&(*t).base.cache_attrs) };
    unsafe { std::ptr::drop_in_place(t) };
}

/// Sweep destructor for [`PlainAttribute`]: reclaim `QuasiImmutField`, drop
/// `cache_attrs` and `name`. Map/Python refs are GC-owned.
///
/// # Safety
/// `obj_addr` must point at a swept [`PlainAttribute`].
pub unsafe fn plain_attribute_destructor(obj_addr: usize) {
    let p = obj_addr as *mut PlainAttribute;
    unsafe { drop_cache_attrs_table(&(*p).base.cache_attrs) };
    unsafe { std::ptr::drop_in_place(p) };
}

/// `rgc.register_custom_trace_hook` for [`CachedAttributeHolder`].
///
/// # Safety
/// `obj_addr` must point at a live [`CachedAttributeHolder`].
pub unsafe fn cached_attribute_holder_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut PyObjectRef),
) {
    let h = unsafe { &mut *(obj_addr as *mut CachedAttributeHolder) };
    visit_map_ref_slot(h.attr.as_ptr(), f);
}

/// Sweep destructor for [`CachedAttributeHolder`].
///
/// # Safety
/// `obj_addr` must point at a swept [`CachedAttributeHolder`].
pub unsafe fn cached_attribute_holder_destructor(obj_addr: usize) {
    unsafe { std::ptr::drop_in_place(obj_addr as *mut CachedAttributeHolder) };
}

#[cfg(test)]
mod first_field_inheritance_tests {
    use super::*;
    use crate::quasiimmut::{null_mutate_lock, null_mutate_word};

    fn dummy_base(class: MapClass) -> AbstractAttribute {
        AbstractAttribute {
            class,
            terminator: std::ptr::null(),
            cache_attrs: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    fn dummy_terminator(w_cls: PyObjectRef) -> Terminator {
        Terminator {
            base: dummy_base(MapClass::Terminator),
            w_cls,
            allow_unboxing: Cell::new(true),
            mutate_allow_unboxing: null_mutate_word(),
            mutate_allow_unboxing_lock: null_mutate_lock(),
            kind: TerminatorKind::NoDict,
            devolved_dict_terminator: Cell::new(std::ptr::null()),
        }
    }

    fn dummy_plain() -> PlainAttribute {
        PlainAttribute {
            base: dummy_base(MapClass::PlainAttribute),
            name: Wtf8Buf::from("x"),
            attrkind: 0,
            storageindex: 0,
            num_attributes: 1,
            back: std::ptr::null(),
            ever_mutated: Cell::new(false),
            mutate_ever_mutated: null_mutate_word(),
            mutate_ever_mutated_lock: null_mutate_lock(),
            order: 0,
            unboxed: None,
        }
    }

    #[test]
    fn downcast_keeps_the_object_start() {
        let mut term = dummy_terminator(0x2000 as PyObjectRef);
        let map = &mut term as *mut Terminator as MapRef;
        let as_term = unsafe { (*map).as_terminator() as *const Terminator };
        assert_eq!(as_term as usize, map as usize);
        assert!(unsafe { (*map).is_terminator() });
        assert!(!unsafe { (*map).is_plain() });

        let mut plain = dummy_plain();
        let map = &mut plain as *mut PlainAttribute as MapRef;
        let as_plain = unsafe { (*map).as_plain() as *const PlainAttribute };
        assert_eq!(as_plain as usize, map as usize);
        assert!(unsafe { (*map).is_plain() });
        assert!(!unsafe { (*map).is_terminator() });
    }

    #[test]
    fn map_terminator_w_cls_reads_terminator_then_w_cls() {
        let w_cls = 0x2000 as PyObjectRef;
        let mut term = dummy_terminator(w_cls);
        let map = &mut term as *mut Terminator as *mut AbstractAttribute;
        unsafe { (*map).terminator = map };
        let map = map as MapRef;
        assert!(std::ptr::eq(unsafe { (*map).terminator }, map));
        assert!(std::ptr::eq(unsafe { (*map).as_terminator().w_cls }, w_cls));
        assert!(std::ptr::eq(unsafe { map_terminator_w_cls(map) }, w_cls));
        assert_eq!(unsafe { map_terminator(map) as usize }, map as usize);
    }

    #[test]
    fn quasiimmut_offsets_are_from_the_object_start() {
        // Walker pins pass the object start (`MapRef` / `*const Terminator` /
        // `*const PlainAttribute`); descr offsets are `offset_of!` on those
        // structs. `struct_ptr + mutate offset` must be the mutate word.
        let w_cls = 0x2000 as PyObjectRef;
        let term = dummy_terminator(w_cls);
        let term_ptr = &term as *const Terminator;
        let map = term_ptr as MapRef;
        assert_eq!(
            unsafe { (*map).as_terminator() as *const Terminator },
            term_ptr
        );
        let via_term = (term_ptr as *const u8)
            .wrapping_add(core::mem::offset_of!(Terminator, mutate_allow_unboxing));
        let direct_term = std::ptr::addr_of!(term.mutate_allow_unboxing) as *const u8;
        assert_eq!(via_term, direct_term);

        let plain = dummy_plain();
        let plain_ptr = &plain as *const PlainAttribute;
        let map = plain_ptr as MapRef;
        assert_eq!(
            unsafe { (*map).as_plain() as *const PlainAttribute },
            plain_ptr
        );
        let via_plain = (plain_ptr as *const u8)
            .wrapping_add(core::mem::offset_of!(PlainAttribute, mutate_ever_mutated));
        let direct_plain = std::ptr::addr_of!(plain.mutate_ever_mutated) as *const u8;
        assert_eq!(via_plain, direct_plain);

        let holder = CachedAttributeHolder {
            order: 0,
            attr: Cell::new(std::ptr::null()),
            mutate_attr: null_mutate_word(),
            mutate_attr_lock: null_mutate_lock(),
            typ: Cell::new(None),
            mutate_typ: null_mutate_word(),
            mutate_typ_lock: null_mutate_lock(),
        };
        let holder_ptr = &holder as *const CachedAttributeHolder;
        let via_attr = (holder_ptr as *const u8)
            .wrapping_add(core::mem::offset_of!(CachedAttributeHolder, mutate_attr));
        let direct_attr = &holder.mutate_attr as *const _ as *const u8;
        assert_eq!(via_attr, direct_attr);
        let via_typ = (holder_ptr as *const u8)
            .wrapping_add(core::mem::offset_of!(CachedAttributeHolder, mutate_typ));
        let direct_typ = &holder.mutate_typ as *const _ as *const u8;
        assert_eq!(via_typ, direct_typ);
    }
}
