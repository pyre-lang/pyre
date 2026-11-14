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
    /// `mapdict.py` `UnboxedPlainAttribute`.
    UnboxedPlainAttribute = 2,
}

/// A shared, interned map node (`mapdict.py` `AbstractAttribute`).
///
/// GC pointer to the object start of a [`Terminator`], [`PlainAttribute`],
/// or [`UnboxedPlainAttribute`].
/// Identity compare (`ptr::eq`) remains valid after a moving collection
/// because both sides are forwarded. The parent struct is first-field-inlined
/// (`lltype.Struct._first_struct` / `_note_inlined_into`), so an upcast or
/// downcast (`cast_pointer`) keeps this address.
pub type MapRef = *const AbstractAttribute;

/// mapdict.py `AbstractAttribute` — the shared prefix of every map node.
///
/// Never allocated on its own. [`Terminator`] and [`PlainAttribute`] inline
/// this as their first field (`GcStruct._first_struct`).
/// [`UnboxedPlainAttribute`] inlines [`PlainAttribute`].
/// `AbstractAttribute._immutable_fields_ = ['terminator']`.
/// `AbstractAttribute._attrs_ = ['terminator', 'space', 'cache_attrs']`
/// (`space` is omitted: pyre's object space is ambient).
#[repr(C)]
#[majit_macros::jit_immutable_fields("terminator")]
pub struct AbstractAttribute {
    /// rclass `OBJECT.typeptr`. Distinguishes Terminator, PlainAttribute,
    /// and UnboxedPlainAttribute.
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

/// mapdict.py `UnboxedPlainAttribute(PlainAttribute)` — an attribute stored
/// unboxed in a shared longlong list at `PlainAttribute.storageindex`;
/// `listindex` is the position in that list.
///
/// `base` is the first-field-inlined parent (`GcStruct._first_struct`).
/// `mapdict.py UnboxedPlainAttribute._immutable_fields_ = ["listindex",
/// "firstunwrapped", "typ"]`.
/// `UnboxedPlainAttribute.__init__` does not call `PlainAttribute.__init__`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("listindex", "firstunwrapped", "typ")]
pub struct UnboxedPlainAttribute {
    /// `PlainAttribute` prefix. Same address as the UnboxedPlainAttribute object.
    pub base: PlainAttribute,
    /// mapdict.py `UnboxedPlainAttribute.typ`.
    pub typ: UnboxType,
    /// mapdict.py `UnboxedPlainAttribute.listindex` — position in the shared
    /// longlong list.
    pub listindex: usize,
    /// mapdict.py `UnboxedPlainAttribute.firstunwrapped` — this attribute is
    /// the first unboxed one to occupy its storage slot (so it allocates the
    /// list).
    pub firstunwrapped: bool,
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

/// GC tid for [`UnboxedPlainAttribute`], assigned at `register_type`.
pub static UNBOXED_PLAIN_ATTRIBUTE_GC_TYPE_ID: TypeIdCell = TypeIdCell::auto();

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

impl GcType for UnboxedPlainAttribute {
    fn type_id() -> u32 {
        UNBOXED_PLAIN_ATTRIBUTE_GC_TYPE_ID.get()
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

    /// `cast_pointer` to [`UnboxedPlainAttribute`] at the same address.
    ///
    /// The class test is `isinstance` ([`Self::is_unboxed_plain`]); this is
    /// the identity downcast (`lltype.cast_pointer` / `GcStruct._first_struct`).
    pub fn as_unboxed_plain(&self) -> &UnboxedPlainAttribute {
        debug_assert!(self.is_unboxed_plain());
        unsafe { &*(self as *const Self as *const UnboxedPlainAttribute) }
    }

    /// `isinstance(self, PlainAttribute)`.
    pub fn is_plain(&self) -> bool {
        matches!(
            self.class,
            MapClass::PlainAttribute | MapClass::UnboxedPlainAttribute
        )
    }

    /// `isinstance(self, UnboxedPlainAttribute)`.
    pub fn is_unboxed_plain(&self) -> bool {
        self.class == MapClass::UnboxedPlainAttribute
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
        // `UnboxedPlainAttribute.storage_needed`: an unboxed attribute only
        // adds a slot when it is the first unboxed one in its slot
        // (`firstunwrapped`); otherwise it packs into the slot the prior
        // unboxed attribute already reserved, so its size is
        // `back.storage_needed()`.
        if self.is_unboxed_plain() {
            let u = self.as_unboxed_plain();
            if u.firstunwrapped {
                u.base.storageindex + 1
            } else {
                unsafe { (*u.base.back).storage_needed() }
            }
        } else {
            self.as_plain().storageindex + 1
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
/// (`typedef.py` `_getusercls` `import_from_mixin`). `getclass` is
/// [`base_user_class_mapdict_getclass`] (`_share_methods`) except the
/// `W_ObjectObject` carrier, which keeps an `import_from_mixin` copy.
/// Stored on the typeptr so a call through an unknown class is a vtable
/// slot (`rclass` `getclsfield` then `FunctionReprBase.call`). Written
/// once at type creation (`_immutable_fields_`).
#[repr(C)]
#[derive(Clone, Copy)]
#[majit_macros::jit_immutable_fields(
    "get_map",
    "set_map",
    "get_storage",
    "set_storage",
    "getclass"
)]
pub struct MapdictMixinAccess {
    pub get_map: fn(crate::PyObjectRef) -> MapRef,
    pub set_map: fn(crate::PyObjectRef, MapRef),
    pub get_storage: fn(crate::PyObjectRef) -> *mut crate::object_array::ItemsBlock,
    pub set_storage: fn(crate::PyObjectRef, *mut crate::object_array::ItemsBlock),
    pub getclass: fn(crate::PyObjectRef) -> crate::PyObjectRef,
}

/// One `mapdict_mixin_access!` static's FnPtr slots, with the FunDecl
/// `name_path` of each nested function. `CallControl.get_jitcode` stamps
/// `JitCode.fnaddr` from `getfunctionptr`; `bytecode_for_address` looks
/// that word up. The pointer is the slot field itself, captured in this
/// crate so a later `fn as usize` in another crate cannot mint a second
/// wasm32 table index.
#[derive(Clone, Copy)]
pub struct MapdictAccessFnCarrier {
    pub get_map_path: &'static str,
    pub get_map: fn(crate::PyObjectRef) -> MapRef,
    pub set_map_path: &'static str,
    pub set_map: fn(crate::PyObjectRef, MapRef),
    pub get_storage_path: &'static str,
    pub get_storage: fn(crate::PyObjectRef) -> *mut crate::object_array::ItemsBlock,
    pub set_storage_path: &'static str,
    pub set_storage: fn(crate::PyObjectRef, *mut crate::object_array::ItemsBlock),
    /// FunDecl path of this table's nested `getclass`. Empty when the
    /// table stores [`base_user_class_mapdict_getclass`]; that path is
    /// published once by [`for_each_mapdict_access_fnaddr`].
    pub getclass_path: &'static str,
    pub getclass: fn(crate::PyObjectRef) -> crate::PyObjectRef,
}

impl MapdictAccessFnCarrier {
    pub const fn from_access(
        get_map_path: &'static str,
        set_map_path: &'static str,
        get_storage_path: &'static str,
        set_storage_path: &'static str,
        getclass_path: &'static str,
        access: MapdictMixinAccess,
    ) -> Self {
        Self {
            get_map_path,
            get_map: access.get_map,
            set_map_path,
            set_map: access.set_map,
            get_storage_path,
            get_storage: access.get_storage,
            set_storage_path,
            set_storage: access.set_storage,
            getclass_path,
            getclass: access.getclass,
        }
    }

    pub fn for_each_slot(self, mut visit: impl FnMut(&'static str, usize)) {
        visit(self.get_map_path, self.get_map as usize);
        visit(self.set_map_path, self.set_map as usize);
        visit(self.get_storage_path, self.get_storage as usize);
        visit(self.set_storage_path, self.set_storage as usize);
        if !self.getclass_path.is_empty() {
            visit(self.getclass_path, self.getclass as usize);
        }
    }
}

// Safety: every field is a `'static` path or a process-global function
// pointer word; sharing across threads is sound.
unsafe impl Sync for MapdictAccessFnCarrier {}

/// Link-time registry of every `mapdict_mixin_access!` slot table.
/// `distributed_slice` rejects wasm32; that target fills
/// [`WASM_MAPDICT_ACCESS_FNADDRS`] from a constructor instead. Read both
/// through [`for_each_mapdict_access_fnaddr`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static PYRE_MAPDICT_ACCESS_FNADDRS: [MapdictAccessFnCarrier] = [..];

#[cfg(target_arch = "wasm32")]
pub static WASM_MAPDICT_ACCESS_FNADDRS: std::sync::Mutex<Vec<MapdictAccessFnCarrier>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(target_arch = "wasm32")]
pub fn register_mapdict_access_fncarrier(carrier: MapdictAccessFnCarrier) {
    WASM_MAPDICT_ACCESS_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(carrier);
}

/// Visit the shared [`base_user_class_mapdict_getclass`] once, then every
/// nested slot function of every `mapdict_mixin_access!` static.
pub fn for_each_mapdict_access_fnaddr(mut visit: impl FnMut(&'static str, usize)) {
    visit(
        concat!(module_path!(), "::base_user_class_mapdict_getclass"),
        base_user_class_mapdict_getclass as *const () as usize,
    );
    #[cfg(not(target_arch = "wasm32"))]
    for carrier in PYRE_MAPDICT_ACCESS_FNADDRS {
        carrier.for_each_slot(&mut visit);
    }
    #[cfg(target_arch = "wasm32")]
    for carrier in WASM_MAPDICT_ACCESS_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        carrier.for_each_slot(&mut visit);
    }
}

#[doc(hidden)]
pub use paste as __paste;

/// `mapdict.py` `BaseUserClassMapdict.getclass` after `_share_methods`
/// (`typedef.py` `_getusercls`): one FunDecl for every non-object carrier.
/// `_get_mapdict_map` is an *indirect* call through the type's
/// `mapdict_access.get_map` slot (`rclass` `getclsfield` then
/// `FunctionReprBase.call`). `W_ObjectObject` keeps an `import_from_mixin`
/// copy (`mapdict_mixin_access!` `copy_getclass`).
pub fn base_user_class_mapdict_getclass(obj: PyObjectRef) -> PyObjectRef {
    match unsafe { (*(*obj).ob_type).mapdict_access } {
        Some(access) => {
            let map = (access.get_map)(obj);
            if map.is_null() {
                return unsafe { (*obj).w_class };
            }
            let w_cls = unsafe { map_terminator_w_cls(map) };
            if w_cls.is_null() {
                unsafe { (*obj).w_class }
            } else {
                w_cls
            }
        }
        None => unsafe { (*obj).w_class },
    }
}

/// Typed `self.map` / `self.storage` accessors and `getclass` for one
/// carrier layout. Bodies are safe `fn` items so they compile to JitCode
/// (`MapdictStorageMixin._get_mapdict_map` / `BaseUserClassMapdict.getclass`).
///
/// Default arm: nested `get_map` / `set_map` / `get_storage` / `set_storage`
/// (`import_from_mixin` copies) and the shared
/// [`base_user_class_mapdict_getclass`] (`_share_methods`).
/// `copy_getclass`: the same four accessors plus a nested `getclass`
/// (`import_from_mixin` copy). Used only for `W_ObjectObject`.
///
/// Also emits one [`MapdictAccessFnCarrier`] so `build_jit_trace_fnaddrs`
/// can publish each slot as `getfunctionptr` without a hand list of
/// carriers. Paths are `{module}::{STATIC}::{field}`, matching the
/// translator family member from `discover_fnptr_field_families`. The
/// shared getclass is published once under its FunDecl path.
#[macro_export]
macro_rules! mapdict_mixin_access {
    (
        $(#[$attr:meta])*
        $vis:vis static $access:ident for $ty:ty $(;)?
    ) => {
        $crate::mapdict_mixin_access! {
            @table
            $(#[$attr])*
            $vis static $access for $ty;
            extra = {};
            getclass = $crate::mapdict::base_user_class_mapdict_getclass,
            getclass_path = ""
        }
    };
    (
        $(#[$attr:meta])*
        $vis:vis static $access:ident for $ty:ty, copy_getclass $(;)?
    ) => {
        $crate::mapdict_mixin_access! {
            @table
            $(#[$attr])*
            $vis static $access for $ty;
            extra = {
                fn getclass(obj: $crate::PyObjectRef) -> $crate::PyObjectRef {
                    majit_ir::icf_identity!(::core::concat!(::core::module_path!(), "::", stringify!($access), "::getclass"));
                    // BaseUserClassMapdict.getclass:
                    // `return self._get_mapdict_map().terminator.w_cls`.
                    let map = majit_ir::jit::promote(unsafe { (*(obj as *const $ty)).map });
                    if map.is_null() {
                        return unsafe { (*obj).w_class };
                    }
                    let w_cls = unsafe { $crate::mapdict::map_terminator_w_cls(map) };
                    if w_cls.is_null() {
                        unsafe { (*obj).w_class }
                    } else {
                        w_cls
                    }
                }
            };
            getclass = getclass,
            getclass_path = ::core::concat!(
                ::core::module_path!(),
                "::",
                stringify!($access),
                "::getclass"
            )
        }
    };
    (
        @table
        $(#[$attr:meta])*
        $vis:vis static $access:ident for $ty:ty;
        extra = { $($extra:tt)* };
        getclass = $getclass:expr,
        getclass_path = $getclass_path:expr
    ) => {
        $(#[$attr])*
        $vis static $access: $crate::mapdict::MapdictMixinAccess = {
            $crate::mapdict_mixin_access!(@accessors $access, $ty, get_map, set_map, get_storage, set_storage);
            $($extra)*
            $crate::mapdict::MapdictMixinAccess {
                get_map,
                set_map,
                get_storage,
                set_storage,
                getclass: $getclass,
            }
        };
        $crate::mapdict_mixin_access!(@fnaddrs $access, $getclass_path);
    };
    (@accessors $access:ident, $ty:ty, $get_map:ident, $set_map:ident, $get_storage:ident, $set_storage:ident) => {
            // Each carrier's slot bodies are byte-identical when the field offsets
            // agree; the token keeps every published fnaddr distinct so
            // bytecode_for_address resolves one jitcode per carrier
            // (`pyjitpl.py` `MetaInterpStaticData.bytecode_for_address`).
            fn $get_map(obj: $crate::PyObjectRef) -> $crate::mapdict::MapRef {
                majit_ir::icf_identity!(::core::concat!(::core::module_path!(), "::", stringify!($access), "::get_map"));
                // MapdictStorageMixin._get_mapdict_map: `return jit.promote(self.map)`.
                majit_ir::jit::promote(unsafe { (*(obj as *const $ty)).map })
            }
            fn $set_map(obj: $crate::PyObjectRef, map: $crate::mapdict::MapRef) {
                majit_ir::icf_identity!(::core::concat!(::core::module_path!(), "::", stringify!($access), "::set_map"));
                unsafe {
                    (*(obj as *mut $ty)).map = map;
                }
            }
            fn $get_storage(obj: $crate::PyObjectRef) -> *mut $crate::object_array::ItemsBlock {
                majit_ir::icf_identity!(::core::concat!(::core::module_path!(), "::", stringify!($access), "::get_storage"));
                unsafe { (*(obj as *const $ty)).storage }
            }
            fn $set_storage(
                obj: $crate::PyObjectRef,
                storage: *mut $crate::object_array::ItemsBlock,
            ) {
                majit_ir::icf_identity!(::core::concat!(::core::module_path!(), "::", stringify!($access), "::set_storage"));
                unsafe {
                    (*(obj as *mut $ty)).storage = storage;
                }
            }
    };
    (@fnaddrs $access:ident, $getclass_path:expr) => {
        $crate::mapdict::__paste::paste! {
            #[cfg(not(target_arch = "wasm32"))]
            #[::linkme::distributed_slice($crate::mapdict::PYRE_MAPDICT_ACCESS_FNADDRS)]
            #[allow(non_upper_case_globals)]
            static [<$access _FNADDRS>]: $crate::mapdict::MapdictAccessFnCarrier =
                $crate::mapdict::MapdictAccessFnCarrier::from_access(
                    ::core::concat!(
                        ::core::module_path!(),
                        "::",
                        stringify!($access),
                        "::get_map"
                    ),
                    ::core::concat!(
                        ::core::module_path!(),
                        "::",
                        stringify!($access),
                        "::set_map"
                    ),
                    ::core::concat!(
                        ::core::module_path!(),
                        "::",
                        stringify!($access),
                        "::get_storage"
                    ),
                    ::core::concat!(
                        ::core::module_path!(),
                        "::",
                        stringify!($access),
                        "::set_storage"
                    ),
                    $getclass_path,
                    $access,
                );

            #[cfg(target_arch = "wasm32")]
            #[::ctor::ctor(unsafe)]
            #[allow(non_snake_case)]
            fn [<__register_ $access _fnaddrs>]() {
                $crate::mapdict::register_mapdict_access_fncarrier(
                    $crate::mapdict::MapdictAccessFnCarrier::from_access(
                        ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            stringify!($access),
                            "::get_map"
                        ),
                        ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            stringify!($access),
                            "::set_map"
                        ),
                        ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            stringify!($access),
                            "::get_storage"
                        ),
                        ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            stringify!($access),
                            "::set_storage"
                        ),
                        $getclass_path,
                        $access,
                    ),
                );
            }
        }
    };
}

/// Load `self.map` through the type's `mapdict_access` slot
/// (`MapdictStorageMixin._get_mapdict_map`). The slot is a vtable field
/// (`rclass` `getclsfield` then `FunctionReprBase.call`).
///
/// # Safety
/// `obj` must be a live mapdict carrier, or null.
#[inline]
pub unsafe fn mapdict_typed_load_map(obj: crate::PyObjectRef) -> MapRef {
    if obj.is_null() {
        return std::ptr::null();
    }
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return std::ptr::null();
    }
    match unsafe { (*tp).mapdict_access } {
        Some(access) => (access.get_map)(obj),
        None => std::ptr::null(),
    }
}

/// Store `self.map` through the type's `mapdict_access` slot
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
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return;
    }
    if let Some(access) = unsafe { (*tp).mapdict_access } {
        (access.set_map)(obj, map);
    }
}

/// Load `self.storage` through the type's `mapdict_access` slot
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
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { (*tp).mapdict_access } {
        Some(access) => (access.get_storage)(obj),
        None => std::ptr::null_mut(),
    }
}

/// Store `self.storage` through the type's `mapdict_access` slot
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
    let tp = unsafe { (*obj).ob_type };
    if tp.is_null() {
        return;
    }
    if let Some(access) = unsafe { (*tp).mapdict_access } {
        (access.set_storage)(obj, storage);
    }
}

const _: () = {
    assert!(std::mem::offset_of!(Terminator, base) == 0);
    assert!(std::mem::offset_of!(PlainAttribute, base) == 0);
    assert!(std::mem::offset_of!(UnboxedPlainAttribute, base) == 0);
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
/// `gettypefor(typeptr)` — the header word. A `_getusercls` carrier
/// dispatches through the type's `getclass` slot (`rclass` `getclsfield`
/// then `FunctionReprBase.call`). Non-object carriers share
/// [`base_user_class_mapdict_getclass`] (`_share_methods`); `W_ObjectObject`
/// keeps an `import_from_mixin` copy. `get_map` promotes `self.map`
/// (`MapdictStorageMixin._get_mapdict_map`) and the body reads
/// `terminator.w_cls`.
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
    let off = unsafe { (*tp).mapdict_offset };
    if off == 0 {
        return unsafe { (*obj).w_class };
    }
    match unsafe { (*tp).mapdict_access } {
        Some(access) => (access.getclass)(obj),
        None => unsafe { (*obj).w_class },
    }
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
/// `obj_addr` must point at a live [`Terminator`], [`PlainAttribute`], or
/// [`UnboxedPlainAttribute`].
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

/// Sweep destructor for [`UnboxedPlainAttribute`]: same treatment as
/// [`plain_attribute_destructor`]. Drop of the inlined [`PlainAttribute`]
/// reclaims the inherited `ever_mutated?` qmut words.
///
/// # Safety
/// `obj_addr` must point at a swept [`UnboxedPlainAttribute`].
pub unsafe fn unboxed_plain_attribute_destructor(obj_addr: usize) {
    let p = obj_addr as *mut UnboxedPlainAttribute;
    unsafe { drop_cache_attrs_table(&(*p).base.base.cache_attrs) };
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
        }
    }

    fn dummy_unboxed() -> UnboxedPlainAttribute {
        UnboxedPlainAttribute {
            base: PlainAttribute {
                base: dummy_base(MapClass::UnboxedPlainAttribute),
                name: Wtf8Buf::from("x"),
                attrkind: 0,
                storageindex: 0,
                num_attributes: 1,
                back: std::ptr::null(),
                ever_mutated: Cell::new(false),
                mutate_ever_mutated: null_mutate_word(),
                mutate_ever_mutated_lock: null_mutate_lock(),
                order: 0,
            },
            typ: UnboxType::Int,
            listindex: 0,
            firstunwrapped: true,
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
        assert!(!unsafe { (*map).is_unboxed_plain() });
        assert!(!unsafe { (*map).is_terminator() });

        let mut unboxed = dummy_unboxed();
        let map = &mut unboxed as *mut UnboxedPlainAttribute as MapRef;
        let as_unboxed = unsafe { (*map).as_unboxed_plain() as *const UnboxedPlainAttribute };
        assert_eq!(as_unboxed as usize, map as usize);
        let as_plain = unsafe { (*map).as_plain() as *const PlainAttribute };
        assert_eq!(as_plain as usize, map as usize);
        assert!(unsafe { (*map).is_plain() });
        assert!(unsafe { (*map).is_unboxed_plain() });
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

        let unboxed = dummy_unboxed();
        let unboxed_ptr = &unboxed as *const UnboxedPlainAttribute;
        let map = unboxed_ptr as MapRef;
        assert_eq!(
            unsafe { (*map).as_unboxed_plain() as *const UnboxedPlainAttribute },
            unboxed_ptr
        );
        let via_unboxed = (unboxed_ptr as *const u8)
            .wrapping_add(core::mem::offset_of!(PlainAttribute, mutate_ever_mutated));
        let direct_unboxed = std::ptr::addr_of!(unboxed.base.mutate_ever_mutated) as *const u8;
        assert_eq!(via_unboxed, direct_unboxed);

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

    #[test]
    fn mapdict_access_fnaddr_path_matches_translator_member() {
        let qualified = "pyre_object::objectobject::W_OBJECT_OBJECT_MAPDICT_ACCESS::getclass";
        let mut found = None;
        super::for_each_mapdict_access_fnaddr(|path, addr| {
            if path == qualified {
                found = Some(addr);
            }
        });
        let addr = found.expect(qualified);
        assert_eq!(
            addr,
            crate::objectobject::W_OBJECT_OBJECT_MAPDICT_ACCESS.getclass as usize
        );
    }

    #[test]
    fn test_correct_method_sharing() {
        // `typedef.py` `_getusercls` / `test_mapdict.py` `test_correct_method_sharing`.
        let shared = super::base_user_class_mapdict_getclass as *const () as usize;
        let bytes = crate::bytesobject::W_BYTES_USER_MAPDICT_ACCESS.getclass as usize;
        let unicode = crate::unicodeobject::W_UNICODE_USER_MAPDICT_ACCESS.getclass as usize;
        assert_eq!(bytes, shared);
        assert_eq!(unicode, shared);
        let object = crate::objectobject::W_OBJECT_OBJECT_MAPDICT_ACCESS.getclass as usize;
        assert_ne!(object, shared);

        let shared_path = "pyre_object::mapdict::base_user_class_mapdict_getclass";
        let mut shared_visits = 0;
        super::for_each_mapdict_access_fnaddr(|path, addr| {
            if path == shared_path {
                shared_visits += 1;
                assert_eq!(addr, shared);
            }
            assert!(
                !path.contains("W_BYTES_USER_MAPDICT_ACCESS::getclass"),
                "default arm must not publish a per-carrier getclass path: {path}"
            );
        });
        assert_eq!(shared_visits, 1);
    }
}
