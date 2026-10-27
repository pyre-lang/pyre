//! Interned mapdict node types (`mapdict.py` `AbstractAttribute`).
//!
//! The map pointer is a field of the instance layout (`MapdictStorageMixin.map`);
//! `Terminator.w_cls` is the Python class (`BaseUserClassMapdict.getclass`).
//! Algorithms that intern and rewrite maps stay in the interpreter mapdict
//! layer; this crate owns the node layout so `w_instance_get_type` can read
//! the class where the data lives.

use crate::pyobject::PyObjectRef;
use crate::quasiimmut::QuasiImmutField;
use parking_lot::Mutex;
use rustpython_wtf8::Wtf8Buf;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::AtomicPtr;

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
///
/// `repr(C)` plus `terminator` as the first payload field matches
/// `AbstractAttribute._attrs_ = ['terminator', ...]` so both variants of
/// [`MapNode`] put that word at one offset after the enum tag. `w_cls`
/// is the next word (`Terminator.__init__`).
#[repr(C)]
pub struct Terminator {
    /// mapdict.py `AbstractAttribute.terminator` — a Terminator points to
    /// itself.
    pub terminator: MapRef,
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
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.allow_unboxing_watchers) };
        }
        self.allow_unboxing.set(v);
    }

    pub fn current_allow_unboxing_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
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
///
/// `terminator` is first so it shares [`Terminator`]'s offset after the
/// [`MapNode`] enum tag (`AbstractAttribute.terminator`).
#[repr(C)]
pub struct PlainAttribute {
    /// mapdict.py `AbstractAttribute.terminator` (= `back.terminator`).
    pub terminator: MapRef,
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
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.ever_mutated_watchers) };
        }
        self.ever_mutated.set(v);
    }

    pub fn current_ever_mutated_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
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

/// Map nodes are `lltype::malloc_raw` allocations that are never freed,
/// never GC objects.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_MapNode: &str = "raw";

/// mapdict.py `AbstractAttribute` plus its two concrete subclasses.
///
/// `repr(C)` pins the enum tag then the variant payload so
/// `AbstractAttribute.terminator` is one offset on both variants.
#[repr(C)]
pub enum MapNode {
    Terminator(Terminator),
    Plain(PlainAttribute),
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
}

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
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.attr_watchers) };
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
            unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.typ_watchers) };
        }
        self.typ.set(t);
    }

    pub fn current_attr_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
        self.attr_watchers.get_current_qmut_instance()
    }

    pub fn attr_qmut_installed(&self) -> bool {
        self.attr_watchers.is_installed()
    }

    pub fn force_attr_qmut(&self) {
        self.attr_watchers.invalidate();
    }

    pub fn current_typ_qmut(&self) -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
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

/// `repr(C)` [`MapNode`] prefix: enum tag then `AbstractAttribute.terminator`.
/// Both variants put `terminator` first in the payload, so this overlay is
/// the typed field read `self.terminator` (`mapdict.py`).
/// `AbstractAttribute._immutable_fields_` includes `terminator`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("terminator")]
struct MapNodeAbstractAttribute {
    _discriminant: isize,
    terminator: MapRef,
}

/// Terminator node overlay: tag, `terminator`, then `Terminator.w_cls`.
/// getclass loads `.terminator.w_cls` through a pointer that is always a
/// Terminator (`AbstractAttribute.__init__` `assert isinstance(terminator,
/// Terminator)`).
/// `AbstractAttribute._immutable_fields_` includes `terminator`;
/// `Terminator._immutable_fields_` includes `w_cls`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("terminator", "w_cls")]
struct MapNodeTerminatorView {
    _discriminant: isize,
    terminator: MapRef,
    w_cls: PyObjectRef,
}

/// Both overlays view a raw [`MapNode`], so they share its flavor.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_MapNodeAbstractAttribute: &str = "raw";
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_MapNodeTerminatorView: &str = "raw";

const _: () = {
    assert!(std::mem::offset_of!(Terminator, terminator) == 0);
    assert!(std::mem::offset_of!(Terminator, w_cls) == std::mem::size_of::<MapRef>());
    assert!(std::mem::offset_of!(PlainAttribute, terminator) == 0);
    assert!(
        std::mem::offset_of!(MapNodeAbstractAttribute, terminator) == std::mem::size_of::<isize>()
    );
    assert!(
        std::mem::offset_of!(MapNodeTerminatorView, w_cls)
            == std::mem::size_of::<isize>() + std::mem::size_of::<MapRef>()
    );
};

/// `mapdict.py` `AbstractAttribute.terminator` then `Terminator.w_cls`.
///
/// Two field reads on the shared prefix, not `as_terminator()`: that match
/// is a `switch` whose missing default is `unreachable/` (`flatten.py`),
/// and the walker hitting it aborts the trace.
///
/// # Safety
/// `map` must be a live interned [`MapNode`].
#[inline]
pub unsafe fn map_terminator_w_cls(map: MapRef) -> PyObjectRef {
    // Live interned nodes have a non-null terminator (`new_terminator`
    // patches the self-pointer; Plain copies `back.terminator`).
    let term = unsafe { (*(map as *const MapNodeAbstractAttribute)).terminator };
    unsafe { (*(term as *const MapNodeTerminatorView)).w_cls }
}

/// `mapdict.py` `BaseUserClassMapdict.getclass`:
/// `return self._get_mapdict_map().terminator.w_cls`.
///
/// Only a `_getusercls` layout (`typedef.py`) is that mixin: `user_base`
/// names the builtin the generated class extends. Exact `object()` is
/// `INSTANCE_TYPE` (`user_base` null) and uses `W_Root.getclass` /
/// `gettypefor(typeptr)` — the header word. A `_getusercls` carrier with
/// a non-null map reads the terminator; a still-zero map falls back to
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
    // `_get_mapdict_map` is `self.map` (`mapdict.py`). The
    // `W_ObjectObjectUserDictWeakrefable` payload is `W_ObjectObject` at
    // offset 0, so the typed field is that word. Generated `_getusercls`
    // payloads put `map` after the builtin prefix; `mapdict_offset` is
    // that mixin word. The layout stores the interned node as `usize`
    // (`W_ObjectObject.map`); `_mapdict_init_empty` is the zero word.
    let map_word = if std::ptr::eq(tp, &crate::pyobject::INSTANCE_USER_TYPE) {
        unsafe { (*(obj as *const crate::objectobject::W_ObjectObject)).map }
    } else {
        let off = unsafe { (*tp).mapdict_offset };
        if off == 0 {
            return unsafe { (*obj).w_class };
        }
        unsafe { majit_ir::llop::gc_load_indexed::<usize, _>(obj, off, 1, 0) }
    };
    if map_word == 0 {
        return unsafe { (*obj).w_class };
    }
    // MapdictStorageMixin._get_mapdict_map: `return jit.promote(self.map)`.
    // getclass is that word then `terminator.w_cls`; without the promote the
    // class stays a red load and every `type()` / `is_exact_type` inlines a
    // fat residual.
    let map = majit_ir::jit::promote(map_word as MapRef);
    let w_cls = unsafe { map_terminator_w_cls(map) };
    if !w_cls.is_null() {
        return w_cls;
    }
    unsafe { (*obj).w_class }
}

#[cfg(test)]
mod prefix_overlay_tests {
    use super::*;

    fn dummy_terminator(w_cls: PyObjectRef) -> Terminator {
        Terminator {
            terminator: std::ptr::null(),
            w_cls,
            allow_unboxing: Cell::new(true),
            allow_unboxing_watchers: QuasiImmutField::new(),
            kind: TerminatorKind::NoDict,
            devolved_dict_terminator: Cell::new(std::ptr::null()),
            cache_attrs: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    #[test]
    fn terminator_prefix_overlay_reads_terminator_and_w_cls() {
        let w_cls = 0x2000 as PyObjectRef;
        let mut node = MapNode::Terminator(dummy_terminator(w_cls));
        let map = &mut node as *mut MapNode;
        // `new_terminator` patches the self-pointer once the address is known.
        unsafe {
            (*map).as_terminator(); // type-check the variant
            if let MapNode::Terminator(t) = &mut *map {
                t.terminator = map;
            }
        }
        let prefix = unsafe { &*(map as *const MapNodeAbstractAttribute) };
        assert!(std::ptr::eq(prefix.terminator, map));
        let view = unsafe { &*(map as *const MapNodeTerminatorView) };
        assert!(std::ptr::eq(view.w_cls, w_cls));
        assert!(std::ptr::eq(unsafe { map_terminator_w_cls(map) }, w_cls));
    }
}
