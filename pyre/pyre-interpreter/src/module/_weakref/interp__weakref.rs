//! pypy/module/_weakref/interp__weakref.py
//!
//! Structural port of `WeakrefLifeline`, `W_WeakrefBase`, `W_Weakref`,
//! `W_AbstractProxy`, `W_Proxy`, `W_CallableProxy`, plus the module-level
//! `getlifeline`, `descr__new__weakref`, `proxy`, `force`, etc.
//!
//! The class hierarchy and method names mirror PyPy so that `__weakref__` and
//! `_weakref.{ref,proxy,...}` go through the same code paths as the original.

#![allow(non_camel_case_types, non_snake_case)]

use crate::{PyError, make_builtin_function, make_builtin_function_with_arity};
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

static WEAKREF_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static PROXY_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static CALLABLE_PROXY_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

// ── Instance attribute names ──────────────────────────────────────────
//
// W_Weakref / W_AbstractProxy still carry these fields as private instance
// attributes. WeakrefLifeline's fields use its typed inline layout instead.

const ATTR_W_OBJ_WEAK: &str = "w_obj_weak";
const ATTR_W_CALLABLE: &str = "w_callable";
const ATTR_W_HASH: &str = "w_hash";

/// Private layout-slot names reserved on a `weakref.ref` subclass that
/// declares a non-empty `__slots__`. Such an instance has no dict, so the
/// three fields above have no other object-resident storage; type creation
/// appends these names to the subclass Layout (`call.rs` `create_all_slots`)
/// and they carry no Member, being interpreter storage rather than attributes
/// the class exposes.
///
/// Each name holds a space, so no class can declare one: `__slots__` entries
/// are rejected unless they are identifiers. A spellable name would collide —
/// `__slots__ = ('__weakref_obj__',)` installs the user's member first and the
/// reserved name is appended after it, so `field_slot` would answer with the
/// user's index and weakref construction would overwrite their slot.
pub const RESERVED_FIELD_SLOTS: [&str; 3] = [
    "weakref ref w_obj_weak",
    "weakref ref w_callable",
    "weakref ref w_hash",
];

/// Layout-slot index reserved for the field `name` on `obj`, walking the
/// layout chain from the object's type toward its base. `None` when no layout
/// in the chain reserves the slot, which is the ordinary dict-backed case.
fn field_slot(obj: PyObjectRef, name: &str) -> Option<u32> {
    let slot_name = match name {
        ATTR_W_OBJ_WEAK => RESERVED_FIELD_SLOTS[0],
        ATTR_W_CALLABLE => RESERVED_FIELD_SLOTS[1],
        ATTR_W_HASH => RESERVED_FIELD_SLOTS[2],
        _ => return None,
    };
    unsafe {
        let w_type = crate::typedef::r#type(obj).map_or(PY_NULL, |p| p.as_ptr());
        if w_type.is_null() {
            return None;
        }
        let mut layout = pyre_object::w_type_get_layout_ptr(w_type);
        while !layout.is_null() {
            let current = &*layout;
            let first_slot = current
                .nslots
                .saturating_sub(current.newslotnames.len() as u32);
            if let Some(offset) = current
                .newslotnames
                .iter()
                .position(|candidate| candidate == slot_name)
            {
                return Some(first_slot + offset as u32);
            }
            layout = current.base_layout;
        }
        None
    }
}

/// Read a per-instance field from its reserved layout slot, or from the
/// instance dict, bypassing the public `getattr` path. The proxy fast-path in
/// `baseobjspace::getattr_str` would otherwise force the receiver and recurse
/// indefinitely while the proxy is reading its OWN `w_obj_weak`/etc.
fn read_attr(obj: PyObjectRef, name: &str) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    if let Some(slot) = field_slot(pyre_object::gc_roots::shadow_stack_get(root_base), name) {
        let value = unsafe {
            crate::objspace::std::mapdict::getslotvalue(
                pyre_object::gc_roots::shadow_stack_get(root_base),
                slot,
            )
        }
        .unwrap_or(PY_NULL);
        if value.is_null() || unsafe { pyre_object::is_none(value) } {
            return PY_NULL;
        }
        return value;
    }
    let w_dict =
        crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(root_base));
    if w_dict.is_null() {
        return PY_NULL;
    }
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    let value = unsafe {
        pyre_object::w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(root_base + 1),
            name,
        )
    }
    .unwrap_or(PY_NULL);
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        return PY_NULL;
    }
    value
}

/// Mirror of `read_attr` for writes — direct slot / dict access keeps lifeline /
/// proxy / weakref bookkeeping out of the fast-path's force loop.
fn write_attr(obj: PyObjectRef, name: &str, value: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let _ = pyre_object::gc_roots::pin_root(value);
    if let Some(slot) = field_slot(pyre_object::gc_roots::shadow_stack_get(root_base), name) {
        unsafe {
            crate::objspace::std::mapdict::setslotvalue(
                pyre_object::gc_roots::shadow_stack_get(root_base),
                slot,
                pyre_object::gc_roots::shadow_stack_get(root_base + 1),
            )
        };
        return;
    }
    let w_dict =
        crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(root_base));
    if !w_dict.is_null() {
        let _ = pyre_object::gc_roots::pin_root(w_dict);
        unsafe {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(root_base + 2),
                name,
                pyre_object::gc_roots::shadow_stack_get(root_base + 1),
            )
        };
    }
}

/// Field bridge for typed `W_Weakref` and mapdict `W_AbstractProxy` objects.
/// `W_WeakrefBase` (interp__weakref.py) supplies the same fields to
/// both; only their physical composition differs.
#[inline]
fn weakref_obj_weak(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_obj_weak(obj) }
    } else {
        read_attr(obj, ATTR_W_OBJ_WEAK)
    }
}

#[inline]
fn weakref_clear(mut obj: PyObjectRef) {
    let slot = pyre_object::with_roots!(obj => weakref_obj_weak(obj));
    if unsafe { pyre_object::weakref::is_gc_weakref_box(slot) } {
        unsafe { pyre_object::weakref::w_gc_weakref_box_clear(slot) };
    } else if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_set_obj_weak(obj, pyre_object::PY_NULL) };
    } else {
        write_attr(obj, ATTR_W_OBJ_WEAK, pyre_object::PY_NULL);
    }
}

#[inline]
fn weakref_callable(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_callable(obj) }
    } else {
        read_attr(obj, ATTR_W_CALLABLE)
    }
}

#[inline]
fn weakref_set_callable(obj: PyObjectRef, value: PyObjectRef) {
    if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_set_callable(obj, value) };
    } else {
        write_attr(obj, ATTR_W_CALLABLE, value);
    }
}

#[inline]
fn weakref_hash(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_hash(obj) }
    } else {
        read_attr(obj, ATTR_W_HASH)
    }
}

#[inline]
fn weakref_set_hash(obj: PyObjectRef, value: PyObjectRef) {
    if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        unsafe { pyre_object::weakref::w_weakref_object_set_hash(obj, value) };
    } else {
        write_attr(obj, ATTR_W_HASH, value);
    }
}

/// Scoped GC root for a freshly-allocated instance still held only in a
/// Rust local. The proxy constructors allocate the instance, then allocate a
/// `GcWeakrefBox` (an `rweakref` `Weakref` via `try_gc_alloc`) and grow the
/// instance dict — each is a nursery allocation that can drive a minor
/// collection. Without a root the not-yet-reachable instance is swept (its
/// header zeroed), and a later `write_attr` / `typedef::r#type` dereferences
/// the dangling pointer.
/// Registering the slot keeps the instance alive and relocates the local
/// in place when the collector promotes it, mirroring `FrameLocalsRoot`.
struct InstanceRoot {
    slot: *mut pyre_object::gc_hook::GCREF,
    registered: bool,
}

impl InstanceRoot {
    fn new(obj: &mut PyObjectRef) -> Self {
        let slot = obj as *mut PyObjectRef as *mut pyre_object::gc_hook::GCREF;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for InstanceRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

// ── Type registration ─────────────────────────────────────────────────

/// pypy/module/_weakref/interp__weakref.py W_Weakref.typedef
///
/// ```python
/// W_Weakref.typedef = TypeDef("weakref",
///     __doc__ = """A weak reference to an object 'obj'.  A 'callback' can be given,
/// which is called with 'obj' as an argument when it is about to be finalized.""",
///     __new__ = interp2app(descr__new__weakref),
///     __init__ = interp2app(W_Weakref.descr__init__weakref),
///     __eq__ = interp2app(W_Weakref.descr__eq__),
///     __ne__ = interp2app(W_Weakref.descr__ne__),
///     __hash__ = interp2app(W_Weakref.descr_hash),
///     __call__ = interp2app(W_Weakref.descr_call),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
/// )
/// ```
fn init_weakref_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // [3.14-spec] PyPy `W_Weakref.typedef` supplies a descriptive string,
    // while CPython 3.14 `_PyWeakref_RefType.tp_doc` is null.  Leave the key
    // to `ensure_common_attributes`, which publishes the observable
    // `ReferenceType.__dict__["__doc__"] is None` without changing PyPy's
    // weakref payload or ownership.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            crate::typedef::make_new_descr(descr__new__weakref_typecall)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__init__",
            make_builtin_function("__init__", descr__init__weakref)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__eq__",
            make_builtin_function_with_arity("__eq__", descr__eq__, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ne__",
            make_builtin_function_with_arity("__ne__", descr__ne__, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__hash__",
            make_builtin_function_with_arity("__hash__", descr_hash, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__call__",
            make_builtin_function_with_arity("__call__", descr_call, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            make_builtin_function_with_arity("__repr__", descr__repr__, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__callback__",
            crate::typedef::make_getset_property_named(
                make_builtin_function_with_arity("__callback__", descr_callback, 2),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                "__callback__",
            )
        )
    };
    // CPython 3.14 Objects/weakrefobject.c:493 — Py_GenericAlias with
    // METH_CLASS.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            ))
        )
    };
}

pub fn weakref_type() -> PyObjectRef {
    WEAKREF_TYPE.get_or_init(|| {
        // CPython exposes `_weakref.ref` as `weakref.ReferenceType`.
        // The dotted builtin name supplies the public module while the type
        // metadata getters expose the final component as the bare name.
        let tp = crate::typedef::make_builtin_type_with_layout(
            "weakref.ReferenceType",
            init_weakref_type,
            crate::typedef::w_object(),
            &pyre_object::weakref::WEAKREF_LAYOUT_TYPE as *const PyType,
        );
        // This mixed-module type is created lazily after the eager
        // `init_typeobjects` registry pass. Bind its translated static layout
        // now, exactly as `py_class_typed!` and `getset_descriptor_type` do,
        // so W_Weakref::allocate_stable stamps the live ReferenceType class.
        pyre_object::set_instantiate(&pyre_object::weakref::WEAKREF_LAYOUT_TYPE, tp);
        tp
    })
}

/// pypy/module/_weakref/interp__weakref.py W_Proxy.typedef
///
/// ```python
/// W_Proxy.typedef = TypeDef("weakproxy",
///     __new__ = interp2app(descr__new__proxy),
///     __hash__ = interp2app(W_Proxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     **proxy_typedef_dict)
/// W_Proxy.typedef.acceptable_as_base_class = False
/// ```
///
/// `__hash__` departs from that quote: `_PyWeakref_ProxyType` leaves
/// `tp_hash` NULL while filling `tp_richcompare`, so slot inheritance never
/// copies `object`'s hash in and the attribute reads as `None`.
fn init_proxy_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            crate::typedef::make_new_descr(descr__new__proxy)
        )
    };
    // A NULL `tp_hash` reads back as `__hash__ is None`, and the generic
    // hash path then reports `unhashable type: 'weakref.ProxyType'` where
    // `W_Proxy.descr__hash__` raises a bare `unhashable type`.
    unsafe { crate::__pyre_put_new!(ns_slot, "__hash__", pyre_object::w_none()) };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            make_builtin_function_with_arity("__repr__", descr__repr__, 1)
        )
    };
    // **proxy_typedef_dict — interp__weakref.py.
    register_proxy_typedef_dict(pyre_object::gc_roots::shadow_stack_get(ns_slot));
}

/// `dont_look_inside`: the `PROXY_TYPE` process-global `OnceLock` read has
/// no extractable graph; the call stays a residual returning the lazily
/// built type object (same shape as [`crate::typedef::w_type`]).
#[majit_macros::dont_look_inside]
pub fn proxy_type() -> PyObjectRef {
    PROXY_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "weakref.ProxyType",
            init_proxy_type,
            crate::typedef::w_object(),
            &pyre_object::weakref::WEAKREF_LAYOUT_TYPE as *const PyType,
        );
        unsafe {
            pyre_object::w_type_set_hasdict(tp, true);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            // `proxy_typedef_dict` `__getattribute__` (`proxy_getattribute`).
            pyre_object::w_type_set_dispatch_own_getattribute(tp);
        }
        tp
    })
}

/// pypy/module/_weakref/interp__weakref.py W_CallableProxy.typedef
///
/// ```python
/// W_CallableProxy.typedef = TypeDef("weakcallableproxy",
///     __new__ = interp2app(descr__new__callableproxy),
///     __hash__ = interp2app(W_CallableProxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     __call__ = interp2app(W_CallableProxy.descr__call__),
///     **callable_proxy_typedef_dict)
/// W_CallableProxy.typedef.acceptable_as_base_class = False
/// ```
///
/// Two departures from that quote, both taken from
/// `_PyWeakref_CallableProxyType`: `tp_hash` stays NULL (see
/// [`init_proxy_type`]), and `tp_richcompare` is `proxy_richcompare`, the
/// same slot the non-callable proxy installs, so the comparison rows of
/// [`register_proxy_typedef_dict`] land on this typedef too.
fn init_callable_proxy_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            crate::typedef::make_new_descr(descr__new__callableproxy)
        )
    };
    // `unhashable type: 'weakref.CallableProxyType'`, for the reason given
    // in `init_proxy_type`.
    unsafe { crate::__pyre_put_new!(ns_slot, "__hash__", pyre_object::w_none()) };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            make_builtin_function_with_arity("__repr__", descr__repr__, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__call__",
            make_builtin_function("__call__", callable_proxy_descr__call__)
        )
    };
    // **callable_proxy_typedef_dict — interp__weakref.py, plus the
    // comparison rows PyPy writes only to `proxy_typedef_dict`.
    register_proxy_typedef_dict(pyre_object::gc_roots::shadow_stack_get(ns_slot));
}

/// `dont_look_inside` for the same reason as [`proxy_type`].
#[majit_macros::dont_look_inside]
pub fn callable_proxy_type() -> PyObjectRef {
    CALLABLE_PROXY_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "weakref.CallableProxyType",
            init_callable_proxy_type,
            crate::typedef::w_object(),
            &pyre_object::weakref::WEAKREF_LAYOUT_TYPE as *const PyType,
        );
        unsafe {
            pyre_object::w_type_set_hasdict(tp, true);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            // `callable_proxy_typedef_dict` shares `proxy_getattribute`.
            pyre_object::w_type_set_dispatch_own_getattribute(tp);
        }
        tp
    })
}

// ── WeakrefLifeline ───────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py
//
// class WeakrefLifeline(W_Root):
//     cached_weakref  = None
//     cached_proxy    = None
//     other_refs_weak = None
//     has_callbacks   = False
//
/// pypy/module/_weakref/interp__weakref.py WeakrefLifeline.__init__
///
/// ```python
/// def __init__(self, space):
///     self.space = space
/// ```
pub fn weakref_lifeline_new() -> PyObjectRef {
    pyre_object::weakref::w_weakref_lifeline_new()
}

/// pypy/module/_weakref/interp__weakref.py:30-33
/// `WeakrefLifeline.append_wref_to`.
fn append_wref_to(self_lifeline: PyObjectRef, w_ref: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_lifeline);
    let _ = pyre_object::gc_roots::pin_root(w_ref);
    let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(
        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
    );
    let _ = pyre_object::gc_roots::pin_root(weak);
    let lifeline = pyre_object::gc_roots::shadow_stack_get(root_base);
    let mut refs = unsafe { pyre_object::weakref::w_weakref_lifeline_other_refs(lifeline) };
    if refs.is_null() {
        refs = pyre_object::w_list_new_object(vec![pyre_object::gc_roots::shadow_stack_get(
            root_base + 2,
        )]);
        unsafe {
            pyre_object::weakref::w_weakref_lifeline_set_other_refs(
                pyre_object::gc_roots::shadow_stack_get(root_base),
                refs,
            )
        };
    } else {
        unsafe {
            pyre_object::w_list_append(refs, pyre_object::gc_roots::shadow_stack_get(root_base + 2))
        };
    }
}

/// pypy/module/_weakref/interp__weakref.py:105-109
/// `WeakrefLifeline.enable_callbacks`.
fn enable_callbacks(self_lifeline: PyObjectRef) {
    if unsafe { pyre_object::weakref::w_weakref_lifeline_has_callbacks(self_lifeline) } {
        return;
    }
    unsafe { pyre_object::weakref::w_weakref_lifeline_set_has_callbacks(self_lifeline) };
    // interp__weakref.py enable_callbacks — the lifeline, not its referent, owns the
    // finalizer. This preserves the collector's ordering for referents that
    // are themselves part of cyclic trash.
    crate::executioncontext::register_finalizer(self_lifeline);
}

/// pypy/module/_weakref/interp__weakref.py get_or_make_weakref
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_weakref(self, w_subtype, w_obj):
///     space = self.space
///     w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///     #
///     if space.is_w(w_weakreftype, w_subtype):
///         if self.cached_weakref is not None:
///             w_cached = self.cached_weakref()
///             if w_cached is not None:
///                 return w_cached
///         w_ref = W_Weakref(space, w_obj, None)
///         self.cached_weakref = weakref.ref(w_ref)
///     else:
///         # subclass: cannot cache
///         w_ref = space.allocate_instance(W_Weakref, w_subtype)
///         W_Weakref.__init__(w_ref, space, w_obj, None)
///         self.append_wref_to(w_ref)
///     return w_ref
/// ```
pub fn get_or_make_weakref(
    mut self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
) -> PyObjectRef {
    let w_weakreftype = weakref_type();
    if w_subtype.is_null() || std::ptr::eq(w_weakreftype, w_subtype) {
        // interp__weakref.py:66-69: cached_weakref is a weakref TO the
        // W_Weakref; w_cached = self.cached_weakref() returns the
        // W_Weakref or None.
        let cached_slot =
            unsafe { pyre_object::weakref::w_weakref_lifeline_cached_weakref(self_lifeline) };
        let cached = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(cached_slot) };
        if !cached.is_null() {
            return cached;
        }
        let mut w_ref =
            pyre_object::with_roots!(self_lifeline => W_Weakref_new(w_subtype, w_obj, PY_NULL));
        let _root = InstanceRoot::new(&mut w_ref);
        let cached = pyre_object::with_roots!(self_lifeline, w_ref => pyre_object::weakref::w_gc_weakref_box_new_or_strong(w_ref));
        unsafe {
            pyre_object::weakref::w_weakref_lifeline_set_cached_weakref(self_lifeline, cached)
        };
        w_ref
    } else {
        // subclass: cannot cache
        let mut w_ref =
            pyre_object::with_roots!(self_lifeline => W_Weakref_new(w_subtype, w_obj, PY_NULL));
        pyre_object::with_roots!(w_ref => append_wref_to(self_lifeline, w_ref));
        w_ref
    }
}

/// pypy/module/_weakref/interp__weakref.py get_or_make_proxy
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_proxy(self, w_obj):
///     space = self.space
///     if self.cached_proxy is not None:
///         w_cached = self.cached_proxy()
///         if w_cached is not None:
///             return w_cached
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, None)
///     else:
///         w_proxy = W_Proxy(space, w_obj, None)
///     self.cached_proxy = weakref.ref(w_proxy)
///     return w_proxy
/// ```
pub fn get_or_make_proxy(mut self_lifeline: PyObjectRef, w_obj: PyObjectRef) -> PyObjectRef {
    // interp__weakref.py: cached_proxy is a weakref TO the W_Proxy /
    // W_CallableProxy; w_cached = self.cached_proxy() returns the proxy
    // or None.
    let cached_slot =
        unsafe { pyre_object::weakref::w_weakref_lifeline_cached_proxy(self_lifeline) };
    let cached = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(cached_slot) };
    if !cached.is_null() {
        return cached;
    }
    let mut w_proxy = if is_callable(w_obj) {
        pyre_object::with_roots!(self_lifeline => W_CallableProxy_new(w_obj, PY_NULL))
    } else {
        pyre_object::with_roots!(self_lifeline => W_Proxy_new(w_obj, PY_NULL))
    };
    let cached = pyre_object::with_roots!(self_lifeline, w_proxy => pyre_object::weakref::w_gc_weakref_box_new_or_strong(w_proxy));
    unsafe { pyre_object::weakref::w_weakref_lifeline_set_cached_proxy(self_lifeline, cached) };
    w_proxy
}

/// pypy/module/_weakref/interp__weakref.py get_any_weakref
///
/// ```python
/// def get_any_weakref(self, space):
///     if self.cached_weakref is not None:
///         w_ref = self.cached_weakref()
///         if w_ref is not None:
///             return w_ref
///     if self.other_refs_weak is not None:
///         w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///         for wref in self.other_refs_weak.items():
///             w_ref = wref()
///             if (w_ref is not None and space.isinstance_w(w_ref, w_weakreftype)):
///                 return w_ref
///     return space.w_None
/// ```
pub fn get_any_weakref(self_lifeline: PyObjectRef) -> PyObjectRef {
    for w_ref in lifeline_refs(self_lifeline) {
        if is_w_weakref(w_ref) {
            return w_ref;
        }
    }
    pyre_object::w_none()
}

/// pypy/module/_weakref/interp__weakref.py make_weakref_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_weakref_with_callback(self, w_subtype, w_obj, w_callable):
///     space = self.space
///     w_ref = space.allocate_instance(W_Weakref, w_subtype)
///     W_Weakref.__init__(w_ref, space, w_obj, w_callable)
///     self.append_wref_to(w_ref)
///     self.enable_callbacks()
///     return w_ref
/// ```
pub fn make_weakref_with_callback(
    mut self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    let mut w_ref =
        pyre_object::with_roots!(self_lifeline => W_Weakref_new(w_subtype, w_obj, w_callable));
    pyre_object::with_roots!(self_lifeline, w_ref => append_wref_to(self_lifeline, w_ref));
    enable_callbacks(self_lifeline);
    w_ref
}

/// pypy/module/_weakref/interp__weakref.py make_proxy_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_proxy_with_callback(self, w_obj, w_callable):
///     space = self.space
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, w_callable)
///     else:
///         w_proxy = W_Proxy(space, w_obj, w_callable)
///     self.append_wref_to(w_proxy)
///     self.enable_callbacks()
///     return w_proxy
/// ```
pub fn make_proxy_with_callback(
    mut self_lifeline: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    let mut w_proxy = if is_callable(w_obj) {
        pyre_object::with_roots!(self_lifeline => W_CallableProxy_new(w_obj, w_callable))
    } else {
        pyre_object::with_roots!(self_lifeline => W_Proxy_new(w_obj, w_callable))
    };
    pyre_object::with_roots!(self_lifeline, w_proxy => append_wref_to(self_lifeline, w_proxy));
    enable_callbacks(self_lifeline);
    w_proxy
}

// ── W_WeakrefBase / W_Weakref ─────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py
//
// class W_WeakrefBase(W_Root):
//     def __init__(self, space, w_obj, w_callable):
//         assert w_callable is not space.w_None
//         self.space = space
//         self.w_obj_weak = weakref.ref(w_obj)
//         self.w_callable = w_callable
//
// class W_Weakref(W_WeakrefBase):
//     def __init__(self, space, w_obj, w_callable):
//         W_WeakrefBase.__init__(self, space, w_obj, w_callable)
//         self.w_hash = None

#[allow(non_snake_case)]
pub fn W_Weakref_new(
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    // typedef.py generic_new_descr → space.allocate_instance(W_Type, w_subtype)
    let actual_type = if w_subtype.is_null() {
        weakref_type()
    } else {
        w_subtype
    };
    // `W_Weakref` (interp__weakref.py) / `descr__new__weakref`
    // (interp__weakref.py): the three fields are interpreter-owned.
    // `allocate_instance(W_Weakref, w_subtype)` (`typedef.py` `_getusercls`)
    // stamps a subclass with `WEAKREF_LAYOUT_USER_TYPE`; an exact `ref` keeps
    // the base payload.
    let exact_type = std::ptr::eq(actual_type, weakref_type());
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let _ = pyre_object::gc_roots::pin_root(w_callable);
    let _ = pyre_object::gc_roots::pin_root(actual_type);
    let w_obj_weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(
        pyre_object::gc_roots::shadow_stack_get(root_base),
    );
    let _ = pyre_object::gc_roots::pin_root(w_obj_weak);
    let callable = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let callable = if !callable.is_null() && !unsafe { pyre_object::is_none(callable) } {
        callable
    } else {
        pyre_object::PY_NULL
    };
    let cls = pyre_object::gc_roots::shadow_stack_get(root_base + 2);
    let obj_weak = pyre_object::gc_roots::shadow_stack_get(root_base + 3);
    if exact_type {
        pyre_object::weakref::w_weakref_object_new(obj_weak, callable, pyre_object::PY_NULL)
    } else {
        // The user-finalizer enqueue `tag_subclass_instance` performs after
        // `user_setup`. `w_class` is already the subclass.
        let obj =
            pyre_object::weakref::w_weakref_user_new(obj_weak, callable, pyre_object::PY_NULL, cls);
        pyre_object::gc_hook::maybe_register_finalizer(obj);
        obj
    }
}

#[allow(non_snake_case)]
pub fn W_Proxy_new(w_obj: PyObjectRef, mut w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::objectobject::w_instance_new;
    let mut obj = w_instance_new(proxy_type());
    let _root = InstanceRoot::new(&mut obj);
    let w_obj_weak = pyre_object::with_roots!(obj, w_callable => pyre_object::weakref::w_gc_weakref_box_new_or_strong(w_obj));
    pyre_object::with_roots!(obj, w_callable => write_attr(obj, ATTR_W_OBJ_WEAK, w_obj_weak));
    let w_callable = if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        w_callable
    } else {
        pyre_object::w_none()
    };
    pyre_object::with_roots!(obj => write_attr(obj, ATTR_W_CALLABLE, w_callable));
    obj
}

#[allow(non_snake_case)]
pub fn W_CallableProxy_new(w_obj: PyObjectRef, mut w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::objectobject::w_instance_new;
    let mut obj = w_instance_new(callable_proxy_type());
    let _root = InstanceRoot::new(&mut obj);
    let w_obj_weak = pyre_object::with_roots!(obj, w_callable => pyre_object::weakref::w_gc_weakref_box_new_or_strong(w_obj));
    pyre_object::with_roots!(obj, w_callable => write_attr(obj, ATTR_W_OBJ_WEAK, w_obj_weak));
    let w_callable = if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        w_callable
    } else {
        pyre_object::w_none()
    };
    pyre_object::with_roots!(obj => write_attr(obj, ATTR_W_CALLABLE, w_callable));
    obj
}

/// pypy/module/_weakref/interp__weakref.py dereference
///
/// ```python
/// @jit.dont_look_inside
/// def dereference(self):
///     w_obj = self.w_obj_weak()
///     return w_obj
/// ```
#[majit_macros::dont_look_inside]
pub fn dereference(w_ref: PyObjectRef) -> PyObjectRef {
    let slot = weakref_obj_weak(w_ref);
    unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(slot) }
}

/// pypy/module/_weakref/interp__weakref.py W_WeakrefBase.descr__repr__
///
/// ```python
/// def descr__repr__(self, space):
///     w_obj = self.dereference()
///     if w_obj is None:
///         state = '; dead'
///     else:
///         typename = space.type(w_obj).getname(space)
///         objname = w_obj.getname(space)
///         if objname and objname != '?':
///             state = "; to '%s' (%s)" % (typename, objname)
///         else:
///             state = "; to '%s'" % (typename,)
///     return self.getrepr(space, self.typedef.name, state)
/// ```
///
/// The state follows `weakref_repr` / `proxy_repr` rather than the quote
/// above: `; to '<fully qualified referent type>' at <referent address>`,
/// closing with ` (<name>)` when a reference's referent answers a string
/// `__name__`.  The flavour is the typedef's name and never the receiver's
/// own class: both proxy types report `weakproxy`, and a `ref` subclass
/// reports `weakref`.
pub fn descr__repr__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let is_proxy = is_w_abstract_proxy(w_self);
    // PyPy's interp2app gateway types W_WeakrefBase.descr__repr__ as a
    // `weakref-or-proxy` method before entering this shared body.
    if !is_w_weakref(w_self) && !is_proxy {
        return Err(PyError::type_error(format!(
            "'weakref-or-proxy' object expected, got '{}' instead",
            crate::baseobjspace::object_functionstr_type_name(w_self)
        )));
    }
    let type_name = if is_proxy { "weakproxy" } else { "weakref" };
    // The `__name__` lookup below can run a user-defined property, which
    // allocates.  Pin the reference and its referent across it and read both
    // back out of their slots, so a collection that moves either one cannot
    // leave a stale address in the repr.
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_self);
    let w_obj = dereference(pyre_object::gc_roots::shadow_stack_get(self_slot));
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Ok(pyre_object::w_str_new_managed(&format!(
            "<{type_name} at {}; dead>",
            crate::display::repr_gc_addr(pyre_object::gc_roots::shadow_stack_get(self_slot))
        )));
    }
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    // `weakref_repr` closes the state with the referent's own `__name__`;
    // `proxy_repr` has no such suffix.
    let name = if is_proxy {
        Wtf8Buf::new()
    } else {
        referent_name_suffix(pyre_object::gc_roots::shadow_stack_get(obj_slot))?
    };
    let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    // The referent's type is spelled with `%T`, the `<module>.<qualname>`
    // form, where `getname` gives the bare name.
    let objtype_name = unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => crate::baseobjspace::type_fully_qualified_name(tp.as_ptr()),
            None => rustpython_wtf8::Wtf8Buf::from("object"),
        }
    };
    // The state also carries the referent's own address, which `getrepr`
    // does not. Both ids are taken from the rooted words: the first hash
    // allocates a shadow and can move the other object.
    let self_addr =
        crate::display::repr_gc_addr(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let obj_addr = crate::display::repr_gc_addr(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(pyre_object::w_str_from_wtf8_managed(
        crate::display::wtf8_format!(
            format!("<{type_name} at {self_addr}; to '"),
            objtype_name,
            format!("' at {obj_addr}"),
            name,
            ">",
        ),
    ))
}

/// `weakref_repr`'s trailing ` (<name>)`: the referent's `__name__` resolved
/// through `_PyObject_LookupSpecial`, which reads the type alone, so neither
/// an instance-dict entry nor a forwarding `__getattr__` can invent one
/// (`test_weakref.py test_repr_failure_gh99184`).  Empty when the type
/// defines no `__name__` and when the answer is not a `str`.
///
/// A `__name__` that raises propagates here.  `weakref_repr` leaves that
/// error indicator set and returns a repr regardless, which `repr()` then
/// reports as an unrelated `SystemError`; the exception the lookup actually
/// raised is the useful one.
fn referent_name_suffix(w_obj: PyObjectRef) -> Result<Wtf8Buf, PyError> {
    let looked_up = unsafe { crate::baseobjspace::lookup_special(w_obj, "__name__")? };
    let Some(w_name) = looked_up else {
        return Ok(Wtf8Buf::new());
    };
    if !unsafe { pyre_object::is_str(w_name) } {
        return Ok(Wtf8Buf::new());
    }
    // The name is written with `%U`, which spells it as it stands; copying it
    // out through a `&str` dropped one holding a lone surrogate.  Owned rather
    // than borrowed because the repr allocates after this returns.
    let mut suffix = Wtf8Buf::new();
    suffix.push_str(" (");
    suffix.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_name) });
    suffix.push_str(")");
    Ok(suffix)
}

/// pypy/module/_weakref/interp__weakref.py descr__init__weakref
///
/// ```python
/// def descr__init__weakref(self, space, w_obj, w_callable=None,
///                          __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError,
///                     "__init__ expected at most 2 arguments")
///     if __args__.keywords:
///         raise oefmt(space.w_TypeError,
///                     "ref() does not take keyword arguments")
/// ```
pub fn descr__init__weakref(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // args[0] = self, args[1] = w_obj, args[2] = w_callable (optional)
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 3 {
        return Err(PyError::type_error(
            "__init__ expected at most 2 arguments".to_string(),
        ));
    }
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(PyError::type_error(
            "ref() does not take keyword arguments".to_string(),
        ));
    }
    Ok(pyre_object::w_none())
}

/// pypy/module/_weakref/interp__weakref.py descr_hash
///
/// ```python
/// def descr_hash(self):
///     if self.w_hash is not None:
///         return self.w_hash
///     w_obj = self.dereference()
///     if w_obj is None:
///         raise oefmt(self.space.w_TypeError, "weak object has gone away")
///     self.w_hash = self.space.hash(w_obj)
///     return self.w_hash
/// ```
pub fn descr_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let current_self = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let cached = weakref_hash(current_self());
    if !cached.is_null() {
        return Ok(cached);
    }
    let w_obj = dereference(current_self());
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::type_error("weak object has gone away".to_string()));
    }
    // `self.w_hash = self.space.hash(w_obj)`: the reference hashes as its
    // referent, so a ref and the object it points at land in the same bucket.
    // `hash_w_strict` can run a user-defined `__hash__`, which allocates, so
    // the referent is rooted across the call and `current_self` re-reads the
    // reference afterwards.
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let h = pyre_object::w_int_new(crate::baseobjspace::hash_w_strict(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    )?);
    let h_slot = pyre_object::gc_roots::shadow_stack_len();
    let h = pyre_object::gc_roots::pin_root(h);
    weakref_set_hash(current_self(), h);
    Ok(pyre_object::gc_roots::shadow_stack_get(h_slot))
}

/// pypy/module/_weakref/interp__weakref.py descr_call
///
/// ```python
/// def descr_call(self):
///     w_obj = self.dereference()
///     if w_obj is None:
///         return self.space.w_None
///     return w_obj
/// ```
pub fn descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = dereference(w_self);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Ok(pyre_object::w_none());
    }
    Ok(w_obj)
}

/// pypy/module/_weakref/interp__weakref.py W_Weakref.compare
///
/// ```python
/// def compare(self, space, w_ref2, invert):
///     if not isinstance(w_ref2, W_Weakref):
///         return space.w_NotImplemented
///     ref1 = self
///     ref2 = w_ref2
///     w_obj1 = ref1.dereference()
///     w_obj2 = ref2.dereference()
///     if w_obj1 is None or w_obj2 is None:
///         w_res = space.is_(ref1, ref2)
///     else:
///         w_res = space.eq(w_obj1, w_obj2)
///     if invert:
///         w_res = space.not_(w_res)
///     return w_res
/// ```
fn compare(
    mut w_self: PyObjectRef,
    mut w_ref2: PyObjectRef,
    invert: bool,
) -> Result<PyObjectRef, PyError> {
    if !is_w_weakref(w_ref2) {
        return Ok(pyre_object::w_not_implemented());
    }
    let mut w_obj1 = pyre_object::with_roots!(w_ref2, w_self => dereference(w_self));
    let w_obj2 = pyre_object::with_roots!(w_obj1, w_ref2, w_self => dereference(w_ref2));
    let w_res = if w_obj1.is_null()
        || unsafe { pyre_object::is_none(w_obj1) }
        || w_obj2.is_null()
        || unsafe { pyre_object::is_none(w_obj2) }
    {
        crate::baseobjspace::is_(w_self, w_ref2)
    } else {
        crate::baseobjspace::compare(w_obj1, w_obj2, crate::baseobjspace::CompareOp::Eq)?
    };
    if invert {
        crate::baseobjspace::not_(w_res)
    } else {
        Ok(w_res)
    }
}

/// pypy/module/_weakref/interp__weakref.py descr__eq__
pub fn descr__eq__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], false)
}

/// pypy/module/_weakref/interp__weakref.py descr__ne__
pub fn descr__ne__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], true)
}

/// pypy/module/_weakref/interp__weakref.py:252-253
/// `__callback__ = GetSetProperty(W_Weakref.descr_callback)`.
/// GetSetProperty passes `(descriptor, instance)` to the getter.
pub fn descr_callback(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_callable = weakref_callable(args[1]);
    if w_callable.is_null() {
        Ok(pyre_object::w_none())
    } else {
        Ok(w_callable)
    }
}

fn is_w_weakref(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { crate::baseobjspace::isinstance_w(obj, weakref_type()) }
}

/// PyPy `app_weakref._remove_dead_weakref`, backed by
/// `W_DictMultiObject.nondescr_delitem_if_value_is`.
///
/// The lookup and deletion operate on the intrinsic dict backing so dict
/// subclass hooks are bypassed.  Rechecking pointer identity immediately
/// before deletion preserves the upstream atomic "delete only this weakref"
/// contract if key comparison mutates the dictionary.
pub fn remove_dead_weakref(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let dict = args
        .first()
        .copied()
        .ok_or_else(|| PyError::type_error("_remove_dead_weakref() missing dict argument"))?;
    let mut key = args
        .get(1)
        .copied()
        .ok_or_else(|| PyError::type_error("_remove_dead_weakref() missing key argument"))?;
    if !unsafe { pyre_object::is_dict(dict) } {
        return Err(PyError::type_error(format!(
            "_remove_dead_weakref() argument 1 must be dict, not {}",
            crate::type_methods::clinic_arg_type_name(dict),
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[dict, key]);
    let mut backing =
        crate::type_methods::resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(base));
    let dict = pyre_object::gc_roots::shadow_stack_get(base);
    key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if backing.is_null() {
        return Err(PyError::type_error(format!(
            "_remove_dead_weakref() argument 1 must be dict, not {}",
            crate::type_methods::clinic_arg_type_name(dict),
        )));
    }
    let Some(stored) =
        pyre_object::with_roots!(backing, key => crate::baseobjspace::finditem(backing, key))?
    else {
        return Ok(pyre_object::w_none());
    };
    // app_weakref.py calls the retrieved value directly (`wr()`) rather than
    // requiring a weakref instance.  Keep backing/key/stored rooted across
    // that arbitrary call, then use the native identity-checked deletion
    // primitive exactly as `delitem_if_value_is(d, key, wr)` does upstream.
    let _roots = pyre_object::gc_roots::push_roots();
    // All three words already exist. Sequential `pin_root` would normalize
    // after the first write and leave the later values unpublished
    // (`RootScope::pin_roots`).
    let base = pyre_object::gc_roots::pin_roots(&[backing, key, stored]);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(base + 2),
        &[],
    )?;
    if !unsafe { pyre_object::is_none(result) } {
        return Ok(pyre_object::w_none());
    }
    let backing = pyre_object::gc_roots::shadow_stack_get(base);
    let key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let stored = pyre_object::gc_roots::shadow_stack_get(base + 2);
    crate::baseobjspace::dict_delitem_if_value_is(backing, key, stored)?;
    Ok(pyre_object::w_none())
}

fn is_callable(obj: PyObjectRef) -> bool {
    // interp__weakref.py:87/124 `space.is_true(space.callable(w_obj))`.
    !obj.is_null() && crate::baseobjspace::callable_w(obj)
}

// ── module-level helpers ──────────────────────────────────────────────

/// pypy/module/_weakref/interp__weakref.py getlifeline
///
/// ```python
/// def getlifeline(space, w_obj):
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         lifeline = WeakrefLifeline(space)
///         w_obj.setweakref(space, lifeline)
///     return lifeline
/// ```
pub fn getlifeline(w_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    if let Some(lifeline) =
        crate::baseobjspace::getweakref(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    {
        return Ok(lifeline);
    }
    let lifeline = weakref_lifeline_new();
    let life_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lifeline);
    crate::baseobjspace::setweakref(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        pyre_object::gc_roots::shadow_stack_get(life_slot),
    )?;
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    crate::cpyext::remember_weakref_lifeline(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        pyre_object::gc_roots::shadow_stack_get(life_slot),
    );
    Ok(pyre_object::gc_roots::shadow_stack_get(life_slot))
}

fn lifeline_refs(lifeline: PyObjectRef) -> Vec<PyObjectRef> {
    let mut refs = Vec::new();
    for weak in unsafe {
        [
            pyre_object::weakref::w_weakref_lifeline_cached_weakref(lifeline),
            pyre_object::weakref::w_weakref_lifeline_cached_proxy(lifeline),
        ]
    } {
        let w_ref = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(weak) };
        if !w_ref.is_null() {
            refs.push(w_ref);
        }
    }
    let others = unsafe { pyre_object::weakref::w_weakref_lifeline_other_refs(lifeline) };
    if !others.is_null() {
        let length = unsafe { pyre_object::w_list_len(others) };
        for i in 0..length {
            let Some(weak) = (unsafe { pyre_object::w_list_getitem(others, i as i64) }) else {
                continue;
            };
            let w_ref = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(weak) };
            if !w_ref.is_null() {
                refs.push(w_ref);
            }
        }
    }
    refs
}

/// `WeakrefLifeline.clear_all_weakrefs` — eagerly make every reference and
/// proxy in this lifeline dead, leaving callback delivery to `_finalize_`.
pub fn clear_all_weakrefs(lifeline: PyObjectRef) {
    for w_ref in lifeline_refs(lifeline) {
        weakref_clear(w_ref);
    }
}

/// `WeakrefLifeline._finalize_` (interp__weakref.py), invoked
/// from pyre's shared finalizer queue when the lifeline becomes unreachable.
pub fn finalize_weakrefs(w_obj: PyObjectRef) {
    let is_lifeline =
        unsafe { pyre_object::py_type_check(w_obj, &pyre_object::weakref::WEAKREF_LIFELINE_TYPE) };
    if !is_lifeline {
        return;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    // RPython's local `items` list is GC-transformed together with every
    // element. A Rust Vec only holds copied raw pointers, so retain the same
    // collection in shadow-stack slots while reads below can allocate.
    let mut ref_slots = Vec::new();
    let others = unsafe { pyre_object::weakref::w_weakref_lifeline_other_refs(w_obj) };
    if !others.is_null() {
        let others_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(others);
        let length = unsafe {
            pyre_object::w_list_len(pyre_object::gc_roots::shadow_stack_get(others_slot))
        };
        for i in 0..length {
            let Some(weak) = (unsafe {
                pyre_object::w_list_getitem(
                    pyre_object::gc_roots::shadow_stack_get(others_slot),
                    i as i64,
                )
            }) else {
                continue;
            };
            let w_ref = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(weak) };
            if !w_ref.is_null() {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(w_ref);
                ref_slots.push(slot);
            }
        }
    }
    // interp__weakref.py:143-145 — detach the shrink-list before callbacks,
    // then activate surviving weakrefs in reverse creation order. The GC's
    // rweakref pass has already invalidated each dead referent.
    unsafe { pyre_object::weakref::w_weakref_lifeline_set_other_refs(w_obj, pyre_object::PY_NULL) };
    for &slot in ref_slots.iter().rev() {
        let w_ref = pyre_object::gc_roots::shadow_stack_get(slot);
        let w_callable = weakref_callable(w_ref);
        if w_callable.is_null() {
            continue;
        }
        let _call_roots = pyre_object::gc_roots::push_roots();
        let call_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_callable);
        let current_ref = || pyre_object::gc_roots::shadow_stack_get(slot);
        let current_callable = || pyre_object::gc_roots::shadow_stack_get(call_base);
        if let Err(error) =
            crate::call::call_function_impl_result(current_callable(), &[current_ref()])
        {
            let ec = crate::call::getexecutioncontext();
            if !ec.is_null() {
                crate::executioncontext::report_error(
                    unsafe { (*ec).space },
                    &error,
                    rustpython_wtf8::Wtf8::new("weakref callback "),
                    current_callable(),
                );
            }
        }
        weakref_set_callable(current_ref(), pyre_object::PY_NULL);
    }
}

/// pypy/module/_weakref/interp__weakref.py descr__new__weakref
///
/// ```python
/// def descr__new__weakref(space, w_subtype, w_obj, w_callable=None,
///                         __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError, "__new__ expected at most 2 arguments")
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_weakref(w_subtype, w_obj)
///     else:
///         return lifeline.make_weakref_with_callback(w_subtype, w_obj, w_callable)
/// ```
pub fn descr__new__weakref(
    w_subtype: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    if args.len() > 2 {
        return Err(PyError::type_error(
            "__new__ expected at most 2 arguments".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let _roots = pyre_object::gc_roots::push_roots();
    // All three words exist. One `pin_roots` is `push_roots` of that live
    // set; sequential `pin_root` normalizes after the first write
    // (`RootScope::pin_roots`).
    let base = pyre_object::gc_roots::pin_roots(&[w_obj, w_callable, w_subtype]);
    let lifeline = getlifeline(pyre_object::gc_roots::shadow_stack_get(base))?;
    let w_obj = pyre_object::gc_roots::shadow_stack_get(base);
    let w_callable = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let w_subtype = pyre_object::gc_roots::shadow_stack_get(base + 2);
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_weakref(lifeline, w_subtype, w_obj))
    } else {
        Ok(make_weakref_with_callback(
            lifeline, w_subtype, w_obj, w_callable,
        ))
    }
}

/// Type-call entry: pyre passes `(cls, *args)` so we strip cls and route
/// through descr__new__weakref with the user-supplied subtype.
fn descr__new__weakref_typecall(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    // PyPy's trailing `__args__` captures keywords separately: __new__ uses
    // only its positional `w_obj` / `w_callable` slots, while __init__ below
    // is responsible for rejecting keywords on the exact builtin type. Do
    // not mistake pyre's flat keyword-marker dict for the callback object.
    let (positional, _kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    descr__new__weakref(args[0], positional)
}

/// pypy/module/_weakref/interp__weakref.py getweakrefcount
///
/// ```python
/// def getweakrefcount(space, w_obj):
///     """Return the number of weak references to 'obj'."""
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         return space.newint(0)
///     else:
///         result = lifeline.traverse(_weakref_count, 0)
///         return space.newint(result)
/// ```
pub fn getweakrefcount(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_int_new(0));
    }
    let w_obj = args[0];
    let count = match crate::baseobjspace::getweakref(w_obj) {
        None => 0,
        Some(lifeline) => lifeline_refs(lifeline).len(),
    };
    Ok(pyre_object::w_int_new(count as i64))
}

/// pypy/module/_weakref/interp__weakref.py getweakrefs
///
/// ```python
/// def getweakrefs(space, w_obj):
///     """Return a list of all weak reference objects that point to 'obj'."""
///     result = []
///     lifeline = w_obj.getweakref()
///     if lifeline is not None:
///         lifeline.traverse(_get_weakrefs, result)
///     return space.newlist(result)
/// ```
pub fn getweakrefs(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_list_new(vec![]));
    }
    let w_obj = args[0];
    let result = crate::baseobjspace::getweakref(w_obj)
        .map(lifeline_refs)
        .unwrap_or_default();
    Ok(pyre_object::w_list_new(result))
}

// ── Proxy ─────────────────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:311-417

// `W_Proxy.descr__hash__` / `W_CallableProxy.descr__hash__`
// (interp__weakref.py) raise a bare `unhashable type`.  Both typedefs bind
// `__hash__` to `None` instead — see `init_proxy_type` — so the generic hash
// path names the type, as a NULL `tp_hash` does on the proxy types.

/// pypy/module/_weakref/interp__weakref.py W_CallableProxy.descr__call__
///
/// ```python
/// def descr__call__(self, space, __args__):
///     w_obj = force(space, self)
///     return space.call_args(w_obj, __args__)
/// ```
pub fn callable_proxy_descr__call__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // `args` is the gateway's native copy; a collection in `force` does not
    // forward it.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    let w_obj = force(_roots.get(base))?;
    // `space.call_args(w_obj, __args__)` preserves both positional and keyword
    // arguments. Re-split pyre's flat builtin kwargs marker before forwarding
    // so the marker dict cannot leak into the referent as a positional value.
    let mut forwarded = Vec::with_capacity(args.len() - 1);
    for i in 1..args.len() {
        forwarded.push(_roots.get(base + i));
    }
    crate::builtins::call_forwarding_args(w_obj, &forwarded)
}

/// pypy/module/_weakref/interp__weakref.py proxy
///
/// ```python
/// def proxy(space, w_obj, w_callable=None):
///     """Create a proxy object that weakly references 'obj'.
/// 'callback', if given, is called with the proxy as an argument when 'obj'
/// is about to be finalized."""
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_proxy(w_obj)
///     else:
///         return lifeline.make_proxy_with_callback(w_obj, w_callable)
/// ```
pub fn proxy(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "proxy() takes at least 1 argument".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let _roots = pyre_object::gc_roots::push_roots();
    // `proxy` / `descr__new__proxy` pass both words under one call.
    // `RootScope::pin_roots` publishes that pair once.
    let base = pyre_object::gc_roots::pin_roots(&[w_obj, w_callable]);
    let lifeline = getlifeline(pyre_object::gc_roots::shadow_stack_get(base))?;
    let w_obj = pyre_object::gc_roots::shadow_stack_get(base);
    let w_callable = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_proxy(lifeline, w_obj))
    } else {
        Ok(make_proxy_with_callback(lifeline, w_obj, w_callable))
    }
}

/// pypy/module/_weakref/interp__weakref.py descr__new__proxy
///
/// ```python
/// def descr__new__proxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError, "cannot create 'weakproxy' instances")
/// ```
///
/// The name in the message is the type's own, not PyPy's internal typedef
/// name: `type_call` formats `"cannot create '%s' instances"` from `tp_name`
/// when `tp_new` is NULL, and the proxy type is named `weakref.ProxyType`.
pub fn descr__new__proxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakref.ProxyType' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py descr__new__callableproxy
///
/// ```python
/// def descr__new__callableproxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError,
///                 "cannot create 'weakcallableproxy' instances")
/// ```
///
/// Named after the type itself, for the reason given in `descr__new__proxy`.
pub fn descr__new__callableproxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakref.CallableProxyType' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py force
///
/// ```python
/// def force(space, proxy):
///     if not isinstance(proxy, W_AbstractProxy):
///         return proxy
///     w_obj = proxy.dereference()
///     if w_obj is None:
///         raise oefmt(space.w_ReferenceError,
///                     "weakly referenced object no longer exists")
///     return w_obj
/// ```
///
/// `proxy_check_ref` spells the same text hyphenated, and that is the
/// wording every dead-proxy operation reports.
pub fn force(proxy: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if !is_w_abstract_proxy(proxy) {
        return Ok(proxy);
    }
    let w_obj = dereference(proxy);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::reference_error(
            "weakly-referenced object no longer exists".to_string(),
        ));
    }
    Ok(w_obj)
}

pub fn is_w_abstract_proxy(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    match crate::typedef::r#type(obj) {
        Some(tp) => {
            std::ptr::eq(tp.as_ptr(), proxy_type())
                || std::ptr::eq(tp.as_ptr(), callable_proxy_type())
        }
        None => false,
    }
}

// ── proxy_typedef_dict / callable_proxy_typedef_dict ──────────────────
//
// pypy/module/_weakref/interp__weakref.py
//
// ```python
// proxy_typedef_dict = {}
// callable_proxy_typedef_dict = {}
// special_ops = {'repr': True, 'hash': True}
//
// for opname, _, arity, special_methods in ObjSpace.MethodTable:
//     if opname in special_ops or not special_methods:
//         continue
//     ...
//     for i in range(forcing_count):
//         code += "    w_obj%s = force(space, w_obj%s)\n" % (i, i)
//     code += "    return space.%s(%s)" % (opname, nonspaceargs)
//     exec py.code.Source(code).compile()
//     ...
// ```
//
// PyPy generates one wrapper per `(opname, special_methods)` row by
// `exec`'ing a string template. pyre cannot synthesize functions at
// runtime — `BuiltinCodeFn` is a fixed `fn(&[PyObjectRef])` pointer —
// so we mirror the loop statically below, one Rust function per
// generated row. Pyre's space op surface is the Python-3 subset of
// PyPy's MethodTable, so the Py-2-only rows (`getslice`, `coerce`,
// `__hex__`, …) are skipped exactly as Python 3 would.

/// Helper used by every proxy wrapper to call a dunder via getattr +
/// `call_function_impl_result`. Used for ops where pyre has no direct
/// `crate::baseobjspace` entry (e.g. `__pos__`, `__abs__`, `__int__`).
fn forward_to_dunder(
    w_obj: PyObjectRef,
    methname: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    // `args` is a native copy; a collection in `getattr_str` does not
    // forward it.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    let method = crate::baseobjspace::getattr_str(w_obj, methname)?;
    let mut forwarded = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        forwarded.push(_roots.get(base + i));
    }
    crate::call::call_function_impl_result(method, &forwarded)
}

// Forwarding wrappers — `force(args[i])` then dispatch to the named
// `crate::baseobjspace` op. The macro arms cover the common arities and
// keep the per-op definitions a single line each, mirroring PyPy's
// generated source.

macro_rules! proxy_unary {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let w_obj0 = force(args[0])?;
            $space_op(w_obj0)
        }
    };
}

macro_rules! proxy_binary {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let w_self = args[0];
            let mut w_other = args[1];
            let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
            let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
            $space_op(w_obj0, w_obj1)
        }
    };
}

macro_rules! proxy_binary_reflected {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            // interp__weakref.py:382-385 — reflected wrappers swap the
            // operands before calling the space op:
            //   `code = code.replace("(w_obj0, w_obj1)", "(w_obj1, w_obj0)")`
            let w_self = args[0];
            let mut w_other = args[1];
            let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
            let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
            $space_op(w_obj1, w_obj0)
        }
    };
}

// Forward + reflected binary ops — interp__weakref.py:376-389.
proxy_binary!(proxy_add, crate::baseobjspace::add);
proxy_binary_reflected!(proxy_radd, crate::baseobjspace::add);
proxy_binary!(proxy_sub, crate::baseobjspace::sub);
proxy_binary_reflected!(proxy_rsub, crate::baseobjspace::sub);
proxy_binary!(proxy_mul, crate::baseobjspace::mul);
proxy_binary_reflected!(proxy_rmul, crate::baseobjspace::mul);
proxy_binary!(proxy_truediv, crate::baseobjspace::truediv);
proxy_binary_reflected!(proxy_rtruediv, crate::baseobjspace::truediv);
proxy_binary!(proxy_floordiv, crate::baseobjspace::floordiv);
proxy_binary_reflected!(proxy_rfloordiv, crate::baseobjspace::floordiv);
proxy_binary!(proxy_mod, crate::baseobjspace::mod_);
proxy_binary_reflected!(proxy_rmod, crate::baseobjspace::mod_);
// pow / rpow — the `ObjSpace.MethodTable` loop in interp__weakref.py
// generates a 3-arg wrapper from
// `('pow', '**', 3, ['__pow__', '__rpow__'])`, whose
// `forcing_count = len(special_methods) = 2` leaves the optional modulo
// operand unforced. `WRAP_TERNARY` dereferences it whenever it is present,
// so a proxy modulus reaches the slot as its referent and a dead one raises
// `ReferenceError`; that is what `proxy_pow_modulus` below restores. The
// two-arg form hands off to `space.pow`, whose forward/reverse dance lets
// `__rpow__` answer; the three-arg form hands off to `space.pow3`, which
// consults only the forward `__pow__` (`DescrOperation.pow`) and never the
// reflected slot.
/// `wrap_ternaryfunc`'s count check, written here rather than declared at the
/// registration: the modulo operand is optional, so the slice is two or three
/// long and there is no single arity for the gateway to enforce.  Without it
/// the reads below ran off the end of a short slice.
fn proxy_pow_arity(args: &[PyObjectRef]) -> Result<(), PyError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(PyError::type_error(format!(
            "expected 1 or 2 arguments, got {}",
            args.len().saturating_sub(1)
        )));
    }
    Ok(())
}

/// `WRAP_TERNARY`'s `if (w != NULL) { UNWRAP(w); }`.  The dereference comes
/// before the `None` test, so a dead proxy modulus raises `ReferenceError`
/// rather than being read as "no modulus".
fn proxy_pow_modulus(w_mod: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if w_mod.is_null() {
        return Ok(PY_NULL);
    }
    force(w_mod)
}

pub fn proxy_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    proxy_pow_arity(args)?;
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_mod = if args.len() < 3 { PY_NULL } else { args[2] };
    let mut w_obj0 = pyre_object::with_roots!(w_other, w_mod => force(w_self))?;
    let mut w_obj1 = pyre_object::with_roots!(w_obj0, w_mod => force(w_other))?;
    let w_obj2 = pyre_object::with_roots!(w_obj0, w_obj1 => proxy_pow_modulus(w_mod))?;
    let has_modulo = !w_obj2.is_null() && !unsafe { pyre_object::is_none(w_obj2) };
    if has_modulo {
        crate::baseobjspace::pow3(w_obj0, w_obj1, w_obj2)
    } else {
        crate::baseobjspace::pow(w_obj0, w_obj1)
    }
}

pub fn proxy_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    proxy_pow_arity(args)?;
    // interp__weakref.py:382-385 — reflected wrapper swaps the first
    // two operands; the modulo argument keeps its slot.
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_mod = if args.len() < 3 { PY_NULL } else { args[2] };
    let mut w_obj0 = pyre_object::with_roots!(w_other, w_mod => force(w_self))?;
    let mut w_obj1 = pyre_object::with_roots!(w_obj0, w_mod => force(w_other))?;
    let w_obj2 = pyre_object::with_roots!(w_obj0, w_obj1 => proxy_pow_modulus(w_mod))?;
    let has_modulo = !w_obj2.is_null() && !unsafe { pyre_object::is_none(w_obj2) };
    if has_modulo {
        crate::baseobjspace::pow3(w_obj1, w_obj0, w_obj2)
    } else {
        crate::baseobjspace::pow(w_obj1, w_obj0)
    }
}
proxy_binary!(proxy_lshift, crate::baseobjspace::lshift);
proxy_binary_reflected!(proxy_rlshift, crate::baseobjspace::lshift);
proxy_binary!(proxy_rshift, crate::baseobjspace::rshift);
proxy_binary_reflected!(proxy_rrshift, crate::baseobjspace::rshift);
proxy_binary!(proxy_and, crate::baseobjspace::and_);
proxy_binary_reflected!(proxy_rand, crate::baseobjspace::and_);
proxy_binary!(proxy_or, crate::baseobjspace::or_);
proxy_binary_reflected!(proxy_ror, crate::baseobjspace::or_);
proxy_binary!(proxy_xor, crate::baseobjspace::xor);
proxy_binary_reflected!(proxy_rxor, crate::baseobjspace::xor);
proxy_binary!(proxy_matmul, crate::baseobjspace::matmul);
proxy_binary_reflected!(proxy_rmatmul, crate::baseobjspace::matmul);

// Inplace ops — interp__weakref.py:367-369. PyPy forces every operand
// (`forcing_count = arity`) and dispatches to `space.inplace_X`. In pyre the
// equivalent space operation is `opcode_ops::binary_value`: it tries the
// `__i*__` slot first and only then falls back to the ordinary binary op.
macro_rules! proxy_inplace {
    ($name:ident, $op:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let w_self = args[0];
            let mut w_other = args[1];
            let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
            let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
            crate::opcode_ops::binary_value(w_obj0, w_obj1, crate::bytecode::BinaryOperator::$op)
        }
    };
}

proxy_inplace!(proxy_iadd, InplaceAdd);
proxy_inplace!(proxy_isub, InplaceSubtract);
proxy_inplace!(proxy_imul, InplaceMultiply);
proxy_inplace!(proxy_itruediv, InplaceTrueDivide);
proxy_inplace!(proxy_ifloordiv, InplaceFloorDivide);
proxy_inplace!(proxy_imod, InplaceRemainder);
proxy_inplace!(proxy_ipow, InplacePower);
proxy_inplace!(proxy_ilshift, InplaceLshift);
proxy_inplace!(proxy_irshift, InplaceRshift);
proxy_inplace!(proxy_iand, InplaceAnd);
proxy_inplace!(proxy_ior, InplaceOr);
proxy_inplace!(proxy_ixor, InplaceXor);
proxy_inplace!(proxy_imatmul, InplaceMatrixMultiply);

// 1-arg unary ops with a direct pyre space op.
proxy_unary!(proxy_len, crate::baseobjspace::len);
proxy_unary!(proxy_neg, crate::baseobjspace::neg);
proxy_unary!(proxy_invert, crate::baseobjspace::invert);
proxy_unary!(proxy_iter, crate::baseobjspace::iter);

// interp__weakref.py:398-414 — these three operations are explicit rather
// than generated from ObjSpace.MethodTable. In particular, __next__ has a
// proxy-specific error when the live referent is not an iterator.
pub fn proxy_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__bytes__", &[])
}

pub fn proxy_reversed(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__reversed__", &[])
}

pub fn proxy_next(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    if unsafe { crate::baseobjspace::lookup(w_obj0, "__next__") }.is_none() {
        return Err(PyError::type_error(
            "Weakref proxy referenced a non-iterator".to_string(),
        ));
    }
    crate::baseobjspace::next(w_obj0)
}

// 1-arg unary ops without a direct space op — fall through to the
// equivalent dunder so the proxy still delegates to the referent.
pub fn proxy_pos(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__pos__", &[])
}

pub fn proxy_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__abs__", &[])
}

pub fn proxy_int(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__int__", &[])
}

pub fn proxy_float(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__float__", &[])
}

pub fn proxy_index(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__index__", &[])
}

pub fn proxy_trunc(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__trunc__", &[])
}

pub fn proxy_str(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    crate::builtins::builtin_str(&[w_obj0])
}

pub fn proxy_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // interp__weakref.py:360 MethodTable row `('nonzero', ..., ['__nonzero__'])`.
    // pyre uses Python-3 truth dispatch (`is_true`) so the wrapper bridges
    // to that and returns the boxed bool.
    let w_obj0 = force(args[0])?;
    Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true(
        w_obj0,
    )?))
}

// 2-/3-arg attribute ops.  The wrapped name goes to `space.getattr` /
// `setattr` / `delattr` unchanged, so a lone-surrogate name reaches the
// referent's WTF-8 lookup.
pub fn proxy_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_name = args[1];
    let w_obj0 = pyre_object::with_roots!(w_name => force(w_self))?;
    crate::baseobjspace::getattr(w_obj0, w_name)
}

pub fn proxy_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_name = args[1];
    let mut w_value = args[2];
    let w_obj0 = pyre_object::with_roots!(w_name, w_value => force(w_self))?;
    crate::baseobjspace::setattr(w_obj0, w_name, w_value)
}

pub fn proxy_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_name = args[1];
    let w_obj0 = pyre_object::with_roots!(w_name => force(w_self))?;
    crate::baseobjspace::delattr(w_obj0, w_name)
}

// Item ops — interp__weakref.py:365 single special method, so
// `forcing_count = len(special_methods) = 1`. The index/value operands
// pass through unchanged so weakproxy/dead-proxy keys and values reach
// the referent's __getitem__/__setitem__/__delitem__ unmodified.
pub fn proxy_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_key = args[1];
    let w_obj0 = pyre_object::with_roots!(w_key => force(w_self))?;
    crate::baseobjspace::getitem(w_obj0, w_key)
}

pub fn proxy_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // Generated wrapper (interp__weakref.py:374): `return space.setitem(...)`.
    let w_self = args[0];
    let mut w_key = args[1];
    let mut w_value = args[2];
    let w_obj0 = pyre_object::with_roots!(w_key, w_value => force(w_self))?;
    crate::baseobjspace::setitem(w_obj0, w_key, w_value)
}

pub fn proxy_delitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_key = args[1];
    let w_obj0 = pyre_object::with_roots!(w_key => force(w_self))?;
    crate::baseobjspace::delitem(w_obj0, w_key)?;
    Ok(pyre_object::w_none())
}

// __format__(self, format_spec) — single special method, force self only.
pub fn proxy_format(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_spec = args[1];
    let w_obj0 = pyre_object::with_roots!(w_spec => force(w_self))?;
    forward_to_dunder(w_obj0, "__format__", &[w_spec])
}

// __contains__(self, needle) — single special method, force self only.
// pyre's contains returns Result<bool>.
pub fn proxy_contains(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_needle = args[1];
    let w_obj0 = pyre_object::with_roots!(w_needle => force(w_self))?;
    let result = crate::baseobjspace::contains(w_obj0, w_needle)?;
    Ok(pyre_object::w_bool_from(result))
}

// Comparison ops — interp__weakref.py:390-391:
//   elif opname in ["lt", "le", "gt", "ge", "eq", "ne"]:
//       proxy_typedef_dict[special_methods[0]] = interp2app(func)
// Each opname's first special method (e.g. `__lt__` for `lt`) is the
// only one registered, and the wrapper dispatches via the matching
// `CompareOp` variant.
pub fn proxy_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Lt)
}

pub fn proxy_le(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Le)
}

pub fn proxy_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Gt)
}

pub fn proxy_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Ge)
}

pub fn proxy_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Eq)
}

pub fn proxy_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Ne)
}

// pypy/interpreter/baseobjspace.py truncatedint_w isinstance/issubtype rows
// — single special method so `forcing_count = 1`. The wrapper hands
// off to `space.isinstance(forced_self, w_obj1)` /
// `space.issubtype(forced_self, w_obj1)`, which is the descroperation
// dispatch that consults `__instancecheck__` / `__subclasscheck__` on
// `type(forced_self)` and falls back to the recursive type-MRO walk.
pub fn proxy_instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_inst = args[1];
    let w_obj0 = pyre_object::with_roots!(w_inst => force(w_self))?;
    let result = crate::baseobjspace::isinstance(w_inst, w_obj0)?;
    Ok(pyre_object::w_bool_from(result))
}

pub fn proxy_subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_cls = args[1];
    let w_obj0 = pyre_object::with_roots!(w_cls => force(w_self))?;
    let result = crate::baseobjspace::issubclass(w_cls, w_obj0)?;
    Ok(pyre_object::w_bool_from(result))
}

// pypy/interpreter/baseobjspace.py:2159 divmod row — two special
// methods (`__divmod__`, `__rdivmod__`) so `forcing_count = 2`. The
// wrapper calls `space.divmod`, which is the descroperation dispatch
// (numeric fast path then forward + reverse `__divmod__`/`__rdivmod__`
// with NotImplemented fallback).
pub fn proxy_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::divmod(w_obj0, w_obj1)
}

pub fn proxy_rdivmod(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // Reflected: swap operands per interp__weakref.py:382-385.
    let w_self = args[0];
    let mut w_other = args[1];
    let mut w_obj0 = pyre_object::with_roots!(w_other => force(w_self))?;
    let w_obj1 = pyre_object::with_roots!(w_obj0 => force(w_other))?;
    crate::baseobjspace::divmod(w_obj1, w_obj0)
}

// Descriptor protocol — `get`/`set`/`delete` rows in MethodTable each
// have a single special method, so `forcing_count = 1`. Only `self` is
// dereferenced; the descriptor instance and value/type operands pass
// through unchanged so identity-sensitive bindings remain intact.
pub fn proxy_get(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_inst = args[1];
    let mut w_owner = args[2];
    let w_obj0 = pyre_object::with_roots!(w_inst, w_owner => force(w_self))?;
    forward_to_dunder(w_obj0, "__get__", &[w_inst, w_owner])
}

pub fn proxy_set(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_inst = args[1];
    let mut w_value = args[2];
    let w_obj0 = pyre_object::with_roots!(w_inst, w_value => force(w_self))?;
    forward_to_dunder(w_obj0, "__set__", &[w_inst, w_value])
}

pub fn proxy_delete(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let mut w_inst = args[1];
    let w_obj0 = pyre_object::with_roots!(w_inst => force(w_self))?;
    forward_to_dunder(w_obj0, "__delete__", &[w_inst])
}

/// pypy/module/_weakref/interp__weakref.py:356-395 register the entries
/// of `proxy_typedef_dict`, plus the six comparison ops.
///
/// Called from both `init_proxy_type` and `init_callable_proxy_type`, so the
/// two typedefs end up with the same set of methods.  PyPy writes the
/// comparison ops to `proxy_typedef_dict` alone; both `_PyWeakref_ProxyType`
/// and `_PyWeakref_CallableProxyType` install `proxy_richcompare`, and
/// without it `p == referent` on a callable proxy falls through to identity
/// and ordering names the proxy type instead of the referents.
fn register_proxy_typedef_dict(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // Forward + reflected binary arithmetic — interp__weakref.py:376-389.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__add__",
            make_builtin_function_with_arity("__add__", proxy_add, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__radd__",
            make_builtin_function_with_arity("__radd__", proxy_radd, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__sub__",
            make_builtin_function_with_arity("__sub__", proxy_sub, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rsub__",
            make_builtin_function_with_arity("__rsub__", proxy_rsub, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__mul__",
            make_builtin_function_with_arity("__mul__", proxy_mul, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", proxy_rmul, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__truediv__",
            make_builtin_function_with_arity("__truediv__", proxy_truediv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rtruediv__",
            make_builtin_function_with_arity("__rtruediv__", proxy_rtruediv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__floordiv__",
            make_builtin_function_with_arity("__floordiv__", proxy_floordiv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rfloordiv__",
            make_builtin_function_with_arity("__rfloordiv__", proxy_rfloordiv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__mod__",
            make_builtin_function_with_arity("__mod__", proxy_mod, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rmod__",
            make_builtin_function_with_arity("__rmod__", proxy_rmod, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__pow__",
            make_builtin_function("__pow__", proxy_pow)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rpow__",
            make_builtin_function("__rpow__", proxy_rpow)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__lshift__",
            make_builtin_function_with_arity("__lshift__", proxy_lshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rlshift__",
            make_builtin_function_with_arity("__rlshift__", proxy_rlshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rshift__",
            make_builtin_function_with_arity("__rshift__", proxy_rshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rrshift__",
            make_builtin_function_with_arity("__rrshift__", proxy_rrshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__and__",
            make_builtin_function_with_arity("__and__", proxy_and, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rand__",
            make_builtin_function_with_arity("__rand__", proxy_rand, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__or__",
            make_builtin_function_with_arity("__or__", proxy_or, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ror__",
            make_builtin_function_with_arity("__ror__", proxy_ror, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__xor__",
            make_builtin_function_with_arity("__xor__", proxy_xor, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rxor__",
            make_builtin_function_with_arity("__rxor__", proxy_rxor, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__matmul__",
            make_builtin_function_with_arity("__matmul__", proxy_matmul, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rmatmul__",
            make_builtin_function_with_arity("__rmatmul__", proxy_rmatmul, 2)
        )
    };
    // baseobjspace.py:2159 divmod row — forward + reflected.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__divmod__",
            make_builtin_function_with_arity("__divmod__", proxy_divmod, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__rdivmod__",
            make_builtin_function_with_arity("__rdivmod__", proxy_rdivmod, 2)
        )
    };

    // Inplace ops — interp__weakref.py:367-369.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__iadd__",
            make_builtin_function_with_arity("__iadd__", proxy_iadd, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__isub__",
            make_builtin_function_with_arity("__isub__", proxy_isub, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__imul__",
            make_builtin_function_with_arity("__imul__", proxy_imul, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__itruediv__",
            make_builtin_function_with_arity("__itruediv__", proxy_itruediv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ifloordiv__",
            make_builtin_function_with_arity("__ifloordiv__", proxy_ifloordiv, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__imod__",
            make_builtin_function_with_arity("__imod__", proxy_imod, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ipow__",
            make_builtin_function_with_arity("__ipow__", proxy_ipow, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ilshift__",
            make_builtin_function_with_arity("__ilshift__", proxy_ilshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__irshift__",
            make_builtin_function_with_arity("__irshift__", proxy_irshift, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__iand__",
            make_builtin_function_with_arity("__iand__", proxy_iand, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ior__",
            make_builtin_function_with_arity("__ior__", proxy_ior, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ixor__",
            make_builtin_function_with_arity("__ixor__", proxy_ixor, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__imatmul__",
            make_builtin_function_with_arity("__imatmul__", proxy_imatmul, 2)
        )
    };

    // Single-dunder rows — interp__weakref.py:393-395.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__format__",
            make_builtin_function_with_arity("__format__", proxy_format, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__str__",
            make_builtin_function_with_arity("__str__", proxy_str, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__len__",
            make_builtin_function_with_arity("__len__", proxy_len, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getattribute__",
            make_builtin_function_with_arity("__getattribute__", proxy_getattribute, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__setattr__",
            make_builtin_function_with_arity("__setattr__", proxy_setattr, 3)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__delattr__",
            make_builtin_function_with_arity("__delattr__", proxy_delattr, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getitem__",
            make_builtin_function_with_arity("__getitem__", proxy_getitem, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__setitem__",
            make_builtin_function_with_arity("__setitem__", proxy_setitem, 3)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__delitem__",
            make_builtin_function_with_arity("__delitem__", proxy_delitem, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__trunc__",
            make_builtin_function_with_arity("__trunc__", proxy_trunc, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__pos__",
            make_builtin_function_with_arity("__pos__", proxy_pos, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__neg__",
            make_builtin_function_with_arity("__neg__", proxy_neg, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__bool__",
            make_builtin_function_with_arity("__bool__", proxy_bool, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__abs__",
            make_builtin_function_with_arity("__abs__", proxy_abs, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__invert__",
            make_builtin_function_with_arity("__invert__", proxy_invert, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__int__",
            make_builtin_function_with_arity("__int__", proxy_int, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__index__",
            make_builtin_function_with_arity("__index__", proxy_index, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__float__",
            make_builtin_function_with_arity("__float__", proxy_float, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__contains__",
            make_builtin_function_with_arity("__contains__", proxy_contains, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__iter__",
            make_builtin_function_with_arity("__iter__", proxy_iter, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__next__",
            make_builtin_function_with_arity("__next__", proxy_next, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__bytes__",
            make_builtin_function_with_arity("__bytes__", proxy_bytes, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__reversed__",
            make_builtin_function_with_arity("__reversed__", proxy_reversed, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__get__",
            make_builtin_function_with_arity("__get__", proxy_get, 3)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__set__",
            make_builtin_function_with_arity("__set__", proxy_set, 3)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__delete__",
            make_builtin_function_with_arity("__delete__", proxy_delete, 2)
        )
    };
    // baseobjspace.py truncatedint_w isinstance / issubtype rows.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__instancecheck__",
            make_builtin_function_with_arity("__instancecheck__", proxy_instancecheck, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__subclasscheck__",
            make_builtin_function_with_arity("__subclasscheck__", proxy_subclasscheck, 2)
        )
    };

    // `proxy_richcompare` — the same slot on both proxy typedefs.  The
    // `ObjSpace.MethodTable` loop in interp__weakref.py writes these rows to
    // `proxy_typedef_dict` only.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__lt__",
            make_builtin_function_with_arity("__lt__", proxy_lt, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__le__",
            make_builtin_function_with_arity("__le__", proxy_le, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__gt__",
            make_builtin_function_with_arity("__gt__", proxy_gt, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ge__",
            make_builtin_function_with_arity("__ge__", proxy_ge, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__eq__",
            make_builtin_function_with_arity("__eq__", proxy_eq, 2)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__ne__",
            make_builtin_function_with_arity("__ne__", proxy_ne, 2)
        )
    };
}

/// Serialize the proxy-delegation tests. They resolve attributes and methods
/// through the process-global proxy typedefs, whose shared map-node
/// `cache_attrs` RefCell (mapdict `find_branch_to_move_into`) is not `Sync`;
/// under cargo's parallel test threads two proxy tests otherwise borrow it at
/// once and panic. Both this module's and `builtins`'s proxy tests take it.
#[cfg(test)]
pub(crate) static PROXY_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Acquire [`PROXY_TEST_LOCK`].  `parking_lot` does not poison, so one test's
/// panic does not cascade into every follow-up.
#[cfg(test)]
pub(crate) fn lock_proxy_tests() -> parking_lot::MutexGuard<'static, ()> {
    PROXY_TEST_LOCK.lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_weakref_uses_typed_fields_and_hash_cache() {
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_str_new("weakref target");
        let weakref = W_Weakref_new(weakref_type(), referent, PY_NULL);
        assert!(unsafe { pyre_object::weakref::is_typed_weakref(weakref) });
        assert!(!unsafe { pyre_object::w_type_get_hasdict(weakref_type()) });
        assert!(
            crate::typedef::r#type(weakref)
                .is_some_and(|tp| std::ptr::eq(tp.as_ptr(), weakref_type()))
        );
        assert!(std::ptr::eq(dereference(weakref), referent));
        let called = crate::call::call_function_impl_result(weakref, &[]).unwrap();
        assert!(std::ptr::eq(called, referent));
        assert!(unsafe { pyre_object::is_none(descr_callback(&[PY_NULL, weakref]).unwrap()) });

        let first = descr_hash(&[weakref]).unwrap();
        let second = descr_hash(&[weakref]).unwrap();
        assert!(std::ptr::eq(first, second));
        assert!(std::ptr::eq(weakref_hash(weakref), first,));
    }

    /// pypy/module/_weakref/interp__weakref.py force —
    /// dead proxy must raise ReferenceError, not RuntimeError.
    #[test]
    fn test_force_dead_proxy_raises_reference_error() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let proxy = W_Proxy_new(pyre_object::w_none(), PY_NULL);
        // Simulate a dead referent by setting the weak slot back to None.
        write_attr(proxy, ATTR_W_OBJ_WEAK, pyre_object::w_none());
        let err = force(proxy).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ReferenceError);
        assert_eq!(
            err.message_text(),
            "weakly-referenced object no longer exists"
        );
    }

    /// `len(proxy)` must dispatch through `proxy_typedef_dict["__len__"]`,
    /// which forces the receiver and calls `space.len(referent)`.
    #[test]
    fn test_proxy_len_delegates_to_referent() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ]);
        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::len(proxy).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 3);
    }

    /// `proxy + x` must call `proxy_typedef_dict["__add__"]` which forces
    /// every operand and dispatches to `space.add(referent, x)`.
    #[test]
    fn test_proxy_add_delegates_to_referent() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_int_new(40);
        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::add(proxy, pyre_object::w_int_new(2)).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 42);
    }

    /// `proxy.attr` must call the `__getattribute__` wrapper which forces
    /// the receiver and reads the attribute off the referent.
    #[test]
    fn test_proxy_getattr_delegates_to_referent() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        // Use a hasdict instance so the underlying object stores
        // attributes in its mapdict.
        let user_type = crate::typedef::make_builtin_type("ProxyTarget", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(user_type, true) };
        let referent = pyre_object::objectobject::w_instance_new(user_type);
        crate::baseobjspace::setattr_str(referent, "x", pyre_object::w_int_new(7)).unwrap();

        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::getattr_str(proxy, "x").unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 7);
    }

    /// `proxy.attr = value` must delegate to the referent's dict.
    #[test]
    fn test_proxy_setattr_delegates_to_referent() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("ProxyTarget2", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(user_type, true) };
        let referent = pyre_object::objectobject::w_instance_new(user_type);

        let proxy = W_Proxy_new(referent, PY_NULL);
        crate::baseobjspace::setattr_str(proxy, "y", pyre_object::w_int_new(11)).unwrap();

        // Read back via the referent directly to confirm the write
        // landed there, not on the proxy itself.
        let result = crate::baseobjspace::getattr_str(referent, "y").unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 11);
    }

    /// `force()` must be a no-op for any non-proxy operand. Otherwise the
    /// `getattr` / `setattr` fast-path would impose proxy semantics on
    /// every Python value.
    #[test]
    fn test_force_is_noop_for_non_proxy() {
        crate::typedef::init_typeobjects();
        let plain = pyre_object::w_int_new(99);
        let result = force(plain).unwrap();
        assert!(std::ptr::eq(result, plain));
    }

    /// `proxy[index]` must NOT force the index operand. PyPy's
    /// interp__weakref.py sets `forcing_count = len(special_methods)`,
    /// so for `__getitem__` (single special method) only `self` is
    /// dereferenced. A weakproxy passed as the index must reach the
    /// referent's `__getitem__` unchanged so identity-keyed dicts work.
    #[test]
    fn test_proxy_getitem_does_not_force_index() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        // Build a list referent so getitem accepts an int index.
        let referent = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(10),
            pyre_object::w_int_new(20),
            pyre_object::w_int_new(30),
        ]);
        let proxy = W_Proxy_new(referent, PY_NULL);
        // Drive the wrapper directly so we exercise its forcing rule.
        let result = proxy_getitem(&[proxy, pyre_object::w_int_new(1)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 20);
    }

    /// `__pow__` is a 3-arg row in MethodTable, but the slice reaching the
    /// wrapper is two long when no modulus is given.
    #[test]
    fn test_proxy_pow_two_arg_form() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_int_new(2);
        let proxy = W_Proxy_new(referent, PY_NULL);
        // 2-arg form: 2 ** 8 == 256
        let result = proxy_pow(&[proxy, pyre_object::w_int_new(8)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 256);
    }

    /// `WRAP_TERNARY` dereferences the modulo operand, so `pow` sees the
    /// referent and not the proxy: `pow(2, 8, proxy_to_100) == 56`.
    #[test]
    fn test_proxy_pow_forces_modulus() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let modulus = W_Proxy_new(pyre_object::w_int_new(100), PY_NULL);
        let proxy = W_Proxy_new(pyre_object::w_int_new(2), PY_NULL);
        let result = proxy_pow(&[proxy, pyre_object::w_int_new(8), modulus]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 56);
        // Reflected wrapper swaps only the first two operands.
        let modulus = W_Proxy_new(pyre_object::w_int_new(100), PY_NULL);
        let proxy = W_Proxy_new(pyre_object::w_int_new(8), PY_NULL);
        let result = proxy_rpow(&[proxy, pyre_object::w_int_new(2), modulus]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 56);
    }

    /// A dead modulo proxy is dereferenced before the `None` test, so it
    /// raises `ReferenceError` instead of reading as "no modulus".
    #[test]
    fn test_proxy_pow_dead_modulus_raises_reference_error() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let dead = W_Proxy_new(pyre_object::w_none(), PY_NULL);
        write_attr(dead, ATTR_W_OBJ_WEAK, pyre_object::w_none());
        for wrapper in [
            proxy_pow as fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>,
            proxy_rpow,
        ] {
            let proxy = W_Proxy_new(pyre_object::w_int_new(2), PY_NULL);
            let err = wrapper(&[proxy, pyre_object::w_int_new(8), dead]).unwrap_err();
            assert_eq!(err.kind, crate::PyErrorKind::ReferenceError);
            assert_eq!(
                err.message_text(),
                "weakly-referenced object no longer exists"
            );
        }
    }

    /// pypy/interpreter/baseobjspace.py — `__instancecheck__`
    /// and `__subclasscheck__` must be present on both proxy typedefs.
    /// pypy/interpreter/baseobjspace.py:2159 — divmod must register
    /// both `__divmod__` and `__rdivmod__`.
    #[test]
    fn test_proxy_typedef_dict_includes_all_explicit_rows() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let weakproxy = proxy_type();
        let callable = callable_proxy_type();
        unsafe {
            for tp in [weakproxy, callable] {
                assert!(crate::baseobjspace::lookup_in_type(tp, "__instancecheck__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__subclasscheck__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__divmod__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__rdivmod__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__matmul__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__rmatmul__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__imatmul__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__bytes__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__reversed__").is_some());
            }
        }
    }

    /// `isinstance(obj, weakproxy)` must dispatch through
    /// `proxy_typedef_dict["__instancecheck__"]` (the proxy_instancecheck
    /// wrapper), force the proxy, then run the recursive type check on
    /// the referent. Verifies the new `baseobjspace::isinstance` shares
    /// the same entry point as the builtin and the proxy wrapper.
    #[test]
    fn test_isinstance_through_proxy() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0))
            .unwrap()
            .as_ptr();
        let proxy = W_Proxy_new(int_type, PY_NULL);
        // 7 is an int — through the proxy it must still resolve.
        let yes = crate::baseobjspace::isinstance(pyre_object::w_int_new(7), proxy).unwrap();
        assert!(yes);
        // A non-int receiver — must be False.
        let no = crate::baseobjspace::isinstance(pyre_object::w_str_new("hi"), proxy).unwrap();
        assert!(!no);
    }

    /// `issubclass(int, weakproxy_around_int)` parity. Same dispatch
    /// path as `isinstance`, but via `__subclasscheck__`.
    #[test]
    fn test_issubclass_through_proxy() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0))
            .unwrap()
            .as_ptr();
        let str_type = crate::typedef::r#type(pyre_object::w_str_new(""))
            .unwrap()
            .as_ptr();
        let proxy = W_Proxy_new(int_type, PY_NULL);
        let yes = crate::baseobjspace::issubclass(int_type, proxy).unwrap();
        assert!(yes);
        let no = crate::baseobjspace::issubclass(str_type, proxy).unwrap();
        assert!(!no);
    }

    /// `isinstance(obj, Checker())` where `Checker` defines
    /// `__instancecheck__`. Verifies the override path fires for
    /// non-proxy user instances acting as classinfo —
    /// pypy/module/__builtin__/abstractinst.py.
    #[test]
    fn test_isinstance_user_instancecheck_override() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("Checker", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__instancecheck__",
                    crate::make_builtin_function("__instancecheck__", |_args| {
                        Ok(pyre_object::w_bool_from(true))
                    })
                )
            };
        });
        let checker = pyre_object::objectobject::w_instance_new(user_type);
        let yes = crate::baseobjspace::isinstance(pyre_object::w_int_new(123), checker).unwrap();
        assert!(yes);
    }

    /// Same shape as `test_isinstance_user_instancecheck_override` but for
    /// `__subclasscheck__`.
    #[test]
    fn test_issubclass_user_subclasscheck_override() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("ClassChecker", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__subclasscheck__",
                    crate::make_builtin_function("__subclasscheck__", |_args| {
                        Ok(pyre_object::w_bool_from(true))
                    })
                )
            };
        });
        let checker = pyre_object::objectobject::w_instance_new(user_type);
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0))
            .unwrap()
            .as_ptr();
        let yes = crate::baseobjspace::issubclass(int_type, checker).unwrap();
        assert!(yes);
    }

    /// `pow(proxy_around_lhs, rhs)` where `lhs.__pow__` returns
    /// NotImplemented and `rhs.__rpow__` returns the actual answer must
    /// fall through to the reflected operand. The pre-fix proxy_pow
    /// called the forward dunder directly and silently swallowed the
    /// NotImplemented sentinel.
    #[test]
    fn test_proxy_pow_falls_through_to_rpow() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__pow__",
                    crate::make_builtin_function("__pow__", |_args| {
                        Ok(pyre_object::w_not_implemented())
                    })
                )
            };
        });
        let rhs_type = crate::typedef::make_builtin_type("PowRhs", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ns = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__rpow__",
                    crate::make_builtin_function("__rpow__", |_args| {
                        Ok(pyre_object::w_int_new(7777))
                    })
                )
            };
        });
        let lhs = pyre_object::objectobject::w_instance_new(lhs_type);
        let rhs = pyre_object::objectobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        // 2-arg form, no modulus.
        let result = proxy_pow(&[proxy_lhs, rhs]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 7777);
    }

    /// 3-arg `pow(proxy, b, c)` reaches `space.pow3`. Python 3.14 extends
    /// reflected-power dispatch to the ternary form: when the forward
    /// `__pow__` returns `NotImplemented`, the right operand's `__rpow__` is
    /// tried with `(base, modulus)` after its receiver rather than raising.
    #[test]
    fn test_proxy_pow_three_arg_uses_rpow() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("Pow3Lhs", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__pow__",
                    crate::make_builtin_function("__pow__", |_args| {
                        Ok(pyre_object::w_not_implemented())
                    })
                )
            };
        });
        let rhs_type = crate::typedef::make_builtin_type("Pow3Rhs", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ns = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__rpow__",
                    // __rpow__(self, base, modulus) — return the modulus so the
                    // assertion proves the ternary argument reached the slot.
                    crate::make_builtin_function("__rpow__", |args| Ok(args[2]))
                )
            };
        });
        let lhs = pyre_object::objectobject::w_instance_new(lhs_type);
        let rhs = pyre_object::objectobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        let result = proxy_pow(&[proxy_lhs, rhs, pyre_object::w_int_new(7)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 7);
    }

    /// `divmod(proxy, rhs)` where `lhs.__divmod__` returns NotImplemented
    /// must fall through to `rhs.__rdivmod__`. Same regression class as
    /// `proxy_pow`.
    #[test]
    fn test_proxy_divmod_falls_through_to_rdivmod() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("DivmodLhsNI", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__divmod__",
                    crate::make_builtin_function("__divmod__", |_args| {
                        Ok(pyre_object::w_not_implemented())
                    })
                )
            };
        });
        let rhs_type = crate::typedef::make_builtin_type("DivmodRhs", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ns = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__rdivmod__",
                    crate::make_builtin_function("__rdivmod__", |_args| {
                        Ok(pyre_object::w_int_new(123456))
                    })
                )
            };
        });
        let lhs = pyre_object::objectobject::w_instance_new(lhs_type);
        let rhs = pyre_object::objectobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        let result = proxy_divmod(&[proxy_lhs, rhs]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 123456);
    }

    /// `proxy_richcompare` sits on both proxy types, so each typedef — not
    /// the inherited `object` operation — must supply the MRO hit for every
    /// comparison slot.  The `ObjSpace.MethodTable` loop in
    /// interp__weakref.py writes these rows to `proxy_typedef_dict` alone.
    #[test]
    fn test_comparison_ops_on_both_proxy_typedefs() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let weakproxy = proxy_type();
        let callable = callable_proxy_type();
        unsafe {
            for tp in [weakproxy, callable] {
                for name in ["__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"] {
                    let (source, _) = crate::baseobjspace::lookup_where_pair(tp, name).unwrap();
                    assert!(std::ptr::eq(source, tp));
                }
                // Forwarded ops should land on both typedefs.
                assert!(crate::baseobjspace::lookup_in_type(tp, "__add__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__len__").is_some());
            }
        }
    }

    /// `weakref_repr` / `proxy_repr`: the flavour is the typedef's name, and
    /// a live state carries the referent's fully qualified type followed by
    /// its address.
    #[test]
    fn test_repr_names_typedef_flavour_and_referent_address() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_str_new("weakref target");
        let weakref = W_Weakref_new(weakref_type(), referent, PY_NULL);
        let text = unsafe { pyre_object::w_str_get_wtf8(descr__repr__(&[weakref]).unwrap()) };
        let text = text.as_str().unwrap();
        assert!(text.starts_with("<weakref at 0x"), "{text}");
        assert!(text.contains("; to 'str' at 0x"), "{text}");

        let proxy = W_Proxy_new(referent, PY_NULL);
        let text = unsafe { pyre_object::w_str_get_wtf8(descr__repr__(&[proxy]).unwrap()) };
        let text = text.as_str().unwrap();
        assert!(text.starts_with("<weakproxy at 0x"), "{text}");
        assert!(text.contains("; to 'str' at 0x"), "{text}");
    }

    /// Neither proxy type carries a hash: `__hash__` reads as `None`, the
    /// way a NULL `tp_hash` does on `_PyWeakref_ProxyType` /
    /// `_PyWeakref_CallableProxyType`.
    #[test]
    fn test_proxy_hash_is_none() {
        let _g = super::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        unsafe {
            for tp in [proxy_type(), callable_proxy_type()] {
                let w_hash = crate::baseobjspace::lookup_in_type(tp, "__hash__").unwrap();
                assert!(pyre_object::is_none(w_hash));
            }
        }
    }
}
