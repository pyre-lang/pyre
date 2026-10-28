//! `pypy/module/__builtin__/descriptor.py` descriptor object ports.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// W_Super — Python `super` proxy object.
//
// PyPy equivalent: pypy/module/__builtin__/descriptor.py W_Super
//
// Stores (super_type, obj_type, obj) and resolves attribute lookups
// starting from the next class after super_type in obj's MRO.

/// super proxy: [ob_type | super_type (cls) | obj_type | obj (self)]
#[pyre_class("super", type_id = 18, static_name = "SUPER", user_layout)]
pub struct W_Super {
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// PyPy `W_Super.w_objtype` — the type returned by `_super_check`.
    pub obj_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Create a new super proxy.
///
/// `objspace.py` `allocate_instance`: `w_subtype` null (or the builtin
/// `super` type) is the base layout; a user subclass is `W_SuperUser`.
/// The three payload refs and `w_subtype` stay pinned across that malloc.
///
/// Exact `W_Super` is `instantiate` / `malloc_fixedsize` (collecting nursery).
/// `_init_subclass` builds `super(w_type, w_type)` as a local; an
/// old-gen proxy joins `old_objects_pointing_to_young` with
/// `super_type` / `obj` the heap type (`incminimark.py`
/// `collect_oldrefs_to_nursery`) and keeps the type alive across
/// every minor. Collecting nursery matches that birth.
pub fn w_super_new(
    super_type: PyObjectRef,
    obj_type: PyObjectRef,
    obj: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[super_type, obj_type, obj, w_subtype]);
    let w_subtype = crate::gc_roots::shadow_stack_get(base + 3);
    if !w_subtype.is_null()
        && !std::ptr::eq(w_subtype, crate::pyobject::get_instantiate(&SUPER_TYPE))
    {
        return W_Super::allocate_instance(
            W_Super {
                ob: PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                super_type: crate::gc_roots::shadow_stack_get(base),
                obj_type: crate::gc_roots::shadow_stack_get(base + 1),
                obj: crate::gc_roots::shadow_stack_get(base + 2),
            },
            w_subtype,
        );
    }
    let mut allocation_root: crate::gc_hook::GCREF = std::ptr::null_mut();
    let mut needs_write_barrier = true;
    let raw = crate::gc_hook::GcAllocOutcome::from_hook(unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            W_SUPER_GC_TYPE_ID,
            W_SUPER_OBJECT_SIZE,
            &mut allocation_root,
            &mut needs_write_barrier,
        )
    })
    .allocated_or_abort(W_SUPER_OBJECT_SIZE)
    .unwrap_or(std::ptr::null_mut());
    let value = W_Super {
        ob: PyObject {
            ob_type: &SUPER_TYPE as *const PyType,
            w_class: crate::pyobject::get_instantiate(&SUPER_TYPE),
        },
        super_type: crate::gc_roots::shadow_stack_get(base),
        obj_type: crate::gc_roots::shadow_stack_get(base + 1),
        obj: crate::gc_roots::shadow_stack_get(base + 2),
    };
    if raw.is_null() {
        let obj = crate::lltype::malloc_typed(value) as PyObjectRef;
        crate::gc_hook::maybe_register_finalizer(obj);
        return obj;
    }
    unsafe {
        std::ptr::write(raw as *mut W_Super, value);
        if needs_write_barrier {
            crate::gc_hook::try_gc_write_barrier_managed(raw);
        }
    }
    let obj = raw as PyObjectRef;
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_super(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SUPER_TYPE) || py_type_check(obj, &SUPER_USER_TYPE) }
}

/// Get the super_type (cls) from a super proxy.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_super_get_type(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).super_type }
}

/// Get the type selected by `_super_check`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_super_get_obj_type(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).obj_type }
}

/// Get the bound object (self) from a super proxy.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).obj }
}

/// Replace the three field-resident pieces of an existing `W_Super`, matching
/// `W_Super.descr_init` assigning `w_starttype`, `w_objtype`, and `w_self`.
///
/// # Safety
/// `obj` must point to a valid `W_Super`.
#[inline]
pub unsafe fn w_super_set_fields(
    obj: PyObjectRef,
    super_type: PyObjectRef,
    obj_type: PyObjectRef,
    bound_obj: PyObjectRef,
) {
    unsafe {
        // `super().__init__(...)` re-initialises a proxy that may already
        // have been promoted, so grey it before the stores the way
        // `w_super_new` does for the freshly allocated one.
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
        let super_obj = obj as *mut W_Super;
        (*super_obj).super_type = super_type;
        (*super_obj).obj_type = obj_type;
        (*super_obj).obj = bound_obj;
    }
}

#[cfg(test)]
mod super_tests {
    use super::*;

    #[test]
    fn w_super_gc_type_id_matches_descr() {
        assert_eq!(W_SUPER_GC_TYPE_ID, 18);
        assert_eq!(
            <W_Super as crate::lltype::GcType>::type_id(),
            W_SUPER_GC_TYPE_ID
        );
        assert_eq!(
            <W_Super as crate::lltype::GcType>::SIZE,
            W_SUPER_OBJECT_SIZE
        );
    }
}

// ── W_Property ─────────────────────────────────────────────────────

/// Python property descriptor object.
///
/// Layout: `[ob_type | fget | fset | fdel | w_doc | w_name | getter_doc |
/// mutate_fget | mutate_fset | mutate_fdel]`
#[pyre_class("property", type_id = 19, static_name = "PROPERTY", user_layout)]
#[majit_macros::jit_immutable_fields("fget?", "fset?", "fdel?")]
pub struct W_Property {
    /// `descriptor.py:175 _immutable_fields_ = ["w_fget?", "w_fset?",
    /// "w_fdel?"]` declares all three quasi-immutable; the hidden watcher
    /// fields below implement the `?` for all three.
    pub fget: PyObjectRef,
    pub fset: PyObjectRef,
    pub fdel: PyObjectRef,
    /// `descriptor.py self.w_doc = space.w_None` — the instance
    /// `__doc__` exposed through `GetSetProperty(get_doc, set_doc)`
    /// (descriptor.py:316-318).  NULL plays None.
    pub w_doc: PyObjectRef,
    /// `descriptor.py self.w_name = None` — set by `__set_name__`
    /// (descriptor.py:274-276) when the property is assigned as a class
    /// attribute.  Surfaced through `__name__` and woven into the
    /// `_properror` accessor messages.  NULL plays unset.
    pub w_name: PyObjectRef,
    /// `descriptor.py:182 self.getter_doc = False` — True when the doc
    /// was copied from `fget.__doc__` (descriptor.py); `_copy`
    /// uses it to drop the inherited doc when the getter is replaced.
    pub getter_doc: bool,
    /// Hidden `mutate_fget` (`quasiimmut.py get_mutate_field_name` of pyre
    /// `fget`; `descriptor.py W_Property._immutable_fields_ = ["w_fget?",
    /// "w_fset?", "w_fdel?"]`).
    ///
    /// Holds no GC pointers, so the derived `PTR_OFFSETS` has nothing to walk
    /// here.  The allocation is [`crate::gc_hook::try_gc_alloc_stable_raw`],
    /// i.e. non-moving, which is [`crate::quasiimmut::QuasiImmutField`]'s
    /// stated precondition: the lock cannot be remapped out from under a
    /// holder.  A GC object's `Drop` never runs, so the field cannot reclaim
    /// its own instance; `property_destructor` in `pyre-jit/src/eval.rs` takes
    /// it back on sweep instead.  `W_TypeObject` and `Function`'s inline
    /// `mutate_<name>` fields carry the same pattern.
    pub mutate_fget: crate::quasiimmut::QuasiImmutField,
    /// Hidden `mutate_fset` (`quasiimmut.py get_mutate_field_name` of pyre
    /// `fset`; upstream `w_fset?`).
    pub mutate_fset: crate::quasiimmut::QuasiImmutField,
    /// Hidden `mutate_fdel` (`quasiimmut.py get_mutate_field_name` of pyre
    /// `fdel`; upstream `w_fdel?`).
    pub mutate_fdel: crate::quasiimmut::QuasiImmutField,
}

/// Allocate a new property object.
///
/// PyPy: W_Property.__init__(space, w_fget, w_fset, w_fdel, w_doc)
pub fn w_property_new(
    fget: PyObjectRef,
    fset: PyObjectRef,
    fdel: PyObjectRef,
    w_subtype: PyObjectRef,
) -> PyObjectRef {
    // Pin the accessors and `w_subtype` across `allocate_instance`
    // (`objspace.py`). A property whose `fget`/`fset`/`fdel` is reachable
    // only through it must be GC-traced. The `w_doc`/`w_name` setters
    // already carry the write barrier (`set_doc`/`set_name`).
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[fget, fset, fdel, w_subtype]);
    W_Property::allocate_instance(
        W_Property {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            fget: crate::gc_roots::shadow_stack_get(base),
            fset: crate::gc_roots::shadow_stack_get(base + 1),
            fdel: crate::gc_roots::shadow_stack_get(base + 2),
            w_doc: PY_NULL,
            w_name: PY_NULL,
            getter_doc: false,
            mutate_fget: crate::quasiimmut::QuasiImmutField::new(),
            mutate_fset: crate::quasiimmut::QuasiImmutField::new(),
            mutate_fdel: crate::quasiimmut::QuasiImmutField::new(),
        },
        crate::gc_roots::shadow_stack_get(base + 3),
    )
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_get_fget(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).fget
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_get_fset(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).fset
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).fdel
}

/// Re-run `property.__init__` on an existing allocation.
///
/// PyPy `descriptor.py:W_Property.init` assigns the three accessors and
/// resets `w_doc` / `getter_doc` before deriving a docstring from the new
/// getter.  CPython 3.14 additionally clears `prop_name` on every init.  Keep
/// those object-resident fields together here instead of maintaining an
/// interpreter-side shadow table.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_reinit(
    obj: PyObjectRef,
    fget: PyObjectRef,
    fset: PyObjectRef,
    fdel: PyObjectRef,
) {
    let prop = obj as *mut W_Property;
    // `rclass.py hook_setfield` emits `jit_force_quasi_immutable`
    // ahead of every store to a `?` field, so the accessors this replaces stop
    // being trace constants before they stop being the live values.  The hook
    // precedes the store and does not consult it, so re-initialising a slot
    // with the value it already holds invalidates as well.  Nothing else
    // revokes them: re-initialising an installed descriptor changes no type's
    // version tag, which is the only other pin a fold over `obj.name` holds.
    // The `is_installed` test is `pyjitpl.py`'s `mutatebox.nonnull()` — a
    // property no loop watches pays one load.
    if (*prop).mutate_fget.is_installed() {
        crate::quasiimmut::sweep_quasi_immut_field(&(*prop).mutate_fget);
    }
    if (*prop).mutate_fset.is_installed() {
        crate::quasiimmut::sweep_quasi_immut_field(&(*prop).mutate_fset);
    }
    if (*prop).mutate_fdel.is_installed() {
        crate::quasiimmut::sweep_quasi_immut_field(&(*prop).mutate_fdel);
    }
    (*prop).fget = fget;
    (*prop).fset = fset;
    (*prop).fdel = fdel;
    (*prop).w_doc = PY_NULL;
    (*prop).w_name = PY_NULL;
    (*prop).getter_doc = false;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// `quasiimmut.py get_current_qmut_instance` for
/// `descriptor.py`'s `w_fget?` — resolved at RECORD time so a write
/// reached later in the same trace sees it, and handed back so the loop
/// compiled from that trace registers on it and `property.__init__` revokes
/// it.  The
/// [`w_type_current_qmut_instance`](crate::typeobject::w_type_current_qmut_instance)
/// shape.
///
/// # Safety
/// `obj` must point to a live [`W_Property`].
pub unsafe fn w_property_current_fget_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(
        (*(obj as *const W_Property))
            .mutate_fget
            .get_current_qmut_instance(),
    )
}

/// The `w_fset?` twin of [`w_property_current_fget_qmut`].
///
/// # Safety
/// `obj` must point to a live [`W_Property`].
pub unsafe fn w_property_current_fset_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(
        (*(obj as *const W_Property))
            .mutate_fset
            .get_current_qmut_instance(),
    )
}

/// The `w_fdel?` twin of [`w_property_current_fget_qmut`].
///
/// # Safety
/// `obj` must point to a live [`W_Property`].
pub unsafe fn w_property_current_fdel_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(
        (*(obj as *const W_Property))
            .mutate_fdel
            .get_current_qmut_instance(),
    )
}

/// `descriptor.py W_Property.get_doc` — returns the raw slot
/// (NULL plays None; the caller wraps).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_get_doc(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).w_doc
}

/// `descriptor.py W_Property.set_doc`, with the Python 3.14
/// `property_set_doc` rule taking precedence: replacing the visible member
/// does not change whether the constructor originally copied it from the
/// getter.  `_copy` still needs that provenance so a later `.getter()` can
/// derive the new getter's docstring.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_set_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_doc = w_doc;
    // Record the old→young edge: `w_doc` is a traced slot and the
    // property may already have been promoted out of the nursery.
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// `descriptor.py:199-204` — stamp a doc inherited from `fget.__doc__`
/// at construction time, marking `getter_doc`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_set_getter_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_doc = w_doc;
    (*prop).getter_doc = true;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// Mark that a property subclass obtained its visible `__doc__` from the
/// getter.  CPython 3.14 stores that visible value in the subclass instance
/// dict (because the subclass class dict shadows property's member), while
/// `_copy` still consults the flag on the native property payload.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_mark_getter_doc(obj: PyObjectRef) {
    (*(obj as *mut W_Property)).getter_doc = true;
}

/// `self.w_name` — NULL plays unset.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_get_name(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).w_name
}

/// `descriptor.py W_Property.set_name` — record the name the
/// property was assigned under.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_property_set_name(obj: PyObjectRef, w_name: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_name = w_name;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_property(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PROPERTY_TYPE) || py_type_check(obj, &PROPERTY_USER_TYPE) }
}

/// `type(obj) is property`, as opposed to [`is_property`]'s layout test.
///
/// `descroperation.py get_and_call_function` spells out why the
/// difference decides who may take an accessor shortcut: `typ = type(w_descr)`
/// then `if typ is Function or typ is FunctionWithFixedCode`, with
/// "isinstance(typ, Function) would not be correct here".  Everything else
/// reaches its accessor through `space.get`, i.e. `type(w_descr).__get__` off
/// the MRO — so calling `fget` in place of `__get__` is licensed only when the
/// descriptor's type is `property` itself and cannot have overridden it.
///
/// Interp-level exactness is the RPython class: `type(w_obj) is
/// W_PropertyObject`. A `_getusercls` subclass instance carries
/// `PROPERTY_USER_TYPE` (`typedef.py`). The Python class is
/// `map.terminator.w_cls` (`BaseUserClassMapdict.getclass`); after
/// `layout_header_class` the header word is `gettypefor(typeptr)` and
/// is not this predicate. Same `user_subclass` test as
/// [`typeptr_is_exact_builtin`].
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_exact_property(obj: PyObjectRef) -> bool {
    unsafe { std::ptr::eq((*obj).ob_type, &PROPERTY_TYPE) }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_property_create() {
        let obj = w_property_new(PY_NULL, PY_NULL, PY_NULL, PY_NULL);
        unsafe {
            assert!(is_property(obj));
            assert!(!is_int(obj));
        }
    }

    #[test]
    fn property_reinit_replaces_accessors_and_clears_metadata() {
        let obj = w_property_new(PY_NULL, PY_NULL, PY_NULL, PY_NULL);
        let old_doc = crate::w_int_new(10);
        let old_name = crate::w_int_new(11);
        let fget = crate::w_int_new(1);
        let fset = crate::w_int_new(2);
        let fdel = crate::w_int_new(3);
        unsafe {
            w_property_set_getter_doc(obj, old_doc);
            w_property_set_name(obj, old_name);
            w_property_reinit(obj, fget, fset, fdel);
            assert_eq!(w_property_get_fget(obj), fget);
            assert_eq!(w_property_get_fset(obj), fset);
            assert_eq!(w_property_get_fdel(obj), fdel);
            assert!(w_property_get_doc(obj).is_null());
            assert!(w_property_get_name(obj).is_null());
            assert!(!(*(obj as *const W_Property)).getter_doc);
        }
    }

    #[test]
    fn w_property_gc_type_id_matches_descr() {
        assert_eq!(W_PROPERTY_GC_TYPE_ID, 19);
        assert_eq!(
            <W_Property as crate::lltype::GcType>::type_id(),
            W_PROPERTY_GC_TYPE_ID
        );
        assert_eq!(
            <W_Property as crate::lltype::GcType>::SIZE,
            W_PROPERTY_OBJECT_SIZE
        );
    }
}
