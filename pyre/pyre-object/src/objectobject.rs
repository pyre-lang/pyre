//! W_ObjectObject — instance of a user-defined class.
//!
//! PyPy equivalent: pypy/objspace/std/objectobject.py → W_ObjectObject
//!
//! A mapdict instance's Python class is `map.terminator.w_cls`
//! (`BaseUserClassMapdict.getclass`). The header `w_class` word is
//! `gettypefor(typeptr)` (`W_Root.getclass`) once a live map is installed.
//! Per-instance attributes live in the mapdict `map`+`storage` pair
//! (`mapdict.py _set_mapdict_map`), matching PyPy's instance attribute layout.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python instance object.
///
/// Layout: `[ob_type | w_class | map | storage]`.
///
/// - `ob_type`: `&INSTANCE_TYPE` for exact `object()`, `&INSTANCE_USER_TYPE`
///   (`typedef.py` `_getusercls(W_ObjectObject)`,
///   `W_ObjectObjectUserDictWeakrefable`) for every other carrier
/// - `w_class`: pointer to the W_TypeObject this is an instance of.
///   A mapdict instance's Python class is `map.terminator.w_cls`
///   (`BaseUserClassMapdict.getclass`).
/// - `map`: the attribute map (`AbstractAttribute` chain) — the
///   `self.map` of `MapdictStorageMixin` (`mapdict.py`)
/// - `storage`: the per-instance attribute storage list — the
///   `self.storage` of `MapdictStorageMixin` (`mapdict.py`)
///
/// A mapdict instance's Python class is `map.terminator.w_cls`
/// (`BaseUserClassMapdict.getclass`). The header word is
/// `gettypefor(typeptr)` (`W_Root.getclass`) once a live map is
/// installed. RPython stores the typeptr on the object (`rclass.py`).
///
/// `map` is an interned `MapNode` address held as a raw word. Node
/// types live in this crate (`mapdict.rs`); intern/rewrite algorithms
/// stay in the interpreter mapdict layer. A null `map`/`storage` is
/// the `_mapdict_init_empty` state with `storage = None` (`mapdict.py`);
/// the real terminator is installed by the mapdict layer on first
/// attribute access.
#[repr(C)]
pub struct W_ObjectObject {
    pub ob_header: PyObject,
    /// `self.map` (`mapdict.py _set_mapdict_map`) — the interned map node's address held as
    /// a raw word, not a pointer.  Map nodes are interned, shared per type and
    /// never freed, so the GC neither owns nor traces this slot
    /// (`object_object_custom_trace` walks only `storage`).  A pointer-typed
    /// field would make the translator lower every read of it to a `Ref`,
    /// which is what the JIT reserves for GC references; the word spelling
    /// keeps the one field on one kind.  Cast to `MapRef` at each use.
    pub map: usize,
    /// `self.storage` (`mapdict.py _mapdict_init_empty`) — a `Ptr(GcArray(OBJECTPTR))` block of
    /// attribute values (`ItemsBlock`, tagged `W_MAPDICT_STORAGE_GC_TYPE_ID`).
    /// null = `None`, the `_mapdict_init_empty` empty state (`mapdict.py`).
    /// The block is a mixed boxed/unboxed array; the mapdict layer
    /// (`pyre-interpreter`) reads/writes it through `crate::object_array`
    /// helpers, and `object_object_custom_trace` walks its boxed slots.
    pub storage: *mut crate::object_array::ItemsBlock,
}

// `storage` is the word immediately after `map` (`MapdictStorageMixin`).
const _: () = {
    assert!(
        std::mem::offset_of!(W_ObjectObject, storage)
            == std::mem::offset_of!(W_ObjectObject, map) + std::mem::size_of::<usize>()
    );
};

/// Fixed payload size of the `[ob_header | map | storage]` instance
/// payload (`framework.py:811`).
pub const W_OBJECT_OBJECT_SIZE: usize = std::mem::size_of::<W_ObjectObject>();

/// `_getusercls(W_ObjectObject)` carrier (`typedef.py`
/// `W_ObjectObjectUserDictWeakrefable`). Layout-identical to
/// [`W_ObjectObject`]; the vtable is a property of this STRUCT
/// (`rclass.new_instance` / `heaptracker.get_vtable_for_gcstruct`).
#[repr(C)]
pub struct W_ObjectObjectUser {
    pub base: W_ObjectObject,
}

const _: () = {
    assert!(std::mem::size_of::<W_ObjectObjectUser>() == std::mem::size_of::<W_ObjectObject>());
    assert!(std::mem::offset_of!(W_ObjectObjectUser, base) == 0);
};

/// Fixed payload size of [`W_ObjectObjectUser`].
pub const W_OBJECT_OBJECT_USER_SIZE: usize = std::mem::size_of::<W_ObjectObjectUser>();

/// GC type id for the `W_ObjectObject` Rust struct. `pyre-jit::eval`
/// registers it through `object_subclass_with_custom_trace` with
/// `W_OBJECT_OBJECT_SIZE` + an `object_object_custom_trace` that
/// traces the off-heap `storage` value slots, so a collection keeps an
/// instance's attribute values reachable and reclaims dead instances.
///
/// This id is the GC header exact `object()` stamps and the
/// `subclass_range` base of `INSTANCE_TYPE`. The collector reads the
/// header for size + custom trace; `ll_issubclass` reads the vtable
/// range. `INSTANCE_TYPE` is this id, a child of `W_ROOT_TYPE`
/// (`W_Root`). Its `_getusercls` child is
/// [`W_OBJECT_OBJECT_USER_GC_TYPE_ID`].
pub const W_OBJECT_OBJECT_GC_TYPE_ID: u32 = 53;

/// GC type id for `W_ObjectObjectUserDictWeakrefable`
/// (`typedef.py` `_getusercls(W_ObjectObject)`). Same size and
/// `object_object_custom_trace` as [`W_OBJECT_OBJECT_GC_TYPE_ID`];
/// that id is the parent.
pub const W_OBJECT_OBJECT_USER_GC_TYPE_ID: u32 = 217;

impl crate::lltype::GcType for W_ObjectObject {
    fn type_id() -> u32 {
        W_OBJECT_OBJECT_GC_TYPE_ID
    }
    const SIZE: usize = W_OBJECT_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_ObjectObjectUser {
    fn type_id() -> u32 {
        W_OBJECT_OBJECT_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_OBJECT_OBJECT_USER_SIZE;
}

/// Allocate a new instance of a user-defined class.
///
/// PyPy equivalent: object.__new__(space, w_type) → allocate_instance
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_instance_new(w_type: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`) for
    // the allocation below. `w_type` is a `W_TypeObject`
    // (`pyre-object::typeobject` GC type id 33) — user-defined types
    // are nursery GC objects, so the pinned typeptr remains a live
    // GC reference across the instance allocation (the collector
    // rewrites the slot if the type moves). The
    // `is_in_nursery` filter in the walker (`majit-gc/src/collector.rs`)
    // keeps the built-in static `PyType` case (e.g. `INT_TYPE`) untouched.
    let _roots = crate::gc_roots::push_roots();
    let w_type = crate::gc_roots::pin_root(w_type);
    let obj = alloc_instance_object(w_type);
    // objspace.py `allocate_instance`: types with `hasuserdel` register the
    // fresh instance on `space.finalizer_queue` immediately after allocation.
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

/// Keep construction at the allocation boundary, so the struct and its field
/// stores stay in one graph for `jtransform.rewrite_op_malloc` to reach.
///
/// Upstream's `instantiate(cls)` is NOT typed by `rtype_malloc` (that one is
/// `@typer_for(lltype.malloc)`); it goes `rtype_instantiate` ->
/// `rclass.rtype_new_instance` -> `RInstance.new_instance`, which emits the
/// `malloc` op itself and then stores each field one at a time
/// (`rclass.py`, `new_instance`: `genop('malloc', ...)` followed by
/// `self.setfield(vptr, '__class__', ...)` and one `setfield` per default).
/// `rewrite_op_malloc` turns that malloc into `new_with_vtable` when
/// `heaptracker.get_vtable_for_gcstruct` answers; `rewrite_op_setfield`
/// then drops the typeptr store unconditionally (`is_typeptr_getset`).
///
/// Exact `object()` and `_getusercls` are two STRUCTs
/// ([`W_ObjectObject`] / [`W_ObjectObjectUser`]), each with a constant
/// vtable, matching the two size descrs already published. The full
/// `typedef.py _getusercls` mixin port (map/storage only on the user
/// class) is still open; these two layout-identical carriers are the
/// malloc/vtable split that makes `new_with_vtable` per branch.
///
/// Each specialized graph allocates through [`crate::lltype::malloc_typed_managed`]
/// (`gct_fv_gc_malloc` / `init_gc_object`). That is `gc_hook::try_gc_alloc`
/// → the installed `GC_ALLOC_HOOK` (`pyre_object_gc_alloc_trampoline` →
/// `majit_gc::alloc_nursery_typed`), the same nursery allocator HEAD's
/// `try_gc_alloc_nursery_raw` used. `fuse_boxing_alloc` (`model.rs`)
/// matches `malloc_typed_managed` in `gc_malloc_flavor` and emits
/// `NewWithVtable` when the type word is a constant; it refuses a moving
/// `New` only for a non-constant type word
/// (`unsupported-malloc-flags-nonmovable`).
fn alloc_instance_object(w_class: PyObjectRef) -> PyObjectRef {
    // `mapdict.py user_setup` → `_mapdict_init_empty(
    // w_subtype.terminator)` (`mapdict.py`): the instance map is
    // the owning type's terminator from construction, and `storage = None`.
    //
    // Reading it before allocation is what makes `_get_mapdict_map`'s
    // `jit.promote(self.map)` promotable (`mapdict.py user_setup` /
    // `_mapdict_init_empty(w_subtype.terminator)`). The type owns that
    // word from `typeobject.py W_TypeObject.__init__`.
    //
    // Zero stays legal: `_getusercls` allocators stamp `map: 0` because
    // `pyre-object` cannot call the interpreter mapdict layer. First
    // attribute access copies `w_subtype.terminator`. The `is_type` test
    // keeps the sentinel used by single-crate tests from being dereferenced
    // as a type.
    // `gct_fv_gc_malloc` bracket (`framework.py`): `w_class` is a live
    // child written after the malloc.  Heap types can sit in the nursery,
    // so the slot is re-read after the allocation and the creation barrier
    // remembers a black instance pointing at a still-white type
    // (`w_bytes_from_block` / `w_cell_new`).
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let w_class = crate::gc_roots::shadow_stack_get(class_slot);
    let map = unsafe {
        if crate::typeobject::is_type(w_class) {
            crate::typeobject::w_type_get_terminator(w_class) as usize
        } else {
            0
        }
    };
    let w_class = crate::gc_roots::shadow_stack_get(class_slot);
    // Exact `object()` keeps `INSTANCE_TYPE`. Anything else — a user class,
    // a builtin that still allocates this attribute bag, a dict subclass, a
    // null sentinel — takes `INSTANCE_USER_TYPE` (`typedef.py`
    // `_getusercls(W_ObjectObject)`). Builtin bags on the user typeptr are
    // interim: `typedef.py` `_getusercls` gives each of those its own
    // interp class.
    if is_exact_object_class(w_class) {
        alloc_exact_object_instance(w_class, map)
    } else {
        alloc_user_object_instance(w_class, map)
    }
}

/// `rclass.new_instance` graph for exact `object()`: one STRUCT, constant
/// [`INSTANCE_TYPE`]. `#[inline(never)]` so `fuse_boxing_alloc` sees
/// `malloc_typed_managed` + constant `ob_type` and emits `new_with_vtable`.
#[inline(never)]
fn alloc_exact_object_instance(w_class: PyObjectRef, map: usize) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let w_class = crate::gc_roots::shadow_stack_get(class_slot);
    let header_class =
        unsafe { layout_header_class(&INSTANCE_TYPE as *const PyType, w_class, map) };
    let header_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(header_class);
    let header_class = crate::gc_roots::shadow_stack_get(header_slot);
    crate::lltype::malloc_typed_managed(W_ObjectObject {
        ob_header: PyObject {
            ob_type: &INSTANCE_TYPE,
            w_class: header_class,
        },
        map,
        storage: std::ptr::null_mut(),
    }) as PyObjectRef
}

/// `rclass.new_instance` graph for `_getusercls(W_ObjectObject)`: one
/// STRUCT, constant [`INSTANCE_USER_TYPE`].
#[inline(never)]
fn alloc_user_object_instance(w_class: PyObjectRef, map: usize) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let w_class = crate::gc_roots::shadow_stack_get(class_slot);
    let header_class =
        unsafe { layout_header_class(&INSTANCE_USER_TYPE as *const PyType, w_class, map) };
    let header_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(header_class);
    let header_class = crate::gc_roots::shadow_stack_get(header_slot);
    crate::lltype::malloc_typed_managed(W_ObjectObjectUser {
        base: W_ObjectObject {
            ob_header: PyObject {
                ob_type: &INSTANCE_USER_TYPE,
                w_class: header_class,
            },
            map,
            storage: std::ptr::null_mut(),
        },
    }) as PyObjectRef
}

/// Typeptr and GC tid [`alloc_instance_object`] stamps for `w_class`.
///
/// Exact `object()` is `w_class` identical to the type object cached on
/// `INSTANCE_TYPE` (`objectobject.py` `allocate_instance` of
/// `W_ObjectObject`). Every other carrier uses `typedef.py`
/// `_getusercls(W_ObjectObject)` (`W_ObjectObjectUserDictWeakrefable`).
///
/// Builtin attribute-bag types that still allocate `W_ObjectObject` land
/// on `INSTANCE_USER_TYPE`. That is interim: `typedef.py` `_getusercls`
/// gives each of those its own interp class.
pub fn instance_typeptr_for(w_class: PyObjectRef) -> (*const PyType, u32) {
    if is_exact_object_class(w_class) {
        (&INSTANCE_TYPE, W_OBJECT_OBJECT_GC_TYPE_ID)
    } else {
        (&INSTANCE_USER_TYPE, W_OBJECT_OBJECT_USER_GC_TYPE_ID)
    }
}

fn is_exact_object_class(w_class: PyObjectRef) -> bool {
    unsafe {
        crate::typeobject::is_type(w_class)
            && std::ptr::eq(w_class, get_instantiate(&INSTANCE_TYPE))
    }
}

/// Python class of `obj` (`W_Root.getclass` / `BaseUserClassMapdict.getclass`).
///
/// A mapdict instance reads `map.terminator.w_cls`; every other object
/// returns the header word (`gettypefor(typeptr)`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_instance_get_type(obj: PyObjectRef) -> PyObjectRef {
    crate::mapdict::instance_getclass(obj)
}

/// Check if an object is an instance of a user-defined class.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_instance(obj: PyObjectRef) -> bool {
    // Exact `object()` is `INSTANCE_TYPE`. Every other carrier is
    // `INSTANCE_USER_TYPE`, a child of that vtable, so the layout test
    // is the subclass range (`rclass.py` `ll_isinstance`).
    ll_isinstance(obj, &INSTANCE_TYPE)
}

/// Exact `_getusercls(W_ObjectObject)` carrier (`INSTANCE_USER_TYPE`).
///
/// `is_instance` is the subclass-range test (`rclass.py` `ll_isinstance`)
/// and includes exact `object()`. This is `ll_type == vtable` against
/// [`INSTANCE_USER_TYPE`]: Python subclasses of `object` share that
/// payload, and exact `object()` (`W_ObjectObject` / `INSTANCE_TYPE`)
/// does not. PyPy's `W_ObjectObject` has no mapdict; the user class is
/// the mapdict carrier (`mapdict.py` `BaseUserClassMapdict.getclass`).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_user_instance(obj: PyObjectRef) -> bool {
    !obj.is_null() && std::ptr::eq((*obj).ob_type, &INSTANCE_USER_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_create_and_check() {
        // A null class is not exact `object()`, so the carrier is the
        // `_getusercls` vtable. `is_instance` reads the published range.
        crate::pyobject::ensure_object_subclass_ranges_initialized();
        let fake_type = PY_NULL;
        let obj = w_instance_new(fake_type);
        unsafe {
            assert!(is_instance(obj));
            assert!(is_user_instance(obj));
            assert!(std::ptr::eq((*obj).ob_type, &INSTANCE_USER_TYPE));
            let hdr = majit_gc::header::header_of(obj as usize);
            assert_eq!((*hdr).type_id(), W_OBJECT_OBJECT_USER_GC_TYPE_ID);
            assert!(!is_int(obj));
            assert!(!crate::typeobject::is_type(obj));
            assert_eq!(w_instance_get_type(obj), fake_type);
        }
    }
}
