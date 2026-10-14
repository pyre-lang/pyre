//! W_ObjectObject — instance of a user-defined class.
//!
//! PyPy equivalent: pypy/objspace/std/objectobject.py → W_ObjectObject
//!
//! An instance holds a pointer to its W_TypeObject (class) in `ob_header.w_class`.
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
/// - `w_class`: pointer to the W_TypeObject this is an instance of
/// - `map`: the attribute map (`AbstractAttribute` chain) — the
///   `self.map` of `MapdictStorageMixin` (`mapdict.py`)
/// - `storage`: the per-instance attribute storage list — the
///   `self.storage` of `MapdictStorageMixin` (`mapdict.py`)
///
/// The Python class is stored in `ob_header.w_class`, shared with all
/// other object types. RPython stores this in `typeptr` (rclass.py).
///
/// `map` is an erased `*const MapNode` (the objspace map node layer in
/// `objspace::std::mapdict` lives in the `pyre-interpreter` crate, which
/// `pyre-object` must not depend on; the interpreter side casts it back).
/// This mirrors the `W_DictObject.dstorage: *mut u8` erasure. A null
/// `map`/`storage` is the `_mapdict_init_empty` state with `storage =
/// None` (`mapdict.py`); the real terminator is installed by the
/// mapdict layer on first attribute access.
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
    // are stable old-gen GC objects, so the pinned typeptr remains a live,
    // non-moving GC reference across the instance allocation. The
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
/// This function is not that shape: it still builds a whole
/// `W_ObjectObject` and `ptr::write`s it into raw bytes obtained from
/// `try_gc_alloc_stable_raw(id, size)`, which carries no `malloc(STRUCT)`
/// operand for `heaptracker.get_vtable_for_gcstruct` to read. Converging
/// means allocating zeroed and storing `ob_type`, `w_class`, `map` and
/// `storage` individually, in `new_instance`'s order.
///
/// Allocate a `W_ObjectObject` through the GC. The header tid is the one
/// [`instance_typeptr_for`] chooses — [`W_OBJECT_OBJECT_GC_TYPE_ID`] for
/// exact `object()`, [`W_OBJECT_OBJECT_USER_GC_TYPE_ID`] otherwise — so
/// `object_object_custom_trace` roots the `storage` value slots and dead
/// instances are reclaimed. When no GC hook is installed (single-crate
/// tests / pre-init snapshot tools) that same header id is stamped on a
/// leaking allocation. `lltype::malloc` still writes id 0, which would
/// disagree with either vtable.
///
/// PRE-EXISTING-ADAPTATION: PyPy instances live in the movable nursery
/// (`rclass`/`gctypelayout` standard `GcStruct`). Pyre allocates them
/// through the stable (non-moving) old-gen allocator instead.
///
/// This note used to record a SIGSEGV as the reason — a movable instance
/// read from `objs[i % 3]` and carried into a method-call guard reading a
/// stale pointer out of the deadframe, reproduced by
/// `synth/inheritance_dispatch` — and named "extend the trace GC-safepoint
/// gcmap to cover transient Ref slots" as the convergence path. That crash
/// no longer reproduces: with this call switched to `try_gc_alloc`,
/// `inheritance_dispatch` passes at the default, 256 KB and 128 KB nursery
/// sizes, the whole `check.py` corpus passes on both backends, and seven
/// GC-sensitive fixtures produce byte-identical output under a 128 KB
/// nursery.
///
/// The switch is not made because it measures nothing. Allocation is about
/// a twentieth of what instantiating an object costs — the rest is call
/// dispatch — and the host-side "nursery" allocator is non-collecting and
/// falls back to old-gen once the nursery is full
/// (`majit-backend-dynasm/src/runner.rs` `dynasm_alloc_nursery_typed`), so
/// the flip only reorders "try nursery, else old-gen" against "always
/// old-gen": minor and major collection counts and peak RSS come out
/// unchanged, and every non-microbenchmark fixture moves within noise.
/// Real convergence needs host-side allocation that may collect, which in
/// turn needs every allocation site to root the raw pointers it holds.
fn alloc_instance_object(w_class: PyObjectRef) -> PyObjectRef {
    // `mapdict.py user_setup` → `_mapdict_init_empty(
    // w_subtype.terminator)` (`mapdict.py`): the instance map is
    // the owning type's terminator from construction, and `storage = None`.
    //
    // Reading it before allocation rather than installing it on first
    // attribute access is what makes `_get_mapdict_map`'s
    // `jit.promote(self.map)` promotable. A deferred install leaves every
    // fresh instance at zero until the first access, so the promoted map guard
    // the JIT bakes cannot hold on the next iteration's fresh instance.
    //
    // Zero stays legal: `pyre-object` cannot build a terminator (it lives in
    // the interpreter's mapdict layer), so a type whose terminator has not
    // been created yet gets one on first access. The `is_type` test keeps the
    // sentinel used by single-crate tests from being dereferenced as a type.
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
    // Exact `object()` keeps `INSTANCE_TYPE`. Anything else — a user class,
    // a builtin that still allocates this attribute bag, a dict subclass, a
    // null sentinel — takes `INSTANCE_USER_TYPE` (`typedef.py`
    // `_getusercls(W_ObjectObject)`). Builtin bags on the user typeptr are
    // interim: `typedef.py` `_getusercls` gives each of those its own
    // interp class.
    let (typeptr, type_id) = instance_typeptr_for(w_class);
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(type_id, W_OBJECT_OBJECT_SIZE);
    let w_class = crate::gc_roots::shadow_stack_get(class_slot);
    let value = W_ObjectObject {
        ob_header: PyObject {
            ob_type: typeptr,
            w_class,
        },
        map,
        storage: std::ptr::null_mut(),
    };
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_ObjectObject, value);
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        raw as PyObjectRef
    } else {
        alloc_instance_object_fallback(value, type_id)
    }
}

/// Host fallback when the GC nursery hook is not installed (lib tests,
/// snapshot tools).  `alloc_with_gc_header` is not in CallRegistry, so
/// the rtyper must not walk this arm from `descr__new__`.
#[majit_macros::dont_look_inside]
fn alloc_instance_object_fallback(value: W_ObjectObject, type_id: u32) -> PyObjectRef {
    majit_gc::header::alloc_with_gc_header(value, type_id) as PyObjectRef
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
    let exact = unsafe {
        crate::typeobject::is_type(w_class)
            && std::ptr::eq(w_class, get_instantiate(&INSTANCE_TYPE))
    };
    if exact {
        (&INSTANCE_TYPE, W_OBJECT_OBJECT_GC_TYPE_ID)
    } else {
        (&INSTANCE_USER_TYPE, W_OBJECT_OBJECT_USER_GC_TYPE_ID)
    }
}

/// Get the class (W_TypeObject) of an instance.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_instance_get_type(obj: PyObjectRef) -> PyObjectRef {
    (*obj).w_class
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
            assert!(std::ptr::eq((*obj).ob_type, &INSTANCE_USER_TYPE));
            let hdr = majit_gc::header::header_of(obj as usize);
            assert_eq!((*hdr).type_id(), W_OBJECT_OBJECT_USER_GC_TYPE_ID);
            assert!(!is_int(obj));
            assert!(!crate::typeobject::is_type(obj));
            assert_eq!(w_instance_get_type(obj), fake_type);
        }
    }
}
