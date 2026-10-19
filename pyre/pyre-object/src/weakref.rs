//! Host-side WEAKREF allocator + dereference.
//!
//! Thin shim over `majit_gc::weakref::Weakref` for pyre-object callers
//! (typeobject.weak_subclasses, future W_Weakref / WeakrefLifeline).
//! `pyre-object` cannot depend on `majit-gc`, so this module mirrors
//! the upstream WEAKREF layout (gctypelayout.py
//! `WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))`)
//! and relies on the runtime GC type registration in `pyre-jit::eval`
//! plus the GC's invalidate_*_weakrefs hooks to keep the slot
//! coherent across collections.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// `interp__weakref.py WeakrefLifeline(W_Root)` — the interpreter-owned
/// bookkeeping object attached to every weak-referenceable referent.
///
/// PyPy stores these four values directly on the translated instance.  Keep
/// the same ownership here: none of them are Python attributes and no
/// instance dictionary participates in weakref bookkeeping.  The first three
/// slots currently contain pyre's managed `GcWeakrefBox` / list carriers;
/// they are ordinary GC references and are therefore discovered by the
/// `#[pyre_class]` offset census.
#[pyre_class("_weakref.WeakrefLifeline", static_name = "WEAKREF_LIFELINE")]
pub struct W_WeakrefLifeline {
    /// `interp__weakref.py:22 cached_weakref = None`.
    pub cached_weakref: PyObjectRef,
    /// `interp__weakref.py:23 cached_proxy = None`.
    pub cached_proxy: PyObjectRef,
    /// `interp__weakref.py:24 other_refs_weak = None`.
    pub other_refs_weak: PyObjectRef,
    /// `interp__weakref.py:25 has_callbacks = False`.
    pub has_callbacks: bool,
}

/// Allocate the hidden typed lifeline with PyPy's four class defaults.
/// `allocate_stable` stamps its static translated-layout `ob_type`; like
/// PyPy's `typedef = None` owner, it has no Python-visible heap type.
pub fn w_weakref_lifeline_new() -> PyObjectRef {
    W_WeakrefLifeline::allocate_stable(W_WeakrefLifeline {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        cached_weakref: PY_NULL,
        cached_proxy: PY_NULL,
        other_refs_weak: PY_NULL,
        has_callbacks: false,
    })
}

#[inline]
pub unsafe fn w_weakref_lifeline_cached_weakref(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_WeakrefLifeline)).cached_weakref }
}

#[inline]
pub unsafe fn w_weakref_lifeline_set_cached_weakref(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_WeakrefLifeline)).cached_weakref = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
pub unsafe fn w_weakref_lifeline_cached_proxy(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_WeakrefLifeline)).cached_proxy }
}

#[inline]
pub unsafe fn w_weakref_lifeline_set_cached_proxy(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_WeakrefLifeline)).cached_proxy = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
pub unsafe fn w_weakref_lifeline_other_refs(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_WeakrefLifeline)).other_refs_weak }
}

#[inline]
pub unsafe fn w_weakref_lifeline_set_other_refs(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_WeakrefLifeline)).other_refs_weak = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
pub unsafe fn w_weakref_lifeline_has_callbacks(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_WeakrefLifeline)).has_callbacks }
}

#[inline]
pub unsafe fn w_weakref_lifeline_set_has_callbacks(obj: PyObjectRef) {
    unsafe { (*(obj as *mut W_WeakrefLifeline)).has_callbacks = true };
}

/// `interp__weakref.py W_Weakref(W_WeakrefBase)` — exact builtin
/// weak references own the base weak slot/callback and their cached hash
/// directly. A user subclass is `typedef.py` `_getusercls`: [`W_WeakrefUser`]
/// appends `MapdictStorageMixin` and carries `__dict__` / `__slots__` there.
#[pyre_class(
    "weakref.ReferenceType",
    type_id = 168,
    static_name = "WEAKREF_LAYOUT",
    user_subclass = "WEAKREF_LAYOUT_USER_TYPE"
)]
pub struct W_Weakref {
    /// `W_WeakrefBase.__init__: self.w_obj_weak = weakref.ref(w_obj)`.
    pub w_obj_weak: PyObjectRef,
    /// `W_WeakrefBase.__init__: self.w_callable = w_callable`.
    pub w_callable: PyObjectRef,
    /// `W_Weakref.__init__: self.w_hash = None`.
    pub w_hash: PyObjectRef,
}

/// `typedef.py` `_getusercls(W_Weakref)`: the base payload plus
/// `MapdictStorageMixin`.
#[repr(C)]
pub struct W_WeakrefUser {
    pub base: W_Weakref,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_WeakrefUser, storage)
            == std::mem::offset_of!(W_WeakrefUser, map) + std::mem::size_of::<usize>()
    );
};

/// User-subclass `weakref.ref` typeptr (`typedef.py` `_getusercls`).
/// Instances share `W_Weakref`'s payload and add mapdict `map` / `storage`.
pub static WEAKREF_LAYOUT_USER_TYPE: PyType = new_user_pytype(
    "weakref.ReferenceType",
    &WEAKREF_LAYOUT_TYPE,
    std::mem::offset_of!(W_WeakrefUser, map),
);

/// User-subclass weakref layout (`typedef.py` `_getusercls`). Appended
/// after `W_LongObjectUser`, ahead of the target-gated tail (`build_gc`).
/// Parents on `W_Weakref`.
pub const W_WEAKREF_USER_GC_TYPE_ID: u32 = 200;
pub const W_WEAKREF_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_WeakrefUser>();

impl crate::lltype::GcType for W_WeakrefUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_WEAKREF_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_WEAKREF_USER_OBJECT_SIZE;
}

/// Allocate the exact builtin W_Weakref payload. The caller has already
/// created/rooted the managed weakref carrier and initialized the public type
/// object that supplies `w_class` through `WEAKREF_LAYOUT_TYPE.instantiate`.
pub fn w_weakref_object_new(
    w_obj_weak: PyObjectRef,
    w_callable: PyObjectRef,
    w_hash: PyObjectRef,
) -> PyObjectRef {
    W_Weakref::allocate_stable(W_Weakref {
        ob: PyObject::default(),
        w_obj_weak,
        w_callable,
        w_hash,
    })
}

/// `allocate_instance(W_WeakrefUser, w_class)`: `map` / `storage` at the
/// `MapdictStorageMixin` initial state. The header is non-moving, the same
/// allocator `W_Weakref::allocate_stable` uses.
pub fn w_weakref_user_new(
    w_obj_weak: PyObjectRef,
    w_callable: PyObjectRef,
    w_hash: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_obj_weak, w_callable, w_hash, w_class]);
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_WEAKREF_USER_GC_TYPE_ID,
        W_WEAKREF_USER_OBJECT_SIZE,
    );
    let body = W_WeakrefUser {
        base: W_Weakref {
            ob: PyObject {
                ob_type: &WEAKREF_LAYOUT_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(base + 3),
            },
            w_obj_weak: crate::gc_roots::shadow_stack_get(base),
            w_callable: crate::gc_roots::shadow_stack_get(base + 1),
            w_hash: crate::gc_roots::shadow_stack_get(base + 2),
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    if raw.is_null() {
        return crate::lltype::malloc_typed(body) as PyObjectRef;
    }
    unsafe {
        std::ptr::write(raw as *mut W_WeakrefUser, body);
        crate::gc_hook::try_gc_write_barrier_managed(raw);
    }
    raw as PyObjectRef
}

#[inline]
pub unsafe fn is_typed_weakref(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &WEAKREF_LAYOUT_TYPE) || py_type_check(obj, &WEAKREF_LAYOUT_USER_TYPE)
    }
}

#[inline]
pub unsafe fn w_weakref_object_obj_weak(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Weakref)).w_obj_weak }
}

#[inline]
pub unsafe fn w_weakref_object_set_obj_weak(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_Weakref)).w_obj_weak = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
pub unsafe fn w_weakref_object_callable(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Weakref)).w_callable }
}

#[inline]
pub unsafe fn w_weakref_object_set_callable(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_Weakref)).w_callable = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
pub unsafe fn w_weakref_object_hash(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Weakref)).w_hash }
}

#[inline]
pub unsafe fn w_weakref_object_set_hash(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_Weakref)).w_hash = value };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// GC type id for the WEAKREF GcStruct. Registered by
/// `pyre-jit::eval::init` after `W_INT_MUTABLE_CELL` and before the
/// per-exception kind loop. A `debug_assert_eq!` in the registration
/// site pins the id to this value so callers can treat it as a
/// constant.
pub const WEAKREF_GC_TYPE_ID: u32 = 51;

/// `WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))`
/// (gctypelayout.py). Single `weakptr` slot at offset 0 — the
/// majit-gc collector reads / writes this slot during
/// `invalidate_young_weakrefs` / `invalidate_old_weakrefs`.
#[repr(C)]
pub struct Weakref {
    pub weakptr: PyObjectRef,
}

/// `sizeof_weakref = llmemory.sizeof(WEAKREF)` (gctypelayout.py).
pub const SIZEOF_WEAKREF: usize = std::mem::size_of::<Weakref>();

impl crate::lltype::GcType for Weakref {
    fn type_id() -> u32 {
        WEAKREF_GC_TYPE_ID
    }
    const SIZE: usize = SIZEOF_WEAKREF;
}

/// Allocate a new WEAKREF GcStruct via the active GC and initialise its single
/// `weakptr` slot to `target`. Always returns a non-null, usable weakref:
/// `weakref.ref(...)` never fails in PyPy, and under
/// `translation.rweakref=False` it is implemented as a strong reference
/// (rweakref.py:11-16). When no collecting GC hook is installed yet (pre-build
/// bootstrap, e.g. a module-level `class B(A)` evaluated before the JIT GC is
/// wired) the hook answers `None` (`GcAllocOutcome::NoRoute`) and this falls
/// back to a Box-immortal `Weakref`: a never-collected slot whose `weakptr`
/// stays valid — the rweakref-off strong-ref mode. When a GC owns the heap
/// and cannot satisfy the malloc (`Some(null)` / `GcAllocOutcome::Failed`),
/// abort (`gc_alloc_failed`, `incminimark.py` `collect_and_reserve`): an
/// unmanaged Weakref is invisible to `invalidate_young_weakrefs`.
///
/// # Safety
///
/// Caller must ensure `target` outlives the weakref or accept that a
/// subsequent collection will null the slot (GC-allocated path only; the
/// Box-immortal bootstrap slot is never cleared).
pub unsafe fn w_weakref_new(target: PyObjectRef) -> *mut Weakref {
    // gct_weakref_create: `push_roots(keep_current_args=True)` then
    // malloc_fixedsize(..., contains_weakptr=True). Nursery overflow is
    // collect_and_reserve, so invalidate_young_weakrefs always sees the
    // new WEAKREF. The no-collect allocator is not a fallback: its spill
    // would birth an old weakref that invalidate_young_weakrefs never visits.
    let mut rooted_target = target;
    // `weakptr` is a weak slot, so the store below deliberately runs no
    // creation barrier; the flag exists only to satisfy the allocator's
    // out-parameter and carries the same default as the other call sites.
    let mut needs_write_barrier = true;
    let payload = unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            WEAKREF_GC_TYPE_ID,
            SIZEOF_WEAKREF,
            &mut rooted_target as *mut PyObjectRef as *mut crate::gc_hook::GCREF,
            &mut needs_write_barrier,
        )
    };
    if let Some(payload) =
        crate::gc_hook::GcAllocOutcome::from_hook(payload).allocated_or_abort(SIZEOF_WEAKREF)
    {
        let wref = payload as *mut Weakref;
        unsafe { (*wref).weakptr = rooted_target };
        return wref;
    }

    crate::lltype::malloc_typed(Weakref {
        weakptr: rooted_target,
    })
}

/// `ll_weakref_deref(wref)` (gctypelayout.py). Reads the
/// `weakptr` slot. Returns null when the GC has already invalidated
/// the target during a minor / major cycle (incminimark.py:3068-3079
/// / :3116-3122).
///
/// # Safety
///
/// `wref` must point to a live WEAKREF GcStruct.
pub unsafe fn w_weakref_deref(wref: *const Weakref) -> PyObjectRef {
    if wref.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*wref).weakptr }
}

// ── GcWeakrefBox wrapper ──────────────────────────────────────────────
//
// pyre's `interp__weakref.rs` still simulates PyPy's W_WeakrefBase on top of
// `W_ObjectObject` + ATTR_* instance-dict slots. Instance-dict slots can only
// hold `PyObjectRef`, not a raw `*mut Weakref`, so this tiny internal PyObject
// wraps the rweakref pointer for storage in those slots. `WeakrefLifeline`
// itself now has its orthodox typed, inline layout above.
//
// Completing W_Weakref / W_AbstractProxy as typed W_Root subclasses will let
// their inline fields own raw `*mut Weakref` values directly. Until that
// follow-up, this wrapper is itself GC-managed and carries the inline field:
// the owning typed lifeline or instance dict traces the box, and the box traces
// the Weakref GcStruct.

/// Internal type tag — used by `py_type_check` to recognise a
/// `GcWeakrefBox` PyObject when it surfaces through a generic slot.
pub static GC_WEAKREF_BOX_TYPE: PyType = new_pytype("__GcWeakrefBox");

/// GC type id assigned to `GcWeakrefBox` — slot 52, immediately after
/// `WEAKREF_GC_TYPE_ID=51`.
pub const GC_WEAKREF_BOX_GC_TYPE_ID: u32 = 52;

#[repr(C)]
pub struct GcWeakrefBox {
    pub ob_header: PyObject,
    /// Strong pointer to a `Weakref` GcStruct. The GC traces this slot
    /// (see `GC_WEAKREF_BOX_GC_PTR_OFFSETS`) so the Weakref struct itself
    /// survives across collections; the `weakptr` slot inside the
    /// Weakref is invalidated separately by
    /// `invalidate_young_weakrefs` / `invalidate_old_weakrefs`.
    pub inner: *mut Weakref,
}

pub const GC_WEAKREF_BOX_OBJECT_SIZE: usize = std::mem::size_of::<GcWeakrefBox>();

/// Byte offset of the inline `*mut Weakref` field the GC must trace
/// (as a strong GcRef) during minor / major collection. Mirrors the
/// `W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS` convention in celldict.rs.
pub const GC_WEAKREF_BOX_GC_PTR_OFFSETS: [usize; 1] = [std::mem::offset_of!(GcWeakrefBox, inner)];

impl crate::lltype::GcType for GcWeakrefBox {
    fn type_id() -> u32 {
        GC_WEAKREF_BOX_GC_TYPE_ID
    }
    const SIZE: usize = GC_WEAKREF_BOX_OBJECT_SIZE;
}

/// Allocate a `GcWeakrefBox` wrapping a fresh rweakref to `target`.
/// Returns null when no GC hook is installed (test environments that
/// did not wire `pyre-jit`) or when `target` itself is null.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_gc_weakref_box_new(target: PyObjectRef) -> PyObjectRef {
    if target.is_null() {
        return std::ptr::null_mut();
    }
    let inner = unsafe { w_weakref_new(target) };
    if inner.is_null() {
        return std::ptr::null_mut();
    }
    let mut value = GcWeakrefBox {
        ob_header: PyObject {
            ob_type: &GC_WEAKREF_BOX_TYPE as *const PyType,
            w_class: get_instantiate(&GC_WEAKREF_BOX_TYPE),
        },
        inner,
    };
    let inner_slot = (&mut value.inner as *mut *mut Weakref).cast::<crate::gc_hook::GCREF>();
    let mut needs_write_barrier = true;
    let raw = unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            GC_WEAKREF_BOX_GC_TYPE_ID,
            GC_WEAKREF_BOX_OBJECT_SIZE,
            inner_slot,
            &mut needs_write_barrier,
        )
    };
    if let Some(raw) = raw {
        if raw.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { std::ptr::write(raw as *mut GcWeakrefBox, value) };
        // A nursery-full allocation may spill the box to old-gen.  Its
        // freshly-written strong `inner` field then needs the creation
        // barrier so the next minor collection sees a young Weakref.
        if needs_write_barrier {
            crate::gc_hook::try_gc_write_barrier(raw);
        }
        return raw as PyObjectRef;
    }

    // Bootstrap/test configurations may expose only the ordinary managed
    // allocator. Keep the wrapper in the collector in that case as well:
    // an off-GC box would leave its managed `inner` pointer outside the root
    // graph. If no managed allocation path is available, return null and let
    // `w_gc_weakref_box_new_or_strong` preserve the target directly.
    let Some(raw) = crate::gc_hook::try_gc_alloc_with_placement(
        GC_WEAKREF_BOX_GC_TYPE_ID,
        GC_WEAKREF_BOX_OBJECT_SIZE,
        &mut needs_write_barrier,
    ) else {
        return std::ptr::null_mut();
    };
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { std::ptr::write(raw as *mut GcWeakrefBox, value) };
    if needs_write_barrier {
        crate::gc_hook::try_gc_write_barrier(raw);
    }
    raw as PyObjectRef
}

/// `isinstance(obj, GcWeakrefBox)` predicate.
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
#[inline]
pub unsafe fn is_gc_weakref_box(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { py_type_check(obj, &GC_WEAKREF_BOX_TYPE) }
}

/// Dereference a `GcWeakrefBox` slot. Returns the original target if
/// still alive, or null after the GC invalidated the underlying
/// rweakref. Returns null for null / non-GcWeakrefBox inputs so callers
/// can use the same code path for "uninitialised slot" / "dead
/// referent".
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_box_deref(obj: PyObjectRef) -> PyObjectRef {
    if !unsafe { is_gc_weakref_box(obj) } {
        return std::ptr::null_mut();
    }
    let wref = unsafe { (*(obj as *const GcWeakrefBox)).inner };
    unsafe { w_weakref_deref(wref) }
}

/// Point an existing box at `target` by installing a fresh rweakref.
///
/// `rweaklist.py add_handle` reuses the slot of a handle whose referent
/// died rather than appending a new one; reusing the box with it keeps the
/// box count at the peak number of simultaneously live handles instead of the
/// number of registrations. The previous inner Weakref becomes
/// unreachable and is collected — it is the *new* one that
/// [`w_weakref_new`] registers with the collector, which is what keeps the weak
/// semantics. The box is an ordinary GC-managed object: keep it rooted while
/// allocating the replacement and run the normal old-to-young write barrier
/// after storing the new `inner`.
///
/// Returns false for a null target or a slot that is not a box.
///
/// # Safety
///
/// `obj` must be null or a live `GcWeakrefBox`, and `target` must be rooted by
/// the caller: allocating the replacement rweakref can collect.
pub unsafe fn w_gc_weakref_box_retarget(obj: PyObjectRef, target: PyObjectRef) -> bool {
    if target.is_null() {
        return false;
    }
    // `rweaklist.py add_handle` keeps the handle in its GC-visible
    // `handles` list. Rust locals are not traced, so mirror the translated
    // livevar explicitly: `w_weakref_new` may collect and relocate the box.
    let _roots = crate::gc_roots::push_roots();
    let box_root = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let rooted_obj = crate::gc_roots::shadow_stack_get(box_root);
    if !unsafe { is_gc_weakref_box(rooted_obj) } {
        return false;
    }
    let inner = unsafe { w_weakref_new(target) };
    if inner.is_null() {
        return false;
    }
    let rooted_obj = crate::gc_roots::shadow_stack_get(box_root);
    unsafe { (*(rooted_obj as *mut GcWeakrefBox)).inner = inner };
    crate::gc_hook::try_gc_write_barrier(rooted_obj as crate::gc_hook::GCREF);
    true
}

/// Clear the wrapped rweakref, mirroring
/// `W_WeakrefBase.clear(): self.w_obj_weak = dead_ref`.
///
/// # Safety
///
/// `obj` must either be null or a live `GcWeakrefBox`.
pub unsafe fn w_gc_weakref_box_clear(obj: PyObjectRef) {
    if !unsafe { is_gc_weakref_box(obj) } {
        return;
    }
    let boxed = obj as *mut GcWeakrefBox;
    let wref = unsafe { (*boxed).inner };
    if !wref.is_null() {
        unsafe { (*wref).weakptr = std::ptr::null_mut() };
    }
}

/// Allocate a GcWeakrefBox for `target`, falling back to a strong
/// PyObjectRef when no GC hook is installed (unit-test environments
/// that did not wire `pyre-jit`). The strong-ref fallback restores
/// the historical instance-dict-slot behavior for tests while
/// production paths get real weak semantics.
///
/// Pair with `w_gc_weakref_box_or_strong_deref` on the reader side.
pub fn w_gc_weakref_box_new_or_strong(mut target: PyObjectRef) -> PyObjectRef {
    // The inner weakref may be allocated before the box allocation fails.
    let wrapped = crate::with_roots!(target => w_gc_weakref_box_new(target));
    if wrapped.is_null() { target } else { wrapped }
}

/// Read a slot written by `w_gc_weakref_box_new_or_strong`. When the slot
/// holds a GcWeakrefBox, deref through the GC weakref. Otherwise treat
/// the slot itself as a strong PyObjectRef (the no-GC fallback path).
///
/// # Safety
///
/// `slot` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_box_or_strong_deref(slot: PyObjectRef) -> PyObjectRef {
    if slot.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { is_gc_weakref_box(slot) } {
        return unsafe { w_gc_weakref_box_deref(slot) };
    }
    slot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lltype::PyreClassPyTypeOf;

    #[test]
    fn weakref_lifeline_uses_inline_pypy_fields() {
        let lifeline = w_weakref_lifeline_new();
        assert!(unsafe { py_type_check(lifeline, &WEAKREF_LIFELINE_TYPE) });
        assert!(unsafe { w_weakref_lifeline_cached_weakref(lifeline) }.is_null());
        assert!(unsafe { w_weakref_lifeline_cached_proxy(lifeline) }.is_null());
        assert!(unsafe { w_weakref_lifeline_other_refs(lifeline) }.is_null());
        assert!(!unsafe { w_weakref_lifeline_has_callbacks(lifeline) });

        let cached_ref = 0x1000_usize as PyObjectRef;
        let cached_proxy = 0x2000_usize as PyObjectRef;
        let other_refs = 0x3000_usize as PyObjectRef;
        unsafe {
            w_weakref_lifeline_set_cached_weakref(lifeline, cached_ref);
            w_weakref_lifeline_set_cached_proxy(lifeline, cached_proxy);
            w_weakref_lifeline_set_other_refs(lifeline, other_refs);
            w_weakref_lifeline_set_has_callbacks(lifeline);
        }
        assert_eq!(
            unsafe { w_weakref_lifeline_cached_weakref(lifeline) },
            cached_ref
        );
        assert_eq!(
            unsafe { w_weakref_lifeline_cached_proxy(lifeline) },
            cached_proxy
        );
        assert_eq!(
            unsafe { w_weakref_lifeline_other_refs(lifeline) },
            other_refs
        );
        assert!(unsafe { w_weakref_lifeline_has_callbacks(lifeline) });

        assert_eq!(
            W_WeakrefLifeline::DESCRIPTOR.ptr_offsets,
            &[
                std::mem::offset_of!(W_WeakrefLifeline, ob)
                    + std::mem::offset_of!(PyObject, w_class),
                std::mem::offset_of!(W_WeakrefLifeline, cached_weakref),
                std::mem::offset_of!(W_WeakrefLifeline, cached_proxy),
                std::mem::offset_of!(W_WeakrefLifeline, other_refs_weak),
            ]
        );
    }

    #[test]
    fn exact_weakref_uses_inline_base_and_hash_fields() {
        let obj_weak = 0x4000_usize as PyObjectRef;
        let callable = 0x5000_usize as PyObjectRef;
        let weakref = w_weakref_object_new(obj_weak, callable, PY_NULL);
        assert!(unsafe { is_typed_weakref(weakref) });
        assert_eq!(unsafe { w_weakref_object_obj_weak(weakref) }, obj_weak);
        assert_eq!(unsafe { w_weakref_object_callable(weakref) }, callable);
        assert!(unsafe { w_weakref_object_hash(weakref) }.is_null());

        let hash = 0x6000_usize as PyObjectRef;
        unsafe {
            w_weakref_object_set_hash(weakref, hash);
            w_weakref_object_set_callable(weakref, PY_NULL);
        }
        assert_eq!(unsafe { w_weakref_object_hash(weakref) }, hash);
        assert!(unsafe { w_weakref_object_callable(weakref) }.is_null());

        assert_eq!(
            W_Weakref::DESCRIPTOR.ptr_offsets,
            &[
                std::mem::offset_of!(W_Weakref, ob) + std::mem::offset_of!(PyObject, w_class),
                std::mem::offset_of!(W_Weakref, w_obj_weak),
                std::mem::offset_of!(W_Weakref, w_callable),
                std::mem::offset_of!(W_Weakref, w_hash),
            ]
        );
    }

    /// `typedef.py` `_getusercls(W_Weakref)`: a subclass instance is
    /// `W_WeakrefUser` carrying `WEAKREF_LAYOUT_USER_TYPE`.
    #[test]
    fn weakref_subclass_instance_carries_user_typeptr() {
        assert_eq!(W_WEAKREF_LAYOUT_GC_TYPE_ID, 168);
        assert_eq!(W_WEAKREF_USER_GC_TYPE_ID, 200);
        assert_eq!(
            W_WEAKREF_LAYOUT_OBJECT_SIZE,
            std::mem::offset_of!(W_Weakref, w_hash) + std::mem::size_of::<PyObjectRef>()
        );
        assert_eq!(
            W_WEAKREF_USER_OBJECT_SIZE,
            W_WEAKREF_LAYOUT_OBJECT_SIZE
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<*mut crate::object_array::ItemsBlock>()
        );
        let obj_weak = 0x4000_usize as PyObjectRef;
        let obj = w_weakref_object_new(obj_weak, PY_NULL, PY_NULL);
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &WEAKREF_LAYOUT_TYPE));
            assert!(is_typed_weakref(obj));
        }
        let obj = w_weakref_user_new(
            obj_weak,
            PY_NULL,
            PY_NULL,
            get_instantiate(&WEAKREF_LAYOUT_TYPE),
        );
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &WEAKREF_LAYOUT_USER_TYPE));
            assert!(is_typed_weakref(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &WEAKREF_LAYOUT_TYPE));
            assert_eq!(w_weakref_object_obj_weak(obj), obj_weak);
        }
    }

    #[test]
    fn w_weakref_new_pre_gc_returns_strong_immortal_ref() {
        // No GC hook is wired in a bare pyre-object unit test, so try_gc_alloc
        // returns None (`GcAllocOutcome::NoRoute`). w_weakref_new must still
        // hand back a usable, non-null weakref whose deref yields the target —
        // PyPy's weakref.ref never fails (a strong reference under
        // translation.rweakref=False). `Some(null)` is OOM (`Failed`) and
        // aborts; it is not this fallback. Before the bootstrap fallback this
        // returned null, which a module-level `class B(A)` recorded into the
        // base's weak_subclasses, dropping the subclass from mutated()/
        // get_subclasses().
        let target = 0xdead_beef_usize as PyObjectRef;
        let wref = unsafe { w_weakref_new(target) };
        assert!(!wref.is_null());
        assert_eq!(unsafe { w_weakref_deref(wref) }, target);
    }
}
