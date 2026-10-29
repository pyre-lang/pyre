//! Virtual references: lazy materialization of virtualized objects.
//!
//! When JIT-compiled code virtualizes an object (e.g., an interpreter frame),
//! external code may still hold a reference to it. A virtual reference is a
//! lightweight wrapper that defers materialization until someone actually
//! accesses (forces) it.
//!
//! The `JitVirtualRef` struct has two fields:
//! - `virtual_token`: while the JIT is running, this is the force_token
//!   (i.e., the JIT frame address). Set to TOKEN_NONE when forced or
//!   when the frame is no longer active.
//! - `forced`: once forced, this points to the materialized object.
//!
//! Mirrors `rpython/jit/metainterp/virtualref.py`.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use majit_gc::header::GCREF;

/// The value [`VREF_GC_TYPE_ID`] holds before `set_vref_gc_type_id` runs.  Zero
/// is a legitimate id, so the sentinel has to be a value the registry never
/// hands out.
const VREF_GC_TYPE_ID_UNSET: u32 = u32::MAX;

/// GC type id for JitVirtualRef, set by `set_vref_gc_type_id()` at startup.
/// RPython registers JIT_VIRTUAL_REF as a real GC type; pyre does the same
/// via `gc.register_type(TypeInfo::with_gc_ptrs(...))` in eval.rs.
static VREF_GC_TYPE_ID: AtomicU32 = AtomicU32::new(VREF_GC_TYPE_ID_UNSET);

/// Set the GC type id for JitVirtualRef. Called once at startup after
/// `gc.register_type()` returns the assigned id.
pub fn set_vref_gc_type_id(type_id: u32) {
    VREF_GC_TYPE_ID.store(type_id, Ordering::Relaxed);
}

/// Get the registered GC type id for JitVirtualRef.
pub(crate) fn vref_gc_type_id() -> u32 {
    VREF_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `rpython/rtyper/rclass.py:OBJECT` — RPython's GC
/// object header.  Every `rclass.OBJECT` subclass starts with a
/// `typeptr` field (a vtable pointer) used for runtime type identity
/// checks (`inst.typeptr == some_vtable`).  Pyre's analogue is the
/// pointer-sized identity word at offset 0.
#[repr(C)]
pub struct ObjectHeader {
    /// `rclass.OBJECT.typeptr` — runtime type identity.  RPython
    /// stores a pointer to the per-class `OBJECT_VTABLE` instance;
    /// pyre stores a pointer-sized identity constant directly.
    pub typeptr: usize,
}

/// `rpython/rlib/jit.py JitVirtualRef`: heap-allocated virtual
/// reference.  Direct port of `virtualref.py:17-20`:
/// ```python
/// self.JIT_VIRTUAL_REF = lltype.GcStruct('JitVirtualRef',
///     ('super', rclass.OBJECT),
///     ('virtual_token', llmemory.GCREF),
///     ('forced', rclass.OBJECTPTR))
/// ```
/// The `super: ObjectHeader` field is the Rust analogue of upstream's
/// `('super', rclass.OBJECT)` inheritance link — every JitVirtualRef
/// starts with the typeptr slot at offset 0, matching the runtime
/// layout `is_virtual_ref` reads through.
///
/// `virtual_token` is `llmemory.GCREF`. `forced` is upstream
/// `rclass.OBJECTPTR`, held as the `GCREF` view `cast_opaque_ptr` gives.
/// Both are GC references (`history.py` `getkind` `'ref'`).
///
/// Both fields are traced GC slots. `virtual_token` contains null, the
/// prebuilt `JITFRAME_DUMMY`, or a JITFRAME GCREF, matching
/// `virtualref.py` and `virtualizable.py:326-330`.  allow-line-citation
#[repr(C)]
pub struct JitVirtualRef {
    /// `('super', rclass.OBJECT)` — typeptr slot at offset 0.
    pub super_: ObjectHeader,
    pub virtual_token: GCREF,
    /// Upstream `rclass.OBJECTPTR`; held as the `GCREF` view `cast_opaque_ptr` gives.
    pub forced: GCREF,
}

/// `virtualref.py:21-23` `jit_virtual_ref_vtable = lltype.malloc(
/// rclass.OBJECT_VTABLE, ..., immortal=True)` — the per-class vtable
/// instance that `is_virtual_ref` compares `inst.typeptr` against.
/// Pyre stores the type-id as a u64 magic constant rather than a
/// real OBJECT_VTABLE pointer; the comparison `header.typeptr ==
/// JIT_VIRTUAL_REF_VTABLE` is the structural equivalent of
/// upstream's `inst.typeptr == self.jit_virtual_ref_vtable`.
// `virtualref.py` allocates an OBJECT_VTABLE pointer, so its translated
// identity occupies one target word rather than an unconditional u64.
pub const JIT_VIRTUAL_REF_VTABLE: usize = 0x4A56_5221; // "JVR!"

/// Free-function form of [`VirtualRefInfo::is_virtual_ref`] for callers that
/// don't hold a `VirtualRefInfo` — e.g. the interpreter's frame-chain force
/// path (`ExecutionContext::force_vref`).  The check reads only the module
/// constant `JIT_VIRTUAL_REF_VTABLE`, so no receiver state is needed.
/// `virtualref.py is_virtual_ref(gcref)`.
///
/// # Safety
/// `ptr` must be null or point to a valid object whose leading
/// pointer-sized word is the `('super', rclass.OBJECT)` typeptr.
#[inline]
pub unsafe fn ptr_is_virtual_ref(ptr: GCREF) -> bool {
    unsafe {
        if ptr.is_null() {
            return false;
        }
        let header = ptr as *const ObjectHeader;
        (*header).typeptr == JIT_VIRTUAL_REF_VTABLE
    }
}

/// `virtualref.py return vref.forced` — the object a vref has already
/// been forced to, read without forcing.
///
/// For readers that cannot run `force_virtual` at all: forcing materializes an
/// object, and a garbage collection has no allocator to do it with.  Inside a
/// collection the only object a vref can name is the one it was already forced
/// to, so a null result means "still virtual" and the reader must stop there
/// rather than invent one.
///
/// # Safety
/// `ptr` must satisfy [`ptr_is_virtual_ref`], which the caller checks first.
#[inline]
pub unsafe fn vref_forced(ptr: GCREF) -> GCREF {
    unsafe { (*(ptr as *const JitVirtualRef)).forced }
}

/// `rpython/rlib/jit.py class InvalidVirtualRef(Exception)` —
/// `force_virtual` raises this when `virtual_token == TOKEN_NONE`
/// but `forced` is null (`virtualref.py`).  Pyre's single
/// canonical definition lives in `crate::jit::InvalidVirtualRef`
/// (mirrors `virtualref.py from rpython.rlib.jit import
/// InvalidVirtualRef`); re-exported here for convenience.
pub use crate::jit::InvalidVirtualRef;

/// Allocate a concrete JitVirtualRef.
/// `virtualref.py virtual_ref_during_tracing(real_object)`.
/// Initializes virtual_token = TOKEN_NONE, forced = real_object.
/// Returns raw pointer; caller owns the allocation.
///
/// `lltype.malloc(self.JIT_VIRTUAL_REF)` is a GC allocation, and it has to be
/// one here too: `forced` is a traced slot of the type registered with
/// [`set_vref_gc_type_id`], so once `ExecutionContext.topframeref` holds the
/// vref instead of the frame, this edge keeps the frame it wraps reachable.
/// A host-heap allocation is invisible to the
/// collector — the root walker's `gc_current_object_address` early-out returns
/// an unowned address unchanged — which drops that edge and lets a live frame
/// be collected out from under the walk.
///
/// Old-gen, not nursery: `virtualref_boxes`, the recorded ops' concrete stamps
/// and the interpreter's `f_backref` chain all hold this address as a raw
/// pointer that no root walker forwards, and MiniMark's old-gen is mark-sweep,
/// so the address is stable across a minor collection.  The `forced` frame may
/// be young, hence the creation write barrier.
///
/// The `Box` fallback covers only the window before `set_vref_gc_type_id` has
/// run (the id still reads its unset sentinel), where there is no registered
/// type to allocate and nothing has handed a vref to the collector yet; it is
/// leaked, and reclamation is what it gives up.  Once the id is set the
/// allocation stays collector-owned or fails loudly — a host box past that
/// point would silently drop the `forced` edge this object exists to hold.
fn alloc_virtual_ref(real_object: GCREF) -> GCREF {
    let vref = JitVirtualRef {
        super_: ObjectHeader {
            typeptr: JIT_VIRTUAL_REF_VTABLE,
        },
        virtual_token: TOKEN_NONE,
        forced: real_object,
    };
    let type_id = vref_gc_type_id();
    if type_id != VREF_GC_TYPE_ID_UNSET {
        let gcref = majit_gc::alloc_oldgen_typed(type_id, std::mem::size_of::<JitVirtualRef>());
        assert!(
            gcref.0 != 0,
            "JitVirtualRef old-gen allocation failed after the type was registered",
        );
        unsafe { std::ptr::write(gcref.0 as *mut JitVirtualRef, vref) };
        // Creation write barrier: an old-gen vref may point at a young frame.
        majit_gc::gc_write_barrier(gcref);
        return gcref.0 as GCREF;
    }
    Box::into_raw(Box::new(vref)) as GCREF
}

/// Token value indicating no JIT frame is active.
/// `virtualizable.py TOKEN_NONE = lltype.nullptr(llmemory.GCREF.TO)`.
pub const TOKEN_NONE: GCREF = std::ptr::null_mut();

/// `virtualizable.py:326`:
/// ```python
/// _DUMMY = lltype.GcStruct('JITFRAME_DUMMY')
/// ```
/// `virtualizable.py` uses a real pointer to the `_dummy` object, so
/// the translated identity word is pointer-sized too.
pub const JITFRAME_DUMMY_VTABLE: usize = 0x4A46_444D; // "JFDM"

/// Lazy initialisation of the `_dummy` address, as a `usize` because raw
/// pointers are not `Sync`.  Zero means "not minted yet"; a null sentinel would
/// collide with `TOKEN_NONE`, so it is not a value this can ever hold.
static TRACING_RESCALL_DUMMY_PTR: AtomicUsize = AtomicUsize::new(0);

const TRACING_RESCALL_DUMMY_GC_TYPE_ID_UNSET: u32 = u32::MAX;
static TRACING_RESCALL_DUMMY_GC_TYPE_ID: AtomicU32 =
    AtomicU32::new(TRACING_RESCALL_DUMMY_GC_TYPE_ID_UNSET);

/// Publish the registered leaf type used by the prebuilt
/// `virtualizable.py JITFRAME_DUMMY` object.
///
/// Registration comes from `build_gc`, so a second call means a second heap.
/// A sentinel minted in the previous one is no longer part of the live heap —
/// `is_managed_heap_object` would stop recognising it and the traced
/// `virtual_token` / `vable_token` slots would be back to holding an address
/// the collector does not own. Drop it so the next request mints in the heap
/// that is now current.
pub fn set_tracing_rescall_dummy_gc_type_id(type_id: u32) {
    TRACING_RESCALL_DUMMY_GC_TYPE_ID.store(type_id, Ordering::Relaxed);
    TRACING_RESCALL_DUMMY_PTR.store(0, Ordering::Relaxed);
}

/// `virtualizable.py _dummy = lltype.malloc(_DUMMY)` — allocate
/// the singleton dummy `JITFRAME_DUMMY` object whose address serves as the
/// tracing sentinel. The object is allocated in the GC old generation with a
/// registered leaf type and held by a program-lifetime root, giving it the
/// prebuilt lifetime of `_dummy` while remaining a legal GCREF.
fn allocate_tracing_rescall_dummy() -> GCREF {
    let value = ObjectHeader {
        typeptr: JITFRAME_DUMMY_VTABLE,
    };
    let type_id = TRACING_RESCALL_DUMMY_GC_TYPE_ID.load(Ordering::Relaxed);
    if type_id == TRACING_RESCALL_DUMMY_GC_TYPE_ID_UNSET {
        // The leaf type is registered by the same setup that installs a
        // collector, so an unset id means there is no managed heap to mint the
        // prebuilt object in — the token protocol is being driven without one,
        // as the walker and dispatch unit tests do. A host allocation keeps the
        // address stable and unique for the process while staying outside the
        // managed heap, where `is_managed_heap_object` rejects it before any
        // tracing path reads its header.
        return Box::into_raw(Box::new(value)) as GCREF;
    }
    let dummy = majit_gc::alloc_oldgen_typed(type_id, std::mem::size_of::<ObjectHeader>());
    assert!(!dummy.is_null(), "JITFRAME_DUMMY old-gen allocation failed");
    unsafe { std::ptr::write(dummy.0 as *mut ObjectHeader, value) };

    // `_dummy` is prebuilt and immortal upstream. A leaked root slot gives the
    // collector the same lifetime; old-gen makes its identity stable.
    let root = Box::into_raw(Box::new(dummy));
    unsafe { majit_gc::gc_add_root(root) };
    dummy.0 as GCREF
}

/// Token value used during tracing when a residual call is in progress.
/// `virtualizable.py:330`:
/// ```python
/// TOKEN_TRACING_RESCALL = lltype.cast_opaque_ptr(llmemory.GCREF, _dummy)
/// ```
/// The returned address is a registered, rooted GC leaf, and stays the same for
/// as long as the heap it was minted in is the current one.
#[inline]
pub fn token_tracing_rescall() -> GCREF {
    let minted = TRACING_RESCALL_DUMMY_PTR.load(Ordering::Relaxed);
    if minted != 0 {
        return minted as GCREF;
    }
    let fresh = allocate_tracing_rescall_dummy() as usize;
    // Racing minters both produce a valid sentinel, but the token protocol
    // compares tokens by address, so exactly one may be published. The loser's
    // object is immortal either way — upstream's `_dummy` is prebuilt.
    match TRACING_RESCALL_DUMMY_PTR.compare_exchange(0, fresh, Ordering::Relaxed, Ordering::Relaxed)
    {
        Ok(_) => fresh as GCREF,
        Err(published) => published as GCREF,
    }
}

/// Virtual reference state for a single reference.
///
/// A virtual reference wraps a virtualizable object during JIT execution.
/// When code outside the JIT tries to access the virtual object, the
/// reference is "forced" -- the virtual object is materialized on the heap.
///
/// `virtualref.py` parity: `descr` / `descr_virtual_token` /
/// `descr_forced` hold the live `cpu.sizeof(JIT_VIRTUAL_REF)` and
/// `cpu.fielddescrof(JIT_VIRTUAL_REF, 'virtual_token' | 'forced')`
/// Arcs.  Pyre's underlying generators (`vref_size_descr()` /
/// `make_vref_field_descr_typed(...)`) cache the Arc identity at
/// module level so every read of these fields returns the same Arc
/// that the optimizer-emit sites stamp into `op.descr`
/// (`optimizeopt/virtualize.rs`) — `Arc::ptr_eq` parity with
/// `history.py:125 cpu.get_latest_descr() is op.getdescr()`.
#[derive(Debug, Clone)]
pub struct VirtualRefInfo {
    /// `virtualref.py:32-33` `self.descr = cpu.sizeof(JIT_VIRTUAL_REF, ...)`
    pub descr: majit_ir::DescrRef,
    /// `virtualref.py:40-41` `self.descr_virtual_token = cpu.fielddescrof(..., 'virtual_token')`
    pub descr_virtual_token: majit_ir::DescrRef,
    /// `virtualref.py:42`    `self.descr_forced        = cpu.fielddescrof(..., 'forced')`
    pub descr_forced: majit_ir::DescrRef,
}

impl Default for VirtualRefInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::resume::VRefInfo for VirtualRefInfo {
    /// virtualref.py continue_tracing(gcref, real_object)
    ///
    /// Mirrors RPython:
    ///   if not self.is_virtual_ref(gcref): return
    ///   assert real_object
    ///   vref = ...; assert vref.virtual_token != TOKEN_TRACING_RESCALL
    ///   vref.virtual_token = TOKEN_NONE
    ///   vref.forced = real_object
    fn continue_tracing(&self, vref: i64, virtual_ref: i64) {
        // `virtualref.py continue_tracing(gcref, real_object)`
        // — delegate to the inherent helper which carries the
        // `is_virtual_ref` guard, `assert real_object`, and
        // `assert vref.virtual_token != TOKEN_TRACING_RESCALL`
        // post-conditions.
        unsafe {
            self.continue_tracing(vref as GCREF, virtual_ref as GCREF);
        }
    }
}

// Project convention: RPython's plain `assert ...` ports to
// `debug_assert!(...)`.  Both elide under the production translation
// (RPython `-O`, pyre `--release`); both fire under `assert ...` /
// `cargo test` (RPython untranslated, pyre debug build).  Where
// upstream uses `if not we_are_translated(): assert ...` the port is
// `#[cfg(debug_assertions)] { assert!(...) }` (always-on under
// debug_assertions, elided in release).
//
// The `debug_assert!` calls below correspond to upstream `assert`
// statements at:
//   virtualref.py:104  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:111  (assert vref.forced)
//   virtualref.py:115  (assert vref.virtual_token == TOKEN_TRACING_RESCALL)
//   virtualref.py:125  (assert real_object)
//   virtualref.py:127  (assert vref.virtual_token != TOKEN_TRACING_RESCALL)
//   virtualref.py:166  (assert vref.forced)
//   virtualref.py:169  (assert not vref.forced)
//   virtualref.py:172  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:173  (assert vref.forced)
impl VirtualRefInfo {
    /// Create a VirtualRefInfo.
    pub fn new() -> Self {
        use crate::optimizeopt::virtualize::{
            VREF_FORCED_FIELD_INDEX, VREF_VIRTUAL_TOKEN_FIELD_INDEX, make_vref_field_descr_pub,
            vref_size_descr,
        };
        VirtualRefInfo {
            descr: vref_size_descr(),
            descr_virtual_token: make_vref_field_descr_pub(VREF_VIRTUAL_TOKEN_FIELD_INDEX),
            descr_forced: make_vref_field_descr_pub(VREF_FORCED_FIELD_INDEX),
        }
    }

    /// `virtualref.py force_virtual(inst)` — force a virtual
    /// reference: materialise the virtual object if needed.
    ///
    /// Returns `Err(InvalidVirtualRef)` when `virtual_token ==
    /// TOKEN_NONE` and `forced` is null — the vref was tracked but
    /// never properly initialised (`virtualref.py`).
    ///
    /// `force_now` mirrors upstream's
    /// `ResumeGuardForcedDescr.force_now(cpu, token)` contract
    /// (`compile.py:966-985`):
    ///
    /// ```python
    /// @staticmethod
    /// def force_now(cpu, token):
    ///     deadframe = cpu.force(token)
    ///     faildescr = cpu.get_latest_descr(deadframe)
    ///     assert isinstance(faildescr, ResumeGuardForcedDescr)
    ///     faildescr.handle_async_forcing(deadframe)
    /// ```
    ///
    /// `force_now` MUST mutate the vref through the writeback the
    /// JIT frame performs during `cpu.force(token)` —
    /// `vref.virtual_token` becomes `TOKEN_NONE` and `vref.forced`
    /// becomes the materialised object.  The post-call assertions
    /// (`virtualref.py`) verify these writebacks; pyre
    /// mirrors them as `debug_assert!`s after `force_now` returns.
    ///
    /// TODO (dependency injection).  Upstream
    /// passes `cpu` directly because `ResumeGuardForcedDescr` lives
    /// alongside the runner; pyre's `VirtualRefInfo` cannot import
    /// `dyn Runner` without a crate-cycle (the runner trait lives
    /// in the backend crate).  The closure parameter is the Rust
    /// adaptation: callers (which have both `cpu` and
    /// `MetaInterp::handle_async_forcing` in scope) bind the
    /// closure to `|vref| { let df = cpu.force(...); ... }` — same
    /// effect, opposite control direction.  Convergence path:
    /// promote `Runner` to a metainterp-visible trait (or pull the
    /// resume-data handling into a separate crate without runner
    /// dependency) so `force_now` can call `cpu.force` directly.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual(
        &self,
        vref_ptr: GCREF,
        force_now: impl FnOnce(*mut JitVirtualRef),
    ) -> Result<GCREF, InvalidVirtualRef> {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            let token = vref.virtual_token;
            if token != TOKEN_NONE {
                if token == token_tracing_rescall() {
                    // `virtualref.py:161-167`: "virtual" is not a
                    // virtual at all during tracing; reset token as
                    // the marker that this "virtual" escapes.
                    debug_assert!(!vref.forced.is_null());
                    vref.virtual_token = TOKEN_NONE;
                } else {
                    // `virtualref.py:168-173`: active-frame token —
                    // delegate to the cpu/descr force_now sequence,
                    // then assert post-conditions on the writeback.
                    debug_assert!(vref.forced.is_null());
                    force_now(vref);
                    debug_assert_eq!(vref.virtual_token, TOKEN_NONE);
                    debug_assert!(!vref.forced.is_null());
                }
            } else if vref.forced.is_null() {
                // `virtualref.py:174-176`: token == TOKEN_NONE and
                // the vref was not forced — invalid.
                return Err(InvalidVirtualRef);
            }
            Ok(vref.forced)
        }
    }

    /// Create a virtual reference during tracing.
    ///
    /// virtualref.py: `virtual_ref_during_tracing(real_object)`
    ///
    /// Allocates a concrete JitVirtualRef on the heap with
    /// virtual_token = TOKEN_NONE, forced = real_object.
    pub fn virtual_ref_during_tracing(&self, real_object: GCREF) -> GCREF {
        alloc_virtual_ref(real_object)
    }

    /// `virtualref.py tracing_before_residual_call(gcref)` —
    /// sets token to `TOKEN_TRACING_RESCALL` so that if the callee
    /// forces the vref, we detect it after the call.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose leading pointer-sized word is the type-tag.
    pub unsafe fn tracing_before_residual_call(&self, vref_ptr: GCREF) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:104 assert vref.virtual_token == TOKEN_NONE`
            debug_assert_eq!(vref.virtual_token, TOKEN_NONE);
            vref.virtual_token = token_tracing_rescall();
        }
    }

    /// `virtualref.py tracing_after_residual_call(gcref)` —
    /// returns `true` if the vref was forced during the residual
    /// call.  If not forced, resets token to `TOKEN_NONE`.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns `False`
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose leading pointer-sized word is the type-tag.
    pub unsafe fn tracing_after_residual_call(&self, vref_ptr: GCREF) -> bool {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return false;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:111 assert vref.forced` — by construction
            // a JitVirtualRef has its `forced` field initialised to a
            // non-null real_object at allocation time.
            debug_assert!(!vref.forced.is_null());
            if vref.virtual_token != TOKEN_NONE {
                // `virtualref.py:113-117` not modified by the residual
                // call; assert it is still TOKEN_TRACING_RESCALL and
                // clear it.
                debug_assert_eq!(vref.virtual_token, token_tracing_rescall());
                vref.virtual_token = TOKEN_NONE;
                false
            } else {
                // `virtualref.py:118-120` token was cleared by the
                // callee — "modified during residual call" marker.
                true
            }
        }
    }

    /// `virtualref.py continue_tracing(gcref, real_object)` —
    /// updates the `forced` field and clears the token.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose leading pointer-sized word is the type-tag.
    pub unsafe fn continue_tracing(&self, vref_ptr: GCREF, real_object: GCREF) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            // `virtualref.py:125 assert real_object` — caller must
            // supply a non-null materialised object.
            debug_assert!(!real_object.is_null());
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py VirtualRefInfo.continue_tracing assert vref.virtual_token != TOKEN_TRACING_RESCALL`
            debug_assert_ne!(vref.virtual_token, token_tracing_rescall());
            vref.virtual_token = TOKEN_NONE;
            // The vref is an old-gen allocation and `real_object` can be young,
            // so the `forced` store needs the generational barrier its creation
            // counterpart in `alloc_virtual_ref` already carries.  That
            // counterpart barriers only inside its GC-allocated arm, so skip
            // an address the collector does not own (the `Box` arm taken
            // before the vref type id is registered).
            if majit_gc::gc_owns_object(vref_ptr as usize) {
                majit_gc::gc_write_barrier(majit_ir::GcRef(vref_ptr as usize));
            }
            vref.forced = real_object;
        }
    }

    /// Check if a virtual reference is currently active (has a JIT frame token).
    ///
    /// virtualref.py: token != TOKEN_NONE and token != TOKEN_TRACING_RESCALL
    pub fn is_active(token: GCREF) -> bool {
        token != TOKEN_NONE && token != token_tracing_rescall()
    }

    /// Check if a virtual reference is forced (token == TOKEN_NONE).
    pub fn is_forced(token: GCREF) -> bool {
        token == TOKEN_NONE
    }

    /// Check if a virtual reference is in a residual call.
    pub fn is_in_residual_call(token: GCREF) -> bool {
        token == token_tracing_rescall()
    }

    /// `virtualref.py is_virtual_ref(gcref)`:
    /// ```python
    /// def is_virtual_ref(self, gcref):
    ///     if not gcref:
    ///         return False
    ///     inst = lltype.cast_opaque_ptr(rclass.OBJECTPTR, gcref)
    ///     return inst.typeptr == self.jit_virtual_ref_vtable
    /// ```
    /// Pyre reads `super_.typeptr` (offset 0, the
    /// `('super', rclass.OBJECT)` slot) and compares it against
    /// `JIT_VIRTUAL_REF_VTABLE`.
    ///
    /// # Safety
    /// `ptr` must point to a valid GCREF object whose leading
    /// pointer-sized word is the typeptr, or be null.
    pub unsafe fn is_virtual_ref(&self, ptr: GCREF) -> bool {
        unsafe { ptr_is_virtual_ref(ptr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_gc::header::{GCREF, GCREFOpaque};

    #[test]
    fn test_token_constants() {
        assert!(
            TOKEN_NONE.is_null(),
            "TOKEN_NONE = nullptr(GCREF) per virtualizable.py:329"
        );
        let rescall = token_tracing_rescall();
        assert!(
            !rescall.is_null(),
            "token_tracing_rescall() = cast_opaque_ptr(_dummy) per virtualizable.py:330 — non-null sentinel",
        );
        assert_ne!(
            TOKEN_NONE, rescall,
            "TOKEN_NONE and token_tracing_rescall() must be distinguishable",
        );
        // Stable across calls — `_dummy = lltype.malloc(_DUMMY)` is a single
        // program-lifetime allocation, so repeated reads must yield the
        // same address.
        assert_eq!(rescall, token_tracing_rescall());
        // Address must point at memory we own (the heap-allocated
        // `ObjectHeader` for `_dummy`), not a poison value like
        // `usize::MAX as *mut u8`.  Read-back of the typeptr field
        // probes that a real `_dummy` allocation backs the token —
        // matches `JITFRAME_DUMMY_VTABLE`.
        unsafe {
            let header = rescall as *const ObjectHeader;
            assert_eq!((*header).typeptr, JITFRAME_DUMMY_VTABLE);
        }
    }

    /// `virtualref.py`: `force_virtual` raises
    /// `InvalidVirtualRef` when `virtual_token == TOKEN_NONE` and
    /// `forced` is null.  pyre surfaces the same shape as
    /// `Err(InvalidVirtualRef)`.
    #[test]
    fn force_virtual_invalid_when_token_none_and_forced_null() {
        let info = VirtualRefInfo::new();
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: TOKEN_NONE,
            forced: std::ptr::null_mut(),
        };
        let vref_ptr = (&mut vref as *mut JitVirtualRef).cast::<GCREFOpaque>();
        let err = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_now must not be called when token == TOKEN_NONE")
            })
        }
        .expect_err(
            "TOKEN_NONE + null forced must surface Err(InvalidVirtualRef) per virtualref.py:174-176",
        );
        assert_eq!(err, InvalidVirtualRef);
    }

    /// `virtualref.py:160-167`: when `virtual_token ==
    /// TOKEN_TRACING_RESCALL`, `force_virtual` resets the token to
    /// `TOKEN_NONE` and returns the existing `forced` value
    /// without calling the materialiser.
    #[test]
    fn force_virtual_tracing_rescall_resets_token() {
        let info = VirtualRefInfo::new();
        let real_obj: GCREF = 0xCAFE_F00Du64 as GCREF;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: token_tracing_rescall(),
            forced: real_obj,
        };
        let vref_ptr = (&mut vref as *mut JitVirtualRef).cast::<GCREFOpaque>();
        let result = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_now must not be called for TOKEN_TRACING_RESCALL branch")
            })
        }
        .expect("tracing-rescall branch must succeed");
        assert_eq!(result, real_obj);
        assert!(
            vref.virtual_token.is_null(),
            "token must be reset to TOKEN_NONE"
        );
    }

    /// `virtualref.py`: when `virtual_token != TOKEN_NONE`
    /// and != `TOKEN_TRACING_RESCALL`, `force_virtual` invokes
    /// `force_now`, which (mirroring upstream's `cpu.force(token)`
    /// and `handle_async_forcing`) must mutate the vref so that
    /// `virtual_token` becomes `TOKEN_NONE` and `forced` becomes
    /// the materialised object.  pyre's closure receives the
    /// `*mut JitVirtualRef` and performs the writeback in-place.
    #[test]
    fn force_virtual_active_frame_invokes_force_now_with_vref_mutation() {
        let info = VirtualRefInfo::new();
        let active_token: GCREF = 0xABCDu64 as GCREF;
        let materialized: GCREF = 0xCAFEu64 as GCREF;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: active_token,
            forced: std::ptr::null_mut(),
        };
        let vref_ptr = (&mut vref as *mut JitVirtualRef).cast::<GCREFOpaque>();
        let result = unsafe {
            info.force_virtual(vref_ptr, |v| {
                // Equivalent of `cpu.force(token)` writeback +
                // `faildescr.handle_async_forcing(deadframe)`:
                // populate forced and clear the token.
                (*v).forced = materialized;
                (*v).virtual_token = TOKEN_NONE;
            })
        }
        .expect("active-frame branch must succeed");
        assert_eq!(result, materialized);
        assert!(vref.virtual_token.is_null());
        assert_eq!(vref.forced, materialized);
    }

    /// `virtualref.py`: `tracing_before_residual_call`
    /// returns silently when the gcref is not a JitVirtualRef.
    /// pyre's `is_virtual_ref` guard mirrors the shape — a
    /// non-vref pointer (here a word whose leading bytes are NOT
    /// `JIT_VIRTUAL_REF_VTABLE`) must be left untouched.
    #[test]
    fn tracing_before_residual_call_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = (&mut not_a_vref as *mut u64).cast::<GCREFOpaque>();
        unsafe {
            info.tracing_before_residual_call(ptr);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py`: `tracing_after_residual_call`
    /// returns `false` for a non-vref input.
    #[test]
    fn tracing_after_residual_call_returns_false_for_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = (&mut not_a_vref as *mut u64).cast::<GCREFOpaque>();
        let forced = unsafe { info.tracing_after_residual_call(ptr) };
        assert!(!forced, "non-vref input must surface as not-forced");
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py`: `continue_tracing` returns silently
    /// when the gcref is not a JitVirtualRef.
    #[test]
    fn continue_tracing_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = (&mut not_a_vref as *mut u64).cast::<GCREFOpaque>();
        let real_obj: GCREF = 0xDEADBEEFu64 as GCREF;
        unsafe {
            info.continue_tracing(ptr, real_obj);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// Sanity: a real `JitVirtualRef` IS recognised by
    /// `is_virtual_ref` and the tracing helpers DO mutate it
    /// (`virtualref.py` happy path).
    #[test]
    fn tracing_before_residual_call_mutates_real_vref() {
        let info = VirtualRefInfo::new();
        let real_obj: GCREF = 0xCAFE_F00Du64 as GCREF;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: TOKEN_NONE,
            forced: real_obj,
        };
        let vref_ptr = (&mut vref as *mut JitVirtualRef).cast::<GCREFOpaque>();
        unsafe {
            info.tracing_before_residual_call(vref_ptr);
        }
        assert_eq!(
            vref.virtual_token,
            token_tracing_rescall(),
            "real vref must transition to TOKEN_TRACING_RESCALL",
        );
    }
}
