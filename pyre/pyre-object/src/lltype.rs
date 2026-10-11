//! `lltype.malloc` parity API — unified allocation lowering for pyre objects.
//!
//! Mirrors RPython's `lltype.malloc(T, flavor='gc')`
//! (`rpython/rtyper/lltypesystem/lltype.py`), the user-facing
//! allocation primitive used throughout PyPy's interpreter
//! (`pypy/objspace/std/intobject.py wrapint` etc.). In RPython's
//! translation pipeline, every `lltype.malloc(T)` call is rewritten by
//! the GC transform (`rpython/memory/gctransform/framework.py gct_fv_gc_malloc
//! gct_fv_gc_malloc`) into a managed allocation surrounded by
//! `push_roots` / `pop_roots`:
//!
//! ```text
//! v_alloc = direct_call(malloc_fast_ptr, c_const_gc,
//!                       c_type_id, c_size, ...)
//! # bracketed by push_roots(hop) / pop_roots(hop, livevars)
//! ```
//!
//! pyre has no equivalent transform stage today — its host code is
//! plain Rust compiled by cargo. This module provides the same API
//! shape (the low-level allocation primitive that the GC transform
//! consumes; PyPy interpreter source-level constructors like
//! `pypy/objspace/std/intobject.py wrapint` are plain
//! `W_IntObject(x)` calls and `lltype.malloc` only emerges from the
//! rtyping stage `rpython/rtyper/rclass.py new_instance`) so that:
//!
//! 1. Object constructors are single allocation calls without
//!    per-callsite TLS hooks or conditional branches.
//! 2. Future GC integration replaces the body of [`malloc`] without
//!    changing any caller.
//!
//! Current body: [`majit_gc::header::alloc_with_gc_header`] — the
//! `malloc_fixedsize` analog that prepends a [`GcHeader`] (type id,
//! `flags=0`) and returns the payload pointer, so the write barrier reads a
//! valid header instead of foreign memory. The GC owns the header
//! (incminimark defines its own `HDR`); the object model delegates and stays
//! header-agnostic. These objects are still leaked host allocations; routing
//! them through the nursery allocator with root push/pop is the remaining
//! GC work.
//!
//! [`GcHeader`]: majit_gc::header::GcHeader

pub use majit_gc::GcType;

/// Process-wide cell that the JIT driver uses to deliver the actual
/// GC tid to a `#[pyre_class]` type after registration.  Two modes:
///
/// 1. `TypeIdCell::auto()` — initialized to [`TypeIdCell::UNASSIGNED`].
///    The driver assigns the next available tid via [`Self::set`] and
///    subsequent runtime reads return it.  This is the ergonomic
///    default for `#[pyre_class("name")]` callers who do not want to
///    reserve a slot up front.
/// 2. `TypeIdCell::with(N)` — pre-initialized with a reserved tid.
///    The driver asserts that its registration order produces the same
///    `N` (drift check).  Used by every legacy `#[pyre_class("…",
///    type_id = N)]` site so the contiguous-monotonic invariant the
///    GC's `pytype_to_tid` table relies on stays self-checking.
///
/// The cell is `repr(transparent)` over `AtomicU32` so its layout is
/// identical to the raw atomic; `const fn` constructors mean it can
/// initialize a `pub static` without `LazyLock` overhead.
#[repr(transparent)]
pub struct TypeIdCell(::std::sync::atomic::AtomicU32);

impl TypeIdCell {
    /// Sentinel meaning "no tid assigned yet".  Picked at the high end
    /// of `u32` so it cannot collide with a real registered tid; the
    /// JIT's contiguous-monotonic table is several orders of magnitude
    /// shorter than `u32::MAX`.
    pub const UNASSIGNED: u32 = u32::MAX;

    /// Cell initialized to [`Self::UNASSIGNED`] — auto-id mode.
    pub const fn auto() -> Self {
        Self(::std::sync::atomic::AtomicU32::new(Self::UNASSIGNED))
    }

    /// Cell pre-initialized with an explicit tid — explicit-id mode
    /// (drift-checked at JIT init).
    pub const fn with(n: u32) -> Self {
        Self(::std::sync::atomic::AtomicU32::new(n))
    }

    /// Current value of the cell.  Returns [`Self::UNASSIGNED`] when
    /// auto-mode and the JIT driver has not registered the type yet.
    #[inline]
    pub fn get(&self) -> u32 {
        self.0.load(::std::sync::atomic::Ordering::Acquire)
    }

    /// Write the runtime-assigned tid into the cell.  Called once per
    /// type by the JIT driver's `register_pyre_class` helper.
    #[inline]
    pub fn set(&self, n: u32) {
        self.0.store(n, ::std::sync::atomic::Ordering::Release)
    }

    /// `true` iff the cell still holds [`Self::UNASSIGNED`].  The JIT
    /// driver uses this to decide between writing (auto) and asserting
    /// (explicit).
    #[inline]
    pub fn is_unassigned(&self) -> bool {
        self.get() == Self::UNASSIGNED
    }
}

/// Compile-time descriptor every `#[pyre_class]` type emits, consumed
/// by the JIT driver's GC registration loop in
/// `pyre/pyre-jit/src/eval.rs`.  Mirrors the per-type tuple PyPy's
/// `framework.py` materializes (TYPE_ID + fixed size + GC
/// pointer offsets) plus the static `PyType` the dispatcher uses to
/// recognise the layout at runtime.
pub struct PyreClassDescriptor {
    /// Static `PyType` pointer used by `py_type_check` and stamped
    /// into `ob_header.ob_type`.
    pub pytype_ptr: *const crate::pyobject::PyType,
    /// Runtime-resolved GC tid cell.  Either pre-initialized with an
    /// explicit `type_id = N` (then drift-checked) or starts at
    /// [`TypeIdCell::UNASSIGNED`] and gets stamped by the JIT driver.
    pub gc_type_id: &'static TypeIdCell,
    /// `GcType::SIZE` for this payload (in bytes).
    pub object_size: usize,
    /// Byte offsets of inline `PyObjectRef` fields the GC must trace.
    pub ptr_offsets: &'static [usize],
    /// Python-visible dotted name (`"collections.deque"`), carried for
    /// diagnostics — the GC-root registration guard reports it by name
    /// when a type with managed children was left unregistered.
    pub pyname: &'static str,
    /// Fully-qualified Rust path of the `PyType` static
    /// [`Self::pytype_ptr`] points at (`"pyre_interpreter::module::thread\
    /// ::local_class::LOCAL_TYPE"`), spelled the way the flowgraph names
    /// the global read.  `#[pyre_class]` derives the static's identifier
    /// from the struct name, so the path exists only after macro
    /// expansion and no source search finds it; carrying it here is what
    /// lets the JIT bind the address without a hand-written table row per
    /// type (`pyre-interpreter/src/jit_fnaddr.rs`
    /// `jit_static_pytype_addrs`).
    ///
    /// Declare the type at module scope.  The path comes from
    /// `module_path!()`, which names the enclosing *module*, so a
    /// `#[pyre_class]` written inside a function body would carry a path
    /// missing that function's segment and bind nothing — unlike the raw
    /// identifier difference (`mod r#struct`), which the consumer side
    /// normalizes away.
    pub pytype_path: &'static str,
    /// Fully-qualified Rust path of the TYPE this describes
    /// (`"pyre_interpreter::pycode::W_LineIter"`), as distinct from
    /// [`Self::pytype_path`], which names the `PyType` static.
    ///
    /// Both name one address; the difference is what a reader can join on.
    /// A flow graph that reads `PyreClassPyTypeOf::PYTYPE` — the associated
    /// const spelling of the same singleton — carries no static path at
    /// all: Charon renders that read `<module>::<Impl>::PYTYPE`, one
    /// spelling for every trait impl in the module. The translator
    /// resolves it through the impl's `Self` type instead, and this is the
    /// key that answer lands on.
    ///
    /// Unlike the rendered path, this one is injective — a Rust type has
    /// exactly one — which is what it must be to also serve as the linkage
    /// symbol `patch_static_addr_constants` re-pairs across the build/run
    /// boundary.
    pub struct_path: &'static str,
    /// `typedef.py` `_getusercls` layout generated by
    /// `#[pyre_class(..., user_layout)]`. The GC trace is
    /// `instance_walk_boxed_storage`; the base payload's fixed offsets stay
    /// on [`Self::ptr_offsets`].
    pub mapdict_user_layout: bool,
}

// Safety: every field is either a static-`'static` reference (PyType,
// gc_type_id, ptr_offsets, pyname), a primitive, or a raw pointer to
// read-only static storage; sharing across threads is sound.
unsafe impl Sync for PyreClassDescriptor {}

/// Link-time registry of every `#[pyre_class]` type's descriptor.
///
/// `#[pyre_class]` appends one element per type (via `linkme`'s
/// `distributed_slice`), so this slice is the whole-program set of GC
/// class descriptors — the Rust stand-in for RPython's translation-time
/// walk over every `GcStruct` in the low-level type graph
/// (`rpython/memory/gctransform/framework.py`), which pyre cannot do
/// because Rust has no such pass.  Two consumers read it: the JIT
/// driver's `build_gc` completeness *oracle*, which asserts right before
/// `freeze_types()` that every descriptor with managed children was
/// actually wired into the GC, and `pyre_class_pytype_addrs`, which binds
/// each `PyType` static's address for the translation boundary.  The
/// oracle degrades gracefully under under-population (it can only
/// under-check, never false-alarm); the address binder does not, because
/// a name bound at build time and missing at runtime keeps the
/// build-process address.  Both populations below therefore have to hold
/// the same set.
///
/// `distributed_slice` rejects `wasm32` ("distributed_slice is not
/// implemented for this platform"), which carries the same set in
/// [`WASM_CLASS_DESCRIPTORS`] instead; read both through
/// [`for_each_class_descriptor`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static PYRE_CLASS_DESCRIPTORS: [&'static PyreClassDescriptor] = [..];

/// The same registry on wasm32, populated at constructor time.
///
/// `linkme` has no wasm32 arm, and the reason is the target rather than
/// the crate: a static with a custom `link_section` must be a plain list
/// of bytes there, with no relocation for a pointer field.  A constructor
/// runs before any exported function of the module, so the set is
/// complete by the time the JIT binds its address table, and it is
/// emitted beside the type — so it inherits that module's `cfg` gates
/// exactly as the link-section element does.
#[cfg(target_arch = "wasm32")]
pub static WASM_CLASS_DESCRIPTORS: std::sync::Mutex<Vec<&'static PyreClassDescriptor>> =
    std::sync::Mutex::new(Vec::new());

/// Append one type's descriptor to [`WASM_CLASS_DESCRIPTORS`].
///
/// Called only from the constructor `#[pyre_class]` emits.
#[cfg(target_arch = "wasm32")]
pub fn register_class_descriptor(descr: &'static PyreClassDescriptor) {
    WASM_CLASS_DESCRIPTORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(descr);
}

/// Visit every registered `#[pyre_class]` descriptor, whichever population
/// the target carries.
pub fn for_each_class_descriptor(mut visit: impl FnMut(&'static PyreClassDescriptor)) {
    #[cfg(not(target_arch = "wasm32"))]
    for descr in PYRE_CLASS_DESCRIPTORS {
        visit(descr);
    }
    #[cfg(target_arch = "wasm32")]
    for descr in WASM_CLASS_DESCRIPTORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        visit(descr);
    }
}

/// Link-time descriptor for a `#[pyre_methods]`-generated `type_object()`
/// accessor.  The accessor's body is the `OnceLock<usize>` type-object cache;
/// its `CELL` read is unliftable, so the JIT stamps the accessor
/// `dont_look_inside` (`majit-translate` `front::llbc_hints`) and residualizes
/// the call.  A residual call needs the callee's runtime address, so each
/// accessor registers its census-visible path and pointer here, the same
/// link-time census `PYRE_CLASS_DESCRIPTORS` uses for pytype statics.
pub struct TypeObjectFnDescriptor {
    /// `concat!(module_path!(), "::type_object")` — the `FunctionPath` the
    /// residual call names, matched against `jit_trace_fnaddrs`.
    pub path: &'static str,
    /// The accessor.  `fn() -> PyObjectRef` is a single-word ref return with
    /// no arguments — the plainest residual-call ABI the codewriter emits.
    pub func: fn() -> crate::PyObjectRef,
}

// Safety: `path` is `'static` and `func` is a plain `fn` pointer to
// process-global code; sharing across threads is sound.
unsafe impl Sync for TypeObjectFnDescriptor {}

/// Link-time registry of every `type_object()` accessor, populated by
/// [`register_type_object_fnaddr!`] so `jit_trace_fnaddrs` publishes each
/// residual-call address without a hand-maintained list (which would
/// mis-resolve inline-mod paths and miss per-module `cfg` gates).
/// `distributed_slice` rejects `wasm32`, which carries the same set in
/// [`WASM_TYPE_OBJECT_FNADDRS`] instead; read both through
/// [`for_each_type_object_fnaddr`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static PYRE_TYPE_OBJECT_FNADDRS: [TypeObjectFnDescriptor] = [..];

/// The same registry on wasm32, populated at constructor time.
///
/// `linkme` has no wasm32 arm, and the reason is the target rather than the
/// crate: a static with a custom `link_section` must be a plain list of bytes
/// there, with no relocation for a pointer field, and a descriptor holds two.
/// A constructor runs before any exported function of the module, so an entry
/// is present by the time the JIT binds its address table, and it is emitted
/// beside the accessor — so it inherits the module's `cfg` gates exactly as
/// the link-section element does.
#[cfg(target_arch = "wasm32")]
pub static WASM_TYPE_OBJECT_FNADDRS: std::sync::Mutex<Vec<TypeObjectFnDescriptor>> =
    std::sync::Mutex::new(Vec::new());

/// Append one accessor to [`WASM_TYPE_OBJECT_FNADDRS`].
///
/// Called only from the constructor [`register_type_object_fnaddr!`] emits.
/// An entry appended after `jit_trace_fnaddrs` has been read is not published,
/// and the residual call keeps the address its build-time snapshot carried.
#[cfg(target_arch = "wasm32")]
pub fn register_type_object_fnaddr(path: &'static str, func: fn() -> crate::PyObjectRef) {
    WASM_TYPE_OBJECT_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(TypeObjectFnDescriptor { path, func });
}

/// Visit every registered `type_object()` accessor, whichever population the
/// target carries.
///
/// The JIT stamps the accessor `dont_look_inside` on every target, so a target
/// that visited none would leave those residual calls bound to the addresses
/// the build-script process recorded, which name nothing in the runtime one.
pub fn for_each_type_object_fnaddr(
    mut visit: impl FnMut(&'static str, fn() -> crate::PyObjectRef),
) {
    #[cfg(not(target_arch = "wasm32"))]
    for desc in PYRE_TYPE_OBJECT_FNADDRS {
        visit(desc.path, desc.func);
    }
    #[cfg(target_arch = "wasm32")]
    for desc in WASM_TYPE_OBJECT_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        visit(desc.path, desc.func);
    }
}

/// Publish the enclosing module's `type_object()` accessor for the JIT's
/// residual call.
///
/// Expanded beside every accessor — `#[pyre_methods]`, `py_class!`,
/// and `py_class_typed!` — so each
/// registration carries whatever `cfg` gates its module already carries.  A
/// central list could not: the accessors sit behind nested gates, and one that
/// fell behind would silently drop an address rather than fail to build.
#[macro_export]
macro_rules! register_type_object_fnaddr {
    () => {
        #[cfg(not(target_arch = "wasm32"))]
        #[::linkme::distributed_slice($crate::lltype::PYRE_TYPE_OBJECT_FNADDRS)]
        #[allow(non_upper_case_globals)]
        static __PYRE_TYPE_OBJECT_FNADDR: $crate::lltype::TypeObjectFnDescriptor =
            $crate::lltype::TypeObjectFnDescriptor {
                path: ::core::concat!(::core::module_path!(), "::type_object"),
                func: type_object,
            };

        #[cfg(target_arch = "wasm32")]
        #[::ctor::ctor(unsafe)]
        fn __pyre_register_type_object_fnaddr() {
            $crate::lltype::register_type_object_fnaddr(
                ::core::concat!(::core::module_path!(), "::type_object"),
                type_object,
            );
        }
    };
}

/// Compile-time bridge between a `#[pyre_class]` struct and its
/// per-type static `PyType` / `PyreClassDescriptor`.  Implemented
/// automatically by `#[pyre_class]`; consumed by `py_class_typed!`
/// to thread the static `PyType` pointer through
/// `make_builtin_type_with_layout` without naming the macro-generated
/// suffixed identifier (`RANDOM_TYPE`, `RANDOM_PYRE_CLASS_DESCRIPTOR`,
/// …) at the call site.
pub trait PyreClassPyTypeOf {
    /// Static `PyType` pointer (`*const pyre_object::PyType`).  Read
    /// by `py_class_typed!` and `<W_X>::allocate` to stamp
    /// `ob_header.ob_type`.
    const PYTYPE: *const crate::pyobject::PyType;
    /// Compile-time descriptor consumed by the JIT driver's
    /// `register_pyre_class` helper in `pyre-jit/src/eval.rs`.
    const DESCRIPTOR: &'static PyreClassDescriptor;
    /// Python-visible dotted name (e.g. `"_random.Random"`) carried
    /// verbatim from `#[pyre_class("…", type_id = N)]`.  Consumed by
    /// `#[pyre_methods]` so the impl block doesn't restate it.
    const PYNAME: &'static str;
    /// CPython 3.14 exposes many module-native types as HEAPTYPE even though
    /// their PyPy implementation remains a builtin `TypeDef`.  The class
    /// declaration records that public owner independently of the `PyType`
    /// storage descriptor.
    const CPYTHON_HEAPTYPE: bool;
    /// CPython's IMMUTABLETYPE axis is independent of HEAPTYPE.  Most modern
    /// extension heap types carry both; a few (for example `_random.Random`)
    /// are mutable.
    const CPYTHON_IMMUTABLETYPE: bool;
    /// The storage class imported `MapdictStorageMixin` (`map` followed by
    /// `storage`). `typedef._getusercls` /
    /// `import_from_mixin(MapdictStorageMixin)`.
    const HAS_MAPDICT_MIXIN: bool;
}

/// `lltype.malloc(T, flavor='gc')` parity, *untyped* (no `GcType` bound).
/// Allocates a fixed-size GC object and returns a raw pointer the caller
/// owns until the GC takes over.
///
/// The header type id is 0 — `OBJECT_GC_TYPE_ID`, the rclass root
/// (`W_Root`). Payloads with no assigned id use this bridge. `W_ObjectObject`
/// does not: its header is `W_OBJECT_OBJECT_GC_TYPE_ID`, matching the
/// `INSTANCE_TYPE` vtable. Any `T` with its own assigned id must use
/// [`malloc_typed`].
///
/// Non-PyObject heap allocations (Strings, raw `Vec`s freed via
/// `Box::from_raw`) belong on [`malloc_raw`], not here: they must NOT migrate
/// to the managed allocator.
#[inline]
pub fn malloc<T>(value: T) -> *mut T {
    // Untyped bridge: rclass-root id (`OBJECT_GC_TYPE_ID` = 0).
    majit_gc::header::alloc_with_gc_header(value, 0)
}

/// Legacy typed allocation outside the managed heap.
///
/// This retains the bootstrap/immortal behavior required by existing leaf
/// boxes and explicit raw-root walkers. New ordinary GC objects must use
/// [`malloc_typed_managed`], whose allocation belongs to the collector and
/// therefore receives its registered trace shape.
///
/// `T::type_id()` is `TypeIdCell::UNASSIGNED` (`u32::MAX`) until the JIT
/// driver registers an auto-id type. That sentinel is not a real id and is
/// written as 0, since `u32::MAX` would index the type table out of bounds in
/// the trace path (`registry.get`); the real id lands once the type is
/// registered.
#[inline]
pub fn malloc_typed<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    let type_id = match T::type_id() {
        // UNASSIGNED is a sentinel, not a real type id; write the object-root
        // id 0 until the JIT driver assigns the real one.
        TypeIdCell::UNASSIGNED => 0,
        id => id,
    };
    majit_gc::header::alloc_with_gc_header(value, type_id)
}

/// [`malloc_typed`] for an immortal leaf singleton — the header carries
/// `init_gc_object_immortal`'s `GCFLAG_NO_HEAP_PTRS | GCFLAG_TRACK_YOUNG_PTRS`,
/// so marking skips the object instead of reading a payload it has no trace
/// shape for.
///
/// `GCFLAG_NO_HEAP_PTRS` is a state — "holds no heap pointer, and is not in
/// `prebuilt_root_objects` yet" — which the write barrier is what ends, on the
/// first pointer stored into the object. Upstream can stamp prebuilt objects
/// that do carry reference fields, because those fields name other prebuilt
/// objects and the GC transform barriers every later store. A box allocated at
/// runtime here satisfies neither half: its reference fields are written
/// straight to memory at construction.
///
/// Hence the restriction to payloads holding no heap reference: `None`,
/// `True`, `False`, `NotImplemented`, `Ellipsis`, interned exact-str wrappers
/// (`w_str_from_wtf8_immortal`), and a box whose constructor runs a write
/// barrier on the whole box before anything can collect and whose every later
/// reference store is barriered, which is the builtin `Function`
/// (`function_new_impl`): that barrier ends `GCFLAG_NO_HEAP_PTRS` with the
/// construction-time fields covered. Any other box with reference fields must
/// use [`malloc_typed`] and be reached through the raw-root walkers instead;
/// `majit_gc::header::alloc_with_gc_header_immortal` records the measured
/// failure that follows from stamping one anyway.
#[inline]
pub fn malloc_typed_immortal<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    let type_id = match T::type_id() {
        TypeIdCell::UNASSIGNED => 0,
        id => id,
    };
    majit_gc::header::alloc_with_gc_header_immortal(value, type_id)
}

/// Managed typed allocation.
///
/// `gct_fv_gc_malloc` / `init_gc_object(result, typeid, flags=0)`
/// (`framework.py:807-856`): route through the installed GC allocator so the
/// type id's registered fixed offsets or custom trace hook are applied. Falls
/// back to [`malloc_typed`] only before the runtime hook is installed (unit
/// tests and bootstrap tools).
#[inline]
pub fn malloc_typed_managed<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    let type_id = match T::type_id() {
        TypeIdCell::UNASSIGNED => 0,
        id => id,
    };
    // `Some(null)` is a GC that owns the heap and could not satisfy the
    // request, which the fallback below must not answer; only a missing hook
    // leaves this path free to take it.
    if let Some(raw) =
        crate::gc_hook::GcAllocOutcome::from_hook(crate::gc_hook::try_gc_alloc(type_id, T::SIZE))
            .allocated_or_abort(T::SIZE)
    {
        unsafe {
            std::ptr::write(raw as *mut T, value);
            // The no-collect allocator may fall back to old-gen when the
            // nursery is full.  In that case the freshly-written payload
            // can already contain nursery references and must enter the
            // remembered set.  Nursery allocations ignore this barrier.
            crate::gc_hook::try_gc_write_barrier_managed(raw);
        }
        return raw as *mut T;
    }
    malloc_typed(value)
}

/// `malloc_fixedsize` (`framework.py` `gct_fv_gc_malloc`) with the address
/// kept stable: `external_malloc(..., alloc_young=True)` (`incminimark.py`).
/// The object is born young, so the first minor collection that does not
/// reach it frees it, and it never relocates, so a method that re-derives
/// `self` from a raw `PyObjectRef` across an allocating call keeps a valid
/// pointer — the one property [`malloc_typed_stable`] was taken for. What
/// the old-generation birth bought on top of that, surviving to the next
/// major with no root at all, is not a property any RPython object has.
///
/// The barrier covers the births the young hook declines (no hook
/// installed, a weakref type, `MAJIT_GC_YOUNG_RAWMALLOC=0`), which land old
/// with `TRACK_YOUNG_PTRS` set; on a young block it is a no-op.
#[inline]
pub fn malloc_typed_young_nonmoving<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    let type_id = match T::type_id() {
        TypeIdCell::UNASSIGNED => 0,
        id => id,
    };
    let raw = crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(type_id, T::SIZE);
    if raw.is_null() {
        malloc_typed(value)
    } else {
        unsafe {
            std::ptr::write(raw as *mut T, value);
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            raw as *mut T
        }
    }
}

/// Stable-address variant of [`malloc_typed`]: routes through the
/// non-moving old-gen allocator (`try_gc_alloc_stable_raw`) so the object
/// never relocates across a later collection.  A self-mutating typed payload
/// whose methods re-derive `self` from a raw `PyObjectRef` across an
/// allocating call needs this: under the movable [`malloc_typed`] a
/// collection triggered mid-method (e.g. by allocating a fresh backing list)
/// relocates the object, leaving that raw `self` pointer stale — the same
/// rationale as `alloc_instance_object`.  Falls back to the movable
/// [`malloc_typed`] when no stable hook is installed (single-crate tests /
/// pre-init snapshot tools).
#[inline]
pub fn malloc_typed_stable<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    let type_id = match T::type_id() {
        TypeIdCell::UNASSIGNED => 0,
        id => id,
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(type_id, T::SIZE);
    if raw.is_null() {
        malloc_typed(value)
    } else {
        unsafe {
            std::ptr::write(raw as *mut T, value);
            // The object is born old-gen with `TRACK_YOUNG_PTRS` set but is
            // not yet in the remembered set, so a minor collection would not
            // scan it — any inline `PyObjectRef` field still pointing into the
            // nursery would be dropped.  Run the barrier once so the object
            // joins the remembered set and the first minor GC forwards its
            // young children, mirroring the hand-written barrier every other
            // stable allocator runs after its raw write (`w_generator_new`,
            // `tupleobject`).
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            raw as *mut T
        }
    }
}

/// `lltype.malloc(T, flavor='raw')` parity. Non-GC heap allocation;
/// caller manages lifetime via `Box::from_raw` later.
///
/// Unlike [`malloc`] / [`malloc_typed`] (which now prepend a [`GcHeader`]
/// via `alloc_with_gc_header`), this stays a bare `Box::into_raw` with no
/// header, so `Box::from_raw` on its output remains sound. Used for
/// `String`s, dict `dstorage`, and other allocations that must NOT migrate
/// to the managed allocator.
///
/// [`GcHeader`]: majit_gc::header::GcHeader
#[inline]
pub fn malloc_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    // GC-flavored mallocs (`malloc` / `malloc_typed`) are leaked in
    // these tests — the managed allocator forbids
    // `Box::from_raw` on its output, so the tests stay forward-compatible
    // by never freeing GC-flavor allocations. Only `malloc_raw`
    // (RPython `flavor='raw'`) is paired with explicit
    // `Box::from_raw` cleanup.

    #[test]
    fn malloc_returns_unique_pointers() {
        let a = malloc(0u64);
        let b = malloc(0u64);
        assert_ne!(a as usize, b as usize);
    }

    #[test]
    fn malloc_writes_value() {
        let p = malloc(42u32);
        unsafe {
            assert_eq!(*p, 42);
        }
    }

    #[test]
    fn malloc_prepends_zeroed_gc_header() {
        let p = malloc(0x1234_5678_u64);
        unsafe {
            assert_eq!(*p, 0x1234_5678);
            // A zeroed GcHeader precedes the payload, so the write barrier
            // reads `*(obj - SIZE)` as TRACK_YOUNG_PTRS=0 and skips it.
            let hdr = majit_gc::header::header_of(p as usize);
            assert_eq!((*hdr).tid_and_flags, 0);
        }
    }

    #[test]
    fn malloc_raw_independent_of_malloc() {
        let a = malloc(1u32);
        let b = malloc_raw(2u32);
        assert_ne!(a as usize, b as usize);
        unsafe {
            assert_eq!(*a, 1);
            assert_eq!(*b, 2);
            // `b` came from `malloc_raw` so explicit cleanup is sound.
            drop(Box::from_raw(b));
        }
    }

    struct DummyPayload(u64);
    impl GcType for DummyPayload {
        fn type_id() -> u32 {
            0xDEAD_BEEF
        }
        const SIZE: usize = std::mem::size_of::<DummyPayload>();
    }

    #[test]
    fn malloc_typed_writes_value_and_reads_type_metadata() {
        assert_eq!(<DummyPayload as GcType>::type_id(), 0xDEAD_BEEF);
        assert_eq!(<DummyPayload as GcType>::SIZE, 8);
        let p = malloc_typed(DummyPayload(7));
        unsafe {
            assert_eq!((*p).0, 7);
            // The header carries the real GC type id, not a 0 placeholder.
            let hdr = majit_gc::header::header_of(p as usize);
            assert_eq!((*hdr).type_id(), 0xDEAD_BEEF);
            assert_eq!((*hdr).flags(), majit_gc::GcFlags::empty());
        }
    }
}
