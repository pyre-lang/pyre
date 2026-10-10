//! RPython `rpython/rtyper/rbuiltin.py` — reprs for `SomeBuiltin`
//! / `SomeBuiltinMethod` annotations.
//!
//! Upstream dispatches every `SomeBuiltin(const)` to
//! [`BuiltinFunctionRepr`] (rbuiltin.py) and every
//! `SomeBuiltinMethod` to `BuiltinMethodRepr` (rbuiltin.py).
//! Both repr types carry no runtime value (the function / bound
//! receiver is known statically) — the calling hop dispatches into a
//! module-level `BUILTIN_TYPER` registry.
//!
//! ## Status of this port
//!
//! Only the narrow surface needed to lift `SomeBuiltin.rtyper_makerepr`
//! out of the `MissingRTypeOperation` fallback in
//! [`crate::translator::rtyper::rmodel::rtyper_makerepr`] is ported
//! today:
//!
//! * [`BuiltinFunctionRepr`] — stateless Void-typed repr that carries
//!   the builtin identifier (pyre: [`HostObject`]).
//! * [`somebuiltin_rtyper_makerepr`] — upstream `SomeBuiltin.
//!   rtyper_makerepr(self, rtyper)` dispatcher, factored into a free
//!   function so [`rmodel`] can route into it without an import cycle.
//!
//! ## Deferred
//!
//! * Concrete `rtype_method_<name>` overrides on each `Repr` subclass
//!   (e.g. `ListRepr.rtype_method_append`, `StringRepr.rtype_method_join`)
//!   — route through `Repr::rtype_method` once the `r<type>.py` ports
//!   land.
//! * `extregistry.specialize_call` (rbuiltin.py) — the
//!   [`crate::translator::rtyper::extregistry::ExtRegistryEntry`] type
//!   has `is_registered` / `lookup` wired for the `_ptr` entry only
//!   (via `_ptrEntry` at lltype.py, which does not override
//!   `specialize_call`, matching upstream `AttributeError` when this
//!   path is hit). The dispatch to `entry.specialize_call` for entry
//!   types whose upstream subclasses do override it (e.g. `llhelper`,
//!   `llmemory.*` family, `objectmodel.*` hints) is not ported yet;
//!   those entry types must first be added as new `ExtRegistryEntry`
//!   enum variants before `findbltintyper` can route them.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use crate::translator::rtyper::rrange::{
    rtype_builtin_enumerate, rtype_builtin_range, rtype_builtin_xrange,
};

/// Process-global counter for the legacy InstanceRepr→PtrRepr swap
/// inside [`rtype_cast_ptr_to_int`].  Each fire is one
/// `cast_ptr_to_int` call whose operand reached the typer without a
/// producer-set `SomeValue::Ptr` annotation — i.e. the struct's
/// `Ptr(GcStruct(...))` is not in the walker lltype catalog yet
/// (typed-ref-someptr-followup progression).
///
/// Progression gate (swap deletion): this counter must read 0
/// across a representative production run.  Read it via
/// [`swap_fallback_hits`]; reset between runs via
/// [`reset_swap_fallback_hits`].
pub(crate) static SWAP_FALLBACK_HITS: AtomicU64 = AtomicU64::new(0);

/// Read the current value of the `cast_ptr_to_int` swap fallback hit
/// counter.  Used by readiness checks + by unit tests.
pub fn swap_fallback_hits() -> u64 {
    SWAP_FALLBACK_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reset the swap fallback hit counter to zero.  Test fixtures call
/// this before exercising a `cast_ptr_to_int` lowering so the
/// post-call observation is independent of any earlier-in-process
/// fires (the registry is process-global so cross-test bleed is
/// otherwise unavoidable).
pub fn reset_swap_fallback_hits() {
    SWAP_FALLBACK_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}

use crate::annotator::model::{SomeBuiltin, SomeBuiltinMethod, SomeValue};
use crate::flowspace::model::{ConstValue, Constant, HOST_ENV, Hlvalue, HostObject};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::extregistry::{self, ExtRegistryEntry};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{ConvertedTo, HighLevelOp, RPythonTyper};

/// RPython `BUILTIN_TYPER = {}` (rbuiltin.py).
///
/// Module-level registry mapping builtin callables to their
/// `rtype_builtin_*` specializer. Upstream populates this dict via
/// the `@typer_for(func)` decorator at module import time
/// (rbuiltin.py); the Rust port populates it via
/// [`typer_for`] calls from module initializers (to be wired when
/// the first concrete `@typer_for` port lands).
///
/// Keyed by [`HostObject`] because upstream uses the Python builtin
/// function object itself as the dict key (Python dict uses `id()`
/// for unhashable callables — Rust mirrors via
/// [`Arc::ptr_eq`]-based identity on [`HostObject`]).
static BUILTIN_TYPER: OnceLock<Mutex<HashMap<HostObject, BuiltinTyperFn>>> = OnceLock::new();

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn builtin_typer_map() -> &'static Mutex<HashMap<HostObject, BuiltinTyperFn>> {
    BUILTIN_TYPER.get_or_init(|| {
        let mut map = HashMap::new();
        install_default_typers(&mut map);
        Mutex::new(map)
    })
}

/// Register the `@typer_for(...)` entries from upstream rbuiltin.py
/// that the Rust port has ported so far. Called once via the
/// [`BUILTIN_TYPER`] `OnceLock` initializer.
///
/// Each `(builtin_name, typer_fn)` pair mirrors one
/// `@typer_for(<python builtin>)` decorator in upstream. The Python
/// builtin is resolved via [`HOST_ENV::lookup_builtin`]; missing
/// entries are silently skipped so bootstrap stays robust when the
/// host environment is partially populated.
///
/// Module-qualified `@typer_for(<module>.<attr>)` entries (rarithmetic,
/// objectmodel, lltype) reach the registry via the separate
/// `module_entries` loop further down — `HOST_ENV.import_module
/// (...).module_get(<attr>)`.
///
/// Outstanding backlog from upstream rbuiltin.py — each batch needs its
/// own dependent infra (HOST_ENV entry, helper-graph registration hook,
/// inputarg coercion primitive, or `Repr` trait extension) before the
/// per-typer body can land:
///
///   * rbuiltin.py — `min` / `max` landed (`ll_min` / `ll_max`
///     helper graphs keyed per argument lltype).
///   * rbuiltin.py — `reversed` (need `Repr::newiter` trait
///     method + iterator repr family)
///   * rbuiltin.py — `object.__init__` is trivial and landed.
///     `EnvironmentError.__init__` stores `errno` / `strerror` /
///     `filename` through `InstanceRepr::setfield`.
///     `WindowsError.__init__` stores `winerror` the same way.
///   * rbuiltin.py — `objectmodel.hlinvoke` (PBC-callable
///     dispatch)
///   * rbuiltin.py — `range` / `xrange` / `enumerate`
///     (delegate to ported `rrange.py`)
///   * rbuiltin.py — `lltype.malloc` / `free` family fully
///     landed (parse_kwds-driven flag plumbing for `flavor` / `zero` /
///     `track_allocation` / `add_memory_pressure` / `nonmovable`,
///     `malloc_varsize` opname switch on `nb_args == 2`).
///     `cast_pointer` / `cast_opaque_ptr` /
///     `length_of_simple_gcarray_from_opaque` / `direct_fieldptr` /
///     `direct_arrayitems` / `direct_ptradd` / `render_immortal` are
///     landed.  Outstanding: `cast_primitive` (rbuiltin.py, needs
///     `gen_cast` helper + the cast-table at `rbuiltin.py+`).
///   * rbuiltin.py — `llmemory.*` family (need `Address` /
///     `_fakeaddress` ports).  `cast_ptr_to_int` /
///     `cast_int_to_ptr` (rbuiltin.py/551) are ported —
///     frontend `expr as T` for `Ref↔Int` emits `Call { target:
///     FunctionPath { segments: ["rpython", "rtyper", "lltypesystem",
///     "lltype", "cast_*"], fun_decl_id: None }, args }`, routed through the
///     `flowspace_adapter` module-qualified HOST_ENV resolver to
///     `BuiltinFunctionRepr.rtype_simple_call → BUILTIN_TYPER →
///     rtype_cast_ptr_to_int / rtype_cast_int_to_ptr`.  The 2-arg
///     upstream shape (`simple_call(lltype.cast_int_to_ptr, T_const,
///     v_int)`) is reduced to a 1-arg call because pyre's surface DSL
///     has no constant carrier for the target Ptr type; the result
///     lltype is recovered from `hop.r_result.lowleveltype` (matches
///     upstream's `resulttype=hop.r_result.lowleveltype`).
///     TODO: `InstanceRepr→PtrRepr` swap in
///     `rtype_cast_ptr_to_int` body awaits typed `&Foo` lift to
///     `SomePtr`.
///   * rbuiltin.py — `objectmodel.free_non_gc_object` landed
///     via `Repr::gc_flavor_str()` (default `None`, `InstanceRepr`
///     overrides) — the `std::any::Any` downcast that the original
///     port used is retired.  `keepalive_until_here` landed
///     (registry-shape).
///   * rbuiltin.py — `llmemory.cast_*_adr` family: two of
///     four landed (`cast_ptr_to_adr` / `cast_int_to_adr`).
///     `cast_adr_to_ptr` / `cast_adr_to_int` both blocked on
///     `raddress.AddressRepr` port — upstream asserts
///     `isinstance(hop.args_r[0], raddress.AddressRepr)` in both
///     bodies (rbuiltin.py, 667).  Free function `offsetof`
///     (rbuiltin.py) blocked on `Symbolic` offset value support
///     in `ConstValue` (upstream returns the `AddressOffset` instance
///     as a Signed constant; pyre's `ConstValue` enum has no
///     symbolic variant).
///   * rbuiltin.py — `objectmodel.instantiate` (PBC handling +
///     `rclass.rtype_new_instance`)
///   * rbuiltin.py — `OrderedDict` / `objectmodel.r_dict` /
///     `objectmodel.r_ordereddict` lower through
///     `OrderedDictRepr::ll_newdict` and, when `custom_eq_hash`,
///     store `fnkeyeq` / `fnkeyhash`.
///   * rbuiltin.py — weakref family low-level path
///     (`weakref_create/deref`, `cast_ptr_to_weakrefptr`,
///     `cast_weakrefptr_to_ptr` — ported; high-level `BaseWeakRefRepr`
///     path + `weakref.ref` alias deferred)
///
/// Each batch is a natural stand-alone commit once its dependent
/// infra (HOST_ENV entry, helper-graph registration hook, `Repr`
/// trait method, or inputarg coercion primitive) lands.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn install_default_typers(map: &mut HashMap<HostObject, BuiltinTyperFn>) {
    let entries: &[(&str, BuiltinTyperFn)] = &[
        // rbuiltin.py:172-176
        ("bool", rtype_builtin_bool),
        // rbuiltin.py:178-184
        ("int", rtype_builtin_int),
        // rbuiltin.py:186-189
        ("float", rtype_builtin_float),
        // rbuiltin.py:191-194
        ("chr", rtype_builtin_chr),
        // rbuiltin.py:196-199
        ("unichr", rtype_builtin_unichr),
        // rbuiltin.py:201-203
        ("unicode", rtype_builtin_unicode),
        // rbuiltin.py:205-207
        ("bytearray", rtype_builtin_bytearray),
        // rbuiltin.py:209-211
        ("list", rtype_builtin_list),
        // rrange.py:96-126 — `@typer_for(range)` / `@typer_for(xrange)`.
        ("range", rtype_builtin_range),
        ("xrange", rtype_builtin_xrange),
        // rbuiltin.py — `typer_for(enumerate)(rrange.rtype_builtin_enumerate)`.
        ("enumerate", rtype_builtin_enumerate),
        // rbuiltin.py:234-238
        ("min", rtype_builtin_min),
        // rbuiltin.py:246-250
        ("max", rtype_builtin_max),
        // rbuiltin.py:709-715
        ("hasattr", rtype_builtin_hasattr),
        // rbuiltin.py:264-267
        ("object.__init__", rtype_object__init__),
        // rbuiltin.py — `EnvironmentError.__init__` field stores.
        ("EnvironmentError.__init__", rtype_EnvironmentError__init__),
        // rbuiltin.py:286-297 — registered only when the host exposes
        // WindowsError.__init__; HOST_ENV lookup preserves that conditional.
        ("WindowsError.__init__", rtype_WindowsError__init__),
        // Pyre-internal front-end pointer-downcast narrow (#298).  Keyed
        // by the `__cast_instance_intrinsic` HOST_ENV singleton (same Arc the
        // adapter resolves the call's callable to), lowers to a
        // `cast_pointer` into the narrowed `InstanceRepr`.
        (
            crate::runtime_names::shims::CAST_INSTANCE,
            rtype_cast_instance_intrinsic,
        ),
        // The erasing twin (`p as *mut u8`).  Its annotator dropped the
        // pointee class, so `hop.r_result` is the classdef-less
        // `InstanceRepr` and the same `cast_pointer` body applies — this
        // time as the upcast direction of `pairtype(InstanceRepr,
        // InstanceRepr).convert_from_to`.
        (
            crate::runtime_names::shims::CAST_ADDRESS,
            rtype_cast_instance_intrinsic,
        ),
        (
            crate::runtime_names::shims::LL_ARRAYMOVE,
            rtype_ll_arraymove_intrinsic,
        ),
    ];
    for (name, typer) in entries {
        if let Some(host) = HOST_ENV.lookup_builtin(name) {
            map.insert(host, *typer);
        }
    }

    // Module-qualified `@typer_for(<module>.<attr>)` entries — RPython
    // resolves these via `from rpython.rlib import rarithmetic` plus
    // `rarithmetic.intmask` attribute lookup at decorator time.  The
    // Rust port mirrors the path by looking up the host module via
    // `HOST_ENV.import_module(...).module_get(<attr>)`.
    let module_entries: &[(&str, &str, BuiltinTyperFn)] = &[
        // rbuiltin.py:220-225
        ("rpython.rlib.rarithmetic", "intmask", rtype_intmask),
        // rbuiltin.py:227-231
        (
            "rpython.rlib.rarithmetic",
            "longlongmask",
            rtype_longlongmask,
        ),
        (
            "rpython.rlib.rarithmetic",
            "ulonglongmask",
            rtype_ulonglongmask,
        ),
        (
            "rpython.rlib.rarithmetic",
            "longlonglongmask",
            rtype_longlonglongmask,
        ),
        (
            "rpython.rlib.rarithmetic",
            "ulonglonglongmask",
            rtype_ulonglonglongmask,
        ),
        // `rarithmetic.r_uint` dispatches via
        // `ExtRegistryEntry::ForType::specialize_call`
        // (rarithmetic.py), routed through
        // `BuiltinFunctionRepr::findbltintyper`
        // when the qualname-keyed BUILTIN_TYPER misses.  No
        // module_entries row is needed here.
        // rbuiltin.py
        (
            "rpython.rlib.objectmodel",
            "keepalive_until_here",
            rtype_keepalive_until_here,
        ),
        // rbuiltin.py:632-640
        (
            "rpython.rlib.objectmodel",
            "free_non_gc_object",
            rtype_free_non_gc_object,
        ),
        // rbuiltin.py
        ("rpython.rlib.objectmodel", "instantiate", rtype_instantiate),
        // rbuiltin.py — `rtype_const_result` is registered for
        // four upstream callables.  `front::mir` synthesises the
        // production dispatch path; these entries keep the registry
        // structurally aligned with upstream until M2.5g lands.
        (
            "rpython.rtyper.lltypesystem.lltype",
            "typeOf",
            rtype_const_result,
        ),
        (
            "rpython.rtyper.lltypesystem.lltype",
            "nullptr",
            rtype_const_result,
        ),
        (
            "rpython.rtyper.lltypesystem.lltype",
            "getRuntimeTypeInfo",
            rtype_const_result,
        ),
        (
            "rpython.rtyper.lltypesystem.lltype",
            "Ptr",
            rtype_const_result,
        ),
        // rbuiltin.py:559-563
        (
            "rpython.rtyper.lltypesystem.lltype",
            "identityhash",
            rtype_identity_hash,
        ),
        // rbuiltin.py:565-572
        (
            "rpython.rtyper.lltypesystem.lltype",
            "runtime_type_info",
            rtype_runtime_type_info,
        ),
        // rbuiltin.py:463-469
        (
            "rpython.rtyper.lltypesystem.lltype",
            "direct_ptradd",
            rtype_direct_ptradd,
        ),
        // rbuiltin.py:446-453
        (
            "rpython.rtyper.lltypesystem.lltype",
            "direct_fieldptr",
            rtype_direct_fieldptr,
        ),
        // rbuiltin.py:455-461
        (
            "rpython.rtyper.lltypesystem.lltype",
            "direct_arrayitems",
            rtype_direct_arrayitems,
        ),
        // rbuiltin.py:429-436
        (
            "rpython.rtyper.lltypesystem.lltype",
            "cast_opaque_ptr",
            rtype_cast_opaque_ptr,
        ),
        // rbuiltin.py:438-444
        (
            "rpython.rtyper.lltypesystem.lltype",
            "length_of_simple_gcarray_from_opaque",
            rtype_length_of_simple_gcarray_from_opaque,
        ),
        // rbuiltin.py:420-427
        (
            "rpython.rtyper.lltypesystem.lltype",
            "cast_pointer",
            rtype_cast_pointer,
        ),
        // rbuiltin.py:471-477
        (
            "rpython.rtyper.lltypesystem.lltype",
            "cast_primitive",
            rtype_cast_primitive,
        ),
        // rbuiltin.py:543-548
        (
            "rpython.rtyper.lltypesystem.lltype",
            "cast_ptr_to_int",
            rtype_cast_ptr_to_int,
        ),
        // rbuiltin.py:551-557
        (
            "rpython.rtyper.lltypesystem.lltype",
            "cast_int_to_ptr",
            rtype_cast_int_to_ptr,
        ),
        // rbuiltin.py:403-410
        (
            "rpython.rtyper.lltypesystem.lltype",
            "render_immortal",
            rtype_render_immortal,
        ),
        // rbuiltin.py:387-401
        ("rpython.rtyper.lltypesystem.lltype", "free", rtype_free),
        // rbuiltin.py:349-385
        ("rpython.rtyper.lltypesystem.lltype", "malloc", rtype_malloc),
        // rbuiltin.py:651-657
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_ptr_to_adr",
            rtype_cast_ptr_to_adr,
        ),
        // rbuiltin.py:680-685
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_int_to_adr",
            rtype_cast_int_to_adr,
        ),
        // rbuiltin.py:659-665
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_adr_to_ptr",
            rtype_cast_adr_to_ptr,
        ),
        // rbuiltin.py:667-678
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_adr_to_int",
            rtype_cast_adr_to_int,
        ),
        // rbuiltin.py:577-585
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "raw_malloc",
            rtype_raw_malloc,
        ),
        // rbuiltin.py:587-591
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "raw_malloc_usage",
            rtype_raw_malloc_usage,
        ),
        // rbuiltin.py:593-600
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "raw_free",
            rtype_raw_free,
        ),
        // rbuiltin.py:602-609
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "raw_memcopy",
            rtype_raw_memcopy,
        ),
        // rbuiltin.py:611-618
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "raw_memclear",
            rtype_raw_memclear,
        ),
        // rbuiltin.py:621-626
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "offsetof",
            rtype_offsetof,
        ),
        // rbuiltin.py:744-758
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "weakref_create",
            rtype_weakref_create,
        ),
        // rbuiltin.py rtype_weakref_deref
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "weakref_deref",
            rtype_weakref_deref,
        ),
        // rbuiltin.py rtype_cast_ptr_to_weakrefptr
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_ptr_to_weakrefptr",
            rtype_cast_ptr_to_weakrefptr,
        ),
        // rbuiltin.py rtype_cast_weakrefptr_to_ptr
        (
            "rpython.rtyper.lltypesystem.llmemory",
            "cast_weakrefptr_to_ptr",
            rtype_cast_weakrefptr_to_ptr,
        ),
        // rbuiltin.py — `@typer_for(weakref.ref)`
        ("weakref", "ref", rtype_weakref_create),
        // `lltype.nullptr` analog for the Rust null-pointer builtins;
        // same four spellings as the `ptr_null_constant` analyzer
        // (annotator/builtin.rs).
        ("core.ptr", "null_mut", rtype_ptr_null),
        ("std.ptr", "null_mut", rtype_ptr_null),
        ("core.ptr", "null", rtype_ptr_null),
        ("std.ptr", "null", rtype_ptr_null),
        // Null `Option<fn>` — `SomePtr(FuncType)`, `lltype.nullptr`.
        ("core.ptr", "null_fn", rtype_null_fn),
        // `core.ptr` shares the `std.ptr` attr instance (model.rs), so
        // one entry covers both spellings.
        ("std.ptr", "eq", rtype_ptr_eq),
        ("std.ptr", "copy_nonoverlapping", rtype_copy_nonoverlapping),
        // `pyre_object::lltype::malloc_raw` — raw (non-GC) typed alloc.
        // The GC sibling `malloc_typed` is fused into `NewWithVtable` by
        // the front-end and never reaches `findbltintyper`; the raw path
        // lowers to a residual external `direct_call` here.
        (
            crate::runtime_names::modules::OBJECT_LLTYPE,
            "malloc_raw",
            rtype_malloc_raw,
        ),
    ];
    for (module_name, attr_name, typer) in module_entries {
        if let Some(host) = HOST_ENV
            .import_module(module_name)
            .and_then(|m| m.module_get(attr_name))
        {
            map.insert(host, *typer);
        }
    }
}

/// Signature of every `rtype_builtin_*` dispatcher in the
/// `BUILTIN_TYPER` registry.
///
/// Upstream typers have the signature `def rtype_builtin_xxx(hop,
/// **kwds_i)` (rbuiltin.py-onwards). The Rust port surfaces
/// `kwds_i` as an explicit `&HashMap<String, usize>` that maps
/// upstream `'i_<name>'` keys to their index in `hop.args_v`. Simple
/// calls pass an empty map; keyword-aware typers (e.g.
/// `rtype_malloc`) read specific keys.
pub type BuiltinTyperFn = fn(&HighLevelOp, &HashMap<String, usize>) -> RTypeResult;

/// RPython `typer_for(func)` decorator (rbuiltin.py).
///
/// ```python
/// def typer_for(func):
///     def wrapped(rtyper_func):
///         BUILTIN_TYPER[func] = rtyper_func
///         return rtyper_func
///     return wrapped
/// ```
///
/// The Rust port folds the two-stage decorator into a single
/// registration call — downstream modules invoke
/// `typer_for(host_obj, rtype_builtin_fn)` at startup.
pub fn typer_for(func: HostObject, rtyper_func: BuiltinTyperFn) {
    builtin_typer_map().lock().insert(func, rtyper_func);
}

/// Non-upstream helper: read-only registry lookup, used by
/// [`BuiltinFunctionRepr::findbltintyper`]. Kept separate so tests
/// can probe the registry without re-entering `findbltintyper`'s
/// fallback branches.
fn lookup_typer(func: &HostObject) -> Option<BuiltinTyperFn> {
    builtin_typer_map().lock().get(func).copied()
}

/// Qualname-keyed typer fallback for host callables that are Rust
/// associated functions rather than module-level functions, so they never
/// enter the `HostObject`-keyed `BUILTIN_TYPER` map built from `(module,
/// attr)` imports.  The four empty-string constructor spellings match the
/// annotator's qualname-keyed `string_constructor` registrations
/// (`annotator/builtin.rs` — `String.new` / `String.with_capacity` /
/// `Wtf8Buf.new` / `Wtf8Buf.with_capacity`), keeping the annotate and rtype
/// stages in agreement on the empty-`SomeString` shell.
fn lookup_typer_by_qualname(qualname: &str) -> Option<BuiltinTyperFn> {
    match qualname {
        "String.new" | "String.with_capacity" | "Wtf8Buf.new" | "Wtf8Buf.with_capacity" => {
            Some(rtype_empty_string_ctor)
        }
        _ => None,
    }
}

/// RPython `class BuiltinFunctionRepr(Repr)` (rbuiltin.py).
///
/// Void-typed repr for a statically known Python builtin callable.
/// `lowleveltype = Void` because the builtin identifier is resolved
/// at typing-time via the `BUILTIN_TYPER` registry (not ported yet).
#[derive(Debug)]
pub struct BuiltinFunctionRepr {
    /// RPython `self.builtinfunc = builtinfunc` (rbuiltin.py).
    /// Carried as the pyre [`HostObject`] wrapping the Python builtin;
    /// future `findbltintyper` ports will read this to pick the
    /// right `rtype_builtin_*` dispatcher.
    pub builtinfunc: HostObject,
    state: ReprState,
    lltype: LowLevelType,
}

impl BuiltinFunctionRepr {
    /// RPython `BuiltinFunctionRepr.__init__(self, builtinfunc)`
    /// (rbuiltin.py).
    pub fn new(builtinfunc: HostObject) -> Self {
        BuiltinFunctionRepr {
            builtinfunc,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }

    /// RPython `BuiltinFunctionRepr.findbltintyper(self, rtyper)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def findbltintyper(self, rtyper):
    ///     "Find the function to use to specialize calls to this built-in func."
    ///     try:
    ///         return BUILTIN_TYPER[self.builtinfunc]
    ///     except (KeyError, TypeError):
    ///         pass
    ///     if extregistry.is_registered(self.builtinfunc):
    ///         entry = extregistry.lookup(self.builtinfunc)
    ///         return entry.specialize_call
    ///     raise TyperError("don't know about built-in function %r" % (
    ///         self.builtinfunc,))
    /// ```
    ///
    /// Upstream accepts `rtyper` as a parameter but never reads it —
    /// the Rust port drops the arg. The `entry.specialize_call`
    /// attribute lookup is delegated to
    /// [`ExtRegistryEntry::specialize_call`], which mirrors upstream's
    /// per-subclass override pattern: subclasses that define
    /// `specialize_call` add an arm returning `Ok(typer_fn)`; subclasses
    /// that do not (e.g. `_ptrEntry` in `lltype.py`, the only
    /// registered variant today) yield the upstream `AttributeError`
    /// surface as a `TyperError`. `ExtRegistryEntry` (extregistry.py)
    /// defines no base `specialize_call`, so this is a per-arm decision.
    pub(crate) fn findbltintyper(&self) -> Result<BuiltinTyperFn, TyperError> {
        if let Some(f) = lookup_typer(&self.builtinfunc) {
            return Ok(f);
        }
        // Qualname fallback for Rust associated-function host callables
        // (`String::new`, `Wtf8Buf::new`, …) that are not module-level
        // functions, so they never enter the `(module, attr)`-imported
        // `BUILTIN_TYPER` map.  Mirrors the annotator's qualname-keyed
        // `BUILTIN_ANALYZERS` dispatch (`annotator/builtin.rs`) — the
        // rtyper twin of `string_constructor`.
        if let Some(f) = lookup_typer_by_qualname(self.builtinfunc.qualname()) {
            return Ok(f);
        }
        let as_const = ConstValue::HostObject(self.builtinfunc.clone());
        if extregistry::is_registered(&as_const) {
            let entry = extregistry::lookup(&as_const)
                .expect("extregistry::is_registered returned true but lookup returned None");
            return entry.specialize_call();
        }
        Err(TyperError::message(format!(
            "don't know about built-in function {:?}",
            self.builtinfunc
        )))
    }

    /// RPython `BuiltinFunctionRepr._call(self, hop2, **kwds_i)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def _call(self, hop2, **kwds_i):
    ///     bltintyper = self.findbltintyper(hop2.rtyper)
    ///     hop2.llops._called_exception_is_here_or_cannot_occur = False
    ///     v_result = bltintyper(hop2, **kwds_i)
    ///     if not hop2.llops._called_exception_is_here_or_cannot_occur:
    ///         raise TyperError("missing hop.exception_cannot_occur() or "
    ///                          "hop.exception_is_here() in %s" % bltintyper)
    ///     return v_result
    /// ```
    ///
    fn _call(&self, hop2: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
        // upstream `extregistry.lookup(self.builtinfunc).specialize_call`
        // returns a *bound method* whose `self` carries entry-instance
        // state. Rust's `BuiltinTyperFn` is a function pointer with no
        // capture, so marker entries (which need `meta` + `marker_kind`
        // to lower into `jit_marker(...)`) bypass the
        // `findbltintyper` → `BuiltinTyperFn` indirection and dispatch
        // through `ExtRegistryEntry::specialize_marker_call` directly.
        // The pre-typer `_called_exception_is_here_or_cannot_occur =
        // false` reset and the post-typer assertion stay intact so the
        // upstream contract on `hop.exception_cannot_occur()` /
        // `hop.exception_is_here()` is enforced identically.
        let as_const = ConstValue::HostObject(self.builtinfunc.clone());
        if extregistry::is_registered(&as_const)
            && let Some(entry) = extregistry::lookup(&as_const)
            && matches!(
                entry,
                ExtRegistryEntry::EnterLeaveMarker { .. } | ExtRegistryEntry::LoopHeader { .. }
            )
        {
            hop2.llops
                .borrow_mut()
                ._called_exception_is_here_or_cannot_occur = false;
            let v_result = entry.specialize_marker_call(hop2, kwds_i)?;
            let checked = hop2
                .llops
                .borrow()
                ._called_exception_is_here_or_cannot_occur;
            if !checked {
                return Err(TyperError::message(
                    "missing hop.exception_cannot_occur() or hop.exception_is_here() in marker \
                     specialize_call",
                ));
            }
            return Ok(v_result);
        }
        let bltintyper = self.findbltintyper()?;
        hop2.llops
            .borrow_mut()
            ._called_exception_is_here_or_cannot_occur = false;
        let v_result = bltintyper(hop2, kwds_i)?;
        let checked = hop2
            .llops
            .borrow()
            ._called_exception_is_here_or_cannot_occur;
        if !checked {
            return Err(TyperError::message(
                "missing hop.exception_cannot_occur() or hop.exception_is_here() in builtin typer",
            ));
        }
        Ok(v_result)
    }
}

impl Repr for BuiltinFunctionRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BuiltinFunctionRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::BuiltinFunctionRepr
    }

    /// RPython `BuiltinFunctionRepr.rtype_simple_call(self, hop)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     hop2 = hop.copy()
    ///     hop2.r_s_popfirstarg()
    ///     return self._call(hop2)
    /// ```
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let hop2 = hop.copy();
        hop2.r_s_popfirstarg();
        self._call(&hop2, &HashMap::new())
    }

    /// RPython `BuiltinFunctionRepr.rtype_call_args(self, hop)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def rtype_call_args(self, hop):
    ///     # calling a built-in function with keyword arguments:
    ///     # mostly for rpython.objectmodel.hint()
    ///     hop, kwds_i = call_args_expand(hop)
    ///
    ///     hop2 = hop.copy()
    ///     hop2.r_s_popfirstarg()
    ///     hop2.r_s_popfirstarg()
    ///     return self._call(hop2, **kwds_i)
    /// ```
    fn rtype_call_args(&self, hop: &HighLevelOp) -> RTypeResult {
        let (hop, kwds_i) = call_args_expand(hop)?;
        let hop2 = hop.copy();
        hop2.r_s_popfirstarg();
        hop2.r_s_popfirstarg();
        self._call(&hop2, &kwds_i)
    }
}

/// RPython `call_args_expand(hop)` (rbuiltin.py).
///
/// ```python
/// def call_args_expand(hop):
///     hop = hop.copy()
///     from rpython.annotator.argument import ArgumentsForTranslation
///     arguments = ArgumentsForTranslation.fromshape(
///             hop.args_s[1].const, # shape
///             range(hop.nb_args-2))
///     assert arguments.w_stararg is None
///     keywords = arguments.keywords
///     # prefix keyword arguments with 'i_'
///     kwds_i = {}
///     for key in keywords:
///         kwds_i['i_' + key] = keywords[key]
///     return hop, kwds_i
/// ```
///
/// The Rust port short-circuits the `ArgumentsForTranslation.fromshape(
/// shape, range(N))` trick — upstream abuses `data_w=range(N)` to let
/// `fromshape` place integer indices into `keywords`. We compute the
/// same `i_<name> → index` map directly from the decoded
/// [`crate::flowspace::argument::CallShape`].
pub fn call_args_expand(
    hop: &HighLevelOp,
) -> Result<(HighLevelOp, HashMap<String, usize>), TyperError> {
    let hop = hop.copy();
    let shape_const = {
        let args_s = hop.args_s.borrow();
        if args_s.len() < 2 {
            return Err(TyperError::message(
                "call_args_expand: hop.args_s has fewer than 2 entries",
            ));
        }
        let Some(cv) = args_s[1].const_() else {
            return Err(TyperError::message(
                "call_args_expand: hop.args_s[1] is not a constant",
            ));
        };
        cv.clone()
    };
    let Some(shape) = crate::annotator::unaryop::decode_call_shape(&shape_const) else {
        return Err(TyperError::message(format!(
            "call_args_expand: hop.args_s[1].const does not decode as a CallShape: {shape_const:?}"
        )));
    };
    if shape.shape_star {
        return Err(TyperError::message(
            "call_args_expand: arguments.w_stararg is None assertion failed",
        ));
    }
    let mut kwds_i = HashMap::new();
    for (offset, key) in shape.shape_keys.iter().enumerate() {
        let idx = shape.shape_cnt + offset;
        kwds_i.insert(format!("i_{key}"), idx);
    }
    Ok((hop, kwds_i))
}

/// RPython `parse_kwds(hop, *argspec_i_r)` (rbuiltin.py).
///
/// ```python
/// def parse_kwds(hop, *argspec_i_r):
///     lst = [i for (i, r) in argspec_i_r if i is not None]
///     lst.sort()
///     if lst != range(hop.nb_args - len(lst), hop.nb_args):
///         raise TyperError("keyword args are expected to be at the end of "
///                          "the 'hop' arg list")
///     result = []
///     for i, r in argspec_i_r:
///         if i is not None:
///             if r is None:
///                 r = hop.args_r[i]
///             result.append(hop.inputarg(r, arg=i))
///         else:
///             result.append(None)
///     del hop.args_v[hop.nb_args - len(lst):]
///     return result
/// ```
///
/// Each `argspec_i_r` entry is `(Some(i), Some(r))` to extract arg `i`
/// converted into `r`, `(Some(i), None)` to use `hop.args_r[i]`, or
/// `(None, None)` for a placeholder (keyword argument not supplied).
///
/// The trailing `del hop.args_v[...]` truncates only `args_v` upstream
/// (asymmetry is upstream-faithful: `args_r` / `args_s` keep their
/// original lengths after keyword consumption).
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn parse_kwds(
    hop: &HighLevelOp,
    argspec_i_r: &[(Option<usize>, Option<Arc<dyn Repr>>)],
) -> Result<Vec<Option<crate::flowspace::model::Hlvalue>>, TyperError> {
    let mut lst: Vec<usize> = argspec_i_r.iter().filter_map(|(i, _)| *i).collect();
    lst.sort();
    let nb_args = hop.nb_args();
    // upstream `if lst != range(hop.nb_args - len(lst), hop.nb_args)`.
    // When `len(lst) > nb_args`, Python's `range(negative, nb_args)`
    // produces values starting below zero — they never match `lst`'s
    // non-negative indices, so the comparison fails and the
    // `TyperError` below fires.  Surface the same TyperError before
    // performing the `usize` subtraction (which would underflow).
    let Some(tail_start) = nb_args.checked_sub(lst.len()) else {
        return Err(TyperError::message(
            "keyword args are expected to be at the end of the 'hop' arg list",
        ));
    };
    let expected: Vec<usize> = (tail_start..nb_args).collect();
    if lst != expected {
        return Err(TyperError::message(
            "keyword args are expected to be at the end of the 'hop' arg list",
        ));
    }
    let mut result: Vec<Option<crate::flowspace::model::Hlvalue>> =
        Vec::with_capacity(argspec_i_r.len());
    for (i_opt, r_opt) in argspec_i_r.iter() {
        if let Some(i) = *i_opt {
            let r_effective: Arc<dyn Repr> = match r_opt {
                Some(r) => r.clone(),
                None => hop.args_r.borrow()[i].clone().ok_or_else(|| {
                    TyperError::message("parse_kwds: hop.args_r[i] is None and no override given")
                })?,
            };
            let v = hop.inputarg(&r_effective, i)?;
            result.push(Some(v));
        } else {
            result.push(None);
        }
    }
    hop.args_v.borrow_mut().truncate(tail_start);
    Ok(result)
}

/// RPython `SomeBuiltin.rtyper_makerepr(self, rtyper)` (rbuiltin.py:
/// :23-27).
///
/// ```python
/// def rtyper_makerepr(self, rtyper):
///     if not self.is_constant():
///         raise TyperError("non-constant built-in function!")
///     return BuiltinFunctionRepr(self.const)
/// ```
///
/// Pyre's port delegates to [`BuiltinFunctionRepr::new`] after
/// asserting `is_constant()`. The [`HostObject`] carrier pulled off
/// `SomeBuiltin.base.const_box` is the pyre equivalent of the Python
/// builtin function object `self.const`.
pub fn somebuiltin_rtyper_makerepr(s_builtin: &SomeBuiltin) -> Result<Arc<dyn Repr>, TyperError> {
    let Some(const_box) = &s_builtin.base.const_box else {
        return Err(TyperError::message("non-constant built-in function!"));
    };
    let ConstValue::HostObject(host) = &const_box.value else {
        return Err(TyperError::message(format!(
            "SomeBuiltin.rtyper_makerepr: expected HostObject const, got {:?}",
            const_box.value
        )));
    };
    Ok(Arc::new(BuiltinFunctionRepr::new(host.clone())) as Arc<dyn Repr>)
}

/// RPython `class BuiltinMethodRepr(Repr)` (rbuiltin.py).
///
/// Bound builtin method repr — stores the receiver annotation and its
/// concrete repr; dispatches `rtype_simple_call` via the `self_repr`'s
/// `rtype_method_<methodname>` lookup.
#[derive(Debug)]
pub struct BuiltinMethodRepr {
    /// RPython `self.s_self = s_self` (rbuiltin.py). Shared
    /// [`Rc`] mirroring upstream's identity-preserved receiver
    /// annotation.
    pub s_self: Rc<SomeValue>,
    /// RPython `self.self_repr = rtyper.getrepr(s_self)` (rbuiltin.py).
    pub self_repr: Arc<dyn Repr>,
    /// RPython `self.methodname = methodname` (rbuiltin.py).
    pub methodname: String,
    state: ReprState,
    /// RPython `self.lowleveltype = self.self_repr.lowleveltype`
    /// (rbuiltin.py:120) — bound methods have no runtime identity
    /// separate from their receiver, so the lowleveltype mirrors the
    /// receiver's directly.
    lltype: LowLevelType,
}

impl BuiltinMethodRepr {
    /// RPython `BuiltinMethodRepr.__init__(self, rtyper, s_self,
    /// methodname)` (rbuiltin.py).
    pub fn new(
        rtyper: &RPythonTyper,
        s_self: Rc<SomeValue>,
        methodname: String,
    ) -> Result<Self, TyperError> {
        let self_repr = rtyper.getrepr(&s_self)?;
        let lltype = self_repr.lowleveltype().clone();
        Ok(BuiltinMethodRepr {
            s_self,
            self_repr,
            methodname,
            state: ReprState::new(),
            lltype,
        })
    }
}

impl Repr for BuiltinMethodRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BuiltinMethodRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::BuiltinMethodRepr
    }

    /// RPython `BuiltinMethodRepr.convert_const(self, obj)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def convert_const(self, obj):
    ///     return self.self_repr.convert_const(obj.__self__)
    /// ```
    ///
    /// `obj` is a bound-method constant; `obj.__self__` is the
    /// receiver. The Rust port unwraps
    /// [`HostObject::bound_method_self`] and delegates to
    /// `self.self_repr.convert_const` on the receiver wrapped as
    /// [`ConstValue::HostObject`].
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "BuiltinMethodRepr.convert_const: expected HostObject bound method, got {value:?}"
            )));
        };
        let receiver = host.bound_method_self().ok_or_else(|| {
            TyperError::message(
                "BuiltinMethodRepr.convert_const: HostObject is not a bound method (no __self__)",
            )
        })?;
        self.self_repr
            .convert_const(&ConstValue::HostObject(receiver.clone()))
    }

    /// RPython `BuiltinMethodRepr.rtype_simple_call(self, hop)`
    /// (rbuiltin.py).
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     # methods: look up the rtype_method_xxx()
    ///     name = 'rtype_method_' + self.methodname
    ///     try:
    ///         bltintyper = getattr(self.self_repr, name)
    ///     except AttributeError:
    ///         raise TyperError("missing %s.%s" % (
    ///             self.self_repr.__class__.__name__, name))
    ///     # hack based on the fact that 'lowleveltype == self_repr.lowleveltype'
    ///     hop2 = hop.copy()
    ///     assert hop2.args_r[0] is self
    ///     if isinstance(hop2.args_v[0], Constant):
    ///         c = hop2.args_v[0].value    # get object from bound method
    ///         c = c.__self__
    ///         hop2.args_v[0] = Constant(c)
    ///     hop2.args_s[0] = self.s_self
    ///     hop2.args_r[0] = self.self_repr
    ///     return bltintyper(hop2)
    /// ```
    ///
    /// The `getattr(self.self_repr, 'rtype_method_' + methodname)`
    /// upstream lookup maps to the [`Repr::rtype_method`] trait method
    /// — each concrete `Repr` overrides `rtype_method` to route by
    /// `method_name`. The `assert hop2.args_r[0] is self` upstream
    /// identity check is dropped in Rust; arg0 rewriting is the
    /// observable effect.
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::flowspace::model::Hlvalue;

        let hop2 = hop.copy();
        // `hack based on the fact that lowleveltype == self_repr.lowleveltype`:
        // if args_v[0] is a Constant bound method, pull __self__ and
        // rebind.
        {
            let mut args_v = hop2.args_v.borrow_mut();
            if let Hlvalue::Constant(c) = &args_v[0]
                && let ConstValue::HostObject(host) = &c.value
                && let Some(receiver) = host.bound_method_self()
            {
                let new_const = Constant::new(ConstValue::HostObject(receiver.clone()));
                args_v[0] = Hlvalue::Constant(new_const);
            }
        }
        *hop2.args_s.borrow_mut().get_mut(0).ok_or_else(|| {
            TyperError::message("BuiltinMethodRepr.rtype_simple_call: hop.args_s is empty")
        })? = (*self.s_self).clone();
        *hop2.args_r.borrow_mut().get_mut(0).ok_or_else(|| {
            TyperError::message("BuiltinMethodRepr.rtype_simple_call: hop.args_r is empty")
        })? = Some(self.self_repr.clone());
        self.self_repr.rtype_method(&self.methodname, &hop2)
    }
}

/// RPython `SomeBuiltinMethod.rtyper_makerepr(self, rtyper)`
/// (rbuiltin.py).
///
/// ```python
/// def rtyper_makerepr(self, rtyper):
///     assert self.methodname is not None
///     result = BuiltinMethodRepr(rtyper, self.s_self, self.methodname)
///     return result
/// ```
///
/// `methodname` is a non-optional `String` in the Rust port, so the
/// `assert self.methodname is not None` is structurally enforced.
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub fn somebuiltinmethod_rtyper_makerepr(
    s_method: &SomeBuiltinMethod,
    rtyper: &RPythonTyper,
) -> Result<Arc<dyn Repr>, TyperError> {
    let repr =
        BuiltinMethodRepr::new(rtyper, s_method.s_self.clone(), s_method.methodname.clone())?;
    Ok(Arc::new(repr) as Arc<dyn Repr>)
}

/// RPython `pairtype(BuiltinMethodRepr, BuiltinMethodRepr).convert_from_to`
/// (rbuiltin.py).
///
/// ```python
/// class __extend__(pairtype(BuiltinMethodRepr, BuiltinMethodRepr)):
///     def convert_from_to((r_from, r_to), v, llops):
///         # convert between two MethodReprs only if they are about the
///         # same methodname.
///         if r_from.methodname != r_to.methodname:
///             return NotImplemented
///         return llops.convertvar(v, r_from.self_repr, r_to.self_repr)
/// ```
///
/// Upstream's `NotImplemented` maps to `Ok(None)` — the pairtype
/// dispatcher ([`crate::translator::rtyper::pairtype::pair_convert_from_to`])
/// treats this as "keep walking `pair_mro`".
///
/// Downcasts `&dyn Repr` → `&dyn Any` → `&BuiltinMethodRepr` via the
/// `Any` supertrait on [`Repr`]. Returns `Ok(None)` if either side
/// isn't actually a [`BuiltinMethodRepr`] (defensive — callers guard
/// with `ReprClassId` but the cast keeps this robust to misuse).
pub fn pair_builtin_method_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &crate::flowspace::model::Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let any_to: &dyn std::any::Any = r_to;
    let Some(from) = any_from.downcast_ref::<BuiltinMethodRepr>() else {
        return Ok(None);
    };
    let Some(to) = any_to.downcast_ref::<BuiltinMethodRepr>() else {
        return Ok(None);
    };
    if from.methodname != to.methodname {
        return Ok(None);
    }
    llops
        .convertvar(v.clone(), from.self_repr.as_ref(), to.self_repr.as_ref())
        .map(Some)
}

/// Top-level dispatcher used by [`crate::translator::rtyper::rmodel::rtyper_makerepr`]
/// — routes `SomeBuiltin` and `SomeBuiltinMethod` to their respective
/// ports. Keeps the dispatch surface inside this module so the
/// rmodel-side arm is a one-liner.
pub fn dispatch_rtyper_makerepr(
    s: &SomeValue,
    rtyper: &RPythonTyper,
) -> Result<Arc<dyn Repr>, TyperError> {
    match s {
        SomeValue::Builtin(b) => somebuiltin_rtyper_makerepr(b),
        SomeValue::BuiltinMethod(m) => somebuiltinmethod_rtyper_makerepr(m, rtyper),
        other => Err(TyperError::message(format!(
            "rbuiltin::dispatch_rtyper_makerepr: unexpected SomeValue variant {other:?}"
        ))),
    }
}

// @typer_for(...) registrations — rbuiltin.py- onwards.
// Each function mirrors one upstream `@typer_for(<python builtin>)`
// decorated `rtype_builtin_*` body. Registration is driven by
// [`install_default_typers`] via the [`BUILTIN_TYPER`] OnceLock init.

fn arg_repr(hop: &HighLevelOp, index: usize) -> Result<Arc<dyn Repr>, TyperError> {
    hop.args_r
        .borrow()
        .get(index)
        .cloned()
        .flatten()
        .ok_or_else(|| {
            TyperError::message(format!(
                "builtin typer: hop.args_r[{index}] is None or out of range"
            ))
        })
}

/// RPython `@typer_for(bool) def rtype_builtin_bool(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(bool)
/// def rtype_builtin_bool(hop):
///     # not called any more?
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_bool(hop)
/// ```
pub fn rtype_builtin_bool(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_bool: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_bool(hop)
}

/// RPython `@typer_for(int) def rtype_builtin_int(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(int)
/// def rtype_builtin_int(hop):
///     if isinstance(hop.args_s[0], annmodel.SomeString):
///         assert 1 <= hop.nb_args <= 2
///         return hop.args_r[0].rtype_int(hop)
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_int(hop)
/// ```
///
/// The two branches call `rtype_int` identically; only the
/// `nb_args` assertion range differs. `SomeString` is matched on the
/// enum tag to match upstream's `isinstance` check.
pub fn rtype_builtin_int(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    let is_string = matches!(hop.args_s.borrow().first(), Some(SomeValue::String(_)));
    let nb = hop.nb_args();
    if is_string {
        if !(1..=2).contains(&nb) {
            return Err(TyperError::message(format!(
                "rtype_builtin_int: SomeString branch expects 1 <= nb_args <= 2, got {nb}"
            )));
        }
    } else if nb != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_int: expected nb_args == 1, got {nb}"
        )));
    }
    arg_repr(hop, 0)?.rtype_int(hop)
}

/// RPython `@typer_for(float) def rtype_builtin_float(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(float)
/// def rtype_builtin_float(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_float(hop)
/// ```
pub fn rtype_builtin_float(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_float: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_float(hop)
}

/// RPython `@typer_for(chr) def rtype_builtin_chr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(chr)
/// def rtype_builtin_chr(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_chr(hop)
/// ```
pub fn rtype_builtin_chr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_chr: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_chr(hop)
}

/// RPython `@typer_for(unichr) def rtype_builtin_unichr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(unichr)
/// def rtype_builtin_unichr(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_unichr(hop)
/// ```
pub fn rtype_builtin_unichr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_unichr: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_unichr(hop)
}

/// RPython `@typer_for(unicode) def rtype_builtin_unicode(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(unicode)
/// def rtype_builtin_unicode(hop):
///     return hop.args_r[0].rtype_unicode(hop)
/// ```
///
/// Upstream does not assert `nb_args`; both `unicode(x)` and
/// `unicode(x, encoding)` flow through the same dispatch.
pub fn rtype_builtin_unicode(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_unicode(hop)
}

/// RPython `@typer_for(bytearray) def rtype_builtin_bytearray(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(bytearray)
/// def rtype_builtin_bytearray(hop):
///     return hop.args_r[0].rtype_bytearray(hop)
/// ```
pub fn rtype_builtin_bytearray(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_bytearray(hop)
}

/// RPython `@typer_for(list) def rtype_builtin_list(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(list)
/// def rtype_builtin_list(hop):
///     return hop.args_r[0].rtype_bltn_list(hop)
/// ```
pub fn rtype_builtin_list(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_bltn_list(hop)
}

/// RPython `@typer_for(min) def rtype_builtin_min(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(min)
/// def rtype_builtin_min(hop):
///     v1, v2 = hop.inputargs(hop.r_result, hop.r_result)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(ll_min, v1, v2)
/// ```
pub fn rtype_builtin_min(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    rtype_builtin_min_max(hop, "ll_min")
}

/// RPython `def ll_min(i1, i2)` (rbuiltin.py).
pub fn ll_min<T: PartialOrd + Copy>(i1: T, i2: T) -> T {
    if i1 < i2 {
        return i1;
    }
    i2
}

/// RPython `@typer_for(max) def rtype_builtin_max(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(max)
/// def rtype_builtin_max(hop):
///     v1, v2 = hop.inputargs(hop.r_result, hop.r_result)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(ll_max, v1, v2)
/// ```
pub fn rtype_builtin_max(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    rtype_builtin_min_max(hop, "ll_max")
}

/// RPython `def ll_max(i1, i2)` (rbuiltin.py).
pub fn ll_max<T: PartialOrd + Copy>(i1: T, i2: T) -> T {
    if i1 > i2 {
        return i1;
    }
    i2
}

/// Shared body of `rtype_builtin_min` / `rtype_builtin_max` — the two
/// upstream functions differ only in the `ll_min` / `ll_max` helper
/// they `gendirectcall`.
fn rtype_builtin_min_max(hop: &HighLevelOp, helper_name: &str) -> RTypeResult {
    // Error text names the user-facing builtin (`min` / `max`), not the
    // internal `ll_*` helper that `gendirectcall` resolves below.
    let builtin = helper_name.strip_prefix("ll_").unwrap_or(helper_name);
    if hop.nb_args() != 2 {
        return Err(TyperError::message(format!(
            "{builtin}: expected nb_args == 2, got {}",
            hop.nb_args()
        )));
    }
    let r_result = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .cloned()
            .ok_or_else(|| TyperError::message(format!("{builtin}: r_result missing")))?
    };
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r_result.as_ref()),
        ConvertedTo::Repr(r_result.as_ref()),
    ])?;
    hop.exception_cannot_occur()?;
    let llt = r_result.lowleveltype().clone();
    let llfunc =
        hop.rtyper
            .lowlevel_helper_function(helper_name, vec![llt.clone(), llt.clone()], llt)?;
    hop.gendirectcall(&llfunc, vlist)
}

/// RPython `@typer_for(rarithmetic.intmask) def rtype_intmask(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(rarithmetic.intmask)
/// def rtype_intmask(hop):
///     hop.exception_cannot_occur()
///     vlist = hop.inputargs(lltype.Signed)
///     return vlist[0]
/// ```
pub fn rtype_intmask(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
    Ok(vlist.into_iter().next())
}

/// RPython `@typer_for(rarithmetic.longlongmask) def rtype_longlongmask(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(rarithmetic.longlongmask)
/// def rtype_longlongmask(hop):
///     hop.exception_cannot_occur()
///     vlist = hop.inputargs(lltype.SignedLongLong)
///     return vlist[0]
/// ```
pub fn rtype_longlongmask(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(
        &LowLevelType::SignedLongLong,
    )])?;
    Ok(vlist.into_iter().next())
}

/// `inputargs` coerces the machine word to `SignedLongLongLong`.
pub fn rtype_longlonglongmask(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(
        &LowLevelType::SignedLongLongLong,
    )])?;
    Ok(vlist.into_iter().next())
}

pub fn rtype_ulonglonglongmask(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(
        &LowLevelType::UnsignedLongLongLong,
    )])?;
    Ok(vlist.into_iter().next())
}

pub fn rtype_ulonglongmask(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(
        &LowLevelType::UnsignedLongLong,
    )])?;
    Ok(vlist.into_iter().next())
}

/// RPython `ForTypeEntry(_about_ = r_uint).specialize_call(hop)`
/// (rarithmetic.py).
///
/// ```python
/// def specialize_call(self, hop):
///     v_result, = hop.inputargs(hop.r_result.lowleveltype)
///     hop.exception_cannot_occur()
///     return v_result
/// ```
///
/// The `r_uint` `Repr` has `lowleveltype == Unsigned`, so
/// `inputargs(hop.r_result.lowleveltype)` coerces the input to
/// Unsigned via `pair(SrcRepr, IntegerRepr<Unsigned>).convert_from_to`
/// emitting `cast_int_to_uint` / `cast_float_to_uint` /
/// `cast_bool_to_uint` per the source repr (rint.py,
/// rint.py, rbool.py).
///
/// TODO: upstream dispatch goes through
/// `extregistry._about_` lookup keyed on the class object; pyre keys
/// on the qualname-keyed BUILTIN_TYPER `module_entries` table because
/// no extregistry port exists yet.  Body is parity-correct — only
/// the dispatch lookup diverges.
pub(super) fn rtype_r_uint(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    let result_lltype = {
        let r_result_borrow = hop.r_result.borrow();
        let r_result = r_result_borrow
            .as_ref()
            .ok_or_else(|| TyperError::message("rtype_r_uint: r_result missing".to_string()))?;
        r_result.lowleveltype().clone()
    };
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&result_lltype)])?;
    hop.exception_cannot_occur()?;
    Ok(vlist.into_iter().next())
}

/// rlib/jit.py — `Entry(_about_=we_are_jitted).\
/// specialize_call(self, hop)`:
///
/// ```python
/// def specialize_call(self, hop):
///     hop.exception_cannot_occur()
///     return hop.inputconst(lltype.Signed, _we_are_jitted)
/// ```
///
/// Emits the `_we_are_jitted` symbolic singleton
/// ([`crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID`])
/// as a `Constant`. pyre's `we_are_jitted() -> bool` annotates
/// `SomeBool`, so `r_result` is `Bool` (vs upstream's `Signed`); the
/// symbolic is stamped at the result repr's lltype. The genc
/// `replace_we_are_jitted` (→ `false`) and the JIT-codewriter
/// `fold_we_are_jitted_true` (→ `true`) both key on the `SpecTag`
/// identity, not the carrier lltype.
pub(super) fn rtype_we_are_jitted(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    let result_lltype = {
        let r_result_borrow = hop.r_result.borrow();
        let r_result = r_result_borrow.as_ref().ok_or_else(|| {
            TyperError::message("rtype_we_are_jitted: r_result missing".to_string())
        })?;
        r_result.lowleveltype().clone()
    };
    hop.exception_cannot_occur()?;
    let c = Constant::with_concretetype(
        ConstValue::SpecTag(crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID),
        result_lltype,
    );
    Ok(Some(Hlvalue::Constant(c)))
}

/// RPython `rlib/nonconst.py specialize_call`
/// `EntryNonConstant.specialize_call(self, hop)`:
///
/// ```python
/// def specialize_call(self, hop):
///     hop.exception_cannot_occur()
///     return hop.inputarg(hop.r_result, arg=0)
/// ```
///
/// The whole of `NonConstant` is an annotation-time effect; at the low level
/// the argument is the result.
pub(super) fn rtype_non_constant(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    let r_result = {
        let borrow = hop.r_result.borrow();
        borrow
            .as_ref()
            .ok_or_else(|| TyperError::message("rtype_non_constant: r_result missing".to_string()))?
            .clone()
    };
    hop.exception_cannot_occur()?;
    Ok(Some(hop.inputarg(&r_result, 0)?))
}

/// Rust source marker for RPython
/// `VirtualizableInstanceRepr.hook_access_field`.
///
/// Upstream emits `jit_force_virtualizable(vinst, cname, flags)`
/// directly. The source marker only names the live frame; this typer
/// appends Void `cname` and flags so the residual matches the 3-arg
/// hook. `jtransform.rewrite_op_jit_force_virtualizable` removes it
/// from looked-inside JIT graphs.
pub(super) fn rtype_jit_force_virtualizable(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_frame = hop
        .args_r
        .borrow()
        .first()
        .and_then(Clone::clone)
        .ok_or_else(|| {
            TyperError::message("rtype_jit_force_virtualizable: frame repr missing".to_string())
        })?;
    let mut vlist = hop.inputargs(vec![ConvertedTo::Repr(r_frame.as_ref())])?;
    hop.exception_cannot_occur()?;
    // `VirtualizableInstanceRepr.hook_access_field` emits
    // `jit_force_virtualizable(vinst, cname, cflags)`. The source marker
    // only names the live frame; synthesize the Void field name and
    // flags dict so `replace_force_virtualizable_with_call` can read
    // `op.args[-1].value.get('access_directly')` and
    // `rewrite_op_jit_force_virtualizable` can file `vable_flags`.
    vlist.push(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::byte_str(""),
        LowLevelType::Void,
    )));
    vlist.push(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Dict(std::collections::HashMap::new()),
        LowLevelType::Void,
    )));
    // This is a call returning None, not an operation with an impossible
    // result. LowLevelOpList.genop(resulttype=lltype.Void) returns its Void
    // Variable, as ExtLoopHeader.specialize_call does for a source marker.
    // Omitting resulttype returns no Variable and makes translate_hl_to_ll
    // require s_ImpossibleValue instead of the marker's s_None annotation.
    Ok(hop.genop(
        "jit_force_virtualizable",
        vlist,
        GenopResult::LLType(LowLevelType::Void),
    ))
}

/// `rlib/jit.py ConditionalCallEntry.specialize_call` for
/// `_jit_conditional_call`.
pub(super) fn rtype_jit_conditional_call(
    hop: &HighLevelOp,
    kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    rtype_conditional_call_entry(hop, kwds_i, false)
}

/// `rlib/jit.py ConditionalCallEntry.specialize_call` for
/// `_jit_conditional_call_value`.
pub(super) fn rtype_jit_conditional_call_value(
    hop: &HighLevelOp,
    kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    rtype_conditional_call_entry(hop, kwds_i, true)
}

/// `ConditionalCallEntry.specialize_call`. `is_value` is
/// `self.instance == _jit_conditional_call_value`.
///
/// ```python
/// if self.instance == _jit_conditional_call:
///     opname = 'jit_conditional_call'
///     COND = lltype.Bool
///     resulttype = None
/// elif self.instance == _jit_conditional_call_value:
///     opname = 'jit_conditional_call_value'
///     COND = hop.r_result
///     resulttype = hop.r_result.lowleveltype
/// args_v = hop.inputargs(COND, lltype.Void, *hop.args_r[2:])
/// args_v[1] = hop.args_r[1].get_concrete_llfn(...)
/// hop.exception_is_here()
/// return hop.genop(opname, args_v, resulttype=resulttype)
/// ```
///
/// Void `genop` uses `GenopResult::LLType(Void)` so
/// `translate_hl_to_ll` keeps a Void Variable for s_None. Flavour
/// `_i` vs `_r` comes from `hop.r_result.lowleveltype`.
fn rtype_conditional_call_entry(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
    is_value: bool,
) -> RTypeResult {
    use crate::translator::rtyper::rpbc::FunctionRepr;
    use crate::translator::rtyper::rtyper::GenopResult;

    if hop.nb_args() < 2 {
        return Err(TyperError::message(
            "ConditionalCallEntry.specialize_call: expected at least condition/value and function",
        ));
    }

    let r_result = hop.r_result.borrow().clone();
    let extra: Vec<Arc<dyn Repr>> = {
        let args_r = hop.args_r.borrow();
        args_r
            .iter()
            .skip(2)
            .map(|r| {
                r.clone().ok_or_else(|| {
                    TyperError::message(
                        "ConditionalCallEntry.specialize_call: missing extra-arg repr",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let bool_ll = LowLevelType::Bool;
    let void_ll = LowLevelType::Void;
    if is_value && r_result.is_none() {
        return Err(TyperError::message(
            "ConditionalCallEntry.specialize_call: r_result missing",
        ));
    }
    let mut converted: Vec<ConvertedTo<'_>> = Vec::with_capacity(hop.nb_args());
    if is_value {
        converted.push(ConvertedTo::Repr(
            r_result.as_ref().expect("checked is_none").as_ref(),
        ));
    } else {
        converted.push(ConvertedTo::LowLevelType(&bool_ll));
    }
    converted.push(ConvertedTo::LowLevelType(&void_ll));
    for r in &extra {
        converted.push(ConvertedTo::Repr(r.as_ref()));
    }
    let mut args_v = hop.inputargs(converted)?;

    let s_fn = hop.args_s.borrow().get(1).cloned().ok_or_else(|| {
        TyperError::message("ConditionalCallEntry.specialize_call: missing function annotation")
    })?;
    args_v[1] = match s_fn {
        SomeValue::PBC(pbc) => {
            let fn_repr = FunctionRepr::new(&hop.rtyper, pbc.clone())?;
            let rest = hop.args_s.borrow()[2..].to_vec();
            fn_repr.get_concrete_llfn(&pbc, rest, hop.position_key.borrow().clone())?
        }
        SomeValue::Ptr(_) => {
            let r_fn = hop.args_r.borrow().get(1).and_then(Clone::clone);
            if let Some(r_fn) = r_fn {
                hop.inputarg(ConvertedTo::Repr(r_fn.as_ref()), 1)?
            } else {
                hop.args_v.borrow()[1].clone()
            }
        }
        other => {
            return Err(TyperError::message(format!(
                "ConditionalCallEntry.specialize_call: function arg should be SomePBC or SomePtr, got {other:?}"
            )));
        }
    };

    hop.exception_is_here()?;
    if is_value {
        Ok(hop.genop(
            "jit_conditional_call_value",
            args_v,
            GenopResult::LLType(r_result.expect("checked is_none").lowleveltype().clone()),
        ))
    } else {
        Ok(hop.genop(
            "jit_conditional_call",
            args_v,
            GenopResult::LLType(LowLevelType::Void),
        ))
    }
}

/// `BigInt::from(i64) -> BigInt` residual lowering, reached through
/// `BuiltinFunctionRepr.findbltintyper` →
/// `extregistry.lookup(BigInt.from).specialize_call`
/// ([`super::extregistry::ExtRegistryEntry::BigIntFrom`]).
///
/// Mirrors `extfunc.py ExternalFunctionRepr.rtype_simple_call`:
///
/// ```python
/// def rtype_simple_call(self, hop):
///     args_r = [rtyper.getrepr(s_arg) for s_arg in self.s_func.args_s]
///     r_result = rtyper.getrepr(self.s_func.s_result)
///     obj = self.get_funcptr(rtyper, args_r, r_result)
///     hop2 = hop.copy(); hop2.r_s_popfirstarg()
///     vlist = [hop2.inputconst(typeOf(obj), obj)] + hop2.inputargs(*args_r)
///     hop2.exception_is_here()
///     return hop2.genop('direct_call', vlist, r_result)
/// ```
///
/// The callee has already been popped off `hop` by
/// `BuiltinFunctionRepr._call` (`rbuiltin.py
/// rtype_simple_call` → `r_s_popfirstarg`), so `hop.args_r[0]` is the
/// `i64` source repr and `hop.r_result` is the classdef-less
/// `SomeInstance` result repr (root `InstanceRepr`, lltype
/// `Ptr(OBJECT)` — the opaque GcRef `bigint_from` projects to).  The
/// residual funcptr carries the conversion's true `(Signed) -> Ptr(OBJECT)`
/// signature; the result lltype is the opaque pointer, never a scalar.
///
/// `BigInt::from(i64)` is infallible, so `exception_cannot_occur` (vs the
/// generic external's `exception_is_here`): the cold long arm has no
/// exception edge off this conversion.
pub fn rtype_bigint_from(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::{
        self, FuncType, functionptr_with_external_name,
    };
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = hop
        .args_r
        .borrow()
        .first()
        .cloned()
        .flatten()
        .ok_or_else(|| {
            TyperError::message("rtype_bigint_from: argument repr missing".to_string())
        })?;
    let arg_lltype = r_arg.lowleveltype().clone();
    let r_result =
        hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message("rtype_bigint_from: r_result missing".to_string())
        })?;
    let result_lltype = r_result.lowleveltype().clone();

    // extfunc.py `get_funcptr` — `functionptr(FT, name,
    // _external_name=name, _callable=...)`.  `_callable` carries the host
    // callable's qualname (`BigInt.from`) for the llinterp/backend label.
    let funcptr = functionptr_with_external_name(
        FuncType {
            args: vec![arg_lltype],
            result: result_lltype,
        },
        "bigint_from",
        Some("BigInt.from".to_string()),
    );
    let funcptr_lltype = LowLevelType::Ptr(Box::new(lltype::typeOf(&funcptr)));

    hop.exception_cannot_occur()?;

    // extfunc.py:62 — `vlist = [hop2.inputconst(typeOf(obj), obj)] +
    // hop2.inputargs(*args_r)`.
    let v_func = HighLevelOp::inputconst(&funcptr_lltype, &ConstValue::LLPtr(Box::new(funcptr)))?;
    let mut vlist = vec![Hlvalue::Constant(v_func)];
    vlist.extend(hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?);

    // extfunc.py — `hop2.genop('direct_call', vlist, r_result)`.
    Ok(hop.genop("direct_call", vlist, GenopResult::Repr(r_result)))
}

/// `rlib/longlong2float.py` — `Float2LongLongEntry.specialize_call`:
/// `hop.inputargs(lltype.Float)`, `hop.exception_cannot_occur()`,
/// `return hop.genop("convert_float_bytes_to_longlong", [v_float], SignedLongLong)`.
pub fn rtype_float2longlong(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "convert_float_bytes_to_longlong",
        vlist,
        GenopResult::LLType(LowLevelType::SignedLongLong),
    ))
}

/// `rlib/longlong2float.py` —
/// `LongLong2FloatEntry.specialize_call`.
pub fn rtype_longlong2float(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(
        &LowLevelType::SignedLongLong,
    )])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "convert_longlong_bytes_to_float",
        vlist,
        GenopResult::LLType(LowLevelType::Float),
    ))
}

/// `@typer_for(pyre_object.lltype.malloc_raw)` — residual external
/// lowering of the raw (non-GC) typed allocation intrinsic
/// `malloc_raw::<T>(value: T) -> *mut T` (`malloc_raw_alloc` annotator
/// in `annotator/builtin.rs`).
///
/// The GC allocation intrinsic `malloc_typed` is fused into
/// `NewWithVtable` by the front-end (it carries a GC vtable to
/// materialise); the raw path has no vtable, so it lowers to a residual
/// `direct_call` carrying THIS callsite's monomorphic `(T-repr) -> Ptr`
/// signature — same shape as [`rtype_bigint_from`] /
/// `extfunc.py ExternalFunctionRepr.rtype_simple_call`. The
/// allocation is treated as non-failing (`exception_cannot_occur`),
/// matching the cold long-box arm whose `BigInt.from` is likewise
/// infallible.
pub fn rtype_malloc_raw(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::{
        self, FuncType, functionptr_with_external_name,
    };
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = hop
        .args_r
        .borrow()
        .first()
        .cloned()
        .flatten()
        .ok_or_else(|| {
            TyperError::message("rtype_malloc_raw: argument repr missing".to_string())
        })?;
    let arg_lltype = r_arg.lowleveltype().clone();
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_malloc_raw: r_result missing".to_string()))?;
    let result_lltype = r_result.lowleveltype().clone();

    let funcptr = functionptr_with_external_name(
        FuncType {
            args: vec![arg_lltype],
            result: result_lltype,
        },
        "malloc_raw",
        Some("malloc_raw".to_string()),
    );
    let funcptr_lltype = LowLevelType::Ptr(Box::new(lltype::typeOf(&funcptr)));

    hop.exception_cannot_occur()?;

    let v_func = HighLevelOp::inputconst(&funcptr_lltype, &ConstValue::LLPtr(Box::new(funcptr)))?;
    let mut vlist = vec![Hlvalue::Constant(v_func)];
    vlist.extend(hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?);

    Ok(hop.genop("direct_call", vlist, GenopResult::Repr(r_result)))
}

/// RPython `@typer_for(hasattr) def rtype_builtin_hasattr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(hasattr)
/// def rtype_builtin_hasattr(hop):
///     hop.exception_cannot_occur()
///     if hop.s_result.is_constant():
///         return hop.inputconst(lltype.Bool, hop.s_result.const)
///     raise TyperError("hasattr is only suported on a constant")
/// ```
pub fn rtype_builtin_hasattr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeObjectTrait;
    use crate::flowspace::model::Hlvalue;

    hop.exception_cannot_occur()?;
    let const_val = {
        let s_result_borrow = hop.s_result.borrow();
        let s = s_result_borrow.as_ref().ok_or_else(|| {
            TyperError::message("rtype_builtin_hasattr: s_result missing".to_string())
        })?;
        if !s.is_constant() {
            return Err(TyperError::message(
                "hasattr is only suported on a constant".to_string(),
            ));
        }
        s.const_().cloned().ok_or_else(|| {
            TyperError::message(
                "rtype_builtin_hasattr: is_constant() true but const_() None".to_string(),
            )
        })?
    };
    let c = HighLevelOp::inputconst(&LowLevelType::Bool, &const_val)?;
    Ok(Some(Hlvalue::Constant(c)))
}

/// RPython `@typer_for(reversed) def rtype_builtin_reversed(hop)`
/// (rbuiltin.py).
///
/// ```python
/// def rtype_builtin_reversed(hop):
///     hop.exception_cannot_occur()
///     return hop.r_result.newiter(hop)
/// ```
pub fn rtype_builtin_reversed(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_builtin_reversed: r_result missing"))?;
    r_result.newiter(hop)
}

/// The receiver of an exception `__init__` hop, as `InstanceRepr`.
///
/// `genop` refuses a raw `args_v` entry, so the self value is the
/// `inputarg` identity conversion.
fn hop_instance_self(
    hop: &HighLevelOp,
    what: &str,
) -> Result<
    (
        std::sync::Arc<crate::translator::rtyper::rclass::InstanceRepr>,
        Hlvalue,
    ),
    TyperError,
> {
    use crate::translator::rtyper::rclass::InstanceRepr;
    use crate::translator::rtyper::rtyper::ConvertedTo;

    if hop.nb_args() == 0 {
        return Err(TyperError::message(format!("{what}: missing self")));
    }
    let r_any = hop.args_r.borrow()[0]
        .clone()
        .ok_or_else(|| TyperError::message(format!("{what}: missing self repr")))?;
    let raw = std::sync::Arc::into_raw(r_any);
    if unsafe { (*raw).type_id() } != std::any::TypeId::of::<InstanceRepr>() {
        let _ = unsafe { std::sync::Arc::from_raw(raw) };
        return Err(TyperError::message(format!(
            "{what}: self is not an InstanceRepr"
        )));
    }
    let r_self = unsafe { std::sync::Arc::from_raw(raw as *const () as *const InstanceRepr) };
    let v_self = hop.inputarg(ConvertedTo::Repr(r_self.as_ref()), 0)?;
    Ok((r_self, v_self))
}

/// RPython `@typer_for(EnvironmentError.__init__) def
/// rtype_EnvironmentError__init__(hop)` (rbuiltin.py).
///
/// ```python
/// hop.exception_cannot_occur()
/// v_self = hop.args_v[0]
/// r_self = hop.args_r[0]
/// if hop.nb_args <= 2:
///     v_errno = hop.inputconst(lltype.Signed, 0)
///     if hop.nb_args == 2:
///         v_strerror = hop.inputarg(rstr.string_repr, arg=1)
///         r_self.setfield(v_self, 'strerror', v_strerror, hop.llops)
/// else:
///     v_errno = hop.inputarg(lltype.Signed, arg=1)
///     v_strerror = hop.inputarg(rstr.string_repr, arg=2)
///     r_self.setfield(v_self, 'strerror', v_strerror, hop.llops)
///     if hop.nb_args >= 4:
///         v_filename = hop.inputarg(rstr.string_repr, arg=3)
///         r_self.setfield(v_self, 'filename', v_filename, hop.llops)
/// r_self.setfield(v_self, 'errno', v_errno, hop.llops)
/// ```
#[allow(non_snake_case)]
pub fn rtype_EnvironmentError__init__(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rstr::string_repr;
    use crate::translator::rtyper::rtyper::ConvertedTo;

    hop.exception_cannot_occur()?;
    let (r_self, v_self) = hop_instance_self(hop, "rtype_EnvironmentError__init__")?;
    let nb = hop.nb_args();
    let signed = LowLevelType::Signed;
    let (v_errno, v_strerror, v_filename) = if nb <= 2 {
        let v_errno = Hlvalue::Constant(HighLevelOp::inputconst(&signed, &ConstValue::Int(0))?);
        let v_strerror = if nb == 2 {
            let r_str = string_repr();
            Some(hop.inputarg(ConvertedTo::Repr(r_str.as_ref()), 1)?)
        } else {
            None
        };
        (v_errno, v_strerror, None)
    } else {
        let r_str = string_repr();
        let v_errno = hop.inputarg(ConvertedTo::from(&signed), 1)?;
        let v_strerror = hop.inputarg(ConvertedTo::Repr(r_str.as_ref()), 2)?;
        let v_filename = if nb >= 4 {
            Some(hop.inputarg(ConvertedTo::Repr(r_str.as_ref()), 3)?)
        } else {
            None
        };
        (v_errno, Some(v_strerror), v_filename)
    };
    let flags = crate::translator::rtyper::rclass::Flags::new();
    let mut llops = hop.llops.borrow_mut();
    if let Some(v_strerror) = v_strerror {
        r_self.setfield(
            v_self.clone(),
            "strerror",
            v_strerror,
            &mut llops,
            false,
            &flags,
        )?;
    }
    if let Some(v_filename) = v_filename {
        r_self.setfield(
            v_self.clone(),
            "filename",
            v_filename,
            &mut llops,
            false,
            &flags,
        )?;
    }
    r_self.setfield(v_self, "errno", v_errno, &mut llops, false, &flags)?;
    Ok(None)
}

/// RPython conditional `@typer_for(WindowsError.__init__) def
/// rtype_WindowsError__init__(hop)` (rbuiltin.py).
///
/// ```python
/// hop.exception_cannot_occur()
/// if hop.nb_args == 2:
///     raise TyperError("WindowsError() should not be called with "
///                      "a single argument")
/// if hop.nb_args >= 3:
///     v_self = hop.args_v[0]
///     r_self = hop.args_r[0]
///     v_error = hop.inputarg(lltype.Signed, arg=1)
///     r_self.setfield(v_self, 'winerror', v_error, hop.llops)
/// ```
#[allow(non_snake_case)]
pub fn rtype_WindowsError__init__(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::ConvertedTo;

    hop.exception_cannot_occur()?;
    if hop.nb_args() == 2 {
        return Err(TyperError::message(
            "WindowsError() should not be called with a single argument",
        ));
    }
    if hop.nb_args() >= 3 {
        let (r_self, v_self) = hop_instance_self(hop, "rtype_WindowsError__init__")?;
        let signed = LowLevelType::Signed;
        let v_error = hop.inputarg(ConvertedTo::from(&signed), 1)?;
        let flags = crate::translator::rtyper::rclass::Flags::new();
        let mut llops = hop.llops.borrow_mut();
        r_self.setfield(v_self, "winerror", v_error, &mut llops, false, &flags)?;
    }
    Ok(None)
}

/// Pack a live `Repr` into `ConstValue::Repr`. The box is leaked so the
/// pointer stays valid for the process and `ConstValue` stays `Send`.
pub(crate) fn repr_const(
    repr: Arc<dyn crate::translator::rtyper::rmodel::Repr>,
) -> crate::flowspace::model::ConstValue {
    crate::flowspace::model::ConstValue::Repr(Box::into_raw(Box::new(repr)) as usize)
}

/// Inverse of [`repr_const`]. Clones the `Arc`; the leaked box stays.
pub(crate) fn repr_from_const(
    value: &crate::flowspace::model::ConstValue,
) -> Option<Arc<dyn crate::translator::rtyper::rmodel::Repr>> {
    let crate::flowspace::model::ConstValue::Repr(ptr) = value else {
        return None;
    };
    if *ptr == 0 {
        return None;
    }
    // SAFETY: `ptr` was produced by `repr_const` and never freed.
    Some(unsafe { Arc::clone(&*(*ptr as *const Arc<dyn crate::translator::rtyper::rmodel::Repr>)) })
}

/// RPython `def rtype_hlinvoke(hop)` (rbuiltin.py).
pub fn rtype_hlinvoke(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    use crate::flowspace::argument::CallShape;

    let (_repr, s_repr) = hop.r_s_popfirstarg();
    let Some(const_value) = s_repr.const_() else {
        return Err(TyperError::message(
            "hlinvoke expects a constant repr as first argument",
        ));
    };
    let r_callable = repr_from_const(const_value)
        .ok_or_else(|| TyperError::message("hlinvoke expects a constant repr as first argument"))?;
    // `Repr.get_r_implfunc` returns the implementation function and how
    // many implicit arguments its signature carries. A method repr returns
    // a different function repr; a function repr returns itself.
    let (r_func, nimplicitarg) = r_callable.get_r_implfunc()?;
    let s_callable = r_callable
        .get_s_callable()
        .ok_or_else(|| TyperError::message("hlinvoke callable has no s_callable"))?;
    let nargs = hop.args_s.borrow().len();
    if nargs == 0 {
        return Err(TyperError::message("hlinvoke: missing callable argument"));
    }
    let nbargs = nargs - 1 + nimplicitarg;
    let shape = CallShape {
        shape_cnt: nbargs,
        shape_keys: Vec::new(),
        shape_star: false,
    };
    let s_sigs = r_func.get_s_signatures(&shape)?;
    if s_sigs.len() != 1 {
        return Err(TyperError::message(format!(
            "cannot hlinvoke callable {} with not uniform annotations: {s_sigs:?}",
            r_callable.repr_string()
        )));
    }
    let (sig_args, s_ret) = s_sigs[0].clone();
    let mut rinputs = Vec::with_capacity(sig_args.len());
    for s_obj in &sig_args {
        rinputs.push(hop.rtyper.getrepr(s_obj)?);
    }
    let rresult = hop.rtyper.getrepr(&s_ret)?;
    let sig_args = sig_args[nimplicitarg..].to_vec();
    let rinputs = rinputs.split_off(nimplicitarg);
    let mut new_args_r = vec![Some(Arc::clone(&r_callable))];
    for r_input in rinputs {
        new_args_r.push(Some(r_input));
    }
    {
        let hop_args_r = hop.args_r.borrow();
        for (i, r_new) in new_args_r.iter().enumerate() {
            let left = hop_args_r
                .get(i)
                .and_then(|r| r.as_ref())
                .map(|r| r.lowleveltype().clone());
            let right = r_new.as_ref().map(|r| r.lowleveltype().clone());
            assert!(
                left == right,
                "hlinvoke arg {i} lowleveltype {left:?} != {right:?}"
            );
        }
    }
    *hop.args_r.borrow_mut() = new_args_r;
    let mut new_args_s = vec![SomeValue::PBC(s_callable.clone())];
    new_args_s.extend(sig_args);
    *hop.args_s.borrow_mut() = new_args_s;
    *hop.s_result.borrow_mut() = Some(s_ret.clone());
    {
        let current = hop.r_result.borrow();
        let current_ll = current.as_ref().map(|r| r.lowleveltype().clone());
        assert!(
            current_ll.as_ref() == Some(rresult.lowleveltype()),
            "hlinvoke r_result lowleveltype {current_ll:?} != {:?}",
            rresult.lowleveltype()
        );
    }
    *hop.r_result.borrow_mut() = Some(rresult);
    hop.dispatch()
}

/// RPython `@typer_for(llmemory.offsetof) def rtype_offsetof(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(llmemory.offsetof)
/// def rtype_offsetof(hop):
///     TYPE, field = hop.inputargs(lltype.Void, lltype.Void)
///     hop.exception_cannot_occur()
///     return hop.inputconst(lltype.Signed,
///                           llmemory.offsetof(TYPE.value, field.value))
/// ```
///
/// `offsetof(TYPE, field)` is `FieldOffset(TYPE, field)` — a `Signed`
/// symbolic constant whose concrete byte value is resolved only at code
/// emission (`Signed._contains_value` accepts `AddressOffset`).
#[allow(non_snake_case)]
pub fn rtype_offsetof(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::llmemory;

    let args = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::LowLevelType(&LowLevelType::Void),
    ])?;
    hop.exception_cannot_occur()?;
    // `TYPE.value` — the struct lltype carried on the first Void constant.
    let Hlvalue::Constant(Constant {
        value: ConstValue::LowLevelType(TYPE),
        ..
    }) = &args[0]
    else {
        return Err(TyperError::message(
            "rtype_offsetof: first argument is not a constant TYPE",
        ));
    };
    // `field.value` — the field name carried on the second Void constant.
    let Hlvalue::Constant(Constant {
        value: ConstValue::ByteStr(field),
        ..
    }) = &args[1]
    else {
        return Err(TyperError::message(
            "rtype_offsetof: second argument is not a constant field name",
        ));
    };
    let field = String::from_utf8_lossy(field);
    let offset = llmemory::offsetof(TYPE, &field).map_err(TyperError::message)?;
    let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::AddressOffset(offset))?;
    Ok(Some(Hlvalue::Constant(c)))
}

/// RPython `@typer_for(objectmodel.instantiate) def rtype_instantiate`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(objectmodel.instantiate)
/// def rtype_instantiate(hop, i_nonmovable=None):
///     hop.exception_cannot_occur()
///     s_class = hop.args_s[0]
///     assert isinstance(s_class, annmodel.SomePBC)
///     v_nonmovable, = parse_kwds(hop, (i_nonmovable, None))
///     nonmovable = (i_nonmovable is not None and v_nonmovable.value)
///     if len(s_class.descriptions) != 1:
///         # instantiate() on a variable class
///         if nonmovable:
///             raise TyperError("instantiate(x, nonmovable=True) cannot be used "
///                              "if x is not a constant class")
///         vtypeptr, = hop.inputargs(rclass.get_type_repr(hop.rtyper))
///         r_class = hop.args_r[0]
///         return r_class._instantiate_runtime_class(hop, vtypeptr,
///                                                   hop.r_result.lowleveltype)
///     classdef = s_class.any_description().getuniqueclassdef()
///     return rclass.rtype_new_instance(hop.rtyper, classdef, hop.llops,
///                                      nonmovable=nonmovable)
/// ```
pub fn rtype_instantiate(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::description::DescEntry;
    use crate::flowspace::model::Hlvalue;

    hop.exception_cannot_occur()?;
    let s_class = hop.args_s.borrow()[0].clone();
    let SomeValue::PBC(s_class) = s_class else {
        return Err(TyperError::message(
            "rtype_instantiate: the class argument is not a SomePBC",
        ));
    };
    let i_nonmovable = kwds_i.get("i_nonmovable").copied();
    let kw = parse_kwds(hop, &[(i_nonmovable, None)])?;
    let nonmovable = i_nonmovable.is_some()
        && matches!(
            kw.into_iter().next().flatten(),
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(true),
                ..
            }))
        );
    if s_class.descriptions.len() != 1 {
        if nonmovable {
            return Err(TyperError::message(
                "instantiate(x, nonmovable=True) cannot be used if x is not a constant class",
            ));
        }
        let r_type = crate::translator::rtyper::rclass::get_type_repr(&hop.rtyper)?;
        let args = hop.inputargs(vec![ConvertedTo::Repr(r_type.as_ref())])?;
        let vtypeptr = args
            .into_iter()
            .next()
            .ok_or_else(|| TyperError::message("rtype_instantiate: missing type pointer"))?;
        let r_any = hop
            .args_r
            .borrow()
            .first()
            .cloned()
            .flatten()
            .ok_or_else(|| TyperError::message("rtype_instantiate: missing class repr"))?;
        let raw = std::sync::Arc::into_raw(r_any);
        if unsafe { (*raw).type_id() }
            != std::any::TypeId::of::<crate::translator::rtyper::rpbc::ClassesPBCRepr>()
        {
            let _ = unsafe { std::sync::Arc::from_raw(raw) };
            return Err(TyperError::message(
                "rtype_instantiate: class repr is not a ClassesPBCRepr",
            ));
        }
        let r_class = unsafe {
            std::sync::Arc::from_raw(
                raw as *const () as *const crate::translator::rtyper::rpbc::ClassesPBCRepr,
            )
        };
        let r_instance = hop
            .r_result
            .borrow()
            .as_ref()
            .ok_or_else(|| TyperError::message("rtype_instantiate: r_result missing"))?
            .lowleveltype()
            .clone();
        let v = r_class._instantiate_runtime_class(hop, vtypeptr, &r_instance)?;
        return Ok(Some(v));
    }
    let Some(DescEntry::Class(classdesc)) = s_class.any_description() else {
        return Err(TyperError::message(
            "rtype_instantiate: the class argument's description is not a ClassDesc",
        ));
    };
    let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(classdesc)
        .map_err(|e| TyperError::message(e.to_string()))?;
    let mut llops = hop.llops.borrow_mut();
    let v_instance = crate::translator::rtyper::rclass::rtype_new_instance(
        &hop.rtyper,
        Some(&classdef),
        &mut llops,
        None,
        nonmovable,
    )?;
    Ok(Some(v_instance))
}

/// RPython `@typer_for(OrderedDict)`, `objectmodel.r_dict`, and
/// `objectmodel.r_ordereddict` `def rtype_dict_constructor(...)`
/// (rbuiltin.py).
///
/// `i_force_non_null` and `i_simple_hash_eq` are ignored: when they
/// matter they have already been applied to `hop.r_result`.
pub fn rtype_dict_constructor(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::{ConstValue, Hlvalue};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::lltypesystem::rordereddict::OrderedDictRepr;
    use crate::translator::rtyper::rtyper::GenopResult;

    hop.exception_cannot_occur()?;
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_dict_constructor: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    let r_dict = any_r.downcast_ref::<OrderedDictRepr>().ok_or_else(|| {
        TyperError::message("rtype_dict_constructor: hop.r_result is not an OrderedDictRepr")
    })?;
    let v_result = r_dict.ll_newdict(hop)?;
    if r_dict.base.custom_eq_hash {
        let (r_eqfn, r_hashfn) = r_dict.base.custom_eq_hash_repr.as_ref().ok_or_else(|| {
            TyperError::message("rtype_dict_constructor: custom_eq_hash reprs missing")
        })?;
        let v_eqfn = hop.inputarg(ConvertedTo::Repr(r_eqfn.as_ref()), 0)?;
        let v_hashfn = hop.inputarg(ConvertedTo::Repr(r_hashfn.as_ref()), 1)?;
        let v_dict = v_result.clone().ok_or_else(|| {
            TyperError::message("rtype_dict_constructor: ll_newdict returned no value")
        })?;
        if !matches!(r_eqfn.lowleveltype(), LowLevelType::Void) {
            let cname =
                HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::byte_str("fnkeyeq"))?;
            hop.genop(
                "setfield",
                vec![v_dict.clone(), Hlvalue::Constant(cname), v_eqfn],
                GenopResult::Void,
            );
        }
        if !matches!(r_hashfn.lowleveltype(), LowLevelType::Void) {
            let cname =
                HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::byte_str("fnkeyhash"))?;
            hop.genop(
                "setfield",
                vec![v_dict, Hlvalue::Constant(cname), v_hashfn],
                GenopResult::Void,
            );
        }
    }
    Ok(v_result)
}

/// RPython `@typer_for(lltype.identityhash) def rtype_identity_hash(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.identityhash)
/// def rtype_identity_hash(hop):
///     vlist = hop.inputargs(hop.args_r[0])
///     hop.exception_cannot_occur()
///     return hop.genop('gc_identityhash', vlist, resulttype=lltype.Signed)
/// ```
pub fn rtype_identity_hash(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "gc_identityhash",
        vlist,
        GenopResult::LLType(LowLevelType::Signed),
    ))
}

/// RPython `@typer_for(lltype.runtime_type_info) def rtype_runtime_type_info(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.runtime_type_info)
/// def rtype_runtime_type_info(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     vlist = hop.inputargs(hop.args_r[0])
///     hop.exception_cannot_occur()
///     return hop.genop('runtime_type_info', vlist,
///                      resulttype=hop.r_result.lowleveltype)
/// ```
///
pub fn rtype_runtime_type_info(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    // upstream `assert isinstance(hop.args_r[0], rptr.PtrRepr)`
    if !matches!(r_arg.repr_class_id(), ReprClassId::PtrRepr) {
        return Err(TyperError::message(format!(
            "rtype_runtime_type_info: hop.args_r[0] must be PtrRepr, got {:?}",
            r_arg.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?;
    hop.exception_cannot_occur()?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message("rtype_runtime_type_info: r_result missing".to_string())
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop("runtime_type_info", vlist, GenopResult::LLType(lltype)))
}

/// RPython `@typer_for(lltype.cast_pointer) def rtype_cast_pointer(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.cast_pointer)
/// def rtype_cast_pointer(hop):
///     assert hop.args_s[0].is_constant()
///     assert isinstance(hop.args_r[1], rptr.PtrRepr)
///     v_type, v_input = hop.inputargs(lltype.Void, hop.args_r[1])
///     hop.exception_cannot_occur()
///     return hop.genop('cast_pointer', [v_input],
///                      resulttype=hop.r_result.lowleveltype)
/// ```
///
pub fn rtype_cast_pointer(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeObjectTrait;
    use crate::translator::rtyper::rtyper::GenopResult;

    // upstream `assert hop.args_s[0].is_constant()`
    {
        let args_s = hop.args_s.borrow();
        let s0 = args_s.first().ok_or_else(|| {
            TyperError::message("rtype_cast_pointer: hop.args_s[0] missing".to_string())
        })?;
        if !s0.is_constant() {
            return Err(TyperError::message(
                "rtype_cast_pointer: hop.args_s[0] must be a constant".to_string(),
            ));
        }
    }
    let r_arg1 = arg_repr(hop, 1)?;
    // upstream `assert isinstance(hop.args_r[1], rptr.PtrRepr)`.
    // Pyre also accepts `InstanceRepr`: a typed receiver lifts to the
    // instance lattice rather than `SomePtr`, and the instance flavour
    // of the same downcast emits the identical `cast_pointer` genop
    // (`pairtype(InstanceRepr, InstanceRepr).convert_from_to`,
    // rclass.py).
    if !matches!(
        r_arg1.repr_class_id(),
        ReprClassId::PtrRepr | ReprClassId::InstanceRepr
    ) {
        return Err(TyperError::message(format!(
            "rtype_cast_pointer: hop.args_r[1] must be PtrRepr or InstanceRepr, got {:?}",
            r_arg1.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::Repr(r_arg1.as_ref()),
    ])?;
    hop.exception_cannot_occur()?;
    let v_input = vlist
        .into_iter()
        .nth(1)
        .ok_or_else(|| TyperError::message("rtype_cast_pointer: missing v_input".to_string()))?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| TyperError::message("rtype_cast_pointer: r_result missing".to_string()))?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop("cast_pointer", vec![v_input], GenopResult::LLType(lltype)))
}

/// RPython `_cast_to_Signed` table (rbuiltin.py) — opname that
/// converts a primitive `t` value to lltype.Signed. `None` means
/// "already Signed, no-op". Returns `None` outright for primitives
/// that have no Signed counterpart.
fn cast_to_signed_opname(t: &LowLevelType) -> Option<Option<&'static str>> {
    match t {
        LowLevelType::Signed => Some(None),
        LowLevelType::Bool => Some(Some("cast_bool_to_int")),
        LowLevelType::Char => Some(Some("cast_char_to_int")),
        LowLevelType::UniChar => Some(Some("cast_unichar_to_int")),
        LowLevelType::Float => Some(Some("cast_float_to_int")),
        LowLevelType::Unsigned => Some(Some("cast_uint_to_int")),
        LowLevelType::SignedLongLong => Some(Some("truncate_longlong_to_int")),
        _ => None,
    }
}

/// RPython `_cast_from_Signed` table (rbuiltin.py) — opname
/// that converts an lltype.Signed value to primitive `t`. `None` means
/// "already Signed, no-op". Returns `None` outright for primitives
/// without a Signed source path.
fn cast_from_signed_opname(t: &LowLevelType) -> Option<Option<&'static str>> {
    match t {
        LowLevelType::Signed => Some(None),
        LowLevelType::Char => Some(Some("cast_int_to_char")),
        LowLevelType::UniChar => Some(Some("cast_int_to_unichar")),
        LowLevelType::Float => Some(Some("cast_int_to_float")),
        LowLevelType::Unsigned => Some(Some("cast_int_to_uint")),
        LowLevelType::SignedLongLong => Some(Some("cast_int_to_longlong")),
        _ => None,
    }
}

/// RPython `gen_cast(llops, TGT, v_value)` (rbuiltin.py).
/// Primitive → primitive arm routes through `Signed` using the
/// `_cast_to_Signed` + `_cast_from_Signed` tables. The Ptr↔Ptr,
/// Ptr↔Address, and Address↔Primitive arms upstream require lltype
/// machinery (`isinstance(TGT.TO, lltype.OpaqueType)`,
/// `cast_adr_to_int`) whose Rust ports are not yet on every call path;
/// each falls through to a fail-loud `TyperError` carrying the type
/// pair so a future port can flip the arm without retrofitting the
/// surface.
pub fn gen_cast(
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    tgt: &LowLevelType,
    v_value: Hlvalue,
) -> Result<Hlvalue, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;
    let orig = hlvalue_concretetype_for_gen_cast(&v_value)?;
    // upstream `if ORIG == TGT: return v_value`
    if &orig == tgt {
        return Ok(v_value);
    }
    // upstream `if isinstance(TGT, Primitive) and isinstance(ORIG,
    // Primitive)` — primitive→primitive routes via Signed.
    if let (Some(to_signed), Some(from_signed)) =
        (cast_to_signed_opname(&orig), cast_from_signed_opname(tgt))
    {
        let mut current = v_value;
        if let Some(op) = to_signed {
            current = llops
                .genop(op, vec![current], GenopResult::LLType(LowLevelType::Signed))
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message(format!("gen_cast: {op} unexpectedly returned void"))
                })?;
        }
        if let Some(op) = from_signed {
            current = llops
                .genop(op, vec![current], GenopResult::LLType(tgt.clone()))
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message(format!("gen_cast: {op} unexpectedly returned void"))
                })?;
        }
        return Ok(current);
    }
    // rbuiltin.py:512 `elif ORIG is Signed and TGT is Bool`
    if matches!(&orig, LowLevelType::Signed) && matches!(tgt, LowLevelType::Bool) {
        return llops
            .genop(
                "int_is_true",
                vec![v_value],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .map(Hlvalue::Variable)
            .ok_or_else(|| TyperError::message("gen_cast: int_is_true returned void".to_string()));
    }
    // rbuiltin.py:515 `else: return llops.genop('cast_primitive', ...)`
    // Both ORIG and TGT are primitives but not in the tables above.
    if orig.is_primitive() && tgt.is_primitive() {
        return llops
            .genop(
                "cast_primitive",
                vec![v_value],
                GenopResult::LLType(tgt.clone()),
            )
            .map(Hlvalue::Variable)
            .ok_or_else(|| {
                TyperError::message("gen_cast: cast_primitive returned void".to_string())
            });
    }
    // upstream `elif isinstance(TGT, lltype.Ptr): ...`
    if let LowLevelType::Ptr(tgt_ptr) = tgt {
        use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
        // Ptr->Ptr: cast_opaque_ptr when either side targets OpaqueType,
        // otherwise cast_pointer.
        if let LowLevelType::Ptr(orig_ptr) = &orig {
            let opname = if matches!(tgt_ptr.TO, PtrTarget::Opaque(_))
                || matches!(orig_ptr.TO, PtrTarget::Opaque(_))
            {
                "cast_opaque_ptr"
            } else {
                "cast_pointer"
            };
            return llops
                .genop(opname, vec![v_value], GenopResult::LLType(tgt.clone()))
                .map(Hlvalue::Variable)
                .ok_or_else(|| TyperError::message(format!("gen_cast: {opname} returned void")));
        }
        // Address->Ptr (`elif ORIG == llmemory.Address`).
        if matches!(&orig, LowLevelType::Address) {
            return llops
                .genop(
                    "cast_adr_to_ptr",
                    vec![v_value],
                    GenopResult::LLType(tgt.clone()),
                )
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message("gen_cast: cast_adr_to_ptr returned void".to_string())
                });
        }
        // rbuiltin.py:524-527 Primitive->Ptr: cast through Signed, then cast_int_to_ptr.
        if orig.is_primitive() {
            let v_signed = gen_cast(llops, &LowLevelType::Signed, v_value)?;
            return llops
                .genop(
                    "cast_int_to_ptr",
                    vec![v_signed],
                    GenopResult::LLType(tgt.clone()),
                )
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message("gen_cast: cast_int_to_ptr returned void".to_string())
                });
        }
    }
    // upstream `elif TGT == llmemory.Address and isinstance(ORIG, lltype.Ptr)`
    if matches!(tgt, LowLevelType::Address) && matches!(&orig, LowLevelType::Ptr(_)) {
        return llops
            .genop(
                "cast_ptr_to_adr",
                vec![v_value],
                GenopResult::LLType(tgt.clone()),
            )
            .map(Hlvalue::Variable)
            .ok_or_else(|| {
                TyperError::message("gen_cast: cast_ptr_to_adr returned void".to_string())
            });
    }
    // rbuiltin.py:528-539 `elif isinstance(TGT, Primitive):`
    // Ptr->Primitive and Address->Primitive route through Signed.
    if tgt.is_primitive() {
        if matches!(&orig, LowLevelType::Ptr(_)) {
            let v_signed = llops
                .genop(
                    "cast_ptr_to_int",
                    vec![v_value],
                    GenopResult::LLType(LowLevelType::Signed),
                )
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message("gen_cast: cast_ptr_to_int returned void".to_string())
                })?;
            return gen_cast(llops, tgt, v_signed);
        }
        if matches!(&orig, LowLevelType::Address) {
            let v_signed = llops
                .genop(
                    "cast_adr_to_int",
                    vec![v_value],
                    GenopResult::LLType(LowLevelType::Signed),
                )
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message("gen_cast: cast_adr_to_int returned void".to_string())
                })?;
            return gen_cast(llops, tgt, v_signed);
        }
    }
    Err(TyperError::message(format!(
        "gen_cast: don't know how to cast from {orig:?} to {tgt:?}"
    )))
}

/// `Hlvalue` concretetype extractor for [`gen_cast`]. Fails loud if
/// either Variable or Constant lacks a concretetype, matching
/// upstream's `v_value.concretetype` direct access semantics
/// (`rbuiltin.py`).
fn hlvalue_concretetype_for_gen_cast(value: &Hlvalue) -> Result<LowLevelType, TyperError> {
    match value {
        Hlvalue::Variable(v) => v.concretetype().ok_or_else(|| {
            TyperError::message("gen_cast: source Variable missing concretetype".to_string())
        }),
        Hlvalue::Constant(c) => c.concretetype.clone().ok_or_else(|| {
            TyperError::message("gen_cast: source Constant missing concretetype".to_string())
        }),
    }
}

/// RPython `@typer_for(lltype.cast_primitive) def rtype_cast_primitive(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.cast_primitive)
/// def rtype_cast_primitive(hop):
///     assert hop.args_s[0].is_constant()
///     TGT = hop.args_s[0].const
///     v_type, v_value = hop.inputargs(lltype.Void, hop.args_r[1])
///     hop.exception_cannot_occur()
///     return gen_cast(hop.llops, TGT, v_value)
/// ```
pub fn rtype_cast_primitive(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeObjectTrait;

    {
        let args_s = hop.args_s.borrow();
        let s0 = args_s.first().ok_or_else(|| {
            TyperError::message("rtype_cast_primitive: hop.args_s[0] missing".to_string())
        })?;
        if !s0.is_constant() {
            return Err(TyperError::message(
                "rtype_cast_primitive: hop.args_s[0] must be a constant".to_string(),
            ));
        }
    }
    let r_arg1 = arg_repr(hop, 1)?;
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::Repr(r_arg1.as_ref()),
    ])?;
    hop.exception_cannot_occur()?;
    let v_value = vlist
        .into_iter()
        .nth(1)
        .ok_or_else(|| TyperError::message("rtype_cast_primitive: missing v_value".to_string()))?;
    // rbuiltin.py `TGT = hop.args_s[0].const`
    let tgt = {
        let args_s = hop.args_s.borrow();
        let s0 = args_s.first().ok_or_else(|| {
            TyperError::message("rtype_cast_primitive: hop.args_s[0] missing".to_string())
        })?;
        let cv = s0.const_().ok_or_else(|| {
            TyperError::message("rtype_cast_primitive: hop.args_s[0] not constant".to_string())
        })?;
        match cv {
            ConstValue::LowLevelType(t) => Ok((**t).clone()),
            _ => Err(TyperError::message(
                "rtype_cast_primitive: TGT not a LowLevelType constant".to_string(),
            )),
        }?
    };
    let result = {
        let mut llops = hop.llops.borrow_mut();
        gen_cast(&mut llops, &tgt, v_value)?
    };
    Ok(Some(result))
}

/// RPython `@typer_for(llmemory.cast_ptr_to_adr) def rtype_cast_ptr_to_adr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(llmemory.cast_ptr_to_adr)
/// def rtype_cast_ptr_to_adr(hop):
///     vlist = hop.inputargs(hop.args_r[0])
///     assert isinstance(vlist[0].concretetype, lltype.Ptr)
///     hop.exception_cannot_occur()
///     return hop.genop('cast_ptr_to_adr', vlist,
///                      resulttype=llmemory.Address)
/// ```
///
pub fn rtype_cast_ptr_to_adr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg0 = arg_repr(hop, 0)?;
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg0.as_ref())])?;
    // upstream `assert isinstance(vlist[0].concretetype, lltype.Ptr)`
    {
        let v0 = vlist.first().ok_or_else(|| {
            TyperError::message("rtype_cast_ptr_to_adr: inputargs returned empty list".to_string())
        })?;
        let concrete = match v0 {
            Hlvalue::Variable(var) => var.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if !matches!(concrete, Some(LowLevelType::Ptr(_))) {
            return Err(TyperError::message(format!(
                "rtype_cast_ptr_to_adr: vlist[0].concretetype must be Ptr(...), got {concrete:?}"
            )));
        }
    }
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_ptr_to_adr",
        vlist,
        GenopResult::LLType(LowLevelType::Address),
    ))
}

/// RPython `@typer_for(getattr(object.__init__, 'im_func', object.__init__))`
/// `def rtype_object__init__(hop)` (rbuiltin.py).
///
/// ```python
/// @typer_for(getattr(object.__init__, 'im_func', object.__init__))
/// def rtype_object__init__(hop):
///     hop.exception_cannot_occur()
/// ```
///
/// Trivial body — only declares the call cannot raise; implicit None
/// return.  Upstream's `getattr(..., 'im_func', ...)` unwraps the
/// Python-2 bound-method `__init__` descriptor; pyre uses a single
/// `"object.__init__"` qualname key in HOST_ENV instead.
#[allow(non_snake_case)]
pub fn rtype_object__init__(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    Ok(None)
}

/// RPython `@typer_for(llmemory.cast_int_to_adr) def rtype_cast_int_to_adr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(llmemory.cast_int_to_adr)
/// def rtype_cast_int_to_adr(hop):
///     v_input, = hop.inputargs(lltype.Signed)
///     hop.exception_cannot_occur()
///     return hop.genop('cast_int_to_adr', [v_input],
///                      resulttype=llmemory.Address)
/// ```
pub fn rtype_cast_int_to_adr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_int_to_adr",
        vlist,
        GenopResult::LLType(LowLevelType::Address),
    ))
}

/// rbuiltin.py:659-665
pub fn rtype_cast_adr_to_ptr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rtyper::GenopResult;

    let r0 = arg_repr(hop, 0)?;
    // rbuiltin.py rtype_cast_adr_to_ptr `assert isinstance(hop.args_r[0], AddressRepr)`
    if r0.repr_class_id() != ReprClassId::AddressRepr {
        return Err(TyperError::message(
            "cast_adr_to_ptr: args_r[0] must be AddressRepr",
        ));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r0.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Void),
    ])?;

    let target_type = {
        let args_s = hop.args_s.borrow();
        let s1 = args_s
            .get(1)
            .ok_or_else(|| TyperError::message("cast_adr_to_ptr: missing TYPE arg".to_string()))?;
        let cv = s1.const_().ok_or_else(|| {
            TyperError::message("cast_adr_to_ptr: TYPE arg not constant".to_string())
        })?;
        match cv {
            ConstValue::LowLevelType(t) => Ok((**t).clone()),
            _ => Err(TyperError::message(
                "cast_adr_to_ptr: TYPE arg not a LowLevelType constant".to_string(),
            )),
        }?
    };

    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_adr_to_ptr",
        vec![vlist[0].clone()],
        GenopResult::LLType(target_type),
    ))
}

/// rbuiltin.py:667-678
pub fn rtype_cast_adr_to_int(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rtyper::GenopResult;

    let r0 = arg_repr(hop, 0)?;
    // rbuiltin.py rtype_cast_adr_to_int `assert isinstance(hop.args_r[0], AddressRepr)`
    if r0.repr_class_id() != ReprClassId::AddressRepr {
        return Err(TyperError::message(
            "cast_adr_to_int: args_r[0] must be AddressRepr",
        ));
    }
    let adr = hop.inputarg(ConvertedTo::Repr(r0.as_ref()), 0)?;

    // rbuiltin.py `mode = hop.args_s[1].const`
    let mode = {
        let args_s = hop.args_s.borrow();
        if args_s.len() == 1 {
            "emulated".to_string()
        } else {
            let s1 = args_s
                .get(1)
                .ok_or_else(|| TyperError::message("cast_adr_to_int: missing mode arg"))?;
            let cv = s1
                .const_()
                .ok_or_else(|| TyperError::message("cast_adr_to_int: mode arg must be constant"))?;
            match cv {
                ConstValue::ByteStr(b) => String::from_utf8(b.clone())
                    .map_err(|_| TyperError::message("cast_adr_to_int: mode not valid UTF-8"))?,
                ConstValue::UniStr(s) => s.clone(),
                _ => {
                    return Err(TyperError::message(
                        "cast_adr_to_int: mode must be a string constant",
                    ));
                }
            }
        }
    };

    let c_mode = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::ByteStr(mode.into_bytes()),
        LowLevelType::Void,
    ));
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_adr_to_int",
        vec![adr, c_mode],
        GenopResult::LLType(LowLevelType::Signed),
    ))
}

/// RPython `@typer_for(llmemory.raw_malloc) def rtype_raw_malloc(hop, i_zero=None)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(llmemory.raw_malloc)
/// def rtype_raw_malloc(hop, i_zero=None):
///     v_size = hop.inputarg(lltype.Signed, arg=0)
///     v_zero, = parse_kwds(hop, (i_zero, None))
///     if v_zero is None:
///         v_zero = hop.inputconst(lltype.Bool, False)
///     hop.exception_cannot_occur()
///     return hop.genop('raw_malloc', [v_size, v_zero],
///                      resulttype=llmemory.Address)
/// ```
pub fn rtype_raw_malloc(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rtyper::GenopResult;

    let i_zero = kwds_i.get("i_zero").copied();
    // upstream `v_size = hop.inputarg(lltype.Signed, arg=0)` — fetch the
    // positional size argument as Signed before parse_kwds truncates the
    // hop's args_v tail.
    let v_size = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 0)?;
    let kw = parse_kwds(hop, &[(i_zero, None)])?;
    let v_zero = match kw.into_iter().next().flatten() {
        Some(v) => v,
        None => Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(false),
            LowLevelType::Bool,
        )),
    };
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "raw_malloc",
        vec![v_size, v_zero],
        GenopResult::LLType(LowLevelType::Address),
    ))
}

/// RPython `@typer_for(llmemory.raw_malloc_usage) def rtype_raw_malloc_usage(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(llmemory.raw_malloc_usage)
/// def rtype_raw_malloc_usage(hop):
///     v_size, = hop.inputargs(lltype.Signed)
///     hop.exception_cannot_occur()
///     return hop.genop('raw_malloc_usage', [v_size], resulttype=lltype.Signed)
/// ```
pub fn rtype_raw_malloc_usage(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "raw_malloc_usage",
        vlist,
        GenopResult::LLType(LowLevelType::Signed),
    ))
}

/// rbuiltin.py:593-600
pub fn rtype_raw_free(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::rtyper::GenopResult;

    let args_s = hop.args_s.borrow();
    if let Some(SomeValue::Address(s_addr)) = args_s.first()
        && s_addr.is_null_address()
    {
        return Err(TyperError::message(
            "raw_free(x) where x is the constant NULL",
        ));
    }
    drop(args_s);

    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Address)])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop("raw_free", vlist, GenopResult::Void))
}

/// rbuiltin.py:602-609
pub fn rtype_raw_memcopy(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::rtyper::GenopResult;

    let args_s = hop.args_s.borrow();
    for s_addr in args_s.iter().take(2) {
        if let SomeValue::Address(s) = s_addr
            && s.is_null_address()
        {
            return Err(TyperError::message("raw_memcopy() with a constant NULL"));
        }
    }
    drop(args_s);

    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Address),
        ConvertedTo::LowLevelType(&LowLevelType::Address),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop("raw_memcopy", vlist, GenopResult::Void))
}

/// rbuiltin.py:611-618
pub fn rtype_raw_memclear(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::rtyper::GenopResult;

    let args_s = hop.args_s.borrow();
    if let Some(SomeValue::Address(s_addr)) = args_s.first()
        && s_addr.is_null_address()
    {
        return Err(TyperError::message(
            "raw_memclear(x, n) where x is the constant NULL",
        ));
    }
    drop(args_s);

    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Address),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop("raw_memclear", vlist, GenopResult::Void))
}

/// rbuiltin.py:753/759/769/777 `assert hop.rtyper.getconfig().translation.rweakref`
fn assert_rweakref(hop: &HighLevelOp) -> Result<(), TyperError> {
    let rweakref = hop
        .rtyper
        .getconfig()
        .map(|c| c.translation.rweakref)
        .unwrap_or(true);
    if !rweakref {
        return Err(TyperError::message(
            "weakref operation requires translation.rweakref=True",
        ));
    }
    Ok(())
}

/// rbuiltin.py rtype_weakref_create
pub fn rtype_weakref_create(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::WEAKREF_PTR;
    use crate::translator::rtyper::rtyper::GenopResult;
    use crate::translator::rtyper::rweakref::as_base_weakref_repr;

    let r0 = arg_repr(hop, 0)?;
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r0.as_ref())])?;
    hop.exception_cannot_occur()?;
    let r_result = hop.r_result.borrow();
    let base_weakref = r_result
        .as_ref()
        .and_then(|r| as_base_weakref_repr(r.as_ref()));
    if let Some(bwr) = base_weakref {
        // rbuiltin.py:751-752 high-level path via BaseWeakRefRepr._weakref_create
        bwr.weakref_create(hop, vlist.into_iter().next().unwrap())
    } else {
        // rbuiltin.py:754-757 low-level path
        assert_rweakref(hop)?;
        Ok(hop.genop(
            "weakref_create",
            vlist,
            GenopResult::LLType(WEAKREF_PTR.clone()),
        ))
    }
}

/// rbuiltin.py rtype_weakref_deref
pub fn rtype_weakref_deref(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::WEAKREF_PTR;
    use crate::translator::rtyper::rtyper::GenopResult;

    assert_rweakref(hop)?;
    let r1 = arg_repr(hop, 1)?;
    // rbuiltin.py:762 `assert v_wref.concretetype == llmemory.WeakRefPtr`
    if r1.lowleveltype() != &*WEAKREF_PTR {
        return Err(TyperError::message(
            "weakref_deref: args_r[1] concretetype must be WeakRefPtr",
        ));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::Repr(r1.as_ref()),
    ])?;

    let target_type = {
        let args_s = hop.args_s.borrow();
        let s0 = args_s
            .first()
            .ok_or_else(|| TyperError::message("weakref_deref: missing PTRTYPE arg"))?;
        let cv = s0
            .const_()
            .ok_or_else(|| TyperError::message("weakref_deref: PTRTYPE arg not constant"))?;
        match cv {
            ConstValue::LowLevelType(t) => Ok((**t).clone()),
            _ => Err(TyperError::message(
                "weakref_deref: PTRTYPE arg not a LowLevelType constant",
            )),
        }?
    };

    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "weakref_deref",
        vec![vlist[1].clone()],
        GenopResult::LLType(target_type),
    ))
}

/// rbuiltin.py rtype_cast_ptr_to_weakrefptr
pub fn rtype_cast_ptr_to_weakrefptr(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::WEAKREF_PTR;
    use crate::translator::rtyper::rtyper::GenopResult;

    assert_rweakref(hop)?;
    let r0 = arg_repr(hop, 0)?;
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r0.as_ref())])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_ptr_to_weakrefptr",
        vlist,
        GenopResult::LLType(WEAKREF_PTR.clone()),
    ))
}

/// rbuiltin.py rtype_cast_weakrefptr_to_ptr
pub fn rtype_cast_weakrefptr_to_ptr(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::lltype::WEAKREF_PTR;
    use crate::translator::rtyper::rtyper::GenopResult;

    assert_rweakref(hop)?;
    let r1 = arg_repr(hop, 1)?;
    // rbuiltin.py:780 `assert v_wref.concretetype == llmemory.WeakRefPtr`
    if r1.lowleveltype() != &*WEAKREF_PTR {
        return Err(TyperError::message(
            "cast_weakrefptr_to_ptr: args_r[1] concretetype must be WeakRefPtr",
        ));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::Repr(r1.as_ref()),
    ])?;

    let target_type = {
        let args_s = hop.args_s.borrow();
        let s0 = args_s
            .first()
            .ok_or_else(|| TyperError::message("cast_weakrefptr_to_ptr: missing PTRTYPE arg"))?;
        let cv = s0.const_().ok_or_else(|| {
            TyperError::message("cast_weakrefptr_to_ptr: PTRTYPE arg not constant")
        })?;
        match cv {
            ConstValue::LowLevelType(t) => Ok((**t).clone()),
            _ => Err(TyperError::message(
                "cast_weakrefptr_to_ptr: PTRTYPE arg not a LowLevelType constant",
            )),
        }?
    };

    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_weakrefptr_to_ptr",
        vec![vlist[1].clone()],
        GenopResult::LLType(target_type),
    ))
}

/// RPython `@typer_for(lltype.malloc) def rtype_malloc(hop, i_flavor=None, ...)`
/// (rbuiltin.py).
///
/// Six keyword-index parameters: `flavor`, `immortal`, `zero`,
/// `track_allocation`, `add_memory_pressure`, `nonmovable`.  Each
/// supplied keyword extends the `flags` dict that becomes the second
/// `genop` argument.  `nb_args == 2` routes through `malloc_varsize`
/// with a second Signed `size` operand.
///
/// Upstream `nonmovable` stores the Hlvalue itself
/// (`flags['nonmovable'] = v_nonmovable`, rbuiltin.py), unlike every
/// other arm which reads `.value`.  `ConstValue::Dict` carries
/// `ConstValue` payloads only and cannot embed an `Hlvalue`, so the
/// line-by-line port is blocked on extending `ConstValue` with an
/// Hlvalue-bearing variant (a separate epic).  Production callers do
/// not pass `nonmovable=`; the kwarg arm surfaces a `TyperError`
/// rather than silently substitute a wrong-shape payload such as
/// `ConstValue::Bool(true)`, which would collapse the wrapped value
/// and lose the structural distinction between a Constant and a
/// Variable carrier.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn rtype_malloc(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rmodel::impossible_repr;
    use crate::translator::rtyper::rtyper::GenopResult;

    use crate::annotator::model::SomeObjectTrait;

    let i_flavor = kwds_i.get("i_flavor").copied();
    let i_immortal = kwds_i.get("i_immortal").copied();
    let i_zero = kwds_i.get("i_zero").copied();
    let i_track_allocation = kwds_i.get("i_track_allocation").copied();
    let i_add_memory_pressure = kwds_i.get("i_add_memory_pressure").copied();
    let i_nonmovable = kwds_i.get("i_nonmovable").copied();

    // upstream `assert hop.args_s[0].is_constant()` (rbuiltin.py:352)
    {
        let args_s = hop.args_s.borrow();
        let s0 = args_s.first().ok_or_else(|| {
            TyperError::message("rtype_malloc: hop.args_s[0] missing".to_string())
        })?;
        if !s0.is_constant() {
            return Err(TyperError::message(
                "rtype_malloc: hop.args_s[0] must be a constant type marker".to_string(),
            ));
        }
    }
    // upstream: vlist = [hop.inputarg(lltype.Void, arg=0)]
    let v_first = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Void), 0)?;
    let mut vlist = vec![v_first];
    let mut opname = String::from("malloc");

    let _ = i_immortal; // upstream `i_immortal` is parse_kwds-only, never read from `flags`
    // upstream `(i_flavor, lltype.Void)` — `impossible_repr` is the
    // RPython `r_void = VoidRepr()` singleton (rmodel.py), the
    // Repr counterpart of `lltype.Void`; routing the flavor slot
    // through it pins the coercion to Void independent of `args_r[i]`.
    let r_void: Arc<dyn Repr> = impossible_repr();
    let kw = parse_kwds(
        hop,
        &[
            (i_flavor, Some(r_void.clone())),
            (i_immortal, None),
            (i_zero, None),
            (i_track_allocation, None),
            (i_add_memory_pressure, None),
            (i_nonmovable, None),
        ],
    )?;
    let v_flavor = &kw[0];
    let v_zero = &kw[2];
    let v_track_allocation = &kw[3];
    let v_add_memory_pressure = &kw[4];
    let v_nonmovable = &kw[5];

    // upstream `v_X.value` reads `.value` from the Hlvalue.  A
    // Constant carries `.value`; a Variable raises AttributeError.
    // Surface a TyperError matching the AttributeError instead of
    // silently dropping the flag — the former is observably the same
    // failure mode upstream, the latter is a silent shape divergence.
    let constant_value = |hl: &Hlvalue, slot: &str| -> Result<ConstValue, TyperError> {
        match hl {
            Hlvalue::Constant(c) => Ok(c.value.clone()),
            Hlvalue::Variable(_) => Err(TyperError::message(format!(
                "rtype_malloc: `{slot}=` keyword value must be a constant — \
                 upstream `v_{slot}.value` access raises AttributeError on a Variable"
            ))),
        }
    };

    let mut flags_map = HashMap::new();
    // upstream: flags = {'flavor': 'gc'}
    flags_map.insert(
        ConstValue::ByteStr(b"flavor".to_vec()),
        ConstValue::ByteStr(b"gc".to_vec()),
    );
    if let Some(v) = v_flavor.as_ref() {
        flags_map.insert(
            ConstValue::ByteStr(b"flavor".to_vec()),
            constant_value(v, "flavor")?,
        );
    }
    if i_zero.is_some() {
        let v = v_zero.as_ref().ok_or_else(|| {
            TyperError::message("rtype_malloc: parse_kwds returned None for `zero` slot")
        })?;
        flags_map.insert(
            ConstValue::ByteStr(b"zero".to_vec()),
            constant_value(v, "zero")?,
        );
    }
    if i_track_allocation.is_some() {
        let v = v_track_allocation.as_ref().ok_or_else(|| {
            TyperError::message(
                "rtype_malloc: parse_kwds returned None for `track_allocation` slot",
            )
        })?;
        flags_map.insert(
            ConstValue::ByteStr(b"track_allocation".to_vec()),
            constant_value(v, "track_allocation")?,
        );
    }
    if i_add_memory_pressure.is_some() {
        let v = v_add_memory_pressure.as_ref().ok_or_else(|| {
            TyperError::message(
                "rtype_malloc: parse_kwds returned None for `add_memory_pressure` slot",
            )
        })?;
        flags_map.insert(
            ConstValue::ByteStr(b"add_memory_pressure".to_vec()),
            constant_value(v, "add_memory_pressure")?,
        );
    }
    if i_nonmovable.is_some() {
        // upstream `flags['nonmovable'] = v_nonmovable` (rbuiltin.py)
        // stores the Hlvalue itself — the only flag arm that does NOT
        // unwrap via `.value`.  `ConstValue::Dict` is keyed and valued
        // over `ConstValue` and cannot carry an `Hlvalue` payload, so
        // an exact-parity port requires extending `ConstValue` with an
        // Hlvalue-bearing variant (a separate epic).  Until then,
        // surface a `TyperError` rather than silently substituting a
        // wrong-shape payload (`ConstValue::Bool(true)` collapses the
        // wrapped value and loses the structural distinction between
        // a constant and a variable carrier).  Production callers do
        // not pass `nonmovable=`, so this never fires today; when one
        // does, the error names the missing carrier.
        let _ = v_nonmovable;
        return Err(TyperError::message(
            "rtype_malloc: `nonmovable=` keyword cannot be ported line-by-line — \
             upstream stores the Hlvalue in the flags dict but `ConstValue::Dict` \
             carries `ConstValue` payloads only.  Extending `ConstValue` with an \
             Hlvalue-bearing variant is a separate epic; until that lands, refuse \
             to silently substitute a wrong-shape payload."
                .to_string(),
        ));
    }
    let cflags = HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::Dict(flags_map))?;
    vlist.push(Hlvalue::Constant(cflags));

    let nb_args = hop.nb_args();
    if !(1..=2).contains(&nb_args) {
        return Err(TyperError::message(format!(
            "rtype_malloc: assertion `1 <= nb_args <= 2` failed (nb_args={nb_args})"
        )));
    }
    if nb_args == 2 {
        let v_size = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 1)?;
        vlist.push(v_size);
        opname.push_str("_varsize");
    }

    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| TyperError::message("rtype_malloc: r_result missing".to_string()))?
            .lowleveltype()
            .clone()
    };
    // upstream rbuiltin.py:383-384
    hop.has_implicit_exception("MemoryError");
    hop.exception_is_here()?;
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(lltype)))
}

/// RPython `@typer_for(lltype.free) def rtype_free(hop, i_flavor, i_track_allocation=None)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.free)
/// def rtype_free(hop, i_flavor, i_track_allocation=None):
///     vlist = [hop.inputarg(hop.args_r[0], arg=0)]
///     v_flavor, v_track_allocation = parse_kwds(hop,
///         (i_flavor, lltype.Void),
///         (i_track_allocation, None))
///     #
///     assert v_flavor is not None and v_flavor.value == 'raw'
///     flags = {'flavor': 'raw'}
///     if i_track_allocation is not None:
///         flags['track_allocation'] = v_track_allocation.value
///     vlist.append(hop.inputconst(lltype.Void, flags))
///     #
///     hop.exception_cannot_occur()
///     hop.genop('free', vlist)
/// ```
///
/// `i_flavor` is the required keyword index; absent or non-`'raw'`
/// flavor surfaces a `TyperError` matching the upstream assertion.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn rtype_free(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rmodel::impossible_repr;
    use crate::translator::rtyper::rtyper::GenopResult;

    let i_flavor = kwds_i.get("i_flavor").copied().ok_or_else(|| {
        TyperError::message(
            "rtype_free: required `flavor` keyword not present in kwds_i".to_string(),
        )
    })?;
    let i_track_allocation = kwds_i.get("i_track_allocation").copied();
    let r_arg0 = arg_repr(hop, 0)?;
    let v_first = hop.inputarg(r_arg0.as_ref(), 0)?;
    let mut vlist = vec![v_first];
    // upstream `(i_flavor, lltype.Void)` — see `rtype_malloc` comment.
    let r_void: Arc<dyn Repr> = impossible_repr();
    let kw = parse_kwds(
        hop,
        &[(Some(i_flavor), Some(r_void)), (i_track_allocation, None)],
    )?;
    let v_flavor = kw[0].as_ref().ok_or_else(|| {
        TyperError::message(
            "rtype_free: parse_kwds returned None for required `flavor` slot".to_string(),
        )
    })?;
    let flavor_value = match v_flavor {
        Hlvalue::Constant(c) => &c.value,
        Hlvalue::Variable(_) => {
            return Err(TyperError::message(
                "rtype_free: `flavor` keyword must be a constant".to_string(),
            ));
        }
    };
    let flavor_is_raw = matches!(flavor_value, ConstValue::ByteStr(b) if b == b"raw")
        || matches!(flavor_value, ConstValue::UniStr(s) if s == "raw");
    if !flavor_is_raw {
        return Err(TyperError::message(format!(
            "rtype_free: assertion `v_flavor.value == 'raw'` failed (got {flavor_value:?})"
        )));
    }
    let mut flags_map = HashMap::new();
    flags_map.insert(
        ConstValue::ByteStr(b"flavor".to_vec()),
        ConstValue::ByteStr(b"raw".to_vec()),
    );
    if i_track_allocation.is_some() {
        let v = kw.get(1).and_then(|slot| slot.as_ref()).ok_or_else(|| {
            TyperError::message(
                "rtype_free: parse_kwds returned None for `track_allocation` slot".to_string(),
            )
        })?;
        match v {
            Hlvalue::Constant(c) => {
                flags_map.insert(
                    ConstValue::ByteStr(b"track_allocation".to_vec()),
                    c.value.clone(),
                );
            }
            Hlvalue::Variable(_) => {
                return Err(TyperError::message(
                    "rtype_free: `track_allocation=` keyword value must be a constant — \
                     upstream `v_track_allocation.value` access raises AttributeError on a Variable"
                        .to_string(),
                ));
            }
        }
    }
    let cflags = HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::Dict(flags_map))?;
    vlist.push(Hlvalue::Constant(cflags));
    hop.exception_cannot_occur()?;
    Ok(hop.genop("free", vlist, GenopResult::Void))
}

/// RPython `@typer_for(objectmodel.free_non_gc_object) def rtype_free_non_gc_object(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(objectmodel.free_non_gc_object)
/// def rtype_free_non_gc_object(hop):
///     hop.exception_cannot_occur()
///     vinst, = hop.inputargs(hop.args_r[0])
///     flavor = hop.args_r[0].gcflavor
///     assert flavor != 'gc'
///     flags = {'flavor': flavor}
///     cflags = hop.inputconst(lltype.Void, flags)
///     return hop.genop('free', [vinst, cflags])
/// ```
///
/// The `flags` Python dict is lowered to `ConstValue::Dict` keyed on
/// the byte-string `"flavor"` with the `Flavor::llflavor()` byte-string
/// as value, matching the runtime shape the `free` llop-handler reads.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn rtype_free_non_gc_object(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rtyper::GenopResult;

    hop.exception_cannot_occur()?;
    let r_arg0 = arg_repr(hop, 0)?;
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg0.as_ref())])?;
    let vinst = vlist.into_iter().next().ok_or_else(|| {
        TyperError::message("rtype_free_non_gc_object: missing vinst".to_string())
    })?;
    // upstream: flavor = hop.args_r[0].gcflavor
    let flavor_str = r_arg0.gc_flavor_str().ok_or_else(|| {
        TyperError::message(
            "rtype_free_non_gc_object: args_r[0] has no gcflavor (not an InstanceRepr-shaped repr)"
                .to_string(),
        )
    })?;
    // upstream: assert flavor != 'gc'
    if flavor_str == "gc" {
        return Err(TyperError::message(
            "rtype_free_non_gc_object: assertion `flavor != 'gc'` failed".to_string(),
        ));
    }
    let mut flags_map = HashMap::new();
    flags_map.insert(
        ConstValue::ByteStr(b"flavor".to_vec()),
        ConstValue::ByteStr(flavor_str.as_bytes().to_vec()),
    );
    let cflags = HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::Dict(flags_map))?;
    Ok(hop.genop(
        "free",
        vec![vinst, Hlvalue::Constant(cflags)],
        GenopResult::Void,
    ))
}

/// RPython `@typer_for(lltype.cast_opaque_ptr) def rtype_cast_opaque_ptr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.cast_opaque_ptr)
/// def rtype_cast_opaque_ptr(hop):
///     assert hop.args_s[0].is_constant()
///     assert isinstance(hop.args_r[1], rptr.PtrRepr)
///     v_type, v_input = hop.inputargs(lltype.Void, hop.args_r[1])
///     hop.exception_cannot_occur()
///     return hop.genop('cast_opaque_ptr', [v_input],
///                      resulttype=hop.r_result.lowleveltype)
/// ```
pub fn rtype_cast_opaque_ptr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeObjectTrait;
    use crate::translator::rtyper::rtyper::GenopResult;

    // upstream `assert hop.args_s[0].is_constant()`
    {
        let args_s = hop.args_s.borrow();
        let s0 = args_s.first().ok_or_else(|| {
            TyperError::message("rtype_cast_opaque_ptr: hop.args_s[0] missing".to_string())
        })?;
        if !s0.is_constant() {
            return Err(TyperError::message(
                "rtype_cast_opaque_ptr: hop.args_s[0] must be a constant".to_string(),
            ));
        }
    }
    let r_arg1 = arg_repr(hop, 1)?;
    // upstream `assert isinstance(hop.args_r[1], rptr.PtrRepr)`
    if !matches!(r_arg1.repr_class_id(), ReprClassId::PtrRepr) {
        return Err(TyperError::message(format!(
            "rtype_cast_opaque_ptr: hop.args_r[1] must be PtrRepr, got {:?}",
            r_arg1.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Void),
        ConvertedTo::Repr(r_arg1.as_ref()),
    ])?;
    hop.exception_cannot_occur()?;
    let v_input = vlist
        .into_iter()
        .nth(1)
        .ok_or_else(|| TyperError::message("rtype_cast_opaque_ptr: missing v_input".to_string()))?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message("rtype_cast_opaque_ptr: r_result missing".to_string())
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop(
        "cast_opaque_ptr",
        vec![v_input],
        GenopResult::LLType(lltype),
    ))
}

/// RPython `@typer_for(lltype.length_of_simple_gcarray_from_opaque)`
/// `def rtype_length_of_simple_gcarray_from_opaque(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.length_of_simple_gcarray_from_opaque)
/// def rtype_length_of_simple_gcarray_from_opaque(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     v_opaque_ptr, = hop.inputargs(hop.args_r[0])
///     hop.exception_cannot_occur()
///     return hop.genop('length_of_simple_gcarray_from_opaque', [v_opaque_ptr],
///                      resulttype=hop.r_result.lowleveltype)
/// ```
pub fn rtype_length_of_simple_gcarray_from_opaque(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    // upstream `assert isinstance(hop.args_r[0], rptr.PtrRepr)` (rbuiltin.py:440)
    if !matches!(r_arg.repr_class_id(), ReprClassId::PtrRepr) {
        return Err(TyperError::message(format!(
            "rtype_length_of_simple_gcarray_from_opaque: hop.args_r[0] must be PtrRepr, got {:?}",
            r_arg.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?;
    hop.exception_cannot_occur()?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message(
                    "rtype_length_of_simple_gcarray_from_opaque: r_result missing".to_string(),
                )
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop(
        "length_of_simple_gcarray_from_opaque",
        vlist,
        GenopResult::LLType(lltype),
    ))
}

/// RPython `@typer_for(lltype.direct_fieldptr) def rtype_direct_fieldptr(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.direct_fieldptr)
/// def rtype_direct_fieldptr(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     assert hop.args_s[1].is_constant()
///     vlist = hop.inputargs(hop.args_r[0], lltype.Void)
///     hop.exception_cannot_occur()
///     return hop.genop('direct_fieldptr', vlist,
///                      resulttype=hop.r_result.lowleveltype)
/// ```
///
pub fn rtype_direct_fieldptr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::annotator::model::SomeObjectTrait;
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    // upstream `assert isinstance(hop.args_r[0], rptr.PtrRepr)`
    if !matches!(r_arg.repr_class_id(), ReprClassId::PtrRepr) {
        return Err(TyperError::message(format!(
            "rtype_direct_fieldptr: hop.args_r[0] must be PtrRepr, got {:?}",
            r_arg.repr_class_id()
        )));
    }
    // upstream `assert hop.args_s[1].is_constant()`
    {
        let args_s = hop.args_s.borrow();
        let s1 = args_s.get(1).ok_or_else(|| {
            TyperError::message("rtype_direct_fieldptr: hop.args_s[1] missing".to_string())
        })?;
        if !s1.is_constant() {
            return Err(TyperError::message(
                "rtype_direct_fieldptr: hop.args_s[1] must be a constant".to_string(),
            ));
        }
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r_arg.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Void),
    ])?;
    hop.exception_cannot_occur()?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message("rtype_direct_fieldptr: r_result missing".to_string())
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop("direct_fieldptr", vlist, GenopResult::LLType(lltype)))
}

/// RPython `@typer_for(lltype.direct_arrayitems) def rtype_direct_arrayitems(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.direct_arrayitems)
/// def rtype_direct_arrayitems(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     vlist = hop.inputargs(hop.args_r[0])
///     hop.exception_cannot_occur()
///     return hop.genop('direct_arrayitems', vlist,
///                      resulttype=hop.r_result.lowleveltype)
/// ```
pub fn rtype_direct_arrayitems(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    // upstream `assert isinstance(hop.args_r[0], rptr.PtrRepr)`
    if !matches!(r_arg.repr_class_id(), ReprClassId::PtrRepr) {
        return Err(TyperError::message(format!(
            "rtype_direct_arrayitems: hop.args_r[0] must be PtrRepr, got {:?}",
            r_arg.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg.as_ref())])?;
    hop.exception_cannot_occur()?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message("rtype_direct_arrayitems: r_result missing".to_string())
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop("direct_arrayitems", vlist, GenopResult::LLType(lltype)))
}

/// RPython `@typer_for(lltype.direct_ptradd) def rtype_direct_ptradd(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.direct_ptradd)
/// def rtype_direct_ptradd(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     vlist = hop.inputargs(hop.args_r[0], lltype.Signed)
///     hop.exception_cannot_occur()
///     return hop.genop('direct_ptradd', vlist,
///                      resulttype=hop.r_result.lowleveltype)
/// ```
///
pub fn rtype_direct_ptradd(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg = arg_repr(hop, 0)?;
    // `assert isinstance(hop.args_r[0], rptr.PtrRepr)`.
    if r_arg.repr_class_id() != ReprClassId::PtrRepr {
        return Err(TyperError::message(format!(
            "rtype_direct_ptradd: hop.args_r[0] must be PtrRepr, got {:?}",
            r_arg.repr_class_id()
        )));
    }
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r_arg.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    hop.exception_cannot_occur()?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        r_result_borrow
            .as_ref()
            .ok_or_else(|| {
                TyperError::message("rtype_direct_ptradd: r_result missing".to_string())
            })?
            .lowleveltype()
            .clone()
    };
    Ok(hop.genop("direct_ptradd", vlist, GenopResult::LLType(lltype)))
}

/// RPython `@typer_for(objectmodel.keepalive_until_here) def rtype_keepalive_until_here(hop)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(objectmodel.keepalive_until_here)
/// def rtype_keepalive_until_here(hop):
///     hop.exception_cannot_occur()
///     for v in hop.args_v:
///         hop.genop('keepalive', [v], resulttype=lltype.Void)
///     return hop.inputconst(lltype.Void, None)
/// ```
pub fn rtype_keepalive_until_here(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rtyper::GenopResult;

    hop.exception_cannot_occur()?;
    let args_v: Vec<Hlvalue> = hop.args_v.borrow().clone();
    for v in args_v {
        let _ = hop.genop("keepalive", vec![v], GenopResult::Void);
    }
    let void_const = HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::None)?;
    Ok(Some(Hlvalue::Constant(void_const)))
}

/// RPython `@typer_for(lltype.render_immortal) def rtype_render_immortal(hop, i_track_allocation=None)`
/// (rbuiltin.py).
///
/// ```python
/// @typer_for(lltype.render_immortal)
/// def rtype_render_immortal(hop, i_track_allocation=None):
///     vlist = [hop.inputarg(hop.args_r[0], arg=0)]
///     v_track_allocation = parse_kwds(hop,
///         (i_track_allocation, None))
///     hop.exception_cannot_occur()
///     if i_track_allocation is None or v_track_allocation.value:
///         hop.genop('track_alloc_stop', vlist)
/// ```
///
/// Upstream quirk: `parse_kwds` returns a list, so `v_track_allocation`
/// is `[hlval]`, not the inner element.  The `or v_track_allocation.value`
/// branch raises `AttributeError` if `i_track_allocation` is `Some` —
/// production callers never supply the `track_allocation` keyword, so
/// the bug never surfaces upstream.  Pyre surfaces a `TyperError` in
/// the same case rather than silently emitting a Void constant, since
/// upstream observably fails the moment the keyword is supplied.
pub fn rtype_render_immortal(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let i_track_allocation = kwds_i.get("i_track_allocation").copied();
    let r_arg0 = arg_repr(hop, 0)?;
    let v_first = hop.inputarg(r_arg0.as_ref(), 0)?;
    let vlist = vec![v_first];
    let _kw = parse_kwds(hop, &[(i_track_allocation, None)])?;
    hop.exception_cannot_occur()?;
    if i_track_allocation.is_some() {
        // upstream `v_track_allocation.value` is `list.value` → AttributeError.
        return Err(TyperError::message(
            "rtype_render_immortal: `track_allocation=` keyword triggers an \
             AttributeError upstream (parse_kwds returns a list, and the \
             upstream `v_track_allocation.value` access fails); production \
             callers never supply the keyword"
                .to_string(),
        ));
    }
    let _ = hop.genop("track_alloc_stop", vlist, GenopResult::Void);
    // upstream has no `return` — implicit `None`.
    Ok(None)
}

/// `std::ptr::eq(a, b)` — the Rust spelling of `a is b` over wrapped
/// objects (`baseobjspace::is_w`).  Identity comparison routes through
/// the full pairtype `rtype_is_` dispatch, exactly like an upstream
/// `is` operation: instance pairs of different classes take the
/// `pairtype(InstanceRepr, InstanceRepr)` common-base arm
/// (rclass.py) before the generic pointer body
/// (rmodel.py) emits `ptr_eq` with a Bool result and
/// constant-folds when the annotator proved the answer.
fn rtype_ptr_eq(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let r0 = arg_repr(hop, 0)?;
    let r1 = arg_repr(hop, 1)?;
    crate::translator::rtyper::pairtype::pair_rtype_is_(r0.as_ref(), r1.as_ref(), hop)
}

/// `std::ptr::copy_nonoverlapping(src, dst, count) -> ()`.
///
/// `count` is an element count. `llmemory.raw_memcopy` takes a byte
/// count (`llmemory.py` `raw_memcopy`). `rgc.ll_arraycopy` multiplies
/// `sizeof(TP.OF) * length` before that call. Scale by `sizeof` of
/// `src`'s pointee, then `cast_ptr_to_adr` so the llop sees `Address`
/// operands (`rtype_raw_memcopy`).
///
/// `BuiltinFunctionRepr.rtype_simple_call` has already popped the
/// callable, so the hop is `(src, dst, count)`.
///
/// `std_ptr_copy_nonoverlapping` returns `s_None`. `consider_op` binds
/// `s_ImpossibleValue` only when the analyzer returns nothing, and
/// `translate_no_return_value` accepts only that. Hand back the Void
/// constant after `raw_memcopy`, as `rtype_keepalive` does for
/// `immutablevalue(None)`.
fn rtype_copy_nonoverlapping(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rtyper::GenopResult;

    if hop.nb_args() != 3 {
        return Err(TyperError::message(format!(
            "copy_nonoverlapping: expected 3 arguments, got {}",
            hop.nb_args()
        )));
    }
    let r0 = arg_repr(hop, 0)?;
    let r1 = arg_repr(hop, 1)?;
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r0.as_ref()),
        ConvertedTo::Repr(r1.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    hop.exception_cannot_occur()?;
    let v_src = ptr_as_address(hop, vlist[0].clone(), r0.lowleveltype())?;
    let v_dst = ptr_as_address(hop, vlist[1].clone(), r1.lowleveltype())?;
    let v_nbytes = scale_copy_count(hop, r0.lowleveltype(), vlist[2].clone())?;
    let _ = hop.genop(
        "raw_memcopy",
        vec![v_src, v_dst, v_nbytes],
        GenopResult::Void,
    );
    let void_const = HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::None)?;
    Ok(Some(Hlvalue::Constant(void_const)))
}

/// `cast_ptr_to_adr` when the operand is a `Ptr` (`raddress.py`
/// `pairtype(PtrRepr, AddressRepr).convert_from_to`). An `Address` is
/// already the llop's operand type (`rtype_raw_memcopy`).
fn ptr_as_address(
    hop: &HighLevelOp,
    v: crate::flowspace::model::Hlvalue,
    ll: &LowLevelType,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;

    match ll {
        LowLevelType::Address => Ok(v),
        LowLevelType::Ptr(_) => hop
            .genop(
                "cast_ptr_to_adr",
                vec![v],
                GenopResult::LLType(LowLevelType::Address),
            )
            .ok_or_else(|| {
                TyperError::message("copy_nonoverlapping: cast_ptr_to_adr returned void")
            }),
        other => Err(TyperError::message(format!(
            "copy_nonoverlapping: src/dst must be Ptr or Address, got {other:?}"
        ))),
    }
}

/// `rgc.ll_arraycopy`: `raw_memcopy(..., sizeof(TP.OF) * length)`.
fn scale_copy_count(
    hop: &HighLevelOp,
    src_ll: &LowLevelType,
    v_count: crate::flowspace::model::Hlvalue,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::lltypesystem::llmemory::sizeof;
    use crate::translator::rtyper::rtyper::GenopResult;

    let pointee = match src_ll {
        LowLevelType::Ptr(ptr) => LowLevelType::from(ptr.TO.clone()),
        LowLevelType::Address => LowLevelType::Char,
        other => {
            return Err(TyperError::message(format!(
                "copy_nonoverlapping: src must be a pointer, got {other:?}"
            )));
        }
    };
    if pointee._is_varsize() {
        return Err(TyperError::message(format!(
            "copy_nonoverlapping: cannot size varsize pointee {pointee:?}"
        )));
    }
    let size = sizeof(&pointee, None).map_err(TyperError::message)?;
    let c_size = HighLevelOp::inputconst(&LowLevelType::Signed, &size)?;
    hop.genop(
        "int_mul",
        vec![v_count, Hlvalue::Constant(c_size)],
        GenopResult::LLType(LowLevelType::Signed),
    )
    .ok_or_else(|| TyperError::message("copy_nonoverlapping: int_mul returned void"))
}

/// `std::ptr::null_mut::<T>()` / `null::<T>()` — the `lltype.nullptr`
/// analog (rbuiltin.py:412-418) for the `{core,std}.ptr.{null_mut,null}`
/// host builtins.  Upstream routes nullptr through `rtype_const_result`
/// because its annotation is a constant SomePtr; the pyre analyzer
/// (`ptr_null_constant`) instead returns a non-constant classdef-less
/// `SomeInstance(can_be_None=True)` so the value stays joinable with
/// typed receivers, and the null value is recovered here from the
/// callable's definition: `convert_const(None)` on the result repr is
/// `null_instance()` (rclass.py `convert_const` None arm).
fn rtype_ptr_null(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;

    hop.exception_cannot_occur()?;
    let r_result_borrow = hop.r_result.borrow();
    let r_result = r_result_borrow
        .as_ref()
        .ok_or_else(|| TyperError::message("rtype_ptr_null: r_result missing".to_string()))?;
    let c = crate::translator::rtyper::rmodel::inputconst(r_result.as_ref(), &ConstValue::None)?;
    Ok(Some(Hlvalue::Constant(c)))
}

/// `core.ptr.null_fn` — `lltype.nullptr(FuncType)` (`rptr.py`
/// `PtrRepr` over a function type). The annotation is `SomePtr`, so the
/// null is an `LLPtr` of the result's function type rather than
/// `convert_const(None)` on an `InstanceRepr`. A constant spelling
/// argument is consumed at annotation (`fn_null_constant`); `r_result`
/// already carries that `FuncType`.
fn rtype_null_fn(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::{ConstValue, Hlvalue};
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget, nullptr};

    hop.exception_cannot_occur()?;
    let r_result_borrow = hop.r_result.borrow();
    let r_result = r_result_borrow
        .as_ref()
        .ok_or_else(|| TyperError::message("rtype_null_fn: r_result missing".to_string()))?;
    let ll = r_result.lowleveltype().clone();
    let LowLevelType::Ptr(ptr) = &ll else {
        return Err(TyperError::message(format!(
            "rtype_null_fn: result is not a pointer, got {ll:?}"
        )));
    };
    let PtrTarget::Func(func) = &ptr.TO else {
        return Err(TyperError::message(format!(
            "rtype_null_fn: result is not a function pointer, got {ll:?}"
        )));
    };
    let null = nullptr(LowLevelType::Func(Box::new(func.clone()))).map_err(TyperError::message)?;
    Ok(Some(Hlvalue::Constant(
        crate::flowspace::model::Constant::with_concretetype(ConstValue::LLPtr(Box::new(null)), ll),
    )))
}

/// Empty-string constructors — `String::new()` / `String::with_capacity(n)`
/// / `Wtf8Buf::new()` / `Wtf8Buf::with_capacity(n)`.  The annotator's
/// `string_constructor` (`annotator/builtin.rs`) shells all four to an empty
/// `SomeString`, so the rtyper folds them to the interned empty `rpy_string`
/// constant (`ByteStr([])` → `StringRepr::convert_const` →
/// `const_str_cache_llstr`).  A capacity argument does not affect the empty
/// content and is dropped: the string value model is immutable, so a later
/// append yields a fresh string rather than mutating this initial value.
fn rtype_empty_string_ctor(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;

    hop.exception_cannot_occur()?;
    let r_result_borrow = hop.r_result.borrow();
    let r_result = r_result_borrow.as_ref().ok_or_else(|| {
        TyperError::message("rtype_empty_string_ctor: r_result missing".to_string())
    })?;
    let c = crate::translator::rtyper::rmodel::inputconst(
        r_result.as_ref(),
        &ConstValue::ByteStr(Vec::new()),
    )?;
    Ok(Some(Hlvalue::Constant(c)))
}

/// RPython `@typer_for(lltype.typeOf)` / `@typer_for(lltype.nullptr)` /
/// `@typer_for(lltype.getRuntimeTypeInfo)` / `@typer_for(lltype.Ptr)`
/// (rbuiltin.py). Single shared body — the result annotation
/// is constant by the time these typers fire, so the typed result is
/// just `inputconst(r_result.lowleveltype, s_result.const)`.
///
/// ```python
/// def rtype_const_result(hop):
///     hop.exception_cannot_occur()
///     return hop.inputconst(hop.r_result.lowleveltype, hop.s_result.const)
/// ```
///
/// Note on dispatch reach: `front::mir` synthesises most `lltype.*`
/// ops directly (no HostObject lookup on the production path).
/// Registration below mirrors upstream's
/// `BUILTIN_TYPER` shape structurally; dispatch flips on once the
/// M2.5g extern-Rust-helper walker lands.
pub fn rtype_const_result(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;

    hop.exception_cannot_occur()?;
    let const_val = hop
        .s_result
        .borrow()
        .as_ref()
        .and_then(|s| s.const_().cloned())
        .ok_or_else(|| {
            TyperError::message("rtype_const_result: s_result is not a constant".to_string())
        })?;
    let lltype = {
        let r_result_borrow = hop.r_result.borrow();
        let r_result = r_result_borrow.as_ref().ok_or_else(|| {
            TyperError::message("rtype_const_result: r_result missing".to_string())
        })?;
        r_result.lowleveltype().clone()
    };
    let c = HighLevelOp::inputconst(&lltype, &const_val)?;
    Ok(Some(Hlvalue::Constant(c)))
}

/// Pointer reprs `rtype_cast_ptr_to_int` consumes without the InstanceRepr
/// swap: `PtrRepr` (`rptr.py`) and `RustVecRepr`, whose lowleveltype is
/// `Ptr(Struct(raw) "RustVec")`.
fn repr_is_ptr_to_int_origin(r: &dyn Repr) -> bool {
    matches!(
        r.repr_class_id(),
        ReprClassId::PtrRepr | ReprClassId::RustVecRepr
    )
}

/// RPython `@typer_for(lltype.cast_ptr_to_int)` (rbuiltin.py):
///
/// ```python
/// @typer_for(lltype.cast_ptr_to_int)
/// def rtype_cast_ptr_to_int(hop):
///     assert isinstance(hop.args_r[0], rptr.PtrRepr)
///     vlist = hop.inputargs(hop.args_r[0])
///     hop.exception_cannot_occur()
///     return hop.genop('cast_ptr_to_int', vlist,
///                      resulttype=lltype.Signed)
/// ```
///
/// Reached via `simple_call(lltype.cast_ptr_to_int, p) →
/// BuiltinFunctionRepr.rtype_simple_call → BUILTIN_TYPER[lltype.\
/// cast_ptr_to_int]` per `rbuiltin.py` registry pattern.
/// `front::mir` lowers `expr as i64` for a Ref-typed source into
/// `Call { target: FunctionPath { segments: ["rpython", "rtyper",
/// "lltypesystem", "lltype", "cast_ptr_to_int"], fun_decl_id: None }, args: [operand] }`,
/// picked up by the flowspace_adapter's module-qualified FunctionPath
/// resolver → HOST_ENV.import_module(...).module_get("cast_ptr_to_int")
/// → BUILTIN_TYPER lookup.
pub fn rtype_cast_ptr_to_int(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::rmodel::PtrRepr;
    use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};

    // `cast_ptr_to_int` is a LOW-LEVEL op (emitted for `expr as i64`
    // on a Ref source, see the doc above): its operand must reach the
    // typer as a `PtrRepr`.  This block relabels a late-arriving
    // `InstanceRepr` operand to `PtrRepr` via its `concretetype`
    // (`InstanceRepr.lowleveltype` IS `Ptr(GcStruct(OBJECT))`,
    // rclass.py/477).  A low-level struct param whose annotation
    // already carries a `SomeValue::Ptr` lands as `PtrRepr` directly
    // (`rmodel.rs`'s `SomeValue::Ptr → PtrRepr`); the swap is
    // load-bearing for low-level operands that reach the typer as
    // `InstanceRepr` instead.
    //
    // Scope caveat (do NOT mis-read this as "drive every struct to
    // SomePtr"): RPython sets the annotation node by ORIGIN, not by
    // eventual LL type.  A value born in the lltype universe
    // (`lltype.malloc` / `GcStruct` / `rffi.CStruct`) is `SomePtr` from
    // annotation (`llannotation.py:185`, `lltype.py`); an  allow-line-citation
    // RPython class instance — and a host-struct method RECEIVER, which
    // is USED like one (classdef-mediated `getattr`/method dispatch,
    // `signature.py` `annotationoftype → SomeInstance`) — is
    // `SomeInstance(ClassDef)` at annotation and only lowered to
    // `Ptr(GcStruct(OBJECT))` at the rtyper.  `SomePtr.getattr` cannot
    // resolve a class-owned host method (PtrRepr → `_nofield`), so
    // annotating a receiver as `SomePtr` is lateral.  Narrowing a
    // receiver to a real `SomeInstance(ClassDef)` is a SEPARATE
    // convergence path (call-propagation / bind_self, gated on a
    // registered host ClassDef origin) — not this gate.
    //
    // Progression gate: SWAP_FALLBACK_HITS reading 0 means every
    // `cast_ptr_to_int` OPERAND already reaches the typer as a
    // registered low-level `PtrRepr` (NOT that every reachable struct
    // lifts to SomePtr); once that holds the swap can be deleted in
    // favour of the line-by-line `rbuiltin.py:541-549` port below.
    let producer_set_someptr = matches!(
        hop.args_v.borrow()[0].clone(),
        Hlvalue::Variable(ref var) if matches!(
            var.annotation.borrow().as_deref(),
            Some(SomeValue::Ptr(_))
        )
    );
    let needs_swap = hop.args_r.borrow()[0]
        .as_ref()
        .is_some_and(|r| !repr_is_ptr_to_int_origin(r.as_ref()));
    if producer_set_someptr {
        debug_assert!(
            !needs_swap,
            "rtype_cast_ptr_to_int: producer-set SomePtr operand reached the typer \
             without args_r[0] resolving to PtrRepr (rtyper makerepr path broken; \
             see rmodel.rs:2545 SomeValue::Ptr → PtrRepr wiring)",
        );
    }
    if needs_swap {
        SWAP_FALLBACK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let v = hop.args_v.borrow()[0].clone();
        let concrete = match &v {
            Hlvalue::Variable(var) => var.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        let ptr = match concrete {
            Some(LowLevelType::Ptr(p)) => *p,
            other => {
                return Err(TyperError::message(format!(
                    "rtype_cast_ptr_to_int: operand concretetype must be Ptr(...) for \
                     late InstanceRepr→PtrRepr swap (typed-ref-someptr-followup-epic \
                     fallback path), got {other:?}"
                )));
            }
        };
        hop.args_r.borrow_mut()[0] = Some(Arc::new(PtrRepr::new(ptr)) as Arc<dyn Repr>);
    }

    let r_arg1 = arg_repr(hop, 0)?;
    // RPython `rbuiltin.py rtype_cast_ptr_to_int` asserts
    // `isinstance(hop.args_r[0], rptr.PtrRepr)`. `RustVecRepr`'s
    // lowleveltype is `Ptr(Struct(raw) "RustVec")`, the same pointer
    // origin.
    assert!(
        repr_is_ptr_to_int_origin(r_arg1.as_ref()),
        "rtype_cast_ptr_to_int: hop.args_r[0] must be a pointer repr after \
         InstanceRepr→PtrRepr relabel, got {:?}",
        r_arg1.repr_class_id()
    );
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg1.as_ref())])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop(
        "cast_ptr_to_int",
        vlist,
        GenopResult::LLType(LowLevelType::Signed),
    ))
}

/// RPython `rtyper.py:478-481` — internal renaming op the rtyper
/// emits to copy a value into a fresh result variable while
/// preserving its lltype.  `front::mir` lowers identity /
/// source-type-unknown casts to `OpKind::UnaryOp { op: "same_as",
/// result_ty, .. }` so the target `ValueType` propagates through the
/// graph (RPython has no `expr as T` syntax — pyre adds it as a Rust
/// adaptation).  The
/// `unsimplify::split_block` Void-variable recreation
/// (unsimplify.rs) and the post-rtyper backendopt pipeline
/// (constfold / removenoops / storesink / inline) also generate
/// `same_as`.  Body parity verbatim: lift the operand's repr,
/// inputargs through that repr, emit the low-level `same_as` with
/// the same lltype.
pub fn rtype_same_as(hop: &HighLevelOp) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_arg1 = arg_repr(hop, 0)?;
    let lltype = r_arg1.lowleveltype().clone();
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_arg1.as_ref())])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop("same_as", vlist, GenopResult::LLType(lltype)))
}

/// RPython `rbuiltin.py @typer_for(lltype.cast_int_to_ptr)`.
///
/// ```python
/// @typer_for(lltype.cast_int_to_ptr)
/// def rtype_cast_int_to_ptr(hop):
///     assert hop.args_s[0].is_constant()
///     v_type, v_input = hop.inputargs(lltype.Void, lltype.Signed)
///     hop.exception_cannot_occur()
///     return hop.genop('cast_int_to_ptr', [v_input],
///                      resulttype=hop.r_result.lowleveltype)
/// ```
///
/// Upstream takes a 2-arg `simple_call(lltype.cast_int_to_ptr, T,
/// v_int)` shape — `args_s[0]` is the constant target type `T`
/// (Void-typed at LL, just a marker) and `args_s[1]` is the
/// integer value.  Upstream `rtype_cast_int_to_ptr` then calls
/// `hop.inputargs(lltype.Void, lltype.Signed)` (rbuiltin.py)
/// — two formal inputs.
///
/// TODO: pyre's frontend emits a 1-arg
/// `Call { target: FunctionPath { segments: ["rpython", "rtyper",
/// "lltypesystem", "lltype", "cast_int_to_ptr"], fun_decl_id: None }, args: [operand] }`
/// because at `Expr::Cast` lowering time the frontend has only
/// `ValueType::Ref` (the opaque high-level surface type), not the
/// concrete Ptr lltype.  This typer therefore takes a single `Signed`
/// inputarg instead of upstream's `(Void, Signed)`.  The result
/// lltype is recovered from `hop.r_result.lowleveltype`, which the
/// rtyper assigned when annotating the result variable — the same
/// concrete Ptr upstream would have read from `PtrT.const`
/// (lltype.py).  The constant carrier already exists
/// (`ConstValue::LowLevelType` carries arbitrary `LowLevelType`
/// including `Ptr`); the blocker is frontend-side access to the
/// concrete Ptr at lowering time.
pub fn rtype_cast_int_to_ptr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};

    let r_result = hop
        .r_result
        .borrow()
        .as_ref()
        .cloned()
        .ok_or_else(|| TyperError::message("rtype_cast_int_to_ptr: missing r_result"))?;
    let result_lltype = r_result.lowleveltype().clone();
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
    hop.exception_cannot_occur()?;
    Ok(hop.genop("cast_int_to_ptr", vlist, GenopResult::LLType(result_lltype)))
}

/// Typer for both front-end pointer-cast markers — the
/// `__cast_instance_intrinsic` downcast narrow (#298) and the
/// `__cast_address_intrinsic` erasure.  Either way the annotator has already
/// decided the result's `InstanceRepr` (the target root, or the
/// classdef-less shell), so `hop.r_result` names the destination and the
/// call lowers to a `cast_pointer` of the pointer operand to that
/// lowleveltype.  This mirrors `pairtype(InstanceRepr,
/// InstanceRepr).convert_from_to` (rclass.py) — the same
/// `genop('cast_pointer', [v], resulttype=r_ins2.lowleveltype)` it emits
/// for the `r_ins1.classdef is None` (classdef-less → concrete) arm, run
/// in one direction by the narrow and the other by the erasure.
/// `args[0]` is the pointer operand; the narrow's `args[1]` is the
/// constant root name (read by the annotator, unused here — it carries
/// no runtime value).
pub fn rtype_cast_instance_intrinsic(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_result =
        hop.r_result.borrow().as_ref().cloned().ok_or_else(|| {
            TyperError::message("rtype_cast_instance_intrinsic: missing r_result")
        })?;
    let result_lltype = r_result.lowleveltype().clone();
    // Validated operand extraction: a malformed call (wrong arity)
    // surfaces a `TyperError` here instead of panicking on a raw
    // `args_v[0]` index, matching the other typers in this module.
    let r_arg0 = arg_repr(hop, 0)?;
    let v_ptr = hop.inputarg(&r_arg0, 0)?;
    hop.exception_cannot_occur()?;
    if r_arg0.lowleveltype() == &result_lltype {
        return Ok(Some(v_ptr));
    }
    // Unrelated raw owners share one address word. `cast_pointer` requires
    // a parent chain (`lltype.castable`); `llmemory.cast_ptr_to_adr` then
    // `cast_adr_to_ptr` is the retype that does not change gc status.
    if let (LowLevelType::Ptr(cur), LowLevelType::Ptr(dest)) =
        (r_arg0.lowleveltype(), &result_lltype)
        && cur._gckind() == crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
        && dest._gckind() == crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
        && crate::translator::rtyper::lltypesystem::lltype::castable(dest, cur).is_err()
    {
        let v_adr = hop
            .genop(
                "cast_ptr_to_adr",
                vec![v_ptr.clone()],
                GenopResult::LLType(LowLevelType::Address),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "rtype_cast_instance_intrinsic: cast_ptr_to_adr produced no value".to_string(),
                )
            })?;
        return Ok(hop.genop(
            "cast_adr_to_ptr",
            vec![v_adr],
            GenopResult::LLType(result_lltype),
        ));
    }
    // RPython `rmodel.externalvsinternal(..., gcref=True)` converts a
    // concrete GC pointer to/from llmemory.GCREF with `cast_opaque_ptr`
    // (`rgcref.py __extend__`).  The same marker also carries ordinary
    // class-pointer narrows, which remain `cast_pointer` per rclass.py.
    let gcref = crate::translator::rtyper::lltypesystem::lltype::GCREF.clone();
    // `cast_pointer` is legal only between related structs: upstream decides
    // that with `castable(PTRTYPE, CURTYPE)` (`lltype.py`), which
    // raises `InvalidCast` for an unrelated pair.  An unrelated pair of GC
    // pointers is exactly the erasure RPython routes through
    // `llmemory.GCREF` — `pairtype(Repr, GCRefRepr).convert_from_to` casts
    // the source to GCREF and `pairtype(GCRefRepr, Repr).convert_from_to`
    // casts GCREF to the destination (`rgcref.py`).  Call those two
    // ported functions rather than re-emitting their ops, so a destination
    // that is not a GC pointer refuses here as it does upstream instead of
    // producing an invalid `cast_pointer`.
    if r_arg0.lowleveltype() != &gcref
        && result_lltype != gcref
        && let (LowLevelType::Ptr(cur), LowLevelType::Ptr(dest)) =
            (r_arg0.lowleveltype(), &result_lltype)
        && crate::translator::rtyper::lltypesystem::lltype::castable(dest, cur).is_err()
    {
        use crate::translator::rtyper::lltypesystem::rgcref::{
            GCRefRepr, pair_gcref_repr_convert_from_to, pair_repr_gcref_convert_from_to,
        };
        let r_gcref = GCRefRepr::make(r_arg0.clone(), &hop.rtyper.gcrefreprcache) as Arc<dyn Repr>;
        let mut llops = hop.llops.borrow_mut();
        let Some(v_gcref) =
            pair_repr_gcref_convert_from_to(r_arg0.as_ref(), r_gcref.as_ref(), &v_ptr, &mut llops)?
        else {
            return Err(TyperError::message(
                "rtype_cast_instance_intrinsic: conversion to GCREF produced no value",
            ));
        };
        let Some(v_result) = pair_gcref_repr_convert_from_to(
            r_gcref.as_ref(),
            r_result.as_ref(),
            &v_gcref,
            &mut llops,
        )?
        else {
            return Err(TyperError::message(format!(
                "rtype_cast_instance_intrinsic: {:?} is not castable from {:?} and \
                 pairtype(GCRefRepr, Repr).convert_from_to refuses a non-GC destination",
                result_lltype,
                r_arg0.lowleveltype(),
            )));
        };
        return Ok(Some(v_result));
    }
    let opname = if r_arg0.lowleveltype() == &gcref || result_lltype == gcref {
        "cast_opaque_ptr"
    } else {
        "cast_pointer"
    };
    Ok(hop.genop(opname, vec![v_ptr], GenopResult::LLType(result_lltype)))
}

/// Restore the MIR adapter marker to RPython
/// `rgc.ll_arraymove(array, source_start, dest_start, length)`.
///
/// `source_start` and `dest_start` arrive as `Signed` variables, so the
/// direction of the move is not known here.  The minted helper carries
/// upstream's runtime `delta` dispatch (`rpython/rlib/rgc.py`,
/// `ll_arraymove`) and is overlap-safe either way, which is also what lets
/// it be minted under the single name `ll_arraymove`: the rtyper caches
/// low-level helpers on `(name, args, result)`, and both directions present
/// the identical key.
pub fn rtype_ll_arraymove_intrinsic(
    hop: &HighLevelOp,
    _kwds_i: &HashMap<String, usize>,
) -> RTypeResult {
    use crate::translator::rtyper::rlist::{FixedSizeListRepr, build_ll_arraymove_helper_graph};

    let r_array = arg_repr(hop, 0)?;
    let any_array: &dyn std::any::Any = r_array.as_ref();
    let fixed = any_array
        .downcast_ref::<FixedSizeListRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "rtype_ll_arraymove_intrinsic requires FixedSizeListRepr items".to_string(),
            )
        })?;
    let item_lltype = fixed.item_lowleveltype();
    let array_lltype = r_array.lowleveltype().clone();
    let mut args = hop.inputargs(vec![
        ConvertedTo::Repr(r_array.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    let array = args.remove(0);
    let source_start = args.remove(0);
    let dest_start = args.remove(0);
    let length = args.remove(0);
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_arraymove".to_string(),
        vec![
            array_lltype,
            LowLevelType::Signed,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Void,
        move |_rtyper, _args, _result| {
            build_ll_arraymove_helper_graph("ll_arraymove", item_lltype.clone())
        },
    )?;
    hop.exception_cannot_occur()?;
    hop.gendirectcall(&helper, vec![array, source_start, dest_start, length])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeBuiltin, SomeBuiltinMethod};
    use crate::flowspace::model::HostObject;

    fn host_bltin(name: &str) -> HostObject {
        HostObject::new_builtin_callable(name)
    }

    #[test]
    fn builtin_function_repr_has_void_lowleveltype() {
        let r = BuiltinFunctionRepr::new(host_bltin("len"));
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::BuiltinFunctionRepr);
    }

    #[test]
    fn builtin_function_repr_convert_const_default_preserves_value_on_void() {
        // Upstream `Repr.convert_const` default (`rmodel.py`) keeps
        // the Python value; `Void._contains_value` (lltype.py)
        // accepts anything, so Void-typed reprs pass through the value
        // unchanged. No `BuiltinFunctionRepr.convert_const` override
        // exists upstream (rbuiltin.py).
        let r = BuiltinFunctionRepr::new(host_bltin("len"));
        let c = r.convert_const(&ConstValue::Int(42)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::Int(42)));
    }

    #[test]
    fn somebuiltin_rtyper_makerepr_constant_branch_returns_builtin_function_repr() {
        let mut s = SomeBuiltin::new("len", None, None);
        // Populate `const_box` so `is_constant()` succeeds.
        s.base.const_box = Some(Constant::new(ConstValue::HostObject(host_bltin("len"))));
        let r = somebuiltin_rtyper_makerepr(&s).unwrap();
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");
    }

    #[test]
    fn somebuiltin_rtyper_makerepr_non_constant_branch_errors() {
        // No const_box set → upstream raises `TyperError("non-constant built-in function!")`.
        let s = SomeBuiltin::new("len", None, None);
        let err = somebuiltin_rtyper_makerepr(&s).unwrap_err();
        assert!(err.to_string().contains("non-constant built-in function"));
    }

    fn dummy_rtyper() -> RPythonTyper {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        RPythonTyper::new(&ann)
    }

    #[test]
    fn somebuiltinmethod_rtyper_makerepr_builds_builtin_method_repr_with_self_repr() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        // s_self = SomeInteger → self_repr = IntegerRepr(Signed).
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let s_method = SomeBuiltinMethod::new("foo", s_self, "foo");
        let r = somebuiltinmethod_rtyper_makerepr(&s_method, &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinMethodRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::BuiltinMethodRepr);
        // lowleveltype mirrors the receiver's repr (IntegerRepr → Signed).
        assert_eq!(r.lowleveltype(), &LowLevelType::Signed);
    }

    #[test]
    fn builtin_method_repr_convert_const_rejects_non_hostobject() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();
        let err = repr.convert_const(&ConstValue::Int(42)).unwrap_err();
        assert!(err.to_string().contains("expected HostObject bound method"));
    }

    #[test]
    fn builtin_method_repr_convert_const_rejects_hostobject_not_bound_method() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();
        // A plain builtin HostObject (not a bound method) → error.
        let err = repr
            .convert_const(&ConstValue::HostObject(host_bltin("len")))
            .unwrap_err();
        assert!(err.to_string().contains("HostObject is not a bound method"));
    }

    fn noop_typer(_hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
        Ok(None)
    }

    #[test]
    fn typer_for_registers_and_lookup_typer_reads_back() {
        let key = host_bltin("rbuiltin_test_registers_and_reads_back");
        assert!(lookup_typer(&key).is_none());
        typer_for(key.clone(), noop_typer);
        assert!(lookup_typer(&key).is_some());
    }

    #[test]
    fn findbltintyper_returns_registered_typer() {
        let key = host_bltin("rbuiltin_test_findbltintyper_returns_registered");
        typer_for(key.clone(), noop_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let f = repr.findbltintyper().expect("typer found");
        // Function pointers compare by address — registered fn must
        // match `noop_typer`.
        assert!(std::ptr::eq(f as *const (), noop_typer as *const ()));
    }

    #[test]
    fn findbltintyper_unknown_host_raises_typererror() {
        // HostObject with no registry entry and not ConstValue::LLPtr
        // (so extregistry::is_registered returns false) → upstream
        // `TyperError("don't know about built-in function %r" % ...)`.
        let key = host_bltin("rbuiltin_test_findbltintyper_unknown_host");
        let repr = BuiltinFunctionRepr::new(key);
        let err = repr.findbltintyper().unwrap_err();
        assert!(
            err.to_string()
                .contains("don't know about built-in function")
        );
    }

    fn dummy_hop() -> HighLevelOp {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let spaceop = SpaceOperation::new(
            OpKind::SimpleCall.opname(),
            Vec::new(),
            crate::flowspace::model::Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops)
    }

    fn typer_calls_exception_cannot_occur(
        hop: &HighLevelOp,
        _kwds_i: &HashMap<String, usize>,
    ) -> RTypeResult {
        hop.exception_cannot_occur().unwrap();
        Ok(None)
    }

    #[test]
    fn call_returns_v_result_when_typer_calls_exception_cannot_occur() {
        let key = host_bltin("rbuiltin_test_call_returns_v_result");
        typer_for(key.clone(), typer_calls_exception_cannot_occur);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        let result = repr._call(&hop, &HashMap::new()).unwrap();
        assert!(result.is_none());
        // Flag was flipped to true by exception_cannot_occur (set back
        // to true after the initial `= false` reset in `_call`).
        assert!(hop.llops.borrow()._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn call_errors_when_typer_skips_exception_guard() {
        let key = host_bltin("rbuiltin_test_call_errors_when_skipped");
        // `noop_typer` never calls exception_is_here/exception_cannot_occur.
        typer_for(key.clone(), noop_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        let err = repr._call(&hop, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing hop.exception_cannot_occur()")
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn rtype_offsetof_folds_field_to_signed_symbolic_field_offset() {
        use crate::translator::rtyper::lltypesystem::llmemory::AddressOffset;
        use crate::translator::rtyper::lltypesystem::lltype::Struct;

        // A STR-like struct so `offsetof(TYPE, "chars")` resolves a field.
        let str_ty = LowLevelType::Struct(Box::new(Struct::new(
            "rpy_string",
            vec![
                ("hash".to_string(), LowLevelType::Signed),
                ("chars".to_string(), LowLevelType::Signed),
            ],
        )));

        // upstream `TYPE, field = hop.inputargs(lltype.Void, lltype.Void)` —
        // two Void constants carrying the struct type and the field name.
        let hop = dummy_hop();
        *hop.args_v.borrow_mut() = vec![
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::LowLevelType(Box::new(str_ty.clone())),
                LowLevelType::Void,
            )),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::ByteStr(b"chars".to_vec()),
                LowLevelType::Void,
            )),
        ];

        let result = rtype_offsetof(&hop, &HashMap::new()).unwrap().unwrap();
        let Hlvalue::Constant(c) = result else {
            panic!("expected a Constant, got {result:?}");
        };
        // `hop.inputconst(lltype.Signed, ...)` — the carrier lltype is Signed.
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
        // The value is the symbolic `FieldOffset(rpy_string, 'chars')`.
        match c.value {
            ConstValue::AddressOffset(AddressOffset::FieldOffset(f)) => {
                assert_eq!(f.fldname, "chars");
                assert_eq!(f.TYPE, str_ty);
            }
            other => panic!("expected a FieldOffset, got {other:?}"),
        }
    }

    #[test]
    fn rtype_simple_call_pops_first_arg_before_dispatching_typer() {
        use crate::flowspace::model::{Hlvalue, Variable};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NB_ARGS_AT_CALL: AtomicUsize = AtomicUsize::new(usize::MAX);
        fn record_nb_args_typer(
            hop: &HighLevelOp,
            _kwds_i: &HashMap<String, usize>,
        ) -> RTypeResult {
            NB_ARGS_AT_CALL.store(hop.nb_args(), Ordering::SeqCst);
            hop.exception_cannot_occur().unwrap();
            Ok(None)
        }

        let key = host_bltin("rbuiltin_test_rtype_simple_call_pops");
        typer_for(key.clone(), record_nb_args_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        // Seed hop with a single arg (the builtin callable itself) so
        // rtype_simple_call -> r_s_popfirstarg leaves 0 args for the typer.
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Builtin(SomeBuiltin::new("unused", None, None)));
        hop.args_r.borrow_mut().push(None);
        NB_ARGS_AT_CALL.store(usize::MAX, Ordering::SeqCst);
        repr.rtype_simple_call(&hop).unwrap();
        assert_eq!(NB_ARGS_AT_CALL.load(Ordering::SeqCst), 0);
        // Original hop is unaffected (upstream `hop.copy()` semantics).
        assert_eq!(hop.nb_args(), 1);
    }

    fn seed_hop_with_shape(
        hop: &HighLevelOp,
        shape_cnt: usize,
        shape_keys: &[&str],
        shape_star: bool,
        extra_args: usize,
    ) {
        use crate::annotator::model::{SomeInteger, SomeTuple};
        use crate::flowspace::model::{Constant, Hlvalue, Variable};

        // Position 0: the builtin callable itself (SomeValue::Builtin,
        // value unused here — just needs to exist for r_s_popfirstarg).
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Builtin(SomeBuiltin::new("unused", None, None)));
        hop.args_r.borrow_mut().push(None);

        // Position 1: the shape constant — SomeTuple of
        // (Int(shape_cnt), Tuple(Str(...)), Bool(shape_star)).
        let shape_const = ConstValue::Tuple(vec![
            ConstValue::Int(shape_cnt as i64),
            ConstValue::Tuple(
                shape_keys
                    .iter()
                    .map(|k| ConstValue::byte_str(*k))
                    .collect(),
            ),
            ConstValue::Bool(shape_star),
        ]);
        let mut s_shape = SomeTuple::new(Vec::new());
        s_shape.base.const_box = Some(Constant::new(shape_const));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s.borrow_mut().push(SomeValue::Tuple(s_shape));
        hop.args_r.borrow_mut().push(None);

        // Extra positions: placeholders, so hop.nb_args() reads as
        // expected by typers.
        for _ in 0..extra_args {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
    }

    #[test]
    fn call_args_expand_builds_kwds_i_from_shape_keys() {
        let hop = dummy_hop();
        // hint(x, category='foo') → shape_cnt=1, shape_keys=['category'].
        seed_hop_with_shape(&hop, 1, &["category"], false, 2);
        let (out_hop, kwds_i) = call_args_expand(&hop).unwrap();
        assert_eq!(kwds_i.len(), 1);
        // shape_cnt=1 → first keyword occupies index 1 (post-shape-header,
        // relative to the shape's own flatten).
        assert_eq!(kwds_i.get("i_category"), Some(&1));
        // hop is copied, not mutated.
        assert_eq!(out_hop.nb_args(), hop.nb_args());
    }

    #[test]
    fn call_args_expand_empty_keywords_yields_empty_kwds_i() {
        let hop = dummy_hop();
        seed_hop_with_shape(&hop, 2, &[], false, 2);
        let (_out_hop, kwds_i) = call_args_expand(&hop).unwrap();
        assert!(kwds_i.is_empty());
    }

    #[test]
    fn call_args_expand_rejects_stararg_shape() {
        let hop = dummy_hop();
        seed_hop_with_shape(&hop, 1, &[], true, 2);
        // HighLevelOp is not Debug, so `.unwrap_err()` won't compile —
        // pattern-match the Result manually.
        match call_args_expand(&hop) {
            Err(err) => assert!(err.to_string().contains("w_stararg is None")),
            Ok(_) => panic!("expected TyperError for shape_star=true"),
        }
    }

    #[test]
    fn rtype_call_args_forwards_kwds_i_and_pops_two_leading_args() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static LAST_CATEGORY_IDX: AtomicUsize = AtomicUsize::new(usize::MAX);
        static LAST_NB_ARGS: AtomicUsize = AtomicUsize::new(usize::MAX);

        fn kwds_observing_typer(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
            LAST_NB_ARGS.store(hop.nb_args(), Ordering::SeqCst);
            LAST_CATEGORY_IDX.store(
                kwds_i.get("i_category").copied().unwrap_or(usize::MAX),
                Ordering::SeqCst,
            );
            hop.exception_cannot_occur().unwrap();
            Ok(None)
        }

        let key = host_bltin("rbuiltin_test_rtype_call_args_forwards");
        typer_for(key.clone(), kwds_observing_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        // hint(x, category='foo') → shape_cnt=1, shape_keys=['category'],
        // extra_args=2 so hop.nb_args()=4 → hop2.nb_args()=2 after the
        // two r_s_popfirstarg calls.
        seed_hop_with_shape(&hop, 1, &["category"], false, 2);
        LAST_CATEGORY_IDX.store(usize::MAX, Ordering::SeqCst);
        LAST_NB_ARGS.store(usize::MAX, Ordering::SeqCst);
        repr.rtype_call_args(&hop).unwrap();
        assert_eq!(LAST_NB_ARGS.load(Ordering::SeqCst), 2);
        assert_eq!(LAST_CATEGORY_IDX.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_kwds_all_none_specs_yield_none_results_and_preserve_args_v() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(None, None), (None, None)];
        let result = parse_kwds(&hop, &specs).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].is_none());
        assert!(result[1].is_none());
        // lst is empty → tail_start == nb_args, truncate is a no-op.
        assert_eq!(hop.nb_args(), 1);
    }

    #[test]
    fn parse_kwds_rejects_misordered_keyword_indices() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        for _ in 0..3 {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
        // nb_args=3, specify i=0 (should be 2) — lst=[0] must equal [2].
        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(Some(0), None)];
        match parse_kwds(&hop, &specs) {
            Err(err) => assert!(
                err.to_string()
                    .contains("keyword args are expected to be at the end")
            ),
            Ok(_) => panic!("expected parse_kwds to reject misordered keyword index"),
        }
    }

    #[test]
    fn parse_kwds_consumes_tail_args_and_truncates_args_v() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        // nb_args = 2: [positional, keyword]. Seed the keyword as a
        // Hlvalue::Constant so inputarg takes the early-return path
        // (avoids convertvar wiring).
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(42),
                LowLevelType::Signed,
            )));
        let mut s_const = SomeInteger::new(false, false);
        s_const.base.const_box = Some(Constant::new(ConstValue::Int(42)));
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_const));
        hop.args_r.borrow_mut().push(None);

        let r: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(Some(1), Some(r.clone()))];
        let result = parse_kwds(&hop, &specs).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());
        // args_v truncated to 1 (tail_start = nb_args(2) - len(lst)(1)).
        assert_eq!(hop.args_v.borrow().len(), 1);
    }

    #[test]
    fn builtin_method_repr_rtype_simple_call_surfaces_missing_rtype_method() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();

        let hop = dummy_hop();
        // Seed args_v[0] with a Variable (non-constant) — no bound-method
        // unwrap path.
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        let err = repr.rtype_simple_call(&hop).unwrap_err();
        // IntegerRepr has no rtype_method_foo override → default trait
        // method raises the "missing <class>.rtype_method_<name>" error.
        let msg = err.to_string();
        assert!(msg.contains("rtype_method_foo"));
    }

    #[test]
    fn builtin_method_repr_rtype_simple_call_rewrites_constant_bound_method_arg0() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Constant, Hlvalue};
        use crate::translator::rtyper::rmodel::Repr;

        // Observer repr to inspect args_v[0] after rtype_simple_call's
        // rewrite. Captures the constant via interior mutability.
        #[derive(Debug)]
        struct ObserverRepr {
            state: ReprState,
            lltype: LowLevelType,
            captured: parking_lot::Mutex<Option<ConstValue>>,
        }
        impl ObserverRepr {
            fn new() -> Arc<Self> {
                Arc::new(ObserverRepr {
                    state: ReprState::new(),
                    lltype: LowLevelType::Signed,
                    captured: parking_lot::Mutex::new(None),
                })
            }
        }
        impl Repr for ObserverRepr {
            fn lowleveltype(&self) -> &LowLevelType {
                &self.lltype
            }
            fn state(&self) -> &ReprState {
                &self.state
            }
            fn class_name(&self) -> &'static str {
                "ObserverRepr"
            }
            fn repr_class_id(&self) -> ReprClassId {
                ReprClassId::Repr
            }
            fn convert_const(&self, _v: &ConstValue) -> Result<Constant, TyperError> {
                Err(TyperError::message("ObserverRepr.convert_const unused"))
            }
            fn rtype_method(&self, _name: &str, hop: &HighLevelOp) -> RTypeResult {
                if let Hlvalue::Constant(c) = &hop.args_v.borrow()[0] {
                    *self.captured.lock() = Some(c.value.clone());
                }
                hop.exception_cannot_occur().unwrap();
                Ok(None)
            }
        }

        let observer: Arc<ObserverRepr> = ObserverRepr::new();
        let self_repr: Arc<dyn Repr> = observer.clone();
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let repr = BuiltinMethodRepr {
            s_self: s_self.clone(),
            self_repr,
            methodname: "foo".into(),
            state: ReprState::new(),
            lltype: LowLevelType::Signed,
        };

        // Build a bound-method HostObject whose __self__ is an int-like
        // builtin-callable (unused — only identity matters for the
        // capture).
        let receiver = host_bltin("receiver_obj");
        let func = host_bltin("receiver_method");
        let origin_class = host_bltin("origin_class");
        let bound = HostObject::new_bound_method(
            "m.receiver_method",
            receiver.clone(),
            func,
            "foo",
            origin_class,
        );

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                bound,
            ))));
        hop.args_s.borrow_mut().push((*s_self).clone());
        hop.args_r.borrow_mut().push(None);

        repr.rtype_simple_call(&hop).unwrap();

        // ObserverRepr captured the rewritten arg0: bound method's
        // HostObject was replaced by its __self__.
        let captured = observer.captured.lock().clone().unwrap();
        match captured {
            ConstValue::HostObject(h) => assert_eq!(h, receiver),
            other => panic!("expected rewritten HostObject, got {other:?}"),
        }
    }

    #[test]
    fn pair_builtin_method_convert_from_to_returns_none_when_methodnames_differ() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = Rc::new(dummy_rtyper());
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let from = BuiltinMethodRepr::new(&rtyper, s_self.clone(), "foo".into()).unwrap();
        let to = BuiltinMethodRepr::new(&rtyper, s_self, "bar".into()).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v = Hlvalue::Variable(Variable::new());
        let result = pair_builtin_method_convert_from_to(&from, &to, &v, &mut llops).unwrap();
        // NotImplemented → Ok(None), pair_mro walker falls through.
        assert!(result.is_none());
    }

    #[test]
    fn pair_builtin_method_convert_from_to_same_methodname_delegates_to_self_repr_convertvar() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = Rc::new(dummy_rtyper());
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let from = BuiltinMethodRepr::new(&rtyper, s_self.clone(), "same".into()).unwrap();
        let to = BuiltinMethodRepr::new(&rtyper, s_self, "same".into()).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v = Hlvalue::Variable(Variable::new());
        // Both self_reprs are IntegerRepr(Signed) → convertvar's
        // same-repr short-circuit returns orig_v unchanged.
        let result = pair_builtin_method_convert_from_to(&from, &to, &v, &mut llops)
            .unwrap()
            .expect("same-methodname path returns Some");
        match (&v, &result) {
            (Hlvalue::Variable(a), Hlvalue::Variable(b)) => assert_eq!(a, b),
            _ => panic!("expected Variable round-trip"),
        }
    }

    #[test]
    fn install_default_typers_registers_baseline_builtins_from_host_env() {
        // rbuiltin.py + 709-715 — `@typer_for(<builtin>)` entries
        // that reach the registry via the bare `lookup_builtin` path
        // (HOST_ENV `builtins` table).  `unichr` is Python 2-only and
        // absent from pyre's HOST_ENV bootstrap, so the silent-skip on
        // missing entries leaves it out of BUILTIN_TYPER — excluded
        // from this assertion deliberately.
        for name in [
            "bool",
            "int",
            "float",
            "chr",
            "unicode",
            "bytearray",
            "list",
            "min",
            "max",
            "hasattr",
        ] {
            let host = HOST_ENV
                .lookup_builtin(name)
                .unwrap_or_else(|| panic!("HOST_ENV missing builtin {name}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `{name}`"
            );
        }
    }

    #[test]
    fn install_default_typers_registers_intmask_longlongmask_from_rarithmetic_module() {
        // rbuiltin.py — `@typer_for(rarithmetic.intmask)` /
        // `@typer_for(rarithmetic.longlongmask)` reach the registry via
        // module-attribute lookup, not the bare `builtins` table.
        for attr in ["intmask", "longlongmask"] {
            let host = HOST_ENV
                .import_module("rpython.rlib.rarithmetic")
                .and_then(|m| m.module_get(attr))
                .unwrap_or_else(|| panic!("HOST_ENV missing rarithmetic.{attr}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `rarithmetic.{attr}`"
            );
        }
    }

    #[test]
    fn install_default_typers_registers_keepalive_until_here_from_objectmodel_module() {
        // rbuiltin.py — `@typer_for(objectmodel.keepalive_until_here)`.
        let host = HOST_ENV
            .import_module("rpython.rlib.objectmodel")
            .and_then(|m| m.module_get("keepalive_until_here"))
            .expect("HOST_ENV missing objectmodel.keepalive_until_here");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `objectmodel.keepalive_until_here`"
        );
    }

    #[test]
    fn install_default_typers_registers_const_result_typers_from_lltype_module() {
        // rbuiltin.py:412-418 — four upstream callables share the
        // `rtype_const_result` body.  Pin the registry hit for each.
        for attr in ["typeOf", "nullptr", "getRuntimeTypeInfo", "Ptr"] {
            let host = HOST_ENV
                .import_module("rpython.rtyper.lltypesystem.lltype")
                .and_then(|m| m.module_get(attr))
                .unwrap_or_else(|| panic!("HOST_ENV missing lltype.{attr}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `lltype.{attr}`"
            );
        }
    }

    #[test]
    fn install_default_typers_registers_identityhash_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.identityhash)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("identityhash"))
            .expect("HOST_ENV missing lltype.identityhash");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.identityhash`"
        );
    }

    #[test]
    fn install_default_typers_registers_runtime_type_info_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.runtime_type_info)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("runtime_type_info"))
            .expect("HOST_ENV missing lltype.runtime_type_info");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.runtime_type_info`"
        );
    }

    #[test]
    fn copy_nonoverlapping_typer_is_shared_by_core_and_std_ptr() {
        let std_host = HOST_ENV
            .import_module("std.ptr")
            .and_then(|m| m.module_get("copy_nonoverlapping"))
            .expect("HOST_ENV missing std.ptr.copy_nonoverlapping");
        let core_host = HOST_ENV
            .import_module("core.ptr")
            .and_then(|m| m.module_get("copy_nonoverlapping"))
            .expect("HOST_ENV missing core.ptr.copy_nonoverlapping");
        assert_eq!(
            std_host, core_host,
            "core.ptr.copy_nonoverlapping must be the std.ptr HostObject"
        );
        assert_eq!(std_host.qualname(), "std.ptr.copy_nonoverlapping");
        assert!(
            lookup_typer(&std_host).is_some(),
            "BUILTIN_TYPER missing std.ptr.copy_nonoverlapping"
        );
        assert!(
            lookup_typer(&core_host).is_some(),
            "the shared HostObject must carry the typer"
        );
    }

    #[test]
    fn copy_nonoverlapping_s_none_simple_call_emits_raw_memcopy() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInstance, SomeInteger, s_none};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        use std::cell::RefCell;

        let host = HOST_ENV
            .import_module("std.ptr")
            .and_then(|m| m.module_get("copy_nonoverlapping"))
            .expect("HOST_ENV missing std.ptr.copy_nonoverlapping");
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        // Instance / OBJECT is a fixed-size GcStruct. SomeString's STR
        // is varsize (`chars` array), so sizeof of that pointee errors.
        let src = Variable::named("src");
        src.annotation
            .replace(Some(Rc::new(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                Default::default(),
            )))));
        let dst = Variable::named("dst");
        dst.annotation
            .replace(Some(Rc::new(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                Default::default(),
            )))));
        let count = Variable::named("count");
        count
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                true, true,
            )))));
        let result = Variable::named("copied");
        result.annotation.replace(Some(Rc::new(s_none())));

        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::HostObject(host))),
                Hlvalue::Variable(src.clone()),
                Hlvalue::Variable(dst.clone()),
                Hlvalue::Variable(count.clone()),
            ],
            Hlvalue::Variable(result.clone()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), Rc::clone(&llops));
        hop.setup().expect("copy_nonoverlapping hop setup");
        // `setup_block_entry` stamps each input's concretetype before
        // `translate_hl_to_ll`. These operands are not block inputs.
        for var in [&src, &dst, &count] {
            let shell = ann.binding(&Hlvalue::Variable(var.clone()));
            let repr = rtyper.getrepr(&shell).expect("operand repr");
            var.set_concretetype(Some(repr.lowleveltype().clone()));
        }
        let mut varmapping = HashMap::new();
        rtyper
            .translate_hl_to_ll(&hop, &mut varmapping)
            .expect("s_None simple_call must agree with the void copy");
        let names: Vec<String> = llops
            .borrow()
            .ops
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert!(
            names.iter().any(|name| name == "raw_memcopy"),
            "raw_memcopy missing from {names:?}"
        );
        assert!(
            names.iter().any(|name| name == "int_mul"),
            "element count must scale by sizeof, missing int_mul in {names:?}"
        );
        assert!(
            names.iter().any(|name| name == "cast_ptr_to_adr"),
            "Ptr operands must convert to Address, missing cast_ptr_to_adr in {names:?}"
        );
        assert_eq!(result.concretetype(), Some(LowLevelType::Void));
    }

    #[test]
    fn install_default_typers_registers_direct_ptradd_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.direct_ptradd)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("direct_ptradd"))
            .expect("HOST_ENV missing lltype.direct_ptradd");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.direct_ptradd`"
        );
    }

    #[test]
    fn install_default_typers_registers_direct_fieldptr_and_arrayitems_from_lltype_module() {
        // rbuiltin.py / 455-461 — `@typer_for(lltype.direct_fieldptr)` /
        // `@typer_for(lltype.direct_arrayitems)`.
        for attr in ["direct_fieldptr", "direct_arrayitems"] {
            let host = HOST_ENV
                .import_module("rpython.rtyper.lltypesystem.lltype")
                .and_then(|m| m.module_get(attr))
                .unwrap_or_else(|| panic!("HOST_ENV missing lltype.{attr}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `lltype.{attr}`"
            );
        }
    }

    #[test]
    fn install_default_typers_registers_cast_opaque_ptr_and_length_of_gcarray_from_lltype_module() {
        // rbuiltin.py / 438-444 — `@typer_for(lltype.cast_opaque_ptr)` /
        // `@typer_for(lltype.length_of_simple_gcarray_from_opaque)`.
        for attr in ["cast_opaque_ptr", "length_of_simple_gcarray_from_opaque"] {
            let host = HOST_ENV
                .import_module("rpython.rtyper.lltypesystem.lltype")
                .and_then(|m| m.module_get(attr))
                .unwrap_or_else(|| panic!("HOST_ENV missing lltype.{attr}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `lltype.{attr}`"
            );
        }
    }

    #[test]
    fn install_default_typers_registers_cast_pointer_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.cast_pointer)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_pointer"))
            .expect("HOST_ENV missing lltype.cast_pointer");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.cast_pointer`"
        );
    }

    #[test]
    fn install_default_typers_registers_render_immortal_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.render_immortal)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("render_immortal"))
            .expect("HOST_ENV missing lltype.render_immortal");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.render_immortal`"
        );
    }

    #[test]
    fn install_default_typers_registers_cast_primitive_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.cast_primitive)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_primitive"))
            .expect("HOST_ENV missing lltype.cast_primitive");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.cast_primitive`"
        );
    }

    #[test]
    fn install_default_typers_registers_raw_malloc_from_llmemory_module() {
        // rbuiltin.py — `@typer_for(llmemory.raw_malloc)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.llmemory")
            .and_then(|m| m.module_get("raw_malloc"))
            .expect("HOST_ENV missing llmemory.raw_malloc");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `llmemory.raw_malloc`"
        );
    }

    #[test]
    fn install_default_typers_registers_raw_malloc_usage_from_llmemory_module() {
        // rbuiltin.py — `@typer_for(llmemory.raw_malloc_usage)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.llmemory")
            .and_then(|m| m.module_get("raw_malloc_usage"))
            .expect("HOST_ENV missing llmemory.raw_malloc_usage");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `llmemory.raw_malloc_usage`"
        );
    }

    #[test]
    fn install_default_typers_registers_cast_ptr_to_adr_from_llmemory_module() {
        // rbuiltin.py — `@typer_for(llmemory.cast_ptr_to_adr)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.llmemory")
            .and_then(|m| m.module_get("cast_ptr_to_adr"))
            .expect("HOST_ENV missing llmemory.cast_ptr_to_adr");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `llmemory.cast_ptr_to_adr`"
        );
    }

    #[test]
    fn install_default_typers_registers_object_init_qualname_builtin() {
        // rbuiltin.py — `@typer_for(object.__init__)`.
        let host = HOST_ENV
            .lookup_builtin("object.__init__")
            .expect("HOST_ENV missing `object.__init__` qualname builtin");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `object.__init__`"
        );
    }

    #[test]
    fn install_default_typers_registers_cast_int_to_adr_from_llmemory_module() {
        // rbuiltin.py — `@typer_for(llmemory.cast_int_to_adr)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.llmemory")
            .and_then(|m| m.module_get("cast_int_to_adr"))
            .expect("HOST_ENV missing llmemory.cast_int_to_adr");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `llmemory.cast_int_to_adr`"
        );
    }

    #[test]
    fn install_default_typers_registers_malloc_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.malloc)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("malloc"))
            .expect("HOST_ENV missing lltype.malloc");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.malloc`"
        );
    }

    #[test]
    fn install_default_typers_registers_free_from_lltype_module() {
        // rbuiltin.py — `@typer_for(lltype.free)`.
        let host = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("free"))
            .expect("HOST_ENV missing lltype.free");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `lltype.free`"
        );
    }

    #[test]
    fn install_default_typers_registers_free_non_gc_object_from_objectmodel_module() {
        // rbuiltin.py — `@typer_for(objectmodel.free_non_gc_object)`.
        let host = HOST_ENV
            .import_module("rpython.rlib.objectmodel")
            .and_then(|m| m.module_get("free_non_gc_object"))
            .expect("HOST_ENV missing objectmodel.free_non_gc_object");
        assert!(
            lookup_typer(&host).is_some(),
            "BUILTIN_TYPER missing entry for `objectmodel.free_non_gc_object`"
        );
    }

    #[test]
    fn rtype_builtin_bool_rejects_nb_args_mismatch() {
        let hop = dummy_hop();
        // nb_args = 0 → should err.
        let err = rtype_builtin_bool(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("rtype_builtin_bool"));
    }

    /// `rvirtualizable.py VirtualizableInstanceRepr.hook_access_field`
    /// emits the force for every redirected field.  Pyre spells that hook as a
    /// callable marker, so this typer is what each frame gateway's force
    /// resolves to, and its emitted shape is what `jtransform.py
    /// rewrite_op_jit_force_virtualizable` matches on when it deletes the op
    /// from a looked-inside graph.  A wrong arity, a lost repr conversion or a
    /// missing exception declaration would break every gateway silently.
    #[test]
    fn rtype_jit_force_virtualizable_emits_a_three_argument_void_llop() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{ConstValue, Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        let frame = Variable::named("v_frame");
        frame.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(frame));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let frame_repr: Arc<dyn Repr> =
            Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(frame_repr));

        // The marker returns None normally. Returning no Hlvalue here would
        // tell translate_hl_to_ll that the call never returns and reject the
        // annotator's s_None result at every descriptor gateway.
        let result = rtype_jit_force_virtualizable(&hop, &HashMap::new())
            .unwrap()
            .expect("a None-returning marker must return its Void Variable");

        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the force llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "jit_force_virtualizable");
        assert_eq!(
            last.args.len(),
            3,
            "hook_access_field emits [vinst, cname, cflags]",
        );
        match &last.args[0] {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("the frame argument must stay a Variable, got {other:?}"),
        }
        match &last.args[1] {
            Hlvalue::Constant(c) => assert_eq!(c.concretetype, Some(LowLevelType::Void)),
            other => panic!("cname must be a Void constant, got {other:?}"),
        }
        match &last.args[2] {
            Hlvalue::Constant(c) => {
                assert_eq!(c.concretetype, Some(LowLevelType::Void));
                assert!(matches!(c.value, ConstValue::Dict(_)));
            }
            other => panic!("cflags must be a Void dict constant, got {other:?}"),
        }
        match &last.result {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Void)),
            other => panic!("the force result must be a Void Variable, got {other:?}"),
        }
        assert!(
            llops._called_exception_is_here_or_cannot_occur,
            "the typer must declare that the force cannot raise",
        );
    }

    /// `ConditionalCallEntry.specialize_call` for
    /// `_jit_conditional_call_value` emits `jit_conditional_call_value`
    /// with args `(value, funcptr, arg)`. Flavour `_i` comes from
    /// `hop.r_result.lowleveltype` (`signed_repr()`).
    #[test]
    fn rtype_jit_conditional_call_value_emits_value_funcptr_arg() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, SomePtr, functionptr};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rmodel::PtrRepr;

        let hop = dummy_hop();
        let r_int: Arc<dyn Repr> = signed_repr();
        *hop.r_result.borrow_mut() = Some(r_int.clone());
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));

        let v_value = Variable::named("v_value");
        v_value.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_value));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        let fptr = functionptr(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            },
            "dummy_cond_call_fn",
            None,
            None,
        );
        let ptr_ty = fptr._TYPE.clone();
        let fptr_ll = LowLevelType::Ptr(Box::new(ptr_ty.clone()));
        let c_func = Constant::with_concretetype(ConstValue::LLPtr(Box::new(fptr)), fptr_ll);
        hop.args_v.borrow_mut().push(Hlvalue::Constant(c_func));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Ptr(SomePtr::new(ptr_ty.clone())));
        hop.args_r
            .borrow_mut()
            .push(Some(Arc::new(PtrRepr::new(ptr_ty))));

        let v_arg = Variable::named("v_arg");
        v_arg.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_arg));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(Some(r_int));

        let result = rtype_jit_conditional_call_value(&hop, &HashMap::new())
            .unwrap()
            .expect("value helper returns a result Variable");

        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the value llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "jit_conditional_call_value");
        assert_eq!(last.args.len(), 3, "args are (value, funcptr, arg)");
        match &last.args[0] {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("value must stay a Variable, got {other:?}"),
        }
        match &last.args[1] {
            Hlvalue::Constant(c) => {
                assert!(matches!(c.value, ConstValue::LLPtr(_)));
                assert!(matches!(c.concretetype, Some(LowLevelType::Ptr(_))));
            }
            other => panic!("funcptr must be a Constant, got {other:?}"),
        }
        match &last.args[2] {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("arg must stay a Variable, got {other:?}"),
        }
        match &last.result {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("value result must be Signed, got {other:?}"),
        }
        assert!(
            llops._called_exception_is_here_or_cannot_occur,
            "ConditionalCallEntry.specialize_call calls exception_is_here",
        );
    }

    /// Void twin of `rtype_jit_conditional_call_value_emits_value_funcptr_arg`.
    #[test]
    fn rtype_jit_conditional_call_emits_cond_funcptr_arg() {
        use crate::annotator::model::{SomeBool, SomeInteger, s_none};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, SomePtr, functionptr};
        use crate::translator::rtyper::rbool::bool_repr;
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rmodel::PtrRepr;

        let hop = dummy_hop();
        *hop.s_result.borrow_mut() = Some(s_none());

        let r_bool: Arc<dyn Repr> = bool_repr();
        let v_cond = Variable::named("v_cond");
        v_cond.set_concretetype(Some(LowLevelType::Bool));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_cond));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Bool(SomeBool::default()));
        hop.args_r.borrow_mut().push(Some(r_bool));

        let fptr = functionptr(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            "dummy_cond_call_void_fn",
            None,
            None,
        );
        let ptr_ty = fptr._TYPE.clone();
        let fptr_ll = LowLevelType::Ptr(Box::new(ptr_ty.clone()));
        let c_func = Constant::with_concretetype(ConstValue::LLPtr(Box::new(fptr)), fptr_ll);
        hop.args_v.borrow_mut().push(Hlvalue::Constant(c_func));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Ptr(SomePtr::new(ptr_ty.clone())));
        hop.args_r
            .borrow_mut()
            .push(Some(Arc::new(PtrRepr::new(ptr_ty))));

        let r_int: Arc<dyn Repr> = signed_repr();
        let v_arg = Variable::named("v_arg");
        v_arg.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_arg));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(Some(r_int));

        let result = rtype_jit_conditional_call(&hop, &HashMap::new())
            .unwrap()
            .expect("void helper returns a Void Variable");

        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the void llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "jit_conditional_call");
        assert_eq!(last.args.len(), 3, "args are (cond, funcptr, arg)");
        match &last.args[0] {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Bool)),
            other => panic!("condition must stay a Variable, got {other:?}"),
        }
        match &last.args[1] {
            Hlvalue::Constant(c) => {
                assert!(matches!(c.value, ConstValue::LLPtr(_)));
                assert!(matches!(c.concretetype, Some(LowLevelType::Ptr(_))));
            }
            other => panic!("funcptr must be a Constant, got {other:?}"),
        }
        match &last.result {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Void)),
            other => panic!("void result must be a Void Variable, got {other:?}"),
        }
        assert!(
            llops._called_exception_is_here_or_cannot_occur,
            "ConditionalCallEntry.specialize_call calls exception_is_here",
        );
    }

    #[test]
    fn rtype_builtin_int_non_string_branch_delegates_to_args_r_rtype_int() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        // Seed the variable with a concretetype so hop.genop's
        // assertion doesn't panic inside IntegerRepr::rtype_int.
        let var = Variable::new();
        var.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(var));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        // IntegerRepr::rtype_int is implemented — non-string branch
        // delegates and returns a variable.
        let result = rtype_builtin_int(&hop, &HashMap::new()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn rtype_builtin_float_rejects_nb_args_2() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        for _ in 0..2 {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
        let err = rtype_builtin_float(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected nb_args == 1"));
    }

    #[test]
    fn rtype_builtin_unichr_rejects_nb_args_mismatch() {
        let hop = dummy_hop();
        let err = rtype_builtin_unichr(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("rtype_builtin_unichr"));
    }

    #[test]
    fn rtype_builtin_unicode_delegates_to_default_rtype_unicode_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_unicode(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("unicode"));
    }

    #[test]
    fn rtype_builtin_bytearray_delegates_to_default_rtype_bytearray_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_bytearray(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("bytearray"));
    }

    #[test]
    fn rtype_builtin_list_delegates_to_default_rtype_bltn_list_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_list(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("bltn_list"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn rtype_EnvironmentError_init_stores_errno_zero_for_a_bare_self() {
        use crate::annotator::classdesc::{Attribute, ClassDef};
        use crate::annotator::model::{SomeInstance, SomeInteger};
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rmodel::Repr;

        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("EnvironmentError", None);
        let mut errno = Attribute::new("errno");
        errno.s_value = SomeValue::Integer(SomeInteger::new(false, false));
        errno.readonly = false;
        classdef
            .borrow_mut()
            .attrs
            .insert("errno".to_string(), errno);
        let r_self = getinstancerepr(&hop.rtyper, Some(&classdef), Flavor::Gc).expect("repr");
        Repr::setup(r_self.as_ref()).expect("setup");
        let v_self = Variable::new();
        v_self.set_concretetype(Some(r_self.lowleveltype().clone()));
        let r_dyn: std::sync::Arc<dyn Repr> = r_self;
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_self));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )));
        hop.args_r.borrow_mut().push(Some(r_dyn));

        rtype_EnvironmentError__init__(&hop, &HashMap::new()).expect("rtype");
        let ops = hop.llops.borrow();
        let store = ops
            .ops
            .iter()
            .find(|op| op.opname == "setfield")
            .expect("errno setfield");
        assert_eq!(store.args.len(), 3);
        let Hlvalue::Constant(name) = &store.args[1] else {
            panic!("field name must be a constant");
        };
        assert_eq!(name.value, ConstValue::ByteStr(b"inst_errno".to_vec()));
        let Hlvalue::Constant(errno) = &store.args[2] else {
            panic!("errno must be a constant");
        };
        assert_eq!(errno.value, ConstValue::Int(0));
    }

    #[test]
    #[allow(non_snake_case)]
    fn rtype_WindowsError_init_stores_winerror_and_rejects_one_argument() {
        use crate::annotator::classdesc::{Attribute, ClassDef};
        use crate::annotator::model::{SomeInstance, SomeInteger};
        use crate::flowspace::model::{Constant, Hlvalue, Variable};
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rmodel::Repr;

        let bare = dummy_hop();
        bare.args_v.borrow_mut().extend([
            Hlvalue::Variable(Variable::new()),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(1),
                LowLevelType::Signed,
            )),
        ]);
        let err = rtype_WindowsError__init__(&bare, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("WindowsError() should not be called with a single argument")
        );

        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("WindowsError", None);
        let mut winerror = Attribute::new("winerror");
        winerror.s_value = SomeValue::Integer(SomeInteger::new(false, false));
        winerror.readonly = false;
        classdef
            .borrow_mut()
            .attrs
            .insert("winerror".to_string(), winerror);
        let r_self = getinstancerepr(&hop.rtyper, Some(&classdef), Flavor::Gc).expect("repr");
        Repr::setup(r_self.as_ref()).expect("setup");
        let v_self = Variable::new();
        v_self.set_concretetype(Some(r_self.lowleveltype().clone()));
        let r_dyn: std::sync::Arc<dyn Repr> = r_self;
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_self),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(7),
                LowLevelType::Signed,
            )),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::ByteStr(b"disk".to_vec()),
                LowLevelType::Void,
            )),
        ]);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )));
        hop.args_r.borrow_mut().push(Some(r_dyn));

        rtype_WindowsError__init__(&hop, &HashMap::new()).expect("rtype");
        let ops = hop.llops.borrow();
        let store = ops
            .ops
            .iter()
            .find(|op| op.opname == "setfield")
            .expect("winerror setfield");
        let Hlvalue::Constant(name) = &store.args[1] else {
            panic!("field name must be a constant");
        };
        assert_eq!(name.value, ConstValue::ByteStr(b"inst_winerror".to_vec()));
        let Hlvalue::Constant(code) = &store.args[2] else {
            panic!("winerror must be a constant");
        };
        assert_eq!(code.value, ConstValue::Int(7));
    }

    #[test]
    fn dispatch_routes_somebuiltin_to_function_repr_and_somebuiltinmethod_to_method_repr() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();

        // SomeBuiltin — constant branch.
        let mut sb = SomeBuiltin::new("abs", None, None);
        sb.base.const_box = Some(Constant::new(ConstValue::HostObject(host_bltin("abs"))));
        let r = dispatch_rtyper_makerepr(&SomeValue::Builtin(sb), &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");

        // SomeBuiltinMethod — now ported.
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let sbm = SomeBuiltinMethod::new("foo", s_self, "foo");
        let r = dispatch_rtyper_makerepr(&SomeValue::BuiltinMethod(sbm), &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinMethodRepr");
    }

    #[test]
    fn swap_fallback_hits_round_trip() {
        // The `cast_ptr_to_int` retirement-readiness counter is
        // process-global; reset before observing so cross-test bleed
        // is excluded.  Subsequent increments through the public
        // `SWAP_FALLBACK_HITS` atomic surface in the reader.
        reset_swap_fallback_hits();
        assert_eq!(swap_fallback_hits(), 0);
        SWAP_FALLBACK_HITS.fetch_add(3, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(swap_fallback_hits(), 3);
        reset_swap_fallback_hits();
        assert_eq!(swap_fallback_hits(), 0);
    }

    fn objectptr_type() -> crate::translator::rtyper::lltypesystem::lltype::Ptr {
        match crate::translator::rtyper::rclass::OBJECTPTR.clone() {
            LowLevelType::Ptr(p) => *p,
            other => panic!("OBJECTPTR must be Ptr, got {other:?}"),
        }
    }

    #[test]
    fn rtype_instantiate_variable_class_indirect_calls_instantiate() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::SomePBC;
        use crate::flowspace::model::{ConstValue, GraphKey, Hlvalue, Variable};
        use crate::translator::rtyper::rclass::{CLASSTYPE, Flavor, OBJECTPTR, getinstancerepr};
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rpbc::ClassesPBCRepr;

        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let left = ClassDef::new_standalone("pkg.Left", None);
        let right = ClassDef::new_standalone("pkg.Right", None);
        crate::translator::rtyper::normalizecalls::create_instantiate_function(&ann, &left)
            .expect("left graph");
        crate::translator::rtyper::normalizecalls::create_instantiate_function(&ann, &right)
            .expect("right graph");
        let s_pbc = SomePBC::new(
            vec![
                DescEntry::Class(left.borrow().classdesc.clone()),
                DescEntry::Class(right.borrow().classdesc.clone()),
            ],
            false,
        );
        assert_eq!(s_pbc.descriptions.len(), 2);
        let r_class = ClassesPBCRepr::new(&hop.rtyper, s_pbc).expect("ClassesPBCRepr");
        let mut expected_graphs = Vec::new();
        for entry in r_class.s_pbc.descriptions.values() {
            let DescEntry::Class(desc) = entry else {
                panic!("class desc");
            };
            let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(desc).unwrap();
            let graph = classdef.borrow().my_instantiate_graph.clone().unwrap();
            expected_graphs.push(GraphKey::of(&graph).as_usize());
        }
        let v_cls = Variable::new();
        v_cls.set_concretetype(Some(CLASSTYPE.clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_cls));
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::PBC(
                r_class.s_pbc.clone(),
            ));
        hop.args_r.borrow_mut().push(Some(
            std::sync::Arc::new(r_class) as std::sync::Arc<dyn Repr>
        ));
        let r_result = getinstancerepr(&hop.rtyper, None, Flavor::Gc).expect("object instance");
        Repr::setup(r_result.as_ref()).expect("setup object");
        *hop.r_result.borrow_mut() = Some(r_result as std::sync::Arc<dyn Repr>);

        let out = rtype_instantiate(&hop, &HashMap::new())
            .expect("rtype_instantiate")
            .expect("cast_pointer result");
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 3);
        assert_eq!(ops.ops[0].opname, "getfield");
        let Hlvalue::Constant(field) = &ops.ops[0].args[1] else {
            panic!("getfield name");
        };
        assert_eq!(field.value, ConstValue::byte_str("instantiate"));
        assert_eq!(ops.ops[1].opname, "indirect_call");
        let Hlvalue::Constant(graphs) = &ops.ops[1].args[1] else {
            panic!("c_graphs");
        };
        assert_eq!(graphs.value, ConstValue::Graphs(expected_graphs));
        let Hlvalue::Variable(called) = &ops.ops[1].result else {
            panic!("indirect_call result");
        };
        assert_eq!(called.concretetype(), Some(OBJECTPTR.clone()));
        assert_eq!(ops.ops[2].opname, "cast_pointer");
        assert_eq!(ops.ops[2].result, out);
    }

    #[test]
    fn rtype_builtin_reversed_calls_r_result_newiter() {
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rrange::{AbstractRangeRepr, RangeIteratorRepr};

        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::SpaceOperation;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call",
                Vec::new(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops,
        );
        let r_rng = AbstractRangeRepr::new(1).expect("step-1 range");
        let iter = std::sync::Arc::new(RangeIteratorRepr::new(&r_rng).expect("range iterator"));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(r_rng.lowleveltype().clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_rng));
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(std::sync::Arc::new(r_rng) as std::sync::Arc<dyn Repr>));
        *hop.r_result.borrow_mut() = Some(iter as std::sync::Arc<dyn Repr>);

        let out = rtype_builtin_reversed(&hop, &HashMap::new())
            .expect("reversed")
            .expect("newiter result");
        let ops = hop.llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert_eq!(ops.ops[0].result, out);
        let Hlvalue::Constant(func) = &ops.ops[0].args[0] else {
            panic!("direct_call arg0 is the helper");
        };
        let rendered = format!("{:?}", func.value);
        assert!(
            rendered.contains("ll_rangeiter"),
            "reversed on a range iterator must call ll_rangeiter, got {rendered}"
        );
    }

    #[test]
    fn rtype_cast_ptr_to_int_with_ptrrepr_does_not_swap() {
        use crate::annotator::model::SomePtr;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rmodel::PtrRepr;
        use std::rc::Rc;

        let hop = dummy_hop();
        let ptr = objectptr_type();
        let var = Variable::named("p");
        var.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        var.annotation
            .replace(Some(Rc::new(SomeValue::Ptr(SomePtr::new(ptr.clone())))));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(var));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Ptr(SomePtr::new(ptr.clone())));
        let ptr_repr: Arc<dyn Repr> = Arc::new(PtrRepr::new(ptr));
        let args_r_before = Arc::as_ptr(&ptr_repr);
        hop.args_r.borrow_mut().push(Some(ptr_repr));

        let result = rtype_cast_ptr_to_int(&hop, &HashMap::new())
            .unwrap()
            .expect("cast_ptr_to_int returns a Signed variable");
        let args_r_after = hop.args_r.borrow()[0].as_ref().map(Arc::as_ptr);
        assert_eq!(
            args_r_after,
            Some(args_r_before),
            "orthodox PtrRepr must not be replaced by the InstanceRepr swap"
        );
        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "cast_ptr_to_int");
        match &last.result {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("expected Signed result, got {other:?}"),
        }
        assert!(
            llops._called_exception_is_here_or_cannot_occur,
            "rbuiltin.py rtype_cast_ptr_to_int calls exception_cannot_occur"
        );
    }

    #[test]
    fn rtype_cast_ptr_to_int_swaps_instancerepr_with_ptr_concretetype() {
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};

        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let inst = getinstancerepr(&hop.rtyper, None, Flavor::Gc).expect("root InstanceRepr");
        let ptr = objectptr_type();
        let var = Variable::named("inst");
        var.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr))));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(var));
        hop.args_s.borrow_mut().push(SomeValue::Instance(
            crate::annotator::model::SomeInstance::new(None, false, Default::default()),
        ));
        hop.args_r.borrow_mut().push(Some(inst as Arc<dyn Repr>));
        assert_eq!(
            hop.args_r.borrow()[0].as_ref().map(|r| r.repr_class_id()),
            Some(ReprClassId::InstanceRepr)
        );

        let result = rtype_cast_ptr_to_int(&hop, &HashMap::new())
            .unwrap()
            .expect("swap still emits cast_ptr_to_int");
        assert_eq!(
            hop.args_r.borrow()[0].as_ref().map(|r| r.repr_class_id()),
            Some(ReprClassId::PtrRepr),
            "the fallback relabels InstanceRepr to PtrRepr"
        );
        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "cast_ptr_to_int");
    }

    #[test]
    fn rtype_cast_ptr_to_int_with_rustvecrepr_does_not_swap() {
        use crate::annotator::model::SomeRustVec;
        use crate::flowspace::model::{Hlvalue, Variable};
        use majit_ir::rvec::VecItemKind;

        reset_swap_fallback_hits();
        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let s_vec = SomeValue::RustVec(SomeRustVec::for_kind(VecItemKind::Int));
        let r = hop.rtyper.getrepr(&s_vec).expect("RustVecRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::RustVecRepr);
        assert!(
            matches!(r.lowleveltype(), LowLevelType::Ptr(_)),
            "RustVecRepr lowleveltype is Ptr"
        );
        let var = Variable::named("l");
        var.set_concretetype(Some(r.lowleveltype().clone()));
        var.annotation
            .replace(Some(std::rc::Rc::new(s_vec.clone())));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(var));
        hop.args_s.borrow_mut().push(s_vec);
        hop.args_r.borrow_mut().push(Some(r));
        let args_r_before = hop.args_r.borrow()[0].as_ref().map(Arc::as_ptr);

        let result = rtype_cast_ptr_to_int(&hop, &HashMap::new())
            .unwrap()
            .expect("cast_ptr_to_int of RustVecRepr returns a Signed variable");
        let args_r_after = hop.args_r.borrow()[0].as_ref().map(Arc::as_ptr);
        assert_eq!(
            args_r_after, args_r_before,
            "RustVecRepr is a pointer repr; rtype_cast_ptr_to_int must not swap it"
        );
        assert_eq!(swap_fallback_hits(), 0);
        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("the llop is emitted");
        assert_eq!(result, last.result);
        assert_eq!(last.opname, "cast_ptr_to_int");
        match &last.result {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Signed)),
            other => panic!("expected Signed result, got {other:?}"),
        }
        assert!(
            llops._called_exception_is_here_or_cannot_occur,
            "rbuiltin.py rtype_cast_ptr_to_int calls exception_cannot_occur"
        );
    }

    #[test]
    fn rtype_direct_ptradd_rejects_instancerepr() {
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};

        let hop = dummy_hop();
        hop.rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let inst = getinstancerepr(&hop.rtyper, None, Flavor::Gc).expect("root InstanceRepr");
        hop.args_r.borrow_mut().push(Some(inst as Arc<dyn Repr>));
        let err = rtype_direct_ptradd(&hop, &HashMap::new()).expect_err("PtrRepr only");
        assert!(
            err.to_string().contains("must be PtrRepr"),
            "rtype_direct_ptradd asserts PtrRepr, got {err}"
        );
    }

    fn dict_constructor_hop() -> (
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        HighLevelOp,
    ) {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "simple_call",
                Vec::new(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops,
        );
        (ann, hop)
    }

    #[test]
    fn rtype_dict_constructor_calls_ll_newdict() {
        use crate::annotator::dictdef::DictDef;
        use crate::annotator::model::{SomeDict, SomeInteger, SomeString, SomeValue};
        use crate::translator::rtyper::rdict::somedict_rtyper_makerepr;

        let (ann, hop) = dict_constructor_hop();
        let _keep_ann = ann;
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let r_dict = somedict_rtyper_makerepr(&SomeDict::new(dictdef), &hop.rtyper).expect("dict");
        *hop.r_result.borrow_mut() = Some(r_dict);

        let out = rtype_dict_constructor(&hop, &HashMap::new())
            .expect("dict constructor")
            .expect("ll_newdict result");
        let ops = hop.llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert_eq!(ops.ops[0].result, out);
        let crate::flowspace::model::Hlvalue::Constant(func) = &ops.ops[0].args[0] else {
            panic!("direct_call arg0 is the helper");
        };
        let rendered = format!("{:?}", func.value);
        assert!(
            rendered.contains("ll_newdict"),
            "dict constructor must call ll_newdict, got {rendered}"
        );
    }

    #[test]
    fn rtype_dict_constructor_stores_custom_eq_hash_fields() {
        use crate::annotator::dictdef::DictDef;
        use crate::annotator::model::{SomeInteger, SomeString, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::lltypesystem::rordereddict::OrderedDictRepr;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rmodel::Repr;

        let (ann, hop) = dict_constructor_hop();
        let _keep_ann = ann;
        let rtyper = hop.rtyper.clone();
        let key = rtyper
            .getrepr(&SomeValue::Integer(SomeInteger::default()))
            .expect("key");
        let value = rtyper
            .getrepr(&SomeValue::String(SomeString::new(false, false)))
            .expect("value");
        let eq = std::sync::Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")))
            as std::sync::Arc<dyn Repr>;
        let hash = std::sync::Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")))
            as std::sync::Arc<dyn Repr>;
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            true,
            false,
            false,
        );
        let r_dict = OrderedDictRepr::new(
            rtyper.self_rc().expect("rtyper rc"),
            key,
            value,
            dictdef,
            Some((eq, hash)),
            false,
            false,
        )
        .expect("ordered dict");
        *hop.r_result.borrow_mut() = Some(std::sync::Arc::new(r_dict) as std::sync::Arc<dyn Repr>);
        hop.args_v.borrow_mut().extend([
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(1),
                LowLevelType::Signed,
            )),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(2),
                LowLevelType::Signed,
            )),
        ]);

        rtype_dict_constructor(&hop, &HashMap::new()).expect("custom dict constructor");
        let ops = hop.llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let fields: Vec<_> = ops
            .ops
            .iter()
            .filter(|op| op.opname == "setfield")
            .map(|op| {
                let Hlvalue::Constant(name) = &op.args[1] else {
                    panic!("field name");
                };
                name.value.clone()
            })
            .collect();
        assert_eq!(
            fields,
            vec![
                ConstValue::byte_str("fnkeyeq"),
                ConstValue::byte_str("fnkeyhash"),
            ]
        );
    }

    fn hlinvoke_function(
        ann: &std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        rtyper: &std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        with_signature: bool,
    ) -> std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> {
        use crate::annotator::description::{CallTableRow, DescEntry, FunctionDesc};
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::{Block, ConstValue, Constant, FunctionGraph, GraphFunc};
        use crate::flowspace::pygraph::PyGraph;
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rpbc::FunctionRepr;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            Signature::new(Vec::new(), None, None),
            None,
            None,
        )));
        if with_signature {
            let family = fd.borrow().base.getcallfamily().expect("callfamily");
            let func = GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default())));
            let graph = FunctionGraph::new("f", Block::shared(Vec::new()));
            {
                let block = graph.returnblock.borrow();
                let crate::flowspace::model::Hlvalue::Variable(ret) = &block.inputargs[0] else {
                    panic!("return var");
                };
                *ret.annotation.borrow_mut() =
                    Some(std::rc::Rc::new(SomeValue::Integer(SomeInteger::default())));
            }
            let py = Rc::new(PyGraph {
                graph: Rc::new(RefCell::new(graph)),
                func,
                signature: RefCell::new(Signature::new(Vec::new(), None, None)),
                defaults: RefCell::new(None),
                access_directly: Cell::new(false),
            });
            let mut row = CallTableRow::new();
            row.insert(fd.borrow().base.identity, Rc::clone(&py));
            fd.borrow().cache.borrow_mut().insert(
                crate::annotator::description::GraphCacheKey::None,
                py.clone(),
            );
            family.borrow_mut().calltables.insert(
                crate::flowspace::argument::CallShape {
                    shape_cnt: 0,
                    shape_keys: Vec::new(),
                    shape_star: false,
                },
                vec![row],
            );
        }
        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        FunctionRepr::new(rtyper, s_pbc).expect("function repr") as std::sync::Arc<dyn Repr>
    }

    #[test]
    fn rtype_hlinvoke_rejects_non_uniform_signatures() {
        use crate::annotator::model::{KnownType, SomeObject, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue};

        let (ann, hop) = dict_constructor_hop();
        let r_fn = hlinvoke_function(&ann, &hop.rtyper, false);
        let mut s_repr = SomeObject::new(KnownType::Object, true);
        s_repr.const_box = Some(Constant::new(repr_const(std::sync::Arc::clone(&r_fn))));
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Object(s_repr), SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([None, Some(r_fn)]);
        hop.args_v.borrow_mut().extend([
            Hlvalue::Constant(Constant::new(ConstValue::None)),
            Hlvalue::Constant(Constant::new(ConstValue::None)),
        ]);

        let err = rtype_hlinvoke(&hop, &HashMap::new()).expect_err("non-uniform");
        let msg = err.to_string();
        assert!(
            msg.contains("cannot hlinvoke"),
            "non-uniform signatures must fail like rbuiltin.py, got {msg}"
        );
    }

    #[test]
    fn rtype_hlinvoke_dispatches_the_callable_repr() {
        use crate::annotator::model::{KnownType, SomeInteger, SomeObject, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue};

        let (ann, hop) = dict_constructor_hop();
        let r_fn = hlinvoke_function(&ann, &hop.rtyper, true);
        let mut s_repr = SomeObject::new(KnownType::Object, true);
        s_repr.const_box = Some(Constant::new(repr_const(std::sync::Arc::clone(&r_fn))));
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Object(s_repr), SomeValue::Impossible]);
        hop.args_r
            .borrow_mut()
            .extend([None, Some(std::sync::Arc::clone(&r_fn))]);
        hop.args_v.borrow_mut().extend([
            Hlvalue::Constant(Constant::new(ConstValue::None)),
            Hlvalue::Constant(Constant::new(ConstValue::None)),
        ]);
        let r_int = hop
            .rtyper
            .getrepr(&SomeValue::Integer(SomeInteger::default()))
            .expect("int repr");
        *hop.r_result.borrow_mut() = Some(r_int);

        rtype_hlinvoke(&hop, &HashMap::new()).expect("hlinvoke dispatch");
        assert!(
            matches!(hop.args_s.borrow().first(), Some(SomeValue::PBC(_))),
            "hlinvoke rewrites args_s[0] to the callable PBC before dispatch"
        );
        assert!(
            std::sync::Arc::ptr_eq(
                hop.args_r.borrow()[0].as_ref().expect("callable repr"),
                &r_fn
            ),
            "hlinvoke installs s_repr.const as args_r[0]"
        );
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "direct_call"),
            "dispatch of a constant function emits direct_call, ops={:?}",
            ops.ops
                .iter()
                .map(|op| op.opname.as_str())
                .collect::<Vec<_>>()
        );
    }
}
