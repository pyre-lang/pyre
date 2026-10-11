//! Built-in function analysers — direct port of
//! `rpython/annotator/builtin.py`.
//!
//! Upstream registers two families of analysers on a module-level
//! `BUILTIN_ANALYZERS` dict:
//!
//! * The `builtin_*` loop at the tail of `builtin.py:190-195` scans
//!   module globals for names starting with `builtin_` and registers
//!   them under the matching `__builtin__` attribute (e.g.
//!   `__builtin__.range` → `builtin_range`).
//! * Explicit `@analyzer_for(<callable>)` decorators register analysers
//!   for specific RPython-rlib helpers (`intmask`, `instantiate`,
//!   `r_dict`, …) plus a handful of stdlib entries (`weakref.ref`,
//!   `sys.getdefaultencoding`, `collections.OrderedDict`, `pdb.set_trace`).
//!
//! The Rust port mirrors that split. Analysers are keyed by the
//! **qualname** of the host callable (the string that
//! `HostObject::qualname()` returns) because:
//!
//! * Rust analysers are first-class function pointers (`BuiltinAnalyzer`),
//!   not Python callables. We cannot key the registry on the callable
//!   itself the way upstream does with `BUILTIN_ANALYZERS[__builtin__.range]`.
//! * `HOST_ENV` installs every builtin/rlib callable under its fully
//!   qualified name (see `flowspace::model::HOST_ENV::bootstrap_*`), so
//!   the qualname is an injective identity for the callable.
//!
//! # Registry
//!
//! A single `OnceLock<HashMap<String, BuiltinAnalyzer>>` owns the
//! registry. First access calls [`register_builtins`] which replicates
//! upstream's mass `for name, value in globals().items(): ...` loop
//! plus every `@analyzer_for(...)` decoration site.
//!
//! # Dispatch
//!
//! [`call_builtin`] is the entry point consumed by
//! `SomeValue::call()` when the callee is `SomeValue::Builtin(_)`. It
//! looks the analyser up by `SomeBuiltin.analyser_name` (which equals
//! the qualname the bookkeeper stored at immutablevalue time), unpacks
//! the call arguments per upstream's `kwds_s['s_'+key] = s_value`
//! rewrite, and hands off to the resolved Rust function.
//!
//! # Deferred entries
//!
//! A few upstream analysers require rtyper-phase machinery that has
//! not landed yet. They are registered but return
//! `AnnotatorError::new("<name>: rtyper-phase only")` so
//! `simplify_graph` paths that accidentally dispatch through them fail
//! loudly instead of silently annotating as `SomeObject`:
//!
//! * `rpython.rlib.objectmodel.hlinvoke` — needs
//!   `rpython.rtyper.llannotation.lltype_to_annotation` and
//!   `rpython.rtyper.rmodel.Repr`.
//! * `unicodedata.decimal` — deliberately errors at annotation time
//!   (matches upstream `builtin.py:300-303`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use super::bookkeeper::Bookkeeper;
use super::classdesc::{ClassDef, ClassDesc};
use super::description::DescEntry;
use super::model::{
    AnnotatorError, SomeBool, SomeByteArray, SomeChar, SomeDict, SomeFloat, SomeInstance,
    SomeInteger, SomeIterator, SomeObjectTrait, SomeString, SomeTuple, SomeUnicodeCodePoint,
    SomeUnicodeString, SomeValue, SomeWeakRef, s_impossible_value, s_none, union, unionof,
};
use crate::flowspace::model::ConstValue;

// Registry.

/// Signature of a builtin analyser.
///
/// * `bk` — active bookkeeper. Every RPython analyser that needs to
///   call `newlist` / `getdictdef` / `immutablevalue` goes through this
///   reference. Upstream pulls the same value via `getbookkeeper()`;
///   passing it explicitly avoids repeated TLS reads.
/// * `args_s` — positional argument annotations, after upstream's
///   `args_s, kwds = args.unpack()` split.
/// * `kwds_s` — keyword argument annotations, already `s_` prefixed to
///   match upstream's `kwds_s['s_'+key] = s_value`. Most analysers do
///   not consume keywords.
///   `SomeBuiltin.call` in `unaryop.py`: `args_s, kwds = args.unpack();
/// return self.analyser(*args_s, **kwds_s)` — args_s passes through as a
///   Python list possibly carrying `None` for unbound caller args.  The
///   analyser body raises AttributeError on the first
///   `s_arg.knowntype` / `.is_constant()` access if it touches an
///   unbound slot.  The Rust port mirrors this by taking
///   `&[Option<SomeValue>]`: each analyser body unwraps via [`arg_at`]
///   at the touch site, panicking with the analyser name and slot
///   index — observationally equivalent to upstream's first-attribute-
///   touch failure, matching `simple_call_SomeBuiltin`'s
///   bind-then-body sequence.
pub type BuiltinAnalyzer = fn(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError>;

/// Per-slot access helper: return `&SomeValue` at index `i`, panicking
/// on `None` to mirror upstream's `args_s[i].knowntype` AttributeError
/// surface at the moment of first attribute touch.  Bodies use this at
/// the access site so an unannotated optional-default slot can still
/// be observed via `args_s.get(i)`-style arity probing before the
/// panic fires.
pub(crate) fn arg_at<'a>(
    args_s: &'a [Option<SomeValue>],
    i: usize,
    analyser: &str,
) -> &'a SomeValue {
    args_s.get(i).and_then(|s| s.as_ref()).unwrap_or_else(|| {
        panic!(
            "{analyser}: positional arg {i} is unannotated \
                 (upstream `args_s[{i}].knowntype` AttributeError)",
        )
    })
}

// (Retired 2026-05-12, strict-parity pass.)  The
// `args_s_concrete_or_panic` eager-prefix helper was a deviation
// from `unaryop.py call simple_call_SomeBuiltin`'s bind-then-body
// sequence (upstream's analyser body raises `AttributeError` on the
// FIRST slot touch, not at a blanket prefix unwrap).  Every
// `builtin_*` analyser in this file now uses [`arg_at`] at the touch
// site directly; per-iteration `arg_at` covers helpers like
// `union_many` / `r_dict_helper` that previously consumed the prefix
// buffer.  Rtyper-phase callers (`SomePtr.call` / `SomeLLPtr.call`)
// touch each slot via `annotation_to_lltype` in a list comprehension
// — those now drive `arg_at` per iteration to mirror upstream
// `lltype.py` AttributeError semantics.

/// Process-wide `BUILTIN_ANALYZERS` table, lazily populated on first
/// access by [`register_builtins`]. Mirrors upstream's module-level
/// `BUILTIN_ANALYZERS = {}` dict (bookkeeper.py).
static BUILTIN_ANALYZERS: OnceLock<HashMap<String, BuiltinAnalyzer>> = OnceLock::new();

/// Lazy accessor for the analyser table. Calls [`register_builtins`]
/// the first time it is hit.
fn analyzers() -> &'static HashMap<String, BuiltinAnalyzer> {
    BUILTIN_ANALYZERS.get_or_init(register_builtins)
}

/// `True` iff the given qualname has a registered analyser.
///
/// Mirrors upstream's `x in BUILTIN_ANALYZERS` membership test at
/// `bookkeeper.py` and `classdesc.py:632`.  allow-line-citation
pub(crate) fn is_registered(qualname: &str) -> bool {
    analyzers().contains_key(qualname)
}

/// Upstream `BUILTIN_ANALYZERS[x]` read. Returns `None` when `qualname`
/// is not registered.
fn lookup(qualname: &str) -> Option<BuiltinAnalyzer> {
    analyzers().get(qualname).copied()
}

/// Per-analyser allow-list of `s_`-prefixed keyword argument names.
///
/// Upstream's dispatch is `self.analyser(*args_s, **kwds_s)`, which
/// raises `TypeError` when an unknown keyword reaches the analyser's
/// Python signature. The Rust dispatcher must reject the same cases
/// before calling the function pointer, so each analyser declares the
/// set of keywords it accepts here. The default (missing entry) means
/// "no keywords accepted".
fn allowed_kwds(qualname: &str) -> &'static [&'static str] {
    match qualname {
        // builtin_int(s_obj, s_base=None)
        "int" => &["s_base"],
        // builtin_enumerate(s_obj, s_start=None)
        "enumerate" => &["s_start"],
        // robjmodel_instantiate(s_clspbc, s_nonmovable=None)
        "rpython.rlib.objectmodel.instantiate" => &["s_nonmovable"],
        // _r_dict_helper(cls, s_eqfn, s_hashfn, s_force_non_null,
        //                s_simple_hash_eq)
        "rpython.rlib.objectmodel.r_dict" | "rpython.rlib.objectmodel.r_ordereddict" => {
            &["s_force_non_null", "s_simple_hash_eq"]
        }
        _ => &[],
    }
}

/// Upstream `SomeBuiltin.call(self, args)` dispatcher
/// (unaryop.py).
///
/// ```python
/// def call(self, args, implicit_init=False):
///     args_s, kwds = args.unpack()
///     kwds_s = {}
///     for key, s_value in kwds.items():
///         kwds_s['s_'+key] = s_value
///     return self.analyser(*args_s, **kwds_s)
/// ```
///
/// In addition to the direct dispatch, the Rust port rejects keyword
/// arguments whose `s_<name>` is not in [`allowed_kwds`] for this
/// `analyser_name`. Upstream gets that check for free from Python's
/// signature binding; the Rust dispatcher emulates it so calls like
/// `list(xs, foo=1)` or `bool(x, foo=1)` fail cleanly instead of
/// silently dropping `foo=1`.
pub(crate) fn call_builtin(
    bk: &Rc<Bookkeeper>,
    analyser_name: &str,
    args_s: &[Option<SomeValue>],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    let Some(analyser) = lookup(analyser_name) else {
        return Err(AnnotatorError::new(format!(
            "SomeBuiltin.call(): no analyser registered for {analyser_name}"
        )));
    };
    let allowed = allowed_kwds(analyser_name);
    for key in kwds_s.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(AnnotatorError::new(format!(
                "{analyser_name}() got an unexpected keyword argument {key:?}"
            )));
        }
    }
    analyser(bk, args_s, kwds_s)
}

/// Upstream `for name, value in globals().items(): if name.startswith('builtin_')` loop
/// (builtin.py:191-195) plus every `@analyzer_for(...)` decoration.
///
/// `__builtin__` aliases like `xrange = builtin_range` are registered
/// explicitly — upstream's `builtin_xrange = builtin_range`
/// assignment (builtin.py:86) places both names into
/// `BUILTIN_ANALYZERS` via the mass scan.
fn register_builtins() -> HashMap<String, BuiltinAnalyzer> {
    use super::bookkeeper::analyzer_for;

    let mut reg: HashMap<String, BuiltinAnalyzer> = HashMap::new();
    // builtin.py — `builtin_range` + `builtin_xrange` alias.
    analyzer_for(&mut reg, "range", builtin_range);
    analyzer_for(&mut reg, "xrange", builtin_range);
    // builtin.py:89-99 — enumerate / reversed.
    analyzer_for(&mut reg, "enumerate", builtin_enumerate);
    analyzer_for(&mut reg, "reversed", builtin_reversed);
    // builtin.py:102-130 — bool / int / float / chr / unichr /
    // unicode / bytearray.
    analyzer_for(&mut reg, "bool", builtin_bool);
    analyzer_for(&mut reg, "int", builtin_int);
    analyzer_for(&mut reg, "float", builtin_float);
    analyzer_for(&mut reg, "chr", builtin_chr);
    analyzer_for(&mut reg, "unichr", builtin_unichr);
    analyzer_for(&mut reg, "unicode", builtin_unicode);
    analyzer_for(&mut reg, "bytearray", builtin_bytearray);
    // builtin.py:133-148 — hasattr.
    analyzer_for(&mut reg, "hasattr", builtin_hasattr);
    // builtin.py:151-188 — tuple / list / zip / min / max.
    analyzer_for(&mut reg, "tuple", builtin_tuple);
    analyzer_for(&mut reg, "list", builtin_list);
    analyzer_for(&mut reg, "zip", builtin_zip);
    analyzer_for(&mut reg, "min", builtin_min);
    analyzer_for(&mut reg, "max", builtin_max);

    // `@analyzer_for(...)` decoration sites.

    // builtin.py:198-214 — object.__init__ / EnvironmentError.__init__ /
    // WindowsError.__init__ — builtin-exception-class skip gate reads
    // these names in classdesc.rs.
    analyzer_for(&mut reg, "object.__init__", object_init);
    analyzer_for(&mut reg, "EnvironmentError.__init__", EnvironmentError_init);
    // `WindowsError` exists only on the `win32` build; upstream wraps
    // the registration in `try: WindowsError; except NameError: pass`.
    // We register it unconditionally so cross-platform annotation runs
    // that exercise stubs carrying `WindowsError` classes succeed
    // identically.
    analyzer_for(&mut reg, "WindowsError.__init__", WindowsError_init);

    // builtin.py:217-293 — sys / rarithmetic / objectmodel helpers.
    analyzer_for(&mut reg, "sys.getdefaultencoding", conf);
    // Analyzer keys mirror the `HostObject::new_builtin_callable`
    // qualname that `flowspace/model.rs::bootstrap_std_modules`
    // assigns ("rarithmetic.intmask" etc.) so the SomeBuiltin-branch gate
    // in `bookkeeper.rs`'s `immutablevalue_hostobject` matches — keying under the full dotted
    // module path silently routes the value to the
    // extregistry / generic fallback path.  The doc-comment paths in
    // each analyser body still reference upstream `rbuiltin.py:221`
    // etc., which is the source-of-truth for the per-callable
    // semantics.
    analyzer_for(&mut reg, "rarithmetic.intmask", rarith_intmask);
    analyzer_for(&mut reg, "rarithmetic.longlongmask", rarith_longlongmask);
    analyzer_for(&mut reg, "rarithmetic.ulonglongmask", rarith_ulonglongmask);
    analyzer_for(
        &mut reg,
        "rarithmetic.longlonglongmask",
        rarith_longlonglongmask,
    );
    analyzer_for(
        &mut reg,
        "rarithmetic.ulonglonglongmask",
        rarith_ulonglonglongmask,
    );
    // `lltype.cast_pointer` (ann_cast_pointer, lltype.py) —
    // keyed under the `HostObject::new_builtin_callable` qualname the
    // lltype HOST_ENV module assigns (`flowspace/model.rs`'s
    // `bootstrap_std_modules`).
    analyzer_for(&mut reg, "lltype.cast_pointer", lltype_cast_pointer);
    // `pyre_object::lltype::malloc[_typed]` — GC allocation intrinsics.
    // Keyed by the qualname the HOST_ENV `pyre_object.lltype` module assigns
    // (`flowspace/model.rs`); recognising it as a builtin keeps its body out
    // of the looked-inside set (the `<T as GcType>::type_id()` accessor wall).
    analyzer_for(
        &mut reg,
        crate::runtime_names::modules::MALLOC,
        malloc_typed_alloc,
    );
    analyzer_for(
        &mut reg,
        crate::runtime_names::modules::MALLOC_TYPED,
        malloc_typed_alloc,
    );
    analyzer_for(
        &mut reg,
        crate::runtime_names::modules::MALLOC_TYPED_MANAGED,
        malloc_typed_alloc,
    );
    analyzer_for(
        &mut reg,
        crate::runtime_names::modules::MALLOC_TYPED_STABLE,
        malloc_typed_alloc,
    );
    // `pyre_object::lltype::malloc_raw` — the raw (non-GC) allocation
    // intrinsic (`lltype.malloc(T, flavor='raw')` parity).  Recognising it
    // as a builtin keeps its `Box::new` / `Box::into_raw` body out of the
    // looked-inside set (neither has an analyser), the same boundary the
    // `malloc_typed` registration draws around `GcType::type_id`.
    analyzer_for(
        &mut reg,
        crate::runtime_names::modules::MALLOC_RAW,
        malloc_raw_alloc,
    );
    // Rust `std::ptr::eq(p, q) -> bool` — registered under the dotted
    // qualname that `HostEnv::bootstrap` assigns to the HOST_ENV stub
    // (`flowspace/model.rs`'s `bootstrap_std_modules`).  Lowers identity
    // checks in
    // `baseobjspace::is_w` / `baseobjspace::is_`.
    analyzer_for(&mut reg, "std.ptr.eq", std_ptr_eq);
    // `core.ptr.copy_nonoverlapping` is the same `HostObject` as
    // `std.ptr.copy_nonoverlapping` (`flowspace/model.rs`), so this
    // qualname covers both spellings.
    analyzer_for(
        &mut reg,
        "std.ptr.copy_nonoverlapping",
        std_ptr_copy_nonoverlapping,
    );
    // Rust `std::ptr::null_mut::<T>()` / `null::<T>()` — the null
    // pointer constant (also the value of
    // `pyre_object::pyobject::PY_NULL`, whose HOST_ENV attr binds to
    // the `core.ptr.null_mut` callable).  All four spellings share one
    // analyzer; `core.*` is the canonical form MIR lowering emits and
    // `std.*` the re-export spelling.
    analyzer_for(&mut reg, "core.ptr.null_mut", ptr_null_constant);
    analyzer_for(&mut reg, "std.ptr.null_mut", ptr_null_constant);
    analyzer_for(&mut reg, "core.ptr.null", ptr_null_constant);
    analyzer_for(&mut reg, "std.ptr.null", ptr_null_constant);
    // `Option<fn>`'s null niche. A GC pointer's null is
    // `ptr_null_constant` (classdef-less `SomeInstance`); a raw function
    // pointer's null is `SomePtr(FuncType)` (`fn_null_constant`), the
    // same `ll_ptrtype` as the matching `fn` value so the two arms union.
    analyzer_for(&mut reg, "core.ptr.null_fn", fn_null_constant);
    // Rust `std::mem::size_of::<T>() -> usize` — compile-time type-size
    // constant called by `lltype::malloc_typed` /
    // `object_array::items_block_layout`.  Returns `SomeInteger`.
    analyzer_for(&mut reg, "std.mem.size_of", std_mem_size_of);
    // Rust `std::mem::align_of::<T>() -> usize` — same compile-time
    // `usize` lattice as `size_of`; called by `object_array.rs` /
    // `pyframe.rs`.
    analyzer_for(&mut reg, "std.mem.align_of", std_mem_align_of);
    // `__cast_instance_intrinsic` — front-end pointer-downcast narrow (#298).
    // `obj as *const RegisteredStruct` lowers to a simple_call against
    // this stub so the classdef-less pointer narrows to
    // `SomeInstance(root)` and a field read on the pointee resolves.
    // Keyed by the qualname `flowspace/model.rs`'s HOST_ENV bootstrap
    // assigns (which also keys the `BUILTIN_TYPER` entry on the same Arc).
    analyzer_for(
        &mut reg,
        crate::runtime_names::shims::CAST_INSTANCE,
        cast_instance_intrinsic,
    );
    // `__cast_address_intrinsic` — the erasing twin.  `p as *mut u8` lowers to
    // a simple_call against this stub so the pointee class is dropped
    // instead of riding along on a value whose declared type is a bare
    // byte pointer.
    analyzer_for(
        &mut reg,
        crate::runtime_names::shims::CAST_ADDRESS,
        cast_address_intrinsic,
    );
    analyzer_for(
        &mut reg,
        crate::runtime_names::shims::LL_ARRAYMOVE,
        ll_arraymove_intrinsic,
    );
    // Rust `String::new()` / `String::with_capacity(n)` construct an empty
    // owned UTF-8 buffer; both annotate as a mutable `SomeString` so the
    // `String.new` / `String.with_capacity` HOST_ENV stubs do not reach the
    // "no analyser registered" error (the formatting helpers in
    // `type_methods.rs` call them).
    analyzer_for(&mut reg, "String.new", string_constructor);
    analyzer_for(&mut reg, "String.with_capacity", string_constructor);
    // External crate `majit_metainterp` bool flag helpers — both
    // qualnames share the same analyzer (returns `SomeBool`).
    analyzer_for(
        &mut reg,
        "majit_metainterp.majit_log_enabled",
        majit_metainterp_bool_flag,
    );
    analyzer_for(
        &mut reg,
        "majit_metainterp.jit.we_are_jitted",
        majit_metainterp_bool_flag,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.we_are_jitted",
        majit_metainterp_bool_flag,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.isconstant",
        majit_metainterp_bool_flag,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.isvirtual",
        majit_metainterp_bool_flag,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.nonconst.non_constant",
        nonconst_non_constant,
    );
    // `rlib/jit.py conditional_call` returns None. One analyzer covers
    // every residual arity (`conditional_call0`..`conditional_call4`).
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call0",
        jit_conditional_call,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call1",
        jit_conditional_call,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call2",
        jit_conditional_call,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call3",
        jit_conditional_call,
    );
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call4",
        jit_conditional_call,
    );
    // `rlib/jit.py ConditionalCallEntry.compute_result_annotation` for
    // `_jit_conditional_call_value`.
    analyzer_for(
        &mut reg,
        "majit_rlib.jit.conditional_call_elidable1",
        jit_conditional_call_elidable,
    );
    // `rlib/jit.py` — the `hint` ExtRegistryEntry's
    // `compute_result_annotation`, the one place upstream MINTS
    // `access_directly` onto an annotation. Upstream spells the kwargs on a
    // single `hint(x, **kwds)`; pyre dispatches one helper per kwarg, so the
    // branch on which kwargs are present becomes the choice of analyzer.
    analyzer_for(
        &mut reg,
        "majit_metainterp.jit.hint_access_directly",
        jit_hint_access_directly,
    );
    analyzer_for(
        &mut reg,
        "majit_metainterp.jit.hint_fresh_virtualizable",
        jit_hint_fresh_virtualizable,
    );
    analyzer_for(
        &mut reg,
        "majit_metainterp.jit.hint_no_access_directly",
        jit_hint_no_access_directly,
    );
    // `rlib/debug.py` — the assertion half of `check_not_access_directly`.
    analyzer_for(
        &mut reg,
        "majit_rlib.debug.check_not_access_directly",
        rlib_check_not_access_directly,
    );
    // Rust primitive type conversion impls — all return `SomeInteger`.
    analyzer_for(&mut reg, "u32.from", primitive_integer_conversion);
    analyzer_for(&mut reg, "i64.from", primitive_integer_conversion);
    analyzer_for(&mut reg, "i64.try_from", primitive_integer_conversion);
    analyzer_for(&mut reg, "usize.try_from", primitive_integer_conversion);
    // `BigInt::from(i64)` — boxes a machine int into the foreign opaque
    // `BigInt` (Python `long`); returns the classdef-less GcRef shell.
    analyzer_for(&mut reg, "BigInt.from", bigint_from);
    analyzer_for(
        &mut reg,
        "longlong2float.float2longlong",
        float2longlong_analyzer,
    );
    analyzer_for(
        &mut reg,
        "longlong2float.longlong2float",
        longlong2float_analyzer,
    );
    // Foreign Rust container constructors — `Vec::new` /
    // `Vec::with_capacity` / `indexmap::IndexMap::new` / `Box::new`.
    // Each returns the classdef-less opaque `SomeInstance` shell (twin
    // `bigint_from`), retiring the "no analyser registered" wall the
    // two-phase prepass hits when residualizing bodies that build these
    // containers (`OpcodeStepExecutor.build_*`, `NamespaceOpcodeHandler.
    // store_*`, `coerce_to_list_for_args`, `FrameBox::new_boxed`,
    // `malloc_raw`).  Keyed by the qualnames `HOST_ENV` bootstrap assigns
    // — note the mixed bare/dotted spellings the census demands.
    analyzer_for(&mut reg, "Vec.new", foreign_container_ctor);
    analyzer_for(&mut reg, "vec.Vec.with_capacity", foreign_container_ctor);
    analyzer_for(&mut reg, "indexmap.IndexMap.new", foreign_container_ctor);
    analyzer_for(&mut reg, "Box.new", foreign_container_ctor);
    // `Box::new_uninit` is the no-arg heap-box twin of `lltype.malloc`.
    analyzer_for(&mut reg, "boxed.Box.new_uninit", foreign_container_ctor);
    // `<[T]>::into_vec` producer — returns a fresh `Vec<T>`, the same
    // opaque container shell as `Vec::new`.
    analyzer_for(
        &mut reg,
        "alloc.boxed.box_assume_init_into_vec_unsafe",
        foreign_container_ctor,
    );
    // The WTF-8 buffer builder — a mutable `SomeString` like `String::*`.
    // (Crate-qualified `string.String.*` needs no row here: its `HostEnv`
    // carrier reuses the bare `String.new` / `with_capacity` callables, whose
    // `String.*` qualnames already resolve to the analyzers above.)
    analyzer_for(&mut reg, "Wtf8Buf.new", string_constructor);
    analyzer_for(&mut reg, "Wtf8Buf.with_capacity", string_constructor);
    // `rarithmetic.r_uint` is routed via
    // `ExtRegistryEntry::ForType` (rarithmetic.py `ForTypeEntry`):
    // bookkeeper's BUILTIN_ANALYZERS miss falls through to
    // `extregistry.lookup`, which returns the SomeBuiltin whose
    // `compute_result_annotation` produces
    // `SomeInteger(knowntype=Ruint, unsigned=True)`.
    // lltype.py — `@analyzer_for(cast_ptr_to_int)`,
    // `@analyzer_for(cast_int_to_ptr)`, `@analyzer_for(direct_ptradd)`.
    // `front::mir` lowers a `Ref ↔ Int` cast and a pointer-typed
    // `<*mut T>::add` to a `simple_call` against the matching
    // `lltype.*` HostObject, so the annotation pass routes through these
    // analyzers when that surface lands in user code.  Keyed by the
    // `qualname` that
    // `HostObject::new_builtin_callable("lltype.cast_ptr_to_int")`
    // produces (`bookkeeper.rs`'s `immutablevalue_hostobject` reads
    // `obj.qualname()` directly
    // for `SomeBuiltin.analyser_name`).
    analyzer_for(&mut reg, "lltype.cast_ptr_to_int", lltype_cast_ptr_to_int);
    analyzer_for(&mut reg, "lltype.cast_int_to_ptr", lltype_cast_int_to_ptr);
    analyzer_for(&mut reg, "lltype.direct_ptradd", lltype_direct_ptradd);
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.instantiate",
        robjmodel_instantiate,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.r_dict",
        robjmodel_r_dict,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.r_ordereddict",
        robjmodel_r_ordereddict,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.hlinvoke",
        robjmodel_hlinvoke,
    );
    analyzer_for(
        &mut reg,
        "objectmodel.keepalive_until_here",
        robjmodel_keepalive_until_here,
    );
    analyzer_for(
        &mut reg,
        "objectmodel.free_non_gc_object",
        robjmodel_free_non_gc_object,
    );

    // builtin.py:295-307 — unicodedata / OrderedDict.
    analyzer_for(&mut reg, "unicodedata.decimal", unicodedata_decimal);
    analyzer_for(&mut reg, "collections.OrderedDict", analyze);

    // builtin.py:314-321 — weakref.ref.
    analyzer_for(&mut reg, "weakref.ref", weakref_ref);

    // builtin.py:335-339 — pdb.set_trace.
    analyzer_for(&mut reg, "pdb.set_trace", pdb_set_trace);

    reg
}

// Helpers.

/// Upstream `constpropagate(func, args_s, s_result)` (builtin.py).
///
/// ```python
/// def constpropagate(func, args_s, s_result):
///     args = []
///     for s in args_s:
///         if not s.is_immutable_constant():
///             return s_result
///         args.append(s.const)
///     try:
///         realresult = func(*args)
///     except (ValueError, OverflowError):
///         return s_result
///     s_realresult = immutablevalue(realresult)
///     if not s_result.contains(s_realresult):
///         raise AnnotatorError(...)
///     return s_realresult
/// ```
///
/// `evaluator` models Python's `func(*args)` — it returns `None` when
/// upstream would have caught `ValueError` / `OverflowError`.
pub fn constpropagate<F>(
    bk: &Rc<Bookkeeper>,
    args_s: &[SomeValue],
    s_result: SomeValue,
    evaluator: F,
) -> Result<SomeValue, AnnotatorError>
where
    F: FnOnce(&[&ConstValue]) -> Option<ConstValue>,
{
    let mut consts: Vec<&ConstValue> = Vec::with_capacity(args_s.len());
    for s in args_s {
        if !s.is_immutable_constant() {
            return Ok(s_result);
        }
        let Some(c) = s.const_() else {
            return Ok(s_result);
        };
        consts.push(c);
    }
    let Some(realresult) = evaluator(&consts) else {
        return Ok(s_result);
    };
    let s_realresult = bk.immutablevalue(&realresult)?;
    if !s_result.contains(&s_realresult) {
        return Err(AnnotatorError::new(format!(
            "constpropagate: {realresult} is not contained in s_result"
        )));
    }
    Ok(s_realresult)
}

/// Upstream `SomeObject.is_immutable_constant` — `True` iff
/// `immutable and is_constant()`. Kept as a thin free helper so the
/// per-analyser sites read like upstream `s.is_immutable_constant()`.
fn is_immutable_constant(s: &SomeValue) -> bool {
    s.immutable() && s.is_constant()
}

/// Upstream `SomeInteger.nonneg` projection — returns `True` iff the
/// annotation is a `SomeInteger` with `nonneg=True`. Other annotations
/// default to `False`.
fn nonneg_of(s: &SomeValue) -> bool {
    match s {
        SomeValue::Integer(i) => i.nonneg,
        _ => false,
    }
}

/// Upstream `s.knowntype == str` projection — matches both `SomeString`
/// and `SomeChar` since both carry `knowntype=str`. Used by
/// [`builtin_int`] to approve `int(string, base)` syntax.
fn is_str_annotation(s: &SomeValue) -> bool {
    matches!(s, SomeValue::String(_) | SomeValue::Char(_))
}

// `builtin_*` analysers (mass-registered via the `builtin_` prefix scan).

/// Upstream `builtin_range(*args)` (builtin.py).
pub fn builtin_range(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // Arity check fires first per `builtin.py:49-84` shape; per-slot
    // access defers through `arg_at` so a `None` slot surfaces as
    // "unannotated arg" at the touch site (matching `unaryop.py call
    // simple_call_SomeBuiltin`'s bind-then-body sequence) instead of
    // the entry-point prefix unwrap.
    let (s_start, s_stop, s_step) = match args_s.len() {
        1 => (
            bk.immutablevalue(&ConstValue::Int(0))?,
            arg_at(args_s, 0, "builtin_range").clone(),
            bk.immutablevalue(&ConstValue::Int(1))?,
        ),
        2 => (
            arg_at(args_s, 0, "builtin_range").clone(),
            arg_at(args_s, 1, "builtin_range").clone(),
            bk.immutablevalue(&ConstValue::Int(1))?,
        ),
        3 => (
            arg_at(args_s, 0, "builtin_range").clone(),
            arg_at(args_s, 1, "builtin_range").clone(),
            arg_at(args_s, 2, "builtin_range").clone(),
        ),
        _ => {
            return Err(AnnotatorError::new("range() takes 1 to 3 arguments"));
        }
    };

    let mut empty = false;
    let step: i64;
    if !is_immutable_constant(&s_step) {
        // upstream `step = 0` signals "variable step".
        step = 0;
    } else {
        step = match s_step.const_() {
            Some(ConstValue::Int(n)) => *n,
            Some(ConstValue::Bool(b)) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            _ => {
                return Err(AnnotatorError::new(
                    "range() with non-integer constant step",
                ));
            }
        };
        if step == 0 {
            return Err(AnnotatorError::new("range() with step zero"));
        }
        if is_immutable_constant(&s_start)
            && is_immutable_constant(&s_stop)
            && let (Some(ConstValue::Int(start)), Some(ConstValue::Int(stop))) =
                (s_start.const_(), s_stop.const_())
        {
            // upstream `if len(xrange(start, stop, step)) == 0:`.
            let range_len: i64 = if step > 0 {
                if *stop > *start { 1 } else { 0 }
            } else if *stop < *start {
                1
            } else {
                0
            };
            if range_len == 0 {
                empty = true;
            }
        }
    }

    let s_item = if empty {
        s_impossible_value()
    } else {
        // upstream: `if step > 0 or s_step.nonneg: nonneg = s_start.nonneg`.
        // `elif step < 0: nonneg = s_stop.nonneg or (s_stop.is_constant()
        //                                            and s_stop.const >= -1)`.
        let nonneg = if step > 0 || nonneg_of(&s_step) {
            nonneg_of(&s_start)
        } else if step < 0 {
            nonneg_of(&s_stop) || matches!(s_stop.const_(), Some(ConstValue::Int(n)) if *n >= -1)
        } else {
            false
        };
        SomeValue::Integer(SomeInteger::new(nonneg, false))
    };

    let s_list = bk.newlist(&[s_item], Some(step))?;
    Ok(SomeValue::List(s_list))
}

/// Upstream `builtin_enumerate(s_obj, s_start=None)` (builtin.py).
pub fn builtin_enumerate(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream: arity check first — `s_start=None` default means a
    // 1-arg call binds `s_start` to Python None.  Defer per-slot
    // access through `arg_at` so the unwrap surfaces at first
    // attribute touch (matching upstream's `s_start.is_constant()`
    // AttributeError) rather than a blanket entry-point panic.
    let (s_obj, s_start_positional) = match args_s.len() {
        1 => (arg_at(args_s, 0, "builtin_enumerate").clone(), None),
        2 => (
            arg_at(args_s, 0, "builtin_enumerate").clone(),
            Some(arg_at(args_s, 1, "builtin_enumerate").clone()),
        ),
        _ => {
            return Err(AnnotatorError::new("enumerate() takes 1 or 2 arguments"));
        }
    };
    // upstream's Python signature binding binds `start` to whichever of
    // positional / keyword was supplied, and raises TypeError on both.
    if s_start_positional.is_some() && kwds.contains_key("s_start") {
        return Err(AnnotatorError::new(
            "enumerate() got multiple values for argument 'start'",
        ));
    }
    // upstream `Argument(start=...)` binds the value as-is (including
    // `None` from `annotator.annotation(v) is None`); pyre's
    // `kwds.get(...).cloned()` returns `Option<Option<SomeValue>>` so
    // `.flatten()` collapses to the Option<SomeValue> form the body
    // consumes via `s_start.as_ref()` (None tolerated).
    let s_start = s_start_positional.or_else(|| kwds.get("s_start").cloned().flatten());
    if let Some(s_start) = s_start.as_ref()
        && !s_start.is_constant()
    {
        return Err(AnnotatorError::new(
            "second argument to enumerate must be constant",
        ));
    }
    let const_start = s_start.and_then(|s| s.const_().cloned());
    Ok(SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
        s_obj,
        vec!["enumerate".to_string()],
        const_start,
    )))
}

/// Upstream `builtin_reversed(s_obj)` (builtin.py).
pub fn builtin_reversed(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("reversed() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_reversed");
    Ok(SomeValue::Iterator(SomeIterator::new(
        s_obj.clone(),
        vec!["reversed".to_string()],
    )))
}

/// Upstream `builtin_bool(s_obj)` (builtin.py).
///
/// Upstream dispatches `s_obj.bool()` which is registered per-`Some*`
/// in `unaryop.py`. The annotator-phase subset used by `builtin_bool`
/// reduces to:
///
/// * constant immutable receivers → `SomeBool(bool(s_obj.const))`
/// * empty-tuple literal `()` → `SomeBool(False)` (unaryop.py)
/// * everything else → unrefined `SomeBool()`
///
/// Knowntypedata propagation (`SomeObject._propagate_knowntypedata`) is
/// an analyser-driven optimisation and is handled by the OpKind::Bool
/// dispatch in `unaryop.rs`; the annotator-phase `builtin_bool` path
/// used by `call_builtin` does not have the HLOperation context to
/// track that side channel and leaves the per-branch refinements to
/// the surrounding flow-level dispatch.
pub fn builtin_bool(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("bool() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_bool");
    let mut r = SomeBool::new();

    // upstream `SomeTuple.bool` (unaryop.py): empty tuple is
    // always falsy.
    if let SomeValue::Tuple(t) = s_obj
        && t.items.is_empty()
    {
        r.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::Bool(false),
        ));
        return Ok(SomeValue::Bool(r));
    }

    // upstream default + primitive overrides: when the operand is a
    // constant immutable value, pin `r.const = bool(value)`.
    if is_immutable_constant(s_obj)
        && let Some(c) = s_obj.const_()
    {
        let truthy = match c {
            ConstValue::Bool(b) => *b,
            ConstValue::Int(n) => *n != 0,
            ConstValue::Int128(n) => *n != 0,
            ConstValue::UInt128(n) => *n != 0,
            ConstValue::Float(bits) => f64::from_bits(*bits) != 0.0,
            ConstValue::ByteStr(s) => !s.is_empty(),
            ConstValue::UniStr(s) => !s.is_empty(),
            ConstValue::None => false,
            ConstValue::Tuple(items) | ConstValue::List(items) => !items.is_empty(),
            ConstValue::Graphs(graphs) => !graphs.is_empty(),
            ConstValue::Dict(m) => !m.is_empty(),
            // HostObject / Function / Code etc. have no `__bool__`
            // override metadata in the Rust port, so take Python's
            // default `True`.
            ConstValue::HostObject(_)
            | ConstValue::Opaque(_)
            | ConstValue::Repr(_)
            | ConstValue::Function(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::LLPtr(_)
            | ConstValue::Code(_)
            | ConstValue::Atom(_)
            | ConstValue::SpecTag(_) => true,
            ConstValue::AddressOffset(_) => true,
            ConstValue::InheritanceId { .. } => true,
            ConstValue::LLAddress(addr) => matches!(
                addr,
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(_)
            ),
            // Placeholder never appears in production flow; stay
            // conservative rather than crash.
            ConstValue::Placeholder => return Ok(SomeValue::Bool(r)),
        };
        r.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::Bool(truthy),
        ));
    }
    Ok(SomeValue::Bool(r))
}

/// Upstream `builtin_int(s_obj, s_base=None)` (builtin.py).
pub fn builtin_int(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream signature: `int(s_obj, s_base=None)` — allow keyword
    // form for parity.  Arity check first so a 1-arg call lets
    // `s_base` default to None without touching slot 1.
    if args_s.is_empty() {
        return Err(AnnotatorError::new("int() takes at least one argument"));
    }
    if args_s.len() > 2 {
        return Err(AnnotatorError::new("int() takes at most 2 arguments"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_int").clone();
    if args_s.len() >= 2 && kwds.contains_key("s_base") {
        return Err(AnnotatorError::new(
            "int() got multiple values for argument 'base'",
        ));
    }
    let s_base = if args_s.len() == 2 {
        Some(arg_at(args_s, 1, "builtin_int").clone())
    } else {
        // upstream `Argument(base=...)` accepts `None` (unbound).
        // pyre's `kwds.get(...).cloned()` returns
        // `Option<Option<SomeValue>>`; `.flatten()` projects to
        // `Option<SomeValue>` to align with the positional arm.
        kwds.get("s_base").cloned().flatten()
    };

    if let SomeValue::Integer(ref i) = s_obj
        && i.unsigned
    {
        return Err(AnnotatorError::new(
            "instead of int(r_uint(x)), use intmask(r_uint(x))",
        ));
    }
    if let Some(ref sb) = s_base {
        let base_is_int = matches!(sb, SomeValue::Integer(i) if !i.unsigned);
        if !base_is_int || !is_str_annotation(&s_obj) {
            return Err(AnnotatorError::new(
                "only int(v|string) or int(string,int) expected",
            ));
        }
    }

    let nonneg = matches!(&s_obj, SomeValue::Integer(i) if i.nonneg);
    let s_result = SomeValue::Integer(SomeInteger::new(nonneg, false));

    let prop_args: Vec<SomeValue> = match &s_base {
        Some(sb) => vec![s_obj.clone(), sb.clone()],
        None => vec![s_obj.clone()],
    };
    constpropagate(bk, &prop_args, s_result, |consts| match consts {
        [ConstValue::Int(n)] => Some(ConstValue::Int(*n)),
        [ConstValue::Bool(b)] => Some(ConstValue::Int(if *b { 1 } else { 0 })),
        [s] if s.as_text().is_some() => s
            .as_text()
            .and_then(|text| text.parse::<i64>().ok())
            .map(ConstValue::Int),
        [ConstValue::Float(bits)] => {
            // upstream `int(float_const)` raises OverflowError when the
            // truncated value does not fit, which `constpropagate`
            // catches and falls back to `s_result` (builtin.py:34-39).
            // Reject non-finite floats and values that would saturate
            // an i64 cast, matching that behaviour instead of pinning a
            // wrong constant.
            let f = f64::from_bits(*bits);
            if !f.is_finite() {
                return None;
            }
            let truncated = f.trunc();
            if truncated < i64::MIN as f64 || truncated >= (i64::MAX as f64 + 1.0) {
                return None;
            }
            Some(ConstValue::Int(truncated as i64))
        }
        [s, ConstValue::Int(base)] if s.as_text().is_some() => {
            if *base >= 2 && *base <= 36 {
                i64::from_str_radix(s.as_text().unwrap().trim(), *base as u32)
                    .ok()
                    .map(ConstValue::Int)
            } else {
                None
            }
        }
        _ => None,
    })
}

/// Upstream `builtin_float(s_obj)` (builtin.py).
pub fn builtin_float(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("float() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_float");
    constpropagate(
        bk,
        std::slice::from_ref(s_obj),
        SomeValue::Float(SomeFloat::new()),
        |consts| match consts {
            [ConstValue::Int(n)] => Some(ConstValue::float(*n as f64)),
            [ConstValue::Bool(b)] => Some(ConstValue::float(if *b { 1.0 } else { 0.0 })),
            [ConstValue::Float(bits)] => Some(ConstValue::Float(*bits)),
            [s] if s.as_text().is_some() => s
                .as_text()
                .and_then(|text| text.trim().parse::<f64>().ok())
                .map(ConstValue::float),
            _ => None,
        },
    )
}

/// Upstream `builtin_chr(s_int)` (builtin.py).
pub fn builtin_chr(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("chr() takes exactly one argument"));
    }
    let s_int = arg_at(args_s, 0, "builtin_chr");
    constpropagate(
        bk,
        std::slice::from_ref(s_int),
        SomeValue::Char(SomeChar::new(false)),
        |consts| match consts {
            [ConstValue::Int(n)] => {
                if (0..=0xff).contains(n) {
                    Some(ConstValue::ByteStr(vec![*n as u8]))
                } else {
                    None
                }
            }
            _ => None,
        },
    )
}

/// Upstream `builtin_unichr(s_int)` (builtin.py).
///
/// Fold the constant case directly, mirroring upstream
/// `constpropagate(unichr, ...)` semantics while pinning the result's
/// `const_box` ourselves.
pub fn builtin_unichr(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("unichr() takes exactly one argument"));
    }
    let s_int = arg_at(args_s, 0, "builtin_unichr");
    let mut result = SomeUnicodeCodePoint::new(false);
    if is_immutable_constant(s_int)
        && let Some(ConstValue::Int(n)) = s_int.const_()
        && (0..=0x10_FFFF).contains(n)
        && let Some(ch) = u32::try_from(*n).ok().and_then(char::from_u32)
    {
        result.inner.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::uni_str(ch.to_string()),
        ));
    }
    Ok(SomeValue::UnicodeCodePoint(result))
}

/// Upstream `builtin_unicode(s_unicode)` (builtin.py).
///
/// Mirrors upstream `unicode(x)` while preserving the explicit
/// `ConstValue::UniStr` type tag.
pub fn builtin_unicode(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("unicode() takes exactly one argument"));
    }
    let s_unicode = arg_at(args_s, 0, "builtin_unicode");
    let mut result = SomeUnicodeString::new(false, false);
    if is_immutable_constant(s_unicode)
        && let Some(s) = s_unicode.const_().and_then(ConstValue::as_text)
    {
        result.inner.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::uni_str(s),
        ));
    }
    Ok(SomeValue::UnicodeString(result))
}

/// Upstream `builtin_bytearray(s_str)` (builtin.py).
pub fn builtin_bytearray(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // `builtin.py` `def builtin_bytearray(s_str):` — Python's
    // call binding (`unaryop.py simple_call_SomeBuiltin` →
    // `signature(builtin_bytearray).bind(...)`) raises `TypeError`
    // when arity disagrees.  Mirror the bind-time check; analyser
    // body still ignores the argument shape and returns
    // `SomeByteArray()`.
    if args_s.len() != 1 {
        return Err(AnnotatorError::new(format!(
            "bytearray() takes exactly one argument ({} given)",
            args_s.len()
        )));
    }
    Ok(SomeValue::ByteArray(SomeByteArray::new(false)))
}

/// Upstream `builtin_hasattr(s_obj, s_attr)` (builtin.py).
pub fn builtin_hasattr(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 2 {
        return Err(AnnotatorError::new("hasattr() takes exactly two arguments"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_hasattr");
    let s_attr = arg_at(args_s, 1, "builtin_hasattr");
    // builtin.py:133-136 — `if not s_attr.is_constant() or not
    // isinstance(s_attr.const, str): bookkeeper.warning("hasattr is not
    // RPythonic enough") ; return SomeBool()`. Upstream's `isinstance(x,
    // str)` is Python 2 bytes-only; pyre uses [`as_pystr`] so a unicode
    // attribute literal short-circuits to the non-constant SomeBool
    // path the same way upstream does. (We skip the warning because
    // the Rust bookkeeper has no warning channel yet.)
    let attr_is_const_str = s_attr.const_().and_then(ConstValue::as_pystr).is_some();
    if !s_attr.is_constant() || !attr_is_const_str {
        return Ok(SomeValue::Bool(SomeBool::new()));
    }
    let Some(attr_name) = s_attr.const_().and_then(ConstValue::as_pystr) else {
        return Ok(SomeValue::Bool(SomeBool::new()));
    };

    let mut r = SomeBool::new();
    if s_obj.is_immutable_constant() {
        // upstream: `r.const = hasattr(s_obj.const, s_attr.const)`.
        // Use the same host descriptor lookup as FrozenDesc, not a separate
        // dict-presence probe which can disagree with getattr.
        if let Some(ConstValue::HostObject(host)) = s_obj.const_() {
            let found = host_hasattr(host, attr_name)?;
            r.base.const_box = Some(super::super::flowspace::model::Constant::new(
                ConstValue::Bool(found),
            ));
        }
    } else if let SomeValue::PBC(pbc) = s_obj {
        // upstream: `isinstance(s_obj, SomePBC) and s_obj.getKind() is FrozenDesc`.
        if pbc.get_kind()? == super::model::DescKind::Frozen {
            // upstream (builtin.py:142-147):
            //
            //     answers = {}
            //     for d in s_obj.descriptions:
            //         answer = (d.s_read_attribute(s_attr.const) !=
            //                   s_ImpossibleValue)
            //         answers[answer] = True
            //     if len(answers) == 1:
            //         r.const, = answers
            let mut answers = indexmap::IndexMap::new();
            for desc in pbc.descriptions.values() {
                // s_read_attribute itself turns only AttributeError into
                // s_ImpossibleValue. Other failures propagate in builtin.py;
                // an unsupported native field is not a missing attribute.
                let result = desc.s_read_attribute(attr_name)?;
                let found = !matches!(result, SomeValue::Impossible);
                answers.insert(found, true);
            }
            if answers.len() == 1 {
                let only = *answers.keys().next().unwrap();
                r.base.const_box = Some(super::super::flowspace::model::Constant::new(
                    ConstValue::Bool(only),
                ));
            }
        }
    }
    Ok(SomeValue::Bool(r))
}

/// Upstream `builtin_tuple(s_iterable)` (builtin.py).
pub fn builtin_tuple(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("tuple() takes exactly one argument"));
    }
    let s_iterable = arg_at(args_s, 0, "builtin_tuple");
    if let SomeValue::Tuple(_) = s_iterable {
        return Ok(s_iterable.clone());
    }
    Err(AnnotatorError::new(
        "tuple(): argument must be another tuple",
    ))
}

/// Upstream `builtin_list(s_iterable)` (builtin.py).
pub fn builtin_list(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("list() takes exactly one argument"));
    }
    let s_iterable = arg_at(args_s, 0, "builtin_list");
    if let SomeValue::List(s_list) = s_iterable {
        // upstream: `s_iterable.listdef.offspring(bk)`.
        return Ok(SomeValue::List(s_list.listdef.offspring(bk, &[])?));
    }
    // upstream: `s_iter = s_iterable.iter(); return bk.newlist(s_iter.next())`.
    let s_iter = SomeIterator::new(s_iterable.clone(), vec![]);
    let s_item = someiterator_next_stub(bk, &s_iter);
    Ok(SomeValue::List(bk.newlist(&[s_item], None)?))
}

/// Upstream `builtin_zip(s_iterable1, s_iterable2)` (builtin.py).
pub fn builtin_zip(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 2 {
        return Err(AnnotatorError::new("zip() takes exactly two arguments"));
    }
    let s_iter1 = SomeIterator::new(arg_at(args_s, 0, "builtin_zip").clone(), vec![]);
    let s_iter2 = SomeIterator::new(arg_at(args_s, 1, "builtin_zip").clone(), vec![]);
    let s_tup = SomeValue::Tuple(SomeTuple::new(vec![
        someiterator_next_stub(bk, &s_iter1),
        someiterator_next_stub(bk, &s_iter2),
    ]));
    Ok(SomeValue::List(bk.newlist(&[s_tup], None)?))
}

/// Upstream `builtin_min(*s_values)` (builtin.py).
pub fn builtin_min(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("min() takes at least one argument"));
    }
    if args_s.len() == 1 {
        // upstream: `s_iter = s_values[0].iter(); return s_iter.next()`.
        let s_iter = SomeIterator::new(arg_at(args_s, 0, "builtin_min").clone(), vec![]);
        return Ok(someiterator_next_stub(bk, &s_iter));
    }
    // upstream: `return union(*s_values)`.
    union_many(args_s, "builtin_min")
}

/// Upstream `builtin_max(*s_values)` (builtin.py).
pub fn builtin_max(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("max() takes at least one argument"));
    }
    if args_s.len() == 1 {
        let s_iter = SomeIterator::new(arg_at(args_s, 0, "builtin_max").clone(), vec![]);
        return Ok(someiterator_next_stub(bk, &s_iter));
    }
    let mut s = union_many(args_s, "builtin_max")?;
    // upstream (builtin.py:182-188):
    //
    //     if type(s) is SomeInteger and not s.nonneg:
    //         nonneg = False
    //         for s1 in s_values:
    //             nonneg |= s1.nonneg
    //         if nonneg:
    //             s = SomeInteger(nonneg=True, knowntype=s.knowntype)
    //
    // `nonneg |= s1.nonneg` is an OR-fold, not AND — any single nonneg
    // operand wins because `max(nonneg, whatever) >= nonneg >= 0`.
    if let SomeValue::Integer(ref i) = s
        && !i.nonneg
    {
        // Per-touch revisit: `union_many` already touched every slot
        // above, so by this point each `args_s[i]` is guaranteed to be
        // `Some(..)` (otherwise the loop would have panicked).
        let any_nonneg = (0..args_s.len()).any(|i| nonneg_of(arg_at(args_s, i, "builtin_max")));
        if any_nonneg {
            // upstream: `SomeInteger(nonneg=True, knowntype=s.knowntype)`
            // — pin the existing knowntype to avoid widening r_uint to int.
            s = SomeValue::Integer(SomeInteger::new_with_knowntype(true, i.base.knowntype));
        }
    }
    Ok(s)
}

// `@analyzer_for(...)` analysers.

/// Upstream `object_init(s_self, *args)` (builtin.py).
pub fn object_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream returns `None` which `immutablevalue` wraps as s_None.
    Ok(s_none())
}

/// Upstream `EnvironmentError_init(s_self, *args)` (builtin.py).
#[allow(non_snake_case)]
pub fn EnvironmentError_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `WindowsError_init(s_self, *args)` (builtin.py).
#[allow(non_snake_case)]
pub fn WindowsError_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `conf()` (builtin.py).
pub fn conf(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::String(SomeString::new(false, false)))
}

/// Upstream `rarith_intmask(s_obj)` (builtin.py).
pub fn rarith_intmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::default()))
}

/// Upstream `rarith_longlongmask(s_obj)` (builtin.py).
pub fn rarith_longlongmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream: `SomeInteger(knowntype=rpython.rlib.rarithmetic.r_longlong)`.
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        false,
        crate::annotator::model::KnownType::LongLong,
    )))
}

/// `r_longlonglong` (`SignedLongLongLong`).
pub fn rarith_longlonglongmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        false,
        crate::annotator::model::KnownType::LongLongLong,
    )))
}

pub fn rarith_ulonglonglongmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        true,
        crate::annotator::model::KnownType::ULongLongLong,
    )))
}

pub fn rarith_ulonglongmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        true,
        crate::annotator::model::KnownType::ULongLong,
    )))
}

/// `longlong2float.float2longlong(floatval)` — upstream
/// `Float2LongLongEntry.compute_result_annotation` returns
/// `SomeInteger(knowntype=r_int64)` (== `r_longlong`).
pub fn float2longlong_analyzer(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        false,
        crate::annotator::model::KnownType::LongLong,
    )))
}

/// `longlong2float.longlong2float(llval)` — upstream
/// `LongLong2FloatEntry.compute_result_annotation` returns `SomeFloat`.
pub fn longlong2float_analyzer(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Float(Default::default()))
}

/// Analyzer for `std::ptr::eq(p1, p2) -> bool` — Rust pointer identity
/// check.  Registered against the `HostObject::new_builtin_callable
/// ("std.ptr.eq")` stub published by [`HostEnv::bootstrap`]
/// (`flowspace/model.rs`'s `bootstrap_std_modules`) so `baseobjspace::is_w` /
/// `baseobjspace::is_` (which lower the Python identity check to
/// `std::ptr::eq(w_one, w_two)`) annotate cleanly.  Mirrors PyPy's
/// `ptr_eq(p, q)` annotator: pure identity comparison returning
/// `SomeBool` regardless of operand types.
fn std_ptr_eq(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Bool(super::model::SomeBool::new()))
}

/// `std::ptr::copy_nonoverlapping(src, dst, count) -> ()`.
///
/// The foreign decl is opaque. A user graph for it has one input slot
/// per argument, and `mergeinputargs` unions every caller's pointee
/// into that slot. `llmemory.raw_memcopy` returns
/// nothing and does not read the pointee, so this analyzer does not
/// either: `()` is `s_None`.
fn std_ptr_copy_nonoverlapping(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 3 {
        return Err(AnnotatorError::new(
            "std.ptr.copy_nonoverlapping expects (src, dst, count)",
        ));
    }
    Ok(s_none())
}

/// Analyzer for `std::ptr::null_mut::<T>()` / `std::ptr::null::<T>()`
/// — the null pointer constant (`lltype.nullptr(T)` analogue;
/// `rbuiltin.py` typer folds it to a null `Ptr` constant).  Pyre's
/// erased-pointer projection keeps pointee classes off the annotation
/// (`annotation_state.rs` `Ref` table row), so the shell is the
/// classdef-less `SomeInstance` with `can_be_none=true`: the
/// `Instance(None) ∪ Instance(cd)` union arm (`model.rs`'s `union`) keeps
/// it joinable with any typed receiver while preserving the
/// nullability bit.
fn ptr_null_constant(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // `ptr::null()` / `ptr::null_mut()` are zero-arg; a call site
    // carrying arguments is a malformed lowering, not a null constant.
    if !args_s.is_empty() || !kwds.is_empty() {
        return Err(AnnotatorError::new(
            "ptr::null()/ptr::null_mut() take no arguments",
        ));
    }
    Ok(SomeValue::Instance(super::model::SomeInstance::new(
        None,
        true,
        std::collections::BTreeMap::new(),
    )))
}

/// Null `Option<fn>` — `lltype.nullptr(FuncType)` as `SomePtr`, not the
/// classdef-less `SomeInstance` that `ptr_null_constant` returns for a
/// GC pointer. `llannotation.py` `pairtype(SomePtr, SomePtr).union`
/// requires the same `ll_ptrtype` as the function-pointer field read.
/// An optional constant spelling is the FUNC type `lltype.nullptr` takes
/// (`getfunctionptr`); without it the coarse empty `FuncType` is used.
fn fn_null_constant(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() {
        return Err(AnnotatorError::new("ptr::null_fn() takes no keywords"));
    }
    match args_s {
        [] => Ok(super::bookkeeper::raw_fn_ptr_somevalue()),
        [Some(s_ty)] => {
            let Some(spelling) = s_ty.const_().and_then(|c| c.as_pystr()) else {
                return Err(AnnotatorError::new(
                    "ptr::null_fn() type argument must be a constant string",
                ));
            };
            super::bookkeeper::fn_ptr_somevalue_for_spelling_in(spelling, Some(bk)).ok_or_else(
                || {
                    AnnotatorError::new(format!(
                        "ptr::null_fn() type argument is not a function pointer: {spelling}"
                    ))
                },
            )
        }
        _ => Err(AnnotatorError::new(
            "ptr::null_fn() takes an optional function-pointer type argument",
        )),
    }
}

/// Analyzer for `std::mem::size_of::<T>() -> usize` — compile-time
/// type-size constant.  Registered against the
/// `HostObject::new_builtin_callable("std.mem.size_of")` stub.
/// Callers in pyre source: `lltype::malloc_typed`,
/// `object_array::items_block_layout`.  Returns `SomeInteger` since
/// the actual value is a compile-time `usize`; the rtyper folds the
/// const at lowering time, but the annotator just needs the lattice
/// position.  The value is a `usize` byte size, so it is non-negative —
/// modeled as `SomeInteger(nonneg=True, unsigned=True)` (the `usize`
/// lattice), strictly more precise than the default and safe under join.
fn std_mem_size_of(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::new(true, true)))
}

/// Analyzer for `std::mem::align_of::<T>() -> usize`.  Identical
/// lattice to [`std_mem_size_of`] (a compile-time non-negative
/// `usize`); registered against the `std.mem.align_of` stub so those
/// callsites do not reach the "no analyser registered" error.
fn std_mem_align_of(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    std_mem_size_of(bk, args_s, kwds)
}

/// Analyzer for `__cast_address_intrinsic` — the front-end pointer-type
/// erasure (`p as *mut u8`).  Upstream a heterogeneous heap block that
/// reaches a GC ownership predicate or a write barrier travels as
/// `llmemory.Address` (`llmemory.cast_ptr_to_adr`), never as
/// `Ptr(instance)`: the hook discriminates blocks by address range, so
/// the pointee type is not merely unused, it must not exist.  Aliasing
/// the cast instead leaves `SomeInstance(PyFrame)` on the argument, and
/// the hook's slot 0 then unions `PyFrame ∪ ItemsBlock` and blocks with
/// "cannot unify instances with no common base class".
///
/// The erased result is the classdef-less pointer shell — the annotation
/// a pointee-less raw pointer already carries here — rather than
/// `SomeAddress`.  That keeps one annotation across both kinds of caller:
/// the ones that erase a typed pointer, and the ones that pass an
/// untyped `*mut u8` field straight through.  `SomeInstance` union takes
/// the classdef-less side as the result (`model.rs`'s `union`), so every such
/// slot merges cleanly.  `args[0]` is the pointer operand.
///
/// The operand's own annotation is not constrained: `cast_ptr_to_adr`
/// takes any `Ptr`, and the operand here is whatever the pointee ADT
/// annotated to — an instance, but also a `SomeList` for a `Vec` whose
/// address is erased.  Only the nullability carries over, since an
/// erasure of a nullable pointer is itself nullable.  (Its narrowing
/// twin does reject a non-pointer operand: a downcast to a named root
/// off a non-pointer is a producer bug, whereas dropping type
/// information off one never is.)
fn cast_address_intrinsic(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 1 {
        return Err(AnnotatorError::new(
            "__cast_address_intrinsic expects exactly one positional pointer argument",
        ));
    }
    let can_be_none = arg_at(args_s, 0, crate::runtime_names::shims::CAST_ADDRESS).can_be_none();
    Ok(SomeValue::Instance(super::model::SomeInstance::new(
        None,
        can_be_none,
        std::collections::BTreeMap::new(),
    )))
}

/// Annotation of the Rust-adapter spelling of `ll_arraymove`.
///
/// The MIR front has already recovered a logical list-items array from the
/// raw pointers before emitting this call, so the receiver arrives as the
/// list, matching the list-taking spelling `rlist.ll_arraymove`
/// (`rpython/rtyper/rlist.py`) rather than the ll-items one
/// (`rgc.ll_arraymove`, `rpython/rlib/rgc.py`), which takes
/// `lst.ll_items()`.
///
/// The declared signature is the whole surface, and both spellings declare
/// the same one: `@signature(types.any(), types.int(), types.int(),
/// types.int(), returns=types.none())` and `@enforceargs(None, int, int,
/// int)`.  `types.any()` / `None` constrain the receiver not at all, so this
/// port constrains it not at all either — only the three indices are checked,
/// and there is no result.
fn ll_arraymove_intrinsic(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 4 {
        return Err(AnnotatorError::new(
            "__majit_ll_arraymove expects (array, source_start, dest_start, length)",
        ));
    }
    for index in 1..4 {
        if !matches!(
            arg_at(args_s, index, crate::runtime_names::shims::LL_ARRAYMOVE),
            SomeValue::Integer(_)
        ) {
            return Err(AnnotatorError::new(format!(
                "__majit_ll_arraymove argument {index} must be SomeInteger"
            )));
        }
    }
    Ok(s_none())
}

/// Analyzer for `__cast_instance_intrinsic` — the front-end pointer-downcast
/// narrow (#298).  `front::mir` aliases the classdef-less pointer for a
/// same-bank `obj as *const RegisteredStruct` cast, so a field read on
/// the pointee blocks on a classdef-less `SomeInstance` (unaryop.rs
/// getattr arm).  The frontend instead lowers such a cast to
/// `simple_call(__cast_instance_intrinsic, operand, const(root))`.
/// The frontend puts the target struct root in `Call.args[1]` as a
/// `ByteStr` Constant.  The result is
/// `SomeInstance(classdef)` for that root so the field read resolves;
/// the rtyper lowers the call to a `cast_pointer` (rclass
/// `pairtype(InstanceRepr, InstanceRepr).convert_from_to`,
/// rclass.py — its `r_ins1.classdef is None` arm already emits the
/// root→concrete cast).  `args[0]` is the pointer operand; `args[1]` is
/// the constant root name.
///
/// A root the bookkeeper deliberately models as a non-instance value —
/// a `FixedObjectArray` projected to its `_items` element list, a
/// `Wtf8Buf` to a byte string (`project_struct_field_type`,
/// in `bookkeeper.rs`) — has its access lowered through `getitem`/byte
/// reads, not a named-field getattr.  An operand that ALREADY carries the
/// modelled variant (`SomeList` for a list root, `SomeString` for a string
/// root) passes through unchanged, exactly as the erasure twin
/// `cast_address_intrinsic` handles a `SomeList` for a `Vec` whose address
/// was erased.  A pointer-shaped operand (`SomeInstance` / `SomePtr` /
/// `SomeAddress`), which is what the GCREF and `GcArray` boundaries hand in,
/// is narrowed to the root's projected annotation instead — carrying the
/// operand's nullability, since `cast_opaque_ptr` answers
/// `nullptr(PTRTYPE.TO)` for a null (`lltype.py`).
fn cast_instance_intrinsic(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // The marker is `Call(["__cast_instance_intrinsic"], [operand, const(root)])`:
    // the pointer operand plus the constant-root string, no
    // keywords.  A different shape is a producer bug, surfaced here
    // rather than silently swallowed.
    if !kwds.is_empty() || args_s.len() != 2 {
        return Err(AnnotatorError::new(
            "__cast_instance_intrinsic expects (operand, constant_root) positional arguments",
        ));
    }
    let root = match args_s
        .get(1)
        .and_then(|o| o.as_ref())
        .and_then(|sv| sv.const_())
        .and_then(|c| c.as_pystr())
    {
        Some(s) => s.to_string(),
        None => {
            return Err(AnnotatorError::new(
                "__cast_instance_intrinsic: target struct root must be a constant string",
            ));
        }
    };
    let operand = arg_at(args_s, 0, crate::runtime_names::shims::CAST_INSTANCE);
    let projected = if root == "GCREF" {
        // RPython GC-transform root slots are llmemory.GCREF.  Rust spells
        // the helper parameter/return as PyObjectRef, but that carrier must
        // not mint a W_Root instance annotation: every concrete GC pointer
        // is cast_opaque_ptr'd into this slot and restored afterward.
        crate::translator::rtyper::llannotation::lltype_to_annotation(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        )
    } else if majit_ir::descr::is_list_container_spelling(&root) {
        // `[T]` / `Vec<T>` dests are the current op's list (`Bookkeeper.getlistdef`),
        // not a fresh ListDef per projection. `setbinding` then `contains`
        // the same object on reflow (`TLS.no_side_effects_in_union`).
        bk.project_list_spelling_at_position(&root)
    } else {
        // One projection per position (`Bookkeeper.getlistdef`,
        // bookkeeper.py). Reflow setbinding then sees the same ListDefs.
        bk.stable_projected_root(&root)
    };
    // Nullability travels with the value, not with the target spelling: a
    // downcast of a nullable pointer is itself nullable, and a null operand
    // is a legal one — `cast_opaque_ptr` answers `nullptr(PTRTYPE.TO)` for it
    // (`lltype.py`).  Computed here so the target-spelling arms
    // below cannot bypass it the way the block further down already honours
    // it for the instance case.
    let operand_can_be_none = match operand {
        SomeValue::Instance(inst) => inst.can_be_none,
        SomeValue::String(str_) => str_.inner.can_be_none,
        SomeValue::None_(_) => true,
        _ => false,
    };
    let with_nullability = |mut projected: SomeValue| {
        if let SomeValue::String(str_) = &mut projected {
            str_.inner.can_be_none = operand_can_be_none;
        }
        projected
    };
    // The symmetric external-item boundary rlist uses: a GCREF read from
    // `GcArray(GCREF)` becomes its concrete external pointer (StringRepr
    // here).  `AbstractBaseListRepr.recast` (`rpython/rtyper/rlist.py`)
    // is the external/internal convert, and the GCREF half of it is
    // `pairtype(GCRefRepr, Repr).convert_from_to`
    // (`rpython/rtyper/lltypesystem/rgcref.py`), which genops
    // `cast_opaque_ptr`.  The synthetic marker carries that target spelling;
    // return the projected string for any GC pointer source, while an
    // already-string source is the identity.
    if matches!(&projected, SomeValue::String(_)) {
        return match operand {
            SomeValue::String(_) => Ok(operand.clone()),
            SomeValue::Instance(_)
            | SomeValue::List(_)
            | SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::None_(_) => Ok(with_nullability(projected)),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-pointer operand for string root {root:?}: {other:?}"
            ))),
        };
    }
    if root == "GCREF" {
        return match operand {
            SomeValue::String(_)
            | SomeValue::Instance(_)
            | SomeValue::List(_)
            | SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::None_(_) => Ok(projected),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-GC-pointer operand for GCREF: {other:?}"
            ))),
        };
    }
    // A physical `TypedItemsBlock` pointer is the Rust owner for RPython's
    // direct `GcArray(Signed|Float)`. The MIR front records that cast as this
    // same pointer-narrow marker with a `[i64]` / `[f64]` target spelling.
    // Return the target SomeList annotation so `getitem` dispatches through
    // ListRepr; the typer below emits the same `cast_pointer` to the target
    // GcArray lowleveltype that an RPython erase/unerase boundary produces.
    // Reaching the marker proves the pointer is immediately dereferenced, so
    // a null value is not an observable successful path.
    // `SomeList` carries no `can_be_None` — upstream's does not either — so
    // the nullability computed above has nowhere to land on this arm; a null
    // operand is still admitted rather than hard-errored.
    // A list root is one `ListDef` for this operation
    // (`Bookkeeper.newlist` / `getlistdef`). `project_struct_field_type`
    // mints a fresh list, and the next reflow's `setbinding` refuses it:
    // the new list does not contain the one already bound.
    if let SomeValue::List(list) = &projected {
        let projected = if bk.position_entered.get() {
            let item = list.listdef.s_value();
            match bk.newlist(&[item], None) {
                Ok(stable) => SomeValue::List(stable),
                Err(err) => {
                    return Err(AnnotatorError::new(format!(
                        "__cast_instance_intrinsic: {err}"
                    )));
                }
            }
        } else {
            projected.clone()
        };
        return match operand {
            SomeValue::List(_) => Ok(operand.clone()),
            SomeValue::Instance(_)
            | SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::None_(_) => Ok(projected),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-pointer operand for list root {root:?}: {other:?}"
            ))),
        };
    }
    // A pointer reaching a Rust tuple (a residual call's tuple result, a
    // `&(A,B)` operand) is narrowed to the `SomeTuple` of that shape; the
    // typer routes the unrelated pointer pair through GCREF
    // (`rgcref.py`).  An operand that already is a tuple passes through.
    if matches!(&projected, SomeValue::Tuple(_)) {
        return match operand {
            SomeValue::Tuple(_) => Ok(operand.clone()),
            SomeValue::Instance(_)
            | SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::None_(_) => Ok(projected),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-pointer operand for tuple root {root:?}: {other:?}"
            ))),
        };
    }
    // `RDict<K, V, S>` projects to `SomeDict` (`OrderedDictRepr`). An
    // operand that already is that dict passes through; a pointer-shaped
    // operand is the dict the `insert` / `get` call is about to use.
    // `SomeDict` carries no `can_be_None`, same as the list arm.
    if matches!(&projected, SomeValue::Dict(_)) {
        return match operand {
            SomeValue::Dict(_) => Ok(operand.clone()),
            SomeValue::Instance(_)
            | SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::None_(_) => Ok(projected),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-pointer operand for dict root {root:?}: {other:?}"
            ))),
        };
    }
    // A pointer to a declared-Raw ADT is an int-bank `SomePtr`. Retargeting
    // `*mut A as *mut B` must replace that pointee, not mint a GC instance.
    // A by-value operand (`push_roots()` → `RootScope`, `VersionTag`) is
    // the same raw ADT: FieldRead already projects SomePtr
    // (`project_struct_field_type`), and `llannotation.py`
    // `pairtype(SomePtr, SomeObject).union` refuses Instance ∪ Ptr.
    if let Some(raw_ptr) = bk.raw_struct_ptr_annotation(&root) {
        return match operand {
            SomeValue::Ptr(_)
            | SomeValue::Address(_)
            | SomeValue::Integer(_)
            | SomeValue::Instance(_) => Ok(raw_ptr),
            other => Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-address operand for raw root {root:?}: {other:?}"
            ))),
        };
    }
    // Carry the operand's nullability onto the downcast result instead
    // of unconditionally narrowing to non-None: a downcast of a nullable
    // pointer is itself nullable.  Same rule as `operand_can_be_none` above,
    // spelled out here because this arm also has to classify the operand
    // variant it does not recognise.
    let can_be_none = match operand {
        SomeValue::Instance(_) | SomeValue::None_(_) => operand_can_be_none,
        SomeValue::Ptr(_) | SomeValue::Address(_) => false,
        // `addr_of_mut!((*entries).items).cast()` reads `[Entry; 0]`.
        // The field projects as a list of `Impossible`. The cast target
        // is `*mut Entry`, the address of that tail.
        SomeValue::List(list) => {
            let item = list.listdef.read_item(None);
            if !matches!(item, SomeValue::Impossible) {
                return Err(AnnotatorError::new(format!(
                    "__cast_instance_intrinsic: non-pointer operand for root {root:?}: {operand:?}"
                )));
            }
            let classdef = bk.getuniqueclassdef_for_struct_root(&root)?;
            return Ok(SomeValue::Instance(super::model::SomeInstance::new(
                Some(classdef),
                operand_can_be_none,
                std::collections::BTreeMap::new(),
            )));
        }
        other => {
            // A root the bookkeeper models as a list or string (a
            // `FixedObjectArray` projected to its `_items` element list,
            // a `Wtf8Buf` projected to a byte string —
            // `project_struct_field_type` in `bookkeeper.rs`) reads
            // through `getitem`/byte access, never a named field, so the
            // pointer→instance narrow is a no-op for it: return the
            // operand unchanged when its variant matches the model, the
            // same operand-agnostic pass-through the erasure twin
            // `cast_address_intrinsic` already performs.  `convert_from_to`
            // (rclass.py) constrains only the destination repr, not
            // the operand shape.  A genuine variant mismatch is still a
            // producer bug, surfaced with the root that was expected.
            if projected.tag() == other.tag() {
                return Ok(operand.clone());
            }
            return Err(AnnotatorError::new(format!(
                "__cast_instance_intrinsic: non-pointer operand for root {root:?}: {other:?}"
            )));
        }
    };
    let classdef = bk.getuniqueclassdef_for_struct_root(&root)?;
    Ok(SomeValue::Instance(super::model::SomeInstance::new(
        Some(classdef),
        can_be_none,
        std::collections::BTreeMap::new(),
    )))
}

/// Analyzer for `String::new()` / `String::with_capacity(n)`.  Both
/// build an empty owned UTF-8 buffer, so the annotation surface is a
/// mutable `SomeString` (not `None`; a `String` may hold interior nul
/// bytes, hence `no_nul = false`).  Registered against the `String.new`
/// / `String.with_capacity` HOST_ENV stubs so those callsites resolve to
/// a real lattice value instead of erroring with "no analyser registered".
fn string_constructor(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::String(SomeString::new(false, false)))
}

/// Analyzer for `BigInt::from(i64)` — the `From<i64>` impl that boxes a
/// machine integer into the arbitrary-precision `BigInt` the interpreter
/// uses for Python `long`.  `BigInt` is a foreign Rust value with no
/// Repr: in the JIT the overflow→long arm is the cold `guard_no_overflow`
/// bailout (`@jit.dont_look_inside` in `rbigint`), never traced inline,
/// so the value stays opaque.  Model the result as a classdef-less
/// `SomeInstance` — the same foreign-opaque GcRef shell
/// `annotation_state::ref_fallback_instance` gives an unregistered
/// host-struct pointee.  It projects to `ValueType::Ref(None)`
/// (`somevalue_to_valuetype`), matching the `result_kind = 'r'` the
/// `bigint_add`/`bigint_sub`/`bigint_mul` residuals produce.
///
/// This retires the `SomeBuiltin.call(): no analyser registered for
/// BigInt.from` wall.  It does not by itself carry the cold long arm to
/// Match: the next operation on the boxed value (the `+`/`-`/`*` operator
/// → a `<BigInt as Add>::add` method call, or `as_bigint`/`bigint_result`'s
/// `.to_i64()`/`.div_floor()`/`.sign()` …) is a method/operator call on
/// the opaque shell that `SomeInstance.getattr` rejects as classdef-less.
/// Residualizing those foreign-method calls on an opaque instance is a
/// separate port.
fn bigint_from(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Instance(SomeInstance::new(
        None,
        false,
        std::collections::BTreeMap::new(),
    )))
}

/// Analyzer for foreign Rust container constructors that box their
/// contents behind an opaque handle the JIT never rtypes: `Vec::new`,
/// `Vec::with_capacity`, `indexmap::IndexMap::new`, and `Box::new`.
/// Each returns a freshly-built owned container/box whose element
/// internals stay foreign — the same wall `BigInt::from` draws around
/// the boxed arbitrary-precision integer.  Model the result as a
/// classdef-less `SomeInstance` (the foreign-opaque GcRef shell
/// `annotation_state::ref_fallback_instance` gives an unregistered
/// host-struct pointee); it projects to `ValueType::Ref(None)`
/// (`somevalue_to_valuetype`).
///
/// Registering these retires the `SomeBuiltin.call(): no analyser
/// registered for <ctor>` wall the two-phase prepass hits when it
/// residualizes a body that builds one of these containers.  A
/// subsequent method/accessor call on the opaque handle (`Vec::index`,
/// `IndexMap::get`, `Box::into_raw`, …) is a separate leaf resolved on
/// its own path (`translate_op` / `CallRegistry`, not the analyser
/// table); this analyser only types the constructor's result.
fn foreign_container_ctor(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Instance(SomeInstance::new(
        None,
        false,
        std::collections::BTreeMap::new(),
    )))
}

/// RPython `rlib/jit.py` — the `hint` ExtRegistryEntry's
/// `compute_result_annotation`, for the `access_directly=True` spelling.
///
/// ```python
/// s_x = annmodel.not_const(s_x)
/// ...
/// if isinstance(s_x, annmodel.SomeInstance):
///     classdesc = s_x.classdef.classdesc
///     virtualizable = classdesc.get_param('_virtualizable_')
///     if s_access_directly.const == True and virtualizable is not None:
///         flags = s_x.flags.copy()
///         flags['access_directly'] = True
///         ...
///     else:
///         ...  # delete both flags
/// return s_x
/// ```
///
/// The `else` arm matters as much as the `if`: a hint on a class that does
/// not declare `_virtualizable_` STRIPS the flags rather than leaving them,
/// so the annotation cannot carry `access_directly` off a non-virtualizable.
///
/// The consumer is `specialize.py default_specialize`, which flags the
/// CALLEE a flagged argument reaches — never the function that spells this
/// hint, whose own arguments arrive unflagged.
fn jit_hint_access_directly(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(hint_virtualizable_flags(args_s, false))
}

/// RPython `rlib/jit.py`, same entry, for the `fresh_virtualizable=True`
/// spelling.
///
/// Upstream carries both kwargs on one call and asserts against the lone
/// form — `assert s_access_directly, "lone fresh_virtualizable hint"`. Pyre
/// splits the call in two, so this helper sets BOTH flags rather than
/// `fresh_virtualizable` alone; the state upstream asserts against is then
/// unreachable instead of merely undiagnosed.
fn jit_hint_fresh_virtualizable(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(hint_virtualizable_flags(args_s, true))
}

/// The body both `hint` spellings share — `rlib/jit.py`'s
/// `SomeInstance` branch.
fn hint_virtualizable_flags(args_s: &[Option<SomeValue>], fresh_virtualizable: bool) -> SomeValue {
    let Some(Some(s_x)) = args_s.first() else {
        // `hint` is an identity on its argument; with nothing bound there is
        // nothing to annotate and nothing to flag.
        return s_impossible_value();
    };
    let s_x = super::model::not_const(s_x);
    let SomeValue::Instance(inst) = &s_x else {
        // Upstream's `isinstance(s_x, SomeInstance)` guard: every other
        // annotation falls through to `return s_x` untouched.
        return s_x;
    };
    let is_virtualizable = inst.classdef.as_ref().is_some_and(|classdef| {
        let classdesc = classdef.borrow().classdesc.clone();
        let param = classdesc.borrow().get_param("_virtualizable_", None, true);
        !matches!(param, crate::flowspace::model::ConstValue::None)
    });
    let mut flags = inst.flags.clone();
    if is_virtualizable {
        flags.insert("access_directly".to_string(), true);
        if fresh_virtualizable {
            flags.insert("fresh_virtualizable".to_string(), true);
        }
    } else {
        flags.remove("access_directly");
        flags.remove("fresh_virtualizable");
    }
    SomeValue::Instance(super::model::SomeInstance::new(
        inst.classdef.clone(),
        inst.can_be_none,
        flags,
    ))
}

/// RPython `rlib/jit.py`, same entry, for the `access_directly=False`
/// spelling — the `else` arm.
///
/// ```python
/// assert s_access_directly.const == False or virtualizable is None
/// if 'access_directly' in s_x.flags or 'fresh_virtualizable' in s_x.flags:
///     flags = s_x.flags.copy()
///     ...  # delete both
/// ```
///
/// It strips unconditionally, where the `True` spelling strips only when the
/// class does not declare `_virtualizable_`. `executioncontext.py` spells it
/// at both points where a frame is handed to arbitrary Python.
fn jit_hint_no_access_directly(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    let Some(Some(s_x)) = args_s.first() else {
        return Ok(s_impossible_value());
    };
    let s_x = super::model::not_const(s_x);
    let SomeValue::Instance(inst) = &s_x else {
        return Ok(s_x);
    };
    let mut flags = inst.flags.clone();
    flags.remove("access_directly");
    flags.remove("fresh_virtualizable");
    Ok(SomeValue::Instance(super::model::SomeInstance::new(
        inst.classdef.clone(),
        inst.can_be_none,
        flags,
    )))
}

/// RPython `rlib/debug.py` — the `check_not_access_directly`
/// ExtRegistryEntry's `compute_result_annotation`.
///
/// ```python
/// def compute_result_annotation(self, s_arg):
///     assert not s_arg.flags.get('access_directly', False)
///     return s_arg
/// ```
///
/// Upstream asserts, i.e. it aborts translation. The failure is a real
/// defect in the caller — a graph reached the marked function carrying the
/// virtualizable hint, which would specialize it — so this raises rather
/// than silently dropping the flag.
fn rlib_check_not_access_directly(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    let Some(Some(s_arg)) = args_s.first() else {
        return Ok(s_impossible_value());
    };
    if let SomeValue::Instance(inst) = s_arg
        && inst.flags.get("access_directly").copied().unwrap_or(false)
    {
        return Err(AnnotatorError::new(
            "check_not_access_directly: argument carries the access_directly hint",
        ));
    }
    Ok(s_arg.clone())
}

/// Analyzer for the `majit_metainterp` crate's `pub fn -> bool` flag
/// helpers.  Single implementation covers both `majit_log_enabled()`
/// (logging gate) and `jit::we_are_jitted()` (JIT-context probe);
/// both return `bool` so the annotation is `SomeBool` regardless of
/// which qualname the dispatcher routed to.
fn majit_metainterp_bool_flag(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Bool(super::model::SomeBool::new()))
}

/// `rlib/nonconst.py EntryNonConstant.compute_result_annotation`:
///
/// ```python
/// def compute_result_annotation(self, s_arg):
///     return not_const(s_arg)
/// ```
///
/// The value is passed through unchanged except that its constancy is
/// dropped, so a caller branching on it keeps both arms.
fn nonconst_non_constant(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    let s_arg = arg_at(args_s, 0, "nonconst.non_constant");
    Ok(super::model::not_const(s_arg))
}

/// `rlib/jit.py ConditionalCallEntry.compute_result_annotation` for
/// `_jit_conditional_call`: the call is void.
fn jit_conditional_call(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(super::model::s_none())
}

/// `rlib/jit.py ConditionalCallEntry.compute_result_annotation` for
/// `_jit_conditional_call_value`:
///
/// ```python
/// s_res = self.bookkeeper.emulate_pbc_call(self.bookkeeper.position_key,
///                                          args_s[1], args_s[2:])
/// return annmodel.unionof(s_res, args_s[0].nonnoneify())
/// ```
///
/// Pyre's fn-const is `SomePtr(FuncType)` rather than always `SomePBC`.
/// A `Ptr(FuncType)` takes RESULT via `lltype_to_annotation`.
fn jit_conditional_call_elidable(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    use super::bookkeeper::EmulatedPbcCallKey;
    use crate::translator::rtyper::llannotation::lltype_to_annotation;
    use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;

    let s_value = arg_at(args_s, 0, "jit.conditional_call_elidable").clone();
    let s_fn = arg_at(args_s, 1, "jit.conditional_call_elidable");
    let s_res = match s_fn {
        SomeValue::PBC(_) => {
            let unique_key = match bk.current_position_key() {
                Some(pk) => EmulatedPbcCallKey::Position(pk),
                None => EmulatedPbcCallKey::Text("jit.conditional_call_elidable".to_string()),
            };
            let call_args: Vec<SomeValue> = (2..args_s.len())
                .map(|i| arg_at(args_s, i, "jit.conditional_call_elidable").clone())
                .collect();
            bk.emulate_pbc_call(unique_key, s_fn, &call_args, &[], None)?
        }
        SomeValue::Ptr(sp) => match &sp.ll_ptrtype.TO {
            PtrTarget::Func(ft) => lltype_to_annotation(ft.result.clone()),
            other => {
                return Err(AnnotatorError::new(format!(
                    "conditional_call_elidable: function arg is SomePtr but not a FuncType, got {other:?}"
                )));
            }
        },
        other => {
            return Err(AnnotatorError::new(format!(
                "conditional_call_elidable: function arg should be SomePBC or SomePtr(FuncType), got {other:?}"
            )));
        }
    };
    unionof([&s_res, &s_value.nonnoneify()]).map_err(|e| AnnotatorError::new(e.to_string()))
}

/// Analyzer for Rust primitive type `From` / `TryFrom` impls
/// (`u32::from`, `i64::from`, `i64::try_from`, `usize::try_from`).
/// All return integer values (the target primitive); `TryFrom` returns
/// `Result<T, ...>` but pyre lowers the unwrap inline so the
/// annotation surface is `SomeInteger`.  Per-primitive precision
/// (knowntype) is left as default since the rtyper folds the
/// conversion at lowering time.
fn primitive_integer_conversion(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::default()))
}

/// Upstream `ann_cast_ptr_to_int(s_ptr)`
/// (rpython/rtyper/lltypesystem/lltype.py).
///
/// ```python
/// @analyzer_for(cast_ptr_to_int)
/// def ann_cast_ptr_to_int(s_ptr): # xxx
///     return SomeInteger()
/// ```
fn lltype_cast_ptr_to_int(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::default()))
}

/// Upstream `ann_direct_ptradd(s_p, s_n)`
/// (`rpython/rtyper/lltypesystem/lltype.py` `ann_direct_ptradd`).
///
/// ```python
/// @analyzer_for(direct_ptradd)
/// def ann_direct_ptradd(s_p, s_n):
///     assert isinstance(s_p, SomePtr), "direct_* of non-pointer: %r" % s_p
///     return s_p
/// ```
///
/// The result annotation is the pointer operand (`ann_direct_ptradd`
/// returns `s_p`).  The operand must be `SomePtr`.
fn lltype_direct_ptradd(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 2 {
        return Err(AnnotatorError::new(
            "direct_ptradd expects a pointer and a count",
        ));
    }
    let s_p = arg_at(args_s, 0, "lltype.direct_ptradd");
    match s_p {
        SomeValue::Ptr(_) => Ok(s_p.clone()),
        // `alloc_zeroed` returns `*mut u8`. That bank is `Ref(None)`,
        // shelled as a classdef-less instance. The add result is a raw
        // pointer.
        SomeValue::Instance(inst) if inst.classdef.is_none() => Ok(raw_alloc_ptr_somevalue()),
        other => Err(AnnotatorError::new(format!(
            "direct_ptradd of non-pointer: {other:?}"
        ))),
    }
}

fn raw_alloc_ptr_somevalue() -> SomeValue {
    use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr, SomePtr};
    let opaque = OpaqueType::gc("MAJIT_REF_OPAQUE");
    let placeholder_ptr = Ptr::from_container_type(
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Opaque(Box::new(opaque)),
    )
    .expect("Opaque container yields a valid Ptr");
    SomeValue::Ptr(SomePtr::new(placeholder_ptr))
}

/// Upstream `ann_cast_int_to_ptr(PtrT, s_int)`
/// (rpython/rtyper/lltypesystem/lltype.py).
///
/// ```python
/// @analyzer_for(cast_int_to_ptr)
/// def ann_cast_int_to_ptr(PtrT, s_int):
///     assert PtrT.is_constant()
///     return SomePtr(ll_ptrtype=PtrT.const)
/// ```
///
/// TODO: the Rust frontend's
/// `Expr::Cast { ValueType::Ref }` carries only the opaque high-level
/// `Ref` (no concrete Ptr lltype is known at lowering time), so the
/// frontend lowers `Int → Ref` to a **1-arg** `simple_call(lltype.\
/// cast_int_to_ptr, v_int)` instead of upstream's 2-arg `simple_call\
/// (lltype.cast_int_to_ptr, PTRTYPE, oddint)`.  Upstream's
/// `ann_cast_int_to_ptr` asserts `PtrT.is_constant()` and returns
/// `SomePtr(ll_ptrtype=PtrT.const)` (lltype.py), reading
/// the concrete Ptr type from the first argument.  Pyre's analyzer
/// cannot recover that concrete Ptr from `args_s[0]` because it
/// isn't there.  We return `SomePtr` with a Gc-Opaque placeholder
/// Ptr (mirroring the opaque Ref bank); the rtyper-phase
/// `rtype_cast_int_to_ptr` recovers the caller-side concrete Ptr
/// from `hop.r_result.lowleveltype()`.  The constant carrier already
/// exists (`ConstValue::LowLevelType(Box<LowLevelType>)`, flowspace/
/// model.rs, supports `LowLevelType::Ptr`); the blocker is the
/// frontend not having the concrete Ptr available at `Expr::Cast`
/// lowering time.  Graduating to the 2-arg form is blocked on
/// threading the rtyper-side annotation back to the frontend (or a
/// deferred 2-arg rewrite in jtransform / annotator once the Ptr is
/// known).
fn lltype_cast_int_to_ptr(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr, SomePtr};
    let opaque = OpaqueType::gc("MAJIT_REF_OPAQUE");
    let placeholder_ptr = Ptr::from_container_type(
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Opaque(Box::new(opaque)),
    )
    .expect("Opaque container yields a valid Ptr");
    Ok(SomeValue::Ptr(SomePtr::new(placeholder_ptr)))
}

/// Upstream `robjmodel_instantiate(s_clspbc, s_nonmovable=None)`
/// (builtin.py:229-242).
pub fn robjmodel_instantiate(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new(
            "instantiate() expected SomePBC as first argument",
        ));
    }
    let s_clspbc_v = arg_at(args_s, 0, "robjmodel_instantiate");
    let SomeValue::PBC(s_clspbc) = s_clspbc_v else {
        return Err(AnnotatorError::new(
            "instantiate() expected SomePBC as first argument",
        ));
    };
    let mut clsdef: Option<Rc<RefCell<ClassDef>>> = None;
    // upstream: `more_than_one = len(s_clspbc.descriptions) > 1`. The
    // `needs_generic_instantiate` dict is consumed downstream by the
    // rtyper (`rpython/rtyper/rclass.py:_get_concrete_class`) to emit
    // a generic allocator; the Rust bookkeeper does not yet carry that
    // map (no rtyper consumer exists) — the flag is recorded here as a
    // comment so the field lands alongside the rtyper port.
    //   let more_than_one = s_clspbc.descriptions.len() > 1;
    for desc in s_clspbc.descriptions.values() {
        let DescEntry::Class(class_desc) = desc else {
            return Err(AnnotatorError::new(
                "instantiate() expects a class PBC — non-class DescEntry",
            ));
        };
        let cdef = ClassDesc::getuniqueclassdef(class_desc)?;
        //   if more_than_one: bk.needs_generic_instantiate[cdef] = True
        clsdef = Some(match clsdef {
            None => cdef,
            Some(prev) => ClassDef::commonbase(&prev, &cdef)
                .ok_or_else(|| AnnotatorError::new("instantiate(): classes have no common base"))?,
        });
    }
    let cdef = clsdef
        .ok_or_else(|| AnnotatorError::new("instantiate() called with empty descriptor set"))?;
    Ok(SomeValue::Instance(SomeInstance::new(
        Some(cdef),
        false,
        Default::default(),
    )))
}

/// `ann_cast_pointer(PtrT, s_p)` (lltype.py): the result of
/// `cast_pointer(PTRTYPE, ptr)` is annotated from the constant target
/// type, independent of the operand's annotation.  Pyre's
/// instance-classed world resolves the constant target class to its
/// unique classdef; the operand's nullability survives the cast (a
/// downcast of a possibly-null pointer stays possibly-null — upstream
/// `SomePtr` does not track null, pyre's `SomeInstance.can_be_none`
/// does).
fn lltype_cast_pointer(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 2 {
        return Err(AnnotatorError::new(
            "cast_pointer() expects (PTRTYPE, ptr) positional arguments",
        ));
    }
    let s_clspbc_v = arg_at(args_s, 0, "lltype_cast_pointer");
    let SomeValue::PBC(s_clspbc) = s_clspbc_v else {
        return Err(AnnotatorError::new(
            "cast_pointer() expected a constant class PBC as first argument",
        ));
    };
    // upstream `assert s_CURTYPE.is_constant()` — a multi-desc PBC is
    // not a constant target type.
    let mut descs = s_clspbc.descriptions.values();
    let (Some(DescEntry::Class(class_desc)), None) = (descs.next(), descs.next()) else {
        return Err(AnnotatorError::new(
            "cast_pointer() target must be a single constant class",
        ));
    };
    let cdef = ClassDesc::getuniqueclassdef(class_desc)?;
    // `assert isinstance(s_p, SomePtr)` (lltype.py) — the operand
    // must be a pointer carrier.  Pyre's carriers are `SomePtr` and
    // `SomeInstance` (classed cast targets and classdef-less erased
    // pointers); anything else is a non-pointer cast.
    let can_be_none = match arg_at(args_s, 1, "lltype_cast_pointer") {
        SomeValue::Instance(inst) => inst.can_be_none,
        SomeValue::Ptr(_) => false,
        // A constant-None operand is a null pointer.  Upstream null
        // constants annotate as SomePtr and pass the lltype.py
        // assert; pyre carries them as SomeNone, and the downcast of
        // a null pointer stays null.
        SomeValue::None_(_) => true,
        other => {
            return Err(AnnotatorError::new(format!(
                "cast_pointer(): casting of non-pointer: {other:?}"
            )));
        }
    };
    Ok(SomeValue::Instance(SomeInstance::new(
        Some(cdef),
        can_be_none,
        Default::default(),
    )))
}

/// Analyzer for `pyre_object::lltype::malloc[_typed]::<T>(value: T) -> *mut T`
/// — the GC allocation intrinsics (`lltype.malloc(STRUCT, flavor='gc')`
/// parity). Registered as builtins so their allocator implementations are
/// NEVER looked-inside. In particular, `malloc_typed` calls
/// `<T as GcType>::type_id()`, a trait accessor with no flowable graph, while
/// the untyped `malloc` reaches the host GC-header allocator. Both are the
/// source spelling consumed by `fuse_boxing_alloc`, just as RPython's rtyper
/// emits one `malloc` op for `jtransform.rewrite_op_malloc`. A successful malloc
/// returns a non-null heap pointer to the same struct the argument
/// constructs, so the result is `SomeInstance(T)` with `can_be_none =
/// false` (OOM takes the MemoryError exception edge, not a null result).
/// The single argument is the constructed struct value (a `SomeInstance`
/// produced by the `SyntheticTransparentCtor`).
fn malloc_typed_alloc(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 1 {
        return Err(AnnotatorError::new(
            "malloc[_typed]() expects a single (value) positional argument",
        ));
    }
    match arg_at(args_s, 0, "malloc[_typed]") {
        SomeValue::Instance(inst) => {
            // lltype.py `malloc(T)` (`:2202-2210`) admits `Struct`, `Array`
            // and `OpaqueType`; anything else falls to
            // `raise TypeError("malloc: unmallocable type")`.  Only the
            // `Struct` arm is reachable through this marker.  (Its result is
            // modelled as a `SomeInstance` rather than upstream `ann_malloc`'s
            // `SomePtr(Ptr(s_T.const))` — pyre's instance annotation IS the
            // struct-pointer annotation.)  A classdef-less
            // `SomeInstance` (object-only; `SomeInstance(classdef=None)`)
            // carries no concrete struct, so it is that `else` arm — reject
            // it here instead of boxing a typeless value the downstream
            // `NewWithVtable` cannot stamp a vtable for.
            if inst.classdef.is_none() {
                return Err(AnnotatorError::new("malloc: unmallocable type"));
            }
            Ok(SomeValue::Instance(SomeInstance::new(
                inst.classdef.clone(),
                false,
                Default::default(),
            )))
        }
        other => Err(AnnotatorError::new(format!(
            "malloc[_typed](): expected a struct value to box, got {other:?}"
        ))),
    }
}

/// Analyzer for `pyre_object::lltype::malloc_raw::<T>(value: T) -> *mut T`
/// — the raw (`lltype.malloc(T, flavor='raw')` parity) heap allocation.
/// Registered as a builtin so the body is NEVER looked-inside: it calls
/// `Box::new` / `Box::into_raw`, neither of which carries an analyser (the
/// `Box.*` HOST_ENV stubs have no analyzer wired), so a body lift would
/// poison every raw-allocating caller with "no analyser registered for
/// Box.new".  Unlike `malloc_typed` (struct-only, `flavor='gc'`), the raw
/// flavor allocates ANY `T` — an opaque foreign payload (`*mut BigInt`, the
/// `int` overflow path), a `*mut String`, a `*mut Vec`, or a pyre struct
/// (`DictStorage`, `FrameBlock`, …) — so the `*mut T` result carries the
/// same value identity as the argument and the arg shape is passed through
/// unchanged.  A successful raw malloc returns a non-null pointer to that
/// payload, so an `Instance` result drops its `can_be_none` (OOM takes the
/// MemoryError edge, not a null result).
///
/// Like [`malloc_typed_alloc`], the result models the freshly-allocated
/// object by its value/instance shape, not `ann_malloc`'s `SomePtr(Ptr(T))`
/// (lltype.py): the allocation feeds the boxing / `NewWithVtable` path
/// that consumes the object directly, so a `SomePtr` wrapper here would
/// diverge from the established `malloc_typed` result modeling.
fn malloc_raw_alloc(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if !kwds.is_empty() || args_s.len() != 1 {
        return Err(AnnotatorError::new(
            "malloc_raw() expects a single (value) positional argument",
        ));
    }
    match arg_at(args_s, 0, "malloc_raw") {
        SomeValue::Instance(inst) => Ok(SomeValue::Instance(SomeInstance::new(
            inst.classdef.clone(),
            false,
            Default::default(),
        ))),
        // Any other payload shape (`SomeString`, `SomeList`, …) round-trips
        // through the `*mut T` pointer unchanged — the raw flavor does not
        // narrow the pointee.
        other => Ok(other.clone()),
    }
}

/// Upstream `robjmodel_r_dict(...)` (builtin.py).
pub fn robjmodel_r_dict(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    r_dict_helper(bk, args_s, kwds, RDictKind::Regular, "robjmodel_r_dict")
}

/// Upstream `robjmodel_r_ordereddict(...)` (builtin.py).
pub fn robjmodel_r_ordereddict(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    r_dict_helper(
        bk,
        args_s,
        kwds,
        RDictKind::Ordered,
        "robjmodel_r_ordereddict",
    )
}

#[derive(Clone, Copy)]
enum RDictKind {
    Regular,
    Ordered,
}

/// Upstream `_r_dict_helper(cls, s_eqfn, s_hashfn, s_force_non_null,
/// s_simple_hash_eq)` (builtin.py:253-268).
///
/// Accepts both positional and keyword forms for the `force_non_null`
/// and `simple_hash_eq` flags — upstream's Python signature binds
/// either path into the same local. Raises the upstream duplicate-
/// argument error if a caller passes both.
fn r_dict_helper(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
    kind: RDictKind,
    analyser: &str,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("r_dict(): missing eq function"));
    }
    if args_s.len() < 2 {
        return Err(AnnotatorError::new("r_dict(): missing hash function"));
    }
    // upstream `_r_dict_helper` binds these positions first; the body
    // touches `s_eqfn`/`s_hashfn` immediately so the per-touch `arg_at`
    // unwrap fires here (matching `unaryop.py call simple_call_
    // SomeBuiltin`'s bind-then-body sequence).
    let s_eqfn = arg_at(args_s, 0, analyser).clone();
    let s_hashfn = arg_at(args_s, 1, analyser).clone();
    if args_s.len() >= 3 && kwds.contains_key("s_force_non_null") {
        return Err(AnnotatorError::new(
            "r_dict() got multiple values for argument 'force_non_null'",
        ));
    }
    if args_s.len() >= 4 && kwds.contains_key("s_simple_hash_eq") {
        return Err(AnnotatorError::new(
            "r_dict() got multiple values for argument 'simple_hash_eq'",
        ));
    }
    if args_s.len() > 4 {
        return Err(AnnotatorError::new("r_dict() takes at most 4 arguments"));
    }
    // Optional positions: a slot present-but-None is treated identically
    // to "absent" here, matching upstream's `args_s.get(N)` semantic
    // (`None` from `get` and `None` from the slot itself both fall
    // through to the kwds lookup).  Required slots above use `arg_at`
    // which panics on None.  `kwds.get(name).cloned().flatten()`
    // projects `Option<Option<SomeValue>>` → `Option<SomeValue>`,
    // collapsing "key absent" and "key present but value unbound"
    // into a single "no concrete annotation" sentinel.
    let s_force = args_s
        .get(2)
        .and_then(|s| s.clone())
        .or_else(|| kwds.get("s_force_non_null").cloned().flatten());
    let s_simple = args_s
        .get(3)
        .and_then(|s| s.clone())
        .or_else(|| kwds.get("s_simple_hash_eq").cloned().flatten());
    let force_non_null = match s_force {
        None => false,
        Some(ref s) => {
            if !s.is_immutable_constant() {
                return Err(AnnotatorError::new(
                    "r_dict(): force_non_null must be constant",
                ));
            }
            matches!(s.const_(), Some(ConstValue::Bool(true)))
        }
    };
    let simple_hash_eq = match s_simple {
        None => false,
        Some(ref s) => {
            if !s.is_immutable_constant() {
                return Err(AnnotatorError::new(
                    "r_dict(): simple_hash_eq must be constant",
                ));
            }
            matches!(s.const_(), Some(ConstValue::Bool(true)))
        }
    };
    let dictdef = bk.getdictdef(true, force_non_null, simple_hash_eq);
    super::dictdef::DictKey::update_rdict_annotations(&dictdef.dictkey_rc(), s_eqfn, s_hashfn)
        .map_err(|e| AnnotatorError::new(e.msg))?;
    // upstream splits `SomeDict` / `SomeOrderedDict` classes; the Rust
    // port collapses them into a single `SomeDict` variant per CLAUDE.md
    // parity rule #1 (see `model.rs`'s `SomeOrderedDict` alias). `RDictKind`
    // is retained
    // in the call site to keep the upstream dispatch readable and to
    // ease re-splitting once the lattice grows `SomeOrderedDict`.
    let _ = kind;
    Ok(SomeValue::Dict(SomeDict::new(dictdef)))
}

/// Upstream `robjmodel_hlinvoke(...)` (builtin.py).
///
/// rtyper-phase only — consumes `rmodel.Repr`, `TyperError`,
/// `lltype_to_annotation`. Returns a clear `AnnotatorError` until the
/// rtyper bridge lands.
pub fn robjmodel_hlinvoke(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "hlinvoke: rtyper-phase only, not supported in annotator",
    ))
}

/// Upstream `robjmodel_keepalive_until_here(*args_s)` (builtin.py).
pub fn robjmodel_keepalive_until_here(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `robjmodel_free_non_gc_object(obj)` (builtin.py).
pub fn robjmodel_free_non_gc_object(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream implicitly returns None from an empty `def`.
    Ok(s_none())
}

/// Upstream `unicodedata_decimal(s_uchr)` (builtin.py).
pub fn unicodedata_decimal(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "unicodedate.decimal() calls should not happen at interp-level",
    ))
}

/// Upstream `analyze()` registered for `OrderedDict` (builtin.py).
pub fn analyze(
    bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream returns `SomeOrderedDict(bk.getdictdef())`; the Rust port
    // collapses `SomeDict`/`SomeOrderedDict` (see `model.rs`'s
    // `SomeOrderedDict` alias).
    let dd = bk.getdictdef(false, false, false);
    Ok(SomeValue::Dict(SomeDict::new(dd)))
}

/// Upstream `weakref_ref(s_obj)` (builtin.py).
pub fn weakref_ref(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new(
            "weakref.ref() takes exactly one argument",
        ));
    }
    let s_obj = arg_at(args_s, 0, "weakref_ref");
    let SomeValue::Instance(inst) = s_obj else {
        return Err(AnnotatorError::new(format!(
            "cannot take a weakref to {s_obj:?}"
        )));
    };
    if inst.can_be_none {
        return Err(AnnotatorError::new(
            "should assert that the instance we take a weakref to cannot be None",
        ));
    }
    Ok(SomeValue::WeakRef(SomeWeakRef::new(inst.classdef.clone())))
}

/// Upstream `pdb_set_trace(*args_s)` (builtin.py).
pub fn pdb_set_trace(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "you left pdb.set_trace() in your interpreter! \
         If you want to attach a gdb instead, call rlib.debug.attach_gdb()",
    ))
}

// Private helpers.

/// Emulates Python's `hasattr(host, name)` for a constant [`HostObject`]
/// receiver.
///
/// Native carrier boundary for builtin.py builtin_hasattr's host lookup.
/// Missing is AttributeError; Unsupported is a porting failure, not a Python
/// exception to fold away. In particular it must not manufacture Bool(false).
fn host_hasattr(
    host: &crate::flowspace::model::HostObject,
    name: &str,
) -> Result<bool, AnnotatorError> {
    use crate::flowspace::model::{HostGetAttrError, host_getattr};
    match host_getattr(host, name) {
        Ok(_) => Ok(true),
        Err(HostGetAttrError::Missing) => Ok(false),
        Err(HostGetAttrError::Unsupported) => Err(AnnotatorError::new(format!(
            "builtin_hasattr({:?}, {name:?}): host getattr is unsupported",
            host.qualname(),
        ))),
    }
}

/// Upstream `union(*s_values)` (model.py).
///
/// n-ary fold with `SomeImpossibleValue` as the identity element,
/// mirroring `annmodel.union(*s_values)` being called with the unpacked
/// `*s_values` tuple.  Each slot is unwrapped via [`arg_at`] at touch
/// time so an unannotated slot surfaces at the first iteration that
/// reads it — matching `unaryop.py call simple_call_SomeBuiltin`'s
/// bind-then-body sequence rather than an entry-point eager prefix
/// unwrap.
fn union_many(s_values: &[Option<SomeValue>], analyser: &str) -> Result<SomeValue, AnnotatorError> {
    let mut acc = s_impossible_value();
    for i in 0..s_values.len() {
        let s = arg_at(s_values, i, analyser);
        acc = union(&acc, s).map_err(|e| AnnotatorError::new(e.msg))?;
    }
    Ok(acc)
}

/// Upstream `s_iterable.iter().next()` — used by [`builtin_list`] /
/// [`builtin_zip`] / [`builtin_min`] / [`builtin_max`].
///
/// Delegates to [`super::unaryop::container_getanyitem`], the shared
/// port of upstream `container.getanyitem(position, *variant)`
/// (unaryop.py / :402-403 / :480-500 / :664-665). The caller
/// passes the iterator's `s_container` plus the variant string from
/// `SomeIterator.variant` (upstream's `*variant` tuple), matching the
/// logic in `unaryop.rs::someiterator_next` without requiring a live
/// `RPythonAnnotator` reference.
///
/// When the iterator wraps another `SomeIterator` (upstream `xs.iter()`
/// then `.iter()` again — returns self via `unaryop.py`), we
/// recurse on the inner iterator so the element annotation still
/// surfaces correctly.
fn someiterator_next_stub(bk: &Rc<Bookkeeper>, it: &SomeIterator) -> SomeValue {
    // upstream `if s_None.contains(self.s_container): return s_ImpossibleValue`
    // (unaryop.py:819-820).
    if matches!(&*it.s_container, SomeValue::None_(_)) {
        return s_impossible_value();
    }
    // upstream `if self.variant and self.variant[0] == "enumerate": ...`
    // (unaryop.py:822-824).
    if it.variant.first().map(String::as_str) == Some("enumerate") {
        let s_item =
            super::unaryop::container_getanyitem(&it.s_container, None, bk.current_position_key());
        return SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            s_item,
        ]));
    }
    // upstream `if variant == ("reversed",): variant = ()` (unaryop.py:826-827).
    let variant: Option<&str> = if it.variant.len() == 1 && it.variant[0] == "reversed" {
        None
    } else {
        it.variant.first().map(String::as_str)
    };
    // When the container is itself a `SomeIterator`, unwrap once — upstream
    // `SomeIterator.iter()` returns self (unaryop.py), so chaining
    // `.iter().next()` reaches the underlying container. The direct
    // `container_getanyitem` dispatch does not recognise Iterator inputs, so
    // we route them through another `someiterator_next_stub` hop first.
    if let SomeValue::Iterator(inner) = &*it.s_container {
        return someiterator_next_stub(bk, inner);
    }
    super::unaryop::container_getanyitem(&it.s_container, variant, bk.current_position_key())
}

// Tests.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    /// `Option<fn>`'s null and a `fn` field read are one `SomePtr`.
    /// `ptr_null_constant` is a classdef-less `SomeInstance` and does not
    /// union with that pointer (`llannotation.py` `pairtype(SomePtr, SomePtr)`).
    #[test]
    fn option_fn_null_unions_with_fn_pointer() {
        use crate::annotator::bookkeeper::raw_fn_ptr_somevalue;
        use crate::annotator::model::unionof;
        let s_fn = raw_fn_ptr_somevalue();
        let s_null =
            fn_null_constant(&bk(), &[], &std::collections::HashMap::new()).expect("null_fn");
        let merged = unionof([&s_fn, &s_null]).expect("Option<fn> arms union");
        assert!(
            matches!(merged, SomeValue::Ptr(_)),
            "null_fn must stay SomePtr, got {merged:?}"
        );
        // `null_mut` is a classdef-less `SomeInstance`. Joining it with a
        // function pointer collapses to that instance, which is the return
        // slot Phase B then cannot convert a `Ptr(FuncType)` into.
        let s_gc_null =
            ptr_null_constant(&bk(), &[], &std::collections::HashMap::new()).expect("null_mut");
        let collapsed = unionof([&s_fn, &s_gc_null]).expect("Instance/Ptr join");
        assert!(
            matches!(collapsed, SomeValue::Instance(_)),
            "null_mut collapses a fn pointer, got {collapsed:?}"
        );
    }

    /// `lltype.nullptr(FUNC)` of a spelled signature is the same
    /// `SomePtr` as a `Some` of that `fn` type (`getfunctionptr`).
    #[test]
    fn typed_null_fn_unions_with_matching_fn_pointer() {
        use crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling;
        use crate::annotator::model::{SomeValue, unionof};
        use crate::flowspace::model::{ConstValue, Constant};

        let spelling = "fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>";
        let s_fn = fn_ptr_somevalue_for_spelling(spelling).expect("fn spelling");
        let mut s_ty = crate::annotator::model::s_str0();
        s_ty.set_const_box(Constant::new(ConstValue::byte_str(spelling)));
        let s_null = fn_null_constant(&bk(), &[Some(s_ty)], &std::collections::HashMap::new())
            .expect("typed null_fn");
        let merged =
            unionof([&s_fn, &s_null]).expect("typed None unions with Some of the same fn type");
        assert!(
            matches!(merged, SomeValue::Ptr(_)),
            "typed null_fn must stay SomePtr, got {merged:?}"
        );
    }

    /// Build a `SomeInstance` whose classdesc does or does not declare
    /// `_virtualizable_` — `ClassDesc.get_param` reads it off the host class.
    fn instance_of(
        bk: &Rc<Bookkeeper>,
        name: &str,
        virtualizable: bool,
        flags: &[&str],
    ) -> SomeValue {
        use crate::flowspace::model::{ConstValue, HostObject};
        let mut members: indexmap::IndexMap<String, ConstValue> = indexmap::IndexMap::new();
        if virtualizable {
            members.insert(
                "_virtualizable_".into(),
                ConstValue::byte_str("locals_cells_stack_w[*]"),
            );
        }
        let pyobj = HostObject::new_class_with_members(name, vec![], members);
        let classdesc =
            super::super::classdesc::ClassDesc::new(bk, pyobj, Some(name.to_string()), None, None)
                .expect("ClassDesc::new");
        let classdef = super::super::classdesc::ClassDef::new(bk, &classdesc);
        let flags = flags
            .iter()
            .map(|f| ((*f).to_string(), true))
            .collect::<std::collections::BTreeMap<_, _>>();
        SomeValue::Instance(super::super::model::SomeInstance::new(
            Some(classdef),
            false,
            flags,
        ))
    }

    fn flags_of(s: &SomeValue) -> Vec<String> {
        let SomeValue::Instance(inst) = s else {
            panic!("not an instance: {s:?}")
        };
        inst.flags.keys().cloned().collect()
    }

    /// `rlib/jit.py` mints the flag only when the class declares
    /// `_virtualizable_`.
    #[test]
    fn the_hint_mints_access_directly_on_a_virtualizable() {
        let bk = bk();
        let s = instance_of(&bk, "PyFrame", true, &[]);
        let out = hint_virtualizable_flags(&[Some(s)], false);
        assert_eq!(flags_of(&out), vec!["access_directly".to_string()]);
    }

    /// The `fresh_virtualizable` spelling carries `access_directly` with it,
    /// so the lone form upstream asserts against is unreachable.
    #[test]
    fn the_fresh_virtualizable_hint_carries_access_directly_too() {
        let bk = bk();
        let s = instance_of(&bk, "PyFrame", true, &[]);
        let out = hint_virtualizable_flags(&[Some(s)], true);
        assert_eq!(
            flags_of(&out),
            vec![
                "access_directly".to_string(),
                "fresh_virtualizable".to_string()
            ]
        );
    }

    /// The `else` arm: a hint on a class with no `_virtualizable_` STRIPS
    /// both flags rather than leaving them in place.
    #[test]
    fn the_hint_strips_the_flags_off_a_non_virtualizable() {
        let bk = bk();
        let s = instance_of(
            &bk,
            "Plain",
            false,
            &["access_directly", "fresh_virtualizable", "other"],
        );
        let out = hint_virtualizable_flags(&[Some(s)], false);
        assert_eq!(flags_of(&out), vec!["other".to_string()]);
    }

    /// `rlib/debug.py` asserts; the port raises instead of dropping the
    /// flag, because the failure is a defect in the caller.
    #[test]
    fn check_not_access_directly_rejects_a_flagged_argument() {
        let bk = bk();
        let flagged = instance_of(&bk, "PyFrame", true, &["access_directly"]);
        assert!(
            rlib_check_not_access_directly(&bk, &[Some(flagged)], &HashMap::new()).is_err(),
            "a flagged argument must abort, not pass"
        );

        let plain = instance_of(&bk, "PyFrame", true, &[]);
        assert_eq!(
            rlib_check_not_access_directly(&bk, &[Some(plain.clone())], &HashMap::new())
                .expect("an unflagged argument passes"),
            plain,
            "the entry is an identity on the annotation it accepts"
        );
    }

    /// Upstream's `isinstance(s_x, SomeInstance)` guard — anything else
    /// returns unchanged.
    #[test]
    fn the_hint_leaves_a_non_instance_alone() {
        let s = SomeValue::Integer(SomeInteger::default());
        let out = hint_virtualizable_flags(&[Some(s.clone())], false);
        assert_eq!(out, super::super::model::not_const(&s));
    }

    fn no_kwds() -> HashMap<String, Option<SomeValue>> {
        HashMap::new()
    }

    /// `ConditionalCallEntry.compute_result_annotation` for
    /// `_jit_conditional_call` is void.
    #[test]
    fn jit_conditional_call_annotates_s_none() {
        let out = jit_conditional_call(&bk(), &[], &no_kwds()).unwrap();
        assert!(
            matches!(out, SomeValue::None_(_)),
            "void conditional_call must be s_None, got {out:?}"
        );
    }

    /// `ConditionalCallEntry.compute_result_annotation` for
    /// `_jit_conditional_call_value` unions the callee result with
    /// `args_s[0].nonnoneify()`.
    #[test]
    fn jit_conditional_call_elidable_unions_callee_result_and_value() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget, SomePtr,
        };

        let s_value = SomeValue::Integer(SomeInteger::new(true, false));
        let s_fn = SomeValue::Ptr(SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            }),
        }));
        let s_arg = SomeValue::Integer(SomeInteger::new(false, false));
        let out = jit_conditional_call_elidable(
            &bk(),
            &[Some(s_value), Some(s_fn), Some(s_arg)],
            &no_kwds(),
        )
        .unwrap();
        match out {
            SomeValue::Integer(i) => assert!(
                !i.nonneg,
                "union of nonneg value and maybe-neg callee result drops nonneg"
            ),
            other => panic!("expected SomeInteger union, got {other:?}"),
        }
    }

    #[test]
    fn malloc_typed_rejects_classdef_less_instance() {
        // lltype.py `malloc(T)`: only `isinstance(T, Struct)` is
        // mallocable. A classdef-less `SomeInstance` (object-only) carries
        // no concrete struct, so it takes the `else: raise
        // TypeError("malloc: unmallocable type")` arm.
        let bk = bk();
        let obj_only = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let err = malloc_typed_alloc(&bk, &[Some(obj_only)], &no_kwds())
            .expect_err("classdef-less instance must be rejected");
        assert!(
            format!("{err:?}").contains("unmallocable type"),
            "expected unmallocable-type error, got {err:?}"
        );
    }

    #[test]
    fn registry_contains_mass_registered_builtins() {
        // upstream builtin.py:191-195 scans `builtin_*` prefixes.
        assert!(is_registered("range"));
        assert!(is_registered("xrange"));
        assert!(is_registered("int"));
        assert!(is_registered("float"));
        assert!(is_registered("chr"));
        assert!(is_registered("tuple"));
        assert!(is_registered("list"));
        assert!(is_registered("zip"));
        assert!(is_registered("min"));
        assert!(is_registered("max"));
        assert!(is_registered(crate::runtime_names::modules::MALLOC));
    }

    #[test]
    fn registry_contains_analyzer_for_entries() {
        // upstream `@analyzer_for(...)` decorations.  Each registry key
        // matches the qualname `HostObject::new_builtin_callable(...)`
        // assigns to its module attribute (see flowspace/model.rs
        // bootstrap), so `immutablevalue_hostobject`'s
        // `is_registered(qualname)` hits.
        assert!(is_registered("object.__init__"));
        assert!(is_registered("sys.getdefaultencoding"));
        assert!(is_registered("rarithmetic.intmask"));
        assert!(is_registered("majit_rlib.jit.conditional_call1"));
        assert!(is_registered("majit_rlib.jit.conditional_call_elidable1"));
        assert!(is_registered("rpython.rlib.objectmodel.instantiate"));
        assert!(is_registered("weakref.ref"));
        assert!(is_registered("pdb.set_trace"));
    }

    #[test]
    fn registry_rejects_unknown_qualname() {
        assert!(!is_registered("completely.made.up.name"));
        assert!(lookup("completely.made.up.name").is_none());
    }

    #[test]
    fn lltype_cast_pointer_types_result_from_constant_target_class() {
        // `ann_cast_pointer` (lltype.py): result annotated from
        // the constant target class; operand nullability survives.
        let bk = bk();
        let host = bk.intern_class_by_qualname("W_CastTarget");
        let s_cls = bk.immutablevalue(&ConstValue::HostObject(host)).unwrap();
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, true, Default::default()));
        let result = lltype_cast_pointer(&bk, &[Some(s_cls), Some(s_ptr)], &no_kwds()).unwrap();
        match result {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.can_be_none,
                    "operand nullability must survive the cast"
                );
                let cdef = inst
                    .classdef
                    .expect("result must carry the target class's unique classdef");
                assert_eq!(cdef.borrow().name, "W_CastTarget");
            }
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn lltype_cast_pointer_rejects_non_class_target() {
        // upstream `assert s_CURTYPE.is_constant()` — a non-PBC first
        // argument is not a constant target type.
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(1)).unwrap();
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let err = lltype_cast_pointer(&bk, &[Some(s_int), Some(s_ptr)], &no_kwds()).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("constant class PBC"),
        );
    }

    #[test]
    fn lltype_cast_pointer_rejects_non_pointer_operand() {
        // `assert isinstance(s_p, SomePtr)` (lltype.py) — a
        // non-pointer second argument is rejected, not silently
        // annotated as the target class.
        let bk = bk();
        let host = bk.intern_class_by_qualname("W_CastTarget");
        let s_cls = bk.immutablevalue(&ConstValue::HostObject(host)).unwrap();
        let s_int = bk.immutablevalue(&ConstValue::Int(1)).unwrap();
        let err = lltype_cast_pointer(&bk, &[Some(s_cls), Some(s_int)], &no_kwds()).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("casting of non-pointer"),
        );
    }

    #[test]
    fn lltype_cast_pointer_accepts_constant_none_as_null() {
        // A constant-None operand is a null pointer (upstream null
        // constants are SomePtr and pass the lltype.py assert);
        // the result is the target class, can_be_none=true.
        let bk = bk();
        let host = bk.intern_class_by_qualname("W_CastTarget");
        let s_cls = bk.immutablevalue(&ConstValue::HostObject(host)).unwrap();
        let s_none = bk.immutablevalue(&ConstValue::None).unwrap();
        let result = lltype_cast_pointer(&bk, &[Some(s_cls), Some(s_none)], &no_kwds()).unwrap();
        match result {
            SomeValue::Instance(inst) => {
                assert!(inst.can_be_none, "null downcast stays nullable");
                let cdef = inst.classdef.expect("target classdef");
                assert_eq!(cdef.borrow().name, "W_CastTarget");
            }
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn builtin_range_single_const_argument_returns_nonneg_list() {
        // upstream builtin_range(10) → list with SomeInteger(nonneg=True),
        // range_step=1.
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let result = builtin_range(&bk, &[Some(s_stop)], &no_kwds()).unwrap();
        match result {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(matches!(item, SomeValue::Integer(ref i) if i.nonneg));
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_range_negative_stop_is_empty_list() {
        // upstream: `len(xrange(0, -5, 1)) == 0` → empty list → s_item =
        // s_ImpossibleValue.
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(-5)).unwrap();
        let result = builtin_range(&bk, &[Some(s_stop)], &no_kwds()).unwrap();
        match result {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(matches!(item, SomeValue::Impossible));
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_range_step_zero_errors() {
        let bk = bk();
        let a = bk.immutablevalue(&ConstValue::Int(0)).unwrap();
        let b = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let c = bk.immutablevalue(&ConstValue::Int(0)).unwrap();
        let err = builtin_range(&bk, &[Some(a), Some(b), Some(c)], &no_kwds()).unwrap_err();
        assert!(err.msg.as_deref().unwrap_or("").contains("step zero"));
    }

    #[test]
    fn builtin_int_const_string_returns_constant_integer() {
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("42")).unwrap();
        let result = builtin_int(&bk, &[Some(s_str)], &no_kwds()).unwrap();
        match result {
            SomeValue::Integer(i) => {
                assert!(i.base.const_box.is_some());
                if let Some(ConstValue::Int(n)) = i.base.const_box.as_ref().map(|c| &c.value) {
                    assert_eq!(*n, 42);
                }
            }
            other => panic!("expected constant SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_non_const_integer_returns_nonneg_somesinteger() {
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let result = builtin_int(&bk, &[Some(s_nonneg)], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::Integer(ref i) if i.nonneg));
    }

    #[test]
    fn builtin_chr_const_integer_returns_constant_char() {
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(65)).unwrap();
        let result = builtin_chr(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match result {
            SomeValue::Char(ch) => {
                assert!(ch.inner.base.const_box.is_some());
                if let Some(ConstValue::ByteStr(s)) =
                    ch.inner.base.const_box.as_ref().map(|c| &c.value)
                {
                    assert_eq!(s.as_slice(), b"A");
                }
            }
            other => panic!("expected constant SomeChar, got {other:?}"),
        }
    }

    #[test]
    fn builtin_list_on_somelist_uses_offspring() {
        let bk = bk();
        let listdef = super::super::listdef::ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let s = SomeValue::List(super::super::model::SomeList::new(listdef));
        let out = builtin_list(&bk, &[Some(s)], &no_kwds()).unwrap();
        assert!(matches!(out, SomeValue::List(_)));
    }

    #[test]
    fn cast_instance_intrinsic_wtf8_root_projects_root_instance_to_somestring() {
        // The string-root marker `cast_instance_call_result("Wtf8", …,
        // ValueType::Str)`.  `project_struct_field_type("Wtf8")` is
        // SomeString; a PyObject instance operand must narrow to that
        // shell rather than keep SomeInstance.
        let bk = bk();
        let classdef = ClassDef::new_standalone("pyobject::PyObject", None);
        let s_obj =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("Wtf8"))
            .expect("Wtf8 root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_obj), Some(s_root)],
            &no_kwds(),
        )
        .expect("string-root cast_instance_intrinsic must accept a PyObject instance");
        assert!(
            matches!(out, SomeValue::String(_)),
            "Wtf8 root must project dest as SomeString, got {out:?}"
        );
    }

    #[test]
    fn cast_instance_intrinsic_bytesblock_root_accepts_byte_list() {
        let bk = bk();
        let listdef = super::super::listdef::ListDef::new(
            Some(bk.clone()),
            super::super::model::s_uint(),
            false,
            false,
        );
        let s_list = SomeValue::List(super::super::model::SomeList::new(listdef));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("BytesBlock"))
            .expect("BytesBlock root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_list), Some(s_root)],
            &no_kwds(),
        )
        .expect("BytesBlock cast must accept a byte list");
        assert!(
            matches!(out, SomeValue::String(_)),
            "BytesBlock root must project dest as SomeString, got {out:?}"
        );
    }

    #[test]
    fn cast_instance_intrinsic_tuple_root_narrows_to_sometuple() {
        // A residual call's `(A, B)` result arrives as a classdef-less
        // pointer; the `Tuple<A,B>` root narrows it to the RPython tuple,
        // and an operand that already is that tuple passes through.
        let bk = bk();
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("Tuple<i64,bool>"))
            .expect("tuple root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr), Some(s_root.clone())],
            &no_kwds(),
        )
        .expect("tuple-root cast_instance_intrinsic must accept a pointer");
        let SomeValue::Tuple(tuple) = &out else {
            panic!("Tuple<i64,bool> root must project to SomeTuple, got {out:?}");
        };
        assert_eq!(tuple.items.len(), 2);
        let again = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(out.clone()), Some(s_root)],
            &no_kwds(),
        )
        .expect("a SomeTuple operand passes through");
        assert_eq!(again, out);
    }

    #[test]
    fn cast_instance_intrinsic_raw_root_narrows_instance_to_someptr() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, PtrTarget};

        let owner = "rawcast::Scope";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            owner.to_string(),
            vec![("save_point".to_string(), "usize".to_string())],
        );
        reg.raw_word_owners.insert(owner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            owner_id,
            crate::call::StructLayout {
                size: 8,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![crate::call::StructFieldLayout {
                    name: "save_point".into(),
                    offset: 0,
                    size: 8,
                    flag: majit_ir::descr::ArrayFlag::Unsigned,
                    field_type: majit_ir::value::Type::Int,
                    rank: None,
                }],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        let classdef = ClassDef::new_standalone(owner, None);
        let s_obj =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str(owner))
            .expect("raw root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_obj), Some(s_root)],
            &no_kwds(),
        )
        .expect("raw-root cast_instance_intrinsic must accept a SomeInstance operand");
        let SomeValue::Ptr(ptr) = out else {
            panic!("raw root must project dest as SomePtr, got {out:?}");
        };
        let PtrTarget::Struct(st) = &ptr.ll_ptrtype.TO else {
            panic!("raw-root SomePtr must point at a Struct");
        };
        assert_eq!(st._name, owner);
        assert_eq!(st._gckind, GcKind::Raw);
    }

    #[test]
    fn cast_instance_intrinsic_object_items_list_root_is_a_typed_list() {
        // `*items_block_items_base(block).add(idx)` narrows its base through
        // `[PyObject]`. The operand is the accessor's nullable classdef-less
        // pointer. `project_struct_field_type` must answer `SomeList` of the
        // element class so `getitem` takes ListRepr (`ll_getitem_fast`,
        // `rlist.py`) instead of `getitem_SomeInstance`.
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        // Production publishes this origin before annotation, so the bare
        // element leaf and `pyobject::PyObject` are one ClassDef.
        let _origins = crate::test_support::register_struct_origins_serialized(HashMap::from([(
            "PyObject".to_string(),
            "pyobject".to_string(),
        )]));
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        reg.fields.insert(
            "PyObject".to_string(),
            reg.fields["pyobject::PyObject"].clone(),
        );
        bk.set_struct_fields(Rc::new(reg));
        let classdef = bk
            .getuniqueclassdef_for_struct_root("pyobject::PyObject")
            .expect("PyObject classdef");
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, true, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("[PyObject]"))
            .expect("list-root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr), Some(s_root)],
            &no_kwds(),
        )
        .expect("object-items list root must accept a classdef-less pointer");
        let SomeValue::List(list) = out else {
            panic!("[PyObject] must project to SomeList, got {out:?}");
        };
        let SomeValue::Instance(item) = list.listdef.s_value() else {
            panic!("object-items element must be an instance");
        };
        assert!(
            item.classdef
                .as_ref()
                .is_some_and(|got| Rc::ptr_eq(got, &classdef)),
            "object-items element must be the PyObject class"
        );
    }

    #[test]
    fn cast_instance_intrinsic_list_root_reuses_position_listdef() {
        // The same `[PyObject]` dest at one op is `Bookkeeper.getlistdef`
        // of that position, so a reflow `contains()` sees one ListDef.
        use crate::annotator::bookkeeper::PositionKey;
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(7, 8, 9)));
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, true, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("[PyObject]"))
            .expect("list-root constant");
        let first = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr.clone()), Some(s_root.clone())],
            &no_kwds(),
        )
        .expect("first [PyObject] cast");
        let second = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr), Some(s_root)],
            &no_kwds(),
        )
        .expect("second [PyObject] cast");
        let (SomeValue::List(a), SomeValue::List(b)) = (first, second) else {
            panic!("[PyObject] must project to SomeList");
        };
        assert!(
            a.listdef.same_as(&b.listdef),
            "two casts of one list spelling at one position share a ListDef"
        );
        assert!(
            a.listdef.same_as(&bk.getlistdef(None)),
            "the cast ListDef is Bookkeeper.getlistdef of the current position"
        );
    }

    #[test]
    fn cast_instance_intrinsic_object_slice_root_narrows_to_somelist() {
        // `emit_object_vec_gcarray` narrows `gcarray_from_pyobject_vec`
        // through this marker with the destination slice's `[Class]`
        // root. That spelling is `project_struct_field_type`'s fixed
        // object list, the `FixedSizeListRepr` identity a
        // `&[PyObjectRef]` parameter already carries; `newlist` reuses
        // the position ListDef (`Bookkeeper.getlistdef`).
        let bk = bk();
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("[PyObject]"))
            .expect("object-slice root constant");
        let out = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr), Some(s_root.clone())],
            &no_kwds(),
        )
        .expect("object-slice-root cast_instance_intrinsic must accept a pointer");
        assert!(
            matches!(out, SomeValue::List(_)),
            "[PyObject] root must project dest as SomeList, got {out:?}"
        );
        let again = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(out.clone()), Some(s_root)],
            &no_kwds(),
        )
        .expect("a SomeList operand passes through");
        assert_eq!(again, out);
    }

    #[test]
    fn cast_instance_intrinsic_list_root_contains_after_item_noneify() {
        // A re-flow of `__cast_instance_intrinsic(v, "[PyObject]")`
        // at the same position must return the same ListDef
        // (`Bookkeeper.getlistdef`). After the item is generalized to
        // `can_be_none=true`, the next call still contains the first
        // binding — the `setbinding` check that used to panic.
        use crate::annotator::bookkeeper::PositionKey;
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        reg.fields.insert(
            "PyObject".to_string(),
            reg.fields["pyobject::PyObject"].clone(),
        );
        bk.set_struct_fields(Rc::new(reg));
        bk.set_position_key(Some(PositionKey::new(1, 2, 0)));

        let s_ptr = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("[PyObject]"))
            .expect("object-slice root constant");
        let args = [Some(s_ptr), Some(s_root)];
        let first = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &args,
            &no_kwds(),
        )
        .expect("first list-root cast");
        let second = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &args,
            &no_kwds(),
        )
        .expect("second list-root cast at the same position");
        let SomeValue::List(first_list) = &first else {
            panic!("expected SomeList, got {first:?}");
        };
        let SomeValue::List(second_list) = &second else {
            panic!("expected SomeList, got {second:?}");
        };
        assert!(
            first_list.listdef.same_as(&second_list.listdef),
            "re-flow at the same position must reuse the position's ListDef"
        );
        let SomeValue::Instance(item) = first_list.listdef.s_value() else {
            panic!(
                "list item must be SomeInstance(PyObject), got {:?}",
                first_list.listdef.s_value()
            );
        };
        assert!(
            !item.can_be_none,
            "projected PyObject item starts non-noneable"
        );

        let noneable = first_list
            .listdef
            .s_value()
            .noneify()
            .expect("SomeInstance.noneify");
        first_list
            .listdef
            .generalize(&noneable)
            .expect("generalize item to can_be_none=true");
        let SomeValue::Instance(item) = first_list.listdef.s_value() else {
            panic!("item must stay SomeInstance after noneify generalize");
        };
        assert!(item.can_be_none);

        let after = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &args,
            &no_kwds(),
        )
        .expect("re-flow after item noneify");
        let SomeValue::List(after_list) = &after else {
            panic!("expected SomeList, got {after:?}");
        };
        assert!(
            after_list.listdef.same_as(&first_list.listdef),
            "re-flow after generalize must still reuse the position's ListDef"
        );
        assert!(
            after.contains(&first),
            "setbinding: new value must contain old after the listdef item noneifies"
        );
    }

    #[test]
    fn cast_instance_reflow_reuses_one_listdef() {
        // Reflow of the same root must hand `setbinding` the same ListDefs.
        // `Bookkeeper.getlistdef` (`bookkeeper.py`) is that cache.
        let bk = bk();
        let s_ptr = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("Tuple<Vec<i64>,Vec<bool>>"))
            .expect("tuple root constant");
        let first = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr.clone()), Some(s_root.clone())],
            &no_kwds(),
        )
        .expect("first projection");
        let second = call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_ptr), Some(s_root)],
            &no_kwds(),
        )
        .expect("reflow projection");
        assert_eq!(first, second);
    }

    #[test]
    fn call_builtin_unknown_name_errors() {
        let bk = bk();
        let err = call_builtin(&bk, "definitely_not_a_builtin", &[], &no_kwds()).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no analyser registered")
        );
    }

    #[test]
    fn copy_nonoverlapping_returns_none_for_an_instance_and_a_string() {
        let bk = bk();
        let inst = SomeValue::Instance(SomeInstance::new(None, false, Default::default()));
        let text = SomeValue::String(SomeString::new(false, false));
        let count = SomeValue::Integer(SomeInteger::new(true, true));
        for (src, dst) in [(&inst, &inst), (&text, &text), (&inst, &text)] {
            let result = call_builtin(
                &bk,
                "std.ptr.copy_nonoverlapping",
                &[Some(src.clone()), Some(dst.clone()), Some(count.clone())],
                &no_kwds(),
            )
            .expect("copy_nonoverlapping annotates");
            assert!(
                matches!(result, SomeValue::None_(_)),
                "copy returns (), got {result:?}"
            );
        }
        let err = call_builtin(
            &bk,
            "std.ptr.copy_nonoverlapping",
            &[Some(inst), Some(count.clone())],
            &no_kwds(),
        )
        .expect_err("two arguments are not the signature");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("expects (src, dst, count)")
        );
    }

    #[test]
    fn call_builtin_dispatches_range() {
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        let result = call_builtin(&bk, "range", &[Some(s_stop)], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::List(_)));
    }

    #[test]
    fn object_init_returns_none() {
        let bk = bk();
        let result = object_init(&bk, &[], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::None_(_)));
    }

    #[test]
    fn pdb_set_trace_always_errors() {
        let bk = bk();
        let err = pdb_set_trace(&bk, &[], &no_kwds()).unwrap_err();
        assert!(err.msg.as_deref().unwrap_or("").contains("pdb.set_trace"));
    }

    #[test]
    fn constpropagate_returns_s_result_when_args_not_constant() {
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_input = SomeValue::Integer(SomeInteger::default());
        let s_result = SomeValue::Integer(SomeInteger::default());
        let out = constpropagate(&bk, &[s_input], s_result.clone(), |_| {
            panic!("evaluator must not be called when args are non-constant");
        })
        .unwrap();
        assert_eq!(out.tag(), s_result.tag());
    }

    #[test]
    fn somebuiltin_call_dispatches_through_registry() {
        // End-to-end wiring check: SomeBuiltin("range") carried inside
        // `SomeValue::call` reaches `call_builtin("range", ...)` and the
        // registered analyser runs. Mirrors upstream's
        // `SomeBuiltin.call(args)` path at unaryop.py.
        use crate::annotator::argument::{ArgumentsForTranslation, simple_args};
        use crate::annotator::model::{SomeBuiltin, SomeValue};

        let bk = bk();
        bk.enter(None);

        let s_stop = bk.immutablevalue(&ConstValue::Int(4)).unwrap();
        let args: ArgumentsForTranslation = simple_args(vec![s_stop]);
        let s_callee = SomeValue::Builtin(SomeBuiltin::new("range", None, Some("range".into())));
        let s_result = s_callee
            .call(&args)
            .expect("SomeBuiltin.call should dispatch");
        assert!(matches!(s_result, Some(SomeValue::List(_))));

        bk.leave();
    }

    #[test]
    fn somebuiltin_call_returns_error_for_unregistered_qualname() {
        use crate::annotator::argument::{ArgumentsForTranslation, simple_args};
        use crate::annotator::model::{SomeBuiltin, SomeValue};

        let bk = bk();
        bk.enter(None);

        let args: ArgumentsForTranslation = simple_args(vec![]);
        let s_callee = SomeValue::Builtin(SomeBuiltin::new(
            "completely.made.up.name",
            None,
            Some("completely.made.up.name".into()),
        ));
        let err = s_callee
            .call(&args)
            .expect_err("unregistered analyser should error");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no analyser registered")
        );

        bk.leave();
    }

    #[test]
    fn builtin_list_on_iterator_routes_through_next() {
        // `list(reversed(xs))` should keep the element type
        // instead of collapsing to SomeImpossibleValue. Upstream chains
        // `s_iterable.iter().next()` which delegates to
        // `SomeIterator.next()` for SomeIterator inputs.
        use crate::annotator::model::{SomeInteger, SomeList};
        let bk = bk();
        let inner = super::super::listdef::ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let inner_list = SomeValue::List(SomeList::new(inner));
        let s_iter =
            SomeValue::Iterator(SomeIterator::new(inner_list, vec!["reversed".to_string()]));
        let out = builtin_list(&bk, &[Some(s_iter)], &no_kwds()).unwrap();
        match out {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(
                    matches!(item, SomeValue::Integer(ref i) if i.nonneg),
                    "expected nonneg SomeInteger element, got {item:?}"
                );
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_max_keeps_nonneg_when_any_operand_is_nonneg() {
        // upstream `nonneg |= s1.nonneg` is an OR-fold —
        // `max(nonneg, maybe_negative)` stays nonneg.
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let a = SomeValue::Integer(SomeInteger::new(true, false));
        let b = SomeValue::Integer(SomeInteger::new(false, false));
        let out = builtin_max(&bk, &[Some(a), Some(b)], &no_kwds()).unwrap();
        match out {
            SomeValue::Integer(i) => assert!(
                i.nonneg,
                "max(nonneg, maybe_neg) must remain nonneg per builtin.py:182-188"
            ),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_rejects_out_of_range_float_constant() {
        // `int(1e20)` would saturate to i64::MAX if we let the
        // Rust cast do its default behaviour — upstream raises OverflowError
        // which constpropagate catches. Non-finite / out-of-range floats
        // must fall through to the conservative `s_result`.
        let bk = bk();
        let huge = ConstValue::float(1.0e20_f64);
        let s_huge = bk.immutablevalue(&huge).unwrap();
        let out = builtin_int(&bk, &[Some(s_huge)], &no_kwds()).unwrap();
        match out {
            SomeValue::Integer(i) => {
                assert!(
                    i.base.const_box.is_none(),
                    "int(out-of-range float) must not pin a constant"
                );
            }
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn registered_class_backed_builtin_lifts_to_somebuiltin() {
        // class-backed builtins (`range`, `int`, `list`, …)
        // are `HostObjectKind::Class` in HOST_ENV but still register
        // analysers. `immutablevalue` must route them to `SomeBuiltin`
        // so that `SomeBuiltin.call()` reaches the analyser, matching
        // upstream bookkeeper.py's BUILTIN_ANALYZERS check
        // firing before the `tp is type` branch.
        use crate::flowspace::model::HOST_ENV;
        let bk = bk();
        let range_host = HOST_ENV
            .lookup_builtin("range")
            .expect("HOST_ENV must expose `range`");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(range_host))
            .expect("range class must annotate");
        assert!(
            matches!(s, SomeValue::Builtin(ref b) if b.analyser_name == "range"),
            "expected SomeBuiltin(range), got {s:?}"
        );
    }

    #[test]
    fn builtin_unichr_const_integer_returns_constant_unicode_cp() {
        // `unichr(65)` must keep the constant via the
        // UnicodeCodePoint variant; constpropagate's re-annotation
        // through immutablevalue needs the explicit unicode tag.
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(65)).unwrap();
        let out = builtin_unichr(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match out {
            SomeValue::UnicodeCodePoint(cp) => {
                match cp.inner.base.const_box.as_ref().map(|c| &c.value) {
                    Some(ConstValue::UniStr(s)) => assert_eq!(s, "A"),
                    other => panic!("expected constant unicode 'A', got {other:?}"),
                }
            }
            other => panic!("expected SomeUnicodeCodePoint, got {other:?}"),
        }
    }

    #[test]
    fn builtin_unicode_const_string_returns_constant_unicode_string() {
        // `unicode("abc")` must return a constant
        // SomeUnicodeString, not raise AnnotatorError.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("abc")).unwrap();
        let out = builtin_unicode(&bk, &[Some(s_str)], &no_kwds()).unwrap();
        match out {
            SomeValue::UnicodeString(us) => {
                match us.inner.base.const_box.as_ref().map(|c| &c.value) {
                    Some(ConstValue::UniStr(s)) => assert_eq!(s, "abc"),
                    other => panic!("expected constant unicode 'abc', got {other:?}"),
                }
            }
            other => panic!("expected SomeUnicodeString, got {other:?}"),
        }
    }

    #[test]
    fn builtin_hasattr_constant_observes_data_descriptor_before_instance_dict() {
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;

        let bk = bk();
        let class = HostObject::new_class("owner.FrozenAttributes", vec![]);
        class.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "owner.FrozenAttributes._freeze_",
                Arc::new(|_| Ok(ConstValue::Bool(true))),
            )),
        );
        class.class_set(
            "value",
            ConstValue::HostObject(HostObject::new_property("value", None, None, None)),
        );
        let instance = HostObject::new_instance(class, vec![]);
        instance.instance_set("value", ConstValue::Int(7)).unwrap();
        let s_obj = bk
            .immutablevalue(&ConstValue::HostObject(instance))
            .unwrap();
        assert!(s_obj.is_immutable_constant());
        let s_attr = bk.immutablevalue(&ConstValue::byte_str("value")).unwrap();
        let out = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds()).unwrap();
        assert!(matches!(out.const_(), Some(ConstValue::Bool(false))));
    }

    #[test]
    fn builtin_hasattr_frozen_family_distinguishes_missing_from_unsupported() {
        use crate::annotator::description::FrozenDesc;
        use crate::annotator::model::SomePBC;
        use crate::flowspace::model::{HostGetAttrError, HostObject};
        use std::cell::RefCell;

        // Two descriptions keep the PBC nonconstant and exercise upstream's
        // FrozenDesc-family branch, not its singleton host-hasattr branch.
        for modes in [[0, 0], [1, 1], [0, 1], [0, 2]] {
            let bk = bk();
            let descriptions = modes.into_iter().enumerate().map(|(i, mode)| {
                let desc = FrozenDesc::new_with_read_attribute(
                    bk.clone(),
                    HostObject::new_module(format!("native_{i}")),
                    Box::new(move |_| match mode {
                        0 => Err(HostGetAttrError::Missing),
                        1 => Ok(ConstValue::Int(42)),
                        _ => Err(HostGetAttrError::Unsupported),
                    }),
                )
                .unwrap();
                DescEntry::Frozen(Rc::new(RefCell::new(desc)))
            });
            let s_obj = SomeValue::PBC(SomePBC::new(descriptions, false));
            assert!(!s_obj.is_constant());
            let s_attr = bk.immutablevalue(&ConstValue::byte_str("field")).unwrap();
            let result = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds());
            if modes.contains(&2) {
                assert!(result.unwrap_err().to_string().contains("unsupported"));
            } else {
                let out = result.unwrap();
                match modes {
                    [0, 0] => assert!(matches!(out.const_(), Some(ConstValue::Bool(false)))),
                    [1, 1] => assert!(matches!(out.const_(), Some(ConstValue::Bool(true)))),
                    _ => assert!(!out.is_constant()),
                }
            }
        }
    }

    #[test]
    fn builtin_hasattr_folds_constant_when_attr_exists_on_class() {
        // for immutable constant receivers, `hasattr(C, "m")`
        // must fold to a constant SomeBool. The HOST_ENV `range` class
        // has no attribute members in this minimal env, so assert the
        // False branch — checks the folding mechanism regardless of
        // which members the host reflects.
        use crate::flowspace::model::HOST_ENV;
        let bk = bk();
        let obj = HOST_ENV.lookup_builtin("range").unwrap();
        let s_obj = bk.immutablevalue(&ConstValue::HostObject(obj)).unwrap();
        let s_attr = bk
            .immutablevalue(&ConstValue::byte_str("no_such_attr"))
            .unwrap();
        let out = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(false)) => {}
                other => panic!("expected constant Bool(false) for missing attr, got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn builtin_bool_folds_constant_none_to_false() {
        // `bool(None)` must pin `r.const = False` instead of
        // returning an unrefined SomeBool.
        let bk = bk();
        let s_none_val = bk.immutablevalue(&ConstValue::None).unwrap();
        let out = builtin_bool(&bk, &[Some(s_none_val)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(false)) => {}
                other => panic!("expected constant Bool(false), got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn builtin_bool_folds_constant_int_to_true() {
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(42)).unwrap();
        let out = builtin_bool(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(true)) => {}
                other => panic!("expected constant Bool(true), got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn call_builtin_rejects_unknown_keyword() {
        // `list(xs, foo=1)` must raise instead of silently
        // ignoring `foo=1`. The dispatcher now rejects kwds outside
        // each analyser's allow-list.
        let bk = bk();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert(
            "s_foo".into(),
            Some(bk.immutablevalue(&ConstValue::Int(1)).unwrap()),
        );
        let err = call_builtin(&bk, "list", &[], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("unexpected keyword argument"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn call_builtin_accepts_declared_keyword() {
        // `int(s, base=16)` is the positive path — `base` is declared
        // allowed for `int` in the allow-list.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("1a")).unwrap();
        let s_base = bk.immutablevalue(&ConstValue::Int(16)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_base".into(), Some(s_base));
        let out = call_builtin(&bk, "int", &[Some(s_str)], &kwds).unwrap();
        match out {
            SomeValue::Integer(i) => match i.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Int(n)) => assert_eq!(*n, 26),
                other => panic!("expected constant Int(26), got {other:?}"),
            },
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_rejects_positional_and_keyword_base() {
        // `int(x, 10, base=16)` must raise the duplicate-arg
        // error that upstream's Python dispatch produces.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("1a")).unwrap();
        let s_base_pos = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let s_base_kw = bk.immutablevalue(&ConstValue::Int(16)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_base".into(), Some(s_base_kw));
        let err = builtin_int(&bk, &[Some(s_str), Some(s_base_pos)], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("multiple values for argument 'base'"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn r_dict_helper_reads_force_non_null_from_keyword() {
        // r_dict(eq, hash, force_non_null=True) must honour
        // the kwd. Previously ignored, leaving the DictDef flag false.
        //
        // Testing through call_builtin would trip
        // `update_rdict_annotations` because our stub eq/hash aren't
        // SomePBC. Instead, extract the flag-binding logic directly by
        // reading the DictDef getdictdef returns once we reproduce the
        // kwd/positional merge. The builder side-effect (getdictdef
        // stores the DictDef in bk.dictdefs keyed on position_key)
        // lets us read back the flags even though update_rdict will
        // fail downstream.
        let bk = bk();
        bk.enter(None);
        let s_force = bk.immutablevalue(&ConstValue::Bool(true)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_force_non_null".into(), Some(s_force));

        // Mirror r_dict_helper's flag extraction — the unit under test.
        let force_non_null = kwds
            .get("s_force_non_null")
            .and_then(|s| s.as_ref())
            .and_then(|s| s.const_().cloned())
            .map(|cv| matches!(cv, ConstValue::Bool(true)))
            .unwrap_or(false);
        assert!(
            force_non_null,
            "kwd extraction must yield force_non_null=true"
        );

        let dd = bk.getdictdef(true, force_non_null, false);
        assert!(
            dd.inner.force_non_null,
            "DictDef::force_non_null must persist the kwd value"
        );
        bk.leave();
    }

    #[test]
    fn builtin_enumerate_rejects_non_constant_start_keyword() {
        // `enumerate(xs, start=<non-constant>)` must raise
        // the upstream "second argument to enumerate must be constant"
        // error instead of silently dropping the kwd.
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start_nonconst = SomeValue::Integer(SomeInteger::default());
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_start".into(), Some(s_start_nonconst));
        let err = builtin_enumerate(&bk, &[Some(s_xs)], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("second argument to enumerate must be constant"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn builtin_enumerate_preserves_constant_start_payload() {
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        let result = builtin_enumerate(&bk, &[Some(s_xs), Some(s_start)], &no_kwds()).unwrap();
        let SomeValue::Iterator(it) = result else {
            panic!("expected SomeIterator");
        };
        assert_eq!(it.variant, vec!["enumerate".to_string()]);
        assert_eq!(it.enumerate_start, Some(ConstValue::Int(3)));
    }

    #[test]
    fn builtin_enumerate_keyword_preserves_constant_start_payload() {
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start = bk.immutablevalue(&ConstValue::Int(5)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_start".into(), Some(s_start));
        let result = builtin_enumerate(&bk, &[Some(s_xs)], &kwds).unwrap();
        let SomeValue::Iterator(it) = result else {
            panic!("expected SomeIterator");
        };
        assert_eq!(it.variant, vec!["enumerate".to_string()]);
        assert_eq!(it.enumerate_start, Some(ConstValue::Int(5)));
    }

    #[test]
    fn ann_cast_ptr_to_int_of_somerustvec_is_someinteger() {
        use crate::annotator::model::SomeRustVec;
        let s_vec = SomeValue::RustVec(SomeRustVec::for_kind(majit_ir::rvec::VecItemKind::Int));
        let out = lltype_cast_ptr_to_int(&bk(), &[Some(s_vec)], &no_kwds())
            .expect("ann_cast_ptr_to_int returns SomeInteger");
        assert!(
            matches!(out, SomeValue::Integer(_)),
            "lltype.py ann_cast_ptr_to_int yields SomeInteger, got {out:?}"
        );
    }

    #[test]
    fn ann_direct_ptradd_rejects_someinstance() {
        use crate::annotator::model::SomePtr;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::lltypesystem::rffi::CCHARP;

        let LowLevelType::Ptr(ptr_ty) = (*CCHARP).clone() else {
            panic!("CCHARP is a pointer");
        };
        let s_p = SomeValue::Ptr(SomePtr::new(*ptr_ty));
        let s_n = SomeValue::Integer(super::super::model::SomeInteger::default());
        let out = lltype_direct_ptradd(&bk(), &[Some(s_p.clone()), Some(s_n.clone())], &no_kwds())
            .expect("ann_direct_ptradd returns s_p");
        assert_eq!(out, s_p);

        let s_shell = SomeValue::Instance(SomeInstance::new(None, true, Default::default()));
        let shelled = lltype_direct_ptradd(&bk(), &[Some(s_shell), Some(s_n.clone())], &no_kwds())
            .expect("classdef-less instance is the alloc_zeroed shell");
        assert!(matches!(shelled, SomeValue::Ptr(_)));

        let classdef = ClassDef::new_standalone("pyobject::PyObject", None);
        let s_inst =
            SomeValue::Instance(SomeInstance::new(Some(classdef), true, Default::default()));
        let err = lltype_direct_ptradd(&bk(), &[Some(s_inst), Some(s_n)], &no_kwds())
            .expect_err("ann_direct_ptradd asserts SomePtr");
        assert!(err.to_string().contains("non-pointer"), "got {err}");
    }

    #[test]
    fn builtin_hasattr_walks_class_mro_for_inherited_attr() {
        // hasattr folding must resolve inherited attributes,
        // not just the immediate class dict. Build base with `m`, then
        // subclass with empty dict — `hasattr(Subclass, "m")` must be
        // folded to True.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let base = HostObject::new_class("pkg.Base", vec![]);
        base.class_set("m", ConstValue::Int(0));
        let sub = HostObject::new_class("pkg.Sub", vec![base]);
        let s_obj = bk.immutablevalue(&ConstValue::HostObject(sub)).unwrap();
        let s_attr = bk.immutablevalue(&ConstValue::byte_str("m")).unwrap();
        let out = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(true)) => {}
                other => panic!("expected Bool(true) for inherited attribute, got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn host_env_builtin_analyzer_keys_audit() {
        // Survey HOST_ENV-registered builtin callables and flag every
        // one whose qualname is NOT keyed in BUILTIN_ANALYZERS.
        // `bookkeeper.rs`'s `immutablevalue_hostobject` gates the SomeBuiltin
        // branch on
        // `is_registered(obj.qualname())`; a False return silently
        // routes the value to the extregistry / generic fallback path
        // — the analyser is effectively dead code.
        use crate::flowspace::model::HOST_ENV;

        let cases: &[(&str, &str)] = &[
            ("rpython.rlib.rarithmetic", "intmask"),
            ("rpython.rlib.rarithmetic", "longlongmask"),
            ("rpython.rlib.rarithmetic", "ulonglongmask"),
            ("rpython.rlib.rarithmetic", "longlonglongmask"),
            ("rpython.rlib.rarithmetic", "ulonglonglongmask"),
            // `rarithmetic.r_uint` is intentionally absent from
            // BUILTIN_ANALYZERS — `immutablevalue_hostobject` misses,
            // then falls through to its own extregistry path
            // which finds `ExtRegistryEntry::ForType` and returns the
            // SomeBuiltin whose `compute_result_annotation` produces
            // `SomeInteger(unsigned=True, knowntype=Ruint)`.  See
            // `flowspace/model.rs::populate_host_env::register_r_uint`.
            ("rpython.rlib.objectmodel", "keepalive_until_here"),
            ("rpython.rlib.objectmodel", "free_non_gc_object"),
            ("rpython.rtyper.lltypesystem.lltype", "cast_ptr_to_int"),
            ("rpython.rtyper.lltypesystem.lltype", "cast_int_to_ptr"),
            ("rpython.rtyper.lltypesystem.lltype", "direct_ptradd"),
        ];
        let mut missing: Vec<(String, String)> = Vec::new();
        for (module_path, attr) in cases {
            let m = HOST_ENV
                .import_module(module_path)
                .unwrap_or_else(|| panic!("module {module_path} not in HOST_ENV"));
            let Some(obj) = m.module_get(attr) else {
                continue;
            };
            let q = obj.qualname().to_string();
            if !super::is_registered(&q) {
                missing.push((format!("{module_path}.{attr}"), q));
            }
        }
        assert!(
            missing.is_empty(),
            "BUILTIN_ANALYZERS missing keys for qualnames bookkeeper.rs:1913 \
             reads — these analyzers are silently unreachable: {missing:#?}"
        );
    }

    #[test]
    fn lltype_cast_analyzer_lookup_matches_host_qualname() {
        // Verify that the analyser registered for
        // `lltype.cast_ptr_to_int` / `lltype.cast_int_to_ptr` is
        // discoverable under the exact qualname that
        // `immutablevalue_hostobject` assigns as `SomeBuiltin.analyser_name` (i.e. what
        // `call_builtin(analyser_name, ..)` will look up).
        use crate::flowspace::model::HOST_ENV;
        let lltype = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .expect("lltype module registered in HOST_ENV bootstrap");
        let cast_p2i = lltype
            .module_get("cast_ptr_to_int")
            .expect("F1 registered cast_ptr_to_int attr on lltype");
        let cast_i2p = lltype
            .module_get("cast_int_to_ptr")
            .expect("F1 registered cast_int_to_ptr attr on lltype");
        let direct_ptradd = lltype
            .module_get("direct_ptradd")
            .expect("F1 registered direct_ptradd attr on lltype");
        // What `immutablevalue_hostobject` will use as analyser_name.
        let q_p2i = cast_p2i.qualname();
        let q_i2p = cast_i2p.qualname();
        let q_add = direct_ptradd.qualname();
        assert!(
            super::is_registered(q_p2i),
            "F2 analyzer must be registered under the qualname \
             bookkeeper produces. Got qualname {q_p2i:?} which is \
             NOT in BUILTIN_ANALYZERS — call_builtin would fail."
        );
        assert!(
            super::is_registered(q_i2p),
            "F2 analyzer must be registered under the qualname \
             bookkeeper produces. Got qualname {q_i2p:?} which is \
             NOT in BUILTIN_ANALYZERS — call_builtin would fail."
        );
        assert!(
            super::is_registered(q_add),
            "F2 analyzer must be registered under the qualname \
             bookkeeper produces. Got qualname {q_add:?} which is \
             NOT in BUILTIN_ANALYZERS — call_builtin would fail."
        );
    }

    #[test]
    fn constpropagate_invokes_evaluator_on_all_constants() {
        let bk = bk();
        let s_a = bk.immutablevalue(&ConstValue::Int(7)).unwrap();
        let s_b = bk.immutablevalue(&ConstValue::Int(8)).unwrap();
        let s_result = SomeValue::Integer(super::super::model::SomeInteger::default());
        let out = constpropagate(&bk, &[s_a, s_b], s_result, |consts| match consts {
            [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(*a + *b)),
            _ => None,
        })
        .unwrap();
        if let SomeValue::Integer(i) = out
            && let Some(ConstValue::Int(n)) = i.base.const_box.as_ref().map(|c| &c.value)
        {
            assert_eq!(*n, 15);
            return;
        }
        panic!("expected constant SomeInteger(15)");
    }
}
