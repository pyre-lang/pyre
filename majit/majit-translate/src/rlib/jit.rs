//! Rust port of `rpython/rlib/jit.py:875-1024` — annotator / rtyper
//! support for the `jit_merge_point` / `can_enter_jit` /
//! `loop_header` markers.
//!
//! The runtime `JitDriver` class itself (rlib/jit.py) lives
//! on the pyre side as `PyPyJitDriver` (`pyre/pyre-jit/src/eval.rs`).
//! This module owns only the translator-side mirror metadata
//! (`JitDriverMeta`) plus the kwarg-validation / cache-union helper
//! consumed by `ExtRegistryEntry::EnterLeaveMarker` /
//! `ExtRegistryEntry::LoopHeader` (in
//! `translator/rtyper/extregistry.rs`).
//!
//! `ExtRegistryEntry::EnterLeaveMarker` and `ExtRegistryEntry::LoopHeader`
//! variants stay anchored on the [`crate::translator::rtyper::extregistry`]
//! enum so `lookup` / `makekey` / `is_registered` go through the same
//! registry as every other extregistry subclass; `compute_result_annotation`
//! itself, however, is upstream-defined in `rlib/jit.py` and lives here
//! to keep the file boundary aligned with upstream.
//!
//! The user-facing no-op hint functions from the earlier half of
//! `rlib/jit.py` (`promote`, `promote_string`, `we_are_jitted`,
//! `record_exact_class`, and related helpers) live in
//! `majit_metainterp::jit`. The proc macro and interpreter examples
//! recognize that public path today, while this crate owns the
//! translator-only extregistry half.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::bookkeeper::{Bookkeeper, EmulatedPbcCallKey};
use crate::annotator::model::{AnnotatorError, SomeValue, s_none};
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue,
    HostObject, Link, SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::lltypesystem::rstr::{
    jit_isconstant_intrinsic_funcptr, jit_isvirtual_intrinsic_funcptr, sub_helper_funcptr_constant,
};
use crate::translator::rtyper::rtyper::{
    LowLevelFunction, RPythonTyper, constant_with_lltype, helper_pygraph_from_graph,
    variable_with_lltype,
};

/// Upstream `jit.py:889` — `self.instance.__name__` on the bound
/// method. Discriminates which member of the
/// `_about_ = (jit_merge_point, can_enter_jit)` tuple triggered the
/// `ExtEnterLeaveMarker` entry; Rust passes the kind explicitly at
/// registration / lookup time because it has no bound-method object
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JitMarkerKind {
    /// `jit_merge_point` (jit.py).
    JitMergePoint,
    /// `can_enter_jit` (jit.py).
    CanEnterJit,
}

impl JitMarkerKind {
    /// Upstream `self.instance.__name__` — the bound-method name for
    /// `(jit_merge_point, can_enter_jit)` registered via `_about_` at
    /// `rlib/jit.py _make_extregistryentries`. Used by
    /// [`ext_enter_leave_marker_compute_result_annotation`] to gate the
    /// `annotate_hooks` branch (`rlib/jit.py`) and as the
    /// `methodname` carried on the [`crate::annotator::model::SomeBuiltin`]
    /// returned by the marker's `compute_annotation` (mirroring upstream
    /// `extregistry.py` base implementation).
    pub fn upstream_method_name(self) -> &'static str {
        match self {
            JitMarkerKind::JitMergePoint => "jit_merge_point",
            JitMarkerKind::CanEnterJit => "can_enter_jit",
        }
    }

    /// Diagnostic analyser-name for the `SomeBuiltin` carrier returned
    /// by `ExtEnterLeaveMarker.compute_annotation`. Not registered in
    /// `BUILTIN_ANALYZERS` — marker dispatch is special-cased in
    /// `SomeValue::call` via `extregistry::lookup(const_box)` rather
    /// than going through the analyser-name table (see
    /// `annotator/model.rs` SomeValue::Builtin call arm).
    pub fn analyser_name(self) -> &'static str {
        match self {
            JitMarkerKind::JitMergePoint => "rlib.jit.ExtEnterLeaveMarker.jit_merge_point",
            JitMarkerKind::CanEnterJit => "rlib.jit.ExtEnterLeaveMarker.can_enter_jit",
        }
    }
}

/// `rlib/jit.py` — `self.instance.__name__` for `loop_header`.
pub const LOOP_HEADER_METHOD_NAME: &str = "loop_header";

/// Diagnostic analyser-name for `ExtLoopHeader.compute_annotation`'s
/// `SomeBuiltin` carrier; same special-cased dispatch as the
/// enter/leave marker (see [`JitMarkerKind::analyser_name`]).
pub const LOOP_HEADER_ANALYSER_NAME: &str = "rlib.jit.ExtLoopHeader.loop_header";

/// Translator-side mirror of the runtime `PyPyJitDriver` (eval.rs)
/// fields that `ExtEnterLeaveMarker` / `ExtLoopHeader` consume. RPython
/// keeps a single `JitDriver` instance reachable from
/// `ExtRegistryEntry.instance.im_self` (rlib/jit.py) — the Rust
/// port stores the same fields here behind an `Arc` so multiple
/// registry variants for the same driver share the metadata identity.
///
/// The metadata is intentionally narrow: only fields that the
/// annotator / codewriter pipeline reads. Runtime hooks captured on
/// the pyre side (`PyPyJitDriver::get_*`) are mirrored as `HostObject`
/// callables once the per-hook wiring lands.
#[derive(Clone, Debug)]
pub struct JitDriverMeta {
    /// Identity of the driver instance — used as the
    /// `_jit_annotation_cache` key (rlib/jit.py:904
    /// `cache = self.bookkeeper._jit_annotation_cache[driver]`).
    pub id: HostObject,
    /// rlib/jit.py:619 / 651 — `name`.
    pub name: String,
    /// rlib/jit.py:649-650 — `greens`. Order is significant
    /// (rlib/jit.py INT/REF/FLOAT heuristic check).
    pub greens: Vec<String>,
    /// rlib/jit.py:652-662 — `reds`.
    pub reds: Vec<String>,
    /// rlib/jit.py:665-668 — `virtualizables`.
    pub virtualizables: Vec<String>,
    /// rlib/jit.py:653 / 661 — True iff `reds='auto'`.
    pub autoreds: bool,
    /// rlib/jit.py:655 / 662 — `len(reds)`; `None` when `autoreds`.
    pub numreds: Option<usize>,
    /// rlib/jit.py:692.
    pub is_recursive: bool,
    /// rlib/jit.py:693 — `vec` (vectorize default False).
    pub vec: bool,
    /// rlib/jit.py / 925-929 — `get_printable_location` callable
    /// for `annotate_hooks`. Wired by the pyre side once the host
    /// callable identity is available.
    pub get_printable_location: Option<HostObject>,
    /// rlib/jit.py:683 / 925-929 — `get_location` callable for
    /// `annotate_hooks`.
    pub get_location: Option<HostObject>,
}

/// rlib/jit.py — `ExtEnterLeaveMarker.compute_result_annotation`.
///
/// ```python
/// def compute_result_annotation(self, **kwds_s):
///     if self.instance.__name__ == 'jit_merge_point':
///         self.annotate_hooks(**kwds_s)
///     driver = self.instance.im_self
///     keys = kwds_s.keys()
///     keys.sort()
///     expected = ['s_' + name for name in driver.greens + driver.reds
///                             if '.' not in name]
///     expected.sort()
///     if keys != expected:
///         raise JitHintError(...)
///     try:
///         cache = self.bookkeeper._jit_annotation_cache[driver]
///     except (AttributeError, KeyError):
///         cache = {}
///         self.bookkeeper._jit_annotation_cache[driver] = cache
///     for key, s_value in kwds_s.items():
///         s_previous = cache.get(key, annmodel.s_ImpossibleValue)
///         s_value = annmodel.unionof(s_previous, s_value)
///         cache[key] = s_value
///     # add the attribute _dont_reach_me_in_del_ ...
///     return annmodel.s_None
/// ```
pub fn ext_enter_leave_marker_compute_result_annotation(
    bookkeeper: &Rc<Bookkeeper>,
    meta: &Arc<JitDriverMeta>,
    marker_kind: JitMarkerKind,
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // rlib/jit.py:889-890 — `if self.instance.__name__ ==
    // 'jit_merge_point': self.annotate_hooks(**kwds_s)`. Upstream
    // sequences annotate_hooks BEFORE the kwds validation below; mirror
    // the order so a missing-kwarg error from inside annotate_hooks
    // (KeyError on `kwds_s['s_' + name]`) surfaces with the same
    // ordering as upstream.
    if marker_kind == JitMarkerKind::JitMergePoint {
        annotate_hooks(bookkeeper, meta, kwds_s)?;
    }

    // rlib/jit.py:892-901 — validate keyword names against
    // `driver.greens + driver.reds`. Dotted greenfield names
    // (e.g. `'frame.code'`) are excluded from the kwds key set
    // because `specialize_call` projects them through field access
    // rather than through a kwarg.
    let mut expected: Vec<String> = meta
        .greens
        .iter()
        .chain(meta.reds.iter())
        .filter(|name| !name.contains('.'))
        .map(|name| format!("s_{name}"))
        .collect();
    expected.sort();

    let mut actual: Vec<String> = kwds_s.keys().cloned().collect();
    actual.sort();

    if actual != expected {
        return Err(AnnotatorError::new(format!(
            "JitDriver({}) marker expects keyword arguments {expected:?}, got {actual:?}",
            meta.name
        )));
    }

    // rlib/jit.py:903-914 — fold into _jit_annotation_cache[driver].
    bookkeeper.union_jit_annotation_kwds(&meta.id, kwds_s)?;

    // rlib/jit.py:916-921 — `try: graph = self.bookkeeper.position_key[0];
    //                            graph.func._dont_reach_me_in_del_ = True
    //                       except (TypeError, AttributeError): pass`.
    //
    // Rust mirrors the silent-swallow contract by walking the optional
    // chain and giving up at the first miss: position_key absent / weak
    // graph dead / `func` absent.
    if let Some(pk) = bookkeeper.current_position_key()
        && let Some(graph_rc) = pk.graph()
        && let Some(func) = graph_rc.borrow_mut().func.as_mut()
    {
        func._dont_reach_me_in_del_ = true;
    }

    // rlib/jit.py:923 — `return annmodel.s_None`.
    Ok(s_none())
}

/// rlib/jit.py — `ExtEnterLeaveMarker.annotate_hooks`.
///
/// ```python
/// def annotate_hooks(self, **kwds_s):
///     driver = self.instance.im_self
///     h = self.annotate_hook
///     h(driver.get_printable_location, driver.greens, **kwds_s)
///     h(driver.get_location, driver.greens, **kwds_s)
/// ```
fn annotate_hooks(
    bookkeeper: &Rc<Bookkeeper>,
    meta: &Arc<JitDriverMeta>,
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<(), AnnotatorError> {
    annotate_hook(
        bookkeeper,
        meta.get_printable_location.as_ref(),
        &meta.greens,
        kwds_s,
    )?;
    annotate_hook(bookkeeper, meta.get_location.as_ref(), &meta.greens, kwds_s)?;
    Ok(())
}

/// rlib/jit.py — `ExtEnterLeaveMarker.annotate_hook`.
///
/// ```python
/// def annotate_hook(self, func, variables, args_s=[], **kwds_s):
///     if func is None:
///         return
///     bk = self.bookkeeper
///     s_func = bk.immutablevalue(func)
///     uniquekey = 'jitdriver.%s' % func.__name__
///     args_s = args_s[:]
///     for name in variables:
///         if '.' not in name:
///             s_arg = kwds_s['s_' + name]
///         else:
///             objname, fieldname = name.split('.')
///             s_instance = kwds_s['s_' + objname]
///             classdesc = s_instance.classdef.classdesc
///             bk.record_getattr(classdesc, fieldname)
///             attrdef = s_instance.classdef.find_attribute(fieldname)
///             s_arg = attrdef.s_value
///             assert s_arg is not None
///         args_s.append(s_arg)
///     bk.emulate_pbc_call(uniquekey, s_func, args_s)
/// ```
///
/// The `args_s=[]` upstream default is dropped here because no
/// in-tree caller forwards a non-empty seed; if one ever lands the
/// signature gains an `args_s_seed: &[SomeValue]` parameter without
/// touching the variable-loop body.
///
/// The dotted-green branch is held back alongside
/// `ExtEnterLeaveMarker.specialize_call` (rlib/jit.py). Until
/// then a dotted variable surfaces as an `AnnotatorError` so the gap
/// fails closed instead of silently dropping the arg.
fn annotate_hook(
    bookkeeper: &Rc<Bookkeeper>,
    func: Option<&HostObject>,
    variables: &[String],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<(), AnnotatorError> {
    // rlib/jit.py — `if func is None: return`.
    let Some(func) = func else { return Ok(()) };

    // rlib/jit.py — `s_func = bk.immutablevalue(func)`. The hook
    // host is wrapped as a `ConstValue::HostObject` so the bookkeeper
    // takes the same branch as upstream's `immutablevalue(func)` for
    // a Python function (returns `SomePBC([funcdesc])`).
    let s_func = bookkeeper.immutablevalue(&ConstValue::HostObject(func.clone()))?;

    // rlib/jit.py — `uniquekey = 'jitdriver.%s' % func.__name__`.
    let unique_key = EmulatedPbcCallKey::Text(format!("jitdriver.{}", func.qualname()));

    // rlib/jit.py:937-949 — build `args_s` from greens.
    let mut args_s: Vec<SomeValue> = Vec::with_capacity(variables.len());
    for name in variables {
        if name.contains('.') {
            // rlib/jit.py:941-948 dotted-green field path. Requires
            // `s_instance.classdef.find_attribute(fieldname)` plus
            // `bk.record_getattr(classdesc, fieldname)` — both land
            // with `ExtEnterLeaveMarker.specialize_call`.
            return Err(AnnotatorError::new(format!(
                "JitDriver dotted greenfield {name:?} reaches annotate_hook before \
                 ExtEnterLeaveMarker.specialize_call port (rlib/jit.py:965-993)",
            )));
        }
        let key = format!("s_{name}");
        let s_arg = kwds_s.get(&key).ok_or_else(|| {
            // rlib/jit.py:940 raises KeyError on `kwds_s['s_'+name]`.
            AnnotatorError::new(format!(
                "JitDriver hook annotate: missing kwarg {key:?} for green {name:?}"
            ))
        })?;
        // upstream pushes the kwds entry to args_s as-is; if the entry
        // is `None` (annotation not yet bound at this fixpoint round),
        // `emulate_pbc_call(args_s)` would AttributeError on the first
        // `s.knowntype` access downstream.  `emulate_pbc_call` still
        // takes `&[SomeValue]` so we surface the unbound-cell case as
        // an `AnnotatorError` here; widening that ABI to
        // `&[Option<SomeValue>]` is a follow-up.
        let s_arg = s_arg.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "JitDriver hook annotate: kwarg {key:?} for green {name:?} \
                 is not yet annotated"
            ))
        })?;
        args_s.push(s_arg.clone());
    }

    // rlib/jit.py — `bk.emulate_pbc_call(uniquekey, s_func, args_s)`.
    // Upstream defaults: `replace=[]`, `callback=None`.
    bookkeeper.emulate_pbc_call(unique_key, &s_func, &args_s, &[], None)?;
    Ok(())
}

/// One conjunct of a `look_inside_iff` predicate over the function's own arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookInsidePred {
    /// `jit.isvirtual` on argument `n`.
    IsVirtual(usize),
    /// `jit.isconstant` on argument `n`.
    IsConstant(usize),
}

/// `rlib/jit.py look_inside_iff(predicate)(func)`: marks `orig` `unroll_safe` +
/// `specialize:call_location`, mints `<name>_trampoline` (`dont_look_inside`,
/// `specialize:call_location`) and `<name>_look_inside_iff`
/// (`if not we_are_jitted() or predicate(*args): return func(*args) else: return trampoline(*args)`).
/// Moves `func.oopspec` onto the trampoline (`trampoline.oopspec = func.oopspec;
/// del func.oopspec`). Returns the wrapper.
pub fn mint_look_inside_iff(
    rtyper: &RPythonTyper,
    name: &str,
    argnames: &[&str],
    args: Vec<LowLevelType>,
    result: LowLevelType,
    orig_builder: impl Fn(&str) -> Result<PyGraph, TyperError> + 'static,
    predicate: &[LookInsidePred],
) -> Result<LowLevelFunction, TyperError> {
    if argnames.len() != args.len() {
        return Err(TyperError::message(format!(
            "look_inside_iff {name}: argnames len {} != args len {}",
            argnames.len(),
            args.len()
        )));
    }

    let orig_name = name.to_string();
    let moved_oopspec = Rc::new(RefCell::new(None::<String>));
    let orig = {
        let moved_oopspec = moved_oopspec.clone();
        rtyper.lowlevel_helper_function_with_builder(
            orig_name.clone(),
            args.clone(),
            result.clone(),
            move |_r, _a, _res| {
                let mut pygraph = orig_builder(&orig_name)?;
                pygraph.func._jit_unroll_safe_ = true;
                pygraph.func.annspecialcase = Some("specialize:call_location".to_string());
                // `rlib/jit.py look_inside_iff`: `trampoline.oopspec = func.oopspec; del func.oopspec`.
                let oopspec = pygraph.func.oopspec.take();
                *moved_oopspec.borrow_mut() = oopspec;
                {
                    let mut inner = pygraph.graph.borrow_mut();
                    if let Some(func) = inner.func.as_mut() {
                        func._jit_unroll_safe_ = true;
                        func.annspecialcase = Some("specialize:call_location".to_string());
                        func.oopspec = None;
                    } else {
                        inner.func = Some(pygraph.func.clone());
                    }
                }
                Ok(pygraph)
            },
        )?
    };
    let trampoline_oopspec = moved_oopspec.borrow().clone();

    let trampoline_name = format!("{name}_trampoline");
    let orig_fn = sub_helper_funcptr_constant(rtyper, &orig)?;
    let trampoline = {
        let trampoline_name = trampoline_name.clone();
        let argnames_owned: Vec<String> = argnames.iter().map(|s| (*s).to_string()).collect();
        let args_owned = args.clone();
        let result_owned = result.clone();
        let orig_fn = orig_fn.clone();
        let trampoline_oopspec = trampoline_oopspec.clone();
        rtyper.lowlevel_helper_function_with_builder(
            trampoline_name.clone(),
            args.clone(),
            result.clone(),
            move |_r, _a, _res| {
                let argnames: Vec<&str> = argnames_owned.iter().map(|s| s.as_str()).collect();
                build_look_inside_iff_trampoline_graph(
                    &trampoline_name,
                    &argnames,
                    &args_owned,
                    &result_owned,
                    orig_fn.clone(),
                    trampoline_oopspec.clone(),
                )
            },
        )?
    };

    let mut predicate_intrinsics: Vec<(usize, Constant)> = Vec::new();
    for pred in predicate {
        match *pred {
            LookInsidePred::IsVirtual(i) => {
                let arg_ty = arg_lltype(name, &args, i)?;
                predicate_intrinsics
                    .push((i, jit_isvirtual_intrinsic_funcptr(rtyper, arg_ty.clone())?));
            }
            LookInsidePred::IsConstant(i) => {
                let arg_ty = arg_lltype(name, &args, i)?;
                predicate_intrinsics
                    .push((i, jit_isconstant_intrinsic_funcptr(rtyper, arg_ty.clone())?));
            }
        }
    }

    let wrapper_name = format!("{name}_look_inside_iff");
    let trampoline_fn = sub_helper_funcptr_constant(rtyper, &trampoline)?;
    let orig_fn = sub_helper_funcptr_constant(rtyper, &orig)?;
    let argnames_owned: Vec<String> = argnames.iter().map(|s| (*s).to_string()).collect();
    let args_owned = args.clone();
    let result_owned = result.clone();
    rtyper.lowlevel_helper_function_with_builder(
        wrapper_name.clone(),
        args,
        result,
        move |_r, _a, _res| {
            let argnames: Vec<&str> = argnames_owned.iter().map(|s| s.as_str()).collect();
            build_look_inside_iff_wrapper_graph(
                &wrapper_name,
                &argnames,
                &args_owned,
                &result_owned,
                orig_fn.clone(),
                trampoline_fn.clone(),
                &predicate_intrinsics,
            )
        },
    )
}

fn arg_lltype<'a>(
    name: &str,
    args: &'a [LowLevelType],
    index: usize,
) -> Result<&'a LowLevelType, TyperError> {
    args.get(index).ok_or_else(|| {
        TyperError::message(format!(
            "look_inside_iff {name}: predicate index {index} out of range ({} args)",
            args.len()
        ))
    })
}

fn look_inside_arg_vars(
    argnames: &[&str],
    args: &[LowLevelType],
) -> Vec<crate::flowspace::model::Variable> {
    argnames
        .iter()
        .zip(args.iter())
        .map(|(name, ty)| variable_with_lltype(name, ty.clone()))
        .collect()
}

fn hl_vars(vars: &[crate::flowspace::model::Variable]) -> Vec<Hlvalue> {
    vars.iter().cloned().map(Hlvalue::Variable).collect()
}

fn bool_const(value: bool) -> Hlvalue {
    constant_with_lltype(ConstValue::Bool(value), LowLevelType::Bool)
}

fn none_void() -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))
}

fn build_direct_call_return_block(
    argnames: &[&str],
    args: &[LowLevelType],
    result: &LowLevelType,
    callee: Constant,
    returnblock: BlockRef,
) -> BlockRef {
    let vars = look_inside_arg_vars(argnames, args);
    let block = Block::shared(hl_vars(&vars));
    let mut call_args = vec![Hlvalue::Constant(callee)];
    call_args.extend(hl_vars(&vars));
    if *result == LowLevelType::Void {
        block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            call_args,
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        block.closeblock(vec![
            Link::new(vec![none_void()], Some(returnblock), None).into_ref(),
        ]);
    } else {
        let res = variable_with_lltype("result", result.clone());
        block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            call_args,
            Hlvalue::Variable(res.clone()),
        ));
        block.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(res)], Some(returnblock), None).into_ref(),
        ]);
    }
    block
}

/// `look_inside_iff`'s `@dont_look_inside` trampoline: one `direct_call`
/// of the original (`rlib/jit.py` `look_inside_iff`). `oopspec` is the
/// spec moved off orig (`trampoline.oopspec = func.oopspec`).
fn build_look_inside_iff_trampoline_graph(
    name: &str,
    argnames: &[&str],
    args: &[LowLevelType],
    result: &LowLevelType,
    orig_fn: Constant,
    oopspec: Option<String>,
) -> Result<PyGraph, TyperError> {
    let vars = look_inside_arg_vars(argnames, args);
    let startblock = Block::shared(hl_vars(&vars));
    let return_var = variable_with_lltype("result", result.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let mut call_args = vec![Hlvalue::Constant(orig_fn)];
    call_args.extend(hl_vars(&vars));
    if *result == LowLevelType::Void {
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            call_args,
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        startblock.closeblock(vec![
            Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
    } else {
        let res = variable_with_lltype("result", result.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            call_args,
            Hlvalue::Variable(res.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(res)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    func._jit_look_inside_ = Some(false);
    func.annspecialcase = Some("specialize:call_location".to_string());
    func.oopspec = oopspec;
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        argnames.iter().map(|s| (*s).to_string()).collect(),
        func,
    ))
}

/// `rlib/jit.py look_inside_iff` wrapper:
///
/// ```python
/// def f(*args):
///     if not we_are_jitted() or predicate(*args):
///         return orig(*args)
///     else:
///         return trampoline(*args)
/// ```
///
/// `we_are_jitted` is the `SpecTag` exitswitch (true link first). Each
/// predicate conjunct is its own block: true continues, false goes to
/// the trampoline.
fn build_look_inside_iff_wrapper_graph(
    name: &str,
    argnames: &[&str],
    args: &[LowLevelType],
    result: &LowLevelType,
    orig_fn: Constant,
    trampoline_fn: Constant,
    predicate_intrinsics: &[(usize, Constant)],
) -> Result<PyGraph, TyperError> {
    let start_vars = look_inside_arg_vars(argnames, args);
    let startblock = Block::shared(hl_vars(&start_vars));
    let return_var = variable_with_lltype("result", result.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let block_orig =
        build_direct_call_return_block(argnames, args, result, orig_fn, graph.returnblock.clone());
    let block_tramp = build_direct_call_return_block(
        argnames,
        args,
        result,
        trampoline_fn,
        graph.returnblock.clone(),
    );

    let mut jitted_target = block_orig.clone();
    for (arg_index, intrinsic) in predicate_intrinsics.iter().rev() {
        let vars = look_inside_arg_vars(argnames, args);
        let block = Block::shared(hl_vars(&vars));
        let flag = variable_with_lltype("pred", LowLevelType::Bool);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(intrinsic.clone()),
                Hlvalue::Variable(vars[*arg_index].clone()),
            ],
            Hlvalue::Variable(flag.clone()),
        ));
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(flag));
        block.closeblock(vec![
            Link::new(
                hl_vars(&vars),
                Some(jitted_target.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                hl_vars(&vars),
                Some(block_tramp.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);
        jitted_target = block;
    }

    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            hl_vars(&start_vars),
            Some(jitted_target),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            hl_vars(&start_vars),
            Some(block_orig),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        argnames.iter().map(|s| (*s).to_string()).collect(),
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use std::rc::Rc;

    fn trivial_void_builder(name: &str, arg_ty: LowLevelType) -> Result<PyGraph, TyperError> {
        let x = variable_with_lltype("x", arg_ty);
        let startblock = Block::shared(vec![Hlvalue::Variable(x)]);
        let return_var = variable_with_lltype("result", LowLevelType::Void);
        let mut graph = FunctionGraph::with_return_var(
            name.to_string(),
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        startblock.closeblock(vec![
            Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let func = GraphFunc::new(
            name.to_string(),
            Constant::new(ConstValue::Dict(Default::default())),
        );
        graph.func = Some(func.clone());
        Ok(helper_pygraph_from_graph(
            graph,
            vec!["x".to_string()],
            func,
        ))
    }

    fn graph_named(ann: &RPythonAnnotator, name: &str) -> crate::flowspace::model::GraphRef {
        ann.translator
            .graphs
            .borrow()
            .iter()
            .find(|g| g.borrow().name == name)
            .unwrap_or_else(|| panic!("missing graph {name}"))
            .clone()
    }

    fn count_predicate_blocks(start: &BlockRef) -> usize {
        let true_link = start.borrow().exits[0]
            .borrow()
            .target
            .clone()
            .expect("true link");
        let mut block = true_link;
        let mut n = 0usize;
        loop {
            let exitswitch = block.borrow().exitswitch.clone();
            match exitswitch {
                Some(Hlvalue::Variable(_)) => {
                    n += 1;
                    let next = block.borrow().exits[0]
                        .borrow()
                        .target
                        .clone()
                        .expect("pred true target");
                    block = next;
                }
                _ => break,
            }
        }
        n
    }

    #[test]
    fn look_inside_iff_mints_orig_trampoline_and_wrapper() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");

        let wrapper = mint_look_inside_iff(
            &rtyper,
            "ll_dummy_look_inside",
            &["x"],
            vec![LowLevelType::Signed],
            LowLevelType::Void,
            |name| {
                let mut py = trivial_void_builder(name, LowLevelType::Signed)?;
                py.func.oopspec = Some("test.oopspec(x)".to_string());
                if let Some(func) = py.graph.borrow_mut().func.as_mut() {
                    func.oopspec = Some("test.oopspec(x)".to_string());
                }
                Ok(py)
            },
            &[LookInsidePred::IsVirtual(0), LookInsidePred::IsConstant(0)],
        )
        .expect("mint_look_inside_iff");

        assert_eq!(wrapper.name, "ll_dummy_look_inside_look_inside_iff");
        let orig = graph_named(&ann, "ll_dummy_look_inside");
        let trampoline = graph_named(&ann, "ll_dummy_look_inside_trampoline");
        let wrapper_graph = graph_named(&ann, "ll_dummy_look_inside_look_inside_iff");

        let orig_func = orig
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("orig GraphFunc");
        assert!(
            orig_func._jit_unroll_safe_,
            "look_inside_iff marks the original unroll_safe"
        );
        assert_eq!(
            orig_func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );

        let tramp_func = trampoline
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("trampoline GraphFunc");
        assert_eq!(tramp_func._jit_look_inside_, Some(false));
        assert_eq!(
            tramp_func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );
        assert_eq!(
            orig_func.oopspec, None,
            "look_inside_iff deletes orig oopspec (`rlib/jit.py look_inside_iff`)"
        );
        assert_eq!(
            tramp_func.oopspec.as_deref(),
            Some("test.oopspec(x)"),
            "look_inside_iff moves oopspec onto the trampoline"
        );

        let wrap_func = wrapper
            .graph
            .as_ref()
            .expect("wrapper PyGraph")
            .func
            .clone();
        assert_eq!(wrap_func.annspecialcase, None);
        assert_eq!(wrap_func.oopspec, None);
        assert_eq!(wrap_func.name, "ll_dummy_look_inside_look_inside_iff");

        let startblock = wrapper_graph.borrow().startblock.clone();
        let start = startblock.borrow();
        assert!(start.operations.is_empty());
        let Some(Hlvalue::Constant(c)) = &start.exitswitch else {
            panic!("exitswitch must be the we_are_jitted SpecTag constant");
        };
        assert_eq!(c.value, ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID));
        assert_eq!(start.exits.len(), 2);
        assert!(
            matches!(
                &start.exits[0].borrow().exitcase,
                Some(Hlvalue::Constant(c)) if c.value == ConstValue::Bool(true)
            ),
            "we_are_jitted true link first"
        );
        drop(start);
        assert_eq!(
            count_predicate_blocks(&startblock),
            2,
            "two conjuncts mint two predicate blocks"
        );
    }

    /// `rlib/jit.py isconstant` accepts any type; `look_inside_iff`
    /// keeps every `isconstant` conjunct (`rordereddict.py`
    /// `ll_dict_lookup` / `_ll_dict_setitem_lookup_done`:
    /// `isvirtual(d) and isconstant(key)`).
    #[test]
    fn look_inside_iff_keeps_isconstant_conjunct_for_ptr_args() {
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr, PtrTarget};
        use crate::translator::rtyper::lltypesystem::rstr::direct_call_funcobjs;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");

        let dict_ty = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Opaque(OpaqueType::gc("DICT")),
        }));
        let key_ty = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Opaque(OpaqueType::gc("KEY")),
        }));
        let orig_d = dict_ty.clone();
        let orig_key = key_ty.clone();
        let wrapper = mint_look_inside_iff(
            &rtyper,
            "ll_dummy_look_inside_ptr_key",
            &["d", "key"],
            vec![dict_ty.clone(), key_ty.clone()],
            LowLevelType::Void,
            move |name| {
                let d = variable_with_lltype("d", orig_d.clone());
                let key = variable_with_lltype("key", orig_key.clone());
                let startblock = Block::shared(vec![Hlvalue::Variable(d), Hlvalue::Variable(key)]);
                let return_var = variable_with_lltype("result", LowLevelType::Void);
                let mut graph = FunctionGraph::with_return_var(
                    name.to_string(),
                    startblock.clone(),
                    Hlvalue::Variable(return_var),
                );
                startblock.closeblock(vec![
                    Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
                ]);
                let func = GraphFunc::new(
                    name.to_string(),
                    Constant::new(ConstValue::Dict(Default::default())),
                );
                graph.func = Some(func.clone());
                Ok(helper_pygraph_from_graph(
                    graph,
                    vec!["d".to_string(), "key".to_string()],
                    func,
                ))
            },
            &[LookInsidePred::IsVirtual(0), LookInsidePred::IsConstant(1)],
        )
        .expect("mint_look_inside_iff");

        assert_eq!(wrapper.name, "ll_dummy_look_inside_ptr_key_look_inside_iff");
        let wrapper_graph = graph_named(&ann, "ll_dummy_look_inside_ptr_key_look_inside_iff");
        assert_eq!(
            count_predicate_blocks(&wrapper_graph.borrow().startblock),
            2,
            "isvirtual and isconstant both mint predicate blocks"
        );

        let funcs = direct_call_funcobjs(&wrapper_graph.borrow());
        let isvirtual: Vec<_> = funcs
            .iter()
            .filter(|f| f._name == "majit_rlib::jit::isvirtual")
            .collect();
        let isconstant: Vec<_> = funcs
            .iter()
            .filter(|f| f._name == "majit_rlib::jit::isconstant")
            .collect();
        assert_eq!(
            isvirtual.len(),
            1,
            "wrapper calls one isvirtual intrinsic, got {:?}",
            funcs.iter().map(|f| &f._name).collect::<Vec<_>>()
        );
        assert_eq!(
            isconstant.len(),
            1,
            "wrapper calls one isconstant intrinsic, got {:?}",
            funcs.iter().map(|f| &f._name).collect::<Vec<_>>()
        );
        assert_eq!(
            isvirtual[0].TYPE.args,
            vec![dict_ty.clone()],
            "isvirtual FuncType arg is the dict Ptr"
        );
        assert_eq!(
            isconstant[0].TYPE.args,
            vec![key_ty],
            "isconstant FuncType arg is the key Ptr"
        );
    }
}
