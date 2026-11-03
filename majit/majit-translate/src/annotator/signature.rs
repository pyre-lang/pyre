//! Signature / annotation-spec helpers.
//!
//! RPython upstream: `rpython/annotator/signature.py` (184 LOC).
//!
//! Ports `annotation`, `annotationoftype`, `_compute_annotation`,
//! `_validate_annotation_size`, `Sig`, `SignatureError`,
//! `finish_type`, `enforce_signature_args`, and
//! `enforce_signature_return`.
//!
//! ## TODO: AnnotationSpec enum
//!
//! Upstream `annotation(t, bookkeeper=None)` accepts a polymorphic
//! Python `t` value — a type object, list, dict, tuple, `SomeObject`
//! instance, `lltype.LowLevelType`, or extregistry entry. Rust has no
//! dynamic-type equivalent, so the input is modelled as the
//! [`AnnotationSpec`] enum. Each enum variant maps to one branch of
//! upstream's `isinstance(t, …)` dispatch.
//!
//! ## Dependency-blocked paths
//!
//! * `extregistry` value-level branch (`_compute_annotation`
//!   signature.py:25-26, 73-76) — wired in
//!   [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue`] via
//!   [`crate::translator::rtyper::extregistry::is_registered`]; per-entry
//!   coverage extends as registrations land.
//! * `extregistry` type-level branch (`annotationoftype`
//!   signature.py) — wired below in [`annotationoftype`] via
//!   [`crate::translator::rtyper::extregistry::is_registered_type`] /
//!   `lookup_type`.
//! * `lltype.LowLevelType` branch (`_compute_annotation`
//!   signature.py:59-60) — `lltype.py` is now ported under
//!   [`crate::translator::rtyper::lltypesystem::lltype`]; the
//!   signature.py:59-60 dispatch (build a `SomeLowLevelType` from a
//!   `LowLevelType` instance) still needs the matching
//!   `SomeLowLevelType` variant on [`super::model::SomeValue`] before
//!   wire-up.
//! * `_annotation_cache` memoization (signature.py, 30-38) —
//!   skipped; callers re-compute on every call. Performance-only;
//!   correctness matches upstream.

use std::rc::Rc;

use super::bookkeeper::Bookkeeper;
use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, SomeBool, SomeDict, SomeFloat, SomeInstance, SomeInteger, SomeList,
    SomeObjectTrait, SomeString, SomeTuple, SomeType, SomeUnicodeString, SomeValue, s_none, union,
};
use crate::flowspace::model::HostObject;

/// Polymorphic input to [`annotation`] / [`annotationoftype`].
///
/// Matches upstream `t` parameter shape (signature.py:30, :80). Each
/// variant maps to one branch of upstream's dispatch.
///
/// **Note on `Hash`**: upstream `_annotation_cache` keys on
/// `_annotation_key(t)` which always returns a hashable atom
/// (signature.py:15-28). The Rust enum carries an `Already(SomeValue)`
/// variant, and `SomeValue` is not `Hash`-able (it's a deep enum
/// containing `Rc<RefCell<ListItem>>` identity cells); we therefore
/// skip the cache — see module doc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnotationSpec {
    /// `bool` type object — upstream `t is bool`.
    Bool,
    /// `int` type object — upstream `t is int`.
    Int,
    /// `float` type object — upstream `t is float`.
    Float,
    /// `str` type object (or subclass per upstream `issubclass(t, str)`).
    Str,
    /// `unicode` type object — upstream `t is unicode`.
    Unicode,
    /// `types.NoneType` / `None` — upstream `t is types.NoneType`.
    NoneType,
    /// `type` type object — upstream `t is type`.
    Type,
    /// User class passed as a host class object. Upstream
    /// `elif bookkeeper and not hasattr(t, '_freeze_'): return
    /// SomeInstance(bookkeeper.getuniqueclassdef(t))`
    /// (signature.py:103-104). The qualified name is preserved by the
    /// carried [`HostObject`] (`.qualname()`).
    UserClass(HostObject),
    /// `[X]` — list-of-X annotation (signature.py).
    List(Box<AnnotationSpec>),
    /// `{K: V}` — dict-with-K-keys-and-V-values annotation
    /// (signature.py:67-70).
    Dict(Box<AnnotationSpec>, Box<AnnotationSpec>),
    /// `(X, Y, ...)` — tuple annotation (signature.py).
    Tuple(Vec<AnnotationSpec>),
    /// Already-computed `SomeValue` — upstream `isinstance(t,
    /// SomeObject)` short-circuit (signature.py:57-58).
    Already(Box<SomeValue>),
}

/// RPython `class SignatureError(AnnotatorError)` (signature.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureError(pub String);

impl SignatureError {
    pub fn new(msg: impl Into<String>) -> Self {
        SignatureError(msg.into())
    }
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SignatureError: {}", self.0)
    }
}

impl std::error::Error for SignatureError {}

impl From<SignatureError> for AnnotatorError {
    fn from(e: SignatureError) -> Self {
        AnnotatorError::new(e.0)
    }
}

/// RPython `_validate_annotation_size(t)` (signature.py).
///
/// Upstream rejects multi-element lists / dicts (which indicate an
/// ill-formed annotation spec); tuples are always accepted regardless
/// of length. The Rust [`AnnotationSpec`] already enforces the
/// single-element invariant structurally (`List(Box<..>)`,
/// `Dict(K, V)`), so this function is a no-op kept for structural
/// parity with the upstream call sites in `_compute_annotation`.
fn validate_annotation_size(_spec: &AnnotationSpec) -> Result<(), SignatureError> {
    Ok(())
}

/// RPython `annotation(t, bookkeeper=None)` (signature.py).
///
/// Upstream memoises lookups in `_annotation_cache` when
/// `bookkeeper is None`; the Rust port skips the cache (perf-only,
/// correctness unchanged). See module doc for rationale.
pub fn annotation(
    spec: &AnnotationSpec,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<SomeValue, SignatureError> {
    compute_annotation(spec, bookkeeper)
}

/// RPython `_compute_annotation(t, bookkeeper=None)` (signature.py).
fn compute_annotation(
    spec: &AnnotationSpec,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<SomeValue, SignatureError> {
    validate_annotation_size(spec)?;
    match spec {
        // upstream: `if isinstance(t, SomeObject): return t`.
        AnnotationSpec::Already(sv) => Ok(sv.as_ref().clone()),
        // upstream: `elif isinstance(t, list): return SomeList(
        //     ListDef(bookkeeper, annotation(t[0]), mutated=True, resized=True))`.
        AnnotationSpec::List(inner) => {
            let s_inner = annotation(inner, bookkeeper)?;
            Ok(SomeValue::List(SomeList::new(ListDef::new(
                bookkeeper.cloned(),
                s_inner,
                true,
                true,
            ))))
        }
        // upstream: `elif isinstance(t, tuple): return SomeTuple(tuple([annotation(i) ...]))`.
        AnnotationSpec::Tuple(items) => {
            let items_s = items
                .iter()
                .map(|i| annotation(i, bookkeeper))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(SomeValue::Tuple(SomeTuple::new(items_s)))
        }
        // upstream: `elif isinstance(t, dict): return SomeDict(
        //     DictDef(bookkeeper, annotation(t.keys()[0]), annotation(t.values()[0])))`.
        AnnotationSpec::Dict(k_spec, v_spec) => {
            let s_k = annotation(k_spec, bookkeeper)?;
            let s_v = annotation(v_spec, bookkeeper)?;
            Ok(SomeValue::Dict(SomeDict::new(DictDef::new(
                bookkeeper.cloned(),
                s_k,
                s_v,
                false,
                false,
                false,
            ))))
        }
        // upstream: `elif type(t) is types.NoneType: return s_None`.
        AnnotationSpec::NoneType => Ok(s_none()),
        // upstream: everything else delegates to annotationoftype.
        // extregistry / lltype branches remain rtyper-phase deps.
        _ => annotationoftype(spec, bookkeeper),
    }
}

/// RPython `annotationoftype(t, bookkeeper=False)` (signature.py).
///
/// Builds the most-precise [`SomeValue`] for a plain Python type
/// object. `bookkeeper` is `Some` when the caller has a live
/// bookkeeper — upstream guards `extregistry` / user-class paths
/// behind that truthiness.
pub fn annotationoftype(
    spec: &AnnotationSpec,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<SomeValue, SignatureError> {
    match spec {
        AnnotationSpec::Bool => Ok(SomeValue::Bool(SomeBool::new())),
        AnnotationSpec::Int => Ok(SomeValue::Integer(SomeInteger::default())),
        AnnotationSpec::Float => Ok(SomeValue::Float(SomeFloat::new())),
        AnnotationSpec::Str => Ok(SomeValue::String(SomeString::default())),
        AnnotationSpec::Unicode => Ok(SomeValue::UnicodeString(SomeUnicodeString::default())),
        AnnotationSpec::NoneType => Ok(s_none()),
        AnnotationSpec::Type => Ok(SomeValue::Type(SomeType::new())),
        AnnotationSpec::UserClass(cls) => {
            // upstream signature.py:98-100:
            //     elif bookkeeper and extregistry.is_registered_type(t):
            //         return (extregistry.lookup_type(t)
            //                 .compute_annotation_bk(bookkeeper))
            //
            // HostObject type entries are registered in extregistry's
            // EXT_REGISTRY_BY_TYPE analogue; misses fall through to
            // the `getuniqueclassdef` UserClass path below.
            if let Some(bk) = bookkeeper
                && crate::translator::rtyper::extregistry::is_registered_type(cls)
            {
                let entry = crate::translator::rtyper::extregistry::lookup_type(cls).expect(
                    "annotationoftype: extregistry.lookup_type must succeed after \
                     is_registered_type",
                );
                return entry.compute_annotation_bk(bk).map_err(|e| {
                    SignatureError::new(format!(
                        "Annotation of registered type {:?}: {}",
                        cls.qualname(),
                        e
                    ))
                });
            }
            // upstream signature.py:103-104:
            //     elif bookkeeper and not hasattr(t, '_freeze_'):
            //         return SomeInstance(bookkeeper.getuniqueclassdef(t))
            // The `hasattr(t, '_freeze_')` guard is the Frozen-PBC branch
            // (handled elsewhere via `getfrozen`); here we assume the
            // caller already dispatched frozen types through the
            // dedicated path.
            if let Some(bk) = bookkeeper {
                let classdef = bk.getuniqueclassdef(cls).map_err(|e| {
                    SignatureError::new(format!(
                        "Annotation of user class {:?}: {}",
                        cls.qualname(),
                        e
                    ))
                })?;
                Ok(SomeValue::Instance(SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                )))
            } else {
                // upstream signature.py:106 — `raise TypeError("Annotation
                // of type %r not supported" % (t,))`.
                Err(SignatureError(format!(
                    "Annotation of type {:?} not supported \
                     (no bookkeeper — signature.py:106)",
                    cls.qualname()
                )))
            }
        }
        // List / Dict / Tuple / Already fall through here when the
        // caller entered via `annotationoftype` directly rather than
        // through `annotation`. Upstream's assertion at
        // signature.py — `assert isinstance(t, (type, types.ClassType))`
        // — rules out containers; map to SignatureError.
        _ => Err(SignatureError(format!(
            "annotationoftype: spec {spec:?} is not a type object"
        ))),
    }
}

/// Upstream `Sig.argtypes[i]` element taxonomy (signature.py).
///
/// Each variant maps to one branch of upstream's dispatch loop:
///
/// | Upstream predicate                                    | Rust variant           |
/// |-------------------------------------------------------|------------------------|
/// | `isinstance(argtype, (FunctionType, MethodType))`     | `DynamicCallable`      |
/// | `argtype is lltype.Void`                              | `Void`                 |
/// | `argtype is None`                                     | `PassThrough`          |
/// | `argtype is NOT_CONSTANT`                             | `NotConstant`          |
/// | `else` (type object / `AnnotationSpec`)               | `Spec(AnnotationSpec)` |
///
/// Keeping the variants distinct preserves upstream's assertions
/// (e.g. `Void` requires the inputcell to be a constant `SomePBC` /
/// `SomeNone`) and the `NOT_CONSTANT → not_const` semantics.
#[derive(Clone)]
pub enum SigArgType {
    /// `types.FunctionType` / `types.MethodType` — upstream invokes
    /// `argtype = argtype(*inputcells)` (signature.py) to
    /// resolve the real argtype at call time. The Rust port models
    /// the callable as a Rust closure that receives the inputcells
    /// and returns the resolved [`SigArgType`].
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    DynamicCallable(Rc<dyn Fn(&[SomeValue]) -> Result<SigArgType, SignatureError>>),
    /// `lltype.Void` — upstream requires the inputcell to be a
    /// constant `SomePBC` / `SomeNone`; passes the inputcell through
    /// unchanged (signature.py:121-127).
    Void,
    /// `argtype is None` — "no change", pass the inputcell through
    /// unchanged (signature.py:128-129).
    PassThrough,
    /// `NOT_CONSTANT` sentinel — upstream drops the `const` tag on
    /// the inputcell via `not_const` (signature.py:130-132).
    NotConstant,
    /// Plain type object / `AnnotationSpec` — upstream's `else`
    /// branch dispatches to `annotation(argtype, bookkeeper=...)`
    /// (signature.py:134).
    Spec(AnnotationSpec),
}

impl std::fmt::Debug for SigArgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SigArgType::DynamicCallable(_) => f.write_str("DynamicCallable(<closure>)"),
            SigArgType::Void => f.write_str("Void"),
            SigArgType::PassThrough => f.write_str("PassThrough"),
            SigArgType::NotConstant => f.write_str("NotConstant"),
            SigArgType::Spec(s) => write!(f, "Spec({s:?})"),
        }
    }
}

impl PartialEq for SigArgType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SigArgType::Void, SigArgType::Void) => true,
            (SigArgType::PassThrough, SigArgType::PassThrough) => true,
            (SigArgType::NotConstant, SigArgType::NotConstant) => true,
            (SigArgType::Spec(a), SigArgType::Spec(b)) => a == b,
            // `DynamicCallable` closures are opaque — match upstream
            // `types.FunctionType` identity by Rc pointer equality.
            (SigArgType::DynamicCallable(a), SigArgType::DynamicCallable(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for SigArgType {}

impl From<AnnotationSpec> for SigArgType {
    fn from(spec: AnnotationSpec) -> Self {
        SigArgType::Spec(spec)
    }
}

/// RPython `class Sig` (signature.py).
///
/// Carries the list of argument types declared via `@signature(...)`.
/// [`Self::call`] walks `inputcells` against `argtypes`, branching per
/// [`SigArgType`] variant (signature.py:118-134).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sig {
    pub argtypes: Vec<SigArgType>,
}

impl Sig {
    /// RPython `Sig.__init__(*argtypes)` (signature.py:110-111).
    pub fn new(argtypes: Vec<SigArgType>) -> Self {
        Sig { argtypes }
    }

    /// Convenience constructor for the common case where every
    /// argtype is a plain [`AnnotationSpec`]. Equivalent to
    /// `Sig::new(specs.into_iter().map(SigArgType::Spec).collect())`.
    pub fn from_specs(specs: Vec<AnnotationSpec>) -> Self {
        Sig {
            argtypes: specs.into_iter().map(SigArgType::Spec).collect(),
        }
    }

    /// RPython `Sig.__call__(self, funcdesc, inputcells)`
    /// (signature.py:113-147).
    ///
    /// ```python
    /// def __call__(self, funcdesc, inputcells):
    ///     args_s = []
    ///     for i, argtype in enumerate(self.argtypes):
    ///         if isinstance(argtype, (types.FunctionType, types.MethodType)):
    ///             argtype = argtype(*inputcells)
    ///         if argtype is lltype.Void: ...
    ///         elif argtype is None:
    ///             args_s.append(inputcells[i])
    ///         elif argtype is NOT_CONSTANT:
    ///             args_s.append(not_const(inputcells[i]))
    ///         else:
    ///             args_s.append(annotation(argtype, bookkeeper=funcdesc.bookkeeper))
    ///     if len(inputcells) != len(args_s):
    ///         raise SignatureError(...)
    ///     for i, (s_arg, s_input) in enumerate(zip(args_s, inputcells)):
    ///         s_input = unionof(s_input, s_arg)
    ///         if not s_arg.contains(s_input):
    ///             raise SignatureError(...)
    ///     inputcells[:] = args_s
    /// ```
    pub fn call(
        &self,
        funcname: &str,
        bookkeeper: &Rc<Bookkeeper>,
        inputcells: &mut [SomeValue],
    ) -> Result<(), SignatureError> {
        // upstream: `args_s = []; for i, argtype in enumerate(self.argtypes):`.
        let mut args_s: Vec<SomeValue> = Vec::with_capacity(self.argtypes.len());
        for (i, argtype_slot) in self.argtypes.iter().enumerate() {
            // upstream signature.py:119-120 — a callable argtype
            // resolves to the real argtype by invoking `argtype(*inputcells)`.
            // Resolve the callable once so the subsequent branches can
            // treat the result uniformly.
            let resolved: SigArgType = match argtype_slot {
                SigArgType::DynamicCallable(f) => f(inputcells)?,
                other => other.clone(),
            };
            match resolved {
                // upstream signature.py — `lltype.Void` branch:
                // assert the inputcell is a constant SomePBC/SomeNone,
                // pass it through unchanged.
                SigArgType::Void => {
                    let Some(s_input) = inputcells.get(i) else {
                        return Err(SignatureError::new(format!(
                            "{funcname}: Void argtype at index {i} has no inputcell",
                        )));
                    };
                    let is_pbc_or_none = matches!(s_input, SomeValue::PBC(_) | SomeValue::None_(_));
                    if !(is_pbc_or_none && s_input.is_constant()) {
                        return Err(SignatureError::new(format!(
                            "{funcname}: Void argtype at index {i} expects a constant SomePBC/SomeNone, got {s_input:?}",
                        )));
                    }
                    args_s.push(s_input.clone());
                }
                // upstream signature.py — `argtype is None`:
                // pass the inputcell through unchanged.
                SigArgType::PassThrough => {
                    args_s.push(inputcells.get(i).cloned().unwrap_or(SomeValue::Impossible));
                }
                // upstream signature.py — `argtype is NOT_CONSTANT`:
                // drop the constant tag via `not_const`.
                SigArgType::NotConstant => {
                    let s_input = inputcells.get(i).cloned().unwrap_or(SomeValue::Impossible);
                    args_s.push(super::model::not_const(&s_input));
                }
                // upstream signature.py:134 — `else: args_s.append(
                // annotation(argtype, bookkeeper=funcdesc.bookkeeper))`.
                SigArgType::Spec(spec) => {
                    args_s.push(annotation(&spec, Some(bookkeeper))?);
                }
                // `DynamicCallable` was resolved above; if `f` returns
                // another `DynamicCallable` we reject to match upstream's
                // `argtype(*inputcells)` returning a callable being the
                // caller's programming error.
                SigArgType::DynamicCallable(_) => {
                    return Err(SignatureError::new(format!(
                        "{funcname}: DynamicCallable at index {i} returned another DynamicCallable",
                    )));
                }
            }
        }
        // upstream: `if len(inputcells) != len(args_s): raise SignatureError`.
        if inputcells.len() != args_s.len() {
            return Err(SignatureError::new(format!(
                "{}: expected {} args, got {}",
                funcname,
                args_s.len(),
                inputcells.len()
            )));
        }
        // upstream: `for i, (s_arg, s_input) in enumerate(zip(args_s, inputcells)):
        //             s_input = unionof(s_input, s_arg);
        //             if not s_arg.contains(s_input): raise SignatureError`.
        for (i, (s_arg, s_input)) in args_s.iter().zip(inputcells.iter()).enumerate() {
            let s_merged =
                super::model::union(s_input, s_arg).map_err(|e| SignatureError::new(e.msg))?;
            if !s_arg.contains(&s_merged) {
                return Err(SignatureError::new(format!(
                    "{} argument {}:\nexpected {:?},\n     got {:?}",
                    funcname,
                    i + 1,
                    s_arg,
                    s_merged
                )));
            }
        }
        // upstream: `inputcells[:] = args_s`.
        for (slot, new_s) in inputcells.iter_mut().zip(args_s) {
            *slot = new_s;
        }
        Ok(())
    }
}

/// Marker types from `rpython/rlib/types.py` referenced by
/// [`finish_type`]. Upstream carries `SelfTypeMarker` and
/// `AnyTypeMarker` classes; Rust collapses them into a single enum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeMarker {
    /// RPython `class SelfTypeMarker` — declared via
    /// `annotation.types.self()`. Upstream raises SignatureError
    /// during `finish_type` unless the enclosing class used
    /// `@rlib.signature.finishsigs()`.
    SelfType,
    /// RPython `class AnyTypeMarker` — "no specific type required".
    /// `finish_type` returns None for this marker.
    AnyType,
}

/// One entry in `enforce_signature_args.paramtypes` — the three
/// shapes upstream `finish_type` dispatches on (signature.py).
pub enum ParamType {
    /// Already a `SomeObject` instance — upstream `isinstance(paramtype,
    /// SomeObject)` branch.
    Annotation(Box<SomeValue>),
    /// `SelfTypeMarker` / `AnyTypeMarker`.
    Marker(TypeMarker),
    /// A callable that takes the bookkeeper and returns a SomeValue.
    /// Upstream: `return paramtype(bookkeeper)` — the callable
    /// pattern used by `rlib/types.py` type factories.
    Builder(fn(&Rc<Bookkeeper>) -> Result<SomeValue, SignatureError>),
}

impl std::fmt::Debug for ParamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamType::Annotation(v) => write!(f, "ParamType::Annotation({v:?})"),
            ParamType::Marker(m) => write!(f, "ParamType::Marker({m:?})"),
            ParamType::Builder(_) => write!(f, "ParamType::Builder(<fn>)"),
        }
    }
}

/// RPython `finish_type(paramtype, bookkeeper, func)` (signature.py).
///
/// Returns `Some(s)` when the paramtype resolves to a concrete
/// annotation, `None` when it is an "any" marker (per upstream
/// `return None` for `AnyTypeMarker`).
pub fn finish_type(
    paramtype: &ParamType,
    bookkeeper: &Rc<Bookkeeper>,
    func_name: &str,
) -> Result<Option<SomeValue>, SignatureError> {
    match paramtype {
        ParamType::Annotation(sv) => Ok(Some(sv.as_ref().clone())),
        ParamType::Marker(TypeMarker::SelfType) => Err(SignatureError(format!(
            "{func_name:?} argument declared as annotation.types.self(); class needs decorator \
             rlib.signature.finishsigs()"
        ))),
        ParamType::Marker(TypeMarker::AnyType) => Ok(None),
        ParamType::Builder(build) => Ok(Some(build(bookkeeper)?)),
    }
}

/// RPython `enforce_signature_args(funcdesc, paramtypes, actualtypes)`
/// (signature.py:163-176).
///
/// Walks `paramtypes` and `actualtypes` in lockstep: for each declared
/// parameter, verify the inferred annotation is contained in the
/// declared one, then overwrite the actual with the declared.
/// `AnyType` markers are skipped. Upstream reads `funcdesc.bookkeeper`
/// / `funcdesc.pyobj` to format the error message; the Rust port
/// takes the bookkeeper + a display name directly so it can run
/// without a `funcdesc` port.
pub fn enforce_signature_args(
    func_name: &str,
    bookkeeper: &Rc<Bookkeeper>,
    paramtypes: &[ParamType],
    actualtypes: &mut [SomeValue],
) -> Result<(), SignatureError> {
    assert_eq!(paramtypes.len(), actualtypes.len());
    let params_s = paramtypes
        .iter()
        .map(|pt| finish_type(pt, bookkeeper, func_name))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, (s_param, s_actual)) in params_s.iter().zip(actualtypes.iter()).enumerate() {
        match s_param {
            None => continue,
            Some(s_param) => {
                if !s_param.contains(s_actual) {
                    return Err(SignatureError(format!(
                        "{func_name:?} argument {pos}:\nexpected {expected:?},\n     got {got:?}",
                        pos = i + 1,
                        expected = s_param,
                        got = s_actual
                    )));
                }
            }
        }
    }
    for (i, s_param) in params_s.iter().enumerate() {
        if let Some(sv) = s_param {
            actualtypes[i] = sv.clone();
        }
    }
    Ok(())
}

/// RPython `enforce_signature_return(funcdesc, sigtype, inferredtype)`
/// (signature.py:178-184).
pub fn enforce_signature_return(
    func_name: &str,
    bookkeeper: &Rc<Bookkeeper>,
    sigtype: &ParamType,
    inferredtype: &SomeValue,
) -> Result<Option<SomeValue>, SignatureError> {
    let s_sigret = finish_type(sigtype, bookkeeper, func_name)?;
    if let Some(ref s_sig) = s_sigret
        && !s_sig.contains(inferredtype)
    {
        return Err(SignatureError(format!(
            "{func_name:?} return value:\nexpected {expected:?},\n     got {got:?}",
            expected = s_sig,
            got = inferredtype
        )));
    }
    Ok(s_sigret)
}

// Keep the unused `union` import doc-visible so future callsites that
// land with the Sig.__call__ port can pick it up without re-adding
// the use statement.
fn _union_hold(a: &SomeValue, b: &SomeValue) -> Result<SomeValue, super::model::UnionError> {
    union(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeInteger, SomeValue};

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn annotation_of_int_is_someinteger() {
        let s = annotation(&AnnotationSpec::Int, None).unwrap();
        assert!(matches!(s, SomeValue::Integer(_)));
    }

    #[test]
    fn annotation_of_bool_is_somebool() {
        let s = annotation(&AnnotationSpec::Bool, None).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
    }

    #[test]
    fn annotation_of_float_str_unicode() {
        assert!(matches!(
            annotation(&AnnotationSpec::Float, None).unwrap(),
            SomeValue::Float(_)
        ));
        assert!(matches!(
            annotation(&AnnotationSpec::Str, None).unwrap(),
            SomeValue::String(_)
        ));
        assert!(matches!(
            annotation(&AnnotationSpec::Unicode, None).unwrap(),
            SomeValue::UnicodeString(_)
        ));
    }

    #[test]
    fn annotation_of_none_is_somenone() {
        let s = annotation(&AnnotationSpec::NoneType, None).unwrap();
        assert!(matches!(s, SomeValue::None_(_)));
    }

    #[test]
    fn annotation_of_type_is_sometype() {
        let s = annotation(&AnnotationSpec::Type, None).unwrap();
        assert!(matches!(s, SomeValue::Type(_)));
    }

    #[test]
    fn annotation_of_list_wraps_element() {
        // [int] → SomeList(ListDef(_, SomeInteger, mutated=True, resized=True))
        let s = annotation(
            &AnnotationSpec::List(Box::new(AnnotationSpec::Int)),
            Some(&bk()),
        )
        .unwrap();
        match s {
            SomeValue::List(l) => {
                assert!(matches!(l.listdef.s_value(), SomeValue::Integer(_)));
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn annotation_of_tuple_walks_items() {
        let s = annotation(
            &AnnotationSpec::Tuple(vec![AnnotationSpec::Int, AnnotationSpec::Str]),
            None,
        )
        .unwrap();
        match s {
            SomeValue::Tuple(t) => {
                assert_eq!(t.items.len(), 2);
                assert!(matches!(t.items[0], SomeValue::Integer(_)));
                assert!(matches!(t.items[1], SomeValue::String(_)));
            }
            other => panic!("expected SomeTuple, got {other:?}"),
        }
    }

    #[test]
    fn annotation_of_dict_walks_key_and_value() {
        let s = annotation(
            &AnnotationSpec::Dict(Box::new(AnnotationSpec::Str), Box::new(AnnotationSpec::Int)),
            Some(&bk()),
        )
        .unwrap();
        match s {
            SomeValue::Dict(d) => {
                assert!(matches!(d.dictdef.s_key(), SomeValue::String(_)));
                assert!(matches!(d.dictdef.s_value(), SomeValue::Integer(_)));
            }
            other => panic!("expected SomeDict, got {other:?}"),
        }
    }

    #[test]
    fn annotation_passes_through_already_somevalue() {
        // upstream: `if isinstance(t, SomeObject): return t`.
        let existing = SomeValue::Integer(SomeInteger::new(true, false));
        let s = annotation(&AnnotationSpec::Already(Box::new(existing.clone())), None).unwrap();
        assert_eq!(s, existing);
    }

    #[test]
    fn annotationoftype_rejects_container_spec() {
        // upstream `annotationoftype` asserts `isinstance(t, type)`.
        let err = annotationoftype(&AnnotationSpec::Tuple(vec![AnnotationSpec::Int]), None)
            .expect_err("container spec must not resolve through annotationoftype");
        assert!(err.0.contains("not a type object"));
    }

    #[test]
    fn user_class_without_bookkeeper_errors() {
        let cls = HostObject::new_class("Foo", vec![]);
        let err = annotationoftype(&AnnotationSpec::UserClass(cls), None)
            .expect_err("user class without bookkeeper must error");
        assert!(err.0.contains("not supported"));
    }

    #[test]
    fn user_class_with_bookkeeper_returns_someinstance() {
        // upstream signature.py:103-104 — `SomeInstance(
        // bookkeeper.getuniqueclassdef(t))`.
        let cls = HostObject::new_class("Foo", vec![]);
        let bk = bk();
        let s = annotation(&AnnotationSpec::UserClass(cls), Some(&bk))
            .expect("user class annotation must resolve");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_some(),
                    "SomeInstance must carry a classdef"
                );
                assert!(!inst.can_be_none);
            }
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn annotationoftype_uses_registered_extregistry_type_before_classdef() {
        let cls = HostObject::new_class("pkg.signature.RegisteredType", vec![]);
        crate::translator::rtyper::extregistry::register_host_type(
            cls.clone(),
            crate::translator::rtyper::extregistry::ExtRegistryEntry::HostTypeAnnotation {
                type_obj: cls.clone(),
                instance: None,
                annotation: crate::translator::rtyper::extregistry::RegisteredAnnotation::Int,
            },
        )
        .expect("register extregistry type");
        let bk = bk();
        let s = annotationoftype(&AnnotationSpec::UserClass(cls), Some(&bk))
            .expect("registered type should resolve via extregistry");
        assert!(matches!(s, SomeValue::Integer(_)));
    }

    #[test]
    fn finish_type_handles_annotation_marker_builder() {
        let bk = bk();

        // ParamType::Annotation
        let sv = SomeValue::Integer(SomeInteger::default());
        let out = finish_type(&ParamType::Annotation(Box::new(sv.clone())), &bk, "f").unwrap();
        assert_eq!(out, Some(sv));

        // ParamType::Marker(AnyType)
        let out = finish_type(&ParamType::Marker(TypeMarker::AnyType), &bk, "f").unwrap();
        assert_eq!(out, None);

        // ParamType::Marker(SelfType) — error
        let err = finish_type(&ParamType::Marker(TypeMarker::SelfType), &bk, "f").unwrap_err();
        assert!(err.0.contains("annotation.types.self()"));

        // ParamType::Builder
        fn build_int(_: &Rc<Bookkeeper>) -> Result<SomeValue, SignatureError> {
            Ok(SomeValue::Integer(SomeInteger::default()))
        }
        let out = finish_type(&ParamType::Builder(build_int), &bk, "f").unwrap();
        assert!(matches!(out, Some(SomeValue::Integer(_))));
    }

    #[test]
    fn enforce_signature_args_rejects_mismatched_param() {
        // Declared: Integer. Actual: String. → SignatureError.
        let bk = bk();
        let paramtypes = vec![ParamType::Annotation(Box::new(SomeValue::Integer(
            SomeInteger::default(),
        )))];
        let mut actualtypes = vec![SomeValue::String(super::super::model::SomeString::default())];
        let err = enforce_signature_args("f", &bk, &paramtypes, &mut actualtypes).unwrap_err();
        assert!(err.0.contains("argument 1"));
    }

    #[test]
    fn enforce_signature_args_overwrites_actual_with_declared() {
        let bk = bk();
        // Declared: nonneg Integer. Actual: signed Integer (contained
        // in nonneg? depends; upstream overwrites with declared).
        let nonneg = SomeInteger::new(true, false);
        let signed = SomeInteger::new(false, false);
        // To avoid SomeInteger containment detail dependence, use a
        // container that unambiguously contains the actual: unspec
        // Integer contains nonneg Integer.
        let paramtypes = vec![ParamType::Annotation(Box::new(SomeValue::Integer(
            SomeInteger::default(),
        )))];
        let mut actualtypes = vec![SomeValue::Integer(nonneg.clone())];
        let _ = signed;
        enforce_signature_args("f", &bk, &paramtypes, &mut actualtypes).unwrap();
        // actualtypes[0] is overwritten with the declared annotation.
        assert_eq!(actualtypes[0], SomeValue::Integer(SomeInteger::default()));
    }

    #[test]
    fn enforce_signature_args_skips_any_marker() {
        let bk = bk();
        let paramtypes = vec![ParamType::Marker(TypeMarker::AnyType)];
        let mut actualtypes = vec![SomeValue::String(super::super::model::SomeString::default())];
        // AnyType must NOT overwrite the actual.
        enforce_signature_args("f", &bk, &paramtypes, &mut actualtypes).unwrap();
        assert!(matches!(actualtypes[0], SomeValue::String(_)));
    }

    #[test]
    fn enforce_signature_return_rejects_mismatched() {
        let bk = bk();
        let sigtype = ParamType::Annotation(Box::new(SomeValue::Integer(SomeInteger::default())));
        let inferred = SomeValue::String(super::super::model::SomeString::default());
        let err = enforce_signature_return("f", &bk, &sigtype, &inferred).unwrap_err();
        assert!(err.0.contains("return value"));
    }

    #[test]
    fn enforce_signature_return_passes_through_any() {
        let bk = bk();
        let sigtype = ParamType::Marker(TypeMarker::AnyType);
        let inferred = SomeValue::String(super::super::model::SomeString::default());
        let out = enforce_signature_return("f", &bk, &sigtype, &inferred).unwrap();
        assert_eq!(out, None);
    }

    #[test]
    fn sig_call_rewrites_inputcells_to_declared_types() {
        // upstream signature.py — Sig.__call__ rebuilds inputcells
        // from annotation(argtype, ...) when the arg contains the input.
        let bk = bk();
        let sig = Sig::from_specs(vec![AnnotationSpec::Int, AnnotationSpec::Bool]);
        let mut inputs = vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Bool(super::super::model::SomeBool::new()),
        ];
        sig.call("f", &bk, &mut inputs)
            .expect("sig.call must succeed");
        // args_s is built from annotation; inputcells are overwritten.
        assert!(matches!(inputs[0], SomeValue::Integer(_)));
        assert!(matches!(inputs[1], SomeValue::Bool(_)));
    }

    #[test]
    fn sig_call_rejects_wrong_length() {
        let bk = bk();
        let sig = Sig::from_specs(vec![AnnotationSpec::Int, AnnotationSpec::Int]);
        let mut inputs = vec![SomeValue::Integer(SomeInteger::new(true, false))];
        let err = sig.call("f", &bk, &mut inputs).unwrap_err();
        assert!(err.0.contains("expected 2 args"));
    }

    #[test]
    fn sig_call_passthrough_leaves_inputcell_untouched() {
        // upstream signature.py — `argtype is None`: inputcell
        // passes through unchanged. The Rust port models this as
        // `SigArgType::PassThrough` (distinct from Spec(Already(X))).
        let bk = bk();
        let sig = Sig::new(vec![SigArgType::PassThrough]);
        let original = SomeValue::String(super::super::model::SomeString::default());
        let mut inputs = vec![original.clone()];
        sig.call("f", &bk, &mut inputs).expect("pass-through");
        assert_eq!(inputs[0], original);
    }

    #[test]
    fn sig_call_not_constant_drops_const_box() {
        // upstream signature.py — `argtype is NOT_CONSTANT`:
        // route the inputcell through `not_const`. For a constant
        // SomeInteger, the const_box should be cleared.
        let bk = bk();
        let mut s_one = SomeInteger::default();
        s_one.base.const_box = Some(super::super::super::flowspace::model::Constant::new(
            super::super::super::flowspace::model::ConstValue::Int(1),
        ));
        let sig = Sig::new(vec![SigArgType::NotConstant]);
        let mut inputs = vec![SomeValue::Integer(s_one)];
        sig.call("f", &bk, &mut inputs).expect("not-constant");
        // Post-call: inputcell is still SomeInteger but non-constant.
        match &inputs[0] {
            SomeValue::Integer(v) => assert!(v.base.const_box.is_none()),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn sig_call_dynamic_callable_resolves_argtype() {
        // upstream signature.py — `argtype(*inputcells)` is
        // invoked when the argtype is a FunctionType/MethodType. The
        // resolved argtype then drives the normal Spec/Void/… dispatch.
        let bk = bk();
        let resolve: Rc<dyn Fn(&[SomeValue]) -> Result<SigArgType, SignatureError>> =
            Rc::new(|_inputcells| Ok(SigArgType::Spec(AnnotationSpec::Int)));
        let sig = Sig::new(vec![SigArgType::DynamicCallable(resolve)]);
        let mut inputs = vec![SomeValue::Integer(SomeInteger::new(true, false))];
        sig.call("f", &bk, &mut inputs).expect("dynamic callable");
        assert!(matches!(inputs[0], SomeValue::Integer(_)));
    }

    #[test]
    fn sig_call_rejects_type_mismatch() {
        // upstream: `if not s_arg.contains(s_input): raise SignatureError`.
        // If unionof(s_input, s_arg) fails structurally, the union error
        // is surfaced via SignatureError (instead of the downstream
        // `.contains()` check) — matches upstream because Python's
        // `unionof(Int, Str)` raises UnionError which propagates up.
        let bk = bk();
        let sig = Sig::from_specs(vec![AnnotationSpec::Int]);
        let mut inputs = vec![SomeValue::String(super::super::model::SomeString::default())];
        let err = sig.call("f", &bk, &mut inputs).unwrap_err();
        // Either the upstream 'argument' message or the union error
        // surfaces — both signal rejection.
        assert!(
            err.0.contains("argument 1") || err.0.contains("pair"),
            "unexpected error message: {}",
            err.0
        );
    }
}
