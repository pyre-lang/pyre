//! RPython `rpython/rtyper/annlowlevel.py` — low-level helper annotation
//! + mix-level helper injection bridge.
//!
//! Upstream docstring:
//!
//! > The code needed to flow and annotate low-level helpers -- the
//! > ll_*() functions
//!
//! This module is the structural bridge between
//! [`crate::annotator::annrpython::RPythonAnnotator`] and
//! [`crate::translator::rtyper::rtyper::RPythonTyper`] for any code
//! that needs to inject a freshly annotated + rtyped helper graph into
//! an in-flight translation:
//!
//! * `ExceptionData.make_helpers` (exceptiondata.py) →
//!   `annmixlevel.getgraph(ll_exception_match, …)` etc.
//! * `normalizecalls.create_instantiate_functions` (normalizecalls.py:
//!   :~300-370) — generates `my_instantiate_graph` on every `ClassDef`
//!   with an `__init__`.
//! * `RPythonTyper.getannmixlevel` (rtyper.py) and the
//!   `self.annmixlevel.finish()` fixpoint step inside
//!   `specialize_more_blocks` (rtyper.py).
//! * `rpython/translator/backendopt/all.py::backend_optimizations`
//!   `secondary=True` path (via `backend_optimize`).
//!
//! ## Status of this port
//!
//! **Skeleton partially filled.** Public functions and methods whose
//! dependencies have landed are ported directly against their
//! upstream counterparts; surfaces that still lack a required Rust
//! carrier keep returning structured pending errors naming the
//! missing primitive.
//!
//! The items whose callable bodies are still scheduled as follow-ups:
//!
//! * The `hlstr` / `llstr` / `hlunicode` / `llunicode` and pointer-cast
//!   helper surfaces are present below, but their bodies still return
//!   structured pending errors until the corresponding `rstr`, `llmemory`,
//!   and cast specialization paths land.
//! * `ADTInterface.__call__` (annlowlevel.py) — the type
//!   surface is present; installing `_annenforceargs_` still lacks a
//!   `ConstValue` carrier for method attributes.
//! * `cachedtype` metaclass (annlowlevel.py) — metaclass
//!   shape has no direct Rust counterpart; callers reach for
//!   `once_cell::sync::Lazy` / `std::sync::LazyLock` + a manual cache
//!   map instead.
//!
//! ## Dependencies already in place
//!
//! * [`crate::annotator::annrpython::RPythonAnnotator::annotate_helper`]
//!   (annrpython.py) — primitive that `getgraph` wraps.
//! * [`crate::annotator::annrpython::RPythonAnnotator::complete_helpers`]
//!   (annrpython.py) — called by `finish_annotate`.
//! * [`crate::annotator::annrpython::RPythonAnnotator::using_policy`]
//!   (annrpython.py) — RAII policy swap guard used by both
//!   `getgraph` and `finish_annotate`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::getrepr`]
//!   (rtyper.py) — consumed by `getdelayedrepr`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::getcallable`]
//!   (rtyper.py) — consumed by `finish_rtype` to resolve
//!   `delayedfuncs`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::call_all_setups`]
//!   (rtyper.py) — called by `finish_rtype` between delayed
//!   resolution passes.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::specialize_more_blocks`]
//!   (rtyper.py) — called by `finish_rtype` to flow the
//!   newly annotated helpers.
//! * [`crate::translator::rtyper::normalizecalls::perform_normalizations`]
//!   (normalizecalls.py) — called by `finish_rtype`.
//! * [`crate::translator::rtyper::rmodel::Repr::set_setup_delayed`] +
//!   `set_setup_maybe_delayed` / `is_setup_delayed` (rmodel.py).
//!
//! Filling the method bodies from these primitives is strictly a
//! line-by-line transliteration of upstream.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::bookkeeper::{self, EmulatedPbcCallKey};
use crate::annotator::description::{FunctionDesc, GraphCacheKey};
use crate::annotator::model::{SomeObjectTrait, SomePtr, SomeValue, not_const};
use crate::annotator::policy::{
    AnnotatorPolicy, PolicyError, PolicyHandle, PolicyOps, Specializer,
    parse_specializer_directive, specializer_from_normalized,
};
use crate::annotator::signature::{AnnotationSpec, Sig, SigArgType, SignatureError};
use crate::flowspace::model::{BlockKey, ConstValue, Constant, GraphKey, GraphRef, HostObject};
use crate::flowspace::pygraph::PyGraph;

use super::error::TyperError;
use super::llannotation::{annotation_to_lltype, lltype_to_annotation};
use super::lltypesystem::lltype::{
    self, _ptr, DelayedPointer, FuncType, LowLevelType, Ptr, PtrTarget,
};
use super::rmodel::Repr;
use super::rmodel::{RTypeResult, inputconst_from_lltype};
use super::rpbc::FunctionRepr;
use super::rtyper::{GenopResult, HighLevelOp, RPythonTyper};

use crate::tool::sourcetools::valid_identifier;

fn already_annotation_sigarg(s_value: SomeValue) -> SigArgType {
    SigArgType::Spec(AnnotationSpec::Already(Box::new(s_value)))
}

fn signature_error(msg: impl Into<String>) -> SignatureError {
    SignatureError::new(msg)
}

fn lowlevel_type_from_const_for_signature(
    value: &ConstValue,
    context: &str,
) -> Result<LowLevelType, SignatureError> {
    match value {
        ConstValue::LowLevelType(t) => Ok((**t).clone()),
        other => Err(signature_error(format!(
            "{context}: expected a LowLevelType constant, got {other:?}"
        ))),
    }
}

fn lowlevel_ptr_type_from_const(value: &ConstValue, context: &str) -> Result<Ptr, TyperError> {
    let ConstValue::LowLevelType(t) = value else {
        return Err(TyperError::message(format!(
            "{context}: expected a LowLevelType constant, got {value:?}"
        )));
    };
    let LowLevelType::Ptr(ptr_t) = &**t else {
        return Err(TyperError::message(format!(
            "{context}: expected Ptr(FuncType), got {:?}",
            t
        )));
    };
    Ok((**ptr_t).clone())
}

fn ptr_to_annotation_sigarg(ptr: Ptr) -> SigArgType {
    already_annotation_sigarg(lltype_to_annotation(LowLevelType::Ptr(Box::new(ptr))))
}

fn lowlevel_attr_from_struct(
    struct_t: &lltype::Struct,
    attr: &str,
) -> Result<LowLevelType, String> {
    if let Some(value) = struct_t._adtmeths.get(attr) {
        return match value {
            ConstValue::LowLevelType(t) => Ok((**t).clone()),
            other => Err(format!(
                "lltype Struct._adtmeths[{attr:?}] is not a LowLevelType: {other:?}"
            )),
        };
    }
    if let Some(field_t) = struct_t._flds.get(attr) {
        return Ok(field_t.clone());
    }
    Err(struct_t._nofield(attr))
}

fn lowlevel_attr_from_type(t: &LowLevelType, attr: &str) -> Result<LowLevelType, String> {
    match t {
        LowLevelType::Struct(struct_t) => lowlevel_attr_from_struct(struct_t, attr),
        LowLevelType::Array(array_t) => match attr {
            "ITEM" | "OF" => Ok(array_t.OF.clone()),
            _ => Err(format!("Array has no attribute {attr:?}")),
        },
        LowLevelType::FixedSizeArray(array_t) => match attr {
            "ITEM" | "OF" => Ok(array_t.OF.clone()),
            _ => Err(format!("FixedSizeArray has no attribute {attr:?}")),
        },
        LowLevelType::ForwardReference(forward_ref) => {
            let resolved = forward_ref
                .resolved()
                .ok_or_else(|| format!("ForwardReference has no resolved attribute {attr:?}"))?;
            lowlevel_attr_from_type(&resolved, attr)
        }
        LowLevelType::Ptr(ptr_t) => lowlevel_attr_from_ptrtarget(&ptr_t.TO, attr),
        other => Err(format!("{other:?} has no low-level attribute {attr:?}")),
    }
}

fn lowlevel_attr_from_ptrtarget(t: &PtrTarget, attr: &str) -> Result<LowLevelType, String> {
    match t {
        PtrTarget::Struct(struct_t) => lowlevel_attr_from_struct(struct_t, attr),
        PtrTarget::Array(array_t) => match attr {
            "ITEM" | "OF" => Ok(array_t.OF.clone()),
            _ => Err(format!("Array has no attribute {attr:?}")),
        },
        PtrTarget::FixedSizeArray(array_t) => match attr {
            "ITEM" | "OF" => Ok(array_t.OF.clone()),
            _ => Err(format!("FixedSizeArray has no attribute {attr:?}")),
        },
        PtrTarget::ForwardReference(forward_ref) => {
            let resolved = forward_ref
                .resolved()
                .ok_or_else(|| format!("ForwardReference has no resolved attribute {attr:?}"))?;
            lowlevel_attr_from_type(&resolved, attr)
        }
        PtrTarget::Func(func_t) => match attr {
            "RESULT" => Ok(func_t.result.clone()),
            _ => Err(format!("FuncType has no low-level attribute {attr:?}")),
        },
        other => Err(format!("{other:?} has no low-level attribute {attr:?}")),
    }
}

pub fn placeholder_sigarg(s: &str) -> Result<SigArgType, TyperError> {
    match s {
        "self" => Ok(SigArgType::DynamicCallable(Rc::new(|inputcells| {
            let Some(s_self) = inputcells.first() else {
                return Err(signature_error("placeholder_sigarg('self'): missing self"));
            };
            match s_self {
                SomeValue::Ptr(_) => Ok(already_annotation_sigarg(s_self.clone())),
                other => Err(signature_error(format!(
                    "placeholder_sigarg('self') expected SomePtr, got {other:?}"
                ))),
            }
        }))),
        "SELF" => Err(TyperError::message(
            "annlowlevel.placeholder_sigarg('SELF') raises NotImplementedError",
        )),
        _ if s.chars().all(|c| c.is_ascii_lowercase()) => {
            let attr = s.to_ascii_uppercase();
            Ok(SigArgType::DynamicCallable(Rc::new(move |inputcells| {
                let Some(s_self) = inputcells.first() else {
                    return Err(signature_error(format!(
                        "placeholder_sigarg({attr:?}): missing self"
                    )));
                };
                let SomeValue::Ptr(s_self) = s_self else {
                    return Err(signature_error(format!(
                        "placeholder_sigarg({attr:?}) expected SomePtr, got {s_self:?}"
                    )));
                };
                let t = lowlevel_attr_from_ptrtarget(&s_self.ll_ptrtype.TO, &attr)
                    .map_err(signature_error)?;
                Ok(already_annotation_sigarg(lltype_to_annotation(t)))
            })))
        }
        _ => Err(TyperError::message(format!(
            "placeholder_sigarg expected lowercase placeholder, got {s:?}"
        ))),
    }
}

pub fn typemeth_placeholder_sigarg(s: &str) -> Result<SigArgType, TyperError> {
    match s {
        "SELF" => Ok(SigArgType::DynamicCallable(Rc::new(|inputcells| {
            let Some(s_type) = inputcells.first() else {
                return Err(signature_error(
                    "typemeth_placeholder_sigarg('SELF'): missing SELF",
                ));
            };
            match s_type {
                SomeValue::PBC(_) if s_type.is_constant() => {
                    Ok(already_annotation_sigarg(s_type.clone()))
                }
                other => Err(signature_error(format!(
                    "typemeth_placeholder_sigarg('SELF') expected constant SomePBC, got {other:?}"
                ))),
            }
        }))),
        "self" => Ok(SigArgType::DynamicCallable(Rc::new(|inputcells| {
            let Some(s_type) = inputcells.first() else {
                return Err(signature_error(
                    "typemeth_placeholder_sigarg('self'): missing SELF",
                ));
            };
            if !matches!(s_type, SomeValue::PBC(_)) || !s_type.is_constant() {
                return Err(signature_error(format!(
                    "typemeth_placeholder_sigarg('self') expected constant SomePBC, got {s_type:?}"
                )));
            }
            let ty = lowlevel_type_from_const_for_signature(
                s_type.const_().expect("is_constant checked"),
                "typemeth_placeholder_sigarg('self')",
            )?;
            let ptr = Ptr::from_container_type(ty).map_err(signature_error)?;
            Ok(ptr_to_annotation_sigarg(ptr))
        }))),
        _ if s.chars().all(|c| c.is_ascii_lowercase()) => {
            let attr = s.to_ascii_uppercase();
            Ok(SigArgType::DynamicCallable(Rc::new(move |inputcells| {
                let Some(s_type) = inputcells.first() else {
                    return Err(signature_error(format!(
                        "typemeth_placeholder_sigarg({attr:?}): missing SELF"
                    )));
                };
                if !matches!(s_type, SomeValue::PBC(_)) || !s_type.is_constant() {
                    return Err(signature_error(format!(
                        "typemeth_placeholder_sigarg({attr:?}) expected constant SomePBC, got {s_type:?}"
                    )));
                }
                let ty = lowlevel_type_from_const_for_signature(
                    s_type.const_().expect("is_constant checked"),
                    &format!("typemeth_placeholder_sigarg({attr:?})"),
                )?;
                let item_t = lowlevel_attr_from_type(&ty, &attr).map_err(signature_error)?;
                Ok(already_annotation_sigarg(lltype_to_annotation(item_t)))
            })))
        }
        _ => Err(TyperError::message(format!(
            "typemeth_placeholder_sigarg expected lowercase placeholder, got {s:?}"
        ))),
    }
}

// annlowlevel.py — KeyComp

/// RPython `class KeyComp(object)` (annlowlevel.py).
///
/// Wrapper used by `LowLevelAnnotatorPolicy.lowlevelspecialize` to
/// build cache keys out of PBC constants and low-level types. The
/// upstream class overrides `__eq__` / `__hash__` / `__str__` so that
/// cache-key comparisons are value-based rather than identity-based
/// even when the wrapped value is a mutable Python object.
///
/// **Port pending.** The concrete `value` payload is one of:
///
/// * a [`LowLevelType`] (stringified via `_short_name` + `'LlT'`),
/// * a Python object with `__name__` or `compact_repr()` + `'Const'`,
/// * or a general `repr(val) + 'Const'`.
///
/// Representing that union as a Rust enum with the same `Eq + Hash`
/// semantics is straightforward once the repr-based path is ported —
/// until then, constructors return this stub.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyComp {
    /// Original upstream attribute `self.val`. Mirrored here verbatim;
    /// kept `pub` so test helpers can observe the seed value.
    pub val: KeyCompValue,
}

/// Internal tag for the wrapped value. Keeps the [`KeyComp`] API
/// uniform while the various branches are ported one at a time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyCompValue {
    /// `lltype.LowLevelType` branch (annlowlevel.py).
    Lltype(LowLevelType),
    /// Generic constant branch (annlowlevel.py:33-39).
    Const(ConstValue),
}

impl KeyComp {
    /// Construct a `KeyComp` from a wrapped value.
    pub fn new(val: KeyCompValue) -> Self {
        KeyComp { val }
    }
}

/// Upstream `KeyComp.__str__` / `__repr__` (annlowlevel.py):
/// `LowLevelType` → `'<short_name>LlT'`; general consts append `'Const'`.
impl std::fmt::Display for KeyComp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.val {
            KeyCompValue::Lltype(t) => write!(f, "{}LlT", t.short_name()),
            KeyCompValue::Const(ConstValue::HostObject(h)) => {
                let name = h.qualname().rsplit('.').next().unwrap_or(h.qualname());
                write!(f, "{}Const", name)
            }
            KeyCompValue::Const(value) => write!(f, "{}Const", value),
        }
    }
}

// annlowlevel.py — LowLevelAnnotatorPolicy

/// RPython `class LowLevelAnnotatorPolicy(AnnotatorPolicy)`
/// (annlowlevel.py).
///
/// Specialization policy that keys the cached graph by the lltype of
/// each positional argument, with constant-PBC arguments promoted to
/// a [`KeyComp`]-wrapped identity key.
///
/// Upstream is a subclass of `AnnotatorPolicy`; the Rust port composes
/// rather than inherits — `base` holds the default policy and
/// overrides are implemented as direct methods on this struct.
#[derive(Clone, Debug)]
pub struct LowLevelAnnotatorPolicy {
    /// RPython `self.rtyper = rtyper` (annlowlevel.py __init__). Weak
    /// because the rtyper owns the strong reference and this policy
    /// lives inside `MixLevelHelperAnnotator` whose lifetime is
    /// ≤ rtyper's.
    pub rtyper: Weak<RPythonTyper>,
    /// Base-policy fallback for non-`ll_`-prefixed callees in the
    /// `MixLevelAnnotatorPolicy` subclass override. Stored on the
    /// struct rather than derived per-call because `AnnotatorPolicy`
    /// is trivially cloneable.
    pub base: AnnotatorPolicy,
}

impl LowLevelAnnotatorPolicy {
    /// RPython `LowLevelAnnotatorPolicy.__init__(self, rtyper=None)`
    /// (annlowlevel.py).
    pub fn new(rtyper: Option<&Rc<RPythonTyper>>) -> Self {
        LowLevelAnnotatorPolicy {
            rtyper: rtyper.map(Rc::downgrade).unwrap_or_default(),
            base: AnnotatorPolicy::new(),
        }
    }

    /// RPython `LowLevelAnnotatorPolicy.lowlevelspecialize(funcdesc,
    /// args_s, key_for_args)` (annlowlevel.py).
    ///
    /// ```python
    /// @staticmethod
    /// def lowlevelspecialize(funcdesc, args_s, key_for_args):
    ///     args_s, key1, builder = flatten_star_args(funcdesc, args_s)
    ///     key = []
    ///     new_args_s = []
    ///     for i, s_obj in enumerate(args_s):
    ///         if i in key_for_args:
    ///             key.append(key_for_args[i])
    ///             new_args_s.append(s_obj)
    ///         elif isinstance(s_obj, annmodel.SomePBC):
    ///             assert s_obj.is_constant(), "ambiguous low-level helper specialization"
    ///             key.append(KeyComp(s_obj.const))
    ///             new_args_s.append(s_obj)
    ///         elif isinstance(s_obj, annmodel.SomeNone):
    ///             key.append(KeyComp(None))
    ///             new_args_s.append(s_obj)
    ///         else:
    ///             new_args_s.append(annmodel.not_const(s_obj))
    ///             try:
    ///                 key.append(annotation_to_lltype(s_obj))
    ///             except ValueError:
    ///                 key.append(s_obj.__class__)
    ///     key = (tuple(key),)
    ///     if key1 is not None:
    ///         key += (key1,)
    ///     flowgraph = funcdesc.cachedgraph(key, builder=builder)
    ///     args_s[:] = new_args_s
    ///     return flowgraph
    /// ```
    ///
    /// Port pending. Depends on
    /// [`crate::annotator::specialize::flatten_star_args`],
    /// [`crate::translator::rtyper::llannotation::annotation_to_lltype`],
    /// and `FunctionDesc::cachedgraph` — all already ported.
    pub fn lowlevelspecialize(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<Option<SomeValue>>,
        key_for_args: &HashMap<usize, KeyComp>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let (args_s_flat, key1, builder) =
            crate::annotator::specialize::flatten_star_args(funcdesc, args_s)
                .map_err(|e| TyperError::message(e.to_string()))?;
        let mut key = Vec::new();
        let mut new_args_s: Vec<Option<SomeValue>> = Vec::new();
        for (i, s_slot) in args_s_flat.iter().enumerate() {
            // upstream `annlowlevel.py:218-241` reads
            // `args_s[i].is_constant() / .const / .__class__`; an
            // unbound cell raises AttributeError there.  Mirror with
            // fail-loud at access on the .as_ref().
            let s_obj = s_slot.as_ref().ok_or_else(|| {
                TyperError::message(format!(
                    "lowlevelspecialize({}): args_s[{i}] is unannotated",
                    funcdesc.name,
                ))
            })?;
            if let Some(keycomp) = key_for_args.get(&i) {
                key.push(GraphCacheKey::KeyComp(keycomp.clone()));
                new_args_s.push(Some(s_obj.clone()));
            } else if let SomeValue::PBC(_pbc) = s_obj {
                if !s_obj.is_constant() {
                    return Err(TyperError::message(
                        "ambiguous low-level helper specialization",
                    ));
                }
                let const_value = s_obj
                    .const_()
                    .cloned()
                    .expect("constant SomePBC must carry const()");
                key.push(GraphCacheKey::KeyComp(KeyComp::new(KeyCompValue::Const(
                    const_value,
                ))));
                new_args_s.push(Some(s_obj.clone()));
            } else if matches!(s_obj, SomeValue::None_(_)) {
                key.push(GraphCacheKey::KeyComp(KeyComp::new(KeyCompValue::Const(
                    ConstValue::None,
                ))));
                new_args_s.push(Some(s_obj.clone()));
            } else {
                new_args_s.push(Some(not_const(s_obj)));
                match annotation_to_lltype(s_obj, None) {
                    Ok(lltype) => key.push(GraphCacheKey::LowLevelType(lltype)),
                    Err(_) => key.push(GraphCacheKey::SomeValueTag(s_obj.tag())),
                }
            }
        }
        let mut key = vec![GraphCacheKey::Tuple(key)];
        if !matches!(key1, GraphCacheKey::None) {
            key.push(key1);
        }
        let flowgraph = funcdesc
            .cachedgraph(GraphCacheKey::Tuple(key), None, builder)
            .map_err(|e| TyperError::message(e.to_string()))?;
        *args_s = new_args_s;
        Ok(flowgraph)
    }

    /// RPython `LowLevelAnnotatorPolicy.default_specialize(funcdesc,
    /// args_s)` (annlowlevel.py). Thin alias:
    ///
    /// ```python
    /// return LowLevelAnnotatorPolicy.lowlevelspecialize(funcdesc, args_s, {})
    /// ```
    pub fn default_specialize(
        &self,
        funcdesc: &crate::annotator::description::FunctionDesc,
        args_s: &mut Vec<Option<SomeValue>>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        self.lowlevelspecialize(funcdesc, args_s, &std::collections::HashMap::new())
    }

    /// RPython `specialize__ll = default_specialize` (annlowlevel.py).
    /// Kept as a distinct method so the
    /// [`crate::annotator::policy::Specializer::Ll`] dispatch can
    /// call it by its upstream attribute name.
    pub fn specialize_ll(
        &self,
        funcdesc: &crate::annotator::description::FunctionDesc,
        args_s: &mut Vec<Option<SomeValue>>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        self.default_specialize(funcdesc, args_s)
    }

    /// RPython `LowLevelAnnotatorPolicy.specialize__ll_and_arg(funcdesc,
    /// args_s, *argindices)` (annlowlevel.py).
    ///
    /// ```python
    /// @staticmethod
    /// def specialize__ll_and_arg(funcdesc, args_s, *argindices):
    ///     keys = {}
    ///     for i in argindices:
    ///         keys[i] = args_s[i].const
    ///     return LowLevelAnnotatorPolicy.lowlevelspecialize(
    ///         funcdesc, args_s, keys)
    /// ```
    ///
    /// Port pending.
    pub fn specialize_ll_and_arg(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<Option<SomeValue>>,
        argindices: &[usize],
    ) -> Result<Rc<PyGraph>, TyperError> {
        let mut keys = HashMap::new();
        for &i in argindices {
            let const_value = args_s
                .get(i)
                .and_then(|s| s.as_ref())
                .and_then(SomeValue::const_)
                .cloned()
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "specialize__ll_and_arg: argument {i} must be constant"
                    ))
                })?;
            keys.insert(i, KeyComp::new(KeyCompValue::Const(const_value)));
        }
        self.lowlevelspecialize(funcdesc, args_s, &keys)
    }
}

impl PolicyOps for LowLevelAnnotatorPolicy {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            return Ok(Specializer::LowLevelDefault);
        };
        let (normalized, parms) = parse_specializer_directive(directive)?;
        match normalized.as_str() {
            "default_specialize" => Ok(Specializer::LowLevelDefault),
            _ => specializer_from_normalized(&normalized, parms),
        }
    }

    fn no_more_blocks_to_annotate(
        &self,
        ann: &RPythonAnnotator,
    ) -> Result<(), crate::annotator::model::AnnotatorError> {
        self.base.no_more_blocks_to_annotate(ann)
    }
}

impl From<LowLevelAnnotatorPolicy> for PolicyHandle {
    fn from(value: LowLevelAnnotatorPolicy) -> Self {
        PolicyHandle::new(value)
    }
}

// annlowlevel.py — annotate_lowlevel_helper

/// RPython `annotate_lowlevel_helper(annotator, ll_function, args_s,
/// policy=None)` (annlowlevel.py).
///
/// ```python
/// def annotate_lowlevel_helper(annotator, ll_function, args_s, policy=None):
///     if policy is None:
///         policy = LowLevelAnnotatorPolicy()
///     return annotator.annotate_helper(ll_function, args_s, policy)
/// ```
///
/// Port pending. The Rust side already has
/// [`RPythonAnnotator::annotate_helper`], so the body reduces to:
/// construct a default `LowLevelAnnotatorPolicy` when `policy` is
/// `None`, then delegate. The one obstacle is that
/// `annotate_helper` currently consumes `AnnotatorPolicy` by value
/// and the low-level variant would need to pass through the ll-policy
/// methods — follow-up work threads the policy as a trait object.
pub fn annotate_lowlevel_helper(
    annotator: &Rc<RPythonAnnotator>,
    ll_function: &HostObject,
    args_s: Vec<SomeValue>,
    policy: Option<LowLevelAnnotatorPolicy>,
) -> Result<Rc<PyGraph>, TyperError> {
    let policy = policy.unwrap_or_else(|| LowLevelAnnotatorPolicy::new(None));
    annotator
        .annotate_helper(ll_function, args_s, Some(PolicyHandle::from(policy)))
        .map_err(|e| TyperError::message(e.to_string()))
}

// annlowlevel.py — MixLevelAnnotatorPolicy

/// RPython `class MixLevelAnnotatorPolicy(LowLevelAnnotatorPolicy)`
/// (annlowlevel.py).
///
/// Specialization policy used by [`MixLevelHelperAnnotator`]. Extends
/// `LowLevelAnnotatorPolicy` so that:
///
/// * Callees whose name starts with `ll_` or `_ll_` go through
///   `LowLevelAnnotatorPolicy.default_specialize` (i.e. lltype-keyed
///   cache).
/// * Every other callee falls back to plain
///   `AnnotatorPolicy.default_specialize`.
///
/// Upstream `__init__` takes an `annhelper` (the
/// `MixLevelHelperAnnotator` instance) and copies its
/// `annhelper.rtyper` onto `self.rtyper`. The Rust port wraps a
/// `LowLevelAnnotatorPolicy` directly because the only rtyper
/// reference the policy ever reads is `self.rtyper` — inheriting
/// from `LowLevelAnnotatorPolicy` via composition captures that
/// shape without the `annhelper` → `rtyper` indirection.
#[derive(Clone, Debug)]
pub struct MixLevelAnnotatorPolicy {
    /// Composed base — `LowLevelAnnotatorPolicy.rtyper` is the only
    /// field upstream inherits via `super().__init__`.
    pub ll: LowLevelAnnotatorPolicy,
}

impl MixLevelAnnotatorPolicy {
    /// RPython `MixLevelAnnotatorPolicy.__init__(self, annhelper)`
    /// (annlowlevel.py).
    pub fn new(annhelper: &MixLevelHelperAnnotator) -> Self {
        MixLevelAnnotatorPolicy {
            ll: LowLevelAnnotatorPolicy {
                rtyper: annhelper.rtyper.clone(),
                base: AnnotatorPolicy::new(),
            },
        }
    }

    /// RPython `MixLevelAnnotatorPolicy.default_specialize(self,
    /// funcdesc, args_s)` (annlowlevel.py).
    ///
    /// ```python
    /// def default_specialize(self, funcdesc, args_s):
    ///     name = funcdesc.name
    ///     if name.startswith('ll_') or name.startswith('_ll_'):
    ///         return super(MixLevelAnnotatorPolicy, self) \
    ///             .default_specialize(funcdesc, args_s)
    ///     else:
    ///         return AnnotatorPolicy.default_specialize(funcdesc, args_s)
    /// ```
    ///
    /// Port pending.
    pub fn default_specialize(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<Option<SomeValue>>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let name = &funcdesc.name;
        if name.starts_with("ll_") || name.starts_with("_ll_") {
            self.ll.default_specialize(funcdesc, args_s)
        } else {
            crate::annotator::specialize::default_specialize(funcdesc, args_s)
                .map_err(|e| TyperError::message(e.to_string()))
        }
    }

    /// RPython `MixLevelAnnotatorPolicy.specialize__arglltype(self,
    /// funcdesc, args_s, i)` (annlowlevel.py).
    ///
    /// ```python
    /// def specialize__arglltype(self, funcdesc, args_s, i):
    ///     key = self.rtyper.getrepr(args_s[i]).lowleveltype
    ///     alt_name = funcdesc.name + "__for_%sLlT" % key._short_name()
    ///     return funcdesc.cachedgraph(key, alt_name=valid_identifier(alt_name))
    /// ```
    ///
    /// Port pending. Consumers already reach
    /// [`RPythonTyper::getrepr`] via
    /// [`LowLevelAnnotatorPolicy::rtyper`].
    pub fn specialize_arglltype(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &[Option<SomeValue>],
        i: usize,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let rtyper = self
            .ll
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelAnnotatorPolicy: rtyper dropped"))?;
        let s = args_s[i].as_ref().ok_or_else(|| {
            TyperError::message(format!("specialize__arglltype: args_s[{i}] is unannotated"))
        })?;
        let key = rtyper.getrepr(s)?.lowleveltype().clone();
        let alt_name = valid_identifier(format!("{}__for_{}LlT", funcdesc.name, key.short_name()));
        funcdesc
            .cachedgraph(GraphCacheKey::LowLevelType(key), Some(&alt_name), None)
            .map_err(|e| TyperError::message(e.to_string()))
    }

    /// RPython `MixLevelAnnotatorPolicy.specialize__genconst(self,
    /// funcdesc, args_s, i)` (annlowlevel.py).
    ///
    /// Upstream comment: `# XXX this is specific to the JIT`. Coerces
    /// `args_s[i]` to a `lltype_to_annotation(TYPE)` and keys on the
    /// resulting lltype.
    ///
    /// Port pending.
    pub fn specialize_genconst(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut [Option<SomeValue>],
        i: usize,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let s = args_s[i].as_ref().ok_or_else(|| {
            TyperError::message(format!("specialize__genconst: args_s[{i}] is unannotated"))
        })?;
        let typ = annotation_to_lltype(s, Some("genconst"))
            .map_err(|e| TyperError::message(e.to_string()))?;
        args_s[i] = Some(lltype_to_annotation(typ.clone()));
        let alt_name = valid_identifier(format!("{}__{}", funcdesc.name, typ.short_name()));
        funcdesc
            .cachedgraph(GraphCacheKey::LowLevelType(typ), Some(&alt_name), None)
            .map_err(|e| TyperError::message(e.to_string()))
    }
}

impl PolicyOps for MixLevelAnnotatorPolicy {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            return Ok(Specializer::MixLevelDefault);
        };
        let (normalized, parms) = parse_specializer_directive(directive)?;
        match normalized.as_str() {
            "default_specialize" => Ok(Specializer::MixLevelDefault),
            "specialize__arglltype" => Ok(Specializer::ArgLltype { parms }),
            "specialize__genconst" => Ok(Specializer::GenConst { parms }),
            _ => specializer_from_normalized(&normalized, parms),
        }
    }

    fn no_more_blocks_to_annotate(
        &self,
        ann: &RPythonAnnotator,
    ) -> Result<(), crate::annotator::model::AnnotatorError> {
        self.ll.base.no_more_blocks_to_annotate(ann)
    }
}

impl From<MixLevelAnnotatorPolicy> for PolicyHandle {
    fn from(value: MixLevelAnnotatorPolicy) -> Self {
        PolicyHandle::new(value)
    }
}

// annlowlevel.py — MixLevelHelperAnnotator

/// One entry of [`MixLevelHelperAnnotator::pending`] — matches the
/// upstream 4-tuple `(ll_function, graph, args_s, s_result)`
/// (annlowlevel.py).
#[derive(Debug, Clone)]
pub struct PendingHelper {
    pub ll_function: HostObject,
    pub graph: Rc<PyGraph>,
    pub args_s: Vec<Option<SomeValue>>,
    pub s_result: SomeValue,
}

/// One entry of [`MixLevelHelperAnnotator::delayedconsts`] — matches
/// upstream `(delayedptr, repr, obj)` (annlowlevel.py).
#[derive(Debug, Clone)]
pub struct DelayedConst {
    pub delayedptr: _ptr,
    pub repr: Arc<dyn Repr>,
    /// Upstream `obj` is a Python value of arbitrary type that the
    /// repr's `convert_const` later lowers.
    pub obj: ConstValue,
}

/// One entry of [`MixLevelHelperAnnotator::delayedfuncs`] — matches
/// upstream `(delayedptr, graph)` (annlowlevel.py).
#[derive(Debug, Clone)]
pub struct DelayedFunc {
    pub delayedptr: _ptr,
    pub graph: Rc<PyGraph>,
}

/// RPython `class PseudoHighLevelCallable(object)`
/// (annlowlevel.py).
#[derive(Debug, Clone)]
pub struct PseudoHighLevelCallable {
    pub llfnptr: _ptr,
    pub args_s: Vec<Option<SomeValue>>,
    pub s_result: SomeValue,
}

impl PseudoHighLevelCallable {
    pub fn new(llfnptr: _ptr, args_s: Vec<Option<SomeValue>>, s_result: SomeValue) -> Self {
        Self {
            llfnptr,
            args_s,
            s_result,
        }
    }

    /// RPython `PseudoHighLevelCallable.__call__` (annlowlevel.py).
    #[allow(non_snake_case)]
    pub fn __call__(&self, _args: &[SomeValue]) -> Result<SomeValue, TyperError> {
        Err(TyperError::message(
            "PseudoHighLevelCallable objects are not really callable directly",
        ))
    }
}

/// RPython `class PseudoHighLevelCallableEntry(ExtRegistryEntry)`
/// (annlowlevel.py).
#[derive(Debug, Clone, Default)]
pub struct PseudoHighLevelCallableEntry;

impl PseudoHighLevelCallableEntry {
    /// RPython `compute_result_annotation(self, *args_s)`
    /// (annlowlevel.py).
    pub fn compute_result_annotation(
        &self,
        instance: &PseudoHighLevelCallable,
        _args_s: &[SomeValue],
    ) -> SomeValue {
        instance.s_result.clone()
    }

    /// RPython `specialize_call(self, hop)` (annlowlevel.py).
    pub fn specialize_call(
        &self,
        instance: &PseudoHighLevelCallable,
        hop: &HighLevelOp,
    ) -> RTypeResult {
        let args_r = instance
            .args_s
            .iter()
            .map(|s| {
                let s = s.as_ref().ok_or_else(|| {
                    TyperError::message(
                        "PseudoHighLevelCallableEntry.specialize_call: missing argument annotation",
                    )
                })?;
                hop.rtyper.getrepr(s)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let r_res = hop.rtyper.getrepr(&instance.s_result)?;
        let vlist = hop.inputargs(args_r.iter().map(|r| r.into()).collect())?;
        let p = instance.llfnptr.clone();
        let func_ptr_type = lltype::typeOf(&p);
        let c_func = inputconst_from_lltype(
            &LowLevelType::Ptr(Box::new(func_ptr_type.clone())),
            &ConstValue::LLPtr(Box::new(p)),
        )?;
        let PtrTarget::Func(functype) = &func_ptr_type.TO else {
            return Err(TyperError::message(
                "PseudoHighLevelCallableEntry.specialize_call: llfnptr is not Ptr(FuncType)",
            ));
        };
        for (r_arg, argtype) in args_r.iter().zip(functype.args.iter()) {
            if r_arg.lowleveltype() != argtype {
                return Err(TyperError::message(format!(
                    "PseudoHighLevelCallableEntry.specialize_call: argument lowleveltype mismatch: got {:?}, expected {:?}",
                    r_arg.lowleveltype(),
                    argtype
                )));
            }
        }
        if r_res.lowleveltype() != &functype.result {
            return Err(TyperError::message(format!(
                "PseudoHighLevelCallableEntry.specialize_call: result lowleveltype mismatch: got {:?}, expected {:?}",
                r_res.lowleveltype(),
                functype.result
            )));
        }
        hop.exception_is_here()?;
        let mut call_args = Vec::with_capacity(vlist.len() + 1);
        call_args.push(crate::flowspace::model::Hlvalue::Constant(c_func));
        call_args.extend(vlist);
        Ok(hop.genop("direct_call", call_args, GenopResult::Repr(r_res)))
    }
}

/// RPython `class LLHelperEntry(ExtRegistryEntry)` (annlowlevel.py).
#[derive(Debug, Clone, Default)]
pub struct LLHelperEntry;

/// RPython `llhelper(F, f)` (annlowlevel.py).
///
/// ```python
/// return lltype.functionptr(F.TO, f.__name__, _callable=f)
/// ```
///
/// Rust's `_func` stores `_callable` as a stable string rather than a live
/// Python function object, so the host-object identity is included to avoid
/// collapsing two distinct function objects that share a display name.
#[allow(non_snake_case)]
pub fn llhelper(F: Ptr, f: &HostObject) -> Result<_ptr, TyperError> {
    let PtrTarget::Func(func_t) = F.TO else {
        return Err(TyperError::message("llhelper expects Ptr(FuncType)"));
    };
    Ok(lltype::functionptr(
        func_t,
        f.simple_name(),
        None,
        Some(format!("{}#{}", f.qualname(), f.identity_id())),
    ))
}

/// RPython `llhelper_args(f, ARGS, RESULT)` (annlowlevel.py).
#[allow(non_snake_case)]
pub fn llhelper_args(f: &HostObject, ARGS: Vec<LowLevelType>, RESULT: LowLevelType) -> _ptr {
    llhelper(
        Ptr {
            TO: PtrTarget::Func(FuncType {
                args: ARGS,
                result: RESULT,
            }),
        },
        f,
    )
    .expect("fresh Ptr(FuncType) must be accepted by llhelper")
}

impl LLHelperEntry {
    /// RPython `compute_result_annotation(self, s_F, s_callable)`
    /// (annlowlevel.py).
    pub fn compute_result_annotation(
        &self,
        s_f: &SomeValue,
        s_callable: &SomeValue,
    ) -> Result<SomeValue, TyperError> {
        if !s_f.is_constant() {
            return Err(TyperError::message(
                "LLHelperEntry.compute_result_annotation: s_F must be constant",
            ));
        }
        let SomeValue::PBC(pbc) = s_callable else {
            return Err(TyperError::message(
                "LLHelperEntry.compute_result_annotation: s_callable must be SomePBC",
            ));
        };
        let f = lowlevel_ptr_type_from_const(
            s_f.const_().expect("is_constant checked"),
            "LLHelperEntry.compute_result_annotation",
        )?;
        let PtrTarget::Func(func) = &f.TO else {
            return Err(TyperError::message(
                "LLHelperEntry.compute_result_annotation: F must be Ptr(FuncType)",
            ));
        };
        let args_s = func
            .args
            .iter()
            .cloned()
            .map(lltype_to_annotation)
            .collect::<Vec<_>>();
        let bookkeeper = bookkeeper::getbookkeeper().ok_or_else(|| {
            TyperError::message(
                "LLHelperEntry.compute_result_annotation missing annotator::bookkeeper::getbookkeeper",
            )
        })?;
        for desc in pbc.descriptions.values() {
            let _funcdesc: Rc<RefCell<FunctionDesc>> = desc.as_function().ok_or_else(|| {
                TyperError::message(
                    "LLHelperEntry.compute_result_annotation: description is not FunctionDesc",
                )
            })?;
            let pyobj = desc.pyobj().ok_or_else(|| {
                TyperError::message(
                    "LLHelperEntry.compute_result_annotation: FunctionDesc.pyobj is missing",
                )
            })?;
            if pbc.is_constant() {
                let Some(ConstValue::HostObject(const_obj)) = s_callable.const_() else {
                    return Err(TyperError::message(
                        "LLHelperEntry.compute_result_annotation: constant SomePBC must carry HostObject",
                    ));
                };
                if const_obj != &pyobj {
                    return Err(TyperError::message(
                        "LLHelperEntry.compute_result_annotation: constant SomePBC does not match desc.pyobj",
                    ));
                }
            }
            let key = EmulatedPbcCallKey::LLHelper {
                callable_id: pyobj.identity_id(),
            };
            let s_res = bookkeeper
                .emulate_pbc_call(key, s_callable, &args_s, &[], None)
                .map_err(|e| TyperError::message(format!("{e:?}")))?;
            if !lltype_to_annotation(func.result.clone()).contains(&s_res) {
                return Err(TyperError::message(format!(
                    "LLHelperEntry.compute_result_annotation: result {:?} is not contained in {:?}",
                    s_res,
                    lltype_to_annotation(func.result.clone())
                )));
            }
        }
        Ok(SomeValue::Ptr(SomePtr::new(f)))
    }

    /// RPython `specialize_call(self, hop)` (annlowlevel.py).
    pub fn specialize_call(&self, hop: &HighLevelOp) -> RTypeResult {
        hop.exception_cannot_occur()?;
        let callable_is_constant = {
            let args_s = hop.args_s.borrow();
            args_s
                .get(1)
                .ok_or_else(|| {
                    TyperError::message("LLHelperEntry.specialize_call: missing callable argument")
                })?
                .is_constant()
        };
        let r_callable = {
            let args_r = hop.args_r.borrow();
            args_r.get(1).and_then(Clone::clone).ok_or_else(|| {
                TyperError::message("LLHelperEntry.specialize_call: missing callable argument repr")
            })?
        };
        if callable_is_constant {
            let r_func = (r_callable.as_ref() as &dyn std::any::Any)
                .downcast_ref::<FunctionRepr>()
                .ok_or_else(|| {
                    TyperError::message(
                        "LLHelperEntry.specialize_call: callable repr is not FunctionRepr",
                    )
                })?;
            return Ok(Some(r_func.get_unique_llfn()?));
        }
        let f = {
            let args_s = hop.args_s.borrow();
            let s_f = args_s.first().ok_or_else(|| {
                TyperError::message("LLHelperEntry.specialize_call: missing F argument")
            })?;
            lowlevel_ptr_type_from_const(
                s_f.const_().ok_or_else(|| {
                    TyperError::message("LLHelperEntry.specialize_call: F must be constant")
                })?,
                "LLHelperEntry.specialize_call",
            )?
        };
        let expected = LowLevelType::Ptr(Box::new(f));
        if r_callable.lowleveltype() != &expected {
            return Err(TyperError::message(format!(
                "LLHelperEntry.specialize_call: callable lowleveltype mismatch: got {:?}, expected {:?}",
                r_callable.lowleveltype(),
                expected
            )));
        }
        Ok(Some(hop.inputarg(&r_callable, 1)?))
    }
}

/// One argument item in an `ADTInterface` signature template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ADTSigArg {
    Placeholder(String),
    Explicit(SigArgType),
}

/// RPython `sigtemplates` value: `name -> (args, result)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ADTSigTemplate {
    pub args: Vec<ADTSigArg>,
    pub result: SigArgType,
}

/// RPython `class ADTInterface(object)` (annlowlevel.py).
#[derive(Debug, Clone)]
pub struct ADTInterface {
    pub sigtemplates: HashMap<String, ADTSigTemplate>,
    pub base: Option<Rc<ADTInterface>>,
    pub sigs: HashMap<String, Sig>,
}

impl ADTInterface {
    /// RPython `ADTInterface.__init__(self, base, sigtemplates)`
    /// (annlowlevel.py).
    pub fn new(
        base: Option<Rc<ADTInterface>>,
        sigtemplates: HashMap<String, ADTSigTemplate>,
    ) -> Result<Self, TyperError> {
        let mut sigs = base
            .as_ref()
            .map(|base| base.sigs.clone())
            .unwrap_or_default();
        for (name, template) in &sigtemplates {
            let first = template.args.first().ok_or_else(|| {
                TyperError::message("ADTInterface signature must have at least one argument")
            })?;
            let make_expand: fn(&str) -> Result<SigArgType, TyperError> = match first {
                ADTSigArg::Placeholder(s) if s == "self" => placeholder_sigarg,
                ADTSigArg::Placeholder(s) if s == "SELF" => typemeth_placeholder_sigarg,
                _ => {
                    return Err(TyperError::message(
                        "ADTInterface signature should start with 'SELF' or 'self'",
                    ));
                }
            };
            let mut sigargs = Vec::with_capacity(template.args.len());
            for arg in &template.args {
                match arg {
                    ADTSigArg::Placeholder(s) => sigargs.push(make_expand(s)?),
                    ADTSigArg::Explicit(arg) => sigargs.push(arg.clone()),
                }
            }
            sigs.insert(name.clone(), Sig::new(sigargs));
        }
        Ok(Self {
            sigtemplates,
            base,
            sigs,
        })
    }

    /// RPython `ADTInterface.__call__(self, adtmeths)` (annlowlevel.py).
    ///
    /// `adtmeths` is the method mapping (`name -> ll function`, stored as
    /// `lltype.Struct._adtmeths`); upstream installs each `sig` as the
    /// method's `_annenforceargs_` and returns the mapping. Body deferred
    /// until the mix-level helper surface lands; arity preserved.
    pub fn __call__(
        &self,
        _adtmeths: HashMap<String, ConstValue>,
    ) -> Result<HashMap<String, ConstValue>, TyperError> {
        Err(TyperError::missing_rtype_operation(
            "annlowlevel.ADTInterface.__call__ deferred: missing ConstValue::_annenforceargs_ carrier"
                .to_string(),
        ))
    }
}

/// RPython `class MixLevelHelperAnnotator(object)` (annlowlevel.py:
/// :126-284).
///
/// Stateful coordinator that buffers helper-graph registrations,
/// annotates them in one pass, and then rtypes them in a second
/// pass. Callers interleave `getgraph` / `delayedfunction` /
/// `constfunc` / `getdelayedrepr` / `delayedconst` calls with
/// regular rtyper work, then invoke [`MixLevelHelperAnnotator::finish`]
/// once when the surrounding phase is done to flush everything.
///
/// Upstream field mapping:
///
/// | Python | Rust |
/// |---|---|
/// | `self.rtyper` | `rtyper: Weak<RPythonTyper>` |
/// | `self.policy` | `policy: MixLevelAnnotatorPolicy` |
/// | `self.pending: list` | `pending: RefCell<Vec<PendingHelper>>` |
/// | `self.delayedreprs: set` | `delayedreprs: RefCell<Vec<Arc<dyn Repr>>>` (pointer-id dedup) |
/// | `self.delayedconsts: list` | `delayedconsts: RefCell<Vec<DelayedConst>>` |
/// | `self.delayedfuncs: list` | `delayedfuncs: RefCell<Vec<DelayedFunc>>` |
/// | `self.newgraphs: set` | `newgraphs: RefCell<Vec<GraphRef>>` (pointer-id dedup) |
///
/// The upstream `set` containers become `Vec` because `dyn Repr` /
/// `GraphRef` have no trivially-hashable identity; Rust dedup mirrors
/// the existing `RPythonTyper::seen_reprs_must_call_setup` pattern.
pub struct MixLevelHelperAnnotator {
    /// RPython `self.rtyper = rtyper` (annlowlevel.py:129).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.policy = MixLevelAnnotatorPolicy(self)`
    /// (annlowlevel.py). Wrapped in `RefCell` so `finish_annotate`
    /// can borrow-swap it into `annotator.using_policy` without an
    /// owned clone per call.
    pub policy: RefCell<MixLevelAnnotatorPolicy>,
    /// RPython `self.pending = []` (annlowlevel.py). Queue of
    /// helper graphs awaiting `finish_annotate`.
    pub pending: RefCell<Vec<PendingHelper>>,
    /// RPython `self.delayedreprs = set()` (annlowlevel.py).
    /// Pointer-identity dedup via the caller; entries are
    /// `Arc<dyn Repr>` whose `set_setup_delayed(false)` is flipped by
    /// `finish_rtype`.
    pub delayedreprs: RefCell<Vec<Arc<dyn Repr>>>,
    /// RPython `self.delayedconsts = []` (annlowlevel.py).
    pub delayedconsts: RefCell<Vec<DelayedConst>>,
    /// RPython `self.delayedfuncs = []` (annlowlevel.py).
    pub delayedfuncs: RefCell<Vec<DelayedFunc>>,
    /// RPython `self.newgraphs = set()` (annlowlevel.py).
    /// Upstream populates this from the `original_graph_count`
    /// checkpoint inside `finish_annotate` / `finish_rtype`.
    pub newgraphs: RefCell<Vec<GraphRef>>,
}

impl MixLevelHelperAnnotator {
    /// RPython `MixLevelHelperAnnotator.__init__(self, rtyper)`
    /// (annlowlevel.py).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        // Build the policy against an incomplete self — upstream
        // passes `self` to `MixLevelAnnotatorPolicy(self)` at the same
        // construction site. The policy only reads
        // `annhelper.rtyper`, so fabricating it from a temporary
        // ref-only struct here mirrors the Python semantics without a
        // partially-initialised moved value.
        let ll = LowLevelAnnotatorPolicy::new(Some(rtyper));
        let policy = MixLevelAnnotatorPolicy { ll };
        MixLevelHelperAnnotator {
            rtyper: Rc::downgrade(rtyper),
            policy: RefCell::new(policy),
            pending: RefCell::new(Vec::new()),
            delayedreprs: RefCell::new(Vec::new()),
            delayedconsts: RefCell::new(Vec::new()),
            delayedfuncs: RefCell::new(Vec::new()),
            newgraphs: RefCell::new(Vec::new()),
        }
    }

    fn add_newgraph(&self, graph: &GraphRef) {
        let mut newgraphs = self.newgraphs.borrow_mut();
        if !newgraphs.iter().any(|existing| Rc::ptr_eq(existing, graph)) {
            newgraphs.push(graph.clone());
        }
    }

    /// RPython `MixLevelHelperAnnotator.getgraph(self, ll_function,
    /// args_s, s_result)` (annlowlevel.py).
    ///
    /// ```python
    /// def getgraph(self, ll_function, args_s, s_result):
    ///     ann = self.rtyper.annotator
    ///     with ann.using_policy(self.policy):
    ///         graph, args_s = ann.get_call_parameters(ll_function, args_s)
    ///     for v_arg, s_arg in zip(graph.getargs(), args_s):
    ///         ann.setbinding(v_arg, s_arg)
    ///     ann.setbinding(graph.getreturnvar(), s_result)
    ///     self.pending.append((ll_function, graph, args_s, s_result))
    ///     return graph
    /// ```
    ///
    /// Port pending.
    pub fn getgraph(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let policy = self.policy.borrow().clone();
        let (graph, args_s) = {
            let _guard = ann.using_policy(policy);
            ann.get_call_parameters(ll_function, args_s)
                .map_err(|e| TyperError::message(e.to_string()))?
        };
        for (v_arg, s_arg) in graph
            .graph
            .borrow()
            .getargs()
            .into_iter()
            .zip(args_s.iter())
        {
            if let crate::flowspace::model::Hlvalue::Variable(mut v_arg) = v_arg
                && let Some(s_arg) = s_arg.clone()
            {
                ann.setbinding(&mut v_arg, s_arg);
            }
        }
        if let crate::flowspace::model::Hlvalue::Variable(mut retvar) =
            graph.graph.borrow().getreturnvar()
        {
            ann.setbinding(&mut retvar, s_result.clone());
        }
        self.pending.borrow_mut().push(PendingHelper {
            ll_function: ll_function.clone(),
            graph: graph.clone(),
            args_s,
            s_result,
        });
        Ok(graph)
    }

    /// RPython `MixLevelHelperAnnotator.delayedfunction(self,
    /// ll_function, args_s, s_result, needtype=False)`
    /// (annlowlevel.py).
    ///
    /// Delegates to [`MixLevelHelperAnnotator::getgraph`] then
    /// [`MixLevelHelperAnnotator::graph2delayed`]; when `needtype` is
    /// true the returned delayed pointer is typed eagerly by
    /// consulting [`MixLevelHelperAnnotator::getdelayedrepr`] for
    /// each arg.
    ///
    /// Port pending.
    pub fn delayedfunction(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
        needtype: bool,
    ) -> Result<_ptr, TyperError> {
        let graph = self.getgraph(ll_function, args_s.clone(), s_result.clone())?;
        let functype = if needtype {
            let args = args_s
                .iter()
                .map(|s_arg| {
                    self.getdelayedrepr(s_arg, false)
                        .map(|r| r.lowleveltype().clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let result = self
                .getdelayedrepr(&s_result, false)?
                .lowleveltype()
                .clone();
            Some(LowLevelType::Func(Box::new(FuncType { args, result })))
        } else {
            None
        };
        self.graph2delayed(&graph, functype)
    }

    /// RPython `MixLevelHelperAnnotator.constfunc(self, ll_function,
    /// args_s, s_result)` (annlowlevel.py).
    ///
    /// ```python
    /// def constfunc(self, ll_function, args_s, s_result):
    ///     p = self.delayedfunction(ll_function, args_s, s_result)
    ///     return Constant(p, lltype.typeOf(p))
    /// ```
    ///
    /// Port pending. Returns a typed `Constant`-wrapping lltype
    /// pointer; the real return type once ported is most likely
    /// `(Hlvalue::Constant, LowLevelType)` so both the Python
    /// `Constant(p, typeOf(p))` halves survive the Rust lowering.
    pub fn constfunc(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
    ) -> Result<(_ptr, LowLevelType), TyperError> {
        let p = self.delayedfunction(ll_function, args_s, s_result, false)?;
        Ok((p.clone(), LowLevelType::Ptr(Box::new(lltype::typeOf(&p)))))
    }

    /// RPython `MixLevelHelperAnnotator.graph2delayed(self, graph,
    /// FUNCTYPE=None)` (annlowlevel.py).
    ///
    /// ```python
    /// def graph2delayed(self, graph, FUNCTYPE=None):
    ///     if FUNCTYPE is None:
    ///         FUNCTYPE = lltype.ForwardReference()
    ///     name = "delayed!%s" % (graph.name,)
    ///     delayedptr = lltype._ptr(lltype.Ptr(FUNCTYPE), name, solid=True)
    ///     self.delayedfuncs.append((delayedptr, graph))
    ///     return delayedptr
    /// ```
    ///
    /// Port pending.
    pub fn graph2delayed(
        &self,
        graph: &Rc<PyGraph>,
        functype: Option<LowLevelType>,
    ) -> Result<_ptr, TyperError> {
        let functype = functype.unwrap_or_else(|| LowLevelType::ForwardReference(Box::default()));
        let _name = format!("delayed!{}", graph.graph.borrow().name);
        let delayedptr = _ptr::new_with_solid(
            Ptr::from_container_type(functype).map_err(TyperError::message)?,
            Err(DelayedPointer),
            true,
        );
        self.delayedfuncs.borrow_mut().push(DelayedFunc {
            delayedptr: delayedptr.clone(),
            graph: graph.clone(),
        });
        Ok(delayedptr)
    }

    /// RPython `MixLevelHelperAnnotator.graph2const(self, graph)`
    /// (annlowlevel.py).
    ///
    /// ```python
    /// def graph2const(self, graph):
    ///     p = self.graph2delayed(graph)
    ///     return Constant(p, lltype.typeOf(p))
    /// ```
    ///
    /// Port pending.
    pub fn graph2const(&self, graph: &Rc<PyGraph>) -> Result<(_ptr, LowLevelType), TyperError> {
        let p = self.graph2delayed(graph, None)?;
        Ok((p.clone(), LowLevelType::Ptr(Box::new(lltype::typeOf(&p)))))
    }

    /// RPython `MixLevelHelperAnnotator.getdelayedrepr(self, s_value,
    /// check_never_seen=True)` (annlowlevel.py).
    ///
    /// ```python
    /// def getdelayedrepr(self, s_value, check_never_seen=True):
    ///     r = self.rtyper.getrepr(s_value)
    ///     if check_never_seen:
    ///         r.set_setup_delayed(True)
    ///         delayed = True
    ///     else:
    ///         delayed = r.set_setup_maybe_delayed()
    ///     if delayed:
    ///         self.delayedreprs.add(r)
    ///     return r
    /// ```
    ///
    /// Port pending.
    pub fn getdelayedrepr(
        &self,
        s_value: &SomeValue,
        check_never_seen: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let r = rtyper.getrepr(s_value)?;
        let delayed = if check_never_seen {
            r.set_setup_delayed(true);
            true
        } else {
            r.set_setup_maybe_delayed()
        };
        if delayed {
            let mut delayedreprs = self.delayedreprs.borrow_mut();
            if !delayedreprs
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &r))
            {
                delayedreprs.push(r.clone());
            }
        }
        Ok(r)
    }

    /// RPython `MixLevelHelperAnnotator.s_r_instanceof(self, cls,
    /// can_be_None=True, check_never_seen=True)` (annlowlevel.py:
    /// :197-202).
    ///
    /// ```python
    /// def s_r_instanceof(self, cls, can_be_None=True, check_never_seen=True):
    ///     classdesc = self.rtyper.annotator.bookkeeper.getdesc(cls)
    ///     classdef = classdesc.getuniqueclassdef()
    ///     s_instance = annmodel.SomeInstance(classdef, can_be_None)
    ///     r_instance = self.getdelayedrepr(s_instance, check_never_seen)
    ///     return s_instance, r_instance
    /// ```
    ///
    /// Port pending.
    pub fn s_r_instanceof(
        &self,
        cls: &HostObject,
        can_be_none: bool,
        check_never_seen: bool,
    ) -> Result<(SomeValue, Arc<dyn Repr>), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let classdesc = ann
            .bookkeeper
            .getdesc(cls)
            .map_err(|e| TyperError::message(e.to_string()))?
            .as_class()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MixLevelHelperAnnotator.s_r_instanceof: {:?} is not a class descriptor",
                    cls.qualname()
                ))
            })?;
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(&classdesc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let s_instance = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            can_be_none,
            Default::default(),
        ));
        let r_instance = self.getdelayedrepr(&s_instance, check_never_seen)?;
        Ok((s_instance, r_instance))
    }

    /// RPython `MixLevelHelperAnnotator.delayedconst(self, repr, obj)`
    /// (annlowlevel.py).
    ///
    /// ```python
    /// def delayedconst(self, repr, obj):
    ///     if repr.is_setup_delayed():
    ///         bk = self.rtyper.annotator.bookkeeper
    ///         bk.immutablevalue(obj)
    ///         delayedptr = lltype._ptr(repr.lowleveltype, "delayed!")
    ///         self.delayedconsts.append((delayedptr, repr, obj))
    ///         return delayedptr
    ///     else:
    ///         return repr.convert_const(obj)
    /// ```
    ///
    /// Port pending.
    pub fn delayedconst(
        &self,
        repr: &Arc<dyn Repr>,
        obj: &ConstValue,
    ) -> Result<Constant, TyperError> {
        if repr.is_setup_delayed() {
            let rtyper = self
                .rtyper
                .upgrade()
                .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
            let ann = rtyper
                .annotator
                .upgrade()
                .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
            ann.bookkeeper
                .immutablevalue(obj)
                .map_err(|e| TyperError::message(e.to_string()))?;
            let LowLevelType::Ptr(ptr_type) = repr.lowleveltype() else {
                return Err(TyperError::message(format!(
                    "MixLevelHelperAnnotator.delayedconst: expected Ptr lowleveltype, got {:?}",
                    repr.lowleveltype()
                )));
            };
            let delayedptr = _ptr::new((**ptr_type).clone(), Err(DelayedPointer));
            self.delayedconsts.borrow_mut().push(DelayedConst {
                delayedptr: delayedptr.clone(),
                repr: repr.clone(),
                obj: obj.clone(),
            });
            Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(delayedptr)),
                repr.lowleveltype().clone(),
            ))
        } else {
            repr.convert_const(obj)
        }
    }

    /// RPython `MixLevelHelperAnnotator.finish(self)`
    /// (annlowlevel.py).
    ///
    /// ```python
    /// def finish(self):
    ///     self.finish_annotate()
    ///     self.finish_rtype()
    /// ```
    pub fn finish(&self) -> Result<(), TyperError> {
        self.finish_annotate()?;
        self.finish_rtype()?;
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.finish_annotate(self)`
    /// (annlowlevel.py).
    ///
    /// Port pending. Upstream body:
    ///
    /// ```python
    /// def finish_annotate(self):
    ///     rtyper = self.rtyper
    ///     ann = rtyper.annotator
    ///     bk = ann.bookkeeper
    ///     translator = ann.translator
    ///     original_graph_count = len(translator.graphs)
    ///     with ann.using_policy(self.policy):
    ///         for ll_function, graph, args_s, s_result in self.pending:
    ///             ann.annotated[graph.returnblock] = graph
    ///             s_function = bk.immutablevalue(ll_function)
    ///             bk.emulate_pbc_call(graph, s_function, args_s)
    ///             self.newgraphs.add(graph)
    ///         ann.complete_helpers()
    ///     for ll_function, graph, args_s, s_result in self.pending:
    ///         s_real_result = ann.binding(graph.getreturnvar())
    ///         if s_real_result != s_result:
    ///             raise Exception("wrong annotation for the result of %r:\n"
    ///                             "originally specified: %r\n"
    ///                             " found by annotating: %r" %
    ///                             (graph, s_result, s_real_result))
    ///     del self.pending[:]
    ///     for graph in translator.graphs[original_graph_count:]:
    ///         self.newgraphs.add(graph)
    /// ```
    pub fn finish_annotate(&self) -> Result<(), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let bk = ann.bookkeeper.clone();
        let translator = ann.translator.clone();
        let original_graph_count = translator.graphs.borrow().len();
        let pending = self.pending.borrow().clone();
        {
            let _guard = ann.using_policy(self.policy.borrow().clone());
            for PendingHelper {
                ll_function,
                graph,
                args_s,
                s_result: _,
            } in &pending
            {
                let returnblock = graph.graph.borrow().returnblock.clone();
                let returnblock_key = BlockKey::of(&returnblock);
                ann.annotated
                    .borrow_mut()
                    .insert(returnblock_key.clone(), Some(graph.graph.clone()));
                ann.all_blocks
                    .borrow_mut()
                    .insert(returnblock_key, returnblock);
                let s_function = bk
                    .immutablevalue(&ConstValue::HostObject(ll_function.clone()))
                    .map_err(|e| TyperError::message(e.to_string()))?;
                // Post-specialize, args_s carries `Some(_)` for every
                // slot (specialize re-wraps after normalize_args).
                // emulate_pbc_call's `&[SomeValue]` shape stays as-is —
                // unwrap each cell here, surfacing any unannotated
                // residue as a fail-loud error matching upstream
                // `bookkeeper.emulate_pbc_call`'s downstream
                // AttributeError surface.
                let args_s_concrete: Vec<SomeValue> = args_s
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        s.clone().ok_or_else(|| {
                            TyperError::message(format!(
                                "MixLevelHelperAnnotator: args_s[{i}] is unannotated post-specialize"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                bk.emulate_pbc_call(
                    crate::annotator::bookkeeper::EmulatedPbcCallKey::Graph(GraphKey::of(
                        &graph.graph,
                    )),
                    &s_function,
                    &args_s_concrete,
                    &[],
                    None,
                )
                .map_err(|e| TyperError::message(e.to_string()))?;
                self.add_newgraph(&graph.graph);
            }
            ann.complete_helpers()
                .map_err(|e| TyperError::message(e.to_string()))?;
        }
        for PendingHelper {
            ll_function: _,
            graph,
            args_s: _,
            s_result,
        } in &pending
        {
            let s_real_result = ann.binding(&graph.graph.borrow().getreturnvar());
            if &s_real_result != s_result {
                return Err(TyperError::message(format!(
                    "wrong annotation for the result of {:?}:\noriginally specified: {:?}\n found by annotating: {:?}",
                    graph.graph.borrow().name,
                    s_result,
                    s_real_result,
                )));
            }
        }
        self.pending.borrow_mut().clear();
        let fresh_graphs: Vec<GraphRef> = translator
            .graphs
            .borrow()
            .iter()
            .skip(original_graph_count)
            .cloned()
            .collect();
        for graph in fresh_graphs {
            self.add_newgraph(&graph);
        }
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.finish_rtype(self)`
    /// (annlowlevel.py).
    ///
    /// Port pending. Upstream body:
    ///
    /// ```python
    /// def finish_rtype(self):
    ///     rtyper = self.rtyper
    ///     translator = rtyper.annotator.translator
    ///     original_graph_count = len(translator.graphs)
    ///     perform_normalizations(rtyper.annotator)
    ///     for r in self.delayedreprs:
    ///         r.set_setup_delayed(False)
    ///     rtyper.call_all_setups()
    ///     for p, repr, obj in self.delayedconsts:
    ///         p._become(repr.convert_const(obj))
    ///     rtyper.call_all_setups()
    ///     for p, graph in self.delayedfuncs:
    ///         self.newgraphs.add(graph)
    ///         real_p = rtyper.getcallable(graph)
    ///         REAL = lltype.typeOf(real_p).TO
    ///         FUNCTYPE = lltype.typeOf(p).TO
    ///         if isinstance(FUNCTYPE, lltype.ForwardReference):
    ///             FUNCTYPE.become(REAL)
    ///         assert FUNCTYPE == REAL
    ///         p._become(real_p)
    ///     rtyper.specialize_more_blocks()
    ///     self.delayedreprs.clear()
    ///     del self.delayedconsts[:]
    ///     del self.delayedfuncs[:]
    ///     for graph in translator.graphs[original_graph_count:]:
    ///         self.newgraphs.add(graph)
    /// ```
    pub fn finish_rtype(&self) -> Result<(), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let translator = ann.translator.clone();
        let original_graph_count = translator.graphs.borrow().len();
        super::normalizecalls::perform_normalizations(ann.as_ref())
            .map_err(|e| TyperError::message(e.to_string()))?;
        let delayedreprs = self.delayedreprs.borrow().clone();
        for r in &delayedreprs {
            r.set_setup_delayed(false);
        }
        rtyper.call_all_setups()?;
        let delayedconsts = self.delayedconsts.borrow().clone();
        for delayed in &delayedconsts {
            let real = delayed.repr.convert_const(&delayed.obj)?;
            let ConstValue::LLPtr(real_p) = &real.value else {
                return Err(TyperError::message(format!(
                    "MixLevelHelperAnnotator.finish_rtype: delayed const for {} did not lower to LLPtr",
                    delayed.repr.repr_string()
                )));
            };
            delayed.delayedptr._become(real_p);
        }
        rtyper.call_all_setups()?;
        let delayedfuncs = self.delayedfuncs.borrow().clone();
        for delayed in &delayedfuncs {
            self.add_newgraph(&delayed.graph.graph);
            let real_p = rtyper.getcallable(&delayed.graph)?;
            let real = lltype::typeOf(&real_p).TO.clone();
            let functype = lltype::typeOf(&delayed.delayedptr).TO.clone();
            if let lltype::PtrTarget::ForwardReference(forward_ref) = &functype {
                forward_ref
                    .r#become(real.clone().into())
                    .map_err(TyperError::message)?;
            }
            assert_eq!(functype, real);
            delayed.delayedptr._become(&real_p);
        }
        rtyper.specialize_more_blocks()?;
        self.delayedreprs.borrow_mut().clear();
        self.delayedconsts.borrow_mut().clear();
        self.delayedfuncs.borrow_mut().clear();
        let fresh_graphs: Vec<GraphRef> = translator
            .graphs
            .borrow()
            .iter()
            .skip(original_graph_count)
            .cloned()
            .collect();
        for graph in fresh_graphs {
            self.add_newgraph(&graph);
        }
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.backend_optimize(self,
    /// **flags)` (annlowlevel.py).
    ///
    /// ```python
    /// def backend_optimize(self, **flags):
    ///     from rpython.translator.backendopt.all import backend_optimizations
    ///     translator = self.rtyper.annotator.translator
    ///     newgraphs = list(self.newgraphs)
    ///     backend_optimizations(translator, newgraphs, secondary=True,
    ///                           inline_graph_from_anywhere=True, **flags)
    ///     self.newgraphs.clear()
    /// ```
    ///
    /// Unreachable in pyre. `backend_optimize` re-optimises the fresh helper
    /// graphs a GC transformer builds, and its only upstream callers are the
    /// translation-time GC passes (`rpython/memory/gctransform/`
    /// {boehm,transform,refcounting,framework}). pyre does not run that
    /// subsystem: it does not translate itself to C, so it needs no C-level GC
    /// insertion; it manages memory with the `majit-gc` runtime collector; and
    /// it inserts JIT GC operations with the backend `GcRewriterAssembler`
    /// (`majit-gc/rewrite.rs`, the `jit/backend/llsupport/rewrite.py` analog),
    /// exactly as upstream does for JIT traces rather than through the
    /// translation-time gctransform. The callee `backend_optimizations` is
    /// itself ported (`backendopt::all`) with a live caller (`driver`); this
    /// method stays fail-loud so any future real caller surfaces instead of
    /// silently running un-optimised helper graphs.
    pub fn backend_optimize(&self) -> Result<(), TyperError> {
        // Skeleton matching the upstream four-line body, kept so the shape is
        // visible at audit time. `backend_optimizations` is reachable code
        // (`backendopt::all`); this wrapper simply has no pyre caller — see the
        // doc above.

        // upstream: translator = self.rtyper.annotator.translator
        let _translator = self
            .rtyper
            .upgrade()
            .and_then(|r| r.annotator.upgrade())
            .map(|a| a.translator.clone());
        // upstream: newgraphs = list(self.newgraphs)
        let _newgraphs: Vec<_> = self.newgraphs.borrow().clone();
        // upstream: backend_optimizations(translator, newgraphs,
        //                                  secondary=True,
        //                                  inline_graph_from_anywhere=True,
        //                                  **flags)
        // Fail before reaching `newgraphs.clear()`: pyre never runs the
        // translation-time gctransform passes that call this, so reaching here
        // means an unexpected caller appeared and should surface loudly.
        Err(TyperError::message(
            "annlowlevel.py:277 MixLevelHelperAnnotator.backend_optimize is unreachable in pyre \
             (only translation-time gctransform passes call it; pyre uses the majit-gc runtime \
             collector and the backend GcRewriterAssembler)",
        ))
        // upstream: self.newgraphs.clear() — only runs after a
        // successful backend_optimizations return; the stub never
        // reaches it.  When the call lands, restore the clear-after
        // ordering.
    }
}

// annlowlevel.py — make_string_entries / hlstr / llstr

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringEntryType {
    Str,
    Unicode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringEntryDirection {
    Hl,
    Ll,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringEntryHelper {
    pub strtype: StringEntryType,
    pub direction: StringEntryDirection,
}

impl StringEntryHelper {
    pub fn call(&self, _arg: ConstValue) -> Result<ConstValue, TyperError> {
        Err(TyperError::message(format!(
            "annlowlevel.py:380 make_string_entries {:?}/{:?} helper port pending",
            self.strtype, self.direction
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HLStrEntry {
    pub helper: StringEntryHelper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LLStrEntry {
    pub helper: StringEntryHelper,
}

pub fn make_string_entries(strtype: StringEntryType) -> (StringEntryHelper, StringEntryHelper) {
    (
        StringEntryHelper {
            strtype,
            direction: StringEntryDirection::Hl,
        },
        StringEntryHelper {
            strtype,
            direction: StringEntryDirection::Ll,
        },
    )
}

#[allow(non_upper_case_globals)]
pub const hlstr: StringEntryHelper = StringEntryHelper {
    strtype: StringEntryType::Str,
    direction: StringEntryDirection::Hl,
};

#[allow(non_upper_case_globals)]
pub const llstr: StringEntryHelper = StringEntryHelper {
    strtype: StringEntryType::Str,
    direction: StringEntryDirection::Ll,
};

#[allow(non_upper_case_globals)]
pub const hlunicode: StringEntryHelper = StringEntryHelper {
    strtype: StringEntryType::Unicode,
    direction: StringEntryDirection::Hl,
};

#[allow(non_upper_case_globals)]
pub const llunicode: StringEntryHelper = StringEntryHelper {
    strtype: StringEntryType::Unicode,
    direction: StringEntryDirection::Ll,
};

// annlowlevel.py cast_object_to_ptr — pointer-cast helpers and extregistry entries

pub fn cast_object_to_ptr(_ptr: LowLevelType, _object: ConstValue) -> Result<Constant, TyperError> {
    Err(TyperError::message(
        "annlowlevel.py:453 cast_object_to_ptr port pending \
         (blocked on cast_pointer extregistry specialization)",
    ))
}

pub fn cast_instance_to_base_ptr(instance: ConstValue) -> Result<Constant, TyperError> {
    cast_object_to_ptr(super::rclass::OBJECTPTR.clone(), instance)
}

pub fn cast_instance_to_gcref(_instance: ConstValue) -> Result<Constant, TyperError> {
    Err(TyperError::message(
        "annlowlevel.py:482 cast_instance_to_gcref port pending \
         (blocked on lltype.cast_opaque_ptr lowering)",
    ))
}

pub fn cast_nongc_instance_to_base_ptr(instance: ConstValue) -> Result<Constant, TyperError> {
    cast_object_to_ptr(super::rclass::NONGCOBJECTPTR.clone(), instance)
}

pub fn cast_nongc_instance_to_adr(_instance: ConstValue) -> Result<Constant, TyperError> {
    Err(TyperError::message(
        "annlowlevel.py:492 cast_nongc_instance_to_adr port pending \
         (blocked on llmemory.cast_ptr_to_adr lowering)",
    ))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CastObjectToPtrEntry;

pub fn cast_base_ptr_to_instance(
    _class: HostObject,
    _ptr: ConstValue,
) -> Result<ConstValue, TyperError> {
    Err(TyperError::message(
        "annlowlevel.py:526 cast_base_ptr_to_instance port pending \
         (blocked on cast_pointer extregistry specialization)",
    ))
}

pub fn cast_base_ptr_to_nongc_instance(
    class: HostObject,
    ptr: ConstValue,
) -> Result<ConstValue, TyperError> {
    cast_base_ptr_to_instance(class, ptr)
}

pub fn cast_gcref_to_instance(
    class: HostObject,
    _ptr: ConstValue,
) -> Result<ConstValue, TyperError> {
    Err(TyperError::message(format!(
        "annlowlevel.py:540 cast_gcref_to_instance({}) port pending \
         (blocked on lltype.cast_opaque_ptr lowering)",
        class.qualname()
    )))
}

pub fn cast_adr_to_nongc_instance(
    class: HostObject,
    _ptr: ConstValue,
) -> Result<ConstValue, TyperError> {
    Err(TyperError::message(format!(
        "annlowlevel.py:546 cast_adr_to_nongc_instance({}) port pending \
         (blocked on llmemory.cast_adr_to_ptr lowering)",
        class.qualname()
    )))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CastBasePtrToInstanceEntry;

// Unit tests — lock in that every stub surface reports its upstream
// line number so follow-up body fills can trip on the exact error
// string.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeInteger, SomeObject, SomePBC};
    use crate::flowspace::model::{
        ConstValue, Constant, GraphFunc, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::rtyper::rmodel::ReprState;
    use crate::translator::rtyper::rtyper::LowLevelOpList;
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    #[derive(Debug)]
    struct TestRepr {
        state: ReprState,
        lowleveltype: LowLevelType,
    }

    impl TestRepr {
        fn new(lowleveltype: LowLevelType) -> Self {
            TestRepr {
                state: ReprState::new(),
                lowleveltype,
            }
        }
    }

    impl Repr for TestRepr {
        fn lowleveltype(&self) -> &LowLevelType {
            &self.lowleveltype
        }

        fn state(&self) -> &ReprState {
            &self.state
        }

        fn class_name(&self) -> &'static str {
            "TestRepr"
        }
    }

    fn make_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let annotator = RPythonAnnotator::new(None, None, None, false);
        let rtyper = annotator.translator.buildrtyper();
        rtyper.initialize_exceptiondata().unwrap();
        (annotator, rtyper)
    }

    fn compiled_host_function(src: &str) -> HostObject {
        let code = rp_compile(src, Mode::Exec, "<test>", Default::default())
            .expect("compile should succeed");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        let func = GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            vec![],
        );
        HostObject::new_user_function(func)
    }

    fn lowlevel_type_const_annotation(t: LowLevelType) -> SomeValue {
        let s = SomeObject {
            const_box: Some(Constant::new(ConstValue::LowLevelType(Box::new(t)))),
            ..SomeObject::default()
        };
        SomeValue::Object(s)
    }

    fn lowlevel_type_pbc_annotation(annotator: &RPythonAnnotator, t: LowLevelType) -> SomeValue {
        let f = compiled_host_function("def ll_marker():\n    return 1\n");
        let entry = annotator
            .bookkeeper
            .getdesc(&f)
            .expect("function desc for low-level type PBC");
        let mut pbc = SomePBC::new([entry], false);
        pbc.base.const_box = Some(Constant::new(ConstValue::LowLevelType(Box::new(t))));
        SomeValue::PBC(pbc)
    }

    fn call_dynamic_sigarg(sigarg: &SigArgType, inputcells: &[SomeValue]) -> SigArgType {
        let SigArgType::DynamicCallable(f) = sigarg else {
            panic!("expected DynamicCallable");
        };
        f(inputcells).expect("dynamic signature argument should expand")
    }

    fn already_annotation(sigarg: SigArgType) -> SomeValue {
        let SigArgType::Spec(AnnotationSpec::Already(s_value)) = sigarg else {
            panic!("expected AnnotationSpec::Already");
        };
        *s_value
    }

    #[test]
    fn llhelper_args_builds_low_level_function_pointer() {
        let f = compiled_host_function("def ll_id(x):\n    return x\n");
        let ptr = llhelper_args(&f, vec![LowLevelType::Signed], LowLevelType::Signed);
        let PtrTarget::Func(func_t) = &ptr._TYPE.TO else {
            panic!("llhelper_args must return Ptr(FuncType)");
        };
        assert_eq!(func_t.args, vec![LowLevelType::Signed]);
        assert_eq!(func_t.result, LowLevelType::Signed);
        let lltype::PtrObj::Strong(lltype::_ptr_obj::Func(func)) = &ptr._obj0 else {
            panic!("llhelper_args must allocate a function container");
        };
        assert_eq!(func._name, "ll_id");
        assert!(
            func._callable
                .as_deref()
                .is_some_and(|name| name.starts_with("ll_id#"))
        );
    }

    #[test]
    fn pseudo_high_level_callable_specialize_call_emits_direct_call() {
        let (_annotator, rtyper) = make_rtyper();
        let func_t = FuncType {
            args: vec![LowLevelType::Signed],
            result: LowLevelType::Signed,
        };
        let fnptr = lltype::functionptr(
            func_t,
            "ll_identity",
            None,
            Some("ll_identity#test".to_string()),
        );
        let instance = PseudoHighLevelCallable::new(
            fnptr.clone(),
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            SomeValue::Integer(SomeInteger::default()),
        );
        let v_arg = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(3),
            LowLevelType::Signed,
        ));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![v_arg.clone()],
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        let r_int = rtyper
            .getrepr(&SomeValue::Integer(SomeInteger::default()))
            .expect("int repr");
        hop.args_v.borrow_mut().push(v_arg);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::default()));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::default()));
        *hop.r_result.borrow_mut() = Some(r_int);

        let result = PseudoHighLevelCallableEntry
            .specialize_call(&instance, &hop)
            .expect("specialize_call")
            .expect("direct_call should produce a result");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("first direct_call arg should be function constant");
        };
        assert_eq!(
            c_func.concretetype,
            Some(LowLevelType::Ptr(Box::new(fnptr._TYPE)))
        );
        assert!(matches!(c_func.value, ConstValue::LLPtr(_)));
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn llhelper_compute_result_annotation_emulates_pbc_call_and_returns_someptr() {
        let (annotator, _rtyper) = make_rtyper();
        let f = compiled_host_function("def ll_id(x):\n    return x\n");
        let entry = annotator
            .bookkeeper
            .getdesc(&f)
            .expect("function desc for llhelper");
        let mut pbc = SomePBC::new([entry], false);
        pbc.base.const_box = Some(Constant::new(ConstValue::HostObject(f)));
        let s_callable = SomeValue::PBC(pbc);
        let f_type = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            }),
        };
        let s_f = lowlevel_type_const_annotation(LowLevelType::Ptr(Box::new(f_type.clone())));
        let _guard = annotator.bookkeeper.at_position(None);

        let result = LLHelperEntry
            .compute_result_annotation(&s_f, &s_callable)
            .expect("compute_result_annotation");
        let SomeValue::Ptr(s_ptr) = result else {
            panic!("llhelper should return SomePtr(F)");
        };
        assert_eq!(s_ptr.ll_ptrtype, f_type);
    }

    #[test]
    fn llhelper_specialize_call_nonconstant_callable_returns_inputarg() {
        let (_annotator, rtyper) = make_rtyper();
        let f_type = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            }),
        };
        let ptr_lltype = LowLevelType::Ptr(Box::new(f_type.clone()));
        let s_f = lowlevel_type_const_annotation(ptr_lltype.clone());
        let s_callable = SomeValue::Ptr(SomePtr::new(f_type));
        let v_f = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::LowLevelType(Box::new(ptr_lltype.clone())),
            LowLevelType::Void,
        ));
        let v_callable = Variable::new();
        v_callable.set_concretetype(Some(ptr_lltype.clone()));
        let v_callable_h = Hlvalue::Variable(v_callable.clone());
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![v_f.clone(), v_callable_h.clone()],
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops.clone());
        let r_callable: Arc<dyn Repr> = Arc::new(TestRepr::new(ptr_lltype));
        hop.args_v.borrow_mut().extend([v_f, v_callable_h]);
        hop.args_s.borrow_mut().extend([s_f, s_callable]);
        hop.args_r
            .borrow_mut()
            .extend([None, Some(r_callable.clone())]);

        let result = LLHelperEntry
            .specialize_call(&hop)
            .expect("specialize_call")
            .expect("non-constant callable should return inputarg");
        let Hlvalue::Variable(result_var) = result else {
            panic!("inputarg should preserve the callable variable");
        };
        assert_eq!(result_var, v_callable);
        assert!(llops.borrow()._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn placeholder_sigarg_dynamic_branches_expand_from_pointer_type() {
        let item = placeholder_sigarg("item").expect("item placeholder");
        let self_arg = placeholder_sigarg("self").expect("self placeholder");
        let ptr = Ptr {
            TO: PtrTarget::Array(lltype::Array::gc(LowLevelType::Signed)),
        };
        let s_self = SomeValue::Ptr(SomePtr::new(ptr.clone()));

        let expanded_item =
            already_annotation(call_dynamic_sigarg(&item, std::slice::from_ref(&s_self)));
        assert!(matches!(expanded_item, SomeValue::Integer(_)));
        let expanded_self = already_annotation(call_dynamic_sigarg(&self_arg, &[s_self]));
        let SomeValue::Ptr(expanded_self) = expanded_self else {
            panic!("self placeholder should return SomePtr");
        };
        assert_eq!(expanded_self.ll_ptrtype, ptr);
    }

    #[test]
    fn typemeth_placeholder_sigarg_dynamic_branches_expand_from_type_constant() {
        let (annotator, _rtyper) = make_rtyper();
        let list_type = LowLevelType::Array(Box::new(lltype::Array::gc(LowLevelType::Signed)));
        let s_type = lowlevel_type_pbc_annotation(&annotator, list_type.clone());

        let expanded_self = already_annotation(call_dynamic_sigarg(
            &typemeth_placeholder_sigarg("self").expect("self typemeth placeholder"),
            std::slice::from_ref(&s_type),
        ));
        let SomeValue::Ptr(s_ptr) = expanded_self else {
            panic!("typemeth self should return Ptr(TYPE)");
        };
        assert_eq!(
            s_ptr.ll_ptrtype,
            Ptr {
                TO: PtrTarget::Array(lltype::Array::gc(LowLevelType::Signed))
            }
        );

        let expanded_self_type = already_annotation(call_dynamic_sigarg(
            &typemeth_placeholder_sigarg("SELF").expect("SELF typemeth placeholder"),
            std::slice::from_ref(&s_type),
        ));
        assert_eq!(expanded_self_type, s_type);

        let expanded_item = already_annotation(call_dynamic_sigarg(
            &typemeth_placeholder_sigarg("item").expect("item typemeth placeholder"),
            &[lowlevel_type_pbc_annotation(&annotator, list_type)],
        ));
        assert!(matches!(expanded_item, SomeValue::Integer(_)));
    }

    #[test]
    fn adtinterface_call_reports_missing_constvalue_annenforceargs_carrier() {
        let interface = ADTInterface {
            sigtemplates: HashMap::new(),
            base: None,
            sigs: HashMap::new(),
        };
        let err = interface.__call__(HashMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing ConstValue::_annenforceargs_ carrier")
        );
    }

    #[test]
    fn mix_level_helper_annotator_pending_queues_start_empty() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        assert!(h.pending.borrow().is_empty());
        assert!(h.delayedreprs.borrow().is_empty());
        assert!(h.delayedconsts.borrow().is_empty());
        assert!(h.delayedfuncs.borrow().is_empty());
        assert!(h.newgraphs.borrow().is_empty());
    }

    #[test]
    fn mix_level_helper_annotator_finish_annotate_marks_returnblock_and_clears_pending() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");

        h.finish_annotate().expect("finish_annotate");
        assert!(h.pending.borrow().is_empty());
        let returnblock_key = BlockKey::of(&graph.graph.borrow().returnblock);
        let annotated = annotator.annotated.borrow();
        let Some(Some(done_graph)) = annotated.get(&returnblock_key) else {
            panic!("return block should be marked annotated");
        };
        assert!(std::rc::Rc::ptr_eq(done_graph, &graph.graph));
        assert!(
            h.newgraphs
                .borrow()
                .iter()
                .any(|newgraph| std::rc::Rc::ptr_eq(newgraph, &graph.graph))
        );
    }

    #[test]
    fn mix_level_helper_annotator_finish_rtype_resolves_delayed_function_and_clears_queues() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");
        let delayed = h.graph2delayed(&graph, None).expect("delayed pointer");

        h.finish_annotate().expect("finish_annotate");
        h.finish_rtype().expect("finish_rtype");

        assert!(h.delayedreprs.borrow().is_empty());
        assert!(h.delayedconsts.borrow().is_empty());
        assert!(h.delayedfuncs.borrow().is_empty());
        let lltype::PtrTarget::ForwardReference(forward_ref) = &lltype::typeOf(&delayed).TO else {
            panic!("expected delayed function type to start as ForwardReference");
        };
        assert!(forward_ref.resolved().is_some());
        let lltype::_ptr_obj::Func(funcobj) = delayed._obj().expect("resolved delayed ptr") else {
            panic!("resolved delayed ptr should expose a func object");
        };
        assert_eq!(
            funcobj.graph,
            Some(crate::flowspace::model::GraphKey::of(&graph.graph).as_usize())
        );
    }

    #[test]
    fn mix_level_helper_annotator_getgraph_populates_pending_queue() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("getgraph should return a helper graph");
        assert!(graph.graph.borrow().name.starts_with("ll_identity"));
        assert_eq!(h.pending.borrow().len(), 1);
        let returnvar = graph.graph.borrow().getreturnvar();
        assert!(matches!(
            annotator.annotation(&returnvar),
            Some(SomeValue::Integer(_))
        ));
    }

    #[test]
    fn mix_level_helper_annotator_graph2delayed_enqueues_delayed_func() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");
        let delayed = h.graph2delayed(&graph, None).expect("delayed ptr");
        assert_eq!(h.delayedfuncs.borrow().len(), 1);
        assert!(matches!(
            lltype::typeOf(&delayed).TO,
            lltype::PtrTarget::ForwardReference(_)
        ));
    }

    #[test]
    fn mix_level_helper_annotator_getdelayedrepr_marks_repr_delayed_and_dedups() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let classdef = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();
        let s_value = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            false,
            Default::default(),
        ));

        let r1 = h
            .getdelayedrepr(&s_value, true)
            .expect("first delayed repr");
        assert!(r1.is_setup_delayed());
        assert_eq!(h.delayedreprs.borrow().len(), 1);

        let r2 = h
            .getdelayedrepr(&s_value, false)
            .expect("second delayed repr");
        assert!(std::sync::Arc::ptr_eq(&r1, &r2));
        assert_eq!(h.delayedreprs.borrow().len(), 1);
    }

    #[test]
    fn mix_level_helper_annotator_s_r_instanceof_builds_someinstance_and_repr() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Target", vec![]);
        let expected = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();

        let (s_instance, r_instance) = h
            .s_r_instanceof(&cls, true, true)
            .expect("s_r_instanceof should succeed");
        let SomeValue::Instance(s_instance) = s_instance else {
            panic!("expected SomeInstance");
        };
        assert!(s_instance.can_be_none);
        let actual = s_instance.classdef.expect("classdef");
        assert!(std::rc::Rc::ptr_eq(&actual, &expected));
        assert!(r_instance.is_setup_delayed());
        assert_eq!(h.delayedreprs.borrow().len(), 1);
    }

    #[test]
    fn mix_level_helper_annotator_delayedconst_enqueues_pointer_for_delayed_repr() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Delayed", vec![]);
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let classdef = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();
        let s_value = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            false,
            Default::default(),
        ));
        let repr = h.getdelayedrepr(&s_value, true).expect("delayed repr");

        let constant = h
            .delayedconst(&repr, &ConstValue::HostObject(inst.clone()))
            .expect("delayed const");
        assert_eq!(h.delayedconsts.borrow().len(), 1);
        let delayedconsts = h.delayedconsts.borrow();
        assert!(matches!(
            &delayedconsts[0].obj,
            ConstValue::HostObject(host) if host == &inst
        ));
        assert_eq!(constant.concretetype, Some(repr.lowleveltype().clone()));
        let ConstValue::LLPtr(ptr) = &constant.value else {
            panic!("expected LLPtr constant");
        };
        assert_eq!(ptr._TYPE, delayedconsts[0].delayedptr._TYPE);
    }

    #[test]
    fn mix_level_helper_annotator_delayedconst_converts_immediately_for_ready_repr() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let s_value = SomeValue::Integer(SomeInteger::default());
        let repr = rtyper.getrepr(&s_value).expect("int repr");
        let expected = repr
            .convert_const(&ConstValue::Int(7))
            .expect("expected const");

        let actual = h
            .delayedconst(&repr, &ConstValue::Int(7))
            .expect("immediate const");
        assert_eq!(actual, expected);
        assert!(h.delayedconsts.borrow().is_empty());
    }

    #[test]
    fn annotate_lowlevel_helper_free_function_builds_helper_graph() {
        let (annotator, _rtyper) = make_rtyper();
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = annotate_lowlevel_helper(
            &annotator,
            &ll_function,
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
        )
        .expect("annotate_lowlevel_helper should annotate the helper");
        assert!(graph.graph.borrow().name.starts_with("ll_identity"));
        let returnvar = graph.graph.borrow().getreturnvar();
        assert!(matches!(
            annotator.annotation(&returnvar),
            Some(SomeValue::Integer(_))
        ));
        assert!(annotator.added_blocks.borrow().is_none());
    }

    #[test]
    fn make_string_entries_returns_hl_and_ll_helpers_for_each_string_type() {
        let (hl, ll) = make_string_entries(StringEntryType::Str);
        assert_eq!(hl, hlstr);
        assert_eq!(ll, llstr);
        let (hl, ll) = make_string_entries(StringEntryType::Unicode);
        assert_eq!(hl, hlunicode);
        assert_eq!(ll, llunicode);
        assert!(
            hl.call(ConstValue::None)
                .unwrap_err()
                .to_string()
                .contains("annlowlevel.py:380 make_string_entries")
        );
    }

    #[test]
    fn pointer_cast_parity_surface_reports_deferred_lowering() {
        let cls = HostObject::new_class("pkg.CastTarget", vec![]);
        assert!(
            cast_instance_to_base_ptr(ConstValue::None)
                .unwrap_err()
                .to_string()
                .contains("annlowlevel.py:453 cast_object_to_ptr")
        );
        assert!(
            cast_base_ptr_to_nongc_instance(cls.clone(), ConstValue::None)
                .unwrap_err()
                .to_string()
                .contains("annlowlevel.py:526 cast_base_ptr_to_instance")
        );
        assert!(
            cast_gcref_to_instance(cls.clone(), ConstValue::None)
                .unwrap_err()
                .to_string()
                .contains("annlowlevel.py:540 cast_gcref_to_instance")
        );
        assert!(
            cast_adr_to_nongc_instance(cls, ConstValue::None)
                .unwrap_err()
                .to_string()
                .contains("annlowlevel.py:546 cast_adr_to_nongc_instance")
        );
        let _entry = CastObjectToPtrEntry;
        let _entry = CastBasePtrToInstanceEntry;
    }
}
