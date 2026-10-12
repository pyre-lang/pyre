//! Port of `rpython/rtyper/extregistry.py`.
//!
//! Upstream provides two parallel registries that
//! `Bookkeeper.immutablevalue(x)` (`bookkeeper.py`) and
//! `signature.annotationoftype(t)` (`signature.py:98-100`) consult
//! before falling back to the generic callable / instance branches:
//!
//! * **Value-level** — `EXT_REGISTRY_BY_VALUE` keyed on the Python
//!   value itself. Driven by `class Entry(ExtRegistryEntry): _about_ =
//!   <value>` registrations (extregistry.py). [`is_registered`] /
//!   [`lookup`] expose this path; the bookkeeper consumes it.
//! * **Type-level** — `EXT_REGISTRY_BY_TYPE` keyed on a Python type
//!   object. Driven by `class Entry(ExtRegistryEntry): _type_ = <type>`
//!   registrations (extregistry.py). [`is_registered_type`] /
//!   [`lookup_type`] expose this path; the signature module consumes
//!   it.
//!
//! ## Rust language adaptation (parity rule #1)
//!
//! Upstream populates both registries lazily through the
//! `AutoRegisteringType` metaclass — every `class Entry(...)` triggers
//! `__init__` which inserts `cls` into the dict keyed on the
//! `_about_` / `_type_` slot. Rust has no metaclass; the
//! [`ExtRegistryEntry`] enum below carries one variant per upstream
//! `_about_` / `_type_` registration, and [`is_registered`] /
//! [`lookup`] match on the carrier shape directly. Adding a new
//! upstream entry means adding a variant + match arm here.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::model::{AnnotatorError, SomeBuiltin, SomeValue, s_none};
use crate::flowspace::model::{ConstValue, Hlvalue, HostObject};
use crate::rlib::jit::{
    JitDriverMeta, JitMarkerKind, ext_enter_leave_marker_compute_result_annotation,
};

use super::error::TyperError;
use super::lltypesystem::lltype;
use super::rbuiltin::BuiltinTyperFn;
use super::rmodel::RTypeResult;
use super::rtyper::{GenopResult, HighLevelOp};

/// One arm per upstream `class Entry(ExtRegistryEntry)` subclass that
/// the Rust port currently consults.
#[derive(Clone, Debug)]
pub enum ExtRegistryEntry {
    /// Upstream `lltype.py _ptrEntry`:
    ///
    /// ```python
    /// class _ptrEntry(ExtRegistryEntry):
    ///     _type_ = _ptr
    ///     def compute_annotation(self):
    ///         from rpython.rtyper.llannotation import SomePtr
    ///         return SomePtr(typeOf(self.instance))
    /// ```
    ///
    /// The `Option<_ptr>` holds the value-level instance when the
    /// entry was constructed via [`lookup`]; `None` when constructed
    /// via [`lookup_type`] (matching upstream `Entry(tp)` /
    /// `Entry(type=tp, instance=None)` at extregistry.py).
    /// Upstream `_ptrEntry.compute_annotation` reads `self.instance`,
    /// so the type-level path raises `AssertionError` upstream — the
    /// Rust [`Self::compute_annotation`] surfaces the same failure as
    /// an [`AnnotatorError`].
    Ptr(Option<lltype::_ptr>),
    /// Generic HostObject value-level entry mirroring the base
    /// `ExtRegistryEntry.compute_annotation` implementation:
    /// construct `SomeBuiltin(analyser, methodname=...)` for the
    /// registered host callable.
    HostValueBuiltin {
        instance: HostObject,
        analyser_name: String,
        methodname: Option<String>,
    },
    /// HostObject type-level entry whose subclass override returns a
    /// fixed annotation. This mirrors `_type_ = T` entries that define
    /// `compute_annotation` instead of using the base builtin-function
    /// implementation.
    HostTypeAnnotation {
        type_obj: HostObject,
        instance: Option<HostObject>,
        annotation: RegisteredAnnotation,
    },
    /// Upstream `rpython/rtyper/extfunc.py ExtFuncEntry`
    /// `class ExtFuncEntry(ExtRegistryEntry)`.
    ExternalFunction(super::extfunc::ExtFuncEntry),
    /// Upstream `rpython/rlib/jit.py ExtEnterLeaveMarker`:
    ///
    /// ```python
    /// class ExtEnterLeaveMarker(ExtRegistryEntry):
    ///     # _about_ assigned per-driver in
    ///     # JitDriver._make_extregistryentries to
    ///     # (driver.jit_merge_point, driver.can_enter_jit).
    /// ```
    ///
    /// Upstream registers two bound methods to a single `_about_`
    /// tuple, dispatched inside `compute_result_annotation` via
    /// `self.instance.__name__` (jit.py). Rust has no native bound
    /// methods, so [`JitMarkerKind`] discriminates which member of
    /// the upstream tuple this entry stands for. The driver metadata
    /// (`greens` / `reds` / hooks) is reachable via `meta` so
    /// `compute_result_annotation` can validate keyword names and
    /// drive `annotate_hooks` without a side registry.
    EnterLeaveMarker {
        meta: Arc<JitDriverMeta>,
        marker_kind: JitMarkerKind,
    },
    /// Upstream `rpython/rlib/jit.py ExtLoopHeader`:
    ///
    /// ```python
    /// class ExtLoopHeader(ExtRegistryEntry):
    ///     # _about_ = self.loop_header (bound method)
    /// ```
    LoopHeader { meta: Arc<JitDriverMeta> },
    /// Upstream `rpython/rlib/rarithmetic.py ForTypeEntry`:
    ///
    /// ```python
    /// class ForTypeEntry(extregistry.ExtRegistryEntry):
    ///     _about_ = int_type
    ///     def compute_result_annotation(self, *args_s, **kwds_s):
    ///         return SomeInteger(knowntype=int_type)
    ///     def specialize_call(self, hop):
    ///         v_result, = hop.inputargs(hop.r_result.lowleveltype)
    ///         hop.exception_cannot_occur()
    ///         return v_result
    /// ```
    ///
    /// Matches the class object created by `build_int` (rarithmetic.py:
    /// 546-600).  `instance` carries the class HostObject so
    /// `compute_result_annotation` can store `knowntype=instance` per
    /// upstream; signedness is derived from the knowntype itself inside
    /// `SomeInteger::new_with_knowntype` (`unsigned = knowntype(-1) > 0`
    /// per model.py:222), so no separate `unsigned` carrier is needed
    /// here — that would duplicate state RPython does not.
    ForType { instance: HostObject },
    /// Upstream `rpython/rlib/nonconst.py EntryNonConstant`:
    ///
    /// ```python
    /// class EntryNonConstant(ExtRegistryEntry):
    ///     _about_ = NonConstant
    ///
    ///     def compute_result_annotation(self, s_arg):
    ///         return not_const(s_arg)
    ///
    ///     def specialize_call(self, hop):
    ///         hop.exception_cannot_occur()
    ///         return hop.inputarg(hop.r_result, arg=0)
    /// ```
    NonConstant,
    /// Upstream `rpython/rlib/jit.py Entry`:
    ///
    /// ```python
    /// class Entry(ExtRegistryEntry):
    ///     _about_ = we_are_jitted
    ///     def compute_result_annotation(self):
    ///         return annmodel.SomeInteger(nonneg=True)
    ///     def specialize_call(self, hop):
    ///         hop.exception_cannot_occur()
    ///         return hop.inputconst(lltype.Signed, _we_are_jitted)
    /// ```
    ///
    /// The entry is a value-level singleton keyed on the
    /// `majit_metainterp.jit.we_are_jitted` host callable. It carries
    /// no per-instance state (the `_about_` value is the function
    /// itself), so it is a fieldless variant.
    ///
    /// The annotation half is served by the
    /// `majit_metainterp.jit.we_are_jitted` `BUILTIN_ANALYZERS` entry
    /// (`annotator/builtin.rs`, `majit_metainterp_bool_flag` →
    /// `SomeBool`), which `Bookkeeper.immutablevalue_hostobject`
    /// resolves and returns *before* the `extregistry.is_registered`
    /// fall-through (bookkeeper.py). So this entry only ever
    /// fires on the rtyper's `findbltintyper` → `specialize_call`
    /// path; its `compute_annotation` is unreachable in practice and
    /// fails closed to surface a bypassed early-return loudly.
    WeAreJitted,
    /// Upstream `rpython/rlib/jit.py ConditionalCallEntry`:
    ///
    /// ```python
    /// class ConditionalCallEntry(ExtRegistryEntry):
    ///     _about_ = _jit_conditional_call, _jit_conditional_call_value
    /// ```
    ///
    /// One class covers both helpers. `is_value` is
    /// `self.instance == _jit_conditional_call_value`.
    ConditionalCall { is_value: bool },
    /// Pyre's explicit source spelling of the force inserted by
    /// `VirtualizableInstanceRepr.hook_access_field`
    /// (`rpython/rtyper/rvirtualizable.py`).  Upstream inserts the
    /// `jit_force_virtualizable` LLOp while rtyping and never annotates a
    /// force-helper body.  Rust needs a callable marker in the source, so
    /// its translation is registered here and specializes to that same
    /// LLOp instead of recursively lifting `executioncontext::force_frame`.
    JitForceVirtualizable,
    /// `BigInt::from(i64) -> BigInt` — the `From<i64>` impl that boxes a
    /// machine int into the foreign opaque `BigInt` (Python `long`) on the
    /// cold `int_add`/`int_sub`/… overflow→long arm.  Modeled as a
    /// residual external, mirroring `register_external` for a foreign C
    /// helper (extfunc.py): the JIT never traces into the arbitrary-
    /// precision body, it emits a `direct_call` whose result is the opaque
    /// classdef-less GcRef (`Ptr(OBJECT)`), not a scalar.
    ///
    /// `From::from` has no `self` receiver, so the front-end lowers it as a
    /// `simple_call` against the `BigInt.from` host callable (not a
    /// `CallTarget::FunctionPath` the `CallRegistry` callee-stub path
    /// handles).  The annotation half is served by the `BigInt.from`
    /// `BUILTIN_ANALYZERS` entry (`annotator/builtin.rs`, `bigint_from` →
    /// the classdef-less `SomeInstance` shell), which
    /// `Bookkeeper.immutablevalue_hostobject` resolves and returns *before*
    /// the `extregistry.is_registered` fall-through (bookkeeper.py).
    /// So this entry only ever fires on the rtyper's `findbltintyper` →
    /// `specialize_call` path (the same dual-registration shape as
    /// [`Self::WeAreJitted`]); its `compute_annotation` is unreachable in
    /// practice and returns the same opaque shell to stay a faithful port.
    BigIntFrom,
    /// `longlong2float.float2longlong(f64) -> i64` — RPython
    /// `Float2LongLongEntry(ExtRegistryEntry)` (`rlib/longlong2float.py`).
    /// Reinterprets the f64 bit pattern as a signed longlong via the
    /// `convert_float_bytes_to_longlong` llop.  Same dual-registration shape as
    /// [`Self::BigIntFrom`]: the annotation half is served by the
    /// `longlong2float.float2longlong` `BUILTIN_ANALYZERS` entry
    /// (`float2longlong_analyzer` → `SomeInteger(r_longlong)`); this entry fires on
    /// the rtyper's `findbltintyper` → `specialize_call` path.
    Float2LongLong,
    /// `longlong2float.longlong2float(i64) -> f64` — RPython
    /// `LongLong2FloatEntry`, the inverse bit reinterpretation.
    LongLong2Float,
}

/// Small Send-able annotation payload for static HostObject type
/// registry entries. Add variants as concrete upstream `_type_`
/// entries land.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegisteredAnnotation {
    Bool,
    Int,
    Str,
}

/// RPython `class FlexibleWeakDict(UserDict.DictMixin)` (`extregistry.py`).
///
/// Upstream stores hashable non-weakrefable keys in `_regdict`,
/// weakrefable keys in `_weakdict`, and unhashable keys wrapped in
/// `Hashable` in `_iddict`. Rust registry keys in this module are
/// already hashable carriers, so this helper exposes the same three
/// stores and routes ordinary insert/get/remove operations through
/// `_regdict`; call sites that need true weak-key behavior should keep
/// using explicit `Weak` carriers as the surrounding port already does.
#[derive(Clone, Debug, Default)]
pub struct FlexibleWeakDict<K, V> {
    pub _regdict: HashMap<K, V>,
    pub _weakdict: HashMap<K, V>,
    pub _iddict: HashMap<K, V>,
}

impl<K, V> FlexibleWeakDict<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    pub fn new() -> Self {
        FlexibleWeakDict {
            _regdict: HashMap::new(),
            _weakdict: HashMap::new(),
            _iddict: HashMap::new(),
        }
    }

    pub fn __setitem__(&mut self, key: K, value: V) {
        self._regdict.insert(key, value);
    }

    pub fn __getitem__(&self, key: &K) -> Option<&V> {
        self._regdict
            .get(key)
            .or_else(|| self._weakdict.get(key))
            .or_else(|| self._iddict.get(key))
    }

    pub fn __delitem__(&mut self, key: &K) -> Option<V> {
        self._regdict
            .remove(key)
            .or_else(|| self._weakdict.remove(key))
            .or_else(|| self._iddict.remove(key))
    }

    pub fn keys(&self) -> Vec<K> {
        self._regdict
            .keys()
            .chain(self._weakdict.keys())
            .chain(self._iddict.keys())
            .cloned()
            .collect()
    }
}

/// Cache-key identity for `SomeBuiltin.rtyper_makekey` after the
/// upstream `extregistry.lookup(const)` remap.
///
/// RPython `ExtRegistryEntry.__hash__` / `__eq__`
/// (extregistry.py) key on `(self.__class__, self.type,
/// self.instance)`. Rust has one enum variant per entry subclass, so
/// the variant tag supplies `self.__class__`; these fields carry the
/// type/instance identities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExtRegistryEntryKey {
    Ptr {
        instance_identity: Option<u64>,
    },
    HostValueBuiltin {
        instance_identity: usize,
    },
    HostTypeAnnotation {
        type_identity: usize,
        instance_identity: Option<usize>,
    },
    ExternalFunction {
        function_identity: usize,
    },
    EnterLeaveMarker {
        driver_identity: usize,
        marker_kind: JitMarkerKind,
    },
    LoopHeader {
        driver_identity: usize,
    },
    ForType {
        instance_identity: usize,
    },
    /// Singleton key for the `we_are_jitted` `_about_` entry
    /// (rlib/jit.py). No per-instance identity — the variant tag
    /// alone supplies `self.__class__` and there is no `self.instance`.
    WeAreJitted,
    /// `ConditionalCallEntry` keyed on `_jit_conditional_call` vs
    /// `_jit_conditional_call_value` (`self.instance`).
    ConditionalCall {
        is_value: bool,
    },
    JitForceVirtualizable,
    /// Singleton key for the `BigInt.from` residual-external entry. No
    /// per-instance identity — the variant tag alone supplies
    /// `self.__class__` and there is no `self.instance`.
    BigIntFrom,
    Float2LongLong,
    LongLong2Float,
    /// Singleton key for the `NonConstant` `_about_` entry
    /// (rlib/nonconst.py). No per-instance identity.
    NonConstant,
}

impl ExtRegistryEntry {
    pub fn makekey(&self) -> ExtRegistryEntryKey {
        match self {
            ExtRegistryEntry::Ptr(instance) => ExtRegistryEntryKey::Ptr {
                instance_identity: instance.as_ref().map(lltype::_ptr::_hashable_identity),
            },
            ExtRegistryEntry::HostValueBuiltin { instance, .. } => {
                ExtRegistryEntryKey::HostValueBuiltin {
                    instance_identity: instance.identity_id(),
                }
            }
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj, instance, ..
            } => ExtRegistryEntryKey::HostTypeAnnotation {
                type_identity: type_obj.identity_id(),
                instance_identity: instance.as_ref().map(HostObject::identity_id),
            },
            ExtRegistryEntry::ExternalFunction(entry) => ExtRegistryEntryKey::ExternalFunction {
                function_identity: entry.function.identity_id(),
            },
            ExtRegistryEntry::EnterLeaveMarker { meta, marker_kind } => {
                ExtRegistryEntryKey::EnterLeaveMarker {
                    driver_identity: meta.id.identity_id(),
                    marker_kind: *marker_kind,
                }
            }
            ExtRegistryEntry::LoopHeader { meta } => ExtRegistryEntryKey::LoopHeader {
                driver_identity: meta.id.identity_id(),
            },
            ExtRegistryEntry::ForType { instance } => ExtRegistryEntryKey::ForType {
                instance_identity: instance.identity_id(),
            },
            ExtRegistryEntry::WeAreJitted => ExtRegistryEntryKey::WeAreJitted,
            ExtRegistryEntry::ConditionalCall { is_value } => {
                ExtRegistryEntryKey::ConditionalCall {
                    is_value: *is_value,
                }
            }
            ExtRegistryEntry::JitForceVirtualizable => ExtRegistryEntryKey::JitForceVirtualizable,
            ExtRegistryEntry::BigIntFrom => ExtRegistryEntryKey::BigIntFrom,
            ExtRegistryEntry::Float2LongLong => ExtRegistryEntryKey::Float2LongLong,
            ExtRegistryEntry::LongLong2Float => ExtRegistryEntryKey::LongLong2Float,
            ExtRegistryEntry::NonConstant => ExtRegistryEntryKey::NonConstant,
        }
    }

    /// RPython `ExtRegistryEntry.compute_annotation_bk(self, bk)`
    /// (extregistry.py). Stashes the bookkeeper on `self` and
    /// dispatches to `compute_annotation`. The Rust port forwards the
    /// bookkeeper into `compute_annotation` directly via the same `&Rc`
    /// — no per-entry mutable state needed.
    pub fn compute_annotation_bk(
        &self,
        _bookkeeper: &Rc<Bookkeeper>,
    ) -> Result<SomeValue, AnnotatorError> {
        self.compute_annotation()
    }

    /// RPython `ExtRegistryEntry.compute_annotation(self)`
    /// (extregistry.py). Subclass override per variant.
    pub fn compute_annotation(&self) -> Result<SomeValue, AnnotatorError> {
        match self {
            // upstream lltype.py:1517-1518:
            //     def compute_annotation(self):
            //         from rpython.rtyper.llannotation import SomePtr
            //         return SomePtr(typeOf(self.instance))
            ExtRegistryEntry::Ptr(Some(instance)) => Ok(super::llannotation::lltype_to_annotation(
                lltype::typeOf(instance),
            )),
            // upstream `_ptrEntry.compute_annotation` reads
            // `self.instance`; with `instance=None` (type-level path)
            // it raises AttributeError. Surface the same failure here.
            ExtRegistryEntry::Ptr(None) => Err(AnnotatorError::new(
                "ExtRegistryEntry::Ptr.compute_annotation: type-level lookup has no instance \
                 (upstream _ptrEntry.compute_annotation requires self.instance)",
            )),
            ExtRegistryEntry::HostValueBuiltin {
                instance,
                analyser_name,
                methodname,
            } => {
                let mut result = SomeBuiltin::new(
                    analyser_name.clone(),
                    None,
                    methodname
                        .clone()
                        .or_else(|| Some(instance.qualname().to_string())),
                );
                result.base.const_box = Some(crate::flowspace::model::Constant::new(
                    ConstValue::HostObject(instance.clone()),
                ));
                Ok(SomeValue::Builtin(result))
            }
            ExtRegistryEntry::HostTypeAnnotation { annotation, .. } => match annotation {
                RegisteredAnnotation::Bool => Ok(SomeValue::Bool(Default::default())),
                RegisteredAnnotation::Int => Ok(SomeValue::Integer(Default::default())),
                RegisteredAnnotation::Str => Ok(SomeValue::String(Default::default())),
            },
            ExtRegistryEntry::ExternalFunction(entry) => {
                Ok(SomeValue::ExternalFunction(entry.compute_annotation(false)))
            }
            // upstream extregistry.py compute_annotation base implementation —
            // marker subclasses do not override `compute_annotation`,
            // so the base returns
            // `SomeBuiltin(self.compute_result_annotation,
            //              methodname=getattr(func, '__name__', None))`.
            // The `SomeBuiltin.call` dispatcher (unaryop.py)
            // then drives the bound-method `compute_result_annotation`
            // — Rust mirrors that by special-casing the
            // `SomeValue::Builtin` arm in `model.rs::SomeValue::call`:
            // when the builtin's `const_box` resolves to a marker
            // entry through `extregistry::lookup`, the dispatcher calls
            // [`Self::compute_annotation_with_kwds`]. The
            // `analyser_name` carried below is diagnostic-only — it is
            // never consulted via `BUILTIN_ANALYZERS`.
            ExtRegistryEntry::EnterLeaveMarker { marker_kind, .. } => {
                Ok(SomeValue::Builtin(SomeBuiltin::new(
                    marker_kind.analyser_name(),
                    None,
                    Some(marker_kind.upstream_method_name().to_string()),
                )))
            }
            ExtRegistryEntry::LoopHeader { .. } => Ok(SomeValue::Builtin(SomeBuiltin::new(
                crate::rlib::jit::LOOP_HEADER_ANALYSER_NAME,
                None,
                Some(crate::rlib::jit::LOOP_HEADER_METHOD_NAME.to_string()),
            ))),
            // upstream extregistry.py compute_annotation base implementation —
            // `ForTypeEntry` inherits `compute_annotation` returning
            // `SomeBuiltin(self.compute_result_annotation,
            //              methodname=getattr(self.instance, '__name__', None))`.
            // `__name__` is the short class name (e.g. "r_uint"), not
            // the dotted qualified path; `HostObject::simple_name` is
            // the canonical short-name accessor (`flowspace/model.rs:
            // 237-242`).  The actual `compute_result_annotation` runs
            // at simple_call time via
            // [`Self::compute_annotation_with_kwds`].
            ExtRegistryEntry::ForType { instance, .. } => {
                let mut result = SomeBuiltin::new(
                    instance.qualname().to_string(),
                    None,
                    Some(instance.simple_name().to_string()),
                );
                result.base.const_box = Some(crate::flowspace::model::Constant::new(
                    ConstValue::HostObject(instance.clone()),
                ));
                Ok(SomeValue::Builtin(result))
            }
            // RPython's `we_are_jitted` ExtRegistryEntry annotates the
            // call `SomeInteger(nonneg=True)` (rlib/jit.py); pyre's
            // `we_are_jitted() -> bool` makes the faithful annotation
            // `SomeBool` — the same one the
            // `majit_metainterp.jit.we_are_jitted` `BUILTIN_ANALYZERS`
            // entry (`majit_metainterp_bool_flag`) serves on the normal
            // `immutablevalue_hostobject` path, which returns before the
            // `extregistry.is_registered` fall-through
            // (bookkeeper.py).  Returning it here keeps the
            // entry a faithful port for any path that consults it.
            ExtRegistryEntry::WeAreJitted => Ok(SomeValue::Bool(Default::default())),
            // `ConditionalCallEntry.compute_result_annotation` is the
            // BUILTIN_ANALYZERS path. This unused fallback matches the
            // void helper's s_None; the value helper's result needs
            // `args_s` and is served by `jit_conditional_call_elidable`.
            ExtRegistryEntry::ConditionalCall { .. } => Ok(s_none()),
            // `VirtualizableInstanceRepr.hook_access_field` emits a Void
            // `jit_force_virtualizable` operation.  The explicit Rust marker
            // therefore has the ordinary ExtRegistry callable annotation and
            // returns None at the call site.
            ExtRegistryEntry::JitForceVirtualizable => Ok(SomeValue::Builtin(SomeBuiltin::new(
                "pyre_interpreter.executioncontext.jit_force_virtualizable",
                None,
                Some("jit_force_virtualizable".to_string()),
            ))),
            // The annotation half is served by the `BigInt.from`
            // `BUILTIN_ANALYZERS` entry (`bigint_from`), which
            // `immutablevalue_hostobject` resolves before the
            // `extregistry.is_registered` fall-through. This entry only
            // fires on the rtyper path; returning the same classdef-less
            // `SomeInstance` shell keeps it a faithful port for any path
            // that consults it.
            ExtRegistryEntry::BigIntFrom => Ok(SomeValue::Instance(
                crate::annotator::model::SomeInstance::new(
                    None,
                    false,
                    std::collections::BTreeMap::new(),
                ),
            )),
            // The annotation half is served by the
            // `longlong2float.float2longlong` `BUILTIN_ANALYZERS` entry
            // (`float2longlong_analyzer`), which `immutablevalue_hostobject`
            // resolves before the `extregistry.is_registered` fall-through
            // (bookkeeper.py). This entry only fires on the rtyper
            // path; returning the same `SomeInteger(r_int64)` result that
            // `Float2LongLongEntry.compute_result_annotation` produces keeps
            // it a faithful port for any path that consults it.
            ExtRegistryEntry::Float2LongLong => Ok(SomeValue::Integer(
                crate::annotator::model::SomeInteger::new_with_knowntype(
                    false,
                    crate::annotator::model::KnownType::LongLong,
                ),
            )),
            ExtRegistryEntry::LongLong2Float => Ok(SomeValue::Float(
                crate::annotator::model::SomeFloat::default(),
            )),
            // `NonConstant` has no `compute_annotation` override upstream, so
            // the base implementation's `SomeBuiltin(compute_result_annotation,
            // methodname=...)` applies.  The call-time annotation is served by
            // the `majit_rlib.nonconst.non_constant` `BUILTIN_ANALYZERS` entry,
            // which `immutablevalue_hostobject` resolves before the
            // `extregistry.is_registered` fall-through.
            ExtRegistryEntry::NonConstant => Ok(SomeValue::Builtin(SomeBuiltin::new(
                "majit_rlib.nonconst.non_constant",
                None,
                Some("non_constant".to_string()),
            ))),
        }
    }

    /// RPython `ExtEnterLeaveMarker.compute_result_annotation(self,
    /// **kwds_s)` (rlib/jit.py) takes the keyword annotation
    /// map; the Rust port passes it as an explicit
    /// `HashMap<String, SomeValue>`. Variants that do not consume
    /// kwds delegate to [`Self::compute_annotation_bk`].
    ///
    /// `kwds_s` keys carry the upstream `'s_'` prefix so the cache
    /// shape under `Bookkeeper._jit_annotation_cache` is line-by-line
    /// with upstream (rlib/jit.py
    /// `expected = ['s_' + name for name ...]`).
    pub fn compute_annotation_with_kwds(
        &self,
        bookkeeper: &Rc<Bookkeeper>,
        kwds_s: &HashMap<String, Option<SomeValue>>,
    ) -> Result<SomeValue, AnnotatorError> {
        match self {
            ExtRegistryEntry::EnterLeaveMarker { meta, marker_kind } => {
                ext_enter_leave_marker_compute_result_annotation(
                    bookkeeper,
                    meta,
                    *marker_kind,
                    kwds_s,
                )
            }
            // rlib/jit.py — `ExtLoopHeader` returns
            // `annmodel.s_None` and ignores any kwds (loop_header
            // takes only `self` upstream).
            ExtRegistryEntry::LoopHeader { .. } => Ok(s_none()),
            ExtRegistryEntry::JitForceVirtualizable => Ok(s_none()),
            // upstream rarithmetic.py — `ForTypeEntry.\
            // compute_result_annotation(self, *args_s, **kwds_s)` returns
            // `SomeInteger(knowntype=int_type)`.  The instance is the int
            // class object (`r_uint`, `r_ulonglong`, ...); pyre records
            // it via `KnownType::Ruint` for r_uint and falls back to the
            // generic untyped class until the KnownType enum gains entries
            // for other ForType instances.  Signedness is derived inside
            // `new_with_knowntype` per model.py:222 — no separate
            // `unsigned` argument is passed (matches `SomeInteger(
            // knowntype=int_type)` upstream).
            ExtRegistryEntry::ForType { instance } => {
                let knowntype = if instance.qualname() == "rarithmetic.r_uint" {
                    crate::annotator::model::KnownType::Ruint
                } else {
                    crate::annotator::model::KnownType::Other
                };
                let _ = bookkeeper;
                Ok(SomeValue::Integer(
                    crate::annotator::model::SomeInteger::new_with_knowntype(false, knowntype),
                ))
            }
            other => other.compute_annotation_bk(bookkeeper),
        }
    }

    /// rbuiltin.py `entry.specialize_call` attribute lookup.
    ///
    /// Upstream `ExtRegistryEntry` (extregistry.py) does not
    /// define a base `specialize_call` method; subclasses that override
    /// it return a typer callable that `findbltintyper` returns
    /// directly, while subclasses that do not override it raise
    /// `AttributeError` at the attribute lookup site. The Rust enum
    /// mirrors per-variant: each arm whose upstream subclass overrides
    /// `specialize_call` returns `Ok(typer_fn)`; arms whose upstream
    /// subclass does not override it surface the AttributeError as a
    /// `TyperError` so the rtyper fails closed at the same point.
    pub(crate) fn specialize_call(&self) -> Result<BuiltinTyperFn, TyperError> {
        match self {
            // lltype.py `_ptrEntry(ExtRegistryEntry)` defines
            // only `_type_` and `compute_annotation` — no
            // `specialize_call`. Upstream raises AttributeError at
            // `entry.specialize_call`.
            ExtRegistryEntry::Ptr(_) => Err(TyperError::message(
                "'ExtRegistryEntry' object has no attribute 'specialize_call'",
            )),
            ExtRegistryEntry::HostValueBuiltin { .. }
            | ExtRegistryEntry::HostTypeAnnotation { .. }
            | ExtRegistryEntry::ExternalFunction(_) => Err(TyperError::message(
                "'ExtRegistryEntry' object has no attribute 'specialize_call'",
            )),
            // Marker variants override `specialize_call` upstream
            // (rlib/jit.py + 1016-1023) — but the upstream
            // `entry.specialize_call` attribute returns a *bound
            // method* whose `self` carries `meta` / `marker_kind`.
            // Rust's `BuiltinTyperFn` is a capture-free function
            // pointer so it cannot embed that context; the marker
            // dispatch lives on [`Self::specialize_marker_call`] and
            // is invoked by `BuiltinFunctionRepr::_call` ahead of the
            // standard `findbltintyper` path. Returning Err here is a
            // defensive signal: any caller that reaches
            // `findbltintyper` for a marker entry has bypassed the
            // marker special case and must be fixed.
            ExtRegistryEntry::EnterLeaveMarker { .. } | ExtRegistryEntry::LoopHeader { .. } => {
                Err(TyperError::message(
                    "ExtRegistryEntry marker variant: specialize_call must dispatch via \
                     specialize_marker_call (rbuiltin.rs::BuiltinFunctionRepr::_call) — \
                     `findbltintyper` cannot capture marker meta in a function pointer",
                ))
            }
            // upstream rarithmetic.py — `ForTypeEntry.\
            // specialize_call(self, hop)`:
            //     v_result, = hop.inputargs(hop.r_result.lowleveltype)
            //     hop.exception_cannot_occur()
            //     return v_result
            // Pyre's `rtype_r_uint` (rbuiltin.rs) mirrors this
            // verbatim — single `inputargs` call against
            // `hop.r_result.lowleveltype`, exception_cannot_occur,
            // then return.  Returning the function pointer here
            // completes the extregistry dispatch surface end-to-end.
            ExtRegistryEntry::ForType { instance, .. } => {
                if instance.qualname() == "rarithmetic.r_uint" {
                    Ok(super::rbuiltin::rtype_r_uint as BuiltinTyperFn)
                } else {
                    Err(TyperError::message(format!(
                        "ExtRegistryEntry::ForType: specialize_call for {} is not yet wired \
                         (rarithmetic.r_uint is the only int-class instance registered today)",
                        instance.qualname()
                    )))
                }
            }
            // rlib/jit.py — `Entry(_about_=we_are_jitted).\
            // specialize_call(self, hop)`:
            //     hop.exception_cannot_occur()
            //     return hop.inputconst(lltype.Signed, _we_are_jitted)
            // `rtype_we_are_jitted` mirrors this, emitting the
            // `_we_are_jitted` symbolic (`constfold::WE_ARE_JITTED_TAG_ID`)
            // at the result repr's lltype.
            ExtRegistryEntry::WeAreJitted => {
                Ok(super::rbuiltin::rtype_we_are_jitted as BuiltinTyperFn)
            }
            // rlib/jit.py — `ConditionalCallEntry.specialize_call`.
            ExtRegistryEntry::ConditionalCall { is_value: false } => {
                Ok(super::rbuiltin::rtype_jit_conditional_call as BuiltinTyperFn)
            }
            ExtRegistryEntry::ConditionalCall { is_value: true } => {
                Ok(super::rbuiltin::rtype_jit_conditional_call_value as BuiltinTyperFn)
            }
            // rlib/nonconst.py — `EntryNonConstant.specialize_call(self, hop)`:
            //     hop.exception_cannot_occur()
            //     return hop.inputarg(hop.r_result, arg=0)
            ExtRegistryEntry::NonConstant => {
                Ok(super::rbuiltin::rtype_non_constant as BuiltinTyperFn)
            }
            ExtRegistryEntry::JitForceVirtualizable => {
                Ok(super::rbuiltin::rtype_jit_force_virtualizable as BuiltinTyperFn)
            }
            // `BigInt.from` lowers to a residual `direct_call` to the
            // external `bigint_from` (extfunc.py:55-64
            // `ExternalFunctionRepr.rtype_simple_call` shape) whose result
            // is the opaque GcRef; `rtype_bigint_from` emits it.
            ExtRegistryEntry::BigIntFrom => {
                Ok(super::rbuiltin::rtype_bigint_from as BuiltinTyperFn)
            }
            // `rlib/longlong2float.py` — `Float2LongLongEntry.specialize_call`
            // emits `genop('convert_float_bytes_to_longlong', [v_float], SignedLongLong)`.
            ExtRegistryEntry::Float2LongLong => {
                Ok(super::rbuiltin::rtype_float2longlong as BuiltinTyperFn)
            }
            // `rlib/longlong2float.py` — inverse bitcast.
            ExtRegistryEntry::LongLong2Float => {
                Ok(super::rbuiltin::rtype_longlong2float as BuiltinTyperFn)
            }
        }
    }

    /// rlib/jit.py — `ExtEnterLeaveMarker.specialize_call(self,
    /// hop, **kwds_i)`.
    ///
    /// ```python
    /// def specialize_call(self, hop, **kwds_i):
    ///     from rpython.rtyper.lltypesystem import lltype
    ///     driver = self.instance.im_self
    ///     greens_v = []
    ///     reds_v = []
    ///     for name in driver.greens:
    ///         if '.' not in name:
    ///             i = kwds_i['i_' + name]
    ///             r_green = hop.args_r[i]
    ///             v_green = hop.inputarg(r_green, arg=i)
    ///         else:
    ///             ...  # dotted greenfield path (rlib/jit.py)
    ///         greens_v.append(v_green)
    ///     for name in driver.reds:
    ///         i = kwds_i['i_' + name]
    ///         r_red = hop.args_r[i]
    ///         v_red = hop.inputarg(r_red, arg=i)
    ///         reds_v.append(v_red)
    ///     hop.exception_cannot_occur()
    ///     vlist = [hop.inputconst(lltype.Void, self.instance.__name__),
    ///              hop.inputconst(lltype.Void, driver)]
    ///     vlist.extend(greens_v)
    ///     vlist.extend(reds_v)
    ///     return hop.genop('jit_marker', vlist, resulttype=lltype.Void)
    /// ```
    ///
    /// rlib/jit.py — `ExtLoopHeader.specialize_call(self, hop)`:
    ///
    /// ```python
    /// def specialize_call(self, hop):
    ///     from rpython.rtyper.lltypesystem import lltype
    ///     driver = self.instance.im_self
    ///     hop.exception_cannot_occur()
    ///     vlist = [hop.inputconst(lltype.Void, 'loop_header'),
    ///              hop.inputconst(lltype.Void, driver)]
    ///     return hop.genop('jit_marker', vlist, resulttype=lltype.Void)
    /// ```
    ///
    /// Dotted greenfields (rlib/jit.py) require the
    /// `r_red._get_field` walk + `_immutable_field` check + the
    /// `driver.ll_greenfields` registration on the runtime driver.
    /// Until those reprs and the `JitDriverMeta::ll_greenfields` slot
    /// land, dotted greens surface a parity-flagged TyperError so the
    /// gap fails closed instead of emitting a half-formed `jit_marker`.
    pub fn specialize_marker_call(
        &self,
        hop: &HighLevelOp,
        kwds_i: &HashMap<String, usize>,
    ) -> RTypeResult {
        match self {
            ExtRegistryEntry::EnterLeaveMarker { meta, marker_kind } => {
                // rlib/jit.py:957-994 greens/reds collection.
                let mut greens_v: Vec<Hlvalue> = Vec::with_capacity(meta.greens.len());
                for name in &meta.greens {
                    if name.contains('.') {
                        // rlib/jit.py:965-993 dotted-green path —
                        // requires `r_red._get_field(fieldname)` +
                        // `_immutable_field` assert + `driver.
                        // ll_greenfields[name] = (GTYPE, mangled_name)`
                        // side-effect on the runtime driver. Defer
                        // until those reprs + a `ll_greenfields`-style
                        // slot on `JitDriverMeta` land.
                        return Err(TyperError::message(format!(
                            "JitDriver({}) marker specialize_call: dotted greenfield \
                             {name:?} requires r_red._get_field + _immutable_field \
                             port from rlib/jit.py:965-993 (not yet ported)",
                            meta.name
                        )));
                    }
                    let v_green = inputarg_for_name(hop, kwds_i, &meta.name, "green", name)?;
                    greens_v.push(v_green);
                }
                let mut reds_v: Vec<Hlvalue> = Vec::with_capacity(meta.reds.len());
                for name in &meta.reds {
                    let v_red = inputarg_for_name(hop, kwds_i, &meta.name, "red", name)?;
                    reds_v.push(v_red);
                }
                // rlib/jit.py:1000.
                hop.exception_cannot_occur()?;
                // rlib/jit.py:1001-1004 — vlist =
                //   [Void(self.instance.__name__), Void(driver), greens..., reds...].
                let mut vlist: Vec<Hlvalue> = Vec::with_capacity(2 + greens_v.len() + reds_v.len());
                vlist.push(Hlvalue::Constant(HighLevelOp::inputconst(
                    &lltype::LowLevelType::Void,
                    &ConstValue::byte_str(marker_kind.upstream_method_name()),
                )?));
                vlist.push(Hlvalue::Constant(HighLevelOp::inputconst(
                    &lltype::LowLevelType::Void,
                    &ConstValue::HostObject(meta.id.clone()),
                )?));
                vlist.extend(greens_v);
                vlist.extend(reds_v);
                // rlib/jit.py:1005-1006 — `hop.genop('jit_marker',
                //   vlist, resulttype=lltype.Void)`.
                Ok(hop.genop("jit_marker", vlist, GenopResult::Void))
            }
            ExtRegistryEntry::LoopHeader { meta } => {
                // rlib/jit.py:1019.
                hop.exception_cannot_occur()?;
                // rlib/jit.py:1020-1021.
                let vlist = vec![
                    Hlvalue::Constant(HighLevelOp::inputconst(
                        &lltype::LowLevelType::Void,
                        &ConstValue::byte_str(crate::rlib::jit::LOOP_HEADER_METHOD_NAME),
                    )?),
                    Hlvalue::Constant(HighLevelOp::inputconst(
                        &lltype::LowLevelType::Void,
                        &ConstValue::HostObject(meta.id.clone()),
                    )?),
                ];
                // rlib/jit.py:1022-1023.
                let _ = kwds_i; // loop_header takes no kwds upstream
                Ok(hop.genop("jit_marker", vlist, GenopResult::Void))
            }
            other => Err(TyperError::message(format!(
                "ExtRegistryEntry::specialize_marker_call called on non-marker variant: {other:?}"
            ))),
        }
    }
}

/// Helper for `ExtEnterLeaveMarker.specialize_call` greens/reds loop —
/// looks up the `i_<name>` index from kwds_i, fetches the matching
/// repr from `hop.args_r`, and emits `hop.inputarg(r, arg=i)`.
fn inputarg_for_name(
    hop: &HighLevelOp,
    kwds_i: &HashMap<String, usize>,
    driver_name: &str,
    role: &str,
    name: &str,
) -> Result<Hlvalue, TyperError> {
    let key = format!("i_{name}");
    let i = *kwds_i.get(&key).ok_or_else(|| {
        TyperError::message(format!(
            "JitDriver({driver_name}) marker specialize_call: missing kwds_i[{key:?}] for \
             {role} {name:?}"
        ))
    })?;
    let r_arg = hop.args_r.borrow()[i].clone().ok_or_else(|| {
        TyperError::message(format!(
            "JitDriver({driver_name}) marker specialize_call: hop.args_r[{i}] missing for \
             {role} {name:?}"
        ))
    })?;
    hop.inputarg(&r_arg, i)
}

// RPython stores these as EXT_REGISTRY_BY_VALUE / EXT_REGISTRY_BY_TYPE dicts
// (extregistry.py). HashMap is therefore the literal Rust analogue,
// not a side table invented for convenience.
static HOST_VALUE_REGISTRY: OnceLock<Mutex<HashMap<HostObject, ExtRegistryEntry>>> =
    OnceLock::new();
static HOST_TYPE_REGISTRY: OnceLock<Mutex<HashMap<HostObject, ExtRegistryEntry>>> = OnceLock::new();

fn host_value_registry() -> &'static Mutex<HashMap<HostObject, ExtRegistryEntry>> {
    HOST_VALUE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn host_type_registry() -> &'static Mutex<HashMap<HostObject, ExtRegistryEntry>> {
    HOST_TYPE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Rust equivalent of `AutoRegisteringType._register_value`.
pub(crate) fn register_host_value(
    host: HostObject,
    entry: ExtRegistryEntry,
) -> Result<(), TyperError> {
    let mut registry = host_value_registry().lock();
    if registry.contains_key(&host) {
        return Err(TyperError::message(format!(
            "duplicate extregistry value registration for {}",
            host.qualname()
        )));
    }
    registry.insert(host, entry);
    Ok(())
}

/// Register `rarithmetic.r_uint` as an `ExtRegistryEntry::ForType`
/// value-level entry mirroring upstream `ForTypeEntry(_about_=r_uint)`
/// (rarithmetic.py).  Called from `HOST_ENV::populate_host_env`
/// so the registration runs once during the first HOST_ENV access; a
/// duplicate registration call is tolerated since `populate_host_env`
/// is itself OnceLock-gated.
pub(crate) fn register_r_uint(host: HostObject) {
    let mut registry = host_value_registry().lock();
    if !registry.contains_key(&host) {
        registry.insert(host.clone(), ExtRegistryEntry::ForType { instance: host });
    }
}

/// Rust equivalent of `AutoRegisteringType._register_type`.
#[cfg(test)]
pub(crate) fn register_host_type(
    host_type: HostObject,
    entry: ExtRegistryEntry,
) -> Result<(), TyperError> {
    let mut registry = host_type_registry().lock();
    if registry.contains_key(&host_type) {
        return Err(TyperError::message(format!(
            "duplicate extregistry type registration for {}",
            host_type.qualname()
        )));
    }
    registry.insert(host_type, entry);
    Ok(())
}

fn bind_lookup_instance(entry: ExtRegistryEntry, instance: &HostObject) -> ExtRegistryEntry {
    match entry {
        ExtRegistryEntry::HostTypeAnnotation {
            type_obj,
            annotation,
            ..
        } => ExtRegistryEntry::HostTypeAnnotation {
            type_obj,
            instance: Some(instance.clone()),
            annotation,
        },
        other => other,
    }
}

fn lookup_host_object(host: &HostObject) -> Option<ExtRegistryEntry> {
    // rlib/jit.py `class Entry(ExtRegistryEntry): _about_ =
    // we_are_jitted`. The `_about_` value is the host callable itself;
    // match it by qualified name (the same key `annotator/builtin.rs`
    // registers for `BUILTIN_ANALYZERS`).
    if host.qualname() == "majit_metainterp.jit.we_are_jitted" {
        return Some(ExtRegistryEntry::WeAreJitted);
    }
    // rlib/jit.py `ConditionalCallEntry._about_ =
    // _jit_conditional_call, _jit_conditional_call_value`.
    if host.qualname() == "majit_rlib.jit.conditional_call_elidable1" {
        return Some(ExtRegistryEntry::ConditionalCall { is_value: true });
    }
    if matches!(
        host.qualname(),
        "majit_rlib.jit.conditional_call0"
            | "majit_rlib.jit.conditional_call1"
            | "majit_rlib.jit.conditional_call2"
            | "majit_rlib.jit.conditional_call3"
            | "majit_rlib.jit.conditional_call4"
    ) {
        return Some(ExtRegistryEntry::ConditionalCall { is_value: false });
    }
    if host.qualname() == "pyre_interpreter.executioncontext.jit_force_virtualizable" {
        return Some(ExtRegistryEntry::JitForceVirtualizable);
    }
    // `rlib/nonconst.py class EntryNonConstant(ExtRegistryEntry): _about_ =
    // NonConstant` — keyed by the same qualname `annotator/builtin.rs`
    // registers for `BUILTIN_ANALYZERS`.
    if host.qualname() == "majit_rlib.nonconst.non_constant" {
        return Some(ExtRegistryEntry::NonConstant);
    }
    // `BigInt.from` residual external — keyed by the qualname the
    // `BigInt.from` `BUILTIN_ANALYZERS` annotation entry uses, so the
    // rtyper's `findbltintyper` → `extregistry.lookup` resolves the same
    // host callable the annotator typed as `SomeBuiltin`.
    if host.qualname() == "BigInt.from" {
        return Some(ExtRegistryEntry::BigIntFrom);
    }
    if host.qualname() == "longlong2float.float2longlong" {
        return Some(ExtRegistryEntry::Float2LongLong);
    }
    if host.qualname() == "longlong2float.longlong2float" {
        return Some(ExtRegistryEntry::LongLong2Float);
    }
    if let Some(entry) = host_value_registry().lock().get(host).cloned() {
        return Some(entry);
    }
    let cls = host.instance_class()?;
    host_type_registry()
        .lock()
        .get(cls)
        .cloned()
        .map(|entry| bind_lookup_instance(entry, host))
}

// Value-level registry — RPython `EXT_REGISTRY_BY_VALUE` (extregistry.py).

/// RPython `extregistry.is_registered(instance)` (extregistry.py).
///
/// Upstream:
///
/// ```python
/// def _lookup_cls(instance):
///     try:
///         return EXT_REGISTRY_BY_VALUE[instance]
///     except (KeyError, TypeError):
///         return _lookup_type_cls(type(instance))
///
/// def is_registered(instance):
///     try:
///         _lookup_cls(instance)
///     except KeyError:
///         return False
///     return True
/// ```
///
/// Upstream falls back to type-level lookup when the value-level
/// dictionary misses; the Rust port mirrors that for HostObject
/// instances via [`HostObject::instance_class`]. `LLPtr` matches the
/// upstream `_ptrEntry._type_=_ptr` fallback (every `_ptr` value's
/// `type(instance)` is `_ptr`).
pub fn is_registered(instance: &ConstValue) -> bool {
    match instance {
        ConstValue::LLPtr(_) => true,
        ConstValue::HostObject(host) => lookup_host_object(host).is_some(),
        _ => false,
    }
}

/// RPython `extregistry.lookup(instance)` (extregistry.py).
///
/// Upstream:
///
/// ```python
/// def lookup(instance):
///     Entry = _lookup_cls(instance)
///     return Entry(type(instance), instance)
/// ```
///
/// The Rust port constructs the [`ExtRegistryEntry`] variant
/// corresponding to the carrier shape, with the value-level instance
/// embedded so `compute_annotation` can read it.
pub fn lookup(instance: &ConstValue) -> Option<ExtRegistryEntry> {
    match instance {
        ConstValue::LLPtr(ptr) => Some(ExtRegistryEntry::Ptr(Some((**ptr).clone()))),
        ConstValue::HostObject(host) => lookup_host_object(host),
        _ => None,
    }
}

// Type-level registry — RPython `EXT_REGISTRY_BY_TYPE` (extregistry.py).

/// RPython `extregistry.is_registered_type(tp)` (extregistry.py).
///
/// Upstream consults `EXT_REGISTRY_BY_TYPE` directly. The Rust port
/// first checks the HostObject registry dict populated by
/// [`register_host_type`].
pub fn is_registered_type(host: &HostObject) -> bool {
    host_type_registry().lock().contains_key(host)
}

/// RPython `extregistry.lookup_type(tp)` (extregistry.py).
///
/// Upstream:
///
/// ```python
/// def lookup_type(tp):
///     Entry = _lookup_type_cls(tp)
///     return Entry(tp)  # instance=None
/// ```
///
/// Constructed entries carry `instance=None` so subclasses whose
/// `compute_annotation` reads `self.instance` (e.g. `_ptrEntry`) raise
/// the same AttributeError upstream produces. The Rust port encodes
/// this via the `Option<_ptr>` slot on [`ExtRegistryEntry::Ptr`].
pub fn lookup_type(host: &HostObject) -> Option<ExtRegistryEntry> {
    host_type_registry().lock().get(host).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::translator::rtyper::lltypesystem::lltype::{RUNTIME_TYPE_INFO, nullptr};

    fn sample_ptr() -> crate::translator::rtyper::lltypesystem::lltype::_ptr {
        nullptr(RUNTIME_TYPE_INFO.clone()).expect("RUNTIME_TYPE_INFO is a container type")
    }

    /// upstream extregistry.py — `is_registered` returns True
    /// for any value whose type is registered (here, `_ptr`).
    #[test]
    fn is_registered_matches_llptr_value() {
        let ptr = sample_ptr();
        let cv = ConstValue::LLPtr(Box::new(ptr));
        assert!(is_registered(&cv));
    }

    /// `lookup` returns the matching entry with the value-level
    /// instance embedded (upstream `Entry(type, instance)` construction).
    #[test]
    fn lookup_returns_ptr_entry_with_instance() {
        let ptr = sample_ptr();
        let cv = ConstValue::LLPtr(Box::new(ptr));
        let entry = lookup(&cv).expect("LLPtr must surface as Ptr entry");
        match entry {
            ExtRegistryEntry::Ptr(Some(_)) => {}
            ExtRegistryEntry::Ptr(None) => panic!("value-level lookup must carry instance"),
            other => panic!("LLPtr lookup returned wrong entry: {other:?}"),
        }
    }

    /// Unregistered HostObject values still miss, matching upstream
    /// `KeyError` from both value and type registries.
    #[test]
    fn is_registered_misses_unregistered_hostobject() {
        let host = HostObject::new_class("UserClass", vec![]);
        let cv = ConstValue::HostObject(host);
        assert!(!is_registered(&cv));
        assert!(lookup(&cv).is_none());
    }

    #[test]
    fn flexible_weak_dict_exposes_three_store_shape() {
        let mut dict = FlexibleWeakDict::new();
        dict.__setitem__("a".to_string(), 1);
        dict._weakdict.insert("b".to_string(), 2);
        dict._iddict.insert("c".to_string(), 3);

        assert_eq!(dict.__getitem__(&"a".to_string()), Some(&1));
        assert_eq!(dict.__getitem__(&"b".to_string()), Some(&2));
        assert_eq!(dict.__getitem__(&"c".to_string()), Some(&3));
        let mut keys = dict.keys();
        keys.sort();
        assert_eq!(
            keys,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(dict.__delitem__(&"a".to_string()), Some(1));
    }

    /// rlib/jit.py `class Entry(ExtRegistryEntry): _about_ =
    /// we_are_jitted` — the `we_are_jitted` host callable surfaces the
    /// `WeAreJitted` entry, whose `specialize_call` resolves the rtyper
    /// path. `compute_annotation` returns `SomeBool` — the faithful
    /// port of RPython's `SomeInteger(nonneg=True)` (rlib/jit.py)
    /// for pyre's bool-typed `we_are_jitted`, the same annotation the
    /// `BUILTIN_ANALYZERS` entry serves.
    #[test]
    fn we_are_jitted_resolves_specialize_call() {
        let host = HostObject::new_builtin_callable("majit_metainterp.jit.we_are_jitted");
        let cv = ConstValue::HostObject(host);
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("we_are_jitted must surface an entry");
        assert!(matches!(entry, ExtRegistryEntry::WeAreJitted));
        assert!(entry.specialize_call().is_ok());
        assert!(matches!(entry.compute_annotation(), Ok(SomeValue::Bool(_))));
    }

    /// rlib/jit.py `ConditionalCallEntry._about_ =
    /// _jit_conditional_call, _jit_conditional_call_value`.
    #[test]
    fn conditional_call_resolves_specialize_call() {
        let void_host = HostObject::new_builtin_callable("majit_rlib.jit.conditional_call1");
        assert!(is_registered(&ConstValue::HostObject(void_host.clone())));
        let void_entry = lookup(&ConstValue::HostObject(void_host))
            .expect("conditional_call1 must surface an entry");
        assert!(matches!(
            void_entry,
            ExtRegistryEntry::ConditionalCall { is_value: false }
        ));
        assert!(void_entry.specialize_call().is_ok());
        assert!(matches!(
            void_entry.compute_annotation(),
            Ok(SomeValue::None_(_))
        ));

        let value_host =
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call_elidable1");
        let value_entry = lookup(&ConstValue::HostObject(value_host))
            .expect("conditional_call_elidable1 must surface an entry");
        assert!(matches!(
            value_entry,
            ExtRegistryEntry::ConditionalCall { is_value: true }
        ));
        assert!(value_entry.specialize_call().is_ok());
    }

    #[test]
    fn jit_force_virtualizable_resolves_to_llop_specializer() {
        let host = HostObject::new_builtin_callable(
            "pyre_interpreter.executioncontext.jit_force_virtualizable",
        );
        let cv = ConstValue::HostObject(host);
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("force marker must surface an entry");
        assert!(matches!(entry, ExtRegistryEntry::JitForceVirtualizable));
        assert!(entry.specialize_call().is_ok());
        let bookkeeper = Rc::new(Bookkeeper::new());
        assert!(matches!(
            entry.compute_annotation_with_kwds(&bookkeeper, &std::collections::HashMap::new()),
            Ok(SomeValue::None_(_))
        ));
    }

    /// `BigInt.from` — the foreign opaque `From<i64>` conversion surfaces
    /// the `BigIntFrom` residual-external entry, keyed by the same qualname
    /// the `BUILTIN_ANALYZERS` annotation entry uses. `specialize_call`
    /// resolves the rtyper residual lowering; `compute_annotation` returns
    /// the classdef-less `SomeInstance` opaque shell (the same one
    /// `bigint_from` produces).
    #[test]
    fn bigint_from_resolves_specialize_call() {
        let host = HostObject::new_builtin_callable("BigInt.from");
        let cv = ConstValue::HostObject(host);
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("BigInt.from must surface an entry");
        assert!(matches!(entry, ExtRegistryEntry::BigIntFrom));
        assert!(entry.specialize_call().is_ok());
        assert!(matches!(
            entry.compute_annotation(),
            Ok(SomeValue::Instance(_))
        ));
    }

    /// Type-level lookup returns None for unregistered host classes.
    #[test]
    fn is_registered_type_misses_unregistered_host_class() {
        let host = HostObject::new_class("UserClass", vec![]);
        assert!(!is_registered_type(&host));
        assert!(lookup_type(&host).is_none());
    }

    #[test]
    fn host_value_registry_hits_registered_hostobject() {
        let host = HostObject::new_builtin_callable("pkg.extregistry.value");
        register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "pkg.extregistry.value".into(),
                methodname: Some("value".into()),
            },
        )
        .expect("register host value");
        let cv = ConstValue::HostObject(host);
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("registered HostObject value must lookup");
        let s = entry
            .compute_annotation()
            .expect("registered host builtin computes annotation");
        assert!(matches!(s, SomeValue::Builtin(_)));
    }

    #[test]
    fn host_type_registry_hits_registered_host_class() {
        let host = HostObject::new_class("pkg.extregistry.Type", vec![]);
        register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Int,
            },
        )
        .expect("register host type");
        assert!(is_registered_type(&host));
        let entry = lookup_type(&host).expect("registered HostObject type must lookup");
        let s = entry
            .compute_annotation()
            .expect("registered host type computes annotation");
        assert!(matches!(s, SomeValue::Integer(_)));
    }

    #[test]
    fn host_value_registry_rejects_duplicate_registration() {
        let host = HostObject::new_builtin_callable("pkg.extregistry.duplicate_value");
        register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "first".into(),
                methodname: None,
            },
        )
        .expect("first registration must succeed");

        let err = register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "second".into(),
                methodname: None,
            },
        )
        .expect_err("duplicate _about_ registration must fail like RPython");
        assert!(format!("{err:?}").contains("duplicate extregistry value registration"));
    }

    #[test]
    fn host_type_registry_rejects_duplicate_registration() {
        let host = HostObject::new_class("pkg.extregistry.DuplicateType", vec![]);
        register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Int,
            },
        )
        .expect("first registration must succeed");

        let err = register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Bool,
            },
        )
        .expect_err("duplicate _type_ registration must fail like RPython");
        assert!(format!("{err:?}").contains("duplicate extregistry type registration"));
    }

    #[test]
    fn host_value_lookup_falls_back_to_registered_instance_class() {
        let cls = HostObject::new_class("pkg.extregistry.FallbackType", vec![]);
        register_host_type(
            cls.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: cls.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Str,
            },
        )
        .expect("register host type");

        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let cv = ConstValue::HostObject(inst.clone());
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("registered instance class must lookup");
        match entry {
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj,
                instance,
                annotation: RegisteredAnnotation::Str,
            } => {
                assert_eq!(type_obj.identity_id(), cls.identity_id());
                assert_eq!(
                    instance.as_ref().map(HostObject::identity_id),
                    Some(inst.identity_id())
                );
            }
            other => panic!("wrong fallback entry: {other:?}"),
        }
    }

    /// Type-level entry construction carries `instance=None` upstream;
    /// the Rust port's `Ptr(None)` mirrors this. `compute_annotation`
    /// surfaces AnnotatorError because `_ptrEntry` reads `self.instance`.
    #[test]
    fn ptr_entry_without_instance_errors_at_compute_annotation() {
        let entry = ExtRegistryEntry::Ptr(None);
        let err = entry
            .compute_annotation()
            .expect_err("type-level _ptrEntry must error without instance");
        assert!(err.to_string().contains("requires self.instance"));
    }

    fn meta_for(name: &str, greens: &[&str], reds: &[&str]) -> Arc<JitDriverMeta> {
        let id = HostObject::new_instance(
            HostObject::new_class(format!("pkg.extregistry.{name}"), vec![]),
            vec![],
        );
        Arc::new(JitDriverMeta {
            id,
            name: name.to_string(),
            greens: greens.iter().map(|s| (*s).to_string()).collect(),
            reds: reds.iter().map(|s| (*s).to_string()).collect(),
            virtualizables: vec![],
            autoreds: false,
            numreds: Some(reds.len()),
            is_recursive: false,
            vec: false,
            get_printable_location: None,
            get_location: None,
        })
    }

    /// rlib/jit.py — `_make_extregistryentries` registers
    /// `(jit_merge_point, can_enter_jit)` to one Entry class and
    /// `loop_header` to another. The Rust port replays the
    /// registration shape through HostObject identities.
    #[test]
    fn marker_registration_round_trips() {
        let meta = meta_for("PyPyJitDriver", &["pc"], &["frame"]);
        let driver_id = meta.id.identity_id();
        let jmp_host = HostObject::new_builtin_callable("pkg.extregistry.driver.jit_merge_point");
        let lh_host = HostObject::new_builtin_callable("pkg.extregistry.driver.loop_header");

        register_host_value(
            jmp_host.clone(),
            ExtRegistryEntry::EnterLeaveMarker {
                meta: meta.clone(),
                marker_kind: JitMarkerKind::JitMergePoint,
            },
        )
        .expect("register jit_merge_point marker");
        register_host_value(
            lh_host.clone(),
            ExtRegistryEntry::LoopHeader { meta: meta.clone() },
        )
        .expect("register loop_header marker");

        let jmp_entry = lookup(&ConstValue::HostObject(jmp_host)).expect("jit_merge_point lookup");
        match jmp_entry {
            ExtRegistryEntry::EnterLeaveMarker {
                meta: m,
                marker_kind: JitMarkerKind::JitMergePoint,
            } => assert_eq!(m.id.identity_id(), driver_id),
            other => panic!("wrong entry for jit_merge_point: {other:?}"),
        }

        let lh_entry = lookup(&ConstValue::HostObject(lh_host)).expect("loop_header lookup");
        match lh_entry {
            ExtRegistryEntry::LoopHeader { meta: m } => {
                assert_eq!(m.id.identity_id(), driver_id)
            }
            other => panic!("wrong entry for loop_header: {other:?}"),
        }
    }

    /// jit.py:889 dispatch on `self.instance.__name__` — Rust mirrors
    /// the discrimination via `JitMarkerKind`. `makekey` must encode
    /// the kind so two markers on the same driver hash to distinct
    /// keys (matching upstream `(self.__class__, self.type,
    /// self.instance)` identity, where `self.instance` is the bound
    /// method object, not the driver).
    #[test]
    fn marker_kind_makekey_distinguishes_methods() {
        let meta = meta_for("MakeKeyDriver", &["pc"], &["frame"]);
        let jmp = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let cej = ExtRegistryEntry::EnterLeaveMarker {
            meta,
            marker_kind: JitMarkerKind::CanEnterJit,
        };
        assert_ne!(jmp.makekey(), cej.makekey());
    }

    /// rbuiltin.rs `BuiltinFunctionRepr::_call` dispatches marker
    /// entries through `specialize_marker_call` directly because the
    /// `BuiltinTyperFn` function pointer cannot capture `meta` /
    /// `marker_kind`. `findbltintyper` must therefore never reach
    /// `specialize_call` for marker variants — this test locks the
    /// defensive Err so a regression that bypasses the marker special
    /// case fails closed instead of silently producing bad lowering.
    #[test]
    fn marker_variants_specialize_call_routes_via_specialize_marker_call() {
        let meta = meta_for("DeferDriver", &["pc"], &["frame"]);
        let entries = [
            ExtRegistryEntry::EnterLeaveMarker {
                meta: meta.clone(),
                marker_kind: JitMarkerKind::JitMergePoint,
            },
            ExtRegistryEntry::EnterLeaveMarker {
                meta: meta.clone(),
                marker_kind: JitMarkerKind::CanEnterJit,
            },
            ExtRegistryEntry::LoopHeader { meta: meta.clone() },
        ];

        for entry in &entries {
            let typ_err = entry
                .specialize_call()
                .expect_err("findbltintyper path is closed for markers");
            assert!(
                format!("{typ_err:?}").contains("specialize_marker_call"),
                "unexpected typer error: {typ_err:?}"
            );
        }
    }

    fn integer_kwd(name: &str) -> (String, Option<SomeValue>) {
        (
            format!("s_{name}"),
            Some(SomeValue::Integer(
                crate::annotator::model::SomeInteger::default(),
            )),
        )
    }

    /// rlib/jit.py:903-914 — happy path. With the kwds keys exactly
    /// matching `'s_' + driver.greens + driver.reds`, the cache is
    /// populated under the driver identity and the call returns
    /// `s_None` per rlib/jit.py:923.
    #[test]
    fn marker_compute_annotation_with_kwds_caches_and_returns_s_none() {
        let meta = meta_for("CacheDriver", &["pc", "is_profiled"], &["frame"]);
        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [
            integer_kwd("pc"),
            integer_kwd("is_profiled"),
            integer_kwd("frame"),
        ]
        .into_iter()
        .collect();

        let result = entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect("happy-path compute_annotation_with_kwds");
        assert!(matches!(result, SomeValue::None_(_)));

        let cache_outer = bk._jit_annotation_cache.borrow();
        let cache = cache_outer
            .get(&meta.id)
            .expect("cache populated under driver identity");
        for key in ["s_pc", "s_is_profiled", "s_frame"] {
            assert!(cache.contains_key(key), "cache missing {key}");
        }
    }

    /// rlib/jit.py:898-901 — keyword set must match exactly.
    /// Missing kwargs (a subset of greens+reds) must raise
    /// `JitHintError` upstream; the Rust port surfaces it as
    /// `AnnotatorError`.
    #[test]
    fn marker_compute_annotation_rejects_missing_kwarg() {
        let meta = meta_for("MissingKwargDriver", &["pc"], &["frame"]);
        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [integer_kwd("pc")].into_iter().collect();

        let err = entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect_err("missing kwarg must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("s_pc") && msg.contains("s_frame"),
            "error must list expected keys, got: {msg}"
        );
    }

    /// rlib/jit.py:898-901 — extra kwargs (not in greens+reds) must
    /// also reject.
    #[test]
    fn marker_compute_annotation_rejects_extra_kwarg() {
        let meta = meta_for("ExtraKwargDriver", &["pc"], &["frame"]);
        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [
            integer_kwd("pc"),
            integer_kwd("frame"),
            integer_kwd("not_a_field"),
        ]
        .into_iter()
        .collect();

        let err = entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect_err("extra kwarg must reject");
        assert!(
            err.to_string().contains("s_not_a_field"),
            "error must surface the offending key: {err}"
        );
    }

    /// rlib/jit.py:911-914 — repeat dispatches union annotations into
    /// the per-driver cache. Two calls with the same key must end up
    /// with one merged entry (not two separate ones).
    #[test]
    fn marker_compute_annotation_unions_multiple_calls() {
        let meta = meta_for("UnionDriver", &["pc"], &["frame"]);
        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [integer_kwd("pc"), integer_kwd("frame")]
            .into_iter()
            .collect();

        for _ in 0..2 {
            entry
                .compute_annotation_with_kwds(&bk, &kwds)
                .expect("each call must succeed");
        }

        let cache_outer = bk._jit_annotation_cache.borrow();
        let cache = cache_outer
            .get(&meta.id)
            .expect("cache populated under driver identity");
        assert_eq!(cache.len(), 2, "two kwarg keys, not duplicated per call");
    }

    /// rlib/jit.py — `ExtLoopHeader.compute_result_annotation`
    /// returns `annmodel.s_None` and accepts no kwargs.
    #[test]
    fn loop_header_compute_annotation_with_kwds_is_s_none() {
        let meta = meta_for("LoopHeaderDriver", &["pc"], &["frame"]);
        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::LoopHeader { meta };
        let kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        let result = entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect("LoopHeader must annotate without kwds");
        assert!(matches!(result, SomeValue::None_(_)));
    }

    /// upstream extregistry.py — base `compute_annotation` returns
    /// `SomeBuiltin(self.compute_result_annotation,
    ///              methodname=getattr(func, '__name__', None))`. The
    /// Rust port mirrors that for both marker variants so the
    /// bookkeeper's `immutablevalue(host)` path lands the marker as a
    /// `SomeValue::Builtin` (the const_box is set by the bookkeeper).
    #[test]
    fn marker_compute_annotation_returns_builtin() {
        let meta = meta_for("CARDriver", &["pc"], &["frame"]);
        for (kind, expected_methodname, expected_analyser) in [
            (
                JitMarkerKind::JitMergePoint,
                "jit_merge_point",
                "rlib.jit.ExtEnterLeaveMarker.jit_merge_point",
            ),
            (
                JitMarkerKind::CanEnterJit,
                "can_enter_jit",
                "rlib.jit.ExtEnterLeaveMarker.can_enter_jit",
            ),
        ] {
            let entry = ExtRegistryEntry::EnterLeaveMarker {
                meta: meta.clone(),
                marker_kind: kind,
            };
            match entry
                .compute_annotation()
                .expect("marker compute_annotation returns SomeBuiltin")
            {
                SomeValue::Builtin(sb) => {
                    assert_eq!(sb.analyser_name, expected_analyser);
                    assert_eq!(sb.methodname.as_deref(), Some(expected_methodname));
                    assert!(
                        sb.base.const_box.is_none(),
                        "const_box stays None until immutablevalue caller injects raw"
                    );
                }
                other => panic!("expected SomeBuiltin for {kind:?}, got {other:?}"),
            }
        }

        let entry = ExtRegistryEntry::LoopHeader { meta };
        match entry
            .compute_annotation()
            .expect("LoopHeader compute_annotation returns SomeBuiltin")
        {
            SomeValue::Builtin(sb) => {
                assert_eq!(sb.analyser_name, "rlib.jit.ExtLoopHeader.loop_header");
                assert_eq!(sb.methodname.as_deref(), Some("loop_header"));
            }
            other => panic!("expected SomeBuiltin for LoopHeader, got {other:?}"),
        }
    }

    /// rlib/jit.py + 925-950 — `jit_merge_point` triggers
    /// `annotate_hooks`, which routes each driver hook through
    /// `bk.emulate_pbc_call('jitdriver.<hook>', s_func, args_s)`. After
    /// dispatch, the unique key must be present in
    /// `Bookkeeper.emulated_pbc_calls`. The downstream `pbc_call` may
    /// surface an error (annotator not driving any reflow), but
    /// `emulate_pbc_call` records the unique-key entry before the
    /// `pbc_call` invocation (both on `bookkeeper.rs`'s `Bookkeeper`), so the
    /// side-effect is observable regardless of dispatch outcome.
    #[test]
    fn marker_jit_merge_point_dispatches_annotate_hook() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::EmulatedPbcCallKey;
        use crate::flowspace::model::{ConstValue, Constant, GraphFunc};

        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let hook_func = HostObject::new_user_function(GraphFunc::new(
            "pkg.driver.get_printable_location",
            globals.clone(),
        ));
        let location_func =
            HostObject::new_user_function(GraphFunc::new("pkg.driver.get_location", globals));

        let mut meta_inner = (*meta_for("HookDriver", &["pc"], &["frame"])).clone();
        meta_inner.get_printable_location = Some(hook_func.clone());
        meta_inner.get_location = Some(location_func.clone());
        let meta = Arc::new(meta_inner);

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [integer_kwd("pc"), integer_kwd("frame")]
            .into_iter()
            .collect();

        // Outcome irrelevant: `emulate_pbc_call` records the unique key
        // before delegating to `pbc_call`, so the side-effect lands
        // even when downstream analysis errors out. Only the first hook
        // is checked because upstream `annotate_hooks` calls
        // `h(get_printable_location, ...)` and `h(get_location, ...)`
        // sequentially without exception isolation; if the first
        // emulate_pbc_call surfaces an error from a stub annotator (no
        // graph in flight), the second hook never runs — matching
        // upstream parity.
        let _ = entry.compute_annotation_with_kwds(&bk, &kwds);

        let emulated = bk.emulated_pbc_calls.borrow();
        let key = EmulatedPbcCallKey::Text(format!("jitdriver.{}", hook_func.qualname()));
        assert!(
            emulated.contains_key(&key),
            "annotate_hook must register {key:?} in emulated_pbc_calls; \
             saw keys = {:?}",
            emulated.keys().collect::<Vec<_>>()
        );
        // location_func registration depends on whether the first
        // emulate_pbc_call surfaced an Err — this test does not
        // discriminate between successful and failed pbc_call to keep
        // the assertion stable across annotator-state changes.
        let _ = location_func;
    }

    /// rlib/jit.py — only `jit_merge_point` runs `annotate_hooks`.
    /// `can_enter_jit` must skip the branch entirely so the hook
    /// callable's annotation is not pre-emulated by every reflow.
    #[test]
    fn marker_can_enter_jit_skips_annotate_hooks() {
        use crate::flowspace::model::{ConstValue, Constant, GraphFunc};

        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let hook_func =
            HostObject::new_user_function(GraphFunc::new("pkg.driver.skip_hook", globals));

        let mut meta_inner = (*meta_for("CanEnterDriver", &["pc"], &["frame"])).clone();
        meta_inner.get_printable_location = Some(hook_func);
        let meta = Arc::new(meta_inner);

        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta,
            marker_kind: JitMarkerKind::CanEnterJit,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [integer_kwd("pc"), integer_kwd("frame")]
            .into_iter()
            .collect();

        entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect("can_enter_jit must succeed without dispatching hooks");

        assert!(
            bk.emulated_pbc_calls.borrow().is_empty(),
            "can_enter_jit must not populate emulated_pbc_calls"
        );
    }

    /// Helpers for the marker specialize_call tests — build a hop
    /// with int-typed greens / reds and a Signed args_r so the
    /// `inputarg(r, i)` path produces real Variables.
    fn signed_arg(name: &str) -> Hlvalue {
        use crate::flowspace::model::Variable;
        let v = Variable::named(name);
        v.set_concretetype(Some(lltype::LowLevelType::Signed));
        Hlvalue::Variable(v)
    }

    fn make_marker_hop(
        arg_names: &[&str],
    ) -> (
        HighLevelOp,
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
    ) {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc as StdRc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = StdRc::new(RPythonTyper::new(&ann));
        let llops = StdRc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let args_v: Vec<Hlvalue> = arg_names.iter().map(|n| signed_arg(n)).collect();
        let result_var = Variable::named("v_result");
        result_var.set_concretetype(Some(lltype::LowLevelType::Void));
        let result_v = Hlvalue::Variable(result_var);
        let spaceop = SpaceOperation::new("simple_call".to_string(), args_v.clone(), result_v);
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops);
        for arg in &args_v {
            hop.args_v.borrow_mut().push(arg.clone());
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::default()));
            let r: std::sync::Arc<dyn Repr> = signed_repr();
            hop.args_r.borrow_mut().push(Some(r));
        }
        *hop.s_result.borrow_mut() = Some(s_none());
        // r_result Void — leave as None; specialize_marker_call emits
        // genop with GenopResult::Void so r_result is unused.
        (hop, ann)
    }

    /// rlib/jit.py:1001-1006 — happy path. EnterLeaveMarker
    /// specialize_call emits `jit_marker(method_name, driver, greens...,
    /// reds...)` on the llops buffer.
    #[test]
    fn marker_specialize_call_emits_jit_marker_for_jit_merge_point() {
        let meta = meta_for("EmitDriver", &["pc"], &["frame"]);
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta: meta.clone(),
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let (hop, _ann) = make_marker_hop(&["pc", "frame"]);
        let kwds_i: HashMap<String, usize> = [("i_pc".to_string(), 0), ("i_frame".to_string(), 1)]
            .into_iter()
            .collect();

        entry
            .specialize_marker_call(&hop, &kwds_i)
            .expect("happy-path specialize_marker_call");

        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("at least one llop emitted");
        assert_eq!(last.opname, "jit_marker");
        // vlist[0] = Void(method_name)
        let arg0 = match &last.args[0] {
            Hlvalue::Constant(c) => c,
            other => panic!("vlist[0] must be a Constant, got {other:?}"),
        };
        match &arg0.value {
            ConstValue::ByteStr(bytes) => {
                assert_eq!(bytes.as_slice(), b"jit_merge_point")
            }
            other => panic!("vlist[0].value must be ByteStr, got {other:?}"),
        }
        // vlist[1] = Void(driver host)
        let arg1 = match &last.args[1] {
            Hlvalue::Constant(c) => c,
            other => panic!("vlist[1] must be a Constant, got {other:?}"),
        };
        match &arg1.value {
            ConstValue::HostObject(host) => {
                assert_eq!(host.identity_id(), meta.id.identity_id())
            }
            other => panic!("vlist[1].value must be HostObject, got {other:?}"),
        }
        // vlist[2..] = greens + reds Variables
        assert_eq!(last.args.len(), 4, "vlist=[name,driver,green,red]");
    }

    /// rlib/jit.py:1019-1023 — LoopHeader emits the same `jit_marker`
    /// op shape but with name=`'loop_header'` and no greens/reds.
    #[test]
    fn loop_header_specialize_call_emits_jit_marker() {
        let meta = meta_for("LoopHeaderEmitDriver", &["pc"], &["frame"]);
        let entry = ExtRegistryEntry::LoopHeader { meta: meta.clone() };
        let (hop, _ann) = make_marker_hop(&[]);
        let kwds_i: HashMap<String, usize> = HashMap::new();

        entry
            .specialize_marker_call(&hop, &kwds_i)
            .expect("LoopHeader specialize_marker_call");

        let llops = hop.llops.borrow();
        let last = llops.ops.last().expect("at least one llop emitted");
        assert_eq!(last.opname, "jit_marker");
        assert_eq!(last.args.len(), 2, "LoopHeader vlist=[name,driver] only");
        match &last.args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(bytes) => assert_eq!(bytes.as_slice(), b"loop_header"),
                other => panic!("vlist[0].value must be ByteStr, got {other:?}"),
            },
            other => panic!("vlist[0] must be Constant, got {other:?}"),
        }
        match &last.args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(host) => {
                    assert_eq!(host.identity_id(), meta.id.identity_id())
                }
                other => panic!("vlist[1].value must be HostObject, got {other:?}"),
            },
            other => panic!("vlist[1] must be Constant, got {other:?}"),
        }
    }

    /// rlib/jit.py:961 / 996 — `kwds_i['i_' + name]` raises KeyError
    /// upstream when missing. Rust mirrors that as a TyperError.
    #[test]
    fn marker_specialize_call_missing_kwds_i_errors() {
        let meta = meta_for("MissingKwdsDriver", &["pc"], &["frame"]);
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta,
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let (hop, _ann) = make_marker_hop(&["pc"]);
        // `i_frame` deliberately omitted; greens succeed, reds fail.
        let kwds_i: HashMap<String, usize> = [("i_pc".to_string(), 0)].into_iter().collect();
        let err = entry
            .specialize_marker_call(&hop, &kwds_i)
            .expect_err("missing i_frame must error");
        assert!(
            err.to_string().contains("i_frame"),
            "expected i_frame in error: {err}"
        );
    }

    /// rlib/jit.py:965-993 dotted greenfield path is deferred — it
    /// must surface a clear TyperError until the supporting
    /// `r_red._get_field` + `_immutable_field` infrastructure lands.
    #[test]
    fn marker_specialize_call_dotted_green_errors() {
        let meta = meta_for("DottedGreenDriver", &["frame.code"], &["frame"]);
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta,
            marker_kind: JitMarkerKind::JitMergePoint,
        };
        let (hop, _ann) = make_marker_hop(&["frame"]);
        let kwds_i: HashMap<String, usize> = [("i_frame".to_string(), 0)].into_iter().collect();
        let err = entry
            .specialize_marker_call(&hop, &kwds_i)
            .expect_err("dotted greenfield must defer to the follow-up port");
        assert!(
            err.to_string().contains("dotted greenfield"),
            "expected dotted-greenfield diagnostic: {err}"
        );
    }

    /// rlib/jit.py:916-921 — after the cache update, the marker writes
    /// `graph.func._dont_reach_me_in_del_ = True` on the graph
    /// reachable from `bookkeeper.position_key`. The Rust port mirrors
    /// that side-effect on `GraphFunc::_dont_reach_me_in_del_`.
    #[test]
    fn marker_sets_dont_reach_me_in_del_on_position_key_graph() {
        use crate::annotator::bookkeeper::PositionKey;
        use crate::flowspace::model::{Block, ConstValue, Constant, FunctionGraph, GraphFunc};
        use std::cell::RefCell;
        use std::rc::Rc as StdRc;

        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let mut graph_func = GraphFunc::new("pkg.dont_reach_me", globals);
        assert!(
            !graph_func._dont_reach_me_in_del_,
            "default GraphFunc starts without the dont-reach flag"
        );
        let startblock = Block::shared(vec![]);
        let mut graph = FunctionGraph::new("dont_reach_me", startblock.clone());
        graph_func._dont_reach_me_in_del_ = false;
        graph.func = Some(graph_func);
        let graph_rc = StdRc::new(RefCell::new(graph));

        let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let pk = PositionKey::from_refs(&graph_rc, &startblock, 0);
        bk.set_position_key(Some(pk));

        let meta = meta_for("DontReachDriver", &["pc"], &["frame"]);
        let entry = ExtRegistryEntry::EnterLeaveMarker {
            meta,
            marker_kind: JitMarkerKind::CanEnterJit,
        };
        let kwds: HashMap<String, Option<SomeValue>> = [integer_kwd("pc"), integer_kwd("frame")]
            .into_iter()
            .collect();
        entry
            .compute_annotation_with_kwds(&bk, &kwds)
            .expect("dispatch with position_key set");

        assert!(
            graph_rc
                .borrow()
                .func
                .as_ref()
                .expect("graph has func")
                ._dont_reach_me_in_del_,
            "marker must set _dont_reach_me_in_del_ on the position_key graph"
        );
    }
}
