//! RPython `rpython/rtyper/rmodel.py` — Repr base class + setup
//! state machine + inputconst / mangle helpers + VoidRepr /
//! SimplePointerRepr leaves.
//!
//! ## Scope of this scaffold
//!
//! Upstream rmodel.py is 474 LOC. This port lands the subset that
//! `rpbc.py FunctionReprBase` / `rclass.py ClassRepr` directly consume:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `setupstate` (`rmodel.py`) | [`setupstate`] enum |
//! | `class Repr(object)` (`rmodel.py`) | [`Repr`] trait + [`ReprState`] state helper |
//! | `class VoidRepr(Repr)` + `impossible_repr` (`rmodel.py`) | [`VoidRepr`] + [`impossible_repr`] |
//! | `class SimplePointerRepr(Repr)` (`rmodel.py`) | [`SimplePointerRepr`] |
//! | `def inputconst(reqtype, value)` (`rmodel.py`) | [`inputconst`] |
//! | `def mangle(prefix, name)` (`rmodel.py`) | [`mangle`] |
//! | `class BrokenReprTyperError(TyperError)` (`rmodel.py`) | already ported in [`error::TyperError::BrokenRepr`][crate::translator::rtyper::error::TyperError::BrokenRepr] |
//!
//! ## Deferred to follow-up commits
//!
//! * Concrete `rtype_*` overrides that take a `HighLevelOp` (e.g.
//!   `PtrRepr.rtype_getattr`, `PtrRepr.rtype_simple_call`, ...) land
//!   with their matching `r*.py` ports. The base `Repr` missing-operation
//!   surface is present here.
//! * `Repr.__getattr__` autosetup side effect (`rmodel.py`) —
//!   Rust has no `__getattr__`. Callers that previously relied on it
//!   must invoke [`Repr::setup`] explicitly before reading derived
//!   fields; this matches upstream's own `setup()` call sequencing in
//!   `rtyper.py:call_all_setups` (`:241`).
//! * `CanBeNull.rtype_bool` ports as [`CanBeNull`] plus the shared
//!   [`can_be_null_rtype_bool`] helper; each upstream `CanBeNull`
//!   inheritor overrides [`Repr::rtype_bool`] and dispatches there.
//! * `IteratorRepr` is represented as a marker trait over the iterator
//!   methods already carried on [`Repr`].
//! * `pairtype(Repr, Repr)` default conversions (`rmodel.py`) —
//!   upstream's double-dispatch mechanism lands with the conversion
//!   table port.
//!
//! ## Parity checkpoints
//!
//! The full dispatch chain upstream is:
//!
//! ```text
//! rtyper.specialize()                           rtyper.py
//!   → specialize_more_blocks()                   rtyper.py
//!     → specialize_block(block)                  rtyper.py
//!       → for hop in highlevelops(...):           rtyper.py
//!           translate_hl_to_ll(hop)               → rtyper.py:translate_op_*
//!             → Repr.rtype_simple_call(hop)       ← pyre lands this in 3.2
//!                → FunctionReprBase.call(hop)     ← pyre lands this in 3.3
//!                   → emit direct_call            ← already backed by jtransform
//! ```
//!
//! This commit scaffolds the bottom of the chain (`Repr` base + leaves)
//! so follow-ups can land in order without retrofitting infrastructure.

#![allow(private_interfaces)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, ThreadId};

use crate::annotator::model::{KnownType, SomeValue};
use crate::flowspace::model::{ConstValue, Constant, Hlvalue};

/// RPython `description.Desc | flowspace.model.Constant` union used
/// by [`Repr::convert_desc_or_const`] (rmodel.py). The
/// concrete Python union is open — upstream accepts any `Desc`
/// subclass or a `Constant` value; the Rust port encodes that via
/// [`DescEntry`] (pointer-id wrapped `Desc`-subclass) and a borrowed
/// [`Constant`].
#[derive(Clone, Debug)]
pub enum DescOrConst {
    /// upstream `description.Desc`-subclass branch.
    Desc(crate::annotator::description::DescEntry),
    /// upstream `flowspace.model.Constant` branch.
    Const(Constant),
}
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::llannotation::lltype_to_annotation;
use crate::translator::rtyper::lltypesystem::lltype::{self, LowLevelType};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, RPythonTyper};

// RPython defines `externalvsinternal` in rmodel.py and lazily imports rclass.
// The implementation lives beside InstanceRepr in rclass.rs; keep the upstream
// module surface here while the Rust implementation stays crate-local.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn externalvsinternal(
    rtyper: &std::rc::Rc<RPythonTyper>,
    item_repr: Arc<dyn Repr>,
    gcref: bool,
) -> Result<(Arc<dyn Repr>, Arc<dyn Repr>), TyperError> {
    crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, gcref)
}

/// Result shape returned by `Repr.rtype_*` methods.
///
/// RPython returns either a low-level `Variable` / `Constant` or `None`.
/// Rust carries both variable and constant cases as [`Hlvalue`].
pub type RTypeResult = Result<Option<Hlvalue>, TyperError>;

fn hop_r_result(hop: &HighLevelOp) -> Result<Arc<dyn Repr>, TyperError> {
    hop.r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))
}

fn hop_arg_const_string(hop: &HighLevelOp, index: usize) -> Result<String, TyperError> {
    let args_s = hop.args_s.borrow();
    let value = args_s
        .get(index)
        .and_then(|s| s.const_())
        .ok_or_else(|| TyperError::message("expected constant string annotation"))?;
    let Some(attr) = value.as_text() else {
        return Err(TyperError::message(format!(
            "expected constant string annotation, got {value:?}"
        )));
    };
    Ok(attr.to_string())
}

fn void_field_const(name: &str) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::byte_str(name),
        LowLevelType::Void,
    ))
}

fn hlvalue_concretetype(value: &Hlvalue) -> Result<LowLevelType, TyperError> {
    match value {
        Hlvalue::Variable(v) => v
            .concretetype()
            .ok_or_else(|| TyperError::message("Variable has no concretetype")),
        Hlvalue::Constant(c) => c
            .concretetype
            .clone()
            .ok_or_else(|| TyperError::message("Constant has no concretetype")),
    }
}

fn ptr_type(ptr: &LowLevelType) -> &lltype::Ptr {
    let LowLevelType::Ptr(ptr) = ptr else {
        panic!("expected Ptr lowleveltype, got {ptr:?}");
    };
    ptr
}

fn ptr_target_struct(target: &lltype::PtrTarget) -> Option<lltype::Struct> {
    match target {
        lltype::PtrTarget::Struct(t) => Some(t.clone()),
        lltype::PtrTarget::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::Struct(t)) => Some(*t),
            _ => None,
        },
        _ => None,
    }
}

fn ptr_target_func(target: &lltype::PtrTarget) -> Option<lltype::FuncType> {
    match target {
        lltype::PtrTarget::Func(t) => Some(t.clone()),
        lltype::PtrTarget::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::Func(t)) => Some(*t),
            _ => None,
        },
        _ => None,
    }
}

fn ptr_target_field_type(target: &lltype::PtrTarget, attr: &str) -> Option<LowLevelType> {
    ptr_target_struct(target).and_then(|struct_t| struct_t.getattr_field_type(attr))
}

fn ptr_target_array_item_type(target: &lltype::PtrTarget) -> Option<LowLevelType> {
    match target {
        lltype::PtrTarget::Array(t) => Some(t.OF.clone()),
        lltype::PtrTarget::FixedSizeArray(t) => Some(t.OF.clone()),
        lltype::PtrTarget::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::Array(t)) => Some(t.OF.clone()),
            Some(LowLevelType::FixedSizeArray(t)) => Some(t.OF.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn ptr_target_fixed_length(target: &lltype::PtrTarget) -> Option<usize> {
    match target {
        lltype::PtrTarget::FixedSizeArray(t) => Some(t.length),
        lltype::PtrTarget::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::FixedSizeArray(t)) => Some(t.length),
            _ => None,
        },
        _ => None,
    }
}

fn lowlevel_array_item_type(container: &LowLevelType) -> Option<LowLevelType> {
    match container {
        LowLevelType::Array(t) => Some(t.OF.clone()),
        LowLevelType::FixedSizeArray(t) => Some(t.OF.clone()),
        LowLevelType::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::Array(t)) => Some(t.OF.clone()),
            Some(LowLevelType::FixedSizeArray(t)) => Some(t.OF.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn lowlevel_container_field_type(container: &LowLevelType, attr: &str) -> Option<LowLevelType> {
    match container {
        LowLevelType::Struct(t) => t.getattr_field_type(attr),
        LowLevelType::ForwardReference(t) => match t.resolved() {
            Some(LowLevelType::Struct(t)) => t.getattr_field_type(attr),
            _ => None,
        },
        _ => None,
    }
}

/// Build a `Constant` carrying a [`LowLevelType`] payload — the
/// `cTEMP` argument shape used by `malloc` / `malloc_varsize`
/// SpaceOperations (and other ops that take a struct/array type as
/// their first arg). Mirrors upstream `inputconst(Void, TYPE)`
/// (rmodel.py).
pub(crate) fn lowlevel_type_const(lltype: LowLevelType) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(lltype)),
        LowLevelType::Void,
    ))
}

/// Build a `Constant` carrying the `{'flavor': 'gc'}` flags dict —
/// the `cflags` argument shape used by `malloc` / `malloc_varsize`
/// SpaceOperations. Mirrors upstream `inputconst(Void, {'flavor':
/// 'gc'})` (rmodel.py, rstr.py:1130).  allow-line-citation
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn gc_flavor_const() -> Result<Hlvalue, TyperError> {
    let flags = HashMap::from([(ConstValue::byte_str("flavor"), ConstValue::byte_str("gc"))]);
    HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::Dict(flags)).map(Hlvalue::Constant)
}

fn rtype_ptr_comparison(r_ptr: &dyn Repr, hop: &HighLevelOp, opname: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::from(r_ptr), ConvertedTo::from(r_ptr)])?;
    Ok(hop.genop(opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

/// RPython `setupstate` (`rmodel.py`).
///
/// ```python
/// class setupstate(object):
///     NOTINITIALIZED = 0
///     INPROGRESS = 1
///     BROKEN = 2
///     FINISHED = 3
///     DELAYED = 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum setupstate {
    /// Initial state; [`Repr::setup`] still has to run.
    NotInitialized = 0,
    /// Inside an active [`Repr::_setup_repr`] call. Re-entry is an
    /// `AssertionError` upstream (`rmodel.py`).
    InProgress = 1,
    /// `_setup_repr()` raised `TyperError`; subsequent `setup()` calls
    /// re-raise `BrokenReprTyperError` (`rmodel.py`).
    Broken = 2,
    /// `_setup_repr()` returned normally; fields are ready to read.
    Finished = 3,
    /// `setup()` deferred pending an outer-pass pre-registration
    /// (`rmodel.py`).
    Delayed = 4,
}

impl setupstate {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => setupstate::NotInitialized,
            1 => setupstate::InProgress,
            2 => setupstate::Broken,
            3 => setupstate::Finished,
            4 => setupstate::Delayed,
            _ => unreachable!("invalid setupstate u8={value}"),
        }
    }
}

/// RPython `Repr._initialized` field (`rmodel.py`).
///
/// Concrete Repr types embed a `ReprState` alongside their own fields.
/// Interior mutability uses [`AtomicU8`] so Repr instances stored in
/// `OnceLock`-backed singletons (`impossible_repr`, per-lltype caches)
/// can survive Rust's `Sync` bound. Upstream Python is single-threaded
/// so the state machine assumes serial calls; pyre's parallel test
/// harness exercises the singletons concurrently, so [`Repr::setup`]
/// uses CAS to claim `InProgress` and tracks the owning [`ThreadId`]
/// in [`ReprState::owner`] to keep RPython's same-thread recursion
/// panic (`rmodel.py`) while letting other threads wait.
#[derive(Debug)]
pub struct ReprState {
    initialized: AtomicU8,
    /// `Some(thread)` while `setup()` runs on this Repr; `None`
    /// otherwise. Read under the implicit happens-before of
    /// [`AtomicU8::compare_exchange`] on `initialized` — observers
    /// that see `InProgress` are guaranteed to see the matching
    /// owner store.
    owner: Mutex<Option<ThreadId>>,
}

impl ReprState {
    /// Construct a fresh state in [`setupstate::NotInitialized`].
    pub fn new() -> Self {
        ReprState {
            initialized: AtomicU8::new(setupstate::NotInitialized as u8),
            owner: Mutex::new(None),
        }
    }

    /// Current state (read).
    pub fn get(&self) -> setupstate {
        setupstate::from_u8(self.initialized.load(Ordering::Acquire))
    }

    /// Force-set (write).
    pub fn set(&self, state: setupstate) {
        self.initialized.store(state as u8, Ordering::Release);
    }

    /// Test-only helper: force `InProgress` on this state and stamp
    /// the owner to the current thread. Lets the
    /// `setup_on_inprogress_panics_like_upstream_assertion` parity
    /// test simulate the "same thread already in setup" scenario the
    /// upstream `raise AssertionError` (`rmodel.py`) panics
    /// on, with pyre's CAS+thread-tracking model preserving that
    /// panic on owner==current-thread.
    #[doc(hidden)]
    pub fn _force_inprogress_for_current_thread(&self) {
        *self.owner.lock() = Some(thread::current().id());
        self.set(setupstate::InProgress);
    }
}

impl Default for ReprState {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `class Repr(object)` (`rmodel.py`).
///
/// Trait object so a `RPythonTyper.reprs: HashMap<SomeValue, Arc<dyn
/// Repr>>` (future 3.2 commit) can store heterogeneous Repr instances.
/// Most default methods match upstream's `Repr` base methods verbatim;
/// only the required fields (`lowleveltype`, state) are abstract.
///
/// The `std::any::Any` supertrait enables `&dyn Repr` → `&dyn Any`
/// upcasting for the rare pairtype paths that need concrete-type
/// introspection (e.g. `pairtype(BuiltinMethodRepr, BuiltinMethodRepr)
/// .convert_from_to` reads both sides' `methodname` + `self_repr`).
/// All concrete `Repr` impls are `'static` today so the bound is
/// satisfied implicitly.
/// `not r_list.listitem.mutated` for the foldable iterator-next selection
/// (`lltypesystem/rlist.py:462-466`): an unmutated list reads each element
/// through the PURE `getarrayitem_pure` (`ll_getitem_foldable_nonneg`). Any
/// non-list container is non-foldable. Mirrors the `rtype_getitem` `basegetitem`
/// gate (`rlist.py:255-258`).
fn list_container_foldable(s: &SomeValue) -> bool {
    matches!(s, SomeValue::List(lst) if !lst.listdef.listitem_rc().borrow().mutated)
}

pub trait Repr: Debug + std::any::Any {
    /// RPython `Repr.lowleveltype` (`rmodel.py` + each subclass).
    fn lowleveltype(&self) -> &LowLevelType;

    /// Access to the embedded [`ReprState`]. Concrete types store one
    /// and return a reference here.
    fn state(&self) -> &ReprState;

    /// RPython `Repr.__repr__` (`rmodel.py`):
    /// `return '<%s %s>' % (self.__class__.__name__, self.lowleveltype)`.
    fn repr_string(&self) -> String {
        format!(
            "<{} {}>",
            self.class_name(),
            self.lowleveltype().short_name()
        )
    }

    /// RPython `Repr.compact_repr` (`rmodel.py`):
    /// `return '%s %s' % (self.__class__.__name__.replace('Repr','R'),
    ///                    self.lowleveltype._short_name())`.
    fn compact_repr(&self) -> String {
        format!(
            "{} {}",
            self.class_name().replace("Repr", "R"),
            self.lowleveltype().short_name()
        )
    }

    /// Concrete class name (for the `__repr__` / `compact_repr`
    /// formatters and TyperError messages). Rust does not have
    /// `__class__.__name__`; each concrete Repr returns its type name
    /// here.
    fn class_name(&self) -> &'static str;

    /// Pairtype-dispatch tag. RPython uses `self.__class__` directly
    /// inside `pairtype(R_A, R_B)` metaclass lookups; Rust exposes the
    /// same identity as the explicit [`super::pairtype::ReprClassId`]
    /// enum. Default [`super::pairtype::ReprClassId::Repr`] matches
    /// upstream's `pairtype(Repr, X)` / `pairtype(X, Repr)` base-class
    /// catch-all. Every concrete Repr overrides this to return its
    /// specific tag.
    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::Repr
    }

    /// RPython `rptr.py`: `InteriorPtrRepr -> InteriorPtrRepr`
    /// conversion is allowed only when `r_from.__dict__ == r_to.__dict__`.
    /// Non-interior reprs return `None`; [`super::pairtype`] uses this
    /// attribute-shaped key for the single conversion arm that needs the
    /// full repr state instead of just `lowleveltype`.
    fn interior_ptr_repr_dict_key(&self) -> Option<lltype::InteriorPtr> {
        None
    }

    /// RPython `InstanceRepr.gcflavor` (`rclass.py`) reach-through.
    /// Returns the upstream-string form (`"gc"` / `"raw"`) for reprs
    /// that carry a GC flavor (`InstanceRepr`); `None` for every other
    /// repr.  Used by `rbuiltin.rtype_free_non_gc_object` to read the
    /// flavor without an `std::any::Any` downcast.  Default `None`
    /// matches upstream where only `InstanceRepr` (and its subclasses)
    /// expose `gcflavor` as a class attribute.
    fn gc_flavor_str(&self) -> Option<&'static str> {
        None
    }

    /// RPython `Repr.setup(self)` (`rmodel.py`).
    ///
    /// ```python
    /// def setup(self):
    ///     if self._initialized == setupstate.FINISHED:
    ///         return
    ///     elif self._initialized == setupstate.BROKEN:
    ///         raise BrokenReprTyperError(
    ///             "cannot setup already failed Repr: %r" %(self,))
    ///     elif self._initialized == setupstate.INPROGRESS:
    ///         raise AssertionError(
    ///             "recursive invocation of Repr setup(): %r" %(self,))
    ///     elif self._initialized == setupstate.DELAYED:
    ///         raise AssertionError(
    ///             "Repr setup() is delayed and cannot be called yet: %r" %(self,))
    ///     assert self._initialized == setupstate.NOTINITIALIZED
    ///     self._initialized = setupstate.INPROGRESS
    ///     try:
    ///         self._setup_repr()
    ///     except TyperError:
    ///         self._initialized = setupstate.BROKEN
    ///         raise
    ///     else:
    ///         self._initialized = setupstate.FINISHED
    /// ```
    fn setup(&self) -> Result<(), TyperError> {
        let state = self.state();
        let me = thread::current().id();
        loop {
            match state.get() {
                setupstate::Finished => return Ok(()),
                setupstate::Broken => {
                    return Err(TyperError::broken_repr(format!(
                        "cannot setup already failed Repr: {}",
                        self.repr_string()
                    )));
                }
                setupstate::InProgress => {
                    // Single-threaded RPython panics on any InProgress
                    // observation (`rmodel.py:45-47`). Pyre's parallel
                    // tests share `OnceLock`-backed singletons, so
                    // distinguish same-thread recursion (true bug)
                    // from cross-thread overlap (wait).
                    if state.owner.lock().as_ref() == Some(&me) {
                        panic!(
                            "recursive invocation of Repr setup(): {}",
                            self.repr_string()
                        );
                    }
                    thread::yield_now();
                    continue;
                }
                setupstate::Delayed => {
                    panic!(
                        "Repr setup() is delayed and cannot be called yet: {}",
                        self.repr_string()
                    );
                }
                setupstate::NotInitialized => {}
            }
            // Try to atomically claim NotInitialized → InProgress.
            // On race, another thread won; loop and re-observe.
            if state
                .initialized
                .compare_exchange(
                    setupstate::NotInitialized as u8,
                    setupstate::InProgress as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            *state.owner.lock() = Some(me);
            let result = self._setup_repr();
            *state.owner.lock() = None;
            match result {
                Ok(()) => {
                    state.set(setupstate::Finished);
                    return Ok(());
                }
                Err(e) => {
                    state.set(setupstate::Broken);
                    return Err(e);
                }
            }
        }
    }

    /// RPython `Repr._setup_repr(self)` (`rmodel.py`).
    ///
    /// Default no-op. Concrete subclasses override for recursive /
    /// two-step initialization.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `Repr.setup_final(self)` (`rmodel.py`).
    ///
    /// ```python
    /// def setup_final(self):
    ///     if self._initialized == setupstate.BROKEN:
    ///         raise BrokenReprTyperError(...)
    ///     assert self._initialized == setupstate.FINISHED
    ///     self._setup_repr_final()
    /// ```
    fn setup_final(&self) -> Result<(), TyperError> {
        let state = self.state();
        match state.get() {
            setupstate::Broken => Err(TyperError::broken_repr(format!(
                "cannot perform setup_final_touch on failed Repr: {}",
                self.repr_string()
            ))),
            setupstate::Finished => self._setup_repr_final(),
            other => panic!(
                "setup_final() on repr with state {other:?}: {}",
                self.repr_string()
            ),
        }
    }

    /// RPython `Repr._setup_repr_final(self)` (`rmodel.py`).
    /// Default no-op.
    fn _setup_repr_final(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `Repr.is_setup_delayed(self)` (`rmodel.py`).
    fn is_setup_delayed(&self) -> bool {
        matches!(self.state().get(), setupstate::Delayed)
    }

    /// RPython `Repr.set_setup_delayed(self, flag)` (`rmodel.py`).
    ///
    /// ```python
    /// def set_setup_delayed(self, flag):
    ///     assert self._initialized in (setupstate.NOTINITIALIZED,
    ///                                  setupstate.DELAYED)
    ///     if flag:
    ///         self._initialized = setupstate.DELAYED
    ///     else:
    ///         self._initialized = setupstate.NOTINITIALIZED
    /// ```
    fn set_setup_delayed(&self, flag: bool) {
        let state = self.state();
        let current = state.get();
        assert!(
            matches!(current, setupstate::NotInitialized | setupstate::Delayed),
            "set_setup_delayed requires NotInitialized/Delayed, got {current:?}"
        );
        if flag {
            state.set(setupstate::Delayed);
        } else {
            state.set(setupstate::NotInitialized);
        }
    }

    /// RPython `Repr.set_setup_maybe_delayed(self)` (`rmodel.py`).
    ///
    /// ```python
    /// def set_setup_maybe_delayed(self):
    ///     if self._initialized == setupstate.NOTINITIALIZED:
    ///         self._initialized = setupstate.DELAYED
    ///     return self._initialized == setupstate.DELAYED
    /// ```
    fn set_setup_maybe_delayed(&self) -> bool {
        let state = self.state();
        if matches!(state.get(), setupstate::NotInitialized) {
            state.set(setupstate::Delayed);
        }
        matches!(state.get(), setupstate::Delayed)
    }

    /// RPython `Repr.get_r_implfunc(self)` (`rmodel.py`).
    ///
    /// ```python
    /// def get_r_implfunc(self):
    ///     raise TyperError("%s has no corresponding implementation function representation" % (self,))
    /// ```
    ///
    /// Returns the `(r_func, nimplicitarg)` pair used by
    /// `rbuiltin.rtype_hlinvoke` (`rbuiltin.py`) to walk from an
    /// opaque callable repr to the underlying `FunctionReprBase` that
    /// exposes `get_s_signatures`. The base trait raises; concrete
    /// reprs override. `FunctionReprBase` returns `(self, 0)`.
    /// `MethodsPBCRepr` returns `(r_class.clsfields[methodname], 1)`.
    /// `MethodOfFrozenPBCRepr` returns `(getrepr(get_s_callable()), 1)`.
    fn get_r_implfunc(&self) -> Result<(std::sync::Arc<dyn Repr>, usize), TyperError> {
        Err(TyperError::message(format!(
            "{} has no corresponding implementation function representation",
            self.repr_string()
        )))
    }

    /// RPython `FunctionReprBase.s_pbc` (`rpbc.py`) read accessor.
    ///
    /// In upstream Python `r_func.s_pbc` is a plain attribute access on
    /// `FunctionRepr` / `FunctionsPBCRepr` / `SmallFunctionSetPBCRepr`
    /// (each subclasses `FunctionReprBase`). Rust composes rather than
    /// inherits (see [`super::rpbc::FunctionReprBase`]'s module doc), so
    /// the polymorphic field access becomes a `Repr` trait method.
    ///
    /// Used by callers that need a callable PBC's shape — notably
    /// `MethodsPBCRepr.redispatch_call` (rpbc.py) — to read
    /// `r_func.s_pbc` for the `subset_of=` argument when narrowing the
    /// per-call `SomePBC`.
    ///
    /// Default `None` matches upstream's `AttributeError`-raising
    /// behaviour for non-FunctionReprBase subclasses; concrete
    /// FunctionReprBase impls override to return `Some(&self.base.s_pbc)`.
    fn pbc_s_pbc(&self) -> Option<&crate::annotator::model::SomePBC> {
        None
    }

    /// RPython `Repr.get_s_callable(self)` (`rmodel.py` / `rpbc.py`).
    ///
    /// `rtype_hlinvoke` reads this, not `s_pbc`. Function reprs return
    /// their `s_pbc`. `MethodOfFrozenPBCRepr` returns the underlying
    /// function, and `MethodsPBCRepr` returns the method PBC.
    fn get_s_callable(&self) -> Option<crate::annotator::model::SomePBC> {
        self.pbc_s_pbc().cloned()
    }

    /// RPython `FunctionReprBase.get_s_signatures(self, shape)` (rpbc.py).
    ///
    /// `rtype_hlinvoke` calls this on the `Repr` returned by
    /// `get_r_implfunc`. The default is the missing-method failure;
    /// `FunctionRepr` forwards to `FunctionReprBase`.
    fn get_s_signatures(
        &self,
        _shape: &crate::flowspace::argument::CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        Err(TyperError::message(format!(
            "{} has no get_s_signatures",
            self.repr_string()
        )))
    }

    /// RPython `Repr.convert_const(self, value)` (`rmodel.py`).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     "Convert the given constant value to the low-level repr of 'self'."
    ///     if not self.lowleveltype._contains_value(value):
    ///         raise TyperError("convert_const(self = %r, value = %r)" % (
    ///             self, value))
    ///     return value
    /// ```
    ///
    /// Upstream returns the Python value directly (the `_contains_value`
    /// predicate doubles as a type cast); the pyre adaptation wraps the
    /// value in a `Constant` whose `concretetype` carries the repr's
    /// lowleveltype, matching what `inputconst` / `specialize_block`
    /// expect downstream.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !self.lowleveltype().contains_value(value) {
            return Err(TyperError::message(format!(
                "convert_const(self = {}, value = {:?})",
                self.repr_string(),
                value
            )));
        }
        Ok(Constant::with_concretetype(
            value.clone(),
            self.lowleveltype().clone(),
        ))
    }

    /// RPython `Repr.special_uninitialized_value(self)` (`rmodel.py`).
    fn special_uninitialized_value(&self) -> Option<ConstValue> {
        None
    }

    /// RPython `Repr.convert_desc(self, desc)` — not defined on the base
    /// class upstream (Python raises `AttributeError` implicitly when
    /// hit). Only PBC / class reprs override it (`rpbc.py`, `:320`,
    /// `:428`, `:647`, `:685`, `:769`, `:878`, `:950` and
    /// `rclass.py`).
    ///
    /// The Rust port surfaces the same "not supported" outcome as a
    /// structured [`TyperError::missing_rtype_operation`] so callers get
    /// a typed error rather than a panic.
    fn convert_desc(
        &self,
        _desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        Err(TyperError::missing_rtype_operation(format!(
            "convert_desc(self = {})",
            self.repr_string()
        )))
    }

    /// RPython `Repr.convert_desc_or_const(self, desc_or_const)`
    /// (`rmodel.py`).
    ///
    /// ```python
    /// def convert_desc_or_const(self, desc_or_const):
    ///     if isinstance(desc_or_const, description.Desc):
    ///         return self.convert_desc(desc_or_const)
    ///     elif isinstance(desc_or_const, Constant):
    ///         return self.convert_const(desc_or_const.value)
    ///     else:
    ///         raise TyperError("convert_desc_or_const expects a Desc"
    ///                          "or Constant: %r" % desc_or_const)
    /// ```
    ///
    /// The Rust port takes a [`DescOrConst`] union so the call site
    /// names its choice explicitly; the two arms dispatch through the
    /// Repr trait's own `convert_desc` / `convert_const` overrides.
    fn convert_desc_or_const(&self, value: &DescOrConst) -> Result<Constant, TyperError> {
        match value {
            DescOrConst::Desc(desc) => self.convert_desc(desc),
            DescOrConst::Const(c) => self.convert_const(&c.value),
        }
    }

    /// RPython `Repr.can_ll_be_null(self, s_value)` (`rmodel.py`).
    ///
    /// Default `true` (conservative) matching upstream.
    fn can_ll_be_null(&self) -> bool {
        true
    }

    /// RPython `Repr.get_ll_eq_function(self)` (`rmodel.py`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     raise TyperError('no equality function for %r' % self)
    /// ```
    ///
    /// The upstream contract is: the base default raises (signalling
    /// "this Repr does not define a structural equality"), concrete
    /// Reprs override either to return `None` (use the primitive
    /// `int_eq`/`float_eq`/... inline op via `gen_eq_function`'s
    /// `eq_funcs[i] or operator.eq` fallback) or to return a callable
    /// helper (e.g. `StringRepr.get_ll_eq_function() -> ll_streq`).
    ///
    /// `rtyper` is threaded so overrides can synthesize per-shape
    /// helpers via [`RPythonTyper::lowlevel_helper_function_with_builder`].
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Err(TyperError::message(format!(
            "no equality function for {}",
            self.repr_string()
        )))
    }

    /// RPython `Repr.get_ll_hash_function(self)` (`rmodel.py`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     raise TyperError('no hashing function for %r' % self)
    /// ```
    ///
    /// Same contract as `get_ll_eq_function`: base raises, concrete
    /// Reprs override (e.g. `IntegerRepr → ll_hash_int`, `NoneRepr →
    /// ll_none_hash`, `StringRepr → ll_strhash`). Unlike eq, there is
    /// no `or _ll_equal` fallback at the caller site — every item Repr
    /// participating in a hashed container must define its hash helper.
    fn get_ll_hash_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Err(TyperError::message(format!(
            "no hashing function for {}",
            self.repr_string()
        )))
    }

    /// RPython `Repr.get_ll_fasthash_function(self)` (`rmodel.py`).
    /// Base returns `None` — hash should be cached in the dict entry.
    /// Concrete Reprs that alias `get_ll_hash_function` override this.
    fn get_ll_fasthash_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `make_missing_op(Repr, opname)` (`rmodel.py`).
    fn missing_rtype_operation(&self, opname: &str) -> TyperError {
        TyperError::missing_rtype_operation(format!(
            "unimplemented operation: '{opname}' on {}",
            self.repr_string()
        ))
    }

    /// RPython `Repr.rtype_getattr(self, hop)` (`rmodel.py`):
    ///
    /// ```python
    /// if s_attr.is_constant() and isinstance(s_attr.const, str):
    ///     attr = s_attr.const
    ///     ...
    /// else:
    ///     raise TyperError("getattr() with a non-constant attribute name")
    /// ```
    ///
    /// `isinstance(s_attr.const, str)` is Python 2 bytes-only — narrow
    /// with [`ConstValue::as_pystr`] so a unicode attribute literal
    /// surfaces the upstream `non-constant attribute name` error
    /// instead of dispatching as if it were a Python 2 `str`.
    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let args_s = hop.args_s.borrow();
        let s_attr = args_s
            .get(1)
            .ok_or_else(|| TyperError::message("getattr() missing attribute argument"))?;
        if let Some(attr) = s_attr.const_().and_then(ConstValue::as_pystr) {
            let s_obj = args_s
                .first()
                .ok_or_else(|| TyperError::message("getattr() missing object argument"))?;
            if s_obj.find_method(attr).is_none() {
                return Err(TyperError::message(format!(
                    "no method {attr} on {s_obj:?}"
                )));
            }
            drop(args_s);
            let v = hop
                .args_v
                .borrow()
                .first()
                .cloned()
                .ok_or_else(|| TyperError::message("getattr() missing object argument"))?;
            if let Hlvalue::Constant(c) = &v {
                return inputconst(self, &c.value).map(|c| Some(Hlvalue::Constant(c)));
            }
            if let Some(value) = hop
                .args_s
                .borrow()
                .first()
                .and_then(|s| s.const_())
                .cloned()
            {
                return inputconst(self, &value).map(|c| Some(Hlvalue::Constant(c)));
            }
            return Ok(Some(v));
        }
        Err(TyperError::message(
            "getattr() with a non-constant attribute name",
        ))
    }

    fn rtype_setattr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("setattr"))
    }

    fn rtype_len(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("len"))
    }

    fn rtype_getitem(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("getitem"))
    }

    fn rtype_setitem(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("setitem"))
    }

    fn rtype_delitem(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("delitem"))
    }

    fn rtype_eq(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("eq"))
    }

    fn rtype_ne(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("ne"))
    }

    /// RPython `Repr.rtype_bool(self, hop)` (`rmodel.py`).
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        match self.rtype_len(hop) {
            Ok(Some(vlen)) => Ok(hop.genop(
                "int_is_true",
                vec![vlen],
                GenopResult::LLType(LowLevelType::Bool),
            )),
            Ok(None) => Err(TyperError::message(format!(
                "rtype_bool({}) returned no length value",
                self.repr_string()
            ))),
            Err(err) if err.is_missing_rtype_operation() => {
                let s_result = hop.s_result.borrow();
                if let Some(s_result) = &*s_result
                    && let Some(value) = s_result.const_()
                {
                    return HighLevelOp::inputconst(&LowLevelType::Bool, value)
                        .map(|c| Some(Hlvalue::Constant(c)));
                }
                Err(TyperError::message(format!(
                    "rtype_bool({}) not implemented",
                    self.repr_string()
                )))
            }
            Err(err) => Err(err),
        }
    }

    fn rtype_simple_call(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("simple_call"))
    }

    // arithmetic / conversion missing-op defaults
    //
    // RPython `rmodel.py:342` registers missing-op stubs for
    // `setattr len contains iter` on the base `Repr`. The pyre port
    // pre-registers the wider slot set that concrete Reprs (`FloatRepr`,
    // `IntegerRepr`, …) override — adding a slot is source-compatible
    // whereas RPython can `setattr` at runtime via `make_missing_op`.
    // Each default raises the same `MissingRTypeOperation` upstream's
    // `make_missing_op` closure does.

    fn rtype_neg(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("neg"))
    }

    fn rtype_neg_ovf(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("neg_ovf"))
    }

    fn rtype_pos(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("pos"))
    }

    fn rtype_abs(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("abs"))
    }

    fn rtype_abs_ovf(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("abs_ovf"))
    }

    fn rtype_int(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("int"))
    }

    fn rtype_float(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("float"))
    }

    fn rtype_invert(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("invert"))
    }

    fn rtype_bin(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("bin"))
    }

    fn rtype_call_args(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("call_args"))
    }

    fn rtype_delattr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("delattr"))
    }

    fn rtype_delslice(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("delslice"))
    }

    fn rtype_getslice(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("getslice"))
    }

    fn rtype_hash(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("hash"))
    }

    fn rtype_hex(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("hex"))
    }

    fn rtype_hint(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("hint"))
    }

    fn rtype_id(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("id"))
    }

    fn rtype_isinstance(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("isinstance"))
    }

    fn rtype_issubtype(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("issubtype"))
    }

    /// RPython `Repr.rtype_iter(self, hop)` (rmodel.py):
    ///
    /// ```python
    /// def rtype_iter(self, hop):
    ///     r_iter = self.make_iterator_repr()
    ///     return r_iter.newiter(hop)
    /// ```
    ///
    /// `foldable` (`not r_list.listitem.mutated`) is read off the container
    /// annotation so an unmutated list's iterator can fold the element load
    /// (`lltypesystem/rlist.py`); it is irrelevant on this path (only
    /// `newiter` runs) but kept consistent with the `next`-op repr.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let foldable = hop
            .args_s
            .borrow()
            .first()
            .map(list_container_foldable)
            .unwrap_or(false);
        let r_iter = self.make_iterator_repr(&[], foldable)?;
        r_iter.newiter(hop)
    }

    /// RPython `Repr.make_iterator_repr(self, *variant)` (rmodel.py):
    /// the base raises a `TyperError`; container reprs (list, range, dict,
    /// str) override to mint their iterator repr. `foldable` selects the
    /// foldable element-load helper for an unmutated list iterator.
    fn make_iterator_repr(
        &self,
        _variant: &[String],
        _foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        Err(self.missing_rtype_operation("make_iterator_repr"))
    }

    /// RPython `IteratorRepr.newiter(self, hop)` — only iterator reprs
    /// implement it (the `ll_*iter` constructor `gendirectcall`); the base
    /// raises.
    fn newiter(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("newiter"))
    }

    fn rtype_long(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("long"))
    }

    fn rtype_next(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("next"))
    }

    fn rtype_oct(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("oct"))
    }

    fn rtype_ord(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("ord"))
    }

    fn rtype_repr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("repr"))
    }

    fn rtype_setslice(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("setslice"))
    }

    fn rtype_str(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("str"))
    }

    fn rtype_type(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("type"))
    }

    fn rtype_chr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("chr"))
    }

    /// RPython `Repr.rtype_unicode(self, hop)` — default routes to the
    /// `MissingRTypeOperation` path. Concrete reprs (e.g. `StringRepr`,
    /// `UnicodeRepr`) override. Used by `@typer_for(unicode)` in
    /// rbuiltin.py.
    fn rtype_unicode(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("unicode"))
    }

    /// RPython `Repr.rtype_bytearray(self, hop)` — default routes to
    /// the `MissingRTypeOperation` path. Used by `@typer_for(bytearray)`
    /// in rbuiltin.py.
    fn rtype_bytearray(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("bytearray"))
    }

    /// RPython `Repr.rtype_bltn_list(self, hop)` — default routes to
    /// the `MissingRTypeOperation` path. The `_bltn` infix
    /// distinguishes the `list(x)` builtin from `x.list()`. Used by
    /// `@typer_for(list)` in rbuiltin.py.
    fn rtype_bltn_list(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(self.missing_rtype_operation("bltn_list"))
    }

    /// RPython `Repr.rtype_unichr(self, hop)` (`rmodel.py`).
    fn rtype_unichr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message(format!(
            "no unichr() support for {}",
            self.repr_string()
        )))
    }

    /// RPython `Repr._freeze_(self)` (`rmodel.py`).
    /// Always true — Repr instances are immutable once created.
    fn freeze(&self) -> bool {
        true
    }

    /// Dispatch table for `rtype_method_<method_name>`.
    ///
    /// Upstream's `BuiltinMethodRepr.rtype_simple_call` (rbuiltin.py:
    /// 125-132) performs a Python-level `getattr(self.self_repr,
    /// 'rtype_method_' + methodname)`. The Rust port materialises the
    /// same lookup as a trait method that every concrete `Repr`
    /// overrides to route each supported `methodname`. The default
    /// surface here raises the same `TyperError` upstream's
    /// `AttributeError` branch does.
    fn rtype_method(&self, method_name: &str, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message(format!(
            "missing {}.rtype_method_{method_name}",
            self.class_name()
        )))
    }
}

/// RPython `inputconst(reqtype, value)` (`rmodel.py`).
///
/// Upstream supports `reqtype` as either `Repr` or `LowLevelType`.
/// Pyre splits into two functions because Rust's type system prefers
/// explicit dispatch over Python-style duck typing. Use
/// [`inputconst_from_lltype`] when the caller already has a bare
/// `LowLevelType`.
///
/// ```python
/// def inputconst(reqtype, value):
///     if isinstance(reqtype, Repr):
///         value = reqtype.convert_const(value)
///         lltype = reqtype.lowleveltype
///     elif isinstance(reqtype, LowLevelType):
///         lltype = reqtype
///     else:
///         raise TypeError(repr(reqtype))
///     if not lltype._contains_value(value):
///         raise TyperError("inputconst(): expected a %r, got %r" %
///                          (lltype, value))
///     c = Constant(value)
///     c.concretetype = lltype
///     return c
/// ```
pub fn inputconst<R: Repr + ?Sized>(
    reqtype: &R,
    value: &ConstValue,
) -> Result<Constant, TyperError> {
    let c = reqtype.convert_const(value)?;
    // `convert_const` already populated `concretetype`; double-check
    // contains_value as upstream does post-convert.
    if !reqtype.lowleveltype().contains_value(&c.value) {
        return Err(TyperError::message(format!(
            "inputconst(): expected a {}, got {:?}",
            reqtype.lowleveltype().short_name(),
            c.value
        )));
    }
    Ok(c)
}

/// RPython `inputconst(LowLevelType, value)` overload (`rmodel.py`).
pub fn inputconst_from_lltype(
    lltype: &LowLevelType,
    value: &ConstValue,
) -> Result<Constant, TyperError> {
    if !lltype.contains_value(value) {
        return Err(TyperError::message(format!(
            "inputconst(): expected a {}, got {:?}",
            lltype.short_name(),
            value
        )));
    }
    Ok(Constant::with_concretetype(value.clone(), lltype.clone()))
}

/// RPython `getgcflavor(classdef)` (`rmodel.py`).
///
/// `rclass.py` carries the same helper in the Rust port because
/// `InstanceRepr` consumes it directly; keep the `rmodel.py` spelling
/// as the public parity surface and delegate to that implementation.
pub fn getgcflavor(
    classdef: &std::rc::Rc<std::cell::RefCell<crate::annotator::classdesc::ClassDef>>,
) -> Result<super::rclass::Flavor, TyperError> {
    super::rclass::getgcflavor(classdef)
}

/// RPython `mangle(prefix, name)` (`rmodel.py`).
///
/// ```python
/// def mangle(prefix, name):
///     """Make a unique identifier from the prefix and the name.  The name
///     is allowed to start with $."""
///     if name.startswith('$'):
///         return '%sinternal_%s' % (prefix, name[1:])
///     else:
///         return '%s_%s' % (prefix, name)
/// ```
pub fn mangle(prefix: &str, name: &str) -> String {
    if let Some(stripped) = name.strip_prefix('$') {
        format!("{prefix}internal_{stripped}")
    } else {
        format!("{prefix}_{name}")
    }
}

/// RPython `class DummyValueBuilder(object)` (`rmodel.py`).
///
/// The lazy `ll_dummy_value` allocation (`rmodel.py`) is ported:
/// it mallocs one immortal placeholder per `TYPE` into
/// `RPythonTyper.cache_dummy_values`. Because pyre stores only `rtyper_id`
/// for identity (not the typer), the `@property` becomes a method that
/// receives the `&RPythonTyper` at call time.
///
/// The producer side — `Repr.get_ll_dummyval_obj` (`rmodel.py`)
/// returning this builder — is still deferred: its only consumers are the
/// dict/ordereddict `ENTRIES.dummy_obj` fields (`rdict.py:109-111`,
/// `rordereddict.py:223-225`), which are not yet ported.
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct DummyValueBuilder {
    rtyper_id: usize,
    TYPE: LowLevelType,
}

#[allow(non_snake_case)]
impl DummyValueBuilder {
    pub fn new(rtyper: &RPythonTyper, TYPE: LowLevelType) -> Self {
        DummyValueBuilder {
            rtyper_id: rtyper as *const RPythonTyper as usize,
            TYPE,
        }
    }

    pub fn rtyper_id(&self) -> usize {
        self.rtyper_id
    }

    pub fn TYPE(&self) -> &LowLevelType {
        &self.TYPE
    }

    pub fn _freeze_(&self) -> bool {
        true
    }

    /// RPython `DummyValueBuilder.ll_dummy_value` (`rmodel.py`).
    ///
    /// The `@property` reads `self.rtyper`; pyre stores only `rtyper_id`
    /// for identity, so the typer is threaded in at call time.
    pub fn ll_dummy_value(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<lltype::LowLevelValue, TyperError> {
        if let Some(p) = rtyper.cache_dummy_values.borrow().get(&self.TYPE) {
            return Ok(p.clone());
        }
        // generate a dummy ptr to an immortal placeholder struct/array
        let n = if self.TYPE._is_varsize() {
            Some(1)
        } else {
            None
        };
        let p = lltype::malloc(self.TYPE.clone(), n, lltype::MallocFlavor::Gc, true)
            .map_err(TyperError::message)?;
        let p = lltype::LowLevelValue::Ptr(Box::new(p));
        rtyper
            .cache_dummy_values
            .borrow_mut()
            .insert(self.TYPE.clone(), p.clone());
        Ok(p)
    }
}

impl PartialEq for DummyValueBuilder {
    fn eq(&self, other: &Self) -> bool {
        self.rtyper_id == other.rtyper_id && self.TYPE == other.TYPE
    }
}

impl Eq for DummyValueBuilder {}

impl Hash for DummyValueBuilder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.TYPE.hash(state);
    }
}

/// RPython `warning(msg)` (`rmodel.py`).
pub fn warning(msg: &str) {
    eprintln!("rtyper WARNING: {msg}");
}

/// RPython `class CanBeNull(object).rtype_bool(self, hop)`
/// (`rmodel.py`).
///
/// ```python
/// class CanBeNull(object):
///     """A mix-in base class for subclasses of Repr that represent None as
///     'null' and true values as non-'null'.
///     """
///     def rtype_bool(self, hop):
///         if hop.s_result.is_constant():
///             return hop.inputconst(Bool, hop.s_result.const)
///         else:
///             vlist = hop.inputargs(self)
///             return hop.genop('ptr_nonzero', vlist, resulttype=Bool)
/// ```
///
/// Python surfaces `CanBeNull` as a mixin class; every `Repr` subclass
/// that inherits from it gets this single method. Rust has no mixins so
/// the body lands as a free helper; each CanBeNull-inheriting `Repr`
/// overrides [`Repr::rtype_bool`] and dispatches here with `self` as
/// the receiver. The constant fast-path is what distinguishes this
/// from the plain `PtrRepr.rtype_bool` (`rmodel.py`) — CanBeNull
/// reprs may live in the Void space (e.g. `FunctionRepr`) where a
/// constant pyobj forces `is_constant()` true.
pub fn can_be_null_rtype_bool(
    r: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> RTypeResult {
    // upstream: `if hop.s_result.is_constant(): return
    //     hop.inputconst(Bool, hop.s_result.const)`.
    let s_const = hop
        .s_result
        .borrow()
        .as_ref()
        .and_then(|s| s.const_())
        .cloned();
    if let Some(value) = s_const {
        let c = inputconst_from_lltype(&LowLevelType::Bool, &value)?;
        return Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)));
    }
    // upstream: `vlist = hop.inputargs(self); return
    //     hop.genop('ptr_nonzero', vlist, resulttype=Bool)`.
    let vlist = hop.inputargs(vec![crate::translator::rtyper::rtyper::ConvertedTo::Repr(
        r,
    )])?;
    Ok(hop.genop(
        "ptr_nonzero",
        vlist,
        crate::translator::rtyper::rtyper::GenopResult::LLType(LowLevelType::Bool),
    ))
}

/// RPython `class CanBeNull(object)` (rmodel.py).
pub trait CanBeNull: Repr {
    fn rtype_bool_can_be_null(&self, hop: &HighLevelOp) -> RTypeResult
    where
        Self: Sized,
    {
        can_be_null_rtype_bool(self, hop)
    }
}

/// RPython `class IteratorRepr(Repr)` (rmodel.py).
///
/// Rust keeps the actual `rtype_iter`, `rtype_method_next` equivalent, and
/// `newiter` hooks on [`Repr`] so all reprs share one dispatch surface; this
/// marker preserves the upstream base-class name for iterator reprs.
pub trait IteratorRepr: Repr {}

// ____________________________________________________________
// Concrete Repr leaves that every downstream port needs.

/// RPython `class VoidRepr(Repr)` (`rmodel.py`).
///
/// ```python
/// class VoidRepr(Repr):
///     lowleveltype = Void
///     def get_ll_eq_function(self): return None
///     def get_ll_hash_function(self): return ll_hash_void
///     get_ll_fasthash_function = get_ll_hash_function
///     def ll_str(self, nothing): raise AssertionError("unreachable code")
/// impossible_repr = VoidRepr()
/// ```
#[derive(Debug)]
pub struct VoidRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl VoidRepr {
    pub fn new() -> Self {
        VoidRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }
}

impl Default for VoidRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for VoidRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "VoidRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::VoidRepr
    }

    /// rmodel.py `def get_ll_eq_function(self): return None`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// rmodel.py `def get_ll_hash_function(self): return ll_hash_void`
    /// where `ll_hash_void(v): return 0` (rmodel.py).
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        rtyper
            .lowlevel_helper_function_with_builder(
                "ll_hash_void".to_string(),
                vec![LowLevelType::Void],
                LowLevelType::Signed,
                |_rtyper, args, _result| build_ll_hash_void_helper_graph("ll_hash_void", &args[0]),
            )
            .map(Some)
    }

    /// rmodel.py `get_ll_fasthash_function = get_ll_hash_function`.
    fn get_ll_fasthash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_hash_function(rtyper)
    }
}

/// RPython `ll_hash_void(v)` (rmodel.py).
pub fn ll_hash_void<T>(_v: T) -> i64 {
    0
}

/// rmodel.py `def ll_hash_void(v): return 0` — single-block
/// helper graph taking one `Void` arg and returning the `Signed`
/// constant `0`.
fn build_ll_hash_void_helper_graph(
    name: &str,
    _arg_type: &LowLevelType,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, BlockRefExt, FunctionGraph, GraphFunc, Link};
    use crate::translator::rtyper::rtyper::{helper_pygraph_from_graph, variable_with_lltype};

    let arg = variable_with_lltype("v", LowLevelType::Void);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg)]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let c_zero = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Int(0),
        LowLevelType::Signed,
    ));
    startblock.closeblock(vec![
        Link::new(vec![c_zero], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["v".to_string()],
        func,
    ))
}

/// RPython `impossible_repr = VoidRepr()` (`rmodel.py`) — the
/// singleton VoidRepr used for `SomeImpossibleValue` (`rmodel.py+`).
///
/// Rust returns an [`Arc<VoidRepr>`] clone of the cached singleton so
/// the pointer identity upstream relies on (`r is impossible_repr`) is
/// preserved through `Arc::ptr_eq`.
pub fn impossible_repr() -> std::sync::Arc<VoidRepr> {
    static REPR: OnceLock<std::sync::Arc<VoidRepr>> = OnceLock::new();
    REPR.get_or_init(|| std::sync::Arc::new(VoidRepr::new()))
        .clone()
}

/// RPython `class SimplePointerRepr(Repr)` (`rmodel.py`).
///
/// ```python
/// class SimplePointerRepr(Repr):
///     "Convenience Repr for simple ll pointer types with no operation on them."
///
///     def __init__(self, lowleveltype):
///         self.lowleveltype = lowleveltype
///
///     def convert_const(self, value):
///         if value is not None:
///             raise TyperError("%r only supports None as prebuilt constant, "
///                              "got %r" % (self, value))
///         return lltype.nullptr(self.lowleveltype.TO)
/// ```
#[derive(Debug)]
pub struct SimplePointerRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl SimplePointerRepr {
    /// `lowleveltype` must be a [`LowLevelType::Ptr`] variant upstream
    /// (`rmodel.py SimplePointerRepr`); enforce via debug assertion.
    pub fn new(lowleveltype: LowLevelType) -> Self {
        debug_assert!(
            matches!(lowleveltype, LowLevelType::Ptr(_)),
            "SimplePointerRepr requires Ptr lowleveltype, got {lowleveltype:?}"
        );
        SimplePointerRepr {
            state: ReprState::new(),
            lltype: lowleveltype,
        }
    }
}

impl Repr for SimplePointerRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SimplePointerRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::SimplePointerRepr
    }

    /// RPython override (`rmodel.py convert_const`): only accept `None`, emit
    /// `nullptr(self.lowleveltype.TO)`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !matches!(value, ConstValue::None) {
            return Err(TyperError::message(format!(
                "{} only supports None as prebuilt constant, got {value:?}",
                self.repr_string()
            )));
        }
        // upstream returns `lltype.nullptr(self.lowleveltype.TO)`; pyre
        // stores the null sentinel via `ConstValue::None` with the
        // Ptr lowleveltype on the Constant, matching what downstream
        // emit_const_r pipes through.
        Ok(Constant::with_concretetype(
            ConstValue::None,
            self.lltype.clone(),
        ))
    }
}

/// RPython `raddress.py` — `class AddressRepr(Repr)`.
/// Singleton repr for `SomeAddress`, lowleveltype = Address.
#[derive(Debug)]
pub struct AddressRepr {
    state: ReprState,
}

impl AddressRepr {
    fn new() -> Self {
        AddressRepr {
            state: ReprState::new(),
        }
    }
}

impl Repr for AddressRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &LowLevelType::Address
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "AddressRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::AddressRepr
    }

    /// raddress.py `assert type(value) is fakeaddress`
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !matches!(value, ConstValue::LLAddress(_)) {
            return Err(TyperError::message(format!(
                "AddressRepr.convert_const: expected LLAddress, got {value:?}",
            )));
        }
        Ok(Constant::with_concretetype(
            value.clone(),
            LowLevelType::Address,
        ))
    }

    /// raddress.py rtype_getattr
    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_access = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        Ok(Some(v_access))
    }

    /// raddress.py `return None` — use default pointer equality
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// raddress.py get_ll_hash_function
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        rtyper
            .lowlevel_helper_function_with_builder(
                "ll_addrhash".to_string(),
                vec![LowLevelType::Address],
                LowLevelType::Signed,
                |_rtyper, args, _result| build_ll_addrhash_helper_graph("ll_addrhash", &args[0]),
            )
            .map(Some)
    }

    /// raddress.py:57
    fn get_ll_fasthash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_hash_function(rtyper)
    }

    /// raddress.py:50-53 `adr_ne(addr, NULL)` → Bool
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::translator::rtyper::rtyper::GenopResult;
        let v_addr = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        let c_null = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null),
            LowLevelType::Address,
        ));
        Ok(hop.genop(
            "adr_ne",
            vec![v_addr, c_null],
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    /// raddress.py `ll_str(a) = ll_int2hex(r_uint(ll_addrhash(a)),
    /// True)`. The default `Repr.rtype_str` (rmodel.py) calls
    /// `hop.gendirectcall(self.ll_str, v_self)`; the `ll_str` body is
    /// itself a helper graph that chains `ll_addrhash` into `ll_int2hex`.
    fn rtype_str(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_addr_str".to_string(),
            vec![LowLevelType::Address],
            crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone(),
            |rtyper, _args, _result| build_ll_address_str_helper_graph(rtyper, "ll_addr_str"),
        )?;
        hop.gendirectcall(&helper, vlist)
    }
}

impl fmt::Display for AddressRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

/// raddress.py `ll_addrhash(addr) = cast_adr_to_int(addr, "forced")`
fn build_ll_addrhash_helper_graph(
    name: &str,
    _arg_type: &LowLevelType,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, FunctionGraph, GraphFunc, Link, SpaceOperation};
    use crate::translator::rtyper::rtyper::{helper_pygraph_from_graph, variable_with_lltype};

    let arg = variable_with_lltype("addr1", LowLevelType::Address);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let cast_var = variable_with_lltype("hashed", LowLevelType::Signed);
    let c_mode = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::ByteStr(b"forced".to_vec()),
        LowLevelType::Void,
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_adr_to_int",
        vec![Hlvalue::Variable(arg), c_mode],
        Hlvalue::Variable(cast_var.clone()),
    ));

    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(cast_var)],
            Some(graph.returnblock.clone()),
            None,
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
        vec!["addr1".to_string()],
        func,
    ))
}

/// raddress.py `ll_str(a)`: chain `ll_addrhash(a)` into
/// `ll_int2hex(r_uint(id), True)`. The `r_uint(id)` wrap is emitted as a
/// `cast_primitive` of the Signed hash to `Unsigned`, feeding the Unsigned
/// specialisation of `ll_int2hex` (`signed_input=false`, sign branch
/// pruned) so a negative address hashes to its full-width unsigned hex.
fn build_ll_address_str_helper_graph(
    rtyper: &super::rtyper::RPythonTyper,
    name: &str,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, FunctionGraph, GraphFunc, Link, SpaceOperation};
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, build_ll_int2hex_helper_graph, sub_helper_funcptr_constant,
    };
    use crate::translator::rtyper::rtyper::{
        constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
    };

    let addrhash = rtyper.lowlevel_helper_function_with_builder(
        "ll_addrhash".to_string(),
        vec![LowLevelType::Address],
        LowLevelType::Signed,
        |_rtyper, _args, _result| {
            build_ll_addrhash_helper_graph("ll_addrhash", &LowLevelType::Address)
        },
    )?;
    let c_addrhash = sub_helper_funcptr_constant(rtyper, &addrhash)?;
    let int2hex = rtyper.lowlevel_helper_function_with_builder(
        "ll_int2hex".to_string(),
        vec![LowLevelType::Unsigned, LowLevelType::Bool],
        STRPTR.clone(),
        |_rtyper, _args, _result| build_ll_int2hex_helper_graph("ll_int2hex", false),
    )?;
    let c_int2hex = sub_helper_funcptr_constant(rtyper, &int2hex)?;

    let addr = variable_with_lltype("a", LowLevelType::Address);
    let startblock = Block::shared(vec![Hlvalue::Variable(addr.clone())]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_id = variable_with_lltype("id", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_addrhash), Hlvalue::Variable(addr)],
        Hlvalue::Variable(v_id.clone()),
    ));
    // r_uint(id): reinterpret the Signed hash as Unsigned for the unsigned
    // ll_int2hex specialisation.
    let v_uid = variable_with_lltype("id", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_primitive",
        vec![Hlvalue::Variable(v_id)],
        Hlvalue::Variable(v_uid.clone()),
    ));
    let c_true = constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let v_result = variable_with_lltype("result", STRPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(c_int2hex),
            Hlvalue::Variable(v_uid),
            c_true,
        ],
        Hlvalue::Variable(v_result.clone()),
    ));

    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
            Some(graph.returnblock.clone()),
            None,
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
        vec!["a".to_string()],
        func,
    ))
}

/// raddress.py — module-global singleton `address_repr = AddressRepr()`.
pub fn address_repr() -> Arc<AddressRepr> {
    use std::sync::OnceLock;
    static REPR: OnceLock<Arc<AddressRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(AddressRepr::new())).clone()
}

/// RPython `raddress.py` — `class TypedAddressAccessRepr(Repr)`.
/// Intermediate repr for `addr.signed[offset]` patterns.
/// `lowleveltype = Address`, stores the access type (Signed/Unsigned/Char/Address/Float).
#[derive(Debug)]
pub struct TypedAddressAccessRepr {
    state: ReprState,
    pub access_type: LowLevelType,
}

impl TypedAddressAccessRepr {
    pub fn new(access_type: LowLevelType) -> Self {
        TypedAddressAccessRepr {
            state: ReprState::new(),
            access_type,
        }
    }
}

impl Repr for TypedAddressAccessRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &LowLevelType::Address
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "TypedAddressAccessRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::TypedAddressAccessRepr
    }
}

impl fmt::Display for TypedAddressAccessRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

// WeakRefRepr and EmulatedWeakRefRepr live in rweakref.rs
// (rweakref.py:51-64 / rweakref.py:67-96)

/// RPython `rptr.py` — `class PtrRepr(Repr)`.
#[derive(Debug)]
pub struct PtrRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl PtrRepr {
    pub fn new(ptrtype: crate::translator::rtyper::lltypesystem::lltype::Ptr) -> Self {
        PtrRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Ptr(Box::new(ptrtype)),
        }
    }

    fn ptrtype(&self) -> &lltype::Ptr {
        ptr_type(&self.lltype)
    }
}

impl Repr for PtrRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "PtrRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::PtrRepr
    }

    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        let r_result = hop_r_result(hop)?;
        // rptr.py:44-46: `if isinstance(hop.s_result, SomeLLADTMeth):
        // return hop.inputarg(hop.r_result, arg=0)`.
        if matches!(&*hop.s_result.borrow(), Some(SomeValue::LLADTMeth(_))) {
            return hop.inputarg(&r_result, 0).map(Some);
        }
        // `is_null` on a bare pointer reaches here as a `SomeBuiltinMethod`
        // (`unaryop.rs ptr_method_is_null`) rather than a `SomeLLADTMeth`.
        // The bound method is represented as its receiver
        // (`BuiltinMethodRepr.lowleveltype == self_repr.lowleveltype`), so
        // forward the pointer value in self's repr and let
        // `BuiltinMethodRepr.rtype_simple_call` dispatch back through
        // `rtype_method`.
        if matches!(&*hop.s_result.borrow(), Some(SomeValue::BuiltinMethod(_))) {
            return hop.inputarg(self, 0).map(Some);
        }
        if self.ptrtype()._example()._lookup_adtmeth(&attr).is_ok() {
            let value = hop
                .s_result
                .borrow()
                .as_ref()
                .and_then(|s| s.const_())
                .ok_or_else(|| {
                    TyperError::message("PtrRepr.rtype_getattr expected constant ADT member")
                })?
                .clone();
            return HighLevelOp::inputconst(&r_result, &value).map(|c| Some(Hlvalue::Constant(c)));
        }
        let field_type = ptr_target_field_type(&self.ptrtype().TO, &attr)
            .ok_or_else(|| TyperError::message(self.ptrtype()._nofield(&attr)))?;
        let newopname = if field_type.is_container_type() {
            if let Some(struct_t) = ptr_target_struct(&self.ptrtype().TO)
                && let Some((first_attr, first_struct)) = struct_t._first_struct()
                && first_attr == attr
                && matches!(&field_type, LowLevelType::Struct(t) if t.as_ref() == first_struct)
            {
                let v_self = hop.inputarg(self, 0)?;
                return Ok(hop.genop(
                    "cast_pointer",
                    vec![v_self],
                    GenopResult::LLType(r_result.lowleveltype().clone()),
                ));
            } else if r_result.class_name() == "InteriorPtrRepr" {
                return hop.inputarg(self, 0).map(Some);
            } else {
                "getsubstruct"
            }
        } else {
            "getfield"
        };
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![ConvertedTo::from(self), ConvertedTo::from(&void)])?;
        Ok(hop.genop(
            newopname,
            vlist,
            GenopResult::LLType(r_result.lowleveltype().clone()),
        ))
    }

    /// `BuiltinMethodRepr.rtype_simple_call` dispatch target for the ptr
    /// bound method seated by the `SomeBuiltinMethod` pass-through in
    /// `rtype_getattr`. `is_null` is the lltype `_ptr` nullity probe —
    /// lower it as `ptr_iszero` (opimpl.py `op_ptr_iszero`).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        if method_name == "is_null" {
            let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
            return Ok(hop.genop("ptr_iszero", vlist, GenopResult::LLType(LowLevelType::Bool)));
        }
        // `Option<RawStruct>::as_ref` / `<*mut T>::as_ref`: the receiver
        // is already the nullable pointer word. Identity, matching the
        // annotator's `ptr_method_as_ref`.
        if method_name == "as_ref" || method_name == "as_mut" {
            return hop.inputarg(self, 0).map(Some);
        }
        Err(TyperError::message(format!(
            "missing PtrRepr.rtype_method_{method_name}"
        )))
    }

    fn rtype_setattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        let field_type = ptr_target_field_type(&self.ptrtype().TO, &attr)
            .ok_or_else(|| TyperError::message(self.ptrtype()._nofield(&attr)))?;
        assert!(!field_type.is_container_type());
        let args_r = hop.args_r.borrow();
        let r_value = args_r
            .get(2)
            .and_then(|r| r.as_ref())
            .ok_or_else(|| TyperError::message("PtrRepr.rtype_setattr missing value repr"))?;
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&void),
            ConvertedTo::from(r_value),
        ])?;
        hop.genop("setfield", vlist, GenopResult::Void);
        Ok(None)
    }

    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        if let Some(length) = ptr_target_fixed_length(&self.ptrtype().TO) {
            return HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(length as i64))
                .map(|c| Some(Hlvalue::Constant(c)));
        }
        let r_result = hop_r_result(hop)?;
        let vlist = hop.inputargs(vec![ConvertedTo::from(self)])?;
        Ok(hop.genop(
            "getarraysize",
            vlist,
            GenopResult::LLType(r_result.lowleveltype().clone()),
        ))
    }

    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::from(self)])?;
        Ok(hop.genop(
            "ptr_nonzero",
            vlist,
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let item_type = ptr_target_array_item_type(&self.ptrtype().TO).ok_or_else(|| {
            TyperError::message(format!(
                "getitem on non-array pointer {:?}",
                self.ptrtype().TO
            ))
        })?;
        let r_result = hop_r_result(hop)?;
        if item_type.is_container_type() {
            if r_result.class_name() == "InteriorPtrRepr" {
                let mut inputs = hop.inputargs(vec![
                    ConvertedTo::from(self),
                    ConvertedTo::from(&LowLevelType::Signed),
                ])?;
                let v_index = inputs
                    .pop()
                    .ok_or_else(|| TyperError::message("PtrRepr.rtype_getitem missing index"))?;
                let v_array = inputs
                    .pop()
                    .ok_or_else(|| TyperError::message("PtrRepr.rtype_getitem missing array"))?;
                let interior_ptr_type = self.ptrtype()._interior_ptr_type_with_index(&item_type);
                let v_interior_ptr = hop
                    .genop(
                        "malloc",
                        vec![
                            lowlevel_type_const(LowLevelType::Struct(Box::new(
                                interior_ptr_type.clone(),
                            ))),
                            gc_flavor_const()?,
                        ],
                        GenopResult::LLType(LowLevelType::Ptr(Box::new(lltype::Ptr {
                            TO: lltype::PtrTarget::Struct(interior_ptr_type),
                        }))),
                    )
                    .ok_or_else(|| TyperError::message("malloc unexpectedly returned no value"))?;
                hop.genop(
                    "setfield",
                    vec![v_interior_ptr.clone(), void_field_const("ptr"), v_array],
                    GenopResult::Void,
                );
                hop.genop(
                    "setfield",
                    vec![v_interior_ptr.clone(), void_field_const("index"), v_index],
                    GenopResult::Void,
                );
                return Ok(Some(v_interior_ptr));
            }
            let vlist = hop.inputargs(vec![
                ConvertedTo::from(self),
                ConvertedTo::from(&LowLevelType::Signed),
            ])?;
            return Ok(hop.genop(
                "getarraysubstruct",
                vlist,
                GenopResult::LLType(r_result.lowleveltype().clone()),
            ));
        }
        let vlist = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&LowLevelType::Signed),
        ])?;
        Ok(hop.genop(
            "getarrayitem",
            vlist,
            GenopResult::LLType(r_result.lowleveltype().clone()),
        ))
    }

    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let item_type = ptr_target_array_item_type(&self.ptrtype().TO).ok_or_else(|| {
            TyperError::message(format!(
                "setitem on non-array pointer {:?}",
                self.ptrtype().TO
            ))
        })?;
        assert!(!item_type.is_container_type());
        let args_r = hop.args_r.borrow();
        let r_value = args_r
            .get(2)
            .and_then(|r| r.as_ref())
            .ok_or_else(|| TyperError::message("PtrRepr.rtype_setitem missing value repr"))?;
        let vlist = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&LowLevelType::Signed),
            ConvertedTo::from(r_value),
        ])?;
        hop.genop("setarrayitem", vlist, GenopResult::Void);
        Ok(None)
    }

    fn rtype_eq(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_ptr_comparison(self, hop, "ptr_eq")
    }

    fn rtype_ne(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_ptr_comparison(self, hop, "ptr_ne")
    }

    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let func_type = ptr_target_func(&self.ptrtype().TO).ok_or_else(|| {
            TyperError::message(format!("calling a non-function {:?}", self.ptrtype().TO))
        })?;
        let args_r = hop.args_r.borrow().clone();
        let mut vlist = Vec::with_capacity(args_r.len() + 1);
        for (i, r_arg) in args_r.iter().enumerate() {
            let r_arg = r_arg
                .as_ref()
                .ok_or_else(|| TyperError::message("PtrRepr.rtype_simple_call missing arg repr"))?;
            vlist.push(hop.inputarg(r_arg, i)?);
        }
        let nexpected = func_type.args.len();
        let nactual = vlist.len().saturating_sub(1);
        if nactual != nexpected {
            return Err(TyperError::message(format!(
                "argcount mismatch:  expected {nexpected} got {nactual}"
            )));
        }
        let opname = if let Some(Hlvalue::Constant(c_func)) = vlist.first() {
            if let ConstValue::LLPtr(ptr) = &c_func.value
                && let Ok(lltype::_ptr_obj::Func(func)) = ptr._obj()
                && let Some(graph_id) = func.graph
            {
                hop.llops.borrow().record_extra_call_by_graph_id(graph_id)?;
            }
            "direct_call"
        } else {
            vlist.push(Hlvalue::Constant(HighLevelOp::inputconst(
                &LowLevelType::Void,
                &ConstValue::None,
            )?));
            "indirect_call"
        };
        hop.exception_is_here()?;
        Ok(hop.genop(opname, vlist, GenopResult::LLType(func_type.result.clone())))
    }
}

fn ptr_from_container_lowleveltype(
    ptrtype: &crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
) -> crate::translator::rtyper::lltypesystem::lltype::Ptr {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

    Ptr {
        TO: match &*ptrtype.PARENTTYPE {
            LowLevelType::Struct(t) => PtrTarget::Struct((**t).clone()),
            LowLevelType::Array(t) => PtrTarget::Array((**t).clone()),
            LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray((**t).clone()),
            LowLevelType::Opaque(t) => PtrTarget::Opaque((**t).clone()),
            LowLevelType::ForwardReference(t) => PtrTarget::ForwardReference((**t).clone()),
            other => panic!("InteriorPtrRepr parent must be a container type, got {other:?}"),
        },
    }
}

fn interior_ptr_lowleveltype(
    ptrtype: &crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
) -> LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

    let parent_ptr = ptr_from_container_lowleveltype(ptrtype);
    if ptrtype.offsets.iter().any(|offset| {
        matches!(
            offset,
            crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Index(_)
        )
    }) {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(parent_ptr._interior_ptr_type_with_index(&ptrtype.TO)),
        }))
    } else {
        LowLevelType::Ptr(Box::new(parent_ptr))
    }
}

/// RPython `rptr.py` — `class InteriorPtrRepr(Repr)`.
#[derive(Debug)]
pub struct InteriorPtrRepr {
    state: ReprState,
    lltype: LowLevelType,
    _ptrtype: crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
    pub v_offsets: Vec<Option<Constant>>,
    pub parentptrtype: crate::translator::rtyper::lltypesystem::lltype::Ptr,
}

impl InteriorPtrRepr {
    pub fn new(ptrtype: crate::translator::rtyper::lltypesystem::lltype::InteriorPtr) -> Self {
        let lltype = interior_ptr_lowleveltype(&ptrtype);
        let mut v_offsets = Vec::new();
        let mut numitemoffsets = 0;
        for offset in &ptrtype.offsets {
            match offset {
                crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Index(_) => {
                    numitemoffsets += 1;
                    v_offsets.push(None);
                }
                crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Field(name) => {
                    v_offsets.push(Some(Constant::with_concretetype(
                        ConstValue::byte_str(name),
                        LowLevelType::Void,
                    )));
                }
            }
        }
        assert!(numitemoffsets <= 1);
        let parentptrtype = ptr_from_container_lowleveltype(&ptrtype);
        InteriorPtrRepr {
            state: ReprState::new(),
            lltype,
            _ptrtype: ptrtype,
            v_offsets,
            parentptrtype,
        }
    }

    fn result_target(&self) -> &LowLevelType {
        self._ptrtype.TO.as_ref()
    }

    fn getinteriorfieldargs(
        &self,
        hop: &HighLevelOp,
        v_self: Hlvalue,
    ) -> Result<Vec<Hlvalue>, TyperError> {
        let mut vlist = Vec::new();
        let has_index = self.v_offsets.iter().any(|offset| offset.is_none());
        let mut nameiter = Vec::<String>::new();
        let mut name_index = 0usize;
        let mut interior_struct = None;
        if has_index {
            let concretetype = hlvalue_concretetype(&v_self)?;
            let ptr = ptr_type(&concretetype);
            let struct_t = ptr_target_struct(&ptr.TO).ok_or_else(|| {
                TyperError::message("InteriorPtrRepr expected struct concretetype")
            })?;
            nameiter = struct_t._names.clone();
            let name = nameiter
                .get(name_index)
                .ok_or_else(|| TyperError::message("interior pointer struct has no fields"))?
                .clone();
            name_index += 1;
            let field_type = struct_t.getattr_field_type(&name).ok_or_else(|| {
                TyperError::message(format!("interior pointer struct missing field {name:?}"))
            })?;
            let v_field = hop
                .genop(
                    "getfield",
                    vec![v_self.clone(), void_field_const(&name)],
                    GenopResult::LLType(field_type),
                )
                .ok_or_else(|| TyperError::message("getfield unexpectedly returned no value"))?;
            vlist.push(v_field);
            interior_struct = Some(struct_t);
        } else {
            vlist.push(v_self.clone());
        }
        for v_offset in &self.v_offsets {
            if let Some(v_offset) = v_offset {
                vlist.push(Hlvalue::Constant(v_offset.clone()));
            } else {
                let struct_t = interior_struct
                    .as_ref()
                    .ok_or_else(|| TyperError::message("missing interior pointer struct"))?;
                let name = nameiter
                    .get(name_index)
                    .ok_or_else(|| {
                        TyperError::message("interior pointer offset has no matching field")
                    })?
                    .clone();
                name_index += 1;
                let field_type = struct_t.getattr_field_type(&name).ok_or_else(|| {
                    TyperError::message(format!("interior pointer struct missing field {name:?}"))
                })?;
                let v_field = hop
                    .genop(
                        "getfield",
                        vec![v_self.clone(), void_field_const(&name)],
                        GenopResult::LLType(field_type),
                    )
                    .ok_or_else(|| {
                        TyperError::message("getfield unexpectedly returned no value")
                    })?;
                vlist.push(v_field);
            }
        }
        if has_index && name_index != nameiter.len() {
            panic!("interior pointer field unpack left unused fields");
        }
        Ok(vlist)
    }
}

impl Repr for InteriorPtrRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "InteriorPtrRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::InteriorPtrRepr
    }

    fn interior_ptr_repr_dict_key(&self) -> Option<lltype::InteriorPtr> {
        Some(self._ptrtype.clone())
    }

    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let mut inputs = hop.inputargs(vec![ConvertedTo::from(self)])?;
        let v_self = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_len missing self"))?;
        let vlist = self.getinteriorfieldargs(hop, v_self)?;
        Ok(hop.genop(
            "getinteriorarraysize",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        let r_result = hop_r_result(hop)?;
        if matches!(&*hop.s_result.borrow(), Some(SomeValue::LLADTMeth(_))) {
            return hop.inputarg(&r_result, 0).map(Some);
        }
        let field_type = lowlevel_container_field_type(self.result_target(), &attr)
            .ok_or_else(|| TyperError::message(format!("missing interior field {attr:?}")))?;
        if field_type.is_container_type() {
            hop.inputarg(self, 0).map(Some)
        } else {
            let void = LowLevelType::Void;
            let mut inputs =
                hop.inputargs(vec![ConvertedTo::from(self), ConvertedTo::from(&void)])?;
            let v_attr = inputs
                .pop()
                .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_getattr missing attr"))?;
            let v_self = inputs
                .pop()
                .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_getattr missing self"))?;
            let mut vlist = self.getinteriorfieldargs(hop, v_self)?;
            vlist.push(v_attr);
            Ok(hop.genop(
                "getinteriorfield",
                vlist,
                GenopResult::LLType(r_result.lowleveltype().clone()),
            ))
        }
    }

    fn rtype_setattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        let field_type = lowlevel_container_field_type(self.result_target(), &attr)
            .ok_or_else(|| TyperError::message(format!("missing interior field {attr:?}")))?;
        assert!(!field_type.is_container_type());
        let args_r = hop.args_r.borrow();
        let r_value = args_r.get(2).and_then(|r| r.as_ref()).ok_or_else(|| {
            TyperError::message("InteriorPtrRepr.rtype_setattr missing value repr")
        })?;
        let void = LowLevelType::Void;
        let mut inputs = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&void),
            ConvertedTo::from(r_value),
        ])?;
        let v_value = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setattr missing value"))?;
        let v_fieldname = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setattr missing field"))?;
        let v_self = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setattr missing self"))?;
        let mut vlist = self.getinteriorfieldargs(hop, v_self)?;
        vlist.push(v_fieldname);
        vlist.push(v_value);
        Ok(hop.genop("setinteriorfield", vlist, GenopResult::Void))
    }

    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let item_type = lowlevel_array_item_type(self.result_target()).ok_or_else(|| {
            TyperError::message(format!(
                "getitem on non-array interior pointer {:?}",
                self.result_target()
            ))
        })?;
        if item_type.is_container_type() {
            let mut inputs = hop.inputargs(vec![
                ConvertedTo::from(self),
                ConvertedTo::from(&LowLevelType::Signed),
            ])?;
            let v_index = inputs.pop().ok_or_else(|| {
                TyperError::message("InteriorPtrRepr.rtype_getitem missing index")
            })?;
            let v_array = inputs.pop().ok_or_else(|| {
                TyperError::message("InteriorPtrRepr.rtype_getitem missing array")
            })?;
            let interior_ptr_type =
                ptr_type(self.lowleveltype())._interior_ptr_type_with_index(&item_type);
            let v_interior_ptr = hop
                .genop(
                    "malloc",
                    vec![
                        lowlevel_type_const(LowLevelType::Struct(Box::new(
                            interior_ptr_type.clone(),
                        ))),
                        gc_flavor_const()?,
                    ],
                    GenopResult::LLType(LowLevelType::Ptr(Box::new(lltype::Ptr {
                        TO: lltype::PtrTarget::Struct(interior_ptr_type),
                    }))),
                )
                .ok_or_else(|| TyperError::message("malloc unexpectedly returned no value"))?;
            hop.genop(
                "setfield",
                vec![v_interior_ptr.clone(), void_field_const("ptr"), v_array],
                GenopResult::Void,
            );
            hop.genop(
                "setfield",
                vec![v_interior_ptr.clone(), void_field_const("index"), v_index],
                GenopResult::Void,
            );
            return Ok(Some(v_interior_ptr));
        }
        let mut inputs = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&LowLevelType::Signed),
        ])?;
        let v_index = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_getitem missing index"))?;
        let v_self = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_getitem missing self"))?;
        let mut vlist = self.getinteriorfieldargs(hop, v_self)?;
        vlist.push(v_index);
        Ok(hop.genop("getinteriorfield", vlist, GenopResult::LLType(item_type)))
    }

    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let item_type = lowlevel_array_item_type(self.result_target()).ok_or_else(|| {
            TyperError::message(format!(
                "setitem on non-array interior pointer {:?}",
                self.result_target()
            ))
        })?;
        assert!(!item_type.is_container_type());
        let args_r = hop.args_r.borrow();
        let r_value = args_r.get(2).and_then(|r| r.as_ref()).ok_or_else(|| {
            TyperError::message("InteriorPtrRepr.rtype_setitem missing value repr")
        })?;
        let mut inputs = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&LowLevelType::Signed),
            ConvertedTo::from(r_value),
        ])?;
        let v_value = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setitem missing value"))?;
        let v_index = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setitem missing index"))?;
        let v_self = inputs
            .pop()
            .ok_or_else(|| TyperError::message("InteriorPtrRepr.rtype_setitem missing self"))?;
        let mut vlist = self.getinteriorfieldargs(hop, v_self)?;
        vlist.push(v_index);
        vlist.push(v_value);
        hop.genop("setinteriorfield", vlist, GenopResult::Void);
        Ok(None)
    }
}

/// RPython `rptr.py` — `class LLADTMethRepr(Repr)`.
#[derive(Debug)]
pub struct LLADTMethRepr {
    state: ReprState,
    lltype: LowLevelType,
    pub func: ConstValue,
    pub ll_ptrtype: crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType,
}

impl LLADTMethRepr {
    pub fn new(adtmeth: &crate::annotator::model::SomeLLADTMeth) -> Self {
        let lltype = match &adtmeth.ll_ptrtype {
            crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::Ptr(ptr) => {
                LowLevelType::Ptr(Box::new(ptr.clone()))
            }
            crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::InteriorPtr(
                ptr,
            ) => interior_ptr_lowleveltype(ptr),
        };
        LLADTMethRepr {
            state: ReprState::new(),
            lltype,
            func: adtmeth.func.clone(),
            ll_ptrtype: adtmeth.ll_ptrtype.clone(),
        }
    }
}

impl Repr for LLADTMethRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "LLADTMethRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::LLADTMethRepr
    }

    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let hop2 = hop.copy();
        let func = self.func.clone();
        let rtyper = &hop.rtyper;
        let annotator = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        let s_func = annotator
            .bookkeeper
            .immutablevalue(&func)
            .map_err(|err| TyperError::message(err.to_string()))?;
        let v_ptr =
            hop2.args_v.borrow().first().cloned().ok_or_else(|| {
                TyperError::message("LLADTMethRepr.rtype_simple_call missing self")
            })?;
        hop2.r_s_popfirstarg();
        hop2.v_s_insertfirstarg(v_ptr, lltype_to_annotation(self.ll_ptrtype.clone()))?;
        hop2.v_s_insertfirstarg(Hlvalue::Constant(Constant::new(func)), s_func)?;
        hop2.dispatch()
    }
}

// ____________________________________________________________
// `rtyper_makekey` / `rtyper_makerepr` per-SomeXxx dispatch
// (rmodel.py + each r*.py `__extend__(SomeXxx)` block).

/// Discriminator inside [`ReprKey::Builtin`] — mirrors upstream
/// `SomeBuiltin.rtyper_makekey`'s `const` slot (rbuiltin.py).
///
/// Upstream swaps `const` for the registered `ExtRegistryEntry`
/// instance when `extregistry.is_registered(const)`; pyre keys on
/// [`crate::translator::rtyper::extregistry::ExtRegistryEntryKey`],
/// mirroring upstream `ExtRegistryEntry.__hash__` over
/// `(entry.__class__, entry.type, entry.instance)`. Hosts fall back to
/// `HostObject` pointer identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltinConstKey {
    /// The const is registered in the extregistry. The key preserves
    /// entry class + type/instance identity instead of collapsing every
    /// variant to a single tag.
    ExtRegistry(crate::translator::rtyper::extregistry::ExtRegistryEntryKey),
    /// Non-registered const. Pyre stores the [`HostObject`] pointer
    /// identity, which is the closest analogue to upstream's
    /// `(self.__class__, const)` tuple where `const` is a Python
    /// callable or attribute.
    Host(usize),
}

/// RPython `S.rtyper_makekey()` upstream tuple hashed by
/// `RPythonTyper.reprs` (rtyper.py+149).
///
/// Upstream returns a tuple like `(S.__class__, knowntype, unsigned)`
/// which Python uses as a dict key. Rust mirrors via a typed enum so
/// each variant carries the right discriminating data for the reprs
/// cache.
///
/// Only `Impossible` is populated in this commit — other SomeValue
/// variants land with their respective concrete Repr ports. The
/// `Pending` variant carries a debug string so unimplemented keys
/// still round-trip through `HashMap<ReprKey, Arc<dyn Repr>>` during
/// bring-up.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReprKey {
    /// RPython `SomeImpossibleValue.rtyper_makekey = (self.__class__,)`
    /// (rmodel.py).
    Impossible,
    /// RPython `SomeInteger.rtyper_makekey = (self.__class__,
    /// self.knowntype)` (`rint.py`).
    Integer(KnownType),
    /// RPython `SomeNone.rtyper_makekey = (self.__class__,)`
    /// (`rnone.py`).
    None_,
    /// RPython `SomeBool.rtyper_makekey = (self.__class__,)`
    /// (`rbool.py`).
    Bool,
    /// RPython `SomeFloat.rtyper_makekey = (self.__class__,)`
    /// (`rfloat.py:71-72`).
    Float,
    /// RPython `SomeSingleFloat.rtyper_makekey = (self.__class__,)`
    /// (`rfloat.py:147-148`).
    SingleFloat,
    /// RPython `SomeLongFloat.rtyper_makekey = (self.__class__,)`
    /// (`rfloat.py:163-164`).
    LongFloat,
    /// RPython `SomeInstance.rtyper_makekey = (self.__class__,
    /// self.classdef)` (rclass.py). `None` mirrors the
    /// `SomeInstance(classdef=None)` sentinel.
    Instance(Option<crate::annotator::description::ClassDefKey>),
    /// RPython `SomeException.rtyper_makekey = (self.__class__,
    /// frozenset(self.classdefs))` (rclass.py). Stored as a
    /// sorted+deduped `Vec<ClassDefKey>` so the frozenset-equality
    /// semantics carry through `HashMap` lookups.
    Exception(Vec<crate::annotator::description::ClassDefKey>),
    /// RPython `SomeType.rtyper_makekey = (self.__class__,)`
    /// (rclass.py).
    Type,
    /// `SomeTypeOf` inherits `SomeType.rtyper_makekey`; because the method
    /// returns `(self.__class__,)`, all `SomeTypeOf` annotations share their
    /// own bucket, distinct from the `SomeType` bucket.
    TypeOf,
    /// RPython `SomeIterator.rtyper_makekey` (rmodel.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     return self.__class__, self.s_container.rtyper_makekey(), self.variant
    /// ```
    ///
    /// Recursively keys on the container's `rtyper_makekey` + the
    /// variant tuple (e.g. `("items",)`, `("keys",)`, `("enumerate",)`).
    Iterator {
        container: Box<ReprKey>,
        variant: Vec<String>,
        /// Upstream's variant tuple carries the constant start of
        /// `("enumerate", start)`; pyre keeps it beside the variant.
        enumerate_start: Option<i64>,
    },
    /// RPython `SomeWeakRef.rtyper_makekey = (self.__class__,)`
    /// (rweakref.py:19-20).
    WeakRef,
    /// RPython `SomeTuple.rtyper_makekey` (rtuple.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     keys = [s_item.rtyper_makekey() for s_item in self.items]
    ///     return tuple([self.__class__] + keys)
    /// ```
    ///
    /// Recursively keys every item — two `SomeTuple`s with identical
    /// element-shape collapse to the same cache entry.
    Tuple(Vec<ReprKey>),
    /// RPython `SomeList.rtyper_makekey` (rlist.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     self.listdef.listitem.dont_change_any_more = True
    ///     return self.__class__, self.listdef.listitem
    /// ```
    ///
    /// `listitem_id` is the `Rc::as_ptr` identity of the list's
    /// `ListItem` — upstream `id(self.listdef.listitem)`. The freeze is
    /// journaled before the write so an added-blocks rollback restores it.
    List(usize),
    /// RPython `SomeDict.rtyper_makekey` (rdict.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     self.dictdef.dictkey  .dont_change_any_more = True
    ///     self.dictdef.dictvalue.dont_change_any_more = True
    ///     return (self.__class__, self.dictdef.dictkey, self.dictdef.dictvalue)
    /// ```
    Dict {
        dictkey_id: usize,
        dictvalue_id: usize,
    },
    /// RPython `SomeString.rtyper_makekey = (self.__class__,)`
    /// (rstr.py:573-574).
    String,
    /// RPython `SomeUnicodeString.rtyper_makekey = (self.__class__,)`
    /// (rstr.py:581-582).
    UnicodeString,
    /// RPython `SomeChar.rtyper_makekey = (self.__class__,)`
    /// (rstr.py:589-590).
    Char,
    /// RPython `SomeUnicodeCodePoint.rtyper_makekey = (self.__class__,)`
    /// (rstr.py:597-598).
    UnicodeCodePoint,
    /// RPython `SomeByteArray.rtyper_makekey = (self.__class__,)`
    /// (rbytearray.py — mirrored for parity even though the repr
    /// itself is pending).
    ByteArray,
    /// RPython `SomeBuiltin.rtyper_makekey` (rbuiltin.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     const = getattr(self, 'const', None)
    ///     if extregistry.is_registered(const):
    ///         const = extregistry.lookup(const)
    ///     return self.__class__, const
    /// ```
    ///
    /// The extregistry remap preserves
    /// `ExtRegistryEntry.__hash__`-equivalent identity. Non-registered
    /// consts key on the [`HostObject`] pointer identity; `None` covers
    /// the non-constant fallback.
    Builtin(Option<BuiltinConstKey>),
    /// RPython `SomeBuiltinMethod.rtyper_makekey` (rbuiltin.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     return (self.__class__, self.methodname, id(self.s_self))
    /// ```
    ///
    /// `s_self_id` is the pointer identity of the receiver annotation
    /// so two `SomeBuiltinMethod`s with distinct receivers stay in
    /// separate cache entries, matching upstream's `id(self.s_self)`.
    BuiltinMethod {
        methodname: String,
        s_self_id: usize,
    },
    /// RPython `SomePBC.rtyper_makekey` (rpbc.py).
    ///
    /// ```python
    /// def rtyper_makekey(self):
    ///     lst = list(self.descriptions)
    ///     lst.sort()
    ///     if self.subset_of:
    ///         t = self.subset_of.rtyper_makekey()
    ///     else:
    ///         t = ()
    ///     return tuple([self.__class__, self.can_be_None] + lst) + t
    /// ```
    ///
    /// `descriptions` is sorted ascending by [`DescKey`] (pointer
    /// identity) to mirror upstream's `lst.sort()`; this means two
    /// `SomePBC`s with identical desc sets + can_be_None + subset_of
    /// collide on the same cache entry regardless of insertion order
    /// — the upstream set-equality contract.
    PBC {
        descriptions: Vec<crate::annotator::description::DescKey>,
        can_be_none: bool,
        subset_of: Option<Box<ReprKey>>,
    },
    /// RPython `SomeAddress.rtyper_makekey = (self.__class__,)`
    /// (raddress.py).
    Address,
    /// RPython `SomeTypedAddressAccess.rtyper_makekey = (self.__class__, self.type)`
    /// (raddress.py).
    TypedAddressAccess(LowLevelType),
    /// RPython `SomePtr.rtyper_makekey = (self.__class__, self.ll_ptrtype)`
    /// (rptr.py). Keyed on the pointer lltype by value: `LowLevelType`'s
    /// `Eq`/`Hash` mirror `lltype.py` `__eq__`/`__hash__`, which
    /// compare `__dict__` structurally (and resolve forward references via
    /// `become()`), so two distinct pointer types never share a bucket and
    /// a resolved forward-reference pointer keys identically to the real
    /// container it points at.
    Ptr(LowLevelType),
    /// RPython `SomeInteriorPtr.rtyper_makekey = (self.__class__,
    /// self.ll_ptrtype)` (rptr.py). Same by-value lltype key as [`Ptr`].
    InteriorPtr(LowLevelType),
    /// RPython `SomeStringBuilder.rtyper_makekey = (self.__class__,)`
    /// (rstring.py). Class-tag singleton.
    StringBuilder,
    /// RPython `SomeUnicodeBuilder.rtyper_makekey = (self.__class__,)`
    /// (rstring.py). Class-tag singleton.
    UnicodeBuilder,
    /// `SomeRustVec`, keyed by its item annotation's key.
    RustVec(Box<ReprKey>),
    /// Pending variant — carries a textual discriminator from
    /// `rtyper_makekey` arm that hasn't been ported yet.
    Pending(String),
}

/// RPython `SomeXxx.rtyper_makekey()` dispatcher.
///
/// Upstream attaches a `rtyper_makekey` method per SomeXxx via the
/// `__extend__` metaclass pattern (rmodel.py, rint.py,
/// rbool.py, ...). Pyre centralises into this match since Rust has
/// no `__extend__` equivalent.
pub fn rtyper_makekey(s_obj: &crate::annotator::model::SomeValue) -> ReprKey {
    use crate::annotator::model::SomeValue;
    match s_obj {
        // rmodel.py: SomeImpossibleValue.rtyper_makekey = (self.__class__,).
        SomeValue::Impossible => ReprKey::Impossible,
        SomeValue::Integer(i) => ReprKey::Integer(i.base.knowntype),
        // rnone.py: SomeNone.rtyper_makekey = (self.__class__,).
        SomeValue::None_(_) => ReprKey::None_,
        // rbool.py: SomeBool.rtyper_makekey = (self.__class__,).
        SomeValue::Bool(_) => ReprKey::Bool,
        // rfloat.py: SomeFloat.rtyper_makekey = (self.__class__,).
        SomeValue::Float(_) => ReprKey::Float,
        // rfloat.py: SomeSingleFloat.rtyper_makekey = (self.__class__,).
        SomeValue::SingleFloat(_) => ReprKey::SingleFloat,
        // rfloat.py: SomeLongFloat.rtyper_makekey = (self.__class__,).
        SomeValue::LongFloat(_) => ReprKey::LongFloat,
        // rclass.py: SomeInstance.rtyper_makekey = (self.__class__, self.classdef).
        SomeValue::Instance(s) => ReprKey::Instance(
            s.classdef
                .as_ref()
                .map(crate::annotator::description::ClassDefKey::from_classdef),
        ),
        // rclass.py: SomeException.rtyper_makekey = (self.__class__, frozenset(self.classdefs)).
        SomeValue::Exception(s) => {
            let mut keys: Vec<crate::annotator::description::ClassDefKey> = s
                .classdefs
                .iter()
                .map(crate::annotator::description::ClassDefKey::from_classdef)
                .collect();
            keys.sort_by_key(|k| k.0);
            keys.dedup();
            ReprKey::Exception(keys)
        }
        // rclass.py: SomeType.rtyper_makekey = (self.__class__,).
        SomeValue::Type(_) => ReprKey::Type,
        // `SomeTypeOf(SomeType)` inherits the same method.  Upstream keys on
        // `self.__class__`, so this is a separate singleton key even though
        // both annotations select the same RootClassRepr.
        SomeValue::TypeOf(_) => ReprKey::TypeOf,
        // rmodel.py — SomeIterator.rtyper_makekey recursively
        // keys on container + variant tuple.
        SomeValue::Iterator(s) => ReprKey::Iterator {
            container: Box::new(rtyper_makekey(&s.s_container)),
            variant: s.variant.clone(),
            enumerate_start: match &s.enumerate_start {
                Some(crate::flowspace::model::ConstValue::Int(start)) => Some(*start),
                _ => None,
            },
        },
        // rweakref.py:19-20 — SomeWeakRef.rtyper_makekey class tag only.
        SomeValue::WeakRef(_) => ReprKey::WeakRef,
        // rtuple.py — SomeTuple.rtyper_makekey recursively keys
        // every item.
        SomeValue::Tuple(s) => ReprKey::Tuple(s.items.iter().map(rtyper_makekey).collect()),
        // rlist.py — SomeList.rtyper_makekey sets
        // `listitem.dont_change_any_more = True` then keys on listitem
        // pointer identity.
        SomeValue::List(s) => {
            let listitem_ref = s.listdef.inner.listitem.borrow();
            crate::annotator::listdef::journal_listitem_mutation(&*listitem_ref);
            listitem_ref.borrow_mut().dont_change_any_more = true;
            ReprKey::List(std::rc::Rc::as_ptr(&*listitem_ref) as usize)
        }
        // rdict.py — SomeDict.rtyper_makekey sets
        // `dictkey.dont_change_any_more = True` and
        // `dictvalue.dont_change_any_more = True`, then keys on the
        // (dictkey, dictvalue) pointer identity pair.
        SomeValue::Dict(s) => {
            let key_ref = s.dictdef.inner.dictkey.borrow();
            let value_ref = s.dictdef.inner.dictvalue.borrow();
            crate::annotator::listdef::journal_listitem_mutation(&*key_ref);
            crate::annotator::listdef::journal_listitem_mutation(&*value_ref);
            key_ref.borrow_mut().dont_change_any_more = true;
            value_ref.borrow_mut().dont_change_any_more = true;
            ReprKey::Dict {
                dictkey_id: std::rc::Rc::as_ptr(&*key_ref) as usize,
                dictvalue_id: std::rc::Rc::as_ptr(&*value_ref) as usize,
            }
        }
        // rstr.py:573-598 — SomeString / SomeUnicodeString / SomeChar
        // / SomeUnicodeCodePoint each key on their class tag alone.
        SomeValue::String(_) => ReprKey::String,
        SomeValue::UnicodeString(_) => ReprKey::UnicodeString,
        SomeValue::Char(_) => ReprKey::Char,
        SomeValue::UnicodeCodePoint(_) => ReprKey::UnicodeCodePoint,
        SomeValue::ByteArray(_) => ReprKey::ByteArray,
        // rbuiltin.py — SomeBuiltin.rtyper_makekey keys on
        // `self.const`, remapping through `extregistry.lookup(const)`
        // when `extregistry.is_registered(const)` is true so that
        // multiple builtin consts registered under the same entry
        // share one cache bucket. The HostObject identity fallback
        // covers the "plain builtin callable" case upstream encodes
        // as `(__class__, const)`.
        SomeValue::Builtin(s) => {
            use crate::translator::rtyper::extregistry;
            let key = s.base.const_box.as_ref().and_then(|c| {
                if extregistry::is_registered(&c.value) {
                    extregistry::lookup(&c.value)
                        .map(|entry| BuiltinConstKey::ExtRegistry(entry.makekey()))
                } else {
                    match &c.value {
                        crate::flowspace::model::ConstValue::HostObject(host) => {
                            Some(BuiltinConstKey::Host(host.identity_id()))
                        }
                        _ => None,
                    }
                }
            });
            ReprKey::Builtin(key)
        }
        // rbuiltin.py — SomeBuiltinMethod.rtyper_makekey keys on
        // `(self.methodname, id(self.s_self))`. `s.s_self` is an `Rc`
        // so its pointer identity is stable across clones of the
        // outer `SomeBuiltinMethod`, matching upstream's
        // `id(self.s_self)` invariant.
        SomeValue::BuiltinMethod(s) => ReprKey::BuiltinMethod {
            methodname: s.methodname.clone(),
            s_self_id: std::rc::Rc::as_ptr(&s.s_self) as usize,
        },
        // rpbc.py: SomePBC.rtyper_makekey lifts descriptions into
        // a sorted tuple prefixed by (class, can_be_None) and appends
        // the recursive `subset_of` key.
        SomeValue::PBC(s_pbc) => {
            let mut descriptions: Vec<crate::annotator::description::DescKey> =
                s_pbc.descriptions.keys().copied().collect();
            // Upstream: `lst.sort()` — descriptions list is sorted so
            // two PBCs with identical desc sets share a cache entry.
            descriptions.sort();
            let subset_of = s_pbc.subset_of.as_ref().map(|sub| {
                Box::new(rtyper_makekey(&crate::annotator::model::SomeValue::PBC(
                    (**sub).clone(),
                )))
            });
            ReprKey::PBC {
                descriptions,
                can_be_none: s_pbc.can_be_none,
                subset_of,
            }
        }
        // raddress.py: SomeAddress.rtyper_makekey = (self.__class__,).
        SomeValue::Address(_) => ReprKey::Address,
        // raddress.py: SomeTypedAddressAccess.rtyper_makekey = (self.__class__, self.type).
        SomeValue::TypedAddressAccess(s) => ReprKey::TypedAddressAccess(s.access_type.clone()),
        // rptr.py/24 — SomePtr / SomeInteriorPtr rtyper_makekey =
        // (self.__class__, self.ll_ptrtype). Key on the lltype itself; its
        // `Eq`/`Hash` (lltype.py `__eq__`/`__hash__`) compare
        // `__dict__` by value with forward references resolved through
        // `become()`, so distinct pointer types never collide and a
        // resolved forward-reference pointer buckets with its real container.
        SomeValue::Ptr(s) => ReprKey::Ptr(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::from(
                s.ll_ptrtype.clone(),
            ),
        ),
        SomeValue::InteriorPtr(s) => ReprKey::InteriorPtr(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::from(
                s.ll_ptrtype.clone(),
            ),
        ),
        // rstring.py/1268 — SomeStringBuilder / SomeUnicodeBuilder
        // rtyper_makekey = (self.__class__,).
        SomeValue::StringBuilder(_) => ReprKey::StringBuilder,
        SomeValue::UnicodeBuilder(_) => ReprKey::UnicodeBuilder,
        SomeValue::RustVec(s) => ReprKey::RustVec(Box::new(rtyper_makekey(&s.s_item))),
        // Remaining variants defer to their r*.rs ports. Emit a
        // deterministic `Pending` key so the reprs cache still
        // distinguishes entries by variant-shape — identical
        // shape-strings collapse to one pending entry (matching
        // upstream `(class, knowntype, ...)` identity for identical
        // tuples).
        other => ReprKey::Pending(format!("{other:?}")),
    }
}

/// RPython `SomeXxx.rtyper_makerepr(rtyper)` dispatcher.
///
/// Upstream r*.py modules each contribute one arm per SomeXxx via
/// `__extend__` (rmodel.py, rint.py, rbool.py, ...). Pyre
/// centralises into this match; as each concrete Repr lands in its
/// r*.rs file the arm flips from `MissingRTypeOperation` to the real
/// constructor.
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub fn rtyper_makerepr(
    s_obj: &crate::annotator::model::SomeValue,
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    use crate::annotator::model::SomeValue;
    let _ = rtyper; // unused until concrete repr ports consume it
    match s_obj {
        // rmodel.py: SomeImpossibleValue.rtyper_makerepr
        // returns the singleton `impossible_repr`.
        SomeValue::Impossible => Ok(impossible_repr() as std::sync::Arc<dyn Repr>),
        // Every other SomeValue variant maps 1:1 to a concrete Repr
        // defined in an r*.py upstream module. Pyre reports a
        // structured MissingRTypeOperation with the expected target
        // module so follow-up cascading ports can anchor on the
        // surface without guessing.
        SomeValue::Integer(i) => Ok(crate::translator::rtyper::rint::getintegerrepr(
            &lltype::build_number(None, i.base.knowntype),
        ) as std::sync::Arc<dyn Repr>),
        // rbool.py: SomeBool.rtyper_makerepr returns the singleton
        // `bool_repr`.
        SomeValue::Bool(_) => {
            Ok(crate::translator::rtyper::rbool::bool_repr() as std::sync::Arc<dyn Repr>)
        }
        // rfloat.py: SomeFloat.rtyper_makerepr returns the
        // module-global `float_repr`.
        SomeValue::Float(_) => {
            Ok(crate::translator::rtyper::rfloat::float_repr() as std::sync::Arc<dyn Repr>)
        }
        // rfloat.py: SomeSingleFloat.rtyper_makerepr returns a
        // fresh `SingleFloatRepr()` per call.
        SomeValue::SingleFloat(_) => Ok(std::sync::Arc::new(
            crate::translator::rtyper::rfloat::SingleFloatRepr::new(),
        ) as std::sync::Arc<dyn Repr>),
        // rfloat.py: SomeLongFloat.rtyper_makerepr returns a
        // fresh `LongFloatRepr()` per call.
        SomeValue::LongFloat(_) => Ok(std::sync::Arc::new(
            crate::translator::rtyper::rfloat::LongFloatRepr::new(),
        ) as std::sync::Arc<dyn Repr>),
        // rstr.py — `SomeChar.rtyper_makerepr` returns
        // `char_repr`. Pyre dispatches via the module-global singleton.
        SomeValue::Char(_) => {
            Ok(crate::translator::rtyper::rstr::char_repr() as std::sync::Arc<dyn Repr>)
        }
        // rstr.py — `SomeUnicodeCodePoint.rtyper_makerepr`
        // returns `unichar_repr`.
        SomeValue::UnicodeCodePoint(_) => {
            Ok(crate::translator::rtyper::rstr::unichar_repr() as std::sync::Arc<dyn Repr>)
        }
        // rstr.py — `SomeString.rtyper_makerepr` /
        // `SomeUnicodeString.rtyper_makerepr` return the module-global
        // `string_repr` / `unicode_repr` (`lltypesystem/rstr.py` /
        // `:1260`) unconditionally. Pyre's per-method `rtype_*` /
        // `convert_const` overrides land incrementally. Until
        // each one lands, `Repr`'s default impls surface a typed
        // `TyperError` from the per-method dispatch — same upstream
        // surface as `AbstractStringRepr` not yet defining the method.
        SomeValue::String(_) => {
            Ok(crate::translator::rtyper::rstr::string_repr() as std::sync::Arc<dyn Repr>)
        }
        SomeValue::UnicodeString(_) => {
            Ok(crate::translator::rtyper::rstr::unicode_repr() as std::sync::Arc<dyn Repr>)
        }
        // rbytearray.py — `SomeByteArray.rtyper_makerepr`
        // imports and returns `lltypesystem.rbytearray.bytearray_repr`.
        SomeValue::ByteArray(_) => Ok(
            crate::translator::rtyper::lltypesystem::rbytearray::bytearray_repr()
                as std::sync::Arc<dyn Repr>,
        ),
        // rclass.py — SomeInstance.rtyper_makerepr.
        SomeValue::Instance(s) => {
            let rtyper_rc = rtyper.self_rc()?;
            let r_inst = crate::translator::rtyper::rclass::getinstancerepr(
                &rtyper_rc,
                s.classdef.as_ref(),
                crate::translator::rtyper::rclass::Flavor::Gc,
            )?;
            Ok(r_inst as std::sync::Arc<dyn Repr>)
        }
        // rclass.py — SomeException.rtyper_makerepr:
        // `return self.as_SomeInstance().rtyper_makerepr(rtyper)`.
        SomeValue::Exception(s) => {
            let as_instance = s.as_some_instance();
            rtyper_makerepr(&as_instance, rtyper)
        }
        SomeValue::Tuple(s_tuple) => {
            // rtuple.py — `return TupleRepr(rtyper, [rtyper.getrepr(s_item)
            // for s_item in self.items])`.
            let mut items_r: Vec<std::sync::Arc<dyn Repr>> =
                Vec::with_capacity(s_tuple.items.len());
            for s_item in &s_tuple.items {
                items_r.push(rtyper.getrepr(s_item)?);
            }
            // `TupleRepr::new` calls `externalvsinternal` per item which
            // needs the owning `Rc<RPythonTyper>` (route to root
            // `getinstancerepr` for GC InstanceRepr items).
            let rtyper_rc = rtyper.self_rc()?;
            Ok(
                std::sync::Arc::new(crate::translator::rtyper::rtuple::TupleRepr::new(
                    &rtyper_rc, items_r,
                )?) as std::sync::Arc<dyn Repr>,
            )
        }
        SomeValue::List(s_list) => {
            // rlist.py `SomeList.rtyper_makerepr`. Three upstream
            // branches; pyre lands only the non-resized one today:
            //
            //   1. `range_step is not None and not mutated and
            //      not SomeImpossibleValue` → `RangeRepr(range_step)`
            //      (lltypesystem/rrange.py). The step-1 form
            //      (`range(n)` / `range(a, b)`) lands on
            //      [`lltypesystem::rrange::RangeRepr`]; the general const-step != 1
            //      and variable-step (`RANGEST`) lengths need
            //      `ll_rangelen`'s `int_floordiv` (not yet a recognised
            //      low-level op), so they fall to the `Err` arm. A
            //      range-list is NOT array-backed, so this never
            //      mis-lowers as a `FixedSizeListRepr` (whose
            //      `getarraysize` / item-load ops would compile to
            //      invalid reads off a non-array).
            //   2. resized listdef → `ListRepr` (`GcStruct("list",
            //      length, items)`) — data shape + `len` land; the
            //      `append`/`getitem`/`setitem`/resize ops are follow-on
            //      slices (`rlist::ListRepr`).
            //   3. non-resized → `FixedSizeListRepr`.
            //
            // Upstream also passes a LAZY `item_repr = lambda:
            // rtyper.getrepr(listitem.s_value)` for recursive list
            // structures; pyre computes it EAGERLY below. Sound for the
            // current usage (list items are `PyObjectRef`, i.e. the root
            // `InstanceRepr`, never the list type itself), so no eager
            // recursion occurs; the lazy form is a parity follow-up.
            // (`externalvsinternal`'s `gcref=True` arm is likewise
            // deferred — `rclass.rs` — and unreached for `PyObjectRef`
            // items.)
            let (resized, range_step, mutated) = {
                let li_rc = s_list.listdef.listitem_rc();
                let li = li_rc.borrow();
                (li.resized, li.range_step, li.mutated)
            };
            let s_value = s_list.listdef.s_value();
            // rlist.py:45 — RangeRepr is selected ONLY when all three hold:
            // `range_step is not None and not mutated and not
            // SomeImpossibleValue`. A mutated (e.g. setitem'd / resized)
            // or still-impossible range-derived list falls through to the
            // resized→ListRepr / non-resized→FixedSizeListRepr branch
            // (rlist.py:54-57), exactly as a non-range list does.
            let range_repr_step =
                range_step.filter(|_| !mutated && !matches!(s_value, SomeValue::Impossible));
            use crate::translator::rtyper::rlist::{FixedSizeListRepr, ListRepr};
            if let Some(step) = range_repr_step {
                // rlist.py:45 — any non-None range_step selects RangeRepr,
                // constant (`RANGE`) or variable (`step == 0` → `RANGEST`).
                Ok(std::sync::Arc::new(
                    crate::translator::rtyper::lltypesystem::rrange::RangeRepr::new(step)?,
                ) as std::sync::Arc<dyn Repr>)
            } else {
                let item_r = rtyper.getrepr(&s_value)?;
                let rtyper_rc = rtyper.self_rc()?;
                let repr: std::sync::Arc<dyn Repr> = if resized {
                    std::sync::Arc::new(ListRepr::new(&rtyper_rc, item_r)?)
                } else {
                    std::sync::Arc::new(FixedSizeListRepr::new(&rtyper_rc, item_r)?)
                };
                Ok(repr)
            }
        }
        // rdict.py — SomeDict.rtyper_makerepr.  annotator/model.py
        // aliases SomeDict = SomeOrderedDict, so the concrete repr is the
        // lltypesystem OrderedDictRepr built by `somedict_rtyper_makerepr`.
        SomeValue::Dict(s_dict) => {
            crate::translator::rtyper::rdict::somedict_rtyper_makerepr(s_dict, rtyper)
        }
        // rmodel.py — SomeIterator.rtyper_makerepr:
        //   r_container = rtyper.getrepr(self.s_container)
        //   if self.variant and self.variant[0] == "enumerate":
        //       from rpython.rtyper.rrange import EnumerateIteratorRepr
        //       r_baseiter = r_container.make_iterator_repr()
        //       return EnumerateIteratorRepr(r_baseiter, self.variant[1])
        //   return r_container.make_iterator_repr(*self.variant)
        SomeValue::Iterator(s_iter) => {
            let r_container = rtyper.getrepr(s_iter.s_container.as_ref())?;
            let foldable = list_container_foldable(s_iter.s_container.as_ref());
            if s_iter.variant.first().map(String::as_str) == Some("enumerate") {
                let r_baseiter = r_container.make_iterator_repr(&[], foldable)?;
                let const_startindex = match &s_iter.enumerate_start {
                    None => None,
                    Some(crate::flowspace::model::ConstValue::Int(start)) => Some(*start),
                    Some(other) => {
                        return Err(TyperError::message(format!(
                            "enumerate(): non-integer constant start {other:?}"
                        )));
                    }
                };
                Ok(std::sync::Arc::new(
                    crate::translator::rtyper::rrange::EnumerateIteratorRepr::new(
                        r_baseiter,
                        const_startindex,
                    )?,
                ) as std::sync::Arc<dyn Repr>)
            } else {
                r_container.make_iterator_repr(&s_iter.variant, foldable)
            }
        }
        // rpbc.py — SomePBC.rtyper_makerepr. Only the degenerate
        // single-FunctionDesc / can_be_None=False branch is ported
        // today; the remaining arms surface as
        // `MissingRTypeOperation` from [`somepbc_rtyper_makerepr`].
        SomeValue::PBC(s_pbc) => {
            let self_rc = rtyper.self_rc()?;
            crate::translator::rtyper::rpbc::somepbc_rtyper_makerepr(s_pbc, &self_rc)
        }
        // extfunc.py — SomeExternalFunction.rtyper_makerepr
        // returns ExternalFunctionRepr(self, impl, fakeimpl).
        SomeValue::ExternalFunction(s_func) => {
            let repr = s_func.rtyper_makerepr(None, None)?;
            Ok(std::sync::Arc::new(repr) as std::sync::Arc<dyn Repr>)
        }
        // rbuiltin.py — SomeBuiltin.rtyper_makerepr /
        // SomeBuiltinMethod.rtyper_makerepr. Routed into the rbuiltin
        // module which owns the concrete repr types.
        SomeValue::Builtin(_) | SomeValue::BuiltinMethod(_) => {
            crate::translator::rtyper::rbuiltin::dispatch_rtyper_makerepr(s_obj, rtyper)
        }
        // rnone.py: SomeNone.rtyper_makerepr returns the singleton
        // `none_repr`.
        SomeValue::None_(_) => {
            Ok(crate::translator::rtyper::rnone::none_repr() as std::sync::Arc<dyn Repr>)
        }
        // rclass.py — SomeType.rtyper_makerepr returns
        // `get_type_repr(rtyper)`.
        SomeValue::Type(_) => crate::translator::rtyper::rclass::get_type_repr(rtyper),
        SomeValue::Object(_) => Err(TyperError::missing_rtype_operation(
            "SomeObject.rtyper_makerepr — no direct rtyper module; see rpython/annotator/model.py",
        )),
        SomeValue::Property(_) => Err(TyperError::missing_rtype_operation(
            "SomeProperty.rtyper_makerepr — no direct rtyper module; see rpython/annotator/model.py",
        )),
        // rweakref.py:13-17
        SomeValue::WeakRef(_) => super::rweakref::weakref_makerepr(rtyper),
        // `SomeTypeOf` is a subclass of `SomeType` and inherits
        // `rclass.py::__extend__(SomeType).rtyper_makerepr`, which returns
        // `get_type_repr(rtyper)`.
        SomeValue::TypeOf(_) => crate::translator::rtyper::rclass::get_type_repr(rtyper),
        SomeValue::Ptr(ptr) => {
            Ok(std::sync::Arc::new(PtrRepr::new(ptr.ll_ptrtype.clone()))
                as std::sync::Arc<dyn Repr>)
        }
        SomeValue::InteriorPtr(ptr) => Ok(std::sync::Arc::new(InteriorPtrRepr::new(
            ptr.ll_ptrtype.clone(),
        )) as std::sync::Arc<dyn Repr>),
        SomeValue::Address(_) => Ok(address_repr() as std::sync::Arc<dyn Repr>),
        // raddress.py: SomeTypedAddressAccess.rtyper_makerepr = TypedAddressAccessRepr(self.type).
        SomeValue::TypedAddressAccess(s) => Ok(std::sync::Arc::new(TypedAddressAccessRepr::new(
            s.access_type.clone(),
        )) as std::sync::Arc<dyn Repr>),
        SomeValue::LLADTMeth(adtmeth) => {
            Ok(std::sync::Arc::new(LLADTMethRepr::new(adtmeth)) as std::sync::Arc<dyn Repr>)
        }
        // rstring.py / 1264-1266 — SomeStringBuilder /
        // SomeUnicodeBuilder rtyper_makerepr return the module-global
        // stringbuilder_repr / unicodebuilder_repr singletons.
        SomeValue::StringBuilder(_) => Ok(
            crate::translator::rtyper::lltypesystem::rbuilder::stringbuilder_repr()
                as std::sync::Arc<dyn Repr>,
        ),
        SomeValue::UnicodeBuilder(_) => Ok(
            crate::translator::rtyper::lltypesystem::rbuilder::unicodebuilder_repr()
                as std::sync::Arc<dyn Repr>,
        ),
        SomeValue::RustVec(s) => {
            let item_repr = rtyper.getrepr(&s.s_item)?;
            Ok(
                std::sync::Arc::new(super::rrustvec::RustVecRepr::new(item_repr)?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
    }
}

// VoidRepr and SimplePointerRepr must be Display-able via Repr's
// `repr_string()` when formatted through Debug. Since `#[derive(Debug)]`
// already handles it, no additional impl is required. (TyperError's
// Display uses repr_string() through self.repr_string() calls.)
impl fmt::Display for VoidRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

impl fmt::Display for SimplePointerRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::rc::Rc;

    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeInteger, SomePtr, SomeString, SomeValue};
    use crate::flowspace::model::{Block, BlockKey, FunctionGraph, SpaceOperation, Variable};
    use crate::translator::rtyper::llannotation::{SomeInteriorPtr, SomeLLADTMeth};
    use crate::translator::rtyper::lltypesystem::lltype::FuncType;
    use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};

    #[derive(Debug)]
    struct TestRepr {
        state: ReprState,
        lltype: LowLevelType,
        class_name: &'static str,
    }

    impl TestRepr {
        fn new(lltype: LowLevelType, class_name: &'static str) -> Self {
            TestRepr {
                state: ReprState::new(),
                lltype,
                class_name,
            }
        }
    }

    impl Repr for TestRepr {
        fn lowleveltype(&self) -> &LowLevelType {
            &self.lltype
        }

        fn state(&self) -> &ReprState {
            &self.state
        }

        fn class_name(&self) -> &'static str {
            self.class_name
        }
    }

    fn rtyper_for_tests() -> RPythonTyper {
        let ann = RPythonAnnotator::new(None, None, None, false);
        RPythonTyper::new(&ann)
    }

    fn live_rtyper_for_hop() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        (ann, rtyper)
    }

    fn const_string_annotation(value: &str) -> SomeValue {
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str(value)));
        SomeValue::String(s_attr)
    }

    #[test]
    fn getgcflavor_reads_alloc_flavor_from_classdesc_like_rmodel() {
        let classdef = crate::annotator::classdesc::ClassDef::new_standalone("pkg.RawC", None);
        assert_eq!(
            getgcflavor(&classdef).unwrap(),
            super::super::rclass::Flavor::Gc
        );
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::byte_str("raw"));
        assert_eq!(
            getgcflavor(&classdef).unwrap(),
            super::super::rclass::Flavor::Raw
        );
    }

    #[test]
    fn dummy_value_builder_freezes_and_compares_by_rtyper_identity_and_type() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper_a = RPythonTyper::new(&ann);
        let rtyper_b = RPythonTyper::new(&ann);
        let a1 = DummyValueBuilder::new(&rtyper_a, LowLevelType::Signed);
        let a2 = DummyValueBuilder::new(&rtyper_a, LowLevelType::Signed);
        let b = DummyValueBuilder::new(&rtyper_b, LowLevelType::Signed);
        let a_bool = DummyValueBuilder::new(&rtyper_a, LowLevelType::Bool);

        assert!(a1._freeze_());
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_ne!(a1, a_bool);
        assert_eq!(a1.TYPE(), &LowLevelType::Signed);
        // A primitive TYPE is unmallocable — `get_ll_dummyval_obj` only ever
        // builds this for a `Ptr`-to-Struct/Array, so `Signed` errors.
        assert!(a1.ll_dummy_value(&rtyper_a).is_err());

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a1.hash(&mut h1);
        a2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn dummy_value_builder_mallocs_immortal_placeholder_and_caches_by_type() {
        use crate::translator::rtyper::lltypesystem::lltype::Struct;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s = LowLevelType::Struct(Box::new(Struct::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let builder = DummyValueBuilder::new(&rtyper, s.clone());

        let v1 = builder
            .ll_dummy_value(&rtyper)
            .expect("dummy value mallocs");
        assert!(matches!(v1, lltype::LowLevelValue::Ptr(_)));
        assert!(rtyper.cache_dummy_values.borrow().contains_key(&s));
        // second call returns the cached placeholder, not a fresh malloc.
        let v2 = builder.ll_dummy_value(&rtyper).expect("cached dummy value");
        assert_eq!(v1, v2);
        assert_eq!(rtyper.cache_dummy_values.borrow().len(), 1);
    }

    fn empty_hop(rtyper: &Rc<RPythonTyper>, opname: &str) -> HighLevelOp {
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                opname.to_string(),
                Vec::new(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops,
        )
    }

    #[test]
    fn voidrepr_lowleveltype_is_void_and_reprs_match_upstream() {
        // rmodel.py: VoidRepr.lowleveltype = Void.
        let r = VoidRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        // rmodel.py `<%s %s>` formatter.
        assert_eq!(r.repr_string(), "<VoidRepr Void>");
        // rmodel.py compact_repr — "VoidRepr" → "VoidR" replacement,
        // then short_name.
        assert_eq!(r.compact_repr(), "VoidR Void");
    }

    #[test]
    fn setup_transitions_notinitialized_to_finished() {
        // rmodel.py setup: NOTINITIALIZED → INPROGRESS → FINISHED.
        let r = VoidRepr::new();
        assert_eq!(r.state().get(), setupstate::NotInitialized);
        r.setup().expect("setup should succeed on default VoidRepr");
        assert_eq!(r.state().get(), setupstate::Finished);
        // Second call returns immediately (rmodel.py:40-41).
        r.setup().expect("setup should be idempotent once FINISHED");
        assert_eq!(r.state().get(), setupstate::Finished);
    }

    #[test]
    fn setup_on_broken_state_raises_broken_repr_typer_error() {
        // rmodel.py:42-44.
        let r = VoidRepr::new();
        r.state().set(setupstate::Broken);
        let err = r.setup().unwrap_err();
        assert!(err.is_broken_repr());
        assert!(err.to_string().contains("<VoidRepr Void>"));
    }

    #[test]
    #[should_panic(expected = "recursive invocation of Repr setup()")]
    fn setup_on_inprogress_panics_like_upstream_assertion() {
        // rmodel.py:45-47 uses `raise AssertionError` — pyre panics.
        // Pyre's CAS+thread-tracking model keys recursion detection on
        // owner==current-thread, so simulate the "this thread is already
        // in setup" state via the `_force_inprogress_for_current_thread`
        // test helper instead of poking `state.set` (which would leave
        // owner=None and cause the cross-thread wait loop to spin).
        let r = VoidRepr::new();
        r.state()._force_inprogress_for_current_thread();
        let _ = r.setup();
    }

    #[test]
    fn set_setup_delayed_toggles_state_both_ways() {
        // rmodel.py set_setup_delayed.
        let r = VoidRepr::new();
        r.set_setup_delayed(true);
        assert_eq!(r.state().get(), setupstate::Delayed);
        assert!(r.is_setup_delayed());
        r.set_setup_delayed(false);
        assert_eq!(r.state().get(), setupstate::NotInitialized);
        assert!(!r.is_setup_delayed());
    }

    #[test]
    fn set_setup_maybe_delayed_only_promotes_from_notinitialized() {
        // rmodel.py set_setup_maybe_delayed.
        let r = VoidRepr::new();
        assert!(r.set_setup_maybe_delayed());
        assert_eq!(r.state().get(), setupstate::Delayed);
        // Already Delayed: returns true (the membership check) without
        // changing state.
        assert!(r.set_setup_maybe_delayed());
    }

    /// rmodel.py — `Repr.get_ll_eq_function` / rmodel.py —
    /// `Repr.get_ll_hash_function` raise `TyperError` at the base.
    /// Concrete Reprs (Integer/Float/Bool/None/String/Tuple/...)
    /// override; the bare base path must surface the error so
    /// callers (e.g. `gen_eq_function` per-item dispatch) loudly fail
    /// for unsupported Reprs instead of silently using a wrong path.
    #[test]
    fn base_repr_get_ll_eq_and_hash_function_defaults_raise_typererror() {
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr, PtrTarget};
        let rtyper = rtyper_for_tests();
        // SimplePointerRepr has no get_ll_eq_function / get_ll_hash_function
        // override → falls back to the base default which raises.
        let r = SimplePointerRepr::new(LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Opaque(OpaqueType::gc("GCREF")),
        })));
        let eq_err = r.get_ll_eq_function(&rtyper).unwrap_err();
        assert!(
            eq_err.to_string().contains("no equality function"),
            "got {eq_err:?}"
        );
        let hash_err = r.get_ll_hash_function(&rtyper).unwrap_err();
        assert!(
            hash_err.to_string().contains("no hashing function"),
            "got {hash_err:?}"
        );
    }

    /// rmodel.py — `VoidRepr` overrides `get_ll_eq_function`
    /// (returns None), `get_ll_hash_function` (returns `ll_hash_void`),
    /// and aliases `get_ll_fasthash_function = get_ll_hash_function`.
    #[test]
    fn voidrepr_eq_none_hash_and_fasthash_resolve_ll_hash_void() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = VoidRepr::new();
        assert!(r.get_ll_eq_function(&rtyper).unwrap().is_none());

        let hashfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("ll_hash_void helper");
        assert_eq!(hashfn.name, "ll_hash_void");
        assert_eq!(hashfn.args, vec![LowLevelType::Void]);
        assert_eq!(hashfn.result, LowLevelType::Signed);

        let fastfn = r
            .get_ll_fasthash_function(&rtyper)
            .unwrap()
            .expect("ll_hash_void helper (fasthash alias)");
        assert_eq!(fastfn.name, "ll_hash_void");
    }

    #[test]
    fn convert_const_on_voidrepr_accepts_any_value() {
        // rmodel.py — `convert_const` delegates to
        // `_contains_value`, and `Void._contains_value` (lltype.py)
        // returns True for any value. So `VoidRepr` accepts None, Int,
        // Bool, etc. as valid constants of lowlevel type Void.
        let r = VoidRepr::new();
        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c = r.convert_const(&ConstValue::Int(42)).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    #[test]
    fn inputconst_wraps_value_and_records_lowleveltype() {
        // rmodel.py inputconst.
        let r = VoidRepr::new();
        let c = inputconst(&r, &ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c2 = inputconst_from_lltype(&LowLevelType::Signed, &ConstValue::Int(7)).unwrap();
        assert_eq!(c2.concretetype.as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn inputconst_from_lltype_rejects_mismatched_constant() {
        let err = inputconst_from_lltype(&LowLevelType::Bool, &ConstValue::Int(0)).unwrap_err();
        assert!(err.to_string().contains("inputconst"));
    }

    #[test]
    fn mangle_uses_internal_prefix_for_dollar_names() {
        // rmodel.py mangle.
        assert_eq!(mangle("cls", "$hidden"), "clsinternal_hidden");
        assert_eq!(mangle("cls", "method"), "cls_method");
    }

    #[test]
    fn impossible_repr_is_shared_singleton_pointer() {
        // `impossible_repr = VoidRepr()` module-level singleton
        // (rmodel.py). Two calls return Arcs to the same instance.
        let a = impossible_repr();
        let b = impossible_repr();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn simple_pointer_repr_only_accepts_none_constant() {
        // rmodel.py SimplePointerRepr.
        let ptr_ty = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            },
        ));
        let r = SimplePointerRepr::new(ptr_ty.clone());
        assert_eq!(r.lowleveltype(), &ptr_ty);

        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&ptr_ty));

        let err = r.convert_const(&ConstValue::Int(0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("only supports None as prebuilt constant"),
            "expected upstream phrase; got {err}"
        );
    }

    #[test]
    fn ptrrepr_rtype_getattr_emits_getfield_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget, Struct};

        let ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "getattr");
        let v_ptr = Variable::new();
        v_ptr.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v
            .borrow_mut()
            .extend([Hlvalue::Variable(v_ptr), void_field_const("x")]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            const_string_annotation("x"),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(impossible_repr() as Arc<dyn Repr>)]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = r_ptr.rtype_getattr(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getfield");
        assert_eq!(ops.ops[0].args.len(), 2);
    }

    #[test]
    fn ptrrepr_rtype_setattr_emits_setfield_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget, Struct};

        let ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "setattr");
        let v_ptr = Variable::new();
        v_ptr.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_ptr),
            void_field_const("x"),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(3),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            const_string_annotation("x"),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(r_ptr_dyn),
            Some(impossible_repr() as Arc<dyn Repr>),
            Some(r_signed),
        ]);

        assert!(r_ptr.rtype_setattr(&hop).unwrap().is_none());
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "setfield");
        assert_eq!(ops.ops[0].args.len(), 3);
    }

    #[test]
    fn ptrrepr_rtype_simple_call_emits_indirect_call_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            }),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "simple_call");
        let v_func = Variable::new();
        v_func.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_func),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(5),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = r_ptr.rtype_simple_call(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "indirect_call");
        assert_eq!(ops.ops[0].args.len(), 3);
    }

    #[test]
    fn ptrrepr_rtype_simple_call_records_extra_call_for_direct_graph_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
        use crate::translator::translator::CallGraphKey;

        let (ann, rtyper) = live_rtyper_for_hop();
        let helper = rtyper
            .lowlevel_helper_function("callee", vec![LowLevelType::Signed], LowLevelType::Signed)
            .expect("helper graph");
        let helper_graph = helper.graph.as_ref().expect("helper graph must exist");
        let func_ptr = rtyper.getcallable(helper_graph).expect("getcallable");
        let PtrTarget::Func(func_type) = func_ptr._TYPE.TO.clone() else {
            panic!("helper callable must be a function pointer");
        };
        let r_ptr = Arc::new(PtrRepr::new(func_ptr._TYPE.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
        let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
            .expect("function ptr constant");

        let caller_start = Block::shared(Vec::new());
        let caller = Rc::new(RefCell::new(FunctionGraph::new(
            "caller",
            caller_start.clone(),
        )));
        ann.translator.graphs.borrow_mut().push(caller.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&caller_start), Some(caller.clone()));
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            Some(caller_start.clone()),
        )));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(func_type.result.clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call",
                vec![
                    Hlvalue::Constant(c_func),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(5),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(r_ptr.ptrtype().clone())),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = r_ptr.rtype_simple_call(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        assert_eq!(llops.borrow().ops[0].opname, "direct_call");

        let callgraph = &ann.translator;
        let callgraph = callgraph.callgraph.borrow();
        let expected_key = CallGraphKey {
            caller: crate::flowspace::model::GraphKey::of(&caller),
            callee: crate::flowspace::model::GraphKey::of(&helper_graph.graph),
            tag_block: BlockKey::of(&caller_start),
            tag_index: 0,
        };
        let edge = callgraph
            .get(&expected_key)
            .expect("direct call should record callgraph edge");
        assert_eq!(
            crate::flowspace::model::GraphKey::of(&edge.caller),
            crate::flowspace::model::GraphKey::of(&caller)
        );
        assert_eq!(
            crate::flowspace::model::GraphKey::of(&edge.callee),
            crate::flowspace::model::GraphKey::of(&helper_graph.graph)
        );
    }

    #[test]
    fn ptrrepr_rtype_len_fixed_array_returns_constant_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{FixedSizeArray, Ptr, PtrTarget};

        let r_ptr = PtrRepr::new(Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 4)),
        });
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "len");
        let out = r_ptr.rtype_len(&hop).unwrap().unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("fixed array len should return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(4));
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
    }

    #[test]
    fn ptrrepr_rtype_getitem_emits_getarrayitem_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let hop = empty_hop(&rtyper, "getitem");
        let v_array = Variable::new();
        v_array.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_array),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(1),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = r_ptr.rtype_getitem(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getarrayitem");
    }

    #[test]
    fn ptrrepr_rtype_setitem_emits_setarrayitem_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let hop = empty_hop(&rtyper, "setitem");
        let v_array = Variable::new();
        v_array.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_array),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(1),
                LowLevelType::Signed,
            )),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(9),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed.clone()), Some(r_signed)]);

        assert!(r_ptr.rtype_setitem(&hop).unwrap().is_none());
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "setarrayitem");
    }

    #[test]
    fn ptrrepr_rtype_getitem_container_result_allocates_interior_ptr_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, PtrTarget, Struct};

        let item = Struct::new("Item", vec![("x".into(), LowLevelType::Signed)]);
        let ptr = Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(item)))),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let r_result: Arc<dyn Repr> = Arc::new(TestRepr::new(
            LowLevelType::Ptr(Box::new(ptr.clone())),
            "InteriorPtrRepr",
        ));
        let hop = empty_hop(&rtyper, "getitem");
        let v_array = Variable::new();
        v_array.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_array),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(2),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed)]);
        *hop.r_result.borrow_mut() = Some(r_result);

        let result = r_ptr.rtype_getitem(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 3);
        assert_eq!(ops.ops[0].opname, "malloc");
        assert_eq!(ops.ops[1].opname, "setfield");
        assert_eq!(ops.ops[2].opname, "setfield");
    }

    #[test]
    fn ptrrepr_rtype_eq_ne_emit_pointer_comparisons_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "eq");
        let v_left = Variable::new();
        v_left.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        let v_right = Variable::new();
        v_right.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v
            .borrow_mut()
            .extend([Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr.clone())),
            SomeValue::Ptr(SomePtr::new(ptr)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn.clone()), Some(r_ptr_dyn)]);

        assert!(r_ptr.rtype_eq(&hop).unwrap().is_some());
        assert!(r_ptr.rtype_ne(&hop).unwrap().is_some());
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops[0].opname, "ptr_eq");
        assert_eq!(ops.ops[1].opname, "ptr_ne");
    }

    #[test]
    fn ptr_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            }),
        };
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(&SomeValue::Ptr(SomePtr::new(ptr.clone())), &rtyper).unwrap();
        assert_eq!(repr.class_name(), "PtrRepr");
        assert_eq!(repr.lowleveltype(), &LowLevelType::Ptr(Box::new(ptr)));
    }

    /// rlist.py:45 — RangeRepr is selected only when `range_step is not
    /// None and not mutated and not SomeImpossibleValue`; a mutated or
    /// still-impossible range-derived list falls through to
    /// FixedSizeListRepr / the deferred resized ListRepr, like any other
    /// list. Regression guard against gating on `range_step.is_some()`
    /// alone.
    #[test]
    fn somelist_rangerepr_gated_on_not_mutated_and_concrete_item() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList};

        let rtyper = rtyper_for_tests();
        let item = SomeValue::Integer(SomeInteger::new(false, false));

        // step==1, not mutated, concrete item → RangeRepr.
        let ldef = ListDef::new(None, item.clone(), false, false);
        ldef.listitem_rc().borrow_mut().range_step = Some(1);
        let repr = rtyper_makerepr(&SomeValue::List(SomeList::new(ldef)), &rtyper).unwrap();
        assert_eq!(repr.class_name(), "RangeRepr");

        // step==1 but mutated → falls through to the non-range branch
        // (FixedSizeListRepr), NOT RangeRepr.
        let ldef_mut = ListDef::new(None, item.clone(), true, false);
        ldef_mut.listitem_rc().borrow_mut().range_step = Some(1);
        let repr_mut = rtyper_makerepr(&SomeValue::List(SomeList::new(ldef_mut)), &rtyper);
        assert!(
            repr_mut
                .map(|r| r.class_name() != "RangeRepr")
                .unwrap_or(true),
            "mutated range-list must not select RangeRepr"
        );

        // step==1 but resized (⇒ mutated) → selects resized ListRepr, never RangeRepr.
        let ldef_resized = ListDef::new(None, item, false, true);
        ldef_resized.listitem_rc().borrow_mut().range_step = Some(1);
        let repr_resized = rtyper_makerepr(&SomeValue::List(SomeList::new(ldef_resized)), &rtyper);
        assert!(
            repr_resized
                .map(|r| r.class_name() != "RangeRepr")
                .unwrap_or(true),
            "resized range-list must not select RangeRepr"
        );

        // step==1, not mutated, but impossible item → not RangeRepr.
        let ldef_imp = ListDef::new(None, SomeValue::Impossible, false, false);
        ldef_imp.listitem_rc().borrow_mut().range_step = Some(1);
        let r_imp = rtyper_makerepr(&SomeValue::List(SomeList::new(ldef_imp)), &rtyper);
        assert!(
            r_imp.map(|r| r.class_name() != "RangeRepr").unwrap_or(true),
            "impossible-item range-list must not select RangeRepr"
        );
    }

    #[test]
    fn interior_ptr_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, PtrTarget, Struct,
        };

        let parent = LowLevelType::Struct(Box::new(Struct::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let interior = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Field("x".into())],
        };
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(
            &SomeValue::InteriorPtr(SomeInteriorPtr::new(interior)),
            &rtyper,
        )
        .unwrap();
        assert_eq!(repr.class_name(), "InteriorPtrRepr");
        assert_eq!(
            repr.lowleveltype(),
            &LowLevelType::Ptr(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::Ptr {
                    TO: PtrTarget::Struct(match parent {
                        LowLevelType::Struct(t) => *t,
                        _ => unreachable!(),
                    }),
                }
            ))
        );
    }

    #[test]
    fn interior_ptr_repr_records_offsets_like_rptr_init() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, PtrTarget, Struct,
        };

        let parent = LowLevelType::Struct(Box::new(Struct::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let repr = InteriorPtrRepr::new(InteriorPtr {
            PARENTTYPE: Box::new(parent),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Field("x".into())],
        });
        assert_eq!(repr.v_offsets.len(), 1);
        let Some(c_offset) = &repr.v_offsets[0] else {
            panic!("field offset should be stored as a Void constant");
        };
        assert_eq!(c_offset.value, ConstValue::byte_str("x"));
        assert_eq!(c_offset.concretetype, Some(LowLevelType::Void));
        assert!(matches!(repr.parentptrtype.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn interiorptr_rtype_getitem_emits_getinteriorfield_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, InteriorOffset, InteriorPtr, Struct,
        };

        let array_type = LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed)));
        let parent = LowLevelType::Struct(Box::new(Struct::new(
            "Holder",
            vec![("items".into(), array_type.clone())],
        )));
        let iptr = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(array_type),
            offsets: vec![InteriorOffset::Field("items".into())],
        };
        let r_iptr = Arc::new(InteriorPtrRepr::new(iptr.clone()));
        let r_iptr_dyn: Arc<dyn Repr> = r_iptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let hop = empty_hop(&rtyper, "getitem");
        let v_self = Variable::new();
        v_self.set_concretetype(Some(r_iptr.lowleveltype().clone()));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_self),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(0),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::InteriorPtr(SomeInteriorPtr::new(iptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_iptr_dyn), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = r_iptr.rtype_getitem(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getinteriorfield");
    }

    #[test]
    fn interiorptr_rtype_setitem_emits_setinteriorfield_like_rptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, InteriorOffset, InteriorPtr, Struct,
        };

        let array_type = LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed)));
        let parent = LowLevelType::Struct(Box::new(Struct::new(
            "Holder",
            vec![("items".into(), array_type.clone())],
        )));
        let iptr = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(array_type),
            offsets: vec![InteriorOffset::Field("items".into())],
        };
        let r_iptr = Arc::new(InteriorPtrRepr::new(iptr.clone()));
        let r_iptr_dyn: Arc<dyn Repr> = r_iptr.clone();
        let (_ann, rtyper) = live_rtyper_for_hop();
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let hop = empty_hop(&rtyper, "setitem");
        let v_self = Variable::new();
        v_self.set_concretetype(Some(r_iptr.lowleveltype().clone()));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_self),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(0),
                LowLevelType::Signed,
            )),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(8),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::InteriorPtr(SomeInteriorPtr::new(iptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_iptr_dyn), Some(r_signed.clone()), Some(r_signed)]);

        assert!(r_iptr.rtype_setitem(&hop).unwrap().is_none());
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "setinteriorfield");
    }

    #[test]
    #[should_panic]
    fn interior_ptr_repr_rejects_multiple_item_offsets_like_rptr_init() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, InteriorOffset, InteriorPtr};

        let _ = InteriorPtrRepr::new(InteriorPtr {
            PARENTTYPE: Box::new(LowLevelType::Array(Box::new(Array::gc(
                LowLevelType::Signed,
            )))),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Index(0), InteriorOffset::Index(1)],
        });
    }

    #[test]
    fn lladtmeth_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelPointerType, Ptr, PtrTarget,
        };

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Signed,
            }),
        };
        let adtmeth = SomeLLADTMeth::new(LowLevelPointerType::Ptr(ptr.clone()), ConstValue::None);
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(&SomeValue::LLADTMeth(adtmeth), &rtyper).unwrap();
        assert_eq!(repr.class_name(), "LLADTMethRepr");
        assert_eq!(repr.lowleveltype(), &LowLevelType::Ptr(Box::new(ptr)));
    }

    #[test]
    fn repr_default_predicates_match_upstream_defaults() {
        // rmodel.py `_freeze_` → True.
        // rmodel.py `special_uninitialized_value` → None.
        // rmodel.py `can_ll_be_null` → True.
        let r = VoidRepr::new();
        assert!(r.freeze());
        assert!(r.special_uninitialized_value().is_none());
        assert!(r.can_ll_be_null());
    }

    // R5 — SomeInstance / SomeException / SomeType rtyper_make{key,repr}.

    #[test]
    fn rtyper_makekey_someinstance_uses_classdef_identity() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::description::ClassDefKey;
        use crate::annotator::model::SomeInstance;
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let s1 = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s2 = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert_eq!(rtyper_makekey(&s1), rtyper_makekey(&s2));
        assert_eq!(
            rtyper_makekey(&s1),
            ReprKey::Instance(Some(ClassDefKey::from_classdef(&classdef)))
        );

        let other = ClassDef::new_standalone("pkg.D", None);
        let s3 = SomeValue::Instance(SomeInstance::new(
            Some(other),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert_ne!(rtyper_makekey(&s1), rtyper_makekey(&s3));
    }

    #[test]
    fn rtyper_makekey_sometype_is_variant_singleton() {
        use crate::annotator::model::SomeType;
        let s = SomeValue::Type(SomeType::new());
        assert_eq!(rtyper_makekey(&s), ReprKey::Type);
    }

    #[test]
    fn rtyper_makekey_sometypeof_is_distinct_variant_singleton() {
        use crate::annotator::model::SomeTypeOf;
        use crate::flowspace::model::Variable;
        let first = SomeValue::TypeOf(SomeTypeOf::new(vec![Rc::new(Variable::new())]));
        let second = SomeValue::TypeOf(SomeTypeOf::new(vec![Rc::new(Variable::new())]));
        assert_eq!(rtyper_makekey(&first), ReprKey::TypeOf);
        assert_eq!(rtyper_makekey(&second), ReprKey::TypeOf);
        assert_ne!(rtyper_makekey(&first), ReprKey::Type);
    }

    #[test]
    fn rtyper_makerepr_someinstance_returns_instance_repr_when_initialized() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeInstance;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("init");

        let classdef = ClassDef::new_standalone("pkg.C", None);
        let sv = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "InstanceRepr");
    }

    /// Dependency anchor for the typed-ref-someptr-followup epic
    /// (`annotation_state.rs` `valuetype_to_someshell::Ref` arm).
    ///
    /// `SomeInstance(classdef=None)` is the projection currently used by
    /// `valuetype_to_someshell` for every `ValueType::Ref` operand;
    /// `rclass.py __extend__` routes `classdef=None` through
    /// `getinstancerepr(rtyper, None, Gc)` -> `buildinstancerepr` with
    /// `unboxed=[]` / `virtualizable=false` and produces a regular
    /// `InstanceRepr` for the abstract `object`-root.  This test locks
    /// that the rtyper side handles the classdef-less projection
    /// cleanly, so subsequent slices that flip producer-side
    /// projections to `SomeValue::Ptr(SomePtr::new(..))` or
    /// `SomeInstance(classdef=Some(..))` have a known-good fallback
    /// they are replacing rather than removing.
    #[test]
    fn rtyper_makerepr_someinstance_classdef_none_returns_instance_repr() {
        use crate::annotator::model::SomeInstance;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("init");

        let sv = SomeValue::Instance(SomeInstance::new(
            None,
            false,
            std::collections::BTreeMap::new(),
        ));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "InstanceRepr");
    }

    #[test]
    fn rtyper_makerepr_sometype_returns_rootclass_repr_when_initialized() {
        use crate::annotator::model::SomeType;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("init");

        let sv = SomeValue::Type(SomeType::new());
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "RootClassRepr");
    }

    #[test]
    fn rtyper_makerepr_sometypeof_inherits_rootclass_repr() {
        use crate::annotator::model::SomeTypeOf;
        use crate::flowspace::model::Variable;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("init");

        let sv = SomeValue::TypeOf(SomeTypeOf::new(vec![Rc::new(Variable::new())]));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "RootClassRepr");
    }

    /// rstr.py — `SomeString.rtyper_makerepr` returns the
    /// module-global `string_repr` (`lltypesystem/rstr.py`).
    #[test]
    fn rtyper_makerepr_somestring_returns_string_repr_singleton() {
        use crate::annotator::model::SomeString;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));

        let sv = SomeValue::String(SomeString::new(false, false));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "StringRepr");
        assert_eq!(
            repr.repr_class_id(),
            crate::translator::rtyper::pairtype::ReprClassId::StringRepr
        );

        let again = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert!(std::sync::Arc::ptr_eq(
            &(repr as std::sync::Arc<dyn Repr>),
            &(again as std::sync::Arc<dyn Repr>),
        ));
    }

    /// rstr.py — `SomeUnicodeString.rtyper_makerepr` returns
    /// the module-global `unicode_repr` (`lltypesystem/rstr.py`).
    #[test]
    fn rtyper_makerepr_someunicodestring_returns_unicode_repr_singleton() {
        use crate::annotator::model::SomeUnicodeString;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));

        let sv = SomeValue::UnicodeString(SomeUnicodeString::new(false, false));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert_eq!(repr.class_name(), "UnicodeRepr");
        assert_eq!(
            repr.repr_class_id(),
            crate::translator::rtyper::pairtype::ReprClassId::UnicodeRepr
        );

        let again = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr");
        assert!(std::sync::Arc::ptr_eq(
            &(repr as std::sync::Arc<dyn Repr>),
            &(again as std::sync::Arc<dyn Repr>),
        ));
    }

    /// `SomeByteArray.rtyper_makerepr` imports and returns
    /// `lltypesystem.rbytearray.bytearray_repr` (`rbytearray.py`).
    #[test]
    fn rtyper_makerepr_somebytearray_returns_bytearray_repr() {
        use crate::annotator::model::SomeByteArray;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));

        let sv = SomeValue::ByteArray(SomeByteArray::new(false));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("ByteArray repr");
        assert_eq!(repr.class_name(), "ByteArrayRepr");
        assert_eq!(
            repr.repr_class_id(),
            crate::translator::rtyper::pairtype::ReprClassId::ByteArrayRepr
        );
    }

    /// rdict.py dispatch — `rtyper_makerepr(SomeValue::Dict)` wires
    /// through to `somedict_rtyper_makerepr` (annotator/model.py:416
    /// aliases SomeDict = SomeOrderedDict), yielding the lltypesystem
    /// `OrderedDictRepr`.  Covers the dispatch arm, not just the factory.
    #[test]
    fn rtyper_makerepr_somedict_returns_ordered_dict_repr() {
        use crate::annotator::dictdef::DictDef;
        use crate::annotator::model::{SomeDict, SomeInteger, SomeString};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");

        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let sv = SomeValue::Dict(SomeDict::new(dictdef));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("dict repr");
        assert_eq!(repr.class_name(), "OrderedDictRepr");
        assert_eq!(
            repr.repr_class_id(),
            crate::translator::rtyper::pairtype::ReprClassId::OrderedDictRepr
        );
    }

    // SomePBC.rtyper_makekey (rpbc.py).

    fn pbc_test_desc_function(
        bk: &Rc<crate::annotator::bookkeeper::Bookkeeper>,
        name: &str,
    ) -> crate::annotator::description::DescEntry {
        use crate::annotator::description::{DescEntry, FunctionDesc};
        use crate::flowspace::argument::Signature;
        DescEntry::function(Rc::new(std::cell::RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))))
    }

    #[test]
    fn rtyper_makekey_somepbc_keys_by_sorted_descs_and_can_be_none() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::model::SomePBC;
        let bk = Rc::new(Bookkeeper::new());
        let f = pbc_test_desc_function(&bk, "f");
        let g = pbc_test_desc_function(&bk, "g");

        let s1 = SomeValue::PBC(SomePBC::new(vec![f.clone(), g.clone()], false));
        // Upstream `lst.sort()` — inserting the same set in reverse
        // order must yield the same cache key.
        let s2 = SomeValue::PBC(SomePBC::new(vec![g, f.clone()], false));
        assert_eq!(rtyper_makekey(&s1), rtyper_makekey(&s2));

        // Differing `can_be_none` must separate entries.
        let s3 = SomeValue::PBC(SomePBC::new(vec![f.clone()], true));
        let s4 = SomeValue::PBC(SomePBC::new(vec![f], false));
        assert_ne!(rtyper_makekey(&s3), rtyper_makekey(&s4));
    }

    #[test]
    fn rtyper_makekey_somepbc_nests_subset_of_recursive_key() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::model::SomePBC;
        let bk = Rc::new(Bookkeeper::new());
        let f = pbc_test_desc_function(&bk, "f");

        let parent = SomePBC::new(vec![f.clone()], false);
        let with_subset =
            SomePBC::with_subset(vec![f.clone()], false, Some(Box::new(parent.clone())));
        let key = rtyper_makekey(&SomeValue::PBC(with_subset));
        match key {
            ReprKey::PBC {
                subset_of: Some(sub),
                ..
            } => {
                // The nested key is the plain no-subset SomePBC's key.
                let bare_key = rtyper_makekey(&SomeValue::PBC(parent));
                assert_eq!(*sub, bare_key);
            }
            other => panic!("expected ReprKey::PBC with subset_of=Some, got {other:?}"),
        }
    }

    // SomeBuiltin / SomeBuiltinMethod.rtyper_makekey (rbuiltin.py).

    #[test]
    fn rtyper_makekey_someiterator_recursively_keys_on_container_and_variant() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeIterator, SomeList};

        let ldef = ListDef::new(None, SomeValue::Impossible, false, false);
        let container = SomeValue::List(SomeList::new(ldef.clone()));

        // Two iterators over the same container with the same variant:
        // same key.
        let a = SomeValue::Iterator(SomeIterator::new(container.clone(), vec!["items".into()]));
        let b = SomeValue::Iterator(SomeIterator::new(container.clone(), vec!["items".into()]));
        assert_eq!(rtyper_makekey(&a), rtyper_makekey(&b));

        // Differing variant separates them (upstream `self.variant`
        // tuple inequality).
        let c = SomeValue::Iterator(SomeIterator::new(container, vec!["keys".into()]));
        assert_ne!(rtyper_makekey(&a), rtyper_makekey(&c));
    }

    #[test]
    fn rtyper_makekey_someweakref_is_class_tag_singleton() {
        use crate::annotator::model::SomeWeakRef;
        let a = SomeValue::WeakRef(SomeWeakRef::new(None));
        let b = SomeValue::WeakRef(SomeWeakRef::new(None));
        assert_eq!(rtyper_makekey(&a), rtyper_makekey(&b));
        assert_eq!(rtyper_makekey(&a), ReprKey::WeakRef);
    }

    #[test]
    fn rtyper_makekey_sometuple_keys_on_element_shape_recursively() {
        use crate::annotator::model::{SomeBool, SomeInteger, SomeTuple};
        let t_bool = SomeValue::Bool(SomeBool::new());
        let t_int = SomeValue::Integer(SomeInteger::new(false, false));

        // Two tuples with the same element shape collapse.
        let a = SomeValue::Tuple(SomeTuple::new(vec![t_bool.clone(), t_int.clone()]));
        let b = SomeValue::Tuple(SomeTuple::new(vec![t_bool.clone(), t_int.clone()]));
        assert_eq!(rtyper_makekey(&a), rtyper_makekey(&b));

        // Changing one element's tag separates them.
        let c = SomeValue::Tuple(SomeTuple::new(vec![t_int.clone(), t_bool.clone()]));
        assert_ne!(rtyper_makekey(&a), rtyper_makekey(&c));
    }

    #[test]
    fn rtyper_makekey_somelist_keys_on_listitem_pointer_identity() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::SomeList;

        let ldef = ListDef::new(None, SomeValue::Impossible, false, false);
        let a = SomeValue::List(SomeList::new(ldef.clone()));
        let b = SomeValue::List(SomeList::new(ldef));
        // Same `ListDef.inner` Rc — same cache entry.
        assert_eq!(rtyper_makekey(&a), rtyper_makekey(&b));

        // Fresh ListDef yields a different listitem identity → separate key.
        let other = SomeValue::List(SomeList::new(ListDef::new(
            None,
            SomeValue::Impossible,
            false,
            false,
        )));
        assert_ne!(rtyper_makekey(&a), rtyper_makekey(&other));
    }

    #[test]
    fn rtyper_makekey_somedict_keys_on_dictkey_and_dictvalue_identities() {
        use crate::annotator::dictdef::DictDef;
        use crate::annotator::model::SomeDict;

        let ddef = DictDef::new(
            None,
            SomeValue::Impossible,
            SomeValue::Impossible,
            false,
            false,
            false,
        );
        let a = SomeValue::Dict(SomeDict::new(ddef.clone()));
        let b = SomeValue::Dict(SomeDict::new(ddef));
        assert_eq!(rtyper_makekey(&a), rtyper_makekey(&b));

        // Fresh DictDef = different key/value identity.
        let other = SomeValue::Dict(SomeDict::new(DictDef::new(
            None,
            SomeValue::Impossible,
            SomeValue::Impossible,
            false,
            false,
            false,
        )));
        assert_ne!(rtyper_makekey(&a), rtyper_makekey(&other));
    }

    #[test]
    fn rtyper_makekey_somelist_sets_listitem_dont_change_any_more() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::SomeList;

        // Seed a mutable-origin ListDef by manually resetting the flag
        // so the makekey effect is observable (upstream mirrors the
        // default-True case from bookkeeper=None, so exercise the
        // bookkeeper-owned shape by flipping it back to False first).
        let ldef = ListDef::new(None, SomeValue::Impossible, false, false);
        let listitem = ldef.inner.listitem.borrow().clone();
        listitem.borrow_mut().dont_change_any_more = false;
        let v = SomeValue::List(SomeList::new(ldef));
        let _ = rtyper_makekey(&v);
        assert!(listitem.borrow().dont_change_any_more);
    }

    #[test]
    fn rtyper_makekey_somedict_sets_dictkey_and_dictvalue_dont_change_any_more() {
        use crate::annotator::dictdef::DictDef;
        use crate::annotator::model::SomeDict;

        let ddef = DictDef::new(
            None,
            SomeValue::Impossible,
            SomeValue::Impossible,
            false,
            false,
            false,
        );
        let dictkey = ddef.inner.dictkey.borrow().clone();
        let dictvalue = ddef.inner.dictvalue.borrow().clone();
        dictkey.borrow_mut().dont_change_any_more = false;
        dictvalue.borrow_mut().dont_change_any_more = false;
        let v = SomeValue::Dict(SomeDict::new(ddef));
        let _ = rtyper_makekey(&v);
        assert!(dictkey.borrow().dont_change_any_more);
        assert!(dictvalue.borrow().dont_change_any_more);
    }

    #[test]
    fn rtyper_makekey_sometring_family_keys_per_class_tag() {
        use crate::annotator::model::{
            SomeByteArray, SomeChar, SomeString, SomeUnicodeCodePoint, SomeUnicodeString,
        };
        assert_eq!(
            rtyper_makekey(&SomeValue::String(SomeString::new(false, false))),
            ReprKey::String
        );
        assert_eq!(
            rtyper_makekey(&SomeValue::UnicodeString(SomeUnicodeString::new(
                false, false
            ))),
            ReprKey::UnicodeString
        );
        assert_eq!(
            rtyper_makekey(&SomeValue::Char(SomeChar::new(false))),
            ReprKey::Char
        );
        assert_eq!(
            rtyper_makekey(&SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(
                false
            ))),
            ReprKey::UnicodeCodePoint
        );
        assert_eq!(
            rtyper_makekey(&SomeValue::ByteArray(SomeByteArray::new(false))),
            ReprKey::ByteArray
        );
    }

    #[test]
    fn rtyper_makekey_somebuiltin_with_hostobject_const_uses_identity_id() {
        use crate::annotator::model::SomeBuiltin;
        use crate::flowspace::model::{ConstValue, Constant, HostObject};
        let host = HostObject::new_builtin_callable("len");
        let mut sb = SomeBuiltin::new("len", None, None);
        sb.base.const_box = Some(Constant::new(ConstValue::HostObject(host.clone())));

        // Two clones of the same SomeBuiltin share const identity →
        // collapse to the same cache entry.
        let mut sb2 = SomeBuiltin::new("len", None, None);
        sb2.base.const_box = Some(Constant::new(ConstValue::HostObject(host.clone())));
        let key_a = rtyper_makekey(&SomeValue::Builtin(sb));
        let key_b = rtyper_makekey(&SomeValue::Builtin(sb2));
        assert_eq!(key_a, key_b);
        assert!(matches!(
            key_a,
            ReprKey::Builtin(Some(BuiltinConstKey::Host(_)))
        ));
    }

    #[test]
    fn rtyper_makekey_somebuiltin_without_const_uses_none_sentinel() {
        use crate::annotator::model::SomeBuiltin;
        let sb = SomeBuiltin::new("len", None, None);
        assert_eq!(
            rtyper_makekey(&SomeValue::Builtin(sb)),
            ReprKey::Builtin(None),
        );
    }

    #[test]
    fn rtyper_makekey_somebuiltin_with_llptr_const_preserves_extregistry_entry_identity() {
        // Upstream rbuiltin.py remaps `extregistry.is_registered(const)`
        // through `extregistry.lookup(const)`. The resulting
        // ExtRegistryEntry hashes on `(entry.__class__, type, instance)`,
        // so two distinct `_ptr` consts must not collapse.
        use crate::annotator::model::SomeBuiltin;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::extregistry::ExtRegistryEntryKey;
        use crate::translator::rtyper::lltypesystem::lltype::{
            _ptr, FuncType, LowLevelType, Ptr, PtrTarget,
        };
        fn make_llptr(arg: LowLevelType) -> _ptr {
            _ptr::new(
                Ptr {
                    TO: PtrTarget::Func(FuncType {
                        args: vec![arg],
                        result: LowLevelType::Void,
                    }),
                },
                Ok(None),
            )
        }
        let mut sb1 = SomeBuiltin::new("fn1", None, None);
        sb1.base.const_box = Some(Constant::new(ConstValue::LLPtr(Box::new(make_llptr(
            LowLevelType::Signed,
        )))));
        let mut sb2 = SomeBuiltin::new("fn2", None, None);
        sb2.base.const_box = Some(Constant::new(ConstValue::LLPtr(Box::new(make_llptr(
            LowLevelType::Float,
        )))));
        let key1 = rtyper_makekey(&SomeValue::Builtin(sb1));
        let key2 = rtyper_makekey(&SomeValue::Builtin(sb2));
        let ReprKey::Builtin(Some(BuiltinConstKey::ExtRegistry(ExtRegistryEntryKey::Ptr {
            instance_identity: Some(identity1),
        }))) = key1
        else {
            panic!("expected ptr extregistry key, got {key1:?}");
        };
        let ReprKey::Builtin(Some(BuiltinConstKey::ExtRegistry(ExtRegistryEntryKey::Ptr {
            instance_identity: Some(identity2),
        }))) = key2
        else {
            panic!("expected ptr extregistry key, got {key2:?}");
        };
        assert_ne!(identity1, identity2);
    }

    #[test]
    fn rtyper_makekey_somebuiltinmethod_keys_by_methodname_and_receiver_identity() {
        use crate::annotator::model::SomeBuiltinMethod;
        let sbm = SomeBuiltinMethod::new("append", SomeValue::Impossible, "append");
        let ReprKey::BuiltinMethod {
            methodname,
            s_self_id,
        } = rtyper_makekey(&SomeValue::BuiltinMethod(sbm))
        else {
            panic!("expected ReprKey::BuiltinMethod");
        };
        assert_eq!(methodname, "append");
        assert_ne!(s_self_id, 0);
    }

    #[test]
    fn rtyper_makekey_somebuiltinmethod_s_self_id_stable_across_clones() {
        // Upstream (rbuiltin.py) uses `id(self.s_self)`, i.e. the
        // identity of the shared Python receiver object. Pyre mirrors
        // that via an `Rc<SomeValue>`; clones of the outer
        // `SomeBuiltinMethod` must keep the same s_self pointer.
        use crate::annotator::model::SomeBuiltinMethod;
        let sbm = SomeBuiltinMethod::new("append", SomeValue::Impossible, "append");
        let cloned = sbm.clone();
        let key_a = rtyper_makekey(&SomeValue::BuiltinMethod(sbm));
        let key_b = rtyper_makekey(&SomeValue::BuiltinMethod(cloned));
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn rtyper_makerepr_someinstance_without_init_surfaces_self_rc_error() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeInstance;
        // No initialize_exceptiondata — self_weak stays empty.
        let rtyper = rtyper_for_tests();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let sv = SomeValue::Instance(SomeInstance::new(
            Some(classdef),
            false,
            std::collections::BTreeMap::new(),
        ));
        let err = rtyper_makerepr(&sv, &rtyper).expect_err("should surface self-weak error");
        assert!(err.to_string().contains("self-weak"));
    }

    #[test]
    fn rtyper_makerepr_stringbuilder_returns_builder_reprs() {
        use crate::annotator::model::{SomeStringBuilder, SomeUnicodeBuilder};
        let rtyper = rtyper_for_tests();

        let sb = SomeValue::StringBuilder(SomeStringBuilder::new());
        let r = rtyper_makerepr(&sb, &rtyper).expect("stringbuilder repr");
        assert_eq!(r.class_name(), "StringBuilderRepr");

        let ub = SomeValue::UnicodeBuilder(SomeUnicodeBuilder::new());
        let r = rtyper_makerepr(&ub, &rtyper).expect("unicodebuilder repr");
        assert_eq!(r.class_name(), "UnicodeBuilderRepr");
    }
}
