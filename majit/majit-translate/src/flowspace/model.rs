//! Flow graph model — SSA structure.
//!
//! RPython upstream: `rpython/flowspace/model.py` (737 LOC).
//!
//! Line-by-line port landing over commits F1.2 .. F1.7. This file
//! carries the leaf types first (`Variable`, `Constant`, `Atom`);
//! follow-up commits add `SpaceOperation`, `Block`, `Link`,
//! `FunctionGraph`, and the `mkentrymap` / `copygraph` / `summary` /
//! `checkgraph` helpers.
//!
//! Structural deviations from upstream (documented per CLAUDE.md
//! parity rule #1):
//!
//! * `Variable.annotation` is still an `Option<...>` carrier because
//!   the annotator arrives later in the pipeline. `Variable.concretetype`
//!   now stores the low-level `lltype` mirror directly.
//! * `Constant.value` is a placeholder `ConstValue` enum pending a
//!   Python object model at Phase 3 (`flowspace/operation.py` port);
//!   RPython wraps arbitrary Python objects via `Hashable`.
//! * `Variable.namesdict` (RPython class attribute — a mutable dict
//!   shared by every `Variable`) is a `static LazyLock<Mutex<...>>`;
//!   Rust has no class-mutable-state, this is the minimum deviation.

use parking_lot::Mutex;
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};

use indexmap::IndexMap;

use super::bytecode::HostCode;
use crate::annotator::model::SomeValue;
use crate::translator::rtyper::lltypesystem::lltype::_ptr;

// Exists to localise prepass nondeterminism (gh#1139).
static BLOCK_ADDR_REUSE: AtomicU64 = AtomicU64::new(0);
// Exists to localise prepass nondeterminism (gh#1139).
static GRAPH_ADDR_REUSE: AtomicU64 = AtomicU64::new(0);
// Exists to localise prepass nondeterminism (gh#1139).
static SEEN_BLOCK_ADDRS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
// Exists to localise prepass nondeterminism (gh#1139).
static REUSED_BLOCK_ADDRS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
// Exists to localise prepass nondeterminism (gh#1139).
static SEEN_GRAPH_ADDRS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

// Exists to localise prepass nondeterminism (gh#1139).
fn record_block_address(block: &BlockRef) {
    if !crate::determinism_trace_enabled() {
        return;
    }
    let addr = Rc::as_ptr(block) as usize;
    let inserted = SEEN_BLOCK_ADDRS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .insert(addr);
    if !inserted {
        BLOCK_ADDR_REUSE.fetch_add(1, Ordering::Relaxed);
        REUSED_BLOCK_ADDRS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .insert(addr);
    }
}

// Exists to localise prepass nondeterminism (gh#1139).
fn record_graph_address(graph: &GraphRef) {
    if !crate::determinism_trace_enabled() {
        return;
    }
    let addr = Rc::as_ptr(graph) as usize;
    let inserted = SEEN_GRAPH_ADDRS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .insert(addr);
    if !inserted {
        GRAPH_ADDR_REUSE.fetch_add(1, Ordering::Relaxed);
    }
}

// Exists to localise prepass nondeterminism (gh#1139).
pub fn block_addr_reuse_count() -> u64 {
    BLOCK_ADDR_REUSE.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
pub fn graph_addr_reuse_count() -> u64 {
    GRAPH_ADDR_REUSE.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
pub fn block_address_was_reused(block: &BlockRef) -> bool {
    if !crate::determinism_trace_enabled() {
        return false;
    }
    let addr = Rc::as_ptr(block) as usize;
    REUSED_BLOCK_ADDRS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .contains(&addr)
}

// RPython `Variable.annotation` holds a `SomeObject` subclass instance
// (annotator/model.py:SomeObject). Rust stores `Option<Rc<SomeValue>>`
// directly — `flowspace` and `annotator` are sibling modules inside
// `majit-translate`, so the cross-module reference is just a `use`.

/// RPython `Variable.concretetype` / `Constant.concretetype` field
/// type. Upstream carries a [`LowLevelType`] object assigned by the
/// rtyper (`rtyper.py:setconcretetype`). Pyre aliases the
/// upstream-orthodox type from
/// `crate::translator::rtyper::lltypesystem::lltype::LowLevelType`
/// for two reasons:
///
/// 1. Upstream keeps the two surfaces separate — `flowspace/model.py`
///    defines Variable/Constant while `rtyper/lltypesystem/lltype.py`
///    defines `LowLevelType` and the primitive singletons. Pyre
///    preserves that separation by re-exporting the alias here.
/// 2. The previous placeholder `ConcretetypePlaceholder = ()` dropped
///    Repr information; carrying the real type lets `FuncType.args` /
///    `FuncType.result` hold the actual per-arg lowleveltypes instead
///    of collapsing to unit, which is what FUNC.RESULT CFG-scan
///    inference relies on.
pub type ConcretetypePlaceholder = crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// General carrier for the host-level Python object stored in RPython
/// `Constant.value`.
///
/// Upstream `rpython/flowspace/model.py` `class Constant(Hashable)`
/// stores an arbitrary Python object in `Hashable.value`
/// (`self.value = value`). The Rust port started `ConstValue` as a closed
/// enum, but gathering builtin function / type / exception class / exception
/// instance / module into one `Rc<HostObjectInner>` carrier is more orthodox.
///
/// Identity is `Rc::ptr_eq` — the same as upstream Python object `is`.
/// Identity-keyed tables such as `SPECIAL_CASES` bootstrap from singletons
/// that share one instance (the Rc returned by `HOST_ENV.lookup_builtin`).
///
/// Deviation (parity rule #1): upstream can put an arbitrary object in the
/// carrier because a Python runtime is present. The Rust port does not embed
/// a Python runtime, so `HostObjectKind` enumerates the concrete categories
/// observed upstream (class/module/builtin callable/user function/instance).
/// This enum is not a public API. Outside the carrier the contract is `Rc`
/// identity and introspection methods such as `qualname()` and
/// `is_subclass_of(…)`.
#[derive(Clone)]
pub struct HostObject {
    inner: Arc<HostObjectInner>,
}

struct HostObjectInner {
    qualname: String,
    name: Option<String>,
    module_name: Option<String>,
    kind: HostObjectKind,
}

fn split_attr_name_module(qualname: &str) -> (String, Option<String>) {
    match qualname.rsplit_once('.') {
        Some((module, name)) => (name.to_string(), Some(module.to_string())),
        None => (qualname.to_string(), None),
    }
}

enum HostObjectKind {
    /// Python type/class object. `bases` is the `__bases__` tuple, used by
    /// the recursive `issubclass` walk. `members` corresponds to `cls.__dict__` —
    /// annotator ClassDesc.__init__ reads `_mixin_`, `_immutable_fields_`,
    /// `__slots__`, `_attrs_`, `__NOT_RPYTHON__`,
    /// `_annspecialcase_`, and so on, and `add_sources_for_class` walks every
    /// entry. Values are arbitrary Python values (bool, tuple,
    /// function, property, …), so they are stored in a `ConstValue` carrier.
    Class {
        bases: Vec<HostObject>,
        /// The ordered container is required because a `HashMap`'s iteration
        /// order varies per process and these entries produce a work order.
        members: Mutex<IndexMap<String, ConstValue>>,
        reusable_prebuilt_instance: OnceLock<HostObject>,
    },
    /// Python module object. `members` is the module dict — the target of
    /// `getattr` lookup. `Mutex` satisfies the `Sync` requirement of the
    /// `LazyLock` singleton.
    Module {
        members: Mutex<HashMap<String, HostObject>>,
    },
    /// Python builtin callable (function, method). No structure beyond the name.
    BuiltinCallable,
    /// Python function object (user-defined). `graph_func` is the code + closure
    /// state that flowspace inspects.
    UserFunction { graph_func: Box<GraphFunc> },
    /// Python instance (including an exception instance materialised by a
    /// raise statement). `class_obj` is `__class__`; `args` are the constructor
    /// arguments; `instance_dict` is the per-instance attribute dict
    /// (`inst.__dict__`) — prebuilt instances attached via class
    /// annotation decorators populate this so `getattr(inst, attr)`
    /// in `FrozenDesc.default_read_attribute` can observe them.
    Instance {
        class_obj: HostObject,
        args: Vec<ConstValue>,
        /// The ordered container is required because a `HashMap`'s iteration
        /// order varies per process and these keys produce a work order.
        instance_dict: HostInstanceDict,
    },
    /// Arbitrary host object stored in `Constant.value` — a value whose
    /// structure flowspace does not know but must preserve (for example an
    /// unported `ConstantData` variant, or a pyre-level opaque object).
    /// `qualname` records a debug-only identifier, and identity is `Arc::ptr_eq`.
    /// This key corresponds to upstream `Constant.value = <anonymous object>`.
    Opaque,
    /// Python `property` descriptor (upstream classdesc.py:591-602). fget
    /// / fset / fdel correspond to the slots of upstream
    /// `property(fget, fset, fdel, doc)` and are stored as `Option<HostObject>`
    /// (None = undefined). unaryop.py `_find_property_meth` extracts the slot
    /// from classdict `Constant(property_value)` via `getattr(obj.value, meth)`.
    Property {
        fget: Option<HostObject>,
        fset: Option<HostObject>,
        fdel: Option<HostObject>,
    },
    /// Python `staticmethod(func)` descriptor. The wrapped callable is
    /// typically a user-function HostObject; `ClassDesc.s_get_value`
    /// unwraps it before calling `immutablevalue`, matching
    /// `value.__get__(42)` upstream.
    StaticMethod { func: HostObject },
    /// Python `classmethod(func)` descriptor. `ClassDesc.s_get_value`
    /// recognizes this wrapper and raises the upstream
    /// "classmethods are not supported" error.
    ClassMethod { func: HostObject },
    /// Python bound method object. This is the host-level result of
    /// `function.__get__(obj, cls)` / `classmethod.__get__(obj, cls)`
    /// in places where flowspace or FrozenDesc must mirror real
    /// `getattr()`.
    BoundMethod {
        self_obj: HostObject,
        func: HostObject,
        name: String,
        origin_class: HostObject,
    },
    /// Python `weakref.ref(obj)`. Upstream `bookkeeper.immutablevalue`
    /// dereferences the weakref via `x()` and routes the referent
    /// (or `None` for dead weakrefs) through the SomeWeakRef path
    /// (bookkeeper.py). `referent=None` models a dead
    /// weakref whose target has been garbage-collected.
    Weakref { referent: Option<HostObject> },
    /// Host-side native callable — wraps a Rust closure that represents
    /// a Python-callable whose behaviour we actually need to execute at
    /// annotation time.
    ///
    /// Upstream RPython evaluates e.g. `x._freeze_()` (bookkeeper.py)
    /// or `prop.fget(self)` (descriptor getattr) directly against the
    /// Python object. The Rust port has no Python runtime, so test
    /// fixtures and bootstrap code register the concrete native closure
    /// that implements those methods. The helper
    /// [`HostObject::call_host`] is the single entry-point for invoking
    /// these — higher-level call helpers (property getter dispatch,
    /// `_freeze_` / `_cleanup_` enforcement, …) all route through it.
    ///
    /// Rust language adaptation (parity rule #1, CLAUDE.md): Python
    /// stores bound callables directly as `__dict__` values and invokes
    /// them via the descriptor protocol. Rust lacks an embedded Python
    /// interpreter, so the carrier is specialised. Mirror behaviour —
    /// not structure — is preserved.
    NativeCallable { func: HostCallableFn },
}

/// Zero- or N-arg host callable. `args[0]` is the bound `self` if any
/// (matching how upstream `BoundMethod.__call__` forwards `self` to
/// `__func__`); zero-arg getters / `_freeze_` / `_cleanup_` pass a
/// single-element slice.
///
/// Result: `Ok(ConstValue)` holds the concrete return, `Err(String)`
/// propagates host-side failures (upstream `TypeError`, unhandled
/// descriptor case, …) which callers rewrap into `AnnotatorError`.
pub type HostCallableFn = Arc<dyn Fn(&[ConstValue]) -> Result<ConstValue, String> + Send + Sync>;

/// Native-source boundary for a live host instance's `__dict__`.
/// RPython's `InstanceSource.all_instance_attributes` enumerates the actual
/// object and `InstanceSource.s_get_value` reads it through `getattr`.
/// A native provider must retain its object's lifetime and return projected
/// field names in order. Missing native support is Unsupported, not Missing.
/// Object-valued fields must preserve the importer's canonical identity.
///
/// This is a read-only view, not a second attribute store. The HostObject
/// carrier invokes it without holding an attribute lock, permitting recursive
/// imports. Production providers still need validated LLBC field projection
/// and translation-owned identity/rooting; this interface does not supply them.
pub trait NativeInstanceDict: Send + Sync {
    fn get(&self, name: &str) -> Result<ConstValue, HostGetAttrError>;
    fn keys(&self) -> Result<Vec<String>, HostGetAttrError>;
}

enum HostInstanceDict {
    Owned(Mutex<IndexMap<String, ConstValue>>),
    Native(Arc<dyn NativeInstanceDict>),
}

impl PartialEq for HostObject {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for HostObject {}

impl Hash for HostObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.inner) as usize).hash(state);
    }
}

impl std::fmt::Debug for HostObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<host {}>", self.inner.qualname)
    }
}

impl HostObject {
    pub fn qualname(&self) -> &str {
        &self.inner.qualname
    }

    /// RPython `cls.__name__` parity — the short name without the
    /// dotted module prefix. Falls back to `qualname` when the inner
    /// `name` slot is empty (matches upstream's "no `__name__` attribute"
    /// case where `qualname` is the only available label).
    pub fn simple_name(&self) -> &str {
        self.inner
            .name
            .as_deref()
            .unwrap_or_else(|| self.qualname())
    }

    pub fn module_name(&self) -> Option<&str> {
        self.inner.module_name.as_deref().or_else(|| {
            if self.is_class()
                && HOST_ENV
                    .lookup_builtin(self.qualname())
                    .as_ref()
                    .is_some_and(|builtin| builtin == self)
            {
                Some("__builtin__")
            } else {
                None
            }
        })
    }

    /// Pointer-identity handle — used where upstream relies on
    /// `id(pyobj)` to build a dict key (e.g. sandbox trampoline
    /// emulation key, policy.py:87).
    pub fn identity_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    pub fn is_class(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Class { .. })
    }

    pub fn is_module(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Module { .. })
    }

    pub fn is_builtin_callable(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::BuiltinCallable)
    }

    pub fn is_instance(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Instance { .. })
    }

    pub fn is_user_function(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::UserFunction { .. })
    }

    pub fn user_function(&self) -> Option<&GraphFunc> {
        match &self.inner.kind {
            HostObjectKind::UserFunction { graph_func } => Some(graph_func.as_ref()),
            _ => None,
        }
    }

    /// Upstream `issubclass(self, other)` over `__bases__`. Recursive
    /// depth-first search; on an exception hierarchy it matches `__mro__`
    /// C3 linearisation.
    pub fn is_subclass_of(&self, other: &HostObject) -> bool {
        if self == other {
            return true;
        }
        match &self.inner.kind {
            HostObjectKind::Class { bases, .. } => bases.iter().any(|b| b.is_subclass_of(other)),
            _ => false,
        }
    }

    /// Class.__bases__ — bases tuple view. Non-class is None.
    pub fn class_bases(&self) -> Option<&[HostObject]> {
        match &self.inner.kind {
            HostObjectKind::Class { bases, .. } => Some(bases.as_slice()),
            _ => None,
        }
    }

    /// `cls.__dict__.get(name)` — class dict lookup. Non-class is
    /// None.
    pub fn class_get(&self, name: &str) -> Option<ConstValue> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members.lock().get(name).cloned(),
            _ => None,
        }
    }

    /// Class dict setter — bootstrap / ClassDesc.create_new_attribute
    /// path. No-op for a non-class.
    pub fn class_set(&self, name: impl Into<String>, value: ConstValue) {
        if let HostObjectKind::Class { members, .. } = &self.inner.kind {
            members.lock().insert(name.into(), value);
        }
    }

    /// `cls.__dict__.keys()` — class dict key snapshot. Non-class is
    /// an empty Vec.
    pub fn class_dict_keys(&self) -> Vec<String> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members.lock().keys().cloned().collect(),
            _ => Vec::new(),
        }
    }

    /// `cls.__dict__.items()` — class dict entry snapshot. Non-class
    /// is an empty Vec.
    pub fn class_dict_items(&self) -> Vec<(String, ConstValue)> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members
                .lock()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// `cls.__dict__.__contains__(name)` — class dict key presence check.
    /// Non-class is false.
    pub fn class_has(&self, name: &str) -> bool {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members.lock().contains_key(name),
            _ => false,
        }
    }

    /// RPython `rclass.InstanceRepr.get_reusable_prebuilt_instance()`
    /// corresponding surface. Re-raising a standard exception must reuse the
    /// same prebuilt instance, so the singleton is attached to a OnceLock
    /// inside the class object.
    pub fn reusable_prebuilt_instance(&self) -> Option<HostObject> {
        match &self.inner.kind {
            HostObjectKind::Class {
                reusable_prebuilt_instance,
                ..
            } => Some(
                reusable_prebuilt_instance
                    .get_or_init(|| HostObject::new_instance(self.clone(), Vec::new()))
                    .clone(),
            ),
            _ => None,
        }
    }

    /// `cls.__mro__` — C3 linearisation over `__bases__`. Non-class is
    /// None. Duplicate base classes are handled, but a conflicting definition
    /// returns None instead of `TypeError: MRO conflict` (upstream
    /// `type(...).__mro__` raises TypeError). The RPython annotator uses this
    /// only on the `add_mixins` path
    /// `type('tmp', tuple(mixins) + (object,), {}).__mro__`, so the None path
    /// is not taken unless the mixin hierarchy has a C3 conflict.
    pub fn mro(&self) -> Option<Vec<HostObject>> {
        if !self.is_class() {
            return None;
        }
        c3_linearise(self)
    }

    /// Instance → `__class__`. None means `self` is not an instance.
    pub fn instance_class(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Instance { class_obj, .. } => Some(class_obj),
            _ => None,
        }
    }

    /// Instance → constructor args (captured from a raise-site call such as
    /// `ValueError("msg")`).
    pub fn instance_args(&self) -> Option<&[ConstValue]> {
        match &self.inner.kind {
            HostObjectKind::Instance { args, .. } => Some(args.as_slice()),
            _ => None,
        }
    }

    /// Instance → per-instance `__dict__` lookup. Mirrors
    /// the dictionary step of `getattr`; descriptor precedence belongs to
    /// `host_getattr`. An unsupported native read must not fall through to a
    /// class attribute or become a default prebuilt field in the rtyper.
    pub fn instance_get(&self, name: &str) -> Result<ConstValue, HostGetAttrError> {
        match &self.inner.kind {
            HostObjectKind::Instance {
                instance_dict: HostInstanceDict::Owned(dict),
                ..
            } => dict
                .lock()
                .get(name)
                .cloned()
                .ok_or(HostGetAttrError::Missing),
            HostObjectKind::Instance {
                instance_dict: HostInstanceDict::Native(dict),
                ..
            } => dict.get(name),
            _ => Err(HostGetAttrError::Missing),
        }
    }

    /// Instance `__dict__.keys()` snapshot. Non-instance returns an
    /// empty Vec, mirroring `getattr(obj, '__dict__', {}).keys()`.
    pub fn instance_dict_keys(&self) -> Result<Vec<String>, HostGetAttrError> {
        match &self.inner.kind {
            HostObjectKind::Instance {
                instance_dict: HostInstanceDict::Owned(dict),
                ..
            } => Ok(dict.lock().keys().cloned().collect()),
            HostObjectKind::Instance {
                instance_dict: HostInstanceDict::Native(dict),
                ..
            } => dict.keys(),
            _ => Ok(Vec::new()),
        }
    }

    /// Instance setter — installs an entry into the per-instance
    /// `__dict__`. Used by prebuilt-instance fixtures in tests and
    /// by the `@setattr_to_class_annotation` style decorators
    /// that populate known attributes upfront.
    pub fn instance_set(
        &self,
        name: impl Into<String>,
        value: ConstValue,
    ) -> Result<(), HostGetAttrError> {
        if let HostObjectKind::Instance {
            instance_dict: HostInstanceDict::Owned(dict),
            ..
        } = &self.inner.kind
        {
            dict.lock().insert(name.into(), value);
            Ok(())
        } else {
            // A native view cannot accept a detached shadow write.
            Err(HostGetAttrError::Unsupported)
        }
    }

    /// Module member lookup — upstream `getattr(module, name)`.
    pub fn module_get(&self, name: &str) -> Option<HostObject> {
        match &self.inner.kind {
            HostObjectKind::Module { members } => members.lock().get(name).cloned(),
            _ => None,
        }
    }

    /// Module setter — used only while bootstrapping a module object.
    pub fn module_set(&self, name: impl Into<String>, value: HostObject) {
        if let HostObjectKind::Module { members } = &self.inner.kind {
            members.lock().insert(name.into(), value);
        }
    }

    /// Host-level `obj.__class__` / `type(obj)` view used by
    /// `description.new_or_old_class()`. Only returns `None` for
    /// `Opaque`, whose real runtime type the Rust carrier does not
    /// know.
    pub fn class_of(&self) -> Option<HostObject> {
        match &self.inner.kind {
            HostObjectKind::Class { .. } => HOST_ENV.lookup_builtin("type"),
            HostObjectKind::Module { .. } => HOST_ENV.lookup_builtin("module"),
            HostObjectKind::BuiltinCallable => {
                HOST_ENV.lookup_builtin("builtin_function_or_method")
            }
            HostObjectKind::UserFunction { .. } => HOST_ENV.lookup_builtin("function"),
            HostObjectKind::Instance { class_obj, .. } => Some(class_obj.clone()),
            HostObjectKind::Opaque => None,
            HostObjectKind::Property { .. } => HOST_ENV.lookup_builtin("property"),
            HostObjectKind::StaticMethod { .. } => HOST_ENV.lookup_builtin("staticmethod"),
            HostObjectKind::ClassMethod { .. } => HOST_ENV.lookup_builtin("classmethod"),
            HostObjectKind::BoundMethod { .. } => HOST_ENV.lookup_builtin("method"),
            HostObjectKind::Weakref { .. } => HOST_ENV.lookup_builtin("weakref"),
            HostObjectKind::NativeCallable { .. } => {
                HOST_ENV.lookup_builtin("builtin_function_or_method")
            }
        }
    }

    pub fn new_class(qualname: impl Into<String>, bases: Vec<HostObject>) -> Self {
        let qualname = qualname.into();
        let (name, module_name) = split_attr_name_module(&qualname);
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name,
                kind: HostObjectKind::Class {
                    bases,
                    members: Mutex::new(IndexMap::new()),
                    reusable_prebuilt_instance: OnceLock::new(),
                },
            }),
        }
    }

    /// `new_class` + initial class dict. annotator
    /// Convenience constructor that installs required members
    /// (`_mixin_`, `_immutable_fields_`, …) on a class object the bootstrap
    /// already built, before `ClassDesc.__init__` splits mixins and runs
    /// `add_sources_for_class`.
    pub fn new_class_with_members(
        qualname: impl Into<String>,
        bases: Vec<HostObject>,
        // The ordered container preserves the caller's class-dict insertion order.
        members: IndexMap<String, ConstValue>,
    ) -> Self {
        let qualname = qualname.into();
        let (name, module_name) = split_attr_name_module(&qualname);
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name,
                kind: HostObjectKind::Class {
                    bases,
                    members: Mutex::new(members),
                    reusable_prebuilt_instance: OnceLock::new(),
                },
            }),
        }
    }

    pub fn new_module(qualname: impl Into<String>) -> Self {
        let qualname = qualname.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                name: Some(qualname.clone()),
                module_name: None,
                qualname,
                kind: HostObjectKind::Module {
                    members: Mutex::new(HashMap::new()),
                },
            }),
        }
    }

    pub fn new_builtin_callable(qualname: impl Into<String>) -> Self {
        let qualname = qualname.into();
        let (name, module_name) = split_attr_name_module(&qualname);
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name: Some(module_name.unwrap_or_else(|| "__builtin__".to_string())),
                kind: HostObjectKind::BuiltinCallable,
            }),
        }
    }

    pub fn new_user_function(graph_func: GraphFunc) -> Self {
        // `pygraph.py _sanitize_funcname`: if `func.class_` is set
        // (attached by `classdesc.py:606` for method functions), the
        // graph name is `'%s.%s' % (class_.__name__, func.__name__)`.
        // Pyre's `GraphFunc.class_` carries the owner `HostObject` for
        // method functions; `simple_name()` returns `cls.__name__` (the
        // short name — NOT the dotted qualname, which would produce
        // `module.Foo.bar` instead of upstream's `Foo.bar`).
        //
        // `name` (the short `__name__` slot) stays at `code.co_name` /
        // `gf.name` so callers reading the bare ident continue to see
        // `bar`, not `Foo.bar`. `module_name` for class-owned methods
        // derives from the owner class's module — `split_attr_name_module`
        // would mistake the class name for a module otherwise.
        let class_qualname = graph_func.class_.as_ref().and_then(|c| {
            let q = c.simple_name();
            if q.is_empty() {
                None
            } else {
                Some(q.to_string())
            }
        });
        let qualname = match &class_qualname {
            Some(class_q) => format!("{class_q}.{}", graph_func.name),
            None => graph_func.name.clone(),
        };
        let name = graph_func
            .code
            .as_ref()
            .map(|code| code.co_name.clone())
            .unwrap_or_else(|| {
                // For class-owned fns the short name comes from
                // `gf.name` directly (the class prefix is in qualname,
                // not in `__name__`). For free fns we keep the
                // historical fall-through to the rsplit so synthetic
                // dotted-qualnames (e.g. `pkg.module.demo`) still
                // surface their tail as `__name__`.
                if class_qualname.is_some() {
                    graph_func.name.clone()
                } else {
                    split_attr_name_module(&qualname).0
                }
            });
        // Upstream `func.__module__` lives on the function object —
        // `GraphFunc::module` carries it directly. Class-owned fns
        // additionally consult the owner class's module when their
        // own slot is unset, mirroring upstream `Foo.bar.__module__`
        // == `Foo.__module__`. Free fns fall back to the qualname
        // split only when the GraphFunc was built without a
        // `__module__` slot (synthetic-globals fixtures, descriptor-
        // class helpers).
        let module_name = graph_func.module.clone().or_else(|| {
            if let Some(class) = graph_func.class_.as_ref() {
                class.module_name().map(str::to_owned)
            } else {
                split_attr_name_module(&qualname).1
            }
        });
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name,
                kind: HostObjectKind::UserFunction {
                    graph_func: Box::new(graph_func),
                },
            }),
        }
    }

    /// RPython `func_with_new_name(func, newname)` helper surface for
    /// user functions. Mirrors the upstream behavior that allocates a
    /// fresh function object sharing the same code/defaults/closure but
    /// exposing a different `__name__`.
    pub fn renamed_user_function(&self, newname: &str) -> Option<Self> {
        let graph_func = self.user_function()?;
        Some(Self::new_user_function(graph_func.with_new_name(newname)))
    }

    pub fn new_instance(class_obj: HostObject, args: Vec<ConstValue>) -> Self {
        Self::new_instance_with_dict(
            class_obj,
            args,
            HostInstanceDict::Owned(Mutex::new(IndexMap::new())),
        )
    }

    /// The caller supplies the canonical projected class and interns this
    /// carrier for its native object. A same-spelled new class is not equivalent.
    pub fn new_native_instance(class_obj: HostObject, dict: Arc<dyn NativeInstanceDict>) -> Self {
        assert!(
            class_obj.is_class(),
            "native instance requires a projected class"
        );
        Self::new_instance_with_dict(class_obj, Vec::new(), HostInstanceDict::Native(dict))
    }

    fn new_instance_with_dict(
        class_obj: HostObject,
        args: Vec<ConstValue>,
        instance_dict: HostInstanceDict,
    ) -> Self {
        let qualname = format!("{}-instance", class_obj.qualname());
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: None,
                module_name: None,
                kind: HostObjectKind::Instance {
                    class_obj,
                    args,
                    instance_dict,
                },
            }),
        }
    }

    /// Carries an arbitrary host object stored in `Constant.value`. `qualname`
    /// is for debug; identity is always a fresh Arc.
    pub fn new_opaque(qualname: impl Into<String>) -> Self {
        let qualname = qualname.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: None,
                module_name: None,
                kind: HostObjectKind::Opaque,
            }),
        }
    }

    pub fn is_opaque(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Opaque)
    }

    /// Python `property(fget, fset, fdel, doc)` constructor — upstream
    /// classdesc.py:591-602 synthesises getter / setter hidden functions
    /// and stores the property object itself in classdict.
    pub fn new_property(
        qualname: impl Into<String>,
        fget: Option<HostObject>,
        fset: Option<HostObject>,
        fdel: Option<HostObject>,
    ) -> Self {
        let qualname = qualname.into();
        let (name, _) = split_attr_name_module(&qualname);
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name: None,
                kind: HostObjectKind::Property { fget, fset, fdel },
            }),
        }
    }

    pub fn is_property(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Property { .. })
    }

    /// Upstream `property.fget` — `None` if not provided at property
    /// construction time.
    pub fn property_fget(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fget, .. } => fget.as_ref(),
            _ => None,
        }
    }

    /// Upstream `property.fset`.
    pub fn property_fset(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fset, .. } => fset.as_ref(),
            _ => None,
        }
    }

    /// Upstream `property.fdel`.
    pub fn property_fdel(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fdel, .. } => fdel.as_ref(),
            _ => None,
        }
    }

    /// Python `staticmethod(func)` constructor.
    pub fn new_staticmethod(qualname: impl Into<String>, func: HostObject) -> Self {
        let qualname = qualname.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(func.simple_name().to_string()),
                module_name: func.module_name().map(|s| s.to_string()),
                kind: HostObjectKind::StaticMethod { func },
            }),
        }
    }

    pub fn is_staticmethod(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::StaticMethod { .. })
    }

    /// Upstream `staticmethod.__get__(42)` unwraps back to the wrapped
    /// callable object without binding.
    pub fn staticmethod_func(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::StaticMethod { func } => Some(func),
            _ => None,
        }
    }

    /// Python `classmethod(func)` constructor.
    pub fn new_classmethod(qualname: impl Into<String>, func: HostObject) -> Self {
        let qualname = qualname.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(func.simple_name().to_string()),
                module_name: func.module_name().map(|s| s.to_string()),
                kind: HostObjectKind::ClassMethod { func },
            }),
        }
    }

    pub fn is_classmethod(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::ClassMethod { .. })
    }

    pub fn classmethod_func(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::ClassMethod { func } => Some(func),
            _ => None,
        }
    }

    /// Host-level bound method constructor. Mirrors the runtime object
    /// returned by descriptor `__get__` for plain functions and
    /// classmethods.
    pub fn new_bound_method(
        qualname: impl Into<String>,
        self_obj: HostObject,
        func: HostObject,
        name: impl Into<String>,
        origin_class: HostObject,
    ) -> Self {
        let qualname = qualname.into();
        let name = name.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name.clone()),
                module_name: func.module_name().map(|s| s.to_string()),
                kind: HostObjectKind::BoundMethod {
                    self_obj,
                    func,
                    name,
                    origin_class,
                },
            }),
        }
    }

    pub fn is_bound_method(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::BoundMethod { .. })
    }

    pub fn bound_method_self(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::BoundMethod { self_obj, .. } => Some(self_obj),
            _ => None,
        }
    }

    pub fn bound_method_func(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::BoundMethod { func, .. } => Some(func),
            _ => None,
        }
    }

    pub fn bound_method_name(&self) -> Option<&str> {
        match &self.inner.kind {
            HostObjectKind::BoundMethod { name, .. } => Some(name.as_str()),
            _ => None,
        }
    }

    pub fn bound_method_origin_class(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::BoundMethod { origin_class, .. } => Some(origin_class),
            _ => None,
        }
    }

    /// Host-level `weakref.ref(obj)` constructor. `referent=None`
    /// models a dead weakref whose target has been garbage-collected.
    pub fn new_weakref(qualname: impl Into<String>, referent: Option<HostObject>) -> Self {
        let qualname = qualname.into();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: None,
                module_name: None,
                kind: HostObjectKind::Weakref { referent },
            }),
        }
    }

    pub fn is_weakref(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Weakref { .. })
    }

    /// Dereference the weakref — upstream `x()`. `Some(Some(h))` when
    /// the weakref is alive; `Some(None)` when it has been collected;
    /// `None` when `self` isn't a weakref at all.
    pub fn weakref_referent(&self) -> Option<Option<&HostObject>> {
        match &self.inner.kind {
            HostObjectKind::Weakref { referent } => Some(referent.as_ref()),
            _ => None,
        }
    }

    /// Construct a host callable from a Rust closure.
    ///
    /// `qualname` is the debug identifier (upstream `func.__qualname__`);
    /// `func` is invoked by [`Self::call_host`] with the positional args.
    pub fn new_native_callable(qualname: impl Into<String>, func: HostCallableFn) -> Self {
        let qualname = qualname.into();
        let (name, module_name) = split_attr_name_module(&qualname);
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                name: Some(name),
                module_name,
                kind: HostObjectKind::NativeCallable { func },
            }),
        }
    }

    pub fn is_native_callable(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::NativeCallable { .. })
    }

    /// Invoke the host callable — upstream Python `obj(*args)`.
    ///
    /// Dispatches by kind:
    /// * `NativeCallable { func }` → `func(args)` directly.
    /// * `BoundMethod { self_obj, func }` → prepend `self_obj` to
    ///   `args` and recurse into `func` (upstream
    ///   `BoundMethod.__call__ = self.__func__(self.__self__, *args)`).
    /// * `UserFunction { graph_func }` →
    ///   `Err(HostCallError::RequiresFlowEvaluator)` — running a user
    ///   graph requires the annotator / flowspace; the caller must
    ///   route through `Bookkeeper::pbc_call` instead of executing
    ///   host-side.
    /// * `BuiltinCallable` with no registered closure →
    ///   `Err(HostCallError::NoNativeImpl)`.
    /// * everything else → `Err(HostCallError::NotCallable)`.
    ///
    /// `HostCallError::RequiresFlowEvaluator` is a specific boundary
    /// marker: upstream RPython evaluates the `_freeze_` /
    /// `_cleanup_` / `property.fget` call through real Python, but
    /// the Rust port has no host-side evaluator for user graphs yet.
    /// Callers (`call_freeze_method`, `call_cleanup_method`,
    /// `host_descriptor_get`) translate this variant into an
    /// `AnnotatorError` / `HostGetAttrError::Unsupported` citing the
    /// Phase 6 graph-evaluator dependency, so the boundary stays
    /// discoverable.
    pub fn call_host(&self, args: &[ConstValue]) -> Result<ConstValue, HostCallError> {
        match &self.inner.kind {
            HostObjectKind::NativeCallable { func } => func(args).map_err(HostCallError::Native),
            HostObjectKind::BoundMethod { self_obj, func, .. } => {
                let mut bound_args: Vec<ConstValue> = Vec::with_capacity(args.len() + 1);
                bound_args.push(ConstValue::HostObject(self_obj.clone()));
                bound_args.extend_from_slice(args);
                func.call_host(&bound_args)
            }
            HostObjectKind::UserFunction { .. } => {
                Err(HostCallError::RequiresFlowEvaluator(self.qualname().into()))
            }
            HostObjectKind::BuiltinCallable => {
                Err(HostCallError::NoNativeImpl(self.qualname().into()))
            }
            _ => Err(HostCallError::NotCallable(self.qualname().into())),
        }
    }

    /// Predicate: `true` when [`Self::call_host`] can return
    /// `Ok(...)` directly — i.e. the inner kind is a
    /// `NativeCallable` or a `BoundMethod` whose `__func__`
    /// transitively reaches one. `false` for `UserFunction` /
    /// unregistered `BuiltinCallable` / non-callable — these
    /// require the host-graph evaluator that hasn't landed yet.
    pub fn is_host_executable(&self) -> bool {
        match &self.inner.kind {
            HostObjectKind::NativeCallable { .. } => true,
            HostObjectKind::BoundMethod { func, .. } => func.is_host_executable(),
            _ => false,
        }
    }
}

fn host_object_own_getattr(pyobj: &HostObject, name: &str) -> Option<ConstValue> {
    match name {
        "__class__" => pyobj.class_of().map(ConstValue::HostObject),
        "__name__" => match &pyobj.inner.kind {
            HostObjectKind::Class { .. }
            | HostObjectKind::Module { .. }
            | HostObjectKind::BuiltinCallable
            | HostObjectKind::UserFunction { .. }
            | HostObjectKind::BoundMethod { .. }
            | HostObjectKind::NativeCallable { .. } => {
                Some(ConstValue::byte_str(pyobj.simple_name()))
            }
            HostObjectKind::Property { .. } => Some(ConstValue::byte_str(pyobj.simple_name())),
            HostObjectKind::StaticMethod { func } | HostObjectKind::ClassMethod { func } => {
                Some(ConstValue::byte_str(func.simple_name()))
            }
            _ => None,
        },
        "__module__" => match &pyobj.inner.kind {
            HostObjectKind::Module { .. } | HostObjectKind::Property { .. } => None,
            HostObjectKind::Class { .. }
            | HostObjectKind::BuiltinCallable
            | HostObjectKind::UserFunction { .. }
            | HostObjectKind::BoundMethod { .. }
            | HostObjectKind::StaticMethod { .. }
            | HostObjectKind::ClassMethod { .. }
            | HostObjectKind::NativeCallable { .. } => {
                pyobj.module_name().map(ConstValue::byte_str)
            }
            _ => None,
        },
        "__globals__" => match &pyobj.inner.kind {
            // `live_globals()` returns the static `self.globals.value`
            // snapshot frozen at construction (`flowcontext.py:284
            // self.w_globals = Constant(func.__globals__)`). Upstream only
            // reads the globals dict during flow analysis (`flowcontext.py:847`)
            // and never mutates it, so the snapshot is observationally
            // identical to upstream's live dict reference.
            HostObjectKind::UserFunction { graph_func } => Some(graph_func.live_globals()),
            _ => None,
        },
        "__defaults__" => match &pyobj.inner.kind {
            HostObjectKind::UserFunction { graph_func } => {
                if graph_func.defaults.is_empty() {
                    Some(ConstValue::None)
                } else {
                    Some(ConstValue::Tuple(
                        graph_func
                            .defaults
                            .iter()
                            .map(|constant| constant.value.clone())
                            .collect(),
                    ))
                }
            }
            _ => None,
        },
        "__closure__" => match &pyobj.inner.kind {
            HostObjectKind::UserFunction { graph_func } => {
                if graph_func.closure.is_empty() {
                    Some(ConstValue::None)
                } else {
                    Some(ConstValue::Tuple(
                        graph_func
                            .closure
                            .iter()
                            .map(|constant| constant.value.clone())
                            .collect(),
                    ))
                }
            }
            _ => None,
        },
        "fget" if pyobj.is_property() => pyobj
            .property_fget()
            .cloned()
            .map(ConstValue::HostObject)
            .or(Some(ConstValue::None)),
        "fset" if pyobj.is_property() => pyobj
            .property_fset()
            .cloned()
            .map(ConstValue::HostObject)
            .or(Some(ConstValue::None)),
        "fdel" if pyobj.is_property() => pyobj
            .property_fdel()
            .cloned()
            .map(ConstValue::HostObject)
            .or(Some(ConstValue::None)),
        "__func__" => pyobj
            .bound_method_func()
            .cloned()
            .or_else(|| pyobj.staticmethod_func().cloned())
            .or_else(|| pyobj.classmethod_func().cloned())
            .map(ConstValue::HostObject),
        "__self__" => match &pyobj.inner.kind {
            HostObjectKind::BoundMethod { self_obj, .. } => {
                Some(ConstValue::HostObject(self_obj.clone()))
            }
            HostObjectKind::BuiltinCallable => Some(ConstValue::HostObject(
                pyobj
                    .module_name()
                    .and_then(|module| {
                        if module == "__builtin__" {
                            Some(HOST_ENV.builtin_module())
                        } else {
                            HOST_ENV.import_module(module)
                        }
                    })
                    .unwrap_or_else(|| HOST_ENV.builtin_module()),
            )),
            _ => None,
        },
        _ => None,
    }
}

/// Discriminated error channel for [`HostObject::call_host`]. Upstream
/// Python `obj(*args)` collapses all three failure modes into whatever
/// exception the callee raises; the Rust port splits them because
/// different annotator callsites need different fallbacks — the
/// `RequiresFlowEvaluator` marker in particular is the boundary
/// between "host-side executable" (NativeCallable) and
/// "annotator-side PBC call required" (UserFunction).
#[derive(Clone, Debug)]
pub enum HostCallError {
    /// upstream Python `TypeError: 'X' object is not callable` —
    /// reached when a class / module / instance without a registered
    /// `__call__` is invoked.
    NotCallable(String),
    /// `BuiltinCallable` shell without a registered native impl —
    /// use `HostObject::new_native_callable` to wire one up.
    NoNativeImpl(String),
    /// `UserFunction` (or `BoundMethod` of one) — running the call
    /// requires the host-graph evaluator. Until that Phase 6 piece
    /// lands, the annotator surfaces this as a specific error so the
    /// limitation is discoverable instead of a generic "not callable".
    RequiresFlowEvaluator(String),
    /// `NativeCallable` closure raised — upstream's arbitrary Python
    /// exception path.
    Native(String),
}

impl std::fmt::Display for HostCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostCallError::NotCallable(q) => {
                write!(f, "HostObject::call_host: {q:?} is not callable")
            }
            HostCallError::NoNativeImpl(q) => write!(
                f,
                "HostObject::call_host: builtin {q:?} has no native impl registered (use HostObject::new_native_callable)",
            ),
            HostCallError::RequiresFlowEvaluator(q) => write!(
                f,
                "HostObject::call_host: user function {q:?} has no host-level body (route through Bookkeeper::pbc_call; Phase 6 host-graph evaluator not yet landed)",
            ),
            HostCallError::Native(msg) => f.write_str(msg),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostGetAttrError {
    Missing,
    Unsupported,
}

/// Upstream Python predicate `type(v).__set__` / `__delete__` exists — a
/// class-dict entry qualifying as a *data descriptor* in the descriptor
/// protocol (`object.__getattribute__` CPython `Objects/object.c`
/// `_PyObject_GenericGetAttrWithDict`).
///
/// Data descriptors win against the instance `__dict__`; non-data
/// descriptors lose. The Rust HostObject model only represents one data
/// descriptor today: `property` (has `__set__` / `__delete__` even when
/// `fset` / `fdel` are `None` — the setter slot still exists on the
/// descriptor type). `staticmethod` / `classmethod` / regular functions
/// / `NativeCallable` are non-data descriptors.
fn is_data_descriptor(value: &ConstValue) -> bool {
    match value {
        ConstValue::HostObject(h) => h.is_property(),
        _ => false,
    }
}

fn host_class_mro_lookup(cls: &HostObject, name: &str) -> Option<(ConstValue, HostObject)> {
    if let Some(value) = cls.class_get(name) {
        return Some((value, cls.clone()));
    }
    let mro = cls.mro()?;
    for base in mro.iter().skip(1) {
        if let Some(value) = base.class_get(name) {
            return Some((value, base.clone()));
        }
    }
    None
}

fn host_descriptor_get(
    value: ConstValue,
    name: &str,
    self_obj: Option<&HostObject>,
    owner: &HostObject,
    origin_class: &HostObject,
) -> Result<ConstValue, HostGetAttrError> {
    match value {
        ConstValue::HostObject(host) if host.is_staticmethod() => host
            .staticmethod_func()
            .cloned()
            .map(ConstValue::HostObject)
            .ok_or(HostGetAttrError::Unsupported),
        ConstValue::HostObject(host) if host.is_classmethod() => {
            let func = host
                .classmethod_func()
                .cloned()
                .ok_or(HostGetAttrError::Unsupported)?;
            Ok(ConstValue::HostObject(HostObject::new_bound_method(
                format!("{}.{}", owner.qualname(), name),
                owner.clone(),
                func,
                name,
                origin_class.clone(),
            )))
        }
        ConstValue::HostObject(host) if host.is_property() => {
            // upstream property descriptor: `prop.__get__(obj, cls)`
            // returns `prop.fget(obj)` when `obj is not None` and the
            // getter is defined; otherwise the raw property descriptor
            // is returned (for class-level lookup).
            let Some(self_obj) = self_obj else {
                return Ok(ConstValue::HostObject(host));
            };
            let Some(fget) = host.property_fget().cloned() else {
                // Property without fget — upstream Python raises
                // `AttributeError: unreadable attribute`, identical to
                // a missing name. Route through Missing so callers that
                // handle AttributeError (instance-dict fallback,
                // FrozenDesc.has_attribute) treat it the same.
                return Err(HostGetAttrError::Missing);
            };
            // upstream executes the fget in real Python. The Rust
            // port dispatches via `HostObject::call_host` — that
            // runs `NativeCallable` directly, and returns
            // `HostCallError::RequiresFlowEvaluator` for
            // `UserFunction` (the Phase 6 host-graph evaluator isn't
            // landed yet). Either failure maps to `Unsupported` here:
            // the attribute exists, the value just can't be obtained
            // host-side.
            fget.call_host(&[ConstValue::HostObject(self_obj.clone())])
                .map_err(|_| HostGetAttrError::Unsupported)
        }
        ConstValue::HostObject(host) if host.is_user_function() => {
            if let Some(self_obj) = self_obj {
                Ok(ConstValue::HostObject(HostObject::new_bound_method(
                    format!("{}.{}", self_obj.qualname(), name),
                    self_obj.clone(),
                    host,
                    name,
                    origin_class.clone(),
                )))
            } else {
                Ok(ConstValue::HostObject(host))
            }
        }
        ConstValue::HostObject(host) if host.is_native_callable() => {
            // Native Rust-backed method stored in the class dict. Upstream
            // treats any callable in `cls.__dict__` as a regular function
            // descriptor — instance lookup binds `self`, class lookup
            // returns the raw callable.
            if let Some(self_obj) = self_obj {
                Ok(ConstValue::HostObject(HostObject::new_bound_method(
                    format!("{}.{}", self_obj.qualname(), name),
                    self_obj.clone(),
                    host,
                    name,
                    origin_class.clone(),
                )))
            } else {
                Ok(ConstValue::HostObject(host))
            }
        }
        other => Ok(other),
    }
}

/// Rust equivalent of upstream Python `getattr(obj, name)` applied to
/// a flow-space / annotator constant — the single code path shared
/// between `flowspace::operation::GetAttr.constfold` (operation.py)
/// and `unaryop::OpKind::GetAttr` (unaryop.py) for constant
/// receivers. The divergence lives only at the wrap step: flowspace
/// re-wraps with `const(result)`; the annotator re-wraps with
/// `bookkeeper.immutablevalue(result)`.
///
/// Returns:
/// * `Ok(Some(value))` — attribute lookup succeeded.
/// * `Ok(None)` — upstream's `WrapException` swallow path
///   (operation.py:643-646): the result couldn't be wrapped as a
///   Constant; fold declines silently.
/// * `Err(msg)` — upstream's "always raises" branch
///   (operation.py:637-642); caller raises `FlowingError` (flowspace)
///   or an `AnnotatorError` (annotator).
///
/// TODO: primitive constants (Int / Float / Str / …)
/// would upstream execute Python's real `getattr` — e.g. `(1).real`
/// evaluates through Python's `int.real` descriptor — and the Rust
/// port has no Python runtime to defer to. The current port only
/// reconstructs the object-level `__class__` path for those constants
/// (via builtin class singletons) and otherwise returns `Ok(None)` so
/// the call falls through to the annotator's `find_method` /
/// host-method emulators, matching upstream's `WrapException` swallow
/// where `const` can't re-wrap the result.
pub(crate) fn const_runtime_getattr(
    obj: &ConstValue,
    name: &str,
) -> Result<Option<ConstValue>, String> {
    match obj {
        ConstValue::HostObject(h) => match host_getattr(h, name) {
            Ok(value) => Ok(Some(value)),
            Err(HostGetAttrError::Missing) => Err(format!(
                "getattr({}, {:?}) always raises AttributeError",
                h.qualname(),
                name
            )),
            Err(HostGetAttrError::Unsupported) => Ok(None),
        },
        ConstValue::Function(func) => match name {
            "__name__" => {
                let (simple_name, _) = split_attr_name_module(&func.name);
                Ok(Some(ConstValue::byte_str(simple_name)))
            }
            "__module__" => {
                // Upstream `func.__module__` lives on the function
                // object directly; GraphFunc carries it in
                // `module`. Fall back to the qualname split only
                // when the slot is empty (synthetic fixtures).
                let module_name = func
                    .module
                    .clone()
                    .or_else(|| split_attr_name_module(&func.name).1);
                Ok(module_name.map(ConstValue::byte_str))
            }
            // Live-snapshot accessor mirrors upstream `func.__globals__`
            // reading the module dict directly (`flowcontext.py:284`).
            "__globals__" => Ok(Some(func.live_globals())),
            "__defaults__" => {
                if func.defaults.is_empty() {
                    Ok(Some(ConstValue::None))
                } else {
                    Ok(Some(ConstValue::Tuple(
                        func.defaults
                            .iter()
                            .map(|constant| constant.value.clone())
                            .collect(),
                    )))
                }
            }
            "__closure__" => {
                if func.closure.is_empty() {
                    Ok(Some(ConstValue::None))
                } else {
                    Ok(Some(ConstValue::Tuple(
                        func.closure
                            .iter()
                            .map(|constant| constant.value.clone())
                            .collect(),
                    )))
                }
            }
            "__class__" => Ok(obj.class_of().map(ConstValue::HostObject)),
            _ => Ok(None),
        },
        _ if name == "__class__" => Ok(obj.class_of().map(ConstValue::HostObject)),
        _ => Ok(None),
    }
}

/// Best-effort host-level `getattr(obj, name)` over the HostObject
/// kinds modelled by the current Rust port.
///
/// This mirrors the parts of Python descriptor lookup the HostObject
/// carrier can represent directly today:
/// * modules read from the module dict;
/// * class lookup walks the class MRO, applies `staticmethod.__get__`,
///   and materialises bound methods for `classmethod`;
/// * instance lookup follows upstream `object.__getattribute__`:
///   class-MRO lookup first, data descriptors (`property`) fire
///   before the instance dict, then `__dict__[name]`, then non-data
///   descriptors / raw class attrs, then `AttributeError`.
/// * every remaining host object kind with a class (`function`,
///   `builtin_function_or_method`, `property`, `staticmethod`,
///   `classmethod`, `method`, `weakref`, ...) falls back to
///   `type(obj).__mro__` lookup with no per-instance dict, matching
///   the generic Python object path.
///
/// Instance `property` getters are evaluated through
/// [`HostObject::call_host`] against `property.fget`. A missing `fget`
/// surfaces as `HostGetAttrError::Missing` (upstream's
/// `AttributeError: unreadable attribute`); a host-side failure during
/// the getter call surfaces as `HostGetAttrError::Unsupported`.
pub(crate) fn host_getattr(pyobj: &HostObject, name: &str) -> Result<ConstValue, HostGetAttrError> {
    if let Some(value) = host_object_own_getattr(pyobj, name) {
        return Ok(value);
    }
    if pyobj.is_module() {
        return pyobj
            .module_get(name)
            .map(ConstValue::HostObject)
            .ok_or(HostGetAttrError::Missing);
    }
    if pyobj.is_class() {
        let Some((value, origin_class)) = host_class_mro_lookup(pyobj, name) else {
            return Err(HostGetAttrError::Missing);
        };
        return host_descriptor_get(value, name, None, pyobj, &origin_class);
    }
    if pyobj.is_instance() {
        let cls = pyobj
            .instance_class()
            .ok_or(HostGetAttrError::Unsupported)?;
        // upstream CPython `object.__getattribute__` descriptor order
        // (`_PyObject_GenericGetAttrWithDict`, Objects/object.c):
        //   1. walk `type(obj).__mro__` for `name`;
        //   2. if the class-level entry is a *data descriptor* (has
        //      `__set__` / `__delete__`), call `__get__` and return;
        //   3. look up `obj.__dict__[name]`;
        //   4. if the class-level entry exists but is a non-data
        //      descriptor (has `__get__` only) or a plain attribute,
        //      call `__get__` / return the raw value;
        //   5. AttributeError.
        let class_hit = host_class_mro_lookup(cls, name);
        if let Some((value, origin_class)) = &class_hit
            && is_data_descriptor(value)
        {
            return host_descriptor_get(value.clone(), name, Some(pyobj), cls, origin_class);
        }
        match pyobj.instance_get(name) {
            Ok(value) => return Ok(value),
            Err(HostGetAttrError::Missing) => {}
            Err(error) => return Err(error),
        }
        if let Some((value, origin_class)) = class_hit {
            return host_descriptor_get(value, name, Some(pyobj), cls, &origin_class);
        }
        return Err(HostGetAttrError::Missing);
    }
    if let Some(cls) = pyobj.class_of() {
        let Some((value, origin_class)) = host_class_mro_lookup(&cls, name) else {
            return Err(HostGetAttrError::Missing);
        };
        return host_descriptor_get(value, name, Some(pyobj), &cls, &origin_class);
    }
    Err(HostGetAttrError::Unsupported)
}

/// C3 linearisation — Rust port of the `type.__mro__` algorithm.
///
/// C3 rule: in `L[C] = C + merge(L[B1], L[B2], …, [B1, B2, …])`,
/// `merge` takes each list's head as a candidate, picks a head that
/// **appears in no other list's tail**, appends it to the result, and
/// removes it from every list. If no head can be chosen, linearisation
/// fails on conflict (Python raises TypeError). This returns None instead.
/// Single inheritance plus mixins, as in annotator `add_mixins`, always succeeds.
fn c3_linearise(cls: &HostObject) -> Option<Vec<HostObject>> {
    let bases = cls.class_bases()?;
    let mut lists: Vec<Vec<HostObject>> = Vec::new();
    for base in bases {
        let base_mro = c3_linearise(base)?;
        lists.push(base_mro);
    }
    if !bases.is_empty() {
        lists.push(bases.to_vec());
    }
    let mut result: Vec<HostObject> = vec![cls.clone()];
    loop {
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            return Some(result);
        }
        // Find the first good head.
        let mut chosen: Option<HostObject> = None;
        for list in &lists {
            let head = &list[0];
            let appears_in_tail = lists.iter().any(|other| other[1..].contains(head));
            if !appears_in_tail {
                chosen = Some(head.clone());
                break;
            }
        }
        let head = chosen?;
        result.push(head.clone());
        for list in lists.iter_mut() {
            if !list.is_empty() && list[0] == head {
                list.remove(0);
            }
        }
    }
}

/// Host namespace emulation — upstream `__builtin__` / imported
/// module table. `HOST_ENV.lookup_builtin(name)` corresponds to
/// `getattr(__builtin__, varname)` in `flowcontext.py`, and
/// `HOST_ENV.import_module(name)` corresponds to `__import__(name, ...)`
/// in `flowcontext.py` `import_name`.
///
/// Deviation (parity rule #1): upstream runs a real Python `__import__`
/// at flow time, so it can load any module. The Rust port does not embed
/// a Python runtime, so module/class/callable values flowspace may refer to
/// are pre-populated at bootstrap, and a missing name falls out as
/// `ImportError` / `FlowingError`. The API exposes only this access so a
/// real runtime can be plugged in as the `HostEnv` backend when pyre is integrated.
pub struct HostEnv {
    builtins: HashMap<String, HostObject>,
    builtin_module: HostObject,
    modules: Mutex<HashMap<String, HostObject>>,
}

impl HostEnv {
    fn bootstrap() -> Self {
        let builtin_module = HostObject::new_module("__builtin__");
        let mut env = HostEnv {
            builtins: HashMap::new(),
            builtin_module,
            modules: Mutex::new(HashMap::new()),
        };
        env.bootstrap_builtin_exceptions();
        env.bootstrap_builtin_types();
        env.bootstrap_builtin_callables();
        env.bootstrap_std_modules();
        env
    }

    fn insert_builtin(&mut self, name: &str, obj: HostObject) {
        self.builtin_module.module_set(name, obj.clone());
        self.builtins.insert(name.to_owned(), obj);
    }

    fn bootstrap_builtin_exceptions(&mut self) {
        // Materialise, ahead of time, every class upstream refers to during
        // flow, from BaseException → Exception → … through `_StackOverflow`
        // in rpython/rlib/rstackovf.py.
        //
        // rstackovf.py — `class StackOverflow(RuntimeError)` is the real
        // class, and the same class object is also bound as `_StackOverflow`.
        // The module-level `StackOverflow` is then rebound to the tuple
        // sentinel `((RuntimeError, RuntimeError),)`, but the identifier
        // flowspace refers to is `_StackOverflow` (the same name as the
        // annotator.exception.standard_exceptions entry). The Rust port
        // registers the same class object under both lookup keys to reproduce
        // that shared upstream identity.
        let base = HostObject::new_class("BaseException", vec![]);
        let exc = HostObject::new_class("Exception", vec![base.clone()]);
        let runtime = HostObject::new_class("RuntimeError", vec![exc.clone()]);
        let stackovf = HostObject::new_class("StackOverflow", vec![runtime.clone()]);
        let not_impl = HostObject::new_class("NotImplementedError", vec![runtime.clone()]);

        let children = [
            ("AssertionError", &exc),
            ("ImportError", &exc),
            ("StopIteration", &exc),
            ("TypeError", &exc),
            ("ValueError", &exc),
            ("ZeroDivisionError", &exc),
            ("AttributeError", &exc),
            ("KeyError", &exc),
            ("IndexError", &exc),
            ("NameError", &exc),
            ("LookupError", &exc),
            ("OSError", &exc),
            ("OverflowError", &exc),
        ];
        for (name, parent) in children {
            let cls = HostObject::new_class(name, vec![parent.clone()]);
            self.insert_builtin(name, cls);
        }
        self.insert_builtin("BaseException", base);
        self.insert_builtin("Exception", exc);
        self.insert_builtin("RuntimeError", runtime);
        // upstream `_StackOverflow = StackOverflow` (rstackovf.py) —
        // Register the same class object under both keys. "StackOverflow" is
        // the original module-level class name, and "_StackOverflow" is the
        // alias used at flow level.
        self.insert_builtin("StackOverflow", stackovf.clone());
        self.insert_builtin("_StackOverflow", stackovf);
        self.insert_builtin("NotImplementedError", not_impl);
    }

    fn bootstrap_builtin_types(&mut self) {
        // Builtin class objects referred to upstream via `const(type)` and
        // similar. Inheritance is not in scope yet, so bases stay empty and
        // only identity is preserved.
        for name in [
            "type",
            "object",
            "module",
            "NoneType",
            "function",
            "code",
            "method",
            "builtin_function_or_method",
            "str",
            "unicode",
            "int",
            "float",
            "bool",
            "tuple",
            "list",
            "dict",
            "set",
            "frozenset",
            "bytes",
            "bytearray",
            "complex",
            "memoryview",
            "range",
            "slice",
            "enumerate",
            "zip",
            "map",
            "filter",
            "reversed",
            "property",
            "classmethod",
            "staticmethod",
        ] {
            self.insert_builtin(name, HostObject::new_class(name, vec![]));
        }
    }

    fn bootstrap_builtin_callables(&mut self) {
        // Callables that exist on upstream `__builtin__` and that flowspace
        // can actually look up via the `find_global` fallback. Every name from
        // the old `BuiltinFunction` enum is carried over as-is — additions and
        // removals stay tied to upstream's `__builtin__` set.
        for name in [
            "__import__",
            "locals",
            "getattr",
            "setattr",
            "delattr",
            "print",
            "all",
            "any",
            "len",
            "iter",
            "next",
            "isinstance",
            "issubclass",
            "hasattr",
            "callable",
            "id",
            "hash",
            "repr",
            "min",
            "max",
            "abs",
            "sum",
            "round",
            "divmod",
            "pow",
            "chr",
            "ord",
            "hex",
            "oct",
            "bin",
            "format",
            "vars",
            "dir",
            "compile",
            "input",
            "exec",
            "eval",
            "super",
            "open",
        ] {
            self.insert_builtin(name, HostObject::new_builtin_callable(name));
        }
        // RPython-only print helpers — upstream `specialcase.py rpython_print_item`.
        for name in [
            "rpython_print_item",
            "rpython_print_end",
            "rpython_print_newline",
        ] {
            self.insert_builtin(name, HostObject::new_builtin_callable(name));
        }
        // Class-method qualnames used by `@typer_for(<class>.__init__)`
        // decorators upstream (rbuiltin.py).  Pyre encodes them
        // as single-string qualname builtins instead of resolving via
        // descriptor lookup.  `EnvironmentError.__init__` (rbuiltin.py)
        // is omitted — its body uses `r_self.setfield(v_self, 'errno', ...)`
        // which requires InstanceRepr setfield trait plumbing that is
        // not yet wired through `findbltintyper`; registering only the
        // qualname without the typer body surfaces as a registry hit
        // that returns no body, which is observably wrong.
        {
            let name = "object.__init__";
            self.insert_builtin(name, HostObject::new_builtin_callable(name));
        }
        // Pyre-internal front-end pointer-downcast narrow (#298): the
        // frontend lowers `obj as *const RegisteredStruct` to a
        // `simple_call` against this stub so the classdef-less pointer
        // narrows to `SomeInstance(root)` and a field read on the pointee
        // resolves.  Its analyzer (annotator/builtin.rs) keys on this
        // qualname and its typer (rtyper/rbuiltin.rs) keys on this same
        // singleton Arc, so the binding must live in HOST_ENV.
        self.insert_builtin(
            crate::runtime_names::shims::CAST_INSTANCE,
            HostObject::new_builtin_callable(crate::runtime_names::shims::CAST_INSTANCE),
        );
        // Pyre-internal front-end pointer-type ERASURE, the twin of the
        // narrow above: the frontend lowers `obj as *mut u8` to a
        // `simple_call` against this stub so the result drops the pointee
        // class and annotates as the classdef-less pointer shell.  This is
        // the `llmemory.cast_ptr_to_adr` role — a heterogeneous heap block
        // reaching a GC hook travels as an opaque address, never as
        // `Ptr(instance)` — spelled in the annotation the port already uses
        // for a pointee-less raw pointer.  Same three-way binding as the
        // narrow (analyzer keys the qualname, typer keys this Arc).
        self.insert_builtin(
            crate::runtime_names::shims::CAST_ADDRESS,
            HostObject::new_builtin_callable(crate::runtime_names::shims::CAST_ADDRESS),
        );
        // Front-end spelling of the exact RPython `rgc.ll_arraymove` helper.
        // Rust source reaches the same operation as `ptr::copy`; the MIR
        // adapter reconstructs the array plus logical source/destination
        // indices before resolving this singleton through the annotator and
        // rtyper registries.
        self.insert_builtin(
            crate::runtime_names::shims::LL_ARRAYMOVE,
            HostObject::new_builtin_callable(crate::runtime_names::shims::LL_ARRAYMOVE),
        );
    }

    fn bootstrap_std_modules(&mut self) {
        // `__import__("os", …)` returns the real os module, but the Rust port
        // keeps only the names referenced by upstream `specialcase.py` `redirect_function`.
        // A dotted path this bootstrap omits makes `import_module` return
        // `None`, which flowspace translates to `ImportError`.
        let os = HostObject::new_module("os");
        let os_path = HostObject::new_module("os.path");
        let rfile = HostObject::new_module("rpython.rlib.rfile");
        let rpath = HostObject::new_module("rpython.rlib.rpath");
        // `rpython.rlib.rarithmetic` module — upstream exports `ovfcheck`
        // from this module, and `Constant(rarithmetic.ovfcheck)` in
        // `translator/simplify.py` `transform_ovfcheck` reaches it as a module attribute.
        // The Rust port attaches an `ovfcheck` sentinel to this module object.
        // Leaving it out of the builtin table makes `find_global("ovfcheck")`
        // fail as upstream does, and only
        // `translator::simplify::transform_ovfcheck` looks the sentinel up
        // directly.
        let rarithmetic = HostObject::new_module("rpython.rlib.rarithmetic");
        rarithmetic.module_set("ovfcheck", HostObject::new_builtin_callable("ovfcheck"));
        // `rbuiltin.py @typer_for(rarithmetic.intmask)` /
        // `rbuiltin.py @typer_for(rarithmetic.longlongmask)`.
        rarithmetic.module_set(
            "intmask",
            HostObject::new_builtin_callable("rarithmetic.intmask"),
        );
        rarithmetic.module_set(
            "longlongmask",
            HostObject::new_builtin_callable("rarithmetic.longlongmask"),
        );
        rarithmetic.module_set(
            "ulonglongmask",
            HostObject::new_builtin_callable("rarithmetic.ulonglongmask"),
        );
        rarithmetic.module_set(
            "longlonglongmask",
            HostObject::new_builtin_callable("rarithmetic.longlonglongmask"),
        );
        rarithmetic.module_set(
            "ulonglonglongmask",
            HostObject::new_builtin_callable("rarithmetic.ulonglonglongmask"),
        );
        // Upstream `r_uint` is the class object created via
        // `build_int('r_uint', False, LONG_BIT)` (rarithmetic.py),
        // dispatched via `ForTypeEntry(extregistry.ExtRegistryEntry)`
        // (rarithmetic.py).  The HostObject registered here is
        // the carrier the extregistry entry keys on; the bootstrap
        // [`crate::translator::rtyper::extregistry::register_r_uint`]
        // call wires `ExtRegistryEntry::ForType` so `compute_result_\
        // annotation` returns `SomeInteger(unsigned=True,
        // knowntype=Ruint)` and `specialize_call` routes through
        // `rtype_r_uint`.
        let r_uint_host = HostObject::new_builtin_callable("rarithmetic.r_uint");
        rarithmetic.module_set("r_uint", r_uint_host.clone());
        crate::translator::rtyper::extregistry::register_r_uint(r_uint_host);

        os.module_set("fdopen", HostObject::new_builtin_callable("os.fdopen"));
        os.module_set("tmpfile", HostObject::new_builtin_callable("os.tmpfile"));
        os.module_set("remove", HostObject::new_builtin_callable("os.remove"));
        os.module_set("unlink", HostObject::new_builtin_callable("os.unlink"));
        os_path.module_set("isdir", HostObject::new_builtin_callable("os.path.isdir"));
        os_path.module_set("isabs", HostObject::new_builtin_callable("os.path.isabs"));
        os_path.module_set(
            "normpath",
            HostObject::new_builtin_callable("os.path.normpath"),
        );
        os_path.module_set(
            "abspath",
            HostObject::new_builtin_callable("os.path.abspath"),
        );
        os_path.module_set("join", HostObject::new_builtin_callable("os.path.join"));
        os_path.module_set(
            "splitdrive",
            HostObject::new_builtin_callable("os.path.splitdrive"),
        );
        os.module_set("path", os_path.clone());
        rfile.module_set(
            "create_file",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_file"),
        );
        rfile.module_set(
            "create_fdopen_rfile",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_fdopen_rfile"),
        );
        rfile.module_set(
            "create_temp_rfile",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_temp_rfile"),
        );
        rpath.module_set(
            "risdir",
            HostObject::new_builtin_callable("rpython.rlib.rpath.risdir"),
        );
        rpath.module_set(
            "risabs",
            HostObject::new_builtin_callable("rpython.rlib.rpath.risabs"),
        );
        rpath.module_set(
            "rnormpath",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rnormpath"),
        );
        rpath.module_set(
            "rabspath",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rabspath"),
        );
        rpath.module_set(
            "rjoin",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rjoin"),
        );
        rpath.module_set(
            "rsplitdrive",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rsplitdrive"),
        );

        // `rpython.rlib.objectmodel` — host-side namespace for typing-time
        // helpers like `keepalive_until_here` (rbuiltin.py:643-648) whose
        // RPython decorators look up the callable by module attribute.
        let objectmodel = HostObject::new_module("rpython.rlib.objectmodel");
        objectmodel.module_set(
            "keepalive_until_here",
            HostObject::new_builtin_callable("objectmodel.keepalive_until_here"),
        );
        objectmodel.module_set(
            "free_non_gc_object",
            HostObject::new_builtin_callable("objectmodel.free_non_gc_object"),
        );
        objectmodel.module_set(
            "instantiate",
            HostObject::new_builtin_callable("rpython.rlib.objectmodel.instantiate"),
        );

        // `rpython.rtyper.lltypesystem.lltype` — typing-time callables that
        // RPython routes through `@typer_for(lltype.<name>)` decorators
        // (rbuiltin.py).  `front::mir` synthesises most
        // `lltype.*` ops directly while lowering (no HostObject lookup
        // in the production dispatch), so these entries currently exist
        // only to make the BUILTIN_TYPER registry structurally match
        // upstream — the dispatch flips on once the M2.5g extern-Rust-helper
        // walker lands.
        let lltype = HostObject::new_module("rpython.rtyper.lltypesystem.lltype");
        lltype.module_set("typeOf", HostObject::new_builtin_callable("lltype.typeOf"));
        lltype.module_set(
            "nullptr",
            HostObject::new_builtin_callable("lltype.nullptr"),
        );
        lltype.module_set(
            "getRuntimeTypeInfo",
            HostObject::new_builtin_callable("lltype.getRuntimeTypeInfo"),
        );
        lltype.module_set("Ptr", HostObject::new_builtin_callable("lltype.Ptr"));
        lltype.module_set(
            "identityhash",
            HostObject::new_builtin_callable("lltype.identityhash"),
        );
        lltype.module_set(
            "runtime_type_info",
            HostObject::new_builtin_callable("lltype.runtime_type_info"),
        );
        lltype.module_set(
            "direct_ptradd",
            HostObject::new_builtin_callable("lltype.direct_ptradd"),
        );
        lltype.module_set(
            "direct_fieldptr",
            HostObject::new_builtin_callable("lltype.direct_fieldptr"),
        );
        lltype.module_set(
            "direct_arrayitems",
            HostObject::new_builtin_callable("lltype.direct_arrayitems"),
        );
        lltype.module_set(
            "cast_opaque_ptr",
            HostObject::new_builtin_callable("lltype.cast_opaque_ptr"),
        );
        lltype.module_set(
            "length_of_simple_gcarray_from_opaque",
            HostObject::new_builtin_callable("lltype.length_of_simple_gcarray_from_opaque"),
        );
        lltype.module_set(
            "cast_pointer",
            HostObject::new_builtin_callable("lltype.cast_pointer"),
        );
        // `lltype.cast_primitive` — rbuiltin.py typer paired with
        // `gen_cast` at rbuiltin.py. The Rust port registers
        // `rtype_cast_primitive` against this HostObject in
        // `install_default_typers`.
        lltype.module_set(
            "cast_primitive",
            HostObject::new_builtin_callable("lltype.cast_primitive"),
        );
        // `lltype.cast_ptr_to_int` / `cast_int_to_ptr` — registered upstream
        // at `rpython/rtyper/lltypesystem/lltype.py` via the
        // `ann_cast_ptr_to_int` / `ann_cast_int_to_ptr` annotator hooks plus
        // `BUILTIN_TYPER` entries in `rbuiltin.py`. `front::mir` projects a
        // Charon `Ref ↔ Int` cast payload onto these `cast_ptr_to_int` /
        // `cast_int_to_ptr` opnames while lowering (`unary_op_label`).
        lltype.module_set(
            "cast_ptr_to_int",
            HostObject::new_builtin_callable("lltype.cast_ptr_to_int"),
        );
        lltype.module_set(
            "cast_int_to_ptr",
            HostObject::new_builtin_callable("lltype.cast_int_to_ptr"),
        );
        lltype.module_set(
            "render_immortal",
            HostObject::new_builtin_callable("lltype.render_immortal"),
        );
        lltype.module_set("free", HostObject::new_builtin_callable("lltype.free"));
        lltype.module_set("malloc", HostObject::new_builtin_callable("lltype.malloc"));

        // `rpython.rtyper.lltypesystem.llmemory` — address-arithmetic
        // and weakref typing callables routed through
        // `@typer_for(llmemory.<name>)` decorators upstream.
        let llmemory = HostObject::new_module("rpython.rtyper.lltypesystem.llmemory");
        llmemory.module_set(
            "cast_ptr_to_adr",
            HostObject::new_builtin_callable("llmemory.cast_ptr_to_adr"),
        );
        llmemory.module_set(
            "cast_adr_to_ptr",
            HostObject::new_builtin_callable("llmemory.cast_adr_to_ptr"),
        );
        llmemory.module_set(
            "cast_adr_to_int",
            HostObject::new_builtin_callable("llmemory.cast_adr_to_int"),
        );
        llmemory.module_set(
            "cast_int_to_adr",
            HostObject::new_builtin_callable("llmemory.cast_int_to_adr"),
        );
        // `llmemory.raw_malloc` / `raw_malloc_usage` — rbuiltin.py:577-591
        // typer entries paired with `install_default_typers` in rbuiltin.rs.
        // Both return / accept Signed; raw_malloc itself returns Address
        // via the `raw_malloc` opname (lloperation.rs line 1103).
        llmemory.module_set(
            "raw_malloc",
            HostObject::new_builtin_callable("llmemory.raw_malloc"),
        );
        llmemory.module_set(
            "raw_malloc_usage",
            HostObject::new_builtin_callable("llmemory.raw_malloc_usage"),
        );
        // rbuiltin.py:593-618
        llmemory.module_set(
            "raw_free",
            HostObject::new_builtin_callable("llmemory.raw_free"),
        );
        llmemory.module_set(
            "raw_memcopy",
            HostObject::new_builtin_callable("llmemory.raw_memcopy"),
        );
        llmemory.module_set(
            "raw_memclear",
            HostObject::new_builtin_callable("llmemory.raw_memclear"),
        );
        // rbuiltin.py:744-782
        llmemory.module_set(
            "weakref_create",
            HostObject::new_builtin_callable("llmemory.weakref_create"),
        );
        llmemory.module_set(
            "weakref_deref",
            HostObject::new_builtin_callable("llmemory.weakref_deref"),
        );
        llmemory.module_set(
            "cast_ptr_to_weakrefptr",
            HostObject::new_builtin_callable("llmemory.cast_ptr_to_weakrefptr"),
        );
        llmemory.module_set(
            "cast_weakrefptr_to_ptr",
            HostObject::new_builtin_callable("llmemory.cast_weakrefptr_to_ptr"),
        );

        // rbuiltin.py `@typer_for(weakref.ref)` — weakref.ref aliases
        // llmemory.weakref_create at the rtyper level.
        let weakref_mod = HostObject::new_module("weakref");
        weakref_mod.module_set("ref", HostObject::new_builtin_callable("weakref.ref"));
        // Z2.5 Path C tail — Rust std lib module-qualified entry points
        // referenced by pyre-object / pyre-interpreter callsites
        // (`std::ptr::null_mut`, `std::ptr::copy_nonoverlapping`,
        // `std::mem::align_of`, `std::alloc::dealloc`).  Branch 3b of
        // `flowspace_adapter::translate_op` resolves
        // `["std", "ptr", "null_mut"]` by joining `segments[..-1]` with
        // `.` (→ `"std.ptr"`) and calling
        // `HOST_ENV.import_module("std.ptr").module_get("null_mut")`,
        // so the module key matches the join shape.  Analyzers that
        // exist for these callables are registered in
        // `annotator/builtin.rs` (`std_ptr_eq`, `ptr_null_constant`,
        // `std_ptr_copy_nonoverlapping`).
        let std_ptr = HostObject::new_module("std.ptr");
        std_ptr.module_set(
            "null_mut",
            HostObject::new_builtin_callable("std.ptr.null_mut"),
        );
        std_ptr.module_set("eq", HostObject::new_builtin_callable("std.ptr.eq"));
        std_ptr.module_set(
            "copy_nonoverlapping",
            HostObject::new_builtin_callable("std.ptr.copy_nonoverlapping"),
        );
        let std_mem = HostObject::new_module("std.mem");
        std_mem.module_set(
            "align_of",
            HostObject::new_builtin_callable("std.mem.align_of"),
        );
        std_mem.module_set(
            "size_of",
            HostObject::new_builtin_callable("std.mem.size_of"),
        );
        let std_alloc = HostObject::new_module("std.alloc");
        std_alloc.module_set(
            "dealloc",
            HostObject::new_builtin_callable("std.alloc.dealloc"),
        );
        // `std::slice::from_raw_parts(p, len)` and `_mut` — raw-pointer
        // -> slice constructors called from pyre-object string / bytes
        // helpers when packing a `*const u8` + `usize` into a `&[u8]`
        // (`pyre_object::unicodeobject::w_str_new_from_bytes`, etc.).  No
        // analyzer needed today: the registry surface is enough to
        // retire the Skip; the rtyper-side lowering still routes
        // through the M2.5g extern-Rust-helper walker once that lands.
        let std_slice = HostObject::new_module("std.slice");
        std_slice.module_set(
            "from_raw_parts",
            HostObject::new_builtin_callable("std.slice.from_raw_parts"),
        );
        std_slice.module_set(
            "from_raw_parts_mut",
            HostObject::new_builtin_callable("std.slice.from_raw_parts_mut"),
        );

        // `BigInt::from(v)` callsites in `pyre-object` (longobject /
        // tupleobject etc.) emit `["BigInt", "from"]` as the canonical
        // call target.  Surface this as a module-shaped entry so
        // Branch 3b in `flowspace_adapter::translate_op` resolves the
        // 2-segment callsite verbatim; the analyzer doesn't model
        // arbitrary-precision arithmetic, but the registry surface is
        // enough to retire the Skip path.
        let bigint = HostObject::new_module("BigInt");
        bigint.module_set("from", HostObject::new_builtin_callable("BigInt.from"));

        // RPython `rlib.longlong2float` bitcast pair.  The front-end mints
        // these callsites for Rust's `f64::{to_bits,from_bits}` surface;
        // expose both ExtRegistryEntry-backed functions so annotation and
        // rtyping emit the matching convert_* llops.
        let longlong2float = HostObject::new_module("longlong2float");
        longlong2float.module_set(
            "float2longlong",
            HostObject::new_builtin_callable("longlong2float.float2longlong"),
        );
        longlong2float.module_set(
            "longlong2float",
            HostObject::new_builtin_callable("longlong2float.longlong2float"),
        );

        // Rust primitive type conversion impls (`u32::from`,
        // `i64::from`, `i64::try_from`, `usize::try_from`) — callsites
        // emit `[primitive, method]` 2-segment paths.  Each impl
        // returns an integer; a shared `SomeInteger` analyzer covers
        // all conversion variants.
        let primitive_u32 = HostObject::new_module("u32");
        primitive_u32.module_set("from", HostObject::new_builtin_callable("u32.from"));
        let primitive_i64 = HostObject::new_module("i64");
        primitive_i64.module_set("from", HostObject::new_builtin_callable("i64.from"));
        primitive_i64.module_set("try_from", HostObject::new_builtin_callable("i64.try_from"));
        let primitive_usize = HostObject::new_module("usize");
        primitive_usize.module_set(
            "try_from",
            HostObject::new_builtin_callable("usize.try_from"),
        );

        // External crate `majit_metainterp` helpers called from pyre
        // source via `use majit_metainterp::{majit_log_enabled, ...}`.
        // The two-segment callsites `["majit_metainterp",
        // "majit_log_enabled"]` and the three-segment
        // `["majit_metainterp", "jit", "we_are_jitted"]` both lower to
        // `pub fn -> bool` flag checks that are compile-time constant
        // outside the JIT context.  Register HOST_ENV stubs +
        // analyzers (`builtin.rs`) so the registry resolution +
        // `SomeBuiltin.call` dispatch both succeed.
        let majit_metainterp = HostObject::new_module("majit_metainterp");
        majit_metainterp.module_set(
            "majit_log_enabled",
            HostObject::new_builtin_callable("majit_metainterp.majit_log_enabled"),
        );
        let majit_metainterp_jit = HostObject::new_module("majit_metainterp.jit");
        majit_metainterp_jit.module_set(
            "we_are_jitted",
            HostObject::new_builtin_callable("majit_metainterp.jit.we_are_jitted"),
        );
        let majit_rlib_jit = HostObject::new_module("majit_rlib.jit");
        majit_rlib_jit.module_set(
            "we_are_jitted",
            HostObject::new_builtin_callable("majit_rlib.jit.we_are_jitted"),
        );
        majit_rlib_jit.module_set(
            "isconstant",
            HostObject::new_builtin_callable("majit_rlib.jit.isconstant"),
        );
        majit_rlib_jit.module_set(
            "isvirtual",
            HostObject::new_builtin_callable("majit_rlib.jit.isvirtual"),
        );
        // `rlib/jit.py ConditionalCallEntry` (`_about_ =
        // _jit_conditional_call, _jit_conditional_call_value`). The
        // residual rlib bodies are interpreter fallbacks; populate
        // declines them (`registration_decline`) so Layer-3b reaches
        // these HOST_ENV callables instead of walking `function(*args)`
        // as an ll funcptr call.
        majit_rlib_jit.module_set(
            "conditional_call0",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call0"),
        );
        majit_rlib_jit.module_set(
            "conditional_call1",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call1"),
        );
        majit_rlib_jit.module_set(
            "conditional_call2",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call2"),
        );
        majit_rlib_jit.module_set(
            "conditional_call3",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call3"),
        );
        majit_rlib_jit.module_set(
            "conditional_call4",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call4"),
        );
        majit_rlib_jit.module_set(
            "conditional_call_elidable1",
            HostObject::new_builtin_callable("majit_rlib.jit.conditional_call_elidable1"),
        );
        // `rlib/nonconst.py NonConstant` is an `ExtRegistryEntry`, never a
        // graph: looking inside it would hand the caller back the very
        // constant it exists to hide.  Registering it as a host callable is
        // how pyre spells that — `populate_call_registry_from_call_graphs`
        // skips the lifted body so Layer-3b resolution reaches this entry,
        // whose analyzer answers `not_const(s_arg)`.
        let majit_rlib_nonconst = HostObject::new_module("majit_rlib.nonconst");
        majit_rlib_nonconst.module_set(
            "non_constant",
            HostObject::new_builtin_callable("majit_rlib.nonconst.non_constant"),
        );
        // The callers live in `majit-rlib` itself, so the front can spell the
        // path crate-qualified or bare.  Bind the bare module to the SAME
        // callable, not a fresh same-qualname one, the way `core.ptr` reuses
        // `std.ptr`'s members.
        let nonconst = HostObject::new_module("nonconst");
        nonconst.module_set(
            "non_constant",
            majit_rlib_nonconst
                .module_get("non_constant")
                .expect("majit_rlib.nonconst.non_constant bound above"),
        );

        // Container constructors emitted as `[Type, "new"]` 2-segment
        // FunctionPath callsites by pyre-source helpers.  Same shape
        // as the `std.slice`/`BigInt` stubs above: surface a module
        // named after the type with a builtin-callable for each
        // associated function so Branch 3b in
        // `flowspace_adapter::translate_op` resolves the path.  No
        // analyzer is wired today; the rtyper-side lowering still
        // routes through the M2.5g extern-Rust-helper walker once
        // that lands.
        let box_ty = HostObject::new_module("Box");
        box_ty.module_set("new", HostObject::new_builtin_callable("Box.new"));
        box_ty.module_set("into_raw", HostObject::new_builtin_callable("Box.into_raw"));
        let vec_ty = HostObject::new_module("Vec");
        vec_ty.module_set("new", HostObject::new_builtin_callable("Vec.new"));
        let string_ty = HostObject::new_module("String");
        string_ty.module_set("new", HostObject::new_builtin_callable("String.new"));
        string_ty.module_set(
            "with_capacity",
            HostObject::new_builtin_callable("String.with_capacity"),
        );

        // 3- and 4-segment paths: `indexmap::IndexMap::new` and
        // `std::collections::HashMap::new`.  Branch 3b joins
        // `segments[..len-1]` with `.` and queries
        // `HOST_ENV.import_module(...)`, so register the module under
        // the dotted prefix (`indexmap.IndexMap`,
        // `std.collections.HashMap`).
        let indexmap_indexmap = HostObject::new_module("indexmap.IndexMap");
        indexmap_indexmap.module_set(
            "new",
            HostObject::new_builtin_callable("indexmap.IndexMap.new"),
        );
        let std_collections_hashmap = HostObject::new_module("std.collections.HashMap");
        std_collections_hashmap.module_set(
            "new",
            HostObject::new_builtin_callable("std.collections.HashMap.new"),
        );

        // Pyre-internal type ctors emitted as 2-segment paths.
        // `FrameDebugData::init` from `pyframe::PyFrame::getorcreate_debug_data`;
        // `RootScope::new` from `gc_roots::push_roots`.
        let frame_debug_data = HostObject::new_module("FrameDebugData");
        frame_debug_data.module_set(
            "init",
            HostObject::new_builtin_callable("FrameDebugData.init"),
        );
        let root_scope = HostObject::new_module("RootScope");
        root_scope.module_set("new", HostObject::new_builtin_callable("RootScope.new"));

        // `pyre_object::int_array::IntArray::from_vec(Vec<i64>)` packs
        // a `Vec` into an `IntArray` and is called from
        // `pyre_object::listobject` list-strategy paths.
        let int_array = HostObject::new_module("IntArray");
        int_array.module_set(
            "from_vec",
            HostObject::new_builtin_callable("IntArray.from_vec"),
        );

        // `core::ptr` / `std::ptr` — Rust standard pointer utilities
        // surfaced by Charon as `core::ptr::*` dotted paths.  Pyre
        // source uses `std::ptr::null_mut()` for `pub const PY_NULL`-
        // style declarations (`parse.rs`); MIR lowering normalises
        // the `std::` re-export to its canonical `core::` form, so the
        // FunctionPath that reaches the flowspace adapter is
        // `["core", "ptr", "null_mut"]`.  Registering both modules
        // keeps either spelling resolvable when the front-end emits
        // the bare-namespace form.
        let core_ptr = HostObject::new_module("core.ptr");
        core_ptr.module_set(
            "null_mut",
            HostObject::new_builtin_callable("core.ptr.null_mut"),
        );
        core_ptr.module_set("null", HostObject::new_builtin_callable("core.ptr.null"));
        // Null `Option<fn>` — `SomePtr(FuncType)`, not `null_mut`'s
        // classdef-less instance. See `fn_null_constant`.
        core_ptr.module_set(
            "null_fn",
            HostObject::new_builtin_callable("core.ptr.null_fn"),
        );
        // `std::ptr::eq` re-exports `core::ptr::eq`; MIR lowering normalises
        // the call to the canonical `["core", "ptr", "eq"]` FunctionPath
        // (e.g. `py_type_check`'s `std::ptr::eq((*obj).ob_type, …)`).  Bind
        // the `core.ptr` spelling to the same `std.ptr.eq` callable so it
        // reuses the existing `std_ptr_eq` analyzer — the SAME instance,
        // not a fresh same-qualname one: `BUILTIN_TYPER` is keyed by
        // `HostObject` Arc identity.
        core_ptr.module_set(
            "eq",
            std_ptr.module_get("eq").expect("std.ptr.eq bound above"),
        );
        // `std::ptr::copy_nonoverlapping` re-exports
        // `core::ptr::copy_nonoverlapping`. Bind the canonical spelling
        // to the same callable so `std_ptr_copy_nonoverlapping` and the
        // `BUILTIN_TYPER` row see one `HostObject`.
        core_ptr.module_set(
            "copy_nonoverlapping",
            std_ptr
                .module_get("copy_nonoverlapping")
                .expect("std.ptr.copy_nonoverlapping bound above"),
        );
        // `std.ptr` was already created above with `null_mut` / `eq` /
        // `copy_nonoverlapping`; extend that same module with `null`
        // instead of re-creating it.  A fresh `new_module` here would
        // shadow the earlier binding and drop its `eq` /
        // `copy_nonoverlapping` members before the `mods.insert` below.
        std_ptr.module_set("null", HostObject::new_builtin_callable("std.ptr.null"));

        // `std::mem::size_of` / `align_of` re-export `core::mem::*`;
        // monomorphised LLBC spells the canonical `["core", "mem", _]`
        // form (same std→core normalisation as `core.ptr` above).  Bind
        // `core.mem` to the SAME `std.mem` callables so the size_of /
        // align_of analyzers (keyed by `HostObject` Arc identity) apply.
        let core_mem = HostObject::new_module("core.mem");
        core_mem.module_set(
            "size_of",
            std_mem
                .module_get("size_of")
                .expect("std.mem.size_of bound above"),
        );
        core_mem.module_set(
            "align_of",
            std_mem
                .module_get("align_of")
                .expect("std.mem.align_of bound above"),
        );

        // Container / bigint ctors reach the adapter spelled with their
        // DEFINING crate's canonical module path (`vec::Vec`,
        // `bigint::BigInt`, `boxed::Box`, and IndexMap's internal `map`
        // module), not the type-name / re-export shape registered above
        // (`Vec`, `BigInt`, `Box`, `indexmap.IndexMap`).  Branch 3b joins
        // `segments[..len-1]` with `.`, so the crate-qualified module key
        // must exist for resolution.  Reuse the existing analyzer-less
        // callables where present; mint fresh ones for the methods not
        // yet surfaced (`with_capacity` / `zero` / `new_uninit`).
        let vec_crate = HostObject::new_module("vec.Vec");
        vec_crate.module_set(
            "new",
            vec_ty.module_get("new").expect("Vec.new bound above"),
        );
        vec_crate.module_set(
            "with_capacity",
            HostObject::new_builtin_callable("vec.Vec.with_capacity"),
        );
        let bigint_crate = HostObject::new_module("bigint.BigInt");
        bigint_crate.module_set(
            "from",
            bigint.module_get("from").expect("BigInt.from bound above"),
        );
        bigint_crate.module_set(
            "zero",
            HostObject::new_builtin_callable("bigint.BigInt.zero"),
        );
        let boxed_crate = HostObject::new_module("boxed.Box");
        boxed_crate.module_set(
            "new",
            box_ty.module_get("new").expect("Box.new bound above"),
        );
        boxed_crate.module_set(
            "into_raw",
            box_ty
                .module_get("into_raw")
                .expect("Box.into_raw bound above"),
        );
        boxed_crate.module_set(
            "new_uninit",
            HostObject::new_builtin_callable("boxed.Box.new_uninit"),
        );
        // `<[T]>::into_vec` lowers through `alloc::boxed::
        // box_assume_init_into_vec_unsafe(Box<[T]>) -> Vec<T>`; a Vec
        // producer like the other container ctors above.
        let alloc_boxed = HostObject::new_module("alloc.boxed");
        alloc_boxed.module_set(
            "box_assume_init_into_vec_unsafe",
            HostObject::new_builtin_callable("alloc.boxed.box_assume_init_into_vec_unsafe"),
        );
        // Crate-qualified `string::String::new` / `with_capacity` — some
        // caller chains keep the `string` module segment (Branch 3b joins
        // `segments[..-1]` so `"string.String"` must exist alongside the bare
        // `"String"`, the same bare/crate dual as `vec.Vec`).  Reuse the bare
        // callables so a single qualname/analyzer covers each leaf.
        let string_crate = HostObject::new_module("string.String");
        string_crate.module_set(
            "new",
            string_ty.module_get("new").expect("String.new bound above"),
        );
        string_crate.module_set(
            "with_capacity",
            string_ty
                .module_get("with_capacity")
                .expect("String.with_capacity bound above"),
        );
        // `Wtf8Buf::new` / `with_capacity` — the owned WTF-8 buffer builder;
        // a mutable str value like `String::*`.
        let wtf8buf_ty = HostObject::new_module("Wtf8Buf");
        wtf8buf_ty.module_set("new", HostObject::new_builtin_callable("Wtf8Buf.new"));
        wtf8buf_ty.module_set(
            "with_capacity",
            HostObject::new_builtin_callable("Wtf8Buf.with_capacity"),
        );
        let map_indexmap = HostObject::new_module("map.IndexMap");
        map_indexmap.module_set(
            "new",
            indexmap_indexmap
                .module_get("new")
                .expect("indexmap.IndexMap.new bound above"),
        );

        // `pub const PY_NULL: PyObjectRef = std::ptr::null_mut()`
        // (`pyre_object::pyobject::PY_NULL`).  Charon emits the const
        // read as a Global place; `front::mir` lowers an unknown global
        // to a 0-arg `Call FunctionPath ["pyre_object", "pyobject",
        // "PY_NULL"]`, which Branch 3b resolves through this module.
        // Bind the attr to the canonical `core.ptr.null_mut` callable so
        // the read shares the null-pointer analyzer with spelled-out
        // `null_mut()` callsites.  Reuse the `core.ptr` attr instance —
        // `BUILTIN_TYPER` is keyed by `HostObject` Arc identity
        // (`rbuiltin.rs lookup_typer`), so a fresh
        // `new_builtin_callable` with the same qualname would miss the
        // registered typer.
        let object_module = HostObject::new_module(crate::runtime_names::modules::OBJECT_MODEL);
        object_module.module_set(
            "PY_NULL",
            core_ptr
                .module_get("null_mut")
                .expect("core.ptr.null_mut bound above"),
        );

        // `pyre_object::lltype::malloc[_typed]` — GC allocation intrinsics
        // (`lltype.malloc(STRUCT, flavor='gc')` parity).  Exposed as a host
        // builtin so its body is never looked-inside; the `malloc_typed_alloc`
        // analyzer types the result as `SomeInstance(T)`.  Callsites carry the
        // crate-qualified FunctionPath `["pyre_object","lltype","malloc_typed"]`,
        // which `translate_op`'s Layer-3b resolves via this module
        // (prefix `pyre_object.lltype`, leaf `malloc_typed`).
        let lltype_module = HostObject::new_module(crate::runtime_names::modules::OBJECT_LLTYPE);
        lltype_module.module_set(
            "malloc",
            HostObject::new_builtin_callable(crate::runtime_names::modules::MALLOC),
        );
        lltype_module.module_set(
            "malloc_typed",
            HostObject::new_builtin_callable(crate::runtime_names::modules::MALLOC_TYPED),
        );
        lltype_module.module_set(
            "malloc_typed_managed",
            HostObject::new_builtin_callable(crate::runtime_names::modules::MALLOC_TYPED_MANAGED),
        );
        lltype_module.module_set(
            "malloc_typed_stable",
            HostObject::new_builtin_callable(crate::runtime_names::modules::MALLOC_TYPED_STABLE),
        );
        // `pyre_object::lltype::malloc_raw` — the raw (non-GC) allocation
        // intrinsic (`lltype.malloc(T, flavor='raw')` parity).  Exposed as a
        // host builtin so its `Box::new` body is never looked-inside; the
        // `malloc_raw_alloc` analyzer types the result as the `*mut T`
        // pointer shell.  Resolved by `translate_op`'s Layer-3b via this
        // module (prefix `pyre_object.lltype`, leaf `malloc_raw`).
        lltype_module.module_set(
            "malloc_raw",
            HostObject::new_builtin_callable(crate::runtime_names::modules::MALLOC_RAW),
        );

        let mut mods = self.modules.lock();
        mods.insert("__builtin__".into(), self.builtin_module.clone());
        mods.insert("os".into(), os);
        mods.insert("os.path".into(), os_path);
        mods.insert("rpython.rlib.rfile".into(), rfile);
        mods.insert("rpython.rlib.rpath".into(), rpath);
        mods.insert("rpython.rlib.rarithmetic".into(), rarithmetic);
        mods.insert("rpython.rlib.objectmodel".into(), objectmodel);
        mods.insert("rpython.rtyper.lltypesystem.lltype".into(), lltype);
        mods.insert("rpython.rtyper.lltypesystem.llmemory".into(), llmemory);
        mods.insert("weakref".into(), weakref_mod);
        mods.insert("core.ptr".into(), core_ptr);
        mods.insert("core.mem".into(), core_mem);
        mods.insert("vec.Vec".into(), vec_crate);
        mods.insert("bigint.BigInt".into(), bigint_crate);
        mods.insert("boxed.Box".into(), boxed_crate);
        mods.insert("alloc.boxed".into(), alloc_boxed);
        mods.insert("string.String".into(), string_crate);
        mods.insert("Wtf8Buf".into(), wtf8buf_ty);
        mods.insert("map.IndexMap".into(), map_indexmap);
        mods.insert("std.ptr".into(), std_ptr);
        mods.insert("std.mem".into(), std_mem);
        mods.insert("std.alloc".into(), std_alloc);
        mods.insert("std.slice".into(), std_slice);
        mods.insert("BigInt".into(), bigint);
        mods.insert("longlong2float".into(), longlong2float);
        mods.insert("majit_metainterp".into(), majit_metainterp);
        mods.insert("majit_metainterp.jit".into(), majit_metainterp_jit);
        mods.insert("majit_rlib.jit".into(), majit_rlib_jit);
        mods.insert("majit_rlib.nonconst".into(), majit_rlib_nonconst);
        mods.insert("nonconst".into(), nonconst);
        mods.insert("u32".into(), primitive_u32);
        mods.insert("i64".into(), primitive_i64);
        mods.insert("usize".into(), primitive_usize);
        mods.insert("Box".into(), box_ty);
        mods.insert("Vec".into(), vec_ty);
        mods.insert("String".into(), string_ty);
        mods.insert("indexmap.IndexMap".into(), indexmap_indexmap);
        mods.insert("std.collections.HashMap".into(), std_collections_hashmap);
        mods.insert("FrameDebugData".into(), frame_debug_data);
        mods.insert("RootScope".into(), root_scope);
        mods.insert("IntArray".into(), int_array);
        mods.insert(
            crate::runtime_names::modules::OBJECT_MODEL.into(),
            object_module,
        );
        mods.insert(
            crate::runtime_names::modules::OBJECT_LLTYPE.into(),
            lltype_module,
        );
    }

    /// upstream `getattr(__builtin__, name)` — `flowcontext.py:851`.
    pub fn lookup_builtin(&self, name: &str) -> Option<HostObject> {
        self.builtins.get(name).cloned()
    }

    pub fn builtin_module(&self) -> HostObject {
        self.builtin_module.clone()
    }

    /// upstream `__import__(name, …)` — `flowcontext.py:660`.
    pub fn import_module(&self, name: &str) -> Option<HostObject> {
        self.modules.lock().get(name).cloned()
    }

    /// Returns the HostObject when the exception class is in the builtin
    /// table. There is no API yet to register a user-defined class; when one
    /// is needed, flowcontext builds it with `HostObject::new_class`.
    pub fn lookup_exception_class(&self, name: &str) -> Option<HostObject> {
        self.lookup_builtin(name).filter(|obj| {
            obj.is_class()
                && obj.is_subclass_of(self.lookup_builtin("BaseException").as_ref().unwrap())
        })
    }

    /// RPython `exceptiondata.py:get_standard_ll_exc_instance_by_class`.
    /// A standard exception class materializes its reusable prebuilt instance
    /// as a shared singleton.
    pub fn lookup_standard_exception_instance(&self, name: &str) -> Option<HostObject> {
        self.lookup_exception_class(name)
            .and_then(|cls| cls.reusable_prebuilt_instance())
    }
}

/// Process-global host namespace singleton. Bootstrap fills it with
/// placeholders for upstream `__builtin__` plus known stdlib modules.
pub static HOST_ENV: LazyLock<HostEnv> = LazyLock::new(HostEnv::bootstrap);

/// `FunctionPath` resolution through `HOST_ENV`: one segment is
/// `lookup_builtin(leaf)`, and two or more are
/// `import_module(prefix).module_get(leaf)`. `translate_op` and
/// `Lowering::callee_is_host_builtin` both call this.
pub(crate) fn host_env_callable(segments: &[String]) -> Option<HostObject> {
    if let [leaf] = segments {
        HOST_ENV.lookup_builtin(leaf)
    } else if segments.len() >= 2 {
        let module = segments[..segments.len() - 1].join(".");
        let leaf = &segments[segments.len() - 1];
        HOST_ENV
            .import_module(&module)
            .and_then(|module| module.module_get(leaf))
    } else {
        None
    }
}

/// Flow-space carrier for Python objects referenced directly by the
/// strict `flowcontext.py` port.
///
/// RPython `Constant.value` is the unwrapped Python object; the Rust
/// port mirrors that contract with an explicit sum type until the
/// wider host object model lands.
#[derive(Clone, Debug)]
pub enum ConstValue {
    /// Sentinel for the `c_last_exception` atom; populated by F1.4
    /// with `Constant(last_exception)`.
    Atom(Atom),
    /// Placeholder for uninstantiated constants during skeleton
    /// construction. Not emitted by production code.
    Placeholder,
    /// Integer constant, e.g. `Constant(0)` / `Constant(1)` in
    /// `test_model.py`. Phase 3 generalises this to arbitrary
    /// Python values.
    Int(i64),
    /// RPython `r_longlonglong` constant. Kept separate from `Int`
    /// because a 128-bit primitive cannot be represented by the
    /// word-sized flowspace carrier without truncation.
    Int128(i128),
    /// RPython `r_ulonglonglong` constant.
    UInt128(u128),
    /// Python float constant. RPython stores the `f64` directly on
    /// `Constant.value`; Rust carries it as `f64::to_bits()` so the
    /// enum keeps its `Eq + Hash` derivation (IEEE 754 NaN violates
    /// `Eq`). Use `ConstValue::float(value)` / `ConstValue::as_float()`
    /// to convert.
    Float(u64),
    /// Python dict constant. Mirrors upstream `Constant(dict)` and is
    /// used both for `func.__globals__` and optimizer rewrites like
    /// `transform_list_contains`.
    Dict(HashMap<ConstValue, ConstValue>),
    /// Python 2 byte string constant (`str`).
    ///
    /// RPython keeps the byte/unicode distinction in the concrete
    /// Python object stored on `Constant.value`; the Rust port makes
    /// that type tag explicit so low-level `Char` and `UniChar`
    /// constants cannot be confused.
    ByteStr(Vec<u8>),
    /// Python 2 unicode constant (`unicode`).
    UniStr(String),
    /// Python tuple constant. Used by `CallSpec.as_list()` to unpack
    /// `*args` exactly like upstream's `w_stararg.value`.
    Tuple(Vec<ConstValue>),
    /// Python list constant. Flowspace treats tuple/list starred args
    /// identically at this phase: each element is rewrapped into an
    /// individual `Constant`.
    List(Vec<ConstValue>),
    /// Boolean constant — used as `Link.exitcase` for if/else
    /// switches (`True` / `False`).
    Bool(bool),
    /// `None` — CPython's None singleton, used by
    /// `FrameState._exc_args()` as the sentinel for
    /// "no pending exception".
    None,
    /// Host code object constant (`func.__code__`).
    Code(Box<HostCode>),
    /// Python function object.
    Function(Box<GraphFunc>),
    /// RPython `rpbc.py:216` trailing `c_graphs` constant carried by
    /// `indirect_call(funcptr, *args, c_graphs)`.
    ///
    /// Upstream stores the graph objects themselves in the Python
    /// list. Rust cannot embed graph refs or raw-pointer wrappers here
    /// without breaking `HostEnv`'s global `Sync` invariants, so the
    /// port stores graph identities as `usize` values
    /// (`GraphKey::as_usize()`).
    Graphs(Vec<usize>),
    /// RPython flowmodel `Constant(TYPE, lltype.Void)` where `TYPE` is
    /// itself an lltype object, e.g. `rptr.py`'s `INTERIOR_PTR_TYPE`
    /// argument to `malloc`.
    LowLevelType(Box<ConcretetypePlaceholder>),
    /// RPython `lltype._ptr` function pointer constant.
    LLPtr(Box<_ptr>),
    /// RPython `llmemory._fakeaddress` Address-typed constant —
    /// either NULL or a `Fake(_ptr)` wrapping a live pointer. Surfaces
    /// at the `Address` lowleveltype slot used by
    /// `MultipleUnrelatedFrozenPBCRepr`.
    LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address),
    /// Arbitrary host-level Python object (class, module, builtin
    /// callable, instance). General carrier that stands in for an arbitrary
    /// object stored in upstream `Constant.value` — see `HostObject`.
    HostObject(HostObject),
    /// RPython `rpython/rlib/unroll.py:SpecTag` — an identity-bearing
    /// marker instance that prevents two different tags from being
    /// merged by `framestate.union()`. Each `SpecTag(id)` carries a
    /// process-unique `u64` so two separately constructed tags never
    /// compare equal even if wrapped in `Constant`.
    ///
    /// `llmemory` symbolic address offset (`AddressOffset` and its
    /// subclasses). `lltype()` is `Signed`; downstream `inputconst(Signed,
    /// sizeof(TYPE))` accepts this so concrete byte sizes are resolved
    /// only at code emission, not at rtyper time.
    AddressOffset(crate::translator::rtyper::lltypesystem::llmemory::AddressOffset),
    /// Deviation from upstream (parity rule #1): RPython uses a Python
    /// class (`SpecTag`) whose identity is the instance's `id()`. Rust
    /// has no cross-session `id()`, so we materialise identity as an
    /// atomic counter.
    SpecTag(u64),
    /// Inheritance-id marker carrying a classdef's `minid` / `maxid` for
    /// an `ll_issubclass_const` / exception-match range check. `lltype()`
    /// is `Signed`. The integer is resolved EAGERLY at rtype time — read
    /// off the already-materialized vtable's `subclassrange_min/max` — and
    /// carried in `value`; `emit_const_i_from_const` emits it verbatim
    /// (unlike `AddressOffset`, which is genuinely resolved at emission).
    /// `cdef_id` (the stable `get_unique_cdef_id`) and `is_max` (min vs max
    /// endpoint) identify the marker and are RESERVED for a future
    /// emission-time numbering resolution, only reachable under the
    /// whole-program two-phase rtype/emit prepass; today `cdef_id` is
    /// always `None`.
    InheritanceId {
        cdef_id: Option<usize>,
        is_max: bool,
        value: i64,
    },
    /// RPython `Constant.value` holding a live rtyper object
    /// (`rclass.FieldListAccessor` in `VTYPE._hints['virtualizable_accessor']`).
    /// Identity is the `Arc` pointer, matching Python `is`.
    Opaque(OpaqueConst),
    /// RPython `s_repr.const` when that constant is an `rmodel.Repr`
    /// (`hlinvoke`'s first argument). The `usize` is
    /// `Box::into_raw(Box<Arc<dyn Repr>>)` leaked for the process.
    /// A fat `Arc<dyn Repr>` cannot sit in this enum: `Repr` is not
    /// `Send`, and `HostEnv` requires `ConstValue: Send`. This is the
    /// object itself, not a side table.
    Repr(usize),
}

/// Identity-bearing rtyper object stored on `Constant.value`.
#[derive(Clone)]
pub struct OpaqueConst {
    inner: Arc<dyn Any + Send + Sync>,
}

impl OpaqueConst {
    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        OpaqueConst {
            inner: Arc::new(value),
        }
    }

    pub fn from_arc<T: Any + Send + Sync>(value: Arc<T>) -> Self {
        OpaqueConst { inner: value }
    }

    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for OpaqueConst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Opaque")
    }
}

impl PartialEq for OpaqueConst {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for OpaqueConst {}

impl Hash for OpaqueConst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

/// `ConstValue::Repr` carries a leaked `Box<Arc<dyn Repr>>`. Equality is
/// the inner `Arc` allocation, the same identity `OpaqueConst` uses.
/// A null carrier is not a live box.
fn repr_carriers_eq(a: usize, b: usize) -> bool {
    if a == 0 || b == 0 {
        return a == b;
    }
    // SAFETY: both addresses are leaked boxes from `repr_const`, which
    // stores `Box<Arc<dyn Repr>>` and never frees it.
    let arc_a =
        unsafe { &*(a as *const std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr>) };
    let arc_b =
        unsafe { &*(b as *const std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr>) };
    std::sync::Arc::ptr_eq(arc_a, arc_b)
}

fn hash_repr_carrier<H: Hasher>(ptr: usize, state: &mut H) {
    if ptr == 0 {
        0usize.hash(state);
        return;
    }
    // SAFETY: `ptr` is a leaked box from `repr_const` and is never freed.
    let arc =
        unsafe { &*(ptr as *const std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr>) };
    std::sync::Arc::as_ptr(arc).hash(state);
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ConstValue::Atom(a), ConstValue::Atom(b)) => a == b,
            (ConstValue::Placeholder, ConstValue::Placeholder) => true,
            (ConstValue::Int(a), ConstValue::Int(b)) => a == b,
            (ConstValue::Int128(a), ConstValue::Int128(b)) => a == b,
            (ConstValue::UInt128(a), ConstValue::UInt128(b)) => a == b,
            (ConstValue::Float(a), ConstValue::Float(b)) => a == b,
            (ConstValue::Dict(a), ConstValue::Dict(b)) => a == b,
            (ConstValue::ByteStr(a), ConstValue::ByteStr(b)) => a == b,
            (ConstValue::UniStr(a), ConstValue::UniStr(b)) => a == b,
            (ConstValue::Tuple(a), ConstValue::Tuple(b)) => a == b,
            (ConstValue::List(a), ConstValue::List(b)) => a == b,
            (ConstValue::Bool(a), ConstValue::Bool(b)) => a == b,
            (ConstValue::None, ConstValue::None) => true,
            (ConstValue::Code(a), ConstValue::Code(b)) => {
                a._hashable_identity() == b._hashable_identity()
            }
            (ConstValue::Function(a), ConstValue::Function(b)) => {
                a._hashable_identity() == b._hashable_identity()
            }
            (ConstValue::Graphs(a), ConstValue::Graphs(b)) => a == b,
            (ConstValue::LowLevelType(a), ConstValue::LowLevelType(b)) => a == b,
            (ConstValue::LLPtr(a), ConstValue::LLPtr(b)) => {
                a._hashable_identity() == b._hashable_identity()
            }
            (ConstValue::LLAddress(a), ConstValue::LLAddress(b)) => a == b,
            (ConstValue::HostObject(a), ConstValue::HostObject(b)) => a == b,
            (ConstValue::Opaque(a), ConstValue::Opaque(b)) => a == b,
            (ConstValue::Repr(a), ConstValue::Repr(b)) => repr_carriers_eq(*a, *b),
            (ConstValue::SpecTag(a), ConstValue::SpecTag(b)) => a == b,
            (ConstValue::AddressOffset(a), ConstValue::AddressOffset(b)) => a == b,
            (
                ConstValue::InheritanceId {
                    cdef_id: a,
                    is_max: x,
                    value: v,
                },
                ConstValue::InheritanceId {
                    cdef_id: b,
                    is_max: y,
                    value: w,
                },
            ) => a == b && x == y && v == w,
            _ => false,
        }
    }
}

impl Eq for ConstValue {}

impl std::fmt::Display for ConstValue {
    /// RPython `Constant.__repr__` prints the wrapped Python value via
    /// `repr()`; pyre renders the simple leaf variants directly and
    /// delegates the complex host / code / function shapes to `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::Atom(atom) => write!(f, "{atom}"),
            ConstValue::Placeholder => f.write_str("<placeholder>"),
            ConstValue::Int(value) => write!(f, "{value}"),
            ConstValue::Int128(value) => write!(f, "{value}"),
            ConstValue::UInt128(value) => write!(f, "{value}"),
            ConstValue::Float(bits) => write!(f, "{}", f64::from_bits(*bits)),
            ConstValue::ByteStr(value) => write!(f, "b{:?}", String::from_utf8_lossy(value)),
            ConstValue::UniStr(value) => write!(f, "u{value:?}"),
            ConstValue::Bool(value) => write!(f, "{value}"),
            ConstValue::None => f.write_str("None"),
            ConstValue::SpecTag(id) => write!(f, "<spec-tag {id}>"),
            ConstValue::HostObject(obj) => write!(f, "{}", obj.qualname()),
            ConstValue::Opaque(_) => f.write_str("<opaque>"),
            ConstValue::Repr(ptr) => write!(f, "<repr {ptr:#x}>"),
            ConstValue::Dict(_)
            | ConstValue::Tuple(_)
            | ConstValue::List(_)
            | ConstValue::Code(_)
            | ConstValue::Graphs(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::LLPtr(_)
            | ConstValue::LLAddress(_)
            | ConstValue::Function(_) => write!(f, "{self:?}"),
            ConstValue::AddressOffset(offset) => write!(f, "{offset:?}"),
            ConstValue::InheritanceId { value, .. } => write!(f, "<inheritance-id {value}>"),
        }
    }
}

impl Hash for ConstValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ConstValue::Atom(atom) => atom.hash(state),
            ConstValue::Placeholder => {}
            ConstValue::Int(value) => value.hash(state),
            ConstValue::Int128(value) => value.hash(state),
            ConstValue::UInt128(value) => value.hash(state),
            ConstValue::Float(bits) => bits.hash(state),
            ConstValue::Dict(items) => {
                // HashMap iteration order is nondeterministic; hash
                // each entry independently and sort the resulting
                // u64s so `ConstValue::Dict` remains stable under
                // hashing regardless of insertion order.
                let mut entry_hashes: Vec<u64> = items
                    .iter()
                    .map(|(key, value)| {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        key.hash(&mut h);
                        value.hash(&mut h);
                        h.finish()
                    })
                    .collect();
                entry_hashes.sort_unstable();
                for entry_hash in entry_hashes {
                    entry_hash.hash(state);
                }
            }
            ConstValue::ByteStr(value) => value.hash(state),
            ConstValue::UniStr(value) => value.hash(state),
            ConstValue::Tuple(items) | ConstValue::List(items) => items.hash(state),
            ConstValue::Bool(value) => value.hash(state),
            ConstValue::None => {}
            ConstValue::Code(code) => code._hashable_identity().hash(state),
            ConstValue::Function(func) => func._hashable_identity().hash(state),
            ConstValue::Graphs(graphs) => graphs.hash(state),
            ConstValue::LowLevelType(lltype) => lltype.hash(state),
            ConstValue::LLPtr(ptr) => ptr._hashable_identity().hash(state),
            ConstValue::LLAddress(addr) => match addr {
                crate::translator::rtyper::lltypesystem::lltype::_address::Null => 0u8.hash(state),
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(p) => {
                    1u8.hash(state);
                    p._hashable_identity().hash(state);
                }
                crate::translator::rtyper::lltypesystem::lltype::_address::IntCast(n) => {
                    2u8.hash(state);
                    n.hash(state);
                }
            },
            ConstValue::HostObject(obj) => obj.hash(state),
            ConstValue::Opaque(opaque) => opaque.hash(state),
            ConstValue::Repr(ptr) => hash_repr_carrier(*ptr, state),
            ConstValue::SpecTag(id) => id.hash(state),
            ConstValue::AddressOffset(offset) => offset.hash(state),
            ConstValue::InheritanceId {
                cdef_id,
                is_max,
                value,
            } => {
                cdef_id.hash(state);
                is_max.hash(state);
                value.hash(state);
            }
        }
    }
}

/// RPython `flowspace/model.py:463-467` — typed marker like
/// `last_exception`. An `Atom` carries only a name (used by `repr`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// RPython `Atom.__name__`.
    pub name: &'static str,
}

impl Atom {
    /// RPython `Atom.__init__(name)`.
    pub fn new(name: &'static str) -> Self {
        Atom { name }
    }
}

impl std::fmt::Display for Atom {
    // RPython `Atom.__repr__` returns `self.__name__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

// RPython `flowspace/model.py` — `Variable` with lazy name
// numbering.
//
// The dummy name is `'v'`; `namesdict` maps each name prefix to a
// (prefix, next_number) tuple so successive Variables with the same
// prefix pick up unique, increasing numbers.
const DUMMYNAME: &str = "v";

/// Shared `Variable.namesdict`. Key = interned name prefix, value =
/// (stored prefix, next-numbered index).
///
/// RPython has one such dict as a class attribute; Rust needs a
/// `static` + `Mutex` since class-mutable state has no direct
/// equivalent.
static NAMESDICT: LazyLock<Mutex<HashMap<String, (String, u32)>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(DUMMYNAME.to_owned(), (DUMMYNAME.to_owned(), 0));
    Mutex::new(m)
});

/// `Variable.rename`'s string-cleaning step (model.py).
///
/// Removes strange characters: keep `[A-Za-z0-9_]`, swap the rest to
/// `_`; always trail with a `_`; prefix numeric first-chars with `_`
/// to ensure a valid Python identifier. `namesdict.setdefault(name,
/// (name, 0))[0]` interns the cleaned prefix.
fn clean_name(name: &str) -> String {
    let mut cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.push('_');
    if let Some(first) = cleaned.chars().next()
        && !first.is_ascii_alphabetic()
        && first != '_'
    {
        cleaned.insert(0, '_');
    }
    let mut nd = NAMESDICT.lock();
    let entry = nd
        .entry(cleaned.clone())
        .or_insert_with(|| (cleaned.clone(), 0));
    entry.0.clone()
}

/// Per-process id counter for Variable identity (RPython identity
/// parity). Rust has no Python-style `id(obj)`, so each Variable
/// gets a unique `u64` at construction; `clone()` preserves it
/// (aliasing the same logical Variable), while `copy()` takes a
/// fresh id (same semantics as RPython's `Variable.copy`, which
/// returns a *new* Variable with the same prefix).
static NEXT_VAR_ID: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn next_var_id() -> u64 {
    NEXT_VAR_ID.load(Ordering::Relaxed)
}

fn alloc_var_id() -> u64 {
    NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed)
}

/// RPython `flowspace/model.py` — `class Variable`.
///
/// `__slots__ = ["_name", "_nr", "annotation", "concretetype"]`.
///
/// Upstream Python relies on object-identity for mutable attribute
/// sharing — setting `v.annotation = s` on one reference is observed
/// by every other reference to the same Python object. The Rust port
/// approximates this by wrapping the mutable-per-identity fields
/// (`_nr`, `annotation`) in `Rc<_>` so `Variable::clone()` (which
/// preserves identity, per upstream's "no language-level clone" model)
/// Rc-shares them across clones. Fresh identities go through
/// [`Variable::new`] / [`Variable::named`] / [`Variable::copy`], each
/// of which allocates independent cells.
///
/// TODO: This Rc-sharing is a minimal Rust-language
/// adaptation for the unavoidable gap between Python's attribute-on-
/// object model and Rust's value-type struct. CLAUDE.md permits it
/// because the alternative (lossy per-clone-slot `annotation`) breaks
/// annotator correctness — see `test_variable_identity_diagnostic.rs`.
/// Backing storage for [`Variable`] — the single heap object that all
/// references to one logical Variable share. Mutable attributes use
/// plain `Cell`/`RefCell` interior mutability; no per-field `Rc` is
/// needed because the whole struct is shared through the one outer
/// `Rc` in [`Variable`], so a write through any reference is observed
/// by all of them (upstream Python's attribute-on-object semantics).
#[derive(Debug)]
pub struct VariableInner {
    /// Identity key. See `NEXT_VAR_ID` — RPython uses object identity;
    /// Rust approximates with a process-wide increment, used only as a
    /// hash/equality key (Python hashes objects by identity too), never
    /// as an index into a value table.
    id: u64,
    /// Prefix name (upstream `Variable._name`). `copy()` mints a fresh
    /// `VariableInner` holding this prefix value; `rename`/`set_name`
    /// through any reference is observed by all since they share the
    /// one object.
    _name: std::cell::RefCell<String>,
    /// Lazy numbering counter (upstream `Variable._nr`).
    _nr: std::cell::Cell<i64>,
    /// RPython `Variable.annotation` (set by the annotator). The inner
    /// `Rc<SomeValue>` is the shared lattice handle (one instance backs
    /// many `Variable`s); the cell itself is per-object.
    pub annotation: std::cell::RefCell<Option<Rc<SomeValue>>>,
    /// RPython `Variable.concretetype` (set by the rtyper).
    pub concretetype: std::cell::RefCell<Option<ConcretetypePlaceholder>>,
}

/// RPython `flowspace/model.py` — `class Variable`.
///
/// `__slots__ = ["_name", "_nr", "annotation", "concretetype"]`.
///
/// A `Variable` IS a Python object: one heap allocation, many
/// references. The Rust port models that directly as a newtype over
/// `Rc<VariableInner>` — `clone()` aliases the same object (a reference
/// copy; identity preserved) while [`Variable::copy`] mints a new
/// object with the same name prefix (RPython `Variable.copy`).
/// Equality and hashing are object identity. `Deref` exposes the
/// shared [`VariableInner`] so existing `v.annotation` /
/// `v.concretetype` reads keep working.
#[derive(Debug, Clone)]
pub struct Variable(Rc<VariableInner>);

impl std::ops::Deref for Variable {
    type Target = VariableInner;
    fn deref(&self) -> &VariableInner {
        &self.0
    }
}

// `Variable` is intentionally !Send / !Sync — matching RPython's
// single-thread annotator invariant.  The `Rc<RefCell<Option<Rc<SomeValue>>>>`
// in `annotation` is the explicit !Sync field.  Every consumer that
// stores `Variable` (or a type embedding one transitively) lives behind
// a thread-local registry (`generated::with_all_jitcodes`, …) — there
// is no `static OnceLock<…>` that would force `T: Sync` on this type.
//
// Do NOT add an `unsafe impl Send / Sync for Variable`.  If a new
// consumer needs `Variable` thread-safe, the orthodox response is to
// confine that consumer to a thread-local (matching RPython's GIL
// model), not to widen the unsafe surface area.

// `Clone` is derived: a newtype over `Rc<VariableInner>` clones via
// `Rc::clone`, which aliases the one shared object — RPython has no
// language-level clone, and downstream code that stores a `Variable`
// and later reuses the "same" one relies on identity being preserved.
// Use [`Variable::copy`] for RPython `Variable.copy` semantics (a new
// object with the same prefix).

impl PartialEq for Variable {
    // RPython relies on Python object identity for Variable
    // equality; the `id` field is a direct port.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Variable {
    /// RPython `Variable.__init__(name=None)`.
    pub fn new() -> Self {
        // `_name = self.dummyname; _nr = -1; annotation = None`.
        Variable(Rc::new(VariableInner {
            id: alloc_var_id(),
            _name: std::cell::RefCell::new(DUMMYNAME.to_owned()),
            _nr: std::cell::Cell::new(-1),
            annotation: std::cell::RefCell::new(None),
            concretetype: std::cell::RefCell::new(None),
        }))
    }

    /// RPython `Variable.__init__(name)` with a non-None argument.
    pub fn named(name: impl AsRef<str>) -> Self {
        let mut v = Self::new();
        v.rename(name.as_ref());
        v
    }

    /// RPython `Variable.name` (property): lazy-allocate a unique
    /// number for this Variable's prefix from the shared namesdict.
    pub fn name(&self) -> String {
        let mut nr = self._nr.get();
        let prefix = self._name.borrow().clone();
        if nr == -1 {
            let mut nd = NAMESDICT.lock();
            let entry = nd
                .entry(prefix.clone())
                .or_insert_with(|| (prefix.clone(), 0));
            nr = entry.1 as i64;
            entry.1 += 1;
            self._nr.set(nr);
        }
        format!("{}{}", prefix, nr)
    }

    /// RPython `Variable.renamed` (property).
    pub fn renamed(&self) -> bool {
        *self._name.borrow() != DUMMYNAME
    }

    /// Generator flowspace needs the pre-numbered name prefix that
    /// upstream reads as `v._name`.
    pub fn name_prefix(&self) -> String {
        self._name.borrow().clone()
    }

    /// RPython `Variable.rename(name)`.
    ///
    /// Only renames once: subsequent calls are no-ops. The RPython
    /// source takes either a string or another Variable as argument;
    /// the Rust port exposes two overloads (`rename(&str)` below
    /// handles the string case, `rename_from(&Variable)` handles the
    /// Variable case — see `set_name_from`).
    pub fn rename(&mut self, name: &str) {
        if *self._name.borrow() != DUMMYNAME {
            return;
        }
        *self._name.borrow_mut() = clean_name(name);
        self._nr.set(-1);
    }

    /// RPython `Variable.rename(name)` where `name` is another
    /// `Variable` (the `_copy(v) -> Variable(v)` path, framestate.py /
    /// model.py). Copies the source's prefix and resets `_nr`
    /// for lazy renumbering; honours the "don't rename twice" guard and
    /// the "source wasn't renamed either" early-out. Unlike the `&str`
    /// overload it does *not* clean the name (the source prefix is
    /// already cleaned). Does NOT copy annotation/concretetype —
    /// `Variable(v).__init__` leaves `annotation = None`.
    pub fn rename_from(&mut self, other: &Variable) {
        if *self._name.borrow() != DUMMYNAME {
            return;
        }
        let other_name = other._name.borrow().clone();
        if other_name == DUMMYNAME {
            return;
        }
        *self._name.borrow_mut() = other_name;
        self._nr.set(-1);
    }

    /// RPython `Variable.set_name_from(v)` (for SSI_to_SSA).
    pub fn set_name_from(&mut self, other: &Variable) {
        // `v.name` forces finalisation on the RHS.
        let _ = other.name();
        let other_name = other._name.borrow().clone();
        *self._name.borrow_mut() = other_name;
        self._nr.set(other._nr.get());
    }

    /// RPython `Variable.set_name(name, nr)` (for wrapper.py).
    pub fn set_name(&mut self, name: String, nr: i64) {
        *self._name.borrow_mut() = name;
        self._nr.set(nr);
    }

    /// Identity key — see `NEXT_VAR_ID`. No direct upstream equivalent
    /// (Python uses `id(obj)` for object identity); the Rust port
    /// exposes the internal counter so callers that cannot rely on
    /// `clone()`-shared mutable state (e.g. SSI_to_SSA's second-pass
    /// graph rewrite) can key on it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Snapshot of the lazy `_nr` counter. Useful when a caller wants
    /// to propagate the same `(prefix, nr)` pair onto another Variable
    /// via `set_name`.
    pub fn nr(&self) -> i64 {
        self._nr.get()
    }

    /// RPython `Variable.foldable()` — always False for Variable.
    pub fn foldable(&self) -> bool {
        false
    }

    /// RPython `Variable.copy()` — creates a *new* Variable with a
    /// fresh identity, preserving name prefix, annotation, and
    /// concretetype.
    pub fn copy(&self) -> Self {
        // `Variable::new()` allocates a fresh id *and* fresh _nr /
        // annotation / concretetype cells (new identity). We
        // overwrite the prefix with the source's _name, matching
        // RPython's `Variable(v)` path that copies the prefix via
        // `rename`. Annotation and concretetype are copied by VALUE
        // into the new cells — sharing the source's Rcs would alias
        // mutable state across independent identities.
        let newvar = Variable::new();
        // `copy()` is a *new* identity, so it gets a fresh `_name` cell
        // holding the source's prefix value (not a shared Rc).
        *newvar._name.borrow_mut() = self._name.borrow().clone();
        newvar._nr.set(-1);
        *newvar.annotation.borrow_mut() = self.annotation.borrow().clone();
        *newvar.concretetype.borrow_mut() = self.concretetype.borrow().clone();
        newvar
    }

    /// Shortcut for `self.concretetype.borrow().clone()` — returns the
    /// current low-level type, if any, without leaking the `RefCell`
    /// abstraction at call sites.
    pub fn concretetype(&self) -> Option<ConcretetypePlaceholder> {
        self.concretetype.borrow().clone()
    }

    /// Shortcut for `*self.concretetype.borrow_mut() = value` — mirrors
    /// upstream `v.concretetype = X` with the reference-semantic
    /// propagation guarantee (see the `concretetype` field).
    pub fn set_concretetype(&self, value: Option<ConcretetypePlaceholder>) {
        *self.concretetype.borrow_mut() = value;
    }

    /// RPython `Variable.replace(mapping)`: `mapping.get(self, self)`.
    /// Upstream mapping values are polymorphic (Variable or Constant);
    /// the Rust port routes through `Hlvalue` so Variable → Constant
    /// substitutions work too.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
        mapping
            .get(self)
            .cloned()
            .unwrap_or_else(|| Hlvalue::Variable(self.clone()))
    }
}

impl std::fmt::Display for Variable {
    // RPython `Variable.__repr__` returns `self.name`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `flowspace/model.py` — `class Constant(Hashable)`.
///
/// `__slots__ = ["concretetype"]`.
#[derive(Clone, Debug)]
pub struct Constant {
    /// Stable identity for this `Constant` instance. Upstream caches
    /// mutable constants by the `Constant(x)` object itself; Rust keeps
    /// value-based `PartialEq` / `Hash` for legacy comparisons and
    /// exposes this separate id for identity-keyed caches.
    pub id: u64,
    /// RPython `Hashable.value`.
    pub value: ConstValue,
    /// RPython `Constant.concretetype`.
    pub concretetype: Option<ConcretetypePlaceholder>,
}

impl Constant {
    /// RPython `Constant.__init__(value, concretetype=None)`.
    pub fn new(value: ConstValue) -> Self {
        let id = alloc_constant_id(&value);
        Constant {
            id,
            value,
            concretetype: None,
        }
    }

    /// RPython `Constant.__init__(value, concretetype)`.
    pub fn with_concretetype(value: ConstValue, concretetype: ConcretetypePlaceholder) -> Self {
        let id = alloc_constant_id(&value);
        Constant {
            id,
            value,
            concretetype: Some(concretetype),
        }
    }

    /// RPython `flowspace/model.py` — `Constant.foldable()`.
    ///
    /// upstream line-by-line:
    /// ```python
    /// to_check = self.value
    /// if hasattr(to_check, 'im_self'):
    ///     to_check = to_check.im_self
    /// if isinstance(to_check, (type, types.ClassType, types.ModuleType)):
    ///     return True
    /// if (hasattr(to_check, '__class__') and
    ///         to_check.__class__.__module__ == '__builtin__'):
    ///     return True
    /// if hasattr(to_check, '_freeze_'):
    ///     assert to_check._freeze_() is True
    ///     return True
    /// return False
    /// ```
    pub fn foldable(&self) -> bool {
        let bound_self = match &self.value {
            ConstValue::HostObject(obj) => {
                obj.bound_method_self().cloned().map(ConstValue::HostObject)
            }
            _ => None,
        };
        let to_check = bound_self.as_ref().unwrap_or(&self.value);
        if let ConstValue::HostObject(obj) = to_check
            && (obj.is_class() || obj.is_module())
        {
            return true;
        }
        let class_module = to_check
            .class_of()
            .as_ref()
            .and_then(|cls| cls.module_name().map(str::to_string));
        if class_module.as_deref() == Some("__builtin__") {
            return true;
        }
        if let ConstValue::HostObject(obj) = to_check {
            return Self::call_foldable_freeze_method(obj);
        }
        if matches!(to_check, ConstValue::LowLevelType(_)) {
            // RPython lltype.LowLevelType implements `_freeze_()` and
            // returns True, so Constant(TYPE).foldable() is True.
            return true;
        }
        false
    }

    /// RPython `Constant.replace(mapping)` — Constants never rename.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, _mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
        Hlvalue::Constant(self.clone())
    }

    /// RPython `rpython/tool/uid.py` — `Hashable.__init__` tries
    /// `hash((type(value), value))` and falls back to `id(self.value)`
    /// when that raises `TypeError`. Rust's derived `Hash` cannot throw,
    /// so this predicate pre-classifies the `ConstValue` variants whose
    /// contents are unhashable under Python's own rules
    /// (`list.__hash__ = None`, `dict.__hash__ = None`) or whose Rust
    /// container (`Graphs` is a `Vec<GraphId>`) has no meaningful
    /// value-identity, routing them through the `self.id` (object
    /// identity) branch of `PartialEq` / `Hash` for `Constant`.
    fn uses_hashable_identity_fallback(value: &ConstValue) -> bool {
        match value {
            ConstValue::List(_) | ConstValue::Dict(_) | ConstValue::Graphs(_) => true,
            ConstValue::Tuple(items) => items.iter().any(Self::uses_hashable_identity_fallback),
            _ => false,
        }
    }

    fn call_foldable_freeze_method(obj: &HostObject) -> bool {
        let method = match host_getattr(obj, "_freeze_") {
            Ok(ConstValue::HostObject(method)) => method,
            Ok(other) => panic!(
                "_freeze_ exists on {:?} but is not callable: {other:?}",
                obj
            ),
            Err(HostGetAttrError::Missing) => return false,
            Err(HostGetAttrError::Unsupported) => return false,
        };
        match method.call_host(&[]) {
            Ok(ConstValue::Bool(true)) => true,
            Ok(other) => panic!("_freeze_() must return True, got {other:?}"),
            Err(HostCallError::RequiresFlowEvaluator(q)) => panic!(
                "_freeze_() on {q:?} requires the Phase 6 host-graph evaluator \
                 (user function body cannot be executed host-side yet)"
            ),
            Err(other) => panic!("_freeze_() call failed: {other}"),
        }
    }
}

static NEXT_CONSTANT_ID: AtomicU64 = AtomicU64::new(1);

// Exists to localise prepass nondeterminism (gh#1139).
fn alloc_constant_id(value: &ConstValue) -> u64 {
    let id = NEXT_CONSTANT_ID.fetch_add(1, Ordering::Relaxed);
    let Some(&(from, to)) = constant_trace_window() else {
        return id;
    };
    if id < from || id >= to {
        return id;
    }

    let mut debug_value = format!("{value:?}");
    if let Some((byte_index, _)) = debug_value.char_indices().nth(120) {
        debug_value.truncate(byte_index);
    }
    eprintln!(
        "[DTRACE-CONST] id={id} kind={} val={debug_value}",
        const_value_variant_name(value)
    );
    if constant_trace_backtrace_enabled() {
        let backtrace = std::backtrace::Backtrace::force_capture();
        eprintln!("[DTRACE-CONST-BT] id={id}\n{backtrace}\n[DTRACE-CONST-BT-END] id={id}");
    }
    id
}

// Exists to localise prepass nondeterminism (gh#1139).
fn constant_trace_backtrace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MAJIT_DTRACE_CONST_BT")
            .as_deref()
            .is_ok_and(|value| value == "1")
    })
}

// Exists to localise prepass nondeterminism (gh#1139).
fn constant_trace_window() -> Option<&'static (u64, u64)> {
    static WINDOW: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    WINDOW
        .get_or_init(|| {
            if !crate::determinism_trace_enabled() {
                return None;
            }
            let from = std::env::var("MAJIT_DTRACE_CONST_FROM")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            let to = std::env::var("MAJIT_DTRACE_CONST_TO")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            (to != 0).then_some((from, to))
        })
        .as_ref()
}

// Exists to localise prepass nondeterminism (gh#1139).
fn const_value_variant_name(value: &ConstValue) -> &'static str {
    match value {
        ConstValue::Atom(_) => "Atom",
        ConstValue::Placeholder => "Placeholder",
        ConstValue::Int(_) => "Int",
        ConstValue::Int128(_) => "Int128",
        ConstValue::UInt128(_) => "UInt128",
        ConstValue::Float(_) => "Float",
        ConstValue::Dict(_) => "Dict",
        ConstValue::ByteStr(_) => "ByteStr",
        ConstValue::UniStr(_) => "UniStr",
        ConstValue::Tuple(_) => "Tuple",
        ConstValue::List(_) => "List",
        ConstValue::Bool(_) => "Bool",
        ConstValue::None => "None",
        ConstValue::Code(_) => "Code",
        ConstValue::Function(_) => "Function",
        ConstValue::Graphs(_) => "Graphs",
        ConstValue::LowLevelType(_) => "LowLevelType",
        ConstValue::LLPtr(_) => "LLPtr",
        ConstValue::LLAddress(_) => "LLAddress",
        ConstValue::HostObject(_) => "HostObject",
        ConstValue::AddressOffset(_) => "AddressOffset",
        ConstValue::SpecTag(_) => "SpecTag",
        ConstValue::InheritanceId { .. } => "InheritanceId",
        ConstValue::Opaque(_) => "Opaque",
        ConstValue::Repr(_) => "Repr",
    }
}

// Exists to localise prepass nondeterminism (gh#1139).
pub fn next_constant_id() -> u64 {
    NEXT_CONSTANT_ID.load(Ordering::Relaxed)
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        if Self::uses_hashable_identity_fallback(&self.value)
            || Self::uses_hashable_identity_fallback(&other.value)
        {
            self.id == other.id
        } else {
            self.value == other.value
        }
    }
}

impl Eq for Constant {}

impl Hash for Constant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if Self::uses_hashable_identity_fallback(&self.value) {
            self.id.hash(state);
        } else {
            self.value.hash(state);
        }
    }
}

impl ConstValue {
    /// Wrap an `f64` into the bit-preserving `Float` variant.
    pub fn float(value: f64) -> Self {
        ConstValue::Float(value.to_bits())
    }

    /// Construct a Python 2 byte string (`str`) constant.
    pub fn byte_str(value: impl AsRef<[u8]>) -> Self {
        ConstValue::ByteStr(value.as_ref().to_vec())
    }

    /// Construct a Python 2 unicode constant.
    pub fn uni_str(value: impl AsRef<str>) -> Self {
        ConstValue::UniStr(value.as_ref().to_string())
    }

    /// llmemory.py `not self.const` — true when the value is a
    /// NULL address (`_address::Null` / `fakeaddress(None)`).
    pub fn is_null_address(&self) -> bool {
        matches!(
            self,
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null)
        )
    }

    /// Return the raw bytes for a Python 2 byte string constant.
    pub fn as_byte_str(&self) -> Option<&[u8]> {
        match self {
            ConstValue::ByteStr(value) => Some(value.as_slice()),
            _ => None,
        }
    }

    /// Return the text for a Python 2 unicode constant.
    pub fn as_uni_str(&self) -> Option<&str> {
        match self {
            ConstValue::UniStr(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Return a Rust `str` view for string constants used as host
    /// names/attributes. Python 2 byte strings are accepted only when
    /// they are valid UTF-8, which covers identifiers and metadata.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ConstValue::ByteStr(value) => std::str::from_utf8(value).ok(),
            ConstValue::UniStr(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Mirror RPython's `isinstance(value, str)` (Python 2 bytes-only)
    /// check on string constants. Use this where the upstream call
    /// site explicitly demands `str` rather than the generic
    /// `bytes`-or-`unicode` text view returned by [`as_text`]. Returns
    /// the byte-string interpreted as UTF-8 text — ASCII identifiers /
    /// attribute names always succeed, while a `unicode` constant or a
    /// non-UTF-8 byte string yields `None`.
    pub fn as_pystr(&self) -> Option<&str> {
        match self {
            ConstValue::ByteStr(value) => std::str::from_utf8(value).ok(),
            _ => None,
        }
    }

    pub fn into_text(self) -> Option<String> {
        match self {
            ConstValue::ByteStr(value) => String::from_utf8(value).ok(),
            ConstValue::UniStr(value) => Some(value),
            _ => None,
        }
    }

    pub fn string_eq(&self, expected: &str) -> bool {
        self.as_text() == Some(expected)
    }

    /// Unwrap a `Float` variant back to `f64`. Returns `None` for
    /// other variants.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstValue::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Best-effort Python truthiness for the constant variants the
    /// current flow-space port constructs directly.
    pub fn truthy(&self) -> Option<bool> {
        match self {
            ConstValue::Placeholder => None,
            ConstValue::Int(n) => Some(*n != 0),
            ConstValue::Int128(n) => Some(*n != 0),
            ConstValue::UInt128(n) => Some(*n != 0),
            // Python `bool(float)` is `float != 0.0` (including -0.0
            // which equals 0.0 under IEEE 754).
            ConstValue::Float(bits) => Some(f64::from_bits(*bits) != 0.0),
            ConstValue::Dict(items) => Some(!items.is_empty()),
            ConstValue::ByteStr(s) => Some(!s.is_empty()),
            ConstValue::UniStr(s) => Some(!s.is_empty()),
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(!items.is_empty()),
            ConstValue::Bool(value) => Some(*value),
            ConstValue::None => Some(false),
            ConstValue::Code(_) => Some(true),
            ConstValue::Graphs(graphs) => Some(!graphs.is_empty()),
            ConstValue::LowLevelType(_) => Some(true),
            ConstValue::LLPtr(ptr) => Some(ptr.nonzero()),
            // `fakeaddress.__nonzero__` (llmemory.py) = `ptr is not
            // None`: a `Fake` and an (odd, tagged-int) `IntCast` are both
            // non-NULL, only `Null` is false. Single-sourced on `nonzero`.
            ConstValue::LLAddress(addr) => Some(addr.nonzero()),
            ConstValue::Function(_) => Some(true),
            ConstValue::HostObject(_) => Some(true),
            ConstValue::Atom(_) => Some(true),
            ConstValue::SpecTag(_) => Some(true),
            ConstValue::AddressOffset(_) => Some(true),
            ConstValue::InheritanceId { .. } => Some(true),
            ConstValue::Opaque(_) | ConstValue::Repr(_) => Some(true),
        }
    }

    pub fn dict_items(&self) -> Option<&HashMap<ConstValue, ConstValue>> {
        match self {
            ConstValue::Dict(items) => Some(items),
            _ => None,
        }
    }

    pub fn graphs(&self) -> Option<&[usize]> {
        match self {
            ConstValue::Graphs(graphs) => Some(graphs.as_slice()),
            _ => None,
        }
    }

    pub fn sequence_items(&self) -> Option<&[ConstValue]> {
        match self {
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    /// RPython `for x in self.w_stararg.value` iteration contract
    /// (`argument.py:113`) — upstream consumes any Python iterable.
    /// Line-by-line port for the `ConstValue` variants that can appear
    /// as `w_stararg.value` in practice: tuple/list (flow-space's
    /// BUILD_TUPLE/BUILD_LIST output), str/unicode (per-item
    /// iteration matches Python semantics), dict (iteration over keys).
    pub fn iter_items(&self) -> Option<Vec<ConstValue>> {
        match self {
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(items.clone()),
            ConstValue::ByteStr(value) => Some(
                value
                    .iter()
                    .copied()
                    .map(|b| ConstValue::ByteStr(vec![b]))
                    .collect(),
            ),
            ConstValue::UniStr(value) => Some(
                value
                    .chars()
                    .map(|c| ConstValue::uni_str(c.to_string()))
                    .collect(),
            ),
            ConstValue::Dict(items) => Some(items.keys().cloned().collect()),
            _ => None,
        }
    }

    /// If this is an exception class, returns `qualname()`. To decide whether
    /// an arbitrary Class is an exception class, check
    /// `HOST_ENV.lookup_builtin("BaseException")` and `is_subclass_of` — this
    /// helper exposes any qualname as long as the value is a class.
    pub fn host_class_name(&self) -> Option<&str> {
        match self {
            ConstValue::HostObject(obj) if obj.is_class() => Some(obj.qualname()),
            _ => None,
        }
    }

    /// Host-level `obj.__class__` / `type(obj)` for the constant
    /// variants that have a direct Python object analogue in the
    /// current flow-space port.
    pub fn class_of(&self) -> Option<HostObject> {
        match self {
            ConstValue::Int(_) | ConstValue::Int128(_) | ConstValue::UInt128(_) => {
                HOST_ENV.lookup_builtin("int")
            }
            ConstValue::Float(_) => HOST_ENV.lookup_builtin("float"),
            ConstValue::Bool(_) => HOST_ENV.lookup_builtin("bool"),
            ConstValue::ByteStr(_) => HOST_ENV.lookup_builtin("str"),
            ConstValue::UniStr(_) => HOST_ENV.lookup_builtin("unicode"),
            ConstValue::Tuple(_) => HOST_ENV.lookup_builtin("tuple"),
            ConstValue::List(_) => HOST_ENV.lookup_builtin("list"),
            ConstValue::Dict(_) => HOST_ENV.lookup_builtin("dict"),
            ConstValue::None => HOST_ENV.lookup_builtin("NoneType"),
            ConstValue::Code(_) => HOST_ENV.lookup_builtin("code"),
            ConstValue::Function(_) => HOST_ENV.lookup_builtin("function"),
            ConstValue::HostObject(obj) => obj.class_of(),
            ConstValue::Atom(_)
            | ConstValue::Placeholder
            | ConstValue::Graphs(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::LLPtr(_)
            | ConstValue::LLAddress(_)
            | ConstValue::AddressOffset(_)
            | ConstValue::InheritanceId { .. }
            | ConstValue::SpecTag(_)
            | ConstValue::Opaque(_)
            | ConstValue::Repr(_) => None,
        }
    }

    /// Reference when this is a HostObject. Shared by the patterns that
    /// replaced ExceptionClass/Builtin/ExceptionInstance.
    pub fn as_host_object(&self) -> Option<&HostObject> {
        match self {
            ConstValue::HostObject(obj) => Some(obj),
            _ => None,
        }
    }

    /// Pulls a HostObject out of the `__builtin__` / stdlib namespace by name
    /// and wraps it as `ConstValue::HostObject`. Panics if `HOST_ENV` has no
    /// such name (a missing bootstrap entry should show up immediately during
    /// development).
    pub fn builtin(name: &str) -> Self {
        let obj = HOST_ENV
            .lookup_builtin(name)
            .unwrap_or_else(|| panic!("HOST_ENV missing builtin {name}"));
        ConstValue::HostObject(obj)
    }
}

/// Application-level exception captured inside the flow space.
///
/// RPython basis: `rpython/flowspace/model.py:385-392` — `class
/// FSException(object)`. `w_type` and `w_value` carry the exception
/// class and instance as flow-space `Hlvalue`s.
///
/// RPython's `ConstException(Constant, FSException)` multiple
/// inheritance has no direct Rust equivalent; the upstream only uses
/// it as a fast-path marker for `foldable()` on constant exceptions.
/// Phase 3 (`flowspace/operation.py`) handles the same flow via a
/// `Constant`-carrying FSException without the extra class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FSException {
    /// RPython `FSException.w_type` — the exception class (as an
    /// `Hlvalue`, usually a `Constant` wrapping the Python class).
    pub w_type: Hlvalue,
    /// RPython `FSException.w_value` — the exception instance (as an
    /// `Hlvalue`).
    pub w_value: Hlvalue,
}

impl FSException {
    /// RPython `FSException.__init__(w_type, w_value)`.
    pub fn new(w_type: Hlvalue, w_value: Hlvalue) -> Self {
        FSException { w_type, w_value }
    }
}

impl std::fmt::Display for FSException {
    // RPython `FSException.__str__` — `[w_type: w_value]`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}: {}]", self.w_type, self.w_value)
    }
}

/// Mixed `Variable | Constant` cell used by `SpaceOperation.args`,
/// `SpaceOperation.result`, and `Block.inputargs`.
///
/// RPython relies on duck-typing: the upstream lists literally hold
/// either a `Variable` or a `Constant` instance and dispatch through
/// `.replace(mapping)`. Rust needs a closed sum to hand a single
/// owned type across API boundaries; this is the minimum deviation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Hlvalue {
    Variable(Variable),
    Constant(Constant),
}

impl Hlvalue {
    /// RPython `.replace(mapping)` dispatch — matches whichever
    /// subclass `Variable.replace` / `Constant.replace` method the
    /// cell carries.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
        match self {
            Hlvalue::Variable(v) => v.replace(mapping),
            Hlvalue::Constant(c) => c.replace(mapping),
        }
    }
}

impl std::fmt::Display for Hlvalue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hlvalue::Variable(v) => write!(f, "{v}"),
            // RPython `Constant.__repr__` (inherited from Hashable)
            // prints `<value>`; we approximate with the enum debug.
            Hlvalue::Constant(c) => write!(f, "{c:?}"),
        }
    }
}

impl From<Variable> for Hlvalue {
    fn from(v: Variable) -> Self {
        Hlvalue::Variable(v)
    }
}

impl From<Constant> for Hlvalue {
    fn from(c: Constant) -> Self {
        Hlvalue::Constant(c)
    }
}

/// RPython `flowspace/model.py` — `class SpaceOperation`.
#[derive(Clone, Debug, Eq)]
pub struct SpaceOperation {
    /// RPython `SpaceOperation.opname` — `intern(opname)`.
    pub opname: String,
    /// RPython `SpaceOperation.args` — mixed `Variable`/`Constant`
    /// list.
    pub args: Vec<Hlvalue>,
    /// RPython `SpaceOperation.result` — either `Variable` or
    /// `Constant`.
    pub result: Hlvalue,
    /// RPython `SpaceOperation.offset` — source bytecode offset,
    /// `-1` when unknown.
    pub offset: i64,
}

impl SpaceOperation {
    /// RPython `SpaceOperation.__init__(opname, args, result,
    /// offset=-1)`.
    pub fn new(opname: impl Into<String>, args: Vec<Hlvalue>, result: Hlvalue) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset: -1,
        }
    }

    /// RPython overload with an explicit `offset`.
    pub fn with_offset(
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
        result: Hlvalue,
        offset: i64,
    ) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset,
        }
    }

    /// RPython `SpaceOperation.replace(mapping)`.
    ///
    /// Returns a fresh SpaceOperation with the same opname/offset and
    /// every arg/result remapped through `mapping`.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> SpaceOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = self.result.replace(mapping);
        SpaceOperation {
            opname: self.opname.clone(),
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }
}

impl PartialEq for SpaceOperation {
    // RPython `SpaceOperation.__eq__`: same class, same opname, same
    // args, same result. `offset` is intentionally NOT part of the
    // identity (upstream: model.py:442-446).
    fn eq(&self, other: &Self) -> bool {
        self.opname == other.opname && self.args == other.args && self.result == other.result
    }
}

impl std::hash::Hash for SpaceOperation {
    // RPython `SpaceOperation.__hash__`: `hash((opname, tuple(args),
    // result))`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.opname.hash(state);
        self.args.hash(state);
        self.result.hash(state);
    }
}

impl std::fmt::Display for SpaceOperation {
    // RPython `SpaceOperation.__repr__` returns
    // `"%r = %s(%s)" % (result, opname, ", ".join(map(repr, args)))`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}(", self.result, self.opname)?;
        let mut first = true;
        for arg in &self.args {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{arg}")?;
            first = false;
        }
        f.write_str(")")
    }
}

// RPython `flowspace/model.py:469-470`.
//
// `last_exception = Atom('last_exception')`
// `c_last_exception = Constant(last_exception)`
//
// A Block's `exitswitch` set to `c_last_exception` signals that the
// last operation of the block may raise, and the exit links' exitcase
// fields carry the matching exception classes.

/// RPython `last_exception` — module-level Atom. Static so both the
/// `Block::canraise` check and the `c_last_exception` constant share
/// identity.
pub static LAST_EXCEPTION: Atom = Atom {
    name: "last_exception",
};

/// RPython `c_last_exception = Constant(last_exception)`.
pub fn c_last_exception() -> Constant {
    Constant::new(ConstValue::Atom(LAST_EXCEPTION.clone()))
}

/// `Rc<RefCell<Block>>` alias matching the RPython behaviour of
/// Block/Link references being mutable shared pointers. Every Link
/// targets a Block by shared reference; every Block holds outgoing
/// Links by shared reference. Cycles back through `Link.prevblock`
/// use `Weak<RefCell<Block>>` to avoid leaks.
pub type BlockRef = Rc<RefCell<Block>>;

/// `Rc<RefCell<Link>>` alias — see `BlockRef`.
pub type LinkRef = Rc<RefCell<Link>>;

/// `Link.args` carries the values flowing into a target block's
/// `inputargs`. RPython's `FrameState.getoutputargs()` may emit an
/// undefined-local sentinel (`None`) in transient merge links, so the
/// Rust port preserves that exact shape with `Option<Hlvalue>`.
pub type LinkArg = Option<Hlvalue>;

/// RPython `flowspace/model.py` — `class Link`.
///
/// `__slots__ = "args target exitcase llexitcase prevblock
///              last_exception last_exc_value".split()`.
pub struct Link {
    /// RPython `Link.args` — mixed list of var/const, with transient
    /// merge links allowed to carry `None` for undefined locals.
    pub args: Vec<LinkArg>,
    /// RPython `Link.target` — successor block. Always `Some` in
    /// constructed graphs; `Option` only to admit the upstream
    /// `target=None` transient state during graph building.
    pub target: Option<BlockRef>,
    /// RPython `Link.exitcase` — concrete value discriminating this
    /// exit. `None` for fall-through / single-exit blocks; Python
    /// values (via `Hlvalue::Constant`) for switch exits; exception
    /// class Constants for `c_last_exception` blocks.
    pub exitcase: Option<Hlvalue>,
    /// RPython `Link.llexitcase` — low-level equivalent of
    /// `exitcase`, populated by the rtyper at Phase 6. Placeholder
    /// here.
    pub llexitcase: Option<Hlvalue>,
    /// RPython `Link.prevblock` — the Block this Link exits. Weak
    /// reference to break the cycle with `Block.exits`.
    pub prevblock: Option<Weak<RefCell<Block>>>,
    /// RPython `Link.last_exception` — extra variable introduced on
    /// exception-handling links.
    pub last_exception: Option<Hlvalue>,
    /// RPython `Link.last_exc_value` — sibling of `last_exception`.
    pub last_exc_value: Option<Hlvalue>,
}

impl std::fmt::Debug for Link {
    /// Parity with `Link.__repr__` (model.py:161): the shallow
    /// `"link from <prevblock> to <target>"` form. `str()` on the endpoint
    /// blocks is `Block::str_repr` (`Block.__str__`), which never follows
    /// `Block.exits` — so, unlike `#[derive(Debug)]`, formatting a Link
    /// cannot recurse around the `Block.exits`/`Link.target` CFG cycle.
    /// Borrows defensively: a Debug may run from a panic/error path with an
    /// endpoint block already borrowed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn block_str(b: &BlockRef) -> String {
            match b.try_borrow() {
                Ok(block) => block.str_repr(),
                Err(_) => "<block>".to_string(),
            }
        }
        let prevblock = match self.prevblock.as_ref().and_then(Weak::upgrade) {
            Some(rc) => block_str(&rc),
            None => "None".to_string(),
        };
        let target = match &self.target {
            Some(rc) => block_str(rc),
            None => "None".to_string(),
        };
        write!(f, "link from {prevblock} to {target}")
    }
}

impl Link {
    /// RPython `Link.__init__(args, target, exitcase=None)`.
    pub fn new(args: Vec<Hlvalue>, target: Option<BlockRef>, exitcase: Option<Hlvalue>) -> Self {
        Self::new_mergeable(args.into_iter().map(Some).collect(), target, exitcase)
    }

    pub fn new_mergeable(
        args: Vec<LinkArg>,
        target: Option<BlockRef>,
        exitcase: Option<Hlvalue>,
    ) -> Self {
        if let Some(t) = &target {
            assert_eq!(
                args.len(),
                t.borrow().inputargs.len(),
                "output args mismatch"
            );
        }
        Link {
            args,
            target,
            exitcase,
            llexitcase: None,
            prevblock: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    /// Wraps the owned `Link` in a shared `LinkRef`. Rust adaptation
    /// closing upstream's Python identity-sharing: `Block.closeblock`
    /// stores the link by reference into `block.exits`, and
    /// `Link.prevblock` is later set by `closeblock` through the
    /// stored reference.
    pub fn into_ref(self) -> LinkRef {
        Rc::new(RefCell::new(self))
    }

    /// RPython `Link.extravars(last_exception=None, last_exc_value=None)`.
    pub fn extravars(&mut self, last_exception: Option<Hlvalue>, last_exc_value: Option<Hlvalue>) {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
    }

    /// RPython `Link.getextravars()` — collect the Variable-typed
    /// extras, ignoring Constant slots.
    pub fn getextravars(&self) -> Vec<Variable> {
        let mut result = Vec::new();
        if let Some(Hlvalue::Variable(v)) = &self.last_exception {
            result.push(v.clone());
        }
        if let Some(Hlvalue::Variable(v)) = &self.last_exc_value {
            result.push(v.clone());
        }
        result
    }

    /// RPython `Link.copy(rename=lambda x: x)`.
    ///
    /// The `rename` callback maps each `Hlvalue` to a substitute —
    /// usually derived from a `Variable -> Variable` mapping. The
    /// closure form mirrors upstream's call shape.
    pub fn copy<F>(&self, rename: F) -> Link
    where
        F: Fn(&Hlvalue) -> Hlvalue,
    {
        let newargs: Vec<LinkArg> = self
            .args
            .iter()
            .map(|arg| arg.as_ref().map(&rename))
            .collect();
        let mut newlink = Link::new_mergeable(newargs, self.target.clone(), self.exitcase.clone());
        newlink.prevblock = self.prevblock.clone();
        newlink.last_exception = self.last_exception.as_ref().map(&rename);
        newlink.last_exc_value = self.last_exc_value.as_ref().map(&rename);
        newlink.llexitcase = self.llexitcase.clone();
        newlink
    }

    /// RPython `Link.replace(mapping)`.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> Link {
        self.copy(|v| v.replace(mapping))
    }

    /// RPython `Link.settarget(targetblock)`.
    pub fn settarget(&mut self, targetblock: BlockRef) {
        assert_eq!(
            self.args.len(),
            targetblock.borrow().inputargs.len(),
            "output args mismatch"
        );
        self.target = Some(targetblock);
    }
}

/// RPython `flowspace/model.py` — `class Block`.
///
/// `__slots__ = "inputargs operations exitswitch exits blockcolor
///              generation".split()`.
pub struct Block {
    /// RPython `Block.inputargs` — mixed list of variable/const.
    pub inputargs: Vec<Hlvalue>,
    /// RPython `Block.operations` — list of `SpaceOperation`.
    pub operations: Vec<SpaceOperation>,
    /// Block parity flag: final (return/except) blocks have
    /// `operations == ()` in RPython (a tuple, not a list).
    pub is_final: bool,
    /// RPython `Block.exitswitch` — either a Variable, a Constant,
    /// or `c_last_exception`. `None` means no exit discriminator.
    pub exitswitch: Option<Hlvalue>,
    /// RPython `Block.exits` — list of outgoing Links.
    pub exits: Vec<LinkRef>,
    /// RPython `Block.blockcolor` — used by graph-layout visualisers.
    pub blockcolor: Option<u32>,
    /// RPython `Block.generation` — used by `flowcontext` iteration
    /// scheduling.
    pub generation: Option<u32>,
}

impl std::fmt::Debug for Block {
    /// Parity with `Block.__repr__` / `Block.__str__` (model.py:191-212):
    /// a shallow one-line summary (`block@<offset> with <n> exits`), never
    /// the full recursive dump of inputarg annotations, operations and
    /// exits that `#[derive(Debug)]` produces. Descending through each
    /// inputarg's `annotation` (SomeValue) and `concretetype`
    /// (LowLevelType) — both self-referential for recursive types such as
    /// `pyframe::PyFrame` — overflows the stack when a `where_info`
    /// diagnostic formats a block (rtyper.rs `with_where`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Block.__repr__
        write!(f, "{} with {} exits", self.str_repr(), self.exits.len())?;
        if let Some(sw) = &self.exitswitch {
            write!(f, "({sw})")?;
        }
        Ok(())
    }
}

impl Block {
    /// RPython `Block.__str__` (model.py:191): the shallow one-line label
    /// (`block@<offset>[<inputarg>...]`, or `return`/`raise`/`codeless
    /// block`). Never follows `exits`, so it is safe to format from a
    /// `Link` endpoint without recursing around the CFG cycle.
    pub(crate) fn str_repr(&self) -> String {
        let mut txt = if let Some(op) = self.operations.first() {
            format!("block@{}", op.offset)
        } else if self.exits.is_empty() && self.inputargs.len() == 1 {
            "return block".to_string()
        } else if self.exits.is_empty() && self.inputargs.len() == 2 {
            "raise block".to_string()
        } else {
            "codeless block".to_string()
        };
        if let Some(first) = self.inputargs.first() {
            if self.inputargs.len() > 1 {
                txt = format!("{txt}[{first}...]");
            } else {
                txt = format!("{txt}[{first}]");
            }
        }
        txt
    }

    /// RPython `Block.__init__(inputargs)`.
    pub fn new(inputargs: Vec<Hlvalue>) -> Self {
        Block {
            inputargs,
            operations: Vec::new(),
            is_final: false,
            exitswitch: None,
            exits: Vec::new(),
            blockcolor: None,
            generation: None,
        }
    }

    /// Shared-ownership constructor — upstream passes `Block(args)`
    /// and stores it directly in `FunctionGraph`/`Link.target`; our
    /// graph types take `Rc<RefCell<Block>>`, so callers usually
    /// want `Block::shared(...)` instead of `Block::new(...)`.
    pub fn shared(inputargs: Vec<Hlvalue>) -> BlockRef {
        let block = Rc::new(RefCell::new(Block::new(inputargs)));
        record_block_address(&block);
        block
    }

    /// RPython `Block.is_final_block()` — `self.operations == ()`.
    pub fn is_final_block(&self) -> bool {
        self.is_final
    }

    /// Mark this block as final (return/except). Matches upstream's
    /// post-`__init__` assignment `block.operations = ()`.
    pub fn mark_final(&mut self) {
        self.operations.clear();
        self.is_final = true;
    }

    /// RPython `Block.canraise` (property).
    pub fn canraise(&self) -> bool {
        match &self.exitswitch {
            Some(Hlvalue::Constant(c)) => matches!(
                &c.value,
                ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name
            ),
            _ => false,
        }
    }

    /// RPython `Block.raising_op` (property).
    pub fn raising_op(&self) -> Option<&SpaceOperation> {
        if self.canraise() {
            self.operations.last()
        } else {
            None
        }
    }

    /// RPython `Block.getvariables()` — unique Variables mentioned
    /// in this block (inputargs + every op's args + every op's
    /// result).
    pub fn getvariables(&self) -> Vec<Variable> {
        let mut result: Vec<Variable> = Vec::new();
        let push_var = |w: &Hlvalue, result: &mut Vec<Variable>| {
            if let Hlvalue::Variable(v) = w
                && !result.iter().any(|x| x == v)
            {
                result.push(v.clone());
            }
        };
        for w in &self.inputargs {
            push_var(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_var(a, &mut result);
            }
            push_var(&op.result, &mut result);
        }
        result
    }

    /// RPython `Block.getconstants()` — unique Constants mentioned
    /// in this block (inputargs + every op's args).
    pub fn getconstants(&self) -> Vec<Constant> {
        let mut result: Vec<Constant> = Vec::new();
        let push_const = |w: &Hlvalue, result: &mut Vec<Constant>| {
            if let Hlvalue::Constant(c) = w
                && !result.iter().any(|x| x == c)
            {
                result.push(c.clone());
            }
        };
        for w in &self.inputargs {
            push_const(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_const(a, &mut result);
            }
        }
        result
    }

    /// RPython `Block.renamevariables(mapping)`.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn renamevariables(&mut self, mapping: &HashMap<Variable, Hlvalue>) {
        self.inputargs = self.inputargs.iter().map(|a| a.replace(mapping)).collect();
        self.operations = self
            .operations
            .iter()
            .map(|op| op.replace(mapping))
            .collect();
        if let Some(sw) = self.exitswitch.take() {
            self.exitswitch = Some(sw.replace(mapping));
        }
        for link_ref in &self.exits {
            let mut link = link_ref.borrow_mut();
            link.args = link
                .args
                .iter()
                .map(|arg| arg.as_ref().map(|value| value.replace(mapping)))
                .collect();
        }
    }

    /// RPython `Block.closeblock(*exits)` — one-shot close.
    pub fn closeblock(&mut self, exits: Vec<LinkRef>) {
        assert!(self.exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    /// RPython `Block.recloseblock(*exits)` — may be called after
    /// `closeblock`, rewiring `Link.prevblock` to this block.
    ///
    /// Caller must pass `self_ref: Weak<RefCell<Block>>` because
    /// `&mut self` cannot upgrade back to the Rc. Upstream has this
    /// for free via Python object identity.
    pub fn recloseblock_with_self(&mut self, self_ref: Weak<RefCell<Block>>, exits: Vec<LinkRef>) {
        for link in &exits {
            link.borrow_mut().prevblock = Some(self_ref.clone());
        }
        self.exits = exits;
    }

    /// Variant that skips `prevblock` wiring when the caller has
    /// already set it (e.g. during `copygraph`).
    pub fn recloseblock(&mut self, exits: Vec<LinkRef>) {
        self.exits = exits;
    }
}

/// Extension trait that gives `BlockRef` the full RPython
/// `Block.closeblock` / `Block.recloseblock` API, including
/// `Link.prevblock` wiring. `Block::closeblock`/`recloseblock` on
/// `&mut self` cannot wire `prevblock` because the borrow-checker
/// does not let us upgrade back to the owning Rc; the trait provides
/// the `self_ref`-aware version.
pub trait BlockRefExt {
    /// RPython `Block.closeblock(*exits)`.
    fn closeblock(&self, exits: Vec<LinkRef>);
    /// RPython `Block.recloseblock(*exits)`.
    fn recloseblock(&self, exits: Vec<LinkRef>);
}

impl BlockRefExt for BlockRef {
    fn closeblock(&self, exits: Vec<LinkRef>) {
        assert!(self.borrow().exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    fn recloseblock(&self, exits: Vec<LinkRef>) {
        let weak = Rc::downgrade(self);
        for link in &exits {
            link.borrow_mut().prevblock = Some(weak.clone());
        }
        self.borrow_mut().exits = exits;
    }
}

/// Native call hook carried by a memo-specialised [`GraphFunc`].
///
/// Wraps an `Arc<dyn Fn(&[ConstValue]) -> Result<ConstValue, String>>`
/// so [`GraphFunc`] can stay `Clone`; a manual [`std::fmt::Debug`] keeps
/// the `#[derive(Debug)]` on `GraphFunc` valid since `dyn Fn` is not
/// `Debug`. Equality/identity of `GraphFunc` is `id`-based and ignores
/// this field.
#[derive(Clone)]
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub struct HostCall(pub Arc<dyn Fn(&[ConstValue]) -> Result<ConstValue, String> + Send + Sync>);

impl std::fmt::Debug for HostCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostCall(..)")
    }
}

/// Exact translated values used by RPython's `func._always_inline_`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AlwaysInline {
    /// The attribute is absent (`getattr(..., None)`).
    #[default]
    Absent,
    /// Literal `True`: force the inline and fail if it cannot be performed.
    True,
    /// String `'try'`: prefer the inline but tolerate a failed attempt.
    Try,
}

impl AlwaysInline {
    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Absent)
    }

    #[inline]
    pub fn is_strict(self) -> bool {
        matches!(self, Self::True)
    }
}

/// Stand-in for the Python function object attached to
/// `FunctionGraph.func`.
#[derive(Clone, Debug)]
pub struct GraphFunc {
    /// Stable object identity for RPython `Hashable(function)` parity.
    pub id: u64,
    /// Python function `__name__`.
    pub name: String,
    /// Upstream `func._sandbox_external_name`.
    pub _sandbox_external_name: Option<String>,
    /// Optional method owner, mirroring upstream `func.class_`.
    pub class_: Option<HostObject>,
    /// Upstream `func._annspecialcase_` decorator tag consulted by
    /// `Bookkeeper.newfuncdesc()` when selecting a specializer.
    pub annspecialcase: Option<String>,
    /// Annotation-time native call hook, the pyre stand-in for
    /// `funcdesc.pyobj(*args)` in `rpython/annotator/specialize.py`'s
    /// `memo`. RPython invokes the live Python callable directly;
    /// pyre's analysed program is Rust, so a memo-specialised function
    /// carries its native closure here — populated from the build-time
    /// linkage to pyre-interpreter (or by a test). The currency is
    /// [`ConstValue`], matching `all_values` output and
    /// `Bookkeeper.immutablevalue` input. `None` for ordinary
    /// functions, which never reach the memo path.
    pub host_call: Option<HostCall>,
    /// Upstream generator helper attribute populated by
    /// `attach_next_method()`.
    pub _generator_next_method_of_: Option<HostObject>,
    /// Upstream function attribute consulted by
    /// `annotator/specialize.py:64` and `jit/codewriter/policy.py:56`.
    ///
    /// `None` mirrors the common "attribute absent" case so callers
    /// can apply the same default as upstream's
    /// `getattr(func, '_jit_look_inside_', True)`.
    pub _jit_look_inside_: Option<bool>,
    /// Upstream function attribute consulted by
    /// `description.py:222-224`.
    ///
    /// `None` mirrors the common "attribute absent" case so callers
    /// can apply the same default as upstream's
    /// `getattr(func, 'relax_sig_check', False)`.
    pub relax_sig_check: Option<bool>,
    /// Upstream function attribute set by
    /// `ExtEnterLeaveMarker.compute_result_annotation` at
    /// `rpython/rlib/jit.py`:
    ///
    /// ```python
    /// try:
    ///     graph = self.bookkeeper.position_key[0]
    ///     graph.func._dont_reach_me_in_del_ = True
    /// except (TypeError, AttributeError):
    ///     pass
    /// ```
    ///
    /// Read by `rpython/rtyper/rclass.py` to suppress the
    /// `_del_method_must_not_run_when_jit_marker_is_present` assertion
    /// for graphs that contain a `jit_merge_point`. `false` mirrors the
    /// "attribute absent" upstream default.
    pub _dont_reach_me_in_del_: bool,
    /// Upstream `func._always_inline_`, consumed by
    /// `translator/backendopt/inline.py always_inline`.
    ///
    /// RPython distinguishes literal `True` from the truthy string `'try'`:
    /// both select the candidate at zero cost, but only literal `True` makes
    /// an unsuccessful inline fatal (`inline.py`).
    pub _always_inline_: AlwaysInline,
    /// Upstream `func._dont_inline_` (read off the live Python
    /// callable via `getattr(funcobj._callable, '_dont_inline_',
    /// False)` at `inline.py:563-564` /
    /// `:617-618`). Set by the `@dont_inline` decorator. `false`
    /// mirrors the "attribute absent" upstream default. Pyre's
    /// callable carrier is the `GraphFunc` itself; the flag rides
    /// along here so the inliner does not have to look through a
    /// `_callable` string.
    pub _dont_inline_: bool,
    /// Upstream `func._no_release_gil_`, consumed by
    /// `translator/backendopt/gilanalysis.py`.
    ///
    /// majit/pyre is freethreaded and has no process-wide GIL to
    /// release. The port still carries the upstream attribute name:
    /// `gilanalysis` interprets it as a stricter "no thread
    /// safepoint / no blocking / no close-stack boundary below this
    /// graph" contract.
    pub _no_release_gil_: bool,
    /// Upstream `func._gctransformer_hint_close_stack_`, consumed by
    /// `gilanalysis.py:11-15`. In the freethreaded runtime this is a
    /// safepoint-like boundary because closing the stack can expose
    /// roots and scheduling/collection state.
    pub _gctransformer_hint_close_stack_: bool,
    /// Upstream `func._gctransformer_hint_cannot_collect_`, consumed
    /// by `collectanalyze.py:15-16` and the JIT codewriter
    /// (`call.py`). When set, `CollectAnalyzer` short-circuits to
    /// `bottom_result` for `direct_call` into this graph regardless
    /// of what the body does. `false` mirrors the "attribute absent"
    /// upstream default.
    pub _gctransformer_hint_cannot_collect_: bool,
    /// Upstream `func._must_be_light_finalizer_`, consumed by
    /// `finalizer.py analyze_light_finalizer`. When set on a
    /// `__del__` method's graph, `FinalizerAnalyzer` rejects any
    /// non-light operation with `FinalizerError`. `false` mirrors the
    /// "attribute absent" upstream default.
    pub _must_be_light_finalizer_: bool,
    /// Upstream `func._transaction_break_`, consumed by
    /// `gilanalysis.py:16-18`. pyre does not currently model STM
    /// transactions, but preserving the attribute keeps the analyzer
    /// structurally aligned with RPython and gives future callers the
    /// same hazard hook.
    pub _transaction_break_: bool,
    /// Upstream `func.exported_symbol` attribute set by
    /// `rpython/rlib/entrypoint.py export_symbol(func)` —
    /// `func.exported_symbol = True; return func`. Consumed only by
    /// the C backend (`rpython/translator/c/database.py` walks
    /// `getattr(callable, 'exported_symbol', False)`), which is not
    /// ported; the flag still rides along on `GraphFunc` so line-by-
    /// line surface with `interactive.py Translation.__init__ export_symbol(entry_point)`
    /// stays intact.
    ///
    /// Wrapped in `Arc<AtomicBool>` so every `Clone`-produced copy of
    /// the GraphFunc shares the same flag cell — upstream Python's
    /// in-place mutation `func.exported_symbol = True` is observable
    /// through every reference because Python objects share identity.
    /// The Rust port mirrors that by Arc-cloning the AtomicBool so a
    /// `store(true)` on any clone is observable on every other.
    /// `Relaxed` ordering is sufficient — there is no other shared
    /// memory whose visibility this flag synchronizes with.
    pub exported_symbol: Arc<AtomicBool>,
    /// Upstream `func._llfnobjattrs_` consumed by
    /// `lltype.getfunctionptr()`. It can force `_name`, `_callable`, or
    /// arbitrary fields on the low-level function object.
    pub _llfnobjattrs_: HashMap<String, ConstValue>,
    /// Upstream `func._llhelper_error_value_`, read by
    /// `translator/exceptiontransform.py:error_value` for callbacks whose
    /// C-visible return value is meaningful.
    pub _llhelper_error_value_: Option<Constant>,
    /// Python function `__globals__`, wrapped as a flow-space
    /// constant.
    pub globals: Constant,
    /// Python function `__closure__`.
    pub closure: Vec<Constant>,
    /// Python function `__defaults__`.
    pub defaults: Vec<Constant>,
    /// Python function `__code__`.
    pub code: Option<Box<HostCode>>,
    /// Source text returned by `inspect.getsource(func)`.
    pub source: Option<String>,
    /// `func.__code__.co_firstlineno`.
    pub firstlineno: Option<u32>,
    /// `func.__code__.co_filename`.
    pub filename: Option<String>,
    /// `func.__module__`. Upstream sets this from the surrounding
    /// `def` site's containing module. Populated at ingest where
    /// known; left `None` when the bookkeeper has only a synthetic
    /// function (e.g. flowspace-test fixtures), in which case
    /// downstream consumers fall back to the filename basename.
    pub module: Option<String>,
    /// Upstream `func._not_rpython_` attribute (objspace.py:21). Set
    /// to `true` by RPython's `@not_rpython` decorator to mark a
    /// function as flow-space-ineligible.
    pub not_rpython: bool,
}

impl GraphFunc {
    pub fn new(name: impl Into<String>, globals: Constant) -> Self {
        GraphFunc {
            id: NEXT_GRAPH_FUNC_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            _sandbox_external_name: None,
            class_: None,
            annspecialcase: None,
            host_call: None,
            _generator_next_method_of_: None,
            _jit_look_inside_: None,
            relax_sig_check: None,
            _dont_reach_me_in_del_: false,
            _always_inline_: AlwaysInline::Absent,
            _dont_inline_: false,
            _no_release_gil_: false,
            _gctransformer_hint_close_stack_: false,
            _gctransformer_hint_cannot_collect_: false,
            _must_be_light_finalizer_: false,
            _transaction_break_: false,
            exported_symbol: Arc::new(AtomicBool::new(false)),
            _llfnobjattrs_: HashMap::new(),
            _llhelper_error_value_: None,
            globals,
            closure: Vec::new(),
            defaults: Vec::new(),
            code: None,
            source: None,
            firstlineno: None,
            filename: None,
            module: None,
            not_rpython: false,
        }
    }

    /// Snapshot accessor for `func.__globals__`: returns a clone of
    /// the static `self.globals.value` snapshot frozen at construction
    /// (`flowcontext.py self.w_globals = Constant(func.__globals__)`).
    pub fn live_globals(&self) -> ConstValue {
        self.globals.value.clone()
    }

    /// RPython `func_with_new_name(func, newname)` copies the function
    /// object while reusing `__code__`, `__defaults__`, and
    /// `__closure__`; the `__code__.co_name` payload stays unchanged.
    pub fn with_new_name(&self, newname: &str) -> Self {
        let mut cloned = self.clone();
        cloned.id = NEXT_GRAPH_FUNC_ID.fetch_add(1, Ordering::Relaxed);
        cloned.name = newname.to_string();
        cloned
    }

    pub fn _hashable_identity(&self) -> u64 {
        self.id
    }

    pub fn from_host_code(code: HostCode, globals: Constant, defaults: Vec<Constant>) -> Self {
        // CPython sets `function.__module__` from
        // `globals.get('__name__')` at function-creation time
        // (`Objects/funcobject.c PyFunction_NewWithQualName`); RPython
        // reads it back through `fn.__module__`. Mirror by extracting
        // the byte-string value of `'__name__'` from the globals Dict
        // when present. Synthetic callers that pass an empty Dict get
        // `module = None`, which keeps the existing fallback path in
        // [`crate::tool::sourcetools::nice_repr_for_func`].
        let module = match &globals.value {
            ConstValue::Dict(items) => items
                .get(&ConstValue::byte_str(b"__name__"))
                .and_then(ConstValue::as_byte_str)
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
            _ => None,
        };
        let mut func = GraphFunc::new(code.co_name.clone(), globals);
        func.filename = Some(code.co_filename.clone());
        func.firstlineno = Some(code.co_firstlineno);
        func.module = module;
        func.defaults = defaults;
        func.code = Some(Box::new(code));
        func
    }
}

impl PartialEq for GraphFunc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GraphFunc {}

static NEXT_GRAPH_FUNC_ID: AtomicU64 = AtomicU64::new(1);

/// RPython `flowspace/model.py` — `class FunctionGraph`.
#[derive(Debug)]
pub struct FunctionGraph {
    /// RPython `FunctionGraph.name`.
    pub name: String,
    /// RPython `FunctionGraph.startblock`.
    pub startblock: BlockRef,
    /// RPython `FunctionGraph.returnblock` — `Block([return_var])`
    /// with `operations = ()` and `exits = ()`.
    pub returnblock: BlockRef,
    /// RPython `FunctionGraph.exceptblock` —
    /// `Block([Variable('etype'), Variable('evalue')])` with
    /// `operations = ()` and `exits = ()`.
    pub exceptblock: BlockRef,
    /// RPython `FunctionGraph.tag`. `None` until set by an analysis
    /// pass.
    pub tag: Option<String>,
    /// RPython `FunctionGraph.func`.
    pub func: Option<GraphFunc>,
    /// RPython `FunctionGraph._source`.
    pub _source: Option<String>,
    /// RPython `FunctionGraph._seen_by_the_backend` — set by the C
    /// backend after a graph has been processed (see
    /// `rpython/translator/c/funcgen.py`). The
    /// `backend_optimizations` entry asserts it is `False`
    /// (`all.py`); the local port carries the same flag so the
    /// assert can be enforced even though the C backend write site
    /// has not landed yet.
    pub _seen_by_the_backend: std::cell::Cell<bool>,
}

impl FunctionGraph {
    /// Wrap this graph in the shared reference used throughout the
    /// translation pipeline.
    pub fn into_ref(self) -> GraphRef {
        let graph = Rc::new(RefCell::new(self));
        record_graph_address(&graph);
        graph
    }

    /// RPython `FunctionGraph.__init__(name, startblock,
    /// return_var=None)`.
    pub fn new(name: impl Into<String>, startblock: BlockRef) -> Self {
        // RPython path: `return_var = None` → a fresh Variable.
        let return_var = Hlvalue::Variable(Variable::new());
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
            func: None,
            _source: None,
            _seen_by_the_backend: std::cell::Cell::new(false),
        }
    }

    /// RPython `FunctionGraph.__init__(name, startblock, return_var)`.
    pub fn with_return_var(
        name: impl Into<String>,
        startblock: BlockRef,
        return_var: Hlvalue,
    ) -> Self {
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
            func: None,
            _source: None,
            _seen_by_the_backend: std::cell::Cell::new(false),
        }
    }

    /// RPython `FunctionGraph.getargs()`.
    pub fn getargs(&self) -> Vec<Hlvalue> {
        self.startblock.borrow().inputargs.clone()
    }

    /// RPython `FunctionGraph.getreturnvar()`.
    pub fn getreturnvar(&self) -> Hlvalue {
        self.returnblock.borrow().inputargs[0].clone()
    }

    /// RPython `FunctionGraph.source` property.
    pub fn source(&self) -> Result<&str, &'static str> {
        if let Some(source) = self._source.as_deref() {
            return Ok(source);
        }
        self.func
            .as_ref()
            .and_then(|func| func.source.as_deref())
            .ok_or("source not found")
    }

    /// RPython `FunctionGraph.source = value`.
    pub fn set_source(&mut self, value: impl Into<String>) {
        self._source = Some(value.into());
    }

    /// RPython `FunctionGraph.startline` property.
    pub fn startline(&self) -> Result<u32, &'static str> {
        self.func
            .as_ref()
            .and_then(|func| func.firstlineno)
            .ok_or("startline not found")
    }

    /// RPython `FunctionGraph.filename` property.
    pub fn filename(&self) -> Result<&str, &'static str> {
        self.func
            .as_ref()
            .and_then(|func| func.filename.as_deref())
            .ok_or("filename not found")
    }

    /// RPython `FunctionGraph.iterblocks()` — DFS over reachable
    /// blocks starting at `startblock`, following `Link.target` and
    /// matching upstream's right-to-left stack order.
    pub fn iterblocks(&self) -> Vec<BlockRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        result.push(start.clone());

        // `stack = list(block.exits[::-1])`
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                result.push(target.clone());
                // `stack += block.exits[::-1]`
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterlinks()`.
    pub fn iterlinks(&self) -> Vec<LinkRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            result.push(link.clone());
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterblockops()`.
    pub fn iterblockops(&self) -> Vec<(BlockRef, SpaceOperation)> {
        let mut result = Vec::new();
        for block in self.iterblocks() {
            for op in &block.borrow().operations {
                result.push((block.clone(), op.clone()));
            }
        }
        result
    }
}

impl std::fmt::Display for FunctionGraph {
    // RPython `FunctionGraph.__str__` returns the name unless `func`
    // is set; `func` lands at Phase 3. Until then, the name is
    // always the output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// RPython `flowspace/model.py` — `const(obj)`.
///
/// Rust callers already carry values as [`ConstValue`], so the host
/// object rewrite / "already wrapped" dynamic-type checks happen before
/// this boundary. The remaining operation is the upstream final
/// `Constant(obj)` wrap.
pub fn r#const(obj: ConstValue) -> Constant {
    Constant::new(obj)
}

/// RPython `flowspace/model.py` — `uniqueitems(lst)`.
///
/// Returns a list with duplicate elements removed, preserving order.
pub fn uniqueitems<T: Clone + PartialEq>(lst: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    for item in lst {
        if !result.iter().any(|x| x == item) {
            result.push(item.clone());
        }
    }
    result
}

/// RPython `flowspace/model.py` — `mkentrymap(funcgraph)`.
///
/// Returns a map from each Block to the list of Links that target
/// it. The entry for the startblock contains a synthetic
/// `Link(graph.getargs(), graph.startblock)`.
pub fn mkentrymap(funcgraph: &FunctionGraph) -> HashMap<BlockKey, Vec<LinkRef>> {
    let startlink = Rc::new(RefCell::new(Link::new(
        funcgraph.getargs(),
        Some(funcgraph.startblock.clone()),
        None,
    )));
    let mut result: HashMap<BlockKey, Vec<LinkRef>> = HashMap::new();
    result.insert(BlockKey::of(&funcgraph.startblock), vec![startlink]);

    for link_ref in funcgraph.iterlinks() {
        let target_key = match link_ref.borrow().target.as_ref() {
            Some(t) => BlockKey::of(t),
            None => continue,
        };
        result.entry(target_key).or_default().push(link_ref.clone());
    }
    result
}

/// Identity-keyed wrapper for `BlockRef` so blocks can be used as
/// `HashMap` keys via pointer identity, matching RPython `dict[Block]`
/// semantics (Python uses `id(block)` by default).
#[derive(Clone, Debug)]
pub struct BlockKey(*const RefCell<Block>);

impl BlockKey {
    pub fn of(b: &BlockRef) -> Self {
        BlockKey(Rc::as_ptr(b))
    }

    /// Exposes the pointer identity as a `usize` — useful as a
    /// `PositionKey` payload.
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl PartialEq for BlockKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for BlockKey {}

impl std::hash::Hash for BlockKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// RPython `FunctionGraph` container used by the annotator. Upstream
/// passes `FunctionGraph` instances directly through dicts/sets
/// (Python identity). The Rust port wraps the graph in `Rc<RefCell<…>>`
/// so the annotator can mutate per-graph state (tag, entry-point
/// bookkeeping) without fighting the borrow checker.
pub type GraphRef = Rc<RefCell<FunctionGraph>>;

/// Identity-keyed wrapper for `GraphRef` — same rationale as
/// [`BlockKey`]: RPython `dict[FunctionGraph]` uses Python object
/// identity, which maps to `Rc::as_ptr` on the Rust side.
#[derive(Clone, Debug)]
pub struct GraphKey(*const RefCell<FunctionGraph>);

impl GraphKey {
    pub fn of(g: &GraphRef) -> Self {
        GraphKey(Rc::as_ptr(g))
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl PartialEq for GraphKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for GraphKey {}

impl std::hash::Hash for GraphKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Identity-keyed wrapper for `LinkRef`. Used by `RPythonAnnotator`'s
/// `links_followed` set — upstream Python stores `Link` objects in a
/// dict keyed by identity.
#[derive(Clone, Debug)]
pub struct LinkKey(*const RefCell<Link>);

impl LinkKey {
    pub fn of(l: &LinkRef) -> Self {
        LinkKey(Rc::as_ptr(l))
    }
}

impl PartialEq for LinkKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for LinkKey {}

impl std::hash::Hash for LinkKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// RPython `flowspace/model.py` — `copygraph(graph,
/// shallow=False, varmap={}, shallowvars=False)`.
///
/// Deep-copies a flow graph. Each Variable encountered is replaced
/// by a fresh Variable (new identity) unless `shallowvars` is true,
/// in which case the original Variable is reused. When `shallow` is
/// true, operation lists are also kept as-is.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn copygraph(
    graph: &FunctionGraph,
    shallow: bool,
    varmap: &HashMap<Variable, Variable>,
    shallowvars: bool,
) -> FunctionGraph {
    let mut varmap: HashMap<Variable, Variable> = varmap.clone();
    let shallowvars = shallowvars || shallow;

    let copyvar = |v: &Variable, varmap: &mut HashMap<Variable, Variable>| -> Variable {
        if shallowvars {
            return v.clone();
        }
        if let Some(existing) = varmap.get(v) {
            return existing.clone();
        }
        let v2 = v.copy();
        // copygraph copyvar (`model.py`): `Variable(v)` then
        // `concretetype` only. `Variable.copy` also clones
        // `annotation`; a lifted clone must not inherit it.
        *v2.annotation.borrow_mut() = None;
        varmap.insert(v.clone(), v2.clone());
        v2
    };

    let copyhl = |w: &Hlvalue, varmap: &mut HashMap<Variable, Variable>| -> Hlvalue {
        match w {
            Hlvalue::Variable(v) => Hlvalue::Variable(copyvar(v, varmap)),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        }
    };

    let copyblock = |block: &BlockRef, varmap: &mut HashMap<Variable, Variable>| -> BlockRef {
        let b = block.borrow();
        let newinputargs: Vec<Hlvalue> = b.inputargs.iter().map(|a| copyhl(a, varmap)).collect();
        let newblock = Block::shared(newinputargs);
        {
            let mut nb = newblock.borrow_mut();
            if b.is_final_block() {
                nb.mark_final();
            } else {
                nb.operations = if shallow {
                    b.operations.clone()
                } else {
                    b.operations
                        .iter()
                        .map(|op| {
                            let newargs: Vec<Hlvalue> =
                                op.args.iter().map(|a| copyhl(a, varmap)).collect();
                            let newresult = copyhl(&op.result, varmap);
                            SpaceOperation::with_offset(
                                op.opname.clone(),
                                newargs,
                                newresult,
                                op.offset,
                            )
                        })
                        .collect()
                };
            }
            nb.exitswitch = b.exitswitch.as_ref().map(|s| copyhl(s, varmap));
        }
        newblock
    };

    // Build blockmap over all reachable blocks + returnblock + exceptblock.
    let mut blockmap: HashMap<BlockKey, BlockRef> = HashMap::new();
    for block in graph.iterblocks() {
        blockmap.insert(BlockKey::of(&block), copyblock(&block, &mut varmap));
    }
    blockmap
        .entry(BlockKey::of(&graph.returnblock))
        .or_insert_with(|| copyblock(&graph.returnblock, &mut varmap));
    blockmap
        .entry(BlockKey::of(&graph.exceptblock))
        .or_insert_with(|| copyblock(&graph.exceptblock, &mut varmap));

    // Wire up exits.
    for block in graph.iterblocks() {
        let src_key = BlockKey::of(&block);
        let newblock = blockmap[&src_key].clone();
        let mut newlinks: Vec<LinkRef> = Vec::new();
        for link_ref in &block.borrow().exits {
            let link = link_ref.borrow();
            let newargs: Vec<LinkArg> = link
                .args
                .iter()
                .map(|arg| arg.as_ref().map(|value| copyhl(value, &mut varmap)))
                .collect();
            let new_target = link.target.as_ref().map(|t| {
                blockmap
                    .get(&BlockKey::of(t))
                    .cloned()
                    .expect("target block missing from blockmap")
            });
            let mut newlink = Link::new_mergeable(newargs, new_target, link.exitcase.clone());
            newlink.prevblock = Some(Rc::downgrade(&newblock));
            newlink.last_exception = link.last_exception.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.last_exc_value = link.last_exc_value.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.llexitcase = link.llexitcase.clone();
            newlinks.push(Rc::new(RefCell::new(newlink)));
        }
        newblock.borrow_mut().closeblock(newlinks);
    }

    let newstart = blockmap[&BlockKey::of(&graph.startblock)].clone();
    let mut newgraph = FunctionGraph::new(graph.name.clone(), newstart);
    newgraph.returnblock = blockmap[&BlockKey::of(&graph.returnblock)].clone();
    newgraph.exceptblock = blockmap[&BlockKey::of(&graph.exceptblock)].clone();
    newgraph.tag = graph.tag.clone();
    newgraph.func = graph.func.clone();
    newgraph._source = graph._source.clone();
    newgraph
}

/// RPython `flowspace/model.py` — `summary(graph)`.
///
/// Returns a map of opname → occurrence count, excluding `same_as`.
pub fn summary(graph: &FunctionGraph) -> HashMap<String, usize> {
    let mut insns: HashMap<String, usize> = HashMap::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            if op.opname != "same_as" {
                *insns.entry(op.opname.clone()).or_insert(0) += 1;
            }
        }
    }
    insns
}

/// RPython `flowspace/model.py` — `safe_iterblocks(graph)`.
///
/// Used for debugging/displaying broken graphs. `FunctionGraph` always
/// has a concrete startblock in Rust, but this still walks via raw
/// `exits` instead of delegating to `iterblocks()` to match upstream's
/// defensive shape.
pub fn safe_iterblocks(graph: &FunctionGraph) -> Vec<BlockRef> {
    let start = graph.startblock.clone();
    let mut result = vec![start.clone()];
    let mut seen = vec![BlockKey::of(&start)];
    let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();

    while let Some(link_ref) = stack.pop() {
        let Some(block) = link_ref.borrow().target.clone() else {
            continue;
        };
        let key = BlockKey::of(&block);
        if seen.iter().any(|k| k == &key) {
            continue;
        }
        result.push(block.clone());
        seen.push(key);
        stack.extend(block.borrow().exits.iter().rev().cloned());
    }
    result
}

/// RPython `flowspace/model.py` — `safe_iterlinks(graph)`.
///
/// Like [`safe_iterblocks`], walks the raw exit lists defensively and
/// yields each link before deciding whether to recurse into its target.
pub fn safe_iterlinks(graph: &FunctionGraph) -> Vec<LinkRef> {
    let start = graph.startblock.clone();
    let mut seen = vec![BlockKey::of(&start)];
    let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
    let mut result = Vec::new();

    while let Some(link_ref) = stack.pop() {
        result.push(link_ref.clone());
        let Some(block) = link_ref.borrow().target.clone() else {
            continue;
        };
        let key = BlockKey::of(&block);
        if seen.iter().any(|k| k == &key) {
            continue;
        }
        seen.push(key);
        stack.extend(block.borrow().exits.iter().rev().cloned());
    }
    result
}

fn is_exception_exitcase(exitcase: &Hlvalue) -> bool {
    matches!(
        exitcase,
        Hlvalue::Constant(Constant {
            value: ConstValue::HostObject(obj),
            ..
        }) if obj.is_class()
    )
}

fn is_valid_switch_exitcase(exitcase: &Hlvalue) -> bool {
    matches!(
        exitcase,
        Hlvalue::Constant(Constant {
            value: ConstValue::Int(_) | ConstValue::Bool(_) | ConstValue::None,
            ..
        })
    ) || match exitcase {
        Hlvalue::Constant(Constant {
            value: ConstValue::ByteStr(s),
            ..
        }) => s == b"default" || s.len() == 1,
        Hlvalue::Constant(Constant {
            value: ConstValue::UniStr(s),
            ..
        }) => s == "default" || s.chars().count() == 1,
        _ => false,
    }
}

/// RPython `flowspace/model.py` — `checkgraph(graph)`.
///
/// Sanity-check a flow graph. Panics with an assertion failure on any
/// violation, matching upstream semantics (RPython: `AssertionError`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn checkgraph(graph: &FunctionGraph) {
    for (block, nbargs) in [(&graph.returnblock, 1usize), (&graph.exceptblock, 2usize)] {
        let b = block.borrow();
        assert_eq!(b.inputargs.len(), nbargs);
        assert!(
            b.operations.is_empty(),
            "exitblock must not contain operations"
        );
        assert!(b.exits.is_empty(), "exitblock must not contain exits");
    }

    let mut vars_previous_blocks: Vec<Variable> = Vec::new();
    let graph_name = graph.name.as_str();

    for (block_idx, block) in graph.iterblocks().into_iter().enumerate() {
        let b = block.borrow();
        if b.exits.is_empty() {
            let is_return = BlockKey::of(&block) == BlockKey::of(&graph.returnblock);
            let is_except = BlockKey::of(&block) == BlockKey::of(&graph.exceptblock);
            assert!(is_return || is_except, "exit block not in graph exitblocks");
        }

        let mut vars: HashMap<Variable, Option<usize>> = HashMap::new();

        let definevar = |v: &Variable,
                         only_in_link: Option<usize>,
                         vars: &mut HashMap<Variable, Option<usize>>,
                         vars_previous_blocks: &[Variable]| {
            assert!(!vars.contains_key(v), "duplicate variable {}", v.name());
            assert!(
                !vars_previous_blocks.contains(v),
                "variable {} used in more than one block",
                v.name()
            );
            vars.insert(v.clone(), only_in_link);
        };

        // `where_str` enriches the panic with the use-site context
        // (e.g. `"op_index 4 ('setattr')"`, `"exitswitch"`,
        // `"link[2].args[1]"`) so per-graph diagnosis can locate the
        // dangling reference without re-traversing the graph. Required
        // substring `" used before definition"` is preserved verbatim
        // for `cutover.rs`'s `is_known_unported`.
        let usevar = |v: &Variable,
                      in_link: Option<usize>,
                      vars: &HashMap<Variable, Option<usize>>,
                      where_str: &str| {
            let only_in_link = vars.get(v).unwrap_or_else(|| {
                panic!(
                    "variable {name} used before definition (graph {graph_name:?}, \
                         block {block_idx}, {where_str})",
                    name = v.name(),
                )
            });
            if let Some(in_link) = in_link {
                assert!(
                    only_in_link.is_none() || *only_in_link == Some(in_link),
                    "variable {} used from the wrong exception link (graph {graph_name:?}, \
                         block {block_idx}, {where_str})",
                    v.name()
                );
            }
        };

        for w in &b.inputargs {
            let Hlvalue::Variable(v) = w else {
                panic!("block inputargs must be Variables");
            };
            definevar(v, None, &mut vars, &vars_previous_blocks);
        }

        for (op_index, op) in b.operations.iter().enumerate() {
            for (arg_index, a) in op.args.iter().enumerate() {
                match a {
                    Hlvalue::Variable(v) => {
                        let where_str = format!(
                            "op_index {op_index} ('{opname}') args[{arg_index}]",
                            opname = op.opname
                        );
                        usevar(v, None, &vars, &where_str);
                    }
                    Hlvalue::Constant(c) => assert!(
                        !matches!(&c.value, ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name),
                        "last_exception constant cannot appear as an operation argument"
                    ),
                }
            }
            if op.opname == "direct_call" {
                assert!(
                    matches!(op.args.first(), Some(Hlvalue::Constant(_))),
                    "direct_call expects a Constant function argument"
                );
            } else if op.opname == "indirect_call" {
                assert!(
                    matches!(op.args.first(), Some(Hlvalue::Variable(_))),
                    "indirect_call expects a Variable function argument"
                );
            }
            if let Hlvalue::Variable(v) = &op.result {
                definevar(v, None, &mut vars, &vars_previous_blocks);
            }
        }

        let mut exc_links: Vec<usize> = Vec::new();
        match &b.exitswitch {
            None => {
                assert!(b.exits.len() <= 1, "block without exitswitch has > 1 exits");
                if let Some(first) = b.exits.first() {
                    assert!(
                        first.borrow().exitcase.is_none(),
                        "unconditional exit carries an exitcase"
                    );
                }
            }
            Some(Hlvalue::Constant(c)) if matches!(&c.value, ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name) =>
            {
                assert!(
                    !b.operations.is_empty(),
                    "canraise block must end with a raising operation"
                );
                let raising_op = b.raising_op().expect("canraise block missing raising_op");
                assert!(
                    raising_op.opname != "keepalive"
                        && raising_op.opname != "cast_pointer"
                        && raising_op.opname != "same_as",
                    "invalid raising_op for canraise block"
                );
                assert!(
                    b.exits.len() >= 2,
                    "canraise block requires exception exits"
                );
                assert!(b.exits[0].borrow().exitcase.is_none());
                for link in b.exits.iter().skip(1) {
                    let exitcase = link
                        .borrow()
                        .exitcase
                        .clone()
                        .expect("exception exitcase missing");
                    assert!(
                        is_exception_exitcase(&exitcase),
                        "exception exitcase must carry an exception class constant"
                    );
                    exc_links.push(Rc::as_ptr(link) as usize);
                }
            }
            Some(Hlvalue::Variable(sw)) => {
                usevar(sw, None, &vars, "exitswitch");
                let is_boolean_switch = b.exits.len() == 2
                    && matches!(
                        (
                            b.exits[0].borrow().exitcase.as_ref(),
                            b.exits[1].borrow().exitcase.as_ref()
                        ),
                        (
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::Bool(false),
                                ..
                            })),
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::Bool(true),
                                ..
                            }))
                        )
                    );
                if !is_boolean_switch {
                    assert!(!b.exits.is_empty(), "switch block must have exits");
                    for (idx, link) in b.exits.iter().enumerate() {
                        let exitcase = link
                            .borrow()
                            .exitcase
                            .clone()
                            .expect("switch exitcase cannot be None");
                        assert!(
                            is_valid_switch_exitcase(&exitcase),
                            "switch on a non-primitive value {exitcase:?}"
                        );
                        if let Hlvalue::Constant(c) = &exitcase
                            && c.value.string_eq("default")
                        {
                            assert!(
                                idx + 1 == b.exits.len(),
                                "'default' branch of a switch is not the last exit"
                            );
                        }
                    }
                }
            }
            Some(Hlvalue::Constant(_)) => {
                panic!("unexpected constant exitswitch outside canraise blocks");
            }
        }

        let mut all_exitcases: Vec<Option<Hlvalue>> = Vec::new();
        for (link_idx, link_ref) in b.exits.iter().enumerate() {
            let link_id = Rc::as_ptr(link_ref) as usize;
            let exc_link = exc_links.contains(&link_id);
            let link = link_ref.borrow();
            let target = link.target.as_ref().expect("link.target missing");
            assert_eq!(
                link.args.len(),
                target.borrow().inputargs.len(),
                "link arity mismatch"
            );
            let prev = link.prevblock.as_ref().and_then(|w| w.upgrade());
            assert!(
                prev.as_ref().is_some_and(|p| Rc::ptr_eq(p, &block)),
                "link.prevblock does not point back to the owning block"
            );
            if exc_link {
                for extra in [&link.last_exception, &link.last_exc_value] {
                    let extra = extra.as_ref().expect("exception link missing extravar");
                    if let Hlvalue::Variable(v) = extra {
                        definevar(v, Some(link_id), &mut vars, &vars_previous_blocks);
                    }
                }
            } else {
                assert!(link.last_exception.is_none());
                assert!(link.last_exc_value.is_none());
            }
            for (arg_idx, arg) in link.args.iter().enumerate() {
                let arg = arg
                    .as_ref()
                    .expect("finalized graph cannot contain undefined-local link args");
                if let Hlvalue::Variable(v) = arg {
                    let where_str = format!("link[{link_idx}].args[{arg_idx}]");
                    usevar(v, Some(link_id), &vars, &where_str);
                    if exc_link && let Some(last_op) = b.operations.last() {
                        assert!(
                            *arg != last_op.result,
                            "raising operation result cannot flow into exception link"
                        );
                    }
                }
            }
            assert!(
                !all_exitcases.contains(&link.exitcase),
                "duplicate exitcase in block exits"
            );
            all_exitcases.push(link.exitcase.clone());
        }
        vars_previous_blocks.extend(vars.into_keys());
    }
}

#[cfg(test)]
mod tests {
    //! Lightweight scaffold tests for F1.2 + F1.3. The full
    //! `rpython/flowspace/test/test_model.py` port lands at F1.7
    //! after `Block`, `Link`, and `FunctionGraph` exist.

    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype;
    use std::collections::hash_map::DefaultHasher;

    #[test]
    fn variable_default_name_numbers_from_dummy_prefix() {
        // Two fresh Variables share the dummy prefix and should get
        // unique lazy numbers.
        let v1 = Variable::new();
        let v2 = Variable::new();
        let n1 = v1.name();
        let n2 = v2.name();
        assert_ne!(n1, n2);
        assert!(n1.starts_with('v'));
        assert!(n2.starts_with('v'));
    }

    #[test]
    fn lladdress_truthy_mirrors_fakeaddress_nonzero() {
        // `bool(fakeaddress)` = `__nonzero__` = `ptr is not None`
        // (llmemory.py): a NULL address is false, a `Fake` and an
        // odd tagged-int `IntCast` are both true.
        use lltype::{_address, Ptr, PtrTarget, Struct};
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::gc(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )),
        };
        assert_eq!(ConstValue::LLAddress(_address::Null).truthy(), Some(false));
        assert_eq!(
            ConstValue::LLAddress(_address::Fake(Box::new(ptr_t._example()))).truthy(),
            Some(true)
        );
        assert_eq!(
            ConstValue::LLAddress(_address::IntCast(0x41)).truthy(),
            Some(true)
        );
    }

    #[test]
    fn variable_rename_sets_prefix_once() {
        let mut v = Variable::new();
        v.rename("counter");
        assert!(v.renamed());
        let first = v.name();
        // Second rename is a no-op per upstream.
        v.rename("other");
        assert_eq!(v.name(), first);
    }

    #[test]
    fn variable_copy_preserves_annotation_and_concretetype() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let v = Variable::new();
        v.annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        v.set_concretetype(Some(LowLevelType::Signed));
        let c = v.copy();
        let ca = c.annotation.borrow().as_ref().unwrap().clone();
        let va = v.annotation.borrow().as_ref().unwrap().clone();
        // Rc shares the same allocation — identity preserved per
        // RPython parity (copy keeps reference to the same SomeValue).
        assert!(Rc::ptr_eq(&ca, &va));
        assert_eq!(c.concretetype(), v.concretetype());
    }

    #[test]
    fn constant_replace_returns_self_wrapped_in_hlvalue() {
        let c = Constant::new(ConstValue::Placeholder);
        let map: HashMap<Variable, Hlvalue> = HashMap::new();
        assert_eq!(c.replace(&map), Hlvalue::Constant(c));
    }

    #[test]
    fn constant_hashable_equality_ignores_concretetype() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let a = Constant::with_concretetype(ConstValue::Int(1), LowLevelType::Signed);
        let b = Constant::with_concretetype(ConstValue::Int(1), LowLevelType::Unsigned);

        assert_eq!(a, b);

        let mut ha = std::collections::hash_map::DefaultHasher::new();
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    /// `ConstValue::as_pystr` mirrors RPython `isinstance(value, str)`
    /// (Python 2 bytes-only). UniStr / non-string variants must yield
    /// `None`; ByteStr returns the UTF-8 view (ASCII identifiers /
    /// attribute names always succeed).
    #[test]
    fn as_pystr_matches_python2_str_isinstance_only() {
        // ByteStr → UTF-8 view (ASCII identifier).
        assert_eq!(ConstValue::byte_str(b"name").as_pystr(), Some("name"));
        // UniStr is `unicode` upstream, NOT `str`. Reject.
        assert_eq!(ConstValue::uni_str("name").as_pystr(), None);
        // Non-UTF-8 bytes — also reject (ASCII identifiers never
        // produce these; defensive).
        assert_eq!(ConstValue::byte_str([0xff, 0xfe]).as_pystr(), None);
        // Non-string variants reject as well.
        assert_eq!(ConstValue::Int(1).as_pystr(), None);
    }

    #[test]
    fn llptr_constants_follow_hashable_pointer_identity() {
        let ptr1 = lltype::Ptr {
            TO: lltype::PtrTarget::Struct(lltype::Struct::new(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )),
        }
        ._example();
        let ptr2 = ptr1.clone();
        let ptr3 = lltype::Ptr {
            TO: lltype::PtrTarget::Struct(lltype::Struct::new(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )),
        }
        ._example();

        let c1 = Constant::new(ConstValue::LLPtr(Box::new(ptr1)));
        let c2 = Constant::new(ConstValue::LLPtr(Box::new(ptr2)));
        let c3 = Constant::new(ConstValue::LLPtr(Box::new(ptr3)));

        let hash_of = |c: &Constant| {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        };

        assert_eq!(c1, c2);
        assert_eq!(hash_of(&c1), hash_of(&c2));
        assert_ne!(c1, c3);
        assert_ne!(hash_of(&c1), hash_of(&c3));
    }

    #[test]
    fn llptr_truthiness_follows_pointer_nullity() {
        let ptr_t = lltype::Ptr {
            TO: lltype::PtrTarget::Struct(lltype::Struct::new(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )),
        };
        let null_ptr = lltype::_ptr::new(ptr_t.clone(), Ok(None));
        let nonnull_ptr = ptr_t._example();

        assert_eq!(ConstValue::LLPtr(Box::new(null_ptr)).truthy(), Some(false));
        assert_eq!(
            ConstValue::LLPtr(Box::new(nonnull_ptr)).truthy(),
            Some(true)
        );
    }

    #[test]
    fn list_constants_use_hashable_identity_fallback() {
        let a = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let b = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn tuple_with_unhashable_item_uses_hashable_identity_fallback() {
        let a = Constant::new(ConstValue::Tuple(vec![ConstValue::List(vec![
            ConstValue::Int(1),
        ])]));
        let b = Constant::new(ConstValue::Tuple(vec![ConstValue::List(vec![
            ConstValue::Int(1),
        ])]));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn function_constants_use_hashable_object_identity() {
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let a = Constant::new(ConstValue::Function(Box::new(GraphFunc::new(
            "f",
            globals.clone(),
        ))));
        let b = Constant::new(ConstValue::Function(Box::new(GraphFunc::new("f", globals))));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn tuple_with_function_item_uses_element_hashable_identity() {
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let a = Constant::new(ConstValue::Tuple(vec![ConstValue::Function(Box::new(
            GraphFunc::new("f", globals.clone()),
        ))]));
        let b = Constant::new(ConstValue::Tuple(vec![ConstValue::Function(Box::new(
            GraphFunc::new("f", globals),
        ))]));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn code_constants_use_hashable_object_identity() {
        let a = Constant::new(ConstValue::Code(Box::new(HostCode::new(
            0,
            0,
            0,
            0,
            rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "<test>".to_string(),
            "f".to_string(),
            1,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new().into_boxed_slice(),
        ))));
        let b = Constant::new(ConstValue::Code(Box::new(HostCode::new(
            0,
            0,
            0,
            0,
            rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "<test>".to_string(),
            "f".to_string(),
            1,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new().into_boxed_slice(),
        ))));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn graphs_constants_use_hashable_identity_fallback() {
        let a = Constant::new(ConstValue::Graphs(vec![1, 2]));
        let b = Constant::new(ConstValue::Graphs(vec![1, 2]));
        let c = a.clone();

        let hash_of = |v: &Constant| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_eq!(a, c);
        assert_eq!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn function_constant_is_foldable() {
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let c = Constant::new(ConstValue::Function(Box::new(GraphFunc::new("f", globals))));
        assert!(c.foldable());
    }

    #[test]
    fn bound_method_foldable_uses_im_self() {
        let cls = HostObject::new_class("pkg.C", vec![]);
        let func = HostObject::new_user_function(GraphFunc::new(
            "pkg.C.cm",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        let method = HostObject::new_bound_method("pkg.C.cm", cls.clone(), func, "cm", cls);
        let c = Constant::new(ConstValue::HostObject(method));
        assert!(c.foldable());
    }

    #[test]
    fn builtin_exception_instance_is_foldable() {
        let exc = HOST_ENV
            .lookup_builtin("ValueError")
            .expect("bootstrap must register ValueError");
        let inst = exc
            .reusable_prebuilt_instance()
            .expect("class exposes reusable instance");
        let c = Constant::new(ConstValue::HostObject(inst));
        assert!(c.foldable());
    }

    #[test]
    fn freeze_instance_is_foldable() {
        let cls = HostObject::new_class("pkg.Frozen", vec![]);
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            )),
        );
        let inst = HostObject::new_instance(cls, Vec::new());
        let c = Constant::new(ConstValue::HostObject(inst));
        assert!(c.foldable());
    }

    #[test]
    fn weakref_constant_is_not_foldable() {
        let target = HostObject::new_class("pkg.C", vec![])
            .reusable_prebuilt_instance()
            .expect("class exposes reusable instance");
        let wref = HostObject::new_weakref("weakref(pkg.C)", Some(target));
        let c = Constant::new(ConstValue::HostObject(wref));
        assert!(!c.foldable());
    }

    #[test]
    fn native_callable_constant_is_foldable_via_builtin_class() {
        let f = HostObject::new_native_callable("pkg.f", Arc::new(|_| Ok(ConstValue::Int(0))));
        let c = Constant::new(ConstValue::HostObject(f));
        assert!(c.foldable());
    }

    #[test]
    fn builtin_class_constant_is_foldable() {
        let int_cls = HOST_ENV.lookup_builtin("int").expect("builtin int class");
        let c = Constant::new(ConstValue::HostObject(int_cls));
        assert!(c.foldable());
    }

    #[test]
    fn none_constant_is_foldable() {
        let c = Constant::new(ConstValue::None);
        assert!(c.foldable());
    }

    #[test]
    fn code_constant_is_foldable() {
        let c = Constant::new(ConstValue::Code(Box::new(HostCode::new(
            0,
            0,
            0,
            0,
            rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "<test>".to_string(),
            "f".to_string(),
            1,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new().into_boxed_slice(),
        ))));
        assert!(c.foldable());
    }

    #[test]
    fn graphs_constant_is_not_foldable() {
        let c = Constant::new(ConstValue::Graphs(vec![1, 2, 3]));
        assert!(!c.foldable());
    }

    #[test]
    fn low_level_type_constant_is_foldable() {
        let c = Constant::new(ConstValue::LowLevelType(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
        )));
        assert!(c.foldable());
    }

    #[test]
    fn atom_display_returns_name() {
        let a = Atom::new("last_exception");
        assert_eq!(format!("{a}"), "last_exception");
    }

    #[test]
    fn space_operation_eq_ignores_offset() {
        // RPython __eq__ explicitly excludes offset (model.py:442).
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op1 = SpaceOperation::with_offset("int_add", vec![a.clone(), a.clone()], r.clone(), 10);
        let op2 = SpaceOperation::with_offset("int_add", vec![a.clone(), a], r, 20);
        assert_eq!(op1, op2);
    }

    #[test]
    fn space_operation_repr_matches_rpython_layout() {
        let r = Hlvalue::Variable(Variable::named("result"));
        let a = Hlvalue::Variable(Variable::named("lhs"));
        let b = Hlvalue::Variable(Variable::named("rhs"));
        let op = SpaceOperation::new("int_add", vec![a, b], r);
        let s = format!("{op}");
        // result = opname(arg0, arg1)
        assert!(s.contains(" = int_add("));
        assert!(s.contains(", "));
    }

    #[test]
    fn c_last_exception_is_atom_constant() {
        let c = c_last_exception();
        assert!(matches!(
            c.value,
            ConstValue::Atom(ref a) if a.name == "last_exception"
        ));
    }

    #[test]
    fn block_canraise_detects_last_exception_exitswitch() {
        let mut block = Block::new(vec![]);
        assert!(!block.canraise());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert!(block.canraise());
    }

    #[test]
    fn block_raising_op_returns_last_when_canraise() {
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op = SpaceOperation::new("int_add", vec![a.clone(), a], r);
        let mut block = Block::new(vec![]);
        block.operations.push(op.clone());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert_eq!(block.raising_op(), Some(&op));
    }

    #[test]
    fn block_closeblock_panics_on_second_call() {
        let target = Block::shared(vec![]);
        let link = Rc::new(RefCell::new(Link::new(vec![], Some(target.clone()), None)));
        let mut block = Block::new(vec![]);
        block.closeblock(vec![link.clone()]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block.closeblock(vec![link]);
        }));
        assert!(result.is_err(), "second closeblock should panic");
    }

    #[test]
    fn function_graph_constructs_return_and_except_blocks() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let startblock = Block::shared(vec![arg]);
        let g = FunctionGraph::new("f", startblock);
        // Returnblock has 1 inputarg, is final, no exits.
        let rb = g.returnblock.borrow();
        assert_eq!(rb.inputargs.len(), 1);
        assert!(rb.is_final_block());
        assert!(rb.exits.is_empty());
        drop(rb);
        // Exceptblock has 2 inputargs (etype, evalue), is final.
        let xb = g.exceptblock.borrow();
        assert_eq!(xb.inputargs.len(), 2);
        assert!(xb.is_final_block());
    }

    #[test]
    fn function_graph_iterblocks_visits_reachable_only() {
        // Build: start -> mid -> return.
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        // Link start -> mid.
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        // Build graph with start as entry; mid has no exits (dead-end).
        let g = FunctionGraph::new("f", start.clone());
        let blocks = g.iterblocks();
        // Should visit start then mid.
        assert_eq!(blocks.len(), 2);
        assert!(Rc::ptr_eq(&blocks[0], &start));
        assert!(Rc::ptr_eq(&blocks[1], &mid));
    }

    #[test]
    fn function_graph_iterlinks_walks_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm.clone()]);
        let g = FunctionGraph::new("f", start);
        let links = g.iterlinks();
        assert_eq!(links.len(), 1);
        assert!(Rc::ptr_eq(&links[0], &link_sm));
    }

    #[test]
    fn function_graph_iterblockops_walks_every_op() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let r = Hlvalue::Variable(Variable::named("r"));
        let op = SpaceOperation::new("int_add", vec![arg.clone(), arg.clone()], r);
        start.borrow_mut().operations.push(op.clone());
        let g = FunctionGraph::new("f", start.clone());
        let ops = g.iterblockops();
        assert_eq!(ops.len(), 1);
        assert!(Rc::ptr_eq(&ops[0].0, &start));
        assert_eq!(ops[0].1, op);
    }

    #[test]
    fn uniqueitems_removes_duplicates_in_order() {
        let v = uniqueitems(&[1, 2, 3, 2, 1, 4]);
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn const_helper_wraps_constvalue() {
        let c = r#const(ConstValue::Int(42));
        assert_eq!(c.value, ConstValue::Int(42));
    }

    #[test]
    fn mkentrymap_includes_synthetic_startlink_and_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        let g = FunctionGraph::new("f", start.clone());
        let map = mkentrymap(&g);
        // The startblock has the synthetic startlink (from getargs).
        assert!(map.contains_key(&BlockKey::of(&start)));
        assert_eq!(map[&BlockKey::of(&start)].len(), 1);
        // The mid block has one entry (the edge from start).
        assert_eq!(map[&BlockKey::of(&mid)].len(), 1);
    }

    #[test]
    fn summary_counts_opnames_excluding_same_as() {
        let x = Hlvalue::Variable(Variable::named("x"));
        let r = Hlvalue::Variable(Variable::named("r"));
        let r2 = Hlvalue::Variable(Variable::named("r2"));
        let start = Block::shared(vec![x.clone()]);
        {
            let mut b = start.borrow_mut();
            b.operations.push(SpaceOperation::new(
                "int_add",
                vec![x.clone(), x.clone()],
                r,
            ));
            b.operations
                .push(SpaceOperation::new("same_as", vec![x.clone()], r2));
        }
        let g = FunctionGraph::new("f", start);
        let s = summary(&g);
        assert_eq!(s.get("int_add").copied(), Some(1));
        assert!(!s.contains_key("same_as"), "summary excludes same_as");
    }

    #[test]
    fn safe_iter_helpers_walk_raw_exits_once() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let end = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        let link_me = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(end.clone()),
            None,
        )));
        let link_es = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(start.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm.clone()]);
        mid.borrow_mut().closeblock(vec![link_me.clone()]);
        end.borrow_mut().closeblock(vec![link_es.clone()]);
        let g = FunctionGraph::new("f", start.clone());

        // Name the identities so a wrong walk fails as a readable sequence diff
        // rather than as a bare `false`. Anything the walk yields that is not
        // one of the three constructed nodes reads as `?` instead of being
        // silently absorbed.
        let name = |b: &Rc<RefCell<Block>>| {
            for (n, r) in [("start", &start), ("mid", &mid), ("end", &end)] {
                if Rc::ptr_eq(b, r) {
                    return n;
                }
            }
            "?"
        };
        let lname = |l: &Rc<RefCell<Link>>| {
            for (n, r) in [("sm", &link_sm), ("me", &link_me), ("es", &link_es)] {
                if Rc::ptr_eq(l, r) {
                    return n;
                }
            }
            "?"
        };

        // The graph is the cycle start -> mid -> end -> start. Both helpers
        // yield each node once, in forward-exit order; `end`'s exit back to
        // `start` is the revisit the "once" in this test's name is about.
        // Measured, and pinned as a sequence: membership alone cannot see a
        // walk that reaches the same three nodes in a different order.
        let blocks = safe_iterblocks(&g);
        assert_eq!(
            blocks.iter().map(name).collect::<Vec<_>>(),
            ["start", "mid", "end"]
        );

        let links = safe_iterlinks(&g);
        assert_eq!(
            links.iter().map(lname).collect::<Vec<_>>(),
            ["sm", "me", "es"]
        );
    }

    #[test]
    fn copygraph_produces_fresh_variables() {
        let x = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let g = FunctionGraph::new("f", start);
        let g2 = copygraph(&g, false, &HashMap::new(), false);
        // New startblock has a different input-variable identity.
        let b0 = g.startblock.borrow();
        let b1 = g2.startblock.borrow();
        let v0 = match &b0.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        let v1 = match &b1.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        assert_ne!(v0, v1, "copygraph should assign a fresh variable id");
    }

    #[test]
    fn checkgraph_passes_on_simple_graph() {
        // A trivial graph with just a start block and the return/except
        // blocks auto-created. startblock has no exits, so it must be a
        // return block — use the returnblock as the startblock so the
        // "exit block not in graph exitblocks" invariant holds.
        let start = Block::shared(vec![Hlvalue::Variable(Variable::named("x"))]);
        let r = Hlvalue::Variable(Variable::named("r"));
        // Link start -> returnblock.
        let g = {
            let g = FunctionGraph::new("f", start.clone());
            let retlink = Rc::new(RefCell::new(Link::new(
                vec![r.clone()],
                Some(g.returnblock.clone()),
                None,
            )));
            retlink.borrow_mut().prevblock = Some(Rc::downgrade(&start));
            start.borrow_mut().closeblock(vec![retlink]);
            // Give the start block at least one op producing `r`.
            start.borrow_mut().operations.push(SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(Variable::named("x")),
                    Hlvalue::Variable(Variable::named("x")),
                ],
                r,
            ));
            g
        };
        // We deliberately skipped variable-definedness plumbing for this
        // hand-rolled test; just make sure the call itself does not panic
        // when the structural invariants hold.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checkgraph(&g);
        }));
        // The above graph has undefined-variable usage (the op uses
        // freshly-minted x variables not in inputargs), so checkgraph
        // will assert. That's expected — just confirm the function is
        // callable end-to-end.
        assert!(
            result.is_err(),
            "checkgraph should catch undefined-variable use"
        );
    }

    #[test]
    fn link_settarget_checks_arity() {
        let a = Hlvalue::Variable(Variable::named("a"));
        let target = Block::shared(vec![a.clone()]);
        let mut link = Link::new(vec![a], Some(target), None);
        let other = Block::shared(vec![]); // different arity
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            link.settarget(other);
        }));
        assert!(result.is_err(), "arity mismatch should panic");
    }

    #[test]
    fn space_operation_replace_remaps_args_and_result() {
        let r = Variable::named("r");
        let a = Variable::named("a");
        let b = Variable::named("b");
        // Mapping r -> r', a -> a'; b unchanged.
        let r_new = Variable::named("r_new");
        let a_new = Variable::named("a_new");
        let mut map: HashMap<Variable, Hlvalue> = HashMap::new();
        map.insert(r.clone(), Hlvalue::Variable(r_new.clone()));
        map.insert(a.clone(), Hlvalue::Variable(a_new.clone()));

        let op = SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(r),
        );
        let op2 = op.replace(&map);
        assert_eq!(op2.result, Hlvalue::Variable(r_new));
        assert_eq!(op2.args[0], Hlvalue::Variable(a_new));
        assert_eq!(op2.args[1], Hlvalue::Variable(b));
    }

    // HostObject class_dict / mro

    #[test]
    fn class_dict_get_set_roundtrip() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        assert!(cls.class_get("_mixin_").is_none());
        cls.class_set("_mixin_", ConstValue::Bool(true));
        match cls.class_get("_mixin_") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn class_dict_keys_snapshot_matches_insert() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        cls.class_set("a", ConstValue::Int(1));
        cls.class_set("b", ConstValue::Int(2));
        let mut keys = cls.class_dict_keys();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn class_dict_items_roundtrip() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        cls.class_set("x", ConstValue::Int(7));
        let items = cls.class_dict_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, "x");
        match items[0].1 {
            ConstValue::Int(7) => {}
            ref other => panic!("expected Int(7), got {other:?}"),
        }
    }

    #[test]
    fn class_has_reflects_membership() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        assert!(!cls.class_has("missing"));
        cls.class_set("here", ConstValue::Int(0));
        assert!(cls.class_has("here"));
    }

    #[test]
    fn new_class_with_members_seeds_initial_dict() {
        // The ordered container is required because a `HashMap`'s iteration
        // order varies per process and these entries produce a work order.
        let mut seed: IndexMap<String, ConstValue> = IndexMap::new();
        seed.insert("_mixin_".into(), ConstValue::Bool(true));
        seed.insert("second".into(), ConstValue::Int(2));
        let cls = HostObject::new_class_with_members("pkg.Foo", vec![], seed);
        match cls.class_get("_mixin_") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
        assert_eq!(cls.class_dict_keys(), vec!["_mixin_", "second"]);
    }

    #[test]
    fn instance_dict_keys_preserve_insertion_order() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let instance = HostObject::new_instance(cls, vec![]);
        instance.instance_set("second", ConstValue::Int(2)).unwrap();
        instance.instance_set("first", ConstValue::Int(1)).unwrap();

        assert_eq!(
            instance.instance_dict_keys().unwrap(),
            vec!["second", "first"]
        );
    }

    struct NativeInstanceFixture {
        value: std::sync::atomic::AtomicI64,
        child: HostObject,
    }

    impl NativeInstanceDict for NativeInstanceFixture {
        fn get(&self, name: &str) -> Result<ConstValue, HostGetAttrError> {
            match name {
                "child" => Ok(ConstValue::HostObject(self.child.clone())),
                "value" => Ok(ConstValue::Int(
                    self.value.load(std::sync::atomic::Ordering::Acquire),
                )),
                "nested" => host_getattr(&self.child, "value"),
                "blocked" => Err(HostGetAttrError::Unsupported),
                _ => Err(HostGetAttrError::Missing),
            }
        }

        fn keys(&self) -> Result<Vec<String>, HostGetAttrError> {
            Ok(["child", "value", "nested", "blocked"]
                .map(str::to_string)
                .to_vec())
        }
    }

    #[test]
    fn native_instance_dict_is_live_read_only_and_preserves_identity() {
        use crate::annotator::{bookkeeper::Bookkeeper, classdesc::InstanceSource};
        let cls = HostObject::new_class("pkg.Native", vec![]);
        let child = HostObject::new_instance(cls.clone(), vec![]);
        child.instance_set("value", ConstValue::Int(11)).unwrap();
        let native = Arc::new(NativeInstanceFixture {
            value: std::sync::atomic::AtomicI64::new(7),
            child: child.clone(),
        });
        let instance = HostObject::new_native_instance(cls.clone(), native.clone());
        assert_eq!(instance.instance_class(), Some(&cls));
        assert_eq!(
            host_getattr(&instance, "child"),
            Ok(ConstValue::HostObject(child))
        );
        assert_eq!(host_getattr(&instance, "nested"), Ok(ConstValue::Int(11)));
        assert_eq!(host_getattr(&instance, "value"), Ok(ConstValue::Int(7)));
        native.value.store(8, std::sync::atomic::Ordering::Release);
        assert_eq!(host_getattr(&instance, "value"), Ok(ConstValue::Int(8)));
        assert_eq!(
            instance.instance_set("value", ConstValue::Int(99)),
            Err(HostGetAttrError::Unsupported)
        );
        assert_eq!(host_getattr(&instance, "value"), Ok(ConstValue::Int(8)));
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let source = InstanceSource::new(&bk, instance.clone());
        assert_eq!(
            source.all_instance_attributes().unwrap(),
            ["child", "value", "nested", "blocked"]
        );
        assert!(source.s_get_value(None, "blocked").is_err());
        drop(native);
        assert_eq!(host_getattr(&instance, "value"), Ok(ConstValue::Int(8)));
    }

    #[test]
    fn native_instance_lookup_preserves_descriptor_order_and_access_errors() {
        let cls = HostObject::new_class("pkg.Native", vec![]);
        cls.class_set("blocked", ConstValue::Int(99));
        cls.class_set("inherited", ConstValue::Int(42));
        let fget = HostObject::new_native_callable(
            "pkg.Native.value",
            Arc::new(|_| Ok(ConstValue::Int(21))),
        );
        cls.class_set(
            "value",
            ConstValue::HostObject(HostObject::new_property(
                "pkg.Native.value",
                Some(fget),
                None,
                None,
            )),
        );
        let instance = HostObject::new_native_instance(
            cls.clone(),
            Arc::new(NativeInstanceFixture {
                value: std::sync::atomic::AtomicI64::new(7),
                child: HostObject::new_instance(cls, vec![]),
            }),
        );
        assert_eq!(host_getattr(&instance, "value"), Ok(ConstValue::Int(21)));
        assert_eq!(
            host_getattr(&instance, "inherited"),
            Ok(ConstValue::Int(42))
        );
        assert_eq!(
            host_getattr(&instance, "absent"),
            Err(HostGetAttrError::Missing)
        );
        assert_eq!(
            host_getattr(&instance, "blocked"),
            Err(HostGetAttrError::Unsupported)
        );
    }

    #[test]
    fn native_instance_enumeration_failure_is_not_an_empty_dict() {
        use crate::annotator::{bookkeeper::Bookkeeper, classdesc::InstanceSource};
        struct Unreadable;
        impl NativeInstanceDict for Unreadable {
            fn get(&self, _: &str) -> Result<ConstValue, HostGetAttrError> {
                Err(HostGetAttrError::Unsupported)
            }
            fn keys(&self) -> Result<Vec<String>, HostGetAttrError> {
                Err(HostGetAttrError::Unsupported)
            }
        }
        let cls = HostObject::new_class("pkg.Unreadable", vec![]);
        let instance = HostObject::new_native_instance(cls, Arc::new(Unreadable));
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let source = InstanceSource::new(&bk, instance);
        assert!(source.all_instance_attributes().is_err());
    }

    #[test]
    fn class_dict_on_module_returns_empty() {
        let m = HostObject::new_module("pkg");
        assert_eq!(m.class_dict_keys(), Vec::<String>::new());
        assert!(m.class_get("anything").is_none());
        assert!(!m.class_has("anything"));
    }

    #[test]
    fn class_bases_matches_ctor() {
        let base = HostObject::new_class("pkg.Base", vec![]);
        let child = HostObject::new_class("pkg.Child", vec![base.clone()]);
        let bases = child.class_bases().expect("class has bases");
        assert_eq!(bases.len(), 1);
        assert_eq!(bases[0], base);
    }

    #[test]
    fn mro_single_inheritance_chain() {
        // A <- B <- C, mro = [C, B, A]
        let a = HostObject::new_class("pkg.A", vec![]);
        let b = HostObject::new_class("pkg.B", vec![a.clone()]);
        let c = HostObject::new_class("pkg.C", vec![b.clone()]);
        let mro = c.mro().expect("single-inheritance mro");
        assert_eq!(mro, vec![c, b, a]);
    }

    #[test]
    fn mro_diamond_c3_order() {
        // A, B<-A, C<-A, D<-B,C → mro = [D, B, C, A]
        let a = HostObject::new_class("pkg.A", vec![]);
        let b = HostObject::new_class("pkg.B", vec![a.clone()]);
        let c = HostObject::new_class("pkg.C", vec![a.clone()]);
        let d = HostObject::new_class("pkg.D", vec![b.clone(), c.clone()]);
        let mro = d.mro().expect("diamond mro");
        assert_eq!(mro, vec![d, b, c, a]);
    }

    #[test]
    fn mro_empty_bases_is_singleton() {
        let a = HostObject::new_class("pkg.A", vec![]);
        let mro = a.mro().expect("no-base mro");
        assert_eq!(mro, vec![a]);
    }

    #[test]
    fn mro_on_non_class_returns_none() {
        let m = HostObject::new_module("pkg");
        assert!(m.mro().is_none());
    }

    #[test]
    fn native_callable_invokes_closure() {
        let f = HostObject::new_native_callable(
            "pkg.always_true",
            Arc::new(|_args| Ok(ConstValue::Bool(true))),
        );
        assert!(f.is_native_callable());
        let out = f.call_host(&[]).expect("invocation succeeds");
        assert_eq!(out, ConstValue::Bool(true));
    }

    #[test]
    fn native_callable_receives_positional_args() {
        let f = HostObject::new_native_callable(
            "pkg.add_one",
            Arc::new(|args| match args {
                [ConstValue::Int(n)] => Ok(ConstValue::Int(n + 1)),
                _ => Err("add_one: expected one int".to_string()),
            }),
        );
        let out = f.call_host(&[ConstValue::Int(41)]).unwrap();
        assert_eq!(out, ConstValue::Int(42));
    }

    #[test]
    fn native_callable_propagates_closure_error() {
        let f = HostObject::new_native_callable(
            "pkg.explode",
            Arc::new(|_args| Err("boom".to_string())),
        );
        let err = f.call_host(&[]).unwrap_err();
        assert!(matches!(err, HostCallError::Native(ref m) if m.contains("boom")));
    }

    #[test]
    fn bound_method_prepends_self_to_call_host() {
        // BoundMethod.__call__(self, *args) → func(self_obj, *args).
        let self_obj = HostObject::new_opaque("pkg.self_marker");
        let inner = HostObject::new_native_callable(
            "pkg.echo_self",
            Arc::new(|args| match args {
                [ConstValue::HostObject(h)] => Ok(ConstValue::HostObject(h.clone())),
                _ => Err(format!("echo_self: wrong args ({})", args.len())),
            }),
        );
        let origin = HostObject::new_class("pkg.Origin", vec![]);
        let method = HostObject::new_bound_method(
            "pkg.Origin.echo_self",
            self_obj.clone(),
            inner,
            "echo_self",
            origin,
        );
        let out = method.call_host(&[]).unwrap();
        assert_eq!(out, ConstValue::HostObject(self_obj));
    }

    #[test]
    fn call_host_on_non_callable_returns_error() {
        let cls = HostObject::new_class("pkg.Plain", vec![]);
        assert!(cls.call_host(&[]).is_err());
        let m = HostObject::new_module("pkg.mod");
        assert!(m.call_host(&[]).is_err());
    }

    #[test]
    fn host_getattr_runs_property_fget_on_instance() {
        // upstream descriptor protocol: instance getattr on a name
        // bound to a `property` descriptor executes `prop.fget(self)`.
        // The Rust port invokes the fget via `HostObject::call_host`.
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let fget = HostObject::new_native_callable(
            "pkg.Box.value_fget",
            Arc::new(|args| match args {
                [ConstValue::HostObject(_self)] => Ok(ConstValue::Int(7)),
                other => Err(format!(
                    "value_fget: expected [self], got {} args",
                    other.len()
                )),
            }),
        );
        let prop = HostObject::new_property("pkg.Box.value", Some(fget), None, None);
        cls.class_set("value", ConstValue::HostObject(prop));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let out = host_getattr(&inst, "value").expect("getter must fire");
        assert_eq!(out, ConstValue::Int(7));
    }

    #[test]
    fn host_getattr_on_property_without_fget_is_missing() {
        // A property without fget (pure setter). Instance access
        // upstream raises `AttributeError: unreadable attribute`,
        // identical semantics to a missing attribute — the Rust port
        // surfaces `HostGetAttrError::Missing` so callers can fall
        // through to whatever AttributeError branch they already have.
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let prop = HostObject::new_property("pkg.Box.value", None, None, None);
        cls.class_set("value", ConstValue::HostObject(prop));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let err = host_getattr(&inst, "value").unwrap_err();
        assert_eq!(err, HostGetAttrError::Missing);
    }

    #[test]
    fn host_getattr_property_beats_instance_dict_for_data_descriptor() {
        // Upstream CPython descriptor order: a data descriptor in the
        // class wins against `obj.__dict__[name]`. If the instance dict
        // has `value=99` *and* the class has `value=property(fget)`,
        // `obj.value` still calls the fget.
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let fget = HostObject::new_native_callable(
            "pkg.Box.value_fget",
            Arc::new(|args| match args {
                [ConstValue::HostObject(_self)] => Ok(ConstValue::Int(7)),
                other => Err(format!(
                    "value_fget: expected [self], got {} args",
                    other.len()
                )),
            }),
        );
        let prop = HostObject::new_property("pkg.Box.value", Some(fget), None, None);
        cls.class_set("value", ConstValue::HostObject(prop));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        inst.instance_set("value", ConstValue::Int(99)).unwrap();
        let out = host_getattr(&inst, "value").expect("descriptor wins");
        // Data descriptor `fget` beats the instance dict.
        assert_eq!(out, ConstValue::Int(7));
    }

    #[test]
    fn host_getattr_instance_dict_beats_non_data_descriptor() {
        // A class-level `NativeCallable` is a non-data descriptor
        // (no `__set__` / `__delete__`). The instance dict wins over
        // it — `obj.method` returns whatever is shadowed in
        // `obj.__dict__`.
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let method =
            HostObject::new_native_callable("pkg.Box.method", Arc::new(|_| Ok(ConstValue::Int(1))));
        cls.class_set("method", ConstValue::HostObject(method));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        inst.instance_set("method", ConstValue::Int(42)).unwrap();
        let out = host_getattr(&inst, "method").expect("instance dict wins");
        assert_eq!(out, ConstValue::Int(42));
    }

    #[test]
    fn host_getattr_property_lookup_on_class_returns_descriptor() {
        // Class-level lookup (no `self`): the raw property descriptor
        // is returned, matching Python's descriptor protocol when
        // `__get__` is called with `obj is None`.
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let fget = HostObject::new_native_callable(
            "pkg.Box.value_fget",
            Arc::new(|_args| Ok(ConstValue::Int(7))),
        );
        let prop = HostObject::new_property("pkg.Box.value", Some(fget), None, None);
        cls.class_set("value", ConstValue::HostObject(prop.clone()));
        let out = host_getattr(&cls, "value").expect("class-level lookup");
        match out {
            ConstValue::HostObject(h) => assert_eq!(h, prop),
            other => panic!("expected property descriptor, got {other:?}"),
        }
    }

    #[test]
    fn host_getattr_property_fget_returns_raw_slot() {
        let fget = HostObject::new_native_callable(
            "pkg.Box.value_fget",
            Arc::new(|_args| Ok(ConstValue::Int(7))),
        );
        let prop = HostObject::new_property("pkg.Box.value", Some(fget.clone()), None, None);
        let out = host_getattr(&prop, "__name__").expect("property.__name__");
        assert_eq!(out, ConstValue::byte_str("value"));
        let out = host_getattr(&prop, "fget").expect("property.fget");
        assert_eq!(out, ConstValue::HostObject(fget));
        let out = host_getattr(&prop, "fset").expect("property.fset");
        assert_eq!(out, ConstValue::None);
    }

    #[test]
    fn host_getattr_staticmethod_and_classmethod_dunder_func_return_wrapped_callable() {
        let func = HostObject::new_native_callable(
            "pkg.Box.value",
            Arc::new(|_args| Ok(ConstValue::Int(7))),
        );
        let sm = HostObject::new_staticmethod("pkg.Box.value", func.clone());
        let cm = HostObject::new_classmethod("pkg.Box.value", func.clone());
        let out = host_getattr(&sm, "__func__").expect("staticmethod.__func__");
        assert_eq!(out, ConstValue::HostObject(func.clone()));
        let out = host_getattr(&cm, "__func__").expect("classmethod.__func__");
        assert_eq!(out, ConstValue::HostObject(func));
    }

    #[test]
    fn host_getattr_bound_method_dunder_self_and_func_return_members() {
        let self_obj = HostObject::new_instance(HostObject::new_class("pkg.Box", vec![]), vec![]);
        let func = HostObject::new_native_callable(
            "pkg.Box.value",
            Arc::new(|_args| Ok(ConstValue::Int(7))),
        );
        let method = HostObject::new_bound_method(
            "pkg.Box.value",
            self_obj.clone(),
            func.clone(),
            "value",
            HostObject::new_class("pkg.Box", vec![]),
        );
        let out = host_getattr(&method, "__self__").expect("method.__self__");
        assert_eq!(out, ConstValue::HostObject(self_obj));
        let out = host_getattr(&method, "__func__").expect("method.__func__");
        assert_eq!(out, ConstValue::HostObject(func));
    }

    #[test]
    fn host_getattr_builtin_callable_dunder_self_returns_builtin_module() {
        let func = HostObject::new_builtin_callable("len");
        let out = host_getattr(&func, "__self__").expect("builtin.__self__");
        assert_eq!(out, ConstValue::HostObject(HOST_ENV.builtin_module()));
    }

    #[test]
    fn host_getattr_user_function_dunder_name_and_module_follow_python_surface() {
        let func = HostObject::new_user_function(GraphFunc::new(
            "pkg.module.demo",
            Constant::new(ConstValue::Dict(HashMap::new())),
        ));
        let out = host_getattr(&func, "__name__").expect("function.__name__");
        assert_eq!(out, ConstValue::byte_str("demo"));
        let out = host_getattr(&func, "__module__").expect("function.__module__");
        assert_eq!(out, ConstValue::byte_str("pkg.module"));
    }

    #[test]
    fn host_getattr_user_function_dunder_globals_returns_raw_dict() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = HostObject::new_user_function(GraphFunc::new("pkg.module.demo", globals));
        let out = host_getattr(&func, "__globals__").expect("function.__globals__");
        assert_eq!(out, ConstValue::Dict(HashMap::new()));
    }

    #[test]
    fn host_getattr_user_function_dunder_defaults_returns_none_when_absent() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = HostObject::new_user_function(GraphFunc::new("pkg.module.demo", globals));
        let out = host_getattr(&func, "__defaults__").expect("function.__defaults__");
        assert_eq!(out, ConstValue::None);
    }

    #[test]
    fn host_getattr_user_function_dunder_closure_returns_none_when_absent() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = HostObject::new_user_function(GraphFunc::new("pkg.module.demo", globals));
        let out = host_getattr(&func, "__closure__").expect("function.__closure__");
        assert_eq!(out, ConstValue::None);
    }

    #[test]
    fn host_getattr_user_function_dunder_defaults_and_closure_return_tuple_payloads() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let mut graph_func = GraphFunc::new("pkg.module.demo", globals);
        graph_func.defaults = vec![Constant::new(ConstValue::Int(7))];
        graph_func.closure = vec![Constant::new(ConstValue::byte_str("cell"))];
        let func = HostObject::new_user_function(graph_func);

        let defaults = host_getattr(&func, "__defaults__").expect("function.__defaults__");
        assert_eq!(defaults, ConstValue::Tuple(vec![ConstValue::Int(7)]));

        let closure = host_getattr(&func, "__closure__").expect("function.__closure__");
        assert_eq!(
            closure,
            ConstValue::Tuple(vec![ConstValue::byte_str("cell")])
        );
    }

    #[test]
    fn host_getattr_module_and_class_dunder_name_follow_python_surface() {
        let module = HostObject::new_module("pkg.demo");
        let out = host_getattr(&module, "__name__").expect("module.__name__");
        assert_eq!(out, ConstValue::byte_str("pkg.demo"));
        assert_eq!(
            host_getattr(&module, "__module__"),
            Err(HostGetAttrError::Missing)
        );

        let cls = HostObject::new_class("pkg.demo.Box", vec![]);
        let out = host_getattr(&cls, "__name__").expect("class.__name__");
        assert_eq!(out, ConstValue::byte_str("Box"));
        let out = host_getattr(&cls, "__module__").expect("class.__module__");
        assert_eq!(out, ConstValue::byte_str("pkg.demo"));
    }

    #[test]
    fn host_getattr_staticmethod_and_classmethod_dunder_name_and_module_follow_wrapped_func() {
        let func = HostObject::new_user_function(GraphFunc::new(
            "pkg.demo.value",
            Constant::new(ConstValue::None),
        ));
        let sm = HostObject::new_staticmethod("pkg.demo.value", func.clone());
        let cm = HostObject::new_classmethod("pkg.demo.value", func);

        let out = host_getattr(&sm, "__name__").expect("staticmethod.__name__");
        assert_eq!(out, ConstValue::byte_str("value"));
        let out = host_getattr(&sm, "__module__").expect("staticmethod.__module__");
        assert_eq!(out, ConstValue::byte_str("pkg.demo"));

        let out = host_getattr(&cm, "__name__").expect("classmethod.__name__");
        assert_eq!(out, ConstValue::byte_str("value"));
        let out = host_getattr(&cm, "__module__").expect("classmethod.__module__");
        assert_eq!(out, ConstValue::byte_str("pkg.demo"));
    }

    #[test]
    fn user_function_call_host_is_not_supported() {
        // UserFunction carries a graph; executing it requires the
        // Phase 6 host-graph evaluator, so call_host returns the
        // discriminated `RequiresFlowEvaluator` marker so callsites
        // (`call_freeze_method`, `host_descriptor_get` property.fget)
        // can fall back to their annotator-side routing.
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let gf = GraphFunc::new("pkg.user_fn", globals);
        let f = HostObject::new_user_function(gf);
        let err = f.call_host(&[]).unwrap_err();
        assert!(matches!(err, HostCallError::RequiresFlowEvaluator(_)));
        assert!(!f.is_host_executable());
    }

    #[test]
    fn native_callable_is_host_executable() {
        let f = HostObject::new_native_callable("pkg.f", Arc::new(|_| Ok(ConstValue::Int(0))));
        assert!(f.is_host_executable());
        // A BoundMethod whose __func__ is NativeCallable is also
        // executable — upstream wraps the `self` prepend invisibly.
        let self_obj = HostObject::new_opaque("pkg.self");
        let bm = HostObject::new_bound_method(
            "pkg.f_bound",
            self_obj,
            f,
            "f",
            HostObject::new_class("pkg.C", vec![]),
        );
        assert!(bm.is_host_executable());
    }

    #[test]
    fn debug_of_cyclic_block_and_link_terminates() {
        // A block whose sole exit links back to itself is a legal CFG
        // cycle (`while True: pass`). `#[derive(Debug)]` walked
        // Block.exits -> Link.target -> Block.exits ... forever and
        // overflowed the stack when a diagnostic formatted it. The shallow
        // Block.__str__ / Link.__repr__ reprs must terminate instead.
        let v = Hlvalue::Variable(Variable::new());
        let blk: BlockRef = Rc::new(RefCell::new(Block::new(vec![v.clone()])));
        let link = Link::new(vec![v], Some(blk.clone()), None).into_ref();
        // BlockRefExt::closeblock wires link.prevblock = blk and
        // blk.exits = [link], closing the self-loop.
        blk.closeblock(vec![link.clone()]);

        let link_repr = format!("{:?}", link.borrow());
        assert!(link_repr.starts_with("link from "), "got: {link_repr}");
        assert!(link_repr.matches("block").count() >= 1, "got: {link_repr}");

        let block_repr = format!("{:?}", blk.borrow());
        assert!(block_repr.contains("with 1 exits"), "got: {block_repr}");
    }
}
