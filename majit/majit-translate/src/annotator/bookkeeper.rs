//! Bookkeeper — central state carrier for the annotator.
//!
//! RPython upstream: `rpython/annotator/bookkeeper.py` (614 LOC).
//!
//! Structural port of `immutablevalue` / `newlist` / `newdict` and the
//! descriptor / class machinery (`descs`, `classdefs`, `methoddescs`,
//! `emulated_pbc_calls`, `frozenpbc_attr_families`, …). Fields not yet
//! landed are documented at the site that still dispatches through the
//! deferred branch.
//!
//! ## Dependency-blocked paths
//!
//! * `immutablevalue(HostObject)` — the `extregistry.is_registered(x)`
//!   branch (bookkeeper.py) is wired in
//!   [`Self::immutablevalue_hostobject`] via
//!   [`crate::translator::rtyper::extregistry::is_registered`]; per-entry
//!   value-level coverage extends as registrations land.
//! * `BUILTIN_ANALYZERS` registry — ported in [`super::builtin`].
//!   `SomeBuiltin.analyser_name` is seeded from the host qualname;
//!   `SomeValue::call()` for `Builtin(_)` dispatches through
//!   [`super::builtin::call_builtin`]. Only registered builtins route
//!   through `SomeBuiltin`, matching upstream
//!   `bookkeeper.py`.
//! * `classpbc_attr_families` / `all_specializations` — not yet
//!   mirrored on the Rust bookkeeper; reachable only from rtyper-phase
//!   consumers.

use indexmap::{IndexMap, IndexSet};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

use super::argument::{ArgumentsForTranslation, simple_args};
use super::classdesc::{ClassDef, ClassDesc};
use super::description::{
    CallFamily, ClassAttrFamily, ClassDefKey, DescEntry, DescKey, FrozenAttrFamily, FrozenDesc,
    FunctionDesc, MethodDesc,
};
use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, SomeBool, SomeBuiltin, SomeChar, SomeDict, SomeFloat, SomeInteger, SomeList,
    SomePBC, SomeString, SomeTuple, SomeUnicodeCodePoint, SomeUnicodeString, SomeValue,
    s_impossible_value, s_none, union,
};
use super::policy::AnnotatorPolicy;
use crate::flowspace::argument::CallShape;
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::cpython_code_signature;
use crate::flowspace::model::{BlockRef, ConstValue, Constant, GraphKey, GraphRef, HostObject};
use crate::tool::algo::unionfind::UnionFind;

// Exists to localise prepass nondeterminism (gh#1139).
static REFLOW_FROM_ATTR: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn reflow_from_attr_count() -> u64 {
    REFLOW_FROM_ATTR.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static REFLOW_FROM_PBC: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn reflow_from_pbc_count() -> u64 {
    REFLOW_FROM_PBC.load(Ordering::Relaxed)
}

/// RPython `bookkeeper.position_key` (bookkeeper.py) — the tuple
/// identifying "where in the flow graph the annotator is currently
/// reading/writing a value".
///
/// Upstream stores `(FunctionGraph, Block, operation_index)` directly.
/// The Rust port carries the identity-hash values of the first two
/// components so the struct stays:
///   * cheap to clone / hash (no flowspace import cycle),
///   * free of borrow-lifetime issues inside `read_locations:
///     HashSet<PositionKey>`,
///   * still upstream-shaped as a 3-tuple that `ListItem.read_locations
///     |= other.read_locations` can merge without loss.
///
/// Callers obtain the identity hashes via
/// `Rc::as_ptr(&graph) as usize` / `Rc::as_ptr(&block) as usize`. The
/// `graph_ref` / `block_ref` slots carry the actual `Weak` refs when
/// production code constructs the key via [`Self::from_refs`];
/// upstream `reflowfromposition(position_key)` unpacks `graph, block,
/// index = position_key` and needs those references back.
///
/// Test constructors use [`Self::new`] and leave the weak slots
/// dangling — downstream consumers that need the graph/block refs
/// detect this via `Weak::upgrade()` returning `None`.
///
/// Hash / Eq always use the three integer identities, so synthetic
/// test keys and production keys with equal hashes are
/// indistinguishable (mirroring upstream's Python tuple equality via
/// `id()`).
#[derive(Clone, Debug)]
pub struct PositionKey {
    /// Identity hash of the enclosing `FunctionGraph` — upstream
    /// `position_key[0]`.
    pub(crate) graph_id: usize,
    /// Identity hash of the enclosing `Block` — upstream
    /// `position_key[1]`.
    pub(crate) block_id: usize,
    /// Operation index inside the block — upstream `position_key[2]`.
    pub(crate) op_index: usize,
    /// Weak reference to the enclosing `FunctionGraph`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub(crate) graph_ref:
        Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::FunctionGraph>>>,
    /// Weak reference to the enclosing `Block`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub(crate) block_ref: Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::Block>>>,
}

impl PartialEq for PositionKey {
    fn eq(&self, other: &Self) -> bool {
        self.graph_id == other.graph_id
            && self.block_id == other.block_id
            && self.op_index == other.op_index
    }
}

impl Eq for PositionKey {}

impl std::hash::Hash for PositionKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.graph_id.hash(state);
        self.block_id.hash(state);
        self.op_index.hash(state);
    }
}

/// Upstream `analyzer_for(func)` decorator (bookkeeper.py).
pub fn analyzer_for(
    reg: &mut HashMap<String, super::builtin::BuiltinAnalyzer>,
    qualname: &str,
    analyser: super::builtin::BuiltinAnalyzer,
) {
    if reg.insert(qualname.to_string(), analyser).is_some() {
        panic!("bookkeeper.rs: duplicate BUILTIN_ANALYZERS entry for {qualname}");
    }
}

impl PositionKey {
    /// Synthetic constructor — test-only. Fills the identity triple
    /// directly and leaves the Weak refs empty.
    #[cfg(test)]
    pub(crate) fn new(graph_id: usize, block_id: usize, op_index: usize) -> Self {
        PositionKey {
            graph_id,
            block_id,
            op_index,
            graph_ref: None,
            block_ref: None,
        }
    }

    /// Production constructor — derives the identity hashes from
    /// `Rc::as_ptr` (matches upstream's Python tuple identity via
    /// `id()`), and retains `Weak` refs so consumers like
    /// `reflowfromposition` can upgrade back to the live
    /// `FunctionGraph` / `Block`.
    pub(crate) fn from_refs(graph: &GraphRef, block: &BlockRef, op_index: usize) -> Self {
        PositionKey {
            graph_id: Rc::as_ptr(graph) as usize,
            block_id: Rc::as_ptr(block) as usize,
            op_index,
            graph_ref: Some(Rc::downgrade(graph)),
            block_ref: Some(Rc::downgrade(block)),
        }
    }

    /// Upgrade the weak graph reference, if the key carries one and
    /// the target is still alive. Mirrors upstream's `graph, _, _ =
    /// position_key` tuple unpack.
    pub(crate) fn graph(&self) -> Option<GraphRef> {
        self.graph_ref.as_ref().and_then(|w| w.upgrade())
    }

    /// Upgrade the weak block reference. Mirrors upstream's `_, block,
    /// _ = position_key` unpack.
    pub(crate) fn block(&self) -> Option<BlockRef> {
        self.block_ref.as_ref().and_then(|w| w.upgrade())
    }
}

/// RPython `class Bookkeeper` (bookkeeper.py).
pub struct Bookkeeper {
    /// RPython `self.annotator = annotator` (bookkeeper.py). A weak
    /// backlink to the owning `RPythonAnnotator`; stored as
    /// `Weak<RPythonAnnotator>` to break the Rc cycle (annotator owns
    /// the bookkeeper via `Rc`). Upgraded on demand via
    /// [`Self::annotator`].
    ///
    /// Test-only `Bookkeeper::new()` leaves this slot empty (the tests
    /// never call into annotator-dependent code paths).
    pub annotator: RefCell<Weak<crate::annotator::annrpython::RPythonAnnotator>>,
    /// RPython `self.policy = annotator.policy` (bookkeeper.py).
    pub policy: AnnotatorPolicy,
    /// RPython `self.position_key = None` initial (bookkeeper.py:147).
    /// The annotator driver (`RPythonAnnotator.reflow`) writes into
    /// this slot around each reflow block so `read_item` / `agree`
    /// pick it up. Interior mutability because callers hold
    /// `Rc<Bookkeeper>` sharers.
    pub(crate) position_key: RefCell<Option<PositionKey>>,
    /// RPython `self.listdefs = {}` (bookkeeper.py). Keyed by
    /// position — callers hitting the same position twice share the
    /// ListDef so merging re-entries stay identity-equal. The key is
    /// `Option<PositionKey>` because upstream uses `self.position_key`
    /// directly as the dict key (bookkeeper.py:180
    /// `self.listdefs[self.position_key]`); when `position_key` is
    /// `None`, upstream still caches under the `None` key — so we do
    /// the same rather than building a fresh ListDef per call outside
    /// a reflow frame.
    pub(crate) listdefs: RefCell<HashMap<Option<PositionKey>, ListDef>>,
    /// RPython `self.dictdefs = {}` (bookkeeper.py). Same
    /// `Option<PositionKey>` key semantics as `listdefs`.
    pub(crate) dictdefs: RefCell<HashMap<Option<PositionKey>, DictDef>>,
    /// RPython `self.descs = {}` (bookkeeper.py). Maps
    /// `Constant(pyobj)` to a FunctionDesc / ClassDesc / FrozenDesc /
    /// MethodDesc / MethodOfFrozenDesc per bookkeeper.py getdesc. The
    /// Rust port keys directly on [`HostObject`] (which already has
    /// `Arc::ptr_eq` identity) via [`DescEntry`].
    /// `IndexMap`, not `HashMap`: `normalize_class_pbcs` iterates
    /// `descs.values()` to resolve DescKeys back to `ClassDesc`s and folds
    /// them into a `commonbase`, feeding classdef numbering.  The key is a
    /// pointer identity, so only insertion order (IndexMap) is
    /// reproducible — a `HashMap` would number classdefs run-varyingly.
    pub(crate) descs: RefCell<IndexMap<HostObject, DescEntry>>,
    /// RPython `self.classdefs = []` (bookkeeper.py). Populated by
    /// `ClassDesc._init_classdef` (classdesc.py). ClassDef
    /// identity is Rc pointer equality — matches upstream's Python
    /// `cls is other` comparisons.
    pub classdefs: RefCell<Vec<Rc<RefCell<ClassDef>>>>,
    /// RPython `self.methoddescs = {}` (bookkeeper.py). Keyed by
    /// `(funcdesc, originclassdef, selfclassdef, name, flags)` tuple
    /// so repeated `getmethoddesc(...)` calls with the same inputs
    /// share identity, per bookkeeper.py.
    pub(crate) methoddescs: RefCell<HashMap<MethodDescKey, Rc<RefCell<MethodDesc>>>>,
    /// RPython `self.frozenpbc_attr_families = UnionFind(FrozenAttrFamily)`
    /// (bookkeeper.py:63).
    pub(crate) frozenpbc_attr_families: RefCell<UnionFind<DescKey, Rc<RefCell<FrozenAttrFamily>>>>,
    /// RPython `self.classpbc_attr_families = {}` (bookkeeper.py) —
    /// lazy `attrname -> UnionFind(ClassAttrFamily)` map materialised by
    /// `get_classpbc_attr_families(attrname)` (bookkeeper.py).
    /// `IndexMap`: `normalize_class_pbcs` iterates this map to build the
    /// commonbase/access-set numbering; upstream is a Python dict.
    pub(crate) classpbc_attr_families:
        RefCell<IndexMap<String, UnionFind<DescKey, Rc<RefCell<ClassAttrFamily>>>>>,
    /// RPython `self.pbc_maximal_call_families = UnionFind(CallFamily)`
    /// (bookkeeper.py:64).
    pub(crate) pbc_maximal_call_families: RefCell<UnionFind<DescKey, Rc<RefCell<CallFamily>>>>,
    /// RPython `self.emulated_pbc_calls = {}` (bookkeeper.py:66).
    /// `IndexMap`, not `HashMap`: `compute_at_fixpoint` iterates
    /// `.values()` to drive `consider_call_site`, and upstream walks
    /// `emulated_pbc_calls.itervalues()` in dict insertion order.  A
    /// `HashMap` would consider the emulated calls in a run-varying order,
    /// perturbing union-seed order and which callee fails first.
    pub(crate) emulated_pbc_calls: RefCell<IndexMap<EmulatedPbcCallKey, (SomePBC, Vec<SomeValue>)>>,
    /// RPython `bookkeeper._jit_annotation_cache = {}`
    /// (rlib/jit.py:903-914), populated lazily by
    /// `ExtEnterLeaveMarker.compute_result_annotation`.
    ///
    /// Outer key: JitDriver `HostObject` identity — RPython keys on
    /// the driver object directly (`cache = self.bookkeeper.
    /// _jit_annotation_cache[driver]`), so the Rust port keys on the
    /// `HostObject` whose `Hash` / `Eq` already match upstream's
    /// `id(driver)` semantics. Storing the `HostObject` (not just
    /// `usize`) preserves the strong reference upstream holds via the
    /// dict, avoiding identity reuse if the underlying object is
    /// dropped. Inner map: kwarg name → merged annotation across all
    /// `jit_merge_point` / `can_enter_jit` call sites for that
    /// driver. Upstream merges via `annmodel.unionof`; the Rust port
    /// uses [`super::model::union`] with
    /// [`super::model::s_impossible_value`] as the
    /// seed for the first hit.
    pub _jit_annotation_cache: RefCell<HashMap<HostObject, HashMap<String, SomeValue>>>,
    /// RPython `self.immutable_cache = {}` (bookkeeper.py), keyed by
    /// the identity of `Constant(x)` for the List / Dict / OrderedDict /
    /// r_dict branches of `immutablevalue` (bookkeeper.py).
    /// Upstream's cache memoises the returned `SomeList` / `SomeDict` so
    /// downstream `generalize_key` / `generalize` mutations on the
    /// cached `DictDef` / `ListDef` persist across calls.
    ///
    /// `transform_list_contains` (translator/transform.py)
    /// relies on this: it calls `self.annotation(Constant(dict))` after
    /// rewriting the contains lhs, then mutates the returned SomeDict's
    /// dictdef via `generalize_key`. Without the cache the second
    /// `annotation()` call builds a fresh DictDef and the mutation
    /// dissolves.
    pub immutable_cache: RefCell<HashMap<u64, SomeValue>>,
    /// RPython `self.pending_specializations = []` (bookkeeper.py).
    ///
    /// List of callbacks drained by
    /// `AnnotatorPolicy.no_more_blocks_to_annotate` before the final
    /// annotation fixpoint. Each callback (e.g. `MemoTable.finish`) is
    /// fallible; upstream lets the exception propagate, so the drain
    /// surfaces the first `Err` instead of swallowing or panicking on it.
    pub pending_specializations: RefCell<Vec<Box<dyn Fn() -> Result<(), AnnotatorError>>>>,
    /// RPython `self.all_specializations = {}` (bookkeeper.py).
    ///
    /// One entry per memo-specialised function, keyed by the funcdesc's
    /// pointer identity (upstream keys by the `funcdesc` object itself,
    /// specialize.py). Each value is the
    /// `UnionFind(compute_one_result)` upstream uses (specialize.py) —
    /// the families of argument tuples that can be called together, merged
    /// on `union`, with each result living in its
    /// [`super::specialize::MemoTable`]'s `table` — paired with a per-
    /// family host-call error latch (see [`super::specialize::MemoFamily`]).
    /// Consumed by [`super::specialize::memo`].
    pub all_specializations: RefCell<HashMap<usize, super::specialize::MemoFamily>>,
    /// RPython `hasattr(self, 'position_key')` (bookkeeper.py:99).
    ///
    /// Upstream distinguishes "no position entered" (attribute absent)
    /// from "entered with position_key = None" (attribute present but
    /// None). Rust carries [`Self::position_key`] as `Option<_>` in both
    /// cases, so this flag tracks the enter/leave invariant explicitly.
    pub position_entered: std::cell::Cell<bool>,
    /// RPython `self.needs_generic_instantiate = {}` (bookkeeper.py).
    ///
    /// Populated by `merge_classpbc_getattr_into_classdef`
    /// (normalizecalls.py) for every class-PBC call family that
    /// spans more than one class — `ClassesPBCRepr.call()` needs the
    /// generic instantiator. Drained by
    /// `create_instantiate_functions` (normalizecalls.py).
    ///
    /// Upstream is a `{}` dict keyed by `ClassDef` identity with every
    /// value set to `True` — i.e. used as a set with upstream's
    /// insertion-order iteration. Pyre mirrors the dict shape via a
    /// `BTreeMap<ClassDefKey, Rc<RefCell<ClassDef>>>`: keys are the
    /// `ClassDefKey` pointer-identity wrapper (matching upstream's
    /// `id(classdef)` hash) and the value retains the `Rc` needed by
    /// consumers. Iteration order is deterministic (sorted by the
    /// pointer-derived key) which differs from upstream's insertion
    /// order, but the consumer
    /// ([`crate::translator::rtyper::normalizecalls::create_instantiate_functions`])
    /// processes each classdef independently so iteration order is not
    /// observable.
    pub(crate) needs_generic_instantiate:
        RefCell<std::collections::BTreeMap<ClassDefKey, Rc<RefCell<ClassDef>>>>,
    /// TODO: no upstream equivalent.  Pyre-only struct-field
    /// metadata snapshot (`struct_name -> [(field_name, type_string)]`)
    /// used by [`Self::getuniqueclassdef_for_struct_root`] to project the
    /// registered `ClassDef.attrs`.  `None` for unit-test fixtures that
    /// build the bookkeeper directly, in which case
    /// `getuniqueclassdef_for_struct_root` leaves `attrs` empty.
    pub struct_fields: RefCell<Option<Rc<crate::front::StructFieldRegistry>>>,
    /// TODO: no upstream equivalent (RPython has no Rust enums).  Enum
    /// type-root name (dual-keyed: qualified path and bare leaf) →
    /// `discriminant value → variant name` table.  Consulted by
    /// [`Self::enum_variant_narrowing_knowntypedata`] to build the
    /// discriminant→variant narrowing `knowntypedata` the `__discriminant`
    /// getattr attaches.  `None` for unit-test fixtures that build the
    /// bookkeeper directly.
    pub enum_variant_by_discriminant: RefCell<Option<Rc<HashMap<String, HashMap<i64, String>>>>>,
    /// TODO: no upstream equivalent.  Interning table mapping a struct
    /// type-root name to its canonical host class `HostObject`.  Because
    /// `HostObject` equality is `Arc` pointer identity (`impl PartialEq for
    /// HostObject` in `flowspace/model.rs`),
    /// two `HostObject::new_class(name, ...)` calls produce distinct
    /// descs/classdefs; interning resolves a type-root string to one
    /// class OBJECT exactly once so subsequent `getuniqueclassdef`
    /// lookups key the same `ClassDef` by identity — the pyre analog of
    /// `annotationoftype(t) -> getuniqueclassdef(t)` (signature.py)
    /// where `t` is the already-resolved class object.  Used by
    /// [`Self::getuniqueclassdef_for_struct_root`].
    pub struct_root_classes: RefCell<HashMap<String, HostObject>>,
    /// Class key (`canonical_struct_name`) of the interpreter's error
    /// carrier object, the `Deref::Target` of the `E` of `Result<T, E>`.
    /// The object is what the program raises, so it is minted as an
    /// `Exception` subclass — `class OperationError(Exception)`
    /// (`pypy/interpreter/error.py`).
    exception_carrier: RefCell<Option<String>>,
    /// Canonical key of the transparent handle (`carrier_path`) when it
    /// differs from [`Self::exception_carrier`]. The handle intern's as
    /// a subclass of the object class so each keeps its own layout.
    exception_carrier_handle: RefCell<Option<String>>,
    /// TODO: no upstream equivalent.  Qualified trait path → owner
    /// root of its only concrete impl in the analyzed LLBC world
    /// (computed in `lib.rs` from `concrete_trait_methods`; multi-impl
    /// traits are absent).  The key is the trait's full `name_path()`
    /// so leaf-name collisions between distinct traits cannot pool
    /// impl owners.  `derive_subject_inputcells` consults this when a
    /// `Ref` parameter's `class_root` is a bound-trait path (generic
    /// receiver) — the unique impl's struct root then seeds the
    /// receiver's `ClassDef` through
    /// [`Self::getuniqueclassdef_for_struct_root`].
    /// RPython's annotator never needs this: it sees the concrete
    /// receiver class at every call site (`classdesc.py lookup`).
    pub trait_unique_impls: RefCell<HashMap<String, String>>,
    /// Shared `CallControl` layout table. The raw-pointer seed builds an
    /// `lltype.Struct` from these records and caches it on `StructLayout.ll_struct`.
    /// `None` for unit-test fixtures that never register a layout.
    pub struct_layouts: RefCell<Option<crate::codewriter::call::StructLayoutTable>>,
    /// Trait qualified-path (`name_path()`) → base `HostObject` for a
    /// receiver-driven method-dispatch family registered through
    /// [`Self::register_trait_family`] (receiver-dispatch configuration).
    /// `derive_subject_inputcells` seeds a `dyn Trait` receiver whose
    /// `class_root` is a key here with the base `ClassDef`, so
    /// `getattr(receiver, method)` resolves the impl-subclass
    /// `MethodDesc` family (attrfamily merge) rather than blocking on
    /// the classdef-less shell.  Empty unless a consumer opts a trait
    /// in; pyre production registers none.
    pub trait_family_bases: RefCell<HashMap<String, HostObject>>,
    /// TODO: no upstream equivalent.  Struct names first interned by
    /// [`Self::project_struct_field_type`]'s bare-name arm whose
    /// registry rows have not been projected yet — drained at the end
    /// of [`Self::getuniqueclassdef_for_struct_root`] so a class
    /// never exposes its untyped FORCE shells to subject flow (a
    /// later projection would be a non-monotonic attr flip).
    pub(crate) pending_struct_row_projection: RefCell<Vec<String>>,
    /// Canonical struct names whose rows [`Self::project_struct_rows`]
    /// already projected (dedups the pending drain and the bare-name
    /// arm's first-sight detection).
    pub(crate) projected_struct_rows: RefCell<std::collections::HashSet<String>>,
}

impl std::fmt::Debug for Bookkeeper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bookkeeper")
            .field("position_key", &self.position_key.borrow())
            .field("listdefs_len", &self.listdefs.borrow().len())
            .field("dictdefs_len", &self.dictdefs.borrow().len())
            .field("descs_len", &self.descs.borrow().len())
            .field("classdefs_len", &self.classdefs.borrow().len())
            .field("methoddescs_len", &self.methoddescs.borrow().len())
            .finish()
    }
}

/// Key for the `Bookkeeper.methoddescs` cache. Upstream uses a tuple
/// `(funcdesc, originclassdef, selfclassdef, name, tuple(flags.items()))`
/// — Python hashes on object identity for `funcdesc` / `classdef`, and
/// on value for the rest. The Rust port mirrors that by keying on the
/// pointer identity of the two descriptor Rcs plus the stringified
/// name + flags.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MethodDescKey {
    /// RPython `funcdesc` — pointer identity via [`DescKey::from_rc`].
    pub(crate) funcdesc_id: DescKey,
    /// RPython `originclassdef` — `ClassDefKey` already carries the
    /// pointer identity.
    pub(crate) originclassdef: ClassDefKey,
    /// RPython `selfclassdef` — `None` for unbound methods.
    pub(crate) selfclassdef: Option<ClassDefKey>,
    /// RPython `name`.
    pub(crate) name: String,
    /// RPython `tuple(flags.items())` — flattened sort-stable flag
    /// entries.
    pub(crate) flags: Vec<(String, bool)>,
}

/// Hashable identity for `Bookkeeper.emulated_pbc_calls`.
///
/// Upstream accepts any hashable Python object. The currently-ported
/// callers need a stable position key plus the r_dict eq/hash
/// pseudo-call identities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum EmulatedPbcCallKey {
    Position(PositionKey),
    Graph(GraphKey),
    ClassDef(ClassDefKey),
    RDictCall {
        item_id: usize,
        role: &'static str,
    },
    Text(String),
    /// RPython `('sandboxing', s_func.const)` tuple (policy.py). The
    /// inner identity is the `HostObject` pointer for the external
    /// callable being sandboxed.
    Sandboxing {
        callable_id: usize,
    },
    /// RPython `annlowlevel.py` key `(llhelper, desc.pyobj)`.
    LLHelper {
        callable_id: usize,
    },
}

/// RPython `Bookkeeper.pbc_call`'s `emulated` parameter
/// (bookkeeper.py). Encodes the Python-side three-state
/// polymorphism (`None` / `True` / `<position_key>`) as a Rust enum:
///
/// * `None` — real call: use `self.position_key`; pull
///   `s_previous_result` from the current op's annotation.
/// * `True` — fully emulated: `whence=None`, `op=None`,
///   `s_previous_result=s_ImpossibleValue`.
/// * `Callback(position)` — callback-style: `whence=position`, same
///   `op=None` / `s_previous_result=s_ImpossibleValue` as `True`.
#[derive(Clone, Debug)]
pub(crate) enum PbcCallEmulated {
    None,
    True,
    Callback(PositionKey),
}

impl EmulatedPbcCallKey {
    /// Build the sandbox-trampoline key used by
    /// `AnnotatorPolicy::no_more_blocks_to_annotate` (policy.py).
    pub(crate) fn sandboxing(func_const: &crate::flowspace::model::ConstValue) -> Self {
        use crate::flowspace::model::ConstValue;
        let callable_id = match func_const {
            ConstValue::HostObject(obj) => obj.identity_id(),
            // Non-HostObject consts reach here only if the policy walker
            // stumbles over a non-sandboxable callable; fall back to the
            // pointer of the enum discriminant so two such entries stay
            // distinct yet the key remains stable for that
            // call site.
            _ => std::ptr::addr_of!(*func_const) as usize,
        };
        EmulatedPbcCallKey::Sandboxing { callable_id }
    }
}

/// Variant wrapper over the two concrete attribute-family types that
/// `pbc_getattr` (bookkeeper.py) consumes polymorphically —
/// `ClassAttrFamily` for ClassDesc PBCs, `FrozenAttrFamily` for
/// FrozenDesc PBCs. Exposes the exact trio upstream touches:
/// `read_locations`, `get_s_value(attr)`, `set_s_value(attr, ...)`.
enum PbcAttrFamily {
    Class(Rc<RefCell<ClassAttrFamily>>),
    Frozen(Rc<RefCell<FrozenAttrFamily>>),
}

impl PbcAttrFamily {
    fn add_read_location(&self, pos: PositionKey) {
        match self {
            PbcAttrFamily::Class(rc) => {
                rc.borrow_mut().read_locations.insert(pos, ());
            }
            PbcAttrFamily::Frozen(rc) => {
                rc.borrow_mut().read_locations.insert(pos, ());
            }
        }
    }

    fn read_locations(&self) -> Vec<PositionKey> {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow().read_locations.keys().cloned().collect(),
            PbcAttrFamily::Frozen(rc) => rc.borrow().read_locations.keys().cloned().collect(),
        }
    }

    fn get_s_value(&self, attr: &str) -> SomeValue {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow().get_s_value(attr),
            PbcAttrFamily::Frozen(rc) => rc.borrow().get_s_value(attr),
        }
    }

    fn set_s_value(&self, attr: &str, v: SomeValue) {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow_mut().set_s_value(attr, v),
            PbcAttrFamily::Frozen(rc) => rc.borrow_mut().set_s_value(attr, v),
        }
    }
}

/// `SomePtr(Ptr(Struct raw))` for `class_root`, or `None` when that owner
/// is not a Raw struct with registry rows.
fn raw_struct_ptr_from_layout(
    registry: &crate::front::StructFieldRegistry,
    layouts: &crate::codewriter::call::StructLayoutTable,
    class_root: &str,
) -> Option<super::model::SomeValue> {
    use crate::translator::rtyper::lltypesystem::lltype::{
        ForwardReference, GcKind, LowLevelType, Ptr, PtrTarget,
    };

    // A fieldless enum value is the integer tag. A payload enum's address
    // is `Ptr` of the raw struct; the base still carries only
    // `__discriminant` (`rclass.py` `getinstancerepr`).
    let lookup = generic_owner_lookup(class_root);
    if registry.is_enum_base(lookup) && !registry.enum_base_has_payload(lookup) {
        return None;
    }
    // A closure env is not a C raw struct. `tyref_input_class_root` keeps
    // the full `name_path` so capture fields project through SomeInstance
    // getattr (`project_struct_field_type`). Seeding SomePtr made
    // getattr(`__pos_0`) use the lltype row, which spells a captured
    // `&usize` as Address (`pointer_field_lltype` of a scalar pointee).
    // `base + 1` then became Address+Int, and `shadow_stack_get` merged
    // r_uint ∪ Address. RPython forbids closures; the env stays the
    // instance whose capture rows `project_struct_field_type` already
    // peels to the integer word.
    let leaf = class_root.rsplit("::").next().unwrap_or(class_root);
    if majit_charon_reader::ullbc::is_closure_leaf(leaf) {
        return None;
    }
    // `GcKind::Raw` is every struct that is not on the GC field-0 chain.
    // A dict strategy or a type object is still an instance; only a
    // word-storage ADT is `SomePtr`. An empty set is a fixture that never
    // harvested owners, so the gckind test below stays in force.
    if !registry.raw_word_owners.is_empty()
        && !registry.raw_word_owners.contains(lookup)
        && !registry.raw_word_owners.contains(class_root)
    {
        return None;
    }
    let mut building: HashMap<String, ForwardReference> = HashMap::new();
    let container = owner_ll_container(registry, layouts, class_root, &mut building)?;
    let LowLevelType::Struct(st) = container else {
        return None;
    };
    if st._gckind != GcKind::Raw {
        return None;
    }
    let ptr = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*st),
    }));
    Some(crate::translator::rtyper::llannotation::lltype_to_annotation(ptr))
}

fn layout_for_owner(
    layouts: &crate::codewriter::call::StructLayoutTable,
    name: &str,
) -> Option<(
    majit_ir::descr::StructId,
    std::rc::Rc<crate::codewriter::call::StructLayout>,
)> {
    let sid = majit_ir::descr::struct_id_for_name(name)?;
    let layout = layouts.borrow().get(&sid)?.clone();
    Some((sid, layout))
}

/// Container lltype of `name`: a finished `Struct`, or the `ForwardReference`
/// already being built for a pointer back to it.
fn owner_ll_container(
    registry: &crate::front::StructFieldRegistry,
    layouts: &crate::codewriter::call::StructLayoutTable,
    name: &str,
    building: &mut HashMap<
        String,
        crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    >,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::{ForwardReference, GcKind, LowLevelType};

    // Field rows and layouts are registered under the bare owner. The
    // `<...>` suffix is only the concrete arguments for field spellings.
    // A generic instantiation must not reuse the bare owner's one
    // `ll_struct`: `Raw<f64>` and `Raw<usize>` share that slot.
    let lookup = generic_owner_lookup(name);
    let generic = name.contains('<');
    let (_sid, layout) = layout_for_owner(layouts, lookup)?;
    if generic {
        if let Some(st) = layout.ll_struct_by_args.borrow().get(name).cloned() {
            return Some(LowLevelType::Struct(Box::new(st)));
        }
    } else if let Some(st) = layout.ll_struct.borrow().clone() {
        return Some(LowLevelType::Struct(Box::new(st)));
    }
    if let Some(fwd) = building.get(name) {
        return Some(LowLevelType::ForwardReference(Box::new(fwd.clone())));
    }
    let rows = registry.field_rows(lookup)?;
    if rows.is_empty() {
        return None;
    }
    let gckind = layout.gckind;
    let fwd = if gckind == GcKind::Gc {
        ForwardReference::gc()
    } else {
        ForwardReference::new()
    };
    building.insert(name.to_string(), fwd.clone());
    let mut fields = Vec::with_capacity(rows.len());
    for (index, (field_name, spelling)) in rows.iter().enumerate() {
        if field_name.is_empty() {
            continue;
        }
        let concrete = instantiate_field_spelling(registry, name, spelling);
        fields.push((
            field_name.clone(),
            field_spelling_lltype(registry, layouts, &concrete, gckind, index, building),
        ));
    }
    if fields.is_empty() {
        building.remove(name);
        return None;
    }
    let st = crate::translator::rtyper::lltypesystem::lltype::Struct::registry_fields(
        name, fields, gckind,
    );
    fwd.r#become(LowLevelType::Struct(Box::new(st.clone())))
        .expect("registry struct gckind matches its ForwardReference");
    if generic {
        layout
            .ll_struct_by_args
            .borrow_mut()
            .insert(name.to_string(), st.clone());
    } else {
        *layout.ll_struct.borrow_mut() = Some(st.clone());
    }
    building.remove(name);
    Some(LowLevelType::Struct(Box::new(st)))
}

/// Registry and layout keys drop the argument list (`Raw<f64>` → `Raw`).
fn generic_owner_lookup(name: &str) -> &str {
    name.split_once('<').map(|(bare, _)| bare).unwrap_or(name)
}

/// A template row spells a parameter as `??TypeVar#N` (declaration index)
/// or, for one argument, as a bare name such as `T`.
fn instantiate_field_spelling(
    registry: &crate::front::StructFieldRegistry,
    owner: &str,
    spelling: &str,
) -> String {
    let args = owner_type_args(owner);
    if spelling.contains("??TypeVar") {
        return substitute_typevar_markers(spelling, &args);
    }
    if spelling.contains(['<', ':', '*', '[', '&'])
        || scalar_lltype(spelling).is_some()
        || registry.field_rows(spelling).is_some()
    {
        return spelling.to_string();
    }
    if args.len() == 1 {
        return args[0].to_string();
    }
    spelling.to_string()
}

fn owner_type_args(owner: &str) -> Vec<&str> {
    let Some((_, args)) = owner.split_once('<') else {
        return Vec::new();
    };
    // One delimiter. `Raw<Option<f64>>` must keep the inner `>`.
    split_generic_args(args.strip_suffix('>').unwrap_or(args))
}

/// `??TypeVar#N` is declaration parameter `N`. A bare `??TypeVar` is
/// parameter 0 when the owner has one argument.
fn substitute_typevar_markers(spelling: &str, args: &[&str]) -> String {
    let mut out = String::new();
    let mut rest = spelling;
    while let Some(idx) = rest.find("??TypeVar") {
        out.push_str(&rest[..idx]);
        rest = &rest[idx + "??TypeVar".len()..];
        if let Some(digits) = rest.strip_prefix('#') {
            let nlen = digits.chars().take_while(|c| c.is_ascii_digit()).count();
            if nlen > 0 {
                let n: usize = digits[..nlen].parse().unwrap_or(usize::MAX);
                match args.get(n) {
                    Some(arg) => out.push_str(arg),
                    None => {
                        out.push_str("??TypeVar#");
                        out.push_str(&digits[..nlen]);
                    }
                }
                rest = &digits[nlen..];
                continue;
            }
        }
        if args.len() == 1 {
            out.push_str(args[0]);
        } else {
            out.push_str("??TypeVar");
        }
    }
    out.push_str(rest);
    out
}

fn field_spelling_lltype(
    registry: &crate::front::StructFieldRegistry,
    layouts: &crate::codewriter::call::StructLayoutTable,
    spelling: &str,
    parent_gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind,
    field_index: usize,
    building: &mut HashMap<
        String,
        crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    >,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};

    let spelling = spelling.trim();
    // `Box<[T]>` / `Box<str>` / `Box<dyn Trait>` is two words. Peeling it
    // like a thin `Box<T>` leaves `pointer_field_lltype` with an unsized
    // pointee, which collapses to one `Address` and drops `.len()`.
    if let Some(fat) = fat_box_field_lltype(spelling) {
        return fat;
    }
    // `*mut dyn Trait` / `*const [T]` / `&dyn Trait` / `Option<*mut dyn ..>`
    // is two words (data + metadata). A one-word Address places later
    // fields one word early (`lltype` models the fat pointer as a struct).
    if let Some(fat) = fat_dyn_pointer_field_lltype(spelling) {
        return fat;
    }
    if let Some(pointee) = peel_one_pointer(spelling) {
        // `get_type_flag("*const u8")` is an unsigned int word. A `*mut u8`
        // or `*mut i8` inside a Raw owner is that same address
        // (`addrinfo.ai_canonname`). A mutable byte pointer in a managed
        // owner stays the Ref erasure below.
        if matches!(pointee.as_str(), "u8" | "i8")
            && (spelling.starts_with("*const ") || parent_gckind == GcKind::Raw)
        {
            return LowLevelType::Address;
        }
        return pointer_field_lltype(registry, layouts, &pointee, building);
    }
    if let Some(scalar) = scalar_lltype(spelling) {
        return scalar;
    }
    // `Option<NonZeroUsize>` is the niche word, not an Option container.
    if let Some(niche) = niche_word_lltype(spelling) {
        return niche;
    }
    if is_string_spelling(spelling) {
        return crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
    }
    let Some(container) = owner_ll_container(registry, layouts, spelling, building) else {
        return LowLevelType::Address;
    };
    // A by-value field is the container itself. An unresolved forward
    // reference cannot be inlined (`_note_inlined_into`), and a Gc container
    // can only be inlined as field 0 of a Gc struct.
    let child_gc = container._gckind();
    let unresolved =
        matches!(container, LowLevelType::ForwardReference(ref fwd) if fwd.resolved().is_none());
    let can_inline = !unresolved
        && (child_gc == GcKind::Raw
            || (child_gc == GcKind::Gc && parent_gckind == GcKind::Gc && field_index == 0));
    if can_inline && container.is_container_type() {
        container
    } else {
        LowLevelType::Address
    }
}

fn pointer_field_lltype(
    registry: &crate::front::StructFieldRegistry,
    layouts: &crate::codewriter::call::StructLayoutTable,
    pointee: &str,
    building: &mut HashMap<
        String,
        crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    >,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget};

    let pointee = pointee.trim();
    if is_string_spelling(pointee) {
        return crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
    }
    // `*mut u8` / `*mut i8` in a managed owner stay the Ref erasure.
    // Raw owners and `*const` byte pointers return Address above.
    // A wider scalar pointer is the raw address word.
    if matches!(pointee, "u8" | "i8") {
        return crate::translator::rtyper::lltypesystem::lltype::GCREF.clone();
    }
    if let Some(scalar) = scalar_lltype(pointee) {
        let _ = scalar;
        return LowLevelType::Address;
    }
    let Some(container) = owner_ll_container(registry, layouts, pointee, building) else {
        return LowLevelType::Address;
    };
    let to = match container {
        LowLevelType::Struct(st) => PtrTarget::Struct(*st),
        LowLevelType::ForwardReference(fwd) => PtrTarget::ForwardReference(*fwd),
        _ => return LowLevelType::Address,
    };
    LowLevelType::Ptr(Box::new(Ptr { TO: to }))
}

/// `Option<NonZeroUsize>` / `Option<NonZero<usize>>` is one integer word.
fn niche_word_lltype(
    spelling: &str,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    let spelling = spelling.trim();
    let start = spelling.rfind("Option<")?;
    let prefix = &spelling[..start];
    if prefix.contains(['<', '>']) || !(prefix.is_empty() || prefix.ends_with("::")) {
        return None;
    }
    let inner = spelling[start + "Option<".len()..]
        .strip_suffix('>')?
        .trim();
    let inner_leaf = inner.rsplit("::").next().unwrap_or(inner).trim();
    let word = if let Some(arg) = inner_leaf
        .strip_prefix("NonZero<")
        .and_then(|rest| rest.strip_suffix('>'))
    {
        arg.trim()
    } else {
        let rest = inner_leaf.strip_prefix("NonZero")?;
        match rest {
            "Usize" => "usize",
            "Isize" => "isize",
            "U8" => "u8",
            "U16" => "u16",
            "U32" => "u32",
            "U64" => "u64",
            "U128" => "u128",
            "I8" => "i8",
            "I16" => "i16",
            "I32" => "i32",
            "I64" => "i64",
            "I128" => "i128",
            _ => return None,
        }
    };
    scalar_lltype(word)
}

fn scalar_lltype(
    spelling: &str,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    Some(match spelling {
        "i8" | "i16" | "i32" | "isize" => LowLevelType::Signed,
        "u8" | "u16" | "u32" | "usize" => LowLevelType::Unsigned,
        // A 32-bit target's machine word is not `i64`. RPython uses
        // `SignedLongLong` / `UnsignedLongLong` for those fields.
        "i64" if crate::layout::target_word_size() == 4 => LowLevelType::SignedLongLong,
        "u64" if crate::layout::target_word_size() == 4 => LowLevelType::UnsignedLongLong,
        "i64" => LowLevelType::Signed,
        "u64" => LowLevelType::Unsigned,
        "i128" => LowLevelType::SignedLongLongLong,
        "u128" => LowLevelType::UnsignedLongLongLong,
        "f64" => LowLevelType::Float,
        "f32" => LowLevelType::SingleFloat,
        "bool" => LowLevelType::Bool,
        "char" => LowLevelType::Char,
        "()" => LowLevelType::Void,
        _ => return None,
    })
}

fn is_string_spelling(spelling: &str) -> bool {
    matches!(
        spelling,
        "str" | "String" | "std::string::String" | "alloc::string::String" | "Utf8Str"
    ) || spelling.ends_with("::Utf8Str")
}

/// One pointer word: `&T` / `&mut T` / `*const T` / `*mut T` / `Box<T>` /
/// `NonNull<T>`, plus `Option` of one of those (the null niche).
fn peel_one_pointer(spelling: &str) -> Option<String> {
    let t = spelling.trim();
    if crate::fat_ptr_layout::spelling_is_fat_ptr(t) {
        return None;
    }
    if let Some(rest) = t
        .strip_prefix("*const ")
        .or_else(|| t.strip_prefix("*mut "))
    {
        return Some(rest.trim().to_string());
    }
    if let Some(rest) = t.strip_prefix("&mut ") {
        return Some(rest.trim().to_string());
    }
    if let Some(rest) = t.strip_prefix('&') {
        let rest = rest.trim();
        if let Some(after_life) = rest.strip_prefix('\'') {
            let rest = after_life
                .split_once(char::is_whitespace)
                .map(|(_, rhs)| rhs.trim())
                .unwrap_or(after_life);
            return Some(rest.to_string());
        }
        return Some(rest.to_string());
    }
    if let Some(inner) = angle_wrapper_inner(t, &["Box", "NonNull"]) {
        if angle_wrapper_inner(t, &["Box"]).is_some() && is_fat_pointee(&inner) {
            return None;
        }
        return Some(inner);
    }
    if let Some(inner) = angle_wrapper_inner(t, &["Option"])
        && peel_one_pointer(&inner).is_some()
    {
        return peel_one_pointer(&inner);
    }
    None
}

/// `Box<[T]>`, `Box<str>`, and `Box<dyn Trait>`: data word then length word
/// (`fat_ptr_layout`). A thin `Box<T>` stays on `peel_one_pointer`.
fn fat_box_field_lltype(
    spelling: &str,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    let inner = angle_wrapper_inner(spelling, &["Box"])?;
    if !is_fat_pointee(&inner) {
        return None;
    }
    two_word_fat_struct(spelling)
}

/// `*mut dyn Trait` / `*const [T]` / `&dyn Trait` / `Option<*mut dyn ..>`:
/// data + metadata. `Box<[T]>` stays on `fat_box_field_lltype`.
fn fat_dyn_pointer_field_lltype(
    spelling: &str,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    if !crate::fat_ptr_layout::spelling_is_fat_ptr(spelling) {
        return None;
    }
    if angle_wrapper_inner(spelling, &["Box"]).is_some() {
        return None;
    }
    two_word_fat_struct(spelling)
}

fn two_word_fat_struct(
    spelling: &str,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};

    let data = LowLevelType::Address;
    let len = LowLevelType::Signed;
    let layout = crate::fat_ptr_layout::probe();
    let fields = if layout.len_offset < layout.data_offset {
        vec![("len".to_string(), len), ("data".to_string(), data)]
    } else {
        vec![("data".to_string(), data), ("len".to_string(), len)]
    };
    Some(LowLevelType::Struct(Box::new(
        crate::translator::rtyper::lltypesystem::lltype::Struct::registry_fields(
            spelling,
            fields,
            GcKind::Raw,
        ),
    )))
}

fn is_fat_pointee(inner: &str) -> bool {
    let inner = inner.trim();
    if inner == "str" || inner.ends_with("::str") || inner.starts_with("dyn ") {
        return true;
    }
    inner
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(|body| !body.contains(';'))
}

fn angle_wrapper_inner(spelling: &str, leaves: &[&str]) -> Option<String> {
    let t = spelling.trim();
    let (head, rest) = t.split_once('<')?;
    if !rest.ends_with('>') {
        return None;
    }
    let leaf = head.trim().rsplit("::").next().unwrap_or(head.trim());
    if !leaves.iter().any(|name| *name == leaf) {
        return None;
    }
    let inner = &rest[..rest.len() - 1];
    if top_level_comma(inner) {
        return None;
    }
    Some(inner.trim().to_string())
}

fn top_level_comma(spelling: &str) -> bool {
    let mut depth = 0i32;
    for ch in spelling.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

impl Bookkeeper {
    /// RPython `Bookkeeper.__init__(self, annotator)` (bookkeeper.py).
    /// Once the annotator driver lands, this constructor takes an
    /// `annotator` backlink; for now it just initialises the bare
    /// storage slots.
    pub fn new() -> Self {
        Self::new_with_policy(AnnotatorPolicy::new())
    }

    pub fn new_with_policy(policy: AnnotatorPolicy) -> Self {
        Bookkeeper {
            annotator: RefCell::new(Weak::new()),
            policy,
            position_key: RefCell::new(None),
            listdefs: RefCell::new(HashMap::new()),
            dictdefs: RefCell::new(HashMap::new()),
            descs: RefCell::new(IndexMap::new()),
            classdefs: RefCell::new(Vec::new()),
            methoddescs: RefCell::new(HashMap::new()),
            frozenpbc_attr_families: RefCell::new(UnionFind::new(|desc: &DescKey| {
                Rc::new(RefCell::new(FrozenAttrFamily::new(*desc)))
            })),
            classpbc_attr_families: RefCell::new(IndexMap::new()),
            pbc_maximal_call_families: RefCell::new(UnionFind::new(|desc: &DescKey| {
                Rc::new(RefCell::new(CallFamily::new(*desc)))
            })),
            emulated_pbc_calls: RefCell::new(IndexMap::new()),
            _jit_annotation_cache: RefCell::new(HashMap::new()),
            immutable_cache: RefCell::new(HashMap::new()),
            pending_specializations: RefCell::new(Vec::new()),
            all_specializations: RefCell::new(HashMap::new()),
            position_entered: std::cell::Cell::new(false),
            needs_generic_instantiate: RefCell::new(std::collections::BTreeMap::new()),
            struct_fields: RefCell::new(None),
            enum_variant_by_discriminant: RefCell::new(None),
            struct_root_classes: RefCell::new(HashMap::new()),
            exception_carrier: RefCell::new(None),
            exception_carrier_handle: RefCell::new(None),
            trait_unique_impls: RefCell::new(HashMap::new()),
            struct_layouts: RefCell::new(None),
            trait_family_bases: RefCell::new(HashMap::new()),
            pending_struct_row_projection: RefCell::new(Vec::new()),
            projected_struct_rows: RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// TODO: no upstream equivalent.  Wire the pyre-only
    /// `StructFieldRegistry` so [`Self::getuniqueclassdef_for_struct_root`]
    /// can project the registered `ClassDef.attrs` from struct-field
    /// metadata.
    ///
    /// May be called before or after roots are registered:
    /// [`Self::getuniqueclassdef_for_struct_root`] re-traverses and
    /// re-projects on every call, so a root cached before the registry
    /// arrived has its attrs back-filled on the next lookup (RPython grows
    /// class attrs as annotation proceeds).  Idempotent: a second call
    /// overwrites the previous registry.
    pub fn set_struct_fields(&self, registry: Rc<crate::front::StructFieldRegistry>) {
        *self.struct_fields.borrow_mut() = Some(registry);
    }

    /// Name the interpreter's error carrier object (see the
    /// `exception_carrier` field doc).  An empty path names none.
    /// First mint of this key receives base `Exception` via
    /// [`Self::default_class_bases`]. Call this before the struct-root
    /// prologue so a later `intern_class_by_qualname` header walk cannot
    /// freeze a different base.
    pub fn set_exception_carrier(&self, carrier_path: &str) {
        *self.exception_carrier.borrow_mut() = (!carrier_path.is_empty()).then(|| {
            majit_ir::descr::canonical_struct_name(&normalize_class_qualname(carrier_path))
        });
    }

    /// Intern the transparent handle as a subclass of the object class.
    /// Identity-aliasing the two spellings onto one `ClassDef` made
    /// getattr of the handle's `.0` / a `MutRef<Handle>.value` resolve
    /// on the payload, which has no such field. The handle keeps its
    /// own layout (`ClassDesc._init_classdef`); `issubclass` still
    /// reaches `Exception` through the object. No-op when no object
    /// class is named or the handle is the same key.
    pub fn alias_exception_carrier_handle(self: &Rc<Self>, handle_path: &str) {
        if handle_path.is_empty() {
            return;
        }
        let handle_key =
            majit_ir::descr::canonical_struct_name(&normalize_class_qualname(handle_path));
        let Some(class_host) = self.exception_carrier_class() else {
            return;
        };
        let class_key = self.exception_carrier.borrow().clone().unwrap_or_default();
        if handle_key == class_key {
            return;
        }
        *self.exception_carrier_handle.borrow_mut() = Some(handle_key.clone());
        let _ = self.intern_class_by_qualname_with_bases(&handle_key, vec![class_host]);
    }

    /// The error carrier's class object, minted on first request.  `None`
    /// when no carrier is named. Empty `__bases__` so
    /// [`Self::default_class_bases`] attaches `Exception` on first mint.
    pub fn exception_carrier_class(self: &Rc<Self>) -> Option<HostObject> {
        let key = self.exception_carrier.borrow().clone()?;
        Some(self.intern_class_by_qualname_with_bases(&key, vec![]))
    }

    pub(crate) fn exception_carrier_handle_key(&self, root: &str) -> bool {
        let key = majit_ir::descr::canonical_struct_name(&normalize_class_qualname(root));
        self.exception_carrier_handle.borrow().as_deref() == Some(key.as_str())
    }

    /// Whether `spelling` names the registered exception carrier handle or
    /// object class. `tyref_is_result_of_carrier` compares the error ADT
    /// to `ErrorCarrierSpec.carrier_path`; this is that compare on a
    /// Charon type string.
    fn exception_carrier_type_spelling(&self, spelling: &str) -> bool {
        let t = peel_ref_and_ptr_prefixes(spelling);
        if self.exception_carrier_handle_key(t) {
            return true;
        }
        let key = majit_ir::descr::canonical_struct_name(&normalize_class_qualname(t));
        if self.exception_carrier.borrow().as_deref() == Some(key.as_str()) {
            return true;
        }
        let leaf = t.rsplit("::").next().unwrap_or(t);
        if self
            .exception_carrier_handle
            .borrow()
            .as_deref()
            .and_then(|k| k.rsplit("::").next())
            == Some(leaf)
        {
            return true;
        }
        self.exception_carrier
            .borrow()
            .as_deref()
            .and_then(|k| k.rsplit("::").next())
            == Some(leaf)
    }

    /// The `__bases__` a first mint of class `key` receives when nothing
    /// else supplies one: `Exception` for the error carrier, none otherwise.
    fn default_class_bases(&self, key: &str) -> Vec<HostObject> {
        if self.exception_carrier.borrow().as_deref() == Some(key) {
            let exception = crate::flowspace::model::HOST_ENV
                .lookup_exception_class("Exception")
                .expect("HOST_ENV missing builtin Exception");
            return vec![exception];
        }
        if self.exception_carrier_handle_key(key)
            && let Some(object_key) = self.exception_carrier.borrow().clone()
            && let Some(host) = self.struct_root_classes.borrow().get(&object_key)
        {
            return vec![host.clone()];
        }
        Vec::new()
    }

    /// Share `CallControl`'s layout table. The handle is the existing map,
    /// not a copy of the built lltypes.
    pub fn set_struct_layouts(&self, layouts: crate::codewriter::call::StructLayoutTable) {
        *self.struct_layouts.borrow_mut() = Some(layouts);
    }

    /// `SomePtr(Ptr(Struct))` for a pointer to the Raw owner `class_root`.
    /// `None` when the owner is not a registered Raw struct with field rows
    /// (a fieldless enum stays the integer shell).
    pub fn raw_struct_ptr_annotation(&self, class_root: &str) -> Option<super::model::SomeValue> {
        let registry = self.struct_fields.borrow();
        let registry = registry.as_ref()?;
        let layouts = self.struct_layouts.borrow().clone()?;
        raw_struct_ptr_from_layout(registry, &layouts, class_root)
    }

    /// SomeInstance of a registered named struct — the shell
    /// `robjmodel_instantiate` writes. Variant payloads (`__pos_N`) use
    /// this so the constructor does not union Instance with the SomePtr
    /// `project_struct_field_type` would mint for a raw-word owner.
    fn struct_instance_annotation(self: &Rc<Self>, field_ty: &str) -> Option<SomeValue> {
        let stripped = field_ty
            .trim()
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim_start_matches("*const ")
            .trim_start_matches("*mut ")
            .trim();
        let named_root = majit_ir::descr::strip_generic_args(stripped);
        let registered = {
            let guard = self.struct_fields.borrow();
            guard
                .as_ref()
                .map(|r| r.fields.contains_key(named_root.as_ref()))
                .unwrap_or(false)
        };
        if !registered {
            return None;
        }
        {
            let canonical = majit_ir::descr::canonical_struct_name(named_root.as_ref());
            if !self.projected_struct_rows.borrow().contains(&canonical) {
                self.pending_struct_row_projection
                    .borrow_mut()
                    .push(named_root.to_string());
            }
        }
        let host = self.intern_class_by_qualname(named_root.as_ref());
        match self.getuniqueclassdef(&host) {
            Ok(classdef) => Some(SomeValue::Instance(super::model::SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            ))),
            Err(_) => None,
        }
    }

    /// True when `field_ty` is a stored raw pointer, or a named struct
    /// whose stored fields include one. A zero-sized field occupies no
    /// word (`StructLayout.fields[..].size`, Charon layout size 0), so a
    /// marker ADT is not a stored pointer. Variant payloads of
    /// stored-pointer owners stay SomePtr so `getfield_raw` matches the
    /// named-field projection. A classed scalar ADT (Fmt, Utf16Or32Form,
    /// RootScope) keeps the SomeInstance `robjmodel_instantiate` writes.
    fn struct_ty_has_raw_pointer_field(&self, field_ty: &str) -> bool {
        let stripped = field_ty
            .trim()
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim();
        if registry_ty_is_raw_ptr(stripped) {
            return true;
        }
        let named_root = majit_ir::descr::strip_generic_args(stripped);
        self.named_ty_has_stored_raw_pointer(named_root.as_ref())
    }

    /// A named ADT stores a raw pointer when a layout field of size > 0
    /// has a `RawPtr` registry type. Size-0 fields are omitted from
    /// `StructLayout.fields` (and Charon reports size 0), so they are
    /// not stored. Without a layout, only `RawPtr` registry rows count —
    /// a marker ADT is a different constructor, not `*mut`/`*const`.
    fn named_ty_has_stored_raw_pointer(&self, named: &str) -> bool {
        let guard = self.struct_fields.borrow();
        let Some(reg) = guard.as_ref() else {
            return false;
        };
        let Some(rows) = reg.field_rows(named) else {
            return false;
        };
        if let Some(layout) = self.layout_of_named(named) {
            return layout.fields.iter().any(|field| {
                field.size > 0
                    && rows
                        .iter()
                        .any(|(name, ty)| name == &field.name && registry_ty_is_raw_ptr(ty))
            });
        }
        rows.iter().any(|(_, ty)| registry_ty_is_raw_ptr(ty))
    }

    fn layout_of_named(
        &self,
        name: &str,
    ) -> Option<std::rc::Rc<crate::codewriter::call::StructLayout>> {
        let layouts = self.struct_layouts.borrow();
        let layouts = layouts.as_ref()?;
        layout_for_owner(layouts, generic_owner_lookup(name)).map(|(_, layout)| layout)
    }

    /// Promote a positional payload SomePtr to SomeInstance only for a
    /// classed scalar ADT written by `robjmodel_instantiate` (Fmt /
    /// Utf16Or32Form / RootScope as an enum payload). A closure capture
    /// and a nullable pointer word keep the SomePtr the writer stores.
    fn variant_payload_promotes_to_instance(
        &self,
        owner: &str,
        field_name: &str,
        field_ty: &str,
        s_value: &SomeValue,
    ) -> bool {
        if !majit_charon_reader::ullbc::is_positional_field_name(field_name)
            || !matches!(s_value, SomeValue::Ptr(_))
        {
            return false;
        }
        let owner_leaf = owner.rsplit("::").next().unwrap_or(owner);
        if majit_charon_reader::ullbc::is_closure_leaf(owner_leaf) {
            return false;
        }
        if self.field_ty_is_nullable_ptr(field_ty) {
            return false;
        }
        !self.struct_ty_has_raw_pointer_field(field_ty)
    }

    /// A niche-optimized pointer-sized enum is the nullable `lltype.Ptr`
    /// (`None` is the null niche), not a classed instance. Decided from
    /// the registered layout (size == pointer word and the host tag is
    /// `TagEncoding::Niche`) or, when that host fact is absent, from a
    /// fieldless enum base (`is_enum_base && !enum_base_has_payload`):
    /// intern would mint `Instance(enum)` over the already-collapsed Ptr.
    /// Every payload enum's base is also `__discriminant` only — the
    /// variants hold the payload (`rclass.py` variant subclass) — so
    /// `is_enum_base` alone is not this fact. `BuiltinEncoder` is that
    /// payload enum: `Option<BuiltinEncoder>::Some.__pos_0` is Instance
    /// so instantiate can setattr a `Utf8` variant.
    fn field_ty_is_nullable_ptr(&self, field_ty: &str) -> bool {
        let stripped = field_ty
            .trim()
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim();
        let named = majit_ir::descr::strip_generic_args(stripped);
        if let Some(layout) = self.layout_of_named(named.as_ref())
            && layout.size == crate::layout::target_word_size()
            && layout.host.as_ref().is_some_and(|host| {
                matches!(
                    host.tag.as_ref(),
                    Some(tag) if matches!(
                        tag.encoding,
                        majit_charon_reader::ullbc::TagEncoding::Niche { .. }
                    )
                )
            })
        {
            return true;
        }
        self.struct_fields.borrow().as_ref().is_some_and(|reg| {
            reg.is_enum_base(named.as_ref()) && !reg.enum_base_has_payload(named.as_ref())
        })
    }

    /// TODO: no upstream equivalent.  Wire the enum
    /// `discriminant → variant` tables consumed by
    /// [`Self::enum_variant_narrowing_knowntypedata`].  Idempotent: a
    /// second call overwrites the previous map.
    pub fn set_enum_variant_by_discriminant(&self, map: Rc<HashMap<String, HashMap<i64, String>>>) {
        *self.enum_variant_by_discriminant.borrow_mut() = Some(map);
    }

    /// TODO: no upstream equivalent.  Wire the trait →
    /// unique-concrete-impl-owner map (see
    /// [`Self::trait_unique_impls`]).  Idempotent: a second call
    /// overwrites the previous map.
    pub fn set_trait_unique_impls(&self, map: HashMap<String, String>) {
        *self.trait_unique_impls.borrow_mut() = map;
    }

    /// TODO: no upstream equivalent.  The full set of struct type-root
    /// keys in the snapshot registry (`StructFieldRegistry.fields`).
    /// The session-prologue inheritance-id pass
    /// (`CallRegistry::ensure_session`) iterates these to eagerly
    /// pre-register the build-time-closed `W_*` struct hierarchy before
    /// running [`crate::translator::rtyper::normalizecalls::assign_inheritance_ids`].
    /// Returns owned `String`s (not an iterator) so the registry borrow
    /// is released before the caller re-borrows it through
    /// [`Self::getuniqueclassdef_for_struct_root`].  Empty when no
    /// registry is set (unit-test fixtures), degrading the pass to a
    /// no-op over whatever classdefs already exist.
    pub fn struct_root_names(&self) -> Vec<String> {
        let mut roots: Vec<String> = self
            .struct_fields
            .borrow()
            .as_ref()
            .map(|reg| reg.fields.keys().cloned().collect())
            .unwrap_or_default();
        // `reg.fields` is a `HashMap`, so its `keys()` order varies
        // run-to-run.  This Vec drives `getuniqueclassdef_for_struct_root`
        // in `ensure_session`, whose first-touch order assigns the global
        // `NEXT_CLASSDEF_ID` counter (`get_unique_cdef_id`), and those ids
        // are the sort witness for the inheritance-id markers.  An unsorted
        // order therefore makes classdef numbering — and the downstream
        // annotate/rtype classification — nondeterministic.  Sort for a
        // stable mint order, matching `pre_register_enum_variant_classes`.
        roots.sort();
        roots
    }

    /// True when `leaf` is the trait-leaf of a registered dispatch family
    /// ([`Self::trait_family_bases`], keyed by the trait's full
    /// `name_path()`).  A method call lowered as `FunctionPath [<leaf>,
    /// method]` (`front::mir`'s `CallKind::Trait` arm spells the trait by
    /// leaf) routes through the receiver's getattr when this matches and
    /// the direct-path registry has no entry (the required-method,
    /// `>=2`-impl case).  Empty unless a consumer opts a trait in.
    pub fn is_registered_trait_family_leaf(&self, leaf: &str) -> bool {
        self.trait_family_bases
            .borrow()
            .keys()
            .any(|path| path.rsplit("::").next().unwrap_or(path) == leaf)
    }

    /// Resolve a bare trait `leaf` to the unique registered dispatch-family
    /// base spelling ([`Self::trait_family_bases`] key = the trait's
    /// full `name_path()`).  The inline-Field `CallTarget::Indirect` carries
    /// the trait by leaf (`dyn_indirect_target` strips the vtable path to its
    /// leaf, matching `register_trait_method`), but the receiver-narrowing
    /// cast (`__cast_instance_intrinsic`) and the family base mint both key on the
    /// qualified spelling — so the leaf must map back to it.  Returns `None`
    /// when the leaf matches no family or matches more than one qualified
    /// path (ambiguous — the caller then keeps the classdef-less receiver
    /// rather than casting to an arbitrary base).
    pub fn registered_trait_family_base_root(&self, leaf: &str) -> Option<String> {
        let bases = self.trait_family_bases.borrow();
        let mut matches = bases
            .keys()
            .filter(|path| path.rsplit("::").next().unwrap_or(path) == leaf);
        let first = matches.next()?.clone();
        match matches.next() {
            Some(_) => None,
            None => Some(first),
        }
    }

    /// No upstream equivalent — forced by pyre's numbering order, not a
    /// casual workaround.  Upstream runs `assign_inheritance_ids` ONCE
    /// inside `RPythonTyper.specialize`, AFTER `annotator.complete()`
    /// reaches its fixpoint, when every instantiated class — variant
    /// classes included — already exists; the single numbering pass sees
    /// the complete set and needs no pre-mint.  pyre cannot follow that
    /// order: it runs incremental per-subject `specialize_more_blocks`
    /// (no single post-annotation driver), and `ClassRepr.fill_vtable_root`
    /// bakes `minid`/`maxid` as eager `Signed` constants into the
    /// cross-graph-cached vtable, so the ids must be stable in the session
    /// prologue — before any per-subject annotation discovers a
    /// lazily-minted variant class.  This pre-mint is the prologue-time
    /// analogue of upstream's "all classes exist before numbering"
    /// invariant, restricted to the enum-variant subtrees that would
    /// otherwise be minted lazily mid-session: it pre-mints the variant
    /// subclasses of every registered enum so the session-prologue
    /// [`crate::translator::rtyper::normalizecalls::assign_inheritance_ids`]
    /// pass numbers each `enum-base + variant-children` subtree as one
    /// contiguous bracket.  Called from `CallRegistry::ensure_session`
    /// AFTER the struct-root loop (so the enum-base classdefs already
    /// exist, UNNUMBERED) and BEFORE the single `assign_inheritance_ids`.
    ///
    /// A variant ctor instantiation (`PyError::type_error` →
    /// `PyErrorKind::TypeError`) mints its variant class LAZILY during
    /// annotation, after the prologue numbering has already baked the
    /// enum base's `[minid,maxid]` bracket.  An on-demand
    /// `assign_inheritance_ids` re-run then cannot number the fresh
    /// child (its bracket would have to nest inside the parent's baked
    /// range — not append-safe), so `ClassesPBCRepr.redispatch_call`
    /// Skip-classifies the instantiation.  Pre-minting the whole subtree
    /// here leaves every member UNNUMBERED at the single numbering pass,
    /// so the contiguous bracket is assigned once and never shifts.
    ///
    /// Spelling MUST match the annotation path or the pre-mint is a
    /// phantom that never matches the lazily-minted class, so each variant
    /// is interned through the SAME [`Self::intern_enum_variant_host`]
    /// primitive the discriminant-narrowing resolver
    /// ([`Self::getuniqueclassdef_for_enum_variant`]) and the variant ctor
    /// arm (`flowspace_adapter`'s `SyntheticTransparentCtor`) route through:
    /// base = `intern_class_by_qualname(canonical_struct_name(root))`,
    /// variant = `intern_class_by_qualname_with_bases(
    /// "{canon_root}::{variant}", [base])`.  All three sites thus resolve
    /// one `HostObject` per variant under the `::`-qualified key.
    /// `enum_variant_by_discriminant` dual-publishes each enum under both
    /// its `::`-qualified path and its bare leaf; iterate the qualified
    /// keys only.  Gated on the same `is_enum_base` predicate the ctor arm
    /// uses (a flat class whose sole row is the synthetic `__discriminant`
    /// tag), so the pre-mint covers exactly the variant subclasses the
    /// adapter can mint — payload-bearing enums included, since their
    /// payloads live under `{enum}::{variant}` keys, not on the base.
    pub fn pre_register_enum_variant_classes(self: &Rc<Self>) {
        // Collect (qualified_root, [variant names]) for every enum base,
        // releasing both registry borrows before minting (the intern /
        // getuniqueclassdef calls re-borrow `struct_fields` and
        // `struct_root_classes`).  Sorted for a deterministic mint
        // order so the numbering bracket is reproducible.
        let mut pairs: Vec<(String, Vec<String>)> = {
            let variant_guard = self.enum_variant_by_discriminant.borrow();
            let fields_guard = self.struct_fields.borrow();
            let (Some(variants), Some(reg)) = (variant_guard.as_ref(), fields_guard.as_ref())
            else {
                return;
            };
            variants
                .iter()
                // A `::`-qualified template root (`module::Enum`) or a
                // per-instantiation root (`Result<Tuple>`, #100) — both
                // name a real enum base.  The bare leaf duplicate
                // (`Enum`) is the other key the registration publishes,
                // skipped here so the subtree numbers once.
                .filter(|(root, _)| root.contains("::") || root.contains('<'))
                .filter_map(|(root, by_discr)| {
                    let leaf = root.rsplit("::").next().unwrap_or(root);
                    // The bare leaf is the usual enum-base alias. Harden
                    // withdraws it when another declaration shares the leaf
                    // (`eval::Code` beside `module::struct::Code`); the
                    // qualified row is still the discriminant-only base, so
                    // a miss on the leaf must not skip the subtree.
                    let stripped = crate::front::mir::strip_crate_prefix(root);
                    let is_base = reg.is_enum_base(leaf)
                        || reg.is_enum_base(root)
                        || (stripped.as_str() != root && reg.is_enum_base(&stripped));
                    is_base.then(|| {
                        let mut names: Vec<String> = by_discr.values().cloned().collect();
                        names.sort();
                        names.dedup();
                        (root.clone(), names)
                    })
                })
                .collect()
        };
        pairs.sort();

        for (root, variant_names) in pairs {
            // Each variant is resolved through the canonical resolver, which
            // materializes the discriminant-only base first
            // ([`Self::getuniqueclassdef_for_struct_root`]) before interning
            // the variant subclass — so the pre-mint and the
            // discriminant-narrowing resolver share one base lineage and the
            // variant subtree still numbers as one contiguous bracket.
            for variant in &variant_names {
                // Resolve each variant through the canonical resolver rather
                // than a bare intern.  It mints the discriminant-only base,
                // interns the variant subclass under the same `::`-qualified
                // key the discriminant-narrowing resolver and the variant ctor
                // arm use (so all three resolve ONE variant classdef, no
                // `.`-vs-`::` sibling split the single numbering pass misses),
                // AND projects the variant's payload rows — draining any
                // struct the payload first reaches — before it returns.
                //
                // The payload projection is what a bare intern skipped: a
                // variant field typed `*mut PyObject` would otherwise stay an
                // untyped FORCE shell until subject-flow narrowing first
                // resolves it (`enum_variant_narrowing_knowntypedata` ->
                // `getuniqueclassdef_for_enum_variant`), where a first-intern
                // of the pointee struct mid-fixpoint generalises an
                // already-annotated cell.  Projecting here — at prologue time,
                // before `assign_inheritance_ids` and any subject flow, with
                // the pending drain inside the call — makes the variant's
                // payload classdefs order-independent, the same contract the
                // struct-root loop above relies on.
                let _ = self.getuniqueclassdef_for_enum_variant(&root, variant);
            }
        }
    }

    /// Push a classdef into [`Self::needs_generic_instantiate`] unless
    /// it is already present (upstream `dict[cdef] = True` idempotence).
    pub fn push_needs_generic_instantiate(&self, classdef: &Rc<RefCell<ClassDef>>) {
        self.needs_generic_instantiate
            .borrow_mut()
            .entry(ClassDefKey::from_classdef(classdef))
            .or_insert_with(|| classdef.clone());
    }

    /// Wire up the `self.annotator` backlink. Invoked from
    /// [`RPythonAnnotator::new`] via `Rc::new_cyclic` so the
    /// bookkeeper's weak reference points to the final `Rc<Self>`.
    pub fn set_annotator(&self, ann: Weak<crate::annotator::annrpython::RPythonAnnotator>) {
        *self.annotator.borrow_mut() = ann;
    }

    /// RPython `self.annotator` attribute access — upgrades the weak
    /// backlink to an `Rc<RPythonAnnotator>`. Panics if the backlink
    /// is absent or the annotator has been dropped; both are programmer
    /// errors mirroring upstream's assumption that `self.annotator`
    /// is always live.
    pub fn annotator(&self) -> Rc<crate::annotator::annrpython::RPythonAnnotator> {
        self.annotator
            .borrow()
            .upgrade()
            .expect("Bookkeeper.annotator backlink is absent or dropped")
    }

    /// Non-panicking variant of [`Self::annotator`].  Returns `None`
    /// when the backlink is absent (test fixtures that wire the
    /// bookkeeper without an `RPythonAnnotator`) or when the
    /// `Rc<RPythonAnnotator>` has been dropped.  Production callers
    /// that need the annotator should prefer [`Self::annotator`];
    /// this variant is for diagnostic side-channels that should
    /// degrade gracefully when the annotator is not yet attached
    /// (e.g. lift-error recording from a registry pre-pass that may
    /// also be exercised in standalone fixture setups).
    pub fn try_annotator(&self) -> Option<Rc<crate::annotator::annrpython::RPythonAnnotator>> {
        self.annotator.borrow().upgrade()
    }

    /// RPython `Bookkeeper.warning(self, msg)` (bookkeeper.py).
    ///
    /// ```python
    /// def warning(self, msg):
    ///     return self.annotator.warning(msg)
    /// ```
    pub fn warning(&self, msg: impl Into<String>) {
        let msg = msg.into();
        self.annotator().warning(&msg);
    }

    /// RPython `bookkeeper.position_key = ...` assignment. Returns the
    /// previous value so callers can restore it around a nested reflow
    /// (matches upstream bookkeeper.py:278 `@contextmanager
    /// position()`).
    #[cfg(test)]
    pub(crate) fn set_position_key(&self, pk: Option<PositionKey>) -> Option<PositionKey> {
        self.position_key.replace(pk)
    }

    /// Current `bookkeeper.position_key`. Returns `None` when no
    /// reflow frame is active (upstream's initial
    /// `self.position_key = None`).
    pub(crate) fn current_position_key(&self) -> Option<PositionKey> {
        self.position_key.borrow().clone()
    }

    /// RPython `Bookkeeper.enter(self, position_key)` (bookkeeper.py).
    ///
    /// Installs the position and registers `self` as the thread-local
    /// bookkeeper so [`getbookkeeper`] returns it. Asserts that no
    /// `enter` is currently active — matches upstream's `not hasattr`
    /// check.
    pub(crate) fn enter(self: &Rc<Self>, position_key: Option<PositionKey>) {
        assert!(!self.position_entered.get(), "don't call enter() nestedly");
        self.position_entered.set(true);
        self.position_key.replace(position_key);
        // Upstream: `TLS.bookkeeper = self` (bookkeeper.py).
        super::model::TLS.with(|state| state.borrow_mut().bookkeeper = Some(Rc::clone(self)));
    }

    /// RPython `Bookkeeper.leave(self)` (bookkeeper.py).
    ///
    /// Clears both the position slot and the thread-local bookkeeper
    /// hook. Safe to call only after a matching [`Self::enter`].
    pub(crate) fn leave(&self) {
        self.position_entered.set(false);
        self.position_key.replace(None);
        // Upstream: `del TLS.bookkeeper` (bookkeeper.py).
        super::model::TLS.with(|state| state.borrow_mut().bookkeeper = None);
    }

    /// RPython `Bookkeeper.at_position(self, pos)` (bookkeeper.py).
    ///
    /// RAII port of the upstream `@contextmanager`. The `pos=None`
    /// fast-path (line 99-101) short-circuits the enter/leave pair when
    /// the bookkeeper is already inside a reflow frame — used by
    /// `compute_at_fixpoint` to let nested callers reuse the ambient
    /// position.
    pub(crate) fn at_position(self: &Rc<Self>, pos: Option<PositionKey>) -> PositionGuard {
        if self.position_entered.get() && pos.is_none() {
            // Upstream: `if hasattr(self, 'position_key') and pos is None: yield; return`
            PositionGuard {
                bk: Rc::clone(self),
                skip_leave: true,
            }
        } else {
            self.enter(pos);
            PositionGuard {
                bk: Rc::clone(self),
                skip_leave: false,
            }
        }
    }

    /// RPython `Bookkeeper.check_no_flags_on_instances(self)`
    /// (bookkeeper.py) — post-annotation sanity check invoked
    /// by `RPythonAnnotator.validate()`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def check_no_flags_on_instances(self):
    ///     seen = set()
    ///     def check_no_flags(s_value_or_def):
    ///         if isinstance(s_value_or_def, SomeInstance):
    ///             assert not s_value_or_def.flags, ...
    ///             check_no_flags(s_value_or_def.classdef)
    ///         elif isinstance(s_value_or_def, SomeList):
    ///             check_no_flags(s_value_or_def.listdef.listitem)
    ///         elif isinstance(s_value_or_def, SomeDict):
    ///             check_no_flags(s_value_or_def.dictdef.dictkey)
    ///             check_no_flags(s_value_or_def.dictdef.dictvalue)
    ///         elif isinstance(s_value_or_def, SomeTuple):
    ///             for s_item in s_value_or_def.items:
    ///                 check_no_flags(s_item)
    ///         elif isinstance(s_value_or_def, ClassDef):
    ///             if s_value_or_def in seen: return
    ///             seen.add(s_value_or_def)
    ///             for attr in s_value_or_def.attrs.itervalues():
    ///                 check_no_flags(attr.s_value)
    ///         elif isinstance(s_value_or_def, ListItem):
    ///             if s_value_or_def in seen: return
    ///             seen.add(s_value_or_def)
    ///             check_no_flags(s_value_or_def.s_value)
    ///
    ///     for clsdef in self.classdefs:
    ///         check_no_flags(clsdef)
    /// ```
    pub fn check_no_flags_on_instances(&self) {
        let mut seen_classdefs: HashSet<ClassDefKey> = HashSet::new();
        let mut seen_listitems: HashSet<usize> = HashSet::new();
        for clsdef in self.classdefs.borrow().iter() {
            check_no_flags_classdef(clsdef, &mut seen_classdefs, &mut seen_listitems);
        }
    }

    /// RPython `Bookkeeper.compute_at_fixpoint(self)`
    /// (bookkeeper.py) — invoked at the tail of
    /// `RPythonAnnotator.simplify()`.
    ///
    /// ```python
    /// def compute_at_fixpoint(self):
    ///     # getbookkeeper() needs to work during this function, so provide
    ///     # one with a dummy position
    ///     with self.at_position(None):
    ///         for call_op in self.annotator.call_sites():
    ///             self.consider_call_site(call_op)
    ///         for pbc, args_s in self.emulated_pbc_calls.itervalues():
    ///             args = simple_args(args_s)
    ///             pbc.consider_call_site(args, s_ImpossibleValue, None)
    ///         self.emulated_pbc_calls = {}
    /// ```
    ///
    /// Structural 1:1 port. `SomePBC::consider_call_site` currently
    /// routes through `DescKind.consider_call_site` with a bridging
    /// gap (see that method's doc); the outer scaffolding — at_position
    /// guard, call_sites drain, emulated_pbc_calls drain + clear — is
    /// parity-faithful.
    ///
    /// Errors from `consider_call_site` propagate verbatim: upstream
    /// `bookkeeper.py` runs both inner loops without a
    /// `try`/`except`, so a raised exception terminates
    /// `compute_at_fixpoint` and unwinds out of
    /// `RPythonAnnotator.simplify`.  Pyre's port mirrors this with
    /// early `?`-propagation; callers that previously ignored the
    /// `Result` (the `let _ = ...` shape that Issue 3.1 flagged) are
    /// migrated to propagate or to panic on uncaught errors,
    /// matching upstream's Python-exception semantics.
    pub fn compute_at_fixpoint(self: &Rc<Self>) -> Result<(), AnnotatorError> {
        let _guard = self.at_position(None);
        let ann = self.annotator();
        for (call_op, op_key) in ann.call_sites_with_positions() {
            self.consider_call_site(&call_op, Some(op_key))?;
        }
        // Snapshot values so we can mutate emulated_pbc_calls inside the
        // loop (upstream does `for pbc, args_s in
        // self.emulated_pbc_calls.itervalues()` while not mutating).
        let emulated: Vec<(SomePBC, Vec<SomeValue>)> =
            self.emulated_pbc_calls.borrow().values().cloned().collect();
        for (pbc, args_s) in emulated {
            // upstream: `args = simple_args(args_s)`;
            //            `pbc.consider_call_site(args, s_ImpossibleValue, None)`.
            let args = simple_args(args_s);
            pbc.consider_call_site(&args, &SomeValue::Impossible, None)?;
        }
        // upstream: `self.emulated_pbc_calls = {}`.
        self.emulated_pbc_calls.borrow_mut().clear();
        Ok(())
    }

    /// RPython `Bookkeeper.consider_call_site(self, call_op)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def consider_call_site(self, call_op):
    ///     from rpython.rtyper.llannotation import SomeLLADTMeth, lltype_to_annotation
    ///     annotation = self.annotator.annotation
    ///     s_callable = annotation(call_op.args[0])
    ///     args_s = [annotation(arg) for arg in call_op.args[1:]]
    ///     if isinstance(s_callable, SomeLLADTMeth):
    ///         adtmeth = s_callable
    ///         s_callable = self.immutablevalue(adtmeth.func)
    ///         args_s = [lltype_to_annotation(adtmeth.ll_ptrtype)] + args_s
    ///     if isinstance(s_callable, SomePBC):
    ///         s_result = annotation(call_op.result)
    ///         if s_result is None:
    ///             s_result = s_ImpossibleValue
    ///         args = call_op.build_args(args_s)
    ///         s_callable.consider_call_site(args, s_result, call_op)
    /// ```
    ///
    /// `call_op.build_args`
    /// is opname-sensitive (`simple_call` →
    /// `ArgumentsForTranslation(list(args_s))`, `call_args` →
    /// `fromshape`); the simple_call path uses [`simple_args`].
    pub(crate) fn consider_call_site(
        self: &Rc<Self>,
        call_op: &crate::flowspace::model::SpaceOperation,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        let ann = self.annotator();
        let Some(mut s_callable) = ann.annotation(&call_op.args[0]) else {
            return Ok(());
        };
        // upstream `args_s = [annotation(arg) for arg in call_op.args[1:]]`
        // (`bookkeeper.py:156`) preserves positional arity *and* the
        // `None` sentinel for not-yet-annotated args (Python list of
        // `SomeValue or None`).  Pyre's `arguments_w:
        // Vec<Option<SomeValue>>` carries None unchanged so downstream
        // `pbc_call → FunctionDesc.pycall → recursivecall` can route
        // through unbound positions exactly as upstream
        // `description.py` does (each unbound position skips
        // its `binding_join` until the fixpoint round populates it).
        let mut args_s: Vec<Option<SomeValue>> = call_op
            .args
            .iter()
            .skip(1)
            .map(|a| ann.annotation(a))
            .collect();
        if let SomeValue::LLADTMeth(adtmeth) = &s_callable {
            let func = adtmeth.func.clone();
            let ll_ptrtype = adtmeth.ll_ptrtype.clone();
            s_callable = self.immutablevalue(&func)?;
            // upstream `bookkeeper.py:158`: args_s = [lltype_to_
            // annotation(adtmeth.ll_ptrtype)] + args_s — the prepended
            // ll_ptrtype is always concrete.
            args_s.insert(
                0,
                Some(crate::translator::rtyper::llannotation::lltype_to_annotation(ll_ptrtype)),
            );
        }
        if let SomeValue::PBC(pbc) = &s_callable {
            let s_result = ann
                .annotation(&call_op.result)
                .unwrap_or(SomeValue::Impossible);
            // upstream: `args = call_op.build_args(args_s)` —
            // opname-sensitive: `simple_call` → `ArgumentsForTranslation
            // (list(args_s))`, `call_args` → `fromshape(args_s[0].const,
            // args_s[1:])`.
            let args = build_args_for_op(&call_op.opname, &args_s)?;
            pbc.consider_call_site(&args, &s_result, op_key)?;
        }
        Ok(())
    }

    /// RPython `Bookkeeper.getattr_locations(self, clsdesc, attrname)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def getattr_locations(self, clsdesc, attrname):
    ///     attrdef = clsdesc.classdef.find_attribute(attrname)
    ///     return attrdef.read_locations
    /// ```
    ///
    /// Returns a snapshot of the attribute's read locations — callers
    /// iterate + reflow outside the borrow so [`ClassDef::find_attribute`]
    /// can reacquire the RefCell if it generalizes the attribute.
    pub(crate) fn getattr_locations(
        &self,
        classdesc: &Rc<RefCell<ClassDesc>>,
        attrname: &str,
    ) -> Result<IndexSet<PositionKey>, AnnotatorError> {
        // upstream: `attrdef = clsdesc.classdef.find_attribute(attrname)`
        let classdef = classdesc
            .borrow()
            .classdef
            .clone()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| {
                AnnotatorError::new(
                    "Bookkeeper.getattr_locations: classdesc.classdef is not set \
                     (requires ClassDesc.getuniqueclassdef — classdesc.py:699-702)"
                        .to_string(),
                )
            })?;
        // `find_attribute` may `generalize_attr` as a side effect; run
        // it outside any borrow of `classdef`.
        let _ = ClassDef::find_attribute(&classdef, attrname)?;
        // Now re-lookup the attribute to read its read_locations.
        let attr = classdef.borrow();
        let attrdef = attr.attrs.get(attrname).ok_or_else(|| {
            AnnotatorError::new(format!(
                "Bookkeeper.getattr_locations: attribute {:?} missing on {:?}",
                attrname, attr.name
            ))
        })?;
        Ok(attrdef.read_locations.clone())
    }

    /// RPython `Bookkeeper.record_getattr(self, clsdesc, attrname)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def record_getattr(self, clsdesc, attrname):
    ///     locations = self.getattr_locations(clsdesc, attrname)
    ///     locations.add(self.position_key)
    /// ```
    ///
    /// The Rust port cannot return the mutable set by reference without
    /// extending the RefCell borrow across the caller's chain, so it
    /// inserts directly into the attribute's `read_locations`.
    pub fn record_getattr(
        &self,
        classdesc: &Rc<RefCell<ClassDesc>>,
        attrname: &str,
    ) -> Result<(), AnnotatorError> {
        let classdef = classdesc
            .borrow()
            .classdef
            .clone()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| {
                AnnotatorError::new(
                    "Bookkeeper.record_getattr: classdesc.classdef is not set".to_string(),
                )
            })?;
        // Ensure the attribute exists (upstream's
        // `clsdesc.classdef.find_attribute(attrname)` side effect).
        let _ = ClassDef::find_attribute(&classdef, attrname)?;
        // upstream: `locations.add(self.position_key)`
        if let Some(pk) = self.current_position_key() {
            let mut classdef_mut = classdef.borrow_mut();
            let trace_class_name =
                crate::determinism_trace_enabled().then(|| classdef_mut.name.clone());
            if let Some(attrdef) = classdef_mut.attrs.get_mut(attrname)
                && attrdef.read_locations.insert(pk)
                && let Some(class_name) = trace_class_name
            {
                eprintln!(
                    "[DTRACE-ATTR] record_getattr class={class_name} attr={attrname} nloc={}",
                    attrdef.read_locations.len()
                );
            }
        }
        Ok(())
    }

    /// RPython `Bookkeeper.update_attr(self, clsdef, attrdef)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def update_attr(self, clsdef, attrdef):
    ///     locations = self.getattr_locations(clsdef.classdesc, attrdef.name)
    ///     for position in locations:
    ///         self.annotator.reflowfromposition(position)
    ///     attrdef.validate(homedef=clsdef)
    /// ```
    ///
    /// Rust signature takes the attribute name (rather than an
    /// `&mut Attribute`) because the attribute lives inside
    /// `clsdef.borrow_mut().attrs`; validate is executed after a
    /// `remove`/`insert` cycle so the attribute is the exclusive mutator
    /// while the `homedef` is borrowed read-only.
    pub fn update_attr(
        &self,
        clsdef: &Rc<RefCell<ClassDef>>,
        attr_name: &str,
    ) -> Result<(), AnnotatorError> {
        let classdesc = clsdef.borrow().classdesc.clone();
        // upstream: `locations = self.getattr_locations(clsdef.classdesc, attrdef.name)`
        let locations = self.getattr_locations(&classdesc, attr_name)?;
        if crate::determinism_trace_enabled() {
            eprintln!(
                "[DTRACE-ATTR] update_attr class={} attr={attr_name} nloc={}",
                clsdef.borrow().name,
                locations.len()
            );
        }
        // upstream: `for position in locations: self.annotator.reflowfromposition(position)`
        let Some(ann) = self.annotator.borrow().upgrade() else {
            // upstream always has `self.annotator`; if the backlink is
            // not wired yet (tests constructing a Bookkeeper without an
            // annotator) skip reflow silently — same effect as upstream
            // when the pending-block queue is empty.
            return Ok(());
        };
        for position in locations {
            REFLOW_FROM_ATTR.fetch_add(1, Ordering::Relaxed);
            ann.reflowfromposition(&position);
        }
        // upstream: `attrdef.validate(homedef=clsdef)`
        let taken = clsdef.borrow_mut().attrs.remove(attr_name);
        if let Some(mut attrdef) = taken {
            let result = attrdef.validate(clsdef);
            clsdef
                .borrow_mut()
                .attrs
                .insert(attr_name.to_string(), attrdef);
            result.map_err(|e| AnnotatorError::new(e.to_string()))?;
        }
        Ok(())
    }

    /// RPython `Bookkeeper.valueoftype(self, t)` (bookkeeper.py).
    ///
    /// Thin wrapper around [`crate::annotator::signature::annotationoftype`]
    /// used by `binaryop.is_` and the PBC-call site machinery to seed
    /// annotations for type-level constants.
    pub fn valueoftype(
        self: &Rc<Self>,
        spec: &crate::annotator::signature::AnnotationSpec,
    ) -> Result<SomeValue, crate::annotator::signature::SignatureError> {
        crate::annotator::signature::annotationoftype(spec, Some(self))
    }

    /// RPython `Bookkeeper.getlistdef(**flags_if_new)` (bookkeeper.py).
    ///
    /// Returns the (cached or freshly constructed) ListDef for the
    /// bookkeeper's current position. Upstream stores flags inside the
    /// `listitem.__dict__`; Rust carries the `range_step` flag
    /// explicitly (the only non-default flag any caller passes — see
    /// bookkeeper.py).
    ///
    /// The current position — including `None` — is used as the cache
    /// key directly, matching upstream's `self.listdefs[self.position_
    /// key]` indexing. Two calls with no active position share the
    /// same ListDef just like two calls inside the same reflow frame
    /// would.
    pub fn getlistdef(self: &Rc<Self>, range_step: Option<i64>) -> ListDef {
        let pk = self.current_position_key();
        let mut listdefs = self.listdefs.borrow_mut();
        if let Some(existing) = listdefs.get(&pk) {
            return existing.clone();
        }
        let new_ld = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
        if let Some(step) = range_step {
            let li = new_ld.inner.listitem.borrow().clone();
            li.borrow_mut().range_step = Some(step);
        }
        listdefs.insert(pk, new_ld.clone());
        new_ld
    }

    /// Bind a list-typed spelling (`Vec<T>`, `VecDeque<T>`, `[T; N]`) to
    /// the `ListDef` of the current `position_key`.
    ///
    /// Field rows mint a fresh `ListDef` into `ClassDef.attrs[name].s_value`
    /// (`Attribute.s_value`). `__cast_instance_intrinsic` / FUNC.ARGS
    /// re-project the same spelling at an operation, and that site is
    /// `Bookkeeper.getlistdef` — two analyses of one op share the ListDef
    /// so `setbinding` / `SomeObject.contains` can succeed under
    /// `TLS.no_side_effects_in_union`. Named structs that *model* as a
    /// list (`FixedObjectArray._items`) keep the attribute identity:
    /// this only rebinds spellings `is_list_container_spelling` accepts.
    pub fn project_list_spelling_at_position(self: &Rc<Self>, spelling: &str) -> SomeValue {
        let projected = self.project_struct_field_type(spelling);
        if !majit_ir::descr::is_list_container_spelling(spelling) {
            return projected;
        }
        let SomeValue::List(list) = projected else {
            return projected;
        };
        let listdef = self.getlistdef(None);
        let _ = listdef.generalize(&list.listdef.s_value());
        SomeValue::List(SomeList::new(listdef))
    }

    /// RPython `Bookkeeper.newlist(*s_values, **flags)` (bookkeeper.py).
    pub fn newlist(
        self: &Rc<Self>,
        s_values: &[SomeValue],
        range_step: Option<i64>,
    ) -> Result<SomeList, AnnotatorError> {
        let listdef = self.getlistdef(range_step);
        for s_value in s_values {
            listdef
                .generalize(s_value)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        if let Some(step) = range_step {
            listdef
                .generalize_range_step(Some(step))
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeList::new(listdef))
    }

    /// RPython `Bookkeeper.getdictdef(is_r_dict=False,
    /// force_non_null=False, simple_hash_eq=False)` (bookkeeper.py).
    ///
    /// `None` position caches just like `Some(pk)`, matching upstream's
    /// `self.dictdefs[self.position_key]` indexing. See [`Self::
    /// getlistdef`] for the rationale.
    pub fn getdictdef(
        self: &Rc<Self>,
        is_r_dict: bool,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> DictDef {
        let pk = self.current_position_key();
        let mut dictdefs = self.dictdefs.borrow_mut();
        if let Some(existing) = dictdefs.get(&pk) {
            return existing.clone();
        }
        let new_dd = DictDef::new(
            Some(self.clone()),
            SomeValue::Impossible,
            SomeValue::Impossible,
            is_r_dict,
            force_non_null,
            simple_hash_eq,
        );
        dictdefs.insert(pk, new_dd.clone());
        new_dd
    }

    /// RPython `Bookkeeper.newdict()` (bookkeeper.py).
    pub fn newdict(self: &Rc<Self>) -> SomeDict {
        SomeDict::new(self.getdictdef(false, false, false))
    }

    /// RPython `Bookkeeper.getdesc(pyobj)` (bookkeeper.py).
    ///
    /// Returns the cached [`DescEntry`] for `pyobj`, or constructs a
    /// fresh one per upstream's `isinstance` dispatch. The Rust port
    /// branches on [`HostObject`] kind:
    ///   * `UserFunction` → [`Self::newfuncdesc`]
    ///   * `Class` → `ClassDesc` (c1 shell — c2c replaces this with the
    ///     full `ClassDesc::__init__` body; the shell still satisfies
    ///     identity-caching so basedesc lookups land in a shared Rc)
    ///   * `BoundMethod` → `MethodDesc` / `MethodOfFrozenDesc`
    ///   * `Instance` / `BuiltinCallable` / `Module` / `Opaque` →
    ///     [`Self::getfrozen`] (upstream's `_freeze_` fallback)
    pub(crate) fn getdesc(
        self: &Rc<Self>,
        pyobj: &HostObject,
    ) -> Result<DescEntry, AnnotatorError> {
        if let Some(existing) = self.descs.borrow().get(pyobj) {
            return Ok(existing.clone());
        }
        let entry = if pyobj.is_user_function() {
            // upstream `newfuncdesc` already returns a MemoDesc or
            // FunctionDesc per the specializer.
            self.newfuncdesc(pyobj)?
        } else if pyobj.is_class() {
            // upstream bookkeeper.py:367-373 — pyobj is `object` check
            // raises, and `__builtin__` module check routes to
            // `getfrozen`. The Rust port currently treats every non-
            // builtin HostObject-class as a ClassDesc; builtin types
            // aren't modelled as HostObject::Class yet (they show up as
            // BuiltinCallable / primitive ConstValue), so the branch
            // isn't reachable. `object` identity isn't materialised
            // either.
            let name = pyobj.qualname().to_string();
            let desc_rc = ClassDesc::new(self, pyobj.clone(), Some(name), None, None)?;
            DescEntry::Class(desc_rc)
        } else if pyobj.is_bound_method() {
            let self_obj = pyobj.bound_method_self().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.getdesc(bound method): missing __self__")
            })?;
            let func = pyobj.bound_method_func().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.getdesc(bound method): missing __func__")
            })?;
            // Keep the whole `FuncDescEntry` (not `as_function()`, which
            // unwraps a memo to its base) so a bound memo method's
            // `MethodDesc.funcdesc` retains the exact MemoDesc identity.
            let funcdesc = self
                .getdesc(func)?
                .as_func_entry()
                .cloned()
                .ok_or_else(|| {
                    AnnotatorError::new(
                        "Bookkeeper.getdesc(bound method): __func__ is not a function",
                    )
                })?;

            if self_obj.is_instance() {
                call_cleanup_method(self_obj)?;
            }
            if self_obj.is_instance() && call_freeze_method(self_obj)? {
                let frozendesc = self.getdesc(self_obj)?.as_frozen().ok_or_else(|| {
                    AnnotatorError::new(
                        "Bookkeeper.getdesc(bound method): frozen self did not produce FrozenDesc",
                    )
                })?;
                DescEntry::MethodOfFrozen(Rc::new(RefCell::new(
                    super::description::MethodOfFrozenDesc::new(self.clone(), funcdesc, frozendesc),
                )))
            } else {
                let (origin_class, name) = origin_of_meth(pyobj)?;
                let self_class = if self_obj.is_class() {
                    self_obj.clone()
                } else {
                    self_obj.instance_class().cloned().ok_or_else(|| {
                        AnnotatorError::new(
                            "Bookkeeper.getdesc(bound method): regular method self has no class",
                        )
                    })?
                };
                let classdef = self.getuniqueclassdef(&self_class)?;
                if self_obj.is_instance() {
                    super::classdesc::ClassDef::see_instance(&classdef, self_obj)?;
                }
                let _ = super::classdesc::ClassDef::find_attribute(&classdef, &name)?;
                DescEntry::Method(self.getmethoddesc(
                    &funcdesc,
                    ClassDefKey::from_classdef(&self.getuniqueclassdef(origin_class)?),
                    Some(ClassDefKey::from_classdef(&classdef)),
                    &name,
                    std::collections::BTreeMap::new(),
                ))
            }
        } else if pyobj.is_builtin_callable()
            || pyobj.is_instance()
            || pyobj.is_module()
            || pyobj.is_opaque()
        {
            DescEntry::Frozen(self.getfrozen(pyobj)?)
        } else {
            return Err(AnnotatorError::new(format!(
                "Bookkeeper.getdesc({:?}): unexpected prebuilt constant",
                pyobj.qualname()
            )));
        };
        self.descs.borrow_mut().insert(pyobj.clone(), entry.clone());
        Ok(entry)
    }

    /// RPython `Bookkeeper.newfuncdesc(pyfunc)` (bookkeeper.py).
    ///
    /// Rust port: pull signature / defaults from the HostObject's
    /// [`crate::flowspace::model::GraphFunc`], and request a
    /// specializer from `AnnotatorPolicy.get_specializer(tag)`. When the
    /// specializer is `memo`, return a [`DescEntry::Memo`] wrapping the
    /// `FunctionDesc` (upstream `if specializer is memo: return MemoDesc(
    /// ...)`, bookkeeper.py); otherwise a [`DescEntry::Function`].
    pub(crate) fn newfuncdesc(
        self: &Rc<Self>,
        pyfunc: &HostObject,
    ) -> Result<DescEntry, AnnotatorError> {
        let gf = pyfunc.user_function().ok_or_else(|| {
            AnnotatorError::new(format!(
                "newfuncdesc({:?}) called on non-user-function HostObject",
                pyfunc.qualname()
            ))
        })?;
        // upstream bookkeeper.py `signature = cpython_code_signature(pyfunc.__code__)`.
        let signature = match gf.code.as_ref() {
            Some(code) => cpython_code_signature(code),
            // No HostCode attached — upstream hits the
            // `_generator_next_method_of_` branch (bookkeeper.py)
            // or fails. The Rust port defaults to the single-arg
            // `Signature(['entry'])` matching upstream's generator
            // fallback so tests that wire GraphFunc without a HostCode
            // still traverse.
            None => Signature::new(vec!["entry".to_string()], None, None),
        };
        self.newfuncdesc_with_signature(pyfunc, signature)
    }

    /// The native-source adapter supplies the signature read from its own
    /// code object (the LLBC declaration), instead of fabricating HostCode
    /// bytecode. All callable attributes, policy selection and MemoDesc
    /// construction still follow Bookkeeper.newfuncdesc's single path.
    pub(crate) fn newfuncdesc_with_signature(
        self: &Rc<Self>,
        pyfunc: &HostObject,
        signature: Signature,
    ) -> Result<DescEntry, AnnotatorError> {
        let gf = pyfunc.user_function().ok_or_else(|| {
            AnnotatorError::new("native function descriptor requires a user function")
        })?;
        let name = gf.name.clone();
        let defaults = if gf.defaults.is_empty() {
            None
        } else {
            Some(gf.defaults.clone())
        };
        // Prefer the live annotator policy (respects `using_policy`
        // swap) when the backlink is present; fall back to the
        // bookkeeper-owned static `policy` slot in unit-test
        // configurations that construct `Bookkeeper::new()` without
        // attaching an annotator.
        let specializer = match self.annotator.borrow().upgrade() {
            Some(ann) => ann
                .policy
                .borrow()
                .get_specializer(gf.annspecialcase.as_deref())
                .map_err(|e| AnnotatorError::new(e.to_string()))?,
            None => self
                .policy
                .get_specializer(gf.annspecialcase.as_deref())
                .map_err(|e| AnnotatorError::new(e.to_string()))?,
        };
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            self.clone(),
            Some(pyfunc.clone()),
            name,
            signature,
            defaults,
            Some(specializer),
        )));
        // upstream: `if specializer is memo: return MemoDesc(...) else:
        // return FunctionDesc(...)`.
        if fd.borrow().is_memo() {
            Ok(DescEntry::memo(Rc::new(RefCell::new(
                super::description::MemoDesc::new(fd),
            ))))
        } else {
            Ok(DescEntry::function(fd))
        }
    }

    /// RPython `Bookkeeper.getfrozen(pyobj)` (bookkeeper.py).
    pub fn getfrozen(
        self: &Rc<Self>,
        pyobj: &HostObject,
    ) -> Result<Rc<RefCell<FrozenDesc>>, AnnotatorError> {
        let fd = FrozenDesc::new(self.clone(), pyobj.clone())?;
        Ok(Rc::new(RefCell::new(fd)))
    }

    /// RPython `Bookkeeper.getmethoddesc(funcdesc, originclassdef,
    /// selfclassdef, name, flags={})` (bookkeeper.py).
    ///
    /// Caches MethodDescs by the `(funcdesc-id, origindef-id,
    /// selfdef-id, name, flags)` tuple — upstream's Python tuple hash
    /// keyed on identity for the descriptor / classdef entries.
    pub(crate) fn getmethoddesc(
        self: &Rc<Self>,
        funcdesc: &super::description::FuncDescEntry,
        originclassdef: ClassDefKey,
        selfclassdef: Option<ClassDefKey>,
        name: &str,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Rc<RefCell<MethodDesc>> {
        let flags_vec: Vec<(String, bool)> = flags.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let key = MethodDescKey {
            // `FuncDescEntry::desc_key()` is the memo wrapper's identity
            // for a memo (the base FunctionDesc's for a plain function),
            // so a memo method keys to the same identity the PBC set
            // carries.
            funcdesc_id: funcdesc.desc_key(),
            originclassdef,
            selfclassdef,
            name: name.to_string(),
            flags: flags_vec,
        };
        if let Some(existing) = self.methoddescs.borrow().get(&key) {
            return existing.clone();
        }
        let md = MethodDesc::new(
            self.clone(),
            funcdesc.clone(),
            originclassdef,
            selfclassdef,
            name,
            flags,
        );
        let rc = Rc::new(RefCell::new(md));
        self.methoddescs.borrow_mut().insert(key, rc.clone());
        rc
    }

    /// Codewriter-time method-desc discovery.
    ///
    /// Walks `receiver_classdef`'s MRO for the first `SomePBC` attr
    /// under `name`, then applies `classdesc.py:344-365` filtering
    /// rules inline to collect all qualifying MethodDescs.
    ///
    /// Classdef-attrs read-only: uses `get_owner` / `attrs.get`
    /// instead of `find_attribute` (which calls `locate_attribute`
    /// → `generalize_attr_internal` mutating classdef attrs — a
    /// phase violation at codewriter time). Does prime the
    /// `Bookkeeper.methoddescs` cache via `getmethoddesc` calls.
    pub fn getmethoddesc_for_attribute(
        self: &Rc<Self>,
        receiver_classdef: &Rc<RefCell<super::classdesc::ClassDef>>,
        name: &str,
    ) -> Vec<Rc<RefCell<super::description::MethodDesc>>> {
        use super::description::DescEntry;
        use super::model::SomeValue;
        let receiver_key = super::description::ClassDefKey::from_classdef(receiver_classdef);
        let mro = super::classdesc::ClassDef::getmro(receiver_classdef);
        for cdef in mro {
            let s_value = cdef.borrow().attrs.get(name).map(|a| a.s_value.clone());
            let Some(SomeValue::PBC(pbc)) = s_value else {
                continue;
            };
            // classdesc.py:344-365 filtering rules applied inline.
            // Does not call ClassDef::lookup_filter (which calls
            // bind_self); instead calls getmethoddesc directly to
            // prime the methoddescs cache.
            let mut d: Vec<Rc<RefCell<super::description::MethodDesc>>> = Vec::new();
            let mut uplookup: Option<Rc<RefCell<super::classdesc::ClassDef>>> = None;
            let mut up_md: Option<Rc<RefCell<super::description::MethodDesc>>> = None;
            for entry in pbc.descriptions.values() {
                let DescEntry::Method(md) = entry else {
                    continue;
                };
                let (funcdesc, originclassdef, name_owned, flags, existing_self) = {
                    let md_borrow = md.borrow();
                    (
                        md_borrow.funcdesc.clone(),
                        md_borrow.originclassdef,
                        md_borrow.name.clone(),
                        md_borrow.flags.clone(),
                        md_borrow.selfclassdef,
                    )
                };
                if existing_self.is_some() {
                    // Already-bound: kept verbatim (classdesc.py:347).
                    d.push(self.getmethoddesc(
                        &funcdesc,
                        originclassdef,
                        existing_self,
                        &name_owned,
                        flags,
                    ));
                    continue;
                }
                let Some(methclassdef) = self.lookup_classdef(originclassdef) else {
                    continue;
                };
                let methclassdef_is_receiver = Rc::ptr_eq(&methclassdef, receiver_classdef);
                let meth_issub_receiver = methclassdef.borrow().issubclass(receiver_classdef);
                let receiver_issub_meth = receiver_classdef.borrow().issubclass(&methclassdef);
                if !methclassdef_is_receiver && meth_issub_receiver {
                    // Subclass-origin: bind to methclassdef
                    // (classdesc.py:349-364).
                    let methclassdef_key =
                        super::description::ClassDefKey::from_classdef(&methclassdef);
                    d.push(self.getmethoddesc(
                        &funcdesc,
                        originclassdef,
                        Some(methclassdef_key),
                        &name_owned,
                        flags,
                    ));
                } else if receiver_issub_meth {
                    // Upward: track best (classdesc.py:351-356).
                    let promote = match &uplookup {
                        None => true,
                        Some(cur) => methclassdef.borrow().issubclass(cur),
                    };
                    if promote {
                        uplookup = Some(methclassdef.clone());
                        up_md = Some(md.clone());
                    }
                }
            }
            // Upward best-match bound to receiver
            // (classdesc.py:366-367).
            if let Some(up_md_rc) = up_md {
                let (funcdesc, originclassdef, name_owned, flags) = {
                    let b = up_md_rc.borrow();
                    (
                        b.funcdesc.clone(),
                        b.originclassdef,
                        b.name.clone(),
                        b.flags.clone(),
                    )
                };
                d.push(self.getmethoddesc(
                    &funcdesc,
                    originclassdef,
                    Some(receiver_key),
                    &name_owned,
                    flags,
                ));
            }
            // classdesc.py:369-374 — once the owner's PBC is found,
            // return the filtered result even if empty. Does not
            // continue to parent attrs.
            return d;
        }
        Vec::new()
    }

    /// RPython `Bookkeeper.get_classpbc_attr_families(attrname)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def get_classpbc_attr_families(self, attrname):
    ///     map = self.classpbc_attr_families
    ///     try:
    ///         access_sets = map[attrname]
    ///     except KeyError:
    ///         access_sets = map[attrname] = UnionFind(description.ClassAttrFamily)
    ///     return access_sets
    /// ```
    ///
    /// Rust exposes the lookup via a closure so the returned mutable
    /// handle doesn't escape the `RefCell::borrow_mut()` borrow. The
    /// attrname's UnionFind is materialised on first access with the
    /// `ClassAttrFamily`-factory, matching upstream's lazy creation.
    pub(crate) fn with_classpbc_attr_families<T>(
        &self,
        attrname: &str,
        f: impl FnOnce(&mut UnionFind<DescKey, Rc<RefCell<ClassAttrFamily>>>) -> T,
    ) -> T {
        let mut map = self.classpbc_attr_families.borrow_mut();
        let entry = map.entry(attrname.to_string()).or_insert_with(|| {
            UnionFind::new(|desc: &DescKey| Rc::new(RefCell::new(ClassAttrFamily::new(*desc))))
        });
        f(entry)
    }

    /// RPython `bookkeeper.classdefs.append(classdef)` — invoked from
    /// `ClassDesc._init_classdef` (classdesc.py). Callers hand over
    /// the fresh `Rc<RefCell<ClassDef>>` so the bookkeeper retains the
    /// identity alongside every other reachable classdef.
    pub fn register_classdef(self: &Rc<Self>, classdef: Rc<RefCell<ClassDef>>) {
        self.classdefs.borrow_mut().push(classdef);
    }

    pub(crate) fn lookup_classdef(&self, key: ClassDefKey) -> Option<Rc<RefCell<ClassDef>>> {
        self.classdefs
            .borrow()
            .iter()
            .find(|classdef| ClassDefKey::from_raw(Rc::as_ptr(classdef) as usize) == key)
            .cloned()
    }

    /// Snapshot of every registered classdef — test helper + upstream
    /// `bookkeeper.classdefs` read access.
    pub fn classdef_snapshot(&self) -> Vec<Rc<RefCell<ClassDef>>> {
        self.classdefs.borrow().clone()
    }

    /// Redirect a bare-leaf `root` whose field alias was withdrawn by
    /// `harden_duplicate_leaf_metadata` (a cross-namespace leaf collision —
    /// e.g. a struct leaf colliding with an enum variant of the same name) to
    /// the unique surviving *qualified* struct key that carries that leaf.
    /// A withdrawn bare leaf has no `fields` entry, so `project_struct_rows`
    /// finds nothing and the classdef stays attrs-empty (every field getattr
    /// on the cast result then blocks). The qualified key survives hardening
    /// (only the bare alias is removed), so resolving through it projects the
    /// struct's rows. Returns `None` — leaving the current bare-leaf behaviour
    /// — unless there is exactly one surviving non-enum-variant qualified key
    /// with this leaf: a genuinely ambiguous leaf (>=2 real struct keys) stays
    /// unresolved, and an unwithdrawn leaf is untouched.
    fn redirect_withdrawn_struct_leaf(self: &Rc<Self>, root: &str) -> Option<String> {
        // Only a bare leaf can be a withdrawal victim; a qualified or
        // per-instantiation spelling already survives hardening.
        if root.contains("::") || root.contains('<') {
            return None;
        }
        let guard = self.struct_fields.borrow();
        let reg = guard.as_ref()?;
        if reg.fields.contains_key(root) {
            return None;
        }
        let mut hit: Option<&String> = None;
        for k in reg.fields.keys() {
            let Some((head, leaf)) = k.rsplit_once("::") else {
                continue;
            };
            if leaf != root {
                continue;
            }
            // An `{Enum}::{Variant}` key's tail is a variant name, not a
            // struct leaf — the enum base carries the `__discriminant` row.
            if reg.is_enum_base(head) {
                continue;
            }
            if hit.is_some() {
                return None;
            }
            hit = Some(k);
        }
        hit.cloned()
    }

    /// Resolve a struct type-root name to its canonical, bookkeeper-
    /// REGISTERED `ClassDef`, with instance attributes projected from
    /// the threaded `StructFieldRegistry`.
    ///
    /// pyre's `W_IntObject` /
    /// `PyFrame` / `W_DictObject` Rust structs are ports of RPython
    /// classes (`W_Root` subclasses), so their annotation is
    /// `SomeInstance(classdef)` — `InstanceRepr._setup_repr`
    /// (`rclass.py`) later lowers `classdef.attrs` into
    /// `Ptr(GcStruct(OBJECT, inst_<field>...))`.  RPython derives a
    /// `ClassDef` ONLY by class-object identity
    /// (`getuniqueclassdef(cls)` -> `getdesc(cls)` -> id() key,
    /// `bookkeeper.py`, `uid.py`); there is no
    /// name->classdef path.  Pyre mirrors this by resolving the
    /// type-root string to a host class OBJECT exactly once
    /// ([`Self::intern_class_by_qualname`]), then routing through the
    /// existing [`Self::getuniqueclassdef`] so the `ClassDef` lands in
    /// `descs`/`classdefs` keyed by identity.
    ///
    /// The struct fields are instance attributes: RPython discovers them
    /// by annotating `setattr`/`__init__`, but pyre has no such pass for
    /// host structs, so the registry projection (via
    /// [`Self::project_struct_field_type`]) stands in.  The whole reachable
    /// struct-field graph is registered and projected — an explicit
    /// work-list bounds recursion at the reachable-struct count — so a
    /// field typed as another struct resolves to a fully populated inner
    /// `SomeInstance(classdef)`, not an attrs-empty inner stub.
    ///
    /// Order-independent w.r.t. [`Self::set_struct_fields`]: the call
    /// re-traverses and re-projects every time, so a root registered
    /// before the field registry arrived has its (and its transitive
    /// structs') attrs back-filled on a later call — RPython grows class
    /// attrs as annotation proceeds.  Identity is stable across calls
    /// (the host class is interned once), so the back-fill mutates the
    /// already-published `ClassDef` rather than minting a fresh one.
    pub fn getuniqueclassdef_for_struct_root(
        self: &Rc<Self>,
        root: &str,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        let determinism_trace = crate::determinism_trace_enabled();
        let trace_root = root;
        if determinism_trace {
            eprintln!("[DTRACE-CLASS] struct_root root={trace_root}");
        }
        let redirected = self.redirect_withdrawn_struct_leaf(root);
        let root: &str = redirected.as_deref().unwrap_or(root);
        // Pass 1 — traverse the registry's struct-field graph from `root`,
        // registering an identity-keyed `ClassDef` in `descs` for every
        // reachable struct (via `intern_class_by_qualname` ->
        // `getuniqueclassdef`).  `project_struct_field_type`'s bare-name arm
        // resolves inner struct references through the same identity path,
        // so it sees the registered classdef.  Registration is idempotent
        // (the identity cache is the memo), so a root reached before the
        // field registry arrived still has its transitive structs
        // discovered — and its attrs re-projected in pass 2 — on a later
        // call.  Cycles are broken by testing membership against `graph`
        // itself (the
        // insertion-order result list): the reachable-struct count is small
        // (bounded by the registry, not the 200+ struct-chain depth that
        // would overflow a recursive walk), so a linear scan over `graph`
        // needs no parallel hash index and lets every node move into
        // `graph` without a clone.
        let mut worklist: Vec<String> = vec![root.to_string()];
        let mut graph: Vec<String> = Vec::new();
        while let Some(n) = worklist.pop() {
            if graph.contains(&n) {
                continue;
            }
            // Register the node's identity ClassDef in `descs` (RPython's
            // HostObject-identity cache, bookkeeper.py:361) via
            // `intern_class_by_qualname` -> `getuniqueclassdef`.  Idempotent:
            // interning resolves the type-root name to one stable
            // `HostObject`, so `getuniqueclassdef` returns the same
            // `ClassDef` on every call — the identity cache is the memo, no
            // name-keyed side table needed.
            let host = self.intern_class_by_qualname(&n);
            // Materialise the classdef lineage base-most first:
            // `getdesc` and `ClassDef::new` each recurse into their
            // base (classdesc.py / :672) and both memoise, so
            // pre-seeding every base keeps the native recursion depth
            // at one frame even when the embedded-header chain is
            // registry-deep (same worklist discipline as `graph`).
            let mut lineage: Vec<HostObject> = vec![host];
            while let Some(base) = lineage
                .last()
                .and_then(|h| h.class_bases().and_then(|b| b.first().cloned()))
            {
                lineage.push(base);
            }
            for h in lineage.iter().rev() {
                self.getuniqueclassdef(h)?;
            }
            let referenced: Vec<String> = {
                let guard = self.struct_fields.borrow();
                if let Some(reg) = guard.as_ref() {
                    if let Some(fields) = reg.fields.get(&n) {
                        let mut out: Vec<String> = Vec::new();
                        for (field_name, field_ty) in fields {
                            if field_name == "__class__" {
                                continue;
                            }
                            collect_referenced_struct_names(field_ty, reg, &mut out);
                        }
                        out
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            for r in referenced {
                if !graph.contains(&r) {
                    worklist.push(r);
                }
            }
            // `n` is no longer borrowed (registry lookup done); move it into
            // the ordered result list without a clone.
            graph.push(n);
        }
        if determinism_trace {
            eprintln!(
                "[DTRACE-CLASS] struct_graph root={trace_root} n={} structs={}",
                graph.len(),
                graph.join(",")
            );
        }
        // Pass 2 — project each reachable node's fields into its
        // `classdef.attrs`.  All nodes are published from pass 1, so a
        // field referencing another struct resolves to the registered
        // inner classdef.  Idempotent: the `Impossible` guard fills an attr
        // exactly once, so re-running after the field registry arrives
        // back-fills a root that was cached attrs-empty without disturbing
        // attrs already set (RPython grows class attrs as annotation
        // proceeds).
        for n in &graph {
            self.project_struct_rows(n)?;
        }
        // Drain structs first interned by `project_struct_field_type`'s
        // bare-name arm during the loop above.  Pass-1's
        // `collect_referenced_struct_names` string parser and the
        // projector resolve type strings independently; a name only
        // the projector reaches would otherwise keep its untyped
        // FORCE shells until a LATER root intern projects it — a
        // non-monotonic mid-session attr flip for any block that
        // already read the shell (`setbinding: new value does not
        // contain old`).  Projecting here closes the window: rows
        // land before this struct_root call returns, i.e. before any
        // subject flow can read the class's attrs.
        loop {
            let next = self.pending_struct_row_projection.borrow_mut().pop();
            let Some(n) = next else { break };
            self.project_struct_rows(&n)?;
        }
        let host = self.intern_class_by_qualname(root);
        self.getuniqueclassdef(&host)
    }

    /// TODO: no upstream equivalent.  The closest thing upstream,
    /// `Bookkeeper.getdesc` / `getuniqueclassdef` (`bookkeeper.py`,
    /// `:353-363`), keys identity on the Python object; pyre has no such
    /// object and keys on a qualname.
    /// Resolve the subclass `ClassDef` for one variant of a Rust enum —
    /// a subclass of the enum's discriminant-only base. Composes the canonical
    /// interning primitives rather than minting a parallel enum-class
    /// registry: the base is the struct-root `ClassDef`
    /// ([`Self::getuniqueclassdef_for_struct_root`]); the variant is
    /// interned with that base as its sole `__bases__`
    /// ([`Self::intern_class_by_qualname_with_bases`]), so the resulting
    /// `ClassDef.basedef` chains through the base. That makes
    /// `commonbase(base, variant) == base`, which is exactly what
    /// `pairtype(SomeInstance, SomeInstance).improve` (binaryop.py)
    /// needs to narrow a `SomeInstance(base)` to `SomeInstance(variant)`.
    /// Identity is the canonical struct-root cache keyed by
    /// `canonical_struct_name`; the base `enum_root` and the
    /// `canon_root::variant` path ([`Self::intern_enum_variant_host`])
    /// each normalise to one stable `HostObject`, so repeated calls
    /// return the same `Rc` — and the variant `Rc` is the very one the
    /// prologue pre-mint numbered.
    pub fn getuniqueclassdef_for_enum_variant(
        self: &Rc<Self>,
        enum_root: &str,
        variant_name: &str,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        // Base first — registers the discriminant-only root and publishes
        // its identity-keyed HostObject in the struct-root cache.
        self.getuniqueclassdef_for_struct_root(enum_root)?;
        // Variant subclass — interned WITH the base so ClassDesc::new
        // wires basedef. The SAME interning primitive the variant
        // CONSTRUCTOR path uses ([`Self::intern_enum_variant_host`]), so
        // both sides resolve one class object. Must precede any plain
        // intern of this path (intern_*_with_bases returns the
        // first-minted class verbatim).
        let variant_host = self.intern_enum_variant_host(enum_root, variant_name);
        // Project the variant's OWN payload rows onto the subclass —
        // the RPython sum-type layout where each subclass carries its
        // own fields (`rclass.py:499-518`), the base only the discriminant.
        // A getattr on a narrowed `SomeInstance(variant)` then resolves
        // the payload on the variant via the MRO (variant → base), with
        // the variant's own field type.  The rows are registered under
        // the canonical variant key in `front::mir`.
        let canon_root = majit_ir::descr::canonical_struct_name(enum_root);
        let variant_path = format!("{canon_root}::{variant_name}");
        self.project_struct_rows(&variant_path)?;
        // A variant payload can be the first place a dependency struct is
        // reached (`ConstantData::Complex.value: Complex<f64>`).  The payload
        // projector queues that struct to avoid recursively borrowing the
        // registry, just like an ordinary struct row does.  Drain it before
        // publishing the variant classdef; otherwise the variant's `value`
        // attr may be classed while `value.re` / `value.im` still see the
        // FORCE shell.  This is the variant counterpart of the pending drain
        // in `getuniqueclassdef_for_struct_root`.
        loop {
            let next = self.pending_struct_row_projection.borrow_mut().pop();
            let Some(n) = next else { break };
            self.project_struct_rows(&n)?;
        }
        let classdef = self.getuniqueclassdef(&variant_host)?;
        // Every payload row on a Rust enum variant is assigned by that
        // variant's constructor. In RPython terms these are mutable instance
        // attributes, not readonly class attributes: `Attribute.modified`
        // runs during annotation before `InstanceRepr._setup_repr` builds its
        // field table. Pyre pre-mints variants at session start so inheritance
        // ids stay stable; mark the projected payload rows at that same point
        // to preserve RPython's attrs-before-repr ordering even when another
        // subject caches the repr before this constructor is annotated.
        {
            let mut classdef_mut = classdef.borrow_mut();
            for attr in classdef_mut.attrs.values_mut() {
                attr.modified(Some(&classdef))?;
            }
        }
        Ok(classdef)
    }

    /// Register a trait family for receiver-driven method dispatch: a base
    /// `ClassDef` `base_root` with each `(impl_root, members)` interned as a
    /// subclass, carrying method MEMBERS so `getattr(instance, method)`
    /// resolves to a `MethodDesc` PBC family — the RPython instance-method
    /// [`MethodsPBCRepr`](crate::translator::rtyper::rpbc::MethodsPBCRepr)
    /// dispatch path.
    ///
    /// Mirrors [`Self::getuniqueclassdef_for_enum_variant`] (a base plus
    /// subclasses interned WITH that base as their sole `__bases__`, so
    /// `ClassDef.basedef`/`subdefs`/`getmro` chain), but seeds method members
    /// through [`HostObject::new_class_with_members`].  Plain pyre struct
    /// classdefs are minted member-less by
    /// [`Self::intern_class_by_qualname`] (`HostObject::new_class`) and carry
    /// only fields (`project_struct_rows`); a `dyn Trait` receiver therefore
    /// annotates to a classdef-less shell and every `getattr` on it fails.
    /// The members seeded here are what `ClassDesc::find_source_for` reads via
    /// `pyobj.class_get(name)` → `add_source_attribute`, so the trait methods
    /// become classdict attributes and `s_getattr` yields the PBC family.
    ///
    /// `base_members` (trait default-body methods) attach to the base;
    /// each impl's `members` (its required-method overrides) attach to that
    /// subclass — mirroring the trait's default-vs-required split, so the MRO
    /// getattr binds each method to its correct owner.
    ///
    /// OPT-IN: only a trait registered through this entry point gets a
    /// member-carrying host in `struct_root_classes`.  Every other
    /// (unregistered) multi-impl trait keeps its current classdef-less /
    /// fail-loud disposition, so this cannot perturb the annotation of pyre
    /// production dispatch that was never registered.
    ///
    /// Returns the base `ClassDef`.
    pub fn register_trait_family(
        self: &Rc<Self>,
        base_root: &str,
        // The ordered containers preserve the host class-dict insertion order.
        base_members: IndexMap<String, ConstValue>,
        impls: Vec<(String, IndexMap<String, ConstValue>)>,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        // Base first — its identity-keyed HostObject must be published in
        // `struct_root_classes` before the subclasses intern, so each
        // subclass's `__bases__` resolves to this exact base host (the same
        // discipline `getuniqueclassdef_for_enum_variant` relies on).
        let base_key = majit_ir::descr::canonical_struct_name(base_root);
        let base_host = crate::flowspace::model::HostObject::new_class_with_members(
            base_key.clone(),
            Vec::new(),
            base_members,
        );
        self.struct_root_classes
            .borrow_mut()
            .insert(base_key, base_host.clone());
        // Index the base by the raw `base_root` spelling too, so the
        // receiver seed (`derive_subject_inputcells`) can match a
        // `dyn Trait` receiver's `class_root` (the trait's `name_path()`)
        // directly without re-canonicalising.
        self.trait_family_bases
            .borrow_mut()
            .insert(base_root.to_string(), base_host.clone());
        let base_cd = self.getuniqueclassdef(&base_host)?;
        // Subclasses — interned WITH the base host so `ClassDesc::new` wires
        // `basedef` and `_init_classdef` pushes each into `base.subdefs`.
        for (impl_root, members) in impls {
            let impl_key = majit_ir::descr::canonical_struct_name(&impl_root);
            let impl_host = crate::flowspace::model::HostObject::new_class_with_members(
                impl_key.clone(),
                vec![base_host.clone()],
                members,
            );
            self.struct_root_classes
                .borrow_mut()
                .insert(impl_key, impl_host.clone());
            self.getuniqueclassdef(&impl_host)?;
        }
        Ok(base_cd)
    }

    /// Intern (first-mint-wins) and return the canonical variant-subclass
    /// `HostObject` for `{enum_root}::{variant_name}`, wiring the enum's
    /// discriminant-only base as its sole `__bases__` (`rclass.py:517-518`:
    /// `self.rbase = getinstancerepr(rtyper, self.classdef.basedef, ...)`).
    ///
    /// Both the discriminant-narrowing path
    /// ([`Self::getuniqueclassdef_for_enum_variant`]) and the variant
    /// CONSTRUCTOR path (`flowspace_adapter`'s `SyntheticTransparentCtor`
    /// arm) intern through here, so a constructed `Some(x)` and a matched
    /// `Some(x)` resolve to ONE class object — the single-class-per-variant
    /// identity RPython gets from one Python class object
    /// (`bookkeeper.py` keys `getuniqueclassdef` by the class object,
    /// not a name string).  Splitting them into sibling classdefs would
    /// union to the base (losing the payload narrowing) and disagree on
    /// the payload attr/field owner.
    ///
    /// The cache key is `canonical_struct_name(enum_root)::variant_name`.
    /// A leaf `enum_root` (`"Color"`, the ctor side, which carries the
    /// enum leaf as its owner tail) and a canonical `enum_root`
    /// (`"module::Color"`, the narrowing side, which reads the base
    /// classdef's canonical name) normalise to the same key because
    /// `canonical_struct_name` resolves the leaf through
    /// `STRUCT_ORIGIN_REGISTRY` — the same normalisation the base itself
    /// went through.
    ///
    /// `enum_root` must carry the LLBC instantiation suffix when the ADT
    /// is generic (`Option<i64>`, not `Option`). `canonical_struct_name`
    /// reattaches a `<…>` that is already on the spelling and otherwise
    /// leaves the name ungeneric, so two instantiations interned under
    /// the bare template share one `ClassDef` and their payloads union
    /// (`bookkeeper.py` `getuniqueclassdef(cls)` — one class per class).
    pub fn intern_enum_variant_host(
        self: &Rc<Self>,
        enum_root: &str,
        variant_name: &str,
    ) -> HostObject {
        let canon_root = majit_ir::descr::canonical_struct_name(enum_root);
        // Field reads spell the crate-stripped template
        // (`pyopcode::StepResult`).  A monomorphized or crate-qualified
        // root (`pyre_interpreter::pyopcode::StepResult<*mut PyObject>`)
        // subclasses that template so the variant unions with the field.
        let template = {
            let bare = canon_root.split('<').next().unwrap_or(&canon_root);
            crate::front::mir::strip_crate_prefix(bare)
        };
        let template_host = self.intern_class_by_qualname(&template);
        let enum_host = if canon_root == template {
            template_host
        } else {
            self.intern_class_by_qualname_with_bases(&canon_root, vec![template_host])
        };
        let variant_path = format!("{canon_root}::{variant_name}");
        self.intern_class_by_qualname_with_bases(&variant_path, vec![enum_host])
    }

    /// Build the discriminant→variant narrowing `knowntypedata` for a
    /// Rust enum's `__discriminant` read.  Each entry is keyed by the
    /// integer tag value and narrows the `receiver` variable to the
    /// variant subclass ([`Self::getuniqueclassdef_for_enum_variant`]),
    /// so a `match disc { k => ... }` switch refines `SomeInstance(base)`
    /// to `SomeInstance(variant_k)` in arm `k` through `follow_link`'s
    /// `improve` (binaryop.py).  `enum_root` is the receiver class
    /// name; the variant table is dual-keyed by qualified path and bare
    /// leaf, so either spelling resolves.  A receiver already narrowed to
    /// a variant subclass (`importing::GcdCache::Miss`) has no `<…>` for
    /// [`majit_ir::descr::strip_instantiation_suffix`] to cut, and its
    /// final segment is the variant, not the enum leaf — walk each `::`
    /// parent (and that parent's bare leaf) so the read resolves the same
    /// table as the base.  Returns `Ok(None)` when the class is not a
    /// registered enum root (the read then stays a plain `SomeInteger`).
    /// Once a registered root's table is found, a variant classdef that
    /// cannot be built is an [`AnnotatorError`], not a silent `None`.
    pub fn enum_variant_narrowing_knowntypedata(
        self: &Rc<Self>,
        enum_root: &str,
        receiver: &Rc<crate::flowspace::model::Variable>,
    ) -> Result<Option<super::model::KnownTypeData>, AnnotatorError> {
        let by_discr = {
            // The discriminant→variant table is keyed by the bare
            // charon-template root (one template per generic ADT), so a
            // per-instantiation receiver name (`Result<bool>`) must drop
            // its `<…>` suffix to resolve.  Bare names pass through
            // unchanged.  A variant-qualified name with no `<…>`
            // (`module::Enum::Variant`) is not reduced by that cut; each
            // `::` parent is tried, exact key then bare leaf, until a
            // registered root hits.
            let lookup_root = majit_ir::descr::strip_instantiation_suffix(enum_root);
            let guard = self.enum_variant_by_discriminant.borrow();
            let Some(map) = guard.as_ref() else {
                return Ok(None);
            };
            let mut cursor = lookup_root;
            let found = loop {
                if let Some(hit) = map.get(cursor) {
                    break Some(hit);
                }
                if let Some(leaf) = cursor.rsplit("::").next()
                    && leaf != cursor
                    && let Some(hit) = map.get(leaf)
                {
                    break Some(hit);
                }
                let Some((parent, _)) = cursor.rsplit_once("::") else {
                    break None;
                };
                cursor = parent;
            };
            let Some(found) = found else {
                return Ok(None);
            };
            found.clone()
        };
        // A receiver already narrowed to a variant subclass
        // (`Option<T>::None`) reaches this read with `enum_root` carrying
        // the `::variant` tail; appending another variant
        // (`Option<T>::None::None`) deepens the per-instantiation classdef
        // every pass, so the lattice never reaches a fixpoint.  Re-narrow
        // against the base enum — strip a trailing `::<variant>` segment so
        // the discriminant maps back to the SAME variant subclasses
        // (idempotent on re-read).
        let base_enum_root: &str = {
            let mut base = enum_root;
            for variant_name in by_discr.values() {
                if let Some(stripped) = base.strip_suffix(&format!("::{variant_name}")) {
                    base = stripped;
                    break;
                }
            }
            base
        };
        let mut ktd = super::model::KnownTypeData::new();
        for (discr, variant_name) in &by_discr {
            let variant_cd =
                self.getuniqueclassdef_for_enum_variant(base_enum_root, variant_name)?;
            let s_variant = SomeValue::Instance(super::model::SomeInstance::new(
                Some(variant_cd),
                false,
                std::collections::BTreeMap::new(),
            ));
            super::model::add_knowntypedata(
                &mut ktd,
                super::model::ExitCaseKey::Int(*discr),
                std::slice::from_ref(receiver),
                s_variant,
            );
        }
        Ok(Some(ktd))
    }

    /// TODO: no upstream equivalent.  The closest thing upstream,
    /// `Bookkeeper.getdesc` / `getuniqueclassdef` (`bookkeeper.py`,
    /// `:353-363`), keys identity on the Python object; pyre has no such
    /// object and keys on a qualname.
    /// Project one struct's registry rows into its `ClassDef.attrs`
    /// — the pass-2 body of [`Self::getuniqueclassdef_for_struct_root`].
    fn project_struct_rows(self: &Rc<Self>, n: &str) -> Result<(), AnnotatorError> {
        // A monomorphised generic spelling (`Option<*mut PyObject>`,
        // `Result<Tuple>`) normally shares the bare template's rows. A
        // payload variant is different: `register_ref_enum_instantiation_rows`
        // publishes an exact suffixed key with its substituted payload type.
        // Consult that exact key first, then strip the `<…>` argument span
        // only as the template fallback — matching
        // `StructFieldRegistry::lookup_fields` while preserving the concrete
        // variant row RPython's completed ClassDef would carry.
        //
        // A per-shape positional aggregate (`Tuple<FrameDebugData>` or
        // `Array<PyObjectRef;2>`) is the exception: it has no template — its
        // concrete element spellings are registered under the full shaped
        // key, and stripping would look up a bare root that carries no rows.
        // Without the exact lookup every `__pos_N` keeps the untyped FORCE shell
        // (`valuetype_to_someshell(Ref) -> SomeInstance(classdef=None)`), which
        // `generalize_attr` can only widen, so a reference-typed tuple element
        // stays classdef-less however precisely its writers are annotated.
        let filtered_nullable_fields: Vec<String> = {
            if !majit_ir::descr::is_shaped_tuple_name(n)
                && !majit_ir::descr::is_shaped_array_name(n)
            {
                Vec::new()
            } else {
                let guard = self.struct_fields.borrow();
                guard
                    .as_ref()
                    .and_then(|r| r.fields.get(n))
                    .into_iter()
                    .flat_map(|fields| fields.iter())
                    .filter(|(_, ty)| is_nullable_sum_spelling(ty))
                    .map(|(name, _)| name.clone())
                    .collect()
            }
        };
        let mut fields: Vec<(String, String)> = {
            if let Some(rows) = crate::front::mir::mut_ref_shape_rows(n) {
                rows.clone()
            } else {
                let guard = self.struct_fields.borrow();
                match guard.as_ref().and_then(|r| {
                    if majit_ir::descr::is_shaped_tuple_name(n)
                        || majit_ir::descr::is_shaped_array_name(n)
                    {
                        r.fields.get(n).cloned()
                    } else {
                        r.fields.get(n).cloned().or_else(|| {
                            r.fields
                                .get(majit_ir::descr::strip_generic_args(n).as_ref())
                                .cloned()
                        })
                    }
                }) {
                    Some(f) => f,
                    None => return Ok(()),
                }
            }
        };
        if majit_ir::descr::is_shaped_tuple_name(n) || majit_ir::descr::is_shaped_array_name(n) {
            // Every `__pos_N` of a shaped positional aggregate is written by
            // the frontend's constructor sequence (synthetic ctor then one
            // `setattr` per element), so a projected row has to agree with the
            // value that sequence produces. A sum-typed element does not: the
            // constructor materializes a variant INSTANCE, while
            // `project_struct_field_type` models the spelling as the nullable
            // payload — `Option<Vec<*mut PyObject>>` projects to list-or-none
            // and then fails `generalize_attr` against
            // `Instance(Option<Vec<*mut PyObject>>::None)`.  Leave those rows
            // to the constructor.
            fields.retain(|(_, ty)| !is_nullable_sum_spelling(ty));
        }
        // A per-instantiation variant carries concrete payload rows keyed
        // under its full `<…>`-suffixed spelling
        // (`Option<Vec<u8>>::Some` → `__pos_0: Vec<u8>`, registered by
        // `register_ref_enum_instantiation_rows`), while the stripped
        // template row is the generic `??TypeVar` that projects to
        // `Impossible`. Substitute that unresolved template field with the
        // exact row. The registrar has already rejected unresolved,
        // multi-word, and layout-ambiguous payloads, so every row reaching
        // this point is safe to project.
        //
        // This ordering is load-bearing parity with RPython:
        // `RPythonTyper.specialize` runs only after annotation has completed
        // every `ClassDef.attrs` entry. Pyre's incremental session can cache
        // an `InstanceRepr` before a later constructor flows its `setattr`;
        // pre-projecting the concrete row keeps repr setup independent of
        // subject order instead of relying on that constructor side effect.
        {
            let guard = self.struct_fields.borrow();
            if let Some(exact) = guard.as_ref().and_then(|r| r.fields.get(n)) {
                for (fname, fty) in &mut fields {
                    if let Some((_, exact_ty)) = exact.iter().find(|(en, _)| en == fname) {
                        if fty.contains("??") && !exact_ty.contains("??") {
                            *fty = exact_ty.clone();
                        }
                    }
                }
            }
        }
        // Bare and qualified spellings of one struct intern to the
        // same canonical class — project its rows once.
        let canonical = majit_ir::descr::canonical_struct_name(n);
        if !self.projected_struct_rows.borrow_mut().insert(canonical) {
            return Ok(());
        }
        let host = self.intern_class_by_qualname(n);
        let classdef = self.getuniqueclassdef(&host)?;
        // The constructor materializes the nullable sum variant in each
        // filtered tuple field.  `register_struct_fields` may already have
        // seeded that same slot with the untyped Ref force shell, but the
        // constructor's `generalize_attr` must start from lattice bottom or
        // the classless shell wins the union and discards the variant
        // ClassDef.  Reset ONLY rows removed by the nullable-sum retain;
        // real annotation flow remains monotonic.
        if !filtered_nullable_fields.is_empty() {
            let mut classdef_mut = classdef.borrow_mut();
            for field_name in &filtered_nullable_fields {
                let Some(attr) = classdef_mut.attrs.get_mut(field_name) else {
                    continue;
                };
                let is_untyped_force_shell = matches!(
                    &attr.s_value,
                    SomeValue::Instance(inst)
                        if inst.classdef.is_none()
                            && !inst.can_be_none
                            && inst.flags.is_empty()
                            && inst.base.const_box.is_none()
                );
                if is_untyped_force_shell {
                    attr.s_value = SomeValue::Impossible;
                }
            }
        }
        for (field_name, field_ty) in &fields {
            if field_name == "__class__" {
                continue;
            }
            let s_value = self.project_struct_field_type(field_ty);
            // A variant payload (`__pos_N`) is written by
            // `robjmodel_instantiate` + setattr, which is always
            // SomeInstance (`builtin.py` `robjmodel_instantiate`).
            // Flattening a classed scalar ADT (Fmt / Utf16Or32Form)
            // to SomePtr made `llannotation.py`
            // `pairtype(SomePtr, SomeObject).union` refuse Instance ∪
            // Ptr at the constructor. A payload whose rows include a
            // raw pointer (PosixMap.ptr: *mut u8) stays SomePtr so
            // `getfield_raw` matches the named-field projection
            // (`raw_struct_ptr_annotation`). Named raw fields
            // (`mapped`, `sin_addr`) and FUNC.ARGS `&Raw` are not
            // `__pos_N` and stay SomePtr either way.
            //
            // A closure capture is the captured value, not a variant
            // constructor: `__cast_instance_intrinsic` / FUNC.ARGS of a
            // raw owner write SomePtr (`load_attr` RootScope). Promoting
            // that slot to Instance made setattr refuse Ptr ∪ Instance.
            // A niche-optimized pointer-sized enum already collapsed to
            // the nullable Ptr (`project_struct_field_type` Option arm,
            // None is the null word); promoting that to Instance(enum)
            // is the mmap_get_attr_i64 `Ptr(MappedObj) ∪ Instance(enum)`
            // refusal.
            let s_value =
                if self.variant_payload_promotes_to_instance(n, field_name, field_ty, &s_value) {
                    self.struct_instance_annotation(field_ty).unwrap_or(s_value)
                } else {
                    s_value
                };
            let mut classdef_mut = classdef.borrow_mut();
            let attr = classdef_mut
                .attrs
                .entry(field_name.clone())
                .or_insert_with(|| super::classdesc::Attribute::new(field_name.clone()));
            // The slot may also hold the untyped FORCE_ATTRIBUTES
            // shell: `register_struct_fields` seeds Ref-typed rows
            // through `valuetype_to_someshell`, which renders every
            // `ValueType::Ref(_)` as the bare
            // `SomeInstance(classdef=None)` before this projection
            // runs (`_init_classdef` applies the force list
            // eagerly).  That shell carries strictly less
            // information than this registry projection (typed
            // lists, dicts, strings, classed instances), so
            // replace it the same way the `Impossible` placeholder
            // is filled — but ONLY that exact shell; any other
            // value was produced by real annotation flow and stays
            // (RPython grows class attrs monotonically).
            let is_untyped_force_shell = matches!(
                &attr.s_value,
                SomeValue::Instance(inst)
                    if inst.classdef.is_none()
                        && !inst.can_be_none
                        && inst.flags.is_empty()
                        && inst.base.const_box.is_none()
            );
            // `tyref_to_attr_value_type` puts `Ptr(FuncType)` in the Int
            // bank (`history.getkind`), so FORCE shells a `fn` /
            // `Option<fn>` field as the Int-bank shell
            // (`valuetype_to_someshell(Int)`: no const, signed,
            // nonneg=false). The layout spelling is still a function
            // pointer, and `project_struct_field_type` answers
            // `SomePtr`. Replace ONLY that exact FORCE shell on a
            // function-pointer spelling: it carries strictly less
            // information than the registry projection, and leaving it
            // makes a `fn` field read union `Integer ∪ Ptr(Func)` with
            // `fn_null_constant`. Any other value was produced by real
            // annotation flow and stays.
            let is_fn_integer_force_shell = fn_ptr_somevalue_for_spelling_in(field_ty, Some(self))
                .is_some()
                && crate::codewriter::annotation_state::valuetype_to_someshell(
                    &crate::model::ValueType::Int,
                )
                .is_some_and(|shell| shell == attr.s_value);
            if matches!(attr.s_value, SomeValue::Impossible)
                || (is_untyped_force_shell && !matches!(s_value, SomeValue::Impossible))
                || is_fn_integer_force_shell
            {
                attr.s_value = s_value;
            }
            // A shaped aggregate's `__pos_N` is written by the constructor
            // (`generalize_attr` of the element annotation). An untyped
            // `Ref` force shell contains every instance, so that write
            // cannot install the element's classdef. Drop the shell back
            // to lattice bottom and let the constructor's value be the
            // item annotation `SomeIterator.next` reads.
            if (majit_ir::descr::is_shaped_tuple_name(n)
                || majit_ir::descr::is_shaped_array_name(n))
                && field_name.starts_with("__pos_")
                && matches!(
                    &attr.s_value,
                    SomeValue::Instance(inst)
                        if inst.classdef.is_none()
                            && !inst.can_be_none
                            && inst.flags.is_empty()
                            && inst.base.const_box.is_none()
                )
            {
                attr.s_value = SomeValue::Impossible;
            }
        }
        Ok(())
    }

    /// TODO: no upstream equivalent.  The closest thing upstream,
    /// `Bookkeeper.getdesc` / `getuniqueclassdef` (`bookkeeper.py`,
    /// `:353-363`), keys identity on the Python object; pyre has no such
    /// object and keys on a qualname.
    /// Resolve a struct type-root name to its canonical host class
    /// `HostObject`, minting it on first sight and caching by name in
    /// [`Self::struct_root_classes`].  Because `HostObject`
    /// equality is `Arc` pointer identity, this interning is what makes
    /// `getuniqueclassdef` return the SAME `ClassDef` for repeated
    /// lookups of one type-root — the pyre analog of resolving a type
    /// name to one class object before `getuniqueclassdef(cls)`.
    ///
    /// Embedded-header subclassing: a struct whose FIRST field embeds
    /// another registered type root BY VALUE (`W_IntObject { ob_header:
    /// PyObject, intval: i64 }`, `LoopBlock { base: FrameBlock }`) is
    /// the Rust spelling of upstream's class hierarchy
    /// (`W_IntObject(W_Root)`, `LoopBlock(FrameBlock)`) — base the
    /// minted class on the header root's class, the same shared-Arc
    /// shape [`Self::intern_class_by_qualname_with_bases`] uses for
    /// enum variants, so `ClassDef::commonbase` unions siblings to the
    /// header root and `cast_pointer`-style refinement connects the
    /// `InstanceRepr`s.  The base is interned under its BARE LEAF —
    /// the spelling `tyref_class_root` seeds params with — so the
    /// subclass points at the same class Arc the rest of the session
    /// uses.
    ///
    /// Each chain node's cache key is resolved through
    /// `canonical_struct_name`, so the bare and qualified spellings of
    /// one struct intern to one `HostObject` — name strings are a
    /// resolution input, never the identity itself
    /// (`getuniqueclassdef(cls)` keys by class object,
    /// bookkeeper.py).  Crate-included `::` names and dotted
    /// constructor qualnames shed exactly one local-crate segment and
    /// converge on the crate-relative key before this walk; a bare leaf
    /// whose origin was tombstoned by
    /// `harden_duplicate_leaf_metadata` stays unresolved and keeps an
    /// attrs-empty classdef because the field registry withdrew the
    /// bare alias.
    ///
    /// The header chain is collected iteratively with a revisit guard:
    /// registry field-type strings strip references
    /// (`tyref_to_ast_string` renders `&T` as `T`), so a reference
    /// cycle (`A.next: &B`, `B.prev: &A`) is indistinguishable from
    /// by-value embedding here — a revisited leaf ends the chain (a
    /// reference loop is linkage, not a header hierarchy), and a
    /// pathological registry chain must not exhaust the native stack
    /// (same O(reachable)-heap discipline as
    /// [`Self::getuniqueclassdef_for_struct_root`]'s worklist).
    pub fn intern_class_by_qualname(self: &Rc<Self>, name: &str) -> HostObject {
        // Collect `name`'s embedded-header chain, outermost first,
        // stopping at an already-interned class (its Arc becomes the
        // deepest base) or at a chain end (scalar / unregistered /
        // revisited first field).  The class-cache key is the canonical
        // `module::Leaf` (`canonical_struct_name`); the field-registry
        // lookup uses the raw spelling (the registry carries both
        // qualified and bare keys).
        let mut chain: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut base_host: Option<HostObject> = None;
        let mut cur = normalize_class_qualname(name);
        loop {
            let key = majit_ir::descr::canonical_struct_name(&cur);
            if let Some(existing) = self.struct_root_classes.borrow().get(&key) {
                if chain.is_empty() {
                    return existing.clone();
                }
                base_host = Some(existing.clone());
                break;
            }
            seen.insert(cur.rsplit("::").next().unwrap_or(&cur).to_string());
            chain.push(cur.clone());
            let next = self.struct_fields.borrow().as_ref().and_then(|reg| {
                // Constructor, crate-included, and crate-relative spellings
                // have already converged before entering the walk.  Keep this
                // reduction local too because subsequent header/enum links
                // can still be emitted as dotted registry names.
                let lookup = cur.replace('.', "::");
                // An enum-variant key `{enum_base}::{variant}` subclasses its
                // enum base — the discriminant-only sum-type root
                // (`rclass.py:499-518`).  Checked before the header convention
                // because a variant's first field is its payload, not a
                // header: without this the session prologue
                // (`CallRegistry::ensure_session`), which pre-mints every
                // `struct_fields` key including the variant keys, would mint
                // the variant base-less, and first-mint-wins would freeze
                // that, dropping the subclass link the narrowing relies on.
                //
                // A monomorphized variant (`StepResult<*mut PyObject>::Continue`)
                // is the same sum type as the template.  The registry keys the
                // template, not the `<…>` suffix, so strip the suffix before
                // the parent test or the variant is minted base-less and a
                // later `__pos_0` write cannot commonbase with the enum.
                let lookup_for_enum = majit_ir::descr::strip_generic_args(&lookup);
                if let Some((parent, _variant)) = lookup_for_enum.rsplit_once("::") {
                    let parent_leaf = parent.rsplit("::").next().unwrap_or(parent);
                    if !seen.contains(parent_leaf) {
                        // The ctor spells the crate (`pyre_interpreter::pyopcode::
                        // StepResult`); the registry often has only the
                        // crate-relative key (`pyopcode::StepResult`) or the
                        // bare leaf.  Parent to that registered key, not the
                        // longer spelling, or the two intern as distinct classes.
                        let mut candidate = parent;
                        loop {
                            let spelled = if candidate.starts_with("pyre_interpreter::")
                                || candidate.starts_with("pyre_object::")
                                || candidate.starts_with("pyre_module::")
                                || candidate.starts_with("pyre_jit::")
                            {
                                crate::front::mir::strip_crate_prefix(candidate)
                            } else {
                                candidate.to_string()
                            };
                            // The field annotation interns the crate-stripped
                            // spelling.  Parent to that string, not the longer
                            // registry key, or the two ClassDefs stay siblings.
                            let in_variant_table = self
                                .enum_variant_by_discriminant
                                .borrow()
                                .as_ref()
                                .is_some_and(|map| {
                                    map.contains_key(candidate)
                                        || map.contains_key(spelled.as_str())
                                        || map.keys().any(|k| {
                                            crate::front::mir::strip_crate_prefix(k) == spelled
                                        })
                                });
                            if in_variant_table
                                || reg.enum_base_registry_key(candidate).is_some()
                                || reg.is_enum_base(candidate)
                                || reg.is_enum_base(&spelled)
                            {
                                return Some(spelled);
                            }
                            match candidate.split_once("::") {
                                Some((_, rest)) => candidate = rest,
                                None => break,
                            }
                        }
                    }
                }
                // A generic enum instantiation (`Option<X>`, `Result<T,E>`) and
                // a collapsed non-suffixed spelling of the same enum
                // (`core.option.Option`, from a ctor whose ADT head lost its
                // type argument) both denote ONE class — RPython has no
                // generics, so every spelling of an enum is the same class.
                // Subclass each under the bare enum leaf (`Option` / `Result`)
                // so two sibling instantiations, and a collapsed value, share a
                // base and `commonbase` to it at a phi merge instead of failing
                // `mergeinputargs` with "no common base class".  The bare leaf
                // is the discriminant-only root (its only row is
                // `__discriminant`), so it carries no payload attr the
                // per-instantiation `__pos_N` writes could conflict on.
                if lookup.contains('<') {
                    let stripped = majit_ir::descr::strip_generic_args(&lookup);
                    // Field annotations use the crate-stripped spelling
                    // (`pyopcode::StepResult` via `strip_crate_prefix`).
                    // Parent the monomorphized class to that spelling so a
                    // `StepResult<*mut PyObject>::Continue` value unions
                    // with the field's class.
                    let spelled = crate::front::mir::strip_crate_prefix(stripped.as_ref());
                    let spelled_leaf = spelled.rsplit("::").next().unwrap_or(&spelled);
                    let is_enum = reg.is_enum_base(stripped.as_ref())
                        || reg.is_enum_base(&spelled)
                        || reg.enum_base_registry_key(stripped.as_ref()).is_some();
                    if is_enum && spelled != lookup && !seen.contains(spelled_leaf) {
                        return Some(spelled);
                    }
                }
                if reg.is_enum_base(&lookup) {
                    let stripped = majit_ir::descr::strip_generic_args(&lookup);
                    let bare_leaf = stripped.rsplit("::").next().unwrap_or(&stripped);
                    if bare_leaf != lookup.as_str() && !seen.contains(bare_leaf) {
                        return Some(bare_leaf.to_string());
                    }
                }
                // `Entry<K,V>` is one struct body per monomorphization. The
                // instantiations subclass the template so a phi can meet on
                // `Entry` while each `key` write stays on its own class.
                let stripped = majit_ir::descr::strip_generic_args(&lookup);
                let template_leaf = stripped.rsplit("::").next().unwrap_or(stripped.as_ref());
                if stripped.as_ref() != lookup.as_str()
                    && template_leaf == "Entry"
                    && (stripped.ends_with("rordereddict_entries::Entry")
                        || stripped.as_ref() == "Entry")
                {
                    let bare = if reg.fields.contains_key(stripped.as_ref()) {
                        Some(stripped.as_ref())
                    } else if reg.fields.contains_key(template_leaf) {
                        Some(template_leaf)
                    } else {
                        None
                    };
                    if let Some(bare) = bare {
                        let bare_leaf = bare.rsplit("::").next().unwrap_or(bare);
                        if !seen.contains(bare_leaf) {
                            return Some(bare.to_string());
                        }
                    }
                }
                // The registry publishes a full declaration path and a bare
                // convenience alias.  `harden_duplicate_leaf_metadata`
                // withdraws the latter when it is not injective, so falling
                // back to it is the name-carrier equivalent of resolving one
                // live class object and fails closed on duplicate leaves.
                let rows = reg.fields.get(&lookup).or_else(|| {
                    let leaf = lookup.rsplit("::").next()?;
                    reg.fields.get(leaf)
                })?;
                let (first_name, first_ty) = rows.first()?;
                // Only header-conventional first fields mark subclassing
                // (`ob_header: PyObject` / `base: FrameBlock`).  A
                // by-value first field of another registered type is
                // otherwise plain composition (`PyError { kind:
                // PyErrorKind, … }`), not a class hierarchy.
                if first_name != "ob_header" && first_name != "base" {
                    return None;
                }
                let leaf = first_ty.rsplit("::").next().unwrap_or(first_ty);
                (reg.fields.contains_key(leaf) && !seen.contains(leaf)).then(|| leaf.to_string())
            });
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        // Mint base-most first so each class links its base's Arc.
        for node in chain.iter().rev() {
            let key = majit_ir::descr::canonical_struct_name(node);
            let bases = match &base_host {
                Some(base) => vec![base.clone()],
                None => self.default_class_bases(&key),
            };
            let host = crate::flowspace::model::HostObject::new_class(key.clone(), bases);
            crate::virtualizable_decl::stamp_host_virtualizable(&host, &key);
            self.struct_root_classes
                .borrow_mut()
                .insert(key, host.clone());
            base_host = Some(host);
        }
        base_host.expect("chain holds at least `name` itself")
    }

    /// TODO: no upstream equivalent.  The closest thing upstream,
    /// `Bookkeeper.getdesc` / `getuniqueclassdef` (`bookkeeper.py`,
    /// `:353-363`), keys identity on the Python object; pyre has no such
    /// object and keys on a qualname.
    /// [`Self::intern_class_by_qualname`] with explicit `__bases__` for
    /// the first mint.  First-mint wins: a cache hit returns the
    /// existing class regardless of `bases`, so every site minting a
    /// given qualname must derive the same bases (the variant-ctor arm
    /// in `flowspace_adapter::translate_op` derives them from the
    /// ctor's own `owner_path`, which is identical at every site).
    pub fn intern_class_by_qualname_with_bases(
        self: &Rc<Self>,
        name: &str,
        bases: Vec<HostObject>,
    ) -> HostObject {
        // Same normalization as [`Self::intern_class_by_qualname`]: both
        // entry points answer one identity question, so they must derive one
        // key.  Keying only the base through the strip splits every enum
        // variant in two — `pyre_object::m::E` and `m::E` intern to one base
        // but to `pyre_object::m::E::Some` and `m::E::Some` — which is the
        // sibling-classdef split `intern_enum_variant_host`'s contract
        // forbids.
        let key = majit_ir::descr::canonical_struct_name(&normalize_class_qualname(name));
        if let Some(existing) = self.struct_root_classes.borrow().get(&key) {
            return existing.clone();
        }
        let bases = if bases.is_empty() {
            self.default_class_bases(&key)
        } else {
            bases
        };
        let host = crate::flowspace::model::HostObject::new_class(key.clone(), bases);
        crate::virtualizable_decl::stamp_host_virtualizable(&host, &key);
        self.struct_root_classes
            .borrow_mut()
            .insert(key, host.clone());
        host
    }

    /// Prebuilt singleton instance of the interned class for `qualname`.
    ///
    /// RPython `InstanceRepr.get_reusable_prebuilt_instance` caches one
    /// instance per classdef on the rtyper; here the instance lives on
    /// the interned class object (`HostObject::reusable_prebuilt_instance`)
    /// in [`Self::struct_root_classes`]. A synthetic unit-variant path
    /// interns through [`Self::intern_enum_variant_host`] so the instance's
    /// `__class__` is the variant ClassDef (Continue → StepResult<..> →
    /// StepResult). `getuniqueclassdef(value.__class__)` then returns that
    /// ClassDef (`classdesc.py getuniqueclassdef`, `rclass.py
    /// InstanceRepr.convert_const`). The declared discriminant is a
    /// class-dict constant on that host (`ClassDesc.classdict`).
    pub fn intern_unit_variant_prebuilt_instance(
        self: &Rc<Self>,
        qualname: &str,
        tag: Option<i64>,
    ) -> Option<HostObject> {
        crate::translator::rtyper::unit_variant_fold::intern_unit_variant_prebuilt_instance(
            self, qualname, tag,
        )
    }

    /// True when `name` is a type-root key in the snapshot
    /// struct-field registry ([`Self::struct_fields`]).  Enums and
    /// structs both register under their qualified path AND bare leaf
    /// (`front/mir.rs` `TypeDeclKind::Enum` / `Struct` arms), so a
    /// ctor's `owner_path` last segment answers true exactly when the
    /// owner is itself an ADT (an enum-variant ctor) rather than a
    /// module (a struct ctor).
    pub fn is_known_struct_root(&self, name: &str) -> bool {
        self.struct_fields
            .borrow()
            .as_ref()
            .is_some_and(|reg| reg.fields.contains_key(name))
    }

    /// TODO: no upstream equivalent.  The closest thing upstream,
    /// `Bookkeeper.getdesc` / `getuniqueclassdef` (`bookkeeper.py`,
    /// `:353-363`), keys identity on the Python object; pyre has no such
    /// object and keys on a qualname.
    /// True when type-root `root` OWNS a field named `name`, or when any
    /// variant of it does — the enum sum-type layout puts a payload row on
    /// the variant, not on the discriminant-only base, so an owner-only test
    /// would miss it (`owner_or_variant_has_field`, `front/semantic.rs`).
    /// The cutover's class-dict method population consults this so a method
    /// member never shadows a same-named instance field — a function source
    /// for a field attribute would union-conflict in `generalize_attr`.
    pub fn struct_root_has_field(&self, root: &str, name: &str) -> bool {
        self.struct_fields
            .borrow()
            .as_ref()
            .is_some_and(|reg| reg.owner_or_variant_has_field(root, name))
    }

    /// A Rust tuple of shape `Tuple<A,B>` is the RPython tuple `(A,B)`:
    /// the `SomeTuple` of its projected items.
    pub fn project_shaped_tuple(self: &Rc<Self>, shape: &str) -> SomeValue {
        let items = shape
            .strip_prefix("Tuple<")
            .and_then(|rest| rest.strip_suffix('>'))
            .unwrap_or_default();
        self.project_struct_field_type(&format!("({items})"))
    }

    /// The `SomeRustVec` a `Vec<T>` spelling projects to — by value, behind a
    /// borrow or behind a raw pointer, all of which carry the header address
    /// — when its item is one word
    /// (`majit_ir::rvec::rust_vec_item_kind_for_spelling`).  Every other
    /// spelling answers `None`.
    pub fn project_rust_vec(&self, ty: &str) -> Option<SomeValue> {
        let kind =
            majit_ir::rvec::rust_vec_item_kind_for_spelling(ty, crate::layout::target_word_size())?;
        Some(SomeValue::RustVec(super::model::SomeRustVec::for_kind(
            kind,
        )))
    }

    /// TODO: no upstream equivalent.  Project a Rust type
    /// string (`"Vec<i32>"`, `"Option<PyFrame>"`, `"HashMap<String,
    /// Box<W_Obj>>"`, …) into a `SomeValue` matching what RPython
    /// `s_getattr` would observe for a class attribute seeded with a
    /// value of that type.  Used by
    /// [`Self::getuniqueclassdef_for_struct_root`] to project the
    /// registered `ClassDef.attrs`.  Bare named types
    /// (`PyFrame`, `W_DictObject`) resolve to
    /// `SomeInstance(stub)` only when the registry contains a matching
    /// entry; unknown bare names fall to `Impossible`.
    pub fn project_struct_field_type(self: &Rc<Self>, field_ty: &str) -> SomeValue {
        let t = field_ty.trim();
        // A `Vec` of one-word items is the raw `{ptr, len, cap}` header
        // `RustVecRepr` lowers, not an RPython list.
        if let Some(s_vec) = self.project_rust_vec(t) {
            return s_vec;
        }
        // `pyre_object::PyObjectRef` is a Rust type alias for
        // `*mut pyobject::PyObject`.  Charon expands aliases in MIR type
        // nodes, but the source-derived struct-field registry deliberately
        // keeps the source spelling, so flexible-array tails such as
        // `FixedObjectArray._items: [PyObjectRef; 0]` reach this projector
        // with the alias still intact.  RPython has no alias object at this
        // point: the corresponding `W_Root` array item is the root instance
        // annotation itself.  Resolve the alias before the ordinary named
        // struct arm so list item repr construction sees that same root
        // `ClassDef`, rather than minting a classdef-less placeholder.
        if t == "PyObjectRef" || t.ends_with("::PyObjectRef") {
            return self.project_struct_field_type("pyobject::PyObject");
        }
        // A function-pointer field is `Ptr(FuncType)` (`lltype.py FuncType`,
        // `SomePtr`), not a classdef-less `SomeInstance`.  Raw `fn` has
        // `_gckind == 'raw'`, so `getkind` is Signed — the dual-gate twin
        // of the legacy walker's address word (`gateway::builtin_code_get`).
        if let Some(s_fn) = fn_ptr_somevalue_for_spelling_in(t, Some(self)) {
            // `lltype.FuncType` / `SomePtr` (`lltype.py` `SomePtr`). A raw
            // function pointer is not a GC instance. `Option<fn(...)>`'s
            // null (`fn_null_constant`) uses the same signature `FuncType`
            // so the two arms union (`llannotation.py`
            // `pairtype(SomePtr, SomePtr).union`).
            return s_fn;
        }
        // A raw-pointer field (`*const T` / `*mut T`) holds a one-word
        // pointer, not a `T`.  Projecting the pointee is right for an
        // aggregate target (`*mut Vec<T>` / `*mut Struct`): the structural
        // shell is already GcRef-kind and downstream element/field access
        // relies on it.  But for a scalar / unit pointee the pointee shell
        // (Signed / Void / …) discards the pointer-ness and diverges from
        // the legacy walker, which folds every raw-pointer field-read to
        // `Ref` → GcRef (the FieldRead op `ty` is
        // `tyref_to_value_type(raw ptr) = Ref(None)`).  Mirror that fold —
        // and the FORCE_ATTRIBUTES `tyref_to_attr_value_type` raw-ptr →
        // `Ref(None)` projection — by shelling a scalar/unit-pointee raw
        // pointer as the conservative classdef-less Ref instance
        // (`getkind = 'ref'`), the GC-safe choice for a field that may
        // hold a managed pointer.
        if let Some(pointee) = t
            .strip_prefix("*const ")
            .or_else(|| t.strip_prefix("*mut "))
        {
            let s_pointee = self.project_struct_field_type(pointee.trim());
            if matches!(
                s_pointee,
                SomeValue::Integer(_)
                    | SomeValue::Bool(_)
                    | SomeValue::Float(_)
                    | SomeValue::SingleFloat(_)
                    | SomeValue::Char(_)
                    | SomeValue::None_(_)
            ) {
                crate::codewriter::annotation_state::record_classdef_less_mint(
                    "project_struct_field_type.raw_ptr_scalar",
                    t,
                );
                return SomeValue::Instance(super::model::SomeInstance::new(
                    None,
                    false,
                    std::collections::BTreeMap::new(),
                ));
            }
            return s_pointee;
        }
        let stripped = t
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim_start_matches("*const ")
            .trim_start_matches("*mut ")
            .trim();
        // `rstring.py` `StringBuilder` (`pyre_object::rstring`): the
        // `rutf8.py` `Utf8StringBuilder._s` attribute is `SomeStringBuilder`.
        if stripped == "StringBuilder" || stripped.ends_with("rstring::StringBuilder") {
            return SomeValue::StringBuilder(super::model::SomeStringBuilder::new());
        }
        match stripped {
            "i8" | "i16" | "i32" | "i64" | "isize" => {
                return super::model::s_int();
            }
            "u8" | "u16" | "u32" | "u64" | "usize" => {
                return super::model::s_uint();
            }
            "f32" => return SomeValue::SingleFloat(super::model::SomeSingleFloat::new()),
            "f64" => return SomeValue::Float(super::model::SomeFloat::new()),
            "bool" => return super::model::s_bool(),
            // Rust `String`/`str` fields are byte string values: string
            // literals lower through `__str_const` into
            // `ConstValue::ByteStr` constants (stamped `Ptr(STR)`, the
            // rtyper's byte `StringRepr`), so the field projection must
            // meet the same `SomeString` — projecting `s_unicode0` here
            // raised `str ∪ unicode` where a literal merged into a
            // String-typed attr cell.
            // RustPython's compiler constants store strings in the
            // dependency-owned `Wtf8Buf` newtype.  The MIR front already
            // erases `Wtf8`/`Wtf8Buf` to the same immutable byte-string
            // value as `String`/`str` (their methods are view adapters over
            // that value), so a struct payload row must project to that
            // exact annotation too.  Leaving the dependency-opaque name to
            // the registered-struct arm below produces `Impossible` and
            // blocks `ConstantData::Str.value`.
            // `BytesBlock` is pyre's concrete storage port of RPython `STR`:
            // a `hash` header (`STR.become(GcStruct('rpy_string', ('hash', Signed), ...)`, `rstr.py`) followed by the `chars` array, whose length is the array's own length word.
            // `bytes_block_chars` folds the Rust raw-slice view back to this
            // owner, so the owner must carry the same `SomeString` annotation
            // as the slice it represents, not a nominal Rust-struct class.
            "String" | "str" | "Wtf8" | "Wtf8Buf" | "BytesBlock" | "Utf8Str" => {
                return super::model::s_str0();
            }
            // `malachite_bigint::BigInt` is deliberately foreign and opaque
            // to translation.  Its sanctioned conversion seam is a residual
            // call, but the value occupying `ConstantData::Integer.value`
            // still has RPython's one-GC-reference shape.  Match the
            // `BigInt.from` builtin analyzer and foreign-method cutover:
            // retain a classdef-less `SomeInstance`, never lattice bottom.
            "BigInt" => {
                crate::codewriter::annotation_state::record_classdef_less_mint(
                    "project_struct_field_type.BigInt",
                    "BigInt",
                );
                return SomeValue::Instance(super::model::SomeInstance::new(
                    None,
                    false,
                    std::collections::BTreeMap::new(),
                ));
            }
            "char" => return SomeValue::Char(super::model::SomeChar::new(false)),
            "()" => return super::model::s_none(),
            _ => {}
        }
        for list_wrapper in ["Vec<", "VecDeque<", "Slice<"] {
            if let Some(inner) = strip_generic_one(stripped, list_wrapper) {
                let s_inner = self.project_struct_field_type(inner);
                // A projected field list is a fresh `ListDef` stored on
                // `ClassDef.attrs[name].s_value` (`Attribute.s_value`).
                // `Bookkeeper.getlistdef` keys by allocation site, not
                // item type; two fields with the same item spelling stay
                // distinct until `ListDef.agree` / `union` join values
                // that actually flow together.
                let listdef =
                    super::listdef::ListDef::new(Some(self.clone()), s_inner, false, false);
                return SomeValue::List(super::model::SomeList::new(listdef));
            }
        }
        if let Some(rest) = stripped.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let inner = match rest.find(';') {
                Some(semi) => rest[..semi].trim(),
                None => rest.trim(),
            };
            let s_inner = self.project_struct_field_type(inner);
            let listdef = super::listdef::ListDef::new(Some(self.clone()), s_inner, false, false);
            return SomeValue::List(super::model::SomeList::new(listdef));
        }
        if let Some(inner) = strip_generic_one(stripped, "Option<") {
            // `Option<fn>` is the nullable `Ptr(FuncType)`, not
            // `union(SomePtr, s_None)`. `s_None` is a classdef-less
            // instance, and that union collapses to a GC ref while the
            // value itself is an int-bank address.
            if let Some(s_fn) = fn_ptr_somevalue_for_spelling_in(inner.trim(), Some(self)) {
                return s_fn;
            }
            let s_inner = self.project_struct_field_type(inner);
            // The same join is refused for a raw-struct payload
            // (`llannotation.py` `pairtype(SomePtr, SomeObject).union`).
            // None is the null word of that Ptr.
            if matches!(s_inner, SomeValue::Ptr(_)) {
                return s_inner;
            }
            let s_none = super::model::s_none();
            return super::model::unionof([&s_inner, &s_none]).unwrap_or(SomeValue::Impossible);
        }
        if let Some(inner) = strip_generic_one(stripped, "Result<") {
            let parts = split_generic_args(inner);
            if parts.len() == 2 {
                let s_ok = self.project_struct_field_type(parts[0]);
                let s_err = self.project_struct_field_type(parts[1]);
                return super::model::unionof([&s_ok, &s_err]).unwrap_or(SomeValue::Impossible);
            }
        }
        for dict_wrapper in ["HashMap<", "BTreeMap<", "IndexMap<"] {
            if let Some(inner) = strip_generic_one(stripped, dict_wrapper) {
                let parts = split_generic_args(inner);
                if parts.len() == 2 {
                    let s_key = self.project_struct_field_type(parts[0]);
                    let s_val = self.project_struct_field_type(parts[1]);
                    let dictdef = super::dictdef::DictDef::new(
                        Some(self.clone()),
                        s_key,
                        s_val,
                        false,
                        false,
                        false,
                    );
                    return SomeValue::Dict(super::model::SomeDict::new(dictdef));
                }
            }
        }
        for set_wrapper in ["HashSet<", "BTreeSet<", "IndexSet<"] {
            if let Some(inner) = strip_generic_one(stripped, set_wrapper) {
                let s_key = self.project_struct_field_type(inner);
                let s_val = super::model::s_none();
                let dictdef = super::dictdef::DictDef::new(
                    Some(self.clone()),
                    s_key,
                    s_val,
                    false,
                    false,
                    false,
                );
                return SomeValue::Dict(super::model::SomeDict::new(dictdef));
            }
        }
        if let Some(inner) = stripped.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            let parts = split_generic_args(inner);
            if !parts.is_empty() {
                let items: Vec<SomeValue> = parts
                    .iter()
                    .map(|p| self.project_struct_field_type(p))
                    .collect();
                return SomeValue::Tuple(super::model::SomeTuple::new(items));
            }
        }
        for wrapper in [
            "Rc<",
            "Arc<",
            "Box<",
            "RefCell<",
            "Cell<",
            "Pin<",
            "NonNull<",
            "NonZero<",
            "MaybeUninit<",
            "ManuallyDrop<",
            "UnsafeCell<",
            "Wrapping<",
            "Reverse<",
        ] {
            if let Some(inner) = strip_generic_one(stripped, wrapper) {
                return self.project_struct_field_type(inner);
            }
        }
        if let Some(inner) = strip_generic_one(stripped, "Cow<") {
            let parts = split_generic_args(inner);
            if parts.len() == 2 {
                return self.project_struct_field_type(parts[1]);
            }
        }
        // `FixedObjectArray { len: usize, _items: [PyObjectRef; 0] }` is a
        // flexible object array.  An `arr[idx]` access lowers (front-end
        // ArrayRead -> flowspace `getitem`) onto the receiver, so the
        // receiver must model as the element list — not the wrapping
        // struct, whose `getitem` over `SomeInstance(None)` would rewrite
        // to `getattr("__getitem__")` and dead-end.  The list identity is
        // `ClassDef.attrs["_items"].s_value` (`Attribute.s_value`); two
        // projections of this field share that attribute rather than
        // minting a second `ListDef`.
        let canonical_stripped = majit_ir::descr::canonical_struct_name(stripped);
        if canonical_stripped == "FixedObjectArray"
            || canonical_stripped == "object_array::FixedObjectArray"
            || canonical_stripped.ends_with("::object_array::FixedObjectArray")
        {
            let items_ty = self
                .struct_fields
                .borrow()
                .as_ref()
                .and_then(|reg| reg.field_type(stripped, "_items").map(str::to_string));
            if let Some(items_ty) = items_ty {
                if let Ok(classdef) = self.getuniqueclassdef_for_struct_root(stripped)
                    && let Some(s_items) =
                        super::classdesc::ClassDef::about_attribute(&classdef, "_items")
                    && matches!(s_items, SomeValue::List(_))
                {
                    return s_items;
                }
                return self.project_struct_field_type(&items_ty);
            }
        }
        // Named generic structs in Charon's registry are stored under their
        // declaration template (`CodeObject`), while a concrete field row can
        // spell the use-site instantiation (`CodeObject<ConstantData>`).
        // RPython has no generic class instantiations: both spellings denote
        // the same classdef.  This is distinct from Option/Result and
        // container wrappers handled above, and from per-shape tuple/enum
        // classes whose dedicated paths never reach this arm.
        let named_root = if strip_generic_one(stripped, "Complex<").is_some()
            && self
                .struct_fields
                .borrow()
                .as_ref()
                .is_some_and(|r| r.fields.contains_key("num_complex::Complex"))
        {
            // Charon renders compiler-core's `Complex64` alias payload as
            // the unqualified `Complex<f64>`, while the opaque dependency
            // declaration is registered under `num_complex::Complex`.
            // Resolve that exact generic spelling to its declaration
            // identity; an unrelated local `Complex` is untouched unless
            // the real dependency root is present in this program.
            std::borrow::Cow::Borrowed("num_complex::Complex")
        } else {
            majit_ir::descr::strip_generic_args(stripped)
        };
        let registered = {
            let guard = self.struct_fields.borrow();
            guard
                .as_ref()
                .map(|r| r.fields.contains_key(named_root.as_ref()))
                .unwrap_or(false)
        };
        if !registered {
            return SomeValue::Impossible;
        }
        // A by-value raw struct is Ptr of a Raw Struct:
        // `history.getkind` Ptr(raw) → "int";
        // `jtransform.rewrite_op_getfield` `_gckind=='raw'` → getfield_raw;
        // `rewrite_op_getsubstruct` → int_add(ptr, ofs).
        // SomeInstance here made dual-gate report legacy=Signed, real=GcRef
        // on a nested FieldRead (MappedObj payload, sockaddr_in.sin_addr,
        // RootScope, VersionTag).
        if let Some(s_ptr) = self.raw_struct_ptr_annotation(named_root.as_ref()) {
            return s_ptr;
        }
        {
            let guard = self.struct_fields.borrow();
            if let Some(reg) = guard.as_ref() {
                let lookup = named_root.as_ref();
                if reg.is_enum_base(lookup) && !reg.enum_base_has_payload(lookup) {
                    // Fieldless enum value is the integer tag
                    // (`rclass.py` getinstancerepr of a discriminant-only base).
                    return super::model::s_int();
                }
            }
        }
        // Resolve the struct-typed field to its identity-registered
        // `ClassDef` through `descs` (`intern_class_by_qualname` ->
        // `getuniqueclassdef`), the same identity path
        // `getuniqueclassdef_for_struct_root` registers it on.  Pass-1 of
        // that method has already registered every reachable struct, so this
        // returns the same `ClassDef` Rc whose `attrs` pass-2 fills in place.
        //
        // Pass-1's string parser may MISS a name this arm resolves
        // (the two walk type strings independently); queue such a
        // first-sight struct so the enclosing struct_root call's
        // pending drain projects its rows before any subject flow can
        // read the class's untyped FORCE shells.
        {
            let canonical = majit_ir::descr::canonical_struct_name(named_root.as_ref());
            if !self.projected_struct_rows.borrow().contains(&canonical) {
                self.pending_struct_row_projection
                    .borrow_mut()
                    .push(named_root.to_string());
            }
        }
        let host = self.intern_class_by_qualname(named_root.as_ref());
        match self.getuniqueclassdef(&host) {
            Ok(classdef) => SomeValue::Instance(super::model::SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
            Err(_) => SomeValue::Impossible,
        }
    }

    /// RPython `Bookkeeper.getuniqueclassdef(cls)` (bookkeeper.py):
    ///
    /// ```python
    /// def getuniqueclassdef(self, cls):
    ///     assert not isinstance(cls, type(Exception)) or cls is type(Exception)
    ///     desc = self.getdesc(cls)
    ///     return desc.getuniqueclassdef()
    /// ```
    ///
    /// The `type(Exception)` assertion guards against PyPy's old-style
    /// exception metaclass trap; the Rust port doesn't model old-style
    /// classes so the assertion is omitted.
    pub fn getuniqueclassdef(
        self: &Rc<Self>,
        cls: &HostObject,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        let entry = self.getdesc(cls)?;
        match entry {
            DescEntry::Class(cd_rc) => ClassDesc::getuniqueclassdef(&cd_rc),
            _ => Err(AnnotatorError::new(format!(
                "Bookkeeper.getuniqueclassdef({:?}): not a class",
                cls.qualname()
            ))),
        }
    }

    /// RPython `Bookkeeper.new_exception(self, exc_classes)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def new_exception(self, exc_classes):
    ///     clsdefs = {self.getuniqueclassdef(cls) for cls in exc_classes}
    ///     return SomeException(clsdefs)
    /// ```
    pub fn new_exception(
        self: &Rc<Self>,
        exc_classes: &[HostObject],
    ) -> Result<super::model::SomeException, super::model::AnnotatorError> {
        let mut clsdefs: Vec<Rc<RefCell<ClassDef>>> = Vec::new();
        for cls in exc_classes {
            let cd = self.getuniqueclassdef(cls)?;
            clsdefs.push(cd);
        }
        Ok(super::model::SomeException::new(clsdefs))
    }

    /// RPython `Bookkeeper.pbc_call(self, pbc, args, emulated=None)`
    /// (bookkeeper.py).
    ///
    /// ```python
    /// def pbc_call(self, pbc, args, emulated=None):
    ///     if emulated is None:
    ///         whence = self.position_key
    ///         fn, block, i = self.position_key
    ///         op = block.operations[i]
    ///         s_previous_result = self.annotator.annotation(op.result)
    ///         if s_previous_result is None:
    ///             s_previous_result = s_ImpossibleValue
    ///     else:
    ///         if emulated is True:
    ///             whence = None
    ///         else:
    ///             whence = emulated
    ///         op = None
    ///         s_previous_result = s_ImpossibleValue
    ///     results = []
    ///     for desc in pbc.descriptions:
    ///         results.append(desc.pycall(whence, args, s_previous_result, op))
    ///     s_result = unionof(*results)
    ///     return s_result
    /// ```
    ///
    /// RPython `Bookkeeper.pbc_getattr(pbc, s_attr)` (bookkeeper.py).
    ///
    /// ```python
    /// def pbc_getattr(self, pbc, s_attr):
    ///     assert s_attr.is_constant()
    ///     attr = s_attr.const
    ///     descs = list(pbc.descriptions)
    ///     first = descs[0]
    ///     if len(descs) == 1:
    ///         return first.s_read_attribute(attr)
    ///     change = first.mergeattrfamilies(descs[1:], attr)
    ///     attrfamily = first.getattrfamily(attr)
    ///     position = self.position_key
    ///     attrfamily.read_locations[position] = True
    ///     actuals = []
    ///     for desc in descs:
    ///         actuals.append(desc.s_read_attribute(attr))
    ///     s_result = unionof(*actuals)
    ///     s_oldvalue = attrfamily.get_s_value(attr)
    ///     attrfamily.set_s_value(attr, unionof(s_result, s_oldvalue))
    ///     if change:
    ///         for position in attrfamily.read_locations:
    ///             self.annotator.reflowfromposition(position)
    ///     if isinstance(s_result, SomeImpossibleValue):
    ///         for desc in descs:
    ///             try:
    ///                 attrs = desc.read_attribute('_attrs_')
    ///             except AttributeError:
    ///                 continue
    ///             if isinstance(attrs, Constant):
    ///                 attrs = attrs.value
    ///             if attr in attrs:
    ///                 raise HarmlesslyBlocked("getattr on enforced attr")
    ///     return s_result
    /// ```
    pub fn pbc_getattr(
        self: &Rc<Self>,
        pbc: &SomePBC,
        s_attr: &SomeValue,
    ) -> Result<SomeValue, super::model::AnnotatorException> {
        use super::classdesc::{ClassDesc, ClassDictEntry};
        use super::description::FrozenDesc;
        use super::model::{AnnotatorException, HarmlesslyBlocked, unionof};

        // upstream: `assert s_attr.is_constant(); attr = s_attr.const`.
        let Some(attr_const) = s_attr.const_() else {
            return Err(AnnotatorError::new(format!(
                "pbc_getattr: s_attr must be constant, got {s_attr:?}"
            ))
            .into());
        };
        // bookkeeper.py `pbc_getattr(self, pbc, s_attr)` —
        // upstream `attr = s_attr.const` with no `isinstance(...,
        // str)` gate. Use [`as_text`] so both `ByteStr` and `UniStr`
        // pass through, mirroring the upstream constant-only check
        // (string equality and the s_read_attribute lookup work
        // across Python 2's bytes/unicode boundary).
        let Some(attr_name) = attr_const.as_text().map(str::to_owned) else {
            return Err(AnnotatorError::new(format!(
                "pbc_getattr: attr must be a string, got {attr_const:?}"
            ))
            .into());
        };

        // upstream: `descs = list(pbc.descriptions); first = descs[0]`.
        let descs: Vec<DescEntry> = pbc.descriptions.values().cloned().collect();
        let first = descs[0].clone();

        // upstream: `if len(descs) == 1: return first.s_read_attribute(attr)`.
        if descs.len() == 1 {
            return Ok(first.s_read_attribute(&attr_name)?);
        }

        // upstream multi-desc path — mergeattrfamilies + getattrfamily +
        // attrfamily bookkeeping. Dispatch by Desc kind, since ClassDesc
        // and FrozenDesc each own their own UnionFind (bookkeeper.py:62-63).
        let (change, pbc_family) = match &first {
            DescEntry::Class(first_cd) => {
                let others_cd: Vec<Rc<RefCell<ClassDesc>>> =
                    descs[1..].iter().filter_map(|d| d.as_class()).collect();
                let change = ClassDesc::mergeattrfamilies(first_cd, &others_cd, &attr_name);
                let family = ClassDesc::getattrfamily(first_cd, &attr_name).ok_or_else(|| {
                    AnnotatorError::new(
                        "pbc_getattr: ClassDesc.getattrfamily unavailable (bookkeeper dropped)",
                    )
                })?;
                (change, PbcAttrFamily::Class(family))
            }
            DescEntry::Frozen(first_fd) => {
                let others_refs: Vec<Rc<RefCell<FrozenDesc>>> =
                    descs[1..].iter().filter_map(|d| d.as_frozen()).collect();
                let borrows: Vec<_> = others_refs.iter().map(|rc| rc.borrow()).collect();
                let others_ptrs: Vec<&FrozenDesc> = borrows.iter().map(|b| &**b).collect();
                let change = first_fd.borrow().mergeattrfamilies(&others_ptrs)?;
                let family = first_fd.borrow().getattrfamily()?;
                (change, PbcAttrFamily::Frozen(family))
            }
            _ => {
                return Err(AnnotatorError::new(format!(
                    "pbc_getattr: multi-desc PBC of kind {:?} has no attrfamily support",
                    first.kind()
                ))
                .into());
            }
        };

        // upstream: `position = self.position_key;
        //            attrfamily.read_locations[position] = True`.
        if let Some(pos) = self.position_key.borrow().clone() {
            pbc_family.add_read_location(pos);
        }

        // upstream: `actuals = [desc.s_read_attribute(attr) for desc in descs]
        //            s_result = unionof(*actuals)`.
        let mut actuals: Vec<SomeValue> = Vec::with_capacity(descs.len());
        for d in &descs {
            actuals.push(d.s_read_attribute(&attr_name)?);
        }
        let s_result = unionof(actuals.iter())
            .map_err(|e| AnnotatorError::new(format!("pbc_getattr unionof failed: {e:?}")))?;

        // upstream: `s_oldvalue = attrfamily.get_s_value(attr);
        //            attrfamily.set_s_value(attr, unionof(s_result, s_oldvalue))`.
        let s_old = pbc_family.get_s_value(&attr_name);
        let merged = unionof([&s_result, &s_old])
            .map_err(|e| AnnotatorError::new(format!("pbc_getattr merge unionof failed: {e:?}")))?;
        pbc_family.set_s_value(&attr_name, merged);

        // upstream: `if change: for position in read_locations: reflowfromposition`.
        if change {
            let positions = pbc_family.read_locations();
            if let Some(ann) = self.annotator.borrow().upgrade() {
                for pos in positions {
                    REFLOW_FROM_PBC.fetch_add(1, Ordering::Relaxed);
                    ann.reflowfromposition(&pos);
                }
            }
        }

        // upstream: HarmlesslyBlocked when s_result is Impossible and
        // attr is in some desc's enforced `_attrs_` set.
        if matches!(s_result, SomeValue::Impossible) {
            for d in &descs {
                let Some(cdesc) = d.as_class() else {
                    continue;
                };
                let attrs_entry = ClassDesc::read_attribute(&cdesc, "_attrs_");
                let Some(ClassDictEntry::Constant(c)) = attrs_entry else {
                    continue;
                };
                let contains = match &c.value {
                    // upstream `_attrs_` is typically a tuple of strings.
                    ConstValue::Tuple(vs) | ConstValue::List(vs) => {
                        vs.iter().any(|v| v.as_text() == Some(attr_name.as_str()))
                    }
                    _ => false,
                };
                if contains {
                    return Err(AnnotatorException::Harmless(HarmlesslyBlocked));
                }
            }
        }

        Ok(s_result)
    }

    /// Python's three-state `emulated` parameter (`None` / `True` /
    /// `<position_key>`) maps onto [`PbcCallEmulated`] in Rust.
    pub(crate) fn pbc_call(
        self: &Rc<Self>,
        pbc: &SomePBC,
        args: &super::argument::ArgumentsForTranslation,
        emulated: PbcCallEmulated,
    ) -> Result<SomeValue, AnnotatorError> {
        use super::model::s_impossible_value;
        // upstream bookkeeper.py:516-531 — 3-way branch on `emulated`.
        let (whence, op_key, s_previous_result) = match &emulated {
            PbcCallEmulated::None => {
                // upstream: `whence = self.position_key;
                //            op = block.operations[i];
                //            s_previous_result = annotation(op.result)
                //                                 or s_ImpossibleValue`.
                let pk = self.current_position_key();
                let whence: Option<(
                    crate::flowspace::model::GraphRef,
                    crate::flowspace::model::BlockRef,
                    usize,
                )> = pk.as_ref().and_then(|p| match (p.graph(), p.block()) {
                    (Some(g), Some(b)) => Some((g, b, p.op_index)),
                    _ => None,
                });
                let s_prev = if let Some(pk_ref) = pk.as_ref() {
                    if let (Some(block), Some(ann)) =
                        (pk_ref.block(), self.annotator.borrow().upgrade())
                    {
                        let i = pk_ref.op_index;
                        let block_borrow = block.borrow();
                        if i < block_borrow.operations.len() {
                            let result_var = block_borrow.operations[i].result.clone();
                            ann.annotation(&result_var)
                                .unwrap_or_else(s_impossible_value)
                        } else {
                            s_impossible_value()
                        }
                    } else {
                        s_impossible_value()
                    }
                } else {
                    s_impossible_value()
                };
                (whence, pk, s_prev)
            }
            PbcCallEmulated::True => {
                // upstream: `whence = None; op = None;
                //            s_previous_result = s_ImpossibleValue`.
                (None, None, s_impossible_value())
            }
            PbcCallEmulated::Callback(callback) => {
                // upstream: `whence = emulated; op = None;
                //            s_previous_result = s_ImpossibleValue`.
                let whence = match (callback.graph(), callback.block()) {
                    (Some(g), Some(b)) => Some((g, b, callback.op_index)),
                    _ => None,
                };
                (whence, None, s_impossible_value())
            }
        };

        // upstream: `for desc in pbc.descriptions:
        //             results.append(desc.pycall(whence, args, s_previous_result, op))`.
        let mut results: Vec<SomeValue> = Vec::with_capacity(pbc.descriptions.len());
        for entry in pbc.descriptions.values() {
            let r = match entry {
                // upstream `desc.pycall` virtual dispatch: a memo desc
                // runs `MemoDesc.pycall` (union-of-results annotation, or
                // the dispatch graph's return var); a plain function runs
                // `FunctionDesc.pycall`.
                super::description::DescEntry::Func(fe) => {
                    if let Some(md) = fe.as_memo() {
                        md.borrow().pycall(
                            whence.clone(),
                            args,
                            &s_previous_result,
                            op_key.clone(),
                        )?
                    } else {
                        fe.func().borrow().pycall(
                            whence.clone(),
                            args,
                            &s_previous_result,
                            op_key.clone(),
                        )?
                    }
                }
                super::description::DescEntry::Method(md) => {
                    md.borrow()
                        .pycall(whence.clone(), args, &s_previous_result, op_key.clone())?
                }
                super::description::DescEntry::MethodOfFrozen(mfd) => {
                    mfd.borrow()
                        .pycall(whence.clone(), args, &s_previous_result, op_key.clone())?
                }
                super::description::DescEntry::Class(cd) => super::classdesc::ClassDesc::pycall(
                    cd,
                    whence.clone(),
                    args,
                    &s_previous_result,
                    op_key.clone(),
                )?,
                super::description::DescEntry::Frozen(_) => {
                    return Err(AnnotatorError::new("pbc_call: FrozenDesc is not callable"));
                }
            };
            results.push(r);
        }
        // upstream: `s_result = unionof(*results)`.
        super::model::unionof(results.iter())
            .map_err(|e| AnnotatorError::new(format!("pbc_call unionof: {}", e)))
    }

    /// rlib/jit.py:903-914 — fold `kwds_s` into
    /// [`Self::_jit_annotation_cache`] under `driver`, unioning
    /// against any previous annotation seen for the same kwarg key.
    ///
    /// Upstream:
    ///
    /// ```python
    /// for key, s_value in kwds_s.items():
    ///     s_previous = cache.get(key, annmodel.s_ImpossibleValue)
    ///     s_value = annmodel.unionof(s_previous, s_value)
    ///     cache[key] = s_value
    /// ```
    ///
    /// `kwds_s` keys carry the upstream `'s_'` prefix (rlib/jit.py:895
    /// `expected = ['s_' + name for name in ...]`), so callers must
    /// preserve the prefix to keep the cache shape line-by-line with
    /// upstream.
    pub fn union_jit_annotation_kwds(
        self: &Rc<Self>,
        driver: &HostObject,
        kwds_s: &HashMap<String, Option<SomeValue>>,
    ) -> Result<(), AnnotatorError> {
        let mut cache_outer = self._jit_annotation_cache.borrow_mut();
        let cache = cache_outer.entry(driver.clone()).or_default();
        for (key, s_value) in kwds_s {
            // upstream `bookkeeper.py union_jit_annotation_kwds` iterates
            // `kwds_s.items()` and `union`s each `s_value` into the cache.
            // A `None` value (unbound annotation, mirroring
            // `annotator.annotation(v) is None`) contributes nothing to
            // the union — preserve the previous cache entry verbatim
            // instead of widening it through `union(prev, None)`.
            let Some(s_value) = s_value else { continue };
            let s_previous = cache.remove(key).unwrap_or_else(s_impossible_value);
            let s_unioned = union(&s_previous, s_value).map_err(|e| {
                AnnotatorError::new(format!(
                    "_jit_annotation_cache union for key {key:?}: {e:?}"
                ))
            })?;
            cache.insert(key.clone(), s_unioned);
        }
        Ok(())
    }

    /// RPython `Bookkeeper.emulate_pbc_call(self, unique_key, pbc,
    /// args_s, replace=[], callback=None)` (bookkeeper.py).
    ///
    /// ```python
    /// def emulate_pbc_call(self, unique_key, pbc, args_s,
    ///                      replace=[], callback=None):
    ///     with self.at_position(None):
    ///         emulated_pbc_calls = self.emulated_pbc_calls
    ///         prev = [unique_key]
    ///         prev.extend(replace)
    ///         for other_key in prev:
    ///             if other_key in emulated_pbc_calls:
    ///                 del emulated_pbc_calls[other_key]
    ///         emulated_pbc_calls[unique_key] = pbc, args_s
    ///
    ///         args = simple_args(args_s)
    ///         if callback is None:
    ///             emulated = True
    ///         else:
    ///             emulated = callback
    ///         return self.pbc_call(pbc, args, emulated=emulated)
    /// ```
    pub(crate) fn emulate_pbc_call(
        self: &Rc<Self>,
        unique_key: EmulatedPbcCallKey,
        pbc: &SomeValue,
        args_s: &[SomeValue],
        replace: &[EmulatedPbcCallKey],
        callback: Option<PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        let SomeValue::PBC(pbc) = pbc else {
            return Err(AnnotatorError::new(format!(
                "Bookkeeper.emulate_pbc_call expects SomePBC, got {pbc:?}"
            )));
        };
        // upstream: `with self.at_position(None):`
        let _guard = self.at_position(None);
        {
            let mut emulated_map = self.emulated_pbc_calls.borrow_mut();
            // upstream: `prev = [unique_key]; prev.extend(replace);
            //            for other_key in prev:
            //                if other_key in emulated_pbc_calls:
            //                    del emulated_pbc_calls[other_key]`.
            // `shift_remove`, not `remove`/`swap_remove`: preserve the
            // insertion order of the surviving entries so the fixpoint
            // `values()` walk stays deterministic.
            emulated_map.shift_remove(&unique_key);
            for key in replace {
                emulated_map.shift_remove(key);
            }
            // upstream: `emulated_pbc_calls[unique_key] = pbc, args_s`.
            emulated_map.insert(unique_key, (pbc.clone(), args_s.to_vec()));
        }
        // upstream: `args = simple_args(args_s)`.
        let args = simple_args(args_s.to_vec());
        // upstream: `emulated = True if callback is None else callback`.
        let emulated = match callback {
            None => PbcCallEmulated::True,
            Some(cb) => PbcCallEmulated::Callback(cb),
        };
        // upstream: `return self.pbc_call(pbc, args, emulated=emulated)`.
        self.pbc_call(pbc, &args, emulated)
    }

    /// RPython `Bookkeeper.immutablevalue(x)` (bookkeeper.py).
    ///
    /// "The most precise SomeValue instance that contains the
    /// immutable value x."
    ///
    fn immutable_list_with_key(
        self: &Rc<Self>,
        key: Option<&Constant>,
        x: &ConstValue,
        items: &[ConstValue],
    ) -> Result<SomeValue, AnnotatorError> {
        if let Some(key) = key
            && let Some(hit) = self.immutable_cache.borrow().get(&key.id).cloned()
        {
            return Ok(hit);
        }
        let listdef = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
        let mut result = SomeList::new(listdef.clone());
        result.base.const_box = Some(key.cloned().unwrap_or_else(|| Constant::new(x.clone())));
        if let Some(key) = key {
            self.immutable_cache
                .borrow_mut()
                .insert(key.id, SomeValue::List(result.clone()));
        }
        for e in items {
            let s_e = self.immutablevalue(e)?;
            listdef
                .generalize(&s_e)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeValue::List(result))
    }

    fn immutable_dict_with_key(
        self: &Rc<Self>,
        key: Option<&Constant>,
        x: &ConstValue,
        items: &HashMap<ConstValue, ConstValue>,
    ) -> Result<SomeValue, AnnotatorError> {
        if let Some(key) = key
            && let Some(hit) = self.immutable_cache.borrow().get(&key.id).cloned()
        {
            return Ok(hit);
        }
        let dictdef = DictDef::new(
            Some(self.clone()),
            SomeValue::Impossible,
            SomeValue::Impossible,
            false,
            false,
            false,
        );
        let mut result = SomeDict::new(dictdef.clone());
        result.base.const_box = Some(key.cloned().unwrap_or_else(|| Constant::new(x.clone())));
        if let Some(key) = key {
            self.immutable_cache
                .borrow_mut()
                .insert(key.id, SomeValue::Dict(result.clone()));
        }
        for (k, v) in items {
            let s_k = self.immutablevalue(k)?;
            let s_v = self.immutablevalue(v)?;
            dictdef
                .generalize_key(&s_k)
                .map_err(|e| AnnotatorError::new(e.msg))?;
            dictdef
                .generalize_value(&s_v)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeValue::Dict(result))
    }

    /// RPython `Bookkeeper.immutablevalue(Constant(x))` for callers that
    /// still have the original `Constant` object and therefore its
    /// identity. List/dict branches must preserve that identity so
    /// `self.immutable_cache[key]` matches upstream.
    pub fn immutableconstant(self: &Rc<Self>, c: &Constant) -> Result<SomeValue, AnnotatorError> {
        match &c.value {
            ConstValue::List(items) => self.immutable_list_with_key(Some(c), &c.value, items),
            ConstValue::Dict(items) => self.immutable_dict_with_key(Some(c), &c.value, items),
            // `ConstUInt` is stored as `ConstValue::Int` with
            // `concretetype = Unsigned` (`flowspace_adapter`). The value
            // branch only sees the signed carrier, so `u64::MAX` becomes
            // `-1` and a later `r_uint` binding no longer contains it.
            ConstValue::Int(i)
                if matches!(
                    c.concretetype,
                    Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Unsigned)
                ) =>
            {
                let mut s = SomeInteger::new(true, true);
                s.base.const_box = Some(Constant::with_concretetype(
                    ConstValue::Int(*i),
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Unsigned,
                ));
                Ok(SomeValue::Integer(s))
            }
            _ => self.immutablevalue(&c.value),
        }
    }

    /// Input is a flowspace [`ConstValue`] — the Rust-side counterpart
    /// to upstream's Python constant. Primitive branches (bool / int /
    /// float / str / char / unicode / bytearray / tuple / None) are
    /// ported line-by-line. Callers that still have the original
    /// [`Constant`] object should use [`Self::immutableconstant`] so
    /// list/dict branches can preserve `Constant(x)` identity.
    ///
    /// The function / class / bound-method / weakref / frozen-PBC /
    /// property / instance branches (bookkeeper.py:218-348) route
    /// through [`Self::immutablevalue_hostobject`]. The `_ptr`
    /// extregistry branch (bookkeeper.py:312-314 via lltype.py:_ptrEntry)
    /// routes through `translator/rtyper/extregistry.rs`.
    pub fn immutablevalue(self: &Rc<Self>, x: &ConstValue) -> Result<SomeValue, AnnotatorError> {
        match x {
            ConstValue::Bool(b) => {
                let mut s = SomeBool::new();
                s.base.const_box = Some(Constant::new(ConstValue::Bool(*b)));
                Ok(SomeValue::Bool(s))
            }
            ConstValue::Int(i) => {
                // upstream: `result = SomeInteger(nonneg = x>=0)`.
                let mut s = SomeInteger::new(*i >= 0, false);
                s.base.const_box = Some(Constant::new(ConstValue::Int(*i)));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::Int128(i) => {
                let mut s = SomeInteger::new_with_knowntype(
                    *i >= 0,
                    crate::annotator::model::KnownType::LongLongLong,
                );
                s.base.const_box = Some(Constant::new(ConstValue::Int128(*i)));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::UInt128(i) => {
                let mut s = SomeInteger::new_with_knowntype(
                    true,
                    crate::annotator::model::KnownType::ULongLongLong,
                );
                s.base.const_box = Some(Constant::new(ConstValue::UInt128(*i)));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::Float(_) => {
                let mut s = SomeFloat::new();
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Float(s))
            }
            ConstValue::ByteStr(s) => {
                let no_nul = !s.contains(&0);
                let result = if s.len() == 1 {
                    // upstream: `result = SomeChar(no_nul=no_nul)`.
                    let mut ch = SomeChar::new(no_nul);
                    ch.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::Char(ch)
                } else {
                    // upstream: `result = SomeString(no_nul=no_nul)`.
                    let mut st = SomeString::new(false, no_nul);
                    st.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::String(st)
                };
                Ok(result)
            }
            ConstValue::UniStr(s) => {
                let no_nul = !s.contains('\x00');
                let result = if s.chars().count() == 1 {
                    let mut ch = SomeUnicodeCodePoint::new(no_nul);
                    ch.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::UnicodeCodePoint(ch)
                } else {
                    let mut st = SomeUnicodeString::new(false, no_nul);
                    st.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::UnicodeString(st)
                };
                Ok(result)
            }
            ConstValue::None => Ok(s_none()),
            ConstValue::Tuple(items) => {
                let items_s = items
                    .iter()
                    .map(|v| self.immutablevalue(v))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SomeValue::Tuple(SomeTuple::new(items_s)))
            }
            ConstValue::List(items) => self.immutable_list_with_key(None, x, items),
            ConstValue::Dict(items) => self.immutable_dict_with_key(None, x, items),
            ConstValue::HostObject(obj) => self.immutablevalue_hostobject(obj, x),
            ConstValue::Function(func) => {
                let host = HostObject::new_user_function((**func).clone());
                self.immutablevalue_hostobject(&host, &ConstValue::HostObject(host.clone()))
            }
            ConstValue::LLPtr(_) => {
                assert!(crate::translator::rtyper::extregistry::is_registered(x));
                let entry = crate::translator::rtyper::extregistry::lookup(x).expect(
                    "Bookkeeper.immutablevalue(LLPtr): extregistry.lookup must succeed after \
                     is_registered",
                );
                let mut result = entry.compute_annotation_bk(self)?;
                // bookkeeper.py:350 — after the extregistry branch,
                // upstream falls through to the common `result.const = x`.
                // Keep that const_box on the SomePtr itself; otherwise
                // SomePtr.bool/is_immutable_constant diverge from RPython.
                match &mut result {
                    SomeValue::Ptr(ptr) => {
                        ptr.base.const_box = Some(Constant::new(x.clone()));
                    }
                    other => panic!(
                        "Bookkeeper.immutablevalue(LLPtr): _ptrEntry must return SomePtr, got {other:?}"
                    ),
                }
                Ok(result)
            }
            ConstValue::AddressOffset(_) => {
                // AddressOffset.annotation() returns SomeInteger()
                let mut s = SomeInteger::new(false, false);
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::InheritanceId { .. } => {
                // A symbolic inheritance id is a Signed integer marker.
                let mut s = SomeInteger::new(false, false);
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::LLAddress(_) => {
                // `isinstance(x, llmemory.fakeaddress)` arm of
                // `bookkeeper.py immutablevalue` — a prebuilt address
                // constant annotates as `SomeAddress()`; the const_box
                // keeps the concrete address so `is_null_address` and
                // constant folding can read it back.
                let mut s = super::model::SomeAddress::new();
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Address(s))
            }
            ConstValue::Repr(_) => {
                // `Repr._freeze_` returns true, so this is the `_freeze_`
                // arm: `SomePBC([getdesc(x)])`. A Repr is not a HostObject,
                // so the frozen description is built for this bookkeeper
                // without a host key.
                let entry =
                    DescEntry::Frozen(Rc::new(RefCell::new(FrozenDesc::prebuilt(self.clone()))));
                let mut pbc = SomePBC::new(vec![entry], false);
                pbc.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::PBC(pbc))
            }
            ConstValue::Code(_)
            | ConstValue::Graphs(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::SpecTag(_)
            | ConstValue::Atom(_)
            | ConstValue::Placeholder
            | ConstValue::Opaque(_) => {
                // Code / Graphs / SpecTag / Atom / Placeholder cover
                // internal flowspace / host-carrier values that
                // upstream never feeds into immutablevalue. Keep the
                // fail-fast stub so any unexpected call-site surfaces
                // a clear error rather than silent stub-SomePBC.
                Err(AnnotatorError::new(format!(
                    "Bookkeeper.immutablevalue({x:?}): internal ConstValue variant \
                     has no upstream immutablevalue branch"
                )))
            }
        }
    }

    /// Narrow dispatch for the `ConstValue::HostObject` arm of
    /// [`Self::immutablevalue`]. Mirrors upstream
    /// `bookkeeper.py` — builtin-analyser / callable / class /
    /// frozen fallbacks.
    ///
    /// The user-function and class branches now route through
    /// [`Self::getdesc`], which produces the real
    /// [`DescEntry::Function`] / [`DescEntry::Class`] wrapping the
    /// shared `Rc<RefCell<…>>` instance from `self.descs`. The
    /// resulting [`SomePBC`] has `descriptions` populated with actual
    /// Desc objects (fix for reviewer pre-existing #1: SomePBC
    /// descriptions were just `kind+name` stubs).
    fn immutablevalue_hostobject(
        self: &Rc<Self>,
        obj: &HostObject,
        raw: &ConstValue,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream bookkeeper.py — `tp is weakref.ReferenceType`:
        //   x1 = x()
        //   if x1 is None:
        //       result = SomeWeakRef(None)    # dead weakref
        //   else:
        //       s1 = self.immutablevalue(x1)
        //       assert isinstance(s1, SomeInstance)
        //       result = SomeWeakRef(s1.classdef)
        //
        // Upstream checks this FIRST (right after the dict cache
        // path), before property/callable/class — weakref objects
        // are themselves callable, so the order matters on upstream.
        // `HostObjectKind` variants are disjoint in the Rust port,
        // so the branches below are observationally order-
        // independent; the sequence here mirrors upstream anyway.
        if obj.is_weakref() {
            let mut wr = match obj.weakref_referent().flatten() {
                // dead weakref.
                None => super::model::SomeWeakRef::new(None),
                Some(referent) => {
                    let s1 = self.immutablevalue(&ConstValue::HostObject(referent.clone()))?;
                    let SomeValue::Instance(inst) = s1 else {
                        return Err(AnnotatorError::new(format!(
                            "Bookkeeper.immutablevalue(weakref): target {referent:?} does not \
                             annotate as SomeInstance (upstream asserts isinstance(s1, SomeInstance))"
                        )));
                    };
                    super::model::SomeWeakRef::new(inst.classdef.clone())
                }
            };
            wr.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::WeakRef(wr));
        }
        // upstream bookkeeper.py:307-308 — `elif tp is property: return SomeProperty(x)`.
        if obj.is_property() {
            return Ok(SomeValue::Property(super::model::SomeProperty::new(obj)));
        }
        // upstream bookkeeper.py — `elif ishashable(x) and x
        // in BUILTIN_ANALYZERS: result = SomeBuiltin(...)`.
        if ishashable(obj) && super::builtin::is_registered(obj.qualname()) {
            let module_name = match crate::flowspace::model::host_getattr(obj, "__module__") {
                Ok(value) => value.into_text().unwrap_or_else(|| "unknown".to_string()),
                Err(crate::flowspace::model::HostGetAttrError::Missing) => "unknown".to_string(),
                Err(crate::flowspace::model::HostGetAttrError::Unsupported) => {
                    return Err(AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): getattr({:?}, '__module__') unsupported",
                        obj.qualname()
                    )));
                }
            };
            let name = match crate::flowspace::model::host_getattr(obj, "__name__") {
                Ok(value) => value.as_text().map(str::to_owned).ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): {:?}.__name__ is not a string: {value:?}",
                        obj.qualname()
                    ))
                })?,
                Err(err) => {
                    return Err(AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): getattr({:?}, '__name__') failed: {err:?}",
                        obj.qualname()
                    )));
                }
            };
            let mut sb = SomeBuiltin::new(
                obj.qualname().to_string(),
                None,
                Some(format!("{module_name}.{name}")),
            );
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:312-314:
        //
        //     elif extregistry.is_registered(x):
        //         entry = extregistry.lookup(x)
        //         result = entry.compute_annotation_bk(self)
        //
        // The HostObject side of the value-level registry currently has
        // no entries (see [`extregistry::is_registered`] doc) — adding
        // one means extending the match arm there, the call below
        // automatically picks it up. Until then this branch declines
        // and falls through to the is_class / is_user_function / etc.
        // checks just like upstream's `elif extregistry.is_registered`
        // declines when `x` is not registered.
        let host_const = ConstValue::HostObject(obj.clone());
        if crate::translator::rtyper::extregistry::is_registered(&host_const) {
            let entry = crate::translator::rtyper::extregistry::lookup(&host_const).expect(
                "Bookkeeper.immutablevalue_hostobject: extregistry.lookup must succeed after \
                 is_registered",
            );
            let mut result = entry.compute_annotation_bk(self)?;
            // upstream bookkeeper.py:347 — `try: result.const = x;
            // except AttributeError: pass`. SomeValue::set_const_box
            // mirrors the swallow contract.
            result.set_const_box(Constant::new(raw.clone()));
            return Ok(result);
        }
        // upstream bookkeeper.py:315-316 — `elif tp is type: result =
        // SomeConstantType(x, self)`. Implemented as a constant
        // SomePBC over the real [`ClassDesc`] returned by [`Self::getdesc`].
        if obj.is_class() {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py — `elif callable(x):` splits
        // into im_self/__self__ bound-method routing (→ find_method on
        // s_self) and the plain-callable fallback (→
        // SomePBC([getdesc(x)])). The Rust port encodes these as two
        // disjoint `HostObjectKind` variants that appear in the same
        // position as upstream's single `callable(x)` branch.
        if obj.is_bound_method() {
            let self_obj = obj.bound_method_self().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.immutablevalue(bound method): missing __self__")
            })?;
            let s_self = self.immutablevalue(&ConstValue::HostObject(self_obj.clone()))?;
            if let Some(name) = obj.bound_method_name()
                && let Some(result) = s_self.find_method(name)
            {
                return Ok(result);
            }
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        if obj.is_user_function() {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            // Keep explicit const_box for parity with upstream
            // `self.const = x` (bookkeeper.py's SomePBC callers pin
            // const with the raw HostObject); SomePBC::new's single-
            // desc hack already sets this to an equivalent value but
            // we retain the write for clarity when `raw != pyobj`.
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // Fallback `callable(x)` arm for `BuiltinCallable` HostObjects
        // whose qualname is not registered in `BUILTIN_ANALYZERS`
        // (e.g. the bare `std.ptr.eq` / `std.mem.align_of` /
        // `std.alloc.dealloc` host stubs published by
        // `HostEnv::bootstrap`).  Mirrors upstream `bookkeeper.py`
        // `elif callable(x): result = SomePBC([self.getdesc(x)])`:
        // PyPy doesn't distinguish "analyzer-backed builtin" from
        // "bare-stub callable" at the immutablevalue boundary — both
        // route through the same `callable(x)` arm.  Pyre splits the
        // registered-analyzer path (returns `SomeBuiltin` with an
        // analyzer ref) from the unregistered fallback (returns
        // `SomeBuiltin` with `analyzer = None`), so a call-site
        // dispatch on this `SomeBuiltin` still fails loud when the
        // analyser is genuinely missing — but `immutablevalue` itself
        // succeeds, matching upstream's "annotate first, fail at the
        // call site only" lifecycle.
        if obj.is_builtin_callable() {
            let qualname = obj.qualname().to_string();
            let mut sb = SomeBuiltin::new(qualname.clone(), None, Some(qualname));
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:332-338 — `elif hasattr(x, '_freeze_'):
        // assert x._freeze_() is True; result = SomePBC([getdesc(x)])`.
        // Check BEFORE the instance branch so any HostObject carrying
        // a `_freeze_` member routes to the frozen-PBC path regardless
        // of its kind (Instance / Module / Opaque all surface their
        // attribute dict the same way).
        if call_freeze_method(obj)? {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:339-345 — `hasattr(x, '__class__') and
        // x.__class__.__module__ != '__builtin__'` →
        // `getuniqueclassdef(x.__class__) + see_instance + SomeInstance`.
        if let Some(class_obj) = obj.instance_class() {
            // upstream bookkeeper.py:341-342:
            //     if hasattr(x, '_cleanup_'): x._cleanup_()
            call_cleanup_method(obj)?;
            // A unit-variant singleton is folded to a prebuilt instance whose
            // class host is interned through `intern_enum_variant_host`
            // (`unit_variant_fold::intern_unit_variant_prebuilt_instance`)
            // so `value.__class__` is the variant ClassDef.
            // `getuniqueclassdef(x.__class__)` (bookkeeper.py) then returns
            // that ClassDef (`rclass.py` `InstanceRepr.convert_const`).
            let classdef = self.getuniqueclassdef(class_obj)?;
            super::classdesc::ClassDef::see_instance(&classdef, obj)?;
            let mut inst = super::model::SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            );
            inst.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Instance(inst));
        }
        // upstream bookkeeper.py:349 — final `raise AnnotatorError("Don't
        // know how to represent %r" % (x,))`. Module / Opaque / weakref
        // / extregistry HostObject kinds without `_freeze_` land here
        // because upstream also raises for them.
        Err(AnnotatorError::new(format!(
            "Don't know how to represent {raw:?}"
        )))
    }
}

impl Default for Bookkeeper {
    fn default() -> Self {
        Self::new()
    }
}

/// One spelling rule for the class cache key, shared by
/// [`Bookkeeper::intern_class_by_qualname`] and
/// [`Bookkeeper::intern_class_by_qualname_with_bases`].
///
/// A Rust declaration reaches the annotator under three spellings:
/// crate-included `pyre_object::unicodeobject::W_UnicodeObject`,
/// crate-relative `unicodeobject::W_UnicodeObject`, and a dotted constructor
/// qualname.  RPython keys all reads and constructors on the one live class
/// object (`bookkeeper.py` — `obj_key = Constant(pyobj)`), so normalize
/// dots and strip one crate prefix only when StructId resolution proves that
/// both spellings denote the same declaration.
///
/// Strip only while a module path survives.  A Rust `::` spelling is stripped
/// only when the process-wide StructId resolver proves the full and
/// crate-relative names denote the same declaration; class identity must not
/// depend on the thread-local call-alias registry or on which pipeline last
/// ran on this thread. Collapsing
/// `pyre_object::PyObject` to the bare `PyObject` is what
/// `harden_duplicate_leaf_metadata` exists to undo — the chain walk needs
/// `ob_header` / `base` to reach the superclass, and a bare leaf collides
/// across modules.
pub(crate) fn normalize_class_qualname(name: &str) -> String {
    let lookup = name.replace('.', "::");
    lookup
        .split_once("::")
        .filter(|(_root, rest)| {
            rest.contains("::")
                && (name.contains('.')
                    || matches!(
                        (
                            majit_ir::descr::struct_id_for_name(&lookup),
                            majit_ir::descr::struct_id_for_name(rest),
                        ),
                        (Some(full), Some(relative)) if full == relative
                    ))
        })
        .map(|(_, rest)| rest.to_string())
        .unwrap_or(lookup)
}

/// Walk `field_ty`'s type-string structure and collect every bare-named
/// type that the `StructFieldRegistry` recognises.  Used by
/// [`Bookkeeper::getuniqueclassdef_for_struct_root`]'s pass-1 discovery
/// to register a ClassDef for every reachable struct before any
/// attribute is projected.  Recursion bound is type-string nesting
/// (Vec<Vec<…>>) which is shallow, NOT the registry's field-graph
/// depth.
fn collect_referenced_struct_names(
    field_ty: &str,
    reg: &crate::front::StructFieldRegistry,
    out: &mut Vec<String>,
) {
    let stripped = field_ty
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ")
        .trim();
    match stripped {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" | "f32"
        | "f64" | "bool" | "String" | "str" | "char" | "()" => return,
        _ => {}
    }
    for wrapper in [
        "Vec<",
        "VecDeque<",
        "Option<",
        "Rc<",
        "Arc<",
        "Box<",
        "RefCell<",
        "Cell<",
        "Pin<",
        "NonNull<",
        "NonZero<",
        "MaybeUninit<",
        "ManuallyDrop<",
        "UnsafeCell<",
        "Wrapping<",
        "Reverse<",
        "HashSet<",
        "BTreeSet<",
        "IndexSet<",
    ] {
        if let Some(inner) = strip_generic_one(stripped, wrapper) {
            collect_referenced_struct_names(inner, reg, out);
            return;
        }
    }
    for wrapper in ["HashMap<", "BTreeMap<", "IndexMap<", "Result<", "Cow<"] {
        if let Some(inner) = strip_generic_one(stripped, wrapper) {
            for part in split_generic_args(inner) {
                collect_referenced_struct_names(part, reg, out);
            }
            return;
        }
    }
    if let Some(rest) = stripped.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        for part in split_generic_args(rest) {
            collect_referenced_struct_names(part, reg, out);
        }
        return;
    }
    if let Some(rest) = stripped.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let inner = match rest.find(';') {
            Some(semi) => rest[..semi].trim(),
            None => rest.trim(),
        };
        collect_referenced_struct_names(inner, reg, out);
        return;
    }
    if reg.fields.contains_key(stripped) {
        out.push(stripped.to_string());
    }
}

/// Whether a field-type spelling names a sum type that
/// [`Bookkeeper::project_struct_field_type`] models as its *payload* plus
/// `None` — the nullable-pointer shape (`Option<Vec<T>>` → list-or-none).
/// That model describes a field lowered to one nullable word; it does NOT
/// describe a slot the frontend fills with a materialized variant instance
/// (`simple_call(<host Option<Vec<*mut PyObject>>::None>)`), whose
/// annotation is `Instance(Option<…>::None)` and cannot union with the
/// payload.
fn is_nullable_sum_spelling(field_ty: &str) -> bool {
    let t = field_ty
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim();
    strip_generic_one(t, "Option<").is_some() || strip_generic_one(t, "Result<").is_some()
}

/// `SomePtr(Ptr(FuncType([], Void)))` — the annotation of a raw function
/// pointer whose signature was erased to the coarse field-registry
/// spelling (`fn` / `Option<fn>`). Shared with an untyped `fn_null_constant`
/// so a coarse null `Option<fn>` and a coarse `fn` field read are one
/// `ll_ptrtype`.
pub(crate) fn raw_fn_ptr_somevalue() -> SomeValue {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};
    SomeValue::Ptr(super::model::SomePtr::new(Ptr {
        TO: PtrTarget::Func(empty_functype()),
    }))
}

/// `SomePtr(FuncType)` when `spelling` names a function pointer or a
/// nullable `Option<fn>`. The `Int` bank (`history.getkind` of
/// `Ptr(FuncType)`) still shells through `valuetype_to_someshell` as
/// `SomeInteger`; this recovers the pointer lattice those Int-typed
/// producers lost. A full `fn(inputs) -> output` spelling yields the
/// same `FuncType` as `getfunctionptr` / `declared_funcptr_type_from_legacy`
/// for that signature.
#[cfg(test)]
pub(crate) fn fn_ptr_somevalue_for_spelling(spelling: &str) -> Option<SomeValue> {
    fn_ptr_somevalue_for_spelling_in(spelling, getbookkeeper().as_deref())
}

pub(crate) fn fn_ptr_somevalue_for_spelling_in(
    spelling: &str,
    bk: Option<&Bookkeeper>,
) -> Option<SomeValue> {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};
    let ft = func_type_from_fn_spelling(spelling, bk)?;
    Some(SomeValue::Ptr(super::model::SomePtr::new(Ptr {
        TO: PtrTarget::Func(ft),
    })))
}

fn empty_functype() -> crate::translator::rtyper::lltypesystem::lltype::FuncType {
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType};
    FuncType {
        args: vec![],
        result: LowLevelType::Void,
    }
}

/// `FuncType` of a Rust fn-pointer spelling. One Rust `fn` type is one
/// `FuncType` (`lltype.py` `FuncType` / `getfunctionptr`), whether the
/// producer is a parameter seed, a field read, or a reified fn item.
pub(crate) fn func_type_from_fn_spelling(
    spelling: &str,
    bk: Option<&Bookkeeper>,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::FuncType> {
    use crate::translator::rtyper::lltypesystem::lltype::FuncType;
    let t = spelling.trim();
    let inner = if let Some(inner) = strip_generic_one(t, "Option<") {
        inner.trim()
    } else {
        t
    };
    if !is_fn_type_spelling(inner) {
        return None;
    }
    let rest = inner.strip_prefix("unsafe ").unwrap_or(inner).trim();
    let after_fn = rest.strip_prefix("fn")?.trim();
    if after_fn.is_empty() {
        return Some(empty_functype());
    }
    let Some((args_str, after_args)) = split_paren_group(after_fn) else {
        return Some(empty_functype());
    };
    let args = if args_str.trim().is_empty() {
        Vec::new()
    } else {
        split_generic_args(args_str)
            .into_iter()
            .map(fn_ptr_arg_lltype)
            .collect()
    };
    let after = after_args.trim();
    let result = if after.is_empty() {
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void
    } else if let Some(ret) = after.strip_prefix("->") {
        fn_ptr_result_lltype(ret.trim(), bk)
    } else {
        return None;
    };
    Some(FuncType { args, result })
}

/// First parenthesized group of a `fn(...)` spelling, with nested
/// `()` / `<>` / `[]` depth. `None` when the group is unclosed.
fn split_paren_group(input: &str) -> Option<(&str, &str)> {
    let bytes = input.as_bytes();
    if bytes.first() != Some(&b'(') {
        return None;
    }
    let mut depth: i32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => {
                depth -= 1;
                if depth == 0 && b == b')' {
                    return Some((&input[1..i], &input[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

fn peel_ref_and_ptr_prefixes(spelling: &str) -> &str {
    let mut t = spelling.trim();
    loop {
        let next = t
            .strip_prefix('&')
            .or_else(|| t.strip_prefix("*const "))
            .or_else(|| t.strip_prefix("*mut "))
            .map(|s| s.trim_start_matches("mut ").trim());
        match next {
            Some(n) if n != t => t = n,
            _ => break,
        }
    }
    t
}

/// Ok payload of `Result<T, E>` when `E` is the bookkeeper's exception
/// carrier (`tyref_is_result_of_carrier`). Other `Result` types stay
/// the ADT; `FuncType.RESULT` is then `valuetype_to_lltype` of that
/// payload, matching `getcallable` / `getfunctionptr`.
fn result_ok_of_exception_carrier<'a>(
    spelling: &'a str,
    bk: Option<&Bookkeeper>,
) -> Option<&'a str> {
    let bk = bk?;
    let args = spelling.trim().strip_prefix("Result<")?.strip_suffix('>')?;
    let parts = split_generic_args(args);
    if parts.len() == 2 && bk.exception_carrier_type_spelling(parts[1].trim()) {
        Some(parts[0].trim())
    } else {
        None
    }
}

/// Arg lltype of a fn-pointer spelling. Matches `valuetype_to_lltype` of
/// the corresponding `ValueType` (`declared_funcptr_type_from_legacy`):
/// integers / bool / float stay primitive; `str` is `STRPTR`; everything
/// else — slices, ADTs, `Result`, pointers, `()` — is `OBJECTPTR`.
fn fn_ptr_arg_lltype(
    spelling: &str,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{GCREF, LowLevelType};
    let t = peel_ref_and_ptr_prefixes(spelling);
    match t {
        "i8" | "i16" | "i32" | "i64" | "isize" | "char" => LowLevelType::Signed,
        "i128" => LowLevelType::SignedLongLongLong,
        "u8" | "u16" | "u32" | "u64" | "usize" => LowLevelType::Unsigned,
        "u128" => LowLevelType::UnsignedLongLongLong,
        "bool" => LowLevelType::Bool,
        "f32" => LowLevelType::SingleFloat,
        "f64" => LowLevelType::Float,
        "str" | "String" | "Wtf8" | "Wtf8Buf" => {
            crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone()
        }
        "StringBuilder" => {
            crate::translator::rtyper::lltypesystem::rbuilder::STRINGBUILDERPTR.clone()
        }
        "gcref" | "GCREF" => GCREF.clone(),
        other if is_fn_type_spelling(other) => LowLevelType::Signed,
        _ => crate::translator::rtyper::rclass::OBJECTPTR.clone(),
    }
}

/// Result lltype of a fn-pointer spelling. `getfunctionptr` / `getcallable`
/// set `FuncType.RESULT` to the callee return variable's lltype
/// (`bindingrepr(v).lowleveltype`); `valuetype_to_lltype` is that
/// projection from the return `ValueType`. A `Result<T, E>` whose `E` is
/// the exception carrier is the residual-call payload `T`
/// (`tyref_is_result_of_carrier`).
fn fn_ptr_result_lltype(
    spelling: &str,
    bk: Option<&Bookkeeper>,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::model::ValueType;
    use crate::translator::rtyper::lltypesystem::lltype::{GCREF, LowLevelType};
    if let Some(ok) = result_ok_of_exception_carrier(spelling, bk) {
        return fn_ptr_result_lltype(ok, bk);
    }
    let t = spelling.trim();
    if t == "()" {
        return LowLevelType::Void;
    }
    let peeled = peel_ref_and_ptr_prefixes(t);
    if peeled == "gcref" || peeled == "GCREF" {
        return GCREF.clone();
    }
    let vt = match peeled {
        "bool" => ValueType::Bool,
        "i8" | "i16" | "i32" | "i64" | "isize" | "char" => ValueType::Int,
        "i128" => ValueType::Int128,
        "u8" | "u16" | "u32" | "u64" | "usize" => ValueType::Unsigned,
        "u128" => ValueType::UInt128,
        "f32" => ValueType::SingleFloat,
        "f64" => ValueType::Float,
        "str" | "String" | "Wtf8" | "Wtf8Buf" => ValueType::Str,
        "StringBuilder" => ValueType::StringBuilder,
        other if is_fn_type_spelling(other) => ValueType::Int,
        _ => ValueType::Ref(None),
    };
    crate::translator::rtyper::cutover::valuetype_to_lltype(&vt)
        .unwrap_or_else(|| crate::translator::rtyper::rclass::OBJECTPTR.clone())
}

/// Charon spelling of a Rust function-pointer type (`fn`, `fn(…)`,
/// `unsafe fn(…)`).  Not a named struct.
fn is_fn_type_spelling(t: &str) -> bool {
    let t = t.trim();
    t == "fn" || t.starts_with("fn(") || t.starts_with("fn ") || t.starts_with("unsafe fn")
}

/// Strip `Wrapper<` prefix and matching `>` suffix from a type string,
/// returning the inner generic-args slice unchanged.  Returns `None` if
/// the prefix is absent OR the suffix is not `>`.
fn strip_generic_one<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    input.strip_prefix(prefix).and_then(|s| s.strip_suffix('>'))
}

/// The field-registry encoding of a Charon `RawPtr` node. A marker ADT
/// wrapping a pointer is a different constructor and does not match.
fn registry_ty_is_raw_ptr(ty: &str) -> bool {
    let t = ty.trim();
    t.starts_with("*mut ") || t.starts_with("*const ")
}

/// Split a generic-args slice on top-level `,` boundaries, respecting
/// nested `<>` / `()` / `[]` depth.
fn split_generic_args(input: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut depth: i32 = 0;
    let mut start: usize = 0;
    let bytes = input.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                let segment = input[start..i].trim();
                if !segment.is_empty() {
                    out.push(segment);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// RPython `hasattr(x, '_freeze_'); assert x._freeze_() is True`
/// (bookkeeper.py:332-333).
///
/// Returns `Ok(true)` when `_freeze_` is present and returns `True`;
/// `Ok(false)` when `_freeze_` is absent (caller falls through to the
/// instance / fallback branch); `Err` when `_freeze_` is present but
/// the invocation fails or returns something other than `True`
/// (mirrors the Python `assert x._freeze_() is True` AssertionError).
///
/// Upstream RPython runs the `_freeze_` body through real Python. The
/// Rust port routes through [`HostObject::call_host`], which handles
/// `NativeCallable` (test fixtures / bootstrap builtins register
/// these) directly and returns
/// [`HostCallError::RequiresFlowEvaluator`] for a `UserFunction`
/// body. The latter surfaces as an explicit
/// "host-graph evaluator not yet landed" `AnnotatorError`
/// rather than a generic call failure — this keeps the boundary
/// between host-executable and flowspace-required callables
/// discoverable at the callsite.
fn call_freeze_method(obj: &HostObject) -> Result<bool, AnnotatorError> {
    let method = match lookup_zero_arg_method(obj, "_freeze_")? {
        Some(m) => m,
        None => return Ok(false),
    };
    let result = method.call_host(&[]).map_err(|err| {
        AnnotatorError::new(match err {
            crate::flowspace::model::HostCallError::RequiresFlowEvaluator(q) => format!(
                "_freeze_() on {q:?} requires the host-graph evaluator \
                 (user function body cannot be executed host-side yet)"
            ),
            other => format!("_freeze_() call failed: {other}"),
        })
    })?;
    match result {
        ConstValue::Bool(true) => Ok(true),
        other => Err(AnnotatorError::new(format!(
            "_freeze_() must return True, got {other:?}"
        ))),
    }
}

/// RPython `if hasattr(x, '_cleanup_'): x._cleanup_()` (bookkeeper.py:341-342).
///
/// Fires the cleanup method for side-effects and ignores the return
/// value. Absence of `_cleanup_` is silently accepted. A `UserFunction`
/// body surfaces the same `host-graph evaluator` marker as
/// `call_freeze_method` so the two boundary limitations are visible
/// at a single grep point.
fn call_cleanup_method(obj: &HostObject) -> Result<(), AnnotatorError> {
    let Some(method) = lookup_zero_arg_method(obj, "_cleanup_")? else {
        return Ok(());
    };
    method.call_host(&[]).map(|_| ()).map_err(|err| {
        AnnotatorError::new(match err {
            crate::flowspace::model::HostCallError::RequiresFlowEvaluator(q) => format!(
                "_cleanup_() on {q:?} requires the host-graph evaluator \
                 (user function body cannot be executed host-side yet)"
            ),
            other => format!("_cleanup_() call failed: {other}"),
        })
    })
}

/// Helper: resolve a zero-arg method `name` on `obj`, binding `self`
/// where applicable (Instance / Module / Class host kinds).
///
/// Returns `Some(callable)` when the name is present and resolves to
/// a callable HostObject (BoundMethod or NativeCallable); `None`
/// otherwise. Non-callable attributes that would raise `TypeError`
/// at call time are deliberately returned as `Some` so
/// [`HostObject::call_host`] can surface the failure with the
/// upstream shape.
fn lookup_zero_arg_method(
    obj: &HostObject,
    name: &str,
) -> Result<Option<HostObject>, AnnotatorError> {
    use crate::flowspace::model::host_getattr;
    match host_getattr(obj, name) {
        Ok(ConstValue::HostObject(h)) => Ok(Some(h)),
        Ok(other) => Err(AnnotatorError::new(format!(
            "{} exists on {:?} but is not callable: {other:?}",
            name,
            obj.qualname()
        ))),
        Err(crate::flowspace::model::HostGetAttrError::Missing) => Ok(None),
        Err(crate::flowspace::model::HostGetAttrError::Unsupported) => Err(AnnotatorError::new(
            format!("{} lookup on {:?} is unsupported", name, obj.qualname()),
        )),
    }
}

/// RPython `CallOp.build_args(args_s)` dispatch by opname
/// (operation.py for `simple_call`, 699-701 for `call_args`).
///
/// `simple_call` wraps `args_s` in a flat `ArgumentsForTranslation`;
/// `call_args` reads the encoded call shape out of `args_s[0].const`
/// (`ConstValue::Tuple([Int(cnt), Tuple([Str(k0), …]), Bool(star)])`,
/// see `flowcontext::build_call_shape_constant`) and reconstructs a
/// CallShape + tail `args_s[1..]`.
// check_no_flags_on_instances walker (bookkeeper.py)

/// Entry point for the recursive sanity walk — inspects a `ClassDef`
/// and recurses through its attributes. `seen_classdefs` /
/// `seen_listitems` break cycles for attrs whose annotation cycles back.
fn check_no_flags_classdef(
    clsdef: &Rc<RefCell<ClassDef>>,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    let key = ClassDefKey::from_classdef(clsdef);
    if !seen_classdefs.insert(key) {
        return;
    }
    let attrs_snapshot: Vec<SomeValue> = clsdef
        .borrow()
        .attrs
        .values()
        .map(|a| a.s_value.clone())
        .collect();
    for s_attr in attrs_snapshot {
        check_no_flags_value(&s_attr, seen_classdefs, seen_listitems);
    }
}

/// upstream `check_no_flags(SomeInstance | SomeList | SomeDict |
/// SomeTuple)` arms; recurses into container element types.
fn check_no_flags_value(
    s: &SomeValue,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    match s {
        SomeValue::Instance(inst) => {
            assert!(
                inst.flags.is_empty(),
                "instance annotation with flags escaped to the heap"
            );
            if let Some(classdef) = &inst.classdef {
                check_no_flags_classdef(classdef, seen_classdefs, seen_listitems);
            }
        }
        SomeValue::List(list) => {
            let listitem = list.listdef.listitem_rc();
            check_no_flags_listitem(&listitem, seen_classdefs, seen_listitems);
        }
        SomeValue::Dict(dict) => {
            let dictkey = dict.dictdef.dictkey_rc();
            check_no_flags_listitem(&dictkey, seen_classdefs, seen_listitems);
            let dictvalue = dict.dictdef.dictvalue_rc();
            check_no_flags_listitem(&dictvalue, seen_classdefs, seen_listitems);
        }
        SomeValue::Tuple(tup) => {
            for item in &tup.items {
                check_no_flags_value(item, seen_classdefs, seen_listitems);
            }
        }
        _ => {}
    }
}

fn check_no_flags_listitem(
    li: &Rc<RefCell<super::listdef::ListItem>>,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    let id = Rc::as_ptr(li) as usize;
    if !seen_listitems.insert(id) {
        return;
    }
    let s_value = li.borrow().s_value.clone();
    check_no_flags_value(&s_value, seen_classdefs, seen_listitems);
}

pub(crate) fn build_args_for_op(
    opname: &str,
    args_s: &[Option<SomeValue>],
) -> Result<ArgumentsForTranslation, AnnotatorError> {
    match opname {
        "simple_call" => Ok(super::argument::simple_args_opt(args_s.to_vec())),
        "call_args" => {
            if args_s.is_empty() {
                return Err(AnnotatorError::new(
                    "build_args_for_op(call_args): missing shape argument",
                ));
            }
            // RPython `call_args` puts a concrete shape Constant at
            // position 0; pyre's frontend mirrors this in
            // `flowcontext::build_call_shape_constant`, so position 0
            // is always bound (matched by the unwrap below — None
            // here is a producer bug).
            let shape_arg = args_s[0].as_ref().ok_or_else(|| {
                AnnotatorError::new("build_args_for_op(call_args): args_s[0] is unbound")
            })?;
            let shape_const = shape_arg.const_().ok_or_else(|| {
                AnnotatorError::new("build_args_for_op(call_args): args_s[0] is not a Constant")
            })?;
            let shape = call_shape_from_const(shape_const)?;
            // Position 0 was the shape Constant; remaining positions
            // may carry None for unbound caller args.
            let mut data_w: Vec<Option<SomeValue>> = Vec::with_capacity(args_s.len() - 1);
            data_w.extend_from_slice(&args_s[1..]);
            Ok(ArgumentsForTranslation::fromshape(&shape, data_w))
        }
        other => Err(AnnotatorError::new(format!(
            "build_args_for_op: unsupported call opname {other:?}"
        ))),
    }
}

/// Decode the `CallShape` tuple embedded in `call_args` operations.
/// Mirrors the encoding produced by
/// `flowcontext::build_call_shape_constant`:
/// `Tuple([Int(shape_cnt), Tuple([Str(key)*]), Bool(shape_star)])`.
fn call_shape_from_const(cv: &ConstValue) -> Result<CallShape, AnnotatorError> {
    let items = match cv {
        ConstValue::Tuple(items) => items,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: expected ConstValue::Tuple",
            ));
        }
    };
    if items.len() != 3 {
        return Err(AnnotatorError::new(
            "call_shape_from_const: tuple must have 3 elements",
        ));
    }
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_cnt is not Int",
            ));
        }
    };
    let keys_tuple = match &items[1] {
        ConstValue::Tuple(k) => k,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_keys is not Tuple",
            ));
        }
    };
    let mut shape_keys = Vec::with_capacity(keys_tuple.len());
    for k in keys_tuple {
        if let Some(s) = k.as_text() {
            shape_keys.push(s.to_string());
        } else {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_keys element is not Str",
            ));
        }
    }
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_star is not Bool",
            ));
        }
    };
    Ok(CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}

/// RPython `getbookkeeper()` free function (bookkeeper.py).
///
/// ```python
/// def getbookkeeper():
///     try:
///         return TLS.bookkeeper
///     except AttributeError:
///         return None
/// ```
pub fn getbookkeeper() -> Option<Rc<Bookkeeper>> {
    super::model::TLS.with(|state| state.borrow().bookkeeper.clone())
}

/// RPython `immutablevalue(x)` free function (bookkeeper.py).
///
/// Delegates to [`Bookkeeper::immutablevalue`] on the thread-local
/// bookkeeper. Panics when called without a live bookkeeper — upstream
/// raises `AttributeError: 'NoneType' object has no attribute
/// 'immutablevalue'` in the same situation.
pub fn immutablevalue(x: &ConstValue) -> Result<SomeValue, AnnotatorError> {
    let bk = getbookkeeper().expect("immutablevalue() called without an active bookkeeper");
    bk.immutablevalue(x)
}

/// RPython `origin_of_meth(boundmeth)` (bookkeeper.py).
pub fn origin_of_meth(boundmeth: &HostObject) -> Result<(&HostObject, String), AnnotatorError> {
    let origin_class = boundmeth.bound_method_origin_class().ok_or_else(|| {
        AnnotatorError::new(format!(
            "could not match bound-method to attribute name: {boundmeth:?}"
        ))
    })?;
    let name = boundmeth.bound_method_name().ok_or_else(|| {
        AnnotatorError::new(format!(
            "could not match bound-method to attribute name: {boundmeth:?}"
        ))
    })?;
    Ok((origin_class, name.to_string()))
}

/// RPython `ishashable(x)` (bookkeeper.py).
pub fn ishashable(_x: &HostObject) -> bool {
    true
}

/// RAII guard returned by [`Bookkeeper::at_position`]. Mirrors the
/// upstream `@contextmanager` exit — calls [`Bookkeeper::leave`] on
/// drop unless the fast-path at bookkeeper.py skipped the
/// initial enter.
pub(crate) struct PositionGuard {
    bk: Rc<Bookkeeper>,
    skip_leave: bool,
}

impl Drop for PositionGuard {
    fn drop(&mut self) {
        if !self.skip_leave {
            self.bk.leave();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeChar, SomeFloat};
    use crate::flowspace::model::GraphFunc;
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn compiled_graph_func(src: &str, defaults: Vec<Constant>) -> GraphFunc {
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
        GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            defaults,
        )
    }

    #[test]
    fn check_no_flags_on_instances_accepts_empty_flags() {
        use crate::annotator::classdesc::{ClassDef, ClassDesc};
        let bk = bk();
        let pyobj = crate::flowspace::model::HostObject::new_class("pkg.A", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.A".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.classdefs.borrow_mut().push(classdef);
        // No attrs, no SomeInstance carries flags — walker must not panic.
        bk.check_no_flags_on_instances();
    }

    #[test]
    fn immutablevalue_preserves_full_width_integer_known_types_and_constants() {
        use crate::annotator::model::{KnownType, SomeObjectTrait};

        for (value, expected_known, expected_nonneg) in [
            (
                ConstValue::Int128(i128::MIN),
                KnownType::LongLongLong,
                false,
            ),
            (ConstValue::Int128(i128::MAX), KnownType::LongLongLong, true),
            (
                ConstValue::UInt128(u128::MAX),
                KnownType::ULongLongLong,
                true,
            ),
        ] {
            let s = bk()
                .immutablevalue(&value)
                .expect("wide integer constant annotates");
            let SomeValue::Integer(integer) = s else {
                panic!("wide integer constant must annotate as SomeInteger")
            };
            assert_eq!(integer.knowntype(), expected_known);
            assert_eq!(integer.nonneg, expected_nonneg);
            assert_eq!(
                integer
                    .base
                    .const_box
                    .as_ref()
                    .map(|constant| &constant.value),
                Some(&value),
                "wide integer constant must remain lossless in the annotator"
            );
        }
    }

    #[test]
    #[should_panic(expected = "instance annotation with flags escaped to the heap")]
    fn check_no_flags_on_instances_panics_on_non_empty_flags() {
        use crate::annotator::classdesc::{ClassDef, ClassDesc};
        use crate::annotator::model::SomeInstance;
        let bk = bk();
        let pyobj = crate::flowspace::model::HostObject::new_class("pkg.A", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.A".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        // Plant an attribute whose value is a SomeInstance with
        // non-empty flags — upstream's assertion must fire.
        let mut flags = std::collections::BTreeMap::new();
        flags.insert("nonneg".to_string(), true);
        let bad = SomeValue::Instance(SomeInstance::new(Some(classdef.clone()), false, flags));
        let mut attrs_attr = crate::annotator::classdesc::Attribute::new("x");
        attrs_attr.s_value = bad;
        classdef
            .borrow_mut()
            .attrs
            .insert("x".to_string(), attrs_attr);
        bk.classdefs.borrow_mut().push(classdef);
        bk.check_no_flags_on_instances();
    }

    /// `class OperationError(Exception)`: the error carrier is minted as an
    /// `Exception` subclass whichever entry point mints it first, and a
    /// caught `Exception` narrowed to it and the carrier's own instances
    /// share one lattice.
    #[test]
    fn the_error_carrier_is_minted_as_an_exception_subclass() {
        for bases_entry in [false, true] {
            let bk = bk();
            bk.set_exception_carrier("error::PyErrorObject");
            let host = if bases_entry {
                bk.intern_class_by_qualname_with_bases("error::PyErrorObject", vec![])
            } else {
                bk.intern_class_by_qualname("error::PyErrorObject")
            };
            let carrier = bk.getuniqueclassdef(&host).expect("carrier classdef");
            let exception = bk
                .getuniqueclassdef(
                    &crate::flowspace::model::HOST_ENV
                        .lookup_exception_class("Exception")
                        .unwrap(),
                )
                .expect("Exception classdef");
            assert!(carrier.borrow().issubclass(&exception));
            let s_carrier = crate::annotator::model::SomeValue::Instance(
                crate::annotator::model::SomeInstance::new(
                    Some(carrier.clone()),
                    false,
                    Default::default(),
                ),
            );
            let s_exception = crate::annotator::model::SomeValue::Instance(
                crate::annotator::model::SomeInstance::new(
                    Some(exception.clone()),
                    false,
                    Default::default(),
                ),
            );
            assert_eq!(
                crate::annotator::model::unionof([&s_carrier, &s_exception]).unwrap(),
                s_exception
            );
        }
        // Another struct stays base-less.
        let bk = bk();
        bk.set_exception_carrier("error::PyErrorObject");
        let other = bk.intern_class_by_qualname("pyre_object::pyobject::PyObject");
        assert!(other.class_bases().map_or(true, |b| b.is_empty()));
    }

    #[test]
    fn the_error_carrier_handle_is_a_subclass_not_an_alias() {
        let bk = bk();
        bk.set_exception_carrier("error::PyErrorObject");
        bk.alias_exception_carrier_handle("error::PyError");
        let object = bk.exception_carrier_class().expect("object class");
        let handle = bk.intern_class_by_qualname("error::PyError");
        let handle_cd = bk.getuniqueclassdef(&handle).expect("handle classdef");
        let object_cd = bk.getuniqueclassdef(&object).expect("object classdef");
        assert!(
            !Rc::ptr_eq(&handle_cd, &object_cd),
            "the handle keeps its own ClassDef"
        );
        assert!(handle_cd.borrow().issubclass(&object_cd));
    }

    #[test]
    fn intern_class_stamps_the_declared_virtualizable_list() {
        crate::virtualizable_decl::register_virtualizable_declarations([(
            "Frame".to_string(),
            vec!["last_instr".to_string()],
        )]);
        let bk = bk();
        let host = bk.intern_class_by_qualname("Frame");
        let listed = host
            .class_get("_virtualizable_")
            .expect("the declared class carries _virtualizable_");
        let ConstValue::List(items) = listed else {
            panic!("_virtualizable_ must be a list");
        };
        assert!(
            items
                .iter()
                .any(|item| matches!(item, ConstValue::ByteStr(bytes) if bytes == b"last_instr"))
        );
        let classdesc = crate::annotator::classdesc::ClassDesc::new(
            &bk,
            host,
            Some("Frame".into()),
            None,
            None,
        )
        .expect("ClassDesc");
        assert!(
            !matches!(
                classdesc.borrow().get_param("_virtualizable_", None, true),
                ConstValue::None
            ),
            "classdesc.get_param('_virtualizable_') must see the class attribute"
        );
        crate::virtualizable_decl::register_virtualizable_declarations(std::iter::empty::<(
            String,
            Vec<String>,
        )>());
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_registers_by_identity_and_projects_fields() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![
                ("pycode".to_string(), "PyCode".to_string()),
                ("depth".to_string(), "i64".to_string()),
            ],
        );
        reg.fields.insert(
            "PyCode".to_string(),
            vec![("argcount".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        let cd1 = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers");
        // Identity: a second lookup returns the SAME ClassDef Rc — the
        // type-root string resolves to one interned host class object.
        let cd2 = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame re-lookup");
        assert!(
            Rc::ptr_eq(&cd1, &cd2),
            "type-root resolves to one identity-keyed ClassDef"
        );

        // Orthodox registration: the ClassDef lands in
        // `bookkeeper.classdefs` (not a side-cache-only stub).
        let registered = bk.classdef_snapshot().iter().any(|c| Rc::ptr_eq(c, &cd1));
        assert!(
            registered,
            "ClassDef must be registered in bookkeeper.classdefs by identity"
        );

        // Instance attributes projected from the registry.
        {
            let g = cd1.borrow();
            let depth = g.attrs.get("depth").expect("depth attr projected");
            assert!(
                matches!(depth.s_value, SomeValue::Integer(_)),
                "i64 field -> SomeInteger"
            );
            let pycode = g.attrs.get("pycode").expect("pycode attr projected");
            assert!(
                matches!(pycode.s_value, SomeValue::Instance(_)),
                "struct-typed field -> SomeInstance(inner classdef)"
            );
        }

        // The transitively-referenced inner struct is ALSO registered
        // with its own fields projected (no attrs-empty inner stub).
        let inner = bk
            .getuniqueclassdef_for_struct_root("PyCode")
            .expect("PyCode registers");
        assert!(
            inner.borrow().attrs.contains_key("argcount"),
            "inner struct's own fields are projected too"
        );

        // Order-independent identity: the inner classdef reached through
        // `PyFrame.pycode` is the SAME Rc as the standalone `PyCode`
        // lookup.  With a single identity route there is no name-keyed
        // path that could mint a divergent unregistered stub.
        let pycode_via_field = {
            let g = cd1.borrow();
            match &g.attrs.get("pycode").expect("pycode attr").s_value {
                SomeValue::Instance(si) => si.classdef.clone(),
                _ => None,
            }
        };
        assert!(
            pycode_via_field
                .as_ref()
                .is_some_and(|c| Rc::ptr_eq(c, &inner)),
            "PyFrame.pycode inner classdef must be the same Rc as the standalone PyCode lookup"
        );
    }

    #[test]
    fn compiler_constant_payload_types_project_without_lattice_bottom() {
        use crate::annotator::model::SomeValue;
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "CodeObject".to_string(),
            vec![("max_stackdepth".to_string(), "u32".to_string())],
        );
        reg.fields.insert(
            "num_complex::Complex".to_string(),
            vec![
                ("re".to_string(), "f64".to_string()),
                ("im".to_string(), "f64".to_string()),
            ],
        );
        reg.fields.insert(
            "Complex".to_string(),
            reg.fields["num_complex::Complex"].clone(),
        );
        bk.set_struct_fields(Rc::new(reg));

        assert!(
            matches!(bk.project_struct_field_type("BigInt"), SomeValue::Instance(ref s) if s.classdef.is_none()),
            "dependency-opaque compiler BigInt remains one GC reference"
        );
        assert!(
            matches!(bk.project_struct_field_type("fn"), SomeValue::Ptr(_)),
            "a bare fn field is Ptr(FuncType), not a classdef-less instance"
        );
        assert!(
            matches!(
                bk.project_struct_field_type("fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>"),
                SomeValue::Ptr(_)
            ),
            "a Rust fn-pointer spelling is Ptr(FuncType)"
        );
        assert!(
            matches!(
                bk.project_struct_field_type("Wtf8Buf"),
                SomeValue::String(_)
            ),
            "Wtf8Buf shares the immutable string annotation"
        );

        let SomeValue::Instance(complex) = bk.project_struct_field_type("Complex<f64>") else {
            panic!("Complex<f64> must project to its registered struct class")
        };
        let complex_cd = complex.classdef.expect("Complex classdef");
        let projected_complex = bk
            .getuniqueclassdef_for_struct_root("num_complex::Complex")
            .expect("Complex rows project");
        assert!(Rc::ptr_eq(&complex_cd, &projected_complex));
        let complex_attrs = &complex_cd.borrow().attrs;
        assert!(matches!(
            complex_attrs.get("re").map(|a| &a.s_value),
            Some(SomeValue::Float(_))
        ));
        assert!(matches!(
            complex_attrs.get("im").map(|a| &a.s_value),
            Some(SomeValue::Float(_))
        ));

        let SomeValue::Instance(code) =
            bk.project_struct_field_type("Box<CodeObject<ConstantData>>")
        else {
            panic!("concrete CodeObject instantiation must use the template class")
        };
        let code_cd = code.classdef.expect("CodeObject classdef");
        let template = bk
            .getuniqueclassdef_for_struct_root("CodeObject")
            .expect("CodeObject template classdef");
        assert!(
            code_cd.borrow().attrs.contains_key("max_stackdepth"),
            "template rows must be projected onto the concrete generic use"
        );
        assert!(
            Rc::ptr_eq(&code_cd, &template),
            "RPython has one CodeObject class, not per-generic classdefs"
        );
    }

    #[test]
    fn fn_ptr_spellings_of_one_rust_type_share_one_functype() {
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, PtrTarget};
        use crate::translator::rtyper::rclass::OBJECTPTR;

        let with_ref = "fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>";
        let stripped = "fn([PyObjectRef]) -> Result<PyObjectRef, PyError>";
        let a = fn_ptr_somevalue_for_spelling(with_ref).expect("ref spelling is a fn pointer");
        let b = fn_ptr_somevalue_for_spelling(stripped).expect("stripped spelling is a fn pointer");
        let SomeValue::Ptr(pa) = &a else {
            panic!("ref spelling must seed SomePtr, got {a:?}");
        };
        let SomeValue::Ptr(pb) = &b else {
            panic!("stripped spelling must seed SomePtr, got {b:?}");
        };
        assert_eq!(
            pa.ll_ptrtype, pb.ll_ptrtype,
            "one Rust fn type is one FuncType (`getfunctionptr`)"
        );
        let expected = FuncType {
            args: vec![OBJECTPTR.clone()],
            result: OBJECTPTR.clone(),
        };
        assert_eq!(
            pa.ll_ptrtype.TO,
            PtrTarget::Func(expected),
            "BuiltinCodeFn args/result rtype as OBJECTPTR (`valuetype_to_lltype` / `return_token_to_lltype`)"
        );
        crate::annotator::model::union(&a, &b)
            .expect("equal FuncTypes union (`llannotation.py` SomePtr.union)");
        let option_spelling = "Option<fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>>";
        let c = fn_ptr_somevalue_for_spelling(option_spelling)
            .expect("Option of that fn type is the same pointer");
        let SomeValue::Ptr(pc) = &c else {
            panic!("Option<fn> spelling must seed SomePtr, got {c:?}");
        };
        assert_eq!(
            pa.ll_ptrtype, pc.ll_ptrtype,
            "Option wrapping does not change FuncType (`getfunctionptr`)"
        );
        crate::annotator::model::union(&a, &c)
            .expect("Option<fn(ARGS)->RET> unions with fn(ARGS)->RET");
        let empty = raw_fn_ptr_somevalue();
        assert!(
            crate::annotator::model::union(&empty, &a).is_err(),
            "distinct FuncTypes must not union; the seed is the signature, not a merge of pointer types"
        );
    }

    #[test]
    fn fn_ptr_result_lltype_follows_callee_return_valuetype() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType, PtrTarget};
        use crate::translator::rtyper::rclass::OBJECTPTR;

        let math = fn_ptr_somevalue_for_spelling("fn(f64) -> Result<f64, MathError>")
            .expect("math Result is a fn pointer");
        let SomeValue::Ptr(p_math) = &math else {
            panic!("math spelling must seed SomePtr, got {math:?}");
        };
        assert_eq!(
            p_math.ll_ptrtype.TO,
            PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Float],
                result: OBJECTPTR.clone(),
            }),
            "a non-carrier Result stays the ADT (`getfunctionptr` of that return annotation)"
        );

        let bk = bk();
        bk.set_exception_carrier("error::PyErrorObject");
        bk.alias_exception_carrier_handle("error::PyError");
        let unit = fn_ptr_somevalue_for_spelling_in("fn() -> Result<(), PyError>", Some(&bk))
            .expect("carrier Result is a fn pointer");
        let SomeValue::Ptr(p_unit) = &unit else {
            panic!("unit spelling must seed SomePtr, got {unit:?}");
        };
        assert_eq!(
            p_unit.ll_ptrtype.TO,
            PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
            "carrier Result<(), E> is Void (`getcallable` of the success return)"
        );
        let signed =
            fn_ptr_somevalue_for_spelling_in("fn() -> Result<i64, error::PyError>", Some(&bk))
                .expect("carrier i64 Result is a fn pointer");
        let SomeValue::Ptr(p_signed) = &signed else {
            panic!("signed spelling must seed SomePtr, got {signed:?}");
        };
        assert_eq!(
            p_signed.ll_ptrtype.TO,
            PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Signed,
            }),
            "carrier Result<i64, E> is Signed (`valuetype_to_lltype` of the payload)"
        );
    }

    #[test]
    fn bytes_block_projects_to_rpython_string_storage() {
        use crate::annotator::model::SomeValue;

        let bk = bk();
        assert!(
            matches!(
                bk.project_struct_field_type("BytesBlock"),
                SomeValue::String(_)
            ),
            "BytesBlock is the owner of RPython STR.chars, not a nominal instance"
        );
        assert!(
            matches!(
                bk.project_struct_field_type("*const BytesBlock"),
                SomeValue::String(_)
            ),
            "a BytesBlock pointer must keep the same string annotation"
        );
    }

    #[test]
    fn struct_root_alias_and_registered_root_share_one_classdef() {
        use crate::annotator::model::SomeValue;
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

        let SomeValue::Instance(alias_item) = bk.project_struct_field_type("PyObjectRef") else {
            panic!("PyObjectRef must project to the PyObject instance annotation")
        };
        let alias_class = alias_item.classdef.expect("typed PyObjectRef classdef");
        let root_classdef = bk
            .getuniqueclassdef_for_struct_root("pyobject::PyObject")
            .expect("PyObject classdef");
        assert!(
            Rc::ptr_eq(&alias_class, &root_classdef),
            "the source alias and registered PyObject root must share one ClassDef"
        );

        let SomeValue::List(array) = bk.project_struct_field_type("[PyObjectRef; 0]") else {
            panic!("a PyObjectRef array tail must project to a list")
        };
        let SomeValue::Instance(array_item) = array.listdef.s_value() else {
            panic!("the projected array item must be a PyObject instance")
        };
        assert!(
            array_item
                .classdef
                .as_ref()
                .is_some_and(|classdef| Rc::ptr_eq(classdef, &root_classdef))
        );

        let SomeValue::List(qualified_array) =
            bk.project_struct_field_type("[pyre_object::PyObjectRef; 0]")
        else {
            panic!("a qualified PyObjectRef array tail must project to a list")
        };
        let SomeValue::Instance(qualified_item) = qualified_array.listdef.s_value() else {
            panic!("the qualified array item must be a PyObject instance")
        };
        assert!(
            qualified_item
                .classdef
                .as_ref()
                .is_some_and(|classdef| Rc::ptr_eq(classdef, &root_classdef))
        );
    }

    #[test]
    fn qualified_fixed_object_array_projects_typed_instance_items() {
        use crate::annotator::model::SomeValue;
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyre_object::object_array::FixedObjectArray".to_string(),
            vec![
                ("len".to_string(), "usize".to_string()),
                (
                    "_items".to_string(),
                    "[pyre_object::PyObjectRef; 0]".to_string(),
                ),
            ],
        );
        reg.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        let SomeValue::List(array) =
            bk.project_struct_field_type("*mut pyre_object::object_array::FixedObjectArray")
        else {
            panic!("qualified FixedObjectArray pointer must project to its item list")
        };
        let SomeValue::Instance(item) = array.listdef.s_value() else {
            panic!("FixedObjectArray item must be a PyObject instance")
        };
        assert!(
            item.classdef.is_some(),
            "PyObject item must keep its ClassDef"
        );
    }

    #[test]
    #[ignore]
    fn real_llbc_pyframe_locals_array_projects_typed_instance_items() {
        use crate::annotator::model::SomeValue;
        use majit_charon_reader::Llbc;

        let interpreter_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-interpreter.ullbc"
        );
        let object_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-object.ullbc"
        );
        let interpreter = Llbc::load(interpreter_path).expect("load real interpreter LLBC");
        let object = Llbc::load(object_path).expect("load real object LLBC");
        let program = crate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
            &[object, interpreter],
            crate::HostStaticAddrs::default(),
            &["pyframe"],
            &["__metadata_only__"],
        )
        .expect("derive real interpreter metadata");
        let field_ty = program
            .struct_fields
            .field_type("PyFrame", "locals_cells_stack_w")
            .expect("PyFrame locals array field")
            .to_string();
        let fixed_array_rows: Vec<_> = program
            .struct_fields
            .fields
            .iter()
            .filter(|(key, _)| key.ends_with("FixedObjectArray"))
            .map(|(key, rows)| (key.clone(), rows.clone()))
            .collect();
        let _registry =
            crate::test_support::register_struct_origins_serialized(program.struct_origins);

        let bk = bk();
        bk.set_struct_fields(Rc::new(program.struct_fields));
        let pyframe = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame classdef");
        let locals = pyframe
            .borrow()
            .attrs
            .get("locals_cells_stack_w")
            .expect("locals array attr")
            .s_value
            .clone();
        let SomeValue::List(array) = locals else {
            panic!(
                "{field_ty} must project to a list, got {locals:?}; registry rows: {fixed_array_rows:?}"
            )
        };
        let SomeValue::Instance(item) = array.listdef.s_value() else {
            panic!("{field_ty} item must be an instance")
        };
        assert!(
            item.classdef.is_some(),
            "{field_ty} item must retain the PyObject ClassDef"
        );
    }

    #[test]
    #[ignore]
    fn real_merged_llbc_memoryview_closure_tuple_projects_instance_item() {
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::rmodel::Repr;
        use majit_charon_reader::Llbc;

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../build/llbc/");
        let object = Llbc::load(format!("{root}pyre-object.ullbc")).expect("load real object LLBC");
        let interpreter = Llbc::load(format!("{root}pyre-interpreter.ullbc"))
            .expect("load real interpreter LLBC");
        let jit = Llbc::load(format!("{root}pyre-jit.ullbc")).expect("load real jit LLBC");
        let program = crate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
            &[object, interpreter, jit],
            crate::HostStaticAddrs::default(),
            &["builtins"],
            &["memoryview_is_native_release_descr", "call_once"],
        )
        .expect("derive merged production metadata");
        let tuple = "Tuple<*mut PyObject>";
        let target = program
            .functions
            .iter()
            .find(|function| function.name == "memoryview_is_native_release_descr")
            .expect("memoryview caller graph");
        let written = target
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                crate::model::OpKind::FieldWrite { field, value, .. }
                    if field.owner_root.as_deref() == Some(tuple) =>
                {
                    value.as_variable().cloned()
                }
                _ => None,
            })
            .expect("memoryview closure Args tuple payload write");
        assert!(
            target
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    op.result.as_ref() == Some(&written)
                        && matches!(
                            &op.kind,
                            crate::model::OpKind::Call {
                                target: crate::model::CallTarget::FunctionPath { segments, .. },
                                ..
                            } if crate::model::cast_instance_root(&op.kind) == Some("PyObject")
                        )
                }),
            "the tuple writer must preserve the source PyObject class"
        );
        let rows = program.struct_fields.fields.get(tuple).unwrap_or_else(|| {
            panic!(
                "missing {tuple}; matching keys: {:?}",
                program
                    .struct_fields
                    .fields
                    .keys()
                    .filter(|key| key.contains("Tuple<*mut PyObject"))
                    .collect::<Vec<_>>()
            )
        });
        assert_eq!(
            rows,
            &vec![("__pos_0".to_string(), "*mut PyObject".to_string())]
        );
        let _registry =
            crate::test_support::register_struct_origins_serialized(program.struct_origins.clone());
        for (qualified, fields) in &program.struct_field_attrs {
            crate::annotator::classdesc::register_struct_fields(qualified, fields);
        }

        let bk = bk();
        bk.set_struct_fields(Rc::new(program.struct_fields));
        let tuple_cd = bk
            .getuniqueclassdef_for_struct_root(tuple)
            .expect("closure Args tuple classdef");
        let item = tuple_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("closure Args item attr")
            .s_value
            .clone();
        let SomeValue::Instance(item) = item else {
            panic!("closure Args item must be an instance, got {item:?}")
        };
        let item_cd = item
            .classdef
            .expect("closure Args item must retain the PyObject classdef");

        let registry = crate::translator::rtyper::call_registry::CallRegistry::new(bk.clone());
        let (_, rtyper) = registry
            .ensure_session()
            .expect("create shared rtyper session");
        let tuple_repr = crate::translator::rtyper::rclass::getinstancerepr(
            &rtyper,
            Some(&tuple_cd),
            crate::translator::rtyper::rclass::Flavor::Gc,
        )
        .expect("closure Args tuple repr");
        crate::translator::rtyper::rmodel::Repr::setup(tuple_repr.as_ref())
            .expect("set up closure Args tuple repr");
        let item_repr = tuple_repr
            .fields()
            .get("__pos_0")
            .expect("closure Args item repr")
            .1
            .clone();
        let expected_repr = crate::translator::rtyper::rclass::getinstancerepr(
            &rtyper,
            Some(&item_cd),
            crate::translator::rtyper::rclass::Flavor::Gc,
        )
        .expect("PyObject repr");
        crate::translator::rtyper::rmodel::Repr::setup(expected_repr.as_ref())
            .expect("set up PyObject repr");
        assert_eq!(
            item_repr.lowleveltype(),
            expected_repr.lowleveltype(),
            "closure Args tuple field repr must be the concrete PyObject repr"
        );
    }

    #[test]
    fn nonheader_dotted_constructor_reuses_crate_relative_struct_identity() {
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        let rows = vec![("__pos_0".to_string(), "u64".to_string())];
        reg.fields.insert(
            "pyre_object::celldict::VersionTag".to_string(),
            rows.clone(),
        );
        reg.fields.insert("VersionTag".to_string(), rows);
        bk.set_struct_fields(Rc::new(reg));

        let from_field = bk.intern_class_by_qualname("celldict::VersionTag");
        let from_ctor = bk.intern_class_by_qualname("pyre_object.celldict.VersionTag");
        assert_eq!(from_field, from_ctor);
        let field_class = bk
            .getuniqueclassdef(&from_field)
            .expect("field-side VersionTag classdef");
        let ctor_class = bk
            .getuniqueclassdef(&from_ctor)
            .expect("constructor-side VersionTag classdef");
        assert!(Rc::ptr_eq(&field_class, &ctor_class));
    }

    #[test]
    fn header_struct_spellings_reuse_one_class_identity_and_base() {
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        let rows = vec![
            ("ob_header".to_string(), "PyObject".to_string()),
            ("value".to_string(), "*mut Wtf8Buf".to_string()),
        ];
        reg.fields.insert(
            "fixture_crate::unicodeobject::W_UnicodeObject".to_string(),
            rows.clone(),
        );
        reg.fields.insert("W_UnicodeObject".to_string(), rows);
        reg.fields.insert(
            "PyObject".to_string(),
            vec![("ob_type".to_string(), "*const PyType".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        let from_field = bk.intern_class_by_qualname("unicodeobject::W_UnicodeObject");
        let from_ctor = bk.intern_class_by_qualname("fixture_crate.unicodeobject.W_UnicodeObject");
        assert_eq!(from_field, from_ctor);

        let classdef = bk.getuniqueclassdef(&from_field).expect("header classdef");
        assert!(
            classdef.borrow().basedef.is_some(),
            "normalizing the class identity must retain its embedded-header base"
        );
    }

    #[test]
    fn getuniqueclassdef_for_enum_variant_subclasses_the_base() {
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Discriminant-only base; the variant carries its own payload.
        reg.fields.insert(
            "Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let variant = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant registers");

        // The variant is a subclass of the base — its basedef chain
        // reaches the base, so commonbase(base, variant) == base.
        assert!(
            variant.borrow().issubclass(&base),
            "variant ClassDef must be a subclass of the base"
        );
        let common = crate::annotator::classdesc::ClassDef::commonbase(&base, &variant)
            .expect("commonbase exists");
        assert!(
            Rc::ptr_eq(&common, &base),
            "commonbase(base, variant) must be the base"
        );

        // Identity: a second call returns the same Rc (canonical cache).
        let variant2 = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant re-lookup");
        assert!(Rc::ptr_eq(&variant, &variant2), "variant identity stable");

        // The narrowing resolver, the prologue pre-mint, and the ctor arm
        // must all intern ONE variant classdef.  A `.`-vs-`::` split would
        // mint a second, distinct classdef that the single
        // `assign_inheritance_ids` pass never reaches; assert the resolver
        // returns the classdef the pre-mint / ctor obtain through the
        // shared `intern_enum_variant_host` primitive (the `::`-qualified
        // key).
        let premint_host = bk.intern_enum_variant_host("Color", "Rgb");
        let premint = bk
            .getuniqueclassdef(&premint_host)
            .expect("pre-mint variant classdef");
        assert!(
            Rc::ptr_eq(&variant, &premint),
            "narrowing variant must be the pre-mint/ctor classdef from intern_enum_variant_host"
        );

        // The payoff: improve() narrows SomeInstance(base) to
        // SomeInstance(variant) given the variant as the refinement —
        // exactly what the discriminant-keyed knowntypedata will feed it.
        let s_base = SomeValue::Instance(SomeInstance::new(
            Some(base.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_variant = SomeValue::Instance(SomeInstance::new(
            Some(variant.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let narrowed = crate::annotator::binaryop::improve(&s_base, &s_variant);
        match narrowed {
            SomeValue::Instance(si) => assert!(
                si.classdef
                    .as_ref()
                    .is_some_and(|c| Rc::ptr_eq(c, &variant)),
                "improve(base, variant) narrows to the variant classdef"
            ),
            other => panic!("expected SomeInstance(variant), got {other:?}"),
        }
    }

    #[test]
    fn generic_enum_variant_shares_a_base_with_the_unsuffixed_enum() {
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Only the crate-relative key is registered.  A bare `StepResult`
        // beside it would make the suffix match ambiguous.
        for base in ["pyopcode::StepResult"] {
            reg.fields.insert(
                base.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
        }
        reg.fields.insert(
            "pyre_interpreter::pyopcode::StepResult::Continue".to_string(),
            vec![],
        );
        bk.set_struct_fields(Rc::new(reg));
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let variant_host = bk.intern_enum_variant_host(
            "pyre_interpreter::pyopcode::StepResult<*mut PyObject>",
            "Continue",
        );
        let variant = bk
            .getuniqueclassdef(&variant_host)
            .expect("generic Continue registers");
        let short = bk
            .getuniqueclassdef_for_struct_root("pyopcode::StepResult")
            .expect("unsuffixed StepResult registers");
        assert!(
            variant.borrow().issubclass(&short),
            "generic Continue must subclass the registered pyopcode::StepResult"
        );
        let dotted = bk.intern_class_by_qualname(
            "pyre_interpreter.pyopcode.StepResult<*mut PyObject>.Continue",
        );
        let dotted_cd = bk
            .getuniqueclassdef(&dotted)
            .expect("dotted Continue registers");
        assert!(
            dotted_cd.borrow().issubclass(&short),
            "dotted ctor spelling must subclass pyopcode::StepResult"
        );
        let common = crate::annotator::classdesc::ClassDef::commonbase(&variant, &short)
            .expect("Continue and pyopcode::StepResult share a base");
        assert!(
            Rc::ptr_eq(&common, &short),
            "common base must be the registered enum class"
        );
        let s_variant = SomeValue::Instance(SomeInstance::new(
            Some(variant),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_short = SomeValue::Instance(SomeInstance::new(
            Some(short),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert!(
            crate::annotator::model::union(&s_variant, &s_short).is_ok(),
            "setattr of Continue onto the unsuffixed enum must union"
        );
    }

    #[test]
    fn generic_enum_instantiations_keep_distinct_payload_classdefs() {
        // Two Rust instantiations of one generic enum are two classes
        // (`bookkeeper.py` `getuniqueclassdef(cls)`). Their `Some.__pos_0`
        // attrs must not share one Attribute: a shared ClassDef unions
        // `r_uint` with `int`.
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "option::Option".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        for (root, payload) in [
            ("option::Option<usize>", "usize"),
            ("option::Option<i64>", "i64"),
        ] {
            reg.fields.insert(
                root.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
            reg.fields.insert(
                format!("{root}::Some"),
                vec![("__pos_0".to_string(), payload.to_string())],
            );
        }
        bk.set_struct_fields(Rc::new(reg));

        let some_usize = bk
            .getuniqueclassdef_for_enum_variant("option::Option<usize>", "Some")
            .expect("Option<usize>::Some");
        let some_i64 = bk
            .getuniqueclassdef_for_enum_variant("option::Option<i64>", "Some")
            .expect("Option<i64>::Some");
        assert!(
            !Rc::ptr_eq(&some_usize, &some_i64),
            "Option<usize>::Some and Option<i64>::Some must be distinct ClassDefs, got {}",
            some_usize.borrow().name
        );
        assert_ne!(some_usize.borrow().name, some_i64.borrow().name);
        assert!(
            some_usize.borrow().name.contains("<usize>"),
            "usize instantiation must keep its suffix, got {}",
            some_usize.borrow().name
        );
        assert!(
            some_i64.borrow().name.contains("<i64>"),
            "i64 instantiation must keep its suffix, got {}",
            some_i64.borrow().name
        );

        let unsigned = SomeValue::Integer(SomeInteger::new(true, true));
        let signed = SomeValue::Integer(SomeInteger::new(true, false));
        ClassDef::generalize_attr(&some_usize, "__pos_0", Some(unsigned.clone()))
            .expect("usize Some payload");
        ClassDef::generalize_attr(&some_i64, "__pos_0", Some(signed.clone()))
            .expect("i64 Some payload");
        match &some_usize.borrow().attrs["__pos_0"].s_value {
            SomeValue::Integer(si) => assert!(si.unsigned, "usize payload must stay r_uint"),
            other => panic!("expected unsigned Integer, got {other:?}"),
        }
        match &some_i64.borrow().attrs["__pos_0"].s_value {
            SomeValue::Integer(si) => assert!(!si.unsigned, "i64 payload must stay signed"),
            other => panic!("expected signed Integer, got {other:?}"),
        }
    }

    #[test]
    fn intern_enum_variant_host_keeps_suffix_after_template_some() {
        // A template `Option::Some` interned first must not steal the
        // `Option<usize>::Some` key (`bookkeeper.py` `getuniqueclassdef(cls)`).
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "option::Option".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "option::Option::Some".to_string(),
            vec![("__pos_0".to_string(), "??TypeVar".to_string())],
        );
        reg.fields.insert(
            "option::Option<usize>".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "option::Option<usize>::Some".to_string(),
            vec![("__pos_0".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));
        let template = bk.intern_enum_variant_host("Option", "Some");
        let suffixed = bk.intern_enum_variant_host("core::option::Option<usize>", "Some");
        assert_ne!(
            template,
            suffixed,
            "template Some must not be the usize instantiation, template={} suffixed={}",
            template.qualname(),
            suffixed.qualname()
        );
        assert!(
            suffixed.qualname().contains("<usize>"),
            "ctor intern of Option<usize>::Some must keep the suffix, got {}",
            suffixed.qualname()
        );
        let cd = bk
            .getuniqueclassdef_for_enum_variant("core::option::Option<usize>", "Some")
            .expect("Option<usize>::Some classdef");
        assert!(
            cd.borrow().name.contains("<usize>"),
            "ClassDef name must keep the suffix, got {}",
            cd.borrow().name
        );
    }

    #[test]
    fn unit_variant_prebuilt_continue_unions_with_stepresult_field() {
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::flowspace::model::ConstValue;
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyopcode::StepResult".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "pyopcode::StepResult<*mut PyObject>::Continue".to_string(),
            vec![],
        );
        bk.set_struct_fields(Rc::new(reg));
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let instance = bk
            .intern_unit_variant_prebuilt_instance(
                "pyre_interpreter.pyopcode.StepResult<*mut PyObject>.Continue",
                Some(0),
            )
            .expect("prebuilt Continue");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(instance))
            .expect("prebuilt Continue annotates");
        let short = bk
            .getuniqueclassdef_for_struct_root("pyopcode::StepResult")
            .expect("field class");
        let SomeValue::Instance(si) = s else {
            panic!("expected SomeInstance, got {s:?}");
        };
        let variant = si.classdef.expect("classdef");
        assert!(
            variant.borrow().issubclass(&short),
            "folded Continue must subclass pyopcode::StepResult"
        );
        let s_variant = SomeValue::Instance(SomeInstance::new(
            Some(variant),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_short = SomeValue::Instance(SomeInstance::new(
            Some(short),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert!(
            crate::annotator::model::union(&s_variant, &s_short).is_ok(),
            "setattr of folded Continue onto the enum field must union"
        );
    }

    #[test]
    fn enum_variant_subclasses_base_even_when_struct_root_preregistered_first() {
        // The session prologue (`CallRegistry::ensure_session`) pre-mints
        // EVERY `struct_fields` key through `getuniqueclassdef_for_struct_root`
        // — including the variant keys — BEFORE any narrowing's
        // `getuniqueclassdef_for_enum_variant` runs.  Since
        // `intern_class_by_qualname` is first-mint-wins, the variant must
        // acquire its enum base on that first mint, not lose it.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Prologue order: pre-register every struct root, variant keys first.
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let variant = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant registers");
        assert!(
            variant.borrow().issubclass(&base),
            "variant must subclass the base even when pre-registered as a struct root first"
        );
    }

    /// `Code` in `lookup_fmt` is a mixed enum: ten fieldless variants plus
    /// one payload variant (`Int { signed }`).  `front/mir.rs` dual-publishes
    /// the discriminant-only base and registers a payload-less variant with
    /// empty rows (a lone `__discriminant` row is the `is_enum_base`
    /// sentinel).  The session prologue first-mints every registry key
    /// through `intern_class_by_qualname`; if that walk skips an empty-row
    /// `{enum}::{variant}` key, the fieldless variant is frozen base-less,
    /// unioning it with `Code::Int` finds no common base, and the
    /// `__discriminant` getattr on `code` lands on the root instance.
    #[test]
    fn payload_less_enum_variant_subclasses_base_and_unions_to_discriminant_root() {
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Dual-publish like `front/mir.rs` TypeDeclKind::Enum.
        for base in ["Code", "module::struct::Code"] {
            reg.fields.insert(
                base.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
        }
        for variant_key in [
            "Code::Int",
            "module::struct::Code::Int",
            "Code::FloatComplex",
            "module::struct::Code::FloatComplex",
            "Code::DoubleComplex",
            "module::struct::Code::DoubleComplex",
        ] {
            let rows = if variant_key.ends_with("::Int") {
                vec![("signed".to_string(), "bool".to_string())]
            } else {
                // Payload-less: empty rows, not a `__discriminant` sentinel.
                vec![]
            };
            reg.fields.insert(variant_key.to_string(), rows);
        }
        bk.set_struct_fields(Rc::new(reg));

        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "FloatComplex".to_string());
        by_discr.insert(1, "DoubleComplex".to_string());
        by_discr.insert(2, "Int".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("module::struct::Code".to_string(), by_discr.clone());
        map.insert("Code".to_string(), by_discr);
        bk.set_enum_variant_by_discriminant(Rc::new(map));

        // Prologue order: every registry key, then the enum-variant pre-mint.
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        bk.pre_register_enum_variant_classes();

        let base = bk
            .getuniqueclassdef_for_struct_root("Code")
            .expect("Code base registers");
        assert!(
            base.borrow().attrs.contains_key("__discriminant"),
            "enum base must carry the synthetic __discriminant row"
        );
        assert!(
            base.borrow().basedef.is_none(),
            "enum base is the discriminant-only root, not a further subclass"
        );

        let int_v = bk
            .getuniqueclassdef_for_enum_variant("Code", "Int")
            .expect("Code::Int registers");
        let float_v = bk
            .getuniqueclassdef_for_enum_variant("Code", "FloatComplex")
            .expect("Code::FloatComplex registers");
        let double_v = bk
            .getuniqueclassdef_for_enum_variant("Code", "DoubleComplex")
            .expect("Code::DoubleComplex registers");

        let interned_float = bk.intern_class_by_qualname("Code::FloatComplex");
        let interned_float_cd = bk
            .getuniqueclassdef(&interned_float)
            .expect("interned FloatComplex classdef");
        assert!(
            Rc::ptr_eq(&interned_float_cd, &float_v),
            "intern_class_by_qualname of the empty-row key must be the variant classdef"
        );

        assert!(
            int_v.borrow().issubclass(&base),
            "payload variant Code::Int must subclass Code"
        );
        assert!(
            float_v.borrow().issubclass(&base),
            "payload-less variant Code::FloatComplex must subclass Code \
             (empty rows are not a reason to mint it base-less)"
        );
        assert!(
            double_v.borrow().issubclass(&base),
            "payload-less variant Code::DoubleComplex must subclass Code"
        );

        let s_int = SomeValue::Instance(SomeInstance::new(
            Some(int_v),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_float = SomeValue::Instance(SomeInstance::new(
            Some(float_v),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_double = SomeValue::Instance(SomeInstance::new(
            Some(double_v),
            false,
            std::collections::BTreeMap::new(),
        ));
        let u = union(&s_float, &s_int).unwrap_or_else(|err| {
            panic!("union(FloatComplex, Int) must share the Code base, got {err}")
        });
        let u = union(&u, &s_double).unwrap_or_else(|err| {
            panic!("union(Code, DoubleComplex) must stay on the Code base, got {err}")
        });
        let SomeValue::Instance(inst) = u else {
            panic!("union of Code variants must stay an instance, got {u:?}");
        };
        assert!(
            inst.classdef.as_ref().is_some_and(|c| Rc::ptr_eq(c, &base)),
            "union of payload-less and payload Code variants must land on the \
             Code enum base (classdef={:?}), not the root instance",
            inst.classdef.as_ref().map(|c| c.borrow().name.clone())
        );
        assert!(!inst.can_be_none, "Code is a non-nullable sum type");
    }

    /// Harden withdraws the bare `Code` alias when `eval::Code` shares the
    /// leaf. The prologue must still mint `module::struct::Code`'s variant
    /// subclasses from the qualified `__discriminant` row.
    #[test]
    fn withdrawn_bare_enum_leaf_still_pre_registers_the_qualified_base() {
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let base = "pyre_interpreter::module::struct::Code";
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            base.to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(format!("{base}::Pad"), vec![]);
        reg.fields.insert(
            format!("{base}::Int"),
            vec![("signed".to_string(), "bool".to_string())],
        );
        reg.fields.insert(
            "pyre_interpreter::eval::Code".to_string(),
            vec![("name".to_string(), "String".to_string())],
        );
        assert!(
            !reg.is_enum_base("Code"),
            "the bare leaf is withdrawn; the qualified path is the enum base"
        );
        bk.set_struct_fields(Rc::new(reg));

        let mut by_discr = HashMap::new();
        by_discr.insert(0, "Pad".to_string());
        by_discr.insert(1, "Int".to_string());
        let mut map = HashMap::new();
        map.insert(base.to_string(), by_discr);
        bk.set_enum_variant_by_discriminant(Rc::new(map));

        assert!(
            bk.classdef_snapshot().is_empty(),
            "the registry alone does not mint classdefs"
        );
        bk.pre_register_enum_variant_classes();
        let names: Vec<String> = bk
            .classdef_snapshot()
            .iter()
            .map(|classdef| classdef.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|name| name.ends_with("Code::Pad")),
            "qualified Code variants must be pre-minted, got {names:?}"
        );
        assert!(
            names.iter().any(|name| name.ends_with("Code::Int")),
            "the payload variant must be pre-minted with the fieldless one, got {names:?}"
        );
        let base_cd = bk
            .getuniqueclassdef_for_struct_root(base)
            .expect("qualified Code base");
        let pad = bk
            .getuniqueclassdef_for_enum_variant(base, "Pad")
            .expect("Pad");
        assert!(
            pad.borrow().issubclass(&base_cd),
            "Pad must subclass the discriminant-only Code base"
        );
        assert!(base_cd.borrow().attrs.contains_key("__discriminant"));
    }

    #[test]
    fn enum_variant_ctor_helper_and_narrowing_resolve_one_classdef_with_payload_attr() {
        // The variant CONSTRUCTOR path (`flowspace_adapter`, via
        // `intern_enum_variant_host`) and the discriminant-NARROWING path
        // (`getuniqueclassdef_for_enum_variant`) must resolve the SAME
        // class object — RPython's single-class-per-variant identity
        // (`ClassDesc.getuniqueclassdef`, `classdesc.py`) — and it
        // must carry the variant's payload
        // attr.  Before the unification the ctor minted a dotted-qualname
        // sibling classdef with no payload attrs.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Canonical `module::Enum` + `module::Enum::Variant` spellings —
        // already `::`-qualified, so `canonical_struct_name` passes them
        // through and the test never touches the process-global
        // `STRUCT_ORIGIN_REGISTRY`.
        reg.fields.insert(
            "shapes::Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "shapes::Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Constructor side: the adapter interns through this helper.
        let ctor_host = bk.intern_enum_variant_host("shapes::Color", "Rgb");
        let ctor_classdef = bk
            .getuniqueclassdef(&ctor_host)
            .expect("ctor variant classdef");
        // Narrowing side: the public API the discriminant read uses.
        let narrow_classdef = bk
            .getuniqueclassdef_for_enum_variant("shapes::Color", "Rgb")
            .expect("narrowing variant classdef");

        assert!(
            Rc::ptr_eq(&ctor_classdef, &narrow_classdef),
            "constructor and narrowing must resolve ONE variant class object"
        );
        assert!(
            narrow_classdef.borrow().attrs.contains_key("rgb"),
            "the unified variant class must carry its payload attr"
        );
        // And it subclasses the discriminant-only base.
        let base = bk
            .getuniqueclassdef_for_struct_root("shapes::Color")
            .expect("base registers");
        assert!(
            narrow_classdef.borrow().issubclass(&base),
            "variant subclasses the enum base"
        );
    }

    #[test]
    fn enum_variant_ctor_bare_leaf_canonicalises_to_one_classdef_with_narrowing() {
        // The production ctor side carries the enum's *bare leaf* as its
        // owner tail (`flowspace_adapter`), while the narrowing side reads
        // the base classdef's *canonical* `module::Enum` name
        // (`unaryop.rs`).  They unify only because `front::mir` registers
        // the leaf → module origin so `canonical_struct_name(leaf)` lands
        // on the same `module::Enum` spelling.  Without that origin the
        // bare leaf passes through and the two sides mint sibling
        // classdefs.
        //
        // The unique leaf below does NOT make the write below safe against a
        // concurrent test, and an earlier version of this comment claimed it
        // did. A unique KEY protects a concurrent reader looking up a
        // DIFFERENT key; `register_struct_origins` is `*guard = origins`, so it
        // discards the whole table, and what it destroys is whatever some other
        // test registered — a direction key uniqueness cannot reach.
        //
        // What actually holds today is narrower and needs re-checking before it
        // is relied on: this is the ONLY `#[test]` in the `majit-translate`
        // lib-test binary that writes `STRUCT_ORIGIN_REGISTRY`, so there is no
        // second writer to race. Adding one re-opens the hazard.
        //
        // Not hypothetical: the sibling registry `STRUCT_ID_BY_NAME` gained a
        // second lib-binary `#[test]` writer and the pair then failed 9 runs in
        // 12, while passing in isolation and under `--test-threads=1`. The
        // remedy used there is `register_struct_ids_serialized`
        // (`codewriter/assembler.rs`, in its `mod tests`); mirror it here
        // rather than reasoning about key collisions.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "probemod::Newt".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "probemod::Newt::Eft".to_string(),
            vec![("eft".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));
        // The origin `front::mir` now registers for an enum leaf.
        let _registry = crate::test_support::register_struct_origins_serialized(
            [("Newt".to_string(), "probemod".to_string())].into(),
        );

        // Ctor side interns through `intern_enum_variant_host` with the
        // bare leaf as owner tail.
        let ctor_host = bk.intern_enum_variant_host("Newt", "Eft");
        let ctor_classdef = bk
            .getuniqueclassdef(&ctor_host)
            .expect("ctor variant classdef");
        // Narrowing side passes the base classdef's canonical name.
        let narrow_classdef = bk
            .getuniqueclassdef_for_enum_variant("probemod::Newt", "Eft")
            .expect("narrowing variant classdef");

        assert!(
            Rc::ptr_eq(&ctor_classdef, &narrow_classdef),
            "bare-leaf ctor and canonical narrowing must resolve ONE class object"
        );
        assert!(
            narrow_classdef.borrow().attrs.contains_key("eft"),
            "the unified variant class must carry its payload attr"
        );
    }

    #[test]
    fn enum_variant_narrowing_knowntypedata_keys_int_cases_to_variant_subclasses() {
        use crate::annotator::model::{ExitCaseKey, SomeInstance, SomeValue};
        use crate::flowspace::model::Variable;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Color".to_string(),
            vec![
                ("__discriminant".to_string(), "i64".to_string()),
                ("rgb".to_string(), "i64".to_string()),
            ],
        );
        bk.set_struct_fields(Rc::new(reg));

        // discriminant 0 -> Rgb, 1 -> Named.
        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "Rgb".to_string());
        by_discr.insert(1, "Named".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("Color".to_string(), by_discr);
        bk.set_enum_variant_by_discriminant(Rc::new(map));

        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let receiver = Rc::new(Variable::new());
        let ktd = bk
            .enum_variant_narrowing_knowntypedata("Color", &receiver)
            .expect("variant classdefs build")
            .expect("enum root has a variant table");

        // One knowntypedata case per variant, keyed by the integer tag.
        assert_eq!(ktd.len(), 2, "one knowntypedata case per variant");
        for (discr, variant_name) in [(0, "Rgb"), (1, "Named")] {
            let constraints = ktd
                .get(&ExitCaseKey::Int(discr))
                .unwrap_or_else(|| panic!("missing Int({discr}) case"));
            let s_refined = constraints
                .get(&receiver)
                .expect("receiver variable is the narrowing key");
            let variant = bk
                .getuniqueclassdef_for_enum_variant("Color", variant_name)
                .expect("variant registers");
            // The refinement narrows SomeInstance(base) to the variant —
            // exactly what follow_link's improve does in arm `discr`.
            let s_base = SomeValue::Instance(SomeInstance::new(
                Some(base.clone()),
                false,
                std::collections::BTreeMap::new(),
            ));
            let narrowed = crate::annotator::binaryop::improve(&s_base, s_refined);
            match narrowed {
                SomeValue::Instance(si) => assert!(
                    si.classdef
                        .as_ref()
                        .is_some_and(|c| Rc::ptr_eq(c, &variant)),
                    "Int({discr}) narrows the receiver to {variant_name}"
                ),
                other => panic!("expected SomeInstance({variant_name}), got {other:?}"),
            }
        }

        // A class with no registered variant table yields no narrowing.
        let other_recv = Rc::new(Variable::new());
        assert!(
            bk.enum_variant_narrowing_knowntypedata("NotAnEnum", &other_recv)
                .expect("unregistered class is not an annotator error")
                .is_none(),
            "non-enum class yields no knowntypedata"
        );
    }

    #[test]
    fn enum_variant_narrowing_is_idempotent_on_already_narrowed_receiver() {
        // A `__discriminant` re-read on a receiver already narrowed to a
        // variant subclass reaches the narrowing with `enum_root` carrying
        // the `::variant` tail (`Opt<i64>::A`).  It must re-derive the base
        // enum's variant subclasses (idempotent), NOT append another variant
        // (`Opt<i64>::A::A`) — otherwise the per-instantiation classdef
        // deepens every annotation pass and never reaches a fixpoint.  The
        // generic `<…>` instantiation is required to reproduce: the table
        // lookup strips it (`strip_instantiation_suffix`), so the narrowed
        // root resolves the same variant table while the variant-build sees
        // the full narrowed name.
        use crate::annotator::model::{ExitCaseKey, SomeValue};
        use crate::flowspace::model::Variable;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Opt<i64>".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Table keyed by the bare template root (`strip_instantiation_suffix`
        // reduces both `Opt<i64>` and `Opt<i64>::A` to `Opt`).
        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "A".to_string());
        by_discr.insert(1, "B".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("Opt".to_string(), by_discr);
        bk.set_enum_variant_by_discriminant(Rc::new(map));

        let receiver = Rc::new(Variable::new());
        let from_base = bk
            .enum_variant_narrowing_knowntypedata("Opt<i64>", &receiver)
            .expect("variant classdefs build")
            .expect("base enum resolves the variant table");
        let from_variant = bk
            .enum_variant_narrowing_knowntypedata("Opt<i64>::A", &receiver)
            .expect("variant classdefs build")
            .expect("an already-narrowed receiver still resolves the base table");

        let classdef_of = |ktd: &crate::annotator::model::KnownTypeData, discr: i64| match ktd
            .get(&ExitCaseKey::Int(discr))
            .and_then(|c| c.get(&receiver))
            .expect("case present")
        {
            SomeValue::Instance(si) => si.classdef.clone().expect("variant carries a classdef"),
            other => panic!("expected SomeInstance, got {other:?}"),
        };
        for discr in [0i64, 1i64] {
            assert!(
                Rc::ptr_eq(
                    &classdef_of(&from_base, discr),
                    &classdef_of(&from_variant, discr)
                ),
                "Int({discr}): narrowing an already-narrowed receiver must reuse \
                 the base variant classdef, not deepen it (::A::A)"
            );
        }
    }

    #[test]
    fn enum_variant_narrowing_resolves_module_qualified_variant_classdef() {
        // `importing::GcdCache::Miss` has no `<…>` for
        // `strip_instantiation_suffix` to cut, and the table is published
        // under the bare leaf `GcdCache` plus the crate-qualified root —
        // not under `Miss` or `importing::GcdCache`.  A `__discriminant`
        // read on the already-narrowed variant classdef must attach the
        // same knowntypedata as a read on the base.
        use crate::annotator::model::{ExitCaseKey, SomeValue};
        use crate::flowspace::model::Variable;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        for name in [
            "GcdCache",
            "importing::GcdCache",
            "pyre_interpreter::importing::GcdCache",
        ] {
            reg.fields.insert(
                name.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
        }
        bk.set_struct_fields(Rc::new(reg));

        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "Miss".to_string());
        by_discr.insert(1, "Ready".to_string());
        by_discr.insert(2, "Initializing".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("GcdCache".to_string(), by_discr.clone());
        map.insert(
            "pyre_interpreter::importing::GcdCache".to_string(),
            by_discr,
        );
        bk.set_enum_variant_by_discriminant(Rc::new(map));

        let receiver = Rc::new(Variable::new());
        let from_base = bk
            .enum_variant_narrowing_knowntypedata("importing::GcdCache", &receiver)
            .expect("variant classdefs build")
            .expect("module-qualified base resolves via the bare leaf");
        let from_variant = bk
            .enum_variant_narrowing_knowntypedata("importing::GcdCache::Miss", &receiver)
            .expect("variant classdefs build")
            .expect("variant classdef resolves the same enum table");
        assert_eq!(from_base.len(), 3);
        assert_eq!(from_variant.len(), 3);
        for discr in [0i64, 1, 2] {
            let classdef_of = |ktd: &crate::annotator::model::KnownTypeData| match ktd
                .get(&ExitCaseKey::Int(discr))
                .and_then(|c| c.get(&receiver))
            {
                Some(SomeValue::Instance(si)) => si.classdef.clone().expect("classdef"),
                other => panic!("Int({discr}) expected SomeInstance, got {other:?}"),
            };
            assert!(
                Rc::ptr_eq(&classdef_of(&from_base), &classdef_of(&from_variant)),
                "Int({discr}): variant-qualified receiver must reuse the base variant classdef"
            );
        }
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_terminates_on_cyclic_graph() {
        // Localizes the route-b stack overflow: the seed fired
        // for `FixedObjectArray` (fields `len: usize`, `_items:
        // [PyObjectRef; 0]`) and the build aborted with a stack overflow.
        // Reproduce the real shape — an array struct referencing the
        // boxed-object root, which cycles back through the type object —
        // and assert transitive registration TERMINATES (returns a
        // populated classdef) rather than recursing forever.  If this test
        // overflows, the overflow is IN transitive registration; if it
        // passes, the overflow is downstream (rtyper repr setup).
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "FixedObjectArray".to_string(),
            vec![
                ("len".to_string(), "usize".to_string()),
                ("_items".to_string(), "[PyObjectRef; 0]".to_string()),
            ],
        );
        reg.fields.insert(
            "PyObjectRef".to_string(),
            vec![("typ".to_string(), "W_TypeObject".to_string())],
        );
        reg.fields.insert(
            "W_TypeObject".to_string(),
            vec![
                // cycle: W_TypeObject -> PyObjectRef -> W_TypeObject
                ("base".to_string(), "PyObjectRef".to_string()),
                // self-reference: W_TypeObject -> W_TypeObject
                ("mro".to_string(), "Vec<W_TypeObject>".to_string()),
            ],
        );
        bk.set_struct_fields(Rc::new(reg));

        let cd = bk
            .getuniqueclassdef_for_struct_root("FixedObjectArray")
            .expect("cyclic graph registers without overflow");
        // `_items` projects to a SomeList of the boxed-object instance.
        assert!(
            cd.borrow().attrs.contains_key("_items"),
            "array field projected"
        );
        // The cyclic inner structs are registered exactly once each.
        let ty = bk
            .getuniqueclassdef_for_struct_root("W_TypeObject")
            .expect("W_TypeObject registers");
        let ty2 = bk
            .getuniqueclassdef_for_struct_root("W_TypeObject")
            .expect("W_TypeObject re-lookup");
        assert!(
            Rc::ptr_eq(&ty, &ty2),
            "cycle node interned to one identity-keyed ClassDef"
        );
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_handles_deep_plus_cyclic_chain() {
        // Deep-chain regression gate for the iterative worklist.
        // The 2026-05-29 route-b measurement overflowed the native stack in
        // the RECURSIVE transitive walk of `FixedObjectArray`.  The walk now
        // drives an explicit `Vec` worklist (no native recursion over the
        // struct-field graph), so a chain far deeper than any native stack
        // can hold must register in O(reachable) heap without overflowing.
        // A back-edge from the tail to the head folds a cycle into the same
        // fixture, pinning depth and cycle handling together: if this test
        // ever overflows, the transitive registration regressed back to a
        // recursive walk.  `DEPTH` is chosen well past the frame budget a
        // recursive walk would exhaust on the default 8 MiB main stack.
        use crate::front::StructFieldRegistry;
        const DEPTH: usize = 2000;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        for i in 0..DEPTH {
            // `Si { next: S{i+1} }` — each link references the next struct.
            reg.fields.insert(
                format!("S{i}"),
                vec![("next".to_string(), format!("S{}", i + 1))],
            );
        }
        // Terminal node: a scalar leaf plus a back-edge to the head, so the
        // reachable graph is both `DEPTH`-deep and cyclic.
        reg.fields.insert(
            format!("S{DEPTH}"),
            vec![
                ("leaf".to_string(), "i64".to_string()),
                ("back".to_string(), "S0".to_string()),
            ],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Registering from the head walks the full depth + the back-edge
        // cycle without overflowing.
        let head = bk
            .getuniqueclassdef_for_struct_root("S0")
            .expect("deep + cyclic graph registers without stack overflow");
        assert!(
            matches!(
                head.borrow().attrs.get("next").map(|a| &a.s_value),
                Some(SomeValue::Instance(_))
            ),
            "head's struct-typed field projects to SomeInstance"
        );

        // A node midway down the chain is registered with its field
        // projected — the worklist reached the full depth (a recursive walk
        // would have overflowed long before here).
        let mid = format!("S{}", DEPTH / 2);
        let mid_cd = bk
            .getuniqueclassdef_for_struct_root(&mid)
            .unwrap_or_else(|e| panic!("deep node {mid} registers: {e}"));
        assert!(
            mid_cd.borrow().attrs.contains_key("next"),
            "deep node's field is projected"
        );
        let mid_cd2 = bk
            .getuniqueclassdef_for_struct_root(&mid)
            .expect("deep node re-lookup");
        assert!(
            Rc::ptr_eq(&mid_cd, &mid_cd2),
            "deep node interned to one identity-keyed ClassDef"
        );

        // The scalar leaf at the tail projects as an integer — confirming
        // the worklist reached the terminal node past the cycle.
        let tail = bk
            .getuniqueclassdef_for_struct_root(&format!("S{DEPTH}"))
            .expect("tail registers");
        assert!(
            matches!(
                tail.borrow().attrs.get("leaf").map(|a| &a.s_value),
                Some(SomeValue::Integer(_))
            ),
            "tail scalar leaf projects to SomeInteger"
        );
    }

    #[test]
    fn a_raw_vec_pointer_field_projects_to_somerustvec() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "W_TypeObject".to_string(),
            vec![(
                "weak_subclasses".to_string(),
                "*mut Vec<*mut Weakref>".to_string(),
            )],
        );
        bk.set_struct_fields(Rc::new(reg));
        let cd = bk
            .getuniqueclassdef_for_struct_root("W_TypeObject")
            .expect("W_TypeObject registers");
        let attr = cd
            .borrow()
            .attrs
            .get("weak_subclasses")
            .expect("weak_subclasses attr projected")
            .s_value
            .clone();
        assert!(
            matches!(attr, SomeValue::RustVec(_)),
            "a *mut Vec of one-word items is SomeRustVec, got {attr:?}"
        );
    }

    #[test]
    fn named_raw_struct_field_projects_to_someptr() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, PtrTarget};

        let inner = "rawproj::PosixMap";
        let parent = "rawproj::Mapped";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("ptr".into(), "*mut u8".into()),
                ("len".into(), "usize".into()),
            ],
        );
        reg.fields.insert(
            parent.to_string(),
            vec![("mapped".into(), inner.to_string())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
            crate::call::StructLayout {
                size: 16,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![
                    crate::call::StructFieldLayout {
                        name: "ptr".into(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    crate::call::StructFieldLayout {
                        name: "len".into(),
                        offset: 8,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Unsigned,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        let projected = bk.project_struct_field_type(inner);
        let SomeValue::Ptr(ptr) = projected else {
            panic!("raw nested struct must project to SomePtr, got {projected:?}");
        };
        let PtrTarget::Struct(st) = &ptr.ll_ptrtype.TO else {
            panic!("SomePtr target must be a Struct");
        };
        assert_eq!(st._gckind, GcKind::Raw);

        let parent_cd = bk
            .getuniqueclassdef_for_struct_root(parent)
            .expect("parent registers");
        let mapped = parent_cd
            .borrow()
            .attrs
            .get("mapped")
            .expect("mapped attr projected")
            .s_value
            .clone();
        assert!(
            matches!(mapped, SomeValue::Ptr(_)),
            "parent.mapped must be SomePtr, got {mapped:?}"
        );
    }

    #[test]
    fn option_some_payload_of_pointer_raw_struct_stays_ptr() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let inner = "rawproj::PosixMap";
        let some = "Option<rawproj::PosixMap>::Some";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("ptr".into(), "*mut u8".into()),
                ("len".into(), "usize".into()),
            ],
        );
        reg.fields.insert(
            some.to_string(),
            vec![("__pos_0".into(), inner.to_string())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
            crate::call::StructLayout {
                size: 16,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![
                    crate::call::StructFieldLayout {
                        name: "ptr".into(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    crate::call::StructFieldLayout {
                        name: "len".into(),
                        offset: 8,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Unsigned,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        assert!(
            matches!(bk.project_struct_field_type(inner), SomeValue::Ptr(_)),
            "by-value raw field type stays SomePtr"
        );

        let some_cd = bk
            .getuniqueclassdef_for_struct_root(some)
            .expect("Option::Some registers");
        let payload = some_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("payload attr projected")
            .s_value
            .clone();
        assert!(
            matches!(payload, SomeValue::Ptr(_)),
            "Option::Some.__pos_0 of a pointer-carrying raw struct stays SomePtr, got {payload:?}"
        );
    }

    #[test]
    fn option_some_payload_of_scalar_raw_struct_stays_instance() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let inner = "rawproj::Fmt";
        let some = "Option<rawproj::Fmt>::Some";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("size".into(), "usize".into()),
                ("needcount".into(), "bool".into()),
            ],
        );
        reg.fields.insert(
            some.to_string(),
            vec![("__pos_0".into(), inner.to_string())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
            crate::call::StructLayout {
                size: 16,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![
                    crate::call::StructFieldLayout {
                        name: "size".into(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Unsigned,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    crate::call::StructFieldLayout {
                        name: "needcount".into(),
                        offset: 8,
                        size: 1,
                        flag: majit_ir::descr::ArrayFlag::Unsigned,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        assert!(
            matches!(bk.project_struct_field_type(inner), SomeValue::Ptr(_)),
            "by-value scalar raw struct still projects SomePtr as a named field"
        );

        let some_cd = bk
            .getuniqueclassdef_for_struct_root(some)
            .expect("Option::Some registers");
        let payload = some_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("payload attr projected")
            .s_value
            .clone();
        assert!(
            matches!(payload, SomeValue::Instance(_)),
            "Option::Some.__pos_0 of a classed scalar ADT stays SomeInstance so instantiate can setattr it, got {payload:?}"
        );
    }

    #[test]
    fn option_some_payload_of_payload_enum_stays_instance() {
        // `BuiltinEncoder`: the base is `__discriminant` only, the
        // `Utf16Or32` variant carries `Utf16Or32Form`. A named-field
        // projection is still SomePtr (`raw_struct_ptr_annotation`);
        // Option::Some.__pos_0 stays SomeInstance so instantiate can
        // setattr a `Utf8` variant (`llannotation.py`
        // pairtype(SomePtr, SomeObject).union).
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let inner = "type_methods::BuiltinEncoder";
        let variant = "type_methods::BuiltinEncoder::Utf16Or32";
        let form = "type_methods::Utf16Or32Form";
        let some = "Option<type_methods::BuiltinEncoder>::Some";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![("__discriminant".into(), "u8".into())],
        );
        reg.fields
            .insert(variant.to_string(), vec![("__pos_0".into(), form.into())]);
        reg.fields.insert(
            form.to_string(),
            vec![
                ("is32".into(), "bool".into()),
                ("order".into(), "ByteOrder".into()),
                ("bom".into(), "bool".into()),
            ],
        );
        reg.fields
            .insert(some.to_string(), vec![("__pos_0".into(), inner.into())]);
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
            crate::call::StructLayout {
                size: 8,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![crate::call::StructFieldLayout {
                    name: "__discriminant".into(),
                    offset: 0,
                    size: 1,
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

        assert!(
            matches!(bk.project_struct_field_type(inner), SomeValue::Ptr(_)),
            "payload-enum named field still projects SomePtr"
        );

        let some_cd = bk
            .getuniqueclassdef_for_struct_root(some)
            .expect("Option::Some registers");
        let payload = some_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("payload attr projected")
            .s_value
            .clone();
        assert!(
            matches!(payload, SomeValue::Instance(_)),
            "Option::Some.__pos_0 of a payload enum stays SomeInstance so instantiate can setattr a fieldless variant, got {payload:?}"
        );
    }

    #[test]
    fn option_some_payload_of_rootscope_stays_instance() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let inner = "rawproj::RootScope";
        let some = "Option<rawproj::RootScope>::Some";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("save_point".into(), "usize".into()),
                ("_not_send".into(), "PhantomData<*const ()>".into()),
            ],
        );
        reg.fields.insert(
            some.to_string(),
            vec![("__pos_0".into(), inner.to_string())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
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

        let some_cd = bk
            .getuniqueclassdef_for_struct_root(some)
            .expect("Option::Some registers");
        let payload = some_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("payload attr projected")
            .s_value
            .clone();
        assert!(
            matches!(payload, SomeValue::Instance(_)),
            "PhantomData<*const ()> is not a stored pointer; RootScope payload stays SomeInstance, got {payload:?}"
        );
    }

    #[test]
    fn closure_capture_of_raw_struct_stays_ptr() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let inner = "rawproj::RootScope";
        let owner = "eval::<Impl>::load_attr::closure";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
                (owner.to_string(), Some(owner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("save_point".into(), "usize".into()),
                ("_not_send".into(), "PhantomData<*const ()>".into()),
            ],
        );
        reg.fields.insert(
            owner.to_string(),
            vec![("__pos_0".into(), inner.to_string())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
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

        let cd = bk
            .getuniqueclassdef_for_struct_root(owner)
            .expect("closure env registers");
        let capture = cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("capture attr projected")
            .s_value
            .clone();
        assert!(
            matches!(capture, SomeValue::Ptr(_)),
            "a closure capture of a raw owner stays SomePtr so setattr of the cast value unions, got {capture:?}"
        );
    }

    #[test]
    fn option_some_payload_of_option_raw_ptr_stays_ptr() {
        use crate::front::StructFieldRegistry;
        use crate::front::host_layout::HostLayout;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;
        use majit_charon_reader::ullbc::{TagEncoding, TagLayout};

        let inner = "rawproj::PosixMap";
        let option_root = "Option";
        let some = "Option<Option<rawproj::PosixMap>>::Some";
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let option_id = majit_ir::descr::StructId::from_canonical(option_root);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (inner.to_string(), Some(inner_id)),
                (option_root.to_string(), Some(option_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            inner.to_string(),
            vec![
                ("ptr".into(), "*mut u8".into()),
                ("len".into(), "usize".into()),
            ],
        );
        // Enum-base row: intern would mint Instance(Option) if the payload
        // were promoted. The host layout is the niche-optimized pointer word.
        reg.fields.insert(
            option_root.to_string(),
            vec![("__discriminant".into(), "u8".into())],
        );
        reg.fields.insert(
            some.to_string(),
            vec![("__pos_0".into(), "Option<rawproj::PosixMap>".into())],
        );
        reg.raw_word_owners.insert(inner.to_string());
        bk.set_struct_fields(Rc::new(reg));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            inner_id,
            crate::call::StructLayout {
                size: 16,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![
                    crate::call::StructFieldLayout {
                        name: "ptr".into(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    crate::call::StructFieldLayout {
                        name: "len".into(),
                        offset: 8,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Unsigned,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        let word = crate::layout::target_word_size();
        cc.set_struct_layout(
            option_id,
            crate::call::StructLayout {
                size: word,
                align: word,
                gckind: GcKind::Raw,
                fields: vec![],
                host: Some(HostLayout {
                    size: word as u64,
                    align: word as u64,
                    variant_field_offsets: vec![vec![], vec![0]],
                    tag: Some(TagLayout {
                        offset: 0,
                        signed: false,
                        bits: (word * 8) as u32,
                        encoding: TagEncoding::Niche {
                            untagged_variant: 1,
                            niche_variants: 0..=0,
                            niche_start: 0,
                        },
                    }),
                }),
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        let some_cd = bk
            .getuniqueclassdef_for_struct_root(some)
            .expect("nested Option::Some registers");
        let payload = some_cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("payload attr projected")
            .s_value
            .clone();
        assert!(
            matches!(payload, SomeValue::Ptr(_)),
            "Option::Some.__pos_0 of Option<Raw> stays the nullable Ptr, got {payload:?}"
        );
    }

    #[test]
    fn closure_env_stays_instance_even_when_raw_word_owner() {
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let owner = "descroperation::try_dispatch_ternary_pow_special::closure#2";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields
            .insert(owner.to_string(), vec![("__pos_0".into(), "&usize".into())]);
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
                    name: "__pos_0".into(),
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

        let projected = bk.project_struct_field_type(owner);
        assert!(
            matches!(projected, SomeValue::Instance(_)),
            "a closure env stays SomeInstance so capture getattr peels &usize to r_uint, got {projected:?}"
        );

        let cd = bk
            .getuniqueclassdef_for_struct_root(owner)
            .expect("closure env registers");
        let capture = cd
            .borrow()
            .attrs
            .get("__pos_0")
            .expect("capture attr projected")
            .s_value
            .clone();
        match capture {
            SomeValue::Integer(i) => assert!(
                i.unsigned,
                "captured &usize must project r_uint, got signed"
            ),
            other => panic!("captured &usize must project SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn fieldless_enum_projects_to_signed_integer() {
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "rawproj::StrategyKind".to_string(),
            vec![("__discriminant".to_string(), "u8".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        match bk.project_struct_field_type("rawproj::StrategyKind") {
            SomeValue::Integer(i) => assert!(
                !i.unsigned,
                "fieldless enum must project Signed, got unsigned"
            ),
            other => panic!("fieldless enum must project SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn projected_field_listdefs_are_keyed_by_attribute_position() {
        // Two structs with the same item spelling keep distinct ListDefs
        // (`Attribute.s_value` / `Bookkeeper.getlistdef` by position).
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "OwnerDeque".to_string(),
            vec![("items".to_string(), "VecDeque<PyObjectRef>".to_string())],
        );
        reg.fields.insert(
            "OwnerArray".to_string(),
            vec![("items".to_string(), "[PyObjectRef; 4]".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        let deque_cd = bk
            .getuniqueclassdef_for_struct_root("OwnerDeque")
            .expect("OwnerDeque registers");
        let array_cd = bk
            .getuniqueclassdef_for_struct_root("OwnerArray")
            .expect("OwnerArray registers");

        let deque_list = match &deque_cd.borrow().attrs.get("items").expect("items").s_value {
            SomeValue::List(l) => l.listdef.clone(),
            other => panic!("OwnerDeque.items must be SomeList, got {other:?}"),
        };
        let array_list = match &array_cd.borrow().attrs.get("items").expect("items").s_value {
            SomeValue::List(l) => l.listdef.clone(),
            other => panic!("OwnerArray.items must be SomeList, got {other:?}"),
        };
        assert!(
            !deque_list.same_as(&array_list),
            "fields on different structs must not share a ListDef"
        );

        deque_list.resize().expect("resize OwnerDeque.items");
        assert!(
            deque_list.listitem_rc().borrow().resized,
            "resized flag lives on the deque field's ListDef"
        );
        assert!(
            !array_list.listitem_rc().borrow().resized,
            "resizing one field must leave the other unchanged"
        );

        let deque_again = bk
            .getuniqueclassdef_for_struct_root("OwnerDeque")
            .expect("OwnerDeque re-lookup");
        let deque_list_again = match &deque_again
            .borrow()
            .attrs
            .get("items")
            .expect("items")
            .s_value
        {
            SomeValue::List(l) => l.listdef.clone(),
            other => panic!("OwnerDeque.items must stay SomeList, got {other:?}"),
        };
        assert!(
            deque_list.same_as(&deque_list_again),
            "two reads of the same field must return the same ListDef"
        );
    }

    #[test]
    fn list_spelling_at_a_position_reuses_getlistdef() {
        // `__cast_instance_intrinsic` of `[T]` re-projects the spelling
        // at an op; identity is `Bookkeeper.getlistdef`, not the item type.
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(11, 22, 3)));
        let first = bk.project_list_spelling_at_position("[PyObjectRef]");
        let second = bk.project_list_spelling_at_position("[PyObjectRef]");
        let (SomeValue::List(a), SomeValue::List(b)) = (first, second) else {
            panic!("[PyObjectRef] must project to SomeList");
        };
        assert!(
            a.listdef.same_as(&b.listdef),
            "two projections of one list spelling at one position share a ListDef"
        );
        assert!(
            a.listdef.same_as(&bk.getlistdef(None)),
            "the bound ListDef is Bookkeeper.getlistdef of the current position"
        );

        bk.set_position_key(Some(PositionKey::new(11, 22, 4)));
        let other_pos = bk.project_list_spelling_at_position("[PyObjectRef]");
        let SomeValue::List(c) = other_pos else {
            panic!("[PyObjectRef] must project to SomeList at a second position");
        };
        assert!(
            !a.listdef.same_as(&c.listdef),
            "distinct positions keep distinct ListDefs"
        );
        a.listdef.resize().expect("resize first position");
        assert!(
            !c.listdef.listitem_rc().borrow().resized,
            "resizing one position must leave the other unchanged"
        );
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_reprojects_after_late_registry() {
        // Order-independence: a root resolved BEFORE the field registry
        // arrives is registered attrs-empty, then back-filled — at the SAME
        // identity — once the registry is supplied and the lookup re-runs.
        // RPython grows `classdef.attrs` as annotation proceeds; the same
        // `ClassDef` object accrues attributes, never a fresh divergent one.
        use crate::front::StructFieldRegistry;
        let bk = bk();

        // Registry-less lookup: identity-registered, but no fields to
        // project, so `attrs` stays empty.
        let early = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers without a registry");
        assert!(
            bk.classdef_snapshot().iter().any(|c| Rc::ptr_eq(c, &early)),
            "registry-less root is still identity-registered"
        );
        assert!(
            !early.borrow().attrs.contains_key("depth"),
            "no registry -> no projected attrs"
        );

        // Registry arrives late.
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![
                ("pycode".to_string(), "PyCode".to_string()),
                ("depth".to_string(), "i64".to_string()),
            ],
        );
        reg.fields.insert(
            "PyCode".to_string(),
            vec![("argcount".to_string(), "i64".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Re-lookup re-projects onto the SAME ClassDef Rc.
        let late = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame re-lookup after registry");
        assert!(
            Rc::ptr_eq(&early, &late),
            "late re-projection keeps the original identity"
        );
        {
            let g = late.borrow();
            assert!(
                matches!(
                    g.attrs.get("depth").map(|a| &a.s_value),
                    Some(SomeValue::Integer(_))
                ),
                "depth back-filled after late registry"
            );
            assert!(
                matches!(
                    g.attrs.get("pycode").map(|a| &a.s_value),
                    Some(SomeValue::Instance(_))
                ),
                "pycode back-filled as SomeInstance after late registry"
            );
        }

        // The transitive inner struct discovered on the late pass is
        // registered with its own field projected too.
        let inner = bk
            .getuniqueclassdef_for_struct_root("PyCode")
            .expect("PyCode registers on the late pass");
        assert!(
            inner.borrow().attrs.contains_key("argcount"),
            "inner struct projected on the late pass"
        );
    }

    #[test]
    fn immutablevalue_int_sets_nonneg_when_ge_zero() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        match s {
            SomeValue::Integer(si) => {
                assert!(si.nonneg);
                assert!(si.base.const_box.is_some());
            }
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_int_negative_not_nonneg() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(-1)).unwrap();
        match s {
            SomeValue::Integer(si) => assert!(!si.nonneg),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_bool() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Bool(true)).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
    }

    #[test]
    fn immutablevalue_single_char_str_is_somechar() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::byte_str("a")).unwrap();
        assert!(matches!(s, SomeValue::Char(_)));
    }

    #[test]
    fn immutablevalue_multichar_str_is_somestring() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::byte_str("hello")).unwrap();
        assert!(matches!(s, SomeValue::String(_)));
    }

    #[test]
    fn immutablevalue_str_with_nul_clears_no_nul() {
        let bk = bk();
        let with_nul = ConstValue::byte_str("a\x00b");
        let s = bk.immutablevalue(&with_nul).unwrap();
        match s {
            SomeValue::String(st) => assert!(!st.inner.no_nul),
            other => panic!("expected SomeString, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_float_is_somefloat() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Float(1.5_f64.to_bits()))
            .unwrap();
        match s {
            SomeValue::Float(_) => {}
            other => panic!("expected SomeFloat, got {other:?}"),
        }
        let _ = SomeFloat::new();
    }

    #[test]
    fn immutablevalue_none_is_s_none() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::None).unwrap();
        assert!(matches!(s, SomeValue::None_(_)));
    }

    #[test]
    fn immutablevalue_tuple_walks_items() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Bool(false),
            ]))
            .unwrap();
        match s {
            SomeValue::Tuple(t) => {
                assert_eq!(t.items.len(), 2);
                assert!(matches!(t.items[0], SomeValue::Integer(_)));
                assert!(matches!(t.items[1], SomeValue::Bool(_)));
            }
            other => panic!("expected SomeTuple, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_list_generalizes_elements() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::List(vec![
                ConstValue::Int(1),
                ConstValue::Int(-1),
            ]))
            .unwrap();
        match s {
            SomeValue::List(sl) => {
                // Element type widened from {nonneg Int, signed Int}
                // to generic Int (nonneg=false after merge with -1).
                if let SomeValue::Integer(si) = sl.listdef.s_value() {
                    assert!(!si.nonneg);
                } else {
                    panic!("expected Int listdef s_value");
                }
                assert!(sl.base.const_box.is_some());
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn immutableconstant_list_reuses_same_constant_identity() {
        let bk = bk();
        let key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let a = bk.immutableconstant(&key).unwrap();
        let b = bk.immutableconstant(&key).unwrap();
        let (SomeValue::List(a), SomeValue::List(b)) = (a, b) else {
            panic!("expected SomeList");
        };
        assert!(a.listdef.same_as(&b.listdef));
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(key.id));
    }

    #[test]
    fn immutableconstant_list_distinguishes_equal_but_distinct_constants() {
        let bk = bk();
        let a_key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let b_key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let a = bk.immutableconstant(&a_key).unwrap();
        let b = bk.immutableconstant(&b_key).unwrap();
        let (SomeValue::List(a), SomeValue::List(b)) = (a, b) else {
            panic!("expected SomeList");
        };
        assert!(!a.listdef.same_as(&b.listdef));
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(a_key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(b_key.id));
    }

    #[test]
    fn immutableconstant_dict_distinguishes_equal_but_distinct_constants() {
        let bk = bk();
        let mut a_items = HashMap::new();
        a_items.insert(ConstValue::Int(1), ConstValue::None);
        let mut b_items = HashMap::new();
        b_items.insert(ConstValue::Int(1), ConstValue::None);
        let a_key = Constant::new(ConstValue::Dict(a_items));
        let b_key = Constant::new(ConstValue::Dict(b_items));
        let a = bk.immutableconstant(&a_key).unwrap();
        let b = bk.immutableconstant(&b_key).unwrap();
        let (SomeValue::Dict(a), SomeValue::Dict(b)) = (a, b) else {
            panic!("expected SomeDict");
        };
        assert_ne!(a.dictdef, b.dictdef);
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(a_key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(b_key.id));
    }

    #[test]
    fn immutablevalue_llptr_returns_someptr() {
        use crate::annotator::model::SomeObjectTrait;
        use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype;
        use std::cell::RefCell;
        use std::rc::Rc;

        let bk = bk();
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(lltype::LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = lltype::getfunctionptr(&graph, lltype::_getconcretetype).unwrap();
        let expected_type = lltype::typeOf(&ptr);

        let s = bk
            .immutablevalue(&ConstValue::LLPtr(Box::new(ptr)))
            .expect("LLPtr must route through extregistry SomePtr");

        match s {
            SomeValue::Ptr(p) => {
                assert_eq!(p.ll_ptrtype, expected_type);
                assert!(p.immutable());
                assert!(!p.can_be_none());
                assert!(p.is_constant());
                assert!(matches!(
                    p.base.const_box.as_ref().map(|c| &c.value),
                    Some(ConstValue::LLPtr(_))
                ));
            }
            other => panic!("expected SomePtr, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_class_returns_constant_pbc() {
        // upstream bookkeeper.py — `tp is type` produces
        // `SomeConstantType(x, self)`, a SomePBC subclass with
        // `const = x`. The Rust port emits a Class-kind `SomePBC`
        // with `const_box` set; `SomeConstantType` collapses into the
        // PBC subclass because our PBC doesn't carry a Python-class
        // inheritance shadow.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let class = HostObject::new_class("Foo", vec![]);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(class))
            .expect("class HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.values().next().unwrap().kind(),
                    DescKind::Class
                );
                assert!(pbc.base.const_box.is_some());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_user_function_returns_function_pbc() {
        // upstream bookkeeper.py — `callable(x)` falls into
        // `SomePBC([self.getdesc(x)])`. Narrow Rust port emits a
        // Function-kind Desc stub; real FunctionDesc wiring lands
        // when bookkeeper commit 2 ports getdesc.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let func = HostObject::new_user_function(compiled_graph_func(
            "def f(self):\n    return self\n",
            Vec::new(),
        ));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(func))
            .expect("user-function HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.values().next().unwrap().kind(),
                    DescKind::Function
                );
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_const_function_returns_function_pbc() {
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = GraphFunc::new("f", globals);
        let s = bk
            .immutablevalue(&ConstValue::Function(Box::new(func)))
            .expect("ConstValue::Function must route to function SomePBC");

        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.values().next().unwrap().kind(),
                    DescKind::Function
                );
                assert!(pbc.base.const_box.is_some());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_builtin_callable_returns_somebuiltin() {
        // upstream bookkeeper.py — BUILTIN_ANALYZERS lookup
        // produces SomeBuiltin with methodname
        // `getattr(x, "__module__", "unknown") + "." + x.__name__`.
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let bltn = HostObject::new_builtin_callable("sys.getdefaultencoding");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(bltn))
            .expect("builtin HostObject must produce SomeBuiltin");
        let SomeValue::Builtin(sb) = s else {
            panic!("expected SomeBuiltin");
        };
        assert_eq!(sb.methodname.as_deref(), Some("sys.getdefaultencoding"));
    }

    #[test]
    fn immutablevalue_builtin_class_uses_builtin_module_and_name_for_methodname() {
        use crate::annotator::model::SomeValue;
        let bk = bk();
        let int_cls = crate::flowspace::model::HOST_ENV
            .lookup_builtin("int")
            .expect("builtin int class");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(int_cls))
            .expect("builtin class HostObject must produce SomeBuiltin");
        let SomeValue::Builtin(sb) = s else {
            panic!("expected SomeBuiltin");
        };
        assert_eq!(sb.methodname.as_deref(), Some("__builtin__.int"));
    }

    #[test]
    fn immutablevalue_property_returns_someproperty() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let fget = HostObject::new_user_function(GraphFunc::new("fget", globals));
        let prop = HostObject::new_property("pkg.C.x", Some(fget.clone()), None, None);

        let s = bk
            .immutablevalue(&ConstValue::HostObject(prop))
            .expect("property HostObject must produce SomeProperty");

        match s {
            SomeValue::Property(prop) => assert_eq!(prop.fget, Some(fget)),
            other => panic!("expected SomeProperty, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_bound_method_returns_method_pbc() {
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.C", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.C.m", inst, func, "m", cls);

        let s = bk
            .immutablevalue(&ConstValue::HostObject(meth))
            .expect("bound method HostObject must produce SomePBC");

        match s {
            SomeValue::PBC(pbc) => assert_eq!(pbc.get_kind().unwrap(), DescKind::Method),
            other => panic!("expected method SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_dead_weakref_returns_none_classdef() {
        // upstream bookkeeper.py:300-302 — `x1 = x(); if x1 is None:
        // result = SomeWeakRef(None)`.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let dead = HostObject::new_weakref("weakref.ref", None);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(dead))
            .expect("dead weakref must route to SomeWeakRef");
        match s {
            SomeValue::WeakRef(wr) => assert!(wr.classdef.is_none()),
            other => panic!("expected SomeWeakRef(None), got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_live_weakref_pulls_instance_classdef() {
        // upstream bookkeeper.py — `s1 = immutablevalue(x1);
        // assert isinstance(s1, SomeInstance); result = SomeWeakRef(s1.classdef)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Target", vec![]);
        let inst = HostObject::new_instance(cls, vec![]);
        let wref = HostObject::new_weakref("weakref.ref", Some(inst));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(wref))
            .expect("live weakref must route to SomeWeakRef");
        match s {
            SomeValue::WeakRef(wr) => assert!(wr.classdef.is_some()),
            other => panic!("expected SomeWeakRef(Some(classdef)), got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_module_without_freeze_raises() {
        // upstream bookkeeper.py:349 — a raw Module hits the final
        // `raise AnnotatorError("Don't know how to represent %r")`
        // fallthrough because `module.__class__.__module__ ==
        // '__builtin__'` and it lacks `_freeze_`.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        let err = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect_err("raw module must raise AnnotatorError");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("Don't know how to represent")
        );
    }

    #[test]
    fn immutablevalue_module_with_freeze_routes_to_some_pbc() {
        // upstream bookkeeper.py:332-338 — any host with `_freeze_`
        // routes through SomePBC([getdesc(x)]). Upstream first checks
        // `hasattr(x, '_freeze_')` then asserts `x._freeze_() is True`,
        // so the Rust port registers `_freeze_` as a NativeCallable
        // that returns `Bool(true)`.
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        mod_obj.module_set(
            "_freeze_",
            HostObject::new_native_callable(
                "m._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            ),
        );
        let s = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect("module with _freeze_ must route to SomePBC");
        assert!(matches!(s, SomeValue::PBC(_)));
    }

    #[test]
    fn immutablevalue_user_class_instance_routes_to_some_instance() {
        // upstream bookkeeper.py:339-345 — user-class instance
        // (non-`_freeze_`, non-builtin class) routes to
        // `SomeInstance(getuniqueclassdef(x.__class__))` with the
        // instance recorded via `classdef.see_instance(x)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Holder", vec![]);
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        inst.instance_set("x", ConstValue::Int(42)).unwrap();
        let s = bk
            .immutablevalue(&ConstValue::HostObject(inst.clone()))
            .expect("user-class instance must route to SomeInstance");
        match s {
            SomeValue::Instance(si) => {
                assert!(si.classdef.is_some(), "classdef must be attached");
                // `see_instance` records the instance in classdef.instances_seen.
                let classdef = si.classdef.unwrap();
                assert!(
                    classdef
                        .borrow()
                        .instances_seen
                        .contains(&inst.identity_id())
                );
            }
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_freeze_instance_routes_to_some_pbc() {
        // upstream bookkeeper.py — `hasattr(x, '_freeze_')` +
        // `assert x._freeze_() is True` routes to
        // `SomePBC([getdesc(x)])`. The Rust port registers `_freeze_`
        // as a NativeCallable on the class dict so the descriptor
        // protocol binds `self` and `call_host` executes it.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Frozen", vec![]);
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect("_freeze_ instance must route to SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_freeze_returning_false_raises_annotator_error() {
        // upstream bookkeeper.py `assert x._freeze_() is True` —
        // a `_freeze_` returning False triggers AssertionError. The
        // Rust port surfaces this as an `AnnotatorError`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.NotFrozen", vec![]);
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.NotFrozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(false))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let err = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect_err("_freeze_ returning False must raise");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("_freeze_() must return True")
        );
    }

    #[test]
    fn immutablevalue_non_callable_freeze_raises_annotator_error() {
        // upstream bookkeeper.py performs `x._freeze_()`;
        // if `_freeze_` exists but is not callable, Python raises
        // `TypeError`. The Rust port surfaces that as an AnnotatorError.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.BadFrozen", vec![]);
        cls.class_set("_freeze_", ConstValue::Bool(true));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let err = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect_err("non-callable _freeze_ must raise");
        assert!(err.msg.unwrap_or_default().contains("_freeze_ exists on"));
    }

    #[test]
    fn immutablevalue_runs_cleanup_on_instance() {
        // upstream bookkeeper.py:341-342:
        //     if hasattr(x, '_cleanup_'): x._cleanup_()
        // The Rust port invokes `_cleanup_` for side effects. We verify
        // the call happens via an AtomicBool (HostCallableFn requires
        // `Send + Sync`, so a plain `Cell` is not shareable into the
        // closure).
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Cleaned", vec![]);
        let called = Arc::new(AtomicBool::new(false));
        let called_cl = called.clone();
        cls.class_set(
            "_cleanup_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Cleaned._cleanup_",
                Arc::new(move |_args| {
                    called_cl.store(true, Ordering::SeqCst);
                    Ok(ConstValue::None)
                }),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let _ = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect("instance with _cleanup_ must not fail");
        assert!(
            called.load(Ordering::SeqCst),
            "_cleanup_ must have been invoked"
        );
    }

    #[test]
    fn getdesc_bound_method_of_frozen_instance_runs_cleanup_and_returns_method_of_frozen() {
        use crate::annotator::model::DescKind;
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.Frozen", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_cl = cleaned.clone();
        cls.class_set(
            "_cleanup_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._cleanup_",
                Arc::new(move |_args| {
                    cleaned_cl.store(true, Ordering::SeqCst);
                    Ok(ConstValue::None)
                }),
            )),
        );
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.Frozen.m", inst, func, "m", cls);

        let entry = bk.getdesc(&meth).expect("frozen bound method desc");

        assert_eq!(entry.kind(), DescKind::MethodOfFrozen);
        assert!(
            cleaned.load(Ordering::SeqCst),
            "_cleanup_ must run before frozen bound-method classification"
        );
    }

    #[test]
    fn getdesc_bound_method_freeze_returning_false_raises() {
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        use std::sync::Arc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.NotFrozen", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.NotFrozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(false))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.NotFrozen.m", inst, func, "m", cls);

        let err = bk
            .getdesc(&meth)
            .expect_err("_freeze_ returning False must raise");

        assert!(
            err.msg
                .unwrap_or_default()
                .contains("_freeze_() must return True")
        );
    }

    #[test]
    fn newlist_creates_somelist_and_generalizes() {
        // Two SomeInteger variants so the `Int ∪ Int` pair-union
        // widens them (signed `∪` nonneg → signed).
        let bk = bk();
        let s_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let s_signed = SomeValue::Integer(SomeInteger::new(false, false));
        let out = bk.newlist(&[s_nonneg, s_signed], None).unwrap();
        // Element type is now signed Int (widened from nonneg).
        if let SomeValue::Integer(si) = out.listdef.s_value() {
            assert!(!si.nonneg);
        } else {
            panic!("expected SomeInteger listdef element");
        }
    }

    #[test]
    fn newdict_creates_someordicteddict_equivalent() {
        let bk = bk();
        let out = bk.newdict();
        // Fresh-position newdict without subsequent generalize_key /
        // generalize_value carries Impossible for both.
        assert!(matches!(out.dictdef.s_key(), SomeValue::Impossible));
        assert!(matches!(out.dictdef.s_value(), SomeValue::Impossible));
    }

    #[test]
    fn getlistdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(1, 2, 0)));
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(3, 4, 0)));
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn getlistdef_caches_under_none_position() {
        // upstream bookkeeper.py:180 indexes `self.listdefs[self.
        // position_key]` — when position_key is `None`, both calls
        // land on the same dict entry. Rust port mirrors this by
        // using `Option<PositionKey>` as the cache key; two `getlist
        // def` calls outside a reflow frame must share the same
        // ListDef.
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_under_none_position() {
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn unicode_through_const_unistr() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::uni_str("abc")).unwrap();
        match s {
            SomeValue::UnicodeString(st) => assert!(st.inner.no_nul),
            other => panic!("expected SomeUnicodeString, got {other:?}"),
        }
    }

    #[test]
    fn char_has_no_nul() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::byte_str("x")).unwrap();
        match s {
            SomeValue::Char(_) => {
                let c = SomeChar::new(true);
                assert!(c.inner.no_nul);
            }
            other => panic!("expected SomeChar, got {other:?}"),
        }
    }

    // classdesc: descs / classdefs / getdesc

    #[test]
    fn getdesc_for_user_function_returns_function_entry() {
        let bk = bk();
        let gf = GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let entry = bk.getdesc(&host).unwrap();
        assert!(matches!(entry, DescEntry::Func(_)));
        let fd = entry.as_function().unwrap();
        assert_eq!(fd.borrow().name, "f");
    }

    #[test]
    fn getdesc_caches_same_pyobj() {
        // Two getdesc calls with the same HostObject return the same
        // Rc (identity equal).
        let bk = bk();
        let gf = GraphFunc::new("g", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let a = bk.getdesc(&host).unwrap();
        let b = bk.getdesc(&host).unwrap();
        assert_eq!(a, b);
        // Same underlying FunctionDesc Rc — pointer-identity.
        let a_fd = a.as_function().unwrap();
        let b_fd = b.as_function().unwrap();
        assert!(Rc::ptr_eq(&a_fd, &b_fd));
    }

    #[test]
    fn getdesc_for_class_returns_class_entry_shell() {
        let bk = bk();
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let entry = bk.getdesc(&cls).unwrap();
        assert!(matches!(entry, DescEntry::Class(_)));
        let cd = entry.as_class().unwrap();
        assert_eq!(cd.borrow().name, "pkg.Foo");
    }

    #[test]
    fn unit_variant_prebuilt_instance_is_of_interned_class() {
        // The prebuilt instance's `__class__` is the interned class
        // object (`rclass.py InstanceRepr.convert_const` /
        // `classdesc.py getuniqueclassdef`). getuniqueclassdef of that
        // class has the Continue → StepResult<*mut PyObject> →
        // StepResult chain.
        use crate::annotator::classdesc::ClassDef;
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyopcode::StepResult".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "pyopcode::StepResult<*mut PyObject>::Continue".to_string(),
            vec![],
        );
        bk.set_struct_fields(Rc::new(reg));
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let dotted = "pyre_interpreter.pyopcode.StepResult<*mut PyObject>.Continue";
        let interned = bk.intern_enum_variant_host(
            "pyre_interpreter::pyopcode::StepResult<*mut PyObject>",
            "Continue",
        );
        let instance = bk
            .intern_unit_variant_prebuilt_instance(dotted, Some(0))
            .expect("unit-variant prebuilt instance");
        let cls = instance.class_of().expect("instance class");
        assert_eq!(cls, interned);
        let cd = bk
            .getuniqueclassdef(&cls)
            .expect("getuniqueclassdef of the interned class");
        let mro: Vec<String> = ClassDef::getmro(&cd)
            .into_iter()
            .map(|c| c.borrow().name.clone())
            .collect();
        assert!(
            mro.iter().any(|n| n.contains("Continue")),
            "MRO must include Continue, got {mro:?}"
        );
        assert!(
            mro.iter().any(|n| n.contains("StepResult<*mut PyObject>")),
            "MRO must include StepResult<*mut PyObject>, got {mro:?}"
        );
        let short = bk
            .getuniqueclassdef_for_struct_root("pyopcode::StepResult")
            .expect("field class");
        assert!(
            cd.borrow().issubclass(&short),
            "Continue must subclass StepResult, MRO {mro:?}"
        );
        assert!(
            mro.iter().any(|n| {
                let leaf = n.rsplit("::").next().unwrap_or(n);
                leaf == "StepResult"
            }),
            "MRO must include StepResult, got {mro:?}"
        );
    }

    #[test]
    fn getdesc_for_instance_returns_frozen_entry() {
        let bk = bk();
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let inst = HostObject::new_instance(cls, vec![]);
        let entry = bk.getdesc(&inst).unwrap();
        assert!(matches!(entry, DescEntry::Frozen(_)));
    }

    #[test]
    fn getdesc_for_builtin_callable_returns_frozen_entry() {
        // bookkeeper.py treats builtin callables as frozen PBCs via
        // the _freeze_ fallback.
        let bk = bk();
        let obj = HostObject::new_builtin_callable("len");
        let entry = bk.getdesc(&obj).unwrap();
        assert!(matches!(entry, DescEntry::Frozen(_)));
    }

    #[test]
    fn getmethoddesc_caches_identity() {
        // Two calls with the same funcdesc/classdefs/name/flags
        // return the same MethodDesc Rc.
        let bk = bk();
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let origin = crate::annotator::description::ClassDefKey::from_raw(1);
        let self_def = Some(crate::annotator::description::ClassDefKey::from_raw(2));
        let flags = std::collections::BTreeMap::new();
        let a = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        let b = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        assert!(Rc::ptr_eq(&a, &b));
    }

    /// `getmethoddesc_for_attribute` mirrors the regular-method branch
    /// of `bookkeeper.py getdesc`: walks the receiver classdef's
    /// MRO read-only, picks the first Method PBC, and routes through
    /// `getmethoddesc` so the upstream `methoddescs` cache is primed.
    /// Missing attribute → `None`.
    #[test]
    fn getmethoddesc_for_attribute_walks_mro_and_primes_cache() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();
        let pyobj = HostObject::new_class("PyFrame", vec![]);
        let desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, pyobj, "PyFrame".into()),
        ));
        let classdef = crate::annotator::classdesc::ClassDef::new(&bk, &desc);
        let classdef_key = crate::annotator::description::ClassDefKey::from_classdef(&classdef);

        // Mint a MethodDesc via getmethoddesc and seed the attrs.
        let gf = GraphFunc::new("push", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let md = bk.getmethoddesc(
            &fd,
            classdef_key,
            Some(classdef_key),
            "push",
            std::collections::BTreeMap::new(),
        );
        let pbc = SomePBC::new([DescEntry::Method(md.clone())], false);
        let mut attr = Attribute::new("push");
        attr.s_value = SomeValue::PBC(pbc);
        classdef.borrow_mut().attrs.insert("push".into(), attr);

        // Idempotent on cache hit: returns the cached MethodDesc rc.
        let results = bk.getmethoddesc_for_attribute(&classdef, "push");
        assert_eq!(results.len(), 1, "expected single Method PBC under push");
        assert!(Rc::ptr_eq(&results[0], &md));

        // Missing attribute → empty Vec.
        assert!(
            bk.getmethoddesc_for_attribute(&classdef, "absent")
                .is_empty()
        );
    }

    /// `bookkeeper.py` getdesc bound-method branch and
    /// `description.py MethodDesc.bind_self` rebind the descriptor
    /// to the receiver classdef.  Walking from a subclass receiver
    /// must produce a MethodDesc whose `selfclassdef` is the receiver
    /// classdef key, even when the attr stores an unbound entry on a
    /// base class (`selfclassdef = None`).  `originclassdef` stays the
    /// base — that's where the method body lives.
    #[test]
    fn getmethoddesc_for_attribute_rebinds_unbound_to_receiver() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        // Base class carries the unbound MethodDesc in attrs.
        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        // Derived class inherits from Base — MRO walks Derived → Base.
        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Seed Base.attrs["m"] with an unbound MethodDesc
        // (selfclassdef = None) so the walker has to perform the bind.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let unbound = bk.getmethoddesc(&fd, base_key, None, "m", std::collections::BTreeMap::new());
        assert_eq!(unbound.borrow().selfclassdef, None);
        let pbc = SomePBC::new([DescEntry::Method(unbound.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived must return a MethodDesc bound to
        // Derived (upward best-match branch — Derived.issubclass(Base))
        // with originclassdef preserved as Base.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let bound = &results[0];
        assert_eq!(bound.borrow().originclassdef, base_key);
        assert_eq!(bound.borrow().selfclassdef, Some(derived_key));
        // The unbound carrier is untouched; the bound result is a
        // distinct MethodDesc rc (cache key differs by selfclassdef).
        assert!(!Rc::ptr_eq(bound, &unbound));
    }

    /// `classdesc.py lookup_filter` only rebinds MDs whose
    /// `selfclassdef is None`.  Already-bound MDs (from instance
    /// attribute origin) are appended with their existing
    /// `selfclassdef` preserved.  Walking from a derived receiver
    /// must NOT clobber a MD already bound to its origin classdef.
    #[test]
    fn getmethoddesc_for_attribute_preserves_already_bound_selfclassdef() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);

        // Seed Base.attrs["m"] with a MethodDesc already bound to Base
        // (selfclassdef = Some(base_key)) — emulates an instance
        // attribute carrying a pre-bound bound-method object.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let already_bound = bk.getmethoddesc(
            &fd,
            base_key,
            Some(base_key),
            "m",
            std::collections::BTreeMap::new(),
        );
        assert_eq!(already_bound.borrow().selfclassdef, Some(base_key));
        let pbc = SomePBC::new([DescEntry::Method(already_bound.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived must preserve the existing selfclassdef =
        // Some(base_key) — NOT rebind to Derived.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let resolved = &results[0];
        assert_eq!(resolved.borrow().originclassdef, base_key);
        assert_eq!(
            resolved.borrow().selfclassdef,
            Some(base_key),
            "already-bound selfclassdef must be preserved, not rebound to receiver",
        );
        assert_ne!(derived_key, base_key, "fixture sanity: keys differ");
    }

    /// `classdesc.py lookup_filter` first branch: when an
    /// unbound MD's `originclassdef` is a strict subclass of the
    /// receiver, the method is kept and bound to `methclassdef`
    /// (NOT `receiver`).  Mirrors the comment:
    ///   "bind the method by giving it a selfclassdef.  Use the
    ///    more precise subclass that it's coming from."
    #[test]
    fn getmethoddesc_for_attribute_subclass_origin_binds_to_methclassdef() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Seed Base.attrs["m"] with an unbound MethodDesc whose
        // originclassdef is Derived (a strict subclass of the
        // receiver Base) — mimics a PBC union carrying a subclass
        // override under a base-class attribute name.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let subclass_origin = bk.getmethoddesc(
            &fd,
            derived_key,
            None,
            "m",
            std::collections::BTreeMap::new(),
        );
        let pbc = SomePBC::new([DescEntry::Method(subclass_origin.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Base sees originclassdef = Derived as a strict
        // subclass — bind to Derived (methclassdef), not Base.
        let results = bk.getmethoddesc_for_attribute(&base_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let bound = &results[0];
        assert_eq!(bound.borrow().originclassdef, derived_key);
        assert_eq!(
            bound.borrow().selfclassdef,
            Some(derived_key),
            "subclass-origin branch binds to methclassdef (Derived), not receiver (Base)",
        );
        assert_ne!(base_key, derived_key, "fixture sanity");
    }

    /// classdesc.py lookup_filter collects ALL matching descs
    /// into `d`, not just the first.  A PBC with both an already-bound
    /// MD and an unbound upward-match MD must return both entries.
    #[test]
    fn getmethoddesc_for_attribute_collects_multi_desc_pbc() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Two FunctionDescs so the MDs have distinct cache keys.
        let gf1 = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host1 = HostObject::new_user_function(gf1);
        let fd1 = bk
            .getdesc(&host1)
            .unwrap()
            .as_func_entry()
            .cloned()
            .unwrap();
        let gf2 = GraphFunc::new("m2", Constant::new(ConstValue::Dict(Default::default())));
        let host2 = HostObject::new_user_function(gf2);
        let fd2 = bk
            .getdesc(&host2)
            .unwrap()
            .as_func_entry()
            .cloned()
            .unwrap();

        // MD1: already-bound to Base (selfclassdef = Some(base_key)).
        let already_bound = bk.getmethoddesc(
            &fd1,
            base_key,
            Some(base_key),
            "m",
            std::collections::BTreeMap::new(),
        );
        // MD2: unbound, originclassdef = Base.
        // Walking from Derived → upward match → bound to Derived.
        let unbound =
            bk.getmethoddesc(&fd2, base_key, None, "m", std::collections::BTreeMap::new());

        let pbc = SomePBC::new(
            [
                DescEntry::Method(already_bound.clone()),
                DescEntry::Method(unbound.clone()),
            ],
            false,
        );
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived: already-bound MD preserved +
        // unbound MD bound to Derived as upward best-match = 2 entries.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(
            results.len(),
            2,
            "multi-desc PBC must collect all matching MDs, not just the first",
        );
        // First entry: already-bound, selfclassdef preserved.
        assert_eq!(results[0].borrow().selfclassdef, Some(base_key));
        // Second entry: upward best-match, bound to receiver (Derived).
        assert_eq!(results[1].borrow().selfclassdef, Some(derived_key));
    }

    #[test]
    fn register_classdef_appends_to_snapshot() {
        let bk = bk();
        let cd_rc = ClassDef::new_standalone("pkg.C", None);
        bk.register_classdef(cd_rc.clone());
        let snap = bk.classdef_snapshot();
        assert_eq!(snap.len(), 1);
        assert!(Rc::ptr_eq(&snap[0], &cd_rc));
    }

    #[test]
    fn descentry_identity_eq() {
        let bk = bk();
        let gf = GraphFunc::new("h", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let a = bk.getdesc(&host).unwrap();
        let b = bk.getdesc(&host).unwrap();
        assert_eq!(a.desc_key(), b.desc_key());
    }

    #[test]
    fn enter_leave_registers_tls() {
        let bk = bk();
        assert!(getbookkeeper().is_none());
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        assert!(getbookkeeper().is_some());
        assert!(bk.position_entered.get());
        assert_eq!(bk.current_position_key(), Some(PositionKey::new(1, 1, 0)));
        bk.leave();
        assert!(getbookkeeper().is_none());
        assert!(!bk.position_entered.get());
        assert!(bk.current_position_key().is_none());
    }

    #[test]
    #[should_panic(expected = "don't call enter() nestedly")]
    fn enter_twice_panics() {
        let bk = bk();
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        // Second enter without leave — matches upstream's `assert not hasattr`.
        bk.enter(Some(PositionKey::new(2, 2, 0)));
    }

    #[test]
    fn at_position_raii_enters_and_leaves() {
        let bk = bk();
        {
            let _guard = bk.at_position(Some(PositionKey::new(3, 4, 0)));
            assert!(bk.position_entered.get());
            assert!(getbookkeeper().is_some());
        }
        assert!(!bk.position_entered.get());
        assert!(getbookkeeper().is_none());
    }

    #[test]
    fn at_position_none_fast_path_skips_when_entered() {
        let bk = bk();
        bk.enter(Some(PositionKey::new(5, 5, 0)));
        {
            // Upstream fast-path: hasattr(self, 'position_key') and pos is None
            let _guard = bk.at_position(None);
            assert!(bk.position_entered.get(), "outer enter stays active");
            // Fast-path does not clobber position.
            assert_eq!(bk.current_position_key(), Some(PositionKey::new(5, 5, 0)));
        }
        // Guard drop must NOT leave because we skipped the enter.
        assert!(bk.position_entered.get());
        bk.leave();
    }

    #[test]
    fn valueoftype_delegates_to_annotationoftype() {
        use crate::annotator::signature::AnnotationSpec;
        let bk = bk();
        let bool_ann = bk.valueoftype(&AnnotationSpec::Bool).unwrap();
        assert!(matches!(bool_ann, SomeValue::Bool(_)));
        let int_ann = bk.valueoftype(&AnnotationSpec::Int).unwrap();
        assert!(matches!(int_ann, SomeValue::Integer(_)));
    }

    #[test]
    fn consider_call_site_routes_lladtmeth_through_immutablevalue_and_prepends_ptr() {
        use crate::annotator::model::SomeLLADTMeth;
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::pygraph::PyGraph;
        use crate::translator::rtyper::llannotation::lltype_to_annotation;
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType, Ptr};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let _guard = ann.bookkeeper.at_position(None);

        let gf = compiled_graph_func("def f(self):\n    return self\n", Vec::new());
        let func = HostObject::new_user_function(gf.clone());
        let fd = ann
            .bookkeeper
            .getdesc(&func)
            .unwrap()
            .as_function()
            .unwrap();
        fd.borrow().cache.borrow_mut().insert(
            "".into(),
            Rc::new(PyGraph::new(gf.clone(), gf.code.as_ref().unwrap())),
        );
        let ll_ptrtype = Ptr {
            TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        };
        let mut v_callable = Variable::named("adtmeth");
        ann.setbinding(
            &mut v_callable,
            SomeValue::LLADTMeth(SomeLLADTMeth::new(
                crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::Ptr(
                    ll_ptrtype.clone(),
                ),
                ConstValue::HostObject(func.clone()),
            )),
        );
        let call_op = SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(v_callable)],
            Hlvalue::Variable(Variable::named("r")),
        );

        ann.bookkeeper
            .consider_call_site(&call_op, None)
            .expect("SomeLLADTMeth call site must flow through the wrapped function");

        let family = fd.borrow().base.getcallfamily().unwrap();
        let expected_shape =
            build_args_for_op("simple_call", &[Some(lltype_to_annotation(ll_ptrtype))])
                .unwrap()
                .rawshape();
        let family_ref = family.borrow();
        let table = family_ref
            .calltables
            .get(&expected_shape)
            .expect("SomeLLADTMeth must prepend ll_ptrtype to the call args");
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn immutablevalue_free_function_uses_tls() {
        use crate::flowspace::model::ConstValue;
        let bk = bk();
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        let s = immutablevalue(&ConstValue::Bool(true)).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
        bk.leave();
    }

    #[test]
    fn annotator_backlink_upgrades_when_alive() {
        // bookkeeper.py — `self.annotator = annotator`. In Rust
        // the backlink is a Weak<RPythonAnnotator>; while the outer
        // `Rc<RPythonAnnotator>` is alive, `bookkeeper.annotator()`
        // must return a live Rc to the same driver.
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let got = ann.bookkeeper.annotator();
        assert!(Rc::ptr_eq(&got, &ann));
    }

    #[test]
    #[should_panic(expected = "Bookkeeper.annotator backlink is absent or dropped")]
    fn annotator_backlink_panics_when_unwired() {
        // A bare Bookkeeper::new() (test-only constructor) leaves the
        // annotator slot empty; any attempt to upgrade panics.
        let bk = Bookkeeper::new();
        let _ = bk.annotator();
    }

    #[test]
    fn compute_at_fixpoint_enters_position_and_clears_emulated() {
        // bookkeeper.py — compute_at_fixpoint wraps the work
        // in `with self.at_position(None)` and clears
        // `emulated_pbc_calls` at the end. Both effects are
        // observable even when the inner loops find no work.
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        // Before: no position entered; emulated_pbc_calls starts empty.
        assert!(!ann.bookkeeper.position_entered.get());
        assert!(ann.bookkeeper.emulated_pbc_calls.borrow().is_empty());

        // Run — should not panic, should leave position_entered false
        // (at_position guard exits via Drop).
        ann.bookkeeper
            .compute_at_fixpoint()
            .expect("empty annotator state must not error");
        assert!(!ann.bookkeeper.position_entered.get());
        assert!(ann.bookkeeper.emulated_pbc_calls.borrow().is_empty());
    }

    #[test]
    fn struct_root_pass2_replaces_untyped_force_shell() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        // Production order (lib.rs registration): the FORCE table is
        // populated from ValueType rows BEFORE any classdef exists, so
        // `_init_classdef` seeds `varnames` with the classdef-less
        // `SomeInstance` shell (`valuetype_to_someshell(Ref)`).  Pass-2
        // of `getuniqueclassdef_for_struct_root` must replace that
        // shell with the registry-string projection (`Box<[String]>`
        // -> SomeList), not skip it as already-filled.
        crate::annotator::classdesc::register_struct_fields(
            "CodeObject",
            &[("varnames".to_string(), crate::model::ValueType::Ref(None))],
        );
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![("code".to_string(), "CodeObject".to_string())],
        );
        reg.fields.insert(
            "CodeObject".to_string(),
            vec![("varnames".to_string(), "Box<[String]>".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));
        let _root = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers");
        let cd = bk
            .getuniqueclassdef_for_struct_root("CodeObject")
            .expect("CodeObject registered");
        let g = cd.borrow();
        let varnames = g.attrs.get("varnames").expect("varnames attr present");
        assert!(
            matches!(varnames.s_value, SomeValue::List(_)),
            "varnames must project to SomeList, got {:?}",
            varnames.s_value
        );
    }

    #[test]
    fn boxed_slice_field_keeps_its_length_word() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let got = fat_box_field_lltype("Box<[String]>").expect("Box<[String]> is fat");
        let LowLevelType::Struct(st) = got else {
            panic!("boxed slice must be the two-word value");
        };
        assert_eq!(st._names, vec!["data".to_string(), "len".to_string()]);
        assert!(matches!(st._flds.get("len"), Some(LowLevelType::Signed)));
        assert!(fat_box_field_lltype("Box<CodeObject>").is_none());
        assert!(peel_one_pointer("Box<CodeObject>").is_some());
        assert!(peel_one_pointer("Box<[String]>").is_none());
    }

    #[test]
    fn fat_raw_pointer_field_keeps_its_metadata_word() {
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};

        let word = crate::layout::target_word_size();
        let spelling = "Option<*mut dyn AsyncActionOps>";
        let fat = fat_dyn_pointer_field_lltype(spelling).expect("unsized raw pointer is two words");
        assert!(peel_one_pointer(spelling).is_none());
        assert!(peel_one_pointer("&dyn Storage").is_none());
        assert!(peel_one_pointer("*const [u8]").is_none());
        assert!(fat_dyn_pointer_field_lltype("*const [u8]").is_some());
        assert!(fat_dyn_pointer_field_lltype("Box<[u8]>").is_none());
        let holder = crate::translator::rtyper::lltypesystem::lltype::Struct::registry_fields(
            "ExecutionContext",
            vec![
                ("check_signal_action".to_string(), fat),
                ("sys_exc_value".to_string(), LowLevelType::Address),
            ],
            GcKind::Gc,
        );
        // Charon `TypeLayout.variant_layouts[0].field_offsets` for a
        // fat pointer then a word field: the trailing field is two words
        // in (`std::mem::offset_of!` of the same layout).
        let llbc_offsets = [0usize, 2 * word];
        let mut computed = Vec::new();
        let mut offset = 0usize;
        for name in &holder._names {
            computed.push(offset);
            let field = holder._flds.get(name).expect("named field");
            offset += match field {
                LowLevelType::Struct(st) => st._names.len() * word,
                _ => word,
            };
        }
        assert_eq!(computed, llbc_offsets);
        assert_eq!(
            computed[1] - computed[0],
            2 * word,
            "later fields must keep the metadata word"
        );
    }

    #[test]
    fn struct_root_pass2_replaces_integer_force_shell_of_fn_field() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        // `tyref_to_attr_value_type` banks `Ptr(FuncType)` as Int, so FORCE
        // shells the field `SomeInteger`. Pass-2 must replace that with the
        // layout spelling's `SomePtr` (`project_struct_field_type("fn")`).
        crate::annotator::classdesc::register_struct_fields(
            "CallbackHolder",
            &[("slot".to_string(), crate::model::ValueType::Int)],
        );
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "CallbackHolder".to_string(),
            vec![("slot".to_string(), "fn".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));
        let cd = bk
            .getuniqueclassdef_for_struct_root("CallbackHolder")
            .expect("CallbackHolder registers");
        let g = cd.borrow();
        let slot = g.attrs.get("slot").expect("slot attr present");
        assert!(
            matches!(slot.s_value, SomeValue::Ptr(_)),
            "fn field must project to SomePtr(FuncType), got {:?}",
            slot.s_value
        );
    }

    #[test]
    fn struct_root_pass2_keeps_integer_force_shell_of_non_fn_int_field() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        // A non-fn Int-bank field (`usize`) is also FORCE-shelled as
        // `valuetype_to_someshell(Int)`. Pass-2 must leave that shell:
        // the Integer→Ptr replacement is only the exact FORCE shell of
        // a function-pointer spelling. Replacing it would install the
        // `usize` projection (`s_uint`, unsigned) over the FORCE shell.
        crate::annotator::classdesc::register_struct_fields(
            "LenHolder",
            &[("len".to_string(), crate::model::ValueType::Int)],
        );
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "LenHolder".to_string(),
            vec![("len".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));
        let cd = bk
            .getuniqueclassdef_for_struct_root("LenHolder")
            .expect("LenHolder registers");
        let g = cd.borrow();
        let len = g.attrs.get("len").expect("len attr present");
        let force = crate::codewriter::annotation_state::valuetype_to_someshell(
            &crate::model::ValueType::Int,
        );
        assert_eq!(
            Some(&len.s_value),
            force.as_ref(),
            "non-fn Int field must keep the Int FORCE shell, got {:?}",
            len.s_value
        );
    }

    #[test]
    fn shaped_tuple_nullable_rows_keep_variant_classdefs_after_constructor_setattr() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::front::StructFieldRegistry;

        let bk = bk();
        let tuple = "Tuple<Option<Vec<*mut PyObject>>,Result<Vec<*mut PyObject>,PyErr>>";
        crate::annotator::classdesc::register_struct_fields(
            tuple,
            &[
                ("__pos_0".to_string(), crate::model::ValueType::Ref(None)),
                ("__pos_1".to_string(), crate::model::ValueType::Ref(None)),
            ],
        );
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            tuple.to_string(),
            vec![
                (
                    "__pos_0".to_string(),
                    "Option<Vec<*mut PyObject>>".to_string(),
                ),
                (
                    "__pos_1".to_string(),
                    "Result<Vec<*mut PyObject>,PyErr>".to_string(),
                ),
            ],
        );
        reg.fields.insert(
            "Option<Vec<*mut PyObject>>".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Option<Vec<*mut PyObject>>::Some".to_string(),
            vec![("__pos_0".to_string(), "Vec<*mut PyObject>".to_string())],
        );
        reg.fields.insert(
            "Result<Vec<*mut PyObject>,PyErr>".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Result<Vec<*mut PyObject>,PyErr>::Ok".to_string(),
            vec![("__pos_0".to_string(), "Vec<*mut PyObject>".to_string())],
        );
        bk.set_struct_fields(Rc::new(reg));

        // Minting the shape class projects its rows (`_init_classdef`), so
        // the untyped force shell is already reset when the classdef is
        // returned; a later projection is a no-op.
        let tuple_host = bk.intern_class_by_qualname(tuple);
        let tuple_cd = bk.getuniqueclassdef(&tuple_host).expect("tuple classdef");
        bk.project_struct_rows(tuple).expect("tuple rows project");
        {
            let attrs = tuple_cd.borrow();
            for field in ["__pos_0", "__pos_1"] {
                assert!(
                    matches!(
                        attrs.attrs.get(field).expect("tuple field").s_value,
                        SomeValue::Impossible
                    ),
                    "{field} filtered row must reset its force shell before setattr"
                );
            }
        }

        let some_cd = bk
            .getuniqueclassdef_for_enum_variant("Option<Vec<*mut PyObject>>", "Some")
            .expect("Option::Some");
        let ok_cd = bk
            .getuniqueclassdef_for_enum_variant("Result<Vec<*mut PyObject>,PyErr>", "Ok")
            .expect("Result::Ok");
        for (name, classdef) in [("Option::Some", &some_cd), ("Result::Ok", &ok_cd)] {
            let payload = classdef.borrow().attrs["__pos_0"].s_value.clone();
            classdef
                .borrow_mut()
                .attrs
                .get_mut("__pos_0")
                .expect("variant payload")
                .modified(Some(classdef))
                .expect("constructor setattr marks payload mutable");
            ClassDef::generalize_attr(classdef, "__pos_0", Some(payload))
                .expect("constructor payload setattr");
            let attrs = classdef.borrow();
            assert!(
                matches!(
                    attrs.attrs.get("__pos_0").map(|attr| &attr.s_value),
                    Some(SomeValue::RustVec(_))
                ),
                "{name} concrete one-word-item Vec payload must be projected to the raw Vec \
                 header before repr setup"
            );
            assert!(
                !attrs.attrs["__pos_0"].readonly,
                "{name} constructor payload must be an instance field before repr setup"
            );
        }
        for (field, classdef) in [("__pos_0", some_cd.clone()), ("__pos_1", ok_cd.clone())] {
            ClassDef::generalize_attr(
                &tuple_cd,
                field,
                Some(SomeValue::Instance(SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                ))),
            )
            .expect("constructor setattr");
        }
        let attrs = tuple_cd.borrow();
        for (field, expected) in [("__pos_0", some_cd), ("__pos_1", ok_cd)] {
            let value = &attrs.attrs.get(field).expect("tuple field").s_value;
            eprintln!("probe {field} after constructor setattr: {value:?}");
            assert!(
                matches!(value, SomeValue::Instance(inst)
                if inst.classdef.as_ref().is_some_and(|cd| Rc::ptr_eq(cd, &expected))),
                "{field} must retain variant ClassDef, got {value:?}"
            );
        }
    }

    /// Fixture for the methods-on-classdef capability that `dyn Trait`
    /// receiver-driven dispatch (receiver-dispatch configuration) needs.
    ///
    /// A ≥2-impl trait family registered through `register_trait_family`
    /// must (a) link a base ClassDef to its impl subclasses, (b) carry the
    /// trait methods as members so `getattr(instance, method)` annotates to
    /// a `MethodDesc` PBC family instead of a classdef-less shell, and
    /// (c) let the existing `MethodsPBCRepr` port lower that family to a
    /// vtable/method dispatch.  Plain pyre struct classdefs carry fields
    /// only, so this method-member path is exercised here for the first time.
    #[test]
    fn register_trait_family_getattr_yields_method_pbc_and_rtypes_via_methods_pbc_repr() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::rpbc::MethodsPBCRepr;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        use indexmap::IndexMap;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();

        // A method member is a user function backed by a GraphFunc — the
        // same shape `HostObject::new_class`'s `__dict__` would carry, and
        // what `add_source_attribute` turns into a MethodDesc.
        let member = |name: &str| -> ConstValue {
            ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                name.to_string(),
                Constant::new(ConstValue::Dict(Default::default())),
            )))
        };

        // Base trait `T` with a default-body method `shared`; three impls
        // A/B/C each override `shared`.  This is the ≥2-impl multi-impl
        // family shape that annotates classdef-less today.
        // The ordered container preserves the host class-dict insertion order.
        let mut base_members = IndexMap::new();
        base_members.insert("shared".to_string(), member("T::shared"));
        let mut impls = Vec::new();
        for impl_name in ["A", "B", "C"] {
            let mut m = IndexMap::new();
            m.insert(
                "shared".to_string(),
                member(&format!("{impl_name}::shared")),
            );
            impls.push((impl_name.to_string(), m));
        }
        let base_cd = bk
            .register_trait_family("T", base_members, impls)
            .expect("register_trait_family must succeed");

        // (a) base ↔ subclass links.
        assert_eq!(
            base_cd.borrow().subdefs.len(),
            3,
            "base classdef must link 3 impl subclasses"
        );

        // (b) getattr(instance_of_T, "shared") resolves to a MethodDesc PBC
        // family — NOT `SomeInstance(classdef=None)`, NOT "attribute not
        // found".  `s_getattr` on the base ClassDef is exactly what
        // `SomeInstance.getattr` (in `unaryop.rs`) calls for an instance of
        // that classdef; populate the attr across the tree first (the
        // fixpoint analog).
        ClassDef::check_missing_attribute_update(&base_cd, "shared")
            .expect("attr population must succeed");
        let s_shared = ClassDef::s_getattr(&base_cd, "shared", &std::collections::BTreeMap::new())
            .expect("s_getattr(shared) must resolve");
        let pbc = match &s_shared {
            SomeValue::PBC(p) => p.clone(),
            other => panic!("expected a method PBC for `shared`, got {other:?}"),
        };
        let n_methods = pbc
            .descriptions
            .values()
            .filter(|d| d.as_method().is_some())
            .count();
        assert!(
            n_methods >= 3,
            "expected >=3 MethodDescs in the `shared` family (the subclass \
             overrides), got {n_methods}: {pbc:?}"
        );

        // (c) the `MethodsPBCRepr` port — possibly never exercised on a
        // pyre-synthesized method PBC — must accept the family and resolve
        // its owner + bound-`self` InstanceRepr (the vtable dispatch seed).
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = MethodsPBCRepr::new(&rtyper, pbc)
            .expect("MethodsPBCRepr must accept the pyre-synthesized method family");
        assert_eq!(repr.methodname, "shared");
        assert!(
            Rc::ptr_eq(&repr.classdef, &base_cd),
            "MethodsPBCRepr must fold the family's owner to the base classdef"
        );
    }

    /// Diagnostic for the LinkedList wiring: the real dispatch blockers
    /// (`_get_2_values` / `head` / `dup`) are trait REQUIRED methods —
    /// declared on the trait, bodied only on the impl subclasses.  This
    /// probes a base-instance `getattr` of a method that exists ONLY on the
    /// subclasses (absent from the base member map).
    ///
    /// FINDING: it resolves to the full impl family anyway.  RPython's
    /// attrfamily merge homes a same-named attribute defined across sibling
    /// subclasses onto their common base, so `s_getattr(base, "req")` returns
    /// the {A,B,C}::req MethodDesc PBC even though the base declares no `req`.
    /// => piece-3 registration can seed the impls ONLY; it need NOT synthesize
    /// a base declaration for required (default-body-less) trait methods.
    #[test]
    fn register_trait_family_subclass_only_method_surfaces_on_base() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeValue;
        use indexmap::IndexMap;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let member = |name: &str| -> ConstValue {
            ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                name.to_string(),
                Constant::new(ConstValue::Dict(Default::default())),
            )))
        };

        // Base `T` declares NO `req`; each impl bodies `req` — the
        // required-method-with-no-default-body shape.
        let mut impls = Vec::new();
        for impl_name in ["A", "B", "C"] {
            // The ordered container preserves the host class-dict insertion order.
            let mut m = IndexMap::new();
            m.insert("req".to_string(), member(&format!("{impl_name}::req")));
            impls.push((impl_name.to_string(), m));
        }
        let base_cd = bk
            .register_trait_family("T", IndexMap::new(), impls)
            .expect("register_trait_family must succeed");

        ClassDef::check_missing_attribute_update(&base_cd, "req").expect("attr population");
        let s_req = ClassDef::s_getattr(&base_cd, "req", &std::collections::BTreeMap::new())
            .expect("s_getattr(req) must not error");
        let n_methods = match &s_req {
            SomeValue::PBC(p) => p
                .descriptions
                .values()
                .filter(|d| d.as_method().is_some())
                .count(),
            other => panic!("expected a method PBC for subclass-only `req`, got {other:?}"),
        };
        // The attrfamily merge surfaces the subclass-only method on the base:
        // one MethodDesc per impl, so a base-typed / trait-object receiver
        // dispatches across all impls without a base declaration.
        assert_eq!(
            n_methods, 3,
            "subclass-only `req` must surface all 3 impl MethodDescs on the \
             base via attrfamily merge, got {n_methods}: {s_req:?}"
        );
    }
}
