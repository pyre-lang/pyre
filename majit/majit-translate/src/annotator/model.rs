//! RPython type lattice — the `SomeValue` ↔ `SomeObject` hierarchy.
//!
//! RPython upstream: `rpython/annotator/model.py` (855 LOC).
//!
//! Commit split:
//!
//! * **A4.1 (this commit)** — Crate skeleton: `SomeValue` enum with
//!   the primitive-type variants declared, `SomeObjectTrait` shell
//!   with the property methods most touched by the annotator
//!   (`knowntype`, `immutable`, `is_constant`, `can_be_none`,
//!   `contains`). Individual `SomeXxx` structs exist as empty/default
//!   shells; method bodies that require the bookkeeper / classdef /
//!   union dispatch land in A4.2–A4.8.
//! * **A4.2** — Flesh out `SomeObject` base (const box, fmt helpers).
//! * **A4.3** — `SomeInteger` bounds (nonneg / unsigned / knowntype),
//!   `SomeBool`, `SomeFloat`, `SomeChar`, `SomeUnicodeCodePoint`.
//! * **A4.4** — `SomeString`, `SomeUnicodeString`, `SomeList`,
//!   `SomeTuple`, `SomeDict`, `SomeIterator`.
//! * **A4.5** — `SomeInstance`, `SomePBC`, `SomeBuiltin`,
//!   `SomeImpossibleValue`, `SomePtr`, `SomeAddress`,
//!   `SomeTypedAddressAccess`.
//! * **A4.6** — `union()` 2D match dispatch.
//! * **A4.7** — `unionof`, `read_can_only_throw`, `add_knowntypedata`.
//! * **A4.8** — Port `rpython/annotator/test/test_model.py` verbatim.
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Python `class SomeObject` + `class SomeInteger(SomeFloat)` +
//!   `class SomeBool(SomeInteger)` chain is modelled as a closed
//!   `SomeValue` **enum** with one struct variant per RPython
//!   subclass. RPython consumers pattern-match on `isinstance(s,
//!   SomeInteger)` thousands of times; Rust `match` on `SomeValue` is
//!   the direct mirror. Method contracts shared by every variant live
//!   on the [`SomeObjectTrait`] trait, which [`SomeValue`] forwards to
//!   via a delegating match.
//!
//! * RPython metaclass magic (`extendabletype`, `doubledispatch`, the
//!   `ConstAccessDelegator` for `self.const` → `self.const_box.value`)
//!   collapses into direct Rust fields — `const_box: Option<Constant>`
//!   is the same thing.
//!
//! * `intersection(s1, s2)` / `difference(s1, s2)` double-dispatch
//!   registries (annotator/model.py) become free functions in
//!   A4.6 with a single `match (SomeValue, SomeValue)` body.

use core::fmt;
use std::cell::RefCell;
use std::rc::Rc;

use super::super::flowspace::model::{BlockKey, BlockRef, Constant, HostObject, Variable};
use super::bookkeeper::Bookkeeper;
use super::classdesc::ClassDef;
pub use crate::translator::rtyper::llannotation::{SomeInteriorPtr, SomeLLADTMeth};

// State / TLS (model.py).

/// RPython `class State(object)` (model.py).
///
/// ```python
/// class State(object):
///     # A global attribute :-(  Patch it with 'True' to enable checking of
///     # the no_nul attribute...
///     check_str_without_nul = False
///     allow_int_to_float = True
/// TLS = State()
/// ```
///
/// The `bookkeeper` slot is assigned dynamically upstream (bookkeeper.py:89
/// — `TLS.bookkeeper = self` inside `Bookkeeper.enter`). Rust can't add
/// attributes at runtime, so the slot is declared explicitly here and
/// `Bookkeeper::enter` / `leave` set / clear it.
pub struct State {
    /// RPython `State.check_str_without_nul` (model.py).
    pub check_str_without_nul: bool,
    /// RPython `State.allow_int_to_float` (model.py).
    pub allow_int_to_float: bool,
    /// RPython `TLS.bookkeeper` (set dynamically by `Bookkeeper.enter`).
    pub bookkeeper: Option<Rc<Bookkeeper>>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub const fn new() -> Self {
        State {
            check_str_without_nul: false,
            allow_int_to_float: true,
            bookkeeper: None,
        }
    }
}

thread_local! {
    /// RPython `TLS = State()` (model.py).
    ///
    /// A single process-wide `State` singleton upstream — modelled as
    /// a `thread_local!` here because `Rc<Bookkeeper>` is `!Send`. The
    /// annotator runs single-threaded, so this matches upstream
    /// semantics exactly.
    pub static TLS: RefCell<State> = const { RefCell::new(State::new()) };
}

// KnownType — mirror of upstream `SomeObject.knowntype` class attribute.

/// RPython stores `knowntype` as a live Python type object
/// (`int`, `float`, `str`, the classdef's `classdesc.pyobj`, …).
/// The Rust port carries the same information as an enum tag so that
/// `s.knowntype()` round-trips through `match` without pulling in
/// `HostObject` here.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KnownType {
    /// `object` — the RPython `SomeObject` fallback.
    Object,
    /// `type` — the RPython `SomeType.knowntype = type`.
    Type,
    /// `int` — the RPython `SomeInteger.knowntype = int`, modulo the
    /// `SomeInteger.__init__` branch that promotes `unsigned=True` to
    /// `r_uint`.
    Int,
    /// `r_uint` — the RPython sister of `int`, set when
    /// `SomeInteger.unsigned` is true.
    Ruint,
    /// `r_longlong` — RPython's signed 64-bit integer type.
    LongLong,
    /// `r_longlonglong` — RPython's signed extended integer type.
    LongLongLong,
    /// `r_ulonglong` — RPython's unsigned 64-bit integer type.
    ULongLong,
    /// `r_ulonglonglong` — RPython's unsigned extended integer type.
    ULongLongLong,
    /// `bool` — the RPython `SomeBool.knowntype = bool`.
    Bool,
    /// `float` — the RPython `SomeFloat.knowntype = float`.
    Float,
    /// `r_singlefloat` — the RPython `SomeSingleFloat.knowntype`.
    Singlefloat,
    /// `r_longfloat` — the RPython `SomeLongFloat.knowntype`.
    Longfloat,
    /// `str` — the RPython `SomeString.knowntype = str`.
    Str,
    /// `unicode` — the RPython `SomeUnicodeString.knowntype = unicode`.
    Unicode,
    /// `bytearray` — the RPython `SomeByteArray.knowntype = bytearray`.
    Bytearray,
    /// `NoneType` — the RPython `SomeNone.knowntype = type(None)`.
    NoneType,
    /// `builtin_function_or_method` — the RPython
    /// `SomeBuiltin.knowntype = BuiltinFunctionType`.
    BuiltinFunctionOrMethod,
    /// `property` — the RPython `SomeProperty.knowntype = type(property)`.
    PropertyType,
    /// `ReferenceType` — the RPython
    /// `SomeWeakRef.knowntype = weakref.ReferenceType`.
    WeakrefReference,
    /// `lltype._ptr` — the RPython `SomePtr.knowntype = _ptr`.
    LlPtr,
    /// `llmemory.Address` — the RPython `SomeAddress.knowntype`.
    Address,
    /// `list` — the RPython `SomeList.knowntype = list`.
    List,
    /// `tuple` — the RPython `SomeTuple.knowntype = tuple`.
    Tuple,
    /// `dict` — the RPython `SomeDict.knowntype = dict` before the
    /// `SomeDict = SomeOrderedDict` rebind (`model.py`).
    Dict,
    /// `collections.OrderedDict` — `SomeOrderedDict.knowntype` after
    /// `SomeDict = SomeOrderedDict` (`model.py`).
    OrderedDict,
    /// `types.FunctionType` — `FunctionDesc.knowntype` (`description.py`).
    FunctionType,
    /// `types.MethodType` — `MethodDesc.knowntype` / `MethodOfFrozenDesc.knowntype`
    /// (`description.py`).
    MethodType,
    /// `type(iter([]))` — `SomeIterator.knowntype` (`model.py`). Python 3
    /// names this class `list_iterator`.
    Iterator,
    /// `classdef.classdesc` — the RPython `SomeInstance.knowntype`.
    /// The `usize` is `ClassDesc.identity`; `None` when `classdef` is
    /// `None` (`knowntype = None` upstream).
    Instance(Option<usize>),
    /// Residual knowntype this port has not yet given its own tag
    /// (`FrozenDesc.knowntype = type(pyobj)`, …).
    Other,
}

impl fmt::Display for KnownType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KnownType::Object => f.write_str("object"),
            KnownType::Type => f.write_str("type"),
            KnownType::Int => f.write_str("int"),
            KnownType::Ruint => f.write_str("r_uint"),
            KnownType::LongLong => f.write_str("r_longlong"),
            KnownType::LongLongLong => f.write_str("r_longlonglong"),
            KnownType::ULongLong => f.write_str("r_ulonglong"),
            KnownType::ULongLongLong => f.write_str("r_ulonglonglong"),
            KnownType::Bool => f.write_str("bool"),
            KnownType::Float => f.write_str("float"),
            KnownType::Singlefloat => f.write_str("r_singlefloat"),
            KnownType::Longfloat => f.write_str("r_longfloat"),
            KnownType::Str => f.write_str("str"),
            KnownType::Unicode => f.write_str("unicode"),
            KnownType::Bytearray => f.write_str("bytearray"),
            KnownType::NoneType => f.write_str("NoneType"),
            KnownType::BuiltinFunctionOrMethod => f.write_str("builtin_function_or_method"),
            KnownType::PropertyType => f.write_str("property"),
            KnownType::WeakrefReference => f.write_str("ReferenceType"),
            KnownType::LlPtr => f.write_str("_ptr"),
            KnownType::Address => f.write_str("Address"),
            KnownType::List => f.write_str("list"),
            KnownType::Tuple => f.write_str("tuple"),
            KnownType::Dict => f.write_str("dict"),
            KnownType::OrderedDict => f.write_str("OrderedDict"),
            KnownType::FunctionType => f.write_str("function"),
            KnownType::MethodType => f.write_str("method"),
            KnownType::Iterator => f.write_str("list_iterator"),
            // `SomeInstance.knowntype` is the classdesc, or `None`.
            KnownType::Instance(None) => f.write_str("None"),
            KnownType::Instance(Some(id)) => write!(f, "classdesc{id}"),
            KnownType::Other => f.write_str("<other>"),
        }
    }
}

/// RPython `commonbase(cls1, cls2)` (model.py), specialized to
/// the Rust port's [`KnownType`] carrier for live Python type objects.
pub fn commonbase(cls1: KnownType, cls2: KnownType) -> KnownType {
    if cls1 == cls2 {
        return cls1;
    }
    match (cls1, cls2) {
        // Python `bool` subclasses `int`.
        (KnownType::Bool, KnownType::Int) | (KnownType::Int, KnownType::Bool) => KnownType::Int,
        // Unequal tags, including two `Instance` classdescs, share only
        // `object`. Class-hierarchy `commonbase` lives on `ClassDesc`.
        _ => KnownType::Object,
    }
}

/// True when two `ClassDef` name strings denote ONE struct reached by two
/// frontend name spellings: a dot-joined crate-included constructor qualname
/// (`pyre_object.celldict.VersionTag`, the flowspace_adapter
/// `SyntheticTransparentCtor` arm) vs the crate-stripped `::` field-read
/// spelling (`celldict::VersionTag`).  Pyre has no single class object to key
/// identity on, so a non-header leaf struct minted from both spellings becomes
/// two distinct base-less `ClassDef`s and `ClassDef::commonbase` finds no
/// shared base — the union then fails "no common base class" even though it is
/// one type.  Reduce the dotted crate-included spelling to the crate-stripped
/// `::` form (the spelling `canonical_struct_name` already yields for the
/// field-read side) and compare.
///
/// Constructors and field reads now normalize at
/// `Bookkeeper::intern_class_by_qualname`, including header-bearing structs;
/// normalization keeps the crate-relative module path so their full base-chain
/// walk still runs.  Keep this compare-time fallback for registry-absent or
/// ambiguous constructor spellings: it fires only where `commonbase` is
/// already None.
///
/// It runs the intern's own key derivation — [`normalize_class_qualname`] then
/// `canonical_struct_name` — rather than a second rule of its own.  A separate
/// rule disagreed on exactly the spelling the intern was written for: the local
/// dotted-only strip answered `false` for
/// `("pyre_object::m::T", "m::T")` while the intern maps both to `m::T`.
/// Upstream keys identity on the object in one place
/// (`bookkeeper.py`), so one derivation is the shape to hold.
///
/// Caution carried forward: collapsing an `ob_header`-bearing `W_*` struct
/// EARLY — that is, keying `struct_root_classes` on the collapse — starved its
/// base-chain walk, so it minted base-less under first-mint-wins and lost its
/// `W_Root` base, a measured +184 phaseA cascade across `_io`/`_pickle`.  The
/// normalization above is a crate-root strip only; it does not collapse the
/// module path, so the walk still runs.  Any future widening of it has to
/// re-measure that number.
/// `option::Option<??TypeVar>::None`: the payload type variable was not
/// substituted, so the variant stayed a class instead of a null pointer.
fn unresolved_option_none_name(name: &str) -> bool {
    name.contains("Option<") && name.contains("??") && name.ends_with("::None")
}

fn same_struct_identity(a: &str, b: &str) -> bool {
    fn identity(name: &str) -> String {
        majit_ir::descr::canonical_struct_name(
            &crate::annotator::bookkeeper::normalize_class_qualname(name),
        )
    }
    identity(a) == identity(b)
}

// SomeObject base — RPython `model.py`.

/// RPython `class SomeObject(object)` (model.py).
///
/// Acts both as the "universal" annotation (`object`) and as the
/// shared base-state carried by every subclass via composition. The
/// `SomeValue::Object` enum variant wraps this struct directly; every
/// other `SomeXxx` struct embeds it as `pub base: SomeObject` so
/// upstream's `super().__init__()` behaviour is preserved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeObject {
    /// RPython `self.const_box`. Populated via the
    /// `ConstAccessDelegator` when `self.const = xyz` is assigned;
    /// `is_constant()` returns true whenever this is `Some`.
    pub const_box: Option<Constant>,
    /// RPython `self.knowntype` — fetched via [`Self::knowntype`].
    pub knowntype: KnownType,
    /// RPython `self.immutable` — class attribute, not usually reset
    /// per-instance. Carried on the base so `SomeObjectTrait::immutable`
    /// is a trivial accessor.
    pub immutable: bool,
}

impl SomeObject {
    pub fn new(knowntype: KnownType, immutable: bool) -> Self {
        SomeObject {
            const_box: None,
            knowntype,
            immutable,
        }
    }
}

impl Default for SomeObject {
    fn default() -> Self {
        // RPython `SomeObject` defaults: `knowntype = object`,
        // `immutable = False`, no const box.
        SomeObject::new(KnownType::Object, false)
    }
}

// SomeObjectTrait — the method contract shared by every Some* variant.

/// Methods every `SomeValue` variant forwards. Mirrors the subset of
/// upstream `SomeObject` methods that the annotator calls through the
/// Python class MRO rather than via `isinstance` dispatch. Further
/// methods (`noneify`, `nonnoneify`, `set_knowntypedata`, …) land in
/// A4.6 alongside the union dispatch.
pub trait SomeObjectTrait {
    /// RPython `s.knowntype` (class attribute or `__init__` override).
    fn knowntype(&self) -> KnownType;

    /// RPython `s.immutable` class attribute.
    fn immutable(&self) -> bool;

    /// RPython `s.is_constant()` (model.py). True iff the
    /// underlying const slot carries a wrapped Python value.
    fn is_constant(&self) -> bool;

    /// RPython `s.is_immutable_constant()` (model.py).
    fn is_immutable_constant(&self) -> bool {
        self.immutable() && self.is_constant()
    }

    /// RPython `s.can_be_none()` (model.py).
    fn can_be_none(&self) -> bool;
}

// Concrete Some* variants (A4.1 shells).

/// RPython `class SomeType(SomeObject)` (model.py).
/// Stands for a `type` value; upstream sets `can_be_none = False`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeType {
    pub base: SomeObject,
}

impl SomeType {
    pub fn new() -> Self {
        SomeType {
            base: SomeObject::new(KnownType::Type, true),
        }
    }
}

impl Default for SomeType {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeType {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeFloat(SomeObject)` (model.py).
/// Stands for a float (or, when `allow_int_to_float` is set, an int
/// promoted to float).
#[derive(Clone, Debug)]
pub struct SomeFloat {
    pub base: SomeObject,
}

impl SomeFloat {
    pub fn new() -> Self {
        SomeFloat {
            base: SomeObject::new(KnownType::Float, true),
        }
    }

    fn constant_float_bits(&self) -> Option<u64> {
        match self.base.const_box.as_ref().map(|c| &c.value) {
            Some(super::super::flowspace::model::ConstValue::Float(bits)) => Some(*bits),
            _ => None,
        }
    }
}

impl PartialEq for SomeFloat {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(lhs_bits), Some(rhs_bits)) =
            (self.constant_float_bits(), other.constant_float_bits())
        {
            let lhs = f64::from_bits(lhs_bits);
            let rhs = f64::from_bits(rhs_bits);
            if lhs.is_nan() && rhs.is_nan() {
                return true;
            }
            if lhs == 0.0 && rhs == 0.0 {
                return lhs_bits == rhs_bits;
            }
        }
        self.base == other.base
    }
}

impl Eq for SomeFloat {}

impl Default for SomeFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeFloat {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeSingleFloat(SomeObject)` (model.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeSingleFloat {
    pub base: SomeObject,
}

impl SomeSingleFloat {
    pub fn new() -> Self {
        SomeSingleFloat {
            base: SomeObject::new(KnownType::Singlefloat, true),
        }
    }
}

impl Default for SomeSingleFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeSingleFloat {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeLongFloat(SomeObject)` (model.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeLongFloat {
    pub base: SomeObject,
}

impl SomeLongFloat {
    pub fn new() -> Self {
        SomeLongFloat {
            base: SomeObject::new(KnownType::Longfloat, true),
        }
    }
}

impl Default for SomeLongFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeLongFloat {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeInteger(SomeFloat)` (model.py).
///
/// Carries `nonneg` / `unsigned` flags and a `knowntype` that can be
/// `int` or one of the `r_uint` / `r_long` family. For A4.1 we stub
/// the flags as `bool` + track the distinction via [`KnownType`]; the
/// `base_int` subclass hierarchy lands with the rlib/rarithmetic port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeInteger {
    pub base: SomeObject,
    /// RPython `self.nonneg` — known to be ≥ 0.
    pub nonneg: bool,
    /// RPython `self.unsigned` — `knowntype is r_uint`.
    pub unsigned: bool,
    /// Branch-refinement facts for an integer-valued exitswitch, keyed
    /// by the case constant ([`ExitCaseKey::Int`]). Upstream defines
    /// `set_knowntypedata` only on `SomeBool` (model.py), since
    /// `SomeBool(SomeInteger)` and its only multi-way narrowing comes
    /// from boolean `is`/`isinstance`/comparison ops. The exitswitch
    /// consumer reads it generically — `getattr(annotation,
    /// "knowntypedata", {})` (annrpython.py) — so an integer
    /// discriminant `SwitchInt` carries its per-case constraints here.
    pub knowntypedata: Option<KnownTypeData>,
}

impl SomeInteger {
    /// RPython `SomeInteger.__init__(nonneg=False, unsigned=None,
    /// knowntype=None)` (model.py:211-224).
    pub fn new(nonneg: bool, unsigned: bool) -> Self {
        let knowntype = if unsigned {
            KnownType::Ruint
        } else {
            KnownType::Int
        };
        SomeInteger {
            base: SomeObject::new(knowntype, true),
            // upstream: `self.nonneg = unsigned or nonneg`.
            nonneg: unsigned || nonneg,
            unsigned,
            knowntypedata: None,
        }
    }

    /// Construct with an explicit [`KnownType`] — mirrors RPython's
    /// `SomeInteger(knowntype=...)` keyword call used by binaryop.py's
    /// `compute_restype(...)` path (binaryop.py:201-202, :218-219).
    ///
    /// upstream model.py:211-224 derives `unsigned` from the knowntype
    /// itself (`knowntype(-1) > 0`) and raises
    /// `TypeError('Conflicting specification for SomeInteger')` when
    /// an explicit `unsigned` is passed alongside a non-None
    /// `knowntype`. The Rust port follows that contract: `unsigned`
    /// is strictly derived, so callers cannot pin both signedness and
    /// `KnownType::Ruint` to contradictory values.
    pub fn new_with_knowntype(nonneg: bool, knowntype: KnownType) -> Self {
        // upstream: `unsigned = self.knowntype(-1) > 0`. The unsigned
        // rarithmetic integer families treat -1 as a positive value;
        // all other integer KnownTypes are signed.
        let unsigned = matches!(
            knowntype,
            KnownType::Ruint | KnownType::ULongLong | KnownType::ULongLongLong
        );
        SomeInteger {
            base: SomeObject::new(knowntype, true),
            nonneg: unsigned || nonneg,
            unsigned,
            knowntypedata: None,
        }
    }

    /// RPython `SomeBool.set_knowntypedata` (model.py), reused
    /// for an integer discriminant. Same set-once assertion and
    /// falsy-inner-dict pruning as [`SomeBool::set_knowntypedata`].
    pub fn set_knowntypedata(&mut self, mut data: KnownTypeData) {
        assert!(
            self.knowntypedata.is_none(),
            "assert not hasattr(self, 'knowntypedata')"
        );
        data.retain(|_case, inner| !inner.is_empty());
        if !data.is_empty() {
            self.knowntypedata = Some(data);
        }
    }
}

impl Default for SomeInteger {
    fn default() -> Self {
        Self::new(false, false)
    }
}

impl SomeObjectTrait for SomeInteger {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeBool(SomeInteger)` (model.py).
///
/// Upstream `SomeBool.__init__` takes no args — the class attributes
/// fix `knowntype = bool`, `nonneg = True`, `unsigned = False`.
/// Key for [`KnownTypeData`] — the branch's `exitcase` value, the
/// typed Rust projection of RPython's generic exitcase key
/// (`knowntypedata.get(link.exitcase, {})`, annrpython.py). A bool
/// exitswitch keys by the branch truth value; an integer exitswitch —
/// e.g. an enum discriminant `SwitchInt` — keys by the case constant.
/// RPython keys directly by the Python exitcase object (`True`/`False`/
/// an int); the sum type makes the two cases explicit and hashable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExitCaseKey {
    Bool(bool),
    Int(i128),
}

/// RPython `knowntypedata` carrier type — `defaultdict(dict)` keyed by
/// the branch `exitcase` value ([`ExitCaseKey`]), inner dict keyed by
/// [`Variable`] identity. Populated by [`add_knowntypedata`] and
/// merged by [`merge_knowntypedata`].
pub type KnownTypeData =
    std::collections::HashMap<ExitCaseKey, std::collections::HashMap<Rc<Variable>, SomeValue>>;

/// `knowntypedata` (set_knowntypedata in model.py) stores the
/// branch-refinement facts for a bool-valued variable: a map from the
/// boolean truth value to the variables whose annotation can be
/// narrowed (and what to narrow to) in that branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeBool {
    pub base: SomeObject,
    /// RPython `self.knowntypedata` (model.py). Absent when
    /// `set_knowntypedata` was never called or when all branches were
    /// pruned. Upstream uses a `defaultdict(dict)` keyed by `bool`;
    /// the Rust port keeps the inner map empty when there is nothing
    /// to refine.
    pub knowntypedata: Option<KnownTypeData>,
}

impl SomeBool {
    pub fn new() -> Self {
        SomeBool {
            base: SomeObject::new(KnownType::Bool, true),
            knowntypedata: None,
        }
    }

    /// RPython `SomeBool.set_knowntypedata` (model.py).
    ///
    /// Assertion + falsy-inner-dict pruning: drop any truth key whose
    /// inner dict is empty, then store only if the outer dict is
    /// non-empty. Upstream asserts that `knowntypedata` is set at most
    /// once per SomeBool instance.
    pub fn set_knowntypedata(&mut self, mut data: KnownTypeData) {
        assert!(
            self.knowntypedata.is_none(),
            "assert not hasattr(self, 'knowntypedata')"
        );
        data.retain(|_truth, inner| !inner.is_empty());
        if !data.is_empty() {
            self.knowntypedata = Some(data);
        }
    }
}

impl Default for SomeBool {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeBool {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeStringOrUnicode(SomeObject)` (model.py).
///
/// Shared state for `SomeString`, `SomeUnicodeString`, `SomeByteArray`.
/// A4.1 stores the two flag bits directly; the full `nonnulify` /
/// `nonnoneify` transformer methods land with A4.4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeStringOrUnicode {
    pub base: SomeObject,
    /// RPython `self.can_be_None`.
    pub can_be_none: bool,
    /// RPython `self.no_nul`.
    pub no_nul: bool,
}

impl SomeStringOrUnicode {
    pub fn new(knowntype: KnownType, immutable: bool, can_be_none: bool, no_nul: bool) -> Self {
        SomeStringOrUnicode {
            base: SomeObject::new(knowntype, immutable),
            can_be_none,
            no_nul,
        }
    }
}

/// RPython `class SomeString(SomeStringOrUnicode)` (model.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeString {
    pub inner: SomeStringOrUnicode,
}

impl SomeString {
    pub fn new(can_be_none: bool, no_nul: bool) -> Self {
        SomeString {
            inner: SomeStringOrUnicode::new(KnownType::Str, true, can_be_none, no_nul),
        }
    }
}

impl Default for SomeString {
    fn default() -> Self {
        Self::new(false, false)
    }
}

impl SomeObjectTrait for SomeString {
    fn knowntype(&self) -> KnownType {
        self.inner.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.inner.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        self.inner.can_be_none
    }
}

/// RPython `class SomeUnicodeString(SomeStringOrUnicode)`
/// (model.py:296-302).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeUnicodeString {
    pub inner: SomeStringOrUnicode,
}

impl SomeUnicodeString {
    pub fn new(can_be_none: bool, no_nul: bool) -> Self {
        SomeUnicodeString {
            inner: SomeStringOrUnicode::new(KnownType::Unicode, true, can_be_none, no_nul),
        }
    }
}

impl Default for SomeUnicodeString {
    fn default() -> Self {
        Self::new(false, false)
    }
}

impl SomeObjectTrait for SomeUnicodeString {
    fn knowntype(&self) -> KnownType {
        self.inner.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.inner.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        self.inner.can_be_none
    }
}

/// RPython `class SomeStringBuilder(SomeObject)` (`rlib/rstring.py`).
///
/// The annotator shape for a `StringBuilder()` value.  It carries no
/// payload; its `method_*` annotations describe the builder call surface,
/// and (once wired) the rtyper binds it to `StringBuilderRepr` via
/// `rtyper_makerepr` (`rlib/rstring.py`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeStringBuilder {
    pub base: SomeObject,
}

impl SomeStringBuilder {
    pub fn new() -> Self {
        SomeStringBuilder {
            base: SomeObject::new(KnownType::Object, false),
        }
    }

    /// `method_append(self, s_str)` (rstring.py) → `s_None`.
    /// `AbstractStringBuilderRepr.rtype_method_append` also accepts
    /// `SomeUnicodeCodePoint` (`ll_append_char`).
    pub fn method_append(&self, s_str: &SomeValue) -> SomeValue {
        assert!(
            matches!(
                s_str,
                SomeValue::None_(_)
                    | SomeValue::String(_)
                    | SomeValue::Char(_)
                    | SomeValue::UnicodeCodePoint(_)
            ),
            "StringBuilder.append expects str / char"
        );
        s_none()
    }

    /// `method_append_slice(self, s_str, s_start, s_end)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_slice(
        &self,
        s_str: &SomeValue,
        s_start: &SomeValue,
        s_end: &SomeValue,
    ) -> SomeValue {
        assert!(matches!(s_str, SomeValue::None_(_) | SomeValue::String(_)));
        assert!(matches!(s_start, SomeValue::Integer(_)));
        assert!(matches!(s_end, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_append_multiple_char(self, s_char, s_times)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_multiple_char(
        &self,
        s_char: &SomeValue,
        s_times: &SomeValue,
    ) -> SomeValue {
        assert!(matches!(s_char, SomeValue::String(_) | SomeValue::Char(_)));
        assert!(matches!(s_times, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_append_charpsize(self, s_ptr, s_size)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_charpsize(&self, s_ptr: &SomeValue, s_size: &SomeValue) -> SomeValue {
        assert!(matches!(s_ptr, SomeValue::Ptr(_)));
        assert!(matches!(s_size, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_getlength(self)` (rstring.py) →
    /// `SomeInteger(nonneg=True)`.
    pub fn method_getlength(&self) -> SomeValue {
        SomeValue::Integer(SomeInteger::new(true, false))
    }

    /// `method_build(self)` (rstring.py) →
    /// `SomeString(can_be_None=False)`.
    pub fn method_build(&self) -> SomeValue {
        SomeValue::String(SomeString::new(false, false))
    }
}

impl Default for SomeStringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `class SomeUnicodeBuilder(SomeObject)` (`rlib/rstring.py`).
/// The `UnicodeBuilder()` mirror of [`SomeStringBuilder`]; the char /
/// string arguments and `method_build` result are the unicode variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeUnicodeBuilder {
    pub base: SomeObject,
}

impl SomeUnicodeBuilder {
    pub fn new() -> Self {
        SomeUnicodeBuilder {
            base: SomeObject::new(KnownType::Object, false),
        }
    }

    /// `method_append(self, s_str)` (rstring.py) → `s_None`.
    pub fn method_append(&self, s_str: &SomeValue) -> SomeValue {
        assert!(
            matches!(
                s_str,
                SomeValue::None_(_) | SomeValue::UnicodeCodePoint(_) | SomeValue::UnicodeString(_)
            ),
            "UnicodeBuilder.append expects unicode / unichar"
        );
        s_none()
    }

    /// `method_append_slice(self, s_str, s_start, s_end)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_slice(
        &self,
        s_str: &SomeValue,
        s_start: &SomeValue,
        s_end: &SomeValue,
    ) -> SomeValue {
        assert!(matches!(
            s_str,
            SomeValue::None_(_) | SomeValue::UnicodeString(_)
        ));
        assert!(matches!(s_start, SomeValue::Integer(_)));
        assert!(matches!(s_end, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_append_multiple_char(self, s_char, s_times)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_multiple_char(
        &self,
        s_char: &SomeValue,
        s_times: &SomeValue,
    ) -> SomeValue {
        assert!(matches!(s_char, SomeValue::UnicodeCodePoint(_)));
        assert!(matches!(s_times, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_append_charpsize(self, s_ptr, s_size)`
    /// (rstring.py) → `s_None`.
    pub fn method_append_charpsize(&self, s_ptr: &SomeValue, s_size: &SomeValue) -> SomeValue {
        assert!(matches!(s_ptr, SomeValue::Ptr(_)));
        assert!(matches!(s_size, SomeValue::Integer(_)));
        s_none()
    }

    /// `method_getlength(self)` (rstring.py) →
    /// `SomeInteger(nonneg=True)`.
    pub fn method_getlength(&self) -> SomeValue {
        SomeValue::Integer(SomeInteger::new(true, false))
    }

    /// `method_build(self)` (rstring.py) →
    /// `SomeUnicodeString(can_be_None=False)`.
    pub fn method_build(&self) -> SomeValue {
        SomeValue::UnicodeString(SomeUnicodeString::new(false, false))
    }
}

impl Default for SomeUnicodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeStringBuilder {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        // upstream: default `SomeObject.can_be_none()` = True, no
        // override on SomeStringBuilder.
        true
    }
}

impl SomeObjectTrait for SomeUnicodeBuilder {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        // upstream: default `SomeObject.can_be_none()` = True, no
        // override on SomeUnicodeBuilder.
        true
    }
}

/// A Rust `alloc::vec::Vec<T>` with one-word items (`majit_ir::rvec`).
///
/// The value is a pointer to the raw `{ptr, len, cap}` header, not an RPython
/// list, so it is a separate annotation from [`SomeList`] and never unions
/// with one. The item annotation is fixed by the Rust item type; two
/// `SomeRustVec`s union their items.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeRustVec {
    pub base: SomeObject,
    pub s_item: Box<SomeValue>,
}

impl SomeRustVec {
    pub fn new(s_item: SomeValue) -> Self {
        SomeRustVec {
            base: SomeObject::new(KnownType::Object, false),
            s_item: Box::new(s_item),
        }
    }

    /// The `SomeRustVec` whose items are the one-word values of `kind`: a
    /// signed integer, a float, or a classdef-less instance (a managed object
    /// reference whose class the Rust item type does not name).
    pub fn for_kind(kind: majit_ir::rvec::VecItemKind) -> Self {
        use majit_ir::rvec::VecItemKind;
        let s_item = match kind {
            VecItemKind::Int => SomeValue::Integer(SomeInteger::new(false, false)),
            VecItemKind::Float => SomeValue::Float(SomeFloat::new()),
            VecItemKind::Ref => SomeValue::Instance(SomeInstance::new(
                None,
                false,
                std::collections::BTreeMap::new(),
            )),
        };
        SomeRustVec::new(s_item)
    }

    /// Register kind of the items: an integer is `Int`, a float `Float`, an
    /// instance (a managed object reference) `Ref`.
    pub fn item_kind(&self) -> Option<majit_ir::rvec::VecItemKind> {
        use majit_ir::rvec::VecItemKind;
        match &*self.s_item {
            SomeValue::Integer(_) => Some(VecItemKind::Int),
            SomeValue::Float(_) => Some(VecItemKind::Float),
            SomeValue::Instance(_) => Some(VecItemKind::Ref),
            _ => None,
        }
    }
}

impl SomeObjectTrait for SomeRustVec {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeByteArray(SomeStringOrUnicode)`
/// (model.py:304-306). Differs from its siblings in `immutable = False`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeByteArray {
    pub inner: SomeStringOrUnicode,
}

impl SomeByteArray {
    pub fn new(can_be_none: bool) -> Self {
        SomeByteArray {
            // `no_nul` is asserted to require `immutable`; bytearray
            // is mutable, so `no_nul` is always False here.
            inner: SomeStringOrUnicode::new(KnownType::Bytearray, false, can_be_none, false),
        }
    }
}

impl Default for SomeByteArray {
    fn default() -> Self {
        Self::new(false)
    }
}

impl SomeObjectTrait for SomeByteArray {
    fn knowntype(&self) -> KnownType {
        self.inner.base.knowntype
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.inner.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        self.inner.can_be_none
    }
}

/// RPython `class SomeChar(SomeString)` (model.py).
///
/// A character is a length-1 string with `can_be_None = False`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeChar {
    pub inner: SomeStringOrUnicode,
}

impl SomeChar {
    pub fn new(no_nul: bool) -> Self {
        SomeChar {
            inner: SomeStringOrUnicode::new(KnownType::Str, true, false, no_nul),
        }
    }
}

impl Default for SomeChar {
    fn default() -> Self {
        Self::new(false)
    }
}

impl SomeObjectTrait for SomeChar {
    fn knowntype(&self) -> KnownType {
        self.inner.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.inner.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeUnicodeCodePoint(SomeUnicodeString)`
/// (model.py:318-324).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeUnicodeCodePoint {
    pub inner: SomeStringOrUnicode,
}

impl SomeUnicodeCodePoint {
    pub fn new(no_nul: bool) -> Self {
        SomeUnicodeCodePoint {
            inner: SomeStringOrUnicode::new(KnownType::Unicode, true, false, no_nul),
        }
    }
}

impl Default for SomeUnicodeCodePoint {
    fn default() -> Self {
        Self::new(false)
    }
}

impl SomeObjectTrait for SomeUnicodeCodePoint {
    fn knowntype(&self) -> KnownType {
        self.inner.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.inner.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// Container-family Some* variants (A4.4).
//
// upstream: `class SomeList(SomeObject)` (model.py),
// `class SomeTuple(SomeObject)` (model.py),
// `class SomeDict(SomeObject)` / `SomeOrderedDict` (model.py),
// `class SomeIterator(SomeObject)` (model.py).
//
// The upstream `listdef.ListDef` / `dictdef.DictDef` classes live in
// rpython/annotator/listdef.py + dictdef.py and carry classdef /
// bookkeeper references; the real implementations live in
// [`super::listdef`] / [`super::dictdef`] and are re-exported here
// so model.rs stays the single import surface for `SomeList` /
// `SomeDict`-wielding code.

/// RPython `rpython/annotator/listdef.py:ListDef` — re-export of
/// [`super::listdef::ListDef`].
pub use super::listdef::ListDef;

/// RPython `rpython/annotator/dictdef.py:DictDef` — re-export of
/// [`super::dictdef::DictDef`].
pub use super::dictdef::DictDef;

/// RPython `class SomeList(SomeObject)` (model.py).
/// Homogeneous list of unknown length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeList {
    pub base: SomeObject,
    pub listdef: ListDef,
}

impl SomeList {
    pub fn new(listdef: ListDef) -> Self {
        SomeList {
            base: SomeObject::new(KnownType::List, false),
            listdef,
        }
    }
}

impl SomeObjectTrait for SomeList {
    fn knowntype(&self) -> KnownType {
        // `SomeList.knowntype = list`.
        KnownType::List
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        // upstream: `def can_be_none(self): return True`.
        true
    }
}

/// RPython `class SomeTuple(PureOperation-alike, SomeObject)`
/// (model.py:357-371). Fixed-length tuple; when every element is
/// constant, `self.const` becomes the tuple of constants. We track
/// elements directly; the `self.const` shortcut is computed on demand
/// via `is_constant()` + `SomeObject.const_box`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeTuple {
    pub base: SomeObject,
    pub items: Vec<SomeValue>,
}

impl SomeTuple {
    pub fn new(items: Vec<SomeValue>) -> Self {
        fn tuple_const_value(items: &[SomeValue]) -> Option<crate::flowspace::model::ConstValue> {
            use crate::flowspace::model::ConstValue;

            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let value = if let Some(value) = item.const_() {
                    value.clone()
                } else if let SomeValue::Tuple(t) = item {
                    t.base.const_box.as_ref().map(|c| c.value.clone())?
                } else {
                    return None;
                };
                out.push(value);
            }
            Some(ConstValue::Tuple(out))
        }

        let mut base = SomeObject::new(KnownType::Tuple, true);
        if let Some(value) = tuple_const_value(&items) {
            base.const_box = Some(Constant::new(value));
        }
        SomeTuple { base, items }
    }
}

impl SomeObjectTrait for SomeTuple {
    fn knowntype(&self) -> KnownType {
        // `SomeTuple.knowntype = tuple`.
        KnownType::Tuple
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        // upstream: `SomeTuple.__init__` sets `self.const` when every
        // item is constant. We mirror that by reading through the
        // items.
        self.base.const_box.is_some() || self.items.iter().all(|i| i.is_constant())
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeDict(SomeObject)` (model.py) — after the
/// `SomeDict = SomeOrderedDict` assignment at model.py:416, every dict
/// annotation is ordered. We collapse that into a single `SomeDict`
/// variant per CLAUDE.md parity rule #1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeDict {
    pub base: SomeObject,
    pub dictdef: DictDef,
}

impl SomeDict {
    pub fn new(dictdef: DictDef) -> Self {
        SomeDict {
            base: SomeObject::new(KnownType::OrderedDict, false),
            dictdef,
        }
    }
}

impl SomeObjectTrait for SomeDict {
    fn knowntype(&self) -> KnownType {
        // After `SomeDict = SomeOrderedDict` (`model.py`), every dict
        // annotation carries `SomeOrderedDict.knowntype = OrderedDict`.
        KnownType::OrderedDict
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        true
    }
}

/// RPython `SomeDict = SomeOrderedDict` (model.py).
pub type SomeOrderedDict = SomeDict;

/// RPython `class SomeIterator(SomeObject)` (model.py).
/// Wraps a container's element annotation; `variant` captures the
/// upstream `*variant` tuple (e.g. `"items"`, `"keys"`, `"values"`
/// for dict iterators).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeIterator {
    pub base: SomeObject,
    pub s_container: Box<SomeValue>,
    pub variant: Vec<String>,
    /// Upstream `SomeIterator(self, "enumerate", const)` carries the
    /// optional enumerate start constant as the second entry in the
    /// `variant` tuple. Rust keeps the marker list plus the explicit
    /// payload so callers can preserve the original constant without
    /// stringifying it away.
    pub enumerate_start: Option<crate::flowspace::model::ConstValue>,
}

impl SomeIterator {
    pub fn new(s_container: SomeValue, variant: Vec<String>) -> Self {
        Self::new_with_enumerate_start(s_container, variant, None)
    }

    pub fn new_with_enumerate_start(
        s_container: SomeValue,
        variant: Vec<String>,
        enumerate_start: Option<crate::flowspace::model::ConstValue>,
    ) -> Self {
        SomeIterator {
            base: SomeObject::new(KnownType::Iterator, false),
            s_container: Box::new(s_container),
            variant,
            enumerate_start,
        }
    }
}

impl SomeObjectTrait for SomeIterator {
    fn knowntype(&self) -> KnownType {
        // `SomeIterator.knowntype = type(iter([]))` (`model.py`).
        KnownType::Iterator
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// Instance / PBC / Builtin / Exception / None / Weakref / TypeOf variants (A4.5).

// `ClassDef` lives in [`super::classdesc`]; the Rust enum references
// classes through `Rc<RefCell<classdesc::ClassDef>>` with identity
// equality via `Rc::ptr_eq`, matching upstream Python class identity.

/// Identity equality for `Option<Rc<RefCell<ClassDef>>>` — two `Some`
/// values are equal only if they point at the same `Rc`; `None` is
/// equal to `None`.
fn classdef_opt_eq(a: &Option<Rc<RefCell<ClassDef>>>, b: &Option<Rc<RefCell<ClassDef>>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// Identity-equality membership test for a `Vec<Rc<RefCell<ClassDef>>>`.
fn classdef_vec_contains(v: &[Rc<RefCell<ClassDef>>], needle: &Rc<RefCell<ClassDef>>) -> bool {
    v.iter().any(|c| Rc::ptr_eq(c, needle))
}

/// `type(desc)` dispatched on in upstream's `SomePBC.getKind()`
/// (model.py:558-566). The Rust port closes the set explicitly.
///
/// The real [`super::description::DescEntry`] holds the `Rc<RefCell<…>>`
/// to each Desc subclass; `DescKind` is the enum-of-variants projection
/// used whenever the annotator needs to branch on `type(desc)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DescKind {
    /// RPython `description.FunctionDesc`.
    Function,
    /// RPython `description.MemoDesc` (subclass of `FunctionDesc`).
    /// Upstream `getKind` keys on `desc.__class__`, so a memo PBC is a
    /// distinct kind: mixing it with a plain `FunctionDesc` in one PBC
    /// raises "mixing several kinds of PBCs". Every kind-driven operation
    /// (`consider_call_site`, `simplify_desc_set`) is inherited from
    /// `FunctionDesc`, so a homogeneous memo PBC routes through the
    /// Function path.
    Memo,
    /// RPython `classdesc.ClassDesc`.
    Class,
    /// RPython `description.MethodDesc`.
    Method,
    /// RPython `description.FrozenDesc`.
    Frozen,
    /// RPython `description.MethodOfFrozenDesc`.
    MethodOfFrozen,
}

/// RPython `class SomeInstance(SomeObject)` (model.py).
///
/// Equality on `classdef` is Python-identity (`Rc::ptr_eq`) matching
/// upstream's `cls is other_cls` semantics — the manual `PartialEq`
/// impl below routes the field through [`classdef_opt_eq`].
#[derive(Clone, Debug)]
pub struct SomeInstance {
    pub base: SomeObject,
    /// RPython `self.classdef`. `None` denotes `object`-only instances
    /// (upstream: `SomeInstance(classdef=None)`).
    pub classdef: Option<Rc<RefCell<ClassDef>>>,
    pub can_be_none: bool,
    /// RPython `self.flags = flags` (model.py).
    ///
    /// Upstream is a Python `dict` whose values are the booleans /
    /// None sentinels produced by `binaryop.py:679`
    /// (`ins2.flags[key] != value` comparisons) and
    /// `description.py:492`. The Rust port stores a
    /// `BTreeMap<String, bool>` — `BTreeMap` so the field stays
    /// `Eq`-comparable through a stable iteration order. Upstream
    /// non-boolean flag payloads (None sentinels observed only at
    /// classdesc.py lookup_filter) will widen this value type when the bookkeeper
    /// lands.
    pub flags: std::collections::BTreeMap<String, bool>,
}

impl SomeInstance {
    /// RPython `SomeInstance.__init__(classdef, can_be_None=False, flags={})`.
    pub fn new(
        classdef: Option<Rc<RefCell<ClassDef>>>,
        can_be_none: bool,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Self {
        let knowntype = KnownType::Instance(
            classdef
                .as_ref()
                .map(|cd| cd.borrow().classdesc.borrow().identity.0),
        );
        SomeInstance {
            base: SomeObject::new(knowntype, false),
            classdef,
            can_be_none,
            flags,
        }
    }
}

impl PartialEq for SomeInstance {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && classdef_opt_eq(&self.classdef, &other.classdef)
            && self.can_be_none == other.can_be_none
            && self.flags == other.flags
    }
}

impl Eq for SomeInstance {}

impl SomeObjectTrait for SomeInstance {
    fn knowntype(&self) -> KnownType {
        // `SomeInstance.knowntype = classdef.classdesc if classdef else None`.
        KnownType::Instance(
            self.classdef
                .as_ref()
                .map(|cd| cd.borrow().classdesc.borrow().identity.0),
        )
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        self.can_be_none
    }
}

/// RPython `class SomeException(SomeObject)` (model.py). Set of
/// exception classdefs obeying `type(exc) in self.classes`.
///
/// Equality on `classdefs` is identity-based (`Rc::ptr_eq` per entry,
/// order-independent) — the manual `PartialEq` impl routes through
/// [`classdef_vec_contains`].
#[derive(Clone, Debug)]
pub struct SomeException {
    pub base: SomeObject,
    pub classdefs: Vec<Rc<RefCell<ClassDef>>>,
}

impl SomeException {
    pub fn new(classdefs: Vec<Rc<RefCell<ClassDef>>>) -> Self {
        let mut unique: Vec<Rc<RefCell<ClassDef>>> = Vec::new();
        for classdef in classdefs {
            if !classdef_vec_contains(&unique, &classdef) {
                unique.push(classdef);
            }
        }
        SomeException {
            base: SomeObject::new(KnownType::Other, false),
            classdefs: unique,
        }
    }

    /// RPython `SomeException.as_SomeInstance()` (model.py).
    pub fn as_some_instance(&self) -> SomeValue {
        let instances: Vec<SomeValue> = self
            .classdefs
            .iter()
            .cloned()
            .map(|classdef| {
                SomeValue::Instance(SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                ))
            })
            .collect();
        unionof(&instances).expect("SomeException.as_some_instance() must union its classdefs")
    }
}

impl PartialEq for SomeException {
    fn eq(&self, other: &Self) -> bool {
        if self.base != other.base || self.classdefs.len() != other.classdefs.len() {
            return false;
        }
        for c in &self.classdefs {
            if !classdef_vec_contains(&other.classdefs, c) {
                return false;
            }
        }
        true
    }
}

impl Eq for SomeException {}

impl SomeObjectTrait for SomeException {
    fn knowntype(&self) -> KnownType {
        // `SomeException` does not override `SomeObject.knowntype = object`.
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomePBC(SomeObject)` (model.py). "Prebuilt
/// constant" — a set of descriptions representing a closed family of
/// callables / classes / frozen instances.
///
/// `descriptions` carries real [`super::description::DescEntry`]
/// values (one per upstream Desc subclass, each wrapping an
/// `Rc<RefCell<…>>` of the concrete description instance). The
/// constructor runs [`simplify`] and the model.py
/// `knowntype` / `self.const = desc.pyobj` / multi-desc kind
/// enforcement branches (ClassDesc / MethodOfFrozenDesc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomePBC {
    pub base: SomeObject,
    /// RPython `self.descriptions` (model.py:522): a Python set of
    /// [`DescEntry`] objects. The Rust port uses
    /// `BTreeMap<DescKey, DescEntry>` — keyed by pointer identity to
    /// mirror `set(descriptions)` membership while keeping iteration
    /// deterministic.
    pub(crate) descriptions:
        std::collections::BTreeMap<super::description::DescKey, super::description::DescEntry>,
    pub can_be_none: bool,
    /// RPython `self.subset_of` — pointer to a wider PBC; `None` for
    /// the top PBC.
    pub subset_of: Option<Box<SomePBC>>,
}

impl SomePBC {
    /// RPython `SomePBC.__init__(descriptions, can_be_None=False,
    /// subset_of=None)` (model.py:519-553).
    pub(crate) fn new(
        descriptions: impl IntoIterator<Item = super::description::DescEntry>,
        can_be_none: bool,
    ) -> Self {
        Self::with_subset(descriptions, can_be_none, None)
    }

    pub(crate) fn with_subset(
        descriptions: impl IntoIterator<Item = super::description::DescEntry>,
        can_be_none: bool,
        subset_of: Option<Box<SomePBC>>,
    ) -> Self {
        let map: std::collections::BTreeMap<
            super::description::DescKey,
            super::description::DescEntry,
        > = descriptions
            .into_iter()
            .map(|e| (e.desc_key(), e))
            .collect();
        assert!(!map.is_empty(), "SomePBC must be non-empty");
        let mut pbc = SomePBC {
            base: SomeObject::new(KnownType::Other, true),
            descriptions: map,
            can_be_none,
            subset_of,
        };
        // upstream model.py — `self.simplify()` (dedup +
        // `getKind()` mixed-kind check). The constructor is infallible, so
        // a mixed-kind error here is not propagated at construction;
        // instead it surfaces at every use through the propagating
        // `get_kind()?` (consider_call_site, rtyper) and eagerly in
        // `union` (the realistic source of a heterogeneous set, which
        // rejects e.g. Function + Memo before this point). The outcome —
        // a mixed PBC is rejected — matches upstream; only the timing
        // differs. `simplify_desc_set` still runs on the homogeneous
        // (Ok) path.
        let _ = pbc.simplify();
        // upstream model.py — `knowntype = reduce(commonbase,
        // [x.knowntype for x in descriptions])`.
        pbc.base.knowntype = pbc
            .descriptions
            .values()
            .map(|desc| desc.knowntype())
            .reduce(commonbase)
            .unwrap_or(KnownType::Object);
        // upstream model.py — single-desc pyobj const hack:
        //     if len(descriptions) == 1 and not can_be_None:
        //         desc, = descriptions
        //         if desc.pyobj is not None:
        //             self.const = desc.pyobj
        if pbc.descriptions.len() == 1 && !pbc.can_be_none {
            let entry = pbc.descriptions.values().next().unwrap();
            if let Some(pyobj) = entry.pyobj_const() {
                pbc.base.const_box = Some(Constant::new(pyobj));
            }
        }
        // upstream model.py:538-553 — multi-desc kind enforcement.
        if pbc.descriptions.len() > 1
            && let Ok(kind) = pbc.get_kind()
        {
            match kind {
                DescKind::Class => {
                    // upstream: `for desc in descriptions: desc.getuniqueclassdef()`.
                    //   "a PBC of several classes: enforce them all
                    //   to be built, without support for specialization."
                    for entry in pbc.descriptions.values() {
                        if let Some(cd) = entry.as_class() {
                            let _ = super::classdesc::ClassDesc::getuniqueclassdef(&cd);
                        }
                    }
                }
                DescKind::MethodOfFrozen => {
                    // upstream: `funcdescs = set(desc.funcdesc for
                    //   desc in descriptions)`; mixing funcdescs
                    //   raises AnnotatorError.
                    let mut funcdesc_keys: std::collections::BTreeSet<super::description::DescKey> =
                        std::collections::BTreeSet::new();
                    for entry in pbc.descriptions.values() {
                        if let Some(mof) = entry.as_method_of_frozen() {
                            funcdesc_keys.insert(mof.borrow().funcdesc.desc_key());
                        }
                    }
                    if funcdesc_keys.len() > 1 {
                        panic!(
                            "AnnotatorError: You can't mix a set of methods on a frozen PBC in \
                                 RPython that are different underlying functions"
                        );
                    }
                }
                _ => {}
            }
        }
        pbc
    }

    /// RPython `SomePBC.any_description()` (model.py).
    pub(crate) fn any_description(&self) -> Option<&super::description::DescEntry> {
        self.descriptions.values().next()
    }

    /// RPython `SomePBC.getKind()` (model.py).
    ///
    /// Returns the common [`DescKind`] of every description in the
    /// PBC. Upstream raises `AnnotatorError("mixing several kinds of
    /// PBCs: %r")` when the set straddles two subclasses of `Desc`.
    pub fn get_kind(&self) -> Result<DescKind, AnnotatorError> {
        let mut kinds: std::collections::BTreeSet<DescKind> = std::collections::BTreeSet::new();
        for entry in self.descriptions.values() {
            kinds.insert(entry.kind());
        }
        if kinds.len() > 1 {
            return Err(AnnotatorError::new(format!(
                "mixing several kinds of PBCs: {kinds:?}"
            )));
        }
        kinds
            .into_iter()
            .next()
            .ok_or_else(|| AnnotatorError::new("empty SomePBC descriptions"))
    }

    /// RPython `SomePBC.simplify()` (model.py).
    ///
    /// ```python
    /// def simplify(self):
    ///     kind = self.getKind()
    ///     if len(self.descriptions) > 1:
    ///         kind.simplify_desc_set(self.descriptions)
    /// ```
    ///
    pub fn simplify(&mut self) -> Result<(), AnnotatorError> {
        // upstream: `kind.simplify_desc_set(self.descriptions)` —
        // polymorphic dispatch on `type(desc)`. Desc.simplify_desc_set
        // (description.py) is the staticmethod no-op base;
        // MethodDesc overrides it (description.py).
        let kind = self.get_kind()?;
        if self.descriptions.len() > 1 {
            match kind {
                DescKind::Method => {
                    super::description::MethodDesc::simplify_desc_set(&mut self.descriptions);
                }
                DescKind::Function
                | DescKind::Memo
                | DescKind::Class
                | DescKind::Frozen
                | DescKind::MethodOfFrozen => {
                    super::description::simplify_desc_set_default(&mut self.descriptions);
                }
            }
        }
        Ok(())
    }

    /// RPython `SomePBC.consider_call_site(self, args, s_result, call_op)`
    /// (model.py:576-578).
    ///
    /// ```python
    /// def consider_call_site(self, args, s_result, call_op):
    ///     descs = list(self.descriptions)
    ///     self.getKind().consider_call_site(descs, args, s_result, call_op)
    /// ```
    ///
    /// Kind dispatch routes through the staticmethod on each Desc
    /// subclass (description.py / 458-465 / 627-634, and
    /// classdesc.py's ClassDesc.consider_call_site). Rust port
    /// dispatches via variant match: FunctionDesc is fully ported;
    /// Method / Class / Frozen / MethodOfFrozen keep structural
    /// no-op bodies until their own `consider_call_site` methods
    /// land.
    pub(crate) fn consider_call_site(
        &self,
        args: &super::argument::ArgumentsForTranslation,
        s_result: &SomeValue,
        op_key: Option<super::bookkeeper::PositionKey>,
    ) -> Result<(), AnnotatorError> {
        let descs: Vec<super::description::DescEntry> =
            self.descriptions.values().cloned().collect();
        if descs.is_empty() {
            return Ok(());
        }
        match self.get_kind()? {
            // MemoDesc inherits FunctionDesc.consider_call_site; a
            // homogeneous memo PBC's `as_function()` yields each wrapped
            // base FunctionDesc.
            DescKind::Function | DescKind::Memo => {
                let fns: Vec<_> = descs.iter().filter_map(|d| d.as_function()).collect();
                super::description::FunctionDesc::consider_call_site(&fns, args, s_result, op_key)?;
            }
            DescKind::Method => {
                // description.py:458-465.
                let mds: Vec<_> = descs.iter().filter_map(|d| d.as_method()).collect();
                super::description::MethodDesc::consider_call_site(&mds, args, s_result, op_key)?;
            }
            DescKind::Class => {
                // classdesc.py (phase 1 only — __init__
                // recursion deferred, see ClassDesc::consider_call_site
                // for the full story).
                let cds: Vec<_> = descs.iter().filter_map(|d| d.as_class()).collect();
                super::classdesc::ClassDesc::consider_call_site(&cds, args, s_result, op_key)?;
            }
            DescKind::MethodOfFrozen => {
                // description.py:627-634.
                let mofds: Vec<_> = descs
                    .iter()
                    .filter_map(|d| d.as_method_of_frozen())
                    .collect();
                super::description::MethodOfFrozenDesc::consider_call_site(
                    &mofds, args, s_result, op_key,
                )?;
            }
            DescKind::Frozen => {
                // Upstream `Desc` base has no `consider_call_site`; in
                // Python `self.getKind().consider_call_site(...)` on a
                // FrozenDesc raises AttributeError. Surface that as an
                // AnnotatorError so callers (`Bookkeeper.consider_call_site`,
                // `compute_at_fixpoint`) observe the rejection via the
                // normal error channel rather than silently succeeding.
                return Err(AnnotatorError::new(
                    "FrozenDesc has no consider_call_site (upstream AttributeError)",
                ));
            }
        }
        Ok(())
    }

    /// RPython `SomePBC.nonnoneify()` (model.py).
    pub fn nonnoneify(&self) -> SomePBC {
        SomePBC::with_subset(
            self.descriptions.values().cloned(),
            false,
            self.subset_of.clone(),
        )
    }

    /// RPython `SomePBC.noneify()` (model.py).
    pub fn noneify(&self) -> SomePBC {
        SomePBC::with_subset(
            self.descriptions.values().cloned(),
            true,
            self.subset_of.clone(),
        )
    }
}

/// RPython `class SomeConstantType(SomePBC)` (model.py).
///
/// The Rust representation uses `SomePBC` plus `const_box` for this
/// subclass state; exposing the upstream name keeps the annotator and
/// binaryop surfaces aligned.
pub type SomeConstantType = SomePBC;

impl SomeObjectTrait for SomePBC {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        // upstream: `SomePBC.__init__` sets `self.const` when
        // `len(descriptions) == 1 and not can_be_None and desc.pyobj
        // is not None` (model.py). The constructor already
        // pins `const_box` via [`DescEntry::pyobj_const`], so this check
        // reduces to inspecting `const_box`.
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        self.can_be_none
    }
}

/// RPython `class SomeNone(SomeObject)` (model.py).
/// Zero-size marker type — every `SomeNone` value is the constant
/// `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeNone {
    pub base: SomeObject,
}

impl SomeNone {
    pub fn new() -> Self {
        let mut base = SomeObject::new(KnownType::NoneType, true);
        base.const_box = Some(Constant::new(
            super::super::flowspace::model::ConstValue::None,
        ));
        SomeNone { base }
    }
}

impl Default for SomeNone {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeNone {
    fn knowntype(&self) -> KnownType {
        KnownType::NoneType
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        true
    }
    fn can_be_none(&self) -> bool {
        true
    }
}

/// RPython `class SomeProperty(SomeObject)` (model.py).
///
/// Used only as the annotation carrier for immutable `property`
/// descriptors returned by `bookkeeper.immutablevalue`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeProperty {
    pub base: SomeObject,
    pub fget: Option<HostObject>,
    pub fset: Option<HostObject>,
}

impl SomeProperty {
    pub fn new(prop: &HostObject) -> Self {
        SomeProperty {
            base: SomeObject::new(KnownType::PropertyType, true),
            fget: prop.property_fget().cloned(),
            fset: prop.property_fset().cloned(),
        }
    }
}

impl SomeObjectTrait for SomeProperty {
    fn knowntype(&self) -> KnownType {
        KnownType::PropertyType
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// Re-export of [`SomePtr`], which RPython declares at
/// `rpython/rtyper/lltypesystem/lltype.py` (`class SomePtr(SomeObject)`).
/// The struct lives next to its upstream home in
/// [`crate::translator::rtyper::lltypesystem::lltype`]; this re-export
/// lets the `SomeValue::Ptr` variant and local call-sites keep
/// referring to it through the annotator-model module path.
pub use crate::translator::rtyper::lltypesystem::lltype::SomePtr;

/// Re-export of [`SomeAddress`], which RPython declares at
/// `rpython/rtyper/lltypesystem/llmemory.py` (`class SomeAddress(SomeObject)`).
pub use crate::translator::rtyper::lltypesystem::llmemory::SomeAddress;
/// `rpython/rtyper/lltypesystem/llmemory.py` (`class SomeTypedAddressAccess(SomeObject)`).
pub use crate::translator::rtyper::lltypesystem::llmemory::SomeTypedAddressAccess;

/// RPython `SomeObject.needs_sandboxing` side-attribute payload
/// (attached by `rtyper/extfunc.py:ExtFuncEntry.compute_annotation`
/// when `config.translation.sandbox` is on). Upstream uses ad-hoc
/// Python `setattr`; the Rust lattice carries it as an optional field
/// on [`SomeBuiltin`]. Populated only when the external-function
/// registration path flags the callable as sandboxed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxingPayload {
    /// Matches upstream `s_func.name` (extfunc.py).
    pub name: String,
    /// Matches upstream `s_func.args_s` — parameter annotations.
    pub args_s: Vec<SomeValue>,
    /// Matches upstream `s_func.s_result` — result annotation.
    pub s_result: Box<SomeValue>,
}

/// RPython `class SomeBuiltin(SomeObject)` (model.py).
///
/// Stands for a built-in function or method. `analyser` in upstream is
/// a callable hook; Rust port carries an opaque identifier because
/// first-class Rust closures don't round-trip through `==` / `Debug`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeBuiltin {
    pub base: SomeObject,
    /// Opaque name identifying the special-cased analyser
    /// (e.g. `"getattr"`, `"isinstance"`, `"len"`). Populated from
    /// specialcase.rs when the bookkeeper is wired in.
    pub analyser_name: String,
    /// RPython `self.s_self` — bound-method receiver annotation.
    pub s_self: Option<Box<SomeValue>>,
    /// RPython `self.methodname`.
    pub methodname: Option<String>,
    /// RPython `hasattr(s_func, 'needs_sandboxing')` payload
    /// (policy.py:85 gate). Set on external-function annotations when
    /// `config.translation.sandbox` is enabled (extfunc.py).
    /// `None` for all non-sandboxed callables.
    pub needs_sandboxing: Option<SandboxingPayload>,
}

impl SomeBuiltin {
    pub fn new(
        analyser_name: impl Into<String>,
        s_self: Option<SomeValue>,
        methodname: Option<String>,
    ) -> Self {
        SomeBuiltin {
            base: SomeObject::new(KnownType::BuiltinFunctionOrMethod, true),
            analyser_name: analyser_name.into(),
            s_self: s_self.map(Box::new),
            methodname,
            needs_sandboxing: None,
        }
    }
}

impl SomeObjectTrait for SomeBuiltin {
    fn knowntype(&self) -> KnownType {
        KnownType::BuiltinFunctionOrMethod
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeBuiltinMethod(SomeBuiltin)` (model.py).
///
/// Bound-method variant of [`SomeBuiltin`]. Upstream models this as a
/// distinct subclass whose constructor requires both the receiver and
/// method name; keep the same shape here instead of overloading
/// `SomeBuiltin` with additional ad-hoc flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeBuiltinMethod {
    pub base: SomeObject,
    /// Opaque identifier for the special-cased analyser hook.
    pub analyser_name: String,
    /// Bound-method receiver annotation (`self.s_self` upstream).
    ///
    /// Stored as an `Rc` (rather than a `Box`) so that `rtyper_makekey`
    /// can hash the receiver by its stable pointer identity — upstream's
    /// `id(self.s_self)` (rbuiltin.py) reuses the same Python
    /// object for every clone, and `Box` addresses would move on every
    /// `.clone()` of `SomeBuiltinMethod`.
    pub s_self: Rc<SomeValue>,
    /// Bound method name (`self.methodname` upstream).
    pub methodname: String,
}

impl SomeBuiltinMethod {
    pub fn new(
        analyser_name: impl Into<String>,
        s_self: SomeValue,
        methodname: impl Into<String>,
    ) -> Self {
        SomeBuiltinMethod {
            base: SomeObject::new(KnownType::BuiltinFunctionOrMethod, true),
            analyser_name: analyser_name.into(),
            s_self: Rc::new(s_self),
            methodname: methodname.into(),
        }
    }

    /// RPython `SomeBuiltinMethod.call(self, args, implicit_init=False)`
    /// (unaryop.py). Returns `None` for a void analyser (a method
    /// body that falls off the end), `Some(_)` otherwise.
    pub fn call(
        &self,
        args: &super::argument::ArgumentsForTranslation,
    ) -> Result<Option<SomeValue>, AnnotatorError> {
        let bk = super::bookkeeper::getbookkeeper().ok_or_else(|| {
            AnnotatorError::new("SomeBuiltinMethod.call() called without an active bookkeeper")
        })?;
        let ann = bk.annotator();
        super::unaryop::call_builtin_method(&ann, self, args)
    }
}

impl SomeObjectTrait for SomeBuiltinMethod {
    fn knowntype(&self) -> KnownType {
        KnownType::BuiltinFunctionOrMethod
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeImpossibleValue(SomeObject)` (model.py).
/// Stored here as an explicit struct variant so `is_immutable_constant`
/// / `annotationcolor` can be accessed consistently; the enum
/// `SomeValue::Impossible` remains as a zero-state convenience alias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeImpossibleValue {
    pub base: SomeObject,
}

impl SomeImpossibleValue {
    pub fn new() -> Self {
        SomeImpossibleValue {
            base: SomeObject::new(KnownType::Object, true),
        }
    }
}

impl Default for SomeImpossibleValue {
    fn default() -> Self {
        Self::new()
    }
}

impl SomeObjectTrait for SomeImpossibleValue {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        false
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class SomeWeakRef(SomeObject)` (model.py).
/// Stands for a `weakref.ref` whose target has a known classdef.
///
/// Equality on `classdef` is identity-based (`Rc::ptr_eq`).
#[derive(Clone, Debug)]
pub struct SomeWeakRef {
    pub base: SomeObject,
    /// RPython `self.classdef` — `None` for known-dead weakrefs.
    pub classdef: Option<Rc<RefCell<ClassDef>>>,
}

impl SomeWeakRef {
    pub fn new(classdef: Option<Rc<RefCell<ClassDef>>>) -> Self {
        SomeWeakRef {
            base: SomeObject::new(KnownType::WeakrefReference, true),
            classdef,
        }
    }
}

impl PartialEq for SomeWeakRef {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base && classdef_opt_eq(&self.classdef, &other.classdef)
    }
}

impl Eq for SomeWeakRef {}

impl SomeObjectTrait for SomeWeakRef {
    fn knowntype(&self) -> KnownType {
        KnownType::WeakrefReference
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        // upstream: default `SomeObject.can_be_none()` = True, no
        // override on SomeWeakRef.
        true
    }
}

/// RPython `class SomeTypeOf(SomeType)` (model.py). Used by
/// `typeof(args_v)` to track a type derived from a specific variable.
///
/// Upstream stores the actual `args_v` list — `Variable` objects —
/// so downstream refinement sites (`is__default`, `isinstance`, …) can
/// dispatch knowntypedata keyed on the underlying variables. The Rust
/// port matches: `is_type_of: Vec<Rc<Variable>>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeTypeOf {
    pub base: SomeObject,
    /// RPython `self.is_type_of` — the `args_v` list.
    pub is_type_of: Vec<Rc<Variable>>,
}

impl SomeTypeOf {
    pub fn new(is_type_of: Vec<Rc<Variable>>) -> Self {
        SomeTypeOf {
            base: SomeObject::new(KnownType::Type, true),
            is_type_of,
        }
    }
}

impl SomeObjectTrait for SomeTypeOf {
    fn knowntype(&self) -> KnownType {
        KnownType::Type
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// SomeValue closed enum.

/// Closed sum of every RPython `SomeXxx` subclass.
///
/// Variants land incrementally per the commit split documented at the
/// top of this file. A4.1 carries the primitive-family (Object /
/// Type / Float / Integer / Bool / String / Char / UnicodeString /
/// UnicodeCodePoint / ByteArray / SingleFloat / LongFloat) plus the
/// `Impossible` variant that anchors the lattice bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SomeValue {
    /// RPython `class SomeImpossibleValue(SomeObject)` (model.py —
    /// lands fully in A4.5). Placeholder until then; no state.
    Impossible,
    Object(SomeObject),
    Type(SomeType),
    Float(SomeFloat),
    SingleFloat(SomeSingleFloat),
    LongFloat(SomeLongFloat),
    Integer(SomeInteger),
    Bool(SomeBool),
    String(SomeString),
    UnicodeString(SomeUnicodeString),
    ByteArray(SomeByteArray),
    Char(SomeChar),
    UnicodeCodePoint(SomeUnicodeCodePoint),
    List(SomeList),
    Tuple(SomeTuple),
    Dict(SomeDict),
    Iterator(SomeIterator),
    Instance(SomeInstance),
    Exception(SomeException),
    PBC(SomePBC),
    None_(SomeNone),
    Property(SomeProperty),
    Ptr(SomePtr),
    InteriorPtr(SomeInteriorPtr),
    Address(SomeAddress),
    TypedAddressAccess(SomeTypedAddressAccess),
    LLADTMeth(SomeLLADTMeth),
    ExternalFunction(crate::translator::rtyper::extfunc::SomeExternalFunction),
    Builtin(SomeBuiltin),
    BuiltinMethod(SomeBuiltinMethod),
    WeakRef(SomeWeakRef),
    TypeOf(SomeTypeOf),
    StringBuilder(SomeStringBuilder),
    UnicodeBuilder(SomeUnicodeBuilder),
    RustVec(SomeRustVec),
}

/// Discriminant-only view of [`SomeValue`]. Parity mirror of RPython's
/// `type(s_x)` / `isinstance(s_x, SomeInteger)` pattern, which the
/// dispatch registries (operation.py:226-239, 268-281) use as the
/// lookup key.
///
/// Order matches the [`SomeValue`] variant declaration so the mapping
/// stays in sync with the enum itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SomeValueTag {
    Impossible,
    Object,
    Type,
    Float,
    SingleFloat,
    LongFloat,
    Integer,
    Bool,
    String,
    UnicodeString,
    ByteArray,
    Char,
    UnicodeCodePoint,
    List,
    Tuple,
    Dict,
    Iterator,
    Instance,
    Exception,
    PBC,
    None_,
    Property,
    Ptr,
    InteriorPtr,
    Address,
    TypedAddressAccess,
    LLADTMeth,
    ExternalFunction,
    Builtin,
    BuiltinMethod,
    WeakRef,
    TypeOf,
    StringBuilder,
    UnicodeBuilder,
    RustVec,
}

impl SomeValueTag {
    /// RPython `type(self_cls).__mro__` (operation.py:213-218). Returns
    /// the dispatch-lookup walk: the tag itself first, then its parent
    /// annotation tags up to [`SomeValueTag::Object`]. Registries use
    /// this to emulate RPython's `@register(Some_cls)` MRO fallback.
    pub fn mro(self) -> &'static [SomeValueTag] {
        use SomeValueTag as T;
        match self {
            // Primitive number chain: SomeBool < SomeInteger < SomeFloat < SomeObject
            // (model.py:164-242). Object is the root fallback.
            T::Bool => &[T::Bool, T::Integer, T::Float, T::Object],
            T::Integer => &[T::Integer, T::Float, T::Object],
            T::Float => &[T::Float, T::Object],
            T::SingleFloat => &[T::SingleFloat, T::Object],
            T::LongFloat => &[T::LongFloat, T::Object],
            // Type chain: SomeTypeOf < SomeType < SomeObject (model.py).
            T::TypeOf => &[T::TypeOf, T::Type, T::Object],
            T::Type => &[T::Type, T::Object],
            // String family: `SomeString`, `SomeUnicodeString` and
            // `SomeByteArray` share a `SomeStringOrUnicode` base
            // (model.py) that carries no pairtype
            // registration, so naming it here would resolve nothing — those
            // three resolve to themselves then Object. The two element tags
            // do inherit from a registered class: `SomeChar(SomeString)`
            // and `SomeUnicodeCodePoint(SomeUnicodeString)` (model.py). That
            // edge is load-bearing — `pairtype(SomeChar, SomeChar)`
            // (binaryop.py) defines only `union`, so `chr + chr` reaches
            // `pairtype(SomeString, SomeString).add` (binaryop.py)
            // through it, and a char operand serves every other `SomeString`
            // pair the same way.
            T::String => &[T::String, T::Object],
            T::UnicodeString => &[T::UnicodeString, T::Object],
            T::ByteArray => &[T::ByteArray, T::Object],
            T::Char => &[T::Char, T::String, T::Object],
            T::UnicodeCodePoint => &[T::UnicodeCodePoint, T::UnicodeString, T::Object],
            T::List => &[T::List, T::Object],
            T::Tuple => &[T::Tuple, T::Object],
            T::Dict => &[T::Dict, T::Object],
            T::Iterator => &[T::Iterator, T::Object],
            T::Instance => &[T::Instance, T::Object],
            T::Exception => &[T::Exception, T::Instance, T::Object],
            T::PBC => &[T::PBC, T::Object],
            T::None_ => &[T::None_, T::Object],
            T::Property => &[T::Property, T::Object],
            T::Ptr => &[T::Ptr, T::Object],
            T::InteriorPtr => &[T::InteriorPtr, T::Ptr, T::Object],
            T::Address => &[T::Address, T::Object],
            T::TypedAddressAccess => &[T::TypedAddressAccess, T::Object],
            T::LLADTMeth => &[T::LLADTMeth, T::Object],
            T::ExternalFunction => &[T::ExternalFunction, T::Object],
            T::Builtin => &[T::Builtin, T::Object],
            T::BuiltinMethod => &[T::BuiltinMethod, T::Object],
            T::WeakRef => &[T::WeakRef, T::Object],
            // rlib/rstring.py SomeStringBuilder / SomeUnicodeBuilder are
            // flat SomeObject subclasses; dispatch resolves to self then
            // Object.
            T::StringBuilder => &[T::StringBuilder, T::Object],
            T::UnicodeBuilder => &[T::UnicodeBuilder, T::Object],
            T::RustVec => &[T::RustVec, T::Object],
            // SomeImpossibleValue is the lattice bottom; it still
            // deserves an Object fallback so registries can bind
            // catch-all defaults keyed on Object.
            T::Impossible => &[T::Impossible, T::Object],
            T::Object => &[T::Object],
        }
    }
}

impl SomeValue {
    /// RPython `getattr(annotation, "knowntypedata", {})`
    /// (annrpython.py) — the exitswitch consumer reads branch
    /// constraints generically, without type-checking the annotation.
    /// `SomeBool` always carries it after a comparison; an integer
    /// discriminant `SomeInteger` carries per-case constraints once a
    /// `SwitchInt` exitswitch is annotated. Every other annotation
    /// returns the `{}` default (`None`).
    pub fn knowntypedata(&self) -> Option<&KnownTypeData> {
        match self {
            SomeValue::Bool(b) => b.knowntypedata.as_ref(),
            SomeValue::Integer(i) => i.knowntypedata.as_ref(),
            _ => None,
        }
    }

    /// RPython `type(self)` in dispatch keys — returns a [`SomeValueTag`]
    /// that identifies the variant without carrying any payload.
    pub fn tag(&self) -> SomeValueTag {
        use SomeValueTag as T;
        match self {
            SomeValue::Impossible => T::Impossible,
            SomeValue::Object(_) => T::Object,
            SomeValue::Type(_) => T::Type,
            SomeValue::Float(_) => T::Float,
            SomeValue::SingleFloat(_) => T::SingleFloat,
            SomeValue::LongFloat(_) => T::LongFloat,
            SomeValue::Integer(_) => T::Integer,
            SomeValue::Bool(_) => T::Bool,
            SomeValue::String(_) => T::String,
            SomeValue::UnicodeString(_) => T::UnicodeString,
            SomeValue::ByteArray(_) => T::ByteArray,
            SomeValue::Char(_) => T::Char,
            SomeValue::UnicodeCodePoint(_) => T::UnicodeCodePoint,
            SomeValue::List(_) => T::List,
            SomeValue::Tuple(_) => T::Tuple,
            SomeValue::Dict(_) => T::Dict,
            SomeValue::Iterator(_) => T::Iterator,
            SomeValue::Instance(_) => T::Instance,
            SomeValue::Exception(_) => T::Exception,
            SomeValue::PBC(_) => T::PBC,
            SomeValue::None_(_) => T::None_,
            SomeValue::Property(_) => T::Property,
            SomeValue::Ptr(_) => T::Ptr,
            SomeValue::InteriorPtr(_) => T::InteriorPtr,
            SomeValue::Address(_) => T::Address,
            SomeValue::TypedAddressAccess(_) => T::TypedAddressAccess,
            SomeValue::LLADTMeth(_) => T::LLADTMeth,
            SomeValue::ExternalFunction(_) => T::ExternalFunction,
            SomeValue::Builtin(_) => T::Builtin,
            SomeValue::BuiltinMethod(_) => T::BuiltinMethod,
            SomeValue::WeakRef(_) => T::WeakRef,
            SomeValue::TypeOf(_) => T::TypeOf,
            SomeValue::StringBuilder(_) => T::StringBuilder,
            SomeValue::UnicodeBuilder(_) => T::UnicodeBuilder,
            SomeValue::RustVec(_) => T::RustVec,
        }
    }

    /// Shorthand for [`SomeValue::Object`] with `SomeObject::default()`
    /// — the upstream `SomeObject()` constructor.
    pub fn object() -> Self {
        SomeValue::Object(SomeObject::default())
    }

    /// RPython `s.contains(other)` (model.py). Delegates to the
    /// module-level [`contains`] helper once [`union`] is available
    /// (A4.6).
    pub fn contains(&self, other: &SomeValue) -> bool {
        contains(self, other)
    }

    /// RPython `s.noneify()` (model.py plus per-subclass
    /// overrides).
    pub fn noneify(&self) -> Result<SomeValue, UnionError> {
        match self {
            SomeValue::String(s) => Ok(SomeValue::String(SomeString::new(true, s.inner.no_nul))),
            SomeValue::UnicodeString(s) => Ok(SomeValue::UnicodeString(SomeUnicodeString::new(
                true,
                s.inner.no_nul,
            ))),
            SomeValue::List(s) => Ok(SomeValue::List(SomeList::new(s.listdef.clone()))),
            SomeValue::Dict(s) => Ok(SomeValue::Dict(SomeDict::new(s.dictdef.clone()))),
            // `SomeInstance.noneify` reconstructs as
            // `SomeInstance(self.classdef, can_be_None=True)`, taking the
            // `flags={}` default: the flags do NOT survive. Carrying them
            // over would leave `access_directly` set on an annotation that
            // reached `None`, and `policy.py:71-83` turns an over-set flag
            // into an aborted translation, not a missed optimisation.
            SomeValue::Instance(s) => Ok(SomeValue::Instance(SomeInstance::new(
                s.classdef.clone(),
                true,
                std::collections::BTreeMap::new(),
            ))),
            SomeValue::PBC(s) => Ok(SomeValue::PBC(SomePBC::with_subset(
                s.descriptions.values().cloned(),
                true,
                s.subset_of.clone(),
            ))),
            SomeValue::WeakRef(s) => Ok(SomeValue::WeakRef(SomeWeakRef::new(s.classdef.clone()))),
            // rstring.py/1271 — SomeStringBuilder / SomeUnicodeBuilder
            // `noneify(self)` returns `self`.
            SomeValue::StringBuilder(_) | SomeValue::UnicodeBuilder(_) => Ok(self.clone()),
            _ => Err(UnionError {
                lhs: Box::new(self.clone()),
                rhs: Box::new(s_none()),
                msg: "RPython noneify() not supported for this annotation".into(),
            }),
        }
    }

    /// RPython `s.nonnoneify()` (model.py plus per-subclass
    /// overrides).
    pub fn nonnoneify(&self) -> SomeValue {
        match self {
            SomeValue::String(s) => SomeValue::String(SomeString::new(false, s.inner.no_nul)),
            SomeValue::UnicodeString(s) => {
                SomeValue::UnicodeString(SomeUnicodeString::new(false, s.inner.no_nul))
            }
            SomeValue::ByteArray(_) => SomeValue::ByteArray(SomeByteArray::new(false)),
            // `SomeInstance.nonnoneify` drops the flags for the same
            // reason `noneify` does — it reconstructs with the `flags={}`
            // default.
            SomeValue::Instance(s) => SomeValue::Instance(SomeInstance::new(
                s.classdef.clone(),
                false,
                std::collections::BTreeMap::new(),
            )),
            SomeValue::PBC(s) => SomeValue::PBC(SomePBC::with_subset(
                s.descriptions.values().cloned(),
                false,
                s.subset_of.clone(),
            )),
            SomeValue::None_(_) => SomeValue::Impossible,
            _ => self.clone(),
        }
    }

    /// RPython `s.nonnulify()` for `SomeStringOrUnicode`
    /// (model.py:281-285).
    pub fn nonnulify(&self) -> Result<SomeValue, UnionError> {
        match self {
            SomeValue::String(s) => Ok(SomeValue::String(SomeString::new(
                s.inner.can_be_none,
                true,
            ))),
            SomeValue::UnicodeString(s) => Ok(SomeValue::UnicodeString(SomeUnicodeString::new(
                s.inner.can_be_none,
                true,
            ))),
            _ => Err(UnionError {
                lhs: Box::new(self.clone()),
                rhs: Box::new(self.clone()),
                msg: "RPython nonnulify() not supported for this annotation".into(),
            }),
        }
    }

    /// RPython `s.const` property (ConstAccessDelegator at model.py
    /// top-level; resolves to `s.const_box.value`).
    ///
    /// Upstream raises `AttributeError` when `const_box` is absent;
    /// the Rust port returns `None` instead. Callers guard with
    /// [`SomeObjectTrait::is_constant`] to avoid the miss case —
    /// mirroring `if s.is_constant(): r = s.const` in the Python
    /// sources. The trailing underscore in the method name works
    /// around Rust's reserved `const` keyword.
    pub fn const_(&self) -> Option<&crate::flowspace::model::ConstValue> {
        let cb: Option<&Constant> = match self {
            SomeValue::Impossible => None,
            SomeValue::Object(s) => s.const_box.as_ref(),
            SomeValue::Type(s) => s.base.const_box.as_ref(),
            SomeValue::Float(s) => s.base.const_box.as_ref(),
            SomeValue::SingleFloat(s) => s.base.const_box.as_ref(),
            SomeValue::LongFloat(s) => s.base.const_box.as_ref(),
            SomeValue::Integer(s) => s.base.const_box.as_ref(),
            SomeValue::Bool(s) => s.base.const_box.as_ref(),
            SomeValue::String(s) => s.inner.base.const_box.as_ref(),
            SomeValue::UnicodeString(s) => s.inner.base.const_box.as_ref(),
            SomeValue::ByteArray(s) => s.inner.base.const_box.as_ref(),
            SomeValue::Char(s) => s.inner.base.const_box.as_ref(),
            SomeValue::UnicodeCodePoint(s) => s.inner.base.const_box.as_ref(),
            SomeValue::List(s) => s.base.const_box.as_ref(),
            SomeValue::Tuple(s) => s.base.const_box.as_ref(),
            SomeValue::Dict(s) => s.base.const_box.as_ref(),
            SomeValue::Iterator(s) => s.base.const_box.as_ref(),
            SomeValue::Instance(s) => s.base.const_box.as_ref(),
            SomeValue::Exception(s) => s.base.const_box.as_ref(),
            SomeValue::PBC(s) => s.base.const_box.as_ref(),
            SomeValue::None_(s) => s.base.const_box.as_ref(),
            SomeValue::Property(s) => s.base.const_box.as_ref(),
            SomeValue::Ptr(s) => s.base.const_box.as_ref(),
            SomeValue::InteriorPtr(s) => s.base.const_box.as_ref(),
            SomeValue::Address(s) => s.base.const_box.as_ref(),
            SomeValue::TypedAddressAccess(s) => s.base.const_box.as_ref(),
            SomeValue::LLADTMeth(s) => s.base.const_box.as_ref(),
            SomeValue::ExternalFunction(s) => s.base.const_box.as_ref(),
            SomeValue::Builtin(s) => s.base.const_box.as_ref(),
            SomeValue::BuiltinMethod(s) => s.base.const_box.as_ref(),
            SomeValue::WeakRef(s) => s.base.const_box.as_ref(),
            SomeValue::TypeOf(s) => s.base.const_box.as_ref(),
            SomeValue::StringBuilder(s) => s.base.const_box.as_ref(),
            SomeValue::UnicodeBuilder(s) => s.base.const_box.as_ref(),
            SomeValue::RustVec(s) => s.base.const_box.as_ref(),
        };
        cb.map(|c| &c.value)
    }

    /// Polymorphic `result.const = x` setter — mirrors upstream
    /// `bookkeeper.py:347` (`try: result.const = x; except AttributeError: pass`).
    ///
    /// Upstream wraps the assignment in `try/except` because some
    /// `SomeXXX` subclasses define `const` as a property without a
    /// setter (raising `AttributeError`). The Rust port routes every
    /// variant to its `SomeObject.const_box`; the few variants
    /// upstream rejects (`SomeImpossibleValue`) match `Impossible` here
    /// and silently no-op, matching the AttributeError swallow.
    pub fn set_const_box(&mut self, c: Constant) {
        match self {
            SomeValue::Impossible => {} // upstream AttributeError swallow.
            SomeValue::Object(s) => s.const_box = Some(c),
            SomeValue::Type(s) => s.base.const_box = Some(c),
            SomeValue::Float(s) => s.base.const_box = Some(c),
            SomeValue::SingleFloat(s) => s.base.const_box = Some(c),
            SomeValue::LongFloat(s) => s.base.const_box = Some(c),
            SomeValue::Integer(s) => s.base.const_box = Some(c),
            SomeValue::Bool(s) => s.base.const_box = Some(c),
            SomeValue::String(s) => s.inner.base.const_box = Some(c),
            SomeValue::UnicodeString(s) => s.inner.base.const_box = Some(c),
            SomeValue::ByteArray(s) => s.inner.base.const_box = Some(c),
            SomeValue::Char(s) => s.inner.base.const_box = Some(c),
            SomeValue::UnicodeCodePoint(s) => s.inner.base.const_box = Some(c),
            SomeValue::List(s) => s.base.const_box = Some(c),
            SomeValue::Tuple(s) => s.base.const_box = Some(c),
            SomeValue::Dict(s) => s.base.const_box = Some(c),
            SomeValue::Iterator(s) => s.base.const_box = Some(c),
            SomeValue::Instance(s) => s.base.const_box = Some(c),
            SomeValue::Exception(s) => s.base.const_box = Some(c),
            SomeValue::PBC(s) => s.base.const_box = Some(c),
            SomeValue::None_(s) => s.base.const_box = Some(c),
            SomeValue::Property(s) => s.base.const_box = Some(c),
            SomeValue::Ptr(s) => s.base.const_box = Some(c),
            SomeValue::InteriorPtr(s) => s.base.const_box = Some(c),
            SomeValue::Address(s) => s.base.const_box = Some(c),
            SomeValue::TypedAddressAccess(s) => s.base.const_box = Some(c),
            SomeValue::LLADTMeth(s) => s.base.const_box = Some(c),
            SomeValue::ExternalFunction(s) => s.base.const_box = Some(c),
            SomeValue::Builtin(s) => s.base.const_box = Some(c),
            SomeValue::BuiltinMethod(s) => s.base.const_box = Some(c),
            SomeValue::WeakRef(s) => s.base.const_box = Some(c),
            SomeValue::TypeOf(s) => s.base.const_box = Some(c),
            SomeValue::StringBuilder(s) => s.base.const_box = Some(c),
            SomeValue::UnicodeBuilder(s) => s.base.const_box = Some(c),
            SomeValue::RustVec(s) => s.base.const_box = Some(c),
        }
    }

    /// RPython `SomeObject.find_method(self, name)` (unaryop.py).
    pub fn find_method(&self, name: &str) -> Option<SomeValue> {
        super::unaryop::find_method(self, name).map(SomeValue::BuiltinMethod)
    }

    /// RPython `SomeObject.call(self, args, implicit_init=False)` and the
    /// `SomeBuiltinMethod` / `SomePBC` / `SomeNone` overrides
    /// (unaryop.py, 961-967, 985-987, 1011-1012).
    ///
    /// Returns `Ok(None)` for a void result — a builtin method analyser
    /// whose body falls off the end (`unaryop.py simple_call`
    /// returns `s_func.call(argspec)`, which may be Python `None`). All
    /// other callable kinds yield a concrete annotation wrapped in
    /// `Some(_)`; an unrefined/None receiver still blocks via
    /// `Some(SomeImpossibleValue())`.
    pub fn call(
        &self,
        args: &super::argument::ArgumentsForTranslation,
    ) -> Result<Option<SomeValue>, AnnotatorError> {
        match self {
            SomeValue::PBC(pbc) => {
                let bk = super::bookkeeper::getbookkeeper().ok_or_else(|| {
                    AnnotatorError::new("SomePBC.call() called without an active bookkeeper")
                })?;
                bk.pbc_call(pbc, args, super::bookkeeper::PbcCallEmulated::None)
                    .map(Some)
            }
            SomeValue::Ptr(ptr) => ptr.call(args).map(Some),
            SomeValue::InteriorPtr(ptr) => ptr.call(args).map(Some),
            SomeValue::LLADTMeth(adtmeth) => adtmeth.call(args).map(Some),
            SomeValue::ExternalFunction(ext) => ext.call(args),
            SomeValue::None_(_) => Ok(Some(s_impossible_value())),
            SomeValue::BuiltinMethod(method) => method.call(args),
            SomeValue::Builtin(sb) => {
                // upstream `SomeBuiltin.call(args)` (unaryop.py):
                //
                //     args_s, kwds = args.unpack()
                //     kwds_s = {}
                //     for key, s_value in kwds.items():
                //         kwds_s['s_'+key] = s_value
                //     return self.analyser(*args_s, **kwds_s)
                let bk = super::bookkeeper::getbookkeeper().ok_or_else(|| {
                    AnnotatorError::new("SomeBuiltin.call() called without an active bookkeeper")
                })?;
                let (args_s_opt, kwds) = args
                    .unpack()
                    .map_err(|err| AnnotatorError::new(err.getmsg()))?;
                // `model.py SomeBuiltin.call` passes args_s
                // (which may carry None for unbound caller args) directly
                // to `analyser(*args_s, **kwds_s)`; the analyser body
                // raises AttributeError on the first attribute access of
                // an unbound slot.  Pyre threads `&[Option<SomeValue>]`
                // through `call_builtin`; each analyser body unwraps via
                // [`builtin::arg_at`] at the touch site, panicking with
                // the analyser name + slot index at the same iteration
                // upstream would raise (`unaryop.py` bind-then-body
                // sequence).
                // upstream `model.py:1184 kwds_s['s_'+key] = s_value`
                // — passes the value through unchanged (None or
                // SomeValue).  Pyre's `BuiltinAnalyzer` ABI now matches
                // `HashMap<String, Option<SomeValue>>` so the `None`
                // sentinel propagates all the way to the per-touch
                // unwrap inside each analyser body.
                let kwds_s: std::collections::HashMap<String, Option<SomeValue>> = kwds
                    .into_iter()
                    .map(|(k, v)| (format!("s_{k}"), v))
                    .collect();
                // upstream `SomeBuiltin.analyser` is the bound method
                // `entry.compute_result_annotation` for builtins
                // produced via `extregistry.py` —
                // `SomeBuiltin(self.compute_result_annotation, ...)`.
                // Rust splits dispatch in two: the registered
                // `BUILTIN_ANALYZERS` table covers the `bookkeeper.py`
                // builtin path, and entries from the value-level
                // extregistry (rlib/jit.py markers, etc.) carry their
                // host identity in `const_box`. Re-look-up the
                // ExtRegistryEntry via that identity and dispatch
                // through `compute_annotation_with_kwds` so the
                // marker's `compute_result_annotation` body fires
                // exactly as `analyser(*args_s, **kwds_s)` does
                // upstream.
                if let Some(c) = sb.base.const_box.as_ref()
                    && let crate::flowspace::model::ConstValue::HostObject(_) = &c.value
                    && let Some(entry) = crate::translator::rtyper::extregistry::lookup(&c.value)
                    && matches!(
                        entry,
                        crate::translator::rtyper::extregistry::ExtRegistryEntry::EnterLeaveMarker { .. }
                            | crate::translator::rtyper::extregistry::ExtRegistryEntry::LoopHeader { .. }
                            | crate::translator::rtyper::extregistry::ExtRegistryEntry::ForType { .. }
                            | crate::translator::rtyper::extregistry::ExtRegistryEntry::JitForceVirtualizable
                            | crate::translator::rtyper::extregistry::ExtRegistryEntry::JitForceQuasiImmutable
                    )
                {
                    return entry.compute_annotation_with_kwds(&bk, &kwds_s).map(Some);
                }
                super::builtin::call_builtin(&bk, &sb.analyser_name, &args_s_opt, &kwds_s).map(Some)
            }
            // Upstream raises the bare message; the callee's lattice tag is
            // added because pyre's panic path drops the graph/block context
            // upstream's annotator attaches, and the tag is what localizes
            // the producer (an `Integer`/`Bool` here means the callee
            // position holds a *field* value, not a callable).
            other => Err(AnnotatorError::new(format!(
                "Cannot prove that the object is callable: {:?}",
                other.tag()
            ))),
        }
    }
}

impl SomeObjectTrait for SomeValue {
    fn knowntype(&self) -> KnownType {
        match self {
            SomeValue::Impossible => KnownType::Object,
            SomeValue::Object(s) => s.knowntype,
            SomeValue::Type(s) => s.knowntype(),
            SomeValue::Float(s) => s.knowntype(),
            SomeValue::SingleFloat(s) => s.knowntype(),
            SomeValue::LongFloat(s) => s.knowntype(),
            SomeValue::Integer(s) => s.knowntype(),
            SomeValue::Bool(s) => s.knowntype(),
            SomeValue::String(s) => s.knowntype(),
            SomeValue::UnicodeString(s) => s.knowntype(),
            SomeValue::ByteArray(s) => s.knowntype(),
            SomeValue::Char(s) => s.knowntype(),
            SomeValue::UnicodeCodePoint(s) => s.knowntype(),
            SomeValue::List(s) => s.knowntype(),
            SomeValue::Tuple(s) => s.knowntype(),
            SomeValue::Dict(s) => s.knowntype(),
            SomeValue::Iterator(s) => s.knowntype(),
            SomeValue::Instance(s) => s.knowntype(),
            SomeValue::Exception(s) => s.knowntype(),
            SomeValue::PBC(s) => s.knowntype(),
            SomeValue::None_(s) => s.knowntype(),
            SomeValue::Property(s) => s.knowntype(),
            SomeValue::Ptr(s) => s.knowntype(),
            SomeValue::InteriorPtr(s) => s.knowntype(),
            SomeValue::Address(s) => s.knowntype(),
            SomeValue::TypedAddressAccess(s) => s.knowntype(),
            SomeValue::LLADTMeth(s) => s.knowntype(),
            SomeValue::ExternalFunction(s) => s.knowntype(),
            SomeValue::Builtin(s) => s.knowntype(),
            SomeValue::BuiltinMethod(s) => s.knowntype(),
            SomeValue::WeakRef(s) => s.knowntype(),
            SomeValue::TypeOf(s) => s.knowntype(),
            SomeValue::StringBuilder(s) => s.knowntype(),
            SomeValue::UnicodeBuilder(s) => s.knowntype(),
            SomeValue::RustVec(s) => s.knowntype(),
        }
    }

    fn immutable(&self) -> bool {
        match self {
            SomeValue::Impossible => true,
            SomeValue::Object(s) => s.immutable,
            SomeValue::Type(s) => s.immutable(),
            SomeValue::Float(s) => s.immutable(),
            SomeValue::SingleFloat(s) => s.immutable(),
            SomeValue::LongFloat(s) => s.immutable(),
            SomeValue::Integer(s) => s.immutable(),
            SomeValue::Bool(s) => s.immutable(),
            SomeValue::String(s) => s.immutable(),
            SomeValue::UnicodeString(s) => s.immutable(),
            SomeValue::ByteArray(s) => s.immutable(),
            SomeValue::Char(s) => s.immutable(),
            SomeValue::UnicodeCodePoint(s) => s.immutable(),
            SomeValue::List(s) => s.immutable(),
            SomeValue::Tuple(s) => s.immutable(),
            SomeValue::Dict(s) => s.immutable(),
            SomeValue::Iterator(s) => s.immutable(),
            SomeValue::Instance(s) => s.immutable(),
            SomeValue::Exception(s) => s.immutable(),
            SomeValue::PBC(s) => s.immutable(),
            SomeValue::None_(s) => s.immutable(),
            SomeValue::Property(s) => s.immutable(),
            SomeValue::Ptr(s) => s.immutable(),
            SomeValue::InteriorPtr(s) => s.immutable(),
            SomeValue::Address(s) => s.immutable(),
            SomeValue::TypedAddressAccess(s) => s.immutable(),
            SomeValue::LLADTMeth(s) => s.immutable(),
            SomeValue::ExternalFunction(s) => s.immutable(),
            SomeValue::Builtin(s) => s.immutable(),
            SomeValue::BuiltinMethod(s) => s.immutable(),
            SomeValue::WeakRef(s) => s.immutable(),
            SomeValue::TypeOf(s) => s.immutable(),
            SomeValue::StringBuilder(s) => s.immutable(),
            SomeValue::UnicodeBuilder(s) => s.immutable(),
            SomeValue::RustVec(s) => s.immutable(),
        }
    }

    fn is_constant(&self) -> bool {
        match self {
            SomeValue::Impossible => false,
            SomeValue::Object(s) => s.const_box.is_some(),
            SomeValue::Type(s) => s.is_constant(),
            SomeValue::Float(s) => s.is_constant(),
            SomeValue::SingleFloat(s) => s.is_constant(),
            SomeValue::LongFloat(s) => s.is_constant(),
            SomeValue::Integer(s) => s.is_constant(),
            SomeValue::Bool(s) => s.is_constant(),
            SomeValue::String(s) => s.is_constant(),
            SomeValue::UnicodeString(s) => s.is_constant(),
            SomeValue::ByteArray(s) => s.is_constant(),
            SomeValue::Char(s) => s.is_constant(),
            SomeValue::UnicodeCodePoint(s) => s.is_constant(),
            SomeValue::List(s) => s.is_constant(),
            SomeValue::Tuple(s) => s.is_constant(),
            SomeValue::Dict(s) => s.is_constant(),
            SomeValue::Iterator(s) => s.is_constant(),
            SomeValue::Instance(s) => s.is_constant(),
            SomeValue::Exception(s) => s.is_constant(),
            SomeValue::PBC(s) => s.is_constant(),
            SomeValue::None_(s) => s.is_constant(),
            SomeValue::Property(s) => s.is_constant(),
            SomeValue::Ptr(s) => s.is_constant(),
            SomeValue::InteriorPtr(s) => s.is_constant(),
            SomeValue::Address(s) => s.is_constant(),
            SomeValue::TypedAddressAccess(s) => s.is_constant(),
            SomeValue::LLADTMeth(s) => s.is_constant(),
            SomeValue::ExternalFunction(s) => s.is_constant(),
            SomeValue::Builtin(s) => s.is_constant(),
            SomeValue::BuiltinMethod(s) => s.is_constant(),
            SomeValue::WeakRef(s) => s.is_constant(),
            SomeValue::TypeOf(s) => s.is_constant(),
            SomeValue::StringBuilder(s) => s.is_constant(),
            SomeValue::UnicodeBuilder(s) => s.is_constant(),
            SomeValue::RustVec(s) => s.is_constant(),
        }
    }

    fn can_be_none(&self) -> bool {
        match self {
            // SomeImpossibleValue carries no value, so "can be None"
            // collapses to False (upstream: `SomeImpossibleValue.can_be_none`
            // is never true). Documented early so A4.5 does not have to
            // change the arm.
            SomeValue::Impossible => false,
            // The bare SomeObject answers True (model.py default).
            SomeValue::Object(_) => true,
            SomeValue::Type(s) => s.can_be_none(),
            SomeValue::Float(s) => s.can_be_none(),
            SomeValue::SingleFloat(s) => s.can_be_none(),
            SomeValue::LongFloat(s) => s.can_be_none(),
            SomeValue::Integer(s) => s.can_be_none(),
            SomeValue::Bool(s) => s.can_be_none(),
            SomeValue::String(s) => s.can_be_none(),
            SomeValue::UnicodeString(s) => s.can_be_none(),
            SomeValue::ByteArray(s) => s.can_be_none(),
            SomeValue::Char(s) => s.can_be_none(),
            SomeValue::UnicodeCodePoint(s) => s.can_be_none(),
            SomeValue::List(s) => s.can_be_none(),
            SomeValue::Tuple(s) => s.can_be_none(),
            SomeValue::Dict(s) => s.can_be_none(),
            SomeValue::Iterator(s) => s.can_be_none(),
            SomeValue::Instance(s) => s.can_be_none(),
            SomeValue::Exception(s) => s.can_be_none(),
            SomeValue::PBC(s) => s.can_be_none(),
            SomeValue::None_(s) => s.can_be_none(),
            SomeValue::Property(s) => s.can_be_none(),
            SomeValue::Ptr(s) => s.can_be_none(),
            SomeValue::InteriorPtr(s) => s.can_be_none(),
            SomeValue::Address(s) => s.can_be_none(),
            SomeValue::TypedAddressAccess(s) => s.can_be_none(),
            SomeValue::LLADTMeth(s) => s.can_be_none(),
            SomeValue::ExternalFunction(s) => s.can_be_none(),
            SomeValue::Builtin(s) => s.can_be_none(),
            SomeValue::BuiltinMethod(s) => s.can_be_none(),
            SomeValue::WeakRef(s) => s.can_be_none(),
            SomeValue::TypeOf(s) => s.can_be_none(),
            SomeValue::StringBuilder(s) => s.can_be_none(),
            SomeValue::UnicodeBuilder(s) => s.can_be_none(),
            SomeValue::RustVec(s) => s.can_be_none(),
        }
    }
}

/// RPython `bind_callables_under(s_value, classdef, name)` — the
/// method defined on `SomeObject` / `SomePBC` / `SomeNone` via the
/// `__extend__(...)` decorator in `rpython/annotator/unaryop.py`,
/// 989-991, 1001-1002.
///
/// ```python
/// # SomeObject default:
/// def bind_callables_under(self, classdef, name):
///     return self
/// # SomePBC:
/// def bind_callables_under(self, classdef, name):
///     d = [desc.bind_under(classdef, name) for desc in self.descriptions]
///     return SomePBC(d, can_be_None=self.can_be_None)
/// # SomeNone:
/// def bind_callables_under(self, classdef, name):
///     return self
/// ```
pub(crate) fn bind_callables_under(
    s_value: &SomeValue,
    classdef: &Rc<std::cell::RefCell<super::classdesc::ClassDef>>,
    name: &str,
) -> SomeValue {
    match s_value {
        SomeValue::PBC(pbc) => {
            let bound: Vec<_> = pbc
                .descriptions
                .values()
                .map(|desc| desc.bind_under(classdef, name))
                .collect();
            SomeValue::PBC(SomePBC::new(bound, pbc.can_be_none))
        }
        // Default SomeObject and SomeNone — both return self.
        _ => s_value.clone(),
    }
}

// UnionError — raised by A4.6's union dispatch.

/// RPython `class UnionError(Exception)` (model.py — exact line
/// varies by upstream revision). Produced by the union dispatch when
/// two annotations cannot be merged. Carried here for type completeness;
/// A4.6 populates the payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionError {
    pub lhs: Box<SomeValue>,
    pub rhs: Box<SomeValue>,
    pub msg: String,
}

impl fmt::Display for UnionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnionError: {} ∪ {} — {}",
            self.lhs.knowntype(),
            self.rhs.knowntype(),
            self.msg
        )
    }
}

impl std::error::Error for UnionError {}

// Module-level singletons (model.py:685-694).

/// RPython `s_None = SomeNone()` (model.py).
pub(crate) fn s_none() -> SomeValue {
    SomeValue::None_(SomeNone::new())
}

/// RPython `s_ImpossibleValue = SomeImpossibleValue()` (model.py).
pub(crate) fn s_impossible_value() -> SomeValue {
    SomeValue::Impossible
}

/// RPython `s_Bool = SomeBool()` (model.py).
pub(crate) fn s_bool() -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

/// RPython `s_Int = SomeInteger()` (model.py).
pub(crate) fn s_int() -> SomeValue {
    SomeValue::Integer(SomeInteger::default())
}

/// `SomeInteger(unsigned=True)` — an r_uint value, the shell an unsigned
/// scalar (`u8`..`u64`, `usize`) carries.  Matches `valuetype_to_someshell`'s
/// `ValueType::Unsigned` projection so the string- and TyRef-derived field
/// seeds agree; a signed shell here walls `r_uint ∪ int` when the write side
/// keeps the unsigned knowntype.
pub(crate) fn s_uint() -> SomeValue {
    SomeValue::Integer(SomeInteger::new(false, true))
}

/// RPython `s_Str0 = SomeString(no_nul=True)` (model.py).
pub(crate) fn s_str0() -> SomeValue {
    SomeValue::String(SomeString::new(false, true))
}

// AnnotatorError + helpers (model.py + 787-795).

/// RPython `class AnnotatorError(Exception)` (model.py). Base
/// error raised by the annotator outside of the structural `UnionError`
/// path.
///
/// `block` carries the `__annotator_block` debug attribute that upstream
/// attaches at `rpython/annotator/annrpython.py:402` so
/// `rpython/tool/error.py` can render a "Processing block:" section
/// when the annotator aborts. Only the pointer identity is kept (as a
/// `usize`) so the error stays `Send` for `panic_any`; a full block
/// lookup can go through the translator's block table when
/// `rpython/tool/error.py` is ported. The field is populated by
/// [`set_annotator_block`] from `processblock` and stays `None`
/// otherwise. Equality ignores `block` since upstream compares error
/// values by message.
#[derive(Clone, Debug)]
pub struct AnnotatorError {
    pub msg: Option<String>,
    pub source: Option<String>,
    pub block: Option<usize>,
}

impl AnnotatorError {
    pub fn new(msg: impl Into<String>) -> Self {
        AnnotatorError {
            msg: Some(msg.into()),
            source: None,
            block: None,
        }
    }

    /// upstream `annrpython.py:402-405`:
    ///
    /// ```python
    /// if not hasattr(e, '__annotator_block'):
    ///     setattr(e, '__annotator_block', block)
    /// ```
    ///
    /// Attaches the offending block to the error only on the first
    /// re-raise so the innermost `processblock` wins.
    pub fn set_annotator_block(&mut self, block: &BlockRef) {
        if self.block.is_none() {
            self.block = Some(BlockKey::of(block).as_usize());
        }
    }
}

impl PartialEq for AnnotatorError {
    fn eq(&self, other: &Self) -> bool {
        self.msg == other.msg && self.source == other.source
    }
}

impl Eq for AnnotatorError {}

impl fmt::Display for AnnotatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Line-by-line port of `AnnotatorError.__str__`
        // (`rpython/annotator/model.py`):
        //
        //     def __str__(self):
        //         s = "\n\n%s" % self.msg
        //         if self.source is not None:
        //             s += "\n\n"
        //             s += self.source
        //         return s
        //
        // Upstream's `%s` formats `None` literally as `"None"` when
        // `self.msg is None` — pyre matches via `Option::Display`
        // (None → "None") to keep producer-side telemetry symmetric.
        // The `block` slot is pyre-only diagnostic state; rendering it
        // here would deviate from upstream and is the reporting
        // layer's concern, not the exception's.
        match &self.msg {
            Some(msg) => write!(f, "\n\n{msg}")?,
            None => write!(f, "\n\nNone")?,
        }
        if let Some(source) = &self.source {
            write!(f, "\n\n{source}")?;
        }
        Ok(())
    }
}

impl std::error::Error for AnnotatorError {}

/// Read a caught panic payload as text.
///
/// Upstream's handlers catch a typed exception and read its fields —
/// `flowin` (annrpython.py) catches `AnnotatorError` to attach `source`,
/// `gottypererror` (rtyper.py) catches `TyperError` to attach `where` — and
/// the message text is produced once, at the terminal report (`complete`,
/// `'\n-----'.join(str(e) for e in self.errors)`). A pyre unwind carries
/// either the error object or an already-formatted string, so the object is
/// read first and the string is the fallback.
///
/// Every consumer of an annotator unwind must go through this one reader.
/// A site that hand-rolls `downcast_ref::<String>` reads an object payload
/// as "unrecognised", and the dual gate then treats a routine unported
/// subject as a real bug because [`crate::translator::rtyper::cutover::is_known_unported`]
/// never sees the message.
///
/// `None` is reserved for a payload that is neither — each caller keeps its
/// own wording for that case.
pub fn panic_payload_text(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(err) = payload.downcast_ref::<AnnotatorError>() {
        return Some(err.to_string());
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return Some(text.clone());
    }
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        return Some((*text).to_string());
    }
    None
}

impl AnnotatorError {
    /// Name the annotator entry point that was driving when this error
    /// surfaced, without changing the error's class.
    ///
    /// Upstream raises `AnnotatorError` from inside `compute_at_fixpoint`
    /// and `complete_pending_blocks` and lets it propagate as itself; the
    /// class is what tells a handler which half of the translator failed,
    /// and `gottypererror` (rtyper.py) keeps `TyperError` separate for the
    /// other half.
    /// A caller that reformats one into the other erases that distinction,
    /// after which the stage is recoverable only by matching substrings of
    /// the rendered text.
    pub fn in_stage(mut self, stage: &str) -> Self {
        self.msg = Some(match self.msg.take() {
            Some(msg) => format!("{stage} failed: {msg}"),
            None => format!("{stage} failed"),
        });
        self
    }
}

// union() dispatch (A4.6) — model.py + binaryop.py pair().union().

/// Short identity of a `union` operand for the unhandled-pair error text.
///
/// The bare [`SomeValueTag`] is often not enough to act on: `Instance`
/// covers every classdef, and the failing pair is usually diagnosed by
/// *which* class collided, not by the variant. This renders the one
/// discriminating field for the two payload-bearing variants that show up
/// in unhandled pairs and falls back to the tag for the rest, so the error
/// stays one short line. Same motivation as the `SomeInstance ∪
/// SomeInstance` arm naming its two classdefs.
fn union_operand_id(s: &SomeValue) -> String {
    match s {
        SomeValue::Instance(inst) => match &inst.classdef {
            Some(cd) => format!("Instance({})", cd.borrow().name),
            None => "Instance(classdef-less)".to_string(),
        },
        SomeValue::Ptr(ptr) => {
            use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
            let target = match &ptr.ll_ptrtype.TO {
                PtrTarget::Struct(st) => st._name.clone(),
                PtrTarget::Opaque(op) => op.tag.clone(),
                PtrTarget::Func(_) => "Func".to_string(),
                PtrTarget::Array(_) => "Array".to_string(),
                PtrTarget::FixedSizeArray(_) => "FixedSizeArray".to_string(),
                PtrTarget::ForwardReference(_) => "ForwardReference".to_string(),
            };
            format!("Ptr({target})")
        }
        other => format!("{:?}", other.tag()),
    }
}

/// RPython `union(s1, s2)` (model.py).
///
/// The join operation in the lattice of annotations. Returns the most
/// precise `SomeValue` that contains both inputs, or raises
/// [`UnionError`] when the two are structurally incompatible.
///
/// Upstream dispatches to `pair(s1, s2).union()` which walks a
/// `DoubleDispatchRegistry` populated by `rpython/annotator/binaryop.py`
/// (859 LOC of per-pair handlers). The current port ships a
/// primitive-family subset sufficient to exercise
/// `test_model.py::{test_equality, test_contains, test_list_union,
/// test_list_contains, test_signedness}`; the binaryop port
/// extends this match with the container / PBC / instance pairs.
pub fn union(s1: &SomeValue, s2: &SomeValue) -> Result<SomeValue, UnionError> {
    // upstream: `if s1 == s2: return s1`. Identity short-circuit
    // (comment in model.py notes that most pair().union() methods
    // handle the case incorrectly in the face of constants).
    if s1 == s2 {
        return Ok(s1.clone());
    }
    match (s1, s2) {
        // SomeImpossibleValue is the bottom element: `impossible ∪ X = X`.
        (SomeValue::Impossible, other) | (other, SomeValue::Impossible) => Ok(other.clone()),

        // SomeInteger ∪ SomeInteger — binaryop.py union signedness-
        // proof dispatch. Direct port of the upstream body:
        //
        //     if int1.unsigned == int2.unsigned:
        //         knowntype = compute_restype(int1.knowntype, int2.knowntype)
        //     else:
        //         t1, t2 = int1.knowntype, int2.knowntype
        //         if t1 is bool: t1 = int
        //         if t2 is bool: t2 = int
        //         if t2 is int:
        //             if int2.nonneg == False:
        //                 raise UnionError(int1, int2, "RPython cannot prove …")
        //             knowntype = t1
        //         elif t1 is int:
        //             if int1.nonneg == False:
        //                 raise UnionError(...)
        //             knowntype = t2
        //         else:
        //             raise UnionError(int1, int2)
        //     return SomeInteger(nonneg=int1.nonneg and int2.nonneg,
        //                        knowntype=knowntype)
        (SomeValue::Integer(int1), SomeValue::Integer(int2)) => {
            let knowntype = if int1.unsigned == int2.unsigned {
                crate::rlib::rarithmetic::compute_restype(int1.base.knowntype, int2.base.knowntype)
            } else {
                // bool → int promotion before the signedness compare.
                // SomeInteger itself never carries KnownType::Bool
                // (SomeBool is a distinct enum variant), so the
                // promotion is a no-op here. Kept as a comment for
                // parity visibility.
                let t1 = int1.base.knowntype;
                let t2 = int2.base.knowntype;
                if matches!(t2, KnownType::Int) {
                    if !int2.nonneg {
                        return Err(UnionError {
                            lhs: Box::new(s1.clone()),
                            rhs: Box::new(s2.clone()),
                            msg:
                                "RPython cannot prove that these integers are of the same signedness"
                                    .into(),
                        });
                    }
                    t1
                } else if matches!(t1, KnownType::Int) {
                    if !int1.nonneg {
                        return Err(UnionError {
                            lhs: Box::new(s1.clone()),
                            rhs: Box::new(s2.clone()),
                            msg:
                                "RPython cannot prove that these integers are of the same signedness"
                                    .into(),
                        });
                    }
                    t2
                } else {
                    return Err(UnionError {
                        lhs: Box::new(s1.clone()),
                        rhs: Box::new(s2.clone()),
                        msg: "RPython cannot unify these integer types".into(),
                    });
                }
            };
            // `knowntypedata` on `SomeInteger` is a pyre-only refinement
            // (upstream carries it only on `SomeBool`). Intersect it when
            // both sides carry it — the same invariant the `SomeBool` arm
            // below protects: dropping the merged refinement makes
            // `union(new, old) != new`, which breaks `contains` (and hence
            // `setbinding`) for a discriminant integer that generalises its
            // prior binding.
            let mut si = SomeInteger::new_with_knowntype(int1.nonneg && int2.nonneg, knowntype);
            if let (Some(k1), Some(k2)) = (&int1.knowntypedata, &int2.knowntypedata) {
                si.set_knowntypedata(merge_knowntypedata(k1, k2));
            }
            Ok(SomeValue::Integer(si))
        }

        // pairtype(SomeBool, SomeBool).union() (binaryop.py):
        // keep the constant only when both sides agree, and intersect
        // knowntypedata when both sides carry it. Dropping the merged
        // knowntypedata here would break `contains` (and hence
        // `setbinding`) for refinement-carrying bools: union(new, old)
        // must reproduce `new` when `new` generalises `old`.
        (SomeValue::Bool(b1), SomeValue::Bool(b2)) => {
            let mut s = SomeBool::new();
            if let (Some(c1), Some(c2)) = (&b1.base.const_box, &b2.base.const_box)
                && c1.value == c2.value
            {
                s.base.const_box = Some(Constant::new(c1.value.clone()));
            }
            if let (Some(k1), Some(k2)) = (&b1.knowntypedata, &b2.knowntypedata) {
                s.set_knowntypedata(merge_knowntypedata(k1, k2));
            }
            Ok(SomeValue::Bool(s))
        }

        // SomeBool ↔ SomeInteger: bool is a subtype of int, so the
        // union widens to SomeInteger. Upstream:
        // `__union__(SomeBool, SomeInteger)` / reverse — both return
        // SomeInteger with nonneg=True (because bool is always ≥ 0).
        (SomeValue::Bool(_), SomeValue::Integer(b)) => {
            Ok(SomeValue::Integer(SomeInteger::new(b.nonneg, b.unsigned)))
        }
        (SomeValue::Integer(a), SomeValue::Bool(_)) => {
            Ok(SomeValue::Integer(SomeInteger::new(a.nonneg, a.unsigned)))
        }

        // SomeFloat ↔ SomeFloat (different constants / NaN) → generic
        // SomeFloat. `SomeFloat.__eq__` in model.py:170-180 already
        // handles NaN equality; the fallthrough here covers constants
        // that are not bit-equal but still "the same" float.
        (SomeValue::Float(_), SomeValue::Float(_)) => Ok(SomeValue::Float(SomeFloat::new())),

        // SomeInteger ↔ SomeFloat: upstream's `TLS.allow_int_to_float`
        // defaults to True, so the union widens to SomeFloat.
        (SomeValue::Integer(_), SomeValue::Float(_))
        | (SomeValue::Float(_), SomeValue::Integer(_)) => Ok(SomeValue::Float(SomeFloat::new())),
        (SomeValue::Bool(_), SomeValue::Float(_)) | (SomeValue::Float(_), SomeValue::Bool(_)) => {
            Ok(SomeValue::Float(SomeFloat::new()))
        }

        // SomeString ∪ SomeString — merge can_be_None / no_nul.
        (SomeValue::String(a), SomeValue::String(b)) => Ok(SomeValue::String(SomeString::new(
            a.inner.can_be_none || b.inner.can_be_none,
            a.inner.no_nul && b.inner.no_nul,
        ))),

        // SomeChar ↔ SomeChar (different no_nul) → SomeChar.
        (SomeValue::Char(a), SomeValue::Char(b)) => Ok(SomeValue::Char(SomeChar::new(
            a.inner.no_nul && b.inner.no_nul,
        ))),
        // SomeChar ↔ SomeString — union widens to SomeString (char is
        // a one-character string; upstream
        // `__union__(SomeChar, SomeString)` returns SomeString).
        (SomeValue::Char(a), SomeValue::String(b)) | (SomeValue::String(b), SomeValue::Char(a)) => {
            Ok(SomeValue::String(SomeString::new(
                b.inner.can_be_none,
                a.inner.no_nul && b.inner.no_nul,
            )))
        }

        // SomeUnicodeString ∪ SomeUnicodeString — mirror of String case.
        (SomeValue::UnicodeString(a), SomeValue::UnicodeString(b)) => {
            Ok(SomeValue::UnicodeString(SomeUnicodeString::new(
                a.inner.can_be_none || b.inner.can_be_none,
                a.inner.no_nul && b.inner.no_nul,
            )))
        }
        (SomeValue::UnicodeCodePoint(a), SomeValue::UnicodeCodePoint(b)) => {
            Ok(SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(
                a.inner.no_nul && b.inner.no_nul,
            )))
        }
        (SomeValue::UnicodeCodePoint(a), SomeValue::UnicodeString(b))
        | (SomeValue::UnicodeString(b), SomeValue::UnicodeCodePoint(a)) => {
            Ok(SomeValue::UnicodeString(SomeUnicodeString::new(
                b.inner.can_be_none,
                a.inner.no_nul && b.inner.no_nul,
            )))
        }

        // SomeList ∪ SomeList — upstream dispatches through
        // `pair(SomeList, SomeList).union()` in binaryop.py which
        // calls `self.listdef.union(other.listdef)`.
        // `ListDef::union_with` mutates both listdefs' listitems in
        // place (via interior mutability) and patches sister
        // `ListDefInner`s so post-merge `same_as` returns True.
        //
        // Under `contains()`, the `SideEffectFreeGuard` is active ⇒
        // `merge_items` raises UnionError without mutating, which
        // propagates out so contains returns False. That matches
        // upstream's `TLS.no_side_effects_in_union` semantics.
        (SomeValue::List(a), SomeValue::List(b)) => {
            a.listdef.union_with(&b.listdef)?;
            Ok(SomeValue::List(SomeList::new(a.listdef.clone())))
        }

        // Two Rust `Vec`s union their item annotations. A Rust `Vec` never
        // unions with an RPython list: it is a raw header, not a list.
        (SomeValue::RustVec(a), SomeValue::RustVec(b)) => Ok(SomeValue::RustVec(SomeRustVec::new(
            union(&a.s_item, &b.s_item)?,
        ))),

        // SomeDict ∪ SomeDict — same shape as SomeList: mutate both
        // dictdef's key/value cells in place via
        // `DictDef::union_with`.
        (SomeValue::Dict(a), SomeValue::Dict(b)) => {
            a.dictdef.union_with(&b.dictdef)?;
            Ok(SomeValue::Dict(SomeDict::new(a.dictdef.clone())))
        }

        // SomeTuple ∪ SomeTuple — only same-length tuples can union.
        // Upstream: different-length tuples return SomeObject; we mirror
        // by raising UnionError (A4.6 covers the common case, not the
        // SomeObject widening fallback yet).
        (SomeValue::Tuple(a), SomeValue::Tuple(b)) => {
            if a.items.len() != b.items.len() {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: format!(
                        "SomeTuple length mismatch ({} vs {})",
                        a.items.len(),
                        b.items.len()
                    ),
                });
            }
            let mut items = Vec::with_capacity(a.items.len());
            for (x, y) in a.items.iter().zip(&b.items) {
                items.push(union(x, y)?);
            }
            Ok(SomeValue::Tuple(SomeTuple::new(items)))
        }

        // SomeInstance ∪ SomeInstance — upstream dispatches through
        // `pair(SomeInstance, SomeInstance).union()` in binaryop.py
        // which walks the classdef MRO to find the common base and
        // keeps only flags that agree on both sides.
        (SomeValue::Instance(a), SomeValue::Instance(b)) => {
            let can_be_none = a.can_be_none || b.can_be_none;
            let mut flags = a.flags.clone();
            flags.retain(|key, value| b.flags.get(key) == Some(value));
            // binaryop.py union unions two SomeInstances through
            // `commonbase(classdef1, classdef2)`.  The `classdef is
            // None` case is a special case that yields `basedef = None`;
            // when BOTH sides carry a classdef and `commonbase` returns
            // None, RPython raises `UnionError` — it does NOT widen two
            // unrelated classdefs to the classdef-less top.  The
            // dual-gate catches that UnionError and Skips the graph
            // (legacy fallback), which is the parity-correct outcome.
            let merged_classdef = match (&a.classdef, &b.classdef) {
                (Some(ca), Some(cb)) => match ClassDef::commonbase(ca, cb) {
                    Some(base) => Some(base),
                    None if same_struct_identity(&ca.borrow().name, &cb.borrow().name) => {
                        // One struct reached by two frontend name spellings
                        // (dot-joined crate-included ctor qualname vs the
                        // crate-stripped `::` field-read spelling): two
                        // base-less leaves with no shared base but the same
                        // canonical identity.  Unify to the lhs class.
                        Some(ca.clone())
                    }
                    None if unresolved_option_none_name(&ca.borrow().name)
                        || unresolved_option_none_name(&cb.borrow().name) =>
                    {
                        // `RDict::get` returns `Option<V>`. A pointer payload
                        // is stored as the object itself, and the unresolved
                        // `None` variant is that object's null. The merge is
                        // the payload class, nullable.
                        let payload = if unresolved_option_none_name(&ca.borrow().name) {
                            cb.clone()
                        } else {
                            ca.clone()
                        };
                        return Ok(SomeValue::Instance(SomeInstance::new(
                            Some(payload),
                            true,
                            flags,
                        )));
                    }
                    None => {
                        // Name the two colliding classdefs so the
                        // skip-classified panic path is diagnosable
                        // instead of a blind `<other> ∪ <other>`.  The
                        // "cannot unify instances with no common base
                        // class" phrase is preserved verbatim so the
                        // dual-gate skip classifier still matches.
                        return Err(UnionError {
                            lhs: Box::new(s1.clone()),
                            rhs: Box::new(s2.clone()),
                            msg: format!(
                                "RPython cannot unify instances with no \
                                 common base class: {} ∪ {}",
                                ca.borrow().name,
                                cb.borrow().name
                            ),
                        });
                    }
                },
                // special case only: a classdef-less top instance on
                // either side unions to the classdef-less top.
                _ => None,
            };
            Ok(SomeValue::Instance(SomeInstance::new(
                merged_classdef,
                can_be_none,
                flags,
            )))
        }

        // A classdef-less top instance — pyre's erased "top reference",
        // e.g. a `&Foo` input whose Ref target `valuetype_to_someshell`
        // dropped to `SomeInstance(None)` — absorbs an opaque low-level
        // pointer denoting the same erased reference.  The Ref input
        // lifts inconsistently (classdef-less `SomeInstance` on one CFG
        // edge, `SomePtr` on another) and merges at `mergeinputargs`;
        // both are type-erased pointers, so the union is the classdef-
        // less top — the same result as the `_ => None` special case
        // above where a classdef-less instance absorbs any instance.  A
        // classdef-BEARING instance ∪ `Ptr` stays genuinely incompatible
        // and falls through to the loud default.
        (SomeValue::Instance(inst), SomeValue::Ptr(_))
        | (SomeValue::Ptr(_), SomeValue::Instance(inst))
            if inst.classdef.is_none() =>
        {
            Ok(SomeValue::Instance(SomeInstance::new(
                None,
                inst.can_be_none,
                std::collections::BTreeMap::new(),
            )))
        }

        // binaryop.py forwards exception/instance and exception/None
        // through `SomeException.as_SomeInstance()`.
        (SomeValue::Exception(exc), SomeValue::Instance(inst)) => {
            union(&exc.as_some_instance(), &SomeValue::Instance(inst.clone()))
        }
        (SomeValue::Exception(exc), SomeValue::None_(_)) => {
            union(&exc.as_some_instance(), &s_none())
        }
        (SomeValue::Instance(inst), SomeValue::Exception(exc)) => {
            union(&SomeValue::Instance(inst.clone()), &exc.as_some_instance())
        }
        (SomeValue::None_(_), SomeValue::Exception(exc)) => {
            union(&s_none(), &exc.as_some_instance())
        }

        // SomeException ∪ SomeException — the set of exception
        // classdefs is the **union** of the two sides (upstream
        // binaryop.py::__union__(SomeException, SomeException)).
        (SomeValue::Exception(a), SomeValue::Exception(b)) => {
            let mut classdefs = a.classdefs.clone();
            for c in &b.classdefs {
                if !classdef_vec_contains(&classdefs, c) {
                    classdefs.push(c.clone());
                }
            }
            Ok(SomeValue::Exception(SomeException::new(classdefs)))
        }

        // `pair(SomeIterator, SomeIterator).union()` in binaryop.py:
        // container union plus exact variant match.
        (SomeValue::Iterator(a), SomeValue::Iterator(b)) => {
            if a.variant != b.variant || a.enumerate_start != b.enumerate_start {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: "RPython cannot unify incompatible iterator variants".into(),
                });
            }
            let s_container = union(&a.s_container, &b.s_container)?;
            Ok(SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
                s_container,
                a.variant.clone(),
                a.enumerate_start.clone(),
            )))
        }

        // llannotation.py — low-level pointers union only when
        // they carry the same `ll_ptrtype`.
        (SomeValue::Ptr(a), SomeValue::Ptr(b)) => {
            if a.ll_ptrtype != b.ll_ptrtype {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: "RPython cannot unify distinct low-level pointer types".into(),
                });
            }
            Ok(SomeValue::Ptr(SomePtr::new(a.ll_ptrtype.clone())))
        }
        (SomeValue::InteriorPtr(a), SomeValue::InteriorPtr(b)) => {
            if a.ll_ptrtype != b.ll_ptrtype {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: "RPython cannot unify distinct low-level pointer types".into(),
                });
            }
            Ok(SomeValue::InteriorPtr(SomeInteriorPtr::new(
                a.ll_ptrtype.clone(),
            )))
        }

        // llannotation.py — `pair(SomeAddress, SomeAddress).union()`
        (SomeValue::Address(_), SomeValue::Address(_)) => {
            Ok(SomeValue::Address(SomeAddress::new()))
        }

        // llannotation.py __extend__ — `pair(SomeTypedAddressAccess,
        // SomeTypedAddressAccess).union()`:
        //   `assert s_taa1.type == s_taa2.type; return s_taa1`.
        (SomeValue::TypedAddressAccess(a), SomeValue::TypedAddressAccess(b)) => {
            if a.access_type != b.access_type {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: "cannot unify typed address accesses of distinct types".into(),
                });
            }
            Ok(SomeValue::TypedAddressAccess(a.clone()))
        }

        // `pair(SomeBuiltinMethod, SomeBuiltinMethod).union()` in
        // binaryop.py: analyser/methodname must match; `s_self`
        // widens by union.
        (SomeValue::BuiltinMethod(a), SomeValue::BuiltinMethod(b)) => {
            if a.analyser_name != b.analyser_name || a.methodname != b.methodname {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: "RPython cannot unify distinct builtin methods".into(),
                });
            }
            let s_self = union(&a.s_self, &b.s_self)?;
            Ok(SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                a.analyser_name.clone(),
                s_self,
                a.methodname.clone(),
            )))
        }

        // SomePBC ∪ SomePBC — description set union, subject to the
        // mixed-kind check (upstream: AnnotatorError if the two PBCs
        // have different `getKind()`). can_be_none OR, subset_of
        // forgotten when the sets no longer agree.
        (SomeValue::PBC(a), SomeValue::PBC(b)) => {
            let can_be_none = a.can_be_none || b.can_be_none;
            // upstream model.py: `getKind()` raises
            // `AnnotatorError("mixing several kinds of PBCs")` for a
            // single PBC whose descriptions span multiple Desc
            // subclasses. The union also widens across the two sides
            // — mixed kinds on EITHER side or BETWEEN sides surfaces
            // as `UnionError` here.
            let unpack = |s: &SomePBC, orig: &SomeValue| -> Result<DescKind, UnionError> {
                s.get_kind().map_err(|e| UnionError {
                    lhs: Box::new(orig.clone()),
                    rhs: Box::new(orig.clone()),
                    msg: e
                        .msg
                        .unwrap_or_else(|| "mixing several kinds of PBCs".into()),
                })
            };
            let a_kind = unpack(a, s1)?;
            let b_kind = unpack(b, s2)?;
            if a_kind != b_kind {
                return Err(UnionError {
                    lhs: Box::new(s1.clone()),
                    rhs: Box::new(s2.clone()),
                    msg: format!("mixing several kinds of PBCs: {a_kind:?} and {b_kind:?}"),
                });
            }
            // upstream: `descriptions = a.descriptions | b.descriptions`
            //   (Python set union); keyed-by-identity BTreeMap merge.
            let mut descriptions_map = a.descriptions.clone();
            for (k, v) in &b.descriptions {
                descriptions_map.entry(*k).or_insert_with(|| v.clone());
            }
            let descriptions = descriptions_map.into_values();
            let merged = SomePBC::with_subset(descriptions, can_be_none, None);
            Ok(SomeValue::PBC(merged))
        }

        // SomeWeakRef ∪ SomeWeakRef — upstream widens the classdef to
        // the common base when both sides carry one.
        (SomeValue::WeakRef(a), SomeValue::WeakRef(b)) => {
            let merged_classdef: Option<Rc<RefCell<ClassDef>>> = match (&a.classdef, &b.classdef) {
                (None, _) | (_, None) => None,
                (Some(ca), Some(cb)) => match ClassDef::commonbase(ca, cb) {
                    Some(base) => Some(base),
                    None => {
                        return Err(UnionError {
                            lhs: Box::new(s1.clone()),
                            rhs: Box::new(s2.clone()),
                            msg: format!(
                                "RPython cannot unify weakrefs with no \
                                     common base class: {} ∪ {}",
                                ca.borrow().name,
                                cb.borrow().name
                            ),
                        });
                    }
                },
            };
            Ok(SomeValue::WeakRef(SomeWeakRef::new(merged_classdef)))
        }

        // mixing Nones with other objects: upstream binaryop.py routes
        // these through `obj.noneify()`.
        (SomeValue::None_(_), obj) | (obj, SomeValue::None_(_)) => obj.noneify(),

        // Default fallback.  Reaching it does NOT imply an arm waiting to
        // be ported: upstream's union table is same-kind almost
        // throughout — `binaryop.py` defines `union` on
        // `pairtype(SomeInteger, SomeInteger)`, `(SomeString, SomeString)`,
        // `(SomeList, SomeList)`, `(SomeInstance, SomeInstance)` and their
        // siblings, plus the `SomeNone`/`SomeImpossibleValue` bridges and a
        // single cross-kind `(SomeUnicodeString, SomeInteger)`.  Everything
        // else resolves through the MRO to `pairtype(SomeObject, SomeObject)
        // .union`, which raises; the ll level is explicit about it, with
        // `llannotation.py`'s `pairtype(SomePtr, SomeObject).union` and its
        // `SomeAddress` twin raising in their own bodies.
        //
        // So a CROSS-kind pair here is a producer divergence — two levels of
        // one value met at a merge upstream never builds — and porting an arm
        // for it would invent a lattice join RPython does not have.  A
        // SAME-kind pair is the porting gap.  `classify_unported_reason`
        // splits the two on the operand names below, so the message must keep
        // naming both sides.
        _ => Err(UnionError {
            lhs: Box::new(s1.clone()),
            rhs: Box::new(s2.clone()),
            // Name the two colliding operands, for the same reason the
            // `SomeInstance` arm above names its classdefs: both sides
            // often render as the bare `<other>` `KnownType`, which makes
            // the skip-classified panic undiagnosable.  The classifier
            // phrase is kept verbatim and first — `cutover.rs`'s `is_known_unported`
            // and `classify_unported_reason` match it with `contains`.
            msg: format!(
                "pair(s1, s2).union() found no arm: {} ∪ {}",
                union_operand_id(s1),
                union_operand_id(s2)
            ),
        }),
    }
}

/// RPython `not_const(s_obj)` (model.py).
///
/// ```python
/// def not_const(s_obj):
///     if s_obj.is_constant() and not isinstance(s_obj, (SomePBC, SomeNone)):
///         new_s_obj = SomeObject.__new__(s_obj.__class__)
///         dic = new_s_obj.__dict__ = s_obj.__dict__.copy()
///         if 'const' in dic:
///             del new_s_obj.const
///         else:
///             del new_s_obj.const_box
///         s_obj = new_s_obj
///     return s_obj
/// ```
///
/// Rust port clears `SomeObject.const_box` on the cloned variant;
/// `SomePBC` / `SomeNone` are explicitly excluded (upstream's constancy
/// on these is structural, not stored in `const_box`).
pub fn not_const(s: &SomeValue) -> SomeValue {
    if !s.is_constant() {
        return s.clone();
    }
    if matches!(s, SomeValue::PBC(_) | SomeValue::None_(_)) {
        return s.clone();
    }
    let mut cloned = s.clone();
    match &mut cloned {
        SomeValue::Object(b) => b.const_box = None,
        SomeValue::Type(v) => v.base.const_box = None,
        SomeValue::Float(v) => v.base.const_box = None,
        SomeValue::SingleFloat(v) => v.base.const_box = None,
        SomeValue::LongFloat(v) => v.base.const_box = None,
        SomeValue::Integer(v) => v.base.const_box = None,
        SomeValue::Bool(v) => v.base.const_box = None,
        SomeValue::String(v) => v.inner.base.const_box = None,
        SomeValue::UnicodeString(v) => v.inner.base.const_box = None,
        SomeValue::ByteArray(v) => v.inner.base.const_box = None,
        SomeValue::Char(v) => v.inner.base.const_box = None,
        SomeValue::UnicodeCodePoint(v) => v.inner.base.const_box = None,
        SomeValue::List(v) => v.base.const_box = None,
        SomeValue::Tuple(v) => v.base.const_box = None,
        SomeValue::Dict(v) => v.base.const_box = None,
        SomeValue::Iterator(v) => v.base.const_box = None,
        SomeValue::Instance(v) => v.base.const_box = None,
        SomeValue::Exception(v) => v.base.const_box = None,
        SomeValue::Property(v) => v.base.const_box = None,
        SomeValue::Ptr(v) => v.base.const_box = None,
        SomeValue::InteriorPtr(v) => v.base.const_box = None,
        SomeValue::Address(v) => v.base.const_box = None,
        SomeValue::TypedAddressAccess(v) => v.base.const_box = None,
        SomeValue::LLADTMeth(v) => v.base.const_box = None,
        SomeValue::ExternalFunction(v) => v.base.const_box = None,
        SomeValue::Builtin(v) => v.base.const_box = None,
        SomeValue::BuiltinMethod(v) => v.base.const_box = None,
        SomeValue::WeakRef(v) => v.base.const_box = None,
        SomeValue::TypeOf(v) => v.base.const_box = None,
        SomeValue::StringBuilder(v) => v.base.const_box = None,
        SomeValue::UnicodeBuilder(v) => v.base.const_box = None,
        SomeValue::RustVec(v) => v.base.const_box = None,
        // Impossible / PBC / None_ handled above (early-return).
        SomeValue::Impossible | SomeValue::PBC(_) | SomeValue::None_(_) => unreachable!(),
    }
    cloned
}

/// RPython `unionof(*somevalues)` (model.py). Returns the most
/// precise `SomeValue` containing every input; panics propagate
/// [`UnionError`] per upstream's implicit `Exception` raise.
pub fn unionof<'a, I: IntoIterator<Item = &'a SomeValue>>(
    somevalues: I,
) -> Result<SomeValue, UnionError> {
    let mut acc = SomeValue::Impossible;
    for s in somevalues {
        if acc != *s {
            acc = union(&acc, s)?;
        }
    }
    Ok(acc)
}

/// RPython `SomeObject.contains(other)` now that `union` exists
/// (model.py:94-100). Supersedes the A4.1 equality-only fallback.
pub(crate) fn contains(a: &SomeValue, b: &SomeValue) -> bool {
    if a == b {
        return true;
    }
    // RPython `SomeObject.contains` wraps `union(self, other)` inside
    // `TLS.no_side_effects_in_union += 1` so any `ListItem.merge` /
    // `DictKey.merge` call raises UnionError rather than mutating.
    // Mirror that contract via the Rust `SideEffectFreeGuard`.
    let _guard = super::listdef::SideEffectFreeGuard::enter();
    match union(a, b) {
        Ok(u) => &u == a,
        Err(_) => false,
    }
}

/// RPython `intersection(s_obj1, s_obj2)` (model.py +
/// doubledispatch registrations at 464-512).
///
/// ```python
/// @doubledispatch
/// def intersection(s_obj1, s_obj2):
///     """Return the intersection of two annotations, or an over-approximation thereof"""
///     raise NotImplementedError
/// ```
///
/// The @doubledispatch default raises NotImplementedError; the concrete
/// overloads in model.py cover (SomeInstance, SomeInstance),
/// (SomeException, SomeInstance), (SomeInstance, SomeException).
/// The Rust port preserves the strict contract: unregistered pairs
/// panic rather than silently widen.
pub fn intersection(s_obj1: &SomeValue, s_obj2: &SomeValue) -> SomeValue {
    match (s_obj1, s_obj2) {
        // @intersection.register(SomeInstance, SomeInstance) — model.py.
        (SomeValue::Instance(s_inst1), SomeValue::Instance(s_inst2)) => {
            intersection_Instance(s_inst1, s_inst2)
        }

        // @intersection.register(SomeException, SomeInstance) — model.py.
        (SomeValue::Exception(s_exc), SomeValue::Instance(s_inst)) => {
            intersection_Exception_Instance(s_exc, s_inst)
        }

        // @intersection.register(SomeInstance, SomeException) — model.py.
        //
        // def intersection_Exception_Instance(s_inst, s_exc):
        //     return intersection(s_exc, s_inst)
        (SomeValue::Instance(_), SomeValue::Exception(_)) => intersection(s_obj2, s_obj1),

        // @doubledispatch default: `raise NotImplementedError`.
        _ => panic!(
            "intersection: NotImplementedError for ({:?}, {:?})",
            s_obj1.knowntype(),
            s_obj2.knowntype()
        ),
    }
}

/// RPython `difference(s_obj1, s_obj2)` (model.py +
/// doubledispatch registrations at 474-512).
///
/// ```python
/// @doubledispatch
/// def difference(s_obj1, s_obj2):
///     """Return the set difference of two annotations, or an over-approximation thereof"""
///     raise NotImplementedError
/// ```
pub fn difference(s_obj1: &SomeValue, s_obj2: &SomeValue) -> SomeValue {
    match (s_obj1, s_obj2) {
        // @difference.register(SomeInstance, SomeInstance) — model.py.
        (SomeValue::Instance(s_inst1), SomeValue::Instance(s_inst2)) => {
            difference_Instance_Instance(s_inst1, s_inst2)
        }

        // @difference.register(SomeException, SomeInstance) — model.py.
        (SomeValue::Exception(s_exc), SomeValue::Instance(s_inst)) => {
            difference_Exception_Instance(s_exc, s_inst)
        }

        _ => panic!(
            "difference: NotImplementedError for ({:?}, {:?})",
            s_obj1.knowntype(),
            s_obj2.knowntype()
        ),
    }
}

/// RPython `intersection_Instance` registered for
/// `(SomeInstance, SomeInstance)` (model.py).
#[allow(non_snake_case)]
pub fn intersection_Instance(s_inst1: &SomeInstance, s_inst2: &SomeInstance) -> SomeValue {
    let can_be_none = s_inst1.can_be_none && s_inst2.can_be_none;
    let (cd1, cd2) = match (&s_inst1.classdef, &s_inst2.classdef) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            // upstream treats None classdef (the "generic SomeInstance")
            // as the top — intersect returns the more specific side.
            let classdef = s_inst1
                .classdef
                .clone()
                .or_else(|| s_inst2.classdef.clone());
            return SomeValue::Instance(SomeInstance::new(
                classdef,
                can_be_none,
                std::collections::BTreeMap::new(),
            ));
        }
    };
    if cd1.borrow().issubclass(cd2) {
        SomeValue::Instance(SomeInstance::new(
            Some(Rc::clone(cd1)),
            can_be_none,
            std::collections::BTreeMap::new(),
        ))
    } else if cd2.borrow().issubclass(cd1) {
        SomeValue::Instance(SomeInstance::new(
            Some(Rc::clone(cd2)),
            can_be_none,
            std::collections::BTreeMap::new(),
        ))
    } else {
        s_impossible_value()
    }
}

/// RPython `difference_Instance_Instance` registered for
/// `(SomeInstance, SomeInstance)` (model.py).
#[allow(non_snake_case)]
pub fn difference_Instance_Instance(s_inst1: &SomeInstance, s_inst2: &SomeInstance) -> SomeValue {
    match (&s_inst1.classdef, &s_inst2.classdef) {
        (Some(cd1), Some(cd2)) if cd1.borrow().issubclass(cd2) => s_impossible_value(),
        _ => SomeValue::Instance(s_inst1.clone()),
    }
}

/// RPython `intersection_Exception_Instance` registered for
/// `(SomeException, SomeInstance)` (model.py).
#[allow(non_snake_case)]
pub fn intersection_Exception_Instance(s_exc: &SomeException, s_inst: &SomeInstance) -> SomeValue {
    let Some(target) = &s_inst.classdef else {
        // Upstream's classdef is always populated for exception exits;
        // an unclassified SomeInstance degrades to the full set.
        return SomeValue::Exception(s_exc.clone());
    };
    let classdefs: Vec<Rc<RefCell<ClassDef>>> = s_exc
        .classdefs
        .iter()
        .filter(|c| c.borrow().issubclass(target))
        .cloned()
        .collect();
    if classdefs.is_empty() {
        s_impossible_value()
    } else {
        SomeValue::Exception(SomeException::new(classdefs))
    }
}

/// RPython `difference_Exception_Instance` registered for
/// `(SomeException, SomeInstance)` (model.py).
#[allow(non_snake_case)]
pub fn difference_Exception_Instance(s_exc: &SomeException, s_inst: &SomeInstance) -> SomeValue {
    let Some(target) = &s_inst.classdef else {
        // With no concrete classdef, the exclusion matches nothing.
        return SomeValue::Exception(s_exc.clone());
    };
    let classdefs: Vec<Rc<RefCell<ClassDef>>> = s_exc
        .classdefs
        .iter()
        .filter(|c| !c.borrow().issubclass(target))
        .cloned()
        .collect();
    if classdefs.is_empty() {
        s_impossible_value()
    } else {
        SomeValue::Exception(SomeException::new(classdefs))
    }
}

/// RPython `class HarmlesslyBlocked(Exception)` (model.py).
///
/// ```python
/// class HarmlesslyBlocked(Exception):
///     """Raised by the unaryop/binaryop to signal a harmless kind of
///     BlockedInference: the current block is blocked, but not in a way
///     that gives 'Blocked block' errors at the end of annotation."""
/// ```
///
/// Marker type — carries no payload. Used as a zero-sized Result error
/// so the flowin except-clause can distinguish harmless blocks.
#[derive(Clone, Debug)]
pub struct HarmlesslyBlocked;

impl core::fmt::Display for HarmlesslyBlocked {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HarmlesslyBlocked")
    }
}

impl std::error::Error for HarmlesslyBlocked {}

/// Either `AnnotatorError` or `HarmlesslyBlocked`.
///
/// In upstream RPython, these are two independent exception classes
/// (`AnnotatorError` / `HarmlesslyBlocked`, both `Exception` subclasses)
/// and they propagate up through the same call stack. Rust functions
/// that can raise *either* (e.g. `ClassDef.s_getattr` at
/// classdesc.py — which raises `HarmlesslyBlocked` in the
/// enforced-attrs branch) return this union so the flowin loop can
/// discriminate them without collapsing `HarmlesslyBlocked` into a
/// generic `AnnotatorError`.
#[derive(Clone, Debug)]
pub enum AnnotatorException {
    /// upstream `AnnotatorError(...)`.
    Annotator(AnnotatorError),
    /// upstream `HarmlesslyBlocked(...)`.
    Harmless(HarmlesslyBlocked),
}

impl From<AnnotatorError> for AnnotatorException {
    fn from(e: AnnotatorError) -> Self {
        AnnotatorException::Annotator(e)
    }
}

impl From<HarmlesslyBlocked> for AnnotatorException {
    fn from(e: HarmlesslyBlocked) -> Self {
        AnnotatorException::Harmless(e)
    }
}

impl fmt::Display for AnnotatorException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnnotatorException::Annotator(e) => e.fmt(f),
            AnnotatorException::Harmless(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for AnnotatorException {}

/// RPython `read_can_only_throw(opimpl, *args)` (model.py).
///
/// ```python
/// def read_can_only_throw(opimpl, *args):
///     can_only_throw = getattr(opimpl, "can_only_throw", None)
///     if can_only_throw is None or isinstance(can_only_throw, list):
///         return can_only_throw
///     return can_only_throw(*args)
/// ```
pub fn read_can_only_throw(
    can_only_throw: &crate::flowspace::operation::CanOnlyThrow,
    args_s: &[SomeValue],
) -> Option<Vec<crate::flowspace::operation::BuiltinException>> {
    use crate::flowspace::operation::CanOnlyThrow;
    match can_only_throw {
        CanOnlyThrow::Absent => None,
        CanOnlyThrow::List(xs) => Some(xs.clone()),
        CanOnlyThrow::Callable(f) => f(args_s),
    }
}

/// RPython `add_knowntypedata(ktd, truth, vars, s_obj)` (model.py).
///
/// Populates an exitcase-keyed table tracking "if this branch takes the
/// `case` exit, these variables have annotation `s_obj`". RPython keys
/// by the `link.exitcase` value, which for a bool exitswitch is
/// `True`/`False` and for an integer exitswitch is the case constant.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn add_knowntypedata(
    ktd: &mut KnownTypeData,
    case: ExitCaseKey,
    vars: &[Rc<Variable>],
    s_obj: SomeValue,
) {
    let entry = ktd.entry(case).or_default();
    for v in vars {
        entry.insert(Rc::clone(v), s_obj.clone());
    }
}

/// RPython `merge_knowntypedata(ktd1, ktd2)` (model.py).
///
/// Intersection of the two tables: a variable survives only if both
/// branches refined it, and the resulting annotation is the union of
/// the two refinements.
fn instance_class_name(s: &SomeValue) -> Option<String> {
    let SomeValue::Instance(inst) = s else {
        return None;
    };
    inst.classdef.as_ref().map(|cd| cd.borrow().name.clone())
}

/// When two discriminant refinements are the same variant leaf and only
/// one spelling is monomorphized, keep that spelling.
fn prefer_instantiated_variant(s1: &SomeValue, s2: &SomeValue) -> Option<SomeValue> {
    let n1 = instance_class_name(s1)?;
    let n2 = instance_class_name(s2)?;
    let bare_leaf = |n: &str| -> String {
        let leaf = n.rsplit("::").next().unwrap_or(n);
        leaf.split('<').next().unwrap_or(leaf).to_string()
    };
    if bare_leaf(&n1) != bare_leaf(&n2) {
        return None;
    }
    match (n1.contains('<'), n2.contains('<')) {
        (true, false) => Some(s1.clone()),
        (false, true) => Some(s2.clone()),
        _ => None,
    }
}

pub fn merge_knowntypedata(ktd1: &KnownTypeData, ktd2: &KnownTypeData) -> KnownTypeData {
    let mut r: KnownTypeData = std::collections::HashMap::new();
    for (case, constraints) in ktd1 {
        let Some(other) = ktd2.get(case) else {
            continue;
        };
        for (v, s1) in constraints {
            let Some(s2) = other.get(v) else { continue };
            if let Ok(u) = unionof([s1, s2]) {
                r.entry(*case).or_default().insert(Rc::clone(v), u);
            } else if let Some(kept) = prefer_instantiated_variant(s1, s2) {
                // `Result::Ok` and `Result<T,E>::Ok` do not union, but they
                // are one variant.  Keeping the instantiation preserves the
                // match-arm payload; dropping it makes `__pos_0` meet on the
                // enum base.
                r.entry(*case).or_default().insert(Rc::clone(v), kept);
            }
        }
    }
    r
}

/// RPython `typeof(args_v)` (model.py).
///
/// Builds a [`SomeTypeOf`] carrying the provided variables, with a
/// fast path that pins `.const` when the single argument is a
/// `SomeException` whose classdefs singleton identifies the type.
pub fn r#typeof(args_v: &[Rc<Variable>]) -> SomeValue {
    if args_v.is_empty() {
        return SomeValue::Type(SomeType::new());
    }
    let result = SomeTypeOf::new(args_v.iter().map(Rc::clone).collect());
    // TODO(Commit 2+): Exception-const fast path (model.py)
    // requires downcasting `v.annotation` to `SomeValue::Exception`
    // and projecting `classdefs -> classdesc.pyobj` onto `result.const`.
    // The HostObject plumbing for that path lands with the exception-
    // handling commits.
    SomeValue::TypeOf(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::flowspace::model::GraphFunc;
    use std::rc::Rc;

    #[test]
    fn rust_vec_unions_its_items_and_never_meets_a_list() {
        use majit_ir::rvec::VecItemKind;
        let a = SomeValue::RustVec(SomeRustVec::new(SomeValue::Integer(SomeInteger::new(
            true, false,
        ))));
        let b = SomeValue::RustVec(SomeRustVec::for_kind(VecItemKind::Int));
        let SomeValue::RustVec(joined) = union(&a, &b).expect("RustVec ∪ RustVec") else {
            panic!("RustVec ∪ RustVec must stay a RustVec");
        };
        assert_eq!(joined.item_kind(), Some(VecItemKind::Int));
        assert_eq!(
            *joined.s_item,
            SomeValue::Integer(SomeInteger::new(false, false))
        );
        let list = SomeValue::List(SomeList::new(ld(SomeValue::Integer(SomeInteger::new(
            false, false,
        )))));
        assert!(union(&b, &list).is_err());
        assert!(union(&list, &b).is_err());
        for kind in VecItemKind::ALL {
            assert_eq!(SomeRustVec::for_kind(kind).item_kind(), Some(kind));
        }
    }

    /// `ListDef::new(None, s_item, false, false)` shortcut — matches
    /// upstream `ListDef(None, s_item)` signature for test fixtures.
    fn ld(s_item: SomeValue) -> ListDef {
        ListDef::new(None, s_item, false, false)
    }

    /// `ListDef::new(Some(<bk>), s_item, false, false)` shortcut —
    /// matches upstream `ListDef('dummy', s_item)` signature.
    fn ld_mutable(s_item: SomeValue) -> ListDef {
        ListDef::new(
            Some(Rc::new(super::super::bookkeeper::Bookkeeper::new())),
            s_item,
            false,
            false,
        )
    }

    /// Build a stub [`DescEntry::Function`] for SomePBC lattice tests.
    ///
    /// RPython tests that exercise `SomePBC` membership build fresh
    /// `FunctionDesc` instances via `getdesc(pyobj)`. The Rust port
    /// mirrors that path: each call produces a distinct-identity
    /// `FunctionDesc`, which is what upstream would also produce for
    /// fresh pyobjs. Callers that need two PBCs to share a description
    /// (upstream: pyobj identity round-trips through `getdesc`) should
    /// `.clone()` the returned `DescEntry`.
    fn fake_function_entry(
        bk: &Rc<super::super::bookkeeper::Bookkeeper>,
        name: &str,
    ) -> super::super::description::DescEntry {
        use super::super::super::flowspace::argument::Signature;
        use super::super::description::{DescEntry, FunctionDesc};
        use std::cell::RefCell;
        DescEntry::function(Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))))
    }

    /// Build a stub [`DescEntry::Memo`] (a MemoDesc wrapping a
    /// FunctionDesc) for SomePBC mixed-kind tests.
    fn fake_memo_entry(
        bk: &Rc<super::super::bookkeeper::Bookkeeper>,
        name: &str,
    ) -> super::super::description::DescEntry {
        use super::super::super::flowspace::argument::Signature;
        use super::super::description::{DescEntry, FunctionDesc, MemoDesc};
        use std::cell::RefCell;
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        )));
        DescEntry::memo(Rc::new(RefCell::new(MemoDesc::new(fd))))
    }

    /// Build a stub [`DescEntry::Class`] for SomePBC lattice tests.
    fn fake_class_entry(
        bk: &Rc<super::super::bookkeeper::Bookkeeper>,
        name: &str,
    ) -> super::super::description::DescEntry {
        use super::super::super::flowspace::model::HostObject;
        use super::super::description::DescEntry;
        let pyobj = HostObject::new_class(name, vec![]);
        let classdesc =
            super::super::classdesc::ClassDesc::new(bk, pyobj, Some(name.to_string()), None, None)
                .expect("fake_class_entry: ClassDesc::new");
        DescEntry::Class(classdesc)
    }

    #[test]
    fn someobject_defaults_match_upstream() {
        let s = SomeValue::object();
        assert_eq!(s.knowntype(), KnownType::Object);
        assert!(!s.immutable());
        assert!(!s.is_constant());
        assert!(s.can_be_none());
    }

    #[test]
    fn commonbase_matches_known_type_subset() {
        assert_eq!(commonbase(KnownType::Bool, KnownType::Int), KnownType::Int);
        assert_eq!(commonbase(KnownType::Int, KnownType::Bool), KnownType::Int);
        assert_eq!(commonbase(KnownType::Str, KnownType::Str), KnownType::Str);
        assert_eq!(
            commonbase(KnownType::Int, KnownType::Float),
            KnownType::Object
        );
    }

    #[test]
    fn someinteger_nonneg_implied_by_unsigned() {
        // upstream: `self.nonneg = unsigned or nonneg` at model.py.
        let s = SomeInteger::new(false, true);
        assert!(s.nonneg);
        assert!(s.unsigned);
        assert_eq!(s.knowntype(), KnownType::Ruint);
        assert!(!s.can_be_none());
    }

    #[test]
    fn someinteger_signed_by_default() {
        let s = SomeInteger::default();
        assert!(!s.nonneg);
        assert!(!s.unsigned);
        assert_eq!(s.knowntype(), KnownType::Int);
    }

    #[test]
    fn somebool_matches_upstream_shape() {
        let s = SomeBool::new();
        assert_eq!(s.knowntype(), KnownType::Bool);
        assert!(s.immutable());
        assert!(!s.can_be_none());
    }

    #[test]
    fn somefloat_and_siblings_are_immutable() {
        for s in [
            SomeValue::Float(SomeFloat::new()),
            SomeValue::SingleFloat(SomeSingleFloat::new()),
            SomeValue::LongFloat(SomeLongFloat::new()),
        ] {
            assert!(s.immutable());
            assert!(!s.can_be_none());
        }
    }

    #[test]
    fn somefloat_constant_eq_matches_upstream_nan_and_signed_zero() {
        let mut nan1 = SomeFloat::new();
        nan1.base.const_box = Some(Constant::new(
            super::super::super::flowspace::model::ConstValue::Float(f64::NAN.to_bits()),
        ));
        let mut nan2 = SomeFloat::new();
        nan2.base.const_box = Some(Constant::new(
            super::super::super::flowspace::model::ConstValue::Float(
                f64::from_bits(0x7ff8_0000_0000_0001).to_bits(),
            ),
        ));
        assert_eq!(nan1, nan2);

        let mut pos_zero = SomeFloat::new();
        pos_zero.base.const_box = Some(Constant::new(
            super::super::super::flowspace::model::ConstValue::Float(0.0f64.to_bits()),
        ));
        let mut neg_zero = SomeFloat::new();
        neg_zero.base.const_box = Some(Constant::new(
            super::super::super::flowspace::model::ConstValue::Float((-0.0f64).to_bits()),
        ));
        assert_ne!(pos_zero, neg_zero);
    }

    #[test]
    fn somestring_can_be_none_flag_round_trips() {
        let s = SomeString::new(true, false);
        assert!(s.can_be_none());
        let s = SomeString::new(false, true);
        assert!(!s.can_be_none());
        assert!(s.inner.no_nul);
    }

    #[test]
    fn somebytearray_is_mutable() {
        let s = SomeByteArray::new(true);
        assert!(!s.immutable());
        assert_eq!(s.knowntype(), KnownType::Bytearray);
    }

    #[test]
    fn somechar_is_not_nullable() {
        let s = SomeChar::new(true);
        assert!(!s.can_be_none());
        assert!(s.inner.no_nul);
        // SomeChar shares knowntype `str` with SomeString.
        assert_eq!(s.knowntype(), KnownType::Str);
    }

    #[test]
    fn someunicodecodepoint_is_unicode_flavoured() {
        let s = SomeUnicodeCodePoint::new(false);
        assert_eq!(s.knowntype(), KnownType::Unicode);
        assert!(!s.can_be_none());
    }

    #[test]
    fn contains_uses_union_dispatch() {
        // A4.6 now delegates `SomeValue::contains` to the union-powered
        // helper (model.py). Signed int contains all nonneg
        // ints; the reverse fails.
        let signed = SomeValue::Integer(SomeInteger::new(false, false));
        let nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        assert!(signed.contains(&nonneg));
        assert!(signed.contains(&signed));
        assert!(!nonneg.contains(&signed));
    }

    #[test]
    fn impossible_can_not_be_none() {
        let s = SomeValue::Impossible;
        assert!(!s.can_be_none());
        assert!(!s.is_constant());
    }

    #[test]
    fn somelist_matches_upstream_defaults() {
        // upstream: `SomeList.can_be_none() → True`,
        // `SomeList.knowntype = list`, `immutable = False`.
        let listdef = ld(SomeValue::Integer(SomeInteger::default()));
        let s = SomeValue::List(SomeList::new(listdef));
        assert!(!s.immutable());
        assert!(s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::List);
    }

    #[test]
    fn somelist_eq_uses_listdef_same_as() {
        // upstream: `SomeList.__eq__` (model.py:339-348) calls
        // `self.listdef.same_as(other.listdef)`. Identity is based on
        // `Rc::ptr_eq` over the listitem cell, so two independently
        // constructed SomeLists compare UN-equal even when their
        // element types agree — matches upstream exactly.
        let a = SomeList::new(ld(SomeValue::Integer(SomeInteger::default())));
        let b = SomeList::new(ld(SomeValue::Integer(SomeInteger::default())));
        let c = SomeList::new(ld(SomeValue::Float(SomeFloat::new())));
        // Distinct ListDefs, regardless of element-type equality, are
        // never same_as.
        assert_ne!(a, b);
        assert_ne!(a, c);
        // Shared listdef identity (via Clone) → same_as = true.
        let d = SomeList::new(a.listdef.clone());
        assert_eq!(a, d);
    }

    #[test]
    fn sometuple_all_constant_items_propagates_is_constant() {
        // upstream: `SomeTuple.__init__` sets `self.const` when every
        // item is constant (model.py). A SomeInteger with a
        // const_box counts as constant; a vanilla one does not.
        let mut item = SomeInteger::default();
        item.base.const_box = Some(Constant::new(
            super::super::super::flowspace::model::ConstValue::Int(7),
        ));
        let tuple_all_const =
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(item.clone())]));
        assert!(tuple_all_const.is_constant());
        assert_eq!(
            tuple_all_const.const_(),
            Some(&super::super::super::flowspace::model::ConstValue::Tuple(
                vec![super::super::super::flowspace::model::ConstValue::Int(7),]
            ))
        );
        // An empty tuple is trivially "all constant" and so counts as
        // a constant tuple too (matches upstream's empty-loop for-else).
        let tuple_empty = SomeValue::Tuple(SomeTuple::new(Vec::new()));
        assert!(tuple_empty.is_constant());
        assert_eq!(
            tuple_empty.const_(),
            Some(&super::super::super::flowspace::model::ConstValue::Tuple(
                vec![]
            ))
        );
        // One non-constant item breaks the invariant.
        let tuple_mixed = SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(item),
            SomeValue::Integer(SomeInteger::default()),
        ]));
        assert!(!tuple_mixed.is_constant());
        assert_eq!(tuple_mixed.const_(), None);
    }

    #[test]
    fn sometuple_is_immutable_and_not_nullable() {
        let s = SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Impossible]));
        assert!(s.immutable());
        assert!(!s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::Tuple);
    }

    #[test]
    fn somedict_matches_upstream_defaults() {
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let s = SomeValue::Dict(SomeDict::new(dictdef));
        assert!(!s.immutable());
        assert!(s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::OrderedDict);
    }

    #[test]
    fn someiterator_wraps_container_and_is_not_nullable() {
        let s = SomeValue::Iterator(SomeIterator::new(
            SomeValue::List(SomeList::new(ld(
                SomeValue::Integer(SomeInteger::default()),
            ))),
            vec!["items".to_string()],
        ));
        assert!(!s.can_be_none());
        assert!(!s.immutable());
    }

    #[test]
    fn union_iterators_require_matching_variant_and_union_container() {
        let a = SomeValue::Iterator(SomeIterator::new(
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::new(true, false),
            )))),
            vec!["items".to_string()],
        ));
        let b = SomeValue::Iterator(SomeIterator::new(
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::new(false, false),
            )))),
            vec!["items".to_string()],
        ));
        let merged = union(&a, &b).unwrap();
        let SomeValue::Iterator(iter) = merged else {
            panic!("expected SomeIterator");
        };
        let SomeValue::List(list) = *iter.s_container else {
            panic!("expected SomeList container");
        };
        let SomeValue::Integer(elem) = list.listdef.s_value() else {
            panic!("expected SomeInteger element");
        };
        assert!(!elem.nonneg);

        let keys = SomeValue::Iterator(SomeIterator::new(
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::default(),
            )))),
            vec!["keys".to_string()],
        ));
        assert!(union(&a, &keys).is_err());
    }

    #[test]
    fn union_iterators_require_matching_enumerate_start_payload() {
        let a = SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::default(),
            )))),
            vec!["enumerate".to_string()],
            Some(crate::flowspace::model::ConstValue::Int(1)),
        ));
        let b = SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::default(),
            )))),
            vec!["enumerate".to_string()],
            Some(crate::flowspace::model::ConstValue::Int(2)),
        ));
        assert!(union(&a, &b).is_err());
    }

    #[test]
    fn someinstance_tracks_can_be_none_flag() {
        let cdef = Some(ClassDef::new_standalone("my_pkg.C", None));
        let s = SomeValue::Instance(SomeInstance::new(
            cdef.clone(),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert!(!s.can_be_none());
        let s = SomeValue::Instance(SomeInstance::new(
            cdef,
            true,
            std::collections::BTreeMap::new(),
        ));
        assert!(s.can_be_none());
    }

    #[test]
    fn someinstance_knowntype_is_classdesc_identity() {
        // `SomeInstance.knowntype = classdef.classdesc if classdef else None`.
        let classdef_less = SomeValue::Instance(SomeInstance::new(
            None,
            false,
            std::collections::BTreeMap::new(),
        ));
        assert_eq!(classdef_less.knowntype(), KnownType::Instance(None));

        let a = ClassDef::new_standalone("pkg.A", None);
        let b = ClassDef::new_standalone("pkg.B", None);
        let s_a = SomeValue::Instance(SomeInstance::new(
            Some(a.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_a2 = SomeValue::Instance(SomeInstance::new(
            Some(a),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_b = SomeValue::Instance(SomeInstance::new(
            Some(b),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert_eq!(s_a.knowntype(), s_a2.knowntype());
        assert_ne!(s_a.knowntype(), s_b.knowntype());
        assert!(matches!(s_a.knowntype(), KnownType::Instance(Some(_))));
    }

    #[test]
    fn someiterator_knowntype_is_list_iterator() {
        // `SomeIterator.knowntype = type(iter([]))` (`model.py`).
        let s = SomeValue::Iterator(SomeIterator::new(
            SomeValue::Integer(SomeInteger::default()),
            vec![],
        ));
        assert_eq!(s.knowntype(), KnownType::Iterator);
        assert_eq!(s.knowntype().to_string(), "list_iterator");
    }

    #[test]
    fn someexception_knowntype_inherits_object() {
        // `SomeException` does not override `SomeObject.knowntype`.
        let ve = ClassDef::new_standalone("ValueError", None);
        let s = SomeValue::Exception(SomeException::new(vec![ve]));
        assert_eq!(s.knowntype(), KnownType::Object);
    }

    #[test]
    fn somepbc_get_kind_detects_single_family() {
        // upstream: `getKind` returns the unique kind of the set.
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let pbc = SomePBC::new(
            vec![fake_function_entry(&bk, "a"), fake_function_entry(&bk, "b")],
            false,
        );
        assert_eq!(pbc.get_kind().unwrap(), DescKind::Function);
    }

    #[test]
    fn somepbc_get_kind_errors_on_mixed_set() {
        // upstream: raises AnnotatorError when kinds differ
        // (model.py:565).
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let pbc = SomePBC::new(
            vec![fake_function_entry(&bk, "a"), fake_class_entry(&bk, "B")],
            false,
        );
        let err = pbc.get_kind().expect_err("mixed kinds must error");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("mixing several kinds of PBCs")
        );
    }

    #[test]
    fn somepbc_single_desc_without_pyobj_is_not_constant() {
        // upstream: model.py — is_constant holds only when
        // `len(descriptions) == 1 and not can_be_None and desc.pyobj
        // is not None`. `fake_function_entry` creates a FunctionDesc
        // with `pyobj = None`, so a single-desc SomePBC built from it
        // has no witnessed Python value and therefore must NOT claim
        // constant-ness — that was the pyobj gate the A4.5
        // approximation was missing.
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let pbc = SomePBC::new(vec![fake_function_entry(&bk, "f")], false);
        assert!(!pbc.is_constant());
        let pbc_nullable = SomePBC::new(vec![fake_function_entry(&bk, "f")], true);
        assert!(!pbc_nullable.is_constant());
    }

    #[test]
    fn somepbc_constructor_deduplicates_description_set() {
        // Upstream set membership is by identity (`desc is desc`).
        // Cloning the `Rc` preserves identity so the BTreeMap dedups.
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let desc = fake_function_entry(&bk, "f");
        let pbc = SomePBC::new(vec![desc.clone(), desc], false);
        assert_eq!(pbc.descriptions.len(), 1);
    }

    #[test]
    fn somenone_is_always_constant_and_nullable() {
        let s = SomeValue::None_(SomeNone::new());
        assert!(s.is_constant());
        assert!(s.can_be_none());
        assert!(s.immutable());
        assert_eq!(s.knowntype(), KnownType::NoneType);
        let SomeValue::None_(none) = s else {
            panic!("expected SomeNone");
        };
        assert_eq!(
            none.base.const_box.as_ref().map(|c| &c.value),
            Some(&super::super::super::flowspace::model::ConstValue::None)
        );
    }

    #[test]
    fn somebuiltin_is_immutable_and_not_nullable() {
        let s = SomeValue::Builtin(SomeBuiltin::new("getattr", None, None));
        assert!(s.immutable());
        assert!(!s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::BuiltinFunctionOrMethod);
    }

    #[test]
    fn somebuiltinmethod_is_immutable_and_not_nullable() {
        let s = SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
            "getitem",
            SomeValue::List(SomeList::new(ld(
                SomeValue::Integer(SomeInteger::default()),
            ))),
            "__getitem__",
        ));
        assert!(s.immutable());
        assert!(!s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::BuiltinFunctionOrMethod);
    }

    #[test]
    fn somelladtmeth_is_immutable_and_not_nullable() {
        let s = SomeValue::LLADTMeth(SomeLLADTMeth::new(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::Ptr(
                crate::translator::rtyper::lltypesystem::lltype::Ptr {
                    TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(
                        crate::translator::rtyper::lltypesystem::lltype::FuncType {
                            args: vec![],
                            result:
                                crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
                        },
                    ),
                },
            ),
            ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                "f",
                Constant::new(ConstValue::Dict(Default::default())),
            ))),
        ));
        assert!(s.immutable());
        assert!(!s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::Object);
    }

    #[test]
    fn someptr_call_returns_annotation_of_low_level_result() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Bool,
            }),
        });
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::new(false, false)))],
            None,
            None,
        );

        let result = s_ptr
            .call(&args)
            .expect("low-level function pointer call must succeed");
        assert!(matches!(result, SomeValue::Bool(_)));
    }

    #[test]
    fn someptr_call_rejects_keyword_arguments() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![],
            Some(std::collections::HashMap::from([(
                "x".into(),
                Some(SomeValue::Integer(SomeInteger::new(false, false))),
            )])),
            None,
        );

        let err = s_ptr
            .call(&args)
            .expect_err("low-level function pointer call must reject kwargs");
        assert!(err.msg.unwrap_or_default().contains("keyword arguments"));
    }

    #[test]
    fn somevalue_call_routes_through_someptr_branch() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s = SomeValue::Ptr(SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Bool,
            }),
        }));
        let args = super::super::argument::ArgumentsForTranslation::new(vec![], None, None);

        let result = s
            .call(&args)
            .expect("SomeValue::call must route SomePtr through SomePtr.call");
        assert!(matches!(result, Some(SomeValue::Bool(_))));
    }

    #[test]
    fn someptr_getattr_rejects_nonconstant_field_name() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let s_attr = SomeString::new(false, false);
        let err = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect_err("SomePtr.getattr must reject non-constant field names");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("non-constant field-name")
        );
    }

    #[test]
    fn someptr_setattr_rejects_nonconstant_field_name() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let s_attr = SomeString::new(false, false);
        let err = s_ptr
            .setattr(
                &SomeValue::String(s_attr),
                &SomeValue::Integer(SomeInteger::new(false, false)),
            )
            .expect_err("SomePtr.setattr must reject non-constant field names");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("non-constant field-name")
        );
    }

    #[test]
    fn someptr_getattr_surfaces_missing_field_error_on_function_pointer_subset() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("missing")));
        let err = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect_err("function-pointer subset has no low-level fields");
        assert!(err.msg.unwrap_or_default().contains("has no field"));
    }

    #[test]
    fn someptr_setattr_surfaces_missing_field_error_on_function_pointer_subset() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("missing")));
        let err = s_ptr
            .setattr(
                &SomeValue::String(s_attr),
                &SomeValue::Integer(SomeInteger::new(false, false)),
            )
            .expect_err("function-pointer subset has no low-level fields");
        assert!(err.msg.unwrap_or_default().contains("has no field"));
    }

    #[test]
    fn someptr_getattr_reads_struct_field_annotation() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelType, Ptr, PtrTarget, Struct,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("x")));

        let result = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect("SomePtr.getattr must read struct field annotations");
        assert!(matches!(result, SomeValue::Integer(_)));
    }

    #[test]
    fn someptr_getattr_as_ref_is_identity_bound_method() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget, Struct};

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Struct(Struct::new("MappedObj", vec![])),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("as_ref")));
        let result = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect("SomePtr.getattr as_ref");
        match result {
            SomeValue::BuiltinMethod(m) => {
                assert_eq!(m.analyser_name, "ptr_method_as_ref");
                assert!(matches!(m.s_self.as_ref(), SomeValue::Ptr(_)));
            }
            other => panic!("as_ref must bind ptr_method_as_ref, got {other:?}"),
        }
    }

    #[test]
    fn someptr_setattr_checks_struct_field_annotation() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelType, Ptr, PtrTarget, Struct,
        };

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("x")));

        let result = s_ptr
            .setattr(
                &SomeValue::String(s_attr),
                &SomeValue::Integer(SomeInteger::new(false, false)),
            )
            .expect("SomePtr.setattr must accept matching struct field annotations");
        assert!(matches!(result, SomeValue::Impossible));
    }

    #[test]
    fn someptr_getattr_wraps_struct_adtmethod_as_lladtmeth() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget, Struct};

        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Struct(Struct::with_adtmeths(
                "S",
                vec![],
                vec![(
                    "f".into(),
                    ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                        "f",
                        Constant::new(ConstValue::Dict(Default::default())),
                    ))),
                )],
            )),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("f")));

        let result = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect("SomePtr.getattr must bind struct adt methods");
        assert!(matches!(result, SomeValue::LLADTMeth(_)));
    }

    #[test]
    fn someptr_getattr_exposes_nested_struct_field_as_someptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelType, Ptr, PtrTarget, Struct,
        };

        let inner = Struct::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::Struct(Struct::new(
                "Outer",
                vec![("x".into(), LowLevelType::Struct(Box::new(inner)))],
            )),
        });
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("x")));

        let result = s_ptr
            .getattr(&SomeValue::String(s_attr))
            .expect("SomePtr.getattr must expose nested struct fields as SomePtr");
        assert!(matches!(result, SomeValue::Ptr(_)));
    }

    #[test]
    fn someptr_len_returns_constant_for_fixedsize_array_pointer() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FixedSizeArray, LowLevelType, Ptr, PtrTarget,
        };

        let ann = super::super::annrpython::RPythonAnnotator::new(None, None, None, false);
        let _guard = ann.bookkeeper.at_position(None);
        let s_ptr = SomePtr::new(Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 3)),
        });
        let result = s_ptr
            .len()
            .expect("SomePtr.len must read fixed-size-array length");
        let SomeValue::Integer(i) = result else {
            panic!("expected SomeInteger");
        };
        assert_eq!(i.base.const_box.unwrap().value, ConstValue::Int(3));
    }

    #[test]
    fn find_method_wraps_list_append_as_somebuiltinmethod() {
        let s_list = SomeValue::List(SomeList::new(ld(
            SomeValue::Integer(SomeInteger::default()),
        )));
        let method = s_list
            .find_method("append")
            .expect("list.append must be recognized by find_method");
        let SomeValue::BuiltinMethod(method) = method else {
            panic!("expected SomeBuiltinMethod");
        };
        assert_eq!(method.analyser_name, "list_method_append");
        assert_eq!(method.methodname, "append");
    }

    #[test]
    fn somevalue_call_routes_through_builtinmethod_append() {
        let ann = super::super::annrpython::RPythonAnnotator::new(None, None, None, false);
        let _guard = ann.bookkeeper.at_position(None);
        let list = SomeList::new(ListDef::new(
            Some(ann.bookkeeper.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        ));
        let s_list = SomeValue::List(list.clone());
        let method = s_list
            .find_method("append")
            .expect("list.append must be recognized by find_method");
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::new(false, false)))],
            None,
            None,
        );

        let result = method.call(&args).expect("append call must succeed");

        assert!(result.is_none(), "method_append is a void analyser → None");
        let SomeValue::Integer(item) = list.listdef.s_value() else {
            panic!("expected SomeInteger list item");
        };
        assert!(!item.nonneg, "append should widen the list item annotation");
    }

    #[test]
    fn union_builtin_methods_require_same_identity_and_union_self() {
        let a = SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
            "getitem",
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::new(true, false),
            )))),
            "__getitem__",
        ));
        let b = SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
            "getitem",
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::new(false, false),
            )))),
            "__getitem__",
        ));
        let merged = union(&a, &b).unwrap();
        let SomeValue::BuiltinMethod(method) = merged else {
            panic!("expected SomeBuiltinMethod");
        };
        let SomeValue::List(list) = method.s_self.as_ref().clone() else {
            panic!("expected SomeList receiver");
        };
        let SomeValue::Integer(elem) = list.listdef.s_value() else {
            panic!("expected SomeInteger element");
        };
        assert!(!elem.nonneg);

        let different_name = SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
            "getitem",
            SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
                SomeInteger::default(),
            )))),
            "__setitem__",
        ));
        assert!(union(&a, &different_name).is_err());
    }

    #[test]
    fn someimpossiblevalue_mirrors_upstream_defaults() {
        let s = SomeImpossibleValue::new();
        assert!(s.immutable());
        assert!(!s.is_constant());
        assert!(!s.can_be_none());
    }

    #[test]
    fn someweakref_allows_none_target() {
        let s = SomeValue::WeakRef(SomeWeakRef::new(None));
        assert!(s.immutable());
        // Weakrefs can always be None → upstream default applies.
        assert!(s.can_be_none());
        assert_eq!(s.knowntype(), KnownType::WeakrefReference);
    }

    #[test]
    fn sometypeof_keeps_variable_refs() {
        let v1 = Rc::new(Variable::named("v1"));
        let v2 = Rc::new(Variable::named("v2"));
        let t = SomeTypeOf::new(vec![v1, v2]);
        assert_eq!(t.is_type_of.len(), 2);
        assert_eq!(t.knowntype(), KnownType::Type);
    }

    #[test]
    fn singleton_helpers_match_upstream_definitions() {
        assert_eq!(s_int().knowntype(), KnownType::Int);
        assert!(matches!(s_impossible_value(), SomeValue::Impossible));
        assert!(matches!(s_none(), SomeValue::None_(_)));
        // s_Str0 carries no_nul = True.
        let SomeValue::String(s) = s_str0() else {
            panic!("s_str0 must be SomeString");
        };
        assert!(s.inner.no_nul);
    }

    #[test]
    fn union_identity_returns_self() {
        let a = SomeValue::Integer(SomeInteger::new(true, false));
        assert_eq!(union(&a, &a).unwrap(), a);
    }

    #[test]
    fn union_impossible_is_identity() {
        let a = SomeValue::Integer(SomeInteger::default());
        assert_eq!(union(&SomeValue::Impossible, &a).unwrap(), a);
        assert_eq!(union(&a, &SomeValue::Impossible).unwrap(), a);
    }

    #[test]
    fn union_integer_merges_nonneg_and_unsigned() {
        let nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let signed = SomeValue::Integer(SomeInteger::new(false, false));
        // AND of nonneg flags → not nonneg.
        let u = union(&nonneg, &signed).unwrap();
        if let SomeValue::Integer(s) = u {
            assert!(!s.nonneg);
            assert!(!s.unsigned);
        } else {
            panic!("expected SomeInteger");
        }
    }

    #[test]
    fn union_int_and_float_widens_to_float() {
        let i = SomeValue::Integer(SomeInteger::default());
        let f = SomeValue::Float(SomeFloat::new());
        assert!(matches!(union(&i, &f).unwrap(), SomeValue::Float(_)));
        assert!(matches!(union(&f, &i).unwrap(), SomeValue::Float(_)));
    }

    #[test]
    fn union_bool_and_int_widens_to_int() {
        let b = SomeValue::Bool(SomeBool::new());
        let i = SomeValue::Integer(SomeInteger::new(true, false));
        assert!(matches!(union(&b, &i).unwrap(), SomeValue::Integer(_)));
        assert!(matches!(union(&i, &b).unwrap(), SomeValue::Integer(_)));
    }

    #[test]
    fn union_char_and_string_widens_to_string() {
        let c = SomeValue::Char(SomeChar::new(true));
        let s = SomeValue::String(SomeString::new(false, true));
        let u = union(&c, &s).unwrap();
        assert!(matches!(u, SomeValue::String(_)));
    }

    #[test]
    fn union_tuple_length_mismatch_errors() {
        let a = SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
            SomeInteger::default(),
        )]));
        let b = SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
        ]));
        assert!(union(&a, &b).is_err());
    }

    #[test]
    fn union_list_element_types_merge() {
        // ListDef::mutable = bookkeeper-present path so merge is
        // allowed (matches upstream `ListDef('dummy', ...)`).
        let a = SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
            SomeInteger::new(true, false),
        ))));
        let b = SomeValue::List(SomeList::new(ld_mutable(SomeValue::Integer(
            SomeInteger::new(false, false),
        ))));
        let u = union(&a, &b).unwrap();
        let SomeValue::List(s) = u else {
            panic!("expected SomeList");
        };
        let SomeValue::Integer(elem) = s.listdef.s_value() else {
            panic!("expected integer element");
        };
        assert!(!elem.nonneg);
    }

    #[test]
    fn union_none_and_instance_makes_instance_nullable() {
        let cdef = Some(ClassDef::new_standalone("pkg.X", None));
        let inst = SomeValue::Instance(SomeInstance::new(
            cdef.clone(),
            false,
            std::collections::BTreeMap::new(),
        ));
        let none = SomeValue::None_(SomeNone::new());
        let u = union(&none, &inst).unwrap();
        if let SomeValue::Instance(s) = u {
            assert!(s.can_be_none);
        } else {
            panic!("expected SomeInstance");
        }
    }

    #[test]
    fn union_same_classdef_instances_keeps_only_matching_flags() {
        let mut flags_a = std::collections::BTreeMap::new();
        flags_a.insert("immutable".to_string(), true);
        let mut flags_b = std::collections::BTreeMap::new();
        flags_b.insert("immutable".to_string(), true);
        flags_b.insert("pinned".to_string(), true);

        let cdef = Some(ClassDef::new_standalone("pkg.X", None));
        let a = SomeValue::Instance(SomeInstance::new(cdef.clone(), false, flags_a));
        let b = SomeValue::Instance(SomeInstance::new(cdef.clone(), true, flags_b));
        let u = union(&a, &b).unwrap();
        let SomeValue::Instance(inst) = u else {
            panic!("expected SomeInstance");
        };
        assert!(classdef_opt_eq(&inst.classdef, &cdef));
        assert!(inst.can_be_none);
        assert_eq!(inst.flags.len(), 1);
        assert!(inst.flags.contains_key("immutable"));
        assert!(!inst.flags.contains_key("pinned"));
    }

    #[test]
    fn union_instance_uses_common_base_classdef() {
        let base = ClassDef::new_standalone("pkg.Base", None);
        let sub = ClassDef::new_standalone("pkg.Sub", Some(&base));
        let a = SomeValue::Instance(SomeInstance::new(
            Some(sub),
            false,
            std::collections::BTreeMap::new(),
        ));
        let b = SomeValue::Instance(SomeInstance::new(
            Some(base.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let u = union(&a, &b).unwrap();
        let SomeValue::Instance(inst) = u else {
            panic!("expected SomeInstance");
        };
        assert!(classdef_opt_eq(&inst.classdef, &Some(base)));
    }

    /// `SomeInstance.noneify` / `nonnoneify` reconstruct with the
    /// `flags={}` default, so a flag does not cross either of them.
    /// `intersection_Instance` already does the same. The only flag that
    /// reaches here in production is `access_directly`, whose consumer
    /// (`policy.py:71-83`) aborts translation when it is set on a graph the
    /// codewriter declines — so an over-kept flag is a build failure.
    #[test]
    fn noneify_and_nonnoneify_drop_instance_flags() {
        let mut flags = std::collections::BTreeMap::new();
        flags.insert("access_directly".to_string(), true);
        let inst = SomeValue::Instance(SomeInstance::new(None, false, flags));

        let noneified = inst.noneify().expect("instance noneifies");
        let SomeValue::Instance(n) = &noneified else {
            panic!("noneify kept the instance shape, got {noneified:?}")
        };
        assert!(n.can_be_none);
        assert!(n.flags.is_empty(), "noneify kept {:?}", n.flags);

        let SomeValue::Instance(nn) = inst.nonnoneify() else {
            panic!("nonnoneify kept the instance shape")
        };
        assert!(!nn.can_be_none);
        assert!(nn.flags.is_empty(), "nonnoneify kept {:?}", nn.flags);
    }

    #[test]
    fn intersection_and_difference_wrappers_match_pypy_registrations() {
        let base = ClassDef::new_standalone("pkg.Base", None);
        let sub = ClassDef::new_standalone("pkg.Sub", Some(&base));
        let base_inst =
            SomeInstance::new(Some(base.clone()), true, std::collections::BTreeMap::new());
        let sub_inst =
            SomeInstance::new(Some(sub.clone()), false, std::collections::BTreeMap::new());

        let inter = intersection_Instance(&base_inst, &sub_inst);
        let SomeValue::Instance(inter_inst) = inter else {
            panic!("expected SomeInstance");
        };
        assert!(classdef_opt_eq(&inter_inst.classdef, &Some(sub.clone())));
        assert!(!inter_inst.can_be_none);

        assert!(matches!(
            difference_Instance_Instance(&sub_inst, &base_inst),
            SomeValue::Impossible
        ));

        let exc = SomeException::new(vec![base.clone(), sub.clone()]);
        let inter_exc = intersection_Exception_Instance(&exc, &base_inst);
        let SomeValue::Exception(inter_exc) = inter_exc else {
            panic!("expected SomeException");
        };
        assert_eq!(inter_exc.classdefs.len(), 2);

        assert!(matches!(
            difference_Exception_Instance(&exc, &base_inst),
            SomeValue::Impossible
        ));
    }

    #[test]
    fn union_object_and_classed_instance_widens_to_object() {
        let a = SomeValue::Instance(SomeInstance::new(
            None,
            false,
            std::collections::BTreeMap::new(),
        ));
        let b = SomeValue::Instance(SomeInstance::new(
            Some(ClassDef::new_standalone("pkg.X", None)),
            false,
            std::collections::BTreeMap::new(),
        ));
        let u = union(&a, &b).unwrap();
        let SomeValue::Instance(inst) = u else {
            panic!("expected SomeInstance");
        };
        assert!(inst.classdef.is_none());
    }

    #[test]
    fn union_instances_with_no_common_base_raises() {
        // Two standalone classes carry classdefs and share no base.
        // binaryop.py union raises UnionError when both sides have a
        // classdef and `commonbase` returns None — it does NOT widen
        // unrelated classdefs to the classdef-less top.
        let a = SomeValue::Instance(SomeInstance::new(
            Some(ClassDef::new_standalone("pkg.A", None)),
            false,
            std::collections::BTreeMap::new(),
        ));
        let b = SomeValue::Instance(SomeInstance::new(
            Some(ClassDef::new_standalone("pkg.B", None)),
            false,
            std::collections::BTreeMap::new(),
        ));
        assert!(union(&a, &b).is_err());
    }

    #[test]
    fn union_exception_unions_classdef_sets() {
        let ve = ClassDef::new_standalone("ValueError", None);
        let te = ClassDef::new_standalone("TypeError", None);
        let ke = ClassDef::new_standalone("KeyError", None);
        let a = SomeValue::Exception(SomeException::new(vec![ve.clone(), te.clone()]));
        let b = SomeValue::Exception(SomeException::new(vec![te.clone(), ke.clone()]));
        let u = union(&a, &b).unwrap();
        let SomeValue::Exception(exc) = u else {
            panic!("expected SomeException");
        };
        assert_eq!(exc.classdefs.len(), 3);
    }

    #[test]
    fn union_exception_and_none_routes_through_someinstance() {
        let ve = ClassDef::new_standalone("ValueError", None);
        let exc = SomeValue::Exception(SomeException::new(vec![ve.clone()]));
        let u = union(&exc, &s_none()).unwrap();
        let SomeValue::Instance(inst) = u else {
            panic!("expected SomeInstance");
        };
        assert!(classdef_opt_eq(&inst.classdef, &Some(ve)));
        assert!(inst.can_be_none);
    }

    #[test]
    fn union_pbc_merges_descriptions() {
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let a = SomeValue::PBC(SomePBC::new(vec![fake_function_entry(&bk, "f1")], false));
        let b = SomeValue::PBC(SomePBC::new(vec![fake_function_entry(&bk, "f2")], true));
        let u = union(&a, &b).unwrap();
        let SomeValue::PBC(pbc) = u else {
            panic!("expected SomePBC");
        };
        assert_eq!(pbc.descriptions.len(), 2);
        assert!(pbc.can_be_none);
    }

    #[test]
    fn somepbc_knowntype_uses_commonbase_fold() {
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let function_pbc = SomePBC::new(vec![fake_function_entry(&bk, "f")], false);
        assert_eq!(function_pbc.knowntype(), KnownType::FunctionType);

        let class_pbc = SomePBC::new(vec![fake_class_entry(&bk, "C")], false);
        assert_eq!(class_pbc.knowntype(), KnownType::Type);
    }

    #[test]
    fn union_pbc_mixed_kinds_errors() {
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let a = SomeValue::PBC(SomePBC::new(vec![fake_function_entry(&bk, "f")], false));
        let b = SomeValue::PBC(SomePBC::new(vec![fake_class_entry(&bk, "C")], false));
        assert!(union(&a, &b).is_err());
    }

    #[test]
    fn union_pbc_function_memo_mixed_errors() {
        // MemoDesc is a distinct DescKind::Memo, so a FunctionDesc PBC
        // unioned with a MemoDesc PBC is a mixed-kind set and must be
        // rejected — matching upstream where FunctionDesc + MemoDesc are
        // different classes (model.py getKind).
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let a = SomeValue::PBC(SomePBC::new(vec![fake_function_entry(&bk, "f")], false));
        let b = SomeValue::PBC(SomePBC::new(vec![fake_memo_entry(&bk, "g")], false));
        assert!(union(&a, &b).is_err());
    }

    #[test]
    fn union_none_and_pbc_makes_pbc_nullable() {
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let pbc = SomeValue::PBC(SomePBC::new(vec![fake_function_entry(&bk, "f")], false));
        let none = SomeValue::None_(SomeNone::new());
        let u = union(&none, &pbc).unwrap();
        let SomeValue::PBC(merged) = u else {
            panic!("expected SomePBC");
        };
        assert!(merged.can_be_none);
    }

    #[test]
    fn union_none_and_pbc_preserves_subset_of() {
        // Share identity via Rc::clone so parent and child reference
        // the same FunctionDesc — matches upstream pyobj → getdesc
        // identity round-trip.
        let bk = Rc::new(super::super::bookkeeper::Bookkeeper::new());
        let f = fake_function_entry(&bk, "f");
        let parent = SomePBC::new(vec![f.clone()], false);
        let child = SomeValue::PBC(SomePBC::with_subset(
            vec![f],
            false,
            Some(Box::new(parent.clone())),
        ));
        let merged = union(&s_none(), &child).unwrap();
        let SomeValue::PBC(pbc) = merged else {
            panic!("expected SomePBC");
        };
        assert!(pbc.can_be_none);
        assert_eq!(pbc.subset_of, Some(Box::new(parent)));
    }

    #[test]
    fn union_weakref_pairs_follow_instance_shape() {
        let x = ClassDef::new_standalone("pkg.X", None);
        let a = SomeValue::WeakRef(SomeWeakRef::new(Some(x.clone())));
        let b = SomeValue::WeakRef(SomeWeakRef::new(Some(x.clone())));
        assert!(union(&a, &b).is_ok());

        let c = SomeValue::WeakRef(SomeWeakRef::new(Some(ClassDef::new_standalone(
            "pkg.Y", None,
        ))));
        assert!(union(&a, &c).is_err());

        let dead = SomeValue::WeakRef(SomeWeakRef::new(None));
        let u = union(&a, &dead).unwrap();
        let SomeValue::WeakRef(w) = u else {
            panic!("expected SomeWeakRef");
        };
        assert!(w.classdef.is_none());
    }

    #[test]
    fn union_integer_signedness_mismatch_with_nonneg_proves_signed() {
        // binaryop.py:189-193 — signed vs unsigned where the signed
        // side is proven ≥ 0 → unify as the unsigned type.
        let signed_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let unsigned = SomeValue::Integer(SomeInteger::new(false, true));
        let r = union(&signed_nonneg, &unsigned).expect("should unify");
        match r {
            SomeValue::Integer(i) => {
                assert!(i.unsigned, "unsigned side wins when signed is nonneg");
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn union_integer_signedness_mismatch_without_nonneg_errors() {
        // binaryop.py:190-192 — signed int with no nonneg proof unioned
        // against r_uint must raise UnionError.
        let signed = SomeValue::Integer(SomeInteger::new(false, false));
        let unsigned = SomeValue::Integer(SomeInteger::new(false, true));
        assert!(
            union(&signed, &unsigned).is_err(),
            "unprovable signedness should error"
        );
    }

    #[test]
    fn union_typed_address_access_requires_matching_type() {
        use crate::translator::rtyper::lltypesystem::llmemory::SomeTypedAddressAccess;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        // llannotation.py __extend__ — same access type unions to s_taa1.
        let signed =
            SomeValue::TypedAddressAccess(SomeTypedAddressAccess::new(LowLevelType::Signed));
        let signed2 =
            SomeValue::TypedAddressAccess(SomeTypedAddressAccess::new(LowLevelType::Signed));
        assert!(union(&signed, &signed2).is_ok());
        // Distinct access types fail (upstream `assert s_taa1.type == s_taa2.type`).
        let charr = SomeValue::TypedAddressAccess(SomeTypedAddressAccess::new(LowLevelType::Char));
        assert!(union(&signed, &charr).is_err());
    }

    #[test]
    fn unionof_folds_over_inputs() {
        let ints = vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(true, false)),
        ];
        let u = unionof(&ints).unwrap();
        if let SomeValue::Integer(s) = u {
            assert!(!s.nonneg);
        } else {
            panic!("expected SomeInteger");
        }
    }

    #[test]
    fn unionof_empty_is_impossible() {
        let empty: [&SomeValue; 0] = [];
        let u = unionof(empty).unwrap();
        assert!(matches!(u, SomeValue::Impossible));
    }

    #[test]
    fn contains_delegates_to_union() {
        // test_model.py: `SomeInteger(unsigned=True).contains(SomeInteger(nonneg=True))`.
        // unsigned implies nonneg; nonneg ⊂ unsigned.
        let unsigned = SomeValue::Integer(SomeInteger::new(false, true));
        let nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        assert!(unsigned.contains(&nonneg));
        // But the reverse fails: a signed-but-nonneg set does not
        // contain r_uint values.
        assert!(!nonneg.contains(&unsigned));
    }

    #[test]
    fn add_knowntypedata_populates_table() {
        let mut ktd: KnownTypeData = std::collections::HashMap::new();
        let x = Rc::new(Variable::named("x"));
        let y = Rc::new(Variable::named("y"));
        add_knowntypedata(
            &mut ktd,
            ExitCaseKey::Bool(true),
            &[Rc::clone(&x), Rc::clone(&y)],
            SomeValue::Integer(SomeInteger::default()),
        );
        assert_eq!(ktd[&ExitCaseKey::Bool(true)].len(), 2);
        assert!(ktd[&ExitCaseKey::Bool(true)].contains_key(&x));
        assert!(!ktd.contains_key(&ExitCaseKey::Bool(false)));
    }

    #[test]
    fn merge_knowntypedata_intersects_and_unions() {
        let x = Rc::new(Variable::named("x"));
        let y = Rc::new(Variable::named("y"));
        let mut k1: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut k1,
            ExitCaseKey::Bool(true),
            &[Rc::clone(&x), Rc::clone(&y)],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut k2: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut k2,
            ExitCaseKey::Bool(true),
            &[Rc::clone(&x)],
            SomeValue::Integer(SomeInteger::default()),
        );
        let merged = merge_knowntypedata(&k1, &k2);
        // Only x survives (present in both); y dropped.
        assert_eq!(merged[&ExitCaseKey::Bool(true)].len(), 1);
        assert!(merged[&ExitCaseKey::Bool(true)].contains_key(&x));
        assert!(!merged[&ExitCaseKey::Bool(true)].contains_key(&y));
    }

    #[test]
    fn bool_union_merges_knowntypedata_and_agreeing_const() {
        // binaryop.py union. A refinement-carrying non-constant bool
        // must contain a constant bool carrying the same refinement —
        // union() has to reproduce the merged knowntypedata, not drop
        // it, or setbinding's contains check fails on reflow.
        let x = Rc::new(Variable::named("x"));
        let mut ktd: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut ktd,
            ExitCaseKey::Bool(true),
            &[Rc::clone(&x)],
            SomeValue::Integer(SomeInteger::new(true, false)),
        );
        let mut general = SomeBool::new();
        general.set_knowntypedata(ktd.clone());
        let mut const_false = SomeBool::new();
        const_false.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        const_false.set_knowntypedata(ktd);
        let general = SomeValue::Bool(general);
        let const_false = SomeValue::Bool(const_false);

        let u = union(&general, &const_false).unwrap();
        assert_eq!(u, general, "disagreeing const drops, ktd survives");
        assert!(general.contains(&const_false));

        // Agreeing constants survive the union.
        let u2 = union(&const_false, &const_false.clone()).unwrap();
        match u2 {
            SomeValue::Bool(b) => {
                assert!(matches!(
                    b.base.const_box.as_ref().map(|c| &c.value),
                    Some(ConstValue::Bool(false))
                ));
                assert!(b.knowntypedata.is_some());
            }
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn somebool_set_knowntypedata_rejects_empty_inner() {
        let mut ktd: KnownTypeData = std::collections::HashMap::new();
        ktd.insert(ExitCaseKey::Bool(true), std::collections::HashMap::new()); // empty inner
        let mut b = SomeBool::new();
        b.set_knowntypedata(ktd);
        assert!(
            b.knowntypedata.is_none(),
            "empty inner dict should be pruned"
        );
    }

    #[test]
    fn someinteger_carries_knowntypedata_keyed_by_int_case() {
        // An integer discriminant SwitchInt attaches per-case
        // constraints keyed by ExitCaseKey::Int(k), the same set-once /
        // empty-pruning contract as SomeBool.
        let x = Rc::new(Variable::named("x"));
        let mut ktd: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut ktd,
            ExitCaseKey::Int(2),
            &[Rc::clone(&x)],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut i = SomeInteger::default();
        i.set_knowntypedata(ktd);
        let data = i.knowntypedata.as_ref().expect("ktd must be populated");
        assert!(data.contains_key(&ExitCaseKey::Int(2)));
        assert!(!data.contains_key(&ExitCaseKey::Int(0)));
    }

    #[test]
    fn knowntypedata_accessor_reads_generically() {
        // `getattr(annotation, "knowntypedata", {})` (annrpython.py)
        // reads off SomeBool and SomeInteger; any other annotation
        // returns the `{}` default (None).
        let x = Rc::new(Variable::named("x"));

        let mut b_ktd: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut b_ktd,
            ExitCaseKey::Bool(true),
            &[Rc::clone(&x)],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut b = SomeBool::new();
        b.set_knowntypedata(b_ktd);
        assert!(SomeValue::Bool(b).knowntypedata().is_some());

        let mut i_ktd: KnownTypeData = std::collections::HashMap::new();
        add_knowntypedata(
            &mut i_ktd,
            ExitCaseKey::Int(1),
            &[Rc::clone(&x)],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut i = SomeInteger::default();
        i.set_knowntypedata(i_ktd);
        assert!(SomeValue::Integer(i).knowntypedata().is_some());

        // A plain integer and a non-carrier annotation read as None.
        assert!(
            SomeValue::Integer(SomeInteger::default())
                .knowntypedata()
                .is_none()
        );
        assert!(SomeValue::Type(SomeType::new()).knowntypedata().is_none());
    }

    /// rstring.py — `SomeStringBuilder` getlength/build annotate to
    /// `SomeInteger(nonneg=True)` and `SomeString(can_be_None=False)`.
    #[test]
    fn some_string_builder_method_annotations() {
        let b = SomeStringBuilder::new();
        match b.method_getlength() {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("getlength must be SomeInteger(nonneg), got {other:?}"),
        }
        match b.method_build() {
            SomeValue::String(s) => assert!(!s.can_be_none()),
            other => panic!("build must be SomeString(can_be_None=False), got {other:?}"),
        }
        assert!(matches!(
            b.method_append(&SomeValue::String(SomeString::new(false, false))),
            SomeValue::None_(_)
        ));
        // `AbstractStringBuilderRepr.rtype_method_append` accepts
        // `SomeUnicodeCodePoint` as the char arm (`ll_append_char`).
        assert!(matches!(
            b.method_append(&SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(
                false
            ))),
            SomeValue::None_(_)
        ));
        assert!(matches!(b.method_getlength(), SomeValue::Integer(_)));
    }

    /// rstring.py — `SomeUnicodeBuilder.build` annotates to
    /// `SomeUnicodeString(can_be_None=False)`; getlength stays the nonneg
    /// integer.
    #[test]
    fn some_unicode_builder_method_annotations() {
        let b = SomeUnicodeBuilder::new();
        match b.method_getlength() {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("getlength must be SomeInteger(nonneg), got {other:?}"),
        }
        match b.method_build() {
            SomeValue::UnicodeString(s) => assert!(!s.can_be_none()),
            other => panic!("build must be SomeUnicodeString(can_be_None=False), got {other:?}"),
        }
        assert!(matches!(
            b.method_append(&SomeValue::UnicodeString(SomeUnicodeString::new(
                false, false
            ))),
            SomeValue::None_(_)
        ));
    }
}
