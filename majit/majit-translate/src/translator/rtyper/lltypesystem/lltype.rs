//! RPython `rpython/rtyper/lltypesystem/lltype.py`.
#![allow(non_camel_case_types, non_snake_case)]
//!
//! Currently ports two surfaces:
//! * Function-pointer surface consumed by `translator/simplify.py:get_graph`:
//!   [`_ptr`], [`_func`], [`FuncType`], [`functionptr`], [`getfunctionptr`],
//!   [`_getconcretetype`].
//! * [`LowLevelType`] primitive enum consumed by `rmodel.py`'s
//!   `Repr.lowleveltype` attribute and by `inputconst(reqtype, value)`.
//!   The Rust adaptation collapses upstream's class hierarchy
//!   (`LowLevelType` → `Primitive` / `Number` / `Ptr` / `Struct` / `Array`
//!   at `lltype.py,...`) into an enum so `Repr`
//!   implementations can pattern-match on kind without Rust trait-object
//!   downcasts. The three variants currently populated (`Void`, `Bool`,
//!   `Signed`, `Float`, `Char`, `UniChar`, `Unsigned`, `SingleFloat`,
//!   `LongFloat`, `SignedLongLong`, `UnsignedLongLong`, `Ptr`) cover every
//!   type used by `rpbc.py FunctionRepr` / `rclass.py InstanceRepr` /
//!   `FunctionReprBase.call` — additional container kinds (`Struct`,
//!   `Array`, `ForwardReference`) are not yet populated.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use crate::annotator::model::KnownType;
use crate::flowspace::model::{ConcretetypePlaceholder, ConstValue, GraphKey, GraphRef, Hlvalue};
use crate::translator::rtyper::error::TyperError;

thread_local! {
    /// RPython `_ptr._become()` mutates the pointer object in place so
    /// every alias starts resolving to the real target. Rust `_ptr`
    /// values are plain cloned structs, so the narrow annlowlevel port
    /// records the post-`_become` target by pointer identity here and
    /// routes read-side operations through the redirect.
    static PTR_BECOME_TARGETS: RefCell<HashMap<u64, _ptr>> = RefCell::new(HashMap::new());

    /// `_subarray._cache` (lltype.py) — keyed by the parent
    /// container's identity and the `baseoffset_or_fieldname`, so two
    /// derivations of the same interior pointer reuse one `_subarray` (and so
    /// one container identity). Upstream uses a `WeakKeyDictionary` plus
    /// `_cleanup_cache`; the cache holds each `_subarray` strongly, so a plain
    /// strong map is the consistent adaptation.
    static SUBARRAY_CACHE: RefCell<HashMap<(usize, ParentIndex), _subarray>> =
        RefCell::new(HashMap::new());

    /// `_arraylenref._cache` (lltype.py, 2091-2098) — the array
    /// container's identity → its single length pseudo-reference.
    static ARRAYLENREF_CACHE: RefCell<HashMap<usize, _arraylenref>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelayedPointer;

/// RPython `class State(object)` (`lltype.py`) backing the module-level
/// recursion guard state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State;

/// RPython `class _uninitialized(object)` (`lltype.py`) sentinel for
/// uninitialized low-level memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct _uninitialized {
    pub TYPE: LowLevelType,
}

/// RPython `NFOUND = object()` (`lltype.py`) sentinel.
pub static NFOUND: LazyLock<&'static str> = LazyLock::new(|| "NFOUND");

/// RPython `Typedef(LowLevelType)` (`lltype.py`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Typedef {
    pub OF: LowLevelType,
    pub c_name: Option<String>,
}

impl Typedef {
    pub fn new(OF: LowLevelType, c_name: Option<String>) -> Self {
        Typedef { OF, c_name }
    }
}

/// RPython `STRUCT_BY_FLAVOR` (`lltype.py`).
///
/// Upstream uses a dict keyed by allocation flavor. Keep the same shape
/// instead of routing through a Rust enum table.
pub static STRUCT_BY_FLAVOR: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| HashMap::from([("raw", "Struct"), ("gc", "GcStruct")]));

/// RPython `FORWARDREF_BY_FLAVOR` (`lltype.py`).
///
/// Upstream uses a dict keyed by allocation flavor. Keep the same dict-like
/// surface for line-by-line ports that look up the constructor by flavor.
pub static FORWARDREF_BY_FLAVOR: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        HashMap::from([
            ("raw", "ForwardReference"),
            ("gc", "GcForwardReference"),
            ("prebuilt", "FuncForwardReference"),
        ])
    });

/// RPython `_numbertypes` (`lltype.py`): Python numeric carrier type
/// name to the corresponding low-level Number token.
#[allow(non_upper_case_globals)]
pub static _numbertypes: LazyLock<HashMap<&'static str, LowLevelType>> = LazyLock::new(|| {
    HashMap::from([
        ("int", LowLevelType::Signed),
        ("r_uint", LowLevelType::Unsigned),
        ("r_longlong", LowLevelType::SignedLongLong),
        ("r_longlonglong", LowLevelType::SignedLongLongLong),
        ("r_ulonglong", LowLevelType::UnsignedLongLong),
        ("r_ulonglonglong", LowLevelType::UnsignedLongLongLong),
    ])
});

/// RPython `_to_primitive` (`lltype.py`): low-level primitive token
/// name to the conversion routine name used by `cast_primitive`.
#[allow(non_upper_case_globals)]
pub static _to_primitive: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("Char", "chr"),
        ("UniChar", "unichr"),
        ("Float", "float"),
        ("Signed", "int"),
        ("Unsigned", "r_uint"),
        ("SignedLongLong", "r_longlong"),
        ("SignedLongLongLong", "r_longlonglong"),
        ("UnsignedLongLong", "r_ulonglong"),
        ("UnsignedLongLongLong", "r_ulonglonglong"),
    ])
});

/// RPython `UninitializedMemoryAccess` (`lltype.py`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UninitializedMemoryAccess;

/// RPython `_ptrEntry(ExtRegistryEntry)` (`lltype.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct _ptrEntry;

/// RPython `staticAdtMethod` (`lltype.py`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct staticAdtMethod<T> {
    pub obj: T,
}

impl<T> staticAdtMethod<T> {
    pub fn new(obj: T) -> Self {
        staticAdtMethod { obj }
    }

    pub fn get(&self) -> &T {
        &self.obj
    }
}

fn deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.lltype.{name} — lltype helper deferred"
    ))
}

pub fn constPtr() -> Result<(), TyperError> {
    Err(deferred("constPtr"))
}

pub fn ann_typeOf() -> Result<(), TyperError> {
    Err(deferred("ann_typeOf"))
}

pub fn cast_primitive() -> Result<(), TyperError> {
    Err(deferred("cast_primitive"))
}

pub fn ann_cast_primitive() -> Result<(), TyperError> {
    Err(deferred("ann_cast_primitive"))
}

pub fn _cast_whatever() -> Result<(), TyperError> {
    Err(deferred("_cast_whatever"))
}

/// RPython `lltype._castdepth(outside, inside)` — signed parent-chain
/// distance used by `jtransform.py Transformer._is_rclass_instance`.
pub fn _castdepth(outside: &Struct, inside: &Struct) -> i32 {
    castdepth(outside, inside)
}

pub fn ann_cast_pointer() -> Result<(), TyperError> {
    Err(deferred("ann_cast_pointer"))
}

pub fn ann_cast_opaque_ptr() -> Result<(), TyperError> {
    Err(deferred("ann_cast_opaque_ptr"))
}

pub fn length_of_simple_gcarray_from_opaque() -> Result<(), TyperError> {
    Err(deferred("length_of_simple_gcarray_from_opaque"))
}

pub fn ann_length_of_simple_gcarray_from_opaque() -> Result<(), TyperError> {
    Err(deferred("ann_length_of_simple_gcarray_from_opaque"))
}

pub fn ann_direct_fieldptr() -> Result<(), TyperError> {
    Err(deferred("ann_direct_fieldptr"))
}

pub fn ann_direct_arrayitems() -> Result<(), TyperError> {
    Err(deferred("ann_direct_arrayitems"))
}

pub fn ann_direct_ptradd() -> Result<(), TyperError> {
    Err(deferred("ann_direct_ptradd"))
}

pub fn _struct_variety() -> Result<(), TyperError> {
    Err(deferred("_struct_variety"))
}

pub fn _get_empty_instance_of_struct_variety() -> Result<(), TyperError> {
    Err(deferred("_get_empty_instance_of_struct_variety"))
}

pub fn ann_malloc() -> Result<(), TyperError> {
    Err(deferred("ann_malloc"))
}

pub fn ann_free() -> Result<(), TyperError> {
    Err(deferred("ann_free"))
}

pub fn render_immortal() -> Result<(), TyperError> {
    Err(deferred("render_immortal"))
}

pub fn ann_render_immortal() -> Result<(), TyperError> {
    Err(deferred("ann_render_immortal"))
}

pub fn _make_scoped_allocator() -> Result<(), TyperError> {
    Err(deferred("_make_scoped_allocator"))
}

pub fn scoped_alloc() -> Result<(), TyperError> {
    Err(deferred("scoped_alloc"))
}

pub fn ann_nullptr() -> Result<(), TyperError> {
    Err(deferred("ann_nullptr"))
}

pub fn ann_cast_ptr_to_int() -> Result<(), TyperError> {
    Err(deferred("ann_cast_ptr_to_int"))
}

pub fn ann_cast_int_to_ptr() -> Result<(), TyperError> {
    Err(deferred("ann_cast_int_to_ptr"))
}

pub fn ann_getRuntimeTypeInfo() -> Result<(), TyperError> {
    Err(deferred("ann_getRuntimeTypeInfo"))
}

pub fn ann_runtime_type_info() -> Result<(), TyperError> {
    Err(deferred("ann_runtime_type_info"))
}

pub fn ann_identityhash() -> Result<(), TyperError> {
    Err(deferred("ann_identityhash"))
}

pub fn typeMethod<F>(func: F) -> F {
    func
}

pub fn dissect_ll_instance() -> Result<(), TyperError> {
    Err(deferred("dissect_ll_instance"))
}

/// RPython `frozendict` (`lltype.py`): a dict whose hash is
/// computed from sorted items. This keeps `_flds` / `_adtmeths` /
/// `_hints` order-insensitive for equality and hashing while `_names`
/// carries the explicit field order, as in upstream `Struct.__init__`.
#[derive(Clone, Debug)]
pub struct frozendict<V> {
    items: Vec<(String, V)>,
}

impl<V> frozendict<V> {
    pub fn new(items: Vec<(String, V)>) -> Self {
        let mut seen: Vec<String> = Vec::with_capacity(items.len());
        for (key, _) in &items {
            if seen.iter().any(|existing| existing == key) {
                panic!("frozendict duplicate key {:?}", key);
            }
            seen.push(key.clone());
        }
        frozendict { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.items
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    pub fn first(&self) -> Option<(&String, &V)> {
        self.items.first().map(|(key, value)| (key, value))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.items.iter().map(|(key, value)| (key, value))
    }

    pub fn to_vec(&self) -> Vec<(String, V)>
    where
        V: Clone,
    {
        self.items.clone()
    }
}

impl<V> From<Vec<(String, V)>> for frozendict<V> {
    fn from(value: Vec<(String, V)>) -> Self {
        frozendict::new(value)
    }
}

impl<'a, V> IntoIterator for &'a frozendict<V> {
    type Item = (&'a String, &'a V);
    type IntoIter =
        std::iter::Map<std::slice::Iter<'a, (String, V)>, fn(&(String, V)) -> (&String, &V)>;

    fn into_iter(self) -> Self::IntoIter {
        fn as_refs<V>((key, value): &(String, V)) -> (&String, &V) {
            (key, value)
        }
        self.items.iter().map(as_refs::<V>)
    }
}

impl<V: PartialEq> PartialEq for frozendict<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.items.iter().all(|(key, value)| {
                other
                    .get(key)
                    .is_some_and(|other_value| other_value == value)
            })
    }
}

impl<V: Eq> Eq for frozendict<V> {}

impl<V: Hash> Hash for frozendict<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut items: Vec<(&String, &V)> = self.iter().collect();
        items.sort_by_key(|(left_key, _)| *left_key);
        items.len().hash(state);
        for (key, value) in items {
            key.hash(state);
            value.hash(state);
        }
    }
}

/// RPython `lltype.Void` singleton surface. Pure re-export of the enum
/// variant so `rmodel.rs` / `rpbc.rs` / `rclass.rs` ports can mirror
/// upstream `from rpython.rtyper.lltypesystem.lltype import Void` reads.
pub const VOID: LowLevelType = LowLevelType::Void;
/// RPython `lltype.Bool` singleton surface.
pub const BOOL: LowLevelType = LowLevelType::Bool;
/// RPython `lltype.Signed` singleton surface.
pub const SIGNED: LowLevelType = LowLevelType::Signed;
/// RPython `lltype.Float` singleton surface.
pub const FLOAT: LowLevelType = LowLevelType::Float;
/// RPython `lltype.Char` singleton surface.
pub const CHAR: LowLevelType = LowLevelType::Char;
/// RPython `lltype.UniChar` singleton surface.
pub const UNICHAR: LowLevelType = LowLevelType::UniChar;

impl LowLevelType {
    /// RPython `isinstance(self, Primitive)` — true for the scalar
    /// types Number / Bool / Float / Char / UniChar / Void. `Ptr` is
    /// NOT a Primitive in upstream's class hierarchy.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            LowLevelType::Void
                | LowLevelType::Signed
                | LowLevelType::Unsigned
                | LowLevelType::SignedLongLong
                | LowLevelType::SignedLongLongLong
                | LowLevelType::UnsignedLongLong
                | LowLevelType::UnsignedLongLongLong
                | LowLevelType::Bool
                | LowLevelType::Float
                | LowLevelType::SingleFloat
                | LowLevelType::LongFloat
                | LowLevelType::Char
                | LowLevelType::UniChar
                | LowLevelType::Address
        )
    }

    /// RPython `LowLevelType._contains_value(value)` — used by
    /// `Repr.convert_const` (`rmodel.py`) and by `inputconst`
    /// (`rmodel.py`) as the "does this low-level type admit this
    /// Python value as a prebuilt constant" check.
    ///
    /// Upstream dispatches through each subclass's `_enforce` /
    /// `_contains_value` implementation; the Rust port pattern-matches
    /// on variant + [`ConstValue`]. Returns `true` if `value` is a
    /// valid constant of kind `self`. Unsupported variants (e.g. rich
    /// container wrappers outside the covered set) conservatively
    /// accept [`ConstValue::Placeholder`] and reject everything else,
    /// matching upstream's TyperError raising surface downstream.
    pub fn contains_value(&self, value: &ConstValue) -> bool {
        // Upstream special-cases `Placeholder` (used by normalizecalls
        // sentinel `description.NODEFAULT`) as a universally acceptable
        // constant while its holder recomputes the real type. Mirror
        // that tolerance so the normalizecalls rewrite branch does not
        // trip convert_const validation during mid-pipeline rewrites.
        if matches!(value, ConstValue::Placeholder) {
            return true;
        }
        match self {
            // upstream `lltype.py` — `Void._contains_value`
            // returns True for *any* value, not only None. Repr.py's
            // convert_const path relies on this to allow Void slots to
            // carry arbitrary Python sentinels during constant folding.
            LowLevelType::Void => true,
            // upstream `Bool = Primitive("Bool", False)`.
            LowLevelType::Bool => matches!(value, ConstValue::Bool(_)),
            // upstream `Signed` also accepts Symbolic values whose
            // `lltype()` is Signed; `llmemory.sizeof(TYPE)` returns an
            // `ItemOffset(TYPE)` symbolic on the typed-address path.
            LowLevelType::Signed => {
                matches!(
                    value,
                    ConstValue::Int(_)
                        | ConstValue::AddressOffset(_)
                        | ConstValue::InheritanceId { .. }
                )
            }
            // Upstream checks `typeOf(value)` against the exact Number
            // primitive (`lltype.py` `_contains_value` →
            // `isCompatibleType(typeOf(value), self)`). `Unsigned =
            // build_number("Unsigned", r_uint)` so the constant is
            // `ConstValue::UInt`. On a 64-bit host `r_ulonglong is
            // r_uint`, so `UnsignedLongLong` is the same type.
            // `_numbertypes[r_int] = _numbertypes[int]`: `SignedLongLong`
            // is `Signed` and its constants stay `Int`.
            LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => {
                matches!(value, ConstValue::UInt(_))
            }
            LowLevelType::SignedLongLong => matches!(value, ConstValue::Int(_)),
            LowLevelType::SignedLongLongLong => {
                matches!(value, ConstValue::Int(_) | ConstValue::Int128(_))
            }
            LowLevelType::UnsignedLongLongLong => {
                // `r_uint` is not `r_ulonglonglong` (`typeOf`).
                matches!(value, ConstValue::Int(_) | ConstValue::UInt128(_))
            }
            // upstream `Float` / `SingleFloat` / `LongFloat` accept
            // Python `float`.
            LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
                matches!(value, ConstValue::Float(_))
            }
            LowLevelType::Char => matches!(value, ConstValue::ByteStr(s) if s.len() == 1),
            LowLevelType::UniChar => {
                matches!(value, ConstValue::UniStr(s) if s.chars().count() == 1)
            }
            // Address is a Primitive with `_defl = NULL`. Concrete
            // values are llmemory.fakeaddress instances; pyre carries
            // them as `ConstValue::LLAddress`.
            LowLevelType::Address => matches!(value, ConstValue::LLAddress(_)),
            // upstream `Ptr(FuncType)` accepts `_ptr` instances — pyre's
            // `ConstValue::LLPtr` is the direct equivalent.
            LowLevelType::Ptr(_) => matches!(value, ConstValue::LLPtr(_)),
            LowLevelType::Struct(_)
            | LowLevelType::Array(_)
            | LowLevelType::FixedSizeArray(_)
            | LowLevelType::Opaque(_)
            | LowLevelType::Func(_)
            | LowLevelType::ForwardReference(_)
            | LowLevelType::InteriorPtr(_) => false,
        }
    }

    /// RPython `LowLevelType._short_name` (`lltype.py` default,
    /// `lltype.py` FuncType override, `lltype.py` Ptr
    /// override). Used by Repr's diagnostic messages (`rmodel.py`).
    /// Primitive types fall back to their class name; FuncType composes
    /// args/result recursively; Ptr prefixes with `"Ptr "` (not `"* "`,
    /// which is `Ptr.__str__`).
    pub fn short_name(&self) -> String {
        match self {
            LowLevelType::Void => "Void".to_string(),
            LowLevelType::Bool => "Bool".to_string(),
            LowLevelType::Signed => "Signed".to_string(),
            LowLevelType::Unsigned => "Unsigned".to_string(),
            LowLevelType::SignedLongLong => "SignedLongLong".to_string(),
            LowLevelType::SignedLongLongLong => "SignedLongLongLong".to_string(),
            LowLevelType::UnsignedLongLong => "UnsignedLongLong".to_string(),
            LowLevelType::UnsignedLongLongLong => "UnsignedLongLongLong".to_string(),
            LowLevelType::Float => "Float".to_string(),
            LowLevelType::SingleFloat => "SingleFloat".to_string(),
            LowLevelType::LongFloat => "LongFloat".to_string(),
            LowLevelType::Char => "Char".to_string(),
            LowLevelType::UniChar => "UniChar".to_string(),
            LowLevelType::Address => "Address".to_string(),
            LowLevelType::Struct(t) => t._short_name(),
            LowLevelType::Array(t) => t._short_name(),
            LowLevelType::FixedSizeArray(t) => t._short_name(),
            LowLevelType::Opaque(t) => t.tag.clone(),
            LowLevelType::Func(t) => t._short_name(),
            LowLevelType::ForwardReference(t) => t
                .resolved()
                .map_or_else(|| "ForwardReference".to_string(), |real| real.short_name()),
            // upstream `Ptr._short_name = "Ptr %s" % (self.TO._short_name(),)`
            // (`lltype.py`). Ptr.__str__ is `"* %s"` — a different
            // method, not used here.
            LowLevelType::Ptr(ptr) => format!("Ptr {}", ptr._to_short_name()),
            // upstream `_InteriorPtr._short_name` is not directly
            // defined — falls back to `LowLevelType._short_name` =
            // `str(self)` = class name. Match with `"InteriorPtr"`.
            LowLevelType::InteriorPtr(_) => "InteriorPtr".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttributeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GcKind {
    Raw,
    Gc,
    Prebuilt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowLevelValue {
    Void,
    Signed(i64),
    Unsigned(u64),
    Bool(bool),
    Float(u64),
    SingleFloat(u32),
    LongFloat(u64),
    Char(char),
    UniChar(char),
    /// RPython `llmemory.Address` values — `fakeaddress(ptr)` / `NULL`.
    /// The NULL sentinel is [`_address::Null`]; richer `fakeaddress`
    /// bodies will extend [`_address`] as llmemory wrappers land.
    Address(_address),
    Struct(Box<_struct>),
    Array(Box<_array>),
    Opaque(Box<_opaque>),
    Ptr(Box<_ptr>),
    InteriorPtr(Box<_interior_ptr>),
}

/// RPython `_fakeaddress(self, ptr)` (`llmemory.py:441-570`).
///
/// Upstream carries the underlying `_ptr` (or `None` for NULL) and
/// exposes `_cast_to_ptr` / `_cast_to_int` / dereference helpers.
/// Pyre carries the two arms consumers actually hit today — the NULL
/// sentinel + a `Fake(_ptr)` wrapper produced by
/// [`MultipleUnrelatedFrozenPBCRepr.convert_pbc`] via
/// `llmemory.fakeaddress(pbcptr)`. Cast / dereference helpers extend
/// `_address` later as further consumers land.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum _address {
    /// `NULL = fakeaddress(None)` (`llmemory.py`).
    Null,
    /// `fakeaddress(ptr)` — wraps a live `_ptr` so it can flow
    /// through `Address`-typed slots (e.g. `MUFrozenPBCRepr` whose
    /// lowleveltype is `llmemory.Address`). Upstream stores this on
    /// the dict-by-identity `converted_pbc_cache`; pyre keys the same
    /// cache by `DescKey` and stores the [`_address::Fake`] payload
    /// directly.
    ///
    /// `_address` no longer derives `Hash` because `_ptr` does not
    /// have a stable structural hash; the `Fake` arm carries a Box
    /// to keep the variant fixed-size.
    Fake(Box<_ptr>),
    /// Pyre's carrier for an `OpKind::ConstRefAddr` — a `Ref`-typed static
    /// constant whose host-evaluator value is a raw pointer **integer**, with
    /// no live `_ptr`/container to wrap. Upstream's rtyper holds the prebuilt
    /// object's real `_ptr` (via `convert_const`); pyre has only the integer,
    /// so it carries it directly.
    ///
    /// This is **not** the `cast_int_to_adr` tagged-integer fakeaddress: that
    /// path builds a proper `_NONGCREF` `_ptr` and wraps it as [`Fake`]
    /// (llmemory.py, `cast_int_to_ptr`). A `ConstRefAddr` raw pointer is
    /// typically aligned (even), which `cast_int_to_ptr` rejects, so it never
    /// reaches `Fake`; it flows only as a `Ref` constant and is never re-cast
    /// to a pointer through `cast_adr_to_ptr`.
    IntCast(i64),
}

impl _address {
    /// `fakeaddress.__eq__` (`llmemory.py`). Compares the two
    /// addresses by their underlying container: `obj1 = self.ptr._obj`,
    /// `obj2 = other.ptr._obj` (`None` for NULL), `return obj1 == obj2`.
    ///
    /// `_fixup()` (llmemory.py) is the identity here because the
    /// `llarena` fake-arena rebind it performs for a freed referent is
    /// unported; a freed `Fake` therefore reaches `_getobj(check=True)`
    /// and panics exactly as upstream `._obj` would on a non-arena freed
    /// pointer. Upstream catches only `DelayedPointer`, falling back to
    /// pointer identity (`self.ptr is other.ptr`).
    pub fn _eq(&self, other: &_address) -> bool {
        // A `ConstRefAddr` raw-pointer carrier compares by its raw integer —
        // two `Ref` statics are the same iff their prebuilt addresses match.
        // It is never equal to a NULL or container-backed address.
        match (self, other) {
            (_address::IntCast(a), _address::IntCast(b)) => return a == b,
            (_address::IntCast(_), _) | (_, _address::IntCast(_)) => return false,
            _ => {}
        }
        let obj = |a: &_address| -> Result<Option<_ptr_obj>, DelayedPointer> {
            match a {
                _address::Null => Ok(None),
                _address::Fake(p) => p._getobj(true),
                _address::IntCast(_) => unreachable!("int-cast handled above"),
            }
        };
        match (obj(self), obj(other)) {
            (Ok(o1), Ok(o2)) => o1 == o2,
            // `except lltype.DelayedPointer: return self.ptr is other.ptr`
            _ => match (self, other) {
                (_address::Fake(p1), _address::Fake(p2)) => p1._identity == p2._identity,
                _ => false,
            },
        }
    }

    /// `fakeaddress.__ne__` (`llmemory.py`): `not (self == other)`.
    pub fn _ne(&self, other: &_address) -> bool {
        !self._eq(other)
    }

    /// `fakeaddress.__nonzero__` (`llmemory.py`): `self.ptr is not
    /// None`. `Fake` wraps a live pointer (a tagged-integer `_NONGCREF`
    /// pointer included, whose `_obj0` is non-`None`) and `IntCast` a raw
    /// `ConstRefAddr` pointer; only `Null` — the normalization of `_obj0 is
    /// None` (llmemory.py) — is the NULL address.
    pub fn nonzero(&self) -> bool {
        matches!(self, _address::Fake(_) | _address::IntCast(_))
    }

    /// `fakeaddress.__lt__` (`llmemory.py`). For the convenience of
    /// debugging the GCs, NULL is the smallest address; two non-NULL
    /// fakeaddresses cannot be ordered (upstream `TypeError`, surfaced here
    /// as `Err` so the fold declines).
    pub fn _lt(&self, other: &_address) -> Result<bool, String> {
        if !other.nonzero() {
            return Ok(false); // self < NULL => False
        }
        if !self.nonzero() {
            return Ok(true); // NULL < non-null-other => True
        }
        Err("cannot compare non-NULL fakeaddresses with '<'".to_string())
    }

    /// `fakeaddress.__le__` (`llmemory.py`): `self == other or self
    /// < other`.
    pub fn _le(&self, other: &_address) -> Result<bool, String> {
        Ok(self._eq(other) || self._lt(other)?)
    }

    /// `fakeaddress.__gt__` (`llmemory.py`): `not (self == other or
    /// self < other)`.
    pub fn _gt(&self, other: &_address) -> Result<bool, String> {
        Ok(!(self._eq(other) || self._lt(other)?))
    }

    /// `fakeaddress.__ge__` (`llmemory.py`): `not (self < other)`.
    pub fn _ge(&self, other: &_address) -> Result<bool, String> {
        Ok(!self._lt(other)?)
    }

    /// `fakeaddress.__sub__` (`llmemory.py`), the
    /// `isinstance(other, fakeaddress)` arm: `addr1 - addr2` is `0` when the
    /// two addresses are equal, otherwise `TypeError("cannot subtract
    /// fakeaddresses in general")` (surfaced as `Err` so the fold declines).
    ///
    /// The other two arms of `__sub__` are not reachable through `op_adr_delta`
    /// (`opimpl.py` passes two checked addresses): the
    /// `isinstance(other, AddressOffset)` arm (`self + (-other)`) belongs to
    /// `op_adr_sub`, and the integer arm is rejected upstream.
    pub fn _delta(&self, other: &_address) -> Result<i64, String> {
        if self._eq(other) {
            Ok(0)
        } else {
            Err("cannot subtract fakeaddresses in general".to_string())
        }
    }
}

#[derive(Clone, Debug)]
pub enum LowLevelType {
    Void,
    Signed,
    Unsigned,
    SignedLongLong,
    SignedLongLongLong,
    UnsignedLongLong,
    UnsignedLongLongLong,
    Bool,
    Float,
    SingleFloat,
    LongFloat,
    Char,
    UniChar,
    /// RPython `Address = lltype.Primitive("Address", NULL)`
    /// (`llmemory.py`). Represents the primitive address type used
    /// by `MultipleUnrelatedFrozenPBCRepr.lowleveltype` and `adr_eq` /
    /// `adr_ne` operations. Values are [`LowLevelValue::Address`].
    Address,
    Func(Box<FuncType>),
    Struct(Box<Struct>),
    Array(Box<Array>),
    FixedSizeArray(Box<FixedSizeArray>),
    Opaque(Box<OpaqueType>),
    ForwardReference(Box<ForwardReference>),
    Ptr(Box<Ptr>),
    InteriorPtr(Box<InteriorPtr>),
}

impl PartialEq for LowLevelType {
    fn eq(&self, other: &Self) -> bool {
        // Cycle short-circuit: two ForwardReferences sharing the
        // same Arc compare equal without descending into resolved
        // types (closes `OBJECT_VTABLE → instantiate → OBJECTPTR →
        // OBJECT → CLASSTYPE → OBJECT_VTABLE`).
        if let (LowLevelType::ForwardReference(left_fwd), LowLevelType::ForwardReference(right_fwd)) =
            (self, other)
            && Arc::ptr_eq(&left_fwd.target, &right_fwd.target)
        {
            return true;
        }
        // Saferecursive guard — when one side is a ForwardReference
        // and we're already comparing it elsewhere on the stack
        // (cycle through the resolved type), short-circuit to `true`.
        // Mirrors RPython `saferecursive(safe_equal, True)` from
        // `lltype.py` — re-entering the same comparison means
        // the outer call hasn't returned False, so the optimistic
        // cycle assumption is "equal". The re-entry case happens for
        // `Struct == ForwardReference` asymmetric pairs where the
        // Struct contains a Ptr looping back to the same fwd.
        if let LowLevelType::ForwardReference(forward_ref) = self
            && let Some(real) = forward_ref.resolved()
        {
            let id = Arc::as_ptr(&forward_ref.target) as *const _ as usize;
            if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                return true;
            }
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
            let r = real == *other;
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
            return r;
        }
        if let LowLevelType::ForwardReference(forward_ref) = other
            && let Some(real) = forward_ref.resolved()
        {
            let id = Arc::as_ptr(&forward_ref.target) as *const _ as usize;
            if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                return true;
            }
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
            let r = *self == real;
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
            return r;
        }
        match (self, other) {
            (LowLevelType::Void, LowLevelType::Void)
            | (LowLevelType::Signed, LowLevelType::Signed)
            | (LowLevelType::Unsigned, LowLevelType::Unsigned)
            | (LowLevelType::SignedLongLong, LowLevelType::SignedLongLong)
            | (LowLevelType::SignedLongLongLong, LowLevelType::SignedLongLongLong)
            | (LowLevelType::UnsignedLongLong, LowLevelType::UnsignedLongLong)
            | (LowLevelType::UnsignedLongLongLong, LowLevelType::UnsignedLongLongLong)
            | (LowLevelType::Bool, LowLevelType::Bool)
            | (LowLevelType::Float, LowLevelType::Float)
            | (LowLevelType::SingleFloat, LowLevelType::SingleFloat)
            | (LowLevelType::LongFloat, LowLevelType::LongFloat)
            | (LowLevelType::Char, LowLevelType::Char)
            | (LowLevelType::UniChar, LowLevelType::UniChar)
            | (LowLevelType::Address, LowLevelType::Address) => true,
            (LowLevelType::Func(left), LowLevelType::Func(right)) => left == right,
            (LowLevelType::Struct(left), LowLevelType::Struct(right)) => left == right,
            (LowLevelType::Array(left), LowLevelType::Array(right)) => left == right,
            (LowLevelType::FixedSizeArray(left), LowLevelType::FixedSizeArray(right)) => {
                left == right
            }
            (LowLevelType::Opaque(left), LowLevelType::Opaque(right)) => left == right,
            (LowLevelType::ForwardReference(left), LowLevelType::ForwardReference(right)) => {
                left == right
            }
            (LowLevelType::Ptr(left), LowLevelType::Ptr(right)) => left == right,
            (LowLevelType::InteriorPtr(left), LowLevelType::InteriorPtr(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for LowLevelType {}

impl Hash for LowLevelType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LowLevelType::Void => 0_u8.hash(state),
            LowLevelType::Signed => 1_u8.hash(state),
            LowLevelType::Unsigned => 2_u8.hash(state),
            LowLevelType::SignedLongLong => 3_u8.hash(state),
            LowLevelType::SignedLongLongLong => 4_u8.hash(state),
            LowLevelType::UnsignedLongLong => 5_u8.hash(state),
            LowLevelType::UnsignedLongLongLong => 6_u8.hash(state),
            LowLevelType::Bool => 7_u8.hash(state),
            LowLevelType::Float => 8_u8.hash(state),
            LowLevelType::SingleFloat => 9_u8.hash(state),
            LowLevelType::LongFloat => 10_u8.hash(state),
            LowLevelType::Char => 11_u8.hash(state),
            LowLevelType::UniChar => 12_u8.hash(state),
            LowLevelType::Func(t) => {
                13_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Struct(t) => {
                14_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Array(t) => {
                15_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::FixedSizeArray(t) => {
                16_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Opaque(t) => {
                17_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::ForwardReference(t) => {
                // Saferecursive cycle guard — when the resolved type
                // contains a Ptr that loops back to this same
                // ForwardReference (e.g. `OBJECT_VTABLE.instantiate
                // → Ptr(FuncType([], OBJECTPTR)) → OBJECT.typeptr →
                // CLASSTYPE → OBJECT_VTABLE`), hashing `real`
                // unconditionally recurses forever. Mirrors RPython
                // `saferecursive(get_hash, 0)` (lltype.py) — on
                // re-entry hash a constant 0 so deterministic across
                // runs and identity-independent.
                let id = Arc::as_ptr(&t.target) as *const _ as usize;
                let already = FORWARD_REF_HASH_GUARD.with(|g| g.borrow().contains(&id));
                if already {
                    0_u8.hash(state);
                    return;
                }
                if let Some(real) = t.resolved() {
                    FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().insert(id));
                    real.hash(state);
                    FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().remove(&id));
                } else {
                    18_u8.hash(state);
                    t.hash(state);
                }
            }
            LowLevelType::Ptr(t) => {
                19_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::InteriorPtr(t) => {
                20_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Address => 21_u8.hash(state),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub args: Vec<ConcretetypePlaceholder>,
    pub result: ConcretetypePlaceholder,
}

/// RPython `Struct`/`GcStruct` (`lltype.py`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Struct {
    pub _name: String,
    pub _flds: frozendict<ConcretetypePlaceholder>,
    pub _names: Vec<String>,
    pub _adtmeths: frozendict<ConstValue>,
    pub _hints: frozendict<ConstValue>,
    pub _arrayfld: Option<String>,
    pub _gckind: GcKind,
    /// RPython `RttiStruct._runtime_type_info` (`lltype.py`).
    /// `None` for plain `Struct`/`GcStruct` without rtti; populated by
    /// `Struct::gc_rtti` (or a later `_install_extras(rtti=True)`
    /// port) with a freshly-minted opaque whose identity distinguishes
    /// two structurally-equal `GcStruct(..., rtti=True)` builds — the
    /// same distinction upstream Python makes via per-instance
    /// `_runtime_type_info` attrs.
    pub _runtime_type_info: Option<Box<_opaque>>,
}

impl std::fmt::Debug for Struct {
    /// Parity with `Struct.__str__` short form (lltype.py):
    /// `"%s %s { %s }" % (self.__class__.__name__, self._name,
    /// ', '.join(self._names))` — the class name (`Struct`/`GcStruct`) and
    /// field NAMES only, never the field TYPES in `_flds`. Descending into
    /// `_flds` re-expands recursive / DAG-shared struct graphs (e.g.
    /// `pyframe::PyFrame` reached through a `Ptr` in its own fields) into a
    /// combinatorially huge string, overflowing the stack when a
    /// `where_info` diagnostic formats a variable's concretetype.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {{ {} }}",
            self._class_name(),
            self._name,
            self._names.join(", ")
        )
    }
}

/// RPython `Array`/`GcArray` (`lltype.py`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Array {
    pub OF: ConcretetypePlaceholder,
    pub _hints: frozendict<ConstValue>,
    pub _gckind: GcKind,
}

/// RPython `FixedSizeArray` (`lltype.py`) — structurally a
/// `Struct` with fields `item0..itemN-1`. The Rust port keeps `OF` and
/// `length` direct so lookups match array-indexing semantics without
/// walking a `Struct._flds` list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedSizeArray {
    pub OF: ConcretetypePlaceholder,
    pub length: usize,
    pub _hints: frozendict<ConstValue>,
    pub _gckind: GcKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpaqueType {
    pub tag: String,
    pub _gckind: GcKind,
}

/// Flattens upstream's three sibling classes
/// `ForwardReference(_gckind='raw')`, `GcForwardReference(_gckind='gc')`,
/// and `FuncForwardReference(_gckind='prebuilt')` (`lltype.py`)
/// into one struct with an explicit `_gckind` field. Upstream `become()`
/// rebinds `self.__class__` / `self.__dict__` to the realcontainertype
/// (`lltype.py`) so every pointer to the same `ForwardReference`
/// observes the resolved type. Rust cannot re-tag enum variants in place,
/// so clones share a mutable target cell and pointer-op sites unwrap it.
#[derive(Clone)]
pub struct ForwardReference {
    pub _gckind: GcKind,
    target: Arc<Mutex<Option<LowLevelType>>>,
}

impl std::fmt::Debug for ForwardReference {
    /// The resolved type sits in `target` and can point back at this same
    /// forward ref (`OBJECT_VTABLE` → `instantiate` → … → `OBJECT_VTABLE`).
    /// Derived `Debug` follows that arc and overflows the stack, or builds
    /// one combinatorial string, the same reason [`Struct`]'s `Debug` does
    /// not print `_flds`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A resolved struct prints its short name (`GcStruct PyFrame { … }`).
        // `Struct`'s `Debug` lists field names only, so this does not follow
        // a pointer back into this forward ref. Any other target stays a flag:
        // printing it would walk the cycle this impl exists to cut.
        let slot = self.target.try_lock();
        if let Some(LowLevelType::Struct(resolved)) = slot.as_ref().and_then(|slot| slot.as_ref()) {
            return write!(
                f,
                "ForwardReference {{ _gckind: {:?}, resolved: {resolved:?} }}",
                self._gckind
            );
        }
        let resolved = slot.is_some_and(|slot| slot.is_some());
        write!(
            f,
            "ForwardReference {{ _gckind: {:?}, resolved: {resolved} }}",
            self._gckind
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InteriorOffset {
    Field(String),
    Index(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InteriorPtr {
    pub PARENTTYPE: Box<LowLevelType>,
    pub TO: Box<LowLevelType>,
    pub offsets: Vec<InteriorOffset>,
}

#[derive(Clone, Debug)]
pub enum PtrTarget {
    Func(FuncType),
    Struct(Struct),
    Array(Array),
    FixedSizeArray(FixedSizeArray),
    Opaque(OpaqueType),
    ForwardReference(ForwardReference),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ptr {
    pub TO: PtrTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelPointerType {
    Ptr(Ptr),
    InteriorPtr(InteriorPtr),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelAdtMember {
    Value(ConstValue),
    Method {
        ll_ptrtype: LowLevelPointerType,
        func: ConstValue,
    },
}

impl From<Ptr> for LowLevelType {
    fn from(value: Ptr) -> Self {
        LowLevelType::Ptr(Box::new(value))
    }
}

impl From<InteriorPtr> for LowLevelType {
    fn from(value: InteriorPtr) -> Self {
        LowLevelType::InteriorPtr(Box::new(value))
    }
}

impl From<LowLevelPointerType> for LowLevelType {
    fn from(value: LowLevelPointerType) -> Self {
        match value {
            LowLevelPointerType::Ptr(ptr) => LowLevelType::Ptr(Box::new(ptr)),
            LowLevelPointerType::InteriorPtr(ptr) => LowLevelType::InteriorPtr(Box::new(ptr)),
        }
    }
}

impl From<PtrTarget> for LowLevelType {
    fn from(value: PtrTarget) -> Self {
        match value {
            PtrTarget::Func(func) => LowLevelType::Func(Box::new(func)),
            PtrTarget::Struct(t) => LowLevelType::Struct(Box::new(t)),
            PtrTarget::Array(t) => LowLevelType::Array(Box::new(t)),
            PtrTarget::FixedSizeArray(t) => LowLevelType::FixedSizeArray(Box::new(t)),
            PtrTarget::Opaque(t) => LowLevelType::Opaque(Box::new(t)),
            PtrTarget::ForwardReference(t) => LowLevelType::ForwardReference(Box::new(t)),
        }
    }
}

impl PartialEq for PtrTarget {
    fn eq(&self, other: &Self) -> bool {
        LowLevelType::from(self.clone()) == LowLevelType::from(other.clone())
    }
}

impl Eq for PtrTarget {}

impl Hash for PtrTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        LowLevelType::from(self.clone()).hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct _func {
    pub TYPE: FuncType,
    pub _name: String,
    pub graph: Option<usize>,
    pub _callable: Option<String>,
    pub attrs: HashMap<String, ConstValue>,
}

impl _func {
    pub fn new(
        TYPE: FuncType,
        _name: String,
        graph: Option<usize>,
        _callable: Option<String>,
        attrs: HashMap<String, ConstValue>,
    ) -> Self {
        _func {
            TYPE,
            _name,
            graph,
            _callable,
            attrs,
        }
    }
}

/// `_struct._fields` is wrapped in `Arc<Mutex<...>>` so a `_struct`
/// `#[derive(Clone)]` shares its field storage with its clones.
/// Mirrors RPython's pointer-by-identity semantics on Python objects:
/// `iprebuiltinstances[value] = result; init(value, result)` (rclass.py
/// :799-800) requires that mutations to `result` mid-init are visible
/// to recursive `convert_const(value)` cache hits.
///
/// Without the Arc-share, `_ptr.clone()` deep-copies `_obj0` →
/// `_struct.clone()` → `_fields.clone()` (owned `Vec`), so two clones
/// of the same prebuilt instance carry independent field storage and
/// intra-class circular prebuilt instances (`a.b.back == a`) wire
/// `back` to a stale snapshot rather than the in-progress `a`. This
/// is the structural pre-requisite for
/// `InstanceRepr::convert_const_exact` byte-for-byte parity with
/// `rclass.py`.
///
/// `Arc<Mutex<>>` (rather than `Rc<RefCell<>>`) because
/// `LowLevelValue` carriers (e.g. `static LazyLock<LowLevelType>` for
/// `STR`/`UNICODE`) traverse the Send/Sync boundary; `_struct` itself
/// must therefore be Send+Sync. Single-threaded RPython has no need
/// for the lock; pyre's lock acquisitions are uncontended in practice
/// (the tracer/codewriter share the per-process repr cache without
/// concurrent mutation of any single `_struct`).
#[derive(Debug)]
pub struct StructCore {
    pub TYPE: Struct,
    pub _fields: Mutex<Vec<(String, LowLevelValue)>>,
    /// `_parentable` state — `_storage`/`_wrparent`/… on the container
    /// object (lltype.py). Owned inline by the Core.
    pub(crate) _parentable: Parentable,
}

#[derive(Clone, Debug)]
pub struct _struct(pub(crate) Arc<StructCore>);

impl std::ops::Deref for _struct {
    type Target = StructCore;
    fn deref(&self) -> &StructCore {
        &self.0
    }
}

impl _struct {
    /// `_struct.__init__` (lltype.py): fresh container, live storage.
    pub(crate) fn from_parts(TYPE: Struct, fields: Vec<(String, LowLevelValue)>) -> Self {
        _struct(Arc::new(StructCore {
            TYPE,
            _fields: Mutex::new(fields),
            _parentable: Parentable::new_inline(),
        }))
    }

    /// `id(self)` / `is` container identity (lltype.py _identityhash) — the
    /// `Arc` allocation address. Stable for the lifetime of the `Arc`;
    /// every value-clone shares the same `Arc`, hence the same identity.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
}

/// `_array.items` is wrapped in `Arc<Mutex<...>>` for the same shared-
/// storage discipline as `_struct._fields` — see the `_struct` doc
/// above for the parity rationale.
#[derive(Debug)]
pub struct ArrayCore {
    pub TYPE: ArrayContainer,
    pub items: Mutex<Vec<LowLevelValue>>,
    /// `_parentable` state (lltype.py). Owned inline by the Core.
    pub(crate) _parentable: Parentable,
}

#[derive(Clone, Debug)]
pub struct _array(pub(crate) Arc<ArrayCore>);

impl std::ops::Deref for _array {
    type Target = ArrayCore;
    fn deref(&self) -> &ArrayCore {
        &self.0
    }
}

impl _array {
    /// `_array.__init__` (lltype.py): fresh container, live storage.
    pub(crate) fn from_parts(TYPE: ArrayContainer, items: Vec<LowLevelValue>) -> Self {
        _array(Arc::new(ArrayCore {
            TYPE,
            items: Mutex::new(items),
            _parentable: Parentable::new_inline(),
        }))
    }

    /// `id(self)` / `is` container identity (lltype.py _identityhash) — the
    /// `Arc` allocation address. Stable for the lifetime of the `Arc`;
    /// every value-clone shares the same `Arc`, hence the same identity.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArrayContainer {
    Array(Array),
    FixedSizeArray(FixedSizeArray),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum _ptr_obj {
    Func(_func),
    Struct(_struct),
    Array(_array),
    Opaque(_opaque),
    Wref(_wref),
    /// `_subarray` interior view (lltype.py), built by
    /// `direct_fieldptr`/`direct_arrayitems`/`direct_ptradd`.
    Subarray(_subarray),
    /// `_arraylenref` array-length pseudo-reference (lltype.py), built by
    /// `ArrayLengthOffset.ref`.
    ArrayLenRef(_arraylenref),
    /// `_endmarker_struct` end-of-array sentinel (lltype.py:168), built by
    /// `ItemOffset.ref` for a reference exactly to the array end.
    EndMarker(_endmarker),
    /// Tagged-integer pointer payload — `_ptr(PTRTYPE, oddint, solid=True)`
    /// stores the bare odd integer in `_obj0` (lltype.py,
    /// `cast_int_to_ptr`). Upstream `_obj0` is a plain `int` here; modeled as
    /// a dedicated variant since the int is not a container. Consumed by
    /// `_cast_to_int` (lltype.py) and round-tripped through
    /// `cast_adr_to_ptr` of a `cast_int_to_adr` tagged address.
    IntCast(i64),
}

/// `_parentable._wrparent = weakref.ref(parent)` (lltype.py). A weak
/// handle to a parent container — one of the three `_parentable` subclasses
/// (`_struct`/`_array`/`_opaque`). Held weak so the child→parent link does
/// not keep the parent alive; `_parentstructure`/`_was_freed` upgrade it and
/// raise "already garbage collected parent" when it is gone (lltype.py).
#[derive(Clone, Debug)]
pub enum WeakContainer {
    Struct(std::sync::Weak<StructCore>),
    Array(std::sync::Weak<ArrayCore>),
    Opaque(std::sync::Weak<OpaqueCore>),
}

impl WeakContainer {
    /// `weakref.ref(parent)` — downgrade a strong container handle.
    fn downgrade(obj: &_ptr_obj) -> Option<WeakContainer> {
        match obj {
            _ptr_obj::Struct(s) => Some(WeakContainer::Struct(Arc::downgrade(&s.0))),
            _ptr_obj::Array(a) => Some(WeakContainer::Array(Arc::downgrade(&a.0))),
            _ptr_obj::Opaque(o) => Some(WeakContainer::Opaque(Arc::downgrade(&o.0))),
            _ => None,
        }
    }

    /// Upgrade back to the strong container, or `None` if collected.
    fn upgrade(&self) -> Option<_ptr_obj> {
        match self {
            WeakContainer::Struct(w) => w.upgrade().map(|c| _ptr_obj::Struct(_struct(c))),
            WeakContainer::Array(w) => w.upgrade().map(|c| _ptr_obj::Array(_array(c))),
            WeakContainer::Opaque(w) => w.upgrade().map(|c| _ptr_obj::Opaque(_opaque(c))),
        }
    }
}

/// `_ptr._obj0` slot (lltype.py `_obj0` attribute). `_setobj`
/// (lltype.py) classifies the target into one of these forms when
/// a pointer is set:
/// * `Null` — `pointing_to is None`.
/// * `Strong(obj)` — solid pointer, or `_T._gckind != 'raw'`, or a
///   `FuncType` pointer; `_obj0 = pointing_to` keeps the container alive.
/// * `Weak(wref)` — a raw, non-solid, non-func container; `_obj0 =
///   weakref.ref(pointing_to)` so the pointer does not keep it alive.
/// * `Delayed` — a forward `DelayedPointer` whose target is not yet known.
#[derive(Clone, Debug)]
pub enum PtrObj {
    Null,
    Strong(_ptr_obj),
    Weak(WeakContainer),
    Delayed(DelayedPointer),
}

#[derive(Debug)]
pub struct OpaqueCore {
    pub TYPE: OpaqueType,
    /// Optional human-readable name — `_opaque(TYPE, _name=name, **attrs)`
    /// kwarg upstream (used by `opaqueptr` to mint named containers such
    /// as `GCSTRUCT._runtime_type_info`). Plain `_opaque(TYPE)` defaults
    /// to the upstream `"?"` marker.
    pub _name: Option<String>,
    /// Upstream `opaqueptr(..., about=self)` stores the source type on
    /// the opaque itself (`lltype.py`).
    pub about: Option<LowLevelType>,
    /// Upstream `_attach_runtime_type_info_funcptr()` stores validated
    /// helper pointers directly on the RTTI opaque (`lltype.py`):
    /// `self._runtime_type_info.query_funcptr = funcptr`. The store mutates
    /// the existing opaque in place, so these are `Mutex` cells behind the
    /// shared `Arc<OpaqueCore>` — the funcptr attach keeps the opaque's
    /// `Arc::as_ptr` identity, matching `id(self._runtime_type_info)`.
    pub query_funcptr: Mutex<Option<Box<_ptr>>>,
    pub destructor_funcptr: Mutex<Option<Box<_ptr>>>,
    /// `cast_opaque_ptr(..., 'hidden', container=..., ORIGTYPE=...,
    /// solid=...)` stores the original container and its pointer type on
    /// the opaque so a later opaque→concrete cast can rebuild the real
    /// pointer (`lltype.py`). `None` on every other opaque.
    pub container: Option<Box<_ptr_obj>>,
    pub ORIGTYPE: Option<LowLevelType>,
    pub solid: bool,
    /// `_parentable` state (lltype.py); `_opaque` is a
    /// `_parentable` (lltype.py). Owned inline by the Core.
    pub(crate) _parentable: Parentable,
}

#[derive(Clone, Debug)]
pub struct _opaque(pub(crate) Arc<OpaqueCore>);

impl std::ops::Deref for _opaque {
    type Target = OpaqueCore;
    fn deref(&self) -> &OpaqueCore {
        &self.0
    }
}

impl _opaque {
    /// `_opaque.__init__` (lltype.py): fresh container, live storage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        TYPE: OpaqueType,
        _name: Option<String>,
        about: Option<LowLevelType>,
        query_funcptr: Option<Box<_ptr>>,
        destructor_funcptr: Option<Box<_ptr>>,
        container: Option<Box<_ptr_obj>>,
        ORIGTYPE: Option<LowLevelType>,
        solid: bool,
    ) -> Self {
        _opaque(Arc::new(OpaqueCore {
            TYPE,
            _name,
            about,
            query_funcptr: Mutex::new(query_funcptr),
            destructor_funcptr: Mutex::new(destructor_funcptr),
            container,
            ORIGTYPE,
            solid,
            _parentable: Parentable::new_inline(),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct _interior_ptr {
    pub _T: LowLevelType,
    pub _parent: LowLevelValue,
    pub _offsets: Vec<InteriorOffset>,
}

impl PartialEq for _interior_ptr {
    fn eq(&self, other: &Self) -> bool {
        if self._TYPE() != other._TYPE() {
            panic!("comparing {:?} and {:?}", self._TYPE(), other._TYPE());
        }
        self._obj() == other._obj()
    }
}

impl Eq for _interior_ptr {}

impl _func {
    /// RPython `_abstract_ptr.__call__` / `_func` arg validation
    /// (`lltype.py`). Arity mismatch and per-arg type
    /// mismatch raise `TypeError`; a None `_callable` raises
    /// `RuntimeError`. The Rust port models `_callable` as an optional
    /// name string rather than a real closure, so the return value is
    /// the result type's default — matching the `_container_example`
    /// closure `def ex(*args): return self.RESULT._defl()`.
    pub fn call(&self, args: &[LowLevelValue]) -> LowLevelValue {
        if args.len() != self.TYPE.args.len() {
            panic!(
                "calling {:?} with wrong argument number: got {}, expected {}",
                self.TYPE,
                args.len(),
                self.TYPE.args.len(),
            );
        }
        for (i, (arg, expected)) in args.iter().zip(self.TYPE.args.iter()).enumerate() {
            let got = typeOf_value(arg);
            if got == *expected {
                continue;
            }
            // upstream lltype.py:1357-1376 special cases:
            // - Void arg: `ARG == Void` accepts any value.
            // - Pointer arg expected, None actual: accepted.
            // - ContainerType expected, Ptr(ContainerType) actual:
            //   accepted (backends unwrap).
            if matches!(expected, LowLevelType::Void) {
                continue;
            }
            if matches!(expected, LowLevelType::Ptr(_)) && matches!(arg, LowLevelValue::Void) {
                continue;
            }
            if expected.is_container_type()
                && let LowLevelValue::Ptr(ptr) = arg
            {
                let expected_ptr = LowLevelType::Ptr(Box::new(Ptr {
                    TO: match expected {
                        LowLevelType::Func(t) => PtrTarget::Func((**t).clone()),
                        LowLevelType::Struct(t) => PtrTarget::Struct((**t).clone()),
                        LowLevelType::Array(t) => PtrTarget::Array((**t).clone()),
                        LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray((**t).clone()),
                        LowLevelType::Opaque(t) => PtrTarget::Opaque((**t).clone()),
                        LowLevelType::ForwardReference(t) => {
                            PtrTarget::ForwardReference((**t).clone())
                        }
                        _ => unreachable!(),
                    },
                }));
                if expected_ptr == LowLevelType::Ptr(Box::new(ptr._TYPE.clone())) {
                    continue;
                }
            }
            panic!(
                "calling {:?} with wrong argument type at index {}: expected {:?}, got {:?}",
                self.TYPE, i, expected, got
            );
        }
        if self._callable.is_none() {
            panic!("calling undefined function {:?}", self._name);
        }
        self.TYPE.result._defl()
    }
}

/// Upstream `_container` inherits Python `object`'s identity-based
/// `__eq__` / `__hash__` (`lltype.py`); two containers compare
/// equal iff they are the same Python object (`id(self) == id(other)`).
/// The `Arc`-backed containers (`_struct` / `_array` / `_opaque`) derive
/// `is`-identity from `Arc::as_ptr` directly (see `_struct::identity`),
/// matching `lltype.py`. This counter only serves the remaining
/// `#[derive(Clone)]` value containers (`_subarray`, `_arraylenref`,
/// `_endmarker`, `_wref`) that are not yet behind an `Arc`: cloning such a
/// value must keep the same identity, which a stored counter (not the
/// address of a freely-duplicated body) provides.
fn fresh_low_level_container_identity() -> usize {
    static NEXT_LOW_LEVEL_CONTAINER_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_LOW_LEVEL_CONTAINER_ID.fetch_add(1, Ordering::Relaxed)
}

fn fresh_low_level_pointer_identity() -> u64 {
    static NEXT_LOW_LEVEL_POINTER_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_LOW_LEVEL_POINTER_ID.fetch_add(1, Ordering::Relaxed) as u64
}

/// Shared, clone-aliased `_parentable` state — upstream
/// `class _parentable(_container)` slots `_storage`, `_wrparent`,
/// `_parent_type`, `_parent_index` (lltype.py). Held behind an
/// `Arc` so every value-clone of a container observes `_free` and the
/// parent link, exactly the way Python's reference objects share one
/// `__dict__`. `_struct`/`_array`/`_opaque` each embed one — they are the
/// three `_parentable` subclasses (`_func`/`_wref` are plain
/// `_container`s, lltype.py, and carry no `Parentable`).
#[derive(Debug)]
pub struct Parentable {
    /// `_storage` (lltype.py:1663): `true` = live (upstream "use default
    /// storage"), `false` = freed (upstream `None`). The ctypes-object
    /// variant is not modeled (ll2ctypes is dead in pyre). `AtomicBool`
    /// keeps the hot `_check`/`_was_freed` read off a mutex.
    storage_live: std::sync::atomic::AtomicBool,
    /// `_wrparent`/`_parent_type`/`_parent_index`/`_keepparent`
    /// (lltype.py), installed by `_setparentstructure`.
    parent: Mutex<Option<ParentLink>>,
}

/// `_parentable._wrparent` + `_parent_type` + `_parent_index` + `_keepparent`
/// (lltype.py).
struct ParentLink {
    /// `_wrparent = weakref.ref(parent)` (lltype.py) — a weak handle to
    /// the parent *container object* (`_struct`/`_array`/`_opaque`), upgraded
    /// by `_parentstructure()` (lltype.py) so `parentlink` /
    /// `_normalizedcontainer` / `_cast_to` up-casts can re-point at its
    /// inlined fields. `None` after the parent is collected.
    wrparent: WeakContainer,
    /// `_parent_type` (lltype.py) — `typeOf(parent)`. Read by the
    /// `_cast_to` up-cast walk (the widened-to type check) and the
    /// error-message/normalizedcontainer hints.
    parent_type: LowLevelType,
    /// `_parent_index` (lltype.py:1696) — the field name or item index that
    /// holds this sub-container inside the parent.
    parent_index: ParentIndex,
    /// `_keepparent` (lltype.py:1697-1702) — a strong ref to the parent kept
    /// only when this child is the parent's first inlined field of matching
    /// `_gckind`, so the shared allocation stays alive as long as the child.
    /// `None` otherwise.
    keepparent: Option<_ptr_obj>,
}

impl std::fmt::Debug for ParentLink {
    /// Non-recursive: `wrparent` is a `Weak` so printing it does not recurse
    /// into the target. `keepparent` (when `Some`) is a strong `_ptr_obj` to
    /// the parent container; that container's own `ParentLink` could chain
    /// further up, making the depth unbounded for deep struct hierarchies.
    /// Show only the type/index slots, like upstream's `_parent_type`/
    /// `_parent_index`; omit `wrparent`/`keepparent`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentLink")
            .field("parent_type", &self.parent_type)
            .field("parent_index", &self.parent_index)
            .finish_non_exhaustive()
    }
}

/// `_parent_index` (lltype.py:1696): a struct field name or an array item
/// index. Inner values are read by the deferred `_parentstructure()` walk
/// (see [`ParentLink::parent_index`]). The item index is signed: a
/// `direct_ptradd` backward shift produces a `_subarray` at `base + n < 0`
/// (lltype.py), which only fails when actually dereferenced out of
/// bounds, not when the interior pointer is built.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParentIndex {
    Field(String),
    Item(i64),
}

impl Parentable {
    /// `_parentable.__init__` (lltype.py) — fresh container,
    /// live storage, no parent.
    fn new() -> Arc<Self> {
        Arc::new(Parentable {
            storage_live: std::sync::atomic::AtomicBool::new(true),
            parent: Mutex::new(None),
        })
    }

    /// Inline (non-Arc) ctor — the Core now owns the `Parentable` directly.
    fn new_inline() -> Self {
        Parentable {
            storage_live: std::sync::atomic::AtomicBool::new(true),
            parent: Mutex::new(None),
        }
    }

    /// `_parentable._was_freed` (lltype.py): freed iff this
    /// container's own storage was freed, or — walking `_wrparent` — its
    /// parent structure is freed. Raises if the `_wrparent` weakref was
    /// garbage collected.
    fn was_freed(&self) -> bool {
        if !self.storage_live.load(Ordering::Relaxed) {
            return true;
        }
        match &*self.parent.lock() {
            None => false,
            Some(link) => match link.wrparent.upgrade() {
                None => panic!(
                    "accessing sub-container, but already garbage collected parent {:?}",
                    link.parent_type
                ),
                Some(parent) => parentable_of_obj(&parent)
                    .expect("a parent container is always a `_parentable`")
                    .was_freed(),
            },
        }
    }

    /// `_parentable._check` (lltype.py): accessing freed storage
    /// is an error. Upstream checks `self._storage is None` then recurses
    /// through `_parentstructure()` (which calls `parent._check()`); the
    /// net condition is "freed anywhere in the chain", i.e. `_was_freed()`.
    fn check(&self) {
        if self.was_freed() {
            panic!("accessing freed container");
        }
    }

    /// `_parentable._free` (lltype.py): `self._check()` (no
    /// double-frees) then `self._storage = None`.
    ///
    /// `_protect`/`_unprotect` (lltype.py) are intentionally not
    /// ported — their sole consumer is `ll2ctypes` (dead in pyre) and the
    /// freed-bit model carries no `_storage` object to round-trip.
    fn free(&self) {
        self.check();
        self.storage_live.store(false, Ordering::Relaxed);
    }

    /// `_parentable._setparentstructure(parent, parentindex)`
    /// (lltype.py). `parent` is the parent container object;
    /// `_wrparent = weakref.ref(parent)`, `_parent_type = typeOf(parent)`.
    /// `_keepparent` keeps a strong ref to the parent only when this child
    /// (`child_type` = `self._TYPE`) is the parent struct's first inlined
    /// field of matching `_gckind`.
    fn set_parent_structure(
        &self,
        child_type: &LowLevelType,
        parent_container: _ptr_obj,
        parent_type: LowLevelType,
        parent_index: ParentIndex,
    ) {
        let wrparent = WeakContainer::downgrade(&parent_container)
            .expect("a parent container is always Struct/Array/Opaque");
        // `_keepparent` (lltype.py:1697-1702): keep a strong ref only when the
        // parent is a Struct with names, this child is its first field
        // (`parentindex in (names[0], 0)`), and child and parent share the
        // same `_gckind`. `FixedSizeArray(Struct)` (lltype.py) is a Struct
        // subclass with synthetic field names `item0, item1, …`, so a child
        // inlined as `item0` (`_names[0]`) of a matching-`_gckind` fixedsize
        // array parent also qualifies.
        let same_gckind = child_type._gckind() == parent_type._gckind();
        let is_first_field = match &parent_type {
            // Plain Struct: first field is `_names[0]` (non-empty).
            LowLevelType::Struct(st) => {
                !st._names.is_empty() && parent_first_field_match(&st._names, &parent_index)
            }
            // FixedSizeArray: a real parent always has at least `item0`
            // (`length > 0`); the first item is field `item0`, i.e. the
            // integer-`0` disjunct of upstream `(names[0], 0)`.
            LowLevelType::FixedSizeArray(arr) => {
                arr.length > 0 && parent_index == ParentIndex::Field("item0".to_string())
            }
            _ => false,
        };
        let keepparent = if is_first_field && same_gckind {
            Some(parent_container.clone())
        } else {
            None
        };
        *self.parent.lock() = Some(ParentLink {
            wrparent,
            parent_type,
            parent_index,
            keepparent,
        });
    }

    /// `_parentable._parentstructure(check=True)` (lltype.py): hand
    /// back the parent container object, running its `_check()` first when
    /// `check`. `None` for a container with no parent link.
    fn parentstructure(&self, check: bool) -> Option<_ptr_obj> {
        let guard = self.parent.lock();
        let link = guard.as_ref()?;
        let parent = match link.wrparent.upgrade() {
            None => panic!(
                "accessing sub-container, but already garbage collected parent {:?}",
                link.parent_type
            ),
            Some(parent) => parent,
        };
        if check {
            parentable_of_obj(&parent)
                .expect("a parent container is always a `_parentable`")
                .check();
        }
        Some(parent)
    }

    /// `container._keepparent = container._parentstructure()` (constfold.py:
    /// 137, `fixup_solid`). Forces a strong ref to the (upgraded) parent
    /// regardless of the first-field/`_gckind` rule `set_parent_structure`
    /// applies, so a pointer to an inlined part keeps the whole parent alive.
    /// No-op when the container has no parent link (`_parentstructure()` is
    /// `None`), matching `_keepparent = None`.
    fn force_keepparent(&self) {
        let parent = self.parentstructure(false);
        if let Some(link) = self.parent.lock().as_mut() {
            link.keepparent = parent;
        }
    }

    /// True when this container's recorded weak parent still upgrades to
    /// `candidate` (the same live container `Arc`). An interior container
    /// memoized by the parent's `Arc::as_ptr` address fails this check once
    /// the original parent is gone — the weak ref is dead (or, on a reused
    /// address, points at a different `Arc`) — so the caller recomputes. This
    /// emulates the `WeakKeyDictionary` eviction plus the `RuntimeError:
    /// pointer comparison with a freed structure` retry that upstream
    /// `_subarray._makeptr` performs (lltype.py).
    fn parent_is(&self, candidate: &_ptr_obj) -> bool {
        match self.parent.lock().as_ref() {
            None => false,
            Some(link) => match link.wrparent.upgrade() {
                Some(parent) => parent.same_arc(candidate),
                None => false,
            },
        }
    }

    /// `_parentable._parent_index` (lltype.py) — `self`'s field name or
    /// item index inside its parent, or `None` when unparented.
    fn parent_index(&self) -> Option<ParentIndex> {
        self.parent.lock().as_ref().map(|l| l.parent_index.clone())
    }

    /// `_parentable._parent_type` (lltype.py) — `typeOf(parent)`, or
    /// `None` when unparented. Read by the `_cast_to` up-cast walk to check
    /// the widened-to type against `PTRTYPE.TO`.
    fn parent_type(&self) -> Option<LowLevelType> {
        self.parent.lock().as_ref().map(|l| l.parent_type.clone())
    }
}

/// The shared `_parentable` state of a container *object* (`_struct`/
/// `_array`/`_opaque`), or `None` for the non-parentable `_func`/`_wref`.
/// Companion to [`parentable_of`] for the `_ptr_obj` (pointed-to container)
/// representation that `_parentstructure` hands back.
/// `parentindex in (self._parent_type._names[0], 0)` (lltype.py): the
/// child is the parent struct's first field — either its first named field,
/// or item index 0. `names` is the parent struct's `_names` (non-empty).
fn parent_first_field_match(names: &[String], parent_index: &ParentIndex) -> bool {
    match parent_index {
        ParentIndex::Field(name) => name == &names[0],
        ParentIndex::Item(i) => *i == 0,
    }
}

fn parentable_of_obj(obj: &_ptr_obj) -> Option<&Parentable> {
    match obj {
        _ptr_obj::Struct(s) => Some(&s._parentable),
        _ptr_obj::Array(a) => Some(&a._parentable),
        _ptr_obj::Opaque(o) => Some(&o._parentable),
        _ptr_obj::Subarray(s) => Some(&*s._parentable),
        _ptr_obj::EndMarker(e) => Some(&*e._parentable),
        // `_arraylenref` is a `_parentable` upstream (lltype.py) but is
        // modeled here without one: it never sets a parent link and is never
        // itself a parent of inlined children, so the parent-chain walk would
        // see an empty link either way. STRUCTURAL DEVIATION (not functional):
        // converging means adding an empty `_parentable` and reconciling the
        // explicit `_ptr_obj::ArrayLenRef` arms that currently assume it is not
        // a parentable (`arraylenref has no container parent`). A tagged-int
        // payload (`cast_int_to_ptr`) is a bare integer, not a container.
        _ptr_obj::Func(_) | _ptr_obj::Wref(_) | _ptr_obj::ArrayLenRef(_) | _ptr_obj::IntCast(_) => {
            None
        }
    }
}

/// `getattr(parent, name)` reinterpreted as the inlined sub-container object
/// it holds — the first-field identity probe for the `_cast_to` up-cast
/// (`getattr(parent, PARENTTYPE._names[0]) != struc`, lltype.py). `None`
/// when `parent` is not a `_struct` or the field is not an inlined container.
fn first_field_container(parent: &_ptr_obj, name: &str) -> Option<_ptr_obj> {
    let _ptr_obj::Struct(s) = parent else {
        return None;
    };
    match s._getattr(name)? {
        LowLevelValue::Struct(c) => Some(_ptr_obj::Struct(*c)),
        LowLevelValue::Array(c) => Some(_ptr_obj::Array(*c)),
        LowLevelValue::Opaque(c) => Some(_ptr_obj::Opaque(*c)),
        _ => None,
    }
}

/// `top_container(container)` (lltype.py): walk the parent chain to
/// the outermost container that owns the shared allocation. A container with
/// no `_parentable` or no parent link is already the top. Uses `check=false`
/// on the walk — the decision it feeds (`_subarray`'s raw keepalive) only
/// reads the top container's `_gckind`, so a freed-storage check here would be
/// spurious.
pub fn top_container(container: &_ptr_obj) -> _ptr_obj {
    let mut top = container.clone();
    loop {
        let parent = parentable_of_obj(&top).and_then(|p| p.parentstructure(false));
        match parent {
            Some(p) => top = p,
            None => break,
        }
    }
    top
}

/// `parentlink(container)` (lltype.py): the parent container object
/// and the index of `container` within it, or `(None, None)` when `container`
/// has no parent (top-level allocation) or is not a `_parentable`.
pub fn parentlink(container: &_ptr_obj) -> (Option<_ptr_obj>, Option<ParentIndex>) {
    let Some(p) = parentable_of_obj(container) else {
        return (None, None);
    };
    match p.parentstructure(true) {
        Some(parent) => (Some(parent), p.parent_index()),
        None => (None, None),
    }
}

/// The shared `_parentable` state of a `LowLevelValue` that is itself an
/// inlined container (`_struct`/`_array`/`_opaque`), or `None` for a
/// primitive / pointer field. Used to wire `_setparentstructure` at
/// materialization (`_struct`/`_array.__init__` pass `parent=self`).
fn parentable_of(value: &LowLevelValue) -> Option<&Parentable> {
    match value {
        LowLevelValue::Struct(s) => Some(&s._parentable),
        LowLevelValue::Array(a) => Some(&a._parentable),
        LowLevelValue::Opaque(o) => Some(&o._parentable),
        _ => None,
    }
}

/// `typeOf(value)` for a `LowLevelValue` that is itself an inlined container
/// (`_struct`/`_array`/`_opaque`) — the child's `_TYPE`, passed to
/// `_setparentstructure` so the `_keepparent` `_gckind` comparison can run.
/// `None` for a primitive / pointer field.
fn child_container_type(value: &LowLevelValue) -> Option<LowLevelType> {
    match value {
        LowLevelValue::Struct(s) => Some(LowLevelType::Struct(Box::new(s.TYPE.clone()))),
        LowLevelValue::Array(a) => Some(match &a.TYPE {
            ArrayContainer::Array(t) => LowLevelType::Array(Box::new(t.clone())),
            ArrayContainer::FixedSizeArray(t) => LowLevelType::FixedSizeArray(Box::new(t.clone())),
        }),
        LowLevelValue::Opaque(o) => Some(LowLevelType::Opaque(Box::new(o.TYPE.clone()))),
        _ => None,
    }
}

/// `_container._as_ptr(self)` (lltype.py): `_ptr(Ptr(self._TYPE),
/// self, solid)` for an inlined container value (`_struct`/`_array`/
/// `_opaque`). `None` for a non-container `LowLevelValue`. Used by
/// `AddressOffset::ref` to hand back a pointer to a navigated sub-container.
pub(crate) fn container_value_as_ptr(value: &LowLevelValue, solid: bool) -> Option<_ptr> {
    let (obj, target) = match value {
        LowLevelValue::Struct(s) => (
            _ptr_obj::Struct((**s).clone()),
            PtrTarget::Struct(s.TYPE.clone()),
        ),
        LowLevelValue::Array(a) => {
            let target = match &a.TYPE {
                ArrayContainer::Array(t) => PtrTarget::Array(t.clone()),
                ArrayContainer::FixedSizeArray(t) => PtrTarget::FixedSizeArray(t.clone()),
            };
            (_ptr_obj::Array((**a).clone()), target)
        }
        LowLevelValue::Opaque(o) => (
            _ptr_obj::Opaque((**o).clone()),
            PtrTarget::Opaque(o.TYPE.clone()),
        ),
        _ => return None,
    };
    Some(_ptr::new_with_solid(
        Ptr { TO: target },
        Ok(Some(obj)),
        solid,
    ))
}

/// `_struct.__init__` (lltype.py) — build a `_struct` whose
/// inlined sub-container fields link back to it as their `_wrparent`
/// (`parent=self, parentindex=fld`). Single construction path so both
/// `_container_example` and `malloc` wire parent links identically.
fn build_struct(type_: Struct, fields: Vec<(String, LowLevelValue)>) -> _struct {
    // Upstream `_setparentstructure(parent=self, ...)` passes the parent
    // container object, which exists (has identity) before `__init__` runs.
    // Build the `_struct` first, then link each inlined sub-container to it;
    // the clone shares `_fields`/`_parentable` so it is an identity alias.
    let s = _struct::from_parts(type_, fields);
    let parent_obj = _ptr_obj::Struct(s.clone());
    let parent_type = LowLevelType::Struct(Box::new(s.TYPE.clone()));
    for (name, value) in s._fields.lock().iter() {
        if let Some(child) = parentable_of(value) {
            let child_type = child_container_type(value)
                .expect("a `_parentable` field is always a container value");
            child.set_parent_structure(
                &child_type,
                parent_obj.clone(),
                parent_type.clone(),
                ParentIndex::Field(name.clone()),
            );
        }
    }
    s
}

/// `_array.__init__` (lltype.py) and `_fixedsizearray.__init__`
/// (lltype.py) — build an `_array` whose inlined item slots link
/// back to it as their `_wrparent`. A plain `_array` passes the integer
/// item index as `parentindex` (`parent=self, parentindex=j`,
/// lltype.py); a `_fixedsizearray` passes the synthesized field name
/// `"item%d" % j` (`parentindex=fld`, lltype.py).
fn build_array(type_: ArrayContainer, items: Vec<LowLevelValue>) -> _array {
    // Same construction order as `build_struct`: materialize the `_array`,
    // then link each inlined item to the (identity-aliased) parent object.
    let a = _array::from_parts(type_, items);
    let parent_obj = _ptr_obj::Array(a.clone());
    let parent_type = match &a.TYPE {
        ArrayContainer::Array(arr) => LowLevelType::Array(Box::new(arr.clone())),
        ArrayContainer::FixedSizeArray(arr) => LowLevelType::FixedSizeArray(Box::new(arr.clone())),
    };
    let fixedsize = matches!(&a.TYPE, ArrayContainer::FixedSizeArray(_));
    for (j, item) in a.items.lock().iter().enumerate() {
        if let Some(child) = parentable_of(item) {
            let child_type = child_container_type(item)
                .expect("a `_parentable` item is always a container value");
            let parent_index = if fixedsize {
                ParentIndex::Field(format!("item{j}"))
            } else {
                ParentIndex::Item(j as i64)
            };
            child.set_parent_structure(
                &child_type,
                parent_obj.clone(),
                parent_type.clone(),
                parent_index,
            );
        }
    }
    a
}

impl _subarray {
    /// `_parentable._was_freed` (lltype.py) — delegates to the
    /// shared `_parentable`, which walks the parent chain.
    pub fn _was_freed(&self) -> bool {
        self._parentable.was_freed()
    }

    /// `_subarray.getlength` (lltype.py) — the `_TYPE` is always a
    /// `FixedSizeArray`, so its declared `.length` (1).
    pub fn getlength(&self) -> usize {
        self.TYPE.length
    }

    /// `_subarray.getbounds` (lltype.py): a struct-field subarray
    /// spans `(0, 1)`; an array subarray shifts the parent array's bounds by
    /// `-baseoffset`, so a backward `direct_ptradd` base makes index 0 fall
    /// below `start` (the `_ptr.__getitem__` bounds check then rejects it as an
    /// out-of-bounds read instead of reaching a negative parent index).
    pub fn getbounds(&self) -> (i64, i64) {
        match self
            ._parentable
            .parent_index()
            .expect("a _subarray always has a parent index")
        {
            ParentIndex::Field(_) => (0, 1),
            ParentIndex::Item(base) => {
                let parent = self
                    ._parentable
                    .parentstructure(true)
                    .expect("a _subarray always has a parent");
                let _ptr_obj::Array(a) = parent else {
                    panic!("_subarray over an array item expects an Array parent");
                };
                let (start, stop) = a.getbounds();
                (start as i64 - base, stop as i64 - base)
            }
        }
    }

    /// `_subarray.getitem` (lltype.py): redirect into the parent at
    /// `base + index`. A struct-field subarray (`_parent_index` is a name)
    /// reads the field (index must be 0); an array subarray (`_parent_index`
    /// is the integer base) reads the parent item at `base + index`.
    pub fn getitem(&self, index: usize) -> Option<LowLevelValue> {
        let parent = self
            ._parentable
            .parentstructure(true)
            .expect("a _subarray always has a parent");
        match self
            ._parentable
            .parent_index()
            .expect("a _subarray always has a parent index")
        {
            ParentIndex::Field(fieldname) => {
                assert_eq!(index, 0);
                let _ptr_obj::Struct(s) = parent else {
                    panic!("_subarray over a struct field expects a Struct parent");
                };
                s._getattr(&fieldname)
            }
            ParentIndex::Item(base) => {
                let _ptr_obj::Array(a) = parent else {
                    panic!("_subarray over an array item expects an Array parent");
                };
                // `self._parentstructure().getitem(baseoffset + index)`
                // (lltype.py). A backward `base` makes the effective index
                // negative; `_ptr.__getitem__` already rejects such an index via
                // `getbounds()` (`_subarray.getbounds` shifts the parent bounds
                // by `-base`), so the `try_from` is a defensive backstop for a
                // direct container access that bypassed the pointer check.
                let effective = base + index as i64;
                let effective =
                    usize::try_from(effective).expect("out-of-bounds read of a backward _subarray");
                a.getitem(effective)
            }
        }
    }

    /// `_subarray.setitem` (lltype.py): write back into the parent
    /// at `base + index`. A struct-field subarray (`_parent_index` a name)
    /// sets the field (index must be 0); an array subarray (`_parent_index`
    /// the integer base) sets the parent item at `base + index`.
    pub fn setitem(&self, index: usize, value: LowLevelValue) -> bool {
        let parent = self
            ._parentable
            .parentstructure(true)
            .expect("a _subarray always has a parent");
        match self
            ._parentable
            .parent_index()
            .expect("a _subarray always has a parent index")
        {
            ParentIndex::Field(fieldname) => {
                assert_eq!(index, 0);
                let _ptr_obj::Struct(s) = parent else {
                    panic!("_subarray over a struct field expects a Struct parent");
                };
                s._setattr(&fieldname, value)
            }
            ParentIndex::Item(base) => {
                let _ptr_obj::Array(a) = parent else {
                    panic!("_subarray over an array item expects an Array parent");
                };
                // As in `getitem`, `_ptr.__setitem__`'s `getbounds()` check
                // rejects a negative effective index first; the `try_from` is a
                // backstop for direct container access.
                let effective = base + index as i64;
                let effective = usize::try_from(effective)
                    .expect("out-of-bounds write of a backward _subarray");
                a.setitem(effective, value)
            }
        }
    }

    /// `_subarray._makeptr(parent, baseoffset_or_fieldname, solid)`
    /// (lltype.py). `ITEMTYPE` is the parent struct field type
    /// (`direct_fieldptr`, `_parent_index` a name) or the parent array `OF`
    /// (`direct_arrayitems`/`direct_ptradd`, `_parent_index` an integer); the
    /// subarray `_TYPE` is `FixedSizeArray(ITEMTYPE, 1)`. The `_subarray` is
    /// memoized per `(parent, baseoffset_or_fieldname)` through [`SUBARRAY_CACHE`]
    /// so re-deriving the same interior pointer reuses one container identity.
    fn _makeptr(parent: &_ptr_obj, key: ParentIndex, solid: bool) -> Result<_ptr, String> {
        // `_subarray._cache` is a `WeakKeyDictionary` keyed by the parent
        // object (lltype.py). The keyed `Arc::as_ptr` address can be
        // reused after a parent is dropped, so a cache hit is accepted only
        // when the stored entry's weak `_wrparent` still upgrades to *this*
        // parent (`parent_is`); a stale hit on a reused address recomputes,
        // mirroring upstream's `except RuntimeError: _cleanup_cache(); retry`
        // (lltype.py).
        let parent_id = match parent {
            _ptr_obj::Struct(s) => s.identity(),
            _ptr_obj::Array(a) => a.identity(),
            _ => return Err("_subarray._makeptr: parent must be a struct or array".into()),
        };
        let cache_key = (parent_id, key.clone());
        let cached = SUBARRAY_CACHE
            .with(|c| c.borrow().get(&cache_key).cloned())
            .filter(|sub| sub._parentable.parent_is(parent));
        let sub = match cached {
            Some(sub) => sub,
            None => {
                let item_type = match (&key, parent) {
                    (ParentIndex::Field(name), _ptr_obj::Struct(s)) => s
                        .TYPE
                        ._flds
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("{} has no field {name:?}", s.TYPE._name))?,
                    (ParentIndex::Item(_), _ptr_obj::Array(a)) => match &a.TYPE {
                        ArrayContainer::Array(t) => t.OF.clone(),
                        ArrayContainer::FixedSizeArray(t) => t.OF.clone(),
                    },
                    _ => return Err("_subarray._makeptr: parent/index kind mismatch".into()),
                };
                let arraytype = FixedSizeArray::new(item_type, 1);
                let sub = _subarray {
                    _identity: fresh_low_level_container_identity(),
                    TYPE: arraytype,
                    _parentable: Parentable::new(),
                };
                let child_type = LowLevelType::FixedSizeArray(Box::new(sub.TYPE.clone()));
                sub._parentable.set_parent_structure(
                    &child_type,
                    parent.clone(),
                    parent._container_type(),
                    key,
                );
                // `_subarray.__init__` (lltype.py): also keep the
                // parent strongly when the top container is `raw` — the
                // subarray shares its allocation, so it must keep it alive;
                // inside a gc object it is someone else's job. The `ll2ctypes`
                // `_storage.contents` disjunct is dead in pyre.
                if top_container(parent)._container_type()._gckind() == GcKind::Raw {
                    sub._parentable.force_keepparent();
                }
                SUBARRAY_CACHE.with(|c| c.borrow_mut().insert(cache_key, sub.clone()));
                sub
            }
        };
        Ok(_ptr::new_with_solid(
            Ptr {
                TO: PtrTarget::FixedSizeArray(sub.TYPE.clone()),
            },
            Ok(Some(_ptr_obj::Subarray(sub))),
            solid,
        ))
    }
}

/// `class _arraylenref(_parentable)` (lltype.py) — a pseudo-reference
/// to the length field of an array, used only internally by `llmemory` to
/// implement `ArrayLengthOffset`. Its `_TYPE` is always `FixedSizeArray(Signed,
/// 1)` and `getitem(0)` reads the parent array's current length.
///
/// As with [`_subarray`] the weak `_cache` (array → `_arraylenref`) is
/// modeled by the strong [`ARRAYLENREF_CACHE`] keyed on the parent array's
/// container identity, so re-deriving a length pointer reuses one identity.
/// `setitem` (in-place shrink, lltype.py) is ported in full.
#[derive(Clone, Debug)]
pub struct _arraylenref {
    pub _identity: usize,
    /// `self.array` (lltype.py:2072) — the parent array container whose length
    /// this pseudo-reference projects.
    pub array: Box<_array>,
}

impl PartialEq for _arraylenref {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _arraylenref {}

impl Hash for _arraylenref {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl _arraylenref {
    /// `_arraylenref.getlength` (lltype.py) — always 1.
    pub fn getlength(&self) -> usize {
        1
    }

    /// `_arraylenref.getbounds` (lltype.py) — always `(0, 1)`.
    pub fn getbounds(&self) -> (i64, i64) {
        (0, 1)
    }

    /// `_arraylenref.getitem` (lltype.py): `assert index == 0;
    /// return self.array.getlength()`.
    pub fn getitem(&self, index: usize) -> Option<LowLevelValue> {
        assert_eq!(index, 0);
        Some(LowLevelValue::Signed(self.array.getlength() as i64))
    }

    /// `_arraylenref.setitem` (lltype.py): `assert index == 0`;
    /// writing a length smaller than the array's current length shrinks it in
    /// place, an equal length is a no-op, and growing is rejected.
    pub fn setitem(&self, index: usize, value: LowLevelValue) -> bool {
        assert_eq!(index, 0);
        let LowLevelValue::Signed(newlen) = value else {
            panic!("_arraylenref.setitem expects a Signed length, got {value:?}");
        };
        let cur = self.array.getlength() as i64;
        if newlen != cur {
            if newlen > cur {
                panic!("can't grow an array in-place");
            }
            self.array.shrinklength(newlen as usize);
        }
        true
    }

    /// `_arraylenref._makeptr(array, solid)` (lltype.py) —
    /// `_ptr(Ptr(FixedSizeArray(Signed, 1)), lenref, solid)`. The `lenref` is
    /// memoized per array container identity through [`ARRAYLENREF_CACHE`], so
    /// re-deriving the length pointer of one array reuses its identity.
    pub(crate) fn _makeptr(array: Box<_array>, solid: bool) -> _ptr {
        let arraytype = FixedSizeArray::new(LowLevelType::Signed, 1);
        // The cache value (`_arraylenref`) holds the parent array strongly
        // through its `array: Box<_array>` field, so the parent `Arc` stays
        // alive as long as this key lives — its `Arc::as_ptr` address cannot
        // be reused while it remains a key, making the stored identity safe.
        let array_id = array.identity();
        let cached = ARRAYLENREF_CACHE.with(|c| c.borrow().get(&array_id).cloned());
        let lenref = cached.unwrap_or_else(|| {
            let lenref = _arraylenref {
                _identity: fresh_low_level_container_identity(),
                array,
            };
            ARRAYLENREF_CACHE.with(|c| c.borrow_mut().insert(array_id, lenref.clone()));
            lenref
        });
        _ptr::new_with_solid(
            Ptr {
                TO: PtrTarget::FixedSizeArray(arraytype),
            },
            Ok(Some(_ptr_obj::ArrayLenRef(lenref))),
            solid,
        )
    }
}

/// `class _endmarker_struct(lltype._struct)` (lltype.py) — the sentinel
/// `_struct` `ItemOffset.ref` returns for a reference *exactly* to the end of an
/// array. Its `_TYPE` is the array item struct type `A`; it carries a parent
/// link back to the array at the end index but exposes no fields — every field
/// access raises (`__getattr__`, lltype.py).
///
/// The weak `_end_markers` memo (array → endmarker, llmemory.py) is not
/// modeled (the same value-model adaptation as [`_subarray`] /
/// [`_arraylenref`]); the address-fold consumer threads a single derived
/// pointer and never re-derives, so the cache's `is`-identity stability is
/// unexercised.
#[derive(Clone, Debug)]
pub struct _endmarker {
    pub _identity: usize,
    pub TYPE: Struct,
    /// Shared `_parentable` state — `_endmarker_struct` is a `_struct`, hence a
    /// `_parentable`, linked to the parent array at the end index.
    _parentable: Arc<Parentable>,
}

impl PartialEq for _endmarker {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _endmarker {}

impl Hash for _endmarker {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl _endmarker {
    /// `_parentable._was_freed` — delegates to the shared `_parentable`.
    pub fn _was_freed(&self) -> bool {
        self._parentable.was_freed()
    }

    /// True when this end marker's weak `_wrparent` still upgrades to `parent`
    /// — used to reject a stale `_end_markers` cache hit on a reused
    /// `Arc::as_ptr` address (see [`Parentable::parent_is`]).
    pub(crate) fn parent_is(&self, parent: &_ptr_obj) -> bool {
        self._parentable.parent_is(parent)
    }

    /// `_endmarker_struct(A, parent=array, parentindex=index)`
    /// (llmemory.py) — the sentinel `_struct` linked to the parent array
    /// at the end index. `A` is the array item struct type. Memoization per
    /// array (`_end_markers`, llmemory.py) is the caller's responsibility,
    /// matching upstream where the cache lives in `llmemory`.
    pub(crate) fn new(item_type: &Struct, parent: &_ptr_obj, index: usize) -> _endmarker {
        let endmarker = _endmarker {
            _identity: fresh_low_level_container_identity(),
            TYPE: item_type.clone(),
            _parentable: Parentable::new(),
        };
        let child_type = LowLevelType::Struct(Box::new(endmarker.TYPE.clone()));
        endmarker._parentable.set_parent_structure(
            &child_type,
            parent.clone(),
            parent._container_type(),
            ParentIndex::Item(index as i64),
        );
        endmarker
    }

    /// `_endmarker_struct(...)._as_ptr()` (llmemory.py) — a `Ptr(A)` to
    /// this sentinel.
    pub(crate) fn _as_ptr(&self, solid: bool) -> _ptr {
        _ptr::new_with_solid(
            Ptr {
                TO: PtrTarget::Struct(self.TYPE.clone()),
            },
            Ok(Some(_ptr_obj::EndMarker(self.clone()))),
            solid,
        )
    }
}

/// `direct_fieldptr(structptr, fieldname)` (lltype.py) — a
/// `Ptr(FixedSizeArray(FIELD, 1))` interior pointer to `structptr.fieldname`,
/// usable in `getarrayitem(0)`/`setarrayitem(0)`.
pub fn direct_fieldptr(structptr: &_ptr, fieldname: &str) -> Result<_ptr, String> {
    let curtype: LowLevelType = structptr._TYPE.TO.clone().into();
    let LowLevelType::Struct(st) = &curtype else {
        return Err("direct_fieldptr: not a struct".into());
    };
    if st._flds.get(fieldname).is_none() {
        return Err(format!("{} has no field {fieldname:?}", st._name));
    }
    if !structptr.nonzero() {
        return Err("direct_fieldptr: NULL argument".into());
    }
    let obj = structptr
        ._obj()
        .map_err(|_| "direct_fieldptr: delayed pointer".to_string())?;
    _subarray::_makeptr(
        &obj,
        ParentIndex::Field(fieldname.to_string()),
        structptr._solid,
    )
}

/// `direct_arrayitems(arrayptr)` (lltype.py) — a
/// `Ptr(FixedSizeArray(ITEM, 1))` interior pointer to the array's first item,
/// usable in `getarrayitem(n)`/`direct_ptradd(n)`.
pub fn direct_arrayitems(arrayptr: &_ptr) -> Result<_ptr, String> {
    let curtype: LowLevelType = arrayptr._TYPE.TO.clone().into();
    if !matches!(
        curtype,
        LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_)
    ) {
        return Err("direct_arrayitems: not an array".into());
    }
    if !arrayptr.nonzero() {
        return Err("direct_arrayitems: NULL argument".into());
    }
    let obj = arrayptr
        ._obj()
        .map_err(|_| "direct_arrayitems: delayed pointer".to_string())?;
    _subarray::_makeptr(&obj, ParentIndex::Item(0), arrayptr._solid)
}

/// `direct_ptradd(ptr, n)` (lltype.py) — shift an interior pointer
/// built by `direct_arrayitems()` forward/backward by `n` items, re-resolving
/// it as `parent[base + n]`. The non-`_subarray` case (a bare nolength C-like
/// array) delegates upstream to `rffi.ptradd`, which is
/// `ll2ctypes.force_ptradd` (rffi.py, ll2ctypes.py) — runtime ctypes
/// pointer arithmetic over a real `_rctypes` object. ll2ctypes is dead at fold
/// time (no live ctypes object backs a fold constant), so there is no
/// constant-fold value to produce and the fold declines, exactly as upstream
/// would have nothing to compute without the runtime pointer. A backward shift
/// can make `base + n` negative — upstream builds the `_subarray` regardless
/// and only errors on an out-of-bounds dereference, so the signed index is
/// threaded straight through.
pub fn direct_ptradd(ptr: &_ptr, n: i64) -> Result<_ptr, String> {
    if !ptr.nonzero() {
        return Err("direct_ptradd: NULL argument".into());
    }
    let obj = ptr
        ._obj()
        .map_err(|_| "direct_ptradd: delayed pointer".to_string())?;
    if !matches!(&obj, _ptr_obj::Subarray(_)) {
        return Err(
            "direct_ptradd: non-_subarray delegates to rffi.ptradd (ll2ctypes runtime), \
             which has no constant-fold value"
                .into(),
        );
    }
    let (parent, base) = parentlink(&obj);
    let parent = parent.ok_or_else(|| "direct_ptradd: subarray has no parent".to_string())?;
    let base = match base {
        Some(ParentIndex::Item(j)) => j,
        Some(ParentIndex::Field(_)) => {
            return Err("direct_ptradd: cannot shift a struct-field subarray".into());
        }
        None => return Err("direct_ptradd: subarray has no parent index".into()),
    };
    _subarray::_makeptr(&parent, ParentIndex::Item(base + n), ptr._solid)
}

/// RPython `fixup_solid(p)` (`constfold.py`):
///
/// ```python
/// def fixup_solid(p):
///     container = p._obj
///     assert isinstance(container, lltype._parentable)
///     container._keepparent = container._parentstructure()
///     return container._as_ptr()
/// ```
///
/// Pins the parent of an inlined sub-pointer so the parent keeps the inlined
/// part alive. `_setparentstructure` stores the parent as a `Weak`
/// (`ParentLink::wrparent`, lltype.py) and keeps it strong only when the
/// `_gckind` / first-field rule applies (`ParentLink::keepparent`,
/// lltype.py). `fixup_solid` forces `keepparent` to the parent
/// regardless of that rule (`Parentable::force_keepparent`), so an interior
/// part that is *not* the parent's first field — a non-first struct field, an
/// array item, a subarray — still keeps its (otherwise weakly-held) parent
/// alive. The body validates the container is a
/// `_parentable` and returns `container._as_ptr()`: a fresh solid pointer to
/// the container (`_ptr(Ptr(typeOf(container)), container, solid=True)`,
/// `_container._as_ptr`, lltype.py). `_ptr` equality is by container
/// identity, so the result still equals the input pointer.
pub fn fixup_solid(ptr: &_ptr) -> Result<_ptr, String> {
    let container = ptr
        ._obj()
        .map_err(|_| "fixup_solid: delayed pointer".to_string())?;
    let Some(parentable) = parentable_of_obj(&container) else {
        return Err(format!("fixup_solid: {container:?} is not a _parentable"));
    };
    // `container._keepparent = container._parentstructure()` (constfold.py):
    // pin the parent strongly so the inlined part keeps the whole object alive,
    // independent of the first-field/`_gckind` rule. With the weak `_wrparent`
    // (lltype.py) this is load-bearing — an interior part returned as a
    // constant would otherwise let its parent drop.
    parentable.force_keepparent();
    Ok(_ptr::new_with_solid(
        Ptr::from_container_type(container._container_type())?,
        Ok(Some(container)),
        true,
    ))
}

impl PartialEq for _struct {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for _struct {}

impl Hash for _struct {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

impl PartialEq for _array {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for _array {}

impl Hash for _array {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

/// `_subarray` (lltype.py) — a `_parentable` view over one slot of
/// a parent array or struct field, built only by [`direct_fieldptr`] /
/// [`direct_arrayitems`]. Its `_TYPE` is always `FixedSizeArray(ITEMTYPE, 1)`,
/// and its `_parent_index` is the parent field name (`direct_fieldptr`) or the
/// item base offset (`direct_arrayitems`); `getitem(i)` redirects into the
/// parent at `base + i`.
///
/// Upstream's `_subarray._cache` weak memo (lltype.py) is not
/// modelled as a weak map. The `_subarray`'s `_setparentstructure` ports
/// `_keepparent` (lltype.py) via the shared `_parentable`. The cache
/// gives `is`-identity stability to independently re-derived interior
/// pointers; without it two `direct_fieldptr(p, "f")` calls yield subarrays
/// with distinct `_identity`. The sole consumer here (`AddressOffset::ref` →
/// the address-fold table) threads a single derived pointer and never
/// re-derives, so the stability difference is unexercised. A faithful
/// weak-keyed cache would key on `Arc::as_ptr` (as `parent_id` already does)
/// but drop the entry when the parent is reclaimed — requiring a weak-map
/// structure not yet implemented.
#[derive(Clone, Debug)]
pub struct _subarray {
    pub _identity: usize,
    pub TYPE: FixedSizeArray,
    /// Shared `_parentable` state (lltype.py); `_subarray` is a
    /// `_parentable` (lltype.py).
    _parentable: Arc<Parentable>,
}

impl PartialEq for _subarray {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _subarray {}

impl Hash for _subarray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl PartialEq for _opaque {
    /// `_opaque.__eq__` (lltype.py). A `'hidden'` opaque carries
    /// a `container`; two such opaques are equal iff their normalized
    /// containers are equal (so two casts of the same allocation match),
    /// regardless of `_identity`. Any other opaque compares by identity
    /// (`self is other`).
    fn eq(&self, other: &Self) -> bool {
        match (&self.container, &other.container) {
            (Some(_), Some(_)) => self._normalizedcontainer() == other._normalizedcontainer(),
            _ => self.identity() == other.identity(),
        }
    }
}

impl Eq for _opaque {}

impl Hash for _opaque {
    /// `_opaque.__hash__` (lltype.py) — hash by the normalized
    /// container when present, else by identity (`_parentable.__hash__`).
    /// Consistent with [`PartialEq`].
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.container {
            Some(c) => c._normalizedcontainer().hash(state),
            None => self.identity().hash(state),
        }
    }
}

impl _opaque {
    /// `id(self)` / `is` container identity (lltype.py _identityhash) — the
    /// `Arc` allocation address. Stable for the lifetime of the `Arc`;
    /// every value-clone shares the same `Arc`, hence the same identity.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }

    /// `_parentable._free` (lltype.py) — `_opaque` is a `_parentable`;
    /// delegates to the shared `_parentable` state.
    pub fn _free(&self) {
        self._parentable.free();
    }

    /// `_parentable._was_freed` (lltype.py) — delegates to the
    /// shared `_parentable`, which walks the `_wrparent` chain.
    pub fn _was_freed(&self) -> bool {
        self._parentable.was_freed()
    }

    /// `_opaque._normalizedcontainer` (lltype.py): a `'hidden'`
    /// opaque normalizes to its stowed container's normal form (the int /
    /// `_carry_around_for_tests` short-circuits are not modeled in pyre).
    /// A container-less opaque would fall to `_parentable._normalizedcontainer`
    /// (the first-inlined-field walk), which needs the deferred
    /// object-returning `_parentstructure()`; it is already normal here.
    fn _normalizedcontainer(&self) -> _ptr_obj {
        match &self.container {
            Some(c) => c._normalizedcontainer(),
            None => _ptr_obj::Opaque(self.clone()),
        }
    }
}

/// llmemory.py — `class _wref(lltype._container)`. A gc-managed
/// container holding a weak reference to another gc container;
/// `_TYPE = WeakRef`, `_gckind = 'gc'`.
///
/// Upstream stores `weakref.ref(normalizeptr(ptarget)._obj)` (or
/// `lambda: None` for a dead wref) and `_dereference` returns
/// `obj._as_ptr()` unless `obj is None or obj._was_freed()` (llmemory.py:
/// 865-879). `_obref` reuses the same [`PtrObj`] classification as
/// `_ptr._obj0`: `Weak` for an `Arc<XCore>` container (`_struct`/`_array`/
/// `_opaque`, where `std::sync::Weak` is the direct analog of
/// `weakref.ref`), `Null` for the `lambda: None` dead wref, and `Delayed`
/// when the target was delayed at construction (upstream raises it out of
/// `normalizeptr(ptarget)._obj`).
///
/// `weakref.ref` upstream accepts *any* container; the not-yet-`Arc`
/// interior containers (`_subarray`/`_arraylenref`/`_endmarker`/`_wref`)
/// have no weak form here, so they are held `Strong` instead of rejected —
/// the same fallback `_ptr._setobj` uses, observationally identical under
/// pyre's never-free model (a strongly-held referent never reports
/// `_was_freed`, so `_dereference` returns it just like a live weakref).
///
/// The `Weak` referent stays alive only through its other strong holders,
/// exactly like a Python weakref. Translation-time gc containers are held
/// strongly by the database / prebuilt-constant tables for the whole
/// translation, so the upgrade resolves alive in practice; should a referent
/// ever be reclaimed, `_dereference` returns null — the faithful weakref
/// result. `_was_freed()` (the second null path) only fires once `_free`
/// has run on the container; pyre has no arena that nulls a central storage
/// slot, so it stays false unless an `llinterp`-style `_free` caller runs.
#[derive(Clone, Debug)]
pub struct _wref {
    pub _identity: usize,
    // `Box` breaks the `_wref` → `PtrObj::Strong(_ptr_obj)` →
    // `_ptr_obj::Wref(_wref)` size cycle (a wref can name a wref).
    pub _obref: Box<PtrObj>,
}

impl PartialEq for _wref {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _wref {}

impl Hash for _wref {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl _wref {
    /// llmemory.py `_wref.__init__(self, ptarget)`:
    ///
    /// ```python
    /// if ptarget is None:
    ///     self._obref = lambda: None
    /// else:
    ///     obj = lltype.normalizeptr(ptarget)._obj
    ///     self._obref = weakref.ref(obj)
    /// ```
    ///
    /// `obj = normalizeptr(ptarget)._obj` resolves the referent container;
    /// `weakref.ref(obj)` keeps a weak handle. A `None` target is the
    /// `lambda: None` dead wref (`PtrObj::Null`); a delayed target carries
    /// its `DelayedPointer` through (upstream raises it).
    pub fn new(ptarget: Option<&_ptr>) -> Self {
        let obref = match ptarget {
            None => PtrObj::Null,
            Some(p) => match p._getobj(true) {
                // A target delayed at construction propagates its
                // `DelayedPointer` (upstream raises it out of `._obj`).
                Err(delayed) => PtrObj::Delayed(delayed),
                // A null target has no `_obj`; treat as the dead wref.
                Ok(None) => PtrObj::Null,
                Ok(Some(obj)) => {
                    // `obj = normalizeptr(ptarget)._obj` (llmemory.py __init__):
                    // un-hide an opaque and promote to the normalized
                    // container before `weakref.ref(obj)`. `weakref.ref`
                    // accepts any container; an `Arc<XCore>` referent
                    // downgrades to a real `Weak`, while a not-yet-`Arc`
                    // interior container (`_subarray`/`_arraylenref`/
                    // `_endmarker`/`_wref`) has no weak form and is held
                    // `Strong` — the same fallback `_setobj` uses, never-free
                    // so observationally a live weakref.
                    let container = obj._normalizedcontainer();
                    WeakContainer::downgrade(&container)
                        .map(PtrObj::Weak)
                        .unwrap_or(PtrObj::Strong(container))
                }
            },
        };
        _wref {
            _identity: fresh_low_level_container_identity(),
            _obref: Box::new(obref),
        }
    }

    /// llmemory.py `_wref._dereference(self)`:
    ///
    /// ```python
    /// def _dereference(self):
    ///     obj = self._obref()
    ///     if obj is None or obj._was_freed():
    ///         return None
    ///     else:
    ///         return obj._as_ptr()
    /// ```
    ///
    /// `obj = self._obref()` upgrades the weakref; a collected referent
    /// (`None`) and the `lambda: None` dead wref both yield null, as does a
    /// container reporting `_was_freed()` (false unless a `_free` caller has
    /// run, e.g. from `llinterp`). A `Strong` fallback referent (a non-`Arc`
    /// interior container) is never collected, so it dereferences unless it
    /// reports `_was_freed()`. A target that was delayed at construction
    /// propagates `DelayedPointer`, exactly as upstream lets it escape.
    pub fn _dereference(&self) -> Result<Option<_ptr>, DelayedPointer> {
        match &*self._obref {
            PtrObj::Delayed(_) => Err(DelayedPointer),
            PtrObj::Null => Ok(None),
            PtrObj::Weak(wref) => match wref.upgrade() {
                None => Ok(None),
                Some(obj) if obj._was_freed() => Ok(None),
                Some(obj) => Ok(Some(obj._as_ptr(true))),
            },
            PtrObj::Strong(obj) if obj._was_freed() => Ok(None),
            PtrObj::Strong(obj) => Ok(Some(obj._as_ptr(true))),
        }
    }

    /// `_container._as_ptr(self)` — `_ptr(Ptr(self._TYPE), self, solid=True)`
    /// with `self._TYPE == WeakRef`.
    pub fn _as_ptr(self) -> _ptr {
        _ptr::new_with_solid(
            Ptr {
                TO: PtrTarget::Opaque(OpaqueType::gc("WeakRef")),
            },
            Ok(Some(_ptr_obj::Wref(self))),
            true,
        )
    }
}

impl _struct {
    /// `_parentable._free` (lltype.py) — delegates to the shared
    /// `_parentable` state.
    pub fn _free(&self) {
        self._parentable.free();
    }

    /// `_parentable._was_freed` (lltype.py) — delegates to the
    /// shared `_parentable`, which walks the `_wrparent` chain.
    pub fn _was_freed(&self) -> bool {
        self._parentable.was_freed()
    }

    /// `_struct._fields` is an `Arc<Mutex<...>>`-shared cell so the
    /// returned value must be cloned out of the lock guard. Cheap for
    /// the primitive variants of `LowLevelValue`; `Struct`/`Array`
    /// variants Clone the inner `_struct`/`_array` (which themselves
    /// share their inner `_fields`/`items` via `Arc::clone` per the
    /// shared-storage discipline).
    pub fn _getattr(&self, field_name: &str) -> Option<LowLevelValue> {
        self._fields
            .lock()
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, value)| value.clone())
    }

    pub fn _setattr(&self, field_name: &str, value: LowLevelValue) -> bool {
        let mut fields = self._fields.lock();
        let Some((_, slot)) = fields.iter_mut().find(|(name, _)| name == field_name) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Recursively descend through nested Struct fields named by `path`
    /// and `_setattr(field, value)` at the bottom. Returns `false` when
    /// any path step is not a Struct field. Used by
    /// [`_ptr::setattr_at_path`] to mutate vtable substructs in place
    /// without going through `_ptr.getattr` (which returns a detached
    /// copy for non-Gc Struct fields).
    pub fn _setattr_descending(&self, path: &[&str], field: &str, value: LowLevelValue) -> bool {
        if path.is_empty() {
            return self._setattr(field, value);
        }
        // Pull the inner Struct out of the lock guard before recursing
        // so the outer guard is released — the inner `_struct` carries
        // its own `Arc<Mutex<_fields>>` cell, and `LowLevelValue::Struct`
        // Clone shares the same Arc storage by `Arc::clone`.
        let inner = {
            let fields = self._fields.lock();
            let Some((_, slot)) = fields.iter().find(|(name, _)| name == path[0]) else {
                return false;
            };
            let LowLevelValue::Struct(inner) = slot else {
                return false;
            };
            (**inner).clone()
        };
        inner._setattr_descending(&path[1..], field, value)
    }
}

impl _array {
    /// `_parentable._free` (lltype.py) — delegates to the shared
    /// `_parentable` state.
    pub fn _free(&self) {
        self._parentable.free();
    }

    /// `_parentable._was_freed` (lltype.py) — delegates to the
    /// shared `_parentable`, which walks the `_wrparent` chain.
    pub fn _was_freed(&self) -> bool {
        self._parentable.was_freed()
    }

    pub fn getlength(&self) -> usize {
        self.items.lock().len()
    }

    pub fn getbounds(&self) -> (usize, usize) {
        (0, self.items.lock().len())
    }

    /// `_array.shrinklength` (lltype.py): `del self.items[newlength:]`.
    pub fn shrinklength(&self, newlength: usize) {
        self.items.lock().truncate(newlength);
    }

    /// Cloned out of the lock guard — same rationale as `_struct::_getattr`.
    pub fn getitem(&self, index: usize) -> Option<LowLevelValue> {
        self.items.lock().get(index).cloned()
    }

    pub fn setitem(&self, index: usize, value: LowLevelValue) -> bool {
        // `assert typeOf(value) == self._TYPE.OF` (lltype.py) — the
        // element-type invariant guards every `_array.setitem`, not just the
        // `_ptr.__setitem__` path.
        assert_eq!(
            typeOf_value(&value),
            self.OF(),
            "{:?} items: expect {:?} got {:?}",
            self.TYPE,
            self.OF(),
            typeOf_value(&value)
        );
        let mut items = self.items.lock();
        let Some(slot) = items.get_mut(index) else {
            // `_array.setitem` special case (lltype.py): writing
            // `'\x00'` to the one slot past the end of an
            // `extra_item_after_alloc` array (the trailing NUL of a STR)
            // is a no-op, not an out-of-bounds error.
            if index == items.len()
                && matches!(value, LowLevelValue::Char('\0'))
                && self.extra_item_after_alloc()
            {
                return true;
            }
            return false;
        };
        *slot = value;
        true
    }

    /// `self._TYPE.OF` — the array element type, shared by both the
    /// variable-length `Array` and the `FixedSizeArray` carriers.
    fn OF(&self) -> ConcretetypePlaceholder {
        match &self.TYPE {
            ArrayContainer::Array(t) => t.OF.clone(),
            ArrayContainer::FixedSizeArray(t) => t.OF.clone(),
        }
    }

    /// `self._TYPE._hints.get('extra_item_after_alloc')` — checked for
    /// truthiness, not mere presence (a `0`/`False` hint is falsy in
    /// upstream). Only a variable-length `Array` (e.g. `STR.chars`) carries
    /// it; a `FixedSizeArray` never does.
    fn extra_item_after_alloc(&self) -> bool {
        match &self.TYPE {
            ArrayContainer::Array(t) => match t._hints.get("extra_item_after_alloc") {
                Some(ConstValue::Int(n)) => *n != 0,
                Some(ConstValue::Bool(b)) => *b,
                _ => false,
            },
            ArrayContainer::FixedSizeArray(_) => false,
        }
    }
}

#[allow(dead_code)]
fn ptr_from_parent_type(parent_type: &LowLevelType) -> Ptr {
    match parent_type {
        LowLevelType::Struct(t) => Ptr {
            TO: PtrTarget::Struct((**t).clone()),
        },
        LowLevelType::Array(t) => Ptr {
            TO: PtrTarget::Array((**t).clone()),
        },
        LowLevelType::FixedSizeArray(t) => Ptr {
            TO: PtrTarget::FixedSizeArray((**t).clone()),
        },
        LowLevelType::Opaque(t) => Ptr {
            TO: PtrTarget::Opaque((**t).clone()),
        },
        LowLevelType::ForwardReference(t) => Ptr {
            TO: PtrTarget::ForwardReference((**t).clone()),
        },
        other => panic!("InteriorPtr parent must be a container type, got {other:?}"),
    }
}

/// RPython `_func.__eq__` (`lltype.py`) overrides `_container`'s
/// identity default with structural comparison on `self.__dict__ ==
/// other.__dict__`. Two `_func`s from separate `functionptr()` /
/// `_container_example()` calls that share `_TYPE`/`_name`/`_callable`/
/// `graph` compare equal — this is how `build_concrete_calltable`'s
/// `lookup` merges rows pointing at the same graph.
impl PartialEq for _func {
    fn eq(&self, other: &Self) -> bool {
        self.TYPE == other.TYPE
            && self._name == other._name
            && self._callable == other._callable
            && self.graph == other.graph
            && self.attrs == other.attrs
    }
}

impl Eq for _func {}

impl Hash for _func {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.TYPE.hash(state);
        self._name.hash(state);
        self._callable.hash(state);
        self.graph.hash(state);
        let mut attrs: Vec<_> = self.attrs.iter().collect();
        attrs.sort_by_key(|(a, _)| *a);
        for (key, value) in attrs {
            key.hash(state);
            value.hash(state);
        }
    }
}

#[derive(Clone, Debug)]
pub struct _ptr {
    pub _identity: u64,
    pub _TYPE: Ptr,
    pub _solid: bool,
    /// `_ptr._obj0` (lltype.py). `_setobj` (lltype.py)
    /// classifies the target as `Null`/`Strong`/`Weak`/`Delayed`; the
    /// weak form is `weakref.ref(pointing_to)` for raw, non-solid,
    /// non-func containers, so `_weak` state is `matches!(_obj0,
    /// PtrObj::Weak(_))` (lltype.py `_set_weak`).
    pub _obj0: PtrObj,
}

/// RPython `_abstract_ptr.__eq__` (`lltype.py`). Order:
/// 1. `_TYPE` mismatch → `TypeError` (Rust panic).
/// 2. Either side `DelayedPointer` → fall back to pointer identity.
/// 3. Otherwise compare `_obj0` values directly — null-null is equal,
///    null-nonnull is unequal, and non-null values use `_ptr_obj` eq
///    (which respects `_container` identity for Struct/Array/Opaque
///    and `_func.__eq__`'s structural dict comparison for Func).
impl PartialEq for _ptr {
    fn eq(&self, other: &Self) -> bool {
        let self_resolved = self._resolved_ptr();
        let other_resolved = other._resolved_ptr();
        if self_resolved._TYPE != other_resolved._TYPE {
            panic!(
                "comparing {:?} and {:?}",
                self_resolved._TYPE, other_resolved._TYPE
            );
        }
        // `self._obj == other._obj` (lltype.py): `_obj` is `_getobj(
        // check=True)`, so the freed-storage `_check()` runs as part of the
        // comparison. A `DelayedPointer` on either side falls back to `_ptr`
        // identity (`return self is other`, lltype.py).
        match (self_resolved._getobj(true), other_resolved._getobj(true)) {
            (Ok(a), Ok(b)) => a == b,
            _ => self._identity == other._identity,
        }
    }
}

impl Eq for _ptr {}

impl _ptr {
    pub fn new(_TYPE: Ptr, _obj0: Result<Option<_ptr_obj>, DelayedPointer>) -> Self {
        Self::new_with_solid(_TYPE, _obj0, false)
    }

    pub fn new_with_solid(
        _TYPE: Ptr,
        _obj0: Result<Option<_ptr_obj>, DelayedPointer>,
        _solid: bool,
    ) -> Self {
        // RPython `_ptr.__init__` (lltype.py): `_set_weak(False)`
        // then `_setobj(pointing_to, solid)`. A `DelayedPointer` target is
        // stored verbatim as the delayed `_obj0` form.
        let mut p = _ptr {
            _identity: fresh_low_level_pointer_identity(),
            _TYPE,
            _solid: false,
            _obj0: PtrObj::Null,
        };
        match _obj0 {
            Ok(pointing_to) => p._setobj(pointing_to, _solid),
            Err(delayed) => {
                p._obj0 = PtrObj::Delayed(delayed);
                p._solid = _solid;
            }
        }
        p
    }

    /// RPython `_ptr._setobj(self, pointing_to, solid=False)`
    /// (lltype.py). A `None` target is null; a solid pointer,
    /// a non-raw `_gckind`, or a `FuncType` pointer keeps a `Strong`
    /// reference (`_obj0 = pointing_to`); otherwise the raw, non-solid
    /// container is held weakly (`_obj0 = weakref.ref(pointing_to)`).
    /// Non-container raw targets (the tagged-int `IntCast` carrier, `Func`)
    /// have no weak form and stay `Strong`.
    ///
    /// PRE-EXISTING-ADAPTATION (blocker: `_subarray`/`_arraylenref` not yet
    /// `Arc`-backed). Upstream `weakref.ref(pointing_to)` weakens any
    /// `_parentable`, including the interior `_subarray`/`_arraylenref`
    /// containers; those are still `#[derive(Clone)]` value types keyed by a
    /// `_identity` counter, so `WeakContainer::downgrade` (which needs an
    /// `Arc<XCore>`) returns `None` and the `unwrap_or` keeps them `Strong`.
    /// This matches the pre-`Arc` behaviour (all `_obj0` were strong), so it
    /// is no regression, but a raw non-solid pointer to such an interior
    /// container is strong where upstream is weak. Converging needs the
    /// interior containers moved behind `Arc`, like `_struct`/`_array`/
    /// `_opaque`.
    pub fn _setobj(&mut self, pointing_to: Option<_ptr_obj>, solid: bool) {
        self._obj0 = match pointing_to {
            None => PtrObj::Null,
            Some(obj) => {
                let is_func = matches!(self._TYPE.TO, PtrTarget::Func(_));
                if solid || self._TYPE._gckind() != GcKind::Raw || is_func {
                    PtrObj::Strong(obj)
                } else {
                    // Struct/Array/Opaque downgrade to a real weak ref; the
                    // not-yet-`Arc` interior containers (`_subarray`/
                    // `_arraylenref`) and the non-container `IntCast`/`Func`
                    // carriers have no weak form and stay `Strong`.
                    WeakContainer::downgrade(&obj)
                        .map(PtrObj::Weak)
                        .unwrap_or(PtrObj::Strong(obj))
                }
            }
        };
        self._solid = solid;
    }

    pub fn _hashable_identity(&self) -> u64 {
        self._identity
    }

    pub fn _togckind(&self) -> GcKind {
        self._TYPE._gckind()
    }

    pub fn _needsgc(&self) -> bool {
        self._TYPE._needsgc()
    }

    fn _resolved_ptr(&self) -> _ptr {
        let mut current = self.clone();
        loop {
            let next = PTR_BECOME_TARGETS
                .with(|targets| targets.borrow().get(&current._identity).cloned());
            match next {
                Some(target) => current = target,
                None => return current,
            }
        }
    }

    /// RPython `_ptr._become(self, other)` (`lltype.py`).
    ///
    /// ```python
    /// def _become(self, other):
    ///     assert self._TYPE == other._TYPE
    ///     assert not self._weak
    ///     self._setobj(other._obj, other._solid)
    /// ```
    ///
    /// The Python object mutates in place. Rust `_ptr` values are copied
    /// by clone, so this records the resolved target by `_identity` and
    /// read-side operations consult the redirect table.
    pub fn _become(&self, other: &_ptr) {
        assert_eq!(
            self._TYPE, other._TYPE,
            "_ptr._become: type mismatch (self={:?}, other={:?})",
            self._TYPE, other._TYPE,
        );
        // `assert not self._weak` (lltype.py): `_setobj` updates `_weak`
        // in place on each `_become`, so the guard must read self's *current*
        // weak state. Pyre records becomes in a redirect table instead of
        // mutating `_obj0`, so the live state is the resolved target's `_obj0`;
        // a fresh `self` resolves to itself. This rejects a second `_become`
        // after self has been pointed at a weak target.
        assert!(
            !matches!(self._resolved_ptr()._obj0, PtrObj::Weak(_)),
            "_ptr._become: cannot reassign a weak pointer",
        );
        // `self._setobj(other._obj, ...)`: `other._obj` is `_getobj(check=True)`
        // (lltype.py), so a dead weak referent or freed storage raises
        // here at `_become` time, not at a later dereference. A null or delayed
        // `other` resolves without error and stays lazy through the redirect.
        let _ = other._getobj(true);
        let resolved = other._resolved_ptr();
        PTR_BECOME_TARGETS.with(|targets| {
            targets.borrow_mut().insert(self._identity, resolved);
        });
    }

    /// RPython `_abstract_ptr._getobj(self, check=True)`
    /// (`lltype.py`). Resolves `_obj0`, running `obj._check()`
    /// (which raises on access to freed storage) when `check`. A null
    /// pointer is `Ok(None)`; a delayed pointer is `Err(DelayedPointer)`.
    /// The `_obj` property is `_getobj(check=True)`.
    fn _getobj(&self, check: bool) -> Result<Option<_ptr_obj>, DelayedPointer> {
        let resolved = self._resolved_ptr();
        match resolved._obj0 {
            PtrObj::Null => Ok(None),
            PtrObj::Delayed(_) => Err(DelayedPointer),
            PtrObj::Strong(obj) => {
                if check {
                    obj._check();
                }
                Ok(Some(obj))
            }
            // `obj = obj()` upgrades the weakref; a `None` upgrade is
            // `RuntimeError("accessing already garbage collected %r")`.
            PtrObj::Weak(w) => match w.upgrade() {
                None => panic!("accessing already garbage collected {:?}", resolved._TYPE),
                Some(obj) => {
                    if check {
                        obj._check();
                    }
                    Ok(Some(obj))
                }
            },
        }
    }

    /// The resolved `_obj0` in the old `Result<Option<_ptr_obj>,
    /// DelayedPointer>` shape (`_getobj(check=False)`): a weak slot is
    /// upgraded to its strong container without running the freed-storage
    /// check. Used by readers that previously matched `_obj0` directly.
    pub fn _obj0_value(&self) -> Result<Option<_ptr_obj>, DelayedPointer> {
        self._getobj(false)
    }

    /// The `_obj` property (`_getobj(check=True)`, lltype.py). Returns
    /// the underlying container; null pointers panic at the dereference
    /// site (upstream raises `AttributeError` on `self._obj.getattr(...)`
    /// implicitly, so either outcome is a programmer error).
    pub fn _obj(&self) -> Result<_ptr_obj, DelayedPointer> {
        match self._getobj(true)? {
            Some(obj) => Ok(obj),
            None => panic!("null low-level pointer has no underlying object"),
        }
    }

    /// RPython `_ptr._was_freed` (lltype.py):
    ///
    /// ```python
    /// def _was_freed(self):
    ///     return (type(self._obj0) not in (type(None), int) and
    ///             self._getobj(check=False)._was_freed())
    /// ```
    ///
    /// A null pointer (`_obj0 == None`) and a tagged int are never freed;
    /// a real container delegates to its own `_was_freed`. A delayed
    /// pointer surfaces as `Err(DelayedPointer)` because `_getobj`
    /// (lltype.py) raises on it — this propagates to the caller
    /// (`_wref._dereference`) rather than masquerading as not-freed. Uses
    /// `check=False` to avoid recursing into `_check`.
    pub fn _was_freed(&self) -> Result<bool, DelayedPointer> {
        match self._getobj(false)? {
            Some(obj) => Ok(obj._was_freed()),
            None => Ok(false),
        }
    }

    /// RPython `_abstract_ptr._same_obj` (`lltype.py`):
    /// `self._obj == other._obj`. Both sides go through `_getobj(check=True)`
    /// so a freed pointer raises; null-null compares equal; a delayed
    /// pointer on either side surfaces as `Err(DelayedPointer)` (the caller
    /// `__eq__` falls back to `self is other`).
    pub fn _same_obj(&self, other: &_ptr) -> Result<bool, DelayedPointer> {
        let a = self._getobj(true)?;
        let b = other._getobj(true)?;
        Ok(a == b)
    }

    /// RPython `_abstract_ptr.__nonzero__` (`lltype.py`):
    /// `self._obj is not None`, i.e. `_getobj(check=True)` — a freed pointer
    /// raises; a delayed pointer is assumed non-null.
    pub fn nonzero(&self) -> bool {
        match self._getobj(true) {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(DelayedPointer) => true,
        }
    }

    /// RPython `_ptr._cast_to(self, PTRTYPE)` (`lltype.py`).
    ///
    /// Walks the struct super-chain to produce a re-typed pointer that
    /// aliases the same underlying allocation. Down-casts (positive
    /// `castable` depth) walk via the first field name (`super`);
    /// up-casts via `_parentstructure()`.
    ///
    /// Down-casts and identity casts re-point through the first field name.
    /// Null casts return `Ptr(target)._defl()`. Up-casts walk up the parent
    /// chain via `_parentstructure()` ([`Parentable::parentstructure`]),
    /// checking at each step that `struc` is the parent's first inlined field
    /// (`getattr(parent, PARENTTYPE._names[0]) != struc`) and then verifying
    /// the widened-to type equals `PTRTYPE.TO`.
    pub fn _cast_to(&self, ptrtype: &Ptr) -> Result<_ptr, String> {
        let down_or_up = castable(ptrtype, &self._TYPE)?;
        if down_or_up == 0 {
            return Ok(self.clone());
        }
        // upstream: `if not self: return PTRTYPE._defl()`.
        if !self.nonzero() {
            return Ok(ptrtype._defl());
        }
        // upstream (lltype.py:1428): `if isinstance(self._obj, int): return
        // _ptr(PTRTYPE, self._obj, solid=True)` — a tagged-integer pointer
        // re-tags to the target type, keeping its integer payload.
        if let Ok(Some(_ptr_obj::IntCast(n))) = self._obj0_value() {
            return Ok(_ptr::new_with_solid(
                ptrtype.clone(),
                Ok(Some(_ptr_obj::IntCast(n))),
                true,
            ));
        }
        if down_or_up < 0 {
            // upstream (lltype.py:1436-1451): walk *up* the parent chain
            // `-down_or_up` times via `_parentstructure()`, checking at each
            // step that `struc` is the parent's first inlined field, then
            // verify the widened-to type equals `PTRTYPE.TO`.
            let mut u = -down_or_up;
            let mut struc = self
                ._obj()
                .map_err(|_| "_ptr._cast_to: delayed pointer".to_string())?;
            let mut parent_type: Option<LowLevelType> = None;
            while u > 0 {
                let parentable = parentable_of_obj(&struc)
                    .ok_or_else(|| format!("_ptr._cast_to: {struc:?} is not a _parentable"))?;
                let parent = parentable
                    .parentstructure(true)
                    .ok_or_else(|| format!("widening to trash: {:?}", self._TYPE))?;
                let ptype = parentable
                    .parent_type()
                    .expect("a parented container always records its parent type");
                // `getattr(parent, PARENTTYPE._names[0]) != struc` → InvalidCast.
                let LowLevelType::Struct(ps) = &ptype else {
                    return Err(format!(
                        "_ptr._cast_to: up-cast parent {ptype:?} is not a struct"
                    ));
                };
                let first_name = ps._names.first().ok_or_else(|| {
                    format!("_ptr._cast_to: parent struct {:?} has no fields", ps._name)
                })?;
                match first_field_container(&parent, first_name) {
                    Some(fc) if fc == struc => {}
                    _ => {
                        return Err(format!("InvalidCast: {:?} to {:?}", self._TYPE, ptrtype));
                    }
                }
                struc = parent;
                parent_type = Some(ptype);
                u -= 1;
            }
            let parent_type = parent_type.expect("an up-cast walks at least once");
            let target_type: LowLevelType = ptrtype.TO.clone().into();
            if parent_type != target_type {
                return Err(format!(
                    "widening {:?} inside {parent_type:?} instead of {target_type:?}",
                    self._TYPE
                ));
            }
            return Ok(_ptr::new_with_solid(
                ptrtype.clone(),
                Ok(Some(struc)),
                self._solid,
            ));
        }
        // upstream: `while down_or_up: p = getattr(p, typeOf(p).TO._names[0])`.
        // The first field of a Struct in our port is the leading entry
        // of `_flds`; for the instance/class hierarchy that is
        // `("super", parent_struct)`. Walk by repeated `_obj.getattr`
        // re-pointing through the chain.
        let mut current = self.clone();
        let mut steps = down_or_up;
        while steps > 0 {
            let to_struct = match &current._TYPE.TO {
                PtrTarget::Struct(s) => s.clone(),
                PtrTarget::ForwardReference(fwd) => match fwd.resolved() {
                    Some(LowLevelType::Struct(s)) => *s,
                    _ => {
                        return Err(format!(
                            "_ptr._cast_to: ForwardReference {fwd:?} did not \
                             resolve to a Struct"
                        ));
                    }
                },
                _ => {
                    return Err(format!(
                        "_ptr._cast_to: cast walk requires Struct target, got {:?}",
                        current._TYPE
                    ));
                }
            };
            let first_name = to_struct._names.first().ok_or_else(|| {
                format!(
                    "_ptr._cast_to: struct {:?} has no fields to walk",
                    to_struct._name
                )
            })?;
            let lv = current.getattr(first_name)?;
            let LowLevelValue::Ptr(next) = lv else {
                return Err(format!(
                    "_ptr._cast_to: vtable.{first_name} did not yield a Ptr value"
                ));
            };
            current = *next;
            steps -= 1;
        }
        // Re-wrap with the target Ptr type while preserving the
        // underlying object and solid bit. Upstream `_ptr(PTRTYPE,
        // p._obj, solid=self._solid)`.
        let obj0 = current._obj0_value();
        Ok(_ptr::new_with_solid(ptrtype.clone(), obj0, self._solid))
    }

    pub fn _identityhash(&self) -> i64 {
        assert_eq!(self._TYPE._gckind(), GcKind::Gc);
        assert!(self.nonzero());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self._obj()
            .expect("gc identityhash requires a concrete underlying object")
            .hash(&mut hasher);
        hasher.finish() as i64
    }

    pub fn call(&self, args: &[LowLevelValue]) -> LowLevelValue {
        match self
            ._obj()
            .expect("low-level function pointer example must expose an underlying object")
        {
            _ptr_obj::Func(func) => func.call(args),
            _ptr_obj::Struct(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Array(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Opaque(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Wref(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Subarray(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::ArrayLenRef(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::EndMarker(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::IntCast(_) => panic!("{:?} instance is not callable", self._TYPE),
        }
    }

    /// RPython `_abstract_ptr._fixedlength` (`lltype.py`).
    /// Always calls `len(self)` first, which raises `TypeError` for
    /// non-array pointers (so `_fixedlength` cannot silently succeed
    /// on a struct/function/opaque pointer). Returns `Ok(Some(n))` for
    /// FixedSizeArray and `Ok(None)` for plain `Array`.
    pub fn _fixedlength(&self) -> Result<Option<i64>, String> {
        let length = self.len()?;
        let is_fixed_size = match &self._TYPE.TO {
            PtrTarget::FixedSizeArray(_) => true,
            PtrTarget::ForwardReference(forward_ref) => matches!(
                forward_ref.resolved(),
                Some(LowLevelType::FixedSizeArray(_))
            ),
            _ => false,
        };
        Ok(if is_fixed_size {
            Some(length as i64)
        } else {
            None
        })
    }

    fn _as_parent_value(&self) -> LowLevelValue {
        match self
            ._obj()
            .expect("low-level pointer example must expose an underlying object")
        {
            _ptr_obj::Func(_) => panic!("function pointer has no container parent"),
            _ptr_obj::Struct(obj) => LowLevelValue::Struct(Box::new(obj.clone())),
            _ptr_obj::Array(obj) => LowLevelValue::Array(Box::new(obj.clone())),
            _ptr_obj::Opaque(obj) => LowLevelValue::Opaque(Box::new(obj.clone())),
            _ptr_obj::Wref(_) => panic!("weakref has no container parent"),
            _ptr_obj::Subarray(_) => panic!("subarray has no container parent"),
            _ptr_obj::ArrayLenRef(_) => panic!("arraylenref has no container parent"),
            _ptr_obj::EndMarker(_) => panic!("endmarker has no container parent"),
            _ptr_obj::IntCast(_) => panic!("tagged-int pointer has no container parent"),
        }
    }

    /// Delegate to [`Ptr::_interior_ptr_type_with_index`] — the method
    /// is defined on the pointer *type* upstream; the value wrapper is
    /// kept here so existing `_ptr`-side call sites stay ergonomic.
    pub fn _interior_ptr_type_with_index(&self, to: &LowLevelType) -> Struct {
        self._TYPE._interior_ptr_type_with_index(to)
    }

    pub fn _expose(&self, offset: InteriorOffset, value: LowLevelValue) -> LowLevelValue {
        let wrap_as_interior_ptr = self._TYPE._gckind() == GcKind::Gc;
        match value {
            LowLevelValue::Struct(obj) => {
                if wrap_as_interior_ptr && obj.TYPE._gckind == GcKind::Raw {
                    LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                        _T: LowLevelType::Struct(Box::new(obj.TYPE.clone())),
                        _parent: self._as_parent_value(),
                        _offsets: vec![offset],
                    }))
                } else {
                    LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                        Ptr {
                            TO: PtrTarget::Struct(obj.TYPE.clone()),
                        },
                        Ok(Some(_ptr_obj::Struct(*obj))),
                        self._solid,
                    )))
                }
            }
            LowLevelValue::Array(obj) => {
                let target = match &obj.TYPE {
                    ArrayContainer::Array(array_t) => PtrTarget::Array(array_t.clone()),
                    ArrayContainer::FixedSizeArray(array_t) => {
                        PtrTarget::FixedSizeArray(array_t.clone())
                    }
                };
                if wrap_as_interior_ptr && target._gckind() == GcKind::Raw {
                    LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                        _T: LowLevelType::from(target.clone()),
                        _parent: self._as_parent_value(),
                        _offsets: vec![offset],
                    }))
                } else {
                    LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                        Ptr { TO: target },
                        Ok(Some(_ptr_obj::Array(*obj))),
                        self._solid,
                    )))
                }
            }
            LowLevelValue::Opaque(obj) => LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                Ptr {
                    TO: PtrTarget::Opaque(obj.TYPE.clone()),
                },
                Ok(Some(_ptr_obj::Opaque(*obj))),
                self._solid,
            ))),
            other => other,
        }
    }

    pub fn _lookup_adtmeth(&self, member_name: &str) -> Result<LowLevelAdtMember, AttributeError> {
        match &self._TYPE.TO {
            PtrTarget::Struct(struct_t) => {
                struct_t._lookup_adtmeth(member_name, &LowLevelPointerType::Ptr(self._TYPE.clone()))
            }
            PtrTarget::ForwardReference(forward_ref) => {
                let Some(LowLevelType::Struct(struct_t)) = forward_ref.resolved() else {
                    return Err(AttributeError);
                };
                struct_t._lookup_adtmeth(member_name, &LowLevelPointerType::Ptr(self._TYPE.clone()))
            }
            _ => Err(AttributeError),
        }
    }

    pub fn getattr(&self, field_name: &str) -> Result<LowLevelValue, String> {
        if (matches!(&self._TYPE.TO, PtrTarget::Struct(_))
            || matches!(
                &self._TYPE.TO,
                PtrTarget::ForwardReference(forward_ref)
                    if matches!(forward_ref.resolved(), Some(LowLevelType::Struct(_)))
            ))
            && let _ptr_obj::Struct(obj) = self._obj().map_err(|_| {
                format!(
                    "delayed pointer {:?} has no field {:?}",
                    self._TYPE, field_name
                )
            })?
            && let Some(value) = obj._getattr(field_name)
        {
            return Ok(self._expose(InteriorOffset::Field(field_name.to_owned()), value.clone()));
        }
        match self._lookup_adtmeth(field_name) {
            Ok(_) => {
                panic!("_ptr.__getattr__ adtmethod path must be handled by SomePtr.getattr first")
            }
            Err(AttributeError) => Err(self._TYPE._nofield(field_name)),
        }
    }

    pub fn getitem(&self, index: usize) -> Result<LowLevelValue, String> {
        if matches!(
            &self._TYPE.TO,
            PtrTarget::Array(_) | PtrTarget::FixedSizeArray(_)
        ) || matches!(
            &self._TYPE.TO,
            PtrTarget::ForwardReference(forward_ref)
                if matches!(
                    forward_ref.resolved(),
                    Some(LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_))
                )
        ) {
            // `o = self._obj.getitem(i)` (lltype.py) — dispatched to the
            // container, so `_subarray`/`_arraylenref` interior pointers (from
            // `direct_fieldptr`/`direct_arrayitems`/`ArrayLengthOffset`) read
            // through too, not only `_array`.
            let obj = self
                ._obj()
                .map_err(|_| format!("delayed pointer {:?} is not an array", self._TYPE))?;
            // `start, stop = self._obj.getbounds(); if not (start <= i < stop):
            // raise IndexError("array index out of bounds")` (lltype.py-
            // 1293) — the bounds check happens at the pointer level, ahead of
            // the container `getitem`, so an out-of-range index (a backward
            // `_subarray`, a non-zero `_arraylenref`) is an array-index error
            // rather than a container panic.
            if let Some((start, stop)) = obj.container_getbounds() {
                let i = index as i64;
                if !(start <= i && i < stop) {
                    return Err(format!("array index out of bounds: {index}"));
                }
            }
            if let Some(value) = obj.container_getitem(index) {
                return Ok(self._expose(InteriorOffset::Index(index), value));
            }
            return Err(format!("array index out of bounds: {index}"));
        }
        Err(format!("{:?} instance is not an array", self._TYPE))
    }

    pub fn setattr(&mut self, field_name: &str, val: LowLevelValue) -> Result<(), String> {
        let struct_t = match &self._TYPE.TO {
            PtrTarget::Struct(struct_t) => Some(struct_t.clone()),
            PtrTarget::ForwardReference(forward_ref) => match forward_ref.resolved() {
                Some(LowLevelType::Struct(struct_t)) => Some(*struct_t),
                _ => None,
            },
            _ => None,
        };
        if let Some(struct_t) = &struct_t
            && let Some(field_t) = struct_t.getattr_field_type(field_name)
        {
            let got_t = typeOf_value(&val);
            if field_t != got_t {
                return Err(format!(
                    "{} instance field {:?}:\nexpects {:?}\n    got {:?}",
                    struct_t._name, field_name, field_t, got_t
                ));
            }
            let obj = self
                ._obj0_value()
                .map_err(|_| {
                    format!(
                        "delayed pointer {:?} has no field {:?}",
                        self._TYPE, field_name
                    )
                })?
                .expect("non-null struct pointer must expose an underlying object");
            let _ptr_obj::Struct(obj) = obj else {
                panic!("struct pointer must expose a struct object");
            };
            if !obj._setattr(field_name, val) {
                panic!("struct object must expose declared field {field_name:?}");
            }
            return Ok(());
        }
        Err(self._TYPE._nofield(field_name))
    }

    /// Variant of [`Self::setattr`] that descends through nested Struct
    /// fields named by `path` before writing `field`. The pointer
    /// continues to alias the same underlying allocation (unlike
    /// `_ptr.getattr("super")` which returns a detached copy of the
    /// substruct). Required by `ClassRepr.init_vtable` to write
    /// `subclassrange_min/max` / `rtti` on the OBJECT_VTABLE substruct
    /// nested inside a per-class vtable allocation.
    pub fn setattr_at_path(
        &mut self,
        path: &[&str],
        field: &str,
        val: LowLevelValue,
    ) -> Result<(), String> {
        let obj = self
            ._obj0_value()
            .map_err(|_| {
                format!(
                    "delayed pointer {:?} has no nested field {:?}",
                    self._TYPE, field
                )
            })?
            .expect("non-null struct pointer must expose an underlying object");
        let _ptr_obj::Struct(obj) = obj else {
            return Err(format!(
                "setattr_at_path: pointer {:?} does not target a Struct",
                self._TYPE
            ));
        };
        if !obj._setattr_descending(path, field, val) {
            return Err(format!(
                "setattr_at_path: nested write at path={:?} field={:?} failed",
                path, field
            ));
        }
        Ok(())
    }

    pub fn setitem(&mut self, index: usize, val: LowLevelValue) -> Result<(), String> {
        let expected = match &self._TYPE.TO {
            PtrTarget::Array(array_t) => Some(array_t.OF.clone()),
            PtrTarget::FixedSizeArray(array_t) => Some(array_t.OF.clone()),
            PtrTarget::ForwardReference(forward_ref) => match forward_ref.resolved() {
                Some(LowLevelType::Array(array_t)) => Some(array_t.OF.clone()),
                Some(LowLevelType::FixedSizeArray(array_t)) => Some(array_t.OF.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(expected) = expected else {
            return Err(format!("{:?} instance is not an array", self._TYPE));
        };
        let got_t = typeOf_value(&val);
        if expected != got_t {
            return Err(format!(
                "{:?} items:\nexpect {:?}\n   got {:?}",
                self._TYPE, expected, got_t
            ));
        }
        // `self._obj.setitem(i, val)` (lltype.py) — dispatched to the
        // container so `_subarray` interior pointers (from `direct_fieldptr`/
        // `direct_arrayitems`) write back through their parent, not only
        // `_array`. Element-type validity is checked above and re-asserted by
        // the container's own `setitem`.
        let obj = self
            ._obj()
            .map_err(|_| format!("delayed pointer {:?} is not an array", self._TYPE))?;
        // `start, stop = self._obj.getbounds(); if not (start <= i < stop):
        // raise IndexError` (lltype.py) — bounds-check at the pointer
        // level before dispatching, mirroring `__getitem__`.
        if let Some((start, stop)) = obj.container_getbounds() {
            let i = index as i64;
            if !(start <= i && i < stop) {
                return Err(format!("array index out of bounds: {index}"));
            }
        }
        if !obj.container_setitem(index, val) {
            return Err(format!("array index out of bounds: {index}"));
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize, String> {
        if (matches!(
            &self._TYPE.TO,
            PtrTarget::Array(_) | PtrTarget::FixedSizeArray(_)
        ) || matches!(
            &self._TYPE.TO,
            PtrTarget::ForwardReference(forward_ref)
                if matches!(
                    forward_ref.resolved(),
                    Some(LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_))
                )
        )) && let _ptr_obj::Array(obj) = self
            ._obj()
            .map_err(|_| format!("delayed pointer {:?} has no length", self._TYPE))?
        {
            return Ok(obj.getlength());
        }
        Err(format!("{:?} instance has no length attribute", self._TYPE))
    }

    pub fn is_empty(&self) -> Result<bool, String> {
        self.len().map(|len| len == 0)
    }
}

impl _interior_ptr {
    pub fn _togckind(&self) -> GcKind {
        self._T._gckind()
    }

    pub fn nonzero(&self) -> bool {
        panic!("do not test an interior pointer for nullity")
    }

    pub fn _same_obj(&self, other: &_interior_ptr) -> bool {
        self._obj() == other._obj()
    }

    pub fn _obj(&self) -> LowLevelValue {
        let mut ob = self._parent.clone();
        for offset in &self._offsets {
            ob = match (offset, ob) {
                (InteriorOffset::Field(name), LowLevelValue::Struct(obj)) => obj
                    ._getattr(name)
                    .unwrap_or_else(|| panic!("interior ptr field {name:?} missing")),
                (InteriorOffset::Index(index), LowLevelValue::Array(obj)) => obj
                    .getitem(*index)
                    .unwrap_or_else(|| panic!("interior ptr index path missing item {index}")),
                (offset, other) => panic!("invalid interior ptr offset {offset:?} on {other:?}"),
            };
        }
        ob
    }

    /// Materialise this interior pointer into a plain container `_ptr` over
    /// the same shared object. `_expose` chooses the interior form for an
    /// inlined Raw container under a Gc parent (e.g. `rpy_string.chars`);
    /// callers that only need to read/write through the container — the
    /// interpreter's `getsubstruct`/`setarrayitem` handlers — take the plain
    /// pointer over the identical `_container`, so writes still persist.
    pub fn _as_container_ptr(&self, solid: bool) -> Result<_ptr, String> {
        let ptr_obj = match self._obj() {
            LowLevelValue::Array(a) => _ptr_obj::Array(*a),
            LowLevelValue::Struct(s) => _ptr_obj::Struct(*s),
            other => {
                return Err(format!(
                    "interior pointer target is not a container: {other:?}"
                ));
            }
        };
        Ok(ptr_obj._as_ptr(solid))
    }

    pub fn _TYPE(&self) -> InteriorPtr {
        InteriorPtr {
            PARENTTYPE: Box::new(typeOf_value(&self._parent)),
            TO: Box::new(self._T.clone()),
            offsets: self._offsets.clone(),
        }
    }

    pub fn _lookup_adtmeth(&self, member_name: &str) -> Result<LowLevelAdtMember, AttributeError> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(AttributeError);
        };
        obj.TYPE
            ._lookup_adtmeth(member_name, &LowLevelPointerType::InteriorPtr(self._TYPE()))
    }

    /// RPython `_abstract_ptr._fixedlength` (`lltype.py`),
    /// inherited by `_interior_ptr`. Always runs `len()` first so
    /// non-array interior pointers surface the upstream `TypeError`.
    pub fn _fixedlength(&self) -> Result<Option<i64>, String> {
        let length = self.len()?;
        match self._obj() {
            LowLevelValue::Array(obj) => match obj.TYPE {
                ArrayContainer::FixedSizeArray(_) => Ok(Some(length as i64)),
                ArrayContainer::Array(_) => Ok(None),
            },
            _ => Ok(None),
        }
    }

    pub fn _expose(&self, offset: InteriorOffset, value: LowLevelValue) -> LowLevelValue {
        let typ = typeOf_value(&value);
        match typ {
            LowLevelType::Struct(_)
            | LowLevelType::Array(_)
            | LowLevelType::FixedSizeArray(_)
            | LowLevelType::Opaque(_) => {
                assert_eq!(typ._gckind(), GcKind::Raw);
                let mut offsets = self._offsets.clone();
                offsets.push(offset);
                LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                    _T: typ,
                    _parent: self._parent.clone(),
                    _offsets: offsets,
                }))
            }
            _ => value,
        }
    }

    pub fn getattr(&self, field_name: &str) -> Result<LowLevelValue, String> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let Some(value) = obj._getattr(field_name) else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        Ok(self._expose(InteriorOffset::Field(field_name.to_owned()), value.clone()))
    }

    pub fn setitem(&mut self, index: usize, val: LowLevelValue) -> Result<(), String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!("{:?} instance is not an array", self._TYPE()));
        };
        let expected = match obj.TYPE {
            ArrayContainer::Array(ref array_t) => array_t.OF.clone(),
            ArrayContainer::FixedSizeArray(ref array_t) => array_t.OF.clone(),
        };
        let got_t = typeOf_value(&val);
        if expected != got_t {
            return Err(format!(
                "{:?} items:\nexpect {:?}\n   got {:?}",
                self._TYPE(),
                expected,
                got_t
            ));
        }
        let LowLevelValue::Array(obj) = self._resolve_mut()? else {
            unreachable!();
        };
        if !obj.setitem(index, val) {
            return Err(format!("array index out of bounds: {index}"));
        }
        Ok(())
    }

    pub fn getitem(&self, index: usize) -> Result<LowLevelValue, String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!("{:?} instance is not an array", self._TYPE()));
        };
        let Some(value) = obj.getitem(index) else {
            return Err(format!("array index out of bounds: {index}"));
        };
        Ok(self._expose(InteriorOffset::Index(index), value.clone()))
    }

    pub fn setattr(&mut self, field_name: &str, val: LowLevelValue) -> Result<(), String> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let Some(field_t) = obj.TYPE.getattr_field_type(field_name) else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let got_t = typeOf_value(&val);
        if field_t != got_t {
            return Err(format!(
                "{:?} field {:?}:\nexpects {:?}\n    got {:?}",
                self._TYPE(),
                field_name,
                field_t,
                got_t
            ));
        }
        let LowLevelValue::Struct(obj) = self._resolve_mut()? else {
            unreachable!();
        };
        if !obj._setattr(field_name, val) {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize, String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!(
                "{:?} instance has no length attribute",
                self._TYPE()
            ));
        };
        Ok(obj.getlength())
    }

    pub fn is_empty(&self) -> Result<bool, String> {
        self.len().map(|len| len == 0)
    }

    pub fn call(&self, _args: &[LowLevelValue]) -> LowLevelValue {
        panic!("{:?} instance is not callable", self._TYPE())
    }

    /// Walks `_parent` by `_offsets` and returns the terminal element
    /// as an owned `LowLevelValue`. The shared-storage discipline of
    /// `_struct._fields` and `_array.items` (`Arc<Mutex<...>>`) means
    /// owned clones still alias the underlying
    /// container, so subsequent `_setattr` / `setitem` calls on the
    /// returned owned value mutate the same backing storage as the
    /// original `_parent` chain.
    fn _resolve_mut(&self) -> Result<LowLevelValue, String> {
        fn descend(
            current: LowLevelValue,
            offsets: &[InteriorOffset],
        ) -> Result<LowLevelValue, String> {
            if offsets.is_empty() {
                return Ok(current);
            }
            match (&offsets[0], current) {
                (InteriorOffset::Field(name), LowLevelValue::Struct(obj)) => {
                    let next = obj
                        ._getattr(name)
                        .ok_or_else(|| format!("interior ptr field {name:?} missing"))?;
                    descend(next, &offsets[1..])
                }
                (InteriorOffset::Index(index), LowLevelValue::Array(obj)) => {
                    let next = obj
                        .getitem(*index)
                        .ok_or_else(|| format!("array index out of bounds: {index}"))?;
                    descend(next, &offsets[1..])
                }
                (offset, other) => Err(format!(
                    "invalid interior ptr offset {offset:?} on {other:?}"
                )),
            }
        }
        descend(self._parent.clone(), &self._offsets)
    }
}

impl LowLevelType {
    pub fn _defl(&self) -> LowLevelValue {
        match self {
            LowLevelType::Void => LowLevelValue::Void,
            LowLevelType::Signed => LowLevelValue::Signed(0),
            LowLevelType::Unsigned => LowLevelValue::Unsigned(0),
            LowLevelType::SignedLongLong => LowLevelValue::Signed(0),
            LowLevelType::SignedLongLongLong => LowLevelValue::Signed(0),
            LowLevelType::UnsignedLongLong => LowLevelValue::Unsigned(0),
            LowLevelType::UnsignedLongLongLong => LowLevelValue::Unsigned(0),
            LowLevelType::Bool => LowLevelValue::Bool(false),
            LowLevelType::Float => LowLevelValue::Float(0.0f64.to_bits()),
            LowLevelType::SingleFloat => LowLevelValue::SingleFloat(0.0f32.to_bits()),
            LowLevelType::LongFloat => LowLevelValue::LongFloat(0.0f64.to_bits()),
            LowLevelType::Char => LowLevelValue::Char('\0'),
            LowLevelType::UniChar => LowLevelValue::UniChar('\0'),
            // upstream `Address = Primitive("Address", NULL)` with
            // `NULL = fakeaddress(None)` — `_defl` returns the NULL
            // sentinel.
            LowLevelType::Address => LowLevelValue::Address(_address::Null),
            LowLevelType::Func(_) => {
                panic!("FuncType has no standalone low-level value default")
            }
            LowLevelType::Struct(struct_t) => {
                LowLevelValue::Struct(Box::new(struct_t._container_example()))
            }
            LowLevelType::Array(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::FixedSizeArray(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::Opaque(opaque_t) => {
                LowLevelValue::Opaque(Box::new(opaque_t._container_example()))
            }
            LowLevelType::ForwardReference(fwd) => match fwd.resolved() {
                // Once a ForwardReference has `become` its real type, its
                // `_defl()` must mirror the resolved container — upstream
                // mutates `fwd.__class__` / `fwd.__dict__` at `become` so the
                // `_defl` lookup dispatches through the real type directly
                // (lltype.py). Pyre carries the `ForwardReference`
                // wrapper, so resolve explicitly here.
                Some(resolved) => resolved._defl(),
                None => panic!("ForwardReference must be resolved before _defl()"),
            },
            LowLevelType::Ptr(ptr) => LowLevelValue::Ptr(Box::new(ptr._defl())),
            LowLevelType::InteriorPtr(ptr) => LowLevelValue::InteriorPtr(Box::new(ptr._example())),
        }
    }
}

pub fn typeOf(ptr: &_ptr) -> Ptr {
    ptr._TYPE.clone()
}

pub fn identityhash(p: &_ptr) -> i64 {
    assert!(p.nonzero());
    p._identityhash()
}

/// RPython `free(p, flavor, track_allocation=True)` (`lltype.py`).
/// The leakfinder side effect is intentionally absent; pyre does not port
/// RPython's allocation tracker, but it preserves the flavor/type guards and
/// clears the same `_parentable` storage bit through `_free()`.
pub fn free(p: &_ptr, flavor: &str, _track_allocation: bool) -> Result<(), String> {
    if flavor.starts_with("gc") {
        return Err("gc flavor free".to_string());
    }
    if p._togckind() != GcKind::Raw {
        return Err("free(): only for pointers to non-gc containers".to_string());
    }
    let obj = p
        ._obj()
        .map_err(|_| "free(): delayed pointer has no concrete container".to_string())?;
    obj._free()
}

pub fn typeOf_value(value: &LowLevelValue) -> ConcretetypePlaceholder {
    match value {
        LowLevelValue::Void => LowLevelType::Void,
        LowLevelValue::Signed(_) => LowLevelType::Signed,
        LowLevelValue::Unsigned(_) => LowLevelType::Unsigned,
        LowLevelValue::Bool(_) => LowLevelType::Bool,
        LowLevelValue::Float(_) => LowLevelType::Float,
        LowLevelValue::SingleFloat(_) => LowLevelType::SingleFloat,
        LowLevelValue::LongFloat(_) => LowLevelType::LongFloat,
        LowLevelValue::Char(_) => LowLevelType::Char,
        LowLevelValue::UniChar(_) => LowLevelType::UniChar,
        LowLevelValue::Address(_) => LowLevelType::Address,
        LowLevelValue::Struct(obj) => LowLevelType::Struct(Box::new(obj.TYPE.clone())),
        LowLevelValue::Array(obj) => match &obj.TYPE {
            ArrayContainer::Array(array_t) => LowLevelType::Array(Box::new(array_t.clone())),
            ArrayContainer::FixedSizeArray(array_t) => {
                LowLevelType::FixedSizeArray(Box::new(array_t.clone()))
            }
        },
        LowLevelValue::Opaque(obj) => LowLevelType::Opaque(Box::new(obj.TYPE.clone())),
        LowLevelValue::Ptr(ptr) => LowLevelType::Ptr(Box::new(typeOf(ptr))),
        LowLevelValue::InteriorPtr(ptr) => LowLevelType::InteriorPtr(Box::new(ptr._TYPE())),
    }
}

/// RPython `safe_equal(x, y)` (`lltype.py`). The Rust port routes
/// recursion-sensitive low-level comparisons through their own `PartialEq`
/// implementations, so this public helper is the same equality surface.
pub fn safe_equal<T: PartialEq>(x: &T, y: &T) -> bool {
    x == y
}

/// RPython `isCompatibleType(TYPE1, TYPE2)` (`lltype.py`):
/// dispatches to `TYPE1._is_compatible(TYPE2)`. The Rust port folds
/// `LowLevelType._is_compatible = __eq__` into [`PartialEq`].
pub fn isCompatibleType(TYPE1: &LowLevelType, TYPE2: &LowLevelType) -> bool {
    TYPE1 == TYPE2
}

/// RPython `enforce(TYPE, value)` (`lltype.py`): return `value`
/// if `typeOf(value) == TYPE`, otherwise raise `TypeError`.
pub fn enforce(TYPE: &LowLevelType, value: LowLevelValue) -> Result<LowLevelValue, String> {
    let got = typeOf_value(&value);
    if isCompatibleType(&got, TYPE) {
        Ok(value)
    } else {
        Err(format!("expected {TYPE:?}, got {got:?}"))
    }
}

/// RPython `normalizeptr(p, check=True)` (`lltype.py`): cast a
/// pointer to the largest containing structure, unwrapping hidden opaques.
/// Null pointers return `None`; tagged integer pointers and special
/// carry-around-for-tests pointers are already normalized.
pub fn normalizeptr(p: &_ptr, check: bool) -> Result<Option<_ptr>, String> {
    let Some(obj) = p
        ._getobj(check)
        .map_err(|_| "normalizeptr() cannot resolve delayed pointer".to_string())?
    else {
        return Ok(None);
    };
    if matches!(p._obj0_value(), Ok(Some(_ptr_obj::IntCast(_)))) {
        return Ok(Some(p.clone()));
    }
    let container = obj._normalizedcontainer();
    if container == obj {
        return Ok(Some(p.clone()));
    }
    let ptr_t = Ptr::from_container_type(container._container_type())?;
    Ok(Some(_ptr::new_with_solid(
        ptr_t,
        Ok(Some(container)),
        p._solid,
    )))
}

impl FuncType {
    pub fn _container_example(&self) -> _func {
        // upstream `FuncType._container_example` (lltype.py)
        // constructs `_func(self, _callable=ex)` where `ex` is a fresh
        // closure returning `self.RESULT._defl()`. Rust stores
        // `_callable` as an Option<String>; the marker `"<example>"`
        // keeps `_func.call` from tripping the "undefined function"
        // raise while the arg validation still runs.
        _func::new(
            self.clone(),
            "<example>".into(),
            None,
            Some("<example>".into()),
            HashMap::new(),
        )
    }

    /// RPython `FuncType._short_name` (`lltype.py`) — composes
    /// args/result short-names into `"Func(arg1, arg2, ...)->result"`.
    /// `saferecursive` guard is approximated by not recursing through
    /// `LowLevelType::Func` (returns bare `"Func(...)"`); the full
    /// TLS-backed recursion guard isn't needed until a downstream caller
    /// builds mutually-recursive FuncTypes.
    pub fn _short_name(&self) -> String {
        let args: Vec<String> = self.args.iter().map(LowLevelType::short_name).collect();
        format!("Func({})->{}", args.join(", "), self.result.short_name())
    }
}

impl Struct {
    pub fn new(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::_build(name, fields, GcKind::Raw, vec![], vec![], true)
    }

    pub fn gc(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::_build(name, fields, GcKind::Gc, vec![], vec![], true)
    }

    /// Struct whose field names are registry spellings. Synthetic positional
    /// names (`__pos_0`) are the getattr key, so the leading-`_` `NameError`
    /// in `Struct.__init__` does not apply to them.
    pub fn registry_fields(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        gckind: GcKind,
    ) -> Self {
        Self::_build(name, fields, gckind, vec![], vec![], false)
    }

    pub fn with_adtmeths(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        adtmeths: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Raw, adtmeths, vec![], true)
    }

    /// Raw `Struct(name, *fields, hints={...})`. Upstream
    /// `rpython/rtyper/lltypesystem/lltype.py Struct.__init__`
    /// forwards the `hints` kwarg through `_install_extras`.
    pub fn with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Raw, vec![], hints, true)
    }

    /// `GcStruct(name, *fields, hints={...})`. Same as
    /// [`Struct::gc`] plus the `hints` dict upstream passes through
    /// `Struct.__init__` kwargs.
    pub fn gc_with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Gc, vec![], hints, true)
    }

    /// `GcStruct(name, *fields, adtmeths={...}, hints={...})`.
    /// `get_ll_dict` (`rordereddict.py`) installs both the `dict` hint
    /// and the `keyhash` / `keyeq` / `r_rdict_eqfn` / `r_rdict_hashfn` /
    /// `paranoia` adtmeths on the same `dicttable` struct.
    pub fn gc_with_adtmeths_and_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        adtmeths: Vec<(String, ConstValue)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Gc, adtmeths, hints, true)
    }

    /// `GcStruct(name, *fields, rtti=True)` — upstream
    /// `RttiStruct._install_extras(rtti=True)` (`lltype.py`) mints
    /// an `_opaque` of type `RuntimeTypeInfo` with `_name` = the struct
    /// name and stores it under `_runtime_type_info`. `getRuntimeTypeInfo`
    /// / `attachRuntimeTypeInfo` later surface it wrapped in
    /// `Ptr(RuntimeTypeInfo)`. Must be called on a fresh struct-type —
    /// the opaque's `_identity` is per-call and therefore per struct.
    pub fn gc_rtti(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::gc_rtti_with_hints(name, fields, vec![])
    }

    /// `GcStruct(name, *fields, hints=hints, rtti=True)` — upstream
    /// funnels both options through `Struct.__init__(**kwds)` so they
    /// compose freely (`lltype.py`). Used e.g. for
    /// `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
    /// hints={...}, rtti=True)` (`rclass.py`).
    pub fn gc_rtti_with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        let mut result = Self::_build(name, fields, GcKind::Gc, vec![], hints, true);
        let rtti_ptr = opaqueptr_with_attrs(
            RUNTIME_TYPE_INFO.clone(),
            &result._name,
            Some(LowLevelType::Struct(Box::new(result.clone()))),
        )
        .expect("opaqueptr(RuntimeTypeInfo, ...) must succeed for gc_rtti()");
        let Ok(Some(_ptr_obj::Opaque(rtti_opaque))) = rtti_ptr._obj0_value() else {
            panic!("opaqueptr(RuntimeTypeInfo, ...) must yield an opaque container");
        };
        result._runtime_type_info = Some(Box::new(rtti_opaque));
        result
    }

    /// Unified constructor mirroring RPython `Struct.__init__` /
    /// `Struct._install_extras` (`lltype.py, 208-210`). Enforces:
    /// * field names must not begin with `_` (`NameError` upstream);
    /// * repeated field names are rejected (`TypeError` upstream);
    /// * a non-raw container type can only be inlined as the first
    ///   field of a struct with matching `_gckind`;
    /// * the last field is allowed to be varsize (recorded in
    ///   `_arrayfld`).
    fn _build(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        gckind: GcKind,
        adtmeths: Vec<(String, ConstValue)>,
        hints: Vec<(String, ConstValue)>,
        reject_underscore_names: bool,
    ) -> Self {
        let mut seen: Vec<String> = Vec::with_capacity(fields.len());
        let first_name = fields.first().map(|(n, _)| n.clone());
        for (i, (fname, ftyp)) in fields.iter().enumerate() {
            if reject_underscore_names && fname.starts_with('_') {
                panic!(
                    "{}: field name {:?} should not start with an underscore",
                    name, fname
                );
            }
            if seen.iter().any(|existing| existing == fname) {
                panic!("{}: repeated field name", name);
            }
            seen.push(fname.clone());
            if ftyp.is_container_type() {
                let child_gc = ftyp._gckind();
                if child_gc != GcKind::Raw {
                    let is_first = i == 0;
                    let same_gc = child_gc == gckind;
                    if !is_first || !same_gc {
                        panic!(
                            "{}: cannot inline {:?} container {:?}",
                            name, child_gc, ftyp
                        );
                    }
                }
            }
        }
        let _ = first_name;
        let _arrayfld = fields.last().and_then(|(n, t)| {
            if t._is_varsize() {
                Some(n.clone())
            } else {
                None
            }
        });
        let names = fields.iter().map(|(n, _)| n.clone()).collect();
        let result = Struct {
            _name: name.into(),
            _flds: frozendict::new(fields),
            _names: names,
            _adtmeths: frozendict::new(adtmeths),
            _hints: frozendict::new(hints),
            _arrayfld,
            _gckind: gckind,
            _runtime_type_info: None,
        };
        let parent = LowLevelType::Struct(Box::new(result.clone()));
        for (i, (_, typ)) in result._flds.iter().enumerate() {
            typ._note_inlined_into(&parent, i == 0, i + 1 == result._flds.len())
                .unwrap_or_else(|err| panic!("{err}"));
        }
        result
    }

    /// RPython `Struct._is_varsize` (`lltype.py`) — a struct is
    /// varsize iff its last field is a varsize container recorded in
    /// `_arrayfld`.
    pub fn _is_varsize(&self) -> bool {
        self._arrayfld.is_some()
    }

    /// Upstream `self.__class__.__name__` for the `Struct`/`GcStruct` pair:
    /// `GcStruct` when gc-kinded, else `Struct`. The Rust port collapses the
    /// `Struct`/`RttiStruct`/`GcStruct` class hierarchy into one type, so the
    /// class name is recovered from `_gckind`.
    fn _class_name(&self) -> &'static str {
        match self._gckind {
            GcKind::Gc => "GcStruct",
            _ => "Struct",
        }
    }

    /// RPython `Struct._short_name` (`lltype.py`) —
    /// `"<class_name> <struct_name>"` where `class_name` is `Struct` or
    /// `GcStruct` depending on `_gckind`.
    pub fn _short_name(&self) -> String {
        format!("{} {}", self._class_name(), self._name)
    }

    /// RPython `Struct._first_struct` (`lltype.py`). Returns the
    /// leading field name and type iff it is a struct of matching
    /// `_gckind`; used by rtyper to walk gc-inlined struct chains.
    pub fn _first_struct(&self) -> Option<(String, &Struct)> {
        let first_name = self._names.first()?;
        let first_type = self._flds.get(first_name)?;
        let LowLevelType::Struct(first_struct) = first_type else {
            return None;
        };
        if self._gckind != first_struct._gckind {
            return None;
        }
        Some((first_name.clone(), first_struct.as_ref()))
    }

    /// Owned variant of [`Self::_first_struct`] that resolves a
    /// leading `ForwardReference` field by cloning out the resolved
    /// struct body. Required by `castdepth` to walk the inheritance
    /// chain produced by `InstanceRepr._setup_repr` (where the
    /// immediate parent is wrapped in `ForwardReference` to support
    /// `_become`-style late resolution).
    pub fn _first_struct_owned(&self) -> Option<(String, Struct)> {
        let first_name = self._names.first()?.clone();
        let first_type = self._flds.get(&first_name)?.clone();
        let first_struct: Struct = match first_type {
            LowLevelType::Struct(boxed) => *boxed,
            LowLevelType::ForwardReference(fwd) => match fwd.resolved()? {
                LowLevelType::Struct(boxed) => *boxed,
                _ => return None,
            },
            _ => return None,
        };
        if self._gckind != first_struct._gckind {
            return None;
        }
        Some((first_name, first_struct))
    }

    /// RPython `Struct._is_atomic` (`lltype.py`). All fields must
    /// themselves be atomic (primitive or opaque-with-no-gc-children).
    pub fn _is_atomic(&self) -> bool {
        self._flds.iter().all(|(_, typ)| typ._is_atomic())
    }

    /// RPython `Struct._names_without_voids` (`lltype.py`).
    /// Returns field names whose type is not `Void`.
    pub fn _names_without_voids(&self) -> Vec<String> {
        self._names
            .iter()
            .filter(|name| {
                self._flds
                    .get(name)
                    .is_some_and(|typ| !matches!(typ, LowLevelType::Void))
            })
            .cloned()
            .collect()
    }

    /// RPython `Struct._note_inlined_into` (`lltype.py`). Checks
    /// structural rules at the parent container's inlining call site.
    /// Varsize structs cannot be inlined; gc structs can only be
    /// inlined as the first field of another gc struct.
    pub fn _note_inlined_into(&self, parent: &LowLevelType, first: bool) -> Result<(), String> {
        if self._arrayfld.is_some() {
            return Err("cannot inline a var-sized struct inside another container".to_string());
        }
        if self._gckind == GcKind::Gc {
            let parent_is_gc_struct = matches!(
                parent,
                LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
            );
            if !first || !parent_is_gc_struct {
                return Err(
                    "a GcStruct can only be inlined as the first field of another GcStruct"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// RPython `Struct._immutable_field(field)` (`lltype.py`).
    /// Returns `true` iff the struct carries `'immutable'` in `_hints`
    /// or the field is named in `_hints['immutable_fields']`.
    pub fn _immutable_field(&self, field: &str) -> bool {
        if matches!(self._hints.get("immutable"), Some(ConstValue::Bool(true))) {
            return true;
        }
        if let Some(ConstValue::Dict(fields)) = self._hints.get("immutable_fields")
            && fields.contains_key(&ConstValue::byte_str(field))
        {
            return true;
        }
        false
    }

    pub fn _container_example(&self) -> _struct {
        let fields: Vec<(String, LowLevelValue)> = self
            ._names
            .iter()
            .map(|name| {
                let typ = self
                    ._flds
                    .get(name)
                    .expect("Struct._names entry must exist in _flds");
                (name.clone(), typ._defl())
            })
            .collect();
        build_struct(self.clone(), fields)
    }

    pub fn getattr_field_type(&self, field_name: &str) -> Option<ConcretetypePlaceholder> {
        self._flds.get(field_name).cloned()
    }

    pub fn _lookup_adtmeth(
        &self,
        member_name: &str,
        ll_ptrtype: &LowLevelPointerType,
    ) -> Result<LowLevelAdtMember, AttributeError> {
        let Some(adtmember) = self._adtmeths.get(member_name) else {
            return Err(AttributeError);
        };
        match adtmember {
            ConstValue::HostObject(_) => Ok(LowLevelAdtMember::Method {
                ll_ptrtype: ll_ptrtype.clone(),
                func: adtmember.clone(),
            }),
            _ => Ok(LowLevelAdtMember::Value(adtmember.clone())),
        }
    }

    pub fn _nofield(&self, name: &str) -> String {
        format!("struct {} has no field {:?}", self._name, name)
    }
}

impl Array {
    pub fn new(of: ConcretetypePlaceholder) -> Self {
        Self::_build(of, GcKind::Raw, vec![])
    }

    pub fn gc(of: ConcretetypePlaceholder) -> Self {
        Self::_build(of, GcKind::Gc, vec![])
    }

    /// Raw `Array(OF, hints={...})`. Upstream
    /// `rpython.rtyper.lltypesystem.lltype.Array.__init__`
    /// + `_install_extras` forwards the `hints` kwarg to
    ///   `self._hints`. Used by
    ///   `SmallFunctionSetPBCRepr._setup_repr`, which
    ///   builds `Array(self.pointer_repr.lowleveltype,
    ///                hints={'nolength': True, 'immutable': True,
    ///                       'static_immutable': True})`.
    pub fn with_hints(of: ConcretetypePlaceholder, hints: Vec<(String, ConstValue)>) -> Self {
        Self::_build(of, GcKind::Raw, hints)
    }

    /// `GcArray(OF, hints={...})`. Same as [`Array::gc`] plus a
    /// `hints` dict mirrored from upstream `_install_extras`.
    pub fn gc_with_hints(of: ConcretetypePlaceholder, hints: Vec<(String, ConstValue)>) -> Self {
        Self::_build(of, GcKind::Gc, hints)
    }

    /// Unified constructor mirroring `Array.__init__` / `_install_extras`
    /// (`lltype.py`). Rejects any non-raw container as the item
    /// type (`lltype.py`) — a gc array-of-gc-containers would
    /// double-manage lifetimes.
    fn _build(
        of: ConcretetypePlaceholder,
        gckind: GcKind,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        if of.is_container_type() && of._gckind() != GcKind::Raw {
            panic!(
                "cannot have a {:?} container as array item type",
                of._gckind()
            );
        }
        let result = Array {
            OF: of,
            _hints: frozendict::new(hints),
            _gckind: gckind,
        };
        let parent = LowLevelType::Array(Box::new(result.clone()));
        result
            .OF
            ._note_inlined_into(&parent, false, false)
            .unwrap_or_else(|err| panic!("{err}"));
        result
    }

    pub fn _container_example(&self) -> _array {
        build_array(ArrayContainer::Array(self.clone()), vec![self.OF._defl()])
    }

    /// RPython `Array._is_atomic` (`lltype.py`) — an array is
    /// atomic iff its item type is atomic.
    pub fn _is_atomic(&self) -> bool {
        self.OF._is_atomic()
    }

    /// RPython `Array._short_name` (`lltype.py`). Includes
    /// `_gckind`-derived class prefix and the item short name.
    pub fn _short_name(&self) -> String {
        let kind = match self._gckind {
            GcKind::Gc => "GcArray",
            _ => "Array",
        };
        format!("{} {}", kind, self.OF.short_name())
    }

    /// RPython `Array._note_inlined_into` (`lltype.py`). Arrays
    /// only inline as the *last* field of a `Struct`; gc arrays cannot
    /// inline at all; no-length arrays cannot inline inside a GcStruct.
    pub fn _note_inlined_into(&self, parent: &LowLevelType, last: bool) -> Result<(), String> {
        let parent_is_struct = matches!(parent, LowLevelType::Struct(_));
        if !last || !parent_is_struct {
            return Err(
                "cannot inline an array in another container unless as the last field of a structure"
                    .to_string(),
            );
        }
        if self._gckind == GcKind::Gc {
            return Err("cannot inline a GC array inside a structure".to_string());
        }
        let parent_is_gc_struct = matches!(
            parent,
            LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
        );
        let has_nolength = matches!(self._hints.get("nolength"), Some(ConstValue::Bool(true)));
        if parent_is_gc_struct && has_nolength {
            return Err("cannot inline a no-length array inside a GcStruct".to_string());
        }
        Ok(())
    }

    /// RPython `Array._immutable_field` (`lltype.py`). Returns
    /// the `immutable` hint flag.
    pub fn _immutable_field(&self) -> bool {
        matches!(self._hints.get("immutable"), Some(ConstValue::Bool(true)))
    }
}

impl FixedSizeArray {
    pub fn new(of: ConcretetypePlaceholder, length: usize) -> Self {
        Self::_build(of, length, GcKind::Raw, vec![])
    }

    /// Unified constructor mirroring `FixedSizeArray.__init__`
    /// (`lltype.py`) — same item-type restrictions as `Array`
    /// apply (`lltype.py`).
    fn _build(
        of: ConcretetypePlaceholder,
        length: usize,
        gckind: GcKind,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        if of.is_container_type() && of._gckind() != GcKind::Raw {
            panic!(
                "cannot have a {:?} container as array item type",
                of._gckind()
            );
        }
        let result = FixedSizeArray {
            OF: of,
            length,
            _hints: frozendict::new(hints),
            _gckind: gckind,
        };
        let parent = LowLevelType::FixedSizeArray(Box::new(result.clone()));
        result
            .OF
            ._note_inlined_into(&parent, false, false)
            .unwrap_or_else(|err| panic!("{err}"));
        result
    }

    pub fn _container_example(&self) -> _array {
        let items: Vec<LowLevelValue> = (0..self.length).map(|_| self.OF._defl()).collect();
        build_array(ArrayContainer::FixedSizeArray(self.clone()), items)
    }

    /// RPython `FixedSizeArray._short_name` (`lltype.py`).
    pub fn _short_name(&self) -> String {
        format!("FixedSizeArray {} {}", self.length, self.OF.short_name())
    }

    /// RPython `FixedSizeArray._is_atomic` — inherited from Struct;
    /// because every `item%d` field shares `OF`, the test reduces to
    /// `OF._is_atomic()`.
    pub fn _is_atomic(&self) -> bool {
        self.OF._is_atomic()
    }
}

impl OpaqueType {
    pub fn new(tag: &str) -> Self {
        OpaqueType {
            tag: tag.into(),
            _gckind: GcKind::Raw,
        }
    }

    pub fn gc(tag: &str) -> Self {
        OpaqueType {
            tag: tag.into(),
            _gckind: GcKind::Gc,
        }
    }

    pub fn _container_example(&self) -> _opaque {
        _opaque::from_parts(
            self.clone(),
            Some("?".to_string()),
            None,
            None,
            None,
            None,
            None,
            false,
        )
    }

    /// RPython `OpaqueType._note_inlined_into` (`lltype.py`).
    /// Raw opaque values may be inlined; gc opaque values may not.
    pub fn _note_inlined_into(&self, parent: &LowLevelType) -> Result<(), String> {
        if self._gckind == GcKind::Gc {
            return Err(format!(
                "{:?} cannot be inlined in {:?}",
                self._short_name(),
                parent.short_name()
            ));
        }
        Ok(())
    }

    pub fn _short_name(&self) -> String {
        match self._gckind {
            GcKind::Gc => format!("{} (gcopaque)", self.tag),
            _ => format!("{} (opaque)", self.tag),
        }
    }
}

/// RPython `RuntimeTypeInfo = OpaqueType("RuntimeTypeInfo")` (lltype.py).
///
/// Module-level singleton opaque type token used as the target of
/// `Ptr(RuntimeTypeInfo)` for GC-tracked instances. R3 helpers
/// (`attachRuntimeTypeInfo`, `getRuntimeTypeInfo`, `ClassRepr::fill_vtable_root`'s
/// `rtti` slot) consume this as the well-known opaque type.
pub static RUNTIME_TYPE_INFO: LazyLock<LowLevelType> =
    LazyLock::new(|| LowLevelType::Opaque(Box::new(OpaqueType::new("RuntimeTypeInfo"))));

/// `_WeakRefType(_gckind='gc')` (llmemory.py).
pub static WEAKREF: LazyLock<LowLevelType> =
    LazyLock::new(|| LowLevelType::Opaque(Box::new(OpaqueType::gc("WeakRef"))));

/// `WeakRefPtr = Ptr(WeakRef)` (llmemory.py).
pub static WEAKREF_PTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(OpaqueType::gc("WeakRef")),
    }))
});

/// `_NONGCREF = lltype.Ptr(lltype.OpaqueType('NONGCREF'))` (llmemory.py).
/// The raw opaque pointer type `cast_int_to_adr` casts a tagged integer
/// through.
pub static NONGCREF: LazyLock<Ptr> = LazyLock::new(|| Ptr {
    TO: PtrTarget::Opaque(OpaqueType::new("NONGCREF")),
});

/// `GCREF = lltype.Ptr(lltype.GcOpaqueType('GCREF'))` (llmemory.py) — the
/// element type of the `GCARRAY_OF_PTR` placeholder array (`array_type_match`).
pub static GCREF: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(OpaqueType::gc("GCREF")),
    }))
});

fn new_opaque_container(TYPE: OpaqueType, name: &str, about: Option<LowLevelType>) -> _opaque {
    _opaque::from_parts(
        TYPE,
        Some(name.to_string()),
        about,
        None,
        None,
        None,
        None,
        false,
    )
}

/// `opaqueptr(PTRTYPE.TO, 'hidden', container=container, ORIGTYPE=...,
/// solid=...)` — a `'hidden'` opaque produced by `cast_opaque_ptr` that
/// stows the original container so the inverse cast can recover it
/// (`lltype.py`).
fn opaqueptr_hidden(
    TYPE: OpaqueType,
    container: _ptr_obj,
    origtype: Option<LowLevelType>,
    solid: bool,
) -> Result<_ptr, String> {
    let obj = _opaque::from_parts(
        TYPE.clone(),
        Some("hidden".to_string()),
        None,
        None,
        None,
        Some(Box::new(container)),
        origtype,
        solid,
    );
    let ptr_t = Ptr::from_container_type(LowLevelType::Opaque(Box::new(TYPE)))?;
    // `opaqueptr` always returns `_ptr(Ptr(TYPE), o, solid=True)`
    // (lltype.py) — the pointer to a hidden opaque is itself always
    // solid, independent of the `solid` attr recorded on the `_opaque`
    // (which carries the original cast pointer's solidity).
    Ok(_ptr::new_with_solid(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque(obj))),
        true,
    ))
}

fn expect_rtti_struct(T: &LowLevelType) -> Result<&Struct, String> {
    match T {
        LowLevelType::Struct(struct_t) if struct_t._gckind == GcKind::Gc => Ok(struct_t.as_ref()),
        _ => Err(format!("expected a RttiStruct: {}", T.short_name())),
    }
}

fn expect_rtti_struct_mut(T: &mut LowLevelType) -> Result<&mut Struct, String> {
    let short_name = T.short_name();
    match T {
        LowLevelType::Struct(struct_t) if struct_t._gckind == GcKind::Gc => Ok(struct_t.as_mut()),
        _ => Err(format!("expected a RttiStruct: {}", short_name)),
    }
}

fn attach_runtime_type_info_missing_error(struct_t: &Struct) -> String {
    format!(
        "attachRuntimeTypeInfo: {} must have been built with the rtti=True argument",
        struct_t._short_name()
    )
}

fn castdepth(outside: &Struct, inside: &Struct) -> i32 {
    if outside == inside {
        return 0;
    }
    let mut dwn = 0;
    // Walk an owned chain so we can resolve `ForwardReference` parents
    // produced by `_setup_repr` for the inheritance hierarchy.
    let mut current_owned = outside.clone();
    loop {
        let Some((_, first_type)) = current_owned._first_struct_owned() else {
            break;
        };
        dwn += 1;
        if &first_type == inside {
            return dwn;
        }
        current_owned = first_type;
    }
    -1
}

/// RPython `castable(PTRTYPE, CURTYPE)` (`lltype.py`). Returns
/// the signed depth distance between two `Ptr` types: positive when
/// `PTRTYPE` is reached by walking inward (through `_first_struct`),
/// negative when reached outward (parent), zero when identical. Errors
/// when the types are not castable (gc-status mismatch / non-Struct
/// targets / no chain).
pub fn castable(ptrtype: &Ptr, curtype: &Ptr) -> Result<i32, String> {
    castable_ptr_types(ptrtype, curtype)
}

fn castable_ptr_types(ptrtype: &Ptr, curtype: &Ptr) -> Result<i32, String> {
    if curtype._gckind() != ptrtype._gckind() {
        return Err(format!(
            "cast_pointer() cannot change the gc status: {:?} to {:?}",
            curtype, ptrtype
        ));
    }
    if curtype == ptrtype {
        return Ok(0);
    }
    let (PtrTarget::Struct(curstruc), PtrTarget::Struct(ptrstruc)) = (&curtype.TO, &ptrtype.TO)
    else {
        return Err(format!(
            "invalid cast between {:?} and {:?}",
            curtype, ptrtype
        ));
    };
    let d = castdepth(curstruc, ptrstruc);
    if d >= 0 {
        return Ok(d);
    }
    let u = castdepth(ptrstruc, curstruc);
    if u == -1 {
        return Err(format!(
            "invalid cast between {:?} and {:?}",
            curtype, ptrtype
        ));
    }
    Ok(-u)
}

fn validate_rtti_helper_ptr(
    funcptr: &_ptr,
    gcstruct: &Struct,
    result_type: &LowLevelType,
    error_label: &str,
) -> Result<(), String> {
    let T = typeOf(funcptr);
    let PtrTarget::Func(func_t) = &T.TO else {
        return Err(format!(
            "expected a {} function implementation, got: {:?}",
            error_label, funcptr
        ));
    };
    let expected_self_ptr =
        Ptr::from_container_type(LowLevelType::Struct(Box::new(gcstruct.clone())))
            .expect("Ptr(GcStruct) must be constructible for RTTI helper validation");
    let arg_ok = func_t.args.len() == 1
        && matches!(
            func_t.args.first(),
            Some(LowLevelType::Ptr(arg_ptr))
                if matches!(castable_ptr_types(arg_ptr, &expected_self_ptr), Ok(depth) if depth >= 0)
        );
    if !(arg_ok && func_t.result == *result_type) {
        return Err(format!(
            "expected a {} function implementation, got: {:?}",
            error_label, funcptr
        ));
    }
    Ok(())
}

impl Default for ForwardReference {
    fn default() -> Self {
        Self::new()
    }
}

impl ForwardReference {
    pub fn new() -> Self {
        ForwardReference {
            _gckind: GcKind::Raw,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn gc() -> Self {
        ForwardReference {
            _gckind: GcKind::Gc,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn prebuilt() -> Self {
        ForwardReference {
            _gckind: GcKind::Prebuilt,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn r#become(&self, realcontainertype: LowLevelType) -> Result<(), String> {
        if !realcontainertype.is_container_type() {
            return Err(format!(
                "ForwardReference can only be to a container, not {realcontainertype:?}"
            ));
        }
        if realcontainertype._gckind() != self._gckind {
            return Err(
                "become() gives conflicting gckind, use the correct XxForwardReference".to_string(),
            );
        }
        *self.target.lock() = Some(realcontainertype);
        Ok(())
    }

    pub fn resolved(&self) -> Option<LowLevelType> {
        self.target.lock().clone()
    }

    /// Pointer identity of the shared `become()` target cell. Stable and
    /// unique per forward reference (clones share the `Arc`), and read
    /// without locking the `Mutex` — usable as a deterministic key
    /// component even while another thread holds the lock.
    pub fn target_id(&self) -> usize {
        Arc::as_ptr(&self.target) as *const () as usize
    }
}

impl PartialEq for ForwardReference {
    fn eq(&self, other: &Self) -> bool {
        // Identity short-circuit: clones of the same `ForwardReference`
        // share the `Arc<Mutex<_>>` target, so pointer equality on the
        // Arc proves equivalence without recursing into the resolved
        // type — required to break cycles like `OBJECT_VTABLE →
        // OBJECTPTR → OBJECT → CLASSTYPE → OBJECT_VTABLE` (the
        // `instantiate` funcptr field closes the cycle and structural
        // recursion through `resolved == resolved` would not terminate).
        if Arc::ptr_eq(&self.target, &other.target) {
            return true;
        }
        match (self.resolved(), other.resolved()) {
            (Some(left), Some(right)) => {
                let id = Arc::as_ptr(&self.target) as *const _ as usize;
                if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                    // Already comparing this fwd elsewhere on the
                    // stack — short-circuit to `true` per RPython
                    // `saferecursive(safe_equal, True)` (lltype.py).
                    return true;
                }
                FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
                let r = left == right;
                FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
                r
            }
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => self._gckind == other._gckind,
        }
    }
}

impl Eq for ForwardReference {}

thread_local! {
    /// TLS-based saferecursive guard. Tracks the set of
    /// `ForwardReference` Arc pointers currently being compared / hashed
    /// on this thread's stack, so re-entry on the same `Arc` short-
    /// circuits to identity comparison / identity hashing instead of
    /// recursing forever through a cyclic type graph.
    static FORWARD_REF_EQ_GUARD: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    static FORWARD_REF_HASH_GUARD: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

impl Hash for ForwardReference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let id = Arc::as_ptr(&self.target) as *const _ as usize;
        let already = FORWARD_REF_HASH_GUARD.with(|g| g.borrow().contains(&id));
        if already {
            // Re-entry on the same fwd — hash a constant 0 per
            // RPython `saferecursive(get_hash, 0)` (lltype.py).
            // Identity hashing here would make hash values depend
            // on Arc address, breaking deterministic equality
            // contract for structurally-equal cyclic types.
            0_u8.hash(state);
            return;
        }
        match self.resolved() {
            Some(real) => {
                FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().insert(id));
                real.hash(state);
                FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().remove(&id));
            }
            None => panic!("ForwardReference object is not hashable"),
        }
    }
}

impl LowLevelType {
    pub fn is_container_type(&self) -> bool {
        matches!(
            self,
            LowLevelType::Func(_)
                | LowLevelType::Struct(_)
                | LowLevelType::Array(_)
                | LowLevelType::FixedSizeArray(_)
                | LowLevelType::Opaque(_)
                | LowLevelType::ForwardReference(_)
        )
    }

    /// RPython `LowLevelType._is_varsize` (`lltype.py`). Default
    /// is `False`; only `Array` and `Struct` with a trailing varsize
    /// field report `True`.
    pub fn _is_varsize(&self) -> bool {
        match self {
            LowLevelType::Array(_) => true,
            LowLevelType::Struct(t) => t._is_varsize(),
            LowLevelType::ForwardReference(t) => t.resolved().is_some_and(|t| t._is_varsize()),
            _ => false,
        }
    }

    /// RPython `LowLevelType._is_atomic` (`lltype.py`). Default
    /// is `False`; primitives report `True` and `Struct` is atomic iff
    /// all fields are atomic.
    pub fn _is_atomic(&self) -> bool {
        match self {
            LowLevelType::Void
            | LowLevelType::Bool
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Float
            | LowLevelType::SingleFloat
            | LowLevelType::LongFloat
            | LowLevelType::Char
            | LowLevelType::UniChar
            | LowLevelType::Address => true,
            LowLevelType::Struct(t) => t._is_atomic(),
            _ => false,
        }
    }

    /// RPython `LowLevelType._note_inlined_into` / container overrides
    /// (`lltype.py, 305-312, 441-448, 592-596`).
    pub fn _note_inlined_into(
        &self,
        parent: &LowLevelType,
        first: bool,
        last: bool,
    ) -> Result<(), String> {
        match self {
            LowLevelType::Void
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Bool
            | LowLevelType::Float
            | LowLevelType::SingleFloat
            | LowLevelType::LongFloat
            | LowLevelType::Char
            | LowLevelType::UniChar
            | LowLevelType::Address
            | LowLevelType::Ptr(_)
            | LowLevelType::InteriorPtr(_) => Ok(()),
            LowLevelType::Struct(t) => t._note_inlined_into(parent, first),
            LowLevelType::FixedSizeArray(t) => {
                if t._gckind == GcKind::Gc {
                    let parent_is_gc_struct = matches!(
                        parent,
                        LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
                    );
                    if !first || !parent_is_gc_struct {
                        return Err(
                            "a GcStruct can only be inlined as the first field of another GcStruct"
                                .to_string(),
                        );
                    }
                }
                Ok(())
            }
            LowLevelType::Array(t) => t._note_inlined_into(parent, last),
            LowLevelType::Opaque(t) => t._note_inlined_into(parent),
            LowLevelType::Func(_) => Err(format!(
                "FuncType cannot be inlined in {:?}",
                parent.short_name()
            )),
            LowLevelType::ForwardReference(t) => match t.resolved() {
                Some(real) => real._note_inlined_into(parent, first, last),
                None => Err(format!(
                    "ForwardReference cannot be inlined in {:?}",
                    parent.short_name()
                )),
            },
        }
    }
}

impl Ptr {
    pub fn _defl(&self) -> _ptr {
        _ptr::new(self.clone(), Ok(None))
    }

    /// RPython `Ptr.__new__(cls, TO)` validation + construction
    /// (`lltype.py`). Takes a container-kind `LowLevelType`
    /// variant and packs it into the matching `PtrTarget` arm. The
    /// upstream `WeakValueDictionary` cache is omitted — Rust
    /// `PartialEq` derives structural equality that covers the identity
    /// role of the cache.
    pub fn from_container_type(T: LowLevelType) -> Result<Self, String> {
        let TO = match T {
            LowLevelType::Func(t) => PtrTarget::Func(*t),
            LowLevelType::Struct(t) => PtrTarget::Struct(*t),
            LowLevelType::Array(t) => PtrTarget::Array(*t),
            LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray(*t),
            LowLevelType::Opaque(t) => PtrTarget::Opaque(*t),
            LowLevelType::ForwardReference(t) => PtrTarget::ForwardReference(*t),
            other => {
                return Err(format!(
                    "can only point to a Container type, not to {other:?}"
                ));
            }
        };
        Ok(Ptr { TO })
    }

    /// Short-name of the pointer's target container type. Called by
    /// `LowLevelType::short_name` for `"Ptr %s"` formatting (upstream
    /// `lltype.py`).
    pub fn _to_short_name(&self) -> String {
        match &self.TO {
            PtrTarget::Func(t) => t._short_name(),
            PtrTarget::Struct(t) => t._short_name(),
            PtrTarget::Array(t) => t._short_name(),
            PtrTarget::FixedSizeArray(t) => t._short_name(),
            PtrTarget::Opaque(t) => t.tag.clone(),
            PtrTarget::ForwardReference(t) => t
                .resolved()
                .map_or_else(|| "ForwardReference".to_string(), |real| real.short_name()),
        }
    }

    pub fn _example(&self) -> _ptr {
        _ptr::new_with_solid(
            self.clone(),
            Ok(Some(match &self.TO {
                PtrTarget::Func(func_t) => _ptr_obj::Func(func_t._container_example()),
                PtrTarget::Struct(struct_t) => _ptr_obj::Struct(struct_t._container_example()),
                PtrTarget::Array(array_t) => _ptr_obj::Array(array_t._container_example()),
                PtrTarget::FixedSizeArray(array_t) => _ptr_obj::Array(array_t._container_example()),
                PtrTarget::Opaque(opaque_t) => _ptr_obj::Opaque(opaque_t._container_example()),
                PtrTarget::ForwardReference(forward_ref) => {
                    match forward_ref
                        .resolved()
                        .expect("ForwardReference must be resolved before _example()")
                    {
                        LowLevelType::Func(func_t) => _ptr_obj::Func(func_t._container_example()),
                        LowLevelType::Struct(struct_t) => {
                            _ptr_obj::Struct(struct_t._container_example())
                        }
                        LowLevelType::Array(array_t) => {
                            _ptr_obj::Array(array_t._container_example())
                        }
                        LowLevelType::FixedSizeArray(array_t) => {
                            _ptr_obj::Array(array_t._container_example())
                        }
                        LowLevelType::Opaque(opaque_t) => {
                            _ptr_obj::Opaque(opaque_t._container_example())
                        }
                        LowLevelType::ForwardReference(_) => {
                            panic!("ForwardReference target must resolve to a concrete container")
                        }
                        LowLevelType::Ptr(_)
                        | LowLevelType::InteriorPtr(_)
                        | LowLevelType::Void
                        | LowLevelType::Signed
                        | LowLevelType::Unsigned
                        | LowLevelType::SignedLongLong
                        | LowLevelType::SignedLongLongLong
                        | LowLevelType::UnsignedLongLong
                        | LowLevelType::UnsignedLongLongLong
                        | LowLevelType::Bool
                        | LowLevelType::Float
                        | LowLevelType::SingleFloat
                        | LowLevelType::LongFloat
                        | LowLevelType::Char
                        | LowLevelType::UniChar
                        | LowLevelType::Address => {
                            panic!("ForwardReference target must resolve to a container type")
                        }
                    }
                }
            })),
            true,
        )
    }

    pub fn _nofield(&self, name: &str) -> String {
        match &self.TO {
            PtrTarget::Struct(struct_t) => struct_t._nofield(name),
            PtrTarget::Func(_)
            | PtrTarget::Array(_)
            | PtrTarget::FixedSizeArray(_)
            | PtrTarget::Opaque(_)
            | PtrTarget::ForwardReference(_) => {
                format!("{:?} instance has no field {:?}", self, name)
            }
        }
    }

    pub fn _gckind(&self) -> GcKind {
        self.TO._gckind()
    }

    pub fn _needsgc(&self) -> bool {
        !matches!(self.TO._gckind(), GcKind::Raw | GcKind::Prebuilt)
    }

    /// RPython `Ptr._interior_ptr_type_with_index(TO)` (`lltype.py`).
    /// Builds the GcStruct that represents an interior-pointer-with-index
    /// into a gc array. Invariants:
    /// * `self.TO._gckind` must be `gc` (upstream asserts this);
    /// * the resulting struct is a GcStruct, not a raw Struct;
    /// * the `interior_ptr_type` hint flags the struct as synthetic;
    /// * when `TO` is a Struct, its `_adtmeths` are copied.
    pub fn _interior_ptr_type_with_index(&self, to: &LowLevelType) -> Struct {
        assert_eq!(
            self.TO._gckind(),
            GcKind::Gc,
            "interior_ptr_type_with_index requires gc parent, got {:?}",
            self.TO._gckind()
        );
        let adtmeths = match to {
            LowLevelType::Struct(struct_t) => struct_t._adtmeths.to_vec(),
            _ => vec![],
        };
        let hints = vec![("interior_ptr_type".into(), ConstValue::Bool(true))];
        Struct::_build(
            "Interior",
            vec![
                ("ptr".into(), LowLevelType::Ptr(Box::new(self.clone()))),
                ("index".into(), LowLevelType::Signed),
            ],
            GcKind::Gc,
            adtmeths,
            hints,
            true,
        )
    }
}

/// RPython `nullptr(T)` (`lltype.py`):
///
/// ```python
/// def nullptr(T):
///     return Ptr(T)._defl()
/// ```
///
/// Returns a null pointer value whose `Ptr` type wraps the given container
/// `T`. Upstream uses `_ptr(self, None)` directly in `Ptr._defl`; the Rust
/// port follows the same two-step `Ptr::from_container_type(T)?._defl()`.
pub fn nullptr(T: LowLevelType) -> Result<_ptr, String> {
    Ok(Ptr::from_container_type(T)?._defl())
}

/// RPython `cast_ptr_to_int(ptr)` (`lltype.py`): delegate to
/// `_ptr._cast_to_int`. Null pointers become 0, tagged integer pointers keep
/// their payload, and ordinary pointers use their pointer identity.
pub fn cast_ptr_to_int(ptr: &_ptr) -> Result<i64, String> {
    match ptr._getobj(true) {
        Ok(None) => Ok(0),
        Ok(Some(_ptr_obj::IntCast(n))) => Ok(n),
        Ok(Some(_)) => Ok(ptr._hashable_identity() as i64),
        Err(DelayedPointer) => Err("DelayedPointer".into()),
    }
}

/// RPython `cast_int_to_ptr(PTRTYPE, oddint)` (`lltype.py`):
///
/// ```python
/// def cast_int_to_ptr(PTRTYPE, oddint):
///     if oddint == 0:
///         return nullptr(PTRTYPE.TO)
///     if not (oddint & 1):
///         raise ValueError("only odd integers can be cast back to ptr")
///     return _ptr(PTRTYPE, oddint, solid=True)
/// ```
///
/// Builds a tagged-integer pointer whose `_obj0` is the bare odd integer
/// ([`_ptr_obj::IntCast`]). `cast_int_to_adr` casts a raw integer through
/// `_NONGCREF` with this.
pub fn cast_int_to_ptr(PTRTYPE: &Ptr, oddint: i64) -> Result<_ptr, String> {
    if oddint == 0 {
        return nullptr(PTRTYPE.TO.clone().into());
    }
    if oddint & 1 == 0 {
        return Err("only odd integers can be cast back to ptr".into());
    }
    Ok(_ptr::new_with_solid(
        PTRTYPE.clone(),
        Ok(Some(_ptr_obj::IntCast(oddint))),
        true,
    ))
}

/// RPython `cast_pointer(PTRTYPE, ptr)` (`lltype.py`):
///
/// ```python
/// def cast_pointer(PTRTYPE, ptr):
///     CURTYPE = typeOf(ptr)
///     if not isinstance(CURTYPE, Ptr) or not isinstance(PTRTYPE, Ptr):
///         raise TypeError("can only cast pointers to other pointers")
///     return ptr._cast_to(PTRTYPE)
/// ```
///
/// Ports the entry point used by `ll_cast_to_object` (`rclass.py`)
/// and other repr-side static-cast call sites. The actual chain walk
/// lives on [`_ptr::_cast_to`].
pub fn cast_pointer(ptrtype: &Ptr, ptr: &_ptr) -> Result<_ptr, String> {
    ptr._cast_to(ptrtype)
}

impl _ptr_obj {
    /// `self._obj.getitem(i)` for the array-like containers `_ptr.__getitem__`
    /// reaches (lltype.py): `_array`, plus the `_subarray` /
    /// `_arraylenref` interior containers built by `direct_fieldptr` /
    /// `direct_arrayitems` / `ArrayLengthOffset`. Non-array containers (and an
    /// `_endmarker`, which exposes no items) yield `None`.
    fn container_getitem(&self, index: usize) -> Option<LowLevelValue> {
        match self {
            _ptr_obj::Array(a) => a.getitem(index),
            _ptr_obj::Subarray(s) => s.getitem(index),
            _ptr_obj::ArrayLenRef(l) => l.getitem(index),
            _ => None,
        }
    }

    /// `self._obj.setitem(i, val)` (lltype.py). `_arraylenref.setitem` is
    /// the in-place length shrink (lltype.py); the field-less
    /// `_endmarker` exposes no items, so it declines.
    fn container_setitem(&self, index: usize, value: LowLevelValue) -> bool {
        match self {
            _ptr_obj::Array(a) => a.setitem(index, value),
            _ptr_obj::Subarray(s) => s.setitem(index, value),
            _ptr_obj::ArrayLenRef(l) => l.setitem(index, value),
            _ => false,
        }
    }

    /// `self._obj.getbounds()` (lltype.py, :1314) — the `(start, stop)`
    /// the `_ptr.__getitem__`/`__setitem__` bounds check consults before
    /// dispatching to `getitem`/`setitem`. Non-array containers yield `None`.
    fn container_getbounds(&self) -> Option<(i64, i64)> {
        match self {
            _ptr_obj::Array(a) => {
                let (start, stop) = a.getbounds();
                Some((start as i64, stop as i64))
            }
            _ptr_obj::Subarray(s) => Some(s.getbounds()),
            _ptr_obj::ArrayLenRef(l) => Some(l.getbounds()),
            _ => None,
        }
    }

    /// `lltype.typeOf(container)` — the container's `LowLevelType`.
    fn _container_type(&self) -> LowLevelType {
        match self {
            _ptr_obj::Func(f) => LowLevelType::Func(Box::new(f.TYPE.clone())),
            _ptr_obj::Struct(s) => LowLevelType::Struct(Box::new(s.TYPE.clone())),
            _ptr_obj::Array(a) => match &a.TYPE {
                ArrayContainer::Array(t) => LowLevelType::Array(Box::new(t.clone())),
                ArrayContainer::FixedSizeArray(t) => {
                    LowLevelType::FixedSizeArray(Box::new(t.clone()))
                }
            },
            _ptr_obj::Opaque(o) => LowLevelType::Opaque(Box::new(o.TYPE.clone())),
            _ptr_obj::Wref(_) => WEAKREF.clone(),
            _ptr_obj::Subarray(s) => LowLevelType::FixedSizeArray(Box::new(s.TYPE.clone())),
            _ptr_obj::ArrayLenRef(_) => {
                LowLevelType::FixedSizeArray(Box::new(FixedSizeArray::new(LowLevelType::Signed, 1)))
            }
            _ptr_obj::EndMarker(e) => LowLevelType::Struct(Box::new(e.TYPE.clone())),
            _ptr_obj::IntCast(_) => panic!("tagged-int pointer has no container type"),
        }
    }

    /// `self is other` for two `Arc`-backed containers — same allocation,
    /// compared by `Arc::as_ptr` identity. Only the `_struct`/`_array`/
    /// `_opaque` parents an interior container memoizes against reach here;
    /// other variants are never cache parents and compare unequal.
    fn same_arc(&self, other: &_ptr_obj) -> bool {
        match (self, other) {
            (_ptr_obj::Struct(a), _ptr_obj::Struct(b)) => a.identity() == b.identity(),
            (_ptr_obj::Array(a), _ptr_obj::Array(b)) => a.identity() == b.identity(),
            (_ptr_obj::Opaque(a), _ptr_obj::Opaque(b)) => a.identity() == b.identity(),
            _ => false,
        }
    }

    /// `_container._as_ptr(self)` (lltype.py) — `_ptr(Ptr(typeOf
    /// (self)), self, solid)`. The upstream base method works for *every*
    /// `_container`, so this dispatches across all container variants: a
    /// `_wref` strong-fallback referent dereferences through here
    /// (`_wref._dereference`), and a weakref-to-weakref lands on the `Wref`
    /// arm. The `PtrTarget` per variant matches each container's own
    /// `_makeptr`/`_as_ptr`. Only the tagged-int `IntCast` carrier is not a
    /// container.
    fn _as_ptr(&self, solid: bool) -> _ptr {
        let target = match self {
            _ptr_obj::Struct(s) => PtrTarget::Struct(s.TYPE.clone()),
            _ptr_obj::Array(a) => match &a.TYPE {
                ArrayContainer::Array(t) => PtrTarget::Array(t.clone()),
                ArrayContainer::FixedSizeArray(t) => PtrTarget::FixedSizeArray(t.clone()),
            },
            _ptr_obj::Opaque(o) => PtrTarget::Opaque(o.TYPE.clone()),
            _ptr_obj::Func(f) => PtrTarget::Func(f.TYPE.clone()),
            // `_subarray._makeptr`: `Ptr(FixedSizeArray(ITEMTYPE, 1))`.
            _ptr_obj::Subarray(s) => PtrTarget::FixedSizeArray(s.TYPE.clone()),
            // `_arraylenref._makeptr`: `Ptr(FixedSizeArray(Signed, 1))`.
            _ptr_obj::ArrayLenRef(_) => {
                PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 1))
            }
            // `_endmarker_struct._as_ptr`: `Ptr(A)` (the item struct type).
            _ptr_obj::EndMarker(e) => PtrTarget::Struct(e.TYPE.clone()),
            // `_wref._as_ptr`: `Ptr(WeakRef)`.
            _ptr_obj::Wref(_) => PtrTarget::Opaque(OpaqueType::gc("WeakRef")),
            _ptr_obj::IntCast(_) => {
                panic!("tagged-int pointer has no container to take _as_ptr of")
            }
        };
        _ptr::new_with_solid(Ptr { TO: target }, Ok(Some(self.clone())), solid)
    }

    /// The container's `_was_freed()`. `_struct`/`_array`/`_opaque` are
    /// `_parentable` and consult their shared `_parentable` state;
    /// `_func`/`_wref` are plain `_container`s whose `_was_freed` is always
    /// `False` (lltype.py).
    fn _was_freed(&self) -> bool {
        match self {
            _ptr_obj::Struct(s) => s._was_freed(),
            _ptr_obj::Array(a) => a._was_freed(),
            _ptr_obj::Opaque(o) => o._was_freed(),
            _ptr_obj::Subarray(s) => s._was_freed(),
            _ptr_obj::EndMarker(e) => e._was_freed(),
            _ptr_obj::Func(_)
            | _ptr_obj::Wref(_)
            | _ptr_obj::ArrayLenRef(_)
            | _ptr_obj::IntCast(_) => false,
        }
    }

    /// `_container._free()` for variants that carry `_parentable` storage.
    /// Plain `_container` values (`_func`, `_wref`) have no storage slot to
    /// clear; tagged integer carriers are not freeable containers.
    fn _free(&self) -> Result<(), String> {
        match self {
            _ptr_obj::Struct(s) => {
                s._free();
                Ok(())
            }
            _ptr_obj::Array(a) => {
                a._free();
                Ok(())
            }
            _ptr_obj::Opaque(o) => {
                o._free();
                Ok(())
            }
            _ptr_obj::Subarray(s) => {
                s._parentable.free();
                Ok(())
            }
            _ptr_obj::EndMarker(e) => {
                e._parentable.free();
                Ok(())
            }
            _ptr_obj::Func(_) | _ptr_obj::Wref(_) | _ptr_obj::ArrayLenRef(_) => Ok(()),
            _ptr_obj::IntCast(_) => Err("free(): tagged integer pointer has no container".into()),
        }
    }

    /// `_container._check()` dispatched per variant. The three `_parentable`
    /// containers raise on access to freed storage (lltype.py);
    /// `_func`/`_wref` are plain `_container`s whose `_check` is a no-op
    /// (lltype.py). Called by `_ptr._getobj(check=True)`.
    fn _check(&self) {
        match self {
            _ptr_obj::Struct(s) => s._parentable.check(),
            _ptr_obj::Array(a) => a._parentable.check(),
            _ptr_obj::Opaque(o) => o._parentable.check(),
            _ptr_obj::Subarray(s) => s._parentable.check(),
            _ptr_obj::EndMarker(e) => e._parentable.check(),
            _ptr_obj::Func(_)
            | _ptr_obj::Wref(_)
            | _ptr_obj::ArrayLenRef(_)
            | _ptr_obj::IntCast(_) => {}
        }
    }

    /// `_container._normalizedcontainer()` — `_opaque` rewrites itself
    /// (lltype.py), otherwise `_parentable._normalizedcontainer`
    /// (lltype.py) walks `_struct`/`_array` up to the enclosing
    /// struct while this container is the parent's first inlined struct field,
    /// returning the promoted parent container object.
    fn _normalizedcontainer(&self) -> _ptr_obj {
        if let _ptr_obj::Opaque(o) = self {
            return o._normalizedcontainer();
        }
        let mut container = self.clone();
        while let Some(parentable) = parentable_of_obj(&container) {
            // `parent = container._parentstructure(check); if parent is None:
            //  break; index = container._parent_index; T = typeOf(parent)`.
            let Some(parent) = parentable.parentstructure(true) else {
                break;
            };
            // `if not isinstance(T, Struct) or T._first_struct()[0] != index
            //  or isinstance(T, FixedSizeArray): break`. A `FixedSizeArray`
            // parent is a distinct lltype here, so the `Struct` match already
            // excludes it.
            let Some(LowLevelType::Struct(parent_struct)) = parentable.parent_type() else {
                break;
            };
            let Some((first_name, _)) = parent_struct._first_struct() else {
                break;
            };
            if parentable.parent_index() != Some(ParentIndex::Field(first_name)) {
                break;
            }
            container = parent;
        }
        container
    }
}

/// RPython `cast_opaque_ptr(PTRTYPE, ptr)` (`lltype.py`) — casts
/// between an `OpaqueType` pointer and a concrete container pointer (or
/// between two opaque pointers), keeping the gc kind. A concrete→opaque
/// cast mints a `'hidden'` opaque that stows the original container so the
/// inverse cast can rebuild the real pointer.
///
/// Two upstream branches are not modelled:
/// - the `_cast_to_ptr` (lltype.py) / `_cast_to_opaque` (lltype.py)
///   container hooks are `hasattr`-guarded and no pyre container defines
///   them, so those branches are dead exactly as the `hasattr` checks make
///   them dead upstream;
/// - the tagged-int container case (`isinstance(container, int)`,
///   lltype.py) maps to pyre's `_ptr_obj::IntCast` carrier and is
///   handled below.
pub fn cast_opaque_ptr(PTRTYPE: &Ptr, ptr: &_ptr) -> Result<_ptr, String> {
    let curtype = ptr._TYPE.clone();
    if curtype == *PTRTYPE {
        return Ok(ptr.clone());
    }
    if curtype._gckind() != PTRTYPE._gckind() {
        return Err(format!(
            "cast_opaque_ptr() cannot change the gc status: {} to {}",
            LowLevelType::Ptr(Box::new(curtype)).short_name(),
            LowLevelType::Ptr(Box::new(PTRTYPE.clone())).short_name(),
        ));
    }
    let cur_opaque = matches!(curtype.TO, PtrTarget::Opaque(_));
    let exp_opaque = matches!(PTRTYPE.TO, PtrTarget::Opaque(_));
    let expected_to = || LowLevelType::from(PTRTYPE.TO.clone());

    if cur_opaque && !exp_opaque {
        // opaque -> concrete
        if !ptr.nonzero() {
            return nullptr(expected_to());
        }
        let _ptr_obj::Opaque(opaque) = obj_of(ptr)? else {
            return Err(format!("{ptr:?} does not come from a container"));
        };
        let container = opaque
            .container
            .clone()
            .ok_or_else(|| format!("{ptr:?} does not come from a container"))?;
        // upstream (lltype.py:998): `if isinstance(container, int): return
        // _ptr(PTRTYPE, container, solid=True)` — a hidden opaque holding a
        // tagged-integer container re-tags it to the target type.
        if let _ptr_obj::IntCast(n) = *container {
            return Ok(_ptr::new_with_solid(
                PTRTYPE.clone(),
                Ok(Some(_ptr_obj::IntCast(n))),
                true,
            ));
        }
        let p = _ptr::new_with_solid(
            Ptr::from_container_type(container._container_type())?,
            Ok(Some(*container)),
            opaque.solid,
        );
        cast_pointer(PTRTYPE, &p)
    } else if !cur_opaque && exp_opaque {
        // concrete -> opaque
        let PtrTarget::Opaque(opaque_t) = &PTRTYPE.TO else {
            unreachable!()
        };
        if !ptr.nonzero() {
            return nullptr(expected_to());
        }
        let container = obj_of(ptr)?;
        opaqueptr_hidden(
            opaque_t.clone(),
            container,
            Some(LowLevelType::Ptr(Box::new(curtype))),
            ptr._solid,
        )
    } else if cur_opaque && exp_opaque {
        // opaque -> opaque
        let PtrTarget::Opaque(opaque_t) = &PTRTYPE.TO else {
            unreachable!()
        };
        if !ptr.nonzero() {
            return nullptr(expected_to());
        }
        let _ptr_obj::Opaque(opaque) = obj_of(ptr)? else {
            return Err(format!("{ptr:?} does not come from a container"));
        };
        let container = opaque
            .container
            .clone()
            .ok_or_else(|| format!("{ptr:?} does not come from a container"))?;
        opaqueptr_hidden(opaque_t.clone(), *container, None, opaque.solid)
    } else {
        Err(format!(
            "invalid cast_opaque_ptr(): {} -> {}",
            LowLevelType::Ptr(Box::new(curtype)).short_name(),
            LowLevelType::Ptr(Box::new(PTRTYPE.clone())).short_name(),
        ))
    }
}

/// `ptr._obj` for a cast — surfaces a delayed pointer as an error rather
/// than panicking, matching the `try/except AttributeError` upstream uses
/// to detect a container-less pointer.
fn obj_of(ptr: &_ptr) -> Result<_ptr_obj, String> {
    ptr._obj()
        .map_err(|_| format!("{ptr:?} is a delayed pointer"))
}

/// RPython `getRuntimeTypeInfo(GCSTRUCT)` (`lltype.py`):
///
/// ```python
/// def getRuntimeTypeInfo(GCSTRUCT):
///     if not isinstance(GCSTRUCT, RttiStruct):
///         raise TypeError(...)
///     if GCSTRUCT._runtime_type_info is None:
///         raise ValueError("no attached runtime type info for GcStruct %s" % GCSTRUCT._name)
///     return _ptr(Ptr(RuntimeTypeInfo), GCSTRUCT._runtime_type_info)
/// ```
///
/// Returns a `Ptr(RuntimeTypeInfo)` pointing at the struct's pre-attached
/// opaque. Errors if the struct was built without `gc_rtti` (no opaque)
/// or if a non-Struct `LowLevelType` is passed. Both conditions match
/// upstream's `TypeError` / `ValueError`.
pub fn getRuntimeTypeInfo(T: &LowLevelType) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct(T)?;
    let Some(rtti_opaque) = &struct_t._runtime_type_info else {
        return Err(format!(
            "no attached runtime type info for GcStruct {}",
            struct_t._name
        ));
    };
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

/// RPython `runtime_type_info(p)` (`lltype.py`): find the top
/// container's RTTI, then validate an attached query funcptr when present.
pub fn runtime_type_info(p: &_ptr) -> Result<_ptr, String> {
    let LowLevelType::Ptr(ptr_t) = typeOf_value(&LowLevelValue::Ptr(Box::new(p.clone()))) else {
        return Err(format!(
            "runtime_type_info on non-RttiStruct pointer: {p:?}"
        ));
    };
    let PtrTarget::Struct(static_struct) = &ptr_t.TO else {
        return Err(format!(
            "runtime_type_info on non-RttiStruct pointer: {p:?}"
        ));
    };
    if static_struct._gckind != GcKind::Gc {
        return Err(format!(
            "runtime_type_info on non-RttiStruct pointer: {p:?}"
        ));
    }
    let struct_obj = p
        ._obj()
        .map_err(|_| "runtime_type_info() cannot resolve delayed pointer".to_string())?;
    let top_parent = top_container(&struct_obj);
    let result = getRuntimeTypeInfo(&top_parent._container_type())?;
    let static_info = getRuntimeTypeInfo(&LowLevelType::Struct(Box::new(static_struct.clone())))?;
    let _ptr_obj::Opaque(static_rtti) = static_info._obj().map_err(|_| {
        "runtime_type_info() static RuntimeTypeInfo pointer resolved as delayed".to_string()
    })?
    else {
        return Err("runtime_type_info() static RuntimeTypeInfo is not opaque".to_string());
    };
    let query_funcptr = static_rtti.query_funcptr.lock().clone();
    if let Some(query_funcptr) = query_funcptr {
        let PtrTarget::Func(func_t) = &query_funcptr._TYPE.TO else {
            return Err("runtime_type_info query_funcptr is not a function pointer".to_string());
        };
        let Some(LowLevelType::Ptr(query_arg_t)) = func_t.args.first() else {
            return Err("runtime_type_info query_funcptr has no pointer argument".to_string());
        };
        let casted = cast_pointer(query_arg_t, p)?;
        let LowLevelValue::Ptr(result2) =
            query_funcptr.call(&[LowLevelValue::Ptr(Box::new(casted))])
        else {
            return Err("runtime_type_info query_funcptr did not return a pointer".to_string());
        };
        if result != *result2 {
            return Err(format!(
                "runtime type-info function for {p:?} returned {result2:?}, should have been {result:?}"
            ));
        }
    }
    Ok(result)
}

/// RPython `attachRuntimeTypeInfo(GCSTRUCT, funcptr=None, destrptr=None)`
/// (`lltype.py`):
///
/// ```python
/// def attachRuntimeTypeInfo(GCSTRUCT, funcptr=None, destrptr=None):
///     if not isinstance(GCSTRUCT, RttiStruct):
///         raise TypeError(...)
///     GCSTRUCT._attach_runtime_type_info_funcptr(funcptr, destrptr)
///     return _ptr(Ptr(RuntimeTypeInfo), GCSTRUCT._runtime_type_info)
/// ```
///
/// Rust keeps the no-extra-args wrapper for existing call sites and
/// exposes the full mutable port in
/// [`attachRuntimeTypeInfo_with_ptrs`], which stores helper pointers on
/// the `_runtime_type_info` opaque just like upstream.
pub fn attachRuntimeTypeInfo(T: &LowLevelType) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct(T)?;
    let Some(rtti_opaque) = &struct_t._runtime_type_info else {
        return Err(attach_runtime_type_info_missing_error(struct_t));
    };
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

pub fn attachRuntimeTypeInfo_with_ptrs(
    T: &mut LowLevelType,
    funcptr: Option<_ptr>,
    destrptr: Option<_ptr>,
) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct_mut(T)?;
    let struct_snapshot = struct_t.clone();
    if struct_t._runtime_type_info.is_none() {
        return Err(attach_runtime_type_info_missing_error(&struct_snapshot));
    }
    let runtime_type_info_ptr = LowLevelType::Ptr(Box::new(Ptr::from_container_type(
        RUNTIME_TYPE_INFO.clone(),
    )?));
    if let Some(funcptr) = funcptr.as_ref() {
        validate_rtti_helper_ptr(
            funcptr,
            &struct_snapshot,
            &runtime_type_info_ptr,
            "runtime type info",
        )?;
    }
    if let Some(destrptr) = destrptr.as_ref() {
        validate_rtti_helper_ptr(
            destrptr,
            &struct_snapshot,
            &LowLevelType::Void,
            "destructor",
        )?;
    }
    // `self._runtime_type_info.query_funcptr = funcptr` (lltype.py):
    // mutate the existing RTTI opaque in place through its `Mutex` cells, so
    // its `Arc::as_ptr` identity is unchanged across the attach. A `None`
    // argument leaves the existing value, matching the `if ... is not None`
    // guards upstream.
    let rtti_opaque = struct_t._runtime_type_info.as_ref().expect("checked above");
    if let Some(funcptr) = funcptr {
        *rtti_opaque.query_funcptr.lock() = Some(Box::new(funcptr));
    }
    if let Some(destrptr) = destrptr {
        *rtti_opaque.destructor_funcptr.lock() = Some(Box::new(destrptr));
    }
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

/// RPython `opaqueptr(TYPE, name, **attrs)` (`lltype.py`):
///
/// ```python
/// def opaqueptr(TYPE, name, **attrs):
///     if not isinstance(TYPE, OpaqueType):
///         raise TypeError("opaqueptr() for OpaqueTypes only")
///     o = _opaque(TYPE, _name=name, **attrs)
///     return _ptr(Ptr(TYPE), o, solid=True)
/// ```
///
/// Mints a pointer to a freshly-constructed `_opaque` container stamped
/// with the given human-readable `name`. Used by
/// `RttiStruct._install_extras(rtti=True)` to register the struct's
/// `_runtime_type_info` opaque (`lltype.py`). The helper keeps
/// the public two-arg surface for existing Rust callers and routes the
/// actual construction through a private attrs-aware helper.
pub fn opaqueptr(TYPE: LowLevelType, name: &str) -> Result<_ptr, String> {
    opaqueptr_with_attrs(TYPE, name, None)
}

fn opaqueptr_with_attrs(
    TYPE: LowLevelType,
    name: &str,
    about: Option<LowLevelType>,
) -> Result<_ptr, String> {
    let LowLevelType::Opaque(opaque_t) = &TYPE else {
        return Err(format!("opaqueptr() for OpaqueTypes only, got {TYPE:?}"));
    };
    let obj = new_opaque_container((**opaque_t).clone(), name, about);
    let ptr_t = Ptr::from_container_type(TYPE)?;
    Ok(_ptr::new_with_solid(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque(obj))),
        true,
    ))
}

/// Allocation flavor for `malloc(T, ..., flavor=...)`, mirroring upstream
/// `lltype.py` string kwarg (`'gc'` | `'raw'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MallocFlavor {
    Gc,
    Raw,
}

/// RPython `lltype.malloc(T, n=None, flavor='gc', immortal=False, ...)`
/// (`lltype.py`).
///
/// ```python
/// def malloc(T, n=None, flavor='gc', immortal=False, zero=False, ...):
///     ...
///     if isinstance(T, Struct):
///         o = _struct(T, n, initialization=initialization)
///     elif isinstance(T, Array):
///         o = _array(T, n, initialization=initialization)
///     elif isinstance(T, OpaqueType):
///         assert n is None
///         o = _opaque(T, initialization=initialization)
///     ...
///     return _ptr(Ptr(T), o, solid)
/// ```
///
/// This port covers the subset R3/RString needs: `flavor='gc'` or
/// `'raw'` × `immortal=True/False` × Struct / Array / OpaqueType,
/// including varsize Structs whose trailing field is an `Array`.
/// `zero`, `track_allocation`, `add_memory_pressure`, and
/// `nonmovable` upstream kwargs are still not accepted. The `solid`
/// kwarg threaded through upstream `_ptr(Ptr(T), o, solid)` is
/// tracked on `_ptr._solid` (upstream lltype.py `immortal or
/// flavor == 'raw'`).
pub fn malloc(
    T: LowLevelType,
    n: Option<usize>,
    flavor: MallocFlavor,
    immortal: bool,
) -> Result<_ptr, String> {
    let obj: _ptr_obj = match &T {
        LowLevelType::Struct(struct_t) => {
            if struct_t._arrayfld.is_some() {
                let n = n.ok_or_else(|| {
                    format!("malloc: varsize struct {:?} requires n", struct_t._name)
                })?;
                let fields = struct_t
                    ._names
                    .iter()
                    .map(|name| {
                        let typ = struct_t
                            ._flds
                            .get(name)
                            .expect("Struct._names entry must exist in _flds");
                        if Some(name) == struct_t._arrayfld.as_ref() {
                            let LowLevelType::Array(array_t) = typ else {
                                panic!("Struct._arrayfld must name an Array field");
                            };
                            let inner_items: Vec<LowLevelValue> =
                                (0..n).map(|_| array_t.OF._defl()).collect();
                            let arr = build_array(
                                ArrayContainer::Array((**array_t).clone()),
                                inner_items,
                            );
                            (name.clone(), LowLevelValue::Array(Box::new(arr)))
                        } else {
                            (name.clone(), typ._defl())
                        }
                    })
                    .collect();
                _ptr_obj::Struct(build_struct((**struct_t).clone(), fields))
            } else {
                if n.is_some() {
                    return Err(format!(
                        "malloc: non-varsize struct {:?} got n={n:?}",
                        struct_t._name
                    ));
                }
                _ptr_obj::Struct(struct_t._container_example())
            }
        }
        LowLevelType::Array(array_t) => {
            let items: Vec<LowLevelValue> = match n {
                Some(n) => (0..n).map(|_| array_t.OF._defl()).collect(),
                None => array_t._container_example().items.lock().clone(),
            };
            _ptr_obj::Array(build_array(
                ArrayContainer::Array((**array_t).clone()),
                items,
            ))
        }
        LowLevelType::Opaque(opaque_t) => {
            if n.is_some() {
                return Err("malloc: OpaqueType does not accept n".to_string());
            }
            _ptr_obj::Opaque(opaque_t._container_example())
        }
        other => {
            return Err(format!("malloc: unmallocable type {other:?}"));
        }
    };
    // upstream lltype.py:2211-2212 — gc flavor on non-gc, non-immortal
    // struct/array/opaque is rejected. We only enforce it when !immortal
    // because the init_vtable call site is immortal and can target
    // either a gc or non-gc prototype container.
    if flavor == MallocFlavor::Gc && !immortal && T._gckind() != GcKind::Gc {
        return Err(format!(
            "gc flavor malloc of a non-GC non-immortal structure {:?}",
            T.short_name()
        ));
    }
    let solid = immortal || flavor == MallocFlavor::Raw;
    let ptr_t = Ptr::from_container_type(T)?;
    Ok(_ptr::new_with_solid(ptr_t, Ok(Some(obj)), solid))
}

impl InteriorPtr {
    pub fn _example(&self) -> _interior_ptr {
        let parent = match &*self.PARENTTYPE {
            LowLevelType::Struct(struct_t) => {
                LowLevelValue::Struct(Box::new(struct_t._container_example()))
            }
            LowLevelType::Array(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::FixedSizeArray(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::Opaque(opaque_t) => {
                LowLevelValue::Opaque(Box::new(opaque_t._container_example()))
            }
            other => panic!("InteriorPtr parent must be container, got {other:?}"),
        };
        _interior_ptr {
            _T: (*self.TO).clone(),
            _parent: parent,
            _offsets: self.offsets.clone(),
        }
    }
}

pub fn functionptr(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
) -> _ptr {
    functionptr_with_attrs(TYPE, name, graph, _callable, HashMap::new())
}

/// `rpython/rtyper/extfunc.py:72-74, 87-89`
/// `functionptr(FT, name, _external_name=name, _callable=...)`.
pub fn functionptr_with_external_name(
    TYPE: FuncType,
    name: &str,
    _callable: Option<String>,
) -> _ptr {
    let mut attrs = HashMap::new();
    attrs.insert("_external_name".to_string(), ConstValue::byte_str(name));
    functionptr_with_attrs(TYPE, name, None, _callable, attrs)
}

fn functionptr_with_attrs(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
    attrs: HashMap<String, ConstValue>,
) -> _ptr {
    _ptr::new(
        Ptr {
            TO: PtrTarget::Func(TYPE.clone()),
        },
        Ok(Some(_ptr_obj::Func(_func::new(
            TYPE,
            name.to_string(),
            graph,
            _callable,
            attrs,
        )))),
    )
}

pub fn build_number(_name: Option<()>, knowntype: KnownType) -> LowLevelType {
    match knowntype {
        KnownType::Int => LowLevelType::Signed,
        KnownType::Ruint => LowLevelType::Unsigned,
        KnownType::LongLong => LowLevelType::SignedLongLong,
        KnownType::LongLongLong => LowLevelType::SignedLongLongLong,
        KnownType::ULongLong => LowLevelType::UnsignedLongLong,
        KnownType::ULongLongLong => LowLevelType::UnsignedLongLongLong,
        other => panic!("lltype.build_number() does not support knowntype {other}"),
    }
}

impl LowLevelType {
    pub fn _gckind(&self) -> GcKind {
        match self {
            LowLevelType::Func(_) => GcKind::Raw,
            LowLevelType::Struct(t) => t._gckind,
            LowLevelType::Array(t) => t._gckind,
            LowLevelType::FixedSizeArray(t) => t._gckind,
            LowLevelType::Opaque(t) => t._gckind,
            LowLevelType::ForwardReference(t) => t._gckind,
            LowLevelType::Ptr(t) => t._gckind(),
            LowLevelType::InteriorPtr(t) => t.TO._gckind(),
            other => panic!("{other:?} is not a container type"),
        }
    }
}

impl PtrTarget {
    pub fn _gckind(&self) -> GcKind {
        match self {
            PtrTarget::Func(_) => GcKind::Raw,
            PtrTarget::Struct(t) => t._gckind,
            PtrTarget::Array(t) => t._gckind,
            PtrTarget::FixedSizeArray(t) => t._gckind,
            PtrTarget::Opaque(t) => t._gckind,
            PtrTarget::ForwardReference(t) => t._gckind,
        }
    }
}

pub fn _getconcretetype(v: &Hlvalue) -> Result<ConcretetypePlaceholder, TyperError> {
    // RPython `rtyper.py getcallable.getconcretetype` does not
    // invent a fallback type: the callable path obtains
    // `self.bindingrepr(v).lowleveltype`. This helper mirrors the raw
    // `v.concretetype` lookup used by lltype-level tests and fails when
    // the rtyper has not assigned a concrete type yet.
    let concretetype = match v {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    };
    concretetype.ok_or_else(|| TyperError::message("missing concretetype for getfunctionptr"))
}

pub fn getfunctionptr<F>(graph: &GraphRef, getconcretetype: F) -> Result<_ptr, TyperError>
where
    F: Fn(&Hlvalue) -> Result<ConcretetypePlaceholder, TyperError>,
{
    let ft = {
        let graph_b = graph.borrow();
        let mut llinputs = Vec::new();
        for arg in graph_b.getargs() {
            llinputs.push(getconcretetype(&arg)?);
        }
        let lloutput = getconcretetype(&graph_b.getreturnvar())?;
        FuncType {
            args: llinputs,
            result: lloutput,
        }
    };
    Ok(functionptr_for_graph_with_type(graph, ft))
}

/// Build a function pointer for `graph` from an explicitly supplied
/// `FuncType` instead of walking the graph's args for their (possibly
/// not-yet-assigned) concretetypes.
///
/// [`getfunctionptr`] delegates here after deriving the `FuncType` from
/// the rtyped args; the fn-const materialisation fallback
/// (`flowspace_adapter::translate_op`) calls this directly with the
/// callee's DECLARED LLBC signature when the callee's pygraph is lifted
/// but not yet rtyped (populate-time, before Phase B), where every arg's
/// `concretetype` is still `None`.  The name / `_callable` / attrs /
/// graph-backlink are resolved identically to `getfunctionptr`, so the
/// pointer differs from a rtyped one only in the arg/result placeholders.
pub fn functionptr_for_graph_with_type(graph: &GraphRef, TYPE: FuncType) -> _ptr {
    let graph_b = graph.borrow();
    let mut name = graph_b.name.clone();
    let mut callable = graph_b.func.as_ref().map(|func| func.name.clone());
    let mut attrs = HashMap::new();
    if let Some(func) = &graph_b.func {
        attrs = func._llfnobjattrs_.clone();
        if let Some(forced_name) = attrs.remove("_name").and_then(ConstValue::into_text) {
            name = forced_name;
        }
        if let Some(forced_callable) = attrs.remove("_callable") {
            callable = Some(match forced_callable {
                ConstValue::ByteStr(name) => {
                    String::from_utf8(name).unwrap_or_else(|err| format!("{:?}", err.into_bytes()))
                }
                ConstValue::UniStr(name) => name,
                ConstValue::Function(func) => func.name,
                ConstValue::HostObject(obj) => obj.qualname().to_string(),
                other => format!("{other:?}"),
            });
        }
    }
    drop(graph_b);
    functionptr_with_attrs(
        TYPE,
        &name,
        Some(GraphKey::of(graph).as_usize()),
        callable,
        attrs,
    )
}

/// RPython `class SomePtr(SomeObject)` (lltype.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomePtr {
    pub base: crate::annotator::model::SomeObject,
    pub ll_ptrtype: Ptr,
}

impl SomePtr {
    pub fn new(ll_ptrtype: Ptr) -> Self {
        SomePtr {
            base: crate::annotator::model::SomeObject::new(KnownType::LlPtr, true),
            ll_ptrtype,
        }
    }
}

impl crate::annotator::model::SomeObjectTrait for SomePtr {
    fn knowntype(&self) -> KnownType {
        KnownType::LlPtr
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, Constant, FunctionGraph, GraphFunc, Variable};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn runtime_type_info_singleton_is_raw_opaque_with_upstream_tag() {
        let LowLevelType::Opaque(inner) = &*RUNTIME_TYPE_INFO else {
            panic!("RUNTIME_TYPE_INFO must resolve to LowLevelType::Opaque");
        };
        assert_eq!(inner.tag, "RuntimeTypeInfo");
        assert_eq!(inner._gckind, GcKind::Raw);
    }

    fn fwd_ptr_to_named_struct(name: &str) -> LowLevelType {
        let body = Struct::gc_with_hints(name, vec![("hash".into(), LowLevelType::Signed)], vec![]);
        let fwd = ForwardReference::gc();
        fwd.r#become(LowLevelType::Struct(Box::new(body))).unwrap();
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd),
        }))
    }

    #[test]
    fn distinct_resolved_forward_reference_pointers_compare_unequal() {
        // rptr.py SomePtr.rtyper_makekey keys ll_ptrtype by value, so two
        // differently-named resolved forward-reference pointers must land in
        // distinct rtyper repr-cache buckets. `LowLevelType`'s structural
        // `Eq` (lltype.py `__eq__`) resolves each forward reference
        // through `become()` before comparing `__dict__`.
        let a = fwd_ptr_to_named_struct("rpy_a");
        let b = fwd_ptr_to_named_struct("rpy_b");
        assert_ne!(a, b);
    }

    #[test]
    fn runtime_type_info_is_stable_across_lookups() {
        let first: *const LowLevelType = &*RUNTIME_TYPE_INFO;
        let second: *const LowLevelType = &*RUNTIME_TYPE_INFO;
        assert_eq!(first, second);
    }

    #[test]
    fn ptr_from_container_type_packs_struct_into_ptr_target_struct() {
        let s = Struct::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = Ptr::from_container_type(T).unwrap();
        assert!(matches!(p.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn ptr_from_container_type_rejects_primitive_types() {
        let err = Ptr::from_container_type(LowLevelType::Signed).unwrap_err();
        assert!(
            err.contains("can only point to a Container type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nullptr_of_runtime_type_info_is_null_ptr_to_opaque_target() {
        let p = nullptr(RUNTIME_TYPE_INFO.clone()).unwrap();
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        assert!(matches!(&p._obj0, PtrObj::Null));
    }

    #[test]
    fn nullptr_rejects_non_container_type() {
        let err = nullptr(LowLevelType::Float).unwrap_err();
        assert!(
            err.contains("can only point to a Container type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cast_pointer_identity_returns_same_pointer_value() {
        let s = Struct::_build(
            "vtable",
            vec![("super".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = malloc(T.clone(), None, MallocFlavor::Gc, true).unwrap();
        let ptr_T = Ptr::from_container_type(T).unwrap();
        let q = cast_pointer(&ptr_T, &p).unwrap();
        // Identity cast preserves both _identity (same allocation) and
        // _TYPE.
        assert_eq!(p._hashable_identity(), q._hashable_identity());
    }

    #[test]
    fn cast_pointer_downcast_walks_first_field_super_chain() {
        // Build a parent gc struct (rtti=true by `gc_rtti`) and a
        // sub-struct whose first field is `("super", parent)`. cast
        // from sub→parent should yield a pointer whose _TYPE matches
        // parent and which aliases the same allocation.
        let parent = Struct::gc_rtti("parent", vec![("typeptr".into(), LowLevelType::Signed)]);
        let parent_T = LowLevelType::Struct(Box::new(parent.clone()));
        let sub = Struct::gc_rtti(
            "sub",
            vec![
                ("super".into(), parent_T.clone()),
                ("data".into(), LowLevelType::Signed),
            ],
        );
        let sub_T = LowLevelType::Struct(Box::new(sub));
        let sub_ptr = malloc(sub_T, None, MallocFlavor::Gc, true).unwrap();

        let parent_ptr_T = Ptr::from_container_type(parent_T).unwrap();
        let cast = cast_pointer(&parent_ptr_T, &sub_ptr).unwrap();
        // The cast result's `_TYPE` is `Ptr(parent)`.
        let PtrTarget::Struct(s) = &cast._TYPE.TO else {
            panic!("cast result must be Ptr(Struct)");
        };
        assert_eq!(s._name, "parent");
    }

    #[test]
    fn cast_pointer_null_yields_null_of_target_type() {
        let parent = Struct::gc_rtti("parent_n", vec![]);
        let parent_T = LowLevelType::Struct(Box::new(parent));
        let parent_ptr_T = Ptr::from_container_type(parent_T.clone()).unwrap();
        let null = nullptr(parent_T).unwrap();
        let cast = cast_pointer(&parent_ptr_T, &null).unwrap();
        assert!(!cast.nonzero(), "null cast must remain null");
    }

    #[test]
    fn malloc_immortal_gc_struct_produces_live_struct_container() {
        let s = Struct::_build(
            "vtable",
            vec![("super".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = malloc(T, None, MallocFlavor::Gc, true).unwrap();
        let Ok(Some(_ptr_obj::Struct(inner))) = p._obj0_value() else {
            panic!("malloc(Struct, immortal=true) must produce Struct container");
        };
        assert_eq!(inner.TYPE._name, "vtable");
        assert!(inner._getattr("super").is_some());
    }

    #[test]
    fn malloc_immortal_gc_opaque_allocates_opaque_container() {
        let T = RUNTIME_TYPE_INFO.clone();
        let p = malloc(T, None, MallocFlavor::Gc, true).unwrap();
        assert!(matches!(p._obj0_value(), Ok(Some(_ptr_obj::Opaque(_)))));
        assert!(p._solid);
    }

    #[test]
    fn malloc_rejects_non_container_type() {
        let err = malloc(LowLevelType::Signed, None, MallocFlavor::Gc, true).unwrap_err();
        assert!(err.contains("unmallocable type"), "unexpected error: {err}");
    }

    #[test]
    fn malloc_varsize_struct_initialises_trailing_array_to_requested_length() {
        let s = Struct::_build(
            "S",
            vec![
                ("x".into(), LowLevelType::Signed),
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(Array::new(LowLevelType::Char))),
                ),
            ],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            Some(3),
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let Ok(Some(_ptr_obj::Struct(s))) = p._obj0_value() else {
            panic!("malloc should return a live Struct pointer");
        };
        let items_value = s
            ._getattr("items")
            .expect("varsize Struct must expose 'items' field");
        let LowLevelValue::Array(items) = items_value else {
            panic!("varsize array field should be initialised as an Array");
        };
        assert_eq!(items.getlength(), 3);
    }

    #[test]
    fn malloc_rejects_gc_flavor_non_immortal_on_non_gc_struct() {
        let s = Struct::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![],
            vec![],
            true,
        );
        let err = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            false,
        )
        .unwrap_err();
        assert!(
            err.contains("gc flavor malloc of a non-GC"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn public_type_helpers_follow_upstream_names() {
        assert!(safe_equal(&LowLevelType::Signed, &LowLevelType::Signed));
        assert!(isCompatibleType(
            &LowLevelType::Signed,
            &LowLevelType::Signed
        ));
        assert!(!isCompatibleType(
            &LowLevelType::Signed,
            &LowLevelType::Bool
        ));

        let value = LowLevelValue::Signed(7);
        assert_eq!(
            enforce(&LowLevelType::Signed, value.clone()).unwrap(),
            value
        );
        assert!(enforce(&LowLevelType::Bool, LowLevelValue::Signed(7)).is_err());
    }

    #[test]
    fn normalizeptr_null_pointer_returns_none() {
        let T = LowLevelType::Struct(Box::new(Struct::new(
            "thing",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let p = nullptr(T).unwrap();
        assert!(normalizeptr(&p, true).unwrap().is_none());
    }

    #[test]
    fn free_top_level_surface_marks_raw_container_freed() {
        let s = Struct::new("thing", vec![("x".into(), LowLevelType::Signed)]);
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap();
        free(&p, "raw", true).unwrap();
        assert!(p._was_freed().unwrap());
    }

    #[test]
    fn runtime_type_info_top_level_surface_returns_attached_rtti() {
        let s = Struct::gc_rtti("R", vec![("x".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let expected = getRuntimeTypeInfo(&T).unwrap();
        let p = malloc(T, None, MallocFlavor::Gc, true).unwrap();
        assert_eq!(runtime_type_info(&p).unwrap(), expected);
    }

    #[test]
    fn opaqueptr_mints_named_opaque_container_and_wraps_in_ptr() {
        let p = opaqueptr(RUNTIME_TYPE_INFO.clone(), "ExceptionFoo").unwrap();
        let Ok(Some(_ptr_obj::Opaque(inner))) = p._obj0_value() else {
            panic!("opaqueptr must produce an Opaque container");
        };
        assert_eq!(inner._name.as_deref(), Some("ExceptionFoo"));
        assert!(inner.about.is_none());
        assert_eq!(inner.TYPE.tag, "RuntimeTypeInfo");
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        assert!(p._solid);
    }

    #[test]
    fn opaqueptr_rejects_non_opaque_type() {
        let err = opaqueptr(LowLevelType::Signed, "x").unwrap_err();
        assert!(
            err.contains("opaqueptr() for OpaqueTypes only"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn signed_contains_address_offset_symbolic() {
        let offset = ConstValue::AddressOffset(
            crate::translator::rtyper::lltypesystem::llmemory::AddressOffset::ItemOffset(
                crate::translator::rtyper::lltypesystem::llmemory::ItemOffset {
                    TYPE: LowLevelType::Signed,
                    repeat: 1,
                },
            ),
        );
        assert!(LowLevelType::Signed.contains_value(&offset));
        assert!(!LowLevelType::Unsigned.contains_value(&offset));
    }

    #[test]
    fn gc_rtti_struct_has_runtime_type_info_opaque_named_after_struct() {
        let s = Struct::gc_rtti("ExceptionFoo", vec![("msg".into(), LowLevelType::Signed)]);
        let rtti = s
            ._runtime_type_info
            .as_ref()
            .expect("gc_rtti must attach _runtime_type_info");
        assert_eq!(rtti.TYPE.tag, "RuntimeTypeInfo");
        assert_eq!(rtti._name.as_deref(), Some("ExceptionFoo"));
        let Some(LowLevelType::Struct(about)) = rtti.about.as_ref() else {
            panic!("gc_rtti must record about=self on the opaque");
        };
        assert_eq!(about._name, "ExceptionFoo");
    }

    #[test]
    fn gc_struct_without_rtti_leaves_runtime_type_info_none() {
        let s = Struct::gc("PlainStruct", vec![("x".into(), LowLevelType::Signed)]);
        assert!(s._runtime_type_info.is_none());
    }

    #[test]
    fn get_runtime_type_info_returns_ptr_to_attached_opaque() {
        let s = Struct::gc_rtti("ExceptionBar", vec![("msg".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let p = getRuntimeTypeInfo(&T).unwrap();
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        let Ok(Some(_ptr_obj::Opaque(inner))) = p._obj0_value() else {
            panic!("getRuntimeTypeInfo must produce an Opaque container");
        };
        assert_eq!(inner._name.as_deref(), Some("ExceptionBar"));
    }

    #[test]
    fn get_runtime_type_info_errors_when_struct_lacks_rtti() {
        let s = Struct::gc("NoRttiStruct", vec![("x".into(), LowLevelType::Signed)]);
        let err = getRuntimeTypeInfo(&LowLevelType::Struct(Box::new(s))).unwrap_err();
        assert!(
            err.contains("no attached runtime type info"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn get_runtime_type_info_rejects_raw_structs() {
        let s = Struct::new("RawStruct", vec![("x".into(), LowLevelType::Signed)]);
        let err = getRuntimeTypeInfo(&LowLevelType::Struct(Box::new(s))).unwrap_err();
        assert!(
            err.contains("expected a RttiStruct"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn get_runtime_type_info_errors_on_non_struct_type() {
        let err = getRuntimeTypeInfo(&LowLevelType::Signed).unwrap_err();
        assert!(
            err.contains("expected a RttiStruct"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn attach_runtime_type_info_returns_same_opaque_as_get() {
        let s = Struct::gc_rtti("AttachedStruct", vec![("x".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let from_attach = attachRuntimeTypeInfo(&T).unwrap();
        let from_get = getRuntimeTypeInfo(&T).unwrap();
        let (Ok(Some(_ptr_obj::Opaque(a))), Ok(Some(_ptr_obj::Opaque(b)))) =
            (from_attach._obj0_value(), from_get._obj0_value())
        else {
            panic!("both helpers must produce Opaque containers");
        };
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a._name, b._name);
    }

    #[test]
    fn attach_runtime_type_info_errors_when_gc_struct_lacks_rtti() {
        let s = Struct::gc("AttachedStruct", vec![("x".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let err = attachRuntimeTypeInfo(&T).unwrap_err();
        assert!(
            err.contains("must have been built with the rtti=True argument"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn attach_runtime_type_info_with_ptrs_stores_query_and_destructor() {
        let mut T = LowLevelType::Struct(Box::new(Struct::gc_rtti(
            "AttachedStruct",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let self_ptr_t = Ptr::from_container_type(T.clone()).unwrap();
        let rtti_ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone()).unwrap();
        let query = functionptr(
            FuncType {
                args: vec![LowLevelType::Ptr(Box::new(self_ptr_t.clone()))],
                result: LowLevelType::Ptr(Box::new(rtti_ptr_t)),
            },
            "query",
            None,
            Some("<query>".into()),
        );
        let destr = functionptr(
            FuncType {
                args: vec![LowLevelType::Ptr(Box::new(self_ptr_t))],
                result: LowLevelType::Void,
            },
            "destr",
            None,
            Some("<destr>".into()),
        );
        let attached =
            attachRuntimeTypeInfo_with_ptrs(&mut T, Some(query.clone()), Some(destr.clone()))
                .unwrap();
        let LowLevelType::Struct(struct_t) = &T else {
            panic!("attachRuntimeTypeInfo_with_ptrs must keep T as a struct");
        };
        let rtti = struct_t
            ._runtime_type_info
            .as_ref()
            .expect("rtti opaque must still be present");
        assert_eq!(
            rtti.query_funcptr
                .lock()
                .as_ref()
                .map(|ptr| ptr._hashable_identity()),
            Some(query._hashable_identity())
        );
        assert_eq!(
            rtti.destructor_funcptr
                .lock()
                .as_ref()
                .map(|ptr| ptr._hashable_identity()),
            Some(destr._hashable_identity())
        );
        let Ok(Some(_ptr_obj::Opaque(attached_rtti))) = attached._obj0_value() else {
            panic!("attachRuntimeTypeInfo_with_ptrs must return the RTTI opaque pointer");
        };
        assert!(attached_rtti.query_funcptr.lock().is_some());
        assert!(attached_rtti.destructor_funcptr.lock().is_some());
    }

    #[test]
    fn functionptr_keeps_graph_on_funcobj() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = getfunctionptr(&graph, _getconcretetype).unwrap();
        let funcobj = ptr._obj().unwrap();
        let _ptr_obj::Func(funcobj) = funcobj else {
            panic!("functionptr must expose a function object");
        };
        assert_eq!(funcobj.graph, Some(GraphKey::of(&graph).as_usize()));
    }

    #[test]
    fn getfunctionptr_calls_getconcretetype_for_args_and_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn counting_getconcretetype(v: &Hlvalue) -> Result<ConcretetypePlaceholder, TyperError> {
            let _ = v;
            CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(LowLevelType::Void)
        }

        let a0 = Variable::new();
        a0.set_concretetype(Some(LowLevelType::Signed));
        let a1 = Variable::new();
        a1.set_concretetype(Some(LowLevelType::Bool));
        let ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let start = Rc::new(RefCell::new(Block::new(vec![
            Hlvalue::Variable(a0),
            Hlvalue::Variable(a1),
        ])));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        CALLS.store(0, Ordering::Relaxed);

        let _ = getfunctionptr(&graph, counting_getconcretetype).unwrap();

        assert_eq!(CALLS.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn getfunctionptr_rejects_missing_concretetype_instead_of_void_fallback() {
        let arg = Hlvalue::Variable(crate::flowspace::model::Variable::new());
        let start = Rc::new(RefCell::new(Block::new(vec![arg])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));

        assert!(getfunctionptr(&graph, _getconcretetype).is_err());
    }

    #[test]
    fn getfunctionptr_copies_llfnobjattrs_from_graph_func() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "graph_name",
            start,
            Hlvalue::Variable(ret),
        )));
        let mut func = GraphFunc::new(
            "py_func",
            Constant::new(ConstValue::Dict(std::sync::Arc::new(HashMap::new()))),
        );
        func._llfnobjattrs_
            .insert("_name".to_string(), ConstValue::byte_str("forced_name"));
        func._llfnobjattrs_.insert(
            "_callable".to_string(),
            ConstValue::byte_str("forced_callable"),
        );
        func._llfnobjattrs_
            .insert("extra".to_string(), ConstValue::Int(7));
        graph.borrow_mut().func = Some(func);

        let ptr = getfunctionptr(&graph, _getconcretetype).unwrap();
        let _ptr_obj::Func(funcobj) = ptr._obj().unwrap() else {
            panic!("functionptr must expose a function object");
        };
        assert_eq!(funcobj._name, "forced_name");
        assert_eq!(funcobj._callable.as_deref(), Some("forced_callable"));
        assert_eq!(funcobj.attrs.get("extra"), Some(&ConstValue::Int(7)));
    }

    #[test]
    fn lowleveltype_primitive_contains_value_matches_upstream_enforce() {
        use crate::flowspace::model::ConstValue;

        // RPython `LowLevelType._contains_value` (`lltype.py`):
        // Void admits any value as a valid constant. Non-Void primitives
        // still dispatch through isCompatibleType.
        assert!(LowLevelType::Void.contains_value(&ConstValue::None));
        assert!(LowLevelType::Void.contains_value(&ConstValue::Bool(true)));
        assert!(LowLevelType::Void.contains_value(&ConstValue::Int(7)));

        // RPython `Bool` Primitive accepts bools.
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(false)));
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(true)));
        assert!(!LowLevelType::Bool.contains_value(&ConstValue::Int(0)));

        // RPython `Signed` / `SignedLongLong` accept Python `int`
        // (`typeOf(42) == Signed`; `SignedLongLong is Signed` on 64-bit).
        // `Unsigned` / `UnsignedLongLong` accept `r_uint`
        // (`typeOf(r_uint(42)) == Unsigned`).
        assert!(LowLevelType::Signed.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::Unsigned.contains_value(&ConstValue::UInt(42)));
        assert!(!LowLevelType::Unsigned.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::UnsignedLongLong.contains_value(&ConstValue::UInt(42)));
        assert!(LowLevelType::SignedLongLong.contains_value(&ConstValue::Int(42)));
        assert!(!LowLevelType::SignedLongLong.contains_value(&ConstValue::UInt(42)));
        assert!(LowLevelType::SignedLongLongLong.contains_value(&ConstValue::Int128(i128::MIN)));
        assert!(LowLevelType::UnsignedLongLongLong.contains_value(&ConstValue::UInt128(u128::MAX)));
        assert!(!LowLevelType::UnsignedLongLongLong.contains_value(&ConstValue::UInt(1)));
        assert!(!LowLevelType::SignedLongLongLong.contains_value(&ConstValue::UInt128(u128::MAX)));
        assert!(!LowLevelType::UnsignedLongLongLong.contains_value(&ConstValue::Int128(i128::MIN)));
        assert!(!LowLevelType::Signed.contains_value(&ConstValue::Bool(true)));

        // RPython `Float` / `SingleFloat` / `LongFloat` accept Python float.
        // pyre stores floats as u64 bit-patterns in ConstValue::Float.
        assert!(LowLevelType::Float.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::SingleFloat.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::LongFloat.contains_value(&ConstValue::Float(0)));
        assert!(!LowLevelType::Float.contains_value(&ConstValue::Int(0)));
    }

    #[test]
    fn lowleveltype_char_unichar_contains_value_enforces_type_and_single_char_length() {
        use crate::flowspace::model::ConstValue;
        assert!(LowLevelType::Char.contains_value(&ConstValue::byte_str(b"a")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::byte_str(b"ab")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::byte_str(b"")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::uni_str("a")));

        assert!(LowLevelType::UniChar.contains_value(&ConstValue::uni_str("a")));
        assert!(LowLevelType::UniChar.contains_value(&ConstValue::uni_str("π")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::uni_str("πi")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::uni_str("")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::byte_str(b"a")));

        // Non-string variants stay rejected.
        assert!(!LowLevelType::Char.contains_value(&ConstValue::Int(0)));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::Bool(false)));
    }

    #[test]
    fn lowleveltype_placeholder_value_is_universally_accepted() {
        use crate::flowspace::model::ConstValue;

        // Placeholder sentinel (`description.NODEFAULT` upstream) must
        // pass `_contains_value` so the normalizecalls rewrite branch
        // can stash it as a transient row-level padding without
        // tripping convert_const validation. See rmodel.rs's
        // inputconst port for the load-bearing use.
        for lltype in [
            LowLevelType::Void,
            LowLevelType::Bool,
            LowLevelType::Signed,
            LowLevelType::Float,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::Ptr(Box::new(Ptr {
                TO: PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            })),
        ] {
            assert!(
                lltype.contains_value(&ConstValue::Placeholder),
                "Placeholder must be universally acceptable (lltype={lltype:?})"
            );
        }
    }

    #[test]
    fn lowleveltype_primitive_short_name_matches_upstream_class_name() {
        // rmodel.py `<%s %s>` formatter and rmodel.py
        // `compact_repr` both consume `lowleveltype._short_name()`
        // (Primitive) or `lowleveltype.__name__` (Ptr). Lock in the
        // upstream strings.
        assert_eq!(LowLevelType::Void.short_name(), "Void");
        assert_eq!(LowLevelType::Bool.short_name(), "Bool");
        assert_eq!(LowLevelType::Signed.short_name(), "Signed");
        assert_eq!(LowLevelType::Unsigned.short_name(), "Unsigned");
        assert_eq!(LowLevelType::SignedLongLong.short_name(), "SignedLongLong");
        assert_eq!(
            LowLevelType::UnsignedLongLong.short_name(),
            "UnsignedLongLong"
        );
        assert_eq!(LowLevelType::Float.short_name(), "Float");
        assert_eq!(LowLevelType::SingleFloat.short_name(), "SingleFloat");
        assert_eq!(LowLevelType::LongFloat.short_name(), "LongFloat");
        assert_eq!(LowLevelType::Char.short_name(), "Char");
        assert_eq!(LowLevelType::UniChar.short_name(), "UniChar");
        assert_eq!(LowLevelType::Address.short_name(), "Address");
    }

    #[test]
    fn lowleveltype_address_is_primitive_atomic_and_equal_to_self() {
        // rpython/rtyper/lltypesystem/llmemory.py:650 —
        // `Address = lltype.Primitive("Address", NULL)`. Address must
        // behave as an atomic primitive (is_primitive / _is_atomic) and
        // its default value is the NULL sentinel.
        assert!(LowLevelType::Address.is_primitive());
        assert!(LowLevelType::Address._is_atomic());
        assert_eq!(LowLevelType::Address, LowLevelType::Address);
        assert_ne!(LowLevelType::Address, LowLevelType::Signed);
    }

    #[test]
    fn lowleveltype_address_defl_returns_null_sentinel() {
        // upstream `NULL = fakeaddress(None)`; `Address._defl()` returns
        // NULL. Pyre models the NULL sentinel as `_address::Null` until
        // richer fakeaddress bodies land.
        match LowLevelType::Address._defl() {
            LowLevelValue::Address(_address::Null) => {}
            other => panic!("expected NULL address, got {other:?}"),
        }
    }

    #[test]
    fn typeof_value_address_round_trips_to_address_lowleveltype() {
        let v = LowLevelValue::Address(_address::Null);
        assert_eq!(typeOf_value(&v), LowLevelType::Address);
    }

    #[test]
    fn lowleveltype_address_contains_value_accepts_fakeaddress_constants() {
        use crate::flowspace::model::ConstValue;
        assert!(!LowLevelType::Address.contains_value(&ConstValue::Int(0)));
        assert!(!LowLevelType::Address.contains_value(&ConstValue::None));
        assert!(
            LowLevelType::Address.contains_value(&ConstValue::LLAddress(_address::Null)),
            "Address must accept llmemory.NULL fakeaddress"
        );
        assert!(LowLevelType::Address.contains_value(&ConstValue::Placeholder));
    }

    #[test]
    fn lowleveltype_ptr_short_name_follows_upstream_prefix() {
        // upstream `Ptr._short_name` (`lltype.py`):
        // `'Ptr %s' % self.TO._short_name()`. Note this is distinct
        // from `Ptr.__str__` which uses `"* %s"` prefix.
        let ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Void, LowLevelType::Void],
                result: LowLevelType::Void,
            }),
        }));
        assert_eq!(ptr.short_name(), "Ptr Func(Void, Void)->Void");
    }

    #[test]
    fn lowleveltype_module_constants_match_variant_singletons() {
        // `VOID` / `BOOL` / `SIGNED` / `FLOAT` / `CHAR` / `UNICHAR` are
        // the pyre re-exports of upstream's `lltype.Void` / `lltype.Bool`
        // / ... (which are instance singletons). Lock the identity so
        // downstream `use lltype::{Void, Bool, ...}` imports see a
        // stable match source.
        assert_eq!(VOID, LowLevelType::Void);
        assert_eq!(BOOL, LowLevelType::Bool);
        assert_eq!(SIGNED, LowLevelType::Signed);
        assert_eq!(FLOAT, LowLevelType::Float);
        assert_eq!(CHAR, LowLevelType::Char);
        assert_eq!(UNICHAR, LowLevelType::UniChar);
    }

    #[test]
    fn delayed_pointer_raises_on_obj_access() {
        let ptr = _ptr::new(
            Ptr {
                TO: PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            },
            Err(DelayedPointer),
        );
        assert_eq!(ptr._obj(), Err(DelayedPointer));
    }

    #[test]
    fn delayed_pointer_equality_requires_same_ptr_instance() {
        let delayed1 = _ptr::new(
            Ptr {
                TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
            },
            Err(DelayedPointer),
        );
        let delayed2 = _ptr::new(delayed1._TYPE.clone(), Err(DelayedPointer));
        assert!(delayed1 == delayed1);
        assert!(delayed1 != delayed2);
    }

    #[test]
    #[should_panic(expected = "comparing")]
    fn ptr_equality_rejects_different_pointer_types() {
        let left = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let right = Ptr {
            TO: PtrTarget::Struct(Struct::new("T", vec![("y".into(), LowLevelType::Signed)])),
        }
        ._example();
        let _ = left == right;
    }

    #[test]
    fn ptr_needsgc_tracks_target_gckind() {
        let raw_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        assert!(!raw_ptr._needsgc());

        let gc_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        assert!(gc_ptr._needsgc());
    }

    #[test]
    fn gc_ptr_identityhash_tracks_underlying_object_identity() {
        let ptr1 = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let ptr2 = _ptr {
            _identity: ptr1._identity,
            _TYPE: ptr1._TYPE.clone(),
            _solid: ptr1._solid,
            _obj0: ptr1._obj0.clone(),
        };
        let ptr3 = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        assert_eq!(identityhash(&ptr1), identityhash(&ptr2));
        assert_ne!(identityhash(&ptr1), identityhash(&ptr3));
    }

    #[test]
    #[should_panic]
    fn raw_ptr_identityhash_rejects_non_gc_pointer() {
        let ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let _ = identityhash(&ptr);
    }

    #[test]
    fn ptr_same_obj_requires_same_underlying_object() {
        let ptr1 = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let ptr2 = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        assert!(!ptr1._same_obj(&ptr2).unwrap());
        assert!(ptr1._same_obj(&ptr1).unwrap());
    }

    #[test]
    fn interior_ptr_same_obj_compares_current_target_object() {
        let iptr1 = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let iptr2 = iptr1.clone();
        assert!(iptr1._same_obj(&iptr2));
    }

    #[test]
    fn interior_ptr_equality_matches_current_target_object() {
        let iptr1 = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let iptr2 = iptr1.clone();
        assert!(iptr1 == iptr2);
    }

    #[test]
    #[should_panic(expected = "comparing")]
    fn interior_ptr_equality_rejects_different_pointer_types() {
        let left = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let right = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new("S", vec![("y".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("y".into())],
        };
        let _ = left == right;
    }

    #[test]
    fn forward_reference_become_rejects_non_container() {
        let forward_ref = ForwardReference::new();
        let err = forward_ref
            .r#become(LowLevelType::Signed)
            .expect_err("ForwardReference must reject non-container targets");
        assert!(err.contains("can only be to a container"));
    }

    #[test]
    fn forward_reference_become_rejects_conflicting_gckind() {
        let forward_ref = ForwardReference::gc();
        let err = forward_ref
            .r#become(LowLevelType::Struct(Box::new(Struct::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .expect_err("GcForwardReference must reject raw targets");
        assert!(err.contains("conflicting gckind"));
    }

    #[test]
    fn forward_reference_become_allows_resolved_struct_example() {
        let forward_ref = ForwardReference::new();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(Struct::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .unwrap();
        let ptr = Ptr {
            TO: PtrTarget::ForwardReference(forward_ref),
        }
        ._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn forward_reference_clones_observe_become() {
        // lltype.py mutates the ForwardReference object in place;
        // all aliases must observe the resolved container.
        let forward_ref = ForwardReference::new();
        let alias = forward_ref.clone();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(Struct::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .unwrap();

        let ptr = Ptr {
            TO: PtrTarget::ForwardReference(alias),
        }
        ._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn resolved_forward_reference_compares_as_real_container() {
        // lltype.py rebinds __class__/__dict__; after become(),
        // the forward reference participates in LowLevelType equality/hash
        // as the real container type.
        let real = LowLevelType::Struct(Box::new(Struct::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let forward_ref = ForwardReference::new();
        forward_ref.r#become(real.clone()).unwrap();
        let resolved = LowLevelType::ForwardReference(Box::new(forward_ref));

        assert_eq!(resolved, real);

        let mut left = std::collections::hash_map::DefaultHasher::new();
        resolved.hash(&mut left);
        let mut right = std::collections::hash_map::DefaultHasher::new();
        real.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    fn cyclic_forward_reference_hash_uses_zero_on_re_entry_so_isomorphic_cycles_match() {
        // Companion to the eq cycle test: hash and eq must agree
        // (Eq+Hash contract). Two distinct ForwardReferences with
        // self-referential isomorphic Struct shapes compare equal,
        // therefore must hash equal. RPython `saferecursive(get_hash,
        // 0)` (lltype.py) yields 0 on re-entry; hashing the Arc
        // identity instead would diverge per allocation.
        let fwd_a = ForwardReference::gc();
        let s_a = Struct::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_a.clone()),
                })),
            )],
        );
        fwd_a.r#become(LowLevelType::Struct(Box::new(s_a))).unwrap();

        let fwd_b = ForwardReference::gc();
        let s_b = Struct::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_b.clone()),
                })),
            )],
        );
        fwd_b.r#become(LowLevelType::Struct(Box::new(s_b))).unwrap();

        assert!(!Arc::ptr_eq(&fwd_a.target, &fwd_b.target));
        let lhs = LowLevelType::ForwardReference(Box::new(fwd_a));
        let rhs = LowLevelType::ForwardReference(Box::new(fwd_b));

        let mut left_hasher = std::collections::hash_map::DefaultHasher::new();
        lhs.hash(&mut left_hasher);
        let mut right_hasher = std::collections::hash_map::DefaultHasher::new();
        rhs.hash(&mut right_hasher);
        assert_eq!(left_hasher.finish(), right_hasher.finish());
    }

    #[test]
    fn debug_of_recursive_struct_type_terminates() {
        // A recursive struct type — a struct with a `Ptr` field pointing
        // back to itself, e.g. `pyframe::PyFrame` — is legal. Formatting
        // its `LowLevelType` must terminate instead of recursing forever:
        // `Struct`'s `Debug` prints field NAMES only (`Struct.__str__`
        // short form, lltype.py), so it never descends into
        // `f_back`'s self-referential pointer type.
        let fwd = ForwardReference::gc();
        let s = Struct::gc(
            "PyFrame",
            vec![(
                "f_back".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            )],
        );
        fwd.r#become(LowLevelType::Struct(Box::new(s.clone())))
            .unwrap();

        // Must format without overflowing the stack, both directly and
        // through the resolved forward reference.
        let via_struct = format!("{:?}", LowLevelType::Struct(Box::new(s)));
        // `Struct::gc` is gc-kinded, so the class name is `GcStruct`
        // (upstream `self.__class__.__name__`), not the base `Struct`.
        assert!(via_struct.contains("GcStruct PyFrame"), "{via_struct}");
        assert!(via_struct.contains("f_back"), "{via_struct}");

        let via_fwd = format!("{:?}", LowLevelType::ForwardReference(Box::new(fwd)));
        assert!(via_fwd.contains("PyFrame"), "{via_fwd}");
    }

    #[test]
    fn cyclic_forward_reference_equality_short_circuits_to_true_on_re_entry() {
        // Two distinct ForwardReferences fwd_a and fwd_b, each
        // resolved to a Struct containing a Ptr looping back to
        // itself. Comparing them recursively re-enters the same
        // ForwardReference comparison; lltype.py
        // `saferecursive(safe_equal, True)` short-circuits to True
        // on re-entry so structurally identical cyclic types compare
        // equal. Returning False there would propagate up through
        // the Struct field comparison and report unequal.
        let fwd_a = ForwardReference::gc();
        let s_a = Struct::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_a.clone()),
                })),
            )],
        );
        fwd_a.r#become(LowLevelType::Struct(Box::new(s_a))).unwrap();

        let fwd_b = ForwardReference::gc();
        let s_b = Struct::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_b.clone()),
                })),
            )],
        );
        fwd_b.r#become(LowLevelType::Struct(Box::new(s_b))).unwrap();

        assert!(!Arc::ptr_eq(&fwd_a.target, &fwd_b.target));
        let lhs = LowLevelType::ForwardReference(Box::new(fwd_a));
        let rhs = LowLevelType::ForwardReference(Box::new(fwd_b));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ptr_to_resolved_forward_reference_compares_as_ptr_to_real_container() {
        let real_struct = Struct::new("S", vec![("x".into(), LowLevelType::Signed)]);
        let forward_ref = ForwardReference::new();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(real_struct.clone())))
            .unwrap();

        let through_forward_ref = Ptr {
            TO: PtrTarget::ForwardReference(forward_ref),
        };
        let direct = Ptr {
            TO: PtrTarget::Struct(real_struct),
        };

        assert_eq!(through_forward_ref, direct);

        let mut left = std::collections::hash_map::DefaultHasher::new();
        through_forward_ref.hash(&mut left);
        let mut right = std::collections::hash_map::DefaultHasher::new();
        direct.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    #[should_panic(expected = "ForwardReference object is not hashable")]
    fn unresolved_forward_reference_hash_panics() {
        // lltype.py — unresolved ForwardReference.__hash__ raises.
        let forward_ref = ForwardReference::new();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        forward_ref.hash(&mut hasher);
    }

    #[test]
    #[should_panic(expected = "_ptr._become: type mismatch")]
    fn ptr_become_panics_on_type_mismatch() {
        // lltype.py — `assert self._TYPE == other._TYPE` in `_become`.
        let self_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("A", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let other_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("B", vec![("y".into(), LowLevelType::Signed)])),
        }
        ._example();
        self_ptr._become(&other_ptr);
    }

    #[test]
    #[should_panic(expected = "_ptr._become: cannot reassign a weak pointer")]
    fn ptr_become_panics_when_self_is_weak() {
        // lltype.py — `assert not self._weak` in `_become`.
        let mut self_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        // Force `self_ptr` into the weak `_obj0` form. `_keep` retains a
        // strong handle so the downgrade has a live referent.
        let _keep = self_ptr._obj0_value().unwrap().unwrap();
        self_ptr._obj0 = PtrObj::Weak(WeakContainer::downgrade(&_keep).unwrap());
        let other_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        self_ptr._become(&other_ptr);
    }

    #[test]
    fn struct_pointer_example_exposes_field_defaults() {
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn struct_pointer_lookup_adtmeth_binds_to_same_ptrtype() {
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::with_adtmeths(
                "S",
                vec![],
                vec![(
                    "f".into(),
                    ConstValue::HostObject(crate::flowspace::model::HostObject::new_user_function(
                        crate::flowspace::model::GraphFunc::new(
                            "f",
                            crate::flowspace::model::Constant::new(ConstValue::Dict(
                                Default::default(),
                            )),
                        ),
                    )),
                )],
            )),
        };
        let ptr = ptr_t._example();
        let LowLevelAdtMember::Method { ll_ptrtype, .. } = ptr._lookup_adtmeth("f").unwrap() else {
            panic!("expected bound adt method");
        };
        assert_eq!(ll_ptrtype, LowLevelPointerType::Ptr(ptr_t));
    }

    #[test]
    fn struct_pointer_getattr_exposes_inlined_struct_field_as_pointer() {
        let inner = Struct::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Ptr {
            TO: PtrTarget::Struct(Struct::new(
                "Outer",
                vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::Ptr(inner_ptr) = value else {
            panic!("expected nested struct field to be exposed as a pointer");
        };
        assert_eq!(inner_ptr._TYPE.TO, PtrTarget::Struct(inner));
    }

    #[test]
    fn gc_struct_pointer_getattr_exposes_raw_struct_field_as_interior_ptr() {
        let inner = Struct::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Ptr {
            TO: PtrTarget::Struct(Struct::gc(
                "Outer",
                vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::InteriorPtr(inner_ptr) = value else {
            panic!("expected raw field of gc struct to be exposed as an interior ptr");
        };
        assert_eq!(
            inner_ptr._TYPE(),
            InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(Struct::gc(
                    "Outer",
                    vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
                ),))),
                TO: Box::new(LowLevelType::Struct(Box::new(inner))),
                offsets: vec![InteriorOffset::Field("x".into())],
            }
        );
    }

    #[test]
    fn array_pointer_len_and_getitem_follow_array_surface() {
        let ptr_t = Ptr {
            TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.len().unwrap(), 1);
        assert_eq!(ptr.getitem(0).unwrap(), LowLevelValue::Signed(0));
        assert_eq!(ptr._fixedlength(), Ok(None));
    }

    #[test]
    fn gc_array_pointer_getitem_exposes_raw_struct_item_as_interior_ptr() {
        let ptr_t = Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(Struct::new(
                "Item",
                vec![("x".into(), LowLevelType::Signed)],
            ))))),
        };
        let value = ptr_t._example().getitem(0).unwrap();
        let LowLevelValue::InteriorPtr(iptr) = value else {
            panic!("expected gc array item to be exposed as an interior ptr");
        };
        assert_eq!(iptr._offsets, vec![InteriorOffset::Index(0)]);
    }

    #[test]
    fn gc_struct_pointer_getattr_exposes_opaque_field_as_ptr_not_interior_ptr() {
        let outer = Ptr {
            TO: PtrTarget::Struct(Struct::gc(
                "Outer",
                vec![(
                    "x".into(),
                    LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                )],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::Ptr(opaque_ptr) = value else {
            panic!("expected opaque field of gc struct to stay a plain ptr");
        };
        assert_eq!(
            opaque_ptr._TYPE.TO,
            PtrTarget::Opaque(OpaqueType::new("OpaqueX"))
        );
    }

    #[test]
    fn fixedsize_array_pointer_reports_fixed_length() {
        let ptr_t = Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 3)),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.len().unwrap(), 3);
        assert_eq!(ptr._fixedlength(), Ok(Some(3)));
    }

    #[test]
    fn array_pointer_setitem_checks_item_type() {
        let mut ptr = Ptr {
            TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
        }
        ._example();
        ptr.setitem(0, LowLevelValue::Signed(7)).unwrap();
        assert_eq!(ptr.getitem(0).unwrap(), LowLevelValue::Signed(7));
        let err = ptr
            .setitem(0, LowLevelValue::Bool(false))
            .expect_err("array setitem must reject mismatched item types");
        assert!(err.contains("expect"));
    }

    #[test]
    fn fixedsize_array_pointer_getitem_reports_out_of_bounds() {
        let ptr = Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 0)),
        }
        ._example();
        let err = ptr
            .getitem(0)
            .expect_err("zero-length fixed-size array must reject index 0");
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn interior_ptr_obj_uses_actual_index_offset() {
        let parent = Struct::new(
            "S",
            vec![(
                "arr".into(),
                LowLevelType::FixedSizeArray(Box::new(FixedSizeArray::new(
                    LowLevelType::Signed,
                    3,
                ))),
            )],
        )
        ._container_example();
        // The `_struct._fields` cell is shared via `Arc<Mutex<>>` so
        // modifying `arr.items` propagates back to `parent._fields`.
        let arr_value = parent
            ._getattr("arr")
            .expect("parent must expose 'arr' field");
        let LowLevelValue::Array(arr) = arr_value else {
            panic!("expected array field");
        };
        arr.setitem(1, LowLevelValue::Signed(7));
        let iptr = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(parent)),
            _offsets: vec![
                InteriorOffset::Field("arr".into()),
                InteriorOffset::Index(1),
            ],
        };
        assert_eq!(iptr._obj(), LowLevelValue::Signed(7));
    }

    #[test]
    fn interior_ptr_setitem_updates_parent_storage() {
        let mut iptr = _interior_ptr {
            _T: LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new(
                    "S",
                    vec![(
                        "arr".into(),
                        LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
                    )],
                )
                ._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("arr".into())],
        };
        iptr.setitem(0, LowLevelValue::Signed(9)).unwrap();
        let LowLevelValue::Array(arr) = iptr._obj() else {
            panic!("expected array after interior setitem");
        };
        assert_eq!(arr.getitem(0), Some(LowLevelValue::Signed(9)));
    }

    #[test]
    fn interior_ptr_exposes_opaque_child_as_interior_ptr() {
        let iptr = _interior_ptr {
            _T: LowLevelType::Struct(Box::new(Struct::new(
                "S",
                vec![(
                    "x".into(),
                    LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                )],
            ))),
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new(
                    "S",
                    vec![(
                        "x".into(),
                        LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                    )],
                )
                ._container_example(),
            )),
            _offsets: vec![],
        };
        let value = iptr.getattr("x").unwrap();
        let LowLevelValue::InteriorPtr(opaque_iptr) = value else {
            panic!("expected interior ptr to keep opaque child as interior ptr");
        };
        assert_eq!(
            opaque_iptr._TYPE(),
            InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(Struct::new(
                    "S",
                    vec![(
                        "x".into(),
                        LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                    )],
                )))),
                TO: Box::new(LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX")))),
                offsets: vec![InteriorOffset::Field("x".into())],
            }
        );
    }

    #[test]
    #[should_panic(expected = "do not test an interior pointer for nullity")]
    fn interior_ptr_nonzero_panics() {
        let iptr = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                Struct::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let _ = iptr.nonzero();
    }

    #[test]
    #[should_panic(expected = "field name")]
    fn struct_new_rejects_underscore_prefix_field() {
        // lltype.py:267-269 — NameError on leading underscore.
        let _ = Struct::new("S", vec![("_hidden".into(), LowLevelType::Signed)]);
    }

    #[test]
    #[should_panic(expected = "repeated field name")]
    fn struct_new_rejects_repeated_field_name() {
        // lltype.py:271-272 — TypeError on repeated field.
        let _ = Struct::new(
            "S",
            vec![
                ("x".into(), LowLevelType::Signed),
                ("x".into(), LowLevelType::Bool),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "cannot inline")]
    fn struct_new_rejects_gc_container_inlined_past_first_field() {
        // lltype.py:274-279 — a gc container can only be inlined as the
        // first field of a struct with matching gckind.
        let gc_inner = Struct::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = Struct::gc(
            "Outer",
            vec![
                ("first".into(), LowLevelType::Signed),
                ("bad".into(), LowLevelType::Struct(Box::new(gc_inner))),
            ],
        );
    }

    #[test]
    fn struct_new_allows_gc_first_field_of_matching_gckind() {
        // lltype.py:275-276 — ok to inline XxContainer as first field of
        // XxStruct when _gckinds match.
        let gc_inner = Struct::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Struct::gc(
            "Outer",
            vec![
                (
                    "head".into(),
                    LowLevelType::Struct(Box::new(gc_inner.clone())),
                ),
                ("x".into(), LowLevelType::Signed),
            ],
        );
        assert_eq!(outer._names, vec!["head".to_string(), "x".to_string()]);
    }

    #[test]
    fn struct_short_name_prefixes_with_kind() {
        // lltype.py _short_name.
        let raw = Struct::new("S", vec![("x".into(), LowLevelType::Signed)]);
        assert_eq!(raw._short_name(), "Struct S");
        let gc = Struct::gc("S", vec![("x".into(), LowLevelType::Signed)]);
        assert_eq!(gc._short_name(), "GcStruct S");
    }

    #[test]
    fn struct_is_atomic_walks_fields() {
        // lltype.py _is_atomic.
        let plain = Struct::new(
            "S",
            vec![
                ("a".into(), LowLevelType::Signed),
                ("b".into(), LowLevelType::Bool),
            ],
        );
        assert!(plain._is_atomic());
        let with_opaque = Struct::new(
            "S",
            vec![(
                "o".into(),
                LowLevelType::Opaque(Box::new(OpaqueType::new("T"))),
            )],
        );
        assert!(!with_opaque._is_atomic());
    }

    #[test]
    fn struct_names_without_voids_filters_voids() {
        // lltype.py _names_without_voids.
        let s = Struct::new(
            "S",
            vec![
                ("keep".into(), LowLevelType::Signed),
                ("drop".into(), LowLevelType::Void),
            ],
        );
        assert_eq!(s._names_without_voids(), vec!["keep".to_string()]);
    }

    #[test]
    fn frozendict_fields_are_order_insensitive_for_extras() {
        // lltype.py, 208-210 — _adtmeths/_hints are frozendict,
        // so dict item order must not affect type equality or hash.
        let left = Struct::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![
                ("a".into(), ConstValue::Bool(true)),
                ("b".into(), ConstValue::Bool(false)),
            ],
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("render_as_void".into(), ConstValue::Bool(false)),
            ],
            true,
        );
        let right = Struct::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![
                ("b".into(), ConstValue::Bool(false)),
                ("a".into(), ConstValue::Bool(true)),
            ],
            vec![
                ("render_as_void".into(), ConstValue::Bool(false)),
                ("immutable".into(), ConstValue::Bool(true)),
            ],
            true,
        );

        assert_eq!(left, right);

        let mut left_hash = std::collections::hash_map::DefaultHasher::new();
        left.hash(&mut left_hash);
        let mut right_hash = std::collections::hash_map::DefaultHasher::new();
        right.hash(&mut right_hash);
        assert_eq!(left_hash.finish(), right_hash.finish());
    }

    #[test]
    fn struct_first_struct_matches_leading_gc_child() {
        // lltype.py _first_struct.
        let inner = Struct::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Struct::gc(
            "Outer",
            vec![
                ("head".into(), LowLevelType::Struct(Box::new(inner.clone()))),
                ("x".into(), LowLevelType::Signed),
            ],
        );
        let (name, child) = outer._first_struct().expect("leading gc struct expected");
        assert_eq!(name, "head");
        assert_eq!(child._name, "Inner");
        assert_eq!(_castdepth(&outer, &inner), 1);
        assert_eq!(_castdepth(&outer, &outer), 0);
        assert_eq!(_castdepth(&inner, &outer), -1);
    }

    #[test]
    fn struct_is_varsize_tracks_trailing_array_field() {
        // lltype.py — trailing Array field sets _arrayfld;
        // FixedSizeArray does not.
        let varsize = Struct::new(
            "Vs",
            vec![
                ("len".into(), LowLevelType::Signed),
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
                ),
            ],
        );
        assert!(varsize._is_varsize());
        assert_eq!(varsize._arrayfld.as_deref(), Some("items"));

        let fixed = Struct::new(
            "Fs",
            vec![(
                "items".into(),
                LowLevelType::FixedSizeArray(Box::new(FixedSizeArray::new(
                    LowLevelType::Signed,
                    4,
                ))),
            )],
        );
        assert!(!fixed._is_varsize());
        assert_eq!(fixed._arrayfld, None);
    }

    #[test]
    fn struct_note_inlined_into_rejects_gc_past_first_field() {
        // lltype.py — _note_inlined_into guard.
        let gc_child = Struct::gc("Child", vec![("x".into(), LowLevelType::Signed)]);
        let gc_parent = LowLevelType::Struct(Box::new(Struct::gc("Parent", vec![])));
        assert!(gc_child._note_inlined_into(&gc_parent, true).is_ok());
        let err = gc_child
            ._note_inlined_into(&gc_parent, false)
            .expect_err("gc struct inlined past first field must be rejected");
        assert!(err.contains("GcStruct can only be inlined"));
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn struct_new_rejects_array_in_non_last_field() {
        // lltype.py calls Array._note_inlined_into for each field.
        let _ = Struct::new(
            "S",
            vec![
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
                ),
                ("tail".into(), LowLevelType::Signed),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "cannot be inlined")]
    fn struct_new_rejects_gc_opaque_even_as_first_gc_field() {
        // lltype.py — GcOpaqueType is never inlineable.
        let _ = Struct::gc(
            "S",
            vec![(
                "opaque".into(),
                LowLevelType::Opaque(Box::new(OpaqueType::gc("O"))),
            )],
        );
    }

    #[test]
    #[should_panic(expected = "cannot have")]
    fn array_new_rejects_gc_container_item() {
        // lltype.py — Array cannot have a gc container item.
        let gc_inner = Struct::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = Array::new(LowLevelType::Struct(Box::new(gc_inner)));
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn array_new_rejects_raw_array_item() {
        // lltype.py calls OF._note_inlined_into(self, first=False, last=False).
        let inner = Array::new(LowLevelType::Signed);
        let _ = Array::new(LowLevelType::Array(Box::new(inner)));
    }

    #[test]
    #[should_panic(expected = "cannot have")]
    fn fixedsize_array_new_rejects_gc_container_item() {
        // lltype.py — same restriction on FixedSizeArray.
        let gc_inner = Struct::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = FixedSizeArray::new(LowLevelType::Struct(Box::new(gc_inner)), 4);
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn fixedsize_array_new_rejects_raw_array_item() {
        // lltype.py:521 applies the same inline-position check to OF.
        let inner = Array::new(LowLevelType::Signed);
        let _ = FixedSizeArray::new(LowLevelType::Array(Box::new(inner)), 4);
    }

    #[test]
    fn array_new_allows_raw_struct_item() {
        // lltype.py __init__ — raw container items are fine.
        let raw_inner = Struct::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let arr = Array::new(LowLevelType::Struct(Box::new(raw_inner)));
        assert_eq!(arr._gckind, GcKind::Raw);
    }

    #[test]
    fn array_short_name_prefixes_with_kind() {
        // lltype.py — Array/GcArray _short_name.
        assert_eq!(
            Array::new(LowLevelType::Signed)._short_name(),
            "Array Signed"
        );
        assert_eq!(
            Array::gc(LowLevelType::Signed)._short_name(),
            "GcArray Signed"
        );
    }

    #[test]
    fn fixedsize_array_short_name_carries_length_and_item() {
        // lltype.py _short_name.
        assert_eq!(
            FixedSizeArray::new(LowLevelType::Signed, 3)._short_name(),
            "FixedSizeArray 3 Signed"
        );
    }

    #[test]
    fn array_is_atomic_walks_item_type() {
        assert!(Array::new(LowLevelType::Signed)._is_atomic());
        let with_opaque = Array::new(LowLevelType::Opaque(Box::new(OpaqueType::new("T"))));
        assert!(!with_opaque._is_atomic());
    }

    #[test]
    fn array_note_inlined_into_requires_last_struct_slot() {
        // lltype.py — last field of a Struct only.
        let arr = Array::new(LowLevelType::Signed);
        let parent_struct = LowLevelType::Struct(Box::new(Struct::new("S", vec![])));
        assert!(arr._note_inlined_into(&parent_struct, true).is_ok());
        let err = arr
            ._note_inlined_into(&parent_struct, false)
            .expect_err("array inlined in non-last position must be rejected");
        assert!(err.contains("last field"));
    }

    #[test]
    fn array_note_inlined_into_rejects_gc_array() {
        // lltype.py:445-446 — gc arrays never inline.
        let gc_arr = Array::gc(LowLevelType::Signed);
        let parent_struct = LowLevelType::Struct(Box::new(Struct::new("S", vec![])));
        let err = gc_arr
            ._note_inlined_into(&parent_struct, true)
            .expect_err("gc array must not inline");
        assert!(err.contains("GC array"));
    }

    #[test]
    fn null_ptr_equality_handles_both_sides_null() {
        // lltype.py:1185-1195 — pointer equality should handle null-null
        // (equal) and null-nonnull (unequal) without panicking at _obj().
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let null_a = _ptr::new(ptr_t.clone(), Ok(None));
        let null_b = _ptr::new(ptr_t.clone(), Ok(None));
        assert!(
            null_a == null_b,
            "two null pointers of same type compare equal"
        );
        let nonnull = ptr_t._example();
        assert!(null_a != nonnull, "null vs nonnull must be unequal");
        assert!(null_a._same_obj(&null_b).unwrap());
        assert!(!null_a._same_obj(&nonnull).unwrap());
    }

    #[test]
    #[should_panic(expected = "wrong argument number")]
    fn func_call_rejects_wrong_arity() {
        // lltype.py:1352-1354.
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let _ = f.call(&[]);
    }

    #[test]
    #[should_panic(expected = "wrong argument type")]
    fn func_call_rejects_wrong_arg_type() {
        // lltype.py:1355-1380 — per-arg type mismatch raises TypeError.
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let _ = f.call(&[LowLevelValue::Bool(true)]);
    }

    #[test]
    #[should_panic(expected = "undefined function")]
    fn func_call_rejects_none_callable() {
        // lltype.py:1381-1383 — RuntimeError on None _callable.
        let f = _func::new(
            FuncType {
                args: vec![],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            None,
            HashMap::new(),
        );
        let _ = f.call(&[]);
    }

    #[test]
    fn func_call_accepts_matching_args() {
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Bool,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let result = f.call(&[LowLevelValue::Signed(42)]);
        assert_eq!(result, LowLevelValue::Bool(false));
    }

    #[test]
    fn ptr_fixedlength_rejects_non_array_ptr() {
        // lltype.py — _fixedlength calls len() first, which
        // raises TypeError for non-array pointers. Rust port surfaces
        // this as Err(String).
        let struct_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        }
        ._example();
        let err = struct_ptr
            ._fixedlength()
            .expect_err("_fixedlength on struct ptr must error");
        assert!(err.contains("no length"));
    }

    #[test]
    fn interior_ptr_type_with_index_is_gcstruct_with_interior_hint() {
        // lltype.py — result is GcStruct, not raw Struct, and
        // carries the interior_ptr_type hint.
        let gc_parent_ptr = Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(Struct::new(
                "Item",
                vec![("x".into(), LowLevelType::Signed)],
            ))))),
        };
        let interior_struct =
            gc_parent_ptr._interior_ptr_type_with_index(&LowLevelType::Struct(Box::new(
                Struct::new("Item", vec![("x".into(), LowLevelType::Signed)]),
            )));
        assert_eq!(
            interior_struct._gckind,
            GcKind::Gc,
            "interior_ptr_type_with_index must produce GcStruct"
        );
        assert!(
            interior_struct
                ._hints
                .get("interior_ptr_type")
                .is_some_and(|v| matches!(v, ConstValue::Bool(true))),
            "interior_ptr_type hint must be present"
        );
        assert_eq!(
            interior_struct._names,
            vec!["ptr".to_string(), "index".to_string()]
        );
    }

    #[test]
    #[should_panic(expected = "requires gc parent")]
    fn interior_ptr_type_with_index_rejects_raw_parent() {
        // lltype.py — `assert self.TO._gckind == 'gc'`.
        let raw_ptr = Ptr {
            TO: PtrTarget::Struct(Struct::new("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let _ = raw_ptr._interior_ptr_type_with_index(&LowLevelType::Signed);
    }

    #[test]
    fn interior_ptr_type_with_index_copies_struct_adtmeths() {
        // lltype.py — when TO is a Struct, _adtmeths propagates.
        let meth_name = "adt_probe".to_string();
        let adtmeths = vec![(meth_name.clone(), ConstValue::Bool(true))];
        let to_struct = Struct::with_adtmeths(
            "Item",
            vec![("x".into(), LowLevelType::Signed)],
            adtmeths.clone(),
        );
        let gc_parent_ptr = Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(to_struct.clone())))),
        };
        let interior =
            gc_parent_ptr._interior_ptr_type_with_index(&LowLevelType::Struct(Box::new(to_struct)));
        assert_eq!(interior._adtmeths, frozendict::new(adtmeths));
    }

    #[test]
    fn freed_container_reports_was_freed_and_kills_wref_deref() {
        // `_free` flips the container's shared `_parentable`
        // storage; `_was_freed` (lltype.py) and `_wref._dereference`
        // (llmemory.py) observe it. Before `_free` the referent is
        // live; after, the weakref dereferences to `None`.
        let s = Struct::_build(
            "gcthing",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let target = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let wref = _wref::new(Some(&target));

        assert!(!target._was_freed().unwrap());
        assert!(wref._dereference().unwrap().is_some());

        let _ptr_obj::Struct(container) = target._obj().unwrap() else {
            panic!("gc struct ptr must hold a Struct container");
        };
        container._free();

        assert!(target._was_freed().unwrap());
        assert!(wref._dereference().unwrap().is_none());
    }

    #[test]
    fn inlined_sub_container_is_freed_when_parent_is_freed() {
        // `_struct.__init__` wires `_setparentstructure`
        // on each inlined sub-container (lltype.py); `_was_freed`
        // walks the `_wrparent` chain (lltype.py), so freeing the
        // parent makes the inlined sub-struct report freed too.
        let inner = Struct::new("inner", vec![("x".into(), LowLevelType::Signed)]);
        let outer = Struct::_build(
            "outer",
            vec![("sub".into(), LowLevelType::Struct(Box::new(inner)))],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let parent = malloc(
            LowLevelType::Struct(Box::new(outer)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();

        let _ptr_obj::Struct(parent_container) = parent._obj().unwrap() else {
            panic!("gc struct ptr must hold a Struct container");
        };
        let Some(LowLevelValue::Struct(sub)) = parent_container._getattr("sub") else {
            panic!("inlined sub field must hold a Struct container");
        };

        assert!(!sub._was_freed());
        parent_container._free();
        // The sub-container's own storage was never freed; it reports freed
        // only by walking `_wrparent` to the freed parent.
        assert!(sub._was_freed());
    }

    #[test]
    fn parentlink_returns_parent_container_and_field_index() {
        // `_parentstructure()` / `parentlink` (lltype.py, 1123-1128):
        // an inlined sub-container hands back its enclosing container *object*
        // and the field index it occupies; a top-level allocation has no
        // parent.
        let inner = Struct::new("inner", vec![("x".into(), LowLevelType::Signed)]);
        let outer = Struct::_build(
            "outer",
            vec![("sub".into(), LowLevelType::Struct(Box::new(inner)))],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let parent = malloc(
            LowLevelType::Struct(Box::new(outer)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let parent_obj = parent._obj().unwrap();
        let _ptr_obj::Struct(parent_container) = parent_obj.clone() else {
            panic!("gc struct ptr must hold a Struct container");
        };
        let Some(LowLevelValue::Struct(sub)) = parent_container._getattr("sub") else {
            panic!("inlined sub field must hold a Struct container");
        };

        // A top-level allocation is unparented.
        assert!(matches!(parentlink(&parent_obj), (None, None)));

        // The inlined sub-struct points back at `outer` via field "sub", and
        // the returned parent is the same container object (by identity).
        let (got_parent, got_index) = parentlink(&_ptr_obj::Struct(*sub));
        assert_eq!(got_parent, Some(parent_obj));
        match got_index {
            Some(ParentIndex::Field(name)) => assert_eq!(name, "sub"),
            other => panic!("expected Field(\"sub\"), got {other:?}"),
        }
    }

    #[test]
    fn direct_fieldptr_makes_subarray_over_struct_field() {
        // `direct_fieldptr(structptr, "x")` (lltype.py) → a
        // `Ptr(FixedSizeArray(Signed, 1))` interior pointer whose `getitem(0)`
        // reads the named field; a missing field declines.
        let s = Struct::gc(
            "point",
            vec![
                ("x".into(), LowLevelType::Signed),
                ("y".into(), LowLevelType::Signed),
            ],
        );
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let _ptr_obj::Struct(container) = p._obj().unwrap() else {
            panic!("gc struct ptr must hold a Struct container");
        };
        container._setattr("x", LowLevelValue::Signed(7));

        let fieldptr = direct_fieldptr(&p, "x").unwrap();
        let _ptr_obj::Subarray(sub) = fieldptr._obj().unwrap() else {
            panic!("direct_fieldptr yields a _subarray interior pointer");
        };
        assert_eq!(sub.getitem(0), Some(LowLevelValue::Signed(7)));

        assert!(direct_fieldptr(&p, "z").is_err());
    }

    #[test]
    fn subarray_makeptr_memoizes_per_parent_and_offset() {
        // `_subarray._cache` (lltype.py): re-deriving the same
        // interior pointer reuses one `_subarray` (so the two fakeaddresses
        // would compare equal); a different field is a distinct container.
        let s = Struct::gc(
            "point",
            vec![
                ("x".into(), LowLevelType::Signed),
                ("y".into(), LowLevelType::Signed),
            ],
        );
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let x1 = direct_fieldptr(&p, "x").unwrap();
        let x2 = direct_fieldptr(&p, "x").unwrap();
        let y1 = direct_fieldptr(&p, "y").unwrap();
        assert_eq!(x1._obj().unwrap(), x2._obj().unwrap());
        assert_ne!(x1._obj().unwrap(), y1._obj().unwrap());
    }

    #[test]
    fn arraylenref_makeptr_memoizes_per_array() {
        // `_arraylenref._cache` (lltype.py): the length pointer of
        // one array is the same container on re-derivation.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let _ptr_obj::Array(arr) = arrayptr._obj().unwrap() else {
            panic!("array ptr must hold an Array container");
        };
        let l1 = _arraylenref::_makeptr(Box::new(arr.clone()), true);
        let l2 = _arraylenref::_makeptr(Box::new(arr), true);
        assert_eq!(l1._obj().unwrap(), l2._obj().unwrap());
    }

    #[test]
    fn ptr_getitem_setitem_through_subarray_interior_pointer() {
        // `_ptr.__getitem__/__setitem__` delegate to `self._obj.getitem/
        // setitem` (lltype.py/1320), so a `direct_fieldptr` _subarray
        // pointer both reads and writes the parent struct field.
        let s = Struct::gc("point", vec![("x".into(), LowLevelType::Signed)]);
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let _ptr_obj::Struct(container) = p._obj().unwrap() else {
            panic!("gc struct ptr must hold a Struct container");
        };
        container._setattr("x", LowLevelValue::Signed(5));

        let mut fieldptr = direct_fieldptr(&p, "x").unwrap();
        assert_eq!(fieldptr.getitem(0).unwrap(), LowLevelValue::Signed(5));
        fieldptr.setitem(0, LowLevelValue::Signed(9)).unwrap();
        // The write lands in the parent struct field, visible both through
        // the parent container and a re-read of the interior pointer.
        assert_eq!(container._getattr("x"), Some(LowLevelValue::Signed(9)));
        assert_eq!(fieldptr.getitem(0).unwrap(), LowLevelValue::Signed(9));
    }

    #[test]
    fn direct_ptradd_shifts_backward_into_range() {
        // `direct_ptradd(ptr, n)` (lltype.py) builds a `_subarray`
        // at `base + n` for any sign of the shift; only an out-of-bounds
        // dereference errors. Shift forward to base 2 then back to base 1.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let _ptr_obj::Array(arr) = arrayptr._obj().unwrap() else {
            panic!("array ptr must hold an Array container");
        };
        assert!(arr.setitem(1, LowLevelValue::Signed(20)));

        let items = direct_arrayitems(&arrayptr).unwrap();
        let fwd = direct_ptradd(&items, 2).unwrap();
        let back = direct_ptradd(&fwd, -1).unwrap();
        let _ptr_obj::Subarray(sub) = back._obj().unwrap() else {
            panic!("direct_ptradd yields a _subarray interior pointer");
        };
        assert_eq!(sub.getitem(0), Some(LowLevelValue::Signed(20)));
    }

    #[test]
    fn fixup_solid_returns_solid_pointer_to_same_container() {
        // `fixup_solid(p)` (constfold.py) pins the parent of an inlined
        // sub-pointer and returns `container._as_ptr()` — a solid pointer.
        // `_setparentstructure` already applies the `_keepparent` rule
        // (lltype.py) on construction, so the keepalive is already
        // in place; `fixup_solid`'s observable effect here is the solid flag.
        // `_ptr` equality is by container identity, so the result still equals
        // the input. Both a plain array container and a `direct_arrayitems`
        // `_subarray` interior pointer are `_parentable`.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let fixed = fixup_solid(&arrayptr).unwrap();
        assert_eq!(fixed, arrayptr);
        assert!(fixed._solid);
        let items = direct_arrayitems(&arrayptr).unwrap();
        let fixed_items = fixup_solid(&items).unwrap();
        assert_eq!(fixed_items, items);
        assert!(fixed_items._solid);
    }

    #[test]
    fn arraylenref_setitem_shrinks_parent_array() {
        // `_arraylenref.setitem(0, smaller)` (lltype.py) shrinks the
        // parent array in place; the length pointer then reads the new length.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let _ptr_obj::Array(arr) = arrayptr._obj().unwrap() else {
            panic!("array ptr must hold an Array container");
        };
        let mut lenptr = _arraylenref::_makeptr(Box::new(arr.clone()), true);
        assert_eq!(lenptr.getitem(0).unwrap(), LowLevelValue::Signed(3));
        lenptr.setitem(0, LowLevelValue::Signed(1)).unwrap();
        assert_eq!(lenptr.getitem(0).unwrap(), LowLevelValue::Signed(1));
        // The shrink is visible through the shared parent container.
        assert_eq!(arr.getlength(), 1);
    }

    #[test]
    fn ptr_getitem_arraylenref_out_of_bounds_errs_not_panics() {
        // `_ptr.__getitem__` bounds-checks via `getbounds()` (lltype.py-
        // 1293) ahead of the container. An `_arraylenref` spans `(0, 1)`, so
        // index 1 is an out-of-bounds error, not the `assert index == 0` panic.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let _ptr_obj::Array(arr) = arrayptr._obj().unwrap() else {
            panic!("array ptr must hold an Array container");
        };
        let lenptr = _arraylenref::_makeptr(Box::new(arr), true);
        assert!(lenptr.getitem(0).is_ok());
        assert!(lenptr.getitem(1).is_err());
    }

    #[test]
    fn ptr_getitem_backward_subarray_index_zero_errs_not_panics() {
        // A backward `direct_ptradd` base shifts the `_subarray` bounds so that
        // index 0 falls below `start` (lltype.py); `_ptr.__getitem__`
        // reports an out-of-bounds error rather than reaching the negative
        // parent index that would panic the container `getitem`.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let items = direct_arrayitems(&arrayptr).unwrap();
        let back = direct_ptradd(&items, -1).unwrap();
        // bounds = (0 - (-1), 3 - (-1)) = (1, 4): index 0 is below start.
        assert!(back.getitem(0).is_err());
        // index 1 reads the parent's item 0 (effective = -1 + 1 = 0).
        assert!(back.getitem(1).is_ok());
    }

    #[test]
    fn cast_pointer_up_cast_round_trips_through_first_field() {
        // A `GcStruct` whose first inlined field is the base struct models the
        // rclass inheritance layout. A pointer down-cast to the base field
        // up-casts back to the original container (lltype.py).
        let object_t = Struct::gc("object", vec![("typeptr".into(), LowLevelType::Signed)]);
        let sub_t = Struct::gc(
            "subclass",
            vec![
                (
                    "super".into(),
                    LowLevelType::Struct(Box::new(object_t.clone())),
                ),
                ("x".into(), LowLevelType::Signed),
            ],
        );
        let subptr = malloc(
            LowLevelType::Struct(Box::new(sub_t.clone())),
            None,
            MallocFlavor::Gc,
            true,
        )
        .unwrap();

        let object_ptrtype =
            Ptr::from_container_type(LowLevelType::Struct(Box::new(object_t))).unwrap();
        let sub_ptrtype = Ptr::from_container_type(LowLevelType::Struct(Box::new(sub_t))).unwrap();

        // down-cast to the base field, then up-cast back to the subclass.
        let objptr = cast_pointer(&object_ptrtype, &subptr).unwrap();
        let back = cast_pointer(&sub_ptrtype, &objptr).unwrap();
        assert_eq!(back, subptr);
    }

    #[test]
    #[should_panic(expected = "accessing freed container")]
    fn double_free_panics() {
        // `_free` calls `_check` first, so a second `_free`
        // on the same container raises (lltype.py).
        let s = Struct::new("thing", vec![("x".into(), LowLevelType::Signed)]);
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap();
        let _ptr_obj::Struct(container) = p._obj().unwrap() else {
            panic!("struct ptr must hold a Struct container");
        };
        container._free();
        container._free();
    }

    #[test]
    #[should_panic(expected = "accessing freed container")]
    fn nonzero_on_freed_pointer_raises() {
        // `__nonzero__` is `_getobj(check=True)` (lltype.py), so a
        // freed pointer raises on truthiness like every other dereference,
        // rather than silently testing not-null.
        let s = Struct::new("thing", vec![("x".into(), LowLevelType::Signed)]);
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap();
        let _ptr_obj::Struct(container) = p._obj().unwrap() else {
            panic!("struct ptr must hold a Struct container");
        };
        container._free();
        let _ = p.nonzero();
    }

    #[test]
    fn array_setitem_trailing_nul_special_case() {
        use crate::flowspace::model::ConstValue;
        // `Array(Char, hints={'extra_item_after_alloc': 1})` like STR.chars:
        // writing '\x00' to the one slot past the end is a no-op success;
        // any other out-of-bounds write fails (lltype.py).
        let with_extra = Array::with_hints(
            LowLevelType::Char,
            vec![("extra_item_after_alloc".into(), ConstValue::Int(1))],
        );
        let arr = build_array(
            ArrayContainer::Array(with_extra),
            vec![LowLevelValue::Char('a'), LowLevelValue::Char('b')],
        );
        assert!(arr.setitem(2, LowLevelValue::Char('\0')));
        assert!(!arr.setitem(2, LowLevelValue::Char('z')));
        assert!(!arr.setitem(3, LowLevelValue::Char('\0')));
        assert!(arr.setitem(1, LowLevelValue::Char('c')));

        // Without the hint, the trailing NUL write is a plain out-of-bounds
        // failure.
        let plain = build_array(
            ArrayContainer::Array(Array::new(LowLevelType::Char)),
            vec![LowLevelValue::Char('a')],
        );
        assert!(!plain.setitem(1, LowLevelValue::Char('\0')));

        // A falsy `extra_item_after_alloc` hint (`0`) does not enable the
        // special case — `.get(...)` truthiness, not key presence.
        let falsy_hint = Array::with_hints(
            LowLevelType::Char,
            vec![("extra_item_after_alloc".into(), ConstValue::Int(0))],
        );
        let falsy = build_array(
            ArrayContainer::Array(falsy_hint),
            vec![LowLevelValue::Char('a')],
        );
        assert!(!falsy.setitem(1, LowLevelValue::Char('\0')));
    }

    #[test]
    #[should_panic(expected = "items: expect")]
    fn array_setitem_rejects_mismatched_item_type() {
        // `assert typeOf(value) == self._TYPE.OF` (lltype.py) — a
        // `Char` write to a `Signed`/`extra_item_after_alloc` array fails the
        // element-type invariant rather than slipping through the trailing-NUL
        // special case.
        use crate::flowspace::model::ConstValue;
        let signed_extra = Array::with_hints(
            LowLevelType::Signed,
            vec![("extra_item_after_alloc".into(), ConstValue::Int(1))],
        );
        let arr = build_array(
            ArrayContainer::Array(signed_extra),
            vec![LowLevelValue::Signed(1)],
        );
        arr.setitem(1, LowLevelValue::Char('\0'));
    }

    #[test]
    fn fakeaddress_eq_compares_underlying_container() {
        // `fakeaddress.__eq__` (llmemory.py): NULL == NULL, NULL is
        // unequal to a live address, and two `Fake` addresses are equal iff
        // they wrap the same underlying container.
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let p = ptr_t._example();
        let same = p.clone(); // same container identity
        let other = ptr_t._example(); // fresh _container_example → distinct identity

        let null = _address::Null;
        let fake = _address::Fake(Box::new(p));
        let fake_same = _address::Fake(Box::new(same));
        let fake_other = _address::Fake(Box::new(other));

        assert!(null._eq(&_address::Null));
        assert!(!null._eq(&fake));
        assert!(!fake._eq(&null));
        assert!(fake._eq(&fake_same));
        assert!(!fake._eq(&fake_other));

        // `__ne__` is the negation of `__eq__`.
        assert!(null._ne(&fake));
        assert!(!fake._ne(&fake_same));
        assert!(fake._ne(&fake_other));
    }

    #[test]
    fn fakeaddress_ordering_null_is_smallest() {
        // `fakeaddress.__lt__`/`__le__`/`__gt__`/`__ge__` (llmemory.py):
        // NULL is the smallest address; two non-NULL addresses cannot be
        // ordered (TypeError → `Err`).
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let null = _address::Null;
        let fake = _address::Fake(Box::new(ptr_t._example()));
        let fake2 = _address::Fake(Box::new(ptr_t._example()));

        assert!(!null.nonzero());
        assert!(fake.nonzero());

        assert_eq!(null._lt(&fake), Ok(true)); // NULL < non-null
        assert_eq!(fake._lt(&null), Ok(false)); // non-null < NULL
        assert_eq!(null._lt(&null), Ok(false)); // NULL < NULL
        assert!(fake._lt(&fake2).is_err()); // two non-NULL → TypeError

        assert_eq!(fake._ge(&null), Ok(true)); // not (non-null < NULL)
        assert_eq!(null._le(&fake), Ok(true)); // NULL <= non-null
        assert_eq!(fake._gt(&null), Ok(true)); // non-null > NULL
        assert!(fake._le(&fake2).is_err()); // composes the un-orderable `<`
    }

    #[test]
    fn fakeaddress_delta_zero_when_equal_else_typeerror() {
        // `fakeaddress.__sub__` (llmemory.py), fakeaddress arm: equal
        // addresses subtract to 0; distinct addresses raise TypeError (`Err`).
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
        };
        let p = ptr_t._example();
        let fake = _address::Fake(Box::new(p.clone()));
        let fake_same = _address::Fake(Box::new(p));
        let fake_other = _address::Fake(Box::new(ptr_t._example()));

        assert_eq!(_address::Null._delta(&_address::Null), Ok(0));
        assert_eq!(fake._delta(&fake_same), Ok(0));
        assert!(fake._delta(&fake_other).is_err());
    }

    #[test]
    fn hidden_opaque_eq_hash_by_normalized_container() {
        // lltype.py __eq__ — two `'hidden'` opaques wrapping the same
        // allocation compare equal and hash equal despite distinct opaque
        // `_identity`; a different allocation is unequal.
        use std::collections::hash_map::DefaultHasher;
        let opaque_hash = |o: &_opaque| {
            let mut h = DefaultHasher::new();
            o.hash(&mut h);
            h.finish()
        };
        let s = Struct::_build(
            "gcs",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
            true,
        );
        let LowLevelType::Ptr(opaque_ptr_t) = WEAKREF_PTR.clone() else {
            panic!("WEAKREF_PTR must be a Ptr");
        };
        let make_hidden = |ty: &LowLevelType| {
            let p = malloc(ty.clone(), None, MallocFlavor::Gc, true).unwrap();
            let op = cast_opaque_ptr(&opaque_ptr_t, &p).unwrap();
            let _ptr_obj::Opaque(o) = op._obj().unwrap() else {
                panic!("cast_opaque_ptr must yield an opaque container");
            };
            o
        };

        let struct_ty = LowLevelType::Struct(Box::new(s));
        let p = malloc(struct_ty.clone(), None, MallocFlavor::Gc, true).unwrap();
        let o1 = {
            let _ptr_obj::Opaque(o) = cast_opaque_ptr(&opaque_ptr_t, &p).unwrap()._obj().unwrap()
            else {
                panic!("opaque");
            };
            o
        };
        let o2 = {
            let _ptr_obj::Opaque(o) = cast_opaque_ptr(&opaque_ptr_t, &p).unwrap()._obj().unwrap()
            else {
                panic!("opaque");
            };
            o
        };
        // Distinct opaque objects, same underlying allocation.
        assert_ne!(o1.identity(), o2.identity());
        assert_eq!(o1, o2);
        assert_eq!(opaque_hash(&o1), opaque_hash(&o2));

        // A separate allocation normalizes to a different container.
        let o3 = make_hidden(&struct_ty);
        assert_ne!(o1, o3);
    }

    #[test]
    fn module_level_parity_maps_keep_upstream_names() {
        assert_eq!(*NFOUND, "NFOUND");
        assert_eq!(STRUCT_BY_FLAVOR.get("raw"), Some(&"Struct"));
        assert_eq!(STRUCT_BY_FLAVOR.get("gc"), Some(&"GcStruct"));
        assert_eq!(FORWARDREF_BY_FLAVOR.get("raw"), Some(&"ForwardReference"));
        assert_eq!(
            FORWARDREF_BY_FLAVOR.get("prebuilt"),
            Some(&"FuncForwardReference")
        );
        assert_eq!(_numbertypes.get("int"), Some(&LowLevelType::Signed));
        assert_eq!(_numbertypes.get("r_uint"), Some(&LowLevelType::Unsigned));
        assert_eq!(_to_primitive.get("Char"), Some(&"chr"));
        assert_eq!(_to_primitive.get("Signed"), Some(&"int"));

        let _state = State;
        let _uninit = _uninitialized {
            TYPE: LowLevelType::Signed,
        };
        let typedef = Typedef::new(LowLevelType::Signed, Some("SignedAlias".into()));
        assert_eq!(typedef.OF, LowLevelType::Signed);
        let _err = UninitializedMemoryAccess;
        let _entry = _ptrEntry;
        let method = staticAdtMethod::new("adt");
        assert_eq!(method.get(), &"adt");
        assert_eq!(typeMethod(7), 7);
    }

    #[test]
    fn deferred_lltype_helpers_name_the_original_surface() {
        let err = cast_primitive().expect_err("cast_primitive lowering is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("cast_primitive"));

        let err = ann_runtime_type_info().expect_err("runtime analyzer is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ann_runtime_type_info"));

        let err = dissect_ll_instance().expect_err("dissection helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("dissect_ll_instance"));
    }

    #[test]
    fn cast_ptr_to_int_handles_null_tagged_and_live_pointer() {
        let s = Struct::new("thing", vec![("x".into(), LowLevelType::Signed)]);
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(s.clone()),
        };

        let null = nullptr(LowLevelType::Struct(Box::new(s.clone()))).unwrap();
        assert_eq!(cast_ptr_to_int(&null), Ok(0));

        let tagged = cast_int_to_ptr(&ptr_t, 17).unwrap();
        assert_eq!(cast_ptr_to_int(&tagged), Ok(17));

        let live = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap();
        assert_eq!(cast_ptr_to_int(&live), Ok(live._hashable_identity() as i64));
    }
}
