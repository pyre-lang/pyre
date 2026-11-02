//! RPython `rpython/rlib/rarithmetic.py` — integer arithmetic helpers.
//!
//! Only the subset consumed by the annotator port lands here. The
//! full RPython rarithmetic module covers ~700 LOC of overflow-safe
//! operators, r_uint / r_longlong / r_singlefloat types, ovfcheck
//! decorators, etc. Those land as their call sites in downstream
//! ports come online.

use indexmap::IndexMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::annotator::model::{KnownType, RIntClass};
use crate::flowspace::model::{ConstValue, HostObject};

/// RPython `LONG_BIT` (`rarithmetic.py`): the width of `lltype.Signed` in bits.
pub fn long_bit() -> u16 {
    (crate::layout::target_signed_size() * 8) as u16
}

/// Incremented on every `type(name, ...)` in `build_int`.
static NEXT_RINT_ID: AtomicU32 = AtomicU32::new(0);

/// Host class object for each created `RIntClass`, indexed by `id`.
/// Owned here the way the generated class object lives on the
/// `rarithmetic` module in upstream.
static RINT_CLASS_OBJECTS: LazyLock<Mutex<Vec<HostObject>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// `base_int(long)` (`rarithmetic.py`). This port's `long` is host `int`.
/// Created here so `_inttypes` seeding can subclass it without entering
/// `HOST_ENV`; the interned builtin class is the same object HOST_ENV
/// publishes as `"int"`.
static BASE_INT: LazyLock<HostObject> = LazyLock::new(|| {
    let host = HostObject::new_class(
        "rpython.rlib.rarithmetic.base_int",
        vec![crate::flowspace::model::builtin_int_class()],
    );
    crate::translator::rtyper::extregistry::register_base_int(host.clone());
    host
});

/// `signed_int` / `unsigned_int` (`rarithmetic.py`).
/// `SIGNED` is on the host class; `typemap = {}` is the class-body dict
/// `base_int._widen` reads and updates. Each of the two class objects
/// owns its own map — subclasses created by `build_int` inherit it.
struct SignedUnsignedInt {
    host: HostObject,
    typemap: Mutex<IndexMap<(KnownType, KnownType), KnownType>>,
}

fn signed_unsigned_int(qualname: &'static str, signed: bool) -> SignedUnsignedInt {
    let mut members = IndexMap::new();
    members.insert("SIGNED".into(), ConstValue::Bool(signed));
    SignedUnsignedInt {
        host: HostObject::new_class_with_members(qualname, vec![BASE_INT.clone()], members),
        typemap: Mutex::new(IndexMap::new()),
    }
}

/// `signed_int` (`rarithmetic.py`): `SIGNED = True`, `typemap = {}`.
static SIGNED_INT: LazyLock<SignedUnsignedInt> =
    LazyLock::new(|| signed_unsigned_int("rpython.rlib.rarithmetic.signed_int", true));

/// `unsigned_int` (`rarithmetic.py`): `SIGNED = False`, `typemap = {}`.
static UNSIGNED_INT: LazyLock<SignedUnsignedInt> =
    LazyLock::new(|| signed_unsigned_int("rpython.rlib.rarithmetic.unsigned_int", false));

/// `base_int` class object (`rarithmetic.py`).
pub fn base_int_class() -> HostObject {
    BASE_INT.clone()
}

/// `signed_int` class object (`rarithmetic.py`).
pub fn signed_int_class() -> HostObject {
    SIGNED_INT.host.clone()
}

/// `unsigned_int` class object (`rarithmetic.py`).
pub fn unsigned_int_class() -> HostObject {
    UNSIGNED_INT.host.clone()
}

/// `self.typemap` in `base_int._widen`: the `typemap` on `signed_int` or
/// `unsigned_int` according to `self_type.SIGNED`.
fn typemap_for(
    self_type: KnownType,
) -> Option<&'static Mutex<IndexMap<(KnownType, KnownType), KnownType>>> {
    let (signed, _) = integer_signed_and_bits(self_type)?;
    Some(if signed {
        &SIGNED_INT.typemap
    } else {
        &UNSIGNED_INT.typemap
    })
}

fn intern_name(name: &str) -> &'static str {
    Box::leak(name.to_owned().into_boxed_str())
}

/// `MASK = (2 ** bits) - 1` as a host integer constant (`build_int`).
/// Same width rule as `RIntClass::mask`; the carrier follows the value.
fn mask_const(bits: u16) -> ConstValue {
    let mask = mask_u128(bits);
    if let Ok(n) = i64::try_from(mask) {
        ConstValue::Int(n)
    } else if let Ok(n) = u64::try_from(mask) {
        ConstValue::UInt(n)
    } else {
        ConstValue::UInt128(mask)
    }
}

/// `type(name, (base_int_type,), {'MASK', 'BITS', 'SIGN'})` in `build_int`.
fn create_rint_class(name: &str, sign: bool, bits: u16) -> RIntClass {
    let name = intern_name(name);
    let base = if sign {
        signed_int_class()
    } else {
        unsigned_int_class()
    };
    let mut members = IndexMap::new();
    members.insert("MASK".into(), mask_const(bits));
    members.insert("BITS".into(), ConstValue::Int(bits as i64));
    members.insert("SIGN".into(), ConstValue::Bool(sign));
    let host = HostObject::new_class_with_members(
        format!("rpython.rlib.rarithmetic.{name}"),
        vec![base],
        members,
    );
    let class = {
        let mut objects = RINT_CLASS_OBJECTS.lock();
        let id = NEXT_RINT_ID.fetch_add(1, Ordering::Relaxed);
        debug_assert_eq!(id as usize, objects.len());
        objects.push(host.clone());
        RIntClass {
            name,
            sign,
            bits,
            id,
        }
    };
    // Registries after `RINT_CLASS_OBJECTS` is released. Lock order
    // with `_inttypes` held by `build_int` / `seed_inttypes` is
    // `_inttypes` then `RINT_CLASS_OBJECTS` then the extregistry maps.
    crate::translator::rtyper::extregistry::register_build_int_class(host, KnownType::RInt(class));
    class
}

fn create_int_type(name: &str, sign: bool, bits: u16) -> KnownType {
    KnownType::RInt(create_rint_class(name, sign, bits))
}

/// Seed helper matching `build_int` without `force_creation`: return the
/// `(sign, bits)` entry when present, otherwise create and record it.
fn register_inttype(
    map: &mut HashMap<(bool, u16), KnownType>,
    name: &str,
    sign: bool,
    bits: u16,
    predefined: Option<KnownType>,
) -> KnownType {
    if let Some(&existing) = map.get(&(sign, bits)) {
        return existing;
    }
    let created = predefined.unwrap_or_else(|| create_int_type(name, sign, bits));
    map.insert((sign, bits), created);
    created
}

fn seed_inttypes() -> HashMap<(bool, u16), KnownType> {
    let mut map = HashMap::new();
    let long_bit = long_bit();
    // Module-level order in `rarithmetic.py`: `r_int`, `r_uint`,
    // `r_longlong`, `r_ulonglong`, `r_longlonglong`, `r_ulonglonglong`,
    // then the `if r_longlong is not r_int` / `else` `r_int32`/`r_uint32`
    // branch.
    register_inttype(&mut map, "r_int", true, long_bit, Some(KnownType::Rint));
    register_inttype(&mut map, "r_uint", false, long_bit, Some(KnownType::Ruint));
    let r_longlong = register_inttype(&mut map, "r_longlong", true, 64, Some(KnownType::LongLong));
    register_inttype(
        &mut map,
        "r_ulonglong",
        false,
        64,
        Some(KnownType::ULongLong),
    );
    register_inttype(
        &mut map,
        "r_longlonglong",
        true,
        128,
        Some(KnownType::LongLongLong),
    );
    register_inttype(
        &mut map,
        "r_ulonglonglong",
        false,
        128,
        Some(KnownType::ULongLongLong),
    );
    if r_longlong == KnownType::Rint {
        register_inttype(&mut map, "r_int32", true, 32, None);
        register_inttype(&mut map, "r_uint32", false, 32, None);
    }
    map
}

/// RPython `_inttypes` (`rarithmetic.py`): keyed by `(sign, bits)`.
#[allow(non_upper_case_globals)]
pub static _inttypes: LazyLock<Mutex<HashMap<(bool, u16), KnownType>>> =
    LazyLock::new(|| Mutex::new(seed_inttypes()));

fn no_predefined_int(sign: bool, bits: u16) -> String {
    format!(
        "TypeError: No predefined {}int{}",
        if sign { "" } else { "u" },
        bits
    )
}

/// `type(name, ...)` in `build_int`, after `_inttypes` lookup and the
/// `name is None` TypeError. The 128-bit carrier limit is ours; it
/// sits here so a missing predefined type still reports
/// `TypeError: No predefined [u]intN`.
fn create_int_type_checked(name: Option<&str>, sign: bool, bits: u16) -> Result<KnownType, String> {
    let name = name.ok_or_else(|| no_predefined_int(sign, bits))?;
    if bits > 128 {
        return Err(format!(
            "build_int: {bits}-bit integers exceed the 128-bit value carrier"
        ));
    }
    Ok(create_int_type(name, sign, bits))
}

/// RPython `build_int(name, sign, bits, force_creation)` (`rarithmetic.py`).
///
/// Looks up `_inttypes[sign, bits]` unless `force_creation`. Missing and
/// `name` is `None` raises `TypeError: No predefined [u]intN`. Otherwise
/// `type(name, ...)` builds a fresh class; it is stored in `_inttypes`
/// unless forced.
pub fn build_int(
    name: Option<&str>,
    sign: bool,
    bits: u16,
    force_creation: bool,
) -> Result<KnownType, String> {
    if force_creation {
        return create_int_type_checked(name, sign, bits);
    }
    // Hold `_inttypes` across lookup, `type(...)` and insert (`build_int`).
    // `create_rint_class` takes `RINT_CLASS_OBJECTS` then the extregistry
    // maps after releasing it. Lock order is `_inttypes` then
    // `RINT_CLASS_OBJECTS` then the registries. Forced creation does not
    // touch `_inttypes`.
    let mut map = _inttypes.lock();
    if let Some(&existing) = map.get(&(sign, bits)) {
        return Ok(existing);
    }
    let created = create_int_type_checked(name, sign, bits)?;
    map.insert((sign, bits), created);
    Ok(created)
}

/// The host class object `build_int` created for `c`.
pub fn rint_class_object(c: RIntClass) -> HostObject {
    RINT_CLASS_OBJECTS.lock()[c.id as usize].clone()
}

/// `build_int(Some(name), sign, bits, false)` unwrapped as [`RIntClass`].
#[cfg(test)]
pub(crate) fn built_rint(name: &str, sign: bool, bits: u16) -> RIntClass {
    match build_int(Some(name), sign, bits, false).unwrap_or_else(|e| panic!("{e}")) {
        KnownType::RInt(c) => c,
        other => panic!("expected KnownType::RInt for {name}, got {other:?}"),
    }
}

/// Payload of `signed_int.__new__` OverflowError (`rarithmetic.py`).
pub fn signed_int_overflow_message(value: impl fmt::Display, bits: u16) -> String {
    format!("{value} does not fit in signed {bits}-bit integer")
}

/// Normalize `val` the way `signed_int.__new__` / `unsigned_int.__new__`
/// do, without wrapping a Python instance.
pub fn rint_normalize(class: RIntClass, value: i128) -> Result<i128, String> {
    if class.sign {
        let max = (class.mask() >> 1) as i128;
        let min = -max - 1;
        if value > max || value < min {
            return Err(signed_int_overflow_message(value, class.bits));
        }
        Ok(value)
    } else {
        Ok((value as u128 & class.mask()) as i128)
    }
}

/// `signed_int.__new__` / `unsigned_int.__new__` (`rarithmetic.py`).
///
/// Unsigned masks with `RIntClass::mask`. Signed raises when `value`
/// is outside `[-2**(bits-1), 2**(bits-1)-1]`. The Err string is the
/// OverflowError payload; callers that stringify the exception add
/// the `OverflowError: ` prefix themselves. Each `Ok` is a fresh
/// `ConstValue::RInt` instance.
pub fn rint_new(class: RIntClass, value: i128) -> Result<ConstValue, String> {
    Ok(ConstValue::rint(class, rint_normalize(class, value)?))
}

/// Stored Number i128 interpreted by `class.sign`.
///
/// Unsigned classes store the bit pattern (`unsigned_int.__new__` masks
/// into a non-negative integer; `u128::MAX` is `-1i128`). Every numeric
/// reader goes through [`rint_value`] rather than treating the i128 as signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RIntValue {
    Signed(i128),
    Unsigned(u128),
}

impl RIntValue {
    pub fn as_u128(self) -> u128 {
        match self {
            Self::Signed(n) => n as u128,
            Self::Unsigned(n) => n,
        }
    }

    pub fn as_f64(self) -> f64 {
        match self {
            Self::Signed(n) => n as f64,
            Self::Unsigned(n) => n as f64,
        }
    }

    pub fn as_i64(self) -> i64 {
        match self {
            Self::Signed(n) => n as i64,
            Self::Unsigned(n) => n as i64,
        }
    }

    pub fn as_i128(self) -> Option<i128> {
        match self {
            Self::Signed(n) => Some(n),
            Self::Unsigned(n) => i128::try_from(n).ok(),
        }
    }

    pub fn is_zero(self) -> bool {
        match self {
            Self::Signed(n) => n == 0,
            Self::Unsigned(n) => n == 0,
        }
    }

    pub fn is_negative(self) -> bool {
        matches!(self, Self::Signed(n) if n < 0)
    }

    /// Numeric ordering of `base_int` / `int` / `long` values.
    pub fn cmp_numeric(self, other: Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Signed(a), Self::Signed(b)) => a.cmp(&b),
            (Self::Unsigned(a), Self::Unsigned(b)) => a.cmp(&b),
            (Self::Signed(a), Self::Unsigned(b)) => {
                if a < 0 {
                    std::cmp::Ordering::Less
                } else {
                    (a as u128).cmp(&b)
                }
            }
            (Self::Unsigned(a), Self::Signed(b)) => {
                if b < 0 {
                    std::cmp::Ordering::Greater
                } else {
                    a.cmp(&(b as u128))
                }
            }
        }
    }
}

impl fmt::Display for RIntValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Signed(n) => write!(f, "{n}"),
            Self::Unsigned(n) => write!(f, "{n}"),
        }
    }
}

/// Interpret a stored Number i128 by `class.sign`.
///
/// Unsigned stores the bit pattern (`u128::MAX` is `-1i128`).
pub fn rint_value(class: RIntClass, v: i128) -> RIntValue {
    if class.sign {
        RIntValue::Signed(v)
    } else {
        RIntValue::Unsigned(v as u128)
    }
}

/// `SIGNED` / `BITS` on a `base_int` subclass, or `None` when `t` is
/// not an rarithmetic integer class.
pub(crate) fn integer_signed_and_bits(t: KnownType) -> Option<(bool, u16)> {
    match t {
        KnownType::Int | KnownType::Rint => Some((true, long_bit())),
        KnownType::Ruint => Some((false, long_bit())),
        KnownType::LongLong => Some((true, 64)),
        KnownType::ULongLong => Some((false, 64)),
        KnownType::LongLongLong => Some((true, 128)),
        KnownType::ULongLongLong => Some((false, 128)),
        KnownType::RInt(c) => Some((c.sign, c.bits)),
        _ => None,
    }
}

fn try_compute_restype(self_type: KnownType, other_type: KnownType) -> Option<KnownType> {
    // Upstream: `if self_type is other_type:`.
    if self_type == other_type {
        if self_type == KnownType::Bool {
            return Some(KnownType::Int);
        }
        return Some(self_type);
    }
    // Upstream: `if other_type in (bool, int, long):`.
    if matches!(other_type, KnownType::Bool | KnownType::Int) {
        if self_type == KnownType::Bool {
            return Some(KnownType::Int);
        }
        return Some(self_type);
    }
    // Upstream: `if self_type in (bool, int, long):`.
    if matches!(self_type, KnownType::Bool | KnownType::Int) {
        return Some(other_type);
    }
    // Upstream: `if self_type is float or other_type is float:`.
    if self_type == KnownType::Float || other_type == KnownType::Float {
        return Some(KnownType::Float);
    }
    // Upstream: `if self_type.SIGNED == other_type.SIGNED:
    //            return build_int(None, self_type.SIGNED,
    //                             max(self_type.BITS, other_type.BITS))`.
    if let (Some((s1, b1)), Some((s2, b2))) = (
        integer_signed_and_bits(self_type),
        integer_signed_and_bits(other_type),
    ) {
        if s1 == s2 {
            return build_int(None, s1, b1.max(b2), false).ok();
        }
    }
    None
}

/// RPython `compute_restype(self_type, other_type)` (rarithmetic.py).
///
/// ```python
/// def compute_restype(self_type, other_type):
///     if self_type is other_type:
///         if self_type is bool:
///             return int
///         return self_type
///     if other_type in (bool, int, long):
///         if self_type is bool:
///             return int
///         return self_type
///     if self_type in (bool, int, long):
///         return other_type
///     if self_type is float or other_type is float:
///         return float
///     if self_type.SIGNED == other_type.SIGNED:
///         return build_int(None, self_type.SIGNED,
///                          max(self_type.BITS, other_type.BITS))
///     raise AssertionError("Merging these types (%s, %s) is not supported" % (self_type, other_type))
/// ```
///
/// Rust port scopes the domain to [`KnownType`] variants the annotator
/// actually exercises. Upstream `long` is Python 2's arbitrary-precision
/// int, folded into [`KnownType::Int`] for Rust. `r_int` maps to
/// [`KnownType::Rint`] and is not in the `(bool, int, long)`
/// short-circuit. `r_uint` maps to [`KnownType::Ruint`]. `r_longlong` /
/// `r_ulonglong` map to the corresponding 64-bit [`KnownType`] variants
/// when they are not interned as `r_int` / `r_uint`.
pub fn compute_restype(self_type: KnownType, other_type: KnownType) -> KnownType {
    if let Some(restype) = try_compute_restype(self_type, other_type) {
        return restype;
    }
    if let (Some((s1, b1)), Some((s2, b2))) = (
        integer_signed_and_bits(self_type),
        integer_signed_and_bits(other_type),
    ) {
        if s1 == s2 {
            panic!(
                "{}",
                build_int(None, s1, b1.max(b2), false).expect_err("merge failed")
            );
        }
    }
    panic!(
        "compute_restype: unsupported merge of {:?} and {:?}",
        self_type, other_type
    );
}

/// `signed_int.__new__` / `unsigned_int.__new__` / `int(value)` for a restype.
pub fn construct_int_type(restype: KnownType, value: RIntValue) -> Option<ConstValue> {
    match restype {
        KnownType::Float => Some(ConstValue::float(value.as_f64())),
        KnownType::Int => i64::try_from(value.as_i128()?).ok().map(ConstValue::Int),
        KnownType::Bool => Some(ConstValue::Bool(!value.is_zero())),
        KnownType::RInt(c) => {
            let n = if c.sign {
                value.as_i128()?
            } else {
                value.as_u128() as i128
            };
            rint_new(c, n).ok()
        }
        KnownType::Rint | KnownType::LongLong => {
            let n = value.as_i128()?;
            let bits = integer_signed_and_bits(restype)?.1;
            let max = (mask_u128(bits) >> 1) as i128;
            let min = -max - 1;
            if n > max || n < min {
                return None;
            }
            i64::try_from(n).ok().map(ConstValue::Int)
        }
        KnownType::Ruint | KnownType::ULongLong => {
            let bits = integer_signed_and_bits(restype)?.1;
            let masked = value.as_u128() & mask_u128(bits);
            u64::try_from(masked).ok().map(ConstValue::UInt)
        }
        KnownType::LongLongLong => value.as_i128().map(ConstValue::Int128),
        KnownType::ULongLongLong => Some(ConstValue::UInt128(value.as_u128())),
        _ => None,
    }
}

fn mask_u128(bits: u16) -> u128 {
    if bits == 0 {
        0
    } else if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// `base_int._widen(self, other, value)` (`rarithmetic.py`).
///
/// ```python
/// self_type = type(self)
/// other_type = type(other)
/// try:
///     return self.typemap[self_type, other_type](value)
/// except KeyError:
///     pass
/// restype = compute_restype(self_type, other_type)
/// self.typemap[self_type, other_type] = restype
/// return restype(value)
/// ```
///
/// `self.typemap` is `signed_int.typemap` or `unsigned_int.typemap`.
/// Mixed signedness is `AssertionError` in `compute_restype` — the fold
/// declines (`None`). The typemap lock is not held across
/// `compute_restype` (`build_int` takes `_inttypes`).
pub fn _widen(self_type: KnownType, other_type: KnownType, value: RIntValue) -> Option<ConstValue> {
    let key = (self_type, other_type);
    let typemap = typemap_for(self_type);
    if let Some(typemap) = typemap {
        if let Some(&restype) = typemap.lock().get(&key) {
            return construct_int_type(restype, value);
        }
    }
    let restype = try_compute_restype(self_type, other_type)?;
    if let Some(typemap) = typemap {
        typemap.lock().insert(key, restype);
    }
    construct_int_type(restype, value)
}

/// `r_singlefloat(floatval)` (`rarithmetic.py`): `struct.pack("f", floatval)`.
/// `ieee.float_pack` raises `OverflowError` when a finite value overflows
/// the format (`"float too large to pack in this format"`). Inf/NaN pack.
pub fn r_singlefloat_bits(floatval: f64) -> Result<u32, &'static str> {
    if floatval.is_nan() || floatval.is_infinite() {
        return Ok((floatval as f32).to_bits());
    }
    let packed = floatval as f32;
    if packed.is_infinite() {
        return Err("float too large to pack in this format");
    }
    Ok(packed.to_bits())
}

/// `r_longfloat(floatval)` (`rarithmetic.py`): stores `floatval` as a Python float.
pub fn r_longfloat_bits(floatval: f64) -> u64 {
    floatval.to_bits()
}

/// RPython `signedtype(t)` (rarithmetic.py).
///
/// ```python
/// @specialize.memo()
/// def signedtype(t):
///     if t in (bool, int, long):
///         return True
///     else:
///         return t.SIGNED
/// ```
pub fn signedtype(t: KnownType) -> bool {
    match t {
        // Upstream: `if t in (bool, int, long): return True`.
        KnownType::Bool | KnownType::Int => true,
        // Upstream: `return t.SIGNED`. `r_int` is a `signed_int`
        // subclass, not builtin `int`.
        KnownType::Rint | KnownType::LongLong | KnownType::LongLongLong => true,
        KnownType::Ruint | KnownType::ULongLong | KnownType::ULongLongLong => false,
        KnownType::RInt(c) => c.sign,
        _ => false,
    }
}

/// RPython `normalizedinttype(t)` (`rarithmetic.py`).
///
/// ```python
/// def normalizedinttype(t):
///     if t is int:
///         return int
///     if t.BITS <= r_int.BITS:
///         return build_int(None, t.SIGNED, r_int.BITS)
///     else:
///         assert t.BITS <= r_longlong.BITS
///         return build_int(None, t.SIGNED, r_longlong.BITS)
/// ```
pub fn normalizedinttype(t: KnownType) -> KnownType {
    if t == KnownType::Int {
        return t;
    }
    let (signed, bits) = integer_signed_and_bits(t)
        .unwrap_or_else(|| panic!("normalizedinttype: {t:?} is not an integer type"));
    let r_int_bits = long_bit();
    if bits <= r_int_bits {
        build_int(None, signed, r_int_bits, false).unwrap_or_else(|e| panic!("{e}"))
    } else {
        // `r_longlong = build_int('r_longlong', True, 64)`.
        let r_longlong_bits = integer_signed_and_bits(KnownType::LongLong)
            .expect("r_longlong")
            .1;
        assert!(
            bits <= r_longlong_bits,
            "normalizedinttype: {bits} > r_longlong.BITS"
        );
        build_int(None, signed, r_longlong_bits, false).unwrap_or_else(|e| panic!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_bool_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Bool),
            KnownType::Int
        );
    }

    #[test]
    fn int_int_stays_int() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_ruint_picks_ruint() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Ruint),
            KnownType::Ruint
        );
        assert_eq!(
            compute_restype(KnownType::Ruint, KnownType::Int),
            KnownType::Ruint
        );
    }

    #[test]
    fn bool_int_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_float_widens_to_float() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Float),
            KnownType::Float
        );
    }

    #[test]
    fn r_singlefloat_bits_raises_on_overflow_and_packs_inf() {
        assert!(r_singlefloat_bits(1e300).is_err());
        assert!(r_singlefloat_bits(f64::INFINITY).unwrap() == f32::INFINITY.to_bits());
        assert!(r_singlefloat_bits(2.1).is_ok());
    }

    #[test]
    fn normalizedinttype_widens_to_r_int() {
        assert_eq!(normalizedinttype(KnownType::Int), KnownType::Int);
        assert_eq!(normalizedinttype(KnownType::Rint), KnownType::Rint);
        assert_eq!(normalizedinttype(KnownType::Ruint), KnownType::Ruint);
        let uchar = built_rint("r_uchar", false, 8);
        let schar = built_rint("r_schar", true, 8);
        assert_eq!(normalizedinttype(KnownType::RInt(uchar)), KnownType::Ruint);
        assert_eq!(normalizedinttype(KnownType::RInt(schar)), KnownType::Rint);
    }

    #[test]
    fn signedtype_follows_rarithmetic_signed_flag() {
        assert!(signedtype(KnownType::Bool));
        assert!(signedtype(KnownType::Int));
        assert!(signedtype(KnownType::Rint));
        assert!(signedtype(KnownType::LongLong));
        assert!(!signedtype(KnownType::Ruint));
        assert!(!signedtype(KnownType::ULongLong));
        let uchar = built_rint("r_uchar", false, 8);
        let short = built_rint("r_short", true, 16);
        assert!(!signedtype(KnownType::RInt(uchar)));
        assert!(signedtype(KnownType::RInt(short)));
    }

    #[test]
    fn build_int_maps_predefined_and_force_creation() {
        let long_bit = long_bit();
        assert_eq!(build_int(None, true, long_bit, false), Ok(KnownType::Rint));
        assert_eq!(
            build_int(None, false, long_bit, false),
            Ok(KnownType::Ruint)
        );
        let expected_64 = if long_bit == 64 {
            KnownType::Rint
        } else {
            KnownType::LongLong
        };
        assert_eq!(build_int(None, true, 64, false), Ok(expected_64));

        let err = build_int(None, false, 24, false).expect_err("no predefined uint24");
        assert_eq!(err, "TypeError: No predefined uint24");

        let a = build_int(Some("r_forced"), true, 16, true).unwrap();
        let b = build_int(Some("r_forced"), true, 16, true).unwrap();
        assert_ne!(a, b);
        let forced = build_int(Some("r_uint_forced"), false, long_bit, true).unwrap();
        assert_ne!(forced, KnownType::Ruint);
        match forced {
            KnownType::RInt(c) => {
                assert!(!c.sign);
                assert_eq!(c.bits, long_bit);
                assert_eq!(c.name, "r_uint_forced");
            }
            other => panic!("force_creation builds RInt, got {other:?}"),
        }
    }

    #[test]
    fn rint_new_unsigned_masks_and_signed_overflows() {
        let uchar = built_rint("r_uchar", false, 8);
        let schar = built_rint("r_schar", true, 8);
        assert_eq!(rint_new(uchar, -1), Ok(ConstValue::rint(uchar, 255)));
        assert_eq!(rint_new(uchar, 256), Ok(ConstValue::rint(uchar, 0)));
        let err = rint_new(schar, 128).expect_err("signed overflow");
        assert_eq!(err, "128 does not fit in signed 8-bit integer");
        assert_eq!(rint_value(uchar, 255), RIntValue::Unsigned(255));
        assert_eq!(rint_value(schar, -1), RIntValue::Signed(-1));
        let u128_forced = match build_int(Some("r_u128_forced"), false, 128, true)
            .unwrap_or_else(|e| panic!("{e}"))
        {
            KnownType::RInt(c) => c,
            other => panic!("expected RInt, got {other:?}"),
        };
        assert_eq!(rint_value(u128_forced, -1), RIntValue::Unsigned(u128::MAX));
        assert_eq!(rint_value(u128_forced, -1).as_u128(), u128::MAX);
        assert_eq!(rint_value(u128_forced, -1).as_f64(), u128::MAX as f64);
    }

    #[test]
    fn created_class_subclasses_signed_or_unsigned_int() {
        let uchar = built_rint("r_uchar", false, 8);
        let uchar_host = rint_class_object(uchar);
        let unsigned_int = unsigned_int_class();
        let bases = uchar_host.class_bases().expect("r_uchar is a class");
        assert!(
            bases.iter().any(|b| b == &unsigned_int),
            "r_uchar bases: {:?}",
            bases
                .iter()
                .map(|b| b.qualname().to_string())
                .collect::<Vec<_>>()
        );
        assert_eq!(uchar_host.class_get("SIGN"), Some(ConstValue::Bool(false)));
        assert_eq!(uchar_host.class_get("BITS"), Some(ConstValue::Int(8)));
        assert_eq!(uchar_host.class_get("MASK"), Some(ConstValue::Int(255)));
        assert_eq!(
            unsigned_int.class_get("SIGNED"),
            Some(ConstValue::Bool(false))
        );

        let short = built_rint("r_short", true, 16);
        let short_host = rint_class_object(short);
        let signed_int = signed_int_class();
        let bases = short_host.class_bases().expect("r_short is a class");
        assert!(bases.iter().any(|b| b == &signed_int));
        assert_eq!(short_host.class_get("SIGN"), Some(ConstValue::Bool(true)));
        assert_eq!(short_host.class_get("BITS"), Some(ConstValue::Int(16)));
        assert_eq!(short_host.class_get("MASK"), Some(ConstValue::Int(0xFFFF)));
        assert_eq!(signed_int.class_get("SIGNED"), Some(ConstValue::Bool(true)));
        let base_int = base_int_class();
        assert!(
            signed_int
                .class_bases()
                .unwrap()
                .iter()
                .any(|b| b == &base_int)
        );
        assert!(
            unsigned_int
                .class_bases()
                .unwrap()
                .iter()
                .any(|b| b == &base_int)
        );
        let host_int = crate::flowspace::model::builtin_int_class();
        assert!(
            base_int.is_subclass_of(&host_int),
            "base_int(long): this port's long is builtin int"
        );
        assert_eq!(
            host_int,
            crate::flowspace::model::HOST_ENV
                .lookup_builtin("int")
                .expect("builtin int")
        );
    }

    #[test]
    fn compute_restype_rint_and_short_keeps_rint() {
        // `r_int` is not in `(bool, int, long)`; a short-circuit that
        // listed it would return the narrower `r_short`.
        let short = KnownType::RInt(built_rint("r_short", true, 16));
        assert_eq!(compute_restype(KnownType::Rint, short), KnownType::Rint);
        assert_eq!(compute_restype(short, KnownType::Rint), KnownType::Rint);
    }

    #[test]
    fn widen_uchar_wraps_through_unsigned_new() {
        let uchar = built_rint("r_uchar", false, 8);
        let t = KnownType::RInt(uchar);
        assert_eq!(
            _widen(t, t, RIntValue::Unsigned(300)),
            Some(ConstValue::rint(uchar, 44))
        );
        assert_eq!(
            _widen(t, KnownType::Int, RIntValue::Unsigned(1)),
            Some(ConstValue::rint(uchar, 1))
        );
    }

    #[test]
    fn widen_typemap_caches_per_class() {
        let s = match build_int(Some("r_widen_typemap_s"), true, 20, true)
            .unwrap_or_else(|e| panic!("{e}"))
        {
            KnownType::RInt(c) => KnownType::RInt(c),
            other => panic!("expected RInt, got {other:?}"),
        };
        let u = match build_int(Some("r_widen_typemap_u"), false, 20, true)
            .unwrap_or_else(|e| panic!("{e}"))
        {
            KnownType::RInt(c) => KnownType::RInt(c),
            other => panic!("expected RInt, got {other:?}"),
        };
        let first_s = _widen(s, KnownType::Int, RIntValue::Signed(1));
        let second_s = _widen(s, KnownType::Int, RIntValue::Signed(1));
        assert_eq!(first_s, second_s);
        let first_u = _widen(u, KnownType::Int, RIntValue::Unsigned(1));
        let second_u = _widen(u, KnownType::Int, RIntValue::Unsigned(1));
        assert_eq!(first_u, second_u);

        let signed_map = SIGNED_INT.typemap.lock();
        let unsigned_map = UNSIGNED_INT.typemap.lock();
        assert_eq!(
            signed_map
                .keys()
                .filter(|k| **k == (s, KnownType::Int))
                .count(),
            1
        );
        assert_eq!(
            unsigned_map
                .keys()
                .filter(|k| **k == (u, KnownType::Int))
                .count(),
            1
        );
        assert_eq!(signed_map.get(&(s, KnownType::Int)).copied(), Some(s));
        assert_eq!(unsigned_map.get(&(u, KnownType::Int)).copied(), Some(u));
        assert!(signed_map.get(&(u, KnownType::Int)).is_none());
        assert!(unsigned_map.get(&(s, KnownType::Int)).is_none());
    }

    #[test]
    fn build_int_rejects_width_above_128() {
        let err = build_int(None, false, 256, false).expect_err("no predefined uint256");
        assert_eq!(err, "TypeError: No predefined uint256");
        let err = build_int(Some("r_too_wide"), true, 129, false).expect_err("129 bits");
        assert_eq!(
            err,
            "build_int: 129-bit integers exceed the 128-bit value carrier"
        );
        let err = build_int(Some("r_too_wide"), false, 256, true).expect_err("256 bits forced");
        assert_eq!(
            err,
            "build_int: 256-bit integers exceed the 128-bit value carrier"
        );
    }

    #[test]
    fn build_int_63_bit_mask_fits_i64() {
        for (name, sign) in [("r_u63", false), ("r_s63", true)] {
            let class =
                match build_int(Some(name), sign, 63, true).unwrap_or_else(|e| panic!("{e}")) {
                    KnownType::RInt(c) => c,
                    other => panic!("expected RInt for {name}, got {other:?}"),
                };
            assert_eq!(class.bits, 63);
            assert_eq!(class.sign, sign);
            assert_eq!(
                rint_class_object(class).class_get("MASK"),
                Some(ConstValue::Int(9223372036854775807))
            );
        }
    }
}
