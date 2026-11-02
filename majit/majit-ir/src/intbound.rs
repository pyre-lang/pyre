/// Integer bound tracking for the optimizer.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intutils.py.
///
/// The abstract domain tracks a (signed) upper and lower bound (both ends
/// inclusive) for every integer variable in the trace. It also tracks which bits
/// of a range are known 0 or known 1 (the remaining bits are unknown). The ranges
/// and the known bits feed back into each other, ie we can improve the range if
/// some upper bits have known values, and we can learn some known bits from the
/// range too.
///
/// A tristate number represents partial knowledge about an integer:
/// - tvalue: the known bits (where tmask is 0)
/// - tmask: 1 where the bit is unknown, 0 where it's known
/// - the combination tvalue=1, tmask=1 at the same bit position is forbidden
pub use crate::optimize::InvalidLoop;

// ── Free helper functions ──

/// Sets all unknowns determined by `mask` in `value` bit-wise to 0 and
/// returns the result.
#[inline(always)]
fn unmask_zero(value: u64, mask: u64) -> u64 {
    value & !mask
}

/// Sets all unknowns determined by `mask` in `value` bit-wise to 1 and
/// returns the result.
#[inline(always)]
fn unmask_one(value: u64, mask: u64) -> u64 {
    value | mask
}

/// Returns `v` with all bits except the most significant bit set to 0.
#[inline(always)]
fn msbonly(v: u64) -> u64 {
    v & (1u64 << 63)
}

/// Calculate next power of 2 minus one, greater than or equal to n.
/// Smears the highest set bit down to all lower positions.
#[inline]
fn next_pow2_m1(mut n: u64) -> u64 {
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n
}

/// Calculates a bitmask in which only the leading zeros of `n` are set (1).
#[inline(always)]
fn leading_zeros_mask(n: u64) -> u64 {
    !next_pow2_m1(n)
}

/// Returns val with all bits deleted but the lowest one that was set.
#[inline(always)]
fn lowest_set_bit_only(val: u64) -> u64 {
    let working = !val;
    let increased = working.wrapping_add(1);
    (working ^ increased) & val
}

/// Flip the most significant bit.
#[inline(always)]
fn flip_msb(val: u64) -> u64 {
    val ^ (1u64 << 63)
}

/// Returns `true` iff `tvalue` and `tmask` form a valid tristate number
/// (no bit position has both tvalue=1 and tmask=1).
#[inline(always)]
fn is_valid_tnum(tvalue: u64, tmask: u64) -> bool {
    (tvalue & tmask) == 0
}

/// Returns a * b. If the multiplication overflows, returns MAXINT or MININT
/// (whichever is closer to the "correct" value).
fn saturating_mul(a: i64, b: i64) -> i64 {
    match a.checked_mul(b) {
        Some(v) => v,
        None => {
            if same_sign(a, b) {
                i64::MAX
            } else {
                i64::MIN
            }
        }
    }
}

/// Return true iff a and b have the same sign.
#[inline(always)]
fn same_sign(a: i64, b: i64) -> bool {
    (a ^ b) >= 0
}

/// Python-style floor division: rounds toward negative infinity.
#[inline]
fn py_div(a: i64, b: i64) -> i64 {
    // In Python: (-7) // 2 == -4, but in Rust/C: (-7) / 2 == -3
    let q = a / b;
    let r = a % b;
    // If the remainder is non-zero and the signs of a and b differ,
    // we need to subtract 1 from the quotient.
    if r != 0 && ((r ^ b) < 0) { q - 1 } else { q }
}

// ── IntBound ──

/// An abstract bound on an integer value.
///
/// Combines interval analysis [lower, upper] with known-bits analysis (tvalue, tmask).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntBound {
    /// Lower bound (inclusive). i64::MIN means unbounded below.
    pub lower: i64,
    /// Upper bound (inclusive). i64::MAX means unbounded above.
    pub upper: i64,
    /// Known bits: where tmask bit is 0, tvalue bit is the known value.
    pub tvalue: u64,
    /// Unknown bit mask: 1 means the bit is unknown.
    pub tmask: u64,
}

impl IntBound {
    // ── Constructors ──

    /// Internal constructor. Callers should prefer the static constructors below.
    fn new_raw(lower: i64, upper: i64, tvalue: u64, tmask: u64, do_shrinking: bool) -> Self {
        debug_assert!(
            is_valid_tnum(tvalue, tmask),
            "invalid tnum: tvalue={tvalue:#x}, tmask={tmask:#x}"
        );
        let mut b = IntBound {
            lower,
            upper,
            tvalue,
            tmask,
        };
        if do_shrinking {
            b.shrink();
        }
        b
    }

    /// Construct from lower, upper, tvalue, tmask with shrinking.
    pub fn new(lower: i64, upper: i64, tvalue: u64, tmask: u64) -> Self {
        Self::new_raw(lower, upper, tvalue, tmask, true)
    }

    /// Completely unbounded (contains every integer).
    pub fn unbounded() -> Self {
        IntBound {
            lower: i64::MIN,
            upper: i64::MAX,
            tvalue: 0,
            tmask: u64::MAX,
        }
    }

    /// Exact constant.
    pub fn from_constant(value: i64) -> Self {
        IntBound {
            lower: value,
            upper: value,
            tvalue: value as u64,
            tmask: 0,
        }
    }

    /// TODO: RPython's keyword-default constructor
    /// `IntBound(lower=lower, upper=upper)` (intutils.py) leaves
    /// `tvalue` / `tmask` at the `TNUM_ONLY_VALUE_DEFAULT` (= 0) and
    /// `TNUM_ONLY_MASK_DEFAULT` (= TNUM_UNKNOWN = r_uint(-1)) defaults
    /// while still running `shrink()`. Rust has no keyword defaults so
    /// `IntBound::new(lower, upper, 0, u64::MAX)` would have to repeat
    /// the magic constants at every call site (~150 sites). This named
    /// factory captures the exact same default semantics in one place.
    ///
    /// Bounded in [lower, upper] with shrinking; tvalue/tmask = TNUM_UNKNOWN.
    pub fn bounded(lower: i64, upper: i64) -> Self {
        debug_assert!(lower <= upper, "bounded: lower({lower}) > upper({upper})");
        Self::new_raw(lower, upper, 0, u64::MAX, true)
    }

    /// Known to be non-negative.
    pub fn nonnegative() -> Self {
        Self::bounded(0, i64::MAX)
    }

    /// From known bits with shrinking.
    pub fn from_knownbits(tvalue: u64, tmask: u64) -> Self {
        let tvalue = unmask_zero(tvalue, tmask);
        Self::new_raw(i64::MIN, i64::MAX, tvalue, tmask, true)
    }

    // ── Queries ──

    /// Whether the value is exactly known.
    pub fn is_constant(&self) -> bool {
        debug_assert!(
            (self.lower == self.upper) == (self.tmask == 0),
            "constant invariant violated: lower={}, upper={}, tmask={:#x}",
            self.lower,
            self.upper,
            self.tmask
        );
        self.tmask == 0
    }

    /// intutils.py `get_constant_int` — return the known constant value
    /// (panics if not constant).
    pub fn get_constant_int(&self) -> i64 {
        debug_assert!(self.is_constant());
        self.lower
    }

    /// Whether the value is known to be non-negative.
    pub fn known_nonnegative(&self) -> bool {
        self.lower >= 0
    }

    /// intutils.py `IntBound.getnullness()` parity (line-by-line port).
    ///
    /// ```python
    /// def getnullness(self):
    ///     if self.known_gt_const(0) or \
    ///        self.known_lt_const(0) or \
    ///        self.tvalue != 0:
    ///         return INFO_NONNULL
    ///     if self.is_constant() and self.get_constant_int() == 0:
    ///         return INFO_NULL
    ///     return INFO_UNKNOWN
    /// ```
    ///
    /// Returns RPython's `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py) so callers can compare directly against the
    /// upstream constants without remapping.
    pub fn getnullness(&self) -> i8 {
        if self.known_gt_const(0) || self.known_lt_const(0) || self.tvalue != 0 {
            crate::optimize::INFO_NONNULL
        } else if self.is_constant() && self.get_constant_int() == 0 {
            crate::optimize::INFO_NULL
        } else {
            crate::optimize::INFO_UNKNOWN
        }
    }

    /// intutils.py `_are_knownbits_implied` — true when the stored
    /// known-bits tnum equals the tnum implied by the lower/upper bounds.
    pub fn are_knownbits_implied(&self) -> bool {
        let (implied_tvalue, implied_tmask) = self._tnum_implied_by_bounds();
        self.tvalue == implied_tvalue && self.tmask == implied_tmask
    }

    /// Whether this abstract integer is unbounded.
    pub fn is_unbounded(&self) -> bool {
        self.lower == i64::MIN
            && self.upper == i64::MAX
            && self.tvalue == 0
            && self.tmask == u64::MAX
    }

    /// Whether the value is known to represent a boolean (0 or 1).
    pub fn is_bool(&self) -> bool {
        self.known_nonnegative() && self.upper <= 1
    }

    /// Whether this abstract integer contains `value`.
    pub fn contains(&self, value: i64) -> bool {
        if value < self.lower || value > self.upper {
            return false;
        }
        let u_vself = unmask_zero(self.tvalue, self.tmask);
        let u_value = unmask_zero(value as u64, self.tmask);
        u_vself == u_value
    }

    /// Check if all numbers are between lower and upper.
    pub fn is_within_range(&self, lower: i64, upper: i64) -> bool {
        lower <= self.lower && self.upper <= upper
    }

    // ── Signed comparisons ──

    pub fn known_lt_const(&self, value: i64) -> bool {
        self.upper < value
    }

    pub fn known_le_const(&self, value: i64) -> bool {
        self.upper <= value
    }

    pub fn known_gt_const(&self, value: i64) -> bool {
        self.lower > value
    }

    pub fn known_ge_const(&self, value: i64) -> bool {
        self.lower >= value
    }

    pub fn known_eq_const(&self, value: i64) -> bool {
        self.is_constant() && self.lower == value
    }

    /// Whether the value is known to be less than `other`.
    pub fn known_lt(&self, other: &IntBound) -> bool {
        self.known_lt_const(other.lower)
    }

    /// Whether the value is known to be less than or equal to `other`.
    pub fn known_le(&self, other: &IntBound) -> bool {
        self.known_le_const(other.lower)
    }

    /// Whether the value is known to be greater than `other`.
    pub fn known_gt(&self, other: &IntBound) -> bool {
        other.known_lt(self)
    }

    /// Whether the value is known to be greater than or equal to `other`.
    pub fn known_ge(&self, other: &IntBound) -> bool {
        other.known_le(self)
    }

    /// Return true if the sets of numbers self and other must be disjoint.
    pub fn known_ne(&self, other: &IntBound) -> bool {
        if self.known_lt(other) || self.known_gt(other) {
            return true;
        }
        let both_known = self.tmask | other.tmask;
        if unmask_zero(self.tvalue, both_known) != unmask_zero(other.tvalue, both_known) {
            return true;
        }
        let mut newself = self.clone();
        newself.intersect(other).is_err()
    }

    // ── Unsigned comparisons ──

    fn _known_same_sign(&self, other: &IntBound) -> bool {
        if self.known_nonnegative() && other.known_nonnegative() {
            return true;
        }
        self.known_lt_const(0) && other.known_lt_const(0)
    }

    pub fn get_minimum_unsigned_by_knownbits(&self) -> u64 {
        unmask_zero(self.tvalue, self.tmask)
    }

    pub fn get_maximum_unsigned_by_knownbits(&self) -> u64 {
        unmask_one(self.tvalue, self.tmask)
    }

    pub fn known_unsigned_lt(&self, other: &IntBound) -> bool {
        if self._known_same_sign(other) && self.known_lt(other) {
            return true;
        }
        let other_min = other.get_minimum_unsigned_by_knownbits();
        let self_max = self.get_maximum_unsigned_by_knownbits();
        self_max < other_min
    }

    pub fn known_unsigned_le(&self, other: &IntBound) -> bool {
        if self._known_same_sign(other) && self.known_le(other) {
            return true;
        }
        let other_min = other.get_minimum_unsigned_by_knownbits();
        let self_max = self.get_maximum_unsigned_by_knownbits();
        self_max <= other_min
    }

    pub fn known_unsigned_gt(&self, other: &IntBound) -> bool {
        other.known_unsigned_lt(self)
    }

    pub fn known_unsigned_ge(&self, other: &IntBound) -> bool {
        other.known_unsigned_le(self)
    }

    // ── Mutation (make_le, make_lt, make_ge, make_gt, etc.) ──

    /// Constrain self to only contain values <= other.upper.
    pub fn make_le(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        self.make_le_const(other.upper)
    }

    pub fn make_le_const(&mut self, value: i64) -> Result<bool, InvalidLoop> {
        if value < self.upper {
            if value < self.lower {
                return Err(InvalidLoop("make_le_const: empty interval"));
            }
            self.upper = value;
            self.shrink();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Constrain self to only contain values < other.upper.
    pub fn make_lt(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        self.make_lt_const(other.upper)
    }

    pub fn make_lt_const(&mut self, value: i64) -> Result<bool, InvalidLoop> {
        if value == i64::MIN {
            return Err(InvalidLoop("intbound can't be made smaller than MININT"));
        }
        self.make_le_const(value - 1)
    }

    /// Constrain self to only contain values >= other.lower.
    pub fn make_ge(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        self.make_ge_const(other.lower)
    }

    pub fn make_ge_const(&mut self, value: i64) -> Result<bool, InvalidLoop> {
        if value > self.lower {
            if value > self.upper {
                return Err(InvalidLoop("make_ge_const: empty interval"));
            }
            self.lower = value;
            self.shrink();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Constrain self to only contain values > other.lower.
    pub fn make_gt(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        self.make_gt_const(other.lower)
    }

    pub fn make_gt_const(&mut self, value: i64) -> Result<bool, InvalidLoop> {
        if value == i64::MAX {
            return Err(InvalidLoop("make_gt_const: can't be > MAXINT"));
        }
        self.make_ge_const(value + 1)
    }

    pub fn make_eq_const(&mut self, intval: i64) -> Result<(), InvalidLoop> {
        if !self.contains(intval) {
            return Err(InvalidLoop("constant int is outside of interval"));
        }
        self.upper = intval;
        self.lower = intval;
        self.tvalue = intval as u64;
        self.tmask = 0;
        Ok(())
    }

    pub fn make_ne_const(&mut self, intval: i64) -> bool {
        if self.lower < intval && intval == self.upper {
            self.upper -= 1;
            self.shrink();
            return true;
        }
        if self.lower == intval && intval < self.upper {
            self.lower += 1;
            self.shrink();
            return true;
        }
        false
    }

    pub fn make_bool(&mut self) {
        let _ = self.intersect_const(0, 1);
    }

    pub fn make_unsigned_le(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        if other.known_nonnegative() {
            self.intersect_const(0, other.upper)
        } else {
            Ok(false)
        }
    }

    pub fn make_unsigned_lt(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        if other.known_nonnegative() {
            debug_assert!(other.upper >= 0);
            if other.upper == 0 {
                return Err(InvalidLoop("make_unsigned_lt: other.upper == 0"));
            }
            self.intersect_const(0, other.upper - 1)
        } else {
            Ok(false)
        }
    }

    pub fn make_unsigned_ge(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        if other.upper < 0 {
            let changed = self.make_lt_const(0)?;
            Ok(self.make_ge(other)? || changed)
        } else if self.known_nonnegative() && other.known_nonnegative() {
            self.make_ge(other)
        } else {
            Ok(false)
        }
    }

    pub fn make_unsigned_gt(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        if other.upper < 0 {
            let changed = self.make_lt_const(0)?;
            Ok(self.make_gt(other)? || changed)
        } else if self.known_nonnegative() && other.known_nonnegative() {
            self.make_gt(other)
        } else {
            Ok(false)
        }
    }

    // ── Intersection ──

    /// Intersect bounds with range [lower, upper].
    pub fn intersect_const(&mut self, lower: i64, upper: i64) -> Result<bool, InvalidLoop> {
        self.intersect_const_inner(lower, upper, true)
    }

    fn intersect_const_inner(
        &mut self,
        lower: i64,
        upper: i64,
        do_shrinking: bool,
    ) -> Result<bool, InvalidLoop> {
        let mut changed = false;
        if lower > self.lower {
            if lower > self.upper {
                return Err(InvalidLoop("intersect_const: empty interval (lower)"));
            }
            self.lower = lower;
            changed = true;
        }
        if upper < self.upper {
            if upper < self.lower {
                return Err(InvalidLoop("intersect_const: empty interval (upper)"));
            }
            self.upper = upper;
            changed = true;
        }
        if changed && do_shrinking {
            self.shrink();
        }
        Ok(changed)
    }

    /// Intersect this bound with another (tighten).
    pub fn intersect(&mut self, other: &IntBound) -> Result<bool, InvalidLoop> {
        if self.known_gt(other) || self.known_lt(other) {
            return Err(InvalidLoop("two integer ranges don't overlap"));
        }

        let r = self.intersect_const_inner(other.lower, other.upper, false)?;

        let (tvalue, tmask, valid) = self._tnum_intersect(other.tvalue, other.tmask);
        if !valid {
            return Err(InvalidLoop("knownbits contradict each other"));
        }
        if self.tmask != tmask {
            let changed = self.set_tvalue_tmask(tvalue, tmask);
            debug_assert!(changed);
            Ok(changed)
        } else if r {
            self.shrink();
            Ok(r)
        } else {
            Ok(r)
        }
    }

    pub fn set_tvalue_tmask(&mut self, tvalue: u64, tmask: u64) -> bool {
        let changed = self.tvalue != tvalue || self.tmask != tmask;
        if changed {
            self.tvalue = tvalue;
            self.tmask = tmask;
            self.shrink();
        }
        changed
    }

    // ── Transfer functions ──

    /// Bound after addition.
    pub fn add_bound(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_add(other);

        let lower = match self.lower.checked_add(other.lower) {
            Some(v) => v,
            None => return IntBound::from_knownbits(tvalue, tmask),
        };
        let upper = match self.upper.checked_add(other.upper) {
            Some(v) => v,
            None => return IntBound::from_knownbits(tvalue, tmask),
        };
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// add self + constant
    pub fn add(&self, value: i64) -> IntBound {
        self.add_bound(&IntBound::from_constant(value))
    }

    /// Returns true if self + other can never overflow.
    pub fn add_bound_cannot_overflow(&self, other: &IntBound) -> bool {
        self.upper.checked_add(other.upper).is_some()
            && self.lower.checked_add(other.lower).is_some()
    }

    /// Return the bound that self + other must have, if no overflow occurred.
    pub fn add_bound_no_overflow(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_add(other);
        let lower = self.lower.checked_add(other.lower).unwrap_or(i64::MIN);
        let upper = self.upper.checked_add(other.upper).unwrap_or(i64::MAX);
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// Return the bound that self + self must have (multiply by 2), if no overflow.
    pub fn mul2_bound_no_overflow(&self) -> IntBound {
        let (tvalue, tmask) = self._tnum_lshift(1);
        let lower = self.lower.checked_add(self.lower).unwrap_or(i64::MIN);
        let upper = self.upper.checked_add(self.upper).unwrap_or(i64::MAX);
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// Bound after subtraction.
    pub fn sub_bound(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_sub(other);

        let lower = match self.lower.checked_sub(other.upper) {
            Some(v) => v,
            None => return IntBound::from_knownbits(tvalue, tmask),
        };
        let upper = match self.upper.checked_sub(other.lower) {
            Some(v) => v,
            None => return IntBound::from_knownbits(tvalue, tmask),
        };
        IntBound::new(lower, upper, tvalue, tmask)
    }

    pub fn sub_bound_cannot_overflow(&self, other: &IntBound) -> bool {
        self.lower.checked_sub(other.upper).is_some()
            && self.upper.checked_sub(other.lower).is_some()
    }

    /// Return the bound that self - other must have, if no overflow occurred.
    pub fn sub_bound_no_overflow(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_sub(other);
        let lower = self.lower.checked_sub(other.upper).unwrap_or(i64::MIN);
        let upper = self.upper.checked_sub(other.lower).unwrap_or(i64::MAX);
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// Bound after multiplication.
    pub fn mul_bound(&self, other: &IntBound) -> IntBound {
        let v1 = self.upper.checked_mul(other.upper);
        let v2 = self.upper.checked_mul(other.lower);
        let v3 = self.lower.checked_mul(other.upper);
        let v4 = self.lower.checked_mul(other.lower);

        match (v1, v2, v3, v4) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                let lower = a.min(b).min(c).min(d);
                let upper = a.max(b).max(c).max(d);
                IntBound::bounded(lower, upper)
            }
            _ => IntBound::unbounded(),
        }
    }

    /// mul_bound_no_overflow is same as mul_bound (can be improved).
    pub fn mul_bound_no_overflow(&self, other: &IntBound) -> IntBound {
        self.mul_bound(other)
    }

    pub fn mul_bound_cannot_overflow(&self, other: &IntBound) -> bool {
        self.upper.checked_mul(other.upper).is_some()
            && self.upper.checked_mul(other.lower).is_some()
            && self.lower.checked_mul(other.upper).is_some()
            && self.lower.checked_mul(other.lower).is_some()
    }

    /// Returns the bound of self**2, if no overflow occurred.
    pub fn square_bound_no_overflow(&self) -> IntBound {
        let val0 = saturating_mul(self.lower, self.lower);
        let val1 = saturating_mul(self.upper, self.upper);
        let mut lower = val0.min(val1);
        let upper = val0.max(val1);
        if !same_sign(self.upper, self.lower) {
            // 0 is contained in the range
            lower = 0;
        }
        IntBound::bounded(lower, upper)
    }

    /// Python-style floor division bound.
    pub fn py_div_bound(&self, other: &IntBound) -> IntBound {
        // We need 0 not in other's interval; also check that other doesn't straddle 0
        if !(other.contains(0) || other.lower < 0 && 0 < other.upper) {
            // py_div uses unchecked `/` and `%`, which panic on i64::MIN / -1.
            // Guard before evaluating the corners so we hit the fallback safely.
            if other.contains(-1) && (self.lower == i64::MIN || self.upper == i64::MIN) {
                return IntBound::unbounded();
            }
            let a = py_div(self.upper, other.upper);
            let b = py_div(self.upper, other.lower);
            let c = py_div(self.lower, other.upper);
            let d = py_div(self.lower, other.lower);
            let lower = a.min(b).min(c).min(d);
            let upper = a.max(b).max(c).max(d);
            IntBound::bounded(lower, upper)
        } else {
            IntBound::unbounded()
        }
    }

    /// Python-style modulo bound.
    pub fn mod_bound(&self, other: &IntBound) -> IntBound {
        if other.is_constant() && other.get_constant_int() == 0 {
            return IntBound::unbounded();
        }
        // with Python's modulo:  0 <= (x % pos) < pos
        //                        neg < (x % neg) <= 0
        let upper = if other.upper > 0 { other.upper - 1 } else { 0 };
        let lower = if other.lower < 0 { other.lower + 1 } else { 0 };
        IntBound::bounded(lower, upper)
    }

    /// Bound after left shift.
    pub fn lshift_bound(&self, other: &IntBound) -> IntBound {
        let (mut tvalue, mut tmask) = (0u64, u64::MAX); // TNUM_UNKNOWN
        if other.is_constant() {
            let c_other = other.get_constant_int();
            if c_other >= 64 {
                tvalue = 0;
                tmask = 0; // TNUM_KNOWN_ZERO
            } else if (0..64).contains(&c_other) {
                let shift = c_other as u32;
                let r = self._tnum_lshift(shift);
                tvalue = r.0;
                tmask = r.1;
            }
            // else: bits are unknown because arguments invalid
        }

        if other.known_nonnegative() && other.known_lt_const(64) {
            // Try to compute interval bounds
            let results = [
                checked_shl_i64(self.upper, other.upper),
                checked_shl_i64(self.upper, other.lower),
                checked_shl_i64(self.lower, other.upper),
                checked_shl_i64(self.lower, other.lower),
            ];
            if let (Some(a), Some(b), Some(c), Some(d)) =
                (results[0], results[1], results[2], results[3])
            {
                let lower = a.min(b).min(c).min(d);
                let upper = a.max(b).max(c).max(d);
                return IntBound::new(lower, upper, tvalue, tmask);
            }
        }

        IntBound::from_knownbits(tvalue, tmask)
    }

    /// Bound after arithmetic right shift.
    pub fn rshift_bound(&self, other: &IntBound) -> IntBound {
        let (mut tvalue, mut tmask) = (0u64, u64::MAX); // TNUM_UNKNOWN
        if other.is_constant() {
            let c_other = other.get_constant_int();
            if c_other >= 64 {
                // shift value out to the right, but do sign extend
                if msbonly(self.tmask) != 0 {
                    // sign bit is unknown => result unknown
                    // tvalue, tmask already TNUM_UNKNOWN
                } else if msbonly(self.tvalue) != 0 {
                    // sign is known 1 => all bits become 1
                    tvalue = u64::MAX;
                    tmask = 0;
                } else {
                    // sign is known 0 => result is 0
                    tvalue = 0;
                    tmask = 0;
                }
            } else if c_other >= 0 {
                let shift = c_other as u32;
                let r = self._tnum_rshift(shift);
                tvalue = r.0;
                tmask = r.1;
            }
            // else: bits are unknown because arguments invalid
        }

        let mut lower = i64::MIN;
        let mut upper = i64::MAX;
        if other.known_nonnegative() && other.known_lt_const(64) {
            let vals = [
                self.upper >> other.upper as u32,
                self.upper >> other.lower as u32,
                self.lower >> other.upper as u32,
                self.lower >> other.lower as u32,
            ];
            lower = vals[0].min(vals[1]).min(vals[2]).min(vals[3]);
            upper = vals[0].max(vals[1]).max(vals[2]).max(vals[3]);
        }
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// Bound after unsigned (logical) right shift.
    pub fn urshift_bound(&self, other: &IntBound) -> IntBound {
        let mut lower = i64::MIN;
        let mut upper = i64::MAX;
        let (mut tvalue, mut tmask) = (0u64, u64::MAX); // TNUM_UNKNOWN

        if other.is_constant() {
            let c_other = other.get_constant_int();
            if c_other >= 64 {
                tvalue = 0;
                tmask = 0; // TNUM_KNOWN_ZERO
            } else if c_other >= 0 {
                let shift = c_other as u32;
                let r = self._tnum_urshift(shift);
                tvalue = r.0;
                tmask = r.1;
                if self.lower >= 0 {
                    upper = ((self.upper as u64) >> shift) as i64;
                    lower = ((self.lower as u64) >> shift) as i64;
                }
            }
            // else: bits are unknown because arguments invalid
        }
        IntBound::new(lower, upper, tvalue, tmask)
    }

    /// Bound after bitwise AND.
    pub fn and_bound(&self, other: &IntBound) -> IntBound {
        let pos1 = self.known_nonnegative();
        let pos2 = other.known_nonnegative();

        let mut lower = i64::MIN;
        let mut upper = i64::MAX;
        if pos1 || pos2 {
            lower = 0;
        }
        if pos1 {
            upper = self.upper;
        }
        if pos2 {
            upper = upper.min(other.upper);
        }

        let (res_tvalue, res_tmask) = self._tnum_and(other);
        IntBound::new(lower, upper, res_tvalue, res_tmask)
    }

    /// Bound after bitwise OR.
    pub fn or_bound(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_or(other);
        IntBound::from_knownbits(tvalue, tmask)
    }

    /// Bound after bitwise XOR.
    pub fn xor_bound(&self, other: &IntBound) -> IntBound {
        let (tvalue, tmask) = self._tnum_xor(other);
        IntBound::from_knownbits(tvalue, tmask)
    }

    /// Arithmetic negation bound.
    pub fn neg_bound(&self) -> IntBound {
        let res = self.invert_bound();
        res.add(1)
    }

    /// Bitwise NOT bound.
    pub fn invert_bound(&self) -> IntBound {
        let upper = !self.lower;
        let lower = !self.upper;
        let tvalue = unmask_zero(!self.tvalue, self.tmask);
        let tmask = self.tmask;
        IntBound::new(lower, upper, tvalue, tmask)
    }

    // ── Backwards transfer functions ──

    pub fn and_bound_backwards(&self, result: &IntBound) -> Result<IntBound, InvalidLoop> {
        let (tvalue, tmask, valid) = self._tnum_and_backwards(result);
        if !valid {
            return Err(InvalidLoop("inconsistency in and_bound_backwards"));
        }
        Ok(IntBound::from_knownbits(tvalue, tmask))
    }

    pub fn or_bound_backwards(&self, result: &IntBound) -> Result<IntBound, InvalidLoop> {
        let (tvalue, tmask, valid) = self._tnum_or_backwards(result);
        if !valid {
            return Err(InvalidLoop("inconsistency in or_bound_backwards"));
        }
        Ok(IntBound::from_knownbits(tvalue, tmask))
    }

    pub fn rshift_bound_backwards(&self, other: &IntBound) -> IntBound {
        if !other.is_constant() {
            return IntBound::unbounded();
        }
        let c_other = other.get_constant_int();
        let (mut tvalue, mut tmask) = (0u64, u64::MAX);
        if (0..64).contains(&c_other) {
            let shift = c_other as u32;
            tvalue = self.tvalue << shift;
            tmask = self.tmask << shift;
            // shift ? in from the right
            tmask |= (1u64 << shift) - 1;
        }
        IntBound::from_knownbits(tvalue, tmask)
    }

    /// urshift_bound_backwards is the same as rshift_bound_backwards.
    pub fn urshift_bound_backwards(&self, other: &IntBound) -> IntBound {
        self.rshift_bound_backwards(other)
    }

    pub fn lshift_bound_backwards(&self, other: &IntBound) -> Result<IntBound, InvalidLoop> {
        if !other.is_constant() {
            return Ok(IntBound::unbounded());
        }
        let c_other = other.get_constant_int();
        let (mut tvalue, mut tmask) = (0u64, u64::MAX);
        if (0..64).contains(&c_other) {
            let shift = c_other as u32;
            tvalue = self.tvalue >> shift;
            tmask = self.tmask >> shift;
            // shift ? in from the left
            let s_tmask = !(u64::MAX >> shift);
            tmask |= s_tmask;
            let inconsistent = self.tvalue & ((1u64 << shift) - 1);
            if inconsistent != 0 {
                return Err(InvalidLoop(
                    "lshift_bound_backwards inconsistent known bits",
                ));
            }
        }
        Ok(IntBound::from_knownbits(tvalue, tmask))
    }

    pub fn lshift_bound_cannot_overflow(&self, other: &IntBound) -> bool {
        if other.known_nonnegative() && other.known_lt_const(64) {
            let results = [
                checked_shl_i64(self.upper, other.upper),
                checked_shl_i64(self.upper, other.lower),
                checked_shl_i64(self.lower, other.upper),
                checked_shl_i64(self.lower, other.lower),
            ];
            results.iter().all(|r| r.is_some())
        } else {
            false
        }
    }

    // ── Widen ──

    pub fn widen(&self) -> IntBound {
        let mut info = self.clone();
        info.widen_update();
        info
    }

    pub fn widen_update(&mut self) {
        if self.lower < i64::MIN / 2 {
            self.lower = i64::MIN;
        }
        if self.upper > i64::MAX / 2 {
            self.upper = i64::MAX;
        }
        self.tvalue = 0;
        self.tmask = u64::MAX;
        self.shrink();
    }

    // ── Internal: tristate number operations ──

    #[inline(always)]
    fn _tnum_add(&self, other: &IntBound) -> (u64, u64) {
        let sum_values = self.tvalue.wrapping_add(other.tvalue);
        let sum_masks = self.tmask.wrapping_add(other.tmask);
        let all_carries = sum_values.wrapping_add(sum_masks);
        let val_carries = all_carries ^ sum_values;
        let tmask = self.tmask | other.tmask | val_carries;
        let tvalue = unmask_zero(sum_values, tmask);
        (tvalue, tmask)
    }

    #[inline(always)]
    fn _tnum_sub(&self, other: &IntBound) -> (u64, u64) {
        let diff_values = self.tvalue.wrapping_sub(other.tvalue);
        let val_borrows =
            diff_values.wrapping_add(self.tmask) ^ diff_values.wrapping_sub(other.tmask);
        let tmask = self.tmask | other.tmask | val_borrows;
        let tvalue = unmask_zero(diff_values, tmask);
        (tvalue, tmask)
    }

    #[inline(always)]
    fn _tnum_and(&self, other: &IntBound) -> (u64, u64) {
        let self_pmask = self.tvalue | self.tmask;
        let other_pmask = other.tvalue | other.tmask;
        let and_vals = self.tvalue & other.tvalue;
        (and_vals, self_pmask & other_pmask & !and_vals)
    }

    #[inline(always)]
    fn _tnum_or(&self, other: &IntBound) -> (u64, u64) {
        let union_vals = self.tvalue | other.tvalue;
        let union_masks = self.tmask | other.tmask;
        (union_vals, union_masks & !union_vals)
    }

    #[inline(always)]
    fn _tnum_xor(&self, other: &IntBound) -> (u64, u64) {
        let xor_vals = self.tvalue ^ other.tvalue;
        let union_masks = self.tmask | other.tmask;
        (unmask_zero(xor_vals, union_masks), union_masks)
    }

    #[inline(always)]
    fn _tnum_lshift(&self, shift: u32) -> (u64, u64) {
        let tvalue = self.tvalue << shift;
        let tmask = self.tmask << shift;
        (tvalue, tmask)
    }

    #[inline(always)]
    fn _tnum_rshift(&self, shift: u32) -> (u64, u64) {
        // arithmetic right shift (sign-extending)
        let tvalue = ((self.tvalue as i64) >> shift) as u64;
        let tmask = ((self.tmask as i64) >> shift) as u64;
        (tvalue, tmask)
    }

    #[inline(always)]
    fn _tnum_urshift(&self, shift: u32) -> (u64, u64) {
        let tvalue = self.tvalue >> shift;
        let tmask = self.tmask >> shift;
        (tvalue, tmask)
    }

    #[inline(always)]
    fn _tnum_intersect(&self, other_tvalue: u64, other_tmask: u64) -> (u64, u64, bool) {
        let union_val = self.tvalue | other_tvalue;
        let either_known = self.tmask & other_tmask;
        let both_known = self.tmask | other_tmask;
        let unmasked_self = unmask_zero(self.tvalue, both_known);
        let unmasked_other = unmask_zero(other_tvalue, both_known);
        let tvalue = unmask_zero(union_val, either_known);
        let valid = unmasked_self == unmasked_other;
        (tvalue, either_known, valid)
    }

    #[inline(always)]
    fn _tnum_and_backwards(&self, result: &IntBound) -> (u64, u64, bool) {
        let tvalue = result.tvalue;
        let tmask = ((!self.tvalue) | result.tmask) & !tvalue;
        let inconsistent = result.tvalue & !self.tmask & !self.tvalue;
        (tvalue, tmask, inconsistent == 0)
    }

    #[inline(always)]
    fn _tnum_or_backwards(&self, result: &IntBound) -> (u64, u64, bool) {
        let zeros = !result.tmask & !result.tvalue;
        let tvalue = result.tvalue & !self.tvalue & !self.tmask;
        let tmask = !(zeros | tvalue);
        let inconsistent = self.tvalue & zeros;
        (tvalue, tmask, inconsistent == 0)
    }

    // ── Shrinking ──

    /// Shrink the bounds and the knownbits to be more precise, without
    /// changing the set of integers that is represented by self.
    fn shrink(&mut self) {
        let changed = self._shrink_bounds_by_knownbits();
        let changed2 = self._shrink_knownbits_by_bounds();
        if !changed && !changed2 {
            return;
        }
        // One more pass (should be idempotent after two passes)
        let changed_again = self._shrink_bounds_by_knownbits();
        let changed_again2 = self._shrink_knownbits_by_bounds();
        debug_assert!(
            !changed_again && !changed_again2,
            "shrinking was not idempotent after two passes"
        );
    }

    fn _shrink_bounds_by_knownbits(&mut self) -> bool {
        let min_by_knownbits = self._get_minimum_signed_by_knownbits_atleast(self.lower);
        let max_by_knownbits = self._get_maximum_signed_by_knownbits_atmost(self.upper);
        let changed = self.lower < min_by_knownbits || self.upper > max_by_knownbits;
        if changed {
            self.lower = min_by_knownbits;
            self.upper = max_by_knownbits;
        }
        changed
    }

    fn _shrink_knownbits_by_bounds(&mut self) -> bool {
        let (tvalue, tmask, _valid) = self._tnum_improve_knownbits_by_bounds();
        let changed = self.tvalue != tvalue || self.tmask != tmask;
        if changed {
            self.tmask = tmask;
            self.tvalue = tvalue;
        }
        changed
    }

    fn _tnum_implied_by_bounds(&self) -> (u64, u64) {
        let hbm_bounds = leading_zeros_mask((self.lower as u64) ^ (self.upper as u64));
        let bounds_common = (self.lower as u64) & hbm_bounds;
        let tmask = !hbm_bounds;
        (unmask_zero(bounds_common, tmask), tmask)
    }

    fn _tnum_improve_knownbits_by_bounds(&self) -> (u64, u64, bool) {
        let (tvalue, tmask) = self._tnum_implied_by_bounds();
        self._tnum_intersect(tvalue, tmask)
    }

    fn _get_minimum_signed_by_knownbits(&self) -> i64 {
        (self.tvalue | msbonly(self.tmask)) as i64
    }

    fn _get_maximum_signed_by_knownbits(&self) -> i64 {
        let unsigned_mask = self.tmask & !msbonly(self.tmask);
        (self.tvalue | unsigned_mask) as i64
    }

    fn _get_minimum_signed_by_knownbits_atleast(&self, threshold: i64) -> i64 {
        let max_kb = self._get_maximum_signed_by_knownbits();
        if max_kb < threshold {
            // No valid value can be >= threshold, but we don't want to
            // panic in the shrink path. Return threshold or lower+1 as fallback.
            return threshold;
        }
        let min_kb = self._get_minimum_signed_by_knownbits();
        if min_kb >= threshold {
            return min_kb;
        }

        let u_min_threshold = threshold as u64;
        let (working_min, cl2set, set2cl) = self._helper_min_max_prepare(u_min_threshold);
        if working_min == u_min_threshold {
            threshold
        } else if cl2set > set2cl {
            self._helper_min_case1(working_min, cl2set)
        } else {
            self._helper_min_case2(working_min, set2cl)
        }
    }

    fn _get_maximum_signed_by_knownbits_atmost(&self, threshold: i64) -> i64 {
        let min_kb = self._get_minimum_signed_by_knownbits();
        if min_kb > threshold {
            return threshold;
        }
        let max_kb = self._get_maximum_signed_by_knownbits();
        if max_kb <= threshold {
            return max_kb;
        }

        let u_max_threshold = threshold as u64;
        let (working_max, cl2set, set2cl) = self._helper_min_max_prepare(u_max_threshold);
        if working_max == u_max_threshold {
            threshold
        } else if cl2set < set2cl {
            self._helper_max_case1(working_max, set2cl)
        } else {
            self._helper_max_case2(working_max, cl2set)
        }
    }

    #[inline(always)]
    fn _helper_min_max_prepare(&self, u_threshold: u64) -> (u64, u64, u64) {
        let mut working_value = u_threshold;
        working_value &= unmask_one(self.tvalue, self.tmask); // clear known 0s
        working_value |= self.tvalue; // set known 1s
        let cl2set = !u_threshold & working_value;
        let set2cl = u_threshold & !working_value;
        (working_value, cl2set, set2cl)
    }

    #[inline(always)]
    fn _helper_min_case1(&self, working_min: u64, cl2set: u64) -> i64 {
        let clear_mask = leading_zeros_mask(cl2set >> 1);
        let working_min = working_min & (clear_mask | !self.tmask);
        working_min as i64
    }

    #[inline(always)]
    fn _helper_min_case2(&self, working_min: u64, set2cl: u64) -> i64 {
        let mut working_min = flip_msb(working_min);
        let possible_bits = !working_min & self.tmask & leading_zeros_mask(set2cl);
        let bit_to_set = lowest_set_bit_only(possible_bits);
        working_min |= bit_to_set;
        let clear_mask = leading_zeros_mask(bit_to_set) | bit_to_set | !self.tmask;
        working_min &= clear_mask;
        flip_msb(working_min) as i64
    }

    #[inline(always)]
    fn _helper_max_case1(&self, working_max: u64, set2cl: u64) -> i64 {
        let set_mask = next_pow2_m1(set2cl >> 1) & self.tmask;
        let working_max = working_max | set_mask;
        working_max as i64
    }

    #[inline(always)]
    fn _helper_max_case2(&self, working_max: u64, cl2set: u64) -> i64 {
        let mut working_max = flip_msb(working_max);
        let possible_bits = working_max & self.tmask & leading_zeros_mask(cl2set);
        let bit_to_clear = lowest_set_bit_only(possible_bits);
        working_max &= !bit_to_clear;
        let set_mask = next_pow2_m1(bit_to_clear >> 1) & self.tmask;
        working_max |= set_mask;
        flip_msb(working_max) as i64
    }

    // ── Debug/Display helpers ──

    /// Return a string representation of the knownbits.
    pub fn knownbits_string(&self) -> String {
        let mut results = Vec::with_capacity(64);
        for bit in 0..64u32 {
            if self.tmask & (1u64 << bit) != 0 {
                results.push('?');
            } else {
                results.push(if (self.tvalue >> bit) & 1 != 0 {
                    '1'
                } else {
                    '0'
                });
            }
        }
        results.reverse();
        results.into_iter().collect()
    }

    fn _debug_check(&self) -> bool {
        if self.lower > self.upper {
            return false;
        }
        let min_kb = self._get_minimum_signed_by_knownbits();
        let max_kb = self._get_maximum_signed_by_knownbits();
        if min_kb > self.upper {
            return false;
        }
        if max_kb < self.lower {
            return false;
        }
        if min_kb > max_kb {
            return false;
        }
        true
    }
}

/// Checked left shift for i64 that returns None on overflow.
fn checked_shl_i64(value: i64, shift: i64) -> Option<i64> {
    if !(0..64).contains(&shift) {
        return None;
    }
    let shift = shift as u32;
    // Check if shifting would overflow
    let result = value.checked_shl(shift)?;
    // Verify round-trip
    if (result >> shift) != value {
        return None;
    }
    Some(result)
}

impl Default for IntBound {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl std::fmt::Display for IntBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_constant() {
            write!(f, "({})", self.get_constant_int())
        } else if self.lower == i64::MIN && self.upper == i64::MAX {
            write!(f, "(?)")
        } else {
            let lower_s = if self.lower == i64::MIN {
                String::new()
            } else {
                format!("{} <= ", self.lower)
            };
            let upper_s = if self.upper == i64::MAX {
                String::new()
            } else {
                format!(" <= {}", self.upper)
            };
            write!(f, "({}?{})", lower_s, upper_s)
        }
    }
}

/// Shared-identity `IntBound` handle. `getintbound` / `setintbound`
/// mint through [`IntBoundRc::new`]. A one-word refcount (no `Weak`).
/// The count is a `Cell`, so a handle stays on the allocating thread
/// and each value is an ordinary allocation. Clone is a count bump;
/// the last drop frees the allocation.
pub struct IntBoundRc {
    ptr: std::ptr::NonNull<IntBoundInner>,
}

#[repr(C)]
struct IntBoundInner {
    strong: std::cell::Cell<usize>,
    value: std::cell::RefCell<IntBound>,
}

fn alloc_int_bound_inner() -> std::ptr::NonNull<IntBoundInner> {
    let layout = std::alloc::Layout::new::<IntBoundInner>();
    let base = unsafe { std::alloc::alloc(layout) }.cast::<IntBoundInner>();
    assert!(!base.is_null(), "IntBoundInner alloc failed");
    unsafe { std::ptr::NonNull::new_unchecked(base) }
}

fn free_int_bound_inner(p: std::ptr::NonNull<IntBoundInner>) {
    let layout = std::alloc::Layout::new::<IntBoundInner>();
    unsafe { std::alloc::dealloc(p.as_ptr().cast(), layout) };
}

impl IntBoundRc {
    pub fn new(bound: IntBound) -> Self {
        let ptr = alloc_int_bound_inner();
        unsafe {
            ptr.as_ptr().write(IntBoundInner {
                strong: std::cell::Cell::new(1),
                value: std::cell::RefCell::new(bound),
            });
        }
        IntBoundRc { ptr }
    }

    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }

    pub fn as_ptr(this: &Self) -> *const () {
        this.ptr.as_ptr() as *const ()
    }

    pub fn into_raw(this: Self) -> *const () {
        let p = Self::as_ptr(&this);
        std::mem::forget(this);
        p
    }

    /// # Safety
    /// `inner` must be a pointer previously returned by [`Self::into_raw`].
    pub unsafe fn from_raw(inner: *const ()) -> Self {
        IntBoundRc {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(inner.cast_mut().cast()) },
        }
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, IntBound> {
        unsafe { self.ptr.as_ref().value.borrow() }
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, IntBound> {
        unsafe { self.ptr.as_ref().value.borrow_mut() }
    }
}

impl Clone for IntBoundRc {
    fn clone(&self) -> Self {
        let inner = unsafe { self.ptr.as_ref() };
        inner.strong.set(inner.strong.get() + 1);
        IntBoundRc { ptr: self.ptr }
    }
}

impl Drop for IntBoundRc {
    fn drop(&mut self) {
        let inner = unsafe { self.ptr.as_ref() };
        let n = inner.strong.get() - 1;
        if n == 0 {
            unsafe {
                std::ptr::drop_in_place(&mut (*self.ptr.as_ptr()).value);
            }
            free_int_bound_inner(self.ptr);
        } else {
            inner.strong.set(n);
        }
    }
}

impl std::fmt::Debug for IntBoundRc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IntBoundRc").field(&*self.borrow()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction tests ──

    #[test]
    fn test_constant() {
        let b = IntBound::from_constant(42);
        assert!(b.is_constant());
        assert_eq!(b.get_constant_int(), 42);
        assert_eq!(b.lower, 42);
        assert_eq!(b.upper, 42);
        assert_eq!(b.tvalue, 42);
        assert_eq!(b.tmask, 0);
    }

    #[test]
    fn test_constant_negative() {
        let b = IntBound::from_constant(-1);
        assert!(b.is_constant());
        assert_eq!(b.get_constant_int(), -1);
        assert_eq!(b.tvalue, u64::MAX);
        assert_eq!(b.tmask, 0);
    }

    #[test]
    fn test_constant_zero() {
        let b = IntBound::from_constant(0);
        assert!(b.is_constant());
        assert_eq!(b.get_constant_int(), 0);
        assert_eq!(b.tvalue, 0);
        assert_eq!(b.tmask, 0);
    }

    #[test]
    fn test_unbounded() {
        let b = IntBound::unbounded();
        assert!(!b.is_constant());
        assert_eq!(b.lower, i64::MIN);
        assert_eq!(b.upper, i64::MAX);
        assert!(b.is_unbounded());
    }

    #[test]
    fn test_bounded() {
        let b = IntBound::bounded(0, 100);
        assert!(b.known_nonnegative());
        assert!(!b.is_constant());
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, 100);
    }

    #[test]
    fn test_bounded_exact() {
        let b = IntBound::bounded(42, 42);
        assert!(b.is_constant());
        assert_eq!(b.get_constant_int(), 42);
    }

    #[test]
    fn test_nonnegative() {
        let b = IntBound::nonnegative();
        assert!(b.known_nonnegative());
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, i64::MAX);
    }

    #[test]
    fn test_from_knownbits() {
        // All bits known
        let b = IntBound::from_knownbits(0xFF, 0);
        assert!(b.is_constant());
        assert_eq!(b.get_constant_int(), 0xFF);

        // Low byte known to be 0xFF, rest unknown
        let b = IntBound::from_knownbits(0xFF, !0xFFu64);
        assert!(!b.is_constant());
        // tvalue should have low byte 0xFF, rest cleared
        assert_eq!(b.tvalue & 0xFF, 0xFF);
    }

    // ── Query tests ──

    #[test]
    fn test_contains() {
        let b = IntBound::bounded(0, 100);
        assert!(b.contains(0));
        assert!(b.contains(50));
        assert!(b.contains(100));
        assert!(!b.contains(-1));
        assert!(!b.contains(101));

        let b = IntBound::from_constant(42);
        assert!(b.contains(42));
        assert!(!b.contains(43));
    }

    #[test]
    fn test_contains_knownbits() {
        // Even numbers only (bit 0 known to be 0)
        let b = IntBound::from_knownbits(0, 0xFFFFFFFFFFFFFFFE);
        assert!(b.contains(0));
        assert!(b.contains(2));
        assert!(!b.contains(1));
        assert!(!b.contains(3));
    }

    #[test]
    fn test_is_bool() {
        let b = IntBound::bounded(0, 1);
        assert!(b.is_bool());

        let b = IntBound::from_constant(0);
        assert!(b.is_bool());

        let b = IntBound::from_constant(1);
        assert!(b.is_bool());

        let b = IntBound::bounded(0, 2);
        assert!(!b.is_bool());

        let b = IntBound::bounded(-1, 1);
        assert!(!b.is_bool());
    }

    // ── Comparison tests ──

    #[test]
    fn test_known_comparisons() {
        let a = IntBound::bounded(0, 5);
        let b = IntBound::bounded(10, 20);
        assert!(a.known_lt(&b));
        assert!(a.known_le(&b));
        assert!(!a.known_gt(&b));
        assert!(b.known_gt(&a));
        assert!(b.known_ge(&a));
    }

    #[test]
    fn test_known_lt_const() {
        let b = IntBound::bounded(0, 5);
        assert!(b.known_lt_const(6));
        assert!(!b.known_lt_const(5));
        assert!(!b.known_lt_const(3));
    }

    #[test]
    fn test_known_le_const() {
        let b = IntBound::bounded(0, 5);
        assert!(b.known_le_const(5));
        assert!(b.known_le_const(6));
        assert!(!b.known_le_const(4));
    }

    #[test]
    fn test_known_gt_const() {
        let b = IntBound::bounded(10, 20);
        assert!(b.known_gt_const(9));
        assert!(!b.known_gt_const(10));
    }

    #[test]
    fn test_known_ge_const() {
        let b = IntBound::bounded(10, 20);
        assert!(b.known_ge_const(10));
        assert!(!b.known_ge_const(11));
    }

    #[test]
    fn test_known_eq_const() {
        let b = IntBound::from_constant(42);
        assert!(b.known_eq_const(42));
        assert!(!b.known_eq_const(43));

        let b = IntBound::bounded(40, 44);
        assert!(!b.known_eq_const(42));
    }

    #[test]
    fn test_known_ne() {
        let a = IntBound::bounded(0, 5);
        let b = IntBound::bounded(10, 20);
        assert!(a.known_ne(&b));

        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(5, 20);
        assert!(!a.known_ne(&b));
    }

    // ── Unsigned comparison tests ──

    #[test]
    fn test_known_unsigned_lt() {
        let a = IntBound::bounded(0, 5);
        let b = IntBound::bounded(10, 20);
        assert!(a.known_unsigned_lt(&b));
        assert!(!b.known_unsigned_lt(&a));
    }

    #[test]
    fn test_known_unsigned_le() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(10, 20);
        assert!(a.known_unsigned_le(&b));
    }

    // ── Mutation tests ──

    #[test]
    fn test_make_le() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(0, 50);
        let changed = a.make_le(&b).unwrap();
        assert!(changed);
        assert_eq!(a.upper, 50);
    }

    #[test]
    fn test_make_lt() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(0, 50);
        let changed = a.make_lt(&b).unwrap();
        assert!(changed);
        assert_eq!(a.upper, 49);
    }

    #[test]
    fn test_make_ge() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(50, 100);
        let changed = a.make_ge(&b).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 50);
    }

    #[test]
    fn test_make_gt() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(50, 100);
        let changed = a.make_gt(&b).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 51);
    }

    #[test]
    fn test_make_le_invalid() {
        let mut a = IntBound::bounded(50, 100);
        let result = a.make_le_const(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_ge_invalid() {
        let mut a = IntBound::bounded(0, 50);
        let result = a.make_ge_const(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_eq_const() {
        let mut a = IntBound::bounded(0, 100);
        a.make_eq_const(42).unwrap();
        assert!(a.is_constant());
        assert_eq!(a.get_constant_int(), 42);
    }

    #[test]
    fn test_make_eq_const_invalid() {
        let mut a = IntBound::bounded(0, 10);
        assert!(a.make_eq_const(42).is_err());
    }

    #[test]
    fn test_make_ne_const() {
        let mut a = IntBound::bounded(0, 10);
        // Removing the upper bound
        assert!(a.make_ne_const(10));
        assert_eq!(a.upper, 9);

        // Removing the lower bound
        let mut a = IntBound::bounded(0, 10);
        assert!(a.make_ne_const(0));
        assert_eq!(a.lower, 1);

        // Removing a middle value: no change
        let mut a = IntBound::bounded(0, 10);
        assert!(!a.make_ne_const(5));
    }

    #[test]
    fn test_make_bool() {
        let mut a = IntBound::bounded(0, 100);
        a.make_bool();
        assert_eq!(a.lower, 0);
        assert_eq!(a.upper, 1);
    }

    // ── Intersection tests ──

    #[test]
    fn test_intersect() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(50, 200);
        let changed = a.intersect(&b).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 50);
        assert_eq!(a.upper, 100);
    }

    #[test]
    fn test_intersect_no_overlap() {
        let mut a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(20, 30);
        assert!(a.intersect(&b).is_err());
    }

    #[test]
    fn test_intersect_const() {
        let mut a = IntBound::bounded(0, 100);
        let changed = a.intersect_const(20, 80).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 20);
        assert_eq!(a.upper, 80);
    }

    #[test]
    fn test_intersect_knownbits() {
        // a knows low bit is 0 (even)
        let mut a = IntBound::from_knownbits(0, 0xFFFFFFFFFFFFFFFE);
        // b knows low bit is 0 too, and second bit is 1
        let b = IntBound::from_knownbits(0b10, 0xFFFFFFFFFFFFFFFC);
        let changed = a.intersect(&b).unwrap();
        assert!(changed);
        // Now we should know both low bits: bit0=0, bit1=1
        assert_eq!(a.tvalue & 3, 0b10);
        assert_eq!(a.tmask & 3, 0);
    }

    #[test]
    fn test_intersect_knownbits_contradiction() {
        // a knows low bit is 0
        let mut a = IntBound::from_knownbits(0, 0xFFFFFFFFFFFFFFFE);
        // b knows low bit is 1
        let b = IntBound::from_knownbits(1, 0xFFFFFFFFFFFFFFFE);
        assert!(a.intersect(&b).is_err());
    }

    // ── Addition tests ──

    #[test]
    fn test_add_constants() {
        let a = IntBound::from_constant(10);
        let b = IntBound::from_constant(20);
        let result = a.add_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 30);
    }

    #[test]
    fn test_add_bounded() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(0, 20);
        let result = a.add_bound(&b);
        assert_eq!(result.lower, 0);
        assert_eq!(result.upper, 30);
    }

    #[test]
    fn test_add_negative() {
        let a = IntBound::from_constant(-5);
        let b = IntBound::from_constant(3);
        let result = a.add_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -2);
    }

    #[test]
    fn test_add_overflow() {
        let a = IntBound::from_constant(i64::MAX);
        let b = IntBound::from_constant(1);
        let result = a.add_bound(&b);
        // lower overflow: MAX + 1 overflows, so bounds come from knownbits only.
        // The tnum addition gives us the wrapped value (i64::MIN), which is the
        // correct modular result. The knownbits are fully known (both constants).
        // So the result is from_knownbits which gives us i64::MIN as constant.
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), i64::MIN);
    }

    #[test]
    fn test_add_bound_cannot_overflow() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(0, 10);
        assert!(a.add_bound_cannot_overflow(&b));

        let a = IntBound::bounded(0, i64::MAX);
        let b = IntBound::bounded(1, i64::MAX);
        assert!(!a.add_bound_cannot_overflow(&b));
    }

    #[test]
    fn test_add_helper() {
        let a = IntBound::bounded(0, 100);
        let result = a.add(5);
        assert_eq!(result.lower, 5);
        assert_eq!(result.upper, 105);
    }

    // ── Subtraction tests ──

    #[test]
    fn test_sub_bounded() {
        let a = IntBound::bounded(10, 20);
        let b = IntBound::bounded(0, 5);
        let result = a.sub_bound(&b);
        assert_eq!(result.lower, 5);
        assert_eq!(result.upper, 20);
    }

    #[test]
    fn test_sub_constants() {
        let a = IntBound::from_constant(30);
        let b = IntBound::from_constant(10);
        let result = a.sub_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 20);
    }

    #[test]
    fn test_sub_overflow() {
        let a = IntBound::from_constant(i64::MIN);
        let b = IntBound::from_constant(1);
        let result = a.sub_bound(&b);
        // RPython's test_sub_bound_bug checks the machine-integer wraparound.
        assert!(result.contains(i64::MAX));
    }

    #[test]
    fn test_sub_bound_cannot_overflow() {
        let a = IntBound::bounded(10, 20);
        let b = IntBound::bounded(0, 5);
        assert!(a.sub_bound_cannot_overflow(&b));

        let a = IntBound::bounded(i64::MIN, 0);
        let b = IntBound::bounded(1, i64::MAX);
        assert!(!a.sub_bound_cannot_overflow(&b));
    }

    // ── Multiplication tests ──

    #[test]
    fn test_mul_constants() {
        let a = IntBound::from_constant(6);
        let b = IntBound::from_constant(7);
        let result = a.mul_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 42);
    }

    #[test]
    fn test_mul_bounded() {
        let a = IntBound::bounded(2, 3);
        let b = IntBound::bounded(4, 5);
        let result = a.mul_bound(&b);
        assert_eq!(result.lower, 8);
        assert_eq!(result.upper, 15);
    }

    #[test]
    fn test_mul_negative() {
        let a = IntBound::bounded(-3, -2);
        let b = IntBound::bounded(4, 5);
        let result = a.mul_bound(&b);
        assert_eq!(result.lower, -15);
        assert_eq!(result.upper, -8);
    }

    #[test]
    fn test_mul_cross_zero() {
        let a = IntBound::bounded(-3, 3);
        let b = IntBound::bounded(-5, 5);
        let result = a.mul_bound(&b);
        assert_eq!(result.lower, -15);
        assert_eq!(result.upper, 15);
    }

    #[test]
    fn test_mul_overflow() {
        let a = IntBound::bounded(0, i64::MAX);
        let b = IntBound::bounded(2, 3);
        let result = a.mul_bound(&b);
        assert!(result.is_unbounded());
    }

    #[test]
    fn test_mul_bound_cannot_overflow() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(0, 10);
        assert!(a.mul_bound_cannot_overflow(&b));

        let a = IntBound::bounded(0, i64::MAX);
        let b = IntBound::bounded(2, 3);
        assert!(!a.mul_bound_cannot_overflow(&b));
    }

    #[test]
    fn test_square_bound_no_overflow() {
        let a = IntBound::bounded(2, 5);
        let result = a.square_bound_no_overflow();
        assert_eq!(result.lower, 4);
        assert_eq!(result.upper, 25);

        // Cross zero: 0 is in the range
        let a = IntBound::bounded(-3, 5);
        let result = a.square_bound_no_overflow();
        assert_eq!(result.lower, 0);
        assert_eq!(result.upper, 25);
    }

    // ── Division tests ──

    #[test]
    fn test_py_div_constants() {
        let a = IntBound::from_constant(7);
        let b = IntBound::from_constant(2);
        let result = a.py_div_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 3);
    }

    #[test]
    fn test_py_div_negative() {
        let a = IntBound::from_constant(-7);
        let b = IntBound::from_constant(2);
        let result = a.py_div_bound(&b);
        // Python: -7 // 2 == -4
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -4);
    }

    #[test]
    fn test_py_div_by_zero_interval() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(-5, 5); // contains 0
        let result = a.py_div_bound(&b);
        assert!(result.is_unbounded());
    }

    #[test]
    fn test_py_div_bounded() {
        let a = IntBound::bounded(10, 20);
        let b = IntBound::bounded(2, 5);
        let result = a.py_div_bound(&b);
        assert_eq!(result.lower, 2);
        assert_eq!(result.upper, 10);
    }

    // ── Mod tests ──

    #[test]
    fn test_mod_positive_divisor() {
        let a = IntBound::unbounded();
        let b = IntBound::bounded(1, 10);
        let result = a.mod_bound(&b);
        assert_eq!(result.lower, 0);
        assert_eq!(result.upper, 9);
    }

    #[test]
    fn test_mod_negative_divisor() {
        let a = IntBound::unbounded();
        let b = IntBound::bounded(-10, -1);
        let result = a.mod_bound(&b);
        assert_eq!(result.lower, -9);
        assert_eq!(result.upper, 0);
    }

    #[test]
    fn test_mod_zero_divisor() {
        let a = IntBound::unbounded();
        let b = IntBound::from_constant(0);
        let result = a.mod_bound(&b);
        assert!(result.is_unbounded());
    }

    #[test]
    fn test_mod_both_signs() {
        let a = IntBound::unbounded();
        let b = IntBound::bounded(-5, 10);
        let result = a.mod_bound(&b);
        assert_eq!(result.lower, -4);
        assert_eq!(result.upper, 9);
    }

    // ── Bitwise AND tests ──

    #[test]
    fn test_and_constants() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::from_constant(0x0F);
        let result = a.and_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0x0F);
    }

    #[test]
    fn test_and_nonneg_upper() {
        let a = IntBound::bounded(0, 255);
        let b = IntBound::bounded(0, 15);
        let result = a.and_bound(&b);
        assert!(result.known_nonnegative());
        assert!(result.upper <= 15);
    }

    #[test]
    fn test_and_knownbits() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::unbounded();
        let result = a.and_bound(&b);
        // Low byte may be anything but high bytes are 0
        // a is constant 0xFF, so AND preserves only low 8 bits at most
        // The tnum result should reflect this
        assert_eq!(result.tmask & !0xFFu64, 0); // high bits must be known 0
    }

    // ── Bitwise OR tests ──

    #[test]
    fn test_or_constants() {
        let a = IntBound::from_constant(0xF0);
        let b = IntBound::from_constant(0x0F);
        let result = a.or_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0xFF);
    }

    #[test]
    fn test_or_knownbits() {
        // a = 0b????0000, b = 0b00001111
        let a = IntBound::from_knownbits(0, 0xF0u64 | !0xFFu64);
        let b = IntBound::from_constant(0x0F);
        let result = a.or_bound(&b);
        // Low nibble should be all 1s (known from b)
        assert_eq!(result.tvalue & 0x0F, 0x0F);
    }

    // ── Bitwise XOR tests ──

    #[test]
    fn test_xor_constants() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::from_constant(0x0F);
        let result = a.xor_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0xF0);
    }

    #[test]
    fn test_xor_same_value() {
        let a = IntBound::from_constant(42);
        let result = a.xor_bound(&a);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0);
    }

    // ── Negation and inversion tests ──

    #[test]
    fn test_neg_bound() {
        let a = IntBound::from_constant(42);
        let result = a.neg_bound();
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -42);
    }

    #[test]
    fn test_neg_bound_range() {
        let a = IntBound::bounded(10, 20);
        let result = a.neg_bound();
        assert_eq!(result.lower, -20);
        assert_eq!(result.upper, -10);
    }

    #[test]
    fn test_invert_bound() {
        let a = IntBound::from_constant(0);
        let result = a.invert_bound();
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -1);
    }

    #[test]
    fn test_invert_bound_range() {
        let a = IntBound::bounded(10, 20);
        let result = a.invert_bound();
        assert_eq!(result.lower, -21);
        assert_eq!(result.upper, -11);
    }

    // ── Left shift tests ──

    #[test]
    fn test_lshift_constant() {
        let a = IntBound::from_constant(1);
        let shift = IntBound::from_constant(3);
        let result = a.lshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 8);
    }

    #[test]
    fn test_lshift_bounded() {
        let a = IntBound::bounded(1, 3);
        let shift = IntBound::from_constant(2);
        let result = a.lshift_bound(&shift);
        assert_eq!(result.lower, 4);
        assert_eq!(result.upper, 12);
    }

    #[test]
    fn test_lshift_large_shift() {
        let a = IntBound::from_constant(1);
        let shift = IntBound::from_constant(64);
        let result = a.lshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0);
    }

    #[test]
    fn test_lshift_knownbits() {
        let a = IntBound::from_constant(0b101);
        let shift = IntBound::from_constant(2);
        let result = a.lshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0b10100);
    }

    // ── Right shift tests ──

    #[test]
    fn test_rshift_constant() {
        let a = IntBound::from_constant(16);
        let shift = IntBound::from_constant(2);
        let result = a.rshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 4);
    }

    #[test]
    fn test_rshift_negative() {
        let a = IntBound::from_constant(-16);
        let shift = IntBound::from_constant(2);
        let result = a.rshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -4);
    }

    #[test]
    fn test_rshift_bounded() {
        let a = IntBound::bounded(16, 32);
        let shift = IntBound::from_constant(2);
        let result = a.rshift_bound(&shift);
        assert_eq!(result.lower, 4);
        assert_eq!(result.upper, 8);
    }

    #[test]
    fn test_rshift_large_shift_positive() {
        let a = IntBound::from_constant(100);
        let shift = IntBound::from_constant(64);
        let result = a.rshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0);
    }

    #[test]
    fn test_rshift_large_shift_negative() {
        let a = IntBound::from_constant(-100);
        let shift = IntBound::from_constant(64);
        let result = a.rshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -1);
    }

    // ── Unsigned right shift tests ──

    #[test]
    fn test_urshift_constant() {
        let a = IntBound::from_constant(16);
        let shift = IntBound::from_constant(2);
        let result = a.urshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 4);
    }

    #[test]
    fn test_urshift_negative() {
        let a = IntBound::from_constant(-1);
        let shift = IntBound::from_constant(1);
        let result = a.urshift_bound(&shift);
        // -1 as u64 is 0xFFFF...FFFF, >> 1 = 0x7FFF...FFFF = i64::MAX
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), i64::MAX);
    }

    #[test]
    fn test_urshift_large_shift() {
        let a = IntBound::from_constant(-1);
        let shift = IntBound::from_constant(64);
        let result = a.urshift_bound(&shift);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0);
    }

    // ── Backwards transfer function tests ──

    #[test]
    fn test_and_bound_backwards() {
        // self = constant 0xFF, result = constant 0x0F
        // => other must have bit pattern 0x0F in the known-1 positions of self
        let self_bound = IntBound::from_constant(0xFF);
        let result_bound = IntBound::from_constant(0x0F);
        let other = self_bound.and_bound_backwards(&result_bound).unwrap();
        // Low nibble of other should be 0x0F, upper nibble should be 0
        assert_eq!(other.tvalue & 0xFF, 0x0F);
    }

    #[test]
    fn test_or_bound_backwards() {
        // self = constant 0xF0, result = constant 0xFF
        // => other must have 0x0F set (since self didn't provide those bits)
        let self_bound = IntBound::from_constant(0xF0);
        let result_bound = IntBound::from_constant(0xFF);
        let other = self_bound.or_bound_backwards(&result_bound).unwrap();
        assert_eq!(other.tvalue & 0x0F, 0x0F);
    }

    #[test]
    fn test_rshift_bound_backwards() {
        let a = IntBound::from_constant(0x0F);
        let shift = IntBound::from_constant(4);
        let result = a.rshift_bound_backwards(&shift);
        // Should have 0x0F << 4 = 0xF0 as the known bits, with low 4 bits unknown
        assert_eq!(result.tvalue & 0xF0, 0xF0);
        assert_eq!(result.tmask & 0x0F, 0x0F); // low 4 bits unknown
    }

    #[test]
    fn test_lshift_bound_backwards() {
        let a = IntBound::from_constant(0xF0);
        let shift = IntBound::from_constant(4);
        let result = a.lshift_bound_backwards(&shift).unwrap();
        // Should have 0xF0 >> 4 = 0x0F as the known bits, with high bits unknown
        assert_eq!(result.tvalue & 0x0F, 0x0F);
    }

    // ── Tnum operation tests ──

    #[test]
    fn test_tnum_add() {
        let a = IntBound::from_constant(3);
        let b = IntBound::from_constant(5);
        let (tv, tm) = a._tnum_add(&b);
        assert_eq!(tv, 8);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_sub() {
        let a = IntBound::from_constant(10);
        let b = IntBound::from_constant(3);
        let (tv, tm) = a._tnum_sub(&b);
        assert_eq!(tv, 7);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_and() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::from_constant(0x0F);
        let (tv, tm) = a._tnum_and(&b);
        assert_eq!(tv, 0x0F);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_or() {
        let a = IntBound::from_constant(0xF0);
        let b = IntBound::from_constant(0x0F);
        let (tv, tm) = a._tnum_or(&b);
        assert_eq!(tv, 0xFF);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_xor() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::from_constant(0x0F);
        let (tv, tm) = a._tnum_xor(&b);
        assert_eq!(tv, 0xF0);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_lshift() {
        let a = IntBound::from_constant(1);
        let (tv, tm) = a._tnum_lshift(3);
        assert_eq!(tv, 8);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_rshift() {
        let a = IntBound::from_constant(16);
        let (tv, tm) = a._tnum_rshift(2);
        assert_eq!(tv, 4);
        assert_eq!(tm, 0);
    }

    #[test]
    fn test_tnum_urshift() {
        let a = IntBound::from_constant(16);
        let (tv, tm) = a._tnum_urshift(2);
        assert_eq!(tv, 4);
        assert_eq!(tm, 0);
    }

    // ── Widen tests ──

    #[test]
    fn test_widen() {
        let a = IntBound::bounded(0, 100);
        let w = a.widen();
        assert_eq!(w.lower, 0);
        assert_eq!(w.upper, 100);

        let a = IntBound::bounded(i64::MIN / 2 - 1, i64::MAX / 2 + 1);
        let w = a.widen();
        assert_eq!(w.lower, i64::MIN);
        assert_eq!(w.upper, i64::MAX);
    }

    // ── Shrinking tests ──

    #[test]
    fn test_shrink_learns_bits_from_bounds() {
        // [0, 7] should learn that bits 63..3 are 0
        let b = IntBound::bounded(0, 7);
        // High bits should be known to be 0
        assert_eq!(b.tvalue >> 3, 0);
        assert_eq!(b.tmask >> 3, 0);
    }

    #[test]
    fn test_shrink_constant_detection() {
        // If lower == upper, all bits become known
        let b = IntBound::bounded(42, 42);
        assert_eq!(b.tmask, 0);
        assert_eq!(b.tvalue, 42);
    }

    #[test]
    fn test_shrink_knownbits_tighten_bounds() {
        // If we know the sign bit is 0, lower should be >= 0
        let b = IntBound::from_knownbits(0, u64::MAX >> 1); // sign bit = 0
        assert!(b.lower >= 0);
    }

    // ── Display test ──

    #[test]
    fn test_display() {
        let b = IntBound::from_constant(42);
        assert_eq!(format!("{}", b), "(42)");

        let b = IntBound::unbounded();
        assert_eq!(format!("{}", b), "(?)");
    }

    // ── Knownbits string test ──

    #[test]
    fn test_knownbits_string() {
        let b = IntBound::from_constant(5); // 0b101
        let s = b.knownbits_string();
        assert_eq!(s.len(), 64);
        assert!(s.ends_with("101"));
        // All characters should be 0 or 1 (no ?)
        assert!(!s.contains('?'));
    }

    #[test]
    fn test_knownbits_string_unknown() {
        let b = IntBound::unbounded();
        let s = b.knownbits_string();
        assert_eq!(s.len(), 64);
        // All bits unknown
        for c in s.chars() {
            assert_eq!(c, '?');
        }
    }

    // ── is_within_range tests ──

    #[test]
    fn test_is_within_range() {
        let b = IntBound::bounded(10, 20);
        assert!(b.is_within_range(0, 100));
        assert!(b.is_within_range(10, 20));
        assert!(!b.is_within_range(11, 20));
        assert!(!b.is_within_range(10, 19));
    }

    // ── Python-style division helper tests ──

    #[test]
    fn test_py_div_helper() {
        assert_eq!(py_div(7, 2), 3);
        assert_eq!(py_div(-7, 2), -4);
        assert_eq!(py_div(7, -2), -4);
        assert_eq!(py_div(-7, -2), 3);
        assert_eq!(py_div(6, 3), 2);
        assert_eq!(py_div(-6, 3), -2);
    }

    // ── Free function tests ──

    #[test]
    fn test_unmask_zero() {
        assert_eq!(unmask_zero(0xFF, 0x0F), 0xF0);
        assert_eq!(unmask_zero(0xFF, 0), 0xFF);
        assert_eq!(unmask_zero(0xFF, u64::MAX), 0);
    }

    #[test]
    fn test_unmask_one() {
        assert_eq!(unmask_one(0, 0xFF), 0xFF);
        assert_eq!(unmask_one(0, 0), 0);
    }

    #[test]
    fn test_next_pow2_m1() {
        assert_eq!(next_pow2_m1(0), 0);
        assert_eq!(next_pow2_m1(1), 1);
        assert_eq!(next_pow2_m1(2), 3);
        assert_eq!(next_pow2_m1(3), 3);
        assert_eq!(next_pow2_m1(4), 7);
        assert_eq!(next_pow2_m1(5), 7);
        assert_eq!(next_pow2_m1(8), 15);
    }

    #[test]
    fn test_leading_zeros_mask() {
        assert_eq!(leading_zeros_mask(0), u64::MAX);
        assert_eq!(leading_zeros_mask(1), !1u64);
        assert_eq!(leading_zeros_mask(u64::MAX), 0);
    }

    #[test]
    fn test_msbonly() {
        assert_eq!(msbonly(0), 0);
        assert_eq!(msbonly(1u64 << 63), 1u64 << 63);
        assert_eq!(msbonly(u64::MAX), 1u64 << 63);
        assert_eq!(msbonly(0x7FFFFFFFFFFFFFFF), 0);
    }

    #[test]
    fn test_flip_msb() {
        assert_eq!(flip_msb(0), 1u64 << 63);
        assert_eq!(flip_msb(1u64 << 63), 0);
    }

    #[test]
    fn test_lowest_set_bit_only() {
        assert_eq!(lowest_set_bit_only(0b1010), 0b0010);
        assert_eq!(lowest_set_bit_only(0b1000), 0b1000);
        assert_eq!(lowest_set_bit_only(0b0001), 0b0001);
    }

    #[test]
    fn test_is_valid_tnum() {
        assert!(is_valid_tnum(0, 0));
        assert!(is_valid_tnum(0xFF, 0));
        assert!(is_valid_tnum(0, u64::MAX));
        assert!(is_valid_tnum(0xF0, 0x0F));
        assert!(!is_valid_tnum(0xFF, 0xFF));
        assert!(!is_valid_tnum(1, 1));
    }

    #[test]
    fn test_saturating_mul() {
        assert_eq!(saturating_mul(3, 4), 12);
        assert_eq!(saturating_mul(i64::MAX, 2), i64::MAX);
        assert_eq!(saturating_mul(i64::MIN, 2), i64::MIN);
        assert_eq!(saturating_mul(i64::MAX, -2), i64::MIN);
    }

    #[test]
    fn test_same_sign() {
        assert!(same_sign(1, 2));
        assert!(same_sign(-1, -2));
        assert!(same_sign(0, 0));
        assert!(same_sign(0, 1));
        assert!(!same_sign(1, -1));
        assert!(!same_sign(-1, 1));
    }

    // ── Edge case / regression tests ──

    #[test]
    fn test_add_minint_maxint() {
        let a = IntBound::from_constant(i64::MIN);
        let b = IntBound::from_constant(i64::MAX);
        let result = a.add_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), -1);
    }

    #[test]
    fn test_sub_minint() {
        let a = IntBound::from_constant(0);
        let b = IntBound::from_constant(i64::MIN);
        let result = a.sub_bound(&b);
        // 0 - MININT overflows, so should not be constant
        // It returns from knownbits
        assert!(!result.is_constant() || result.lower == i64::MIN);
    }

    #[test]
    fn test_and_with_neg() {
        let a = IntBound::from_constant(-1);
        let b = IntBound::from_constant(0xFF);
        let result = a.and_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0xFF);
    }

    #[test]
    fn test_or_with_zero() {
        let a = IntBound::from_constant(0);
        let b = IntBound::from_constant(0xFF);
        let result = a.or_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), 0xFF);
    }

    #[test]
    fn test_xor_with_all_ones() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::from_constant(-1); // all ones
        let result = a.xor_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant_int(), !0xFFi64);
    }

    #[test]
    fn test_intersect_identity() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(0, 100);
        let changed = a.intersect(&b).unwrap();
        assert!(!changed);
        assert_eq!(a.lower, 0);
        assert_eq!(a.upper, 100);
    }

    #[test]
    fn test_intersect_subset() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(20, 80);
        let changed = a.intersect(&b).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 20);
        assert_eq!(a.upper, 80);
    }

    #[test]
    fn test_lshift_variable_shift() {
        let a = IntBound::bounded(1, 2);
        let shift = IntBound::bounded(1, 3);
        let result = a.lshift_bound(&shift);
        assert_eq!(result.lower, 2); // 1 << 1
        assert_eq!(result.upper, 16); // 2 << 3
    }

    #[test]
    fn test_rshift_variable_shift() {
        let a = IntBound::bounded(16, 32);
        let shift = IntBound::bounded(1, 3);
        let result = a.rshift_bound(&shift);
        assert_eq!(result.lower, 2); // 16 >> 3
        assert_eq!(result.upper, 16); // 32 >> 1
    }

    #[test]
    fn test_debug_check() {
        let b = IntBound::from_constant(42);
        assert!(b._debug_check());

        let b = IntBound::unbounded();
        assert!(b._debug_check());

        let b = IntBound::bounded(0, 100);
        assert!(b._debug_check());
    }

    #[test]
    fn test_add_tnum_propagation() {
        // Adding two even numbers should give an even number
        // Even: bit 0 is known 0
        let a = IntBound::from_knownbits(0, 0xFFFFFFFFFFFFFFFE);
        let b = IntBound::from_knownbits(0, 0xFFFFFFFFFFFFFFFE);
        let result = a.add_bound(&b);
        // The result's bit 0 should be known 0 (even + even = even)
        assert_eq!(result.tvalue & 1, 0);
        assert_eq!(result.tmask & 1, 0);
    }

    #[test]
    fn test_lshift_knownbits_zero_fill() {
        // After left shift by 2, low 2 bits should be known 0
        let a = IntBound::unbounded();
        let shift = IntBound::from_constant(2);
        let result = a.lshift_bound(&shift);
        assert_eq!(result.tvalue & 3, 0);
        assert_eq!(result.tmask & 3, 0);
    }

    #[test]
    fn test_bounded_knownbits_interaction() {
        // [0, 3] should know that all bits above bit 1 are 0
        let b = IntBound::bounded(0, 3);
        // bits above position 1 should be known 0
        assert_eq!(b.tmask >> 2, 0);
        assert_eq!(b.tvalue >> 2, 0);
    }

    #[test]
    fn test_make_unsigned_le() {
        let other = IntBound::bounded(0, 100);
        let mut a = IntBound::unbounded();
        let changed = a.make_unsigned_le(&other).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 0);
        assert_eq!(a.upper, 100);
    }

    #[test]
    fn test_make_unsigned_lt() {
        let other = IntBound::bounded(0, 100);
        let mut a = IntBound::unbounded();
        let changed = a.make_unsigned_lt(&other).unwrap();
        assert!(changed);
        assert_eq!(a.lower, 0);
        assert_eq!(a.upper, 99);
    }

    #[test]
    fn test_lshift_bound_cannot_overflow() {
        let a = IntBound::bounded(0, 10);
        let shift = IntBound::from_constant(2);
        assert!(a.lshift_bound_cannot_overflow(&shift));

        let a = IntBound::bounded(0, i64::MAX);
        let shift = IntBound::from_constant(1);
        assert!(!a.lshift_bound_cannot_overflow(&shift));
    }

    #[test]
    fn test_mul2_bound_no_overflow() {
        let a = IntBound::bounded(0, 10);
        let result = a.mul2_bound_no_overflow();
        assert_eq!(result.lower, 0);
        assert_eq!(result.upper, 20);
        // Result should be known even
        assert_eq!(result.tvalue & 1, 0);
        assert_eq!(result.tmask & 1, 0);
    }

    #[test]
    fn test_clone_bound() {
        let a = IntBound::bounded(10, 20);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_default() {
        let b = IntBound::default();
        assert!(b.is_unbounded());
    }

    #[test]
    fn int_bound_rc_shares_identity_and_mutation() {
        let a = IntBoundRc::new(IntBound::unbounded());
        let b = a.clone();
        assert!(IntBoundRc::ptr_eq(&a, &b));
        a.borrow_mut().lower = 3;
        assert_eq!(b.borrow().lower, 3);
        let c = IntBoundRc::new(IntBound::unbounded());
        assert!(!IntBoundRc::ptr_eq(&a, &c));
    }
}
