//! RPython's `rpython.rlib.rbigint`, ported structurally to Rust.
//!
//! Upstream pin:
//! - repository revision: `pypy/main` at
//!   `1f81807bcfde3dfd76d96e576bdb5f69188713ff`
//! - `rpython/rlib/rbigint.py` blob:
//!   `ce4b24400fec311792fddd4554a469d33ba73290`
//!
//! The method and helper order follows the upstream file.  Digits use the
//! existing `GcArray(Signed)` representation (`TypedItemsBlock`) directly:
//! `_digits` is the GC-array pointer and `_size` is the signed logical length.
//! This is deliberately not a Rust `Vec`; the storage shape is part of the
//! RPython port and must remain visible to translation and the GC.

use std::borrow::Cow;
use std::cmp::Ordering;

/// The GC boundary this port needs and `rbigint.py` does not have, because
/// upstream's translator generates it. Kept out of this file so what remains
/// here is the port of the upstream one.
mod gc;

use gc::PendingPartsCacheDigitRoot;
pub use gc::*;

use crate::lltypesystem::rlist::{
    Digits, TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TypedItemsBlock, typed_items_block_capacity,
    typed_items_block_items_base,
};

pub const SUPPORT_INT128: bool = true;
pub const BYTEORDER: &str = if cfg!(target_endian = "little") {
    "little"
} else {
    "big"
};

// rbigint.py — pyre's supported target is 64-bit and majit has the
// corresponding SignedLongLongLong / UnsignedLongLongLong low-level types.
pub const SHIFT: i64 = 63;
pub type Digit = i64;
pub type UDigit = u64;
pub type WideDigit = i128;
pub type UWideDigit = u128;
pub const MASK: Digit = i64::MAX;
pub const FLOAT_MULTIPLIER: f64 = (1_u64 << SHIFT) as f64;
pub const BASE_AS_FLOAT: f64 = FLOAT_MULTIPLIER;

pub const MIN_INT_VALUE: i64 = -1_i64 << SHIFT;
pub const ULONGLONG_BOUND: u64 = 1_u64 << 63;
pub const LONGLONG_MIN: i64 = i64::MIN;
pub const BITCOUNT_K1: u64 = 0x5555_5555_5555_5555;
pub const BITCOUNT_K2: u64 = 0x3333_3333_3333_3333;
pub const BITCOUNT_K4: u64 = 0x0f0f_0f0f_0f0f_0f0f;
pub const BITCOUNT_KF: u64 = 0x0101_0101_0101_0101;

#[majit_macros::always_inline]
#[inline]
pub const fn int_in_valid_range(x: i64) -> bool {
    x != (-9223372036854775807_i64 - 1)
}

pub const USE_KARATSUBA: bool = true;
pub const KARATSUBA_CUTOFF: i64 = 19;
pub const KARATSUBA_SQUARE_CUTOFF: i64 = 2 * KARATSUBA_CUTOFF;
pub const FIVEARY_CUTOFF: i64 = 8;
/// rbigint.py `LimitHolder` / module-global `HOLDER`.
///
/// PyPy deliberately keeps these algorithm cutoffs on one mutable,
/// process-global object (its tests temporarily tune `DIV_LIMIT`). Atomics are
/// only the Rust synchronization envelope around that same shared owner; TLS
/// or per-call copies would change both ownership and observability.
#[allow(non_snake_case)]
pub struct LimitHolder {
    pub DIV_LIMIT: std::sync::atomic::AtomicI64,
    pub STR2INT_LIMIT: std::sync::atomic::AtomicI64,
    pub MINSIZE_STR2INT: std::sync::atomic::AtomicI64,
}

#[allow(non_snake_case)]
pub static HOLDER: LimitHolder = LimitHolder {
    DIV_LIMIT: std::sync::atomic::AtomicI64::new(21),
    STR2INT_LIMIT: std::sync::atomic::AtomicI64::new(2048),
    MINSIZE_STR2INT: std::sync::atomic::AtomicI64::new(4000),
};

#[inline]
fn holder_limit(slot: &std::sync::atomic::AtomicI64) -> i64 {
    slot.load(std::sync::atomic::Ordering::Relaxed)
}

// rbigint.py `@specialize.argtype(0) _mask_digit`.
//
// RPython emits one graph for every concrete argument type.  Keep those
// graphs separate here too: collapsing all callers through `u128` introduces
// wide casts and wide bit operations into paths whose RPython graph only uses
// Signed or Unsigned.  `UDIGIT_MASK` is `intmask` on our 64-bit target, so
// every specialization returns the signed STORE_TYPE.
#[inline]
fn _mask_digit(x: Digit) -> Digit {
    x & MASK as Digit
}

#[inline]
fn _mask_udigit(x: UDigit) -> Digit {
    (x & MASK as UDigit) as Digit
}

#[inline]
fn _mask_widedigit(x: WideDigit) -> Digit {
    (x & MASK as WideDigit) as Digit
}

#[inline]
fn _mask_uwidedigit(x: UWideDigit) -> Digit {
    (x & MASK as UWideDigit) as Digit
}

#[inline]
fn _widen_digit(x: Digit) -> WideDigit {
    x as WideDigit
}

#[inline]
fn _unsigned_widen_digit(x: Digit) -> UWideDigit {
    x as UDigit as UWideDigit
}

// rbigint.py `@specialize.argtype(0) _store_digit`.  This is only a
// cast to STORE_TYPE; masking belongs to callers exactly as it does upstream.
#[inline]
fn _store_digit(x: Digit) -> Digit {
    x
}

#[inline]
fn _store_udigit(x: UDigit) -> Digit {
    x as Digit
}

#[inline]
fn _store_widedigit(x: WideDigit) -> Digit {
    x as Digit
}

#[inline]
fn _store_uwidedigit(x: UWideDigit) -> Digit {
    x as Digit
}

#[majit_macros::always_inline]
#[inline]
fn _load_unsigned_digit(x: Digit) -> UDigit {
    x as UDigit
}

/// Address of `l->items[x]`, the way `rtype_getitem` reaches it after
/// translation.
///
/// `rtype_getitem` / `rtype_setitem` (rlist.py, :272-303) select
/// `dum_checkidx` only where the source catches `IndexError`; every digit
/// access takes the default `dum_nocheck`, under which `ll_getitem_nonneg`
/// keeps `ll_assert(index >= 0, ...)` and folds the length test away. Going
/// through a `&[Digit]` instead reloads the block's length header and branches
/// on it once per digit, which the translated form never does.
///
/// # Safety
///
/// `digits` must be a live digit block and `x` an index within its capacity.
#[majit_macros::always_inline]
#[inline]
unsafe fn digits_item(digits: *mut TypedItemsBlock, x: i64) -> *mut Digit {
    debug_assert!(x >= 0, "unexpectedly negative digit index");
    debug_assert!(!digits.is_null());
    debug_assert!((x as usize) < unsafe { typed_items_block_capacity(digits) });
    unsafe {
        (digits as *mut u8)
            .add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET)
            .cast::<Digit>()
            .add(x as usize)
    }
}

/// Copy `digits` into a freshly allocated block. An empty source writes the
/// `NULLDIGIT` sentinel that `rbigint.__init__` stores for `NULLDIGITS`.
unsafe fn fill_digit_block(block: *mut TypedItemsBlock, digits: &[Digit]) {
    unsafe {
        let base = typed_items_block_items_base(block) as *mut Digit;
        if digits.is_empty() {
            *base = NULLDIGIT;
        } else {
            let mut i = 0;
            while i < digits.len() {
                *base.add(i) = digits[i];
                i += 1;
            }
        }
    }
}

pub const NULLDIGIT: Digit = 0;
pub const ONEDIGIT: Digit = 1;

// rbigint.py / 3560.  The translated module owns these objects
// process-globally; every zero/one result aliases the corresponding immutable
// prebuilt digit array instead of allocating another GcArray(Signed).
//
// Store stable raw RBigInt addresses in OnceLock<usize>, following pyre's
// existing process-global immortal-object convention.  The payload boxes
// themselves never move; their `_digits` slots are visited by
// `walk_rbigint_cache_digit_slots`, just like RPython's translated prebuilt
// root graph.  Do not make these TLS: PyPy has one module-global owner.
static NULLRBIGINT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static ONERBIGINT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static ONENEGATIVERBIGINT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static FIVERBIGINT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// The four owners above, in the order [`PREBUILT_DIGITS`] indexes them.
fn prebuilt_slots() -> [&'static std::sync::OnceLock<usize>; 4] {
    [&NULLRBIGINT, &ONERBIGINT, &ONENEGATIVERBIGINT, &FIVERBIGINT]
}

/// Digit-array identity of each prebuilt, held together and readable without
/// touching the objects themselves.
///
/// A source line upstream spells `return NULLRBIGINT` — it hands back the
/// module-global object, and nothing is rediscovered. pyre's `RBigInt` is a
/// by-value handle that only becomes a heap object at the boxing step, so the
/// boxing step is where that identity has to be recovered, on every allocation,
/// almost always answering "not a prebuilt".
///
/// `prebuilt` gives each of the four its own one-element digit array, so that
/// answer is decided by the digit pointer alone. Keeping the four pointers in
/// one array makes the common case four plain loads out of a single cache line,
/// instead of four `OnceLock` state words each followed by a dereference into a
/// separately allocated payload. A candidate match still reads the object, so
/// the test stays exactly as exact as it was.
static PREBUILT_DIGITS: [std::sync::atomic::AtomicUsize; 4] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

fn _check_digits(digits: &[Digit]) {
    for &digit in digits {
        debug_assert!(digit >= 0);
        debug_assert_eq!(digit as UDigit & MASK as UDigit, digit as UDigit);
    }
}

#[inline]
pub const fn intsign(i: i64) -> i64 {
    if i == 0 {
        0
    } else if i < 0 {
        -1
    } else {
        1
    }
}

/// `rpython.rlib.rbigint.rbigint`.
///
/// `_size` is `sign * number_of_significant_digits`.  Zero has `_size == 0`
/// and still owns a one-element `_digits` array containing `0`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("_digits[*]", "_size")]
pub struct RBigInt {
    _digits: *mut TypedItemsBlock,
    _size: i64,
}

/// `rpython.rlib.rstring.NumberStringParser`.
///
/// This deliberately remains an iterator-like parser over the original
/// string.  In particular, underscore validation happens as digits are
/// consumed, and the power-of-two conversion rewinds over the same stream.
pub struct NumberStringParser<'a> {
    s: &'a str,
    pub start: i64,
    pub end: i64,
    pub sign: i64,
    pub original_base: i64,
    pub base: i64,
    allow_underscores: bool,
    i: i64,
}

impl<'a> NumberStringParser<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: &'a str,
        base: i64,
        allow_underscores: bool,
        no_implicit_octal: bool,
        start: i64,
        end: Option<i64>,
        max_str_digits: i64,
        disallow_whitespace_after_sign: bool,
    ) -> Result<Self, RBigIntError> {
        if !s.is_ascii() {
            return Err(RBigIntError::ParseString);
        }
        let mut parser = Self {
            s,
            start,
            end: end.unwrap_or(s.len() as i64),
            sign: 1,
            original_base: base,
            base,
            allow_underscores,
            i: start,
        };
        if parser.start > parser.end || parser.end > s.len() as i64 {
            return Err(RBigIntError::ParseString);
        }
        parser._strip_spaces();
        if parser._startswith1(b'-') {
            parser.sign = -1;
            parser.start += 1;
            if !disallow_whitespace_after_sign {
                parser._strip_spaces();
            }
        } else if parser._startswith1(b'+') {
            parser.start += 1;
            if !disallow_whitespace_after_sign {
                parser._strip_spaces();
            }
        }

        let mut parsed_base = base;
        if parsed_base == 0 {
            if parser._startswith2(b"0x") || parser._startswith2(b"0X") {
                parsed_base = 16;
            } else if parser._startswith2(b"0b") || parser._startswith2(b"0B") {
                parsed_base = 2;
            } else if parser._startswith1(b'0') {
                if no_implicit_octal && !(parser._startswith2(b"0o") || parser._startswith2(b"0O"))
                {
                    // RPython uses pseudo-base 1 to make only zero valid.
                    parsed_base = 1;
                } else {
                    parsed_base = 8;
                }
            } else {
                parsed_base = 10;
            }
        } else if !(2..=36).contains(&parsed_base) {
            return Err(RBigIntError::InvalidBase);
        }
        parser.base = parsed_base;

        // This check intentionally precedes base-prefix removal.  It permits
        // the CPython-compatible `0x_1`, but rejects `_1`.
        if parser._startswith1(b'_') {
            return Err(RBigIntError::ParseString);
        }
        if parser.base == 16 && (parser._startswith2(b"0x") || parser._startswith2(b"0X")) {
            parser.start += 2;
        }
        if parser.base == 8 && (parser._startswith2(b"0o") || parser._startswith2(b"0O")) {
            parser.start += 2;
        }
        if parser.base == 2 && (parser._startswith2(b"0b") || parser._startswith2(b"0B")) {
            parser.start += 2;
        }
        if parser.start == parser.end {
            return Err(RBigIntError::ParseString);
        }
        parser.i = parser.start;
        if max_str_digits > 0 {
            // Keep the upstream count over `self.s`, not merely the active
            // slice.  NumberStringParser does this before consuming digits.
            let length =
                parser.end - parser.start - s.bytes().filter(|&c| c == b'_').count() as i64;
            if length > max_str_digits {
                return Err(RBigIntError::MaxStrDigits);
            }
        }
        Ok(parser)
    }

    #[inline]
    fn _startswith1(&self, prefix: u8) -> bool {
        self.start < self.end && self.s.as_bytes()[self.start as usize] == prefix
    }

    #[inline]
    fn _startswith2(&self, prefix: &[u8; 2]) -> bool {
        self.start + 1 < self.end
            && self.s.as_bytes()[self.start as usize] == prefix[0]
            && self.s.as_bytes()[(self.start + 1) as usize] == prefix[1]
    }

    fn _strip_spaces(&mut self) {
        let bytes = self.s.as_bytes();
        while self.start < self.end && is_parser_space(bytes[self.start as usize]) {
            self.start += 1;
        }
        while self.start < self.end && is_parser_space(bytes[(self.end - 1) as usize]) {
            self.end -= 1;
        }
    }

    pub fn rewind(&mut self) {
        self.i = self.start;
    }

    pub fn next_digit(&mut self) -> Result<i64, RBigIntError> {
        if self.i >= self.end {
            return Ok(-1);
        }
        let bytes = self.s.as_bytes();
        let mut c = bytes[self.i as usize];
        if self.allow_underscores && c == b'_' {
            self.i += 1;
            if self.i >= self.end {
                return Err(RBigIntError::ParseString);
            }
            c = bytes[self.i as usize];
        }
        let digit = ascii_digit(c).ok_or(RBigIntError::ParseString)?;
        if digit >= self.base {
            return Err(RBigIntError::ParseString);
        }
        self.i += 1;
        Ok(digit)
    }

    fn _all_digits10(&mut self) -> Result<(Cow<'a, str>, i64, i64), RBigIntError> {
        for index in self.start..self.end {
            let c = self.s.as_bytes()[index as usize];
            if !c.is_ascii_digit() {
                if c == b'_' && self.allow_underscores {
                    break;
                }
                return Err(RBigIntError::ParseString);
            }
        }
        if self.s.as_bytes()[self.start as usize..self.end as usize]
            .iter()
            .all(u8::is_ascii_digit)
        {
            return Ok((Cow::Borrowed(self.s), self.start, self.end));
        }

        debug_assert!(self.allow_underscores);
        let capacity = usize::try_from(self.end - self.start).map_err(|_| RBigIntError::Memory)?;
        let mut builder = String::new();
        builder
            .try_reserve_exact(capacity)
            .map_err(|_| RBigIntError::Memory)?;
        loop {
            let digit = self.next_digit()?;
            if digit < 0 {
                let end = builder.len() as i64;
                return Ok((Cow::Owned(builder), 0, end));
            }
            builder.push((b'0' + digit as u8) as char);
        }
    }

    pub fn prev_digit(&mut self) -> Result<i64, RBigIntError> {
        if self.i <= self.start {
            return Err(RBigIntError::ParseString);
        }
        self.i -= 1;
        let mut c = self.s.as_bytes()[self.i as usize];
        if self.allow_underscores && c == b'_' {
            if self.i == 0 {
                return Err(RBigIntError::ParseString);
            }
            self.i -= 1;
            c = self.s.as_bytes()[self.i as usize];
        }
        ascii_digit(c).ok_or(RBigIntError::ParseString)
    }
}

#[inline]
const fn is_parser_space(c: u8) -> bool {
    matches!(c, b' ' | b'\x0c' | b'\n' | b'\r' | b'\t' | b'\x0b')
}

#[inline]
const fn ascii_digit(c: u8) -> Option<i64> {
    if c >= b'0' && c <= b'9' {
        Some((c - b'0') as i64)
    } else if c >= b'A' && c <= b'Z' {
        Some((c - b'A') as i64 + 10)
    } else if c >= b'a' && c <= b'z' {
        Some((c - b'a') as i64 + 10)
    } else {
        None
    }
}

// The object is immutable after construction in every public operation.
// Its backing GcArray contains scalar words and is safe to read concurrently.
unsafe impl Send for RBigInt {}
unsafe impl Sync for RBigInt {}

/// RPython's `tuple2` of two `rbigint`s — the return type of `rbigint.divmod`
/// and `rbigint.int_divmod` (rbigint.py / 1050, both `@jit.elidable`).
///
/// RPython gives a multiple-return elidable a real GC struct, so the trace is
/// one `call_pure_r` yielding this pointer plus two `getfield_gc_r`. Rust
/// returns `(RBigInt, RBigInt)` by value, which has no address the JIT can name
/// — hence this explicit struct, whose two fields are the traced edges.
#[repr(C)]
pub struct RBigIntPair {
    pub item0: *mut RBigInt,
    pub item1: *mut RBigInt,
}

pub const RBIGINT_PAIR_SIZE: usize = std::mem::size_of::<RBigIntPair>();
pub const RBIGINT_PAIR_ITEM0_OFFSET: usize = std::mem::offset_of!(RBigIntPair, item0);
pub const RBIGINT_PAIR_ITEM1_OFFSET: usize = std::mem::offset_of!(RBigIntPair, item1);

impl Clone for RBigInt {
    fn clone(&self) -> Self {
        // Python assignment shares an immutable `rbigint` object.  A clone of
        // the Rust handle therefore shares the immutable GcArray as well; only
        // freshly allocated arithmetic results ever mutate digit storage.
        Self {
            _digits: self._digits,
            _size: self._size,
        }
    }
}

impl std::fmt::Debug for RBigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RBigInt")
            .field("_digits", &self.digits())
            .field("_size", &self._size)
            .finish()
    }
}

impl std::fmt::Display for RBigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.str(0).map_err(|_| std::fmt::Error)?;
        f.write_str(&value)
    }
}

impl PartialEq for RBigInt {
    // rbigint.py: Python `__eq__`/`__ne__` exist for tests only and
    // carry @not_rpython. This Rust comparison trait is the same host-facing
    // adapter; translated code calls the inherent elidable `RBigInt::eq`.
    #[majit_macros::not_rpython]
    fn eq(&self, other: &Self) -> bool {
        RBigInt::eq(self, other)
    }
}

impl Eq for RBigInt {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RBigIntSign {
    Minus,
    NoSign,
    Plus,
}

impl RBigInt {
    /// Rust ownership spelling for an RPython reference assignment.
    ///
    /// The host needs a shallow value clone, but source translation aliases
    /// this exact method's input and output so upstream fast paths that return
    /// `self`/`other` do not allocate a new GC payload. Do not use this for
    /// `rbigint.neg`, which deliberately constructs a fresh handle.
    #[doc(hidden)]
    #[inline]
    pub fn translated_alias(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn prebuilt(index: usize, digit: Digit, sign: i64) -> Self {
        let raw = *prebuilt_slots()[index].get_or_init(|| {
            let block = unsafe { Digits::prebuilt(digit) };
            let value = Self {
                _digits: block,
                _size: sign,
            };
            let raw = Box::into_raw(Box::new(value)) as usize;
            // Publish the digit identity only once the object it names exists;
            // `prebuilt_payload_pointer` matches on this word and then reads
            // the object through the owner slot.
            PREBUILT_DIGITS[index].store(block as usize, std::sync::atomic::Ordering::Release);
            raw
        }) as *const Self;
        // RPython assignment returns the process-global rbigint itself. The
        // host needs a shallow owned handle; source translation aliases the
        // exact immortal payload address instead of allocating a clone.
        unsafe { (&*raw).translated_alias() }
    }

    #[inline]
    fn negative_one() -> Self {
        Self::prebuilt(2, ONEDIGIT, -1)
    }

    #[inline]
    fn five() -> Self {
        Self::prebuilt(3, 5, 1)
    }

    /// rbigint.py `__init__(digits=NULLDIGITS, sign=0, size=0)`.
    ///
    /// Adopts `digits` by copying them into a fresh `Digits::new` block.
    /// `digits` must not alias another bigint's GcArray: that allocation
    /// can collect and move the source under this slice. Aliasing callers
    /// use [`RBigInt::new_from_rooted`] so the source handle is re-read
    /// after the malloc, the way the translator keeps the source on the
    /// shadow stack. Stack arrays (`fromint`, 1-2 digit `mul` results)
    /// stay valid across the allocation and pay no heap copy.
    pub fn new(digits: &[Digit], sign: i64, size: i64) -> Self {
        // Upstream calls `_check_digits` only under
        // `if not we_are_translated()`.  The translated constructor relies on
        // the same digit invariant rather than carrying test-only validation
        // into the flow graph.
        debug_assert!(size >= 0);
        let logical_len = if size == 0 { digits.len() as i64 } else { size };
        let block = unsafe { Digits::new(digits.len().max(1)) };
        unsafe {
            fill_digit_block(block, digits);
        }
        Self {
            _digits: block,
            _size: logical_len * sign,
        }
    }

    /// `rbigint.__init__` over a slice of a live source whose `_digits`
    /// slot is already on the shadow stack. `Digits::new` may collect;
    /// the range is re-derived from `src` afterwards.
    fn new_from_rooted(
        src: &RBigIntGcRoot,
        start: usize,
        end: usize,
        sign: i64,
        size: i64,
    ) -> Self {
        debug_assert!(size >= 0);
        debug_assert!(start <= end);
        let len = end - start;
        let logical_len = if size == 0 { len as i64 } else { size };
        let block = unsafe { Digits::new(len.max(1)) };
        unsafe {
            fill_digit_block(block, &src.digits()[start..end]);
        }
        Self {
            _digits: block,
            _size: logical_len * sign,
        }
    }

    /// Fallible form of [`RBigInt::new`] for upstream paths whose translated
    /// allocation has an explicit MemoryError edge.
    #[allow(dead_code)] // rbigint.py `rbigint.__init__`
    fn try_new(digits: &[Digit], sign: i64, size: i64) -> Result<Self, RBigIntError> {
        debug_assert!(size >= 0);
        let logical_len = if size == 0 {
            i64::try_from(digits.len()).map_err(|_| RBigIntError::Memory)?
        } else {
            size
        };
        let allocation_size = digits.len().max(1);
        let block = unsafe { Digits::try_new(allocation_size) }.ok_or(RBigIntError::Memory)?;
        unsafe {
            fill_digit_block(block, digits);
        }
        Ok(Self {
            _digits: block,
            _size: logical_len.checked_mul(sign).ok_or(RBigIntError::Memory)?,
        })
    }

    /// Fallible [`RBigInt::new_from_rooted`].
    fn try_new_from_rooted(
        src: &RBigIntGcRoot,
        start: usize,
        end: usize,
        sign: i64,
        size: i64,
    ) -> Result<Self, RBigIntError> {
        debug_assert!(size >= 0);
        debug_assert!(start <= end);
        let len = end - start;
        let logical_len = if size == 0 {
            i64::try_from(len).map_err(|_| RBigIntError::Memory)?
        } else {
            size
        };
        let allocation_size = len.max(1);
        let block = unsafe { Digits::try_new(allocation_size) }.ok_or(RBigIntError::Memory)?;
        unsafe {
            fill_digit_block(block, &src.digits()[start..end]);
        }
        Ok(Self {
            _digits: block,
            _size: logical_len.checked_mul(sign).ok_or(RBigIntError::Memory)?,
        })
    }

    /// Allocate the translated equivalent of `[NULLDIGIT] * size`:
    /// `ll_alloc_and_set(LIST, size, 0)`, which is `ll_newlist` followed by
    /// `rgc.ll_arrayclear` (rtyper/rlist.py).
    #[inline]
    fn with_size(size: i64, sign: i64) -> Self {
        debug_assert!(size >= 0);
        let block = unsafe { Digits::alloc_and_set_zero(size as usize) };
        Self {
            _digits: block,
            _size: size * sign,
        }
    }

    /// `[NULLDIGIT] * size` with operand `_digits` as stack livevars.
    ///
    /// The digit-array malloc is the only collecting step; the subsequent
    /// fill loop only stores into that array. `a`/`b` are by-value handles —
    /// their addresses are not GC object starts — so the livevars are the
    /// `GcArray(Signed)` bodies. Reload `_digits` after the malloc and return
    /// a plain handle for the result; an owner-root `Deref` on every digit
    /// store is not a livevar.
    fn with_size_livevars(size: i64, sign: i64, a: &mut RBigInt, b: &mut RBigInt) -> Self {
        debug_assert!(size >= 0);
        let mut roots = [majit_ir::GcRef::NULL; 2];
        Self::with_size_reloading_digits(size, sign, a, b, &mut roots)
    }

    fn with_size_reloading_digits(
        size: i64,
        sign: i64,
        a: &mut RBigInt,
        b: &mut RBigInt,
        roots: &mut [majit_ir::GcRef],
    ) -> Self {
        debug_assert!(roots.len() >= 2);
        roots[0] = majit_ir::GcRef(a._digits as usize);
        roots[1] = majit_ir::GcRef(b._digits as usize);
        let block = unsafe { Digits::alloc_and_set_zero_rooted(size as usize, roots) };
        a._digits = roots[0].0 as *mut _;
        b._digits = roots[1].0 as *mut _;
        Self {
            _digits: block,
            _size: size * sign,
        }
    }

    /// Host livevars for `[NULLDIGIT] * size` while two operand handles stay
    /// live across it and across later collecting calls. The owner-root slots
    /// keep each handle's `_digits` edge, which is what `gct_fv_gc_malloc`
    /// reloads into `a`/`b` after that malloc.
    fn with_size_keeping_handles(
        size: i64,
        sign: i64,
        a: &RBigInt,
        b: &RBigInt,
    ) -> (RBigIntGcRoot, RBigIntGcRoot, RBigIntGcRoot) {
        let a_hold = live_rbigint(a);
        let b_hold = live_rbigint(b);
        let z = RBigIntGcRoot::new(Self::with_size(size, sign));
        (a_hold, b_hold, z)
    }

    /// The explicit Rust form of RPython's implicit `gc_malloc_array`
    /// `MemoryError` edge. Used by operations whose public Rust signature
    /// already carries `RBigIntError`.
    #[inline]
    fn try_with_size(size: i64, sign: i64) -> Result<Self, RBigIntError> {
        let allocation_size = usize::try_from(size).map_err(|_| RBigIntError::Memory)?;
        let block = unsafe { Digits::try_alloc_and_set_zero(allocation_size) }
            .ok_or(RBigIntError::Memory)?;
        Ok(Self {
            _digits: block,
            _size: size * sign,
        })
    }

    #[inline]
    pub fn zero() -> Self {
        Self::prebuilt(0, NULLDIGIT, 0)
    }

    #[inline]
    pub fn one() -> Self {
        Self::prebuilt(1, ONEDIGIT, 1)
    }

    #[inline]
    fn digits(&self) -> &[Digit] {
        unsafe {
            std::slice::from_raw_parts(
                typed_items_block_items_base(self._digits) as *const Digit,
                typed_items_block_capacity(self._digits),
            )
        }
    }

    #[inline]
    pub fn get_sign(&self) -> i64 {
        intsign(self._size)
    }

    #[inline]
    fn _set_sign(&mut self, sign: i64) {
        debug_assert!(matches!(sign, -1..=1));
        self._size = self._size.abs() * sign;
    }

    #[majit_macros::always_inline]
    #[inline]
    pub fn digit(&self, x: i64) -> Digit {
        unsafe { *digits_item(self._digits, x) }
    }

    #[majit_macros::always_inline]
    #[inline]
    pub fn widedigit(&self, x: i64) -> WideDigit {
        _widen_digit(unsafe { *digits_item(self._digits, x) })
    }

    #[majit_macros::always_inline]
    #[inline]
    pub fn uwidedigit(&self, x: i64) -> UWideDigit {
        _unsigned_widen_digit(unsafe { *digits_item(self._digits, x) })
    }

    #[majit_macros::always_inline]
    #[inline]
    pub fn udigit(&self, x: i64) -> UDigit {
        _load_unsigned_digit(unsafe { *digits_item(self._digits, x) })
    }

    #[majit_macros::always_inline]
    #[inline]
    fn setdigit(&mut self, x: i64, val: Digit) {
        let val = _mask_digit(val);
        debug_assert!(val >= 0);
        unsafe { *digits_item(self._digits, x) = _store_digit(val) };
    }

    // rbigint.py `@specialize.argtype(2) setdigit`, Unsigned graph.
    #[majit_macros::always_inline]
    #[inline]
    fn setdigit_udigit(&mut self, x: i64, val: UDigit) {
        let val = _mask_udigit(val);
        debug_assert!(val >= 0);
        unsafe { *digits_item(self._digits, x) = _store_digit(val) };
    }

    // rbigint.py `@specialize.argtype(2) setdigit`, LONG_TYPE graph.
    #[majit_macros::always_inline]
    #[inline]
    fn setdigit_widedigit(&mut self, x: i64, val: WideDigit) {
        let val = _mask_widedigit(val);
        debug_assert!(val >= 0);
        unsafe { *digits_item(self._digits, x) = _store_digit(val) };
    }

    // rbigint.py `@specialize.argtype(2) setdigit`, ULONG_TYPE graph.
    #[majit_macros::always_inline]
    #[inline]
    fn setdigit_uwidedigit(&mut self, x: i64, val: UWideDigit) {
        let val = _mask_uwidedigit(val);
        debug_assert!(val >= 0);
        unsafe { *digits_item(self._digits, x) = _store_digit(val) };
    }

    #[majit_macros::always_inline]
    #[inline]
    pub fn numdigits(&self) -> i64 {
        let w = self._size.abs();
        let w = if w == 0 { 1 } else { w };
        debug_assert!(w > 0);
        w
    }

    /// rbigint.py `fromint`.
    #[majit_macros::jit_elidable]
    pub fn fromint(intval: i64) -> Self {
        let (sign, ival) = if intval < 0 {
            // `-r_uint(intval)` in rbigint.py: negate in the unsigned
            // domain so the most-negative machine integer remains valid.
            // An i128 negation makes rustc inject an irrelevant i128::MIN
            // overflow assertion into MIR; RPython has no such operation.
            (-1, (!intval as u64) + 1)
        } else if intval > 0 {
            (1, intval as u64)
        } else {
            return Self::zero();
        };

        let carry = ival >> SHIFT;
        if carry != 0 {
            Self::new(
                &[_store_udigit(ival & MASK as UDigit), _store_udigit(carry)],
                sign,
                2,
            )
        } else {
            Self::new(&[_store_udigit(ival & MASK as UDigit)], sign, 1)
        }
    }

    /// rbigint.py `frombool`.
    #[majit_macros::jit_elidable]
    pub fn frombool(value: bool) -> Self {
        if value { Self::one() } else { Self::zero() }
    }

    /// rbigint.py `fromlong`; Python-only upstream helper, represented
    /// with Rust's widest built-in signed integer for port tests.
    #[majit_macros::not_rpython]
    pub fn fromlong(value: i128) -> Self {
        let (digits, sign) = args_from_long(value);
        Self::new(&digits, sign, digits.len() as i64)
    }

    /// rbigint.py `fromfloat`.
    #[majit_macros::jit_elidable]
    pub fn fromfloat(dval: f64) -> Result<Self, RBigIntError> {
        if dval.is_infinite() {
            return Err(RBigIntError::InfiniteFloat);
        }
        if dval.is_nan() {
            return Err(RBigIntError::NanFloat);
        }
        Ok(Self::_fromfloat_finite(dval))
    }

    /// rbigint.py `_fromfloat_finite`.
    #[majit_macros::jit_elidable]
    pub fn _fromfloat_finite(mut dval: f64) -> Self {
        let sign = if dval < 0.0 {
            dval = -dval;
            -1
        } else {
            1
        };
        let (mut frac, expo) = float_frexp(dval);
        if expo <= 0 {
            return Self::zero();
        }
        let ndig = (expo as i64 - 1) / SHIFT + 1;
        let mut value = Self::with_size(ndig, sign);
        frac *= 2_f64.powi((expo - 1) % SHIFT as i32 + 1);
        let mut i = ndig;
        while i > 0 {
            i -= 1;
            let bits = frac as Digit;
            value.setdigit(i, bits);
            frac -= bits as f64;
            frac *= FLOAT_MULTIPLIER;
        }
        value
    }

    /// rbigint.py `fromrarith_int`.
    #[majit_macros::jit_elidable]
    pub fn fromrarith_int(value: i64) -> Self {
        let (digits, sign) = args_from_rarith_int(value);
        Self::new(&digits, sign, digits.len() as i64)
    }

    /// Unsigned specialization generated by rbigint.py
    /// `@specialize.argtype(0)` for `r_uint`.
    #[majit_macros::jit_elidable]
    pub fn fromrarith_uint(value: u64) -> Self {
        let (digits, sign) = args_from_rarith_uint(value);
        Self::new(&digits, sign, digits.len() as i64)
    }

    /// rbigint.py `fromdecimalstr`.
    #[majit_macros::jit_elidable]
    pub fn fromdecimalstr(s: &str) -> Self {
        _decimalstr_to_bigint(s, 0, s.len() as i64)
    }

    /// rbigint.py `fromstr`.
    #[majit_macros::jit_elidable]
    pub fn fromstr(s: &str, base: i64, allow_underscores: bool) -> Result<Self, RBigIntError> {
        if !s.is_ascii() {
            return Err(RBigIntError::ParseString);
        }
        let bytes = s.as_bytes();
        let mut start = 0;
        let mut end = bytes.len();
        while start < end && is_parser_space(bytes[start]) {
            start += 1;
        }
        while start < end && is_parser_space(bytes[end - 1]) {
            end -= 1;
        }
        let stripped = &s[start..end];
        let mut parser_end = stripped.len() as i64;
        if parser_end > 0
            && matches!(stripped.as_bytes()[(parser_end - 1) as usize], b'l' | b'L')
            && base < 22
        {
            parser_end -= 1;
        }
        let mut parser = NumberStringParser::new(
            stripped,
            base,
            allow_underscores,
            false,
            0,
            Some(parser_end),
            0,
            false,
        )?;
        Self::_from_numberstring_parser(&mut parser)
    }

    /// rbigint.py `_from_numberstring_parser`.
    pub fn _from_numberstring_parser(
        parser: &mut NumberStringParser<'_>,
    ) -> Result<Self, RBigIntError> {
        parse_digit_string(parser)
    }

    #[majit_macros::jit_elidable]
    pub fn frombytes(bytes: &[u8], byteorder: &str, signed: bool) -> Result<Self, RBigIntError> {
        if byteorder != "big" && byteorder != "little" {
            return Err(RBigIntError::InvalidEndianness);
        }
        let is_big = byteorder == "big";
        if bytes.is_empty() {
            return Ok(Self::zero());
        }
        let msb = if is_big {
            bytes[0]
        } else {
            bytes[bytes.len() - 1]
        };
        let sign = if msb >= 0x80 && signed { -1 } else { 1 };
        // Upstream grows a temporary list (hinted with
        // `len(s) * 8 // LONG_BIT + 1`) and then passes `digits[:]` to the
        // rbigint constructor.  Reserve enough fixed-width slots for that
        // append phase here; below we make the same exact-length final copy.
        let max_digits = bytes
            .len()
            .checked_mul(8)
            .and_then(|value| value.checked_add(SHIFT as usize - 1))
            .map(|value| value / SHIFT as usize)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or(RBigIntError::Memory)?;
        let mut result = Self::try_with_size(max_digits, sign)?;
        let mut accum = 0_u128;
        let mut accumbits = 0_i64;
        let mut out = 0;
        let mut carry = 1_u128;

        let mut i = 0;
        while i < bytes.len() {
            let byte = if is_big {
                bytes[bytes.len() - 1 - i]
            } else {
                bytes[i]
            };
            let mut c = byte as u128;
            if sign == -1 {
                c = (0xff ^ c) + carry;
                carry = c >> 8;
                c &= 0xff;
            }
            accum |= c << accumbits;
            accumbits += 8;
            if accumbits >= SHIFT {
                result.setdigit_uwidedigit(out, accum);
                out += 1;
                accum >>= SHIFT;
                accumbits -= SHIFT;
            }
            i += 1;
        }
        if accumbits != 0 {
            result.setdigit_uwidedigit(out, accum);
            out += 1;
        }
        // `digits[:]` in rbigint.py `frombytes` discards the temporary
        // list's spare capacity.  Keeping `max_digits` as the final
        // GcArray capacity makes the translated storage shape depend on a
        // reservation detail. The source handle stays rooted across the
        // exact-length `Digits::new`.
        let sourced = RBigIntGcRoot::new(result);
        let mut result = Self::try_new_from_rooted(&sourced, 0, out as usize, sign, out)?;
        result._normalize();
        Ok(result)
    }

    #[majit_macros::jit_elidable]
    pub fn tobytes(
        &self,
        nbytes: i64,
        byteorder: &str,
        signed: bool,
    ) -> Result<Vec<u8>, RBigIntError> {
        if byteorder != "big" && byteorder != "little" {
            return Err(RBigIntError::InvalidEndianness);
        }
        if !signed && self.get_sign() == -1 {
            return Err(RBigIntError::InvalidSignedness);
        }

        let mut j = 0;
        let imax = self.numdigits();
        let mut accum = 0_i128;
        let mut accumbits = 0_i64;
        // StringBuilder(nbytes) in rbigint.py carries the translated
        // gc_malloc/MemoryError edge.  Preserve it explicitly instead of
        // truncating i64 to usize on wasm32 or panicking on capacity overflow.
        let capacity = usize::try_from(nbytes).map_err(|_| RBigIntError::Memory)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(capacity)
            .map_err(|_| RBigIntError::Memory)?;
        let mut carry = 1_i128;
        let mut i = 0;
        while i < imax {
            let mut d = self.widedigit(i);
            if self.get_sign() == -1 {
                d = (d ^ MASK as i128) + carry;
                carry = d >> SHIFT;
                d &= MASK as i128;
            }
            accum |= d << accumbits;
            if i == imax - 1 {
                let mut s = if self.get_sign() == -1 {
                    d ^ MASK as i128
                } else {
                    d
                };
                while s != 0 {
                    s >>= 1;
                    accumbits += 1;
                }
            } else {
                accumbits += SHIFT;
            }
            while accumbits >= 8 {
                if j >= nbytes {
                    return Err(RBigIntError::Overflow);
                }
                j += 1;
                result.push((accum & 0xff) as u8);
                accum >>= 8;
                accumbits -= 8;
            }
            i += 1;
        }
        if accumbits != 0 {
            if j >= nbytes {
                return Err(RBigIntError::Overflow);
            }
            j += 1;
            if self.get_sign() == -1 {
                accum |= (-1_i128) << accumbits;
            }
            result.push((accum & 0xff) as u8);
        }
        let signbyte = if self.get_sign() == -1 { 0xff } else { 0 };
        while j < nbytes {
            result.push(signbyte);
            j += 1;
        }
        if j == nbytes && nbytes > 0 && signed {
            let msb = result[(nbytes - 1) as usize];
            if (self.get_sign() == -1) != (msb >= 0x80) {
                return Err(RBigIntError::Overflow);
            }
        }
        if byteorder == "big" {
            result.reverse();
        }
        Ok(result)
    }

    /// rbigint.py `toint`.
    #[majit_macros::jit_elidable]
    pub fn toint(&self) -> Result<i64, RBigIntError> {
        if self.numdigits() > MAX_DIGITS_THAT_CAN_FIT_IN_INT {
            return Err(RBigIntError::Overflow);
        }
        self._toint_helper()
    }

    #[majit_macros::jit_elidable]
    fn _toint_helper(&self) -> Result<i64, RBigIntError> {
        let x = self._touint_helper()?;
        if self.get_sign() >= 0 {
            let res = x as i64;
            if res < 0 {
                return Err(RBigIntError::Overflow);
            }
            Ok(res)
        } else {
            let res = 0_u64.wrapping_sub(x) as i64;
            if res >= 0 {
                return Err(RBigIntError::Overflow);
            }
            Ok(res)
        }
    }

    pub fn fits_int(&self) -> bool {
        let n = self.numdigits();
        if n < MAX_DIGITS_THAT_CAN_FIT_IN_INT {
            return true;
        }
        if n > MAX_DIGITS_THAT_CAN_FIT_IN_INT {
            return false;
        }
        let Ok(x) = self._touint_helper() else {
            return false;
        };
        if self.get_sign() >= 0 {
            x as i64 >= 0
        } else {
            (0_u64.wrapping_sub(x) as i64) < 0
        }
    }

    #[majit_macros::jit_elidable]
    #[inline]
    pub fn tolonglong(&self) -> Result<i64, RBigIntError> {
        _AsLongLong(self)
    }

    #[inline]
    pub fn tobool(&self) -> bool {
        self.get_sign() != 0
    }

    #[majit_macros::jit_elidable]
    pub fn touint(&self) -> Result<u64, RBigIntError> {
        if self.get_sign() == -1 {
            return Err(RBigIntError::NegativeToUnsigned);
        }
        self._touint_helper()
    }

    #[majit_macros::jit_elidable]
    fn _touint_helper(&self) -> Result<u64, RBigIntError> {
        let mut x = 0_u64;
        let mut i = self.numdigits();
        while i > 0 {
            i -= 1;
            let previous = x;
            x = x.wrapping_shl(SHIFT as u32).wrapping_add(self.udigit(i));
            if (x >> SHIFT) != previous {
                return Err(RBigIntError::Overflow);
            }
        }
        Ok(x)
    }

    #[majit_macros::jit_elidable]
    #[inline]
    pub fn toulonglong(&self) -> Result<u64, RBigIntError> {
        if self.get_sign() == -1 {
            return Err(RBigIntError::NegativeToUnsigned);
        }
        _AsULonglong_ignore_sign(self)
    }

    #[majit_macros::jit_elidable]
    #[inline]
    pub fn uintmask(&self) -> u64 {
        self.ulonglongmask()
    }

    #[majit_macros::jit_elidable]
    pub fn ulonglongmask(&self) -> u64 {
        make_unsigned_mask_conversion(self)
    }

    #[majit_macros::jit_elidable]
    pub fn tofloat(&self) -> Result<f64, RBigIntError> {
        _AsDouble(self)
    }

    /// Render this value in `digits`' base, between `prefix` and `suffix`.
    ///
    /// A conversion long enough to exhaust the nursery collects part-way
    /// through, so a caller must not read an `&RBigInt` — this receiver
    /// included — that it obtained before the call. `self` is consumed into a
    /// by-value handle before the first collection point and is never read
    /// again here, so a caller that needs the receiver afterwards must
    /// re-derive it from whatever roots it.
    #[majit_macros::jit_elidable]
    pub fn format(
        &self,
        digits: &str,
        prefix: &str,
        suffix: &str,
        max_str_digits: i64,
    ) -> Result<String, RBigIntError> {
        _format(self, digits, prefix, suffix, max_str_digits)
    }

    #[majit_macros::jit_elidable]
    pub fn repr(&self) -> Result<String, RBigIntError> {
        match self.toint() {
            Ok(value) => Ok(format!("{value}L")),
            Err(RBigIntError::Overflow) => self.format(BASE10, "", "L", 0),
            Err(error) => Err(error),
        }
    }

    #[majit_macros::jit_elidable]
    pub fn str(&self, max_str_digits: i64) -> Result<String, RBigIntError> {
        match self.toint() {
            Ok(value) => Ok(value.to_string()),
            Err(RBigIntError::Overflow) => self.format(BASE10, "", "", max_str_digits),
            Err(error) => Err(error),
        }
    }

    #[majit_macros::jit_elidable]
    pub fn eq(&self, other: &Self) -> bool {
        if self.get_sign() != other.get_sign() || self.numdigits() != other.numdigits() {
            return false;
        }
        let mut i = 0;
        let ld = self.numdigits();
        while i < ld {
            if self.digit(i) != other.digit(i) {
                return false;
            }
            i += 1;
        }
        true
    }

    #[majit_macros::jit_elidable]
    pub fn int_eq(&self, iother: i64) -> bool {
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.eq(&other);
        }
        if self.numdigits() > 1 {
            return false;
        }
        self.get_sign() * self.digit(0) == iother
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }

    #[inline]
    pub fn int_ne(&self, iother: i64) -> bool {
        !self.int_eq(iother)
    }

    #[majit_macros::jit_elidable]
    pub fn lt(&self, other: &Self) -> bool {
        let selfsign = self.get_sign();
        let othersign = other.get_sign();
        if selfsign > othersign {
            return false;
        }
        if selfsign < othersign {
            return true;
        }
        let ld1 = self.numdigits();
        let ld2 = other.numdigits();
        if ld1 > ld2 {
            return othersign <= 0;
        } else if ld1 < ld2 {
            return othersign > 0;
        }
        let mut i = ld1;
        while i > 0 {
            i -= 1;
            let d1 = self.digit(i);
            let d2 = other.digit(i);
            if d1 < d2 {
                return othersign > 0;
            } else if d1 > d2 {
                return othersign <= 0;
            }
        }
        false
    }

    #[majit_macros::jit_elidable]
    pub fn int_lt(&self, iother: i64) -> bool {
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.lt(&other);
        }
        _x_int_lt(self, iother, false)
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        !other.lt(self)
    }

    #[inline]
    pub fn int_le(&self, iother: i64) -> bool {
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.le(&other);
        }
        _x_int_lt(self, iother, true)
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        other.lt(self)
    }

    #[inline]
    pub fn int_gt(&self, iother: i64) -> bool {
        !self.int_le(iother)
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        !self.lt(other)
    }

    #[inline]
    pub fn int_ge(&self, iother: i64) -> bool {
        !self.int_lt(iother)
    }

    #[majit_macros::jit_elidable]
    pub fn hash(&self) -> i64 {
        _hash(self)
    }

    #[majit_macros::jit_elidable]
    pub fn add(&self, other: &Self) -> Self {
        let selfsign = self.get_sign();
        let othersign = other.get_sign();
        if selfsign == 0 {
            return other.translated_alias();
        }
        if othersign == 0 {
            return self.translated_alias();
        }
        let mut result = if selfsign == othersign {
            _x_add(self, other)
        } else {
            _x_sub(other, self)
        };
        result._set_sign(result.get_sign() * othersign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn int_add(&self, iother: i64) -> Self {
        let selfsign = self.get_sign();
        if selfsign == 0 {
            return Self::fromint(iother);
        }
        if iother == 0 {
            return self.translated_alias();
        }
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.add(&other);
        }

        let othersign = intsign(iother);
        let mut result = if selfsign == othersign {
            _x_int_add(self, iother)
        } else {
            let mut result = _x_int_sub(self, iother);
            result._set_sign(-result.get_sign());
            result
        };
        result._set_sign(result.get_sign() * othersign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn add_int_int_bigint_result(iself: i64, iother: i64) -> Self {
        if !int_in_valid_range(iself) || !int_in_valid_range(iother) {
            return Self::fromint(iself).int_add(iother);
        }
        Self::_add_int_int_helper(iself, iother)
    }

    fn _add_int_int_helper(iself: i64, iother: i64) -> Self {
        if iself == 0 {
            return Self::fromint(iother);
        }
        if iother == 0 {
            return Self::fromint(iself);
        }
        let sign1 = intsign(iself);
        let sign2 = intsign(iother);
        let v1 = iself.unsigned_abs();
        let v2 = iother.unsigned_abs();
        let (sign, ures) = if sign1 == sign2 {
            (sign1, v1 as UWideDigit + v2 as UWideDigit)
        } else if v1 >= v2 {
            (sign1, (v1 - v2) as UWideDigit)
        } else {
            (sign2, (v2 - v1) as UWideDigit)
        };
        if ures == 0 {
            return Self::zero();
        }
        let carry = ures >> SHIFT;
        if carry != 0 {
            Self::new(
                &[
                    _store_uwidedigit(ures & MASK as UWideDigit),
                    _store_uwidedigit(carry),
                ],
                sign,
                2,
            )
        } else {
            Self::new(&[_store_uwidedigit(ures & MASK as UWideDigit)], sign, 1)
        }
    }

    #[majit_macros::jit_elidable]
    pub fn sub(&self, other: &Self) -> Self {
        let selfsign = self.get_sign();
        let othersign = other.get_sign();
        if othersign == 0 {
            return self.translated_alias();
        }
        if selfsign == 0 {
            let other = live_rbigint(other);
            return Self::new_from_rooted(
                &other,
                0,
                other.numdigits() as usize,
                -othersign,
                other.numdigits(),
            );
        }
        let mut result = if selfsign == othersign {
            _x_sub(self, other)
        } else {
            _x_add(self, other)
        };
        result._set_sign(result.get_sign() * selfsign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn int_sub(&self, iother: i64) -> Self {
        let selfsign = self.get_sign();
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.sub(&other);
        }
        if iother == 0 {
            return self.translated_alias();
        }
        if selfsign == 0 {
            return Self::fromint(-iother);
        }
        let mut result = if selfsign == intsign(iother) {
            _x_int_sub(self, iother)
        } else {
            _x_int_add(self, iother)
        };
        result._set_sign(result.get_sign() * selfsign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn sub_int_int_bigint_result(iself: i64, iother: i64) -> Self {
        if !int_in_valid_range(iself) || !int_in_valid_range(iother) {
            return Self::fromint(iself).int_sub(iother);
        }
        Self::_add_int_int_helper(iself, -iother)
    }

    #[majit_macros::jit_elidable]
    pub fn mul(&self, other: &Self) -> Self {
        let mut this = self;
        let mut that = other;
        let mut selfsize = this.numdigits();
        let mut othersize = that.numdigits();
        let selfsign = this.get_sign();
        let othersign = that.get_sign();

        if selfsize > othersize {
            std::mem::swap(&mut this, &mut that);
            std::mem::swap(&mut selfsize, &mut othersize);
        }
        if selfsign == 0 || othersign == 0 {
            return Self::zero();
        }

        let mut result;
        if selfsize == 1 {
            if this.digit(0) == ONEDIGIT {
                let that = live_rbigint(that);
                return Self::new_from_rooted(
                    &that,
                    0,
                    othersize as usize,
                    selfsign * othersign,
                    othersize,
                );
            } else if othersize == 1 {
                let res = that.uwidedigit(0) * this.udigit(0) as UWideDigit;
                let carry = res >> SHIFT;
                if carry != 0 {
                    return Self::new(
                        &[
                            _store_uwidedigit(res & MASK as UWideDigit),
                            _store_uwidedigit(carry),
                        ],
                        selfsign * othersign,
                        2,
                    );
                }
                return Self::new(
                    &[_store_uwidedigit(res & MASK as UWideDigit)],
                    selfsign * othersign,
                    1,
                );
            }
            result = _x_mul(this, that, this.digit(0));
        } else if USE_KARATSUBA {
            let cutoff = if std::ptr::eq(this, that) {
                KARATSUBA_SQUARE_CUTOFF
            } else {
                KARATSUBA_CUTOFF
            };
            if selfsize <= cutoff {
                result = _x_mul(this, that, 0);
            } else {
                result = _k_mul(this, that);
            }
        } else {
            result = _x_mul(this, that, 0);
        }
        result._set_sign(selfsign * othersign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn int_mul(&self, iother: i64) -> Self {
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.mul(&other);
        }
        let selfsign = self.get_sign();
        if selfsign == 0 || iother == 0 {
            return Self::zero();
        }
        let asize = self.numdigits();
        let digit = iother.abs();
        let othersign = intsign(iother);
        if digit == 1 {
            if othersign == 1 {
                return self.translated_alias();
            }
            let this = live_rbigint(self);
            return Self::new_from_rooted(&this, 0, asize as usize, selfsign * othersign, asize);
        } else if asize == 1 {
            let res = self.uwidedigit(0) * digit as UWideDigit;
            let carry = res >> SHIFT;
            if carry != 0 {
                return Self::new(
                    &[
                        _store_uwidedigit(res & MASK as UWideDigit),
                        _store_uwidedigit(carry),
                    ],
                    selfsign * othersign,
                    2,
                );
            }
            return Self::new(
                &[_store_uwidedigit(res & MASK as UWideDigit)],
                selfsign * othersign,
                1,
            );
        }

        let mut result = if digit & (digit - 1) == 0 {
            self.lqshift(PTWOTABLE[digit.trailing_zeros() as usize])
        } else {
            _muladd1(self, digit, 0)
        };
        result._set_sign(selfsign * othersign);
        result
    }

    #[majit_macros::jit_elidable]
    pub fn mul_int_int_bigint_result(iself: i64, iother: i64) -> Self {
        if !int_in_valid_range(iself) {
            return Self::fromint(iself).int_mul(iother);
        }
        if iself == 0 || iother == 0 {
            return Self::zero();
        }
        let selfsign = intsign(iself);
        let othersign = intsign(iother);
        let res = iself.unsigned_abs() as UWideDigit * iother.unsigned_abs() as UWideDigit;
        let carry = res >> SHIFT;
        if carry != 0 {
            Self::new(
                &[
                    _store_uwidedigit(res & MASK as UWideDigit),
                    _store_uwidedigit(carry),
                ],
                selfsign * othersign,
                2,
            )
        } else {
            Self::new(
                &[_store_uwidedigit(res & MASK as UWideDigit)],
                selfsign * othersign,
                1,
            )
        }
    }

    #[majit_macros::jit_elidable]
    pub fn truediv(&self, other: &Self) -> Result<f64, RBigIntError> {
        _bigint_true_divide(self, other)
    }

    pub fn floordiv(&self, other: &Self) -> Result<Self, RBigIntError> {
        let (div, _) = self.divmod(other)?;
        Ok(div)
    }

    #[inline]
    pub fn div(&self, other: &Self) -> Result<Self, RBigIntError> {
        self.floordiv(other)
    }

    #[majit_macros::jit_elidable]
    pub fn int_floordiv(&self, iother: i64) -> Result<Self, RBigIntError> {
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.floordiv(&other);
        }
        if iother == 0 {
            return Err(RBigIntError::DivisionByZero);
        }
        let digit = iother.abs();
        let selfsign = self.get_sign();
        if selfsign == 1 && iother > 0 {
            if digit == 1 {
                return Ok(self.translated_alias());
            } else if digit & (digit - 1) == 0 {
                return Ok(self.rqshift(PTWOTABLE[digit.trailing_zeros() as usize]));
            }
        }
        let (mut div, rem) = _divrem1(self, digit);
        let othersign = intsign(iother);
        if rem != 0 && selfsign * othersign == -1 {
            if div.get_sign() == 0 {
                return Ok(Self::negative_one());
            }
            div = div.int_add(1);
        }
        div._set_sign(selfsign * othersign);
        div._normalize();
        Ok(div)
    }

    #[inline]
    pub fn int_div(&self, iother: i64) -> Result<Self, RBigIntError> {
        self.int_floordiv(iother)
    }

    pub fn r#mod(&self, other: &Self) -> Result<Self, RBigIntError> {
        let (_, modulo) = self.divmod(other)?;
        Ok(modulo)
    }

    #[majit_macros::jit_elidable]
    pub fn int_mod(&self, iother: i64) -> Result<Self, RBigIntError> {
        if iother == 0 {
            return Err(RBigIntError::DivisionByZero);
        }
        let selfsign = self.get_sign();
        if selfsign == 0 {
            return Ok(Self::zero());
        }
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.r#mod(&other);
        }

        let digit = iother.abs();
        let mut modulo = if digit == 1 {
            return Ok(Self::zero());
        } else if digit == 2 {
            let modm = self.digit(0) & 1;
            if modm != 0 {
                return Ok(if iother < 0 {
                    Self::negative_one()
                } else {
                    Self::one()
                });
            }
            return Ok(Self::zero());
        } else if digit & (digit - 1) == 0 {
            self.int_and_(digit - 1)
        } else {
            let rem = _int_rem_core(self, digit);
            if rem == 0 {
                return Ok(Self::zero());
            }
            Self::new(&[rem as Digit], if selfsign < 0 { -1 } else { 1 }, 1)
        };
        if modulo.get_sign() * intsign(iother) == -1 {
            modulo = modulo.int_add(iother);
        }
        Ok(modulo)
    }

    #[majit_macros::jit_elidable]
    pub fn int_mod_int_result(&self, iother: i64) -> Result<i64, RBigIntError> {
        if iother == 0 {
            return Err(RBigIntError::DivisionByZero);
        }
        let selfsign = self.get_sign();
        if selfsign == 0 {
            return Ok(0);
        }
        if !int_in_valid_range(iother) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.r#mod(&other)?.toint();
        }

        let digit = iother.abs();
        let mut modulo = if digit == 1 {
            0
        } else if digit == 2 {
            let modm = self.digit(0) & 1;
            if modm != 0 {
                if iother < 0 { -1 } else { 1 }
            } else {
                0
            }
        } else if digit & (digit - 1) == 0 {
            self.int_and_(digit - 1).toint()?
        } else {
            _int_rem_core(self, digit) as i64 * selfsign
        };
        if intsign(modulo) * intsign(iother) == -1 {
            modulo += iother;
        }
        Ok(modulo)
    }

    #[majit_macros::jit_elidable]
    pub fn divmod(&self, other: &Self) -> Result<(Self, Self), RBigIntError> {
        let selfsign = self.get_sign();
        let othersign = other.get_sign();
        if othersign == 0 {
            return Err(RBigIntError::DivisionByZero);
        }
        if selfsign == 0 {
            return Ok((Self::zero(), Self::zero()));
        }
        let this = live_rbigint(self);
        let other = live_rbigint(other);
        if other.numdigits() == 1 && !(othersign == -1 && selfsign != othersign) {
            let otherint = other.digit(0) * othersign;
            debug_assert!(int_in_valid_range(otherint));
            return this.int_divmod(otherint);
        }
        let div_limit = holder_limit(&HOLDER.DIV_LIMIT);
        if this.numdigits() * 10 > other.numdigits() * 12 && other.numdigits() > div_limit * 2 {
            let (div, modulo) = divmod_big(&this, &other)?;
            // The check's `mul` and `add` collect.
            let div = RBigIntGcRoot::new(div);
            let modulo = RBigIntGcRoot::new(modulo);
            debug_assert!(div.mul(&other).add(&modulo).eq(&this));
            return Ok((div.translated_alias(), modulo.translated_alias()));
        }
        this._divmod_small(&other)
    }

    fn _divmod_small(&self, other: &Self) -> Result<(Self, Self), RBigIntError> {
        let this = live_rbigint(self);
        let other = live_rbigint(other);
        let (div, modulo) = _divrem(&this, &other)?;
        let mut div = RBigIntGcRoot::new(div);
        let mut modulo = RBigIntGcRoot::new(modulo);
        if modulo.get_sign() * other.get_sign() == -1 {
            modulo = RBigIntGcRoot::new(modulo.add(&other));
            if div.get_sign() == 0 {
                return Ok((Self::negative_one(), (*modulo).clone()));
            }
            div = RBigIntGcRoot::new(div.int_sub(1));
        }
        Ok(((*div).clone(), (*modulo).clone()))
    }

    #[majit_macros::jit_elidable]
    pub fn int_divmod(&self, iother: i64) -> Result<(Self, Self), RBigIntError> {
        if iother == 0 {
            return Err(RBigIntError::DivisionByZero);
        }
        let selfsign = self.get_sign();
        let othersign = intsign(iother);
        if !int_in_valid_range(iother) || (othersign == -1 && selfsign != othersign) {
            let this = live_rbigint(self);
            let other = Self::fromint(iother);
            return this.divmod(&other);
        }
        let digit = iother.abs();
        let (div, rem) = _divrem1(self, digit);
        let mut div = RBigIntGcRoot::new(div);
        if div._size != 0 {
            div._set_sign(selfsign * othersign);
        }
        let mut rem = rem;
        if selfsign < 0 {
            rem = -rem;
        }
        if rem != 0 && selfsign * othersign == -1 {
            rem += iother;
            if div.get_sign() == 0 {
                div = RBigIntGcRoot::new(Self::negative_one());
            } else {
                div = RBigIntGcRoot::new(div.int_sub(1));
            }
        }
        let rem = Self::fromint(rem);
        Ok(((*div).clone(), rem))
    }

    #[majit_macros::jit_elidable]
    pub fn pow(&self, other: &Self, modulus: Option<&Self>) -> Result<Self, RBigIntError> {
        let selfsign = self.get_sign();
        let othersign = other.get_sign();
        if othersign < 0 {
            return Err(if modulus.is_some() {
                RBigIntError::NegativeExponentWithModulus
            } else {
                RBigIntError::NegativeExponent
            });
        }

        // Root every handle this graph rereads after `neg` / `r#mod` /
        // `_help_mult`. Those are `Digits::new` (`collect_and_reserve`).
        let this = live_rbigint(self);
        let other = live_rbigint(other);

        let mut negative_output = false;
        let modulus_root = if let Some(modulus) = modulus {
            let modulussign = modulus.get_sign();
            if modulussign == 0 {
                return Err(RBigIntError::ZeroModulus);
            }
            Some(if modulussign < 0 {
                negative_output = true;
                RBigIntGcRoot::new(modulus.neg())
            } else {
                live_rbigint(modulus)
            })
        } else {
            None
        };
        if let Some(modulus) = modulus_root.as_ref() {
            if modulus.numdigits() == 1 && modulus.digit(0) == ONEDIGIT {
                return Ok(Self::zero());
            }
        } else if othersign == 0 {
            return Ok(Self::one());
        } else if selfsign == 0 {
            return Ok(Self::zero());
        }

        // Keep `base` as a guard, not `&*this`. A `&RBigInt` taken out of the
        // cell is not reloaded after the next collecting `_help_mult`.
        let base_owned = if let Some(modulus) = modulus_root.as_ref() {
            if selfsign < 0 || this.numdigits() > modulus.numdigits() {
                Some(RBigIntGcRoot::new(this.r#mod(modulus)?))
            } else {
                None
            }
        } else {
            None
        };
        let base: &RBigIntGcRoot = base_owned.as_ref().unwrap_or(&this);

        let mut size_b = other.numdigits();
        if modulus_root.is_none() && size_b == 1 {
            let digit = other.digit(0);
            if digit == ONEDIGIT {
                return Ok(base.translated_alias());
            } else if base.numdigits() == 1 {
                let adigit = base.udigit(0);
                if adigit == 1 {
                    return Ok(if selfsign == -1 && digit & 1 != 0 {
                        Self::negative_one()
                    } else {
                        Self::one()
                    });
                } else if adigit & (adigit - 1) == 0 {
                    let exponent_minus_one = digit as u64 - 1;
                    let shift = exponent_minus_one
                        .checked_mul((PTWOTABLE[adigit.trailing_zeros() as usize] - 1) as u64)
                        .and_then(|value| value.checked_add(exponent_minus_one))
                        .and_then(|value| i64::try_from(value).ok())
                        .ok_or(RBigIntError::Memory)?;
                    let mut ret = base.lshift(shift)?;
                    if selfsign == -1 && digit & 1 == 0 {
                        ret._set_sign(1);
                    }
                    return Ok(ret);
                }
            }
        }

        let mut z = RBigIntGcRoot::new(Self::one());
        if size_b <= FIVEARY_CUTOFF {
            while size_b > 0 {
                size_b -= 1;
                let bi = other.udigit(size_b);
                let mut j = 1_u64 << (SHIFT - 1);
                while j != 0 {
                    z = RBigIntGcRoot::new(_help_mult(&z, &z, modulus_root.as_deref())?);
                    if bi & j != 0 {
                        z = RBigIntGcRoot::new(_help_mult(&z, base, modulus_root.as_deref())?);
                    }
                    j >>= 1;
                }
            }
        } else {
            let mut table: [RBigIntGcRoot; 32] =
                std::array::from_fn(|_| RBigIntGcRoot::new(Self::one()));
            let mut i = 1;
            while i < 32 {
                table[i] =
                    RBigIntGcRoot::new(_help_mult(&table[i - 1], base, modulus_root.as_deref())?);
                i += 1;
            }
            const JMAPPING: [i32; 5] = [0, 2, 4, 1, 3];
            let mut j = JMAPPING[(size_b % 5) as usize];
            let mut accum = 0_u64;
            loop {
                j -= 5;
                let index;
                if j >= 0 {
                    index = (accum >> j) & 0x1f;
                } else {
                    if size_b == 0 {
                        break;
                    }
                    size_b -= 1;
                    let bi = other.udigit(size_b);
                    index = ((accum << (-j)) | (bi >> (j + SHIFT as i32))) & 0x1f;
                    accum = bi;
                    j += SHIFT as i32;
                }
                let mut k = 0;
                while k < 5 {
                    z = RBigIntGcRoot::new(_help_mult(&z, &z, modulus_root.as_deref())?);
                    k += 1;
                }
                if index != 0 {
                    z = RBigIntGcRoot::new(_help_mult(
                        &z,
                        &table[index as usize],
                        modulus_root.as_deref(),
                    )?);
                }
            }
            debug_assert_eq!(j, -5);
        }
        if negative_output && z.get_sign() != 0 {
            z = RBigIntGcRoot::new(
                z.sub(
                    modulus_root
                        .as_ref()
                        .expect("negative modulus was normalized"),
                ),
            );
        }
        Ok(z.translated_alias())
    }

    #[majit_macros::jit_elidable]
    pub fn int_pow(&self, iother: i64, modulus: Option<&Self>) -> Result<Self, RBigIntError> {
        let mut negative_output = false;
        if iother < 0 {
            return Err(if modulus.is_some() {
                RBigIntError::NegativeExponentWithModulus
            } else {
                RBigIntError::NegativeExponent
            });
        }

        let selfsign = self.get_sign();
        debug_assert!(iother >= 0);
        let this = live_rbigint(self);
        let modulus_root = if let Some(modulus) = modulus {
            let modulussign = modulus.get_sign();
            if modulussign == 0 {
                return Err(RBigIntError::ZeroModulus);
            }
            Some(if modulussign < 0 {
                negative_output = true;
                RBigIntGcRoot::new(modulus.neg())
            } else {
                live_rbigint(modulus)
            })
        } else {
            None
        };

        if let Some(modulus) = modulus_root.as_ref() {
            if modulus.numdigits() == 1 && modulus.digit(0) == ONEDIGIT {
                return Ok(Self::zero());
            }
        } else if iother == 0 {
            return Ok(Self::one());
        } else if selfsign == 0 {
            return Ok(Self::zero());
        } else if iother == 1 {
            return Ok(this.translated_alias());
        } else if this.numdigits() == 1 {
            let adigit = this.udigit(0);
            if adigit == 1 {
                return Ok(if selfsign == -1 && iother & 1 != 0 {
                    Self::negative_one()
                } else {
                    Self::one()
                });
            } else if adigit & (adigit - 1) == 0 {
                let exponent_minus_one =
                    u64::try_from(iother - 1).expect("non-negative exponent was checked above");
                let shift = exponent_minus_one
                    .checked_mul((PTWOTABLE[adigit.trailing_zeros() as usize] - 1) as u64)
                    .and_then(|value| value.checked_add(exponent_minus_one))
                    .and_then(|value| i64::try_from(value).ok())
                    .ok_or(RBigIntError::Memory)?;
                let mut ret = this.lshift(shift)?;
                if selfsign == -1 && iother & 1 == 0 {
                    ret._set_sign(1);
                }
                return Ok(ret);
            }
        }

        let base_owned = if let Some(modulus) = modulus_root.as_ref() {
            if selfsign < 0 || this.numdigits() > modulus.numdigits() {
                Some(RBigIntGcRoot::new(this.r#mod(modulus)?))
            } else {
                None
            }
        } else {
            None
        };
        let base: &RBigIntGcRoot = base_owned.as_ref().unwrap_or(&this);

        let mut z = RBigIntGcRoot::new(Self::one());
        let mut j = 1_i64 << 62;
        while j != 0 {
            z = RBigIntGcRoot::new(_help_mult(&z, &z, modulus_root.as_deref())?);
            if iother & j != 0 {
                z = RBigIntGcRoot::new(_help_mult(&z, base, modulus_root.as_deref())?);
            }
            j >>= 1;
        }

        if negative_output && z.get_sign() != 0 {
            z = RBigIntGcRoot::new(
                z.sub(
                    modulus_root
                        .as_ref()
                        .expect("negative modulus was normalized"),
                ),
            );
        }
        Ok(z.translated_alias())
    }

    #[majit_macros::jit_elidable]
    pub fn neg(&self) -> Self {
        let mut result = self.clone();
        result._set_sign(-self.get_sign());
        result
    }

    #[majit_macros::jit_elidable]
    pub fn abs(&self) -> Self {
        if self.get_sign() != -1 {
            self.translated_alias()
        } else {
            self.neg()
        }
    }

    #[majit_macros::jit_elidable]
    pub fn invert(&self) -> Self {
        if self.get_sign() == 0 {
            return Self::negative_one();
        }
        let mut ret = self.int_add(1);
        ret._set_sign(-ret.get_sign());
        ret
    }

    #[majit_macros::jit_elidable]
    #[majit_macros::always_inline]
    #[inline]
    pub fn lshift(&self, int_other: i64) -> Result<Self, RBigIntError> {
        let selfsign = self.get_sign();
        if int_other < 0 {
            return Err(RBigIntError::NegativeShift);
        } else if int_other == 0 || selfsign == 0 {
            return Ok(self.translated_alias());
        }
        let mut wordshift = int_other / SHIFT;
        let remshift = int_other % SHIFT;
        let this = live_rbigint(self);
        if remshift == 0 {
            let newsize = this
                .numdigits()
                .checked_add(wordshift)
                .ok_or(RBigIntError::Memory)?;
            let mut result = Self::try_with_size(newsize, selfsign)?;
            // Past the last collecting call: read digits through a plain handle.
            let this: &RBigInt = &this;
            let mut i = 0;
            while i < this.numdigits() {
                result.setdigit(wordshift + i, this.digit(i));
                i += 1;
            }
            return Ok(result);
        }

        let hishift = SHIFT - remshift;
        let oldsize = this.numdigits();
        let mut newsize = oldsize
            .checked_add(wordshift)
            .and_then(|size| size.checked_add(1))
            .ok_or(RBigIntError::Memory)?;
        let mut z = Self::try_with_size(newsize, selfsign)?;
        // Past the last collecting call: read digits through a plain handle.
        let this: &RBigInt = &this;
        let mut j = 0;
        let mut prevdigit = 0_u64;
        while j < oldsize {
            let digit = this.udigit(j);
            let newdigit = (digit << remshift) | (prevdigit >> hishift);
            z.setdigit_udigit(wordshift, newdigit);
            prevdigit = digit;
            wordshift += 1;
            j += 1;
        }
        newsize -= 1;
        z.setdigit_udigit(newsize, prevdigit >> hishift);
        z._normalize();
        Ok(z)
    }

    /// rbigint.py `lqshift`.
    #[majit_macros::jit_elidable]
    #[majit_macros::always_inline]
    #[inline]
    pub fn lqshift(&self, int_other: i64) -> Self {
        debug_assert!(int_other > 0 && int_other < SHIFT);
        let this = live_rbigint(self);
        let oldsize = this.numdigits();
        let selfsign = this.get_sign();
        let mut z = Self::with_size(oldsize + 1, selfsign);
        // Past the last collecting call: read digits through a plain handle.
        let this: &RBigInt = &this;
        let hishift = SHIFT - int_other;
        let mut prevdigit = 0_u64;
        let mut i = 0;
        while i < oldsize {
            let digit = this.udigit(i);
            let newdigit = (digit << int_other) | (prevdigit >> hishift);
            z.setdigit_udigit(i, newdigit);
            prevdigit = digit;
            i += 1;
        }
        z.setdigit_udigit(oldsize, prevdigit >> hishift);
        z._normalize();
        z
    }

    #[majit_macros::jit_elidable]
    pub fn lshift_int_int_bigint_result(iself: i64, int_other: i64) -> Result<Self, RBigIntError> {
        if !int_in_valid_range(iself) {
            return Self::fromint(iself).lshift(int_other);
        }
        if int_other < 0 {
            return Err(RBigIntError::NegativeShift);
        }
        if iself == 0 {
            return Ok(Self::zero());
        }
        if int_other == 0 {
            return Ok(Self::fromint(iself));
        }

        let selfsign = intsign(iself);
        let wordshift = int_other / SHIFT;
        let remshift = int_other % SHIFT;
        let ival = iself.unsigned_abs();
        if remshift != 0 {
            let hishift = SHIFT - remshift;
            let lowerdigit = ival << remshift;
            let upperdigit = ival >> hishift;
            if upperdigit != 0 {
                let size = wordshift.checked_add(2).ok_or(RBigIntError::Memory)?;
                let mut result = Self::try_with_size(size, selfsign)?;
                result.setdigit_udigit(wordshift, lowerdigit);
                result.setdigit_udigit(wordshift + 1, upperdigit);
                return Ok(result);
            }
            let size = wordshift.checked_add(1).ok_or(RBigIntError::Memory)?;
            let mut result = Self::try_with_size(size, selfsign)?;
            result.setdigit_udigit(wordshift, lowerdigit);
            return Ok(result);
        }
        let size = wordshift.checked_add(1).ok_or(RBigIntError::Memory)?;
        let mut result = Self::try_with_size(size, selfsign)?;
        result.setdigit_udigit(wordshift, ival);
        Ok(result)
    }

    #[majit_macros::jit_elidable]
    #[majit_macros::always_inline_try]
    pub fn rshift(&self, int_other: i64, dont_invert: bool) -> Result<Self, RBigIntError> {
        if int_other < 0 {
            return Err(RBigIntError::NegativeShift);
        } else if int_other == 0 {
            return Ok(self.translated_alias());
        }
        let this = live_rbigint(self);
        let selfsign = this.get_sign();
        if selfsign == -1 && !dont_invert {
            let a = this.invert().rshift(int_other, false)?;
            return Ok(a.invert());
        }

        let mut wordshift = int_other / SHIFT;
        let newsize = this.numdigits() - wordshift;
        if newsize <= 0 {
            return Ok(Self::zero());
        }

        let loshift = int_other % SHIFT;
        let hishift = SHIFT - loshift;
        let mut z = Self::with_size(newsize, selfsign);
        // Past the last collecting call: read digits through a plain handle.
        let this: &RBigInt = &this;
        let mut i = 0;
        while i < newsize {
            let mut newdigit = this.digit(wordshift) >> loshift;
            if i + 1 < newsize {
                newdigit |= this.digit(wordshift + 1).wrapping_shl(hishift as u32);
            }
            z.setdigit(i, newdigit);
            i += 1;
            wordshift += 1;
        }
        z._normalize();
        Ok(z)
    }

    #[majit_macros::jit_elidable]
    pub fn rqshift(&self, int_other: i64) -> Self {
        debug_assert!(int_other >= 0);
        let this = live_rbigint(self);
        let mut wordshift = int_other / SHIFT;
        let loshift = int_other % SHIFT;
        let newsize = this.numdigits() - wordshift;
        if newsize <= 0 {
            return Self::zero();
        }
        let hishift = SHIFT - loshift;
        let selfsign = this.get_sign();
        let mut z = Self::with_size(newsize, selfsign);
        // Past the last collecting call: read digits through a plain handle.
        let this: &RBigInt = &this;
        let mut i = 0;
        while i < newsize {
            let digit = this.udigit(wordshift);
            let mut newdigit = digit >> loshift;
            if i + 1 < newsize {
                newdigit |= this.udigit(wordshift + 1) << hishift;
            }
            z.setdigit_udigit(i, newdigit);
            i += 1;
            wordshift += 1;
        }
        z._normalize();
        z
    }

    #[majit_macros::jit_elidable]
    pub fn abs_rshift_and_mask(&self, bigshiftcount: u64, mask: i64) -> i64 {
        debug_assert!(mask >= 0);
        let wordshift = bigshiftcount / SHIFT as u64;
        let numdigits = self.numdigits() as u64;
        if wordshift >= numdigits {
            return 0;
        }
        let wordshift = wordshift as i64;
        let loshift = (bigshiftcount - wordshift as u64 * SHIFT as u64) as i64;
        let mut lastdigit = self.digit(wordshift) >> loshift;
        if mask > (MASK as Digit >> loshift) && wordshift + 1 < self.numdigits() {
            let hishift = SHIFT - loshift;
            lastdigit |= self.digit(wordshift + 1).wrapping_shl(hishift as u32);
        }
        lastdigit & mask
    }

    pub fn from_list_n_bits(list: &[i64], nbits: i64) -> Result<Self, RBigIntError> {
        if list.is_empty() {
            return Ok(Self::zero());
        }
        let mut z;
        if nbits == SHIFT {
            z = Self::with_size(list.len() as i64, 1);
            let mut i = 0;
            while i < list.len() as i64 {
                z.setdigit(i, list[i as usize]);
                i += 1;
            }
        } else {
            if !(1..SHIFT).contains(&nbits) {
                return Err(RBigIntError::InvalidBitWidth);
            }
            let length = list.len() as i64 * nbits / SHIFT + 1;
            z = Self::with_size(length, 1);
            let mut out = 0;
            let mut i = 0_i64;
            let mut accum = 0_u128;
            for &input in list {
                accum |= (input as u128) << i as u32;
                let original_i = i;
                i += nbits;
                if i > SHIFT {
                    z.setdigit_uwidedigit(out, accum);
                    out += 1;
                    accum = (input as u128) >> (SHIFT - original_i) as u32;
                    i -= SHIFT;
                }
            }
            debug_assert!(out < length);
            z.setdigit_uwidedigit(out, accum);
        }
        z._normalize();
        Ok(z)
    }

    #[majit_macros::jit_elidable]
    pub fn and_(&self, other: &Self) -> Self {
        _bitwise_and(self, other)
    }

    #[majit_macros::jit_elidable]
    pub fn int_and_(&self, iother: i64) -> Self {
        _int_bitwise_and(self, iother)
    }

    #[majit_macros::jit_elidable]
    pub fn xor(&self, other: &Self) -> Self {
        _bitwise_xor(self, other)
    }

    #[majit_macros::jit_elidable]
    pub fn int_xor(&self, iother: i64) -> Self {
        _int_bitwise_xor(self, iother)
    }

    #[majit_macros::jit_elidable]
    pub fn or_(&self, other: &Self) -> Self {
        _bitwise_or(self, other)
    }

    #[majit_macros::jit_elidable]
    pub fn int_or_(&self, iother: i64) -> Self {
        _int_bitwise_or(self, iother)
    }

    #[majit_macros::jit_elidable]
    pub fn oct(&self) -> Result<String, RBigIntError> {
        if self.get_sign() == 0 {
            Ok("0L".to_string())
        } else {
            self.format(BASE8, "0", "L", 0)
        }
    }

    #[majit_macros::jit_elidable]
    pub fn hex(&self) -> Result<String, RBigIntError> {
        self.format(BASE16, "0x", "L", 0)
    }

    /// rbigint.py `log`.
    #[majit_macros::jit_elidable]
    pub fn log(&self, base: f64) -> Result<f64, RBigIntError> {
        if base == 10.0 {
            return _loghelper_log10(self);
        }
        if base == 2.0 {
            return _loghelper_log2(self);
        }
        let mut result = _loghelper_ln(self)?;
        if base != 0.0 {
            result /= base.ln();
        }
        Ok(result)
    }

    /// rbigint.py `tolong`; Python-only upstream helper, represented
    /// with Rust's widest built-in signed integer for port tests.
    #[majit_macros::not_rpython]
    pub fn tolong(&self) -> Result<i128, RBigIntError> {
        let mut value = 0_u128;
        let mut i = self.numdigits();
        while i > 0 {
            i -= 1;
            let digit = self.udigit(i) as u128;
            if value > (u128::MAX - digit) >> SHIFT {
                return Err(RBigIntError::Overflow);
            }
            value = (value << SHIFT) + digit;
        }
        if self.get_sign() >= 0 {
            i128::try_from(value).map_err(|_| RBigIntError::Overflow)
        } else if value == 1_u128 << 127 {
            Ok(i128::MIN)
        } else {
            i128::try_from(value)
                .map(|value| -value)
                .map_err(|_| RBigIntError::Overflow)
        }
    }

    /// rbigint.py `_normalize`.
    #[majit_macros::always_inline]
    #[inline]
    pub fn _normalize(&mut self) {
        let mut i = self.numdigits();
        while i > 1 && self.digit(i - 1) == NULLDIGIT {
            i -= 1;
        }
        debug_assert!(i > 0);
        self._size = i * self.get_sign();
        if i == 1 && self.digit(0) == NULLDIGIT {
            self._size = 0;
            self._digits = Self::zero()._digits;
        }
    }

    #[majit_macros::jit_elidable]
    pub fn bit_length(&self) -> Result<i64, RBigIntError> {
        let i = self.numdigits();
        if i == 1 && self.digit(0) == 0 {
            return Ok(0);
        }
        let msd = self.digit(i - 1);
        let msd_bits = bits_in_digit(msd);
        let bits = (i - 1).checked_mul(SHIFT).ok_or(RBigIntError::Overflow)? + msd_bits;
        Ok(bits)
    }

    #[majit_macros::jit_elidable]
    pub fn bit_count(&self) -> Result<i64, RBigIntError> {
        let mut result = 0_i64;
        let mut i = 0;
        while i < self.numdigits() {
            result = result
                .checked_add(bit_count_digit(self.digit(i)))
                .ok_or(RBigIntError::Overflow)?;
            i += 1;
        }
        Ok(result)
    }

    pub fn gcd(&self, other: &Self) -> Result<Self, RBigIntError> {
        let this = live_rbigint(self);
        let other = live_rbigint(other);
        gcd_lehmer(this.abs(), other.abs())
    }

    pub fn isqrt(&self) -> Result<Self, RBigIntError> {
        let this = live_rbigint(self);
        if this.int_lt(0) {
            return Err(RBigIntError::NegativeSquareRoot);
        }
        if this.int_eq(0) {
            return Ok(Self::zero());
        }
        let c = (this.bit_length()? - 1) / 2;
        let mut a = RBigIntGcRoot::new(Self::one());
        let mut d = 0;
        let top = bits_in_digit(c as Digit);
        let mut s = top;
        while s > 0 {
            s -= 1;
            let e = d;
            d = c >> s;
            let shifted_a = RBigIntGcRoot::new(a.lshift(d - e - 1)?);
            let shifted_self = RBigIntGcRoot::new(this.rshift(2 * c - e - d + 1, false)?);
            let quot = RBigIntGcRoot::new(shifted_self.floordiv(&a)?);
            a = RBigIntGcRoot::new(shifted_a.add(&quot));
        }
        let square = RBigIntGcRoot::new(a.mul(&a));
        Ok(a.int_sub(square.gt(&this) as i64))
    }
}

/// Rust ecosystem compatibility surface kept outside the line-by-line
/// `rbigint` method block above. These adapters replaced Malachite/num-bigint
/// call sites during the port; separating them preserves the exact upstream
/// method order for source auditing without changing their native ABI.
impl RBigInt {
    pub fn from_u128(value: u128) -> Self {
        if value == 0 {
            return Self::zero();
        }
        let digits = digits_from_nonneg_u128(value);
        Self::new(&digits, 1, digits.len() as i64)
    }

    #[inline]
    pub fn sign(&self) -> RBigIntSign {
        match self.get_sign() {
            -1 => RBigIntSign::Minus,
            0 => RBigIntSign::NoSign,
            1 => RBigIntSign::Plus,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn bits(&self) -> u64 {
        self.bit_length()
            .expect("an allocated 64-bit RBigInt cannot overflow Signed bit_length") as u64
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.get_sign() == 0
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.int_eq(1)
    }

    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        self.toint().ok()
    }

    #[inline]
    pub fn to_i32(&self) -> Option<i32> {
        self.toint()
            .ok()
            .and_then(|value| i32::try_from(value).ok())
    }

    #[inline]
    pub fn to_i128(&self) -> Option<i128> {
        self.tolong().ok()
    }

    #[inline]
    pub fn to_u64(&self) -> Option<u64> {
        self.touint().ok()
    }

    #[inline]
    pub fn to_u32(&self) -> Option<u32> {
        self.touint()
            .ok()
            .and_then(|value| u32::try_from(value).ok())
    }

    #[inline]
    pub fn to_usize(&self) -> Option<usize> {
        self.touint()
            .ok()
            .and_then(|value| usize::try_from(value).ok())
    }

    /// Compatibility with compiler/marshal word serialization. The returned
    /// words are little-endian and the conversion is linear over RPython's
    /// 63-bit digits.
    pub fn to_u32_digits(&self) -> (RBigIntSign, Vec<u32>) {
        if self.get_sign() == 0 {
            return (RBigIntSign::NoSign, Vec::new());
        }
        let mut words = Vec::with_capacity(self.bits().div_ceil(32) as usize);
        let mut accumulator = 0_u128;
        let mut accumulator_bits = 0_u32;
        let mut i = 0;
        while i < self.numdigits() {
            accumulator |= (self.udigit(i) as u128) << accumulator_bits;
            accumulator_bits += SHIFT as u32;
            while accumulator_bits >= 32 {
                words.push(accumulator as u32);
                accumulator >>= 32;
                accumulator_bits -= 32;
            }
            i += 1;
        }
        if accumulator_bits != 0 {
            words.push(accumulator as u32);
        }
        while words.last() == Some(&0) {
            words.pop();
        }
        (self.sign(), words)
    }

    #[inline]
    pub fn to_f64(&self) -> Option<f64> {
        self.tofloat().ok()
    }

    pub fn from_f64(value: f64) -> Option<Self> {
        Self::fromfloat(value).ok()
    }

    pub fn parse_bytes(bytes: &[u8], radix: u32) -> Option<Self> {
        let source = std::str::from_utf8(bytes).ok()?;
        Self::fromstr(source, radix as i64, false).ok()
    }

    pub fn from_bytes_le(sign: RBigIntSign, bytes: &[u8]) -> Self {
        let mut value =
            Self::frombytes(bytes, "little", false).expect("byte order is statically valid");
        if sign == RBigIntSign::Minus && value.tobool() {
            value._set_sign(-1);
        }
        value
    }

    pub fn to_str_radix(&self, radix: u32) -> String {
        const ALPHABET: &str = "0123456789abcdefghijklmnopqrstuvwxyz";
        self.format(&ALPHABET[..radix as usize], "", "", 0)
            .expect("radix must be in 2..=36")
    }
}

/// Host-only wide integer specialization used by upstream-Python parity tests.
/// Translated RPython integer specializations use the machine-width
/// `digits_from_nonneg_long` below.
fn digits_from_nonneg_u128(mut value: u128) -> Vec<Digit> {
    let mut digits = Vec::new();
    loop {
        digits.push(_store_uwidedigit(value & MASK as UWideDigit));
        value >>= SHIFT;
        if value == 0 {
            return digits;
        }
    }
}

impl PartialOrd for RBigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RBigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.lt(other) {
            Ordering::Less
        } else if self.eq(other) {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }
}

macro_rules! impl_rbigint_from_signed {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for RBigInt {
                fn from(value: $ty) -> Self {
                    RBigInt::fromint(value as i64)
                }
            }
        )*
    };
}

macro_rules! impl_rbigint_from_unsigned {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for RBigInt {
                fn from(value: $ty) -> Self {
                    RBigInt::from_u128(value as u128)
                }
            }
        )*
    };
}

impl_rbigint_from_signed!(i8, i16, i32, i64, isize);
impl_rbigint_from_unsigned!(u8, u16, u32, u64, u128, usize);

impl From<i128> for RBigInt {
    fn from(value: i128) -> Self {
        RBigInt::fromlong(value)
    }
}

impl TryFrom<&RBigInt> for i64 {
    type Error = RBigIntError;
    fn try_from(value: &RBigInt) -> Result<Self, Self::Error> {
        value.toint()
    }
}

impl TryFrom<&RBigInt> for u64 {
    type Error = RBigIntError;
    fn try_from(value: &RBigInt) -> Result<Self, Self::Error> {
        value.touint()
    }
}

impl TryFrom<&RBigInt> for u32 {
    type Error = RBigIntError;
    fn try_from(value: &RBigInt) -> Result<Self, Self::Error> {
        value
            .touint()
            .and_then(|value| u32::try_from(value).map_err(|_| RBigIntError::Overflow))
    }
}

impl TryFrom<&RBigInt> for i128 {
    type Error = RBigIntError;
    fn try_from(value: &RBigInt) -> Result<Self, Self::Error> {
        value.tolong()
    }
}

macro_rules! impl_rbigint_binary_op {
    ($trait:ident, $method:ident, $rbigint_method:ident) => {
        impl std::ops::$trait for RBigInt {
            type Output = RBigInt;
            fn $method(self, other: RBigInt) -> RBigInt {
                RBigInt::$rbigint_method(&self, &other)
            }
        }

        impl<'a> std::ops::$trait<&'a RBigInt> for RBigInt {
            type Output = RBigInt;
            fn $method(self, other: &'a RBigInt) -> RBigInt {
                RBigInt::$rbigint_method(&self, other)
            }
        }

        impl std::ops::$trait<RBigInt> for &RBigInt {
            type Output = RBigInt;
            fn $method(self, other: RBigInt) -> RBigInt {
                RBigInt::$rbigint_method(self, &other)
            }
        }

        impl<'a> std::ops::$trait<&'a RBigInt> for &RBigInt {
            type Output = RBigInt;
            fn $method(self, other: &'a RBigInt) -> RBigInt {
                RBigInt::$rbigint_method(self, other)
            }
        }
    };
}

impl_rbigint_binary_op!(Add, add, add);
impl_rbigint_binary_op!(Sub, sub, sub);
impl_rbigint_binary_op!(Mul, mul, mul);
impl_rbigint_binary_op!(BitAnd, bitand, and_);
impl_rbigint_binary_op!(BitOr, bitor, or_);
impl_rbigint_binary_op!(BitXor, bitxor, xor);

macro_rules! impl_rbigint_scalar_ops {
    ($($ty:ty),* $(,)?) => {
        $(
            impl std::ops::Add<$ty> for RBigInt {
                type Output = RBigInt;
                fn add(self, other: $ty) -> RBigInt {
                    // `RBigInt::from` collects.
                    let this = RBigIntGcRoot::new(self);
                    let other = RBigInt::from(other);
                    RBigInt::add(&this, &other)
                }
            }
            impl std::ops::Sub<$ty> for RBigInt {
                type Output = RBigInt;
                fn sub(self, other: $ty) -> RBigInt {
                    // `RBigInt::from` collects.
                    let this = RBigIntGcRoot::new(self);
                    let other = RBigInt::from(other);
                    RBigInt::sub(&this, &other)
                }
            }
            impl std::ops::Mul<$ty> for RBigInt {
                type Output = RBigInt;
                fn mul(self, other: $ty) -> RBigInt {
                    // `RBigInt::from` collects.
                    let this = RBigIntGcRoot::new(self);
                    let other = RBigInt::from(other);
                    RBigInt::mul(&this, &other)
                }
            }
            impl std::ops::Rem<$ty> for RBigInt {
                type Output = RBigInt;
                fn rem(self, other: $ty) -> RBigInt {
                    // `RBigInt::from` collects.
                    let this = RBigIntGcRoot::new(self);
                    let other = RBigInt::from(other);
                    _divrem(&this, &other).expect("division by zero").1
                }
            }
        )*
    };
}

impl_rbigint_scalar_ops!(i32, i64, u32, u64, usize);

macro_rules! impl_rbigint_division_op {
    ($trait:ident, $method:ident, $tuple_index:tt) => {
        impl std::ops::$trait for RBigInt {
            type Output = RBigInt;
            fn $method(self, other: RBigInt) -> RBigInt {
                _divrem(&self, &other)
                    .expect("division by zero")
                    .$tuple_index
            }
        }

        impl<'a> std::ops::$trait<&'a RBigInt> for RBigInt {
            type Output = RBigInt;
            fn $method(self, other: &'a RBigInt) -> RBigInt {
                _divrem(&self, other)
                    .expect("division by zero")
                    .$tuple_index
            }
        }

        impl std::ops::$trait<RBigInt> for &RBigInt {
            type Output = RBigInt;
            fn $method(self, other: RBigInt) -> RBigInt {
                _divrem(self, &other)
                    .expect("division by zero")
                    .$tuple_index
            }
        }

        impl<'a> std::ops::$trait<&'a RBigInt> for &RBigInt {
            type Output = RBigInt;
            fn $method(self, other: &'a RBigInt) -> RBigInt {
                _divrem(self, other).expect("division by zero").$tuple_index
            }
        }
    };
}

impl_rbigint_division_op!(Div, div, 0);
impl_rbigint_division_op!(Rem, rem, 1);

impl std::ops::Neg for RBigInt {
    type Output = RBigInt;
    fn neg(self) -> RBigInt {
        RBigInt::neg(&self)
    }
}

impl std::ops::Neg for &RBigInt {
    type Output = RBigInt;
    fn neg(self) -> RBigInt {
        RBigInt::neg(self)
    }
}

impl std::ops::Not for RBigInt {
    type Output = RBigInt;
    fn not(self) -> RBigInt {
        self.invert()
    }
}

impl std::ops::Not for &RBigInt {
    type Output = RBigInt;
    fn not(self) -> RBigInt {
        self.invert()
    }
}

impl<T> std::ops::Shl<T> for RBigInt
where
    T: TryInto<u64>,
{
    type Output = RBigInt;
    fn shl(self, shift: T) -> RBigInt {
        let shift = shift.try_into().ok().expect("negative shift count");
        self.lshift(i64::try_from(shift).expect("shift count exceeds i64"))
            .expect("nonnegative shift")
    }
}

impl<T> std::ops::Shl<T> for &RBigInt
where
    T: TryInto<u64>,
{
    type Output = RBigInt;
    fn shl(self, shift: T) -> RBigInt {
        let shift = shift.try_into().ok().expect("negative shift count");
        self.lshift(i64::try_from(shift).expect("shift count exceeds i64"))
            .expect("nonnegative shift")
    }
}

impl<T> std::ops::Shr<T> for RBigInt
where
    T: TryInto<u64>,
{
    type Output = RBigInt;
    fn shr(self, shift: T) -> RBigInt {
        let shift = shift.try_into().ok().expect("negative shift count");
        self.rshift(
            i64::try_from(shift).expect("shift count exceeds i64"),
            false,
        )
        .expect("nonnegative shift")
    }
}

impl<T> std::ops::Shr<T> for &RBigInt
where
    T: TryInto<u64>,
{
    type Output = RBigInt;
    fn shr(self, shift: T) -> RBigInt {
        let shift = shift.try_into().ok().expect("negative shift count");
        self.rshift(
            i64::try_from(shift).expect("shift count exceeds i64"),
            false,
        )
        .expect("nonnegative shift")
    }
}

impl std::ops::AddAssign<RBigInt> for RBigInt {
    fn add_assign(&mut self, other: RBigInt) {
        *self = RBigInt::add(self, &other);
    }
}

impl std::ops::AddAssign<&RBigInt> for RBigInt {
    fn add_assign(&mut self, other: &RBigInt) {
        *self = RBigInt::add(self, other);
    }
}

impl std::ops::SubAssign<RBigInt> for RBigInt {
    fn sub_assign(&mut self, other: RBigInt) {
        *self = RBigInt::sub(self, &other);
    }
}

impl std::ops::MulAssign<RBigInt> for RBigInt {
    fn mul_assign(&mut self, other: RBigInt) {
        *self = RBigInt::mul(self, &other);
    }
}

impl num_traits::Zero for RBigInt {
    fn zero() -> Self {
        RBigInt::zero()
    }

    fn is_zero(&self) -> bool {
        RBigInt::is_zero(self)
    }
}

impl num_traits::One for RBigInt {
    fn one() -> Self {
        RBigInt::one()
    }

    fn is_one(&self) -> bool {
        RBigInt::is_one(self)
    }
}

impl num_traits::ToPrimitive for RBigInt {
    fn to_i64(&self) -> Option<i64> {
        RBigInt::to_i64(self)
    }

    fn to_u64(&self) -> Option<u64> {
        RBigInt::to_u64(self)
    }

    fn to_i128(&self) -> Option<i128> {
        RBigInt::to_i128(self)
    }

    fn to_u128(&self) -> Option<u128> {
        if self.get_sign() < 0 || self.numdigits() > 3 {
            return None;
        }
        let mut value = 0_u128;
        let mut i = self.numdigits();
        while i > 0 {
            i -= 1;
            let digit = self.udigit(i) as u128;
            if value > (u128::MAX - digit) >> SHIFT {
                return None;
            }
            value = (value << SHIFT) + digit;
        }
        Some(value)
    }
}

impl num_traits::FromPrimitive for RBigInt {
    fn from_i64(value: i64) -> Option<Self> {
        Some(RBigInt::fromint(value))
    }

    fn from_u64(value: u64) -> Option<Self> {
        Some(RBigInt::from_u128(value as u128))
    }

    fn from_i128(value: i128) -> Option<Self> {
        Some(RBigInt::fromlong(value))
    }

    fn from_u128(value: u128) -> Option<Self> {
        Some(RBigInt::from_u128(value))
    }

    fn from_f64(value: f64) -> Option<Self> {
        RBigInt::from_f64(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RBigIntError {
    Memory,
    Overflow,
    NegativeToUnsigned,
    InvalidEndianness,
    InvalidSignedness,
    DivisionByZero,
    NegativeShift,
    InvalidBitWidth,
    NegativeExponent,
    NegativeExponentWithModulus,
    ZeroModulus,
    NegativeSquareRoot,
    InfiniteFloat,
    NanFloat,
    FloatDivisionOverflow,
    MaxStrDigits,
    InvalidBase,
    ParseString,
    LogDomain,
}

pub const MAX_DIGITS_THAT_CAN_FIT_IN_INT: i64 = 2;

fn float_frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        let scaled = (x * 18_014_398_509_481_984.0).to_bits();
        let mantissa_bits = (scaled & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
        let exponent = (((scaled >> 52) & 0x7ff) as i32) - 1022 - 54;
        return (f64::from_bits(mantissa_bits), exponent);
    }
    let mantissa_bits = (bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
    (f64::from_bits(mantissa_bits), exp_field - 1022)
}

/// rbigint.py `_help_mult`.
fn _help_mult(
    x: &RBigInt,
    y: &RBigInt,
    modulus: Option<&RBigInt>,
) -> Result<RBigInt, RBigIntError> {
    let x = live_rbigint(x);
    let y = live_rbigint(y);
    let modulus = modulus.map(live_rbigint);
    let mut result = RBigIntGcRoot::new(x.mul(&y));
    if let Some(ref modulus) = modulus {
        result = RBigIntGcRoot::new(result.r#mod(modulus)?);
    }
    Ok(result.translated_alias())
}

/// rbigint.py `digits_from_nonneg_long`.
fn digits_from_nonneg_long(mut value: i64) -> Vec<Digit> {
    debug_assert!(value >= 0);
    let mut digits = Vec::new();
    loop {
        digits.push(_store_digit(_mask_digit(value & MASK as Digit)));
        value >>= SHIFT;
        if value == 0 {
            return digits;
        }
    }
}

/// Unsigned graph emitted by rbigint.py
/// `@specialize.argtype(0) digits_from_nonneg_long`.
fn digits_from_nonneg_ulong(mut value: u64) -> Vec<Digit> {
    let mut digits = Vec::new();
    loop {
        digits.push(_store_digit(_mask_udigit(value & MASK as UDigit)));
        value >>= SHIFT;
        if value == 0 {
            return digits;
        }
    }
}

/// rbigint.py `digits_for_most_neg_long`.
fn digits_for_most_neg_long(mut value: i64) -> Vec<Digit> {
    let mut digits = Vec::new();
    while _mask_digit(value) == 0 {
        digits.push(NULLDIGIT);
        value >>= SHIFT;
    }
    value = -value;
    debug_assert_eq!(value & MASK as Digit, value);
    digits.push(_store_digit(value));
    digits
}

/// rbigint.py `args_from_rarith_int1`.
fn args_from_rarith_int1(value: i64) -> (Vec<Digit>, i64) {
    if value > 0 {
        (digits_from_nonneg_long(value), 1)
    } else if value == 0 {
        (vec![NULLDIGIT], 0)
    } else if value != i64::MIN {
        (digits_from_nonneg_long(-value), -1)
    } else {
        (digits_for_most_neg_long(value), -1)
    }
}

/// rbigint.py `args_from_rarith_int`.
fn args_from_rarith_int(value: i64) -> (Vec<Digit>, i64) {
    args_from_rarith_int1(value)
}

/// Unsigned graph emitted by rbigint.py
/// `@specialize.argtype(0) args_from_rarith_int1`.
fn args_from_rarith_uint1(value: u64) -> (Vec<Digit>, i64) {
    if value == 0 {
        (vec![NULLDIGIT], 0)
    } else {
        (digits_from_nonneg_ulong(value), 1)
    }
}

/// Unsigned graph emitted by rbigint.py
/// `@specialize.argtype(0) args_from_rarith_int`.
fn args_from_rarith_uint(value: u64) -> (Vec<Digit>, i64) {
    args_from_rarith_uint1(value)
}

/// rbigint.py `args_from_long`.
#[majit_macros::not_rpython]
fn args_from_long(value: i128) -> (Vec<Digit>, i64) {
    if value > 0 {
        (digits_from_nonneg_u128(value as u128), 1)
    } else if value == 0 {
        (vec![NULLDIGIT], 0)
    } else {
        let magnitude = if value == i128::MIN {
            1_u128 << 127
        } else {
            (-value) as u128
        };
        (digits_from_nonneg_u128(magnitude), -1)
    }
}

/// rbigint.py `_x_add`.
fn _x_add(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let mut size_a = a.numdigits();
    let mut size_b = b.numdigits();
    let (mut a, mut b) = if size_a < size_b {
        std::mem::swap(&mut size_a, &mut size_b);
        (b.translated_alias(), a.translated_alias())
    } else {
        (a.translated_alias(), b.translated_alias())
    };
    let mut z = RBigInt::with_size_livevars(size_a + 1, 1, &mut a, &mut b);
    let mut i = 0;
    let mut carry = 0_u64;
    while i < size_b {
        carry = carry.wrapping_add(a.udigit(i)).wrapping_add(b.udigit(i));
        z.setdigit_udigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    while i < size_a {
        carry = carry.wrapping_add(a.udigit(i));
        z.setdigit_udigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    z.setdigit_udigit(i, carry);
    z._normalize();
    z.translated_alias()
}

/// rbigint.py `_x_int_add`.
fn _x_int_add(a: &RBigInt, b: i64) -> RBigInt {
    let a = live_rbigint(a);
    let size_a = a.numdigits();
    let mut z = RBigInt::with_size(size_a + 1, 1);
    // Past the last collecting call: read digits through a plain handle.
    let a: &RBigInt = &a;
    let mut carry = a.udigit(0).wrapping_add(b.unsigned_abs());
    z.setdigit_udigit(0, carry);
    carry >>= SHIFT;
    let mut i = 1;
    while i < size_a {
        carry = carry.wrapping_add(a.udigit(i));
        z.setdigit_udigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    z.setdigit_udigit(i, carry);
    z._normalize();
    z
}

/// rbigint.py `_x_sub`.
fn _x_sub(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let mut size_a = a.numdigits();
    let mut size_b = b.numdigits();
    let mut sign = 1;
    let (mut a, mut b) = if size_a < size_b {
        sign = -1;
        std::mem::swap(&mut size_a, &mut size_b);
        (b.translated_alias(), a.translated_alias())
    } else if size_a == size_b {
        let mut i = size_a;
        while i > 0 && a.digit(i - 1) == b.digit(i - 1) {
            i -= 1;
        }
        if i == 0 {
            return RBigInt::zero();
        }
        if a.digit(i - 1) < b.digit(i - 1) {
            sign = -1;
            size_a = i;
            size_b = i;
            (b.translated_alias(), a.translated_alias())
        } else {
            size_a = i;
            size_b = i;
            (a.translated_alias(), b.translated_alias())
        }
    } else {
        (a.translated_alias(), b.translated_alias())
    };

    let mut z = RBigInt::with_size_livevars(size_a, sign, &mut a, &mut b);
    let mut borrow = 0_u64;
    let mut i = 0;
    while i < size_b {
        borrow = a.udigit(i).wrapping_sub(b.udigit(i)).wrapping_sub(borrow);
        z.setdigit_udigit(i, borrow);
        borrow = (borrow >> SHIFT) & 1;
        i += 1;
    }
    while i < size_a {
        borrow = a.udigit(i).wrapping_sub(borrow);
        z.setdigit_udigit(i, borrow);
        borrow = (borrow >> SHIFT) & 1;
        i += 1;
    }
    debug_assert_eq!(borrow, 0);
    z._normalize();
    z.translated_alias()
}

/// rbigint.py `_x_int_sub`.
fn _x_int_sub(a: &RBigInt, b: i64) -> RBigInt {
    let a = live_rbigint(a);
    let size_a = a.numdigits();
    let bdigit = b.unsigned_abs();
    if size_a == 1 {
        let adigit = a.digit(0);
        if adigit as u64 == bdigit {
            return RBigInt::zero();
        }
        return if adigit as u64 > bdigit {
            RBigInt::fromint((adigit as u64 - bdigit) as i64)
        } else {
            RBigInt::fromint(-((bdigit - adigit as u64) as i64))
        };
    }

    let mut z = RBigInt::with_size(size_a, 1);
    // Past the last collecting call: read digits through a plain handle.
    let a: &RBigInt = &a;
    let mut borrow = a.udigit(0).wrapping_sub(bdigit);
    z.setdigit_udigit(0, borrow);
    borrow = (borrow >> SHIFT) & 1;
    let mut i = 1;
    while i < size_a {
        borrow = a.udigit(i).wrapping_sub(borrow);
        z.setdigit_udigit(i, borrow);
        borrow = (borrow >> SHIFT) & 1;
        i += 1;
    }
    debug_assert_eq!(borrow, 0);
    z._normalize();
    z
}

// rbigint.py `ptwotable`.
//
// Every upstream lookup first takes an absolute digit and proves that it is a
// power of two. The immutable dict's positive and negative keys therefore
// collapse to this dense exponent-indexed table without changing lookup
// semantics. Keeping it as a module-global array makes the prebuilt owner
// visible to source translation while avoiding a foreign Rust HashMap.
const fn make_ptwotable() -> [i64; SHIFT as usize] {
    let mut table = [0_i64; SHIFT as usize];
    let mut exponent = 1;
    while exponent < table.len() {
        table[exponent] = exponent as i64;
        exponent += 1;
    }
    table
}

const PTWOTABLE: [i64; SHIFT as usize] = make_ptwotable();

/// rbigint.py `_x_mul`.
fn _x_mul(a: &RBigInt, b: &RBigInt, digit: Digit) -> RBigInt {
    let size_a = a.numdigits();
    let size_b = b.numdigits();

    if std::ptr::eq(a, b) {
        let mut a = a.translated_alias();
        let mut b = b.translated_alias();
        let mut z = RBigInt::with_size_livevars(size_a + size_b, 1, &mut a, &mut b);
        let mut i = 0;
        while i < size_a {
            let mut f = a.uwidedigit(i);
            let mut pz = i << 1;
            let mut pa = i + 1;
            let mut carry = z.uwidedigit(pz) + f * f;
            z.setdigit_uwidedigit(pz, carry);
            pz += 1;
            carry >>= SHIFT;
            debug_assert!(carry <= MASK as UWideDigit);
            f <<= 1;
            while pa < size_a {
                carry += z.uwidedigit(pz) + a.uwidedigit(pa) * f;
                pa += 1;
                z.setdigit_uwidedigit(pz, carry);
                pz += 1;
                carry >>= SHIFT;
            }
            if carry != 0 {
                carry += z.udigit(pz) as UWideDigit;
                z.setdigit_uwidedigit(pz, carry);
                pz += 1;
                carry >>= SHIFT;
            }
            if carry != 0 {
                let acc = z.udigit(pz) as UWideDigit + carry;
                z.setdigit_uwidedigit(pz, acc);
            }
            debug_assert_eq!(carry >> SHIFT, 0);
            i += 1;
        }
        z._normalize();
        return z.translated_alias();
    } else if digit != 0 {
        if digit & (digit - 1) == 0 {
            return b.lqshift(PTWOTABLE[digit.trailing_zeros() as usize]);
        }
        return _muladd1(b, digit, 0);
    }

    // a is not b
    let mut a = a.translated_alias();
    let mut b = b.translated_alias();
    let mut z = RBigInt::with_size_livevars(size_a + size_b, 1, &mut a, &mut b);
    let mut i = 0;
    let size_a1 = size_a - 1;
    let size_b1 = size_b - 1;
    while i < size_a1 {
        let f0 = a.uwidedigit(i);
        let f1 = a.uwidedigit(i + 1);
        let mut pz = i;
        let mut carry = z.uwidedigit(pz) + b.uwidedigit(0) * f0;
        z.setdigit_uwidedigit(pz, carry);
        pz += 1;
        carry >>= SHIFT;
        let mut j = 0;
        while j < size_b1 {
            carry += z.uwidedigit(pz) + b.uwidedigit(j + 1) * f0 + b.uwidedigit(j) * f1;
            z.setdigit_uwidedigit(pz, carry);
            pz += 1;
            carry >>= SHIFT;
            j += 1;
        }
        carry += z.uwidedigit(pz) + b.uwidedigit(size_b1) * f1;
        z.setdigit_uwidedigit(pz, carry);
        pz += 1;
        carry >>= SHIFT;
        if carry != 0 {
            z.setdigit_uwidedigit(pz, carry);
        }
        debug_assert_eq!(carry >> SHIFT, 0);
        i += 2;
    }
    if size_a & 1 != 0 {
        let mut pz = size_a1;
        let f = a.uwidedigit(pz);
        let mut pb = 0;
        let mut carry = 0_u128;
        while pb < size_b {
            carry += z.uwidedigit(pz) + b.uwidedigit(pb) * f;
            pb += 1;
            z.setdigit_uwidedigit(pz, carry);
            pz += 1;
            carry >>= SHIFT;
        }
        if carry != 0 {
            let acc = z.udigit(pz) as UWideDigit + carry;
            z.setdigit_uwidedigit(pz, acc);
        }
    }
    z._normalize();
    z.translated_alias()
}

/// rbigint.py `_kmul_split`.
fn _kmul_split(n: &RBigInt, size: i64) -> (RBigInt, RBigInt) {
    let n = live_rbigint(n);
    let size_n = n.numdigits();
    let size_lo = size_n.min(size);
    // rbigint.py `_kmul_split` copies `n._digits[:size_lo]` once into the
    // new rbigint. `RBigInt::new` is that copy (`Digits::new`).
    let lo = if size_lo == 0 {
        RBigInt::zero()
    } else {
        RBigInt::new_from_rooted(&n, 0, size_lo as usize, 1, size_lo)
    };
    // `RBigInt::new` for `hi` can collect; keep `lo._digits` live.
    let mut lo = RBigIntGcRoot::new(lo);
    let hi = if size_lo == size_n {
        RBigInt::zero()
    } else {
        RBigInt::new_from_rooted(&n, size_lo as usize, size_n as usize, 1, size_n - size_lo)
    };
    let mut hi = RBigIntGcRoot::new(hi);
    lo.normalize();
    hi.normalize();
    ((*hi).clone(), (*lo).clone())
}

/// rbigint.py `_k_mul`.
fn _k_mul(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let asize = a.numdigits();
    let bsize = b.numdigits();
    // rbigint.py `_k_mul` keeps `ret`, `a`, `b` as ordinary locals; the GC
    // transform reloads them after `[NULLDIGIT] * (asize+bsize)` and after
    // each recursive `mul`. Host handles store `_digits` by value, so that
    // edge lives on owner-root slots (`with_size_keeping_handles`) and the
    // result buffer stays rooted across the splits and recursive muls.
    let same = std::ptr::eq(a, b);
    let (a, b, mut ret) = RBigInt::with_size_keeping_handles(asize + bsize, 1, a, b);
    let shift = bsize >> 1;
    let (bh, bl) = _kmul_split(&b, shift);
    let bh = RBigIntGcRoot::new(bh);
    let bl = RBigIntGcRoot::new(bl);

    if !same && asize <= shift {
        let t1 = RBigIntGcRoot::new(a.mul(&bl));
        {
            // No collecting call inside the copy: plain handles.
            let (ret, t1): (&mut RBigInt, &RBigInt) = (&mut ret, &t1);
            let mut i = 0;
            while i < t1.numdigits() {
                ret.setdigit(i, t1.digit(i));
                i += 1;
            }
        }
        let t2 = RBigIntGcRoot::new(a.mul(&bh));
        let i = ret.numdigits() - shift;
        _v_iadd(&mut ret, shift, i, &t2, t2.numdigits());
        ret.normalize();
        return ret.translated_alias();
    }
    let a_parts = if same {
        None
    } else {
        let (ah, al) = _kmul_split(&a, shift);
        Some((RBigIntGcRoot::new(ah), RBigIntGcRoot::new(al)))
    };
    let (ah, al) = match &a_parts {
        None => (&bh, &bl),
        Some((ah, al)) => (ah, al),
    };

    let t1 = RBigIntGcRoot::new(ah.mul(&bh));
    debug_assert!(t1.get_sign() >= 0);
    debug_assert!(2 * shift + t1.numdigits() <= ret.numdigits());
    {
        // No collecting call inside the copy: plain handles.
        let (ret, t1): (&mut RBigInt, &RBigInt) = (&mut ret, &t1);
        let mut i = 0;
        while i < t1.numdigits() {
            ret.setdigit(2 * shift + i, t1.digit(i));
            i += 1;
        }
    }

    let t2 = RBigIntGcRoot::new(al.mul(&bl));
    debug_assert!(t2.get_sign() >= 0);
    debug_assert!(t2.numdigits() <= 2 * shift);
    {
        // No collecting call inside the copy: plain handles.
        let (ret, t2): (&mut RBigInt, &RBigInt) = (&mut ret, &t2);
        let mut i = 0;
        while i < t2.numdigits() {
            ret.setdigit(i, t2.digit(i));
            i += 1;
        }
    }

    let i = ret.numdigits() - shift;
    _v_isub(&mut ret, shift, i, &t2, t2.numdigits());
    _v_isub(&mut ret, shift, i, &t1, t1.numdigits());

    let t1 = RBigIntGcRoot::new(_x_add(ah, al));
    let t3 = RBigIntGcRoot::new(if same {
        t1.mul(&t1)
    } else {
        let t2 = RBigIntGcRoot::new(_x_add(&bh, &bl));
        t1.mul(&t2)
    });
    debug_assert!(t3.get_sign() >= 0);
    _v_iadd(&mut ret, shift, i, &t3, t3.numdigits());
    ret.normalize();
    ret.translated_alias()
}

/// rbigint.py `_inplace_divrem1`.
fn _inplace_divrem1(pout: &mut RBigInt, pin: &RBigInt, n: Digit) -> Digit {
    debug_assert!(n > 0 && n <= MASK as Digit);
    let mut rem = 0_u128;
    let mut size = pin.numdigits();
    while size > 0 {
        size -= 1;
        rem = (rem << SHIFT) | pin.udigit(size) as UWideDigit;
        let hi = rem / n as UWideDigit;
        pout.setdigit_uwidedigit(size, hi);
        rem -= hi * n as UWideDigit;
    }
    rem as Digit
}

/// rbigint.py `_divrem1`.
#[majit_macros::jit_elidable]
fn _divrem1(a: &RBigInt, n: Digit) -> (RBigInt, Digit) {
    debug_assert!(n > 0 && n <= MASK as Digit);
    let a = live_rbigint(a);
    let size = a.numdigits();
    let mut z = RBigInt::with_size(size, 1);
    let rem = _inplace_divrem1(&mut z, &a, n);
    z._normalize();
    (z, rem)
}

/// rbigint.py `_int_rem_core`.
fn _int_rem_core(a: &RBigInt, digit: Digit) -> Digit {
    let mut size = a.numdigits() - 1;
    if size > 0 {
        let mut wrem = a.widedigit(size);
        while size > 0 {
            size -= 1;
            wrem = ((wrem << SHIFT) | a.widedigit(size)) % digit as WideDigit;
        }
        _store_widedigit(wrem)
    } else {
        _store_digit(a.digit(0) % digit)
    }
}

/// rbigint.py `_v_iadd`.
fn _v_iadd(x: &mut RBigInt, xofs: i64, m: i64, y: &RBigInt, n: i64) -> UDigit {
    debug_assert!(m >= n);
    let mut carry = 0_u64;
    let mut i = xofs;
    let mut iend = xofs + n;
    while i < iend {
        carry = carry
            .wrapping_add(x.udigit(i))
            .wrapping_add(y.udigit(i - xofs));
        x.setdigit_udigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    iend = xofs + m;
    while carry != 0 && i < iend {
        carry = carry.wrapping_add(x.udigit(i));
        x.setdigit_udigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    carry
}

/// rbigint.py `_v_isub`.
fn _v_isub(x: &mut RBigInt, xofs: i64, m: i64, y: &RBigInt, n: i64) -> UDigit {
    debug_assert!(m >= n);
    let mut borrow = 0_u64;
    let mut i = xofs;
    let mut iend = xofs + n;
    while i < iend {
        borrow = x
            .udigit(i)
            .wrapping_sub(y.udigit(i - xofs))
            .wrapping_sub(borrow);
        x.setdigit_udigit(i, borrow);
        borrow = (borrow >> SHIFT) & 1;
        i += 1;
    }
    iend = xofs + m;
    while borrow != 0 && i < iend {
        borrow = x.udigit(i).wrapping_sub(borrow);
        x.setdigit_udigit(i, borrow);
        borrow = (borrow >> SHIFT) & 1;
        i += 1;
    }
    borrow
}

/// rbigint.py `_muladd1`.
fn _muladd1(a: &RBigInt, n: Digit, extra: Digit) -> RBigInt {
    debug_assert!(n > 0);
    let a = live_rbigint(a);
    let size_a = a.numdigits();
    let mut z = RBigInt::with_size(size_a + 1, 1);
    // Past the last collecting call: read digits through a plain handle.
    let a: &RBigInt = &a;
    debug_assert_eq!(extra & MASK as Digit, extra);
    let mut carry = _unsigned_widen_digit(extra);
    let mut i = 0;
    while i < size_a {
        carry += a.uwidedigit(i) * n as UWideDigit;
        z.setdigit_uwidedigit(i, carry);
        carry >>= SHIFT;
        i += 1;
    }
    z.setdigit_uwidedigit(i, carry);
    z._normalize();
    z
}

/// rbigint.py `_v_lshift`.
fn _v_lshift(z: &mut RBigInt, a: &RBigInt, m: i64, d: i64) -> UWideDigit {
    let mut carry = 0_u128;
    let mut i = 0;
    while i < m {
        let acc = (a.uwidedigit(i) << d as u32) | carry;
        z.setdigit_uwidedigit(i, acc);
        carry = acc >> SHIFT;
        i += 1;
    }
    carry
}

/// rbigint.py `_v_rshift`.
fn _v_rshift(z: &mut RBigInt, a: &RBigInt, m: i64, d: i64) -> UWideDigit {
    let mut carry = 0_u128;
    let mask = (1_u128 << d as u32) - 1;
    let mut i = m;
    while i > 0 {
        i -= 1;
        let acc = (carry << SHIFT) | a.udigit(i) as UWideDigit;
        carry = acc & mask;
        z.setdigit_uwidedigit(i, acc >> d as u32);
    }
    carry
}

/// rbigint.py `_x_divrem`.
pub(crate) fn _x_divrem(v1: &RBigInt, w1: &RBigInt) -> (RBigInt, RBigInt) {
    let v1 = live_rbigint(v1);
    let w1 = live_rbigint(w1);
    let mut size_v = v1.numdigits();
    let size_w = w1.numdigits();
    debug_assert!(size_v >= size_w && size_w > 1);
    let mut v = live_rbigint(&RBigInt::with_size(size_v + 1, 1));
    let mut w = live_rbigint(&RBigInt::with_size(size_w, 1));

    let d = SHIFT - bits_in_digit(w1.digit(size_w - 1));
    let carry = _v_lshift(&mut w, &w1, size_w, d);
    debug_assert_eq!(carry, 0);
    let carry = _v_lshift(&mut v, &v1, size_v, d);
    if carry != 0 || v.digit(size_v - 1) >= w.digit(size_w - 1) {
        v.setdigit_uwidedigit(size_v, carry);
        size_v += 1;
    }

    let mut k = size_v - size_w;
    if k == 0 {
        // Upstream's `assert _v_rshift(...) == 0` contains the operation that
        // materialises the remainder.  Do not put the call itself inside a
        // Rust `debug_assert`, which disappears in release builds.
        let carry = _v_rshift(&mut w, &v, size_w, d);
        debug_assert_eq!(carry, 0);
        w.normalize();
        // rbigint.py deliberately does not return NULLRBIGINT here:
        // callers of this internal division helper may modify the result.
        // Keep both the rbigint value and its digit array fresh.
        return (RBigInt::new(&[NULLDIGIT], 0, 0), w.clone());
    }
    let mut a = live_rbigint(&RBigInt::with_size(k, 1));
    {
        // Past the last collecting call: the division loop runs on plain
        // handles.
        let v: &mut RBigInt = &mut v;
        let w: &RBigInt = &w;
        let a: &mut RBigInt = &mut a;
        let wm1 = w.widedigit(size_w - 1);
        let wm2 = w.widedigit(size_w - 2);
        let mut j = size_v - 1;
        while k > 0 {
            k -= 1;
            let vtop = if j >= size_v { 0 } else { v.widedigit(j) };
            debug_assert!(vtop <= wm1);
            let vv = (vtop << SHIFT) | v.widedigit(j - 1);
            debug_assert!(vv >= 0 && wm1 >= 1);
            let mut q = vv / wm1;
            let mut r = vv % wm1;
            let vj2 = v.digit(j - 2) as WideDigit;
            while wm2 * q > ((r << SHIFT) | vj2) {
                q -= 1;
                r += wm1;
            }

            let mut zhi = 0_i128;
            let mut i = 0;
            while i < size_w {
                let z = v.widedigit(k + i) + zhi - q * w.widedigit(i);
                v.setdigit_widedigit(k + i, z);
                zhi = z >> SHIFT;
                i += 1;
            }
            if vtop + zhi < 0 {
                let mut carry = 0_u64;
                i = 0;
                while i < size_w {
                    carry = carry
                        .wrapping_add(v.udigit(k + i))
                        .wrapping_add(w.udigit(i));
                    v.setdigit_udigit(k + i, carry);
                    carry >>= SHIFT;
                    i += 1;
                }
                q -= 1;
            }
            a.setdigit_widedigit(k, q);
            j -= 1;
        }
    }
    // Same side-effecting upstream assertion as the k == 0 arm above.
    let carry = _v_rshift(&mut w, &v, size_w, d);
    debug_assert_eq!(carry, 0);
    a.normalize();
    w.normalize();
    (a.clone(), w.clone())
}

/// rbigint.py `_divrem`.
#[majit_macros::jit_elidable]
pub fn _divrem(a: &RBigInt, b: &RBigInt) -> Result<(RBigInt, RBigInt), RBigIntError> {
    let a = live_rbigint(a);
    let b = live_rbigint(b);
    let size_a = a.numdigits();
    let size_b = b.numdigits();
    if b.get_sign() == 0 {
        return Err(RBigIntError::DivisionByZero);
    }
    if size_a < size_b || (size_a == size_b && a.digit(size_a - 1) < b.digit(size_b - 1)) {
        return Ok((RBigInt::zero(), a.translated_alias()));
    }
    let (mut z, mut rem) = if size_b == 1 {
        let (z, urem) = _divrem1(&a, b.digit(0));
        let z = RBigIntGcRoot::new(z);
        let rem = RBigInt::new(&[urem as Digit], if urem != 0 { 1 } else { 0 }, 1);
        ((*z).clone(), rem)
    } else {
        _x_divrem(&a, &b)
    };
    if a.get_sign() != b.get_sign() {
        z._set_sign(-z.get_sign());
    }
    if a.get_sign() < 0 && rem.get_sign() != 0 {
        rem._set_sign(-rem.get_sign());
    }
    Ok((z, rem))
}

/// rbigint.py `_extract_digits`.
fn _extract_digits(a: &RBigInt, startindex: i64, numdigits: i64) -> RBigInt {
    if startindex >= a.numdigits() {
        return RBigInt::zero();
    }
    let stop = (startindex + numdigits).min(a.numdigits());
    if stop == startindex {
        return RBigInt::zero();
    }
    let a = live_rbigint(a);
    let mut result =
        RBigInt::new_from_rooted(&a, startindex as usize, stop as usize, 1, stop - startindex);
    result._normalize();
    result
}

/// rbigint.py `div2n1n`.
fn div2n1n(
    a_container: &RBigInt,
    a_startindex: i64,
    b: &RBigInt,
    n_s: i64,
) -> Result<(RBigInt, RBigInt), RBigIntError> {
    let a_container = live_rbigint(a_container);
    let b = live_rbigint(b);
    if n_s <= holder_limit(&HOLDER.DIV_LIMIT) {
        let a = _extract_digits(&a_container, a_startindex, 2 * n_s);
        if a.get_sign() == 0 {
            return Ok((RBigInt::zero(), RBigInt::zero()));
        }
        return _divrem(&a, &b);
    }
    debug_assert_eq!(n_s & 1, 0);
    let half_n_s = n_s >> 1;
    let b1 = RBigIntGcRoot::new(_extract_digits(&b, half_n_s, half_n_s));
    let b2 = RBigIntGcRoot::new(_extract_digits(&b, 0, half_n_s));
    let (q1, r1) = div3n2n(
        &a_container,
        a_startindex + n_s,
        &a_container,
        a_startindex + half_n_s,
        &b,
        &b1,
        &b2,
        half_n_s,
    )?;
    let q1 = RBigIntGcRoot::new(q1);
    let r1 = RBigIntGcRoot::new(r1);
    let (q2, r) = div3n2n(&r1, 0, &a_container, a_startindex, &b, &b1, &b2, half_n_s)?;
    // `_full_digits_lshift_then_or` collects while `r` is live.
    let r = RBigIntGcRoot::new(r);
    let q = _full_digits_lshift_then_or(&q1, half_n_s, &q2)?;
    Ok((q, r.translated_alias()))
}

/// rbigint.py `div3n2n`.
#[allow(clippy::too_many_arguments)]
fn div3n2n(
    a12_container: &RBigInt,
    a12_startindex: i64,
    a3_container: &RBigInt,
    a3_startindex: i64,
    b: &RBigInt,
    b1: &RBigInt,
    b2: &RBigInt,
    n_s: i64,
) -> Result<(RBigInt, RBigInt), RBigIntError> {
    // rbigint.py `div3n2n` keeps a12, a3, b, b1, b2 as ordinary locals
    // across `div2n1n` and `q*b2`. Host handles store `_digits` by value.
    let a12_container = live_rbigint(a12_container);
    let a3_container = live_rbigint(a3_container);
    let b = live_rbigint(b);
    let b1 = live_rbigint(b1);
    let b2 = live_rbigint(b2);
    let (q, r) = div2n1n(&a12_container, a12_startindex, &b1, n_s)?;
    let mut q = RBigIntGcRoot::new(q);
    let mut r = RBigIntGcRoot::new(r);
    if r.get_sign() == 0 {
        r = RBigIntGcRoot::new(_extract_digits(&a3_container, a3_startindex, n_s));
    } else {
        let r_size = r.numdigits();
        let combined_size = n_s.checked_add(r_size).ok_or(RBigIntError::Memory)?;
        let mut combined = RBigInt::try_with_size(combined_size, 1)?;
        // Past the last collecting call: read digits through a plain handle.
        let a3_container: &RBigInt = &a3_container;
        let r_digits: &RBigInt = &r;
        let stop = (a3_startindex + n_s).min(a3_container.numdigits());
        let mut source = a3_startindex;
        let mut index = 0;
        while source < stop {
            combined.setdigit(index, a3_container.digit(source));
            source += 1;
            index += 1;
        }
        index = n_s;
        let mut i = 0;
        while i < r_size {
            combined.setdigit(index, r_digits.digit(i));
            index += 1;
            i += 1;
        }
        combined._normalize();
        r = RBigIntGcRoot::new(combined);
    }
    if q.get_sign() == 0 {
        return Ok(((*q).clone(), (*r).clone()));
    }
    let qb2 = RBigIntGcRoot::new(q.mul(&b2));
    r = RBigIntGcRoot::new(r.sub(&qb2));
    while r.get_sign() < 0 {
        q = RBigIntGcRoot::new(q.int_sub(1));
        r = RBigIntGcRoot::new(r.add(&b));
    }
    Ok(((*q).clone(), (*r).clone()))
}

/// rbigint.py `_full_digits_lshift_then_or`.
fn _full_digits_lshift_then_or(a: &RBigInt, n: i64, b: &RBigInt) -> Result<RBigInt, RBigIntError> {
    if a.get_sign() == 0 {
        return Ok(b.translated_alias());
    }
    let a = live_rbigint(a);
    let b = live_rbigint(b);
    let bdigits = b.numdigits();
    debug_assert!(bdigits <= n);
    let result_size = a.numdigits().checked_add(n).ok_or(RBigIntError::Memory)?;
    let mut result = RBigInt::try_with_size(result_size, 1)?;
    // Past the last collecting call: read digits through a plain handle.
    let a: &RBigInt = &a;
    let b: &RBigInt = &b;
    let mut i = 0;
    while i < bdigits {
        result.setdigit(i, b.digit(i));
        i += 1;
    }
    i = 0;
    while i < a.numdigits() {
        result.setdigit(n + i, a.digit(i));
        i += 1;
    }
    Ok(result)
}

/// rbigint.py `_divmod_fast_pos`.
fn _divmod_fast_pos(a: &RBigInt, b: &RBigInt) -> Result<(RBigInt, RBigInt), RBigIntError> {
    let a = live_rbigint(a);
    let b = live_rbigint(b);
    let n = b.bit_length()?;
    let m = a.bit_length()?;
    if m < n {
        return Ok((RBigInt::zero(), a.translated_alias()));
    }
    let mut new_n = SHIFT * holder_limit(&HOLDER.DIV_LIMIT);
    while new_n < n {
        new_n <<= 1;
    }
    let rest_shift = new_n - n;
    let a_shifted;
    let b_shifted;
    let (a, b) = if rest_shift != 0 {
        a_shifted = RBigIntGcRoot::new(a.lshift(rest_shift)?);
        b_shifted = RBigIntGcRoot::new(b.lshift(rest_shift)?);
        debug_assert_eq!(b_shifted.bit_length(), Ok(new_n));
        (&a_shifted, &b_shifted)
    } else {
        (&a, &b)
    };
    let n_s = new_n / SHIFT;

    let chunk_count = a
        .numdigits()
        .checked_add(n_s - 1)
        .ok_or(RBigIntError::Memory)?
        / n_s;
    let chunk_capacity = usize::try_from(chunk_count).map_err(|_| RBigIntError::Memory)?;
    let mut a_digits_base_two_pow_n = Vec::new();
    a_digits_base_two_pow_n
        .try_reserve_exact(chunk_capacity)
        .map_err(|_| RBigIntError::Memory)?;
    let mut start = 0;
    while start < a.numdigits() {
        // rbigint.py constructs these chunks directly from
        // `a._digits[i:stop]`; unlike `_extract_digits`, it deliberately does
        // not normalize them.  The lower chunks are base-2**n digits and must
        // retain their fixed `n_s`-digit width (including leading zero
        // machine digits) for the recursive Burnikel-Ziegler slice offsets.
        let stop = (start + n_s).min(a.numdigits());
        let digits_len = stop - start;
        a_digits_base_two_pow_n.push(RBigIntGcRoot::new(RBigInt::try_new_from_rooted(
            a,
            start as usize,
            stop as usize,
            1,
            digits_len,
        )?));
        start = stop;
    }
    let mut a_digits_index = a_digits_base_two_pow_n.len() as i64 - 1;
    debug_assert!(a_digits_index >= 0);
    let mut r = if a_digits_base_two_pow_n[a_digits_index as usize].ge(b) {
        RBigIntGcRoot::new(RBigInt::zero())
    } else {
        let value = a_digits_base_two_pow_n[a_digits_index as usize].translated_alias();
        a_digits_index -= 1;
        RBigIntGcRoot::new(value)
    };

    let mut q_digits: Option<RBigIntGcRoot> = None;
    let mut q_index_start = a_digits_index * n_s;
    while a_digits_index >= 0 {
        let arg1 = RBigIntGcRoot::new(_full_digits_lshift_then_or(
            &r,
            n_s,
            &a_digits_base_two_pow_n[a_digits_index as usize],
        )?);
        let (q_digit, next_r) = div2n1n(&arg1, 0, b, n_s)?;
        let q_digit = RBigIntGcRoot::new(q_digit);
        r = RBigIntGcRoot::new(next_r);
        if q_digits.is_none() {
            let q_size = q_index_start
                .checked_add(q_digit.numdigits())
                .ok_or(RBigIntError::Memory)?;
            q_digits = Some(RBigIntGcRoot::new(RBigInt::try_with_size(q_size, 1)?));
        }
        if let Some(q_digits) = &mut q_digits {
            let mut i = 0;
            while i < q_digit.numdigits() {
                q_digits.setdigit(q_index_start + i, q_digit.digit(i));
                i += 1;
            }
        }
        q_index_start -= n_s;
        a_digits_index -= 1;
    }
    if rest_shift != 0 {
        r = RBigIntGcRoot::new(r.rshift(rest_shift, false)?);
    }
    let mut q = q_digits
        .map(|value| value.translated_alias())
        .unwrap_or_else(RBigInt::zero);
    q._normalize();
    r.normalize();
    Ok((q, r.translated_alias()))
}

/// rbigint.py `divmod_big`.
pub fn divmod_big(a: &RBigInt, b: &RBigInt) -> Result<(RBigInt, RBigInt), RBigIntError> {
    let a = live_rbigint(a);
    let b = live_rbigint(b);
    if b.get_sign() == 0 {
        return Err(RBigIntError::DivisionByZero);
    } else if b.get_sign() < 0 {
        let (q, r) = divmod_big(&a.neg(), &b.neg())?;
        return Ok((q, r.neg()));
    } else if a.get_sign() < 0 {
        let (q, r) = divmod_big(&a.invert(), &b)?;
        // Each `invert` and the `add` collect: `r`, then `q.invert()`, stay
        // rooted across the next one, and `b` is read only after the last
        // `invert`, as the translated locals would be.
        let r = RBigIntGcRoot::new(r);
        let q = RBigIntGcRoot::new(q.invert());
        let r_inverted = r.invert();
        let r = b.add(&r_inverted);
        return Ok((q.translated_alias(), r));
    } else if a.get_sign() == 0 {
        return Ok((RBigInt::zero(), RBigInt::zero()));
    }
    _divmod_fast_pos(&a, &b)
}

/// rbigint.py `_x_int_lt`.
fn _x_int_lt(a: &RBigInt, b: i64, eq: bool) -> bool {
    let mut osign = 1;
    if b == 0 {
        osign = 0;
    } else if b < 0 {
        osign = -1;
    }

    if a.get_sign() > osign {
        return false;
    } else if a.get_sign() < osign {
        return true;
    }

    let digits = a.numdigits();
    if digits > 1 {
        return osign != 1;
    }

    let d1 = a.get_sign() * a.digit(0);
    if eq {
        if d1 <= b {
            return true;
        }
    } else if d1 < b {
        return true;
    }
    false
}

/// rbigint.py `_AsScaledDouble`.
#[allow(non_snake_case)]
fn _AsScaledDouble(value: &RBigInt) -> (f64, i64) {
    const NBITS_WANTED: i64 = 57;
    if value.get_sign() == 0 {
        return (0.0, 0);
    }
    let mut i = value.numdigits() - 1;
    let sign = value.get_sign();
    let mut x = value.digit(i) as f64;
    let mut nbitsneeded = NBITS_WANTED - 1;
    while i > 0 && nbitsneeded > 0 {
        i -= 1;
        x = x * FLOAT_MULTIPLIER + value.digit(i) as f64;
        nbitsneeded -= SHIFT;
    }
    debug_assert!(x > 0.0);
    (x * sign as f64, i)
}

/// rbigint.py `_AsDouble`.
#[majit_macros::dont_look_inside]
#[allow(non_snake_case)]
fn _AsDouble(value: &RBigInt) -> Result<f64, RBigIntError> {
    const DBL_MANT_DIG: i64 = 53;
    const DBL_MAX_EXP: i64 = 1024;

    let sign = value.get_sign();
    if sign == 0 {
        return Ok(0.0);
    }
    // `rshift` / `lshift` collect. Keep `n` on the shadow stack so the
    // subsequent `eq` still reads the live digit array.
    let n = if sign < 0 {
        RBigIntGcRoot::new(value.neg())
    } else {
        live_rbigint(value)
    };

    let exp = n.bit_length()?;
    let shift = DBL_MANT_DIG + 2 - exp;
    let mut q;
    if shift >= 0 {
        q = n.ulonglongmask() << shift as u32;
    } else {
        let shift = (-shift) as u64;
        let n2 = RBigIntGcRoot::new(n.rshift(shift as i64, false)?);
        q = n2.ulonglongmask();
        // Shift before `n` is read: `lshift` collects.
        let shifted = n2.lshift(shift as i64)?;
        if !n.eq(&shifted) {
            q |= 1;
        }
    }

    q = (q >> 2) + ((q & 2 != 0 && q & 5 != 0) as u64);
    if exp > DBL_MAX_EXP || (exp == DBL_MAX_EXP && q == 1_u64 << DBL_MANT_DIG) {
        return Err(RBigIntError::Overflow);
    }

    let mut result = _float_ldexp(q as f64, exp - DBL_MANT_DIG);
    if sign < 0 {
        result = -result;
    }
    Ok(result)
}

/// rbigint.py `@specialize.arg(0) _loghelper(log, arg)`.
///
/// RPython emits one graph per constant function argument.  These three
/// bodies are those concrete graphs; there must be no runtime discriminator.
fn _loghelper_ln(arg: &RBigInt) -> Result<f64, RBigIntError> {
    let (x, exponent) = _AsScaledDouble(arg);
    if x <= 0.0 {
        return Err(RBigIntError::LogDomain);
    }
    Ok(x.ln() + exponent as f64 * SHIFT as f64 * 2.0_f64.ln())
}

fn _loghelper_log10(arg: &RBigInt) -> Result<f64, RBigIntError> {
    let (x, exponent) = _AsScaledDouble(arg);
    if x <= 0.0 {
        return Err(RBigIntError::LogDomain);
    }
    Ok(x.log10() + exponent as f64 * SHIFT as f64 * 2.0_f64.log10())
}

fn _loghelper_log2(arg: &RBigInt) -> Result<f64, RBigIntError> {
    let (x, exponent) = _AsScaledDouble(arg);
    if x <= 0.0 {
        return Err(RBigIntError::LogDomain);
    }
    Ok(x.log2() + exponent as f64 * SHIFT as f64 * 2.0_f64.log2())
}

/// rbigint.py `bits_in_digit`.
const fn make_bit_length_table() -> [u8; 1 << 8] {
    let mut table = [0_u8; 1 << 8];
    let mut i = 1;
    while i < table.len() {
        table[i] = 1 + table[i / 2];
        i += 1;
    }
    table
}

const BIT_LENGTH_TABLE: [u8; 1 << 8] = make_bit_length_table();

pub fn bits_in_digit(mut d: Digit) -> i64 {
    debug_assert!(d >= 0);
    let mut d_bits = 0;
    while d >= 1 << 8 {
        d_bits += 8;
        d >>= 8;
    }
    d_bits + BIT_LENGTH_TABLE[d as usize] as i64
}

/// rbigint.py `bit_length_int`.
#[majit_macros::jit_elidable]
pub fn bit_length_int(mut value: i64) -> i64 {
    let mut count;
    if value < 0 {
        // Upstream's overflow-safe transformation for the most-negative int.
        value = -((value + 1) >> 1);
        count = 1;
    } else {
        count = 0;
    }
    if value >= 1_i64 << 32 {
        value >>= 32;
        count += 32;
    }
    if value >= 1_i64 << 16 {
        value >>= 16;
        count += 16;
    }
    if value >= 1_i64 << 8 {
        value >>= 8;
        count += 8;
    }
    count + BIT_LENGTH_TABLE[value as usize] as i64
}

pub fn bit_count_digit(val: Digit) -> i64 {
    _bitcount64(val as u64)
}

#[majit_macros::jit_elidable]
fn _bitcount64(x: u64) -> i64 {
    _bitcount64_ops(x, BITCOUNT_K1, BITCOUNT_K2, BITCOUNT_K4, BITCOUNT_KF) as i64
}

#[majit_macros::always_inline]
fn _bitcount64_ops(mut x: u64, k1: u64, k2: u64, k4: u64, kf: u64) -> u64 {
    x -= (x >> 1) & k1;
    x = (x & k2) + ((x >> 2) & k2);
    x = (x + (x >> 4)) & k4;
    x.wrapping_mul(kf) >> 56
}

/// rbigint.py `_truediv_result`.
#[inline]
fn _truediv_result(result: f64, negate: bool) -> f64 {
    if negate { -result } else { result }
}

fn _truediv_overflow<T>() -> Result<T, RBigIntError> {
    Err(RBigIntError::FloatDivisionOverflow)
}

/// RPython's `math.ldexp(value, exponent)` spelling used by `_AsDouble` and
/// `_bigint_true_divide`.  Computing `value * 2.0.powi(exponent)` directly is
/// not equivalent: for a subnormal result the power-of-two factor can become
/// zero before it is multiplied by the exactly-representable mantissa.
#[inline]
fn _float_ldexp(mut value: f64, mut exponent: i64) -> f64 {
    while exponent > 1023 {
        value *= 2.0_f64.powi(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= 2.0_f64.powi(-1022);
        exponent += 1022;
    }
    value * 2.0_f64.powi(exponent as i32)
}

/// rbigint.py `_bigint_true_divide`.
fn _bigint_true_divide(a: &RBigInt, b: &RBigInt) -> Result<f64, RBigIntError> {
    const DBL_MANT_DIG: i64 = 53;
    const DBL_MAX_EXP: i64 = 1024;
    const DBL_MIN_EXP: i64 = -1021;
    const MANT_DIG_DIGITS: i64 = DBL_MANT_DIG / SHIFT;
    const MANT_DIG_BITS: i64 = DBL_MANT_DIG % SHIFT;

    // `lshift` / `rshift` / `_divrem` each allocate (`ll_newlist` /
    // `malloc_fast`). The operands' `_digits` must sit on the shadow stack
    // for the whole graph, matching the frame slots the GC transform would
    // emit around those calls.
    let a = live_rbigint(a);
    let b = live_rbigint(b);
    let negate = (a.get_sign() < 0) ^ (b.get_sign() < 0);
    if !b.tobool() {
        return Err(RBigIntError::DivisionByZero);
    }
    if !a.tobool() {
        return Ok(_truediv_result(0.0, negate));
    }
    let mut a_size = a.numdigits();
    let mut b_size = b.numdigits();
    let a_is_small = a_size <= MANT_DIG_DIGITS
        || (a_size == MANT_DIG_DIGITS + 1 && a.udigit(MANT_DIG_DIGITS) >> MANT_DIG_BITS == 0);
    let b_is_small = b_size <= MANT_DIG_DIGITS
        || (b_size == MANT_DIG_DIGITS + 1 && b.udigit(MANT_DIG_DIGITS) >> MANT_DIG_BITS == 0);
    if a_is_small && b_is_small {
        a_size -= 1;
        let mut da = a.digit(a_size) as f64;
        while a_size > 0 {
            a_size -= 1;
            da = da * FLOAT_MULTIPLIER + a.digit(a_size) as f64;
        }
        b_size -= 1;
        let mut db = b.digit(b_size) as f64;
        while b_size > 0 {
            b_size -= 1;
            db = db * FLOAT_MULTIPLIER + b.digit(b_size) as f64;
        }
        return Ok(_truediv_result(da / db, negate));
    }

    let mut diff = a_size - b_size;
    diff = diff * SHIFT + bits_in_digit(a.digit(a_size - 1)) - bits_in_digit(b.digit(b_size - 1));
    if diff > DBL_MAX_EXP {
        return _truediv_overflow();
    } else if diff < DBL_MIN_EXP - DBL_MANT_DIG - 1 {
        return Ok(_truediv_result(0.0, negate));
    }
    let shift = diff.max(DBL_MIN_EXP) - DBL_MANT_DIG - 2;
    let mut inexact = false;
    let x = if shift <= 0 {
        RBigIntGcRoot::new(a.lshift(-shift)?)
    } else {
        let x = RBigIntGcRoot::new(a.rshift(shift, true)?);
        // Shift before `a` is read: `lshift` collects.
        let shifted = x.lshift(shift)?;
        if !a.eq(&shifted) {
            inexact = true;
        }
        x
    };
    let (x, rem) = _divrem(&x, &b)?;
    if rem.tobool() {
        inexact = true;
    }
    debug_assert!(x.tobool());
    let mut x_size = x.numdigits();
    let x_bits = (x_size - 1) * SHIFT + bits_in_digit(x.digit(x_size - 1));
    let extra_bits = x_bits.max(DBL_MIN_EXP - shift) - DBL_MANT_DIG;
    debug_assert!(extra_bits == 2 || extra_bits == 3);
    let mask = 1_u64 << (extra_bits - 1);
    let mut low = x.udigit(0) | inexact as u64;
    if low & mask != 0 && low & (3 * mask - 1) != 0 {
        low += mask;
    }
    let x_digit_0 = low & !(mask - 1);
    x_size -= 1;
    let mut dx = 0.0;
    while x_size > 0 {
        dx += x.digit(x_size) as f64;
        dx *= FLOAT_MULTIPLIER;
        x_size -= 1;
    }
    dx += x_digit_0 as f64;
    if shift + x_bits >= DBL_MAX_EXP
        && (shift + x_bits > DBL_MAX_EXP || dx == 2_f64.powi(x_bits as i32))
    {
        return _truediv_overflow();
    }
    Ok(_truediv_result(_float_ldexp(dx, shift), negate))
}

pub const BASE8: &str = "01234567";
pub const BASE10: &str = "0123456789";
pub const BASE16: &str = "0123456789abcdef";

/// rbigint.py `_format_base2_notzero`.
fn _format_base2_notzero(
    a: &RBigInt,
    digits: &str,
    prefix: &str,
    suffix: &str,
    _max_str_digits: i64,
) -> Result<String, RBigIntError> {
    let digit_bytes = digits.as_bytes();
    let base = digit_bytes.len() as i64;
    let mut accum = 0_u128;
    let mut accumbits = 0_i32;
    let mut basebits = 0_i32;
    let mut i = base;
    while i > 1 {
        basebits += 1;
        i >>= 1;
    }
    let size_a = a.numdigits();
    // rbigint.py allocates `[chr(0)] * i`; its translated allocation can
    // raise MemoryError.  Keep that edge instead of allowing Rust sizing
    // arithmetic to wrap or an infallible allocation to abort the process.
    let payload_bits = size_a
        .checked_mul(SHIFT)
        .and_then(|value| value.checked_add(basebits as i64 - 1))
        .ok_or(RBigIntError::Memory)?;
    let encoded_digits = payload_bits / basebits as i64;
    let capacity = 5_i64
        .checked_add(i64::try_from(prefix.len()).map_err(|_| RBigIntError::Memory)?)
        .and_then(|value| value.checked_add(suffix.len() as i64))
        .and_then(|value| value.checked_add(encoded_digits))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(RBigIntError::Memory)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| RBigIntError::Memory)?;
    result.resize(capacity, 0_u8);
    let mut next_char_index = capacity;
    let mut j = suffix.len();
    while j > 0 {
        next_char_index -= 1;
        j -= 1;
        result[next_char_index] = suffix.as_bytes()[j];
    }
    i = 0;
    while i < size_a {
        accum |= a.uwidedigit(i) << accumbits as u32;
        accumbits += SHIFT as i32;
        debug_assert!(accumbits >= basebits);
        loop {
            let cdigit = (accum & (base as u128 - 1)) as usize;
            next_char_index -= 1;
            result[next_char_index] = digit_bytes[cdigit];
            accumbits -= basebits;
            accum >>= basebits as u32;
            if i < size_a - 1 {
                if accumbits < basebits {
                    break;
                }
            } else if accum == 0 {
                break;
            }
        }
        i += 1;
    }
    j = prefix.len();
    while j > 0 {
        next_char_index -= 1;
        j -= 1;
        result[next_char_index] = prefix.as_bytes()[j];
    }
    if a.get_sign() < 0 {
        next_char_index -= 1;
        result[next_char_index] = b'-';
    }
    result.copy_within(next_char_index.., 0);
    result.truncate(capacity - next_char_index);
    Ok(String::from_utf8(result).expect("rbigint format digit alphabets are ASCII"))
}

struct PartsCacheBase {
    mindigits: i64,
    lowest_part: i64,
    // rbigint.py stores a list of rbigint object references, not inline
    // value copies.  The outer Arc is pyre's immutable-reader snapshot of that
    // one shared list; each element Arc preserves the identity of the cached
    // rbigint while a later append publishes a longer snapshot.
    //
    // The mutex is only the free-threaded synchronization envelope around
    // PyPy's mutable list.  Readers clone one Arc, rather than cloning the
    // complete list on every formatting call, and never retain this lock
    // across a bigint allocation/GC safepoint.
    parts_cache: parking_lot::Mutex<std::sync::Arc<Vec<std::sync::Arc<RBigInt>>>>,
}

impl PartsCacheBase {
    fn base_parameters(base: i64) -> (i64, i64) {
        let mut mindigits = 1;
        let mut curr = base;
        while let Some(next) = curr.checked_mul(base) {
            if next == MASK {
                break;
            }
            curr = next;
            mindigits += 1;
        }
        (mindigits, curr)
    }

    fn new(base: i64) -> Self {
        let (mindigits, curr) = Self::base_parameters(base);
        Self {
            mindigits,
            lowest_part: curr,
            parts_cache: parking_lot::Mutex::new(std::sync::Arc::new(vec![std::sync::Arc::new(
                RBigInt::fromint(curr),
            )])),
        }
    }

    /// Build the translated prebuilt `_parts_cache_10` value.
    ///
    /// In rbigint.py this object is constructed at module import and
    /// belongs to the translated prebuilt root graph.  Its first digit array
    /// therefore must not be a nursery allocation lazily attempted from the
    /// collector's first root walk.
    fn new_prebuilt_decimal() -> Self {
        let (mindigits, curr) = Self::base_parameters(10);
        let block = unsafe { Digits::prebuilt(curr) };
        let part = RBigInt {
            _digits: block,
            _size: 1,
        };
        Self {
            mindigits,
            lowest_part: curr,
            parts_cache: parking_lot::Mutex::new(std::sync::Arc::new(vec![std::sync::Arc::new(
                part,
            )])),
        }
    }

    #[inline]
    fn parts_snapshot(&self) -> std::sync::Arc<Vec<std::sync::Arc<RBigInt>>> {
        self.parts_cache.lock().clone()
    }

    /// Publish `expected[-1] ** 2` exactly as `_format` appends to
    /// `pcb.parts_cache`.  A competing formatter may win the append; in that
    /// case its longer snapshot is already the shared PyPy list state.
    fn append_square(
        &self,
        expected: &std::sync::Arc<Vec<std::sync::Arc<RBigInt>>>,
    ) -> Result<(), RBigIntError> {
        let last = expected.last().expect("parts cache starts non-empty");

        // Do every bigint/GC allocation outside the list lock.  MiniMark's
        // STW root walker locks this same shared list to forward `_digits`.
        let next = std::sync::Arc::new(last.int_pow(2, None)?);
        let _next_root = unsafe { PendingPartsCacheDigitRoot::new(&next) };
        let extended_len = expected.len().checked_add(1).ok_or(RBigIntError::Memory)?;
        let mut extended = Vec::new();
        extended
            .try_reserve_exact(extended_len)
            .map_err(|_| RBigIntError::Memory)?;
        extended.extend(expected.iter().cloned());
        extended.push(next);
        let extended = std::sync::Arc::new(extended);

        let mut shared = self.parts_cache.lock();
        if std::sync::Arc::ptr_eq(&shared, expected) {
            *shared = extended;
        }
        Ok(())
    }
}

type PartsCacheRef = std::sync::Arc<PartsCacheBase>;

static PARTS_CACHE: std::sync::LazyLock<parking_lot::Mutex<Vec<Option<PartsCacheRef>>>> =
    std::sync::LazyLock::new(|| {
        // rbigint.py constructs the 34-slot owner and immediately
        // fills slot 10 - 3 through `_parts_cache_10`.  Preserve that eager
        // content rather than treating decimal like the other lazy bases.
        let mut parts_cache = vec![None; 34];
        parts_cache[10 - 3] = Some(std::sync::Arc::new(PartsCacheBase::new_prebuilt_decimal()));
        parking_lot::Mutex::new(parts_cache)
    });

/// Force construction of rbigint.py's module-global `_parts_cache` and
/// `_parts_cache_10` during runtime initialization.
pub fn initialize_rbigint_parts_cache() {
    std::sync::LazyLock::force(&PARTS_CACHE);
}

/// rbigint.py `_PartsCache.get_cached_parts`.
///
/// Upstream owns one process-global 34-slot list. The mutex is only Rust's
/// synchronization wrapper around that same shared owner; this must never be
/// changed to TLS or a per-call cache.
fn get_cached_parts(base: i64) -> PartsCacheRef {
    let index = base - 3;
    {
        let all = PARTS_CACHE.lock();
        if let Some(cache) = &all[index as usize] {
            return cache.clone();
        }
    }

    // Construct outside the owner lock: `fromint` allocates a digit array and
    // the collector walks this same process-global owner.
    let initial = std::sync::Arc::new(PartsCacheBase::new(base));
    let initial_part = {
        let parts = initial.parts_cache.lock();
        parts[0].clone()
    };
    let _initial_root = unsafe { PendingPartsCacheDigitRoot::new(&initial_part) };
    let mut all = PARTS_CACHE.lock();
    all[index as usize].get_or_insert(initial).clone()
}

fn _format_int_general(mut val: i64, digits: &str) -> String {
    let digit_bytes = digits.as_bytes();
    let base = digit_bytes.len() as i64;
    let mut out = Vec::new();
    while val != 0 {
        out.push(digit_bytes[(val % base) as usize]);
        val /= base;
    }
    out.reverse();
    String::from_utf8(out).expect("format digits are ASCII")
}

#[inline]
fn _format_int10(val: i64, _digits: &str) -> String {
    val.to_string()
}

const fn make_format10_table2() -> [u8; 200] {
    let mut result = [0_u8; 200];
    let mut i = 0;
    while i < 100 {
        result[2 * i] = b'0' + (i / 10) as u8;
        result[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    result
}

const FORMAT10_TABLE2: [u8; 200] = make_format10_table2();

/// rbigint.py `_format_int10_18digits`.
fn _format_int10_18digits(mut val: i64, builder: &mut String) {
    debug_assert!(val < 10_i64.pow(18));
    let top2 = val / 10_i64.pow(16);
    val %= 10_i64.pow(16);
    let a = val / 10_i64.pow(8);
    let b = val % 10_i64.pow(8);
    let aa = a / 10_i64.pow(4);
    let ab = a % 10_i64.pow(4);
    let ba = b / 10_i64.pow(4);
    let bb = b % 10_i64.pow(4);
    let pairs = [
        top2,
        aa / 100,
        aa % 100,
        ab / 100,
        ab % 100,
        ba / 100,
        ba % 100,
        bb / 100,
        bb % 100,
    ];
    for pair in pairs {
        let index = 2 * pair as usize;
        builder.push(FORMAT10_TABLE2[index] as char);
        builder.push(FORMAT10_TABLE2[index + 1] as char);
    }
}

/// rbigint.py
/// `@specialize.arg(6) _format_recursive(..., _format_int, ...)`.
///
/// This is the `_format_int10` graph.  Keeping it separate from the general
/// graph preserves RPython's constant callable identity checks and removes a
/// runtime discriminator from recursive calls.
#[expect(
    clippy::too_many_arguments,
    reason = "The signature is the direct specialized _format_recursive graph from rbigint.py; callable identity and argument order are part of RPython specialization parity"
)]
fn _format_recursive_decimal(
    x: RBigInt,
    mut i: i64,
    output: &mut String,
    pcb: &PartsCacheBase,
    pts: &[std::sync::Arc<RBigInt>],
    digits: &str,
    size_prefix: i64,
    max_str_digits: i64,
) -> Result<(), RBigIntError> {
    // `x` stays live across `divmod` and across the recursive call, both of
    // which can now collect.  RPython's stack map roots it there; a Rust local
    // has no stack map, so it is rooted explicitly for the whole body.  `top`
    // is moved into the callee, which roots it the same way, and `pts`/`pcb`
    // are forwarded by `walk_rbigint_cache_digit_slots`.
    let mut x = crate::rbigint::gc::RBigIntGcRoot::new(x);
    while i > 0 {
        let (top, low) = x.divmod(&pts[i as usize])?;
        x.set(low);
        if top.tobool() || output.len() as i64 != size_prefix {
            _format_recursive_decimal(
                top,
                i - 1,
                output,
                pcb,
                pts,
                digits,
                size_prefix,
                max_str_digits,
            )?;
        }
        i -= 1;
    }
    let mindigits = pcb.mindigits;
    let mut curlen = output.len() as i64;
    let (high, low) = _format_lowest_level_divmod_int_results(&x, pcb.lowest_part);
    let mut lowdone = false;
    if curlen == size_prefix {
        if high != 0 {
            let s = _format_int10(high, digits);
            output.push_str(&s);
            curlen += s.len() as i64;
        } else {
            if low != 0 {
                let s = _format_int10(low, digits);
                output.push_str(&s);
                curlen += s.len() as i64;
            }
            lowdone = true;
        }
    } else {
        if mindigits == 18 {
            _format_int10_18digits(high, output);
        } else {
            let s = _format_int10(high, digits);
            for _ in s.len() as i64..mindigits {
                output.push(digits.as_bytes()[0] as char);
            }
            output.push_str(&s);
        }
        curlen += mindigits;
    }
    if !lowdone {
        if mindigits == 18 {
            _format_int10_18digits(low, output);
        } else {
            let s = _format_int10(low, digits);
            for _ in s.len() as i64..mindigits {
                output.push(digits.as_bytes()[0] as char);
            }
            output.push_str(&s);
        }
        curlen += mindigits;
    }
    if max_str_digits > 0 && curlen - size_prefix > max_str_digits {
        return Err(RBigIntError::MaxStrDigits);
    }
    Ok(())
}

/// The `_format_int_general` graph emitted by upstream's
/// `@specialize.arg(6) _format_recursive`.
#[expect(
    clippy::too_many_arguments,
    reason = "The signature is the direct specialized _format_recursive graph from rbigint.py; grouping state would obscure recursive-call and specialization parity"
)]
fn _format_recursive_general(
    x: RBigInt,
    mut i: i64,
    output: &mut String,
    pcb: &PartsCacheBase,
    pts: &[std::sync::Arc<RBigInt>],
    digits: &str,
    size_prefix: i64,
    max_str_digits: i64,
) -> Result<(), RBigIntError> {
    // `x` stays live across `divmod` and across the recursive call, both of
    // which can now collect.  RPython's stack map roots it there; a Rust local
    // has no stack map, so it is rooted explicitly for the whole body.  `top`
    // is moved into the callee, which roots it the same way, and `pts`/`pcb`
    // are forwarded by `walk_rbigint_cache_digit_slots`.
    let mut x = crate::rbigint::gc::RBigIntGcRoot::new(x);
    while i > 0 {
        let (top, low) = x.divmod(&pts[i as usize])?;
        x.set(low);
        if top.tobool() || output.len() as i64 != size_prefix {
            _format_recursive_general(
                top,
                i - 1,
                output,
                pcb,
                pts,
                digits,
                size_prefix,
                max_str_digits,
            )?;
        }
        i -= 1;
    }
    let mindigits = pcb.mindigits;
    let mut curlen = output.len() as i64;
    let (high, low) = _format_lowest_level_divmod_int_results(&x, pcb.lowest_part);
    let mut lowdone = false;
    if curlen == size_prefix {
        if high != 0 {
            let s = _format_int_general(high, digits);
            output.push_str(&s);
            curlen += s.len() as i64;
        } else {
            if low != 0 {
                let s = _format_int_general(low, digits);
                output.push_str(&s);
                curlen += s.len() as i64;
            }
            lowdone = true;
        }
    } else {
        let s = _format_int_general(high, digits);
        for _ in s.len() as i64..mindigits {
            output.push(digits.as_bytes()[0] as char);
        }
        output.push_str(&s);
        curlen += mindigits;
    }
    if !lowdone {
        let s = _format_int_general(low, digits);
        for _ in s.len() as i64..mindigits {
            output.push(digits.as_bytes()[0] as char);
        }
        output.push_str(&s);
        curlen += mindigits;
    }
    if max_str_digits > 0 && curlen - size_prefix > max_str_digits {
        return Err(RBigIntError::MaxStrDigits);
    }
    Ok(())
}

#[majit_macros::always_inline]
fn _format_lowest_level_divmod_int_results(x: &RBigInt, iother: i64) -> (i64, i64) {
    if !x.tobool() {
        return (0, 0);
    }
    debug_assert!(iother > 0);
    let size = x.numdigits() - 1;
    let mut rem = if size == 1 {
        let rem = x.uwidedigit(1);
        debug_assert!(rem < iother as UWideDigit);
        rem
    } else {
        0
    };
    rem = (rem << SHIFT) | x.uwidedigit(0);
    let div = rem / iother as UWideDigit;
    rem -= div * iother as UWideDigit;
    (div as i64, rem as i64)
}

/// rbigint.py `_format`.
fn _format(
    x: &RBigInt,
    digits: &str,
    prefix: &str,
    suffix: &str,
    max_str_digits: i64,
) -> Result<String, RBigIntError> {
    if x.get_sign() == 0 {
        let capacity = prefix
            .len()
            .checked_add(1)
            .and_then(|value| value.checked_add(suffix.len()))
            .ok_or(RBigIntError::Memory)?;
        let mut output = String::new();
        output
            .try_reserve_exact(capacity)
            .map_err(|_| RBigIntError::Memory)?;
        output.push_str(prefix);
        output.push('0');
        output.push_str(suffix);
        return Ok(output);
    }
    let base = digits.len() as i64;
    if !(2..=36).contains(&base) {
        return Err(RBigIntError::InvalidBase);
    }
    if base & (base - 1) == 0 {
        return _format_base2_notzero(x, digits, prefix, suffix, max_str_digits);
    }
    let negative = x.get_sign() < 0;
    // `append_square` and the recursive formatter allocate. Keep `x` on the
    // shadow stack for every `lt` / `divmod` that follows.
    let x = if negative {
        RBigIntGcRoot::new(x.neg())
    } else {
        live_rbigint(x)
    };
    let pcb = get_cached_parts(base);
    let mindigits = pcb.mindigits;
    let mut stringsize = mindigits;
    let mut pts = pcb.parts_snapshot();

    // rbigint.py grows the one shared list by repeated squaring.
    // Refreshing the immutable Arc snapshot after each attempted append is the
    // free-threaded equivalent of observing `pts.append(...)` in place.
    while pts
        .last()
        .expect("parts cache starts non-empty")
        .as_ref()
        .lt(&x)
    {
        pcb.append_square(&pts)?;
        pts = pcb.parts_snapshot();
    }

    let mut startindex = 0;
    while startindex < pts.len() as i64 && pts[startindex as usize].as_ref().lt(&x) {
        stringsize = stringsize.checked_mul(2).ok_or(RBigIntError::Memory)?;
        startindex += 1;
    }
    startindex -= 1;
    stringsize = stringsize
        .checked_add(i64::try_from(prefix.len()).map_err(|_| RBigIntError::Memory)?)
        .and_then(|value| value.checked_add(suffix.len() as i64))
        .and_then(|value| value.checked_add(negative as i64))
        .ok_or(RBigIntError::Memory)?;
    let capacity = usize::try_from(stringsize).map_err(|_| RBigIntError::Memory)?;
    let mut output = String::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| RBigIntError::Memory)?;
    if negative {
        output.push('-');
    }
    output.push_str(prefix);
    if startindex < 0 {
        let value = x.toint()?;
        if digits == BASE10 {
            output.push_str(&_format_int10(value, digits));
        } else {
            output.push_str(&_format_int_general(value, digits));
        }
    } else {
        let size_prefix = output.len() as i64;
        if digits == BASE10 {
            _format_recursive_decimal(
                x.translated_alias(),
                startindex,
                &mut output,
                &pcb,
                &pts,
                digits,
                size_prefix,
                max_str_digits,
            )?;
        } else {
            _format_recursive_general(
                x.translated_alias(),
                startindex,
                &mut output,
                &pcb,
                &pts,
                digits,
                size_prefix,
                max_str_digits,
            )?;
        }
    }
    output.push_str(suffix);
    Ok(output)
}

/// The operand prologue of `rbigint.py` `_bitwise`:
/// `if a.get_sign() < 0: a = a.invert(); maska = MASK`, then the same for
/// `b`. Each `invert` collects, so the other operand is held on an
/// owner-root slot across it.
fn bitwise_prep_operands(a: &RBigInt, b: &RBigInt) -> (RBigIntGcRoot, RBigIntGcRoot, Digit, Digit) {
    let b = live_rbigint(b);
    let (a, maska) = if a.get_sign() < 0 {
        (RBigIntGcRoot::new(a.invert()), MASK as Digit)
    } else {
        (live_rbigint(a), 0)
    };
    let (b, maskb) = if b.get_sign() < 0 {
        (RBigIntGcRoot::new(b.invert()), MASK as Digit)
    } else {
        (b, 0)
    };
    (a, b, maska, maskb)
}

/// rbigint.py `@specialize.arg(1) _bitwise(a, '&', b)`.
fn _bitwise_and(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let (a, b, mut maska, mut maskb) = bitwise_prep_operands(a, b);

    let mut negz = false;
    let mut op_is_and = true;
    if maska != 0 && maskb != 0 {
        op_is_and = false;
        maska ^= MASK as Digit;
        maskb ^= MASK as Digit;
        negz = true;
    }

    let size_a = a.numdigits();
    let size_b = b.numdigits();
    let size_z = if op_is_and {
        if maska != 0 {
            size_b
        } else if maskb != 0 {
            size_a
        } else {
            size_a.min(size_b)
        }
    } else {
        size_a.max(size_b)
    };
    // `a` and `b` are the livevars across `[NULLDIGIT] * size_z`; the loop
    // reads the reloaded plain handles.
    let mut a = a.translated_alias();
    let mut b = b.translated_alias();
    let mut z = RBigInt::with_size_livevars(size_z, 1, &mut a, &mut b);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i < size_b {
            b.digit(i) ^ maskb
        } else {
            maskb
        };
        let value = if op_is_and { diga & digb } else { diga | digb };
        z.setdigit(i, value);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// The `'|'` graph emitted by upstream's `@specialize.arg(1) _bitwise`.
fn _bitwise_or(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let (a, b, mut maska, mut maskb) = bitwise_prep_operands(a, b);

    let mut negz = false;
    let mut op_is_and = false;
    if maska != 0 || maskb != 0 {
        op_is_and = true;
        maska ^= MASK as Digit;
        maskb ^= MASK as Digit;
        negz = true;
    }

    let size_a = a.numdigits();
    let size_b = b.numdigits();
    let size_z = if op_is_and {
        if maska != 0 {
            size_b
        } else if maskb != 0 {
            size_a
        } else {
            size_a.min(size_b)
        }
    } else {
        size_a.max(size_b)
    };
    // `a` and `b` are the livevars across `[NULLDIGIT] * size_z`; the loop
    // reads the reloaded plain handles.
    let mut a = a.translated_alias();
    let mut b = b.translated_alias();
    let mut z = RBigInt::with_size_livevars(size_z, 1, &mut a, &mut b);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i < size_b {
            b.digit(i) ^ maskb
        } else {
            maskb
        };
        let value = if op_is_and { diga & digb } else { diga | digb };
        z.setdigit(i, value);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// The `'^'` graph emitted by upstream's `@specialize.arg(1) _bitwise`.
fn _bitwise_xor(a: &RBigInt, b: &RBigInt) -> RBigInt {
    let (a, b, mut maska, maskb) = bitwise_prep_operands(a, b);

    let mut negz = false;
    if maska != maskb {
        maska ^= MASK as Digit;
        negz = true;
    }

    let size_a = a.numdigits();
    let size_b = b.numdigits();
    let size_z = size_a.max(size_b);
    // `a` and `b` are the livevars across `[NULLDIGIT] * size_z`; the loop
    // reads the reloaded plain handles.
    let mut a = a.translated_alias();
    let mut b = b.translated_alias();
    let mut z = RBigInt::with_size_livevars(size_z, 1, &mut a, &mut b);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i < size_b {
            b.digit(i) ^ maskb
        } else {
            maskb
        };
        z.setdigit(i, diga ^ digb);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// rbigint.py
/// `@specialize.arg(1) _int_bitwise(a, '&', b)`.
fn _int_bitwise_and(a: &RBigInt, mut b: i64) -> RBigInt {
    // `_bitwise_and` already roots both operands; this graph's `fromint`
    // and `with_size` (`ll_newlist` → `malloc_fast`) are the same collecting
    // allocations, so `a`'s `_digits` sits in `RBigIntGcRoot` here too.
    let a_live = live_rbigint(a);
    if !int_in_valid_range(b) {
        let other = RBigInt::fromint(b);
        return _bitwise_and(&a_live, &other);
    }
    let a_inverted = (a_live.get_sign() < 0).then(|| RBigIntGcRoot::new(a_live.invert()));
    let mut maska = if a_inverted.is_some() {
        MASK as Digit
    } else {
        0
    };
    let mut maskb = if b < 0 {
        b = !b;
        MASK as Digit
    } else {
        0
    };

    let mut negz = false;
    let mut op_is_and = true;
    if maska != 0 && maskb != 0 {
        op_is_and = false;
        maska ^= MASK as Digit;
        maskb ^= MASK as Digit;
        negz = true;
    }

    let size_a = a_inverted.as_deref().unwrap_or(&a_live).numdigits();
    let size_z = if op_is_and {
        if maska != 0 {
            1
        } else if maskb != 0 {
            size_a
        } else {
            1
        }
    } else {
        size_a
    };
    let mut z = RBigInt::with_size(size_z, 1);
    // `a = a.invert()` rebinds `a`; the handle is taken after the
    // collecting malloc so the digit loop reads the forwarded array.
    let a: &RBigInt = a_inverted.as_deref().unwrap_or(&a_live);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i == 0 { b ^ maskb } else { maskb };
        let value = if op_is_and { diga & digb } else { diga | digb };
        z.setdigit(i, value);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// The `'|'` graph emitted by upstream's `@specialize.arg(1) _int_bitwise`.
fn _int_bitwise_or(a: &RBigInt, mut b: i64) -> RBigInt {
    let a_live = live_rbigint(a);
    if !int_in_valid_range(b) {
        let other = RBigInt::fromint(b);
        return _bitwise_or(&a_live, &other);
    }
    let a_inverted = (a_live.get_sign() < 0).then(|| RBigIntGcRoot::new(a_live.invert()));
    let mut maska = if a_inverted.is_some() {
        MASK as Digit
    } else {
        0
    };
    let mut maskb = if b < 0 {
        b = !b;
        MASK as Digit
    } else {
        0
    };

    let mut negz = false;
    let mut op_is_and = false;
    if maska != 0 || maskb != 0 {
        op_is_and = true;
        maska ^= MASK as Digit;
        maskb ^= MASK as Digit;
        negz = true;
    }

    let size_a = a_inverted.as_deref().unwrap_or(&a_live).numdigits();
    let size_z = if op_is_and {
        if maska != 0 {
            1
        } else if maskb != 0 {
            size_a
        } else {
            1
        }
    } else {
        size_a
    };
    let mut z = RBigInt::with_size(size_z, 1);
    // `a = a.invert()` rebinds `a`; the handle is taken after the
    // collecting malloc so the digit loop reads the forwarded array.
    let a: &RBigInt = a_inverted.as_deref().unwrap_or(&a_live);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i == 0 { b ^ maskb } else { maskb };
        let value = if op_is_and { diga & digb } else { diga | digb };
        z.setdigit(i, value);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// The `'^'` graph emitted by upstream's `@specialize.arg(1) _int_bitwise`.
fn _int_bitwise_xor(a: &RBigInt, mut b: i64) -> RBigInt {
    let a_live = live_rbigint(a);
    if !int_in_valid_range(b) {
        let other = RBigInt::fromint(b);
        return _bitwise_xor(&a_live, &other);
    }
    let a_inverted = (a_live.get_sign() < 0).then(|| RBigIntGcRoot::new(a_live.invert()));
    let mut maska = if a_inverted.is_some() {
        MASK as Digit
    } else {
        0
    };
    let maskb = if b < 0 {
        b = !b;
        MASK as Digit
    } else {
        0
    };

    let mut negz = false;
    if maska != maskb {
        maska ^= MASK as Digit;
        negz = true;
    }

    let size_a = a_inverted.as_deref().unwrap_or(&a_live).numdigits();
    let size_z = size_a;
    let mut z = RBigInt::with_size(size_z, 1);
    // `a = a.invert()` rebinds `a`; the handle is taken after the
    // collecting malloc so the digit loop reads the forwarded array.
    let a: &RBigInt = a_inverted.as_deref().unwrap_or(&a_live);
    let mut i = 0;
    while i < size_z {
        let diga = if i < size_a {
            a.digit(i) ^ maska
        } else {
            maska
        };
        let digb = if i == 0 { b ^ maskb } else { maskb };
        z.setdigit(i, diga ^ digb);
        i += 1;
    }
    z._normalize();
    if negz { z.invert() } else { z }
}

/// rbigint.py `_AsLongLong`.
#[allow(non_snake_case)]
fn _AsLongLong(value: &RBigInt) -> Result<i64, RBigIntError> {
    let x = _AsULonglong_ignore_sign(value)?;
    if x >= ULONGLONG_BOUND {
        if x == ULONGLONG_BOUND && value.get_sign() < 0 {
            Ok(-9223372036854775807_i64 - 1)
        } else {
            Err(RBigIntError::Overflow)
        }
    } else if value.get_sign() < 0 {
        Ok(-(x as i64))
    } else {
        Ok(x as i64)
    }
}

/// rbigint.py `_AsULonglong_ignore_sign`.
#[allow(non_snake_case)]
fn _AsULonglong_ignore_sign(value: &RBigInt) -> Result<u64, RBigIntError> {
    let mut x = 0_u64;
    let mut i = value.numdigits();
    while i > 0 {
        i -= 1;
        let previous = x;
        x = x.wrapping_shl(SHIFT as u32).wrapping_add(value.udigit(i));
        if (x >> SHIFT) != previous {
            return Err(RBigIntError::Overflow);
        }
    }
    Ok(x)
}

/// rbigint.py `make_unsigned_mask_conversion`.
///
/// Both RPython instantiations (`r_uint` and `r_ulonglong`) are 64-bit on
/// pyre's supported target, so they specialize to this one concrete graph.
fn make_unsigned_mask_conversion(value: &RBigInt) -> u64 {
    _As_unsigned_mask(value)
}

#[allow(non_snake_case)]
fn _As_unsigned_mask(value: &RBigInt) -> u64 {
    let mut x = 0_u64;
    let mut i = value.numdigits();
    while i > 0 {
        i -= 1;
        x = x.wrapping_shl(SHIFT as u32).wrapping_add(value.udigit(i));
    }
    if value.get_sign() < 0 {
        x.wrapping_neg()
    } else {
        x
    }
}

/// rbigint.py `_hash`.
fn _hash(v: &RBigInt) -> i64 {
    let mut i = v.numdigits();
    let sign = v.get_sign();
    let mut x = 0_u64;
    while i > 0 {
        i -= 1;
        x = (x << SHIFT) | (x >> (u64::BITS as i64 - SHIFT));
        let digit = v.udigit(i);
        let (sum, overflowed) = x.overflowing_add(digit);
        x = sum;
        if overflowed {
            x = x.wrapping_add(1);
        }
    }
    (x as i64).wrapping_mul(sign)
}

// rbigint.py — precomputed chunk sizes for non-power-of-two parsers.
pub const fn digits_max_for_base(base: i64) -> Digit {
    let mut dec_per_digit = 1_i64;
    let mut power = base as UWideDigit;
    while power < MASK as UWideDigit {
        dec_per_digit += 1;
        power *= base as UWideDigit;
    }
    dec_per_digit -= 1;

    let mut result = 1_u128;
    let mut i = 0;
    while i < dec_per_digit {
        result *= base as UWideDigit;
        i += 1;
    }
    result as Digit
}

const fn make_base_max() -> [Digit; 37] {
    let mut result = [0_i64; 37];
    result[1] = 1;
    let mut base = 2;
    while base < result.len() {
        result[base] = digits_max_for_base(base as i64);
        base += 1;
    }
    result
}

pub const BASE_MAX: [Digit; 37] = make_base_max();
pub const DEC_MAX: Digit = digits_max_for_base(10);

/// rbigint.py `_decimalstr_to_bigint`.
fn _decimalstr_to_bigint(s: &str, start: i64, lim: i64) -> RBigInt {
    // Like upstream, this helper is only for a decimal string that has
    // already been parsed and validated. Validation belongs to
    // NumberStringParser, not this arithmetic graph.
    debug_assert!(s.is_ascii());
    debug_assert!(start < lim && lim <= s.len() as i64);
    let bytes = s.as_bytes();
    let mut p = start;
    let mut negative = false;
    if bytes[p as usize] == b'-' {
        negative = true;
        p += 1;
    } else if bytes[p as usize] == b'+' {
        p += 1;
    }
    debug_assert!(p < lim);

    let mut a = RBigInt::zero();
    let mut tens = 1_i64;
    let mut dig = 0_i64;
    while p < lim {
        let c = bytes[p as usize];
        debug_assert!(c.is_ascii_digit());
        dig = dig * 10 + (c - b'0') as i64;
        p += 1;
        tens *= 10;
        if tens == DEC_MAX || p == lim {
            a = _muladd1(&a, tens, dig);
            tens = 1;
            dig = 0;
        }
    }
    if negative && a.get_sign() == 1 {
        a._set_sign(-1);
    }
    a
}

/// rbigint.py `parse_digit_string`.
pub fn parse_digit_string(parser: &mut NumberStringParser<'_>) -> Result<RBigInt, RBigIntError> {
    let base = parser.base;
    if base >= 2 && (base & (base - 1)) == 0 {
        return parse_string_from_binary_base(parser);
    }
    if base == 10 && parser.end - parser.start > holder_limit(&HOLDER.MINSIZE_STR2INT) {
        let (s, start, end) = parser._all_digits10()?;
        let mut a =
            _str_to_int_big_base10(s.as_ref(), start, end, holder_limit(&HOLDER.STR2INT_LIMIT))?;
        a._set_sign(a.get_sign() * parser.sign);
        return Ok(a);
    }

    let mut a = RBigInt::zero();
    let digitmax = BASE_MAX[base as usize];
    let mut baseexp = 1_i64;
    let mut dig = 0_i64;
    loop {
        let digit = parser.next_digit()?;
        if baseexp == digitmax || digit < 0 {
            a = _muladd1(&a, baseexp, dig);
            if digit < 0 {
                break;
            }
            dig = digit;
            baseexp = base;
        } else {
            dig = dig * base + digit;
            baseexp *= base;
        }
    }
    a._set_sign(a.get_sign() * parser.sign);
    Ok(a)
}

/// The upstream object is a transient `dict<int, rbigint>` local to one
/// conversion.  The recursive algorithm creates only a small number of
/// entries (13 for the upstream 6000-digit test), so a direct vector of
/// key/value pairs preserves that ownership without introducing a side-table
/// or process-global map.
struct FivePowCache {
    entries: Vec<(i64, RBigIntGcRoot)>,
}

impl FivePowCache {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn get(&self, key: i64) -> Option<RBigInt> {
        self.entries
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, value)| value.translated_alias())
    }

    fn contains(&self, key: i64) -> bool {
        self.entries.iter().any(|(candidate, _)| *candidate == key)
    }

    fn insert(&mut self, key: i64, value: RBigInt) {
        debug_assert!(!self.contains(key));
        self.entries.push((key, RBigIntGcRoot::new(value)));
    }
}

/// rbigint.py `_str_to_int_big_w5pow`.
fn _str_to_int_big_w5pow(
    w: i64,
    mem: &mut FivePowCache,
    limit: i64,
) -> Result<RBigInt, RBigIntError> {
    if let Some(result) = mem.get(w) {
        return Ok(result);
    }
    let result = if w <= limit {
        RBigInt::five().int_pow(w, None)?
    } else if w > 0 && mem.contains(w - 1) {
        let prev = RBigIntGcRoot::new(mem.get(w - 1).expect("contains was checked above"));
        prev.int_mul(5)
    } else {
        let w2 = w >> 1;
        let smaller = RBigIntGcRoot::new(_str_to_int_big_w5pow(w2, mem, limit)?);
        let larger = _str_to_int_big_w5pow(w - w2, mem, limit)?;
        smaller.mul(&larger)
    };
    mem.insert(w, result.translated_alias());
    Ok(result)
}

/// rbigint.py `_str_to_int_big_inner10`.
fn _str_to_int_big_inner10(
    s: &str,
    a: i64,
    b: i64,
    mem: &mut FivePowCache,
    limit: i64,
) -> Result<RBigInt, RBigIntError> {
    let diff = b - a;
    if diff <= limit {
        return Ok(_decimalstr_to_bigint(s, a, b));
    }
    let mid = a + (diff + 1) / 2;
    let right = RBigIntGcRoot::new(_str_to_int_big_inner10(s, mid, b, mem, limit)?);
    let mut left = RBigIntGcRoot::new(_str_to_int_big_inner10(s, a, mid, mem, limit)?);
    // A method call derefs its receiver before its argument runs, and
    // `_str_to_int_big_w5pow` collects: run it first so `left` reloads after.
    let w5pow = RBigIntGcRoot::new(_str_to_int_big_w5pow(b - mid, mem, limit)?);
    left = RBigIntGcRoot::new(left.mul(&w5pow));
    left = RBigIntGcRoot::new(left.lshift(b - mid)?);
    Ok(right.add(&left))
}

/// rbigint.py `_str_to_int_big_base10`.
fn _str_to_int_big_base10(
    s: &str,
    start: i64,
    end: i64,
    limit: i64,
) -> Result<RBigInt, RBigIntError> {
    let mut mem = FivePowCache::new();
    _str_to_int_big_inner10(s, start, end, &mut mem, limit)
}

/// rbigint.py `parse_string_from_binary_base`.
fn parse_string_from_binary_base(
    parser: &mut NumberStringParser<'_>,
) -> Result<RBigInt, RBigIntError> {
    let bits_per_char: i64 = match parser.base {
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        _ => unreachable!("caller checks that the base is a supported power of two"),
    };

    let mut n = 0_i64;
    while parser.next_digit()? >= 0 {
        n = n.checked_add(1).ok_or(RBigIntError::ParseString)?;
    }
    let bits = n
        .checked_mul(bits_per_char)
        .and_then(|value| value.checked_add(SHIFT - 1))
        .ok_or(RBigIntError::ParseString)?;
    let b = (bits / SHIFT).max(1);
    let mut z = RBigInt::with_size(b, parser.sign);

    let mut accum = 0_u128;
    let mut bits_in_accum = 0_i64;
    let mut pdigit = 0_i64;
    for _ in 0..n {
        let k = parser.prev_digit()? as u128;
        accum |= k << bits_in_accum as u32;
        bits_in_accum += bits_per_char;
        if bits_in_accum >= SHIFT {
            z.setdigit_uwidedigit(pdigit, accum);
            pdigit += 1;
            debug_assert!(pdigit <= b);
            accum >>= SHIFT;
            bits_in_accum -= SHIFT;
        }
    }
    if bits_in_accum != 0 {
        z.setdigit_uwidedigit(pdigit, accum);
    }
    z._normalize();
    Ok(z)
}

/// rbigint.py `gcd_binary`.
#[majit_macros::jit_elidable]
pub fn gcd_binary(mut a: i64, mut b: i64) -> i64 {
    debug_assert!(a >= 0 && b >= 0);
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let mut shift = 0;
    while (a | b) & 1 == 0 {
        a >>= 1;
        b >>= 1;
        shift += 1;
    }
    while a & 1 == 0 {
        a >>= 1;
    }
    while b & 1 == 0 {
        b >>= 1;
    }
    while a != b {
        let difference = (a - b).abs();
        b = a.min(b);
        a = difference;
        while a & 1 == 0 {
            a >>= 1;
        }
    }
    a << shift
}

/// rbigint.py `lehmer_xgcd`.
fn lehmer_xgcd(mut a: u64, mut b: u64) -> (i128, i128, i128, i128) {
    let (mut s_old, mut s_new) = (1_i128, 0_i128);
    let (mut t_old, mut t_new) = (0_i128, 1_i128);
    while b >> (SHIFT >> 1) != 0 {
        let q = a / b;
        let r = a % b;
        a = b;
        b = r;
        (s_old, s_new) = (s_new, s_old - q as i128 * s_new);
        (t_old, t_new) = (t_new, t_old - q as i128 * t_new);
    }
    (s_old, t_old, s_new, t_new)
}

/// rbigint.py `gcd_lehmer`.
#[majit_macros::jit_elidable]
fn gcd_lehmer(a: RBigInt, b: RBigInt) -> Result<RBigInt, RBigIntError> {
    let mut a = RBigIntGcRoot::new(a);
    let mut b = RBigIntGcRoot::new(b);
    if a.lt(&b) {
        std::mem::swap(&mut a, &mut b);
    }
    while b.numdigits() > 1 {
        let mut a_ms = a.udigit(a.numdigits() - 1);
        let mut x = 0_i64;
        while a_ms & (0xff_u64 << (SHIFT - 8)) == 0 {
            a_ms <<= 8;
            x += 8;
        }
        while a_ms & (1_u64 << (SHIFT - 1)) == 0 {
            a_ms <<= 1;
            x += 1;
        }
        a_ms |= a.udigit(a.numdigits() - 2) >> (SHIFT - x);

        let b_ms = if a.numdigits() == b.numdigits() {
            (b.udigit(b.numdigits() - 1) << x) | (b.udigit(b.numdigits() - 2) >> (SHIFT - x))
        } else if a.numdigits() == b.numdigits() + 1 {
            b.udigit(b.numdigits() - 1) >> (SHIFT - x)
        } else {
            0
        };
        if b_ms >> ((SHIFT + 1) >> 1) == 0 {
            let remainder = RBigIntGcRoot::new(a.r#mod(&b)?);
            std::mem::swap(&mut a, &mut b);
            b = remainder;
            continue;
        }

        let (s_old, t_old, s_new, t_new) = lehmer_xgcd(a_ms, b_ms);
        debug_assert!(
            [s_old, t_old, s_new, t_new]
                .iter()
                .all(|&v| i64::try_from(v).is_ok())
        );
        let a_s = RBigIntGcRoot::new(a.int_mul(s_new as i64));
        let b_t = RBigIntGcRoot::new(b.int_mul(t_new as i64));
        let n_a = RBigIntGcRoot::new(RBigIntGcRoot::new(a_s.add(&b_t)).abs());
        let a_s = RBigIntGcRoot::new(a.int_mul(s_old as i64));
        let b_t = RBigIntGcRoot::new(b.int_mul(t_old as i64));
        b = RBigIntGcRoot::new(RBigIntGcRoot::new(a_s.add(&b_t)).abs());
        a = n_a;
        if a.lt(&b) {
            std::mem::swap(&mut a, &mut b);
        }
    }
    if !b.tobool() {
        return Ok((*a).clone());
    }
    a = RBigIntGcRoot::new(a.r#mod(&b)?);
    Ok(RBigInt::fromint(gcd_binary(b.toint()?, a.toint()?)))
}

/// rbigint.py `frombytes_int`.
#[majit_macros::jit_elidable]
pub fn frombytes_int(bytes: &[u8], byteorder: &str, signed: bool) -> Result<i64, RBigIntError> {
    if byteorder != "big" && byteorder != "little" {
        return Err(RBigIntError::InvalidEndianness);
    }
    if bytes.is_empty() {
        return Ok(0);
    }
    let msb = if byteorder == "big" {
        bytes[0]
    } else {
        bytes[bytes.len() - 1]
    };
    let sign = if msb >= 0x80 && signed { -1 } else { 1 };
    let mut result = 0_u64;
    let mut bitpos = 0_i64;
    let mut pad_byte = 0xdead_u64;

    let mut offset = 0_i64;
    while offset < bytes.len() as i64 {
        let c = if byteorder == "big" {
            bytes[bytes.len() - 1 - offset as usize] as u64
        } else {
            bytes[offset as usize] as u64
        };
        if bitpos == u64::BITS as i64 - 8 {
            pad_byte = if signed && c & (1 << 7) != 0 {
                0xff
            } else {
                0x00
            };
        }
        if bitpos >= u64::BITS as i64 {
            if c != pad_byte {
                return Err(RBigIntError::Overflow);
            }
        } else {
            result |= c << bitpos;
        }
        bitpos += 8;
        offset += 1;
    }
    if signed && bitpos <= u64::BITS as i64 {
        let sign_bit = 1_u64 << (bitpos - 1);
        result = (result ^ sign_bit).wrapping_sub(sign_bit);
    }
    let result = result as i64;
    debug_assert!(sign != -1 || result < 0);
    if !signed && result < 0 {
        return Err(RBigIntError::Overflow);
    }
    Ok(result)
}

/// rbigint.py `tobytes_int`.
#[majit_macros::jit_elidable]
pub fn tobytes_int(
    intval: i64,
    nbytes: i64,
    byteorder: &str,
    signed: bool,
) -> Result<Vec<u8>, RBigIntError> {
    if byteorder != "big" && byteorder != "little" {
        return Err(RBigIntError::InvalidEndianness);
    }
    if !signed && intval < 0 {
        return Err(RBigIntError::InvalidSignedness);
    }
    // Keep the same allocation edge as rbigint.py's StringBuilder(nbytes).
    // In particular, a signed-to-unsigned cast must not turn a negative
    // length into an enormous allocation request.
    let capacity = usize::try_from(nbytes).map_err(|_| RBigIntError::Memory)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| RBigIntError::Memory)?;
    let mut currval = intval;
    let mut byte = 0_u8;
    for _ in 0..nbytes {
        byte = (currval & 0xff) as u8;
        result.push(byte);
        currval >>= 8;
    }
    if currval != 0 && currval != -1 {
        return Err(RBigIntError::Overflow);
    }
    if nbytes > 0 && signed && ((intval < 0) != (byte >= 0x80)) {
        return Err(RBigIntError::Overflow);
    }
    if byteorder == "big" {
        result.reverse();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_representation_and_fromint() {
        let zero = RBigInt::fromint(0);
        assert_eq!(zero.get_sign(), 0);
        assert_eq!(zero.numdigits(), 1);
        assert_eq!(zero.digit(0), 0);

        let max = RBigInt::fromint(i64::MAX);
        assert_eq!(max.get_sign(), 1);
        assert_eq!(max.numdigits(), 1);
        assert_eq!(max.digit(0), i64::MAX);

        let min = RBigInt::fromint(i64::MIN);
        assert_eq!(min.get_sign(), -1);
        assert_eq!(min.numdigits(), 2);
        assert_eq!(min.digit(0), 0);
        assert_eq!(min.digit(1), 1);

        // rbigint.__init__: `(size or len(digits)) * sign`.  The backing
        // GcArray still has one sentinel slot, but that storage minimum must
        // not turn an empty digit list into a logically nonzero bigint.
        let empty_positive = RBigInt::new(&[], 1, 0);
        assert_eq!(empty_positive.get_sign(), 0);
        assert_eq!(empty_positive.numdigits(), 1);
        assert_eq!(empty_positive.digit(0), 0);
    }

    #[test]
    fn test_prebuilt_rbigints_share_upstream_digit_arrays() {
        let zero_a = RBigInt::zero();
        let zero_b = RBigInt::fromint(0);
        let zero_c = RBigInt::frombool(false);
        assert_eq!(zero_a._digits, zero_b._digits);
        assert_eq!(zero_a._digits, zero_c._digits);

        let one_a = RBigInt::one();
        let one_b = RBigInt::frombool(true);
        assert_eq!(one_a._digits, one_b._digits);

        let minus_one_a = RBigInt::negative_one();
        let minus_one_b = RBigInt::zero().invert();
        assert_eq!(minus_one_a._digits, minus_one_b._digits);

        let five_a = RBigInt::five();
        let five_b = RBigInt::five();
        assert_eq!(five_a._digits, five_b._digits);

        let magnitude = RBigInt::one().lshift(200).unwrap().int_add(17);
        let negative = magnitude.neg();
        assert_eq!(
            magnitude._digits, negative._digits,
            "rbigint.neg must share the upstream immutable digit array"
        );
        assert_eq!(negative.get_sign(), -1);
        let absolute = negative.abs();
        assert_eq!(
            negative._digits, absolute._digits,
            "rbigint.abs must share the upstream immutable digit array"
        );
        assert!(absolute.eq(&magnitude));

        let raw_zero_a = alloc_rbigint_nursery(RBigInt::zero());
        let raw_zero_b = alloc_rbigint_nursery_collecting(RBigInt::frombool(false));
        let raw_zero_c = alloc_rbigint_stable(RBigInt::fromint(0));
        assert_eq!(raw_zero_a, raw_zero_b);
        assert_eq!(raw_zero_a, raw_zero_c);
    }

    #[test]
    fn test_clone_allocation_bypasses_prebuilt_payload_canonicalization() {
        let prebuilts = [
            RBigInt::zero(),
            RBigInt::one(),
            RBigInt::negative_one(),
            RBigInt::five(),
        ];

        for value in prebuilts {
            let prebuilt = gc::prebuilt_payload_pointer(&value).expect("prebuilt payload");
            let original_sign = unsafe { (*prebuilt).get_sign() };
            let original_digits = unsafe { (*prebuilt)._digits };
            let cloned = alloc_rbigint_clone_nursery_collecting(value.clone());

            assert_ne!(
                cloned, prebuilt,
                "the clone residual must allocate a fresh rbigint handle"
            );
            assert_eq!(
                unsafe { (*cloned)._digits },
                original_digits,
                "the fresh handle must retain rbigint's shallow digit sharing"
            );

            // Generated neg/abs code mutates the cloned handle's `_size`.
            // Exercise the same write directly, including zero where changing
            // the raw size makes aliasing observable despite `_set_sign(0)`.
            unsafe {
                if original_sign == 0 {
                    (*cloned)._size = 1;
                } else {
                    (*cloned)._set_sign(-original_sign);
                }
                assert_eq!(
                    (*prebuilt).get_sign(),
                    original_sign,
                    "mutating the fresh handle must not corrupt the immortal prebuilt"
                );
            }
        }
    }

    #[test]
    fn test_sign_only_results_copy_upstream_digit_slice() {
        // rbigint.py/810/854 uses `_digits[:size]` for these sign-only
        // arithmetic results.  This deliberately differs from neg()/abs(),
        // which pass the existing `_digits` list without slicing.
        let value = RBigInt::fromint(123_456_789);
        let zero_minus = RBigInt::zero().sub(&value);
        assert_eq!(zero_minus.get_sign(), -1);
        assert!(!std::ptr::eq(zero_minus._digits, value._digits));
        assert!(zero_minus.eq(&value.neg()));

        let one_times = RBigInt::one().mul(&value);
        assert_eq!(one_times.get_sign(), 1);
        assert!(!std::ptr::eq(one_times._digits, value._digits));
        assert!(one_times.eq(&value));

        let negative_one_times = RBigInt::negative_one().mul(&value);
        assert_eq!(negative_one_times.get_sign(), -1);
        assert!(!std::ptr::eq(negative_one_times._digits, value._digits));
        assert!(negative_one_times.eq(&value.neg()));

        let int_negative_one_times = value.int_mul(-1);
        assert_eq!(int_negative_one_times.get_sign(), -1);
        assert!(!std::ptr::eq(int_negative_one_times._digits, value._digits));
        assert!(int_negative_one_times.eq(&value.neg()));
    }

    #[test]
    fn test_frombool_and_int_conversions() {
        assert_eq!(RBigInt::frombool(false).toint(), Ok(0));
        assert_eq!(RBigInt::frombool(true).toint(), Ok(1));
        for value in [0, 1, -1, 42, -42, i64::MAX, i64::MIN] {
            let big = RBigInt::fromint(value);
            assert_eq!(big.toint(), Ok(value));
            assert!(big.fits_int());
        }
        assert_eq!(
            RBigInt::fromint(-1).touint(),
            Err(RBigIntError::NegativeToUnsigned)
        );
        let signed_overflow = RBigInt::one().lshift(63).unwrap();
        assert_eq!(signed_overflow.toint(), Err(RBigIntError::Overflow));
        assert!(!signed_overflow.fits_int());
        assert_eq!(signed_overflow.touint(), Ok(1_u64 << 63));
        let unsigned_overflow = RBigInt::one().lshift(64).unwrap();
        assert_eq!(unsigned_overflow.touint(), Err(RBigIntError::Overflow));
        assert_eq!(unsigned_overflow.toulonglong(), Err(RBigIntError::Overflow));
    }

    #[test]
    fn test_upstream_conversion_and_log_helpers() {
        for value in [i64::MIN, -1, 0, 1, i64::MAX] {
            let from_long = RBigInt::fromlong(value as i128);
            let from_rarith = RBigInt::fromrarith_int(value);
            assert!(from_long.eq(&from_rarith));
            assert_eq!(from_long.tolong(), Ok(value as i128));
        }
        for value in [0_u64, 1, i64::MAX as u64, 1_u64 << 63, u64::MAX] {
            let from_long = RBigInt::fromlong(value as i128);
            let from_rarith = RBigInt::fromrarith_uint(value);
            assert!(from_long.eq(&from_rarith));
            assert_eq!(from_long.tolong(), Ok(value as i128));
        }
        for value in [i128::MIN, i128::MAX] {
            let from_long = RBigInt::fromlong(value);
            assert_eq!(from_long.tolong(), Ok(value));
        }
        assert_eq!(
            RBigInt::one().lshift(127).unwrap().tolong(),
            Err(RBigIntError::Overflow)
        );

        for value in [0_i64, 1, 2, 3, 255, 256, i64::MAX, i64::MIN] {
            let expected = if value == 0 {
                0
            } else {
                (u64::BITS - value.unsigned_abs().leading_zeros()) as i64
            };
            assert_eq!(bit_length_int(value), expected);
        }
        let huge = RBigInt::fromint(3).int_pow(300, None).unwrap();
        let expected = 300.0 * 3_f64.ln();
        assert!((huge.log(0.0).unwrap() - expected).abs() < 1e-12 * expected);
        assert!((huge.log(2.0).unwrap() - expected / 2_f64.ln()).abs() < 1e-12 * expected);
        assert!((huge.log(10.0).unwrap() - expected / 10_f64.ln()).abs() < 1e-12 * expected);
        assert_eq!(RBigInt::zero().log(0.0), Err(RBigIntError::LogDomain));
        assert_eq!(RBigInt::fromint(-1).log(0.0), Err(RBigIntError::LogDomain));
    }

    #[test]
    fn test_upstream_machine_word_conversion_boundaries() {
        // Exact vectors from test_rbigint.py::test_longlong,
        // test_uintmask, test_ulonglongmask, test_toulonglong, and
        // test_fits_int.
        let signed_bound = RBigInt::one().lshift(63).unwrap();
        assert_eq!(signed_bound.int_sub(1).tolonglong(), Ok(i64::MAX));
        assert_eq!(signed_bound.neg().tolonglong(), Ok(i64::MIN));
        assert_eq!(signed_bound.tolonglong(), Err(RBigIntError::Overflow));
        assert_eq!(
            signed_bound.neg().int_sub(1).tolonglong(),
            Err(RBigIntError::Overflow)
        );

        assert_eq!(RBigInt::fromint(-1).uintmask(), u64::MAX);
        assert_eq!(RBigInt::zero().uintmask(), 0);
        assert_eq!(RBigInt::fromint(i64::MAX).uintmask(), i64::MAX as u64);
        assert_eq!(signed_bound.uintmask(), 1_u64 << 63);

        assert_eq!(RBigInt::fromint(-1).ulonglongmask(), u64::MAX);
        assert_eq!(RBigInt::zero().ulonglongmask(), 0);
        assert_eq!(RBigInt::fromint(i64::MAX).ulonglongmask(), i64::MAX as u64);
        let nine_to_fifty = RBigInt::fromint(9).int_pow(50, None).unwrap();
        assert_eq!(nine_to_fifty.ulonglongmask(), 9_u64.wrapping_pow(50));
        assert_eq!(
            nine_to_fifty.neg().ulonglongmask(),
            0_u64.wrapping_sub(9_u64.wrapping_pow(50))
        );
        assert_eq!(
            RBigInt::fromint(-1).toulonglong(),
            Err(RBigIntError::NegativeToUnsigned)
        );

        for value in [0, 42, -42, i64::MAX, i64::MIN] {
            assert!(RBigInt::fromint(value).fits_int());
        }
        assert!(!signed_bound.fits_int());
        assert!(!signed_bound.neg().int_sub(1).fits_int());
        assert!(!RBigInt::fromdecimalstr("-73786976294838206459").fits_int());
        assert!(!RBigInt::one().lshift(1000).unwrap().fits_int());
    }

    #[test]
    fn test_machine_int_byte_helpers() {
        for value in [i64::MIN, -300, -1, 0, 1, 127, 128, 255, 256, i64::MAX] {
            for byteorder in ["little", "big"] {
                let bytes = tobytes_int(value, 8, byteorder, true).unwrap();
                assert_eq!(frombytes_int(&bytes, byteorder, true), Ok(value));
                if value >= 0 {
                    let bytes = tobytes_int(value, 8, byteorder, false).unwrap();
                    assert_eq!(frombytes_int(&bytes, byteorder, false), Ok(value));
                }
            }
        }
        assert_eq!(
            tobytes_int(128, 1, "big", true),
            Err(RBigIntError::Overflow)
        );
        assert_eq!(
            tobytes_int(-1, 1, "big", false),
            Err(RBigIntError::InvalidSignedness)
        );
        assert_eq!(tobytes_int(0, -1, "big", false), Err(RBigIntError::Memory));
        assert_eq!(
            RBigInt::zero().tobytes(-1, "big", false),
            Err(RBigIntError::Memory)
        );
        assert_eq!(
            frombytes_int(&[0x80, 0, 0, 0, 0, 0, 0, 0], "big", false),
            Err(RBigIntError::Overflow)
        );
        assert_eq!(
            frombytes_int(
                &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                "big",
                true
            ),
            Ok(-1)
        );
    }

    #[test]
    fn test_eq_ne_and_order() {
        let values = [i64::MIN, -50, -2, -1, 0, 1, 2, 10, 50, i64::MAX];
        for x in values {
            for y in values {
                let a = RBigInt::fromint(x);
                let b = RBigInt::fromint(y);
                assert_eq!(a.eq(&b), x == y);
                assert_eq!(a.ne(&b), x != y);
                assert_eq!(a.lt(&b), x < y);
                assert_eq!(a.le(&b), x <= y);
                assert_eq!(a.gt(&b), x > y);
                assert_eq!(a.ge(&b), x >= y);
                assert_eq!(a.int_eq(y), x == y);
                assert_eq!(a.int_lt(y), x < y);
                assert_eq!(a.int_le(y), x <= y);
                assert_eq!(a.int_gt(y), x > y);
                assert_eq!(a.int_ge(y), x >= y);
            }
        }
    }

    #[test]
    fn test_normalize() {
        let mut one = RBigInt::new(&[1, 0], 1, 2);
        one._normalize();
        assert_eq!(one.numdigits(), 1);
        let mut zero = RBigInt::new(&[0, 0, 0], 1, 3);
        zero._normalize();
        assert_eq!(zero.numdigits(), 1);
        assert_eq!(zero._size, 0);
        assert_eq!(
            zero._digits,
            RBigInt::zero()._digits,
            "_normalize must replace a zero result with upstream NULLDIGITS"
        );
        assert!(one.sub(&one).eq(&zero));
    }

    #[test]
    fn test_normalize_on_a_rooted_handle_publishes_the_sentinel() {
        // `_kmul_split` holds both halves on owner-root slots and normalises
        // them there. A high half of one zero digit collapses to zero, and
        // `_normalize` installs the shared zero digit array; the slot has to
        // carry that store, or the read that follows restores the freshly
        // allocated block the value no longer describes and nothing roots.
        let n = RBigInt::new(&[7, NULLDIGIT], 1, 2);
        let (hi, lo) = _kmul_split(&n, 1);
        assert_eq!(lo._size, 1);
        assert_eq!(hi._size, 0);
        assert_eq!(
            hi._digits,
            RBigInt::zero()._digits,
            "a rooted `_normalize` must leave the zero result on NULLDIGITS"
        );
    }

    #[test]
    fn test_bitwise_zero_result_is_nulldigits() {
        // `_bitwise` ends with `z._normalize(); return z`: a result that
        // collapses to zero carries NULLDIGITS, not the block
        // `[NULLDIGIT] * size_z` allocated for it.
        let zero_digits = RBigInt::zero()._digits;
        let one = RBigInt::fromint(1);
        let two = RBigInt::fromint(2);
        let big = RBigInt::fromint(1).lshift(200).unwrap();
        let cases = [
            ("and", one.and_(&two)),
            ("and", big.and_(&one)),
            ("xor", big.xor(&big)),
            ("or", RBigInt::zero().or_(&RBigInt::zero())),
        ];
        for (op, result) in cases {
            assert_eq!(result._size, 0, "{op}");
            assert_eq!(
                result._digits, zero_digits,
                "a zero `{op}` result must be normalized onto NULLDIGITS"
            );
        }
    }

    #[test]
    fn test_divmod_fast_pos_zero_remainder_is_nulldigits() {
        // `_divmod_fast_pos` ends with `r._normalize()` before returning `r`.
        let b = RBigInt::fromint(3)
            .pow(&RBigInt::fromint(300), None)
            .unwrap();
        let q = RBigInt::fromint(7)
            .pow(&RBigInt::fromint(200), None)
            .unwrap();
        let a = q.mul(&b);
        let (qq, r) = _divmod_fast_pos(&a, &b).unwrap();
        assert!(qq.eq(&q));
        assert_eq!(r._size, 0);
        assert_eq!(
            r._digits,
            RBigInt::zero()._digits,
            "a zero remainder must be normalized onto NULLDIGITS"
        );
    }

    #[test]
    fn test_upstream_specialized_digit_helpers() {
        // rbigint.py.  These are distinct RPython graphs selected by
        // `@specialize.argtype(0)`, and `_store_digit` itself must remain a
        // plain STORE_TYPE cast rather than silently masking its input.
        assert_eq!(_mask_digit(-1), MASK as Digit);
        assert_eq!(_mask_udigit(u64::MAX), MASK as Digit);
        assert_eq!(_mask_widedigit(-1), MASK as Digit);
        assert_eq!(_mask_uwidedigit(u128::MAX), MASK as Digit);
        assert_eq!(_store_digit(-1), -1);
        assert_eq!(_store_udigit(u64::MAX), -1);
        assert_eq!(_store_widedigit(i128::MAX), -1);
        assert_eq!(_store_uwidedigit(u128::MAX), -1);

        let mut value = RBigInt::with_size(4, 1);
        value.setdigit(0, -1);
        value.setdigit_udigit(1, u64::MAX);
        value.setdigit_widedigit(2, -1);
        value.setdigit_uwidedigit(3, u128::MAX);
        assert_eq!(value.digits(), &[MASK as Digit; 4]);
    }

    #[test]
    fn test_upstream_internal_single_digit_and_karatsuba_helpers() {
        // Direct vectors from TestInternalFunctions in test_rbigint.py.
        for (x, y) in [
            (1_238_585_838_347_i64, 3_i64),
            (1_234_123_412_311_231, 1_231_231),
            (99, 100),
        ] {
            let mut quotient = RBigInt::fromint(x);
            // A shallow RBigInt clone retains the same `_digits` GC array,
            // exercising upstream's explicitly-supported `pin is pout` case
            // without creating aliased Rust references to one stack value.
            let pin = quotient.clone();
            let remainder = _inplace_divrem1(&mut quotient, &pin, y);
            quotient._normalize();
            assert_eq!(quotient.toint(), Ok(x / y));
            assert_eq!(remainder, x % y);
        }
        let mut alias_out = RBigInt::new(&[99, 99], 1, 2);
        let alias_pin = alias_out.clone();
        let _ = _inplace_divrem1(&mut alias_out, &alias_pin, 100);

        let dividend = RBigInt::fromint(1_238_585_838_347);
        let (quotient, remainder) = _divrem1(&dividend, 3);
        assert_eq!(quotient.toint(), Ok(1_238_585_838_347 / 3));
        assert_eq!(remainder, 1_238_585_838_347 % 3);

        let product = _muladd1(&dividend, 3, 42);
        assert_eq!(product.toint(), Ok(1_238_585_838_347 * 3 + 42));

        let mut add_target = RBigInt::new(&[MASK as Digit; 10], 1, 10);
        let one = RBigInt::one();
        let add_m = add_target.numdigits() - 1;
        assert_eq!(_v_iadd(&mut add_target, 1, add_m, &one, 1), 1);
        assert_eq!(add_target.tolong(), Ok(MASK as i128));

        let mut sub_digits = vec![0_i64; 11];
        sub_digits[0] = MASK as Digit;
        sub_digits[10] = 1;
        let mut sub_target = RBigInt::new(&sub_digits, 1, sub_digits.len() as i64);
        let sub_m = sub_target.numdigits() - 1;
        assert_eq!(_v_isub(&mut sub_target, 1, sub_m, &one, 1), 0);
        assert_eq!(&sub_target.digits()[..10], &[MASK as Digit; 10]);
        assert_eq!(sub_target.digit(10), 0);

        let split = 5;
        let mut split_digits = vec![0_i64; split];
        split_digits.extend_from_slice(&[MASK as Digit; 5]);
        let split_value = RBigInt::new(&split_digits, 1, split_digits.len() as i64);
        let (hi, lo) = _kmul_split(&split_value, split as i64);
        assert!(lo.is_zero());
        assert_eq!(hi.digits()[..hi.numdigits() as usize], [MASK as Digit; 5]);
    }

    #[test]
    fn test_add_sub_matches_machine_int_when_in_range() {
        let values = [-1000_i64, -50, -2, -1, 0, 1, 2, 10, 50, 1000];
        for x in values {
            for y in values {
                let a = RBigInt::fromint(x);
                let b = RBigInt::fromint(y);
                assert_eq!(a.add(&b).toint(), Ok(x + y));
                assert_eq!(a.sub(&b).toint(), Ok(x - y));
                assert_eq!(a.mul(&b).toint(), Ok(x * y));
                assert_eq!(a.int_add(y).toint(), Ok(x + y));
                assert_eq!(a.int_sub(y).toint(), Ok(x - y));
                assert_eq!(a.int_mul(y).toint(), Ok(x * y));
            }
        }
    }

    #[test]
    fn test_add_sub_cross_digit_boundary() {
        let max = RBigInt::fromint(i64::MAX);
        let sum = max.add(&RBigInt::one());
        assert_eq!(sum.get_sign(), 1);
        assert_eq!(sum.numdigits(), 2);
        assert_eq!(sum.digit(0), 0);
        assert_eq!(sum.digit(1), 1);
        assert!(!sum.fits_int());
        assert_eq!(sum.sub(&RBigInt::one()).toint(), Ok(i64::MAX));

        let min = RBigInt::fromint(i64::MIN);
        let below = min.sub(&RBigInt::one());
        assert_eq!(below.get_sign(), -1);
        assert!(!below.fits_int());
        assert_eq!(below.add(&RBigInt::one()).toint(), Ok(i64::MIN));
    }

    #[test]
    fn test_bit_length() {
        for value in [0_i64, 1, 2, 3, 7, 8, 255, 256, i64::MAX, i64::MIN] {
            let expected = if value == 0 {
                0
            } else {
                (u64::BITS - value.unsigned_abs().leading_zeros()) as i64
            };
            assert_eq!(RBigInt::fromint(value).bit_length(), Ok(expected));
        }
    }

    #[test]
    fn test_mul_cross_digit_boundary() {
        let max = RBigInt::fromint(i64::MAX);
        let square = max.mul(&max);
        assert_eq!(square.get_sign(), 1);
        assert_eq!(square.numdigits(), 2);
        assert_eq!(square.digit(0), 1);
        assert_eq!(square.udigit(1), MASK as u64 - 1);
    }

    #[test]
    fn test_karatsuba_mul_and_square() {
        // (B^19 + 1)(B^19 + 2) = B^38 + 3B^19 + 2.  Twenty
        // digits crosses KARATSUBA_CUTOFF exactly as upstream.
        let mut a_digits = [0_i64; 20];
        a_digits[0] = 1;
        a_digits[19] = 1;
        let mut b_digits = a_digits;
        b_digits[0] = 2;
        let a = RBigInt::new(&a_digits, 1, 20);
        let b = RBigInt::new(&b_digits, 1, 20);
        let product = a.mul(&b);
        assert_eq!(product.numdigits(), 39);
        assert_eq!(product.digit(0), 2);
        assert_eq!(product.digit(19), 3);
        assert_eq!(product.digit(38), 1);
        for i in 1..38 {
            if i != 19 {
                assert_eq!(product.digit(i), 0);
            }
        }

        // Forty digits crosses KARATSUBA_SQUARE_CUTOFF.
        let mut square_digits = [0_i64; 40];
        square_digits[0] = 1;
        square_digits[39] = 1;
        let value = RBigInt::new(&square_digits, 1, 40);
        let square = value.mul(&value);
        assert_eq!(square.numdigits(), 79);
        assert_eq!(square.digit(0), 1);
        assert_eq!(square.digit(39), 2);
        assert_eq!(square.digit(78), 1);
    }

    #[test]
    fn test_upstream_karatsuba_and_lopsided_regressions() {
        // test_rbigint.py::test_karatsuba_not_used_bug.  Build the oracle as
        // shifts/additions so it is independent of the multiplication path.
        let a = RBigInt::one().lshift(2000).unwrap().int_add(1);
        let b = RBigInt::one().lshift(5000).unwrap().int_add(7);
        let expected = RBigInt::one()
            .lshift(7000)
            .unwrap()
            .add(&RBigInt::fromint(7).lshift(2000).unwrap())
            .add(&RBigInt::one().lshift(5000).unwrap())
            .int_add(7);
        assert!(a.mul(&b).eq(&expected));

        // test_overzealous_assertion: this used to trip an internal
        // Karatsuba assertion for very differently-sized negative operands.
        let left = RBigInt::negative_one().lshift(10_000).unwrap();
        let right = RBigInt::negative_one().lshift(3_000).unwrap();
        assert!(left.mul(&right).eq(&RBigInt::one().lshift(13_000).unwrap()));
    }

    #[test]
    fn test_shift_and_invert_match_machine_int_when_in_range() {
        let values = [-1000_i64, -50, -2, -1, 0, 1, 2, 10, 50, 1000];
        for value in values {
            let big = RBigInt::fromint(value);
            assert_eq!(big.invert().toint(), Ok(!value));
            for shift in 0..=10 {
                assert_eq!(big.lshift(shift).unwrap().toint(), Ok(value << shift));
                assert_eq!(
                    big.rshift(shift, false).unwrap().toint(),
                    Ok(value >> shift)
                );
            }
        }
        assert_eq!(RBigInt::one().lshift(-1), Err(RBigIntError::NegativeShift));
        assert_eq!(
            RBigInt::one().rshift(-1, false),
            Err(RBigIntError::NegativeShift)
        );
    }

    #[test]
    fn test_shift_cross_digit_boundary() {
        let shifted = RBigInt::fromint(i64::MAX).lshift(1).unwrap();
        assert_eq!(shifted.numdigits(), 2);
        assert_eq!(shifted.udigit(0), MASK as u64 - 1);
        assert_eq!(shifted.digit(1), 1);
        assert_eq!(shifted.rshift(1, false).unwrap().toint(), Ok(i64::MAX));
        assert_eq!(
            RBigInt::fromint(i64::MIN)
                .rshift(63, false)
                .unwrap()
                .toint(),
            Ok(-1)
        );
        assert_eq!(RBigInt::one().lshift(i64::MAX), Err(RBigIntError::Memory));
        assert_eq!(
            RBigInt::lshift_int_int_bigint_result(1, i64::MAX),
            Err(RBigIntError::Memory)
        );
        // test_shift_optimization: zero returns before computing or
        // allocating the enormous word-shift.
        assert!(RBigInt::zero().lshift(i64::MAX).unwrap().int_eq(0));
    }

    #[test]
    fn test_upstream_quick_shift_vectors() {
        for x in 0_i64..10 {
            for y in (1_u32..161).step_by(16) {
                let num = RBigInt::fromint(x).lshift(y as i64).unwrap().int_add(x);
                let negative = num.neg();
                for shift in 1_u32..31 {
                    assert!(
                        num.lqshift(shift as i64)
                            .eq(&num.lshift(shift as i64).unwrap())
                    );
                    assert!(
                        num.rqshift(shift as i64)
                            .eq(&num.rshift(shift as i64, false).unwrap())
                    );
                    assert!(
                        negative
                            .lqshift(shift as i64)
                            .eq(&negative.lshift(shift as i64).unwrap())
                    );
                }
            }
        }
        for x in (MASK as u128 - 10)..=(MASK as u128 + 9) {
            let value = RBigInt::from_u128(x);
            assert!(
                value
                    .rqshift(SHIFT)
                    .eq(&value.rshift(SHIFT, false).unwrap())
            );
            assert!(
                value
                    .rqshift(SHIFT + 1)
                    .eq(&value.rshift(SHIFT + 1, false).unwrap())
            );
        }
    }

    #[test]
    fn test_bitwise_matches_machine_int() {
        let values = [
            i64::MIN,
            -1000,
            -50,
            -2,
            -1,
            0,
            1,
            2,
            10,
            50,
            1000,
            i64::MAX,
        ];
        for x in values {
            for y in values {
                let a = RBigInt::fromint(x);
                let b = RBigInt::fromint(y);
                assert_eq!(a.and_(&b).toint(), Ok(x & y));
                assert_eq!(a.or_(&b).toint(), Ok(x | y));
                assert_eq!(a.xor(&b).toint(), Ok(x ^ y));
                assert_eq!(a.int_and_(y).toint(), Ok(x & y));
                assert_eq!(a.int_or_(y).toint(), Ok(x | y));
                assert_eq!(a.int_xor(y).toint(), Ok(x ^ y));
            }
        }

        // `_int_bitwise` must retain all of a positive bigint's digits when
        // ANDing with a negative machine integer: the negative operand is
        // sign-extended with MASK above its one explicit digit.
        let multi_digit = RBigInt::one()
            .lshift(2 * SHIFT + 7)
            .unwrap()
            .int_add(0b1011);
        assert!(multi_digit.int_and_(-1).eq(&multi_digit));
        assert!(multi_digit.int_and_(-2).eq(&multi_digit.int_sub(1)));
        let minint_expected = multi_digit.rqshift(SHIFT).lshift(SHIFT).unwrap();
        assert!(multi_digit.int_and_(i64::MIN).eq(&minint_expected));
        assert!(
            multi_digit
                .int_and_(-2)
                .eq(&multi_digit.and_(&RBigInt::fromint(-2)))
        );
    }

    #[test]
    fn test_specialized_int_paths_match_general_bigint_paths() {
        let mut bigints = vec![
            RBigInt::zero(),
            RBigInt::one(),
            RBigInt::negative_one(),
            RBigInt::new(&[0, 1], 1, 2),
            RBigInt::new(&[MASK, 1], 1, 2),
            RBigInt::new(&[7, 0, 3], 1, 3),
            RBigInt::new(&[MASK, MASK, 5], 1, 3),
        ];
        let negatives: Vec<_> = bigints
            .iter()
            .filter(|value| value.get_sign() > 0)
            .map(RBigInt::neg)
            .collect();
        bigints.extend(negatives);

        let ints = [
            i64::MIN,
            -i64::MAX,
            -1_000_003,
            -2,
            -1,
            0,
            1,
            2,
            1_000_003,
            i64::MAX,
        ];
        for value in &bigints {
            for iother in ints {
                let other = RBigInt::fromint(iother);
                assert!(value.int_add(iother).eq(&value.add(&other)));
                assert!(value.int_sub(iother).eq(&value.sub(&other)));
                assert!(value.int_mul(iother).eq(&value.mul(&other)));
                assert!(value.int_and_(iother).eq(&value.and_(&other)));
                assert!(value.int_or_(iother).eq(&value.or_(&other)));
                assert!(value.int_xor(iother).eq(&value.xor(&other)));
                assert_eq!(value.int_eq(iother), value.eq(&other));
                assert_eq!(value.int_lt(iother), value.lt(&other));
                assert_eq!(value.int_le(iother), value.le(&other));
                assert_eq!(value.int_gt(iother), value.gt(&other));
                assert_eq!(value.int_ge(iother), value.ge(&other));

                if iother != 0 {
                    let int_q = value.int_floordiv(iother).unwrap();
                    let general_q = value.floordiv(&other).unwrap();
                    assert!(int_q.eq(&general_q));

                    let int_r = value.int_mod(iother).unwrap();
                    let general_r = value.r#mod(&other).unwrap();
                    assert!(int_r.eq(&general_r));
                    assert_eq!(value.int_mod_int_result(iother), general_r.toint());

                    let (int_q, int_r) = value.int_divmod(iother).unwrap();
                    let (general_q, general_r) = value.divmod(&other).unwrap();
                    assert!(int_q.eq(&general_q));
                    assert!(int_r.eq(&general_r));
                }
            }
        }

        for iself in ints {
            for iother in ints {
                let left = RBigInt::fromint(iself);
                let right = RBigInt::fromint(iother);
                assert!(RBigInt::add_int_int_bigint_result(iself, iother).eq(&left.add(&right)));
                assert!(RBigInt::sub_int_int_bigint_result(iself, iother).eq(&left.sub(&right)));
                assert!(RBigInt::mul_int_int_bigint_result(iself, iother).eq(&left.mul(&right)));
            }
            for shift in [0_i64, 1, SHIFT - 1, SHIFT, 64, 127] {
                assert!(
                    RBigInt::lshift_int_int_bigint_result(iself, shift)
                        .unwrap()
                        .eq(&RBigInt::fromint(iself).lshift(shift).unwrap())
                );
            }
            assert_eq!(
                RBigInt::lshift_int_int_bigint_result(iself, -1),
                Err(RBigIntError::NegativeShift)
            );
        }
    }

    #[test]
    fn test_from_list_n_bits() {
        let value = RBigInt::from_list_n_bits(&[0x7f, 0x01], 8).unwrap();
        assert_eq!(value.toint(), Ok(0x017f));
        let base_digits = RBigInt::from_list_n_bits(&[MASK, 1], SHIFT).unwrap();
        assert_eq!(base_digits.numdigits(), 2);
        assert_eq!(base_digits.udigit(0), MASK as u64);
        assert_eq!(base_digits.digit(1), 1);
    }

    #[test]
    fn test_divmod_matches_python_floor_semantics() {
        let values = [-1000_i64, -50, -13, -2, -1, 0, 1, 2, 10, 13, 50, 1000];
        for x in values {
            for y in values {
                if y == 0 {
                    continue;
                }
                let mut q = x / y;
                let mut r = x % y;
                if r != 0 && (r < 0) != (y < 0) {
                    q -= 1;
                    r += y;
                }
                let a = RBigInt::fromint(x);
                let b = RBigInt::fromint(y);
                let (big_q, big_r) = a.divmod(&b).unwrap();
                assert_eq!(big_q.toint(), Ok(q), "{x} // {y}");
                assert_eq!(big_r.toint(), Ok(r), "{x} % {y}");
                assert_eq!(a.int_floordiv(y).unwrap().toint(), Ok(q));
                assert_eq!(a.int_mod(y).unwrap().toint(), Ok(r));
                assert_eq!(a.int_mod_int_result(y), Ok(r));
            }
        }
    }

    #[test]
    fn test_multidigit_division() {
        // (B^4 - 1) / (B^2 - 1) == B^2 + 1.
        let a = RBigInt::new(&[MASK; 4], 1, 4);
        let b = RBigInt::new(&[MASK; 2], 1, 2);
        let (q, r) = a.divmod(&b).unwrap();
        assert_eq!(q.numdigits(), 3);
        assert_eq!(q.digit(0), 1);
        assert_eq!(q.digit(1), 0);
        assert_eq!(q.digit(2), 1);
        assert_eq!(r.get_sign(), 0);
        assert!(q.mul(&b).add(&r).eq(&a));

        let negative = a.neg();
        let (q, r) = negative.divmod(&b).unwrap();
        assert!(q.mul(&b).add(&r).eq(&negative));
        assert!(r.get_sign() >= 0);
    }

    #[test]
    fn test_upstream_rare_x_divrem_quotient_correction() {
        // Explicit upstream vector for the rare `_x_divrem` branch that adds
        // the divisor back after its quotient estimate was one too large.
        let dividend = RBigInt::fromdecimalstr(
            "2401064762424988628303678384283622960038813848808995811101817752058392725584695633",
        );
        let divisor =
            RBigInt::fromdecimalstr("510439143470502793407446782273075179624699774495710665331026");
        let expected_q = RBigInt::fromdecimalstr("4703919738795935662080");
        let expected_r =
            RBigInt::fromdecimalstr("510439143470502793407446782273075179622080336837243909001553");
        let (q, r) = _x_divrem(&dividend, &divisor);
        assert!(q.eq(&expected_q));
        assert!(r.eq(&expected_r));
        assert!(q.mul(&divisor).add(&r).eq(&dividend));
    }

    #[test]
    fn test_x_divrem_zero_quotient_is_fresh() {
        // rbigint.py cannot return NULLRBIGINT because callers may
        // modify this internal result.  Equal-sized, two-digit operands with
        // dividend < divisor take the otherwise uncommon k == 0 branch.
        let dividend = RBigInt::new(&[1, 1], 1, 2);
        let divisor = RBigInt::new(&[2, 1], 1, 2);
        let (mut quotient, remainder) = _x_divrem(&dividend, &divisor);
        assert_eq!(quotient.get_sign(), 0);
        assert!(remainder.eq(&dividend));
        assert_ne!(quotient._digits, RBigInt::zero()._digits);

        quotient.setdigit(0, 1);
        assert_eq!(RBigInt::zero().digit(0), 0);
    }

    #[test]
    fn test_pow_and_modular_pow() {
        assert_eq!(
            RBigInt::fromint(2).int_pow(10, None).unwrap().toint(),
            Ok(1024)
        );
        assert_eq!(
            RBigInt::fromint(-2).int_pow(9, None).unwrap().toint(),
            Ok(-512)
        );
        assert_eq!(
            RBigInt::fromint(-2).int_pow(10, None).unwrap().toint(),
            Ok(1024)
        );
        let modulus = RBigInt::fromint(13);
        assert_eq!(
            RBigInt::fromint(2)
                .int_pow(5, Some(&modulus))
                .unwrap()
                .toint(),
            Ok(6)
        );
        let negative_modulus = RBigInt::fromint(-13);
        assert_eq!(
            RBigInt::fromint(2)
                .int_pow(5, Some(&negative_modulus))
                .unwrap()
                .toint(),
            Ok(-7)
        );
    }

    #[test]
    fn test_int_pow_specialized_path_matches_bigint_exponent_path() {
        let bases = [-65_i64, -64, -8, -2, -1, 0, 1, 2, 8, 64, 65];
        let exponents = [0_i64, 1, 2, 3, 4, 7, 31, 63];
        let moduli = [-65_i64, -13, -1, 1, 13, 65];
        for base in bases {
            let value = RBigInt::fromint(base);
            for exponent in exponents {
                let big_exponent = RBigInt::fromint(exponent);
                let specialized = value.int_pow(exponent, None).unwrap();
                let general = value.pow(&big_exponent, None).unwrap();
                assert!(
                    specialized.eq(&general),
                    "int_pow mismatch for {base} ** {exponent}"
                );
                for modulus in moduli {
                    let modulus = RBigInt::fromint(modulus);
                    let specialized = value.int_pow(exponent, Some(&modulus)).unwrap();
                    let general = value.pow(&big_exponent, Some(&modulus)).unwrap();
                    assert!(
                        specialized.eq(&general),
                        "int_pow mismatch for pow({base}, {exponent}, {modulus:?})"
                    );
                }
            }
        }

        let zero = RBigInt::zero();
        assert_eq!(
            RBigInt::fromint(2).int_pow(3, Some(&zero)),
            Err(RBigIntError::ZeroModulus)
        );
        assert_eq!(
            RBigInt::fromint(2).int_pow(-1, None),
            Err(RBigIntError::NegativeExponent)
        );
        assert_eq!(
            RBigInt::fromint(2).int_pow(-1, Some(&RBigInt::fromint(13))),
            Err(RBigIntError::NegativeExponentWithModulus)
        );
        assert_eq!(
            RBigInt::fromint(64)
                .pow(&RBigInt::one(), Some(&RBigInt::fromint(63)))
                .unwrap()
                .toint(),
            Ok(1)
        );
        assert_eq!(
            RBigInt::fromint(-64)
                .pow(&RBigInt::one(), Some(&RBigInt::fromint(-65)))
                .unwrap()
                .toint(),
            Ok(-64)
        );

        let modulus = RBigInt::fromint(i64::MAX);
        for (base, exponent, expected) in [
            (-5, 1_i64 << 31, 5_400_123_348_685_254_823),
            (-5, (1_i64 << 32) - 1, 5_943_747_623_342_280_032),
            (-5, 1_i64 << 32, 7_174_750_030_707_703_068),
            (-2, 1_i64 << 31, 4),
            (-2, (1_i64 << 32) - 1, 9_223_372_036_854_775_799),
            (-2, 1_i64 << 32, 16),
            (2, 1_i64 << 31, 4),
            (2, (1_i64 << 32) - 1, 8),
            (2, 1_i64 << 32, 16),
            (5, 1_i64 << 31, 5_400_123_348_685_254_823),
            (5, (1_i64 << 32) - 1, 3_279_624_413_512_495_775),
            (5, 1_i64 << 32, 7_174_750_030_707_703_068),
        ] {
            assert_eq!(
                RBigInt::fromint(base)
                    .int_pow(exponent, Some(&modulus))
                    .unwrap()
                    .toint(),
                Ok(expected),
                "upstream int_pow_big vector failed for {base} ** {exponent}"
            );
        }
    }

    #[test]
    fn test_fiveary_pow_path() {
        let mut exponent_digits = [0_i64; FIVEARY_CUTOFF as usize + 1];
        exponent_digits[FIVEARY_CUTOFF as usize] = 1;
        let exponent = RBigInt::new(&exponent_digits, 1, exponent_digits.len() as i64);
        let modulus = RBigInt::fromint(3);
        let result = RBigInt::fromint(2).pow(&exponent, Some(&modulus)).unwrap();
        assert_eq!(result.toint(), Ok(1));
    }

    #[test]
    fn test_upstream_modular_pow_regressions() {
        // test_pow_lll_bug / test_pow_lll_bug2 from upstream.
        let two = RBigInt::fromint(2);
        let exponent = RBigInt::fromdecimalstr(
            "2655689964083835493447941032762343136647965588635159615997220691002017799304",
        );
        for (modulus, expected) in [(37, 9), (1291, 931), (67_889, 39_464)] {
            assert_eq!(
                two.pow(&exponent, Some(&RBigInt::fromint(modulus)))
                    .unwrap()
                    .toint(),
                Ok(expected)
            );
        }

        let exponent = RBigInt::fromdecimalstr(
            "5100894665148900058249470019412564146962964987365857466751243988156579407594163282788332839328303748028644825680244165072186950517295679131100799612871613064597",
        );
        assert_eq!(
            two.pow(&exponent, Some(&RBigInt::fromint(538_564)))
                .unwrap()
                .toint(),
            Ok(163_464)
        );
    }

    #[test]
    fn test_bit_count_gcd_and_isqrt() {
        for value in [0_i64, 1, 2, 3, 7, 8, 255, 256, i64::MAX, i64::MIN] {
            assert_eq!(
                RBigInt::fromint(value).bit_count(),
                Ok(value.unsigned_abs().count_ones() as i64)
            );
        }
        assert_eq!(
            RBigInt::fromint(48)
                .gcd(&RBigInt::fromint(-18))
                .unwrap()
                .toint(),
            Ok(6)
        );
        for value in 0_i64..500 {
            let expected = (value as f64).sqrt() as i64;
            assert_eq!(
                RBigInt::fromint(value).isqrt().unwrap().toint(),
                Ok(expected)
            );
        }
        assert_eq!(
            RBigInt::fromint(-1).isqrt(),
            Err(RBigIntError::NegativeSquareRoot)
        );
    }

    #[test]
    fn test_multidigit_gcd_and_isqrt() {
        let b_minus_one = RBigInt::new(&[MASK], 1, 1);
        let b_squared_minus_one = RBigInt::new(&[MASK, MASK], 1, 2);
        assert!(
            b_squared_minus_one
                .gcd(&b_minus_one)
                .unwrap()
                .eq(&b_minus_one)
        );

        let root = RBigInt::new(&[1, 1], 1, 2);
        let square = root.mul(&root);
        assert!(square.isqrt().unwrap().eq(&root));
        assert!(square.int_sub(1).isqrt().unwrap().eq(&root.int_sub(1)));
    }

    #[test]
    fn test_fromdecimalstr_and_fromstr() {
        let decimal = "12345678901234567890523897987";
        assert_eq!(RBigInt::fromdecimalstr(decimal).str(0).unwrap(), decimal);
        assert_eq!(
            RBigInt::fromdecimalstr(&format!("-{decimal}"))
                .str(0)
                .unwrap(),
            format!("-{decimal}")
        );
        assert!(RBigInt::fromdecimalstr("+0").int_eq(0));
        assert!(RBigInt::fromdecimalstr("-0").int_eq(0));

        assert_eq!(RBigInt::fromstr("123L", 0, false).unwrap().toint(), Ok(123));
        assert_eq!(
            RBigInt::fromstr("123L  ", 0, false).unwrap().toint(),
            Ok(123)
        );
        assert_eq!(RBigInt::fromstr("123L", 4, false).unwrap().toint(), Ok(27));
        assert_eq!(
            RBigInt::fromstr("123L", 30, false).unwrap().toint(),
            Ok(27000 + 1800 + 90 + 21)
        );
        assert_eq!(
            RBigInt::fromstr("123L", 22, false).unwrap().toint(),
            Ok(10648 + 968 + 66 + 21)
        );
        assert_eq!(
            RBigInt::fromstr("123L", 21, false).unwrap().toint(),
            Ok(441 + 42 + 3)
        );
        assert_eq!(
            RBigInt::fromstr("1891_234_17_4_19731_9", 0, true)
                .unwrap()
                .toint(),
            Ok(1_891_234_174_197_319)
        );
        assert_eq!(
            RBigInt::fromstr("0x_abcdef", 0, true).unwrap().toint(),
            Ok(0xabcdef)
        );
        assert_eq!(RBigInt::fromstr("077", 0, false).unwrap().toint(), Ok(0o77));
        assert_eq!(
            RBigInt::fromstr("L", 0, false),
            Err(RBigIntError::ParseString)
        );
        assert_eq!(
            RBigInt::fromstr("1__2", 10, true),
            Err(RBigIntError::ParseString)
        );
        assert_eq!(
            RBigInt::fromstr("1_", 10, true),
            Err(RBigIntError::ParseString)
        );
    }

    #[test]
    fn test_number_string_parser_state_and_limits() {
        let mut parser =
            NumberStringParser::new("-99", 10, false, false, 0, None, 0, false).unwrap();
        assert_eq!(parser.sign, -1);
        assert_eq!(parser.next_digit(), Ok(9));
        assert_eq!(parser.next_digit(), Ok(9));
        assert_eq!(parser.next_digit(), Ok(-1));
        parser.rewind();
        assert_eq!(parser.next_digit(), Ok(9));
        assert_eq!(parser.next_digit(), Ok(9));
        assert_eq!(parser.next_digit(), Ok(-1));

        let mut invalid_octal = NumberStringParser::new(
            "077777777777777777777777777777",
            0,
            false,
            true,
            0,
            None,
            0,
            false,
        )
        .unwrap();
        assert_eq!(
            RBigInt::_from_numberstring_parser(&mut invalid_octal),
            Err(RBigIntError::ParseString)
        );
        let mut zeroes = NumberStringParser::new("000", 0, false, true, 0, None, 0, false).unwrap();
        assert!(
            RBigInt::_from_numberstring_parser(&mut zeroes)
                .unwrap()
                .int_eq(0)
        );

        let too_long = "1".repeat(1000);
        assert!(matches!(
            NumberStringParser::new(&too_long, 10, false, false, 0, None, 999, false),
            Err(RBigIntError::MaxStrDigits)
        ));
    }

    #[test]
    fn test_parse_power_of_two_bases_and_large_decimal() {
        for (base, source) in [
            (2, "10110011100011110000111100001111"),
            (4, "123003210123003210123003210"),
            (8, "765432107654321076543210"),
            (16, "fedcba9876543210fedcba9876543210"),
            (32, "vutsrqponmlkjihgfedcba9876543210"),
        ] {
            let value = RBigInt::fromstr(source, base, false).unwrap();
            let alphabet = &BASE16[..base.min(16) as usize];
            if base <= 16 {
                assert_eq!(value.format(alphabet, "", "", 0).unwrap(), source);
            }
            assert_eq!(value.get_sign(), 1);
        }

        let source = "123952".repeat(1000);
        let mut mem = FivePowCache::new();
        let direct =
            _str_to_int_big_inner10(&source, 0, source.len() as i64, &mut mem, 20).unwrap();
        assert_eq!(mem.entries.len(), 13);
        let parsed = RBigInt::fromstr(&source, 10, false).unwrap();
        assert!(parsed.eq(&direct));
        assert_eq!(parsed.str(0).unwrap(), source);

        let underscored = "1_1".repeat(2001);
        let expected = "11".repeat(2001);
        assert_eq!(
            RBigInt::fromstr(&underscored, 10, true)
                .unwrap()
                .str(0)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn test_frombytes_tobytes_roundtrip() {
        // Literal vectors from test_rbigint.py::test_frombytes/test_tobytes.
        assert!(RBigInt::frombytes(&[], "big", true).unwrap().int_eq(0));
        assert_eq!(
            RBigInt::frombytes(&[0xff, 0x12, 0x34, 0x56], "big", false)
                .unwrap()
                .tolong(),
            Ok(0xff12_3456)
        );
        assert_eq!(
            RBigInt::frombytes(&[0xff, 0x12, 0x34, 0x56], "little", false)
                .unwrap()
                .tolong(),
            Ok(0x5634_12ff)
        );
        assert_eq!(
            RBigInt::frombytes(&[0x82], "big", true).unwrap().toint(),
            Ok(-126)
        );
        for (value, nbytes, byteorder, signed, expected) in [
            (0, 1, "big", true, vec![0x00]),
            (1, 2, "big", true, vec![0x00, 0x01]),
            (-129, 2, "big", true, vec![0xff, 0x7f]),
            (-129, 2, "little", true, vec![0x7f, 0xff]),
            (65_535, 3, "big", true, vec![0x00, 0xff, 0xff]),
            (-65_536, 3, "little", true, vec![0x00, 0x00, 0xff]),
            (65_535, 2, "big", false, vec![0xff, 0xff]),
            (-8_388_608, 3, "little", true, vec![0x00, 0x00, 0x80]),
        ] {
            assert_eq!(
                RBigInt::fromint(value).tobytes(nbytes, byteorder, signed),
                Ok(expected)
            );
        }

        for value in -300_i64..=300 {
            for byteorder in ["little", "big"] {
                let big = RBigInt::fromint(value);
                let bytes = big.tobytes(2, byteorder, true).unwrap();
                let roundtrip = RBigInt::frombytes(&bytes, byteorder, true).unwrap();
                assert_eq!(roundtrip.toint(), Ok(value));
                if value >= 0 {
                    let bytes = big.tobytes(2, byteorder, false).unwrap();
                    let roundtrip = RBigInt::frombytes(&bytes, byteorder, false).unwrap();
                    assert_eq!(roundtrip.toint(), Ok(value));
                }
            }
        }

        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let huge = RBigInt::frombytes(&bytes, "big", false).unwrap();
        assert_eq!(
            huge.tobytes(bytes.len() as i64, "big", false).unwrap(),
            bytes
        );

        // rbigint.py constructs the result from `digits[:]`: the final
        // GcArray has the number of appended base-2**SHIFT digits, not the
        // temporary list's capacity hint.
        let exact_digit_bytes = vec![0x7f; 63];
        let exact = RBigInt::frombytes(&exact_digit_bytes, "big", false).unwrap();
        assert_eq!(exact.numdigits(), 8);
        assert_eq!(exact.digits().len(), 8);
        let normalized_zero = RBigInt::frombytes(&[0; 63], "big", false).unwrap();
        assert_eq!(normalized_zero.get_sign(), 0);
        assert_eq!(normalized_zero.digits().len(), 1);

        assert_eq!(
            RBigInt::fromint(128).tobytes(1, "big", true),
            Err(RBigIntError::Overflow)
        );
        assert_eq!(
            RBigInt::fromint(-129).tobytes(1, "big", true),
            Err(RBigIntError::Overflow)
        );
        assert_eq!(
            RBigInt::fromint(-1).tobytes(1, "big", false),
            Err(RBigIntError::InvalidSignedness)
        );
    }

    #[test]
    fn test_fromfloat_and_tofloat() {
        for value in [
            -9_007_199_254_740_991_i64,
            -1_234_567_890,
            -1,
            0,
            1,
            1_234_567_890,
            9_007_199_254_740_991,
        ] {
            let float = value as f64;
            assert_eq!(RBigInt::fromfloat(float).unwrap().toint(), Ok(value));
            assert_eq!(RBigInt::fromint(value).tofloat(), Ok(float));
        }
        assert!(RBigInt::fromfloat(-0.0).unwrap().int_eq(0));
        assert_eq!(
            RBigInt::fromfloat(f64::INFINITY),
            Err(RBigIntError::InfiniteFloat)
        );
        assert_eq!(RBigInt::fromfloat(f64::NAN), Err(RBigIntError::NanFloat));

        // test_rbigint.py::test_tofloat_precision: exercise every binary
        // exponent in the upstream range, not just representative samples.
        for sign in [1_i64, -1] {
            for power in 0_i64..100 {
                let base = RBigInt::fromint((1_i64 << 53) + 1).lshift(power).unwrap();
                let rounded_up = base.int_add(1);
                let rounded_up = if sign < 0 {
                    rounded_up.neg()
                } else {
                    rounded_up
                };
                let expected_up =
                    sign as f64 * (((1_u64 << 53) + 2) as f64 * 2_f64.powi(power as i32));
                assert_eq!(rounded_up.tofloat().unwrap(), expected_up);
                assert!(
                    RBigInt::fromfloat(expected_up)
                        .unwrap()
                        .eq(&RBigInt::fromint((1_i64 << 53) + 2)
                            .lshift(power)
                            .unwrap()
                            .int_mul(sign))
                );

                let tie = if sign < 0 { base.neg() } else { base };
                let expected_tie = sign as f64 * ((1_u64 << 53) as f64 * 2_f64.powi(power as i32));
                assert_eq!(tie.tofloat().unwrap(), expected_tie);
                assert!(
                    RBigInt::fromfloat(expected_tie)
                        .unwrap()
                        .eq(&RBigInt::fromint(1_i64 << 53)
                            .lshift(power)
                            .unwrap()
                            .int_mul(sign))
                );
            }
        }
    }

    #[test]
    fn test_hash_and_true_division() {
        for value in [i64::MIN, -1_000_000, -1, 0, 1, 1_000_000, i64::MAX] {
            assert_eq!(RBigInt::fromint(value).hash(), value);
        }
        let values = [-1000_i64, -13, -2, -1, 0, 1, 2, 13, 1000];
        for x in values {
            for y in values {
                if y != 0 {
                    assert_eq!(
                        RBigInt::fromint(x).truediv(&RBigInt::fromint(y)).unwrap(),
                        x as f64 / y as f64
                    );
                }
            }
        }
        let huge = RBigInt::one().lshift(1000).unwrap();
        assert_eq!(
            huge.truediv(&RBigInt::fromint(3)).unwrap(),
            2_f64.powi(1000) / 3.0
        );
        let too_huge = RBigInt::one().lshift(2000).unwrap();
        assert_eq!(
            too_huge.truediv(&RBigInt::one()),
            Err(RBigIntError::FloatDivisionOverflow)
        );
        assert_eq!(RBigInt::one().truediv(&too_huge).unwrap(), 0.0);

        // Exact boundary vectors from test_truediv_precision,
        // test_truediv_overflow, and test_truediv_overflow2.
        let precise_numerator = RBigInt::fromint(12_345).lshift(30).unwrap();
        let precise_denominator = RBigInt::fromint(7)
            .int_pow(81, None)
            .unwrap()
            .int_mul(98_765);
        assert_eq!(
            precise_numerator.truediv(&precise_denominator).unwrap(),
            4.7298422347492634e-61
        );

        let overflowing = RBigInt::one()
            .lshift(1024)
            .unwrap()
            .sub(&RBigInt::one().lshift(1024 - 53 - 1).unwrap());
        let just_below = overflowing.int_sub(1);
        assert_eq!(just_below.truediv(&RBigInt::one()).unwrap(), f64::MAX);
        assert_eq!(
            just_below.truediv(&RBigInt::negative_one()).unwrap(),
            -f64::MAX
        );
        assert_eq!(
            just_below.neg().truediv(&RBigInt::negative_one()).unwrap(),
            f64::MAX
        );
        assert_eq!(
            overflowing.truediv(&RBigInt::one()),
            Err(RBigIntError::FloatDivisionOverflow)
        );
        let twice_just_below = overflowing.int_mul(2).int_sub(10);
        assert_eq!(
            twice_just_below.truediv(&RBigInt::fromint(2)).unwrap(),
            f64::MAX
        );
        assert_eq!(
            twice_just_below.truediv(&RBigInt::fromint(-2)).unwrap(),
            -f64::MAX
        );

        // `math.ldexp` parity at the subnormal boundary.  A direct
        // `dx * 2.0.powi(shift)` loses these values when the scale factor
        // underflows before multiplication.
        let power_of_two = |exponent| RBigInt::one().lshift(exponent).unwrap();
        assert_eq!(
            RBigInt::one().truediv(&power_of_two(1030)).unwrap(),
            f64::from_bits(0x0000_1000_0000_0000)
        );
        assert_eq!(
            RBigInt::fromint(7).truediv(&power_of_two(1074)).unwrap(),
            f64::from_bits(0x0000_0000_0000_0007)
        );
        assert_eq!(
            power_of_two(53)
                .int_add(1)
                .truediv(&power_of_two(1075).int_add(7))
                .unwrap(),
            f64::from_bits(0x0010_0000_0000_0000)
        );
    }

    #[test]
    fn test_burnikel_ziegler_divmod_cutoff() {
        let div_limit = holder_limit(&HOLDER.DIV_LIMIT);
        let divisor_digits = vec![MASK; (div_limit * 2 + 1) as usize];
        let divisor = RBigInt::new(&divisor_digits, 1, divisor_digits.len() as i64);
        let mut quotient_digits = [0_i64; 20];
        quotient_digits[0] = 1;
        quotient_digits[19] = 1;
        let quotient = RBigInt::new(&quotient_digits, 1, quotient_digits.len() as i64);
        let remainder = RBigInt::fromint(7);
        let dividend = divisor.mul(&quotient).add(&remainder);
        assert!(dividend.numdigits() * 10 > divisor.numdigits() * 12);
        assert!(divisor.numdigits() > div_limit * 2);

        let (actual_q, actual_r) = dividend.divmod(&divisor).unwrap();
        assert!(actual_q.eq(&quotient));
        assert!(actual_r.eq(&remainder));

        let (negative_q, negative_r) = dividend.neg().divmod(&divisor).unwrap();
        assert!(
            negative_q
                .mul(&divisor)
                .add(&negative_r)
                .eq(&dividend.neg())
        );
        assert!(negative_r.get_sign() >= 0);

        let negative_divisor = divisor.neg();
        let (negative_q, negative_r) = dividend.divmod(&negative_divisor).unwrap();
        assert!(
            negative_q
                .mul(&negative_divisor)
                .add(&negative_r)
                .eq(&dividend)
        );
        assert!(negative_r.get_sign() <= 0);

        // test_rbigint.py::test_divmod_big2.  This shape contains long
        // zero-filled spans inside the base-2**n chunks.  PyPy deliberately
        // keeps those chunks at fixed `n_S` width instead of normalizing each
        // one before the Burnikel-Ziegler recursion.
        let shared_shift = 100_i64 * SHIFT;
        let sparse_dividend = RBigInt::fromint(2)
            .add(&RBigInt::fromint(5).lshift(SHIFT).unwrap())
            .lshift(shared_shift)
            .unwrap();
        let sparse_divisor = RBigInt::fromint(5).lshift(shared_shift).unwrap();
        let expected_q = RBigInt::one().lshift(SHIFT).unwrap();
        let expected_r = RBigInt::fromint(2).lshift(shared_shift).unwrap();
        let (actual_q, actual_r) = divmod_big(&sparse_dividend, &sparse_divisor).unwrap();
        assert!(actual_q.eq(&expected_q));
        assert!(actual_r.eq(&expected_r));
    }

    #[test]
    fn test_burnikel_ziegler_modular_pow_4093_bit_modulus() {
        // Regression found by the runtime GC stress.  This modulus selects
        // Burnikel-Ziegler division with a non-power-of-two significant digit
        // count; every intermediate modular reduction must still satisfy the
        // recursive div2n1n slice-width invariants.
        let one = RBigInt::one();
        let mut x = one
            .lshift(4095)
            .unwrap()
            .add(&one.lshift(2057).unwrap())
            .int_add(0x0123_4567_89ab_cdef);
        let mut y = one
            .lshift(4087)
            .unwrap()
            .add(&one.lshift(1999).unwrap())
            .int_add(0x0fed_cba9_8765_4321);
        x = x
            .mul(&y.or_(&one))
            .add(&x)
            .and_(&one.lshift(4096).unwrap().int_sub(1));
        y = y
            .int_mul(1_000_003)
            .add(&x)
            .int_add(17)
            .and_(&one.lshift(4096).unwrap().int_sub(1));
        let base = x.xor(&y).int_add(3);
        let modulus = one.lshift(4093).unwrap().int_sub(159);
        let exponent = RBigInt::fromint(13);

        let result = base.pow(&exponent, Some(&modulus)).unwrap();
        assert!(result.get_sign() >= 0);
        assert!(result.lt(&modulus));
    }

    #[test]
    fn test_format_power_of_two_bases() {
        for value in [i64::MIN, -1_000_000, -1, 0, 1, 1_000_000, i64::MAX] {
            let big = RBigInt::fromint(value);
            assert_eq!(
                big.format("01", "", "", 0).unwrap(),
                if value < 0 {
                    format!("-{:b}", value.unsigned_abs())
                } else {
                    format!("{:b}", value as u64)
                }
            );
            assert_eq!(
                big.format(BASE8, "", "", 0).unwrap(),
                if value < 0 {
                    format!("-{:o}", value.unsigned_abs())
                } else {
                    format!("{:o}", value as u64)
                }
            );
            assert_eq!(
                big.format(BASE16, "", "", 0).unwrap(),
                if value < 0 {
                    format!("-{:x}", value.unsigned_abs())
                } else {
                    format!("{:x}", value as u64)
                }
            );
        }
        assert_eq!(RBigInt::fromint(255).hex().unwrap(), "0xffL");
        assert_eq!(RBigInt::fromint(-255).hex().unwrap(), "-0xffL");
        assert_eq!(RBigInt::fromint(8).oct().unwrap(), "010L");
    }

    #[test]
    fn test_upstream_exact_tostring_vectors() {
        // Literal vectors from test_rbigint.py::test_tostring.
        let zero = RBigInt::zero();
        assert_eq!(zero.str(0).unwrap(), "0");
        assert_eq!(zero.repr().unwrap(), "0L");
        assert_eq!(zero.hex().unwrap(), "0x0L");
        assert_eq!(zero.oct().unwrap(), "0L");

        let value = RBigInt::fromlong(-18_471_379_832_321);
        assert_eq!(value.str(0).unwrap(), "-18471379832321");
        assert_eq!(value.repr().unwrap(), "-18471379832321L");
        assert_eq!(value.hex().unwrap(), "-0x10ccb4088e01L");
        assert_eq!(value.oct().unwrap(), "-0414626402107001L");
        assert_eq!(
            value.format(".!", "", "", 0).unwrap(),
            "-!....!!..!!..!.!!.!......!...!...!!!........!"
        );
        assert_eq!(
            value.format("abcdefghijkl", "<<", ">>", 0).unwrap(),
            "-<<cakdkgdijffjf>>"
        );

        let huge =
            RBigInt::fromdecimalstr("-18471379832321000000000000000000000000000000000000000000");
        assert_eq!(
            huge.str(0).unwrap(),
            "-18471379832321000000000000000000000000000000000000000000"
        );
        assert_eq!(
            huge.repr().unwrap(),
            "-18471379832321000000000000000000000000000000000000000000L"
        );
        assert_eq!(
            huge.hex().unwrap(),
            "-0xc0d9a6f41fbcf1718b618443d45516a051e40000000000L"
        );
        assert_eq!(
            huge.oct().unwrap(),
            "-014033151572037571705614266060420752125055201217100000000000000L"
        );
    }

    #[test]
    fn test_recursive_decimal_and_general_format() {
        let power = RBigInt::fromint(10).int_pow(100, None).unwrap();
        let value = power.int_add(123);
        let expected = format!("1{}123", "0".repeat(97));
        assert_eq!(value.str(0).unwrap(), expected);
        assert_eq!(
            value.format(BASE10, "prefix:", ":suffix", 0).unwrap(),
            format!("prefix:{expected}:suffix")
        );
        assert_eq!(value.str(50), Err(RBigIntError::MaxStrDigits));

        let three_power = RBigInt::fromint(3).int_pow(200, None).unwrap();
        assert_eq!(
            three_power.format("012", "", "", 0).unwrap(),
            format!("1{}", "0".repeat(200))
        );
        // Exercise the process-global cache twice; contents and output must be
        // stable rather than rebuilt per thread or per call.
        assert_eq!(value.str(0).unwrap(), expected);
    }

    #[test]
    fn test_parts_cache_decimal_prebuilt_owner_and_snapshot_identity() {
        initialize_rbigint_parts_cache();

        let cache_from_owner = {
            let all = PARTS_CACHE.lock();
            assert_eq!(all.len(), 34);
            all[10 - 3]
                .as_ref()
                .expect("_parts_cache_10 must be eagerly populated")
                .clone()
        };
        let cache_from_lookup = get_cached_parts(10);
        assert!(std::sync::Arc::ptr_eq(
            &cache_from_owner,
            &cache_from_lookup
        ));

        let first = cache_from_lookup.parts_snapshot();
        let second = cache_from_lookup.parts_snapshot();
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "a cache hit must clone one shared-list handle, not the full parts vector"
        );
        assert_eq!(cache_from_lookup.mindigits, 18);
        assert_eq!(cache_from_lookup.lowest_part, 10_i64.pow(18));
        assert!(first[0].int_eq(10_i64.pow(18)));
    }

    #[test]
    fn test_parts_cache_growth_preserves_cached_rbigint_identity() {
        let cache = get_cached_parts(33);
        let before = cache.parts_snapshot();
        let target = before
            .last()
            .expect("parts cache starts non-empty")
            .int_add(1);

        let mut after = before.clone();
        while after
            .last()
            .expect("parts cache starts non-empty")
            .as_ref()
            .lt(&target)
        {
            cache.append_square(&after).unwrap();
            after = cache.parts_snapshot();
        }

        assert!(after.len() > before.len());
        for (old, current) in before.iter().zip(after.iter()) {
            assert!(
                std::sync::Arc::ptr_eq(old, current),
                "list growth must retain the exact cached rbigint object"
            );
        }
    }

    #[test]
    fn test_parts_cache_concurrent_append_converges_on_one_shared_list() {
        let cache = get_cached_parts(29);
        let before = cache.parts_snapshot();
        let target = std::sync::Arc::new(
            before
                .last()
                .expect("parts cache starts non-empty")
                .int_pow(16, None)
                .unwrap(),
        );
        let expected_len = before.len() + 4;

        let mut workers = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let target = target.clone();
            workers.push(std::thread::spawn(move || {
                loop {
                    let parts = cache.parts_snapshot();
                    if !parts
                        .last()
                        .expect("parts cache starts non-empty")
                        .as_ref()
                        .lt(target.as_ref())
                    {
                        break;
                    }
                    cache.append_square(&parts).unwrap();
                }
            }));
        }
        for worker in workers {
            worker
                .join()
                .expect("parts-cache formatter worker panicked");
        }

        let after = cache.parts_snapshot();
        assert_eq!(after.len(), expected_len);
        assert!(
            after
                .last()
                .expect("parts cache starts non-empty")
                .as_ref()
                .eq(target.as_ref())
        );
        for (old, current) in before.iter().zip(after.iter()) {
            assert!(std::sync::Arc::ptr_eq(old, current));
        }
    }

    #[test]
    fn test_decimal_recursive_parts_keep_lowest_level_bounds() {
        let source = "69092838422151607430870816441919864609";
        let value = RBigInt::fromdecimalstr(source);
        let cache = get_cached_parts(10);
        let mut parts = cache.parts_snapshot();
        assert_eq!(cache.mindigits, 18);
        assert_eq!(cache.lowest_part, 10_i64.pow(18));
        assert!(parts[0].int_eq(10_i64.pow(18)));
        let squared = RBigInt::fromdecimalstr("1000000000000000000000000000000000000");
        while parts
            .last()
            .expect("parts cache starts non-empty")
            .as_ref()
            .lt(&value)
        {
            cache.append_square(&parts).unwrap();
            parts = cache.parts_snapshot();
        }
        assert!(parts[1].as_ref().eq(&squared));

        let (top, bottom) = value.divmod(&parts[1]).unwrap();
        assert!(top.int_eq(69));
        assert!(top.mul(&parts[1]).add(&bottom).eq(&value));
        let (high, low) = _format_lowest_level_divmod_int_results(&bottom, cache.lowest_part);
        assert!(high < cache.lowest_part);
        assert!(low < cache.lowest_part);
        assert_eq!(value.str(0).unwrap(), source);
    }

    #[test]
    fn add_sub_mul_and_bitwise_match_i128_arithmetic() {
        // The residual `jit_bigint_*` paths call these bodies directly on the
        // operand payloads; the digit-list malloc inside them is the one
        // collecting call, rooted as `gct_fv_gc_malloc` roots its livevars.
        let pairs = [
            (0_i64, 0_i64),
            (0, 1),
            (1, 0),
            (i64::MAX, 1),
            (i64::MAX, i64::MAX),
            (i64::MIN, i64::MIN),
            (-1, 1),
            (i64::MIN, 1),
            (-7, -3),
        ];
        for (left, right) in pairs {
            let a = RBigInt::fromint(left);
            let b = RBigInt::fromint(right);
            let (l, r) = (left as i128, right as i128);
            assert_eq!(a.add(&b).to_i128(), Some(l + r), "{left} + {right}");
            assert_eq!(a.sub(&b).to_i128(), Some(l - r), "{left} - {right}");
            assert_eq!(a.mul(&b).to_i128(), Some(l * r), "{left} * {right}");
            assert_eq!(a.and_(&b).to_i128(), Some(l & r), "{left} & {right}");
            assert_eq!(a.or_(&b).to_i128(), Some(l | r), "{left} | {right}");
            assert_eq!(a.xor(&b).to_i128(), Some(l ^ r), "{left} ^ {right}");
        }
        // Two-limb operands so `_x_add` / `_x_mul` walk more than one digit.
        let a = RBigInt::fromint(i64::MAX)
            .lshift(SHIFT)
            .unwrap()
            .int_add(i64::MAX);
        let b = a.translated_alias();
        let a_value = a.to_i128().expect("two-limb operand fits in i128");
        assert_eq!(a.add(&b).to_i128(), a_value.checked_mul(2));
        assert_eq!(a.sub(&b).get_sign(), 0);
        assert!(a.mul(&b).floordiv(&b).unwrap().eq(&a));
        assert!(a.and_(&b).eq(&a));
        assert!(a.or_(&b).eq(&a));
        assert_eq!(a.xor(&b).get_sign(), 0);
    }
}
