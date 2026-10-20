//! Port of `rpython/rtyper/lltypesystem/opimpl.py`.
//!
//! Upstream defines per-`LLOp` fold callables that
//! `translator/backendopt/constfold.py:34-46` invokes via
//! `op = getattr(llop, spaceop.opname); op(RESTYPE, *args)`. Each
//! callable raises on type mismatch; the constfold catch-all converts
//! that into "do not fold". The Rust port mirrors the same shape: each
//! `op_<name>` takes `&[ConstValue]` and returns `Option<ConstValue>`,
//! where `None` is the convergent analogue of upstream's exception
//! path (TypeError / OverflowError / AssertionError → no fold).
//!
//! Coverage: every op with a primitive carrier in [`ConstValue`] is
//! ported (Int / Bool / Float / ByteStr / UniStr / LLPtr).
//! Ops requiring lltype-runtime carriers (`r_longlonglong`,
//! `r_ulonglonglong`, `lltype._parentable`) intentionally have no fold
//! here — once the backing carrier lands in `lltype.rs`, the
//! corresponding `op_<name>` function lifts in line-by-line from
//! upstream.
//!
//! Symbolic-carrier branches (PARITY by structural unreachability):
//! upstream `op_int_add` `opimpl.py`, `op_int_sub` `:215-220`,
//! `op_int_mul` `:269-272`, `op_int_eq` `:107-118` carry
//! `assert isinstance(x, …, llmemory.AddressOffset)` /
//! `assert isinstance(x, llgroup.CombinedSymbolic)` clauses that are
//! pure type-validation: they raise on a wrong-typed input, which
//! `constfold.py:34-46` catches as "do not fold". The Rust port's
//! match arms have the equivalent shape — non-`Int` inputs return
//! `None`, the convergent analogue. `op_int_xor` `:261-267` does
//! convert `AddressAsInt` via `cast_adr_to_int(x.adr)`, but a
//! symbolic-int input never reaches the Rust folder because
//! [`ConstValue`] carries no `AddressAsInt` variant — call sites
//! producing such an input simply never appear. Adding the symbolic-
//! int variants to `ConstValue` would let those branches fire and
//! match upstream byte-for-byte; until then the absence of the
//! branches is observably equivalent, not divergent.
//!
//! Convergence path: `lloperation.rs::LLOp` should grow a
//! `fold: Option<fn(&[ConstValue]) -> Option<ConstValue>>` field that
//! the registry below populates, so `constfold.rs::eval_llop` collapses
//! to `op_desc.fold.and_then(|f| f(args))`. The current free-function
//! registry [`get_op_impl`] is the smaller-scope precursor.

#![allow(non_camel_case_types, non_upper_case_globals)]

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::llmemory::{cast_adr_to_ptr, cast_int_to_adr};
use crate::translator::rtyper::lltypesystem::lltype::{_address, LowLevelType};

/// Fold callable signature. `None` mirrors upstream's "exception →
/// no fold" path (`constfold.py:34-46`).
pub type FoldFn = fn(&[ConstValue]) -> Option<ConstValue>;

/// RPython `ops_returning_a_bool`.
pub const ops_returning_a_bool: &[&str] = &["gt", "ge", "lt", "le", "eq", "ne", "bool", "is_true"];

/// RPython `r_longlonglong_arg = r_longlonglong`.
pub type r_longlonglong_arg = i128;

/// RPython `r_longlonglong_result = r_longlonglong`.
pub type r_longlonglong_result = i128;

/// RPython `argtype_by_name` keys. The values are host Python type objects
/// upstream; the Rust fold path enforces them through `ConstValue` carriers.
pub const argtype_by_name: &[&str] = &[
    "int", "float", "uint", "llong", "ullong", "lllong", "ulllong",
];

/// RPython `no_op(x)`.
pub fn no_op(value: &ConstValue) -> ConstValue {
    value.clone()
}

/// RPython `_normalize(x)`. Pointer-to-rpy-string normalization depends on
/// the live lltype string object; constant folding keeps the value untouched.
pub fn _normalize(value: &ConstValue) -> ConstValue {
    value.clone()
}

/// RPython `op_gc_writebarrier_before_move(array)`.
pub fn op_gc_writebarrier_before_move(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_debug_print(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_debug_start(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_debug_stop(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_have_debug_prints_for(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_jit_record_exact_value(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_jit_enter_portal_frame(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_jit_leave_portal_frame(_args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_raw_load(_restype: &LowLevelType, _args: &[ConstValue]) -> Option<ConstValue> {
    None
}

pub fn op_gc_load_indexed(_restype: &LowLevelType, _args: &[ConstValue]) -> Option<ConstValue> {
    None
}

/// RPython `opimpl.op_bool_not` (`opimpl.py`).
pub fn op_bool_not(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::Bool(!b)),
        _ => None,
    }
}

/// RPython `opimpl.op_same_as` (`opimpl.py`).
pub fn op_same_as(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [value] => Some(value.clone()),
        _ => None,
    }
}

// int_*

/// RPython `opimpl.op_int_is_true` derived from
/// `flowspace.operation.op.is_true.pyfunc` (`opimpl.py get_primitive_op_src
/// get_primitive_op_src`). For Python `int` the truth test is
/// `value != 0`.
pub fn op_int_is_true(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => Some(ConstValue::Bool(*n != 0)),
        _ => None,
    }
}

/// `op_int_neg` — `intmask(-x)` (`opimpl.py get_primitive_op_src`).
pub fn op_int_neg(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => Some(ConstValue::Int(n.wrapping_neg())),
        _ => None,
    }
}

/// `op_int_abs` — `intmask(abs(x))`. `abs(MIN)` overflows back to
/// `MIN` under intmask wrap-around, which matches `i64::wrapping_abs`.
pub fn op_int_abs(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => Some(ConstValue::Int(n.wrapping_abs())),
        _ => None,
    }
}

/// `op_int_invert` — `intmask(~x)`.
pub fn op_int_invert(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => Some(ConstValue::Int(!n)),
        _ => None,
    }
}

/// RPython `op_int_add` (`opimpl.py`) — `intmask(x + y)`.
pub fn op_int_add(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a.wrapping_add(*b))),
        _ => None,
    }
}

/// RPython `op_int_sub` (`opimpl.py`).
pub fn op_int_sub(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a.wrapping_sub(*b))),
        _ => None,
    }
}

/// RPython `op_int_mul` (`opimpl.py`).
pub fn op_int_mul(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a.wrapping_mul(*b))),
        _ => None,
    }
}

/// RPython `op_int_floordiv` (`opimpl.py`). Upstream computes
/// `r = x // y` (Python floor div) then adjusts by `+1` when the
/// signs of `x` and `y` differ and the remainder is non-zero — the
/// net effect is **C-style truncating division**. Rust's
/// `i64::checked_div` is already C truncating, so the per-bit
/// adjustment in upstream is unnecessary here. The `INT_MIN / -1`
/// overflow corresponds to upstream's `lltype.enforce(Signed, 2**63)`
/// raising `OverflowError`; `checked_div` returning `None` matches
/// the ZeroDivisionError path too.
pub fn op_int_floordiv(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(_), ConstValue::Int(0)] => None,
        [ConstValue::Int(a), ConstValue::Int(b)] => a.checked_div(*b).map(ConstValue::Int),
        _ => None,
    }
}

/// RPython `op_int_mod` (`opimpl.py`). Same Python-floor →
/// C-truncate adjustment as `op_int_floordiv`. Rust's
/// `i64::wrapping_rem` is already C truncating, including
/// `wrapping_rem(MIN, -1) = 0`, which matches upstream where
/// `enforce(Signed, 0)` succeeds.
pub fn op_int_mod(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(_), ConstValue::Int(0)] => None,
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a.wrapping_rem(*b))),
        _ => None,
    }
}

/// RPython `op_int_lshift` derived via `get_primitive_op_src`. Python
/// `x << y` with non-negative `y` is mathematical multiplication by
/// `2**y`; `intmask` truncates to `Signed`. For `y >= 64` every bit
/// of `x` shifts past the sign position, so the truncated result is
/// `0`. Negative `y` raises `ValueError` upstream → no fold.
pub fn op_int_lshift(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => {
            if *b < 0 {
                return None;
            }
            if *b >= i64::BITS as i64 {
                Some(ConstValue::Int(0))
            } else {
                Some(ConstValue::Int(a.wrapping_shl(*b as u32)))
            }
        }
        _ => None,
    }
}

/// RPython `op_int_rshift` — Python arithmetic right shift. `y >= 64`
/// collapses to sign-extension (-1 for negative `x`, 0 otherwise).
/// Negative `y` raises `ValueError` upstream → no fold.
pub fn op_int_rshift(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => {
            if *b < 0 {
                return None;
            }
            if *b >= i64::BITS as i64 {
                Some(ConstValue::Int(if *a < 0 { -1 } else { 0 }))
            } else {
                Some(ConstValue::Int(*a >> *b))
            }
        }
        _ => None,
    }
}

/// RPython `op_int_and`, `op_int_or`, `op_int_xor` (`opimpl.py`).
pub fn op_int_and(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a & b)),
        _ => None,
    }
}

pub fn op_int_or(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a | b)),
        _ => None,
    }
}

pub fn op_int_xor(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(a ^ b)),
        _ => None,
    }
}

/// RPython `op_int_eq`, `op_int_ne`, `op_int_lt`, `op_int_le`,
/// `op_int_gt`, `op_int_ge` (`opimpl.py` + comparators
/// derived via `get_primitive_op_src`).
pub fn op_int_lt(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a < b)),
        _ => None,
    }
}

pub fn op_int_le(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a <= b)),
        _ => None,
    }
}

pub fn op_int_eq(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a == b)),
        _ => None,
    }
}

pub fn op_int_ne(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a != b)),
        _ => None,
    }
}

pub fn op_int_gt(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a > b)),
        _ => None,
    }
}

pub fn op_int_ge(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Bool(a >= b)),
        _ => None,
    }
}

/// RPython `op_int_between(a, b, c)` (`opimpl.py`) —
/// `a <= b < c`.
pub fn op_int_between(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a), ConstValue::Int(b), ConstValue::Int(c)] => {
            Some(ConstValue::Bool(a <= b && b < c))
        }
        _ => None,
    }
}

/// RPython `op_int_force_ge_zero(a)` (`opimpl.py`) —
/// `0 if a < 0 else a`.
pub fn op_int_force_ge_zero(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(a)] => Some(ConstValue::Int(if *a < 0 { 0 } else { *a })),
        _ => None,
    }
}

// float_*

/// `op_float_*` are derived via `get_primitive_op_src` (`opimpl.py`)
/// for `argtype = float`. RPython folds them as direct IEEE 754
/// arithmetic; only division by zero raises `ZeroDivisionError`
/// upstream, so that case refuses to fold.
fn float_pair(args: &[ConstValue]) -> Option<(f64, f64)> {
    match args {
        [ConstValue::Float(a), ConstValue::Float(b)] => {
            Some((f64::from_bits(*a), f64::from_bits(*b)))
        }
        _ => None,
    }
}

pub fn op_float_is_true(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => Some(ConstValue::Bool(f64::from_bits(*bits) != 0.0)),
        _ => None,
    }
}

pub fn op_float_neg(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => Some(ConstValue::float(-f64::from_bits(*bits))),
        _ => None,
    }
}

pub fn op_float_abs(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => Some(ConstValue::float(f64::from_bits(*bits).abs())),
        _ => None,
    }
}

pub fn op_float_add(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::float(a + b))
}

pub fn op_float_sub(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::float(a - b))
}

pub fn op_float_mul(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::float(a * b))
}

pub fn op_float_truediv(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    if b == 0.0 {
        // Python `x / 0.0` raises ZeroDivisionError → no fold.
        return None;
    }
    Some(ConstValue::float(a / b))
}

pub fn op_float_lt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a < b))
}

pub fn op_float_le(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a <= b))
}

pub fn op_float_eq(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a == b))
}

pub fn op_float_ne(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a != b))
}

pub fn op_float_gt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a > b))
}

pub fn op_float_ge(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = float_pair(args)?;
    Some(ConstValue::Bool(a >= b))
}

// llong_*
//
// Upstream `op_llong_*` (`opimpl.py op_llong_floordiv`) operates on `r_longlong_arg`,
// which on 64-bit hosts aliases the regular Python int (see
// `opimpl.py:23-28`: `if r_longlong is r_int: r_longlong_arg =
// (r_longlong, int, long)`). The Rust port keeps the
// [`ConstValue::Int(i64)`] carrier for `LowLevelType::SignedLongLong`
// per `lltype.rs`, so every `llong_*` arm mirrors its
// `int_*` counterpart. On 32-bit hosts this would diverge — convergence
// path: add a separate `ConstValue::LongLong` variant once
// `lltype.rs` carries `r_longlong` distinctly.

/// RPython `op_llong_is_true` (derived via `get_primitive_op_src`,
/// `opimpl.py`).
pub fn op_llong_is_true(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_is_true(args)
}

pub fn op_llong_neg(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_neg(args)
}

pub fn op_llong_abs(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_abs(args)
}

pub fn op_llong_invert(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_invert(args)
}

pub fn op_llong_add(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_add(args)
}

pub fn op_llong_sub(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_sub(args)
}

pub fn op_llong_mul(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_mul(args)
}

/// RPython `op_llong_floordiv` (`opimpl.py`). Same C-truncating
/// semantics as `op_int_floordiv`.
pub fn op_llong_floordiv(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_floordiv(args)
}

/// RPython `op_llong_mod` (`opimpl.py`).
pub fn op_llong_mod(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_mod(args)
}

/// RPython `op_llong_lshift` (`opimpl.py`) — `r_longlong_result(x << y)`.
pub fn op_llong_lshift(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_lshift(args)
}

/// RPython `op_llong_rshift` (`opimpl.py`).
pub fn op_llong_rshift(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_rshift(args)
}

pub fn op_llong_and(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_and(args)
}

pub fn op_llong_or(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_or(args)
}

pub fn op_llong_xor(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_xor(args)
}

pub fn op_llong_lt(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_lt(args)
}

pub fn op_llong_le(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_le(args)
}

pub fn op_llong_eq(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_eq(args)
}

pub fn op_llong_ne(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_ne(args)
}

pub fn op_llong_gt(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_gt(args)
}

pub fn op_llong_ge(args: &[ConstValue]) -> Option<ConstValue> {
    op_int_ge(args)
}

// uint_* and ullong_*
//
// Upstream `op_uint_*` (`opimpl.py op_uint_lshift`) operates on `r_uint` —
// `unsigned long` in C, `u64` on 64-bit hosts. The carrier is
// [`ConstValue::UInt`] (`get_primitive_op_src` checks `isinstance(x, r_uint)`).
// Results are `UInt` so `Hashable.key = (type(value), value)` keeps them
// apart from signed `Int`. On a 64-bit host `r_ulonglong is r_uint`
// (`rarithmetic.build_int`).
//
// The wrap-around math (add/sub/mul/and/or/xor/eq/ne) is identical to
// the signed family — `i64::wrapping_add` and `u64::wrapping_add`
// produce the same bit pattern. The differences land in:
//   - comparisons (`uint_lt`/`le`/`gt`/`ge`): unsigned ordering
//   - `uint_floordiv`/`uint_mod`: unsigned division semantics
//   - `uint_rshift`: logical (zero-fill) right shift

fn uint_bits(value: &ConstValue) -> Option<u64> {
    // `get_primitive_op_src` checks `isinstance(x, r_uint)`.
    match value {
        ConstValue::UInt(n) => Some(*n),
        _ => None,
    }
}

fn uint_pair(args: &[ConstValue]) -> Option<(u64, u64)> {
    match args {
        [a, b] => Some((uint_bits(a)?, uint_bits(b)?)),
        _ => None,
    }
}

fn uint_arg(args: &[ConstValue]) -> Option<u64> {
    match args {
        [a] => uint_bits(a),
        _ => None,
    }
}

fn uint_const(n: u64) -> ConstValue {
    ConstValue::UInt(n)
}

/// `rarithmetic.is_valid_int` is range-based: it accepts `r_uint`
/// values inside the signed machine range (`-maxint-1 <= r <= maxint`).
fn valid_int(value: &ConstValue) -> Option<i64> {
    match value {
        ConstValue::Int(n) => Some(*n),
        ConstValue::UInt(n) if *n <= i64::MAX as u64 => Some(*n as i64),
        _ => None,
    }
}

/// RPython `op_uint_is_true` derived via `get_primitive_op_src`.
pub fn op_uint_is_true(args: &[ConstValue]) -> Option<ConstValue> {
    Some(ConstValue::Bool(uint_arg(args)? != 0))
}

pub fn op_uint_invert(args: &[ConstValue]) -> Option<ConstValue> {
    Some(uint_const(!uint_arg(args)?))
}

pub fn op_uint_add(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a.wrapping_add(b)))
}

pub fn op_uint_sub(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a.wrapping_sub(b)))
}

pub fn op_uint_mul(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a.wrapping_mul(b)))
}

/// High 64 bits of the unsigned 64x64 -> 128 product — the overflow high
/// word `checked_mul` tests. Matches the backend `OpCode::UintMulHigh`
/// (`umulhi`) and `bhimpl_uint_mul_high`.
pub fn op_uint_mul_high(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const((((a as u128) * (b as u128)) >> 64) as u64))
}

pub fn op_uint_floordiv(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    if b == 0 {
        return None;
    }
    Some(uint_const(a / b))
}

pub fn op_uint_mod(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    if b == 0 {
        return None;
    }
    Some(uint_const(a % b))
}

pub fn op_uint_and(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a & b))
}

pub fn op_uint_or(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a | b))
}

pub fn op_uint_xor(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(uint_const(a ^ b))
}

/// RPython `op_uint_lshift` (`opimpl.py`) — `r_uint(x << y)`.
/// `assert isinstance(x, r_uint)` and `assert is_valid_int(y)`.
/// `rarithmetic.is_valid_int` accepts `r_uint` counts in the signed
/// machine range. Negative `y` raises ValueError upstream → no fold;
/// `y >= 64` truncates to 0 in u64 space.
pub fn op_uint_lshift(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::UInt(a), b] => {
            let b = valid_int(b)?;
            if b < 0 {
                return None;
            }
            if b >= u64::BITS as i64 {
                Some(uint_const(0))
            } else {
                Some(uint_const(a.wrapping_shl(b as u32)))
            }
        }
        _ => None,
    }
}

/// RPython `op_uint_rshift` (`opimpl.py`) — logical right
/// shift in u64 space. `assert isinstance(x, r_uint)` and
/// `assert is_valid_int(y)`. `rarithmetic.is_valid_int` accepts
/// `r_uint` counts in the signed machine range. `y >= 64`
/// collapses to 0.
pub fn op_uint_rshift(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::UInt(a), b] => {
            let b = valid_int(b)?;
            if b < 0 {
                return None;
            }
            if b >= u64::BITS as i64 {
                Some(uint_const(0))
            } else {
                Some(uint_const(a >> (b as u32)))
            }
        }
        _ => None,
    }
}

pub fn op_uint_lt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a < b))
}

pub fn op_uint_le(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a <= b))
}

pub fn op_uint_eq(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a == b))
}

pub fn op_uint_ne(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a != b))
}

pub fn op_uint_gt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a > b))
}

pub fn op_uint_ge(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = uint_pair(args)?;
    Some(ConstValue::Bool(a >= b))
}

// `ullong_*` shares the u64 carrier with `uint_*` on the host targets
// pyre actually compiles for. Each `op_ullong_*` delegates per
// upstream's `r_ulonglong` semantics matching `r_uint` semantics for
// 64-bit-wide unsigned arithmetic.

pub fn op_ullong_is_true(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_is_true(args)
}

pub fn op_ullong_invert(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_invert(args)
}

pub fn op_ullong_add(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_add(args)
}

pub fn op_ullong_sub(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_sub(args)
}

pub fn op_ullong_mul(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_mul(args)
}

pub fn op_ullong_floordiv(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_floordiv(args)
}

pub fn op_ullong_mod(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_mod(args)
}

pub fn op_ullong_and(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_and(args)
}

pub fn op_ullong_or(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_or(args)
}

pub fn op_ullong_xor(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_xor(args)
}

/// RPython `op_ullong_lshift` (`opimpl.py`).
pub fn op_ullong_lshift(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_lshift(args)
}

/// RPython `op_ullong_rshift` (`opimpl.py`).
pub fn op_ullong_rshift(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_rshift(args)
}

pub fn op_ullong_lt(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_lt(args)
}

pub fn op_ullong_le(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_le(args)
}

pub fn op_ullong_eq(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_eq(args)
}

pub fn op_ullong_ne(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_ne(args)
}

pub fn op_ullong_gt(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_gt(args)
}

pub fn op_ullong_ge(args: &[ConstValue]) -> Option<ConstValue> {
    op_uint_ge(args)
}

// ptr_*

/// RPython `op_ptr_eq` (`opimpl.py`).
///
/// Upstream calls `checkptr(ptr1); checkptr(ptr2)` first, which
/// raises when either argument is not an `lltype.Ptr` value. The
/// Rust port returns `None` (refuse to fold) when either operand is
/// not an `LLPtr` — matching upstream's "non-pointer args don't
/// fold" semantics. Pointer identity over the `LLPtr` carrier lives
/// in [`ConstValue::PartialEq`] (uses `_hashable_identity`), so the
/// LLPtr/LLPtr case delegates.
pub fn op_ptr_eq(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [a @ ConstValue::LLPtr(_), b @ ConstValue::LLPtr(_)] => Some(ConstValue::Bool(a == b)),
        _ => None,
    }
}

/// RPython `op_ptr_ne` (`opimpl.py`). Same `checkptr`
/// discipline as `op_ptr_eq` — non-pointer operands refuse to fold.
pub fn op_ptr_ne(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [a @ ConstValue::LLPtr(_), b @ ConstValue::LLPtr(_)] => Some(ConstValue::Bool(a != b)),
        _ => None,
    }
}

/// RPython `op_ptr_nonzero` (`opimpl.py`) — `bool(ptr)`.
/// Upstream's `checkptr(p)` enforces `Ptr` typeOf; the Rust port
/// matches the `LLPtr` carrier.
pub fn op_ptr_nonzero(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLPtr(p)] => Some(ConstValue::Bool(p.nonzero())),
        _ => None,
    }
}

/// RPython `op_ptr_iszero` (`opimpl.py`) — `not bool(ptr)`.
pub fn op_ptr_iszero(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLPtr(p)] => Some(ConstValue::Bool(!p.nonzero())),
        _ => None,
    }
}

/// RPython `op_adr_eq` (`opimpl.py`): `checkadr; return addr1 ==
/// addr2`. The `checkadr` discipline (operands must be `Address`) is the
/// `LLAddress` carrier match; non-address operands refuse to fold. The
/// `==` delegates to `fakeaddress.__eq__` ([`_address::_eq`]).
pub fn op_adr_eq(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => Some(ConstValue::Bool(a._eq(b))),
        _ => None,
    }
}

/// RPython `op_adr_ne` (`opimpl.py`): `checkadr; return addr1 !=
/// addr2`, delegating to `fakeaddress.__ne__` ([`_address::_ne`]).
pub fn op_adr_ne(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => Some(ConstValue::Bool(a._ne(b))),
        _ => None,
    }
}

/// RPython `op_adr_lt` (`opimpl.py`): `checkadr; return addr1 <
/// addr2`. `fakeaddress.__lt__` raises `TypeError` for two non-NULL
/// addresses; `Err` there maps to "no fold" (the op stays runtime).
pub fn op_adr_lt(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => a._lt(b).ok().map(ConstValue::Bool),
        _ => None,
    }
}

/// RPython `op_adr_le` (`opimpl.py`): `checkadr; return addr1 <=
/// addr2`, delegating to `fakeaddress.__le__` ([`_address::_le`]).
pub fn op_adr_le(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => a._le(b).ok().map(ConstValue::Bool),
        _ => None,
    }
}

/// RPython `op_adr_gt` (`opimpl.py`): `checkadr; return addr1 >
/// addr2`, delegating to `fakeaddress.__gt__` ([`_address::_gt`]).
pub fn op_adr_gt(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => a._gt(b).ok().map(ConstValue::Bool),
        _ => None,
    }
}

/// RPython `op_adr_ge` (`opimpl.py`): `checkadr; return addr1 >=
/// addr2`, delegating to `fakeaddress.__ge__` ([`_address::_ge`]).
pub fn op_adr_ge(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => a._ge(b).ok().map(ConstValue::Bool),
        _ => None,
    }
}

/// RPython `op_adr_delta` (`opimpl.py`): `checkadr(addr1);
/// checkadr(addr2); return addr1 - addr2`. With two fakeaddresses the
/// subtraction ([`_address::_delta`], `fakeaddress.__sub__` llmemory.py)
/// yields `Signed 0` when the addresses are equal and raises `TypeError`
/// otherwise; the `Err` maps to "no fold" so the op stays runtime.
///
pub fn op_adr_delta(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(a), ConstValue::LLAddress(b)] => {
            a._delta(b).ok().map(ConstValue::Int)
        }
        _ => None,
    }
}

/// RPython `op_adr_add` (`opimpl.py`): `checkadr(addr); assert
/// typeOf(offset) is Signed; return addr + offset`. `fakeaddress.__add__`
/// (llmemory.py) raises `NullAddressError` on a NULL base and
/// otherwise returns `fakeaddress(offset.ref(self.ptr))`. The offset is the
/// `ConstValue::AddressOffset` carrier; [`AddressOffset::r#ref`] declines
/// (→ no fold) for the bases / element kinds it cannot navigate yet.
pub fn op_adr_add(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(addr), ConstValue::AddressOffset(off)] => {
            // Only a `Fake` base carries a `_ptr` to navigate; a NULL base is
            // `NullAddressError` and an int-cast base has no container pointer
            // — both decline to fold.
            let _address::Fake(ptr) = addr else {
                return None;
            };
            off.r#ref(ptr)
                .ok()
                .map(|p| ConstValue::LLAddress(_address::Fake(Box::new(p))))
        }
        // `fakeaddress.__add__`: a plain `0` offset returns the address
        // unchanged (llmemory.py). The `offset` is typed `Signed`,
        // which admits an `Int` carrier as well as `AddressOffset`
        // (`Signed::contains_value`).
        [ConstValue::LLAddress(addr), ConstValue::Int(0)] => {
            Some(ConstValue::LLAddress(addr.clone()))
        }
        _ => None,
    }
}

/// RPython `op_adr_sub` (`opimpl.py`): `addr - offset`. With an
/// `AddressOffset` operand `fakeaddress.__sub__` is `self + (-offset)`
/// (llmemory.py); a non-negatable offset (`AddressOffset::neg` →
/// `None`) declines, as does the underlying `op_adr_add`.
pub fn op_adr_sub(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::LLAddress(addr), ConstValue::AddressOffset(off)] => {
            let neg = off.clone().neg()?;
            op_adr_add(&[
                ConstValue::LLAddress(addr.clone()),
                ConstValue::AddressOffset(neg),
            ])
        }
        // `fakeaddress.__sub__`: a plain `0` offset returns self
        // (llmemory.py).
        [ConstValue::LLAddress(addr), ConstValue::Int(0)] => {
            Some(ConstValue::LLAddress(addr.clone()))
        }
        _ => None,
    }
}

/// RPython `op_cast_int_to_adr(int)` (`opimpl.py`):
/// `llmemory.cast_int_to_adr(int)` (llmemory.py). Folds the
/// `cast_int_to_ptr` + `cast_ptr_to_adr` composition: `0` → NULL address,
/// an odd integer → a tagged-integer `_NONGCREF` `_ptr` wrapped as the
/// fakeaddress (`_address::Fake`), an even non-zero integer → declines (its
/// upstream `ll2ctypes` resolution is runtime-only). See [`cast_int_to_adr`].
pub fn op_cast_int_to_adr(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => cast_int_to_adr(*n).map(ConstValue::LLAddress),
        _ => None,
    }
}

/// RPython `op_cast_adr_to_ptr(TYPE, adr)` (`opimpl.py`,
/// `need_result_type = True`): `llmemory.cast_adr_to_ptr(adr, TYPE)`. `TYPE`
/// is the operation result's concretetype (`constfold.py:36` passes RESTYPE
/// first); it must be a `Ptr`. An int-cast address declines (no tagged-int
/// round-trip ported).
pub fn op_cast_adr_to_ptr(restype: &LowLevelType, args: &[ConstValue]) -> Option<ConstValue> {
    let [ConstValue::LLAddress(adr)] = args else {
        return None;
    };
    let LowLevelType::Ptr(expected) = restype else {
        return None;
    };
    cast_adr_to_ptr(adr, expected)
        .ok()
        .map(|p| ConstValue::LLPtr(Box::new(p)))
}

// cast_*_to_* (primitive-only carriers)

/// RPython `op_cast_int_to_float` (`opimpl.py`).
pub fn op_cast_int_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(i)] => Some(ConstValue::float(*i as f64)),
        _ => None,
    }
}

/// RPython `op_cast_float_to_int` (`opimpl.py`) —
/// `intmask(int(f))`. NaN / ±inf raise upstream and refuse to fold.
/// Finite floats — including those outside `[i64::MIN, i64::MAX]` —
/// fold to the exact 64-bit truncation of `int(f)`, matching
/// upstream's arbitrary-precision `int()` followed by a 64-bit
/// `intmask`. The bit-decomposition path is shared with
/// [`op_cast_float_to_longlong`] via [`float_to_signed64_intmask`].
pub fn op_cast_float_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => {
            let f = f64::from_bits(*bits);
            if !f.is_finite() {
                return None;
            }
            Some(ConstValue::Int(float_to_signed64_intmask(f)))
        }
        _ => None,
    }
}

/// `intmask(int(f))` semantics for a finite `f`. Decomposes the
/// IEEE 754 representation directly:
///
/// * `mantissa` = `(bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000`
///   (53-bit unsigned, with the implicit leading 1 reattached).
/// * `unbiased = exp - 1023 - 52` — power of two the mantissa is
///   scaled by.
/// * Right-shift the mantissa by `-unbiased` when negative
///   (truncates toward zero), left-shift it by `unbiased` modulo
///   `2^64` when non-negative. Apply the sign by two's-complement
///   negation.
///
/// Returns `0` for `f == 0.0` (both signs). Returns `0` when
/// `unbiased >= 64` (the integer is divisible by `2^64`).
fn float_to_signed64_intmask(f: f64) -> i64 {
    if f == 0.0 {
        return 0;
    }
    let bits = f.to_bits();
    let sign = (bits >> 63) & 1;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        // Subnormal — magnitude < 1.0, truncates to 0.
        return 0;
    }
    let mantissa: u64 = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
    let unbiased = exp - 1023 - 52;
    let raw: u64 = if unbiased >= 0 {
        if unbiased >= 64 {
            0
        } else {
            mantissa.wrapping_shl(unbiased as u32)
        }
    } else {
        let shift = (-unbiased) as u32;
        if shift >= 64 { 0 } else { mantissa >> shift }
    };
    let abs = raw as i64;
    if sign == 1 { abs.wrapping_neg() } else { abs }
}

/// RPython `op_cast_bool_to_int` (`opimpl.py`).
pub fn op_cast_bool_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::Int(*b as i64)),
        _ => None,
    }
}

/// RPython `op_cast_bool_to_float` (`opimpl.py`).
pub fn op_cast_bool_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::float(if *b { 1.0 } else { 0.0 })),
        _ => None,
    }
}

/// RPython `op_cast_int_to_char` (`opimpl.py`) —
/// `chr(b)`. Out-of-range ints raise ValueError upstream → no fold.
/// The Rust carrier for `char` is a length-1 [`ConstValue::ByteStr`].
pub fn op_cast_int_to_char(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] if (0..=0xFF).contains(n) => Some(ConstValue::ByteStr(vec![*n as u8])),
        _ => None,
    }
}

/// RPython `op_cast_char_to_int` (`opimpl.py`) —
/// `ord(b)` on a length-1 string. Other shapes refuse to fold.
pub fn op_cast_char_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::ByteStr(bytes)] if bytes.len() == 1 => Some(ConstValue::Int(bytes[0] as i64)),
        _ => None,
    }
}

/// RPython `op_cast_int_to_unichar` (`opimpl.py`) —
/// `unichr(b)`. The Rust carrier for `unichar` is a length-1
/// [`ConstValue::UniStr`].
pub fn op_cast_int_to_unichar(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => {
            let cp = u32::try_from(*n).ok()?;
            let ch = char::from_u32(cp)?;
            Some(ConstValue::UniStr(ch.to_string()))
        }
        _ => None,
    }
}

/// RPython `op_cast_unichar_to_int` (`opimpl.py`) — `ord(b)`.
pub fn op_cast_unichar_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::UniStr(s)] => {
            let mut chars = s.chars();
            let first = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(ConstValue::Int(first as i64))
        }
        _ => None,
    }
}

// wide-int casts (i64 / u64 carrier preserves bit pattern)
//
// On 64-bit hosts signed long-long collapses onto [`ConstValue::Int`]
// (`SignedLongLong is Signed`) and unsigned long-long onto
// [`ConstValue::UInt`] (`r_ulonglong is r_uint`).
// The `cast_int_to_longlong` / `cast_int_to_uint` family is therefore
// identity at the bit-pattern level. Upstream's per-type asserts
// remain available for parity once the carriers diverge.

/// RPython `op_cast_int_to_uint` (`opimpl.py`) — `r_uint(b)`.
/// `assert is_valid_int(b)`. `rarithmetic.is_valid_int` is range-based
/// and accepts `r_uint` values inside the signed machine range.
pub fn op_cast_int_to_uint(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [b] => Some(ConstValue::UInt(valid_int(b)? as u64)),
        _ => None,
    }
}

/// RPython `op_cast_uint_to_int` (`opimpl.py`) — `intmask(b)`.
/// `assert type(b) is r_uint`.
pub fn op_cast_uint_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::UInt(n)] => Some(ConstValue::Int(*n as i64)),
        _ => None,
    }
}

/// RPython `op_cast_int_to_longlong` (`opimpl.py`).
/// `r_longlong_result(b)` is `long(b)` on 64-bit, which fits the
/// existing i64 carrier identically.
pub fn op_cast_int_to_longlong(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(_)] => Some(args[0].clone()),
        _ => None,
    }
}

/// RPython `op_truncate_longlong_to_int` (`opimpl.py`) —
/// `intmask(b)`. Identity on i64.
pub fn op_truncate_longlong_to_int(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(_)] => Some(args[0].clone()),
        _ => None,
    }
}

/// RPython `op_cast_bool_to_uint` (`opimpl.py`).
pub fn op_cast_bool_to_uint(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::UInt(u64::from(*b))),
        _ => None,
    }
}

/// RPython `op_cast_uint_to_float` (`opimpl.py`) — `float(u)`
/// in u64 space (preserves precision for values up to 2^53; large
/// u64 values are folded with the standard `u64 as f64` rounding,
/// matching upstream's `float(int(...))` lossy conversion).
pub fn op_cast_uint_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [n] => Some(ConstValue::float(uint_bits(n)? as f64)),
        _ => None,
    }
}

/// RPython `op_cast_longlong_to_float` (`opimpl.py`). On 64-bit
/// hosts where `r_longlong is r_int`, this is `float(i)` — identical
/// to `cast_int_to_float`.
pub fn op_cast_longlong_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    op_cast_int_to_float(args)
}

/// RPython `op_cast_ulonglong_to_float` (`opimpl.py`). On
/// 64-bit hosts this is `float(u64_value)`, matching
/// `op_cast_uint_to_float`.
pub fn op_cast_ulonglong_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    op_cast_uint_to_float(args)
}

/// RPython `op_cast_float_to_uint` (`opimpl.py`) —
/// `r_uint(long(f))`. Upstream's `long(f)` truncates toward zero
/// into a Python arbitrary-precision integer, and `r_uint(...)`
/// then wraps the value modulo 2^64 (on 64-bit hosts; modulo 2^32
/// on 32-bit). Pyre targets 64-bit, so the result is `trunc(f)
/// mod 2^64` for every finite `f`. NaN / inf surface as upstream's
/// `OverflowError` / `ValueError` and refuse to fold.
///
/// Implementation: `f as i64` / `f as u64` saturate outside
/// `[-2^63, 2^63)` / `[0, 2^64)` respectively, so they cannot be
/// used for floats with `|f| >= 2^63`. Compute the wrapped low 64
/// bits directly from the IEEE-754 mantissa + exponent — this is
/// exact for every finite f64 because every f64 with `|f| >= 2^53`
/// is already integral.
pub fn op_cast_float_to_uint(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => {
            let f = f64::from_bits(*bits);
            if !f.is_finite() {
                return None;
            }
            Some(ConstValue::UInt(float_trunc_mod_2_pow_64(f)))
        }
        _ => None,
    }
}

/// Compute `trunc(f) mod 2^64` for any finite `f64`, matching
/// upstream `r_uint(long(f))` / `r_ulonglong(long(f))` wrap
/// (`opimpl.py op_cast_float_to_uint`). The exact wrap is recoverable from the
/// IEEE-754 representation alone because:
///
/// * For `|f| < 1`: `trunc(f) == 0`.
/// * For `|f| >= 1`: the f64 mantissa carries 53 bits of integer
///   value at position `exp - 52`, where `exp` is the unbiased
///   exponent (`>= 0` since `|f| >= 1`). Shifting the mantissa by
///   `exp - 52` reconstructs the bottom 64 bits of the truncated
///   integer; bits that would land beyond position 63 are dropped
///   (which matches `mod 2^64`).
///
/// `panic`s only on caller misuse (NaN / inf) — caller filters.
fn float_trunc_mod_2_pow_64(f: f64) -> u64 {
    debug_assert!(f.is_finite());
    let bits = f.to_bits();
    let sign = (bits >> 63) & 1;
    let raw_exp = ((bits >> 52) & 0x7FF) as i64;
    if raw_exp == 0 {
        // Subnormal or signed zero — `|f| < 1`, `trunc(f) == 0`.
        return 0;
    }
    let exp = raw_exp - 1023;
    if exp < 0 {
        // `|f| < 1` (normal but below 1), `trunc(f) == 0`.
        return 0;
    }
    // Implicit leading 1 + 52 explicit mantissa bits = 53-bit
    // integer at position `exp - 52`.
    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let unsigned_trunc = if exp >= 52 {
        let shift = (exp - 52) as u32;
        if shift >= 64 {
            // Whole 53-bit integer shifts out of the low 64 bits —
            // `trunc(f) mod 2^64 == 0` (upstream `r_uint(huge_int) ==
            // 0` whenever `huge_int` is a multiple of 2^64).
            0
        } else {
            // `wrapping_shl` here matches `(mantissa << shift) mod
            // 2^64` for `shift < 64`.
            mantissa.wrapping_shl(shift)
        }
    } else {
        // 0 <= exp < 52 — shift right to drop the fractional bits.
        mantissa >> (52 - exp) as u32
    };
    if sign == 0 {
        unsigned_trunc
    } else {
        // Negative: `r_uint(-n) == (2^64 - n) mod 2^64` =
        // `n.wrapping_neg()`.
        unsigned_trunc.wrapping_neg()
    }
}

/// RPython `op_cast_float_to_longlong` (`opimpl.py`).
///
/// Upstream:
/// ```python
/// def op_cast_float_to_longlong(f):
///     assert type(f) is float
///     r = float(0x100000000)
///     small = f / r
///     high = int(small)
///     truncated = int((small - high) * r)
///     return r_longlong_result(high) * 0x100000000 + truncated
/// ```
///
/// The high/truncated split is upstream's workaround for hosts
/// without arbitrary-precision integer truncation (`int(f)` for
/// very large `f`): split `f` into high-32 and low-32 components
/// computed entirely inside `i64` arithmetic, then reassemble via
/// `r_longlong_result(high) * 2^32 + truncated`.
///
/// Pyre's port mirrors the algorithm verbatim — `float as i64` is
/// safe inside `[-2^63, 2^63)`, and the high/truncated decomposition
/// keeps both intermediate values inside `i64` for every `f` whose
/// final result fits in `i64`. NaN / inf reject; finite floats
/// outside `[-2^63, 2^63)` saturate via `as i64` and the wrap
/// matches upstream's overflow behavior on the same machine.
pub fn op_cast_float_to_longlong(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => {
            let f = f64::from_bits(*bits);
            if !f.is_finite() {
                return None;
            }
            // Upstream `:438-441` decomposes f / 2^32 into high/low
            // halves so `int()` never sees a value outside Python 2's
            // CPython int range. Pyre operates on f64 directly via
            // [`float_to_signed64_intmask`], which gives the same
            // numerical result without the high/low intermediate —
            // both paths reduce to `int(f) mod 2^64` reinterpreted as
            // a signed 64-bit. Boundary cases verified against
            // CPython in the unit tests below.
            Some(ConstValue::Int(float_to_signed64_intmask(f)))
        }
        _ => None,
    }
}

/// RPython `op_cast_float_to_ulonglong` (`opimpl.py`).
pub fn op_cast_float_to_ulonglong(args: &[ConstValue]) -> Option<ConstValue> {
    op_cast_float_to_uint(args)
}

/// RPython `op_convert_float_bytes_to_longlong` (`opimpl.py`)
/// — `float2longlong(a)`, the reinterpret-cast from f64 bit pattern
/// to i64.
pub fn op_convert_float_bytes_to_longlong(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Float(bits)] => Some(ConstValue::Int(*bits as i64)),
        _ => None,
    }
}

/// RPython `op_convert_longlong_bytes_to_float` (`opimpl.py`)
/// — `longlong2float(a)`, the reinterpret-cast from i64 bit pattern
/// to f64.
pub fn op_convert_longlong_bytes_to_float(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Int(n)] => Some(ConstValue::Float(*n as u64)),
        _ => None,
    }
}

// char_* / unichar_* comparisons

/// RPython `char_lt`/`char_le`/`char_eq`/`char_ne`/`char_gt`/`char_ge`
/// derived via `get_primitive_op_src` (`opimpl.py`). The
/// length-1 carrier is a 1-byte [`ConstValue::ByteStr`].
fn char_pair(args: &[ConstValue]) -> Option<(u8, u8)> {
    match args {
        [ConstValue::ByteStr(a), ConstValue::ByteStr(b)] if a.len() == 1 && b.len() == 1 => {
            Some((a[0], b[0]))
        }
        _ => None,
    }
}

pub fn op_char_lt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a < b))
}

pub fn op_char_le(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a <= b))
}

pub fn op_char_eq(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a == b))
}

pub fn op_char_ne(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a != b))
}

pub fn op_char_gt(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a > b))
}

pub fn op_char_ge(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = char_pair(args)?;
    Some(ConstValue::Bool(a >= b))
}

/// RPython `op_unichar_eq`/`op_unichar_ne` (`opimpl.py`).
fn unichar_pair(args: &[ConstValue]) -> Option<(char, char)> {
    match args {
        [ConstValue::UniStr(a), ConstValue::UniStr(b)] => {
            let mut ai = a.chars();
            let mut bi = b.chars();
            let (a0, b0) = (ai.next()?, bi.next()?);
            if ai.next().is_some() || bi.next().is_some() {
                return None;
            }
            Some((a0, b0))
        }
        _ => None,
    }
}

pub fn op_unichar_eq(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = unichar_pair(args)?;
    Some(ConstValue::Bool(a == b))
}

pub fn op_unichar_ne(args: &[ConstValue]) -> Option<ConstValue> {
    let (a, b) = unichar_pair(args)?;
    Some(ConstValue::Bool(a != b))
}

// likely / unlikely

/// RPython `op_likely` / `op_unlikely` (`opimpl.py`) —
/// identity on `bool`. The annotation is a JIT hint; constant folding
/// just unwraps the value.
pub fn op_likely(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::Bool(*b)),
        _ => None,
    }
}

pub fn op_unlikely(args: &[ConstValue]) -> Option<ConstValue> {
    match args {
        [ConstValue::Bool(b)] => Some(ConstValue::Bool(*b)),
        _ => None,
    }
}

// registry

// TODO: opimpl missing-ops audit.
//
// Upstream `opimpl.py` defines per-LLOp fold callables; the local
// registry below is the subset whose carriers all live in
// `flowspace::ConstValue`. Convergence path for each missing op
// reduces to "land the carrier, then lift the upstream body".
// Categorized blockers (each entry cites `opimpl.py:<line>`):
//
// 1. lltype struct/array/interior carrier (5 ops):
//      op_getsubstruct          :138-143
//      op_getarraysubstruct     :145-150
//      op_getarraysize          :179-181
//      op_getfield              :583-588
//      op_getarrayitem          :590-595
//    Convergence: `lltype.rs::Ptr` + immutable-field bookkeeping
//    (`rpython/rtyper/lltypesystem/lltype.py:_immutable_field`) and
//    a `ConstValue::LLPtr`-aware deref.
//
// 2. lltype interior-pointer carrier (4 ops):
//      op_getinteriorarraysize  :152-160
//      op_getinteriorfield      :162-177
//      op_direct_fieldptr       :183-186
//      op_direct_arrayitems     :188-190
//    Convergence: `lltype._interior_ptr` analogue.
//
// 3. raw pointer arithmetic (1 op):
//      op_direct_ptradd         :192-202
//    Convergence: typed raw-pointer carrier with bounds metadata.
//
// 4. llmemory address carrier — all landed (adr_lt/le/eq/ne/gt/ge/
//    delta/add/sub + cast_int_to_adr + cast_adr_to_ptr). `cast_adr_to_ptr`
//    (`opimpl.py`, `need_result_type`) folds through the
//    RESTYPE threaded by `constfold.rs::eval_llop`.
//
// 5. lltype.cast_primitive / cast_pointer with TYPE arg (2 ops):
//      op_cast_primitive        :383-386
//      op_cast_pointer          :477-480
//    Both are `need_result_type`; `constfold.rs::eval_llop` already
//    threads RESTYPE (see group 4). Convergence: add the RESTYPE-taking
//    `op_cast_primitive` / `op_cast_pointer` bodies + their eval_llop
//    arms (cast_pointer needs the `LLPtr` carrier walk, cast_primitive
//    the primitive-reinterpret over RESTYPE).
//
// 6. r_longlonglong / r_ulonglonglong 128-bit integer carrier
//    (6 ops):
//      op_lllong_floordiv       :314-320
//      op_lllong_mod            :322-328
//      op_lllong_lshift         :350-353
//      op_lllong_rshift         :355-358
//      op_ulllong_lshift        :370-373
//      op_ulllong_rshift        :375-378
//    Convergence: `ConstValue::Int128(i128)` / `UInt128(u128)`
//    variants. Aliasing the existing llong/ullong handlers is
//    rejected — `ConstValue::Int(i64)` cannot represent the full
//    128-bit range, so an alias would silently fold values that
//    upstream's `r_longlonglong_arg` assertion catches.
//
// 7. llgroup combined-symbolic / vtable carrier (7 ops):
//      op_get_group_member          :662-667
//      op_get_next_group_member     :669-674
//      op_is_group_member_nonzero   :676-681
//      op_extract_ushort            :684-687
//      op_combine_ushort            :689-691
//      op_gc_gettypeptr_group       :693-704
//      op_get_member_index          :706-707
//    Convergence: `llgroup::CombinedSymbolic` carrier port (used by
//    GC vtable layouts, currently absent from `ConstValue`).
//
// 8. Intentionally unfoldable (side-effecting): the registry omits
//    these by design because constfold rejects ops with side effects
//    (`constfold.py:34-46` "exception → no fold" path). Listed for
//    completeness so a reader does not mistake omission for a port
//    gap. 28 ops total:
//      op_gc_writebarrier_before_copy/move    :555-581
//      op_debug_flush_log/print/start/stop    :605-616
//      op_debug_offset/flush                  :617-621
//      op_have_debug_prints/_for              :623-627
//      op_debug_nonnull_pointer               :629-630
//      op_debug_fatalerror                    :728-733
//      op_gc_stack_bottom                     :632-633
//      op_jit_force_virtualizable/virtual     :635-642
//      op_jit_is_virtual                      :641-642
//      op_jit_force_quasi_immutable           :644-645
//      op_jit_record_exact_class/value        :647-651
//      op_jit_ffi_save_result                 :653-654
//      op_jit_enter/leave_portal_frame        :656-660
//      op_gc_writebarrier                     :709-710
//      op_gc_bit                              :712-715
//      op_shrink_array                        :717-718
//      op_ll_read_timestamp                   :720-722
//      op_ll_get_timestamp_unit               :724-726
//      op_raw_store/raw_load                  :735-756
//      op_gc_store/gc_load_indexed            :742-767
//      op_gc_store_indexed                    :768-777
//      op_gc_ignore_finalizer                 :787-788
//      op_gc_move_out_of_nursery              :790-791
//      op_gc_increase_root_stack_depth        :793-794
//      op_revdb_do_next_call                  :796-797

/// Mirror of upstream's `getattr(opimpls, 'op_' + opname, None)` — the
/// per-op fold callable, or `None` if no implementation exists for
/// this opname (e.g. carriers not yet ported, or ops with side
/// effects).
pub fn get_op_impl(opname: &str) -> Option<FoldFn> {
    static REGISTRY: OnceLock<HashMap<&'static str, FoldFn>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| {
        let mut m: HashMap<&'static str, FoldFn> = HashMap::new();
        m.insert("bool_not", op_bool_not);
        m.insert("same_as", op_same_as);

        m.insert("int_is_true", op_int_is_true);
        m.insert("int_neg", op_int_neg);
        m.insert("int_abs", op_int_abs);
        m.insert("int_invert", op_int_invert);
        m.insert("int_add", op_int_add);
        m.insert("int_sub", op_int_sub);
        m.insert("int_mul", op_int_mul);
        m.insert("int_floordiv", op_int_floordiv);
        m.insert("int_mod", op_int_mod);
        m.insert("int_lshift", op_int_lshift);
        m.insert("int_rshift", op_int_rshift);
        m.insert("int_and", op_int_and);
        m.insert("int_or", op_int_or);
        m.insert("int_xor", op_int_xor);
        m.insert("int_lt", op_int_lt);
        m.insert("int_le", op_int_le);
        m.insert("int_eq", op_int_eq);
        m.insert("int_ne", op_int_ne);
        m.insert("int_gt", op_int_gt);
        m.insert("int_ge", op_int_ge);
        m.insert("int_between", op_int_between);
        m.insert("int_force_ge_zero", op_int_force_ge_zero);

        m.insert("llong_is_true", op_llong_is_true);
        m.insert("llong_neg", op_llong_neg);
        m.insert("llong_abs", op_llong_abs);
        m.insert("llong_invert", op_llong_invert);
        m.insert("llong_add", op_llong_add);
        m.insert("llong_sub", op_llong_sub);
        m.insert("llong_mul", op_llong_mul);
        m.insert("llong_floordiv", op_llong_floordiv);
        m.insert("llong_mod", op_llong_mod);
        m.insert("llong_lshift", op_llong_lshift);
        m.insert("llong_rshift", op_llong_rshift);
        m.insert("llong_and", op_llong_and);
        m.insert("llong_or", op_llong_or);
        m.insert("llong_xor", op_llong_xor);
        m.insert("llong_lt", op_llong_lt);
        m.insert("llong_le", op_llong_le);
        m.insert("llong_eq", op_llong_eq);
        m.insert("llong_ne", op_llong_ne);
        m.insert("llong_gt", op_llong_gt);
        m.insert("llong_ge", op_llong_ge);

        m.insert("uint_is_true", op_uint_is_true);
        m.insert("uint_invert", op_uint_invert);
        m.insert("uint_add", op_uint_add);
        m.insert("uint_sub", op_uint_sub);
        m.insert("uint_mul", op_uint_mul);
        m.insert("uint_mul_high", op_uint_mul_high);
        m.insert("uint_floordiv", op_uint_floordiv);
        m.insert("uint_mod", op_uint_mod);
        m.insert("uint_lshift", op_uint_lshift);
        m.insert("uint_rshift", op_uint_rshift);
        m.insert("uint_and", op_uint_and);
        m.insert("uint_or", op_uint_or);
        m.insert("uint_xor", op_uint_xor);
        m.insert("uint_lt", op_uint_lt);
        m.insert("uint_le", op_uint_le);
        m.insert("uint_eq", op_uint_eq);
        m.insert("uint_ne", op_uint_ne);
        m.insert("uint_gt", op_uint_gt);
        m.insert("uint_ge", op_uint_ge);

        m.insert("ullong_is_true", op_ullong_is_true);
        m.insert("ullong_invert", op_ullong_invert);
        m.insert("ullong_add", op_ullong_add);
        m.insert("ullong_sub", op_ullong_sub);
        m.insert("ullong_mul", op_ullong_mul);
        m.insert("ullong_floordiv", op_ullong_floordiv);
        m.insert("ullong_mod", op_ullong_mod);
        m.insert("ullong_lshift", op_ullong_lshift);
        m.insert("ullong_rshift", op_ullong_rshift);
        m.insert("ullong_and", op_ullong_and);
        m.insert("ullong_or", op_ullong_or);
        m.insert("ullong_xor", op_ullong_xor);
        m.insert("ullong_lt", op_ullong_lt);
        m.insert("ullong_le", op_ullong_le);
        m.insert("ullong_eq", op_ullong_eq);
        m.insert("ullong_ne", op_ullong_ne);
        m.insert("ullong_gt", op_ullong_gt);
        m.insert("ullong_ge", op_ullong_ge);

        m.insert("cast_int_to_uint", op_cast_int_to_uint);
        m.insert("cast_uint_to_int", op_cast_uint_to_int);
        m.insert("cast_int_to_longlong", op_cast_int_to_longlong);
        m.insert("truncate_longlong_to_int", op_truncate_longlong_to_int);
        m.insert("cast_bool_to_uint", op_cast_bool_to_uint);
        m.insert("cast_uint_to_float", op_cast_uint_to_float);
        m.insert("cast_longlong_to_float", op_cast_longlong_to_float);
        m.insert("cast_ulonglong_to_float", op_cast_ulonglong_to_float);
        m.insert("cast_float_to_uint", op_cast_float_to_uint);
        m.insert("cast_float_to_longlong", op_cast_float_to_longlong);
        m.insert("cast_float_to_ulonglong", op_cast_float_to_ulonglong);
        m.insert(
            "convert_float_bytes_to_longlong",
            op_convert_float_bytes_to_longlong,
        );
        m.insert(
            "convert_longlong_bytes_to_float",
            op_convert_longlong_bytes_to_float,
        );

        m.insert("float_is_true", op_float_is_true);
        m.insert("float_neg", op_float_neg);
        m.insert("float_abs", op_float_abs);
        m.insert("float_add", op_float_add);
        m.insert("float_sub", op_float_sub);
        m.insert("float_mul", op_float_mul);
        m.insert("float_truediv", op_float_truediv);
        m.insert("float_lt", op_float_lt);
        m.insert("float_le", op_float_le);
        m.insert("float_eq", op_float_eq);
        m.insert("float_ne", op_float_ne);
        m.insert("float_gt", op_float_gt);
        m.insert("float_ge", op_float_ge);

        m.insert("ptr_eq", op_ptr_eq);
        m.insert("ptr_ne", op_ptr_ne);
        m.insert("ptr_nonzero", op_ptr_nonzero);
        m.insert("ptr_iszero", op_ptr_iszero);

        m.insert("adr_lt", op_adr_lt);
        m.insert("adr_le", op_adr_le);
        m.insert("adr_eq", op_adr_eq);
        m.insert("adr_ne", op_adr_ne);
        m.insert("adr_gt", op_adr_gt);
        m.insert("adr_ge", op_adr_ge);
        m.insert("adr_delta", op_adr_delta);
        m.insert("adr_add", op_adr_add);
        m.insert("adr_sub", op_adr_sub);
        m.insert("cast_int_to_adr", op_cast_int_to_adr);

        m.insert("cast_int_to_float", op_cast_int_to_float);
        m.insert("cast_float_to_int", op_cast_float_to_int);
        m.insert("cast_bool_to_int", op_cast_bool_to_int);
        m.insert("cast_bool_to_float", op_cast_bool_to_float);
        m.insert("cast_int_to_char", op_cast_int_to_char);
        m.insert("cast_char_to_int", op_cast_char_to_int);
        m.insert("cast_int_to_unichar", op_cast_int_to_unichar);
        m.insert("cast_unichar_to_int", op_cast_unichar_to_int);

        m.insert("char_lt", op_char_lt);
        m.insert("char_le", op_char_le);
        m.insert("char_eq", op_char_eq);
        m.insert("char_ne", op_char_ne);
        m.insert("char_gt", op_char_gt);
        m.insert("char_ge", op_char_ge);

        m.insert("unichar_eq", op_unichar_eq);
        m.insert("unichar_ne", op_unichar_ne);

        m.insert("likely", op_likely);
        m.insert("unlikely", op_unlikely);

        m
    });
    registry.get(opname).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i(n: i64) -> ConstValue {
        ConstValue::Int(n)
    }
    fn u(n: u64) -> ConstValue {
        ConstValue::UInt(n)
    }
    fn b(v: bool) -> ConstValue {
        ConstValue::Bool(v)
    }
    fn f(v: f64) -> ConstValue {
        ConstValue::float(v)
    }

    #[test]
    fn top_level_parity_surface_keeps_unfoldable_op_names_unregistered() {
        assert!(ops_returning_a_bool.contains(&"eq"));
        assert!(argtype_by_name.contains(&"lllong"));
        assert_eq!(no_op(&i(3)), i(3));
        assert_eq!(_normalize(&i(4)), i(4));
        assert_eq!(op_gc_writebarrier_before_move(&[]), None);
        assert_eq!(op_debug_print(&[]), None);
        assert_eq!(op_debug_start(&[]), None);
        assert_eq!(op_debug_stop(&[]), None);
        assert_eq!(op_have_debug_prints_for(&[]), None);
        assert_eq!(op_jit_record_exact_value(&[]), None);
        assert_eq!(op_jit_enter_portal_frame(&[]), None);
        assert_eq!(op_jit_leave_portal_frame(&[]), None);
        assert_eq!(op_raw_load(&LowLevelType::Signed, &[]), None);
        assert_eq!(op_gc_load_indexed(&LowLevelType::Signed, &[]), None);

        for opname in [
            "gc_writebarrier_before_move",
            "debug_print",
            "debug_start",
            "debug_stop",
            "have_debug_prints_for",
            "jit_record_exact_value",
            "jit_enter_portal_frame",
            "jit_leave_portal_frame",
            "raw_load",
            "gc_load_indexed",
        ] {
            assert!(
                get_op_impl(opname).is_none(),
                "{opname} must remain outside constfold registry"
            );
        }
    }

    #[test]
    fn int_arithmetic_wraps_around() {
        assert_eq!(op_int_add(&[i(i64::MAX), i(1)]), Some(i(i64::MIN)));
        assert_eq!(op_int_sub(&[i(i64::MIN), i(1)]), Some(i(i64::MAX)));
        assert_eq!(op_int_mul(&[i(i64::MAX), i(2)]), Some(i(-2)));
        assert_eq!(op_int_neg(&[i(i64::MIN)]), Some(i(i64::MIN)));
        assert_eq!(op_int_abs(&[i(i64::MIN)]), Some(i(i64::MIN)));
    }

    #[test]
    fn int_floordiv_truncates_toward_zero() {
        // Upstream `op_int_floordiv` is C-style truncating division
        // (the `+1` adjustment in `opimpl.py` converts Python
        // floor div back to truncation), matching `i64::checked_div`.
        assert_eq!(op_int_floordiv(&[i(-7), i(2)]), Some(i(-3)));
        assert_eq!(op_int_floordiv(&[i(7), i(-2)]), Some(i(-3)));
        assert_eq!(op_int_floordiv(&[i(7), i(2)]), Some(i(3)));
        // INT_MIN / -1 refuses to fold (would overflow Signed).
        assert_eq!(op_int_floordiv(&[i(i64::MIN), i(-1)]), None);
        // Division by zero refuses to fold.
        assert_eq!(op_int_floordiv(&[i(1), i(0)]), None);
    }

    #[test]
    fn int_mod_truncates_toward_zero() {
        // Upstream `op_int_mod` is C-style truncating mod (the
        // `r -= y` adjustment in `opimpl.py` converts Python
        // floor mod back to truncation), matching `i64::wrapping_rem`.
        // Sign of the result tracks the dividend.
        assert_eq!(op_int_mod(&[i(-7), i(2)]), Some(i(-1)));
        assert_eq!(op_int_mod(&[i(7), i(-2)]), Some(i(1)));
        assert_eq!(op_int_mod(&[i(i64::MIN), i(-1)]), Some(i(0)));
        assert_eq!(op_int_mod(&[i(1), i(0)]), None);
    }

    #[test]
    fn int_lshift_handles_shift_count_bounds() {
        assert_eq!(op_int_lshift(&[i(1), i(0)]), Some(i(1)));
        assert_eq!(op_int_lshift(&[i(1), i(63)]), Some(i(i64::MIN)));
        assert_eq!(op_int_lshift(&[i(1), i(64)]), Some(i(0)));
        assert_eq!(op_int_lshift(&[i(1), i(100)]), Some(i(0)));
        assert_eq!(op_int_lshift(&[i(1), i(-1)]), None);
    }

    #[test]
    fn int_rshift_handles_sign_and_overflow() {
        assert_eq!(op_int_rshift(&[i(8), i(1)]), Some(i(4)));
        assert_eq!(op_int_rshift(&[i(-8), i(1)]), Some(i(-4)));
        assert_eq!(op_int_rshift(&[i(-1), i(64)]), Some(i(-1)));
        assert_eq!(op_int_rshift(&[i(7), i(64)]), Some(i(0)));
        assert_eq!(op_int_rshift(&[i(7), i(-1)]), None);
    }

    #[test]
    fn int_between_inclusive_lower_exclusive_upper() {
        assert_eq!(op_int_between(&[i(0), i(0), i(1)]), Some(b(true)));
        assert_eq!(op_int_between(&[i(0), i(1), i(1)]), Some(b(false)));
        assert_eq!(op_int_between(&[i(0), i(-1), i(1)]), Some(b(false)));
    }

    #[test]
    fn int_force_ge_zero_clamps_negatives() {
        assert_eq!(op_int_force_ge_zero(&[i(-5)]), Some(i(0)));
        assert_eq!(op_int_force_ge_zero(&[i(0)]), Some(i(0)));
        assert_eq!(op_int_force_ge_zero(&[i(5)]), Some(i(5)));
    }

    #[test]
    fn cast_int_float_round_trip() {
        assert_eq!(op_cast_int_to_float(&[i(7)]), Some(f(7.0)));
        assert_eq!(op_cast_float_to_int(&[f(3.7)]), Some(i(3)));
        assert_eq!(op_cast_float_to_int(&[f(-3.7)]), Some(i(-3)));
        // NaN / inf refuse to fold.
        assert_eq!(op_cast_float_to_int(&[f(f64::NAN)]), None);
        assert_eq!(op_cast_float_to_int(&[f(f64::INFINITY)]), None);
        assert_eq!(op_cast_float_to_int(&[f(f64::NEG_INFINITY)]), None);
    }

    #[test]
    fn cast_float_to_int_wraps_large_finite() {
        // Upstream `intmask(int(2.0**63))` == -2^63 (high bit set).
        assert_eq!(
            op_cast_float_to_int(&[f((1u64 << 63) as f64)]),
            Some(i(i64::MIN)),
        );
        // 2^63 + 2^11 = a representable float just above i64::MAX —
        // `intmask` wraps to the corresponding negative.
        let above_max = (1u64 << 63) as f64 + (1u64 << 11) as f64;
        let expected = (((1u128 << 63) + (1u128 << 11)) as i64)
            .wrapping_neg()
            .wrapping_neg();
        assert_eq!(
            op_cast_float_to_int(&[f(above_max)]),
            Some(i(expected.wrapping_neg().wrapping_neg())),
        );
        // 2^64 → mod 2^64 == 0.
        assert_eq!(op_cast_float_to_int(&[f((1u128 << 64) as f64)]), Some(i(0)),);
        // -2^63 fits exactly.
        assert_eq!(
            op_cast_float_to_int(&[f(i64::MIN as f64)]),
            Some(i(i64::MIN))
        );
        // Subnormals truncate to zero.
        assert_eq!(op_cast_float_to_int(&[f(0.5)]), Some(i(0)));
        assert_eq!(op_cast_float_to_int(&[f(-0.5)]), Some(i(0)));
    }

    #[test]
    fn cast_float_to_longlong_matches_cast_float_to_int_for_finite() {
        // Both ops reduce to `int(f) mod 2^64` on 64-bit hosts —
        // upstream's high/low decomposition is a CPython int-overflow
        // workaround, not a different algorithm.
        for &v in &[
            3.7_f64,
            -3.7,
            0.0,
            (1u64 << 63) as f64,
            (1u128 << 64) as f64,
        ] {
            assert_eq!(
                op_cast_float_to_longlong(&[f(v)]),
                op_cast_float_to_int(&[f(v)]),
                "mismatch at f = {v}",
            );
        }
    }

    #[test]
    fn cast_bool_round_trip() {
        assert_eq!(op_cast_bool_to_int(&[b(true)]), Some(i(1)));
        assert_eq!(op_cast_bool_to_int(&[b(false)]), Some(i(0)));
        assert_eq!(op_cast_bool_to_float(&[b(true)]), Some(f(1.0)));
        assert_eq!(op_cast_bool_to_float(&[b(false)]), Some(f(0.0)));
    }

    #[test]
    fn cast_char_round_trip() {
        assert_eq!(
            op_cast_int_to_char(&[i(b'A' as i64)]),
            Some(ConstValue::ByteStr(vec![b'A']))
        );
        assert_eq!(op_cast_int_to_char(&[i(-1)]), None);
        assert_eq!(op_cast_int_to_char(&[i(256)]), None);
        assert_eq!(
            op_cast_char_to_int(&[ConstValue::ByteStr(vec![b'A'])]),
            Some(i(65))
        );
        // Non-singleton bytes refuse to fold (upstream `assert
        // type(b) is str and len(b) == 1`).
        assert_eq!(op_cast_char_to_int(&[ConstValue::ByteStr(vec![])]), None);
        assert_eq!(
            op_cast_char_to_int(&[ConstValue::ByteStr(vec![b'A', b'B'])]),
            None
        );
    }

    #[test]
    fn cast_unichar_round_trip() {
        assert_eq!(
            op_cast_int_to_unichar(&[i('가' as i64)]),
            Some(ConstValue::UniStr("가".to_string()))
        );
        // Surrogate code points are invalid → no fold.
        assert_eq!(op_cast_int_to_unichar(&[i(0xD800)]), None);
        assert_eq!(op_cast_int_to_unichar(&[i(-1)]), None);
        assert_eq!(
            op_cast_unichar_to_int(&[ConstValue::UniStr("가".to_string())]),
            Some(i('가' as i64))
        );
        assert_eq!(
            op_cast_unichar_to_int(&[ConstValue::UniStr("ab".to_string())]),
            None
        );
    }

    #[test]
    fn char_comparisons_use_byte_value() {
        let a = ConstValue::ByteStr(vec![b'a']);
        let z = ConstValue::ByteStr(vec![b'z']);
        assert_eq!(op_char_lt(&[a.clone(), z.clone()]), Some(b(true)));
        assert_eq!(op_char_le(&[a.clone(), a.clone()]), Some(b(true)));
        assert_eq!(op_char_eq(&[a.clone(), a.clone()]), Some(b(true)));
        assert_eq!(op_char_ne(&[a.clone(), z.clone()]), Some(b(true)));
        assert_eq!(op_char_gt(&[z.clone(), a.clone()]), Some(b(true)));
        assert_eq!(op_char_ge(&[a.clone(), a.clone()]), Some(b(true)));
        // Non-1-byte byte-strings refuse to fold.
        assert_eq!(op_char_eq(&[ConstValue::ByteStr(vec![]), a]), None);
    }

    #[test]
    fn unichar_comparisons_use_codepoint() {
        let a = ConstValue::UniStr("가".to_string());
        let b_ = ConstValue::UniStr("나".to_string());
        assert_eq!(op_unichar_eq(&[a.clone(), a.clone()]), Some(b(true)));
        assert_eq!(op_unichar_ne(&[a.clone(), b_]), Some(b(true)));
        // Multi-char unistrings refuse to fold.
        assert_eq!(op_unichar_eq(&[ConstValue::UniStr("ab".into()), a]), None);
    }

    #[test]
    fn likely_unlikely_are_identity_on_bool() {
        assert_eq!(op_likely(&[b(true)]), Some(b(true)));
        assert_eq!(op_unlikely(&[b(false)]), Some(b(false)));
    }

    #[test]
    fn float_arithmetic_and_zero_division() {
        assert_eq!(op_float_add(&[f(1.5), f(2.5)]), Some(f(4.0)));
        assert_eq!(op_float_sub(&[f(1.5), f(2.5)]), Some(f(-1.0)));
        assert_eq!(op_float_mul(&[f(1.5), f(2.0)]), Some(f(3.0)));
        assert_eq!(op_float_truediv(&[f(3.0), f(2.0)]), Some(f(1.5)));
        // Zero division refuses to fold (Python ZeroDivisionError).
        assert_eq!(op_float_truediv(&[f(1.0), f(0.0)]), None);
        assert_eq!(op_float_truediv(&[f(1.0), f(-0.0)]), None);
        assert_eq!(op_float_neg(&[f(2.0)]), Some(f(-2.0)));
        assert_eq!(op_float_abs(&[f(-2.0)]), Some(f(2.0)));
        assert_eq!(op_float_is_true(&[f(0.0)]), Some(b(false)));
        assert_eq!(op_float_is_true(&[f(0.5)]), Some(b(true)));
    }

    #[test]
    fn float_comparisons_follow_ieee_754() {
        assert_eq!(op_float_lt(&[f(1.0), f(2.0)]), Some(b(true)));
        assert_eq!(op_float_le(&[f(2.0), f(2.0)]), Some(b(true)));
        assert_eq!(op_float_eq(&[f(2.0), f(2.0)]), Some(b(true)));
        assert_eq!(op_float_ne(&[f(2.0), f(3.0)]), Some(b(true)));
        assert_eq!(op_float_gt(&[f(3.0), f(2.0)]), Some(b(true)));
        assert_eq!(op_float_ge(&[f(2.0), f(2.0)]), Some(b(true)));
        // NaN compares unequal even to itself.
        assert_eq!(op_float_eq(&[f(f64::NAN), f(f64::NAN)]), Some(b(false)));
        assert_eq!(op_float_lt(&[f(f64::NAN), f(0.0)]), Some(b(false)));
    }

    #[test]
    fn llong_aliases_int_on_64bit_carrier() {
        // `r_longlong is r_int` on 64-bit hosts (`opimpl.py`),
        // so every llong_* op produces the same result as int_* on
        // identical inputs.
        assert_eq!(op_llong_add(&[i(i64::MAX), i(1)]), Some(i(i64::MIN)));
        assert_eq!(op_llong_floordiv(&[i(-7), i(2)]), Some(i(-3)));
        assert_eq!(op_llong_mod(&[i(7), i(-2)]), Some(i(1)));
        assert_eq!(op_llong_lshift(&[i(1), i(63)]), Some(i(i64::MIN)));
        assert_eq!(op_llong_rshift(&[i(-8), i(1)]), Some(i(-4)));
    }

    #[test]
    fn uint_uses_unsigned_division_and_comparison() {
        let umax = u(u64::MAX);
        // Unsigned: u64::MAX > 1 (signed: -1 < 1).
        assert_eq!(op_uint_lt(&[umax.clone(), u(1)]), Some(b(false)));
        assert_eq!(op_uint_gt(&[umax.clone(), u(1)]), Some(b(true)));
        // Unsigned: u64::MAX // 2 == 0x7FFFFFFFFFFFFFFF.
        assert_eq!(
            op_uint_floordiv(&[umax.clone(), u(2)]),
            Some(u(0x7FFFFFFFFFFFFFFF))
        );
        // Logical right shift fills with zero.
        assert_eq!(
            op_uint_rshift(&[umax.clone(), i(1)]),
            Some(u(0x7FFFFFFFFFFFFFFF))
        );
        // Modular arithmetic agrees with signed for add/sub/mul.
        assert_eq!(op_uint_add(&[u(u64::MAX), u(1)]), Some(u(0)));
        // Division by zero refuses to fold.
        assert_eq!(op_uint_floordiv(&[u(1), u(0)]), None);
        // Negative shift refuses to fold.
        assert_eq!(op_uint_lshift(&[u(1), i(-1)]), None);
        // `rarithmetic.is_valid_int` accepts `r_uint` in signed range.
        assert_eq!(op_uint_lshift(&[u(1), u(1)]), Some(u(2)));
        assert_eq!(op_uint_rshift(&[u(4), u(1)]), Some(u(2)));
        assert_eq!(op_cast_int_to_uint(&[u(1)]), Some(u(1)));
        // `r_uint` above `maxint` fails `is_valid_int`.
        assert_eq!(op_uint_lshift(&[u(1), u(i64::MAX as u64 + 1)]), None);
        assert_eq!(op_cast_int_to_uint(&[u(i64::MAX as u64 + 1)]), None);
        // `assert type(b) is r_uint`.
        assert_eq!(op_cast_uint_to_int(&[i(1)]), None);
    }

    #[test]
    fn ullong_matches_uint_on_64bit_carrier() {
        // `r_ulonglong` ≡ `r_uint` ≡ u64 on the 64-bit carrier.
        assert_eq!(op_ullong_lt(&[u(u64::MAX), u(1)]), Some(b(false)));
        assert_eq!(
            op_ullong_floordiv(&[u(u64::MAX), u(2)]),
            Some(u(0x7FFFFFFFFFFFFFFF))
        );
        assert_eq!(
            op_ullong_rshift(&[u(u64::MAX), i(1)]),
            Some(u(0x7FFFFFFFFFFFFFFF))
        );
    }

    #[test]
    fn wide_int_casts_are_bit_pattern_identity_on_64bit_carrier() {
        let v = i(0x1234_5678);
        assert_eq!(
            op_cast_int_to_uint(std::slice::from_ref(&v)),
            Some(u(0x1234_5678))
        );
        assert_eq!(op_cast_uint_to_int(&[u(0x1234_5678)]), Some(v.clone()));
        assert_eq!(
            op_cast_int_to_longlong(std::slice::from_ref(&v)),
            Some(v.clone())
        );
        assert_eq!(
            op_truncate_longlong_to_int(std::slice::from_ref(&v)),
            Some(v.clone())
        );
        assert_eq!(op_cast_bool_to_uint(&[b(true)]), Some(u(1)));
    }

    #[test]
    fn cast_float_uint_round_trip_within_u64_range() {
        assert_eq!(op_cast_uint_to_float(&[u(1024)]), Some(f(1024.0)));
        assert_eq!(
            op_cast_uint_to_float(&[u(u64::MAX)]),
            Some(f(u64::MAX as f64))
        );
        assert_eq!(op_cast_float_to_uint(&[f(1024.0)]), Some(u(1024)));
        // Negative finite floats fold via 2's-complement wrap —
        // upstream `r_uint(long(-1.0))` is `0xFFFFFFFFFFFFFFFF`.
        // Truncates toward zero (`-1.5` → `-1`, not `-2`).
        assert_eq!(op_cast_float_to_uint(&[f(-1.0)]), Some(u(u64::MAX)));
        assert_eq!(op_cast_float_to_uint(&[f(-1.5)]), Some(u(u64::MAX)));
        assert_eq!(op_cast_float_to_uint(&[f(-2.0)]), Some(u((-2i64) as u64)));
        // Wrap, not saturate, near the i64::MAX boundary. `2^63`
        // should produce the bit pattern `0x8000_0000_0000_0000`.
        assert_eq!(
            op_cast_float_to_uint(&[f(9223372036854775808.0)]), // 2^63
            Some(u(1u64 << 63)),
        );
        assert_eq!(
            op_cast_float_to_ulonglong(&[f(9223372036854775808.0)]),
            Some(u(1u64 << 63)),
        );
        // 2^65 → upstream `r_uint(2^65)` is `2^65 mod 2^64 == 0`. The
        // bit-shift implementation walks the IEEE-754 mantissa /
        // exponent and folds correctly across the full finite range.
        assert_eq!(
            op_cast_float_to_uint(&[f(36893488147419103232.0)]), // 2^65
            Some(u(0)),
        );
        // -2^65 wraps to `(-2^65) mod 2^64 == 0` for the same reason.
        assert_eq!(
            op_cast_float_to_uint(&[f(-36893488147419103232.0)]),
            Some(u(0)),
        );
        // 2^63 + 2^32 (still inside finite-float exact range) — wrap
        // produces 0x8000_0000_0000_0000 + 2^32 in u64.
        assert_eq!(
            op_cast_float_to_uint(&[f(9223372041149743104.0)]),
            Some(u((1u64 << 63).wrapping_add(0x1_0000_0000))),
        );
        // NaN / inf refuse to fold.
        assert_eq!(op_cast_float_to_uint(&[f(f64::NAN)]), None);
        assert_eq!(op_cast_float_to_uint(&[f(f64::INFINITY)]), None);
        assert_eq!(op_cast_float_to_uint(&[f(f64::NEG_INFINITY)]), None);
    }

    #[test]
    fn convert_float_longlong_bytes_round_trip() {
        let value = 1.5_f64;
        let bits = value.to_bits() as i64;
        assert_eq!(
            op_convert_float_bytes_to_longlong(&[f(value)]),
            Some(i(bits))
        );
        assert_eq!(
            op_convert_longlong_bytes_to_float(&[i(bits)]),
            Some(f(value))
        );
    }

    #[test]
    fn registry_lookup_returns_callable_for_known_op() {
        let f = get_op_impl("int_add").expect("int_add must be in registry");
        assert_eq!(f(&[i(1), i(2)]), Some(i(3)));
        assert!(get_op_impl("definitely_not_an_op").is_none());
    }

    #[test]
    fn adr_eq_ne_fold_over_address_carrier() {
        use crate::translator::rtyper::lltypesystem::lltype::_address;
        let null = || ConstValue::LLAddress(_address::Null);
        let f_eq = get_op_impl("adr_eq").expect("adr_eq must be registered");
        let f_ne = get_op_impl("adr_ne").expect("adr_ne must be registered");
        // NULL == NULL folds to true; != folds to false.
        assert_eq!(f_eq(&[null(), null()]), Some(ConstValue::Bool(true)));
        assert_eq!(f_ne(&[null(), null()]), Some(ConstValue::Bool(false)));
        // `checkadr`: non-address operands refuse to fold.
        assert_eq!(f_eq(&[i(1), i(1)]), None);

        // Ordering ops are registered; NULL < NULL folds to false, and a
        // non-orderable pair (handled in lltype tests) declines to fold here.
        for op in ["adr_lt", "adr_le", "adr_gt", "adr_ge"] {
            assert!(get_op_impl(op).is_some(), "{op} must be registered");
        }
        let f_lt = get_op_impl("adr_lt").unwrap();
        assert_eq!(f_lt(&[null(), null()]), Some(ConstValue::Bool(false)));

        // adr_delta of two equal addresses folds to Signed 0; non-address
        // operands refuse to fold (`checkadr`).
        let f_delta = get_op_impl("adr_delta").expect("adr_delta must be registered");
        assert_eq!(f_delta(&[null(), null()]), Some(ConstValue::Int(0)));
        assert_eq!(f_delta(&[i(1), i(1)]), None);
    }

    #[test]
    fn adr_add_folds_through_container_array_navigation() {
        use crate::translator::rtyper::lltypesystem::llmemory::{
            AddressOffset, ArrayItemsOffset, ItemOffset,
        };
        use crate::translator::rtyper::lltypesystem::lltype::{
            _address, _ptr_obj, Array, LowLevelType, MallocFlavor, ParentIndex, Struct, malloc,
            parentlink,
        };

        // `GcArray(Struct('item', ('x', Signed)))` of length 3.
        let item = Struct::new("item", vec![("x".into(), LowLevelType::Signed)]);
        let item_ty = LowLevelType::Struct(Box::new(item));
        let array_ty = LowLevelType::Array(Box::new(Array::gc(item_ty.clone())));
        let arrayptr = malloc(array_ty.clone(), Some(3), MallocFlavor::Gc, true).unwrap();

        let f_add = get_op_impl("adr_add").expect("adr_add must be registered");

        // `arrayadr + ArrayItemsOffset(ARRAY)` → pointer to item 0.
        let arrayadr = ConstValue::LLAddress(_address::Fake(Box::new(arrayptr.clone())));
        let items_off =
            ConstValue::AddressOffset(AddressOffset::ArrayItemsOffset(ArrayItemsOffset(array_ty)));
        let Some(ConstValue::LLAddress(_address::Fake(item0ptr))) = f_add(&[arrayadr, items_off])
        else {
            panic!("ArrayItemsOffset fold must yield a fake address");
        };

        // `firstitemadr + ItemOffset(ITEM, 1)` → pointer to item 1.
        let item0adr = ConstValue::LLAddress(_address::Fake(item0ptr));
        let item_off = ConstValue::AddressOffset(AddressOffset::ItemOffset(ItemOffset {
            TYPE: item_ty,
            repeat: 1,
        }));
        let Some(ConstValue::LLAddress(_address::Fake(item1ptr))) = f_add(&[item0adr, item_off])
        else {
            panic!("ItemOffset fold must yield a fake address");
        };

        // The navigated pointer is the array's element at index 1: its
        // parentlink points back at the array container with item index 1.
        let (parent, index) = parentlink(&item1ptr._obj().unwrap());
        assert!(matches!(parent, Some(_ptr_obj::Array(_))));
        match index {
            Some(ParentIndex::Item(1)) => {}
            Some(ParentIndex::Field(ref name)) if name == "item1" => {}
            other => panic!("expected item index 1, got {other:?}"),
        }

        // A NULL base is `NullAddressError` and declines to fold.
        assert_eq!(
            op_adr_add(&[
                ConstValue::LLAddress(_address::Null),
                ConstValue::AddressOffset(AddressOffset::ArrayItemsOffset(ArrayItemsOffset(
                    LowLevelType::Signed,
                ))),
            ]),
            None
        );
    }

    #[test]
    fn adr_add_sub_zero_offset_returns_address_unchanged() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            _address, Array, LowLevelType, MallocFlavor, malloc,
        };

        // `fakeaddress.__add__/__sub__`: a plain `0` offset returns the address
        // unchanged (llmemory.py, 486-487). The `Int(0)` carrier is a
        // valid `Signed` offset alongside `AddressOffset`.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(1), MallocFlavor::Gc, true).unwrap();
        let addr = ConstValue::LLAddress(_address::Fake(Box::new(arrayptr)));

        assert_eq!(
            op_adr_add(&[addr.clone(), ConstValue::Int(0)]),
            Some(addr.clone())
        );
        assert_eq!(
            op_adr_sub(&[addr.clone(), ConstValue::Int(0)]),
            Some(addr.clone())
        );
        // A non-zero plain int is `NotImplemented` upstream → declines to fold.
        assert_eq!(op_adr_add(&[addr.clone(), ConstValue::Int(8)]), None);
        assert_eq!(op_adr_sub(&[addr, ConstValue::Int(8)]), None);
    }

    #[test]
    fn adr_add_folds_through_primitive_array_navigation() {
        use crate::translator::rtyper::lltypesystem::llmemory::{
            AddressOffset, ArrayItemsOffset, ItemOffset,
        };
        use crate::translator::rtyper::lltypesystem::lltype::{
            _address, _ptr_obj, Array, LowLevelType, LowLevelValue, MallocFlavor, malloc,
        };

        // `GcArray(Signed)` of length 3, item 1 set to 42.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty.clone(), Some(3), MallocFlavor::Gc, true).unwrap();
        let _ptr_obj::Array(arr) = arrayptr._obj().unwrap() else {
            panic!("array ptr must hold an Array container");
        };
        assert!(arr.setitem(1, LowLevelValue::Signed(42)));

        let f_add = get_op_impl("adr_add").expect("adr_add must be registered");

        // `arrayadr + ArrayItemsOffset(ARRAY)` → `direct_arrayitems` interior
        // pointer to item 0 (a `_subarray`).
        let arrayadr = ConstValue::LLAddress(_address::Fake(Box::new(arrayptr)));
        let items_off =
            ConstValue::AddressOffset(AddressOffset::ArrayItemsOffset(ArrayItemsOffset(array_ty)));
        let Some(ConstValue::LLAddress(_address::Fake(item0ptr))) = f_add(&[arrayadr, items_off])
        else {
            panic!("ArrayItemsOffset fold must yield a fake address");
        };

        // `firstitemadr + ItemOffset(Signed, 1)` → `direct_ptradd` to item 1.
        let item0adr = ConstValue::LLAddress(_address::Fake(item0ptr));
        let item_off = ConstValue::AddressOffset(AddressOffset::ItemOffset(ItemOffset {
            TYPE: LowLevelType::Signed,
            repeat: 1,
        }));
        let Some(ConstValue::LLAddress(_address::Fake(item1ptr))) = f_add(&[item0adr, item_off])
        else {
            panic!("ItemOffset fold must yield a fake address");
        };

        // The item-1 subarray reads the parent array's element 1 (= 42).
        let _ptr_obj::Subarray(sub) = item1ptr._obj().unwrap() else {
            panic!("a primitive-array interior pointer holds a _subarray");
        };
        assert_eq!(sub.getitem(0), Some(LowLevelValue::Signed(42)));
    }

    #[test]
    fn cast_int_to_adr_folds_per_cast_int_to_ptr_semantics() {
        use crate::translator::rtyper::lltypesystem::lltype::{_address, _ptr_obj};
        let cast = get_op_impl("cast_int_to_adr").expect("cast_int_to_adr must be registered");
        // 0 routes through `cast_int_to_ptr` → `nullptr` → the NULL address.
        assert_eq!(cast(&[i(0)]), Some(ConstValue::LLAddress(_address::Null)));
        // An odd integer casts through `cast_int_to_ptr(_NONGCREF, odd)` to a
        // tagged-integer `_ptr`, wrapped as the fakeaddress (`Fake`).
        let Some(ConstValue::LLAddress(_address::Fake(p))) = cast(&[i(0x41)]) else {
            panic!("an odd integer must build a tagged-int fakeaddress");
        };
        assert!(matches!(p._obj0_value(), Ok(Some(_ptr_obj::IntCast(0x41)))));
        // An even non-zero integer raises `ValueError` upstream (its
        // `ll2ctypes` resolution is runtime-only) → declines.
        assert_eq!(cast(&[i(0x40)]), None);
        // A non-integer operand declines.
        assert_eq!(cast(&[f(1.0)]), None);
    }

    #[test]
    fn cast_adr_to_ptr_folds_fake_and_null_addresses() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            _address, Array, LowLevelType, MallocFlavor, malloc,
        };

        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty, Some(3), MallocFlavor::Gc, true).unwrap();
        let restype = LowLevelType::Ptr(Box::new(arrayptr._TYPE.clone()));

        // A fake address over a live ptr casts back to the same-typed ptr.
        let fakeadr = ConstValue::LLAddress(_address::Fake(Box::new(arrayptr.clone())));
        assert_eq!(
            op_cast_adr_to_ptr(&restype, &[fakeadr]),
            Some(ConstValue::LLPtr(Box::new(arrayptr)))
        );

        // A NULL address casts to the null ptr of the expected type.
        let nulladr = ConstValue::LLAddress(_address::Null);
        let Some(ConstValue::LLPtr(p)) = op_cast_adr_to_ptr(&restype, &[nulladr]) else {
            panic!("a NULL address must cast to a null ptr");
        };
        assert!(p._obj0_value().is_ok_and(|o| o.is_none()));

        // A tagged-int fakeaddress (`cast_int_to_adr` of an odd integer)
        // re-casts its `_NONGCREF` opaque pointer through the real cast
        // pipeline. Re-casting to a concrete array type fails — the bare-int
        // container has no `_obj.container` (upstream `InvalidCast`) — so the
        // fold declines.
        let cast_int = get_op_impl("cast_int_to_adr").unwrap();
        let Some(intadr) = cast_int(&[i(0x41)]) else {
            panic!("an odd integer must build a tagged-int fakeaddress");
        };
        assert_eq!(op_cast_adr_to_ptr(&restype, &[intadr]), None);
    }
}
