//! Concrete execution for tracing — RPython `executor.execute()` parity.
//!
//! Each function computes a concrete result from concrete arguments,
//! without recording any IR operations. This mirrors RPython's executor
//! module which dispatches to `BlackholeInterpreter.bhimpl_*` methods
//! for concrete computation during tracing.

use super::state::ConcreteValue;

// ── Integer arithmetic (RPython bhimpl_int_*) ──

pub fn bhimpl_int_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_add(y)),
        _ => ConcreteValue::Null,
    }
}

/// executor.py do_int_add_ovf: ovfcheck(a + b)
/// Returns (result, ovf_flag). On overflow: result=0, ovf_flag=true.
pub fn bhimpl_int_add_ovf(a: ConcreteValue, b: ConcreteValue) -> (ConcreteValue, bool) {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => match x.checked_add(y) {
            Some(z) => (ConcreteValue::Int(z), false),
            None => (ConcreteValue::Int(0), true),
        },
        _ => (ConcreteValue::Null, false),
    }
}

pub fn bhimpl_int_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_sub(y)),
        _ => ConcreteValue::Null,
    }
}

/// executor.py do_int_sub_ovf: ovfcheck(a - b)
/// Returns (result, ovf_flag). On overflow: result=0, ovf_flag=true.
pub fn bhimpl_int_sub_ovf(a: ConcreteValue, b: ConcreteValue) -> (ConcreteValue, bool) {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => match x.checked_sub(y) {
            Some(z) => (ConcreteValue::Int(z), false),
            None => (ConcreteValue::Int(0), true),
        },
        _ => (ConcreteValue::Null, false),
    }
}

pub fn bhimpl_int_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_mul(y)),
        _ => ConcreteValue::Null,
    }
}

/// executor.py do_int_mul_ovf: ovfcheck(a * b)
/// Returns (result, ovf_flag). On overflow: result=0, ovf_flag=true.
pub fn bhimpl_int_mul_ovf(a: ConcreteValue, b: ConcreteValue) -> (ConcreteValue, bool) {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => match x.checked_mul(y) {
            Some(z) => (ConcreteValue::Int(z), false),
            None => (ConcreteValue::Int(0), true),
        },
        _ => (ConcreteValue::Null, false),
    }
}

/// The `int_py_div` residual call's target: Python's floor quotient.
///
/// `rint.py ll_int_py_div`: truncate as C does, then correct by the
/// sign of the residue. `r * y` and `x - r * y` both stay in range because
/// `r` is the truncating quotient, so the only wrap is the `INT_MIN / -1`
/// corner the trace guards out ahead of the call.
pub fn bhimpl_int_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => {
            let r = x.wrapping_div(y);
            let p = r.wrapping_mul(y);
            let u = if y < 0 {
                p.wrapping_sub(x)
            } else {
                x.wrapping_sub(p)
            };
            ConcreteValue::Int(r.wrapping_add(u >> (i64::BITS - 1)))
        }
        _ => ConcreteValue::Null,
    }
}

/// The `int_py_mod` residual call's target: Python's floor remainder.
///
/// `rint.py ll_int_py_mod`: truncate as C does, then add the divisor
/// back exactly when the remainder carries the wrong sign. Adding `y` to a
/// remainder of the opposite sign cannot leave the range, which is why the
/// correction is masked out of `y` rather than computed as `(r + y) % y` —
/// that intermediate overflows for a remainder near the type's limit.
pub fn bhimpl_int_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => {
            let r = x.wrapping_rem(y);
            let u = if y < 0 { r.wrapping_neg() } else { r };
            ConcreteValue::Int(r.wrapping_add(y & (u >> (i64::BITS - 1))))
        }
        _ => ConcreteValue::Null,
    }
}

/// `support.py _ll_2_int_floordiv` — the truncating primitive, as distinct from
/// the floor helper above that the `int_py_div` call reaches.
pub fn _ll_2_int_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => ConcreteValue::Int(x.wrapping_div(y)),
        _ => ConcreteValue::Null,
    }
}

/// `support.py _ll_2_int_mod` — see [`_ll_2_int_floordiv`].
pub fn _ll_2_int_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => ConcreteValue::Int(x.wrapping_rem(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_and(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x & y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_or(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x | y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_xor(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x ^ y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_lshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shl(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_rshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shr(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_neg(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(x.wrapping_neg()),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_invert(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(!x),
        _ => ConcreteValue::Null,
    }
}

// ── Integer comparison (RPython bhimpl_int_lt etc.) ──

pub fn bhimpl_int_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}

// ── Float arithmetic (RPython bhimpl_float_*) ──

pub fn bhimpl_float_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x + y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x - y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x * y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_truediv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x / y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float((x / y).floor()),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x % y),
        _ => ConcreteValue::Null,
    }
}

// ── Float comparison ──

pub fn bhimpl_float_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}

// ── Unified dispatch (RPython executor.execute() parity) ──

/// Dispatch concrete execution by opcode.
/// RPython: executor.execute(cpu, metainterp, opnum, descr, *argboxes)
/// Returns (result, ovf_flag). ovf_flag is true only for Int*Ovf opcodes on overflow.
pub fn execute_opcode(opcode: majit_ir::OpCode, args: &[ConcreteValue]) -> (ConcreteValue, bool) {
    use majit_ir::OpCode;
    match opcode {
        // Integer arithmetic
        OpCode::IntAdd => {
            if args.len() >= 2 {
                (bhimpl_int_add(args[0], args[1]), false)
            } else {
                (ConcreteValue::Null, false)
            }
        }
        // executor.py do_int_add_ovf: ovfcheck(a + b), ovf_flag on overflow
        OpCode::IntAddOvf => {
            if args.len() >= 2 {
                bhimpl_int_add_ovf(args[0], args[1])
            } else {
                (ConcreteValue::Null, false)
            }
        }
        OpCode::IntSub => {
            if args.len() >= 2 {
                (bhimpl_int_sub(args[0], args[1]), false)
            } else {
                (ConcreteValue::Null, false)
            }
        }
        // executor.py do_int_sub_ovf: ovfcheck(a - b), ovf_flag on overflow
        OpCode::IntSubOvf => {
            if args.len() >= 2 {
                bhimpl_int_sub_ovf(args[0], args[1])
            } else {
                (ConcreteValue::Null, false)
            }
        }
        OpCode::IntMul => {
            if args.len() >= 2 {
                (bhimpl_int_mul(args[0], args[1]), false)
            } else {
                (ConcreteValue::Null, false)
            }
        }
        // executor.py do_int_mul_ovf: ovfcheck(a * b), ovf_flag on overflow
        OpCode::IntMulOvf => {
            if args.len() >= 2 {
                bhimpl_int_mul_ovf(args[0], args[1])
            } else {
                (ConcreteValue::Null, false)
            }
        }
        OpCode::IntAnd => {
            let v = if args.len() >= 2 {
                bhimpl_int_and(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntOr => {
            let v = if args.len() >= 2 {
                bhimpl_int_or(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntXor => {
            let v = if args.len() >= 2 {
                bhimpl_int_xor(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntLshift => {
            let v = if args.len() >= 2 {
                bhimpl_int_lshift(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntRshift => {
            let v = if args.len() >= 2 {
                bhimpl_int_rshift(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntNeg => {
            let v = if args.len() >= 1 {
                bhimpl_int_neg(args[0])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntInvert => {
            let v = if args.len() >= 1 {
                bhimpl_int_invert(args[0])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        // Integer comparison
        OpCode::IntLt => {
            let v = if args.len() >= 2 {
                bhimpl_int_lt(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntLe => {
            let v = if args.len() >= 2 {
                bhimpl_int_le(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntEq => {
            let v = if args.len() >= 2 {
                bhimpl_int_eq(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntNe => {
            let v = if args.len() >= 2 {
                bhimpl_int_ne(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntGt => {
            let v = if args.len() >= 2 {
                bhimpl_int_gt(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::IntGe => {
            let v = if args.len() >= 2 {
                bhimpl_int_ge(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        // Float arithmetic
        OpCode::FloatAdd => {
            let v = if args.len() >= 2 {
                bhimpl_float_add(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatSub => {
            let v = if args.len() >= 2 {
                bhimpl_float_sub(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatMul => {
            let v = if args.len() >= 2 {
                bhimpl_float_mul(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatTrueDiv => {
            let v = if args.len() >= 2 {
                bhimpl_float_truediv(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatFloorDiv => {
            let v = if args.len() >= 2 {
                bhimpl_float_floordiv(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatMod => {
            let v = if args.len() >= 2 {
                bhimpl_float_mod(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        // Float comparison
        OpCode::FloatLt => {
            let v = if args.len() >= 2 {
                bhimpl_float_lt(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatLe => {
            let v = if args.len() >= 2 {
                bhimpl_float_le(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatEq => {
            let v = if args.len() >= 2 {
                bhimpl_float_eq(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatNe => {
            let v = if args.len() >= 2 {
                bhimpl_float_ne(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatGt => {
            let v = if args.len() >= 2 {
                bhimpl_float_gt(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        OpCode::FloatGe => {
            let v = if args.len() >= 2 {
                bhimpl_float_ge(args[0], args[1])
            } else {
                ConcreteValue::Null
            };
            (v, false)
        }
        // Unknown opcode — no concrete result
        _ => (ConcreteValue::Null, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn py_div(x: i64, y: i64) -> i64 {
        bhimpl_int_floordiv(ConcreteValue::Int(x), ConcreteValue::Int(y))
            .getint()
            .unwrap()
    }

    fn py_mod(x: i64, y: i64) -> i64 {
        bhimpl_int_mod(ConcreteValue::Int(x), ConcreteValue::Int(y))
            .getint()
            .unwrap()
    }

    /// The truncating primitives beside them round toward zero; these two
    /// round toward negative infinity, which is what `int_py_div` /
    /// `int_py_mod` are called for.
    #[test]
    fn the_floor_helpers_round_toward_negative_infinity() {
        for (x, y, div, rem) in [
            (7_i64, 3_i64, 2_i64, 1_i64),
            (-7, 3, -3, 2),
            (7, -3, -3, -2),
            (-7, -3, 2, -1),
            (9, 3, 3, 0),
            (-9, 3, -3, 0),
            (9, -3, -3, 0),
        ] {
            assert_eq!(py_div(x, y), div, "{x} // {y}");
            assert_eq!(py_mod(x, y), rem, "{x} % {y}");
        }
    }

    /// A remainder near the type's limit: `(r + y) % y` overflows on the way
    /// to the answer, while masking the divisor out of the remainder's sign
    /// does not.
    #[test]
    fn the_floor_helpers_hold_at_the_limits() {
        // The same floor, computed with room to spare so the reference cannot
        // share a wrap with what it is checking.
        fn floor_div_wide(x: i128, y: i128) -> i128 {
            let q = x / y;
            if x % y != 0 && (x < 0) != (y < 0) {
                q - 1
            } else {
                q
            }
        }

        for (x, y) in [
            (i64::MAX - 1, i64::MAX),
            (-1, i64::MIN),
            (i64::MAX, i64::MIN),
            (i64::MIN, i64::MAX),
            (i64::MIN + 1, i64::MAX),
            (i64::MIN, 3),
            (i64::MAX, -3),
            (i64::MIN, -3),
        ] {
            let q = floor_div_wide(i128::from(x), i128::from(y));
            let r = i128::from(x) - q * i128::from(y);
            assert_eq!(i128::from(py_mod(x, y)), r, "{x} % {y}");
            // `INT_MIN // -1` is the one quotient that does not fit; the trace
            // guards it out ahead of the call, so it is not asserted here.
            if let Ok(expected) = i64::try_from(q) {
                assert_eq!(py_div(x, y), expected, "{x} // {y}");
            }
        }
    }
}
