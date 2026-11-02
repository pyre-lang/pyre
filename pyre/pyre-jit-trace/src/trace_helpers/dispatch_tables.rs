//! Operator → opcode dispatch tables. The data analog of `resoperation.py`
//! / `lloperation.py`: maps `BinaryOperator` / `ComparisonOperator` to the
//! `majit_ir::OpCode` and per-op flags.

use super::*;
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};

/// Int binary operator dispatch: `(base_op, inplace_op, opcode, has_overflow, needs_concrete_check)`.
pub const INT_BINOP_TABLE: &[(BinaryOperator, BinaryOperator, majit_ir::OpCode, bool, bool)] = &[
    (
        BinaryOperator::Add,
        BinaryOperator::InplaceAdd,
        majit_ir::OpCode::IntAddOvf,
        true,
        false,
    ),
    (
        BinaryOperator::Subtract,
        BinaryOperator::InplaceSubtract,
        majit_ir::OpCode::IntSubOvf,
        true,
        false,
    ),
    (
        BinaryOperator::Multiply,
        BinaryOperator::InplaceMultiply,
        majit_ir::OpCode::IntMulOvf,
        true,
        false,
    ),
    (
        BinaryOperator::And,
        BinaryOperator::InplaceAnd,
        majit_ir::OpCode::IntAnd,
        false,
        false,
    ),
    (
        BinaryOperator::Or,
        BinaryOperator::InplaceOr,
        majit_ir::OpCode::IntOr,
        false,
        false,
    ),
    (
        BinaryOperator::Xor,
        BinaryOperator::InplaceXor,
        majit_ir::OpCode::IntXor,
        false,
        false,
    ),
    (
        BinaryOperator::Lshift,
        BinaryOperator::InplaceLshift,
        majit_ir::OpCode::IntLshift,
        false,
        true,
    ),
    (
        BinaryOperator::Rshift,
        BinaryOperator::InplaceRshift,
        majit_ir::OpCode::IntRshift,
        false,
        true,
    ),
];

/// Look up int binary operator dispatch entry.
pub fn int_binop_lookup(op: BinaryOperator) -> Option<(majit_ir::OpCode, bool, bool)> {
    INT_BINOP_TABLE
        .iter()
        .find(|(base, inplace, _, _, _)| *base == op || *inplace == op)
        .map(|(_, _, opcode, ovf, concrete)| (*opcode, *ovf, *concrete))
}

/// Float binary operator dispatch: `(base_op, inplace_op, opcode)`.
///
/// resoperation.py:959-962: only FLOAT_ADD, FLOAT_SUB, FLOAT_MUL, FLOAT_TRUEDIV.
/// lloperation.py:260-261: no float_floordiv, no float_mod, no float_pow.
/// FloorDivide → _divmod_w() residual call (floatobject.py).
/// Remainder → math_fmod residual call (floatobject.py:520).
/// Power → ll_math_pow residual call (ll_math.py).
pub const FLOAT_BINOP_TABLE: &[(BinaryOperator, BinaryOperator, majit_ir::OpCode)] = &[
    (
        BinaryOperator::Add,
        BinaryOperator::InplaceAdd,
        majit_ir::OpCode::FloatAdd,
    ),
    (
        BinaryOperator::Subtract,
        BinaryOperator::InplaceSubtract,
        majit_ir::OpCode::FloatSub,
    ),
    (
        BinaryOperator::Multiply,
        BinaryOperator::InplaceMultiply,
        majit_ir::OpCode::FloatMul,
    ),
    (
        BinaryOperator::TrueDivide,
        BinaryOperator::InplaceTrueDivide,
        majit_ir::OpCode::FloatTrueDiv,
    ),
];

/// Look up float binary operator dispatch entry.
/// Returns None for FloorDivide, Remainder, Power and unsupported ops.
pub fn float_binop_lookup(op: BinaryOperator) -> Option<majit_ir::OpCode> {
    FLOAT_BINOP_TABLE
        .iter()
        .find(|(base, inplace, _)| *base == op || *inplace == op)
        .map(|(_, _, opcode)| *opcode)
}

/// Comparison operator dispatch: `(comp_op, int_opcode, float_opcode)`.
pub const COMPARE_TABLE: &[(ComparisonOperator, majit_ir::OpCode, majit_ir::OpCode)] = &[
    (
        ComparisonOperator::Less,
        majit_ir::OpCode::IntLt,
        majit_ir::OpCode::FloatLt,
    ),
    (
        ComparisonOperator::LessOrEqual,
        majit_ir::OpCode::IntLe,
        majit_ir::OpCode::FloatLe,
    ),
    (
        ComparisonOperator::Greater,
        majit_ir::OpCode::IntGt,
        majit_ir::OpCode::FloatGt,
    ),
    (
        ComparisonOperator::GreaterOrEqual,
        majit_ir::OpCode::IntGe,
        majit_ir::OpCode::FloatGe,
    ),
    (
        ComparisonOperator::Equal,
        majit_ir::OpCode::IntEq,
        majit_ir::OpCode::FloatEq,
    ),
    (
        ComparisonOperator::NotEqual,
        majit_ir::OpCode::IntNe,
        majit_ir::OpCode::FloatNe,
    ),
];

/// Look up comparison operator dispatch for int operands.
pub fn int_compare_lookup(op: ComparisonOperator) -> majit_ir::OpCode {
    COMPARE_TABLE
        .iter()
        .find(|(cmp, _, _)| *cmp == op)
        .map(|(_, int_op, _)| *int_op)
        .expect("all ComparisonOperator variants are covered")
}

/// Look up comparison operator dispatch for float operands.
pub fn float_compare_lookup(op: ComparisonOperator) -> majit_ir::OpCode {
    COMPARE_TABLE
        .iter()
        .find(|(cmp, _, _)| *cmp == op)
        .map(|(_, _, float_op)| *float_op)
        .expect("all ComparisonOperator variants are covered")
}
