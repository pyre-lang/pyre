//! Test-only trace comparison utilities — no RPython equivalent.
//! Used by integration tests for structural trace parity validation.

use crate::history::TreeLoop;
use indexmap::IndexMap;
use majit_ir::{Op, OpRef, Type};

/// A small, stable parity case format for comparing majit traces against
/// RPython-derived expectations.
///
/// The key rule is that proc-macro expansion details do not matter. Only the
/// normalized runtime seam matters: the recorded IR operations, their argument
/// order, and any explicit guard fail-args.
#[derive(Clone, Debug)]
pub struct TraceParityCase<'a> {
    pub name: &'a str,
    pub rpython_reference: &'a str,
    pub expected_lines: &'a [&'a str],
}

#[derive(Default)]
struct VarRenumbering {
    next_id: u32,
    ids: IndexMap<OpRef, u32>,
}

impl VarRenumbering {
    fn id_for(&mut self, opref: OpRef) -> u32 {
        if let Some(&id) = self.ids.get(&opref) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.ids.insert(opref, id);
        id
    }
}

fn render_arg(arg: OpRef, constants: &IndexMap<u32, i64>, vars: &mut VarRenumbering) -> String {
    // history.py/268/314 — inline-Const variants carry value inline.
    if let Some(value) = arg
        .inline_const_bits()
        .or_else(|| constants.get(&arg.raw()).copied())
    {
        value.to_string()
    } else {
        format!("v{}", vars.id_for(arg))
    }
}

fn render_op(op: &Op, constants: &IndexMap<u32, i64>, vars: &mut VarRenumbering) -> String {
    let args = op
        .getarglist()
        .iter()
        .map(|arg| render_arg(arg.to_opref(), constants, vars))
        .collect::<Vec<_>>()
        .join(", ");

    let mut line = if op.opcode.is_guard() || op.opcode.result_type() == Type::Void {
        format!("{:?}({args})", op.opcode)
    } else {
        format!("v{} = {:?}({args})", vars.id_for(op.pos().get()), op.opcode)
    };

    if let Some(fail_args) = op.guard_fail_args() {
        let fail_args = fail_args
            .iter()
            .map(|arg| render_arg(arg.to_opref(), constants, vars))
            .collect::<Vec<_>>()
            .join(", ");
        line.push_str(&format!(" [fail_args={fail_args}]"));
    }

    line
}

/// Normalize a completed trace into stable line strings for parity checks.
///
/// Inputargs are pre-numbered in declaration order so that parity cases
/// can reference them as `v0`, `v1`, … regardless of which ops happen
/// to touch them first. Without this pre-population, the first
/// inputarg touched by a later op would take the next available vN,
/// shifting op-result IDs and breaking the stable line format
/// parity cases expect (mirrors RPython `opimpl_*` trace shapes that
/// assume the inputarg slot numbering is the canonical prefix).
pub fn normalize_trace(trace: &TreeLoop, constants: &IndexMap<u32, i64>) -> Vec<String> {
    let mut vars = VarRenumbering::default();
    for inputarg in &trace.inputargs {
        // Pre-allocate v0..vN for the inputarg slots. RPython inputargs
        // are concrete InputArgInt/InputArgFloat/InputArgRef instances;
        // pre-number the matching typed OpRef so later render_arg hits
        // reuse these IDs.
        vars.id_for(OpRef::input_arg_typed(inputarg.index, inputarg.tp.get()));
    }
    trace
        .ops
        .iter()
        .map(|op| render_op(op, constants, &mut vars))
        .collect()
}

/// Normalize an op slice into the same stable line format used by
/// [`normalize_trace`].
pub fn normalize_ops(ops: &[Op], constants: &IndexMap<u32, i64>) -> Vec<String> {
    let mut vars = VarRenumbering::default();
    ops.iter()
        .map(|op| render_op(op, constants, &mut vars))
        .collect()
}

/// Assert that a trace matches a normalized parity case.
pub fn assert_trace_parity(
    trace: &TreeLoop,
    constants: &IndexMap<u32, i64>,
    case: &TraceParityCase<'_>,
) {
    let actual = normalize_trace(trace, constants);
    let expected = case
        .expected_lines
        .iter()
        .map(|line| (*line).to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        actual, expected,
        "trace parity mismatch for {} (RPython reference: {})",
        case.name, case.rpython_reference
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::Trace;
    use crate::{TraceCtx, make_fail_descr};
    use majit_ir::{OpCode, Type};

    fn finish_trace_ctx(
        mut ctx: TraceCtx,
        finish_args: &[OpRef],
    ) -> (TreeLoop, IndexMap<u32, i64>) {
        ctx.finish(finish_args, make_fail_descr(finish_args.len()));
        let constants = IndexMap::new();
        let trace = ctx.into_tree_loop();
        (trace, constants)
    }

    #[test]
    fn trace_ctx_constants_match_parity_case() {
        let mut recorder = Trace::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let one = ctx.const_int(1);
        let two = ctx.const_int(2);
        let sum = ctx.record_op(OpCode::IntAdd, &[i0, one]);
        let prod = ctx.record_op(OpCode::IntMul, &[sum, two]);
        let (trace, constants) = finish_trace_ctx(ctx, &[prod]);

        let case = TraceParityCase {
            name: "trace_ctx_constant_arith",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: special_int_add / opimpl_int_mul",
            expected_lines: &["v1 = IntAdd(v0, 1)", "v2 = IntMul(v1, 2)", "Finish(v2)"],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn guard_fail_args_are_part_of_normalized_trace() {
        let mut recorder = Trace::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let zero = ctx.const_int(0);
        let cond = ctx.record_op(OpCode::IntGt, &[i0, zero]);
        let g = ctx.record_guard(OpCode::GuardTrue, &[cond], 1);
        ctx.finish(&[i0], make_fail_descr(1));
        ctx.set_fail_args(g, &[i0]);
        let constants = IndexMap::new();
        let trace = ctx.into_tree_loop();

        let case = TraceParityCase {
            name: "guard_fail_args_surface_in_seam",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: implement_guard_value",
            expected_lines: &[
                "v1 = IntGt(v0, 0)",
                "GuardTrue(v1) [fail_args=v0]",
                "Finish(v0)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn unary_bitwise_and_comparison_ops_normalize_stably() {
        let mut recorder = Trace::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let i1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let neg = ctx.record_op(OpCode::IntNeg, &[i0]);
        let anded = ctx.record_op(OpCode::IntAnd, &[neg, i1]);
        let seven = ctx.const_int(7);
        let zero = ctx.const_int(0);
        let xored = ctx.record_op(OpCode::IntXor, &[anded, seven]);
        let cond = ctx.record_op(OpCode::IntGe, &[xored, zero]);
        let g = ctx.record_guard(OpCode::GuardTrue, &[cond], 2);
        ctx.finish(&[xored], make_fail_descr(1));
        ctx.set_fail_args(g, &[i0, i1]);
        let constants = IndexMap::new();
        let trace = ctx.into_tree_loop();

        let case = TraceParityCase {
            name: "unary_bitwise_and_comparison_ops",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_and / opimpl_int_xor / opimpl_int_ge",
            expected_lines: &[
                "v2 = IntNeg(v0)",
                "v3 = IntAnd(v2, v1)",
                "v4 = IntXor(v3, 7)",
                "v5 = IntGe(v4, 0)",
                "GuardTrue(v5) [fail_args=v0, v1]",
                "Finish(v4)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn shift_ops_match_expected_trace_shape() {
        let mut recorder = Trace::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let i1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let one = ctx.const_int(1);
        let two = ctx.const_int(2);
        let lhs = ctx.record_op(OpCode::IntLshift, &[i0, one]);
        let rhs = ctx.record_op(OpCode::IntRshift, &[i1, two]);
        let merged = ctx.record_op(OpCode::IntOr, &[lhs, rhs]);
        let (trace, constants) = finish_trace_ctx(ctx, &[merged]);

        let case = TraceParityCase {
            name: "shift_ops",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_lshift / opimpl_int_rshift",
            expected_lines: &[
                "v2 = IntLshift(v0, 1)",
                "v3 = IntRshift(v1, 2)",
                "v4 = IntOr(v2, v3)",
                "Finish(v4)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }
}
