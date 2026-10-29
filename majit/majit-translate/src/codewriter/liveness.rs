//! Liveness computation for flattened JitCode instructions.
//!
//! RPython equivalent: `rpython/jit/codewriter/liveness.py`.
//!
//! Expands `-live-` markers in the flattened instruction sequence to
//! include all values that are alive at that point (written before and
//! read afterwards). This information is used by guard operations in the
//! meta-interpreter to know which values to save on failure.
//!
//! The algorithm is a backward dataflow analysis that iterates to fixpoint.

use std::collections::{HashMap, HashSet};

use crate::flatten::{FlatOp, Label, RegKind, Register, SSARepr};
use crate::regalloc::RegAllocator;

/// Compute liveness for a flattened function.
///
/// RPython: `liveness.py::compute_liveness(ssarepr)`.
///
/// Modifies the flattened ops in place: each `FlatOp::Live` marker
/// gets its `live_values` set populated with all [`Register`]s alive
/// at that point in the instruction sequence.
///
/// `regallocs` supplies the per-`Variable` `(kind, color)` mapping used
/// to convert FlatOp::Op operand `Variable`s to [`Register`]s for
/// the alive set — RPython works directly on Registers because
/// `serialize_op` already projects `Variable`→`Register` via
/// `getcolor`; pyre still walks the conversion here at the liveness
/// boundary because `FlatOp::Op` carries the pre-flatten
/// `SpaceOperation` whose operand reads are kind-driven.
/// RPython liveness.py.
///
/// Operand kinds read through `Variable.concretetype` directly via
/// `variable_to_register`, so this entry no longer needs the
/// surrounding `FunctionGraph` — the kind lives on every operand
/// Variable itself (upstream `Variable.concretetype` parity).
pub fn compute_liveness(flattened: &mut SSARepr, regallocs: &HashMap<RegKind, RegAllocator>) {
    let mut label2alive: HashMap<Label, HashSet<Register>> = HashMap::new();

    loop {
        if !compute_liveness_pass(&mut flattened.insns, &mut label2alive, regallocs) {
            break;
        }
    }
    remove_repeated_live(&mut flattened.insns);
}

/// Resolve a `Variable` from a `FlatOp::Op` operand to its
/// [`Register`].
///
/// **Structural divergence (TODO)**: PyPy `liveness.py:67` walks
/// instructions whose register operands are already
/// [`Register`] / `ListOfKind` because `flatten_list()`
/// (`flatten.py`) projected `Variable` → `Register` at
/// flatten time.  Pyre's `FlatOp::Op` still carries
/// [`crate::model::SpaceOperation`] with `Variable` slots, so the
/// liveness pass has to redo the `getcolor` lookup here.  The fix
/// is to migrate `SpaceOperation` slots to `Register` so liveness
/// can read the kind off the operand directly; until that lands
/// this helper preserves the same `(kind, color)` answer.
///
/// **RPython invariant** (`flatten.py` `getcolor`): every
/// `Variable` has a single `(kind, color)` via
/// `getkind(v.concretetype)` + `regallocs[kind]`.  Reads kind via
/// `FunctionGraph::concretetype_of(var)` and color via
/// `RegAllocator::color_for_variable(var)`; a miss panics — PyPy
/// would never fall back to other classes.
fn variable_to_register(
    var: &crate::flowspace::model::Variable,
    regallocs: &HashMap<RegKind, RegAllocator>,
) -> Option<Register> {
    use crate::model::ConcreteType;
    use crate::model::FunctionGraph;
    let declared = FunctionGraph::concretetype_of(var);
    let kind = match declared {
        ConcreteType::Signed => Some(RegKind::Int),
        ConcreteType::GcRef => Some(RegKind::Ref),
        ConcreteType::Float => Some(RegKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    };
    if let Some(kind) = kind {
        let ra = regallocs.get(&kind).unwrap_or_else(|| {
            panic!(
                "variable_to_register: graph declared kind {kind:?} for {var:?} \
                 but regallocs map is missing the entry",
            )
        });
        let color = ra.color_for_variable(var).unwrap_or_else(|| {
            let other_classes: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .filter(|k| **k != kind)
                .filter(|k| {
                    regallocs
                        .get(*k)
                        .is_some_and(|ra| ra.contains_variable(var))
                })
                .copied()
                .collect();
            panic!(
                "variable_to_register: graph declared kind {kind:?} for {var:?} \
                 but regallocs[{kind:?}] has no coloring (other classes with a \
                 coloring: {other_classes:?})",
            )
        });
        return Some(Register::new(kind, color));
    }
    // Void / Unknown — fall through to KINDS scan.
    let mut found: Option<Register> = None;
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        if let Some(ra) = regallocs.get(&kind)
            && let Some(color) = ra.color_for_variable(var)
        {
            if let Some(prev) = found {
                panic!(
                    "variable_to_register: Variable {var:?} colored in multiple \
                         regalloc classes ({:?} and {kind:?}) — RPython `getkind` must \
                         give exactly one",
                    prev.kind,
                );
            }
            found = Some(Register::new(kind, color));
        }
    }
    found
}

/// RPython liveness.py: remove_repeated_live.
///
/// Merges consecutive `-live-` markers into a single one (union of
/// all live registers). Labels between them are preserved.
pub fn remove_repeated_live(ops: &mut Vec<FlatOp>) {
    let mut result: Vec<FlatOp> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if !matches!(&ops[i], FlatOp::Live { .. }) {
            result.push(ops[i].clone());
            i += 1;
            continue;
        }
        let mut labels = Vec::new();
        let mut lives = Vec::new();
        while i < ops.len() {
            match &ops[i] {
                FlatOp::Live { live_values } => {
                    lives.push(live_values.clone());
                    i += 1;
                }
                FlatOp::Label(_) => {
                    labels.push(ops[i].clone());
                    i += 1;
                }
                _ => break,
            }
        }
        result.extend(labels);
        if lives.len() == 1 {
            // RPython `remove_repeated_live`: a lone marker is moved after
            // interleaved labels but otherwise preserved byte-for-byte. The
            // union/sort step belongs only to a genuinely repeated run.
            result.push(FlatOp::Live {
                live_values: lives.pop().unwrap(),
            });
            continue;
        }
        let mut merged_live: HashSet<Register> = HashSet::new();
        for live in lives {
            merged_live.extend(live);
        }
        // Stable order so the final bytecode encoding is reproducible.
        let mut merged: Vec<Register> = merged_live.into_iter().collect();
        merged.sort_by_key(|r| (r.kind as u8, r.index));
        result.push(FlatOp::Live {
            live_values: merged,
        });
    }
    *ops = result;
}

/// One backward pass of liveness analysis.
/// Returns true if any label's alive set grew (needs another iteration).
///
/// RPython: `_compute_liveness_must_continue(ssarepr, label2alive)`.
///
/// Walks backward through the instruction sequence. At each `-live-`
/// marker, expands it to include all values alive at that point.
/// Reads each `FlatOp::Op` operand's kind via
/// `FunctionGraph::concretetype_of(&var)` and color via
/// `RegAllocator::color_for_variable(&var)`, matching upstream
/// `flatten.py getcolor` line-for-line.
fn compute_liveness_pass(
    ops: &mut [FlatOp],
    label2alive: &mut HashMap<Label, HashSet<Register>>,
    regallocs: &HashMap<RegKind, RegAllocator>,
) -> bool {
    let mut alive: HashSet<Register> = HashSet::new();
    let mut must_continue = false;

    // `def_value` and `use_value` route a `FlatOp::Op`-side
    // [`Variable`] through regalloc to its [`Register`] before
    // joining the alive set.  Void / unallocated values are silently
    // dropped (RPython's `flatten.py if v.concretetype is not
    // lltype.Void` makes the same filter at flatten time).
    let def_value = |alive: &mut HashSet<Register>, var: &crate::flowspace::model::Variable| {
        if let Some(r) = variable_to_register(var, regallocs) {
            alive.remove(&r);
        }
    };
    let use_value = |alive: &mut HashSet<Register>, var: &crate::flowspace::model::Variable| {
        if let Some(r) = variable_to_register(var, regallocs) {
            alive.insert(r);
        }
    };

    for i in (0..ops.len()).rev() {
        match &ops[i] {
            FlatOp::Label(label) => {
                let label = *label;
                let alive_at_point = label2alive.entry(label).or_default();
                let prev_len = alive_at_point.len();
                alive_at_point.extend(alive.iter());
                if alive_at_point.len() != prev_len {
                    must_continue = true;
                }
            }
            FlatOp::Live { live_values } => {
                // RPython liveness.py:44-52: `-live-` markers are
                // expanded to the full set of [`Register`]s alive at
                // this point.  Pre-seeded values (e.g. forced by
                // jtransform) merge in here.
                for r in live_values {
                    alive.insert(*r);
                }
                ops[i] = FlatOp::Live {
                    live_values: alive.iter().copied().collect(),
                };
            }
            FlatOp::EndOfBlock => {
                alive.clear();
            }
            FlatOp::Unreachable => {
                // Same liveness semantics as `EndOfBlock`: the
                // instruction stream past this point cannot execute,
                // so registers that are only live in the dead tail
                // must NOT leak backward into earlier `-live-`
                // markers.  Without this clear, regalloc / resume
                // would over-pin those slots for no observable use.
                alive.clear();
            }
            FlatOp::Op(inner_op) => {
                if let Some(result) = inner_op.result.as_ref() {
                    def_value(&mut alive, result);
                }
                for var in crate::inline::op_variable_refs(&inner_op.kind) {
                    use_value(&mut alive, &var);
                }
            }
            FlatOp::Jump(label) => {
                let label = *label;
                if let Some(alive_at_target) = label2alive.get(&label) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::CatchException { target } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfExceptionMismatch { target, .. } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfNot { cond, target } => {
                let target = *target;
                alive.insert(*cond);
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfNotOp { args, target, .. } => {
                // Fused guard (`goto_if_not_<op>`): a pure use of each
                // comparison operand, plus the false-path target merge —
                // same backward semantics as `GotoIfNot`.
                let target = *target;
                for arg in args {
                    alive.insert(*arg);
                }
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Switch { value, targets } => {
                alive.insert(*value);
                for (_, target) in targets {
                    if let Some(alive_at_target) = label2alive.get(target) {
                        alive.extend(alive_at_target.iter());
                    }
                }
            }
            FlatOp::IntBinOpJumpIfOvf {
                target,
                lhs,
                rhs,
                dst,
                ..
            } => {
                alive.remove(dst);
                alive.insert(*lhs);
                alive.insert(*rhs);
                if let Some(alive_at_target) = label2alive.get(target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Move { dst, src } => {
                // `flatten.py:333` — `int_copy %src -> %dst`.
                // Backward: dst is defined, register source is used,
                // constant source contributes nothing.
                alive.remove(dst);
                if let crate::flatten::RegOrConst::Reg(r) = src {
                    alive.insert(*r);
                }
            }
            FlatOp::Push(src) => {
                // `flatten.py:329` — `int_push %src` reads `src` into
                // the per-kind tmpreg.  Backward: a pure use of src.
                alive.insert(*src);
            }
            FlatOp::Pop(dst) => {
                // `flatten.py:331` — `int_pop -> %dst` writes tmpreg
                // into dst.  Backward: a pure def of dst.
                alive.remove(dst);
            }
            FlatOp::LastException { dst } | FlatOp::LastExcValue { dst } => {
                // Register operand carries (kind, color); the alive
                // set is Register-keyed so the def removes the
                // matching slot directly without any Variable bridge.
                alive.remove(dst);
            }
            FlatOp::Reraise => {}
            FlatOp::IntReturn(v) | FlatOp::RefReturn(v) | FlatOp::FloatReturn(v) => {
                // Backward: the return value is alive at this point;
                // after it (forward) nothing is.  RegOrConst::Reg
                // contributes its Register to the alive set;
                // Constants don't.
                alive.clear();
                if let crate::flatten::RegOrConst::Reg(r) = v {
                    alive.insert(*r);
                }
            }
            FlatOp::VoidReturn => {
                alive.clear();
            }
            FlatOp::Raise(v) => {
                alive.clear();
                if let crate::flatten::RegOrConst::Reg(r) = v {
                    alive.insert(*r);
                }
            }
        }
    }

    must_continue
}

pub use majit_jitcode::codewriter::liveness::*;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::flatten::FlatOp;
    use crate::model::{OpKind, SpaceOperation, ValueType};

    #[test]
    fn lone_live_marker_keeps_its_original_register_sequence() {
        // RPython `remove_repeated_live`: labels move before a lone marker,
        // but the marker itself bypasses the multi-marker set/sort path.
        let live = vec![
            Register::new(RegKind::Int, 2),
            Register::new(RegKind::Int, 1),
            Register::new(RegKind::Int, 2),
        ];
        let mut ops = vec![
            FlatOp::Live {
                live_values: live.clone(),
            },
            FlatOp::Label(Label(7)),
        ];

        remove_repeated_live(&mut ops);

        assert!(matches!(ops.first(), Some(FlatOp::Label(Label(7)))));
        assert!(matches!(
            ops.get(1),
            Some(FlatOp::Live { live_values }) if live_values == &live
        ));
    }

    #[test]
    fn basic_liveness() {
        // v0 = Input
        // v1 = ConstInt(42)
        // v2 = BinOp(v0, v1)
        // Return v2
        let regallocs: HashMap<RegKind, RegAllocator> = HashMap::new();
        let v0 = crate::flowspace::model::Variable::new();
        let v1 = crate::flowspace::model::Variable::new();
        let v2 = crate::flowspace::model::Variable::new();
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![
                FlatOp::Label(Label(0)),
                FlatOp::op(SpaceOperation {
                    result: Some(v0.clone()),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                }),
                FlatOp::op(SpaceOperation {
                    result: Some(v1.clone()),
                    kind: OpKind::ConstInt(42),
                }),
                FlatOp::op(SpaceOperation {
                    result: Some(v2),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: v0,
                        rhs: v1,
                        result_ty: ValueType::Int,
                    },
                }),
            ],
            num_blocks: 1,
            insns_pos: None,
        };

        // Should not panic.  Phase 3 added the `regallocs` parameter
        // for the FlatOp::Op `Variable → Register` bridge; pass an
        // empty map since this fixture has no inputargs that exercise
        // the conversion.
        compute_liveness(&mut flat, &regallocs);
    }

    /// A `-live-` at a call between window formation and keepalive lists
    /// the owner (`liveness.py` `compute_liveness` / `-live-` args).
    #[test]
    fn keepalive_forces_owner_live_at_preceding_call() {
        use crate::flatten::RegOrConst;
        use crate::model::{CallTarget, ConcreteType, FunctionGraph, SpaceOperation};

        let owner = crate::flowspace::model::Variable::new();
        FunctionGraph::set_concretetype_of_inline(&owner, ConcreteType::GcRef);
        let window = crate::flowspace::model::Variable::new();
        FunctionGraph::set_concretetype_of_inline(&window, ConcreteType::Signed);
        let owner_reg = Register::new(RegKind::Ref, 0);
        let window_reg = Register::new(RegKind::Int, 0);
        let mut ref_coloring = crate::regalloc::Coloring::default();
        ref_coloring.insert(owner.clone(), 0);
        let mut int_coloring = crate::regalloc::Coloring::default();
        int_coloring.insert(window.clone(), 0);
        let mut regallocs = HashMap::new();
        regallocs.insert(
            RegKind::Ref,
            crate::regalloc::RegAllocator {
                coloring: ref_coloring,
                num_regs: 1,
            },
        );
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocator {
                coloring: int_coloring,
                num_regs: 1,
            },
        );
        let mut flat = SSARepr {
            name: "window_keepalive".into(),
            insns: vec![
                FlatOp::Label(Label(0)),
                FlatOp::op(SpaceOperation {
                    result: Some(owner.clone()),
                    kind: OpKind::Input {
                        name: "owner".into(),
                        ty: ValueType::Ref(None),
                        class_root: None,
                    },
                }),
                FlatOp::op(SpaceOperation {
                    result: Some(window.clone()),
                    kind: OpKind::ConstInt(0),
                }),
                FlatOp::Live {
                    live_values: vec![],
                },
                FlatOp::op(SpaceOperation {
                    result: None,
                    kind: OpKind::Call {
                        target: CallTarget::function_path(["window_param"]),
                        args: crate::model::call_args(vec![window.clone()]),
                        result_ty: ValueType::Void,
                    },
                }),
                FlatOp::Live {
                    live_values: vec![owner_reg],
                },
                FlatOp::IntReturn(RegOrConst::Reg(window_reg)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        compute_liveness(&mut flat, &regallocs);
        let call_idx = flat
            .insns
            .iter()
            .position(|op| {
                matches!(
                    op,
                    FlatOp::Op(inner) if matches!(inner.kind, OpKind::Call { .. })
                )
            })
            .expect("call");
        let FlatOp::Live { live_values } = &flat.insns[call_idx - 1] else {
            panic!(
                "expected -live- immediately before the call; insns={:?}",
                flat.insns
            );
        };
        assert!(
            live_values.contains(&owner_reg),
            "owner must be live at the call -live-; insns={:?}",
            flat.insns
        );
    }
}
