//! Flow-space interpreter frame state.
//!
//! RPython upstream: `rpython/flowspace/framestate.py` (148 LOC).
//!
//! `FrameState` captures the interpreter's local variables, value
//! stack, pending exception, block stack, and next-instruction offset
//! at a point in the flow-space walk. It is the merge unit consumed
//! by `flowcontext.mergeblock` when two execution paths rejoin.
//!
//! ## Deviations from upstream (parity rule #1)
//!
//! * RPython `locals_w` holds a list whose cells are `Variable`,
//!   `Constant`, or `None` (undefined local). Rust encodes this as
//!   `Vec<Option<Hlvalue>>` — the closed-sum `Hlvalue` already covers
//!   `Variable | Constant`, and `Option` carries the `None` case.
//! * RPython `stack` holds a list whose cells are `Variable`,
//!   `Constant`, *or* `FlowSignal` (signals are placed on the stack
//!   between SETUP_*/POP_BLOCK). Rust introduces a `StackElem` enum;
//!   upstream duck-typing is closed as one of two variants.
//! * `FrameBlock` is imported from `flowcontext`;
//!   `FlowSignal` likewise. RPython uses lazy imports inside function
//!   bodies (`from rpython.flowspace.flowcontext import FlowSignal`)
//!   to break a module cycle. Rust has no lazy `use`; the cycle is
//!   broken by sharing the types in `flowcontext.rs`.
//! * RPython's `union()` dispatches on Python type-equality
//!   (`type(w1) is not type(w2)`). Rust collapses this to
//!   `FlowSignalTag` comparison — semantically identical for the
//!   closed variant set.

use super::flowcontext::{FlowSignal, FrameBlock};
use super::model::{ConstValue, Constant, FSException, Hlvalue, Variable};

/// Per-cell payload in the flow-space value stack.
///
/// RPython stack cells are any of `Variable | Constant | FlowSignal`;
/// in Rust we close this as a tagged union — `Hlvalue` (Variable or
/// Constant) or a `FlowSignal` instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackElem {
    /// A concrete flow-space value on the stack.
    Value(Hlvalue),
    /// A pending translator-level control-flow signal (Return / Raise
    /// / Break / Continue / RaiseImplicit) placed on the stack by
    /// SETUP_FINALLY etc.
    Signal(FlowSignal),
}

impl StackElem {
    /// Treat the cell as an `Hlvalue` where possible; signals have no
    /// direct `Hlvalue` equivalent and return `None`.
    pub fn as_hlvalue(&self) -> Option<&Hlvalue> {
        match self {
            StackElem::Value(h) => Some(h),
            StackElem::Signal(_) => None,
        }
    }
}

impl From<Variable> for StackElem {
    fn from(v: Variable) -> Self {
        StackElem::Value(Hlvalue::Variable(v))
    }
}

impl From<Constant> for StackElem {
    fn from(c: Constant) -> Self {
        StackElem::Value(Hlvalue::Constant(c))
    }
}

impl From<Hlvalue> for StackElem {
    fn from(h: Hlvalue) -> Self {
        StackElem::Value(h)
    }
}

impl From<FlowSignal> for StackElem {
    fn from(s: FlowSignal) -> Self {
        StackElem::Signal(s)
    }
}

/// Interpreter frame snapshot used as the merge unit in flow-space.
///
/// RPython basis: `framestate.py` — `class FrameState`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameState {
    /// RPython `FrameState.locals_w` — slot `None` stands in for
    /// upstream's literal `None` (undefined local).
    pub locals_w: Vec<Option<Hlvalue>>,
    /// RPython `FrameState.stack` — mixed value / signal stack.
    pub stack: Vec<StackElem>,
    /// RPython `FrameState.last_exception`.
    pub last_exception: Option<FSException>,
    /// RPython `FrameState.blocklist`.
    pub blocklist: Vec<FrameBlock>,
    /// RPython `FrameState.next_offset`.
    pub next_offset: i64,
    /// RPython `FrameState._mergeable` — cached flattening of
    /// `locals_w + recursively_flatten(stack) + [exc_type, exc_value]`.
    /// RPython stores this on `self._mergeable` behind the `mergeable`
    /// property; we keep the same lazy-cache shape via `Option`.
    mergeable_cache: Option<Vec<MergeCell>>,
}

/// Slot shape of `FrameState.mergeable`.
///
/// RPython `mergeable` is a flat list whose cells are `Variable`,
/// `Constant`, or `None`. Rust closes this as `Option<Hlvalue>`; the
/// `None` variant mirrors upstream's literal `None` (undefined local
/// or missing exception).
pub type MergeCell = Option<Hlvalue>;

/// Raised when two `FrameState`s cannot be merged.
///
/// RPython basis: `framestate.py` — `class UnionError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionError;

impl std::fmt::Display for UnionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UnionError: the two states should be merged.")
    }
}

impl std::error::Error for UnionError {}

impl FrameState {
    /// RPython `FrameState.__init__`.
    pub fn new(
        locals_w: Vec<Option<Hlvalue>>,
        stack: Vec<StackElem>,
        last_exception: Option<FSException>,
        blocklist: Vec<FrameBlock>,
        next_offset: i64,
    ) -> Self {
        FrameState {
            locals_w,
            stack,
            last_exception,
            blocklist,
            next_offset,
            mergeable_cache: None,
        }
    }

    /// RPython `FrameState.mergeable` property.
    ///
    /// Returns the flattened view used for state comparison and
    /// merging: `locals_w + recursively_flatten(stack) + [exc_type,
    /// exc_value]`.
    pub fn mergeable(&mut self) -> &[MergeCell] {
        if self.mergeable_cache.is_none() {
            self.mergeable_cache = Some(self.build_mergeable());
        }
        self.mergeable_cache.as_ref().unwrap()
    }

    /// Immutable accessor for cases where the caller cannot mutate
    /// `self` (e.g. from a `&FrameState` in `union` / `matches`).
    /// RPython mutates `self._mergeable` inside the property getter;
    /// we split read and cached-read to match Rust borrow rules while
    /// preserving the one-pass build.
    pub fn mergeable_view(&self) -> Vec<MergeCell> {
        if let Some(cached) = &self.mergeable_cache {
            return cached.clone();
        }
        self.build_mergeable()
    }

    fn build_mergeable(&self) -> Vec<MergeCell> {
        let mut data: Vec<MergeCell> = Vec::new();
        data.extend(self.locals_w.iter().cloned());
        // RPython `recursively_flatten(self.stack)` — signals on the
        // stack unwrap into their `args`.
        data.extend(recursively_flatten(&self.stack).into_iter().map(Some));
        // `[exc_type, exc_value]` tail per upstream — `Constant(None)`
        // sentinels when there is no pending exception.
        match &self.last_exception {
            None => {
                data.push(Some(Hlvalue::Constant(Constant::new(ConstValue::None))));
                data.push(Some(Hlvalue::Constant(Constant::new(ConstValue::None))));
            }
            Some(exc) => {
                data.push(Some(exc.w_type.clone()));
                data.push(Some(exc.w_value.clone()));
            }
        }
        data
    }

    /// RPython `FrameState.copy` — "Make a copy of this state in which
    /// all Variables are fresh."
    pub fn copy(&self) -> FrameState {
        let exc = self
            .last_exception
            .as_ref()
            .map(|e| FSException::new(copy_hlvalue(&e.w_type), copy_hlvalue(&e.w_value)));
        let locals_w = self
            .locals_w
            .iter()
            .map(|cell| cell.as_ref().map(copy_hlvalue))
            .collect();
        let stack = self.stack.iter().map(copy_stack_elem).collect();
        FrameState {
            locals_w,
            stack,
            last_exception: exc,
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
            mergeable_cache: None,
        }
    }

    /// RPython `FrameState.getvariables`.
    pub fn getvariables(&self) -> Vec<Variable> {
        self.mergeable_view()
            .into_iter()
            .filter_map(|cell| match cell {
                Some(Hlvalue::Variable(v)) => Some(v),
                _ => None,
            })
            .collect()
    }

    /// RPython `FrameState.matches` — "Two states match if they only
    /// differ by using different Variables at the same place."
    pub fn matches(&self, other: &FrameState) -> bool {
        assert_eq!(
            self.blocklist, other.blocklist,
            "matches: blocklist mismatch"
        );
        assert_eq!(
            self.next_offset, other.next_offset,
            "matches: next_offset mismatch"
        );
        let a = self.mergeable_view();
        let b = other.mergeable_view();
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(w1, w2)| match (w1, w2) {
            (None, None) => true,
            (Some(Hlvalue::Variable(_)), Some(Hlvalue::Variable(_))) => true,
            (Some(x), Some(y)) => x == y,
            _ => false,
        })
    }

    /// RPython `FrameState._exc_args`. Upstream returns `[w_type,
    /// w_value]` with `Constant(None)` stand-ins for the no-exception
    /// case.
    fn exc_args(&self) -> [Hlvalue; 2] {
        match &self.last_exception {
            None => [
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
            ],
            Some(exc) => [exc.w_type.clone(), exc.w_value.clone()],
        }
    }

    /// RPython `FrameState.union`. Returns a state at least as general
    /// as both `self` and `other`, or `None` when the states cannot be
    /// merged (propagates upstream's `UnionError` as `Option::None`).
    pub fn union(&self, other: &FrameState) -> Option<FrameState> {
        let locals = union_optlist(&self.locals_w, &other.locals_w).ok()?;
        let stack = union_stack(&self.stack, &other.stack).ok()?;
        let exc = if self.last_exception.is_none() && other.last_exception.is_none() {
            None
        } else {
            let a = self.exc_args();
            let b = other.exc_args();
            let w_type = union(Some(&a[0]), Some(&b[0])).ok()?;
            let w_value = union(Some(&a[1]), Some(&b[1])).ok()?;
            // `union()` only returns `None` for the undefined-local
            // path (one side is None). Exception slots are never
            // undefined — both sides carry Constant(None) sentinels
            // when the exception is absent — so unwrapping is safe.
            Some(FSException::new(w_type?, w_value?))
        };
        Some(FrameState::new(
            locals,
            stack,
            exc,
            self.blocklist.clone(),
            self.next_offset,
        ))
    }

    /// RPython `FrameState.getoutputargs` — "Return the output
    /// arguments needed to link self to targetstate."
    ///
    /// Upstream (`framestate.py`) iterates over
    /// `targetstate.mergeable`, and for every cell that is a
    /// `Variable` on the target side, appends `self.mergeable[i]`
    /// — whatever it is — to the result list. `self.mergeable[i]`
    /// may be a `Variable`, a `Constant`, *or* Python `None`
    /// (undefined-local sentinel from `locals_w`). The Rust
    /// counterpart therefore returns `Vec<MergeCell>` (=
    /// `Vec<Option<Hlvalue>>`): `None` carries the undefined-local
    /// slot through intact, matching upstream's list-with-None
    /// semantics.
    pub fn getoutputargs(&self, targetstate: &FrameState) -> Vec<MergeCell> {
        let mergeable = self.mergeable_view();
        let target = targetstate.mergeable_view();
        let mut result: Vec<MergeCell> = Vec::new();
        for (i, w_target) in target.iter().enumerate() {
            if matches!(w_target, Some(Hlvalue::Variable(_))) {
                result.push(mergeable.get(i).cloned().unwrap_or(None));
            }
        }
        result
    }
}

// free functions

/// RPython `framestate.py:_copy` — deep-copy a cell so every Variable
/// becomes a fresh one.
fn copy_hlvalue(h: &Hlvalue) -> Hlvalue {
    match h {
        Hlvalue::Variable(v) => Hlvalue::Variable(v.copy()),
        other => other.clone(),
    }
}

fn copy_stack_elem(e: &StackElem) -> StackElem {
    match e {
        StackElem::Value(h) => StackElem::Value(copy_hlvalue(h)),
        StackElem::Signal(sig) => {
            let fresh_args: Vec<Hlvalue> = sig.args().iter().map(copy_hlvalue).collect();
            StackElem::Signal(FlowSignal::rebuild_with_args(sig.tag(), fresh_args))
        }
    }
}

/// RPython `framestate.py:_union` — pairwise union over two equal-length
/// sequences of cells. Returns `UnionError` if any pair refuses to merge
/// (upstream `union` raises; we propagate as `Result::Err`).
fn union_optlist(
    seq1: &[Option<Hlvalue>],
    seq2: &[Option<Hlvalue>],
) -> Result<Vec<Option<Hlvalue>>, UnionError> {
    assert_eq!(
        seq1.len(),
        seq2.len(),
        "_union: mismatched sequence lengths"
    );
    seq1.iter()
        .zip(seq2.iter())
        .map(|(a, b)| union(a.as_ref(), b.as_ref()))
        .collect()
}

/// RPython `union(w1, w2)` (framestate.py).
///
/// Input cells are `Option<&Hlvalue>`: `None` is upstream's literal
/// `None`, which "kills" a union by returning `None`.
///
/// Returns `Ok(Some(generalised))` on normal merge, `Ok(None)` on the
/// undefined-local kill path, and `Err(UnionError)` when the two
/// Constants carry mismatched SpecTag-like markers (or FlowSignal
/// type mismatch inside the stack walker).
pub fn union(w1: Option<&Hlvalue>, w2: Option<&Hlvalue>) -> Result<Option<Hlvalue>, UnionError> {
    match (w1, w2) {
        (Some(a), Some(b)) if a == b => Ok(Some(a.clone())),
        (None, _) | (_, None) => Ok(None),
        (Some(Hlvalue::Variable(_)), _) | (_, Some(Hlvalue::Variable(_))) => {
            // `Variable()` (framestate.py `union`): a fresh phi, no
            // annotation and no concretetype.
            Ok(Some(Hlvalue::Variable(Variable::new())))
        }
        (Some(Hlvalue::Constant(c1)), Some(Hlvalue::Constant(c2))) => {
            if matches!(c1.value, ConstValue::SpecTag(_))
                || matches!(c2.value, ConstValue::SpecTag(_))
            {
                Err(UnionError)
            } else {
                Ok(Some(Hlvalue::Variable(Variable::new())))
            }
        }
    }
}

/// Stack union. Per upstream, stack cells may hold FlowSignal as well
/// as Variable/Constant; same-type signals union pairwise through
/// `rebuild`, different-type signals raise `UnionError`, and a signal
/// paired with a non-signal value also raises.
pub(crate) fn union_stack(a: &[StackElem], b: &[StackElem]) -> Result<Vec<StackElem>, UnionError> {
    assert_eq!(a.len(), b.len(), "stack length mismatch in union");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (StackElem::Value(v1), StackElem::Value(v2)) => {
                // Non-signal pair dispatches to the regular `union`.
                let merged = union(Some(v1), Some(v2))?;
                Ok(StackElem::Value(merged.unwrap_or_else(|| {
                    // `union()` only returns `None` on the
                    // undefined-local path (one side is `None`);
                    // both stack values here are `Some`, so this
                    // arm is unreachable.
                    unreachable!("union of two Somes returned None")
                })))
            }
            (StackElem::Signal(s1), StackElem::Signal(s2)) => {
                if s1.tag() != s2.tag() {
                    return Err(UnionError);
                }
                let merged_args: Vec<Hlvalue> = s1
                    .args()
                    .iter()
                    .zip(s2.args().iter())
                    .map(|(a1, a2)| {
                        union(Some(a1), Some(a2)).and_then(|merged| merged.ok_or(UnionError))
                    })
                    .collect::<Result<_, _>>()?;
                Ok(StackElem::Signal(FlowSignal::rebuild_with_args(
                    s1.tag(),
                    merged_args,
                )))
            }
            _ => Err(UnionError),
        })
        .collect()
}

/// RPython `framestate.py` — `recursively_flatten`.
///
/// Unpacks any `FlowSignal` cells in the stack into their `args`,
/// recursively. RPython operates in-place on a copy; we return a
/// flattened `Vec<Hlvalue>` for use by `mergeable`.
pub fn recursively_flatten(lst: &[StackElem]) -> Vec<Hlvalue> {
    // Fast path: no signals → just unwrap Values. Mirrors upstream's
    // `else: return lst` when no FlowSignal is found.
    let has_signal = lst.iter().any(|e| matches!(e, StackElem::Signal(_)));
    if !has_signal {
        return lst
            .iter()
            .map(|e| match e {
                StackElem::Value(h) => h.clone(),
                StackElem::Signal(_) => unreachable!(),
            })
            .collect();
    }
    let mut out: Vec<Hlvalue> = Vec::new();
    let mut work: Vec<StackElem> = lst.to_vec();
    let mut i = 0;
    while i < work.len() {
        match work[i].clone() {
            StackElem::Value(h) => {
                out.push(h);
                i += 1;
            }
            StackElem::Signal(sig) => {
                let args = sig.args();
                // Expand in place: replace the signal cell with its
                // args (as Values), then re-visit the same `i` to
                // handle any nested signals introduced by the
                // expansion (upstream's loop walks `while i <
                // len(lst)` and re-checks the newly-inserted cells).
                let replacement: Vec<StackElem> = args.into_iter().map(StackElem::Value).collect();
                work.splice(i..=i, replacement);
            }
        }
    }
    // `out` already has the expanded-value tail; but we walked `work`
    // start-to-end so the loop above only pushes Values. Anything we
    // skipped (signals) got replaced by Values via splice.
    out
}

#[cfg(test)]
mod test {
    use super::super::flowcontext::{FlowSignal, FrameBlockKind};
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // RPython basis: `test/test_framestate.py`. Upstream tests drive
    // `FlowContext.getstate/setstate`, which require F3.4+F3.7
    // (`FlowContext` and `PyGraph`). The upstream tests are ported
    // alongside F3.8 once those dependencies exist. F3.1 instead
    // builds FrameStates by hand to cover the `matches`/`copy`/
    // `union`/`getvariables`/`getoutputargs` shapes.

    static SPEC_TAG_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fresh_spectag_constant() -> Constant {
        let n = SPEC_TAG_COUNTER.fetch_add(1, Ordering::Relaxed);
        Constant::new(ConstValue::SpecTag(n))
    }

    fn build_state(locals: Vec<Option<Hlvalue>>) -> FrameState {
        FrameState::new(locals, Vec::new(), None, Vec::new(), 0)
    }

    #[test]
    fn matches_equal_states() {
        let v = Hlvalue::Variable(Variable::new());
        let fs1 = build_state(vec![
            Some(v.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        ]);
        let fs2 = build_state(vec![
            Some(v.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        ]);
        assert!(fs1.matches(&fs2));
    }

    #[test]
    fn matches_treats_any_two_variables_as_equal_slot() {
        let fs1 = build_state(vec![Some(Hlvalue::Variable(Variable::new()))]);
        let fs2 = build_state(vec![Some(Hlvalue::Variable(Variable::new()))]);
        assert!(fs1.matches(&fs2));
    }

    #[test]
    fn matches_fails_when_constants_differ() {
        let fs1 = build_state(vec![Some(Hlvalue::Constant(Constant::new(
            ConstValue::Int(1),
        )))]);
        let fs2 = build_state(vec![Some(Hlvalue::Constant(Constant::new(
            ConstValue::Int(2),
        )))]);
        assert!(!fs1.matches(&fs2));
    }

    #[test]
    fn copy_returns_fresh_variables() {
        let v = Variable::new();
        let fs1 = build_state(vec![Some(Hlvalue::Variable(v.clone()))]);
        let fs2 = fs1.copy();
        // matches accepts different Variables at the same slot,
        // verifying upstream's "only Variables may change" invariant.
        assert!(fs1.matches(&fs2));
        // But the actual Variable objects differ by identity.
        match (&fs1.locals_w[0], &fs2.locals_w[0]) {
            (Some(Hlvalue::Variable(a)), Some(Hlvalue::Variable(b))) => assert_ne!(a, b),
            _ => panic!("expected two variables"),
        }
    }

    #[test]
    fn union_two_equal_states_matches_both() {
        let v = Hlvalue::Variable(Variable::new());
        let fs1 = build_state(vec![Some(v.clone())]);
        let fs2 = build_state(vec![Some(v)]);
        let fs3 = fs1.union(&fs2).expect("equal states must union");
        assert!(fs3.matches(&fs1));
        assert!(fs3.matches(&fs2));
    }

    #[test]
    fn union_generalises_to_variable_when_constant_meets_variable() {
        let fs1 = build_state(vec![Some(Hlvalue::Constant(Constant::new(
            ConstValue::Int(1),
        )))]);
        let fs2 = build_state(vec![Some(Hlvalue::Variable(Variable::new()))]);
        let fs3 = fs1.union(&fs2).expect("variable/constant must union");
        // Result should be a Variable slot.
        assert!(matches!(&fs3.locals_w[0], Some(Hlvalue::Variable(_))));
    }

    #[test]
    fn union_refuses_spectag_mismatch() {
        let fs1 = build_state(vec![Some(Hlvalue::Constant(fresh_spectag_constant()))]);
        let fs2 = build_state(vec![Some(Hlvalue::Constant(fresh_spectag_constant()))]);
        assert!(fs1.union(&fs2).is_none());
    }

    #[test]
    fn getvariables_filters_to_variables() {
        let v = Variable::new();
        let fs = build_state(vec![
            Some(Hlvalue::Variable(v.clone())),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Int(7)))),
            None,
        ]);
        let vars = fs.getvariables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], v);
    }

    #[test]
    fn getoutputargs_reads_source_at_target_variable_slots() {
        let v_src = Hlvalue::Variable(Variable::new());
        let fs1 = build_state(vec![
            Some(v_src.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        ]);
        let fs2 = build_state(vec![
            Some(Hlvalue::Variable(Variable::new())),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        ]);
        let out = fs1.getoutputargs(&fs2);
        // Only slot[0] in fs2 is a Variable, so we pick fs1's slot[0].
        assert_eq!(out, vec![Some(v_src)]);
    }

    #[test]
    fn getoutputargs_preserves_undefined_local_at_variable_target() {
        // RPython `test_framestate.py:test_getoutputargs` — when fs2's
        // slot is a fresh Variable and fs1's slot is Python `None`
        // (undefined local), upstream's `result.append(mergeable[i])`
        // appends the `None`. We mirror that via `MergeCell::None`.
        let fs1 = build_state(vec![None]);
        let fs2 = build_state(vec![Some(Hlvalue::Variable(Variable::new()))]);
        let out = fs1.getoutputargs(&fs2);
        assert_eq!(out, vec![None]);
    }

    #[test]
    fn mergeable_appends_constant_none_sentinels_when_no_exception() {
        let fs = build_state(vec![Some(Hlvalue::Variable(Variable::new()))]);
        let view = fs.mergeable_view();
        // locals_w (1) + stack (0) + exc_type (1) + exc_value (1) = 3
        assert_eq!(view.len(), 3);
        assert_eq!(
            view[1],
            Some(Hlvalue::Constant(Constant::new(ConstValue::None)))
        );
        assert_eq!(
            view[2],
            Some(Hlvalue::Constant(Constant::new(ConstValue::None)))
        );
    }

    #[test]
    fn mergeable_flattens_signal_stack_cells() {
        let v = Variable::new();
        let mut fs = FrameState::new(
            vec![Some(Hlvalue::Variable(v.clone()))],
            vec![StackElem::Signal(FlowSignal::Return {
                w_value: Hlvalue::Variable(Variable::new()),
            })],
            None,
            Vec::new(),
            0,
        );
        let view = fs.mergeable().to_vec();
        // locals_w(1) + Return-args-expanded(1) + exc sentinels(2) = 4
        assert_eq!(view.len(), 4);
    }

    #[test]
    fn recursively_flatten_no_signal_is_passthrough() {
        let v1 = Hlvalue::Variable(Variable::new());
        let v2 = Hlvalue::Constant(Constant::new(ConstValue::Int(7)));
        let flat =
            recursively_flatten(&[StackElem::Value(v1.clone()), StackElem::Value(v2.clone())]);
        assert_eq!(flat, vec![v1, v2]);
    }

    #[test]
    fn recursively_flatten_unrolls_signal_args() {
        let a = Hlvalue::Variable(Variable::new());
        let b = Hlvalue::Constant(Constant::new(ConstValue::Int(3)));
        let flat = recursively_flatten(&[StackElem::Signal(FlowSignal::Raise {
            w_exc: FSException::new(a.clone(), b.clone()),
        })]);
        assert_eq!(flat, vec![a, b]);
    }

    #[test]
    fn blocklist_participates_in_matches_assertion() {
        let block = FrameBlock::new(0, 0, FrameBlockKind::Finally);
        let v = Variable::new();
        let fs1 = FrameState::new(
            vec![Some(Hlvalue::Variable(v.clone()))],
            Vec::new(),
            None,
            vec![block.clone()],
            0,
        );
        let fs2 = FrameState::new(
            vec![Some(Hlvalue::Variable(v))],
            Vec::new(),
            None,
            vec![block],
            0,
        );
        assert!(fs1.matches(&fs2));
    }
}
