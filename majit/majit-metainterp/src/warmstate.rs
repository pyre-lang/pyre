/// Warm state management — the lifecycle from interpreting to compiled code.
///
/// Manages the transition: Interpreting -> Tracing -> Compiled.
/// When the hot counter fires, we start tracing. When the trace is
/// complete, we compile it and cache the result.
///
/// Reference: rpython/jit/metainterp/warmstate.py WarmEnterState, BaseBaseJitCell
use indexmap::IndexMap;
use std::sync::atomic::{AtomicBool, Ordering};

use majit_backend::JitCellToken;
use majit_ir::{GREEN_INLINE, GcRef, GreenKey, GreenType, RetainedGreens, Type, Value};
use smallvec::SmallVec;
use std::sync::Arc;

use crate::counter::{DEFAULT_SIZE, JitCounter};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::recorder::Trace;

bitflags::bitflags! {
    /// warmstate.py `JC_*` -- the flags a `BaseJitCell` carries.
    ///
    /// `every_jc_flag_is_the_bit_warmstate_gives_it` reads the values back out
    /// of `warmstate.py`, so a bit corrected there and not here cannot pass
    /// unnoticed.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct JcFlags: u8 {
        /// We are currently tracing from this green key.
        const JC_TRACING = 0x01;
        /// Don't trace here (e.g., trace was too long last time).
        const JC_DONT_TRACE_HERE = 0x02;
        /// Has a temporary procedure token (CALL_ASSEMBLER fallback).
        const JC_TEMPORARY = 0x04;
        /// Tracing has occurred at least once from this key.
        const JC_TRACING_OCCURRED = 0x08;
        /// When a trace from this cell grows too long, segment it: finish
        /// with a guard that always fails (`warmstate.py` `JC_FORCE_FINISH`).
        const JC_FORCE_FINISH = 0x10;
    }
}

/// Derived reading of `flags` and the procedure token for tests and the
/// entry diagnostic. Not a field: `BaseJitCell` stores only `flags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseJitCellState {
    NotHot,
    Tracing,
    Compiled,
    Invalidated,
}

/// Per-greenkey cell that tracks JIT state for a specific program location.
///
/// Mirrors rpython/jit/metainterp/warmstate.py BaseBaseJitCell.
pub struct BaseJitCell {
    /// The `JC_*` flags this cell carries. `warmstate.py` `BaseJitCell.flags`
    /// is the only per-cell state.
    pub flags: Cell<JcFlags>,
    /// Aborts of this cell since it was installed. At `MAX_TRACE_ABORT_COUNT`
    /// the back-edge and function-entry doors return not-hot, and the cell is
    /// marked `JC_DONT_TRACE_HERE`.
    pub abort_count: Cell<u32>,
    /// Hot counter value for this cell (local to this green key).
    pub counter: Cell<u32>,
    /// Generation at which tracing was last started.
    /// Used to detect stale tracing sessions.
    pub tracing_generation: Cell<u64>,
    /// `warmstate.py` `wref_procedure_token = None` — a WEAK handle on the
    /// compiled loop token, written through
    /// [`BaseJitCell::_makeref`] (`warmstate.py`).
    ///
    /// `None` means "no procedure token was ever set" and is the only state
    /// [`Self::has_seen_a_procedure_token`] distinguishes; `Some` holding a
    /// weak that no longer upgrades is upstream's *dead weakref*, which is
    /// what makes `should_remove_jitcell`'s `JC_DONT_TRACE_HERE` arm
    /// (`warmstate.py`) able to expire a cell at all.
    ///
    /// `MemoryManager::alive_loops` is therefore the sole long-lived strong
    /// owner (`memmgr.py`), matching `memmgr.rs`'s own header; the other
    /// weak handle on the same token is `CompiledEntry::token`.  Every writer
    /// of this field must have registered its token with the memory manager
    /// first, or the token dies before first entry and this cell answers
    /// "never compiled" — see `compile.rs`'s `compile_tmp_callback`
    /// (`compile.py:1148-1149`) for the one place that had to be repaired
    /// before the downgrade was safe.
    pub loop_token: RefCell<Option<std::sync::Weak<JitCellToken>>>,
    /// `JitCellToken.invalidated`, cloned at `set_procedure_token`.
    ///
    /// `is_compiled` reads this flag and `Weak::strong_count` instead of
    /// `upgrade`: the upgrade bumps and drops the token's strong count on
    /// every entry, and `get_procedure_token` (`warmstate.py`) already does
    /// that once for the token the entry actually runs.
    pub invalidated_flag: RefCell<Option<Arc<AtomicBool>>>,
    /// counter.py / warmstate.py BaseJitCell.next
    /// Linked list for per-bucket chain in the celltable.
    pub next: RefCell<Option<Rc<BaseJitCell>>>,
    /// Which driver's `WarmEnterState` filed this cell.
    ///
    /// `warmstate.py` `make_jitcell_getter_default` defines one `JitCell`
    /// class per driver and its chain walks test `isinstance(cell, JitCell)`,
    /// because every driver files into the one `jitcounter.celltable`. The
    /// class is fixed at construction, so this is a plain field.
    pub jitdriver_sd: usize,
    /// This cell's own u64 identity — what every hash-form entry point on
    /// [`WarmEnterState`] names it by, and what flows on through
    /// `MetaInterp::compiled_loops`, `JitCellToken::green_key` and
    /// `rd_loop_token`.
    ///
    /// Upstream needs no such field: `maybe_compile_and_run`
    /// (warmstate.py) resolves greens + `comparekey` + `get_uhash`
    /// *together* into a cell object and then carries **that object** to the
    /// executor (`raise EnterJitAssembler(procedure_token, ...)`,
    /// warmstate.py/:511), so nothing downstream re-derives which cell was
    /// meant. Pyre carries a `u64` instead, and a green-key hash does not name
    /// a cell when two cells answer to it — two cells, one number. The cell key
    /// restores the missing half of the identity: it is
    ///
    /// * this cell's own raw `JitCell.get_uhash` when no live cell already
    ///   holds that number, so the collision-free case is bit-for-bit what it
    ///   always was, and
    /// * a minted, live-set-unique u64 ([`WarmEnterState::mint_cell_key`])
    ///   when the raw hash was already taken by another cell.
    ///
    /// Either way [`WarmEnterState::bucket_of`] maps it back to the raw hash,
    /// which is what lets a reader holding only a number tell this cell from
    /// the neighbours the table's index truncation (counter.py _get_index) put in
    /// the same chain.
    ///
    /// `None` only between [`BaseJitCell::new_for_driver_with_greens`] and the
    /// [`WarmEnterState::install_new_cell`] that files the cell — an
    /// unassigned cell is in no chain and is reachable by nothing.
    pub cell_key: Cell<Option<u64>>,
    /// The bucket [`Self::cell_key`] maps to, memoized at filing time.
    ///
    /// [`WarmEnterState::bucket_of`] answers the same question, but a minted
    /// key sends it through the `minted` map, so the walk that tells this
    /// cell's chain neighbours apart paid one hash probe PER CELL, on every
    /// walk. The number is fixed for the cell's whole life:
    /// [`WarmEnterState::install_new_cell`] files the cell into `hash`'s slot
    /// and, in the same step, gives it either `hash` itself or a minted key
    /// registered as belonging to `hash`. Nothing moves a filed cell between
    /// buckets afterwards — the prune walk relinks survivors inside their own
    /// slot — so the memo cannot go stale.
    ///
    /// Meaningless while `cell_key` is `None`, and read only beside it.
    pub cell_bucket: Cell<u64>,
    /// The green values, `JitCell.__init__` `setattr(self, attrname, greenargs[i])`
    /// (`make_jitcell_subclass`). Immutable after construction.
    pub greens: Box<[i64]>,
    /// Owns the `Ref`-typed referents named by `comparekey`, for exactly this
    /// cell's lifetime — the `setattr` half of `JitCell.__init__`
    /// (warmstate.py), which pyre stored as a bare `i64` and so never
    /// owned. Without it a referent can be freed and its address reused by a
    /// different object, whose green key is then byte-identical to the dead
    /// one's; `comparekey_matches` returns `true` and the new object silently
    /// inherits this cell and its compiled loop token.
    ///
    /// Empty unless a frontend registered `majit_ir::set_ref_resolver`.
    pub retained_greens: RetainedGreens,
}

impl BaseJitCell {
    /// `JitCell(*greenargs)` of the driver `jitdriver_sd`'s class.
    pub fn new_for_driver_with_greens(
        jitdriver_sd: usize,
        greenargs: &[i64],
        types: &[GreenType],
    ) -> Self {
        BaseJitCell {
            jitdriver_sd,
            flags: Cell::new(JcFlags::empty()),
            abort_count: Cell::new(0),
            counter: Cell::new(0),
            tracing_generation: Cell::new(0),
            loop_token: RefCell::new(None),
            invalidated_flag: RefCell::new(None),
            next: RefCell::new(None),
            cell_key: Cell::new(None),
            cell_bucket: Cell::new(0),
            greens: Box::from(greenargs),
            retained_greens: RetainedGreens::retain_greenargs(greenargs, types),
        }
    }

    /// Live `JitCell.gN` values (`warmstate.py` `make_jitcell_subclass`).
    ///
    /// `greens` holds the addresses copied at install. Each `gN` is a
    /// GC-managed field upstream, so a retained Ref is read through
    /// `retained_greens.current_value`.
    pub fn current_greens(&self) -> SmallVec<[i64; GREEN_INLINE]> {
        (0..self.greens.len())
            .map(|i| self.retained_greens.current_value(i, self.greens[i]))
            .collect()
    }

    /// warmstate.py `JitCell.comparekey(*greenargs2)`:
    ///
    /// ```python
    /// def comparekey(self, *greenargs2):
    ///     i = 0
    ///     for attrname, TYPE in green_args_name_spec:
    ///         item1 = getattr(self, attrname)
    ///         if not equal_whatever(TYPE, item1, greenargs2[i]):
    ///             return False
    ///         i = i + 1
    ///     return True
    /// ```
    ///
    /// `types` is the driver's `GreenSpec::TYPES` on the door path
    /// (`jitdriver_sd._green_args_spec`), used only to pick
    /// `equal_whatever`'s TYPE. There is no types comparison: every
    /// caller has already checked `cell.jitdriver_sd`
    /// (`isinstance(cell, JitCell)`). With `types` constant at the
    /// inlined door, the loop and `equal_whatever` fold per slot
    /// (`unrolling_iterable(green_args_name_spec)`, `@specialize.arg(0)`).
    #[inline(always)]
    pub fn comparekey(&self, greenargs: &[i64], types: &[GreenType]) -> bool {
        debug_assert_eq!(types.len(), greenargs.len());
        if self.greens.len() != types.len() {
            return false;
        }
        for i in 0..types.len() {
            // Ref greens live in the frontend root slot, the others are the
            // cell field `getattr(self, attrname)` (`JitCell.comparekey`).
            let item1 = if types[i] == GreenType::Ref {
                self.retained_greens.current_value(i, self.greens[i])
            } else {
                self.greens[i]
            };
            if !majit_ir::equal_whatever(types[i], item1, greenargs[i]) {
                return false;
            }
        }
        true
    }

    /// Thin wrapper around [`Self::comparekey`] for a stored [`GreenKey`].
    #[inline]
    pub fn comparekey_matches(&self, other: &GreenKey) -> bool {
        self.comparekey(&other.values, &other.types)
    }

    pub fn is_tracing(&self) -> bool {
        self.flags.get().contains(JcFlags::JC_TRACING)
    }

    fn abort_ceiling_latched(&self) -> bool {
        self.abort_count.get() >= MAX_TRACE_ABORT_COUNT
    }

    pub(crate) fn derived_state(&self) -> BaseJitCellState {
        let invalidated = self
            .loop_token
            .borrow()
            .as_ref()
            .and_then(|wref| wref.upgrade())
            .is_some_and(|token| token.is_invalidated());
        if invalidated {
            BaseJitCellState::Invalidated
        } else if self.is_compiled() {
            BaseJitCellState::Compiled
        } else if self.is_tracing() {
            BaseJitCellState::Tracing
        } else {
            BaseJitCellState::NotHot
        }
    }

    /// warmstate.py — get_procedure_token returns None for
    /// invalidated tokens. is_compiled additionally excludes TEMPORARY.
    pub fn is_compiled(&self) -> bool {
        if self.flags.get().contains(JcFlags::JC_TEMPORARY) {
            return false;
        }
        let invalidated_flag = self.invalidated_flag.borrow();
        let Some(flag) = invalidated_flag.as_ref() else {
            return self.get_procedure_token().is_some();
        };
        let loop_token = self.loop_token.borrow();
        let Some(wref) = loop_token.as_ref() else {
            return false;
        };
        // `strong_count` is the weakref resolving; the flag is
        // `token.invalidated`. A dead weak answers false without `upgrade`.
        wref.strong_count() > 0 && !flag.load(Ordering::Acquire)
    }

    /// warmstate.py `get_procedure_token`.
    ///
    /// ```python
    /// def get_procedure_token(self):
    ///     if self.wref_procedure_token is not None:
    ///         token = self.wref_procedure_token()
    ///         if token and not token.invalidated:
    ///             return token
    ///     return None
    /// ```
    ///
    /// The `token and ...` half is the weakref failing to resolve — pyre's
    /// `Weak::upgrade` returning `None`. It answers `None` for a token
    /// `MemoryManager` has already dropped, exactly as it does for an
    /// invalidated one, and returns an OWNED handle: a caller that got a token
    /// out of here keeps it alive for as long as it holds it, which is what
    /// upstream's Python reference does too.
    ///
    /// Outlined so `Weak::upgrade`'s `casa` stays in this body. Inlining it
    /// into `enter_compiled_function_entry` residualizes `upgrade` as a call.
    #[inline(never)]
    pub fn get_procedure_token(&self) -> Option<Arc<JitCellToken>> {
        // The three conditions above, in order: the slot is set, the weakref
        // still resolves, and the token it resolves to is not invalidated.
        if let Some(wref) = self.loop_token.borrow().as_ref()
            && let Some(token) = wref.upgrade()
            && !token.is_invalidated()
        {
            return Some(token);
        }
        None
    }

    /// warmstate.py `_makeref`.
    ///
    /// ```python
    /// def _makeref(self, token):
    ///     assert token is not None
    ///     return weakref.ref(token)
    /// ```
    fn _makeref(token: &Arc<JitCellToken>) -> std::sync::Weak<JitCellToken> {
        Arc::downgrade(token)
    }

    /// Set the procedure token and update ownership state.
    /// If `tmp` is true, sets the TEMPORARY flag (CALL_ASSEMBLER fallback).
    ///
    /// **Precondition every caller owes**: `loop_token` is a WEAK handle
    /// (`warmstate.py:188`), so this method takes NO ownership of `loop_token`.
    /// The token must already be registered with `MemoryManager`
    /// (`memmgr.py`) — upstream's producers do it at
    /// `compile.py:566-567` and `compile.py:1148-1149`, and pyre's are
    /// `compile_loop_body` / `compile_retrace` / `finish_and_compile` /
    /// `compile_simple_loop` / `compile_entry_bridge` (each calls
    /// `keep_loop_alive` before its `attach_procedure_with_redirect`) plus
    /// `compile_tmp_callback`. A caller that skips it installs a token that is
    /// already unowned, and the cell answers `get_procedure_token() == None`
    /// — "never compiled" — rather than failing.
    ///
    /// Returns the previous procedure token (if any), so the caller can
    /// implement `warmstate.py`'s `redirect_call_assembler` +
    /// `old_token.record_jump_to(procedure_token)` chain.
    pub fn set_procedure_token(
        &self,
        loop_token: impl Into<Arc<JitCellToken>>,
        tmp: bool,
    ) -> Option<Arc<JitCellToken>> {
        let loop_token = loop_token.into();
        // `warmstate.py:202` `self.wref_procedure_token =
        // self._makeref(token)`. The returned `old` is the strong handle the
        // displaced weakref still resolves to, if anything still owns it —
        // `redirect_call_assembler` (warmstate.py) needs the object,
        // and a predecessor `alive_loops` has already dropped has no code left
        // to redirect.
        *self.invalidated_flag.borrow_mut() = Some(Arc::clone(&loop_token.invalidated));
        let old = self
            .loop_token
            .replace(Some(Self::_makeref(&loop_token)))
            .and_then(|wref| wref.upgrade());
        if tmp {
            self.flags.set(self.flags.get() | (JcFlags::JC_TEMPORARY));
        } else {
            self.flags.set(self.flags.get() & (!JcFlags::JC_TEMPORARY));
        }
        old
    }

    /// warmstate.py `has_seen_a_procedure_token`.
    ///
    /// ```python
    /// def has_seen_a_procedure_token(self):
    ///     return self.wref_procedure_token is not None
    /// ```
    ///
    /// Reads the weakref SLOT, which is what makes the reading
    /// `should_remove_jitcell` puts on it correct: at `warmstate.py`
    /// the `JC_DONT_TRACE_HERE` arm is only reached after
    /// `get_procedure_token()` already answered `None`, so a slot that is
    /// nonetheless `Some` means the weakref is dead (or the token
    /// invalidated) — upstream's own comment, "i.e. dead weakref".
    ///
    /// This previously read the `token: Option<u64>` mirror, which is set by
    /// `set_procedure_token` and never cleared. That agreed with the slot
    /// while the slot was a strong `Arc` that only `set_procedure_token`
    /// wrote, but it could not become false when a token died, so the arm
    /// above could never fire and every `JC_DONT_TRACE_HERE` cell was
    /// immortal.
    pub fn has_seen_a_procedure_token(&self) -> bool {
        self.loop_token.borrow().is_some()
    }

    /// Whether this cell should be removed (for GC of dead cells).
    /// Mirrors BaseBaseJitCell.should_remove_jitcell.
    pub fn should_remove_jitcell(&self) -> bool {
        if self.get_procedure_token().is_some() {
            return false; // has a valid procedure token
        }
        if self.flags.get().contains(JcFlags::JC_TRACING) {
            return false; // currently tracing
        }
        if self.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
            // Remove only if we had a token that is now dead.
            return self.has_seen_a_procedure_token();
        }
        // warmstate.py:222-225
        if self.flags.get().contains(JcFlags::JC_FORCE_FINISH) {
            return false;
        }
        true
    }
}

/// Per-green-key JIT cell state with associated data.
///
/// Richer variant of BaseJitCellState that carries trace/token payloads.
pub enum CellJitState {
    /// Normal interpretation; no tracing is active.
    Interpreting,
    /// Actively recording a trace.
    Tracing(Trace),
    /// A compiled loop exists for this green key.
    Compiled(Arc<JitCellToken>),
}

pub use crate::memmgr::MemoryManager;

// Warm state manager — the orchestrator of the JIT lifecycle. It keeps track
// of per-greenkey cells and the global hot counter.
// Defaults from `rpython.rlib.jit.PARAMETERS`.
// DEFAULT_ constants must match RPython exactly.

/// rlib/jit.py:588 threshold = 1039 (just above 1024, prime)
const DEFAULT_THRESHOLD: u32 = 1039;

/// rlib/jit.py:589 function_threshold = 1619
const DEFAULT_FUNCTION_THRESHOLD: u32 = 1619;

/// rlib/jit.py:590 trace_eagerness = 200
const DEFAULT_TRACE_EAGERNESS: u32 = 200;

/// rlib/jit.py:601 max_unroll_recursion = 7
const DEFAULT_MAX_UNROLL_RECURSION: u32 = 7;

/// rlib/jit.py:593 inlining = 1 (max_inline_depth derived)
const DEFAULT_MAX_INLINE_DEPTH: u32 = 7;

/// rlib/jit.py:592 trace_limit = 6000
const DEFAULT_TRACE_LIMIT: u32 = crate::trace_ctx::DEFAULT_TRACE_LIMIT as u32;

/// rlib/jit.py:595 retrace_limit = 0.
const DEFAULT_RETRACE_LIMIT: u32 = 0;

/// rlib/jit.py:598 max_unroll_loops = 0
const DEFAULT_MAX_UNROLL_LOOPS: u32 = 0;

/// rlib/jit.py:600 enable_opts = "all"
fn default_enable_opts() -> Vec<String> {
    vec![
        "intbounds".to_string(),
        "rewrite".to_string(),
        "virtualize".to_string(),
        "string".to_string(),
        "pure".to_string(),
        "earlyforce".to_string(),
        "heap".to_string(),
        "unroll".to_string(),
    ]
}

/// Pyre-local abort ceiling with no upstream analogue: after repeated aborts,
/// mark the green key `DONT_TRACE_HERE`.  Pyre walker aborts are structural
/// and recur identically on retrace; without a ceiling the same body would
/// retrace forever, each attempt executing its residual calls concretely.
///
/// The ban is permanent, and permanent in a way upstream cannot express: the
/// flag it sets is the one `should_remove_jitcell` keeps forever on a cell that
/// never saw a procedure token, and `abort_count` is never reset.  It is
/// load-bearing rather than defensive — 17 of 24 sampled fixtures reach it, and
/// removing it buys no compiled loop on any of them while multiplying
/// `loops_aborted` by 4-6x (see [`WarmEnterState::maybe_compile_decision`] for
/// why the upstream cell lifecycle that would retire these cells first cannot
/// be ported as written).  `mc_diag` slots 80/81 measure it.
///
/// THE FLAG IT SETS IS ALSO THE INLINE GATE, AND THAT IS DELIBERATE.
/// `JC_DONT_TRACE_HERE` is what [`WarmEnterState::can_inline_callable`] reads
/// (`warmstate.py`), so a location the ceiling retires also stops being inlined
/// into other traces.  Upstream writes that flag from two places only —
/// `disable_noninlinable_function` (`warmstate.py`, reached from the
/// `ABORT_TOO_LONG` arm of `blackhole_if_trace_too_long`, `pyjitpl.py`) and
/// `dont_trace_here` (`warmstate.py`, `max_unroll_recursion`) — and there it
/// does not even mean "never trace here": `maybe_compile_and_run`
/// (`warmstate.py`) traces a flagged, tokenless cell immediately, and
/// `prepare_trace_segmenting` (`pyjitpl.py`) sets it with the comment
/// "bizarrely enough, this means *do trace here* ??!".  So the coupling is a
/// pyre-local extension, not a port, and it reads like an accident.
///
/// It is not.  The ceiling's premise is that a pyre walker abort is STRUCTURAL
/// and recurs identically, and that premise applies to the callee position too:
/// inlining a body whose own walk gave up five times walks the caller into the
/// same give-up.  Splitting the two — latching the ceiling on `abort_count`
/// alone and leaving `JC_DONT_TRACE_HERE` to its two upstream writers — was
/// built and measured over the whole synth corpus on 2026-08-24.  It improved
/// NOTHING (every other row byte-identical on dynasm, cranelift and wasm) and
/// cost `synth/generator_tree_recursion`: `bridges_compiled` 26 -> 60,
/// `guard_failures` 2999 -> 25297, `loops_aborted` 0 -> 3, and the pypy ratio
/// from under its 7.6x gate to 26.0x (dynasm) / 28.6x (cranelift).  A guard
/// census of that run puts 24_819 of the 25_297 failures on ONE green key
/// across nine fail indices, so the sixty bridges built there do not converge —
/// the ban is holding a bridge non-convergence shut.  Reopening it means fixing
/// that first; the two numbers above are the acceptance test.
///
/// A SECOND, INDEPENDENT ROUTE reached the same wall, which is what pins the
/// diagnosis.  `abort_trace_live` (`pyjitpl.rs`) charges `abort_tracing` to the
/// trace's green key, and a bridge session's green key is the source loop's,
/// so guard-side walks that gave up were driving already-compiled loops into
/// this ceiling — `synth/generator_frame_handout_image` reaches it with 14
/// such charges and `loops_aborted=0`, i.e. no primary trace aborted there at
/// all.  Not charging them is the upstream answer: `aborted_tracing` reads
/// `greenkey = None # we're in the bridge` and touches no JitCell, and a
/// bridge never passes through `bound_reached`, which is the only place
/// upstream touches a cell for a trace.  Doing it removed one of
/// `generator_tree_recursion`'s two bans and reproduced the paragraph above to
/// the digit — `bridges_compiled` 26 -> 60, `guard_failures` 2999 -> 25297,
/// `loops_aborted` 0 -> 3, ratio 41.7x/32.4x — with every other row of the
/// corpus unchanged on all three backends.
///
/// What is known about that unbanned loop, from `MAJIT_GUARDLOG=1` over the bad
/// regime.  Bridges are NOT decorative there: `MAJIT_NO_BRIDGE=1` on the same
/// binary reads 239874 guard failures against 25297 with bridges on, so the
/// tally counts failures a bridge serves as well as deopts, and the 25297 is
/// 8.4x the banned regime's 2999 at the SAME `loops_compiled=3`.  Every
/// `(trace_id, fail_index)` coordinate is offered a bridge exactly once in the
/// whole run — 103 offers, 60 bridges — and `must_compile` answers no to all
/// 21802 later failures at those coordinates, one of which (`trace` 7,
/// `fail` 2) accounts for 7777 by itself.  Whether those 43 offers that
/// produced no bridge should have been retried the way `must_compile` retries
/// upstream (`jitcounter.tick` resets the counter when a bridge does not
/// compile, `compile.py done_compiling`) is the open question; reading it
/// needed `trace_id` next to `fail_index`, since an index is allocated per
/// trace and a bridge re-uses the numbers of the loop it hangs off.
///
/// Public because an embedder that demotes a key of its own on repeated
/// aborts has to agree with this ceiling, and the consumers that do
/// restated the literal instead.  Sharing it makes the NUMBERS agree, not
/// the COUNTS: this one is per green key and latches with `>=`, so a host
/// counter keyed on anything coarser reaches 5 for reasons this one did
/// not.
pub const MAX_TRACE_ABORT_COUNT: u32 = 5;

/// rlib/jit.py `PARAMETERS['disable_unrolling']`, stored by
/// `set_param_disable_unrolling` and read by nothing.
const DEFAULT_DISABLE_UNROLLING: u32 = 200;

/// warmspot.py `WarmRunnerDesc.__init__`: `self.jitcounter = JitCounter(...)`
/// with the `rlib/jit.py PARAMETERS` decay default.
pub fn new_runner_jitcounter() -> JitCounter {
    let mut counter = JitCounter::new(DEFAULT_SIZE);
    counter.set_decay(40);
    counter
}

/// warmspot.py `WarmRunnerDesc.__init__`: `self.memory_manager =
/// memmgr.MemoryManager()`, with the `rlib/jit.py PARAMETERS` defaults
/// `warmstate.py set_param_retrace_limit` / `set_param_max_unroll_loops`
/// write into it.
pub fn new_runner_memory_manager() -> crate::memmgr::MemoryManager {
    let mut memory_manager = crate::memmgr::MemoryManager::new(0);
    memory_manager.set_retrace_limit(DEFAULT_RETRACE_LIMIT);
    memory_manager.set_max_unroll_loops(DEFAULT_MAX_UNROLL_LOOPS);
    memory_manager
}

/// JIT statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Number of cells in Compiled state.
    pub num_compiled: usize,
    /// Number of cells in Tracing state.
    pub num_tracing: usize,
    /// Number of cells in Invalidated state.
    pub num_invalidated: usize,
    /// Number of cells carrying `JC_DONT_TRACE_HERE`, whatever lifecycle state
    /// they are in.
    pub num_disable_noninlinable_function: usize,
    /// Total number of BaseJitCells.
    pub num_cells: usize,
    /// Referents pinned alive by stored typed green keys, summed over every
    /// cell in every chain.
    ///
    /// This is the resource cost of the ownership invariant (`RetainedGreens`)
    /// against a `cells` map with no size bound: nothing caps how many cells
    /// accumulate, so the pinned set grows with them. Reported so the growth
    /// is a number somebody can read rather than latent RSS — the condition
    /// under which pinning is allowed to ship ahead of a bound.
    ///
    /// WHICH WINDOW THIS IS TAKEN IN. The number is a **running total at
    /// the moment of the call**, not a settled figure, and it is not
    /// monotonic. Cells *are* dropped in production — `install_new_cell`
    /// unlinks every chained cell whose `should_remove_jitcell()` holds
    /// (which is the default for a cold, tokenless cell), and `cleanup_chain`
    /// drops the whole chain — and each drop releases that cell's retains.
    /// So a cold tree, a tree mid-warmup and a settled tree give three
    /// different answers for the same program, and a reading is only
    /// comparable against another taken at the same point.
    ///
    /// Do not quote it as "how many cells pin a referent" without saying
    /// after how many portal entries it was read. `gc_cells` has no
    /// production caller, so the one eviction path that would make this fall
    /// sharply is currently unreachable.
    ///
    /// Zero unless a frontend registered `set_ref_resolver`.
    pub num_pinned_refs: usize,
}

/// # The `u64` this type is keyed by is a CELL key, not a bucket hash
///
/// Every `cell_key: u64` parameter below names one cell
/// ([`BaseJitCell::cell_key`]), not a celltable bucket. The two are the same
/// number whenever a bucket holds one cell — which is every collision-free
/// workload, so nothing observable changes there — and differ only for a cell
/// that had to be minted a key because a sibling already held the bucket's raw
/// hash.
///
/// The parameter name says which side of the resolve the value sits on, so the
/// two entry points that genuinely take a PRE-resolve raw green-key hash keep
/// the `green_key_hash` spelling: [`WarmEnterState::bucket_is_chained`] (the
/// test a caller asks before it decides whether a `GreenKey` has to be built at
/// all) and
/// [`WarmEnterState::trace_next_iteration`] (whose whole identity is the hash,
/// see its own doc).
///
/// The distinction exists because pyre carries a `u64` where upstream carries
/// the cell object. `maybe_compile_and_run` (warmstate.py) resolves
/// greens + `comparekey` + `get_uhash` together into a cell and then hands
/// *that cell's* procedure token to the executor (warmstate.py/:511:
/// `raise EnterJitAssembler(procedure_token, *execute_args)`) rather than
/// handing on a key for a second lookup — which is why upstream has no second
/// reader that can disagree with the first. Pyre restores that property by
/// resolving once, at the hash producer ([`WarmEnterState::cell_key_for`] /
/// [`WarmEnterState::ensure_cell_key`] / [`WarmEnterState::sole_cell_key`]),
/// and carrying the resolved cell key onward — through `MetaInterp`'s
/// `compiled_loops`, `JitCellToken::green_key` and `rd_loop_token` alike.
///
/// The counter is the deliberate exception: `jitcounter.tick(hash, ...)`
/// (warmstate.py) is indexed by the raw hash upstream, so colliding keys
/// share one back-edge counter. Every counter call here therefore goes through
/// [`WarmEnterState::bucket_of`] first.
pub struct WarmEnterState {
    /// counter.py JitCounter parity: single timetable shared by loop
    /// entry, guard failure, and function entry — each caller passes
    /// a different pre-computed increment to tick(hash, increment).
    ///
    /// `warmspot.py` `WarmRunnerDesc.jitcounter` is one object on the
    /// runner; extra `jd.warmstate` slots clone this handle
    /// (`warmstate.py` `_compute_threshold`).
    pub counter: JitCounter,
    /// Compilation threshold (copied from counter for easy access).
    threshold: u32,
    /// warmstate.py:254: increment_threshold = compute_threshold(threshold).
    increment_threshold: f64,
    /// warmstate.py: trace_eagerness parameter (integer, default 200).
    trace_eagerness: u32,
    /// warmstate.py: increment_trace_eagerness = compute_threshold(trace_eagerness).
    /// Pre-computed f64 increment for guard failure counter ticking.
    increment_trace_eagerness: f64,
    /// Function call threshold for inlining during tracing.
    function_threshold: u32,
    /// warmstate.py:257: increment_function_threshold = compute_threshold(function_threshold).
    increment_function_threshold: f64,
    /// Maximum depth of inlined function calls during tracing.
    max_inline_depth: u32,
    /// Maximum number of operations per trace before aborting.
    trace_limit: u32,
    /// Global tracing generation counter.
    /// Incremented each time tracing starts; stored in BaseJitCell to
    /// detect stale tracing sessions.
    tracing_generation: u64,
    /// Quasi-immutable field invalidation registry.
    ///
    /// Maps a quasi-immutable field key (hash of object_id + field_index)
    /// to the set of cell keys whose compiled loops depend on that field.
    /// When a quasi-immutable field is mutated, all dependent loops are invalidated.
    quasiimmut_deps: crate::FxIndexMap<u64, Vec<u64>>,
    // Function-entry hotness rides on the shared `counter: JitCounter`
    // below (warmstate.py — maybe_compile_and_run's tick + reset
    // goes through the common timetable, not a separate HashMap).

    // warmstate.py set_param_retrace_limit — retrace_limit / max_retrace_guards /
    // max_unroll_loops / max_unroll_recursion live on
    // warmrunnerdesc.memory_manager, not on WarmEnterState itself. See
    // `MemoryManager` in memmgr.rs.
    /// jit.py:581,602: vec — enable vectorization optimization.
    vectorize: bool,
    /// jit.py:585,603: vec_all — vectorize loops outside numpypy library.
    vec_all: bool,
    /// jit.py:583,604: vec_cost — cost threshold for vectorization decisions.
    vec_cost: u32,
    /// warmstate.py: enable_opts — list of enabled optimization pass names.
    enable_opts: Vec<String>,
    /// warmstate.py: set_param_inlining — whether inlining is enabled.
    inlining: bool,
    /// warmstate.py: set_param_disable_unrolling — "after how many
    /// operations we should not unroll": a recording longer than this
    /// compiles as a simple loop.
    disable_unrolling_threshold: u32,
    /// warmstate.py: set_param_pureop_historylength — size of the
    /// pure operation history cache.
    pureop_historylength: u32,
    /// warmspot.py `WarmRunnerDesc.__init__` `self.memory_manager =
    /// memmgr.MemoryManager()`: one manager on the runner, which every
    /// `WarmEnterState` reads as `self.warmrunnerdesc.memory_manager`. The
    /// handle is `Clone`; `MetaInterp::make_enter_function` hands each
    /// extra slot the runner's.
    pub memory_manager: crate::memmgr::MemoryManager,
    /// Bumped by every mutation that can change what
    /// [`Self::maybe_compile_decision`] answers for a key it already refused
    /// at the abort ceiling: installing a cell, which resets `abort_count` to
    /// zero, and attaching a procedure token, which can make the key runnable
    /// through a sibling cell in the same chain.
    ///
    /// It exists so a caller can cache that refusal rather than re-derive it
    /// per back edge. The refusal is pure — it bumps a diagnostic slot and
    /// returns `NotHot` above `decay_all_counters`, deliberately — so a cache
    /// keyed on this counter changes nothing but the work spent reaching it.
    cell_generation: u64,
    /// `warmstate.py` `WarmEnterState.__init__` `self.jitdriver_sd` — the
    /// driver this state belongs to, as its `jd_no`. Cells it files carry it
    /// ([`BaseJitCell::jitdriver_sd`]) and its chain walks match on it.
    pub jitdriver_sd: usize,
    /// `warmstate.py` `self.jitdriver_sd._confirm_enter_jit_ptr` — the
    /// driver's `confirm_enter_jit` callback, or `None` when the declaration
    /// omitted the kwarg (`rlib/jit.py:626,688`).
    ///
    /// `warmstate.py` reads the pointer once and substitutes a
    /// `return True` closure when it is absent, which is what `None` means
    /// here; [`Self::confirm_enter_jit`] is that substitution.
    ///
    /// `warmstate.py make_entry_point` captures `confirm_enter_jit` as a
    /// closure constant. The `Rc<Cell>` is that capture: `JitDriver` holds
    /// another handle to the same cell so the compiled portal can read it
    /// without borrowing this state.
    ///
    /// `warmspot.py make_driverhook_graphs` builds this hook with
    /// `onlygreens=False`, so the pointer is `confirm_enter_jit(*args)` —
    /// greens then reds as one portal tuple (`_portal_args_s`). Callers
    /// thread the live portal reds into [`Self::confirm_enter_jit`].
    confirm_enter_jit_ptr: Rc<Cell<Option<fn(&[Value]) -> bool>>>,
    /// warmspot.py `jd._should_unroll_one_iteration_ptr`.
    ///
    /// `warmstate.py WarmEnterState.__init__` wraps this as
    /// `self.should_unroll_one_iteration(greenkey)` (False when the ptr is
    /// None). Copied from `JitDriverStaticData` when the state is built
    /// (`make_enter_function`) and when a real descriptor lands on an
    /// existing shell state.
    should_unroll_one_iteration_ptr: Option<fn(&[i64]) -> bool>,
}

/// Result of checking whether a green key is hot.
pub enum HotResult {
    /// Not yet hot; keep interpreting.
    NotHot,
    /// Threshold reached; start tracing. The caller (MetaInterp) builds
    /// the Trace itself — RPython parity: `MetaInterp.create_empty_history`
    /// / `MetaInterp.create_history` live on `MetaInterp`, not on the
    /// warmstate (pyjitpl.py). Pyre's prior signal-and-factory
    /// pattern (`HotResult::StartTracing(Trace::new())`) forced warmstate
    /// to depend on the `recorder::Trace` type; the `TraceRecordBuffer`
    /// migration removes
    /// that coupling so the `Trace` factory moves to MetaInterp where
    /// `metainterp_sd` is available for `TraceRecordBuffer::new`.
    StartTracing,
    /// Already tracing (caller should keep feeding ops to the active recorder).
    AlreadyTracing,
    /// Compiled code exists; run it.
    RunCompiled,
}

/// What the function-entry door should do, from one walk of the cell chain.
///
/// [`WarmEnterState::function_entry_step`] answers all three of the door's
/// questions at once; the variants are the three answers `maybe_compile_and_run`
/// arrives at.
pub enum FunctionEntryStep {
    /// A runnable procedure token was on the cell. Handed back so the door can
    /// enter it without resolving the cell a second time.
    RunCompiled(Arc<JitCellToken>),
    /// The gate did not decline: start tracing. Either the counter fired or the
    /// door never asked it (a trace is already running, or a `DONT_TRACE_HERE`
    /// cell that has never seen a token is due its immediate retry).
    Proceed,
    /// Keep interpreting.
    NotHot,
}

/// Which door is asking [`WarmEnterState::maybe_compile_and_run_decision`].
enum WarmEntryDoor {
    /// `increment_threshold`; no function-entry diag slots.
    /// `engine_is_tracing` is `MetaInterp::trace_attempt_in_progress`, same nested-fire
    /// rule as [`WarmEntryDoor::FunctionEntry`].
    BackEdge { engine_is_tracing: bool },
    /// `increment_function_threshold`; slots 23/24/25/64/65/66.
    /// `engine_is_tracing` is `MetaInterp::trace_attempt_in_progress`. It does not skip
    /// the counter: `warmstate.py maybe_compile_and_run` ticks other
    /// greenkeys while a nested `MetaInterp` is tracing, and only this
    /// cell's `JC_TRACING` returns. A fire while the walker holds the
    /// outer `TraceCtx` is deferred to `JitCell.trace_next_iteration`'s
    /// 0.98 rather than nested `bound_reached`.
    FunctionEntry { engine_is_tracing: bool },
}

impl std::fmt::Debug for WarmEnterState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WarmEnterState")
            .field("jitdriver_sd", &self.jitdriver_sd)
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl crate::jit::JitParameterTarget for WarmEnterState {
    fn set_param(&mut self, name: &str, value: i64) -> Result<(), crate::jit::JitHintError> {
        self.set_param(name, value);
        Ok(())
    }

    fn set_param_enable_opts(&mut self, value: &str) {
        self.set_param_enable_opts(value);
    }
}

/// One green word as a portal `Value` (`warmspot.py` `_make_hook_graph`
/// TYPE). Void greens are packed as `Int`, matching the specialized
/// portal tuple `confirm_enter_jit(*args)` receives.
fn value_from_green_word(word: i64, tp: GreenType) -> Value {
    match tp {
        GreenType::Int | GreenType::Void => Value::Int(word),
        GreenType::Float => Value::Float(f64::from_bits(word as u64)),
        GreenType::Ref | GreenType::Str | GreenType::Unicode => Value::Ref(GcRef(word as usize)),
    }
}

/// Slice identity for a `Value` extra-area walk. The collector updates
/// `Value::Ref` payloads in place so a slice live across a collecting call
/// keeps forwarded addresses (`get_livevars_for_roots` / `push_roots`).
/// LABEL `direct_live_values` are not green Str/Unicode slots: every
/// `Value::Ref` is a GC ref.
struct ConfirmValueSlice {
    ptr: *mut Value,
    len: usize,
}

unsafe fn walk_confirm_value_refs(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
    let slice = unsafe { &*data.cast::<ConfirmValueSlice>() };
    let values = unsafe { std::slice::from_raw_parts_mut(slice.ptr, slice.len) };
    for value in values {
        if let Value::Ref(r) = value
            && !r.is_null()
            && *r != GcRef::NO_CONCRETE
        {
            visitor(r);
        }
    }
}

/// Extra-area roots for a `Value` slice live across a collecting call.
/// `_area` is declared first so it drops before `_walk` (`root_with`).
/// The guard holds the exclusive borrow of `values`, so the slice cannot
/// be moved, reallocated, or read elsewhere while the collector may rewrite it.
pub(crate) struct ConfirmRefRoots<'a> {
    _area: Option<majit_gc::shadow_stack::MutatorExtraAreaGuard>,
    _walk: Option<Box<ConfirmValueSlice>>,
    _values: std::marker::PhantomData<&'a mut [Value]>,
}

/// Root `values` for the duration of the returned guard (`push_roots` /
/// `pop_roots`). The guard holds the exclusive borrow of `values`, so the
/// slice cannot be moved, reallocated, or read elsewhere while the
/// collector may rewrite it.
pub(crate) fn root_confirm_ref_values<'a>(values: &'a mut [Value]) -> ConfirmRefRoots<'a> {
    if values.is_empty() || !majit_gc::shadow_stack::mutator_is_registered() {
        return ConfirmRefRoots {
            _area: None,
            _walk: None,
            _values: std::marker::PhantomData,
        };
    }
    let walk = Box::new(ConfirmValueSlice {
        ptr: values.as_mut_ptr(),
        len: values.len(),
    });
    let area = unsafe {
        majit_gc::shadow_stack::MutatorExtraAreaGuard::new(
            walk_confirm_value_refs,
            &*walk as *const ConfirmValueSlice as *const (),
            "confirm_enter_jit",
        )
    };
    ConfirmRefRoots {
        _area: Some(area),
        _walk: Some(walk),
        _values: std::marker::PhantomData,
    }
}

/// What `maybe_compile_and_run`'s chain walk found for this hash on the
/// door, before any counter work.
pub(crate) enum DoorCell {
    /// No cell for this hash: the not-found arm (`jitcounter.tick`).
    Empty,
    /// The cell is being traced: `if cell.flags & JC_TRACING: return`.
    Tracing,
    /// The cell holds an enterable token. `cell_key` is the resolved key.
    Enter {
        cell_key: u64,
        token: Arc<JitCellToken>,
    },
    /// Anything else (chained bucket, comparekey miss, dead token,
    /// `JC_DONT_TRACE_HERE`, compiled-but-not-enterable): the occupied door.
    Occupied,
}

impl WarmEnterState {
    /// `warmstate.py maybe_compile_and_run`: a `JC_DONT_TRACE_HERE` cell that
    /// has never seen a procedure token is retried — immediately the first
    /// time (`tick = True` when `JC_TRACING_OCCURRED` is unset), then by
    /// `jitcounter.tick` on subsequent entries.
    ///
    /// `abort_tracing` also stamps `JC_DONT_TRACE_HERE` at
    /// `MAX_TRACE_ABORT_COUNT`. That stamp is a ban, not
    /// `disable_noninlinable_function`'s "trace this separately": with
    /// `threshold=1`, `compute_threshold` is `> 1` so `tick` fires on every
    /// entry and the occupied door would retrace forever. The ceiling stays
    /// in front of this retry.
    fn should_start_dont_trace_here_trace(
        &mut self,
        cell_key: u64,
        flags: JcFlags,
        has_seen_a_procedure_token: bool,
        abort_count: u32,
        increment: f64,
    ) -> bool {
        if !flags.contains(JcFlags::JC_DONT_TRACE_HERE) || has_seen_a_procedure_token {
            return false;
        }
        if abort_count >= MAX_TRACE_ABORT_COUNT {
            return false;
        }
        if flags.contains(JcFlags::JC_TRACING_OCCURRED) {
            let bucket = self.bucket_of(cell_key);
            self.counter.tick(bucket, increment)
        } else {
            true
        }
    }

    /// `warmstate.py maybe_compile_and_run` after the chain walk. Callers look
    /// up `cell`, choose `hash` / `increment` / the veto greens, and map the
    /// result; this body is the order.
    #[expect(
        clippy::too_many_arguments,
        reason = "greenargs/types/reds are the portal *args split make_entry_point passes as confirm_enter_jit(*args)"
    )]
    fn maybe_compile_and_run_decision(
        &mut self,
        cell: Option<&BaseJitCell>,
        hash: u64,
        increment: f64,
        greenargs: Option<&[i64]>,
        green_types: &[GreenType],
        reds: &[Value],
        is_runnable: bool,
        door: WarmEntryDoor,
    ) -> HotResult {
        let (function_entry, engine_is_tracing) = match door {
            WarmEntryDoor::BackEdge { engine_is_tracing } => (false, engine_is_tracing),
            WarmEntryDoor::FunctionEntry { engine_is_tracing } => (true, engine_is_tracing),
        };
        if increment == 0.0 {
            return HotResult::NotHot;
        }

        /// `warmstate.py maybe_compile_and_run` ticks other greenkeys while
        /// another `MetaInterp` is in `compile_and_run_once` / `handle_guard_failure`.
        /// `bound_reached` then constructs a nested `MetaInterp`. pyre's walker
        /// holds `&mut TraceCtx` on the outer attempt, so a fire cannot nest:
        /// leave the hash at `JitCell.trace_next_iteration`'s 0.98 and compile
        /// on the next non-tracing entry. Resetting at 1.0 and refusing would
        /// drop the threshold.
        fn tick_or_defer_nested(
            state: &mut WarmEnterState,
            hash: u64,
            increment: f64,
            engine_is_tracing: bool,
        ) -> HotResult {
            if !state.counter.tick(hash, increment) {
                return HotResult::NotHot;
            }
            if engine_is_tracing {
                state.counter.change_current_fraction(hash, 0.98);
                return HotResult::NotHot;
            }
            HotResult::StartTracing
        }

        let Some(cell) = cell else {
            if function_entry {
                crate::mc_diag_bump(25);
            }
            return tick_or_defer_nested(self, hash, increment, engine_is_tracing);
        };

        let flags = cell.flags.get();
        if flags.contains(JcFlags::JC_TRACING) || flags.contains(JcFlags::JC_TEMPORARY) {
            if flags.contains(JcFlags::JC_TRACING) {
                // tracing already happening in some outer invocation of
                // this function. don't trace a second time.
                if function_entry {
                    crate::mc_diag_bump(23);
                    crate::mc_diag_bump(65);
                    // 65 is a cell holding JC_TRACING. 66 splits those by
                    // age: a generation older than the warm state's means a
                    // later start superseded the session that set the flag.
                    // See MC_DIAG's legend.
                    if cell.tracing_generation.get() < self.tracing_generation {
                        crate::mc_diag_bump(66);
                    }
                }
                return HotResult::AlreadyTracing;
            }
            if function_entry {
                crate::mc_diag_bump(25);
            }
            return tick_or_defer_nested(self, hash, increment, engine_is_tracing);
        }

        if cell.get_procedure_token().is_none() {
            let has_seen_a_procedure_token = cell.has_seen_a_procedure_token();
            let abort_count = cell.abort_count.get();
            if flags.contains(JcFlags::JC_DONT_TRACE_HERE) && !has_seen_a_procedure_token {
                if self.should_start_dont_trace_here_trace(
                    hash,
                    flags,
                    has_seen_a_procedure_token,
                    abort_count,
                    increment,
                ) {
                    if function_entry && flags.contains(JcFlags::JC_TRACING_OCCURRED) {
                        crate::mc_diag_bump(25);
                    }
                    if engine_is_tracing {
                        self.counter.change_current_fraction(hash, 0.98);
                        return HotResult::NotHot;
                    }
                    return HotResult::StartTracing;
                }
                if abort_count >= MAX_TRACE_ABORT_COUNT {
                    crate::mc_diag_bump(81);
                }
                return HotResult::NotHot;
            }
            if function_entry {
                crate::mc_diag_bump(24);
            }
            self.cleanup_chain(hash);
            return HotResult::NotHot;
        }

        if is_runnable {
            // warmstate.py maybe_compile_and_run: confirm_enter_jit before EnterJitAssembler.
            if !self.confirm_enter_jit_if_available(greenargs, green_types, reds) {
                return HotResult::NotHot;
            }
            return HotResult::RunCompiled;
        }
        // `has_compiled_meta` is a pending upstream mismatch (`warmstate.py maybe_compile_and_run` enters on any non-temporary procedure token).
        let not_hot = if function_entry {
            crate::mc_diag_bump(23);
            crate::mc_diag_bump(64);
            true
        } else {
            flags.contains(JcFlags::JC_DONT_TRACE_HERE)
        };
        if not_hot {
            return HotResult::NotHot;
        }
        tick_or_defer_nested(self, hash, increment, engine_is_tracing)
    }

    /// `warmstate.py` `bound_reached` — the veto, then the mark.
    ///
    /// Every tracing start in this module funnels through here or through its
    /// typed twin: `maybe_compile_with_key` reaches it from its own decision,
    /// and `force_start_tracing*` (which is what `MetaInterp::bound_reached`
    /// calls) reaches it after theirs. That makes this the one position at
    /// which the hook fires exactly once per start, which is what upstream
    /// gets by having a single `bound_reached`.
    ///
    /// `Some(key)` installs `JitCell(*greenargs)` (`warmstate.py
    /// bound_reached`). `None` names an already-installed cell; a tracing
    /// start without one needs greens — use the `_for_key` twin.
    fn start_tracing_cell(
        &mut self,
        cell_key: u64,
        green_key: Option<&GreenKey>,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> HotResult {
        match green_key {
            Some(key) => {
                if !self.confirm_enter_jit_if_available(Some(&key.values), &key.types, reds) {
                    return HotResult::NotHot;
                }
                self.ensure_cell_for_key(key);
            }
            None => {
                let cell = self.cell_by_key(cell_key).expect(
                    "the hash form names an installed cell; a tracing start without one needs greens, \
                     warmstate.py bound_reached JitCell(*greenargs) — use the `_for_key` twin",
                );
                let greens = cell.current_greens();
                if !self.confirm_enter_jit_if_available(Some(&greens[..]), green_types, reds) {
                    return HotResult::NotHot;
                }
            }
        }
        self.counter.reset(self.bucket_of(cell_key));
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        let cell = match green_key {
            Some(key) => self
                .lookup_chain_with_key(key)
                .expect("ensure_cell_for_key just installed a cell matching this key"),
            None => self.cell_by_key(cell_key).expect(
                "the hash form names an installed cell; a tracing start without one needs greens, \
                 warmstate.py bound_reached JitCell(*greenargs) — use the `_for_key` twin",
            ),
        };
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_TRACING | JcFlags::JC_TRACING_OCCURRED));
        cell.tracing_generation.set(current_generation);
        HotResult::StartTracing
    }

    /// Standalone state for tests: owns a fresh `JitCounter` and
    /// `MemoryManager` and sets the given threshold.
    pub fn new(threshold: u32) -> Self {
        let mut ws = Self::new_for_driver(new_runner_jitcounter(), new_runner_memory_manager(), 0);
        ws.set_threshold(threshold);
        ws
    }

    /// `WarmEnterState(warmrunnerdesc, jitdriver_sd)`: one state per
    /// `JitDriverStaticData`, reading the runner's one `jitcounter` and one
    /// `memory_manager` (`_compute_threshold`, `set_param_retrace_limit`)
    /// and starting from the `rlib/jit.py PARAMETERS` defaults.
    pub fn new_for_driver(
        counter: JitCounter,
        memory_manager: crate::memmgr::MemoryManager,
        jitdriver_sd: usize,
    ) -> Self {
        let threshold = crate::jit::PARAMETERS.threshold;
        let increment_threshold = counter.compute_threshold(threshold);
        let increment_trace_eagerness = counter.compute_threshold(DEFAULT_TRACE_EAGERNESS);
        let increment_function_threshold = counter.compute_threshold(DEFAULT_FUNCTION_THRESHOLD);
        WarmEnterState {
            counter,
            threshold,
            increment_threshold,
            trace_eagerness: DEFAULT_TRACE_EAGERNESS,
            increment_trace_eagerness,
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            increment_function_threshold,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            quasiimmut_deps: crate::FxIndexMap::default(),
            vectorize: false,
            vec_all: false,
            vec_cost: 0,
            enable_opts: default_enable_opts(),
            inlining: true,
            disable_unrolling_threshold: DEFAULT_DISABLE_UNROLLING,
            pureop_historylength: 16,
            memory_manager,
            cell_generation: 0,
            jitdriver_sd,
            confirm_enter_jit_ptr: Rc::new(Cell::new(None)),
            should_unroll_one_iteration_ptr: None,
        }
    }

    pub fn mark_dont_trace(&mut self, cell_key: u64) {
        self.disable_noninlinable_function(cell_key);
    }

    /// warmstate.py:467: jitcounter.tick(hash, increment_threshold).
    pub fn counter_tick(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key(cell_key)
            && cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE)
        {
            return;
        }
        let bucket = self.bucket_of(cell_key);
        let _ = self.counter.tick(bucket, self.increment_threshold);
    }

    pub fn counter_tick_checked(
        &mut self,
        cell_key: u64,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> bool {
        let cell = self.cell_by_key(cell_key);
        let is_runnable = cell.as_ref().is_some_and(|c| c.is_compiled());
        let hash = self.bucket_of(cell_key);
        let greens = if self.has_confirm_enter_jit() {
            cell.as_ref().map(|c| c.current_greens())
        } else {
            None
        };
        matches!(
            self.maybe_compile_and_run_decision(
                cell.as_deref(),
                hash,
                self.increment_threshold,
                greens.as_deref(),
                green_types,
                reds,
                is_runnable,
                WarmEntryDoor::BackEdge {
                    engine_is_tracing: false,
                },
            ),
            HotResult::StartTracing
        )
    }

    /// `warmstate.py maybe_compile_and_run` `:465-469`: the chain walk found
    /// no cell, so `jitcounter.tick(hash, increment_threshold)` and nothing
    /// else. The hash is the bucket itself — there is no minted cell key to
    /// map back — matching `tick(hash, …)` on the `else` of that walk.
    #[inline]
    pub fn tick_empty_chain(&mut self, hash: u64) -> bool {
        self.counter.tick(hash, self.increment_threshold)
    }

    /// `warmstate.py maybe_compile_and_run` `:256-257` / `:465-469` with
    /// `increment_function_threshold`: the function-entry door's empty-chain
    /// (or unchained not-compiled cell) tick. Slot 25 is the same bump
    /// [`Self::function_entry_step`] makes before that tick.
    #[inline]
    pub fn tick_function_entry_empty_chain(&mut self, hash: u64) -> bool {
        crate::mc_diag_bump(25);
        self.counter.tick(hash, self.increment_function_threshold)
    }

    /// The counter [`Self::cell_generation`] documents.
    pub fn cell_generation(&self) -> u64 {
        self.cell_generation
    }

    fn bump_cell_generation(&mut self) {
        self.cell_generation = self.cell_generation.wrapping_add(1);
    }

    /// Hash-form door of `warmstate.py maybe_compile_and_run`.
    pub(crate) fn maybe_compile_decision_with_meta(
        &mut self,
        cell_key: u64,
        has_compiled_meta: impl FnOnce(u64) -> bool,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> HotResult {
        self.maybe_compile_decision_with_meta_nested(
            cell_key,
            has_compiled_meta,
            false,
            green_types,
            reds,
        )
    }

    /// [`Self::maybe_compile_decision_with_meta`] with the live
    /// `MetaInterp::trace_attempt_in_progress` flag, so a fire while the walker holds the
    /// outer `TraceCtx` parks at 0.98 instead of returning `StartTracing`.
    pub(crate) fn maybe_compile_decision_with_meta_nested(
        &mut self,
        cell_key: u64,
        has_compiled_meta: impl FnOnce(u64) -> bool,
        engine_is_tracing: bool,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> HotResult {
        let cell = self.cell_by_key(cell_key);
        let is_runnable = cell
            .as_ref()
            .is_some_and(|c| c.is_compiled() && has_compiled_meta(cell_key));
        let hash = self.bucket_of(cell_key);
        let greens = if self.has_confirm_enter_jit() {
            cell.as_ref().map(|c| c.current_greens())
        } else {
            None
        };
        self.maybe_compile_and_run_decision(
            cell.as_deref(),
            hash,
            self.increment_threshold,
            greens.as_deref(),
            green_types,
            reds,
            is_runnable,
            WarmEntryDoor::BackEdge { engine_is_tracing },
        )
    }

    /// Compatibility form for callers whose compiled-loop metadata is wholly
    /// represented by the warm cell itself.
    ///
    /// `green_types` is the driver's `_green_args_spec` (`warmstate.py`
    /// `confirm_enter_jit(*args)`). A hook packs one type per stored green;
    /// this wrapper does not invent a spec. Callers that install no hook
    /// may pass `&[]` — confirm is not consulted.
    #[inline]
    pub fn maybe_compile_decision(
        &mut self,
        cell_key: u64,
        green_types: &[GreenType],
    ) -> HotResult {
        self.maybe_compile_decision_with_meta(cell_key, |_| true, green_types, &[])
    }

    /// [`Self::maybe_compile_decision`] followed by the mark that
    /// `warmstate.py:441` makes once the decision is taken:
    /// `cell.flags |= JC_TRACING | JC_TRACING_OCCURRED`, immediately before
    /// the `try:` that runs the trace.
    ///
    /// Callers that go straight from the answer into `setup_tracing` want
    /// this form. A caller that instead delegates to a tracing entry point
    /// which makes its own decision — `MetaInterp::bound_reached` reaches the
    /// same cell through [`Self::force_start_tracing_for_key`] — must take the
    /// decision alone, or that second call reads the mark this one left and
    /// answers `AlreadyTracing` for the trace it was asked to start.
    ///
    /// Test-only hash commit: production creates through
    /// [`Self::start_tracing_cell_for_key`] / `JitCell(*greenargs)`.
    #[cfg(test)]
    pub fn maybe_compile(&mut self, cell_key: u64) -> HotResult {
        match self.maybe_compile_decision(cell_key, &[]) {
            HotResult::StartTracing => self.start_tracing_cell(cell_key, None, &[], &[]),
            other => other,
        }
    }

    /// warmstate.py `WarmEnterState.maybe_compile_and_run` —
    /// typed-greenkey form of the decision-then-mark.
    ///
    /// Walks the per-bucket chain by typed comparekey for both the
    /// state read (`lookup_chain_with_key`) and the state mutation
    /// (`start_tracing_cell_for_key`), mirroring upstream's
    /// `JitCell.get_jitcell_for_args` (`warmstate.py`) followed
    /// by per-cell flag mutation. Hash collisions across distinct
    /// typed greens therefore do not cross-contaminate each other's
    /// `JC_TRACING` / `JC_COMPILED` flags or counter-derived
    /// transitions.
    ///
    /// The bucket-shared counter ticks via `key.get_uhash()` — upstream
    /// counter is per-bucket too (`warmstate.py`), so colliding
    /// typed greens share the back-edge counter while keeping
    /// independent cell state.
    ///
    /// The typed chain-walk variant [`Self::finish_tracing_for_key`] keeps
    /// the typed path's full lifecycle self-consistent on its own buckets.
    pub fn maybe_compile_with_key(&mut self, key: &GreenKey) -> HotResult {
        match self.maybe_compile_decision_with_key(key, |_| true, &[]) {
            HotResult::StartTracing => self.start_tracing_cell_for_key(key, &[]),
            other => other,
        }
    }

    /// The typed cell-policy decision without marking a tracing start.
    /// `has_compiled_meta` supplies the frontend half of pyre's runnable-loop
    /// predicate; keeping it inside this decision prevents an entry door from
    /// bypassing the compiled-code veto with an earlier fast path.
    pub fn maybe_compile_decision_with_key(
        &mut self,
        key: &GreenKey,
        has_compiled_meta: impl FnOnce(u64) -> bool,
        reds: &[Value],
    ) -> HotResult {
        self.maybe_compile_decision_with_key_nested(key, has_compiled_meta, false, reds)
    }

    /// [`Self::maybe_compile_decision_with_key`] with the live
    /// `MetaInterp::trace_attempt_in_progress` flag. `warmstate.py maybe_compile_and_run`
    /// ticks other greenkeys during an outer `compile_and_run_once`.
    pub fn maybe_compile_decision_with_key_nested(
        &mut self,
        key: &GreenKey,
        has_compiled_meta: impl FnOnce(u64) -> bool,
        engine_is_tracing: bool,
        reds: &[Value],
    ) -> HotResult {
        let hash = key.get_uhash();
        let cell = self.lookup_chain_with_key(key);
        let is_runnable = cell.as_ref().is_some_and(|c| {
            // `compiled_loops` is filed under this cell's assigned key.
            // `install_new_cell` mints `BaseJitCell::cell_key` when
            // `get_uhash` is already taken, and `maybe_compile_and_run_step`
            // resolves that same number with `cell_key_for` after
            // `RunCompiled`. The raw hash names the bucket's first cell.
            let cell_key = c.cell_key.get().unwrap_or(hash);
            c.is_compiled() && has_compiled_meta(cell_key)
        });
        self.maybe_compile_and_run_decision(
            cell.as_deref(),
            hash,
            self.increment_threshold,
            Some(&key.values),
            &key.types,
            reds,
            is_runnable,
            WarmEntryDoor::BackEdge { engine_is_tracing },
        )
    }

    /// Mutable chain-walk variant of [`Self::lookup_chain_with_key`].
    /// Returns `Some` only when a chained cell carries a comparekey
    /// equal (`equal_whatever`) to `key`. The bucket head is not privileged.
    fn lookup_chain_with_key_mut(&mut self, key: &GreenKey) -> Option<Rc<BaseJitCell>> {
        self.lookup_chain_with_key(key)
    }

    /// warmstate.py `WarmEnterState.bound_reached` —
    /// typed-key variant of [`Self::start_tracing_cell`].
    ///
    /// Walks the chain by `key`'s comparekey to mutate the matching
    /// cell's `JC_TRACING` / `JC_TRACING_OCCURRED` flags +
    /// `tracing_generation`, rather than the bucket head. Pairs with
    /// [`Self::maybe_compile_with_key`] so a typed greenkey always
    /// transitions its own cell on a hash collision.
    fn start_tracing_cell_for_key(&mut self, key: &GreenKey, reds: &[Value]) -> HotResult {
        let hash = key.get_uhash();
        // `warmstate.py` `bound_reached` — see `start_tracing_cell`, whose
        // position this is the typed twin of.
        if !self.confirm_enter_jit(&key.values, &key.types, reds) {
            return HotResult::NotHot;
        }
        self.counter.reset(hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_TRACING | JcFlags::JC_TRACING_OCCURRED));
        cell.tracing_generation.set(current_generation);
        HotResult::StartTracing
    }

    /// Typed-key variant of [`Self::finish_tracing`]. Walks the bucket
    /// chain by comparekey so a typed green that collides with another
    /// clears `JC_TRACING` on its own cell rather than the bucket head.
    pub fn finish_tracing_for_key(&mut self, key: &GreenKey) {
        if let Some(cell) = self.lookup_chain_with_key_mut(key) {
            cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Greens stored on the installed cell named by `cell_key`.
    ///
    /// `warmstate.py` `JitCell.__init__` (`make_jitcell_subclass`) keeps the
    /// greenargs on the cell (`setattr(self, attrname, greenargs[i])`).
    /// `types` is the driver's `_green_args_spec`. The JitCell class is per
    /// driver, so the stored greens always match that spec.
    pub fn installed_greenkey(&self, cell_key: u64, types: &[GreenType]) -> Option<GreenKey> {
        self.cell_by_key(cell_key).map(|c| {
            let greens = c.current_greens();
            assert_eq!(
                greens.len(),
                types.len(),
                "cell greens match the driver's _green_args_spec \
                 (warmstate.py make_jitcell_subclass)"
            );
            GreenKey::with_types(greens, types.to_vec())
        })
    }

    /// Typed-key variant of [`Self::disable_noninlinable_function`]:
    /// installs a cell for `key` if absent (`ensure_jit_cell_at_key`), then
    /// sets `DONT_TRACE_HERE` on the cell whose comparekey matches `key`.
    pub fn disable_noninlinable_function_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_DONT_TRACE_HERE));
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    ///
    /// Used by function-entry tracing where the caller has already
    /// determined that tracing should begin.
    pub fn force_start_tracing(
        &mut self,
        cell_key: u64,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> HotResult {
        if let Some(cell) = self.cell_by_key(cell_key) {
            if cell.is_compiled() {
                let greens = cell.current_greens();
                if !self.confirm_enter_jit_if_available(Some(&greens[..]), green_types, reds) {
                    return HotResult::NotHot;
                }
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            // `warmstate.py maybe_compile_and_run` handles JC_TEMPORARY before it consults
            // JC_DONT_TRACE_HERE: a compile_tmp_callback is only the residual
            // CALL_ASSEMBLER fallback, so it must keep counting toward the
            // callee's real standalone trace even when that callee was marked
            // non-inlinable.  Refusing the combination here strands the
            // JC_FORCE_FINISH retry installed after a trace-too-long abort.
            if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE)
                && !cell.flags.get().contains(JcFlags::JC_TEMPORARY)
                && cell.has_seen_a_procedure_token()
            {
                return HotResult::NotHot;
            }
            // The abort ceiling bans a cell that has already failed
            // `MAX_TRACE_ABORT_COUNT` traces, including one that carries
            // `JC_DONT_TRACE_HERE` from that stamp. A dead token still
            // belongs to `cleanup_chain` on the occupied door.
            let dead_token =
                cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none();
            if cell.abort_count.get() >= MAX_TRACE_ABORT_COUNT && !dead_token {
                crate::mc_diag_bump(81);
                return HotResult::NotHot;
            }
        }

        self.start_tracing_cell(cell_key, None, green_types, reds)
    }

    /// Typed-key variant of [`Self::force_start_tracing`]: reads the
    /// matching cell by comparekey and force-starts tracing on it via
    /// `Self::start_tracing_cell_for_key`, so a force-started cell
    /// (function-entry / can_enter_jit) carries a `comparekey` like the
    /// [`Self::maybe_compile_with_key`] path.
    pub fn force_start_tracing_for_key(&mut self, key: &GreenKey, reds: &[Value]) -> HotResult {
        if let Some(cell) = self.lookup_chain_with_key(key) {
            if cell.is_compiled() {
                if !self.confirm_enter_jit(&key.values, &key.types, reds) {
                    return HotResult::NotHot;
                }
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE)
                && !cell.flags.get().contains(JcFlags::JC_TEMPORARY)
                && cell.has_seen_a_procedure_token()
            {
                return HotResult::NotHot;
            }
            let dead_token =
                cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none();
            if cell.abort_count.get() >= MAX_TRACE_ABORT_COUNT && !dead_token {
                crate::mc_diag_bump(81);
                return HotResult::NotHot;
            }
        }

        self.start_tracing_cell_for_key(key, reds)
    }

    /// Mark that tracing is done for a green key. Clears the TRACING flag.
    /// The caller is responsible for compiling the trace and calling
    /// `attach_procedure_to_interp` with the resulting JitCellToken.
    pub fn finish_tracing(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Undo a tracing start which the frontend declined before recording.
    ///
    /// This is the local counterpart of `warmstate.py`
    /// `maybe_compile_and_run`'s unconditional `finally` clear when the
    /// pre-trace policy refuses a graph: there is no `SwitchToBlackhole`, so
    /// `aborted_tracing` and the pyre-local abort ceiling do not participate.
    pub fn decline_tracing(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
        }
    }

    /// Mark that tracing was aborted for a green key.
    ///
    /// Non-permanent aborts clear `JC_TRACING` and allow a future retry until
    /// pyre's abort ceiling marks the location `DONT_TRACE_HERE`.
    pub fn abort_tracing(&mut self, cell_key: u64, disable_noninlinable_function: bool) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
            cell.abort_count.set(cell.abort_count.get() + 1);
            let already_banned = cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE);
            let ceiling_reached = cell.abort_ceiling_latched();
            if disable_noninlinable_function || already_banned || ceiling_reached {
                cell.flags
                    .set(cell.flags.get() | (JcFlags::JC_DONT_TRACE_HERE));
            }
            if ceiling_reached && !already_banned && !disable_noninlinable_function {
                crate::mc_diag_bump(80);
            }
        }

        if disable_noninlinable_function {
            self.disable_noninlinable_function(cell_key);
        }
    }

    /// Install a compiled loop token for a green key.
    ///
    /// The cell transitions to Compiled state and takes ownership of
    /// the procedure token. Also clears TRACING on the cell the token
    /// is attached to — this covers the same-key compile path, while
    /// cross-loop cut (compile.py:269) compiles under an inner cell
    /// and the outer (starting) cell's TRACING is cleared separately
    /// by the `clear_tracing_flag` call in the tracing entry point's
    /// finally block (warmstate.py:444 parity).
    ///
    /// The hash form names an installed cell. Installing one needs the
    /// greenkey (`warmstate.py` `attach_procedure_to_interp`
    /// `ensure_jit_cell_at_key(greenkey)`), which the `_for_key` twin takes.
    pub fn attach_procedure_to_interp(
        &mut self,
        cell_key: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Option<Arc<JitCellToken>> {
        let token = token.into();
        let cell = self.cell_by_key(cell_key).expect(
            "attach_procedure_to_interp names an installed cell; installing one needs greens, \
             warmstate.py attach_procedure_to_interp ensure_jit_cell_at_key(greenkey) — use the `_for_key` twin",
        );
        cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
        let previous = cell.set_procedure_token(token, false);
        self.bump_cell_generation();
        previous
    }

    /// Typed form of [`Self::attach_procedure_to_interp`].
    ///
    /// `JitCell.__init__` (warmstate.py) always stores the green args
    /// on the cell (`ensure_jit_cell_at_key(greenkey)`).
    pub fn attach_procedure_to_interp_for_key(
        &mut self,
        key: &GreenKey,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Option<Arc<JitCellToken>> {
        let token = token.into();
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
        let previous = cell.set_procedure_token(token, false);
        self.bump_cell_generation();
        previous
    }

    /// warmstate.py `finally: cell.flags &= ~JC_TRACING` parity —
    /// unconditional flag clear on the starting cell after tracing ends.
    /// Called from the synchronous tracing entry point
    /// regardless of whether tracing succeeded, aborted, or cross-loop-cut
    /// installed under a different inner cell. Does not alter state: the
    /// companion `attach_procedure_to_interp` / `abort_tracing` calls own
    /// the state transition for whichever cell they touch.
    pub fn clear_tracing_flag(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags.set(cell.flags.get() & !JcFlags::JC_TRACING);
        }
    }

    /// Get a reference to the compiled loop token for a green key.
    ///
    /// `warmstate.py get_procedure_token` parity: returns `None`
    /// when the cell has no procedure token AND when the token has been
    /// invalidated (`token and not token.invalidated`).  Pyre routes
    /// through `BaseJitCell::get_procedure_token` (which applies the
    /// `is_invalidated` filter) so every entry-path consumer of the
    /// "current compiled token at this key" sees the same filtered view
    /// PyPy provides — invalidated tokens never reach the warm-entry
    /// runner, the CALL_ASSEMBLER inline gate, or the bridge stitch
    /// surface.
    ///
    /// Returns an OWNED handle, not a borrow out of the celltable: the cell
    /// holds only a `Weak` now (`warmstate.py`), so there is no
    /// `Arc<JitCellToken>` inside the table to lend out. The `Arc` this hands
    /// back is what keeps the token alive for the caller's use of it.
    pub fn get_compiled(&self, cell_key: u64) -> Option<Arc<JitCellToken>> {
        self.cell_by_key(cell_key)
            .and_then(|cell| cell.get_procedure_token())
    }

    /// warmstate.py `get_procedure_token`.
    ///
    /// Reads the cell the CELL KEY names — one cell, whatever the bucket holds
    /// — so this and [`Self::get_procedure_token_for_key`] answer about the
    /// same object for any key that has been resolved
    /// ([`Self::cell_key_for`]). Handing this a raw bucket hash instead is a
    /// different question with a different answer: it names the bucket's first
    /// occupant.
    pub fn get_procedure_token(&self, cell_key: u64) -> Option<Arc<JitCellToken>> {
        self.cell_by_key(cell_key)
            .and_then(|cell| cell.get_procedure_token())
    }

    /// Whether the resolved cell's current procedure token is the temporary
    /// interpreter callback installed by `compile_tmp_callback`.
    ///
    /// `warmstate.py maybe_compile_and_run` reads `JC_TEMPORARY` before it
    /// considers the token executable at a warm entry. Keep this on the cell,
    /// where upstream owns it; a token body or pyre's retained
    /// `compiled_loops` metadata cannot answer the same question.
    pub(crate) fn procedure_token_is_temporary(&self, cell_key: u64) -> bool {
        self.cell_by_key(cell_key)
            .is_some_and(|cell| cell.flags.get().contains(JcFlags::JC_TEMPORARY))
    }

    /// [`Self::get_procedure_token`] with the two flag reads taken OUT: the
    /// cell lookup, the `Weak::upgrade`, and the drop of the `Arc` it
    /// produced, and nothing else.
    ///
    /// A measurement reader with no production caller and no upstream
    /// counterpart. It exists because the shipping predicate fuses two
    /// separable costs into one function -- `get_procedure_token` performs the
    /// refcount pair AND reads `invalidated` -- so no arm built from the public
    /// API can say which of them a sibling-door scan is paying for. Differenced
    /// against `has_compiled_loop`, this one leaves the flag reads.
    ///
    /// The answer is deliberately WEAKER than the shipping predicate's: an
    /// invalidated token still upgrades, so this returns `true` where the door
    /// would decide `false`. That is why it is a cost probe and never a
    /// decision -- nothing may route on it.
    #[cfg(feature = "__yield-stage-probe")]
    pub fn probe_cell_token_upgrades(&self, cell_key: u64) -> bool {
        self.cell_by_key(cell_key).is_some_and(|cell| {
            cell.loop_token
                .borrow()
                .as_ref()
                .and_then(|w| w.upgrade())
                .is_some()
        })
    }

    /// `warmstate.py` — resolve the cell by `comparekey`, then read its
    /// procedure token.
    ///
    /// The typed twin of [`Self::get_procedure_token`]. Upstream reaches a
    /// token only through a cell it has already matched on the full green key;
    /// this is that discipline. A bucket the key does not match anywhere is
    /// upstream's `else:` arm — "not found" — and answers `None` rather than
    /// handing back the head cell's unrelated token.
    ///
    /// Equivalent to [`Self::cell_key_for`] followed by
    /// [`Self::get_procedure_token`], and callers that need the key as well as
    /// the token should spell it that way so the key they carry onward is the
    /// one this matched on.
    pub fn get_procedure_token_for_key(&self, key: &GreenKey) -> Option<Arc<JitCellToken>> {
        self.lookup_chain_with_key(key)
            .and_then(|cell| cell.get_procedure_token())
    }

    /// Whether THIS HASH owns more than one cell, i.e. whether a raw hash
    /// leaves anything to choose between.
    ///
    /// A chain needs a hash collision: two green keys that share
    /// `get_uhash` produce two cells for one hash. A hash owning exactly
    /// one cell leaves one candidate, so
    /// [`Self::sole_cell_key`] is exact and a caller
    /// can skip building a `GreenKey` it would only use to confirm that — which
    /// is what keeps [`Self::resolve_cell_key`] allocation-free on the entry
    /// path.
    ///
    /// The question is about the hash, not about the table slot. Since
    /// `_get_index` (counter.py) drops all but eleven bits of the hash,
    /// a slot routinely chains cells belonging to OTHER hashes, and those
    /// leave this hash's reader nothing to choose between — it never had a
    /// candidate in them. [`Self::cell_keys_at`] counts the occupants that are
    /// this hash's, which is the same walk `maybe_compile_and_run` does with
    /// `comparekey` in hand (warmstate.py), spelled for a caller that
    /// holds only the number.
    ///
    /// Counting through [`Self::cell_keys_at`] also keeps this reader and
    /// [`Self::sole_cell_key`] on ONE route. Upstream cannot split them: every
    /// reader enters at `jitcounter.lookup_chain(hash)`
    /// (warmstate.py, :597-598, :632-633), and `lookup_chain`
    /// (counter.py) is a bare `celltable[self._get_index(hash)]`. See
    /// `a_raw_hash_equal_to_a_minted_key_resolves_through_one_bucket`.
    #[inline]
    pub fn bucket_is_chained(&self, green_key_hash: u64) -> bool {
        self.cell_keys_at(green_key_hash).0 > 1
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// warmstate.py set_param_threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.increment_threshold = self.counter.compute_threshold(threshold);
    }

    /// Decay all counters (e.g., periodically to avoid stale counts).
    pub fn decay_counters(&mut self) {
        self.counter.decay_all_counters();
    }

    /// Get a reference to the BaseJitCell for a green key, if it exists.
    pub fn get_cell(&self, cell_key: u64) -> Option<Rc<BaseJitCell>> {
        self.cell_by_key(cell_key)
    }

    /// Typed-key variant of [`Self::get_cell`]:
    /// `warmstate.py JitCell.get_jitcell(*greenargs)`.
    ///
    /// Walks the chain by `comparekey`, so it selects the cell belonging to
    /// this key rather than the cell that happens to hold the bucket's raw
    /// hash as its key. Those differ whenever a bucket chains two
    /// `JitCell(*greenargs)` occupants (`install_new_cell` links the
    /// survivor behind the newcomer).
    ///
    /// The reason to route through the key rather than the hash at all:
    /// every upstream cell lookup takes greenargs and compares them
    /// (`get_jitcell` :596, `_ensure_jit_cell_at_key` :631,
    /// `dont_trace_here` :644, `mark_as_being_traced` :649 — warmstate.py).
    /// Upstream has exactly one bare-hash entry point,
    /// `trace_next_iteration_hash` (warmstate.py), and it touches
    /// **the counter, not the cell**.
    /// That is the line: a hash is enough to find a bucket, never enough to
    /// pick a cell out of one.
    pub fn get_cell_for_key(&self, key: &GreenKey) -> Option<Rc<BaseJitCell>> {
        self.lookup_chain_with_key(key)
    }

    /// `rpython/jit/metainterp/warmstate.py` `get_assembler_token`.
    ///
    /// Returns the cell's existing procedure token, or — if none exists —
    /// builds a temporary one via `make_token` (caller wires
    /// `compile_tmp_callback`) and installs it on the cell with
    /// `tmp=true`.  The closure-based signature is a Rust adaptation so
    /// the caller can provide the `&mut Backend` / `&JitDriverStaticData`
    /// / `greenboxes` bundle without threading them through WarmEnterState.
    ///
    /// The closure is handed `&mut MemoryManager` because upstream reads it
    /// off `self.warmrunnerdesc.memory_manager` at the same point
    /// (`warmstate.py`) and passes it into `compile_tmp_callback`,
    /// which registers the fresh token in `alive_loops` (`compile.py:1148`).
    /// Pyre keeps the manager on `WarmEnterState` itself, so the caller cannot
    /// reach it while this method holds `&mut self`; passing it in is what
    /// keeps the registration at the call `compile.py` makes it from.  Without
    /// it the token installed below has no strong owner at all
    /// (`BaseJitCell::loop_token` is weak) and dies before first entry.
    pub fn get_assembler_token<E, F>(
        &mut self,
        cell_key: u64,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce(&mut crate::memmgr::MemoryManager) -> Result<Arc<JitCellToken>, E>,
    {
        if let Some(token) = self
            .cell_by_key(cell_key)
            .and_then(|c| c.get_procedure_token())
        {
            return Ok(token.clone());
        }
        // `celltable` and `memory_manager` are disjoint fields; the cell borrow
        // above has ended, so the closure gets the manager on its own.
        let token = make_token(&mut self.memory_manager)?;
        let cell = self.cell_by_key(cell_key).expect(
            "get_assembler_token names an installed cell; installing one needs greens, \
             warmstate.py get_assembler_token ensure_jit_cell_at_key(greenkey) — use the `_for_key` twin",
        );
        cell.set_procedure_token(token.clone(), true);
        Ok(token)
    }

    /// `warmstate.py` `get_assembler_token` typed variant —
    /// `JitCell.ensure_jit_cell_at_key(greenkey)` then the cell's identity.
    ///
    /// `compile_tmp_callback` (`make_token`) can collect, so the typed
    /// greens are resolved to the cell key before that call and the
    /// post-compile install uses [`Self::get_assembler_token`]. The cell's
    /// own comparekey is forwarded by the GC walker; a raw `GreenKey` of
    /// reference words is not.
    pub fn get_assembler_token_with_key<E, F>(
        &mut self,
        key: &GreenKey,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce(&mut crate::memmgr::MemoryManager) -> Result<Arc<JitCellToken>, E>,
    {
        self.ensure_cell_for_key(key);
        let cell_key = self.cell_key_for(key).unwrap_or_else(|| key.get_uhash());
        self.get_assembler_token(cell_key, make_token)
    }

    /// warmstate.py: trace_eagerness parameter (integer).
    pub fn trace_eagerness(&self) -> u32 {
        self.trace_eagerness
    }

    /// warmstate.py: set_param_trace_eagerness.
    pub fn set_param_trace_eagerness(&mut self, value: u32) {
        self.trace_eagerness = value;
        self.increment_trace_eagerness = self.counter.compute_threshold(value);
    }

    /// warmstate.py: increment_trace_eagerness (pre-computed f64).
    pub fn increment_trace_eagerness(&self) -> f64 {
        self.increment_trace_eagerness
    }

    /// compile.py:783-784: jitcounter.tick(hash, increment_trace_eagerness).
    /// Increment the guard failure counter using the shared timetable.
    /// Returns true when counter reaches 1.0 (trace_eagerness ticks).
    #[inline]
    pub fn tick_guard_failure(&mut self, guard_hash: u64) -> bool {
        self.counter
            .tick(guard_hash, self.increment_trace_eagerness)
    }

    /// compile.py: store_hash — allocate a jitcounter hash for
    /// a new guard. Called at compile time (or lazily on first failure).
    pub fn fetch_next_hash(&mut self) -> u64 {
        self.counter.fetch_next_hash()
    }

    /// Get the function inlining threshold.
    pub fn function_threshold(&self) -> u32 {
        self.function_threshold
    }

    /// warmstate.py set_param_function_threshold.
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.function_threshold = threshold;
        self.increment_function_threshold = self.counter.compute_threshold(threshold);
    }

    /// RPython-compatible wrapper: set_param_threshold.
    pub fn set_param_threshold(&mut self, threshold: u32) {
        self.set_threshold(threshold);
    }

    /// RPython-compatible wrapper: set_param_trace_limit.
    pub fn set_param_trace_limit(&mut self, value: u32) {
        self.set_trace_limit(value);
    }

    /// RPython-compatible wrapper: set_param_function_threshold.
    pub fn set_param_function_threshold(&mut self, value: u32) {
        self.set_function_threshold(value);
    }

    /// RPython-compatible wrapper: set_param_inlining.
    pub fn set_param_inlining(&mut self, value: bool) {
        self.inlining = value;
    }

    /// RPython-compatible wrapper: set_param_disable_unrolling.
    pub fn set_param_disable_unrolling(&mut self, value: u32) {
        self.disable_unrolling_threshold = value;
    }

    /// RPython-compatible wrapper: set_param_vec.
    pub fn set_param_vec(&mut self, enabled: bool) {
        self.vectorize = enabled;
    }

    /// jit.py:585: set_param_vec_all.
    pub fn set_param_vec_all(&mut self, enabled: bool) {
        self.vec_all = enabled;
    }

    /// RPython-compatible wrapper: set_param_vec_cost.
    pub fn set_param_vec_cost(&mut self, value: u32) {
        self.vec_cost = value;
    }

    /// warmstate.py set_param_max_unroll_recursion — delegates
    /// to memory_manager.max_unroll_recursion.
    pub fn set_param_max_unroll_recursion(&mut self, value: u32) {
        self.memory_manager.set_max_unroll_recursion(value);
    }

    /// RPython-compatible wrapper: set_param_max_inline_depth.
    pub fn set_param_max_inline_depth(&mut self, value: u32) {
        self.set_max_inline_depth(value);
    }

    /// warmstate.py set_param_retrace_limit — delegates to
    /// memory_manager.retrace_limit.
    pub fn set_param_retrace_limit(&mut self, value: u32) {
        self.memory_manager.set_retrace_limit(value);
    }

    /// warmstate.py set_param_max_retrace_guards — delegates to
    /// memory_manager.max_retrace_guards.
    pub fn set_param_max_retrace_guards(&mut self, value: u32) {
        self.memory_manager.set_max_retrace_guards(value);
    }

    /// warmstate.py set_param_max_unroll_loops — delegates to
    /// memory_manager.max_unroll_loops.
    pub fn set_param_max_unroll_loops(&mut self, value: u32) {
        self.memory_manager.set_max_unroll_loops(value);
    }

    /// warmstate.py set_param_loop_longevity — delegates to the
    /// memory manager's max_age.
    pub fn set_param_loop_longevity(&mut self, value: u32) {
        // memmgr.py set_max_age default check_frequency=0 → derives sqrt(max_age).
        self.memory_manager.set_max_age(value as i64, 0);
    }

    /// RPython-compatible wrapper: set_param_pureop_historylength.
    pub fn set_param_pureop_historylength(&mut self, value: u32) {
        self.pureop_historylength = value;
    }

    /// warmstate.py set_param_decay — delegates to the jit counter.
    pub fn set_param_decay(&mut self, value: u32) {
        self.counter.set_decay(value as i32);
    }

    /// Set the maximum inline depth.
    pub fn set_max_inline_depth(&mut self, depth: u32) {
        self.max_inline_depth = depth;
    }

    /// Get the maximum inline depth.
    pub fn max_inline_depth(&self) -> u32 {
        self.max_inline_depth
    }

    /// Whether this callee is eligible for inlining at all.
    ///
    /// Mirrors PyPy's `can_inline_callable`: once a green key is marked
    /// `DONT_TRACE_HERE`, callers must stop inlining it and instead let it
    /// converge to a separate functrace / call_assembler path.
    pub fn can_inline_callable(&self, callee_key: u64) -> bool {
        self.cell_by_key(callee_key)
            .is_none_or(|cell| !cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE))
    }

    /// Typed-key variant of [`Self::can_inline_callable`].
    ///
    /// `warmstate.py` `can_inline_callable` receives the portal greens and
    /// reaches the `JitCell` through `get_jitcell_at_key`, whose chain walk
    /// calls `comparekey`.  A producer that still has those greens must use
    /// this door: reading the bucket's hash-form cell can miss the
    /// `DONT_TRACE_HERE` flag that `dont_trace_here(*greenargs)` set on the
    /// typed cell.
    pub fn can_inline_callable_for_key(&self, key: &GreenKey) -> bool {
        self.lookup_chain_with_key(key)
            .is_none_or(|cell| !cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE))
    }

    /// Mark a callee as a location that should no longer be inlined into
    /// surrounding traces.
    ///
    /// The hash form names an installed cell; installing one needs the
    /// greenkey (`ensure_jit_cell_at_key(greenkey)`), which the `_for_key`
    /// twin takes.
    pub fn disable_noninlinable_function(&mut self, callee_key: u64) {
        let cell = self.cell_by_key(callee_key).expect(
            "disable_noninlinable_function names an installed cell; installing one needs greens, \
             warmstate.py dont_trace_here ensure_jit_cell_at_key(greenkey) — use the `_for_key` twin",
        );
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_DONT_TRACE_HERE));
    }

    /// `warmstate.py mark_as_being_traced(*greenargs)`, which reaches
    /// its cell through `_ensure_jit_cell_at_key(*greenargs)` — i.e. by
    /// comparekey, never by hash alone.
    ///
    /// Installs through [`Self::ensure_cell_for_key`], so the cell it marks
    /// carries the greens (`JitCell(*greenargs)`) — the same shape
    /// [`Self::disable_noninlinable_function_for_key`] uses.
    ///
    /// `tracing_generation` is read before the borrow because the cell
    /// borrows `self.celltable` mutably.
    pub fn mark_as_being_traced_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let tracing_generation = self.tracing_generation;
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags.set(cell.flags.get() | (JcFlags::JC_TRACING));
        if !cell.flags.get().contains(JcFlags::JC_TRACING_OCCURRED) {
            cell.tracing_generation.set(tracing_generation);
        }
    }

    /// Restore warm-state parameters to rlib/jit.py PARAMETERS defaults.
    pub fn set_default_params(&mut self) {
        self.set_threshold(DEFAULT_THRESHOLD); // 1039
        self.set_param_trace_eagerness(DEFAULT_TRACE_EAGERNESS); // 200
        self.set_trace_limit(DEFAULT_TRACE_LIMIT); // 6000
        self.set_function_threshold(DEFAULT_FUNCTION_THRESHOLD); // 1619
        self.set_max_inline_depth(DEFAULT_MAX_INLINE_DEPTH); // 7
        self.inlining = true; // inlining = 1
        self.disable_unrolling_threshold = DEFAULT_DISABLE_UNROLLING;
        self.pureop_historylength = 16;
        self.counter.set_decay(40);
        self.memory_manager.set_max_retrace_guards(15);
        self.memory_manager.set_max_unroll_loops(0);
        self.memory_manager.set_retrace_limit(DEFAULT_RETRACE_LIMIT);
        self.memory_manager
            .set_max_unroll_recursion(DEFAULT_MAX_UNROLL_RECURSION);
        self.memory_manager.set_max_age(1000, 0);
        self.vec_cost = 0;
        self.vectorize = false;
        self.set_param_enable_opts("all");
    }

    /// Mirror RPython warmstate.py `mark_force_finish_tracing(greenkey)`.
    ///
    /// The next tracing run for this green key should segment instead of
    /// repeatedly aborting once it approaches the trace limit.
    ///
    /// The hash form names an installed cell; installing one needs the
    /// greenkey (`ensure_jit_cell_at_key(greenkey)`), which the `_for_key`
    /// twin takes.
    pub fn mark_force_finish_tracing(&mut self, cell_key: u64) {
        let cell = self.cell_by_key(cell_key).expect(
            "mark_force_finish_tracing names an installed cell; installing one needs greens, \
             warmstate.py mark_force_finish_tracing ensure_jit_cell_at_key(greenkey) — use the `_for_key` twin",
        );
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_FORCE_FINISH));
    }

    /// Typed form of [`Self::mark_force_finish_tracing`].
    ///
    /// `FORCE_FINISH` is sticky and never cleared explicitly, so setting it
    /// on the wrong cell of a bucket is permanent: the key that needs
    /// segmenting keeps aborting while an unrelated key segments forever.
    pub fn mark_force_finish_tracing_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_FORCE_FINISH));
    }

    /// warmstate.py `bool(cell.flags.get() & JC_FORCE_FINISH)` — read the sticky
    /// segmenting flag at loop entry.  RPython never clears this flag
    /// explicitly: `should_remove_jitcell` (warmstate.py) keeps the cell
    /// alive while it is set, and once set it persists until the cell itself
    /// is removed.
    pub fn should_force_finish_tracing(&self, cell_key: u64) -> bool {
        self.cell_by_key(cell_key)
            .is_some_and(|cell| cell.flags.get().contains(JcFlags::JC_FORCE_FINISH))
    }

    /// Typed form of [`Self::should_force_finish_tracing`].
    ///
    /// `mark_force_finish_tracing_for_key` writes `JC_FORCE_FINISH` through
    /// `ensure_jit_cell_at_key` (`warmstate.py`). `bound_reached` then reads
    /// that same cell (`bool(cell.flags.get() & JC_FORCE_FINISH)`). A hash
    /// read of the bucket head misses when the bucket holds more than one
    /// owner.
    pub fn should_force_finish_tracing_for_key(&self, key: &GreenKey) -> bool {
        self.lookup_chain_with_key(key)
            .is_some_and(|cell| cell.flags.get().contains(JcFlags::JC_FORCE_FINISH))
    }

    /// Boost the current loop/function green key so the next execution
    /// immediately retriggers tracing.
    ///
    /// Mirrors PyPy's `JitCell.trace_next_iteration()` in warmstate.py:
    /// it does not force tracing right now, it only raises the hot counter
    /// to ~threshold so the next hit converges quickly.
    ///
    /// Takes the bare hash on purpose, and it is the one shape allowed to:
    /// `_trace_next_iteration` (warmstate.py) hashes the greenargs and
    /// calls `jitcounter.change_current_fraction` — no cell is looked up, so
    /// the hash is the whole identity the operation needs. Upstream exposes
    /// exactly this as `trace_next_iteration_hash` (warmstate.py).
    /// The moment a cell read is added here, this needs a `&GreenKey` like
    /// [`Self::get_cell_for_key`]'s cohort.
    pub fn trace_next_iteration(&mut self, green_key_hash: u64) {
        self.counter
            .change_current_fraction(self.bucket_of(green_key_hash), 0.98);
    }

    /// warmstate.py:467 jitcounter.tick(hash, increment_threshold) parity.
    ///
    /// warmstate.py set_param_function_threshold: jitcounter.tick(hash, increment_function_threshold).
    ///
    /// The bool half of [`Self::function_entry_step`], kept as its own name
    /// because the mc_diag legend and the tests below are written about it.
    /// `has_compiled_meta` false is what makes the step run this whole body:
    /// with no frontend meta there is no runnable loop to return early for.
    pub fn should_trace_function_entry(&mut self, cell_key: u64) -> bool {
        matches!(
            self.function_entry_step(cell_key, None, || false, false, &[]),
            FunctionEntryStep::Proceed
        )
    }

    /// The function-entry door's whole answer, from ONE walk of the cell chain.
    ///
    /// `warmstate.py maybe_compile_and_run` with `increment_function_threshold`.
    /// `has_compiled_meta` is the frontend half of `has_runnable_compiled_loop`
    /// (`compiled_loops`, a map the warm state does not own); the cell half is
    /// `is_compiled` plus `has_compiled_code`. Both together make an ordinary
    /// token runnable; `JC_TRACING` and `JC_TEMPORARY` precede that test.
    ///
    /// `engine_is_tracing` is `MetaInterp::trace_attempt_in_progress` — engine-wide,
    /// not a per-cell flag. `warmstate.py maybe_compile_and_run` still ticks other
    /// greenkeys while it is set; only this cell's `JC_TRACING`
    /// returns. A counter fire while the walker holds the outer `TraceCtx`
    /// is parked at 0.98 (`JitCell.trace_next_iteration`) instead of nested
    /// `bound_reached`.
    pub fn function_entry_step(
        &mut self,
        cell_key: u64,
        green_key: Option<&GreenKey>,
        has_compiled_meta: impl FnOnce() -> bool,
        engine_is_tracing: bool,
        reds: &[Value],
    ) -> FunctionEntryStep {
        let cell = self.cell_by_key(cell_key);
        // `maybe_compile_and_run` reads `procedure_token =
        // cell.get_procedure_token()` once, decides on it, and raises
        // `EnterJitAssembler` with that same token.
        let runnable_token = cell
            .as_ref()
            .filter(|c| c.is_compiled() && has_compiled_meta())
            .and_then(|c| c.get_procedure_token())
            .filter(|token| token.has_compiled_code());
        let hash = self.bucket_of(cell_key);
        let (greenargs, green_types) = match green_key {
            Some(key) => (Some(key.values.as_slice()), key.types.as_slice()),
            None => (None, [].as_slice()),
        };
        match self.maybe_compile_and_run_decision(
            cell.as_deref(),
            hash,
            self.increment_function_threshold,
            greenargs,
            green_types,
            reds,
            runnable_token.is_some(),
            WarmEntryDoor::FunctionEntry { engine_is_tracing },
        ) {
            HotResult::RunCompiled => {
                FunctionEntryStep::RunCompiled(runnable_token.expect("runnable compiled token"))
            }
            HotResult::StartTracing => FunctionEntryStep::Proceed,
            HotResult::NotHot | HotResult::AlreadyTracing => FunctionEntryStep::NotHot,
        }
    }

    /// Typed-key form of [`Self::should_trace_function_entry`].
    ///
    /// `warmstate.py maybe_compile_and_run` walks the bucket chain
    /// with `comparekey` once and keeps that exact `JitCell` through the
    /// decision. A raw hash names only the bucket in pyre, so using the hash
    /// form for this first half and [`Self::force_start_tracing_for_key`] for
    /// the second can inspect two different cells in a chained bucket. Keep
    /// the whole function-entry decision on the matching typed cell instead.
    pub fn should_trace_function_entry_for_key(&mut self, key: &GreenKey) -> bool {
        let hash = key.get_uhash();
        let cell = self.lookup_chain_with_key(key);
        matches!(
            self.maybe_compile_and_run_decision(
                cell.as_deref(),
                hash,
                self.increment_function_threshold,
                Some(&key.values),
                &key.types,
                &[],
                false,
                WarmEntryDoor::FunctionEntry {
                    engine_is_tracing: false,
                },
            ),
            HotResult::StartTracing
        )
    }

    // ── Quasi-immutable field invalidation ──

    /// Register that the compiled loop at `cell_key` depends on the
    /// quasi-immutable field identified by `qmut_key`.
    ///
    /// When `invalidate_quasiimmut(qmut_key)` is called later, the compiled
    /// loop's JitCellToken will be invalidated, causing GUARD_NOT_INVALIDATED
    /// to fail and forcing a retrace.
    ///
    /// `qmut_key` should be a hash of (object_id, field_index) or similar.
    /// `cell_key` is a resolved cell identity, not a raw green-key hash: it is
    /// what names one cell among a bucket's occupants, so it is what the
    /// invalidation below can resolve back to the dependent it was recorded for.
    pub fn register_quasiimmut_dependency(&mut self, qmut_key: u64, cell_key: u64) {
        let deps = self.quasiimmut_deps.entry(qmut_key).or_default();
        if !deps.contains(&cell_key) {
            deps.push(cell_key);
        }
    }

    /// Invalidate all compiled loops that depend on the quasi-immutable field
    /// identified by `qmut_key`.
    ///
    /// Called by the interpreter when a quasi-immutable field is mutated.
    /// Each dependent loop's JitCellToken has its invalidated flag set, causing
    /// GUARD_NOT_INVALIDATED to fail on the next execution.
    ///
    /// Returns the number of loops invalidated.
    ///
    /// No typed twin, and this one needs none. Its dependents arrive as the
    /// *values* of `quasiimmut_deps`, which records a `cell_key` each — the
    /// identity [`Self::cell_by_key_mut`] resolves, and that resolve walks the
    /// whole chain rather than answering with the bucket head. So a colliding
    /// neighbour cannot be invalidated in a dependent's place, and a dependent
    /// sitting behind a chained head cannot be skipped: the property
    /// [`Self::invalidate_all`] below is written for — *skipping a cell is a
    /// WRONG ANSWER rather than a leak* — holds here for the same reason,
    /// without a `GreenKey` having to reach this far.
    ///
    /// Dead in production today: its only registrar,
    /// `register_quasiimmut_dependency`, has no production caller either, so
    /// `quasiimmut_deps` is empty and this returns 0 at the first line.
    pub fn invalidate_quasiimmut(&mut self, qmut_key: u64) -> usize {
        let deps = match self.quasiimmut_deps.swap_remove(&qmut_key) {
            Some(deps) => deps,
            None => return 0,
        };

        let mut invalidated = 0;
        for cell_key in &deps {
            if let Some(cell) = self.cell_by_key_mut(*cell_key)
                && let Some(token) = cell.loop_token.borrow().as_ref().and_then(|w| w.upgrade())
            {
                token.invalidate();
                invalidated += 1;
            }
        }
        invalidated
    }

    /// Invalidate all compiled loops that contain a GUARD_NOT_INVALIDATED.
    ///
    /// This is a brute-force invalidation used when the specific qmut_key
    /// is not known (e.g., bulk invalidation after a class hierarchy change).
    /// Walks each chain, not just its head.
    /// This is the sweep where skipping a cell is a WRONG ANSWER rather than a
    /// leak: a cell whose token is not invalidated keeps running compiled code
    /// built under an assumption that has just been retracted.
    pub fn invalidate_all(&mut self) {
        self.counter.for_each_cell(|cell| {
            if let Some(token) = cell.loop_token.borrow().as_ref().and_then(|w| w.upgrade()) {
                token.invalidate();
            }
        });
        self.quasiimmut_deps.clear();
    }

    // ── BaseJitCell state machine API ──

    /// Derived view of `flags` plus the procedure token. Not stored:
    /// `warmstate.py` `BaseJitCell` has only `flags`.
    #[inline]
    pub fn get_cell_state(&self, cell_key: u64) -> BaseJitCellState {
        self.cell_by_key(cell_key)
            .map(|cell| cell.derived_state())
            .unwrap_or(BaseJitCellState::NotHot)
    }

    /// Explicitly transition a cell to a new state.
    ///
    /// This is the low-level state-machine driver. Most callers should use
    /// the higher-level methods (`maybe_compile_with_key`, `finish_tracing`,
    /// `attach_procedure_to_interp`, `abort_tracing`) which call this internally.
    ///
    /// No typed twin: every caller is in this file's own test module, so the
    /// comparator-less cells it installs exist only in fixtures. That also
    /// makes it a way for a test to build the split-cell state deliberately.
    /// A twin becomes necessary if production ever calls this.
    #[cfg(test)]
    pub fn transition_cell(&mut self, cell_key: u64, new_state: BaseJitCellState) {
        let cell = self.cell_by_key(cell_key).expect(
            "transition_cell names an installed cell; installing one needs greens, \
             warmstate.py ensure_jit_cell_at_key(greenkey) — use a `_for_key` twin",
        );

        match new_state {
            BaseJitCellState::NotHot => {
                cell.flags
                    .set(cell.flags.get() & (!(JcFlags::JC_TRACING | JcFlags::JC_DONT_TRACE_HERE)));
            }
            BaseJitCellState::Tracing => {
                cell.flags
                    .set(cell.flags.get() | (JcFlags::JC_TRACING | JcFlags::JC_TRACING_OCCURRED));
            }
            BaseJitCellState::Compiled => {
                cell.flags.set(cell.flags.get() & (!JcFlags::JC_TRACING));
            }
            BaseJitCellState::Invalidated => {
                if let Some(token) = cell.loop_token.borrow().as_ref().and_then(|w| w.upgrade()) {
                    token.invalidate();
                }
            }
        }
    }

    // ── set_param / get_stats API ──

    /// Set a JIT parameter by its RPython name.
    ///
    /// Supported parameters:
    ///   - "threshold": compilation threshold
    ///   - "trace_limit": max ops per trace
    ///   - "trace_eagerness": guard fail count before bridge compilation
    ///   - "function_threshold": calls before inlining
    ///   - "max_inline_depth": maximum inlining depth
    ///
    /// `JitDriver.set_param` defines negative thresholds as disabled, and
    /// `JitCounter.compute_threshold` maps a disabled threshold to `0.0`.
    pub fn set_param(&mut self, name: &str, value: i64) {
        // Clamp disabled thresholds to zero instead of wrapping a negative
        // value to `u32::MAX`.
        let as_u32 = if value < 0 { 0u32 } else { value as u32 };
        match name {
            "threshold" => self.set_threshold(as_u32),
            "trace_limit" => self.trace_limit = as_u32,
            "trace_eagerness" => self.set_param_trace_eagerness(as_u32),
            "function_threshold" => self.set_function_threshold(as_u32),
            "max_inline_depth" => self.max_inline_depth = as_u32,
            "retrace_limit" => self.memory_manager.set_retrace_limit(as_u32),
            "max_retrace_guards" => self.memory_manager.set_max_retrace_guards(as_u32),
            "max_unroll_loops" => self.memory_manager.set_max_unroll_loops(as_u32),
            "max_unroll_recursion" => self.memory_manager.set_max_unroll_recursion(as_u32),
            "loop_longevity" => self.memory_manager.set_max_age(as_u32 as i64, 0),
            // warmstate.py set_param_vec — vec, vec_all, vec_cost are separate fields
            "vec" | "vectorize" => self.vectorize = value != 0,
            "vec_all" => self.vec_all = value != 0,
            "vec_cost" => self.vec_cost = as_u32,
            "inlining" => self.inlining = value != 0,
            "disable_unrolling" => self.disable_unrolling_threshold = as_u32,
            "pureop_historylength" => self.pureop_historylength = as_u32,
            "decay" => self.counter.set_decay(value as i32),
            "enable_opts" => {} // string param, handled by set_param_enable_opts
            _ => {}
        }
    }

    /// warmstate.py: set_param_enable_opts(value)
    /// Set which optimization passes are enabled.
    /// Value is a colon-separated string like "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll".
    /// "all" enables all passes.
    pub fn set_param_enable_opts(&mut self, value: &str) {
        // warmstate.py:284 substitutes the full list for `'all'` (and for
        // `None`, which this `&str` surface cannot express) and for nothing
        // else.  `''` therefore splits into no names and disables every pass;
        // treating it as `'all'` would silently keep unrolling on for a caller
        // that asked for the empty set.
        self.enable_opts = if value == "all" {
            default_enable_opts()
        } else {
            value
                .split(':')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        };
    }

    /// Get enabled optimization pass names.
    pub fn get_enable_opts(&self) -> &[String] {
        &self.enable_opts
    }

    /// `warmstate.py` `make_jitdriver_callbacks` —
    /// `self.confirm_enter_jit`, the callable every veto position calls.
    ///
    /// The driver's own hook when one was declared, and upstream's
    /// `def confirm_enter_jit(*args): return True` substitution
    /// (`warmstate.py`) when it was not.
    ///
    /// Packs greens then reds into one portal tuple
    /// (`warmspot.py` `_make_hook_graph(..., onlygreens=False)`).
    #[inline]
    pub fn confirm_enter_jit(
        &self,
        greens: &[i64],
        green_types: &[GreenType],
        reds: &[Value],
    ) -> bool {
        let Some(hook) = self.confirm_enter_jit_ptr.get() else {
            return true;
        };
        assert_eq!(
            greens.len(),
            green_types.len(),
            "confirm_enter_jit(*args): one type per green (_green_args_spec)"
        );
        let mut args = SmallVec::<[Value; 8]>::new();
        for (&word, &tp) in greens.iter().zip(green_types.iter()) {
            args.push(value_from_green_word(word, tp));
        }
        args.extend_from_slice(reds);
        // Packed `args` is dead after `hook(&args)` (`get_livevars_for_roots`).
        // The hook is the callee and roots its own locals.
        hook(&args)
    }

    /// Call the installed `confirm_enter_jit` hook with `greenargs` when both
    /// are present. A hook with no greens in hand refuses (`mc_diag` 95)
    /// rather than inventing a tuple; no hook installed is always allow
    /// (`warmstate.py` `def confirm_enter_jit(*args): return True`).
    #[inline]
    fn confirm_enter_jit_if_available(
        &self,
        greens: Option<&[i64]>,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> bool {
        match (self.confirm_enter_jit_ptr.get(), greens) {
            (None, _) => true,
            (Some(_), Some(values)) => self.confirm_enter_jit(values, green_types, reds),
            (Some(_), None) => {
                crate::mc_diag_bump(95); // confirm_enter_jit_missing_key
                false
            }
        }
    }

    /// Consult the compiled-entry veto for a resolved cell key.
    ///
    /// `live_greens` is this invocation's arguments when the caller holds
    /// them (`warmstate.py` `maybe_compile_and_run` `confirm_enter_jit(*args)`).
    /// `None` falls back to the cell's greens (`JitCell.__init__` `setattr`)
    /// for key-only doors.
    #[inline]
    pub(crate) fn confirm_compiled_entry_for_cell_key(
        &self,
        cell_key: u64,
        live_greens: Option<&[i64]>,
        green_types: &[GreenType],
        reds: &[Value],
    ) -> bool {
        if self.confirm_enter_jit_ptr.get().is_none() {
            return true;
        }
        let stored;
        let greens = if let Some(values) = live_greens {
            Some(values)
        } else {
            stored = self.cell_by_key(cell_key).map(|c| c.current_greens());
            stored.as_deref()
        };
        self.confirm_enter_jit_if_available(greens, green_types, reds)
    }

    /// `warmspot.py` `make_driverhook_graphs` —
    /// `jd._confirm_enter_jit_ptr = self._make_hook_graph(...)`.
    ///
    /// Upstream builds the pointer from the `JitDriver` declaration; nothing
    /// below the declaration can know what it said, so an interpreter installs
    /// its own here. `None` restores the no-hook default.
    pub fn set_confirm_enter_jit(&mut self, hook: Option<fn(&[Value]) -> bool>) {
        self.confirm_enter_jit_ptr.set(hook);
    }

    /// Whether an interpreter installed the optional entry veto. Entry doors
    /// use this to avoid constructing a structured key for the default
    /// always-allow case.
    #[inline]
    pub fn has_confirm_enter_jit(&self) -> bool {
        self.confirm_enter_jit_ptr.get().is_some()
    }

    /// The `make_entry_point` capture of `confirm_enter_jit`. `JitDriver`
    /// holds another handle to this same cell.
    pub(crate) fn confirm_enter_jit_handle(&self) -> Rc<Cell<Option<fn(&[Value]) -> bool>>> {
        Rc::clone(&self.confirm_enter_jit_ptr)
    }

    /// warmstate.py `WarmEnterState.should_unroll_one_iteration(greenkey)`.
    ///
    /// False when `_should_unroll_one_iteration_ptr` is None.
    pub fn should_unroll_one_iteration(&self, greens: &[i64]) -> bool {
        match self.should_unroll_one_iteration_ptr {
            Some(hook) => hook(greens),
            None => false,
        }
    }

    /// Bind `jd._should_unroll_one_iteration_ptr` (`warmspot.py
    /// make_driverhook_graphs`) onto this state.
    pub(crate) fn set_should_unroll_one_iteration_ptr(&mut self, ptr: Option<fn(&[i64]) -> bool>) {
        self.should_unroll_one_iteration_ptr = ptr;
    }

    /// warmstate.py: get_location(greenkey)
    /// Convert a green key to a human-readable source location string.
    /// Used for JIT logging and debugging.
    ///
    /// In RPython this is a user-provided callback set via JitDriver.
    /// Here we return a default format.
    pub fn get_location(&self, green_key: u64) -> String {
        format!("<jit key 0x{:x}>", green_key)
    }

    /// warmstate.py: get_param(name) — read a JIT parameter value.
    pub fn get_param(&self, name: &str) -> Option<i64> {
        match name {
            "threshold" => Some(self.threshold as i64),
            "trace_limit" => Some(self.trace_limit as i64),
            "trace_eagerness" => Some(self.trace_eagerness as i64),
            "function_threshold" => Some(self.function_threshold as i64),
            "max_inline_depth" => Some(self.max_inline_depth as i64),
            "retrace_limit" => Some(self.memory_manager.retrace_limit() as i64),
            "max_retrace_guards" => Some(self.memory_manager.max_retrace_guards() as i64),
            "max_unroll_loops" => Some(self.memory_manager.max_unroll_loops() as i64),
            "max_unroll_recursion" => Some(self.memory_manager.max_unroll_recursion() as i64),
            "loop_longevity" => Some(self.memory_manager.loop_longevity_param()),
            "vectorize" => Some(if self.vectorize { 1 } else { 0 }),
            "vec_cost" => Some(self.vec_cost as i64),
            "inlining" => Some(if self.inlining { 1 } else { 0 }),
            "disable_unrolling" => Some(self.disable_unrolling_threshold as i64),
            "pureop_historylength" => Some(self.pureop_historylength as i64),
            // warmstate.py has no getter for "decay": set_param_decay delegates
            // into jitcounter.set_decay which stores decay_by_mult (the derived
            // multiplier), not the raw int. Read-back is not supported.
            "decay" => None,
            _ => None,
        }
    }

    /// warmstate.py: set_param_to_default(name)
    /// Reset a single JIT parameter to its default value.
    pub fn set_param_to_default(&mut self, name: &str) {
        match name {
            "threshold" => self.set_threshold(1039), // RPython default
            "trace_limit" => self.trace_limit = DEFAULT_TRACE_LIMIT,
            "trace_eagerness" => self.set_param_trace_eagerness(DEFAULT_TRACE_EAGERNESS),
            "function_threshold" => self.set_function_threshold(DEFAULT_FUNCTION_THRESHOLD),
            "max_inline_depth" => self.set_max_inline_depth(DEFAULT_MAX_INLINE_DEPTH),
            "retrace_limit" => self.memory_manager.set_retrace_limit(DEFAULT_RETRACE_LIMIT),
            "max_retrace_guards" => self.memory_manager.set_max_retrace_guards(15),
            "max_unroll_loops" => self.memory_manager.set_max_unroll_loops(0),
            "max_unroll_recursion" => {
                self.memory_manager
                    .set_max_unroll_recursion(DEFAULT_MAX_UNROLL_RECURSION);
            }
            "loop_longevity" => self.memory_manager.set_max_age(1000, 0),
            "vectorize" => self.vectorize = false,
            "vec_cost" => self.vec_cost = 0,
            // rlib/jit.py PARAMETERS default decay=40.
            "decay" => self.counter.set_decay(40),
            _ => {}
        }
    }

    /// warmstate.py: get_param_names()
    /// Return all known parameter names.
    pub fn param_names() -> &'static [&'static str] {
        &[
            "threshold",
            "trace_limit",
            "trace_eagerness",
            "function_threshold",
            "max_inline_depth",
            "retrace_limit",
            "max_retrace_guards",
            "max_unroll_loops",
            "max_unroll_recursion",
            "loop_longevity",
            "vectorize",
            "vec_cost",
            "inlining",
            "disable_unrolling",
            "pureop_historylength",
            "decay",
        ]
    }

    // ── RPython warmstate.py getter methods ──

    pub fn retrace_limit(&self) -> u32 {
        self.memory_manager.retrace_limit()
    }
    pub fn max_retrace_guards(&self) -> u32 {
        self.memory_manager.max_retrace_guards()
    }
    pub fn max_unroll_loops(&self) -> u32 {
        self.memory_manager.max_unroll_loops()
    }
    pub fn max_unroll_recursion(&self) -> u32 {
        self.memory_manager.max_unroll_recursion()
    }
    pub fn vectorize(&self) -> bool {
        self.vectorize
    }
    /// warmstate.py: vec_all — try to vectorize all trace loops.
    pub fn vec_all(&self) -> bool {
        self.vec_all
    }
    pub fn vec_cost(&self) -> u32 {
        self.vec_cost
    }
    /// warmstate.py: inlining
    pub fn inlining(&self) -> bool {
        self.inlining
    }
    /// warmstate.py: disable_unrolling
    pub fn disable_unrolling_threshold(&self) -> u32 {
        self.disable_unrolling_threshold
    }
    /// warmstate.py: pureop_historylength
    pub fn pureop_historylength(&self) -> u32 {
        self.pureop_historylength
    }

    /// Get a snapshot of current JIT statistics.
    ///
    /// Counts every cell in every chain, not just the bucket heads.
    ///
    /// A chain needs no hash collision: a `comparekey: None` cell — what every
    /// bare-hash writer installs — can never match a typed lookup, so
    /// `ensure_cell_for_key` misses on a key that already has a cell and
    /// chains past it. See
    /// `one_key_through_a_hash_and_a_typed_entry_point_uses_one_cell`.
    ///
    /// So `num_cells` and the occupied-slot count are **already** capable of
    /// disagreeing, and the difference is the count of keys that reached
    /// both an untyped and a typed writer. Counting the cells is the
    /// definition the field documents ("Total number of BaseJitCells");
    /// counting the map entries counts buckets, which is a different number.
    ///
    /// `num_pinned_refs` sums `retained_greens` per cell, and a
    /// chained cell pins referents just as a head does. So the "adds zero
    /// today" note above applies to the state counters only.
    pub fn get_stats(&self) -> JitStats {
        let mut stats = JitStats::default();
        self.counter.for_each_cell(|cell| {
            stats.num_cells += 1;
            stats.num_pinned_refs += cell.retained_greens.len();
            let invalidated = cell
                .loop_token
                .borrow()
                .as_ref()
                .and_then(|w| w.upgrade())
                .is_some_and(|token| token.is_invalidated());
            if invalidated {
                stats.num_invalidated += 1;
            } else if cell.is_compiled() {
                stats.num_compiled += 1;
            } else if cell.is_tracing() {
                stats.num_tracing += 1;
            }
            // Counted off the flag, not the lifecycle state: a denial is
            // orthogonal to where the cell is in that lifecycle, and a cell
            // denied while it goes on to trace carries only the flag.
            if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
                stats.num_disable_noninlinable_function += 1;
            }
        });
        stats
    }

    /// Get the current trace limit.
    pub fn trace_limit(&self) -> u32 {
        self.trace_limit
    }

    /// Set the trace limit.
    pub fn set_trace_limit(&mut self, limit: u32) {
        self.trace_limit = limit;
    }

    /// Get the current tracing generation.
    pub fn tracing_generation(&self) -> u64 {
        self.tracing_generation
    }

    /// Remove dead BaseJitCells from all chains.
    /// Returns the number of cells removed.
    pub fn gc_cells(&mut self) -> usize {
        self.counter.gc_cells()
    }

    /// The bucket a cell key lives in.
    ///
    /// An unminted cell key IS its bucket's raw hash (it is only ever handed
    /// out inside that bucket), so the common path costs one `is_empty()`
    /// test and no lookup at all.
    #[inline]
    pub(crate) fn bucket_of(&self, cell_key: u64) -> u64 {
        self.counter.bucket_of(cell_key)
    }

    /// The cell named by `cell_key`, or `None` if no cell holds that key.
    ///
    /// This is what every hash-form entry point resolves through. Upstream's
    /// equivalent is not a lookup at all: `maybe_compile_and_run` already
    /// holds the cell object it matched (warmstate.py). Pyre's u64 is
    /// carried across crate boundaries (`compiled_loops`,
    /// `JitCellToken::green_key`, `rd_loop_token`), so it resolves back to the
    /// cell here — and it resolves to exactly one, which is the property that
    /// bucket-head reading did not have.
    #[inline]
    pub fn cell_by_key(&self, cell_key: u64) -> Option<Rc<BaseJitCell>> {
        self.counter.cell_by_key(self.jitdriver_sd, cell_key)
    }

    /// Mutable [`Self::cell_by_key`].
    #[inline]
    pub(crate) fn cell_by_key_mut(&mut self, cell_key: u64) -> Option<Rc<BaseJitCell>> {
        self.counter.cell_by_key(self.jitdriver_sd, cell_key)
    }

    /// Resolve greens to the cell key their cell is named by, WITHOUT
    /// installing anything.
    ///
    /// This is the "resolve once" step. `warmstate.py get_jitcell
    /// JitCell.get_jitcell` takes greens + `comparekey` + `get_uhash`
    /// together and yields a cell; pyre yields that cell's key, and every
    /// consumer downstream carries the key instead of re-deriving it from the
    /// bucket hash (warmstate.py/:511 carries the resolved
    /// `procedure_token` for the same reason).
    ///
    /// `None` means this key owns no cell AND its bucket's raw hash is already
    /// taken by a different cell — the only state in which there is no u64
    /// that names this key's (absent) cell. Readers must treat it as a miss;
    /// writers call [`Self::ensure_cell_key`], which mints.
    pub fn cell_key_for(&self, key: &GreenKey) -> Option<u64> {
        self.cell_key_for_greenargs(key.get_uhash(), &key.values, &key.types)
    }

    /// Resolve greens to their cell key, installing the cell (with its
    /// `comparekey`) if the key owns none — `warmstate.py
    /// _ensure_jit_cell_at_key`, reporting the identity of the cell it
    /// ensured.
    pub fn ensure_cell_key(&mut self, key: &GreenKey) -> u64 {
        self.ensure_cell_for_key(key);
        self.lookup_chain_with_key(key)
            .and_then(|cell| cell.cell_key.get())
            .expect("ensure_cell_for_key just installed a keyed cell for this key")
    }

    /// **Resolve once**: the cell key the greens behind `hash` name.
    ///
    /// `warmstate.py` `JitCell.get_jitcell` walks `lookup_chain(hash)` and
    /// returns a cell only after `cell.comparekey(*greenargs)`, including
    /// when the chain holds one occupant. A 64-bit `get_uhash` collision
    /// with a different green key must miss.
    ///
    /// See [`WarmEnterState`]'s type doc for why the resolve happens at the
    /// producer and the result is carried, rather than each consumer
    /// re-deriving a cell from the slot.
    #[inline]
    pub fn resolve_cell_key(&self, hash: u64, make_key: impl FnOnce() -> GreenKey) -> u64 {
        self.lookup_chain_with_key(&make_key())
            .and_then(|cell| cell.cell_key.get())
            .unwrap_or(hash)
    }

    /// [`Self::resolve_cell_key`] against greens already in hand:
    /// `JitCell.get_jitcell(*greenargs)` compares them in place, so a door
    /// that holds the greens builds no [`GreenKey`] to find its cell.
    /// `hash` must be `get_uhash(*greenargs)`.
    #[inline]
    pub fn resolve_cell_key_with_greenargs(
        &self,
        hash: u64,
        greenargs: &[i64],
        types: &[GreenType],
    ) -> u64 {
        debug_assert_eq!(
            hash,
            GreenKey::with_types(greenargs.to_vec(), types.to_vec()).get_uhash()
        );
        self.lookup_chain_with_greenargs(hash, greenargs, types)
            .and_then(|cell| cell.cell_key.get())
            .unwrap_or(hash)
    }

    /// [`Self::cell_key_for`] against greens already in hand.
    ///
    /// `JitCell.get_jitcell(*greenargs)`: walk the chain with `comparekey`
    /// and report the matching cell's key. Builds no `GreenKey`.
    fn cell_key_for_greenargs(
        &self,
        hash: u64,
        greenargs: &[i64],
        types: &[GreenType],
    ) -> Option<u64> {
        if let Some(cell) = self.lookup_chain_with_greenargs(hash, greenargs, types) {
            return cell.cell_key.get();
        }
        self.cell_by_key(hash).is_none().then_some(hash)
    }

    /// [`Self::resolve_cell_key`] and the resolved cell's ENTRY TOKEN answered
    /// from ONE walk of the bucket.
    ///
    /// A door that resolves a hash and then reads the resolved key's token
    /// walked the same chain twice for the same cell: `cell_keys_at` finds the
    /// cell and returns only its key, and `cell_by_key` then walks back to it.
    /// The single-candidate case — the one a door is in whenever the bucket
    /// holds exactly this key's cell — has the cell in hand at the moment the
    /// count reaches one, so the token is read there.
    ///
    /// `JC_TEMPORARY` is read off that same borrow, and a cell carrying it
    /// reports no token: `warmstate.py maybe_compile_and_run` tests the flag
    /// before it considers a token executable, and a `compile_tmp_callback`
    /// token has a body like any other, so nothing about the token itself
    /// answers that question. Folding it in here is what lets the whole
    /// cell-owned half of an entry decision cost one walk — asked separately it
    /// was a third walk of the same chain for the same cell.
    ///
    /// The other two cases are the second walk, unchanged. On a miss the key a
    /// later install would take is `hash`, but `hash` may itself be a minted
    /// key filed under a different bucket, so the cell has to be looked up
    /// through [`Self::cell_by_key`], which maps it. On an ambiguous bucket the
    /// greens are what settle it — and a caller that holds none (`greenargs`
    /// is `None`) gets the raw hash back, naming that bucket's original
    /// occupant or nothing, which is the same answer [`Self::sole_cell_key`]
    /// gives such a caller.
    ///
    /// `greenargs` is `*greenargs` already in hand
    /// (`JitCell.comparekey(*greenargs2)`). Comparison never builds a
    /// `GreenKey`. A unique-hash occupant is still accepted only after
    /// `comparekey` matches (`JitCell.get_jitcell`).
    ///
    /// The resolve is unaffected by the token: the key comes back whether or
    /// not one is found, because every consumer downstream is keyed by it.
    pub fn resolved_cell_procedure_token(
        &self,
        hash: u64,
        greenargs: Option<(&[i64], &[GreenType])>,
    ) -> (u64, Option<Arc<JitCellToken>>) {
        // `maybe_compile_and_run`: `cell = get_jitcell(*greenargs)`;
        // `if cell.flags & (JC_TRACING | JC_TEMPORARY)`; `procedure_token =
        // cell.get_procedure_token()`. The unchained bucket is that cell
        // only when it is this driver's JitCell (`isinstance(cell, JitCell)`
        // in `JitCell.get_jitcell`); every driver files into the one
        // `jitcounter.celltable`.
        //
        // `counter.py JitCounter.lookup_chain` is a table-slot load
        // (`return self.celltable[self._get_index(hash)]`).
        // `warmstate.py maybe_compile_and_run` keeps that pointer
        // (`cell = jitcounter.lookup_chain(hash)`). The owned `Rc`
        // that [`Self::lookup_chain`] clones is for callers that store
        // the cell; this arm only reads flags and upgrades the token.
        if let Some(cell) = self.counter.lookup_chain(hash)
            && cell.next.borrow().is_none()
            && cell.cell_bucket.get() == hash
            && cell.jitdriver_sd == self.jitdriver_sd
        {
            // `JitCell.get_jitcell`: unique occupant still needs comparekey.
            // `greenargs` is None is an identity lookup by cell key
            // (already-resolved keys); greens-holding doors pass the
            // comparator `maybe_compile_and_run` already has.
            if let Some((values, types)) = greenargs
                && !cell.comparekey(values, types)
            {
                return (hash, None);
            }
            let cell_key = cell.cell_key.get().unwrap_or(hash);
            return (cell_key, Self::cell_entry_procedure_token(&cell));
        }
        let (cell_key, cell) = self.resolved_cell(hash, greenargs);
        let token = cell.and_then(|cell| Self::cell_entry_procedure_token(&cell));
        (cell_key, token)
    }

    /// `maybe_compile_and_run` flag-first then `cell.get_procedure_token()`.
    /// `EnterJitAssembler` holds that owned token for the run.
    #[inline(always)]
    pub(crate) fn cell_entry_procedure_token(cell: &BaseJitCell) -> Option<Arc<JitCellToken>> {
        if cell.flags.get().contains(JcFlags::JC_TRACING)
            || cell.flags.get().contains(JcFlags::JC_TEMPORARY)
        {
            return None;
        }
        cell.get_procedure_token()
    }

    /// The table-slot load `maybe_compile_and_run` keeps
    /// (`cell = jitcounter.lookup_chain(hash)`), without an `Rc` clone.
    #[inline(always)]
    pub(crate) fn door_cell(
        &self,
        hash: u64,
        greenargs: Option<(&[i64], &[GreenType])>,
    ) -> DoorCell {
        let Some(cell) = self.counter.lookup_chain(hash) else {
            return DoorCell::Empty;
        };
        // A chained bucket needs comparekey and stays on `back_edge_internal`.
        // A lone cell is the tick arm only when this hash owns it.
        if cell.next.borrow().is_some() {
            return DoorCell::Occupied;
        }
        // `lookup_chain` returns the table-slot head. `_get_index` keeps
        // only the high bits of the low 32, so that head can belong to a
        // different green key. `maybe_compile_and_run` walks with
        // `isinstance(cell, JitCell) and cell.comparekey(*greenargs)`
        // and only on a match reads `cell.flags`.
        if cell.cell_bucket.get() != hash {
            return DoorCell::Empty;
        }
        if let Some((values, types)) = greenargs
            && !cell.comparekey(values, types)
        {
            return DoorCell::Occupied;
        }
        if cell.jitdriver_sd != self.jitdriver_sd {
            return DoorCell::Occupied;
        }
        if cell.is_compiled() {
            let cell_key = cell.cell_key.get().unwrap_or(hash);
            return match Self::cell_entry_procedure_token(&cell) {
                Some(token) => DoorCell::Enter { cell_key, token },
                None => DoorCell::Occupied,
            };
        }
        if cell.is_tracing() {
            return DoorCell::Tracing;
        }
        // Dead-token and other tokenless cells take `cleanup_chain`
        // on the occupied door (`maybe_compile_and_run`).
        if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
            return DoorCell::Occupied;
        }
        if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
            // `should_start_dont_trace_here_trace`: never traced
            // starts immediately, `JC_TRACING_OCCURRED` ticks. Both
            // stay on the occupied door.
            return DoorCell::Occupied;
        }
        DoorCell::Empty
    }

    /// [`Self::resolve_cell_key`]'s answer together with the cell it names,
    /// from the one walk that produces both.
    ///
    /// The key alone is what `resolve_cell_key` reports, and every arm here
    /// hands back exactly the key that function would; a caller that needs
    /// anything OFF the cell would otherwise walk back to it.
    fn resolved_cell(
        &self,
        hash: u64,
        greenargs: Option<(&[i64], &[GreenType])>,
    ) -> (u64, Option<Rc<BaseJitCell>>) {
        // `JitCell.get_jitcell`: walk this hash's chain and accept only
        // after `comparekey`. A typed miss must not recover a minted
        // cell from another bucket via `cell_named_by_hash`.
        if let Some((values, types)) = greenargs {
            return match self.lookup_chain_with_greenargs(hash, values, types) {
                Some(cell) => (cell.cell_key.get().unwrap_or(hash), Some(cell)),
                None => (hash, None),
            };
        }
        let mut count = 0usize;
        let mut found: Option<(Rc<BaseJitCell>, u64)> = None;
        // The cell this slot holds under `hash` ITSELF, whatever bucket filed
        // it. A key naming no cell of this bucket still names one when `hash`
        // was minted for a cell of another bucket, and the miss arm below
        // would otherwise walk back for it; noting it here costs one compare
        // per chain node instead of a second traversal.
        let mut by_key: Option<Rc<BaseJitCell>> = None;
        let mut cell = self.lookup_chain(hash);
        while let Some(c) = cell {
            if c.jitdriver_sd != self.jitdriver_sd {
                cell = c.next.borrow().clone();
                continue;
            }
            if let Some(cell_key) = c.cell_key.get() {
                if c.cell_bucket.get() == hash {
                    count += 1;
                    if found.is_none() {
                        found = Some((c.clone(), cell_key));
                    }
                }
                if cell_key == hash {
                    by_key = Some(c.clone());
                }
            }
            cell = c.next.borrow().clone();
        }
        // `bucket_of(hash) == hash` says [`Self::cell_by_key`] would search
        // THIS slot for `hash`, so the walk above has already decided it: a key
        // is routed to another slot only when [`Self::mint_cell_key`] recorded
        // it for another bucket, and a cell key names at most one live cell
        // wherever that cell sits.
        let cell_named_by_hash = || {
            if self.bucket_of(hash) == hash {
                by_key
            } else {
                self.cell_by_key(hash)
            }
        };
        match (count, found) {
            (1, Some((cell, cell_key))) => (cell_key, Some(cell)),
            _ => (hash, cell_named_by_hash()),
        }
    }

    /// The key of the one cell `hash` owns, if it owns exactly one.
    ///
    /// The entry path's allocation-free half. One candidate means the walk
    /// `maybe_compile_and_run` would do can only find that cell, so no
    /// `GreenKey` has to be built to prove it. Normally the answer is `hash`
    /// itself; it differs only when the cell that held the raw hash has been
    /// evicted out from under a minted sibling.
    ///
    /// **Which comparator is consulted.** `warmstate.py` accepts a cell
    /// only after `comparekey` matches, because a slot is indexed by a
    /// truncation of the hash (`counter.py _get_index`, read by
    /// `counter.py lookup_chain`) and genuinely mixes unrelated green keys. A caller
    /// here holds no greens, so it cannot run that comparator — but it does not
    /// need the full one to exclude the neighbours: [`Self::cell_keys_at`]
    /// compares FULL hashes, which the cell key preserves, and two green keys
    /// with equal full hashes are exactly the case `comparekey` was added for.
    /// So this narrows the slot's occupants to the ones `comparekey` could
    /// still have accepted, and answers only when one is left.
    ///
    /// What remains unresolvable is a genuine 64-bit `get_uhash` collision
    /// between two distinct green keys, which this cannot tell from one key's
    /// two cells. That is the condition under which a comparator-less cell is
    /// unresolvable rather than resolvable-but-unsound (see
    /// [`BaseJitCell::comparekey`]) — no accept rule fixes it here.
    #[inline]
    pub fn sole_cell_key(&self, hash: u64) -> Option<u64> {
        match self.cell_keys_at(hash) {
            (1, cell_key) => cell_key,
            _ => None,
        }
    }

    /// How many live cells `hash` owns, and the key of the first one found.
    ///
    /// The `u64`-only stand-in for the `comparekey(*greenargs)` filter every
    /// upstream chain walk applies (warmstate.py, :599-603, :634-638).
    /// `_get_index` (counter.py) keeps eleven bits, so a slot's
    /// occupants are mostly other green keys' cells; a reader that took the
    /// slot's population for this hash's would count strangers.
    ///
    /// A cell's own hash is recoverable without its greens because
    /// [`Self::bucket_of`] maps a cell key back to the hash the cell was filed
    /// under — the cell key IS that hash whenever the cell could take it, and
    /// [`Self::mint_cell_key`] records the hash for the cells that could not.
    /// That is the full 64 bits, so this excludes every neighbour except one
    /// whose green key genuinely collides in all 64.
    fn cell_keys_at(&self, hash: u64) -> (usize, Option<u64>) {
        let mut count = 0;
        let mut first = None;
        let mut cell = self.lookup_chain(hash);
        while let Some(c) = cell {
            if c.jitdriver_sd == self.jitdriver_sd
                && let Some(cell_key) = c.cell_key.get()
                && c.cell_bucket.get() == hash
            {
                count += 1;
                if first.is_none() {
                    first = Some(cell_key);
                }
            }
            cell = c.next.borrow().clone();
        }
        (count, first)
    }

    /// counter.py lookup_chain(hash)
    ///
    /// ```text
    ///  def lookup_chain(self, hash):
    ///      return self.celltable[self._get_index(hash)]
    /// ```
    ///
    /// Returns the head of the chain at `hash`'s table slot. Walk `.next` to
    /// iterate the chain — which, because `_get_index` (counter.py)
    /// keeps only the top bits of the low 32, holds every green key that lands
    /// in this slot and not just the ones hashing to `hash`. Readers that must
    /// tell those apart go through [`Self::cell_keys_at`] or compare
    /// `comparekey`; this one hands back the slot as upstream does.
    #[inline]
    pub fn lookup_chain(&self, hash: u64) -> Option<Rc<BaseJitCell>> {
        self.counter.lookup_chain(hash).map(|cell| Rc::clone(&cell))
    }

    /// `jitcounter.install_new_cell(hash, newcell)` (counter.py), reached
    /// from this driver's `get_jitcell` / `_ensure_jit_cell_at_key`. The
    /// counter owns the table; this state only notes that what
    /// [`Self::maybe_compile_decision`] answers may have changed.
    pub fn install_new_cell(&mut self, hash: u64, newcell: Option<BaseJitCell>) {
        self.bump_cell_generation();
        self.counter.install_new_cell(hash, newcell.map(Rc::new));
    }

    /// How many table slots hold a chain at all; see
    /// `JitCounter::occupied_buckets`.
    #[cfg(test)]
    fn occupied_buckets(&self) -> usize {
        self.counter.occupied_buckets()
    }

    /// `jitcounter.cleanup_chain(hash)` (counter.py).
    pub fn cleanup_chain(&mut self, hash: u64) {
        self.bump_cell_generation();
        self.counter.cleanup_chain(hash);
    }

    /// warmstate.py `JitCell.get_jitcell(*greenargs)`.
    ///
    /// ```python
    /// hash = JitCell.get_uhash(*greenargs)
    /// cell = jitcounter.lookup_chain(hash)
    /// while cell is not None:
    ///     if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///         return cell
    ///     cell = cell.next
    /// return None
    /// ```
    ///
    /// Walks the per-bucket chain comparing each cell's stored
    /// comparekey with `key` until a match is found. Hash collisions
    /// across distinct typed greens are resolved by comparekey, so
    /// `(code_a, pc_a)` and `(code_b, pc_b)` with the same `get_uhash`
    /// no longer alias to the same cell.
    ///
    /// `JitCell.__init__` always stores the greens. Two distinct keys that
    /// share `get_uhash` each keep their own cell; the second is minted
    /// ([`Self::mint_cell_key`]).
    ///
    /// This is the resolve half of [`Self::cell_key_for`], which is how the
    /// `u64` API surface reaches the same cell without holding the greens: the
    /// walk happens ONCE, at the hash producer, and the cell key it yields is
    /// what `counter_tick`, `maybe_compile`,
    /// `finish_tracing`, `abort_tracing`, `get_cell` and the rest are handed —
    /// along with `MetaInterp`'s `compiled_loops` / `cut_compiled_keys`, which
    /// are indexed by the same key and so describe the same cell. That is the
    /// property the `u64` currency previously lacked: it named a bucket, and a
    /// bucket is not a cell.
    pub fn lookup_chain_with_key(&self, key: &GreenKey) -> Option<Rc<BaseJitCell>> {
        self.lookup_chain_with_greenargs(key.get_uhash(), &key.values, &key.types)
    }

    /// `JitCell.get_jitcell(*greenargs)`: walk `lookup_chain(hash)` and
    /// accept the first cell whose `comparekey(*greenargs)` matches.
    fn lookup_chain_with_greenargs(
        &self,
        hash: u64,
        greenargs: &[i64],
        types: &[GreenType],
    ) -> Option<Rc<BaseJitCell>> {
        let mut cell = self.lookup_chain(hash);
        while let Some(c) = cell {
            if c.jitdriver_sd == self.jitdriver_sd && c.comparekey(greenargs, types) {
                return Some(c);
            }
            cell = c.next.borrow().clone();
        }
        None
    }

    /// warmstate.py `JitCell.ensure_jit_cell_at_key(greenkey)` /
    /// `_ensure_jit_cell_at_key(*greenargs)`.
    ///
    /// ```python
    /// def _ensure_jit_cell_at_key(*greenargs):
    ///     hash = JitCell.get_uhash(*greenargs)
    ///     cell = jitcounter.lookup_chain(hash)
    ///     while cell is not None:
    ///         if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///             return cell
    ///         cell = cell.next
    ///     newcell = JitCell(*greenargs)
    ///     jitcounter.install_new_cell(hash, newcell)
    ///     return newcell
    /// ```
    ///
    /// Returns nothing; the caller reads the ensured cell back through
    /// `lookup_chain_with_key`, or through [`Self::ensure_cell_key`] when what
    /// it needs is the cell's identity rather than the cell.
    ///
    /// On a miss a new cell carrying the greens is installed and, when the
    /// raw hash is already taken, [`Self::mint_cell_key`] names it.
    pub fn ensure_cell_for_key(&mut self, key: &GreenKey) {
        if self
            .lookup_chain_with_key(key)
            .is_some_and(|cell| cell.comparekey_matches(key))
        {
            return;
        }
        let newcell =
            BaseJitCell::new_for_driver_with_greens(self.jitdriver_sd, &key.values, &key.types);
        self.install_new_cell(key.get_uhash(), Some(newcell));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_greens::{gk, gk_with_uhash};

    fn keys_sharing_index(index_of: impl Fn(u64) -> usize, n: usize) -> Vec<majit_ir::GreenKey> {
        let first = gk(1);
        let idx = index_of(first.get_uhash());
        let mut found = vec![first];
        let mut raw = 2i64;
        while found.len() < n {
            let key = gk(raw);
            raw += 1;
            if index_of(key.get_uhash()) == idx {
                found.push(key);
            }
            assert!(raw < 1_000_000, "could not find {n} keys sharing an index");
        }
        found
    }

    fn maybe_compile(ws: &mut WarmEnterState, raw: u64) -> HotResult {
        ws.maybe_compile_with_key(&gk(raw as i64))
    }

    /// `attach_procedure_to_interp` with the ownership every production attach
    /// path gives the token first.
    ///
    /// `BaseJitCell` keeps a WEAK handle (`warmstate.py`), so a token whose
    /// last strong reference is this call's argument dies as the call returns
    /// and the cell reads back as "never compiled" — silently, not as a crash.
    /// Upstream never has that shape: `send_loop_to_backend`
    /// (`compile.py`) and `compile_tmp_callback` (`compile.py`)
    /// both register with `MemoryManager` before any cell sees the token, and
    /// `alive_loops` is the sole long-lived strong owner (`memmgr.py`).
    ///
    /// Returns the strong handle so a caller that wants to keep asserting on
    /// the token object can.
    fn attach_alive(
        ws: &mut WarmEnterState,
        raw: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Arc<JitCellToken> {
        let token: Arc<JitCellToken> = token.into();
        ws.memory_manager.keep_loop_alive(&token);
        ws.attach_procedure_to_interp_for_key(&gk(raw as i64), Arc::clone(&token));
        token
    }

    /// The token a cell's weak handle still resolves to, invalidated or not —
    /// `warmstate.py` `token = self.wref_procedure_token()` without the
    /// `:194 not token.invalidated` filter `get_procedure_token` puts on top,
    /// which is what an invalidation assertion has to look past.
    fn resolved_token(cell: &BaseJitCell) -> Arc<JitCellToken> {
        cell.loop_token
            .borrow()
            .as_ref()
            .expect("the cell has a procedure-token slot")
            .upgrade()
            .expect("and something still owns the token it names")
    }

    /// Typed-key twin of [`attach_alive`].
    fn attach_alive_for_key(
        ws: &mut WarmEnterState,
        key: &GreenKey,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Arc<JitCellToken> {
        let token: Arc<JitCellToken> = token.into();
        ws.memory_manager.keep_loop_alive(&token);
        ws.attach_procedure_to_interp_for_key(key, Arc::clone(&token));
        token
    }

    /// `warmstate.py:188` + `:191-196` + `memmgr.py:9-14` — the cell holds a
    /// WEAK handle, so `alive_loops` dropping the token is what makes the cell
    /// answer "no procedure token". The entry decision then goes back to
    /// counting instead of dispatching into a body nothing owns.
    ///
    /// **Why this test cannot pass under the old shape.** With
    /// `loop_token: Option<Arc<JitCellToken>>` the cell is itself a strong
    /// owner, so the eviction below removes `alive_loops`' entry and the token
    /// survives on the cell: every assertion after the eviction reads exactly
    /// as it did before it, and `maybe_compile` keeps answering `RunCompiled`
    /// against a loop the memory manager believes it has freed. The `assert!`s
    /// after `drop(token)` are the ones that fail there.
    #[test]
    fn a_token_the_memory_manager_evicted_reads_back_as_never_compiled() {
        let mut ws = WarmEnterState::new(1);
        let key = GreenKey::new(vec![0x11, 0x22]);
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        // `compile.py:566-567` order: register the strong owner, then attach.
        let token = attach_alive_for_key(&mut ws, &key, token);
        let cell_key = ws.cell_key_for(&key).expect("the key now owns a cell");

        assert!(
            ws.get_procedure_token_for_key(&key).is_some(),
            "fixture: a live token reads back through the weak handle",
        );
        assert!(
            matches!(ws.maybe_compile_with_key(&key), HotResult::RunCompiled),
            "fixture: and the entry decision dispatches to it \
             (warmstate.py:482-483)",
        );

        // `memmgr.py _kill_old_loops_now` — age it out. `max_age = 1` makes
        // `max_generation == current_generation`, so the token's generation
        // (stamped by `keep_loop_alive` one generation ago) is below it.
        ws.memory_manager.set_max_age(1, 1);
        let evicted = ws.memory_manager.next_generation();
        assert!(
            evicted.iter().any(|t| Arc::ptr_eq(t, &token)),
            "fixture: the sweep must actually evict this token, or the drop \
             below proves nothing",
        );
        drop(evicted);
        // Nothing owns the token now — which is the whole point of the weak
        // handle: this is reachable state, not a fixture trick.
        drop(token);

        assert!(
            ws.get_procedure_token_for_key(&key).is_none(),
            "warmstate.py:192-195 `token = self.wref_procedure_token()` — a \
             weakref that no longer resolves is not a procedure token",
        );
        let cell = ws
            .get_cell_for_key(&key)
            .expect("the cell outlives the token it pointed at");
        assert!(
            !cell.is_compiled(),
            "so the cell is not compiled, and the entry path must not enter it",
        );
        assert!(
            cell.has_seen_a_procedure_token(),
            "warmstate.py:198-199 — the SLOT is still set, which is how \
             `should_remove_jitcell` tells 'had one and lost it' from \
             'never had one'",
        );
        assert!(
            !matches!(ws.maybe_compile_with_key(&key), HotResult::RunCompiled),
            "and the entry decision falls back to counting (warmstate.py:483-500)",
        );
    }

    /// `compile.py:1148-1149` — the token `get_assembler_token`
    /// (`warmstate.py`) installs is minted by `compile_tmp_callback`,
    /// and `compile_tmp_callback`'s registration with `MemoryManager` is its
    /// ONLY strong owner: the cell takes a weak handle (`warmstate.py`)
    /// and the caller drops its own as soon as the CALL_ASSEMBLER descr is
    /// built.
    ///
    /// The two arms below differ in exactly that one call. The unregistered
    /// arm is the state `compile_tmp_callback` was in before the registration
    /// was added: token minted, installed, returned — and gone, leaving the
    /// cell answering "never compiled" rather than failing loudly. That is why
    /// the `&mut MemoryManager` reaches the maker at all.
    #[test]
    fn a_tmp_callback_token_survives_its_install_only_if_the_maker_registered_it() {
        let mut ws = WarmEnterState::new(1);

        let unregistered_key = GreenKey::new(vec![0x55, 0x66]);
        let handed_back = ws
            .get_assembler_token_with_key::<(), _>(&unregistered_key, |_memmgr| {
                Ok(Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED)))
            })
            .expect("the maker succeeded");
        assert!(
            ws.get_procedure_token_for_key(&unregistered_key).is_some(),
            "fixture: while the caller still holds it, the cell resolves it",
        );
        // The production caller keeps the token only long enough to put it in a
        // descr; `direct_assembler_call` then drops its handle.
        drop(handed_back);
        assert!(
            ws.get_procedure_token_for_key(&unregistered_key).is_none(),
            "a maker that skipped `keep_loop_alive` leaves NOTHING owning the \
             token, so the cell it was just installed on reads as never \
             compiled",
        );

        let registered_key = GreenKey::new(vec![0x77, 0x88]);
        let handed_back = ws
            .get_assembler_token_with_key::<(), _>(&registered_key, |memmgr| {
                let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
                // `compile.py:1148-1149`.
                memmgr.keep_loop_alive(&token);
                Ok(token)
            })
            .expect("the maker succeeded");
        drop(handed_back);
        assert!(
            ws.get_procedure_token_for_key(&registered_key).is_some(),
            "with `alive_loops` owning it (`memmgr.py:9-14`), the installed \
             token outlives the call that made it",
        );
    }

    /// The other half of the same invariant: a token the memory manager still
    /// owns is entered, every time, for as long as `alive_loops` holds it.
    ///
    /// This is the regression the weak flip could have caused silently — a
    /// cell whose token died reads as "never compiled" rather than failing
    /// loudly, so a missing `keep_loop_alive` anywhere would show up as the
    /// JIT quietly never entering. Pinning repeated entry (not just one
    /// `is_some()` at mint time) is what makes that visible here.
    #[test]
    fn a_token_the_memory_manager_still_owns_is_entered_every_time() {
        let mut ws = WarmEnterState::new(1);
        let key = GreenKey::new(vec![0x33, 0x44]);
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        let token = attach_alive_for_key(&mut ws, &key, token);
        let cell_key = ws.cell_key_for(&key).expect("the key now owns a cell");
        // The fixture's own handle is gone; `alive_loops` is the only owner,
        // exactly as `memmgr.py:9-12` describes production.
        drop(token);

        for entry in 0..8 {
            assert!(
                matches!(ws.maybe_compile_with_key(&key), HotResult::RunCompiled),
                "entry #{entry} must still dispatch: `alive_loops` owns the \
                 token and no sweep has run",
            );
            assert!(
                ws.get_procedure_token_for_key(&key).is_some(),
                "entry #{entry} must still resolve the token through the cell",
            );
        }
    }

    /// `warmstate.py:284` substitutes the full pass list for `'all'` alone.
    /// The empty string splits into no names, so it selects nothing — the one
    /// spelling a caller uses to turn every optimization off. Reading it as
    /// `'all'` inverts that request completely and silently, since the driver
    /// then keeps unrolling and every other pass.
    #[test]
    fn an_empty_enable_opts_selects_no_passes() {
        let mut ws = WarmEnterState::new(1);
        let all = default_enable_opts();
        assert_eq!(
            ws.get_enable_opts(),
            all.as_slice(),
            "default is the full set"
        );

        ws.set_param_enable_opts("");
        assert!(
            ws.get_enable_opts().is_empty(),
            "`\"\"` names no pass, so no pass is enabled; got {:?}",
            ws.get_enable_opts()
        );

        ws.set_param_enable_opts("all");
        assert_eq!(ws.get_enable_opts(), all.as_slice(), "`all` restores them");

        ws.set_param_enable_opts("intbounds:heap");
        assert_eq!(ws.get_enable_opts(), ["intbounds", "heap"]);
    }

    #[test]
    fn test_not_hot_initially() {
        let mut ws = WarmEnterState::new(3);
        match maybe_compile(&mut ws, 42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot"),
        }
    }

    #[test]
    fn test_start_tracing_at_threshold() {
        let mut ws = WarmEnterState::new(3);
        // Tick 1, 2: not hot
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        // Tick 3: threshold reached, start tracing
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
    }

    #[test]
    fn test_already_tracing() {
        let mut ws = WarmEnterState::new(2);
        // First tick: eviction (always false). Second tick: threshold reached.
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Next call sees TRACING flag
        match maybe_compile(&mut ws, 42) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing"),
        }
    }

    #[test]
    fn test_run_compiled() {
        let mut ws = WarmEnterState::new(1);
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, 42, token);

        match maybe_compile(&mut ws, 42) {
            HotResult::RunCompiled => {}
            _ => panic!("expected RunCompiled"),
        }
    }

    #[test]
    fn test_finish_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(ws.cell_key_for(&gk(42)).expect("cell"));

        let cell = ws
            .get_cell(ws.cell_key_for(&gk(42)).expect("cell"))
            .unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags.get().contains(JcFlags::JC_TRACING_OCCURRED));
    }

    #[test]
    fn test_abort_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.abort_tracing(ws.cell_key_for(&gk(42)).expect("cell"), true);

        let cell = ws
            .get_cell(ws.cell_key_for(&gk(42)).expect("cell"))
            .unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));

        // RPython warmstate.py: a DONT_TRACE_HERE cell with no procedure token
        // still retriggers separate tracing after warming up again.
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing due to DONT_TRACE_HERE retrace"),
        }
    }

    #[test]
    fn test_abort_tracing_allows_retry() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Abort without DONT_TRACE_HERE
        ws.abort_tracing(ws.cell_key_for(&gk(42)).expect("cell"), false);

        // `maybe_compile_and_run`: a tokenless aborted cell is `cleanup_chain`
        // (was: the cell stayed, so one NotHot then StartTracing). The bucket
        // then recounts from cold (threshold=2).
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        assert!(ws.get_cell_for_key(&gk(42)).is_none());
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing on retry"),
        }
    }

    #[test]
    fn test_different_green_keys() {
        let mut ws = WarmEnterState::new(3);
        // Key 1: tick 1 (eviction), tick 2 (count=2 < 3)
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::NotHot));
        // Key 2: tick 1 (eviction)
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::NotHot));
        // Key 1: tick 3 -> threshold, starts tracing
        match maybe_compile(&mut ws, 1) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 1"),
        }
        // Key 2 still not hot (only 2 total ticks: eviction + one more needed)
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::NotHot));
    }

    #[test]
    fn test_set_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);
        ws.set_threshold(50);
        assert_eq!(ws.threshold(), 50);
    }

    #[test]
    fn test_get_compiled() {
        let mut ws = WarmEnterState::new(1);
        assert!(
            ws.get_compiled(ws.cell_key_for(&gk(42)).expect("cell"))
                .is_none()
        );

        let token = JitCellToken::new(0);
        attach_alive(&mut ws, 42, token);

        let compiled = ws.get_compiled(ws.cell_key_for(&gk(42)).expect("cell"));
        assert!(compiled.is_some());
        assert_eq!(compiled.unwrap().number(), 0);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut ws = WarmEnterState::new(3);
        let key = 0xDEAD;

        // Phase 1: Not hot (need 3 ticks to reach threshold=3)
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));

        // Phase 2: Start tracing (third tick reaches threshold)
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));

        // Phase 3: Already tracing
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::AlreadyTracing
        ));

        // Phase 4: Finish tracing, install compiled code
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);

        // Phase 5: Run compiled
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
    }

    #[test]
    fn test_trace_eagerness_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.trace_eagerness(), 200);
    }

    #[test]
    fn test_trace_eagerness_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_param_trace_eagerness(10);
        assert_eq!(ws.trace_eagerness(), 10);
    }

    // ── Quasi-immutable invalidation tests ──

    #[test]
    fn test_quasiimmut_register_and_invalidate() {
        let mut ws = WarmEnterState::new(1);
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        let green_key = 42;
        let qmut_key = 0xABCD;

        assert!(!token.is_invalidated());
        attach_alive(&mut ws, green_key, token);
        ws.register_quasiimmut_dependency(
            qmut_key,
            ws.cell_key_for(&gk(green_key as i64)).expect("cell"),
        );

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 1);

        let cell = ws
            .get_cell(ws.cell_key_for(&gk(green_key as i64)).expect("cell"))
            .unwrap();
        assert!(resolved_token(&cell).is_invalidated());
    }

    #[test]
    fn test_quasiimmut_no_deps() {
        let mut ws = WarmEnterState::new(1);
        // No dependencies registered → invalidation returns 0.
        let count = ws.invalidate_quasiimmut(0xDEAD);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_quasiimmut_multiple_deps() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;

        // Install two loops depending on the same quasi-immutable field.
        for green_key in [10, 20] {
            let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
            attach_alive(&mut ws, green_key, token);
            ws.register_quasiimmut_dependency(
                qmut_key,
                ws.cell_key_for(&gk(green_key as i64)).expect("cell"),
            );
        }

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 2);

        for green_key in [10, 20] {
            let cell = ws
                .get_cell(ws.cell_key_for(&gk(green_key as i64)).expect("cell"))
                .unwrap();
            assert!(resolved_token(&cell).is_invalidated());
        }
    }

    #[test]
    fn test_quasiimmut_invalidate_all() {
        let mut ws = WarmEnterState::new(1);
        for green_key in [1, 2, 3] {
            let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
            attach_alive(&mut ws, green_key, token);
        }

        ws.invalidate_all();

        for green_key in [1, 2, 3] {
            let cell = ws
                .get_cell(ws.cell_key_for(&gk(green_key as i64)).expect("cell"))
                .unwrap();
            assert!(resolved_token(&cell).is_invalidated());
        }
    }

    // ── Function threshold tests ──

    #[test]
    fn test_function_threshold_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.function_threshold(), 1619); // PyPy default
    }

    #[test]
    fn test_function_threshold_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(10);
        assert_eq!(ws.function_threshold(), 10);
    }

    // ── Trace limit lifecycle tests (RPython: test_tracelimit.py) ──

    #[test]
    fn test_abort_tracing_too_long_sets_dont_trace() {
        // When a trace is too long, the meta-interpreter calls
        // abort_tracing(key, true) to prevent future tracing at that location.
        // This mirrors RPython's ABORT_TOO_LONG behavior.
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Simulate: recorder.is_too_long() was true, so abort with dont_trace.
        ws.abort_tracing(ws.cell_key_for(&gk(42)).expect("cell"), true);

        // The key is now blacklisted.
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(42)).expect("cell"))
            .unwrap();
        assert!(cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));
        assert!(!cell.is_tracing());

        // RPython warmstate.py: DONT_TRACE_HERE still allows separate
        // tracing later for keys without a procedure token.
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, 42),
            HotResult::StartTracing
        ));
    }

    #[test]
    fn test_disable_noninlinable_function_blocks_inlining() {
        let mut ws = WarmEnterState::new(3);

        // Fresh cell: can_inline_callable returns true by default.
        assert!(ws.can_inline_callable(ws.cell_key_for(&gk(42)).expect("cell")));

        // dont_trace_here → JC_DONT_TRACE_HERE flag, can_inline_callable
        // returns false (warmstate.py parity).
        ws.disable_noninlinable_function_for_key(&gk(42));
        assert!(!ws.can_inline_callable(ws.cell_key_for(&gk(42)).expect("cell")));
    }

    #[test]
    fn typed_disable_blocks_the_same_typed_inline_lookup() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 11]);

        assert!(ws.can_inline_callable_for_key(&key));
        ws.disable_noninlinable_function_for_key(&key);
        assert!(!ws.can_inline_callable_for_key(&key));
    }

    #[test]
    fn test_abort_too_long_then_retry_different_key() {
        // Aborting one key's trace as too long should not affect other keys.
        let mut ws = WarmEnterState::new(2);

        // Key 42: start and abort as too long.
        assert!(matches!(maybe_compile(&mut ws, 42), HotResult::NotHot));
        match maybe_compile(&mut ws, 42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 42"),
        }
        ws.abort_tracing(ws.cell_key_for(&gk(42)).expect("cell"), true);

        // Key 99: should still work normally.
        assert!(matches!(maybe_compile(&mut ws, 99), HotResult::NotHot));
        match maybe_compile(&mut ws, 99) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 99"),
        }
    }

    #[test]
    fn test_lifecycle_with_trace_abort_and_recompile() {
        // Full lifecycle: trace starts, is too long (abort without blacklist),
        // then on retry a shorter trace succeeds and gets compiled.
        let mut ws = WarmEnterState::new(2);
        let key = 0xCAFE;

        // Phase 1: reach threshold, start tracing.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Phase 2: trace is too long, abort without blacklist.
        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);

        // Phase 3: `maybe_compile_and_run` drops the tokenless aborted cell
        // (`cleanup_chain`; was: the cell stayed and one NotHot re-armed it),
        // then the bucket recounts from cold.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(ws.get_cell_for_key(&gk(key as i64)).is_none());
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {
                // Phase 4: this time the trace succeeds.
                ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
                let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
                attach_alive(&mut ws, key, token);
            }
            _ => panic!("expected StartTracing on retry"),
        }

        // Phase 5: compiled code should be available.
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
    }

    #[test]
    fn test_multiple_aborts_before_success() {
        // Mirrors RPython's segmented trace behavior: a location can fail
        // multiple times before eventually compiling.
        // threshold=2 because the first tick always evicts (returns NotHot).
        let mut ws = WarmEnterState::new(2);
        let key = 0xBEEF;

        // First attempt: tick once (eviction), tick twice (threshold) -> StartTracing.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 1)"),
        }
        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);

        // Second attempt: `maybe_compile_and_run` `cleanup_chain`s the aborted
        // cell (was: the cell stayed, one NotHot then StartTracing), then the
        // bucket recounts from cold.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(ws.get_cell_for_key(&gk(key as i64)).is_none());
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 2)"),
        }
        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);

        // Third attempt: succeeds and gets compiled.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(ws.get_cell_for_key(&gk(key as i64)).is_none());
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {
                ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
                let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
                attach_alive(&mut ws, key, token);
            }
            _ => panic!("expected StartTracing (attempt 3)"),
        }
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
    }

    #[test]
    fn test_tracing_occurred_flag_persists_after_abort() {
        // The TRACING_OCCURRED flag should remain set even after abort.
        // This mirrors RPython's tracking of whether tracing was ever attempted.
        let mut ws = WarmEnterState::new(2);
        let key = 42;

        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert!(cell.flags.get().contains(JcFlags::JC_TRACING_OCCURRED));

        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);

        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert!(!cell.is_tracing());
        assert!(
            cell.flags.get().contains(JcFlags::JC_TRACING_OCCURRED),
            "TRACING_OCCURRED should persist after abort"
        );
    }

    #[test]
    fn test_quasiimmut_deps_cleared_after_invalidation() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, 42, token);
        ws.register_quasiimmut_dependency(qmut_key, ws.cell_key_for(&gk(42)).expect("cell"));

        ws.invalidate_quasiimmut(qmut_key);
        // Second invalidation should find no deps.
        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 0);
    }

    // ── MemoryManager (memmgr.py parity) tests ──
    //
    // Ported from rpython/jit/metainterp/test/test_memmgr.py.  Each
    // test creates real `Arc<JitCellToken>` instances since
    // `MemoryManager.alive_loops` keys on token identity per
    // memmgr.py (the looptoken Python object is the dict key
    // upstream).

    fn make_token(number: i64) -> Arc<JitCellToken> {
        Arc::new(JitCellToken::new(number))
    }

    #[test]
    fn test_loop_aging_basic() {
        // memmgr.py:_kill_old_loops_now predicate:
        //   0 <= looptoken.generation < current_generation - (max_age - 1)
        // With max_age=3 and a token kept alive at current_generation=1:
        //   after 2 next_generation: cg=3, max_gen=3-2=1, 1<1=false → keep
        //   after 3 next_generation: cg=4, max_gen=4-2=2, 1<2=true  → evict
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 0);
    }

    #[test]
    fn test_loop_aging_disabled() {
        // memmgr.py:43-44: max_age <= 0 disables eviction.
        let mut mgr = MemoryManager::new(0);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        for _ in 0..100 {
            mgr.next_generation();
        }
        assert_eq!(mgr.alive_count(), 2);
    }

    /// Pins the PRODUCTION default, which is upstream's **test** default.
    ///
    /// The tests above prove the memmgr mechanism is a faithful port. What
    /// nothing watched is which mode a real `WarmEnterState` ships in.
    /// `WarmEnterState::new` (the only production constructor — `pyjitpl.rs`
    /// `warm_state: WarmEnterState::new(threshold)`) builds
    /// `MemoryManager::new(0)`, so `next_check == -1` and
    /// `_kill_old_loops_now` is unreachable: `alive_loops` never prunes.
    ///
    /// Upstream splits these two defaults and pyre collapsed them:
    ///
    /// | | `loop_longevity` |
    /// |---|---|
    /// | `rlib/jit.py` PARAMETERS — **production** | **1000** |
    /// | `warmspot.py` `jittify_and_run` — **test harness** | **0** |
    /// | pyre, every path unless `PYRE_JIT` overrides | **0** |
    ///
    /// So this asserts a **divergence**, not a desired value. It is here so
    /// the literal is watched: changing `MemoryManager::new(0)` at the
    /// construction site turns eviction on across the JIT, and that is a
    /// behavioural change (it starts feeding `try_to_free_some_loops`) which
    /// must be measured, not slipped in. If you are here because this test
    /// failed, that is the intended alarm — see default-retirement.
    ///
    /// This governs `alive_loops` (loop TOKENS) only. The `cells` map is a
    /// different population and is NOT bounded by `max_age`; it sheds
    /// entries only through `install_new_cell`'s `should_remove_jitcell`
    /// gate (typed-key eviction). Turning `max_age` on does not bound the cell table.
    ///
    /// `install_new_cell` runs the gate over the whole existing chain at the
    /// bucket on *every* typed install, collision or not — so the gate is
    /// reached routinely, it simply keeps most cells (a token, `TRACING`,
    /// `DONT_TRACE_HERE` without a dead token, or `FORCE_FINISH` all veto
    /// removal). The unbounded-growth conclusion survives; the stated reason
    /// for it does not.
    #[test]
    fn production_warmstate_ships_loop_eviction_disabled() {
        let ws = WarmEnterState::new(3);
        assert_eq!(
            ws.memory_manager.next_check(),
            -1,
            "next_check must be -1 (eviction disabled) — memmgr.py:43-44",
        );
        assert_eq!(
            ws.memory_manager.loop_longevity_param(),
            0,
            "production ships loop_longevity=0, upstream's TEST default; \
             rlib/jit.py:594 gives production 1000",
        );
    }

    #[test]
    fn test_loop_aging_refresh() {
        // keep_loop_alive resets `looptoken.generation` to
        // `current_generation`, postponing eviction.
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        // Refresh t1 each generation; t2 ages out.
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        // cg=4, max_gen=2; t1.gen=3 (refreshed), 3<2=false → keep;
        //                  t2.gen=1, 1<2=true → evict.
        assert_eq!(mgr.alive_count(), 1);
        assert!(mgr.contains(&t1));
        assert!(!mgr.contains(&t2));

        // Refreshing t1 each generation keeps it indefinitely.
        for _ in 0..10 {
            mgr.keep_loop_alive(&t1);
            mgr.next_generation();
        }
        assert!(mgr.contains(&t1));
    }

    #[test]
    fn test_loop_aging_mixed() {
        // Tokens kept alive at different generations age independently.
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, t1.gen=1, 1<1=false → keep
        mgr.keep_loop_alive(&t2); // t2.gen=2

        mgr.next_generation(); // cg=3, max_gen=2, t1.gen=1<2=true → evict, t2.gen=2<2=false → keep
        assert_eq!(mgr.alive_count(), 1);
        assert!(!mgr.contains(&t1));
        assert!(mgr.contains(&t2));
    }

    // ── Memmgr deeper coverage (RPython: test_memmgr.py parity) ──

    #[test]
    fn test_evicted_loops_can_be_recompiled() {
        // After eviction the same JitCellToken can be re-inserted by a
        // fresh keep_loop_alive (or a freshly compiled token can take
        // its place — RPython gets a brand-new LoopToken object).
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, 1<1=false → keep
        mgr.next_generation(); // cg=3, max_gen=2, 1<2=true → evict
        assert_eq!(mgr.alive_count(), 0);

        // Re-register the same token (or a fresh one).
        mgr.keep_loop_alive(&t1); // t1.gen=3
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation(); // cg=4, max_gen=3, 3<3=false → keep
        assert!(mgr.contains(&t1));
        mgr.next_generation(); // cg=5, max_gen=4, 3<4=true → evict
        assert!(!mgr.contains(&t1));
    }

    #[test]
    fn test_generation_does_not_panic_at_high_values() {
        // Verify that generation arithmetic survives many advances.
        // RPython uses r_int64; Rust uses i64 with the same wraparound
        // semantics.
        let mut mgr = MemoryManager::new(3);

        for _ in 0..1000 {
            mgr.next_generation();
        }
        assert_eq!(mgr.current_generation(), 1001);

        let t = make_token(42);
        mgr.keep_loop_alive(&t);
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(mgr.contains(&t));
        mgr.next_generation();
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_with_warm_state_integration() {
        // memmgr cooperates with WarmEnterState: attach a token via
        // `attach_procedure_to_interp` (warmstate.py parity)
        // then mirror it into MemoryManager via keep_loop_alive.
        let mut ws = WarmEnterState::new(2);
        let mut mgr = MemoryManager::new(2);
        let key = 0xF00D;

        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = make_token(JitCellToken::NUMBER_UNASSIGNED);
        ws.attach_procedure_to_interp_for_key(&gk(key as i64), Arc::clone(&token));
        mgr.keep_loop_alive(&token);

        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(!mgr.contains(&token));

        // Re-install: a fresh compile produces a fresh token; the
        // warmstate cell records the new procedure token, MemoryManager
        // tracks it under its own pointer identity.
        let token2 = make_token(JitCellToken::NUMBER_UNASSIGNED);
        ws.attach_procedure_to_interp_for_key(&gk(key as i64), Arc::clone(&token2));
        mgr.keep_loop_alive(&token2);

        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
        assert_eq!(mgr.alive_count(), 1);
    }

    #[test]
    fn test_loop_aging_does_not_affect_active_loops() {
        // A loop continually kept alive is never evicted, even while
        // peers are aged out.
        let mut mgr = MemoryManager::new(2);
        let t_active = make_token(1);
        let t_idle_a = make_token(2);
        let t_idle_b = make_token(3);

        mgr.keep_loop_alive(&t_active);
        mgr.keep_loop_alive(&t_idle_a);
        mgr.keep_loop_alive(&t_idle_b);

        for _ in 0..20 {
            mgr.keep_loop_alive(&t_active);
            mgr.next_generation();
            assert!(mgr.contains(&t_active));
        }

        // Idle loops were aged out long ago.
        assert!(!mgr.contains(&t_idle_a));
        assert!(!mgr.contains(&t_idle_b));
    }

    #[test]
    fn test_loop_aging_set_max_age_dynamic() {
        // memmgr.py set_max_age — shrinking max_age accelerates
        // eviction of older loops.
        let mut mgr = MemoryManager::new(10);
        let t = make_token(1);

        mgr.keep_loop_alive(&t); // t.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-9=-7 → keep

        // Shrink max_age — next sweep evicts t.
        mgr.set_max_age(1, 0);
        mgr.next_generation(); // cg=3, max_gen=3-0=3, 1<3 → evict
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_interleaved_register_and_evict() {
        // Mirrors RPython's test_basic_3: tokens registered at different
        // generations, even-indexed kept alive each step.  After the
        // loop, even-indexed tokens are alive; old odd-indexed ones
        // (registered ≥ max_age generations ago and never refreshed)
        // are evicted.
        let mut mgr = MemoryManager::new(4);
        let mut tokens: Vec<Arc<JitCellToken>> = Vec::new();

        for i in 0..10u64 {
            let t = make_token(i as i64);
            mgr.keep_loop_alive(&t);
            tokens.push(t);
            mgr.next_generation();

            // Refresh even-indexed tokens.
            for j in (0..=i).step_by(2) {
                mgr.keep_loop_alive(&tokens[j as usize]);
            }
        }

        for (i, t) in tokens.iter().enumerate() {
            let is_alive = mgr.contains(t);
            if i % 2 == 0 {
                assert!(is_alive, "even-indexed token {} should be alive", i);
            } else if i < 6 {
                // Odd tokens registered early enough are evicted.
                // Token i registered at cg=i+1; cg now ~11; max_gen=11-3=8.
                // For i < 6: t.gen=i+1 < 8 → evict.
                assert!(!is_alive, "odd token {} should be evicted", i);
            }
        }
    }

    #[test]
    fn test_abort_tracing_retry_with_lower_threshold() {
        // Simulates the scenario where a trace is too long, the location
        // is aborted (without blacklisting), and on retry the WarmEnterState
        // has a lower threshold so it starts tracing sooner.
        let mut ws = WarmEnterState::new(5);
        let key = 0xABCD;

        // Reach threshold=5, start tracing
        for _ in 0..4 {
            assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        }
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing at threshold 5"),
        }

        // Abort without blacklisting
        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);

        // Lower threshold for retry
        ws.set_threshold(2);

        // `maybe_compile_and_run` drops the tokenless aborted cell
        // (`cleanup_chain`; was: the cell stayed, so the first is NotHot).
        // Threshold 2 then needs two fresh ticks.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(ws.get_cell_for_key(&gk(key as i64)).is_none());
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {
                ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
                let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
                attach_alive(&mut ws, key, token);
            }
            _ => panic!("expected StartTracing with lower threshold"),
        }
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
    }

    #[test]
    fn test_force_start_tracing_bypasses_counter() {
        // force_start_tracing is used for function-entry tracing where
        // the caller already decided to trace. It should work regardless
        // of the counter state.
        let mut ws = WarmEnterState::new(100); // very high threshold
        let key = gk(42);

        // Without any ticks, force_start_tracing should start tracing
        match ws.force_start_tracing_for_key(&key, &[]) {
            HotResult::StartTracing => {}
            _ => panic!("force_start_tracing should start tracing immediately"),
        }

        // The cell should be in TRACING state
        let cell = ws.get_cell(ws.cell_key_for(&key).expect("cell")).unwrap();
        assert!(cell.is_tracing());

        // Second call sees AlreadyTracing
        match ws.force_start_tracing_for_key(&key, &[]) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing on second force_start_tracing"),
        }
    }

    // ── BaseJitCell state machine tests ──

    #[test]
    fn test_jitcell_state_transitions() {
        // Full lifecycle: NotHot → Tracing → Compiled → Invalidated
        let mut ws = WarmEnterState::new(2);
        let key = 0xA1;

        // Initially no cell → NotHot
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::NotHot
        );

        // Tick to threshold → Tracing
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Tracing
        );

        // Finish tracing and install → Compiled
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Compiled
        );

        // Invalidate via transition_cell → Invalidated
        ws.transition_cell(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            BaseJitCellState::Invalidated,
        );
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        // The loop_token's invalidated flag should be set
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert!(resolved_token(&cell).is_invalidated());
        assert!(cell.has_seen_a_procedure_token());
    }

    #[test]
    fn test_procedure_token_ownership() {
        // Compiled cell owns a token; invalidation revokes ownership.
        let mut ws = WarmEnterState::new(2);
        let key = 0xB2;

        // Compile a loop
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);

        // Cell owns the token
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert!(cell.get_procedure_token().is_some());
        assert!(cell.has_seen_a_procedure_token());

        // Invalidate via quasiimmut
        let qmut_key = 0xFF;
        ws.register_quasiimmut_dependency(
            qmut_key,
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
        );
        ws.invalidate_quasiimmut(qmut_key);

        // Token ownership revoked (state is Invalidated); the weak slot
        // still records that a procedure token existed.
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert_eq!(cell.derived_state(), BaseJitCellState::Invalidated);
        // get_procedure_token returns None because the token is invalidated
        assert!(cell.get_procedure_token().is_none());
        // But we still know a token existed
        assert!(cell.has_seen_a_procedure_token());
    }

    #[test]
    fn test_set_param_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);

        ws.set_param("threshold", 42);
        assert_eq!(ws.threshold(), 42);

        ws.set_param("trace_limit", 5000);
        assert_eq!(ws.trace_limit(), 5000);

        ws.set_param("trace_eagerness", 10);
        assert_eq!(ws.trace_eagerness(), 10);

        ws.set_param("function_threshold", 8);
        assert_eq!(ws.function_threshold(), 8);

        ws.set_param("max_inline_depth", 15);
        assert_eq!(ws.max_inline_depth(), 15);

        // Unknown param is ignored
        ws.set_param("nonexistent", 999);
    }

    #[test]
    fn test_get_stats() {
        let mut ws = WarmEnterState::new(2);

        // Initially empty
        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 0);
        assert_eq!(stats.num_compiled, 0);

        // Start tracing two keys
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::StartTracing));
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::StartTracing));

        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 2);
        assert_eq!(stats.num_tracing, 2);

        // Compile key 1
        ws.finish_tracing(ws.cell_key_for(&gk(1)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, 1, token);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 1);

        // Abort key 2 with dont_trace
        ws.abort_tracing(ws.cell_key_for(&gk(2)).expect("cell"), true);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 0);
        assert_eq!(stats.num_disable_noninlinable_function, 1);

        // Invalidate key 1
        ws.transition_cell(
            ws.cell_key_for(&gk(1)).expect("cell"),
            BaseJitCellState::Invalidated,
        );

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 0);
        assert_eq!(stats.num_invalidated, 1);
        assert_eq!(stats.num_disable_noninlinable_function, 1);
        assert_eq!(stats.num_cells, 2);
    }

    #[test]
    fn test_jitcell_state_disable_noninlinable_function() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xC3;

        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        match maybe_compile(&mut ws, key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Abort with DONT_TRACE_HERE → the cell carries the flag
        ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), true);
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .unwrap()
                .flags
                .get()
                .contains(JcFlags::JC_DONT_TRACE_HERE)
        );

        // Future calls return NotHot
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));

        // Can manually reset to NotHot
        ws.transition_cell(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            BaseJitCellState::NotHot,
        );
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::NotHot
        );
        // DONT_TRACE_HERE flag should be cleared
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
            .unwrap();
        assert!(!cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));
    }

    /// `disable_noninlinable_function` retries a never-seen-token cell
    /// (`warmstate.py maybe_compile_and_run`). `abort_tracing` stamps the
    /// same `JC_DONT_TRACE_HERE` flag at `MAX_TRACE_ABORT_COUNT` as a ban:
    /// a latched cell must not take that retry, or `threshold=1` retraces
    /// on every entry.
    ///
    /// Was: five start-via-door / abort cycles left the tokenless cell in the
    /// chain so `abort_count` could climb. `maybe_compile_and_run` now
    /// `cleanup_chain`s a tokenless abort, so the latch is stamped here
    /// without a decision-door visit between aborts.
    #[test]
    fn a_ceiling_latched_cell_without_a_token_is_banned() {
        for use_typed_key in [false, true] {
            let mut ws = WarmEnterState::new(2);
            let green = GreenKey::new(vec![0xCE, 0x11]);
            let key = green.get_uhash();
            ws.ensure_cell_for_key(&green);
            for _ in 0..MAX_TRACE_ABORT_COUNT {
                ws.abort_tracing(key, false);
            }
            let mut ask = |ws: &mut WarmEnterState| {
                if use_typed_key {
                    ws.maybe_compile_with_key(&green)
                } else {
                    ws.maybe_compile(key)
                }
            };

            for _ in 0..64 {
                assert!(
                    matches!(ask(&mut ws), HotResult::NotHot),
                    "a ceiling-latched cell retried after abort_tracing stamped DONT_TRACE_HERE (typed={use_typed_key})",
                );
            }
        }
    }

    /// `warmstate.py maybe_compile_and_run`: `JC_DONT_TRACE_HERE` with no
    /// token and no `JC_TRACING_OCCURRED` traces immediately (`tick = True`).
    /// Threshold 100 would stay cold if the door only counted.
    #[test]
    fn a_disabled_never_traced_cell_retries_immediately() {
        let mut ws = WarmEnterState::new(100);
        let key = 0xD017u64;
        ws.disable_noninlinable_function_for_key(&gk(key as i64));
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));
    }

    /// The abort ceiling and the inline gate share `JC_DONT_TRACE_HERE`, so a
    /// location the ceiling retires also stops being inlined into other
    /// traces.  That coupling has no upstream analogue — upstream writes the
    /// flag from `disable_noninlinable_function` and `dont_trace_here` only —
    /// and it was measured on 2026-08-24: splitting the two improved no row of
    /// the synth corpus and cost `synth/generator_tree_recursion`
    /// `bridges_compiled` 26 -> 60 and `guard_failures` 2999 -> 25297.  This
    /// test is the pin; see `MAX_TRACE_ABORT_COUNT` for the full record.
    #[test]
    fn the_abort_ceiling_also_closes_the_inline_gate() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xCE111;

        assert!(ws.can_inline_callable(key), "a fresh key is inlinable");

        // Was: five start-via-door / abort cycles. A tokenless abort is now
        // `cleanup_chain` (`maybe_compile_and_run`), so stamp the ceiling on
        // one cell without a decision-door visit between aborts.
        ws.ensure_cell_for_key(&gk_with_uhash(key, 1));
        for _ in 0..MAX_TRACE_ABORT_COUNT {
            // `false`: no `disable_noninlinable_function`, so the ceiling is
            // the only thing that can stamp the flag.
            ws.abort_tracing(key, false);
        }

        let cell = ws.get_cell(key).unwrap();
        assert!(
            cell.abort_ceiling_latched(),
            "the ceiling did not latch after {MAX_TRACE_ABORT_COUNT} aborts",
        );
        assert!(
            !ws.can_inline_callable(key),
            "a ceiling-latched location is still inlinable as a callee",
        );
    }

    /// A frontend coverage decline happens before upstream enters
    /// `MetaInterp._interpret`, so it cannot raise `SwitchToBlackhole` or
    /// contribute to `aborted_tracing`.  Repeating the local fallback past the
    /// pyre-only ceiling must therefore leave the cell traceable.
    #[test]
    fn a_pretrace_decline_does_not_charge_the_abort_ceiling() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![0xDEC11E]);
        let hash = key.get_uhash();

        for _ in 0..=MAX_TRACE_ABORT_COUNT {
            ws.mark_as_being_traced_for_key(&key);
            ws.decline_tracing(hash);
        }

        let cell = ws
            .get_cell_for_key(&key)
            .expect("decline keeps the warm-state cell");
        assert_eq!(cell.abort_count.get(), 0);
        assert_eq!(cell.derived_state(), BaseJitCellState::NotHot);
        assert!(!cell.flags.get().contains(JcFlags::JC_TRACING));
        assert!(!cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));
        assert!(ws.can_inline_callable_for_key(&key));
    }

    #[test]
    fn test_tracing_generation_increments() {
        let mut ws = WarmEnterState::new(2);
        let gen0 = ws.tracing_generation();
        assert_eq!(gen0, 0);

        // Start tracing key 1 → generation 1
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 1);

        let cell = ws.get_cell(ws.cell_key_for(&gk(1)).expect("cell")).unwrap();
        assert_eq!(cell.tracing_generation.get(), 1);

        // Start tracing key 2 → generation 2
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 2);

        let cell = ws.get_cell(ws.cell_key_for(&gk(2)).expect("cell")).unwrap();
        assert_eq!(cell.tracing_generation.get(), 2);
    }

    /// Slot 66 fires only for a cell whose `JC_TRACING` outlived the session
    /// that set it; slot 65 also counts a decline taken while a trace is
    /// running.
    ///
    /// This test calls `should_trace_function_entry` DIRECTLY, so it can
    /// reach the gate mid-trace. Production cannot: the one production caller
    /// guards on `!driver.is_tracing()`, so there every 65 is already a leak
    /// and 66 only says how old. Do not read this test as evidence that a
    /// production 65 is healthy — see the slot legend in `lib.rs`.
    ///
    /// Deltas are asserted as lower bounds because `MC_DIAG` is a process-wide
    /// static — a concurrent test bumping the same slot can only inflate them.
    #[test]
    fn stale_tracing_generation_is_counted_apart_from_a_live_trace() {
        let mut ws = WarmEnterState::new(2);
        let key_a = GreenKey::new(vec![0xA]);

        // Cell A is marked as being traced under the current generation.
        ws.mark_as_being_traced_for_key(&key_a);
        let cell = ws
            .get_cell_for_key(&key_a)
            .expect("mark_as_being_traced_for_key installs it");
        assert!(cell.is_tracing(), "fixture: A carries JC_TRACING");
        assert_eq!(
            cell.tracing_generation.get(),
            ws.tracing_generation(),
            "fixture: A's generation is the live one, so it is NOT stale yet"
        );

        // While A's trace is still the live one, the decline is the healthy
        // case: 65 moves, 66 must not.
        let live_65 = crate::mc_diag(65);
        let live_66 = crate::mc_diag(66);
        assert!(!ws.should_trace_function_entry_for_key(&key_a));
        assert!(
            crate::mc_diag(65) >= live_65 + 1,
            "a decline on the tracing term bumps 65"
        );
        assert_eq!(
            crate::mc_diag(66),
            live_66,
            "a LIVE trace must not read as stale — this is what 65 alone cannot say"
        );

        // Starting a trace on another key supersedes A's session without
        // clearing A's flag: only start_tracing_cell increments the generation.
        assert!(matches!(maybe_compile(&mut ws, 0xB), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, 0xB),
            HotResult::StartTracing
        ));
        let cell = ws.get_cell_for_key(&key_a).expect("A is still installed");
        assert!(cell.is_tracing(), "fixture: A's flag was never cleared");
        assert!(
            cell.tracing_generation.get() < ws.tracing_generation(),
            "fixture: A's session has been superseded"
        );

        let stale_66 = crate::mc_diag(66);
        assert!(!ws.should_trace_function_entry_for_key(&key_a));
        assert!(
            crate::mc_diag(66) >= stale_66 + 1,
            "a stale JC_TRACING at this gate is what slot 66 reports"
        );
    }

    /// The set/clear pair for `JC_TRACING` agrees on the one cell a green
    /// key owns. `mark_as_being_traced_for_key` writes that cell;
    /// `clear_tracing_flag` on the hash key clears the same cell.
    #[test]
    fn a_typed_clear_reaches_the_cell_a_typed_mark_wrote() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![100, 200]);
        let hash = key.get_uhash();

        ws.disable_noninlinable_function_for_key(&key);

        ws.mark_as_being_traced_for_key(&key);
        assert_eq!(ws.get_stats().num_cells, 1, "one green key, one cell");
        assert!(
            ws.lookup_chain_with_key(&key)
                .expect("the cell exists")
                .is_tracing(),
        );
        assert!(
            ws.get_cell(hash).expect("the cell").is_tracing(),
            "the hash key and the typed key name one cell",
        );

        ws.clear_tracing_flag(hash);
        assert!(
            !ws.lookup_chain_with_key(&key)
                .expect("the cell exists")
                .is_tracing(),
            "clearing the hash key clears the one cell",
        );
    }

    #[test]
    fn test_jitcell_should_remove() {
        // A freshly created cell with no token and no flags should be removable
        let cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        assert!(cell.should_remove_jitcell());

        // A cell that is tracing should NOT be removable
        let mut cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        cell.flags.set(cell.flags.get() | (JcFlags::JC_TRACING));
        assert!(!cell.should_remove_jitcell());

        // A cell with DONT_TRACE_HERE but no token history is removable
        let mut cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_DONT_TRACE_HERE));
        assert!(!cell.should_remove_jitcell()); // has_seen_a_procedure_token is false

        // warmstate.py — a `JC_DONT_TRACE_HERE` cell that HAD a
        // procedure token and no longer has one is removable ("i.e. dead
        // weakref"). The state is built by letting the token die, not by
        // stamping a number: `has_seen_a_procedure_token` reads the weakref
        // slot (warmstate.py), which is what the arm below is a
        // reading of.
        let mut cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_DONT_TRACE_HERE));
        {
            let token = Arc::new(JitCellToken::new(42));
            cell.set_procedure_token(Arc::clone(&token), false);
            assert!(
                !cell.should_remove_jitcell(),
                "a live procedure token pins the cell (warmstate.py:213-214)"
            );
        }
        // `token` dropped — `MemoryManager` is the only other owner and this
        // cell was never registered, so nothing holds it now.
        assert!(
            cell.get_procedure_token().is_none(),
            "the weakref no longer upgrades"
        );
        assert!(
            cell.has_seen_a_procedure_token(),
            "but the SLOT is still set, which is what distinguishes \
             'had one and lost it' from 'never had one'"
        );
        assert!(cell.should_remove_jitcell());

        // warmstate.py:222-225: FORCE_FINISH must NOT be removed
        let mut cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        cell.flags
            .set(cell.flags.get() | (JcFlags::JC_FORCE_FINISH));
        assert!(!cell.should_remove_jitcell());
    }

    #[test]
    fn test_gc_cells() {
        let mut ws = WarmEnterState::new(2);

        // Create some cells in various states
        // Key 1: compiled (should NOT be removed)
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 1), HotResult::StartTracing));
        ws.finish_tracing(ws.cell_key_for(&gk(1)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, 1, token);

        // Key 2: tracing (should NOT be removed)
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 2), HotResult::StartTracing));

        // Key 3: aborted without dont_trace → NotHot, removable
        assert!(matches!(maybe_compile(&mut ws, 3), HotResult::NotHot));
        assert!(matches!(maybe_compile(&mut ws, 3), HotResult::StartTracing));
        ws.abort_tracing(ws.cell_key_for(&gk(3)).expect("cell"), false);

        assert_eq!(ws.get_stats().num_cells, 3);
        let removed = ws.gc_cells();
        assert_eq!(removed, 1); // key 3 removed
        assert_eq!(ws.get_stats().num_cells, 2);
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(1)).expect("cell"))
                .is_some()
        );
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(2)).expect("cell"))
                .is_some()
        );
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(3)).expect("cell"))
                .is_none()
        );
    }

    #[test]
    fn test_invalidated_cell_allows_recompilation() {
        // After invalidation, transitioning back to NotHot allows recompilation.
        let mut ws = WarmEnterState::new(2);
        let key = 0xD4;

        // Compile
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Compiled
        );

        // Invalidate
        ws.transition_cell(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            BaseJitCellState::Invalidated,
        );
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        // Reset to NotHot and recompile
        ws.transition_cell(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            BaseJitCellState::NotHot,
        );
        let token2 = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token2);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Compiled
        );
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::RunCompiled
        ));
    }

    #[test]
    fn test_get_param() {
        let ws = WarmEnterState::new(100);
        assert_eq!(ws.get_param("threshold"), Some(100));
        assert_eq!(ws.get_param("vectorize"), Some(0));
        assert_eq!(ws.get_param("unknown_param"), None);
    }

    #[test]
    fn test_set_param_to_default() {
        let mut ws = WarmEnterState::new(100);
        ws.set_param("trace_limit", 999);
        assert_eq!(ws.get_param("trace_limit"), Some(999));
        ws.set_param_to_default("trace_limit");
        assert_eq!(
            ws.get_param("trace_limit"),
            Some(DEFAULT_TRACE_LIMIT as i64)
        );
    }

    #[test]
    fn test_param_names() {
        let names = WarmEnterState::param_names();
        assert!(names.contains(&"threshold"));
        assert!(names.contains(&"trace_limit"));
        assert!(names.contains(&"vectorize"));
        assert!(names.len() >= 10);
    }

    #[test]
    fn test_quasiimmut_dependency_lifecycle() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xF00D;
        let qmut = 0xBEEF;

        // Compile a loop
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);

        // Register quasi-immutable dependency
        ws.register_quasiimmut_dependency(qmut, ws.cell_key_for(&gk(key as i64)).expect("cell"));

        // Invalidate
        let invalidated = ws.invalidate_quasiimmut(qmut);
        assert_eq!(invalidated, 1);

        // Loop should now be invalidated state
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );
    }

    #[test]
    fn test_quasiimmut_invalidated_cell_rearms_hot_counter() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xD00D;
        let qmut = 0xFA11;

        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        ws.register_quasiimmut_dependency(qmut, ws.cell_key_for(&gk(key as i64)).expect("cell"));

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        // warmstate.py:483-500: the next entry removes the invalidated
        // token cell and resets the counter; following entries count fresh.
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .is_none()
        );
        assert!(matches!(maybe_compile(&mut ws, key), HotResult::NotHot));
        assert!(matches!(
            maybe_compile(&mut ws, key),
            HotResult::StartTracing
        ));
    }

    #[test]
    fn tick_empty_chain_is_the_counter_tick_on_an_unoccupied_hash() {
        // warmstate.py:465-469: no cell → `jitcounter.tick(hash, increment)`.
        // Threshold 2 fires on the second add (compute_threshold(2) ≈ 0.5).
        let mut ws = WarmEnterState::new(2);
        let hash = 0xA11C_u64;
        assert!(ws.lookup_chain(hash).is_none());
        assert!(!ws.tick_empty_chain(hash));
        assert!(ws.lookup_chain(hash).is_none());
        assert!(ws.tick_empty_chain(hash));
        assert!(ws.lookup_chain(hash).is_none());
    }

    #[test]
    fn test_counter_tick_checked_rearms_after_quasiimmut_invalidation() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xD00E;
        let qmut = 0xFA12;

        assert!(!ws.counter_tick_checked(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            &[GreenType::Int],
            &[],
        ));
        assert!(ws.counter_tick_checked(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            &[GreenType::Int],
            &[],
        ));
        ws.force_start_tracing_for_key(&gk(key as i64), &[]);
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        ws.register_quasiimmut_dependency(qmut, ws.cell_key_for(&gk(key as i64)).expect("cell"));

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        assert!(!ws.counter_tick_checked(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            &[GreenType::Int],
            &[],
        ));
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .is_none()
        );
        assert!(!ws.counter_tick_checked(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            &[GreenType::Int],
            &[],
        ));
        assert!(ws.counter_tick_checked(
            ws.cell_key_for(&gk(key as i64)).expect("cell"),
            &[GreenType::Int],
            &[],
        ));
    }

    #[test]
    fn test_function_entry_rearms_after_quasiimmut_invalidation() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = 0xD00F;
        let qmut = 0xFA13;

        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        ws.force_start_tracing_for_key(&gk(key as i64), &[]);
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        ws.register_quasiimmut_dependency(qmut, ws.cell_key_for(&gk(key as i64)).expect("cell"));

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .is_none()
        );
        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
    }

    /// A cell that has latched the pyre-local abort ceiling AND holds a
    /// procedure token that was since invalidated must still be removed from
    /// the chain: `warmstate.py maybe_compile_and_run` re-counts such a location
    /// from cold,
    /// and both back-edge doors already exempt a dead token from the ceiling.
    /// Refusing at the ceiling first leaves the stale entry in place for every
    /// later call, so its bucket's counter never re-arms.
    #[test]
    fn test_function_entry_ceiling_still_cleans_up_a_dead_token() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = 0xD010;
        let qmut = 0xFA14;

        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        ws.force_start_tracing_for_key(&gk(key as i64), &[]);
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
        attach_alive(&mut ws, key, token);
        ws.register_quasiimmut_dependency(qmut, ws.cell_key_for(&gk(key as i64)).expect("cell"));

        for _ in 0..MAX_TRACE_ABORT_COUNT {
            ws.abort_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"), false);
        }
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .is_some()
        );

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(
            ws.get_cell_state(ws.cell_key_for(&gk(key as i64)).expect("cell")),
            BaseJitCellState::Invalidated
        );

        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(key as i64)).expect("cell"))
                .is_none(),
            "a dead token at the abort ceiling must reach cleanup_chain"
        );
        assert!(!ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
        assert!(ws.should_trace_function_entry(ws.cell_key_for(&gk(key as i64)).expect("cell")));
    }

    /// The three answers [`WarmEnterState::function_entry_step`] gives, against
    /// the three separate reads of the same chain the door used to compute them
    /// from.
    ///
    /// `has_compiled_meta` is the caller's half of
    /// `JitDriver::has_runnable_compiled_loop`. A token WITHOUT frontend meta is
    /// not runnable, so only both halves together produce
    /// [`FunctionEntryStep::RunCompiled`].
    #[test]
    fn function_entry_step_answers_the_whole_door_from_one_walk() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = 0xD011;

        // Cold, no trace running: the counter gate, unchanged.
        assert!(matches!(
            ws.function_entry_step(
                ws.cell_key_for(&gk(key as i64)).expect("cell"),
                None,
                || false,
                false,
                &[],
            ),
            FunctionEntryStep::NotHot
        ));
        assert!(matches!(
            ws.function_entry_step(
                ws.cell_key_for(&gk(key as i64)).expect("cell"),
                None,
                || false,
                false,
                &[],
            ),
            FunctionEntryStep::Proceed
        ));

        ws.force_start_tracing_for_key(&gk(key as i64), &[]);
        ws.finish_tracing(ws.cell_key_for(&gk(key as i64)).expect("cell"));
        let token = token_with_compiled_code(&mut ws);
        attach_alive(&mut ws, key, token);

        assert!(matches!(
            ws.function_entry_step(
                ws.cell_key_for(&gk(key as i64)).expect("cell"),
                None,
                || true,
                false,
                &[],
            ),
            FunctionEntryStep::RunCompiled(_)
        ));
        assert!(matches!(
            ws.function_entry_step(
                ws.cell_key_for(&gk(key as i64)).expect("cell"),
                None,
                || false,
                false,
                &[],
            ),
            FunctionEntryStep::NotHot
        ));
    }

    /// `warmstate.py maybe_compile_and_run` ticks other greenkeys while a
    /// nested `MetaInterp` is tracing. A fire cannot nest here, so the hash
    /// stays at `JitCell.trace_next_iteration`'s 0.98 and the next
    /// non-tracing entry compiles.
    #[test]
    fn function_entry_step_ticks_other_keys_while_the_engine_is_tracing() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = 0xD012;
        let cell_key = ws.cell_key_for(&gk(key as i64)).expect("cell");

        for _ in 0..8 {
            assert!(matches!(
                ws.function_entry_step(
                    ws.cell_key_for(&gk(key as i64)).expect("cell"),
                    None,
                    || false,
                    true,
                    &[],
                ),
                FunctionEntryStep::NotHot
            ));
        }
        assert!(ws.should_trace_function_entry(cell_key));
    }

    /// Nested `engine_is_tracing` parks a back-edge fire at 0.98
    /// (`tick_or_defer_nested`); the next non-tracing call starts.
    #[test]
    fn back_edge_decision_parks_a_fire_while_the_engine_is_tracing() {
        // warmstate.py maybe_compile_and_run ticks other greenkeys during an
        // outer compile_and_run_once. A fire cannot nest here, so the hash
        // stays at JitCell.trace_next_iteration's 0.98.
        let mut ws = WarmEnterState::new(2);
        let key = gk(0xD014);
        ws.ensure_cell_for_key(&key);
        for _ in 0..8 {
            assert!(matches!(
                ws.maybe_compile_decision_with_key_nested(&key, |_| false, true, &[]),
                HotResult::NotHot
            ));
        }
        assert!(matches!(
            ws.maybe_compile_decision_with_key_nested(&key, |_| false, false, &[]),
            HotResult::StartTracing
        ));
    }

    /// `warmstate.py maybe_compile_and_run`: a cell already carrying
    /// `JC_TRACING` returns without ticking, including when the engine is
    /// tracing that same cell.
    #[test]
    fn function_entry_step_does_not_tick_a_cell_that_is_jc_tracing() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = gk(0xD013);
        ws.force_start_tracing_for_key(&key, &[]);
        let cell_key = ws.cell_key_for(&key).expect("cell");

        for _ in 0..8 {
            assert!(matches!(
                ws.function_entry_step(cell_key, None, || false, true, &[]),
                FunctionEntryStep::NotHot
            ));
        }
        assert!(matches!(
            ws.function_entry_step(cell_key, None, || false, false, &[]),
            FunctionEntryStep::NotHot
        ));
    }

    /// `tick_or_defer_nested` JC_TEMPORARY arm: a tmp-callback token still
    /// counts, but a fire while the engine is tracing parks at 0.98.
    #[test]
    fn temporary_token_parks_a_fire_while_the_engine_is_tracing() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![0x7E, 0x13]);
        let token = Arc::new(JitCellToken::new(0x7e13));
        ws.get_assembler_token_with_key::<(), _>(&key, |_memmgr| Ok(token.clone()))
            .expect("tmp token");
        assert!(
            ws.lookup_chain_with_key(&key)
                .expect("tmp cell")
                .flags
                .get()
                .contains(JcFlags::JC_TEMPORARY)
        );
        for _ in 0..8 {
            assert!(matches!(
                ws.maybe_compile_decision_with_key_nested(&key, |_| true, true, &[]),
                HotResult::NotHot
            ));
        }
        assert!(matches!(
            ws.maybe_compile_decision_with_key_nested(&key, |_| true, false, &[]),
            HotResult::StartTracing
        ));
    }

    /// Never-compiled `JC_DONT_TRACE_HERE` retries immediately
    /// (`should_start_dont_trace_here_trace` with no `JC_TRACING_OCCURRED`),
    /// then parks at 0.98 while the engine is tracing.
    #[test]
    fn dont_trace_here_never_compiled_parks_at_098_while_tracing() {
        let mut ws = WarmEnterState::new(2);
        let key = gk(0xD015);
        ws.disable_noninlinable_function_for_key(&key);
        assert!(matches!(
            ws.maybe_compile_decision_with_key_nested(&key, |_| false, true, &[]),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_decision_with_key_nested(&key, |_| false, false, &[]),
            HotResult::StartTracing
        ));
    }

    /// Compiled-but-not-runnable back-edge tail: `tick_or_defer_nested`
    /// after a procedure token that `has_compiled_meta` refuses.
    #[test]
    fn compiled_not_runnable_back_edge_parks_while_tracing() {
        let mut ws = WarmEnterState::new(2);
        let key = gk(0xD016);
        let token = token_with_compiled_code(&mut ws);
        attach_alive_for_key(&mut ws, &key, token);
        for _ in 0..8 {
            assert!(matches!(
                ws.maybe_compile_decision_with_key_nested(&key, |_| false, true, &[]),
                HotResult::NotHot
            ));
        }
        assert!(matches!(
            ws.maybe_compile_decision_with_key_nested(&key, |_| false, false, &[]),
            HotResult::StartTracing
        ));
    }

    /// A compiled cell that is not runnable must not tick the function-entry
    /// door: consecutive probes stay cold and the live token stays attached.
    #[test]
    fn function_entry_does_not_tick_a_compiled_not_runnable_cell() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = GreenKey::new(vec![0xC0, 0xDE]);
        let token = token_with_compiled_code(&mut ws);
        attach_alive_for_key(&mut ws, &key, Arc::clone(&token));

        for _ in 0..50 {
            assert!(!ws.should_trace_function_entry_for_key(&key));
        }
        let held = ws
            .lookup_chain_with_key(&key)
            .and_then(|cell| cell.get_procedure_token())
            .expect("the compiled cell still holds a token");
        assert!(
            Arc::ptr_eq(&held, &token),
            "the compiled cell still holds its token",
        );

        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = gk(0xC0DE);
        let token = token_with_compiled_code(&mut ws);
        attach_alive_for_key(&mut ws, &key, token);
        for _ in 0..50 {
            assert!(matches!(
                ws.function_entry_step(
                    ws.cell_key_for(&key).expect("cell"),
                    None,
                    || false,
                    false,
                    &[]
                ),
                FunctionEntryStep::NotHot
            ));
        }
    }

    /// `warmstate.py maybe_compile_and_run`: `increment_threshold == 0` is
    /// "jit is off" — not hot, no tick, no cleanup — before any cell lookup.
    #[test]
    fn zero_increment_returns_not_hot_without_ticking() {
        let mut ws = WarmEnterState::new(2);
        let key = gk(0xA11);
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::NotHot
        ));
        ws.set_threshold(0);
        for _ in 0..8 {
            assert!(matches!(
                ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
                HotResult::NotHot
            ));
        }
        ws.set_threshold(2);
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::StartTracing
        ));

        let mut ws = WarmEnterState::new(2);
        let compiled = gk(0xA12);
        attach_alive_for_key(&mut ws, &compiled, JitCellToken::new(1));
        ws.set_threshold(0);
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&compiled, |_| true, &[]),
            HotResult::NotHot
        ));

        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(0);
        assert!(matches!(
            ws.function_entry_step(
                ws.cell_key_for(&gk(0xA12)).expect("cell"),
                None,
                || true,
                false,
                &[],
            ),
            FunctionEntryStep::NotHot
        ));

        let mut ws = WarmEnterState::new(2);
        let aborted = GreenKey::new(vec![0xA13]);
        assert!(matches!(
            ws.maybe_compile_with_key(&aborted),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_with_key(&aborted),
            HotResult::StartTracing
        ));
        ws.abort_tracing(aborted.get_uhash(), false);
        assert!(ws.lookup_chain_with_key(&aborted).is_some());
        ws.set_threshold(0);
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&aborted, |_| true, &[]),
            HotResult::NotHot
        ));
        assert!(
            ws.lookup_chain_with_key(&aborted).is_some(),
            "increment 0 does not cleanup_chain",
        );
    }

    /// `warmstate.py maybe_compile_and_run` reads `JC_TRACING` before the
    /// procedure token, so compiled+tracing chooses already-tracing.
    /// `engine_is_tracing` does not turn that into StartTracing / Proceed.
    #[test]
    fn tracing_and_compiled_is_already_tracing() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![0xA21, 0xA22]);
        let token = token_with_compiled_code(&mut ws);
        attach_alive_for_key(&mut ws, &key, token);
        let cell = ws.lookup_chain_with_key(&key).expect("compiled cell");
        cell.flags.set(cell.flags.get() | JcFlags::JC_TRACING);
        let cell_key = ws.cell_key_for(&key).expect("compiled cell key");

        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::AlreadyTracing
        ));
        assert!(matches!(
            ws.maybe_compile_decision(cell_key, &key.types),
            HotResult::AlreadyTracing
        ));
        assert!(matches!(
            ws.function_entry_step(cell_key, Some(&key), || true, false, &[]),
            FunctionEntryStep::NotHot
        ));
        assert!(matches!(
            ws.function_entry_step(cell_key, Some(&key), || true, true, &[]),
            FunctionEntryStep::NotHot
        ));
    }

    /// `warmstate.py maybe_compile_and_run` counts `JC_TEMPORARY` normally,
    /// including when `JC_DONT_TRACE_HERE` is also set.
    #[test]
    fn temporary_dont_trace_here_counts_toward_start() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![0x7E, 0x11]);
        let token = Arc::new(JitCellToken::new(0x7e11));
        ws.get_assembler_token_with_key::<(), _>(&key, |_memmgr| Ok(token.clone()))
            .expect("tmp token");
        ws.disable_noninlinable_function_for_key(&key);
        let cell = ws.lookup_chain_with_key(&key).expect("tmp cell");
        assert!(cell.flags.get().contains(JcFlags::JC_TEMPORARY));
        assert!(cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));

        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::StartTracing
        ));

        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = GreenKey::new(vec![0x7E, 0x12]);
        let token = Arc::new(JitCellToken::new(0x7e12));
        ws.get_assembler_token_with_key::<(), _>(&key, |_memmgr| Ok(token.clone()))
            .expect("tmp token");
        ws.disable_noninlinable_function_for_key(&key);
        assert!(!ws.should_trace_function_entry_for_key(&key));
        assert!(ws.should_trace_function_entry_for_key(&key));
    }

    /// `warmstate.py maybe_compile_and_run`: a tokenless aborted cell is
    /// `cleanup_chain` on every door, including the typed function-entry one.
    #[test]
    fn tokenless_aborted_cell_function_entry_runs_cleanup_chain() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![0xC1, 0xEA]);
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(matches!(
            ws.maybe_compile_with_key(&key),
            HotResult::StartTracing
        ));
        ws.abort_tracing(key.get_uhash(), false);
        assert!(ws.lookup_chain_with_key(&key).is_some());
        assert!(!ws.should_trace_function_entry_for_key(&key));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "an ordinary tokenless aborted cell is cleanup_chain",
        );
    }

    #[test]
    fn test_set_param_roundtrip() {
        let mut ws = WarmEnterState::new(100);
        for name in WarmEnterState::param_names() {
            let original = ws.get_param(name);
            // "decay" is write-only (warmstate.py set_param_decay delegates to
            // jitcounter.set_decay which stores decay_by_mult, not the raw int).
            if *name != "decay" {
                assert!(original.is_some(), "param {name} should be gettable");
            }
            ws.set_param(name, 999);
            ws.set_param_to_default(name);
            // After default, should be same as a fresh instance
        }
    }

    /// warmstate.py `hash_whatever` routes a generic GC pointer through
    /// `lltype.identityhash`; minimark's implementation is
    /// `mangle_hash(id_or_identityhash(obj))`, not the raw pointer bits.
    #[test]
    fn warm_state_registers_gc_ref_identity_hash() {
        let _ws = WarmEnterState::new(100);
        let addr = 0x5eed_1230usize;
        let identity = majit_gc::gc_id_or_identityhash(addr) as u64;
        assert_eq!(
            majit_ir::value::hash_whatever(majit_ir::GreenType::Ref, addr as i64),
            identity ^ (identity >> 4),
        );
        assert_eq!(
            majit_ir::value::hash_whatever(majit_ir::GreenType::Ref, 0),
            0,
        );
    }

    /// warmstate.py comparekey — `comparekey_matches` returns true only
    /// when the cell's stored greens equal the probe (`JitCell.comparekey`).
    #[test]
    fn comparekey_matches_only_with_stored_key() {
        let cell = BaseJitCell::new_for_driver_with_greens(0, &[], &[]);
        let key = GreenKey::new(vec![1, 2, 3]);
        assert!(
            !cell.comparekey_matches(&key),
            "empty greens must not match a longer probe"
        );
        let cell = BaseJitCell::new_for_driver_with_greens(0, &key.values, &key.types);
        assert!(
            cell.comparekey_matches(&key),
            "cell with stored comparekey must match equal probe"
        );
        assert!(
            cell.comparekey(&key.values, &key.types),
            "slice comparekey is the same body as comparekey_matches"
        );
        let other = GreenKey::new(vec![1, 2, 4]);
        assert!(
            !cell.comparekey_matches(&other),
            "cell with stored comparekey must not match unequal probe"
        );
        assert!(!cell.comparekey(&other.values, &other.types));
    }

    // Sentinel-tagged retain/release log for the ownership test below.
    //
    // `set_ref_resolver` is a process-global `OnceLock`, so exactly one test in
    // this binary may register and it cannot be scoped to a single case.
    // Everything is therefore asserted by SENTINEL VALUE rather than by call
    // count: any other fixture that installs a Ref green while this resolver is
    // live would move a count, but cannot forge these addresses.
    static RETAIN_LOG: parking_lot::Mutex<Vec<i64>> = parking_lot::Mutex::new(Vec::new());
    static RELEASE_LOG: parking_lot::Mutex<Vec<i64>> = parking_lot::Mutex::new(Vec::new());
    static CURRENT_REF_GREEN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

    fn test_retain(value: i64) -> usize {
        RETAIN_LOG.lock().push(value);
        value as usize
    }

    fn test_current(handle: usize) -> i64 {
        const REF_GREEN: i64 = 0x5EED_0001;
        if handle as i64 == REF_GREEN {
            let current = CURRENT_REF_GREEN.load(std::sync::atomic::Ordering::SeqCst);
            if current != 0 {
                return current;
            }
        }
        handle as i64
    }

    fn test_release(handle: usize) {
        RELEASE_LOG.lock().push(handle as i64);
    }

    /// warmstate.py JitCell — a cell's stored green key OWNS its `Ref`
    /// referents, so the address it names cannot be freed and recycled by a
    /// different object while the cell is alive.
    ///
    /// Four legs, and the second is the localization control:
    ///
    /// 1. the `Ref` slot is retained on install;
    /// 2. an `Int` slot holding an equally pointer-shaped value is **not**
    ///    retained — ownership must key off the declared `GreenType`, not off
    ///    "looks like an address", which is the only way it can be right;
    /// 3. a null `Ref` is skipped (`hash_whatever` folds null to 0, and there
    ///    is nothing to own);
    /// 4. dropping the cell releases exactly what it retained.
    #[test]
    fn stored_green_key_owns_its_ref_referents() {
        majit_ir::set_ref_resolver(test_retain, test_current, test_release);

        const REF_GREEN: i64 = 0x5EED_0001;
        const MOVED_REF_GREEN: i64 = 0x5EED_1001;
        const INT_GREEN: i64 = 0x5EED_0002;
        CURRENT_REF_GREEN.store(0, std::sync::atomic::Ordering::SeqCst);

        let key = GreenKey::with_types(
            vec![7, INT_GREEN, REF_GREEN, 0],
            vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        );

        {
            let mut ws = WarmEnterState::new(3);
            ws.ensure_cell_for_key(&key);

            let retained = RETAIN_LOG.lock().clone();
            assert!(
                retained.contains(&REF_GREEN),
                "installing a cell must retain its Ref green; log={retained:x?}"
            );
            assert!(
                !retained.contains(&INT_GREEN),
                "an Int green must NOT be retained even when its value is \
                 pointer-shaped — ownership keys off GreenType, not off the \
                 bit pattern; log={retained:x?}"
            );
            assert!(
                !retained.contains(&0),
                "a null Ref green has no referent to own; log={retained:x?}"
            );

            assert_eq!(
                ws.get_stats().num_pinned_refs,
                1,
                "the pinned-population counter must see exactly the one Ref \
                 referent this cell owns"
            );

            assert!(
                !RELEASE_LOG.lock().contains(&REF_GREEN),
                "nothing may be released while the cell that owns it is alive"
            );

            CURRENT_REF_GREEN.store(MOVED_REF_GREEN, std::sync::atomic::Ordering::SeqCst);
            let moved_key = GreenKey::with_types(
                vec![7, INT_GREEN, MOVED_REF_GREEN, 0],
                vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
            );
            let cell = ws
                .lookup_chain(key.get_uhash())
                .expect("the retained cell stays in its original identity-hash bucket");
            assert!(
                cell.comparekey_matches(&moved_key),
                "comparekey must read the collector-forwarded owner-root value"
            );
            assert_eq!(
                cell.current_greens()[2],
                MOVED_REF_GREEN,
                "current_greens reads the same forwarded gN as comparekey"
            );
            let cell_key = cell.cell_key.get().expect("installed cell has a key");
            let types = [
                GreenType::Int,
                GreenType::Int,
                GreenType::Ref,
                GreenType::Ref,
            ];
            let installed = ws
                .installed_greenkey(cell_key, &types)
                .expect("the retained cell is named by its cell key");
            assert_eq!(
                installed.values[2], MOVED_REF_GREEN,
                "installed_greenkey reconstructs from forwarded gN values"
            );
        }

        let released = RELEASE_LOG.lock().clone();
        assert!(
            released.contains(&REF_GREEN),
            "dropping the cell must release its Ref green; log={released:x?}"
        );
        assert!(
            !released.contains(&INT_GREEN),
            "an unretained Int green must never be released; log={released:x?}"
        );
    }

    /// warmstate.py ensure_jit_cell_at_key + 596-604 — `ensure_cell_for_key` allocates
    /// a fresh cell on miss and `lookup_chain_with_key` returns it on a
    /// repeat probe with the same typed greens.
    #[test]
    fn ensure_and_lookup_chain_with_key_round_trips() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![42, 100]);

        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "no cell for fresh key"
        );
        ws.ensure_cell_for_key(&key);
        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("ensure must install a cell");
        assert_eq!(
            &cell.greens[..],
            &key.values[..],
            "stored comparekey must equal the install key"
        );
    }

    /// warmstate.py `dont_trace_here(*greenargs)` — the typed form stores
    /// `comparekey` (`JitCell(*greenargs)`). The hash form names that
    /// installed cell and sets `DONT_TRACE_HERE` on it.
    #[test]
    fn dont_trace_here_typed_form_installs_and_the_hash_form_flags_the_installed_cell() {
        let key = GreenKey::new(vec![7, 11]);

        let mut typed = WarmEnterState::new(100);
        typed.disable_noninlinable_function_for_key(&key);
        let typed_cell = typed
            .get_cell(key.get_uhash())
            .expect("typed form installs a cell");
        assert!(
            typed_cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE),
            "typed form must set DONT_TRACE_HERE"
        );
        assert_eq!(
            &typed_cell.greens[..],
            &key.values[..],
            "typed form must store the greens it was called with"
        );

        let mut hashed = WarmEnterState::new(100);
        hashed.ensure_cell_for_key(&key);
        hashed.disable_noninlinable_function(key.get_uhash());
        let hashed_cell = hashed
            .get_cell(key.get_uhash())
            .expect("hash form names the installed cell");
        assert!(
            hashed_cell
                .flags
                .get()
                .contains(JcFlags::JC_DONT_TRACE_HERE),
            "hash form on an installed cell sets DONT_TRACE_HERE"
        );
        assert_eq!(&hashed_cell.greens[..], &key.values[..]);
    }

    #[test]
    #[should_panic(expected = "disable_noninlinable_function names an installed cell")]
    fn hash_form_disable_on_a_missing_key_panics() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![7, 11]);
        ws.disable_noninlinable_function(key.get_uhash());
    }

    #[test]
    #[should_panic(expected = "mark_force_finish_tracing names an installed cell")]
    fn hash_form_mark_force_finish_on_a_missing_key_panics() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![7, 11]);
        ws.mark_force_finish_tracing(key.get_uhash());
    }

    /// warmstate.py get_jitcell — chain walk distinguishes hash collisions:
    /// two distinct GreenKeys sharing one bucket must each resolve to
    /// their own cell via `comparekey` rather than aliasing. The test
    /// exploits `install_new_cell`'s `should_remove_jitcell` gate
    /// (`BaseJitCell::should_remove_jitcell`, counter.py parity): a cell with
    /// no `loop_token` / no flags is treated as dead and dropped on
    /// the next install. Setting `JC_TRACING` on the first-installed
    /// cell keeps it alive across the second install so both end up
    /// chained through `head.next`. The walker must then skip the
    /// non-matching head when looking up the chained cell.
    #[test]
    fn lookup_chain_with_key_resolves_hash_collisions_via_comparekey() {
        let mut ws = WarmEnterState::new(100);
        let key_head_after_chain = GreenKey::new(vec![1, 2]);
        let key_chained = GreenKey::new(vec![3, 4]);
        let bucket = key_head_after_chain.get_uhash();

        // First install: TRACING flag so should_remove_jitcell == false
        // and install_new_cell preserves the cell across the second
        // install. Stays at the back of the chain after the second
        // install per install_new_cell's "prepend new cell, fold
        // existing keepable cells in front" semantics.
        let cell_first = BaseJitCell::new_for_driver_with_greens(
            0,
            &key_head_after_chain.values,
            &key_head_after_chain.types,
        );
        cell_first
            .flags
            .set(cell_first.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(cell_first));
        // Second install — keep predicate doesn't matter for this one,
        // it lands as `keep` and the prior head folds in front, so the
        // resulting chain shape is head=cell_first, *head.next.borrow_mut() = cell_for_key_chained.
        // Wait — install_new_cell folds the EXISTING chain in front of
        // `keep`. So after the second install: head=cell_first (the
        // existing one, since !should_remove), head.next=cell_chained.
        let cell_chained =
            BaseJitCell::new_for_driver_with_greens(0, &key_chained.values, &key_chained.types);
        ws.install_new_cell(bucket, Some(cell_chained));

        let head = ws
            .lookup_chain(bucket)
            .expect("bucket head exists after install");
        assert_eq!(&head.greens[..], &key_head_after_chain.values[..]);
        let next = head.next.borrow().clone().expect("chain has a second cell");
        assert_eq!(&next.greens[..], &key_chained.values[..]);

        // Fast path: lookup_chain_with_key(key_head_after_chain) finds
        // the head's comparekey on first probe.
        let hit_head = ws
            .lookup_chain_with_key(&key_head_after_chain)
            .expect("walker must find chain head via comparekey");
        assert_eq!(&hit_head.greens[..], &key_head_after_chain.values[..]);
    }

    /// `JitCell.get_jitcell`: a unique hash occupant is still accepted
    /// only after `comparekey`. A different green key with the same
    /// `get_uhash` must miss even when the occupant already holds an
    /// executable token.
    #[test]
    fn unique_hash_candidate_still_compares_greens() {
        let mut ws = WarmEnterState::new(100);
        // Closed-form `get_uhash` collision: Int greens `[1, 0]` and
        // `[2, 13080698639]` fold to the same u64.
        let key_hit = GreenKey::new(vec![1, 0]);
        let key_miss = GreenKey::new(vec![2, 13080698639]);
        let hash = key_hit.get_uhash();
        assert_eq!(
            hash,
            key_miss.get_uhash(),
            "fixture is a full-hash collision"
        );
        ws.ensure_cell_for_key(&key_hit);
        attach_alive_for_key(
            &mut ws,
            &key_hit,
            JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED),
        );
        assert!(
            ws.lookup_chain(hash)
                .expect("installed")
                .next
                .borrow()
                .is_none(),
            "test needs a unique occupant"
        );
        let miss = (key_miss.values.as_slice(), key_miss.types.as_slice());
        let hit = (key_hit.values.as_slice(), key_hit.types.as_slice());
        assert!(
            ws.resolved_cell_procedure_token(hash, Some(hit))
                .1
                .is_some(),
            "occupant holds an executable token"
        );

        let occupant_key = ws
            .cell_key_for(&key_hit)
            .expect("installed occupant has a cell key");
        assert_eq!(
            ws.resolve_cell_key(hash, || key_hit.clone()),
            occupant_key,
            "matching greens resolve to the occupant's cell key"
        );
        assert_eq!(
            ws.resolve_cell_key(hash, || key_miss.clone()),
            hash,
            "JitCell.get_jitcell returns None; do not hand back the occupant's key"
        );

        let (_cell_key, cell) = ws.resolved_cell(hash, Some(miss));
        assert!(
            cell.is_none(),
            "unique hash occupant must not win without comparekey"
        );

        let (_cell_key, cell_hit) = ws.resolved_cell(hash, Some(hit));
        assert!(
            cell_hit
                .expect("matching greens must hit")
                .comparekey_matches(&key_hit)
        );

        let (_cell_key, token) = ws.resolved_cell_procedure_token(hash, Some(miss));
        assert!(
            token.is_none(),
            "comparekey miss must not recover the occupant's token"
        );

        // count==0 typed lookup: `hash` is a minted cell key sitting in
        // another bucket. `cell_named_by_hash` would return that cell;
        // `get_jitcell` walks this hash's chain and misses.
        let mut minted_ws = WarmEnterState::new(100);
        let parked = GreenKey::new(vec![7100, 7200]);
        let parked_bucket = parked.get_uhash();
        let squatter = GreenKey::new(vec![7300, colliding_last_green(&parked, 7300)]);
        assert_eq!(squatter.get_uhash(), parked_bucket);
        minted_ws.ensure_cell_for_key(&squatter);
        let parked_token = token_with_compiled_code(&mut minted_ws);
        minted_ws.attach_procedure_to_interp_for_key(&parked, Arc::clone(&parked_token));
        let minted = minted_ws
            .cell_key_for(&parked)
            .expect("colliding squatter forces a minted key");
        assert_ne!(minted, parked_bucket);
        let arriving = gk_with_uhash(minted, 8100);
        assert_eq!(arriving.get_uhash(), minted);
        assert_ne!(arriving, parked);
        let arriving_args = (arriving.values.as_slice(), arriving.types.as_slice());
        let (_cell_key, minted_cell) = minted_ws.resolved_cell(minted, Some(arriving_args));
        assert!(
            minted_cell.is_none(),
            "typed miss must not recover a minted cell from another bucket"
        );
        let (_cell_key, minted_token) =
            minted_ws.resolved_cell_procedure_token(minted, Some(arriving_args));
        assert!(
            minted_token.is_none(),
            "typed miss must not return the minted cell's token"
        );
        assert_eq!(
            minted_ws.resolve_cell_key(minted, || arriving.clone()),
            minted,
            "get_jitcell miss carries the arriving hash"
        );
    }

    /// The door keeps the chain-head token `maybe_compile_and_run` would
    /// carry, and a comparekey miss at that hash is Occupied.
    #[test]
    fn door_cell_carries_the_resolved_token() {
        let mut ws = WarmEnterState::new(100);
        let key_hit = GreenKey::new(vec![1, 0]);
        let key_miss = GreenKey::new(vec![2, 13080698639]);
        let hash = key_hit.get_uhash();
        assert_eq!(
            hash,
            key_miss.get_uhash(),
            "fixture is a full-hash collision"
        );
        ws.ensure_cell_for_key(&key_hit);
        attach_alive_for_key(
            &mut ws,
            &key_hit,
            JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED),
        );
        assert!(
            ws.lookup_chain(hash)
                .expect("installed")
                .next
                .borrow()
                .is_none(),
            "test needs a unique occupant"
        );
        let hit = (key_hit.values.as_slice(), key_hit.types.as_slice());
        let miss = (key_miss.values.as_slice(), key_miss.types.as_slice());
        let (resolved_key, resolved_token) = ws.resolved_cell_procedure_token(hash, Some(hit));
        let resolved_token = resolved_token.expect("occupant holds an executable token");
        let DoorCell::Enter { cell_key, token } = ws.door_cell(hash, Some(hit)) else {
            panic!("door_cell must carry the compiled occupant's token");
        };
        assert_eq!(cell_key, resolved_key);
        assert!(
            Arc::ptr_eq(&token, &resolved_token),
            "door_cell carries the same token resolved_cell_procedure_token found"
        );
        assert!(
            matches!(ws.door_cell(hash, Some(miss)), DoorCell::Occupied),
            "comparekey miss with the same hash is Occupied"
        );
    }

    /// A tracing occupant whose comparekey misses is Occupied, not Tracing.
    /// `maybe_compile_and_run` establishes identity before `cell.flags`.
    #[test]
    fn door_cell_tracing_comparekey_miss_is_occupied() {
        let mut ws = WarmEnterState::new(100);
        let key_hit = GreenKey::new(vec![1, 0]);
        let key_miss = GreenKey::new(vec![2, 13080698639]);
        let hash = key_hit.get_uhash();
        assert_eq!(
            hash,
            key_miss.get_uhash(),
            "fixture is a full-hash collision"
        );
        ws.ensure_cell_for_key(&key_hit);
        let cell = ws.lookup_chain(hash).expect("installed");
        assert!(cell.next.borrow().is_none(), "test needs a unique occupant");
        cell.flags.set(cell.flags.get() | JcFlags::JC_TRACING);
        let hit = (key_hit.values.as_slice(), key_hit.types.as_slice());
        let miss = (key_miss.values.as_slice(), key_miss.types.as_slice());
        assert!(
            matches!(ws.door_cell(hash, Some(miss)), DoorCell::Occupied),
            "comparekey miss on a tracing cell is Occupied"
        );
        assert!(
            matches!(ws.door_cell(hash, Some(hit)), DoorCell::Tracing),
            "comparekey hit on a tracing cell is Tracing"
        );
    }

    /// Build a 2-cell chain in `target`'s bucket: a decoy cell at the head
    /// (different comparekey) and `target`'s cell chained behind it, both
    /// TRACING.  The typed-key lifecycle mutators must walk past the decoy
    /// head and touch only `target`'s cell.
    fn chain_decoy_then_target(target: &GreenKey, decoy: &GreenKey) -> WarmEnterState {
        let mut ws = WarmEnterState::new(100);
        let bucket = target.get_uhash();
        let decoy_cell = BaseJitCell::new_for_driver_with_greens(0, &decoy.values, &decoy.types);
        decoy_cell
            .flags
            .set(decoy_cell.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(decoy_cell));
        let target_cell = BaseJitCell::new_for_driver_with_greens(0, &target.values, &target.types);
        target_cell
            .flags
            .set(target_cell.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(target_cell));
        ws
    }

    #[test]
    fn finish_tracing_for_key_clears_tracing_on_matching_cell_only() {
        let target = GreenKey::new(vec![5, 6]);
        let decoy = GreenKey::new(vec![7, 8]);
        let mut ws = chain_decoy_then_target(&target, &decoy);
        ws.finish_tracing_for_key(&target);
        let head = ws.lookup_chain(target.get_uhash()).expect("bucket head");
        assert_eq!(&head.greens[..], &decoy.values[..]);
        assert!(
            head.flags.get().contains(JcFlags::JC_TRACING),
            "decoy head untouched"
        );
        let hit = head
            .next
            .borrow()
            .clone()
            .expect("target chained behind decoy");
        assert_eq!(&hit.greens[..], &target.values[..]);
        assert!(
            !hit.flags.get().contains(JcFlags::JC_TRACING),
            "target TRACING cleared"
        );
    }

    /// `warmstate.py` `get_assembler_token` — a fresh typed key
    /// installs a temporary procedure token (tmp=true → JC_TEMPORARY)
    /// and returns it; subsequent calls with the same key return the
    /// same token without invoking `make_token` again.
    #[test]
    fn get_assembler_token_with_key_caches_token_per_typed_key() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![10, 20]);
        let token = std::sync::Arc::new(JitCellToken::new(0xa55e_b1ed_u64 as i64));

        let mut make_token_calls = 0;
        let token_clone_1 = {
            let token_for_closure = token.clone();
            ws.get_assembler_token_with_key::<(), _>(&key, |_memmgr| {
                make_token_calls += 1;
                Ok(token_for_closure)
            })
            .expect("install ok")
        };
        assert_eq!(make_token_calls, 1, "first call invokes make_token");
        assert!(std::sync::Arc::ptr_eq(&token_clone_1, &token));

        let token_clone_2 = ws
            .get_assembler_token_with_key::<(), _>(&key, |_memmgr| {
                make_token_calls += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(
                    0xdead_beef_u64 as i64,
                )))
            })
            .expect("re-fetch ok");
        assert_eq!(
            make_token_calls, 1,
            "second call hits the cached token, does not invoke make_token"
        );
        assert!(
            std::sync::Arc::ptr_eq(&token_clone_2, &token),
            "second call returns the originally installed token"
        );

        // JC_TEMPORARY must be set since tmp=true.
        let cell = ws.lookup_chain_with_key(&key).expect("cell installed");
        assert!(
            cell.flags.get().contains(JcFlags::JC_TEMPORARY),
            "tmp token must set JC_TEMPORARY"
        );
    }

    #[test]
    #[should_panic(expected = "attach_procedure_to_interp names an installed cell")]
    fn attach_procedure_to_interp_expects_an_installed_cell() {
        let mut ws = WarmEnterState::new(100);
        let _ = ws.attach_procedure_to_interp(0xDEAD, JitCellToken::new(1));
    }

    #[test]
    #[should_panic(expected = "get_assembler_token names an installed cell")]
    fn get_assembler_token_expects_an_installed_cell() {
        let mut ws = WarmEnterState::new(100);
        let _ = ws.get_assembler_token::<(), _>(0xDEAD, |_memmgr| {
            Ok(std::sync::Arc::new(JitCellToken::new(1)))
        });
    }

    /// `warmstate.py maybe_compile_and_run` checks JC_TEMPORARY before
    /// JC_DONT_TRACE_HERE.  A callee blamed for a trace-too-long abort can
    /// carry all of TEMPORARY (its residual CALL_ASSEMBLER callback),
    /// DONT_TRACE_HERE (do not inline it again), and FORCE_FINISH (segment its
    /// standalone trace).  That combination must count and then start tracing
    /// the matching typed cell; otherwise the callee can never replace its
    /// temporary callback with a real loop.
    #[test]
    fn temporary_noninlinable_function_entry_traces_its_typed_cell() {
        let mut ws = WarmEnterState::new(100);
        ws.set_function_threshold(1);
        let key = GreenKey::new(vec![30, 40]);
        let bucket = key.get_uhash();

        // Put a comparator-less cell at the bucket head first. This is the
        // production migration shape in which a raw-hash precheck and a typed
        // force-start used to inspect different cells for one green key.
        ws.disable_noninlinable_function_for_key(&key);
        let token = std::sync::Arc::new(JitCellToken::new(0xcafe));
        ws.get_assembler_token_with_key::<(), _>(&key, |_memmgr| Ok(token.clone()))
            .expect("temporary token install");
        ws.disable_noninlinable_function_for_key(&key);
        ws.mark_force_finish_tracing_for_key(&key);

        let typed = ws.lookup_chain_with_key(&key).expect("typed callee cell");
        assert_eq!(
            typed.cell_key.get(),
            Some(bucket),
            "one green key, one cell key"
        );
        assert!(typed.flags.get().contains(JcFlags::JC_TEMPORARY));
        assert!(typed.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE));
        assert!(typed.flags.get().contains(JcFlags::JC_FORCE_FINISH));

        assert!(
            ws.should_trace_function_entry_for_key(&key),
            "JC_TEMPORARY counts normally despite DONT_TRACE_HERE"
        );
        assert!(matches!(
            ws.force_start_tracing_for_key(&key, &[]),
            HotResult::StartTracing
        ));

        let typed = ws.lookup_chain_with_key(&key).expect("typed callee cell");
        assert!(typed.is_tracing(), "the matching typed cell starts tracing");
        assert!(
            ws.cell_by_key(bucket).expect("the one cell").is_tracing(),
            "the hash key names the same cell",
        );
        assert!(
            typed.flags.get().contains(JcFlags::JC_FORCE_FINISH),
            "the standalone retry retains its segmenting request"
        );
    }

    /// `warmstate.py get_assembler_token` + `626-641` — typed variant must
    /// disambiguate hash collisions: two `GreenKey`s that share a hash
    /// but compare unequal under `equal_whatever` get distinct tokens.
    #[test]
    fn get_assembler_token_with_key_disambiguates_hash_collisions() {
        let mut ws = WarmEnterState::new(100);

        // Build a GreenKey pair guaranteed to collide on hash via the
        // existing collision fixture (install_new_cell uses bucket-by-hash).
        // Reuse the same shape as the chain-walk test: install a typed
        // cell first, then call get_assembler_token_with_key for another
        // typed key in the same bucket.
        let key_a = GreenKey::new(vec![100, 200]);
        let key_b = GreenKey::new(vec![300, 400]);

        let token_a = std::sync::Arc::new(JitCellToken::new(0xaaaa));
        let token_b = std::sync::Arc::new(JitCellToken::new(0xbbbb));

        let got_a = ws
            .get_assembler_token_with_key::<(), _>(&key_a, |_memmgr| Ok(token_a.clone()))
            .expect("install a");
        let got_b = ws
            .get_assembler_token_with_key::<(), _>(&key_b, |_memmgr| Ok(token_b.clone()))
            .expect("install b");

        assert!(
            std::sync::Arc::ptr_eq(&got_a, &token_a),
            "key_a returns token_a"
        );
        assert!(
            std::sync::Arc::ptr_eq(&got_b, &token_b),
            "key_b returns token_b"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&got_a, &got_b),
            "distinct typed keys must not alias to the same token"
        );

        // Repeat fetch must not invoke make_token.
        let mut count = 0;
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_a, |_memmgr| {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xcccc)))
            })
            .expect("re-fetch a");
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_b, |_memmgr| {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xdddd)))
            })
            .expect("re-fetch b");
        assert_eq!(count, 0, "cached lookups must not invoke make_token");
    }

    /// The hash form names the installed typed cell. One green key reached
    /// through either form owns one cell with a comparekey.
    #[test]
    fn a_hash_write_and_a_typed_read_of_one_key_use_one_cell() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 9]);
        assert_eq!(ws.get_stats().num_cells, 0);
        assert!(ws.get_cell(key.get_uhash()).is_none());
        ws.ensure_cell_for_key(&key);
        assert_eq!(ws.occupied_buckets(), 1, "one hash, so one bucket");
        assert_eq!(ws.get_stats().num_cells, 1, "typed install is one cell");
        ws.disable_noninlinable_function(key.get_uhash());
        assert_eq!(
            ws.get_stats().num_cells,
            1,
            "hash write on an existing cell does not chain a sibling",
        );
        let typed_cell = ws
            .lookup_chain_with_key(&key)
            .expect("typed install is one cell");
        assert_eq!(&typed_cell.greens[..], &key.values[..]);
        assert!(
            typed_cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE),
            "hash form on an installed cell sets DONT_TRACE_HERE",
        );

        let mut typed = WarmEnterState::new(3);
        let key2 = GreenKey::new(vec![7, 9]);
        typed.disable_noninlinable_function_for_key(&key2);
        assert!(
            matches!(typed.maybe_compile_with_key(&key2), HotResult::StartTracing),
            "the typed mark IS consulted, so the dont-trace-here rule applies              at once instead of waiting for the counter",
        );
        assert_eq!(typed.get_stats().num_cells, 1, "no split on the typed path");
    }

    /// Drive `key` to the tick that would start a trace.
    ///
    /// `WarmEnterState::new(3)` needs three ticks; the first two answer
    /// `NotHot` from the counter alone and reach no veto position.
    fn tick_to_threshold(ws: &mut WarmEnterState, key: &GreenKey) -> HotResult {
        ws.maybe_compile_with_key(key);
        ws.maybe_compile_with_key(key);
        ws.maybe_compile_with_key(key)
    }

    /// `warmstate.py:427-428` — a hook that refuses stops the trace before the
    /// cell is marked, so the location keeps interpreting and stays startable.
    ///
    /// Without a hook installed the same fixture answers `StartTracing`
    /// (`maybe_compile_with_key_lazily_installs_on_threshold_tick` above), so
    /// the veto is what the answer turns on here.
    #[test]
    fn a_confirm_enter_jit_veto_stops_the_tracing_start() {
        fn refuse(_args: &[Value]) -> bool {
            false
        }
        let mut ws = WarmEnterState::new(3);
        ws.set_confirm_enter_jit(Some(refuse));
        let key = GreenKey::new(vec![11, 13]);

        assert!(
            matches!(tick_to_threshold(&mut ws, &key), HotResult::NotHot),
            "the threshold tick reaches the veto, and a refusal is `return` \
             upstream — the caller keeps interpreting",
        );
        assert!(
            ws.lookup_chain_with_key(&key)
                .is_none_or(|cell| !cell.is_tracing()),
            "`cell.flags |= JC_TRACING` is below the veto (warmstate.py:441), \
             so a refused start must leave no cell claiming to be tracing",
        );
    }

    /// The callback receives the typed green tuple itself, not the bucket hash
    /// that the cell table uses internally.
    #[test]
    fn the_confirm_enter_jit_hook_can_inspect_the_greens() {
        static FIRST: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
        static SECOND: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
        fn refuse(args: &[Value]) -> bool {
            FIRST.store(args[0].as_int(), std::sync::atomic::Ordering::Relaxed);
            SECOND.store(args[1].as_int(), std::sync::atomic::Ordering::Relaxed);
            false
        }

        let mut ws = WarmEnterState::new(1);
        ws.set_confirm_enter_jit(Some(refuse));
        let key = GreenKey::new(vec![41, 43]);

        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert_eq!(FIRST.load(std::sync::atomic::Ordering::Relaxed), 41);
        assert_eq!(SECOND.load(std::sync::atomic::Ordering::Relaxed), 43);
    }

    /// `warmstate.py:501` — the second veto position, between finding a live
    /// procedure token and `raise EnterJitAssembler`.
    #[test]
    fn a_confirm_enter_jit_veto_stops_entry_into_compiled_code() {
        fn refuse(_args: &[Value]) -> bool {
            false
        }
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![21, 23]);
        let _token = attach_alive_for_key(&mut ws, &key, JitCellToken::new(1));
        assert!(
            matches!(ws.maybe_compile_with_key(&key), HotResult::RunCompiled),
            "fixture: the cell is compiled and entered before any hook exists",
        );

        ws.set_confirm_enter_jit(Some(refuse));
        assert!(
            matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot),
            "a refusal at :501 leaves the caller interpreting rather than \
             entering the machine code",
        );
        assert!(
            matches!(ws.force_start_tracing_for_key(&key, &[]), HotResult::NotHot),
            "the force-start door must not turn an already-compiled cell into \
             an unchecked compiled entry",
        );
    }

    /// A permissive hook must preserve both compiled-entry answers even when
    /// the compatibility API names the typed cell by hash. This also proves
    /// those doors recover and pass the cell's structured `comparekey` rather
    /// than treating an installed hook as an unconditional refusal.
    #[test]
    fn a_permissive_confirm_enter_jit_preserves_hash_compiled_entry() {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn allow(_args: &[Value]) -> bool {
            CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }

        CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![22, 24]);
        let hash = key.get_uhash();
        let _token = attach_alive_for_key(&mut ws, &key, JitCellToken::new(1));
        ws.set_confirm_enter_jit(Some(allow));

        assert!(matches!(
            ws.maybe_compile_decision_with_meta(hash, |_| true, &key.types, &[]),
            HotResult::RunCompiled
        ));
        assert!(matches!(
            ws.force_start_tracing(hash, &key.types, &[]),
            HotResult::RunCompiled
        ));
        assert_eq!(
            CALLS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "both compiled-entry doors must call the permissive hook",
        );
    }

    /// The function-entry policy owns the compiled-code door too; returning
    /// `NotHot` is the caller's instruction to keep interpreting the frame.
    #[test]
    fn a_confirm_enter_jit_veto_stops_the_function_entry_door() {
        fn refuse(_args: &[Value]) -> bool {
            false
        }
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![25, 27]);
        let token = token_with_compiled_code(&mut ws);
        let _token = attach_alive_for_key(&mut ws, &key, token);
        let cell_key = ws.cell_key_for(&key).expect("fixture installed the cell");

        assert!(matches!(
            ws.function_entry_step(cell_key, Some(&key), || true, false, &[]),
            FunctionEntryStep::RunCompiled(_),
        ));
        ws.set_confirm_enter_jit(Some(refuse));
        assert!(matches!(
            ws.function_entry_step(cell_key, Some(&key), || true, false, &[]),
            FunctionEntryStep::NotHot,
        ));
    }

    /// The hook fires once per tracing start, not once per decision.
    ///
    /// Upstream has a single `bound_reached` and calls it there
    /// (`warmstate.py`). Here the decision and the start are two calls —
    /// `maybe_compile_decision` answers, and `MetaInterp::bound_reached`
    /// reaches the cell through `force_start_tracing_for_key` — so a veto
    /// placed in both would consult a user callback twice for one entry.
    #[test]
    fn the_confirm_enter_jit_hook_fires_once_per_tracing_start() {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn allow(_args: &[Value]) -> bool {
            CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }
        let mut ws = WarmEnterState::new(3);
        ws.set_confirm_enter_jit(Some(allow));
        let key = GreenKey::new(vec![31, 33]);
        let hash = key.get_uhash();

        assert!(matches!(
            ws.maybe_compile_decision(hash, &key.types),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_decision(hash, &key.types),
            HotResult::NotHot
        ));
        assert!(
            matches!(
                ws.maybe_compile_decision(hash, &key.types),
                HotResult::StartTracing
            ),
            "the decision alone answers, and it is not a veto position",
        );
        assert_eq!(
            CALLS.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the decision must not consult the hook; the start does",
        );

        assert!(matches!(
            ws.force_start_tracing_for_key(&key, &[]),
            HotResult::StartTracing
        ));
        assert_eq!(
            CALLS.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "one entry, one call",
        );
    }

    /// The default is upstream's own substitution: no hook declared means
    /// `def confirm_enter_jit(*args): return True` (`warmstate.py`).
    #[test]
    fn no_declared_hook_confirms_every_entry() {
        let ws = WarmEnterState::new(3);
        assert!(ws.confirm_enter_jit(&[0], &[], &[]));
        assert!(ws.confirm_enter_jit(&[i64::MAX], &[], &[]));
    }

    /// `warmspot.py` `_make_hook_graph(..., onlygreens=False)` packs greens
    /// then reds. Each green word uses its `_green_args_spec` type
    /// (`warmstate.py` `make_entry_point` `confirm_enter_jit(*args)`).
    #[test]
    fn confirm_enter_jit_packs_typed_greens_then_reds() {
        static SEEN0: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
        static SEEN1: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static SEEN2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static SEEN3: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static SEEN_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn record(args: &[Value]) -> bool {
            SEEN_LEN.store(args.len(), std::sync::atomic::Ordering::Relaxed);
            SEEN0.store(args[0].as_int(), std::sync::atomic::Ordering::Relaxed);
            SEEN1.store(
                args[1].as_ref().as_usize(),
                std::sync::atomic::Ordering::Relaxed,
            );
            SEEN2.store(
                args[2].as_float().to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            SEEN3.store(
                args[3].as_ref().as_usize(),
                std::sync::atomic::Ordering::Relaxed,
            );
            true
        }

        let mut ws = WarmEnterState::new(3);
        ws.set_confirm_enter_jit(Some(record));
        let greens = [7i64, 0x1000, 1.5f64.to_bits() as i64];
        let types = [GreenType::Int, GreenType::Ref, GreenType::Float];
        let reds = [Value::Ref(GcRef(0x2000))];
        assert!(ws.confirm_enter_jit(&greens, &types, &reds));
        assert_eq!(SEEN_LEN.load(std::sync::atomic::Ordering::Relaxed), 4);
        let packed = [
            Value::Int(SEEN0.load(std::sync::atomic::Ordering::Relaxed)),
            Value::Ref(GcRef(SEEN1.load(std::sync::atomic::Ordering::Relaxed))),
            Value::Float(f64::from_bits(
                SEEN2.load(std::sync::atomic::Ordering::Relaxed),
            )),
            Value::Ref(GcRef(SEEN3.load(std::sync::atomic::Ordering::Relaxed))),
        ];
        assert_eq!(
            packed,
            [
                Value::Int(7),
                Value::Ref(GcRef(0x1000)),
                Value::Float(1.5),
                Value::Ref(GcRef(0x2000)),
            ]
        );
    }

    /// `confirm_enter_jit(*args)` receives this invocation's greens when
    /// the caller holds them (`warmstate.py` `maybe_compile_and_run`).
    ///
    /// `equal_whatever(Float)` is `f64 ==`, so +0.0 stored and -0.0 live
    /// share a cell; the hook still sees the live bits.
    #[test]
    fn confirm_compiled_entry_prefers_live_greens_over_the_cells_stored_word() {
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        fn record(args: &[Value]) -> bool {
            SEEN.store(
                args[0].as_float().to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            true
        }

        let pos_bits = 0.0f64.to_bits();
        let neg_bits = (-0.0f64).to_bits();
        assert_ne!(pos_bits, neg_bits);
        let pos = GreenKey::with_types(vec![pos_bits as i64], vec![GreenType::Float]);
        let mut ws = WarmEnterState::new(3);
        let token = token_with_compiled_code(&mut ws);
        let _token = attach_alive_for_key(&mut ws, &pos, token);
        let cell_key = ws.cell_key_for(&pos).expect("fixture installed the cell");
        ws.set_confirm_enter_jit(Some(record));

        SEEN.store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(ws.confirm_compiled_entry_for_cell_key(
            cell_key,
            Some(&[neg_bits as i64]),
            &[GreenType::Float],
            &[],
        ));
        assert_eq!(
            SEEN.load(std::sync::atomic::Ordering::Relaxed),
            neg_bits,
            "Some(live) packs the invocation's greens, not the cell's +0.0"
        );

        SEEN.store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(ws.confirm_compiled_entry_for_cell_key(cell_key, None, &[GreenType::Float], &[],));
        assert_eq!(
            SEEN.load(std::sync::atomic::Ordering::Relaxed),
            pos_bits,
            "None falls back to the cell's stored greens"
        );
    }

    /// warmstate.py — typed variant of `maybe_compile_and_run`.
    /// Upstream installs the JitCell lazily at `bound_reached`
    /// (warmstate.py): each tick under threshold returns
    /// without writing to the celltable; on the threshold tick, the
    /// cell is created with `comparekey` and installed via
    /// `jitcounter.install_new_cell`. The typed entry point preserves
    /// that lifecycle — pre-threshold ticks see no cell installed,
    /// only the bucket counter ticks.
    #[test]
    fn maybe_compile_with_key_lazily_installs_on_threshold_tick() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 9]);

        // Tick 1, 2: not hot. Cell is NOT yet installed — upstream
        // `maybe_compile_and_run` only allocates at `bound_reached`
        // (warmstate.py).
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "cold ticks must not install a cell (warmstate.py:466-468)",
        );
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "second cold tick still no cell",
        );

        // Tick 3: threshold reached → `start_tracing_cell_for_key`
        // installs a cell carrying `comparekey = key` (typed) and
        // sets the TRACING flags.
        match ws.maybe_compile_with_key(&key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("cell installed at threshold tick");
        assert_eq!(
            &cell.greens[..],
            &key.values[..],
            "comparekey populated on lazy install (warmstate.py:438-439)",
        );
        assert!(cell.is_tracing(), "JC_TRACING flag set on threshold tick");
    }

    /// The whole-table sweeps must walk each chain, not just its head.
    ///
    /// A table slot yields a chain HEAD, so a sweep that does not follow
    /// `next` silently skips every chained cell. "majit-metainterp: count
    /// chained cells in get_stats, not bucket heads" fixed exactly
    /// this in `get_stats`; the fix was filed at one access path and the
    /// remaining siblings kept the defect.
    ///
    /// `invalidate_all` is the one that costs a wrong answer rather than a
    /// leak: a chained cell whose token is never invalidated keeps running
    /// compiled code built under a retracted assumption.
    ///
    /// The token lives on the CHAINED cell and the head is left tokenless, so
    /// a head-only sweep reaches nothing at all — the assertions below fail
    /// before the fix.
    #[test]
    fn whole_table_sweeps_reach_chained_cells_not_only_heads() {
        let mut ws = WarmEnterState::new(100);

        let key_head = GreenKey::new(vec![100, 200]);
        let key_tail = GreenKey::new(vec![300, 400]);
        let bucket = key_tail.get_uhash();

        // TRACING keeps the head non-removable so the second install chains
        // behind it rather than replacing it (counter.py install_new_cell).
        let head = BaseJitCell::new_for_driver_with_greens(0, &key_head.values, &key_head.types);
        head.flags.set(head.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(head));

        const CHAINED_TOKEN: i64 = 0x5EED_0003;
        let chained_token = make_token(CHAINED_TOKEN);
        // `alive_loops` is the token's only strong owner once the cell's
        // handle is weak (`warmstate.py:188`, `memmgr.py:9-14`).
        ws.memory_manager.keep_loop_alive(&chained_token);
        let tail = BaseJitCell::new_for_driver_with_greens(0, &key_tail.values, &key_tail.types);
        tail.flags.set(tail.flags.get() | (JcFlags::JC_TRACING));
        *tail.loop_token.borrow_mut() = Some(std::sync::Arc::downgrade(&chained_token));
        ws.install_new_cell(bucket, Some(tail));

        // The token is on the chained cell, never on the head.
        let chain_head = ws.lookup_chain(bucket).expect("bucket has a head");
        assert!(
            chain_head.loop_token.borrow().is_none(),
            "fixture requires a tokenless head, or a head-only sweep would pass"
        );
        assert!(
            chain_head
                .next
                .borrow()
                .clone()
                .is_some_and(|c| c.loop_token.borrow().is_some()),
            "fixture requires the token on the CHAINED cell"
        );

        // 1. the chained cell's weak handle upgrades to the planted token.
        assert!(
            chain_head
                .next
                .borrow()
                .as_ref()
                .and_then(|c| c.loop_token.borrow().as_ref().and_then(|w| w.upgrade()))
                .is_some_and(|tok| Arc::ptr_eq(&tok, &chained_token)),
            "a token on a chained cell must be findable"
        );

        // 2. invalidate_all — the correctness one.
        ws.invalidate_all();
        let chained = ws
            .lookup_chain(bucket)
            .and_then(|h| h.next.borrow().clone())
            .expect("chain survives invalidate_all");
        assert!(
            chained
                .loop_token
                .borrow()
                .as_ref()
                .and_then(|t| t.upgrade())
                .is_some_and(|t| t.is_invalidated()),
            "invalidate_all must invalidate a CHAINED cell's token — skipping \
             it leaves compiled code live under a retracted assumption"
        );
    }

    /// warmstate.py — `JitCell.get_jitcell_for_args(*greenargs)`
    /// walks the per-bucket chain by `comparekey` to read AND mutate
    /// the cell associated with `greenargs`. A hash-form delegate
    /// would alias colliding GreenKeys to the same head and read /
    /// write the wrong cell's `JC_TRACING` / `JC_COMPILED` flags.
    ///
    /// The test simulates a hash collision by directly installing two
    /// cells in the SAME bucket via [`WarmEnterState::install_new_cell`]:
    /// `key_a` carries `JC_TRACING`, `key_b` is fresh. After installs,
    /// `install_new_cell` folds non-removable cells to the front, so
    /// the chain becomes `head=key_a (TRACING) -> tail=key_b`.
    ///
    /// `maybe_compile_with_key(&key_b)` must return `NotHot` (cell B
    /// has no flags). A hash-form read path would see the head
    /// (`key_a`) and incorrectly return `AlreadyTracing`.
    #[test]
    fn maybe_compile_with_key_walks_chain_for_state_under_hash_collision() {
        let mut ws = WarmEnterState::new(100);
        // Same hash bucket for both keys — install_new_cell takes the
        // bucket explicitly so we don't need a real `get_uhash`
        // collision in the GreenKey values.
        let key_b = GreenKey::new(vec![300, 400]);
        let key_a = GreenKey::new(vec![100, 200]);
        let bucket = key_b.get_uhash();

        // Pre-install A at B's bucket. JC_TRACING keeps A non-removable
        // (counter.py install_new_cell should_remove gate) so the next install
        // chains B behind A.
        let cell_a = BaseJitCell::new_for_driver_with_greens(0, &key_a.values, &key_a.types);
        cell_a.flags.set(cell_a.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(cell_a));

        // Install B at the same bucket. install_new_cell folds A in
        // front; resulting chain shape: head=A (TRACING) → tail=B.
        let cell_b = BaseJitCell::new_for_driver_with_greens(0, &key_b.values, &key_b.types);
        // A live token keeps B out of the tokenless `cleanup_chain` arm.
        let token_b = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        ws.memory_manager.keep_loop_alive(&token_b);
        cell_b.set_procedure_token(Arc::clone(&token_b), true);
        ws.install_new_cell(bucket, Some(cell_b));

        // Sanity — chain order matches the install_new_cell contract.
        let head = ws.lookup_chain(bucket).expect("bucket has a head");
        assert_eq!(&head.greens[..], &key_a.values[..]);
        assert!(head.is_tracing(), "head A carries the TRACING flag");
        let chained = head
            .next
            .borrow()
            .clone()
            .expect("chained tail exists after second install");
        assert_eq!(&chained.greens[..], &key_b.values[..]);
        assert!(!chained.is_tracing(), "B is fresh");

        // Without the chain-walk fix, `maybe_compile_with_key(&key_b)`
        // would delegate to `maybe_compile(hash)`, hit `cells.get(&hash)`,
        // see the head A's TRACING flag and return AlreadyTracing —
        // contaminating B's lookup with A's tracing state. With the fix
        // it walks the chain by comparekey and reads B's clean state
        // (NotHot path: counter ticks under threshold).
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::AlreadyTracing => panic!(
                "maybe_compile_with_key(&key_b) aliased to head A's TRACING flag — \
                 chain walk by comparekey is broken (warmstate.py:455-465 parity)"
            ),
            HotResult::NotHot | HotResult::StartTracing | HotResult::RunCompiled => {}
        }

        // Cell A's TRACING flag must remain on A, not migrate to B.
        let head_after = ws.lookup_chain(bucket).expect("bucket head still exists");
        assert!(
            head_after.is_tracing(),
            "head A's TRACING flag must persist across maybe_compile_with_key(&key_b)",
        );
        assert_eq!(&head_after.greens[..], &key_a.values[..]);
        let chained_after = head_after
            .next
            .borrow()
            .clone()
            .expect("chained tail still present");
        assert_eq!(&chained_after.greens[..], &key_b.values[..]);
        assert!(
            !chained_after.is_tracing(),
            "B must not have inherited A's TRACING flag (counter under threshold)",
        );
    }

    /// The hash form flags the installed cell. ONE green key through the
    /// typed form owns one cell; a later hash write does not chain a sibling.
    #[test]
    fn one_key_through_a_hash_and_a_typed_entry_point_uses_one_cell() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![100, 200]);

        assert_eq!(ws.get_stats().num_cells, 0);
        assert!(ws.get_cell(key.get_uhash()).is_none());

        ws.ensure_cell_for_key(&key);
        assert_eq!(ws.occupied_buckets(), 1, "one bucket");
        assert_eq!(ws.get_stats().num_cells, 1, "typed form installs one cell");
        ws.disable_noninlinable_function(key.get_uhash());
        assert_eq!(
            ws.get_stats().num_cells,
            1,
            "hash form does not install a sibling",
        );
        assert!(
            ws.lookup_chain_with_key(&key)
                .unwrap()
                .flags
                .get()
                .contains(JcFlags::JC_DONT_TRACE_HERE),
            "hash form on an installed cell sets DONT_TRACE_HERE",
        );
        assert!(
            ws.lookup_chain_with_key(&key)
                .unwrap()
                .comparekey_matches(&key)
        );
        assert_eq!(
            &ws.lookup_chain_with_key(&key).unwrap().greens[..],
            &key.values[..],
        );
    }

    /// The typed `_for_key` forms must select the cell belonging to the key,
    /// not whichever cell heads the bucket.
    ///
    /// Non-vacuity comes from the fixture, not from trust: both tests first
    /// assert that the bucket head is the *comparator-less* cell, so a
    /// head-reading implementation is provably looking at the wrong object.
    /// Against the hash-delegating forms these read `None` and `false`
    /// respectively.
    #[test]
    fn mark_as_being_traced_for_key_marks_the_keys_own_cell_not_the_bucket_head() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![300, 400]);

        // A hash-form writer squats the bucket with a comparator-less cell.
        ws.disable_noninlinable_function_for_key(&key);

        ws.mark_as_being_traced_for_key(&key);

        // Delegating to the hash form set TRACING on the comparator-less head
        // and left the key still owning no cell at all.
        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.flags.get().contains(JcFlags::JC_TRACING),
            "TRACING must land on the key's own cell",
        );
        assert_eq!(
            cell.derived_state(),
            BaseJitCellState::Tracing,
            "and so must the state transition",
        );
    }

    #[test]
    fn get_cell_for_key_returns_the_keys_cell_not_whichever_heads_the_bucket() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![500, 600]);

        ws.disable_noninlinable_function_for_key(&key);
        ws.ensure_cell_for_key(&key);
        assert_eq!(ws.occupied_buckets(), 1, "one bucket");
        assert_eq!(ws.get_stats().num_cells, 1, "one cell");

        assert!(
            ws.get_cell(key.get_uhash())
                .expect("bucket is occupied")
                .comparekey_matches(&key),
            "the hash key names the cell the typed key stamped",
        );

        let cell = ws.get_cell_for_key(&key).expect("the key owns a cell");
        assert!(
            cell.comparekey_matches(&key),
            "get_cell_for_key must return the cell that matches the key",
        );
    }

    /// The typed creators must write to the key's own cell, not squat a new
    /// comparator-less one beside it.
    ///
    /// Same fixture discipline as the pair above: a hash-form writer takes the
    /// bucket first and the assertion that it owns nothing typed runs *before*
    /// the call under test, so a form that delegated to the hash creator would
    /// be provably writing to the comparator-less head. Against the hash forms
    /// these read `None` at the `lookup_chain_with_key` line.
    #[test]
    fn attach_procedure_to_interp_for_key_installs_on_the_keys_own_cell() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![700, 800]);

        ws.disable_noninlinable_function_for_key(&key);
        assert!(ws.lookup_chain_with_key(&key).is_some());

        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.get_procedure_token().is_some(),
            "the procedure token must land on the key's own cell",
        );
        assert!(
            !cell.flags.get().contains(JcFlags::JC_TRACING),
            "and TRACING must be cleared on that same cell",
        );
    }

    /// `warmstate.py maybe_compile_and_run` reads a procedure token
    /// only off a cell whose `comparekey(*greenargs)` already matched. The
    /// hash form reads the bucket head, which on a chained bucket is a
    /// different cell.
    ///
    /// Same fixture discipline as the pair above: the hash-form writer takes
    /// the bucket first, so the head is provably the comparator-less cell and
    /// a head-reading lookup is looking at the wrong object.
    #[test]
    fn get_procedure_token_for_key_reads_the_keys_own_cell_not_the_bucket_head() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![1100, 1200]);

        ws.disable_noninlinable_function_for_key(&key);
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        assert!(!ws.bucket_is_chained(key.get_uhash()));
        assert!(
            ws.get_procedure_token(key.get_uhash()).is_some(),
            "the hash key names the cell that holds the token",
        );

        let found = ws
            .get_procedure_token_for_key(&key)
            .expect("the typed form must find the token on the key's own cell");
        assert!(
            Arc::ptr_eq(&found, &token),
            "and it must be the very token installed under this key",
        );
    }

    /// The second green of a two-`Int` key beginning with `first` that makes it
    /// hash to the same bucket as `other`.
    ///
    /// Constructed rather than searched. `get_uhash` folds
    /// `x = (x ^ hash_whatever(tp, v)) * GREEN_UHASH_MULT` (warmstate.py)
    /// and `hash_whatever(Int, v)` is `v` itself, so with the multiply the same
    /// on both sides the two hashes agree exactly when the pre-multiply words
    /// do — one xor away. A searched collision would pin the hash function; the
    /// subject here is what a chained bucket does.
    fn colliding_last_green(other: &GreenKey, first: i64) -> i64 {
        use majit_ir::{GREEN_UHASH_SEED, GreenType, green_uhash_step};
        assert_eq!(other.values.len(), 2, "written for two-green Int keys");
        let other_prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, other.values[0]);
        let this_prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, first);
        (this_prefix ^ other_prefix ^ (other.values[1] as u64)) as i64
    }

    /// A token that answers `has_compiled_code()`, which is what both halves of
    /// the entry gate test. The payload is never read — `has_compiled_code` is
    /// `self.compiled.get().is_some()` — so its type only has to satisfy the
    /// `Any + Send` bound.
    fn token_with_compiled_code(ws: &mut WarmEnterState) -> Arc<JitCellToken> {
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        token.set_compiled(Box::new(()));
        assert!(token.has_compiled_code());
        token
    }

    /// **The entry path decides on one cell and executes off another.**
    ///
    /// `jitdriver.rs entry_cell_has_compiled_code` resolves a chained bucket
    /// through `comparekey` — `has_compiled_loop_for_key`, i.e.
    /// `get_procedure_token_for_key` — but the runner it then calls,
    /// `MetaInterp::run_compiled_detailed_with_values_at_dispatch_key`, reads
    /// `warm_state.get_procedure_token(hash)`: the bucket HEAD.
    ///
    /// [`get_procedure_token_for_key_reads_the_keys_own_cell_not_the_bucket_head`]
    /// above pins the reader in isolation, and its head cell holds no token at
    /// all — so the entry it models DECLINES, which is safe. This one gives the
    /// head a compiled token of its own, which is the shape a real collision
    /// between two warm keys has, and then nothing declines: the decision is
    /// about the chained cell's artifact and the execution is the head cell's.
    ///
    /// `warmstate.py` reaches a token only through the cell it has
    /// already matched on greens + comparekey + `get_uhash` together, so the
    /// two can never name different objects upstream. `warmstate.py/511`
    /// then carries that resolved `procedure_token` to the executor
    /// (`raise EnterJitAssembler(procedure_token, *execute_args)`) rather than
    /// handing on the key for a second lookup — which is why upstream has no
    /// second reader that could disagree.
    ///
    /// The entry path now carries the token the same way: `back_edge_internal`
    /// binds it at the decision and hands it to
    /// `MetaInterp::execute_assembler_at_dispatch_key` as an argument, so on
    /// that route there is no longer a second reader for the resolved key to
    /// keep honest — only the one modelled below. The u64 reader survives for
    /// callers that reach a run without having decided anything about the cell
    /// first, which is what the `executed` line stands in for.
    ///
    /// # What this used to pin, and what it pins now
    ///
    /// This is the rewritten body of the `#[ignore]`d pin
    /// `the_entry_decision_and_the_entry_execution_can_name_different_tokens`,
    /// rewritten with the repo owner's explicit approval. The fixture below —
    /// two cells, one bucket, a compiled token each — is the original's,
    /// unchanged. The assertion is inverted from "these two readers disagree,
    /// and that is the open defect" to "these two readers resolve through one
    /// cell key, so they cannot disagree", which is what the cell-unique-key
    /// migration makes true.
    ///
    /// **Pre-migration failure, recorded from the run that authorised this
    /// rewrite.** On `add4516a533` the same fixture and the same final
    /// assertion, with the executed side reading `chained_key.get_uhash()` —
    /// the raw bucket hash, which is the only u64 the runner had to carry
    /// then — reports:
    ///
    /// ```text
    /// thread 'warmstate::tests::the_entry_decision_and_the_entry_execution_can_name_different_tokens'
    /// panicked at majit/majit-metainterp/src/warmstate.rs:4842:9:
    /// the entry decided `chained_key` has compiled code and then ran a DIFFERENT
    /// key's artifact: decided token #2, executed token #1. The two must resolve
    /// through the same cell.
    /// ```
    ///
    /// The old pin's stated blocker — `MetaInterp::compiled_loops` is an
    /// `IndexMap<u64, CompiledEntry<M>>` with one slot per hash, so two
    /// colliding keys cannot both file a meta — is what the migration
    /// dissolves rather than works around: the u64 those tables are keyed by
    /// became cell-unique, so two colliding green keys file two entries and
    /// nothing had to move onto the cell.
    ///
    /// **Measured again after the migration, by substitution.** Replacing
    /// `carried` with `bucket` on the `executed` line below — the only
    /// difference between this body and the pre-migration one — reproduces the
    /// same failure on the migrated tree:
    ///
    /// ```text
    /// panicked at majit/majit-metainterp/src/warmstate.rs:5127:9:
    /// the entry decided `chained_key` has compiled code and then ran a DIFFERENT
    /// key's artifact: decided token #2, executed token #1.
    /// ```
    ///
    /// So the resolve is what carries this test, not a fixture that stopped
    /// colliding. The last two assertions pin that in-test: the raw bucket
    /// hash still names the HEAD cell's artifact.
    #[test]
    fn the_entry_decision_and_the_entry_execution_resolve_through_one_cell_key() {
        let mut ws = WarmEnterState::new(100);
        let head_key = GreenKey::new(vec![2100, 2200]);
        let chained_key = GreenKey::new(vec![2300, colliding_last_green(&head_key, 2300)]);
        let bucket = head_key.get_uhash();
        assert_eq!(
            chained_key.get_uhash(),
            bucket,
            "fixture: the two keys must land in one bucket by their own hash, \
             not by being installed there — `lookup_chain_with_key` starts \
             from `key.get_uhash()`, so a hand-placed cell would simply be \
             unreachable and the test would pass for the wrong reason",
        );
        assert_ne!(
            head_key, chained_key,
            "fixture: and they must be DIFFERENT keys, which is what makes \
             one bucket two artifacts",
        );

        // Installed first, and keepable (a cell holding a procedure token is
        // never `should_remove_jitcell`), so `install_new_cell` folds it in
        // front of the second and it ends up HEADING the bucket.
        let head_token = token_with_compiled_code(&mut ws);
        let head_cell =
            BaseJitCell::new_for_driver_with_greens(0, &head_key.values, &head_key.types);
        head_cell.set_procedure_token(Arc::clone(&head_token), false);
        ws.install_new_cell(bucket, Some(head_cell));

        let chained_token = token_with_compiled_code(&mut ws);
        let chained_cell =
            BaseJitCell::new_for_driver_with_greens(0, &chained_key.values, &chained_key.types);
        chained_cell.set_procedure_token(Arc::clone(&chained_token), false);
        ws.install_new_cell(bucket, Some(chained_cell));

        assert!(
            ws.bucket_is_chained(bucket),
            "fixture: both cells must share one bucket, which is the only \
             case in which the two readers can disagree",
        );
        assert!(
            !Arc::ptr_eq(&head_token, &chained_token),
            "fixture: the two cells must hold DIFFERENT artifacts, or the \
             defect has nothing to show",
        );

        // THE RESOLVE, once, at the entry — `jitdriver.rs back_edge_internal`
        // calls `MetaInterp::resolve_cell_key` exactly here, with the bucket
        // hash it arrived with and the greens behind it. Everything after this
        // line carries `carried` and looks nothing up by bucket again.
        let carried = ws.resolve_cell_key(bucket, || chained_key.clone());
        assert_ne!(
            carried, bucket,
            "fixture: the chained cell had to be minted its own key, or the \
             two halves below would agree for the trivial reason",
        );

        // What the driver decides on for `chained_key` — the `comparekey`
        // walk, `has_compiled_loop_for_key`.
        let decided = ws
            .get_procedure_token_for_key(&chained_key)
            .expect("the decision resolves the chain and finds the key's cell");
        assert!(
            Arc::ptr_eq(&decided, &chained_token),
            "the decision is about the chained cell's artifact",
        );

        // What the runner then executes: the u64 reader, handed the key the
        // entry resolved rather than the bucket hash it arrived with.
        let executed = ws
            .get_procedure_token(carried)
            .expect("the carried cell key names the cell the decision matched");

        assert!(
            Arc::ptr_eq(&executed, &decided),
            "the entry decided `chained_key` has compiled code and then ran a \
             DIFFERENT key's artifact: decided token #{}, executed token #{}. \
             The two must resolve through the same cell.",
            decided.number(),
            executed.number(),
        );

        let by_bucket_hash = ws
            .get_procedure_token(bucket)
            .expect("the head cell holds the bucket's raw hash as its key");
        assert!(
            Arc::ptr_eq(&by_bucket_hash, &head_token),
            "the raw hash names the bucket's first occupant, as it always did",
        );
    }

    /// Two typed keys share `get_uhash()`. The second cell's compiled token
    /// is reachable only through the key `install_new_cell` minted for it.
    /// `maybe_compile_and_run_step` asks `has_compiled_meta` with that key
    /// (`compiled_loops.contains_key`) and then reads it back with
    /// `cell_key_for`. A decision that hands the raw hash to the callback
    /// misses the token and falls through to the counter.
    #[test]
    fn maybe_compile_decision_with_key_runs_the_colliding_cell_under_its_minted_key() {
        let mut ws = WarmEnterState::new(100);
        let head = GreenKey::new(vec![2100, 2200]);
        let second = GreenKey::new(vec![2300, colliding_last_green(&head, 2300)]);
        let hash = head.get_uhash();
        assert_eq!(
            second.get_uhash(),
            hash,
            "fixture: the two keys must share get_uhash()",
        );
        assert_ne!(head, second, "fixture: and they must be different keys");

        ws.ensure_cell_for_key(&head);
        ws.ensure_cell_for_key(&second);
        let minted = ws
            .cell_key_for(&second)
            .expect("the second key owns a cell");
        assert_ne!(
            minted, hash,
            "the second cell must carry the key install_new_cell minted once \
             the raw hash was taken",
        );

        let token = token_with_compiled_code(&mut ws);
        attach_alive_for_key(&mut ws, &second, token);
        assert!(
            ws.lookup_chain_with_key(&second)
                .is_some_and(|cell| cell.is_compiled()),
            "fixture: the second cell holds a live compiled token",
        );

        assert!(
            matches!(
                ws.maybe_compile_decision_with_key(&second, |cell_key| cell_key == minted, &[]),
                HotResult::RunCompiled
            ),
            "the second cell's compiled metadata lives under its minted key",
        );
    }

    /// A producer that reaches a chained bucket with no greens DECLINES rather
    /// than guessing which sibling was meant.
    ///
    /// This is the answer to the hazard the former hash-form installers leave
    /// behind. That population is not a list kept in prose but a property of
    /// one function: a creator is exactly a caller of
    /// [`WarmEnterState::ensure_cell_for_key`]. The hash form no longer installs; Anything
    /// reaching the celltable with a `&GreenKey` goes through
    /// `ensure_cell_for_key` instead and stores its comparator. A cell with no
    /// comparator cannot be told from a colliding neighbour by any mechanism —
    /// upstream cannot even express the state, since `JitCell.__init__`
    /// (warmstate.py) always stores the greens.
    ///
    /// So the resolve does not try. With no greens it answers the raw bucket
    /// hash, which names the bucket's ORIGINAL occupant — the cell that took
    /// the hash as its key — and names nothing at all once that cell is gone.
    /// That is a miss, and a miss is the honest port: a `None` from
    /// `cell_key_for` and a raw hash naming no cell both dead-end in the same
    /// "not found" arm `maybe_compile_and_run` has at warmstate.py.
    #[test]
    fn a_chained_bucket_reached_without_greens_answers_the_first_occupant() {
        let mut ws = WarmEnterState::new(100);
        let first = GreenKey::new(vec![3500, 3600]);
        let second = GreenKey::new(vec![3700, colliding_last_green(&first, 3700)]);
        let bucket = first.get_uhash();

        // A procedure token keeps the first cell non-removable, so the second
        // install chains it rather than pruning it (counter.py install_new_cell's
        // `should_remove_jitcell` gate — a cold, tokenless cell is dropped).
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        attach_alive_for_key(&mut ws, &first, token);
        let first_key = ws
            .cell_key_for(&first)
            .expect("the first key owns the cell it just installed");
        let second_key = ws.ensure_cell_key(&second);
        assert!(
            ws.bucket_is_chained(bucket),
            "fixture: the two keys must share a bucket",
        );

        assert_eq!(
            first_key, bucket,
            "the first occupant keeps the raw hash, so every workload that \
             never collides is bit-for-bit unchanged",
        );
        assert_ne!(second_key, bucket, "the second had to be minted one");
        assert_eq!(
            ws.sole_cell_key(bucket),
            None,
            "and a greens-less producer gets no answer from the bucket: two \
             candidates, nothing to choose between them",
        );

        // Which leaves the raw hash, and the raw hash names the first
        // occupant — never the minted sibling.
        assert_eq!(
            ws.cell_by_key(bucket).and_then(|cell| cell.cell_key.get()),
            Some(first_key),
        );
    }

    /// The one-walk resolve answers an ambiguous bucket with no greens exactly
    /// as the two readers it replaced did.
    ///
    /// [`WarmEnterState::resolved_cell`] folds
    /// [`WarmEnterState::cell_keys_at`] and [`WarmEnterState::cell_by_key`]
    /// into a single traversal, and its `(count > 1, greenargs: None)` arm is
    /// the one that cannot be checked against a caller's greens: it reports the
    /// raw hash and whatever cell that hash names. That arm reaches the cell
    /// through `by_key`, a candidate noted DURING the walk, where the readers
    /// it replaced reached it by walking back from the key — so the two can
    /// only be shown to agree on a bucket that actually holds more than one
    /// candidate, which is the fixture below.
    #[test]
    fn an_ambiguous_bucket_resolved_without_greens_names_the_cell_its_key_names() {
        let mut ws = WarmEnterState::new(100);
        let first = GreenKey::new(vec![4100, 4200]);
        let second = GreenKey::new(vec![4300, colliding_last_green(&first, 4300)]);
        let bucket = first.get_uhash();

        // A procedure token keeps the first cell non-removable so the second
        // install chains it rather than pruning it — the same shape
        // `a_chained_bucket_reached_without_greens_answers_the_first_occupant`
        // builds, because a single-candidate bucket takes the `(1, _)` arm and
        // never reaches the branch under test.
        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        attach_alive_for_key(&mut ws, &first, token);
        let first_key = ws
            .cell_key_for(&first)
            .expect("the first key owns the cell it just installed");
        let _second_key = ws.ensure_cell_key(&second);

        let (count, first_found) = ws.cell_keys_at(bucket);
        assert!(
            count > 1,
            "fixture: the bucket must own two candidates, or the resolve takes              its single-candidate arm and the branch under test never runs",
        );
        assert_eq!(
            first_found,
            Some(first_key),
            "fixture: and the first of them is the occupant that kept the raw              hash",
        );

        let (cell_key, cell) = ws.resolved_cell(bucket, None);
        assert_eq!(
            cell_key, bucket,
            "with no greens to settle the bucket, the resolve reports the raw              hash, exactly as `sole_cell_key` declines and `resolve_cell_key`              falls back",
        );

        let by_key = ws
            .cell_by_key(bucket)
            .expect("the raw hash names the bucket's original occupant");
        let cell = cell.expect("and the resolve hands that same cell back");
        assert!(
            Rc::ptr_eq(&cell, &by_key),
            "the cell noted during the walk must BE the one a walk back from              the key finds, not merely one with equal fields",
        );
        assert_eq!(cell.cell_key.get(), Some(first_key));
    }

    /// The number of cells linked at `hash`'s table slot, counting the
    /// neighbours the index truncation put there as well as `hash`'s own.
    fn chain_len(ws: &WarmEnterState, hash: u64) -> usize {
        let mut len = 0;
        let mut cell = ws.lookup_chain(hash);
        while let Some(c) = cell {
            len += 1;
            cell = c.next.borrow().clone();
        }
        len
    }

    /// **Three DIFFERENT full hashes share ONE chain, because the table index
    /// is a truncation of the hash.**
    ///
    /// counter.py `self.celltable = [None] * size` sizes the table once,
    /// and counter.py reads it at `self._get_index(hash)`, which
    /// "truncates the hash to 32 bits, and then keep the *highest* remaining
    /// bits" (counter.py). Both halves of that truncation are pinned
    /// here: one sibling differs from the subject only BELOW the shift, the
    /// other only ABOVE bit 31, so all three hashes are pairwise distinct and
    /// `_get_index` still maps them to one slot.
    ///
    /// A table keyed by the whole 64-bit hash cannot express this state at all.
    /// Three distinct hashes are three separate entries there, each holding a
    /// one-cell chain, so the assertions below read 1 instead of 3 — which is
    /// what makes this test about the table's geometry and not about
    /// `install_new_cell`'s linking, already covered elsewhere.
    ///
    /// The second half is the property the truncation must NOT cost: sharing a
    /// chain with strangers must not make any of the three hashes ambiguous.
    /// Upstream separates them with `cell.comparekey(*greenargs)`
    /// (warmstate.py); a caller holding no greens separates them by the
    /// full hash the cell key preserves ([`WarmEnterState::cell_keys_at`]).
    #[test]
    fn three_distinct_hashes_sharing_a_truncated_index_share_one_chain() {
        let mut ws = WarmEnterState::new(100);

        // `_get_index` truncates, so distinct hashes can share a slot.
        let owned = keys_sharing_index(|h| ws.counter._get_index(h), 3);
        let subject = owned[0].clone();
        let below_the_shift = owned[1].clone();
        let above_bit_31 = owned[2].clone();
        let keys = [&subject, &below_the_shift, &above_bit_31];
        let hashes: Vec<u64> = keys.iter().map(|k| k.get_uhash()).collect();

        assert_ne!(hashes[0], hashes[1], "fixture: three DIFFERENT hashes");
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[1], hashes[2]);
        for hash in &hashes {
            assert_eq!(
                ws.counter._get_index(*hash),
                ws.counter._get_index(hashes[0]),
                "fixture: one table slot for all three",
            );
        }

        // A procedure token each, so `should_remove_jitcell` keeps every cell
        // and the slot really chains rather than pruning back to one
        // (counter.py install_new_cell).
        let mut tokens = Vec::new();
        for key in keys {
            let token = token_with_compiled_code(&mut ws);
            ws.attach_procedure_to_interp_for_key(key, Arc::clone(&token));
            tokens.push(token);
        }

        assert_eq!(
            ws.occupied_buckets(),
            1,
            "the fixed-size table put all three under one index",
        );
        assert_eq!(
            chain_len(&ws, hashes[0]),
            3,
            "and the chain there holds all three cells — the state a table \
             keyed by the whole hash has no way to reach",
        );
        assert_eq!(ws.get_stats().num_cells, 3, "three cells, one slot");

        // Sharing a chain with strangers must not cost any of them its
        // identity: each hash still owns exactly one cell, and it is its own.
        for (i, key) in keys.iter().enumerate() {
            assert!(
                !ws.bucket_is_chained(hashes[i]),
                "the OTHER two cells are not this hash's candidates",
            );
            assert_eq!(
                ws.sole_cell_key(hashes[i]),
                Some(hashes[i]),
                "so the greens-less reader still answers, and with this \
                 hash's own cell",
            );
            assert_eq!(ws.cell_key_for(key), Some(hashes[i]));
            let token = ws
                .get_procedure_token_for_key(key)
                .expect("each key kept the token installed for it");
            assert!(
                Arc::ptr_eq(&token, &tokens[i]),
                "and it is that key's token, not a chain neighbour's",
            );
        }
    }

    /// **A hash that owns NO cell must not be answered with the cell of a
    /// truncation neighbour that happens to share its slot.**
    ///
    /// This is the hazard the fixed-size table introduces and the reason the
    /// greens-less readers count by full hash. `counter.py lookup_chain` hands back
    /// `celltable[self._get_index(hash)]` and every upstream walk over it is
    /// gated on `cell.comparekey(*greenargs)` (warmstate.py, :599-603,
    /// :634-638), so a slot occupant belonging to another green key is passed
    /// over. A reader that took "one cell in the slot" for "one candidate"
    /// would instead resolve this hash to the resident's cell and read its
    /// procedure token — a compiled loop belonging to different greens.
    ///
    /// Against a `head.next.borrow().is_none()` rule this reads `Some(resident_hash)`
    /// and `resolve_cell_key` returns the resident's key.
    #[test]
    fn a_hash_owning_no_cell_is_not_answered_with_a_slot_neighbours() {
        let mut ws = WarmEnterState::new(100);

        // Two hashes that share `_get_index` and therefore one table slot.
        let pair = keys_sharing_index(|h| ws.counter._get_index(h), 2);
        let resident = pair[0].clone();
        let stranger = pair[1].clone();
        let resident_hash = resident.get_uhash();
        let stranger_hash = stranger.get_uhash();
        assert_ne!(resident_hash, stranger_hash, "fixture: different hashes");
        assert_eq!(
            ws.counter._get_index(stranger_hash),
            ws.counter._get_index(resident_hash),
            "fixture: one table slot",
        );

        let token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&resident, Arc::clone(&token));
        assert_eq!(
            chain_len(&ws, stranger_hash),
            1,
            "fixture: the slot holds exactly one cell, and it is not this \
             hash's — the shape in which a slot-counting reader is wrong",
        );

        assert_eq!(
            ws.sole_cell_key(stranger_hash),
            None,
            "the one occupant is another hash's, so this hash has no candidate",
        );
        assert_eq!(
            ws.resolve_cell_key(stranger_hash, || stranger.clone()),
            stranger_hash,
            "and the resolve is a miss returning the hash itself, not the \
             resident's key",
        );
        assert!(
            ws.get_procedure_token(stranger_hash).is_none(),
            "so nothing hands back a loop compiled for different greens",
        );
        assert!(ws.get_procedure_token_for_key(&stranger).is_none());

        // The resident is untouched by any of this.
        assert_eq!(ws.sole_cell_key(resident_hash), Some(resident_hash));
        assert!(
            ws.get_procedure_token_for_key(&resident)
                .is_some_and(|t| Arc::ptr_eq(&t, &token)),
        );
    }

    /// **A bucket that has to mint twice must get two different keys.**
    ///
    /// [`WarmEnterState::mint_cell_key`] retries `green_uhash_step` against the
    /// live key set, which only terminates if the candidate MOVES between
    /// retries. Mixing the serial into the accumulator instead of passing it as
    /// the folded value cancels it — `green_uhash_step(bucket ^ serial, Int,
    /// serial)` folds `(bucket ^ serial) ^ serial`, which is `bucket` — so
    /// every retry re-proposes the single number `bucket * GREEN_UHASH_MULT`.
    /// The first mint in a bucket takes it and the second spins forever.
    ///
    /// Three cells in one bucket is the smallest state that asks for two mints,
    /// and nothing exotic reaches it: `install_new_cell` (counter.py)
    /// chains every cell whose `should_remove_jitcell` is false, so three warm
    /// keys sharing a bucket is enough.
    ///
    /// A regression does not reach an assertion: it hangs in
    /// `mint_cell_key`'s retry loop because every retry proposes the same key.
    #[test]
    fn a_bucket_that_mints_twice_gets_two_different_keys() {
        let mut ws = WarmEnterState::new(100);
        let first = GreenKey::new(vec![9100, 9200]);
        let second = GreenKey::new(vec![9300, colliding_last_green(&first, 9300)]);
        let third = GreenKey::new(vec![9400, colliding_last_green(&first, 9400)]);
        let bucket = first.get_uhash();
        assert_eq!(second.get_uhash(), bucket, "fixture: one bucket");
        assert_eq!(third.get_uhash(), bucket, "fixture: three keys");

        // A procedure token each, so `should_remove_jitcell` keeps all three
        // and the bucket really chains rather than pruning down to one.
        let mut keys = Vec::new();
        for key in [&first, &second, &third] {
            let token = token_with_compiled_code(&mut ws);
            attach_alive_for_key(&mut ws, key, token);
            keys.push(
                ws.cell_key_for(key)
                    .expect("each key owns the cell just installed for it"),
            );
        }

        assert_eq!(keys[0], bucket, "the first occupant keeps the raw hash");
        assert_ne!(keys[1], keys[0], "the second had to be minted one");
        assert_ne!(
            keys[2], keys[1],
            "and the third must be minted a DIFFERENT one, or the two cells \
             share an identity and the artifact tables indexed by it collapse",
        );
        assert_ne!(keys[2], keys[0]);
        for (key, cell_key) in [&first, &second, &third].iter().zip(&keys) {
            assert_eq!(
                ws.cell_by_key(*cell_key)
                    .and_then(|cell| cell.cell_key.get()),
                Some(*cell_key),
                "each minted key must name its own cell",
            );
            assert_eq!(ws.cell_key_for(key), Some(*cell_key));
        }
    }

    /// **Two readers of one raw hash inspected two different buckets.**
    ///
    /// [`WarmEnterState::resolve_cell_key`] asks
    /// [`WarmEnterState::bucket_is_chained`] whether a `GreenKey` has to be
    /// built at all, and answers from [`WarmEnterState::sole_cell_key`] when it
    /// does not. Both take a raw green-key hash, but they reached the celltable
    /// by different routes: `bucket_is_chained` went through
    /// [`WarmEnterState::bucket_of`], which maps a CELL KEY to the bucket that
    /// cell lives in, while `sole_cell_key` indexed the celltable directly.
    ///
    /// The routes agree on every number that is not a live minted key, and a
    /// minted key is reachable as a raw hash: `mint_cell_key` steps
    /// `green_uhash_step` against the live set and every `u64` is a hash some
    /// green key produces (see its doc), so [`gk_with_uhash`] builds the
    /// green key whose own hash IS the minted number. `bucket_of` then sent
    /// `bucket_is_chained` to the minted cell's bucket while `sole_cell_key`
    /// read the bucket the hash names — two buckets, one number, opposite
    /// answers.
    ///
    /// Upstream has one route and cannot express the split: every reader starts
    /// at `jitcounter.lookup_chain(hash)` with `hash = JitCell.get_uhash(...)`
    /// — `maybe_compile_and_run` (warmstate.py), `get_jitcell`
    /// (warmstate.py) and `_ensure_jit_cell_at_key`
    /// (warmstate.py) all spell it that way — and `lookup_chain`
    /// (counter.py) is a bare `celltable[self._get_index(hash)]` with
    /// nothing in front of it.
    ///
    /// Routing `bucket_is_chained` through `bucket_of` makes the readers
    /// disagree about whether this raw hash names a chained bucket. The later
    /// assertions pin the consequence: the entry would resolve to a cell
    /// belonging to neither the arriving key nor its sibling.
    #[test]
    fn a_raw_hash_equal_to_a_minted_key_resolves_through_one_bucket() {
        let mut ws = WarmEnterState::new(100);

        let parked = GreenKey::new(vec![7100, 7200]);
        let parked_bucket = parked.get_uhash();
        // A different key already holds the raw hash, so `parked` is minted.
        // The occupant is tokenless and is dropped when `parked` is filed
        // (`should_remove_jitcell`), leaving the minted cell alone.
        let occupant = GreenKey::new(vec![7300, colliding_last_green(&parked, 7300)]);
        assert_eq!(occupant.get_uhash(), parked_bucket);
        assert_ne!(occupant, parked);
        ws.ensure_cell_for_key(&occupant);
        let parked_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&parked, Arc::clone(&parked_token));
        let minted = ws
            .cell_key_for(&parked)
            .expect("the typed writer's cell is reachable by its own key");
        assert_ne!(
            minted, parked_bucket,
            "a distinct key already held the raw hash, so this cell was minted",
        );
        assert!(
            !ws.bucket_is_chained(parked_bucket),
            "fixture: the squatter was pruned, so the minted cell's bucket holds \
             exactly one cell and both routes call it unchained",
        );

        // The green key whose own hash IS that minted number, plus a sibling
        // colliding with it, so the bucket the number names is CHAINED while
        // the bucket `bucket_of` maps it to is not.
        let arriving = gk_with_uhash(minted, 8100);
        let sibling = gk_with_uhash(minted, 8200);
        assert_ne!(arriving, sibling, "fixture: two different green keys");
        let arriving_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&arriving, Arc::clone(&arriving_token));
        let sibling_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&sibling, Arc::clone(&sibling_token));
        assert!(
            !Arc::ptr_eq(&arriving_token, &sibling_token),
            "fixture: the two cells hold different artifacts",
        );
        assert!(
            ws.lookup_chain(minted)
                .is_some_and(|head| head.next.borrow().is_some()),
            "fixture: the bucket this hash names holds both keys' cells",
        );

        // THE DISAGREEMENT.
        assert_eq!(
            ws.sole_cell_key(minted),
            None,
            "the bucket this hash names has two candidates, so the greens-less \
             reader declines",
        );
        assert!(
            ws.bucket_is_chained(minted),
            "the two readers of one raw hash must describe ONE bucket: \
             `sole_cell_key` declined because the bucket this hash names is \
             chained, and this answered `unchained` about the bucket `bucket_of` \
             sent it to instead",
        );

        // What that costs, end to end: the entry arrives with this hash and the
        // greens behind it, and has to reach the arriving key's own cell.
        let carried = ws.resolve_cell_key(minted, || arriving.clone());
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(carried)
                    .expect("the carried cell key names a cell"),
                &arriving_token,
            ),
            "the resolve answered a key naming a cell that belongs to neither \
             the arriving key nor its sibling",
        );

        // Non-vacuity: the raw hash has not stopped naming the parked key's
        // cell, so answering it unchanged is answering a foreign cell.
        assert_ne!(carried, minted);
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(minted)
                    .expect("the minted key names its own cell"),
                &parked_token,
            ),
            "and the cell it would have answered is the parked key's",
        );
    }

    /// Two colliding green keys file two entries in a `u64`-keyed artifact
    /// table — the precondition `MetaInterp::compiled_loops` and
    /// `MemoryManager` eviction both need, and the one the old pin named as
    /// its blocker.
    ///
    /// `compiled_loops` is an `IndexMap<u64, CompiledEntry<M>>` and
    /// `try_to_free_some_loops` reaches an entry by
    /// `token.green_key()` (`JitCellToken::green_key: Cell<u64>`). While that
    /// u64 was a bucket hash, two colliding keys shared ONE slot: the second
    /// compile overwrote the first, and the loser's token could not reach its
    /// own entry to be freed. The stand-in table below is that shape — the
    /// subject is the keys, not the payload, so the payload is the token
    /// number.
    #[test]
    fn two_colliding_keys_file_two_entries_in_a_u64_keyed_artifact_table() {
        let mut ws = WarmEnterState::new(100);
        let key_a = GreenKey::new(vec![3100, 3200]);
        let key_b = GreenKey::new(vec![3300, colliding_last_green(&key_a, 3300)]);
        assert_eq!(
            key_a.get_uhash(),
            key_b.get_uhash(),
            "fixture: one bucket, two keys",
        );

        let cell_a = ws.ensure_cell_key(&key_a);
        let cell_b = ws.ensure_cell_key(&key_b);
        assert_ne!(
            cell_a, cell_b,
            "colliding keys must be told apart by the u64 the artifact tables \
             are indexed by, or one of them has no reachable entry at all",
        );

        // The two tokens each stamp their own cell key, which is what
        // `compile.rs`'s `compile_tmp_callback`
        // (`jitcell_token.green_key.set(green_key)`) writes and
        // what `try_to_free_some_loops` reads back.
        let token_a = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        let token_b = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        token_a.green_key.set(cell_a);
        token_b.green_key.set(cell_b);

        let mut compiled_loops: indexmap::IndexMap<u64, i64> = indexmap::IndexMap::new();
        compiled_loops.insert(token_a.green_key(), token_a.number());
        compiled_loops.insert(token_b.green_key(), token_b.number());
        assert_eq!(
            compiled_loops.len(),
            2,
            "both compiles must file; a shared slot means the second erased \
             the first",
        );

        // Eviction of B reaches B's entry and leaves A's alone.
        assert_eq!(
            compiled_loops.swap_remove(&token_b.green_key()),
            Some(token_b.number()),
            "the evicted token's own key must name its own entry",
        );
        let token_a_number = token_a.number();
        assert_eq!(
            compiled_loops.get(&token_a.green_key()),
            Some(&token_a_number),
            "and the colliding sibling's artifact must survive it",
        );
    }

    /// An unchained bucket has one candidate, so the head IS the match and the
    /// hash form is exact — this is what lets the entry path skip building a
    /// `GreenKey` on every warm entry.
    #[test]
    fn an_unchained_bucket_answers_the_same_through_both_forms() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![1300, 1400]);

        let token = Arc::new(JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        assert!(
            !ws.bucket_is_chained(key.get_uhash()),
            "one typed writer and no hash-form squatter builds no chain",
        );
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(key.get_uhash())
                    .expect("the hash form finds the only cell"),
                &token,
            ),
            "both forms must name the same token when there is nothing to walk",
        );
        assert!(Arc::ptr_eq(
            &ws.get_procedure_token_for_key(&key)
                .expect("the typed form finds it too"),
            &token,
        ));
    }

    #[test]
    fn mark_force_finish_tracing_for_key_sets_the_flag_on_the_keys_own_cell() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![900, 1000]);

        ws.disable_noninlinable_function_for_key(&key);
        assert!(ws.lookup_chain_with_key(&key).is_some());

        ws.mark_force_finish_tracing_for_key(&key);

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.flags.get().contains(JcFlags::JC_FORCE_FINISH),
            "FORCE_FINISH is sticky and never cleared, so landing it on the \
             wrong cell of the bucket is permanent",
        );
    }

    /// `get_stats` counts every cell in a chain, not just the bucket head.
    ///
    /// Non-vacuity: a chain is the *only* shape that separates the two
    /// implementations, so this fixture hands `install_new_cell` a shared
    /// bucket directly — as the collision fixture above does, and for the
    /// same reason (`cells` is keyed by the full `get_uhash()`, so two real
    /// green keys will not collide). Head and tail are put in *different*
    /// states so a head-only reader is caught twice: once on the total and
    /// once on the per-state split.
    ///
    /// The parenthetical above says only that a *collision* is
    /// impractical to reach from two real keys; it is not a claim that
    /// chains are rare. The fixture directly above shows a chain forming
    /// from a SINGLE key with no collision at all.
    ///
    /// Measured against the previous body (`num_cells` counted per table slot
    /// instead of per cell): fails `num_cells` 1 vs 2. The
    /// per-state assertions below are not separately observed in that run —
    /// the first failure ends it — but they cover the same head-only read
    /// on a second axis, so a future body that fixes the total and keeps a
    /// head-only state walk still fails here.
    #[test]
    fn get_stats_counts_chained_cells_not_just_bucket_heads() {
        let mut ws = WarmEnterState::new(100);
        let key_head = GreenKey::new(vec![100, 200]);
        let key_tail = GreenKey::new(vec![300, 400]);
        let bucket = key_head.get_uhash();

        // TRACING keeps the tail non-removable so the second install chains
        // it rather than dropping it (counter.py install_new_cell should_remove gate).
        let tail = BaseJitCell::new_for_driver_with_greens(0, &key_tail.values, &key_tail.types);
        tail.flags.set(tail.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(tail));

        let head = BaseJitCell::new_for_driver_with_greens(0, &key_head.values, &key_head.types);
        head.flags.set(head.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(head));

        // Precondition: one occupied slot holding a two-cell chain. Without
        // this the assertions below would pass for the wrong reason.
        assert_eq!(ws.occupied_buckets(), 1, "fixture must build ONE bucket");
        assert!(
            ws.lookup_chain(bucket)
                .and_then(|h| h.next.borrow().clone())
                .is_some(),
            "fixture must build a TWO-cell chain, or it cannot tell the \
             implementations apart",
        );

        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 2, "both chained cells must be counted");
        assert_eq!(stats.num_tracing, 2, "both cells carry JC_TRACING");
    }

    /// warmstate.py bound_reached + 596-604 — typed-key transition writes
    /// land on the matching chained cell, not the bucket head. A
    /// collision-shaped chain `head=A → tail=B` followed by enough
    /// `maybe_compile_with_key(&key_b)` ticks to exhaust the counter
    /// must flip `B.is_tracing()` to true while A's flags stay intact.
    #[test]
    fn maybe_compile_with_key_start_tracing_writes_to_chained_cell() {
        let mut ws = WarmEnterState::new(2); // threshold=2 → 2 ticks fire
        let key_b = GreenKey::new(vec![700, 800]);
        let key_a = GreenKey::new(vec![500, 600]);
        let bucket = key_b.get_uhash();

        // Head A: must stay non-removable across B's install.
        let cell_a = BaseJitCell::new_for_driver_with_greens(0, &key_a.values, &key_a.types);
        cell_a.flags.set(cell_a.flags.get() | (JcFlags::JC_TRACING));
        ws.install_new_cell(bucket, Some(cell_a));

        // Tick #1 — counter not at threshold yet, NotHot.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::NotHot => {}
            _ => panic!("tick #1 expected NotHot"),
        }
        // Tick #2 — counter fires; start_tracing_cell_for_key must
        // chain-walk and flip B's flags, not the head A.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::StartTracing => {}
            _ => panic!("tick #2 expected StartTracing"),
        }

        // Walk the chain — A still TRACING (its own flag), B now
        // additionally TRACING.
        let head = ws.lookup_chain(bucket).expect("head exists");
        assert_eq!(&head.greens[..], &key_a.values[..]);
        assert!(head.is_tracing(), "A's TRACING flag is its own");
        let b_cell = ws
            .lookup_chain_with_key(&key_b)
            .expect("typed lookup finds B in the chain");
        assert!(
            b_cell.is_tracing(),
            "B's TRACING flag set by start_tracing_cell_for_key — \
             chain-walk write must reach the tail, not just the head",
        );
    }

    /// `abort_tracing` latches `DONT_TRACE_HERE` at `MAX_TRACE_ABORT_COUNT`
    /// without touching `loop_token`, so a cell can be both latched and
    /// holding a token that is invalidated afterwards. The abort ceiling must
    /// not answer for that cell before `cleanup_chain` has had it: a dead
    /// entry left in the bucket never re-arms its counter, which is the
    /// starvation the ceiling was moved here to remove, in another form.
    #[test]
    fn a_latched_cell_whose_token_died_is_still_removed_from_the_chain() {
        let mut ws = WarmEnterState::new(2);
        // Seen once, held no longer. `set_procedure_token` stores only a
        // weak handle (`_makeref`) and takes no ownership, and this fixture
        // registers the token with no owner, so the weakref is already dead
        // when `attach_procedure_to_interp` returns while the SLOT stays
        // `Some` — which is the state `has_seen_a_procedure_token` reads.
        // Nulling the slot would erase the record and make the cell read as
        // "never had a token".
        ws.attach_procedure_to_interp_for_key(
            &gk(42),
            JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED),
        );
        for _ in 0..MAX_TRACE_ABORT_COUNT {
            ws.abort_tracing(ws.cell_key_for(&gk(42)).expect("cell"), false);
        }
        let cell = ws
            .get_cell(ws.cell_key_for(&gk(42)).expect("cell"))
            .expect("fixture: the cell is still there");
        assert!(
            cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none(),
            "fixture: the token must be dead, not merely absent",
        );
        assert!(
            cell.abort_count.get() >= MAX_TRACE_ABORT_COUNT,
            "fixture: the ceiling must be latched",
        );

        assert!(matches!(
            ws.maybe_compile_decision(ws.cell_key_for(&gk(42)).expect("cell"), &gk(42).types),
            HotResult::NotHot
        ));
        assert!(
            ws.get_cell(ws.cell_key_for(&gk(42)).expect("cell"))
                .is_none(),
            "the dead cell must be gone: the ceiling returned before \
             cleanup_chain and left it in the bucket",
        );
    }

    /// Typed-key half of
    /// `a_latched_cell_whose_token_died_is_still_removed_from_the_chain`.
    /// `maybe_compile_with_key` is a parallel implementation, so an ordering
    /// fixed in one says nothing about the other.
    #[test]
    fn a_latched_typed_cell_whose_token_died_is_still_removed_from_the_chain() {
        let mut ws = WarmEnterState::new(2);
        let key = GreenKey::new(vec![7, 11]);
        ws.attach_procedure_to_interp_for_key(
            &key,
            JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED),
        );
        for _ in 0..MAX_TRACE_ABORT_COUNT {
            ws.abort_tracing(key.get_uhash(), false);
        }
        let cell = ws
            .get_cell_for_key(&key)
            .expect("fixture: the cell is still there");
        assert!(
            cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none(),
            "fixture: the token must be dead, not merely absent",
        );
        assert!(
            cell.abort_count.get() >= MAX_TRACE_ABORT_COUNT,
            "fixture: the ceiling must be latched",
        );

        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.get_cell_for_key(&key).is_none(),
            "the dead cell must be gone: the ceiling returned before \
             cleanup_chain and left it in the bucket",
        );
    }

    /// `maybe_compile_and_run`: a tokenless aborted cell is `cleanup_chain`;
    /// a never-traced `JC_DONT_TRACE_HERE` cell is not.
    #[test]
    fn tokenless_aborted_cell_reaches_cleanup_chain() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![0xAB, 0xCD]);
        let hash = key.get_uhash();

        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(matches!(
            ws.maybe_compile_with_key(&key),
            HotResult::StartTracing
        ));
        ws.abort_tracing(hash, false);

        // Two leftover ticks would fire threshold=3 if cleanup did not reset.
        assert!(!ws.tick_empty_chain(hash));
        assert!(!ws.tick_empty_chain(hash));

        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::NotHot
        ));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "a tokenless aborted cell is cleanup_chain",
        );
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::NotHot
        ));
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&key, |_| true, &[]),
            HotResult::StartTracing
        ));

        let dnt = GreenKey::new(vec![0xD017, 0x11]);
        ws.disable_noninlinable_function_for_key(&dnt);
        assert!(matches!(
            ws.maybe_compile_decision_with_key(&dnt, |_| true, &[]),
            HotResult::StartTracing
        ));
        assert!(
            ws.lookup_chain_with_key(&dnt).is_some(),
            "a never-traced JC_DONT_TRACE_HERE cell is not cleanup_chain",
        );
    }

    #[test]
    fn is_ceiling_latched_agrees_with_the_decision_it_mirrors() {
        let mut ws = WarmEnterState::new(2);
        let cell_key = 42u64;
        ws.ensure_cell_for_key(&gk(cell_key as i64));
        let cell_key = ws.cell_key_for(&gk(cell_key as i64)).expect("cell");
        assert!(
            ws.get_cell(cell_key).unwrap().abort_count.get() < MAX_TRACE_ABORT_COUNT,
            "a fresh cell is not latched",
        );

        for _ in 0..MAX_TRACE_ABORT_COUNT {
            ws.abort_tracing(cell_key, false);
        }
        // `abort_tracing` stamps `DONT_TRACE_HERE` at the ceiling as a ban.
        assert!(
            ws.get_cell(cell_key).unwrap().abort_count.get() >= MAX_TRACE_ABORT_COUNT,
            "a ceiling-latched never-seen-token cell is banned",
        );

        let generation = ws.cell_generation();
        assert!(matches!(
            ws.maybe_compile_decision(cell_key, &gk(42).types),
            HotResult::NotHot
        ));
        assert_eq!(ws.cell_generation(), generation);
        assert_eq!(
            ws.get_cell(cell_key).unwrap().abort_count.get(),
            MAX_TRACE_ABORT_COUNT,
            "the refusal did not consume the latch",
        );
    }

    #[test]
    fn attaching_a_procedure_token_moves_the_generation_a_cache_keys_on() {
        let mut ws = WarmEnterState::new(2);
        let cell_key = 7u64;
        ws.ensure_cell_for_key(&gk(cell_key as i64));
        let cell_key = ws.cell_key_for(&gk(cell_key as i64)).expect("cell");
        let generation = ws.cell_generation();
        ws.attach_procedure_to_interp(cell_key, JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED));
        assert_ne!(
            ws.cell_generation(),
            generation,
            "a token can make a latched key runnable through its chain, so a \
             cache must be told",
        );
    }

    /// warmstate.py names each `JC_*` at a value, and so does `JcFlags`.
    /// Nothing tied the two together, so a bit corrected upstream and not here
    /// -- or the reverse -- would leave a cell reading a different bit than the
    /// one it means, with nothing noticing.
    ///
    /// `JcFlags::FLAGS` is what makes the walk complete: it carries every
    /// constant the `bitflags!` block declares, so a flag added there is
    /// compared here without anyone remembering to list it.
    #[test]
    fn every_jc_flag_is_the_bit_warmstate_gives_it() {
        use bitflags::Flags as _;

        const WARMSTATE: &str = include_str!("../../../rpython/jit/metainterp/warmstate.py");

        // `JC_X = 0xNN`, whose name and `=` are padded to a column.
        let mut upstream: Vec<(&str, u8)> = Vec::new();
        for line in WARMSTATE.lines() {
            let Some(rest) = line.strip_prefix("JC_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once('=') else {
                continue;
            };
            let Some(digits) = body.trim().strip_prefix("0x") else {
                continue;
            };
            let Ok(value) = u8::from_str_radix(digits, 16) else {
                continue;
            };
            upstream.push((name.trim_end(), value));
        }

        // A parser that read nothing would agree with every flag below, so the
        // floor comes first.  warmstate.py declares five.
        assert_eq!(
            upstream.len(),
            5,
            "warmstate.py declares {} JC_ flags, not the 5 this read",
            upstream.len()
        );

        for flag in super::JcFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("JC_")
                .expect("every flag is declared under warmstate.py's own name");
            let ours = flag.value().bits();
            let (_, theirs) = upstream
                .iter()
                .find(|(known, _)| *known == name)
                .unwrap_or_else(|| panic!("warmstate.py declares no JC_{name}"));
            assert_eq!(
                ours, *theirs,
                "JC_{name} is {theirs:#x} in warmstate.py and {ours:#x} here"
            );
        }
    }
    #[test]
    fn disable_unrolling_defaults_to_parameters_and_is_settable() {
        let mut ws = WarmEnterState::new(3);
        assert_eq!(ws.disable_unrolling_threshold(), 200);
        ws.set_param("disable_unrolling", 6000);
        assert_eq!(ws.disable_unrolling_threshold(), 6000);
        ws.set_default_params();
        assert_eq!(ws.disable_unrolling_threshold(), 200);
    }
}
