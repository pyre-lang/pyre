//! Residual-call dispatch: the tracer path for a callee that cannot be
//! folded or inlined and must be recorded as a residual `CALL_*` operation.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s
//! `opimpl_residual_call_*`; the executor fast paths call into
//! `majit-metainterp/executor.rs` (`executor.py`). PyPy keeps these opimpls
//! inside `pyjitpl.py`'s `MIFrame`; the split into this file is pyre-local
//! navigability, not a PyPy file boundary.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Covers the per-shape
//! dispatchers (`dispatch_residual_call_{iRd,iIRd,iIRFd}_kind`), the
//! executor fast paths (`try_fold_pure_call_via_executor`,
//! `try_execute_residual_call_via_executor`), opcode selection and arg
//! binding, the pre-call vable/vref sync, result writeback, and the
//! residual-call body classification helpers. The `residual_call_*` opname
//! arms themselves stay in `handle` (mod.rs) and call into these.

use super::symbolic_fold::try_fold_registered_symbolic_residual;
use super::{InlineParentBlackhole, InlineParentFrame, *};
use pyre_interpreter::{locals_w, locals_w_mut};

/// The published fnaddrs whose registered path matches `pred`, collected once.
///
/// `jit_trace_fnaddrs()` builds a fresh `Vec` of every published residual on
/// each call, so asking it a name question per residual call rebuilds and
/// rescans the whole table.  The addresses are fixed for the process, so the
/// answer is a set taken once.
fn fnaddr_set(pred: impl Fn(&str) -> bool) -> std::collections::HashSet<i64> {
    pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .filter_map(|(name, addr)| pred(name).then_some(addr))
        .collect()
}

/// Address a walker fold compares a residual's funcbox against.
///
/// The JitCode constant is patched from `jit_trace_fnaddrs` (`fnaddr_for_target`
/// / `direct_funcptr_value`, then `patch_constants_i_fnaddrs`), so the match
/// must use [`crate::runtime_fnaddr_patch::runtime_fnaddr_by_path`] rather than
/// a second `fn as usize`: wasm32 table slots are not stable across cast sites.
///
/// `None` on every path falls back to `raw` so an unpublished name cannot
/// silently disable the fold on native, where the two casts agree. An
/// unpublished path is a unit-test failure, not a production decline.
fn walker_published_fnaddr(paths: &[&'static str], raw: i64) -> i64 {
    paths
        .iter()
        .copied()
        .find_map(crate::runtime_fnaddr_patch::runtime_fnaddr_by_path)
        .unwrap_or(raw)
}

static TAKE_LAST_EXEC_CTX_FNADDR: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| {
    walker_published_fnaddr(
        &["pyre_interpreter::call::take_last_exec_ctx"],
        pyre_interpreter::call::take_last_exec_ctx as *const () as usize as i64,
    )
});

/// Residuals whose registered path names a `bigint` helper: their Ref slots
/// carry `*mut BigInt` payloads, not `PyObject`s.
static BIGINT_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| fnaddr_set(|name| name.contains("bigint")));

/// The eval-loop `BINARY_OP` helper, which can arrive with
/// `RuntimeHelperKind::None` and so is identified by address.
static BINARY_VALUE_FROM_TAG_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| fnaddr_set(|name| name.ends_with("binary_value_from_tag")));

/// The eval-loop `COMPARE_OP` helper, identified the same way.
static COMPARE_VALUE_FROM_TAG_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| fnaddr_set(|name| name.ends_with("compare_value_from_tag")));

/// `space.newutf8` / `w_str_from_storage_and_length`, identified by path.
static NEWUTF8_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| fnaddr_set(|name| name.ends_with("w_str_from_storage_and_length")));

/// Void `list_write_barrier` only. [`is_list_write_barrier`] also matches
/// `prepare_list_ref_store` / `current_gc_ref`, which return the list or
/// value and must not be recorded as void `CondCallGcWb`.
static VOID_LIST_WRITE_BARRIER_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| {
        fnaddr_set(|name| {
            name.ends_with("::listobject::list_write_barrier")
                || name == "pyre_object::list_write_barrier"
        })
    });

/// Object-strategy in-place store: write-barrier plus the relocated value.
static PREPARE_LIST_REF_STORE_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| {
        fnaddr_set(|name| {
            name.ends_with("::listobject::prepare_list_ref_store")
                || name == "pyre_object::prepare_list_ref_store"
        })
    });

/// Post-safepoint reload residual. Identity after `CondCallGcWb`.
static CURRENT_GC_REF_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| {
        fnaddr_set(|name| {
            name.ends_with("::listobject::current_gc_ref") || name == "pyre_object::current_gc_ref"
        })
    });

/// `frame_locals_proxy::proxy_list_append`, identified by leaf. The slot
/// scan records this wrapper; the append descent substitutes
/// `w_list_append_inner` when the address matches.
static PROXY_LIST_APPEND_FNADDRS: std::sync::LazyLock<std::collections::HashSet<i64>> =
    std::sync::LazyLock::new(|| {
        fnaddr_set(|name| {
            name.rsplit("::").next().is_some_and(|leaf| {
                let leaf = leaf.strip_prefix("__majit_call_target_").unwrap_or(leaf);
                leaf == "proxy_list_append"
            })
        })
    });

pub(crate) fn is_proxy_list_append_residual<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
) -> bool {
    residual_funcptr_addr(ctx, funcptr)
        .is_some_and(|addr| PROXY_LIST_APPEND_FNADDRS.contains(&(addr as i64)))
}

fn residual_funcptr_addr<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
) -> Option<usize> {
    ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
        majit_ir::Value::Int(n) => Some(n as usize),
        _ => None,
    })
}

/// Record the list write-barrier residual as `rewrite.py gen_write_barrier`
/// would: `CondCallGcWb` on the list, not a `CallN`/`CallR` every append.
///
/// `prepare_list_ref_store` still returns the value — only the barrier half
/// becomes `CondCallGcWb`. `current_gc_ref` after that barrier is identity
/// (`CondCallGcWb` is a header-flag check, not a moving collection). After a
/// real collecting residual the reload stays a `Call*`.
///
/// Returns `(recorded, rewrote)`. `rewrote` skips `_record_helper_varargs`
/// heapcache invalidation (`COND_CALL_GC_WB` is not a call).
fn record_list_write_barrier_residual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
    allboxes: &[OpRef],
    call_opcode: OpCode,
    descr: DescrRef,
) -> (OpRef, bool) {
    let Some(addr) = residual_funcptr_addr(ctx, funcptr) else {
        return (
            ctx.trace_ctx
                .record_op_with_descr(call_opcode, allboxes, descr),
            false,
        );
    };
    let addr_i = addr as i64;
    if VOID_LIST_WRITE_BARRIER_FNADDRS.contains(&addr_i) {
        let list_op = *allboxes.get(1).unwrap_or(&allboxes[0]);
        return (
            ctx.trace_ctx.record_op(OpCode::CondCallGcWb, &[list_op]),
            true,
        );
    }
    if PREPARE_LIST_REF_STORE_FNADDRS.contains(&addr_i) {
        let list_op = *allboxes.get(1).unwrap_or(&allboxes[0]);
        let _wb = ctx.trace_ctx.record_op(OpCode::CondCallGcWb, &[list_op]);
        // Executor stamps the (possibly relocated) value onto this box.
        let value_op = *allboxes.get(2).unwrap_or(&allboxes[0]);
        return (value_op, true);
    }
    if CURRENT_GC_REF_FNADDRS.contains(&addr_i) && last_non_guard_is_cond_call_gc_wb(ctx) {
        let obj_op = *allboxes.get(1).unwrap_or(&allboxes[0]);
        return (obj_op, true);
    }
    (
        ctx.trace_ctx
            .record_op_with_descr(call_opcode, allboxes, descr),
        false,
    )
}

/// `CondCallGcWb` cannot raise, but the residual it replaced may have been
/// classified `can_raise` and already grown a `GuardNoException`. Skip those
/// when deciding whether the preceding barrier was a moving safepoint.
fn last_non_guard_is_cond_call_gc_wb<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> bool {
    // Byte-mode `ops()` is empty until materialize. Opcode lives on
    // `FrontendSlot` (`history.py AbstractResOp.getopnum`).
    let n = ctx.trace_ctx.num_ops();
    for i in (0..n).rev() {
        let Some(opcode) = ctx.trace_ctx.opcode_at(i) else {
            continue;
        };
        match opcode {
            OpCode::GuardNoException | OpCode::GuardNotForced | OpCode::GuardException => {
                continue;
            }
            OpCode::CondCallGcWb => return true,
            _ => return false,
        }
    }
    false
}

/// Which of [`flush_active_frame_escape`]'s two flushes committed the resume
/// pc.  They differ in exactly the way the walk-end commit contract cares
/// about, so the epilogue cannot classify the leg without being told.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EscapeResumeKind {
    /// The latched mid-expression operand stack resolved.  The resume pc still
    /// re-runs the escaping opcode — both flushes take the same `py_pc` and the
    /// same `last_instr = py_pc - 1` — but this path is gated:
    /// [`escape_opcode_window_clean`] ran at the residual before the latch.
    Exact,
    /// The latch did not resolve and the flush fell back to the merge-point
    /// state.  Same resume pc, but NO gate ran on this path at any point.
    RerunsOpcode,
}

thread_local! {
    static ACTIVE_FRAME_ESCAPE: std::cell::Cell<Option<(usize, Option<usize>)>> =
        const { std::cell::Cell::new(None) };
    static ACTIVE_FRAME_ESCAPE_STACK: std::cell::RefCell<Option<Vec<OpRef>>> =
        const { std::cell::RefCell::new(None) };
    /// Step-0 attribution probe: classification of the most recent matched
    /// escape, read at the force site to attribute the token clear (and hence
    /// the abort) to the portal or the published-callee disjunct.
    static LAST_ESCAPE_WAS_CALLEE_ONLY: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static COMMITTED_FRAME_ESCAPE_PC: std::cell::Cell<Option<(usize, EscapeResumeKind)>> =
        const { std::cell::Cell::new(None) };
    /// The concrete frame a residual forced that the escape flush did NOT
    /// write: an inline callee published on the execution context, matched by
    /// [`flush_active_frame_escape`]'s second disjunct while the flush stays
    /// keyed on the traced virtualizable (the portal frame).
    ///
    /// `virtualizable.py write_boxes` writes the array of the virtualizable it
    /// forces, with no way to decline.  pyre forces here on behalf of a frame
    /// that is NOT the traced virtualizable, so nothing writes that frame's
    /// `locals_cells_stack_w`, and every value the compiled trace was holding
    /// in a register for it stays absent.  A blackhole level resumed on such a
    /// frame reads those slots as Python NULL.
    static UNFLUSHED_ESCAPED_CALLEE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Pre-flush frame state captured by [`flush_active_frame_escape`] so a
    /// post-call commit withdrawal can put the live frame back.  The legacy
    /// replay's correctness contract is "the live frame still holds pre-walk
    /// state"; a committed force-flush breaks it, so the replay leg restores
    /// this before re-entering.  Which leg runs is only known at walk end, so
    /// the withdrawal arms `ESCAPE_FLUSH_UNDO_PENDING` and the epilogue
    /// decides.
    static ESCAPE_FLUSH_UNDO: std::cell::RefCell<Option<EscapeFlushUndo>> =
        const { std::cell::RefCell::new(None) };
    /// Set when the force arm withdrew its commit: the pre-flush frame has to
    /// come back, but only on the legacy-replay leg (see
    /// `mark_escape_flush_undo_pending`).
    static ESCAPE_FLUSH_UNDO_PENDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Opcode-scoped purity window: `(py_pc, every_prior_residual_reentrant)`
    /// for the Python opcode currently being walked.  Re-executing a committed
    /// escape re-runs the WHOLE opcode, so the latch gate must know whether any
    /// EARLIER residual of the same opcode could have done something — not just
    /// the escaping one (rich-compare fallback chains run two residuals in one
    /// opcode).
    ///
    /// The verdict comes from each residual's DECLARED effect class
    /// ([`escape_opcode_window_note`], `EffectInfo::check_is_elidable` or
    /// `LoopInvariant`), not from counting what happened.  That is the axis
    /// upstream decides on: the licence to place a resume pc behind an executed
    /// residual is the codewriter-time `EF_ELIDABLE_CANNOT_RAISE` registration
    /// (`jtransform.py:620-630`), and upstream's only op counters are
    /// profiling-only and gate nothing (`jitprof.py`, `count_ops` is
    /// `pass`).
    ///
    /// Reset at walk start.  Residuals are the only executor that can put
    /// something irreversible between two points of ONE opcode: the eager
    /// journaled folds (`fbw_store_journal_push` / `fbw_list_journal_push_append` /
    /// `fbw_cell_store_journal_push`) each terminate their own opcode
    /// (STORE_SUBSCR, the `append` CALL, STORE_NAME/STORE_GLOBAL), so no
    /// residual of the same opcode instance can follow one.
    static ESCAPE_OPCODE_WINDOW: std::cell::Cell<Option<(usize, bool)>> =
        const { std::cell::Cell::new(None) };
    /// Force-time image of the single live tracing frame.  The
    /// color-indexed concrete banks cease to exist when dispatch unwinds, so
    /// the MIFrame must be assembled beside `tracing_after_residual_call` and
    /// carried to the walk-end VableEscaped leg.
    static FBW_SINGLE_FRAME_BLACKHOLE: std::cell::RefCell<Option<LatchedSingleFrameBlackhole>> =
        const { std::cell::RefCell::new(None) };
    /// Force-time image of the paused caller chain plus the live innermost
    /// tracing frame.  Like the single-frame latch, this outlives the walk
    /// contexts whose concrete banks supplied it.
    static FBW_MULTI_FRAME_BLACKHOLE: std::cell::RefCell<Option<LatchedMultiFrameBlackhole>> =
        const { std::cell::RefCell::new(None) };
}

/// The concrete red frame owned by the current inlined MIFrame.
///
/// RPython carries this identity directly on every `MIFrame`; pyre's walker
/// brackets the corresponding per-thread execution state with
/// [`InlineConcreteFrameGuard`].  Vable writes use this accessor to keep that
/// frame's own heap image coherent for a later multi-frame blackhole handoff.
///
/// Read back out of the guard's root rather than from a raw copy: the value
/// is compared for identity against the concrete bank's own `Value::Ref`,
/// which the collector forwards, so a relocated frame would make the two
/// disagree and silently drop the mirror write.
pub(crate) fn current_inline_concrete_frame() -> usize {
    INLINE_CONCRETE_FRAME.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(0, |owner_root| owner_root.get().0)
    })
}

pub(crate) struct EscapeFlushUndo {
    frame: usize,
    /// Owner-root for `frame`. A minor collection forwards the nursery
    /// object and then poisons the old address, so a later
    /// `gc_current_object_address` on the captured word cannot find it.
    /// [`Self::current_frame`] re-reads the slot the collector updates.
    frame_root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    last_instr: isize,
    valuestackdepth: usize,
    pub(crate) slots: Vec<pyre_object::PyObjectRef>,
    /// The region as the flush left it.  The restore compares against this to
    /// tell a slot the flush wrote from one the residual's user Python wrote
    /// after the capture; empty means no flush image was recorded, and the
    /// restore then puts every captured slot back as it always did.
    pub(crate) flush_image: Vec<pyre_object::PyObjectRef>,
}

impl EscapeFlushUndo {
    /// Address of the frame now. The captured word stays the identity the
    /// guards compare; dereferences go through the root.
    fn current_frame(&self) -> usize {
        self.frame_root
            .as_ref()
            .map(|root| root.get().0)
            .unwrap_or(self.frame)
    }
}

fn owner_root_if_gc(addr: usize) -> Option<majit_gc::shadow_stack::OwnerRootGuard> {
    (addr != 0 && majit_gc::gc_owns_object(addr))
        .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(addr)))
}

/// Owner-root for the walk's live frame across one residual.
///
/// The call collects. Afterwards the nursery word is poison, and
/// `gc_current_object_address` cannot recover the new address. Drop
/// publishes the forwarded word back onto the snapshot so the next
/// residual does not start from the abandoned address.
struct LiveFrameRoot<Sym: WalkSym> {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    sym: *mut Sym,
}

impl<Sym: WalkSym> LiveFrameRoot<Sym> {
    fn current(&self, fallback: usize) -> usize {
        self.root
            .as_ref()
            .map(|root| root.get().0)
            .unwrap_or(fallback)
    }
}

impl<Sym: WalkSym> Drop for LiveFrameRoot<Sym> {
    fn drop(&mut self) {
        let Some(root) = self.root.as_ref() else {
            return;
        };
        let now = root.get().0;
        if now != 0 && !self.sym.is_null() {
            unsafe { (*self.sym).set_live_vable_frame_addr(now) };
        }
    }
}

/// The operand stack an abort image publishes, resolved from the walker's
/// OpRef mirror while the walk's concrete side tables are still live.
///
/// The detached tracing snapshot cannot supply it.  A root walk keeps its
/// virtualizable purely symbolic — `setarrayitem_vable_via_metainterp` mirrors
/// into a concrete frame only through `current_inline_vable_target`, i.e. only
/// for the frame an INLINE level owns — so the root snapshot's
/// `locals_cells_stack_w` still holds the stack from before the walk ran.  The
/// locals half has no such gap: `write_back_outer_locals` publishes those from
/// the virtualizable shadow.
pub(crate) struct MirrorStackImage {
    /// Python pc of the opcode the walk stopped inside.  Pairs with the slots:
    /// the mirror reflects the depth ON ENTRY to this opcode, which is the
    /// `last_instr = py_pc - 1` coordinate the codewriter stores for it.
    pub(crate) py_pc: usize,
    pub(crate) slots: Vec<pyre_object::PyObjectRef>,
}

pub(crate) struct LatchedSingleFrameBlackhole {
    pub(crate) miframe: majit_metainterp::MIFrame,
    pub(crate) last_exc_value: i64,
    pub(crate) raising_exception: bool,
    /// The walker's resolved operand-stack image, captured by whichever latch
    /// stops the walk.  The adopter uses it for legs that stop INSIDE an opcode
    /// (`WalkAbort` and `VableEscape`) and uses the frame's own snapshot array
    /// for `ABORT_TOO_LONG`, whose post-step coordinate is an opcode boundary
    /// the array does describe.
    ///
    /// `None` means the walk had no resolvable mirror, and the adopt then
    /// declines rather than publishing a partial stack.
    pub(crate) mirror_stack: Option<MirrorStackImage>,
    /// Which latch published this image.  Every latch supplies the resume
    /// coordinate from a different source, so an image that resumes at a
    /// coordinate the jitcode cannot dispatch is only attributable once the
    /// adopter can name its producer.
    pub(crate) origin: &'static str,
}

pub(crate) struct LatchedMultiFrameBlackhole {
    pub(crate) framestack: majit_metainterp::MIFrameStack,
    pub(crate) last_exc_value: i64,
    pub(crate) raising_exception: bool,
    pub(crate) mirror_stack: Option<MirrorStackImage>,
    /// Which latch published this image — see the single-frame counterpart.
    pub(crate) origin: &'static str,
}

pub(crate) fn single_frame_blackhole_cell_ptr()
-> *const std::cell::RefCell<Option<LatchedSingleFrameBlackhole>> {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|cell| cell as *const _)
}

pub(crate) fn take_single_frame_blackhole() -> Option<LatchedSingleFrameBlackhole> {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| slot.borrow_mut().take())
}

/// Read-only counterpart of the single-frame adopter's operand-stack gate, for
/// the legs that publish from the walker mirror instead of from the tracing
/// snapshot's slot array.
///
/// [`latch_abort_blackhole`] proves every capability the snapshot-array source
/// needs, ending at `can_publish_frame_stack`. The mirror carries a condition of
/// its own: `capture_frame_stack_from_mirror` declines when the recorded height
/// disagrees with the forward analysis at the mirror's `py_pc`, rather than
/// publish a shifted stack. A leg whose adopt treats a decline as an assert has
/// to answer that here, before it commits to the abort.
pub(crate) fn latched_single_frame_mirror_publishable() -> bool {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
        let borrow = slot.borrow();
        let Some(latched) = borrow.as_ref() else {
            return false;
        };
        let Some(mirror) = latched.mirror_stack.as_ref() else {
            return false;
        };
        let Ok(jitcode_index) = i32::try_from(latched.miframe.jitcode.index()) else {
            return false;
        };
        let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index);
        let vable_frame = latched
            .miframe
            .ref_value_for_blackhole(frame_reg as usize)
            .unwrap_or(0) as usize;
        vable_frame != 0
            && crate::state::capture_frame_stack_from_mirror(
                vable_frame,
                mirror.py_pc,
                &mirror.slots,
            )
            .is_some()
    })
}

pub(crate) fn multi_frame_blackhole_cell_ptr()
-> *const std::cell::RefCell<Option<LatchedMultiFrameBlackhole>> {
    FBW_MULTI_FRAME_BLACKHOLE.with(|cell| cell as *const _)
}

pub(crate) fn take_multi_frame_blackhole() -> Option<LatchedMultiFrameBlackhole> {
    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| slot.borrow_mut().take())
}

pub(crate) fn multi_frame_blackhole_is_latched() -> bool {
    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| slot.borrow().is_some())
}

/// True when an abort-recovery image is already staged.
///
/// The INNERMOST abort owns the handoff: its [`WalkContext`] is the one holding
/// the concrete banks of the frame the walk actually stopped in, and
/// `build_multi_frame_miframe` already appends every paused caller from
/// [`WalkSession::framestack`].  As the error propagates back out through the
/// enclosing `walk()` frames, each would otherwise re-latch and overwrite that
/// image with one rooted at the caller — resuming the blackhole above the frame
/// whose opcode was left half-executed.
pub(crate) fn abort_blackhole_latched() -> bool {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| slot.borrow().is_some())
        || FBW_MULTI_FRAME_BLACKHOLE.with(|slot| slot.borrow().is_some())
}

pub(crate) fn reset_single_frame_blackhole() {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
        *slot.borrow_mut() = None;
    });
    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

/// Name each decline under `PYRE_FBW_DEBUG_ABORT`, the way `build_multi_frame_
/// miframe`'s `s2dbg!` and `try_adopt_multi_frame_blackhole`'s `mfdbg!` name
/// theirs: an unlatched abort and a latch that was never reached both end in the
/// same replay, and the two want different fixes.
macro_rules! latchdbg {
    ($($a:tt)*) => {
        if fbw_debug_abort_enabled() {
            eprintln!("[latch-decline] {}", format!($($a)*));
        }
    };
}

/// Snapshot the live meta-interpreter framestack for an abort at `resume_pc`.
/// RPython's `run_blackhole_interp_to_cancel_tracing` routes every
/// `SwitchToBlackhole` through `convert_and_run_from_pyjitpl`, which copies the
/// current frame and all paused callers at their current program counters.
/// [`WalkContext`] and [`WalkSession::framestack`] own the corresponding banks.
///
/// The caller must supply the next instruction for a completed step, or the
/// stopped instruction for a capability gap.  Propagation through enclosing
/// `walk()` calls must preserve the first image; otherwise the caller would
/// replace the innermost frame where execution stopped.
///
/// Return `false` without publishing a partial image when any live value is
/// unresolved.  A zero-effect walk may then use the legacy entry replay;
/// an effectful walk must keep recording until a complete image can be built,
/// because replaying it would apply the effect twice.
/// Resolve `WalkFrameStateData::vstack_boxes` — the walker's counterpart of
/// `MIFrame.registers_r` snapshotted by `get_list_of_active_boxes`
/// (`pyjitpl.py`) — into the concrete operand stack an abort image publishes.
///
/// Same resolution [`flush_escape_state_with_latched_stack`] performs for the
/// escape flush, and for the same reason: the walk-abort latch and the
/// vable-escape latch both stop inside a Python opcode, where no other source
/// describes the operand stack.  The tracing snapshot's array is not it — a
/// root walk mirrors nothing into that frame (see [`MirrorStackImage`]).
///
/// A recorded concrete NULL is a REAL operand — it is `PUSH_NULL`'s
/// `self_or_null` sentinel — so only `GcRef::NO_CONCRETE`, which is what
/// "unavailable" looks like, rejects a slot.  Any unresolved slot declines the
/// whole image, leaving the abort on the legacy entry replay.
fn capture_vstack_mirror_image<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    origin: &'static str,
) -> Option<MirrorStackImage> {
    if !ctx.vstack_valid {
        latchdbg!("origin={origin} mirror-invalid");
        return None;
    }
    let mut slots = Vec::with_capacity(ctx.frame_state.borrow().vstack_boxes.len());
    for &opref in ctx.frame_state.borrow().vstack_boxes.iter() {
        match ctx.trace_ctx.concrete_of_opref(opref) {
            Some(majit_ir::Value::Ref(value)) if value != majit_ir::GcRef::NO_CONCRETE => {
                slots.push(value.as_usize() as pyre_object::PyObjectRef);
            }
            other => {
                latchdbg!(
                    "origin={origin} mirror-slot {}/{} unresolved opref={opref:?} \
                     concrete={other:?} pypc={} boxes={:?}",
                    slots.len(),
                    ctx.frame_state.borrow().vstack_boxes.len(),
                    ctx.vstack_cur_pypc,
                    ctx.frame_state.borrow().vstack_boxes,
                );
                return None;
            }
        }
    }
    Some(MirrorStackImage {
        py_pc: ctx.vstack_cur_pypc as usize,
        slots,
    })
}

/// Rebuild frame 0's post-CALL operand stack from the exact caller image
/// captured when the first inline level was pushed.
///
/// RPython keeps this stack in the root `MIFrame.registers_r` while its callee
/// frames are paused above it.  Pyre's live `PyFrame` is still parked at the
/// outer loop header, so the multi-frame blackhole adopter must publish the
/// paused caller image before it drives those frames.  Every pre-existing
/// stack slot comes from `call_stack_overrides`; the top result slot is left
/// null because `drive_multi_frame_blackhole` writes it when the callee returns.
fn capture_root_parent_resume_stack<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<MirrorStackImage> {
    // Every refusal below names itself: an adopter that declines on a missing
    // root-parent stack reports only "not capturable", so without these the
    // seven silent `?` exits are indistinguishable from "never reached".
    macro_rules! need {
        ($opt:expr, $what:literal) => {
            match $opt {
                Some(value) => value,
                None => {
                    latchdbg!("root-parent-stack: {}", $what);
                    return None;
                }
            }
        };
    }
    let session = ctx.session.borrow();
    let parent = need!(
        need!(session.first_inline(), "empty framestack")
            .parents
            .first(),
        "innermost frame has no parent"
    );
    let call_jit_pc = need!(parent.call_jitcode_pc, "parent has no call_jitcode_pc");
    let pjc = need!(
        crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32),
        "parent jitcode index unresolved"
    );
    if pjc.code_ptr.is_null() {
        latchdbg!("root-parent-stack: parent code_ptr null");
        return None;
    }
    let resume_py_pc = need!(
        resolve_parent_resume_py_pc(parent),
        "parent resume py_pc unresolved"
    ) as usize;
    let depth = need!(
        crate::liveness::liveness_for(pjc.code_ptr)
            .depth_at_py_pc()
            .get(resume_py_pc)
            .copied(),
        "no depth_at_py_pc for the parent resume pc"
    ) as usize;
    // `MIFrame`'s operand stack starts after locals *and* cell/freevar
    // entries (`pyframe.py:111`).  `call_stack_overrides` uses that same
    // semantic slot space through `caller_sym.nlocals()`; using only
    // `varnames` here would shift every closure caller's paused stack into
    // its cells when the multi-frame blackhole image is adopted.
    let code = unsafe { &*pjc.code_ptr };
    let nlocals = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let call = need!(
        decode_op_at(&pjc.jitcode.code, call_jit_pc),
        "parent call op does not decode"
    );
    let result_bank = need!(
        call.argcodes.chars().last(),
        "parent call op has no argcodes"
    );
    let result_slot = (result_bank != 'v' && depth != 0).then_some(nlocals + depth - 1);
    let mut slots = Vec::with_capacity(depth);
    for slot in nlocals..nlocals + depth {
        if result_slot == Some(slot) {
            slots.push(pyre_object::PY_NULL);
            continue;
        }
        let Some(value) = parent
            .call_stack_overrides
            .iter()
            .find_map(|&(candidate, value)| (candidate == slot).then_some(value))
            .or_else(|| {
                // `_opimpl_setarrayitem_vable` (`pyjitpl.py`) writes each
                // operand-stack slot through `setarrayitem_vable_via_metainterp`
                // → `store_live_frame_array_slot` onto the inline callee's
                // `callee_shadow.concrete_frame` before a nested sub-walk
                // starts.  `paused_concrete_frame` is that shadow's frame
                // (None on a root parent, whose live PyFrame is still parked
                // at the loop header).  Fill a missing override from that
                // array so a complete image can latch when the CALL-site
                // capture was sparse.
                parent
                    .paused_concrete_frame()
                    .and_then(|frame| crate::state::concrete_stack_value(frame, slot))
            })
        else {
            latchdbg!(
                "root-parent-stack: no call_stack_override for slot {slot} \
                 (nlocals={nlocals} depth={depth} resume_py_pc={resume_py_pc} \
                 call_jit_pc={call_jit_pc} have={:?})",
                parent
                    .call_stack_overrides
                    .iter()
                    .map(|&(s, _)| s)
                    .collect::<Vec<_>>()
            );
            return None;
        };
        slots.push(value);
    }
    Some(MirrorStackImage {
        py_pc: resume_py_pc,
        slots,
    })
}

pub(crate) fn latch_abort_blackhole<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    resume_pc: usize,
    origin: &'static str,
) -> bool {
    if !ctx.is_authoritative_executor {
        latchdbg!("origin={origin} not-authoritative");
        return false;
    }
    let last_exc_value = match ctx.last_exc_value_concrete() {
        ConcreteValue::Ref(value) => value as i64,
        _ => 0,
    };

    if ctx.session.borrow().at_portal() && !ctx.fbw_mode.inline_subwalk {
        let Some((jitcode, cf_addr, live_root_addr)) = (unsafe {
            if ctx.fbw_mode.snapshot_sym.is_null() {
                None
            } else {
                let sym = &*ctx.fbw_mode.snapshot_sym;
                let jitcode = sym.jitcode();
                let forwarded_live_root = match ctx.trace_ctx.lookup_opref_concrete(sym.frame()) {
                    Some(majit_ir::Value::Ref(value)) if value.0 != 0 => value.0,
                    _ => sym.live_vable_frame_addr(),
                };
                (!jitcode.is_null()).then(|| {
                    (
                        (&(*jitcode).payload).jitcode.clone(),
                        sym.tracing_vable_frame_addr(),
                        forwarded_live_root,
                    )
                })
            }
        }) else {
            latchdbg!("origin={origin} no-snapshot-sym-jitcode");
            return false;
        };
        let Some(miframe) = build_trace_too_long_single_frame_miframe(ctx, jitcode, resume_pc)
        else {
            latchdbg!("origin={origin} sf-build-miframe");
            return false;
        };
        // `walk()` has already executed this step, so returning TraceTooLong
        // is safe only when every pre-drive adopter gate is known to pass.
        // RPython's `run_blackhole_interp_to_cancel_tracing()` cannot return
        // to entry replay. Keep the same boundary: incomplete images merely
        // keep recording until a later step supplies a complete handoff.
        let Some(jitcode_index) = i32::try_from(miframe.jitcode.index()).ok() else {
            latchdbg!("origin={origin} sf-jitcode-index");
            return false;
        };
        if ctx.trace_ctx.virtualizable_info().is_none()
            || crate::state::concrete_nlocals(cf_addr).is_none()
        {
            latchdbg!("origin={origin} sf-no-vinfo-or-nlocals");
            return false;
        }
        let root_addr = if live_root_addr != 0 {
            live_root_addr
        } else {
            cf_addr
        };
        let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index);
        let vable_frame = miframe
            .ref_value_for_blackhole(frame_reg as usize)
            .unwrap_or(0) as usize;
        if vable_frame == 0
            || vable_frame != root_addr
            || crate::state::capture_frame_locals(vable_frame).is_none()
            || !crate::state::can_write_back_outer_locals(ctx.trace_ctx, vable_frame)
            || !crate::state::can_publish_frame_stack(cf_addr, vable_frame)
        {
            latchdbg!("origin={origin} sf-vable-frame-mismatch");
            return false;
        }
        // Keep the per-frame red identity seeded by `frame_box`.  The
        // authoritative walk executed against `snapshot_for_tracing`, but the
        // adopter validates this identity and publishes that snapshot's locals
        // to the matching live frame before driving the blackhole. Replacing
        // the register with the snapshot address would collapse those two
        // identities and fail the live-root check.
        // `ctx` IS the root walk on this arm (`framestack.is_empty() &&
        // !inline_subwalk`), so its mirror describes the very frame whose
        // operand stack the adopter publishes.  A root walk never refreshes
        // the virtualizable snapshot array; `_copy_data_from_miframe` copies
        // the live MIFrame banks, which is this mirror.  `ABORT_TOO_LONG`
        // reconciles the mirror to the next instruction first (`walk`), so
        // the captured slots are the post-step stack.  A capability-gap
        // abort stops mid-opcode and needs the same source.
        let mirror_stack = capture_vstack_mirror_image(ctx, origin);
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[latch-accept] origin={origin} single-frame pc={} jitcode={:?} index={:?}",
                miframe.pc,
                miframe.jitcode.name(),
                miframe.jitcode.try_index(),
            );
        }
        FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
            *slot.borrow_mut() = Some(LatchedSingleFrameBlackhole {
                miframe,
                last_exc_value,
                raising_exception: false,
                mirror_stack,
                origin,
            });
        });
        true
    } else if ctx.fbw_mode.inline_subwalk {
        let Some(framestack) =
            build_multi_frame_miframe(ctx, resume_pc, InnermostMiframeBuild::TraceTooLong, origin)
        else {
            latchdbg!("origin={origin} mf-build-miframe");
            return false;
        };
        if !multi_frame_blackhole_preflight(ctx, &framestack, origin) {
            latchdbg!("origin={origin} mf-preflight");
            return false;
        }
        // `ctx` is the innermost callee, so its ordinary vstack mirror names
        // the wrong frame.  Reconstruct frame 0 from the paused caller image
        // captured at the outermost inline push, exactly where RPython keeps
        // that root MIFrame's register bank.  An incomplete image is not a
        // latch: `ABORT_TOO_LONG` with effects cannot decline into replay
        // (`trace_too_long_abort_safe`), so refuse here and keep recording
        // until a later step can hand off (`latch_abort_blackhole`).
        let Some(mirror_stack) = capture_root_parent_resume_stack(ctx) else {
            latchdbg!("origin={origin} mf-root-parent-stack");
            return false;
        };
        FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
            *slot.borrow_mut() = Some(LatchedMultiFrameBlackhole {
                framestack,
                last_exc_value,
                raising_exception: false,
                mirror_stack: Some(mirror_stack),
                origin,
            });
        });
        true
    } else {
        latchdbg!(
            "origin={origin} no-arm framestack_empty={} inline_subwalk={}",
            ctx.session.borrow().at_portal(),
            ctx.fbw_mode.inline_subwalk
        );
        false
    }
}

/// Prepare a quasi-immutable abort raised at `resume_pc`.
///
/// RPython's `_get_opimpl_method` advances `MIFrame.pc` before calling
/// `opimpl_jit_force_quasi_immutable`; `convert_and_run_from_pyjitpl` therefore
/// resumes after the qmut marker.  Pyre synthesizes that marker immediately
/// before the residual operation it protects, so `resume_pc` is the same
/// next-to-run position.  Inline aborts capture the whole framestack; root
/// aborts retain the operand stack for `flush_qmut_abort_state`.
fn prepare_force_quasi_immutable_abort<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    resume_pc: usize,
) {
    if ctx.fbw_mode.inline_subwalk {
        // This is a new unwind, so a retained image from a recovered sub-walk
        // must not replace its coordinate.
        reset_single_frame_blackhole();
        let _ = latch_abort_blackhole(ctx, resume_pc, "force-qmut");
    } else if ctx.vstack_valid && !ctx.trace_ctx.is_bridge_trace {
        fbw_qmut_abort_stack_latch(
            ctx.vstack_cur_pypc as usize,
            ctx.frame_state.borrow().vstack_boxes.clone(),
        );
    }
    // The session flag survives recovered sub-walks; always overwrite it at
    // the raise site so a later root-frame abort cannot inherit `true`.
    ctx.session.borrow_mut().abort_in_subwalk = ctx.fbw_mode.inline_subwalk;
}

/// Read-only counterpart of every adopter gate that can reject a latched
/// multi-frame image.  `ABORT_TOO_LONG` runs after the opcode's effects, so it
/// may publish the image only when the later handoff cannot fall back to entry
/// replay.  RPython needs no split preflight: its per-frame red virtualizable
/// is already the live MIFrame state copied by
/// `convert_and_run_from_pyjitpl`.
fn multi_frame_blackhole_preflight<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    framestack: &majit_metainterp::MIFrameStack,
    origin: &'static str,
) -> bool {
    if ctx.trace_ctx.virtualizable_info().is_none() || ctx.fbw_mode.snapshot_sym.is_null() {
        latchdbg!("origin={origin} pf-no-vinfo-or-sym");
        return false;
    }
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    let snapshot = sym.tracing_vable_frame_addr();
    let live_root = match ctx.trace_ctx.lookup_opref_concrete(sym.frame()) {
        Some(majit_ir::Value::Ref(value)) if value.0 != 0 => value.0,
        _ => sym.live_vable_frame_addr(),
    };
    let root = if live_root != 0 { live_root } else { snapshot };
    // Named by the first capability that refuses, and reported only then.
    // Emitted ahead of the test, the line announced every passing preflight
    // under the decline tag as well, so a census could not tell a rejected root
    // from an accepted one.
    let refused = if crate::state::concrete_nlocals(snapshot).is_none() {
        Some("nlocals")
    } else if crate::state::capture_frame_locals(root).is_none() {
        Some("locals")
    } else if !crate::state::can_write_back_outer_locals(ctx.trace_ctx, root) {
        Some("writeback")
    } else if !crate::state::can_publish_frame_stack(snapshot, root) {
        Some("publish")
    } else {
        None
    };
    if let Some(refused) = refused {
        latchdbg!(
            "origin={origin} pf-root-caps snapshot={snapshot:#x} root={root:#x} refused={refused}"
        );
        return false;
    }

    let mut seen = Vec::with_capacity(framestack.frames.len());
    for (index, frame) in framestack.frames.iter().enumerate() {
        let Ok(jitcode_index) = i32::try_from(frame.jitcode.index()) else {
            latchdbg!("origin={origin} pf-jitcode-index");
            return false;
        };
        let frame_reg = crate::state::portal_red_regs_at(jitcode_index).0;
        if frame_reg == u16::MAX {
            latchdbg!("origin={origin} pf-frame-reg-none");
            return false;
        }
        let Some(frame_ptr) = frame.ref_value_for_blackhole(frame_reg as usize) else {
            latchdbg!(
                "origin={origin} pf-frame-ptr-unset index={index}/{} jitcode={} frame_reg={frame_reg}",
                framestack.frames.len(),
                frame.jitcode.name()
            );
            return false;
        };
        let frame_ptr = frame_ptr as usize;
        let Some(stack_base) = crate::state::concrete_nlocals(frame_ptr) else {
            latchdbg!("origin={origin} pf-nlocals");
            return false;
        };
        let Some(stack_depth) = crate::state::concrete_stack_depth(frame_ptr) else {
            latchdbg!("origin={origin} pf-stack-depth");
            return false;
        };
        let Some(array_len) = crate::state::concrete_frame_array_len(frame_ptr) else {
            latchdbg!("origin={origin} pf-array-len");
            return false;
        };
        if stack_depth < stack_base
            || stack_depth > array_len
            || (index == 0 && frame_ptr != root)
            || (index > 0 && frame_ptr == root)
            || seen.contains(&frame_ptr)
        {
            latchdbg!("origin={origin} pf-shape");
            return false;
        }
        seen.push(frame_ptr);
    }
    true
}

fn build_single_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode: std::sync::Arc<majit_metainterp::jitcode::JitCode>,
    resume_pc: usize,
    lastop_result: Option<(char, usize, i64)>,
) -> Option<majit_metainterp::MIFrame> {
    // An MIFrame is only representable at a `-live-` anchored startpoint: the
    // blackhole reads its registers straight out of the copied banks, so a
    // coordinate whose live set cannot be decoded must decline here rather
    // than yield a frame with every register unset.  All callers pass a
    // post-call `next_pc` today, which always resolves; the precondition is
    // what keeps that true as the pc domain widens.
    let live = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
        i32::try_from(jitcode.index()).ok()?,
        i32::try_from(resume_pc).ok()?,
    )?;
    let mut miframe = majit_metainterp::MIFrame::new(jitcode.clone(), resume_pc);

    // pyjitpl.py make_result_of_lastop: the residual returned before its
    // ordinary dispatcher writeback, while `resume_pc` already points past
    // the call.  Stamp the just-produced value first: its destination is
    // normally live at `resume_pc`, but the concrete shadow still contains
    // Null/old data and must not make the all-live-color pass decline.
    if let Some((bank, color, value)) = lastop_result {
        match bank {
            'i' => *miframe.int_regs.get_mut(color)? = Some(OpRef::const_int(value)),
            'r' => {
                *miframe.ref_regs.get_mut(color)? =
                    Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)))
            }
            'f' => {
                *miframe.float_regs.get_mut(color)? =
                    Some(OpRef::const_float(f64::from_bits(value as u64)))
            }
            'v' => {}
            _ => return None,
        }
    }

    // An `OpRef::None` register is an ABSENT BOX, not an unresolved one: the
    // walk never defined this color.  `blackhole.py _copy_data_from_miframe
    // _copy_data_from_miframe` copies a register only `if box`, so upstream
    // leaves the blackhole's slot exactly as unset as the MIFrame's, and the
    // drive cannot read it — every path that reaches this coordinate with the
    // color live also defines it, or the jitcode would read an undefined
    // register.  It arrives undefined here because a `-live-` set is the union
    // over the paths INTO its coordinate, and the walk took one of them.
    // Declining refused the whole image over a register nothing will read; the
    // shape it kept on the replay path is a `CALL_FUNCTION_EX` whose escape
    // fell back to a legacy replay of an already-executed frame.
    // A color that HAS a box but no recoverable concrete still declines: there
    // the drive can read it and the walk lost the value.
    for &color in &live.int {
        let color = color as usize;
        if miframe.int_regs.get(color).copied().flatten().is_some() {
            continue;
        }
        let ConcreteValue::Int(value) = ctx.concrete_registers_i.get(color).copied()? else {
            if ctx.registers_i.get(color)?.is_none() {
                continue;
            }
            return None;
        };
        *miframe.int_regs.get_mut(color)? = Some(OpRef::const_int(value));
    }
    for &color in &live.ref_ {
        let color = color as usize;
        if miframe.ref_regs.get(color).copied().flatten().is_some() {
            continue;
        }
        let value = match ctx
            .frame_state
            .borrow()
            .concrete_registers_r
            .get(color)
            .copied()?
        {
            ConcreteValue::Ref(value) => value as i64,
            // ConcreteValue::Null is the walker's "unknown" sentinel, not a
            // proven Python null.  The register shadow never held this color's
            // concrete — recover it from the recorded box's producer (a getfield
            // chain rooted at a seeded input arg an inlined sub-walk read but
            // never concretized in this frame's shadow); decline only when even
            // that cannot resolve it.
            _ => {
                let opref = ctx.registers_r.get(color)?;
                if opref.is_none() {
                    continue;
                }
                match ctx.trace_ctx.recover_ref_value(opref, 8) {
                    Some(majit_ir::Value::Ref(gc)) => gc.0 as i64,
                    _ => return None,
                }
            }
        };
        *miframe.ref_regs.get_mut(color)? = Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)));
    }
    for &color in &live.float {
        let color = color as usize;
        if miframe.float_regs.get(color).copied().flatten().is_some() {
            continue;
        }
        let opref = ctx.registers_f.get(color)?;
        if opref.is_none() {
            continue;
        }
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            return None;
        };
        *miframe.float_regs.get_mut(color)? = Some(OpRef::const_float(value));
    }

    // Seed every REMAINING color the walk has a concrete value for, not just the
    // ones live at `resume_pc`.
    //
    // `_copy_data_from_miframe` (blackhole.py) copies
    // `range(num_regs_i/r/f())` — the WHOLE bank, filtering only on "the MIFrame
    // has a box here", never on liveness.  Seeding a liveness-selected subset is
    // the deviation, and it is unsound the moment the drive leaves the straight
    // line: the blackhole runs on to a `jit_merge_point`, whose own live set is
    // generally LARGER (a loop-carried value defined in the prologue and not
    // rewritten in the body is dead at a mid-body pc but live at the header).
    // Those colors then read back NULL, and a NULL written into a live
    // operand-stack slot faults the interpreter.
    //
    // Measured on the `getframe_root_loop_force_*` fixtures: live at the build
    // pc was ref [0, 1, 6] while the merge wanted [0, 1, 2, 3, 4], and 2/3/4 all
    // had concrete values in the walk's own bank the whole time.
    //
    // Seeding cannot introduce a stale value: a color the drive rewrites is
    // overwritten before any read, and a color it does not rewrite still holds
    // what the walk observed at the force point, which is what a merge reached
    // without an intervening definition expects.  Colors with no concrete value
    // stay unset, exactly as an absent upstream box does.
    for (color, slot) in miframe.int_regs.iter_mut().enumerate() {
        if slot.is_none()
            && let Some(ConcreteValue::Int(value)) = ctx.concrete_registers_i.get(color).copied()
        {
            *slot = Some(OpRef::const_int(value));
        }
    }
    for (color, slot) in miframe.ref_regs.iter_mut().enumerate() {
        // `ConcreteValue::Null` is the walker's "unknown" sentinel, not a proven
        // Python null, so it seeds nothing.
        if slot.is_none()
            && let Some(ConcreteValue::Ref(value)) = ctx
                .frame_state
                .borrow()
                .concrete_registers_r
                .get(color)
                .copied()
        {
            *slot = Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)));
        }
    }
    // `num_regs_f()` is the third bank `_copy_data_from_miframe` walks, so the
    // rationale above covers floats too.  There is no float concrete shadow —
    // resolve the recorded box, and leave the color unset when it has none.
    for color in 0..miframe.float_regs.len() {
        if miframe.float_regs[color].is_some() {
            continue;
        }
        let Some(opref) = ctx.registers_f.get(color) else {
            continue;
        };
        if let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) {
            miframe.float_regs[color] = Some(OpRef::const_float(value));
        }
    }

    Some(miframe)
}

/// Build the `ABORT_TOO_LONG` image at an arbitrary post-step JitCode pc.
///
/// Unlike the residual-call handoff above, this coordinate need not be a
/// `-live-` marker: RPython copies the complete MIFrame banks after every
/// `run_one_step`. Requiring marker liveness here would make an effectful
/// straight-line tail unbounded again whenever the limit lands between
/// markers.
pub(super) fn build_trace_too_long_single_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode: std::sync::Arc<majit_metainterp::jitcode::JitCode>,
    resume_pc: usize,
) -> Option<majit_metainterp::MIFrame> {
    // The blackhole `setposition`s straight onto this coordinate and dispatches
    // there, so it has to start an instruction. Dropping the `-live-`
    // requirement above widened the pc domain to every post-step position, and
    // nothing else narrows it back: a coordinate inside an instruction's
    // operands reads the next operand byte as an opcode. That is where
    // `dispatch_step: unwired opcode=0x5 pos=18 jitcode="__new__"` came from —
    // byte 5 is the result-register operand of a `residual_call_ir_r/iIRd>r`
    // spanning pc 8..19, and no build emits byte 5 as an opcode at all.
    //
    // `build_multi_frame_miframe` already declines this class for its own
    // innermost frame. Declining is the supported outcome: every caller falls
    // back to the rollback/replay path.
    if !instruction_starts_at(&jitcode, resume_pc) {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-fill-decline] resume_pc={resume_pc} is not an instruction \
                 start in jitcode {:?} (len {})",
                jitcode.name,
                jitcode.code.len(),
            );
        }
        return None;
    }
    let mut miframe = majit_metainterp::MIFrame::new(jitcode, resume_pc);
    fill_trace_too_long_register_banks(ctx, &mut miframe).then_some(miframe)
}

/// Whether `pc` is an instruction boundary of `jitcode`.
///
/// `startpoints` (`jitcode.py:40`) is the assembler's own record of where
/// instructions begin, and the authority the blackhole already uses for this
/// question — `blackhole.py dispatch_loop` asserts its position against it,
/// and `find_catch_before_resume_live` scans it for op boundaries. Assembled
/// jitcodes always populate it; a hand-built one leaves it `None`
/// (`jitcode.py setup(..., startpoints=None)`), so fall back to walking the
/// fallthrough layout. `decoded_ops` starts at 0 and stops at the first byte it
/// cannot decode, so a pc it never yields is either inside an instruction or
/// past a decode failure — neither is a coordinate the blackhole may dispatch
/// from. The end of the code is a boundary either way: a frame that resumes
/// there returns without dispatching.
fn instruction_starts_at(jitcode: &majit_metainterp::jitcode::JitCode, pc: usize) -> bool {
    if pc == jitcode.code.len() {
        return true;
    }
    match jitcode.startpoints.as_ref() {
        Some(startpoints) => startpoints.contains(&pc),
        None => crate::jitcode_runtime::decoded_ops(&jitcode.code).any(|op| op.pc == pc),
    }
}

/// Fill every currently-known register color, matching
/// `blackhole.py:_copy_data_from_miframe`.
///
/// The vable-escape latch above only needs the resume marker's live set: it
/// resumes immediately after one forcing residual whose result is supplied
/// separately. `ABORT_TOO_LONG` is different — the blackhole can follow
/// arbitrary control flow from a post-step pc before reaching a terminal, so
/// colors outside that marker-local live set may be read later. RPython copies
/// all three complete MIFrame banks; do the same for this abort instead of
/// relying on the narrower resume-liveness cache.
///
/// `_copy_data_from_miframe` (`blackhole.py`) guards every bank entry
/// with `if box is not None` and has no failing path — but there, a `None`
/// register is a **dead** one the jitcode's liveness already cleared, and every
/// register that survives holds a box with a value.  "Live, but the walk knows
/// no value" has no upstream counterpart, and skipping such a color would leave
/// its pre-sized blackhole register at zero for a resume that can follow
/// arbitrary control flow to an instruction that reads it.  So `OpRef::NONE`
/// (dead, upstream's `box is None`) is skipped, and a live color with no
/// concrete refuses the whole frame.
///
/// One live Ref color carries a value neither concrete lookup can express: the
/// dead-`box_bool` marker holds a comparison's raw truth Int in a Ref register.
/// It is unfillable by construction rather than unknown, so the Ref bank
/// reconstructs the singleton instead of refusing — see there.
fn fill_trace_too_long_register_banks<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    miframe: &mut majit_metainterp::MIFrame,
) -> bool {
    // Name the declining bank and color under `PYRE_FBW_DEBUG_ABORT`, the way
    // `build_multi_frame_miframe` names its own: "innermost declined" alone
    // does not say which register had no concrete.
    macro_rules! s2dbg {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-fill-decline] {}", format!($($a)*));
            }
        };
    }
    for color in 0..miframe.int_regs.len() {
        if let Some(value) = ctx
            .concrete_registers_i
            .get(color)
            .copied()
            .and_then(|value| match value {
                ConcreteValue::Int(value) => Some(value),
                ConcreteValue::Bool(value) => Some(i64::from(value)),
                _ => None,
            })
        {
            miframe.int_regs[color] = Some(OpRef::const_int(value));
            continue;
        }
        if miframe.int_regs[color].is_none() {
            let Some(opref) = ctx.registers_i.get(color) else {
                continue;
            };
            if opref == OpRef::NONE {
                continue;
            }
            let Some(majit_ir::Value::Int(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
                s2dbg!("int color={color} opref={opref:?} has no concrete");
                return false;
            };
            miframe.int_regs[color] = Some(OpRef::const_int(value));
        }
    }

    for color in 0..miframe.ref_regs.len() {
        let opref = ctx.registers_r.get(color);
        let forwarded = opref
            .filter(|&value| value != OpRef::NONE)
            .and_then(|value| {
                match ctx
                    .trace_ctx
                    .lookup_opref_concrete(value)
                    .or_else(|| ctx.trace_ctx.recover_ref_value(value, 8))
                {
                    Some(majit_ir::Value::Ref(value)) => Some(value.0 as i64),
                    _ => None,
                }
            });
        let from_shadow = ctx
            .frame_state
            .borrow()
            .concrete_registers_r
            .get(color)
            .copied()
            .and_then(|value| match value {
                ConcreteValue::Ref(value) => Some(value as i64),
                _ => None,
            });
        // Every bool a comparison leaves in a Ref register is now a real Ref —
        // either the `space.newbool` singleton behind its truth guard or the
        // residual box — so `forwarded` reads it directly.  A raw-truth Int was
        // once written into this bank as a dead-box marker, which neither arm
        // above can represent; the color was skipped and the blackhole read
        // NULL, and `is_true(NULL)` is false, sending a `while` straight to its
        // exit arm and dropping every remaining iteration.
        if let Some(value) = forwarded.or(from_shadow) {
            miframe.ref_regs[color] = Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)));
        } else if opref.is_some_and(|value| value != OpRef::NONE) {
            s2dbg!("ref color={color} opref={opref:?} has no concrete");
            return false;
        }
    }

    for color in 0..miframe.float_regs.len() {
        if miframe.float_regs[color].is_some() {
            continue;
        }
        let Some(opref) = ctx.registers_f.get(color) else {
            continue;
        };
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            s2dbg!("float color={color} opref={opref:?} has no concrete");
            return false;
        };
        miframe.float_regs[color] = Some(OpRef::const_float(value));
    }
    true
}

/// Result-register of the CALL this paused parent sits on, if the op returns
/// a value.  `_setup_return_value_*` writes that color when the callee
/// finishes, so the captured image must leave it unset.
fn parent_call_result_color(
    pjc: &crate::pyjitcode::PyJitCode,
    call_jit_pc: usize,
) -> Option<(char, usize)> {
    let call = decode_op_at(&pjc.jitcode.code, call_jit_pc)?;
    let result_bank = call.argcodes.chars().last()?;
    if result_bank == 'v' {
        return None;
    }
    let color = pjc.jitcode.code.get(call.next_pc.checked_sub(1)?)?;
    Some((result_bank, usize::from(*color)))
}

fn parent_resume_pc(
    parent: &InlineParentFrame,
    pjc: &crate::pyjitcode::PyJitCode,
) -> Option<usize> {
    if let Some(concrete) = parent.blackhole.as_ref() {
        return Some(concrete.resume_pc);
    }
    if parent.call_jitcode_pc.is_none() {
        return parent.resume_marker_jit_pc;
    }
    let call_jit_pc = parent.call_jitcode_pc?;
    decode_op_at(&pjc.jitcode.code, call_jit_pc).map(|call| call.next_pc)
}

/// One paused caller as an `MIFrame`. Prefer the CALL-site blackhole capture;
/// if that capture missed, reconstruct from the live banks `attach_live_caller`
/// stored. Continuation tails (`descr_call`, operator) own no banks: they
/// still become their own `MIFrame` at `resume_marker_jit_pc`, which is what
/// `_setup_return_value_r` (`blackhole.py`) needs so the dunder return lands
/// in the tail rather than the caller's result slot.
fn build_parent_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    parent: &InlineParentFrame,
    origin: &'static str,
    index: usize,
) -> Option<majit_metainterp::MIFrame> {
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32) else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: no pyjitcode for parent \
                 jitcode_index={}",
                parent.jitcode_index
            );
        }
        return None;
    };
    if let Some(concrete) = parent.blackhole.as_ref() {
        return copy_captured_parent_blackhole(ctx, &pjc, concrete, origin, index);
    }
    if parent.call_jitcode_pc.is_none()
        && parent.registers_i.is_none()
        && parent.registers_r.is_none()
        && parent.registers_f.is_none()
    {
        return build_codeless_continuation_parent(ctx, parent, &pjc, origin, index);
    }
    build_parent_miframe_from_live_banks(ctx, parent, &pjc, origin, index)
}

/// `descr_call` / operator tails: jitcode and resume pc only, no live banks.
/// `ctor_continuation` keeps the instance in `INSTANCE_REG`; operator tails
/// have nothing live (`get_list_of_active_boxes(in_a_call=True)`).
fn build_codeless_continuation_parent<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    parent: &InlineParentFrame,
    pjc: &crate::pyjitcode::PyJitCode,
    origin: &'static str,
    index: usize,
) -> Option<majit_metainterp::MIFrame> {
    let Some(resume_pc) = parent.resume_marker_jit_pc else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: continuation tail \
                 has no resume_marker_jit_pc"
            );
        }
        return None;
    };
    if !instruction_starts_at(&pjc.jitcode, resume_pc) {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: continuation \
                 resume_pc={resume_pc} is not an instruction start"
            );
        }
        return None;
    }
    let mut miframe = majit_metainterp::MIFrame::new(pjc.jitcode.clone(), resume_pc);
    if let Some(&instance) = parent.boxes.first() {
        let color = crate::ctor_continuation::INSTANCE_REG as usize;
        let Some(majit_ir::Value::Ref(gc)) = ctx.trace_ctx.concrete_of_opref(instance) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: continuation \
                     instance box has no concrete Ref"
                );
            }
            return None;
        };
        if color >= miframe.ref_regs.len() {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: continuation \
                     INSTANCE_REG {color} out of range (len {})",
                    miframe.ref_regs.len()
                );
            }
            return None;
        }
        miframe.ref_regs[color] = Some(OpRef::const_ptr(gc));
    }
    Some(miframe)
}

fn copy_captured_parent_blackhole<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pjc: &crate::pyjitcode::PyJitCode,
    concrete: &InlineParentBlackhole,
    origin: &'static str,
    index: usize,
) -> Option<majit_metainterp::MIFrame> {
    let mut miframe = majit_metainterp::MIFrame::new(pjc.jitcode.clone(), concrete.resume_pc);
    for &(color, value) in &concrete.int_values {
        let bank_len = miframe.int_regs.len();
        let Some(slot) = miframe.int_regs.get_mut(color) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: int color {color} \
                     out of range (len {bank_len})"
                );
            }
            return None;
        };
        *slot = Some(OpRef::const_int(value));
    }
    for &(color, value) in &concrete.ref_values {
        let bank_len = miframe.ref_regs.len();
        let Some(slot) = miframe.ref_regs.get_mut(color) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: ref color {color} \
                     out of range (len {bank_len})"
                );
            }
            return None;
        };
        *slot = Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)));
    }
    for &(color, opref) in &concrete.float_values {
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: float opref {opref:?} \
                     not a stamped Float"
                );
            }
            return None;
        };
        let bank_len = miframe.float_regs.len();
        let Some(slot) = miframe.float_regs.get_mut(color) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: float color {color} \
                     out of range (len {bank_len})"
                );
            }
            return None;
        };
        *slot = Some(OpRef::const_float(value));
    }
    Some(miframe)
}

fn build_parent_miframe_from_live_banks<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    parent: &InlineParentFrame,
    pjc: &crate::pyjitcode::PyJitCode,
    origin: &'static str,
    index: usize,
) -> Option<majit_metainterp::MIFrame> {
    let Some(resume_pc) = parent_resume_pc(parent, pjc) else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: parent.blackhole None \
                 and no CALL resume pc (continuation tail / capture missing)"
            );
        }
        return None;
    };
    if !instruction_starts_at(&pjc.jitcode, resume_pc) {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: resume_pc={resume_pc} \
                 is not an instruction start"
            );
        }
        return None;
    }
    let (Some(registers_i), Some(registers_r), Some(registers_f), Some(frame_state)) = (
        parent.registers_i.as_ref(),
        parent.registers_r.as_ref(),
        parent.registers_f.as_ref(),
        parent.frame_state.as_ref(),
    ) else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[s2-build-decline] origin={origin} frame {index}: parent.blackhole None \
                 and no live banks (descr_call / operator continuation)"
            );
        }
        return None;
    };
    let result = parent
        .call_jitcode_pc
        .and_then(|pc| parent_call_result_color(pjc, pc));
    let mut miframe = majit_metainterp::MIFrame::new(pjc.jitcode.clone(), resume_pc);
    for color in 0..miframe.int_regs.len() {
        if result == Some(('i', color)) {
            continue;
        }
        let Some(opref) = registers_i.get(color) else {
            continue;
        };
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Int(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: live int color \
                     {color} opref={opref:?} has no concrete"
                );
            }
            return None;
        };
        miframe.int_regs[color] = Some(OpRef::const_int(value));
    }
    for color in 0..miframe.ref_regs.len() {
        if result == Some(('r', color)) {
            continue;
        }
        let Some(opref) = registers_r.get(color) else {
            continue;
        };
        if opref.is_none() {
            continue;
        }
        let from_shadow = {
            let borrowed = frame_state.borrow();
            borrowed
                .concrete_registers_r
                .get(color)
                .copied()
                .and_then(|value| match value {
                    ConcreteValue::Ref(value) => Some(value as i64),
                    _ => None,
                })
        };
        let forwarded = match ctx
            .trace_ctx
            .lookup_opref_concrete(opref)
            .or_else(|| ctx.trace_ctx.recover_ref_value(opref, 8))
        {
            Some(majit_ir::Value::Ref(gc)) => Some(gc.0 as i64),
            _ => None,
        };
        if let Some(value) = forwarded.or(from_shadow) {
            miframe.ref_regs[color] = Some(OpRef::const_ptr(majit_ir::GcRef(value as usize)));
        } else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: live ref color \
                     {color} opref={opref:?} has no concrete"
                );
            }
            return None;
        }
    }
    for color in 0..miframe.float_regs.len() {
        if result == Some(('f', color)) {
            continue;
        }
        let Some(opref) = registers_f.get(color) else {
            continue;
        };
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[s2-build-decline] origin={origin} frame {index}: live float color \
                     {color} opref={opref:?} has no concrete"
                );
            }
            return None;
        };
        miframe.float_regs[color] = Some(OpRef::const_float(value));
    }
    Some(miframe)
}

#[derive(Clone, Copy)]
enum InnermostMiframeBuild {
    LiveMarker(Option<(char, usize, i64)>),
    TraceTooLong,
}

fn build_multi_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    resume_pc: usize,
    innermost_build: InnermostMiframeBuild,
    origin: &'static str,
) -> Option<majit_metainterp::MIFrameStack> {
    macro_rules! s2dbg {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-build-decline] {}", format!($($a)*));
            }
        };
    }
    // The build's one success report.  It has to carry its own tag: a census
    // groups these lines by their `[…-decline]` prefix, so a completed build
    // announced under that prefix is counted as a refusal.
    macro_rules! s2built {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-build-ok] {}", format!($($a)*));
            }
        };
    }
    let session = ctx.session.borrow();
    if session.at_portal() {
        s2dbg!(
            "origin={origin} framestack empty depth={} transparent_helper_subwalk={}",
            session.framestack.len(),
            ctx.fbw_mode.transparent_helper_subwalk
        );
        return None;
    }
    let mut frames = majit_metainterp::MIFrameStack::empty();

    for (index, parent) in session
        .framestack
        .iter()
        .flat_map(|inline| inline.parents.iter())
        .enumerate()
    {
        let Some(miframe) = build_parent_miframe(ctx, parent, origin, index) else {
            return None;
        };
        frames.push(miframe);
    }

    // The innermost frame is the callee `ctx` is actively sub-walking. In a
    // sub-walk its identity lives in `inline_callee_consts` — the copied
    // `snapshot_sym` still points at the outer portal, so resolving off it
    // pins the OUTER jitcode while `resume_pc` and the concrete banks are the
    // callee's, landing the resume mid-op in the wrong code. Resolve the
    // callee's own jitcode so all three share its coordinate space; fall back
    // to `snapshot_sym` for a top-level (non-sub-walk) abort.
    let innermost_jitcode = if let Some(consts) = ctx.inline_callee_consts {
        let Some(jc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
            s2dbg!(
                "origin={origin} innermost: no pyjitcode for callee jitcode_index={}",
                consts.jitcode_index
            );
            return None;
        };
        jc.jitcode.clone()
    } else {
        unsafe {
            let sym = &*ctx.fbw_mode.snapshot_sym;
            if sym.jitcode().is_null() {
                s2dbg!("origin={origin} innermost snapshot_sym jitcode null");
                return None;
            }
            (&(*sym.jitcode()).payload).jitcode.clone()
        }
    };
    let innermost = match innermost_build {
        InnermostMiframeBuild::LiveMarker(lastop_result) => {
            build_single_frame_miframe(ctx, innermost_jitcode, resume_pc, lastop_result)
        }
        InnermostMiframeBuild::TraceTooLong => {
            build_trace_too_long_single_frame_miframe(ctx, innermost_jitcode, resume_pc)
        }
    };
    let Some(innermost) = innermost else {
        s2dbg!("origin={origin} innermost build_single_frame_miframe declined");
        return None;
    };
    frames.push(innermost);
    s2built!(
        "origin={origin} BUILT multi-frame depth={}",
        frames.frames.len()
    );
    Some(frames)
}

/// TLS cell pointer for the store-journal root area: the undo stays armed
/// from force time until the abort epilogue consumes it, and its slots can be
/// the ONLY reference to pre-walk locals the flush displaced — the area
/// walker forwards them across any minor collection on that whole window
/// (the resume-ref-roots stack cannot cover it: the residual's post-call
/// `pop_resume_ref_roots_to` truncates past a force-time push).
pub(crate) fn escape_flush_undo_cell_ptr() -> *const std::cell::RefCell<Option<EscapeFlushUndo>> {
    ESCAPE_FLUSH_UNDO.with(|cell| cell as *const _)
}

thread_local! {
    /// The concrete `PyFrame` the innermost inline sub-walk is executing, or
    /// `None` outside one.  Set by [`InlineConcreteFrameGuard`], and held as
    /// a translated-livevar root rather than a raw pointer, the way
    /// [`ResidualFrameChainGuard`] holds the `topframeref` it displaces.
    static INLINE_CONCRETE_FRAME: std::cell::RefCell<
        Option<majit_gc::shadow_stack::OwnerRootGuard>,
    > = const { std::cell::RefCell::new(None) };

    /// The concrete frame currently published on the interpreter chain by a
    /// live [`ResidualFrameChainGuard`], or null.  Distinct from
    /// [`INLINE_CONCRETE_FRAME`], which stays set across the whole sub-walk.
    static PUBLISHED_INLINE_FRAME: std::cell::Cell<*mut pyre_interpreter::PyFrame> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
    /// The inline level's own locals shadow, published alongside
    /// [`PUBLISHED_INLINE_FRAME`] for the duration of one residual call.
    ///
    /// A force inside the residual runs from `force_pyframe`, which reaches
    /// only the `TraceCtx`; `virtualizable.py write_boxes` writes the array of
    /// the virtualizable it forces, so the level's own slot source has to be
    /// reachable there. Publish the shared owner, not a raw pointer into its
    /// RefCell: writeback may collect and forward the same shadow's fields.
    static PUBLISHED_INLINE_SHADOW: std::cell::RefCell<Option<(super::WalkFrameState, u16)>> =
        const { std::cell::RefCell::new(None) };

    /// The frame a live [`LiveLastInstrGuard`] published the executing pc onto,
    /// with the resume coordinate it displaced.
    static PUBLISHED_LAST_INSTR: std::cell::Cell<Option<(usize, isize)>> =
        const { std::cell::Cell::new(None) };
}

/// Preserve the instruction that raised when a residual-call traceback hook
/// has finalized the currently published frame.  On PyPy the dispatch loop's
/// `last_instr` store precedes `handle_operation_error`, so there is no older
/// resume coordinate to restore after that exception leaves the frame.
pub(super) fn finalize_published_last_instr(frame: usize, last_instr: isize) {
    PUBLISHED_LAST_INSTR.with(|slot| {
        if matches!(slot.get(), Some((published, _)) if published == frame) {
            slot.set(Some((frame, last_instr)));
        }
    });
    ESCAPE_FLUSH_UNDO.with(|slot| {
        if let Some(undo) = slot.borrow_mut().as_mut()
            && undo.frame == frame
        {
            undo.last_instr = last_instr;
        }
    });
}

/// `_opimpl_setfield_vable`'s heap half for the recording walk.  While a
/// residual runs concretely the live frame must name the EXECUTING opcode, so
/// that a frame reader inside the callee reports the line the opcode is on —
/// what `eval.rs` `frame.last_instr = pc` gives an interpreted opcode.  The
/// walk dispatches opcodes itself instead of through `execute_opcode_step`, so
/// nothing else advances the field and the frame still carries the pc of the
/// last resume point; `_warnings::setup_context` then keys its registry on
/// that line and re-issues a warning the interpreted run already deduplicated.
///
/// Restored on the way out, because `last_instr` doubles as the resume
/// coordinate: `flush_walk_end_state_to_frame_inner` writes `resume_py_pc - 1`
/// and the escape guard's resume pc is this same pc, so the two meanings
/// differ by exactly one and may only coexist for the residual's duration.
/// Leaving the executing value behind makes an abort replay one opcode late.
struct LiveLastInstrGuard {
    frame: *mut pyre_interpreter::PyFrame,
    /// Re-read on drop. `frame` is the address at enter and is what the
    /// publication cell compares; the object may have moved.
    frame_root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    saved: isize,
    prev: Option<(usize, isize)>,
}

impl LiveLastInstrGuard {
    /// Publishes onto the frame `py_pc` indexes.  Inside an inline sub-walk
    /// that is the callee's concrete frame, NOT the walk's virtualizable: a
    /// sub-walk's pc is in the callee's code (the same reason `escape_stack`
    /// declines to latch a sub-walk's mirror), and writing it onto the outer
    /// frame strands that frame's replay on a stack depth it never had.
    fn enter(live_frame: usize, py_pc: u32, inline_py_pc: Option<u32>) -> Option<Self> {
        let inline = current_inline_concrete_frame() as *mut pyre_interpreter::PyFrame;
        // The frame and the pc have to name the same code.  Retargeting the
        // frame to the callee while keeping the outer walk's `vstack_cur_pypc`
        // publishes a pc from the CALLER's code onto the callee's frame — and
        // for a sub-walk that mirror is never advanced (`vstack_valid` is
        // false), so the published value is a constant 0 and a frame reader
        // inside the callee reports the function's first line.
        let (frame, py_pc) = match (inline.is_null(), inline_py_pc) {
            (false, Some(callee_pc)) => (inline, callee_pc),
            (false, None) => (inline, py_pc),
            (true, _) => (live_frame as *mut pyre_interpreter::PyFrame, py_pc),
        };
        Self::enter_frame(frame, py_pc)
    }

    fn enter_frame(frame: *mut pyre_interpreter::PyFrame, py_pc: u32) -> Option<Self> {
        if frame.is_null() {
            return None;
        }
        let saved = unsafe { (*frame).last_instr };
        unsafe { (*frame).last_instr = py_pc as isize };
        let frame_root = owner_root_if_gc(frame as usize);
        // Save/restore rather than set/clear: a residual can run user code that
        // records a nested walk whose own residual enters a second guard, and
        // clearing on the inner drop would leave the still-live outer
        // publication invisible — `capture_escape_flush_undo` would then snapshot
        // the executing pc as if it were the outer frame's resume coordinate.
        // Same discipline as [`InlineConcreteFrameGuard`] and
        // [`ResidualFrameChainGuard`].
        let prev = PUBLISHED_LAST_INSTR.with(|slot| slot.replace(Some((frame as usize, saved))));
        Some(Self {
            frame,
            frame_root,
            saved,
            prev,
        })
    }
}

impl Drop for LiveLastInstrGuard {
    fn drop(&mut self) {
        let restore = PUBLISHED_LAST_INSTR.with(|slot| {
            let current = slot.replace(self.prev);
            current
                .filter(|(frame, _)| *frame == self.frame as usize)
                .map_or(self.saved, |(_, saved)| saved)
        });
        // A flush that committed onto this frame wrote the resume coordinate
        // itself and is authoritative.  Its undo capture holds the value this
        // guard displaced (see [`capture_escape_flush_undo`]), so a later
        // commit withdrawal still restores the resume pc rather than the
        // executing one.
        // The test is "did an escape happen on THIS frame", not "did a flush
        // write a resume pc".  Both legs arm the capture and both leave the
        // frame heap-authoritative at the published executing coordinate, so
        // narrowing this to the committing leg restores the PRE-call coordinate
        // on the locals-only escapes — the withdrawn-too-early defect
        // `getframe_caller_resume_coord_two_call_sites` measures and prints.
        let flushed = ESCAPE_FLUSH_UNDO.with(|slot| {
            slot.borrow().as_ref().map(|undo| undo.frame) == Some(self.frame as usize)
        });
        if !flushed {
            let frame = self
                .frame_root
                .as_ref()
                .map(|root| root.get().0 as *mut pyre_interpreter::PyFrame)
                .unwrap_or(self.frame);
            unsafe { (*frame).last_instr = restore };
        }
    }
}

/// Names the frame an inline sub-walk executes concretely for the duration of
/// that sub-walk.  Publishing it on the interpreter chain is left to
/// [`ResidualFrameChainGuard`], which brackets only the residual calls.
pub(crate) struct InlineConcreteFrameGuard(Option<majit_gc::shadow_stack::OwnerRootGuard>);

impl InlineConcreteFrameGuard {
    pub(crate) fn enter(frame: *mut pyre_interpreter::PyFrame) -> Self {
        // Set unconditionally, the empty case included: a nested sub-walk
        // whose seed block bailed has no frame of its own, and inheriting the
        // enclosing callee's frame would publish it for the inner callee's
        // residuals — resolving one level too shallow, the error this guard
        // exists to stop.  An empty slot makes `ResidualFrameChainGuard::enter`
        // publish nothing.
        //
        // The displaced root stays in this guard, so the enclosing level's
        // frame keeps its own root for the whole nested sub-walk.
        let entered = (!frame.is_null())
            .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(frame as usize)));
        Self(INLINE_CONCRETE_FRAME.with(|slot| slot.replace(entered)))
    }
}

impl Drop for InlineConcreteFrameGuard {
    fn drop(&mut self) {
        INLINE_CONCRETE_FRAME.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// `executioncontext.py enter` / `leave` around one concretely executed
/// residual call of an inline sub-walk.
///
/// Inlining a call elides the callee's real call sequence, so nothing
/// publishes the callee frame.  PyPy can leave the chain alone — its
/// metainterp never runs the callee for real while tracing — but the walker
/// does, and a residual that reads the chain (`sys._getframe`, a traceback)
/// would otherwise observe the caller as the running frame and resolve one
/// level too shallow.  Scoped to the call itself so the walk's own frame
/// bookkeeping outside it is untouched.
///
/// The displaced caller is parked in `frame.f_backref` and nowhere else:
/// `executioncontext.py enter` writes it there and `leave` reads it back off
/// the frame it is leaving, so the guard needs no saved copy of its own.  That
/// is also what keeps the value rooted — `f_backref` is a traced `Type::Ref`
/// field, so a minor collection inside the residual forwards a nursery
/// `JitVirtualRef` in place, where a raw copy in this struct would go stale.
/// Parking it on the shared shadow stack instead, which this guard used to do,
/// made its restore a LIFO slot the residual could not publish across: the
/// scope is one residual call, so anything the callee pushed sat above the
/// guard's own index and was truncated away when the guard exited.
struct ResidualFrameChainGuard {
    ec: *mut pyre_interpreter::PyExecutionContext,
    /// A root, not a raw copy.  The callee frame an inline sub-walk executes
    /// is a nursery allocation (`emit_new_pyframe_inline_with_params` ->
    /// `NewWithVtable`), so a minor collection inside the residual relocates
    /// it and `Drop`'s `f_backref` / `escaped()` reads would follow a stale
    /// address.  The collector forwards `OWNER_ROOTS` in place
    /// (`shadow_stack::walk_roots`), which is the same reason
    /// [`current_inline_concrete_frame`] reads its frame back out of a root
    /// rather than out of a copy.  Rooting is also why this can be an owner
    /// root and not a shadow-stack slot: the shadow stack is LIFO and the
    /// residual publishes across this scope.
    frame_root: majit_gc::shadow_stack::OwnerRootGuard,
    previous_published: *mut pyre_interpreter::PyFrame,
    previous_shadow: Option<(super::WalkFrameState, u16)>,
    /// Whether this guard performed the chain write, so `Drop` restores only
    /// what it changed.  False when the chain already named `frame`.
    entered: bool,
}

impl ResidualFrameChainGuard {
    fn enter(frame: usize, shadow: Option<(super::WalkFrameState, u16)>) -> Option<Self> {
        let frame = frame as *mut pyre_interpreter::PyFrame;
        if frame.is_null() {
            return None;
        }
        let ec = pyre_interpreter::call::getexecutioncontext()
            as *mut pyre_interpreter::PyExecutionContext;
        if ec.is_null() {
            return None;
        }
        let saved_topframeref = unsafe { (*ec).topframeref };
        // Re-entering the same frame would make it its own caller.  `topframeref`
        // holds a `jit.virtual_ref`, so a vref that NAMES this frame is the same
        // re-entry as the bare pointer; resolve the referent without forcing,
        // because forcing clears `TOKEN_TRACING_RESCALL` and
        // `tracing_after_residual_call` reads that as a callee escape.
        let entered = !std::ptr::eq(
            pyre_interpreter::executioncontext::vref_referent(saved_topframeref),
            frame,
        );
        if entered {
            // Same barrier obligation as the inline-call push: `f_backref` is
            // a traced `Type::Ref` field, `frame` can be old-generation, and
            // `saved_topframeref` can name a young frame.
            pyre_object::gc_hook::try_gc_write_barrier(frame as pyre_object::gc_hook::GCREF);
            majit_gc::bh_probe_note_store(
                frame as usize,
                crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
                2,
            );
            unsafe {
                (*frame).f_backref = saved_topframeref;
                (*ec).topframeref = frame;
            }
        }
        // Published whether or not this guard wrote the chain: `frame` is the
        // one a force inside the residual must redirect its escape onto
        // (`flush_active_frame_escape`), and the chain already naming it makes
        // that more true, not less.
        let previous_published = PUBLISHED_INLINE_FRAME.with(|slot| slot.replace(frame));
        let previous_shadow = PUBLISHED_INLINE_SHADOW.with(|slot| slot.replace(shadow));
        Some(Self {
            ec,
            frame_root: majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(
                frame as usize,
            )),
            previous_published,
            previous_shadow,
            entered,
        })
    }

    /// The frame as the collector last left it.  Every read below goes through
    /// here rather than through a field, so a residual that collected cannot
    /// hand `Drop` an address from before the move.
    fn frame(&self) -> *mut pyre_interpreter::PyFrame {
        self.frame_root.get().0 as *mut pyre_interpreter::PyFrame
    }
}

impl Drop for ResidualFrameChainGuard {
    fn drop(&mut self) {
        unsafe {
            // `executioncontext.py leave`: move the raw caller vref back without
            // forcing it, then, when the frame escaped, force the caller and
            // mark it escaped too.  A frame handed to application code keeps a
            // reference to its caller, so the caller must stay materialised;
            // dropping that propagation would leave the escape recorded only on
            // a frame the walk owns privately.
            PUBLISHED_INLINE_FRAME.with(|slot| slot.set(self.previous_published));
            PUBLISHED_INLINE_SHADOW.with(|slot| slot.replace(self.previous_shadow.take()));
            if self.entered {
                // The value `enter` displaced, read back off the frame it was
                // written to, so a collection during the residual forwarded it
                // in place.  Only the `entered` path wrote `f_backref`, and
                // only that path has anything to restore.
                (*self.ec).topframeref = (*self.frame()).f_backref;
            }
            let frame = self.frame();
            if (*frame).escaped() {
                let f_back = (*frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
            }
        }
    }
}

struct ActiveFrameEscapeGuard {
    prev: Option<(usize, Option<usize>)>,
    prev_stack: Option<Vec<OpRef>>,
}

impl ActiveFrameEscapeGuard {
    fn enter(frame: usize, py_pc: Option<usize>, stack: Option<Vec<OpRef>>) -> Self {
        let current = (frame != 0).then_some((frame, py_pc));
        COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.set(None));
        let latched = if current.is_some() { stack } else { None };
        let prev_stack = ACTIVE_FRAME_ESCAPE_STACK
            .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), latched));
        Self {
            prev: ACTIVE_FRAME_ESCAPE.with(|slot| slot.replace(current)),
            prev_stack,
        }
    }
}

impl Drop for ActiveFrameEscapeGuard {
    fn drop(&mut self) {
        ACTIVE_FRAME_ESCAPE.with(|slot| slot.set(self.prev));
        ACTIVE_FRAME_ESCAPE_STACK.with(|slot| *slot.borrow_mut() = self.prev_stack.take());
    }
}

/// Returns whether `frame` is the one recorded as escaping this residual call
/// (`expected == frame`) — i.e. the traced virtualizable itself was handed to
/// Python, so its tracing token must be forced.  The concrete frame an inline
/// sub-walk published counts as well: it runs under that virtualizable, so
/// handing it out escapes the virtualizable too. Any other frame a residual
/// callee inspects returns false. Independently, the walk-end resume pc is
/// committed only when the state flush succeeds (a merge point with cached
/// depth); a directly matched frame that cannot flush still escaped and must
/// be forced, so for it the two signals are decoupled.
/// Step-0 attribution probe: consume the classification recorded by the most
/// recent matched escape and tally which disjunct owns this token clear.
pub fn attribute_last_escape_force() {
    if let Some(callee_only) = LAST_ESCAPE_WAS_CALLEE_ONLY.with(|c| c.take()) {
        use crate::trace::fbw_diag;
        fbw_diag::bump(if callee_only {
            fbw_diag::ESCAPE_FORCE_BY_CALLEE_ONLY
        } else {
            fbw_diag::ESCAPE_FORCE_BY_PORTAL
        });
    }
}

/// The frame [`flush_active_frame_escape`] forced without writing, if any.
///
/// A blackhole chain that has one of these as a level would run that level
/// against an array the force never materialized, so the adopt declines and
/// the walk keeps the legacy escape/replay path.
pub(crate) fn unflushed_escaped_callee() -> usize {
    UNFLUSHED_ESCAPED_CALLEE.with(|slot| slot.get())
}

/// Clear the record at the start of a walk, so a frame forced by an earlier
/// walk cannot decline this one's adopt.
pub(crate) fn reset_unflushed_escaped_callee() {
    UNFLUSHED_ESCAPED_CALLEE.with(|slot| slot.set(0));
}

pub fn flush_active_frame_escape(ctx: &TraceCtx, frame: *mut pyre_interpreter::PyFrame) -> bool {
    // `executioncontext.py leave` — a frame handed to application code
    // keeps a reference to its caller, so escaping the concrete frame an inline
    // sub-walk published escapes the traced virtualizable it runs under.  The
    // flush stays keyed on that virtualizable, whose resume pc this residual
    // already latched, so the walk resumes forward rather than replaying from
    // entry.
    let escaped_published = PUBLISHED_INLINE_FRAME.with(|slot| {
        let published = slot.get();
        let matched = !published.is_null() && std::ptr::eq(published, frame);
        if matched {
            let f_back = unsafe { (*published).get_f_back() };
            if !f_back.is_null() {
                unsafe { (*f_back).mark_as_escaped() };
            }
        }
        matched
    });
    ACTIVE_FRAME_ESCAPE.with(|slot| {
        if let Some((expected, portal_py_pc)) = slot.get()
            && {
                let escaped_portal = expected == frame as usize;
                if escaped_portal || escaped_published {
                    use crate::trace::fbw_diag;
                    match (escaped_portal, escaped_published) {
                        (true, false) => fbw_diag::bump(fbw_diag::ESCAPE_PORTAL_ONLY),
                        (false, true) => fbw_diag::bump(fbw_diag::ESCAPE_PUBLISHED_CALLEE_ONLY),
                        (true, true) => {
                            fbw_diag::bump(fbw_diag::ESCAPE_PORTAL_AND_PUBLISHED_CALLEE)
                        }
                        (false, false) => {}
                    }
                    LAST_ESCAPE_WAS_CALLEE_ONLY
                        .with(|c| c.set(Some(!escaped_portal && escaped_published)));
                    if !escaped_portal && escaped_published {
                        // `virtualizable.py write_boxes` writes the array of
                        // the virtualizable being forced.  The flush below
                        // stays keyed on the traced virtualizable — the portal
                        // frame — so for this disjunct nothing would write the
                        // frame that was actually handed to the residual, and
                        // every local the compiled trace held in a register for
                        // it would stay absent from `locals_cells_stack_w`.  A
                        // blackhole level later resumed on that frame reads
                        // those slots as Python NULL; the witness is an
                        // `f_locals` write in the resumed tail of a recursive
                        // callee whose following `LOAD_FAST` of a local
                        // assigned one line earlier lowers to
                        // `getarrayitem_vable_r` and answers NULL.
                        //
                        // A bridge-resume sub-walk seeds this level's locals
                        // into its own `CalleeLocalsShadow`
                        // (`bridge_subwalk.rs`), which is what the publish
                        // carries here; when it cannot supply the whole region
                        // the frame is recorded so the blackhole adopt declines
                        // rather than resuming on an unwritten array.
                        //
                        // `take` rather than `get`: this is the force's
                        // write_boxes, and only the FIRST force in a residual
                        // may perform it.  The callee can legitimately mutate
                        // its own fastlocals afterwards — an `f_locals`
                        // write-through is exactly that — and a second force
                        // re-running the write would put the shadow's
                        // pre-residual values back over it.  The guard restores
                        // the previous entry when the residual ends, so the
                        // suppression is scoped to this residual.
                        let written = PUBLISHED_INLINE_SHADOW
                            .with(|slot| slot.take())
                            .is_some_and(|(state, frame_reg)| {
                                // The shadow describes ONE level's frame.  Only
                                // that frame may be written from it: a nested
                                // residual publishes its own, and writing one
                                // level's slots onto another's array would
                                // replace live values with unrelated ones.
                                let matches_frame =
                                    state.borrow().callee_shadow.as_ref().is_some_and(|shadow| {
                                        shadow.concrete_frame == frame as usize
                                    });
                                matches_frame
                                    && super::flush_callee_locals_region(&state, frame, frame_reg)
                            });
                        if !written {
                            UNFLUSHED_ESCAPED_CALLEE.with(|slot| slot.set(frame as usize));
                        }
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-escape-callee] frame={:#x} locals written={written}",
                                frame as usize
                            );
                        }
                    }
                    true
                } else {
                    false
                }
            }
        {
            // Force #2+ within this residual (`enter` resets the committed pc
            // per residual): the live frame is already heap-authoritative
            // from the first flush, and the callee may have legitimately
            // mutated it since (an `f_locals` write-through) — re-flushing
            // would overwrite that mutation with the same walk-end values.
            // The token force is a no-op then too (`force_now` on
            // TOKEN_NONE, virtualizable.py).
            if COMMITTED_FRAME_ESCAPE_PC
                .with(|committed| committed.get())
                .is_some()
            {
                return true;
            }
            capture_escape_flush_undo(expected);
            let (latched, flushed) = if let Some(py_pc) = portal_py_pc {
                let latched = flush_escape_state_with_latched_stack(ctx, expected, py_pc);
                let flushed =
                    latched || crate::state::flush_walk_end_state_to_frame(ctx, expected, py_pc);
                (latched, flushed)
            } else {
                (false, false)
            };
            // Reachability probe: the latched-stack path is gated on
            // `escape_opcode_window_clean`, the plain fallback is not, and this
            // resume pc re-runs the whole escaping opcode.
            if let Some(py_pc) = portal_py_pc
                && !latched
                && flushed
            {
                use crate::trace::fbw_diag;
                fbw_diag::bump(fbw_diag::ESCAPE_PLAIN_FALLBACK);
                if !escape_opcode_window_clean(py_pc) {
                    fbw_diag::bump(fbw_diag::ESCAPE_PLAIN_FALLBACK_UNCLEAN);
                }
            }
            if flushed {
                record_escape_flush_image(expected);
                let kind = if latched {
                    EscapeResumeKind::Exact
                } else {
                    EscapeResumeKind::RerunsOpcode
                };
                if let Some(py_pc) = portal_py_pc {
                    COMMITTED_FRAME_ESCAPE_PC.with(|committed| committed.set(Some((py_pc, kind))));
                }
            } else if crate::state::flush_locals_region_to_frame(ctx, expected) {
                record_escape_flush_image(expected);
                // The locals write claims no resume pc (`flush_locals_region_to_frame`
                // leaves `last_instr` alone) but `LiveLastInstrGuard` still keeps
                // the executing opcode, because this capture is armed. A walk
                // that later commits a resume overwrites that coordinate. One
                // that does not must get the pre-flush frame back: the
                // executing opcode of a `raise` inside `for` is followed by
                // `END_FOR`, and the interpreter pops an iterator the entry
                // stack does not hold. The epilogue restores only when no
                // resume-past leg committed (`take_escape_flush_undo_pending`).
                mark_escape_flush_undo_pending();
            } else {
                // All-or-nothing decline: nothing was written, nothing to undo.
                discard_escape_flush_undo();
            }
            // A declined full flush still escaped the virtualizable, so the
            // locals region is written anyway (`virtualizable.py write_boxes
            // write_boxes` has no decline) — otherwise the callee reads an
            // array of nulls.  That write claims no resume pc.  The capture
            // stays armed and `mark_escape_flush_undo_pending` asks the
            // walk-end epilogue to put the pre-flush frame back when no
            // resume-past leg commits.  Restoring here, inside the residual,
            // would rewind `last_instr` before a later `sys._getframe` read
            // in the same walk (`getframe_caller_resume_coord_two_call_sites`).
            //
            // Upstream reports the escape from the vable token state alone,
            // independent of any resume-image write.  See
            // `virtualizable.py` (`tracing_after_residual_call` /
            // `force_now`), `virtualizable.py` (token states),
            // `virtualref.py` (vref'd inlined callee), and
            // `pyjitpl.py vable_after_residual_call` (unconditional ABORT_ESCAPE).  Runtime
            // forcing also resets the token before writing fields
            // (`resume.py:1405-1408`).  Therefore a matched guard reports the
            // escape even when the flush declined.
            return true;
        }
        false
    })
}

/// Snapshot the frame region a successful escape flush overwrites (the whole
/// `locals_cells_stack_w` array plus `last_instr`/`valuestackdepth`).  GC:
/// the store-journal root area walks the armed capture's slots in place (see
/// [`escape_flush_undo_cell_ptr`]).  A second force in the same residual
/// keeps the OLDEST capture (the true pre-flush state); staleness across walk
/// attempts is impossible — the walk-start reset discards any leftover.
fn capture_escape_flush_undo(frame: usize) {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        let mut slot = slot.borrow_mut();
        match slot.as_ref() {
            // Keep the oldest capture for the SAME frame (true pre-flush state).
            Some(existing) if existing.frame == frame => return,
            // A capture for a different frame is a stale leftover from an
            // unwound path; its frame may be dead — drop without restoring.
            Some(_) => *slot = None,
            None => {}
        }
        let pf = unsafe { &*(frame as *const pyre_interpreter::PyFrame) };
        // With a [`LiveLastInstrGuard`] live on this frame the field holds the
        // EXECUTING pc, not the resume coordinate.  The undo exists to hand a
        // legacy replay a pristine pre-flush frame, and a replay needs the
        // resume pc — so capture the value the guard displaced.
        let last_instr = PUBLISHED_LAST_INSTR
            .with(|slot| slot.get())
            .filter(|(published, _)| *published == frame)
            .map_or(pf.last_instr, |(_, saved)| saved);
        *slot = Some(EscapeFlushUndo {
            frame,
            frame_root: owner_root_if_gc(frame),
            last_instr,
            valuestackdepth: pf.valuestackdepth,
            slots: locals_w!(pf).as_slice().to_vec(),
            flush_image: Vec::new(),
        });
    });
}

/// Report what the post-residual shadow refresh left behind, and optionally
/// import the residual's own fastlocals writes into the shadow.
///
/// Every point that publishes walk state to the frame -- the escape flush, the
/// walk-end flush, the resume image, the undo restore -- writes the shadow over
/// the locals region.  So a fastlocals write made by the residual survives only
/// if the shadow learned about it first.  `flush_image` is the region as the
/// flush left it, so a slot that has moved since is exactly one the residual
/// wrote.
///
/// The import is gated on [`fbw_import_residual_locals_enabled`] because it
/// updates the concrete half alone: the recorded IR still names the box the
/// walk was holding, which is wrong for anything but this one recording.  It is
/// here to answer whether the shadow is the channel, not to ship.
fn report_post_residual_shadow<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    live_frame: usize,
    frame_root: Option<&majit_gc::shadow_stack::OwnerRootGuard>,
    before: Option<&[pyre_object::PyObjectRef]>,
) {
    let diag = fbw_debug_abort_enabled();
    let import = fbw_import_residual_locals_enabled();
    if !diag && !import {
        return;
    }
    let base = match ctx.trace_ctx.virtualizable_info() {
        Some(info) => info.num_static_extra_boxes,
        None => return,
    };
    let heap = ctx.trace_ctx.diag_virtualizable_heap_ptr();
    // Baseline for "what the residual wrote".  A flush rewrote the locals region
    // from the shadow, so once there is a flush image that is the only sound
    // baseline: measuring from before the call would report the flush's own
    // stores as the residual's.  Measured, that difference is the loop
    // variable -- the flush writes it, the residual does not -- and adopting it
    // makes the compiled loop re-read its counter out of the array, one
    // iteration ahead.  With no flush the frame was never rewritten, so the
    // pre-call image is exact.
    let flushed = ESCAPE_FLUSH_UNDO.with(|slot| {
        let slot = slot.borrow();
        slot.as_ref()
            .filter(|undo| {
                (undo.frame == live_frame || undo.current_frame() == live_frame)
                    && !undo.flush_image.is_empty()
            })
            .map(|undo| undo.flush_image.clone())
    });
    let baseline: &[pyre_object::PyObjectRef] = match (flushed.as_deref(), before) {
        (Some(image), _) => image,
        (None, Some(image)) => image,
        (None, None) => {
            if diag {
                eprintln!("[fbw-post-residual] skip=no-baseline frame={live_frame:#x}");
            }
            return;
        }
    };
    let origin = if flushed.is_some() {
        "flush"
    } else {
        "pre-call"
    };
    if live_frame == 0 {
        if diag {
            eprintln!("[fbw-post-residual] skip=no-live-frame origin={origin}");
        }
        return;
    }
    // `flush_image.clone` above can collect. Re-read the owner root after
    // that allocation; the address argument is the word from before it.
    let frame = frame_root.map(|root| root.get().0).unwrap_or(live_frame);
    let pf = unsafe { &*(frame as *const pyre_interpreter::PyFrame) };
    let live = locals_w!(pf).as_slice();
    let n = live.len().min(baseline.len());
    let moved: Vec<(usize, pyre_object::PyObjectRef)> = (0..n)
        .filter(|&i| live[i] != baseline[i])
        .map(|i| (i, live[i]))
        .collect();
    if moved.is_empty() {
        return;
    }
    if diag {
        let stale: Vec<usize> = moved
            .iter()
            .filter(|(abs, live)| {
                ctx.trace_ctx
                    .virtualizable_entry_at(base + abs)
                    .is_none_or(|(_, v)| {
                        crate::state::boxed_slot_value_for_type(majit_ir::Type::Ref, &v) != *live
                    })
            })
            .map(|(abs, _)| *abs)
            .collect();
        eprintln!(
            "[fbw-post-residual] origin={origin} heap={heap:#x} frame={frame:#x} same={} moved={:?} shadow_stale={stale:?}",
            heap == frame,
            moved.iter().map(|(abs, _)| *abs).collect::<Vec<_>>(),
        );
    }
    if import {
        adopt_residual_locals_writes(ctx, base, &moved);
    }
}

/// The frame a residual argument gives a live view of, following the one
/// wrapper a bound read puts in the way.
///
/// `p["x"]` hands the proxy to the residual directly, but `p.get("x")` binds
/// the method first and the CALL then carries a `Method` whose `__self__` is
/// the proxy.  Nothing deeper is followed: a wrapper that hides the receiver
/// from the residual's own arguments hides it from the compiled call too.
fn proxy_viewed_frame(obj: pyre_object::PyObjectRef) -> Option<pyre_object::PyObjectRef> {
    use pyre_interpreter::pyframe::frame_locals_proxy::viewed_frame;
    if obj.is_null() || unsafe { (*obj).ob_type.is_null() } {
        return None;
    }
    if let Some(frame) = viewed_frame(obj) {
        return Some(frame);
    }
    let receiver = unsafe { pyre_object::is_method(obj) }
        .then(|| unsafe { pyre_object::w_method_get_self(obj) })
        .filter(|recv| !recv.is_null())?;
    viewed_frame(receiver)
}

/// Whether a residual's Ref-bank operands may hold something other than an
/// object, so reading a type off one is not sound.
///
/// The operand list carries one word per `arg_types()` entry, which is a list
/// of objects only where every parameter is one machine word wide.  Two
/// families break that, and neither call is executed by the walk:
///
/// * a helper published with a parameter wider than a slot, which
///   [`pyre_interpreter::is_abi_unsound_argument_residual`] names;
/// * an un-lowered helper, whose funcbox is a symbolic hash rather than an
///   address and whose Rust signature nothing checked — `get_and_call_function`
///   takes `args_w: &[PyObjectRef]`, so its fourth Ref slot holds half of a
///   slice rather than an object.
///
/// [`try_execute_residual_call_via_executor`] declines both on the same test,
/// but it runs after this write-back, so the scan cannot wait for its answer.
fn residual_operands_are_not_all_objects<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
) -> bool {
    let Some(&funcbox) = allboxes.first() else {
        return false;
    };
    let Some(majit_ir::Value::Int(addr)) = ctx.trace_ctx.box_value(funcbox) else {
        // A funcbox the walk cannot read is one it cannot classify, and the
        // executor declines on the same condition.
        return true;
    };
    majit_jitcode::codewriter::call::is_symbolic_fnaddr(addr)
        || pyre_interpreter::is_abi_unsound_argument_residual(addr as usize)
        || BIGINT_FNADDRS.contains(&addr)
}

/// Write the traced frame's locals region out before a residual that reads it
/// through a live `FrameLocalsProxy`.
///
/// The proxy reads the frame's ARRAY, while a recording walk holds those locals
/// in the virtualizable's boxes.  The `f_locals` getter fold already writes the
/// region out when it hands a proxy over
/// (`specialize.rs walker_write_back_standard_frame_locals`), but a proxy is a
/// live view and outlives that write: bind one before a loop the JIT afterwards
/// compiles and every later read answers with what the local held at the bind.
/// Measured at `N=4000`, `p = f_locals` outside the loop and `p["x"]` in the
/// body reads 1041 -- the compile threshold -- where the same read spelled
/// `sys._getframe(0).f_locals["x"]`, which re-enters the fold each iteration,
/// reads 3999.
///
/// A force at the read accessor is the shape this does NOT take, and
/// `PyFrame::force_locals` records why: `hook_access_field` runs in residual
/// code while a trace redirects the same accesses to the shadow, so forcing on
/// every read forces the traced ones too and aborts the loop.  The write-back
/// is not a force -- it is a mirror onto the concrete frame plus the same store
/// into the trace, and none of its ops can reach `force_frame`.
///
/// Scoped to the standard virtualizable, gated on BOTH its red box and its
/// concrete pointer, exactly as the getter fold gates itself.  A proxy onto any
/// other frame is left alone: that frame's locals are in its own array already.
///
/// Every spelling of the read reaches the array through a residual that carries
/// the proxy itself -- `p["x"]`, `len(p)`, `dict(p)`, and a helper the proxy is
/// passed to, whose own subscript is the residual that carries it -- with one
/// exception: `p.get("x")` binds `get` first, so the CALL sees a `Method` and
/// the proxy only as its receiver.  [`proxy_viewed_frame`] takes that hop.
///
/// Called beside `vable_and_vrefs_before_residual_call`, which is the only
/// point that satisfies the ordering: the dispatchers `record_op_with_descr`
/// the CALL and only then hand it to
/// [`try_execute_residual_call_via_executor`], so anything emitted from inside
/// the executor lands AFTER the call in the trace.  Measured with the stores
/// there, the walk still answered correctly -- the concrete mirror runs at
/// record time either way -- while the compiled loop read what the PREVIOUS
/// iteration's stores had left: `p["x"]` in a `x = i` body reported a lag of
/// one on 1479 of the 2958 iterations past the compile point on dynasm, and
/// none on cranelift.  Emitted here the stores sit between the token store and
/// the call, where the residual reads them.
pub(crate) fn write_back_locals_for_proxy_reader<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
) {
    let (Some(frame_box), Some(frame_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return;
    };
    if residual_operands_are_not_all_objects(ctx, allboxes) {
        return;
    }
    let mut reads_traced_frame = false;
    for &arg in allboxes {
        let Some(obj) = walker_concrete_ref_object(ctx, arg) else {
            continue;
        };
        if proxy_viewed_frame(obj).is_some_and(|frame| frame as usize == frame_ptr) {
            reads_traced_frame = true;
            break;
        }
    }
    if reads_traced_frame {
        // A decline writes nothing and leaves the read as it is today.
        let _ = crate::jitcode_dispatch::specialize::walker_write_back_standard_frame_locals(
            ctx, frame_box, frame_ptr,
        );
    }
}

/// Adopt into the walk the fastlocals a residual wrote to the live frame.
///
/// The walk holds each local as a box and reads it straight out of the
/// virtualizable shadow (`pyjitpl.py _opimpl_getarrayitem_vable`, standard
/// leg), so a residual that writes the frame's fastlocals behind the walk's
/// back -- an `f_locals` write-through from the callee is the reachable case --
/// leaves every later read, and the values the loop closes on, holding what the
/// local was before the call.
///
/// The standing objection to re-reading a standard virtualizable mid-trace is
/// that the array holds values the trace never wrote.  What answers it is not
/// the force but *`moved`*: the caller diffs the region against the image it
/// took before the call, so only the slots the call itself changed are adopted,
/// and a slot the walk is ahead on -- unchanged during the call -- is not in the
/// set.  That holds on both arms, the one where the residual forced the region
/// out of the shadow first and the one where nothing forced at all.
///
/// The re-read is RECORDED, not folded to the value seen now: compiled code has
/// to perform the same load after its own force, so this emits
/// `GETARRAYITEM_GC_R` off the frame's `locals_cells_stack_w` and hands the
/// shadow that OpRef.  Folding the concrete value in would bake one recording's
/// answer into every execution.
fn adopt_residual_locals_writes<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    base: usize,
    moved: &[(usize, pyre_object::PyObjectRef)],
) {
    if moved.is_empty() || ctx.fbw_mode.snapshot_sym.is_null() {
        return;
    }
    let frame = unsafe { (*ctx.fbw_mode.snapshot_sym).frame() };
    if frame.is_none() {
        return;
    }
    let array = crate::state::frame_locals_cells_stack_array(ctx.trace_ctx, frame);
    for &(abs, live) in moved {
        if ctx.trace_ctx.virtualizable_entry_at(base + abs).is_none() {
            continue;
        }
        let index = ctx.trace_ctx.const_int(abs as i64);
        let item = crate::state::trace_array_getitem_value(ctx.trace_ctx, array, index);
        let value = majit_ir::Value::Ref(majit_ir::GcRef(live as usize));
        ctx.trace_ctx.set_opref_concrete(item, value);
        ctx.trace_ctx
            .set_virtualizable_entry_at(base + abs, item, value);
    }
}

/// Record the locals region as the flush just left it, for the frame the
/// pending capture holds.  Runs after the flush and before the residual's
/// user Python can touch the region, so any later difference from this image
/// is that user code's own write.  A second flush for the same frame supersedes
/// the image: the restore must undo the flush that is actually standing.
fn record_escape_flush_image(frame: usize) {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(undo) = slot.as_mut() else {
            return;
        };
        if undo.frame != frame && undo.current_frame() != frame {
            return;
        }
        // The flush boxes Int/Float locals and can collect. The captured
        // address is not reloaded: the nursery stub is gone once the
        // collection finishes. Read the owner root instead.
        let frame_now = undo.current_frame();
        let pf = unsafe { &*(frame_now as *const pyre_interpreter::PyFrame) };
        undo.flush_image = locals_w!(pf).as_slice().to_vec();
    });
}

/// Put the pre-flush frame state back so the legacy replay re-enters a
/// pristine frame.  Called from the unforced / rootless continuation (where
/// the walk goes on and must not see the moved frame) and, for a withdrawn
/// commit, from the walk-end epilogue once no resume-PAST continuation has
/// claimed the flushed frame.
pub(crate) fn restore_escape_flush_undo() {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        let Some(undo) = slot.borrow_mut().take() else {
            return;
        };
        // A minor collection between the flush and this restore moves a
        // nursery frame and then drops the forwarding stub. Re-read the
        // owner root taken at capture; the captured word is the old address.
        let frame = undo.current_frame();
        unsafe {
            let pf = &mut *(frame as *mut pyre_interpreter::PyFrame);
            let arr_ptr = pf.locals_cells_stack_w;
            let dst = locals_w_mut!(pf);
            let n = undo.slots.len().min(dst.as_slice().len());
            // Three-way merge: pre-flush capture, the image the flush left,
            // and what the frame holds now.  A slot the residual did not touch
            // still reads as the flush left it, and only those go back to their
            // pre-flush value.  A slot that moved since carries a write made
            // after the capture -- `f_locals` write-through from the callee is
            // the reachable case -- and putting the capture back over it would
            // discard that write.  The force-#2 arm above already refuses to
            // re-flush for this reason; this is the same refusal on the way out.
            //
            // With no image (the flush declined, or an arm that never records
            // one) `merge` is false and every captured slot goes back, as before.
            let merge = undo.flush_image.len() >= n;
            // Report-only: `differs` counts the slots that moved since the
            // capture, `kept` those the merge leaves alone.  Gated, so a
            // production build pays nothing.
            if fbw_debug_abort_enabled() {
                let live = dst.as_slice();
                let differs = (0..n).filter(|&i| live[i] != undo.slots[i]).count();
                let kept = if merge {
                    (0..n).filter(|&i| live[i] != undo.flush_image[i]).count()
                } else {
                    0
                };
                if differs != 0 {
                    let moved: Vec<usize> =
                        (0..n).filter(|&i| live[i] != undo.slots[i]).collect();
                    let held: Vec<usize> = if merge {
                        (0..n).filter(|&i| live[i] != undo.flush_image[i]).collect()
                    } else {
                        Vec::new()
                    };
                    eprintln!(
                        "[fbw-undo-clobber] frame={frame:#x} n={n} differs={differs} moved={moved:?} kept={kept} held={held:?}"
                    );
                }
            }
            for (i, &v) in undo.slots.iter().take(n).enumerate() {
                if merge && dst.as_slice()[i] != undo.flush_image[i] {
                    continue;
                }
                dst[i] = v;
            }
            pf.last_instr = undo.last_instr;
            pf.valuestackdepth = undo.valuestackdepth;
            // The forward flush arms this per store; the restore writes values
            // that are already boxed, so nothing allocates between them and one
            // arming covers the whole image.  Without it a pre-flush value that
            // is nursery-young goes back into an old-gen array with no
            // remembered-set entry to re-trace it.
            crate::state::frame_array_write_barrier(frame as *mut u8, arr_ptr);
        }
    });
}

pub(crate) fn discard_escape_flush_undo() {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        *slot.borrow_mut() = None;
    });
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.set(false));
}

/// Note that the forced residual's commit was withdrawn, so the pre-flush
/// frame has to come back IF the walk ends up on the legacy replay.  The
/// restore itself waits for [`take_escape_flush_undo_pending`] at walk end:
/// the resume-PAST continuation keeps the flushed frame (upstream's
/// `virtualizable.py write_boxes` has no undo once the vable is
/// forced), and only the replay-from-entry leg needs the pre-walk state back.
fn mark_escape_flush_undo_pending() {
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.set(true));
}

/// Consume the deferred-restore request armed by the force arm.
pub(crate) fn take_escape_flush_undo_pending() -> bool {
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.replace(false))
}

/// Opcode-scoped effect window check (see [`ESCAPE_OPCODE_WINDOW`]): true iff
/// every earlier residual of the CURRENT Python opcode is declared re-runnable.
/// A pure query — the window is written only by [`escape_opcode_window_note`],
/// so the residual asking the question never disqualifies itself.
///
/// Keyed on `py_pc` alone: an opcode revisited across walked inner-loop
/// iterations still sees the FIRST visit's verdict, so every revisit after any
/// non-re-runnable residual declines the latch — conservative (decline →
/// legacy).  An unexplained latch-decline spike on nested-loop shapes is this;
/// the refinement would reset the window on back-edge re-entry.
fn escape_opcode_window_clean(py_pc: usize) -> bool {
    ESCAPE_OPCODE_WINDOW.with(|slot| match slot.get() {
        Some((pc, clean)) if pc == py_pc => clean,
        _ => true,
    })
}

/// Declare this residual's re-runnability into the opcode window (see
/// [`ESCAPE_OPCODE_WINDOW`]).  Called for every residual that reaches
/// execution, AFTER the latch gate has read the window.
///
/// `reentrant` is the declared effect class, `EF_ELIDABLE_*`
/// (`check_is_elidable`) or `EF_LOOPINVARIANT`.  It is deliberately STRICTER
/// than `provably_side_effect_free`, which additionally exempts
/// [`majit_ir::RuntimeHelperKind::ForIterNext`]: that exemption answers a
/// different question (the consume is the SOURCE of an in-flight item, not a
/// body effect for it), and a user-defined `__next__` runs user bytecode that
/// re-executing the opcode would re-run.
fn escape_opcode_window_note(py_pc: usize, reentrant: bool) {
    ESCAPE_OPCODE_WINDOW.with(|slot| {
        let clean = match slot.get() {
            Some((pc, clean)) if pc == py_pc => clean,
            _ => true,
        };
        slot.set(Some((py_pc, clean && reentrant)));
    });
}

/// Reset the opcode window at walk start so a prior trace's sample cannot
/// alias a same-pc opcode of the new walk.
pub(crate) fn escape_opcode_window_reset() {
    ESCAPE_OPCODE_WINDOW.with(|slot| slot.set(None));
}

/// Resolve the latched operand-stack mirror (`ActiveFrameEscapeGuard::enter`)
/// to concrete refs and flush the walk-end state with the mid-expression
/// stack the vable shadow cannot provide (its stack region is only valid at
/// merge points).  The `_run_forever` continue-forward analog
/// (`blackhole.py`) for the escape abort: the resume state is the exact
/// abort-point frame, so the walk's applied effects stand and nothing
/// replays.  Returns false (no frame mutation) when no stack was latched or
/// any slot lacks a concrete non-null Ref — the caller then falls back to the
/// plain merge-point flush.
fn flush_escape_state_with_latched_stack(ctx: &TraceCtx, frame: usize, py_pc: usize) -> bool {
    ACTIVE_FRAME_ESCAPE_STACK.with(|slot| {
        let latched = slot.borrow();
        let Some(oprefs) = latched.as_ref() else {
            return false;
        };
        flush_with_latched_stack(ctx, frame, py_pc, oprefs)
    })
}

/// Resolve a latched operand-stack OpRef mirror to concrete refs and flush the
/// walk-end state with it, keeping the resolved refs rooted across the call.
///
/// Shared by the escape flush above and the `ABORT_FORCE_QUASIIMMUT` leg
/// ([`flush_qmut_abort_state`]): both resume the interpreter
/// mid-expression, where the vable shadow's stack region reads NULL and only the
/// walk's own mirror can say what the operands are.
pub(crate) fn flush_with_latched_stack(
    ctx: &TraceCtx,
    frame: usize,
    py_pc: usize,
    oprefs: &[OpRef],
) -> bool {
    flush_with_latched_stack_inner(ctx, frame, py_pc, oprefs, None, ConsumedOperands::TosOnly)
}

/// How much of the latch may disagree with the vable shadow because the
/// in-progress opcode has already popped it into registers — see
/// [`flush_with_latched_stack_inner`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConsumedOperands {
    /// One slot, the opcode's TOS.
    TosOnly,
    /// The whole operand run, however deep.
    WholeRun,
}

/// Resume a FOR_ITER body from the complete header stack plus the item the
/// authoritative walk already consumed.  RPython keeps both pieces in the
/// live MIFrame register image; pyre records the item in the FOR_ITER carrier
/// separately from the operand-stack mirror, so they must be reunited before
/// converting the frame to interpreter state.
pub(crate) fn flush_with_latched_stack_and_item(
    ctx: &TraceCtx,
    frame: usize,
    header_py_pc: usize,
    oprefs: &[OpRef],
    push: (pyre_object::PyObjectRef, usize),
) -> bool {
    flush_with_latched_stack_inner(
        ctx,
        frame,
        header_py_pc,
        oprefs,
        Some(push),
        ConsumedOperands::TosOnly,
    )
}

fn flush_with_latched_stack_inner(
    ctx: &TraceCtx,
    frame: usize,
    py_pc: usize,
    oprefs: &[OpRef],
    push: Option<(pyre_object::PyObjectRef, usize)>,
    consumed_operands: ConsumedOperands,
) -> bool {
    {
        let mut stack = Vec::with_capacity(oprefs.len() + usize::from(push.is_some()));
        for &opref in oprefs.iter() {
            match ctx.concrete_of_opref(opref) {
                Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
                    // A recorded concrete NULL is a real operand, not an absent
                    // one — `NO_CONCRETE` is what "unavailable" looks like.  It
                    // is the `self_or_null` sentinel `PUSH_NULL` puts under a
                    // callable, so refusing it made every `CALL` /
                    // `CALL_FUNCTION_EX` escape fall through to the plain flush,
                    // which then declines on that same NULL slot and leaves the
                    // portal replaying the frame from its entry.
                    stack.push(r.as_usize() as pyre_object::PyObjectRef);
                }
                other => {
                    if fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-latched-flush] DECLINE at py_pc={py_pc}: slot {} of {} \
                             opref={opref:?} concrete={other:?}",
                            stack.len(),
                            oprefs.len(),
                        );
                    }
                    return false;
                }
            }
        }
        let resume_py_pc = if let Some((item, body_pc)) = push {
            stack.push(item);
            body_pc
        } else {
            py_pc
        };
        // Why this latch exists, checkable at runtime.  A slot that disagrees
        // with the vable shadow holds a compile-time NULL *constant* rather
        // than a stale or absent value: exactly what `popvalue_maybe_none`
        // writes (`pyframe.py` →
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, ConstPtr.NULL)`
        // via `jtransform.py do_fixed_list_setitem`).  The opcode had already popped the slot
        // before its residual forced.
        //
        // So the array is correct and upstream agrees — RPython's `popvalue`
        // NULLs the slot the same way.  What this latch holds is the operand
        // the in-flight opcode already consumed, which is what upstream keeps
        // in `MIFrame.registers_r` and what `convert_and_run_from_pyjitpl`
        // resumes a blackhole from.  Emitting more vable stores does not
        // remove the need for it: `LOAD_ATTR` already emits the push mirror
        // via `emit_pushvalue_ref!` and its slot still reads NULL, because the
        // pop follows the push.
        //
        // How much of that run may disagree is [`ConsumedOperands`], and the
        // two answers are about WHEN the leg's resume pc runs, not about the
        // latch.
        //
        // `WholeRun` is the qmut abort's: it fires BEFORE its residual
        // executes, so the opcode is entirely ahead of the walk and resuming
        // at it re-runs nothing.  A `CALL` pops its whole callable/self/args
        // run into registers before that residual — for a class statement the
        // jitcode nulls four operand slots between the `LOAD_CONST` of the
        // class name and the `CallFn` the abort fires at — so every one of
        // them reads NULL here and a one-slot exemption declines on the first.
        //
        // `TosOnly` is the escape leg's, and it is a restriction rather than a
        // measurement: that leg's opcode is IN PROGRESS (its residual is what
        // forced), so its `last_instr = pc - 1` says "about to run the CALL"
        // while the callee doing the forcing is already inside it.  Admitting
        // the deeper run there turned `caller_f_lasti_across_residual_call`
        // red — 2 of 20000 reads answered 42, the `LOAD_GLOBAL` ahead of the
        // call, instead of the `CALL` at 52.  Widening it is part of that
        // fixture's own epic, not of this check.
        //
        // Pops run top-down, so the consumed operands are a SUFFIX of the
        // latch.  What stays refused either way is a disagreement against a
        // non-NULL shadow, and a live operand sitting ABOVE a consumed one.
        // Both say the latch is not this frame's continuation.
        let base = ctx
            .virtualizable_info()
            .map_or(usize::MAX, |info| info.num_static_extra_boxes);
        let nlocals = crate::state::concrete_nlocals(frame).unwrap_or(usize::MAX);
        // Lowest slot the in-progress opcode has already popped, once one is
        // seen: everything from here up is its own operand run.
        let mut consumed_from: Option<usize> = None;
        for (rel, &obj) in stack.iter().enumerate() {
            let entry =
                ctx.virtualizable_entry_at(base.saturating_add(nlocals).saturating_add(rel));
            let shadow = entry.map(|(_opref, value)| value);
            let shadow_ptr = match shadow {
                Some(majit_ir::Value::Ref(r)) => Some(r.as_usize()),
                _ => None,
            };
            let agrees = shadow_ptr == Some(obj as usize);
            if !agrees && fbw_debug_abort_enabled() {
                // Report the OpRef too: a live box with a NULL value means
                // the symbolic write landed and only the concrete mirror is
                // absent, which is a different defect from no write at all.
                eprintln!(
                    "[r6-latch] slot {rel}/{} latched=0x{:x} shadow={shadow:?} box={:?} (DISAGREES)",
                    stack.len(),
                    obj as usize,
                    entry.map(|(opref, _)| opref),
                );
            }
            // The orthodox disagreement: the vable slot is NULL while the
            // MIFrame register image still carries the object this opcode
            // consumed out of it.
            let consumed = !agrees
                && obj as usize != 0
                && shadow_ptr == Some(0)
                && (consumed_operands == ConsumedOperands::WholeRun || rel + 1 == stack.len());
            let refusal = if consumed {
                consumed_from.get_or_insert(rel);
                None
            } else if !agrees {
                Some("disagrees against a shadow that is not a consumed NULL")
            } else if consumed_from.is_some() && shadow_ptr != Some(0) {
                Some("is a live operand above the consumed run")
            } else {
                None
            };
            if let Some(refusal) = refusal {
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-latched-flush] DECLINE at py_pc={py_pc}: slot {rel}/{} {refusal}",
                        stack.len(),
                    );
                }
                return false;
            }
        }
        // The flush's Int/Float local boxing can trigger a minor collection;
        // register the resolved refs as resume roots so they are forwarded in
        // place across it (the same discipline as the vable root above).
        let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_raw_parts_mut(
                stack.as_mut_ptr() as *mut i64,
                stack.len(),
            ));
        }
        let committed = crate::state::flush_walk_end_state_to_frame_with_full_stack(
            ctx,
            frame,
            resume_py_pc,
            &stack,
        );
        majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
        committed
    }
}

/// The `ABORT_FORCE_QUASIIMMUT` leg's flush: resume the interpreter at the
/// opcode carrying the forcing write, with the operand stack the walk mirrored
/// on entry to it ([`fbw_qmut_abort_stack_take`]).
pub(crate) fn flush_qmut_abort_state(
    ctx: &TraceCtx,
    frame: usize,
    py_pc: usize,
    oprefs: &[OpRef],
) -> bool {
    flush_with_latched_stack_inner(ctx, frame, py_pc, oprefs, None, ConsumedOperands::WholeRun)
}

/// Take the committed escape resume pc and which flush produced it (the
/// walk-end commit contract turns on the difference — see [`EscapeResumeKind`]).
pub(crate) fn take_committed_frame_escape_pc() -> Option<(usize, EscapeResumeKind)> {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.take())
}

fn committed_frame_escape_pc() -> Option<(usize, EscapeResumeKind)> {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.get())
}

/// Withdraw a committed escape resume pc: the residual that escaped also ran
/// user bytecode, so resuming AT its opcode would re-execute that user body —
/// the legacy replay (whose journals roll back) is the never-double fallback.
fn cancel_committed_frame_escape_pc() {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.set(None));
}

/// Reject a residual_call whose `allboxes` (funcbox + permuted args)
/// contains an `OpRef::NONE`.  RPython's `do_residual_call` resolves
/// each argbox through `env[box]`, where an unbound box is a `KeyError`;
/// recording the op anyway lets `OpRef::NONE` reach the backend's
/// `resolve_opref`, which aborts the process.  Returning
/// `ResidualCallArgUnbound` instead lets the outer walker fall back to a
/// trace abort — the same graceful outcome a pre-seam inline arm reached
/// when its payload read surfaced `GotoIfNotValueNotConcrete`.
pub(crate) fn ensure_residual_call_args_bound(
    allboxes: &[OpRef],
    pc: usize,
) -> Result<(), DispatchError> {
    if let Some(arg_index) = allboxes.iter().position(|b| b.is_none()) {
        return Err(DispatchError::ResidualCallArgUnbound { pc, arg_index });
    }
    Ok(())
}

/// EffectInfo-driven opcode selector shared by `dispatch_residual_call_*`
/// dispatchers. Mirrors `pyjitpl.py do_residual_call`'s
/// precedence:
///   1. **forces branch** (`pyjitpl.py`): outer check on
///      `assembler_call or check_forces_virtual_or_virtualizable()`
///      records `CALL_MAY_FORCE_*` at step 2 and unconditionally fires
///      `GUARD_NOT_FORCED` (`:2079`).  The release-gil sub-case
///      (`pyjitpl.py if effectinfo.is_call_release_gil()`) is
///      handled by [`direct_call_release_gil`] **before** this
///      selector is called — the dispatcher early-returns on
///      `ei.is_call_release_gil()` so this function only ever sees
///      EI values where the sub-case is not active.
///   2. `EF_LOOPINVARIANT` (`:2087-2110`): `CALL_LOOPINVARIANT_*`.
///   3. `check_is_elidable()` (`:2112-2126`): `CALL_PURE_*`.
///   4. default (`:2126`): plain `CALL_*`.
///
/// Returns the `Call*` opcode for the call itself, whether
/// `handle_possible_exception` should emit `GUARD_NO_EXCEPTION`
/// (`check_can_raise(False)`), and whether the unconditional
/// `GUARD_NOT_FORCED` from the forces branch (`pyjitpl.py`)
/// should fire.
///
/// Non-elidable concrete-execute inventory.
///
/// PyPy `do_residual_call` (pyjitpl.py) concrete-executes
/// the helper at trace-record time across the **forces** /
/// **loopinvariant** (cache miss) / **elidable** / **default**
/// branches via `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=is_elidable)`.  Two narrower branches sit
/// outside this uniform call:
///   * `OS_NOT_IN_TRACE` short-circuits through
///     `do_not_in_trace_call` (`pyjitpl.py`) and never
///     reaches `executor.execute_varargs`.
///   * `is_call_release_gil()` runs the helper through
///     `do_call_release_gil` (`pyjitpl.py`), invoking
///     `executor.execute_varargs` directly **before** the recorded
///     `CALL_RELEASE_GIL_*` op is emitted.
/// The recorded opcode kind selected above is for the IR trace
/// only; concrete execution either fired (or was intentionally
/// skipped via the two narrow branches above) before the trace op
/// hits the recorder.
///
/// Per-branch concrete-execute status (every residual call with concrete
/// operands executes during the walk, matching
/// PyPy `do_residual_call` which runs `executor.execute_varargs` for the
/// whole forces branch regardless of EI):
///
/// | EI branch | Selected op | Pyre walker concrete-execute |
/// |---|---|---|
/// | `is_call_release_gil()` | (early-routed to [`direct_call_release_gil`], records `CALL_RELEASE_GIL_*`) | executed as `CallMayForce*` on the **original** `allboxes` via [`try_execute_residual_call_via_executor`] |
/// | `check_forces_virtual_or_virtualizable()` | `CallMayForce*` + `GuardNotForced` | executed via [`try_execute_residual_call_via_executor`] (active vable bracketed by the token protocol) |
/// | `extraeffect == LoopInvariant` | `CallLoopinvariant*` | executed on cache miss; [`loopinvariant_lookup`] reuses the cached OpRef on hit (no execute, no record) |
/// | `check_is_elidable()` | `CallPure*` | cannot-raise calls use [`try_fold_pure_call_via_executor`]; can-raise calls use [`try_execute_residual_call_via_executor`] so the exception is transcribed |
/// | default | `Call*` + (`GuardNoException` iff can_raise) | executed via [`try_execute_residual_call_via_executor`] |
///
/// All three dispatch entry points — [`dispatch_residual_call_iRd_kind`]
/// (`_opimpl_residual_call1`), [`dispatch_residual_call_iIRd_kind`]
/// (`_opimpl_residual_call2`), [`dispatch_residual_call_iIRFd_kind`]
/// (`_opimpl_residual_call3`) — call [`select_residual_call_opcode`],
/// `record_op_with_descr`, then [`try_fold_pure_call_via_executor`] (the
/// cannot-raise pure fast path) + [`try_execute_residual_call_via_executor`]
/// (all other calls, including can-raise pure calls).  The executor
/// self-gates and degrades to recording-only when a precondition fails
/// (non-authoritative walk, non-const funcbox, unpatched symbolic fnaddr).
///
pub(crate) fn select_residual_call_opcode(
    ei: &majit_ir::EffectInfo,
    dst_bank: char,
    caller: &'static str,
) -> (OpCode, bool, bool) {
    // Release-gil sub-case is handled by `direct_call_release_gil`
    // before this selector runs.  Any `is_call_release_gil()` EI
    // reaching here is a dispatcher bug.
    debug_assert!(
        !ei.is_call_release_gil(),
        "{caller}: select_residual_call_opcode received an is_call_release_gil() EI; \
         dispatcher should have routed via direct_call_release_gil first"
    );
    let (call_op, pure_op, may_force_op, loopinvariant_op): (OpCode, OpCode, OpCode, OpCode) =
        match dst_bank {
            'r' => (
                OpCode::CallR,
                OpCode::CallPureR,
                OpCode::CallMayForceR,
                OpCode::CallLoopinvariantR,
            ),
            'i' => (
                OpCode::CallI,
                OpCode::CallPureI,
                OpCode::CallMayForceI,
                OpCode::CallLoopinvariantI,
            ),
            // `_irf_f/iIRFd>f` (`pyjitpl.py opimpl_residual_call_irf_f =
            // _opimpl_residual_call3`, `blackhole.py bhimpl_residual_call_irf_f`).
            // `resoperation.py Type::Float => CallF`. The `_r_f` /
            // `_ir_f` shapes do not exist upstream — the only float-result
            // residual_call variant routes through the `iIRFd` arglist.
            'f' => (
                OpCode::CallF,
                OpCode::CallPureF,
                OpCode::CallMayForceF,
                OpCode::CallLoopinvariantF,
            ),
            // `_*_v/iRd|iIRd|iIRFd` void variants (`pyjitpl.py
            // opimpl_residual_call_r_v = _opimpl_residual_call1`,
            // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
            // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`,
            // `blackhole.py bhimpl_residual_call_*_v`).
            // `resoperation.py Type::Void => CallN`. No dst writeback;
            // `write_residual_call_result_to_dst` no-ops on 'v'.
            'v' => (
                OpCode::CallN,
                OpCode::CallPureN,
                OpCode::CallMayForceN,
                OpCode::CallLoopinvariantN,
            ),
            _ => panic!("{caller}: unsupported dst_bank '{dst_bank}'"),
        };
    if ei.check_forces_virtual_or_virtualizable() {
        // pyjitpl.py forces-virtual-or-virtualizable branch
        // proper: CALL_MAY_FORCE_* + GUARD_NOT_FORCED.
        // `handle_possible_exception` also fires (forces always
        // satisfies check_can_raise).
        (may_force_op, ei.check_can_raise(false), true)
    } else if ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant {
        // pyjitpl.py EF_LOOPINVARIANT branch: CALL_LOOPINVARIANT_*
        // via miframe_execute_varargs(..., exc=False). LoopInvariant
        // never raises (extraeffect=1 < CannotRaise=2 → check_can_raise=False).
        //
        // The `pyjitpl.py call_loopinvariant_known_result` lookup
        // and `pyjitpl.py call_loopinvariant_now_known` cache
        // update are wired at the dispatcher level via
        // [`loopinvariant_lookup`] and [`loopinvariant_now_known`]
        // around the `record_op_with_descr` call — they require the
        // dispatcher's `descr_index` and `arg0_int` so this opcode
        // selector cannot perform them on its own.
        (loopinvariant_op, ei.check_can_raise(false), false)
    } else if ei.check_is_elidable() {
        // pyjitpl.py elidable branch: CALL_PURE_*.
        (pure_op, ei.check_can_raise(false), false)
    } else {
        // pyjitpl.py default branch: CALL_*.
        (call_op, ei.check_can_raise(false), false)
    }
}

/// `MIFrame.execute_varargs(pure=True)` parity for the
/// walker layer: when a residual_call routes to `CallPure*` (elidable +
/// cannot-raise EI per [`select_residual_call_opcode`]) AND every
/// argument in `allboxes` has a known concrete value
/// (`TraceCtx::box_value` returns `Some`), execute the helper at trace
/// time via [`majit_metainterp::executor::execute_pure_call`] and stamp
/// `recorded` with the result.
///
/// PyPy `_opimpl_*` methods (e.g. `_opimpl_setitem`,
/// `_opimpl_setfield_*`) concrete-execute every `do_residual_call`
/// regardless of `check_is_elidable()` — the EI flag only selects the
/// recorded opcode kind (`CALL_PURE_*` vs `CALL_*`), not whether the
/// helper runs.  This function covers the elidable arm; the
/// remaining arms (`CallMayForce*`, `CallLoopinvariant*`, `Call*`, and
/// can-raise `CallPure*`) are concrete-executed by
/// [`try_execute_residual_call_via_executor`]
/// with raised exceptions surfaced through `BH_LAST_EXC_VALUE` so
/// `eval_loop_jit` can route them into the bytecode exception handler.
///
/// RPython upstream `MIFrame.execute_varargs` invokes
/// `executor.execute_varargs(opnum, argboxes, descr, exc=False, pure=True)`
/// which dispatches to `cpu.bh_call_*` and stores the result on
/// `result_box.value` (`pyjitpl.py`).  Pyre's walker observes the
/// same effect through the `set_opref_concrete` stamp — downstream walker
/// chain (sub-jitcode bodies that consume the call result via
/// `concrete_of_opref`) folds end-to-end instead of stalling at
/// `RefOp/IntOp(N)` unknown values.
///
/// **Caller contract**:
/// * `call_opcode` must be one of `CallPureI`/`CallPureR`/`CallPureF`/
///   `CallPureN` — the `select_residual_call_opcode` elidable arm
///   (`pyjitpl.py` proper, `dispatch.rs`).  Other call
///   shapes (`CallMayForce*`, `CallLoopinvariant*`, `Call*`) carry
///   `can_raise=true` or escape semantics that require the full
///   `execute_varargs` MetaInterp seam — they MUST NOT route here.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// Having executed the helper, it continues into
/// `pyjitpl.py record_result_of_call_pure` as `MIFrame.execute_varargs`
/// does, and RETURNS the OpRef that carries the result afterwards: the
/// constant itself when every argbox was a `Const` (the call is cut back
/// out of the trace), and the re-recorded `CallPure*` otherwise, with
/// `call_pure_results` populated for the optimizer.  Callers must use the
/// returned OpRef rather than the one they recorded.
///
/// Best-effort: returns the caller's own OpRef unchanged when any operand
/// lacks a concrete `box_value` (the walker has no way to read the runtime
/// value), or when the arity exceeds `MAX_HOST_CALL_ARITY` (16) — the trace
/// still has the recorded `CallPure*` op for the optimizer to consume later,
/// just without the per-record fold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fold_pure_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    descr: majit_ir::DescrRef,
    patch_pos: majit_metainterp::recorder::TracePosition,
    recorded: OpRef,
) -> OpRef {
    if !matches!(
        call_opcode,
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
    ) {
        return recorded;
    }
    // `MIFrame.execute_varargs(pure=True)` only fires for
    // `EF_ELIDABLE_CANNOT_RAISE`. `select_residual_call_opcode` returns
    // `CallPure*` whenever `check_is_elidable()` is true (including
    // `EF_ELIDABLE_CAN_RAISE`), so re-check the can-raise predicate here
    // before dispatching through the `execute_pure_call` no-metainterp
    // carve-out.  A `EF_ELIDABLE_CAN_RAISE` callee would silently swallow
    // the exception via `BH_LAST_EXC_VALUE` with no metainterp to
    // transcribe it.
    let ei = call_descr.get_extra_info();
    if ei.check_can_raise(false) {
        return recorded;
    }
    if allboxes.is_empty() {
        return recorded;
    }
    // pyjitpl.py `_build_allboxes`: slot 0 is funcbox, slots
    // 1.. are user args in `descr.arg_types()` ABI order.  Walker's
    // [`build_allboxes`] preserves the same layout.
    //
    // `MIFrame.execute_varargs(pure=True)` invariant: it requires
    // `funcbox` to be a Const so its `getint()` is the actual fn
    // pointer.  Non-constant funcboxes carry a stale-stamped Int
    // (from `cast_ptr_to_int` of a Ref-bank receiver, etc.) and
    // dereferencing as a code address yields SIGSEGV.  Skip the fold
    // when the funcbox is non-constant; the recorded `CallPure*` op
    // stays in the trace for the optimizer to consume later.
    if !allboxes[0].is_constant() {
        return recorded;
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return recorded,
    };
    // `jitcode.py::JitCode.__init__` uses NULL for "no address".  It is no
    // more executable than an unpatched symbolic path hash; leave the call
    // unfolded so the main residual-call gate can abort/decline it without
    // jumping through address zero.
    if func_ptr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(func_ptr) {
        return recorded;
    }
    // Cap at MAX_HOST_CALL_ARITY (`call_int_function` / `call_void_function`
    // panic on excess arity).  `allboxes.len() - 1` is the arg count
    // (funcbox doesn't pass through).
    if allboxes.len() - 1 > majit_jitcode::codewriter::insns::MAX_HOST_CALL_ARITY {
        return recorded;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    // `pyjitpl.py:3572` keys `call_pure_results` on
    // `[executor.constant_from_op(a) for a in normargboxes]`, so the `Value`
    // form of every box — funcbox included — is collected here alongside the
    // ABI word the executor takes.
    let mut arg_values = Vec::with_capacity(allboxes.len());
    arg_values.push(majit_ir::Value::Int(func_ptr));
    for &arg in &allboxes[1..] {
        let Some(boxed) = ctx.trace_ctx.box_value(arg) else {
            return recorded;
        };
        arg_values.push(boxed);
        let v = match boxed {
            majit_ir::Value::Int(n) => n,
            majit_ir::Value::Ref(r) => {
                // `usize::MAX` sentinel from `concrete_of_opref` means
                // "no concrete known" — never reach this path because
                // `box_value` returns `None` for un-stamped OpRefs, but
                // belt-and-suspenders against future plumbing.
                if r == majit_ir::GcRef::NO_CONCRETE {
                    return recorded;
                }
                r.as_usize() as i64
            }
            majit_ir::Value::Float(f) => f.to_bits() as i64,
            majit_ir::Value::Void => 0,
        };
        args.push(v);
    }
    // Refuse to invoke the helper when any Ref argument is NULL.
    //
    // `pyjitpl.py record_result_of_call_pure` folds on the weaker
    // test "every argbox is a Const", which admits `ConstPtr(NULL)`.  It can
    // afford to: upstream reaches that line only *after* the call has already
    // been executed for real, and its Const arguments come from boxes the
    // interpreter itself made constant.
    //
    // Pyre's walker folds from a different source of constants.  Its
    // getfield_gc_r handler propagates field reads (including pointer-valued
    // fields like `PyFrame.f_backref`) as concrete values whenever the parent
    // struct is concrete-known, so a top-level frame stamps
    // `Value::Ref(GcRef(0))` onto the recorded `GetfieldGcR` OpRef
    // (`set_opref_concrete`), which `box_value` then hands straight to this
    // fold.  Upstream never derives that NULL in the first place: a pointer
    // becomes nonnull-known there only from the traced program's own null test
    // (`pyjitpl.py _establish_nullity`), so the box stays symbolic and
    // the call is never folded on a NULL it invented.  Executing
    // `helper(NULL)` here would dereference NULL and SEGV.
    //
    // So guard the executor entry against NULL Ref arguments and fall through
    // to recording the IR op as-is.  The downstream optimizer then sees the
    // call op and emits the necessary guards.
    for (i, &arg) in args.iter().enumerate() {
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return recorded;
        }
    }
    let result_i64 = majit_metainterp::executor::execute_pure_call(call_descr, func_ptr, &args);
    // pyjitpl.py `result_box.value = result`: stamp the recorded
    // OpRef with the executed concrete so downstream
    // `concrete_of_opref` / `box_value` consumers see the folded value.
    let result_value = match call_descr.result_type() {
        majit_ir::Type::Int => majit_ir::Value::Int(result_i64),
        majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
        majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
        // void callees discard the result upstream too (`bh_call_v` has
        // no return value); `CallPureN` is included in the matched set
        // only to mirror `MIFrame.execute_varargs(pure=True)`'s handling of all
        // pure shapes — skip the stamp for void.
        majit_ir::Type::Void => return recorded,
    };
    // `pyjitpl.py:1945` follows the executed pure call with
    // `record_result_of_call_pure`: an all-`Const` argument list cuts the call
    // back out of the trace and hands the result back as a constant, and any
    // other argument list populates `call_pure_results` so a later call with
    // the same arguments reuses the value.  Executing the helper without that
    // step leaves a `CallPure*` standing on constants, which is not what an
    // elidable leaf over an immutable block is supposed to record.
    let folded = ctx.trace_ctx.record_result_of_call_pure(
        recorded,
        allboxes,
        &arg_values,
        descr,
        patch_pos,
        call_opcode,
        result_value,
    );
    // Stamp only when the recorded result has a live slot in the active
    // recorder.  A deeper inlined / recursive frame's residual result may be
    // recorded in a context whose position is not allocated in the active
    // recorder; stamping it would violate the `*FrontendOp(pos, value)`
    // invariant.  Skipping leaves the result symbolic so the downstream branch
    // aborts the trace into the trait fallback instead of crashing.  A folded
    // constant carries its value inline and has nothing to stamp.
    if folded.inline_const_to_value().is_none() {
        ctx.trace_ctx.try_set_opref_concrete(folded, result_value);
    }
    folded
}

/// Whether Ref argument `arg_index` of a may-force CALL to `helper` is a
/// **checked** `PY_NULL` sentinel — a slot the helper tests before use, so a
/// concrete NULL there is the operand's normal shape rather than the broken
/// baked-NULL shape [`DispatchError::MayForceNullRefArgUnsupported`] exists to
/// catch.  `nargs` is the callee's argument count, for the trailing-slot entry.
///
/// One table, read by both `walker_abort_if_mayforce_null_ref_arg` (which
/// aborts the walk) and `try_execute_residual_call_via_executor` (which leaves
/// the call symbolic).  They have to agree: an exemption in only the gate lets
/// the walk continue past a call the executor then declines, and a
/// `declined_symbolic` drops the recording iteration's call exactly once
/// (`g(i, d=4)` in a hot loop summed to n-1, with the callee run n-1 times).
///
/// * `CallFn` / `CallKw` arg 1 — `null_or_self`: `bh_call_fn_impl` /
///   `bh_call_kw_N` prepend it as arg0 only when non-null, so NULL is the
///   ordinary plain-call shape.
/// * `CallFunctionEx` args 1 and 3 — `self_or_null` and `kwargs_or_null`: NULL
///   is the ordinary `f(*args)` / no-`**` shape.
/// * `StoreDeref` arg 1 — `value`: DELETE_DEREF lowers to
///   `store_deref_value(cell, none)` after its own bound check
///   (`codewriter.rs` `Instruction::DeleteDeref`) and the helper hands the NULL
///   straight to `w_cell_set` unread.  Declining it left the clear-the-cell
///   half of every `del <cellvar>` and every `except E as e` cleanup recorded
///   but UNEXECUTED, which marks the walk unjournaled — the walk-end flush then
///   declines and the caller replays the region on top of residuals the walk
///   already ran concretely.
/// * `WithExceptStart` arg 1 — `exit_self`: `with_except_start_values` prepends
///   it as arg0 only when non-null, and `LOAD_SPECIAL` leaves it NULL for every
///   already-bound `__exit__`, which is the common `with` shape.  Aborting on it
///   refuses every bridge out of a `with` handler.
/// * `LoadGlobal` arg 0 — `namespace_ptr`: the helper discards it and resolves
///   the namespace from the executing frame or the callee's own promoted
///   `w_code`, because an inlined / chained callee's frame register aliases an
///   outer frame; the operand survives only as the cell-fold recogniser's hint.
///   A nested function's `LOAD_GLOBAL` folds it to the NULL constant.
/// * `RaiseVarargs` trailing arg — `cause`: `normalize_raise_varargs` carries it
///   as the `raise X` (no `from`) sentinel, never dereferenced when null.
/// * a registered leaf the effect info gives no kind for — see
///   [`null_ref_sentinel_of_registered_leaf`].
pub(crate) fn mayforce_null_ref_arg_is_checked_sentinel(
    helper: majit_ir::RuntimeHelperKind,
    target: i64,
    arg_index: usize,
    nargs: usize,
) -> bool {
    use majit_ir::RuntimeHelperKind as K;
    match helper {
        K::CallFn | K::CallKw | K::StoreDeref | K::WithExceptStart => arg_index == 1,
        K::CallFunctionEx => arg_index == 1 || arg_index == 3,
        K::LoadGlobal => arg_index == 0,
        K::RaiseVarargs => arg_index + 1 == nargs,
        _ => null_ref_sentinel_of_registered_leaf(target, arg_index, nargs),
    }
}

/// Whether Ref argument `arg_index` of the registered leaf at `target` is a
/// checked `PY_NULL` sentinel.
///
/// A plain `dont_look_inside*` body has no `RuntimeHelperKind`, so the table
/// above cannot name it and the funcptr is the only coordinate the call
/// carries; resolve it against the published registry the way
/// `records_inside_transparent_helper` does.
fn null_ref_sentinel_of_registered_leaf(target: i64, arg_index: usize, nargs: usize) -> bool {
    if target <= 0 {
        return false;
    }
    pyre_interpreter::jit_trace_fnaddrs()
        .iter()
        .any(|(path, registered)| {
            *registered == target
                && path.rsplit("::").next().is_some_and(|leaf| {
                    let leaf = leaf.strip_prefix("__majit_call_target_").unwrap_or(leaf);
                    match leaf {
                        // `dict_get_plain_applies(dict, key, tail)` — `tail` is
                        // `PY_NULL` for the two-word `d.get(k)` that
                        // `__majit_wrap_dict_descr_get` pads to three words, and
                        // `tail.is_null()` is the first thing the body reads it
                        // with.  Declining it leaves the `bool` result unbound,
                        // and the wrapper's own `goto_if_not` on that result
                        // then aborts the walk with no complete image.
                        "dict_get_plain_applies" => arg_index == 2 && nargs == 3,
                        // `dict_get_slow(dict, key, default, nargs)` — the same
                        // padding reaches `default`, and a concrete NULL there
                        // is itself the proof the word is unread: the wrapper
                        // leaves it `PY_NULL` only for a call of one or two
                        // words and passes that same length as `nargs`, while
                        // every arm that moves `default` on is guarded by
                        // `nargs >= 3`.  Nothing dereferences it even then —
                        // it is copied into the slice `dict_method_get`
                        // rejects on arity.
                        "dict_get_slow" => arg_index == 2 && nargs == 4,
                        // `jit_portal_call_3(callable, null_or_self, a0, a1, a2)`
                        // — `bh_call_fn_impl`'s layout. `null_or_self` is
                        // `PY_NULL` for a plain call; the body tests
                        // `is_null()` and does not pass that word to
                        // `call_function_impl_result`. The codewriter leaves
                        // `runtime_helper` empty, so the `CallFn` row above
                        // does not see the slot.
                        "jit_portal_call_3" => arg_index == 1 && nargs == 5,
                        // `RootScope::pin_root(&self, root)` stores `root` on
                        // the shadow stack and normalizes the slot.
                        // `gc_current_object_address_in_window` returns a null
                        // word unchanged, so a null root is a real pin, not a
                        // deref. Declining it marks the walk unjournaled and
                        // the end flush keeps the legacy loop-entry replay,
                        // which runs the traced iteration's add a second time.
                        "pin_root" => arg_index == 1 && nargs == 2,
                        // `rlist.py ll_alloc_and_set(length, init)` writes
                        // `init` into every slot and does not dereference it.
                        // `vec![PY_NULL; n]` lowers to this helper; a NULL
                        // fill is the empty rest-args vector `W_CData.call`
                        // copies `args_w[1:]` into.
                        "ll_vec_alloc_and_set_r" => arg_index == 1 && nargs == 2,
                        _ => false,
                    }
                })
        })
}

/// Abort the walk when a result-bearing may-force CALL is recorded with a
/// concrete-NULL Ref argument — the specialized direct-call shape whose
/// baked `ptr(0x0)` (the `PUSH_NULL` self-slot) makes the runtime call pass
/// NULL where the callee entry expects its globals/closure, yielding a NULL
/// result (closures / locals-bound callees called in a loop). See
/// [`DispatchError::MayForceNullRefArgUnsupported`].
pub(crate) fn walker_abort_if_mayforce_null_ref_arg<Sym: WalkSym>(
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !matches!(
        call_opcode,
        OpCode::CallMayForceR | OpCode::CallMayForceI | OpCode::CallMayForceF
    ) {
        return Ok(());
    }
    // `allboxes[0]` is the funcbox; `allboxes[1 + i]` aligns with
    // `arg_types[i]` (see `build_allboxes`).  A Ref arg folded to the
    // NULL constant (`GcRef(0)`) is the broken self-slot; the sentinel
    // `GcRef(usize::MAX)` means "no concrete known" and is left alone.
    //
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    let helper = call_descr.get_extra_info().runtime_helper;
    let nargs = call_descr.arg_types().len();
    for (i, &ty) in call_descr.arg_types().iter().enumerate() {
        if ty != majit_ir::Type::Ref {
            continue;
        }
        let Some(&b) = allboxes.get(1 + i) else {
            continue;
        };
        if !matches!(
            ctx.trace_ctx.box_value(b),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ) {
            continue;
        }
        // The checked-sentinel table, shared with
        // `try_execute_residual_call_via_executor` so the two cannot drift.
        // Its funcbox coordinate names a leaf the effect info gives no
        // `RuntimeHelperKind` for, and is resolved on this arm alone.
        let target = match allboxes.first().and_then(|&f| ctx.trace_ctx.box_value(f)) {
            Some(majit_ir::Value::Int(addr)) => addr,
            _ => 0,
        };
        if mayforce_null_ref_arg_is_checked_sentinel(helper, target, i, nargs) {
            continue;
        }
        // Phase-1 diagnostic (gh#343 depth-2): pinpoint which Ref arg
        // folded to concrete NULL and its provenance.  Gated on
        // `PYRE_P2_DIAG` (the depth-2 framestack-walk diag flag) and
        // computed only on the abort path, so the default trace path
        // pays nothing.
        if p2_diag_enabled() {
            eprintln!(
                "[p2-mayforce] NULL Ref arg: pc={pc} call_opcode={call_opcode:?} \
                 helper={helper:?} arg_index={i} nargs={nargs} funcbox={:?}(={:?})",
                allboxes.first(),
                allboxes.first().and_then(|&f| ctx.trace_ctx.box_value(f)),
            );
            for (j, &aty) in call_descr.arg_types().iter().enumerate() {
                let ab = allboxes.get(1 + j).copied();
                eprintln!(
                    "[p2-mayforce]   arg[{j}] ty={aty:?} opref={ab:?} val={:?}",
                    ab.and_then(|b| ctx.trace_ctx.box_value(b)),
                );
            }
        }
        return Err(DispatchError::MayForceNullRefArgUnsupported { pc });
    }
    Ok(())
}

/// Diagnostic (`PYRE_FBW_DEBUG_ABORT`): dump the Python coordinate and per-arg
/// provenance behind a ValueUnavailable residual decline — the resume py_pc,
/// the decoded Python opcode, the declined arg's OpRef and whether it is a
/// constant, and its `box_value`.  Attributes an unj_val census walk to a
/// knowable-but-unpopulated value versus a genuinely-symbolic heap object
/// without re-instrumenting.
pub(crate) fn probe_resid_decline_ctx<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    why: &str,
    op_pc: usize,
    arg_index: usize,
    arg: OpRef,
    allboxes: &[OpRef],
) {
    let sym = ctx.fbw_mode.snapshot_sym;
    let (py_pc, opcode) = if !sym.is_null() {
        let s = unsafe { &*sym };
        if !s.jitcode().is_null() {
            let jc = unsafe { &*s.jitcode() };
            let code = (!jc.payload.code_ptr.is_null()).then(|| unsafe { &*jc.payload.code_ptr });
            let (pc, op) =
                crate::py_coord::py_op_name_for_jitcode_pc(&jc.payload.metadata, code, op_pc);
            (Some(pc), op)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    let box_v = ctx.trace_ctx.box_value(arg);
    let arg_id = if arg.is_constant() || arg.is_none() {
        "const".to_string()
    } else {
        format!("r{}", arg.raw())
    };
    // The declined arg's semantic register slot and its index-keyed concrete
    // shadow (`concrete_registers_r`): a `Null` shadow at a found slot with a
    // `None` box_value is the bridge-resume seed gap (neither store populated).
    let reg_slot = ctx.registers_r.iter().position(|r| r == arg);
    let shadow = reg_slot.and_then(|s| {
        ctx.frame_state
            .borrow()
            .concrete_registers_r
            .get(s)
            .copied()
    });
    eprintln!(
        "[fbw-resid-decline] {why} op_pc={op_pc} py_pc={py_pc:?} py_op={opcode:?} \
         arg_index={arg_index} arg={arg_id} box_value={box_v:?} reg_slot={reg_slot:?} \
         reg_shadow={shadow:?} nargs={}",
        allboxes.len() - 1,
    );
}

/// Resolve a Ref register through the rebuilt carrier frame's semantic
/// operand-stack slot. RPython's MIFrame stores the Box at that slot; the flat
/// post-regalloc color is only an encoding detail and can name a freshly
/// recorded load whose runtime value was lost after heapcache invalidation.
fn carrier_stack_box_for_ref_arg<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
    arg: OpRef,
) -> Option<OpRef> {
    if !ctx.fbw_mode.carrier_resume || !ctx.vstack_valid {
        return None;
    }
    let consts = ctx.inline_callee_consts?;
    let color = ctx.registers_r.iter().position(|value| value == arg)?;
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(ctx.inline_w_code() as pyre_object::PyObjectRef)
    } as *const pyre_interpreter::CodeObject;
    if raw_code.is_null() {
        return None;
    }
    let nlocals = unsafe { (&(*raw_code).varnames).len() };
    let maps =
        crate::state::bridge_semantic_maps_from_jitcode_pc(consts.jitcode_index, op_pc as i32);
    let semantic = crate::state::semantic_ref_slot_for_reg_color(
        nlocals,
        maps.stack_depth_at_pc,
        &maps.pcdep_entries,
        color,
    )?;
    let stack_slot = semantic.checked_sub(nlocals)?;
    ctx.frame_state
        .borrow()
        .vstack_boxes
        .get(stack_slot)
        .copied()
        .filter(|&value| value != OpRef::NONE)
}

/// Make a carrier-resumed `CallFn` consume the Box objects owned by its
/// reconstructed MIFrame, as `pyjitpl.py do_residual_call` does. Prefer the
/// semantic color-to-stack mapping; for a post-regalloc color whose mapping is
/// stale, use the CALL operand suffix only when the encoded box has lost its
/// concrete value and the frame box still has one. Keeping an already-concrete
/// encoded box preserves its SSA identity for the optimizer.
fn repair_carrier_call_ref_args<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    r_args: &mut [OpRef],
) {
    if !ctx.fbw_mode.carrier_resume || helper != majit_ir::RuntimeHelperKind::CallFn {
        return;
    }
    for arg in r_args.iter_mut() {
        if let Some(frame_box) = carrier_stack_box_for_ref_arg(ctx, op_pc, *arg) {
            *arg = frame_box;
        }
    }
    if !ctx.vstack_valid || ctx.frame_state.borrow().vstack_boxes.len() < r_args.len() {
        return;
    }
    let start = ctx.frame_state.borrow().vstack_boxes.len() - r_args.len();
    for (arg, &frame_box) in r_args
        .iter_mut()
        .zip(&ctx.frame_state.borrow().vstack_boxes[start..])
    {
        if ctx.trace_ctx.concrete_of_opref(*arg).is_none()
            && frame_box != OpRef::NONE
            && ctx.trace_ctx.concrete_of_opref(frame_box).is_some()
        {
            *arg = frame_box;
        }
    }
}

/// Whether a residual call is a self-recursive call to the walk's own code —
/// the `CALL_ASSEMBLER` fold target running as a plain residual because the
/// fold declined (no compiled token yet, a non-concrete argument during a
/// bridge resume, etc.).  Mirrors the callee/self resolution in
/// `try_walker_call_assembler_self_recursive`.  Keeps the recursion itself out
/// of the foreign-body-residual latch so pure recursion (`fib`) still folds.
pub(crate) fn residual_callee_is_walk_self_recursive<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    helper: majit_ir::RuntimeHelperKind,
) -> bool {
    if helper != majit_ir::RuntimeHelperKind::CallFn {
        return false;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return false;
    }
    // A `bh_call_fn` residual is `[funcptr, callable, null_or_self, arg0, ...]`;
    // the Python callable is `allboxes[1]`.
    let Some(&callable_box) = allboxes.get(1) else {
        return false;
    };
    let Some(majit_ir::Value::Ref(callable_ref)) = ctx.trace_ctx.box_value(callable_box) else {
        return false;
    };
    if callable_ref == majit_ir::GcRef::NO_CONCRETE || callable_ref.as_usize() == 0 {
        return false;
    }
    let callable = callable_ref.as_usize() as pyre_object::PyObjectRef;
    unsafe {
        let Some((w_code, _nparams, _has_closure)) = resolve_inlinable_callee(callable) else {
            return false;
        };
        let sym = &*sym_ptr;
        if sym.jitcode().is_null() {
            return false;
        }
        let caller_code =
            pyre_interpreter::live_code_wrapper((*sym.jitcode()).raw_code() as *const ())
                as *const ();
        w_code as usize == caller_code as usize
    }
}

/// Whether the callable's frame needs cells. The self-rec CALL_ASSEMBLER
/// fold is Branch A (`ncells == 0`); a closure or cell-bearing body
/// declines there and must not be inlined past `execute_frame` either.
fn residual_callable_has_closure_or_cells<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    callable: OpRef,
) -> bool {
    let Some(majit_ir::Value::Ref(callable_ref)) = ctx.trace_ctx.box_value(callable) else {
        return false;
    };
    if callable_ref == majit_ir::GcRef::NO_CONCRETE || callable_ref.as_usize() == 0 {
        return false;
    }
    let obj = callable_ref.as_usize() as pyre_object::PyObjectRef;
    unsafe {
        let Some((w_code, _nparams, has_closure)) = resolve_inlinable_callee(obj) else {
            return false;
        };
        if has_closure {
            return true;
        }
        let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject;
        !raw.is_null() && pyre_interpreter::ncells(&*raw) != 0
    }
}

/// `dont_look_inside_cannot_raise` helpers whose descr stays
/// `EF_RANDOM_EFFECTS`. Recording the call inside a transparent helper
/// is sound: they do not call Python, so `GUARD_NOT_FORCED` does not fail.
/// `module_spec_get_initializing` answers `-1` instead of running `__eq__`,
/// `__get__`, or `__bool__`. `bootstrap_handle_fromlist` and
/// `default_importlib_import_word` are the same kind of lookup: a miss is
/// null, and neither calls `_handle_fromlist`.
fn records_inside_transparent_helper(addr: i64) -> bool {
    if addr <= 0 {
        return false;
    }
    pyre_interpreter::jit_trace_fnaddrs()
        .iter()
        .any(|(path, registered)| *registered == addr && transparent_helper_recordable_leaf(path))
}

fn transparent_helper_recordable_leaf(path: &str) -> bool {
    path.rsplit("::").next().is_some_and(|leaf| {
        let leaf = leaf.strip_prefix("__majit_call_target_").unwrap_or(leaf);
        matches!(
            leaf,
            "tuple_from_exact_list"
                | "dict_get_plain"
                | "dict_get_plain_applies"
                | "exc_init_one_positional"
                | "value_error_one_arg"
                | "extend_from_frame_locals_proxy"
                | "proxy_list_new"
                | "proxy_list_append"
                | "append_extra_locals"
                | "append_extra_locals_items"
                | "module_spec_get_initializing"
                | "bootstrap_handle_fromlist"
                | "default_importlib_import_word"
        )
    })
}

/// `jit_portal_call_3` is `space.call_method` (`baseobjspace.py`) as five
/// words. The codewriter leaves `runtime_helper` empty; the inliner enters
/// only for `CallFn`. Other helpers keep their own tag, so a `store_subscr`
/// residual is not read as a Python call.
fn portal_user_call_helper<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
    helper: majit_ir::RuntimeHelperKind,
) -> majit_ir::RuntimeHelperKind {
    if helper != majit_ir::RuntimeHelperKind::None {
        return helper;
    }
    let Some(majit_ir::Value::Int(addr)) = ctx.trace_ctx.box_value(funcptr) else {
        return helper;
    };
    if pyre_interpreter::importing::residual_addr_is_jit_portal_call_3(addr) {
        majit_ir::RuntimeHelperKind::CallFn
    } else {
        helper
    }
}

/// Funcptr word of a residual that is NULL or a `symbolic_fnaddr` hash.
/// Other `ResidualDecline::Symbolic` causes (non-const funcbox, LoadConst)
/// stay recorded: those calls are real at runtime.
fn recorded_symbolic_funcptr<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
) -> Option<i64> {
    let funcbox = allboxes.first()?;
    if !funcbox.is_constant() {
        return None;
    }
    match ctx.trace_ctx.box_value(*funcbox) {
        Some(majit_ir::Value::Int(addr))
            if addr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(addr) =>
        {
            Some(addr)
        }
        _ => None,
    }
}

/// The symbolic decline, minted in one place so it carries provenance.
///
/// Ten preconditions inside [`try_execute_residual_call_via_executor`] end in
/// this same value, and the walk only learns that ONE of them fired.  That is
/// enough while the decline is benign, but a decline taken after the walk has
/// already executed an unrecoverable effect leaves the walk with no sound
/// walk-end road, and then which precondition fired is the question.  Record
/// the caller and the opcode it refused for `PYRE_UNJOURNALED_SITE` to print.
#[track_caller]
fn declined_symbolic(call_opcode: OpCode) -> ResidualExecOutcome {
    // Read the caller OUTSIDE the closure: a closure body is not
    // `#[track_caller]`, so `Location::caller()` inside one names the closure.
    let site = std::panic::Location::caller();
    DECLINED_SYMBOLIC_SITE.with(|c| c.set(Some((site, call_opcode))));
    ResidualExecOutcome::Declined(ResidualDecline::Symbolic)
}

thread_local! {
    static DECLINED_SYMBOLIC_SITE: std::cell::Cell<
        Option<(&'static std::panic::Location<'static>, OpCode)>,
    > = const { std::cell::Cell::new(None) };
}

/// Where the last symbolic decline of this thread was taken, for diagnostics.
pub(crate) fn last_declined_symbolic_site()
-> Option<(&'static std::panic::Location<'static>, OpCode)> {
    DECLINED_SYMBOLIC_SITE.with(|c| c.get())
}

/// Helper kinds whose residual mutates live heap while RETURNING a value, so
/// the `Void`-result write proxy the same discriminator opens with cannot see
/// them.  A kind missing from here is a store the executed-effect odometer does
/// not count, and a nested abort behind it rewinds and runs it twice.
pub(crate) fn helper_kind_writes_live_heap(helper: majit_ir::RuntimeHelperKind) -> bool {
    matches!(
        helper,
        majit_ir::RuntimeHelperKind::StoreSubscr
            | majit_ir::RuntimeHelperKind::SetCurrentException
            | majit_ir::RuntimeHelperKind::StoreDeref
            | majit_ir::RuntimeHelperKind::SetAddMethod
            | majit_ir::RuntimeHelperKind::BareSuperFromFrame
    )
}

/// The standard virtualizable word `try_execute_residual_call_via_executor`
/// saves across a non-forcing residual, rooted for the duration of the call.
struct SavedVableRoot {
    slot: Box<i64>,
    depth: Option<usize>,
}

impl SavedVableRoot {
    fn push(ptr: Option<*const u8>) -> Self {
        let mut root = Self {
            slot: Box::new(ptr.map_or(0, |p| p as usize as i64)),
            depth: None,
        };
        if ptr.is_some() {
            root.depth = Some(majit_gc::shadow_stack::resume_ref_roots_depth());
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(
                    &mut *root.slot,
                ));
            }
        }
        root
    }

    fn forwarded(&self) -> Option<*const u8> {
        self.depth.map(|_| *self.slot as usize as *const u8)
    }
}

impl Drop for SavedVableRoot {
    fn drop(&mut self) {
        if let Some(depth) = self.depth {
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
        }
    }
}

pub(crate) fn try_execute_residual_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
    op_pc: usize,
    blackhole_result: Option<(usize, char, usize)>,
    // `Some` from `journal_walker_namespace_write` for this residual.
    // A namespace helper whose journal declined stays a body effect.
    namespace_write_journaled: bool,
) -> Result<ResidualExecOutcome, DispatchError> {
    // `execute_varargs` clears the metainterp exception slot at residual-call
    // entry, before the helper can either run or leave the call recorded
    // symbolically. A handled exception from an earlier opcode must not survive
    // across the back edge and make a later linear `catch_exception/L` look like
    // it is handling a fresh raise.
    clear_walk_exception(ctx);

    // Orthodox sub-jitcode walk safety (#171 wall-5d): a residual call whose
    // funcbox is NULL or a `symbolic_fnaddr` placeholder — a 64-bit `DefaultHasher`
    // hash of an in-body helper's `CallPath`/`CallTarget`, minted when
    // `jit_trace_fnaddrs()` has no entry for it (e.g. the zero-arg
    // `SyntheticTransparentCtor "Tuple"` unit constructor inside
    // `w_list_append`) — must not be recorded while inlining a sub-jitcode
    // body.  The production fall-throughs below leave such a call symbolic when
    // folding declines, on the contract that it runs at runtime against live
    // state; but a sub-walk's recorded trace is committed and compiled, so the
    // backend bakes the hash as a code address and the trace branches straight
    // to it -> SIGSEGV.  Decline the whole descent so it aborts gracefully at
    // the first un-lowered helper.  Symbolic hashes carry the
    // `SYMBOLIC_FNADDR_BASE` high-16-bit tag no real funcptr can carry.
    //
    // A published address is the weaker claim that the funcptr is real, not
    // that its signature fits the residual ABI.  The executor writes one word
    // per `arg_types()` entry, so a helper taking a fat pointer receives its
    // second half from whichever register the caller last wrote, and reading
    // that as a length dereferences an address nothing chose --- a fault
    // inside the callee, past every check this walk makes.
    // `is_abi_unsound_argument_residual` names exactly the helpers published
    // with such a parameter, so the descent declines before running one and
    // the interpreter makes the call with the argument whole.
    //
    // A may-force residual inside a transparent helper is declined for the
    // resume it would need.  Its GUARD_NOT_FORCED and GUARD_NO_EXCEPTION fail
    // after the Python code it called back into has run: a callback that
    // raises forces the caller's frame vref on `leave`, and so does one that
    // escapes its frame.  `capture_resumedata` resumes such a guard inside the
    // helper's own frame, past the call.  A transparent helper has no frame of
    // its own, so its guards resume at the Python CALL that entered it, and
    // the CALL runs the callback a second time -- or, with the exception still
    // pending at that coordinate, lets it past the CALL's handler.
    let unsupported_funcbox = if ctx.fbw_mode.inline_subwalk
        && allboxes.first().is_some_and(|b| b.is_constant())
        && let Some(majit_ir::Value::Int(addr)) = ctx.trace_ctx.box_value(allboxes[0])
    {
        if addr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(addr) {
            // `symbolic` resolves against `symbolic_fnaddr_paths` in the
            // build's jit metadata, so carry the hash for the report.
            Some(("symbolic", addr, addr))
        } else if pyre_interpreter::is_abi_unsound_argument_residual(addr as usize) {
            // A real funcptr reached here, so there is no hash to carry, and
            // zero is what this variant spells for that.
            Some(("abi-unsound-arg", addr, 0))
        } else if ctx.fbw_mode.transparent_helper_subwalk
            && call_opcode.is_call_may_force()
            && !records_inside_transparent_helper(addr)
            && !pyre_interpreter::is_rewindable_root_bracket_residual(addr as usize)
            && !call_descr.get_extra_info().is_call_release_gil()
            // `pyjitpl.py direct_libffi_call` executes `jit_ffi_call_impl_*`
            // and records `call_release_gil`. The helper is
            // `dont_look_inside` and does not re-enter Python; the forces
            // guard belongs to the recorded release-gil op.
            && call_descr.get_extra_info().oopspecindex != majit_ir::OopSpecIndex::LibffiCall
        {
            // These helpers are `dont_look_inside_cannot_raise`. Their
            // call descr is still `EF_RANDOM_EFFECTS` because the graph
            // is top, so the opcode is may-force. They do not call back
            // into Python, so the guard the comment above is about does
            // not fail and the call can be recorded.
            // `pyjitpl.py` `do_residual_call` still runs
            // `execute_and_record_varargs` on `call_external_function`
            // (`ccall_*`). `direct_call_release_gil` only rewrites the
            // recorded op. The wrapper releases the GIL around the C
            // call, so it does not re-enter Python and `GUARD_NOT_FORCED`
            // cannot fail for a callback.
            Some(("may-force-in-transparent-helper", addr, 0))
        } else {
            None
        }
    } else {
        None
    };
    if let Some((reason, addr, symbolic)) = unsupported_funcbox {
        // Every caller catches this variant to decline its whole fold, so
        // without a report here the helper that stopped the descent is
        // unnameable.
        if fbw_debug_abort_enabled() {
            let path = majit_metainterp::resolve_symbolic_fnaddr_path(addr);
            eprintln!("[subwalk-unsupported] pc={op_pc} {reason}={addr:#x} path={path:?}");
        }
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc: op_pc,
            symbolic,
        });
    }
    // Authoritative-executor gate: fire ONLY when the walk is the sole
    // concrete-execution leg (the production full-body walk and its
    // inline sub-walks; the per-opcode arm walk is retired).  Shadow /
    // diagnostic-probe runs leave the flag `false` so the call is
    // recorded symbolically without re-running its side effects.
    if !ctx.is_authoritative_executor {
        return Ok(declined_symbolic(call_opcode));
    }
    let plain_or_loopinvariant = matches!(
        call_opcode,
        OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
    );
    // `pyjitpl.py do_residual_call` forces branch: every
    // `CallMayForce*` is concrete-executed, with the active
    // virtualizable bracketed by the token protocol (set
    // TOKEN_TRACING_RESCALL before the call, probe-and-clear after —
    // see the doc bullet above).
    let is_may_force = matches!(
        call_opcode,
        OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
    );
    // `MIFrame.execute_varargs` executes an elidable call BEFORE it patches the
    // recorded CALL into CALL_PURE and before `handle_possible_exception`
    // (`pyjitpl.py`).  The cannot-raise half is already executed by
    // `try_fold_pure_call_via_executor`; the can-raise half deliberately
    // reaches this full executor so `BH_LAST_EXC_VALUE` is transcribed into the
    // walker's exception state instead of being swallowed.
    let is_pure = matches!(
        call_opcode,
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
    );
    if is_pure && !call_descr.get_extra_info().check_can_raise(false) {
        // The cannot-raise fast path ran first.  If it could not answer because
        // an operand was symbolic, keeping CALL_PURE in the trace carries no
        // missed effect or exception obligation.
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::PureUnfolded));
    }
    if !plain_or_loopinvariant && !is_may_force && !is_pure {
        return Ok(declined_symbolic(call_opcode));
    }
    if allboxes.is_empty() {
        return Ok(declined_symbolic(call_opcode));
    }
    // Same funcbox-must-be-const invariant as `try_fold_pure_call_via_executor`:
    // a non-const funcbox carries a stale stamp and dereferencing it as a
    // code address SEGVs.  pyjitpl.py forces the funcbox through
    // the executor's `cpu.bh_call_*` `ConstInt.getint()` path which
    // implicitly requires constness too (residual_call descrs always
    // carry a fixed funcptr at translation time).
    if !allboxes[0].is_constant() {
        return Ok(declined_symbolic(call_opcode));
    }
    // The LOAD_CONST helper (oopspec `LoadConst`) has a dedicated fold in the
    // residual_call dispatchers: when the const index AND the code pointer
    // (`frame.pycode`) are both concrete, it materializes `co_consts[idx]`
    // directly and suppresses the residual.  When that fold declines — the
    // promoted `frame.pycode` is concrete for the portal frame but an inlined
    // callee sub-walk does not seed it — the residual is recorded so the
    // loop computes it at runtime from the live frame's real `pycode`.
    // Executing it concretely here would pass the unseeded (null/garbage)
    // code pointer to `bh_load_const_fn`, which dereferences it via
    // `w_code_get_ptr` and faults.  Leave it symbolic, mirroring the fold's
    // "falls through to the generic record" contract.
    if call_descr.get_extra_info().runtime_helper == majit_ir::RuntimeHelperKind::LoadConst {
        return Ok(declined_symbolic(call_opcode));
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return Ok(declined_symbolic(call_opcode)),
    };
    // Safety gate — reject NULL (upstream's "no address" spelling in
    // `jitcode.py::JitCode.__init__`) and `symbolic_fnaddr_for_path`
    // placeholder values that escaped runtime patching.  Pyre's
    // codewriter mints a 64-bit hash of the helper's `CallPath` when
    // the build-time `pyre_interpreter::jit_trace_fnaddrs()` snapshot
    // has no entry for it (`symbolic_fnaddr_for_path` in
    // `majit-translate/src/codewriter/call.rs`).  `runtime_fnaddr_patch` rewrites
    // these to real runtime addresses only when the path appears in
    // both the build-time and runtime registries; helpers absent from
    // the runtime registry retain the hash and dereferencing it as a
    // code address SIGBUSes.  Hashes carry the `SYMBOLIC_FNADDR_BASE`
    // high-16-bit tag no real funcptr can carry on any target (a bit-47
    // range test would misclassify every real funcptr on aarch64 Linux,
    // whose 48-bit VA maps code at 0xaaab…/0xffff…).
    if func_ptr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(func_ptr) {
        return Ok(declined_symbolic(call_opcode));
    }
    // Same unsound-argument set the inline-subwalk gate consults above.
    // Top-level FBW (bridge retrace after a compiled-loop guard) used to
    // skip it and execute the helper with one word per `arg_types()` slot;
    // a fat-pointer parameter then reads the next register as a length and
    // memmoves from a wild address (`exception_reused_object_tb_not_doubled`
    // cranelift SIGSEGV in `try_execute_residual_call_via_executor`).
    if pyre_interpreter::is_abi_unsound_argument_residual(func_ptr as usize) {
        // `declined_symbolic` skips only the tracing-time invocation and
        // leaves the residual CALL in the trace. The backend then supplies
        // one word per `arg_types()` entry to a helper that expects a
        // multiword parameter. Abort the enclosing walk the same way the
        // inline-subwalk gate above does.
        if fbw_debug_abort_enabled() {
            eprintln!("[subwalk-unsupported] pc={op_pc} abi-unsound-arg={func_ptr:#x}");
        }
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc: op_pc,
            symbolic: 0,
        });
    }
    // A residual whose funcptr is a `PyFrame` operand-stack accessor
    // (`pop`/`push`/`peek`/`peek_at`) reads or mutates the live frame's
    // operand stack.  During a walk that stack is empty — the walk holds
    // operand values symbolically in its register banks, not on the real
    // frame (the portal lowers stack ops to vable array writes; these
    // accessors appear only inside inlined callee sub-jitcode bodies such as
    // `pop_value`).  Executing one here underflows `PyFrame::pop`'s
    // `valuestackdepth > stack_base()` assertion against the paused outer
    // frame.  Record it symbolically instead, mirroring the tracer's
    // never-mutate-the-traced-frame discipline; it runs at runtime against a
    // frame whose operand stack the compiled trace's preceding pushes have
    // populated.
    if pyre_interpreter::is_pyframe_operand_stack_accessor(func_ptr as usize) {
        return Ok(declined_symbolic(call_opcode));
    }
    // The `list_write_barrier` residual (the #171 object-append fold's Object
    // arm leaves it as a residual because it is `#[dont_look_inside]`) is pure
    // idempotent GC bookkeeping — re-running it on a body replay only re-adds
    // the list to the remembered set, never doubling user-visible state.  It
    // must still EXECUTE concretely below — the walk mutates the live heap, so
    // the store it guards has really happened — but it is not a
    // body effect: keep it out of the in-flight-FOR_ITER body-effect accounting
    // so an Object-strategy comprehension append (`[(i, i) for …]`, `[None …]`)
    // is not refuse-dropped.  RPython treats the write barrier the same way —
    // `COND_CALL_GC_WB` is never executed by pyjitpl
    // (`rpython/jit/metainterp/executor.py:446`), is neither can-raise nor a
    // call (`resoperation.py:1124-1125`), and is inserted only by the backend
    // GC rewrite pass after optimization (`backend/llsupport/rewrite.py`),
    // so it never participates in the metainterp's side-effect analysis.
    let is_idempotent_gc_barrier = pyre_interpreter::is_list_write_barrier(func_ptr as usize);
    // The void bookkeeping residuals a descent into a translated body meets
    // before it reaches any real work: a stack check and a `OnceLock` lazy
    // init, both of which reach the same state when re-run.  Joined into
    // `provably_side_effect_free` below rather than repeated at each
    // accounting, so every consumer of that predicate — the nested sub-walk
    // decline included — reads one answer.  Before a callee body was traced
    // through, these ran inside one opaque residual that the caller's own
    // recognisers could exempt whole; tracing splits them out, and each split
    // piece has to carry the exemption the whole used to.
    let is_rerunnable_bookkeeping =
        pyre_interpreter::is_rerunnable_bookkeeping_residual(func_ptr as usize);
    // Same re-runnability question as `stack_check`: `dont_look_inside`
    // (`rlib/jit.py dont_look_inside`) residualises the call but does not
    // license the optimizer to drop it (`elidable`).  A rewind re-pins from
    // the save point it finds, so the odometer must not count the bracket
    // the way the descent scan already does not.
    let is_rewindable_root_bracket =
        pyre_interpreter::is_rewindable_root_bracket_residual(func_ptr as usize);
    if allboxes.len() - 1 > majit_jitcode::codewriter::insns::MAX_HOST_CALL_ARITY {
        return Ok(declined_symbolic(call_opcode));
    }
    // A void residual (CALL_N family) is a side effect with no result box, so
    // `do_residual_call` executes it EAGERLY during the walk and resumes the
    // compiled loop at iteration N+1 (the commit-invariant note below) — a
    // deferred void store is lost (its symbolic op fires only for N+1+).  When a
    // void call's arg cannot be resolved to a concrete (`GcRef(usize::MAX)` =
    // "no concrete known", or `None` = unbound), eager execution is impossible
    // AND deferral drops the store, so neither path is correct: abort the trace
    // gracefully (interpreter fallback) rather than silently drop it.  This is
    // the off-by-one for a module-global loop that builds and stores a heap
    // object then reads it back (`g = [n]; ... g[0]`): the `STORE_NAME` is a
    // void `CallN` whose value is the still-virtual `BUILD_LIST` result, so the
    // iteration-N store never reaches the cell.  A value-returning call with a
    // non-concrete arg is safe to leave symbolic — the compiled loop computes
    // its result at runtime with no lost side effect.
    let is_void = matches!(
        call_opcode,
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallLoopinvariantN
    );
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for (arg_index, &arg) in allboxes[1..].iter().enumerate() {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef::NO_CONCRETE {
                    if is_void {
                        return Err(DispatchError::ResidualCallArgUnbound {
                            pc: op_pc,
                            arg_index,
                        });
                    }
                    if fbw_debug_abort_enabled() {
                        probe_resid_decline_ctx(
                            ctx,
                            "NO_CONCRETE",
                            op_pc,
                            arg_index,
                            arg,
                            allboxes,
                        );
                    }
                    return Ok(ResidualExecOutcome::Declined(
                        ResidualDecline::ValueUnavailable,
                    ));
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => {
                if is_void {
                    return Err(DispatchError::ResidualCallArgUnbound {
                        pc: op_pc,
                        arg_index,
                    });
                }
                if fbw_debug_abort_enabled() {
                    probe_resid_decline_ctx(ctx, "box_value=None", op_pc, arg_index, arg, allboxes);
                }
                return Ok(ResidualExecOutcome::Declined(
                    ResidualDecline::ValueUnavailable,
                ));
            }
        };
        args.push(v);
    }
    // NULL-Ref-arg refusal: same SEGV-avoidance contract as the pure
    // fast path (see `try_fold_pure_call_via_executor`'s NULL guard).  Pyre's
    // optimizer emits `guard_nonnull` after this walker fold, so a NULL
    // receiver dereferences before that guard exists; fall through to
    // recording the call op and let the optimizer's guard emission
    // handle it at compile time.
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    // `null_or_self` (arg index 1) is a checked sentinel — `PY_NULL`
    // means "no receiver" and is never dereferenced (`bh_call_fn_impl`
    // prepends it as arg0 only when non-null), so a concrete-NULL there
    // is the normal plain-call shape.  The exemptions are read out of the
    // same `mayforce_null_ref_arg_is_checked_sentinel` table
    // `walker_abort_if_mayforce_null_ref_arg` consults, because they MUST
    // agree — a gate-only exemption leaves a normal no-receiver
    // keyword/star call declined as symbolic here, which drops the
    // recording iteration's call exactly once (`g(i, d=4)` in a hot loop
    // summed to n-1, callee ran n-1 times).
    let helper = call_descr.get_extra_info().runtime_helper;
    // `bh_load_method_self_fn` dereferences `attr` (`is_method`) before it
    // looks at `obj`. A null or unaligned word is not a PyObject — a
    // bit-cast of 1 reaches this executor as an Int box the Ref null-check
    // below does not see. Decline; the compiled call runs with the real
    // operands.
    if helper == majit_ir::RuntimeHelperKind::LoadMethodSelf {
        let object_arg_unusable = args.iter().take(2).any(|&arg| {
            let bits = arg as usize;
            bits == 0 || bits % std::mem::align_of::<usize>() != 0
        });
        if object_arg_unusable {
            return Ok(declined_symbolic(call_opcode));
        }
    }
    for (i, &arg) in args.iter().enumerate() {
        if arg != 0 || !matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) {
            continue;
        }
        // The funcbox coordinate the sentinel table needs to name a leaf the
        // effect info gives no `RuntimeHelperKind` for, resolved on this arm
        // alone so the ordinary residual path never reads the registry.
        let target = match allboxes.first().and_then(|&b| ctx.trace_ctx.box_value(b)) {
            Some(majit_ir::Value::Int(addr)) => addr,
            _ => 0,
        };
        if mayforce_null_ref_arg_is_checked_sentinel(helper, target, i, args.len()) {
            continue;
        }
        // The refusal names the helper and slot it fired on, because the
        // repair for a helper that does check its NULL is a row in
        // `mayforce_null_ref_arg_is_checked_sentinel` and the row needs
        // both coordinates.
        if fbw_debug_abort_enabled() {
            let name = pyre_interpreter::jit_trace_fnaddrs()
                .into_iter()
                .find(|&(_, addr)| addr == target)
                .map_or("-", |(name, _)| name);
            eprintln!(
                "[nullref-refusal] helper={helper:?} target={name}/{target:#x} \
                 arg_index={i} nargs={} pc={op_pc} opcode={call_opcode:?}",
                args.len()
            );
        }
        return Ok(declined_symbolic(call_opcode));
    }
    // #57 (Finding #1, in-place container mutation): an in-flight FOR_ITER
    // body's `acc += delta` is a bare `NB_INPLACE_*` `BinaryOp` residual (args
    // = [lhs, rhs, op_code]) that may mutate its receiver in place at the C
    // level — no Void result, no write tag, no user frame — so none of the
    // body-effect signals below see it.  A committed non-journaled in-place
    // mutation that an aborting walk delivers would be re-run (double); dropped,
    // it would lose the iteration's tail.  Two recoverable shapes are handled
    // here, decided BEFORE any vable/tracing-call state is set up so an early
    // decline strands nothing:
    //
    //  * `acc += [ints]` for two Integer-strategy lists — the extend keeps `acc`
    //    Integer-strategy, so `w_list_int_set_len` can rewind it.  Capture the
    //    pre-extend length; the success arm journals it so the abort rollback
    //    undoes the one extend and the deliver re-applies it exactly once.
    //  * an immutable receiver (`int`/`bool`/`float`/`tuple`/`str`/`bytes`) —
    //    `+=` yields a FRESH object and rebinds the journaled local, so a plain
    //    deliver re-run is exact with no journaling.
    //
    // Any OTHER *exact builtin* receiver — an object-/float-strategy list,
    // `bytearray`, `set`, `dict`, `array`, a mixed `int-list += non-ints` that
    // would change strategy, … — may commit a mutation the rollback cannot
    // rewind, so decline the walk here and let this loop run interpreted
    // (exact), like the gate refusing an unsupported body op.  A user instance
    // must not take that decline: its `__iadd__`/`__isub__`/… has not run yet,
    // while the in-flight `FOR_ITER` item has already been consumed.  The
    // permanent-abort path then drops that item (the conservative delivery
    // gate sees the loop's preceding `STORE_NAME`), silently skipping one
    // augmented-assignment iteration at the trace-entry boundary.  Let the
    // normal residual dispatch execute user special methods; its existing
    // user-frame effect accounting handles any later abort.
    let inplace_list_journal: Option<(pyre_object::PyObjectRef, usize, isize)> =
        if call_descr.get_extra_info().runtime_helper == majit_ir::RuntimeHelperKind::BinaryOp
            && args.len() >= 3
            && pyre_interpreter::runtime_ops::binary_op_tag_is_inplace(args[2])
            && fbw_foriter_inflight_active()
        {
            let lhs = args[0] as usize as pyre_object::PyObjectRef;
            let rhs = args[1] as usize as pyre_object::PyObjectRef;
            unsafe {
                if pyre_object::pyobject::is_exact_list(lhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(lhs)
                    && pyre_object::pyobject::is_exact_list(rhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(rhs)
                {
                    Some((
                        lhs,
                        pyre_object::w_list_len(lhs),
                        pyre_object::listobject::w_list_allocated(lhs),
                    ))
                } else if pyre_object::pyobject::is_int_or_long(lhs)
                    || pyre_object::pyobject::is_bool(lhs)
                    || pyre_object::pyobject::is_float(lhs)
                    || pyre_object::pyobject::is_tuple(lhs)
                    || pyre_object::unicodeobject::is_str(lhs)
                    || pyre_object::bytesobject::is_bytes(lhs)
                {
                    None
                } else if pyre_object::pyobject::is_exact_builtin_instance(lhs) {
                    return Err(DispatchError::InplaceContainerMutationUnsupported { pc: op_pc });
                } else {
                    None
                }
            }
        } else {
            None
        };
    // A `jit_list_append(list, value)` residual (LIST_APPEND opcode, tagged
    // `ListAppendValue`) reaches this generic executor only when the orthodox
    // append fold DECLINED and fell through — most commonly at a realloc
    // boundary, where the backing block is full and the in-place fast-store
    // fold cannot lower the resize (`w_list_append`'s `Vec::push` else-leg is an
    // un-lowered allocating helper).  Executing it here mutates `list` in place
    // (resize + store); journal the pre-append length so an aborting walk's
    // rollback rewinds the one append and the deliver re-applies it exactly once
    // (the same `fbw_list_journal_push_append` contract the fold's own commit uses),
    // making the fall-through abort-safe instead of a silent double.
    let list_append_journal: Option<(pyre_object::PyObjectRef, usize, isize)> =
        if call_descr.get_extra_info().runtime_helper
            == majit_ir::RuntimeHelperKind::ListAppendValue
        {
            let list = args
                .first()
                .map(|&a| a as usize as pyre_object::PyObjectRef);
            list.filter(|&l| !l.is_null() && unsafe { pyre_object::pyobject::is_list(l) })
                .map(|l| unsafe {
                    (
                        l,
                        pyre_object::w_list_len(l),
                        pyre_object::listobject::w_list_allocated(l),
                    )
                })
        } else {
            None
        };
    // `do_residual_call` (pyjitpl.py for CALL_MAY_FORCE_N /
    // CALL_LOOPINVARIANT_N / CALL_N) runs `executor.execute_varargs` for a void
    // call exactly like the value-returning shapes, applying the side effect
    // once during tracing and then resuming the compiled loop at the *next*
    // iteration (`raise_continue_running_normally`, pyjitpl.py, hands
    // back the end-of-iteration-N state so iteration N is never re-run).  Pyre
    // mirrors the second half via the walk-end commit
    // (`flush_walk_end_state_to_frame`, run_perfn_walk): a successful commit
    // adopts the end-of-walk frame so the compiled loop enters at iteration
    // N+1, leaving the eagerly-applied side effect counted once.  Executing
    // void calls here (rather than recording-only) keeps that invariant whole —
    // a deferred void store would be lost on commit (its symbolic op only fires
    // for N+1+), which is why deferral previously forced the no-commit legacy
    // replay.  The replay path that re-runs iteration N has the symmetric
    // hazard for already-executed value calls (e.g. `list.insert` returns the
    // None ref, so it is not a void call yet still mutates) — eager-everything +
    // commit is the single consistent rule, matching `do_residual_call`.
    //
    // The standard virtualizable box pointer for a MayForce residual — a
    // force inside the callee could escape the frame.  None for non-forces
    // opcodes and when no live vable exists (the jitdriver has no standard
    // virtualizable, or unit-test init disabled the heap pointer) — nothing
    // the callee could force.  The token is armed further below, past every
    // decline gate.
    let mut vable_obj_root = if is_may_force {
        ctx.trace_ctx
            .standard_virtualizable_box()
            .and_then(|_| ctx.trace_ctx.virtualizable_heap_ptr())
            .filter(|p| !p.is_null())
            .map(|p| Box::new(p as usize as i64))
    } else {
        None
    };
    // A Python-level callee (e.g. a recursive `fib`) re-enters the
    // interpreter (`eval_loop_jit` → `jit_merge_point`) while this walk still
    // holds the driver in the tracing state.  Suspend re-entrant trace
    // continuation for the duration of the concrete call so the callee runs as
    // plain interpretation instead of starting a nested trace that would share
    // and corrupt this walk's `TraceCtx` (flaky `libsystem_malloc` freelist
    // abort during deep recursion).  Plain C-helper callees never re-enter, so
    // the guard is a no-op for them.
    //
    // In RPython the tracing metainterp and the executing (blackhole /
    // compiled) interpreter are SEPARATE objects, so `do_residual_call`
    // never perturbs the tracer's virtualizable boxes. Pyre shares one
    // `TraceCtx` across the walk and any re-entrant JIT activity the
    // concrete call triggers. `set_vable_ptr` no longer retargets an
    // active ctx (`virtualizable.py write_boxes` stays on the frame the
    // boxes name), but a nested `initialize_virtualizable` or bridge
    // start on this same ctx still would. Snapshot the standard
    // virtualizable pointer and restore it after the call so the walk's
    // subsequent vable token protocol / field reads see the frame being
    // traced, mirroring RPython's separate-state isolation.
    let saved_vable_heap_ptr = ctx.trace_ctx.virtualizable_heap_ptr();
    // The call can allocate, and a minor collection moves a nursery frame.
    // A MayForce call roots the frame through `vable_obj_root`; every other
    // call roots the saved word here, so the restore below writes the
    // forwarded address (`pyjitpl.py` holds the virtualizable in a box the
    // GC traces).
    let saved_vable_root = SavedVableRoot::push(
        saved_vable_heap_ptr.filter(|p| vable_obj_root.is_none() && !p.is_null()),
    );
    // #57 Option C (Finding #1, R1 double-apply guard): whether THIS residual
    // could commit an irreversible heap mutation the journals do not cover,
    // while an in-flight FOR_ITER item is already captured (a consume ran
    // earlier this iteration).  The journaled list ops (setitem / append) run
    // OUTSIDE this executor (`try_walker_orthodox_list_append`) and roll back
    // on abort, so they are not body-effect candidates here.
    //
    // The OLD allow-list (`StoreSubscr` / `CallFn` / `SetCurrentException`
    // only) MISSED the many statement-level mutators that reach this executor
    // concretely, succeed, and carry `RuntimeHelperKind::None` (`store_attr_fn` /
    // `delete_subscr_fn` / `delete_attr_fn` / `list_extend_fn` / `store_name_fn`
    // / `store_global` / `store_slice` …): a missed mutator is a silent double
    // on a body re-run (correctness-FATAL).  Track residuals that WRITE live
    // heap state outside the journals.
    //
    // The write discriminator is the residual's RESULT TYPE plus the
    // value-returning-mutator helper tags — NOT `extraeffect`, which cannot
    // separate a write from a read: `getattr_fn` (a pure `.append` bound-method
    // lookup) and `store_attr_fn` are BOTH `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    // (the analyzer is stubbed, so `write_descrs_*` are empty for both).  A
    // `Void`-result residual produces no value, so it is executed solely for
    // its heap side effect → a write.  Every statement-level mutator above
    // lowers to a `residual_call_*_v` (Void); the benign reads (`getattr`,
    // `load_global`, `load_name`, `load_deref` …) all RETURN a value, so a
    // Void result is a clean write proxy that does not over-refuse a read.
    //
    // The few VALUE-returning writes the Void test cannot see are caught by
    // helper tag: `CallFn` (an opaque Python call returning its None ref may
    // mutate arbitrary state), `StoreSubscr` (a dict/object `o[k]=v` returning
    // the stored value), `SetCurrentException` (the TLS exc-slot write), and
    // `StoreDeref` (an in-place closure-cell write returning the slot value —
    // `nonlocal n; n += 1`).  Over-refusing only these (never a benign read)
    // keeps the journaled-append body (`for_mutate`) delivering, since its
    // `getattr`/append residuals return `Ref` and carry no write tag.
    //
    // Provably read-only/elidable residuals are exempt up front: `@jit.elidable`-
    // class (`check_is_elidable`: `EF_ELIDABLE_*`, the pure executor folds
    // these) or `EF_LOOPINVARIANT` (loop-hoisted).  The `for_iter_next` consume
    // itself ([`RuntimeHelperKind::ForIterNext`]) is excluded — it is the SOURCE of
    // the capture (it runs while the PRIOR iteration's item is still in flight),
    // not a body effect for that prior iteration.  Sampled BEFORE the call so
    // the success arm can flag an effect that committed AFTER the in-flight
    // consume.
    let ei = call_descr.get_extra_info();
    let helper = ei.runtime_helper;
    // The declared re-runnability class, the axis `EffectInfo` is decided on at
    // codewriter time (`jtransform.py:620-630`): re-executing the opcode that
    // contains this residual re-executes the residual, which is harmless only
    // for an elidable or loop-hoisted one.  Feeds [`ESCAPE_OPCODE_WINDOW`].
    let reentrant_residual =
        ei.check_is_elidable() || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant;
    // PyPy traces through `str_descr_new` -> `space.str` and therefore knows
    // that an exact builtin scalar cannot dispatch to user `__str__` code.
    // Pyre reaches the same operation through the opaque `CallFn` helper; if
    // it remains classified as arbitrary here, a loop-bearing inlined caller
    // (for example `fold`'s `for ch in str(value)`) is denied at the nested
    // residual and the whole bridge is thrown away once before that caller is
    // learned as non-inlinable.
    //
    // This is deliberately an observed-value fact, not a blanket blessing of
    // `str(x)`: both the callable and operand must be exact builtins in the
    // concrete execution this walk may have to replay.  Exact immutable scalar
    // formatting allocates only its fresh result and cannot mutate live heap or
    // enter a user frame, so replay has the same safety as an elidable call.
    // A subclass/callable override misses pointer/type identity and keeps the
    // ordinary nested-residual decline.
    let observed_exact_scalar_str =
        helper == majit_ir::RuntimeHelperKind::CallFn && args.len() == 3 && {
            let callable = args[0] as pyre_object::PyObjectRef;
            let operand = args[2] as pyre_object::PyObjectRef;
            let str_type =
                pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::STR_TYPE);
            !callable.is_null()
                && !operand.is_null()
                && std::ptr::eq(callable, str_type)
                && unsafe {
                    pyre_object::is_exact_builtin_instance(operand)
                        && (pyre_object::is_int_or_long(operand)
                            || pyre_object::is_float(operand)
                            || pyre_object::is_complex(operand)
                            || pyre_object::is_str(operand)
                            || pyre_object::is_none(operand))
                }
        };
    // PyPy traces `space.iter(w_exact_unicode)` through
    // `W_UnicodeObject.descr_iter` into a fresh sequence iterator.  Pyre's
    // tagged `GetIter` call has the same no-user-dispatch property for an
    // exact string; subclasses retain the conservative decline.
    let observed_exact_str_iter =
        helper == majit_ir::RuntimeHelperKind::GetIter && args.len() == 1 && {
            let operand = args[0] as pyre_object::PyObjectRef;
            !operand.is_null()
                && unsafe {
                    pyre_object::is_str(operand) && pyre_object::is_exact_builtin_instance(operand)
                }
        };
    // PyPy's `space.ord` reads the immutable unicode payload directly.  Match
    // both canonical builtin-code identity and the observed exact string so a
    // rebound `ord` or a user object cannot enter this replay-safe class.
    let observed_exact_str_ord = helper == majit_ir::RuntimeHelperKind::CallFn
        && args.len() == 3
        && pyre_interpreter::builtins::is_builtin_ord_function(args[0] as pyre_object::PyObjectRef)
        && {
            let operand = args[2] as pyre_object::PyObjectRef;
            !operand.is_null() && unsafe { pyre_object::is_str(operand) }
        };
    // `abstract_isinstance_w` handles a real type through
    // `p_recursive_isinstance_type_w`: first the side-effect-free MRO test,
    // then (only on a miss) `w_inst.__class__` (`abstractinst.py`).  A
    // hit is replay-safe directly.  A miss is replay-safe only when the
    // instance type has already proven and memoized that it inherits
    // `object.__getattribute__` AND its MRO resolves `__class__` to the
    // canonical `object.__class__` descriptor.  The first fact alone is not
    // enough: `@property def __class__` retains the default getattribute path
    // but runs user code here.  Type mutation invalidates both lookup facts.
    //
    // Require the canonical callable and an ordinary class whose metaclass is
    // exactly `type`; tuple/union classinfo and custom `__instancecheck__`
    // remain on the conservative nested-residual decline.
    let observed_replay_safe_isinstance = helper == majit_ir::RuntimeHelperKind::CallFn
        && args.len() == 4
        && args[1] == 0
        && pyre_interpreter::builtins::is_builtin_isinstance_function(
            args[0] as pyre_object::PyObjectRef,
        )
        && {
            let object = args[2] as pyre_object::PyObjectRef;
            let classinfo = args[3] as pyre_object::PyObjectRef;
            !object.is_null()
                && !classinfo.is_null()
                && pyre_interpreter::typedef::r#type(classinfo).is_some_and(|meta| {
                    std::ptr::eq(meta.as_ptr(), pyre_interpreter::typedef::w_type())
                })
                && pyre_interpreter::typedef::r#type(object).is_some_and(|actual| unsafe {
                    pyre_interpreter::baseobjspace::isinstance_w(object, classinfo)
                        || pyre_interpreter::baseobjspace::isinstance_miss_class_lookup_is_pure(
                            actual.as_ptr(),
                        )
                })
        };
    // `tuple(exact_list)` arrives through the generic `CallFn` helper, not the
    // BUILD_TUPLE or CALL_INTRINSIC_1 helpers.  `builtin_tuple`'s exact-list
    // arm reads the strategy-aware length/items through `w_list_len` and
    // `w_list_getitem`, then passes the copied items to `w_tuple_new`
    // (`builtins.rs`).  It neither calls user code nor writes the list, and it
    // always returns a fresh tuple.  The exact callable, no-receiver, one-arg,
    // exact-list checks below are therefore the complete replay-safe surface;
    // every other `CallFn` remains opaque and effectful.  It shares the `CallFn`
    // helper with the `str` and `ord` arms above and is separated from them by
    // the callable identity each one pins, so at most one can hold.
    let replay_safe_tuple_from_list = helper == majit_ir::RuntimeHelperKind::CallFn
        && args.len() == 3
        && args[1] == 0
        && args[0] as usize
            == pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE)
                as usize
        && args[2] != 0
        && unsafe { pyre_object::is_exact_list(args[2] as usize as pyre_object::PyObjectRef) };
    // `operator.index(x)` reaches `space_index`, whose first test is
    // `is_int_or_long`: an int (or long) is returned as-is, before any
    // `__index__` lookup, so no user code runs and nothing is written.  Every
    // other argument type dispatches through `__index__` and stays opaque.
    // Pin the callable the way the `ord` / `isinstance` / `tuple` arms above
    // do — it shares the `CallFn` helper with them and is separated by the
    // identity each one pins, so at most one can hold, and a rebound
    // `operator.index` is a different object that keeps the conservative
    // treatment.  Observe the argument too: the callable alone does not bound
    // what `space_index` will do.  `randrange`'s `_index(start)` is the
    // shape that reaches here: it is the ONLY residual the aborted walk of a
    // `[randrange(k) for _ in ...]` body executes, and classifying it as a live
    // heap write refused the in-flight FOR_ITER delivery and dropped the
    // element.
    let observed_exact_int_index = helper == majit_ir::RuntimeHelperKind::CallFn
        && args.len() == 3
        && args[1] == 0
        && pyre_interpreter::module::operator::is_operator_index_function(
            args[0] as pyre_object::PyObjectRef,
        )
        && {
            let operand = args[2] as pyre_object::PyObjectRef;
            !operand.is_null() && unsafe { pyre_object::pyobject::is_int_or_long(operand) }
        };
    // `do_residual_call` (`pyjitpl.py`) runs at any framestack depth.
    // Exact-int `BINARY_OP` / `COMPARE_OP` has no user dunder and is not
    // the self-recursive `CALL_ASSEMBLER` trampoline the nested-decline
    // hazard names.  Refusing it inside `rec` aborted the inline before
    // `sys._getframe` could force a multi-frame carrier.
    // Name the helper first.  A descended `jit_bigint_*` residual also
    // has two Ref slots, but those are `*mut BigInt` payloads, not
    // `PyObject`s; `is_int` must not see them.
    let opcode_binop_or_compare_fnaddr = BINARY_VALUE_FROM_TAG_FNADDRS.contains(&(func_ptr as i64))
        || COMPARE_VALUE_FROM_TAG_FNADDRS.contains(&(func_ptr as i64));
    let binop_or_compare_helper = matches!(
        helper,
        majit_ir::RuntimeHelperKind::BinaryOp | majit_ir::RuntimeHelperKind::CompareOp
    ) || opcode_binop_or_compare_fnaddr;
    let is_exact_int_word = |word: i64| {
        let obj = word as pyre_object::PyObjectRef;
        !obj.is_null()
            && (word as usize).is_multiple_of(std::mem::align_of::<usize>())
            && unsafe {
                pyre_object::is_int(obj)
                    && !pyre_object::is_bool(obj)
                    && pyre_object::is_exact_builtin_instance(obj)
            }
    };
    let exact_int_binop_operands = binop_or_compare_helper && {
        let ref_operand_words: Vec<i64> = call_descr
            .arg_types()
            .iter()
            .enumerate()
            .filter(|(_, ty)| **ty == majit_ir::Type::Ref)
            .filter_map(|(i, _)| args.get(i).copied())
            .collect();
        ref_operand_words.len() == 2
            && is_exact_int_word(ref_operand_words[0])
            && is_exact_int_word(ref_operand_words[1])
    };
    let observed_exact_int_binop = matches!(
        helper,
        majit_ir::RuntimeHelperKind::BinaryOp | majit_ir::RuntimeHelperKind::CompareOp
    ) && exact_int_binop_operands;
    // A codewriter residual of the same helper can arrive with
    // `RuntimeHelperKind::None`.  The fnaddr only names the helper;
    // user `__add__` / `__eq__` still mutate, so keep the exact-int gate.
    let opcode_binop_or_compare = opcode_binop_or_compare_fnaddr && exact_int_binop_operands;
    // `BUILD_TUPLE` / `BUILD_LIST` create a fresh container from their fresh
    // backing array (`pyopcode.py`).  Re-executing either allocation
    // cannot mutate an object visible before the call.  Upstream records list
    // construction directly as `opimpl_newlist` (`pyjitpl.py`) and
    // allows residual calls at every `MIFrame` depth (`pyjitpl.py`);
    // it has no nested-callee abort for these allocation helpers.
    //
    // Disjoint from the three observed-value classes above: those name `CallFn`
    // and `GetIter` over exact builtin scalars, this one names the two
    // allocation helpers, and no helper kind is in both.
    let replay_safe_fresh_allocation = matches!(
        helper,
        majit_ir::RuntimeHelperKind::NewtupleFromArray
            | majit_ir::RuntimeHelperKind::NewlistFromArray
            | majit_ir::RuntimeHelperKind::BuildStringFromArray
    );
    // A journaled cursor is replay-safe: `fbw_bridge_iter_journal_rollback`
    // puts it back.  A generator, `map`, dict/set iterator, itertools
    // iterator, or user `__next__` records nothing, so the consume is not
    // free — entry replay would call `__next__` again.
    // `convert_and_run_from_pyjitpl` continues from the framestack instead.
    let for_iter_journaled = helper == majit_ir::RuntimeHelperKind::ForIterNext
        && args
            .first()
            .is_some_and(|&iter| fbw_bridge_iter_journalable(iter as pyre_object::PyObjectRef));
    let provably_side_effect_free = reentrant_residual
        || is_rerunnable_bookkeeping
        || is_rewindable_root_bracket
        || for_iter_journaled
        || observed_exact_scalar_str
        || observed_exact_str_iter
        || observed_exact_str_ord
        || observed_replay_safe_isinstance
        || replay_safe_fresh_allocation
        || replay_safe_tuple_from_list
        || observed_exact_int_index
        || observed_exact_int_binop
        || opcode_binop_or_compare;
    let writes_live_heap = call_descr.result_type() == majit_ir::Type::Void
        || (helper == majit_ir::RuntimeHelperKind::CallFn && !replay_safe_tuple_from_list)
        || helper_kind_writes_live_heap(helper);
    // Inside an inline sub-walk, decline before any residual that is not
    // provably side-effect-free.  Ref-result getters/dunders/user `__next__`
    // can mutate live heap through user frames while `writes_live_heap` is
    // false, and rollback would miss that concrete mutation.  The helper no-ops
    // on an empty session framestack, so top-level depth-1 behavior is unchanged.
    if !provably_side_effect_free {
        if std::env::var_os("PYRE_LB_SITE").is_some() {
            // The decline cause the gate prints is `None` on this arm, so name
            // the callee here instead: once a body is traced through, the
            // residual that fails the predicate is one of its whole helper set.
            let name = pyre_interpreter::jit_trace_fnaddrs()
                .iter()
                .find(|(_, a)| *a == func_ptr as i64)
                .map(|(n, _)| *n);
            eprintln!(
                "[lb-exec] pc={op_pc} helper={helper:?} extraeffect={:?} elidable={} \
                 rtype={:?} fn={name:?}",
                ei.extraeffect,
                ei.check_is_elidable(),
                call_descr.result_type(),
            );
        }
        fbw_abort_nested_unjournaled_residual(ctx, op_pc, None)?;
    }
    // The loop-variable binding store is the op at the in-flight FOR_ITER's
    // `body_pc` (the FOR_ITER continue-arm fallthrough), a STORE_NAME/
    // STORE_GLOBAL that writes the just-consumed item to the loop target (a
    // module/global-scope `for i in …`; a function-scope loop var is a
    // STORE_FAST frame local that never becomes a residual).  Re-delivery
    // re-runs the body from `body_pc`, re-storing the SAME re-delivered item to
    // the SAME name — an idempotent write, never an accumulating double.  Like
    // the `is_idempotent_gc_barrier` write barrier it still EXECUTES concretely
    // (the module dict must hold the binding for the walk's remaining reads) but
    // it is not a body effect: keep it out of the R1 in-flight-FOR_ITER
    // accounting so an escaping residual later in the same body does not
    // refuse-drop the whole iteration.
    // `vstack_cur_pypc` is the pc the walk is ABOUT TO ENTER
    // (`reconcile_vstack_at_boundary` sets it to `new_pypc` after reconciling
    // the PREVIOUS opcode), so at this residual it names the opcode being
    // walked.  The loop-var store is recognised by the FOR_ITER body's own
    // relation `body_pc + 1 == vstack_cur_pypc`, i.e. the walk has advanced one
    // opcode past the recorded body pc — not by a next-instr convention.
    let is_loop_var_binding_store = is_loop_var_binding_store(ctx, helper);
    // PUSH_EXC_INFO's carrier clear is void-returning, so `writes_live_heap`
    // alone counts it.  The slot it writes exists only to keep a propagating
    // exception rooted for the collector — nothing reads it back as a value —
    // and `push_exc_info` performs the same clear at the same point, so a
    // re-run of the body reaches the same state rather than doubling anything.
    // That is the `is_idempotent_gc_barrier` category, and BOTH accountings of
    // it read this one binding: the gh#467 heap-write odometer below stated the
    // exemption while the R1 discriminator on the next line did not, so a `for`
    // body whose only committed residual was `except`'s clear had its delivery
    // refused and the whole iteration dropped — the outcome the refusal exists
    // to be safer than.
    let writes_gc_liveness_root_only =
        helper == majit_ir::RuntimeHelperKind::ClearInFlightException;
    // Store / list-append journals have rollback entry points; executing
    // those residuals is recoverable.  A namespace helper is recoverable
    // only when [`journal_walker_namespace_write`] actually recorded the
    // displaced binding.  A non-dict `exec` locals mapping returns `None`
    // — its `__setitem__` / `__delitem__` cannot be undone — so that write
    // stays a body effect.  A transparent helper sub-walk (`write_cell`,
    // `binary_value_from_tag`, …) is not a Python frame: its Void
    // write-barrier residuals belong to the journaled outer store, not to
    // the FOR_ITER body of the live frame.  Marking either class as
    // `body_effect_since_consume` makes [`fbw_foriter_inflight_take`]
    // refuse delivery and skip the cursor restore, which is how a
    // module-level `for` over journaled STORE_NAME cells lost iterations.
    let residual_will_be_journaled = inplace_list_journal.is_some()
        || list_append_journal.is_some()
        || namespace_write_journaled;
    let body_effect_candidate = !provably_side_effect_free
        && !is_idempotent_gc_barrier
        && !is_loop_var_binding_store
        && !writes_gc_liveness_root_only
        && writes_live_heap
        && fbw_foriter_inflight_active()
        && !residual_will_be_journaled
        && !ctx.fbw_mode.transparent_helper_subwalk;
    // `vinfo.tracing_before_residual_call(virtualizable)`
    // heap half: every decline gate has now passed, so the helper WILL
    // execute — set TOKEN_TRACING_RESCALL on the active virtualizable so a
    // force inside the callee is observable afterwards.  Armed AFTER the
    // inline-subwalk decline so a declined residual never strands the token:
    // tracing_before pairs with the tracing_after clear below only for
    // residuals that proceed, mirroring `do_residual_call` where
    // `vable_and_vrefs_before_residual_call` (pyjitpl.py) runs only past
    // the OS_NOT_IN_TRACE / force-virtual short-circuits. RPython mirror:
    // `pyjitpl.py`.
    // `vrefinfo.tracing_before_residual_call(vref)` for every live vref
    // (`pyjitpl.py`), which upstream runs first inside the same
    // `vable_and_vrefs_before_residual_call` the vable half below mirrors.
    // Stamps TOKEN_TRACING_RESCALL so the post-call check can tell "forced by
    // this callee" from "untouched".  Armed here, past every decline gate, for
    // the same reason the vable half is: a declined residual must not strand a
    // token.
    //
    // Gated on `is_may_force` — the walker's `check_forces_virtual_or_
    // virtualizable()` — because `do_residual_call` runs the whole preparation
    // block only for `assembler_call or effectinfo.check_forces_...`
    // (`pyjitpl.py`).  A call that cannot force needs no stamp, and
    // stamping one would leave `tracing_after_residual_call` reading a token
    // nobody will clear.
    if is_may_force {
        // How many pairs the bracket actually walks.  Both halves iterate
        // `virtualref_boxes`, and nothing in-tree reports its length, so
        // "the bracket iterates zero times" has never been a measured claim —
        // only a restatement of which call sites populate it.  Report the
        // count here so a sweep can settle it; gated like the rest of the
        // walk's build reporting.
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[vref-bracket] pairs={}",
                ctx.trace_ctx.virtualref_boxes_len() / 2
            );
        }
        ctx.trace_ctx.vrefs_before_residual_call();
    }
    let live_frame = if ctx.fbw_mode.snapshot_sym.is_null() {
        0
    } else {
        // The symbol stores a raw address. A residual is a collection
        // point (`push_roots` below), so a nursery frame may have been
        // forwarded since the address was captured. RPython's live
        // virtualizable is a GC pointer and is rewritten in place;
        // follow the stub the same way `gc_current_object_address` does.
        let raw = unsafe { (*ctx.fbw_mode.snapshot_sym).live_vable_frame_addr() };
        if raw == 0 {
            0
        } else {
            majit_gc::gc_current_object_address(raw)
        }
    };
    // See [`LiveFrameRoot`].
    let live_frame_root = LiveFrameRoot {
        root: owner_root_if_gc(live_frame),
        sym: ctx.fbw_mode.snapshot_sym as *mut Sym,
    };
    // Resolved against the callee's OWN metadata, because `vstack_cur_pypc` is
    // the outer walk's mirror and a sub-walk never advances it.
    let inline_callee_pc = inline_callee_py_pc(ctx, op_pc);
    let mut saved_last_instr_shadow: Option<(OpRef, Value)> = None;
    let vable_root_depth = if let Some(obj) = vable_obj_root.as_mut() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        // Publish the current Python pc so a force inside the callee reports
        // the executing line rather than the one the last resume point left
        // behind.
        //
        // Only while the pc indexes the walk's own virtualizable.  Inside an
        // inline sub-walk `vstack_cur_pypc` is in the CALLEE's code, and both
        // halves below name the outer frame, so publishing there stamps a
        // foreign pc onto it — `offset2lineno` then resolves it against the
        // outer code object and reports whatever line that byte happens to sit
        // on.  [`LiveLastInstrGuard`] makes the matching retarget for the
        // concrete store, publishing onto the callee's own frame instead.
        if current_inline_concrete_frame() == 0 {
            // A transparent helper is not a Python frame: its
            // `vstack_cur_pypc` stays 0.  Publish the CALL-site opcode
            // instead of writing last_instr=0 over the portal coordinate.
            let py_pc = if ctx.fbw_mode.transparent_helper_subwalk {
                ctx.fbw_mode
                    .inline_caller_py_pc
                    .unwrap_or(ctx.vstack_cur_pypc)
            } else {
                ctx.vstack_cur_pypc
            };
            if ctx.fbw_mode.transparent_helper_subwalk && py_pc == 0 {
                // Keep the last portal store rather than clobber it with 0.
            } else {
                let last_instr = ctx.trace_ctx.const_int(py_pc as i64);
                // Scope the box half to the residual just as
                // `LiveLastInstrGuard` scopes the heap half.  `_opimpl_setfield_vable`
                // leaves both halves equal (`pyjitpl.py`), and
                // `check_synchronized_virtualizable` asserts that invariant
                // (`pyjitpl.py`).
                if let Some(idx) = info.static_field_index_by_name("last_instr")
                    && let Some((prev_op, prev_val)) = ctx.trace_ctx.virtualizable_entry_at(idx)
                {
                    saved_last_instr_shadow = Some((prev_op, prev_val));
                }
                crate::trace_opcode::mirror_vable_static_to_boxes(
                    ctx.trace_ctx,
                    "last_instr",
                    last_instr,
                    majit_ir::Value::Int(py_pc as i64),
                );
                // Record the runtime heap half of the same `_opimpl_setfield_vable`
                // shape.  The mirror above synchronizes the tracing-time shadow
                // and live frame; this store keeps compiled execution paired too.
                //
                // Upstream needs neither at a residual call: its frame readers are
                // traced in and read `last_instr` off the virtual frame.  pyre
                // residualizes them and reads the heap, and the frame-chain walk no
                // longer forces (`ExecutionContext::force_frame`), so without this
                // store nothing keeps the field current in compiled code: left at
                // the last resume point, `_warnings::setup_context` keys its
                // registry on the wrong line and re-issues a warning the
                // interpreted run already deduplicated.
                if let Some(vable_ref) = ctx.trace_ctx.standard_virtualizable_box()
                    && let Some(idx) = info.static_field_index_by_name("last_instr")
                {
                    // Record under the PARENT-STRUCT field descr, the resolution
                    // `vable_setfield` applies through `vable_static_record_descr`;
                    // `vable_setfield_descr` is a raw pass-through and does not.
                    // The vinfo's own `static_field_descrs[i]` numbers the field by
                    // the vinfo `[token, statics, arrays]` order, which diverges
                    // from PyFrame's struct declaration order, so the store pairs
                    // against the wrong slot.  `virtualizable.py:71` builds the
                    // vinfo descrs with `cpu.fielddescrof(VTYPE, name)`, so
                    // upstream's vinfo descr and the descr an ordinary
                    // `setfield_gc` carries are one object and the question cannot
                    // arise there.  Here it decides whether this publish and
                    // `emit_traceback_node`'s store — same frame, same offset —
                    // supersede each other or survive as two independent
                    // locations flushed in an arbitrary order.
                    let descr = info.static_field_struct_descr(idx);
                    ctx.trace_ctx
                        .vable_setfield_descr(vable_ref, last_instr, descr);
                }
            }
        }
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut **obj));
            info.tracing_before_residual_call(**obj as usize as *mut u8);
        }
        Some(root_depth)
    } else {
        None
    };
    // #57 Option C (Finding #1, user-frame signal): the Void/helper-tag write
    // discriminator above cannot see a body effect committed through USER
    // PYTHON CODE by a value-returning (`Ref`), `RuntimeHelperKind::None`,
    // `MayForce` residual: `obj.prop` (a `@property` getter / `__getattr__` /
    // descriptor `__get__`), `a + b` / `a == b` (user `__add__` / `__eq__`),
    // `iter(obj)` (user `__iter__`), `str(obj)` / `f"{obj}"` (user `__str__` /
    // `__format__`), `import name`.  Each RETURNS a value (so the Void proxy
    // misses it) and carries no write tag, yet its getter/dunder/module body
    // may mutate live heap.  Those mutations all run a USER PYTHON FRAME (the
    // getter's bytecode); a pure builtin path (`seen.append`'s C-level
    // bound-method lookup, `int.__add__`) does NOT.  Snapshot the monotonic
    // frame eval-loop entry odometer before the call; if it advanced while an
    // in-flight FOR_ITER item is active, the residual ran user bytecode that
    // could have committed an irreversible body effect → flag it (the success
    // arm compares post-call).  Sampled only when an item is in flight and the
    // residual is not provably read-only (an elidable / loop-invariant fold or
    // the `for_iter_next` consume itself never counts).
    let user_frame_snapshot = (!provably_side_effect_free && fbw_foriter_inflight_active())
        .then(pyre_interpreter::call::frame_entry_count);
    // #493: a NEW consume attempt for a FOR_ITER whose prior item is still in
    // flight means that item's body ran to completion — mark the entry BEFORE
    // the call so an attempt that aborts mid-way (a kept-stack guard on the
    // exhaustion arm) still records the completion; a successful attempt
    // replaces the entry with a fresh one anyway.
    if helper == majit_ir::RuntimeHelperKind::ForIterNext {
        // Journal the pre-advance cursor the way the range and zip FOR_ITER
        // specializations journal theirs.  A bridge recording that does not
        // commit restores it in the walk epilogue; a root walk keeps it until
        // the delivery site, which restores it only when it could not hand the
        // item back at the loop header.
        if let Some(&iter) = args.first() {
            let recorded = fbw_bridge_iter_journal_capture(iter as pyre_object::PyObjectRef);
            debug_assert_eq!(recorded, for_iter_journaled);
        }
        let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
            .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
        fbw_foriter_inflight_mark_attempt(body);
    }
    // gh#467: sample the user-frame odometer UNCONDITIONALLY (not only under an
    // in-flight FOR_ITER) so the concrete-heap-write gate can detect a callee
    // sub-walk mutation committed through a value-returning dunder body — the
    // same user-frame signal Finding #1 uses, generalized past FOR_ITER.
    let heap_write_odometer_before =
        (!provably_side_effect_free).then(pyre_interpreter::call::frame_entry_count);
    // The live frame's locals as they stood before the residual ran.  A
    // residual that writes fastlocals writes THIS object while the walk reads
    // its own copy, so the diff taken afterwards names exactly the slots the
    // shadow has to learn about.
    //
    // The copies live in real shadow-stack slots rather than in a plain `Vec`
    // (`gctransform/framework.py` push_roots/pop_roots around a collection
    // point).  The residual IS a collection point, and a minor collection
    // forwards the frame's own array in place while copies taken outside the
    // root stack keep their pre-collection words.  Diffing against those reads
    // every moved slot as a write, and the adoption below would then put the
    // array's value over a box the walk holds and the frame was never told
    // about.
    let mut residual_locals_roots = ((fbw_debug_abort_enabled()
        || fbw_import_residual_locals_enabled())
        && is_may_force
        && live_frame != 0
        && majit_gc::gc_owns_object(live_frame_root.current(live_frame)))
    .then(|| unsafe {
        // Re-read the owner root. The captured word is not chased: a
        // finished collection has already dropped the forwarding stub
        // (exception_reused_object_tb_not_doubled).
        let live_frame = live_frame_root.current(live_frame);
        let pf = &*(live_frame as *const pyre_interpreter::PyFrame);
        let snapshot = locals_w!(pf).as_slice().to_vec();
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.publish(&snapshot);
        roots.normalize(base, snapshot.len());
        (roots, base, snapshot.len())
    });
    let exec_result = {
        let escape_frame = if is_may_force { live_frame } else { 0 };
        // Latch the operand-stack mirror for the escape flush: at force time
        // the walk-end flush needs the caller's mid-expression stack, which
        // the vable shadow cannot provide.
        //
        // A WRITING residual is latched too, but never to resume AT this
        // opcode: the withdrawal below cancels its commit and restores the
        // pre-flush frame unconditionally, so the interpreter never
        // re-executes it and never re-applies the write.  What the commit
        // leaves behind for it is a WITNESS — it proves every mirror slot
        // resolved to a concrete non-null Ref, i.e. the walk holds a complete
        // mid-expression stack image, which is exactly the precondition for
        // building the blackhole resume PAST the residual.  Forcing is NOT
        // limited to frame-introspection reads (`hook_access_field`,
        // rvirtualizable.py, forces on every redirected-field access,
        // reads AND writes), and every Python-visible frame MUTATOR (the
        // `f_lineno`/`f_trace` setters, `sys.settrace`, `_warnings.warn`,
        // which forces the caller frame for `__name__` and then mutates
        // `__warningregistry__` with no user frame entered) is a
        // Void-returning store or a CALL-shaped helper, so all of them land
        // on the writing side and take that withdrawal.
        //
        // A non-writing residual that entered no user frame keeps its commit
        // and the resume-AT-opcode semantics: what survives the gates there
        // is attribute/item READS of frame-family objects, which are
        // idempotently re-executable (re-execution reads the same flushed
        // values the first execution saw; a token re-force is a no-op).
        //
        // Why the licence is `!writes_live_heap` and not the effect class the
        // rewound residual declares: BOTH upstream-shaped static licences are
        // empty on this path.  `check_is_elidable()`/`EF_LOOPINVARIANT` is
        // disjoint from forcing by construction — `pyjitpl.py:2007` takes the
        // forcing arm iff `check_forces_virtual_or_virtualizable()`
        // (`extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, effectinfo.py)
        // and routes elidable/loop-invariant down the `else` arm at `:2084`.
        // An empty declared write set is unavailable too: every residual that
        // reaches this gate carries `EF_RANDOM_EFFECTS`, whose write-descr sets
        // are `None` — top, not bottom — by upstream's own assertion
        // (effectinfo.py).  So no `EffectInfo` predicate can license
        // this rewind, and the licence has to come from the shape gates above.
        // Measured over `pyre/bench/synth` (312 files, 115 forces): the only
        // shape that commits an `EscapeResumeKind::Exact` is a `LoadAttr`/`Ref`
        // frame-family read, 10 of them, from `getframe_stored_fback_walk` and
        // `getframe_force_cancel_journal`; 5 enter a user frame and have the
        // commit withdrawn below, 5 do not and are re-executed.  Every other
        // force either commits nothing or takes the merge-point fallback, whose
        // `RerunsOpcode` kind `commit_walk_end` refuses outright.
        //
        // A sub-walk's mirror describes the callee frame, not the escape
        // frame, so it is never latched (the nested unjournaled-residual
        // decline above already aborts before this).
        // Not latched on a bridge walk: its abort path bypasses the
        // run_perfn_walk epilogue that adopts (or restores) a committed
        // escape flush, so a commit there would strand a moved frame.
        let escape_py_pc = (!ctx.fbw_mode.inline_subwalk && ctx.vstack_valid)
            .then_some(ctx.vstack_cur_pypc as usize);
        let escape_stack = escape_py_pc
            .filter(|py_pc| {
                escape_frame != 0
                    && !ctx.trace_ctx.is_bridge_trace
                    && escape_opcode_window_clean(*py_pc)
            })
            .map(|_| ctx.frame_state.borrow().vstack_boxes.clone());
        let _frame_escape = ActiveFrameEscapeGuard::enter(escape_frame, escape_py_pc, escape_stack);
        // `executioncontext.py enter` for the inlined callee this residual
        // runs inside of.
        // RPython's residual executes against `metainterp.framestack[-1]`:
        // the concrete frame belongs to the live MIFrame, not to the portal
        // root.  A forward inline publishes the same identity through
        // `InlineConcreteFrameGuard`; a bridge-reconstructed inline carries
        // it on its own `CalleeLocalsShadow`, because that frame must not be
        // installed as the walk's standard virtualizable.  Prefer the
        // per-context frame so bridge tracing cannot collapse `super()` /
        // `sys._getframe()` onto the caller, retaining the TLS value only for
        // older non-shadow inline paths.
        let concrete_inline_frame = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .map(|shadow| shadow.concrete_frame)
            .filter(|&frame| frame != 0)
            .unwrap_or_else(current_inline_concrete_frame);
        // The level's own slot source travels with the frame: a force inside
        // this residual reaches only the `TraceCtx`, and the frame it forces is
        // this callee, not the traced virtualizable.
        let published_shadow = ctx
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(|_| {
                let frame_reg = ctx
                    .inline_callee_consts
                    .and_then(|consts| {
                        crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)
                    })
                    .map(|jitcode| jitcode.metadata.portal_frame_reg)?;
                Some((ctx.frame_state.clone(), frame_reg))
            });
        let _frame_chain = ResidualFrameChainGuard::enter(concrete_inline_frame, published_shadow);
        let live_py_pc = if ctx.fbw_mode.inline_subwalk {
            ctx.vstack_cur_pypc
        } else {
            live_py_pc_from_snapshot(ctx, op_pc).unwrap_or(ctx.vstack_cur_pypc)
        };
        let _callee_last_instr = LiveLastInstrGuard::enter(
            live_frame_root.current(live_frame),
            live_py_pc,
            inline_callee_pc,
        );
        let _caller_last_instr = ctx.fbw_mode.inline_caller_py_pc.map(|py_pc| {
            LiveLastInstrGuard::enter_frame(
                live_frame_root.current(live_frame) as *mut pyre_interpreter::PyFrame,
                py_pc,
            )
        });
        let _suspend = majit_metainterp::TraceContinuationSuspendGuard::enter(ctx.trace_ctx);
        majit_metainterp::executor::execute_residual_call(call_descr, func_ptr, &args)
    };
    // A Ref result is a nursery object. The next residual in this walk
    // (`abs(x - y)` after complex subtract) allocates and can collect it
    // before `set_opref_concrete` roots the recorded op. Pin the live
    // word now and stamp that word below.
    let mut residual_result_scope = None;
    let exec_result = match exec_result {
        Ok(result_i64)
            if !is_void && call_descr.result_type() == majit_ir::Type::Ref && result_i64 != 0 =>
        {
            let obj = result_i64 as usize as pyre_object::PyObjectRef;
            if obj.is_null() {
                Ok(result_i64)
            } else {
                let live = if let Some((ref roots, _, _)) = residual_locals_roots {
                    roots.pin_root(obj)
                } else {
                    let scope = pyre_object::gc_roots::push_roots();
                    let live = scope.pin_root(obj);
                    residual_result_scope = Some(scope);
                    live
                };
                Ok(live as i64)
            }
        }
        other => other,
    };
    let _residual_result_scope = residual_result_scope;
    // Declared only now, so this residual constrains the residuals that FOLLOW
    // it inside the same opcode and never itself: the gate above read the
    // window, and a force inside the callee reads it again from
    // `flush_active_frame_escape` while it is still the gate's view.
    escape_opcode_window_note(ctx.vstack_cur_pypc as usize, reentrant_residual);
    if !provably_side_effect_free && !is_idempotent_gc_barrier {
        fbw_mark_executed_nonpure_residual();
        // Count only a FOREIGN non-pure residual: a self-recursive call is the
        // fold target running because its fold declined, not a body side effect.
        // The executor the `CALL_ASSEMBLER` fold runs to stamp its concrete
        // result is the same thing whatever the callee's identity — that
        // execution IS the folded call, and the trace carries the
        // CALL_ASSEMBLER standing for it.  Reading the fold scope rather than
        // only the callee's code keeps the first successful fold in a body from
        // latching the gate that then declines every later call in the walk.
        if !selfrec_ca_fold_active()
            && !residual_callee_is_walk_self_recursive(ctx, allboxes, helper)
        {
            fbw_mark_executed_body_residual();
        }
    }
    let restored_vable_heap_ptr = vable_obj_root
        .as_ref()
        .map(|obj| **obj as usize as *const u8)
        .or(saved_vable_root.forwarded().or(saved_vable_heap_ptr))
        .unwrap_or(std::ptr::null());
    drop(saved_vable_root);
    ctx.trace_ctx
        .set_virtualizable_heap_ptr(restored_vable_heap_ptr);
    // `pyjitpl.py:2049` step 3, "after this call, check the vrefs.  If any
    // have been forced by the call, then we record in the trace a
    // VIRTUAL_REF_FINISH---before we record any CALL".  Runs before the
    // virtualizable check below, matching the upstream order, and before the
    // CALL op is recorded further down.  A vref the callee handed to Python
    // stops being tracked and its box becomes ConstPtr(NULL), so the resume
    // snapshot no longer claims the frame is still virtual.  Paired with the
    // pre-call stamp, so it carries the same `is_may_force` gate.
    if is_may_force {
        ctx.trace_ctx.vrefs_after_residual_call();
    }
    // `vinfo.tracing_after_residual_call(virtualizable)`
    // heap half: a cleared token means the callee forced the virtualizable —
    // the frame escaped, the trace must abort (pyjitpl.py
    // `SwitchToBlackhole(Counters.ABORT_ESCAPE, raising_exception=True)`).
    // The interpreter resumes from the live frame, which the callee's force
    // path made heap-authoritative. Reload the walk shadow before aborting so
    // later cleanup cannot restore pre-force fields.  An intact token is
    // cleared back to TOKEN_NONE.
    if let Some(obj) = vable_obj_root.as_ref() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let forced = unsafe { info.tracing_after_residual_call(**obj as usize as *mut u8) };
        if let Some(depth) = vable_root_depth {
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
        }
        // Match `LiveLastInstrGuard` by restoring both halves after the
        // residual.  A force is excluded because the callee made the heap
        // authoritative and its escape walk reloads the shadow; restoring a
        // pre-force value would violate `_opimpl_setfield_vable`'s equality
        // (`pyjitpl.py`) checked by
        // `check_synchronized_virtualizable` (`pyjitpl.py`).
        if !forced && let Some((prev_op, prev_val)) = saved_last_instr_shadow.take() {
            crate::trace_opcode::mirror_vable_static_to_boxes(
                ctx.trace_ctx,
                "last_instr",
                prev_op,
                prev_val,
            );
        }
        if forced {
            disarm_folded_inline_callee_after_escape(ctx, op_pc)?;
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[force-shape] helper={helper:?} rtype={:?} writes_live={writes_live_heap} \
                     reentrant={reentrant_residual} commit={} entered_frame={} bh={} \
                     fs={} subwalk={} bridge={} wf={:?} wa={:?} wi={:?} rnd={} fn=0x{:x} \
                     pc={op_pc}",
                    call_descr.result_type(),
                    match committed_frame_escape_pc() {
                        None => "none",
                        Some((_, EscapeResumeKind::Exact)) => "exact",
                        Some((_, EscapeResumeKind::RerunsOpcode)) => "reruns",
                    },
                    heap_write_odometer_before
                        .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before),
                    blackhole_result.is_some(),
                    ctx.session.borrow().framestack.len(),
                    ctx.fbw_mode.inline_subwalk,
                    ctx.trace_ctx.is_bridge_trace,
                    ei._write_descrs_fields.as_ref().map(Vec::len),
                    ei._write_descrs_arrays.as_ref().map(Vec::len),
                    ei._write_descrs_interiorfields.as_ref().map(Vec::len),
                    ei.has_random_effects(),
                    func_ptr as usize,
                );
            }
            // The escaping residual either wrote live heap outside the
            // journals, or entered a user Python frame whose body may have
            // committed irreversible effects.  Either way a committed escape
            // resume would re-execute this opcode and so re-apply that write
            // / re-run that body.  Withdraw the commit AND restore the
            // pre-flush frame so the legacy replay re-enters pristine
            // pre-walk state (the flush moved the live frame mid-iteration;
            // replaying on top of it loses journal-rolled-back effects and
            // re-runs partial state).
            if writes_live_heap
                || heap_write_odometer_before
                    .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before)
            {
                cancel_committed_frame_escape_pc();
                // The restore is DEFERRED to the walk-end epilogue, which is
                // where the legacy replay is actually chosen.  Undoing it here
                // also undid `write_boxes`' locals materialization on the
                // resume-PAST path below — the blackhole then continued into a
                // frame whose fastlocals were back to their unwritten nulls
                // (`x = 1; a = sys._getframe(0); return x` returned NULL).
                // The capture stays armed; `run_perfn_walk` restores it only
                // when neither the force blackhole nor a committed escape pc
                // takes over the continuation.
                mark_escape_flush_undo_pending();
            }
            // The writes-live-heap force shape cannot safely resume AT
            // this opcode, but a top-level one-frame walk can resume PAST it
            // through the existing blackhole when every live color has a
            // concrete value.  Build here, before WalkContext's concrete banks
            // disappear, and consume only in run_perfn_walk's VableEscaped
            // epilogue.  Every condition is additive under an exact opt-in;
            // failure leaves the pre-existing escape/replay path untouched.
            let odometer_unchanged = heap_write_odometer_before
                .is_some_and(|before| pyre_interpreter::call::frame_entry_count() == before);
            if fbw_debug_abort_enabled() && ctx.fbw_mode.inline_subwalk {
                eprintln!(
                    "[s2-gate] inline_subwalk fs={} writes_live={} odo_unchanged={} \
                     committed_none={} not_bridge={} bh_result_some={} sym_nonnull={}",
                    ctx.session.borrow().framestack.len(),
                    writes_live_heap,
                    odometer_unchanged,
                    committed_frame_escape_pc().is_none(),
                    !ctx.trace_ctx.is_bridge_trace,
                    blackhole_result.is_some(),
                    !ctx.fbw_mode.snapshot_sym.is_null(),
                );
            }
            // `vable_after_residual_call` has exactly one continuation for an
            // escape: `load_fields_from_virtualizable()` then
            // `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)`.  It
            // never rewinds — the residual has run, its result is in hand, and
            // the blackhole picks up PAST the call.  A top-level one-frame walk
            // reaches that continuation whenever every live color has a
            // concrete value, so build the resume-past image here, before
            // WalkContext's concrete banks disappear, and consume it in
            // run_perfn_walk's VableEscaped epilogue.  Both the latch
            // (non-bridge, empty framestack, not an inline sub-walk, resolvable
            // snapshot sym) and the adopt (`try_adopt_single_frame_blackhole` →
            // `apply_single_frame_blackhole_crn`, which validates every mapped
            // color and every live operand-stack slot before writing anything)
            // decline to the pre-existing escape/replay path on any unmet
            // condition, so this only ever replaces a replay that would have
            // produced the same state.
            //
            // Neither `writes_live_heap` nor the odometer gates it.  Both
            // describe hazards of RE-RUNNING the escaping opcode, which is what
            // the rewind latch and the legacy replay do; this leg resumes past
            // the opcode with the residual's result (or its raise) spliced in,
            // so the residual runs exactly once no matter what it wrote or
            // whether it entered a Python frame.  `writes_live_heap` is a static
            // helper-kind list that does not even contain `CallFunctionEx`,
            // whose callee is arbitrary user code; gating on it left exactly the
            // frames whose replay is unsound — a re-entry guard plus a
            // non-idempotent store ahead of the escaping call — on the replay
            // path.  Upstream has no counterpart to either gate: ABORT_ESCAPE
            // goes straight to `run_blackhole_interp_to_cancel_tracing`
            // (`pyjitpl.py` → `blackhole.py convert_and_run_from_pyjitpl`),
            // which converts the framestack and runs FORWARD, never replays.
            //
            // An already-committed escape pc does not gate it either.  The
            // withdraw above cancels the commit but only MARKS the frame
            // restore pending, so the image is built over the forced frame it
            // describes; `run_perfn_walk`'s epilogue runs the restore later and
            // only when neither this image nor an escape pc claimed the
            // continuation.  Requiring `committed_frame_escape_pc().is_none()`
            // here instead left the shapes that DID commit — a re-entry guard
            // plus a non-idempotent store ahead of the escaping call — on the
            // replay path.
            if let Some((resume_pc, result_bank, result_color)) = blackhole_result
                && !ctx.fbw_mode.snapshot_sym.is_null()
            {
                let (lastop_result, last_exc_value, raising_exception) = match exec_result {
                    Ok(value) => (
                        (result_bank != 'v').then_some((result_bank, result_color, value)),
                        0,
                        false,
                    ),
                    Err(exc) => (None, exc, true),
                };
                // Same non-bridge latch as the escape-flush commit above:
                // a bridge walk never adopts this image (`run_perfn_walk`
                // epilogue is skipped).
                if ctx.session.borrow().at_portal()
                    && !ctx.fbw_mode.inline_subwalk
                    && !ctx.trace_ctx.is_bridge_trace
                {
                    let jitcode = unsafe {
                        let sym = &*ctx.fbw_mode.snapshot_sym;
                        (!sym.jitcode().is_null())
                            .then(|| (&(*sym.jitcode()).payload).jitcode.clone())
                    };
                    if let Some(miframe) = jitcode.and_then(|jitcode| {
                        build_single_frame_miframe(ctx, jitcode, resume_pc, lastop_result)
                    }) {
                        // `ctx` is the root walk on this arm
                        // (`framestack.is_empty() && !inline_subwalk`).  It
                        // stops inside the residual call, and the resumed
                        // blackhole can reach a `getarrayitem_vable_r` that
                        // reloads an operand from the virtualizable array, so
                        // publish the root stack from the walker's mirror.
                        let mirror_stack = capture_vstack_mirror_image(ctx, "escape-flush");
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[latch-accept] origin=escape-flush single-frame pc={} \
                                 jitcode={:?} index={:?}",
                                miframe.pc,
                                miframe.jitcode.name(),
                                miframe.jitcode.try_index(),
                            );
                        }
                        FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
                            *slot.borrow_mut() = Some(LatchedSingleFrameBlackhole {
                                miframe,
                                last_exc_value,
                                raising_exception,
                                mirror_stack,
                                origin: "escape-flush",
                            });
                        });
                    }
                // An inlined sub-walk adopts the multi-frame blackhole image.
                // The build side (`build_multi_frame_miframe`, the input-arg
                // `_resref` seed, and the getfield-chain `recover_ref_value`)
                // reconstructs the frame stack; the resume side
                // (`drive_multi_frame_blackhole` →
                // `convert_and_run_from_pyjitpl`) publishes each level as the
                // chain reaches it. The blast radius is exactly
                // `inline_subwalk` at a vable escape: this is an `if`/`else
                // if`, and the single-frame arm requires
                // `framestack.is_empty() && !inline_subwalk`.
                //
                // This path was once gated because the walker executes
                // residuals concretely while an inline push does not run the
                // interpreter's call sequence. `ec.topframeref` therefore
                // named the CALLER while an inlined callee body ran, so a
                // `sys._getframe` that was itself the escaping residual read
                // the wrong frame at walk time. Adopting committed that answer
                // where legacy escape/replay discarded it. A `sys._getframe`
                // executed later, inside the blackhole, was always correct.
                // `walker_ec_enter` / `walker_ec_leave` (the port of
                // `executioncontext.py enter`) publish the callee frame at
                // the inlined-call push, which closed the gap.
                // `synth/getframe_while_escaping_read_frame_identity` is the
                // regression guard.
                } else if ctx.fbw_mode.inline_subwalk
                    && let Some(framestack) = build_multi_frame_miframe(
                        ctx,
                        resume_pc,
                        InnermostMiframeBuild::LiveMarker(lastop_result),
                        "escape-flush",
                    )
                {
                    // Frame 0's operand stack needs the same publication the
                    // single-frame arm above gives its own, and for the same
                    // reason: a root walk keeps its virtualizable symbolic, so
                    // nothing the walk pushed or popped reached the live
                    // frame's slot array and it still holds the stack from
                    // before the walk began.  The mirror has to come from the
                    // paused caller image rather than `capture_vstack_mirror_image`,
                    // because `ctx` is the innermost callee on this arm and its
                    // own vstack mirror names that frame, not frame 0.
                    //
                    // Leaving the stack alone resumes the blackhole on the
                    // stale one.  The witness is a nested `for` whose walk
                    // crossed the inner loop's exhaust: the fresh inner
                    // iterator the outer FOR_ITER built exists only in the
                    // walk, the resumed frame finds the previous pass's
                    // exhausted iterator on TOS, and the inner loop ends one
                    // iteration in — leaving every later `i` one pass ahead of
                    // the values the body reads.
                    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
                        *slot.borrow_mut() = Some(LatchedMultiFrameBlackhole {
                            framestack,
                            last_exc_value,
                            raising_exception,
                            mirror_stack: capture_root_parent_resume_stack(ctx),
                            origin: "escape-flush",
                        });
                    });
                }
            }
            // On a kept commit the undo stays armed: the abort epilogue
            // consumes it — discard on adoption, restore when the flush is
            // not adopted.  On the cancel arm the restore is only MARKED
            // pending, so no restore has run by the time this refresh does:
            // the reload is `load_fields_from_virtualizable()`'s either way,
            // i.e. the FORCED frame.  A ladder leg that needs PRE-walk shadow
            // state after a cancelled commit has to re-read it after the
            // walk-end restore, not here.
            ctx.trace_ctx.reload_virtualizable_boxes_from_heap();
            // Read the pre-call image back out of the root slots: a collection
            // inside the residual forwarded them alongside the frame's array,
            // so these words and `locals_w!` name the same objects.
            let residual_locals_before = residual_locals_roots.take().map(|(roots, base, len)| {
                (0..len)
                    .map(|k| roots.get(base + k))
                    .collect::<Vec<pyre_object::PyObjectRef>>()
            });
            report_post_residual_shadow(
                ctx,
                live_frame_root.current(live_frame),
                live_frame_root.root.as_ref(),
                residual_locals_before.as_deref(),
            );
            if fbw_debug_abort_enabled() {
                // `vable_after_residual_call`'s
                // `debug_print('vable escaped during a call in %s')`: name the
                // callee through `get_name_from_address`, falling back to the
                // raw address.  Which helper forced the virtualizable is the
                // only thing that distinguishes one ABORT_ESCAPE from another,
                // and the trace is gone by the time the abort surfaces.
                let addr = match ctx.trace_ctx.box_value(allboxes[0]) {
                    Some(majit_ir::Value::Int(a)) => a,
                    _ => 0,
                };
                match pyre_interpreter::jit_trace_fnaddrs()
                    .into_iter()
                    .find(|(_, a)| *a == addr)
                {
                    Some((name, _)) => eprintln!(
                        "[fbw-escape] pc={op_pc} vable escaped during a call in ConstClass({name})"
                    ),
                    None => {
                        eprintln!(
                            "[fbw-escape] pc={op_pc} vable escaped during a call in {addr:#x}"
                        )
                    }
                }
            }
            // `pyjitpl.py:3389-3392`: ABORT_ESCAPE is raised with
            // `raising_exception=True` "because we must still have the eventual
            // exception raised (this is normally done after the call to
            // `vable_after_residual_call()`)" — the post-call `execute_raised`
            // bookkeeping still owes its work even though the walk is over.
            // Returning here skips the `Err(bh_exc)` arm below, and with it the
            // half of that bookkeeping that has no upstream counterpart: the
            // shared `bh_*` helper published this raise into the backend
            // `_store_exception` cells as well as `BH_LAST_EXC_VALUE`
            // (`publish_residual_call_exception`), and those cells belong to
            // compiled / blackhole execution, which RPython tracing never
            // touches.  A survivor is read by the next compiled trace's
            // `must_save_exception` guard and delivered as that frame's own
            // raise — an exception surfacing out of a frame that raised
            // nothing.  `BH_LAST_EXC_VALUE` stays as `execute_residual_call`
            // left it (cleared on read): the escape's consumers take the
            // exception from `exec_result` directly, so restoring it would only
            // outlive the walk.
            if exec_result.is_err()
                && let Some(cb) = crate::callbacks::try_get()
            {
                (cb.drain_backend_jit_exc)();
            }
            crate::state::note_vable_escape_abort();
            return Err(DispatchError::VableEscapedDuringResidualCall { pc: op_pc });
        }
    }
    // Not forced, so nothing above consumed the pre-call image.  The reachable
    // shape is the same `f_locals` write-through as the forced arm, one
    // recording later: with the callee inlined the store IS the residual, and
    // `framelocalsproxy_setitem`'s own force is gated on the live frame's
    // `vable_token` -- which the walk arms on its snapshot instead -- so the
    // store lands on the array with no escape raised and no shadow reload.
    // The walk would otherwise carry the box it held before the call all the
    // way into the jump arguments the loop closes on.
    //
    // This runs BEFORE the restore below: the residual's write is still in the
    // live frame here, and the restore keeps it anyway (it holds back exactly
    // the slots the flush did not write).
    if let Some((roots, base, len)) = residual_locals_roots.take() {
        let before: Vec<pyre_object::PyObjectRef> = (0..len).map(|k| roots.get(base + k)).collect();
        report_post_residual_shadow(
            ctx,
            live_frame_root.current(live_frame),
            live_frame_root.root.as_ref(),
            Some(&before),
        );
    }
    // A flush that ran without a forced abort (an unarmed token or a missing
    // vable root) must not leak the moved frame into the continuing walk.
    restore_escape_flush_undo();
    // #57 Option C (Finding #1, R1): a residual that is not provably
    // side-effect-free has now EXECUTED AFTER the in-flight FOR_ITER consume —
    // whether it returned a value (Ok) or raised (Err).  The store/append
    // journals roll their entries back on abort (so a body re-run re-applies
    // them once), but a mutation outside those journals (a dict
    // `store_subscr_fn`, an `obj.attr = …` `store_attr_fn`, a `list.extend`,
    // a `del o[k]`, a name/global/deref store …) cannot be undone — delivering
    // the in-flight item and re-running the body would double it.  Flag it so
    // `fbw_foriter_inflight_take` refuses delivery (the legacy drop-on-abort
    // fallback) instead of doubling.
    // The user-frame signal: the residual's concrete execution entered a user
    // Python frame (the odometer advanced), so a value-returning
    // getter/dunder/module body may have mutated live heap outside the
    // journals — a body effect the Void/helper-tag discriminator misses.
    // `for_mutate`'s `seen.append` resolves its bound method at the C level (no
    // user frame), so its snapshot is unchanged and it still DELIVERS.
    //
    // The marking runs on BOTH arms.  A getter that mutates and THEN raises
    // (`for_prop_raise_abort`: `Obj.hits += 1; raise`, caught locally, walk
    // continues, later abort) takes the Err arm but still committed the
    // irreversible effect and still bumped the eval-loop entry odometer; if it
    // marked only on Ok, `fbw_foriter_inflight_take` would see no signal and
    // DELIVER, re-running the getter and DOUBLING `Obj.hits`.  The
    // `for_iter_next` consume itself is exempt (`provably_side_effect_free`
    // leaves both `body_effect_candidate` false and `user_frame_snapshot`
    // None), so a raising `__next__` never self-flags.
    //
    // The odometer bumps at frame ENTRY, so `entered_user_frame` cannot tell a
    // user frame that raised AFTER mutating from one that raised BEFORE: a
    // getter that raises before committing anything is also refused here.  That
    // is a harmless conservative DROP (the legacy bypass still runs the
    // iteration once), never a double — refusing a non-mutating raise costs
    // nothing but the never-double guarantee.
    let entered_user_frame = user_frame_snapshot
        .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before);
    // The consume itself is not a body effect.  Marking it would make
    // `fbw_foriter_inflight_take` take the `REFUSED_EFFECT` arm and keep a
    // journaled cursor advanced, dropping that item.  The fresh capture below
    // still starts with a clear body-effect flag.
    if helper != majit_ir::RuntimeHelperKind::ForIterNext
        && (body_effect_candidate || entered_user_frame)
    {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] body effect committed since consume (helper={helper:?} \
                 extraeffect={:?} result_type={:?} write_discriminator={body_effect_candidate} \
                 entered_user_frame={entered_user_frame})",
                ei.extraeffect,
                call_descr.result_type(),
            );
        }
        fbw_mark_foriter_body_effect_since_consume();
    }
    // gh#467: bump the concrete-heap-write odometer for any residual that is not
    // provably side-effect-free and either writes live heap (a Void / mutator-
    // tagged store the store/append journals do not cover) or entered a user
    // Python frame (a value-returning getter/dunder body that may have mutated).
    // The inline abort-forward-flush gate snapshots this at the CALL and refuses
    // the forward flush if a callee sub-walk moved it — re-executing the CALL
    // would double the effect.
    // `writes_gc_liveness_root_only` (defined with the R1 discriminator above,
    // which reads the same binding): an uncommitted walk has nothing to undo
    // for the exception carrier's clear, so the region it hands back is not
    // irreversible.
    if !provably_side_effect_free
        && !is_idempotent_gc_barrier
        && !writes_gc_liveness_root_only
        && (writes_live_heap
            || heap_write_odometer_before
                .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before))
    {
        // `committed=false effects>0` is the double-apply signature, but the
        // odometer alone does not say WHICH residual made the region
        // irreversible.  Name the bumping call the way the escape diagnostic
        // names a forcing one.
        if fbw_debug_abort_enabled() {
            let name = pyre_interpreter::jit_trace_fnaddrs()
                .iter()
                .find(|(_, a)| *a == func_ptr as i64)
                .map(|(n, _)| *n);
            eprintln!(
                "[fbw-effect] pc={op_pc} helper={helper:?} rtype={:?} extraeffect={:?} \
                 writes_live={writes_live_heap} entered_frame={} fn={:?}/{:#x}",
                call_descr.result_type(),
                ei.extraeffect,
                heap_write_odometer_before
                    .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before),
                name,
                func_ptr as usize,
            );
        }
        fbw_bump_executed_effect("residual");
    } else if helper == majit_ir::RuntimeHelperKind::ForIterNext && !for_iter_journaled {
        // Unjournaled `__next__` does not write through `writes_live_heap` and
        // a dict/set/itertools iterator enters no user frame, so the arm above
        // would not see it.  The odometer is what `fbw_decline_inline_callee`
        // and the bridge drain consult before `try_adopt_blackhole`.
        fbw_bump_executed_effect("residual");
    }
    match exec_result {
        Ok(result_i64) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // #57 (Finding #1): the in-place int-list extend committed; journal
            // its pre-extend length so an aborting walk's rollback rewinds it and
            // the deliver re-applies it exactly once.  `result_i64 == lhs`
            // confirms the in-place mutation (list `__iadd__`/`__imul__` return
            // self) rather than a fresh-object op that merely shared the slot.
            if let Some((lhs, len_before, allocated_before)) = inplace_list_journal {
                if result_i64 as usize == lhs as usize {
                    fbw_list_journal_push_append(lhs, len_before, allocated_before);
                }
            }
            // A folded-decline `jit_list_append` fall-through (realloc-boundary
            // append) mutated `list` in place; journal its pre-append length so
            // the abort rollback rewinds it and the deliver re-applies exactly
            // once.  The append always mutates its receiver (void `0` result),
            // so no in-place `result == lhs` re-check is needed.
            if let Some((list, len_before, allocated_before)) = list_append_journal {
                fbw_list_journal_push_append(list, len_before, allocated_before);
            }
            // pyjitpl.py `result_box.value = result` analogue — stamp
            // the recorded OpRef with the executed concrete so downstream
            // `concrete_of_opref` / `box_value` consumers see the folded
            // value. An executed void helper has nothing to stamp.
            // `pyjitpl.py:2049-2068` checks forced virtual refs before
            // recording the selected CALL operation.  The assembler-call
            // walker uses `OpRef::NONE` while concrete-executing so
            // `vrefs_after_residual_call` can emit VIRTUAL_REF_FINISH before
            // CALL_ASSEMBLER, then stamps the newly recorded call itself.
            if !recorded.is_none() {
                match call_descr.result_type() {
                    majit_ir::Type::Int => {
                        ctx.trace_ctx
                            .set_opref_concrete(recorded, majit_ir::Value::Int(result_i64));
                    }
                    majit_ir::Type::Ref => {
                        ctx.trace_ctx.set_opref_concrete(
                            recorded,
                            majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
                        );
                        // `pyjitpl.py` keeps the residual CALL result in the
                        // MIFrame Ref register consumed by the enclosing
                        // Python opcode.  This direct-record path bypasses
                        // `write_ref_reg`, so publish the same last-Ref value
                        // to the operand-stack mirror explicitly.  Without
                        // it a CALL returning None leaves a NULL hole at the
                        // following POP_TOP and an abort cannot flush past an
                        // already-executed mutation such as `seen.add(w)`.
                        ctx.frame_state.borrow_mut().vstack_last_ref = recorded;
                    }
                    majit_ir::Type::Float => {
                        ctx.trace_ctx.set_opref_concrete(
                            recorded,
                            majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
                        );
                    }
                    majit_ir::Type::Void => {}
                }
            }
            // #57 Option C (capture): this residual is the FOR_ITER advance
            // (`for_iter_next`) — it just advanced the real shared heap
            // iterator (an irreversible side effect with no journal undo).
            // Stash the consumed item + the FOR_ITER body pc (the continue
            // arm's `py_pc + 1` fallthrough) so an aborting walk can DELIVER
            // the in-flight iteration to the live frame instead of dropping
            // it.  A null result is the exhaustion arm (no item, no body
            // runs) — nothing to deliver, leave the stash empty.
            if call_descr.get_extra_info().runtime_helper
                == majit_ir::RuntimeHelperKind::ForIterNext
                && result_i64 != 0
            {
                // The body pc is the FOR_ITER continue-arm fallthrough — the
                // Python bytecode pc of the FOR_ITER opcode plus one (matching
                // `opcode_for_iter`'s `next_instr() == opcode_pc + 1`).
                //
                // Finding #2: derive it from the residual op's OWN JitCode pc
                // (`op_pc`) mapped to its containing Python opcode, NOT the
                // walk-ENTRY coordinate (`entry_py_pc + 1`).  The entry
                // coordinate equals the FOR_ITER fallthrough only when FOR_ITER
                // is the loop-header / walk-entry opcode; a second/nested
                // FOR_ITER reached deeper in a traced body has its own
                // `op_pc`, so the entry coordinate would point at the WRONG
                // body and deliver to the wrong pc.  The fallback (no outer
                // full-body sym / metadata) keeps the entry coordinate, which
                // is correct for the loop-header FOR_ITER.
                let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
                    .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
                fbw_foriter_inflight_capture(
                    result_i64 as usize as pyre_object::PyObjectRef,
                    body,
                    for_iter_journaled,
                );
                // #73/#267: the item lands on the operand-stack TOS through the
                // codewriter's `pin!` slot binding (FOR_ITER lowering), not a
                // `setarrayitem_vable_r` push, and the residual result is
                // stamped via `set_opref_concrete`, not `write_ref_reg` — so
                // neither mirror chokepoint sees the item and `vstack_last_ref`
                // still holds whatever inner box the ForIterNext produced.  Seed
                // it with the item OpRef so the FOR_ITER boundary
                // (`ResultToTos`) places the item, not a stale box, on the new
                // TOS.  This runs for every `ForIterNext` residual once it
                // returns, placing the item OpRef on the new TOS for the
                // FOR_ITER `ResultToTos` boundary.
                ctx.frame_state.borrow_mut().vstack_last_ref = recorded;
                if fbw_debug_abort_enabled() {
                    let item = result_i64 as usize as pyre_object::PyObjectRef;
                    let intval = if unsafe { pyre_object::pyobject::is_int(item) } {
                        Some(unsafe { pyre_object::w_int_get_value(item) })
                    } else {
                        None
                    };
                    eprintln!(
                        "[fbw-foriter] capture item=0x{:x} intval={intval:?} foriter_pc={} body={body:?} \
                         store_journal_len={} append_journal_len={} unjournaled={}",
                        result_i64 as usize,
                        ctx.entry_py_pc(),
                        fbw_store_journal_len(),
                        FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()),
                        fbw_has_unjournaled_effect(),
                    );
                }
            }
        }
        Err(bh_exc) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // `metainterp.execute_raised(exception, constant=False)`
            // analogue — seed the standing exception
            // state so downstream walker chain (`reraise/`,
            // `last_exc_value/>r`, `handle_possible_exception` guard
            // emission) sees a non-null `last_exc_value` and routes
            // through the GuardException path.
            //
            // `execute_residual_call` consumes `BH_LAST_EXC_VALUE` when it
            // returns this `Err`.  Do not republish it: RPython transfers the
            // value into `MetaInterp.last_exc_value` exactly once, and the
            // walker shadow below is that owner.  Do not republish the
            // consumed value into TLS as well.  An
            // uncaught raise is carried by the trace's exception FINISH; an
            // aborted walk re-executes the live opcode.
            let bh_exc_box = ctx.trace_ctx.const_ref(bh_exc);
            ctx.set_last_exc_value(
                bh_exc_box,
                ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef),
            );
            // `execute_raised(..., constant=False)`:
            // a residual exception has not had its class proven by a guard yet.
            ctx.fbw_mode.class_of_last_exc_is_const = false;
            // `execute_raised` records the raise into `last_exc_value`
            // (above) only.  The shared `bh_*` residual helper also
            // published into the backend `_store_exception` cells
            // (`publish_residual_call_exception`), but RPython tracing never
            // touches them — they belong to compiled / blackhole execution.
            // Drain them so an aborted trace's snapshot-side raise cannot
            // leak into the live frame's re-run, where compiled
            // `GUARD_NO_EXCEPTION` would read it as a spurious pending
            // exception (the standing exception lives in `last_exc_value`
            // for the walk's `reraise` / `catch_exception` consumers).
            if let Some(cb) = crate::callbacks::try_get() {
                (cb.drain_backend_jit_exc)();
            }
        }
    }
    Ok(ResidualExecOutcome::Executed(exec_result))
}

/// `pyjitpl.py MetaInterp.direct_call_release_gil` port.
/// Sub-case of the forces-virtual-or-virtualizable branch
/// (`pyjitpl.py` `if effectinfo.is_call_release_gil()`): when the
/// descr's `call_release_gil_target` is a non-NULL `(realfuncaddr,
/// saveerr)` pair, the recorded trace op is `CALL_RELEASE_GIL_*`
/// with a re-shaped arglist:
///
/// ```text
///     realfuncaddr, saveerr = effectinfo.call_release_gil_target
///     funcbox = ConstInt(adr2int(realfuncaddr))
///     savebox = ConstInt(saveerr)
///     opnum   = rop.call_release_gil_for_descr(calldescr)
///     return self.history.record_nospec(
///         opnum, [savebox, funcbox] + argboxes[1:], ..., calldescr)
/// ```
///
/// `argboxes[0]` (the original funcbox) is replaced by the descr's real
/// target address, with `savebox` (`saveerr`) prepended.  The pyre-jit-
/// trace `allboxes` from [`build_allboxes`] starts with `funcptr` at
/// index 0 and the user-side arguments from index 1 onwards, matching
/// upstream's `argboxes[0] = funcbox` convention, so the slice rebuild
/// is `[savebox, funcbox_real] + allboxes[1..]`.
///
/// Mirror of `majit-metainterp/src/pyjitpl.rs
/// direct_call_release_gil` for the pyre-jit-trace dispatcher layer.
/// The two-frame-layer parity (majit `do_residual_call` and
/// pyre-jit-trace `dispatch_residual_call_*`) both implement the same
/// `pyjitpl.py` sub-case independently because the layers
/// receive different argument shapes.  `descr` is consumed (move) into
/// `record_op_with_descr` so the caller must `clone()` it before
/// calling if it needs the original after this returns.
///
/// Also emits the two guards the outer forces branch demands
/// (`pyjitpl.py GUARD_NOT_FORCED` unconditionally,
/// `pyjitpl.py GUARD_NO_EXCEPTION` when
/// `check_can_raise(False)` is true) — keeping guard emission inside
/// this helper means the dispatcher early-returns after a single call.
///
/// **`'r'` bank not supported.**  RPython
/// `resoperation.py call_release_gil_for_descr` has no
/// `CALL_RELEASE_GIL_R` arm (commented out as `# no such thing`),
/// and `:1462 is_call_release_gil` excludes `CALL_RELEASE_GIL_R`
/// from the predicate.  This helper panics on `dst_bank == 'r'` —
/// the closest behaviour to upstream's missing branch is fail-fast,
/// since silently routing to a non-existent OpCode would record an
/// IR op the optimizer / backend cannot consume.  Generic codewriter
/// `emit_residual_call` sites do not manufacture release-gil EIs via
/// `effect_info_for_call_flavor`; release-gil support is limited to
/// explicit via-target lowering that resolves the real call target
/// before materializing the final calldescr.  The panic is defensive
/// against a future producer that introduces a `'r'`-result release-gil
/// callee without first wiring an upstream `CALL_RELEASE_GIL_R` opcode.
///
/// `'i'` / `'f'` / `'v'` are the three result kinds upstream's
/// `call_release_gil_for_descr` accepts (`resoperation.py`).
/// All three are decoded here so the **opcode selection** matches
/// upstream's three-way result-kind table even though only
/// `dispatch_residual_call_iRd_kind` / `_iIRd_kind` currently route
/// `'i'` and `'r'` (the latter rejected per the panic above).
///
/// **Float / Void coverage is opcode-only, not full reuse.**  A
/// future float / void residual-call dispatcher would still have
/// to extend its own callsite to (a) widen `dst_bank` validation,
/// (b) add the corresponding writeback path to
/// `registers_f` / no-writeback, and (c) thread Float-typed
/// `argbox_types` through `build_allboxes` for the `'f'` arg-list
/// case.  This helper produces the right `OpCode::CallReleaseGil*`
/// once those landed; it does not by itself complete the dispatcher.
/// `pyjitpl.py do_residual_call` parity:
///
/// ```python
/// if effectinfo.oopspecindex == effectinfo.OS_NOT_IN_TRACE:
///     return self.metainterp.do_not_in_trace_call(allboxes, descr)
/// ```
///
/// ```python
/// def do_not_in_trace_call(self, allboxes, descr):
///     self.clear_exception()
///     executor.execute_varargs(self.cpu, self, rop.CALL_N, allboxes, descr)
///     if self.last_exc_value:
///         raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
///                                 raising_exception=True)
///     return None
/// ```
///
/// The check sits at the TOP of `do_residual_call`, ahead of the
/// forces-virtual branch, so `vable_and_vrefs_before_residual_call` and
/// the vref bracket never run for this callee: the trace does not contain
/// the call at all, and there is nothing for a force to have escaped
/// through.  `rop.CALL_N` is a void call and the `return None` gives the
/// caller no result box, so no destination register is written either.
///
/// Concrete execution is [`walker_execute_may_force_boxed_outcome`],
/// which resolves each argbox to its recorded concrete and calls through
/// `majit_metainterp::executor::execute_residual_call` without recording
/// IR — `execute_varargs` off the history, exactly what this arm needs.
/// It answers `None` when some argbox has no concrete to resolve (the
/// funcbox is not a constant, or a Ref argbox carries `NO_CONCRETE`);
/// there is no faithful outcome to invent there, so the typed
/// [`DispatchError::NotInTraceRequiresConcreteExecution`] still surfaces.
/// The same error covers a `>X` destination bank. That is upstream's own
/// invariant, not a pyre restriction: `jtransform.py`'s
/// `jit.not_in_trace` arm refuses to emit the call at all unless
/// `op.result.concretetype is lltype.Void` -- *"jit.not_in_trace()
/// function must return None"* -- so only the `residual_call_*_v`
/// spellings can carry this oopspec, and a destination reaching here
/// names a codewriter invariant violation rather than a shape to guess
/// at.
///
/// `effect_info_for_call_flavor` (`flatten.rs` audit table) still never
/// sets `oopspecindex`, so no production residual call reaches this arm
/// today; the tests drive it directly.
#[inline]
pub(crate) fn do_not_in_trace_call_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    pc: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if ei.oopspecindex != OopSpecIndex::NotInTrace {
        return Ok(None);
    }
    if dst_bank != 'v' {
        return Err(DispatchError::NotInTraceRequiresConcreteExecution { pc });
    }
    // `do_not_in_trace_call`: `self.clear_exception()` before the call, so a
    // raise the callee leaves behind is unambiguously its own.
    ctx.clear_last_exc_value();
    let Some(outcome) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr) else {
        return Err(DispatchError::NotInTraceRequiresConcreteExecution { pc });
    };
    match outcome {
        // Normal completion records nothing: upstream returns `None` from
        // `do_not_in_trace_call` and `do_residual_call` passes that straight
        // out, so no ResOperation joins the history.
        Ok(_) => Ok(Some(DispatchOutcome::Continue)),
        // "cannot trace this! it raises, so we have to follow the
        // exception-catching path, but the trace doesn't contain the call at
        // all."
        Err(_) => Ok(Some(DispatchOutcome::SwitchToBlackhole {
            reason: majit_metainterp::counters::ABORT_ESCAPE,
            raising_exception: true,
        })),
    }
}

/// IR-recording portion of `pyjitpl.py
/// vable_and_vrefs_before_residual_call`.  Records
/// `FORCE_TOKEN + SETFIELD_GC(vable_token_descr)` whenever the
/// jitdriver has a standard virtualizable registered for the current
/// frame.  RPython structure:
///
/// ```text
/// def vable_and_vrefs_before_residual_call(self):
///     self.vrefs_before_residual_call()                # heap mutation
///     vinfo = self.jitdriver_sd.virtualizable_info
///     if vinfo is not None:
///         virtualizable_box = self.virtualizable_boxes[-1]
///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
///         vinfo.tracing_before_residual_call(virtualizable) # heap mutation
///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)  # IR
///         self.history.record2(rop.SETFIELD_GC, ..., descr=...)     # IR
/// ```
///
/// In pyre, the IR-recording role and the runtime heap-mutation role
/// are split.  This helper carries the IR portion only; the heap
/// halves of the vable token protocol
/// (`vinfo.tracing_before_residual_call(virtualizable)` /
/// `vinfo.tracing_after_residual_call(virtualizable)`) live with
/// the walk that executes the callee:
/// [`try_execute_residual_call_via_executor`] brackets the concrete
/// `execute_residual_call` with both halves of
/// `vable_and_vrefs_before_residual_call` /
/// `vable_after_residual_call` and surfaces
/// [`DispatchError::VableEscapedDuringResidualCall`] on a detected
/// force (pyjitpl.py ABORT_ESCAPE parity).
///
/// This helper records ONLY the IR portion here and never
/// touches the token; the heap-half token protocol is bracketed by
/// [`try_execute_residual_call_via_executor`], keeping the
/// `*token_ptr == 0` assertion in `tracing_before_residual_call`
/// intact.
///
/// The vref halves are bracketed by that same executor:
/// `TraceCtx::vrefs_before_residual_call` stamps every live vref before the
/// call and `vrefs_after_residual_call` turns any the callee forced into a
/// `VIRTUAL_REF_FINISH` + ConstPtr(NULL) box before the CALL op is recorded.
/// They iterate over the vrefs an inlined call's `enter` pushed
/// (`inline_call.rs::walker_ec_enter`), so a residual that hands an inlined
/// callee's frame to Python stops the trace from claiming it is still virtual.
pub(crate) fn walker_vable_and_vrefs_before_residual_call(ctx: &mut TraceCtx) {
    // pyjitpl.py: vinfo = self.jitdriver_sd.virtualizable_info;
    //                       if vinfo is not None:
    let Some(vable_ref) = ctx.standard_virtualizable_box() else {
        return;
    };
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    // pyjitpl.py: force_token + SETFIELD_GC vable_token_descr
    let force_token = ctx.force_token();
    ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
}

/// The python pc `jit_pc` names in the inline callee's OWN code, or `None`
/// when this walk is not inside an inline callee.  A callee `jit_pc` has no
/// meaning in the outer jitcode's py_pc tables, so every consumer that writes
/// a pc onto the callee's frame has to go through the callee's own metadata.
pub(super) fn inline_callee_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) -> Option<u32> {
    let consts = ctx.inline_callee_consts?;
    let pjc = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)?;
    Some(crate::py_coord::containing_py_pc_for_jitcode_pc(
        &pjc.metadata,
        jit_pc,
    ))
}

fn live_py_pc_from_snapshot<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) -> Option<u32> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return None;
    }
    let jc = unsafe { &*sym.jitcode() };
    Some(crate::py_coord::containing_py_pc_for_jitcode_pc(
        &jc.payload.metadata,
        jit_pc,
    ))
}

/// Convenience wrapper for [`walker_vable_and_vrefs_before_residual_call`].
/// Kept as a thin pass-through so the dispatcher call sites stay
/// readable; collapses to direct `walker_*` once the dispatchers
/// inline.
pub(crate) fn maybe_record_inline_callee_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    let Some(consts) = ctx.inline_callee_consts else {
        return;
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
        return;
    };
    let frame_reg = pjc.metadata.portal_frame_reg as usize;
    let Some(callee_frame) = ctx.registers_r.get(frame_reg) else {
        return;
    };
    if callee_frame == OpRef::NONE {
        return;
    }

    let callee_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, jit_pc);
    let last_instr = ctx.trace_ctx.const_int(callee_py_pc as i64);
    let last_instr_descr = crate::descr::pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, last_instr],
        last_instr_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(callee_frame, last_instr_idx, last_instr);
}

pub(super) fn maybe_publish_inline_callee_last_instr_concrete<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    let Some(consts) = ctx.inline_callee_consts else {
        return;
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
        return;
    };
    let frame_reg = pjc.metadata.portal_frame_reg as usize;
    let callee_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, jit_pc);
    // The walker is the recording-time interpreter.  Keep the concrete
    // MIFrame in step with the emitted store just as
    // `fbw_publish_exit_last_instr` does for the portal frame; otherwise a
    // residual observer during this walk records the constructor's `-1`
    // sentinel and bakes that answer into the compiled trace.  Inline frames
    // are owned by this walk (and may already have escaped), so this write is
    // not speculative — unlike the portal frame's, it is never rolled back for
    // replay.
    //
    // It is still only half a coordinate.  A reader takes its next opcode from
    // `last_instr + 1` and its operand stack from `valuestackdepth`; the walk
    // keeps the operand stack symbolic, so `valuestackdepth` stays at whatever
    // the frame was built with while `last_instr` advances per opcode.
    // `capture_frame_scalars` and `correct_resume_vsd` move the two words
    // together; this site moves one, and nothing downstream re-pairs them.
    // Pairing it here with the forward analysis's depth at `callee_py_pc + 1`
    // was measured and does not fix
    // `bench/synth/_pending/loop_callee_for_header_resume.py`; it does not fix
    // gh#1444 either, whose wrong answer is a depth->=2 multi-frame blackhole
    // adopt (`PYRE_FBW_MULTIFRAME_DEPTH=1` answers correctly).  So the split is
    // a recorded gap rather than either one's cause.
    if let Some(ConcreteValue::Ref(frame_ptr)) = ctx
        .frame_state
        .borrow()
        .concrete_registers_r
        .get(frame_reg)
        .copied()
    {
        if !frame_ptr.is_null() {
            unsafe {
                (*(frame_ptr as *mut pyre_interpreter::PyFrame)).last_instr = callee_py_pc as isize;
            }
        }
    }
}

pub(super) fn record_and_publish_inline_callee_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    maybe_record_inline_callee_last_instr(ctx, jit_pc);
    maybe_publish_inline_callee_last_instr_concrete(ctx, jit_pc);
}

pub(crate) fn disarm_folded_inline_callee_after_escape<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    let Some(consts) = ctx.inline_callee_consts else {
        return Ok(());
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
        return Ok(());
    };
    let state = ctx.frame_state.borrow();
    let Some(shadow) = state.callee_shadow.as_ref() else {
        return Ok(());
    };
    if shadow.fold_frame_reg == u16::MAX || shadow.frame_box == OpRef::NONE {
        return Ok(());
    }
    let frame_reg = pjc.metadata.portal_frame_reg as usize;
    let Some(callee_frame) = ctx.registers_r.get(frame_reg) else {
        return Ok(());
    };
    if callee_frame != shadow.frame_box {
        return Ok(());
    }
    let Some(info) = ctx.trace_ctx.virtualizable_info() else {
        return Ok(());
    };
    let Some(fdescr) = info.array_field_descrs().first().cloned() else {
        return Ok(());
    };
    let Some(adescr) = info.array_descrs.first().cloned() else {
        return Ok(());
    };
    let mut slots: Vec<(i64, OpRef, Value)> = shadow
        .opref
        .iter()
        .filter_map(|(&slot, &value)| {
            (slot >= 0).then(|| {
                let concrete = shadow
                    .concrete
                    .get(&slot)
                    .filter(|entry| entry.frame_reg == shadow.fold_frame_reg)
                    .map(|entry| entry.value)
                    .or_else(|| ctx.trace_ctx.concrete_of_opref(value))
                    .unwrap_or(Value::Void);
                (slot, value, concrete)
            })
        })
        .collect();
    slots.sort_by_key(|(slot, _, _)| *slot);
    drop(state);

    for (slot, value, concrete) in slots {
        let index = ctx.trace_ctx.const_int(slot);
        let store = vable_ops::with_replace_frames(ctx, |ctx| {
            let nonstandard =
                vable_ops::walker_nonstandard_virtualizable(ctx, pc, callee_frame, &fdescr)?;
            Ok(ctx.trace_ctx.vable_setarrayitem_checked(
                nonstandard,
                pc,
                callee_frame,
                index,
                slot,
                fdescr.clone(),
                adescr.clone(),
                value,
                concrete,
                false,
            ))
        })?;
        match store {
            VableArrayStore::Stored(_) => {}
            VableArrayStore::OutOfVable => {}
        }
    }
    if let Some(shadow) = ctx.frame_state.borrow_mut().callee_shadow.as_mut()
        && shadow.frame_box == callee_frame
    {
        shadow.fold_frame_reg = u16::MAX;
    }
    Ok(())
}

/// Publish the opcode this walk is inside onto the portal frame before a
/// residual that hands that frame to Python.
///
/// `try_execute_residual_call_via_executor` does this ahead of a may-force
/// call. `dispatch_bytecode` writes `last_instr` before every opcode; a
/// portal loop carries only the merge point's coordinate in its
/// virtualizable, and a reader inside the residual (`f_lineno` from an
/// audit hook) otherwise reports that line. The mirror keeps the
/// tracing-time shadow and live frame on this opcode.
/// `vable_setfield_descr` is the store compiled code executes: the
/// following call receives the frame, so the heap cache flushes the lazy
/// store before the call.
///
/// Returns `None` when this walk is inside an inline callee — that frame's
/// coordinate is [`record_and_publish_inline_callee_last_instr`], and
/// `vstack_cur_pypc` indexes the outer code.
pub(crate) fn publish_portal_executing_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame_op: OpRef,
    frame_ptr: *mut pyre_interpreter::PyFrame,
) -> Option<u32> {
    if current_inline_concrete_frame() != 0 {
        return None;
    }
    let vable_ref = ctx.trace_ctx.standard_virtualizable_box()?;
    if frame_op != vable_ref {
        return None;
    }
    let py_pc = if ctx.fbw_mode.transparent_helper_subwalk {
        ctx.fbw_mode
            .inline_caller_py_pc
            .unwrap_or(ctx.vstack_cur_pypc)
    } else {
        ctx.vstack_cur_pypc
    };
    if ctx.fbw_mode.transparent_helper_subwalk && py_pc == 0 {
        return None;
    }
    let last_instr = ctx.trace_ctx.const_int(i64::from(py_pc));
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "last_instr",
        last_instr,
        majit_ir::Value::Int(i64::from(py_pc)),
    );
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let idx = info.static_field_index_by_name("last_instr")?;
    // Parent-struct descr, matching the publish in
    // `try_execute_residual_call_via_executor`: the vinfo's own static
    // descr numbers fields in vinfo order, which is not PyFrame's.
    let descr = info.static_field_struct_descr(idx);
    ctx.trace_ctx
        .vable_setfield_descr(vable_ref, last_instr, descr);
    if !frame_ptr.is_null() {
        crate::jitcode_dispatch::fbw_note_last_instr_undo(frame_ptr as usize);
        unsafe { (*frame_ptr).last_instr = py_pc as isize };
    }
    Some(py_pc)
}

pub(crate) fn maybe_walker_vable_and_vrefs_before_residual_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    record_and_publish_inline_callee_last_instr(ctx, jit_pc);
    walker_vable_and_vrefs_before_residual_call(ctx.trace_ctx);
}

/// Write a residual_call's recorded result OpRef into the dst register
/// chosen by `dst_bank`. Centralizes the result writeback so the
/// dispatchers can perform it BEFORE recording the
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` guards, matching
/// `pyjitpl.py _opimpl_residual_call*` ordering: the result
/// must populate `registers_*[dst]` before
/// `handle_possible_exception()` captures the guard's `fail_args`,
/// otherwise a raising call surfaces NONE in the slot the resume
/// snapshot reads.
pub(crate) fn write_residual_call_result_to_dst<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    dst_bank: char,
    result: OpRef,
) -> Result<(), DispatchError> {
    // concrete_of_opref shadow write: route the shadow write through `concrete_of_opref`
    // so a CallPure* descr whose argboxes are all constant (do_residual_call
    // path that lands a constant result via the executor.execute_varargs
    // stamp) propagates concrete to the dst slot.  Falls back to Null when
    // the result has no recorded concrete (matches the pre-#75.F shape for
    // every recorded call without an executed concrete result).
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'r' => {
            write_ref_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'i' => {
            write_int_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        // Void variants (`pyjitpl.py opimpl_residual_call_*_v`):
        // the operand layout has no `>X` dst byte and no register slot to
        // populate. The cached / recorded OpRef is dropped on the floor
        // upstream too (the `_call*` body discards the call result for
        // void).
        'v' => {}
        _ => unreachable!("dst_bank validated by caller"),
    }
    Ok(())
}

pub(crate) fn residual_call_helper_kind_in_body(
    body_code: &[u8],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
) -> Option<majit_ir::RuntimeHelperKind> {
    let descr_index = residual_call_descr_index_in_body(body_code, d)?;
    callee_descr_refs
        .get(descr_index)
        .and_then(|descr| descr.as_call_descr())
        .map(|cd| cd.get_extra_info().runtime_helper)
}

/// Return the per-function descriptor-pool index carried by a residual call
/// in a callee jitcode body.  The layouts mirror the residual dispatchers:
/// the descriptor follows the one or two variable-length argument lists.
pub(crate) fn residual_call_descr_index_in_body(body_code: &[u8], d: &DecodedOp) -> Option<usize> {
    let descr_offset = match d.key {
        "residual_call_r_r/iRd>r" | "residual_call_r_i/iRd>i" | "residual_call_r_v/iRd" => {
            let r_len_pc = d.pc + 2;
            let r_len = *body_code.get(r_len_pc)? as usize;
            1 + 1 + r_len
        }
        "residual_call_ir_r/iIRd>r" | "residual_call_ir_i/iIRd>i" | "residual_call_ir_v/iIRd" => {
            let i_len_pc = d.pc + 2;
            let i_width = 1 + *body_code.get(i_len_pc)? as usize;
            let r_len_pc = d.pc + 1 + 1 + i_width;
            let r_width = 1 + *body_code.get(r_len_pc)? as usize;
            1 + i_width + r_width
        }
        _ => return None,
    };
    Some(decode_descr_index(body_code, d, descr_offset))
}

/// Whether this residual is the binding store immediately following the live
/// in-flight `FOR_ITER`.  Both effect accounting and namespace rollback must
/// use this one predicate so their loop-variable classifications cannot drift.
fn is_loop_var_binding_store<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    helper: majit_ir::RuntimeHelperKind,
) -> bool {
    matches!(
        helper,
        majit_ir::RuntimeHelperKind::StoreName | majit_ir::RuntimeHelperKind::StoreGlobal
    ) && fbw_foriter_inflight_top_body_pc()
        .is_some_and(|body_pc| body_pc + 1 == ctx.vstack_cur_pypc as usize)
}

/// Journal the namespace binding a `STORE_NAME` / `STORE_GLOBAL` /
/// `DELETE_NAME` / `DELETE_GLOBAL` residual is about to overwrite, so a walk
/// that does not commit can put it back before the replay re-runs the region
/// (`FBW_NAMESPACE_STORE_JOURNAL`).  Without it the replay reads the walk's own
/// bindings for every statement ahead of the store.
///
/// The namespace is the one the opcode family writes: `*_NAME` binds the
/// frame's `w_locals`, `*_GLOBAL` its `w_globals`.  Only a real dict is
/// journaled — an `exec(src, mapping)` namespace takes its write through the
/// mapping protocol, which a `setitem`/`delitem` undo would not reverse
/// faithfully — and only for the authoritative executor, the walk whose
/// concrete writes are the ones that land.
fn journal_walker_namespace_write<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    helper: majit_ir::RuntimeHelperKind,
    r_args: &[OpRef],
) -> Option<()> {
    use majit_ir::RuntimeHelperKind as K;
    if !ctx.is_authoritative_executor {
        return None;
    }
    // `dict[str] = value` that `try_walker_orthodox_store_subscr` declined
    // (exact list + int index only).  The generic residual still writes the
    // live dict.  Snapshot the displaced item on the namespace journal so a
    // non-commit epilogue puts it back and the entry replay applies the store
    // once.  A dict or str subclass may override `__setitem__`, so only the
    // exact builtin pair is reversible through `w_dict_setitem_wtf8_no_proxy`.
    if helper == K::StoreSubscr {
        return journal_walker_dict_store_subscr(ctx, r_args);
    }
    let binds_name = matches!(helper, K::StoreName | K::DeleteName);
    if !binds_name && !matches!(helper, K::StoreGlobal | K::DeleteGlobal) {
        return None;
    }
    let (&frame_opref, &name_opref) = (r_args.first()?, r_args.get(1)?);
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
    ) = (
        ctx.trace_ctx.box_value(frame_opref),
        ctx.trace_ctx.box_value(name_opref),
    )
    else {
        return None;
    };
    if frame_ptr == 0 || w_name_ptr == 0 {
        return None;
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let namespace = if binds_name {
        frame.get_w_locals()
    } else {
        frame.get_w_globals()
    };
    if namespace.is_null() {
        return None;
    }
    // SAFETY: `namespace` is a live namespace object the frame holds.
    if !unsafe {
        pyre_object::is_module_dict(namespace)
            || pyre_object::py_type_check(namespace, &pyre_object::DICT_TYPE)
    } {
        return None;
    }
    let name = w_name_ptr as pyre_object::PyObjectRef;
    // SAFETY: a namespace opcode's name operand is the code object's interned
    // `co_names` string, and the dict was just type-checked.
    let displaced =
        unsafe { pyre_object::w_dict_lookup(namespace, name) }.unwrap_or(std::ptr::null_mut());
    // The loop-variable binding store is the op at the in-flight FOR_ITER's
    // `body_pc`, recognised the way the R1 in-flight accounting recognises it.
    // An in-flight delivery re-runs the body from that pc, so this one binding
    // is re-applied by the delivery itself and its rollback closes no road.
    let loop_var = is_loop_var_binding_store(ctx, helper);
    fbw_namespace_store_journal_push(namespace, name, displaced, loop_var, None);
    Some(())
}

/// Journal an exact-dict `STORE_SUBSCR` whose key is an exact `str`.
///
/// Rollback restores through [`fbw_namespace_store_journal_rollback`], which
/// reads the key with `w_str_get_wtf8`.  A missing item is stored as null so
/// the rollback deletes the binding the residual is about to create.
///
/// The snapshot lookup must not run a stored key's Python `__eq__`.
/// `ObjectDictStrategy` does, and the residual store plus a rollback would
/// then run it again.  Only an empty dict, a `UnicodeDictStrategy` dict, or a
/// module dict still on `ModuleDictStrategy` is snapshotted.  Rollback
/// restores that entry only while it is still the object this walk stored.
fn journal_walker_dict_store_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    r_args: &[OpRef],
) -> Option<()> {
    let (&dict_op, &key_op) = (r_args.first()?, r_args.get(1)?);
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(dict_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(key_ptr))),
    ) = (
        ctx.trace_ctx.box_value(dict_op),
        ctx.trace_ctx.box_value(key_op),
    )
    else {
        return None;
    };
    if dict_ptr == 0 || key_ptr == 0 {
        return None;
    }
    let dict = dict_ptr as pyre_object::PyObjectRef;
    let key = key_ptr as pyre_object::PyObjectRef;
    // SAFETY: both addresses are live trace-box references.
    let reversible = unsafe {
        (pyre_object::is_exact_type(dict, &pyre_object::DICT_TYPE)
            || pyre_object::is_module_dict(dict))
            && pyre_object::is_exact_type(key, &pyre_object::STR_TYPE)
    };
    if !reversible {
        return None;
    }
    let displaced = unsafe {
        if pyre_object::is_module_dict(dict) {
            if pyre_object::w_module_dict_is_object_strategy(dict) {
                return None;
            }
            let key_text = pyre_object::unicodeobject::w_str_get_value_opt(key)?;
            pyre_object::w_module_dict_getitem_str(dict, key_text).unwrap_or(std::ptr::null_mut())
        } else if pyre_object::w_dict_is_empty_strategy(dict) {
            std::ptr::null_mut()
        } else {
            pyre_object::w_dict_lookup_str_keyed(dict, key, std::ptr::null_mut())?
        }
    };
    let &value_op = r_args.get(2)?;
    let Some(majit_ir::Value::Ref(majit_ir::GcRef(value_ptr))) = ctx.trace_ctx.box_value(value_op)
    else {
        return None;
    };
    if value_ptr == 0 {
        return None;
    }
    fbw_namespace_store_journal_push(
        dict,
        key,
        displaced,
        false,
        Some(value_ptr as pyre_object::PyObjectRef),
    );
    Some(())
}

/// Mirror `opimpl_jit_force_quasi_immutable` for a module namespace's
/// `version?` before an opaque namespace-write residual.
///
/// The walker cannot see the qmut operation inside that residual, so it checks
/// both the installed watcher and the side-effect-free
/// [`pyre_object::celldict::store_would_bump_version`] predicate here.  It
/// forces the watcher and aborts before executing the residual; forward resume
/// then performs the namespace write exactly once.  No `GUARD_ISNULL` is needed
/// because the residual itself calls `notify_version_watchers`.
fn try_walker_force_quasi_immut_namespace_write<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    helper: majit_ir::RuntimeHelperKind,
    r_args: &[OpRef],
) -> Option<()> {
    use majit_ir::RuntimeHelperKind as K;
    let is_store = matches!(helper, K::StoreName | K::StoreGlobal);
    if !is_store && !matches!(helper, K::DeleteName | K::DeleteGlobal) {
        return None;
    }
    let (&frame_opref, &name_opref) = (r_args.first()?, r_args.get(1)?);
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
    ) = (
        ctx.trace_ctx.box_value(frame_opref),
        ctx.trace_ctx.box_value(name_opref),
    )
    else {
        return None;
    };
    if frame_ptr == 0 || w_name_ptr == 0 {
        return None;
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    // A `*_NAME` opcode writes `getorcreatedebug().w_locals`, a `*_GLOBAL` one
    // writes `w_globals`.  Module-level code aliases the two; a class body's
    // locals are the `newdict(module=True)` namespace `build_class` bound, and
    // `celldict.py setitem_str` reaches `version?` through either of them.
    let w_ns = if matches!(helper, K::StoreName | K::DeleteName) {
        frame.get_w_locals()
    } else {
        frame.get_w_globals()
    };
    if w_ns.is_null() {
        return None;
    }
    // A plain `W_DictObject` for globals (`exec(src, {})`,
    // `FunctionType(code, {})`) has no `version?` field at all, so
    // `hook_setfield` (rclass.py) emits no `jit_force_quasi_immutable`
    // for its write and there is nothing to abandon the trace for.
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(w_ns) };
    // `mutatebox.nonnull()` — nothing is watching, so there is nothing to
    // abandon the trace for. The write still runs its own invalidation.
    if !unsafe {
        pyre_object::dictmultiobject::module_dict_strategy_version_qmut_installed(strategy)
    } {
        return None;
    }
    let Some(name) = (unsafe {
        pyre_object::unicodeobject::w_str_get_value_opt(w_name_ptr as pyre_object::PyObjectRef)
    }) else {
        return None;
    };
    let slot = crate::state::module_dict_cell_slot_direct(w_ns, name);
    let bumps = if is_store {
        let &value_opref = r_args.get(2)?;
        let Some(majit_ir::Value::Ref(majit_ir::GcRef(value_ptr))) =
            ctx.trace_ctx.box_value(value_opref)
        else {
            return None;
        };
        if value_ptr == 0 {
            return None;
        }
        let cell = slot.and_then(|s| crate::state::module_dict_cell_value_direct(w_ns, s));
        unsafe {
            pyre_object::celldict::store_would_bump_version(
                cell,
                value_ptr as pyre_object::PyObjectRef,
            )
        }
    } else {
        // `celldict.py delitem` reaches `mutated()` only after a
        // successful removal — `delitem_str` returns early on a missing key.
        slot.is_some()
    };
    if !bumps {
        return None;
    }
    // Match `do_force_quasi_immutable`: invalidate before raising.  Invalidation
    // removes the installed watcher, so the resumed residual cannot force it
    // a second time.
    unsafe { pyre_object::dictmultiobject::module_dict_strategy_force_version_qmut(strategy) };
    Some(())
}

/// The `CodeObject` of the frame the walk is currently in: the innermost
/// inlined callee while a sub-walk is in progress, else the portal frame.
///
/// A sub-walk inherits `FbwWalkMode::snapshot_sym` from its parent, so that
/// field names the walk's outermost frame however deep the descent is. A
/// helper holding a class statement is an ordinary inline candidate, so
/// reading the snapshot root would decide a callee's class statement by its
/// caller's returns. `WalkSession::framestack`
/// carries the callee levels, and its top is the frame executing the opcode;
/// this is the same derivation `ActiveResumeFrame::current` performs, so the
/// gate and the snapshot encoder name one consistent active frame.
fn walker_active_py_code<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<&'static pyre_interpreter::CodeObject> {
    // Bound before the branch so the session borrow ends with this statement.
    let innermost = super::fbw_state::fbw_innermost_inline_callee_key(ctx);
    if let Some(callee_w_code) = innermost {
        let index = crate::state::ensure_jitcode_index(callee_w_code as *const ())?;
        let payload = crate::state::pyjitcode_for_jitcode_index(index)?;
        let code_ptr = payload.code_ptr;
        return (!code_ptr.is_null()).then(|| unsafe { &*code_ptr });
    }
    let sym = ctx.fbw_mode.snapshot_sym;
    if sym.is_null() {
        return None;
    }
    // SAFETY: identical contract to the other `snapshot_sym` readers — the
    // pointer is set only for the lifetime of the full-body walk.
    let sym = unsafe { &*sym };
    if sym.jitcode().is_null() {
        return None;
    }
    let jitcode = unsafe { &*sym.jitcode() };
    let code_ptr = jitcode.payload.code_ptr;
    (!code_ptr.is_null()).then(|| unsafe { &*code_ptr })
}

/// Mirror `opimpl_jit_force_quasi_immutable` for a class body hidden inside the
/// opaque `build_class` residual.
///
/// `build_class` executes the body in a module-strategy dict, whose first
/// `STORE_NAME` installs and forces `version?`; upstream therefore aborts the
/// trace.  Pyre detects the same condition at the residual boundary.  The
/// namespace does not exist yet, so there is no watcher to record or invalidate
/// before the abort.
fn try_walker_force_quasi_immut_class_body<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    ref_operand_offset: usize,
    r_args: &[OpRef],
) -> bool {
    if r_args.len() < 2 {
        return false;
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    // A bit-cast of a small int is a non-null `Ref` and not a callable.
    // `is_build_class_builtin` loads the type word from it.
    let aligned_object = |bits: usize| bits % std::mem::align_of::<usize>() == 0;
    let mut callable = match arg_concretes.first() {
        Some(ConcreteValue::Ref(value)) if aligned_object(*value as usize) => *value,
        _ => std::ptr::null_mut(),
    };
    if callable.is_null()
        && let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(r_args[0])
        && r != majit_ir::GcRef::NO_CONCRETE
        && aligned_object(r.as_usize())
        && r.as_usize() != 0
    {
        callable = r.as_usize() as pyre_object::PyObjectRef;
    }
    if !pyre_interpreter::builtins::is_build_class_builtin(callable) {
        return false;
    }
    // Index 1 is the CALL's null-or-self slot; the builtin's own
    // `&[PyObjectRef]` starts after it, exactly the tail
    // `try_walker_inline_builtin_call` copies into its wrapper array.
    let mut args = Vec::with_capacity(arg_concretes.len().saturating_sub(2));
    for concrete in arg_concretes.iter().skip(2) {
        let ConcreteValue::Ref(value) = *concrete else {
            return false;
        };
        args.push(value);
    }
    if !pyre_interpreter::call::build_class_body_namespace_is_module_dict(&args) {
        return false;
    }
    true
}

/// `nestedscope.py Cell.get` for the LOAD_DEREF residual.
///
/// ```python
/// def get(self):
///     if jit.isconstant(self):
///         if not self.family.ever_mutated:
///             w_res = self._elidable_get()
///             if w_res is not None:
///                 return w_res
///     ...
///     return self.w_value
/// ```
///
/// A cell the trace already holds as a constant — the ordinary freevar read,
/// whose cell reaches the callee through the pinned callable's quasi-immutable
/// `closure` — folds to its contents under a
/// `QUASIIMMUT_FIELD(family, ever_mutated)`, so the read leaves no residual at
/// all.  Every other shape declines and keeps the residual.
///
/// The `w_res is not None` re-check is load-bearing, not defensive: with
/// `ever_mutated` false, the elidable read may hand back a stale unbound `None`
/// for a cell that has since been bound for the first time (that first binding
/// is deliberately not a mutation).  Dropping the check would fold a wrong
/// answer, so a null falls through to the live residual read.
///
/// !! THE CONSTANT ARM FIRES NOWHERE, AND THE FIRST GUARD IS WHY.
/// Measured 2026-08-26 on release dynasm with every early return named through
/// `decline!`: over the whole corpus the census row read `consulted=59
/// fired=0`, and over the 359 `bench/synth` fixtures holding a nested function
/// every one of the 38 declines reported `cell-not-constant`.  No other reason
/// was reported anywhere.  Its ideal shape — a write-once freevar read from a
/// hot `while` — reported the same single reason, as did a closure callee
/// called from a hot loop, so this was not a coverage gap in the corpus and the
/// later guards are unreached rather than merely quiet.
///
/// So `jit.isconstant` / [`OpRef::is_constant`] is established, not suspected:
/// the cell arrives red because it is read out of the frame's own cells
/// region.  `ever_mutated` was never a candidate for THAT decline —
/// `CellFamily::new` is born `false` and a write-once binding never sets it —
/// and the `can_move` gate on the contents is likewise never reached.
///
/// What a red cell does with that is what changed: it takes
/// [`try_walker_read_deref_cell`] instead of declining.  Re-measured over 483
/// fixtures on 2026-08-27, the row reads `consulted=55 fired=46 suppressed=0`
/// across the 16 fixtures that consult it, all 46 firings are that arm, and
/// every one of the 9 declines reports `READ ever-mutated`.  So the row is
/// covered now, and it is the constant arm that a `spec-folds=load_deref`
/// directive would still not be covering.
fn try_walker_specialize_load_deref<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<OpRef>, DispatchError> {
    /// Name the guard that declined, so that "this fold fires nowhere" can be
    /// attributed to one of them instead of re-derived by reading.
    macro_rules! decline {
        ($why:literal) => {{
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LOAD-DEREF {}", $why);
            }
            return Ok(None);
        }};
    }
    let Some(&cell_op) = r_args.first() else {
        decline!("no-cell-operand");
    };
    let Some(majit_ir::Value::Ref(cell_ref)) = ctx.trace_ctx.box_value(cell_op) else {
        decline!("cell-box-not-ref");
    };
    // nestedscope.py `if jit.isconstant(self)`.  A RED cell takes the read arm
    // below instead of declining: the residual it replaces is one field read
    // behind a `RandomEffects` call boundary, which is worth removing whether
    // or not the cell is green.
    if !cell_op.is_constant() {
        return try_walker_read_deref_cell(ctx, op_pc, cell_op, cell_ref);
    }
    // `NO_CONCRETE` is `usize::MAX - 1`, not zero, so `is_null` does not cover
    // it and `is_cell` would dereference it.  It means "this box carries no
    // concrete half", which declines here for the same reason a null does: the
    // fold answers from the cell it can read, and there is none.  Spelled
    // alongside the null the way `state.rs` reads a `Value::Ref`
    // (`frame_ref.is_null() || frame_ref == majit_ir::GcRef::NO_CONCRETE`).
    let cell = cell_ref.0 as pyre_object::PyObjectRef;
    if cell.is_null() || cell_ref == majit_ir::GcRef::NO_CONCRETE {
        decline!("cell-null-or-no-concrete");
    }
    if !unsafe { pyre_object::is_cell(cell) } {
        decline!("not-a-cell");
    }
    let family = unsafe { pyre_object::w_cell_family(cell) };
    if family.is_null() {
        decline!("family-null");
    }
    // nestedscope.py `if not self.family.ever_mutated`.
    if unsafe { (*family).ever_mutated.get() } {
        decline!("ever-mutated");
    }
    // nestedscope.py `w_res = self._elidable_get(); if w_res is not None`.
    let contents = unsafe { pyre_object::w_cell_get(cell) };
    if contents.is_null() {
        decline!("contents-unbound");
    }
    // A cell holds an arbitrary object, and baking its address needs no
    // movability test.  A RECORDED `ConstPtr` is forwarded at every later
    // stage — the active-trace walk, the retrace and resume-data pools, and
    // `remove_constptrs_in`, which turns each one into a `LoadFromGcTable`
    // whose slot is a root — and the window before that, while the mint sits
    // in the walker's own `registers_r` bank, is covered by
    // `InlineRegisterBankGuard` (`miframe_registers`).  The only other address this fold
    // bakes is the family's, and a family is a `Box::into_raw` leak outside
    // the collector entirely.
    let owner = ctx.trace_ctx.const_ref(family as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::cell_family_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    Ok(Some(ctx.trace_ctx.const_ref(contents as i64)))
}

/// [`try_walker_specialize_load_deref`]'s arm for a cell the trace holds RED —
/// the ordinary freevar read from a frame's own cells region, which is every
/// LOAD_DEREF outside an inlined callee.
///
/// Upstream never needs this arm: `nestedscope.py Cell.get` falls back to
/// `self._elidable_get()` under `jit.isconstant`, and the read it guards is a
/// plain `getfield` the tracer is already inside.  Pyre reaches the same read
/// through a residual (`lower_load_deref_value_hlop_to_insn`), and a residual
/// is not merely a call: `CallFlavor::Plain` resolves to
/// `EffectInfo::MOST_GENERAL`, so every freevar read also arms
/// `GUARD_NOT_FORCED`, `GUARD_NO_EXCEPTION` and `has_random_effects()`'s
/// `clean_caches`.  A loop reading one freevar per iteration therefore loses
/// the optimizer's whole heap cache once per iteration.
///
/// What replaces it is `w_cell_get` spelled out: a class pin on the slot and
/// one `GETFIELD_GC_R(cell, contents)`, which re-reads on every iteration
/// exactly as the residual re-read it.  Nothing is baked.
///
/// It still takes the `ever_mutated` quasi-immutable the fold above takes, and
/// that is NOT because the read is elidable — it is not, `contents` is a
/// mutable field.  It is because a STORE_DEREF cannot invalidate the cached
/// read.  `lower_store_deref_value_hlop_to_insn` binds its residual
/// `PlainCannotRaise`, whose `EffectInfo` carries analyzer-EMPTY write-descr
/// sets — an assertion that the call writes no field at all — so `OptHeap`
/// neither runs `clean_caches` (that needs `RandomEffects`) nor flushes any
/// descr, and a `GETFIELD_GC_R(cell, contents)` recorded before the store is
/// still live after it.  `run_nonlocal` in
/// `bench/synth/closure_freevar_branch_resume` is the witness: `nonlocal n; n
/// = n + 1` over 40000 iterations answered 24417 with the read admitted
/// unconditionally.
///
/// The pair of record-time facts below closes that hole without a write set,
/// which nothing in pyre can mint (`make_call_descr_sized` rejects a non-empty
/// raw descr set once the bitstrings are computed):
///
/// * the contents are BOUND, so the next write to this cell is a
///   bound-to-bound transition or a delete, and
/// * `ever_mutated` is false,
///
/// and `w_cell_set` / `w_cell_delete` both call `set_ever_mutated(true)` for
/// exactly those two, ahead of touching `contents`.  So any store that could
/// stale this read invalidates the quasi-immutable first, and the
/// `GUARD_NOT_INVALIDATED` below is what a STORE_DEREF's own effect set does
/// not say.  The write-once freevar — a closed-over constant, and the
/// `__class__` cell every zero-argument `super()` reads — keeps the read.
///
/// The unbound `NameError` becomes a side exit: `bh_load_deref_value_fn`
/// raises it when the contents are null, so a `GUARD_NONNULL` in that place
/// resumes the residual, which raises it from the eval loop.  A cell that is
/// unbound AT RECORDING keeps the residual instead — the fold would otherwise
/// have to emit a guard it knows fails.
pub(crate) fn try_walker_read_deref_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cell_op: OpRef,
    cell_ref: majit_ir::GcRef,
) -> Result<Option<OpRef>, DispatchError> {
    macro_rules! decline {
        ($why:literal) => {{
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LOAD-DEREF-READ {}", $why);
            }
            return Ok(None);
        }};
    }
    let cell = cell_ref.0 as pyre_object::PyObjectRef;
    if cell.is_null() || cell_ref == majit_ir::GcRef::NO_CONCRETE {
        decline!("cell-null-or-no-concrete");
    }
    if !unsafe { pyre_object::is_cell(cell) } {
        decline!("not-a-cell");
    }
    let family = unsafe { pyre_object::w_cell_family(cell) };
    if family.is_null() {
        decline!("family-null");
    }
    if unsafe { (*family).ever_mutated.get() } {
        decline!("ever-mutated");
    }
    let contents = unsafe { pyre_object::w_cell_get(cell) };
    if contents.is_null() {
        decline!("contents-unbound");
    }
    // Under the single-frame collapse a guard here would resume at the
    // caller's CALL, re-running whatever that callee already did.
    if ctx.fbw_mode.inline_subwalk
        && !crate::jitcode_dispatch::walker_inline_guard_resumes_in_callee(ctx)
    {
        decline!("subwalk-guard-collapses");
    }
    let cell_type = &pyre_object::nestedscope::CELL_TYPE as *const _ as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(cell_op) {
        let type_const = ctx.trace_ctx.const_int(cell_type);
        crate::jitcode_dispatch::walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardClass,
            &[cell_op, type_const],
        )?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(cell_op);
    }
    let owner = ctx.trace_ctx.const_ref(family as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::cell_family_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    let value = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        cell_op,
        crate::descr::cell_contents_descr(),
    );
    if !matches!(
        ctx.trace_ctx.box_value(value),
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
    ) {
        ctx.trace_ctx.set_opref_concrete(
            value,
            majit_ir::Value::Ref(majit_ir::GcRef(contents as usize)),
        );
    }
    crate::jitcode_dispatch::walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardNonnull,
        &[value],
    )?;
    Ok(Some(value))
}

pub(crate) fn walker_pin_plain_ever_mutated<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    plain: *const pyre_interpreter::objspace::std::mapdict::PlainAttribute,
) -> Result<(), DispatchError> {
    if plain.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(plain as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::plain_attribute_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// Mirror `opimpl_jit_force_quasi_immutable` for mapdict writes hidden inside
/// residuals.  Record the target before testing it because
/// `opimpl_record_quasiimmut_field` creates the watcher.  No `GUARD_ISNULL` is
/// needed: these writes remain residual and perform their own invalidation.
fn try_walker_force_quasi_immut_mapdict_write<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    i_args: &[OpRef],
    r_args: &[OpRef],
) -> Option<()> {
    use majit_ir::RuntimeHelperKind as K;
    let is_store = matches!(helper, K::StoreAttr);
    if !is_store && !matches!(helper, K::DeleteAttr) {
        return None;
    }
    // This helper only raises where the root flush can adopt its operand stack.
    // Otherwise the optimizer's `is_still_valid_for` check discards a loop
    // whose recorded qmut value changed.
    if !ctx.vstack_valid || ctx.fbw_mode.inline_subwalk || ctx.trace_ctx.is_bridge_trace {
        return None;
    }

    let &obj_opref = r_args.first()?;
    let &code_opref = r_args.get(if is_store { 2 } else { 1 })?;
    let &name_opref = i_args.first()?;
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_obj_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
        Some(majit_ir::Value::Int(name_idx)),
    ) = (
        ctx.trace_ctx.box_value(obj_opref),
        ctx.trace_ctx.box_value(code_opref),
        ctx.trace_ctx.box_value(name_opref),
    )
    else {
        return None;
    };
    if w_obj_ptr == 0 || w_code_ptr == 0 {
        return None;
    }
    let code = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return None;
        }
        &*(code_ptr as *const pyre_interpreter::CodeObject)
    };
    let name = pyre_interpreter::pyframe::load_name_from_code(code, name_idx as usize)?;
    // `descr_set__name__` stores `name?`. The residual setattr hides
    // `rclass.py hook_setfield`, so the walker asks `mutatebox.nonnull()`
    // here, the way mapdict writes name `setattr_would_force_quasi_immut`.
    if is_store && name == "__name__" {
        let &value_opref = r_args.get(1)?;
        let Some(majit_ir::Value::Ref(majit_ir::GcRef(w_value_ptr))) =
            ctx.trace_ctx.box_value(value_opref)
        else {
            return None;
        };
        if w_value_ptr == 0 {
            return None;
        }
        let w_obj = w_obj_ptr as pyre_object::PyObjectRef;
        let w_value = w_value_ptr as pyre_object::PyObjectRef;
        if unsafe { pyre_object::typeobject::is_type(w_obj) } {
            if unsafe { pyre_interpreter::type_set_name_would_store(w_obj, w_value) }
                && unsafe { pyre_object::typeobject::w_type_w_name_qmut_installed(w_obj) }
            {
                unsafe { pyre_object::typeobject::w_type_force_w_name_qmut(w_obj) };
                return Some(());
            }
            return None;
        }
    }
    let (attrkind, is_slot) = unsafe {
        pyre_interpreter::objspace::std::mapdict::classify_mapdict_write_attr(
            w_obj_ptr as pyre_object::PyObjectRef,
            name,
        )
    }?;
    if attrkind == pyre_interpreter::objspace::std::mapdict::INVALID {
        return None;
    }
    let attrname = rustpython_wtf8::Wtf8::new(if is_slot { "slot" } else { name });
    let target = if is_store {
        let &value_opref = r_args.get(1)?;
        let Some(majit_ir::Value::Ref(majit_ir::GcRef(w_value_ptr))) =
            ctx.trace_ctx.box_value(value_opref)
        else {
            return None;
        };
        if w_value_ptr == 0 {
            return None;
        }
        unsafe {
            pyre_interpreter::objspace::std::mapdict::setattr_would_force_quasi_immut(
                w_obj_ptr as pyre_object::PyObjectRef,
                attrname,
                attrkind,
                w_value_ptr as pyre_object::PyObjectRef,
            )
        }
    } else {
        unsafe {
            pyre_interpreter::objspace::std::mapdict::delattr_would_force_quasi_immut(
                w_obj_ptr as pyre_object::PyObjectRef,
                attrname,
                attrkind,
            )
        }
    }?;

    use pyre_interpreter::objspace::std::mapdict::MapdictQmutTarget as T;
    // `op_pc` is the walker's jit pc — the resume coordinate
    // `walker_capture_snapshot_for_last_guard` stamps on the drained
    // `GUARD_NOT_INVALIDATED`, the same one every other fold guard uses. It is
    // not the Python pc the operand-stack latch below carries.
    match target {
        T::PlainEverMutated(plain) => walker_pin_plain_ever_mutated(ctx, op_pc, plain).ok()?,
        T::TerminatorAllowUnboxing(term) => {
            crate::jitcode_dispatch::walker_pin_terminator_allow_unboxing(ctx, op_pc, term).ok()?
        }
        T::HolderTyp(holder) => {
            crate::jitcode_dispatch::walker_pin_holder_typ(ctx, op_pc, holder).ok()?
        }
        T::HolderAttr(holder) => {
            crate::jitcode_dispatch::walker_pin_holder_attr(ctx, op_pc, holder).ok()?
        }
    }
    let installed = unsafe {
        match target {
            T::PlainEverMutated(plain) => pyre_interpreter::objspace::std::mapdict::plain_attribute_ever_mutated_qmut_installed(plain),
            T::TerminatorAllowUnboxing(term) => pyre_interpreter::objspace::std::mapdict::terminator_allow_unboxing_qmut_installed(term),
            T::HolderTyp(holder) => pyre_interpreter::objspace::std::mapdict::holder_typ_qmut_installed(holder),
            T::HolderAttr(holder) => pyre_interpreter::objspace::std::mapdict::holder_attr_qmut_installed(holder),
        }
    };
    if !installed {
        return None;
    }
    unsafe {
        match target {
            T::PlainEverMutated(plain) => {
                pyre_interpreter::objspace::std::mapdict::plain_attribute_force_ever_mutated_qmut(
                    plain,
                )
            }
            T::TerminatorAllowUnboxing(term) => {
                pyre_interpreter::objspace::std::mapdict::terminator_force_allow_unboxing_qmut(term)
            }
            T::HolderTyp(holder) => {
                pyre_interpreter::objspace::std::mapdict::holder_force_typ_qmut(holder)
            }
            T::HolderAttr(holder) => {
                pyre_interpreter::objspace::std::mapdict::holder_force_attr_qmut(holder)
            }
        }
    }
    Some(())
}

/// The `RuntimeHelperKind` a residual's fold ladder may recognise, which is
/// its own except while a profile function is installed.
///
/// A profiled frame owes `c_call` / `c_return` around every builtin call its
/// bytecode makes — `baseobjspace.py call_valuestack`, `pyopcode.py`
/// CALL_FUNCTION_KW / CALL_FUNCTION_EX and `callmethod.py CALL_METHOD_KW` hand
/// `call_args_and_c_profile` the executing frame.  Upstream's compiled loop
/// carries that reporting because its tracer traced THROUGH `call_args`, and
/// its own `test_cprofile_builtin` unpacks exactly ONE loop out of a profiled
/// `lst.append` / `lst.pop` pair: the answer there is to record the reporting,
/// never to decline the trace.
///
/// The residual helpers take that same diversion
/// (`call_jit.rs residual_call_c_profile_frame`), so the reporting rides in the
/// trace as a `CALL_MAY_FORCE` to a helper that runs it — for exactly the three
/// call helpers, which are the whole population that owes an event.  Every
/// other kind reaching a residual dispatcher erases an operator dispatch, which
/// reports nothing on cpython, pypy3 or pyre alike.
///
/// What the walker owes in return is to leave those three ALONE.  A fold, a
/// descent into the builtin's own jitcode, or a `CALL_ASSEMBLER` replaces the
/// residual with something that reaches no call door, and none of them can
/// report: measured 2026-08-26, admitting a profiled trace while the folds
/// stayed live stopped `len`'s `c_call` at the entry threshold, 1041, however
/// long the tail.  Answering `None` is how they are left alone — no fold arm
/// tests for it, so each declines exactly as it does for a shape it cannot
/// handle, and the uniform decline path is the generic residual.
///
/// The two doors into a PYTHON callee owe `call` / `return` from `pyframe.py
/// execute_frame` rather than `c_call`, and are closed for a trace function and
/// a profiler alike by [`super::ec_hook_installed`].
fn walker_foldable_runtime_helper<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
) -> majit_ir::RuntimeHelperKind {
    let owes_c_call_events = matches!(
        ei.runtime_helper,
        majit_ir::RuntimeHelperKind::CallFn
            | majit_ir::RuntimeHelperKind::CallKw
            | majit_ir::RuntimeHelperKind::CallFunctionEx
    );
    if owes_c_call_events && ctx.session.borrow().is_being_profiled {
        majit_ir::RuntimeHelperKind::None
    } else {
        ei.runtime_helper
    }
}

/// `residual_call` shape `iRd>X` dispatcher. Reads `funcptr (i)`,
/// R-list args, and `descr`, runs `_build_allboxes` to produce the
/// callee's ABI-ordered arglist, classifies the call by `EffectInfo`
/// via [`select_residual_call_opcode`], records the matching
/// kind-coded `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` /
/// `Call*` op, emits `GUARD_NOT_FORCED` on the forces branch, emits
/// `GUARD_NO_EXCEPTION` if the classification says `can_raise`, and
/// writes the recorded result OpRef into the dst register chosen by
/// `dst_bank`.
///
/// RPython parity: `pyjitpl.py _opimpl_residual_call1` →
/// `do_residual_or_indirect_call` → `do_residual_call`
/// (pyjitpl.py). `pyjitpl.py opimpl_residual_call_r_i =
/// _opimpl_residual_call1` and `:1347 opimpl_residual_call_r_r =
/// _opimpl_residual_call1` confirm both kind variants share the
/// `_call1` body. The `_X` suffix is the *call's return kind* — mapping
/// comes from `do_residual_call`'s `descr.get_normalized_result_type()`
/// dispatch (pyjitpl.py) and `select_residual_call_opcode`'s
/// kind-keyed opcode tables.
///
/// `dst_bank` selects where the call's result lands:
/// * `'r'`: caller's `registers_r[dst]` — Ref-typed `Call*` family
///   (`_r_r/iRd>r`, `pyjitpl.py opimpl_residual_call_r_r`).
/// * `'i'`: caller's `registers_i[dst]` — Int-typed `Call*` family
///   (`_r_i/iRd>i`, `pyjitpl.py opimpl_residual_call_r_i`).
/// * `'v'`: void return — operand layout drops the trailing `>X` byte and
///   the writeback no-ops (`_r_v/iRd`, `pyjitpl.py
///   opimpl_residual_call_r_v`, `blackhole.py bhimpl_residual_call_r_v`).
/// (`'f'` is intentionally absent: RPython does not exec-generate
/// `opimpl_residual_call_r_f`. The only float-result residual_call
/// shape is `_irf_f/iIRFd>f`, dispatched by
/// [`dispatch_residual_call_iIRFd_kind`].)
///
/// TODO: walker selects the IR opcode by EffectInfo
/// branch (`CallMayForce*` for forces, `CallLoopinvariant*` for
/// loop-invariant, `CallPure*` for elidable, otherwise `Call*`) via
/// [`select_residual_call_opcode`]. Three sub-cases route through
/// dedicated helpers before the selector:
///   - **libffi** ([`direct_libffi_call`], `pyjitpl.py`) — checked ahead
///     of release-gil when `ei.oopspecindex` is
///     `OopSpecIndex::LibffiCall`, parses the `CIF_DESCRIPTION_P` the
///     call's second argument points at, records one `GETARRAYITEM_RAW`
///     per argument slot and then the `CALL_RELEASE_GIL_*`.  An
///     unsupported argument or result kind declines to the same
///     `resbox is None` fallthrough the two helpers below take.
///   - **release-gil** ([`direct_call_release_gil`], `pyjitpl.py-
///     3681`) — early-return when `ei.is_call_release_gil()`,
///     reshapes the arglist to `[savebox, funcbox] + argboxes[1:]`
///     and records `CALL_RELEASE_GIL_*` instead of `CALL_MAY_FORCE_*`.
///   - **loop-invariant heapcache** ([`loopinvariant_lookup`] /
///     [`loopinvariant_now_known`], `pyjitpl.py`) —
///     short-circuits the record on a heapcache hit and populates
///     the cache after a fresh record.
///
/// Emits `GUARD_NOT_FORCED` on the forces path plus
/// `GUARD_NO_EXCEPTION` whenever `check_can_raise(False)` is true,
/// matching `pyjitpl.py`. After every recorded call op,
/// invalidates the heapcache via
/// `heap_cache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
/// matching `pyjitpl.py _record_helper_varargs` parity (forces
/// branch's `pyjitpl.py` redundantly invalidates with
/// `CALL_MAY_FORCE_*`, equivalent because `select_residual_call_opcode`
/// returns `CallMayForce*` for the forces classification).  Release-gil
/// helper invalidates with `CALL_MAY_FORCE_*` matching
/// `pyjitpl.py`'s `opnum1`. The pre-call vable IR bookkeeping
/// (`pyjitpl.py vable_and_vrefs_before_residual_call`, IR-only
/// portion: FORCE_TOKEN + SETFIELD_GC) is wired via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`].  The
/// after-call helpers (`pyjitpl.py
/// vrefs_after_residual_call` / `vable_after_residual_call`) and the
/// runtime heap mutations on `tracing_before_residual_call` run in the
/// residual-call execution path — see
/// [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
/// split rationale.  The `OS_NOT_IN_TRACE` check fires up front via
/// [`do_not_in_trace_call_result`] — fail-loud guard against future
/// silent TODOs once the `majit-translate` analyzer trio
/// populates `oopspecindex`.
///
/// Still missing relative to upstream `do_residual_call`, all blocked
/// on infrastructure absent from pyre-jit-trace today:
///   - `OS_JIT_FORCE_VIRTUAL` PTR_EQ + GUARD_VALUE prelude
///     (`pyjitpl.py _do_jit_force_virtual`) —
///     walker is fail-loud here via [`do_jit_force_virtual_guard`]
///     (called from each `dispatch_residual_call_*` arm); a producer
///     that emits an `OopSpecIndex::JitForceVirtual` calldescr surfaces
///     `DispatchError::JitForceVirtualRequiresConcreteResolver` instead
///     of silently recording `CALL_MAY_FORCE_*` (this was the prior
///     behaviour and is documented as STRICTER-THAN-PYPY in
///     [`do_jit_force_virtual_guard`]'s docstring). Optimizer pass
///     `OptVirtualize::optimize_jit_force_virtual` (`virtualize.rs`)
///     already handles the constant-token / non-null-forced short-circuit
///     post-trace. Adding the PTR_EQ + GUARD_VALUE prelude (the only
///     way to retire the fail-loud guard) is not yet implemented and
///     would land with the walker; metainterp has a tests-only
///     orthodox port at
///     `majit-metainterp/src/pyjitpl.rs _do_jit_force_virtual`
///     that the converged walker would route through. Production reach
///     today is zero — `jtransform.rs jit.force_virtual` is the only
///     producer and pyre's interpreter does not emit it.
///   - `vrefs_before_residual_call` / `vrefs_after_residual_call` ARE called
///     by the walker, under the `is_may_force` gate that mirrors
///     `do_residual_call`'s `assembler_call or effectinfo.check_forces_...`
///     (`pyjitpl.py`). Their loops are NOT empty: [`walker_ec_enter`]
///     takes a vref of every seeded callee frame via
///     `TraceCtx::opimpl_virtual_ref` — `ExecutionContext.enter`'s vref — and
///     pairs it with `opimpl_virtual_ref_finish` when the frame leaves, so
///     `virtualref_boxes` carries real entries whenever an inlined callee is
///     live across a may-force call. Measured over 431 synth + 93 parity
///     fixtures via the `[vref-bracket] pairs=` report: 5487 bracket entries,
///     686 of them (12.5%) with at least one pair, 66 of the 316 emitting
///     fixtures reaching a nonzero count, maximum 7 pairs. Vable forces are
///     detected separately, by the residual-call execution path's heap-token
///     bracket.
///   - `direct_assembler_call` (`pyjitpl.py`) + KEEPALIVE
///     (`pyjitpl.py`) — only fire when `assembler_call=True`
///     in `do_residual_call`. Walker's residual_call dispatchers are
///     never called with `assembler_call=True`; the parallel
///     `inline_call_*/dR>X` family routes through
///     [`dispatch_inline_call_dr_kind`] instead. Adding the path would
///     require the codewriter to emit a new `assembler_call` shape, not
///     a walker-side change.
///   - Per-PC liveness narrowing for the snapshot that
///     `walker_capture_snapshot_for_last_guard` attaches
///     (`pyjitpl.py _get_list_of_active_boxes`). Walker's
///     helper today snapshots every non-`OpRef::NONE` register across
///     all three banks; RPython narrows the box list via
///     `jitcode.get_live_vars_info(pc, op_live)` so dead registers are
///     pruned before the snapshot.  The walker has no `op_live` byte
///     reader plumbed through `SubJitCodeBody` yet — follow-up
///     once the codewriter exposes the per-PC liveness table on the
///     callee body slice.  Over-capture is correctness-preserving:
///     `store_final_boxes_in_guard` filters dead boxes from the
///     snapshot via the optimizer's liveness pass.
fn try_walker_lower_getexecutioncontext<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    funcptr: OpRef,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' || !r_args.is_empty() {
        return Ok(None);
    }
    let Some(majit_ir::Value::Int(funcaddr)) = ctx.trace_ctx.box_value(funcptr) else {
        return Ok(None);
    };
    if funcaddr != *TAKE_LAST_EXEC_CTX_FNADDR {
        return Ok(None);
    }

    // `ObjSpace.getexecutioncontext` returns `threadlocals.get_ec()` in the
    // translated arm (`pypy/interpreter/baseobjspace.py`). Inside the portal
    // that object is already the second red (`interp_jit.py
    // PyPyJitDriver.reds = ['frame', 'ec']`), so the meta-trace carries the
    // red itself; it does not retain a TLS call before every EC field access.
    //
    // Charon cannot extract Rust's `thread_local!` body, which leaves
    // `take_last_exec_ctx` as a residual leaf inside otherwise translated
    // helpers such as `get_sys_exception`. Connect that leaf to the same live
    // red here. Do not fold to the recording pointer: a compiled loop may run
    // on another OS thread, and the red is what gives that entry its own EC.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let concrete_ec = pyre_interpreter::call::getexecutioncontext();
    if concrete_ec.is_null() {
        return Ok(None);
    }
    write_ref_reg(
        ctx,
        op_pc,
        dst,
        ec,
        ConcreteValue::Ref(concrete_ec as usize as pyre_object::PyObjectRef),
    )?;
    Ok(Some(()))
}

pub(crate) fn dispatch_residual_call_iRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) opens every residual call
    // with metainterp.clear_exception(), so a caught exception's
    // last_exc_value never survives past the next call — the
    // opimpl_catch_exception assert (pyjitpl.py) relies on it.
    // Clear at the arm entry so declined/folded paths uphold the same
    // invariant as the concrete-execution success arm.
    ctx.clear_last_exc_value();
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (mut r_args, arg_width) = read_ref_var_list(code, op, 1, ctx)?;
    // #62: env-gated recognition probe (no-op unless PYRE_DIAG_INLINE_RECOG
    // set; full-body-walk authoritative path only).  First slice of the
    // call-inlining feature — confirms callable->JitCode recognition before
    // sub-walk wiring.
    if ctx.is_authoritative_executor && std::env::var("PYRE_DIAG_INLINE_RECOG").is_ok() {
        let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
        diagnose_inline_recognition(&arg_concretes, op.pc);
    }
    let descr_offset = 1 + arg_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    // RPython `do_residual_or_indirect_call` always receives a
    // CallDescr (pyjitpl.py). Codewriter emits only CallDescrs
    // for residual_call slots; surface a typed error if a test fixture
    // (or future deviation) routes a non-CallDescr here.
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    // Void shape `_r_v/iRd` (`pyjitpl.py opimpl_residual_call_r_v =
    // _opimpl_residual_call1`) has no trailing `>X` dst byte. The
    // result OpRef is discarded by `write_residual_call_result_to_dst`'s
    // `'v'` arm, so `dst` is irrelevant on the void path; reading the
    // byte would walk past the operand list.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    let ei = call_descr.get_extra_info();
    repair_carrier_call_ref_args(ctx, op.pc, ei.runtime_helper, &mut r_args);
    // Resolve LOAD_GLOBAL's semantic namespace before any specialization reads
    // the ref args and before `_build_allboxes` copies them.  PyPy's
    // `LOAD_GLOBAL` asks the live MIFrame for `get_w_globals()`; keeping that
    // one OpRef in both representations prevents the cell fold from seeing the
    // codewriter's null trace-time placeholder while the surviving residual
    // sees the later replacement.
    replace_movable_load_global_namespace_with_frame_globals(ctx, ei.runtime_helper, &mut r_args);

    // The identity the fold ladder below is allowed to recognise, which under
    // a profile function is not this call's own — see
    // [`walker_foldable_runtime_helper`].
    let foldable_runtime_helper = walker_foldable_runtime_helper(ctx, ei);
    // Residual-call entry mirrors `execute_varargs`: even when the walker
    // folds the call or leaves it recorded symbolically, stale handled
    // exceptions from earlier opcodes are not visible to the following
    // linear `catch_exception/L`.
    clear_walk_exception(ctx);

    // `pyopcode.py IMPORT_NAME` traces `get_builtin`, `getdebug`, and
    // `get_w_globals` as reads of the live red frame. Non-null debugdata
    // globals stay in the fold. Null debugdata globals descend
    // `pyframe::PyFrame::get_w_globals`.
    if ctx.is_authoritative_executor
        && matches!(
            foldable_runtime_helper,
            majit_ir::RuntimeHelperKind::LoadImport
                | majit_ir::RuntimeHelperKind::LoadImportLocals
                | majit_ir::RuntimeHelperKind::LoadImportGlobals
        )
        && let Some(&frame_op) = r_args.first()
        && try_walker_import_frame_read_fold(
            ctx,
            op.pc,
            foldable_runtime_helper,
            frame_op,
            dst,
            dst_bank,
        )?
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadImportGlobals
        && let Some(&frame_op) = r_args.first()
        && try_walker_descend_frame_get_w_globals(ctx, op.pc, frame_op, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `space.getexecutioncontext()` in an inlined translated helper is the
    // portal's already-live EC red. Rust's opaque TLS accessor otherwise
    // survives as a residual call and separates the handler's EC store from
    // the helper's read, preventing both heap forwarding and exception
    // virtualization.
    if try_walker_lower_getexecutioncontext(ctx, op.pc, funcptr, &r_args, dst_bank, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Offer the specific pop fold before generic builtin inlining. Now that
    // `list.pop` has a `__majit_wrap_*` gateway, the generic path finds its
    // jitcode and otherwise descends into the wrapper until `w_list_len`'s
    // lock seam. Restrict this to the root walk: guards emitted in an inline
    // subwalk resume at the caller CALL boundary and could repeat an earlier
    // caller side effect.
    if ctx.is_authoritative_executor
        && !ctx.fbw_mode.inline_subwalk
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_list_pop(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `isinstance(x, C)` for a class whose metaclass is exactly `type`: the
    // answer is `issubtype`'s elidable MRO test on two promoted types
    // (typeobject.py), so pin both and bake it instead of leaving the opaque
    // `bh_call_fn(isinstance_builtin, NULL, x, C)` residual.  Read-only like
    // the `len` fold, so no sub-walk restriction; a tuple or union classinfo, a
    // custom `__instancecheck__`, and an unversioned type fall through to the
    // generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.runtime_helper == majit_ir::RuntimeHelperKind::CallFn
    {
        if spec_gate(SpecFold::BuiltinIsinstance, || {
            try_walker_specialize_builtin_isinstance(ctx, code, op, &r_args, dst)
        })?
        .is_some()
        {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // `ABCMeta.__instancecheck__` is Python.  Enter it before the wrapper
        // descent: an `Err` from that descent aborts the trace.
        if let Some(inlined) =
            try_walker_inline_isinstance_dunder(ctx, op, code, &r_args, call_descr, dst, dst_bank)?
        {
            return Ok(inlined);
        }
    }

    // The class namespace does not exist until this residual runs, so the
    // enclosing frame's non-constant return is the available signal that the
    // fresh class can reach the trace.  Check before the call so forward resume
    // executes the class body once.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && walker_active_py_code(ctx)
            .is_none_or(|active| !pyre_interpreter::code_returns_only_constants(active))
        && try_walker_force_quasi_immut_class_body(ctx, code, op, 1, &r_args)
    {
        crate::state::note_force_quasi_immut_abort();
        prepare_force_quasi_immutable_abort(ctx, op.pc);
        return Err(DispatchError::ForceQuasiImmutable { pc: op.pc });
    }

    // `function.py funccall_valuestack`'s `_code_of_sys_exc_info` test
    // precedes `code.fast_natural_arity` dispatch, so this `exc_info_direct`
    // fold sits immediately before the `fastcall_0`-equivalent
    // `try_walker_inline_builtin_call` descent below.  The generated CALL has
    // already become a CallFn residual at this seam, so reproduce the source
    // fast path here with this inline level's own red frame and the shared
    // EC red.  When the look-ahead declines, the descent is the `fastcall_0`
    // fallback onto the regular builtin wrapper.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::SysExcInfo, || {
            try_walker_specialize_sys_exc_info(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // BuiltinCode.func is an indirect PBC target exactly like RPython's
    // gateway wrappers.  Enter its generated JitCode before considering the
    // user-function-only full-body walk below.
    if let Some(inlined) = try_walker_inline_builtin_call(
        ctx,
        op,
        code,
        funcptr,
        1,
        &r_args,
        call_descr,
        foldable_runtime_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    // Float slot wrappers have no `__majit_wrap_*` jitcode, so the builtin
    // descent above declines them. Descend `_float_*` before the user-function
    // walk.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn {
        if let Some(inlined) = try_walker_inline_float_slot(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
        if let Some(inlined) = try_walker_inline_int_slot(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
    }

    // #62 slice (3c): attempt full-body-walk inline of a user-function call
    // unconditionally. Eligible exact-positional closure-free
    // calls sub-walk the callee body in place of the residual; ineligible
    // calls (including every non-`call_fn` helper, gated on `runtime_helper`)
    // fall through with no IR emitted.
    //
    // A nested self-rec (`plain` closing over `best`) cannot take
    // CALL_ASSEMBLER (that fold is Branch A: no cells). Inlining it
    // instead compiles a function-entry call that skips
    // `execute_frame`'s depth charge, so a JIT-hot run rides the
    // native stack budget (~8000) past `sys.getrecursionlimit()`.
    // Leave those on the residual so `execute_frame` still charges.
    // Module-level self-rec (`fib`) has no cells and still unrolls.
    let skip_cell_self_rec = r_args.first().is_some_and(|&callable| {
        residual_callee_is_walk_self_recursive(ctx, &[funcptr, callable], foldable_runtime_helper)
            && residual_callable_has_closure_or_cells(ctx, callable)
    });
    if !skip_cell_self_rec {
        if let Some(inlined) = try_walker_inline_user_call(
            ctx,
            op,
            code,
            1,
            funcptr,
            &r_args,
            call_descr,
            portal_user_call_helper(ctx, funcptr, foldable_runtime_helper),
            dst_bank,
            dst,
        )? {
            return Ok(inlined);
        }
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
    {
        if let Some(inlined) = try_walker_inline_exception_string_override(
            ctx, op, code, funcptr, &r_args, call_descr, dst,
        )? {
            return Ok(inlined);
        }
        // hash(x) over a user instance: inline __hash__ plus normalization.
        if let Some(inlined) =
            try_walker_inline_hash_builtin(ctx, op, code, funcptr, &r_args, call_descr, dst)?
        {
            return Ok(inlined);
        }
        // A class is not a `Function`, so the user-call route above declined it.
        if let Some(inlined) =
            try_walker_inline_type_call(ctx, op, code, funcptr, &r_args, call_descr, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        // `float` subclass construction: `w_float_subclass_new` +
        // `tag_subclass_instance`. `try_walker_inline_type_call` declines a
        // `__new__` that is not `object`'s.
        if let Some(inlined) =
            try_walker_inline_float_subclass_new(ctx, op, &r_args, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        if let Some(inlined) = try_walker_inline_int_subclass_new(ctx, op, &r_args, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        // `object.__new__(cls)`, which a Python `__new__` ends in.
        if let Some(inlined) = try_walker_inline_object_new(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
    }

    // FORMAT_WITH_SPEC: an exact `int` plus a constant decimal spec
    // descends `format_int_decimal` (`format_int_or_long` /
    // `_fill_number`).  A width-less spec with no forced sign descends
    // `descr_str` instead.  Tried before the Python `__format__`
    // inline so a builtin `int.__format__` never takes that route.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::FormatWithSpec
        && ctx.is_authoritative_executor
        && dst_bank == 'r'
        && spec_gate(SpecFold::FormatWithSpecInt, || {
            try_walker_specialize_format_with_spec_int(ctx, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // FORMAT_WITH_SPEC over a receiver whose `__format__` is Python: inline
    // the resolved body in place of the opaque format residual.  Keyed off
    // the helper tag, so every other `residual_call_r_r` falls straight
    // through.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::FormatWithSpec {
        if let Some(inlined) =
            try_walker_inline_format(ctx, op, code, &r_args, call_descr, dst, dst_bank)?
        {
            return Ok(inlined);
        }
    }

    // #62: a self-recursive call the inline path declined (e.g. the
    // branchy `fib`) gets a direct `CALL_ASSEMBLER` to its own loop token
    // instead of the heavyweight func-entry
    // residency residual. Independent of inline eligibility.
    if let Some(ca) = try_walker_call_assembler_self_recursive(
        ctx,
        op,
        code,
        funcptr,
        &r_args,
        call_descr,
        foldable_runtime_helper,
        dst_bank,
        dst,
    )? {
        return Ok(ca);
    }

    // `_r_*` shape: argboxes = R-list only; argbox_types = [Ref; n].
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(
        funcptr,
        &r_args,
        &argbox_types,
        call_descr.arg_types(),
        None,
    );
    if let Err(e) = ensure_residual_call_args_bound(&allboxes, op.pc) {
        if fbw_debug_abort_enabled() {
            let len_pc = op.pc + 1 + 1;
            let n = code[len_pc] as usize;
            let regs: Vec<u8> = code[len_pc + 1..len_pc + 1 + n].to_vec();
            let funcaddr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
                majit_ir::Value::Int(n) => Some(n as u64),
                _ => None,
            });
            eprintln!(
                "[fbw-unbound] pc={} regs={:?} r_args={:?} func={:?} runtime_helper={:?}",
                op.pc,
                regs,
                r_args,
                funcaddr.map(|a| format!("{a:#x}")),
                ei.runtime_helper,
            );
        }
        return Err(e);
    }

    // Optional diagnostic for iRd-shape residual calls.  The STORE_SUBSCR
    // specialization keys on a fn-pointer match against `bh_store_subscr_fn`
    // plus `r_args.len() == 3` with `dst_bank == 'v'`; logging raw addresses
    // here makes mismatches visible without affecting production when the
    // env var is unset.
    if crate::probe_subscr_enabled() {
        let funcptr_addr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
            majit_ir::Value::Int(n) => Some(n as u64),
            _ => None,
        });
        let arg_addrs: Vec<Option<u64>> = r_args
            .iter()
            .map(|&op| {
                ctx.trace_ctx.box_value(op).and_then(|v| match v {
                    majit_ir::Value::Ref(r) => Some(r.as_usize() as u64),
                    _ => None,
                })
            })
            .collect();
        eprintln!(
            "[MAJIT_PROBE_SUBSCR] dispatch_residual_call_iRd_kind pc={} dst_bank={} r_args.len={} funcptr_addr={:?} arg_addrs={:?}",
            op.pc,
            dst_bank,
            r_args.len(),
            funcptr_addr.map(|a| format!("{:#x}", a)),
            arg_addrs
                .iter()
                .map(|o| o.map(|a| format!("{:#x}", a)))
                .collect::<Vec<_>>(),
        );
    }

    // Drain the snapshot-capture failure the `WalkerFrameOps`
    // `generate_guard` impl latched (its `()` trait signature has no error
    // channel): a guard recorded without a resume snapshot must abort the
    // walk.
    if let Some(e) = ctx.pending_guard_snapshot_error.take() {
        return Err(e);
    }

    // StoreName/StoreGlobal: descend `typeobject.py write_cell` for an
    // in-place cell.  A replacing write still forces `version?` below.
    //
    // Two staleness bugs were fixed before enabling this unconditionally:
    // (1) the fold now eagerly applies the concrete
    // `cell.intvalue` write (journaled in [`FBW_CELL_STORE_JOURNAL`]) —
    // without it the walk's remaining concrete execution read the pre-store
    // global and the next LOAD fold's cache-hit sanity check tripped;
    // (2) `int_mutable_cell_value_descr` is a singleton `Arc` so the
    // optimizer's `cached_fields` (keyed by `descr_identity`) connects the
    // store's lazy `setfield_gc_i` to the LOAD's `getfield_gc_i` —
    // per-call fresh Arcs let `force_lazy_sets_for_guard` flush the store
    // BELOW an emitted load of the same cell (the nested module-loop
    // `i = i + 1; while i < n` read the pre-increment value and ran one
    // extra iteration).
    //
    // An exception table is a reason to skip: folding the residual drops
    // `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`, whose kept-stack snapshot
    // is what blackhole/peel resume uses for the live FOR_ITER iterator.
    // Without it, `ABORT_TOO_LONG` inside an inlined callee leaves a NULL
    // TOS on the portal frame (`trace_too_long_inline_multiframe`).
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && r_args.len() == 3
        && matches!(
            ei.runtime_helper,
            majit_ir::RuntimeHelperKind::StoreName | majit_ir::RuntimeHelperKind::StoreGlobal
        )
    {
        if let (Some(&frame_opref), Some(&name_opref), Some(&value_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2))
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_store_name_cell_fold(
                    ctx,
                    op.pc,
                    ei.runtime_helper,
                    frame_ptr,
                    w_name_ptr,
                    value_opref,
                )? {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // pyjitpl.py `opimpl_jit_force_quasi_immutable`, asked at the
    // residual boundary because pyre's namespace write lives inside one.
    //
    // Ordered AFTER the cell fold above (a successful in-place fold IS the
    // no-bump case) and BEFORE `try_execute_residual_call_via_executor`, so
    // nothing has been applied when the abort fires and the resume re-runs the
    // whole opcode cleanly. Executing first — the order `do_not_in_trace_call`
    // uses for its own contract — would double-apply the store or the delete.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && try_walker_force_quasi_immut_namespace_write(ctx, ei.runtime_helper, &r_args).is_some()
    {
        crate::state::note_force_quasi_immut_abort();
        prepare_force_quasi_immutable_abort(ctx, op.pc);
        return Err(DispatchError::ForceQuasiImmutable { pc: op.pc });
    }

    // pyjitpl.py OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) =
        do_not_in_trace_call_result(ctx, ei, &allboxes, call_descr, dst_bank, op.pc)?
    {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — walker
    // can't reproduce `_do_jit_force_virtual` without a concrete
    // `vref_ptr` resolver; surface a typed error rather than silently
    // recording `CALL_MAY_FORCE_*`.
    do_jit_force_virtual_guard(ei, op.pc)?;

    if ctx.is_authoritative_executor && dst_bank == 'i' && r_args.len() == 1 {
        // A TO_BOOL / POP_JUMP truth residual walks
        // `opcode_ops::truth_value` (`is_true`). That elides the may-force
        // call whose force/exc guards mis-resume the kept short-circuit
        // stack (`(i % 7) and ...`, and the boxed bool a COMPARE_OP leaves
        // for `if a == b:`).
        if foldable_runtime_helper == majit_ir::RuntimeHelperKind::Truth {
            if let Some(outcome) = try_walker_orthodox_truth(ctx, op.pc, r_args[0], dst, dst_bank)?
            {
                return Ok((outcome, op.next_pc));
            }
            // `is_true_lookup` is `dont_look_inside`. An overriding Python
            // `__bool__` is inlined here, the way `len` inlines `__len__`,
            // instead of recording the lookup as RandomEffects.
            if let Some(routed) =
                try_walker_inline_truth_bool(ctx, op, code, funcptr, &r_args, call_descr, dst)?
            {
                return Ok(routed);
            }
        }
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && r_args.len() == 1
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::UnaryInvert
    {
        if let Some(outcome) = try_walker_orthodox_unary_invert(ctx, op.pc, &r_args, dst, dst_bank)?
        {
            return Ok((outcome, op.next_pc));
        }
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && r_args.len() == 1
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::UnaryPositive
    {
        if let Some(outcome) = try_walker_orthodox_unary_pos(ctx, op.pc, &r_args, dst, dst_bank)? {
            return Ok((outcome, op.next_pc));
        }
    }

    // BINARY_SLICE walks `runtime_ops::binary_slice_values_inner` so
    // `is_list` / `is_str` / `is_tuple` are recorded guards and a custom
    // `__index__` is the body's `eval_slice_index`. Bytes / bytearray /
    // user `__getitem__` stay in `binary_slice_getitem_fallback`. Not a
    // spec-fold row.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && r_args.len() == 3
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::BinarySlice
    {
        if let Some(outcome) = try_walker_orthodox_binary_slice(ctx, op, &r_args, dst, dst_bank)? {
            return Ok((outcome, op.next_pc));
        }
    }

    // STORE_SUBSCR `list[int] = value` walks `w_list_setitem_inner`,
    // eliding the `CALL_MAY_FORCE` that would force the virtualizable every
    // iteration.  Falls through to the generic residual otherwise (SAFE).
    // Full-body walks only: the eager store rides `FBW_STORE_JOURNAL`,
    // whose commit/rollback epilogues run on FBW walk ends.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::StoreSubscr
    {
        if try_walker_orthodox_store_subscr(ctx, op.pc, &r_args)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // #171 setslice inline: `target[const_slice] = source` for a
        // same-length, step-1, Integer↔Integer slice — fold the assignment
        // into per-element getarrayitem/setarrayitem on the int_items blocks so
        // a virtualizable BUILD_LIST source temp is consumed without forcing.
        // Declines to the opaque residual for any shape it cannot reproduce
        // faithfully (SAFE — always byte-correct).
        if spec_gate(SpecFold::Setslice, || {
            try_walker_specialize_setslice(ctx, op.pc, &r_args)
        })?
        .is_some()
        {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if ctx.trace_ctx.is_bridge_trace && fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-store-fallthrough] bridge STORE_SUBSCR fell to GENERIC residual at pc={} \
                 (specialization declined — unjournaled concrete store)",
                op.pc
            );
        }
    }

    // Range GET_ITER: virtualize exact machine-word `range` into the same
    // `W_IntRangeIterator` shape PyPy's inlined `descr_iter` would trace.
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::GetIter
    {
        if let Some(iter_op) = spec_gate(SpecFold::GetIter, || {
            try_walker_specialize_get_iter(ctx, op.pc, &r_args, dst, dst_bank)
        })? {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, iter_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // The instance arm of `iter`: a user class's `__iter__` is Python, and
        // left residual it costs an interpreter frame and the virtualref
        // bracket that forces the caller's, once per `for` statement.
        if let Some(outcome) = spec_gate(SpecFold::InstanceIter, || {
            try_walker_specialize_instance_iter(
                ctx, op, code, funcptr, &r_args, call_descr, dst, dst_bank,
            )
        })? {
            return Ok(outcome);
        }
    }

    // Range FOR_ITER is a C-level iterator advance.  Re-emit its field
    // updates so the opaque ForIterNext residual cannot invalidate optheap;
    // The user-instance route below separately enters a Python `__next__`;
    // remaining iterator families retain the residual and its Python semantics.
    // In particular, a generic W_SeqIterObject must stay residual: its PyPy
    // `W_SeqIterObject.descr_next` frame catches IndexError from `__getitem__`
    // and turns it into iterator exhaustion.  Inlining only `__getitem__`
    // drops that catch frame, so GUARD_NO_EXCEPTION leaks IndexError on deopt.
    // The specialization supplies the same Ref result that the residual would,
    // including NULL for exhaustion, so the codewriter's trailing
    // GuardNonnull remains the only loop-exit guard.
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::ForIterNext
    {
        if let Some(item_op) = spec_gate(SpecFold::ForIterNext, || {
            try_walker_specialize_for_iter_next(ctx, op.pc, &r_args, dst, dst_bank)
        })? {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, item_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // Every list iterator records `W_FastListIterObject.descr_next`
        // (`list_iter_descr_next`) by sub-walking the interpreter body.
        // Not a spec-fold row.
        if let Some(item_op) =
            try_walker_orthodox_list_iter_next(ctx, op.pc, &r_args, dst, dst_bank)?
        {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, item_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if let Some(inlined) = spec_gate(SpecFold::InstanceNext, || {
            try_walker_specialize_instance_next(
                ctx, op, code, funcptr, &r_args, call_descr, dst, dst_bank,
            )
        })? {
            return Ok(inlined);
        }
        // A suspended generator is neither a C-level iterator nor a user-class
        // instance, so neither route above sees it and the residual advances
        // it by calling back into the interpreter.
        if let Some(resumed) = spec_gate(SpecFold::GeneratorNext, || {
            try_walker_specialize_generator_next(
                ctx, op, funcptr, &r_args, call_descr, dst, dst_bank,
            )
        })? {
            return Ok(resumed);
        }
    }

    // Emit MAKE_FUNCTION's `Function.__init__` as New + SetField so a `def` in a
    // loop body virtualizes away instead of allocating through the opaque
    // residual.  Falls through to the residual for every shape the emit cannot
    // reproduce constant-for-constant (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::MakeFunction
        && spec_gate(SpecFold::MakeFunction, || {
            try_walker_specialize_make_function(ctx, op.pc, &r_args, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #195 / #73: virtualize an arity-2 plain-int BUILD_TUPLE
    // (`newtuple_from_array`) by descending `w_specialised_tuple_ii_new`
    // (`makespecialisedtuple2` `Cls_ii`). `fuse_boxing_alloc` records
    // `new_with_vtable` + `value0` / `value1`, so the backing array and the
    // partner UNPACK_SEQUENCE reads DCE to a pure-int loop. Falls through
    // to the opaque residual for any other shape (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::NewtupleFromArray
        && spec_gate(SpecFold::Newtuple, || {
            try_walker_specialize_newtuple(ctx, op.pc, &r_args, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // The arities `makespecialisedtuple2` does not claim take the canonical
    // array-backed `W_TupleObject` shape instead, so a non-escaping BUILD_TUPLE
    // of any width folds away rather than allocating through the opaque
    // residual.  Reached only after the `spec_ii` fold above declines; falls
    // through to the residual for any shape it cannot reproduce (SAFE — never
    // declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::NewtupleFromArray
        && spec_gate(SpecFold::NewtupleObject, || {
            try_walker_specialize_newtuple_object(ctx, op.pc, &r_args, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: virtualize a non-escaping BUILD_LIST (`newlist_from_array`) by
    // decomposing it into the `opimpl_newlist` shape (`pyjitpl.py`) —
    // `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
    // choosing the storage strategy from the concrete element shadows exactly
    // like `w_list_new` / `list_strategy_for`, so the traced object matches what
    // the blackhole rebuilds on deopt.  Falls through to the opaque residual
    // for any shape it cannot reproduce faithfully (empty list, non-const
    // array length, an element without a concrete Ref shadow) — SAFE, never
    // declined.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::NewlistFromArray
        && spec_gate(SpecFold::Newlist, || {
            try_walker_specialize_newlist(ctx, op.pc, &r_args, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: specialize `lst.append(x)` so its array ops reach the trace,
    // replacing the opaque `bh_call_fn` residual (orthodox descent of the
    // real `w_list_append` body; see `try_walker_orthodox_list_append`).  The
    // call arrives as `CallFn` with `dst_bank == 'r'` (the None result is a
    // Ref, not void) and `r_args = [bound-method, PY_NULL, value]`.  Falls
    // through to the residual for any non-matching shape (SAFE).  The eager append rides
    // `FBW_LIST_EFFECT_JOURNAL`, whose commit/rollback epilogues run on FBW walk
    // ends (same lifecycle as the STORE_SUBSCR store journal).
    //
    // Both loop and function-entry (no-loop) traces are eligible.  A
    // no-loop helper compiled from entry (e.g. `def push(a, v): a.append(v)`
    // called in a hot loop) traces with `header_pc == 0`; its spare-capacity
    // guard's resume reconstructs the receiver from the call-site coordinate
    // published below (`collect_outer_active_boxes` at the CALL py_pc), which
    // preserves the receiver local across the fold's mid-statement guards in
    // both trace kinds.  The earlier loop-only restriction was a carryover
    // from the retired hand-rolled fold (#227), whose function-entry exit
    // layout dropped the receiver; the orthodox descent's resume coordinate
    // does not, verified by a two-list alternating-receiver append stress
    // (any wrong receiver box corrupts the cross-checked lists) and the
    // parity suite folding function-entry helper appends on both backends.
    // Inside an inlined callee the same descent runs only when
    // `walker_inline_guard_resumes_in_callee` holds.  `w_list_append_inner`
    // is a transparent helper: `walker_capture_transparent_helper_snapshot`
    // resumes at the paused Python CALL from
    // `compute_inline_helper_call_entry_frame`, so a guard re-executes that
    // call rather than the caller's earlier stores.  A shorter parent chain
    // still collapses onto `entry_py_pc` / `outer_active_boxes` and stays on
    // the residual, whose resume is past the call.
    // #171 ORTHODOX descent: descend the real `w_list_append` body,
    // recording its array ops native.
    // A recognition or body-sub-walk decline falls through to the generic
    // residual below.
    let list_append_inline_ok =
        !ctx.fbw_mode.inline_subwalk || walker_inline_guard_resumes_in_callee(ctx);
    if ctx.is_authoritative_executor
        && list_append_inline_ok
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_list_append(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171 ORTHODOX descent for the LIST_APPEND opcode (comprehension append,
    // e.g. `[f(x) for x in xs]`).  The codewriter lowers LIST_APPEND to a void
    // `jit_list_append(list, value)` residual tagged `ListAppendValue` (the
    // list is the peeked receiver, the value the popped operand — no
    // bound-method callable), so it arrives with `dst_bank == 'v'`.  Fold it
    // through the same `w_list_append` descent as the CallFn method-call form.
    // Same inline resume gate as the CallFn form (`list_append_inline_ok`).
    if ctx.is_authoritative_executor
        && list_append_inline_ok
        && dst_bank == 'v'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::ListAppendValue
    {
        // Descend `w_list_append_inner` (`ll_append` / `_ll_list_resize_ge`)
        // so the walk records `conditional_call` + `setarrayitem`. Fall
        // through to the generic `jit_list_append` residual when recognition
        // or the body walk declines. The commit journals the append
        // (`fbw_list_journal_push_append`) so `fbw_store_journal_rollback`
        // rewinds it on abort; the generic executor records the same
        // pre-append length before running `jit_list_append`. The decline
        // point is side-effect-free — it declines BEFORE emitting any IR.
        if try_walker_orthodox_list_append_opcode(ctx, code, op, &r_args, dst)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // `frame_locals_proxy::proxy_list_append(list, item)` returns the list.
    // The wrapper stays `dont_look_inside` so the slot scan does not follow
    // `w_list_lock` into `list.__init__`. Descend `w_list_append_inner` for
    // that one append and write the list back. A decline keeps the wrapper.
    if ctx.is_authoritative_executor
        && list_append_inline_ok
        && dst_bank == 'r'
        && try_walker_orthodox_proxy_list_append(ctx, code, op, funcptr, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Exact `dict.get(identity_key)` follows the promoted strategy-entry
    // shape produced by tracing `DictStrategy.getitem`: pin the key-set
    // iterator state and read the resolved entry value live.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::BuiltinDictGet, || {
            try_walker_specialize_builtin_dict_get(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // PyPy traces through builtin `getattr` into a property's Python getter.
    // The ordinary LOAD_ATTR spelling has the same inline route above; keep
    // the builtin spelling on it too, including lone-surrogate names carried
    // by PyPy's RPython-string representation.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && let Some(inlined) =
            try_walker_inline_builtin_getattr_property(ctx, op, code, &r_args, call_descr, dst)?
    {
        return Ok(inlined);
    }

    // `getattr(type, name)` whose class-MRO value is returned unchanged:
    // pin receiver, name, and the receiver version, then use the green value.
    // Non-matching shapes fall through to the generic residual.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::BuiltinTypeGetattr, || {
            try_walker_specialize_builtin_type_getattr(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // The same call on an ordinary instance: `getattr(obj, "name")` reaches the
    // one `space.getattr` that `obj.name` does, so it folds to the instance
    // read instead of an opaque MRO-walk residual.  Runs after the type form,
    // which answers the receiver the instance-shape read declines.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::BuiltinGetattr, || {
            try_walker_specialize_builtin_getattr(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // An exact `range(...)` constructor call becomes a virtual W_Range whose
    // four wrapped-int fields can fold directly into GET_ITER virtualization.
    // Non-canonical callables and arguments fall through to the residual.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
    {
        if let Some(outcome) = spec_gate(SpecFold::BuiltinRange, || {
            try_walker_specialize_builtin_range(ctx, code, op, funcptr, &r_args, call_descr, dst)
        })? {
            return Ok((outcome, op.next_pc));
        }
    }

    // Zero-argument `locals()` / `vars()` / `dir()` on the walk's own portal
    // frame: model `fast2locals`' fastlocals reads as `getarrayitem_vable_r`
    // plus a non-forcing dict-build chain — the shape the meta-tracer produces
    // upstream, where `pyframe.py fast2locals` is `@jit.unroll_safe` and
    // therefore looked into.  This arm runs BEFORE
    // `try_execute_residual_call_via_executor` arms the vable token protocol,
    // which is the point: the opaque residual is what turns the locals-read
    // barrier into `VableEscapedDuringResidualCall`.  Any non-matching shape
    // falls through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::BuiltinLocals, || {
            try_walker_specialize_builtin_locals(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `sys._getframe()` / `sys._getframe(depth)` : publish the per-level red
    // frame directly, the shape upstream's
    // `@jit.look_inside_iff(jit.isconstant(depth))` produces by tracing the
    // constant-depth walk through.  Like the `locals()` arm this runs BEFORE
    // `try_execute_residual_call_via_executor` arms the vable token protocol,
    // which is the point: `getframe`'s `mark_as_escaped` / frame forces clear
    // that token from inside the residual and cost the loop.  Armed audit
    // hooks stay on this arm too: the walk is still in the trace and only
    // `trigger_audit_events` is residual.  Any non-matching shape falls
    // through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && let Some(outcome) = spec_gate(SpecFold::SysGetframe, || {
            try_walker_specialize_sys_getframe(ctx, code, op, &r_args, dst)
        })?
    {
        return Ok((outcome, op.next_pc));
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_int_call(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // Ahead of the `bare_super_call` re-route, which sits with the other
    // substitutions below and is what a decline here falls through to.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::BareSuperVirtual, || {
            try_walker_specialize_bare_super_virtual(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::TwoArgSuperCall, || {
            try_walker_specialize_two_arg_super_call(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_float_call(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_complex_call(ctx, code, op, funcptr, &r_args, call_descr, dst)?
            .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_orthodox_str_call(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::MathFrexp, || {
            try_walker_specialize_math_frexp(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // `divmod(a, b)` descends `space.divmod` instead of the opaque
    // `bh_call_fn(divmod, NULL, a, b)` residual. `SubRaise` from that
    // body (a zero divisor) propagates; any other shape falls through
    // to the generic residual.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
    {
        if let Some(outcome) = spec_gate(SpecFold::BuiltinDivmodDescent, || {
            try_walker_orthodox_builtin_divmod(ctx, code, op, &r_args, dst)
        })? {
            return Ok((outcome, op.next_pc));
        }
    }

    // A `raise Type(args)` of a canonical
    // builtin exception class arrives as two residuals — a `CallFn` that
    // constructs the exception, and a `RaiseVarargs`
    // (`normalize_raise_varargs_jit`) that publishes it.  The construct fold
    // (`try_walker_trace_exception_new`) emits the inline virtualizable
    // `NewWithVtable` + SetField shape instead of the opaque `bh_call_fn`
    // constructor call (mirrors the retired trait-side constructor fold); the
    // raise fold (`try_walker_trace_raise_builtin`) then skips the residual
    // publish for a freshly-built exception with no `from` cause, emitting
    // `__context__` as a `SetfieldGc` on the still-virtual exception.
    // Together they drop the two may-force calls (construct + normalize) so
    // the exception virtualizes and DCEs.  Any non-matching shape falls
    // through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && try_walker_trace_exception_new(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // `BaseException___reduce___impl`: `(cls, args)` when `w_dict` is
    // NULL, `(cls, args, dict)` when the pointer is set. Residual
    // `bh_call_fn(__reduce__)` otherwise forces the virtual exception
    // (`exception_reduce`).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
        && spec_gate(SpecFold::ExceptionReduce, || {
            try_walker_specialize_exception_reduce(ctx, code, op, &r_args, dst)
        })?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::RaiseVarargs
        && r_args.is_empty()
        && ctx.frame_state.borrow().current_exception_seed.is_some()
    {
        let seed = ctx.frame_state.borrow().current_exception_seed.unwrap();
        let concrete = ctx.frame_state.borrow().current_exception_seed_concrete;
        if !concrete.is_null() && unsafe { pyre_object::is_exception(concrete) } {
            // `RAISE_VARARGS 0` may use the normalizing nullary helper rather
            // than the raw current-exception helper.  A bridge seed is already
            // a live BaseException, so the helper's successful result is the
            // pending fieldbox itself; retaining that OpRef keeps the value
            // loop-variant in the bridge namespace.
            ctx.trace_ctx.set_opref_concrete(
                seed,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
            );
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', seed)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::RaiseVarargs
        && try_walker_trace_raise_builtin(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // A bare-class `raise Type` has no construct residual to fold; synthesize
    // the zero-argument instance at the raise itself when the operand is a
    // canonical builtin exception class.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::RaiseVarargs
        && try_walker_trace_raise_bare_class(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // Lower the PUSH_EXC_INFO / POP_EXCEPT
    // exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
    // `sys_exc_value` slot, and consume the interpreter-only propagation-root
    // clear without recording a runtime CallN. Recognised by the
    // codewriter-stamped `runtime_helper` tag (not a funcptr address — the
    // residual calls the cross-crate `cpu.{get,set}_current_exception_fn`
    // wrappers).  A balanced PUSH save + POP restore on the same descr-
    // identity field is dead-store-eliminated by the heap optimizer, so a
    // non-escaping exception (built + raised + caught in one trace) stays
    // virtual and DCEs — eliding the per-iteration `set_current_exception`
    // CALL that otherwise forces the exception to materialize.
    if ctx.is_authoritative_executor
        && matches!(
            ei.runtime_helper,
            majit_ir::RuntimeHelperKind::GetCurrentException
                | majit_ir::RuntimeHelperKind::CurrentExceptionOrNone
                | majit_ir::RuntimeHelperKind::SetCurrentException
                | majit_ir::RuntimeHelperKind::ClearInFlightException
        )
        && try_walker_lower_exc_info_residual(
            ctx,
            code,
            op,
            ei.runtime_helper,
            &r_args,
            dst_bank,
            dst,
        )?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    if try_fold_registered_symbolic_residual(ctx, &allboxes, op.pc, dst, dst_bank)? {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `s.add(x)` on an exact `set`: swap the generic `bh_call_fn` target for
    // the direct `set_add` store the SET_ADD accumulator opcode records
    // (`try_walker_specialize_set_add_method`).  This is a substitution, not a
    // fold — the call is still a MayForce residual, still guarded and still
    // concrete-executed by the shared tail below, so the force/exception
    // guards, the heapcache invalidation and the `None` writeback are the ones
    // the generic path already emits.  A decline leaves every binding
    // untouched.
    //
    // The substituted descr carries `RuntimeHelperKind::SetAddMethod` for one
    // reason: `writes_live_heap` below reads the helper tag and the result
    // type, and the generic call it replaces arrived tagged `CallFn`.  An
    // untagged `MOST_GENERAL` descr with a `Ref` result matches neither, so the
    // substitution would quietly stop bumping the executed-effect odometer and
    // stop setting `body_effect_candidate` -- and a nested walk that aborted
    // behind a completed insert would rewind and run it again.  The set table
    // survives that (adding the same element twice is one insert), but the
    // element's `__hash__` need not be Python: a cpyext `tp_hash` runs native
    // code, moves no frame-entry odometer, and can do anything.  Keeping the
    // marker is what makes the substitution invisible to the rollback rules
    // rather than merely survivable by them.
    //
    // The zero-argument `super()` re-route is the second arm of the same shape
    // and is spelled here rather than with the folds above precisely so it can
    // reuse this tail: its leaf can run a `__class__` property, so it needs the
    // concrete execution, the raise transcription and the force bracket that
    // only the generic path performs.  The two arms are mutually exclusive by
    // callable -- `set.add` bound method against the `super` type -- so the
    // order between them is free; `set_add` is asked first only because it is
    // the cheaper recognition.
    //
    // An integer-strategy hit is not this tail.  The contains check does not
    // insert and does not run `__hash__`, so it is not tagged `SetAddMethod`
    // and must not be concrete-executed as `jit_set_add_method`.  `spec_gate`
    // still counts that `Some` as the same row.
    let direct_subst = if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn
    {
        match spec_gate(SpecFold::SetAddMethod, || {
            try_walker_specialize_set_add_method(ctx, code, op, &r_args, dst)
        })? {
            Some(SetAddMethodSpec::Elided) => {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            Some(SetAddMethodSpec::Subst(subst)) => Some(subst),
            None => spec_gate(SpecFold::BareSuperCall, || {
                try_walker_specialize_bare_super_call(ctx, code, op, &r_args)
            })?,
        }
    } else {
        None
    };
    let (funcptr, descr, allboxes) = match direct_subst {
        Some(subst) => (subst.funcptr, subst.descr, subst.allboxes),
        None => (funcptr, descr, allboxes),
    };
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    let ei = call_descr.get_extra_info();

    // pyjitpl.py forces-branch sub-case: when the descr's
    // `call_release_gil_target` is a non-NULL `(realfuncaddr, saveerr)`
    // pair, route through `direct_call_release_gil` which records
    // `CALL_RELEASE_GIL_*` with the upstream-shape arglist
    // `[savebox, funcbox] + argboxes[1:]` (pyjitpl.py).  All
    // other forces-branch paths (CALL_MAY_FORCE_*, the loopinvariant
    // sub-case below, the elidable branch, the default branch) come
    // out of `select_residual_call_opcode`.
    // pyjitpl.py forces-branch sub-case, ahead of the release-gil one:
    //     if effectinfo.oopspecindex == effectinfo.OS_LIBFFI_CALL:
    //         resbox = self.metainterp.direct_libffi_call(allboxes, c_result, descr)
    //     if resbox is None: ...release_gil / may_force...
    // A decline lands on exactly that fallthrough.
    let libffi_recorded = if ei.oopspecindex == majit_ir::OopSpecIndex::LibffiCall {
        match direct_libffi_call(
            ctx,
            ei,
            &allboxes,
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iRd_kind",
        )? {
            LibffiCallOutcome::Raised(outcome) => return Ok((outcome, op.next_pc)),
            LibffiCallOutcome::Recorded => true,
            LibffiCallOutcome::Declined => false,
        }
    } else {
        false
    };
    if libffi_recorded {
        // `direct_libffi_call` recorded the specialized call and both of the
        // forces branch's guards.
    } else if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py EF_LOOPINVARIANT short-circuit. The
        // cached path emits no IR op and no guard, so result-before-
        // guard ordering is moot — write the dst eagerly.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iRd_kind");
        // The abort gate is static (EI flags) and must run BEFORE the
        // concrete executor below: `do_residual_call` (pyjitpl.py)
        // executes the helper only on a path that keeps recording — it has
        // no execute-then-abandon shape.  Aborting after execution would
        // leave the helper's heap/exception effects standing while the
        // declined retrace re-runs the same bytecode, double-applying them.
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` — fires
        // unconditionally on the forces branch.  Records FORCE_TOKEN +
        // SETFIELD_GC IR for the active virtualizable; the runtime heap
        // mutations on `vinfo.tracing_before_residual_call` and
        // `vrefinfo.tracing_before_residual_call` (`pyjitpl.py`)
        // are handled by the residual-call execution path — see
        // [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
        // split rationale.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
            write_back_locals_for_proxy_reader(ctx, &allboxes);
        }

        // pyjitpl.py `execute_and_record_varargs`; may-force
        // calls use `history.record_nospec` and therefore count nothing.
        let profiled_call = matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        );
        if profiled_call {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
        }
        // `list_write_barrier` is pyre's stand-in for the write barrier
        // RPython's GC transform inserts.  `pyjitpl.py` never records
        // `ll_writebarrier` as a call (`executor.py` skips `COND_CALL_GC_WB`);
        // `rewrite.py gen_write_barrier` emits `COND_CALL_GC_WB` after
        // optimize.  A residual `CallN` every Object-strategy append is the
        // slow path of that barrier.  Record `CondCallGcWb` on the list
        // instead: the fast path is a header-flag check, and remembering the
        // `W_ListObject` still lets its custom tracer reach every slot
        // (`list_append_write_barrier_gc`, including `MAJIT_GC_ITEMSBLOCK=0`).
        // The helper still executes concretely below.
        // `pyjitpl.py:1943` takes `patch_pos` before recording the call so
        // `record_result_of_call_pure` can cut it back out.
        let patch_pos = ctx.trace_ctx.get_trace_position();
        let (recorded, is_list_wb) =
            record_list_write_barrier_residual(ctx, funcptr, &allboxes, call_opcode, descr.clone());

        // `MIFrame.execute_varargs(pure=True)` parity: for
        // `CallPure*` whose every argbox carries a known `box_value`,
        // execute the helper now and stamp `recorded` with the result so
        // downstream walker chain (sub-jitcode bodies that consume the
        // result via `concrete_of_opref`) folds end-to-end.  No-op when
        // any argbox is symbolic, when the EI can raise, or for non-pure
        // call opcodes.  The shared executor below handles the can-raise pure
        // case and transfers its exception state like `MIFrame.execute_varargs`.
        let recorded = try_fold_pure_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            descr.clone(),
            patch_pos,
            recorded,
        );
        // `_record_helper_varargs` counts RECORDED_OPS as it records; a
        // may-force call takes `history.record_nospec` and counts nothing.
        // An all-const pure fold cuts the call back out (`record_result_of_call_pure`),
        // so only a standing call op is a recorded op.
        if profiled_call && recorded.inline_const_to_value().is_none() {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }

        // pyjitpl.py `_opimpl_residual_call{1,2,3}` parity
        // for the remaining shapes.  PyPy
        // concrete-executes EVERY residual call regardless of EI — the
        // `exc` flag only selects the *guard* shape downstream
        // (`handle_possible_exception` → `GUARD_EXCEPTION` vs
        // `GUARD_NO_EXCEPTION`), not whether the helper runs.  Without
        // this, walker-recorded non-elidable helpers
        // (`store_subscr_fn`, `set_current_exception`, …) would skip
        // their heap mutation because `eval.rs`'s walker-skip
        // path bypasses `execute_opcode_step` → SIGBUS on the next read
        // of the un-mutated container.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        // A namespace write reaches the executor as an ordinary residual and
        // mutates the frame's dict in place; record what it displaces while the
        // write is still entirely ahead of the walk.
        let namespace_write_journaled =
            journal_walker_namespace_write(ctx, ei.runtime_helper, &r_args).is_some();
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
            namespace_write_journaled,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  A `CallPure*` reaches this
        // dispatcher too when its fold could not answer, and declines as
        // `PureUnfolded` when it also cannot raise — no effect and no
        // exception, so no such obligation; the can-raise half declines as
        // `Symbolic` and keeps it.
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                // The call op is already recorded. A symbolic fnaddr must
                // not stay in it whatever the decline cause (a `PureUnfolded`
                // `CallPure*` keeps the call too): the backend emits a call
                // to the hash. Cut back to the position taken before the
                // record, then abort this walk so the result box is not used.
                if let Some(addr) = recorded_symbolic_funcptr(ctx, &allboxes) {
                    ctx.trace_ctx.cut_trace_with_snapshots(patch_pos);
                    return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                        pc: op.pc,
                        symbolic: addr,
                    });
                }
                fbw_abort_nested_unjournaled_residual(ctx, op.pc, Some(cause))?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // `_record_helper_varargs` parity: every
        // recorded varargs op invalidates the heapcache via
        // `heapcache.invalidate_caches_varargs(opnum, descr,
        // argboxes)`.  Pyre's `record_op_with_descr` does NOT
        // auto-invalidate, so call it explicitly here.  Forces
        // branch (`select_residual_call_opcode` returned a
        // `CallMayForce*`) thus matches `pyjitpl.py` which uses
        // `opnum1 = CALL_MAY_FORCE_*`; non-forces branches
        // (`CallLoopinvariant*`/`CallPure*`/`Call*`) match the
        // `_record_helper_varargs` invocation that runs inside
        // upstream's `executor.execute_varargs(opnum, ...)`.
        // `COND_CALL_GC_WB` is not a call (`resoperation.py`); the
        // rewrite-inserted form never goes through `_record_helper_varargs`.
        if !is_list_wb {
            ctx.trace_ctx
                .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        }
        // pyjitpl.py execute_varargs: `make_result_of_lastop(op)`
        // runs BEFORE `handle_possible_exception()` precisely "because we need
        // the box to show up in get_list_of_active_boxes()".  Write the dst
        // for every non-void result REGARDLESS of whether the helper raised,
        // so the GUARD_NOT_FORCED fail_args snapshot reads the recorded OpRef
        // in the slot the resume position points at — otherwise a raising call
        // surfaces NONE in fail_args for the `>X` slot.  On a raised call the
        // OpRef carries a Null concrete shadow (never read on the exception
        // path); only the *caching* of a raised result is skipped below,
        // matching upstream's `not last_exc_value` pure-cache gate.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        // pyjitpl.py `metainterp.generate_guard(rop.GUARD_NOT_FORCED)`
        // — unconditionally on the forces-virtual-or-virtualizable branch.
        // The walker omits the `vable_after_residual_call(funcbox)`
        // short-circuit (`pyjitpl.py`): the residual-call execution
        // path's heap-token bracket already detects a vable escape and
        // surfaces `VableEscapedDuringResidualCall` before this guard is
        // emitted.
        if emit_guard_not_forced {
            // #73: maintain the `-live-` AFTER anchor.  A
            // residual-call guard reads its resume point at `self.pc` (the
            // `-live-` trailing the call, `pyjitpl.py`).  `op.next_pc` is
            // the first byte after the residual_call opcode, which the
            // `[funcptr, Call, -live-]` layout (jitcode.rs) makes the
            // trailing `-live-` byte.  Side-data only.
            ctx.live_after_jit_pc = op.next_pc;
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        // `metainterp.handle_possible_exception()` —
        // emits `GUARD_EXCEPTION(exc_type)` when the recording-time
        // helper raised (pinning the class for guard recovery), else
        // `GUARD_NO_EXCEPTION`.  The capture ports
        // `capture_resumedata(after_residual_call=True)`
        // (keyed on the guard opcode) so the
        // optimizer's `store_final_boxes_in_guard` finds a
        // `rd_resume_position` advanced *past* the call.
        // `COND_CALL_GC_WB` cannot raise (`resoperation.py`).
        if can_raise && !is_list_wb {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc)?;
                // `handle_possible_exception` routes
                // the raising branch through `finishframe_exception()`
                // immediately after emitting `GUARD_EXCEPTION`, so the
                // remaining bytes of the arm never run.  Surface the
                // outcome to `walk_loop` as `SubRaise`: at top-level it
                // emits the outer `FINISH(exc)` and Terminates the trace;
                // at sub-walk depth it propagates up to the caller's
                // `inline_call_*` handler.  Continuing past this point
                // would record dead arm IR (e.g. the arm's tail
                // `*_return`) onto an exception path and confuse the
                // optimizer's guard-fail snapshot.
                let exc = ctx
                    .last_exc_value()
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete();
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // Request that this residual call's no-exception-guard resume
                // route through the call's OWN post-call catch
                // (`GuardCaptureScope::residual_call_catch_resume`).  The
                // snapshot helper carries the call's jitcode offset only when
                // the CALL pc is actually covered by the code's exception table
                // (checked in `walker_capture_snapshot_for_last_guard_impl`);
                // an uncovered residual keeps the generic fallthrough resume.
                // See the scope field's doc.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        // pyjitpl.py `heapcache.call_loopinvariant_now_known`:
        // populate the cache so a subsequent matching call short-
        // circuits via the lookup above.  No-op for non-loopinvariant
        // EI per `loopinvariant_now_known`'s extraeffect check.
        //
        // Skip on `resid_raised`: caching a `recorded` OpRef with no
        // stamped concrete would propagate the un-stamped value into a
        // subsequent loop iteration's `loopinvariant_lookup` hit,
        // bypassing the actual helper call.
        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    let saved_last_exc_value = ctx.last_exc_value();
    let saved_last_exc_value_concrete = ctx.last_exc_value_concrete();
    let preserve_last_exc_for_handler =
        saved_last_exc_value.is_some() && reads_last_exc_before_next_catch(code, op.next_pc);
    if !preserve_last_exc_for_handler {
        clear_walk_exception(ctx);
    }
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (mut r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let mut descr = read_descr(code, op, descr_offset, ctx)?;
    let original_call_descr =
        descr
            .as_call_descr()
            .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
                pc: op.pc,
                descr_index,
            })?;
    let descr_key = descr.index();
    // Copy the recognition tag out of the descr borrow now: the DELETE_ATTR
    // fold gate below runs after the StoreAttr arm's `descr = specialized_
    // descr` reassignment, which the live `original_call_descr` borrow would
    // otherwise forbid (E0506).
    let runtime_helper_kind = original_call_descr.get_extra_info().runtime_helper;
    repair_carrier_call_ref_args(ctx, op.pc, runtime_helper_kind, &mut r_args);
    // Do this before the fold block below and before `_build_allboxes`: both
    // consumers must observe the same per-MIFrame namespace.  In particular,
    // the portal codewriter deliberately supplies NULL as a hint placeholder;
    // delaying replacement until the generic residual path makes the
    // LoadGlobal cell fold decline on that stale value.
    replace_movable_load_global_namespace_with_frame_globals(ctx, runtime_helper_kind, &mut r_args);
    // Void shape `_ir_v/iIRd` (`pyjitpl.py opimpl_residual_call_ir_v =
    // _opimpl_residual_call2`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // RootScope::get remains a residual unless the translator proves its
    // slot/value pairing. `shadowcolor.expand_one_pop_roots` reconstructs a
    // specific live value from its root slot; equality of an input-dependent
    // index with a slot observed while tracing is not that proof.

    // Flat argboxes = i_args ++ r_args (`boxes2` argcode order).
    // Parallel argbox_types stamps each entry with its source bank so
    // `_build_allboxes`'s type-filter loops can permute correctly.
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len());
    let mut argbox_types: Vec<Type> = Vec::with_capacity(i_args.len() + r_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    let mut allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        original_call_descr.arg_types(),
        None,
    );

    if ctx.is_authoritative_executor && dst_bank == 'r' && i_args.len() == 1 && r_args.len() == 1 {
        let func_addr = match ctx.trace_ctx.box_value(funcptr) {
            Some(majit_ir::Value::Int(n)) => n,
            _ => 0,
        };
        if func_addr != 0 && NEWUTF8_FNADDRS.contains(&func_addr) {
            if let (Some(majit_ir::Value::Int(len)), Some(storage_obj)) = (
                ctx.trace_ctx.box_value(i_args[0]),
                walker_concrete_ref_object(ctx, r_args[0]),
            ) {
                if len >= 0 {
                    let boxed = pyre_object::unicodeobject::w_str_from_storage_and_length(
                        storage_obj as *mut pyre_object::unicodeobject::UnicodeValueStorage,
                        len as usize,
                    );
                    if let Some(result) =
                        try_walker_orthodox_newutf8(ctx, op.pc, r_args[0], i_args[0], boxed)?
                    {
                        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;
                        return Ok((DispatchOutcome::Continue, op.next_pc));
                    }
                }
            }
        }
    }

    // pyjitpl.py `opimpl_jit_force_quasi_immutable` must run before
    // any fold or residual applies the opcode. In particular,
    // `try_walker_specialize_store_attr` mutates `?` fields while resolving,
    // so moving this below the STORE_ATTR fold would hide the force. The abort
    // resumes by re-running the whole opcode, so no part of its write may have
    // happened yet.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && try_walker_force_quasi_immut_mapdict_write(
            ctx,
            op.pc,
            runtime_helper_kind,
            &i_args,
            &r_args,
        )
        .is_some()
    {
        crate::state::note_force_quasi_immut_abort();
        prepare_force_quasi_immutable_abort(ctx, op.pc);
        return Err(DispatchError::ForceQuasiImmutable { pc: op.pc });
    }

    // LOAD_DEREF on a constant cell whose binding was only ever filled once:
    // fold to the contents under a `QUASIIMMUT_FIELD(family, ever_mutated)`
    // instead of the opaque read residual, the shape upstream gets for free
    // from `nestedscope.py Cell.get`.  Read-only, and every unrecognised
    // shape falls through to the residual (SAFE).
    //
    // This is the shape that decides whether a closure callee inlines at all:
    // the sub-walk's `DeferredCall` admission promises it commits nothing, and
    // a `load_deref_value` that stays residual is a nested residual the lever
    // could not inline, so the admission is revoked and the callee denied.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && runtime_helper_kind == majit_ir::RuntimeHelperKind::LoadDeref
    {
        if let Some(value) = spec_gate(SpecFold::LoadDeref, || {
            try_walker_specialize_load_deref(ctx, op.pc, &r_args)
        })? {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, value)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // STORE_ATTR fold (mapdict.py): recognize an existing unboxed
    // integer slot and replace only the generic setattr residual's helper,
    // arguments, and effect.  The transformed CallN continues through the
    // ordinary record + concrete-execute path below; unsupported receivers,
    // descriptors, custom hooks, absent/boxed/float slots, and type-changing
    // values retain the original CallMayForceN unchanged.
    if ctx.is_authoritative_executor
        && original_call_descr.get_extra_info().runtime_helper
            == majit_ir::RuntimeHelperKind::StoreAttr
    {
        if let (Some(&obj_opref), Some(&value_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                // Deterministic immutable-type raise (`int.x = v`) folds to a
                // traced inline exception construction the optimizer can
                // virtualize — tried before the unboxed-slot store fold; the
                // two shapes are disjoint (type receiver vs instance receiver).
                if let Some(outcome) = try_walker_trace_immutable_type_attr_raise(
                    ctx,
                    op,
                    obj_opref,
                    Some(value_opref),
                    w_code_ptr,
                    namei as usize,
                )? {
                    return Ok(outcome);
                }
                if let Some(outcome) = try_walker_trace_readonly_descr_attr_raise(
                    ctx,
                    op,
                    obj_opref,
                    value_opref,
                    w_code_ptr,
                    namei as usize,
                )? {
                    return Ok(outcome);
                }
                // The resolver below COMMITS before it answers: it performs
                // the write and returns, reaching no hook that could refuse
                // it afterwards.  A `BINARY_OP` /
                // `COMPARE_OP` rewind region has to be asked first, then,
                // rather than at the generic residual arm it would
                // eventually take.
                //
                // Reached from a `for` body since `store_attr` joined the
                // deferred-helper list in `fbw_state.rs`: a callee that stores
                // an attribute now descends there as it always did from a
                // `while` body, and the fold below is what erases its
                // residual.
                //
                // The receiver tells the rewind guard whether this store is
                // on an instance the region's own `type` call allocated
                // (`return Nested(self.n + o.n).n`) or on an object that
                // already existed, so it is passed rather than the region
                // being asked about the write alone.
                fbw_binop_rewind_refuse_commit(ctx, op.pc, Some(obj_opref))?;
                if spec_gate(SpecFold::StoreAttrDirect, || {
                    try_walker_specialize_store_attr(
                        ctx,
                        op.pc,
                        obj_opref,
                        value_opref,
                        w_code_ptr,
                        namei as usize,
                    )
                })?
                .is_some()
                {
                    fbw_mark_foriter_body_effect_since_consume();
                    fbw_bump_executed_effect("store_attr_direct");
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // DELETE_ATTR immutable-type raise fold — the deletion twin of the
    // STORE_ATTR fold above.  `bh_delete_attr_fn(obj, code, name_idx)`
    // carries no value operand: r_args = [obj, code], i_args = [name_idx].
    if ctx.is_authoritative_executor
        && runtime_helper_kind == majit_ir::RuntimeHelperKind::DeleteAttr
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(outcome) = try_walker_trace_immutable_type_attr_raise(
                    ctx,
                    op,
                    obj_opref,
                    None,
                    w_code_ptr,
                    namei as usize,
                )? {
                    return Ok(outcome);
                }
            }
        }
    }

    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;

    let ei = call_descr.get_extra_info();
    // The identity the fold ladder below is allowed to recognise, which under
    // a profile function is not this call's own — see
    // [`walker_foldable_runtime_helper`].
    let foldable_runtime_helper = walker_foldable_runtime_helper(ctx, ei);
    // pyjitpl.py OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) =
        do_not_in_trace_call_result(ctx, ei, &allboxes, call_descr, dst_bank, op.pc)?
    {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // Method-form `CALL` helpers lower through the mixed int/ref residual
    // shape (`bh_call_fn_N(callable, null_or_self, args...)` carries the call
    // arity in the Int list).  Share the same user-function inline gate as the
    // plain Ref-only residual, but read the concrete Ref shadows from the
    // shifted R-list offset.
    if let Some(inlined) = try_walker_inline_builtin_call(
        ctx,
        op,
        code,
        funcptr,
        1 + i_width,
        &r_args,
        call_descr,
        foldable_runtime_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    // Float slot wrappers have no `__majit_wrap_*` jitcode, so the builtin
    // descent above declines them. Descend `_float_*` before the user-function
    // walk.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn {
        if let Some(inlined) = try_walker_inline_float_slot(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
        if let Some(inlined) = try_walker_inline_int_slot(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
    }

    if let Some(inlined) = try_walker_inline_user_call(
        ctx,
        op,
        code,
        1 + i_width,
        funcptr,
        &r_args,
        call_descr,
        portal_user_call_helper(ctx, funcptr, foldable_runtime_helper),
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    // A class is not a `Function`, so the user-call route above declined it.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::CallFn {
        if let Some(inlined) =
            try_walker_inline_type_call(ctx, op, code, funcptr, &r_args, call_descr, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        // `float` subclass construction: `w_float_subclass_new` +
        // `tag_subclass_instance`. `try_walker_inline_type_call` declines a
        // `__new__` that is not `object`'s.
        if let Some(inlined) =
            try_walker_inline_float_subclass_new(ctx, op, &r_args, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        if let Some(inlined) = try_walker_inline_int_subclass_new(ctx, op, &r_args, dst_bank, dst)?
        {
            return Ok(inlined);
        }
        // `object.__new__(cls)`, which a Python `__new__` ends in.
        if let Some(inlined) = try_walker_inline_object_new(ctx, op, &r_args, dst_bank, dst)? {
            return Ok(inlined);
        }
    }

    // LoadConst fold: the LOAD_CONST helper (oopspec `LoadConst`, set
    // codewriter-side at flatten.rs
    // `build_residual_call_ir_r_single_ref_plain_insn_from_operands`)
    // re-materializes `co_consts[idx]` on every call.  When both the const
    // index (`i_args[0]`) and the code pointer (`r_args[0]`, the promoted
    // `frame.pycode`) are concrete, fold to the constant ref the call would
    // have produced — the indexed entry is loop-invariant — and suppress the
    // residual.  Falls through to the generic record when either operand is
    // not concrete (the residual stays correct in that case).
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadConst {
        if let (Some(&idx_opref), Some(&code_opref)) = (i_args.first(), r_args.first()) {
            if let (
                Some(majit_ir::Value::Int(consti)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(idx_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                // Read the constant identically to the runtime
                // `bh_load_const_fn`: every constant is the shared
                // `pycode.co_consts_w[index]` object.
                let w_const = unsafe {
                    pyre_interpreter::pycode::w_code_const(
                        w_code_ptr as pyre_object::PyObjectRef,
                        consti as usize,
                    )
                };
                let const_box = ctx.trace_ctx.const_ref(w_const as i64);
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, const_box)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // LoadGlobal fold (#62): mirror the trait LOAD_GLOBAL cell fast-path so
    // the optimizer hoists the module-global lookup out of the loop instead of
    // keeping the opaque CanRaise residual every iteration.  Requires namei
    // (i_args[0]), namespace (r_args[0]), promoted pycode (r_args[1]) and the
    // live frame (r_args[2]) concrete; the cell-strategy module-dict fast path
    // (name present in the executing frame's globals) and the builtins-cell
    // fast path (name absent from globals, resolved via `frame.get_builtin()`)
    // are both foldable.
    //
    // Handler-bearing bodies: `load_global_fn` is `CallFlavor::Plain` (can
    // raise NameError), so a residual lowering keeps a `GUARD_NO_EXCEPTION`
    // that a `catch_exception/L` in the same body may resume into.  The fold
    // emits the lookup as `ElidableCannotRaise` (the name resolves at trace
    // time and the version watchers fail the loop on any rebind/shadow), so a
    // SUCCESSFUL fold provably can't raise NameError and dropping that guard
    // for this load is sound — the handler can never be entered from it.  We
    // therefore attempt the fold even in handler-bearing bodies and keep the
    // residual (with its guard) only when the fold DECLINES.  The builtin
    // raise+catch path needs this so the
    // `raise ValueError`/`except ValueError` class loads fold to const.
    //
    // The fold resolves the `co_names` index the same way
    // `bh_load_global_fn` does and reaches production parity for
    // global-function-call loops when combined with the user-call inlining
    // path. Handler-bearing reachability also includes the builtins fallback.
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadGlobal
    {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                let name_idx = (namei as usize) >> 1;
                if !mark_trace_reads_module_global_from_code(
                    ctx.trace_ctx,
                    ns_ptr as pyre_object::PyObjectRef,
                    w_code_ptr,
                    name_idx,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadGlobal
    {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            // The live frame operand (r_args[2]) is needed for the builtins
            // fallback (`frame.get_builtin()`). Admitted inline callees carry
            // their own seeded portal frame; synthetic walkers may still omit
            // it, in which case only the module-dict cell path is attempted.
            let frame_ptr = r_args
                .get(2)
                .and_then(|&f| ctx.trace_ctx.box_value(f))
                .and_then(|v| match v {
                    majit_ir::Value::Ref(majit_ir::GcRef(p)) => Some(p),
                    _ => None,
                })
                .unwrap_or(0);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_load_global_cell_fold(
                    ctx, op.pc, dst, dst_bank, ns_ptr, w_code_ptr, frame_ptr, namei,
                )? {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadName fold: module-scope LOAD_NAME mirror of the LoadGlobal fold
    // above.  The residual is `bh_load_name_fn(frame, w_name, namei)`, so
    // r_args = [frame, w_name].  `try_walker_load_name_cell_fold` gates module
    // scope at runtime and routes non-module frames back to this residual.
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadName
    {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if !mark_trace_reads_module_global_from_frame_name(
                    ctx.trace_ctx,
                    frame_ptr,
                    w_name_ptr,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }

    // LoadName: descend `getitem_str` = `getdictvalue_no_unwrapping` +
    // `unwrap_cell`.  No handler gate: `except as` is a different
    // `next_instr` green from the inner while (`interp_jit.py
    // pypyjitdriver.greens`).
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadName
    {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_load_name_cell_fold(ctx, op.pc, dst, dst_bank, frame_ptr, w_name_ptr)?
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadSpecial fold: replace the `__enter__` / `__exit__` type lookup with
    // the constant descriptor plus an inline `Method` construction the
    // following CALL virtualizes away.  The residual is
    // `load_special_fn(obj, method_kind)`, so `r_args = [obj]` and
    // `i_args = [method_kind]`.
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadSpecial
    {
        if let (Some(&obj_opref), Some(&kind_opref)) = (r_args.first(), i_args.first()) {
            if let Some(majit_ir::Value::Int(method_kind)) = ctx.trace_ctx.box_value(kind_opref) {
                if spec_gate(SpecFold::LoadSpecialMethod, || {
                    try_walker_specialize_load_special_method(
                        ctx,
                        op.pc,
                        obj_opref,
                        method_kind,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadAttr fold (`mapdict.py LOAD_ATTR_caching`): fold a
    // monomorphic plain instance-attribute read to `guard_class` +
    // `guard_value(map)` + `getfield(storage)` + `getarrayitem(C_index)`,
    // eliding the opaque `getattr_fn` MRO-walk residual.  The residual is
    // `load_attr_fn(obj, code, name_idx)`, so `r_args = [obj, code]` and
    // `i_args = [name_idx]`.  A successful fold provably cannot raise (the map
    // guard proves the attribute is present on this shape), so it is attempted
    // even in handler-bearing bodies; every unfoldable shape falls through to
    // the residual (which keeps its exception guard).
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadAttr
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                // Resolve the name out of the jitcode's own `co_names` here
                // rather than inside the fold (mirrors `bh_load_attr_fn`; the
                // codewriter passes the raw index).  The fold's body is shared
                // with the `getattr(obj, "name")` builtin, which spells the
                // same `space.getattr` but carries its name as a constant
                // operand, so the lookup has to reach it already resolved.
                let attr_name = walker_load_name_from_code(w_code_ptr, namei as usize);
                // `descroperation.py getattr` selects a custom
                // `__getattribute__` before it runs the ordinary descriptor
                // protocol.  PyPy traces into that Python frame; do the same
                // before every default-attribute fold below (all of which
                // correctly decline the override but would otherwise leave
                // it in the opaque residual).
                if let Some(inlined) = try_walker_inline_getattribute_hook(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
                if let Some(attr_name) = attr_name.as_deref()
                    && spec_gate(SpecFold::LoadAttr, || {
                        try_walker_specialize_load_attr(
                            ctx, op.pc, obj_opref, attr_name, dst, dst_bank,
                        )
                    })?
                    .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // `complexobject.py complexwprop`: `real` / `imag` box the
                // lane with `space.newfloat`. Descend that getter here.
                if let Some(attr_name) = attr_name.as_deref()
                    && try_walker_orthodox_complex_member(
                        ctx, op.pc, obj_opref, attr_name, dst, dst_bank,
                    )?
                    .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // A name binding splits `super()` from the later LOAD_ATTR,
                // but PyPy still traces through the selected property's
                // getter.  Give that Python descriptor its inline path before
                // the Python-free super binding fold below declines it.
                if let Some(attr_name) = attr_name.as_deref()
                    && let Some(inlined) = try_walker_inline_super_proxy_property_get(
                        ctx, op, code, &r_args, call_descr, obj_opref, attr_name, dst, dst_bank,
                    )?
                {
                    return Ok(inlined);
                }
                // The plain-slot fold declines a `super` proxy: nothing on
                // it is a mapdict slot, and the MRO suffix `getattribute`
                // walks starts somewhere else entirely.
                if let Some(attr_name) = attr_name.as_deref()
                    && spec_gate(SpecFold::LoadAttrOnSuper, || {
                        try_walker_specialize_load_attr_on_super(
                            ctx, op.pc, obj_opref, attr_name, dst, dst_bank,
                        )
                    })?
                    .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // The plain-slot fold declines a `property` (data descriptor);
                // inline its Python getter instead of the opaque residual.
                if let Some(inlined) = try_walker_inline_property_get(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
                // `Object.descr__getattribute__` gives any other user data
                // descriptor the same traced call shape, with the descriptor,
                // receiver and owner type as explicit arguments.
                if let Some(inlined) = try_walker_inline_data_descriptor_get(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
                // `DescrOperation.get` on a non-data descriptor. Exact wrappers
                // unwrap on their own folds; a subclass `__get__` is recorded
                // here so the override runs.
                if let Some(inlined) = try_walker_inline_nondatadescr_get(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
                // The name resolves nowhere and the type defines `__getattr__`:
                // inline the hook in place of the miss walk plus its frame.
                if let Some(inlined) = try_walker_inline_getattr_hook(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
                // A type receiver whose class-MRO value needs no descriptor
                // binding folds to that value under receiver + version pins.
                if spec_gate(SpecFold::LoadTypeAttr, || {
                    try_walker_specialize_load_type_attr(
                        ctx,
                        op.pc,
                        obj_opref,
                        w_code_ptr,
                        namei as usize,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // A bare `g = o.m` read, which `space.getattr` answers with a
                // `Method` exactly as the fused method-call shape does.  That
                // shape is left to the block below, where `load_method_attr`
                // reaches the same receiver by pushing `[w_descr, w_obj]`
                // without allocating at all; only a read with no
                // `load_method_self` after it is answered here.
                if !next_op_is_load_method_self_for_attr(code, op, ctx, dst)
                    && spec_gate(SpecFold::LoadBoundMethodAttr, || {
                        try_walker_specialize_load_bound_method_attr(
                            ctx,
                            op.pc,
                            obj_opref,
                            w_code_ptr,
                            namei as usize,
                            dst,
                            dst_bank,
                        )
                    })?
                    .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    // STORE_ATTR property setter: the plain-slot store fold above declines a
    // `property` data descriptor; inline its Python setter instead of the
    // opaque `setattr` residual.  `store_attr_fn` r_args = [obj, value, code].
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::StoreAttr
    {
        if let (Some(&obj_opref), Some(&value_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(inlined) = try_walker_inline_property_set(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    value_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
            }
        }
    }
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadAttr
        && next_op_is_load_method_self_for_attr(code, op, ctx, dst)
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if spec_gate(SpecFold::LoadMethodAttr, || {
                    try_walker_specialize_load_method_attr(
                        ctx,
                        op.pc,
                        obj_opref,
                        w_code_ptr,
                        namei as usize,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // `Type.cmethod(...)`: the receiver is a class and the name
                // resolves to a `classmethod`, which `load_method_fast_path`
                // declines (non-instance receiver, non-method descriptor).
                // Write the classmethod's `__func__` so the paired
                // `load_method_self` binds the class and the CALL inlines it.
                if spec_gate(SpecFold::LoadClassmethodAttr, || {
                    try_walker_specialize_load_classmethod_attr(
                        ctx,
                        op.pc,
                        obj_opref,
                        w_code_ptr,
                        namei as usize,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // The `[w_descr, w_obj]` push above is restricted to
                // `flag_method_descriptor` types, so a builtin method on a
                // builtin receiver (`lst.append`) leaves `getattr` to build a
                // `Method`.  Emit that construction instead of the opaque
                // residual so it virtualizes into the following CALL.
                if spec_gate(SpecFold::LoadBoundMethodAttr, || {
                    try_walker_specialize_load_bound_method_attr(
                        ctx,
                        op.pc,
                        obj_opref,
                        w_code_ptr,
                        namei as usize,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    // `super(C, self).name` / `super().name`: emit the `Method`
    // `W_Super.getattribute` builds instead of the opaque residual that
    // rebuilds the proxy, re-walks the MRO suffix and re-binds the descriptor
    // every iteration.  Any non-matching shape falls through to the generic
    // residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadSuperAttr
    {
        if let (
            Some(&global_super_opref),
            Some(&self_opref),
            Some(&cls_opref),
            Some(&code_opref),
            Some(&namei_opref),
        ) = (
            r_args.first(),
            r_args.get(1),
            r_args.get(2),
            r_args.get(4),
            i_args.first(),
        ) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(name) = walker_load_name_from_code(w_code_ptr, namei as usize)
                    && let Some(inlined) = try_walker_inline_super_property_get(
                        ctx,
                        op,
                        code,
                        &r_args,
                        call_descr,
                        global_super_opref,
                        self_opref,
                        cls_opref,
                        &name,
                        dst,
                        dst_bank,
                    )?
                {
                    return Ok(inlined);
                }
                if spec_gate(SpecFold::LoadSuperAttr, || {
                    try_walker_specialize_load_super_attr(
                        ctx,
                        op.pc,
                        global_super_opref,
                        self_opref,
                        cls_opref,
                        w_code_ptr,
                        namei as usize,
                        dst,
                        dst_bank,
                    )
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    // The method form's `[func, self_or_null]` split.  Paired with the fold
    // above: without it the two residuals FORCE the still-virtual `Method`,
    // which costs a real allocation the fold alone would not pay.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::SuperAttrUnwrap
    {
        if let (Some(&raw_opref), Some(&which_opref)) = (r_args.first(), i_args.first()) {
            if let Some(majit_ir::Value::Int(which)) = ctx.trace_ctx.box_value(which_opref) {
                if spec_gate(SpecFold::SuperAttrUnwrap, || {
                    try_walker_fold_super_attr_unwrap(ctx, op.pc, raw_opref, which, dst, dst_bank)
                })?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    if ctx.is_authoritative_executor
        && foldable_runtime_helper == majit_ir::RuntimeHelperKind::LoadMethodSelf
    {
        if let (Some(&namei_opref), Some(&obj_opref), Some(&attr_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1), r_args.get(2))
        {
            let r_len_pc = op.pc + 1 + 1 + i_width;
            let attr_reg = code
                .get(r_len_pc + 1 + 1)
                .copied()
                .map(usize::from)
                .unwrap_or(usize::MAX);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_fold_load_method_self(
                    ctx,
                    op.pc,
                    obj_opref,
                    attr_opref,
                    attr_reg,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // Defer the arg-bound check past the short-circuiting LoadConst /
    // LoadGlobal folds above: each resolves the call to a constant from
    // `i_args`/`r_args` without recording it, so an unbound *trailing* arg
    // is irrelevant when the call folds away. Synthetic walkers can still
    // present an unbound trailing frame operand; the fold elides that call,
    // so the frame box never needs binding. Only a call that survives to a genuine record
    // (BoxInt exec, generic residual below) requires every box bound.
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    // BoxInt fold (#62): `box_int_fn(raw)` allocates a fresh `PyLong`.  The
    // opaque CanRaise residual the generic leg would record blocks the
    // optimizer (no DCE of an unused/round-tripped box).  Emit the
    // virtualizable `new_with_vtable` + `setfield_gc` form (`wrapint`,
    // identical to the BINARY_OP result box) so a following unbox
    // (`getfield_gc_pure`) forwards through the setfield and the box DCEs
    // when it never escapes.  The concrete shadow carries the authentic
    // boxed pointer so downstream specializations still see a concrete int.
    if foldable_runtime_helper == majit_ir::RuntimeHelperKind::BoxInt && dst_bank == 'r' {
        if let Some(&raw_arg) = i_args.first() {
            if let Some(boxed_ptr) = walker_execute_may_force_boxed(ctx, &allboxes, call_descr) {
                let intval =
                    unsafe { pyre_object::w_int_get_value(boxed_ptr as pyre_object::PyObjectRef) };
                let boxed = walker_box_int(ctx, op.pc, raw_arg, intval)?;
                // wrapint allocates; the concrete is that heap box,
                // not the LOAD_SMALL_INT intern table.
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(intval, boxed_ptr));
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, boxed)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // #57: speculative int specialization for the BINARY_OP / COMPARE_OP
    // helper (oopspec `BinaryOp` / `CompareOp`, set codewriter-side at
    // flatten.rs `build_residual_call_ir_r_insn_from_operands`).  When both
    // operands are concrete `W_IntObject`, re-emit the guard_class + unbox
    // + int_OP (+ rebox / bool-box) sequence instead of an opaque
    // CALL_MAY_FORCE, matching the retired trait-side int binop / compare
    // paths.  Falls through to the generic record for
    // non-int operands / deferred operators.
    //
    // The eval-loop codewriter residual can arrive with
    // `RuntimeHelperKind::None`; identify the helper by fnaddr so
    // `binary_op_descent` still runs after `binary_op_int` retired.
    let func_addr = match ctx.trace_ctx.box_value(funcptr) {
        Some(majit_ir::Value::Int(n)) => n,
        _ => 0,
    };
    let opcode_binary_fnaddr = func_addr != 0 && BINARY_VALUE_FROM_TAG_FNADDRS.contains(&func_addr);
    let opcode_compare_fnaddr =
        func_addr != 0 && COMPARE_VALUE_FROM_TAG_FNADDRS.contains(&func_addr);
    let is_binary_op =
        foldable_runtime_helper == majit_ir::RuntimeHelperKind::BinaryOp || opcode_binary_fnaddr;
    let is_compare_op =
        foldable_runtime_helper == majit_ir::RuntimeHelperKind::CompareOp || opcode_compare_fnaddr;
    if is_binary_op || is_compare_op {
        if let Some(&tag_opref) = i_args.first() {
            if let Some(majit_ir::Value::Int(op_tag)) = ctx.trace_ctx.box_value(tag_opref) {
                let specialized = if is_binary_op {
                    let is_subscr = matches!(
                        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
                        Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
                    );
                    if is_subscr {
                        // A user-instance receiver resolves `__getitem__` on
                        // its own type; inline that body instead of leaving
                        // the subscript an opaque residual.  The storage folds
                        // below are for builtin containers, which this
                        // declines.
                        if let Some(inlined) = spec_gate(SpecFold::SubscrUserGetitem, || {
                            try_walker_inline_subscr_getitem(
                                ctx,
                                op,
                                code,
                                Some(funcptr),
                                &r_args,
                                call_descr,
                                dst,
                                dst_bank,
                            )
                        })? {
                            return Ok(inlined);
                        }
                        // Int-strategy miss: `dict.lookup` on dstorage +
                        // `int_lt < 0` then `raise_key_error`
                        // (`ll_dict_getitem_with_hash`).  The hit fold
                        // emits the same lookup's fail arm; residual
                        // CallMayForce would force the virtual KeyError
                        // the inlined except is about to catch.
                        if let Some(outcome) = spec_gate(SpecFold::Subscr, || {
                            try_walker_specialize_subscr_int_miss(ctx, op.pc, &r_args)
                        })? {
                            return Ok((outcome, op.next_pc));
                        }
                        // BINARY_SUBSCR list[int] getitem (int/float storage);
                        // falls through to the generic may-force leg otherwise.
                        spec_gate(SpecFold::Subscr, || {
                            try_walker_specialize_subscr(
                                ctx, op.pc, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )
                        })?
                    } else {
                        // The exact-int zero-divisor raise runs BEFORE the
                        // descent: the descended body reaches its
                        // `ZeroDivisionError` through an opaque published
                        // materialiser, whose concrete result no longer
                        // virtualizes (see
                        // `try_walker_specialize_binary_op_int_zero_div`).
                        if let Some(outcome) = spec_gate(SpecFold::BinaryOpIntZeroDiv, || {
                            try_walker_specialize_binary_op_int_zero_div(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst_bank,
                            )
                        })? {
                            return Ok((outcome, op.next_pc));
                        }
                        // Descend the helper whole ahead of the hand folds, so
                        // their `consulted` counts read whether the descent
                        // took the site.
                        if let Some(outcome) = spec_gate(SpecFold::BinaryOpDescent, || {
                            try_walker_orthodox_binary_op(
                                ctx, op.pc, op_tag, tag_opref, &r_args, dst, dst_bank,
                            )
                        })? {
                            return Ok((outcome, op.next_pc));
                        }
                        // Exact int pairs and long `**` are recorded by
                        // `binary_op_descent` (`long_pow` →
                        // `jit_bigint_int_pow_nomod`).
                        None
                    }
                } else if op_tag == 10 && ctx.is_authoritative_executor {
                    // `op_tag == 10` is CHECK_EXC_MATCH
                    // (`compare_value_from_tag(exc, match_type, 10)`,
                    // `call_jit.rs`).  Fold the match concretely to a
                    // const bool (the immortal TRUE/FALSE singleton) so the
                    // exception stays virtual and DCEs, eliding the may-force
                    // compare + its truth-extract residual.  Declines (falls
                    // through to the remaining compare attempts, which also
                    // decline for Ref operands → generic residual) when an
                    // operand has no concrete shadow or the match target is
                    // not a valid exception class.
                    let keep_last_exc_for_handler =
                        reads_last_exc_before_next_catch(code, op.next_pc);
                    let folded =
                        try_walker_fold_check_exc_match(ctx, op.pc, &r_args, dst, dst_bank)?;
                    if folded.is_some() && keep_last_exc_for_handler {
                        ctx.restore_last_exc_value(
                            saved_last_exc_value,
                            saved_last_exc_value_concrete,
                        );
                    }
                    folded
                } else {
                    // Descend the helper whole ahead of the hand folds, so
                    // their `consulted` counts read whether the descent
                    // took the site. Tags 8/9 (`is` / `is_not`) record
                    // `compare_value_from_tag` → `ObjSpace.is_w` on this
                    // same descent. Bytecode `IS_OP` does not reach here:
                    // it is an `inline_call` of `runtime_ops::is_op`.
                    if let Some(outcome) = spec_gate(SpecFold::CompareOpDescent, || {
                        try_walker_orthodox_compare_op(
                            ctx, op.pc, op_tag, tag_opref, &r_args, dst, dst_bank,
                        )
                    })? {
                        return Ok((outcome, op.next_pc));
                    }
                    // Exact int, long and str compares are recorded by the
                    // descent above (`compare_value_from_tag` →
                    // `compare_slot`). Short exact tuples of ints or None are
                    // folded by `try_walker_fold_small_tuple_eq`. Longer
                    // tuples and subclasses still reach this residual.
                    None
                };
                if specialized.is_some() {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                if is_binary_op {
                    if let Some(inlined) = try_walker_inline_user_binop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                }
                if is_compare_op {
                    if let Some(inlined) = try_walker_inline_user_compareop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                    // CONTAINS_OP is outside `compare_op_from_tag`, so the
                    // rich-compare inline above declines it.
                    if pyre_interpreter::runtime_ops::compare_op_tag_is_contains(op_tag) {
                        if let Some(inlined) = try_walker_inline_contains_dunder(
                            ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                        )? {
                            return Ok(inlined);
                        }
                    }
                }
            }
        }
    }

    // UNPACK_SEQUENCE fold (#73): read an arity-2 specialised int tuple's
    // elements directly so the unpacked items stay unboxed ints (the trait
    // path's value0/value1 fold).  A non-foldable shape falls through to the
    // opaque residual below — correct, no decline.
    if matches!(
        foldable_runtime_helper,
        majit_ir::RuntimeHelperKind::UnpackSequence | majit_ir::RuntimeHelperKind::UnpackItem
    ) && spec_gate(SpecFold::Unpack, || {
        try_walker_specialize_unpack(
            ctx,
            op.pc,
            foldable_runtime_helper,
            &i_args,
            &r_args,
            &allboxes,
            call_descr,
            dst,
            dst_bank,
        )
    })?
    .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    if try_fold_registered_symbolic_residual(ctx, &allboxes, op.pc, dst, dst_bank)? {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py forces-branch sub-case: route release-gil through
    // `direct_call_release_gil`.  Mirrors `dispatch_residual_call_iRd_kind`.
    // pyjitpl.py forces-branch sub-case, ahead of the release-gil one:
    //     if effectinfo.oopspecindex == effectinfo.OS_LIBFFI_CALL:
    //         resbox = self.metainterp.direct_libffi_call(allboxes, c_result, descr)
    //     if resbox is None: ...release_gil / may_force...
    // A decline lands on exactly that fallthrough.
    let libffi_recorded = if ei.oopspecindex == majit_ir::OopSpecIndex::LibffiCall {
        match direct_libffi_call(
            ctx,
            ei,
            &allboxes,
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRd_kind",
        )? {
            LibffiCallOutcome::Raised(outcome) => return Ok((outcome, op.next_pc)),
            LibffiCallOutcome::Recorded => true,
            LibffiCallOutcome::Declined => false,
        }
    } else {
        false
    };
    if libffi_recorded {
        // `direct_libffi_call` recorded the specialized call and both of the
        // forces branch's guards.
    } else if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py EF_LOOPINVARIANT short-circuit; no IR
        // op, no guard, ordering moot.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers run in the residual-call execution path.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
            write_back_locals_for_proxy_reader(ctx, &allboxes);
        }

        let profiled_call = matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        );
        if profiled_call {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
        }
        // `pyjitpl.py:1943` takes `patch_pos` before recording the call so
        // `record_result_of_call_pure` can cut it back out.
        let patch_pos = ctx.trace_ctx.get_trace_position();
        let (recorded, is_list_wb) =
            record_list_write_barrier_residual(ctx, funcptr, &allboxes, call_opcode, descr.clone());

        // `MIFrame.execute_varargs(pure=True)` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        let recorded = try_fold_pure_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            descr.clone(),
            patch_pos,
            recorded,
        );
        // See `dispatch_residual_call_iRd_kind`: count only a standing call.
        if profiled_call && recorded.inline_const_to_value().is_none() {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }

        // Non-elidable concrete-execute parity — see
        // `dispatch_residual_call_iRd_kind` for the full citation.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        // A namespace write reaches the executor as an ordinary residual and
        // mutates the frame's dict in place; record what it displaces while the
        // write is still entirely ahead of the walk.
        let namespace_write_journaled =
            journal_walker_namespace_write(ctx, ei.runtime_helper, &r_args).is_some();
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
            namespace_write_journaled,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  A `CallPure*` reaches this
        // dispatcher too when its fold could not answer, and declines as
        // `PureUnfolded` when it also cannot raise — no effect and no
        // exception, so no such obligation; the can-raise half declines as
        // `Symbolic` and keeps it.
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                // The call op is already recorded. A symbolic fnaddr must
                // not stay in it whatever the decline cause (a `PureUnfolded`
                // `CallPure*` keeps the call too): the backend emits a call
                // to the hash. Cut back to the position taken before the
                // record, then abort this walk so the result box is not used.
                if let Some(addr) = recorded_symbolic_funcptr(ctx, &allboxes) {
                    ctx.trace_ctx.cut_trace_with_snapshots(patch_pos);
                    return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                        pc: op.pc,
                        symbolic: addr,
                    });
                }
                fbw_abort_nested_unjournaled_residual(ctx, op.pc, Some(cause))?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // `_record_helper_varargs` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.  Same invalidation semantics; only the
        // arglist construction differs (boxes2 = i_args ++ r_args).
        // `COND_CALL_GC_WB` is not a call (`resoperation.py`).
        if !is_list_wb {
            ctx.trace_ctx
                .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        }
        // pyjitpl.py _opimpl_residual_call*: result writeback runs
        // BEFORE handle_possible_exception().  See
        // `dispatch_residual_call_iRd_kind` for the full citation.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise && !is_list_wb {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc)?;
                // pyjitpl.py `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value()
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete();
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // The mixed int/ref residual-call shape must use the same
                // exception-region resume as the one-ref shape above.  In
                // particular, UNPACK_SEQUENCE and UNPACK_EX pass their arity
                // as Int arguments and the sequence as a Ref argument.  If
                // their validation raises after a guard failure, resume at the
                // call's own catch so the enclosing handler sees the error;
                // the generic fallthrough may already be outside the covered
                // exception-table range.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `residual_call` shape `iIRFd>X` dispatcher — `_irf_*` arglist with
/// int + ref + float lists before the descr. RPython parity:
/// `pyjitpl.py _opimpl_residual_call3` (`@arguments` argspec
/// `"box", "boxes3", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1` / `_call2`. The
/// `boxes3` argcode (`pyjitpl.py`) decodes three adjacent
/// count-prefixed lists into one concatenated `argboxes` array
/// `[i_args..., r_args..., f_args...]`. `_build_allboxes`
/// (`pyjitpl.py`, ported to [`build_allboxes`]) permutes
/// those to match `descr.get_arg_types()` ABI ordering.
///
/// Operand layout `iIRFd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 1B f-list count + K×1B f-regs + 2B descr + 1B
///   `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iIRd_kind`; all sub-cases (release-gil,
/// loop-invariant, default) route through the same helpers
/// ([`select_residual_call_opcode`], [`direct_call_release_gil`],
/// [`loopinvariant_lookup`] / [`loopinvariant_now_known`]).
#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRFd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    ctx.clear_last_exc_value();
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let (f_args, f_width) = read_float_var_list(code, op, 1 + i_width + r_width, ctx)?;
    let descr_offset = 1 + i_width + r_width + f_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_irf_v/iIRFd` (`pyjitpl.py opimpl_residual_call_irf_v =
    // _opimpl_residual_call3`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args ++ f_args (`boxes3` argcode order).
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    let mut argbox_types: Vec<Type> =
        Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    argboxes.extend_from_slice(&f_args);
    argbox_types.extend(std::iter::repeat(Type::Float).take(f_args.len()));
    let allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        call_descr.arg_types(),
        None,
    );
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    clear_walk_exception(ctx);
    if let Some(outcome) =
        do_not_in_trace_call_result(ctx, ei, &allboxes, call_descr, dst_bank, op.pc)?
    {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    if try_fold_registered_symbolic_residual(ctx, &allboxes, op.pc, dst, dst_bank)? {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py forces-branch sub-case, ahead of the release-gil one:
    //     if effectinfo.oopspecindex == effectinfo.OS_LIBFFI_CALL:
    //         resbox = self.metainterp.direct_libffi_call(allboxes, c_result, descr)
    //     if resbox is None: ...release_gil / may_force...
    // A decline lands on exactly that fallthrough.
    let libffi_recorded = if ei.oopspecindex == majit_ir::OopSpecIndex::LibffiCall {
        match direct_libffi_call(
            ctx,
            ei,
            &allboxes,
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRFd_kind",
        )? {
            LibffiCallOutcome::Raised(outcome) => return Ok((outcome, op.next_pc)),
            LibffiCallOutcome::Recorded => true,
            LibffiCallOutcome::Declined => false,
        }
    } else {
        false
    };
    if libffi_recorded {
        // `direct_libffi_call` recorded the specialized call and both of the
        // forces branch's guards.
    } else if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRFd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRFd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers run in the residual-call execution path.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
            write_back_locals_for_proxy_reader(ctx, &allboxes);
        }

        let profiled_call = matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        );
        if profiled_call {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
        }
        // `pyjitpl.py:1943` takes `patch_pos` before recording the call so
        // `record_result_of_call_pure` can cut it back out.
        let patch_pos = ctx.trace_ctx.get_trace_position();
        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // `MIFrame.execute_varargs(pure=True)` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        let recorded = try_fold_pure_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            descr.clone(),
            patch_pos,
            recorded,
        );
        // See `dispatch_residual_call_iRd_kind`: count only a standing call.
        if profiled_call && recorded.inline_const_to_value().is_none() {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }
        // `boxes3`-shaped may-force residual (`CallMayForce{R,I,F,N}`):
        // execute concretely under the authoritative walk and stamp the
        // result, identically to the `iRd` / `iIRd` siblings.
        // `do_residual_call` (`pyjitpl.py _opimpl_residual_call3`)
        // is arglist-shape-independent, so the float-arg shape needs the same
        // execution path — e.g. a float-returning helper such as `math.sqrt`
        // records here as `iIRFd>f` (empty i/r lists), and its result must be
        // made concrete so the downstream float math can specialize.  Void
        // float-stores (`irf_v`) are caught inside the helper's
        // `result_type() == Void` arm and deferred (#61), so the compiled
        // loop's re-run does not double-apply the store.

        // Non-elidable concrete-execute parity — see
        // `dispatch_residual_call_iRd_kind` for the full citation.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        // A namespace write reaches the executor as an ordinary residual and
        // mutates the frame's dict in place; record what it displaces while the
        // write is still entirely ahead of the walk.
        let namespace_write_journaled =
            journal_walker_namespace_write(ctx, ei.runtime_helper, &r_args).is_some();
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
            namespace_write_journaled,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  A `CallPure*` reaches this
        // dispatcher too when its fold could not answer, and declines as
        // `PureUnfolded` when it also cannot raise — no effect and no
        // exception, so no such obligation; the can-raise half declines as
        // `Symbolic` and keeps it.
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                // The call op is already recorded. A symbolic fnaddr must
                // not stay in it whatever the decline cause (a `PureUnfolded`
                // `CallPure*` keeps the call too): the backend emits a call
                // to the hash. Cut back to the position taken before the
                // record, then abort this walk so the result box is not used.
                if let Some(addr) = recorded_symbolic_funcptr(ctx, &allboxes) {
                    ctx.trace_ctx.cut_trace_with_snapshots(patch_pos);
                    return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                        pc: op.pc,
                        symbolic: addr,
                    });
                }
                fbw_abort_nested_unjournaled_residual(ctx, op.pc, Some(cause))?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRFd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc)?;
                // pyjitpl.py `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value()
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete();
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Shared `I`/`R`/`d` tail of the canonical cond/record layouts
/// (`iiIRd` / `riIRd` after the leading value + funcptr registers).
fn read_cond_record_ir_tail<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, Vec<Type>, DescrRef, usize), DispatchError> {
    let (i_args, i_width) = read_int_var_list(code, op, 2, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 2 + i_width, ctx)?;
    let descr_offset = 2 + i_width + r_width;
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len());
    let mut argbox_types: Vec<Type> = Vec::with_capacity(i_args.len() + r_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    Ok((argboxes, argbox_types, descr, descr_offset))
}

fn cond_record_call_descr<'a>(
    descr: &'a DescrRef,
    code: &[u8],
    op: &DecodedOp,
    descr_offset: usize,
) -> Result<&'a dyn majit_ir::CallDescr, DispatchError> {
    descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index: decode_descr_index(code, op, descr_offset),
        })
}

fn cond_record_concrete_args<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
) -> Option<Vec<i64>> {
    let mut concrete_args = Vec::with_capacity(allboxes.len().saturating_sub(2));
    for boxref in allboxes.iter().skip(2) {
        match ctx.trace_ctx.box_value(*boxref) {
            Some(majit_ir::Value::Int(n)) => concrete_args.push(n),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(p))) => concrete_args.push(p as i64),
            _ => return None,
        }
    }
    Some(concrete_args)
}

fn cond_record_arg_values<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
) -> Option<Vec<majit_ir::Value>> {
    let mut values = Vec::with_capacity(allboxes.len());
    for &boxref in allboxes {
        values.push(ctx.trace_ctx.box_value(boxref)?);
    }
    Some(values)
}

fn cond_record_pure_result(concrete: ConcreteValue) -> Option<majit_ir::Value> {
    match concrete {
        ConcreteValue::Int(n) => Some(majit_ir::Value::Int(n)),
        ConcreteValue::Ref(p) => Some(majit_ir::Value::Ref(majit_ir::GcRef(p as usize))),
        ConcreteValue::Null => None,
        _ => None,
    }
}

/// `MIFrame.execute_varargs` exception bookkeeping after a recorded
/// `COND_CALL` / `COND_CALL_VALUE`. Consumes `BH_LAST_EXC_VALUE` the way
/// `try_execute_residual_call_via_executor` does, then emits
/// `GUARD_EXCEPTION` + `SubRaise` or `GUARD_NO_EXCEPTION`.
fn cond_record_handle_exception<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    ei: &majit_ir::descr::EffectInfo,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let can_raise = ei.check_can_raise(false);
    let bh_exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 {
        let bh_exc_box = ctx.trace_ctx.const_ref(bh_exc);
        ctx.set_last_exc_value(
            bh_exc_box,
            ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef),
        );
        ctx.fbw_mode.class_of_last_exc_is_const = false;
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        debug_assert!(
            can_raise,
            "conditional_call helper raised on a !can_raise EffectInfo"
        );
        if can_raise {
            walker_record_guard_exception(ctx, op.pc)?;
            let exc = ctx
                .last_exc_value()
                .expect("cond_record_handle_exception seeded last_exc_value");
            let exc_concrete = ctx.last_exc_value_concrete();
            return Ok(Some((
                DispatchOutcome::SubRaise { exc, exc_concrete },
                op.next_pc,
            )));
        }
    } else if can_raise {
        ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    Ok(None)
}

fn concrete_from_box_value(value: Option<majit_ir::Value>) -> ConcreteValue {
    match value {
        Some(majit_ir::Value::Int(n)) => ConcreteValue::Int(n),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(p))) => {
            ConcreteValue::Ref(p as pyre_object::PyObjectRef)
        }
        _ => ConcreteValue::Null,
    }
}

fn concrete_ref_is_null(value: ConcreteValue) -> bool {
    match value {
        ConcreteValue::Null => true,
        ConcreteValue::Ref(p) => p.is_null() || p == pyre_object::PY_NULL,
        _ => false,
    }
}

/// `pyjitpl.py opimpl_conditional_call_ir_v` / `do_conditional_call`.
///
/// Operand layout `iiIRd`: condition, funcptr, I-list, R-list, descr.
/// A constant-false condition records nothing so the heapcache can keep
/// the arguments virtual.
pub(crate) fn dispatch_conditional_call_ir_v<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let cond = read_int_reg(code, op, 0, ctx)?;
    let funcptr = read_int_reg(code, op, 1, ctx)?;
    let (argboxes, argbox_types, descr, descr_offset) = read_cond_record_ir_tail(code, op, ctx)?;
    let call_descr = cond_record_call_descr(&descr, code, op, descr_offset)?;
    // pyjitpl.py `opimpl_conditional_call_ir_v`: ConstInt(0) returns without recording.
    if cond.is_constant() {
        let zero = matches!(ctx.trace_ctx.box_value(cond), Some(majit_ir::Value::Int(0)));
        if zero {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    let allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        call_descr.arg_types(),
        Some(cond),
    );
    let ei = call_descr.get_extra_info().clone();
    let arg_types = call_descr.arg_types().to_vec();
    assert!(
        !ei.check_forces_virtual_or_virtualizable(),
        "conditional_call target must not force virtualizable"
    );
    ctx.trace_ctx
        .profiler()
        .count_ops(OpCode::CondCallN, majit_metainterp::counters::OPS);
    ctx.trace_ctx
        .profiler()
        .count_ops(OpCode::CondCallN, majit_metainterp::counters::RECORDED_OPS);
    let _recorded = ctx
        .trace_ctx
        .record_op_with_descr(OpCode::CondCallN, &allboxes, descr);
    // `_record_helper_varargs` runs `heapcache.invalidate_caches_varargs`
    // before it appends. `record_op_with_descr` does not, so do it here
    // the same way residual_call does.
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(OpCode::CondCallN, Some(&ei), &allboxes);
    let cond_true = match ctx.trace_ctx.box_value(cond) {
        Some(majit_ir::Value::Int(n)) => n != 0,
        _ => match read_int_reg_concrete(code, op, 0, ctx) {
            ConcreteValue::Int(n) => n != 0,
            ConcreteValue::Bool(b) => b,
            _ => false,
        },
    };
    ctx.clear_last_exc_value();
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    if cond_true && ctx.is_authoritative_executor {
        if let Some(majit_ir::Value::Int(func_addr)) = ctx.trace_ctx.box_value(funcptr)
            && func_addr != 0
            && !majit_jitcode::codewriter::call::is_symbolic_fnaddr(func_addr)
            && let Some(concrete_args) = cond_record_concrete_args(ctx, &allboxes)
        {
            majit_metainterp::call_void_function_typed(
                func_addr as *const (),
                &concrete_args,
                &arg_types,
            );
        } else {
            // `do_conditional_call` runs `executor.execute_varargs` whenever
            // the condition holds; the body after this op reads what the
            // callee wrote.
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[fbw-cond-call] not executed pc={} funcptr={:?} args={:?}",
                    op.pc,
                    ctx.trace_ctx.box_value(funcptr),
                    allboxes
                        .iter()
                        .map(|&b| (b, ctx.trace_ctx.box_value(b)))
                        .collect::<Vec<_>>(),
                );
            }
            walker_gc_store_not_executed(ctx, op.pc)?;
        }
    }
    if let Some(out) = cond_record_handle_exception(ctx, op, &ei)? {
        return Ok(out);
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py _opimpl_conditional_call_value` / `do_conditional_call(is_value=True)`.
///
/// `iiIRd>i`: value, funcptr, I-list, R-list, descr, dst.
fn dispatch_conditional_call_value_ir<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    first: OpRef,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let funcptr = read_int_reg(code, op, 1, ctx)?;
    let (argboxes, argbox_types, descr, descr_offset) = read_cond_record_ir_tail(code, op, ctx)?;
    let call_descr = cond_record_call_descr(&descr, code, op, descr_offset)?;
    let dst = code[op.pc + 1 + descr_offset + 2] as usize;
    // `_opimpl_conditional_call_value`: Const nonnull returns the value
    // box without recording, so the heapcache can keep args virtual.
    if first.is_constant() {
        let nonnull = match ctx.trace_ctx.box_value(first) {
            Some(majit_ir::Value::Int(n)) => n != 0,
            Some(majit_ir::Value::Ref(majit_ir::GcRef(p))) => p != 0,
            _ => false,
        };
        if nonnull {
            let concrete = concrete_from_box_value(ctx.trace_ctx.box_value(first));
            match dst_bank {
                'i' => write_int_reg(ctx, op.pc, dst, first, concrete)?,
                'r' => write_ref_reg(ctx, op.pc, dst, first, concrete)?,
                _ => unreachable!("cond_call_value dst bank is i or r"),
            }
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    let allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        call_descr.arg_types(),
        Some(first),
    );
    let ei = call_descr.get_extra_info().clone();
    let arg_types = call_descr.arg_types().to_vec();
    assert!(
        !ei.check_forces_virtual_or_virtualizable(),
        "conditional_call_value target must not force virtualizable"
    );
    ctx.trace_ctx
        .profiler()
        .count_ops(opcode, majit_metainterp::counters::OPS);
    ctx.trace_ctx
        .profiler()
        .count_ops(opcode, majit_metainterp::counters::RECORDED_OPS);
    // `do_conditional_call(is_value=True)` → `execute_varargs(..., pure=True)`:
    // snapshot the recorder so `record_result_of_call_pure` can cut an
    // all-constant COND_CALL_VALUE back out.
    let patch_pos = ctx.trace_ctx.get_trace_position();
    let descr_for_pure = descr.clone();
    let recorded = ctx.trace_ctx.record_op_with_descr(opcode, &allboxes, descr);
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(opcode, Some(&ei), &allboxes);
    let should_call = match ctx.trace_ctx.box_value(first) {
        Some(majit_ir::Value::Int(n)) => n == 0,
        Some(majit_ir::Value::Ref(majit_ir::GcRef(p))) => p == 0,
        _ => match dst_bank {
            'i' => matches!(
                read_int_reg_concrete(code, op, 0, ctx),
                ConcreteValue::Int(0)
            ),
            'r' => concrete_ref_is_null(read_ref_reg_concrete(code, op, 0, ctx)),
            _ => false,
        },
    };
    ctx.clear_last_exc_value();
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let (result, concrete) = if should_call && ctx.is_authoritative_executor {
        if let Some(majit_ir::Value::Int(func_addr)) = ctx.trace_ctx.box_value(funcptr)
            && func_addr != 0
            && !majit_jitcode::codewriter::call::is_symbolic_fnaddr(func_addr)
            && let Some(concrete_args) = cond_record_concrete_args(ctx, &allboxes)
        {
            match dst_bank {
                'i' => {
                    let n = majit_metainterp::call_int_function_typed(
                        func_addr as *const (),
                        &concrete_args,
                        &arg_types,
                    );
                    (recorded, ConcreteValue::Int(n))
                }
                'r' => {
                    let p = majit_metainterp::call_ref_function_typed(
                        func_addr as *const (),
                        &concrete_args,
                        &arg_types,
                    );
                    (
                        recorded,
                        ConcreteValue::Ref(p as usize as pyre_object::PyObjectRef),
                    )
                }
                _ => (recorded, ConcreteValue::Null),
            }
        } else {
            (recorded, ConcreteValue::Null)
        }
    } else if should_call {
        (recorded, ConcreteValue::Null)
    } else {
        (
            first,
            concrete_from_box_value(ctx.trace_ctx.box_value(first)),
        )
    };
    // `execute_varargs(..., pure=True)` then `record_result_of_call_pure`.
    // Skip when the walk had no concrete result (non-authoritative /
    // symbolic target): folding a Null would invent a constant.
    // Also skip when the helper raised (`not last_exc_value`).
    let raised = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get()) != 0;
    let result = if raised {
        result
    } else {
        match cond_record_pure_result(concrete) {
            Some(result_value) => {
                if let Some(arg_values) = cond_record_arg_values(ctx, &allboxes) {
                    ctx.trace_ctx.record_result_of_call_pure(
                        recorded,
                        &allboxes,
                        &arg_values,
                        descr_for_pure,
                        patch_pos,
                        opcode,
                        result_value,
                    )
                } else {
                    result
                }
            }
            None => result,
        }
    };
    match dst_bank {
        'i' => write_int_reg(ctx, op.pc, dst, result, concrete)?,
        'r' => write_ref_reg(ctx, op.pc, dst, result, concrete)?,
        _ => unreachable!("cond_call_value dst bank is i or r"),
    }
    if let Some(out) = cond_record_handle_exception(ctx, op, &ei)? {
        return Ok(out);
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

pub(crate) fn dispatch_conditional_call_value_ir_i<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let first = read_int_reg(code, op, 0, ctx)?;
    dispatch_conditional_call_value_ir(code, op, ctx, first, OpCode::CondCallValueI, 'i')
}

pub(crate) fn dispatch_conditional_call_value_ir_r<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let first = read_ref_reg(code, op, 0, ctx)?;
    dispatch_conditional_call_value_ir(code, op, ctx, first, OpCode::CondCallValueR, 'r')
}

/// `pyjitpl.py opimpl_record_known_result_i_ir_v` / `blackhole.py
/// bhimpl_record_known_result_{i,r}_ir_v`. Record only; the body is `pass`.
fn dispatch_record_known_result_ir<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    first: OpRef,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let funcptr = read_int_reg(code, op, 1, ctx)?;
    let (argboxes, argbox_types, descr, descr_offset) = read_cond_record_ir_tail(code, op, ctx)?;
    let call_descr = cond_record_call_descr(&descr, code, op, descr_offset)?;
    let allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        call_descr.arg_types(),
        Some(first),
    );
    let ei = call_descr.get_extra_info().clone();
    // `opimpl_record_known_result_i_ir_v` records via
    // `_record_helper_varargs` and increments only RECORDED_OPS.
    ctx.trace_ctx.profiler().count_ops(
        OpCode::RecordKnownResult,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let _recorded = ctx
        .trace_ctx
        .record_op_with_descr(OpCode::RecordKnownResult, &allboxes, descr);
    ctx.trace_ctx.heapcache_invalidate_caches_varargs(
        OpCode::RecordKnownResult,
        Some(&ei),
        &allboxes,
    );
    Ok((DispatchOutcome::Continue, op.next_pc))
}

pub(crate) fn dispatch_record_known_result_i_ir_v<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let first = read_int_reg(code, op, 0, ctx)?;
    dispatch_record_known_result_ir(code, op, ctx, first)
}

pub(crate) fn dispatch_record_known_result_r_ir_v<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let first = read_ref_reg(code, op, 0, ctx)?;
    dispatch_record_known_result_ir(code, op, ctx, first)
}
