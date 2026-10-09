use majit_backend::{Backend, ExitValueSourceLayout};

use crate::jitexc::{ContinueRunningNormallyArgs, PortalResume};

thread_local! {
    /// Per-thread pool for the structured back-edge blackhole resume.
    /// `build_inline_call_only_bh_builder` wires ~100 string-keyed insns and
    /// their handlers — a fixed dispatch table independent of any trace — so
    /// rebuilding it on every guard failure is pure waste (the dominant
    /// per-deopt cost). Built once, lent via [`BackEdgeBhBuilder`], and
    /// re-pooled on drop, mirroring `call_jit.rs`'s `BH_BUILDER_RD`.
    /// The slot holds a `Box` so a lease moves one pointer rather than the
    /// builder itself (`warmspot.py` keeps one `BlackholeInterpBuilder` and
    /// `blackhole.py resume_in_blackhole` reaches it through
    /// `metainterp_sd.blackholeinterpbuilder`).
    static BACK_EDGE_BH_BUILDER: std::cell::RefCell<Option<Box<crate::blackhole::BlackholeInterpBuilder>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII lease of the thread-local back-edge blackhole builder. Takes the
/// pooled `Box` out for the duration of one resume and returns it on drop,
/// so every early `return` in the resume path re-pools it. If the slot is
/// empty — first use, or a re-entrant resume already holds it — a fresh
/// builder is constructed; the surplus is simply dropped when re-pooling.
struct BackEdgeBhBuilder(Option<Box<crate::blackhole::BlackholeInterpBuilder>>);

impl BackEdgeBhBuilder {
    fn lease() -> Self {
        let builder = BACK_EDGE_BH_BUILDER
            .with(|c| c.borrow_mut().take())
            .unwrap_or_else(|| Box::new(crate::blackhole::build_inline_call_only_bh_builder(&[])));
        Self(Some(builder))
    }
}

impl Drop for BackEdgeBhBuilder {
    fn drop(&mut self) {
        if let Some(builder) = self.0.take() {
            BACK_EDGE_BH_BUILDER.with(|c| *c.borrow_mut() = Some(builder));
        }
    }
}

impl std::ops::Deref for BackEdgeBhBuilder {
    type Target = crate::blackhole::BlackholeInterpBuilder;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("builder leased")
    }
}

impl std::ops::DerefMut for BackEdgeBhBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_deref_mut().expect("builder leased")
    }
}

/// Terminal state from a single-frame tracing-abort blackhole run.
///
/// The terminal interpreter is kept logically alive by moving out its three
/// register banks, matching the back-edge resume path below: a
/// ContinueRunningNormally handoff needs the full color-indexed bank, not only
/// the declared red subset carried by the exception.
pub struct SingleFrameBlackholeResult {
    pub outcome: crate::jitexc::JitException,
    pub registers_i: Vec<i64>,
    pub registers_r: Vec<i64>,
    pub registers_f: Vec<i64>,
    pub position: usize,
    pub last_opcode_position: usize,
    /// The drive's virtualizable, forwarded across every collection the run
    /// performed. The register banks above cannot answer this: a `-live-`
    /// marker clears each Ref the live set omits, so the frame red reads 0
    /// whenever the run stopped past that register's last use.
    pub virtualizable_ptr: i64,
    /// [`crate::blackhole::BlackholeInterpreter::abort_permanent_bail`] of the
    /// frame the run stopped in, carried for the same reason
    /// [`crate::blackhole::BlackholeTerminalImage`] carries it: `outcome`
    /// alone cannot tell a bail that stamped a resume coordinate from one that
    /// stopped mid-instruction.
    pub abort_permanent_bail: bool,
}

/// The blackhole's view of `metainterp_sd.jitdrivers_sd`, cast once and shared.
///
/// `bhimpl_jit_merge_point` indexes the prebuilt `jitdrivers_sd` directly, so
/// nothing about this table is per-resume; the rows only need re-casting
/// because the blackhole names the result kind, the portal runner and the
/// portal calldescr in its own types.  The cast is memoised on the static data
/// that owns the rows and invalidated by `finish_setup_descrs_for_jitdrivers`,
/// so a resume pays one `Arc` clone rather than one `BhCallDescr` per driver.
fn bh_jitdrivers_sd(
    metainterp_sd: &crate::pyjitpl::MetaInterpStaticData,
) -> &std::sync::Arc<[crate::blackhole::BhJitDriverSd]> {
    let table = metainterp_sd
        .bh_jitdrivers_sd
        .get_or_init(|| build_bh_jitdrivers_sd(metainterp_sd));
    debug_assert!(
        table.len() == metainterp_sd.jitdrivers_sd.len()
            && table
                .iter()
                .zip(&metainterp_sd.jitdrivers_sd)
                .all(|(bh, jd)| {
                    bh.portal_runner_ptr.map_or(0, |f| f as usize as i64) == jd.portal_runner_adr
                        && bh.result_type as u8 == bh_return_type(jd.result_type) as u8
                }),
        "bh_jitdrivers_sd outlived a `jitdrivers_sd` write that did not run \
         `finish_setup_descrs_for_jitdrivers`"
    );
    table
}

impl crate::pyjitpl::MetaInterpStaticData {
    /// `metainterp_sd.jitdrivers_sd` in the blackhole's row type, indexed by
    /// `jdindex` like the static data it is cast from.
    pub fn bh_jitdrivers_sd(&self) -> &std::sync::Arc<[crate::blackhole::BhJitDriverSd]> {
        bh_jitdrivers_sd(self)
    }
}

/// `warmspot.py:449` `jd.result_type` projected to the blackhole dispatch char.
fn bh_return_type(result_type: majit_ir::Type) -> crate::blackhole::BhReturnType {
    match result_type {
        majit_ir::Type::Int => crate::blackhole::BhReturnType::Int,
        majit_ir::Type::Ref => crate::blackhole::BhReturnType::Ref,
        majit_ir::Type::Float => crate::blackhole::BhReturnType::Float,
        majit_ir::Type::Void => crate::blackhole::BhReturnType::Void,
    }
}

fn build_bh_jitdrivers_sd(
    metainterp_sd: &crate::pyjitpl::MetaInterpStaticData,
) -> std::sync::Arc<[crate::blackhole::BhJitDriverSd]> {
    metainterp_sd
        .jitdrivers_sd
        .iter()
        .map(|jd| {
            let result_type = bh_return_type(jd.result_type);
            let portal_runner_ptr = (jd.portal_runner_adr != 0).then(|| unsafe {
                std::mem::transmute::<usize, extern "C" fn(i64, i64, i64, i64, i64) -> i64>(
                    jd.portal_runner_adr as usize,
                )
            });
            let mainjitcode_calldescr = jd
                .portal_calldescr
                .as_ref()
                .and_then(|descr| descr.as_call_descr())
                .map(majit_jitcode::jitcode::BhCallDescr::from_call_descr)
                .unwrap_or_default();
            crate::blackhole::BhJitDriverSd {
                result_type,
                portal_runner_ptr,
                mainjitcode_calldescr,
                handle_jitexc_from_bh: jd.handle_jitexc_from_bh,
            }
        })
        .collect()
}

/// Pick the `bh.virtualizable_info` pointer to seed at a guard-failure deopt, or
/// null to leave it unset. A non-null vinfo lets the blackhole run a mid-body
/// vable op (`getfield_vable_*`, or the `int_*_jump_if_ovf` overflow guard on
/// a `[int; virt]` state field). `blackhole.py` `clear_vable_token` is a no-op
/// while the token word is zero, and forces the frame when a compiled loop
/// left one, which is what a deopt of that loop has to do before it reads
/// the fields.
fn seed_deopt_vinfo_ptr(
    vinfo: Option<&std::sync::Arc<crate::virtualizable::VirtualizableInfo>>,
) -> *const crate::virtualizable::VirtualizableInfo {
    match vinfo {
        Some(info) => std::sync::Arc::as_ptr(info),
        None => std::ptr::null(),
    }
}

/// `MAJIT_BH_ENTRY_EXC`: report a `BH_LAST_EXC_VALUE` that survived into a
/// blackhole drive the caller did not describe as raising.
///
/// The clear-before / check-after protocol brackets every call-shaped opcode
/// inside a drive, so within one drive the channel cannot go stale.  What no
/// bracket covers is the window between phases: the value a residual raise left
/// behind is still in TLS when the next drive starts, and a drive entered with
/// `raising_exception == false` seeds `exception_last_value` from its own
/// `last_exc_value` argument instead.  The two can disagree.
///
/// Clearing the channel unconditionally on entry is NOT the fix — `call_jit.rs`
/// deliberately keeps a pending value across its decline path so the blackhole
/// can deliver it — so this reports the disagreement rather than repairing it,
/// and stays off unless armed.  What it is for: deciding whether the window is
/// ever actually observed, which no reading of the code settles.
fn bh_entry_stale_exc_report(last_exc_value: i64, raising_exception: bool, arm: &str) {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ARMED.get_or_init(|| std::env::var_os("MAJIT_BH_ENTRY_EXC").is_some()) {
        return;
    }
    let live = crate::blackhole::BH_LAST_EXC_VALUE.with(|cell| cell.get());
    if live != 0 && live != last_exc_value {
        eprintln!(
            "[bh-entry-exc] {arm}: BH_LAST_EXC_VALUE={live:#x} survives into a \
             drive carrying last_exc_value={last_exc_value:#x} \
             (raising_exception={raising_exception})"
        );
    }
}

/// Run one tracing MIFrame forward in the already-ported blackhole.
///
/// This is the single-frame counterpart of the structured back-edge template
/// at the call site below.  `copy_data_from_miframe` deliberately copies only
/// the typed register values and pc, so all interpreter-global / frame-global
/// state is supplied explicitly here.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn drive_single_frame_blackhole(
    miframe: &mut crate::pyjitpl::MIFrame,
    cpu: &dyn majit_backend::Backend,
    state_field_layout: crate::blackhole::StateFieldLayout,
    virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    mut virtualizable_ptr: i64,
    virtualizable_stack_base: usize,
    metainterp_sd: &crate::pyjitpl::MetaInterpStaticData,
    mut last_exc_value: i64,
    raising_exception: bool,
) -> SingleFrameBlackholeResult {
    bh_entry_stale_exc_report(last_exc_value, raising_exception, "single-frame");
    // The MIFrame is handed here from a force-time TLS latch.  Publish its Ref
    // values before leasing/building the interpreter because first use of the
    // pooled builder may allocate.  Option<i64> is not a contiguous root area,
    // so keep an index-parallel packed root vector and copy forwarding updates
    // back before copy_data_from_miframe reads the MIFrame.
    let mut ref_roots: Vec<(usize, i64)> = (0..miframe.ref_regs.len())
        .filter_map(|index| {
            miframe
                .ref_value_for_blackhole(index)
                .map(|value| (index, value))
        })
        .collect();
    let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    let mut packed_ref_roots: Vec<i64> = ref_roots.iter().map(|(_, value)| *value).collect();
    // `MetaInterp.last_exc_value` is not necessarily present in an MIFrame
    // Ref register (e.g. between `last_exception` and `last_exc_value/>r`).
    // Root it beside the bank for the full drive, matching the multi-frame
    // driver's explicit exception root.
    let exception_root = (last_exc_value != 0).then(|| {
        let index = packed_ref_roots.len();
        packed_ref_roots.push(last_exc_value);
        index
    });
    // The virtualizable is reachable from the Ref bank only while a `-live-`
    // marker still names its register. The marker hook clears every Ref the
    // live set omits, so a run that stops past the frame red's last use leaves
    // that slot at 0 and the bank can no longer answer where the frame moved.
    // Root the address beside the bank, the same way the exception value above
    // is rooted, so the identity survives independently of the liveness set.
    let virtualizable_root = (virtualizable_ptr != 0).then(|| {
        let index = packed_ref_roots.len();
        packed_ref_roots.push(virtualizable_ptr);
        index
    });
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(packed_ref_roots.as_mut_slice());
    }

    let mut builder = BackEdgeBhBuilder::lease();
    builder.set_cpu(cpu);
    builder.setup_cached_control_opcodes(
        metainterp_sd.op_live,
        metainterp_sd.op_catch_exception,
        metainterp_sd.op_rvmprof_code,
    );
    for ((index, value), forwarded) in ref_roots.iter_mut().zip(&packed_ref_roots) {
        *value = *forwarded;
        miframe.set_forwarded_ref_value(*index, *forwarded);
    }
    if let Some(index) = exception_root {
        last_exc_value = packed_ref_roots[index];
    }
    if let Some(index) = virtualizable_root {
        virtualizable_ptr = packed_ref_roots[index];
    }

    builder.setup_jitdrivers_sd(std::sync::Arc::clone(bh_jitdrivers_sd(metainterp_sd)));

    let mut bh = builder.acquire_interp();
    bh.copy_data_from_miframe(miframe);
    bh.state_field_layout = state_field_layout;
    bh.virtualizable_info = virtualizable_info;
    bh.virtualizable_ptr = virtualizable_ptr;
    bh.virtualizable_stack_base = virtualizable_stack_base;

    // The blackhole may allocate/collect while executing.  Its Ref bank is the
    // authoritative live register file and must be forwarded in place.
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(bh.registers_r.as_mut_slice());
    }
    let current_exc = if raising_exception {
        last_exc_value
    } else {
        bh.exception_last_value = last_exc_value;
        0
    };
    let outcome = match bh.resume_mainloop(current_exc) {
        Err(jit_exc) => jit_exc,
        Ok(propagated) => {
            debug_assert!(
                bh.nextblackholeinterp.is_some(),
                "single-frame bottom blackhole returned to a missing caller"
            );
            crate::jitexc::JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(
                propagated as usize,
            ))
        }
    };
    // Read the forwarded frame address out of the root area before dropping it.
    if let Some(index) = virtualizable_root {
        virtualizable_ptr = packed_ref_roots[index];
    }
    majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);

    let result = SingleFrameBlackholeResult {
        outcome,
        registers_i: std::mem::take(&mut bh.registers_i),
        registers_r: std::mem::take(&mut bh.registers_r),
        registers_f: std::mem::take(&mut bh.registers_f),
        position: bh.position,
        last_opcode_position: bh.last_opcode_position,
        virtualizable_ptr,
        abort_permanent_bail: bh.abort_permanent_bail,
    };
    builder.release_interp(bh);
    result
}

/// Terminal state from a multi-frame tracing-abort blackhole run.
///
/// `terminal` is the bottommost (portal-level) frame's register image, present
/// whenever its exception propagated out of `_run_forever`.  A
/// ContinueRunningNormally handoff needs it to materialize that frame's live
/// operand-stack slots, exactly as the single-frame path uses
/// [`SingleFrameBlackholeResult`].
pub struct MultiFrameBlackholeResult {
    pub outcome: crate::jitexc::JitException,
    pub terminal: Option<crate::blackhole::BlackholeTerminalImage>,
}

/// `pyjitpl.py run_blackhole_interp_to_cancel_tracing` input, captured at
/// the abort and cashed in one step later.
///
/// RPython runs the conversion inline in the abort handler because
/// `metainterp.framestack` and the interpreter state are both reachable there.
/// Pyre's abort handler (`merge_point`'s `TraceAction::Abort` arm) sees the
/// framestack and the live sym but not native `state`, and the blackhole needs
/// `JitState::state_field_layout` to know which registers the `state_field`
/// handlers read.  So the arm records both halves here and the
/// `jit_merge_point!` hook — the first point that holds `state` — runs
/// [`JitDriver::run_pending_abort_blackhole`].
pub struct PendingAbortBlackhole {
    /// The aborting walk's frames, outermost first, each already carrying its
    /// own resume `pc` (`blackhole.py _copy_data_from_miframe`).
    pub framestack: crate::pyjitpl::MIFrameStack,
    /// `JitState::collect_scalar_state_field_values` — int scalars followed by
    /// float scalars, read off the sym while it was still live.
    pub scalar_values: Vec<i64>,
    /// `JitState::collect_ref_scalar_state_field_values`.
    pub ref_scalar_values: Vec<i64>,
    /// The walk's `[.. ; virt]` array elements
    /// (`TraceCtx::collect_virtualizable_element_values`).  The walk mutates
    /// the array on the trace-ctx shadow, never on the live struct, so these
    /// have to be pushed into native `state` before the chain runs — the
    /// blackhole's `getarrayitem_vable_*` read the real object.  `None` when
    /// the state has no virtualizable array.
    pub virt_array_values: Option<Vec<i64>>,
    /// The trace's virtualizable identity (`initialize_virtualizable` writes
    /// it onto the ctx; `set_vable_ptr` only seeds `pending_vable_ptr`).
    /// `0` when the state has none.
    pub virtualizable_ptr: i64,
    /// `metainterp.last_exc_value` as `blackhole.py:1811-1814` reads it —
    /// snapshotted at the abort because the accounting that follows clears the
    /// tracing state it lives on.  `0` for a null exception.
    pub last_exc_value: i64,
    /// `SwitchToBlackhole.raising_exception` (`history.py`): true when the
    /// abort was raised at a point where `last_exc_value` is still *supposed to
    /// be raised*, so the conversion hands it to the chain instead of merely
    /// copying it into the first frame's `exception_last_value`.
    pub raising_exception: bool,
}

/// Run a tracing MIFrame chain through the structured multi-frame blackhole
/// conversion.  Every frame shares the portal-level virtualizable layout, but
/// retains its own jitcode, register banks, and blackhole interpreter.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn drive_multi_frame_blackhole(
    builder: &mut crate::blackhole::BlackholeInterpBuilder,
    framestack: &mut crate::pyjitpl::MIFrameStack,
    state_field_layout: crate::blackhole::StateFieldLayout,
    virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    virtualizable_ptr: i64,
    virtualizable_stack_base: usize,
    metainterp_sd: &crate::pyjitpl::MetaInterpStaticData,
    mut last_exc_value: i64,
    raising_exception: bool,
    per_frame: Option<&[(i64, usize)]>,
    on_enter_level: Option<&dyn Fn(i64)>,
    on_leave_level: Option<&dyn Fn(i64)>,
) -> MultiFrameBlackholeResult {
    bh_entry_stale_exc_report(last_exc_value, raising_exception, "multi-frame");
    let mut ref_locations = Vec::new();
    let mut packed_ref_roots = Vec::new();
    for (frame_index, frame) in framestack.frames.iter().enumerate() {
        for color in 0..frame.ref_regs.len() {
            if let Some(value) = frame.ref_value_for_blackhole(color) {
                ref_locations.push((frame_index, color));
                packed_ref_roots.push(value);
            }
        }
    }
    let exception_root = (last_exc_value != 0).then(|| {
        let index = packed_ref_roots.len();
        packed_ref_roots.push(last_exc_value);
        index
    });
    let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(packed_ref_roots.as_mut_slice());
    }

    builder.setup_cached_control_opcodes(
        metainterp_sd.op_live,
        metainterp_sd.op_catch_exception,
        metainterp_sd.op_rvmprof_code,
    );
    for ((frame_index, color), forwarded) in ref_locations
        .into_iter()
        .zip(packed_ref_roots.iter().copied())
    {
        framestack.frames[frame_index].set_forwarded_ref_value(color, forwarded);
    }
    if let Some(index) = exception_root {
        last_exc_value = packed_ref_roots[index];
    }

    builder.setup_jitdrivers_sd(std::sync::Arc::clone(bh_jitdrivers_sd(metainterp_sd)));

    let mut terminal = None;
    let outcome = crate::blackhole::convert_and_run_from_pyjitpl(
        builder,
        framestack,
        last_exc_value,
        raising_exception,
        Some(crate::blackhole::PyjitplBlackholeFrameConfig {
            state_field_layout,
            virtualizable_info,
            virtualizable_ptr,
            virtualizable_stack_base,
            per_frame,
            on_enter_level,
            on_leave_level,
        }),
        Some(&mut terminal),
    );
    majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
    MultiFrameBlackholeResult { outcome, terminal }
}

fn writeback_live_state_scalars_from_blackhole<S: crate::JitState>(
    state: &mut S,
    layout: &crate::blackhole::StateFieldLayout,
    bh: &crate::blackhole::BlackholeInterpreter,
    all_liveness: &[u8],
) {
    use majit_jitcode::liveness::LivenessIterator;

    let info = bh.get_current_position_info();
    if info + 3 > all_liveness.len() {
        return;
    }
    let length_i = all_liveness[info] as u32;
    let length_r = all_liveness[info + 1] as u32;
    let length_f = all_liveness[info + 2] as u32;
    let mut offset = info + 3;

    if length_i != 0 {
        let mut it = LivenessIterator::new(offset, length_i, all_liveness);
        for reg_idx in &mut it {
            let reg_idx = reg_idx as usize;
            let scalars_end = layout.int_scalar_base + layout.num_scalars;
            if (layout.int_scalar_base..scalars_end).contains(&reg_idx)
                && let Some(value) = bh.registers_i.get(reg_idx).copied()
            {
                state.writeback_live_scalar_state_field(reg_idx - layout.int_scalar_base, value);
            }
        }
        offset = it.offset;
    }

    if length_r != 0 {
        let mut it = LivenessIterator::new(offset, length_r, all_liveness);
        for reg_idx in &mut it {
            let reg_idx = reg_idx as usize;
            let scalars_end = layout.ref_scalar_base + layout.num_ref_scalars;
            if (layout.ref_scalar_base..scalars_end).contains(&reg_idx)
                && let Some(value) = bh.registers_r.get(reg_idx).copied()
            {
                state
                    .writeback_live_ref_scalar_state_field(reg_idx - layout.ref_scalar_base, value);
            }
        }
        offset = it.offset;
    }

    if length_f != 0 {
        let mut it = LivenessIterator::new(offset, length_f, all_liveness);
        for reg_idx in &mut it {
            let reg_idx = reg_idx as usize;
            let scalars_end = layout.float_scalar_base + layout.num_float_scalars;
            if (layout.float_scalar_base..scalars_end).contains(&reg_idx)
                && let Some(value) = bh.registers_f.get(reg_idx).copied()
            {
                state.writeback_live_float_scalar_state_field(
                    reg_idx - layout.float_scalar_base,
                    value,
                );
            }
        }
    }
}

/// RAII guard that suspends re-entrant trace continuation on the owning
/// `TraceCtx` for its lifetime, restoring the previous state on drop
/// (nesting-safe).
#[must_use = "the guard suspends trace continuation only while held; a bare \
              `TraceContinuationSuspendGuard::enter(ctx);` drops it immediately"]
pub struct TraceContinuationSuspendGuard<'a> {
    cell: &'a std::cell::Cell<bool>,
    prev: bool,
}

impl<'a> TraceContinuationSuspendGuard<'a> {
    pub fn enter(ctx: &'a crate::trace_ctx::TraceCtx) -> Self {
        Self {
            prev: ctx.trace_continuation_suspended.replace(true),
            cell: &ctx.trace_continuation_suspended,
        }
    }
}

impl Drop for TraceContinuationSuspendGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.prev);
    }
}

/// Diagnostic env gates read once and cached — these are checked on the hot
/// back-edge / guard-failure paths that run every loop iteration, so re-reading
/// the environment per call would add a syscall to each iteration.
///
/// `MAJIT_SPDIAG`: trace the stack-pointer bookkeeping. Public for the same
/// reason [`no_bridge_enabled`] is — a frontend that owns state the diagnostic
/// is about has to report it from its own side, and a variable that only the
/// metainterp honours reads as broken when a frontend's half stays silent.
#[inline]
pub fn spdiag_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_SPDIAG").is_some())
}
/// `MAJIT_NO_BRIDGE`: suppress bridge recording so every guard failure resumes
/// through the blackhole.  Public because a frontend that owns its own
/// guard-failure entry point has to honour it there too — gating only the
/// jitdriver-internal paths leaves the variable set but inert, which reads as
/// "bridges are off" while they keep recording.
pub fn no_bridge_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| crate::jit_env::env_var_os("MAJIT_NO_BRIDGE").is_some())
}
/// `MAJIT_SKIP_BRIDGES=a,b,...` (diagnostic): decline exactly the listed
/// `MAJIT_MAX_BRIDGES` sequence numbers and take every other one.  Where the
/// fuel limit answers "the first N are sound", this answers "is number N the
/// one that corrupts, or is the corruption cumulative" — a distinction the
/// limit cannot draw, because declining N also declines everything after it.
/// Off by default; an unparsable entry is ignored rather than reported, since
/// the variable exists only for a bisect the operator is reading anyway.
fn bridge_fuel_skip_set() -> &'static [u64] {
    static SET: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    SET.get_or_init(|| {
        crate::jit_env::env_var("MAJIT_SKIP_BRIDGES")
            .map(|v| v.split(',').filter_map(|p| p.trim().parse().ok()).collect())
            .unwrap_or_default()
    })
}

fn bridge_fuel_skipped(n: u64) -> bool {
    bridge_fuel_skip_set().contains(&n)
}
/// `MAJIT_MAX_BRIDGES=N` (diagnostic): allow the first N bridge compilations
/// and behave as `MAJIT_NO_BRIDGE` from then on.  Bisecting N names the bridge
/// whose compilation first produces a wrong value, at seconds per run rather
/// than a rebuild per arm.  Consumes fuel only when the rest of `should_bridge`
/// already held, so the count is bridges actually taken — place it last in the
/// `&&` chain.  `MAJIT_BRIDGE_FUEL_LOG` reports each one taken.
///
/// `MAJIT_SKIP_BRIDGES` names positions in this same sequence, so either gate
/// alone starts the count: numbering only when a limit is set would leave the
/// skip list naming positions nothing ever allocates.
///
/// Public for the same reason `no_bridge_enabled` is: a frontend with its own
/// guard-failure entry point spends fuel there too, and a counter that only
/// half the bridges draw from makes `MAJIT_MAX_BRIDGES=0` disagree with
/// `MAJIT_NO_BRIDGE=1` — the arm that reads as "no bridges" still compiles
/// every bridge that entry point reaches.
pub fn bridge_fuel_take() -> bool {
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let limit = *LIMIT
        .get_or_init(|| crate::jit_env::env_var("MAJIT_MAX_BRIDGES").and_then(|v| v.parse().ok()));
    // The sequence number belongs to the pair, not to the limit: the skip list
    // names positions in this same count, so a run that configures only the
    // skip list still has to number the bridges it takes. Numbering only when a
    // limit is set would leave the skip list naming positions nothing allocates.
    if limit.is_none() && bridge_fuel_skip_set().is_empty() {
        return true;
    }
    static USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = USED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if limit.is_some_and(|limit| n >= limit) {
        return false;
    }
    if bridge_fuel_skipped(n) {
        if std::env::var_os("MAJIT_BRIDGE_FUEL_LOG").is_some() {
            eprintln!("@@@FUEL bridge #{n} SKIPPED");
        }
        return false;
    }
    if std::env::var_os("MAJIT_BRIDGE_FUEL_LOG").is_some() {
        eprintln!("@@@FUEL bridge #{n}");
    }
    true
}
/// `MAJIT_NO_GUARD_RESUME_BRIDGE`: decline every bridge entry taken directly
/// from a guard failure. A decline means the blackhole resumes with no
/// bridge.  Narrower than `MAJIT_NO_BRIDGE` only in which entry it declines:
/// this flag returns before `start_bridge_tracing` inside
/// `bridge_from_guard_resume_position`, and that failure records nothing.
/// Costly — the blackhole arm reaches the merge point by interpreting — so
/// it is a bisection tool, not a policy.  Off by default.
fn no_guard_resume_bridge_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_NO_GUARD_RESUME_BRIDGE").is_some())
}
/// Restores the refusal of a guard-resume bridge for a guard whose deferred
/// writes span more than one virtual.
///
/// Off by default, because upstream has no such refusal —
/// `compile.py ResumeGuardDescr.handle_fail` bridges whenever `must_compile`
/// fires. The cut decides a large share of every bridge opportunity, so the
/// two answers have to stay comparable inside ONE binary; a second build is
/// not a control for that question.
fn guard_resume_refuse_pending_fields() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_GUARD_RESUME_REFUSE_PENDING_FIELDS").is_some())
}
fn guardlog_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_GUARDLOG").is_some())
}
fn failvals_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_FAILVALS").is_some())
}
#[inline]
fn portal_rca_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_PORTAL_RCA").is_some())
}

// ── warm-entry stage probe ───────────────────────────────────────────────
//
// A measurement arm, never a shipping one, and behind a cargo feature for that
// reason: a default build has neither the load below nor the loops it feeds.
// Inside the feature the arms are selected at RUN time, because what one stage
// costs against another is a difference that has to be taken inside ONE binary
// — two `cargo build` invocations admit compile drift and stale binaries, and
// neither is visible in the numbers they produce.

/// Extra passes of one stage of a warm compiled entry.
///
/// A compiled entry that finds its artifact already there spends ~100 ns
/// arriving, and four fifths of that is inside [`JitDriver::back_edge_internal`]
/// below. No clock can be put inside it: `Instant::now()` costs a fifth of the
/// budget on the box these were taken on, so five of them would report the
/// clock. Each stage is timed by REPETITION instead — the same entry with every
/// count at zero, then again with one count at `k`, and the difference over `k`
/// is that stage. Everything the two arms share cancels out of it.
///
/// Only stages that are IDEMPOTENT under repetition can be read this way, and
/// the warm entry has one that is emphatically not: **the call**.
/// `execute_assembler_at_dispatch_key` builds the frame, enters the compiled
/// trace and decodes the deadframe it returns (`get_latest_descr_arc` and
/// `decode_exit_slots` are inside it, not after it). It RUNS THE PROGRAM: a
/// second pass would execute the trace again from state the first one already
/// advanced, and would leave through a different exit. There is no honest
/// repeatable form of it, so none is offered — the call is what the residual
/// is, together with the handful of one-shot moves named on the fields below.
#[cfg(feature = "__back-edge-stage-probe")]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct BackEdgeStageRepeats {
    /// Extra gate consultations: `entry_procedure_token` (only for a caller
    /// that did not arrive carrying one, which is the shape the gate itself
    /// takes), `get_compiled_meta(..).cloned()`, [`JitDriver::driver_descriptor_for`]
    /// and `is_compatible`. All four answer the same thing every time they are
    /// asked on one entry, which is what makes them repeatable.
    ///
    /// NOT in this stage, and left in the residual: `resolve_cell_key`, which a
    /// carrying caller skips altogether, and
    /// `take_single_pass_label_entry_dispatch_key_for_back_edge`, which TAKES
    /// the slot — a second call finds it empty, answers `None`, and prices the
    /// other branch rather than this one.
    pub gate: u16,
    /// Extra refills of the entry-argument buffers: `sync_before`,
    /// `extract_live_values_into`, `live_values_match_descriptor`,
    /// `extend_compiled_live_values_into`, and the buffer clears.
    ///
    /// The clears belong to the stage rather than to the machinery:
    /// [`JitDriver::take_entry_scratch`] does exactly them once per entry, and
    /// a refill that skipped them would append to the previous pass instead of
    /// repeating it. What is NOT in the stage is that function's own
    /// `mem::take` — a second take hands back the defaulted struct.
    ///
    /// Armed only for an entry that extracted its arguments from state. One
    /// that arrived with `direct_live_values` ran none of this.
    pub marshal_in: u16,
    /// Extra `restore_values` + `sync_after` on the values a completed run left
    /// behind.
    ///
    /// Armed only on the arms that reach them — a FINISH or a normal back-edge
    /// JUMP. A guard failure goes to `handle_fail` and reaches neither.
    /// `restore_values` is skipped for a FINISH exactly as the shipping arm
    /// skips it: those exit slots are the portal's return value and not the
    /// loop-carried state.
    ///
    /// NOT in this stage: `entry_scratch_out`, which moves the buffers back
    /// into the driver, and the `drop(result)` on the guard-failure path. Both
    /// run once by construction.
    pub marshal_out: u16,
    /// Extra passes of the same loop with NO stage in it: the counter, and the
    /// one optimization barrier every stage's loop also carries. Subtracting it
    /// is what leaves a stage's own cost rather than its cost plus the
    /// machinery that made it repeat.
    pub barrier: u16,
}

#[cfg(feature = "__back-edge-stage-probe")]
impl BackEdgeStageRepeats {
    fn pack(self) -> u64 {
        (self.gate as u64)
            | (self.marshal_in as u64) << 16
            | (self.marshal_out as u64) << 32
            | (self.barrier as u64) << 48
    }

    fn unpack(packed: u64) -> Self {
        Self {
            gate: packed as u16,
            marshal_in: (packed >> 16) as u16,
            marshal_out: (packed >> 32) as u16,
            barrier: (packed >> 48) as u16,
        }
    }
}

/// The counts the next warm entry runs with, on every thread.
///
/// Process-wide rather than thread-local, for the reason the jitframe pool's
/// arm is (`set_jitframe_pool`): what it selects is a strategy, not a state, and
/// a harness flipping arms between two timed batches wants the flip to hold for
/// whichever thread the next entry runs on.
///
/// Four counts packed into one word so the entry path takes exactly ONE relaxed
/// load, which every arm pays and which therefore cancels out of the difference
/// between two of them. There is no unseeded state and no environment gate to
/// seed one from: all-zero is the shipping shape, and
/// [`set_back_edge_stage_repeats`] is the only way out of it.
#[cfg(feature = "__back-edge-stage-probe")]
static BACK_EDGE_STAGE_REPEATS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A stage of the warm compiled entry that the probe can amplify.
///
/// The variants are the field order of [`BackEdgeStageRepeats`], and each one's
/// discriminant is its slot in [`BACK_EDGE_STAGE_PASSES`]. `Barrier` is the arm
/// the other three are differenced against.
#[cfg(feature = "__back-edge-stage-probe")]
#[derive(Clone, Copy)]
#[repr(usize)]
enum BackEdgeStage {
    Gate = 0,
    MarshalIn = 1,
    MarshalOut = 2,
    Barrier = 3,
}

#[cfg(feature = "__back-edge-stage-probe")]
impl BackEdgeStage {
    const COUNT: usize = 4;
}

/// Amplified passes actually executed, one slot per [`BackEdgeStage`].
#[cfg(feature = "__back-edge-stage-probe")]
static BACK_EDGE_STAGE_PASSES: [std::sync::atomic::AtomicU64; BackEdgeStage::COUNT] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Set the repeat counts for subsequent warm entries, answering what they were.
///
/// A setter and not a wrapper around one entry: the entry's whole cost is read
/// through the frontend's own door, so the stages have to be read through it
/// too, or the parts and the whole are about different doors.
#[cfg(feature = "__back-edge-stage-probe")]
pub fn set_back_edge_stage_repeats(repeats: BackEdgeStageRepeats) -> BackEdgeStageRepeats {
    BackEdgeStageRepeats::unpack(
        BACK_EDGE_STAGE_REPEATS.swap(repeats.pack(), std::sync::atomic::Ordering::Relaxed),
    )
}

/// Amplified passes since the process started, as
/// `[gate, marshal_in, marshal_out, barrier]` — the field order of
/// [`BackEdgeStageRepeats`].
///
/// The witness that an arm was REACHED. Two arms that ran the same code differ
/// by nothing but the box, and a stage figure taken off such a pair describes
/// the box; these counts are what tells the two cases apart without a clock.
#[cfg(feature = "__back-edge-stage-probe")]
pub fn back_edge_stage_passes() -> [u64; BackEdgeStage::COUNT] {
    let read = |stage: BackEdgeStage| {
        BACK_EDGE_STAGE_PASSES[stage as usize].load(std::sync::atomic::Ordering::Relaxed)
    };
    [
        read(BackEdgeStage::Gate),
        read(BackEdgeStage::MarshalIn),
        read(BackEdgeStage::MarshalOut),
        read(BackEdgeStage::Barrier),
    ]
}

/// One relaxed load per warm entry. Both arms pay it.
#[cfg(feature = "__back-edge-stage-probe")]
#[inline]
fn back_edge_stage_repeats() -> BackEdgeStageRepeats {
    BackEdgeStageRepeats::unpack(BACK_EDGE_STAGE_REPEATS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Tally one call's worth of amplified passes for a stage.
///
/// Once per call rather than once per pass, and the barrier arm pays it too, so
/// the read-modify-write cancels out of every stage's difference instead of
/// being amplified `k`-fold into it.
#[cfg(feature = "__back-edge-stage-probe")]
#[inline]
fn count_back_edge_stage_passes(stage: BackEdgeStage, repeats: u16) {
    if repeats != 0 {
        BACK_EDGE_STAGE_PASSES[stage as usize]
            .fetch_add(repeats as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

fn format_rca_live_values(labels: Option<&[String]>, values: &[Value]) -> String {
    let mut out = String::new();
    for (idx, value) in values.iter().enumerate() {
        let label = labels
            .and_then(|labels| labels.get(idx).map(String::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("virtualizable_payload[{}]", idx));
        let raw = match value {
            Value::Int(v) => *v,
            Value::Ref(r) => r.as_usize() as i64,
            Value::Float(v) => v.to_bits() as i64,
            Value::Void => 0,
        };
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("  live[{idx:02}] {label}: {:?} raw={raw}\n", value),
        );
    }
    out
}

use crate::TraceAction;
use crate::blackhole::ExceptionState;
use crate::jit_state::JitState;
use crate::pyjitpl::{
    AbortReason, BackEdgeAction, CompiledExitLayout, DetailedDriverRunOutcome, InlineDecision,
    JitStats, MetaInterp,
};
use crate::resume::ResumeLayoutSummary;
use crate::virtualizable::VirtualizableInfo;
use crate::warmstate::{FunctionEntryStep, HotResult, JcFlags};
use majit_gc::GcAllocator;
use majit_ir::OpRef;
use majit_ir::descr::DescrRef;
use majit_ir::{Const, GreenKey, GreenType, JitDriverVar, Type, Value, VarKind, green_type_to_ir};

/// Bucket the variables of a given kind by IR `Type`, returning
/// `(num_int, num_ref, num_float)`. `Type::Void` is skipped per
/// upstream — `bhimpl_jit_merge_point` only encodes `i / R / F` slots.
fn kind_counts(vars: &[JitDriverVar], kind: VarKind) -> (usize, usize, usize) {
    let mut counts = (0usize, 0usize, 0usize);
    for v in vars.iter().filter(|v| v.kind == kind) {
        match v.tp {
            Type::Int => counts.0 += 1,
            Type::Ref => counts.1 += 1,
            Type::Float => counts.2 += 1,
            Type::Void => {}
        }
    }
    counts
}

/// Slice (audit Issue #5) — production dispatch JitCode payload
/// schema cross-check.  Called from `register_dispatch_jitcode` just
/// before the JitCode is stashed as the driver's singleton.
///
/// Walks the body bytes from offset 0 looking for the first
/// `BC_JIT_MERGE_POINT` / `BC_JIT_MERGE_POINT_C` byte; when found,
/// reads the 1-byte jdindex + six [I, R, F, I, R, F] length bytes
/// and compares them to `green_kind_counts` / `red_kind_counts`.
///
/// The walk is structural (peek for the merge-point byte) rather
/// than a full bytecode iterator — false positives are tolerated
/// because pyre's lowerer always emits the merge-point near the
/// start of the dispatch body, and the count check would only
/// false-pass when the body coincidentally starts with a
/// well-formed-looking 7-byte sequence at the same offset.  The
/// check fires in release builds — `warmspot.py
/// make_args_specification` runs at translation time and asserts
/// the marker op's args match the driver schema; the runtime
/// version stays active in production for the same reason
/// (codegen / `declare_schema` drift would otherwise silently
/// install a misshaped dispatch JitCode).
///
/// Translation-time assertion surface — `pyjitpl.py verify_green_args
/// verify_green_args` + `warmspot.py make_args_specification`
/// fail loudly when the dispatch JitCode's `BC_JIT_MERGE_POINT`
/// payload disagrees with the driver schema.  Pyre keeps the same
/// fail-loud contract here: every malformed shape (no descriptor,
/// missing merge offset, truncated payload) is a codegen /
/// `declare_schema` lifecycle bug, not a runtime data condition.
fn validate_dispatch_jitcode_payload<S>(driver: &JitDriver<S>, jc: &crate::jitcode::JitCode)
where
    S: crate::JitState,
{
    let descriptor = driver.descriptor.as_ref().unwrap_or_else(|| {
        panic!(
            "register_dispatch_jitcode: JitDriver has no descriptor — \
             `declare_schema` / `declare_schema_typed` must run before \
             registration (warmspot.py:660-666 make_args_specification \
             parity)",
        )
    });
    use majit_jitcode::codewriter::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};
    let body = &jc.code;
    // Read the opcode byte directly from the captured offset.  Both routes
    // into a body record it at the position they reserve for the opcode:
    // `JitCodeBuilder::jit_merge_point` on the proc-macro route, and
    // `Assembler::encode_op`'s `JitMergePoint` arm on the LLBC one.  No
    // byte-stream scan — mirrors RPython `blackhole.py handler` argcode-based
    // decode (operand bytes that happen to equal `BC_JIT_MERGE_POINT(_C)`
    // cannot trigger a false-positive payload validation).  Pre-Merge-
    // Point body-local lets and other lowerer-emitted prefix opcodes
    // (`bind_pre_merge_point_stmts` in `jitcode_lower/api.rs`) are
    // therefore irrelevant to the validator's correctness.
    let merge_offset = jc.exec.jit_merge_point_offset.unwrap_or_else(|| {
        panic!(
            "register_dispatch_jitcode: dispatch JitCode `{}` has no \
             jit_merge_point_offset — the body was assembled without a \
             `jit_merge_point`, or it came through a route that does not \
             record the offset (`JitCodeBuilder::jit_merge_point` on the \
             proc-macro route, `Assembler::encode_op` then \
             `JitCode::from_canonical` on the LLBC one)",
            jc.name(),
        )
    });
    let opcode = body
        .get(merge_offset)
        .copied()
        .expect("jit_merge_point_offset out of bounds — JitCodeBuilder invariant violated");
    assert!(
        opcode == BC_JIT_MERGE_POINT || opcode == BC_JIT_MERGE_POINT_C,
        "jit_merge_point_offset {merge_offset} does not point at \
         BC_JIT_MERGE_POINT(_C); JitCodeBuilder::jit_merge_point invariant",
    );
    let payload_start = merge_offset + 1;
    let body_len = body.len();
    assert!(
        payload_start + 7 <= body_len,
        "register_dispatch_jitcode: dispatch JitCode `{}` body has \
         {body_len} bytes but BC_JIT_MERGE_POINT(_C) at offset \
         {merge_offset} requires at least 8 (jdindex + 6 list-len bytes). \
         pyjitpl.py:1530 verify_green_args parity surface — payload \
         truncation indicates a codegen lifecycle bug.",
        jc.name(),
    );
    let _jdindex = body[payload_start];
    let mut q = payload_start + 1;
    let (gi, gr, gf) = descriptor.green_kind_counts();
    let (ri, rr, rf) = descriptor.red_kind_counts();
    let (ei, er, ef) = driver.merge_point_identity_reds;
    // Greens are the portal-runner arguments. Reds on the marker are
    // those plus the identity/cell registers `resolve_reds` lists
    // (`identity_slot_registers` / `__flat_array_identity_slots`).
    let expected = [gi, gr, gf, ri + ei, rr + er, rf + ef];
    for (slot, expect) in expected.iter().enumerate() {
        assert!(
            q < body_len,
            "register_dispatch_jitcode: dispatch JitCode `{}` body \
             truncated at slot {slot} — header byte beyond body length \
             {body_len} (codegen / declare_schema mismatch).",
            jc.name(),
        );
        let count = body[q] as usize;
        assert_eq!(
            count, *expect,
            "register_dispatch_jitcode: BC_JIT_MERGE_POINT payload \
             slot {slot} count {count} disagrees with expected {expect} \
             (greens=({gi}, {gr}, {gf}) reds=({ri}, {rr}, {rf}) \
             identity=({ei}, {er}, {ef})). \
             Codegen / declare_schema mismatch.",
        );
        q += 1 + count;
        assert!(
            q <= body_len,
            "register_dispatch_jitcode: dispatch JitCode `{}` body \
             truncated reading slot {slot} register bytes — payload \
             tail at offset {q} exceeds body length {body_len}.",
            jc.name(),
        );
    }
}

/// The seed's entry for a dispatch JitCode that already names its own index.
///
/// Split out so the two numbering authorities read as the two cases they are:
/// this one checks that the seed really is the table the dispatch came from,
/// and returns its entry; the caller's copy is dropped.
fn adopt_seeded_dispatch(
    registry: &[std::sync::Arc<crate::jitcode::JitCode>],
    index: usize,
    dispatch: &crate::jitcode::JitCode,
) -> std::sync::Arc<crate::jitcode::JitCode> {
    let seeded = registry.get(index).unwrap_or_else(|| {
        panic!(
            "dispatch jitcode {:?} names index {index}, which the registry seed \
             does not reach — it holds {} entries. A build-time table that \
             numbers the portal has to hand it over through \
             `RuntimeDescrTable::jitcodes()`, or the portal is registered \
             against a list that is not the one its `j` operands index",
            dispatch.name(),
            registry.len(),
        )
    });
    assert_eq!(
        seeded.name(),
        dispatch.name(),
        "dispatch jitcode {:?} names index {index}, which the seed fills with \
         {:?} — two graphs claim one slot, so the `j` operands written against \
         it resolve to whichever of them the registry happens to hold",
        dispatch.name(),
        seeded.name(),
    );
    // A build-time shell resolves its `j` operands through the shared pool and
    // carries no per-jitcode descrs, so there is nothing here for the walk to
    // reach. A caller that has some is one whose callees were emitted at run
    // time, and adopting would drop them unnumbered.
    assert!(
        dispatch.exec.descrs.is_empty(),
        "dispatch jitcode {:?} is seeded at index {index} but carries {} \
         run-time descrs; adopting the seed's entry would leave them out of \
         the registry",
        dispatch.name(),
        dispatch.exec.descrs.len(),
    );
    assert_eq!(
        seeded.exec.jit_merge_point_offset,
        dispatch.exec.jit_merge_point_offset,
        "dispatch jitcode {:?} is seeded at index {index} but its merge-point \
         offset {:?} disagrees with the seed's {:?}; adopting the seed's entry \
         would install a different body",
        dispatch.name(),
        dispatch.exec.jit_merge_point_offset,
        seeded.exec.jit_merge_point_offset,
    );
    std::sync::Arc::clone(seeded)
}

/// Flatten a dispatch JitCode and everything it can inline-call into the one
/// flat `all_jitcodes` registry, returning it with the dispatch JitCode's
/// `Arc`.
///
/// `codewriter.py make_jitcodes` numbers a single list — `jitcode.index =
/// index` as each is appended — and `resume.py:1338-1340 jitcode =
/// jitcodes[jitcode_pos]` reads that same list back. So the registry has one
/// property to keep: `registry[jitcode.index()]` is that jitcode.
///
/// `seed` is the part of the list a host already numbered at build time, which
/// its own `j` operands are written against. Those indices are not this
/// function's to assign — `JitCode::set_index` is set-once and asserts on a
/// second value, so overwriting one is not even a silent renumbering, it is a
/// panic. Numbering therefore CONTINUES above the seed rather than restarting,
/// which is the same thing upstream does by appending to `all_jitcodes` in one
/// pass.
///
/// The walk starts at the dispatch JitCode, not at the seed: every seeded entry
/// is already present, so its callees are too. It is depth-agnostic, so a
/// nested inline helper stays correctly indexed without a parent-relative walk.
///
/// A dispatch JitCode the seed *already holds* is adopted at the index it
/// names rather than appended. `codewriter.py make_jitcodes` numbers the
/// portal graph in the same pass as every other graph, and `warmspot.py` reads
/// it back as `metainterp_sd.jitcodes[portal_jd.index]` — upstream's portal is
/// a seeded entry. A host that lowers its whole interpreter ahead of time
/// hands over that shape; only one that mints the dispatch body at run time
/// (`JitCodeBuilder`, the `#[jit_interp]` route) has an unnumbered portal to
/// append.
/// The registry for a dispatch jitcode the seed already numbers, or `None` when
/// it carries no index and has to be appended instead.
///
/// Split out because it reads the dispatch and never takes it: a consumer whose
/// table is a build artefact holds every body behind an `Arc` and has no owned
/// copy to give, and the arm it needs is exactly this one.
fn build_seeded_jitcode_registry(
    seed: &[std::sync::Arc<crate::jitcode::JitCode>],
    dispatch: &crate::jitcode::JitCode,
) -> Option<(
    Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    std::sync::Arc<crate::jitcode::JitCode>,
)> {
    let index = dispatch.try_index()?;
    let registry = seed.to_vec();
    // The seed's own `Arc`, never the caller's copy: `codewriter.py:80
    // all_jitcodes[jitcode.index] is jitcode` is the identity every `j` operand
    // at `index` resolves through, and the driver's `mainjitcode` has to be that
    // same object. Nothing is appended, so the registry is the seed — which the
    // build-time table already numbered end to end.
    let seeded = adopt_seeded_dispatch(&registry, index, dispatch);
    Some((registry, seeded))
}

fn build_jitcode_registry(
    seed: &[std::sync::Arc<crate::jitcode::JitCode>],
    dispatch: crate::jitcode::JitCode,
) -> (
    Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    std::sync::Arc<crate::jitcode::JitCode>,
) {
    if let Some(seeded) = build_seeded_jitcode_registry(seed, &dispatch) {
        return seeded;
    }
    let mut registry = seed.to_vec();
    let dispatch_position = registry.len();
    dispatch.set_index(dispatch_position);
    let dispatch_arc = std::sync::Arc::new(dispatch);
    registry.push(dispatch_arc.clone());
    let mut cursor = dispatch_position;
    while cursor < registry.len() {
        let current = registry[cursor].clone();
        cursor += 1;
        for descr in &current.exec.descrs {
            if let Some(sub) = descr.as_jitcode() {
                if registry.iter().any(|j| std::sync::Arc::ptr_eq(j, sub)) {
                    continue;
                }
                // A jitcode the seed does not hold has to be one built at run
                // time, which carries no index yet. One that already names an
                // index came from a build-time table this registry never saw,
                // so the seed is incomplete — and renumbering it is not a
                // recoverable error: it either aborts in `set_index`, or, when
                // the walk order happens to reproduce the build-time position,
                // silently yields a registry whose slots are not the objects a
                // callee's nested frame indices resolve against. Fail at the
                // point that names the cause instead.
                assert!(
                    sub.try_index().is_none(),
                    "jitcode {:?} already names index {:?} but the registry seed does not hold it; \
                     a `RuntimeDescrTable` that hands out `JitCode` entries must implement \
                     `jitcodes()` so the seed covers them",
                    sub.name(),
                    sub.try_index(),
                );
                let idx = registry.len();
                sub.set_index(idx);
                registry.push(sub.clone());
            }
        }
    }
    (registry, dispatch_arc)
}

/// The width of a compiled loop's entry contract, and where the
/// virtualizable sits inside it, for a driver whose entry is NOT a red list.
///
/// `compile.py:431` truncates `loop.inputargs` to `jitdriver_sd.num_red_args`
/// and `compile.py:429` reads the virtualizable out of that prefix by
/// `index_of_virtualizable`. Both spellings work upstream because the compiled
/// entry is invoked with exactly the jitdriver's reds (`warmstate.py execute_assembler
/// execute_assembler`) and the virtualizable is one of them
/// (`warmspot.py jd.index_of_virtualizable = jitdriver.reds.index(vname)`).
///
/// A driver that declares its runtime state as flat fields presents a
/// different entry: one slot per declared field, in the order the state's live
/// values are emitted, with the virtualizable occupying a single slot among
/// them rather than being one of N reds. Its logical red list still exists —
/// it is the merge-point payload schema, in which the whole state is one red —
/// so the red count and the entry width are counts of different things and
/// neither can be derived from the other. Such a driver states both numbers
/// here instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlatEntryContract {
    /// Number of leading inputargs that form the entry contract — the
    /// `compile.py:431 i` the loop's inputargs are truncated to.
    pub len: usize,
    /// Index, within that prefix, of the slot holding the virtualizable
    /// pointer (`compile.py:429 inputargs[index_of_virtualizable]`).
    pub index_of_virtualizable: usize,
}

/// Descriptor for a JitDriver's variable layout.
///
/// Mirrors RPython's `JitDriver(greens=[...], reds=[...])`:
/// - `greens` are compile-time constants identifying the loop header
/// - `reds` are runtime values carried as InputArgs
///
/// The interpreter declares this once per JitDriver and passes it to
/// MetaInterp for structured green/red handling.
#[derive(Clone, Debug)]
pub struct JitDriverStaticData {
    /// warmspot.py / pyjitpl.py parity: registered driver slot index.
    ///
    /// RPython stores this on `jitdriver_sd.index` and threads it
    /// through portal-frame bookkeeping plus `JitCellToken` ownership.
    /// It stays `None` until `register_jitdriver_sd` assigns the slot.
    pub index: Option<usize>,
    /// `rlib/jit.py JitDriver.name`. `compile.py compile_trace` writes
    /// this as `jd_name` on `MARK_START_TRACE`. Default is the upstream
    /// constructor default `'jitdriver'`.
    pub name: String,
    /// All variables in declaration order.
    pub vars: Vec<JitDriverVar>,
    /// Optional name of the virtualizable red variable.
    pub virtualizable: Option<String>,
    /// Set when this driver's compiled entry contract is a flat state-field
    /// prefix rather than the red list. See [`FlatEntryContract`]; `None`
    /// keeps the upstream red-list spelling.
    pub flat_entry: Option<FlatEntryContract>,
    /// warmspot.py:449 jd.result_type — portal function return type.
    /// Determined once at driver setup from the portal's return signature.
    pub result_type: Type,
    /// rpython/rlib/jit.py JitDriver(is_recursive=True) parity.
    /// Mirrors RPython's `jitcode.jitdriver_sd.jitdriver.is_recursive`
    /// access — see pyjitpl.py `is_main_jitcode`.  Pyre stores
    /// the flag directly on the static data so MetaInterp can decide
    /// without touching the runtime `JitDriver` object.
    pub is_recursive: bool,
    /// warmspot.py:921 `jd.mainjitcode = pyjitpl_class(jd, mainjitcode)`.
    ///
    /// The portal jitcode for this driver — the entry point of the
    /// recursive function being JITed.  Read by
    /// `MetaInterp.do_recursive_call` (pyjitpl.py) to set
    /// `portal_code.calldescr` for the residual CALL_ASSEMBLER op.
    pub mainjitcode: Option<std::sync::Arc<crate::jitcode::JitCode>>,
    /// Deferred source for [`Self::mainjitcode`]. `call.py grab_initial_jitcodes`
    /// binds `jd.mainjitcode` during translation; pyre keeps that binding at
    /// driver build and runs the decode on the first `mainjitcode_of` read.
    /// `None` when the eager field is the only source.
    ///
    /// The loader must return one process-wide `Arc`: this struct is `Clone`,
    /// and each clone has its own [`Self::mainjitcode_loaded`] cell.
    pub mainjitcode_loader: Option<fn() -> Option<std::sync::Arc<crate::jitcode::JitCode>>>,
    /// Jitcode produced by [`Self::mainjitcode_loader`] on this clone.
    pub mainjitcode_loaded: std::sync::OnceLock<std::sync::Arc<crate::jitcode::JitCode>>,
    /// warmspot.py:946 `jd.portal_runner_adr = adr_of(portal_runner)`.
    ///
    /// Address of the portal_runner C function — the funcbox for
    /// `do_recursive_call`'s residual CALL_ASSEMBLER op
    /// (pyjitpl.py).
    pub portal_runner_adr: i64,
    /// warmspot.py `jd.handle_jitexc_from_bh`.
    /// Copied onto [`crate::blackhole::BhJitDriverSd`] for
    /// `_handle_jitexception_in_portal`.
    pub handle_jitexc_from_bh: Option<crate::blackhole::PortalRunnerHook>,
    /// jitdriver.py:16 + warmspot.py:520-545 `jd.virtualizable_info`.
    ///
    /// Per-driver `VirtualizableInfo` populated during warmspot setup
    /// (`make_virtualizable_infos`).  Read by
    /// `MIFrame._do_jit_force_virtual` (pyjitpl.py),
    /// `vable_and_vrefs_before_residual_call` (pyjitpl.py), and
    /// peers that consult `self.metainterp.jitdriver_sd.virtualizable_info`.
    /// Pyre keeps a duplicate single-slot on `MetaInterp` for the
    /// production path that hasn't yet rerouted through this slot —
    /// `MetaInterp::register_jitdriver_sd` propagates from MetaInterp
    /// to here when both are present so per-driver readers stay in
    /// sync without forcing the production registration refactor.
    pub virtualizable_info: Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>>,
    /// jitdriver.py:17 + warmspot.py:520-525 `jd.greenfield_info`.
    ///
    /// Per-driver `GreenFieldInfo` populated when `jd.jitdriver.greens`
    /// contains a dotted name.  Read by `_do_jit_force_virtual`
    /// (pyjitpl.py) — the bail check fires when both this slot
    /// and `virtualizable_info` are None.
    pub greenfield_info: Option<crate::greenfield::GreenFieldInfo>,
    /// jitdriver.py:28 + warmspot.py:529/538 `jd.index_of_virtualizable`.
    ///
    /// Index inside the red-arg list pointing to the virtualizable
    /// argument; -1 when the driver has no virtualizable.  Mirrored
    /// by `virtualizable_arg_index()` so legacy callers keep
    /// returning `Option<usize>`.
    pub index_of_virtualizable: i32,
    /// jitdriver.py:11 `self.portal_calldescr ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:1013` `jd.portal_calldescr = self.cpu.calldescrof(...)`.
    /// The `CallDescr` describing the portal_runner signature; read by
    /// `compile_tmp_callback` (`compile.py`) to pick the
    /// `CALL_*` opcode and attach as the residual call's descr.
    pub portal_calldescr: Option<majit_ir::DescrRef>,
    /// jitdriver.py:21 `self.portal_finishtoken ... rpython.jit.metainterp.pyjitpl`.
    ///
    /// `pyjitpl.py:2279-2280` sets this to the type-appropriate
    /// `MetaInterpStaticData.done_with_this_frame_descr_{int,ref,float,void}`
    /// singleton.  Used as the `FINISH` descr emitted by
    /// `compile_tmp_callback` (`compile.py`).
    pub portal_finishtoken: Option<majit_ir::DescrRef>,
    /// jitdriver.py:22 `self.propagate_exc_descr .. rpython.jit.metainterp.pyjitpl`.
    ///
    /// `pyjitpl.py` shares a single `PropagateExceptionDescr`
    /// instance across all jitdrivers.  Used as the `GUARD_NO_EXCEPTION`
    /// descr emitted by `compile_tmp_callback` (`compile.py`).
    pub propagate_exc_descr: Option<majit_ir::DescrRef>,
    /// jitdriver.py:14 `self.red_args_types ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:664` `red_args_types = [history.getkind(TYPE)[0] for
    /// TYPE in jd._JIT_ENTER_FUNCTYPE.ARGS[jd.num_green_args:]]`.
    /// One char per red arg: 'i' / 'r' / 'f'. Consumed by
    /// `compile_tmp_callback` (`compile.py`) to decide the
    /// `CALL_*` opcode for each red and to size `rd_consts` /
    /// `rd_unicodes` arrays.
    ///
    /// Populated at construction from `vars[red].tp.to_char()`.
    pub red_args_types: Vec<char>,
    /// jitdriver.py:20 `self.no_loop_header ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:558` sets True when the codewriter has not emitted a
    /// `loop_header` op for this driver — read by `MIFrame.opimpl_jit_
    /// merge_point` (`pyjitpl.py`) to skip the implicit
    /// `can_enter_jit` check at the merge point.
    ///
    /// Default `false` matches upstream's "attribute absent until
    /// warmspot decides otherwise" state.
    pub no_loop_header: bool,
    /// jitdriver.py `self.assembler_helper_adr` (CALL_ASSEMBLER).
    ///
    /// `warmspot.py:1031-1035` `jd.assembler_helper_adr =
    /// llmemory.cast_ptr_to_adr(self.cpu.cast_ptr_to_int(
    /// assembler_helper_ptr))`. Address of the per-driver
    /// `assembler_helper` callable invoked by the backend's
    /// `CALL_ASSEMBLER` op to handle the `done_with_this_frame_*` /
    /// `propagate_exception` post-loop branch.
    ///
    /// Stays `0`: the backends reach the same post-loop branch through a
    /// registration API instead of a per-driver address, so nothing ever
    /// populates this field. Cranelift and dynasm both call
    /// `register_call_assembler_{blackhole,bridge,force,unbox_int}`
    /// (`majit-backend-dynasm/src/lib.rs:192-198`), which `pyre-jit`
    /// fills in at boot (`call_jit.rs`). The field is kept because
    /// `jitdriver.py` names it; it is not "not yet wired".
    pub assembler_helper_adr: i64,
    /// jitdriver.py `self.vable_token_descr` (CALL_ASSEMBLER).
    ///
    /// `warmspot.py:546` `jd.vable_token_descr = self.cpu.fielddescrof(
    /// VABLERTI, 'vable_token')`. Used by the backend's `CALL_ASSEMBLER`
    /// op to load / clear the virtualizable's `vable_token` slot when
    /// crossing the assembler-helper boundary.
    ///
    /// `None` until the per-driver virtualizable layout is known —
    /// populated alongside `virtualizable_info` once the host runtime
    /// supplies the field descriptor.
    pub vable_token_descr: Option<majit_ir::DescrRef>,
    /// Per-driver override for the `jitcode.py enumerate_vars` frame box
    /// count (`length_i + length_r + length_f`) that `rebuild_from_numbering`
    /// asks for while building guard metadata.
    ///
    /// Upstream needs no such thing: one codewriter run numbers every jitcode
    /// and interns every `-live-` triple into the single
    /// `metainterp_sd.liveness_info` pool (`pyjitpl.py:2264`), so the global
    /// decode is unambiguous. pyre has **two** numbering spaces — the runtime
    /// CodeObject-keyed store jd0 grows via `intern_liveness`, and the
    /// build-time `jitcode_runtime` artifacts a driver over an extracted
    /// interpreter body (jd1 `unpackiterable_driver`) numbers against. A frame
    /// numbered in one space and decoded against the other reads an unrelated
    /// frame's liveness triple.
    ///
    /// Selecting the store therefore has to be keyed on the driver, which is
    /// what this field does: `None` keeps the process-global callback
    /// (`majit_ir::resumedata::set_frame_value_count_fn`), and a driver whose
    /// frames are numbered elsewhere installs its own decoder here.
    ///
    /// A "try the global store, fall back on failure" scheme cannot replace
    /// this. The wrong store frequently *succeeds*: its low indices hold
    /// unrelated jitcodes that decode at the same pc and silently return a
    /// mistyped count. A successful decode is not evidence of the right store.
    pub frame_value_count_fn: Option<fn(i32, i32) -> usize>,
    /// warmstate.py `WarmEnterState.get_unique_id(greenkey)`.
    ///
    /// `newframe` records `ENTER_PORTAL_FRAME(jd_no, unique_id)` from this
    /// hook when a greenkey is present. The greens are the typed key's
    /// values in declaration order — `interp_jit.py get_unique_id(next_instr,
    /// is_being_profiled, bytecode)`. `None` leaves the portal-frame unique
    /// id as the greenkey hash, the previous fallback.
    pub get_unique_id: Option<fn(&[i64]) -> i64>,
    /// warmspot.py `make_enter_function`: `jd.warmstate = WarmEnterState(self, jd)`.
    ///
    /// Absent until `MetaInterp::make_enter_function` runs for this slot.
    /// Clones of the descriptor share the one state.
    pub warmstate: Option<std::rc::Rc<std::cell::RefCell<crate::warmstate::WarmEnterState>>>,
}

impl JitDriverStaticData {
    /// Create a descriptor from green and red variable lists.
    pub fn new(greens: Vec<(&str, Type)>, reds: Vec<(&str, Type)>) -> Self {
        Self::with_virtualizable(greens, reds, None)
    }

    /// Create a descriptor where greens carry a full [`GreenType`]
    /// (preserving STR/UNICODE subtypes) instead of the IR-collapsed
    /// `Type::Ref`.  Mirrors `warmspot.py:663 _green_args_spec`.
    pub fn with_green_types(greens: Vec<(&str, GreenType)>, reds: Vec<(&str, Type)>) -> Self {
        Self::with_green_types_and_virtualizable(greens, reds, None)
    }

    /// [`Self::with_green_types`] + optional virtualizable name.
    pub fn with_green_types_and_virtualizable(
        greens: Vec<(&str, GreenType)>,
        reds: Vec<(&str, Type)>,
        virtualizable: Option<&str>,
    ) -> Self {
        let mut vars = Vec::new();
        for (name, gt) in greens {
            vars.push(JitDriverVar::green_with_type(
                name,
                green_type_to_ir(gt),
                gt,
            ));
        }
        for (name, tp) in reds {
            vars.push(JitDriverVar::red(name, tp));
        }
        Self::finish_with_vars(vars, virtualizable)
    }

    /// Create a descriptor with optional virtualizable metadata.
    pub fn with_virtualizable(
        greens: Vec<(&str, Type)>,
        reds: Vec<(&str, Type)>,
        virtualizable: Option<&str>,
    ) -> Self {
        let mut vars = Vec::new();
        for (name, tp) in greens {
            vars.push(JitDriverVar::green(name, tp));
        }
        for (name, tp) in reds {
            vars.push(JitDriverVar::red(name, tp));
        }
        Self::finish_with_vars(vars, virtualizable)
    }

    fn finish_with_vars(vars: Vec<JitDriverVar>, virtualizable: Option<&str>) -> Self {
        // warmspot.py:664 — `red_args_types` carries one history-kind
        // char per red arg in declaration order. Built from the red-only
        // slice of `vars` so adding a green/red split helper later (S2)
        // does not need a parallel update path here.
        let red_args_types: Vec<char> = vars
            .iter()
            .filter(|v| v.kind == VarKind::Red)
            .map(|v| v.tp.to_char())
            .collect();
        let mut sd = JitDriverStaticData {
            index: None,
            name: "jitdriver".into(),
            vars,
            virtualizable: virtualizable.map(str::to_string),
            flat_entry: None,
            result_type: Type::Ref,
            is_recursive: false,
            mainjitcode: None,
            mainjitcode_loader: None,
            mainjitcode_loaded: std::sync::OnceLock::new(),
            portal_runner_adr: 0,
            handle_jitexc_from_bh: None,
            virtualizable_info: None,
            greenfield_info: None,
            index_of_virtualizable: -1,
            portal_calldescr: None,
            portal_finishtoken: None,
            propagate_exc_descr: None,
            red_args_types,
            no_loop_header: false,
            assembler_helper_adr: 0,
            vable_token_descr: None,
            frame_value_count_fn: None,
            get_unique_id: None,
            warmstate: None,
        };
        // warmspot.py:529/538 — keep `index_of_virtualizable` in sync
        // with the `virtualizable_arg_index()` derived from `reds`.
        // Stays `-1` when the driver has no virtualizable.
        if let Some(idx) = sd.virtualizable_arg_index() {
            sd.index_of_virtualizable = idx as i32;
        }
        sd
    }

    /// `warmspot.py:1013-1017` `jd.portal_calldescr = self.cpu.calldescrof(
    ///     jd._PTR_PORTAL_FUNCTYPE.TO, ...ARGS, ...RESULT, EffectInfo.MOST_GENERAL)`.
    ///
    /// Populates `self.portal_calldescr` from `self.vars` (green + red
    /// types, in declaration order) and `self.result_type`.  Idempotent:
    /// a fresh descriptor is built every call so subsequent mutations of
    /// `result_type` propagate.
    pub fn build_portal_calldescr(&mut self) {
        use majit_ir::{EffectInfo, SimpleCallDescr};
        use std::sync::atomic::{AtomicU32, Ordering};
        // `warmspot.py` passes `EffectInfo.MOST_GENERAL` — the portal
        // runner can do anything.
        static NEXT_IDX: AtomicU32 = AtomicU32::new(0x4100_0000);
        let idx = NEXT_IDX.fetch_add(1, Ordering::Relaxed);
        // `warmspot.py` `jd._PTR_PORTAL_FUNCTYPE.TO.ARGS`: the full
        // argument list of the portal runner, i.e. green + red (same
        // order as `vars`).  pyre keeps the same layout via `VarKind`.
        let arg_types: Vec<Type> = self.vars.iter().map(|v| v.tp).collect();
        // `descr.py:478` `result_signed = True` for signed ints; pyre
        // defaults to signed since `Type::Int` is the 64-bit signed slot.
        let result_size = match self.result_type {
            Type::Void => 0,
            _ => 8,
        };
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleCallDescr::new(
            idx,
            arg_types,
            self.result_type,
            true,
            result_size,
            EffectInfo::MOST_GENERAL,
        ));
        self.portal_calldescr = Some(descr);
    }

    /// Index separating the constructor-preserved `[greens..., reds...]`
    /// layout. Keeping the declaration partition in `vars` lets both accessors
    /// below borrow slices instead of allocating pointer vectors.
    fn green_red_split(&self) -> usize {
        let split = self.vars.partition_point(|var| var.kind == VarKind::Green);
        debug_assert!(
            self.vars[split..]
                .iter()
                .all(|var| var.kind == VarKind::Red),
            "JitDriverStaticData vars must keep greens before reds"
        );
        split
    }

    /// Get only the green variables without allocating.
    pub fn greens(&self) -> &[JitDriverVar] {
        &self.vars[..self.green_red_split()]
    }

    /// Get only the red variables without allocating.
    pub fn reds(&self) -> &[JitDriverVar] {
        &self.vars[self.green_red_split()..]
    }

    /// Number of green variables.
    pub fn num_greens(&self) -> usize {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .count()
    }

    /// Number of red variables.
    pub fn num_reds(&self) -> usize {
        self.vars.iter().filter(|v| v.kind == VarKind::Red).count()
    }

    /// Per-kind counts of green variables, in `(Int, Ref, Float)` order.
    ///
    /// Mirrors the partition `bhimpl_jit_merge_point` payload encodes via
    /// `@arguments("self", "i", "I", "R", "F", ...)` (`blackhole.py`):
    /// the greens are split by IR kind so the typed register lists in the
    /// `BC_JIT_MERGE_POINT` body line up with the JitDriver schema. Slice
    /// 91.5 uses these counts to validate the runtime payload shape.
    pub fn green_kind_counts(&self) -> (usize, usize, usize) {
        kind_counts(&self.vars, VarKind::Green)
    }

    /// Per-kind counts of red variables, in `(Int, Ref, Float)` order.
    ///
    /// Companion to [`Self::green_kind_counts`]; together they reproduce the six
    /// `[I, R, F, I, R, F]` length bytes of `BC_JIT_MERGE_POINT`.
    pub fn red_kind_counts(&self) -> (usize, usize, usize) {
        kind_counts(&self.vars, VarKind::Red)
    }

    /// jitdriver.py:12 `self.num_green_args ... rpython.jit.metainterp.warmspot`.
    ///
    /// Upstream-canonical name; alias of [`Self::num_greens`]. Matches
    /// `warmspot.py` `jd.num_green_args = len(jd._green_args_spec)`
    /// callers (`compile.py` `compile_tmp_callback`,
    /// `pyjitpl.py verify_green_args` green-constant assertion).
    pub fn num_green_args(&self) -> usize {
        self.num_greens()
    }

    /// jitdriver.py:13 `self.num_red_args ... rpython.jit.metainterp.warmspot`.
    ///
    /// Upstream-canonical name; alias of [`Self::num_reds`]. Matches
    /// `warmspot.py:666` `jd.num_red_args = len(red_args_types)` callers
    /// (`compile.py:1107-1118` red-arg loop bounds).
    pub fn num_red_args(&self) -> usize {
        self.num_reds()
    }

    /// `warmspot.py:663` `jd._green_args_spec = [TYPE for ARG, TYPE in
    /// jd._JIT_ENTER_FUNCTYPE.ARGS[:jd.num_green_args]]`.
    ///
    /// Returns the per-green-arg `GreenType` spec used by
    /// `make_unwrap_greenkey` (`warmstate.py`) and
    /// `make_jitcell_subclass` (`warmstate.py`) to dispatch
    /// `equal_whatever` / `hash_whatever`.
    ///
    /// `JitDriverVar.green_type` carries the upstream lltype subtype
    /// (`Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`) when the install path
    /// supplies it via [`JitDriverStaticData::with_green_types`] /
    /// [`JitDriver::declare_schema_typed`]; greens declared via the
    /// canonical [`Type`]-only API fall back to `GreenType::from(tp)`.
    pub fn green_args_spec(&self) -> Vec<GreenType> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .map(|v| v.green_type.unwrap_or_else(|| GreenType::from(v.tp)))
            .collect()
    }

    /// Convert `self.red_args_types` (history-kind char list per
    /// `warmspot.py:664`) into the IR-level `Type` slice consumed by
    /// `compile_tmp_callback` (`compile.py` red-arg loop).
    ///
    /// Pyre stores `red_args_types: Vec<char>` to mirror upstream's
    /// `'i' / 'r' / 'f'` representation; this helper bridges to the IR
    /// `Type` enum without the caller having to re-implement the
    /// mapping. A later cutover will drop the explicit `red_arg_types`
    /// argument from `compile_tmp_callback` and read from the jd
    /// directly via this helper.
    ///
    /// Panics if any entry is not one of 'i', 'r', 'f', 'v'.
    pub fn red_arg_types_as_ir_types(&self) -> Vec<Type> {
        self.red_args_types
            .iter()
            .map(|c| match c {
                'i' => Type::Int,
                'r' => Type::Ref,
                'f' => Type::Float,
                'v' => Type::Void,
                other => panic!(
                    "JitDriverStaticData::red_arg_types_as_ir_types: \
                     unknown history kind char {:?} in red_args_types — \
                     warmspot.py:664 emits only 'i' / 'r' / 'f' / 'v'",
                    other
                ),
            })
            .collect()
    }

    /// `warmstate.py` `make_unwrap_greenkey().unwrap_greenkey(greenkey)`.
    ///
    /// Upstream's `unwrap_greenkey` walks `_green_args_spec` and calls
    /// `unwrap(TYPE, greenbox)` on each `Const` greenbox, returning a
    /// tuple of unwrapped values. Pyre's caller has already extracted
    /// the `i64` payload from the `Const` opref before calling this
    /// helper (the upstream `Const`-isinstance check lives in
    /// `MIFrame::verify_green_args` at `frame.rs` per `pyjitpl.py`),
    /// so the input here is the already-unwrapped value list.
    ///
    /// Panics if `values.len() != self.num_green_args()` — upstream's
    /// spec-driven loop has an implicit assertion via the
    /// `unrolling_iterable` length match.
    pub fn unwrap_greenkey(&self, values: &[i64]) -> GreenKey {
        let spec = self.green_args_spec();
        assert_eq!(
            values.len(),
            spec.len(),
            "unwrap_greenkey: expected {} green values per _green_args_spec, got {}",
            spec.len(),
            values.len(),
        );
        GreenKey::with_types(values.to_vec(), spec)
    }

    /// Get the virtualizable variable, if any.
    pub fn virtualizable(&self) -> Option<&JitDriverVar> {
        let name = self.virtualizable.as_deref()?;
        self.vars.iter().find(|var| var.name == name)
    }

    /// rewrite.py:684 `jd.index_of_virtualizable` parity.
    ///
    /// Returns the index inside the ENTRY arglist — the values the compiled
    /// entry is invoked with — not the absolute index in `vars`. When the reds
    /// ARE the entry, that is a position among the reds and this agrees with
    /// [`Self::red_arg_virtualizable_index`]. When the entry is a flat
    /// state-field prefix the two part company: the reds describe the
    /// merge-point payload, where the whole state is a single red, so the same
    /// number picks out a different value in each list. The declared contract
    /// takes precedence here; see [`FlatEntryContract`].
    ///
    /// A consumer indexing an original CALL_ASSEMBLER red arglist wants the
    /// other accessor, whichever driver it holds.
    pub fn virtualizable_arg_index(&self) -> Option<usize> {
        if let Some(flat) = self.flat_entry {
            return Some(flat.index_of_virtualizable);
        }
        self.red_arg_virtualizable_index()
    }

    /// `warmspot.py jd.index_of_virtualizable = jitdriver.reds.index(vname)`.
    ///
    /// The virtualizable's position among the REDS, and only that — a flat
    /// entry contract does not override it, because the list this indexes is
    /// the red arglist a CALL_ASSEMBLER carries rather than the compiled
    /// entry's. Upstream has one coordinate system and spells it this way;
    /// a consumer whose list is `num_red_args` wide has to as well.
    pub fn red_arg_virtualizable_index(&self) -> Option<usize> {
        let name = self.virtualizable.as_deref()?;
        self.reds().iter().position(|var| var.name == name)
    }

    /// The flat entry contract, when this driver declares one.
    ///
    /// A consumer that needs the compiled loop's entry WIDTH must consult this
    /// rather than `num_reds()`: the two count different things, so a flat
    /// driver answered with a red count gets a truncation point in the middle
    /// of its live entry values. The virtualizable's position inside that
    /// width is already served model-aware by
    /// [`Self::virtualizable_arg_index`].
    pub fn flat_entry_contract(&self) -> Option<FlatEntryContract> {
        self.flat_entry
    }
}

/// Trait implemented by declarative `#[jit_driver]` marker types.
///
/// This provides a stable seam between proc-macro-generated driver metadata
/// and the runtime `JitDriver` orchestration layer.
pub trait DeclarativeJitDriver {
    const GREENS: &'static [&'static str];
    const REDS: &'static [&'static str];
    const NUM_VARS: usize;
    const NUM_GREENS: usize;
    const NUM_REDS: usize;
    const VIRTUALIZABLE: Option<&'static str>;

    fn descriptor(
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<JitDriverStaticData, &'static str>;

    fn green_key(values: &[i64]) -> Result<GreenKey, &'static str>;
}

/// A named entry point registered with a [`JitDriver`].
///
/// Multiple functions can share the same JitDriver and compiled loops
/// by registering additional entry points with distinct green key schemas.
/// This models the warmspot multi-entry pattern where several
/// `jit_merge_point` calls in different functions share one driver.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Human-readable name of this entry point (e.g., function name).
    pub name: String,
    /// warmstate.py:564 `_green_args_spec` — per-entry TYPE used by
    /// `JitCell.comparekey` / `get_uhash`. `GreenType` retains the
    /// STR / UNICODE distinction that IR `Type` collapses into `Ref`.
    pub schema: Vec<GreenType>,
}

/// High-level JIT driver that automates the tracing lifecycle.
///
/// Wraps [`MetaInterp`] and manages symbolic state, replacing ~80 lines
/// of boilerplate in each interpreter's mainloop with two method calls:
///
/// - [`merge_point`](JitDriver::merge_point) — replaces the tracing hook
///   (`jit_merge_point` in RPython)
/// - [`back_edge`](JitDriver::back_edge) — replaces back-edge handling
///   (`can_enter_jit` in RPython)
///
/// # Type Parameter
///
/// `S` implements [`JitState`] and defines how the interpreter's live
/// state maps to/from the JIT's representation.
/// PyPy warmspot.py JitDriver equivalent.
///
/// Manages the tracing/compilation lifecycle for a single jitdriver.
/// PyPy names: JitDriver(greens=[...], reds=[...], is_recursive=True).
pub struct JitDriver<S: JitState> {
    meta: MetaInterp<S::Meta>,
    sym: Option<S::Sym>,
    /// pyjitpl.py reached_loop_header direct compile_trace parity.
    /// The frontend decides whether to call compile_trace(); the driver only
    /// consumes the successful result and switches back to compiled code.
    compile_trace_success: bool,
    /// pyjitpl.py `raise_continue_running_normally` payload for a
    /// successful close-loop edge.
    continue_running_normally_payload: Option<(Vec<Value>, Option<usize>)>,
    descriptor: Option<JitDriverStaticData>,
    /// Memoized [`Self::driver_descriptor_for`] answer.
    ///
    /// `warmspot.py make_driverhook_graphs` builds each
    /// `JitDriverStaticData` once at translation time and stores it in
    /// `metainterp_sd.jitdrivers_sd`; every runtime path then reads that one
    /// object. Here the answer was rebuilt on each consultation instead —
    /// either by cloning `descriptor` or by calling `JitState::driver_descriptor`
    /// — which allocates the `vars` Vec plus one `String` per variable and frees
    /// them again. Entering compiled code consults it, so a call-heavy program
    /// paid that malloc/free pair per entry.
    ///
    /// The outer `Option` is "already resolved", the inner one is the answer.
    /// Both producers are configuration-time constants: `descriptor` is only
    /// written by `declare_schema`, `declare_schema_typed` and
    /// `register_descriptor` (each of which clears this cache), and no
    /// `JitState::driver_descriptor` implementation reads its `meta` argument.
    descriptor_cache: Option<Option<std::sync::Arc<JitDriverStaticData>>>,
    /// Identity/cell reds the lowerer lists on `jit_merge_point` after the
    /// named portal reds (`identity_slot_registers` plus Meta-length
    /// flattened `[int]` cells). Not `JitDriverStaticData::vars`: those
    /// are the portal-runner arguments (`warmspot.py`
    /// `cpu.calldescrof(portal_runner)`). `(int, ref, float)`.
    merge_point_identity_reds: (usize, usize, usize),
    /// Whether the live bridge trace was entered at a guard's own jitcode
    /// position rather than at a merge point the blackhole walked to.
    ///
    /// The abort handling in [`Self::merge_point`] skips its resume handoff
    /// for a bridge, because a bridge used to be started only after the
    /// blackhole had already resumed the interpreter. A bridge entered at the
    /// guard has had no such resume — the walk is the only thing that ran —
    /// so it needs the handoff like a fresh trace does.
    bridge_entered_at_guard_resume: bool,
    /// resume.py: result of rebuild_from_resumedata for bridge tracing.
    /// RPython stores this in MIFrame registers; pyre stores it here
    /// for the caller to initialize PyreSym slot-to-OpRef mapping.
    pub resume_data_result: Option<crate::jit_state::ResumeDataResult>,
    /// RPython pyjitpl.py:3101 parity: true when the current bridge
    /// traces from an exception guard (GUARD_EXCEPTION / GUARD_NO_EXCEPTION).
    /// The caller should emit SAVE_EXC_CLASS + SAVE_EXCEPTION at trace start.
    pub last_bridge_is_exception_guard: bool,
    /// Whether this session's bridge attempt was declined, or had no target to
    /// close against -- the session PHASE, as distinct from the resumekey CLASS
    /// that `MetaInterp::bridge_info` carries.
    ///
    /// Upstream needs no such flag: `reached_loop_header`
    /// (pyjitpl.py) re-attempts `compile_trace` on every loop-header
    /// visit while `not self.partial_trace`, and never writes `self.resumekey`
    /// outside session start (pyjitpl.py / 2939). majit latches the
    /// declined attempt instead, so the latch needs storage of its own --
    /// clearing `bridge_info` to express it also destroys the class that
    /// `compile.py resumekey.compile_and_attach` dispatches on.
    bridge_attempt_declined: bool,
    /// Additional entry points sharing this driver's compiled loops.
    entry_points: Vec<EntryPoint>,
    /// PyPy JitDriver(is_recursive=True): enables max_unroll_recursion
    /// for recursive portal calls (pyjitpl.py _opimpl_recursive_call).
    is_recursive: bool,
    /// Driver-shared `Assembler`: holds the
    /// `all_liveness` payload (`assembler.py:30`) populated incrementally
    /// by `__JitMeta::install_canonical_liveness` (canonical entry) and
    /// each per-pc JitCode build (per-marker triples via
    /// `JitCodeBuilder::finalize_liveness`).
    ///
    /// The Mutex serialises factory invocations from blackhole resume
    /// callers; in practice tracing/resume run single-threaded for the
    /// state-field JIT consumer, so contention is negligible.
    ///
    /// Snapshot contract: factory calls that can append distinct
    /// liveness entries must run before tracing starts, followed by
    /// `sync_liveness_info_from_shared_asm()`.  This preserves PyPy's
    /// `make_jitcodes() -> finish_setup()` order while keeping
    /// `MetaInterpStaticData.liveness_info` immutable during tracing.
    shared_asm: std::sync::Arc<parking_lot::Mutex<majit_jitcode::codewriter::assembler::Assembler>>,
    /// resume.py:1367 — CPU allocation backend for virtual materialization
    /// during blackhole resume. Registered by the embedding interpreter at
    /// startup.
    blackhole_allocator: Option<Box<dyn crate::resume::BlackholeAllocator + Send>>,
    /// warmspot.py handle_jitexception parity: portal runner callback.
    /// Called when ContinueRunningNormally is raised at a recursive portal
    /// jtransform.py `portal_jd.index` — the `jitdrivers_sd` slot this
    /// driver's portal occupies. `call.py:147 jd.mainjitcode` holds the portal
    /// JitCode itself, so this driver stores only the slot; `None` until
    /// `register_dispatch_jitcode` runs.
    portal_jd_index: Option<usize>,
    /// This driver's payload for the per-thread `frame_value_count` store,
    /// built once by `register_dispatch_jitcode`. `None` until then.
    state_field_fvc: Option<StateFieldFvcData>,
    /// One-shot skip for the next [`Self::function_entry_structured`].
    ///
    /// An embedder whose loop-header door can already serve the call (CEL
    /// yields when a sibling loop key is runnable) sets this so the generated
    /// `ll_portal_runner` prefix does not mint a second artifact for the same
    /// activation. Taken, not read: a skipped door must not stay skipped.
    function_entry_suppressed: bool,
    /// Reusable buffers for the compiled-entry argument walk.
    ///
    /// `warmstate.py execute_assembler` builds the entry arguments
    /// with no allocation at all: the reds are already unboxed locals, and
    /// `func_execute_token` writes them straight into the jitframe the backend
    /// owns. Here the same walk minted a fresh `Vec` per entry — one for the
    /// raw live values, one for their types, one for the typed pair, and one
    /// more for each virtualizable box list — so a call-heavy program paid
    /// several malloc/free pairs per entry for buffers whose shape is fixed by
    /// the compiled artifact and therefore identical on every call.
    ///
    /// The driver outlives every entry through it, so the buffers are cleared
    /// and refilled rather than reallocated; after the first entry their
    /// capacity already covers the artifact's inputarg count. Same motivation
    /// as [`Self::descriptor_cache`], one layer further in.
    ///
    /// One set is enough even though a compiled run can re-enter the driver:
    /// the taker holds its buffers as a local for the whole entry, so a nested
    /// entry finds this slot empty and starts from fresh ones. Nesting
    /// therefore costs what it costs today and never aliases; only the
    /// outermost entry's buffers survive to be reused.
    ///
    /// Held behind a pointer because the hand-out and the hand-back are moves:
    /// the struct is five `Vec` headers, so an inline slot moved a hundred and
    /// twenty bytes out of this field and the same back into it on every entry,
    /// for buffers whose contents never move at all. `None` is the window
    /// between the two, which is what a nested entry finds.
    entry_scratch: Option<Box<EntryScratch>>,
    /// Machine-word projection of one guard exit's typed deadframe values.
    /// Kept separate from `entry_scratch`: bridge tracing can re-enter this
    /// driver while the outer guard is still consuming these words, and that
    /// nested entry should still be able to reuse the ordinary entry buffers.
    exit_raw_scratch: Option<Vec<i64>>,
}

/// Per-entry scratch owned by [`JitDriver`]; see [`JitDriver::entry_scratch`].
#[derive(Default)]
struct EntryScratch {
    /// The typed live values handed to the backend as the entry's inputargs.
    live_values: Vec<Value>,
    /// `JitState::extract_live` output, before it is paired with its types.
    raw: Vec<i64>,
    /// `JitState::live_value_types` output.
    types: Vec<Type>,
    /// `JitState::export_virtualizable_boxes_into` field output, in
    /// `VirtualizableInfo::read_boxes` order.
    vable_boxes: Vec<i64>,
    /// `JitState::export_virtualizable_boxes_into` per-array lengths.
    vable_lengths: Vec<usize>,
}

impl EntryScratch {
    /// Drop the previous entry's typed buffers.
    ///
    /// `execute_assembler`'s raw path writes only the unspecialized red words,
    /// so [`JitDriver::take_entry_scratch_raw`] leaves these alone. A fill of
    /// `live_values`, `types`, `vable_boxes` or `vable_lengths` clears them
    /// first; `raw` is not touched here.
    fn clear_typed(&mut self) {
        self.live_values.clear();
        self.types.clear();
        self.vable_boxes.clear();
        self.vable_lengths.clear();
    }
}

thread_local! {
    /// Per-thread publication of the currently-installed state-field JIT's
    /// flat jitcode registry + packed liveness, read by the stateless global
    /// `frame_value_count` callback below. Mirrors pyre-jit-trace's
    /// thread-local `METAINTERP_SD` (`jitcodes` + `liveness_info` + `op_live`).
    static STATE_FIELD_FVC: std::cell::RefCell<Option<StateFieldFvcData>> =
        const { std::cell::RefCell::new(None) };
}

/// Identity stamped on each driver's publication so a thread can tell whose
/// store it is currently holding. Non-zero; `0` is never handed out.
static STATE_FIELD_FVC_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One driver's contribution to the per-thread store, assembled once at
/// `register_dispatch_jitcode` so re-publishing costs three refcount bumps
/// rather than cloning the registry and the liveness buffer.
#[derive(Clone)]
struct StateFieldFvcData {
    /// Which driver built this. Compared against the thread's current store to
    /// skip a redundant install.
    epoch: u64,
    jitcodes: std::sync::Arc<Vec<std::sync::Arc<crate::jitcode::JitCode>>>,
    all_liveness: std::sync::Arc<Vec<u8>>,
    op_live: u8,
}

/// resume.py:1050/1338 `jitcode = metainterp_sd.jitcodes[jitcode_pos]` parity
/// for the state-field JIT compile-time multi-frame decode
/// (`rebuild_from_numbering`). Resolves each frame's jitcode STATELESSLY from
/// the flat global registry by its self-describing absolute index, then
/// derives the per-frame box count from that jitcode's liveness at pc
/// (jitcode.py `enumerate_vars` → `length_i + length_r + length_f`).
/// Structural mirror of `pyre-jit-trace::state::frame_value_count_at`.
fn state_field_frame_value_count(jitcode_index: i32, pc: i32) -> usize {
    STATE_FIELD_FVC.with(|cell| {
        let data = cell.borrow();
        let Some(data) = data.as_ref() else {
            return 0;
        };
        // A negative index or pc is not a registry slot. `as usize`
        // wraps it and `get_live_vars_info` then reads another frame's
        // liveness.
        let Some(index) = usize::try_from(jitcode_index).ok() else {
            return 0;
        };
        let Some(resolved_pc) = usize::try_from(pc).ok() else {
            return 0;
        };
        let Some(jc) = data.jitcodes.get(index) else {
            return 0;
        };
        let off = jc.get_live_vars_info(resolved_pc, data.op_live);
        let all_liveness = &data.all_liveness;
        if off + 2 < all_liveness.len() {
            all_liveness[off] as usize
                + all_liveness[off + 1] as usize
                + all_liveness[off + 2] as usize
        } else {
            0
        }
    })
}

/// Epoch of the state-field store this thread is currently holding, or `0`
/// when no driver has published on it yet. Diagnostic for the single-slot
/// store described on [`JitDriver::republish_state_field_fvc`].
#[doc(hidden)]
pub fn current_state_field_fvc_epoch() -> u64 {
    STATE_FIELD_FVC.with(|cell| cell.borrow().as_ref().map_or(0, |data| data.epoch))
}

/// Publish the driver's flat jitcode registry + packed liveness for the
/// stateless global `frame_value_count` decode. Both compile-time decoders
/// (`compile.rs` build_guard_metadata and the cranelift backend) read the
/// callback via `get_frame_value_count_fn`, so this single registration
/// serves both. Registers the callback once (process-global slot), then
/// refreshes the thread-local payload for the installing driver — but only
/// when the thread is not already holding that driver's store, so the
/// trace-entry calls that keep it honest cost one `u64` compare.
fn install_state_field_fvc(data: &StateFieldFvcData) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        majit_ir::resumedata::set_frame_value_count_fn(state_field_frame_value_count);
    });
    STATE_FIELD_FVC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().is_some_and(|held| held.epoch == data.epoch) {
            return;
        }
        *slot = Some(data.clone());
    });
}

/// Why [`JitDriver::bridge_from_guard_resume_position`] gave the guard's own
/// position up and handed the failure to the blackhole arm instead.
///
/// Its ladder runs before the walk and every rung ends in the same
/// `abort_trace`, which is counted under one `Counters.ABORT_*` id that the
/// unclassified default also uses. The rung is therefore only knowable at the
/// point it is chosen; carrying it out of the closure is what lets
/// [`Self::diag_slot`] tally each one separately.
#[derive(Clone, Copy)]
enum GuardResumeDecline {
    /// No rebuilt resume data, no frame in it, no position to convert, or no
    /// live trace session to seed — there is nothing to enter at.
    NoResumeState,
    /// The frame names a jitcode other than the dispatch one. A position is an
    /// offset into the jitcode the frame NAMES, so entering a different one
    /// there lands mid-instruction in unrelated code.
    ForeignJitcode,
    /// The applying reader met a write it has no applying twin for, so the
    /// trace records a store the heap did not receive.
    ReplayIncomplete,
    /// The register-index count and the rebuilt-value count disagree, so the
    /// two sides read different liveness and no per-register pairing off them
    /// is trustworthy.
    RegCountMismatch,
    /// A live register holds a virtual, which has no concrete value, or nothing
    /// at all. The walk executes, so a register it cannot read a value out of
    /// is not one it can run past.
    UnreadableRegister,
    /// A section names a `jitcode_index` the flat registry does not hand out.
    /// `resume.py:1051` indexes that registry directly, so an index outside it
    /// names no code to enter.
    UnregisteredJitcode,
    /// A section below the top is suspended somewhere that is not the far side
    /// of a `BC_INLINE_CALL`, so nothing there says which callee it is waiting
    /// on or which register takes the result.
    NoInlineCallSite,
    /// The call site's `j` operand does not resolve to a JitCode, so the frame
    /// stacked on top of it is not the one the bytecode names.
    CallSiteNotAJitcode,
    /// The call site names a different callee than the section actually stacked
    /// on it, so the two descriptions of the same call stack disagree.
    CallSiteCalleeMismatch,
    /// The callee ends in a typed return but the call that pushed it has no
    /// slot of that kind, so the pop would have nowhere to put the result.
    CalleeReturnSlotMismatch,
    /// The resume spans more than one frame and the sym reserves an
    /// identity-only int range, which the snapshot trim blanked in every
    /// frame below the root and nothing re-derives.
    ReservedIdentitySlots,
}

impl GuardResumeDecline {
    /// The `MC_DIAG` slot this rung is tallied in.
    const fn diag_slot(self) -> usize {
        match self {
            GuardResumeDecline::NoResumeState => 83,
            GuardResumeDecline::ForeignJitcode => 84,
            GuardResumeDecline::ReplayIncomplete => 85,
            GuardResumeDecline::RegCountMismatch => 86,
            GuardResumeDecline::UnreadableRegister => 87,
            GuardResumeDecline::UnregisteredJitcode => 88,
            GuardResumeDecline::NoInlineCallSite => 89,
            GuardResumeDecline::CallSiteNotAJitcode => 90,
            GuardResumeDecline::CallSiteCalleeMismatch => 91,
            GuardResumeDecline::CalleeReturnSlotMismatch => 92,
            GuardResumeDecline::ReservedIdentitySlots => 93,
        }
    }

    /// The same rung, for the log line that names it without a slot legend.
    const fn label(self) -> &'static str {
        crate::MC_DIAG_LABELS[self.diag_slot()]
    }
}

impl<S: JitState> crate::jit::JitParameterTarget for JitDriver<S> {
    fn set_param(&mut self, name: &str, value: i64) -> Result<(), crate::jit::JitHintError> {
        // warmstate.py JitCounter.compute_threshold: nonpositive thresholds
        // disable counting. The public driver setter takes unsigned values.
        let value = match name {
            "threshold" | "function_threshold" | "trace_eagerness" => value.max(0),
            _ => value,
        };
        self.set_param(name, value);
        Ok(())
    }

    fn set_param_enable_opts(&mut self, value: &str) {
        self.set_param_enable_opts(value);
    }
}

/// Answer of [`JitDriver::enter_compiled_function_entry`].
enum SteadyCompiledEntry {
    /// No compiled token for this hash. The cold door still runs.
    Miss,
    /// A procedure token exists, and the frontend meta does not. The
    /// occupied door (`function_entry_internal`) owns that case.
    NeedsInternal,
    /// This call is finished. `None` is a decline, `Some` a resume pc.
    Done(Option<usize>),
}

/// Answer of [`JitDriver::function_entry_runner`].
///
/// `warmspot.py ll_portal_runner` calls `maybe_compile_and_run` before the
/// portal body. `warmstate.py execute_assembler` returns a
/// `DoneWithThisFrameDescrInt` / `DoneWithThisFrameDescrRef` result straight
/// out of that runner. [`FunctionEntryRunner::Finished`] is that word (an
/// int, or a ref address stored as `i64`). [`FunctionEntryRunner::Resume`]
/// and [`FunctionEntryRunner::Run`] still enter the portal body; those arms
/// keep the finish latch the inline door drains.
pub enum FunctionEntryRunner {
    Finished(i64),
    Resume(usize),
    Run,
}

fn map_runner_resume(resume: Option<usize>) -> FunctionEntryRunner {
    match resume {
        Some(pc) => FunctionEntryRunner::Resume(pc),
        None => FunctionEntryRunner::Run,
    }
}

/// `warmstate.py` `maybe_compile_and_run` before it builds assembler args.
///
/// [`BackEdgeWarmth::Interpret`] is the not-found arm after `jitcounter.tick`
/// returned false: no cell install, no descriptor, no live-value list.
/// [`BackEdgeWarmth::Trace`] is that same tick returning true; the counter is
/// already reset, so the caller publishes live values and calls
/// [`JitDriver::commit_back_edge_trace`] without ticking again.
/// [`BackEdgeWarmth::Full`] is every arm that continues past the cheap lookup
/// (a compiled token, a chain, a dead token, `JC_DONT_TRACE_HERE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackEdgeWarmth {
    Interpret,
    Trace,
    Full,
}

impl<S: JitState> JitDriver<S> {
    /// Create a new JitDriver with the given hot-counting threshold.
    pub fn new(threshold: u32) -> Self {
        let mut meta = MetaInterp::new(threshold);
        // `warmspot.py` sets `jd.result_type` from the portal return kind
        // once, when the driver is built. A state with no `FinishReturn`
        // leaves the `Type::Ref` `MetaInterp::new` starts from.
        if let Some(tp) = S::PORTAL_RESULT_TYPE {
            meta.set_result_type(tp);
        }
        if let Some(info) = S::__build_virtualizable_info() {
            meta.set_virtualizable_info(info);
        }
        JitDriver {
            meta,
            sym: None,
            compile_trace_success: false,
            continue_running_normally_payload: None,
            descriptor: None,
            descriptor_cache: None,
            merge_point_identity_reds: (0, 0, 0),
            bridge_entered_at_guard_resume: false,
            resume_data_result: None,
            last_bridge_is_exception_guard: false,
            bridge_attempt_declined: false,
            entry_points: Vec::new(),
            is_recursive: false,
            // resume.py `allocate_with_vtable` must return an object.
            // `NullAllocator` leaves a virtual ref as 0, so a vable slot
            // that held one is not a live box after the guard.
            blackhole_allocator: Some(Box::new(crate::resume::LlmodelBlackholeAllocator)),
            portal_jd_index: None,
            state_field_fvc: None,
            function_entry_suppressed: false,
            entry_scratch: Some(Box::default()),
            exit_raw_scratch: Some(Vec::new()),
            #[expect(
                clippy::arc_with_non_send_sync,
                reason = "Arc preserves shared JitCode/descriptor identity across compiled artifacts; the non-Send translator payload is confined to the single-threaded build phase and is never transferred between threads"
            )]
            shared_asm: std::sync::Arc::new(parking_lot::Mutex::new(
                majit_jitcode::codewriter::assembler::Assembler::new(),
            )),
        }
    }

    /// Borrow the driver-shared `Assembler` mutex
    /// so macro-emitted install/prebuild paths can append dispatch and
    /// embedded sub-JitCode liveness entries into a single `all_liveness`
    /// table before `staticdata.liveness_info` is snapshotted.
    ///
    /// Returned `Arc<Mutex<...>>` is cheap to clone; clones share the
    /// same underlying mutex.
    pub fn shared_asm(
        &self,
    ) -> std::sync::Arc<parking_lot::Mutex<majit_jitcode::codewriter::assembler::Assembler>> {
        self.shared_asm.clone()
    }

    /// Copy `shared_asm`'s `all_liveness` byte
    /// stream into `staticdata.liveness_info`, mirroring
    /// `pyjitpl.py MetaInterpStaticData.finish_setup self.liveness_info = "".join(asm.all_liveness)`.
    ///
    /// Must run while `staticdata` Arc is uniquely owned (i.e. before
    /// the first trace clones it), after macro-emitted prebuild has
    /// registered every per-marker liveness entry.
    pub fn sync_liveness_info_from_shared_asm(&mut self) {
        let asm_guard = self.shared_asm.lock();
        self.meta.sync_liveness_info(asm_guard.all_liveness());
    }

    /// Install the state-field JIT canonical liveness payload before
    /// any tracing path runs.  Mirrors RPython `warmspot.py:281-289`'s
    /// `make_jitcodes() → finish_setup(codewriter)` for the narrow
    /// `pyjitpl.py MetaInterpStaticData.finish_setup self.liveness_info = "".join(asm.all_liveness)`
    /// slice — full `finish_setup` requires `CodeWriter` /
    /// `CallControl` construction, which is not yet wired.
    ///
    /// Caller responsibility: build a fresh
    /// `majit_jitcode::codewriter::assembler::Assembler`,
    /// register the canonical
    /// `__JitMeta::canonical_liveness_slots()` triple via
    /// `Assembler::_encode_liveness`, then pass `&asm` here.  The
    /// single-owner `Arc::get_mut` invariant (`MetaInterp::finish_setup`)
    /// applies — call before the first trace.
    pub fn install_canonical_liveness(
        &mut self,
        asm: &majit_jitcode::codewriter::assembler::Assembler,
    ) {
        self.meta.install_canonical_liveness(asm);
    }

    /// Register the dispatch JitCode singleton. Called once at install time
    /// by the macro-generated `__JitMeta::install_canonical_liveness`.
    /// `__trace_<fn>` clones this singleton for every trace iteration;
    /// resume-side `resolve_jitcode` closures clone it for the root frame
    /// (`resume.py`).
    ///
    /// RPython parity: `metainterp_sd.jitcodes[portal_jd.index]` global
    /// registry slot assignment, scoped to the per-`#[jit_interp]` driver.
    pub fn register_dispatch_jitcode(&mut self, jitcode: crate::jitcode::JitCode) {
        let (registry, dispatch_arc) =
            build_jitcode_registry(crate::jitcode::global_build_jitcodes(), jitcode);
        self.adopt_dispatch_registry(registry, dispatch_arc);
    }

    /// Register a dispatch JitCode the build-time table already holds.
    ///
    /// Same registration as [`Self::register_dispatch_jitcode`], for the other
    /// shape of consumer: one whose jitcodes are a build artefact published
    /// behind `Arc`, with no owned copy to hand over. Without this door such a
    /// consumer can only reach [`crate::pyjitpl::MetaInterp::install_jitcodes`],
    /// which publishes the table and nothing else — no `BC_JIT_MERGE_POINT`
    /// payload cross-check against the declared schema, and neither half of the
    /// `call.py grab_initial_jitcodes` back-pointer, so `is_main_jitcode` is
    /// false for the portal and a blackhole chain walks off its bottom.
    ///
    /// The jitcode must already name its index, which a build-time table
    /// assigns to every entry; an unnumbered one has to be appended, and
    /// appending needs ownership.
    pub fn register_dispatch_jitcode_shared(
        &mut self,
        jitcode: &std::sync::Arc<crate::jitcode::JitCode>,
    ) {
        let (registry, dispatch_arc) =
            build_seeded_jitcode_registry(crate::jitcode::global_build_jitcodes(), jitcode)
                .unwrap_or_else(|| {
                    panic!(
                        "register_dispatch_jitcode_shared: dispatch JitCode {:?} names no index,                          so it is not one the build-time table numbered; a jitcode that has to be                          appended must be passed by value through `register_dispatch_jitcode`",
                        jitcode.name(),
                    )
                });
        self.adopt_dispatch_registry(registry, dispatch_arc);
    }

    /// The registration both doors share, once the registry and the portal's
    /// own `Arc` have been decided.
    fn adopt_dispatch_registry(
        &mut self,
        registry: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
        dispatch_arc: std::sync::Arc<crate::jitcode::JitCode>,
    ) {
        // Validate after `build_jitcode_registry` has selected the canonical
        // indexed identity. A seeded caller is only a locator: name, index,
        // and merge offset do not prove equal bodies, and validating that
        // disposable shell before adopting `all_jitcodes[index]` would install
        // unchecked bytecode. `codewriter.py CodeWriter.make_jitcodes` owns
        // the identity invariant `all_jitcodes[jitcode.index] is jitcode`;
        // validate that exact object before publishing it as `mainjitcode`.
        validate_dispatch_jitcode_payload(self, &dispatch_arc);
        // call.py `grab_initial_jitcodes`:
        //
        //     jd.mainjitcode = self.get_jitcode(jd.portal_graph)
        //     jd.mainjitcode.jitdriver_sd = jd
        //
        // Both directions, as upstream sets them: the driver's static data owns
        // the portal JitCode, and the portal JitCode names its driver back.
        // `call.py jd.index = idx` is this driver's own slot — the macro's
        // install pipeline runs `ensure_descriptor_registered` before reaching
        // here, so it is assigned. Fall back to the single-portal slot 0 only
        // when a consumer skipped that step.
        if let Some(seed_index) = dispatch_arc.jitdriver_sd() {
            self.ensure_descriptor_registered_at(seed_index);
        } else {
            self.ensure_descriptor_registered();
        }
        let portal_jd_index = self.index().unwrap_or(0);
        // The back-pointer is what makes this jitcode a *portal* jitcode to
        // every reader of `JitCode::jitdriver_sd`: `pyjitpl.py is_main_jitcode
        // is_main_jitcode`, `pyjitpl.py _try_tco`'s portal-frame opt-out,
        // and `blackhole.py:1764`'s walk up a blackhole chain, which without it
        // runs off the bottom of a `#[jit_interp]` frontend's chain instead of
        // stopping at its root.  Translation-side jitcodes get it from
        // `codewriter.rs`'s `make_jitcodes`; this is the macro frontend's equivalent.
        dispatch_arc.set_jitdriver_sd(portal_jd_index);
        self.meta
            .jitdriver_sd_mut(portal_jd_index)
            .expect("register_dispatch_jitcode: this driver's jitdrivers_sd slot is vacant")
            .mainjitcode = Some(dispatch_arc);
        self.portal_jd_index = Some(portal_jd_index);
        // warmspot.py:281-282 `metainterp_sd.jitcodes = make_jitcodes()` — the
        // drained list is owned by the staticdata, which is what `resume.py:1051`
        // indexes. `Arc` identity carries over from the worklist above, so the
        // `Arc::ptr_eq` dedup and the `set_index` stamps stay valid.
        self.meta.install_jitcodes(registry.clone());
        // Publish the registry + packed liveness for the stateless global
        // `frame_value_count` decode. `install_canonical_liveness` ran just
        // before this call, so staticdata carries the final liveness buffer.
        let all_liveness = self.meta_interp().staticdata.liveness_info.snapshot_vec();
        let op_live = self.meta_interp().staticdata.op_live as u8;
        let data = StateFieldFvcData {
            epoch: STATE_FIELD_FVC_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            jitcodes: std::sync::Arc::new(registry),
            all_liveness: std::sync::Arc::new(all_liveness),
            op_live,
        };
        install_state_field_fvc(&data);
        self.state_field_fvc = Some(data);
    }

    /// pyjitpl.py:3294 `self.jitdriver_sd.mainjitcode` — this driver's portal
    /// JitCode. Returns `None` until `register_dispatch_jitcode` has been
    /// called; upstream has no such window, since `call.py grab_initial_jitcodes` assigns
    /// every driver's `mainjitcode` before the metainterp ever runs.
    pub fn dispatch_jitcode(&self) -> Option<&std::sync::Arc<crate::jitcode::JitCode>> {
        self.meta.mainjitcode_of(self.portal_jd_index?)
    }

    /// `call.py grab_initial_jitcodes` for a portal the build-time table
    /// already holds. Production never goes through the macro frontend's
    /// `register_dispatch_jitcode`; the extracted portal is `jd.mainjitcode`.
    pub fn install_extracted_portal_jitcode(
        &mut self,
        jitcode: std::sync::Arc<crate::jitcode::JitCode>,
    ) {
        let portal_jd_index = self.index().unwrap_or(0);
        jitcode.set_jitdriver_sd(portal_jd_index);
        self.meta
            .jitdriver_sd_mut(portal_jd_index)
            .expect("install_extracted_portal_jitcode: jitdrivers_sd slot is vacant")
            .mainjitcode = Some(jitcode);
        self.portal_jd_index = Some(portal_jd_index);
    }

    /// `call.py grab_initial_jitcodes` binding without decoding the portal.
    /// Same slot assignment as [`Self::install_extracted_portal_jitcode`]; the
    /// jitcode is filled by `MetaInterp::mainjitcode_of` on the first read.
    pub fn install_extracted_portal_jitcode_loader(
        &mut self,
        loader: fn() -> Option<std::sync::Arc<crate::jitcode::JitCode>>,
    ) {
        let portal_jd_index = self.index().unwrap_or(0);
        self.meta
            .jitdriver_sd_mut(portal_jd_index)
            .expect("install_extracted_portal_jitcode_loader: jitdrivers_sd slot is vacant")
            .mainjitcode_loader = Some(loader);
        self.portal_jd_index = Some(portal_jd_index);
    }

    /// Point the per-thread state-field store back at this driver, reusing what
    /// [`Self::register_dispatch_jitcode`] already built.
    ///
    /// That store is a SINGLE slot per thread and is written only when a driver
    /// publishes, so a thread holding more than one driver alive can otherwise
    /// decode against whichever published last. That is not a benign miss:
    /// [`JitDriverStaticData::frame_value_count_fn`] records that the wrong
    /// store frequently *succeeds*, decoding an unrelated jitcode at the same pc
    /// and silently returning a mistyped frame count.
    ///
    /// The store is now re-aimed automatically at both trace entries
    /// ([`Self::force_start_tracing`] and [`Self::start_bridge_tracing`]), which
    /// is what every compile is reached through, so calling this by hand is no
    /// longer required — it stays for consumers that already do, and to let a
    /// pool aim the store at checkout rather than at trace start.
    ///
    /// A driver that has not registered a dispatch jitcode has nothing to
    /// publish and is left alone.
    ///
    /// `warmspot.py WarmRunnerDesc.__init__ metainterp_sd.jitcodes` now owns the table
    /// (`MetaInterp::install_jitcodes`), but one `MetaInterpStaticData` per
    /// driver is still not upstream's ONE table, and this store is a separate
    /// per-thread copy of it besides. Re-aiming on entry only removes the
    /// window in which the wrong one is read.
    pub fn republish_state_field_fvc(&self) {
        if let Some(data) = self.state_field_fvc.as_ref() {
            install_state_field_fvc(data);
        }
    }

    /// This driver's store epoch, or `0` before it registered a dispatch
    /// jitcode. Compare against [`current_state_field_fvc_epoch`] to see whose
    /// store the thread is holding.
    #[doc(hidden)]
    pub fn state_field_fvc_epoch(&self) -> u64 {
        self.state_field_fvc.as_ref().map_or(0, |data| data.epoch)
    }

    /// Register a BlackholeAllocator for virtual materialization during
    /// guard failure blackhole resume. Without this, virtual objects
    /// and raw buffers are allocated as null/zero.
    pub fn register_blackhole_allocator(
        &mut self,
        allocator: impl crate::resume::BlackholeAllocator + Send + 'static,
    ) {
        self.blackhole_allocator = Some(Box::new(allocator));
    }

    /// The allocator a reader materializes virtuals through
    /// (`ResumeDataBoxReader.allocate_with_vtable` is
    /// `execute_new_with_vtable`: the box holds the object it allocated).
    pub fn blackhole_allocator(&self) -> Option<&dyn crate::resume::BlackholeAllocator> {
        self.blackhole_allocator
            .as_deref()
            .map(|allocator| allocator as &dyn crate::resume::BlackholeAllocator)
    }

    /// Frames `start_bridge_tracing` already seated via
    /// `rebuild_from_resumedata`. Zero means that rebuild has not run, so a
    /// portal wrapper may still call it.
    pub fn portal_framestack_len(&self) -> usize {
        self.meta.framestack.frames.len()
    }

    /// `MetaInterp::rebuild_portal_framestack_from_resumedata` with this
    /// driver's allocator, so a virtual the register rebuild materializes
    /// is allocated like one a frame value names.
    #[allow(clippy::too_many_arguments)]
    pub fn rebuild_portal_framestack_from_resumedata(
        &mut self,
        mainjitcode: std::sync::Arc<crate::jitcode::JitCode>,
        frames: &[majit_ir::resumedata::RebuiltFrame],
        fail_values: &[i64],
        fail_types: &[majit_ir::Type],
        materialized: &[Option<std::sync::Arc<crate::jitcode::JitCode>>],
        resume_liveness: &[u8],
        resume_op_live: u8,
        reader: &mut crate::BridgeVirtualCache<'_>,
    ) -> bool {
        self.meta.rebuild_portal_framestack_from_resumedata(
            mainjitcode,
            frames,
            fail_values,
            fail_types,
            materialized,
            resume_liveness,
            resume_op_live,
            reader,
        )
    }

    /// Pre-register the per-driver green / red schema with the embedded
    /// `JitDriverStaticData::vars`.  The macro-emitted install path
    /// (`#[jit_interp]` codegen) calls this before
    /// `ensure_descriptor_registered` runs, so
    /// `green_kind_counts` / `red_kind_counts` reflect the actual schema
    /// the dispatch JitCode body's `BC_JIT_MERGE_POINT` carries —
    /// matching `warmspot.py:663-665` `jd._green_args_spec` /
    /// `red_args_types` derivation from the `JIT_ENTER_FUNCTYPE`.
    ///
    /// Idempotent: if `with_descriptor` already populated the
    /// descriptor (e.g. consumer-side preconfiguration), this is a
    /// no-op.  Otherwise it replaces the empty stub fallback that
    /// `ensure_descriptor_registered` would install.
    pub fn declare_schema(&mut self, greens: Vec<(&str, Type)>, reds: Vec<(&str, Type)>) {
        if self.descriptor.is_some() {
            return;
        }
        self.descriptor = Some(JitDriverStaticData::new(greens, reds));
        self.descriptor_cache = None;
    }

    /// [`Self::declare_schema`] variant whose green list carries the
    /// upstream lltype subtype directly (`GreenType::Str` /
    /// `GreenType::Unicode` / canonical `Int`/`Ref`/`Float`/`Void`),
    /// mirroring `warmspot.py:663 jd._green_args_spec`.  Used by the
    /// `#[jit_interp]` install path so `JitDriverStaticData::
    /// green_args_spec` reports STR/UNICODE without falling back to
    /// `Ref`.  Reds keep IR `Type` (no `equal_whatever` dispatch on
    /// runtime args).
    pub fn declare_schema_typed(
        &mut self,
        greens: Vec<(&str, GreenType)>,
        reds: Vec<(&str, Type)>,
    ) {
        if self.descriptor.is_some() {
            return;
        }
        self.descriptor = Some(JitDriverStaticData::with_green_types(greens, reds));
        self.descriptor_cache = None;
    }

    /// Identity/cell reds the lowerer lists on `jit_merge_point` after the
    /// named portal reds. `register_dispatch_jitcode` adds these to
    /// `red_kind_counts` so the payload length is exact.
    ///
    /// `(int, ref, float)` from `identity_slot_registers` plus Meta-length
    /// flattened `[int]` cells (`__flat_array_identity_slots`).
    pub fn declare_merge_point_identity_reds(&mut self, counts: (usize, usize, usize)) {
        self.merge_point_identity_reds = counts;
    }

    /// Name the descriptor's virtualizable and state its flat entry contract.
    ///
    /// `warmspot.py make_virtualizable_infos` names the virtualizable
    /// on the jitdriver static data, which is what makes
    /// `compile.py`'s field-reload preamble run at all. A driver whose
    /// entry is a flat state-field prefix cannot say it through the reds — see
    /// [`FlatEntryContract`] — so both halves arrive together here, after
    /// [`Self::declare_schema_typed`] has built the descriptor from the
    /// merge-point payload schema.
    ///
    /// No-op when no descriptor exists yet: a virtualizable declaration with
    /// no green/red schema behind it has nothing to index into.
    pub fn declare_flat_entry_contract(
        &mut self,
        virtualizable: &str,
        contract: FlatEntryContract,
    ) {
        let Some(descriptor) = self.descriptor.as_mut() else {
            return;
        };
        descriptor.virtualizable = Some(virtualizable.to_string());
        descriptor.flat_entry = Some(contract);
        // warmspot.py:538 `jd.index_of_virtualizable` — the flat contract's
        // index, not a position in the reds, because the entry it describes is
        // not the red list.
        descriptor.index_of_virtualizable = contract.index_of_virtualizable as i32;
        let registered_index = descriptor.index;
        self.descriptor_cache = None;
        // Registration files a CLONE of the descriptor, so after it there are
        // two copies and the consumers are split between them: a trace start
        // reads the driver's, while `initialize_virtualizable` reads the
        // registered one. Writing only the first is a silent disagreement
        // about whether this driver has a virtualizable at all.
        //
        // Upstream has no such split to keep in step — `warmspot.py:259`
        // builds the static data and `warmspot.py:537` stamps
        // `jd.index_of_virtualizable` on the very object the table holds — so
        // declaring after registration stays a supported order here too, and
        // the registered copy is brought along rather than left behind.
        if let Some(index) = registered_index
            && let Some(registered) = self.meta.jitdriver_sd_mut(index)
        {
            registered.virtualizable = Some(virtualizable.to_string());
            registered.flat_entry = Some(contract);
            registered.index_of_virtualizable = contract.index_of_virtualizable as i32;
        }
    }

    /// Declare `contract` if — and only if — this driver's virtualizable is one
    /// the compiled entry's field-load preamble can serve. Reports whether it
    /// was declared.
    ///
    /// Declaring the contract is what puts the driver on
    /// `compile.py:508-511`'s preamble: the fields ride through the optimizer
    /// as expanded inputargs and are reloaded from the virtualizable at entry
    /// instead of being handed over one argument per element. That only works
    /// for a virtualizable whose arrays the preamble can reach — see
    /// [`crate::virtualizable::VirtualizableInfo::arrays_are_entry_reloadable`]
    /// — and which array storage a state's fields register as is resolved from
    /// the containers the interpreter author declared, at
    /// `VirtualizableInfo`-build time. So the caller supplies the two numbers
    /// only it can know (the width of its own entry and the virtualizable's
    /// position in it) and the gate is answered here, off the vinfo already
    /// installed on this driver.
    ///
    /// The name comes from the same vinfo (`interp_jit.py:25`
    /// `jitdriver_sd.virtualizable_info.name`) rather than from the caller, so
    /// the descriptor cannot end up naming a virtualizable other than the one
    /// the gate was answered about.
    ///
    /// A driver with no virtualizable info, or one holding an array the
    /// preamble cannot reload, is left exactly as it was: no contract, no name,
    /// and the entry keeps whichever shape it already had.
    pub fn arm_flat_entry_contract(&mut self, contract: FlatEntryContract) -> bool {
        let Some(info) = self.meta.virtualizable_info() else {
            return false;
        };
        if !info.arrays_are_entry_reloadable() {
            return false;
        }
        let name = info.name.clone();
        self.declare_flat_entry_contract(&name, contract);
        true
    }

    /// The entry contract this driver's descriptor carries, if one was armed.
    ///
    /// Reads through the driver rather than the registered static data because
    /// `register_descriptor` mirrors the stamped copy back onto the driver, so
    /// this answer is the same before and after registration.
    pub fn flat_entry_contract(&self) -> Option<FlatEntryContract> {
        self.descriptor.as_ref()?.flat_entry_contract()
    }

    pub fn with_descriptor(threshold: u32, descriptor: JitDriverStaticData) -> Self {
        let mut driver = Self::new(threshold);
        // warmstate.py:564 `_green_args_spec` carries lltype TYPE per
        // green arg, including `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
        // distinct from generic Ptr.  `JitDriverVar.green_type` is the
        // STR/UNICODE-preserving slot populated by
        // `JitDriverStaticData::with_green_types` (`warmspot.py:663`
        // `_green_args_spec` parity).  When the descriptor was built
        // through that constructor (e.g. `#[jit_interp]` install path
        // via `declare_schema_typed`), use the per-var subtype directly;
        // otherwise fall back to `GreenType::from(v.tp)` for the
        // canonical Int/Ref/Float/Void mapping.
        let greens: Vec<GreenType> = descriptor
            .greens()
            .iter()
            .map(|v| v.green_type.unwrap_or_else(|| v.tp.into()))
            .collect();
        driver.descriptor = Some(descriptor);
        driver.entry_points.push(EntryPoint {
            name: "primary".to_string(),
            schema: greens,
        });
        // PyPy's `finish_setup` (`pyjitpl.py`) wires
        // `portal_finishtoken` / `propagate_exc_descr` after the driver
        // is present in `metainterp_sd.jitdrivers_sd`.  Register the
        // pre-built descriptor now so the existing
        // `register_jitdriver_sd` tail call performs that wiring on the
        // actual jitdriver slot, not just on the CPU-global propagate
        // descr.  Later macro-emitted `ensure_descriptor_registered`
        // calls are idempotent because the descriptor now has an index.
        driver.ensure_descriptor_registered();
        driver
    }

    /// Registered slot index of this driver's descriptor, populated by
    /// [`crate::pyjitpl::MetaInterpStaticData::register_jitdriver_sd`].  Mirrors the RPython
    /// attribute `jitdriver_sd.index` (`rpython/jit/codewriter/call.py:46-47`)
    /// read by `jtransform.py:1704` and `compile_tmp_callback`.  Returns `None`
    /// before registration runs.
    pub fn index(&self) -> Option<usize> {
        self.descriptor.as_ref().and_then(|d| d.index)
    }

    /// PyPy `call.py:46-47` `for index, jd in enumerate(jitdrivers_sd):
    /// jd.index = index` parity.  If a consumer constructed this driver
    /// via [`JitDriver::with_descriptor`] (the descriptor carries
    /// `greens`/`reds`/`virtualizable` already), reuse THAT descriptor
    /// when registering — `register_descriptor` was previously called
    /// with a freshly-built empty `JitDriverStaticData::new(vec![],
    /// vec![])`, which clobbered the consumer-provided descriptor's
    /// fields and caused `jitdrivers_sd[idx].num_green_args() == 0` even
    /// when the dispatch JitCode body's `BC_JIT_MERGE_POINT` carried a
    /// real `greens` payload (`pyjitpl.py opimpl_jit_merge_point`
    /// would then mis-validate the green count).
    ///
    /// Idempotent: returns immediately when [`JitDriver::index`] is
    /// already `Some(_)`.  When the consumer did NOT pre-build a
    /// descriptor (`self.descriptor` is `None`), falls back to an empty
    /// stub — pyre-only fail-soft, since PyPy's `jitdrivers_sd` list
    /// would never contain such an entry.
    pub fn ensure_descriptor_registered(&mut self) {
        self.ensure_descriptor_registered_with_index(None);
    }

    fn ensure_descriptor_registered_at(&mut self, index: usize) {
        self.ensure_descriptor_registered_with_index(Some(index));
    }

    fn ensure_descriptor_registered_with_index(&mut self, preinstalled_index: Option<usize>) {
        if self.index().is_some() {
            if let Some(index) = preinstalled_index {
                assert_eq!(
                    self.index(),
                    Some(index),
                    "registered driver descriptor index disagrees with its seeded portal",
                );
            }
            return;
        }
        let jd = self
            .descriptor
            .take()
            .unwrap_or_else(|| JitDriverStaticData::new(vec![], vec![]));
        let _ = self.register_descriptor_with_index(jd, preinstalled_index);
    }

    /// Register `jd` with the embedded `MetaInterpStaticData` and stamp the
    /// returned index onto `self.descriptor`. Mirrors RPython's
    /// `CallControl.__init__` (`call.py`) where `jd.index = idx` is
    /// stamped right after `metainterp_sd.jitdrivers_sd.append(jd)`.
    ///
    /// The macro-emitted `__JitMeta::install_canonical_liveness` invokes
    /// [`JitDriver::ensure_descriptor_registered`] once per consumer
    /// startup, before the dispatch JitCode body is built (which bakes
    /// `jd.index` into `BC_JIT_MERGE_POINT` / `BC_LOOP_HEADER` payloads —
    /// see `jtransform.py, :1716`).
    pub fn register_descriptor(&mut self, jd: JitDriverStaticData) -> usize {
        self.register_descriptor_with_index(jd, None)
    }

    fn register_descriptor_with_index(
        &mut self,
        jd: JitDriverStaticData,
        preinstalled_index: Option<usize>,
    ) -> usize {
        let crate::pyjitpl::MetaInterp {
            staticdata,
            backend,
            ..
        } = &mut self.meta;
        let sd_mut = std::sync::Arc::get_mut(staticdata)
            .expect("MetaInterpStaticData must be uniquely owned at register_descriptor time");
        let idx = match preinstalled_index {
            Some(index) => sd_mut.preinstall_jitdriver_sd(jd, index, backend),
            None => sd_mut.register_jitdriver_sd(jd, backend),
        };
        // Mirror onto self.descriptor so JitDriver::index() returns Some(idx)
        // — A.3.1a's accessor reads through descriptor, not the
        // staticdata Vec, so the stamped clone must live on the driver too.
        self.descriptor = Some(sd_mut.jitdrivers_sd[idx].clone());
        self.descriptor_cache = None;
        // warmspot.py `make_enter_functions`: the driver's `warmstate`
        // exists from registration on.
        self.meta.make_enter_function(idx);
        idx
    }

    /// Get compiled loop metadata for the given green key.
    ///
    /// The `Arc` is the entry's own, not a copy: a caller that needs to hold
    /// the metadata across a run clones the refcount.
    ///
    /// Bound on this driver's WarmEnterState (`JitDriver::index`).
    /// `warmstate.py maybe_compile_and_run` / `JitCell.get_procedure_token`
    /// stay on `jitdriver_sd.warmstate`; the metadata row for that cell uses
    /// the executing driver's index.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&std::sync::Arc<S::Meta>> {
        self.meta
            .get_compiled_meta_on_driver(self.index().unwrap_or(0), green_key)
    }

    /// PyPy warmstate.py get_assembler_token(greenkey).
    /// Returns the JitCellToken for the compiled loop at this green key.
    ///
    /// Bound on this driver's WarmEnterState (`JitDriver::index`).
    pub fn get_loop_token(
        &self,
        green_key: u64,
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        self.meta
            .get_loop_token_on_driver(self.index().unwrap_or(0), green_key)
    }

    /// `warmstate.py bound_reached` finally `cell.flags &= ~JC_TRACING`
    /// on that driver's WarmEnterState.
    pub fn abort_entry_tracing(&mut self, green_key: u64) {
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .clear_tracing_flag(green_key);
    }

    pub fn front_target_inputarg_types(&self, green_key: u64) -> Option<Vec<Type>> {
        self.meta
            .front_target_inputarg_types_on_driver(self.index().unwrap_or(0), green_key)
    }

    /// Attach a GC allocator to the active backend.
    pub fn set_gc_allocator(&mut self, gc: Box<dyn GcAllocator>) {
        self.meta.backend_mut().set_gc_allocator(gc);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "gc_box"))]
thread_local! {
    /// Whether this execution thread has finished `install_jitframe_gc`.
    ///
    /// Set both when a per-thread box was installed and when
    /// `gc_sync::is_initialized` was already true and no box was installed.
    /// This is deliberately execution-context state: the collector owns only
    /// temporary JITFRAMEs, deadframes are thread-confined, and no object
    /// identity or root is duplicated. The native backend's active-GC slot has
    /// the same execution-thread lifetime: dropping one of several JitDrivers
    /// does not clear it. Once the interpreter's values move onto this heap,
    /// its real collector replaces this temporary boundary.
    static JITFRAME_GC_INSTALLED: core::cell::Cell<bool> = const {
        core::cell::Cell::new(false)
    };
}

/// Install the smallest collector the interpreter's compiled tier needs.
///
/// RPython's `gc_ll_descr.malloc_jitframe` allocates every entry frame as a
/// typed GC object. The interpreter's compiled tier constructs no heap object
/// yet, so its collector needs exactly that one type. Installation is once per
/// execution thread because the selected native backend owns its active
/// allocator there. When `gc_sync::is_initialized` is already true, this
/// installs nothing and lets the process collector serve.
#[cfg(all(not(target_arch = "wasm32"), feature = "gc_box"))]
pub fn install_jitframe_gc<S: JitState>(driver: &mut JitDriver<S>) {
    JITFRAME_GC_INSTALLED.with(|installed| {
        if installed.get() {
            return;
        }
        // Backend trampolines prefer a per-thread box over `gc_sync`'s
        // process collector (`store_singleton`). Skip the box when that
        // collector is already installed.
        if !majit_gc::gc_sync::is_initialized() {
            let mut gc = majit_gc::collector::MiniMarkGC::new();
            crate::register_active_backend_jitframe_gc_type(&mut gc);
            driver.set_gc_allocator(Box::new(gc));
        }
        installed.set(true);
    });
}

#[cfg(target_arch = "wasm32")]
pub fn install_jitframe_gc<S: JitState>(_driver: &mut JitDriver<S>) {}

impl<S: JitState> JitDriver<S> {
    /// Route compiled `New` / `NewWithVtable` through the installed GC
    /// allocator instead of the backend's `malloc` stub.
    ///
    /// The backends have carried this setting since it was added for aheui's
    /// nursery-backed nodes, and until now none of them had a caller: it is an
    /// inherent method on each backend struct, and `backend_mut` is private, so
    /// no consumer of `JitDriver` could reach it. Only the dynasm backend acts
    /// on it — cranelift already routes `New` through an installed GC, and the
    /// wasm backend takes it as a no-op — so on those two the setting is
    /// already the behaviour and this call is a statement of intent.
    pub fn set_new_via_gc(&mut self, enabled: bool) {
        self.meta.backend_mut().set_new_via_gc(enabled);
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    /// Frontend supplies the byte offset of the type pointer field, mirroring
    /// RPython's `symbolic.get_field_token(rclass.OBJECT, 'typeptr', ...)`.
    /// `None` matches RPython's gcremovetypeptr behavior (no vtable write).
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.meta.backend_mut().set_vtable_offset(offset);
    }

    /// Byte offset of the class word `bh_new_with_vtable` writes beside the
    /// type word. `None` leaves that word unwritten, the same as
    /// [`Self::set_vtable_offset`].
    pub fn set_w_class_offset(&mut self, offset: Option<usize>) {
        self.meta.backend_mut().set_w_class_offset(offset);
    }

    /// `AbstractLLCPU.subclassrange_min_offset`, beside [`Self::set_vtable_offset`].
    pub fn set_subclassrange_min_offset(&mut self, offset: Option<usize>) {
        majit_backend::set_cpu_subclassrange_min_offset(offset);
        #[cfg(all(
            feature = "dynasm",
            not(feature = "cranelift"),
            not(target_arch = "wasm32")
        ))]
        self.meta.backend_mut().set_subclassrange_min_offset(offset);
    }

    /// PyPy JitDriver(is_recursive=True).
    /// Enables max_unroll_recursion for recursive portal calls.
    pub fn set_is_recursive(&mut self, value: bool) {
        self.is_recursive = value;
    }

    /// PyPy warmspot.py set_param_max_unroll_recursion().
    pub fn set_max_unroll_recursion(&mut self, value: usize) {
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .set_param_max_unroll_recursion(value as u32);
    }

    /// RPython-compatible alias used by warmstate/compile wiring.
    pub fn set_param_max_unroll_recursion(&mut self, value: usize) {
        self.set_max_unroll_recursion(value);
    }

    /// warmspot.py:73 `jd.warmstate.set_param_enable_opts(enable_opts)`.
    ///
    /// Takes the `':'`-separated pass names of `ALL_OPTS`; `"all"` restores the
    /// full set. Dropping `"unroll"` selects the simple-loop compilation path
    /// for this driver alone, which is why this is per-driver rather than a
    /// process-wide switch. `set_param` cannot carry it — that surface is
    /// `i64`-valued, and this parameter is a string upstream too.
    pub fn set_param_enable_opts(&mut self, value: &str) {
        // warmspot.py `jd.warmstate.set_param_enable_opts` is bound on
        // that driver's WarmEnterState (`JitDriver::index`).
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .set_param_enable_opts(value);
    }

    /// Whether this driver was declared recursive.
    pub fn is_recursive(&self) -> bool {
        self.is_recursive
    }

    pub fn with_declarative<D: DeclarativeJitDriver>(
        threshold: u32,
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<Self, &'static str> {
        let descriptor = D::descriptor(green_types, red_types)?;
        Ok(Self::with_descriptor(threshold, descriptor))
    }

    /// Whether the driver is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.meta.is_tracing()
    }

    /// warmstate.py `cell.flags & JC_TRACING` for this green key.
    ///
    /// `maybe_compile_and_run` skips only a cell that is already being
    /// traced in an outer invocation of the same greens. A live session
    /// on another key or another driver does not suppress this cell.
    /// Bound on this driver's WarmEnterState (`JitDriver::index`).
    #[inline]
    pub fn cell_is_tracing(&self, green_key: u64) -> bool {
        self.meta
            .warm_state_ref_for_driver(self.index().unwrap_or(0))
            .and_then(|warm| warm.get_cell(green_key))
            .is_some_and(|cell| cell.is_tracing())
    }

    /// True when this green key has already hit `MAX_TRACE_ABORT_COUNT` and
    /// a later back-edge must not enter `bound_reached`.
    ///
    /// `warmstate.py maybe_compile_and_run` answers this from the cell it
    /// already holds; pyre's u64 door has to look the cell up. A dead
    /// procedure token still belongs to `cleanup_chain`, so it is not
    /// treated as a ban.
    #[inline]
    pub fn cell_is_abort_ceiling_banned(&self, green_key: u64) -> bool {
        self.meta
            .warm_state_ref_for_driver(self.index().unwrap_or(0))
            .and_then(|warm| warm.get_cell(green_key))
            .is_some_and(|cell| {
                let dead_token =
                    cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none();
                !dead_token && cell.abort_count.get() >= crate::warmstate::MAX_TRACE_ABORT_COUNT
            })
    }

    /// Single-pass tracing: take the `(walk_final_pc, walk_final_reds, green banks)`
    /// snapshot captured at a terminal trace transition. CloseLoop and Finish
    /// publish it before draining the ctx; a fresh TraceAction::Abort publishes
    /// the current portal pc so the `__merge` wrapper resumes after its
    /// committed prefix. The banks are `TraceCtx::portal_resume_args`,
    /// captured while the ctx is still live. `None` outside single-pass.
    #[inline]
    pub fn take_single_pass_outcome(
        &mut self,
    ) -> Option<(usize, Vec<crate::Value>, ContinueRunningNormallyArgs)> {
        self.meta.single_pass_outcome.take()
    }

    /// Publish `(pc, empty reds, green banks)` before `compile_loop` or
    /// `take_trace_meta` drains the ctx.
    ///
    /// `warmspot.py ll_portal_runner` rebuilds every portal green from
    /// `jitexc.py ContinueRunningNormally`. The banks are the live
    /// merge-point or header registers (`TraceCtx::portal_resume_args`).
    fn publish_single_pass_outcome(&mut self, pc: usize) {
        let args = self
            .meta
            .trace_ctx()
            .map(|ctx| ctx.portal_resume_args())
            .expect(
                "single-pass outcome needs ContinueRunningNormally banks \
                 (warmspot.py handle_jitexception getattr(e, attrname)[count])",
            );
        self.meta.single_pass_outcome = Some((pc, Vec::new(), args));
    }

    /// Return `ContinueRunningNormally` banks to `portal_resume_scratch`.
    ///
    /// The `#[jit_interp]` expansion calls this after assigning greens so
    /// the next compiled entry reuses the Vec capacities. `ResumeAt` has
    /// no banks (`jtransform.py` `promote_greens` already passed).
    pub fn recycle_portal_resume(&mut self, resume: PortalResume) {
        if let Some(args) = resume.into_args() {
            self.meta.recycle_portal_resume_args(args);
        }
    }

    /// Function-entry maps `PortalResume` to a pc and would otherwise drop
    /// the Vec capacities. Recycle first (`blackhole.py`
    /// `recycle_merge_point_args`).
    fn resume_pc_recycling(&mut self, resume: PortalResume) -> Option<usize> {
        let pc = resume.resume_pc();
        self.recycle_portal_resume(resume);
        pc
    }

    /// `MetaInterp.run_blackhole_interp_to_cancel_tracing` delegates to
    /// `blackhole.convert_and_run_from_pyjitpl`.
    ///
    /// A tracing abort can land anywhere in the portal jitcode, including in
    /// the middle of a source opcode whose remaining effects have not run.  The
    /// walk's own registers name no source position there, so RPython does not
    /// read one: it converts `metainterp.framestack` into a blackhole chain,
    /// runs it forward until the bottommost frame reaches its
    /// `jit_merge_point`, and resumes the interpreter from the greens that
    /// merge point reports through `ContinueRunningNormally`, which
    /// `WarmRunnerDesc.handle_jitexception` catches. Every opcode the walk left half-executed
    /// therefore *finishes* instead of being skipped.
    ///
    /// Returns the resume pc when the conversion ran and reached a merge point,
    /// having already written the blackhole's register banks back into `state`.
    /// `None` leaves the abort on the pre-existing source-pc handoff — the walk
    /// did not abort, or the state shape has no seed path.
    ///
    /// **`None` means "declined before the chain ran", and only that.** Once
    /// `drive_multi_frame_blackhole` returns, the chain has executed the rest
    /// of the half-finished opcodes against the real heap; the caller's
    /// source-pc handoff resumes at a pc dispatch advanced before those
    /// opcodes' arms, so answering `None` there runs them a second time —
    /// the very double-application this conversion exists to prevent. Every
    /// post-chain path therefore returns `Some`, using the `single_pass_finish`
    /// + `usize::MAX` "no forward pc" outcome when it has no merge point to
    ///   resume at. Upstream has no such split:
    /// `blackhole.convert_and_run_from_pyjitpl` never returns to its caller.
    ///
    /// **Seeding.** The walk keeps state fields on the sym; the blackhole reads
    /// them out of the identity registers `StateFieldLayout` names.  The abort
    /// arm captured the sym's image, and this places it into the root frame's
    /// banks before the conversion.  A `[.. ; virt]` array needs two more
    /// things: its elements are pushed into native `state` (they
    /// live on the trace-ctx shadow during the walk, and the chain's
    /// `getarrayitem_vable_*` read the real object), and the virtualizable
    /// identity seeded into its own slot plus handed to the chain alongside the
    /// vinfo pointer `seed_deopt_vinfo_ptr` allows.
    ///
    /// One shape still declines: a flattened fixed `[int]` array (`array_lens`),
    /// whose per-element slots have no sym-side collector.  No frontend
    /// currently declares one — the supported loop-carried form is
    /// `[.. ; virt]`.
    pub fn take_interpret_bail_residual(&mut self) -> Option<String> {
        self.meta.interpret_bail_residual.take()
    }

    pub fn interpret_abort_reason_label(&self) -> Option<&'static str> {
        self.meta
            .last_interpret_abort_reason
            .map(abort_counter_name)
    }

    pub fn run_pending_abort_blackhole(
        &mut self,
        state: &mut S,
        env: &S::Env,
    ) -> Option<PortalResume> {
        self.meta.interpret_bail_residual = None;
        let pending = self.meta.pending_abort_blackhole.take()?;
        let PendingAbortBlackhole {
            framestack,
            scalar_values,
            ref_scalar_values,
            virt_array_values,
            virtualizable_ptr,
            last_exc_value,
            raising_exception,
        } = pending;
        // pyjitpl.py popframe appends to free_frames_list; the standalone
        // walk's pool is that list. Recycle on every exit, including the
        // unseeded-array decline below.
        let mut recycle = crate::pyjitpl::RecycleFramestackOnDrop(framestack);
        let framestack = &mut recycle.0;
        let layout = state.state_field_layout();
        if !layout.array_lens.is_empty() {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[bh] abort-blackhole declined: unseeded state shape (arrays={})",
                    layout.array_lens.len(),
                );
            }
            return None;
        }
        // `writeback_virt_array_state_fields_from_values`, run here rather than
        // by the `jit_merge_point!` hook's own virt-array writeback: the chain
        // is about to re-execute the tail of the aborted opcodes against the
        // live struct, so it has to see the walk's elements, not the
        // trace-start ones.
        if let Some(values) = virt_array_values {
            state.writeback_virt_array_state_fields_from_values(&values);
            self.meta.single_pass_virt_array_values = None;
        }
        // Guard-failure resume still uses `seed_deopt_vinfo_ptr` (null for a
        // heap virtualizable). This drive re-executes the aborted opcode,
        // including vable ops inside an inlined callee. Upstream reads that
        // handle from `fielddescr.get_vinfo()` (`blackhole.py`
        // `bhimpl_getfield_vable_*`), which every interpreter can see.
        // pyre keeps it on the interpreter, so the chain — and any child
        // `interpret_unresolved_inline_call` clones from it — must carry the
        // driver's `virtualizable_info` rather than the null resume seed.
        let vinfo_held = self.meta.virtualizable_info().cloned();
        let vinfo_ptr = vinfo_held
            .as_ref()
            .map_or(std::ptr::null(), std::sync::Arc::as_ptr);
        let root = framestack.frames.first_mut()?;
        if let Some(slot) = layout.vable_identity_ref_slot()
            && slot < root.ref_regs.len()
        {
            root.ref_regs[slot] = Some(majit_ir::OpRef::const_ptr(majit_ir::GcRef(
                virtualizable_ptr as usize,
            )));
        }
        // `scalar_values` is `[int scalars.., float scalars..]` — the order
        // `collect_scalar_state_field_values` builds and
        // `writeback_scalar_state_fields_from_values` consumes.
        for (field_idx, value) in scalar_values.iter().take(layout.num_scalars).enumerate() {
            let slot = layout.scalar_slot(field_idx);
            if slot < root.int_regs.len() {
                root.int_regs[slot] = Some(majit_ir::OpRef::const_int(*value));
            }
        }
        for (field_idx, value) in scalar_values.iter().skip(layout.num_scalars).enumerate() {
            let slot = layout.float_scalar_slot(field_idx);
            if slot < root.float_regs.len() {
                root.float_regs[slot] =
                    Some(majit_ir::OpRef::const_float(f64::from_bits(*value as u64)));
            }
        }
        for (field_idx, value) in ref_scalar_values.iter().enumerate() {
            let slot = layout.ref_scalar_slot(field_idx);
            if slot < root.ref_regs.len() {
                root.ref_regs[slot] =
                    Some(majit_ir::OpRef::const_ptr(majit_ir::GcRef(*value as usize)));
            }
        }

        let mut bh_builder = BackEdgeBhBuilder::lease();
        bh_builder.set_cpu(self.meta_interp().blackhole_cpu());
        let staticdata = &self.meta.staticdata;
        let outcome = drive_multi_frame_blackhole(
            &mut bh_builder,
            framestack,
            layout.clone(),
            vinfo_ptr,
            virtualizable_ptr,
            // PyFrame's operand-stack base; a state-field machine's
            // `[.. ; virt]` array starts at 0 and a heap virtualizable reaches
            // here with a null vinfo, so the base is never read.
            0,
            staticdata,
            last_exc_value,
            raising_exception,
            None,
            None,
            None,
        );
        let MultiFrameBlackholeResult { outcome, terminal } = outcome;
        if crate::majit_log_enabled() {
            eprintln!("[bh] abort-blackhole: convert_and_run → {:?}", outcome);
        }
        // The chain has run: it executed the rest of the half-finished opcodes
        // against the real heap.  The sym image the Abort arm staged predates
        // those effects, so applying it would roll `state` back to before them
        // — but only the terminal image can replace it, so keep it as the
        // fallback when the run produced none.
        if terminal.is_some() {
            self.meta.single_pass_scalar_values = None;
            self.meta.single_pass_ref_scalar_values = None;
            self.meta.single_pass_virt_array_values = None;
        } else if crate::majit_log_enabled() {
            eprintln!(
                "[bh] abort-blackhole: no terminal image — state falls back to \
                 the staged sym scalars"
            );
        }
        // `warmspot.py handle_jitexception` reads the terminal frame's
        // banks the same way the guard-failure resume does
        // (`back_edge_structured`'s CRN arm).
        let writeback = |state: &mut S, resume_pc: usize| {
            let Some(terminal) = terminal.as_ref() else {
                return;
            };
            let (ints, refs, floats) = abort_blackhole_restore_banks(&layout, terminal);
            let meta = S::build_meta(state, resume_pc, env);
            state.restore_banked3(&meta, ints, refs, floats);
            state.recover_after_compiled_run();
        };
        match outcome {
            // The bottommost frame reached its `jit_merge_point`
            // (`blackhole.py:1068-1069`): resume the interpreter at the green
            // pc it reported.  Every `greens` declaration puts `pc` first.
            crate::jitexc::JitException::ContinueRunningNormally(args) => {
                let args = *args;
                // `bhimpl_jit_merge_point` already built the six lists.
                // `warmspot.py handle_jitexception` reads them in place.
                let Some(&resume_pc) = args.green_int.first() else {
                    // Unreachable for any declared jitdriver — every `greens`
                    // declaration puts `pc` first.  Do NOT return `None`: that
                    // is the pre-chain decline answer, and the chain has
                    // already run the aborted opcodes' tails against the real
                    // heap, so the caller's source-pc handoff would execute
                    // them a second time.  Ending the dispatch loop is the
                    // only outcome here that cannot double-apply.
                    debug_assert!(false, "merge point reported no green pc");
                    eprintln!(
                        "[bh] abort-blackhole: ContinueRunningNormally with no green pc \
                         AFTER the chain ran — ending the dispatch loop rather than \
                         replaying the aborted opcodes"
                    );
                    writeback(state, usize::MAX);
                    self.meta.single_pass_finish = true;
                    return Some(PortalResume::ContinueRunningNormally(args));
                };
                // A negative green is not a guest pc. `as usize` wraps it
                // and the dispatch loop resumes in unrelated code. The
                // chain has already run, so end it the same way a missing
                // green does. The other greens still go back: the epilogue
                // reads them after the loop ends.
                let Ok(resume_pc) = usize::try_from(resume_pc) else {
                    debug_assert!(false, "merge point reported a negative green pc");
                    eprintln!(
                        "[bh] abort-blackhole: ContinueRunningNormally green pc {resume_pc} \
                         is not a position — ending the dispatch loop"
                    );
                    writeback(state, usize::MAX);
                    self.meta.single_pass_finish = true;
                    return Some(PortalResume::ContinueRunningNormally(args));
                };
                // `iirrr` portal registers pack greens in front of reds
                // (`next_instr`, `is_being_profiled`, then `pycode`). A
                // state with no scalar identity slots has `int_scalar_base`
                // 0, so the int bank's first word is that green pc.
                // `restore` reads `values[0]` as the red frame.
                // `ContinueRunningNormally` already split the colors
                // (`warmspot.py handle_jitexception`); re-enter with the
                // reds. A state-field machine keeps the identity-slot
                // slice: its reds live past `int_scalar_base`.
                let meta = S::build_meta(state, resume_pc, env);
                let (ints, refs, floats) = crn_restore_banks(&layout, terminal.as_ref(), &args);
                state.restore_banked3(&meta, ints, refs, floats);
                state.recover_after_compiled_run();
                Some(PortalResume::ContinueRunningNormally(args))
            }
            // The portal itself returned inside the blackhole
            // (`blackhole.py _done_with_this_frame`).  There is no forward
            // green pc; the native dispatch loop must exit and run its own
            // post-loop return, which is what `single_pass_finish` tells the
            // `jit_merge_point!` hook to do — it breaks before reading the pc.
            crate::jitexc::JitException::DoneWithThisFrameVoid
            | crate::jitexc::JitException::DoneWithThisFrameInt(_)
            | crate::jitexc::JitException::DoneWithThisFrameRef(_)
            | crate::jitexc::JitException::DoneWithThisFrameFloat(_) => {
                // `usize::MAX` is the same "no forward pc" sentinel the
                // guard-failure resume uses for this outcome.  It reaches
                // `build_meta` only as the `header_pc` the `#[jit_interp]`
                // impl ignores, and the hook breaks before using it as a pc.
                //
                // Publish the scalar the chain returned.  A dispatch JitCode
                // that lowered the source post-loop epilogue has already run it
                // here, so the value IS the portal's result; leaving it unread
                // makes the hook break and native Rust run that suffix a second
                // time.  Void carries no scalar and leaves the latch empty.
                // Ref publishes `Value::Ref`, the same latch Int and Float use.
                match &outcome {
                    crate::jitexc::JitException::DoneWithThisFrameInt(v) => {
                        self.meta.single_pass_finish_values =
                            Some(core::iter::once(Value::Int(*v)).collect());
                    }
                    crate::jitexc::JitException::DoneWithThisFrameFloat(v) => {
                        self.meta.single_pass_finish_values =
                            Some(core::iter::once(Value::Float(*v)).collect());
                    }
                    crate::jitexc::JitException::DoneWithThisFrameRef(v) => {
                        self.meta.single_pass_finish_values =
                            Some(core::iter::once(Value::Ref(*v)).collect());
                    }
                    _ => {}
                }
                writeback(state, usize::MAX);
                self.meta.single_pass_finish = true;
                Some(PortalResume::DoneWithThisFrame(Some(
                    greens_from_blackhole_registers(
                        self.dispatch_jitcode().map(|jc| jc.as_ref()),
                        terminal.as_ref(),
                    ),
                )))
            }
            // `blackhole.py _exit_frame_with_exception` → the exception
            // escaped every converted frame.  Hand it to the interpreter's own
            // machinery.
            //
            // `JitState::deliver_blackhole_exception` defaults to `None` — an
            // interpreter with no exception surface has nothing to hand it to.
            // That answer must NOT become this function's `None`, which means
            // "declined before running anything" and sends the caller to the
            // source-pc handoff: the chain has already executed the aborted
            // opcodes' tails against the real heap, so that handoff would run
            // them twice.  Upstream cannot reach this — `blackhole.py
            // convert_and_run_from_pyjitpl` does not return, it propagates the
            // exception out to `warmspot.py handle_jitexception` — so
            // ending the dispatch loop is the closest non-replaying answer.
            // A frontend that wants the exception implements the hook.
            // The chain stopped without finishing the frame
            // (`JitException::BailToInterpreter`).  The frame's own resume
            // coordinate is stamped, but this driver resumes at a GREEN pc and
            // the bail reports a jitcode one, so there is no pc to hand back.
            // End the dispatch loop for the same reason the no-surface
            // exception arm below does: the chain has already executed the
            // aborted opcodes' tails against the real heap, so the `None`
            // source-pc handoff would run them twice.
            crate::jitexc::JitException::BailToInterpreter => {
                // The terminal image is the nested jitcode's register
                // file (e.g. `new` with 5 ints), not the portal
                // virtualizable (`PyFrame` has 6 scalars). Writing it
                // back panics in `virt_restore_scalars_raw` and would
                // smash the live frame residuals already updated.
                // Leave `state` as the heap left it.
                // Upstream's blackhole never returns. The interpret
                // portal panics with this residual instead of rewinding.
                self.meta.interpret_bail_residual = Some(
                    terminal
                        .as_ref()
                        .and_then(|image| image.bail_residual.clone())
                        .unwrap_or_else(|| "BailToInterpreter".to_owned()),
                );
                self.meta.single_pass_finish = true;
                Some(PortalResume::BailToInterpreter(Some(
                    greens_from_blackhole_registers(
                        self.dispatch_jitcode().map(|jc| jc.as_ref()),
                        terminal.as_ref(),
                    ),
                )))
            }
            crate::jitexc::JitException::ExitFrameWithExceptionRef(exc) => {
                let args = Some(greens_from_blackhole_registers(
                    self.dispatch_jitcode().map(|jc| jc.as_ref()),
                    terminal.as_ref(),
                ));
                let Some(resume_pc) = state.deliver_blackhole_exception(exc) else {
                    eprintln!(
                        "[bh] abort-blackhole: exception escaped every converted frame and \
                         this interpreter has no `deliver_blackhole_exception` — ending the \
                         dispatch loop rather than replaying the aborted opcodes"
                    );
                    writeback(state, usize::MAX);
                    self.meta.single_pass_finish = true;
                    return Some(PortalResume::ExitFrameWithException {
                        pc: usize::MAX,
                        args,
                    });
                };
                writeback(state, resume_pc);
                Some(PortalResume::ExitFrameWithException {
                    pc: resume_pc,
                    args,
                })
            }
        }
    }

    fn clear_single_pass_finish(&mut self) {
        self.meta.single_pass_finish = false;
        self.meta.single_pass_finish_values = None;
    }

    /// `pyjitpl.py finishframe` only raises `DoneWithThisFrame*` with this
    /// frame's `resultbox`. A leftover flag or word from a prior walk is
    /// another call's resultbox.
    fn clear_finish_latches(&mut self) {
        self.clear_single_pass_finish();
        self.meta.back_edge_finish_word = None;
    }

    /// Whether the just-published single-pass outcome came from a terminal
    /// dispatch return (`TraceAction::Finish`) rather than a `CloseLoop`
    /// back-edge. A CloseLoop resumes the native loop at the merge-point green
    /// pc and keeps interpreting; a Finish means the interpreted function has
    /// returned, so the native dispatch loop must EXIT and run its own
    /// post-loop return once. The walk-final pc captured for a Finish is the
    /// interpreter pc past the terminal opcode, which only lands past the
    /// program end (making `while pc < size` exit) when the terminal is the
    /// last opcode — a mid-program terminal would otherwise re-enter the loop.
    /// The macro reads this to `break` instead of resuming at that pc.
    /// Consumes (`take`s) the flag.
    pub fn take_single_pass_finish(&mut self) -> bool {
        let finish = std::mem::replace(&mut self.meta.single_pass_finish, false);
        if finish {
            // The Finish arm also publishes `back_edge_finish_word` for the
            // function-entry Run drain. Consuming the flag without the word
            // leaves that resultbox for the next portal call.
            let _ = self.meta.back_edge_finish_word.take();
        }
        finish
    }

    /// Integer FINISH result produced by a single-pass tracing walk.
    pub fn take_single_pass_finish_int(&mut self) -> Option<i64> {
        let values = self.meta.single_pass_finish_values.take()?;
        match values.first() {
            Some(Value::Int(v)) => Some(*v),
            Some(Value::Float(v)) => Some(v.to_bits() as i64),
            _ => None,
        }
    }

    /// Ref FINISH result produced by a single-pass tracing walk.
    ///
    /// `compile.py DoneWithThisFrameDescrRef`: the address
    /// `warmstate.py execute_assembler` returns for a REF portal.
    pub fn take_single_pass_finish_ref(&mut self) -> Option<usize> {
        let values = self.meta.single_pass_finish_values.take()?;
        match values.first() {
            Some(Value::Ref(v)) => Some(v.as_usize()),
            _ => None,
        }
    }

    /// Float FINISH result produced by a single-pass tracing walk.
    pub fn take_single_pass_finish_float(&mut self) -> Option<f64> {
        let values = self.meta.single_pass_finish_values.take()?;
        match values.first() {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(f64::from_bits(*v as u64)),
            _ => None,
        }
    }

    /// The FINISH arguments of the compiled run a `back_edge*` call just made,
    /// when that run ended in FINISH. `compile.py` marks a FINISH descr
    /// `final_descr = True`: the traced function has RETURNED, so the portal
    /// must return rather than resume the interpreter — upstream's `handle_fail`
    /// raises `jitexc.DoneWithThisFrame*` at this point. The `back_edge*`
    /// signature is `Option<resume_pc>` and cannot say "returned", so the value
    /// travels here. Call immediately after `back_edge*`; consumes (`take`s) it.
    ///
    /// A caller that drains this must return from the portal without using the
    /// `Some(resume_pc)` the same call produced: that pc is the back edge, and
    /// resuming there re-runs the loop the compiled run already completed.
    pub fn take_back_edge_finish(&mut self) -> Option<Vec<Value>> {
        // The int fast path publishes one word. `get_result` for
        // `DoneWithThisFrameDescrInt` is that word, not an exit-value buffer.
        if let Some(value) = self.meta.back_edge_finish_word.take() {
            self.clear_single_pass_finish();
            return Some(vec![Value::Int(value)]);
        }
        // The latch holds the exit values in their decoded storage, which is
        // inline for the widths a finish actually returns. Taken rather than
        // copied: the buffer is owned here and has no reader left, so a width
        // that did spill hands its allocation over instead of duplicating it.
        let values = self.meta.back_edge_finish.take();
        if values.is_some() {
            self.clear_single_pass_finish();
        }
        values.map(smallvec::SmallVec::into_vec)
    }

    /// [`Self::take_back_edge_finish`] projected onto one integer word — the
    /// return shape of an `-> i64` `#[jit_interp]` portal. A `Float` finish
    /// argument is projected by its bits, matching how such a portal spells a
    /// float return (`f64::to_bits() as i64`).
    pub fn take_back_edge_finish_int(&mut self) -> Option<i64> {
        if let Some(value) = self.meta.back_edge_finish_word.take() {
            self.clear_single_pass_finish();
            return Some(value);
        }
        let values = self.meta.back_edge_finish.take()?;
        self.clear_single_pass_finish();
        match values.first() {
            Some(Value::Int(v)) => Some(*v),
            Some(Value::Float(v)) => Some(v.to_bits() as i64),
            _ => None,
        }
    }

    /// [`Self::take_back_edge_finish`] projected onto one ref address — the
    /// return shape of a `*mut T` / `*const T` `#[jit_interp]` portal.
    /// `DoneWithThisFrameDescrRef.get_result` is that address.
    pub fn take_back_edge_finish_ref(&mut self) -> Option<usize> {
        if let Some(value) = self.meta.back_edge_finish_word.take() {
            self.clear_single_pass_finish();
            return Some(value as usize);
        }
        let values = self.meta.back_edge_finish.take()?;
        self.clear_single_pass_finish();
        match values.first() {
            Some(Value::Ref(v)) => Some(v.as_usize()),
            _ => None,
        }
    }

    /// [`Self::take_back_edge_finish`] projected onto one float word — the
    /// return shape of an `-> f64` `#[jit_interp]` portal.
    pub fn take_back_edge_finish_float(&mut self) -> Option<f64> {
        if let Some(value) = self.meta.back_edge_finish_word.take() {
            self.clear_single_pass_finish();
            return Some(f64::from_bits(value as u64));
        }
        let values = self.meta.back_edge_finish.take()?;
        self.clear_single_pass_finish();
        match values.first() {
            Some(Value::Float(v)) => Some(*v),
            Some(Value::Int(v)) => Some(f64::from_bits(*v as u64)),
            _ => None,
        }
    }

    /// Push the walk's scalar state fields into native `state`. The walk
    /// advances a scalar (notably a SEL's new `selected`) on the sym only;
    /// native `state` stays at its trace-start value until this write-back.
    /// The values were captured off the still-live sym inside the CloseLoop
    /// arm (`single_pass_scalar_values`, scalar index order) — the arm clears
    /// the sym before returning, so they cannot be re-read here. Consumes
    /// (`take`s) the stash. No-op when no CloseLoop stashed values (the hot
    /// path) or the state has no scalar fields. Called from the whole-circuit
    /// single-pass `jit_merge_point!` branch, before `recover_after_compiled_run`
    /// so recover refines the storage-derived caches (e.g. stacksize from the
    /// now-current selected) from the written-back scalars.
    #[inline]
    pub fn writeback_scalar_state_fields(&mut self, state: &mut S) {
        if let Some(values) = self.meta.single_pass_scalar_values.take() {
            state.writeback_scalar_state_fields_from_values(&values);
        }
    }

    /// Push the walk's ref state fields into native `state`.
    ///
    /// The ref twin of [`Self::writeback_scalar_state_fields`], and its
    /// consumer is the one the twin does not have: a resume whose native
    /// `state` was left behind by a compiled run rather than kept current by
    /// the interpreter. Consumes (`take`s) the stash; no-op when the close
    /// staged none or the state declares no ref fields.
    #[inline]
    pub fn writeback_ref_scalar_state_fields(&mut self, state: &mut S) {
        let Some(values) = self.meta.single_pass_ref_scalar_values.take() else {
            return;
        };
        for (index, value) in values.into_iter().enumerate() {
            state.writeback_live_ref_scalar_state_field(index, value);
        }
    }

    /// Push the walk's loop-carried virtualizable-array element values into
    /// native `state`, the array analog of `writeback_scalar_state_fields`. The
    /// walk mutates the array on the trace-ctx shadow; native `state`'s array is
    /// stale at the close because `synchronize_virtualizable` skips the
    /// write-back when `VirtualizableInfo::outer_executor_owns_state` is set —
    /// the observer/replay merge-point form, where an outer executor owns the
    /// live struct. Without this, the compiled-loop seed
    /// (`extract_live_values` reads native `state`) reflects the trace-start
    /// array, so the loop re-executes the peeled iteration — double-firing any
    /// side-effecting residual. Consumes (`take`s) the stash. No-op when the
    /// state has no virtualizable array. Called from the whole-circuit
    /// single-pass `jit_merge_point!` branch, after `writeback_scalar_state_fields`
    /// and before `recover_after_compiled_run` / `try_resume_into_compiled_loop`.
    #[inline]
    pub fn writeback_virt_array_state_fields(&mut self, state: &mut S) {
        if let Some(values) = self.meta.single_pass_virt_array_values.take() {
            state.writeback_virt_array_state_fields_from_values(&values);
        }
    }

    pub fn can_resume_into_compiled_loop_at_label(&self) -> bool {
        let Some(key) = self.meta.single_pass_compiled_key else {
            if crate::callee_rca_enabled() {
                eprintln!("[callee-rca][direct-entry-can-resume] key=None -> false");
            }
            return false;
        };
        let jd_no = self.index().unwrap_or(0);
        let has_compiled = self.has_compiled_loop(key);
        let loop_header_pc = self.meta.loop_header_pc_for_driver(jd_no, key);
        let dispatch_key = self.meta.front_target_dispatch_key_on_driver(jd_no, key);
        let direct_entry = dispatch_key.is_some_and(|dispatch_key| {
            self.meta
                .entry_procedure_token_on_driver(self.index().unwrap_or(0), key)
                .is_some_and(|token| {
                    self.meta
                        .backend_supports_dispatch_key_entry_for(&token, dispatch_key)
                })
        });
        let ok = has_compiled && loop_header_pc != Some(0) && direct_entry;
        if crate::callee_rca_enabled() {
            eprintln!(
                "[callee-rca][direct-entry-can-resume] key={key} has_compiled={} \
                 loop_header_pc={:?} dispatch_key={:?} -> {}",
                has_compiled, loop_header_pc, dispatch_key, ok,
            );
        }
        ok
    }

    pub fn discard_single_pass_resume(&mut self) {
        self.meta.single_pass_compiled_key = None;
        self.meta.single_pass_scalar_values = None;
        self.meta.single_pass_ref_scalar_values = None;
        self.meta.single_pass_virt_array_values = None;
        self.meta.single_pass_compact_label_values = None;
        self.meta.single_pass_full_live_values = None;
        self.continue_running_normally_payload = None;
    }

    pub fn log_single_pass_parity_resume(&self, resume_pc: usize, state: &S, env: &S::Env) {
        if !portal_rca_enabled() {
            return;
        }
        if resume_pc == usize::MAX {
            // The finish sentinel: the interpreted function returned, so there
            // is no resume coordinate to build a state image against.
            eprintln!("[portal-rca][parity-crn] resume_pc=<finish>");
            return;
        }
        let key = self.meta.single_pass_compiled_key;
        let dispatch_key = key.and_then(|k| {
            self.meta
                .front_target_dispatch_key_on_driver(self.index().unwrap_or(0), k)
        });
        let meta = S::build_meta(state, resume_pc, env);
        eprintln!(
            "[portal-rca][parity-crn] resume_pc={resume_pc} compiled_key={key:?} \
             label_dispatch_key={dispatch_key:?}"
        );
        if let Some(dump) = state.debug_state_fields(&meta) {
            eprintln!("[portal-rca][parity-crn-state]\n{dump}");
        }
    }

    pub fn arm_single_pass_label_entry_on_next_back_edge(&mut self, state: &S) {
        if state.state_field_layout().total_live_values() == 0 {
            if portal_rca_enabled() {
                eprintln!(
                    "[portal-rca][parity-crn-arm-label-entry] state_fields=0 -> byte-identical discard"
                );
            }
            return;
        }
        // `raise_continue_running_normally` returns to the interpreter.
        // The next entry is `maybe_compile_and_run` → `execute_assembler`
        // on the procedure token, not a direct body-LABEL dispatch.
        let _ = state;
    }

    /// Body-LABEL dispatch is not a metainterp exit. `execute_assembler`
    /// enters the procedure token.
    fn take_single_pass_label_entry_dispatch_key_for_back_edge(
        &mut self,
        _green_key: u64,
    ) -> Option<u32> {
        None
    }

    /// Single-pass cross-loop-cut resume: directly enter the loop the
    /// CloseLoop arm just compiled (its key stashed in
    /// `single_pass_compiled_key`) with the walk-final native state, instead
    /// of re-interpreting the walked body. This is a direct TargetToken
    /// entry, so it must skip the peeled preamble and enter at the loop LABEL:
    /// PyPy's `raise_continue_running_normally` does not execute the freshly
    /// compiled token with the post-walk state, and later assembler entry lands
    /// at the TargetToken body address. Cranelift models that body address with
    /// dispatch key `label_block_id + 1`; key 0 is the peeled-preamble path and
    /// would replay side-effecting preamble stores.
    pub fn try_resume_into_compiled_loop(
        &mut self,
        resume_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> Option<usize> {
        let dbg = crate::closedbg_enabled();
        let key = match self.meta.single_pass_compiled_key.take() {
            Some(k) => k,
            None => {
                if dbg {
                    eprintln!("@@@DE try_resume: single_pass_compiled_key=None");
                }
                return None;
            }
        };
        let has = self.has_compiled_loop(key);
        if dbg {
            eprintln!("@@@DE try_resume key={key} has_compiled={has} resume_pc={resume_pc}");
        }
        if !has {
            return None;
        }
        let jd_no = self.index().unwrap_or(0);
        if self.meta.loop_header_pc_for_driver(jd_no, key) == Some(0) {
            return None;
        }
        let dispatch_key = self.meta.front_target_dispatch_key_on_driver(jd_no, key)?;
        let Some(token) = self
            .meta
            .entry_procedure_token_on_driver(self.index().unwrap_or(0), key)
        else {
            return None;
        };
        if !self
            .meta
            .backend_supports_dispatch_key_entry_for(&token, dispatch_key)
        {
            return None;
        }
        let full_live_values = self
            .take_continue_running_normally_payload()
            .map(|(values, _)| values);
        let compact_label_values = match self.meta.single_pass_compact_label_values.take() {
            Some((values_key, values)) if values_key == key => {
                if crate::callee_rca_enabled() {
                    eprintln!(
                        "[callee-rca][direct-entry-pack] key={key} dispatch_key={dispatch_key} \
                         source=compact-label len={}",
                        values.len()
                    );
                }
                Some(values)
            }
            Some((values_key, values)) => {
                if crate::callee_rca_enabled() {
                    eprintln!(
                        "[callee-rca][direct-entry-pack] key={key} dispatch_key={dispatch_key} \
                         stale_compact_key={values_key} stale_len={} -> decline",
                        values.len()
                    );
                }
                None
            }
            None => None,
        };
        // Only the compact values are already in the LABEL's argument order. The
        // walk's full live values are in state-field order, so they get mapped
        // here; `back_edge_internal` takes whatever it is handed as final.
        let direct_live_values = match compact_label_values {
            Some(values) => Some(values),
            None => match full_live_values {
                Some(values) => {
                    let packed = self.meta.pack_front_target_live_values_on_driver(
                        self.index().unwrap_or(0),
                        key,
                        &values,
                    );
                    if crate::callee_rca_enabled() {
                        match packed.as_ref() {
                            Some(packed) => eprintln!(
                                "[callee-rca][direct-entry-pack] key={key} \
                                 dispatch_key={dispatch_key} \
                                 source=full-live-packed full_len={} compact_len={}",
                                values.len(),
                                packed.len()
                            ),
                            None => eprintln!(
                                "[callee-rca][direct-entry-pack] key={key} \
                                 dispatch_key={dispatch_key} full_len={} \
                                 source=full-live-packed -> decline",
                                values.len()
                            ),
                        }
                    }
                    packed
                }
                None => None,
            },
        };
        let r = self
            .back_edge_internal(
                key,
                None,
                None,
                resume_pc,
                state,
                env,
                Some(dispatch_key),
                direct_live_values,
                || {},
            )
            .and_then(|resume| self.resume_pc_recycling(resume));
        if dbg {
            eprintln!("@@@DE back_edge_internal -> {r:?}");
        }
        r
    }

    /// RPython JC_TRACING parity: true only when tracing this specific
    /// key. `target_raw` is the structured `(code_ptr, pc)` greenkey
    /// matching pyjitpl.py:1396-1401's element-wise comparison.
    #[inline]
    pub fn is_tracing_key(&self, target_raw: (usize, usize)) -> bool {
        self.meta.is_tracing_key(target_raw)
    }

    /// Whether the driver is currently tracing a bridge.
    ///
    /// This reports the `self.resumekey` CLASS (pyjitpl.py:2905 / 2939), which
    /// now survives a declined bridge attempt. Both consumers want a live
    /// session rather than the bare class — `close_loop_args_at` reads it
    /// mid-walk to decide whose LABEL types the JUMP must match, and
    /// `eval_with_jit_inner` reads it to force concrete force-helper calls onto
    /// the plain interpreter — but `self.sym` is not the liveness signal:
    /// `start_bridge_tracing` sets `bridge_info` (through
    /// `start_retrace_from_guard`) and then runs `rebuild_from_resumedata`
    /// BEFORE `self.sym = Some(sym)`, so the protection window opens while
    /// `sym` is still `None`. Narrowing this to a live session needs a signal
    /// that spans the whole window, so it is left as the class for now.
    #[inline]
    pub fn is_bridge_tracing(&self) -> bool {
        self.meta.bridge_info().is_some()
    }

    /// The green key of the active trace, if any.
    pub fn current_trace_green_key(&mut self) -> Option<u64> {
        self.meta.trace_ctx().map(|ctx| ctx.green_key())
    }

    /// pyjitpl.py compile_loop + raise_if_successful bookkeeping,
    /// shared by the `CloseLoop` and `CloseLoopWithArgs` arms.  Captures the
    /// `(green_key, header_pc)` BEFORE `compile_loop` drains the trace ctx (so a
    /// later bridge into this loop can resolve its parent loop's header pc — its
    /// JUMP close target), compiles, and on success records the loop-header pc
    /// and stashes the just-compiled key for the single-pass direct-entry hook.
    fn compile_and_record_loop(
        &mut self,
        jump_args: &[OpRef],
        meta: S::Meta,
    ) -> crate::CompileOutcome {
        let loop_header_pc = self.meta.trace_ctx().map(|c| c.header_pc);
        // The merge-point greens this loop closed on — the vocabulary a later
        // bridge closing at the same merge point will present
        // (pyjitpl.py). Captured before `compile_loop` drains the ctx.
        let loop_close_greens = self
            .meta
            .trace_ctx()
            .and_then(|c| c.close_greens.clone().or_else(|| c.header_greens.clone()));
        let loop_green_key = self.current_trace_green_key();
        if crate::closedbg_enabled() {
            eprintln!(
                "@@@CLOSE LOOP-COMPILE green_key={} header_pc={} jump_args={}",
                loop_green_key.map(|k| k as i64).unwrap_or(-1),
                loop_header_pc.map(|p| p as i64).unwrap_or(-1),
                jump_args.len()
            );
        }
        let outcome = self.meta.compile_loop(jump_args, meta);
        if crate::closedbg_enabled() {
            eprintln!(
                "@@@CLOSE LOOP-COMPILE outcome={:?}",
                match &outcome {
                    crate::CompileOutcome::Compiled { .. } => "Compiled",
                    crate::CompileOutcome::Cancelled => "Cancelled",
                    crate::CompileOutcome::Aborted => "Aborted",
                    crate::CompileOutcome::SwitchToBlackhole { .. } => "SwitchToBlackhole",
                }
            );
        }
        if matches!(outcome, crate::CompileOutcome::Compiled { .. }) {
            if let (Some(gk), Some(hp)) = (loop_green_key, loop_header_pc) {
                self.meta.record_loop_header_pc(gk, hp);
            }
            if let Some(k) = self.meta.last_compiled_key() {
                if let Some(hp) = loop_header_pc {
                    self.meta.record_loop_header_pc(k, hp);
                }
                // Key the loop by the greens it closed on, so a bridge reaching
                // this merge point resolves THIS loop's procedure token. Only
                // `last_compiled_key` is registered: on a cross-loop cut that
                // is the inner loop the close point belongs to, whereas
                // `loop_green_key` is the outer trace-start key.
                if let Some(greens) = loop_close_greens {
                    if crate::closedbg_enabled() {
                        eprintln!("@@@CLOSE LOOP-GREENS key={} greens={greens:?}", k as i64);
                    }
                    self.meta.record_loop_header_greens(k, greens);
                } else if crate::closedbg_enabled() {
                    eprintln!("@@@CLOSE LOOP-GREENS key={} greens=NONE", k as i64);
                }
                // Stash the just-compiled loop's key so the merge-point hook
                // can arm the CRN/lazy back-edge label entry with the
                // walk-final state instead of replaying the walked peeled
                // preamble (cross-loop-cut: this is cut_inner_green_key).
                self.meta.single_pass_compiled_key = Some(k);
            }
        }
        outcome
    }

    /// ResumeFromInterpDescr parity: data needed to compile an entry bridge
    /// from the currently active trace.
    ///
    /// An entry bridge starts with the unoptimized interpreter arguments and
    /// closes with a JUMP into an already-compiled loop.  The key it attaches
    /// to is `original_boxes[:num_green_args]` — the greens of the merge point
    /// tracing STARTED at (`_compile_and_run_once`, pyjitpl.py), with
    /// no function-entry versus loop-header distinction; whether the trace came
    /// from the interpreter at all is already decided by the absent bridge
    /// origin at the caller.
    pub fn compile_trace_entry_data(&mut self) -> Option<(u64, S::Meta)> {
        let original_green_key = self.meta.trace_ctx().map(|ctx| ctx.root_green_key())?;
        let trace_meta = self.meta.trace_meta()?.clone();
        Some((original_green_key, trace_meta))
    }

    /// pyjitpl.py:1641-1663, the loop arm of
    /// `_create_segmented_trace_and_blackhole`:
    ///
    /// ```python
    /// target_token = compile.compile_simple_loop(
    ///     metainterp, greenkey, metainterp.history.trace,
    ///     fake_runtime_boxes, enable_opts, cut_at, patch_jumpop_at_end=False)
    /// jd_sd.warmstate.attach_procedure_to_interp(
    ///     greenkey, target_token.targeting_jitcell_token)
    /// ```
    ///
    /// Split out of the `TraceAction::SegmentedLoop` arm so the arm can host
    /// the bridge else-arm (pyjitpl.py) beside it without the two
    /// compiles pushing the shared handoff and blackhole tail apart.
    fn compile_segmented_loop(&mut self, meta: S::Meta) {
        let Some(green_key) = self.meta.compile_simple_loop(meta) else {
            return;
        };
        // pyjitpl.py:2760 reads the greenkey from current_merge_points;
        // without staging the key here, the segmented-loop abort hook
        // reports key 0.
        self.meta.pending_abort_green_key = Some(green_key);
        // pyjitpl.py:1662-1663 line-by-line: pass the compiled
        // `target_token` returned by `compile_simple_loop` to
        // `attach_procedure_to_interp`, NOT a fresh synthetic uncompiled
        // JitCellToken.  pyre's `compile_simple_loop` returns the
        // green_key; the actual `Arc<JitCellToken>` is the `token` field
        // of the freshly-installed `CompiledEntry` in `self.compiled_loops`.
        // `compile_simple_loop` inserted the entry under the compiling
        // driver (`self.jitdriver_sd`); read it back from the same one.
        let jd_no = self.meta.active_jitdriver_sd.unwrap_or(0);
        let install_token = self
            .meta
            .compiled_entry_on_driver(jd_no, green_key)
            .expect(
                "compile_simple_loop returned Some(green_key) ⇒ \
                 compiled_loops has the new CompiledEntry",
            )
            .live_token();
        // `warmstate.py attach_procedure_to_interp` redirect+record_jump_to chain routed
        // through MetaInterp's caller-side helper.  Skip the redirect when
        // MemoryManager has already evicted the freshly-installed token
        // (rare; eviction is independent of this insertion path).
        if let Some(install_token) = install_token {
            self.meta
                .attach_procedure_with_redirect_on_driver(jd_no, green_key, install_token);
        }
    }

    /// RPython rlib.jit.current_trace_length().
    /// Returns the number of ops in the active trace, or -1 if not tracing.
    pub fn current_trace_length(&mut self) -> i64 {
        match self.meta.trace_ctx() {
            Some(ctx) => ctx.num_ops() as i64,
            None => -1,
        }
    }

    /// Abort the active trace and clear driver-owned symbolic state.
    pub fn abort_current_trace(&mut self, permanent: bool) {
        if self.meta.is_tracing() {
            self.meta.abort_trace(permanent);
            self.sym = None;
            self.meta.clear_trace_session();
            self.compile_trace_success = false;
            self.continue_running_normally_payload = None;
        }
    }

    /// reached_loop_header() called compile_trace() successfully and the
    /// driver should switch to compiled code after the merge-point closure
    /// returns.
    ///
    /// TODO: substitutes for `ContinueRunningNormally`
    /// stack unwind (`pyjitpl.py` `raise_if_successful`). RPython
    /// raises across the interpreter to break out of trace recording; the
    /// Rust dispatch loop cannot raise across the FFI boundary into
    /// `merge_point_keyed`, so a one-shot `bool` consumed by
    /// `take_compile_trace_success` substitutes. Same semantics: stop
    /// tracing immediately, switch into compiled code on next iteration.
    pub fn note_compile_trace_success(&mut self) {
        crate::debug::log_one("jit-summary", "compile-trace: note success");
        self.compile_trace_success = true;
        self.continue_running_normally_payload = None;
    }

    fn note_continue_running_normally(
        &mut self,
        values: Option<Vec<Value>>,
        restart_pc: Option<usize>,
    ) {
        self.compile_trace_success = true;
        self.continue_running_normally_payload = values.map(|values| (values, restart_pc));
    }

    pub fn compile_trace_success_pending(&self) -> bool {
        self.compile_trace_success
    }

    fn take_compile_trace_success(&mut self) -> bool {
        std::mem::take(&mut self.compile_trace_success)
    }

    fn take_continue_running_normally_payload(&mut self) -> Option<(Vec<Value>, Option<usize>)> {
        self.continue_running_normally_payload.take()
    }

    /// Tracing hook — call at the top of the dispatch loop.
    ///
    /// If tracing is active, calls `trace_fn` with the active [`MetaInterp`]
    /// (via which the closure can obtain the active [`crate::trace_ctx::TraceCtx`]) and
    /// symbolic state, then handles the result automatically:
    ///
    /// - `CloseLoop` → validates depths, collects jump args, compiles
    /// - `Finish` → compiles a terminal trace ending in `FINISH`
    /// - `Abort` → aborts trace (may retry later)
    /// - `AbortPermanent` → `aborted_tracing`; does not set `JC_DONT_TRACE_HERE`
    /// - `Continue` → no action
    ///
    /// # Example
    ///
    /// ```ignore
    /// driver.merge_point(|meta, sym| {
    ///     let ctx = meta.trace_ctx().expect("merge_point invariant");
    ///     trace_instruction(ctx, sym, program, pc, &state)
    /// });
    /// ```
    #[inline]
    pub fn merge_point<F>(&mut self, mut trace_fn: F)
    where
        F: FnMut(&mut MetaInterp<S::Meta>, &mut S::Sym) -> TraceAction,
    {
        if !self.meta.is_tracing() {
            return;
        }
        // A stash still here belongs to an earlier abort whose consumer never
        // ran (a `jit_merge_point!` expansion without a `state` argument, or a
        // non-macro driver).  It names frames from a finished walk, so drop it
        // rather than let this merge point's handoff read it.
        if let Some(pending) = self.meta.pending_abort_blackhole.take() {
            crate::pyjitpl::recycle_framestack(pending.framestack);
        }
        if self.sym.is_none() || self.meta.trace_meta().is_none() {
            crate::debug::log_one("jit-abort", "mp: abort:sym_none");
            self.meta.abort_trace(false);
            self.sym = None;
            self.meta.clear_trace_session();
            return;
        }

        // pyjitpl.py: when `reached_loop_header` returns WITHOUT
        // raising — the `current_merge_points.append` path at :3058-3060 — the
        // metainterp restores its pc and keeps executing the guest under
        // recording:
        //
        // ```python
        // self.pc = saved_pc
        // ...
        // # and the same MIFrame goes on with `opcode = ...; dispatch`
        // ```
        //
        // No guest instruction runs outside the recorder while the trace is
        // live. This driver's caller works the other way round: a merge point
        // that returns hands control back to the NATIVE dispatch loop, which
        // executes the instruction the merge point sits on
        // (`#pc = __sp_pc; continue;` in the `jit_merge_point!` expansion).
        // For every other outcome that is correct — the trace has ended. For
        // the append path it would execute one guest instruction with no IR op
        // recorded for it, and the next walk segment would resume a whole
        // instruction later with the sym still holding the previous
        // iteration's boxes. So the append path re-enters the walk HERE
        // instead of returning, which is what makes it upstream's "keep
        // interpreting" rather than "hand the guest back".
        //
        // Self-limiting the way upstream's scan is: the flag is set only when
        // the scan found NO same-greenkey merge point, and the same call
        // registers one, so the next visit takes the retrace/abort branch.
        loop {
            // Phase 1: split-borrow self into meta and sym, run closure.
            // self.meta and self.sym are disjoint fields → standard split-borrow.
            let action = {
                let Some(sym) = self.sym.as_mut() else {
                    crate::debug::log_one("jit-abort", "mp: abort:sym_none2");
                    self.meta.abort_trace(false);
                    self.meta.clear_trace_session();
                    return;
                };
                trace_fn(&mut self.meta, sym)
            }; // borrows dropped here

            // pyjitpl.py:1617-1620 places the `force_finish_trace` segmenting
            // check INSIDE the tracing loop, in `MIFrame.debug_merge_point`, so
            // a JC_FORCE_FINISH key segments before the 1.0x too-long check can
            // abort it.  Pyre's counterpart lives at the same place, in the
            // production tracer's `jit_merge_point` arm
            // (`jitcode_dispatch/mod.rs`, whose `create_segmented_trace` the
            // walk returns `DispatchOutcome::SegmentTrace` from); `trace.rs`
            // maps it onto the `TraceAction::SegmentedLoop` /
            // `SegmentedBridge` arm below.  `pyjitpl/dispatch.rs` carries a
            // second copy on `MIFrame::run_one_step`, which production tracing
            // does not go through.  A check here instead would be unreachable:
            // the walk returns `Continue` only once its framestack drains, and
            // an over-budget walk returns `Abort` from inside its own loop.

            // Phase 2: handle trace result with full access to self
            match action {
                TraceAction::Continue => {}
                TraceAction::CompileTrace => {
                    // Successful compile-into-existing-target: raise_if_successful()
                    // raises ContinueRunningNormally (pyjitpl.py), which
                    // bypasses the `except SwitchToBlackhole` handler, so only the
                    // successful live history teardown runs — `aborted_tracing`
                    // accounting is NOT reached on success (the loop/bridge counter
                    // was already bumped at backend-compile time).
                    self.meta.finish_trace_live();
                    // No aborted_tracing follows on success, so drop the
                    // pending_abort_* payload finish_trace_live staged (else it
                    // attaches this key to a later, unrelated abort's hook).
                    self.meta.clear_pending_abort();
                    self.sym = None;
                    self.meta.clear_trace_session();
                    self.compile_trace_success = true;
                    self.continue_running_normally_payload = None;
                    return;
                }
                TraceAction::CloseLoop => {
                    // Snapshot the walk-final (pc, reds) off the active TraceCtx
                    // BEFORE `compile_loop` (below) drains it, stashing onto the
                    // MetaInterp so the `__merge` wrapper can read it after the
                    // trace closes. `None` whenever the walk did not populate the
                    // reds.
                    //
                    // Resume pc is the walk-final green pc the dispatch stashed
                    // (an interpreter program pc, NOT the JitCode op cursor). The
                    // reds vector published here is INTENTIONALLY empty. The
                    // merge-point hook transfers walk-final loop state without it:
                    // scalar state fields (e.g. selected) are captured here from
                    // the still-live sym (`single_pass_scalar_values`) and written
                    // into native `state` by the macro hook after the walk closes,
                    // then `recover` re-derives the storage-backed cache fields
                    // (stacksize, storage refs) from the walk-advanced shared
                    // storage. `dispatch` does populate `ctx.walk_final_reds` from
                    // the merge point's red-operand slots, but that register-bank
                    // snapshot is NOT a valid `restore_values` source for the
                    // state-field model — those slots are type-grouped operand
                    // order, not `collect_jump_args` order, and include reds (e.g.
                    // floats) with no state-field target, so restoring them would
                    // write the wrong values. A JIT whose merge point carries real
                    // red operands would instead source reds here, but none that
                    // close exist today.
                    let pc = self.meta.trace_ctx().and_then(|ctx| ctx.walk_final_pc);
                    if let Some(p) = pc {
                        self.publish_single_pass_outcome(p);
                    }
                    // Capture the walk-final scalar state-field values off the
                    // still-live sym in scalar index order (idx `0..num_scalars`),
                    // BEFORE `self.sym = None` at the end of this arm drops them.
                    // native `state` is not in scope here (only in the
                    // `jit_merge_point!` macro expansion), so stash the values on
                    // the MetaInterp for the macro hook to apply after the close.
                    // Empty when the state has no scalar fields.
                    if let Some(scalars) = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.close_scalar_values.take())
                    {
                        self.meta.single_pass_scalar_values = Some(scalars);
                    } else if let Some(sym) = self.sym.as_ref() {
                        let scalars = S::collect_scalar_state_field_values(sym);
                        if !scalars.is_empty() {
                            self.meta.single_pass_scalar_values = Some(scalars);
                        }
                    }
                    if let Some(ref_scalars) = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.close_ref_scalar_values.take())
                    {
                        self.meta.single_pass_ref_scalar_values = Some(ref_scalars);
                    } else if let Some(sym) = self.sym.as_ref() {
                        let ref_scalars = S::collect_ref_scalar_state_field_values(sym);
                        if !ref_scalars.is_empty() {
                            self.meta.single_pass_ref_scalar_values = Some(ref_scalars);
                        }
                    }
                    // Capture the walk-final loop-carried virt-array element values
                    // off the still-live trace ctx (the walk mutated the ctx shadow,
                    // not native `state`), before `compile_loop` drains the ctx. The
                    // macro hook writes them into native `state` so the compiled loop
                    // resumes at S_{k+1} instead of re-executing the peeled iteration
                    // (pyjitpl.py live_arg_boxes += virtualizable_boxes).
                    // `None` when the state has no virtualizable array.
                    let virt_elems = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.collect_virtualizable_element_values());
                    if let Some(elems) = virt_elems {
                        self.meta.single_pass_virt_array_values = Some(elems);
                    }
                    // pyjitpl.py reached_loop_header: `live_arg_boxes` is built ONCE, here at
                    // the top of `reached_loop_header`, and every consumer below
                    // reads that same list — the bridge's closing JUMP
                    // (line 2988 `compile_trace`), the merge-point scan
                    // (line 3019), and `current_merge_points.append` (line 3059).
                    //
                    // `remove_consts_and_duplicates` RECORDS a `SAME_AS` op for
                    // every constant or duplicate it rewrites, so normalizing once
                    // per consumer appends a duplicate set of them. On the path
                    // where a bridge close falls through to the loop close
                    // (`RetraceNeeded` / `Declined`) that second set lands after
                    // `compile_trace` sampled `potential_retrace_position`, moving
                    // every later trace position off by the number of rewrites.
                    // `reached_loop_header` (pyjitpl.py) runs on a live
                    // MIFrame and builds `live_arg_boxes` from `greenboxes +
                    // redboxes`; upstream has no "no symbolic state" arm, so the
                    // `None` fallback below stands in for an invariant rather than
                    // for a real state. It holds by construction: `merge_point`
                    // returns early when `self.sym.is_none()`, its walk loop head
                    // re-establishes `Some` on every iteration, and `trace_fn`
                    // receives `&mut S::Sym` — never `&mut self` — so nothing
                    // between the loop head and here can write the `Option`.
                    debug_assert!(
                        self.sym.is_some(),
                        "reached_loop_header: sym must be live on the CloseLoop arm",
                    );
                    // `reached_loop_header` built this list from the marker
                    // operands (`prepare_list_of_boxes`) and already ran
                    // `remove_consts_and_duplicates`. Both close routes
                    // (`opimpl_jit_merge_point` and header-revisit
                    // `close_header_revisit`) stash it on `close_jump_boxes`.
                    let live_arg_boxes: Vec<OpRef> = self
                        .meta
                        .trace_ctx()
                        .expect("reached_loop_header: CloseLoop requires a live TraceCtx")
                        .close_jump_boxes
                        .take()
                        .expect(
                            "reached_loop_header: close_jump_boxes must be built at this \
                             merge point (`opimpl_jit_merge_point` / `close_header_revisit`)",
                        )
                        .into_iter()
                        .map(|(o, _)| o)
                        .collect();
                    // pyjitpl.py reached_loop_header:
                    //
                    //     if not self.partial_trace:
                    //         ptoken = self.get_procedure_token(greenboxes)
                    //         if has_compiled_targets(ptoken):
                    //             self.compile_trace(live_arg_boxes, ptoken)
                    //     # raises in case it works
                    //
                    // The walker published the token key it reached (`close_jump_into_key`); this is
                    // the `compile_trace` call, and its success ends the trace exactly the way
                    // `raise_if_successful` (pyjitpl.py) does.
                    // Set when this visit already ran `compile_trace`, so the
                    // bridge-origin block below does not run it a second time.
                    // `reached_loop_header` calls `compile_trace` once per header visit.
                    let mut compile_trace_attempted = false;
                    if let Some(target_key) = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.take_close_jump_into_key())
                    {
                        let mut result = crate::pyjitpl::BridgeCompileResult::Declined;
                        // Whether the close was actually evaluated.  Both conditions below are
                        // transient — `partial_trace` clears with the retrace it belongs to, and a
                        // key acquires compiled targets the moment its loop compiles — and
                        // upstream re-tests them at every later visit of the same header
                        // (pyjitpl.py:3001-3007 runs once per visit, with no memory of the last).
                        // Only an attempt that ran and did not compile may be latched below.
                        let mut attempted = false;
                        let mut continue_running_normally_values = None;
                        // pyjitpl.py `if not self.partial_trace:` — a retrace must not bridge
                        // into the very loop it exists to respecialize.
                        if self.meta.partial_trace().is_none()
                            && self.meta.has_compiled_targets(target_key)
                        {
                            compile_trace_attempted = true;
                            attempted = true;
                            continue_running_normally_values = {
                                let trace_meta = self.meta.trace_meta().cloned();
                                match (trace_meta, self.sym.as_ref()) {
                                    (Some(meta), Some(sym)) => {
                                        self.meta.trace_ctx().and_then(|ctx| {
                                            S::close_loop_live_values(
                                                ctx,
                                                sym,
                                                &meta,
                                                &live_arg_boxes,
                                            )
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            // pyjitpl.py `compile.compile_trace(self, self.resumekey, ...)`:
                            // the resumekey decides the bridge shape.  A guard origin is a
                            // ResumeGuardDescr (`close_bridge` patches the failing guard); no bridge
                            // origin is a ResumeFromInterpDescr entry bridge
                            // (`compile_trace_from_interp`, compile.py:1002-1021).
                            let bridge_origin =
                                self.meta.bridge_info().map(|b| (b.trace_id, b.fail_index));
                            result = match bridge_origin {
                                Some((trace_id, fail_index)) => self.meta.close_bridge(
                                    target_key,
                                    trace_id,
                                    fail_index,
                                    &live_arg_boxes,
                                ),
                                // compile.py:269-270: a cross-loop CUT keeps its cut prefix
                                // in `front_target_tokens[0]`, where a loop closed at its own
                                // header keeps its PREAMBLE — the peeled iteration that
                                // re-derives the specialized label's invariants from the
                                // loop's entry state, which is what makes `jump_to_preamble`
                                // (unroll.py:238-242) sound as the no-matching-label
                                // fallback. The cut prefix's invariants instead hold only
                                // after the guards preceding the cut point have run.
                                // `ResumeFromInterpDescr.get_resumestorage()` is None, so an
                                // entry bridge carries no runtime values, `generate_guards`
                                // rejects every specialized label, and the JUMP always
                                // reaches that slot; on cel's
                                // `batch_chain_eager_fold_traps` the unproven index bound
                                // became an out-of-bounds store and a SIGSEGV on the reload.
                                //
                                // Scoped to this arm alone. A `ResumeGuardDescr` has resume
                                // storage, so a specialized label can match and the fallback
                                // need not be reached — and declining the guard origin too
                                // makes `pi/pi.jinseo` compute wrong digits from
                                // byte 865, same length, no crash. That was
                                // measured at threshold 50, and `MAJIT_THRESHOLD`
                                // is NOT read anywhere — the threshold reaches the
                                // driver only as the `JitDriver::new(threshold)`
                                // argument, so reproducing this means changing the
                                // caller, not setting a variable.
                                //
                                // What the widened arm produces, at the op level: the
                                // declined close returns here, the walk runs on and closes a
                                // second time one interpreter instruction later, and the bridge it
                                // compiles carries that instruction's pop (`IntAdd -1`, two
                                // `GcLoadR`, two `GcStore`) ahead of a `Jump` into a
                                // four-input label, where the close that was declined jumped
                                // into a three-input one.  So the cost is not the lost close
                                // but the resumed walk: declining and then giving the trace
                                // up outright reproduces the un-compiled output exactly.
                                None if self.meta.is_cross_loop_cut_key_on_driver(
                                    self.index().unwrap_or(0),
                                    target_key,
                                ) =>
                                {
                                    crate::pyjitpl::BridgeCompileResult::Declined
                                }
                                None => match self.compile_trace_entry_data() {
                                    Some((original_green_key, entry_meta)) => {
                                        let outcome = self.meta.compile_trace_from_interp(
                                            target_key,
                                            &live_arg_boxes,
                                            original_green_key,
                                            entry_meta,
                                        );
                                        self.meta.classify_compile_outcome(outcome)
                                    }
                                    None => crate::pyjitpl::BridgeCompileResult::Declined,
                                },
                            };
                        }
                        match result {
                            crate::pyjitpl::BridgeCompileResult::Compiled => {
                                // pyjitpl.py raise_if_successful → raise_continue_running_normally:
                                // the trace is over, the interpreter enters the loop it jumped into.
                                self.sym = None;
                                self.meta.clear_trace_session();
                                self.note_continue_running_normally(
                                    continue_running_normally_values,
                                    None,
                                );
                                self.meta.finish_trace_live();
                                self.meta.clear_pending_abort();
                                return;
                            }
                            crate::pyjitpl::BridgeCompileResult::RetraceNeeded => {
                                // pyjitpl.py:3003 and the sibling RetraceNeeded
                                // arm below: `compile_trace` armed
                                // `partial_trace`, so fall through to the
                                // reached_loop_header path that routes
                                // compile_loop -> compile_retrace in this call.
                                self.bridge_attempt_declined = true;
                            }
                            crate::pyjitpl::BridgeCompileResult::Declined => {
                                // pyjitpl.py `raise_if_successful` does not raise
                                // on None, so control reaches the
                                // `current_merge_points` scan (`reached_loop_header`).
                                // No prior same-greenkey entry appends and
                                // tracing continues; the next visit retries
                                // `compile_trace`. A prior entry calls
                                // `compile_loop`, which gives the trace up when
                                // that key still `has_compiled_targets`.
                                //
                                // An unattempted close (`partial_trace` already
                                // set, or the token is gone) reaches the same
                                // scan: `reached_loop_header` skips
                                // `compile_trace` and falls into the
                                // `current_merge_points` loop unchanged. With no
                                // prior same-greenkey entry it appends and keeps
                                // tracing; with one it falls through to
                                // `compile_loop` / `compile_retrace` below.
                                if attempted {
                                    crate::mc_diag_bump(50); // bridge_declined_close
                                } else {
                                    crate::mc_diag_bump(67); // bridge_unattempted_close
                                }
                                if self
                                    .keep_tracing_after_declined_jump(target_key, &live_arg_boxes)
                                {
                                    continue;
                                }
                                self.bridge_attempt_declined = true;
                            }
                            crate::pyjitpl::BridgeCompileResult::Failed => {
                                self.meta.abort_trace(false);
                                self.sym = None;
                                self.meta.clear_trace_session();
                                return;
                            }
                        }
                    }
                    // pyjitpl.py reached_loop_header parity.
                    // Path 1: bridge — only if has_compiled_targets (line 2982).
                    //
                    // pyjitpl.py `if not self.partial_trace:` gates the whole
                    // bridge attempt. Once `retrace_needed` has armed
                    // `partial_trace` the loop being closed is a RETRACE of an
                    // existing loop, so re-entering `compile_trace` would ask the
                    // optimizer to bridge into the very loop the retrace exists to
                    // respecialize — and on `RetraceNeeded` it would arm the
                    // retrace a second time. Fall straight through to the
                    // merge-point scan, which is the only consumer upstream leaves
                    // for a partial trace.
                    // pyjitpl.py `if not self.partial_trace:` is the only
                    // compile_trace gate. Upstream retries on every header
                    // visit; do not skip a later visit after a decline.
                    // `compile_trace_attempted` is that one call for this visit.
                    let has_partial_trace = self.meta.partial_trace().is_some();
                    if !compile_trace_attempted
                        && let Some(bridge) = self.meta.bridge_info().filter(|_| !has_partial_trace)
                    {
                        let bridge_key = bridge.green_key;
                        let bridge_trace_id = bridge.trace_id;
                        let bridge_fail_index = bridge.fail_index;
                        // pyjitpl.py:2981-2982:
                        //     ptoken = self.get_procedure_token(greenboxes)
                        //     if has_compiled_targets(ptoken): ...
                        // `greenboxes` is the *current* merge point's greens — the
                        // one this trace just reached — not the bridge origin from
                        // the failed guard.  A bridge that closes on an inner merge
                        // point (the cross-loop cut) must JUMP into the loop living
                        // AT that merge point: `pc` is a green, baked into each
                        // compiled loop as a constant, so jumping into the origin
                        // loop instead would enter it carrying another pc's reds.
                        let close_greens = self
                            .meta
                            .trace_ctx()
                            .and_then(|ctx| ctx.close_greens.clone());
                        // `get_procedure_token(greenboxes)` (pyjitpl.py) walks
                        // the cell chain and settles a chained bucket on the
                        // greens themselves; a bare bucket hash cannot, because a
                        // bucket whose occupant was filed under a minted cell key
                        // holds no cell AT the hash. `compile_loop` files and
                        // looks its loop up under the resolved key
                        // (`WarmEnterState::resolve_cell_key`), so a door reading
                        // the unresolved hash and a `compile_loop` reading the
                        // resolved one answer differently about one close: the
                        // door reports no target and falls through to a
                        // `compile_loop` that refuses the same key for already
                        // having a loop, giving the trace up as ABORT_BAD_LOOP.
                        // The guard is left as it was, so every later failure
                        // retraces it to the same answer and no bridge is ever
                        // built. Resolve here so both doors read one cell.
                        let close_key = self.meta.trace_ctx().and_then(|ctx| ctx.close_green_key());
                        let resolved_key = match &close_key {
                            Some(key) => {
                                let make_key = || key.clone();
                                // compile.py / pyjitpl.py reached_loop_header
                                // match greens on jitdriver_sd.warmstate.
                                Some(self.meta.resolve_cell_key_on_driver(
                                    self.meta.active_jitdriver_sd.unwrap_or(0),
                                    key.get_uhash(),
                                    Some(&make_key as &dyn Fn() -> majit_ir::GreenKey),
                                ))
                            }
                            // `ctx.green_key` is resolved at trace start.
                            None => self.current_trace_green_key(),
                        };
                        // The cell token a close resolves is `jump_op.getdescr()`
                        // upstream (unroll.py:196 `cell_token =
                        // jump_op.getdescr()`, :321-322 `jitcelltoken =
                        // jump_op.getdescr()`), an object identity that cannot
                        // miss, so the `unwrap_or(bridge_key)` below stands in for
                        // an invariant. It holds: `current_trace_green_key()` is
                        // `trace_ctx().map(|c| c.green_key())` and `merge_point`
                        // only dispatches while `is_tracing()`, so the `or_else`
                        // leg always answers.
                        debug_assert!(
                            resolved_key.is_some(),
                            "reached_loop_header: bridge close must resolve a target green key",
                        );
                        if crate::closedbg_enabled() {
                            eprintln!(
                                "@@@CLOSE BRIDGE-TARGET close_greens={close_greens:?} close_hash={} resolved_key={} origin_key={}",
                                close_key
                                    .as_ref()
                                    .map(|k| k.get_uhash() as i64)
                                    .unwrap_or(-1),
                                resolved_key.map(|k| k as i64).unwrap_or(-1),
                                self.current_trace_green_key()
                                    .map(|k| k as i64)
                                    .unwrap_or(-1),
                            );
                        }
                        let has_targets =
                            resolved_key.is_some_and(|k| self.meta.has_compiled_targets(k));
                        let target_key = resolved_key.unwrap_or(bridge_key);
                        if crate::closedbg_enabled() {
                            let hp = self
                                .meta
                                .trace_ctx()
                                .map(|c| c.header_pc as i64)
                                .unwrap_or(-1);
                            let nops = self
                                .meta
                                .trace_ctx()
                                .map(|c| c.num_ops() as i64)
                                .unwrap_or(-1);
                            eprintln!(
                                "@@@CLOSE BRIDGE bridge_key={} target_key={} has_targets={} header_pc={} num_ops={} -> {}",
                                bridge_key,
                                target_key,
                                has_targets as i32,
                                hp,
                                nops,
                                if has_targets {
                                    "close_bridge"
                                } else {
                                    "NEW-LOOP(degenerate?)"
                                }
                            );
                        }
                        if has_targets {
                            if crate::majit_log_enabled() {
                                eprintln!(
                                    "[bridge] CloseLoop -> close_bridge target_key={} (bridge_origin={}) trace={} fail={}",
                                    target_key, bridge_key, bridge_trace_id, bridge_fail_index
                                );
                            }
                            if self.sym.is_some() {
                                // pyjitpl.py `compile_trace(live_arg_boxes,
                                // ptoken)`: the bridge JUMP passes the very list
                                // built above — not a second, separately
                                // normalized one.
                                let finish_args = live_arg_boxes.clone();
                                let continue_running_normally_values = {
                                    let trace_meta = self.meta.trace_meta().cloned();
                                    match (trace_meta, self.sym.as_ref()) {
                                        (Some(meta), Some(sym)) => {
                                            self.meta.trace_ctx().and_then(|ctx| {
                                                S::close_loop_live_values(
                                                    ctx,
                                                    sym,
                                                    &meta,
                                                    &finish_args,
                                                )
                                            })
                                        }
                                        _ => None,
                                    }
                                };
                                let result = self.meta.close_bridge(
                                    target_key,
                                    bridge_trace_id,
                                    bridge_fail_index,
                                    &finish_args,
                                );
                                match result {
                                    crate::pyjitpl::BridgeCompileResult::Compiled => {
                                        self.sym = None;
                                        self.meta.clear_trace_session();
                                        // pyjitpl.py raise_if_successful():
                                        // successful bridge closure terminates tracing.
                                        self.note_continue_running_normally(
                                            continue_running_normally_values,
                                            None,
                                        );
                                        // Success edge: live teardown only, no
                                        // aborted_tracing accounting (see the
                                        // CompileTrace arm above).
                                        self.meta.finish_trace_live();
                                        self.meta.clear_pending_abort();
                                        return;
                                    }
                                    // pyjitpl.py: after retrace_needed(),
                                    // partial_trace is set on MetaInterp. Fall through
                                    // to compile_loop → compile_retrace in same call.
                                    crate::pyjitpl::BridgeCompileResult::RetraceNeeded => {
                                        // `bridge_info` is pyre's `self.resumekey`
                                        // class discriminator, and upstream writes
                                        // that field exactly twice — pyjitpl.py:2905
                                        // `ResumeFromInterpDescr(original_greenkey)`
                                        // and pyjitpl.py:2939 `= resumedescr` — both
                                        // at session start. `retrace_needed`
                                        // (pyjitpl.py) sets only
                                        // `partial_trace` / `retracing_from` /
                                        // `exported_state` / `heapcache.reset()`, so
                                        // the resumekey a guard-originated session
                                        // carries must survive into
                                        // `compile_retrace`'s
                                        // `resumekey.compile_and_attach`
                                        // (compile.py:392-393). Clearing it here
                                        // turned every such retrace into a
                                        // `ResumeFromInterpDescr` front-door install.
                                        //
                                        // Only the CLASS survives, not the
                                        // attempt: upstream closes this gate
                                        // through `partial_trace`
                                        // (pyjitpl.py), which
                                        // `retrace_needed` always arms because
                                        // compile.py:1079 calls it
                                        // unconditionally. majit calls it only
                                        // when `compile_bridge` produced an
                                        // exported state (`pyjitpl.rs`), so
                                        // without the latch the gate would stay
                                        // open on the `exported == None` edge.
                                        self.bridge_attempt_declined = true;
                                        // Fall through — do NOT return.
                                    }
                                    // pyjitpl.py raise_if_successful() does not
                                    // raise on a None target token, so the trace is
                                    // not given up: fall through to compile_loop the
                                    // way reached_loop_header falls into its
                                    // merge-point scan.
                                    crate::pyjitpl::BridgeCompileResult::Declined => {
                                        crate::mc_diag_bump(50); // bridge_declined_close
                                        // `reached_loop_header` — no prior same-greenkey
                                        // entry appends and the walk continues.
                                        // A prior entry falls through to
                                        // compile_loop.
                                        if self.keep_tracing_after_declined_jump(
                                            target_key,
                                            &live_arg_boxes,
                                        ) {
                                            continue;
                                        }
                                        // The resumekey survives because upstream
                                        // never clears it in this region
                                        // (pyjitpl.py:2979-3060); only the attempt
                                        // latch is set.
                                        self.bridge_attempt_declined = true;
                                    }
                                    crate::pyjitpl::BridgeCompileResult::Failed => {
                                        self.meta.abort_trace(false);
                                        self.sym = None;
                                        self.meta.clear_trace_session();
                                        return;
                                    }
                                }
                            } else {
                                self.meta.abort_trace(false);
                                self.sym = None;
                                self.meta.clear_trace_session();
                                return;
                            }
                        } else {
                            // No targets: consume bridge_info, fall through to
                            // compile_loop (pyjitpl.py). The resumekey
                            // survives because upstream never clears it in this
                            // region (pyjitpl.py); only the attempt latch
                            // is set.
                            crate::mc_diag_bump(51); // bridge_no_targets_close
                            self.bridge_attempt_declined = true;
                        }
                    }
                    let Some(trace_meta) = self.meta.trace_meta().cloned() else {
                        self.meta.abort_trace(false);
                        self.sym = None;
                        self.meta.clear_trace_session();
                        return;
                    };
                    let Some(sym) = self.sym.as_ref() else {
                        self.meta.abort_trace(false);
                        self.meta.clear_trace_session();
                        return;
                    };
                    if S::validate_close(sym, &trace_meta) {
                        // pyjitpl.py:3019/3059: the merge-point scan and the
                        // `current_merge_points.append` both read the one
                        // `live_arg_boxes` built at the top of this arm.
                        let jump_args = live_arg_boxes;
                        // pyjitpl.py parity: compile_loop borrows the
                        // frontend meta the same way `self.history` is a shared
                        // mutable object on the RPython MetaInterp — the caller
                        // does not drain it. Read via `trace_meta().cloned()` so
                        // the `active_trace_session` envelope stays live for the
                        // duration of the compile. `compile_loop` itself owns
                        // the post-exit invariant that session lifetime tracks
                        // `self.tracing` (see `compile_loop` wrapper).
                        let provisional_meta = self.meta.trace_meta().cloned().unwrap();
                        // When cross-loop cut redirects to a different header_pc,
                        // rebuild meta from merge point to get the inner header's
                        // frame layout (vsd, slot_types).
                        let meta = {
                            if let Some((cut_pc, ref cut_boxes)) = self.meta.cross_loop_cut_info() {
                                S::build_meta_from_merge_point(&provisional_meta, cut_pc, cut_boxes)
                            } else {
                                provisional_meta
                            }
                        };
                        let continue_running_normally_values = self
                            .meta
                            .trace_ctx()
                            .and_then(|ctx| S::close_loop_live_values(ctx, sym, &meta, &jump_args));
                        self.meta.single_pass_full_live_values =
                            continue_running_normally_values.clone();
                        let outcome = self.compile_and_record_loop(&jump_args, meta);
                        match outcome {
                            crate::CompileOutcome::Compiled { .. } => {
                                // pyjitpl.py raise_if_successful →
                                // raise_continue_running_normally(live_arg_boxes).
                                self.note_continue_running_normally(
                                    continue_running_normally_values,
                                    None,
                                );
                                // compile_loop already drained tracing+session.
                            }
                            crate::CompileOutcome::Cancelled => {
                                // pyjitpl.py:3018-3032 parity: cancel_count is
                                // incremented inside compile_loop and tracing
                                // continues unless `cancelled_too_many_times`
                                // escalates to Aborted. The wrapper restored
                                // session↔tracing alignment so no extra cleanup
                                // is required here — either the early-Cancelled
                                // path kept both live (tracing continues) or a
                                // late-Cancelled path drained both (next bound
                                // reaches begins a fresh session).
                            }
                            crate::CompileOutcome::Aborted => {
                                self.meta.abort_trace(false);
                            }
                            crate::CompileOutcome::SwitchToBlackhole { reason } => {
                                // `pyjitpl.py compile_loop` /
                                // `_compile_and_run_once`: one
                                // `run_blackhole_interp_to_cancel_tracing`.
                                // The walker has no metainterp framestack here;
                                // `abort_trace` is that catch's accounting and
                                // hands the framestack to the portal blackhole.
                                self.meta.stage_abort_reason(reason);
                                self.meta.abort_trace(false);
                            }
                        }
                    } else {
                        crate::debug::log_one("jit-abort", "mp: abort:validate_close");
                        self.meta.abort_trace(false);
                        self.meta.clear_trace_session();
                    }
                    // pyjitpl.py: a `reached_loop_header` that only
                    // appended a merge point RETURNS — the trace is not over and
                    // the metainterp goes on recording into the next iteration.
                    // Dropping `self.sym` is how this driver says "the trace ended
                    // here" (`merge_point` aborts on a None sym), so that one path
                    // has to keep it — and it must re-enter the walk rather than
                    // let the native dispatch loop run the merge point's own
                    // instruction unrecorded (see the loop header above).
                    if self.meta.take_keep_tracing_after_close() {
                        // The walk-final handoff staged at the top of this arm
                        // describes a trace that ENDED. Discard it: publishing it
                        // would make the `jit_merge_point!` expansion write the
                        // walk state into native `state` and resume the guest at
                        // the merge-point pc, which is exactly the instruction the
                        // next walk segment still has to record.
                        self.meta.single_pass_outcome = None;
                        self.meta.single_pass_scalar_values = None;
                        self.meta.single_pass_ref_scalar_values = None;
                        self.meta.single_pass_virt_array_values = None;
                        // pyjitpl.py:1571/1574 `saved_pc` — the resumed walk
                        // re-enters at the merge point's own guest pc, so the
                        // merge-point op must fall through once instead of
                        // closing again with nothing recorded in between, and
                        // the walk must be seeded THERE rather than wherever
                        // its segment began. `register_retrace_merge_point`
                        // registered this merge point under `close_header_pc`,
                        // the same pc `close_green_pc` names, so resuming
                        // anywhere else also disagrees with that registration.
                        //
                        // The slot is here because nothing available reaches
                        // this branch: it fires on no test in this crate, no
                        // majit example, and no downstream corpus program at
                        // thresholds 2/8/32/128. A zero says "still unreached";
                        // the first non-zero is the first evidence this
                        // resumption is exercised at all.
                        crate::mc_diag_bump(78); // retrace_close_resumed
                        if let Some(ctx) = self.meta.trace_ctx() {
                            ctx.resume_walk_after_close();
                        }
                        continue;
                    }
                    self.sym = None;
                }
                TraceAction::CloseLoopWithArgs {
                    jump_args,
                    loop_header_pc,
                } => {
                    // pyjitpl.py reached_loop_header parity.
                    //
                    // pyjitpl.py `if not self.partial_trace:` gates the whole
                    // bridge attempt, the same way it does on the `CloseLoop` twin
                    // above: once `retrace_needed` has armed `partial_trace` the
                    // loop being closed is a RETRACE, and re-entering
                    // `compile_trace` would both bridge into the loop the retrace
                    // exists to respecialize and arm the retrace a second time.
                    // pyjitpl.py `if not self.partial_trace:` is the only
                    // compile_trace gate. Upstream retries on every header
                    // visit; do not skip a later visit after a decline.
                    let has_partial_trace = self.meta.partial_trace().is_some();
                    if let Some(bridge) = self.meta.bridge_info().filter(|_| !has_partial_trace) {
                        let bridge_key = bridge.green_key;
                        let bridge_trace_id = bridge.trace_id;
                        let bridge_fail_index = bridge.fail_index;
                        let bridge_code_ptr = bridge.code_ptr;
                        // pyjitpl.py: ptoken = self.get_procedure_token(greenboxes)
                        // Use the TARGET loop header's green key, not the bridge origin.
                        let target_key = loop_header_pc
                            .map(|pc| crate::green_key_from_code_ptr(bridge_code_ptr, pc))
                            .unwrap_or(bridge_key);
                        let has_targets = self.meta.has_compiled_targets(target_key);
                        if has_targets {
                            let continue_running_normally_values = {
                                let trace_meta = self.meta.trace_meta().cloned();
                                match (trace_meta, self.sym.as_ref()) {
                                    (Some(meta), Some(sym)) => {
                                        self.meta.trace_ctx().and_then(|ctx| {
                                            S::close_loop_live_values(ctx, sym, &meta, &jump_args)
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            let result = self.meta.close_bridge(
                                target_key,
                                bridge_trace_id,
                                bridge_fail_index,
                                &jump_args,
                            );
                            match result {
                                crate::pyjitpl::BridgeCompileResult::Compiled => {
                                    self.sym = None;
                                    self.meta.clear_trace_session();
                                    // pyjitpl.py raise_if_successful():
                                    // successful bridge closure terminates tracing.
                                    self.note_continue_running_normally(
                                        continue_running_normally_values,
                                        loop_header_pc,
                                    );
                                    // Success edge: live teardown only, no
                                    // aborted_tracing accounting (see the
                                    // CompileTrace arm above).
                                    self.meta.finish_trace_live();
                                    self.meta.clear_pending_abort();
                                    return;
                                }
                                // pyjitpl.py: after retrace_needed(),
                                // partial_trace is set on MetaInterp. Fall through
                                // to compile_loop → compile_retrace in same call.
                                crate::pyjitpl::BridgeCompileResult::RetraceNeeded => {
                                    // Same as the `CloseLoop` twin: `bridge_info`
                                    // is the `self.resumekey` class discriminator
                                    // and upstream writes it only at session start
                                    // (pyjitpl.py:2905 / 2939), never in
                                    // `retrace_needed` (pyjitpl.py), so
                                    // it has to survive into
                                    // `resumekey.compile_and_attach`
                                    // (compile.py:392-393). The attempt latch is
                                    // set for the same reason as on that twin —
                                    // `partial_trace` only closes this gate when
                                    // `compile_bridge` exported a state.
                                    self.bridge_attempt_declined = true;
                                    // Fall through — do NOT return.
                                }
                                // pyjitpl.py raise_if_successful() does not
                                // raise on a None target token, so the trace is not
                                // given up: fall through to compile_loop the way
                                // reached_loop_header falls into its merge-point
                                // scan.
                                crate::pyjitpl::BridgeCompileResult::Declined => {
                                    crate::mc_diag_bump(50); // bridge_declined_close
                                    // `reached_loop_header` — no prior same-greenkey
                                    // entry appends and the walk continues.
                                    // A prior entry falls through to compile_loop.
                                    if self.keep_tracing_after_declined_jump(target_key, &jump_args)
                                    {
                                        continue;
                                    }
                                    // The resumekey survives because upstream never
                                    // clears it in this region
                                    // (pyjitpl.py:2979-3060); only the attempt
                                    // latch is set.
                                    self.bridge_attempt_declined = true;
                                }
                                crate::pyjitpl::BridgeCompileResult::Failed => {
                                    self.meta.abort_trace(false);
                                    self.sym = None;
                                    self.meta.clear_trace_session();
                                    return;
                                }
                            }
                        } else {
                            // No targets: consume bridge_info, fall through. The
                            // resumekey survives because upstream never clears it
                            // in this region (pyjitpl.py:2979-3060); only the
                            // attempt latch is set.
                            crate::mc_diag_bump(51); // bridge_no_targets_close
                            self.bridge_attempt_declined = true;
                        }
                    }
                    // pyjitpl.py: compile_trace already compiled a bridge
                    // to the existing loop. Don't call compile_loop again.
                    if self.compile_trace_success_pending() {
                        self.sym = None;
                        self.meta.clear_trace_session();
                        return;
                    }
                    let Some(trace_meta) = self.meta.trace_meta().cloned() else {
                        self.meta.abort_trace(false);
                        self.sym = None;
                        self.meta.clear_trace_session();
                        return;
                    };
                    let Some(sym) = self.sym.as_ref() else {
                        self.meta.abort_trace(false);
                        self.meta.clear_trace_session();
                        return;
                    };
                    if S::validate_close_with_jump_args(sym, &trace_meta, &jump_args) {
                        // pyjitpl.py:2993-3036 parity: keep `active_trace_session`
                        // alive across compile_loop — the session envelope mirrors
                        // RPython's `self.history` which compile_loop borrows
                        // without consuming.
                        let provisional_meta = self.meta.trace_meta().cloned().unwrap();
                        let meta = {
                            if let Some((cut_pc, ref cut_boxes)) = self.meta.cross_loop_cut_info() {
                                S::build_meta_from_merge_point(&provisional_meta, cut_pc, cut_boxes)
                            } else {
                                provisional_meta
                            }
                        };
                        let continue_running_normally_values = self
                            .meta
                            .trace_ctx()
                            .and_then(|ctx| S::close_loop_live_values(ctx, sym, &meta, &jump_args));
                        self.meta.single_pass_full_live_values =
                            continue_running_normally_values.clone();
                        let outcome = self.compile_and_record_loop(&jump_args, meta);
                        match outcome {
                            crate::CompileOutcome::Compiled { .. } => {
                                // pyjitpl.py raise_if_successful →
                                // raise_continue_running_normally(live_arg_boxes).
                                self.note_continue_running_normally(
                                    continue_running_normally_values,
                                    loop_header_pc,
                                );
                            }
                            crate::CompileOutcome::Cancelled => {
                                // pyjitpl.py:3018-3032 parity: no explicit abort,
                                // tracing continues (or is already drained inside
                                // compile_loop for late-cancel cases). The wrapper
                                // maintains the session↔tracing invariant.
                            }
                            crate::CompileOutcome::Aborted => {
                                self.meta.abort_trace(false);
                            }
                            crate::CompileOutcome::SwitchToBlackhole { reason } => {
                                self.meta.stage_abort_reason(reason);
                                self.meta.abort_trace(false);
                            }
                        }
                    } else {
                        if crate::majit_log_enabled() {
                            eprintln!(
                                "[mp] abort:validate_close_with_jump_args actual_len={}",
                                jump_args.len()
                            );
                        }
                        self.meta.abort_trace(false);
                        self.meta.clear_trace_session();
                    }
                    // `reached_loop_header` keeps tracing after a cancel no
                    // matter how the JUMP arguments were supplied. Same latch
                    // as the `CloseLoop` arm: `resume_walk_after_close` and
                    // leave `self.sym` in place.
                    if self.meta.take_keep_tracing_after_close() {
                        self.meta.single_pass_outcome = None;
                        self.meta.single_pass_scalar_values = None;
                        self.meta.single_pass_ref_scalar_values = None;
                        self.meta.single_pass_virt_array_values = None;
                        crate::mc_diag_bump(78); // retrace_close_resumed
                        if let Some(ctx) = self.meta.trace_ctx() {
                            ctx.resume_walk_after_close();
                        }
                        continue;
                    }
                    self.sym = None;
                }
                TraceAction::Finish {
                    finish_args,
                    finish_arg_types,
                    exit_with_exception,
                    exc_value,
                } => {
                    self.meta.single_pass_finish_values = self
                        .meta
                        .trace_ctx()
                        .map(|ctx| ctx.walk_finish_values.iter().copied().collect());
                    // `pyjitpl.py finishframe`: after `compile_done_with_this_frame`
                    // the portal raises `jitexc.DoneWithThisFrame*` with
                    // `resultbox`. The function-entry Run arm drains
                    // `back_edge_finish_word` first; leaving it empty re-enters
                    // `#portal_ident` at pc 0.
                    if let Some(values) = self.meta.single_pass_finish_values.as_ref() {
                        match values.first() {
                            Some(Value::Int(v)) => {
                                self.meta.back_edge_finish_word = Some(*v);
                            }
                            Some(Value::Float(v)) => {
                                self.meta.back_edge_finish_word = Some(v.to_bits() as i64);
                            }
                            Some(Value::Ref(v)) => {
                                self.meta.back_edge_finish_word = Some(v.as_usize() as i64);
                            }
                            _ => {}
                        }
                    }
                    // A terminal dispatch return is also a valid single-pass
                    // handoff: the interpreted function has returned, so native
                    // execution must EXIT the dispatch loop and run its own
                    // post-loop work exactly once. The scalar/virt-array write-back
                    // still applies (post-loop reads the walk-final state), but the
                    // `single_pass_finish` flag tells the macro to `break` rather
                    // than resume at the captured pc. The captured pc (i0, past the
                    // terminal opcode) only lands past the program end — where
                    // `while pc < size` would exit on resume — when the terminal is
                    // the last opcode; a mid-program terminal would re-enter the
                    // loop body, so the exit is signalled explicitly.
                    //
                    // The flag is not conditional on that pc.  `Finish` IS the
                    // interpreted function returning, so the dispatch loop has
                    // to exit either way; a walk that published its finish
                    // values without a final pc would otherwise leave the hook
                    // with neither a break nor a resume and spin the native
                    // loop.
                    //
                    // The outcome is not conditional on it either, and for the
                    // same reason: the hook reads `single_pass_finish` only
                    // from INSIDE the outcome branch, so publishing nothing
                    // would hide the flag it is supposed to carry.
                    // `usize::MAX` is the no-position sentinel the wrapper
                    // itself returns for a finish, so a consumer that ignores
                    // the flag fails loudly instead of resuming somewhere.
                    let pc = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.walk_final_pc)
                        .unwrap_or(usize::MAX);
                    let finish_key = self.current_trace_green_key();
                    self.publish_single_pass_outcome(pc);
                    if let Some(k) = finish_key {
                        if let Some((_, _, args)) = self.meta.single_pass_outcome.as_ref() {
                            self.meta.record_loop_header_greens(
                                k,
                                (
                                    args.green_int.clone(),
                                    args.green_ref.clone(),
                                    args.green_float.clone(),
                                ),
                            );
                        }
                    }
                    self.meta.single_pass_finish = true;
                    // `virtualizable.py write_from_resume_data_partial` writes
                    // every static field before the portal runner continues.
                    // A `void_return` Finish has no resultbox; the native
                    // suffix reads these reds (`compile_done_with_this_frame`
                    // then `DoneWithThisFrameVoid`).
                    if let Some(scalars) = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.close_scalar_values.take())
                    {
                        self.meta.single_pass_scalar_values = Some(scalars);
                    } else if let Some(sym) = self.sym.as_ref() {
                        let scalars = S::collect_scalar_state_field_values(sym);
                        if !scalars.is_empty() {
                            self.meta.single_pass_scalar_values = Some(scalars);
                        }
                    }
                    if let Some(ref_scalars) = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.close_ref_scalar_values.take())
                    {
                        self.meta.single_pass_ref_scalar_values = Some(ref_scalars);
                    } else if let Some(sym) = self.sym.as_ref() {
                        let ref_scalars = S::collect_ref_scalar_state_field_values(sym);
                        if !ref_scalars.is_empty() {
                            self.meta.single_pass_ref_scalar_values = Some(ref_scalars);
                        }
                    }
                    let virt_elems = self
                        .meta
                        .trace_ctx()
                        .and_then(|ctx| ctx.collect_virtualizable_element_values());
                    if let Some(elems) = virt_elems {
                        self.meta.single_pass_virt_array_values = Some(elems);
                    }
                    // The `raise` half of `pyjitpl.py:2558-2562`.  This arm is the
                    // consumer for `#[jit_interp]`-generated tracers, which reach
                    // it through `merge_point` and whose walk carries an escaping
                    // exception in `BH_LAST_EXC_VALUE` — the only channel a
                    // macro-front-end client has.  The producer
                    // (`dispatch.rs unwind_to_exception_handler`) closes that
                    // channel and hands the value over on the action, so re-open
                    // it here; dropping it would let the portal finish as if the
                    // frame had returned normally.
                    //
                    // Published *before* the compile: upstream snapshots
                    // `excvalue` into a local at :2531 and that local keeps the
                    // ref alive as a GC root across
                    // `compile_exit_frame_with_exception` (:2558).  A raw `i64`
                    // is no root, and `compile_finish_from_active_session`
                    // allocates, so the walked TLS cell has to hold the value
                    // across it.
                    if exit_with_exception && exc_value != 0 {
                        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_value));
                    }
                    // pyjitpl.py:3198-3220 + 3238-3245 parity — upstream's
                    // `compile_done_with_this_frame` and
                    // `compile_exit_frame_with_exception` both call
                    // `compile.compile_trace(self, self.resumekey, exits)` →
                    // `compile.giveup()` on failure.  Both live on
                    // `MetaInterp` in upstream; the shared helper
                    // `compile_finish_from_active_session` dispatches bridge
                    // vs root based on the session's `bridge_info` and picks
                    // the FINISH descr per `exit_with_exception`.  The
                    // TraceAction::Finish arm stays as a thin wrapper that
                    // routes the pyre-dispatch-layer finish emission
                    // (`pyjitpl/dispatch.rs`'s `run_one_step`) into the same
                    // helper, matching the
                    // `except SwitchToBlackhole as stb: self.aborted_tracing`
                    // shape at pyjitpl.py.
                    if let Err(stb) = self.meta.compile_finish_from_active_session(
                        &finish_args,
                        finish_arg_types,
                        exit_with_exception,
                    ) {
                        self.meta.aborted_tracing(stb.reason);
                    }
                    self.sym = None;
                }
                action @ (TraceAction::SegmentedLoop | TraceAction::SegmentedBridge { .. }) => {
                    // pyjitpl.py _create_segmented_trace_and_blackhole,
                    // compile half.  Upstream branches on whether the segmented
                    // trace owns a merge point and entered from the interpreter:
                    //   loop:   target_token = compile.compile_simple_loop(...)
                    //           warmstate.attach_procedure_to_interp(greenkey, token)
                    //   bridge: target_token = compile.compile_trace(
                    //               metainterp, metainterp.resumekey, [exception_box])
                    //           if target_token is not token: compile.giveup()
                    // then leaves through one shared
                    // pyjitpl.py SwitchToBlackhole(ABORT_SEGMENTED_TRACE).
                    // `create_segmented_trace` made that choice while it still
                    // held the `TraceCtx`; the box it carries is the FINISH
                    // operand the bridge arm still has to record.
                    let bridge_exception_box = match action {
                        TraceAction::SegmentedBridge { exception_box } => Some(exception_box),
                        _ => None,
                    };
                    // pyjitpl.py:1671-1672 blackholes back to the interpreter
                    // rather than jumping to compiled code.  Publish the
                    // single-pass handoff first — the walk executed everything
                    // it recorded, so the interpreter has to continue from where
                    // the walk stopped (`create_segmented_trace` stashed that
                    // pc) and with the walk's scalar / virtualizable state.
                    // Read before `take_trace_meta` drains the session.
                    let pc = self.meta.trace_ctx().and_then(|ctx| ctx.walk_final_pc);
                    if let Some(pc) = pc {
                        self.publish_single_pass_outcome(pc);
                        if let Some(scalars) = self
                            .meta
                            .trace_ctx()
                            .and_then(|ctx| ctx.close_scalar_values.take())
                        {
                            self.meta.single_pass_scalar_values = Some(scalars);
                        } else if let Some(sym) = self.sym.as_ref() {
                            let scalars = S::collect_scalar_state_field_values(sym);
                            if !scalars.is_empty() {
                                self.meta.single_pass_scalar_values = Some(scalars);
                            }
                        }
                        let virt_elems = self
                            .meta
                            .trace_ctx()
                            .and_then(|ctx| ctx.collect_virtualizable_element_values());
                        if let Some(elems) = virt_elems {
                            self.meta.single_pass_virt_array_values = Some(elems);
                        }
                    }
                    let abort_reason = match bridge_exception_box {
                        // pyjitpl.py:1665-1668, the else-arm: a guard-origin
                        // bridge has no merge point to make a loop out of, so
                        // the segment is closed as an ordinary bridge.
                        // `compile_finish_from_active_session` IS
                        // `compile.compile_trace(metainterp, self.resumekey,
                        // [exception_box])` for a bridge origin: it records the
                        // FINISH under `sd.exit_frame_with_exception_descr_ref`
                        // and returns `Err` exactly where upstream's
                        // `target_token is not token` reaches `compile.giveup()`
                        // (compile.py:27, `SwitchToBlackhole(ABORT_BRIDGE)`).
                        //
                        // The flag this arm answers to was already being armed
                        // with nothing to act on it: `prepare_trace_segmenting`
                        // (pyjitpl.py) sets FORCE_BRIDGE_SEGMENTING on
                        // the source loop token, `start_retrace_from_guard`
                        // reads it back into `force_finish_trace`
                        // (compile.py:725-731), and the bridge then ran on to
                        // the ordinary over-limit abort — the retracing-forever
                        // outcome the flag is set to prevent.  Upstream's own
                        // note at pyjitpl.py: "creating a segmented bridge
                        // is generally quite safe".
                        Some(exception_box) => {
                            // Read before the compile drains the tracer: the
                            // bridge's own key is what its abort hook reports.
                            let green_key = self.meta.trace_ctx().map(|ctx| ctx.green_key);
                            let result = self.meta.compile_finish_from_active_session(
                                &[exception_box],
                                vec![majit_ir::Type::Int],
                                /* exit_with_exception */ true,
                            );
                            // On the giveup path `abort_trace_live` already
                            // staged this key; on the success path it was
                            // cleared again.  Stage it either way so the hook
                            // below does not report key 0.
                            self.meta.pending_abort_green_key = green_key;
                            match result {
                                Ok(()) => crate::counters::ABORT_SEGMENTED_TRACE,
                                Err(stb) => stb.reason,
                            }
                        }
                        None => {
                            let meta = self.meta.take_trace_meta().unwrap();
                            self.compile_segmented_loop(meta);
                            crate::counters::ABORT_SEGMENTED_TRACE
                        }
                    };
                    // Blackhole transition: clear all driver tracing state.
                    // `take_trace_meta` / `compile_finish_from_active_session`
                    // above drained the M-ownership side;
                    // `leave_profiler_tracing` fires the matching
                    // `pyjitpl.py:2934 finally: profiler.end_tracing()`.
                    // `clear_trace_session` is then a no-op for both effects
                    // (session already drained, profiler already left) and
                    // only cleans `bridge_info`.
                    self.sym = None;
                    self.meta.leave_profiler_tracing();
                    self.meta.clear_trace_session();
                    // pyjitpl.py:1673 `raise SwitchToBlackhole(
                    // Counters.ABORT_SEGMENTED_TRACE)` — upstream's handler
                    // (pyjitpl.py) runs `aborted_tracing(stb.reason)`, so a
                    // segmented trace counts as an abort with its own reason.
                    // A bridge arm that gave up carries `ABORT_BRIDGE` instead,
                    // for the same reason: `giveup` raises its own
                    // SwitchToBlackhole and that is the reason the handler sees.
                    self.meta.aborted_tracing(abort_reason);
                }
                TraceAction::Decline => {
                    // This is a pre-trace coverage refusal, not
                    // `SwitchToBlackhole`.  Upstream's
                    // `maybe_compile_and_run` only clears `JC_TRACING` in its
                    // `finally` arm; charging `abort_trace_live` here made five
                    // unrelated declines trip pyre's local abort ceiling.
                    self.meta.decline_trace_live();
                    self.sym = None;
                    self.bridge_attempt_declined = false;
                    self.meta.clear_trace_session();
                }
                // Consumed inside the metainterp dispatch loop
                // (PyreMetaInterp::step_inline_frame pops the inline frame and
                // records the CALL_ASSEMBLER); it never reaches the jitdriver.
                // Defensive: treat an escaped instance as a plain Abort.
                action @ (TraceAction::RecursiveCallAssembler { .. }
                | TraceAction::Abort
                | TraceAction::SwitchToBlackhole(_)) => {
                    if self.meta.bridge_info().is_some() {
                        crate::debug::log_one("jit-abort", "Abort during bridge tracing");
                    }
                    if self.bridge_attempt_declined {
                        crate::mc_diag_bump(52); // abort_after_declined
                    }
                    // `AbstractResumeGuardDescr.done_compiling` does not stick
                    // a failed bridge. A setup abort or a deterministic bridge
                    // abort leaves the guard's jitcounter where `jitcounter.tick`
                    // reset it, and the next failure ticks again.
                    // pyjitpl.py `run_blackhole_interp_to_cancel_tracing(stb)`
                    // consumes both the reason and raising_exception from the
                    // same signal. The Python worker returns it with the action;
                    // legacy dispatch sites still put it on the trace context.
                    let pending_stb = match &action {
                        TraceAction::SwitchToBlackhole(stb) => Some(*stb),
                        _ => None,
                    };
                    // `history.py:37-43`: false unless the abort site raised at a
                    // point where `last_exc_value` still has to be raised
                    // (`pyjitpl.py:3389-3390` vable escape, `pyjitpl.py:3714-3716`
                    // `do_not_in_trace_call`). A too-long signal carries false,
                    // as upstream's `SwitchToBlackhole(ABORT_TOO_LONG)` does.
                    let abort_raising_exception = pending_stb
                        .as_ref()
                        .is_some_and(|stb| stb.raising_exception);
                    // Explicit SwitchToBlackhole signals bypass the legacy
                    // length classification. In particular, the too-long raise
                    // has already disabled its callee and retired the log.
                    let reason_int = match pending_stb
                        .map(|stb| stb.reason)
                        .or_else(|| self.meta.take_pending_abort_reason())
                    {
                        Some(r) => r,
                        // The standalone JitCodeMachine still returns plain
                        // Abort for its post-step overflow. Keep its legacy
                        // classification here until it owns the MetaInterp
                        // decision. A SwitchToBlackhole always takes the Some
                        // arm, as pyjitpl.py's exception catch does.
                        None => match self.meta.blackhole_if_trace_too_long() {
                            Some(r) => r.as_int(),
                            None => {
                                crate::mc_diag_bump(71);
                                AbortReason::Generic.as_int()
                            }
                        },
                    };
                    // pyjitpl.py run_blackhole_interp_to_cancel_tracing
                    // run_blackhole_interp_to_cancel_tracing: a fresh trace has
                    // executed its portal body forward with real residual heap
                    // effects.  Before dropping that live frame, publish its
                    // current source pc and scalar state for the single-pass
                    // jit_merge_point hook.  The hook restores those scalars,
                    // runs `recover_after_compiled_run`, and continues the native
                    // interpreter from this pc rather than replaying the committed
                    // trace prefix.
                    //
                    // Bridge/compiled guard-failure recovery owns a separate
                    // blackhole resume path below `back_edge_internal`; do not
                    // feed that path through this fresh-trace handoff.
                    //
                    // Unless the bridge was entered at the guard's own position
                    // — then there was no separate resume: the walk is the only
                    // thing that ran, and it ran the failing opcode's tail. Its
                    // frames are the sole record of where that left the
                    // interpreter, so this bridge needs the handoff for the
                    // same reason a fresh trace does.
                    //
                    // `handle_guard_failure` catches `SwitchToBlackhole` with
                    // `run_blackhole_interp_to_cancel_tracing` →
                    // `MetaInterp.aborted_tracing` →
                    // `convert_and_run_from_pyjitpl` on the live framestack.
                    // A deterministic bridge abort (`BC_ABORT`, a refused
                    // walk-local residual) is still that abort: converting
                    // from the abort point finishes the half-opcode. Returning
                    // `None` from `bridge_from_guard_resume_position` would
                    // take `ResumeGuardDescr.handle_fail`'s else-arm
                    // (`resume_in_blackhole` from the guard deadframe) and
                    // replay residuals the walk already executed
                    // (`note_residual_committed` / `ITER_NEXT`).
                    if matches!(
                        action,
                        TraceAction::Abort | TraceAction::SwitchToBlackhole(_)
                    ) && (self.meta.bridge_info().is_none()
                        || self.bridge_attempt_declined
                        || self.bridge_entered_at_guard_resume)
                    {
                        // This gate asks whether the session still has a bridge
                        // artifact to resume into, which is the PHASE, not the
                        // resumekey class. A declined attempt has none, so it takes
                        // the fresh-trace handoff exactly as it did when the decline
                        // arms cleared `bridge_info`.
                        let pc = self.meta.trace_ctx().and_then(|ctx| ctx.walk_final_pc);
                        if let Some(pc) = pc {
                            self.publish_single_pass_outcome(pc);
                            if let Some(scalars) = self
                                .meta
                                .trace_ctx()
                                .and_then(|ctx| ctx.close_scalar_values.take())
                            {
                                self.meta.single_pass_scalar_values = Some(scalars);
                            } else if let Some(sym) = self.sym.as_ref() {
                                let scalars = S::collect_scalar_state_field_values(sym);
                                if !scalars.is_empty() {
                                    self.meta.single_pass_scalar_values = Some(scalars);
                                }
                            }
                            let virt_elems = self
                                .meta
                                .trace_ctx()
                                .and_then(|ctx| ctx.collect_virtualizable_element_values());
                            if let Some(elems) = virt_elems {
                                self.meta.single_pass_virt_array_values = Some(elems);
                            }
                        }
                        // `pyjitpl.py:2954 convert_and_run_from_pyjitpl(self,
                        // stb.raising_exception)`: the source pc above is only a
                        // sound resume position when the walk happened to stop at a
                        // source-opcode boundary.  Stage the framestack and the
                        // sym's state-field image so the `jit_merge_point!` hook can
                        // finish the half-executed opcodes in the blackhole and take
                        // the resume position from the merge point they reach.
                        let mut aborted = self
                            .meta
                            .tracing
                            .as_mut()
                            .and_then(|ctx| ctx.aborted_framestack.take());
                        // `_copy_data_from_miframe` reads `getint()` off the
                        // register box. Rewrite non-const boxes to inline
                        // Consts while the recorder still holds `_res*`.
                        if let Some(ctx) = self.meta.tracing.as_ref() {
                            if let Some(ref mut fs) = aborted {
                                for frame in fs.frames.iter_mut() {
                                    frame.freeze_values_into_const_boxes(ctx);
                                }
                            }
                            for frame in self.meta.framestack.frames.iter_mut() {
                                frame.freeze_values_into_const_boxes(ctx);
                            }
                        }
                        let virt_and_ptr = self.meta.trace_ctx().map(|ctx| {
                            (
                                ctx.collect_virtualizable_element_values(),
                                ctx.virtualizable_heap_ptr().map_or(0, |p| p as i64),
                            )
                        });
                        // Standalone walks publish into `aborted_framestack`.
                        // `MetaInterp::interpret` keeps the same stack on the
                        // MetaInterp (`pyjitpl.py` `_interpret`).
                        let staged = aborted
                            .or_else(|| {
                                (!self.meta.framestack.is_empty()).then(|| {
                                    std::mem::replace(
                                        &mut self.meta.framestack,
                                        crate::pyjitpl::MIFrameStack::empty(),
                                    )
                                })
                            })
                            .map(|framestack| {
                                let (virt_array_values, virtualizable_ptr) =
                                    virt_and_ptr.unwrap_or((None, 0));
                                (framestack, virt_array_values, virtualizable_ptr)
                            });
                        // `blackhole.py convert_and_run_from_pyjitpl` only
                        // needs the metainterp framestack. The sym image is
                        // extra state-field seed for generated machines;
                        // the Python portal has none, so missing `self.sym`
                        // must not drop the conversion.
                        if let Some((framestack, virt_array_values, virtualizable_ptr)) = staged {
                            let (scalar_values, ref_scalar_values) = match self.meta.trace_ctx() {
                                Some(ctx) => (
                                    ctx.close_scalar_values.take().unwrap_or_default(),
                                    ctx.close_ref_scalar_values.take().unwrap_or_default(),
                                ),
                                None => (Vec::new(), Vec::new()),
                            };
                            let (scalar_values, ref_scalar_values) =
                                if scalar_values.is_empty() && ref_scalar_values.is_empty() {
                                    if let Some(sym) = self.sym.as_ref() {
                                        (
                                            S::collect_scalar_state_field_values(sym),
                                            S::collect_ref_scalar_state_field_values(sym),
                                        )
                                    } else {
                                        (Vec::new(), Vec::new())
                                    }
                                } else {
                                    (scalar_values, ref_scalar_values)
                                };
                            self.meta.pending_abort_blackhole = Some(PendingAbortBlackhole {
                                framestack,
                                scalar_values,
                                ref_scalar_values,
                                virt_array_values,
                                virtualizable_ptr,
                                // `blackhole.py:1811-1814` reads
                                // `metainterp.last_exc_value` at conversion time;
                                // snapshot it here because `abort_trace_live` below
                                // tears down the tracing state it belongs to.
                                last_exc_value: self.meta.last_exc_value,
                                raising_exception: abort_raising_exception,
                            });
                        }
                    }
                    self.meta.abort_trace_live(false);
                    // `MetaInterp._interpret` counts every
                    // `SwitchToBlackhole` abort, including a bridge whose
                    // greenkey is None.
                    self.meta.aborted_tracing(reason_int);
                    self.sym = None;
                    self.bridge_attempt_declined = false;
                    self.meta.clear_trace_session();
                }
                TraceAction::AbortPermanent => {
                    // `pyjitpl.py aborted_tracing` does not set
                    // `JC_DONT_TRACE_HERE`. The only upstream writer of that
                    // flag from an abort is `blackhole_if_trace_too_long`
                    // ABORT_TOO_LONG, which names the huge function via
                    // `disable_noninlinable_function`. This arm is a
                    // `SwitchToBlackhole` whose unwind is
                    // `run_blackhole_interp_to_cancel_tracing` →
                    // `aborted_tracing(stb.reason)` — same bookkeeping as
                    // Abort. Passing `permanent=true` would stamp the flag
                    // through `abort_tracing`, and `can_inline_callable`
                    // (`warmstate.py`) reads it, so one unrecordable opcode
                    // would retire the function from inlining everywhere.
                    let is_bridge = self.meta.bridge_info().is_some();
                    if is_bridge {
                        crate::mc_diag_bump(82); // bridge_abort_permanent
                        crate::debug::log_one("jit-abort", "AbortPermanent during bridge tracing");
                    }
                    if self.bridge_attempt_declined {
                        crate::mc_diag_bump(52); // abort_after_declined
                    }
                    self.meta.abort_trace(false);
                    self.sym = None;
                    // The session ends here, so the latch must not outlive it.
                    // This arm does not share the Abort/Decline teardown, so it
                    // clears the latch itself.
                    self.bridge_attempt_declined = false;
                    self.meta.clear_trace_session();
                }
            }
            break;
        }
    }

    /// `pyjitpl.py` `reached_loop_header` after `compile_trace` returns
    /// without raising.
    ///
    /// The reverse scan (`reached_loop_header`) either finds a prior
    /// same-greenkey merge point — the caller then runs `compile_loop`,
    /// which gives up when that key still `has_compiled_targets` — or
    /// appends `(live_arg_boxes, start)` and keeps tracing. The next header
    /// visit retries `compile_trace` (`if not self.partial_trace`).
    ///
    /// Returns `true` when this visit appended and the walk should continue.
    fn keep_tracing_after_declined_jump(
        &mut self,
        target_key: u64,
        live_arg_boxes: &[OpRef],
    ) -> bool {
        let (header_pc, already, key_typed) = {
            let Some(ctx) = self.meta.trace_ctx() else {
                return false;
            };
            // pyjitpl.py same_greenkey: the green key is the identity.
            // `header_pc` recorded below is this close's guest pc.
            let key_typed = ctx
                .close_green_key()
                .filter(|key| key.get_uhash() == target_key);
            (
                ctx.close_header_pc(),
                ctx.find_merge_point_same_greenkey(target_key, key_typed.as_ref())
                    .is_some(),
                key_typed,
            )
        };
        if already {
            return false;
        }
        let green_boxes: Vec<crate::trace_ctx::GreenBox> = live_arg_boxes
            .iter()
            .map(|&op| crate::trace_ctx::GreenBox::new(op, op.ty().unwrap_or(majit_ir::Type::Void)))
            .collect();
        {
            let Some(ctx) = self.meta.trace_ctx() else {
                return false;
            };
            ctx.add_merge_point_with_key(target_key, key_typed, green_boxes, header_pc);
            // `saved_pc`: the resumed walk executes the
            // merge-point instruction once instead of closing again with
            // nothing recorded in between.
            ctx.resume_walk_after_close();
            ctx.close_greens = None;
            ctx.close_green_pc = None;
        }
        // The handoff staged for a trace that ENDED must not publish: the
        // walk is still recording.
        self.meta.single_pass_outcome = None;
        self.meta.single_pass_scalar_values = None;
        self.meta.single_pass_ref_scalar_values = None;
        self.meta.single_pass_virt_array_values = None;
        true
    }

    /// pyjitpl.py `blackhole_if_trace_too_long` + pyjitpl.py
    /// `aborted_tracing(reason)` parity shape.
    ///
    /// Runs the upstream bookkeeping, unwinds the live tracing state via
    /// `abort_trace_live`, then fires `aborted_tracing(reason)` for the
    /// accounting event — all exactly once — and clears driver-owned
    /// per-session state (sym / trace_meta / bridge_info).  Returns
    /// `true` when an abort fired so callers can act on it.
    #[inline]
    pub fn blackhole_if_trace_too_long(&mut self) -> bool {
        let Some(reason) = self.meta.blackhole_if_trace_too_long() else {
            return false;
        };
        self.meta.abort_trace_live(false);
        self.meta.aborted_tracing(reason.as_int());
        self.clear_tracing_session_state();
        true
    }

    #[cold]
    #[inline(never)]
    fn clear_tracing_session_state(&mut self) {
        self.sym = None;
        self.bridge_attempt_declined = false;
        self.meta.clear_trace_session();
    }

    /// Back-edge handler — call when a backward jump is detected.
    ///
    /// Handles hot counting, trace start, and compiled code execution.
    /// Returns `true` if compiled code ran successfully and the interpreter
    /// should jump to `target_pc` with restored state.
    ///
    /// `pre_run` is called before executing compiled code (e.g., to flush
    /// buffered I/O).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if target <= pc {
    ///     if driver.back_edge(target, &mut state, program, || writer.flush()) {
    ///         pc = target;
    ///         continue;
    ///     }
    /// }
    /// ```
    /// Returns `Some(resume)` when compiled code ran (FINISH or guard
    /// failure with blackhole resume). `PortalResume::resume_pc` is the
    /// position; the six lists are `jitexc.py ContinueRunningNormally`.
    /// Returns `None` otherwise.
    pub fn back_edge(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<PortalResume> {
        if let Some(handled) =
            self.try_empty_chain_tick(target_pc as u64, None, target_pc, state, env)
        {
            return handled.map(continue_at_pc);
        }
        self.back_edge_internal(
            target_pc as u64,
            None,
            None,
            target_pc,
            state,
            env,
            None,
            None,
            pre_run,
        )
    }

    pub fn back_edge_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<PortalResume> {
        if let Some(handled) = self.try_empty_chain_tick(green_key, None, target_pc, state, env) {
            return handled.map(continue_at_pc);
        }
        self.back_edge_internal(
            green_key, None, None, target_pc, state, env, None, None, pre_run,
        )
    }

    /// Takes the green key's hash eagerly and the key itself lazily.
    ///
    /// Only the hash is needed unconditionally — it selects the cell and
    /// answers `has_compiled_loop`. The typed [`GreenKey`] is read only by the
    /// cell-resolution/decision path and, on a tracing start, by the tracing
    /// commit. Both are reached only after `back_edge_internal`'s early returns
    /// (already tracing, `!can_trace`, cross-loop-cut decline, and the whole
    /// `has_compiled_loop` branch — i.e. every entry into compiled code).
    /// Building it eagerly spent the key's `values` / `types` vectors on those
    /// paths and dropped them unread.
    ///
    /// This is upstream's split: `get_uhash(*greenargs)` hashes the greens in
    /// place on every back edge, and the greens are stored on the cell only
    /// when one is installed (warmstate.py).
    ///
    /// Not `#[cold]`: this is the interpreter's per-iteration door, and the
    /// empty-chain arm (`warmstate.py`) is the common path. The
    /// compiled-entry / tracing-start bodies stay in `back_edge_internal`,
    /// which is `#[inline(never)]` so they do not land in the dispatch loop.
    #[inline]
    pub fn back_edge_structured(
        &mut self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<PortalResume> {
        if let Some(handled) =
            self.try_empty_chain_tick(green_key_hash, Some(&make_green_key), target_pc, state, env)
        {
            return handled.map(continue_at_pc);
        }
        self.back_edge_internal(
            green_key_hash,
            Some(&make_green_key),
            None,
            target_pc,
            state,
            env,
            None,
            None,
            pre_run,
        )
    }

    /// [`Self::back_edge_structured`] for a caller that has already resolved
    /// the cell and read its token — the resolve happens once, at that caller,
    /// and its result travels here instead of being recomputed.
    ///
    /// `warmstate.py` is this shape: `maybe_compile_and_run` reads the
    /// token at `:483`, decides on it, and passes that object to the executor
    /// as an argument (`raise EnterJitAssembler(procedure_token,
    /// *execute_args)`) rather than handing on a key for the executor to look
    /// up again. A door that instead probes and then calls the hash-taking
    /// entry point pays for two cell resolutions per warm call and, on a
    /// chained bucket, can decide about one cell's token and enter another's.
    ///
    /// `cell_key` must be a [`Self::resolve_cell_key`] result and
    /// `procedure_token` must be the token read from THAT key — normally
    /// [`Self::runnable_procedure_token`] asked with this same `cell_key`.
    /// Neither is re-derived here: the key is used as-is (re-resolving a
    /// resolved key would also rebuild the caller's `GreenKey` on a chained
    /// bucket, which is the allocation the carry exists to avoid), and the
    /// token is entered as given.
    ///
    /// Unannotated, as `warmstate.py maybe_compile_and_run` is: the body is
    /// a call to `back_edge_internal`, which stays out of line.
    pub fn back_edge_resolved(
        &mut self,
        cell_key: u64,
        procedure_token: std::sync::Arc<majit_backend::JitCellToken>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<PortalResume> {
        self.back_edge_internal(
            cell_key,
            None,
            Some(procedure_token),
            target_pc,
            state,
            env,
            None,
            None,
            pre_run,
        )
    }

    pub fn back_edge_or_run_compiled(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(
            target_pc as u64,
            None,
            target_pc,
            state,
            env,
            pre_run,
        )
    }

    pub fn back_edge_or_run_compiled_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(green_key, None, target_pc, state, env, pre_run)
    }

    /// RPython-style jit_merge_point dispatch hook.
    ///
    /// Called from the outer interpreter dispatch loop on every iteration.
    /// This hook only feeds the active trace. Warmup counting, tracing start,
    /// and compiled-code entry belong to `can_enter_jit`/function-entry paths.
    pub fn jit_merge_point_keyed<F>(
        &mut self,
        _green_key: u64,
        target_pc: usize,
        _state: &mut S,
        _env: &S::Env,
        _pre_run: impl FnOnce(),
        trace_fn: F,
    ) -> Option<DetailedDriverRunOutcome>
    where
        F: FnMut(&mut MetaInterp<S::Meta>, &mut S::Sym) -> TraceAction,
    {
        // RPython JC_TRACING parity: in RPython, MetaInterp._interpret()
        // runs all bytecodes within the same portal call. In pyre,
        // eval_loop_jit calls jit_merge_point at each bytecode, so the
        // green_key changes per bytecode (includes pc). Use is_tracing()
        // (not is_tracing_key) to accept all bytecodes from the tracing
        // portal. Nested function calls are prevented by JIT_TRACING
        // flag in eval_loop_jit.
        if self.meta.is_tracing() && !self.meta.trace_continuation_suspended() {
            // pyjitpl.py: record frame.pc for capture_resumedata.
            if let Some(ctx) = self.meta.trace_ctx() {
                ctx.last_traced_pc = target_pc;
            }
            self.merge_point(trace_fn);
            if self.take_compile_trace_success() {
                let (continue_running_normally_values, continue_running_normally_pc) = self
                    .take_continue_running_normally_payload()
                    .map_or((None, None), |(values, pc)| (Some(values), pc));
                // Re-observing a success the CompileTrace arm already tore down
                // (compile_trace_success flag). Live teardown only — the
                // accounting must not fire (it would double-count a single
                // successful compile as two aborts).
                self.meta.abort_trace_live(false);
                self.meta.clear_pending_abort();
                self.sym = None;
                self.meta.clear_trace_session();
                return Some(DetailedDriverRunOutcome::Jump {
                    via_blackhole: false,
                    continue_running_normally_values,
                    continue_running_normally_pc,
                });
            }
        }
        None
    }

    /// RPython-style can_enter_jit path.
    ///
    /// Called from a back-edge. Unlike `jit_merge_point_keyed`, this path
    /// never starts tracing and only runs already-compiled code.
    pub fn can_enter_jit_keyed(
        &mut self,
        green_key: u64,
        _target_pc: usize,
        state: &mut S,
        _env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        if self.meta.is_tracing() {
            return None;
        }
        let single_pass_dispatch_key =
            self.take_single_pass_label_entry_dispatch_key_for_back_edge(green_key);
        if !state.can_trace() {
            return None;
        }
        // `warmstate.py:471-478`: a temporary token is not entered. Reporting
        // `None` rather than an `Abort` outcome keeps the answer the same shape
        // as the "no compiled code at all" case, which is what a tmp-only cell
        // is from this path's point of view.
        if self.has_confirmed_runnable_compiled_loop(green_key) {
            return Some(self.run_compiled_detailed_keyed_with_dispatch_key(
                green_key,
                state,
                pre_run,
                single_pass_dispatch_key,
            ));
        }
        None
    }

    pub fn back_edge_declarative<D: DeclarativeJitDriver>(
        &mut self,
        green_values: &[i64],
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Result<Option<usize>, &'static str> {
        let green_key = D::green_key(green_values)?;
        Ok(self
            .back_edge_structured(
                green_key.hash_u64(),
                || green_key.clone(),
                target_pc,
                state,
                env,
                pre_run,
            )
            .and_then(|resume| self.resume_pc_recycling(resume)))
    }

    /// Whether this guard's resume stream carries deferred heap writes over
    /// more than one virtual.
    ///
    /// Read straight off the guard's own exit layout — the same lookup
    /// `start_retrace_from_guard` makes for `retrace.storage`, but reachable
    /// before a trace session exists, so a decline that turns on it does not
    /// have to open one and tear it down.
    ///
    /// The virtual count is the whole question: a guard carrying deferred
    /// writes over one virtual is served correctly and one carrying two is
    /// not. See the caller for what separates them.
    fn guard_carries_pending_fields(
        &self,
        descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    ) -> bool {
        // `ResumeGuardDescr.get_resumestorage()`: the payload is the descr's.
        crate::resume::get_resumestorage(descr_arc).is_some_and(|storage| {
            !storage.rd_pendingfields().is_none_or(<[_]>::is_empty)
                && storage.rd_virtuals().map_or(0, <[_]>::len) > 1
        })
    }

    /// `compile.py handle_fail`'s bridging arm.
    ///
    /// Upstream's `handle_fail` is an if/else: a guard that `must_compile()`
    /// traces a bridge, and one that does not resumes in the blackhole. Never
    /// both. Tracing starts at the guard itself —
    /// `initialize_state_from_guard_failure` → `rebuild_state_after_failure`
    /// rebuilds the frames and `setup_resume_at_op(pc)` puts each back at its
    /// own position — so the rest of the opcode the guard sits inside is
    /// recorded into the bridge.
    ///
    /// A decline means the blackhole resumes with no bridge. The blackhole
    /// stops at the next merge point and returns that pc, and nothing records
    /// a trace from there.
    ///
    /// Returns the interpreter position to resume at, or `None` when the
    /// bridge could not be entered AT ALL — the caller then takes the
    /// blackhole arm, which is upstream's other branch. Every decline is
    /// therefore sited BEFORE the walk: once the walk has run, the tail has
    /// happened, and handing the same guard to the blackhole would apply it a
    /// second time.
    fn bridge_from_guard_resume_position(
        &mut self,
        descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
        state: &mut S,
        env: &S::Env,
        raw_values: &[i64],
        target_pc: usize,
    ) -> Option<PortalResume> {
        use majit_ir::resumedata::RebuiltValue;
        // Only a state whose machine IS a dispatch jitcode has a guard
        // position to re-enter at; every other JitState resumes through its
        // own frontend.
        let dispatch = self.dispatch_jitcode().cloned()?;
        // The cut below is OFF by default and upstream has no equivalent:
        // `compile.py ResumeGuardDescr.handle_fail` bridges whenever
        // `must_compile` fires, whatever the guard's deferred writes look
        // like. What the guard's pendingfields once threatened was a double
        // application — the entry's applying reader storing them on its way
        // in and the blackhole arm this decline hands the guard to storing
        // them again — and that is answered by SITING the decision ahead of
        // `start_bridge_tracing`, which is where it now is, not by refusing.
        // A guard admitted here reaches exactly one of the two readers.
        //
        // The wrong answers that later narrowed the cut to `nv > 1` were
        // never this entry's either: they are made at the CLOSE, by a loop
        // whose single LABEL its own closing JUMP specialized, and
        // `compile_loop` now declines to compile that shape rather than
        // publishing a label nothing can enter soundly (see
        // `MetaInterp::closing_jump_fixes_label_slots`).
        //
        // Sited here rather than in the ladder below because
        // `compile.py ResumeGuardDescr.handle_fail` runs ONE of resume.py's
        // two readers per guard and never both: `start_bridge_tracing` asks
        // for the applying one, which stores the deferred writes on its way
        // in, and the blackhole arm this decline hands the guard to then
        // stores them a second time.
        // A prior walk's `TraceAction::Finish` leaves `single_pass_finish`
        // standing when the portal runner drained `back_edge_finish_word`
        // and never took the flag. A JUMP close that reuses it reports
        // `DoneWithThisFrame` with no result and the runner re-enters at
        // pc 0. The Finish arm also publishes the word latch for the Run
        // drain; leaving that standing returns the prior resultbox on the
        // next portal call.
        self.clear_finish_latches();
        if no_guard_resume_bridge_enabled() {
            return None;
        }
        if guard_resume_refuse_pending_fields() && self.guard_carries_pending_fields(descr_arc) {
            let why = GuardResumeDecline::ReplayIncomplete;
            crate::mc_diag_bump(why.diag_slot());
            if crate::majit_log_enabled() {
                eprintln!("[bridge] guard-resume entry declined: {}", why.label());
            }
            return None;
        }
        // No blackhole runs before this entry, so its replay is the one that
        // owes the guard's deferred writes to the heap.
        if !self.start_bridge_tracing(descr_arc, state, env, raw_values, target_pc, true, &[]) {
            return None;
        }
        // From here the trace session is LIVE, so a decline has to tear it
        // down rather than return through it.
        type Seeded = Vec<crate::jit_state::GuardResumeFrame>;
        let seeded = (|| -> Result<Seeded, GuardResumeDecline> {
            use GuardResumeDecline as Decline;
            let resume = self
                .resume_data_result
                .as_ref()
                .ok_or(Decline::NoResumeState)?;
            // The slot the decline arm bumps names the rung; a rung this walk
            // can refuse at any depth needs the frame it refused at too, so
            // every decline below also logs. Same policy, and the same reason,
            // as `read_frame_liveness_reg_indices`: a bridge that is never
            // built looks exactly like a bridge that was never attempted, and
            // the difference is the whole diagnosis.
            if crate::bridge_debug_enabled() {
                eprintln!(
                    "[bridgeB] rebuild {} frame(s): {:?}",
                    resume.frames.len(),
                    resume
                        .frames
                        .iter()
                        .map(|f| (f.jitcode_index, f.pc, f.values.len()))
                        .collect::<Vec<_>>(),
                );
            }
            // A frame below the root has had its reserved identity-slot
            // registers blanked to `Const(0)` by the snapshot trim
            // (`get_list_of_active_snapshot_boxes`), and nothing re-derives
            // them: `resume.py write_an_int` writes each section's decoded
            // values, zeros included, into that frame's own bank, and the only
            // root-to-callee propagation deopt performs is on the
            // virtualizable FIELDS, never the register file. The root's
            // registers are not a substitute either — they hold a different
            // quantity at the same index, which the trim's own note records
            // for dualtape (identity in a sub-frame's int reg 2, the scalar
            // `pb` in the root's).
            //
            // So refuse, and refuse HERE: this rung's decline leaves through
            // the same arm as every other one, which returns `None` from
            // `bridge_from_guard_resume_position` and so falls through to
            // `compile.py:711 resume_in_blackhole`, which rebuilds every frame
            // itself. That is what main did for every multi-frame resume, and
            // it is the upstream shape — `pyjitpl.py _handle_guard_failure`
            // reaches `compile.py giveup()` once the frames are known. An
            // abort from inside the walk would NOT do: the walk publishes its
            // handoff only after `run_to_end`, so an early return leaves both
            // handoffs empty and the caller resumes at the loop header with no
            // blackhole at all. The walk has not started here — `merge_point`
            // is below — so the teardown the arm performs is of the trace
            // session `start_bridge_tracing` opened, not of a walk in flight.
            //
            // `None` — every symbol that has not opted in via `split_dispatch`
            // — reserves no identity-only register, so nothing was blanked and
            // this refuses nothing.
            if resume.frames.len() > 1
                && let Some((base, end)) = S::reserved_int_identity_range()
            {
                if crate::bridge_debug_enabled() {
                    eprintln!(
                        "[bridgeB] DECLINE {} frames with reserved identity slots [{base}, {end}): \
                         the snapshot blanked them and nothing re-derives them",
                        resume.frames.len(),
                    );
                }
                return Err(Decline::ReservedIdentitySlots);
            }
            // `resume.py` `jitcode = jitcodes[jitcode_pos]`: the position
            // is an offset into the jitcode the frame NAMES. Entering a
            // different one at that offset lands mid-instruction in unrelated
            // code, and the walk decodes whatever byte is there. The root frame
            // is the one this driver re-enters at, so a root that names any
            // jitcode other than the dispatch one is a decline.
            let dispatch_index = dispatch.try_index().ok_or(Decline::NoResumeState)?;
            let root_index = usize::try_from(
                resume
                    .frames
                    .first()
                    .ok_or(Decline::NoResumeState)?
                    .jitcode_index,
            )
            .map_err(|_| Decline::ForeignJitcode)?;
            if root_index != dispatch_index {
                // Unlike the pending-fields decline this one cannot be
                // hoisted: the frame it reads only exists once
                // `rebuild_from_resumedata` has run, which is inside the call
                // this rung is downstream of. A guard declined here therefore
                // has had its virtuals allocated and its stores applied once
                // before the blackhole applies them again — the re-application
                // is idempotent (the stream carries the values, so neither
                // pass reads what the other wrote) and the first set of
                // virtuals becomes garbage.
                if crate::bridge_debug_enabled() {
                    eprintln!("[bridgeB] DECLINE root frame does not name the dispatch jitcode");
                }
                return Err(Decline::ForeignJitcode);
            }
            // `resume.py _prepare_pendingfields` replays the guard's deferred
            // heap writes through `execute_and_record`, which applies them to
            // the heap AND puts them in the trace. This entry asked for that
            // applying reader, so a guard carrying deferred writes is served
            // here rather than declined — but only as far as the reader has an
            // applying twin for what it meets. Where it did not, it recorded a
            // write it could not perform, and resuming would run against a heap
            // that half-describes what the trace says.
            //
            // Decline exactly there. The blackhole arm applies all of them,
            // which is where these guards were served before.
            if self
                .meta
                .tracing
                .as_ref()
                .is_some_and(|ctx| ctx.bridge_replay_incomplete())
            {
                if crate::bridge_debug_enabled() {
                    eprintln!("[bridgeB] DECLINE the guard carries pending fields");
                }
                return Err(Decline::ReplayIncomplete);
            }
            let fail_types = resume.fail_arg_types.clone();
            // `resume.py:1051` `jitcodes[jitcode_pos]` — the same flat registry
            // the blackhole rebuild resolves through, so a frame names its
            // jitcode absolutely and no walk of a parent's `descrs` is needed
            // to find it. Taken owned because the seeding below borrows
            // `self.meta.tracing` mutably to intern this bridge's constants.
            let jitcodes = self.meta.jitcodes().to_vec();
            let op_live = self.meta.staticdata.op_live as u8;
            let all_liveness = self.meta.staticdata.liveness_info.snapshot_arc();
            // One entry per encoded resume section, in the stream's own order.
            // `opencoder.py SnapshotIterator.__init__` reverses on the WRITER
            // side, so the sections are already caller-first and
            // `rebuild_from_resumedata` just appends them as it reads.
            let mut sections: Vec<(std::sync::Arc<crate::jitcode::JitCode>, usize)> =
                Vec::with_capacity(resume.frames.len());
            for frame in &resume.frames {
                let Some(index) = usize::try_from(frame.jitcode_index).ok() else {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE negative jitcode_index={}",
                            frame.jitcode_index
                        );
                    }
                    return Err(Decline::UnregisteredJitcode);
                };
                let Some(jitcode) = jitcodes.get(index) else {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE unregistered jitcode_index={}",
                            frame.jitcode_index
                        );
                    }
                    return Err(Decline::UnregisteredJitcode);
                };
                let pc = usize::try_from(frame.pc).map_err(|_| Decline::NoResumeState)?;
                sections.push((jitcode.clone(), pc));
            }
            // A frame below the top is suspended inside a `BC_INLINE_CALL`
            // whose operands say which callee it is waiting on and which of its
            // own registers takes the result. Neither is in the resume stream —
            // `opencoder.py _ensure_parent_resumedata` blanks the result
            // register before reading a parent's liveness — so recover them the
            // way `pyjitpl.py make_result_of_lastop` does, off the bytecode,
            // and check that the call found there names the frame that is
            // actually stacked on top of it.
            let mut call_sites = Vec::with_capacity(sections.len());
            for (depth, (jitcode, pc)) in sections.iter().enumerate() {
                let Some((callee, _)) = sections.get(depth + 1) else {
                    call_sites.push(None);
                    continue;
                };
                let Some(site) = jitcode.inline_call_ending_at(*pc) else {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE no inline call ending at depth={depth} pc={pc} jc={:?}",
                            jitcode.name()
                        );
                    }
                    return Err(Decline::NoInlineCallSite);
                };
                // `as_jitcode_owned`, not `as_jitcode`: a self-recursive
                // `#[jit_inline]` helper's back edge is a `Weak` descr, so the
                // owning variant alone misses exactly the recursion this walk
                // exists to resume through. `BC_INLINE_CALL` resolves its `j`
                // operand the same way.
                let Some(named) = jitcode
                    .descr_at(site.sub_idx)
                    .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode_owned)
                else {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE descrs[{}] is not a JitCode at depth={depth}",
                            site.sub_idx,
                        );
                    }
                    return Err(Decline::CallSiteNotAJitcode);
                };
                if named.try_index() != callee.try_index() {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE call at depth={depth} names {:?} not {:?}",
                            named.try_index(),
                            callee.try_index()
                        );
                    }
                    return Err(Decline::CallSiteCalleeMismatch);
                }
                // The pop reads the return kind off the callee's own trailing
                // return opcode and then requires the matching caller slot
                // (`dispatch.rs`, the inline-return arm). A callee that returns
                // a value into no slot would panic there, so decline here
                // instead — the blackhole arm rebuilds every frame either way.
                let returns = crate::jitcode::JitCodeRuntimeExt::trailing_return_info(&**callee)
                    .map(|(kind, _)| kind);
                let slot = site.result_slot();
                if returns.is_some() && slot.map(|(kind, _)| kind) != returns {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE depth={depth} callee returns {returns:?} but the call slot is {slot:?}"
                        );
                    }
                    return Err(Decline::CalleeReturnSlotMismatch);
                }
                call_sites.push(Some(site));
            }
            let allocator = self.blackhole_allocator.as_deref();
            let ctx = self.meta.tracing.as_mut().ok_or(Decline::NoResumeState)?;
            let virtual_count = resume
                .storage
                .as_ref()
                .map_or(0, |storage| storage.rd_virtuals().map_or(0, <[_]>::len));
            let mut virt_cache = match allocator {
                Some(allocator) => crate::BridgeVirtualCache::executing(
                    virtual_count,
                    crate::default_bridge_array_descr,
                    allocator,
                    raw_values,
                    &fail_types,
                ),
                None => {
                    crate::BridgeVirtualCache::new(virtual_count, crate::default_bridge_array_descr)
                }
            };
            let mut frames = Vec::with_capacity(sections.len());
            for (depth, (jitcode, pc)) in sections.into_iter().enumerate() {
                let values = &resume.frames[depth].values;
                // Every frame visits its own liveness at its own position,
                // which is what `ResumeDataBoxReader::consume_boxes` does per
                // `f.get_current_position_info()`: `enumerate_vars` calls one
                // bank callback per live index and stores nothing. Rebuilt
                // values follow that bank order (int, then ref, then float;
                // see `FrameLivenessRegIndices`), so one cursor pairs each
                // index with the next value. The root frame's materialized
                // indices stay on the trace ctx for `setup_bridge_sym`, which
                // is static and has no jitcode.
                //
                // The same two conditions `read_frame_liveness_reg_indices`
                // declines on produce empty banks here, and the count check
                // then declines unless `values` is empty too.
                let header = if !jitcode.can_decode_live_vars(pc, op_live) {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] read_frame_liveness_reg_indices: no liveness startpoint at pc={pc} op_live={op_live} — declining (empty banks)"
                        );
                    }
                    None
                } else {
                    let info = jitcode.get_live_vars_info(pc, op_live);
                    if info + 2 >= all_liveness.len() {
                        if crate::bridge_debug_enabled() {
                            eprintln!(
                                "[bridgeB] read_frame_liveness_reg_indices: liveness info {info} out of range (len={}) at pc={pc} — declining (empty banks)",
                                all_liveness.len()
                            );
                        }
                        None
                    } else {
                        Some((
                            all_liveness[info] as u32,
                            all_liveness[info + 1] as u32,
                            all_liveness[info + 2] as u32,
                            info,
                        ))
                    }
                };
                // `consume_boxes` fills every live register of the frame, so a
                // count that disagrees means the two sides read different
                // liveness and no per-register pairing off them is trustworthy.
                // `enumerate_vars` walks exactly `length_i + length_r + length_f`
                // indices, which is that count; the bytes alone decide it.
                let live_count = match header {
                    Some((length_i, length_r, length_f, _)) => {
                        length_i as usize + length_r as usize + length_f as usize
                    }
                    None => 0,
                };
                if live_count != values.len() {
                    if crate::bridge_debug_enabled() {
                        eprintln!(
                            "[bridgeB] DECLINE depth={depth} pc={pc} liveness {live_count} != values {}",
                            values.len()
                        );
                    }
                    return Err(Decline::RegCountMismatch);
                }
                if crate::bridge_debug_enabled() {
                    let shown = if header.is_some() {
                        crate::resume::read_frame_liveness_reg_indices(
                            &jitcode,
                            pc,
                            op_live,
                            &all_liveness,
                        )
                    } else {
                        crate::resume::FrameLivenessRegIndices::default()
                    };
                    eprintln!(
                        "[bridgeB] liveness pc={pc} i={:?} r={:?} f={:?}",
                        shown.int, shown.ref_, shown.float
                    );
                    let code = &jitcode.code;
                    let n = code.len().saturating_sub(pc).min(16);
                    eprint!("[bridgeB] code[{pc}..]=");
                    for b in &code[pc..pc + n] {
                        eprint!(" {b:02x}");
                    }
                    eprintln!();
                }
                let mut regs = Vec::with_capacity(values.len());
                if let Some((_, _, _, info)) = header {
                    let mut decline: Option<Decline> = None;
                    let mut cursor = 0usize;
                    {
                        let mut consume = |bank: majit_ir::Type, index: u32| {
                            if decline.is_some() {
                                return;
                            }
                            let (opref, value) = match &values[cursor] {
                                RebuiltValue::Box(n, kind) => crate::jit_state::bridge_decode_red(
                                    *n,
                                    *kind,
                                    raw_values,
                                    &fail_types,
                                ),
                                RebuiltValue::Const(c) => {
                                    let bits = c.as_raw_i64();
                                    let opref = match bank {
                                        majit_ir::Type::Ref => ctx.const_ref(bits),
                                        majit_ir::Type::Float => ctx.const_float(bits),
                                        _ => ctx.const_int(bits),
                                    };
                                    (opref, bits)
                                }
                                // resume.py consume_boxes → getvirtual: a virtual
                                // is allocated and the register holds the object.
                                // Unassigned still has nothing to execute.
                                RebuiltValue::Virtual(vidx) => {
                                    let rd_virtuals = resume
                                        .storage
                                        .as_ref()
                                        .map(|storage| storage.rd_virtuals().unwrap_or(&[]));
                                    let opref = crate::materialize_bridge_virtual(
                                        ctx,
                                        *vidx,
                                        rd_virtuals,
                                        resume,
                                        &mut virt_cache,
                                    );
                                    let bits = if opref.is_none() {
                                        None
                                    } else {
                                        virt_cache.concrete_root_of(opref).or_else(|| {
                                            ctx.box_value(opref).and_then(|value| match value {
                                                majit_ir::Value::Ref(r) => {
                                                    Some(r.as_usize() as i64)
                                                }
                                                majit_ir::Value::Int(i) => Some(i),
                                                majit_ir::Value::Float(f) => {
                                                    Some(f.to_bits() as i64)
                                                }
                                                majit_ir::Value::Void => None,
                                            })
                                        })
                                    };
                                    let Some(bits) = bits else {
                                        if crate::bridge_debug_enabled() {
                                            eprintln!(
                                                "[bridgeB] DECLINE depth={depth} virtual {vidx} \
                                                 has no concrete"
                                            );
                                        }
                                        decline = Some(Decline::UnreadableRegister);
                                        return;
                                    };
                                    (opref, bits)
                                }
                                RebuiltValue::Unassigned => {
                                    if crate::bridge_debug_enabled() {
                                        eprintln!(
                                            "[bridgeB] DECLINE depth={depth} unreadable slot {:?}",
                                            values[cursor]
                                        );
                                    }
                                    decline = Some(Decline::UnreadableRegister);
                                    return;
                                }
                            };
                            regs.push(crate::jit_state::GuardResumeReg {
                                bank,
                                index,
                                opref,
                                value,
                            });
                            cursor += 1;
                        };
                        majit_jitcode::codewriter::jitcode::enumerate_vars_by_bank(
                            info,
                            &all_liveness,
                            |bank, index| consume(bank, index),
                        );
                    }
                    if let Some(why) = decline {
                        return Err(why);
                    }
                }
                if crate::bridge_debug_enabled() {
                    eprint!("[bridgeB] seeded");
                    for reg in &regs {
                        eprint!(
                            " {:?}[{}]={:#x}/{:?}",
                            reg.bank, reg.index, reg.value, reg.opref
                        );
                    }
                    eprintln!();
                }
                frames.push(crate::jit_state::GuardResumeFrame {
                    jitcode,
                    pc,
                    regs,
                    result_slot: call_sites[depth].and_then(|site| site.result_slot()),
                    // The `descrs` slot the call that pushed THIS frame named —
                    // it lives on the caller's site, one level out.
                    sub_idx: depth
                        .checked_sub(1)
                        .and_then(|caller| call_sites[caller])
                        .map(|site| site.sub_idx),
                });
            }
            Ok(frames)
        })();
        let mut frames = match seeded {
            Ok(frames) => frames,
            Err(why) => {
                crate::mc_diag_bump(why.diag_slot());
                if crate::majit_log_enabled() {
                    eprintln!("[bridge] guard-resume entry declined: {}", why.label());
                }
                // The ladder above already decided; it is the abort that cannot
                // see which rung chose. Name the reason here so the
                // `abort_trace` below tallies a bridge given up rather than
                // falling into the unclassified default, which shares its id.
                self.meta.stage_abort_reason(crate::counters::ABORT_BRIDGE);
                self.meta.abort_trace(false);
                self.clear_tracing_session_state();
                self.resume_data_result = None;
                self.last_bridge_is_exception_guard = false;
                return None;
            }
        };

        // Generated `setup_bridge_sym` rebinds a missing identity register
        // from the live state. Grain's two reds are those identities and
        // the resume stream can still name a walk-local address; rewrite
        // the concrete shadows before the walk residual-calls through them.
        state.rebind_bridge_reds_from_fail(&mut frames, raw_values);

        self.bridge_entered_at_guard_resume = true;
        let mut entered = false;
        self.merge_point(|meta, sym| {
            // `merge_point` re-enters its closure for the `saved_pc`
            // resumption a declined close asks for. The guard's position is a
            // one-off; a resumption is an ordinary merge-point entry, which
            // the `jit_merge_point!` wrapper owns. Returning here leaves
            // tracing live and hands it over.
            if std::mem::replace(&mut entered, true) {
                return TraceAction::Continue;
            }
            meta.with_trace_ctx_and_token_resolver(
                |ctx,
                 resolve,
                 rec_target,
                 rec_decision,
                 rec_exec,
                 rec_exec_ref,
                 rec_exec_float,
                 rec_exec_void| {
                    let runtime = crate::pyjitpl::ClosureRuntimeWithResolver::new(
                        |pc: usize| pc,
                        resolve,
                        rec_target,
                        rec_decision,
                        rec_exec,
                        rec_exec_ref,
                        rec_exec_float,
                        rec_exec_void,
                    );
                    S::trace_from_guard_resume_position(ctx, sym, &frames, target_pc, &runtime)
                        .unwrap_or(TraceAction::Abort)
                },
            )
            .unwrap_or(TraceAction::Abort)
        });
        self.bridge_entered_at_guard_resume = false;

        // Where the interpreter goes next. This is the `jit_merge_point!`
        // hook's own single-pass handoff, in the same order and with the same
        // steps: the walk advanced the state on the sym, so whichever arm
        // answers has to push it into native `state` before the interpreter
        // reads it. The hook runs for a walk that ended at a merge point; a
        // walk that started at a guard ends the same way and owes the same
        // transfer.
        let outcome = self.take_single_pass_outcome();
        // `pyjitpl.py MetaInterp.run_blackhole_interp_to_cancel_tracing`: the walk stopped
        // somewhere that is not a source-opcode boundary, so the half-executed
        // opcodes are finished in the blackhole and the position comes from
        // the merge point they reach. It outranks the walk's own pc for the
        // same reason it does in the hook. The abort's
        // `ContinueRunningNormally` banks replace the walk's close key:
        // `warmspot.py handle_jitexception` reads the merge point the
        // blackhole reached, not the walk that aborted.
        if let Some(resume) = self.run_pending_abort_blackhole(state, env) {
            let _ = outcome;
            self.writeback_scalar_state_fields(state);
            self.writeback_ref_scalar_state_fields(state);
            self.writeback_virt_array_state_fields(state);
            state.recover_after_compiled_run();
            self.log_single_pass_parity_resume(
                resume.resume_pc().unwrap_or(usize::MAX),
                state,
                env,
            );
            self.arm_single_pass_label_entry_on_next_back_edge(state);
            self.discard_single_pass_resume();
            return Some(resume);
        }
        if let Some((pc, reds, args)) = outcome {
            self.writeback_scalar_state_fields(state);
            self.writeback_ref_scalar_state_fields(state);
            self.writeback_virt_array_state_fields(state);
            if !reds.is_empty() {
                let resumed_meta = state.build_meta(pc, env);
                state.restore_values(&resumed_meta, &reds);
            }
            state.recover_after_compiled_run();
            self.log_single_pass_parity_resume(pc, state, env);
            self.arm_single_pass_label_entry_on_next_back_edge(state);
            self.discard_single_pass_resume();
            // A terminal dispatch return means the interpreted function has
            // returned, so the native dispatch loop must EXIT rather than
            // resume at a position. `single_pass_finish` is how that is said,
            // and it is deliberately left standing here for the caller to
            // consume: the `#[jit_interp]` expansion of both markers reads it
            // straight after the call and `break`s on it, which is the only
            // exit a dispatch loop is required to have.
            // Greens captured before the finish still go back: the epilogue
            // reads them after the break.
            let resume = if self.meta.single_pass_finish
                && (self.meta.single_pass_finish_values.is_some()
                    || self.meta.back_edge_finish_word.is_some())
            {
                PortalResume::DoneWithThisFrame(Some(args))
            } else {
                continue_with_args(args)
            };
            return Some(resume);
        }
        // Neither walk handoff answered. `_trace_and_compile_from_bridge`
        // ends in `ContinueRunningNormally` / `DoneWithThisFrame*` /
        // `ExitFrameWithExceptionRef`; the `must_compile` else-arm is
        // `resume_in_blackhole` and runs only when tracing never started.
        // `ResumeAt` would skip the loop-carried greens
        // `warmspot.py handle_jitexception` assigns.
        crate::mc_diag_bump(79); // guard_resume_bridge_no_handoff
        if let Some(args) = self.meta.trace_ctx().map(|ctx| ctx.portal_resume_args()) {
            state.recover_after_compiled_run();
            return Some(continue_with_args(args));
        }
        // `ResumeGuardDescr.handle_fail` / `resume_in_blackhole` from the
        // caller's deadframe. Reached only when `merge_point` never invoked
        // the walk (`entered` stays false): a live `TraceCtx` with no
        // dispatch jitcode already returned above, and every
        // `TraceAction::Abort` / `SwitchToBlackhole` after
        // `start_bridge_tracing` stages `pending_abort_blackhole` for
        // `MetaInterp.aborted_tracing` → `convert_and_run_from_pyjitpl`.
        debug_assert!(
            !entered,
            "guard-resume walk started; MetaInterp.aborted_tracing must convert \
             the live framestack (blackhole.py convert_and_run_from_pyjitpl) \
             rather than ResumeGuardDescr.handle_fail resume_in_blackhole"
        );
        if crate::majit_log_enabled() {
            eprintln!(
                "[bridge] no-handoff: compile_trace_success={} single_pass_finish={} \
                 interpret_abort={:?} → resume_in_blackhole",
                self.compile_trace_success,
                self.meta.single_pass_finish,
                self.meta.last_interpret_abort_reason,
            );
        }
        state.recover_after_compiled_run();
        None
    }

    /// `warmstate.py` `maybe_compile_and_run`, not-found arm: hash already
    /// in hand, `lookup_chain` once, and if this hash owns no enterable
    /// procedure token, `jitcounter.tick` and return. A compiled token, a
    /// chained bucket, or a `JC_DONT_TRACE_HERE` cell this hash owns
    /// continues into `back_edge_internal`.
    ///
    /// `Some(resume)` means this IS that cold case — including `Some(None)`
    /// when the tick did not overflow.
    #[inline(always)]
    fn try_empty_chain_tick(
        &mut self,
        green_key_hash: u64,
        structured_green_key: Option<&dyn Fn() -> GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> Option<Option<usize>> {
        if self.meta.is_tracing() {
            return Some(None);
        }
        // Cross-loop-cut decline and single-pass label handoff must stay
        // ahead of the counter (`back_edge_internal` cites why). Empty
        // tables are O(1) and allocation-free: check emptiness first so a
        // never-compiled loop does not hash into them.
        if !self.meta.cut_compiled_keys.is_empty() {
            return None;
        }
        // A chained bucket needs comparekey and stays on `back_edge_internal`.
        // A lone cell is the tick arm only when this hash owns it.
        let jd_no = self.index().unwrap_or(0);
        if let Some(cell) = self
            .meta
            .warm_state_ref_for_driver(jd_no)
            .and_then(|warm| warm.lookup_chain(green_key_hash))
        {
            if cell.next.borrow().is_some() {
                return None;
            }
            // `lookup_chain` returns the table-slot head. `_get_index` keeps
            // only the high bits of the low 32, so that head can belong to a
            // different green key. Its compiled / tracing / dead-token /
            // abort-ceiling / `JC_DONT_TRACE_HERE` state is not ours: the
            // not-found arm of `maybe_compile_and_run` ticks this hash.
            if cell.cell_bucket.get() == green_key_hash {
                if cell.is_compiled() {
                    return None;
                }
                if cell.is_tracing() {
                    return Some(None);
                }
                // Dead-token and other tokenless cells take `cleanup_chain`
                // on the occupied door (`maybe_compile_and_run`).
                if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
                    return None;
                }
                if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
                    // `should_start_dont_trace_here_trace`: never traced
                    // starts immediately, `JC_TRACING_OCCURRED` ticks. Both
                    // stay on the occupied door.
                    return None;
                }
            }
        }
        if !state.can_trace() {
            return Some(None);
        }
        if self
            .meta
            .warm_state_for_driver(jd_no)
            .tick_empty_chain(green_key_hash)
            && self.commit_start_tracing(
                green_key_hash,
                structured_green_key,
                target_pc,
                state,
                env,
            )
        {
            return Some(Some(target_pc));
        }
        Some(None)
    }

    /// RPython warmstate.py:482-501 / compile.py:711 parity.
    ///
    /// Returns `Some(pc)` when compiled code ran:
    /// - FINISH → `Some(target_pc)` (loop completed, re-enter at header)
    /// - Guard failure → `Some(resume_pc)` (resume from failure point)
    ///   Returns `None` when no compiled code exists or tracing started.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    #[inline(never)]
    fn back_edge_internal(
        &mut self,
        green_key_hash: u64,
        structured_green_key: Option<&dyn Fn() -> GreenKey>,
        carried_procedure_token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        dispatch_key: Option<u32>,
        direct_live_values: Option<Vec<Value>>,
        pre_run: impl FnOnce(),
    ) -> Option<PortalResume> {
        if self.meta.is_tracing() {
            return None;
        }
        // **Resolve once, then carry.** The parameter is a raw bucket hash;
        // from here down `green_key` names ONE cell, not a celltable bucket —
        // the decision below, the
        // `compiled_loops` meta the runner executes, the invalidation on a
        // shape mismatch and the front-target tables all read through this one
        // number, so they cannot name different cells the way a bucket hash
        // let them (`warmstate.py:458-464` resolves once and :483/:511 carries
        // the resolved token onward for the same reason). Identity for every
        // unchained bucket, so the collision-free path is unchanged.
        //
        // A caller arriving with `carried_procedure_token` has already done
        // this step and read its token from the result, so `green_key_hash` is
        // that resolved key and is taken as-is: resolving a resolved key would
        // rebuild the caller's `GreenKey` on a chained bucket, and the whole
        // point of the carry is that the key, the token and the run come from
        // one resolution.
        //
        // The cell-owned half of the entry decision is taken from that same
        // resolution rather than from later walks back to the cell it already
        // found: `resolve_cell_key`, the token read and the `JC_TEMPORARY` test
        // are three questions about ONE cell, and upstream asks all three off
        // the single `cell` its chain walk bound
        // (`WarmEnterState.maybe_compile_and_run` — "these few lines inline
        // some logic that is also on the JitCell class, to avoid computing the
        // hash several times"). The `compiled_loops` conjunct is still taken
        // below, where the metadata's value is wanted anyway.
        //
        // Which shape the gate below takes, recorded before the token is moved
        // into it, so the amplified gate can take the same one.
        #[cfg(feature = "__back-edge-stage-probe")]
        let carried_token = carried_procedure_token.is_some();
        let (green_key, entry_procedure_token) = match carried_procedure_token {
            Some(token) => (green_key_hash, Some(token)),
            None => self.resolved_entry_procedure_token(green_key_hash, structured_green_key),
        };
        // Single-pass label handoff must stay ahead of the counter so a
        // CloseLoop arm's pending LABEL entry is consumed on the next back
        // edge rather than ticking past it. The table is one `Option`; check
        // emptiness first so a never-armed loop does not enter the take.
        // Assembler entry is the procedure token (`execute_assembler`),
        // not a one-shot body-LABEL dispatch key.
        let single_pass_dispatch_key: Option<u32> = None;
        if !state.can_trace() {
            return None;
        }

        if spdiag_enabled() {
            eprintln!(
                "@@@SPDIAG back_edge target_pc={target_pc} green_key={green_key} has_compiled={}",
                self.meta.has_compiled_loop(green_key)
            );
        }

        // Same decline as the `None if is_cross_loop_cut_key(target_key)` arm
        // of the CloseLoop bridge path above, on the sibling route: that arm
        // refuses an ENTRY BRIDGE into a cross-loop cut's key because the cut
        // artifact's entry invariants hold only after the guards preceding the
        // cut point have run, and a `ResumeFromInterpDescr` carries no runtime
        // values to prove them; on cel's `batch_chain_eager_fold_traps` the
        // unproven index bound became an out-of-bounds store. A direct
        // interpreter back edge proves no more than that entry bridge does and
        // can therefore execute the same invalid store. This route was hidden
        // while the key the interpreter entered by and the
        // key `compile_loop` filed the cut under could not be equal while
        // `can_enter_jit!` read its position green at the back edge instead of
        // at the target; closing that made this route reachable.
        //
        // What the entry would execute, from the cut trace's own head on that
        // fixture: `IntAdd(v5, 1)` / `IntMul` / `RawLoadI` and a later
        // `RawStore`, with no bound guard on either — the guard was in the ops
        // `cut_trace_from` discarded, and `PreambleCompileData` re-optimizes
        // from scratch, so nothing re-derives it. Scoped to the generic case
        // for that reason: the cutting trace's own closing JUMP and the
        // single-pass label handoff pass an explicit dispatch key and arrive
        // with the discarded prefix already run. Refusing before
        // `has_compiled_loop` leaves the merge point to arm tracing normally
        // rather than re-entering an artifact whose entry contract is unmet.
        let dispatch_key = dispatch_key.or(single_pass_dispatch_key);
        // `cut_compiled_keys` is empty until a cross-loop cut installs a
        // loop; check that first so a never-compiled back edge does not
        // hash-probe the set. The decline itself stays ahead of the
        // counter: a cut key must not tick into `bound_reached` either.
        if dispatch_key.is_none()
            && !self.meta.cut_compiled_keys.is_empty()
            && self
                .meta
                .is_cross_loop_cut_key_on_driver(self.index().unwrap_or(0), green_key)
        {
            return None;
        }
        // `warmstate.py maybe_compile_and_run`: a cell whose token was
        // `attached by compile_tmp_callback()` is NOT entered — upstream falls
        // straight to `jitcounter.tick(hash, increment_threshold)` and, on
        // overflow, `bound_reached`. The resolution above reads the
        // cell-owned `JC_TEMPORARY` flag explicitly: a callback token has a
        // body, and pyre may retain frontend metadata for the loop it
        // displaced, so neither token nor metadata presence identifies it.
        // Binding the token rather than testing a predicate and
        // unwrapping afterwards keeps the decision and execution in step — the
        // fall-through below IS the counter processing upstream continues
        // with. Token presence stays one conjunct: it is the
        // `cell.get_procedure_token() is not None` half (code present and not
        // invalidated), which the meta lookup alone does not imply, since
        // `invalidate_loop` deliberately leaves the meta in place.
        // Both halves read through the resolved cell key, so the artifact the
        // entry DECIDES on and the artifact it EXECUTES are the same object.
        // They were not: the decision walked the bucket by `comparekey` while
        // the runner read `compiled_loops[hash]`, so on a chained bucket the
        // entry could decide about one cell's token and run another's.
        //
        // The token itself is bound here rather than only tested, and travels
        // to the run below as an argument. `warmstate.py` reads
        // `procedure_token = cell.get_procedure_token()` once per
        // `maybe_compile_and_run` and `:509-511` carries it out through
        // `raise EnterJitAssembler(procedure_token, *execute_args)`; the
        // runner receives it and never asks the cell again. This is the same
        // resolve-once discipline the cell key above already follows, one
        // level in: the token the gate said yes about IS the token executed,
        // and a warm entry pays for one cell lookup, not one per question it
        // asks about that cell.
        //
        // A caller that already took its own decision on this cell's token
        // hands that same object in, and it is entered unread — the carry
        // reaches one level further out, to that caller, exactly as
        // `EnterJitAssembler` carries `:483`'s read past `maybe_compile_and_run`
        // and into the executor. The meta below is still fetched here because
        // its VALUE is needed, not just its presence.
        // Read once per entry, ahead of every stage, so no stage's difference
        // carries it. See [`BackEdgeStageRepeats`]; a default build has neither
        // this load nor the loops it feeds.
        #[cfg(feature = "__back-edge-stage-probe")]
        let stage_repeats = back_edge_stage_repeats();
        if let Some(procedure_token) = entry_procedure_token
            && let Some(compiled_meta) = self
                .meta
                .get_compiled_meta_on_driver(self.index().unwrap_or(0), green_key)
                .cloned()
        {
            // warmstate.py `WarmEnterState.make_entry_point` /
            // `maybe_compile_and_run`: the confirmation veto sits between
            // finding the procedure token and `EnterJitAssembler`.
            if !self
                .meta
                .warm_state_ref_for_driver(self.index().unwrap_or(0))
                .is_some_and(|warm| warm.confirm_compiled_entry_for_cell_key(green_key))
            {
                return None;
            }
            // The cached descriptor's `Arc` stays on the driver. Cloning it
            // on every entry bumped the count; `warmstate.py execute_assembler`
            // closes over the driver data instead of re-resolving it.
            let descriptor_ptr = self.driver_descriptor_ptr(state, &compiled_meta);
            let descriptor = unsafe { descriptor_ptr.map(|ptr| &*ptr) };
            let vable = descriptor.and_then(JitDriverStaticData::virtualizable);
            if !state.is_compatible(&compiled_meta) {
                self.invalidate_loop(green_key);
                return None;
            }
            // Stage E1, and the barrier the other stages are differenced
            // against. The gate above has just run, so every consultation it
            // makes is repeated here in the same order and under the same
            // condition rather than being reordered into a shape the entry
            // never takes.
            #[cfg(feature = "__back-edge-stage-probe")]
            {
                count_back_edge_stage_passes(BackEdgeStage::Gate, stage_repeats.gate);
                for _ in 0..stage_repeats.gate {
                    let token = if carried_token {
                        None
                    } else {
                        self.meta.entry_procedure_token(green_key)
                    };
                    let repeat_meta = self
                        .meta
                        .get_compiled_meta_on_driver(self.index().unwrap_or(0), green_key)
                        .cloned();
                    let repeat_descriptor = self.driver_descriptor_for(state, &compiled_meta);
                    let compatible = state.is_compatible(&compiled_meta);
                    // Without this the four answers are dead and the loop
                    // prices nothing.
                    std::hint::black_box((token, repeat_meta, repeat_descriptor, compatible));
                    std::hint::black_box(&mut *state);
                }
                // The same loop and the same barrier with no stage in them, so
                // what the amplification itself costs is subtracted rather than
                // reported as a stage.
                count_back_edge_stage_passes(BackEdgeStage::Barrier, stage_repeats.barrier);
                for _ in 0..stage_repeats.barrier {
                    std::hint::black_box(&mut *state);
                }
            }
            // No recording context to move or put back: this function
            // refuses to run at all while `is_tracing`, so `sync_before`
            // writes no trace pointer and the exits below have none to
            // restore. The two `run_compiled_detailed_*` runners carry the
            // save/restore, because a walk does reach those.
            if !self.sync_before(state, &compiled_meta, vable) {
                return None;
            }
            // The entry arguments are assembled in buffers the driver keeps
            // across calls, so a warm entry reuses the capacity the artifact's
            // inputarg shape fixed instead of minting one `Vec` per call —
            // `warmstate.py maybe_compile_and_run` reaches
            // `execute_assembler` with the reds already in hand and allocates
            // nothing to describe them. Taken out of `self` for the duration
            // because assembling them calls back into `self`; every exit below
            // hands it back, and `entry_scratch_out` names that.
            let mut scratch = self.take_entry_scratch();
            // `direct_live_values` arrive already in the target LABEL's argument
            // order; state-extracted ones are in state-field order and have to be
            // mapped before a dispatch-key entry can use them.
            let values_are_label_ordered = direct_live_values.is_some();
            // Stage E2, ahead of the shipping refill rather than after it: each
            // pass ENDS with the clears `take_entry_scratch` performs, so the
            // buffers the real extraction below fills are in exactly the state
            // it would have found them in, whatever any pass answered.
            #[cfg(feature = "__back-edge-stage-probe")]
            if !values_are_label_ordered {
                count_back_edge_stage_passes(BackEdgeStage::MarshalIn, stage_repeats.marshal_in);
                for _ in 0..stage_repeats.marshal_in {
                    let synced = self.sync_before(state, &compiled_meta, vable);
                    state.extract_live_values_into(
                        &compiled_meta,
                        &mut scratch.live_values,
                        &mut scratch.raw,
                        &mut scratch.types,
                    );
                    let matched = Self::live_values_match_descriptor(
                        descriptor.as_deref(),
                        &scratch.live_values,
                        state.state_field_layout().total_live_values(),
                    );
                    let extended = self.extend_compiled_live_values_into(
                        &procedure_token,
                        state,
                        &compiled_meta,
                        vable,
                        &mut scratch.live_values,
                        &mut scratch.vable_boxes,
                        &mut scratch.vable_lengths,
                    );
                    // Answered and not acted on: the shipping pass below takes
                    // the decision, and a probe pass that declined would only
                    // mean this stage cannot be amplified on this shape.
                    std::hint::black_box((synced, matched, extended));
                    std::hint::black_box(&mut *state);
                    scratch.live_values.clear();
                    scratch.raw.clear();
                    scratch.types.clear();
                    scratch.vable_boxes.clear();
                    scratch.vable_lengths.clear();
                }
            }
            if let Some(values) = direct_live_values {
                scratch.live_values.extend(values);
            } else {
                state.extract_live_values_into(
                    &compiled_meta,
                    &mut scratch.live_values,
                    &mut scratch.raw,
                    &mut scratch.types,
                );
                if !Self::live_values_match_descriptor(
                    descriptor.as_deref(),
                    &scratch.live_values,
                    state.state_field_layout().total_live_values(),
                ) {
                    self.entry_scratch_out(scratch);
                    return None;
                }

                if !self.extend_compiled_live_values_into(
                    &procedure_token,
                    state,
                    &compiled_meta,
                    vable,
                    &mut scratch.live_values,
                    &mut scratch.vable_boxes,
                    &mut scratch.vable_lengths,
                ) {
                    self.entry_scratch_out(scratch);
                    return None;
                }
            }
            if dispatch_key.is_some() && !values_are_label_ordered {
                let full_len = scratch.live_values.len();
                let Some(packed) = self.meta.pack_front_target_live_values_on_driver(
                    self.index().unwrap_or(0),
                    green_key,
                    &scratch.live_values,
                ) else {
                    if crate::callee_rca_enabled() {
                        eprintln!(
                            "[callee-rca][direct-entry-pack] key={green_key} \
                             dispatch_key={:?} full_len={full_len} \
                             source=front-target-source-positions -> decline",
                            dispatch_key,
                        );
                    }
                    self.entry_scratch_out(scratch);
                    return None;
                };
                if crate::callee_rca_enabled() {
                    eprintln!(
                        "[callee-rca][direct-entry-pack] key={green_key} dispatch_key={:?} \
                         source=front-target-source-positions full_len={full_len} \
                         compact_len={}",
                        dispatch_key,
                        packed.len()
                    );
                }
                scratch.live_values.clear();
                scratch.live_values.extend(packed);
            }
            if dispatch_key.is_some() && values_are_label_ordered {
                // The compact values are LABEL-ordered by construction, but the
                // same unchecked Ref dereference is downstream of them, so the
                // types are confirmed here as well.
                let Some(types) = self
                    .meta
                    .front_target_inputarg_types_on_driver(self.index().unwrap_or(0), green_key)
                else {
                    self.entry_scratch_out(scratch);
                    return None;
                };
                if types.len() != scratch.live_values.len()
                    || types
                        .iter()
                        .zip(scratch.live_values.iter())
                        .any(|(want, value)| value.get_type() != *want)
                {
                    if crate::callee_rca_enabled() {
                        eprintln!(
                            "[callee-rca][direct-entry-pack] key={green_key} \
                             dispatch_key={:?} source=compact-label label_types={types:?} \
                             value_types={:?} -> decline",
                            dispatch_key,
                            scratch
                                .live_values
                                .iter()
                                .map(|v| v.get_type())
                                .collect::<Vec<_>>(),
                        );
                    }
                    self.entry_scratch_out(scratch);
                    return None;
                }
            }
            let live_values = &scratch.live_values;

            if crate::callee_rca_enabled() {
                let layout = state.state_field_layout();
                let labels = state.debug_state_live_labels(&compiled_meta);
                let compiled_inputs = self
                    .meta
                    .warm_state_ref_for_driver(self.index().unwrap_or(0))
                    .and_then(|ws| ws.get_compiled(green_key))
                    .map(|compiled| compiled.inputarg_types().len())
                    .unwrap_or(0);
                eprintln!(
                    "[callee-rca][entry] green_key={green_key} target_pc={target_pc} \
                     compiled_inputs={} live_values={} layout={:?}",
                    compiled_inputs,
                    live_values.len(),
                    layout,
                );
                if let Some(dump) = state.debug_state_fields(&compiled_meta) {
                    eprintln!("[callee-rca][entry-state]\n{dump}");
                }
                eprintln!(
                    "[callee-rca][entry-live]\n{}",
                    format_rca_live_values(labels.as_deref(), &live_values)
                );
            }

            pre_run();

            let selected_dispatch_key = dispatch_key.unwrap_or(0);
            // One read for the pair below. The flag is fixed at process start,
            // but the compiled run sits between the two sites and is opaque to
            // the optimizer, so asking twice is two loads and two calls.
            let portal_rca = portal_rca_enabled();
            if portal_rca {
                let labels = state.debug_state_live_labels(&compiled_meta);
                eprintln!(
                    "[portal-rca][compiled-entry] green_key={green_key} target_pc={target_pc} \
                     dispatch_key={selected_dispatch_key} live_values={}",
                    live_values.len()
                );
                eprintln!(
                    "[portal-rca][compiled-entry-live]\n{}",
                    format_rca_live_values(labels.as_deref(), &live_values)
                );
            }
            // Same point the RCA line above reports, but unconditional and
            // counted: every decline has already returned, and the compiled
            // body runs on the next statement, so one call here is one entry.
            // Scoped so the borrow ends before `self.meta` is taken mutably.
            if let Some(ref hook) = self.meta.hooks.on_compiled_entry {
                hook(green_key, target_pc);
            }

            // `warmstate.py execute_assembler` reads `jitdriver_sd.result_type`
            // and returns `DoneWithThisFrameDescrInt` / `DoneWithThisFrameDescrRef`
            // `get_result`. Dispatch key 0 is the ordinary token entry. A
            // direct LABEL entry keeps the general result. Int (and every
            // non-ref kind) stays on `poll_raw_int_finish`.
            let polled_raw_int = selected_dispatch_key == 0;
            if polled_raw_int {
                let finished = if self.meta.result_type == Type::Ref {
                    self.meta
                        .poll_raw_ref_finish(&procedure_token, green_key, live_values)
                        .map(|addr| addr as i64)
                } else {
                    self.meta
                        .poll_raw_int_finish(&procedure_token, green_key, live_values)
                };
                if let Some(value) = finished {
                    self.entry_scratch_out(scratch);
                    self.meta.back_edge_finish = None;
                    self.meta.back_edge_finish_word = Some(value);
                    if vable.is_some() {
                        self.sync_after(state, &compiled_meta, vable, None);
                    }
                    self.meta.single_pass_finish = true;
                    // `warmstate.py execute_assembler` `DoneWithThisFrameDescr*`
                    // returns the word. `handle_jitexception` does not read
                    // green banks on that path.
                    return Some(PortalResume::DoneWithThisFrame(None));
                }
            }
            let result = if polled_raw_int {
                self.meta
                    .raw_int_fallback
                    .take()
                    .expect("poll_raw_int_finish stored the general-case result")
            } else {
                self.meta.execute_assembler_at_dispatch_key(
                    &procedure_token,
                    green_key,
                    // The run is handed the meta the gate above already read out of
                    // `compiled_loops`, rather than probing that map a third time
                    // for the same slot. Nothing between the two can replace the
                    // entry: `pre_run` cannot capture the driver and the
                    // `on_compiled_entry` hook is called through `&self.meta`.
                    // Not carried: this entry reads its snapshot from
                    // `compiled_meta` on every arm below, so a copy on the result
                    // would ride out and drop unread.
                    None,
                    live_values,
                    selected_dispatch_key,
                )
            };
            // The compiled body no longer reads its entry arguments. Return
            // these buffers before any guard-resume path can re-enter the
            // driver to trace a bridge.
            self.entry_scratch_out(scratch);
            return self.consume_compiled_entry_result(
                green_key,
                target_pc,
                state,
                env,
                result,
                &compiled_meta,
                vable,
                selected_dispatch_key,
                portal_rca,
            );
        }

        // Re-enter the dispatch loop at `target_pc` so the trace HEADS where its
        // live-in snapshot was TAKEN.
        //
        // `maybe_start_tracing` reads `extract_live_values` off native `state`
        // at this call, but nothing is recorded until the next
        // `jit_merge_point!`. When the caller reaches that merge point without
        // executing anything — a back edge, whose `pc = tgt; continue;` follows
        // this macro immediately — the two coincide and the snapshot is exact.
        //
        // An ENTRY DOOR arms at the pc it is about to execute, so falling
        // through here runs one instruction natively, outside the trace and
        // after the snapshot. The trace then heads one instruction late while
        // its live-ins hold pre-instruction values, and that instruction's
        // write is silently dropped: the walk binds the register to its value
        // at the arming pc, so the write never reaches the compiled body OR
        // native `state`. Returning the target makes the caller `continue` to
        // it, and the merge point there records with tracing already live.
        //
        // For a back edge this is behaviourally identical to the `pc = tgt;
        // continue;` it pre-empts (`target_pc == tgt`); 21 of the corpus's 23
        // `can_enter_jit!` sites are that shape. Re-entry cannot loop: the next
        // pass returns at the `is_tracing()` guard opening this function.
        // Occupied bucket, no enterable token: the typed comparekey walk
        // (`on_back_edge_typed_decision`) ticks the cell the greens own.
        // Frontend snapshot work stays behind `StartTracing`.
        if self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env) {
            return Some(continue_at_pc(target_pc));
        }
        None
    }

    /// Finish, loop-back, or guard failure after the assembler returned.
    ///
    /// `warmstate.py execute_assembler` reads `get_latest_descr` and either
    /// returns `DoneWithThisFrameDescr*.get_result` or calls `handle_fail`.
    /// The int finish that already returned a word never reaches this.
    fn consume_compiled_entry_result(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        result: crate::pyjitpl::CompileResult<S::Meta>,
        compiled_meta: &std::sync::Arc<S::Meta>,
        vable: Option<&JitDriverVar>,
        selected_dispatch_key: u32,
        portal_rca: bool,
    ) -> Option<PortalResume> {
        let mut result = result;
        if portal_rca {
            eprintln!(
                "[portal-rca][compiled-exit] green_key={green_key} \
                 dispatch_key={selected_dispatch_key} is_finish={} fail_index={} \
                 typed_values={:?}",
                result.is_finish, result.fail_index, result.typed_values
            );
        }

        // Stage E4, before the arm split, so it prices one pair of calls
        // rather than a different pair per outcome — and before the FINISH
        // arm takes the exit values out from under it.
        // The exit is consumed here, outside the function that read the
        // repeats for the earlier stages, so it reads them again.
        #[cfg(feature = "__back-edge-stage-probe")]
        if result.is_finish || result.fail_index == u32::MAX {
            let stage_repeats = back_edge_stage_repeats();
            count_back_edge_stage_passes(BackEdgeStage::MarshalOut, stage_repeats.marshal_out);
            for _ in 0..stage_repeats.marshal_out {
                if !result.is_finish && !result.typed_values.is_empty() {
                    state.restore_values(&compiled_meta, &result.typed_values);
                }
                self.sync_after(state, &compiled_meta, vable, None);
                std::hint::black_box(&mut *state);
            }
        }

        if result.is_finish && result.is_exit_frame_with_exception {
            // compile.py `ExitFrameWithExceptionDescrRef.handle_fail`:
            // `value = cpu.get_ref_value(deadframe, 0)` and raise
            // `jitexc.ExitFrameWithExceptionRef(value)`, which the portal
            // runner re-raises as the portal's own exception. It is not a
            // return value, so nothing is published on the finish latch;
            // the exception goes to the same interpreter hook the blackhole
            // `_exit_frame_with_exception` arms use.
            let exc = match result.typed_values.first() {
                Some(Value::Ref(exc)) => *exc,
                other => panic!("exit-with-exception FINISH carried {other:?}, not a Ref"),
            };
            self.sync_after(state, compiled_meta, vable, None);
            let Some(resume_pc) = state.deliver_blackhole_exception(exc) else {
                eprintln!(
                    "[jit] exception escaped the compiled portal and this interpreter has \
                     no `deliver_blackhole_exception` — ending the dispatch loop"
                );
                self.meta.single_pass_finish = true;
                return Some(PortalResume::ExitFrameWithException {
                    pc: usize::MAX,
                    args: None,
                });
            };
            return Some(PortalResume::ExitFrameWithException {
                pc: resume_pc,
                args: None,
            });
        }

        if result.is_finish {
            // compile.py `_DoneWithThisFrameDescr.final_descr = True`:
            // the compiled run ended in FINISH, so the traced function has
            // RETURNED. Upstream `handle_fail` raises `jitexc.DoneWithThisFrame*`
            // and unwinds the portal; there is no resume point past a final
            // descr. `target_pc` is the back edge, so resuming there restarts
            // the loop the run just finished — with only the FINISH's own
            // argument restored, not the loop-carried state — and the caller
            // re-enters compiled code at the next back edge, making one call
            // cost one full compiled run per remaining iteration.
            //
            // Publish the FINISH arguments out of band. Front end A reads the
            // same outcome as `DetailedDriverRunOutcome::Finished` and returns
            // from the portal; front end B's `Option<resume_pc>` signature has
            // no variant for "the function returned", so the `#[jit_interp]`
            // expansion drains this latch right after the call and returns it
            // as the portal's own return value.
            // Taken, not copied: this arm returns below, so the exit values
            // in `result` have no reader past this point.
            self.meta.back_edge_finish_word = None;
            self.meta.back_edge_finish = Some(std::mem::take(&mut result.typed_values));
            // The meta the run was handed, not a second handle to it:
            // cloning `result.meta` here bought a refcount pair for a value
            // already in scope.
            let run_meta = &compiled_meta;
            // The FINISH arguments are NOT the loop-carried state, so they
            // are not written back into it. `warmstate.py:405-419
            // execute_assembler` takes the `DoneWithThisFrameDescr*` fast
            // path straight to `fail_descr.get_result(cpu, deadframe)` and
            // returns; the only thing it reads off the deadframe is the
            // portal's own result, and it touches no interpreter state on
            // the way out. A `restore_values` here decodes the FINISH
            // descr's `fail_arg_types` — one word for an int-returning
            // portal — into slots indexed by the state's live-value
            // layout, so a state with more than one live field indexes
            // past the end of the list it was handed.
            // The descriptor the entry decided on is the one the exit
            // syncs through — resolved once above and carried here rather
            // than asked for again. `warmstate.py/509-511` reads the
            // cell once per `maybe_compile_and_run` and carries the read
            // out to the executor for the same reason; the driver's static
            // data is a configuration-time constant (see
            // `descriptor_cache`), so a second consultation could only
            // return the same object at the cost of one more refcount pair
            // per compiled entry.
            self.sync_after(state, run_meta, vable, None);
            // `compile.py _DoneWithThisFrameDescr.final_descr`: the portal
            // returned. `warmspot.py handle_jitexception` answers
            // `DoneWithThisFrame*` by returning from `ll_portal_runner`
            // and reads no green banks. `ResumeAt` is the JUMP / tick
            // path, whose header `guard_value` already passed.
            self.meta.single_pass_finish = true;
            return Some(PortalResume::DoneWithThisFrame(None));
        }

        // Normal loop back-edge JUMP, not a guard failure.
        if result.fail_index == u32::MAX {
            // Same handle as the FINISH arm above, for the same reason.
            let run_meta = &compiled_meta;
            if !result.typed_values.is_empty() {
                state.restore_values(run_meta, &result.typed_values);
            }
            // Carried from the entry decision above, not re-resolved; see
            // the FINISH arm for why one resolution serves both ends of a
            // compiled entry.
            self.sync_after(state, run_meta, vable, None);
            return Some(continue_at_pc(target_pc));
        }

        // compile.py handle_fail
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        // `compile.py ResumeGuardDescr.handle_fail` / `resume.py
        // blackhole_from_resumedata` read fail values from the deadframe
        // in place (`resume.py ResumeDataDirectReader.decode_int` via
        // `self.cpu.get_int_value(self.deadframe, num)`). No dense list
        // is built unless `must_compile` starts a bridge.
        let descr_arc = result
            .descr_arc
            .take()
            .expect("a guard exit carries its descr");
        // `compile.py handle_fail(self, deadframe, ...)`: the failing
        // guard's own `ResumeGuardDescr` is what every read below is off
        // — its fail-arg types, its `rd_*` resume payload — not a layout
        // assembled per failure from the frontend's records.
        let fd: &dyn majit_ir::FailDescr = descr_arc
            .as_fail_descr()
            .expect("a guard exit carries a FailDescr");
        let guard_value_operand = result.guard_value_operand;
        // `compile.py handle_fail` `resumedescr.rd_loop_token`, resolved
        // once where the run handed the descr back.
        let descr_owning_key = result.rd_loop_token;
        // blackhole.py `_prepare_resume_from_failure(deadframe)`:
        // the pending exception grabbed at guard failure
        // (cpu.grab_exc_value) must seed the blackhole resume so an
        // exception guard unwinds into its handler instead of resuming
        // the no-exception continuation.
        let guard_exc = result.exception.exc_value;
        let savedata = result.savedata;
        // Keep `result` (and its deadframe) until this arm returns so
        // `jf_savedata` stays rooted through `AllVirtuals.show`, matching
        // `compile.py handle_fail(self, deadframe, ...)`.
        // Also park the copied GCREF: `AllVirtuals.show` reads this
        // word after the reconstruction has already allocated.
        let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
        let _savedata_root = unsafe {
            crate::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| savedata.is_some())
        };
        let savedata = savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize));
        // The reconstruction below allocates through the blackhole allocator,
        // so hold the exception where the frontend's root walker can reach
        // it until `prepare_resume_from_failure` hands it to the blackhole.
        let _guard_exc_root = crate::blackhole::GuardExcRoot::park(guard_exc);

        // must_compile tick for bridge threshold counting.
        if crate::majit_log_enabled() {
            let descr_addr = std::sync::Arc::as_ptr(&descr_arc) as *const () as usize;
            eprintln!(
                "[jit] handle_fail: fail_index={} trace_id={} descr_addr={:#x} values={}",
                fail_index,
                trace_id,
                descr_addr,
                fd.fail_arg_types().len()
            );
        }
        if failvals_enabled() {
            let raw_values = result.deadframe.as_ref().map_or_else(Vec::new, |frame| {
                self.meta.raw_exit_slots_from_deadframe(frame, fd)
            });
            eprintln!(
                "@@@FAILVALS fail_index={} resume_pc={} raw_values={:?}",
                fail_index,
                crate::compile::guard_resume_pc(fd)
                    .map(|p| p as i64)
                    .unwrap_or(-1),
                raw_values
            );
        }
        if crate::callee_rca_enabled() {
            let raw_values = result.deadframe.as_ref().map_or_else(Vec::new, |frame| {
                self.meta.raw_exit_slots_from_deadframe(frame, fd)
            });
            eprintln!(
                "[callee-rca][guard-fail] fail_index={} trace_id={} raw_values={:?} exit_types={:?}",
                fail_index,
                trace_id,
                raw_values,
                fd.fail_arg_types(),
            );
            if let Some(dump) = state.debug_state_fields(&compiled_meta) {
                eprintln!("[callee-rca][post-execute-token-state]\n{dump}");
            }
        }
        // Resolved once for the whole event, where the run handed the descr
        // back, and carried here: `compile.py handle_fail` reads
        // `resumedescr.rd_loop_token` once and hands the loop it names to
        // the `must_compile` call. A descr with no owner stamped is the
        // loop that was entered.
        let owning_key = descr_owning_key.unwrap_or(green_key);
        // `guard_value_operand` was read off the deadframe while it was
        // live (`compile.py must_compile` `cpu.get_value_direct`). The
        // fail-values slice is only the fallback for backends that
        // resolve the GUARD_VALUE index from the dense vector.
        let (must_compile, owning_key) = self.meta.must_compile_with_owning_key(
            &descr_arc,
            &[],
            guard_value_operand,
            owning_key,
        );
        // compile.py: must_compile() and not stack_almost_full().
        // MAJIT_NO_BRIDGE (diagnostic): suppress bridge recording so every
        // guard failure resumes via blackhole — isolates bridge-record
        // resume defects from the blackhole path.
        //
        // Pending-field guards bridge through setup_bridge_sym's
        // pending-field prologue (resume.py _prepare_pendingfields).
        let should_bridge = must_compile
            && !majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full()
            && !no_bridge_enabled()
            && bridge_fuel_take();

        // compile.py handle_fail. `must_compile() and not
        // stack_almost_full()` → `_trace_and_compile_from_bridge`, else
        // `resume_in_blackhole`; the two are mutually exclusive and
        // neither returns. majit adapts by returning the interpreter
        // resume pc.
        //
        // The bridging arm enters the walk at the guard's own jitcode
        // position, so the rest of the opcode the guard sits inside is
        // recorded rather than run where the bridge cannot see it. It
        // declines — and only before running anything — for a state with
        // no dispatch jitcode, a resume frame it cannot seed a register
        // file from, or a bridge setup that gave up; the blackhole arm
        // below is then the answer, exactly as when the guard does not
        // `must_compile` at all.
        // Decode a dense list only when this failure will start a
        // bridge (`compile.py ResumeGuardDescr.handle_fail` when
        // `must_compile` fires — about once per 200 failures).
        let mut raw_values_for_bridge: Option<Vec<i64>> = None;
        if should_bridge {
            let mut raw_values = self.take_exit_raw_scratch();
            if let Some(frame) = result.deadframe.as_ref() {
                raw_values.extend(self.meta.raw_exit_slots_from_deadframe(frame, fd));
            }
            // `resume.py` `ResumeDataBoxReader.load_box_from_cpu` reads the
            // deadframe at decode time. This scratch is already a copy of
            // those slots; root its Ref words for the bridge setup below,
            // which allocates (`allocate_with_vtable`) and then reads them.
            // The guard ends before `raw_values` is moved.
            let ref_slot: Vec<bool> = fd
                .fail_arg_types()
                .iter()
                .map(|tp| *tp == majit_ir::Type::Ref)
                .collect();
            let bridge_pc = {
                let _fail_arg_roots = unsafe {
                    crate::resume::DeadFrameRefRoots::enter(&mut raw_values, |index| {
                        ref_slot.get(index).copied().unwrap_or(false)
                    })
                };
                self.bridge_from_guard_resume_position(
                    &descr_arc,
                    state,
                    env,
                    &raw_values,
                    target_pc,
                )
            };
            if let Some(resume) = bridge_pc {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bridge] guard-resume bridge key={} trace={} fail={} resume_pc={}",
                        green_key,
                        trace_id,
                        fail_index,
                        resume.resume_pc().unwrap_or(usize::MAX),
                    );
                }
                self.exit_raw_scratch_out(raw_values);
                return Some(resume);
            }
            raw_values_for_bridge = Some(raw_values);
        }

        // compile.py:711 resume_in_blackhole
        // compile.py `ResumeGuardDescr` storage — `rd_numb` /
        // `rd_consts` / `rd_virtuals` / `rd_pendingfields` borrowed off
        // the failing descr itself (`get_resumestorage(): return self`),
        // so blackhole resume observes the guard-owned pool the GC
        // walker updates, and nothing is built per failure to hold them.
        if let Some(rd_numb) = fd.rd_numb() {
            let rd_consts_slice: &[Const] = fd.rd_consts().unwrap_or(&[]);

            // `resume.py ResumeDataVirtualAdder._number_virtuals` stores
            // `storage.rd_virtuals` once. Resume assigns that list
            // (`AbstractResumeDataReader._prepare_virtuals`). An empty
            // list skips the two `VirtualCache` vecs.
            let guard_virtuals = crate::resume::guard_virtual_infos(fd);
            let rd_virtuals_slice = guard_virtuals.as_ref().map(|infos| infos.as_slice());

            // resume.py:1338-1340: `jitcode = jitcodes[jitcode_pos];
            // curbh.setposition(jitcode, pc)`.  Per-driver
            // pyre layout: root frame resolves to the dispatch
            // JitCode singleton (RPython's `metainterp_sd.jitcodes`
            // for the portal jitdriver slot); sub-frames index into
            // the parent's `descrs` array per `BC_INLINE_CALL`'s
            // `j` argcode (`blackhole.py`).
            //
            // The snapshot's root `jitcode_index` is unused once
            // the dispatch singleton is registered — the closure
            // ignores it and clones `self.dispatch_jitcode`.
            let _ = env;
            // resume.py `jitcode = jitcodes[jitcode_pos]` —
            // resolve every frame statelessly from the flat global
            // registry by its self-describing absolute index (no root/sub
            // branch, no parent-relative descrs walk, no last-frame state).
            // `resume.py`'s `jitcode = jitcodes[jitcode_pos]` indexes the
            // prebuilt list in place.  Borrowing it costs nothing; the
            // `to_vec` that stood here cloned every `Arc` in the registry
            // — two atomic RMWs per jitcode — on a path this fixture takes
            // once per input character.
            let jitcode_registry: &[std::sync::Arc<crate::jitcode::JitCode>] = self.meta.jitcodes();
            let resolve_jitcode =
                |jitcode_index: i32, pc: i32| -> Option<crate::resume::ResolvedJitCode> {
                    // `jitcodes[jitcode_pos]` and `setposition(jitcode, pc)`.
                    // `as usize` wraps a negative index or pc into a huge
                    // slot and resumes mid-instruction in unrelated code.
                    let index = usize::try_from(jitcode_index).ok()?;
                    let resolved_pc = usize::try_from(pc).ok()?;
                    let resolved_jitcode = jitcode_registry.get(index)?.clone();
                    Some(crate::resume::ResolvedJitCode::new(
                        resolved_jitcode,
                        resolved_pc,
                    ))
                };

            let fallback_alloc = crate::resume::NullAllocator;
            let allocator: &dyn crate::resume::BlackholeAllocator = self
                .blackhole_allocator
                .as_deref()
                .unwrap_or(&fallback_alloc);

            // Inline-call-only builder so byte 17 (BC_INLINE_CALL) is
            // wired to handler_inline_call_nested_ext when the
            // multi-frame state-field-JIT chain reconstructs sub-frames
            // via parent.descrs[idx].as_jitcode() above.  3
            // retired the dispatch_one BC_INLINE_CALL legacy arm so an
            // empty builder would now panic on byte 17.  Re-apply
            // staticdata control opcodes after the constructor's
            // BC_LIVE/BC_CATCH_EXCEPTION/BC_RVMPROF_CODE defaults so
            // the values match the metainterp's actual byte
            // assignments.
            let mut bh_builder = BackEdgeBhBuilder::lease();
            bh_builder.set_cpu(self.meta_interp().blackhole_cpu());
            bh_builder.setup_cached_control_opcodes(
                self.meta_interp().staticdata.op_live,
                self.meta_interp().staticdata.op_catch_exception,
                self.meta_interp().staticdata.op_rvmprof_code,
            );
            // `blackhole_from_resumedata` below acquires one interpreter
            // per resumed frame, and every frame that still has a caller
            // takes `bhimpl_jit_merge_point`'s recursive-portal branch
            // (blackhole.py), which indexes `jitdrivers_sd`
            // directly.  The lease is thread-local and carries whatever
            // the previous resume left on it, so seed the table here
            // instead of depending on that.
            let jitdrivers_sd = bh_jitdrivers_sd(&self.meta_interp().staticdata);
            if !std::sync::Arc::ptr_eq(&bh_builder.jitdrivers_sd, jitdrivers_sd) {
                bh_builder.setup_jitdrivers_sd(std::sync::Arc::clone(jitdrivers_sd));
            }
            let all_liveness = self.meta_interp().staticdata.liveness_info.snapshot_arc();
            // `resume.py ResumeDataDirectReader.decode_int` —
            // `self.cpu.get_int_value(self.deadframe, num)`. A GC
            // jitframe is re-read through `OwnerRootGuard`; an off-heap
            // frame is read in place (`llmodel.py get_int_value`).
            // Keep `result` alive so `jf_savedata` stays rooted.
            let n_fail_args = fd.fail_arg_types().len();
            let mut fallback_raw = Vec::new();
            let fail_args =
                match result.deadframe.as_ref().and_then(|frame| {
                    majit_backend::fail_arg_source_from_frame(frame, fd, n_fail_args)
                }) {
                    Some(src) => src,
                    None => {
                        if let Some(frame) = result.deadframe.as_ref() {
                            fallback_raw = self.meta.raw_exit_slots_from_deadframe(frame, fd);
                        }
                        majit_backend::FailArgSource::Slice(&fallback_raw)
                    }
                };
            let bh = crate::resume::blackhole_from_resumedata(
                &mut bh_builder,
                &resolve_jitcode,
                rd_numb,
                rd_consts_slice,
                &all_liveness,
                fail_args,
                Some(fd.fail_arg_types()),
                rd_virtuals_slice,
                Some(fd.rd_pendingfields().unwrap_or(&[])), // rd_guard_pendingfields
                Some(
                    &self.meta_interp().staticdata.virtualref_info as &dyn crate::resume::VRefInfo,
                ),
                // Pass the registered vinfo so the vable section of the
                // resume stream is consumed (avoids a stream desync when
                // the trace carries virtualizable array boxes).
                self.meta
                    .virtualizable_info()
                    .map(|a| a.as_ref() as &dyn crate::resume::VirtualizableInfo),
                None, // ginfo
                savedata.and_then(crate::compile::AllVirtuals::show),
                allocator,
            );
            let (mut bh, vable_ptr) = bh;
            {
                // Thread the state-field register layout onto every frame
                // so the `state_field` handlers map a logical scalar/array
                // index to the flat register slot the resume reader seeded.
                // Inlined sub-frames carry no state-field ops, but the root
                // dispatch frame (the chain tail, which owns the merge point
                // and the loop's state load/store) does, so it must hold the
                // layout before it runs.
                let sf_layout = state.state_field_layout();
                // Seed the reconstructed blackhole chain with the registered
                // virtualizable info + identity pointer so a mid-body
                // vable-array opcode (`getarrayitem_vable_*` /
                // `setarrayitem_vable`) can resolve its vinfo during resume.
                // Two sources:
                //   * the portal-inline experiment (gated), OR
                //   * a state-field machine (no `vable_token` field), whose
                //     `bh_clear_vable_token` is inert (the `state` struct has
                //     no heap token), so a non-null vinfo cannot corrupt it.
                //     The overflow-guard deopt (`int_*_jump_if_ovf` on the
                //     `[int; virt]` regs) is the first path to run a blackhole
                //     vable-array op on such a machine.
                // A heap virtualizable is seeded as well: a guard that fails
                // before a later `getfield_vable_*` has to resolve that
                // descr. `clear_vable_token` is a no-op while the token is
                // zero. The identity pointer must be co-seeded because the GC-root
                // walk (`resume_mainloop`) dereferences `virtualizable_ptr`
                // whenever `virtualizable_info` is non-null.
                let seed_vinfo_ptr = seed_deopt_vinfo_ptr(self.meta.virtualizable_info());
                {
                    let mut current = Some(&mut *bh);
                    while let Some(frame) = current {
                        frame.state_field_layout = sf_layout.clone();
                        if !seed_vinfo_ptr.is_null() {
                            frame.virtualizable_info = seed_vinfo_ptr;
                            frame.virtualizable_ptr = vable_ptr;
                        }
                        current = frame.nextblackholeinterp.as_deref_mut();
                    }
                }
                let exc =
                    crate::blackhole::BlackholeInterpreter::prepare_resume_from_failure(guard_exc);
                // Drive the reconstructed frame chain (blackhole.py _run_forever
                // `_run_forever`): each completed sub-frame passes its
                // return value to its caller (resume_mainloop did the
                // `setup_return_value_*` copy), then we descend to the
                // caller until a frame raises a control exception — CRN at
                // a merge point, DoneWithThisFrame at the root, or an
                // uncaught exception. Unlike `run_forever_with_portal`,
                // which consumes every frame and so leaves no register file
                // to read, keep the terminal frame alive: PyPy re-enters
                // the portal with the CRN reds (warmspot.py:970-983), but
                // those are only the declared subset, so the structural
                // equivalent for majit's live `state` struct is to restore
                // the terminal frame's full register bank
                // (resume.py _callback_i seeded it in slot order) and resume
                // at the CRN green pc.
                let mut cur_exc = exc;
                let outcome = loop {
                    match bh.resume_mainloop(cur_exc) {
                        Ok(next_exc) => match bh.nextblackholeinterp.take() {
                            Some(caller) => {
                                // Layout + vinfo were seeded across the
                                // whole chain before the loop started.
                                bh_builder.release_interp(bh);
                                bh = caller;
                                cur_exc = next_exc;
                            }
                            // The bottommost frame always raises
                            // (done_with_this_frame / exit_frame_with_
                            // exception), so an `Ok` with no caller is
                            // unreachable.  Nothing returned a result on
                            // this route, so name it a bail rather than a
                            // void completion the caller would install.
                            None => break crate::jitexc::JitException::BailToInterpreter,
                        },
                        Err(jit_exc) => break jit_exc,
                    }
                };
                if crate::majit_log_enabled() {
                    eprintln!("[bh] back_edge_internal: chain resume → {:?}", outcome);
                }
                let portal_jc = self.dispatch_jitcode().cloned();
                let banks_from_bh =
                    |this: &mut Self, bh: &crate::blackhole::BlackholeInterpreter| {
                        let jc = portal_jc.as_deref().unwrap_or(bh.jitcode.as_ref());
                        let mut args = this.meta.take_portal_resume_scratch();
                        match jc.merge_point_green_regs() {
                            Some((gi, gr, gf)) => {
                                args.fill_greens_from_named_registers(
                                    gi,
                                    gr,
                                    gf,
                                    &bh.registers_i,
                                    &bh.registers_r,
                                    &bh.registers_f,
                                );
                            }
                            None => args.copy_green_banks_from(&[], &[], &[]),
                        }
                        args
                    };
                // `compile.py` `resume_in_blackhole`. The walk runs the
                // reconstructed chain to the next merge point (or out of
                // the frame) and hands that pc back to the interpreter.
                // It does not record a bridge: `handle_fail` already
                // tried `_trace_and_compile_from_bridge` from the guard's
                // own resumedescr, and a decline is the blackhole branch,
                // not a second trace started at the merge point the walk
                // stopped on.
                let resume_pc = match outcome {
                    // Next merge point reached (loop back-edge): flush the
                    // register file into the live state and resume the
                    // interpreter at the merge point's green pc.
                    crate::jitexc::JitException::ContinueRunningNormally(args) => {
                        let green_int = &args.green_int;
                        // PyPy re-enters the portal with the CRN greens
                        // (warmspot.py:970-983), so the interpreter
                        // resumes at the pc the re-executed
                        // jit_merge_point reported — every `greens`
                        // declaration puts `pc` first (jit_merge_point
                        // bucket order), so it is `green_int[0]`. The
                        // blackhole may have crossed a non-back-edge
                        // path (e.g. a branch fall-through that leaves
                        // the loop), in which case the green pc differs
                        // from the loop-header `target_pc`.
                        // `as usize` wraps a negative green into a huge pc
                        // and the interpreter resumes there. That is not a
                        // position; end the dispatch loop instead of falling
                        // back to the loop-header `target_pc`, which would
                        // re-execute the opcodes the blackhole already ran.
                        let green_pc = green_int
                            .first()
                            .copied()
                            .map(|pc| usize::try_from(pc).unwrap_or(usize::MAX));
                        if portal_rca_enabled() {
                            eprintln!(
                                "[portal-rca][crn] target_pc={} green_int={:?} \
                                 selected_green_pc={:?}",
                                target_pc, green_int, green_pc,
                            );
                        }
                        if let Some(pc) = green_pc {
                            let _ = crate::handle_portal_crn_hook(target_pc, pc);
                        }
                        // `restore_banked` consumes both banks densely
                        // (int slot j at index j, ref scalar j at index
                        // j); the register file holds them at
                        // `scalar_slot(j) = int_scalar_base + j` /
                        // `ref_scalar_slot(j) = ref_scalar_base + j`, so
                        // slice off the dispatch JitCode's argument
                        // prefix in each bank.
                        let layout = state.state_field_layout();
                        let int_base = layout.int_scalar_base.min(bh.registers_i.len());
                        let ref_base = layout.ref_scalar_base.min(bh.registers_r.len());
                        if crate::callee_rca_enabled() {
                            eprintln!(
                                "[callee-rca][crn-pre] green_pc={:?} target_pc={} \
                                 layout={:?} int_base={} ref_base={}",
                                green_pc, target_pc, layout, int_base, ref_base,
                            );
                            eprintln!(
                                "[callee-rca][crn-pre] bh.registers_i(len={})={:?}",
                                bh.registers_i.len(),
                                bh.registers_i,
                            );
                            eprintln!(
                                "[callee-rca][crn-pre] bh.registers_r(len={})={:?}",
                                bh.registers_r.len(),
                                bh.registers_r,
                            );
                            eprintln!(
                                "[callee-rca][crn-pre] int_slice={:?}",
                                &bh.registers_i[int_base..],
                            );
                            eprintln!(
                                "[callee-rca][crn-pre] ref_slice={:?}",
                                &bh.registers_r[ref_base..],
                            );
                            if let Some(dump) = state.debug_state_fields(&compiled_meta) {
                                eprintln!("[callee-rca][crn-state-before]\n{dump}");
                            }
                        }
                        if layout.num_vable_identity_slots == 0 {
                            let float_base = layout.float_scalar_base.min(bh.registers_f.len());
                            state.restore_banked3(
                                &compiled_meta,
                                &bh.registers_i[int_base..],
                                &bh.registers_r[ref_base..],
                                &bh.registers_f[float_base..],
                            );
                        } else {
                            writeback_live_state_scalars_from_blackhole(
                                state,
                                &layout,
                                &bh,
                                &all_liveness,
                            );
                        }
                        // The carried/delta-tracked reds in the deadframe
                        // can lag the authoritative live storage (e.g. a
                        // `stacksize` whose backing stack was popped by the
                        // compiled body but whose red was captured stale).
                        // Re-derive them from storage so the green recomputed
                        // at the resumed `jit_merge_point` (here: `stackok =
                        // req_size <= stacksize`) reflects reality instead of
                        // the stale red, otherwise the interpreter re-enters
                        // the same compiled loop and re-fails the same green
                        // guard with zero forward progress.
                        state.recover_after_compiled_run();
                        if crate::callee_rca_enabled()
                            && let Some(dump) = state.debug_state_fields(&compiled_meta)
                        {
                            eprintln!("[callee-rca][crn-state-after]\n{dump}");
                        }
                        // The register file just flushed above is the state
                        // AT the green pc — the blackhole ran the failing
                        // opcode forward to the next merge point. Resuming
                        // at the loop-header `target_pc` re-executes the
                        // header..green opcodes on post-green state. A CRN
                        // with no green pc has no position to resume at, so
                        // end the dispatch loop the same way a negative
                        // green does.
                        // Copy into the recycled driver buffer before
                        // `recycle_merge_point_args` clears the blackhole
                        // lists. `warmspot.py ll_portal_runner` rebuilds
                        // every green, not only `green_int[0]`.
                        let mut dest = self.meta.take_portal_resume_scratch();
                        dest.copy_from_args(&args);
                        bh.recycle_merge_point_args(args);
                        Some(continue_with_args(dest))
                    }
                    // The interpreted frame ran to completion inside the
                    // blackhole: flush, then force the generated mainloop's
                    // `while pc < len` guard to fail so it exits and
                    // returns the value computed from the flushed state.
                    outcome @ (crate::jitexc::JitException::DoneWithThisFrameVoid
                    | crate::jitexc::JitException::DoneWithThisFrameInt(_)
                    | crate::jitexc::JitException::DoneWithThisFrameRef(_)
                    | crate::jitexc::JitException::DoneWithThisFrameFloat(_)) => {
                        let layout = state.state_field_layout();
                        let int_base = layout.int_scalar_base.min(bh.registers_i.len());
                        let ref_base = layout.ref_scalar_base.min(bh.registers_r.len());
                        let float_base = layout.float_scalar_base.min(bh.registers_f.len());
                        state.restore_banked3(
                            &compiled_meta,
                            &bh.registers_i[int_base..],
                            &bh.registers_r[ref_base..],
                            &bh.registers_f[float_base..],
                        );
                        // `warmspot.py ll_portal_runner` returns
                        // `_done_with_this_frame` / `DoneWithThisFrameDescr*`
                        // `get_result` without re-entering the portal body.
                        // `function_entry_runner` maps `Done(None)` to `Run`,
                        // and that arm drains `back_edge_finish_word` first.
                        // Leaving the latch empty re-enters the portal at the
                        // function-entry pc with the virtualizable the residual
                        // already mutated.
                        match &outcome {
                            crate::jitexc::JitException::DoneWithThisFrameInt(v) => {
                                self.meta.back_edge_finish_word = Some(*v);
                            }
                            crate::jitexc::JitException::DoneWithThisFrameFloat(v) => {
                                self.meta.back_edge_finish_word = Some(v.to_bits() as i64);
                            }
                            crate::jitexc::JitException::DoneWithThisFrameRef(v) => {
                                self.meta.back_edge_finish_word = Some(v.as_usize() as i64);
                            }
                            _ => {}
                        }
                        Some(PortalResume::DoneWithThisFrame(Some(banks_from_bh(
                            self, &bh,
                        ))))
                    }
                    // blackhole.py `_exit_frame_with_exception` →
                    // warmspot.py:998-1005: the resumed chain raised an
                    // exception that escaped every reconstructed frame
                    // (the hand-rolled loop above drove the multi-frame
                    // descent, propagating the exc to the bottommost frame
                    // just like `_run_forever`). resume_in_blackhole has
                    // finished; hand the exception to the interpreter's own
                    // machinery (parity: `eval.rs` re-raises the stored Ref
                    // via `PyError::from_exc_object`) instead of dropping
                    // back to re-execute from the guard pc. Returns the
                    // interpreter's handler pc, or `None` when it has no
                    // exception machinery — then the crude fallback runs,
                    // but an exception-less interpreter never reaches here.
                    // See the twin arm in the tracing-abort adoption
                    // above: a bail names a jitcode coordinate this
                    // driver cannot resume at, and the chain has already
                    // run, so ending the dispatch loop is the closest
                    // non-replaying answer.  The banked state is restored
                    // first because the blackhole registers hold what the
                    // chain computed before it stopped.
                    crate::jitexc::JitException::BailToInterpreter => {
                        let layout = state.state_field_layout();
                        let int_base = layout.int_scalar_base.min(bh.registers_i.len());
                        let ref_base = layout.ref_scalar_base.min(bh.registers_r.len());
                        let float_base = layout.float_scalar_base.min(bh.registers_f.len());
                        state.restore_banked3(
                            &compiled_meta,
                            &bh.registers_i[int_base..],
                            &bh.registers_r[ref_base..],
                            &bh.registers_f[float_base..],
                        );
                        Some(PortalResume::BailToInterpreter(Some(banks_from_bh(
                            self, &bh,
                        ))))
                    }
                    crate::jitexc::JitException::ExitFrameWithExceptionRef(exc_ref) => {
                        let args = Some(banks_from_bh(self, &bh));
                        match state.deliver_blackhole_exception(exc_ref) {
                            Some(pc) => Some(PortalResume::ExitFrameWithException { pc, args }),
                            None => None,
                        }
                    }
                };
                bh_builder.release_interp(bh);
                // `BlackholeInterpBuilder.release_interp` makes the frame
                // reusable as soon as `_run_forever` is done. The
                // interpreter may re-enter the portal on the pc we return,
                // so the builder goes back to the pool before that.
                drop(bh_builder);
                if let Some(resume) = resume_pc {
                    if let Some(raw_values) = raw_values_for_bridge {
                        self.exit_raw_scratch_out(raw_values);
                    }
                    return Some(resume);
                }
            }
        }

        // The chain raised an exception and
        // `deliver_blackhole_exception` returned `None` (the
        // interpreter has no exception machinery — unreachable for an
        // exception-less interpreter): fall back to crude state recovery and
        // resume the interpreter at the guard pc.
        //
        // `AbstractResumeGuardDescr.handle_fail` parity: a guard failure
        // that does not compile is answered by `resume_in_blackhole`, which
        // rebuilds the frame chain and `setposition`s it at the pc the
        // guard's own resume data encodes. The header pc of the guard's
        // recovery frame is that position here. Resolved at the head of
        // this block for two reasons. It sits inside the block because the
        // block is the only reader — every other exit from the guard arm
        // returns a pc the bridge or the blackhole reported — and it sits
        // ahead of the retirement below because the index lookup is keyed on
        // a compiled loop that `remove_compiled_loop` is about to drop:
        // asking afterwards finds nothing and silently answers `target_pc`,
        // which resumes at the loop entry against state the recovery has
        // already rewound. The failing guard's own layout carries the same
        // header pc, so reading it first also answers without a lookup.
        // The backend may have stamped its trace's header pc on the
        // descr (`CompiledTraceInfo`); otherwise the pc is the one the
        // guard's own `rd_numb` encodes for its outermost frame.
        let guard_resume_pc = fd
            .trace_info_any()
            .and_then(|info| {
                info.downcast_ref::<majit_backend::CompiledTraceInfo>()
                    .map(|info| info.header_pc)
            })
            .or_else(|| crate::compile::guard_resume_pc(fd))
            .map(|pc| pc as usize)
            .unwrap_or(target_pc);
        state.recover_after_compiled_run();
        self.invalidate_loop(green_key);
        self.remove_compiled_loop(green_key);
        self.abort_entry_tracing(green_key);
        if let Some(raw_values) = raw_values_for_bridge {
            self.exit_raw_scratch_out(raw_values);
        }
        let _ = guard_resume_pc;
        panic!(
            "compiled guard failure did not produce ContinueRunningNormally \
             from the blackhole (blackhole.py resume_in_blackhole / \
             bhimpl_jit_merge_point)"
        );
    }

    fn back_edge_or_run_compiled_internal(
        &mut self,
        green_key: u64,
        structured_green_key: Option<&dyn Fn() -> GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        if self.meta.is_tracing() {
            return None;
        }
        if let Some(handled) =
            self.try_empty_chain_tick(green_key, structured_green_key, target_pc, state, env)
        {
            let _ = handled;
            return None;
        }
        let single_pass_dispatch_key: Option<u32> = None;
        if !state.can_trace() {
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-tracing",
                    &format!("back_edge blocked: can_trace=false key={green_key}"),
                );
            }
            return None;
        }

        // Same `warmstate.py:471-478` gate as `back_edge_internal`: a cell
        // holding only a `compile_tmp_callback` token has no `compiled_loops`
        // meta, and upstream counts it normally instead of entering it. Without
        // the meta the runner below can only return `Abort`, which stops this
        // route from ever reaching `maybe_start_tracing` — i.e. the counter
        // stops ticking and the real loop is never traced.
        if self.has_confirmed_runnable_compiled_loop(green_key) {
            return Some(self.run_compiled_detailed_keyed_with_dispatch_key(
                green_key,
                state,
                pre_run,
                single_pass_dispatch_key,
            ));
        }

        self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env);
        None
    }

    /// Force-start tracing for a function entry.
    ///
    /// Bypasses the WarmEnterState hot counter (the caller already did its own
    /// counting). Used for function-entry JIT where the threshold is
    /// controlled externally.
    pub fn force_start_tracing(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) {
        // `gctypelayout.py encode_type_shapes_now` closes `type_info_group`
        // at translation. Close before tracing so a reader in this trace
        // sees the frozen table, and JIT-only types can still register
        // after startup.
        majit_gc::ensure_type_registry_closed();
        // Note: no is_hot_or_tracing check here — the caller (try_function_entry_jit)
        // already verified the threshold. force_start_tracing must unconditionally start.
        //
        // Every compile this trace can reach decodes frame value counts through
        // the per-thread store, so aim it at this driver before recording a
        // single op. Free when it already points here (one `u64` compare).
        self.republish_state_field_fvc();
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        // Resolved here with the descriptor and carried to both ends of
        // the run; see `sync_before` for why it is not asked for twice.
        let vable = descriptor
            .as_deref()
            .and_then(JitDriverStaticData::virtualizable);
        // Precondition: not entered while `is_tracing`.  The caller decides
        // that a trace starts here, and one cannot start inside another.  So
        // `sync_before` finds no recording context, writes no trace pointer,
        // and the returns below have none to put back; the two
        // `run_compiled_detailed_*` runners carry the save/restore instead,
        // because a walk does reach those.  `initialize_virtualizable` seeds
        // the cell on the context this call is about to build.
        debug_assert!(
            !self.is_tracing(),
            "force_start_tracing entered while a recording is live"
        );
        if !self.sync_before(state, &meta, vable) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(
            descriptor.as_deref(),
            &live_values,
            state.state_field_layout().total_live_values(),
        ) {
            return;
        }

        if let BackEdgeAction::StartedTracing = self.meta.force_start_tracing(
            green_key,
            (state.code_ptr(), target_pc),
            descriptor.map(|d| (*d).clone()),
            &live_values,
        ) {
            if spdiag_enabled() {
                eprintln!("@@@SPDIAG StartedTracing target_pc={target_pc} green_key={green_key}");
            }
            if let Some(ctx) = self.meta.trace_ctx() {
                ctx.header_pc = target_pc;
            }
            let mut sym = S::create_sym(&meta, target_pc);
            state.initialize_sym(&mut sym, &meta);
            self.sym = Some(sym);
            self.meta.begin_trace_session(meta);
        }
    }

    #[cold]
    #[inline(never)]
    /// RPython warmstate.py bound_reached parity.
    ///
    /// Called when the counter threshold has ALREADY fired (from eval_loop_jit).
    /// Bypasses counter.tick() inside maybe_compile, allowing decay_counters()
    /// to be called before tracing starts.
    pub fn bound_reached(&mut self, green_key: u64, target_pc: usize, state: &mut S, env: &S::Env) {
        if self.meta.is_tracing() {
            return;
        }
        self.take_single_pass_label_entry_dispatch_key_for_back_edge(green_key);
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        // Resolved here with the descriptor and carried to both ends of
        // the run; see `sync_before` for why it is not asked for twice.
        let vable = descriptor
            .as_deref()
            .and_then(JitDriverStaticData::virtualizable);
        if !self.sync_before(state, &meta, vable) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(
            descriptor.as_deref(),
            &live_values,
            state.state_field_layout().total_live_values(),
        ) {
            return;
        }

        if let BackEdgeAction::StartedTracing = self.meta.bound_reached(
            green_key,
            (state.code_ptr(), target_pc),
            None,
            descriptor.map(|d| (*d).clone()),
            &live_values,
        ) {
            if spdiag_enabled() {
                eprintln!("@@@SPDIAG StartedTracing target_pc={target_pc} green_key={green_key}");
            }
            if let Some(ctx) = self.meta.trace_ctx() {
                ctx.header_pc = target_pc;
            }
            let mut sym = S::create_sym(&meta, target_pc);
            state.initialize_sym(&mut sym, &meta);
            self.sym = Some(sym);
            self.meta.begin_trace_session(meta);
        }
    }

    fn maybe_start_tracing(
        &mut self,
        green_key: u64,
        structured_green_key: Option<&dyn Fn() -> GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> bool {
        // Occupied-bucket path: keep the typed comparekey walk so a
        // comparator-less cell at the hash cannot answer for a sibling
        // (`chained_ceiling_latch_does_not_refuse_the_typed_sibling`).
        // The empty-chain door never reaches here; it ticked already.
        match self.meta.on_back_edge_typed_decision_on_driver(
            self.index().unwrap_or(0),
            green_key,
            (state.code_ptr(), target_pc),
        ) {
            HotResult::StartTracing => {
                self.commit_start_tracing(green_key, structured_green_key, target_pc, state, env)
            }
            _ => false,
        }
    }

    /// Same decision as [`Self::try_empty_chain_tick`], without building the
    /// live-value list. A true tick is consumed here and reported as
    /// [`BackEdgeWarmth::Trace`]; [`Self::commit_back_edge_trace`] is the
    /// `bound_reached` half and must not tick again.
    #[inline]
    pub fn back_edge_warmth(&mut self, green_key_hash: u64, state: &S) -> BackEdgeWarmth {
        if self.meta.is_tracing() {
            return BackEdgeWarmth::Full;
        }
        if !self.meta.cut_compiled_keys.is_empty() {
            return BackEdgeWarmth::Full;
        }
        let jd_no = self.index().unwrap_or(0);
        if let Some(cell) = self
            .meta
            .warm_state_ref_for_driver(jd_no)
            .and_then(|warm| warm.lookup_chain(green_key_hash))
        {
            if cell.next.borrow().is_some() {
                return BackEdgeWarmth::Full;
            }
            if cell.cell_bucket.get() == green_key_hash {
                if cell.is_compiled() {
                    return BackEdgeWarmth::Full;
                }
                if cell.is_tracing() {
                    return BackEdgeWarmth::Interpret;
                }
                if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
                    return BackEdgeWarmth::Full;
                }
                if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
                    return BackEdgeWarmth::Full;
                }
            }
        }
        if !state.can_trace() {
            return BackEdgeWarmth::Interpret;
        }
        if self
            .meta
            .warm_state_for_driver(jd_no)
            .tick_empty_chain(green_key_hash)
        {
            BackEdgeWarmth::Trace
        } else {
            BackEdgeWarmth::Interpret
        }
    }

    /// `warmstate.py` `bound_reached` after the caller has published live values.
    /// The counter tick already fired; this does not tick again.
    #[cold]
    #[inline(never)]
    pub fn commit_back_edge_trace(
        &mut self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> bool {
        self.commit_start_tracing(green_key_hash, Some(&make_green_key), target_pc, state, env)
    }

    /// `warmstate.py bound_reached`: the tick has already fired, so build
    /// the frontend snapshot and start the trace. Not reached on a cold
    /// back edge — `maybe_compile_and_run` only calls `bound_reached` when
    /// `tick` returns True.
    #[cold]
    #[inline(never)]
    fn commit_start_tracing(
        &mut self,
        green_key: u64,
        structured_green_key: Option<&dyn Fn() -> GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> bool {
        if spdiag_enabled() {
            eprintln!("@@@SPDIAG maybe_start_tracing target_pc={target_pc} green_key={green_key}");
        }
        // Denominator for slots 62/63, bumped before either refusal can return
        // so both are readable as fractions rather than bare totals.  Only this
        // door is counted: `force_start_tracing` and `bound_reached` open with
        // the same two checks, and folding three doors into one slot would say
        // nothing about any of them.
        crate::mc_diag_bump(61); // mst_entered
        // Slot 61 counts entries into this bound_reached commit, not cold
        // ticks. `warmstate.py maybe_compile_and_run` only calls
        // `bound_reached` when `tick` returns True, so a NotHot back edge
        // never arrives here. Slots 62/63 are refusals of this door and
        // stay zero until 61 has moved. The abort ceiling is NOT consulted
        // separately here. It is a refusal the decision below already owns:
        // `maybe_compile_decision_with_key` takes it in the position
        // `warmstate.py maybe_compile_and_run` gives `confirm_enter_jit`,
        // above the counter tick and above the `decay_all_counters` inside
        // `bound_reached`, so a latched cell reaches neither — which is the
        // whole reason the ceiling decides early. Asking first resolved the
        // same cell through the same typed chain walk a second time, for an
        // answer `is_ceiling_latched_agrees_with_the_decision_it_mirrors`
        // pins equal to the one the decision reaches; `maybe_compile_and_run`
        // binds its cell once ("to avoid computing the hash several times")
        // and answers every question about it off that binding. Slot 81
        // counts the same population either way, because only one of the two
        // refusals can fire for a given edge: every state in which the
        // mirror answered `false` returns above the decision's ceiling arm
        // or fails its condition. A latched cell on the empty-chain door
        // bumps 81 there and never enters this function.

        // The liveness decoder's thread-local payload is only a publication
        // mechanism; ownership stays with this driver. Another live portal may
        // have published after this one was constructed, so re-aim the slot on
        // the ordinary hot-counter trace entry too. `force_start_tracing` and
        // `start_bridge_tracing` already do the same at their entry doors.
        self.republish_state_field_fvc();

        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        // Resolved here with the descriptor and carried to both ends of
        // the run; see `sync_before` for why it is not asked for twice.
        let vable = descriptor
            .as_deref()
            .and_then(JitDriverStaticData::virtualizable);
        if !self.sync_before(state, &meta, vable) {
            crate::mc_diag_bump(62); // mst_sync_before_false
            return false;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(
            descriptor.as_deref(),
            &live_values,
            state.state_field_layout().total_live_values(),
        ) {
            crate::mc_diag_bump(63); // mst_live_values_mismatch
            return false;
        }

        match self.meta.start_back_edge_trace(
            green_key,
            (state.code_ptr(), target_pc),
            // Neither the typed key nor the descriptor is built before the
            // counter decision. Both are consumed only while committing its
            // `StartTracing` answer, so the factory and a borrow avoid a deep
            // clone until this cold arm.
            structured_green_key,
            descriptor.as_deref(),
            &live_values,
        ) {
            BackEdgeAction::Interpret => false,
            BackEdgeAction::StartedTracing => {
                if spdiag_enabled() {
                    eprintln!(
                        "@@@SPDIAG StartedTracing target_pc={target_pc} green_key={green_key}"
                    );
                }
                if let Some(ctx) = self.meta.trace_ctx() {
                    ctx.header_pc = target_pc;
                }
                let mut sym = S::create_sym(&meta, target_pc);
                state.initialize_sym(&mut sym, &meta);
                self.sym = Some(sym);
                self.meta.begin_trace_session(meta);
                true
            }
            BackEdgeAction::AlreadyTracing | BackEdgeAction::RunCompiled => false,
        }
    }

    /// Borrow [`Self::entry_scratch`] for the duration of one compiled entry.
    ///
    /// Handed out by value rather than by reference because assembling the
    /// entry arguments calls back into `self`; every caller must return it
    /// through [`Self::entry_scratch_out`] before leaving, or the next entry
    /// starts from empty buffers (correct, just slower).
    ///
    /// The buffers arrive with their previous contents dropped and their
    /// capacity kept — see `JitState::export_virtualizable_boxes_into`.
    ///
    /// A nested entry finds the slot empty and builds its own set, which is the
    /// `unwrap_or_default` arm; the allocation it costs is the price of nesting
    /// and not of the ordinary entry, which finds the box the last one returned.
    fn take_entry_scratch(&mut self) -> Box<EntryScratch> {
        let mut scratch = self.entry_scratch.take().unwrap_or_default();
        scratch.live_values.clear();
        scratch.raw.clear();
        scratch.types.clear();
        scratch.vable_boxes.clear();
        scratch.vable_lengths.clear();
        scratch
    }

    /// [`Self::take_entry_scratch`] for the raw red path of
    /// `execute_assembler`.
    ///
    /// `maybe_compile_and_run` unspecializes reds into one word buffer and
    /// `func_execute_token` reads that buffer. Only `raw` is cleared. A caller
    /// that fills `live_values`, `types`, `vable_boxes` or `vable_lengths`
    /// clears those first via [`EntryScratch::clear_typed`].
    fn take_entry_scratch_raw(&mut self) -> Box<EntryScratch> {
        let mut scratch = self.entry_scratch.take().unwrap_or_default();
        scratch.raw.clear();
        scratch
    }

    /// Return the buffers [`Self::take_entry_scratch`] handed out.
    fn entry_scratch_out(&mut self, scratch: Box<EntryScratch>) {
        self.entry_scratch = Some(scratch);
    }

    fn take_exit_raw_scratch(&mut self) -> Vec<i64> {
        let mut raw = self.exit_raw_scratch.take().unwrap_or_default();
        raw.clear();
        raw
    }

    fn exit_raw_scratch_out(&mut self, raw: Vec<i64>) {
        self.exit_raw_scratch = Some(raw);
    }

    /// The driver's static data, resolved once and then shared.
    ///
    /// See [`Self::descriptor_cache`] for why the answer is memoized rather
    /// than rebuilt per consultation. Callers that need to hand ownership on
    /// (the trace-start and bridge-setup paths) clone out of the shared value;
    /// the per-entry paths only read through it.
    /// Address of the cached driver descriptor. The `Arc` in
    /// [`Self::descriptor_cache`] keeps it alive for the driver's life.
    fn driver_descriptor_ptr(
        &mut self,
        state: &S,
        meta: &S::Meta,
    ) -> Option<*const JitDriverStaticData> {
        if self.descriptor_cache.is_none() {
            let resolved = self
                .descriptor
                .clone()
                .or_else(|| state.driver_descriptor(meta))
                .map(std::sync::Arc::new);
            self.descriptor_cache = Some(resolved);
        }
        self.descriptor_cache
            .as_ref()
            .and_then(|cached| cached.as_ref().map(|descr| std::sync::Arc::as_ptr(descr)))
    }

    fn driver_descriptor_for(
        &mut self,
        state: &S,
        meta: &S::Meta,
    ) -> Option<std::sync::Arc<JitDriverStaticData>> {
        if let Some(cached) = &self.descriptor_cache {
            return cached.clone();
        }
        let resolved = self
            .descriptor
            .clone()
            .or_else(|| state.driver_descriptor(meta))
            .map(std::sync::Arc::new);
        self.descriptor_cache = Some(resolved.clone());
        resolved
    }

    fn live_values_match_descriptor(
        descriptor: Option<&JitDriverStaticData>,
        live_values: &[Value],
        state_field_live_count: usize,
    ) -> bool {
        let Some(descriptor) = descriptor else {
            return true;
        };
        // State-field JIT whose schema carries a vable_var red: the flat
        // `StateFieldLayout` live values ARE the runtime red shape (the values
        // the compiled entry is invoked with via `extract_live_values` — int
        // slots flow to `registers_i`, ref-bank scalars to `registers_r`).
        // The logical descriptor reds are the BC_JIT_MERGE_POINT payload
        // schema, in which the vable_var array appears as a single box
        // (red_schema's `vable_var` candidate); extract_live flat-expands it
        // into scalar + (ptr, len) i64 slots, so the flat live count cannot
        // equal the logical red count.  Validate the flat live count against
        // the declared flat layout instead — the logical-reds shape check
        // below is a category mismatch for the flat model.  This precedes the
        // reds check so state-field drivers that declare no logical reds are
        // still validated (not blanket-passed).  (`state_field_live_count == 0`
        // for non-state-field drivers such as pyre, which keep the strict reds
        // check.)
        if state_field_live_count != 0 {
            return live_values.len() == state_field_live_count;
        }
        // warmstate.py execute_assembler parity: the runtime entrypoint
        // is called with the JitDriver reds only.  An empty redlist therefore
        // requires zero live values (jtransform.py:1697-1703 validates the
        // redlist strictly); the shape check below covers that case rather
        // than blanket-accepting any live-value shape.  If the state reports
        // any other live-value shape while a descriptor is active, refuse the
        // run instead of silently accepting a pyre-specific prefix match.
        let reds = descriptor.reds();
        if live_values.len() != reds.len() {
            return false;
        }
        reds.iter()
            .zip(live_values.iter())
            .all(|(var, value)| var.tp == value.get_type())
    }

    /// warmstate.py:482-511: extend live values with virtualizable fields
    /// when the compiled loop expects more inputs than currently available.
    /// RPython always has jitdriver_sd available; pyre may have descriptor=None
    /// when re-entering from guard failure, so fall back to virtualizable_info
    /// directly.
    ///
    /// Appends in place and reports success, so the compiled-entry path can
    /// hand it a buffer that already has the artifact's capacity instead of a
    /// freshly minted one. `boxes` and `array_lengths` are scratch for the export;
    /// they arrive cleared and their contents are not read on return. On
    /// `false` `live_values` is restored to the length it arrived with, so a
    /// declining caller sees the buffer it passed.
    fn extend_compiled_live_values_into(
        &self,
        procedure_token: &majit_backend::JitCellToken,
        state: &S,
        meta: &S::Meta,
        virtualizable: Option<&JitDriverVar>,
        live_values: &mut Vec<Value>,
        boxes: &mut Vec<i64>,
        array_lengths: &mut Vec<usize>,
    ) -> bool {
        // `warmstate.py:188 cell.loop_token` is the single PyPy source of
        // truth for the entry-path inputarg shape, and this token IS what
        // `warm_state.get_compiled` returns — so the shape still comes off the
        // cell and never off pyre's `compiled_loops` side table (the
        // retirement target). Arrives resolved for the reason `sync_before`
        // gives: `warmstate.py maybe_compile_and_run` reads the cell's token
        // once and carries the object into `execute_assembler`, where asking
        // the celltable again costs a walk, a `Weak::upgrade` and an `Arc`
        // drop for one length.
        let compiled_inputs = procedure_token.inputarg_types().len();
        if compiled_inputs <= live_values.len() {
            return true;
        }
        // Fall back to virtualizable_info's default name if no descriptor.
        let Some(info) = self.meta.virtualizable_info() else {
            return false;
        };
        // Try the driver's declared virtualizable first (RPython
        // jitdriver_sd.virtualizable), else jitdriver_sd.virtualizable_info.name
        // (interp_jit.py:25). Borrowed rather than cloned: both spellings
        // outlive this call, and the export below only reads the name. Arrives
        // pre-resolved for the reason given on `sync_before` — this is the
        // third consultation on one compiled entry, and the caller already
        // holds the answer.
        let name: &str = match virtualizable {
            Some(virtualizable) => &virtualizable.name,
            None => &info.name,
        };
        if !state.export_virtualizable_boxes_into(meta, name, info, boxes, array_lengths) {
            return false;
        }
        let base = live_values.len();
        live_values.extend(
            boxes
                .iter()
                .zip(info.box_types(array_lengths))
                .map(|(&raw, ty)| crate::pyjitpl::heap_value_for_pub(ty, raw)),
        );
        if live_values.len() != compiled_inputs {
            live_values.truncate(base);
            return false;
        }
        true
    }

    /// Owning form of [`Self::extend_compiled_live_values_into`], for the
    /// bridge-setup and diagnostic paths that hold a `Vec` of their own and
    /// run far from the steady entry path.
    fn extend_compiled_live_values(
        &self,
        green_key: u64,
        state: &S,
        meta: &S::Meta,
        virtualizable: Option<&JitDriverVar>,
        mut live_values: Vec<Value>,
    ) -> Option<Vec<Value>> {
        // The cold form keeps the lookup: its callers hold a green key and no
        // token.
        let procedure_token = self.get_loop_token(green_key)?;
        let mut boxes = Vec::new();
        let mut array_lengths = Vec::new();
        self.extend_compiled_live_values_into(
            &procedure_token,
            state,
            meta,
            virtualizable,
            &mut live_values,
            &mut boxes,
            &mut array_lengths,
        )
        .then_some(live_values)
    }

    /// The declared virtualizable red of the driver this entry runs under,
    /// or `None` for a driver that declares none.
    ///
    /// Taken pre-resolved rather than looked up here: the driver's static
    /// data names its virtualizable by NAME, so asking it costs a scan of
    /// the variable list with a string compare per variable, and both this
    /// and [`Self::sync_after`] would pay it separately on one compiled
    /// entry. `warmspot.py:529-538` resolves the virtualizable's position
    /// once at driver setup (`jd.index_of_virtualizable`) and every runtime
    /// reader indexes with it; the callers here resolve once at the entry
    /// decision and carry the answer to both ends of the run for the same
    /// reason.
    fn sync_before(
        &mut self,
        state: &mut S,
        meta: &S::Meta,
        virtualizable: Option<&JitDriverVar>,
    ) -> bool {
        // Descriptor-gated token-reset path — RPython
        // compile.py::sync_before_jit calls `vinfo.reset_vable_token(virt)`
        // when the JitDriver has a virtualizable declared.  Pyre routes
        // this through the interpreter's `sync_named_virtualizable_before_jit`.
        if let Some(virtualizable) = virtualizable {
            let ok = state.sync_named_virtualizable_before_jit(
                meta,
                &virtualizable.name,
                self.meta.virtualizable_info().map(|a| a.as_ref()),
            );
            if !ok {
                return false;
            }
        }
        // `execute_assembler` clears the vable token only `if vinfo is not None`.
        // A driver with no virtualizable info has nothing to reset.
        if virtualizable.is_none() && self.meta.virtualizable_info().is_none() {
            return true;
        }

        // Refresh the trace-entry vable heap pointer so `initialize_virtualizable`
        // reads array lengths directly from the concrete virtualizable object
        // (pyjitpl.py `vinfo.read_boxes(cpu, virtualizable, startindex)`).
        // `VirtualizableInfo.name` is the canonical identifier declared via
        // `virtualizable!(name = "...")`, so we can feed it straight to the
        // interpreter hooks without going through the descriptor.
        self.meta.set_vable_ptr(std::ptr::null());
        self.meta.set_vable_array_lengths(Vec::new());
        let info_clone = self.meta.virtualizable_info().cloned();
        if let Some(ref info) = info_clone {
            if let Some(ptr) = state.virtualizable_heap_ptr(meta, &info.name, info) {
                self.meta.set_vable_ptr(ptr.cast_const());
                // The entry names the frame this run executes.  The exits
                // put the caller's frame back through
                // `restore_trace_vable_ptr`: `sync_after` for the ones that
                // take it, the decline and finish exits directly.
                if let Some(ctx) = self.meta.tracing.as_mut() {
                    ctx.set_virtualizable_heap_ptr(ptr.cast_const());
                }
            }
            // Fallback cache for layouts that cannot expose array length on
            // the heap object alone (header-less embedded arrays). Unused in
            // production since PyFrame's `locals_cells_stack_w` is embedded
            // with a length_offset; retained so tests that don't stage a
            // fake heap object still reach `initialize_virtualizable`.
            let fallback_lengths = state
                .virtualizable_array_lengths(meta, &info.name, info)
                .unwrap_or_default();
            self.meta.set_vable_array_lengths(fallback_lengths);
        }
        true
    }

    /// Puts back the recording context's virtualizable frame saved before a
    /// compiled entry. `sync_after` does this for the exits that take it; the
    /// decline and finish exits call this directly, because a walk that resumes
    /// recording must flush into its own frame whether or not the entry ran.
    fn restore_trace_vable_ptr(&mut self, saved: Option<*const u8>) {
        if let Some(ptr) = saved {
            if let Some(ctx) = self.meta.tracing.as_mut() {
                ctx.set_virtualizable_heap_ptr(ptr);
            }
        }
    }

    /// Counterpart of [`Self::sync_before`]; takes the same pre-resolved
    /// virtualizable red, and for the same reason. Restores the recording
    /// context's virtualizable pointer saved before that `sync_before`.
    fn sync_after(
        &mut self,
        state: &mut S,
        meta: &S::Meta,
        virtualizable: Option<&JitDriverVar>,
        saved_vable_ptr: Option<*const u8>,
    ) {
        self.restore_trace_vable_ptr(saved_vable_ptr);
        let Some(virtualizable) = virtualizable else {
            return;
        };
        state.sync_named_virtualizable_after_jit(
            meta,
            &virtualizable.name,
            self.meta.virtualizable_info().map(|a| a.as_ref()),
        );
    }

    /// Register an additional entry point for this driver.
    ///
    /// Multiple functions can share the same JitDriver and compiled loops.
    /// Each entry point has a name and a green key schema describing the
    /// types of its green (loop-invariant) variables.
    pub fn register_entry_point(&mut self, name: &str, green_key_schema: &[Type]) {
        // Accept IR `Type` for source compatibility. Callers that need to
        // declare STR/UNICODE greens can call `register_entry_point_typed`.
        let schema: Vec<GreenType> = green_key_schema.iter().copied().map(Into::into).collect();
        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            schema,
        });
    }

    /// warmstate.py:564-565 — register an entry point with an explicit
    /// `GreenType` schema. Frontends that declare STR / UNICODE green
    /// args call this path so `comparekey` / `get_uhash` dispatch through
    /// `equal_whatever` / `hash_whatever` correctly.
    pub fn register_entry_point_typed(&mut self, name: &str, green_key_schema: &[GreenType]) {
        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            schema: green_key_schema.to_vec(),
        });
    }

    /// Return all registered entry points.
    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.entry_points
    }

    /// Look up an entry point by name.
    pub fn find_entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.iter().find(|ep| ep.name == name)
    }

    /// Set virtualizable info for frame virtualization.
    pub fn set_virtualizable_info(&mut self, info: std::sync::Arc<VirtualizableInfo>) {
        self.meta.set_virtualizable_info(info);
    }

    /// Set the compilation threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .set_threshold(threshold);
    }

    /// RPython-compatible wrapper used by compile.py.
    pub fn send_loop_to_backend(
        &mut self,
        jump_args: &[OpRef],
        meta: S::Meta,
    ) -> crate::pyjitpl::CompileOutcome {
        self.meta.compile_loop(jump_args, meta)
    }

    /// RPython-compatible wrapper used by compile.py for bridges.
    pub fn send_bridge_to_backend(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> crate::pyjitpl::BridgeCompileResult {
        self.meta
            .close_bridge(green_key, trace_id, fail_index, finish_args)
    }

    /// Set the trace eagerness (guard failure threshold for bridge tracing).
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .set_param_trace_eagerness(eagerness);
    }

    /// Set a callback for loop compilation events.
    pub fn set_on_compile_loop(
        &mut self,
        f: impl Fn(u64, usize, usize, &[majit_ir::OpCode]) + Send + 'static,
    ) {
        self.meta.set_on_compile_loop(f);
    }

    /// Set a callback for bridge compilation events.
    ///
    /// `f` receives `(green_key, fail_index, num_ops)`. The loop sibling above
    /// cannot stand in for it: a guard that declines to bridge still deopts
    /// through the blackhole and the loop keeps running, so only this count
    /// distinguishes a bridge that was built from one that was passed over.
    pub fn set_on_compile_bridge(&mut self, f: impl Fn(u64, u32, usize) + Send + 'static) {
        self.meta.set_on_compile_bridge(f);
    }

    /// Set a callback for guard failure events.
    pub fn set_on_guard_failure(&mut self, f: impl Fn(u64, u32, u32) + Send + 'static) {
        self.meta.set_on_guard_failure(f);
    }

    /// Set a callback for swallowed compilation panics.
    ///
    /// `f` receives `(green_key)`. The tally this mirrors lives on the driver
    /// and is otherwise readable only through [`Self::get_stats`], which a
    /// process-global reader cannot window.
    pub fn set_on_internal_compile_panic(&mut self, f: impl Fn(u64) + Send + 'static) {
        self.meta.set_on_internal_compile_panic(f);
    }

    /// Set a callback fired immediately before a call enters compiled code.
    /// `f` receives `(green_key, target_pc)`.
    ///
    /// Scope: this driver reaches compiled code through `back_edge_internal`,
    /// which is the site the hook is installed at.  The sibling entry paths in
    /// this file serve other front ends and do not fire it yet, so a zero count
    /// is evidence only for a consumer whose entries all go through the back
    /// edge.
    pub fn set_on_compiled_entry(&mut self, f: impl Fn(u64, usize) + Send + 'static) {
        self.meta.set_on_compiled_entry(f);
    }

    /// Set a callback for trace abort events. `f` receives `(green_key, permanent)`
    /// where `permanent` is true when the abort disables the key permanently.
    pub fn set_on_trace_abort(&mut self, f: impl Fn(u64, bool) + Send + 'static) {
        self.meta.set_on_trace_abort(f);
    }

    /// Set a JIT parameter by name at runtime.
    ///
    /// Supported parameters:
    /// - `"threshold"` — compilation hot-count threshold
    /// - `"trace_eagerness"` — guard failure count before bridge tracing
    /// - `"function_threshold"` — function call count before inlining
    ///
    /// Unknown parameter names are silently ignored.
    pub fn set_param(&mut self, name: &str, value: i64) {
        // rlib/jit.py `set_param(driver, name, value)` writes
        // `jitdriver.warmstate`. `set_param(None, ...)` is
        // `MetaInterp::set_param_all_drivers`.
        self.meta
            .warm_state_for_driver(self.index().unwrap_or(0))
            .set_param(name, value);
    }

    /// Return a snapshot of the cumulative JIT compilation statistics.
    pub fn get_stats(&self) -> JitStats {
        self.meta.get_stats()
    }

    /// Return `(total allocator capacity, live code bytes)` from this
    /// driver's CPU-owned assembler memory manager.
    pub fn assembler_memory_stats(&self) -> (usize, usize) {
        self.meta.assembler_memory_stats()
    }

    /// Read the `i`th reason-keyed abort tally, indexed as
    /// [`crate::jitprof::ABORT_COUNTER_KINDS`]. Breaks a `loops_aborted`
    /// total down by `Counters.ABORT_*` for a caller that cannot reach the
    /// profiler directly.
    pub fn abort_diag(&self, i: usize) -> u64 {
        self.meta.staticdata.profiler.abort_diag(i)
    }

    /// Get direct access to the underlying MetaInterp.
    pub fn meta_interp(&self) -> &MetaInterp<S::Meta> {
        &self.meta
    }

    /// Get mutable access to the underlying MetaInterp.
    pub fn meta_interp_mut(&mut self) -> &mut MetaInterp<S::Meta> {
        &mut self.meta
    }

    /// `compile.py ResumeGuardForcedDescr.force_now` materializes
    /// virtuals through the same resume allocator used by ordinary guard
    /// failure.  In particular, a `jit.virtual_ref` frame must not be decoded
    /// through `NullAllocator`, or its `forced` writeback remains null.
    pub fn force_virtualizable_token(&mut self, token: u64) {
        let fallback_alloc = crate::resume::NullAllocator;
        let allocator: &dyn crate::resume::BlackholeAllocator = self
            .blackhole_allocator
            .as_deref()
            .unwrap_or(&fallback_alloc);
        self.meta
            .force_virtualizable_token_with_allocator(token, allocator);
    }

    /// framework.py `root_walker.walk_roots` parity: visit every Ref-typed
    /// entry in every live compiled trace's `rd_consts` pool so the minor
    /// collector can forward them. See `MetaInterp::walk_rd_consts_refs`
    /// for the full rationale.
    pub fn walk_rd_consts_refs(&mut self, visitor: impl FnMut(&mut majit_ir::GcRef)) {
        self.meta.walk_rd_consts_refs(visitor);
    }

    /// framework.py `root_walker.walk_roots` parity: visit every
    /// inline `OpRef::ConstPtr(GcRef)` slot in the stashed
    /// `partial_trace.ops` op-graph (history.py `ConstPtr.value`).
    /// See `MetaInterp::walk_partial_trace_refs`.
    pub fn walk_partial_trace_refs(&mut self, visitor: impl FnMut(&mut majit_ir::GcRef)) {
        self.meta.walk_partial_trace_refs(visitor);
    }

    /// framework.py `root_walker.walk_roots` parity: visit every
    /// inline `ConstPtr.value` slot in the active recorder's op-graph
    /// (history.py:314). See `MetaInterp::walk_active_trace_refs`.
    pub fn walk_active_trace_refs(&mut self, visitor: impl FnMut(&mut majit_ir::GcRef)) {
        self.meta.walk_active_trace_refs(visitor);
    }

    /// GC walker for ConstPtr GcRefs in snapshot maps that are
    /// live during compilation. See `MetaInterp::walk_compile_snapshot_refs`.
    pub fn walk_compile_snapshot_refs(&mut self, visitor: impl FnMut(&mut majit_ir::GcRef)) {
        self.meta.walk_compile_snapshot_refs(visitor);
    }

    pub fn run_compiled_detailed_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> DetailedDriverRunOutcome {
        let dispatch_key = self.take_single_pass_label_entry_dispatch_key_for_back_edge(green_key);
        self.run_compiled_detailed_keyed_with_dispatch_key(green_key, state, pre_run, dispatch_key)
    }

    fn run_compiled_detailed_keyed_with_dispatch_key(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
        dispatch_key: Option<u32>,
    ) -> DetailedDriverRunOutcome {
        let Some(meta) = self
            .meta
            .get_compiled_meta_on_driver(self.index().unwrap_or(0), green_key)
            .cloned()
        else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        // Resolved here with the descriptor and carried to both ends of
        // the run; see `sync_before` for why it is not asked for twice.
        let vable = descriptor
            .as_deref()
            .and_then(JitDriverStaticData::virtualizable);
        let saved_vable_ptr = self
            .meta
            .tracing
            .as_ref()
            .map(|ctx| ctx.virtualizable_heap_ptr().unwrap_or(std::ptr::null()));
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, vable) {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if crate::debug::have_debug_prints() {
            let vals: Vec<String> = live_values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("[{}]={:?}", i, v))
                .collect();
            crate::debug::log_one(
                "jit-running",
                &format!("run_bridge live_values: {}", vals.join(", ")),
            );
        }
        if !Self::live_values_match_descriptor(
            descriptor.as_deref(),
            &live_values,
            state.state_field_layout().total_live_values(),
        ) {
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let Some(live_values) =
            self.extend_compiled_live_values(green_key, state, &meta, vable, live_values)
        else {
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let mut live_values = live_values;
        // `live_values` is in state-field order; a dispatch-key entry lands on a
        // LABEL whose arguments are its own, so the mapping runs unconditionally.
        if dispatch_key.is_some() {
            let Some(packed) = self.meta.pack_front_target_live_values_on_driver(
                self.index().unwrap_or(0),
                green_key,
                &live_values,
            ) else {
                self.restore_trace_vable_ptr(saved_vable_ptr);
                return DetailedDriverRunOutcome::Abort {
                    restored: false,
                    via_blackhole: false,
                };
            };
            live_values = packed;
        }
        pre_run();
        let jd_no = self.index().unwrap_or(0);
        let result = if let Some(dispatch_key) = dispatch_key {
            self.meta.run_compiled_detailed_with_values_on_driver(
                jd_no,
                green_key,
                &live_values,
                dispatch_key,
            )
        } else {
            self.meta
                .run_compiled_detailed_with_values_on_driver(jd_no, green_key, &live_values, 0)
        };
        let Some(mut result) = result else {
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        if result.is_finish {
            // Taken, not copied: this arm returns below, so the exit values in
            // `result` have no reader past this point.
            let typed_values = std::mem::take(&mut result.typed_values).into_vec();
            let is_exit_frame_with_exception = result.is_exit_frame_with_exception;
            drop(result);
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Finished {
                typed_values,
                via_blackhole: false,
                raw_int_result: self.meta.has_raw_int_finish(),
                is_exit_frame_with_exception,
            };
        }

        let exit_meta = result.meta.take().expect("a detailed run carries its meta");
        // JUMP already decoded on its own arm. A guard-failure through this
        // runner still `restore_values`; re-derive the list from the deadframe.
        let typed_values = if result.fail_index == u32::MAX {
            std::mem::take(&mut result.typed_values)
        } else if let (Some(frame), Some(fd)) = (
            result.deadframe.as_ref(),
            result.descr_arc.as_ref().and_then(|a| a.as_fail_descr()),
        ) {
            crate::pyjitpl::MetaInterp::<S::Meta>::decode_exit_slots(
                self.meta.backend(),
                frame,
                fd.fail_arg_types(),
            )
        } else {
            std::mem::take(&mut result.typed_values)
        };
        state.restore_values(&exit_meta, &typed_values);
        self.sync_after(state, &exit_meta, vable, saved_vable_ptr);
        DetailedDriverRunOutcome::Jump {
            via_blackhole: false,
            continue_running_normally_values: None,
            continue_running_normally_pc: None,
        }
    }

    /// RPython compile.py:714 + pyjitpl.py rebuild_state_after_failure():
    /// run compiled code, distinguish normal JUMP from guard failure, and
    /// allow the caller to rebuild interpreter state and start bridge tracing
    /// from the recovered resume pc.
    /// `warmstate.py` `execute_assembler`: run compiled code and return
    /// the outcome. For guard failures, returns raw exit data without any
    /// state restoration — the caller processes it via handle_fail().
    pub fn run_compiled_detailed_with_bridge_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        _env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> DetailedDriverRunOutcome {
        let Some(meta) = self
            .meta
            .get_compiled_meta_on_driver(self.index().unwrap_or(0), green_key)
            .cloned()
        else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=missing-compiled-meta target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        // Resolved here with the descriptor and carried to both ends of
        // the run; see `sync_before` for why it is not asked for twice.
        let vable = descriptor
            .as_deref()
            .and_then(JitDriverStaticData::virtualizable);
        // warmstate.py:482-511: RPython enters execute_token directly
        // when the green key matches — no is_compatible check.
        // pyre still needs this gate because compiled code reads
        // virtualizable slots by positional index (not by field
        // descriptor), so a valuestackdepth mismatch causes wrong
        // slot reads. Remove when virtualizable field access uses
        // descriptor-based addressing (RPython virtualizable.py parity).
        if !state.is_compatible(&meta) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-skip] key={} reason=incompatible-state target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let saved_vable_ptr = self
            .meta
            .tracing
            .as_ref()
            .map(|ctx| ctx.virtualizable_heap_ptr().unwrap_or(std::ptr::null()));
        if !self.sync_before(state, &meta, vable) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=sync-before target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if crate::debug::have_debug_prints() {
            let vals: Vec<String> = live_values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("[{}]={:?}", i, v))
                .collect();
            crate::debug::log_one(
                "jit-running",
                &format!("BRIDGE live_values: {}", vals.join(", ")),
            );
        }
        if !Self::live_values_match_descriptor(
            descriptor.as_deref(),
            &live_values,
            state.state_field_layout().total_live_values(),
        ) {
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-abort",
                    &format!(
                        "run-compiled-abort key={green_key} reason=live-values-descriptor-mismatch target_pc={target_pc} nvals={}",
                        live_values.len()
                    ),
                );
            }
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let Some(live_values) =
            self.extend_compiled_live_values(green_key, state, &meta, vable, live_values)
        else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=extend-live-values target_pc={}",
                    green_key, target_pc
                );
            }
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        pre_run();
        let Some(mut result) = self.meta.run_compiled_detailed_with_values_on_driver(
            self.index().unwrap_or(0),
            green_key,
            &live_values,
            0,
        ) else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=backend-run target_pc={}",
                    green_key, target_pc
                );
            }
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        let is_finish = result.is_finish;
        let is_exit_frame_with_exception = result.is_exit_frame_with_exception;
        let exit_meta = result.meta.take().expect("a detailed run carries its meta");
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        // Carried past the finish arm below, which is the one that has none.
        let descr_arc = result.descr_arc.take();
        // The pointer travels; the layout behind it is opened past the finish
        // arm below, which is the one that carries none.
        let exit_layout = result.exit_layout.take();
        let guard_value_operand = result.guard_value_operand;
        // See the sibling run loop.
        let descr_owning_key = result.rd_loop_token;
        // llmodel.py grab_exc_value: the pending exception captured at
        // guard failure travels with the GuardFailure outcome so the
        // blackhole resume can seed it (blackhole.py:1794).
        let guard_exc = result.exception.exc_value;
        // compile.py `cpu.get_savedata_ref(deadframe)` — fished here
        // while the CompileResult still owns the word, then handed to
        // `ResumeGuardForcedDescr.handle_fail`. Keep the deadframe
        // until `AllVirtuals.show` so `jf_savedata` stays rooted.
        let savedata = result.savedata;
        let deadframe = result.deadframe.take();

        // memmgr.py: keep_loop_alive(loop_token) on the executing
        // driver's warmstate, matching execute_assembler.
        self.meta
            .keep_loop_alive_on_driver(self.index().unwrap_or(0), green_key);

        if is_finish {
            let typed_values = std::mem::take(&mut result.typed_values).into_vec();
            drop(result);
            self.restore_trace_vable_ptr(saved_vable_ptr);
            return DetailedDriverRunOutcome::Finished {
                typed_values,
                via_blackhole: false,
                raw_int_result: self.meta.has_raw_int_finish(),
                is_exit_frame_with_exception,
            };
        }

        let descr_arc = descr_arc.expect("a guard exit carries its descr");
        // Normal loop back-edge JUMP, not a guard failure.
        if fail_index == u32::MAX {
            state.restore_values(&exit_meta, &result.typed_values);
            self.sync_after(state, &exit_meta, vable, saved_vable_ptr);
            drop(result);
            return DetailedDriverRunOutcome::Jump {
                via_blackhole: false,
                continue_running_normally_values: None,
                continue_running_normally_pc: None,
            };
        }

        // compile.py handle_fail / must_compile: single tick+check.
        // Guard-failure: decode from the deadframe after the split, for
        // callers that still take `GuardFailure.raw_values` (eval.rs).
        let fd: &dyn majit_ir::FailDescr = descr_arc
            .as_fail_descr()
            .expect("a guard exit carries a FailDescr");
        let raw_values = deadframe
            .as_ref()
            .map(|frame| self.meta.raw_exit_slots_from_deadframe(frame, fd))
            .unwrap_or_default();
        drop(result);
        let exit_layout = *exit_layout.expect("a guard exit carries its exit layout");
        let fallback_green_key = if exit_layout.rd_loop_token != 0 {
            exit_layout.rd_loop_token
        } else {
            green_key
        };
        // Carried off the result — see the sibling run loop.
        let owning_key = descr_owning_key.unwrap_or(fallback_green_key);
        let (must_compile, owning_key) = self.meta.must_compile_with_owning_key(
            &descr_arc,
            &raw_values,
            guard_value_operand,
            owning_key,
        );
        // compile.py: must_compile() and not stack_almost_full().
        // `no_bridge_enabled()` is the same opt-out the two sibling loops
        // apply; this is the loop pyre actually runs, so leaving it out made
        // `MAJIT_NO_BRIDGE=1` report a bridge-free run while still compiling
        // every bridge.
        // `is_tracing()` sits with the other conjuncts rather than at the
        // consumer: a bridge cannot be recorded from inside a live recording,
        // and this is the entry that reaches a guard failure with one live.
        // Taking fuel before asking numbered bridges that never compiled,
        // which is the property `MAJIT_MAX_BRIDGES` and the skip list are read
        // for, and printed `@@@GUARD bridge=true` on the same failures.
        let should_bridge = must_compile
            && !majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full()
            && !no_bridge_enabled()
            && !self.is_tracing()
            && bridge_fuel_take();

        // Same `@@@GUARD` line the sibling loops emit. Without it this loop —
        // the one pyre reaches — had no per-guard-failure trace at all, so
        // identifying a failing guard meant reading the whole `MAJIT_LOG`
        // stream. `resume_pc` is resolved by the caller here, so report the
        // guard's own coordinates instead.
        if guardlog_enabled() {
            let preview: Vec<i64> = raw_values.iter().take(6).copied().collect();
            eprintln!(
                "@@@GUARD key={} trace={} fail={} bridge={} nvals={} vals={:?}",
                owning_key,
                trace_id,
                fail_index,
                should_bridge,
                raw_values.len(),
                preview
            );
        }

        // Return raw guard failure data; state restoration is the caller's
        // handle_fail().  The vable pointer `sync_before` moved to this
        // entry's frame is deliberately left naming it: `should_bridge` is
        // false while a recording is live, so the one outcome handle_fail has
        // then is the blackhole, and that resumes on this entry's frame.
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            descr_arc,
            should_bridge,
            owning_key,
            raw_values,
            exit_layout,
            guard_exc,
            savedata,
            deadframe,
        }
    }

    /// Invalidate a compiled loop, forcing fallback to interpretation.
    ///
    /// `cpu.invalidate_loop` runs on the token filed on this driver's
    /// WarmEnterState (`JitDriver::index`). Compile-time
    /// `MetaInterp::invalidate_loop` still reads `get_procedure_token`.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        if let Some(token) = self
            .meta
            .get_procedure_token_on_driver(self.index().unwrap_or(0), green_key)
        {
            token.invalidate();
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-invalidate",
                    &format!("invalidated loop at key={green_key}"),
                );
            }
        }
    }

    /// Drop the metainterpreter metadata for a loop that has already been
    /// invalidated and must be replaced by a newly traced version.
    ///
    /// `invalidate_loop` alone preserves the old trace's target tokens for
    /// quasi-immutable invalidation.  An intentional specialization demotion
    /// instead needs a fresh root trace, so its stale targets must not make
    /// `compile_loop` choose the old loop's bridge path.
    pub fn remove_compiled_loop(&mut self, green_key: u64) {
        self.meta
            .remove_compiled_loop_on_driver(self.index().unwrap_or(0), green_key);
    }

    /// Check whether a compiled loop exists for a given green key.
    ///
    /// `warmstate.py maybe_compile_and_run` is bound on this driver's
    /// WarmEnterState (`JitDriver::index`), not leftover portal slot 0.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.meta
            .entry_procedure_token_on_driver(self.index().unwrap_or(0), green_key)
            .is_some()
    }

    /// [`Self::has_compiled_loop`] with both flag reads removed, so an
    /// amplified arm can price the cell lookup and the `Weak::upgrade`/drop
    /// pair without them. A cost probe with a deliberately weaker answer than
    /// the door's -- see `WarmState::probe_cell_token_upgrades`. Nothing may
    /// route on it.
    #[cfg(feature = "__yield-stage-probe")]
    #[inline]
    pub fn probe_cell_token_upgrades(&self, green_key: u64) -> bool {
        self.meta.probe_cell_token_upgrades(green_key)
    }

    /// Whether the warm-entry runner can actually execute the code at this
    /// green key. Stronger than `has_compiled_loop`: it also requires a
    /// frontend `compiled_loops` meta (`get_compiled_meta`).
    ///
    /// `has_compiled_loop` is true whenever the cell's procedure token has a
    /// compiled backend body (`warmstate.py` gates entry on code
    /// presence). That includes a `compile_tmp_callback` token
    /// (`compile.py:1101-1150`): a CALL_ASSEMBLER fallback stub with a real
    /// body but NO frontend meta, since MetaInterp never inserts it into
    /// `compiled_loops`. RPython's `execute_assembler` can enter such a token
    /// directly — the stub bounces control back to the interpreter
    /// (ContinueRunningNormally). Pyre's warm-entry runner instead needs the
    /// `compiled_loops` meta to interpret guard-failure exit layouts, so
    /// entering a bare tmp callback here has no meta and aborts.
    ///
    /// Gating warm-entry dispatch on this predicate lets a tmp-only cell fall
    /// through to the counter tick / trace-start, which is the pyre-orthodox
    /// realization of the tmp-callback → resume-interp semantics: the back-edge
    /// counter keeps ticking until the real loop compiles (a `compiled_loops`
    /// entry appears) and `redirect_call_assembler` takes over. Entry bridges
    /// (`ResumeFromInterpDescr`, `compile.py:1079-1083`) DO get a
    /// `compiled_loops` entry despite carrying 0 target tokens, so they remain
    /// dispatchable — this predicate only excludes bare tmp callbacks.
    ///
    /// No typed twin, and unlike [`Self::has_compiled_loop_for_key`] that is
    /// not a gap: the strengthening above is pyre-only, so there is no
    /// warmstate.py reader for a twin to mirror. What a twin would add is a
    /// `cell_key_for` resolve in front, and the entry path already resolves at
    /// the producer (`back_edge_internal`, `resolve_cell_key`) — after which
    /// both conjuncts here index by that one cell key, so the disagreement a
    /// twin existed to prevent (`compiled_loops` read at the bare
    /// `key.get_uhash()` while the token half walked the chain) can no longer
    /// be expressed. `warmstate::tests::
    /// the_entry_decision_and_the_entry_execution_resolve_through_one_cell_key`
    /// is the pin for that.
    ///
    /// The boolean form of [`Self::runnable_procedure_token`] and defined in
    /// terms of it, so a caller that decides through this predicate and a
    /// caller that runs through the token cannot be answering about different
    /// cells or different tokens.
    #[inline]
    pub fn has_runnable_compiled_loop(&self, green_key: u64) -> bool {
        self.runnable_procedure_token(green_key).is_some()
    }

    /// Runnable-loop predicate for an execution door: the cell/token and
    /// frontend metadata must exist, then the warm-state entry policy must
    /// allow the transition.
    #[inline]
    fn has_confirmed_runnable_compiled_loop(&self, green_key: u64) -> bool {
        self.has_runnable_compiled_loop(green_key)
            && self
                .meta
                .warm_state_ref_for_driver(self.index().unwrap_or(0))
                .is_some_and(|warm| warm.confirm_compiled_entry_for_cell_key(green_key))
    }

    /// See [`MetaInterp::runnable_generation`]: a number that holds while
    /// [`Self::resolved_entry_procedure_token`] cannot change its answer
    /// for any key, and moves when it may.
    #[inline]
    pub fn runnable_generation(&self) -> u64 {
        self.meta.runnable_generation()
    }

    /// [`Self::resolve_cell_key`] and the whole CELL-OWNED half of
    /// [`Self::runnable_procedure_token`] from one walk of the cell chain.
    ///
    /// The pair is what a door holding a raw bucket hash asks: which cell do my
    /// greens own, and does that cell have code that may be entered. Asked
    /// separately they walked the chain three times for the same cell — once to
    /// resolve, once to read the token, once for `JC_TEMPORARY` — see
    /// [`MetaInterp::resolved_entry_procedure_token`]. The answer is identical
    /// to resolving first and then asking; this only stops paying for the
    /// repeats.
    ///
    /// `JC_TEMPORARY` is part of the answer because retained frontend metadata
    /// must not make a `compile_tmp_callback` token executable as the loop it
    /// displaced, matching the flag-first branch of
    /// `WarmEnterState.maybe_compile_and_run`.
    ///
    /// The `compiled_loops` conjunct is NOT part of it, and the resolved key
    /// travels back so the caller can take it: a door that goes on to run needs
    /// that map's VALUE, so a presence test here would probe it twice for one
    /// entry, and a door that declines is answered by the walk alone.
    #[inline]
    pub fn resolved_entry_procedure_token(
        &self,
        green_key_hash: u64,
        make_green_key: Option<&dyn Fn() -> GreenKey>,
    ) -> (u64, Option<std::sync::Arc<majit_backend::JitCellToken>>) {
        self.meta.resolved_entry_procedure_token_on_driver(
            self.index().unwrap_or(0),
            green_key_hash,
            make_green_key,
        )
    }

    /// The token [`Self::has_runnable_compiled_loop`] says yes about, handed
    /// back so a door can carry it into the run instead of resolving the cell a
    /// second time.
    ///
    /// `warmstate.py` reads `procedure_token = cell.get_procedure_token()`
    /// once and `:509-511` carries that object out through `raise
    /// EnterJitAssembler(procedure_token, *execute_args)`; the executor never
    /// asks the cell again. This is the reader that makes the same carry
    /// expressible for a caller outside this crate.
    ///
    /// How the predicate's two conjuncts divide here: `entry_procedure_token_on_driver`
    /// IS the `has_compiled_loop` half — that predicate is defined as this
    /// token being present, not merely as something equivalent to it — so
    /// binding the token subsumes the first conjunct exactly. It does NOT
    /// subsume the second: `get_compiled_meta` reads the frontend
    /// `compiled_loops` table. Nor does either read subsume the cell-owned
    /// `JC_TEMPORARY` flag: `compile_tmp_callback` can displace a real loop
    /// while that loop's frontend metadata remains. Both tests therefore stay
    /// here explicitly, matching the flag-first branch in upstream
    /// `WarmEnterState.maybe_compile_and_run`.
    ///
    /// `warmstate.py maybe_compile_and_run` is bound on this driver's
    /// WarmEnterState (`JitDriver::index`), not leftover portal slot 0.
    ///
    /// `green_key` must already name one cell — see [`Self::resolve_cell_key`],
    /// whose result is what a door should ask this with. A raw bucket hash
    /// answers for whichever cell heads the bucket.
    #[inline]
    pub fn runnable_procedure_token(
        &self,
        green_key: u64,
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        let jd_no = self.index().unwrap_or(0);
        self.meta
            .entry_procedure_token_on_driver(jd_no, green_key)
            .filter(|_| {
                !self
                    .meta
                    .warm_state_ref_for_driver(jd_no)
                    .is_some_and(|warm| warm.procedure_token_is_temporary(green_key))
                    && self
                        .meta
                        .get_compiled_meta_on_driver(jd_no, green_key)
                        .is_some()
            })
    }

    /// The back-edge door's whole answer from the warm-state policy that owns
    /// both trace-start and compiled-entry decisions.
    ///
    /// `warmstate.py maybe_compile_and_run` is bound on this driver's
    /// WarmEnterState (`JitDriver::index`).
    #[inline]
    pub fn maybe_compile_and_run_step(
        &mut self,
        green_key: u64,
        green_key_raw: (usize, usize),
    ) -> (crate::warmstate::HotResult, u64) {
        self.meta.maybe_compile_and_run_step_on_driver(
            self.index().unwrap_or(0),
            green_key,
            green_key_raw,
        )
    }

    /// `compile.py ResumeFromInterpDescr.compile_and_attach` files the
    /// frontend row on this driver's cell (`JitDriver::index()`), matching
    /// [`Self::maybe_compile_and_run_step`]. Leftover `active_jitdriver_sd`
    /// is the last traced driver.
    #[cfg(test)]
    pub(crate) fn insert_compiled_loop(
        &mut self,
        green_key: u64,
        entry: crate::pyjitpl::CompiledEntry<S::Meta>,
    ) -> Option<crate::pyjitpl::CompiledEntry<S::Meta>> {
        self.meta
            .insert_compiled_loop_on_driver(self.index().unwrap_or(0), green_key, entry)
    }

    /// The function-entry door's whole answer, from one walk of the cell chain.
    ///
    /// `MetaInterp::function_entry_step` joins the two pieces of state that live
    /// outside the warm state -- `compiled_loops`, the frontend half of
    /// [`Self::has_runnable_compiled_loop`], and `is_tracing` -- with the cell
    /// read [`crate::warmstate::WarmEnterState::function_entry_step`] performs.
    /// Neither is a chain walk, so that read is the only walk the door pays for.
    ///
    /// `cell_key` must already name one cell; `green_key_hash` and
    /// `green_key_raw` reconstruct its structured greens only when the
    /// optional veto is installed.
    ///
    /// `warmstate.py maybe_compile_and_run` with
    /// `increment_function_threshold` is bound on this driver's
    /// WarmEnterState (`JitDriver::index`).
    pub fn function_entry_step(
        &mut self,
        cell_key: u64,
        green_key_hash: u64,
        green_key_raw: (usize, usize),
    ) -> FunctionEntryStep {
        self.meta.function_entry_step_on_driver(
            self.index().unwrap_or(0),
            cell_key,
            green_key_hash,
            green_key_raw,
        )
    }

    /// `warmspot.py` `ll_portal_runner`: maybe enter from the function's start.
    ///
    /// `maybe_compile_and_run(increment_function_threshold, *args)` then the
    /// interpreter. A runnable procedure token is entered through
    /// [`Self::back_edge_resolved`] so FINISH, deopt and blackhole resume are
    /// the same as a back-edge run. A counter hit arms tracing at `target_pc`
    /// via [`Self::force_start_tracing`] and returns `None` so the caller
    /// falls into the interpreter; the next merge point records.
    ///
    /// Returns `Some(resume_pc)` only when compiled code ran and left a
    /// resume point. FINISH is published on the same latch as
    /// [`Self::back_edge`].
    ///
    /// Not `#[cold]`: this is the interpreter's per-call door, and the
    /// empty-chain arm (`warmstate.py` with
    /// `increment_function_threshold`) is the common path. Compiled-entry
    /// / tracing-start / chained-resolve bodies stay in
    /// `function_entry_internal`, which is `#[inline(never)]` so they do
    /// not land in the portal.
    #[inline]
    pub fn function_entry_structured(
        &mut self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> Option<usize> {
        match self.function_entry_runner(green_key_hash, make_green_key, target_pc, state, env) {
            FunctionEntryRunner::Finished(value) => {
                self.meta.back_edge_finish_word = Some(value);
                Some(target_pc)
            }
            FunctionEntryRunner::Resume(pc) => Some(pc),
            FunctionEntryRunner::Run => None,
        }
    }

    /// `warmspot.py ll_portal_runner`: the door in front of the portal body.
    ///
    /// Same decisions as [`Self::function_entry_structured`]. The raw compiled
    /// fast path (`warmstate.py execute_assembler`,
    /// `DoneWithThisFrameDescrInt` / `DoneWithThisFrameDescrRef`) returns
    /// [`FunctionEntryRunner::Finished`] and does not write
    /// `back_edge_finish_word`. Tracing, blackhole, the general case and the
    /// typed entry still publish that latch; the runner drains it the way the
    /// inline door does.
    #[inline]
    pub fn function_entry_runner(
        &mut self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> FunctionEntryRunner {
        if std::mem::replace(&mut self.function_entry_suppressed, false) {
            return FunctionEntryRunner::Run;
        }
        // `warmspot.py ll_portal_runner` enters once per portal call. A
        // prior call's Finish may leave `back_edge_finish_word` /
        // `single_pass_finish` standing when the runner drained one latch
        // and not the other. The next door's `take_back_edge_finish_*`
        // would then return that call's resultbox.
        self.clear_finish_latches();
        if self.meta.is_tracing() {
            return FunctionEntryRunner::Run;
        }
        // The raw fast path writes the word here instead of
        // `back_edge_finish_word`, so `execute_assembler`'s register return
        // shape stays intact (`Done(Option<usize>)` still fits one word).
        let mut direct_word = None;
        let entry = self.enter_compiled_function_entry(
            green_key_hash,
            target_pc,
            state,
            env,
            &mut direct_word,
        );
        if let Some(value) = direct_word {
            return FunctionEntryRunner::Finished(value);
        }
        match entry {
            SteadyCompiledEntry::Done(Some(pc)) => return FunctionEntryRunner::Resume(pc),
            SteadyCompiledEntry::Done(None) => return FunctionEntryRunner::Run,
            SteadyCompiledEntry::NeedsInternal => {
                return map_runner_resume(self.function_entry_internal(
                    green_key_hash,
                    make_green_key,
                    target_pc,
                    state,
                    env,
                    false,
                ));
            }
            SteadyCompiledEntry::Miss => {}
        }
        if let Some(handled) = self.try_function_entry_cold_tick(
            green_key_hash,
            &make_green_key,
            target_pc,
            state,
            env,
        ) {
            return map_runner_resume(handled);
        }
        map_runner_resume(self.function_entry_internal(
            green_key_hash,
            make_green_key,
            target_pc,
            state,
            env,
            false,
        ))
    }

    /// Sole compiled cell for `hash`, when no confirm hook is installed.
    ///
    /// `warmstate.py maybe_compile_and_run` walks `lookup_chain` once.
    /// `:473` declines `JC_TRACING | JC_TEMPORARY`. `:483` reads
    /// `get_procedure_token` once. A confirm hook stays on the occupied
    /// door (`:501`).
    #[inline]
    fn compiled_function_token(
        &self,
        hash: u64,
    ) -> Option<(u64, std::sync::Arc<majit_backend::JitCellToken>)> {
        let jd_no = self.index().unwrap_or(0);
        let Some(warm) = self.meta.warm_state_ref_for_driver(jd_no) else {
            return None;
        };
        if warm.has_confirm_enter_jit() {
            return None;
        }
        let cell = warm.lookup_chain(hash)?;
        if cell.next.borrow().is_some() || cell.cell_bucket.get() != hash {
            return None;
        }
        if cell.flags.get().contains(JcFlags::JC_TRACING)
            || cell.flags.get().contains(JcFlags::JC_TEMPORARY)
        {
            return None;
        }
        let cell_key = cell.cell_key.get()?;
        let token = cell.get_procedure_token()?;
        if !token.has_compiled_code() {
            return None;
        }
        Some((cell_key, token))
    }

    /// `warmstate.py maybe_compile_and_run` then `execute_assembler` for a
    /// compiled function-entry token.
    ///
    /// Reds are unspecialized into the driver's scratch (`:503-511`) and the
    /// assembler runs. `execute_assembler` passes those reds straight to
    /// `func_execute_token`. When `vinfo is not None` the only per-entry
    /// touch is `vinfo.clear_vable_token(args[index_of_virtualizable])`;
    /// `index_of_virtualizable` is the value `make_virtualizable_infos`
    /// stored on the driver. A pending label or a cross-loop cut falls
    /// back to [`Self::back_edge_resolved`].
    #[inline(never)]
    fn enter_compiled_function_entry(
        &mut self,
        green_key_hash: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        direct_word: &mut Option<i64>,
    ) -> SteadyCompiledEntry {
        let Some((cell_key, token)) = self.compiled_function_token(green_key_hash) else {
            return SteadyCompiledEntry::Miss;
        };
        if !state.can_trace() {
            return self.enter_compiled_cannot_trace();
        }
        if !self.meta.cut_compiled_keys.is_empty() {
            return self
                .enter_compiled_pending_label_or_cut(cell_key, token, target_pc, state, env);
        }
        let mut scratch = self.take_entry_scratch_raw();
        // `warmstate.py maybe_compile_and_run` unspecializes reds to words and
        // `llmodel.py execute_token` stores them by the token's kinds. A state
        // that cannot produce the words takes the typed entry below.
        let raw_entry = state.fill_entry_raw_reds(&mut scratch.raw);
        if !raw_entry {
            return self
                .enter_compiled_typed_entry(cell_key, token, target_pc, state, env, scratch);
        }
        // `execute_assembler` receives the unspecialized reds.
        // `patch_new_loop_to_load_virtualizable_fields` truncates the
        // loop's inputargs to that prefix, so a patched entry's
        // `inputarg_types` match the reds already filled. A longer list was
        // not patched and still needs the extension in `back_edge_resolved`,
        // which reads typed reds.
        let have = scratch.raw.len();
        if token.inputarg_types().len() > have {
            return self
                .enter_compiled_unpatched_reds(cell_key, token, target_pc, state, env, scratch);
        }
        // `warmstate.py execute_assembler`: `if vinfo is not None:
        // virtualizable = args[index_of_virtualizable];
        // vinfo.clear_vable_token(virtualizable)`. Nothing else — no
        // descriptor walk, no field export.
        self.clear_entry_vable_token_raw(&scratch.raw);
        if self.meta.hooks.on_compiled_entry.is_some() {
            self.enter_compiled_entry_hook(cell_key, target_pc);
        }
        // `warmstate.py execute_assembler` selects the poll from
        // `jitdriver_sd.result_type`. Int portals keep `poll_raw_int_finish_raw`.
        let finished = if self.meta.result_type == Type::Ref {
            self.meta
                .poll_raw_ref_finish_raw(&token, cell_key, &scratch.raw)
                .map(|addr| addr as i64)
        } else {
            self.meta
                .poll_raw_int_finish_raw(&token, cell_key, &scratch.raw)
        };
        if let Some(value) = finished {
            self.entry_scratch_out(scratch);
            self.meta.back_edge_finish = None;
            // `ll_portal_runner` returns `execute_assembler`'s fast-path word
            // without the latch.
            *direct_word = Some(value);
            return SteadyCompiledEntry::Done(None);
        }
        self.enter_compiled_general_case(cell_key, target_pc, state, env, scratch)
    }

    /// `can_trace` declined the compiled entry. The interpreter continues.
    #[cold]
    #[inline(never)]
    fn enter_compiled_cannot_trace(&mut self) -> SteadyCompiledEntry {
        SteadyCompiledEntry::Done(None)
    }

    /// `hooks.on_compiled_entry`, when one is installed.
    #[cold]
    #[inline(never)]
    fn enter_compiled_entry_hook(&self, cell_key: u64, target_pc: usize) {
        if let Some(ref hook) = self.meta.hooks.on_compiled_entry {
            hook(cell_key, target_pc);
        }
    }

    /// Pending label or cross-loop cut: `maybe_compile_and_run` cannot enter
    /// the raw token, so the call goes through `back_edge_resolved`.
    #[cold]
    #[inline(never)]
    fn enter_compiled_pending_label_or_cut(
        &mut self,
        cell_key: u64,
        token: std::sync::Arc<majit_backend::JitCellToken>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> SteadyCompiledEntry {
        SteadyCompiledEntry::Done(
            self.back_edge_resolved(cell_key, token, target_pc, state, env, || {})
                .and_then(|resume| self.resume_pc_recycling(resume)),
        )
    }

    /// Typed reds: `fill_entry_raw_reds` declined, so `execute_assembler`'s
    /// word path does not apply. Meta is read only when the reds or
    /// `is_compatible` need it; then the same `clear_vable_token`, hook,
    /// `poll_raw_int_finish` and finish latches as the raw path.
    #[cold]
    #[inline(never)]
    fn enter_compiled_typed_entry(
        &mut self,
        cell_key: u64,
        token: std::sync::Arc<majit_backend::JitCellToken>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        mut scratch: Box<EntryScratch>,
    ) -> SteadyCompiledEntry {
        scratch.clear_typed();
        // `execute_assembler` does not look up `CompiledEntry`. The cell
        // already holds the procedure token. Meta is only for a state whose
        // reds or `is_compatible` actually read it.
        let compatible = if state.fill_entry_reds_without_meta(&mut scratch.live_values) {
            true
        } else {
            let Some(meta) = self
                .meta
                .get_compiled_meta_on_driver(self.index().unwrap_or(0), cell_key)
            else {
                self.entry_scratch_out(scratch);
                return SteadyCompiledEntry::NeedsInternal;
            };
            let compatible = state.is_compatible(meta);
            if compatible {
                state.extract_live_values_into(
                    meta,
                    &mut scratch.live_values,
                    &mut scratch.raw,
                    &mut scratch.types,
                );
            }
            compatible
        };
        if !compatible {
            self.entry_scratch_out(scratch);
            self.invalidate_loop(cell_key);
            return SteadyCompiledEntry::Done(None);
        }
        // Same inputarg-length test as the raw path. This arm did not fill
        // raw words, so `have` is the typed red count. A longer
        // `inputarg_types` list was not patched. `back_edge_resolved` fills
        // the reds again through `extract_live_values_into`.
        let have = scratch.live_values.len();
        if token.inputarg_types().len() > have {
            self.entry_scratch_out(scratch);
            return SteadyCompiledEntry::Done(
                self.back_edge_resolved(cell_key, token, target_pc, state, env, || {})
                    .and_then(|resume| self.resume_pc_recycling(resume)),
            );
        }
        self.clear_entry_vable_token(&scratch.live_values);
        if let Some(ref hook) = self.meta.hooks.on_compiled_entry {
            hook(cell_key, target_pc);
        }
        let finished = if self.meta.result_type == Type::Ref {
            self.meta
                .poll_raw_ref_finish(&token, cell_key, &scratch.live_values)
                .map(|addr| addr as i64)
        } else {
            self.meta
                .poll_raw_int_finish(&token, cell_key, &scratch.live_values)
        };
        if let Some(value) = finished {
            self.entry_scratch_out(scratch);
            self.meta.back_edge_finish = None;
            self.meta.back_edge_finish_word = Some(value);
            return SteadyCompiledEntry::Done(Some(target_pc));
        }
        self.enter_compiled_general_case(cell_key, target_pc, state, env, scratch)
    }

    /// `inputarg_types` is longer than the unspecialized reds.
    ///
    /// The loop was not patched down to that prefix, so
    /// `back_edge_resolved` extends the typed reds. The discarded fill is
    /// the same call the raw arm made before handing the token on.
    #[cold]
    #[inline(never)]
    fn enter_compiled_unpatched_reds(
        &mut self,
        cell_key: u64,
        token: std::sync::Arc<majit_backend::JitCellToken>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        mut scratch: Box<EntryScratch>,
    ) -> SteadyCompiledEntry {
        scratch.clear_typed();
        let _ = state.fill_entry_reds_without_meta(&mut scratch.live_values);
        self.entry_scratch_out(scratch);
        SteadyCompiledEntry::Done(
            self.back_edge_resolved(cell_key, token, target_pc, state, env, || {})
                .and_then(|resume| self.resume_pc_recycling(resume)),
        )
    }

    /// `poll_raw_int_finish` / `poll_raw_int_finish_raw` returned `None`.
    ///
    /// The trace already ran; `raw_int_fallback` holds the general-case
    /// result. `execute_assembler` then takes `handle_fail` via
    /// `consume_compiled_entry_result`.
    #[cold]
    #[inline(never)]
    fn enter_compiled_general_case(
        &mut self,
        cell_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        scratch: Box<EntryScratch>,
    ) -> SteadyCompiledEntry {
        let result = self
            .meta
            .raw_int_fallback
            .take()
            .expect("poll_raw_int_finish stored the general-case result");
        self.entry_scratch_out(scratch);
        let compiled_meta = self
            .meta
            .get_compiled_meta_on_driver(self.index().unwrap_or(0), cell_key)
            .cloned()
            .expect("compiled meta survived the run");
        SteadyCompiledEntry::Done(
            self.consume_compiled_entry_result(
                cell_key,
                target_pc,
                state,
                env,
                result,
                &compiled_meta,
                None,
                0,
                portal_rca_enabled(),
            )
            .and_then(|resume| self.resume_pc_recycling(resume)),
        )
    }

    /// `execute_assembler`'s `vinfo.clear_vable_token(args[index_of_virtualizable])`.
    ///
    /// The index is [`JitDriverStaticData::index_of_virtualizable`], fixed
    /// when the driver is set up (`warmspot.py make_virtualizable_infos`).
    /// The red at that slot is the virtualizable pointer. No `Arc` clone
    /// and no allocation: the info is the one already on the metainterp.
    fn clear_entry_vable_token(&self, live_values: &[Value]) {
        let Some(descriptor) = self.descriptor.as_ref() else {
            return;
        };
        let index = descriptor.index_of_virtualizable;
        if index < 0 {
            return;
        }
        let Some(info) = self.meta.virtualizable_info() else {
            return;
        };
        let Some(Value::Ref(majit_ir::GcRef(addr))) = live_values.get(index as usize) else {
            return;
        };
        if *addr == 0 {
            return;
        }
        unsafe {
            crate::virtualizable::bh_clear_vable_token(info, *addr as *mut u8);
        }
    }

    /// [`Self::clear_entry_vable_token`] for an `unspecialize_value` word.
    ///
    /// The word at `index_of_virtualizable` is the virtualizable address
    /// (`warmstate.py execute_assembler` `args[index_of_virtualizable]`).
    ///
    /// A zero `vable_token` is the steady entry (`virtualizable.py
    /// clear_vable_token`). Forcing a live token is
    /// [`Self::force_entry_vable_token_raw`].
    #[inline]
    fn clear_entry_vable_token_raw(&self, raw: &[i64]) {
        let Some(descriptor) = self.descriptor.as_ref() else {
            return;
        };
        let index = descriptor.index_of_virtualizable;
        if index < 0 {
            return;
        }
        let Some(info) = self.meta.virtualizable_info() else {
            return;
        };
        let Some(&word) = raw.get(index as usize) else {
            return;
        };
        if word == 0 || !info.has_vable_token() {
            return;
        }
        let token = unsafe {
            let obj = word as usize as *mut u8;
            *(obj.add(info.token_offset) as *const usize)
        };
        if token == 0 {
            return;
        }
        self.force_entry_vable_token_raw(info, word as usize as *mut u8);
    }

    /// `virtualizable.py clear_vable_token` when the token is live:
    /// `force_now`, then the token is clear.
    #[cold]
    #[inline(never)]
    fn force_entry_vable_token_raw(
        &self,
        info: &crate::virtualizable::VirtualizableInfo,
        obj: *mut u8,
    ) {
        unsafe {
            crate::virtualizable::bh_clear_vable_token(info, obj);
        }
    }

    /// `warmstate.py maybe_compile_and_run` `:465-480` with
    /// `increment_function_threshold`: hash already in hand, `lookup_chain`
    /// once, and if there is no enterable procedure token, `jitcounter.tick`
    /// and return. A compiled token, a chained bucket, a dead token, or a
    /// `JC_DONT_TRACE_HERE` cell continues into `function_entry_internal`.
    ///
    /// `Some(resume)` means this IS that cold case — including `Some(None)`
    /// when the tick did not overflow. A True tick is already consumed here
    /// and is handed to `function_entry_internal` as `already_ticked` so
    /// [`crate::warmstate::WarmEnterState::function_entry_step`] does not
    /// increment a second time.
    #[inline(always)]
    fn try_function_entry_cold_tick(
        &mut self,
        green_key_hash: u64,
        make_green_key: &impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) -> Option<Option<usize>> {
        if let Some((cell_key, token)) = self.compiled_function_token(green_key_hash) {
            return Some(
                self.back_edge_resolved(cell_key, token, target_pc, state, env, || {})
                    .and_then(|resume| self.resume_pc_recycling(resume)),
            );
        }
        let jd_no = self.index().unwrap_or(0);
        if let Some(cell) = self
            .meta
            .warm_state_ref_for_driver(jd_no)
            .and_then(|warm| warm.lookup_chain(green_key_hash))
        {
            if cell.next.borrow().is_some() {
                return None;
            }
            if cell.cell_bucket.get() == green_key_hash {
                if cell.is_compiled() {
                    return None;
                }
                if cell.is_tracing() {
                    // Slot 23/65/66 live on `function_entry_step`; keep them
                    // on that walk so a JC_TRACING leak still tallies.
                    return None;
                }
                if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
                    return None;
                }
                if cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE) {
                    // Immediate-Proceed (never traced) vs tick-normally
                    // (TRACING_OCCURRED) stay on the occupied door.
                    return None;
                }
                if cell.flags.get().contains(JcFlags::JC_TEMPORARY) {
                    return None;
                }
            }
            // Occupant of this slot is a different green key. Same as empty
            // for our hash: tick our hash, do not inspect the stranger
            // (`warmstate.py:465-469` not-found arm).
        }
        if self
            .meta
            .warm_state_for_driver(jd_no)
            .tick_function_entry_empty_chain(green_key_hash)
        {
            return Some(self.function_entry_internal(
                green_key_hash,
                make_green_key,
                target_pc,
                state,
                env,
                true,
            ));
        }
        Some(None)
    }

    /// Occupied / overflow half of [`Self::function_entry_structured`].
    ///
    /// `already_ticked` is the empty-chain overflow: the counter already
    /// returned True, so this skips [`Self::function_entry_step`] (which
    /// would tick again) and takes the Proceed arm. RunCompiled, chained
    /// `resolve_cell_key`, decay + `stack_almost_full` +
    /// `force_start_tracing` all stay here.
    #[inline(never)]
    fn function_entry_internal(
        &mut self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        already_ticked: bool,
    ) -> Option<usize> {
        let cell_key = self.resolve_cell_key(green_key_hash, make_green_key);
        let step = if already_ticked {
            FunctionEntryStep::Proceed
        } else {
            self.function_entry_step(cell_key, green_key_hash, (state.code_ptr(), target_pc))
        };
        match step {
            FunctionEntryStep::RunCompiled(token) => self
                .back_edge_resolved(cell_key, token, target_pc, state, env, || {})
                .and_then(|resume| self.resume_pc_recycling(resume)),
            FunctionEntryStep::Proceed => {
                if !state.can_trace() {
                    return None;
                }
                // warmstate.py bound_reached: decay, then refuse a nearly
                // full stack, then start the function-entry trace.
                // `force_start_tracing` republishes the frame decoder.
                self.meta.decay_counters();
                if majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full() {
                    return None;
                }
                self.force_start_tracing(cell_key, target_pc, state, env);
                None
            }
            FunctionEntryStep::NotHot => None,
        }
    }

    /// Skip the next generated function-entry door.
    ///
    /// `warmspot.py` has no equivalent: a portal has one runner. An embedder
    /// that still has a second, loop-header door and cannot yet coexist two
    /// artifacts for one program uses this to yield that other door.
    pub fn suppress_function_entry(&mut self) {
        self.function_entry_suppressed = true;
    }

    /// Turn the raw green-key hash a door arrives with into the key that names
    /// exactly one cell, so the door's decision, its token read and the run it
    /// hands them to are all about that one cell.
    ///
    /// `warmstate.py maybe_compile_and_run` matches greens on this driver's
    /// WarmEnterState (`JitDriver::index`).
    ///
    /// The producer-side half of `MetaInterp::resolve_cell_key`, exposed
    /// because a door outside this crate carries the same obligation as the one
    /// inside it: `warmstate.py` matches the greens with `comparekey`
    /// before `:483` reads anything off the cell. A door that decides on a bare
    /// hash and hands that hash on has matched nothing, and on a chained bucket
    /// can decide about one cell and run another's code.
    ///
    /// `make_green_key` is called only when the bucket actually holds more than
    /// one cell, so the common path builds no `GreenKey`.
    #[inline]
    pub fn resolve_cell_key(
        &self,
        green_key_hash: u64,
        make_green_key: impl Fn() -> GreenKey,
    ) -> u64 {
        self.meta.resolve_cell_key_on_driver(
            self.index().unwrap_or(0),
            green_key_hash,
            Some(&make_green_key),
        )
    }

    /// Typed twin of [`Self::has_compiled_loop`], and the shape upstream
    /// actually has: `warmstate.py maybe_compile_and_run` walks the bucket and
    /// tests `comparekey(*greenargs)` before deciding anything, and only then
    /// reads `cell.get_procedure_token()` and gates on it. The hash form has
    /// no upstream counterpart at all — a bare bucket hash can answer for
    /// whichever cell holds it — so this is the reader to ask wherever the
    /// typed key is in scope at the decision point.
    ///
    /// Bound on this driver's WarmEnterState (`JitDriver::index`), not leftover
    /// portal slot 0.
    ///
    /// Kept as the typed reader API even though its only callers today are
    /// tests: `the_typed_entry_predicate_reads_the_keys_own_cell_not_the_bucket_head`
    /// is what pins the hash/typed disagreement the entry path resolves away,
    /// and a chained bucket is the only state in which the two can differ.
    #[inline]
    pub fn has_compiled_loop_for_key(&self, key: &GreenKey) -> bool {
        self.meta
            .has_compiled_loop_for_key_on_driver(self.index().unwrap_or(0), key)
    }

    /// Actual key the last compile_loop stored under.
    pub fn last_compiled_key(&self) -> Option<u64> {
        self.meta.last_compiled_key()
    }

    /// Owning loop token of the last successful loop/bridge artifact.
    pub fn last_compiled_artifact_token(
        &self,
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        self.meta.last_compiled_artifact_token()
    }

    pub fn clear_last_compiled_artifact_token(&mut self) {
        self.meta.clear_last_compiled_artifact_token();
    }

    /// warmstate.py:437-444 starting cell's green_key (the cell on which
    /// TRACING must be cleared in the finally block). Returns None when
    /// no trace is in progress.
    pub fn starting_green_key(&self) -> Option<u64> {
        self.meta.starting_green_key()
    }

    /// Check if any guard in the compiled trace has Float-typed fail_args.
    pub fn compiled_trace_has_float_guards(&self, green_key: u64) -> bool {
        self.meta.compiled_trace_has_float_guards(green_key)
    }

    /// Check if the compiled trace is safe for bridge compilation.
    pub fn compiled_trace_safe_for_bridge(
        &self,
        green_key: u64,
        slot_types: &[majit_ir::Type],
    ) -> bool {
        self.meta
            .compiled_trace_safe_for_bridge(green_key, slot_types)
    }

    /// Remove all compiled code. Used when guard-fail recovery detects
    /// unrecoverable state (e.g., null Ref slots from incomplete
    /// resume data), preventing repeated entry→guard-fail loops.
    pub fn invalidate_all_compiled(&mut self) {
        self.meta.clear_compiled_loops();
    }

    /// interp_jit.py releaseall → memmgr.py release_all_loops.
    ///
    /// memmgr.py release_all_loops → alive_loops.clear().
    ///
    /// RPython: alive_loops holds the only strong references to
    /// LoopTokens. Clearing it makes tokens eligible for GC collection.
    /// JitCells hold weakrefs, so tokens referenced by an active thread
    /// stack survive until the stack unwinds. Tokens are NOT invalidated.
    ///
    /// majit has no GC or weakrefs. We clear compiled_loops (the
    /// equivalent of alive_loops) which drops majit's strong ownership.
    /// Warm-state cells are untouched — they can re-trigger compilation.
    /// Machine code is not invalidated: if a guard failure path still
    /// references a removed entry it will simply miss the lookup and
    /// fall back to the interpreter, matching the RPython "dangling
    /// weakref → re-compile" path.
    pub fn mark_all_loops_for_release(&mut self) {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][releaseall] releasing {} compiled loops",
                self.meta.compiled_loops.len()
            );
        }
        // `memmgr.py release_all_loops` — the upstream entry
        // point for `jit_hooks.releaseall` (`rlib/jit_hooks.py:131`)
        // calls `memory_manager.alive_loops.clear()`.  3
        // (`memmgr.rs alive_loops` Arc owner) made `alive_loops`
        // a strong owner of every running loop, so releasing here
        // is the only path that drops those Arcs.
        self.meta.release_all_driver_loops();
        self.meta.clear_compiled_loops();
    }

    /// Invalidate compiled code for a specific trace_id, removing the
    /// compiled_loops entry whose root_trace_id matches.
    pub fn invalidate_compiled_trace(&mut self, trace_id: u64) {
        self.meta.invalidate_compiled_trace(trace_id);
    }

    /// warmspot.py:449 — set the per-driver result_type.
    pub fn set_result_type(&mut self, tp: Type) {
        self.meta.set_result_type(tp);
    }

    /// warmstate.py:385 — whether this driver's portal returns a raw int.
    pub fn has_raw_int_finish(&self) -> bool {
        self.meta.has_raw_int_finish()
    }

    /// Get the loop token number for a compiled loop.
    pub fn get_loop_token_number(&self, green_key: u64) -> Option<u64> {
        self.get_loop_token(green_key).map(|t| t.number)
    }

    /// Get the owning token object for a compiled loop.
    pub fn get_loop_token_arc(
        &self,
        green_key: u64,
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        self.get_loop_token(green_key)
    }

    /// Get the pre-allocated token number for the trace being recorded.
    ///
    /// Returns `Some(number)` if `green_key` matches the current trace's
    /// target, enabling self-recursive call_assembler emission.
    pub fn get_pending_token_number(&self, green_key: u64) -> Option<u64> {
        self.meta.get_pending_token_number(green_key)
    }

    /// Get the pre-allocated token object for the trace being recorded.
    pub fn get_pending_token_arc(
        &self,
        green_key: u64,
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        self.meta.get_pending_token_arc(green_key).cloned()
    }

    /// Resolve a portal CALL_ASSEMBLER target through an installed loop token
    /// or an RPython tmp callback (`warmstate.py get_assembler_token`,
    /// `compile.py:1101-1150`). The pending token remains only a
    /// trace-in-progress marker for inline decisions.
    pub fn get_or_make_portal_assembler_token_arc(
        &mut self,
        key: &majit_ir::GreenKey,
        greenboxes: &[Value],
        red_arg_types: &[Type],
    ) -> Option<std::sync::Arc<majit_backend::JitCellToken>> {
        self.meta
            .get_or_make_portal_assembler_token_arc(key, greenboxes, red_arg_types)
    }

    /// Decide how to handle a function call during tracing.
    ///
    /// Returns `Inline` if the callee should be traced through,
    /// `CallAssembler` if it already has compiled code, or
    /// `ResidualCall` if it should be left as an opaque call.
    pub fn should_inline(&mut self, callee_key: u64, callee_raw: (usize, usize)) -> InlineDecision {
        self.meta.should_inline(callee_key, callee_raw)
    }

    /// Inline decision with externally-held ctx (merge_point callback).
    pub fn should_inline_with_ctx(
        &mut self,
        callee_key: u64,
        callee_raw: (usize, usize),
        ctx: &crate::trace_ctx::TraceCtx,
    ) -> InlineDecision {
        self.meta
            .should_inline_with_ctx(callee_key, callee_raw, ctx)
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Pushes an inline frame. No ENTER_PORTAL_FRAME is recorded — an inline
    /// call is not a portal transition, and [`MetaInterp::enter_inline_frame`]
    /// carries the reasoning. A recursive PORTAL entry is the shape that does
    /// record one, in `exec_recursive_call`.
    ///
    /// Returns `true` if inlining started successfully.
    pub fn enter_inline_frame(&mut self, callee_raw: (usize, usize)) -> bool {
        self.meta.enter_inline_frame(callee_raw)
    }

    /// End an inlined function call during tracing.
    ///
    /// Pops the inline frame. No LEAVE_PORTAL_FRAME, for the reason its
    /// [`Self::enter_inline_frame`] counterpart gives.
    pub fn leave_inline_frame(&mut self) {
        self.meta.leave_inline_frame()
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.meta.inline_depth()
    }

    /// RPython equivalent: `opimpl_hint_force_virtualizable(box)`
    ///
    /// Call during tracing when the interpreter encounters
    /// `hint_force_virtualizable`. Emits IR to flush virtualizable
    /// boxes back to the heap.
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        self.meta.opimpl_hint_force_virtualizable(vable_opref);
    }

    pub fn opimpl_getfield_vable_int(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta
            .opimpl_getfield_vable_int(pc, vable_opref, vable_struct_ptr, fielddescr)
    }

    pub fn opimpl_getfield_vable_ref(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta
            .opimpl_getfield_vable_ref(pc, vable_opref, vable_struct_ptr, fielddescr)
    }

    pub fn opimpl_getfield_vable_float(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta
            .opimpl_getfield_vable_float(pc, vable_opref, vable_struct_ptr, fielddescr)
    }

    pub fn opimpl_setfield_vable_int(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_int(pc, vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_setfield_vable_ref(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_ref(pc, vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_setfield_vable_float(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_float(pc, vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_getarrayitem_vable_int(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta.opimpl_getarrayitem_vable_int(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            fdescr,
            adescr,
        )
    }

    pub fn opimpl_getarrayitem_vable_ref(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta.opimpl_getarrayitem_vable_ref(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            fdescr,
            adescr,
        )
    }

    pub fn opimpl_getarrayitem_vable_float(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        self.meta.opimpl_getarrayitem_vable_float(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            fdescr,
            adescr,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn opimpl_setarrayitem_vable_int(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_int(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
            adescr,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn opimpl_setarrayitem_vable_ref(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_ref(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
            adescr,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn opimpl_setarrayitem_vable_float(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_float(
            pc,
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
            adescr,
        );
    }

    pub fn opimpl_arraylen_vable(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> OpRef {
        self.meta
            .opimpl_arraylen_vable(pc, vable_opref, vable_struct_ptr, fdescr, adescr)
    }

    /// Start bridge tracing from a guard failure point.
    ///
    /// Uses the compiled loop's stored meta so that the sym's
    /// storage_layout matches the parent loop's inputargs format.
    /// `resume_pc` is where interpretation resumes after the guard failure.
    pub fn start_bridge_tracing(
        &mut self,
        // `pyjitpl.py handle_guard_failure(self, resumedescr,
        // deadframe)` threads the descr (`resumedescr`) as the
        // canonical bridge-source identity.  The descr Arc returned by
        // `cpu.get_latest_descr` (`history.py:125`) plays the same role
        // — `descr_owning_jct(arc).green_key`, `arc.trace_id()` and
        // `arc.fail_index_per_trace()` are the line-by-line readers.
        // Pyre threads `Arc<dyn Descr>` and obtains the `FailDescr`
        // facet via `as_fail_descr()` (sub-trait of `Descr`).
        descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
        state: &mut S,
        env: &S::Env,
        raw_fail_values: &[i64],
        resume_pc: usize,
        // Which half of `resume.py`'s reader pair the entry needs: true for an
        // entry no direct reader preceded, so the replay must apply each write
        // as `execute_and_record` does as well as record it.
        execute_replay: bool,
        // The objects the direct reader that preceded a recording-only entry
        // allocated, by virtual number (`ResumeDataDirectReader`'s
        // `all_virtuals`). Empty for an applying entry.
        all_virtuals: &[Option<majit_ir::GcRef>],
    ) -> bool {
        // Same close as `force_start_tracing`: bridge codegen reads
        // `type_info_group` (`gctypelayout.py encode_type_shapes_now`).
        majit_gc::ensure_type_registry_closed();
        majit_metainterp::mc_diag_bump(12); // start_bridge_tracing entered
        // Same reason as the primary trace entry: the bridge compile decodes
        // frame value counts through the per-thread store, so aim it here.
        self.republish_state_field_fvc();
        self.bridge_attempt_declined = false;
        // compile.py `_trace_and_compile_from_bridge` raises
        // `compile.giveup()` when the descr's owning JitCellToken weakref
        // is dead (memmgr-evicted).  Pyre signals the same outcome by
        // returning false; also returns false if the descr is not a
        // FailDescr at all (e.g. a synthetic terminal-exit Descr).
        let Some(descr_fd) = descr_arc.as_fail_descr() else {
            majit_metainterp::mc_diag_bump(13); // sbt early: descr not FailDescr
            return false;
        };
        let Some(jct) = majit_backend::descr_owning_jct(descr_fd) else {
            majit_metainterp::mc_diag_bump(14); // sbt early: no owning jct
            return false;
        };
        let green_key = jct.green_key();
        let trace_id = descr_fd.trace_id();
        let fail_index = descr_fd.fail_index_per_trace();
        // `pyjitpl.py` `raise_continue_running_normally` leaves this
        // path once a bridge is attached. A later `must_compile` FIRED
        // here is a pyre re-entry; walking again can abort setup and
        // terminally decline the working source.
        // Prefer the descr-side mark (`assembler.py patch_jump_for_descr`
        // zeroes `adr_jump_offset` once the guard jumps into a bridge).
        // `bridge_attached` returns `Some(false)` while the guard still
        // owns its recovery stub; `None` falls back to the token map.
        let already_compiled = match self.meta.backend.bridge_attached(descr_fd) {
            Some(attached) => attached,
            None => self
                .meta
                .bridge_was_compiled(green_key, trace_id, fail_index),
        };
        if already_compiled {
            return false;
        }
        // compile.py `_trace_and_compile_from_bridge` already gated on
        // `rd_loop_token.loop_token_wref()` (`descr_owning_jct` above).
        // `get_compiled_meta(green_key)` is the leftover numeric table;
        // a live token with no `compiled_loops` row still traces.

        if !state.can_trace() {
            majit_metainterp::mc_diag_bump(16); // sbt early: !can_trace
            return false;
        }

        // RPython parity: bridge tracing starts from the state rebuilt after
        // guard failure, not from the original loop-header metadata.
        // compile.py handle_fail -> pyjitpl.handle_guard_failure ->
        // rebuild_state_after_failure() resumes the current interpreter
        // position and traces forward from there.
        let mut trace_meta = state.build_meta(resume_pc, env);

        // `compile.py:855` `_attrs_` parity: bridge inputarg count is the
        // length of the source descr's `fail_arg_types`.  Read directly
        // from `descr_fd` (the Arc already in scope) — no `(green_key,
        // trace_id, fail_index)` reverse lookup.
        let fail_arg_count = descr_fd.fail_arg_types().len();
        let Some(frontend_fail_values) = raw_fail_values.get(..fail_arg_count) else {
            majit_metainterp::mc_diag_bump(17); // sbt early: fail_values too short
            return false;
        };

        // `pyjitpl.py _prepare_exception_resumption` reads
        // `cpu.grab_exc_value(deadframe)` itself. The bridge entry parked
        // that word in `GUARD_EXC_VALUE` (`GuardExcRoot`); this call does
        // not take a copied `i64`.
        let retrace = match self.meta.handle_guard_failure(
            descr_arc.clone(),
            green_key,
            trace_id,
            fail_index,
            frontend_fail_values,
        ) {
            Some(r) => r,
            None => return false,
        };

        // resume.py rebuild_from_resumedata parity: decode rd_numb
        // to reconstruct the complete frame for bridge tracing.
        // Falls back to update_meta_for_bridge (legacy truncation) when
        // rd_numb is not available.
        //
        // compile.py `ResumeGuardDescr` storage — pass the shared
        // Arc to the decoder so it borrows the guard-owned pool (no
        // clone) and can thread the same handle into
        // `ResumeDataResult`.
        //
        // `pyjitpl.py rebuild_state_after_failure` parity status:
        //
        //   * `ResumeDataResult.virtualref_values` (vref pair stream) is
        //     restored into `TraceCtx.virtualref_boxes` and each pair fires
        //     `vrefinfo.continue_tracing(vref, virtual)`
        //     (`virtualref.py`) inside
        //     `pyre-jit-trace::state::setup_bridge_sym` — matches the
        //     `consume_virtualref_info` half of `consume_vref_and_vable`
        //     (resume.py).
        //   * `ResumeDataResult.virtualizable_boxes` (vable scalar +
        //     array stream) is restored into `sym` via
        //     `seed_virtualizable_boxes`, populating
        //     `ctx.virtualizable_boxes` / `virtualizable_array_lengths`
        //     from resume-decoded boxes (each box carries its concrete).
        //
        //   * `pyjitpl.py self.synchronize_virtualizable()`, the
        //     routine's closing line, writes those same vable boxes back
        //     onto the live virtualizable. It is
        //     `ctx.synchronize_virtualizable_after_guard_failure()` below,
        //     fired once `setup_bridge_sym` has seeded the boxes and only on
        //     an entry that asked to apply — an entry the blackhole preceded
        //     has already had them written.
        //
        //     A frontend whose own guard-failure recovery performs that
        //     write before bridge tracing starts declares
        //     `JitState::SYNCHRONIZES_VIRTUALIZABLE_AFTER_GUARD_FAILURE` and
        //     is skipped here. That is not a preference: this writer
        //     converts every slot through `value_to_raw_bits`, while a
        //     frontend whose Ref array slots hold boxed integers and floats
        //     needs a field-aware conversion that allocates through its own
        //     object model, which cannot be reached from this crate.
        //     Unifying the two (likely a field-convert trait threaded into
        //     `TraceCtx`) is a separate epic.
        let resume_data_result = S::rebuild_from_resumedata(
            &mut trace_meta,
            &retrace.fail_types,
            retrace.storage.as_ref(),
        );
        if resume_data_result.is_none() {
            if crate::bridge_debug_enabled() {
                eprintln!("[bridgeB] resume data could not be rebuilt — giving up on the bridge",);
            }
            self.meta.stage_abort_reason(AbortReason::Bridge.as_int());
            self.meta.abort_trace(false);
            self.clear_tracing_session_state();
            self.resume_data_result = None;
            self.last_bridge_is_exception_guard = false;
            return false;
        }
        // resume.py `ResumeDataBoxReader.__init__` parks storage, then
        // `AbstractResumeDataReader._prepare` (`_prepare_virtuals` then
        // `_prepare_pendingfields`) runs before `rebuild_from_resumedata`'s
        // `newframe` / `consume_boxes`. `handle_guard_failure` already ran
        // `create_history` (`start_retrace_from_guard` set `self.meta.tracing`).
        // Park storage on that live ctx first; `prepare_bridge_resume` is
        // `_prepare`; `rebuild_portal_framestack_from_resumedata` is
        // `newframe` + `consume_boxes`; `setup_bridge_sym` fills the
        // frontend after those frames exist.
        if let Some(bfm) = resume_data_result.as_ref() {
            let ctx = self
                .meta
                .tracing
                .as_mut()
                .expect("bridge: tracing context must be live after start_retrace_from_guard");
            ctx.set_bridge_resume_data(bfm.clone());
        }
        let mut sym = S::create_sym(&trace_meta, resume_pc);
        state.initialize_sym(&mut sym, &trace_meta);
        // `create_sym` numbers the state fields off its own `__offset`, which
        // matches a loop trace's inputargs because `extract_live` walks the
        // same declaration order. This trace's inputargs came from
        // `fail_descr.fail_arg_types()` instead (`start_retrace_from_guard`),
        // so the two orders are unrelated and a surviving mint names whichever
        // failarg happens to sit at that position. `setup_bridge_sym` below is
        // the binding authority here; drop the mints so a field it does not
        // reach reads as unbound rather than as a live reference to a value it
        // never named. The concrete `_value` mirrors `initialize_sym` just
        // seeded are untouched.
        S::clear_sym_inputarg_bindings(&mut sym);
        // Grade the call above, in the one window where it is gradeable.
        //
        // `setup_bridge_sym` rebinds a field to the same value whether it
        // arrived here as `OpRef::NONE` or as the loop-shaped mint
        // `create_sym` left, so every observation taken after it is blind to
        // whether this line ran. Deleting the call is invisible downstream
        // for exactly the kinds that get rebound, which is why two fixtures
        // that once caught it stopped: the repairs made the rebinding
        // unconditional, and an unconditional overwrite erases its target.
        //
        // `None` means the state cannot count and the check is skipped —
        // never that it is clean.
        //
        // Scope. Three conditions have to hold together before this line
        // grades anything, and each is a way for a run to be green without
        // reaching it: `debug_assertions` on, since this is a
        // `debug_assert`; a state that overrides `count_bound_sym_inputargs`,
        // since the `JitState` default returns `None` and only a
        // macro-generated `Sym` carries the override; and a run that forms a
        // bridge and arrives here. Miss any one and the deletion mutant
        // survives with zero firings, so that run says nothing about this
        // line.
        //
        // Do not name a crate here as the place the mutant dies, or as a
        // place it does not. Which crates meet all three conditions is a
        // property of the corpus rather than of this file, so a name reads as
        // a measurement and then goes stale with nothing turning red — which
        // has already happened here in both directions, the second time about
        // this crate's own test targets, some of which do carry a
        // macro-generated `Sym`. Derive the subject from the conditions.
        if let Some(__bound) = S::count_bound_sym_inputargs(&sym) {
            debug_assert_eq!(
                __bound, 0,
                "bridge sym still holds {__bound} loop-shaped OpRef binding(s) \
                 after clear_sym_inputarg_bindings; the recorder numbers a \
                 bridge by the guard's failarg order, so a surviving mint \
                 resolves against whatever that order put at its position \
                 rather than missing"
            );
        }
        self.sym = Some(sym);
        // pyjitpl.py:2890 parity: bridge traces start at resume_pc, not at
        // function entry (pc=0). Set header_pc so init_symbolic correctly
        // detects this is NOT a function-entry trace.
        // pyjitpl.py reached_loop_header parity:
        // Set live lookup callback for has_compiled_targets(greenkey).
        // RPython: ptoken = self.get_procedure_token(greenboxes)
        //          if has_compiled_targets(ptoken): ...
        // Use a raw pointer to self.meta for the callback — safe because
        // bridge tracing runs synchronously within start_bridge_tracing's
        // caller scope, so self.meta outlives the callback.
        let meta_ptr = &self.meta as *const _ as *const ();
        // pyjitpl.py `handle_guard_failure` runs on a MetaInterp whose
        // `jitdriver_sd` is the guard's owning loop's driver. A bridge trace
        // that reaches a loop header (`jit_merge_point` → `compile_loop`)
        // therefore compiles that loop with the same `jitdriver_sd`, so
        // `send_loop_to_backend` (compile.py) emits the virtualizable
        // field-load preamble (`patch_new_loop_to_load_virtualizable_fields`)
        // and the loop's inputarg contract stays reds-only ([frame, ec]).
        // Without seeding the bridge ctx's driver descriptor, the bridge-trace
        // loop compile sees `driver=None`, skips the patch, and bakes the
        // eager-expanded virtualizable fields into the loop inputargs
        // (`index_of_virtualizable = -1`). A recursive CALL_ASSEMBLER into
        // such a loop only delivers [frame, ec] and cannot fill the expanded
        // inputarg slots, dereferencing null at the loop's GUARD_CLASS.
        let bridge_driver_descriptor = self.driver_descriptor_for(state, &trace_meta);
        // The parent loop (the guard's owning loop, keyed by `green_key`) lives
        // at this header pc; the bridge closes by JUMPing into it there, not at
        // its own `resume_pc`. Resolve before the `ctx` mutable borrow below.
        let parent_header_pc = self.meta.loop_header_pc_for_driver(
            jct.outermost_jitdriver_index
                .unwrap_or(self.index().unwrap_or(0)),
            green_key,
        );
        // pyjitpl.py `rebuild_state_after_failure` ends with
        // `synchronize_virtualizable()` / `check_synchronized_virtualizable()`,
        // both of which read the LIVE virtualizable.  Resolve it here (the same
        // `virtualizable_heap_ptr` hook the trace-entry `sync_before` uses) so
        // the bridge's vable seeding has an authority to check the guard's
        // decoded identity against.
        let live_vable = self.meta.virtualizable_info().cloned().and_then(|info| {
            state
                .virtualizable_heap_ptr(&trace_meta, &info.name.clone(), &info)
                .map(|ptr| (info, ptr.cast_const()))
        });
        // `start_retrace_from_guard` above sets `self.meta.tracing = Some(..)`
        // on success (pyjitpl.py:9415). Fail loud rather than skipping bridge
        // header_pc / is_bridge_trace / has_compiled_targets_fn wiring
        // silently.
        let ctx = self
            .meta
            .tracing
            .as_mut()
            .expect("bridge: tracing context must be live after start_retrace_from_guard");
        ctx.header_pc = resume_pc;
        // `ResumeGuardDescr._trace_and_compile_from_bridge` keeps the source
        // loop's outermost driver. Only descriptor-less host entries need the
        // state hook; it must not replace a driver recovered from the guard.
        if ctx.driver_descriptor().is_none()
            && let Some(descriptor) = bridge_driver_descriptor
        {
            ctx.set_driver_descriptor((*descriptor).clone());
        }
        // pyjitpl.py `if not self.partial_trace:` — bridge
        // entry sets the explicit flag so close-loop consumers
        // can apply bridge-only behavior without overloading
        // `has_compiled_targets_fn` presence.
        ctx.is_bridge_trace = true;
        // pyjitpl.py `_handle_guard_failure`:
        //
        //     self.seen_loop_header_for_jdindex = -1
        //     if isinstance(key, compile.ResumeAtPositionDescr):
        //         self.seen_loop_header_for_jdindex = self.jitdriver_sd.index
        //
        // A `ResumeAtPositionDescr` is the descr `inline_short_preamble`
        // stamps onto the guards it replays (unroll.py:337 / :409), so the
        // guard sits at the target loop's entry and the bridge grown from it
        // has to close at the very first merge point it reaches. Pre-arming
        // the flag is what makes that happen: the merge point's own ladder
        // (`if not any_operation` / `portal_call_depth` /
        // `has_compiled_targets`, pyjitpl.py) is skipped and the
        // close protocol runs. Without it the bridge records a whole extra
        // iteration before it can close, and the `bridge_entry_merge_pc`
        // skip-once below sends it past the crossing it was meant to take.
        //
        // The flag is one-shot on both sides: the merge point resets it to
        // -1 (pyjitpl.py) after consuming it, so only the FIRST crossing
        // is forced. `compile_trace`'s sibling read of the same descr class
        // (`inline_short_preamble = False`, compile.py) lives at
        // `pyjitpl.rs`'s bridge compile entry.
        //
        // Upstream writes `self.jitdriver_sd.index` straight into the flag
        // because that value IS the `jdindex` the following merge point
        // carries. Pyre's registered slot and the op's build-time `jdindex`
        // can differ by the `ensure_default_driver_sd` placeholder shift, and
        // the merge point asserts the two agree, so arm a request instead and
        // let the merge point stamp its own `jdindex`.
        //
        // The test is on `get_resumestorage()`, not on the failing descr:
        // upstream's `key` is the resolved storage, so a guard that shares a
        // short-preamble donor's resume data through `ResumeGuardCopiedDescr`
        // arms the flag as well. `compile_trace`'s sibling read is the one
        // that stays unresolved (compile.py:1040 tests `resumekey` itself).
        ctx.bridge_resume_at_position =
            crate::compile::get_resumestorage(&descr_arc).is_resume_at_position();
        if let Some((_, ptr)) = &live_vable {
            ctx.set_virtualizable_heap_ptr(*ptr);
        }
        ctx.set_bridge_source_is_exception_guard(retrace.is_exception_guard);
        ctx.bridge_target_header_pc = parent_header_pc;
        ctx.has_compiled_targets_fn = Some(Box::new(move |gk: u64| -> bool {
            let meta = unsafe { &*(meta_ptr as *const crate::pyjitpl::MetaInterp<S::Meta>) };
            meta.has_compiled_targets(gk)
        }));
        ctx.compiled_key_for_greens_fn = Some(Box::new(
            move |greens: &(Vec<i64>, Vec<i64>, Vec<i64>)| -> Option<u64> {
                let meta = unsafe { &*(meta_ptr as *const crate::pyjitpl::MetaInterp<S::Meta>) };
                meta.compiled_key_for_greens(greens)
            },
        ));
        // pyjitpl.py:1551 first-iteration auto loop-header
        // `if self.metainterp.portal_call_depth: return` parity —
        // sample the live counter through the same self.meta
        // pointer.  Same lifetime invariant as
        // `has_compiled_targets_fn` (bridge tracing runs
        // synchronously within `start_bridge_tracing`'s scope).
        ctx.portal_call_depth_fn = Some(Box::new(move || -> i32 {
            let meta = unsafe { &*(meta_ptr as *const crate::pyjitpl::MetaInterp<S::Meta>) };
            meta.portal_call_depth
        }));
        // resume.py rebuild_from_resumedata parity: map frame locals to bridge InputArg OpRefs
        // so bridge tracing sees locals as symbolic variables, not concrete values.
        // resume.py getvirtual_ptr parity: when frame.values contains
        // RebuiltValue::Virtual entries, setup_bridge_sym must emit
        // NEW_WITH_VTABLE/SETFIELD_GC ops via the active trace ctx so the
        // bridge has fresh local instances instead of stale vable-array reads.
        if let Some(ref bfm) = resume_data_result {
            // bridgeopt.py deserialize_optimizer_knowledge / resume.py:1260 parity: `Box(n, tp)` in  allow-line-citation
            // `rd_numb` indexes the guard's encoder-time liveboxes, whose
            // runtime values are the backend's raw deadframe (the guard's
            // `fail_args` image). `extract_live` returns the virtualizable
            // array in vable layout order — a different stream with a
            // different length. Pass the raw deadframe values that the
            // caller supplied to this guard-failure entry so
            // `decode_concrete(Box(n, tp))` reads `raw_values[n]` and
            // `fail_types[n]` (length-aligned with `fail_descr.fail_arg_types`).
            // `pending_frontend_boxes` is deliberately compacted to the live
            // history InputArgs, as pyjitpl.py
            // `initialize_state_from_guard_failure` filters its holes. Resume
            // decoding still needs the uncompressed deadframe coordinate
            // space, so keep the positional slice the caller passed.
            //
            // `resume.py` `ResumeDataBoxReader.load_box_from_cpu` reads
            // `cpu.get_ref_value(self.deadframe, num)` at the moment the box
            // is decoded. The deadframe stays a traced root, so a collection
            // inside `allocate_with_vtable` forwards the slot before that
            // read. Snapshotting the words here would leave that collection
            // updating the caller's buffer while `setup_bridge_sym` still
            // indexes the copy (`materialize_concrete_virtual_ptr` allocates,
            // then a later field or box reads `fail_values[n]`). The caller
            // keeps this slice on a GC-visible holder for the whole call —
            // the deadframe, or `DeadFrameRefRoots` over the Ref slots.
            // `resume.py` `rebuild_from_resumedata` walks every encoded
            // section: `read_jitcode_pos_pc` then `newframe(jitcode)` then
            // `consume_boxes` with that frame's `get_current_position_info`.
            // Decode each section's live register indices here, where the
            // jitcode table + `liveness_info` are reachable, and stash them
            // on the trace ctx. A JitDriver `setup_bridge_sym` (static,
            // metainterp-blind) reads the root banks back to map each
            // decoded frame value to its sym slot.
            //
            // The frame's `pc` word is a position in the frame's OWN
            // jitcode. A frame resumed inside a runtime-registered jitcode
            // carries a pc the portal jitcode cannot decode, and reading it
            // there yields empty banks: every register of that frame then
            // resumes unset. The dispatch jitcode is only the fallback for
            // a section without an index.
            let bridge_liveness = self.meta.staticdata.liveness_info.snapshot_arc();
            let bridge_reg_indices = bfm.frames.first().and_then(|frame| {
                let Ok(pc) = usize::try_from(frame.pc) else {
                    return Some(crate::resume::FrameLivenessRegIndices::default());
                };
                let jc = usize::try_from(frame.jitcode_index)
                    .ok()
                    .and_then(|pos| {
                        S::resolve_resume_jitcode(pos)
                            .or_else(|| self.meta.staticdata.jitcodes.get(pos).cloned())
                    })
                    .or_else(|| self.dispatch_jitcode().cloned())?;
                Some(crate::resume::read_frame_liveness_reg_indices(
                    &jc,
                    pc,
                    self.meta.staticdata.op_live as u8,
                    &bridge_liveness,
                ))
            });
            // Both `self.sym` (set above via `self.sym = Some(sym)`) and
            // `self.meta.tracing` (set by `start_retrace_from_guard`) must be
            // live by construction at this point. Skipping `setup_bridge_sym`
            // silently would leave the bridge sym uninitialized.
            // resume.py `ResumeDataBoxReader.__init__`: one reader object for
            // the whole `rebuild_from_resumedata` — `_prepare`, then
            // `consume_boxes` per frame, then the frontend's
            // `rebuild_state_after_failure` fill — so `virtuals_cache` and
            // the deadframe values are one object's fields. Built inside the
            // caller's deadframe root window, so the roots it registers nest.
            let replay_allocator: Option<&dyn crate::resume::BlackholeAllocator> = if execute_replay
            {
                self.blackhole_allocator
                    .as_deref()
                    .map(|allocator| allocator as &dyn crate::resume::BlackholeAllocator)
            } else {
                None
            };
            let virtual_count = retrace
                .storage
                .as_deref()
                .map_or(0, |storage| storage.rd_virtuals().map_or(0, <[_]>::len));
            let boxes: crate::VrefVableBoxes;
            let mut reader = match replay_allocator {
                Some(allocator) => crate::BridgeVirtualCache::executing(
                    virtual_count,
                    S::mint_bridge_array_descr,
                    allocator,
                    frontend_fail_values,
                    &retrace.fail_types,
                ),
                None => crate::BridgeVirtualCache::recording(
                    virtual_count,
                    S::mint_bridge_array_descr,
                    all_virtuals,
                ),
            };
            {
                let ctx = self
                    .meta
                    .tracing
                    .as_mut()
                    .expect("bridge: tracing context must be live");
                if let Some(idx) = bridge_reg_indices {
                    ctx.set_bridge_reg_indices(idx);
                }
                // An entry that asked to apply and has no allocator to apply
                // through cannot fall back to recording only: nothing else will
                // write this guard's deferred stores.
                if execute_replay
                    && replay_allocator.is_none()
                    && retrace.storage.as_deref().is_some_and(|storage| {
                        !storage.rd_pendingfields().is_none_or(<[_]>::is_empty)
                            || !storage.rd_virtuals().is_none_or(<[_]>::is_empty)
                    })
                {
                    ctx.mark_bridge_replay_incomplete();
                }
                let sym = self
                    .sym
                    .as_mut()
                    .expect("bridge: sym must be live after S::create_sym");
                S::prepare_bridge_resume(
                    sym,
                    ctx,
                    bfm,
                    retrace.storage.as_deref().and_then(|s| s.rd_virtuals()),
                    frontend_fail_values,
                    &retrace.fail_types,
                    &mut reader,
                );
                // resume.py `rebuild_from_resumedata`:
                // `consume_vref_and_vable_boxes` before the frame loop.
                boxes = S::consume_vref_and_vable_boxes(
                    sym,
                    ctx,
                    bfm,
                    retrace.storage.as_deref().and_then(|s| s.rd_virtuals()),
                    frontend_fail_values,
                    &retrace.fail_types,
                    &mut reader,
                );
            }
            // resume.py `rebuild_from_resumedata`: `newframe(jitcode)` per
            // encoded section then `consume_boxes`. `setup_bridge_sym` is
            // `rebuild_state_after_failure`'s vable / pending / register
            // fill and runs after those frames exist, matching
            // `synchronize_virtualizable` after the newframe loop.
            if let Some(mainjitcode) = self.dispatch_jitcode().cloned() {
                let materialized: Vec<Option<std::sync::Arc<crate::jitcode::JitCode>>> = bfm
                    .frames
                    .iter()
                    .map(|section| {
                        usize::try_from(section.jitcode_index).ok().and_then(|pos| {
                            S::resolve_resume_jitcode(pos)
                                .or_else(|| self.meta.staticdata.jitcodes.get(pos).cloned())
                        })
                    })
                    .collect();
                let liveness = self.meta.staticdata.liveness_info.snapshot_arc();
                let op_live = self.meta.staticdata.op_live as u8;
                // `self.meta` directly: `reader` borrows this driver's
                // allocator for the whole rebuild.
                let rebuilt = self.meta.rebuild_portal_framestack_from_resumedata(
                    mainjitcode,
                    &bfm.frames,
                    frontend_fail_values,
                    &retrace.fail_types,
                    &materialized,
                    liveness.as_ref(),
                    op_live,
                    &mut reader,
                );
                if !rebuilt {
                    drop(reader);
                    self.meta.stage_abort_reason(AbortReason::Bridge.as_int());
                    self.meta.abort_trace(false);
                    self.clear_tracing_session_state();
                    self.resume_data_result = None;
                    self.last_bridge_is_exception_guard = false;
                    return false;
                }
            }
            {
                let sym = self
                    .sym
                    .as_mut()
                    .expect("bridge: sym must be live after S::create_sym");
                let ctx = self
                    .meta
                    .tracing
                    .as_mut()
                    .expect("bridge: tracing context must be live");
                S::setup_bridge_sym(
                    sym,
                    ctx,
                    bfm,
                    retrace.storage.as_deref().and_then(|s| s.rd_virtuals()),
                    frontend_fail_values,
                    &retrace.fail_types,
                    &mut reader,
                    &boxes,
                );
                // pyjitpl.py `MetaInterp.rebuild_state_after_failure` tail.
                // Assembler (virtualizable.py case 4) into tracing (case 2).
                if let Some((info, ptr)) = &live_vable
                    && info.has_vable_token()
                {
                    unsafe {
                        if info.is_token_nonnull_gcref(*ptr) {
                            info.reset_token_gcref(*ptr as *mut u8);
                        }
                    }
                }
                if execute_replay && !S::SYNCHRONIZES_VIRTUALIZABLE_AFTER_GUARD_FAILURE {
                    ctx.synchronize_virtualizable_after_guard_failure();
                }
            }
        }
        self.meta.begin_trace_session(trace_meta);
        // resume.py:1047-1055 parity:
        //   ResumeDataBoxReader.consume_boxes() rebuilds the frame state,
        //   and bridge tracing continues from that restored interpreter
        //   frame.  The authoritative slot metadata is therefore the
        //   state rebuilt by `state.build_meta(resume_pc, env)`, not the
        //   raw `RebuiltFrame.values` section.
        //
        // `RebuiltFrame.values` contains only the frame's register slots;
        // it does NOT have pyre's old synthetic [frame, ni, vsd] bridge
        // header. Reinterpreting those slot values as header-prefixed
        // fail_arg types shifts the frame layout and can mis-bind bridge
        // return values.
        //
        // Keep `trace_meta` as built from the restored state when rd_numb
        // is available; the bridge path should not synthesize a separate
        // header-shaped meta layer.
        //
        // Constants are NOT injected into the pool up-front: resume.py
        // `decode_box` returns live ConstInt/ConstPtr boxes at the moment
        // each tagged value is read, so majit's parity path registers each
        // constant via `ctx.const_int/const_ref/const_float` inside
        // `setup_bridge_sym`'s resolve closure and inside
        // `materialize_bridge_virtual`'s `decode_fieldnum`.
        self.resume_data_result = resume_data_result;
        let code_ptr = state.code_ptr();
        self.meta
            .set_bridge_trace_info(crate::pyjitpl::BridgeTraceInfo {
                green_key,
                trace_id,
                fail_index,
                code_ptr,
                // `pyjitpl.py` `handle_guard_failure(self,
                // resumedescr, deadframe)` parity: thread the source
                // descr Arc obtained from `cpu.get_latest_descr`
                // through the bridge session so `compile_trace_inner`
                // reads it directly (no `(trace_id, fail_index)`
                // reverse lookup).
                source_descr: descr_arc.clone(),
            });
        // RPython pyjitpl.py:2908 — bridge traces start with empty
        // current_merge_points (no loop header to match against).
        let ctx = self
            .meta
            .tracing
            .as_mut()
            .expect("bridge: tracing context must be live");
        ctx.clear_merge_points();

        // RPython pyjitpl.py _prepare_exception_resumption parity:
        // For exception guard bridges, the caller should emit
        // SAVE_EXC_CLASS + SAVE_EXCEPTION at the trace start.
        // Store the flag so it's accessible to the pyre-jit caller.
        self.last_bridge_is_exception_guard = retrace.is_exception_guard;

        true
    }
}

/// A jitdriver with no scalar identity slots. The portal is this shape:
/// its reds are the virtualizable frame and `ec`, not `#[jit_interp]`
/// scalars, and `state_field_layout` stays the empty default.
fn layout_has_no_scalar_identity(layout: &crate::blackhole::StateFieldLayout) -> bool {
    layout.num_scalars == 0
        && layout.num_ref_scalars == 0
        && layout.num_float_scalars == 0
        && layout.num_vable_identity_slots == 0
        && layout.array_lens.is_empty()
}

/// `ContinueRunningNormally` restore inputs.
///
/// No scalar identity slots: the portal calldescr is `iirrr`, so the
/// int bank is greens (`next_instr`, `is_being_profiled`) and the ref
/// bank leads with the green `pycode`. The red frame is
/// `args.red_ref[0]` (`interp_jit.py` reds `['frame', 'ec']`).
/// Passing the int bank to `restore` stores the green pc as that frame.
///
/// A state-field layout keeps the identity-slot slice of the terminal
/// register file. `terminal` is absent only when the chain produced no
/// image; the reds are still the CRN arguments.
fn crn_restore_banks<'a>(
    layout: &crate::blackhole::StateFieldLayout,
    terminal: Option<&'a crate::blackhole::BlackholeTerminalImage>,
    args: &'a crate::jitexc::ContinueRunningNormallyArgs,
) -> (&'a [i64], &'a [i64], &'a [i64]) {
    if layout_has_no_scalar_identity(layout) {
        return (&args.red_int, &args.red_ref, &args.red_float);
    }
    match terminal {
        Some(terminal) => abort_blackhole_restore_banks(layout, terminal),
        None => (&args.red_int, &args.red_ref, &args.red_float),
    }
}

fn abort_counter_name(reason: i32) -> &'static str {
    match reason {
        crate::pyjitpl::counters::ABORT_TOO_LONG => "ABORT_TOO_LONG",
        crate::pyjitpl::counters::ABORT_BRIDGE => "ABORT_BRIDGE",
        crate::pyjitpl::counters::ABORT_BAD_LOOP => "ABORT_BAD_LOOP",
        crate::pyjitpl::counters::ABORT_ESCAPE => "ABORT_ESCAPE",
        crate::pyjitpl::counters::ABORT_FORCE_QUASIIMMUT => "ABORT_FORCE_QUASIIMMUT",
        crate::pyjitpl::counters::ABORT_SEGMENTED_TRACE => "ABORT_SEGMENTED_TRACE",
        _ => "ABORT",
    }
}

/// Declaration-order green banks from a register file.
///
/// `blackhole.py bhimpl_jit_merge_point` indexes the merge-point's green
/// operands, not the JitCode register file by position. A loop-carried
/// green lives in its `join_merge` header register.
#[cfg(test)]
fn greens_from_named_registers(
    green_i: &[u8],
    green_r: &[u8],
    green_f: &[u8],
    registers_i: &[i64],
    registers_r: &[i64],
    registers_f: &[i64],
) -> ContinueRunningNormallyArgs {
    let mut args = ContinueRunningNormallyArgs::default();
    args.fill_greens_from_named_registers(
        green_i,
        green_r,
        green_f,
        registers_i,
        registers_r,
        registers_f,
    );
    args
}

fn greens_from_blackhole_registers(
    portal_jitcode: Option<&crate::jitcode::JitCode>,
    terminal: Option<&crate::blackhole::BlackholeTerminalImage>,
) -> ContinueRunningNormallyArgs {
    let jitcode = portal_jitcode.expect(
        "ContinueRunningNormally banks need the portal JitCode \
         (blackhole.py bhimpl_jit_merge_point)",
    );
    let terminal = terminal.expect(
        "ContinueRunningNormally banks need the terminal register file \
         (blackhole.py bhimpl_jit_merge_point)",
    );
    let mut args = ContinueRunningNormallyArgs::default();
    let Some((gi, gr, gf)) = jitcode.merge_point_green_regs() else {
        return args;
    };
    args.fill_greens_from_named_registers(
        gi,
        gr,
        gf,
        &terminal.registers_i,
        &terminal.registers_r,
        &terminal.registers_f,
    );
    args
}

fn continue_at_pc(pc: usize) -> PortalResume {
    PortalResume::ResumeAt(pc)
}

#[cfg(test)]
mod green_bank_mapping_tests {
    use super::greens_from_named_registers;

    /// `greens = [pc, a, kind, b]`: declaration index 2 is `kind`, whose
    /// join_merge register is i1, not i2. Copying the raw int bank would
    /// write `registers_i[2]` into `kind`.
    #[test]
    fn declaration_order_reads_the_join_merge_register() {
        let green_i = [0u8, 5, 1, 6];
        let registers_i = vec![10, 99, 7, 8, 9, 70, 90];
        let args = greens_from_named_registers(&green_i, &[], &[], &registers_i, &[], &[]);
        assert_eq!(args.green_int, vec![10, 70, 99, 90]);
        assert_eq!(args.green_int[2], 99);
        assert_ne!(args.green_int[2], registers_i[2]);
    }
}

fn continue_with_args(args: ContinueRunningNormallyArgs) -> PortalResume {
    PortalResume::ContinueRunningNormally(args)
}

fn abort_blackhole_restore_banks<'a>(
    layout: &crate::blackhole::StateFieldLayout,
    terminal: &'a crate::blackhole::BlackholeTerminalImage,
) -> (&'a [i64], &'a [i64], &'a [i64]) {
    let int_base = layout.int_scalar_base.min(terminal.registers_i.len());
    let ref_base = layout.ref_scalar_base.min(terminal.registers_r.len());
    let float_base = layout.float_scalar_base.min(terminal.registers_f.len());
    (
        &terminal.registers_i[int_base..],
        &terminal.registers_r[ref_base..],
        &terminal.registers_f[float_base..],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::ReconstructedFrame;
    use majit_ir::{GcRef, OpCode, OpRef, Type, Value};
    use std::sync::Arc;

    // `seed_deopt_vinfo_ptr` decides which guard-failure deopts hand the blackhole
    // a non-null `bh.virtualizable_info`. A state-field machine (no `vable_token`)
    // must be seeded so a mid-body vable-array op — the `int_*_jump_if_ovf` overflow
    // guard on a `[int; virt]` field — resolves its vinfo during resume instead of
    // panicking. A real heap virtualizable (e.g. PyFrame) keeps the prior
    // seeded too, so a later `getfield_vable_*` in that resume can run.
    #[test]
    fn seed_deopt_vinfo_ptr_seeds_state_field_and_skips_heap_virtualizable() {
        use crate::virtualizable::VirtualizableInfo;

        // no vable_token → seed the reconstructed vinfo pointer (always, since
        // its `bh_clear_vable_token` is inert and cannot corrupt the state struct).
        let state_field = std::sync::Arc::new(VirtualizableInfo::without_vable_token());
        assert_eq!(
            seed_deopt_vinfo_ptr(Some(&state_field)),
            std::sync::Arc::as_ptr(&state_field),
            "a state-field machine with no vable_token must seed a non-null vinfo",
        );

        // No vinfo available → null.
        assert!(seed_deopt_vinfo_ptr(None).is_null());

        // A heap virtualizable is seeded too: a guard that fails before a
        // later `getfield_vable_*` otherwise resumes with a null vinfo.
        let heap_vable = std::sync::Arc::new(VirtualizableInfo::new(8));
        assert_eq!(
            seed_deopt_vinfo_ptr(Some(&heap_vable)),
            std::sync::Arc::as_ptr(&heap_vable),
            "a heap virtualizable deopt must seed vinfo so getfield_vable can run",
        );
    }

    #[test]
    fn portal_iirrr_crn_restore_feeds_red_frame_not_green_pc() {
        use crate::blackhole::{BlackholeTerminalImage, StateFieldLayout};
        use crate::jitexc::ContinueRunningNormallyArgs;

        let layout = StateFieldLayout::default();
        let terminal = BlackholeTerminalImage {
            jitcode_index: 0,
            registers_i: vec![34, 0],
            registers_r: vec![0x111, 0x222, 0x333],
            registers_f: Vec::new(),
            position: 0,
            last_opcode_position: 0,
            abort_permanent_bail: false,
            bail_residual: None,
        };
        let args = ContinueRunningNormallyArgs {
            green_int: vec![34, 0],
            green_ref: vec![0x111],
            red_ref: vec![0xABC0, 0xEC],
            ..ContinueRunningNormallyArgs::default()
        };
        let (ints, refs, floats) = crn_restore_banks(&layout, Some(&terminal), &args);
        assert!(
            ints.is_empty(),
            "green pc must not be the first restored int"
        );
        assert_eq!(refs, &[0xABC0, 0xEC]);
        assert!(floats.is_empty());
        assert_ne!(refs.first().copied(), Some(34));
    }

    #[derive(Default)]
    struct TypedRestoreState {
        live_values: Vec<i64>,
        restored_values: Vec<Value>,
        restore_called: bool,
    }

    #[derive(Default)]
    struct CountingDoorState {
        build_meta_calls: std::cell::Cell<usize>,
        extract_live_values_calls: std::cell::Cell<usize>,
        code_ptr: usize,
    }

    #[derive(Default)]
    struct TypedInputState {
        raw_live_values: Vec<i64>,
        typed_live_values: Vec<Value>,
        restored_values: Vec<Value>,
        raw_restore_calls: usize,
        typed_restore_calls: usize,
    }

    #[derive(Default)]
    #[allow(dead_code)]
    struct FrameMetadataState {
        seen_trace_id: Option<u64>,
        seen_header_pc: Option<u64>,
        seen_source_guard: Option<(u64, u32)>,
        restored_values: Vec<Value>,
    }

    #[derive(Default)]
    struct NonTraceableState;

    impl JitState for TypedRestoreState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.restore_called = true;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for CountingDoorState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
            self.build_meta_calls.set(self.build_meta_calls.get() + 1);
        }

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
            self.extract_live_values_calls
                .set(self.extract_live_values_calls.get() + 1);
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn code_ptr(&self) -> usize {
            self.code_ptr
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct IncompatibleDoorState;

    impl JitState for IncompatibleDoorState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            false
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    /// `compile.py make_and_attach_done_descrs([self, cpu])` runs from
    /// `MetaInterp::new`, which `JitDriver::new` calls, and again from
    /// `finish_setup_descrs_for_jitdrivers` when the descriptor is registered.
    /// The int cell is that singleton's address.
    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[test]
    fn driver_setup_publishes_done_int_cell() {
        let mut driver = JitDriver::<CountingDoorState>::new(1);
        let cell = driver.meta.backend().done_with_this_frame_descr_int_cell();
        assert_ne!(cell, 0);
        assert_eq!(
            cell,
            driver
                .meta
                .backend()
                .attached_done_with_this_frame_descr_int()
        );
        driver.ensure_descriptor_registered();
        assert_eq!(
            driver.meta.backend().done_with_this_frame_descr_int_cell(),
            cell
        );
    }

    impl JitState for TypedInputState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.raw_live_values.clone()
        }

        fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
            self.typed_live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.raw_restore_calls += 1;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.typed_restore_calls += 1;
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for FrameMetadataState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &Self::Meta,
            _frame_index: usize,
            _total_frames: usize,
            frame: &ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.seen_trace_id = frame.trace_id;
            self.seen_header_pc = frame.header_pc;
            self.seen_source_guard = frame.source_guard;
            self.restored_values = values.to_vec();
            true
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for NonTraceableState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn can_trace(&self) -> bool {
            false
        }

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    fn compiled_structured_back_edge_fixture() -> (
        JitDriver<TypedRestoreState>,
        u64,
        GreenKey,
        TypedRestoreState,
    ) {
        let code = 0x7100usize;
        let pc = 19usize;
        let key = GreenKey::with_types(
            vec![pc as i64, 0, code as i64],
            vec![Type::Int, Type::Int, Type::Ref],
        );
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        // A real frontend constructs its JitDriver before hashing any Ref
        // greens: WarmEnterState::new installs the GC-owned identity-hash
        // resolver.  Keep the fixture in that order too, otherwise `hash`
        // uses raw pointer bits while the compiled-entry door recomputes the
        // same key through the installed, move-stable identity hash.
        let hash = key.get_uhash();
        // `resume.py blackhole_from_resumedata` indexes
        // `metainterp_sd.jitcodes[jitcode_pos]` unconditionally.  The guard
        // snapshot below names jitcode 0, so give the fixture the same
        // codewriter-owned registry a real translated portal always has.
        // Returning the one restored int also lets the synthetic failing
        // guard finish its reconstructed frame normally.
        let mut resume_asm = majit_jitcode::codewriter::assembler::Assembler::new();
        resume_asm.register_insn("live/", crate::jitcode::insns::BC_LIVE);
        resume_asm.register_insn(
            "catch_exception/L",
            crate::jitcode::insns::BC_CATCH_EXCEPTION,
        );
        resume_asm.register_insn("rvmprof_code/ii", crate::jitcode::insns::BC_RVMPROF_CODE);
        let mut resume_builder = crate::JitCodeBuilder::new();
        resume_builder.live(&mut resume_asm, &[0], &[], &[]);
        resume_builder.int_return(0);
        let resume_jitcode = resume_builder.finish();
        resume_jitcode.set_index(0);
        driver.install_canonical_liveness(&resume_asm);
        driver
            .meta_interp_mut()
            .install_jitcodes(vec![Arc::new(resume_jitcode)]);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let live = [Value::Int(1)];
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(hash, (code, pc), None, None, &live),
            BackEdgeAction::StartedTracing,
        ));
        {
            let ctx = driver
                .meta
                .trace_ctx()
                .expect("trace starts at threshold 1");
            let i0 = OpRef::input_arg_int(0);
            let guard = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0], 0, 0);
            ctx.set_fail_args(guard, &[i0]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert!(driver.has_compiled_loop(hash));
        let state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };
        (driver, hash, key, state)
    }

    /// `warmstate.py maybe_compile_and_run` consults `confirm_enter_jit`
    /// after finding a procedure token and before raising
    /// `EnterJitAssembler`; the structured back-edge door owes the same veto.
    #[test]
    fn a_confirm_enter_jit_veto_stops_the_structured_back_edge_compiled_door() {
        static CONFIRMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn refuse(_key: &GreenKey) -> bool {
            CONFIRMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false
        }

        CONFIRMS.store(0, std::sync::atomic::Ordering::Relaxed);
        let (mut driver, hash, key, mut state) = compiled_structured_back_edge_fixture();
        let compiled_entries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted_entries = Arc::clone(&compiled_entries);
        driver.set_on_compiled_entry(move |_, _| {
            counted_entries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        driver
            .meta
            .warm_state_for_driver(0)
            .set_confirm_enter_jit(Some(refuse));
        let mut pre_ran = false;

        let outcome = driver.back_edge_structured(
            hash,
            || key.clone(),
            19,
            &mut state,
            &(),
            || pre_ran = true,
        );

        assert_eq!(outcome, None);
        assert_eq!(CONFIRMS.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(!pre_ran, "a veto returns before the compiled-entry pre-run");
        assert_eq!(
            compiled_entries.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the rejected structured back edge must not enter compiled code",
        );
    }

    /// The positive control distinguishes a working confirmation gate from a
    /// door that never reaches compiled code at all.
    #[test]
    fn a_permissive_confirm_enter_jit_preserves_the_structured_back_edge_compiled_door() {
        static CONFIRMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn allow(_key: &GreenKey) -> bool {
            CONFIRMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }

        CONFIRMS.store(0, std::sync::atomic::Ordering::Relaxed);
        let (mut driver, hash, key, mut state) = compiled_structured_back_edge_fixture();
        let compiled_entries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted_entries = Arc::clone(&compiled_entries);
        driver.set_on_compiled_entry(move |_, _| {
            counted_entries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        driver
            .meta
            .warm_state_for_driver(0)
            .set_confirm_enter_jit(Some(allow));
        let mut pre_ran = false;

        let outcome = driver.back_edge_structured(
            hash,
            || key.clone(),
            19,
            &mut state,
            &(),
            || pre_ran = true,
        );

        assert!(outcome.is_some(), "the permissive door executes the loop");
        assert_eq!(CONFIRMS.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(pre_ran, "the permissive door reaches the compiled pre-run");
        assert_eq!(
            compiled_entries.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the permissive structured back edge must enter compiled code",
        );
    }

    #[test]
    fn jit_merge_point_only_drives_active_trace() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 123u64;
        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };

        assert!(
            driver
                .jit_merge_point_keyed(
                    key,
                    7,
                    &mut state,
                    &(),
                    || {},
                    |_meta, _sym| { TraceAction::Continue }
                )
                .is_none()
        );
        assert!(
            !driver.is_tracing(),
            "jit_merge_point must not warm up or start tracing by itself"
        );

        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(!driver.is_tracing(), "first back-edge should only warm up");

        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "back-edge should start tracing once hot"
        );
    }

    #[test]
    fn maybe_start_tracing_bumps_its_entry_counter_on_the_live_path() {
        // Slots 62/63 count `commit_start_tracing`'s two refusals and slot 61 is
        // their denominator, so a refusal reading 0 only means "did not fire"
        // once 61 is known to move.  Pin the bump to the live bound_reached
        // path, not to the counter's own definition: an unreachable slot and a
        // correctly-quiet one both read 0.  `MC_DIAG` is process-global and
        // shared with every other test in this binary, so a concurrent bump can
        // only inflate the delta — the assertion below is a lower bound.
        //
        // Cold edges (`warmstate.py`) tick and return without entering
        // this door, so only the overflow that starts tracing is counted.
        let before = crate::mc_diag(61);
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 4242u64;
        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };
        // Threshold 2: the first back edge warms up, the second starts tracing.
        for _ in 0..2 {
            assert!(
                driver
                    .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                    .is_none()
            );
        }
        assert!(
            driver.is_tracing(),
            "drive must reach the StartTracing arm so that neither refusal fires"
        );
        assert!(
            crate::mc_diag(61) >= before + 1,
            "mst_entered did not move on the back edge that started tracing"
        );
    }

    #[test]
    fn cold_back_edges_do_not_build_frontend_state() {
        let mut driver = JitDriver::<CountingDoorState>::new(10);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 4299u64;
        let mut state = CountingDoorState::default();

        for _ in 0..4 {
            assert!(
                driver
                    .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                    .is_none()
            );
        }

        assert!(!driver.is_tracing());
        assert_eq!(state.build_meta_calls.get(), 0);
        assert_eq!(state.extract_live_values_calls.get(), 0);
    }

    #[test]
    fn ceiling_latched_back_edge_skips_frontend_state() {
        use crate::warmstate::MAX_TRACE_ABORT_COUNT;

        let mut driver = JitDriver::<CountingDoorState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 4343u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::StartedTracing
        ));
        driver.meta.abort_trace(false);
        for _ in 1..MAX_TRACE_ABORT_COUNT {
            driver
                .meta
                .warm_state_for_driver(0)
                .abort_tracing(key, false);
        }
        assert!(driver.meta.warm_state_for_driver(0).is_ceiling_latched(key));

        let mut state = CountingDoorState::default();
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert_eq!(state.build_meta_calls.get(), 0);
        assert_eq!(state.extract_live_values_calls.get(), 0);
    }

    #[test]
    fn chained_ceiling_latch_does_not_refuse_the_typed_sibling() {
        use crate::warmstate::MAX_TRACE_ABORT_COUNT;

        let mut driver = JitDriver::<CountingDoorState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let target_pc = 7usize;
        let code_ptr = 0x1234usize;
        let key = GreenKey::with_types(
            vec![target_pc as i64, 0, code_ptr as i64],
            vec![Type::Int, Type::Int, Type::Ref],
        );
        let green_key = key.get_uhash();

        // The key's own cell reaches its ceiling.
        driver
            .meta
            .warm_state_for_driver(0)
            .disable_noninlinable_function_for_key(&key);
        for _ in 0..MAX_TRACE_ABORT_COUNT {
            driver
                .meta
                .warm_state_for_driver(0)
                .abort_tracing_for_key(&key, false);
        }
        assert_eq!(
            driver.meta.warm_state_for_driver(0).get_stats().num_cells,
            1,
            "one green key, one cell",
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .is_dont_trace_here(green_key)
        );
        let typed = driver
            .meta
            .warm_state_for_driver(0)
            .lookup_chain_with_key(&key)
            .expect("the key's cell");
        assert!(
            typed
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_DONT_TRACE_HERE),
            "dont-trace lives on the one cell this green key owns",
        );

        let mut state = CountingDoorState {
            code_ptr,
            ..Default::default()
        };
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(green_key, target_pc, &mut state, &(), || {},)
                .is_none()
        );
    }

    #[test]
    fn a_colliding_green_key_in_a_single_cell_bucket_still_ticks_and_traces() {
        // `_get_index` keeps the high bits of the low 32. Bit 0 never
        // changes that index, so these two hashes share a table slot and
        // differ everywhere the cell's own bucket is compared.
        let occupant = 0x11u64 << 21;
        let colliding = occupant | 1;
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        assert_eq!(
            driver.meta.jitcounter._get_index(occupant),
            driver.meta.jitcounter._get_index(colliding),
            "fixture hashes must share one celltable slot",
        );
        driver
            .meta
            .warm_state_for_driver(0)
            .disable_noninlinable_function(occupant);
        let cell = driver
            .meta
            .warm_state_for_driver(0)
            .lookup_chain(colliding)
            .expect("the colliding hash must see the occupied slot");
        assert!(
            cell.next.borrow().is_none(),
            "the bucket must be a single cell"
        );
        assert_eq!(cell.cell_bucket.get(), occupant);
        assert_ne!(cell.cell_bucket.get(), colliding);
        assert!(
            cell.flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_DONT_TRACE_HERE)
        );

        let increment = driver
            .meta
            .warm_state_for_driver(0)
            .counter
            .compute_threshold(2);
        assert!(
            !driver
                .meta
                .warm_state_for_driver(0)
                .counter
                .would_tick_fire(colliding, increment),
            "the colliding hash has not been counted yet",
        );
        let mut state = CountingDoorState::default();
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(colliding, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(!driver.is_tracing());
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .counter
                .would_tick_fire(colliding, increment),
            "a stranger in the slot must not swallow this hash's tick",
        );
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(colliding, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "the colliding hash must trace once its own counter fires",
        );
    }

    #[test]
    fn a_tokenless_dont_trace_here_cell_retries_from_the_back_edge() {
        // `disable_noninlinable_function` installs `JC_DONT_TRACE_HERE`
        // with no procedure token. `should_start_dont_trace_here_trace`
        // starts that cell immediately; a later `JC_TRACING_OCCURRED`
        // retries by ticking. Threshold 100 would stay cold if the back
        // edge only counted.
        let mut driver = JitDriver::<CountingDoorState>::new(100);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 0xD017_u64;
        driver
            .meta
            .warm_state_for_driver(0)
            .disable_noninlinable_function(key);
        let cell = driver
            .meta
            .warm_state_for_driver(0)
            .lookup_chain(key)
            .expect("disable_noninlinable_function installs a cell");
        assert!(cell.next.borrow().is_none());
        assert_eq!(cell.cell_bucket.get(), key);
        assert!(
            cell.flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_DONT_TRACE_HERE)
        );
        assert!(!cell.has_seen_a_procedure_token());

        let mut state = CountingDoorState::default();
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "a tokenless JC_DONT_TRACE_HERE cell must trace on the first back edge",
        );
    }

    #[test]
    fn function_entry_structured_arms_tracing_on_the_function_threshold() {
        // warmspot.py ll_portal_runner: maybe_compile_and_run uses
        // increment_function_threshold, not the back-edge increment.
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver
            .meta
            .warm_state_for_driver(0)
            .set_function_threshold(2);
        let hash = 0xA11u64;
        let make_key = || GreenKey::with_types(vec![hash as i64], vec![GreenType::Int]);
        let mut state = CountingDoorState {
            build_meta_calls: std::cell::Cell::new(0),
            extract_live_values_calls: std::cell::Cell::new(0),
            code_ptr: 0,
        };
        assert!(
            driver
                .function_entry_structured(hash, make_key, 0, &mut state, &())
                .is_none()
        );
        assert!(
            !driver.is_tracing(),
            "a single entry must not arm tracing below the function threshold"
        );
        assert!(
            driver
                .function_entry_structured(hash, make_key, 0, &mut state, &())
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "the function-entry door must arm tracing when the function threshold fires"
        );
    }

    #[test]
    fn compiled_function_token_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run with increment_function_threshold
        // is bound on that driver's WarmEnterState.
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e14_u64;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(74));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));

        let found = driver.compiled_function_token(key);
        assert!(
            found.is_some_and(
                |(cell_key, got)| cell_key == key && std::sync::Arc::ptr_eq(&got, &token)
            ),
            "function-entry compiled lookup on jd1 reads that driver's cell"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .get_compiled(key)
                .is_none(),
            "portal slot 0 does not hold jd1's compiled cell"
        );
    }

    #[test]
    fn resume_into_compiled_loop_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run / cell.get_procedure_token()
        // is bound on that driver's WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e19_u64;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(79));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));

        assert!(
            driver.has_compiled_loop(key),
            "compiled-loop resume lookup on jd1 reads that driver's cell"
        );
        assert!(
            driver
                .meta
                .entry_procedure_token_on_driver(idx1, key)
                .is_some_and(|got| std::sync::Arc::ptr_eq(&got, &token)),
            "entry token on jd1 is the attached compiled token"
        );
        assert!(
            driver.meta.entry_procedure_token(key).is_none(),
            "portal slot 0 does not hold jd1's compiled cell"
        );
    }

    #[test]
    fn has_compiled_loop_for_key_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run / cell.get_procedure_token()
        // via comparekey is bound on that driver's WarmEnterState
        // (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = GreenKey::new(vec![0x7e1a]);
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(80));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp_for_key(&key, std::sync::Arc::clone(&token));

        assert!(
            driver.has_compiled_loop_for_key(&key),
            "typed compiled-loop lookup on jd1 reads that driver's cell"
        );
        assert!(
            !driver.meta.has_compiled_loop_for_key(&key),
            "portal slot 0 does not hold jd1's compiled cell"
        );
    }

    #[test]
    fn resolve_cell_key_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run matches greens on that driver's
        // WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = GreenKey::with_types(vec![0x7e16], vec![GreenType::Int]);
        let hash = key.get_uhash();
        // Occupy the raw hash so the typed cell mints a distinct identity.
        // A miss on portal 0 also returns `hash`; the minted key is what
        // proves the shipped resolve walked jd1.
        driver
            .meta
            .warm_state_for_driver(idx1)
            .disable_noninlinable_function(hash);
        let cell_key = driver
            .meta
            .warm_state_for_driver(idx1)
            .ensure_cell_key(&key);
        assert_ne!(cell_key, hash, "typed cell on jd1 must mint a distinct key");

        let found = driver.resolve_cell_key(hash, || key.clone());
        assert_eq!(
            found, cell_key,
            "function-entry resolve on jd1 reads that driver's cell"
        );
        let make = || key.clone();
        let portal = driver.meta.resolve_cell_key(hash, Some(&make));
        assert_eq!(
            portal, hash,
            "portal slot 0 does not see jd1's cell and returns the unmatched hash"
        );
    }

    #[test]
    fn invalidate_loop_reads_the_entry_driver() {
        // cpu.invalidate_loop runs on the token filed on that driver's
        // WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e17_u64;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(77));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));

        driver.invalidate_loop(key);
        assert!(
            token.is_invalidated(),
            "invalidate_loop on jd1 hits that driver's token"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .get_procedure_token(key)
                .is_none(),
            "portal slot 0 has no token"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .cell_by_key(key)
                .is_none(),
            "portal slot 0 has no cell to invalidate"
        );
    }

    #[test]
    fn back_edge_incompatible_invalidates_the_entry_driver() {
        // cpu.invalidate_loop runs on the token filed on that driver's
        // WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<IncompatibleDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e18_u64;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(78));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));
        driver.insert_compiled_loop(
            key,
            crate::pyjitpl::CompiledEntry {
                token: std::sync::Arc::downgrade(&token),
                meta: std::sync::Arc::new(()),
                front_target_tokens: Vec::new(),
                front_entry_index: None,
                front_target_source_positions: None,
                root_trace_id: 1,
                traces: crate::FxIndexMap::default(),
                previous_tokens: Vec::new(),
                loop_header_pc: None,
                next_global_opref: 0,
            },
        );

        let mut state = IncompatibleDoorState;
        let result = driver.back_edge_resolved(
            key,
            std::sync::Arc::clone(&token),
            0,
            &mut state,
            &(),
            || {},
        );
        assert!(
            result.is_none(),
            "incompatible compiled entry declines the run"
        );
        assert!(
            token.is_invalidated(),
            "incompatible compiled entry on jd1 invalidates that driver's token"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .get_procedure_token(key)
                .is_none(),
            "portal slot 0 has no token"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .cell_by_key(key)
                .is_none(),
            "portal slot 0 has no cell to invalidate"
        );
    }

    #[test]
    fn runnable_procedure_token_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run is bound on that driver's
        // WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e15_u64;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(76));
        token.set_compiled(Box::new(()));
        driver
            .meta
            .warm_state_for_driver(idx1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));
        driver.insert_compiled_loop(
            key,
            crate::pyjitpl::CompiledEntry {
                token: std::sync::Arc::downgrade(&token),
                meta: std::sync::Arc::new(()),
                front_target_tokens: Vec::new(),
                front_entry_index: None,
                front_target_source_positions: None,
                root_trace_id: 1,
                traces: crate::FxIndexMap::default(),
                previous_tokens: Vec::new(),
                loop_header_pc: None,
                next_global_opref: 0,
            },
        );

        let found = driver.runnable_procedure_token(key);
        assert!(
            found.is_some_and(|got| std::sync::Arc::ptr_eq(&got, &token)),
            "runnable_procedure_token on jd1 reads that driver's cell"
        );
        assert!(
            driver.meta.entry_procedure_token(key).is_none(),
            "portal slot 0 does not hold jd1's compiled cell"
        );
    }

    #[test]
    fn get_compiled_meta_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run / JitCell.get_procedure_token
        // stay on jitdriver_sd.warmstate; the compiled_loops row for that
        // cell uses the executing driver's index (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0x7e1b_u64;
        let meta0 = Arc::new(());
        let meta1 = Arc::new(());
        let entry = |meta: Arc<()>, root_trace_id: u64| crate::pyjitpl::CompiledEntry {
            token: std::sync::Weak::new(),
            meta,
            front_target_tokens: Vec::new(),
            front_entry_index: None,
            front_target_source_positions: None,
            root_trace_id,
            traces: crate::FxIndexMap::default(),
            previous_tokens: Vec::new(),
            loop_header_pc: None,
            next_global_opref: 0,
        };
        driver
            .meta
            .insert_compiled_loop_on_driver(idx0, key, entry(Arc::clone(&meta0), 1));
        driver
            .meta
            .insert_compiled_loop_on_driver(idx1, key, entry(Arc::clone(&meta1), 2));
        driver.meta.active_jitdriver_sd = Some(idx0);
        assert!(
            driver
                .get_compiled_meta(key)
                .is_some_and(|got| Arc::ptr_eq(got, &meta1)),
            "get_compiled_meta on jd1 returns that driver's compiled_loops meta"
        );
    }

    #[test]
    fn back_edge_empty_chain_ticks_the_entry_driver() {
        // warmstate.py maybe_compile_and_run `:465-469` ticks
        // jitdriver_sd.warmstate's counter.
        let mut driver = JitDriver::<CountingDoorState>::new(100);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver.meta.warm_state_for_driver(idx1).set_threshold(1);

        let hash = 0xBE11_u64;
        let state = CountingDoorState::default();
        assert!(
            matches!(driver.back_edge_warmth(hash, &state), BackEdgeWarmth::Trace),
            "empty-chain tick on jd1 fires that driver's threshold"
        );
        let increment = driver
            .meta
            .warm_state_for_driver(0)
            .counter
            .compute_threshold(100);
        assert!(
            !driver
                .meta
                .warm_state_for_driver(0)
                .counter
                .would_tick_fire(hash, increment),
            "portal slot 0 must not consume the back-edge tick"
        );
    }

    #[test]
    fn abort_entry_tracing_clears_the_entry_driver() {
        // warmstate.py bound_reached finally clears JC_TRACING on
        // that driver's cell, not leftover portal slot 0.
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0xAB07_u64;
        driver
            .meta
            .warm_state_for_driver(0)
            .mark_as_being_traced(key);
        driver
            .meta
            .warm_state_for_driver(idx1)
            .mark_as_being_traced(key);
        driver.abort_entry_tracing(key);

        let extra = driver
            .meta
            .warm_state_ref_for_driver(idx1)
            .expect("jd1 warmstate");
        let extra_cell = extra.cell_by_key(key).expect("jd1 cell");
        assert!(
            !extra_cell
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_TRACING),
            "bound_reached finally clears JC_TRACING on that driver's cell"
        );
        assert!(
            !extra_cell
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_DONT_TRACE_HERE),
            "finally does not set JC_DONT_TRACE_HERE"
        );
        drop(extra);
        let primary_cell = driver
            .meta
            .warm_state_for_driver(0)
            .cell_by_key(key)
            .expect("jd0 cell");
        assert!(
            primary_cell
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_TRACING),
            "portal slot 0 keeps its own tracing cell"
        );
    }

    #[test]
    fn cell_is_tracing_reads_the_entry_driver() {
        // warmstate.py cell.flags & JC_TRACING is bound on that driver's
        // WarmEnterState (`JitDriver::index`).
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(driver.index(), Some(1));
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let key = 0xAB08_u64;
        driver
            .meta
            .warm_state_for_driver(idx1)
            .mark_as_being_traced(key);
        assert!(
            driver.cell_is_tracing(key),
            "cell_is_tracing on jd1 reads that driver's JC_TRACING"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .cell_by_key(key)
                .is_none_or(|cell| !cell
                    .flags
                    .get()
                    .contains(crate::warmstate::JcFlags::JC_TRACING)),
            "portal slot 0 does not hold jd1's tracing cell"
        );
    }

    #[test]
    fn set_param_writes_the_entry_driver() {
        // rlib/jit.py set_param(driver, name, value) writes that
        // driver's WarmEnterState, not leftover portal slot 0.
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let mut first = JitDriverStaticData::new(vec![], vec![]);
        first.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx0 = driver.register_descriptor(first);
        let mut second = JitDriverStaticData::new(vec![], vec![]);
        second.portal_runner_adr = portal_runner_helper as *const () as i64;
        let idx1 = driver.register_descriptor(second);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let primary_before = driver.meta.warm_state_for_driver(0).threshold();
        driver.set_param("threshold", 7);
        driver.set_param_enable_opts("intbounds");
        assert_eq!(
            driver.meta.warm_state_for_driver(idx1).threshold(),
            7,
            "set_param on jd1 writes that driver's threshold"
        );
        assert_eq!(
            driver.meta.warm_state_for_driver(0).threshold(),
            primary_before,
            "portal slot 0 keeps its own threshold"
        );
        assert_eq!(
            driver
                .meta
                .warm_state_ref_for_driver(idx1)
                .expect("jd1 warmstate")
                .get_enable_opts(),
            &["intbounds".to_string()][..],
            "set_param_enable_opts on jd1 writes that driver's enable_opts"
        );
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .get_enable_opts()
                .iter()
                .any(|opt| opt == "unroll"),
            "portal slot 0 keeps the default unroll pass"
        );
        driver.set_max_unroll_recursion(3);
        assert_eq!(
            driver
                .meta
                .warm_state_for_driver(idx1)
                .max_unroll_recursion(),
            3,
            "set_max_unroll_recursion on jd1 writes warmrunnerdesc.memory_manager"
        );
        assert_eq!(
            driver.meta.warm_state_for_driver(0).max_unroll_recursion(),
            3,
            "every WarmEnterState reads the one warmrunnerdesc.memory_manager"
        );
        driver.set_threshold(17);
        driver.set_trace_eagerness(4);
        assert_eq!(
            driver.meta.warm_state_for_driver(idx1).threshold(),
            17,
            "set_threshold on jd1 writes that driver's WarmEnterState"
        );
        assert_eq!(
            driver.meta.warm_state_for_driver(idx1).trace_eagerness(),
            4,
            "set_trace_eagerness on jd1 writes that driver's WarmEnterState"
        );
    }

    #[test]
    fn suppress_function_entry_skips_the_next_door_only() {
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver
            .meta
            .warm_state_for_driver(0)
            .set_function_threshold(1);
        let hash = 0xB0Du64;
        let make_key = || GreenKey::with_types(vec![hash as i64], vec![GreenType::Int]);
        let mut state = CountingDoorState {
            build_meta_calls: std::cell::Cell::new(0),
            extract_live_values_calls: std::cell::Cell::new(0),
            code_ptr: 0,
        };
        driver.suppress_function_entry();
        assert!(
            driver
                .function_entry_structured(hash, make_key, 0, &mut state, &())
                .is_none()
        );
        assert!(
            !driver.is_tracing(),
            "a suppressed door must not arm tracing"
        );
        assert!(
            driver
                .function_entry_structured(hash, make_key, 0, &mut state, &())
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "suppress is one-shot: the next door must run"
        );
    }

    #[test]
    fn run_compiled_detailed_keyed_uses_typed_live_inputs() {
        let mut driver = JitDriver::<TypedInputState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 11u64;
        // Input 0 is Int (used as GuardFalse condition); inputs 1/2 are
        // Ref/Float so the typed restore path still exercises mixed types.
        // GUARD_TRUE/FALSE in RPython are typed 'i' (resoperation.py
        // arglist_types) — the guard condition must be Int.
        let typed_live_values = vec![Value::Int(1), Value::Ref(GcRef(0x1234)), Value::Float(3.5)];

        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));

        // Input 0 is Int(1) — nonzero so GuardFalse fails at runtime.
        // Optimizer cannot constant-fold an input arg.
        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let i0 = OpRef::input_arg_int(0);
            let r1 = OpRef::input_arg_ref(1);
            let f2 = OpRef::input_arg_float(2);
            let g = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0, r1, f2], 0, 0);
            ctx.set_fail_args(g, &[i0, r1, f2]);
        };
        driver.meta.compile_loop(
            &[
                OpRef::input_arg_int(0),
                OpRef::input_arg_ref(1),
                OpRef::input_arg_float(2),
            ],
            (),
        );
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedInputState {
            raw_live_values: vec![1, 0, 0],
            typed_live_values: typed_live_values.clone(),
            ..Default::default()
        };
        match driver.run_compiled_detailed_keyed(key, &mut state, || {}) {
            DetailedDriverRunOutcome::Jump { via_blackhole, .. } => {
                assert!(!via_blackhole);
            }
            DetailedDriverRunOutcome::Finished { .. } => {
                panic!("expected Jump outcome, got Finished")
            }
            DetailedDriverRunOutcome::GuardFailure { .. } => {
                panic!("expected Jump outcome, got GuardFailure")
            }
            DetailedDriverRunOutcome::Abort { .. } => panic!("expected Jump outcome, got Abort"),
        }
        assert_eq!(state.restored_values, typed_live_values);
        assert_eq!(state.raw_restore_calls, 0);
        assert_eq!(state.typed_restore_calls, 1);
    }

    #[test]
    fn run_compiled_detailed_keyed_reads_the_entry_driver() {
        // warmstate.py maybe_compile_and_run / execute_assembler
        // run the cell on jitdriver_sd.warmstate (`JitDriver::index()`),
        // not leftover active_jitdriver_sd / portal 0.
        let mut driver = JitDriver::<TypedInputState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 12u64;
        let typed_live_values = vec![Value::Int(1), Value::Ref(GcRef(0x1234)), Value::Float(3.5)];
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let i0 = OpRef::input_arg_int(0);
            let r1 = OpRef::input_arg_ref(1);
            let f2 = OpRef::input_arg_float(2);
            let g = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0, r1, f2], 0, 0);
            ctx.set_fail_args(g, &[i0, r1, f2]);
        }
        driver.meta.compile_loop(
            &[
                OpRef::input_arg_int(0),
                OpRef::input_arg_ref(1),
                OpRef::input_arg_float(2),
            ],
            (),
        );
        let token = driver
            .meta
            .get_procedure_token_on_driver(0, key)
            .expect("compiled token on the tracing driver");
        let entry = driver
            .meta
            .take_compiled_loop(key)
            .expect("compiled_loops row on the tracing driver");
        driver.meta.insert_compiled_loop_on_driver(1, key, entry);
        driver
            .meta
            .warm_state_for_driver(1)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(1)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));
        driver.meta.active_jitdriver_sd = Some(0);
        assert!(
            driver
                .meta
                .run_compiled_detailed_with_values(key, &typed_live_values)
                .is_none(),
            "portal 0 wrapper must miss jd1's compiled_loops row"
        );
        let result = driver
            .meta
            .run_compiled_detailed_with_values_on_driver(1, key, &typed_live_values, 0)
            .expect("execute_assembler reads jitdriver_sd.warmstate for jd1");
        assert!(!result.is_finish);
    }

    #[test]
    fn single_pass_label_entry_handoff_expires_on_first_unrelated_back_edge() {
        let mut driver = JitDriver::<TypedRestoreState>::new(100);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let loop_a = 11u64;
        let loop_b = 22u64;
        let mut state = TypedRestoreState::default();

        // Models pyjitpl.py raise_continue_running_normally: the
        // LABEL handoff is only for the next back-edge after the parity resume.
        // If loop B is that next back-edge, loop A's handoff is stale and must
        // not survive until a later fresh loop-A entry.
        assert!(
            driver
                .back_edge_keyed(loop_b, 22, &mut state, &(), || {})
                .is_none()
        );
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(loop_b, 22, &mut state, &(), || {})
                .is_none()
        );
        let _ = loop_a;
    }

    #[test]
    fn test_threshold_1_fires_on_first_tick() {
        // counter.py: compute_threshold(1) = 1.0/(1-0.001) ≈ 1.001
        // First tick adds ≈1.001 to 0.0, reaching ≥1.0 immediately.
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 500u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::StartedTracing
        ));
    }

    /// `declare_flat_entry_contract` fills in the two things a flat entry
    /// cannot say through the red list, on the descriptor the schema call just
    /// built — and re-answers `virtualizable_arg_index` in that model, so the
    /// consumers that read it as a position in the entry arglist see the
    /// virtualizable's slot rather than its position among the reds.
    #[test]
    fn declare_flat_entry_contract_states_the_entry_shape_the_reds_cannot() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.declare_schema_typed(
            vec![("pc", GreenType::Int), ("program", GreenType::Ref)],
            vec![("state", Type::Ref)],
        );
        let before = driver
            .descriptor
            .as_ref()
            .expect("schema builds a descriptor");
        assert_eq!(before.flat_entry_contract(), None);
        assert_eq!(before.virtualizable_arg_index(), None);

        driver.declare_flat_entry_contract(
            "state",
            FlatEntryContract {
                len: 2,
                index_of_virtualizable: 1,
            },
        );
        let after = driver.descriptor.as_ref().expect("descriptor survives");
        assert_eq!(after.virtualizable.as_deref(), Some("state"));
        assert_eq!(after.flat_entry_contract().map(|f| f.len), Some(2));
        assert_eq!(after.virtualizable_arg_index(), Some(1));
        assert_eq!(after.index_of_virtualizable, 1);
        // The reds still describe the merge-point payload, unchanged.
        assert_eq!(after.num_reds(), 1);
    }

    /// Build a driver whose virtualizable holds one array.
    fn driver_with_one_vable_array() -> JitDriver<TypedRestoreState> {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.declare_schema_typed(vec![("pc", GreenType::Int)], vec![("state", Type::Ref)]);
        let mut info = crate::virtualizable::VirtualizableInfo::without_vable_token();
        info.name = "state".to_string();
        info.add_array_field(
            "regs",
            Type::Int,
            0,
            0,
            8,
            majit_ir::descr::make_array_descr(8, 8, Type::Int),
        );
        driver
            .meta
            .set_virtualizable_info(info.finalize_arc(majit_ir::descr::make_size_descr(16)));
        driver
    }

    /// A virtualizable whose arrays the entry preamble can reload gets armed:
    /// `warmspot.py:520-545` names it on the static data, which is what makes
    /// `compile.py` run.
    #[test]
    fn arming_declares_the_contract_for_a_reloadable_virtualizable() {
        let mut driver = driver_with_one_vable_array();
        assert!(driver.arm_flat_entry_contract(FlatEntryContract {
            len: 2,
            index_of_virtualizable: 1,
        }));
        assert_eq!(
            driver.flat_entry_contract(),
            Some(FlatEntryContract {
                len: 2,
                index_of_virtualizable: 1,
            })
        );
        // The name comes from the vinfo the gate was answered about, so the
        // descriptor cannot end up naming a different virtualizable.
        assert_eq!(
            driver.descriptor.as_ref().unwrap().virtualizable.as_deref(),
            Some("state")
        );
    }

    /// With no virtualizable there is nothing for the preamble to reload from,
    /// so there is no contract to declare either.
    #[test]
    fn arming_declines_a_driver_with_no_virtualizable_info() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.declare_schema_typed(vec![("pc", GreenType::Int)], vec![("state", Type::Ref)]);
        assert!(!driver.arm_flat_entry_contract(FlatEntryContract {
            len: 2,
            index_of_virtualizable: 1,
        }));
        assert_eq!(driver.flat_entry_contract(), None);
    }

    #[test]
    fn test_set_param_threshold() {
        let mut driver = JitDriver::<TypedRestoreState>::new(10);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        // Initially threshold is 10 — not hot after 2 ticks.
        let key = 100u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));

        // Lower threshold to 2 via set_param — next tick should start tracing.
        // (threshold=1 fires on first tick: increment=1/0.999≈1.001 ≥ 1.0)
        driver.set_param("threshold", 2);
        let key2 = 200u64;
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::StartedTracing
        ));
    }

    #[test]
    fn test_get_stats_after_compile() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 300u64;

        // Stats should be zero initially.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.loops_aborted, 0);

        // Warm up and start tracing. Pass a single live value so that
        // setup_tracing (pyjitpl.rs) registers one inputarg via
        // recorder.record_input_arg — this gives the trace a valid
        // `OpRef::input_arg_int(0)` inputarg that the GuardTrue below references.
        // Without the inputarg, TraceIterator (opencoder.py)
        // correctly rejects a TAGBOX reference to an unseeded cache
        // slot (`_get(i)` asserts `res is not None`).
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0i64]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0i64]),
            BackEdgeAction::StartedTracing
        ));

        // Record minimal trace and compile.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            ctx.const_int(42);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0], 0, 0);
            ctx.set_fail_args(g, &[i0]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert!(driver.has_compiled_loop(key));

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.loops_aborted, 0);
        assert_eq!(stats.bridges_compiled, 0);
    }

    /// State whose walk advances a scalar state field the recover hook cannot
    /// re-derive (a `selected`-style storage index). The Sym carries the
    /// scalar; `collect_scalar_state_field_values` reads it (at close time,
    /// while the sym is live) and `writeback_scalar_state_fields_from_values`
    /// pushes the captured value into native `state`.
    #[derive(Default)]
    struct ScalarWalkState {
        // Native scalar; frozen at trace-start until the close-time write-back.
        selected: i64,
        // Set by the write-back so the test can observe the applied value.
        writeback_applied: Option<Vec<i64>>,
    }

    #[derive(Default)]
    struct ScalarWalkSym {
        // The walk mutates this in the merge-point closure (a
        // BC_STORE_STATE_FIELD analog); native `state.selected` stays frozen.
        selected: i64,
    }

    impl JitState for ScalarWalkState {
        type Meta = ();
        type Sym = ScalarWalkSym;
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            // One inputarg so the trace records a valid input.
            vec![self.selected]
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
            ScalarWalkSym::default()
        }

        fn initialize_sym(&self, sym: &mut Self::Sym, _meta: &Self::Meta) {
            // Seed the sym's scalar mirror from the trace-start native value.
            sym.selected = self.selected;
        }

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        // The bug: this must observe the walk-final scalar captured while the
        // sym was still live, NOT the trace-start value and NOT a no-op.
        fn collect_scalar_state_field_values(sym: &Self::Sym) -> Vec<i64> {
            vec![sym.selected]
        }

        fn writeback_scalar_state_fields_from_values(&mut self, values: &[i64]) {
            self.selected = values[0];
            self.writeback_applied = Some(values.to_vec());
        }

        // Force the abort branch of the CloseLoop arm — no backend compile is
        // needed to exercise the capture-before-clear / apply-after-clear
        // ordering, which is where the bug lived.
        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            false
        }
    }

    /// `validate_close` accepts, so `CloseLoopWithArgs` reaches
    /// `compile_and_record_loop`.
    #[derive(Default)]
    struct AcceptCloseState {
        selected: i64,
    }

    #[derive(Default)]
    struct AcceptCloseSym {
        selected: i64,
    }

    impl JitState for AcceptCloseState {
        type Meta = ();
        type Sym = AcceptCloseSym;
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            vec![self.selected]
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
            AcceptCloseSym::default()
        }

        fn initialize_sym(&self, sym: &mut Self::Sym, _meta: &Self::Meta) {
            sym.selected = self.selected;
        }

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            vec![OpRef::input_arg_int(0)]
        }

        fn collect_scalar_state_field_values(_sym: &Self::Sym) -> Vec<i64> {
            Vec::new()
        }

        fn writeback_scalar_state_fields_from_values(&mut self, _values: &[i64]) {}

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    /// `reached_loop_header` keeps tracing after `compile_retrace` returns
    /// None under the unroll budget. The close arrived as
    /// `CloseLoopWithArgs`; the driver must consume
    /// `keep_tracing_after_close` and leave the walk running.
    #[test]
    fn close_loop_with_args_retrace_cancel_keeps_walk() {
        let mut driver = JitDriver::<AcceptCloseState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver
            .meta
            .warm_state_for_driver(0)
            .set_param("max_unroll_loops", 1);
        let mut state = AcceptCloseState::default();
        let key = 42u64;
        driver.force_start_tracing(key, 0, &mut state, &());
        assert!(driver.is_tracing());
        let start = driver.meta.trace_ctx().unwrap().current_merge_points[0].position;
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(9));
        token.set_compiled(Box::new(()));
        token.record_target_token(crate::history::TargetToken::new_loop(1).as_jump_target_descr());
        driver
            .meta
            .warm_state_for_driver(0)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(0)
            .attach_procedure_to_interp(key, std::sync::Arc::clone(&token));
        driver.meta.partial_trace = Some(crate::pyjitpl::PartialTrace {
            ops: Vec::new(),
            inputargs: Vec::new(),
        });
        driver.meta.retracing_from = Some(start);
        let calls = std::cell::Cell::new(0u32);
        driver.merge_point(|_meta, _sym| {
            let n = calls.get();
            calls.set(n + 1);
            if n > 0 {
                return TraceAction::Continue;
            }
            TraceAction::CloseLoopWithArgs {
                jump_args: vec![OpRef::input_arg_int(0)],
                loop_header_pc: None,
            }
        });
        assert_eq!(calls.get(), 2, "the close must re-enter the walk");
        assert!(driver.sym.is_some(), "the driver walk stays alive");
        assert!(driver.is_tracing());
        assert!(
            !driver.meta.take_keep_tracing_after_close(),
            "CloseLoopWithArgs must consume the latch"
        );
        assert!(driver.meta.trace_ctx().unwrap().take_merge_point_resumed());
    }

    /// Regression: the whole-circuit single-pass CloseLoop arm captures the
    /// walk-final scalar state fields off the still-live sym (into
    /// `single_pass_scalar_values`) BEFORE it clears `self.sym`, and
    /// `writeback_scalar_state_fields` applies that stash to native `state`
    /// afterward. Previously the write-back read `self.sym` directly, but the
    /// CloseLoop arm had already set `self.sym = None`, so the write-back was
    /// always a silent no-op — a program whose walk advanced a scalar (an
    /// instruction switching the selected storage, say) would seed native
    /// state from the stale trace-start value.
    #[test]
    fn single_pass_close_writes_back_walk_final_scalar() {
        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        // Trace-start native state: selected == 3.
        let mut state = ScalarWalkState {
            selected: 3,
            writeback_applied: None,
        };
        let target_pc = 7usize;
        let key = target_pc as u64;

        // Start a real trace: this seeds `driver.sym` via create_sym +
        // initialize_sym (sym.selected == 3) and installs the trace session.
        driver.force_start_tracing(key, target_pc, &mut state, &());
        assert!(
            driver.is_tracing(),
            "force_start_tracing must start a trace"
        );

        // Drive the merge point to a CloseLoop. The closure stands in for the
        // walk: it advances the scalar on the sym (selected 3 -> 9) and stamps
        // the walk-final pc so the single-pass outcome is published, then
        // closes the loop.
        driver.merge_point(|meta, sym| {
            sym.selected = 9;
            if let Some(ctx) = meta.trace_ctx() {
                ctx.walk_final_pc = Some(target_pc);
                // `reached_loop_header` stashes JUMP boxes; this fixture
                // has no merge-point reds.
                ctx.close_jump_boxes = Some(Vec::new());
            }
            TraceAction::CloseLoop
        });

        // merge_point cleared the sym as part of CloseLoop handling.
        assert!(
            driver.sym.is_none(),
            "CloseLoop handling must clear the sym before returning",
        );

        // Mimic the `jit_merge_point!(...; state)` macro hook: the walk-final
        // outcome is published, then the write-back applies the captured
        // walk-final scalar into native state.
        let outcome = driver.take_single_pass_outcome();
        assert!(
            outcome.is_some(),
            "CloseLoop must publish the single-pass outcome",
        );
        driver.writeback_scalar_state_fields(&mut state);

        assert_eq!(
            state.selected, 9,
            "native state must receive the walk-final scalar (9), not stay frozen at the trace-start value (3)",
        );
        assert_eq!(
            state.writeback_applied,
            Some(vec![9]),
            "write-back must run with the walk-final captured values, not be a no-op",
        );
    }

    /// Regression: an exception that drains the walk's frame stack must still
    /// be delivered after the FINISH is compiled.
    ///
    /// `dispatch.rs unwind_to_exception_handler` closes the
    /// `BH_LAST_EXC_VALUE` channel when it builds the exception-carrying
    /// FINISH (leaving it set makes an interpreter re-surface the trace's
    /// exception at its next residual-call boundary), and hands the concrete
    /// value over on the action instead.  `#[jit_interp]` clients reach that
    /// action through `merge_point` and have no second carrier, so this arm
    /// re-publishes it — without that, the portal finishes as if the frame had
    /// returned normally and the exception is silently dropped.
    #[test]
    fn finish_with_exception_republishes_the_escaping_exception() {
        const ESCAPING_EXC: i64 = 0xc0ff_ee00;

        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        let mut state = ScalarWalkState {
            selected: 3,
            writeback_applied: None,
        };
        let target_pc = 7usize;
        driver.force_start_tracing(target_pc as u64, target_pc, &mut state, &());
        assert!(
            driver.is_tracing(),
            "force_start_tracing must start a trace"
        );

        // The producer cleared the channel; only the action carries the value.
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));

        driver.merge_point(|meta, _sym| {
            let exc_box = meta
                .trace_ctx()
                .expect("tracing ⇒ trace ctx")
                .const_ref(ESCAPING_EXC);
            TraceAction::Finish {
                finish_args: vec![exc_box],
                finish_arg_types: vec![majit_ir::Type::Ref],
                exit_with_exception: true,
                exc_value: ESCAPING_EXC,
            }
        });

        let delivered = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
        assert_eq!(
            delivered, ESCAPING_EXC,
            "the exception the trace finished with must survive the compile half",
        );
    }

    #[test]
    fn cell_is_tracing_is_per_green_key() {
        const A: u64 = 11;
        const B: u64 = 22;
        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let mut state = ScalarWalkState::default();
        driver.force_start_tracing(A, A as usize, &mut state, &());
        assert!(driver.is_tracing());
        assert!(driver.cell_is_tracing(A));
        assert!(!driver.cell_is_tracing(B));
    }

    #[test]
    fn too_long_unwind_preserves_the_root_spared_by_the_raise_site() {
        const ROOT: u64 = 7;
        const CALLEE: u64 = 0xa11;
        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let mut state = ScalarWalkState::default();
        driver.force_start_tracing(ROOT, ROOT as usize, &mut state, &());
        assert!(driver.is_tracing());

        driver.merge_point(|meta, _sym| {
            let start = meta.trace_ctx().unwrap().get_trace_position();
            meta.push_portal_trace_position(0, Some((CALLEE, None)), start);
            let ctx = meta.tracing.as_mut().unwrap();
            ctx.record_op(
                majit_ir::OpCode::IntAdd,
                &[majit_ir::OpRef::const_int(0), majit_ir::OpRef::const_int(0)],
            );
            let end = ctx.get_trace_position();
            ctx.set_trace_limit(0);
            meta.push_portal_trace_position(0, None, end);
            let reason = meta.blackhole_if_trace_too_long().unwrap();
            assert_eq!(reason, AbortReason::TooLong);
            assert!(meta.portal_trace_positions.is_none());
            assert!(meta.take_pending_abort_reason().is_none());
            TraceAction::SwitchToBlackhole(crate::SwitchToBlackhole {
                reason: reason.as_int(),
                raising_exception: false,
            })
        });

        assert!(!driver.is_tracing());
        {
            let warmstate = driver.meta.warm_state_for_driver(0);
            assert!(!warmstate.can_inline_callable(CALLEE));
            assert!(warmstate.can_inline_callable(ROOT));
            assert!(!warmstate.should_force_finish_tracing(ROOT));
        }
        // A fresh attempt must remain possible after the callee is disabled.
        driver.force_start_tracing(ROOT, ROOT as usize, &mut state, &());
        assert!(driver.is_tracing());
        assert_eq!(
            driver.meta.portal_trace_positions.as_ref().map(Vec::len),
            Some(0)
        );
    }

    /// `pyjitpl.py aborted_tracing` does not set `JC_DONT_TRACE_HERE`.
    /// A BRIDGE's green key names the SOURCE LOOP — a loop that already
    /// compiled — and `can_inline_callable` (`warmstate.py`) reads that
    /// flag, so `abort_trace(true)` would additionally make the enclosing
    /// function non-inlinable everywhere. `handle_guard_failure` never
    /// reaches that flag: its unwind ends in
    /// `run_blackhole_interp_to_cancel_tracing` → `aborted_tracing(reason)`.
    #[test]
    fn an_abort_permanent_on_a_bridge_does_not_retire_the_source_loop() {
        #[derive(Debug)]
        struct BridgeOriginDescr;
        impl majit_ir::Descr for BridgeOriginDescr {}

        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let mut state = ScalarWalkState {
            selected: 3,
            writeback_applied: None,
        };

        let primary_pc = 7usize;
        let primary_key = primary_pc as u64;
        driver.force_start_tracing(primary_key, primary_pc, &mut state, &());
        assert!(driver.is_tracing(), "fixture: the primary trace started");
        driver.merge_point(|_meta, _sym| TraceAction::AbortPermanent);
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .can_inline_callable(primary_key),
            "a primary AbortPermanent must not stamp JC_DONT_TRACE_HERE",
        );

        let bridge_pc = 11usize;
        let bridge_key = bridge_pc as u64;
        driver.force_start_tracing(bridge_key, bridge_pc, &mut state, &());
        assert!(driver.is_tracing(), "fixture: the bridge trace started");
        driver
            .meta
            .set_bridge_trace_info(crate::pyjitpl::BridgeTraceInfo {
                green_key: bridge_key,
                trace_id: 1,
                fail_index: 0,
                code_ptr: 0,
                source_descr: std::sync::Arc::new(BridgeOriginDescr),
            });
        let before = crate::mc_diag(82);
        driver.merge_point(|_meta, _sym| TraceAction::AbortPermanent);
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .can_inline_callable(bridge_key),
            "an AbortPermanent reached from a guard retired the source loop",
        );
        // `MC_DIAG` is a process-wide static, so a concurrent test can only
        // inflate the delta.
        assert!(
            crate::mc_diag(82) > before,
            "the bridge arm must name itself in slot 82",
        );
    }

    /// `pyjitpl.py aborted_tracing` does not set `JC_DONT_TRACE_HERE`.
    /// `disable_noninlinable_function` is only the ABORT_TOO_LONG arm of
    /// `blackhole_if_trace_too_long`. A primary `AbortPermanent` is the
    /// same `SwitchToBlackhole` unwind: clear `JC_TRACING`, leave the
    /// cell inlinable.
    #[test]
    fn an_abort_permanent_on_a_primary_trace_clears_tracing_without_dont_trace_here() {
        let mut driver = JitDriver::<ScalarWalkState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let mut state = ScalarWalkState {
            selected: 3,
            writeback_applied: None,
        };

        let primary_pc = 7usize;
        let primary_key = primary_pc as u64;
        driver.force_start_tracing(primary_key, primary_pc, &mut state, &());
        assert!(driver.is_tracing(), "fixture: the primary trace started");
        assert!(
            driver
                .meta
                .warm_state_for_driver(0)
                .cell_by_key(primary_key)
                .is_some_and(|cell| cell.is_tracing()),
            "force_start_tracing marked JC_TRACING"
        );
        driver.merge_point(|_meta, _sym| TraceAction::AbortPermanent);
        assert!(!driver.is_tracing(), "AbortPermanent ended the session");
        let cell = driver
            .meta
            .warm_state_for_driver(0)
            .cell_by_key(primary_key)
            .expect("the aborted cell remains");
        assert!(
            !cell
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_TRACING),
            "aborted_tracing clears JC_TRACING"
        );
        assert!(
            !cell
                .flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_DONT_TRACE_HERE),
            "aborted_tracing does not set JC_DONT_TRACE_HERE"
        );
    }

    #[test]
    fn test_set_param_unknown_ignored() {
        let mut driver = JitDriver::<TypedRestoreState>::new(5);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        // Unknown params should not panic or cause any side effects.
        driver.set_param("nonexistent_param", 999);
        driver.set_param("", 0);
        driver.set_param("enable_opts", 1);

        // Driver should still work normally after unknown params.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
    }

    #[test]
    fn test_new_auto_registers_virtualizable_info_from_state() {
        #[derive(Default)]
        struct AutoVableState;

        impl JitState for AutoVableState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
            fn __build_virtualizable_info()
            -> Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>> {
                let mut info = crate::virtualizable::VirtualizableInfo::new(24);
                info.add_field("pc", Type::Int, 8);
                info.add_array_field(
                    "locals_w",
                    Type::Ref,
                    16,
                    0,
                    0,
                    majit_ir::make_array_descr(0, 8, Type::Ref),
                );
                Some(info.finalize_arc(majit_ir::descr::make_size_descr(64)))
            }
        }

        let driver = JitDriver::<AutoVableState>::new(5);
        let info = driver
            .meta_interp()
            .virtualizable_info()
            .expect("driver should auto-register __build_virtualizable_info");
        assert_eq!(info.token_offset, 24);
        assert_eq!(info.static_fields.len(), 1);
        assert_eq!(info.array_fields.len(), 1);
        assert_eq!(info.static_fields[0].name, "pc");
        assert_eq!(info.array_fields[0].name, "locals_w");
    }

    // ── Event-driven JitHookInterface parity tests ──
    // These exercise the full compilation pipeline (tracing → optimize → Cranelift)
    // and verify hooks fire with correct metadata, unlike wiring-only tests that
    // call hook closures directly.

    #[test]
    fn test_hook_on_compile_fires_through_real_pipeline() {
        use parking_lot::Mutex;
        use std::sync::Arc;

        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let compile_events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = compile_events.clone();
        driver.set_on_compile_loop(move |green_key, ops_before, ops_after, _opcodes| {
            events.lock().push((green_key, ops_before, ops_after));
        });

        let key = 42u64;

        // Warm up: first back_edge increments counter, second starts tracing.
        // Pass one live value so input_arg_int(0) = i0 is available.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        // Record a real trace: IntAdd + GuardTrue(input) + JUMP(sum)
        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0); // input arg from on_back_edge
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
            sum
        };

        // Close: JUMP args must match body inputargs after preamble peeling.
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        let events = compile_events.lock();
        assert_eq!(events.len(), 1, "on_compile_loop should fire exactly once");
        assert_eq!(events[0].0, key, "green_key should match");
        assert!(events[0].1 > 0, "num_ops_before should be positive");
        assert!(events[0].2 > 0, "num_ops_after should be positive");
    }

    /// Record a straight-line trace of `n_adds` `IntAdd`s closed by a guard.
    /// A longer trace ends at a higher `next_global_opref` high-water.
    /// `n_adds` must be at least 1 — an op-less trace compiles to `Cancelled`.
    fn record_trace_of(driver: &mut JitDriver<TypedRestoreState>, n_adds: usize) -> OpRef {
        let ctx = driver.meta.trace_ctx().expect("should be tracing");
        let i0 = OpRef::input_arg_int(0);
        let c1 = ctx.const_int(1);
        let mut sum = i0;
        for _ in 0..n_adds {
            sum = ctx.record_op(OpCode::IntAdd, &[sum, c1]);
        }
        let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
        ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
        ctx.set_fail_args(g, &[sum]);
        sum
    }

    /// Drive one compile at `key`: climb the back-edge counter until tracing
    /// starts, record `n_adds` adds, compile.
    fn compile_once_at(driver: &mut JitDriver<TypedRestoreState>, key: u64, n_adds: usize) {
        for _ in 0..10 {
            driver.meta.on_back_edge(key, &[0]);
            if driver.meta.trace_ctx().is_some() {
                break;
            }
        }
        assert!(
            driver.meta.trace_ctx().is_some(),
            "back edges did not start a trace at key={key}"
        );
        let sum = record_trace_of(driver, n_adds);
        driver.meta.compile_loop(&[sum], ());
    }

    /// Compile twice at one green key and hand back both entries' carried
    /// state.
    ///
    /// The second compile only reaches a `take_entry_for_replace` branch if
    /// the key is traceable *again* while its `compiled_loops` entry is still
    /// present. Both trace-start doors (`on_back_edge`, `force_start_tracing`)
    /// refuse a key whose cell `is_compiled()`, which is
    /// `get_procedure_token().is_some()` filtered on `!is_invalidated()`
    /// (`warmstate.rs`'s `BaseJitCell::is_compiled`). Invalidating the procedure token
    /// clears that predicate and touches no `compiled_loops` entry, so it
    /// re-opens tracing and leaves the entry to be displaced — the same
    /// desynchronisation quasi-immut invalidation produces in production.
    ///
    /// Returns `(first.next_global_opref, replacement.next_global_opref,
    /// replacement.loop_header_pc)`.
    fn compile_twice_at_one_green_key(
        key: u64,
        header_pc: usize,
        first_adds: usize,
        second_adds: usize,
    ) -> (u32, u32, Option<usize>) {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        compile_once_at(&mut driver, key, first_adds);
        let first_opref = driver
            .meta
            .compiled_entry(key)
            .expect("compiled entry")
            .next_global_opref;
        // `compile_loop` can only ever carry `loop_header_pc` forward, never
        // mint one, so stamp the first entry the way a closing bridge does.
        driver.meta.record_loop_header_pc(key, header_pc);

        driver.meta.warm_state_for_driver(0).invalidate_all();
        assert!(
            driver.meta.has_compiled_loop_entry(key),
            "invalidating the token must leave the entry to be displaced"
        );

        compile_once_at(&mut driver, key, second_adds);
        assert_eq!(
            driver.get_stats().loops_compiled,
            2,
            "the second compile did not happen"
        );
        let replacement = &driver.meta.compiled_entry(key).expect("compiled entry");
        (
            first_opref,
            replacement.next_global_opref,
            replacement.loop_header_pc,
        )
    }

    /// `compile_loop`'s replace branch must raise the fresh high-water to the
    /// displaced entry's, so OpRefs minted for a later bridge stay disjoint
    /// from those the retired loop's bridges already hold.
    #[test]
    fn a_replacing_compile_carries_the_displaced_entrys_next_global_opref() {
        // The `max` is only observable when the replacement's own trace ends
        // lower than the displaced entry's. Measure the short trace's
        // standalone high-water on this tree rather than pinning a constant.
        let mut probe = JitDriver::<TypedRestoreState>::new(2);
        probe.meta.finish_setup_descrs_for_jitdrivers();
        compile_once_at(&mut probe, 91, 1);
        let short_alone = probe
            .meta
            .compiled_entry(91)
            .expect("compiled entry")
            .next_global_opref;

        let (first, replacement, _) = compile_twice_at_one_green_key(77, 4242, 8, 1);
        assert!(
            short_alone < first,
            "fixture is not discriminating: a short trace alone reaches \
             {short_alone}, the long first compile reaches {first} — the \
             carry cannot be distinguished from a fresh computation"
        );
        assert_eq!(
            replacement, first,
            "the replacement entry must inherit the displaced entry's OpRef \
             high-water"
        );
    }

    /// `compile_loop`'s replace branch must carry the displaced entry's
    /// `loop_header_pc`, or every bridge aiming its closing JUMP at that
    /// header is stranded.
    #[test]
    fn a_replacing_compile_carries_the_displaced_entrys_loop_header_pc() {
        let (_, _, loop_header_pc) = compile_twice_at_one_green_key(78, 4242, 8, 1);
        assert_eq!(
            loop_header_pc,
            Some(4242),
            "the replacement entry must inherit the displaced entry's close \
             target"
        );
    }

    #[test]
    fn test_hook_get_stats_matches_real_compile_count() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        // Initially zero.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.bridges_compiled, 0);

        // Compile two distinct loops.
        for key in [100u64, 200u64] {
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::Interpret
            ));
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::StartedTracing
            ));
            {
                let ctx = driver.meta.trace_ctx().expect("should be tracing");
                let i0 = OpRef::input_arg_int(0);
                let c1 = ctx.const_int(1);
                let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
                let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
                ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
                ctx.set_fail_args(g, &[sum]);
            }
            driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
            assert!(driver.has_compiled_loop(key));
        }

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 2);
        assert_eq!(stats.loops_aborted, 0);
    }

    #[test]
    fn test_hook_on_compile_bridge_fires_through_real_pipeline() {
        use parking_lot::Mutex;
        use std::sync::Arc;

        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let bridge_events: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = bridge_events.clone();
        driver.meta.set_on_compile_bridge(move |gk, fi, nops| {
            ev.lock().push((gk, fi, nops));
        });

        let key = 50u64;

        // Step 1: Compile a loop with a guard on a non-constant condition.
        // The guard must survive optimization, so use an input arg as condition.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));
        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0); // input arg (non-constant = won't be folded)
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
            sum
        };
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        // Identify the fail_index assigned to the first guard in the optimized trace.
        // build_guard_metadata assigns fail_index sequentially starting from 0.
        let fail_index = 0u32;

        // Step 2: Start bridge tracing via start_retrace (simulates guard failure path).
        assert!(driver.meta.start_retrace(key, fail_index, &[0]));

        // Step 3: Record a bridge trace and compile it via close_bridge.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing bridge");
            let i0 = OpRef::input_arg_int(0); // bridge input from start_retrace
            let c2 = ctx.const_int(2);
            ctx.record_op(OpCode::IntAdd, &[i0, c2]);
        }
        // Use the actual root trace_id allocated at compile_loop time.
        // RPython has no `trace_id == 0 → root_trace_id` normalize step;
        // the descr identity carries the exact trace_id, and tests must
        // pass the same identity.
        let trace_id = driver
            .meta
            .compiled_root_trace_id(key)
            .expect("compiled loop should have a root trace_id");
        let result =
            driver
                .meta
                .close_bridge(key, trace_id, fail_index, &[OpRef::input_arg_int(0)]);
        assert_eq!(
            result,
            crate::pyjitpl::BridgeCompileResult::Compiled,
            "bridge should compile successfully"
        );

        let events = bridge_events.lock();
        assert_eq!(
            events.len(),
            1,
            "on_compile_bridge should fire exactly once"
        );
        assert_eq!(events[0].0, key, "bridge green_key should match");
        assert_eq!(events[0].1, fail_index, "bridge fail_index should match");
        assert!(events[0].2 > 0, "bridge num_ops should be positive");

        // Stats should reflect 1 loop + 1 bridge.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.bridges_compiled, 1);
    }

    /// A [`Census`](crate::embed::Census) reading must report the bridge while
    /// the driver that compiled it is STILL ALIVE.
    ///
    /// The census counters are process-global and the bridge tally is not: it
    /// lives on the driver. Reading it only out of drivers already handed to
    /// `Census::absorb` is correct for a one-shot driver and reports zero
    /// forever for a pool, which is the shape every real embedder has — and a
    /// zero there is indistinguishable from "no bridge was compiled". So this
    /// asserts the count with the driver held, and never absorbs it.
    ///
    /// It repeats the pipeline of the test above rather than sharing it,
    /// because what is under test is the OTHER half: that `Census::install`
    /// wires `set_on_compile_bridge` at all. A driver built by that test would
    /// pass this one with the hook missing.
    #[test]
    fn test_census_counts_a_bridge_from_a_live_driver() {
        use crate::embed::Census;

        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        Census::install(&mut driver);

        let census = Census::begin();
        assert_eq!(
            census.counts().bridges_compiled,
            0,
            "the window opens before anything has compiled"
        );

        let key = 51u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));
        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
            sum
        };
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        let fail_index = 0u32;
        assert!(driver.meta.start_retrace(key, fail_index, &[0]));
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing bridge");
            let i0 = OpRef::input_arg_int(0);
            let c2 = ctx.const_int(2);
            ctx.record_op(OpCode::IntAdd, &[i0, c2]);
        }
        let trace_id = driver
            .meta
            .compiled_root_trace_id(key)
            .expect("compiled loop should have a root trace_id");
        assert_eq!(
            driver
                .meta
                .close_bridge(key, trace_id, fail_index, &[OpRef::input_arg_int(0)]),
            crate::pyjitpl::BridgeCompileResult::Compiled,
        );

        let counts = census.counts();
        assert_eq!(
            counts.bridges_compiled, 1,
            "the driver is still alive and holds the only bridge tally: {counts}"
        );
        assert_eq!(
            counts.bridges_compiled,
            driver.get_stats().bridges_compiled,
            "the callback count and the driver's own tally are the same number"
        );
    }

    #[test]
    fn compiling_a_loop_moves_the_runnable_generation() {
        let mut driver = JitDriver::<NonTraceableState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let green_key = 405u64;
        let cold = driver.runnable_generation();
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[], 0, 0);
            ctx.set_fail_args(g, &[]);
        }
        let tracing = driver.runnable_generation();
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert!(driver.has_runnable_compiled_loop(green_key));
        let compiled = driver.runnable_generation();
        assert_ne!(
            compiled, tracing,
            "the compile made the key runnable, so a reader caching the \
             answer from the trace-time number must be told"
        );
        // Once compiled, asking is free of side effects on the number.
        assert!(driver.has_runnable_compiled_loop(green_key));
        assert_eq!(driver.runnable_generation(), compiled);
        assert_ne!(cold, compiled);
    }

    #[test]
    fn test_start_bridge_tracing_skips_non_traceable_state() {
        let mut driver = JitDriver::<NonTraceableState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let green_key = 404u64;
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::StartedTracing
        ));
        {
            // Use input arg so optimizer cannot fold. Input is 0 (falsy)
            // at runtime → GuardTrue fails.
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[], 0, 0);
            ctx.set_fail_args(g, &[]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert!(driver.has_compiled_loop(green_key));
        let failure = driver
            .meta
            .run_compiled_detailed(green_key, &[0])
            .expect("guard should fail");
        let fail_values = crate::compile::raw_exit_values(&failure.typed_values);
        let descr_arc = failure
            .descr_arc
            .clone()
            .expect("a guard exit carries its descr");
        drop(failure);

        let started = driver.start_bridge_tracing(
            &descr_arc,
            &mut NonTraceableState,
            &(),
            &fail_values,
            0,
            false,
            &[],
        );
        assert!(!started);
        assert!(!driver.meta.is_tracing());
    }

    #[test]
    fn bridge_entry_preserves_the_source_driver_through_symbolic_setup() {
        struct BridgeState;
        impl JitState for BridgeState {
            type Meta = ();
            type Sym = ();
            type Env = ();
            fn build_meta(&self, _: usize, _: &()) {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                vec![0]
            }
            fn create_sym(_: &(), _: usize) {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                vec![]
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
            fn rebuild_from_resumedata(
                _: &mut (),
                fail_arg_types: &[Type],
                storage: Option<&Arc<crate::resume::ResumeStorage>>,
            ) -> Option<crate::ResumeDataResult> {
                // This guard has no live state to reload. Unlike the default
                // trait stub, the fixture admits the symbolic setup path.
                assert!(fail_arg_types.is_empty());
                Some(crate::ResumeDataResult {
                    frames: vec![],
                    virtualizable_boxes: vec![],
                    virtualref_values: vec![],
                    storage: storage.cloned(),
                    num_failargs: 0,
                    fail_arg_types: vec![],
                })
            }
        }
        let mut driver = JitDriver::<BridgeState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let host_driver = driver
            .register_descriptor(JitDriverStaticData::new(vec![], vec![("value", Type::Int)]));
        let source_driver = driver
            .meta
            .register_jitdriver_sd(JitDriverStaticData::new(vec![], vec![("value", Type::Int)]));
        assert_ne!(host_driver, source_driver);
        let descriptor = driver.meta.staticdata.jitdrivers_sd[source_driver].clone();
        let green_key = 406;
        assert!(matches!(
            driver
                .meta
                .force_start_tracing(green_key, (0, 0), Some(descriptor), &[Value::Int(0)],),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().unwrap();
            let guard = ctx.record_guard(OpCode::GuardTrue, &[OpRef::input_arg_int(0)], 0);
            ctx.capture_snapshot_for_last_guard(&[], 0, 0);
            ctx.set_fail_args(guard, &[]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        // warmstate.py maybe_compile_and_run reads the compiling
        // jitdriver_sd.warmstate cell, not leftover portal 0.
        let failure = driver
            .meta
            .run_compiled_detailed_with_values_on_driver(
                source_driver,
                green_key,
                &[Value::Int(0)],
                0,
            )
            .unwrap();
        let fail_values = crate::compile::raw_exit_values(&failure.typed_values);
        let descr = failure.descr_arc.clone().unwrap();
        let mut state = BridgeState;
        assert!(driver.start_bridge_tracing(&descr, &mut state, &(), &fail_values, 0, false, &[]));
        assert_eq!(driver.meta.active_jitdriver_sd, Some(source_driver));
        assert_eq!(
            driver
                .meta
                .trace_ctx()
                .and_then(|ctx| ctx.driver_descriptor())
                .and_then(|descriptor| descriptor.index),
            Some(source_driver),
        );
    }

    #[test]
    fn start_bridge_tracing_starts_when_compiled_loops_misses() {
        // compile.py `_trace_and_compile_from_bridge` gates on
        // `rd_loop_token.loop_token_wref()`, not the leftover
        // `compiled_loops` row.
        struct BridgeState;
        impl JitState for BridgeState {
            type Meta = ();
            type Sym = ();
            type Env = ();
            fn build_meta(&self, _: usize, _: &()) {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                vec![0]
            }
            fn create_sym(_: &(), _: usize) {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                vec![]
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
            fn rebuild_from_resumedata(
                _: &mut (),
                fail_arg_types: &[Type],
                storage: Option<&Arc<crate::resume::ResumeStorage>>,
            ) -> Option<crate::ResumeDataResult> {
                assert!(fail_arg_types.is_empty());
                Some(crate::ResumeDataResult {
                    frames: vec![],
                    virtualizable_boxes: vec![],
                    virtualref_values: vec![],
                    storage: storage.cloned(),
                    num_failargs: 0,
                    fail_arg_types: vec![],
                })
            }
        }
        let mut driver = JitDriver::<BridgeState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let source_driver = driver
            .register_descriptor(JitDriverStaticData::new(vec![], vec![("value", Type::Int)]));
        let descriptor = driver.meta.staticdata.jitdrivers_sd[source_driver].clone();
        let green_key = 407;
        assert!(matches!(
            driver
                .meta
                .force_start_tracing(green_key, (0, 0), Some(descriptor), &[Value::Int(0)],),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().unwrap();
            let guard = ctx.record_guard(OpCode::GuardTrue, &[OpRef::input_arg_int(0)], 0);
            ctx.capture_snapshot_for_last_guard(&[], 0, 0);
            ctx.set_fail_args(guard, &[]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        let failure = driver
            .meta
            .run_compiled_detailed_with_values_on_driver(
                source_driver,
                green_key,
                &[Value::Int(0)],
                0,
            )
            .unwrap();
        let fail_values = crate::compile::raw_exit_values(&failure.typed_values);
        let descr = failure.descr_arc.clone().unwrap();
        assert!(driver.meta.take_compiled_loop(green_key).is_some());
        assert!(driver.meta.get_compiled_meta(green_key).is_none());
        let mut state = BridgeState;
        assert!(
            driver.start_bridge_tracing(&descr, &mut state, &(), &fail_values, 0, false, &[]),
            "live loop_token_wref starts a bridge without compiled_loops"
        );
        assert!(driver.meta.is_tracing());
    }

    #[test]
    fn setup_bridge_sym_runs_after_rebuild_seats_frames() {
        // `rebuild_state_after_failure` runs `newframe` / `consume_boxes`
        // then `synchronize_virtualizable`. `setup_bridge_sym` fills the
        // frontend after those frames exist.
        use std::cell::Cell;

        use majit_ir::resumedata::RebuiltFrame;
        use majit_jitcode::insns::BC_LIVE;
        use majit_jitcode::jitcode::JitCodeBody;

        thread_local! {
            static FRAMESTACK_LEN_AT_PREPARE: Cell<Option<usize>> = const { Cell::new(None) };
            static FRAMESTACK_LEN_AT_SETUP: Cell<Option<usize>> = const { Cell::new(None) };
            static META_PTR: Cell<*const crate::pyjitpl::MetaInterp<()>> =
                const { Cell::new(std::ptr::null()) };
        }

        struct MetaPtrReset;
        impl Drop for MetaPtrReset {
            fn drop(&mut self) {
                META_PTR.with(|p| p.set(std::ptr::null()));
            }
        }

        struct BridgePendingOrderState;
        impl JitState for BridgePendingOrderState {
            type Meta = ();
            type Sym = ();
            type Env = ();
            fn build_meta(&self, _: usize, _: &()) {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                vec![0]
            }
            fn create_sym(_: &(), _: usize) {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                vec![]
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
            fn rebuild_from_resumedata(
                _: &mut (),
                fail_arg_types: &[Type],
                storage: Option<&Arc<crate::resume::ResumeStorage>>,
            ) -> Option<crate::ResumeDataResult> {
                assert!(fail_arg_types.is_empty());
                Some(crate::ResumeDataResult {
                    frames: vec![RebuiltFrame {
                        jitcode_index: 0,
                        pc: 0,
                        values: vec![],
                    }],
                    virtualizable_boxes: vec![],
                    virtualref_values: vec![],
                    storage: storage.cloned(),
                    num_failargs: 0,
                    fail_arg_types: vec![],
                })
            }
            fn prepare_bridge_resume(
                _: &mut (),
                _: &mut crate::TraceCtx,
                _: &crate::ResumeDataResult,
                _: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                _: &[i64],
                _: &[Type],
                _: &mut crate::BridgeVirtualCache<'_>,
            ) {
                FRAMESTACK_LEN_AT_PREPARE.with(|c| {
                    c.set(META_PTR.with(|p| {
                        let ptr = p.get();
                        if ptr.is_null() {
                            None
                        } else {
                            Some(unsafe { &*ptr }.framestack.frames.len())
                        }
                    }))
                });
            }
            fn setup_bridge_sym(
                _: &mut (),
                _: &mut crate::TraceCtx,
                _: &crate::ResumeDataResult,
                _: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                _: &[i64],
                _: &[Type],
                _: &mut crate::BridgeVirtualCache<'_>,
                _: &crate::VrefVableBoxes,
            ) {
                FRAMESTACK_LEN_AT_SETUP.with(|c| {
                    c.set(META_PTR.with(|p| {
                        let ptr = p.get();
                        if ptr.is_null() {
                            None
                        } else {
                            Some(unsafe { &*ptr }.framestack.frames.len())
                        }
                    }))
                });
            }
        }

        FRAMESTACK_LEN_AT_PREPARE.with(|c| c.set(None));
        FRAMESTACK_LEN_AT_SETUP.with(|c| c.set(None));
        let mut driver = JitDriver::<BridgePendingOrderState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver.register_descriptor(JitDriverStaticData::new(vec![], vec![("value", Type::Int)]));

        let jc = crate::jitcode::JitCode::new("bridge-pending-order");
        let mut startpoints = indexmap::IndexSet::new();
        startpoints.insert(0);
        jc.set_body(JitCodeBody {
            code: vec![BC_LIVE, 0, 0, 0],
            startpoints: Some(startpoints),
            ..JitCodeBody::default()
        });
        let jitcode = std::sync::Arc::new(jc);
        {
            let sd = std::sync::Arc::get_mut(&mut driver.meta.staticdata)
                .expect("staticdata uniquely owned before tracing");
            sd.op_live = i32::from(BC_LIVE);
            sd.jitcodes.push(jitcode.clone());
        }
        driver.install_extracted_portal_jitcode(jitcode);

        let green_key = 407u64;
        assert!(matches!(
            driver
                .meta
                .force_start_tracing(green_key, (0, 0), None, &[Value::Int(0)]),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().unwrap();
            let guard = ctx.record_guard(OpCode::GuardTrue, &[OpRef::input_arg_int(0)], 0);
            ctx.capture_snapshot_for_last_guard(&[], 0, 0);
            ctx.set_fail_args(guard, &[]);
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        let failure = driver
            .meta
            .run_compiled_detailed(green_key, &[0])
            .expect("guard should fail");
        let fail_values = crate::compile::raw_exit_values(&failure.typed_values);
        let descr = failure.descr_arc.clone().unwrap();
        drop(failure);

        META_PTR.with(|p| p.set(&driver.meta as *const _));
        let _reset = MetaPtrReset;
        let mut state = BridgePendingOrderState;
        assert!(driver.start_bridge_tracing(&descr, &mut state, &(), &fail_values, 0, false, &[]));
        assert_eq!(
            FRAMESTACK_LEN_AT_PREPARE.with(|c| c.get()),
            Some(0),
            "_prepare runs while framestack is empty"
        );
        assert_eq!(
            FRAMESTACK_LEN_AT_SETUP.with(|c| c.get()),
            Some(1),
            "setup_bridge_sym runs after rebuild_from_resumedata seats frames"
        );
        assert_eq!(
            driver.meta.framestack.frames.len(),
            1,
            "rebuild_portal_framestack_from_resumedata seats frames before setup_bridge_sym"
        );
        assert_eq!(
            driver.portal_framestack_len(),
            1,
            "call_jit skips a second rebuild_from_resumedata when frames are already seated"
        );
    }

    // ── Multi-entry point lifecycle tests ──
    // Parity with warmspot.py multi-driver entry semantics: multiple functions
    // can share the same JitDriver and compiled loops via register_entry_point.

    #[test]
    fn test_multi_entry_point_registration() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.meta.finish_setup_descrs_for_jitdrivers();

        // Initially no entry points.
        assert!(driver.entry_points().is_empty());

        // Register two entry points with different green key schemas.
        driver.register_entry_point("loop_main", &[Type::Int]);
        driver.register_entry_point("loop_helper", &[Type::Int, Type::Ref]);

        assert_eq!(driver.entry_points().len(), 2);

        // Verify first entry point.
        let ep0 = driver
            .find_entry_point("loop_main")
            .expect("should find loop_main");
        assert_eq!(ep0.name, "loop_main");
        assert_eq!(ep0.schema, vec![GreenType::Int]);

        // Verify second entry point.
        let ep1 = driver
            .find_entry_point("loop_helper")
            .expect("should find loop_helper");
        assert_eq!(ep1.name, "loop_helper");
        assert_eq!(ep1.schema, vec![GreenType::Int, GreenType::Ref]);

        // Non-existent entry point returns None.
        assert!(driver.find_entry_point("nonexistent").is_none());
    }

    #[test]
    fn test_multi_entry_points_share_compiled_loops() {
        // Two entry points register different functions on the same driver.
        // A loop compiled from one entry point is visible to back_edge calls
        // from any entry point, since they share the same MetaInterp and
        // compiled loop table (keyed by green_key, not by entry point name).
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver.register_entry_point("func_a", &[Type::Int]);
        driver.register_entry_point("func_b", &[Type::Int]);

        // Both entry points are registered.
        assert_eq!(driver.entry_points().len(), 2);

        // Compile a loop from "func_a"'s perspective using green_key=10.
        let key = 10u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
            sum
        };
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        // "func_b" can see the compiled loop via the same green_key.
        // The compiled loop is shared — it doesn't matter which entry point
        // triggered the compilation.
        assert!(
            driver.has_compiled_loop(key),
            "compiled loop should be visible from any entry point"
        );

        // A different green key from "func_b" does not see the loop.
        let other_key = 20u64;
        assert!(
            !driver.has_compiled_loop(other_key),
            "different green_key should not match"
        );

        // The driver stats reflect exactly one compiled loop.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
    }

    #[test]
    fn jitdriver_install_canonical_liveness_publishes_macro_emit_composition() {
        // Mirrors the body of the macro-emitted
        // `__JitMeta::install_canonical_liveness`
        // (`majit-macros/src/jit_interp/codegen_state.rs`):
        //   1. Build a fresh `Assembler`.
        //   2. Encode the canonical
        //      `live_slots_for_state_field_jit(num_scalars, &lens, num_virt)`
        //      triple via `_encode_liveness`.
        //   3. Forward to `JitDriver::install_canonical_liveness`.
        //
        // RPython parity: `warmspot.py:281-289` →
        // `pyjitpl.py MetaInterpStaticData.finish_setup self.liveness_info = "".join(asm.all_liveness)`
        // — the metainterp-side test
        // (`pyjitpl.rs::metainterp_install_canonical_liveness_publishes_asm_bytes`)
        // exercises the inner layer; this driver-level test guards the
        // pass-through wrapper the macro actually targets.
        use majit_jitcode::codewriter::assembler::Assembler;

        // Build the canonical liveness exactly the way the macro expansion
        // does (orth-6 helper + orth-2 _encode_liveness + insns
        // registration mirroring `assembler.py:222 self.insns[key] = opnum`).
        let (live_i, live_r, live_f) =
            crate::live_slots_for_state_field_jit(2, &[1], 0, 0, 0, 0, 0, 0);
        let mut asm = Assembler::new();
        let mut scratch = Vec::<u8>::new();
        asm._encode_liveness(&live_i, &live_r, &live_f, &mut scratch);
        asm.register_insn("live/", crate::jitcode::insns::BC_LIVE);
        asm.register_insn(
            "catch_exception/L",
            crate::jitcode::insns::BC_CATCH_EXCEPTION,
        );
        asm.register_insn("rvmprof_code/ii", crate::jitcode::insns::BC_RVMPROF_CODE);
        asm.register_insn("int_return/i", crate::jitcode::insns::BC_INT_RETURN);
        asm.register_insn("ref_return/r", crate::jitcode::insns::BC_REF_RETURN);
        asm.register_insn("float_return/f", crate::jitcode::insns::BC_FLOAT_RETURN);
        asm.register_insn("void_return/", crate::jitcode::insns::BC_VOID_RETURN);
        let expected = asm.all_liveness().to_vec();

        let mut driver = JitDriver::<TypedRestoreState>::new(0);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        driver.install_canonical_liveness(&asm);

        assert_eq!(driver.meta.staticdata.liveness_info, expected);
        // `setup_insns(asm.insns())` (`pyjitpl.py`) resolves
        // `op_live` from the `live/` entry the macro registers above
        // — same data flow as RPython where `make_jitcodes()`
        // populates `asm.insns` before `finish_setup(codewriter)`.
        assert_eq!(
            driver.meta.staticdata.op_live,
            crate::jitcode::insns::BC_LIVE as i32,
        );
    }

    /// `warmspot.py:663-665` `jd._green_args_spec = [...]` /
    /// `jd.num_green_args = len(jd._green_args_spec)` — spec length
    /// must match the number of green vars on the jd.
    #[test]
    fn green_args_spec_length_matches_num_green_args() {
        let sd = JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Ref)],
            vec![("r0", Type::Int)],
        );
        let spec = sd.green_args_spec();
        assert_eq!(spec.len(), sd.num_green_args());
        assert_eq!(spec, vec![GreenType::Int, GreenType::Ref]);
    }

    /// 1 — per-kind partition of greens/reds aligning with
    /// `bhimpl_jit_merge_point`'s `[I, R, F, I, R, F]` payload shape.
    #[test]
    fn green_red_kind_counts_partition_vars_by_ir_type() {
        let sd = JitDriverStaticData::new(
            vec![
                ("g_int_a", Type::Int),
                ("g_ref_a", Type::Ref),
                ("g_int_b", Type::Int),
            ],
            vec![
                ("r_float_a", Type::Float),
                ("r_int_a", Type::Int),
                ("r_float_b", Type::Float),
                ("r_ref_a", Type::Ref),
            ],
        );
        assert_eq!(sd.green_kind_counts(), (2, 1, 0));
        assert_eq!(sd.red_kind_counts(), (1, 1, 2));
    }

    #[test]
    fn green_red_kind_counts_skip_void_slots() {
        let sd = JitDriverStaticData::new(
            vec![("g_void", Type::Void), ("g_int", Type::Int)],
            vec![("r_void", Type::Void), ("r_ref", Type::Ref)],
        );
        assert_eq!(sd.green_kind_counts(), (1, 0, 0));
        assert_eq!(sd.red_kind_counts(), (0, 1, 0));
    }

    /// `warmstate.py` `make_unwrap_greenkey().unwrap_greenkey(greenkey)`
    /// — the resulting `GreenKey` carries values + types in declaration
    /// order so `comparekey` / `get_uhash` dispatch is type-correct.
    #[test]
    fn unwrap_greenkey_uses_green_args_spec_in_declaration_order() {
        let sd = JitDriverStaticData::new(
            vec![("g_int", Type::Int), ("g_ref", Type::Ref)],
            vec![("r_int", Type::Int)],
        );
        let key = sd.unwrap_greenkey(&[42, 0xdeadbeef]);
        assert_eq!(&key.values[..], &[42, 0xdeadbeef]);
        assert_eq!(&key.types[..], &[GreenType::Int, GreenType::Ref]);
    }

    /// `warmstate.py:535-553` — upstream's `unrolling_iterable` loop
    /// implicitly asserts `len(greenkey) == len(_green_args_spec)`. The
    /// pyre helper makes that assertion explicit.
    #[test]
    #[should_panic(expected = "expected 2 green values per _green_args_spec, got 1")]
    fn unwrap_greenkey_panics_when_value_count_mismatches_spec() {
        let sd = JitDriverStaticData::new(vec![("g0", Type::Int), ("g1", Type::Int)], vec![]);
        let _ = sd.unwrap_greenkey(&[42]);
    }

    /// `warmspot.py:664` `red_args_types = [history.getkind(TYPE)[0] ...]`.
    /// The IR-type bridge must preserve declaration order and each char
    /// must round-trip to the matching `Type` variant.
    #[test]
    fn red_arg_types_as_ir_types_maps_chars_in_declaration_order() {
        let sd = JitDriverStaticData::new(
            vec![],
            vec![("r0", Type::Int), ("r1", Type::Ref), ("r2", Type::Float)],
        );
        let mapped = sd.red_arg_types_as_ir_types();
        assert_eq!(mapped, vec![Type::Int, Type::Ref, Type::Float]);
        assert_eq!(mapped.len(), sd.num_red_args());
    }

    /// `warmspot.py:664` only emits 'i' / 'r' / 'f' / 'v'; an unknown
    /// char indicates corrupted state and must surface immediately.
    #[test]
    #[should_panic(expected = "unknown history kind char")]
    fn red_arg_types_as_ir_types_panics_on_unknown_char() {
        let mut sd = JitDriverStaticData::new(vec![], vec![("r0", Type::Int)]);
        sd.red_args_types = vec!['x'];
        let _ = sd.red_arg_types_as_ir_types();
    }

    /// `warmstate.py` `JitCell.comparekey(*greenargs2)` — two
    /// `GreenKey`s built from the same spec + values must compare equal,
    /// and a value mismatch in any position must compare unequal.
    #[test]
    fn unwrap_greenkey_round_trips_through_comparekey_matches() {
        use crate::warmstate::BaseJitCell;

        let sd = JitDriverStaticData::new(vec![("g0", Type::Int), ("g1", Type::Ref)], vec![]);
        let key_a = sd.unwrap_greenkey(&[7, 0x1000]);
        let key_b = sd.unwrap_greenkey(&[7, 0x1000]);
        let key_c = sd.unwrap_greenkey(&[7, 0x2000]);

        let mut cell = BaseJitCell::new();
        *cell.comparekey.borrow_mut() = Some(key_a);
        assert!(cell.comparekey_matches(&key_b));
        assert!(!cell.comparekey_matches(&key_c));
    }

    #[test]
    fn jit_driver_index_returns_descriptor_index() {
        // `JitDriver::index()` must forward `descriptor.index`; returns None
        // before registration stamps it.
        let mut sd = JitDriverStaticData::new(vec![], vec![]);
        sd.index = Some(3);
        let driver = JitDriver::<TypedRestoreState>::with_descriptor(1, sd);
        assert_eq!(driver.index(), Some(3));
    }

    #[test]
    fn jit_driver_index_returns_none_without_descriptor() {
        // A driver constructed without a descriptor (threshold-only) must
        // return None from index().
        let driver = JitDriver::<TypedRestoreState>::new(1);
        assert_eq!(driver.index(), None);
    }

    /// Install the cell shape `warmstate.py get_assembler_token` leaves
    /// behind when it has to synthesise a token: a procedure token carrying a
    /// backend body, flagged `JC_TEMPORARY`, and no `compiled_loops` entry —
    /// MetaInterp never files frontend meta for a `compile_tmp_callback` stub
    /// (`compile.py:1101-1150`).
    fn attach_tmp_callback_cell<S: JitState>(driver: &mut JitDriver<S>, green_key: u64) {
        let token = alive_tmp_token(driver);
        driver
            .meta
            .warm_state_for_driver(0)
            .attach_tmp_callback_to_interp(green_key, token);
    }

    /// Typed twin of [`attach_tmp_callback_cell`]: the cell carries its greens,
    /// as the one `get_assembler_token` files does.
    fn attach_tmp_callback_cell_for_key<S: JitState>(driver: &mut JitDriver<S>, key: &GreenKey) {
        let token = alive_tmp_token(driver);
        driver
            .meta
            .warm_state_for_driver(0)
            .attach_tmp_callback_to_interp_for_key(key, token);
    }

    fn alive_tmp_token<S: JitState>(
        driver: &mut JitDriver<S>,
    ) -> std::sync::Arc<majit_backend::JitCellToken> {
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(
            driver.meta.warm_state_for_driver(0).alloc_token_number(),
        ));
        token.set_compiled(Box::new(()));
        // `compile.py` — `compile_tmp_callback` registers the token
        // with `MemoryManager` before `get_assembler_token` installs it. The
        // cell keeps only a weak handle (`warmstate.py`), so without this
        // the token dies here and the cell reads back as "never compiled".
        driver
            .meta
            .warm_state_for_driver(0)
            .memory_manager
            .keep_loop_alive(&token);
        token
    }

    /// `warmstate.py maybe_compile_and_run`: a cell whose token was
    /// `attached by compile_tmp_callback()` is counted normally, never entered.
    #[test]
    fn a_tmp_callback_only_cell_is_counted_not_entered_at_the_back_edge() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 4413u64;
        attach_tmp_callback_cell(&mut driver, key);

        assert!(
            driver.has_compiled_loop(key),
            "the tmp callback token has a body, so the code-presence predicate says yes",
        );
        assert!(
            !driver.has_runnable_compiled_loop(key),
            "and it has no frontend meta, so the warm-entry runner cannot enter it",
        );

        // Before this fell through, the back edge unwrapped the absent meta and
        // panicked at the loop header of any driver that can mint a tmp token.
        let mut state = TypedRestoreState::default();
        assert_eq!(
            driver.back_edge_keyed(key, 0, &mut state, &(), || {}),
            None,
            "nothing ran and nothing was traced on the first tick",
        );
        assert!(
            !driver.meta.is_tracing(),
            "the first back edge only ticks the counter",
        );

        // The counter keeps ticking, so the real loop still gets traced — the
        // tmp callback delays entry, it does not disable the key.
        driver.back_edge_keyed(key, 0, &mut state, &(), || {});
        assert!(
            driver.meta.is_tracing(),
            "the threshold is reached by counting, exactly as for a cell with no token",
        );
    }

    /// `maybe_compile_and_run` calls `bound_reached(hash, cell, *args)` with
    /// the `JC_TEMPORARY` cell it found. A non-zero `code_ptr` takes
    /// `bound_reached`'s typed arm (`force_start_tracing_for_key`); that arm
    /// must set `JC_TRACING` on this cell and not install a second one.
    /// `code_ptr == 0` stays on `ensure_cell_by_key` and cannot show the split.
    #[test]
    fn a_temporary_cell_starts_tracing_on_the_found_cell() {
        // `WarmEnterState::new` installs the Ref hash resolver. Hash the
        // portal greens after that, or the key will not match the one
        // `with_typed_decision_key` rebuilds.
        let mut driver = JitDriver::<CountingDoorState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let target_pc = 7usize;
        let code_ptr = 0x1234usize;
        let key = GreenKey::with_types(
            vec![target_pc as i64, 0, code_ptr as i64],
            vec![Type::Int, Type::Int, Type::Ref],
        );
        let green_key = majit_ir::pypyjit_greenkey_uhash(target_pc, false, code_ptr as u64);
        assert_eq!(green_key, key.get_uhash());
        attach_tmp_callback_cell_for_key(&mut driver, &key);

        let token_number = {
            let cell = driver
                .meta
                .warm_state_for_driver(0)
                .lookup_chain(green_key)
                .expect("the temporary callback installed one cell");
            assert!(
                cell.next.borrow().is_none(),
                "the fast path only owns a lone cell"
            );
            assert_eq!(cell.cell_bucket.get(), green_key);
            assert!(cell.comparekey_matches(&key));
            assert!(
                cell.flags
                    .get()
                    .contains(crate::warmstate::JcFlags::JC_TEMPORARY)
            );
            assert!(!cell.is_tracing());
            cell.get_procedure_token()
                .expect("the callback token is live")
                .number
        };
        assert_eq!(
            driver.meta.warm_state_for_driver(0).get_stats().num_cells,
            1
        );

        let mut state = CountingDoorState {
            code_ptr,
            ..Default::default()
        };
        assert_eq!(
            driver.back_edge_keyed(green_key, target_pc, &mut state, &(), || {}),
            None,
            "the first tick must not start tracing",
        );
        assert!(!driver.meta.is_tracing());
        assert_eq!(
            driver
                .back_edge_keyed(green_key, target_pc, &mut state, &(), || {})
                .and_then(|resume| resume.resume_pc()),
            Some(target_pc),
        );
        assert!(driver.meta.is_tracing());
        assert_eq!(
            driver.meta.warm_state_for_driver(0).get_stats().num_cells,
            1,
            "bound_reached must mark the found cell, not install another",
        );

        let cell = driver
            .meta
            .warm_state_for_driver(0)
            .lookup_chain(green_key)
            .expect("the original cell is still the bucket");
        assert!(cell.next.borrow().is_none());
        assert!(cell.is_tracing());
        assert!(
            cell.flags
                .get()
                .contains(crate::warmstate::JcFlags::JC_TEMPORARY)
        );
        assert!(cell.comparekey_matches(&key));
        assert_eq!(cell.cell_key.get(), Some(green_key));
        assert_eq!(
            cell.get_procedure_token()
                .expect("the callback token stays on the found cell")
                .number,
            token_number,
        );
    }

    /// `warmstate.py` resolves the cell by `comparekey(*greenargs)`
    /// before reading a token off it. On a chained bucket the head is a
    /// different cell, so the hash form and the typed form answer differently
    /// — and the typed one is the one upstream asks.
    #[test]
    fn the_typed_entry_predicate_reads_the_keys_own_cell_not_the_bucket_head() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = GreenKey::new(vec![1500, 1600]);
        let hash = key.get_uhash();

        // A hash-only writer files the cell and gives it a code-bearing token.
        // The typed writer for the SAME key stamps that cell (`JitCell.__init__`).
        let token = std::sync::Arc::new(majit_backend::JitCellToken::new(
            driver.meta.warm_state_for_driver(0).alloc_token_number(),
        ));
        token.set_compiled(Box::new(()));
        // See `attach_tmp_callback_cell`: `alive_loops` is the token's only
        // strong owner (`memmgr.py:9-14`).
        driver
            .meta
            .warm_state_for_driver(0)
            .memory_manager
            .keep_loop_alive(&token);
        driver
            .meta
            .warm_state_for_driver(0)
            .attach_tmp_callback_to_interp_for_key(&key, token);
        driver
            .meta
            .warm_state_for_driver(0)
            .mark_dont_trace_for_key(&key);

        assert!(
            !driver.meta.green_key_bucket_is_chained(hash),
            "one green key is one cell",
        );
        assert!(
            driver.has_compiled_loop(hash),
            "the cell holds the code-bearing token",
        );
        assert!(
            driver.has_compiled_loop_for_key(&key),
            "the typed predicate reads that same cell",
        );
    }

    /// The `run_compiled` sibling route makes the same decision.
    #[test]
    fn a_tmp_callback_only_cell_is_not_dispatched_by_back_edge_or_run_compiled() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let key = 5161u64;
        attach_tmp_callback_cell(&mut driver, key);

        let mut state = TypedRestoreState::default();
        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 0, &mut state, &(), || {})
                .is_none(),
            "an unenterable cell reports the same `None` as a cell with no code at all",
        );
    }
}

/// pyjitpl.py:3001-3007's JUMP arm, over the one target class it must not
/// take: a loop stored under a cross-loop cut's inner jitcell token
/// (compile.py:269-270).
#[cfg(test)]
mod cross_loop_cut_close_tests {
    use super::*;
    use majit_ir::{OpCode, OpRef};

    /// Greens the compiled inner loop is registered under.
    fn inner_greens() -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        (vec![4242], vec![], vec![])
    }

    /// A trace whose closing JUMP carries ONE int, matching the inner loop's
    /// single inputarg (compile.py:334 `jump.numargs() == label.numargs()`).
    #[derive(Default)]
    struct CutCloseState;

    impl JitState for CutCloseState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            vec![0]
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            vec![OpRef::input_arg_int(0)]
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    /// Trace and compile a one-inputarg loop at `key`, registering it under
    /// `inner_greens()`. `cut` routes `compile_loop` through the cross-loop cut
    /// (compile.py:269-270) so the loop is stored under the inner token.
    fn compile_inner_loop(driver: &mut JitDriver<CutCloseState>, key: u64, cut: bool) {
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef::input_arg_int(0);
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            // The optimizer rejects a guardless loop.
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
            if cut {
                ctx.cut_inner_green_key = Some(key);
            }
        }
        driver.meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert!(driver.has_compiled_loop(key), "inner loop must compile");
        driver.meta.record_loop_header_greens(key, inner_greens());
        assert_eq!(
            driver.meta.is_cross_loop_cut_key(key),
            cut,
            "the cut mark must follow the key compile_loop stored under",
        );
    }

    /// Start an interp-origin trace (pyjitpl.py:2905 `ResumeFromInterpDescr`)
    /// and record one op into it.
    fn start_outer_trace(driver: &mut JitDriver<CutCloseState>, outer_key: u64) {
        let mut state = CutCloseState;
        driver.force_start_tracing(outer_key, outer_key as usize, &mut state, &());
        assert!(driver.is_tracing(), "outer trace must start");
        assert!(
            driver.meta.bridge_info().is_none(),
            "force_start_tracing is the interp origin",
        );
        let ctx = driver.meta.trace_ctx().expect("tracing ⇒ trace ctx");
        let i0 = OpRef::input_arg_int(0);
        let c2 = ctx.const_int(2);
        let sum = ctx.record_op(OpCode::IntAdd, &[i0, c2]);
        let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
        ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
        ctx.set_fail_args(g, &[sum]);
    }

    /// The publish the walker makes at a merge point that already owns a
    /// procedure token (pyjitpl.py `get_procedure_token(greenboxes)`).
    fn close_jumping_into(driver: &mut JitDriver<CutCloseState>, target: u64) {
        driver.merge_point(|meta, _sym| {
            let Some(ctx) = meta.trace_ctx() else {
                return TraceAction::Continue;
            };
            if std::mem::take(&mut ctx.merge_point_resumed) {
                return TraceAction::Continue;
            }
            ctx.close_greens = Some(inner_greens());
            ctx.close_jump_into_key = Some(target);
            ctx.close_jump_boxes = Some(vec![(OpRef::input_arg_int(0), Type::Int)]);
            TraceAction::CloseLoop
        });
    }

    /// A cross-loop cut leaves no PREAMBLE in `front_target_tokens[0]` — that
    /// slot holds the cut prefix, whose entry invariants only the cutting trace
    /// has proven. `jump_to_preamble` (unroll.py) is the fallback
    /// whenever no specialized label matches, and an interp-origin entry bridge
    /// always reaches it, so a cut target would be entered with the specialized
    /// label's bounds unestablished.
    #[test]
    fn interp_origin_close_declines_a_cross_loop_cut_target() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let inner_key = 7031u64;
        compile_inner_loop(&mut driver, inner_key, /* cut */ true);

        start_outer_trace(&mut driver, 7032);
        close_jumping_into(&mut driver, inner_key);

        // pyjitpl.py: `compile_trace` returned without raising, so
        // the trace is NOT given up — the walk keeps recording.
        // Upstream retries compile_trace on the next header visit
        // (`if not self.partial_trace`); do not latch the decline.
        assert!(
            driver.is_tracing(),
            "a cut target must not be entered by an entry bridge",
        );
    }

    /// The control the decline is measured against: the same close into a loop
    /// compiled at its own header does take the JUMP, and a successful
    /// `compile_trace` ends the trace (pyjitpl.py
    /// `raise_if_successful`).
    #[test]
    fn interp_origin_close_takes_a_non_cut_target() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let inner_key = 7051u64;
        compile_inner_loop(&mut driver, inner_key, /* cut */ false);

        start_outer_trace(&mut driver, 7052);
        close_jumping_into(&mut driver, inner_key);

        assert!(
            !driver.is_tracing(),
            "a non-cut target closes: compile_trace raises and tracing ends",
        );
    }

    /// The third outcome the `Declined` arm has to keep apart: closing into a
    /// key that owns no compiled target fails `has_compiled_targets`
    /// (pyjitpl.py `get_procedure_token` returning nothing), so the close
    /// is never evaluated at all.  `result` is still its initializer here, so
    /// this lands in the same `Declined` arm a rejected attempt lands in — and
    /// the two must not report as one event.
    ///
    /// Slot 67 is process-global and this suite runs in parallel, so the
    /// assertion is a before/after bump rather than an absolute count.
    #[test]
    fn interp_origin_close_into_an_uncompiled_target_is_not_a_declined_attempt() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        // Deliberately never passed to `compile_inner_loop`.
        let never_compiled = 7061u64;
        assert!(
            !driver.has_compiled_loop(never_compiled),
            "the fixture's premise: nothing is compiled at this key",
        );

        let before = crate::mc_diag(67);
        start_outer_trace(&mut driver, 7062);
        close_jumping_into(&mut driver, never_compiled);

        assert!(
            crate::mc_diag(67) > before,
            "a close that was never attempted must be counted under its own slot",
        );
        assert!(
            driver.is_tracing(),
            "the walk keeps recording, as on the declined-attempt path",
        );
    }

    /// `close_jumping_into` with the walk-entry bookkeeping the
    /// `jit_merge_point!` expansion performs, and with a `close_green_pc` the
    /// caller chooses.  Returns the `walk_resume_pc` observed at the top of
    /// each walk — one entry per invocation of the closure, which is what the
    /// macro reads to decide where to seed the walk.
    fn close_jumping_into_observing_resume(
        driver: &mut JitDriver<CutCloseState>,
        target: u64,
        close_pc: Option<i64>,
    ) -> Vec<Option<usize>> {
        let mut observed = Vec::new();
        driver.merge_point(|meta, _sym| {
            let Some(ctx) = meta.trace_ctx() else {
                return TraceAction::Continue;
            };
            observed.push(ctx.walk_resume_pc.take());
            if std::mem::take(&mut ctx.merge_point_resumed) {
                return TraceAction::Continue;
            }
            ctx.close_greens = Some(inner_greens());
            ctx.close_green_pc = close_pc;
            ctx.close_jump_into_key = Some(target);
            ctx.close_jump_boxes = Some(vec![(OpRef::input_arg_int(0), Type::Int)]);
            TraceAction::CloseLoop
        });
        observed
    }

    /// Where a declined close re-enters the walk.
    ///
    /// pyjitpl.py:1571/1574 `saved_pc`: upstream consults one merge point per
    /// `opimpl_jit_merge_point` call, so the position that entered the call and
    /// the position that declined cannot differ.  A pyre walk covers a whole
    /// segment, so they can — and only the second one names the state the walk
    /// is carrying.  The driver hands that pc over; the walk is re-seeded from
    /// it rather than from wherever the segment began.
    ///
    /// `close_jumping_into`, which the rest of this module drives, publishes no
    /// `close_green_pc` at all, so none of those tests can tell the two pcs
    /// apart.
    #[test]
    fn a_declined_close_hands_over_the_pc_that_declined() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let inner_key = 7081u64;
        compile_inner_loop(&mut driver, inner_key, /* cut */ true);

        // Deliberately unrelated to the trace's own header, which is what a
        // close reached deeper in the walk than its entry looks like.
        let declining_pc = 385i64;
        start_outer_trace(&mut driver, 7082);
        let seen = close_jumping_into_observing_resume(&mut driver, inner_key, Some(declining_pc));

        assert!(
            driver.is_tracing(),
            "premise: the decline keeps tracing, so there is a re-entry to check",
        );
        assert_eq!(
            seen.len(),
            2,
            "the declined close re-runs the walk exactly once",
        );
        assert_eq!(
            seen[0], None,
            "the first walk is seeded by its caller; nothing has declined yet",
        );
        assert_eq!(
            seen[1],
            Some(declining_pc as usize),
            "the re-entered walk must be seeded at the merge point that \
             declined. Seeded at the pc that entered the walk instead, it \
             re-executes everything between the two against the state it \
             already advanced past, and closes a loop whose registered green \
             key contradicts the state its own JUMP carries",
        );
    }

    /// …and a close that names no pc leaves the seed alone, rather than
    /// pointing the walk at 0.  The pc is the merge point's green
    /// (pyjitpl.py:3005), so a driver whose merge point yields none has
    /// nothing to correct and keeps the pc it was called with.
    #[test]
    fn a_declined_close_without_a_pc_hands_over_nothing() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let inner_key = 7091u64;
        compile_inner_loop(&mut driver, inner_key, /* cut */ true);

        start_outer_trace(&mut driver, 7092);
        let seen = close_jumping_into_observing_resume(&mut driver, inner_key, None);

        assert_eq!(
            seen.len(),
            2,
            "the declined close re-runs the walk exactly once"
        );
        assert!(
            seen.iter().all(Option::is_none),
            "with no pc published there is nothing to re-seed from, and the \
             walk keeps the pc its caller passed",
        );
    }

    /// Retiring the loop retires the cut mark with it, so a later loop compiled
    /// at the same key is not judged by its predecessor.
    #[test]
    fn retiring_a_cut_loop_forgets_its_cut_mark() {
        let mut driver = JitDriver::<CutCloseState>::new(2);
        driver.meta.finish_setup_descrs_for_jitdrivers();
        let inner_key = 7041u64;
        compile_inner_loop(&mut driver, inner_key, /* cut */ true);

        driver.meta.remove_compiled_loop(inner_key);
        assert!(
            !driver.meta.is_cross_loop_cut_key(inner_key),
            "the cut mark must not outlive the loop it describes",
        );
    }
}

#[cfg(test)]
mod jitcode_registry_tests {
    use super::*;
    use crate::jitcode::{JitCode, RuntimeBhDescr};
    use std::sync::Arc;

    #[derive(Default)]
    struct RegistryState;

    impl JitState for RegistryState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    /// The arm a consumer holding only an `Arc` reaches through
    /// `register_dispatch_jitcode_shared`.
    ///
    /// It has to answer for a dispatch it never takes: the seeded case returns
    /// the seed's own entry, and the unnumbered case declines rather than
    /// inventing a position, because appending needs an owned value.
    #[test]
    fn the_seeded_registry_arm_reads_a_dispatch_it_does_not_take() {
        let seed = vec![jitcode("other", Some(0)), jitcode("dispatch", Some(1))];
        let shared = Arc::clone(&seed[1]);

        let (registry, adopted) = build_seeded_jitcode_registry(&seed, &shared)
            .expect("a dispatch the seed numbers is adopted from the seed");
        assert_eq!(registry.len(), seed.len(), "nothing is appended");
        assert!(
            Arc::ptr_eq(&adopted, &seed[1]),
            "the adopted portal is the seed's own entry, which is the identity \
             every `j` operand at that index resolves through",
        );

        let unnumbered = jitcode("runtime_built", None);
        assert!(
            build_seeded_jitcode_registry(&seed, &unnumbered).is_none(),
            "a jitcode carrying no index has to be appended, and appending needs ownership",
        );
    }

    /// A jitcode with a committed body, optionally already numbered the way a
    /// build-time table numbers its entries.
    fn jitcode(name: &str, prenumbered: Option<usize>) -> Arc<JitCode> {
        Arc::new(jitcode_value(name, prenumbered))
    }

    fn seeded_portal(name: &str, index: usize, jitdriver_sd: usize) -> Arc<JitCode> {
        let core = majit_jitcode::jitcode::JitCode::new(name);
        core.set_body(majit_jitcode::jitcode::JitCodeBody {
            code: vec![
                majit_jitcode::codewriter::insns::BC_JIT_MERGE_POINT,
                jitdriver_sd as u8,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            jit_merge_point_offset: Some(0),
            ..Default::default()
        });
        core.set_index(index);
        core.set_jitdriver_sd(jitdriver_sd);
        Arc::new(JitCode::from_canonical(core))
    }

    /// The same, by value, for handing `build_jitcode_registry` the graph its
    /// seed already holds.
    ///
    /// `JitCode` is deliberately not `Clone` — the runtime half is per-object
    /// — so the seed's entry and the registered portal are two
    /// `from_canonical` calls over one canonical body. That is exactly the
    /// shape an LLBC embedder arrives with.
    fn jitcode_value(name: &str, prenumbered: Option<usize>) -> JitCode {
        jitcode_value_with_merge_offset(name, prenumbered, None)
    }

    fn jitcode_value_with_merge_offset(
        name: &str,
        prenumbered: Option<usize>,
        jit_merge_point_offset: Option<usize>,
    ) -> JitCode {
        let core = majit_jitcode::jitcode::JitCode::new(name);
        core.set_body(majit_jitcode::jitcode::JitCodeBody {
            jit_merge_point_offset,
            ..Default::default()
        });
        if let Some(index) = prenumbered {
            core.set_index(index);
        }
        JitCode::from_canonical(core)
    }

    /// A dispatch jitcode that inline-calls each of `callees`.
    fn dispatch_calling(callees: &[Arc<JitCode>]) -> JitCode {
        let core = majit_jitcode::jitcode::JitCode::new("dispatch");
        core.set_body(Default::default());
        let mut dispatch = JitCode::from_canonical(core);
        dispatch.exec.descrs = callees
            .iter()
            .map(|callee| RuntimeBhDescr::JitCode(Arc::clone(callee)))
            .collect();
        dispatch
    }

    /// Every entry sits at the index it names — the property
    /// `resume.py:1338-1340` reads the registry with.
    fn assert_self_indexed(registry: &[Arc<JitCode>]) {
        for (position, jitcode) in registry.iter().enumerate() {
            assert_eq!(
                jitcode.try_index(),
                Some(position),
                "`{}` sits at {position} but names {:?}",
                jitcode.name(),
                jitcode.try_index(),
            );
        }
    }

    /// With no build-time table the dispatch jitcode is the root of the list,
    /// exactly as before a seed existed.
    #[test]
    fn an_unseeded_registry_numbers_the_dispatch_jitcode_first() {
        let helper = jitcode("helper", None);
        let (registry, dispatch) = build_jitcode_registry(&[], dispatch_calling(&[helper]));
        assert_eq!(dispatch.index(), 0);
        assert_eq!(registry.len(), 2);
        assert_self_indexed(&registry);
    }

    /// The subject: a callee that a build-time table already numbered keeps
    /// that number.
    ///
    /// Renumbering it is not a silent drift — `set_index` is set-once and
    /// asserts — so before the seed existed, inline-calling a build-time
    /// jitcode panicked here rather than tracing.
    /// A seed whose inline-called entry sits at an index the run-time
    /// numbering would otherwise hand to something else. `add` at 2 rather
    /// than at the first free slot is the point: a seed one entry long would
    /// pass this test by coincidence.
    fn seed() -> Vec<Arc<JitCode>> {
        vec![
            jitcode("mainloop", Some(0)),
            jitcode("val_add", Some(1)),
            jitcode("add", Some(2)),
        ]
    }

    #[test]
    fn a_prenumbered_callee_keeps_the_index_its_own_operands_use() {
        let seed = seed();
        let (registry, dispatch) = build_jitcode_registry(&seed, dispatch_calling(&seed[2..]));
        assert_eq!(
            seed[2].index(),
            2,
            "the seed's own `j` operands are written against this index",
        );
        assert_eq!(
            dispatch.index(),
            3,
            "the run-time numbering continues above the seed rather than \
             restarting over slots the build already assigned",
        );
        assert_eq!(
            registry.len(),
            4,
            "the callee is seeded, not appended twice"
        );
        assert_self_indexed(&registry);
    }

    /// A seed and a run-time helper in one registry: both numbering authorities
    /// land in one list with no collision and no hole.
    #[test]
    fn a_runtime_helper_is_numbered_above_a_seeded_callee() {
        let seed = seed();
        let helper = jitcode("helper", None);
        let (registry, dispatch) = build_jitcode_registry(
            &seed,
            dispatch_calling(&[Arc::clone(&seed[2]), Arc::clone(&helper)]),
        );
        assert_eq!(dispatch.index(), 3);
        assert_eq!(helper.index(), 4);
        assert_eq!(registry.len(), 5);
        assert_self_indexed(&registry);
    }

    /// A table that hands out `JitCode` entries while leaving
    /// `RuntimeDescrTable::jitcodes()` at its default seeds nothing, so the
    /// walk meets a callee the build already numbered.
    ///
    /// This is the case `set_index` structurally cannot catch: the walk hands
    /// the callee the very index it already holds, and a same-value
    /// `set_index` is a documented no-op. It would therefore have produced a
    /// registry whose slot 1 holds this shell rather than the build-time
    /// object a callee's nested frame indices resolve against — silently. The
    /// panic has to come from here, and has to name the missing method.
    #[test]
    #[should_panic(expected = "must implement `jitcodes()`")]
    fn an_incomplete_seed_names_the_method_that_would_have_supplied_it() {
        let callee = jitcode("build_time_callee", Some(1));
        build_jitcode_registry(&[], dispatch_calling(&[callee]));
    }

    /// The same defect where the walk order does *not* reproduce the
    /// build-time index. `set_index` would abort here too, but on an internal
    /// set-once invariant that names neither the seed nor the table.
    #[test]
    #[should_panic(expected = "already names index Some(7)")]
    fn an_incomplete_seed_is_reported_before_the_set_once_invariant_trips() {
        let callee = jitcode("build_time_callee", Some(7));
        build_jitcode_registry(&[], dispatch_calling(&[callee]));
    }

    /// A portal the build-time table already numbered keeps that number.
    ///
    /// This is the LLBC shape: one lowering pass writes the portal and its
    /// callees into a single table, so the portal arrives already seeded.
    /// Appending it would ask `set_index` to name a second slot for a jitcode
    /// that already names one, which aborts — every such host reached this
    /// function and panicked instead of registering.
    #[test]
    fn a_seeded_dispatch_jitcode_is_adopted_at_the_index_it_names() {
        let seed = seed();
        let portal = jitcode_value("mainloop", Some(0));

        let (registry, dispatch) = build_jitcode_registry(&seed, portal);

        assert_eq!(dispatch.index(), 0, "the portal keeps the seed's number");
        assert_eq!(registry.len(), seed.len(), "nothing is appended");
        assert!(
            std::sync::Arc::ptr_eq(&registry[0], &dispatch),
            "`all_jitcodes[jitcode.index] is jitcode` — the driver's portal has \
             to BE the registry entry, not a second object with the same body",
        );
        assert_self_indexed(&registry);
    }

    /// Adoption is keyed on the index the dispatch names, not on it being
    /// first: a table whose driver names a portal in the middle registers the
    /// entry at that position and no other.
    #[test]
    fn a_seeded_dispatch_is_adopted_wherever_the_table_put_it() {
        let seed = seed();
        let (registry, dispatch) = build_jitcode_registry(&seed, jitcode_value("add", Some(2)));
        assert_eq!(dispatch.index(), 2);
        assert_eq!(dispatch.name(), "add");
        assert!(std::sync::Arc::ptr_eq(&registry[2], &dispatch));
    }

    /// A build-time portal retains the driver slot the codewriter recorded on
    /// it. The second prepass driver is the first non-coincidental case: a
    /// fresh run-time staticdata table would otherwise allocate slot 0 and try
    /// to overwrite the seed's set-once `jitdriver_sd == 1` relationship.
    #[test]
    fn a_seeded_second_driver_registers_at_its_build_time_descriptor_index() {
        let first = jitcode("first_portal", Some(0));
        let second = seeded_portal("second_portal", 1, 1);
        let seed = vec![first, Arc::clone(&second)];
        let mut driver = JitDriver::<RegistryState>::new(2);
        driver.declare_schema(Vec::new(), Vec::new());

        driver.adopt_dispatch_registry(seed, Arc::clone(&second));

        assert_eq!(driver.index(), Some(1));
        assert!(Arc::ptr_eq(
            driver
                .dispatch_jitcode()
                .expect("registration installs the seeded portal"),
            &second,
        ));
    }

    /// Two graphs claiming one slot is not adoptable: the seed's entry and the
    /// registered portal would be different bodies under one number, and every
    /// `j` operand written against it resolves to whichever the registry holds.
    #[test]
    #[should_panic(expected = "two graphs claim one slot")]
    fn a_seeded_slot_holding_another_graph_is_refused() {
        let seed = seed();
        build_jitcode_registry(&seed, jitcode_value("portal", Some(1)));
    }

    /// A name and index are not enough to establish that the registered
    /// dispatch came from the seed's body: the encoder-owned merge-point
    /// offset must identify the same bytecode position as well.
    #[test]
    #[should_panic(expected = "merge-point offset")]
    fn a_same_named_seeded_dispatch_with_a_different_body_is_refused() {
        let seed = vec![Arc::new(jitcode_value_with_merge_offset(
            "mainloop",
            Some(0),
            Some(3),
        ))];
        let dispatch = jitcode_value_with_merge_offset("mainloop", Some(0), Some(7));
        build_jitcode_registry(&seed, dispatch);
    }

    /// Name, index, and merge offset can all agree while the bodies do not.
    /// Registration therefore validates the seed entry selected by the
    /// codewriter's indexed object identity, rather than the locator shell the
    /// caller hands over and then drops.
    #[test]
    #[should_panic(expected = "payload slot 0 count 1 disagrees")]
    fn adoption_validates_the_seeded_object_that_it_installs() {
        let seeded_core = majit_jitcode::jitcode::JitCode::new("mainloop");
        seeded_core.set_body(majit_jitcode::jitcode::JitCodeBody {
            code: vec![
                majit_jitcode::codewriter::insns::BC_JIT_MERGE_POINT,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            jit_merge_point_offset: Some(0),
            ..Default::default()
        });
        seeded_core.set_index(0);
        let seed = vec![Arc::new(JitCode::from_canonical(seeded_core))];
        let locator = seeded_portal("mainloop", 0, 0);
        let locator = Arc::try_unwrap(locator).expect("the locator has one owner");
        let (registry, adopted) = build_jitcode_registry(&seed, locator);
        let mut driver = JitDriver::<RegistryState>::new(2);
        driver.declare_schema(Vec::new(), Vec::new());

        driver.adopt_dispatch_registry(registry, adopted);
    }

    /// A dispatch numbered past the end of the seed names the method that
    /// would have supplied the rest of the table.
    #[test]
    #[should_panic(expected = "does not reach")]
    fn a_dispatch_numbered_above_the_seed_names_the_missing_table() {
        let seed = seed();
        build_jitcode_registry(&seed, jitcode_value("mainloop", Some(9)));
    }

    /// Adoption drops the caller's copy, so run-time descrs on it would never
    /// be numbered. Refuse rather than lose them.
    #[test]
    #[should_panic(expected = "run-time descrs")]
    fn a_seeded_dispatch_carrying_runtime_callees_is_refused() {
        let seed = seed();
        let helper = jitcode("helper", None);
        let mut portal = jitcode_value("mainloop", Some(0));
        portal.exec.descrs = vec![RuntimeBhDescr::JitCode(helper)];
        build_jitcode_registry(&seed, portal);
    }
}
