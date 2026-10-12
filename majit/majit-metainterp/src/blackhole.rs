//! Blackhole interpreter: executes jitcode bytecodes with concrete values.
//!
//! When a guard fails in compiled code, `resume_in_blackhole` reconstructs
//! execution frames from resume data and runs jitcode bytecodes with
//! concrete values, following all code paths (not just the traced one).
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/blackhole.py`.
//!
//! The `resume_in_blackhole` entry itself lives with its caller, as
//! `pyre_jit::eval::resume_in_blackhole_from_exit_layout`: it needs the
//! backend's exit layout plus the driver's `vinfo`/`vrefinfo`/`ginfo`, none
//! of which this crate can name.  What this module owns is everything it
//! calls into — `blackhole_from_resumedata`, `prepare_resume_from_failure`
//! and `run_forever`.

use crate::jitexc::JitException;
use indexmap::IndexMap;
use majit_ir::{GcRef, OpCode, OpRef};

/// blackhole.py:1068 parity: typed payload decoded from merge-point
/// bytecode operands. Corresponds to the 6 lists in
/// ContinueRunningNormally(gi, gr, gf, ri, rr, rf).
///
/// Corresponds to ContinueRunningNormally(gi, gr, gf, ri, rr, rf)
/// in jitexc.py — the typed portal args, NOT live locals.
pub type MergePointArgs = crate::jitexc::ContinueRunningNormallyArgs;

/// Exception state tracked during blackhole execution.
///
/// Mirrors RPython's exception tracking in the meta-interpreter.
/// Guards like GUARD_EXCEPTION and GUARD_NO_EXCEPTION check this state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExceptionState {
    /// The exception class pointer (0 = no exception pending).
    pub exc_class: i64,
    /// The exception value pointer.
    pub exc_value: i64,
    /// executor.py: metainterp.ovf_flag — set by Int*Ovf operations,
    /// checked by GuardNoOverflow / GuardOverflow.
    pub ovf_flag: bool,
}

impl ExceptionState {
    /// Whether an exception is currently pending.
    pub fn is_pending(&self) -> bool {
        self.exc_class != 0
    }

    /// Set a pending exception.
    pub fn set(&mut self, exc_class: i64, exc_value: i64) {
        self.exc_class = exc_class;
        self.exc_value = exc_value;
    }

    /// Clear the pending exception and return (class, value).
    pub fn clear(&mut self) -> (i64, i64) {
        let cls = self.exc_class;
        let val = self.exc_value;
        self.exc_class = 0;
        self.exc_value = 0;
        (cls, val)
    }
}

// RPython blackhole.py parity: BlackholeInterpreter
//
// Jitcode-based blackhole execution. When a guard fails in compiled code,
// resume_in_blackhole reconstructs execution frames from resume data and
// runs jitcode bytecodes with concrete values, following ALL code paths
// (unlike trace IR which only has the traced path).

use crate::jitcode::{self, JitArgKind, JitCode, JitCodeRuntimeExt};
use crate::pyjitpl::{MIFrame, MIFrameStack};
use crate::pyjitpl::{
    call_int_function, call_ref_function, call_void_function, eval_binop_f, eval_binop_i,
};

// BlackholeInterpBuilder instruction setup
// RPython `blackhole.py` `class BlackholeInterpBuilder` combines
// pool management AND dispatch setup. pyre's existing
// `BlackholeInterpBuilder` (below, at the pool management section) is the
// pool manager. The `setup_insns` infrastructure (opcode table + dispatch
// table) is being added incrementally as Phase D of the RPython parity
// plan. Handler function pointers and `dispatch_loop` will be wired in
// as `bhimpl_*` methods are ported one by one.

/// Handler function signature for the codewriter-orthodox dispatch table.
///
/// RPython `blackhole.py` `handler(self, code, position) -> position`.
/// Each handler decodes operands from `code[position..]` based on its
/// argcodes, calls the corresponding `bhimpl_*` method on `bh`, writes
/// results, and returns the updated position.
pub type BhOpcodeHandler =
    fn(bh: &mut BlackholeInterpreter, code: &[u8], position: usize) -> Result<usize, DispatchError>;

/// Default handler installed by `setup_insns` for opcodes whose
/// `bhimpl_*` has not been ported to `wire_bhimpl_handlers` yet.
///
/// RPython `blackhole.py` `setup_insns` immediately resolves every
/// entry via `_get_method` and would raise `AttributeError` right there
/// if a `bhimpl_*` is missing. pyre defers the crash to dispatch time so
/// the runtime can construct a partial builder and `unwired_opnames()`
/// can survey gaps without terminating the process.
///
/// Named (not a closure) so `unwired_opnames()` can compare fn pointers
/// reliably — closures get a fresh anonymous type per call site.
fn unwired_handler_placeholder(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    // Grain never installs MiniMark. A walk that inlined a helper
    // (`malformed`) and aborted leaves the blackhole on that helper
    // body; those Charon-lowered opcodes are not in this table.
    // Leave to the portal merge point rather than panic. Hosts that
    // registered a MemoryError provider still fail loud.
    if majit_backend::memory_error_singleton_ref() == 0 {
        return Err(DispatchError::LeaveFrame);
    }
    panic!(
        "dispatch_step: unwired opcode pos={} entry={} table_len={} \
         jitcode={:?} index={:?} startpoint={} — extend the builder's \
         setup_insns to cover this opname",
        bh.last_opcode_position,
        bh.entry_position,
        bh.dispatch_table.len(),
        bh.jitcode.name,
        bh.jitcode.try_index(),
        bh.jitcode.is_valid_startpoint(bh.last_opcode_position),
    )
}

/// Return type of a blackhole frame.
///
/// RPython: `BlackholeInterpreter._return_type`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BhReturnType {
    Int,
    Ref,
    Float,
    #[default]
    Void,
}

/// Re-export BhDescr from codewriter::jitcode — shared descriptor type
/// between codewriter assembler and blackhole interpreter.
/// RPython `history.py:AbstractDescr` parity.
pub use majit_jitcode::jitcode::{BhCallDescr, BhDescr, DescrTable, EMPTY_DESCR_TABLE};

/// Per-jitdriver static data visible to the blackhole interpreter.
///
/// Mirrors the subset of `metainterp_sd.jitdrivers_sd[jdindex]` that
/// `bhimpl_recursive_call_*` and the jitcode::BC_RECURSIVE_CALL dispatch consult
/// (blackhole.py:1080-1099):
/// - `result_type` selects which `bhimpl_recursive_call_{v,i,r,f}`
///   method handles the call (blackhole.py:1080-1093).
/// - `portal_runner_ptr` / `mainjitcode_calldescr` feed
///   `get_portal_runner` (blackhole.py).
#[derive(Clone, Debug, Default)]
pub struct BhJitDriverSd {
    /// warmspot.py:449 `jd.result_type` projected to the blackhole
    /// dispatch char ('v','i','r','f') via `BhReturnType`.
    pub result_type: BhReturnType,
    /// warmspot.py:1010-1013 `jd.portal_runner_ptr` — the portal runner as a
    /// raw C function matching `mainjitcode_calldescr`. `get_portal_runner`
    /// hands its address to `bh_call_r`, which invokes it via the calldescr
    /// (`"iirrr"`: `next_instr:i, is_being_profiled:i, pycode:r, frame:r,
    /// ec:r`), so the pointee must take those five scalars by the C ABI.
    pub portal_runner_ptr: Option<extern "C" fn(i64, i64, i64, i64, i64) -> i64>,
    /// `jitdriver_sd.mainjitcode.calldescr` — CallDescr of the portal
    /// function returned by `get_portal_runner` for `bh_call_*`.
    pub mainjitcode_calldescr: BhCallDescr,
    /// warmspot.py `jd.handle_jitexc_from_bh`, one per jitdriver.
    /// `_handle_jitexception_in_portal` calls it on the driver whose
    /// `mainjitcode` is this frame's jitcode.
    pub handle_jitexc_from_bh: Option<PortalRunnerHook>,
}

/// How [`BlackholeInterpreter::run`]'s dispatch loop stopped.
///
/// `blackhole.py`'s `dispatch_loop` is `while True:` over `ord(code[position])`
/// and never returns — it only raises, and `run` converts `LeaveFrame` into a
/// normal return while letting everything else through.  So upstream has three
/// exits, and each one leaves different state behind for `_resume_mainloop` to
/// read.  Naming them keeps the caller from having to infer which happened.
///
/// The fourth, [`Self::EndOfCode`], has no upstream counterpart: reading past
/// the last byte is an `IndexError` there.  It exists here because pyre's unit
/// builders emit jitcodes that simply end, and it is a distinct variant rather
/// than a synonym for [`Self::LeaveFrame`] because the two differ in exactly
/// the state a caller goes on to publish — `return_type` and `tmpreg_*` are
/// written by a `*_return` handler and cleared by neither `cleanup_registers`
/// (upstream has no such slot to clear) nor `release_interp`, so a frame that
/// merely ran out of bytes carries the previous occupant of its pooled
/// interpreter to the caller's result register.
#[derive(Debug)]
pub enum BhRunOutcome {
    /// A `*_return` handler raised `LeaveFrame`: `return_type` and the matching
    /// `tmpreg_*` hold this frame's result.
    LeaveFrame,
    /// An exception no `catch_exception` in this frame handled;
    /// `got_exception` and `exception_last_value` carry it.
    Exception,
    /// `blackhole.py:1068` `raise ContinueRunningNormally(*args)`.
    ContinueRunningNormally(Box<MergePointArgs>),
    /// The dispatch loop ran out of bytecode without returning.  No upstream
    /// counterpart — see the type's own docs.
    EndOfCode,
}

/// Signal raised by handlers and propagated up through `dispatch_step` to `run`.
///
/// RPython: `LeaveFrame` exception + `except Exception` in blackhole.py run()
#[derive(Debug)]
pub enum DispatchError {
    /// Normal return from frame (RPython: LeaveFrame).
    LeaveFrame,
    /// Exception raised — must call handle_exception_in_frame.
    ///
    /// `exc` is the exception value (GcRef pointer as i64).  `resume_position`
    /// is the first byte past the raising instruction, which
    /// [`BlackholeInterpreter::dispatch_step`] stores into `self.position`
    /// before the error propagates: `blackhole.py` `_get_method` wraps every
    /// `bhimpl_*` in `except Exception`, advances `position` past the result
    /// register byte, assigns `self.position = position` and only then
    /// re-raises, so `handle_exception_in_frame` reads the coordinate of the
    /// instruction *after* the one that raised.  The field is part of the
    /// variant rather than a convention each producer remembers, because a
    /// producer that forgot it left `self.position` naming an operand byte and
    /// the handler search started one instruction short.
    RaiseException { exc: i64, resume_position: usize },
    /// blackhole.py:1068-1069: raise ContinueRunningNormally(*args)
    /// Bottommost blackhole reached the merge point — restart portal.
    ContinueRunningNormally(Box<MergePointArgs>),
}

/// Jitcode-based blackhole interpreter.
///
/// Executes jitcode bytecodes with concrete values. Each instance
/// represents one execution frame. Frame chain is linked via
/// `nextblackholeinterp`.
///
/// RPython: `BlackholeInterpreter` class in blackhole.py.
///
/// RPython `__init__` receives `builder` and stores:
///   self.cpu = builder.cpu
///   self.dispatch_loop = builder.dispatch_loop
///   self.descrs = builder.descrs
///   self.op_catch_exception = builder.op_catch_exception
///   self.op_rvmprof_code = builder.op_rvmprof_code
#[derive(Clone, Copy)]
struct BlackholeCpuRef(*const (dyn majit_backend::Backend + 'static));

impl BlackholeCpuRef {
    /// Erase the borrow lifetime without manufacturing a long-lived Rust
    /// reference.  The builder is leased only while its owning MetaInterp CPU
    /// is alive; every dereference is tied to the interpreter borrow below.
    fn new(cpu: &dyn majit_backend::Backend) -> Self {
        let ptr = cpu as *const dyn majit_backend::Backend;
        // SAFETY: raw trait-object pointers carry a lifetime bound only at the
        // type level. `BlackholeInterpBuilder::set_cpu` documents and enforces
        // the owner relationship at each production lease.
        Self(unsafe {
            std::mem::transmute::<
                *const dyn majit_backend::Backend,
                *const (dyn majit_backend::Backend + 'static),
            >(ptr)
        })
    }

    fn get(&self) -> &dyn majit_backend::Backend {
        // SAFETY: callers can obtain this only through a live builder/interp
        // whose active lease is nested inside the CPU owner's lifetime.
        unsafe { &*self.0 }
    }
}

pub struct BlackholeInterpreter {
    /// RPython `BlackholeInterpreter.__init__`: `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    cpu: Option<BlackholeCpuRef>,
    /// RPython `blackhole.py:288` `self.descrs = builder.descrs`.
    /// Descriptor table from the assembler. In RPython, `descrs` is a list
    /// of `AbstractDescr` objects carrying field offsets, array item sizes,
    /// etc. In pyre, we store raw offsets (usize) as a simplification —
    /// descriptor-index argcode ('d', 2 bytes) indexes into this table.
    /// `blackhole.py` binds `builder.descrs` by reference and :102-103
    /// stores the assembler list itself, so the table is shared and never
    /// copied; :154 is its only consumer and only reads. This has the same
    /// lifetime shape as the sibling `cpu: Option<&'static dyn Backend>`.
    pub descrs: &'static dyn DescrTable,
    /// RPython `blackhole.py` `self.op_catch_exception = builder.op_catch_exception`.
    /// `BC_ABSENT` when the key is missing — see [`BlackholeInterpBuilder::op_live`].
    pub op_catch_exception: u8,
    /// RPython `blackhole.py:290` `self.op_rvmprof_code = builder.op_rvmprof_code`.
    /// `BC_ABSENT` when the key is missing — see [`BlackholeInterpBuilder::op_live`].
    pub op_rvmprof_code: u8,
    /// RPython `blackhole.py:289` `self.op_live = builder.op_live`.
    /// `BC_ABSENT` when the key is missing — see [`BlackholeInterpBuilder::op_live`].
    pub op_live: u8,
    /// Integer register bank.
    /// Indices 0..num_regs_i are working registers.
    /// Indices num_regs_i..num_regs_i+constants_i.len() hold constants.
    pub registers_i: Vec<i64>,
    /// Reference register bank.
    pub registers_r: Vec<i64>,
    /// Float register bank.
    pub registers_f: Vec<i64>,
    /// Temporary register for int return value.
    pub tmpreg_i: i64,
    /// Temporary register for ref return value.
    pub tmpreg_r: i64,
    /// Temporary register for float return value.
    pub tmpreg_f: i64,
    /// Current jitcode being executed.
    pub jitcode: std::sync::Arc<JitCode>,
    /// Current bytecode position (program counter).
    pub position: usize,
    /// Jitcode pc of the last `-live-` marker this frame executed.
    /// `usize::MAX` when none has run since `setposition` / `release_interp`.
    /// `walk_bh_regs` reads it at collection time and roots only the marker's
    /// live Ref registers (`liveness.py`). `bhimpl_live` itself only records
    /// this and skips `OFFSET_SIZE`.
    last_live_pc: usize,
    /// Caller frame in the blackhole frame chain.
    pub nextblackholeinterp: Option<Box<BlackholeInterpreter>>,
    /// Whether `acquire_interp` registered this interpreter's traced fields
    /// with `shadow_stack::register_bh_interp` for its whole life (the
    /// pooled, boxed case). `run()` roots an unregistered one per call.
    rooted: bool,
    /// Free-list link owned by `BlackholeInterpBuilder.blackholeinterps`.
    /// This is distinct from `nextblackholeinterp`, exactly as upstream's
    /// `back` pool link is distinct from the live caller-frame chain.
    back: Option<Box<BlackholeInterpreter>>,
    /// Return type of this frame.
    pub return_type: BhReturnType,
    /// True when run() hit jitcode::BC_ABORT (unsupported bytecode). Callers
    /// must not treat abort as DoneWithThisFrame — side effects from
    /// partial execution have corrupted state.
    pub aborted: bool,
    /// True when the abort came from pyre's `abort_permanent` marker.  That
    /// marker is a clean bail to the interpreter after the codewriter has
    /// already materialized the frame's resume coordinate.
    pub abort_permanent_bail: bool,
    /// Name of the residual whose unbound call produced this bail.
    /// `None` until a reject site or `resume_mainloop` fills it.
    pub bail_residual: Option<String>,
    /// RPython blackhole.py handle_exception_in_frame parity:
    /// True when a residual call raised an exception (returned NULL ref).
    /// Unlike `aborted`, this indicates a Python-level exception that
    /// should propagate up the blackhole chain, not a JIT infrastructure error.
    pub got_exception: bool,
    /// Position of the last dispatched opcode (before position advances past operands).
    /// Used by handle_exception_in_frame for handler lookup — the faulting instruction
    /// PC, not the next instruction PC. Public so caller-chain propagation in
    /// call_jit.rs can set it to the suspended caller's position.
    pub last_opcode_position: usize,
    /// Position this frame was last seeded at by `setposition`.
    ///
    /// Diagnostic-only: an opcode that reads a register nobody wrote is
    /// either a resume that landed here without restoring it or a forward
    /// run whose producing opcode misbehaved, and only the seed coordinate
    /// separates the two.
    pub entry_position: usize,
    /// blackhole.py:391 exception_last_value: the caught exception object.
    /// Set when handle_exception_in_frame finds a handler.
    /// Read by CheckExcMatch and other exception opcodes in the handler.
    pub exception_last_value: i64,
    /// Callback for the interpreter-level traceback record that occurs before
    /// an exception-table handler is entered.  The generic blackhole carries
    /// only raw frame/exception references; portal integrations install the
    /// callback that understands their frame and JitCode-PC layouts.
    pub record_caught_exception: Option<extern "C" fn(i64, i64, i64, i64)>,
    /// Pointer to this level's virtualizable object.
    ///
    /// The vable bytecodes do NOT read this. `handler_getfield_vable_*` and
    /// `handler_setfield_vable_*` take the struct as their own `r` operand at
    /// their own pc, the shape `bhimpl_getfield_vable_r` has; correspondingly
    /// the vable handlers never consult this field. This pointer
    /// serves only readers that have no operand to read — `record_frame_traceback`
    /// and the `on_enter_level` / `on_leave_level` rooting callbacks — so a
    /// wrong value here corrupts a traceback or a root, never a vable access.
    /// Set during blackhole setup from the guard failure's virtualizable ptr.
    pub virtualizable_ptr: i64,
    /// Pointer to the VirtualizableInfo describing field offsets.
    /// Used by vable bytecodes to compute memory offsets.
    pub virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    /// blackhole.py get_portal_runner / warmspot.py:1010-1013 — per-jitdriver  allow-line-citation
    /// static data indexed by `jdindex`.  Each entry carries the
    /// portal_runner_ptr, mainjitcode.calldescr and result_type that
    /// `get_portal_runner` and the jitcode::BC_RECURSIVE_CALL dispatch consult
    /// (blackhole.py selects `bhimpl_recursive_call_{v,i,r,f}`
    /// from `sd.jitdrivers_sd[jdindex].result_type`).
    ///
    /// Replaces the prior single-driver flat fields
    /// (`portal_runner_ptr`/`mainjitcode_calldescr`) so multi-driver
    /// dispatch matches upstream line-by-line.
    ///
    /// Upstream reads the table through `self.builder.metainterp_sd`
    /// (blackhole.py `self.metainterp_sd = metainterp_sd`), so it is
    /// builder-shared and every interpreter in a chain sees it by
    /// construction.  A pool-owned Rust interpreter cannot hold a `&builder`
    /// back-reference, so this is the builder's snapshot, shared in by
    /// `acquire_interp` the same way `descrs` is (blackhole.py).  Set it
    /// via [`BlackholeInterpBuilder::setup_jitdrivers_sd`] before acquiring,
    /// not on individual frames.  `Arc<[_]>` rather than `Vec`: upstream's
    /// back-reference costs nothing to hand on, and both routes that give a
    /// frame its context — `acquire_interp` and [`Self::clone_context_from`]
    /// — hand it on.
    pub jitdrivers_sd: std::sync::Arc<[BhJitDriverSd]>,
    /// Pyre: absolute start index of the operand stack in
    /// PyFrame.locals_cells_stack_w. RPython does not need this because
    /// its JitCode already operates in register space.
    pub virtualizable_stack_base: usize,
    /// RPython `blackhole.py` `self.dispatch_loop = builder.dispatch_loop`.
    /// Per-instance reference to the builder's dispatch table.  Cloned
    /// from the builder via `Arc::clone` in `acquire_interp` so wired
    /// handler lookup is one indirect call (`self.dispatch_table[opcode]`)
    /// without going back to the builder.
    ///
    /// `Default::default()` starts with an empty table; the builder owns the
    /// wired table and propagates an `Arc` clone via `acquire_interp`, and a
    /// nested inline callee gets the parent's through
    /// [`Self::clone_context_from`].
    pub(crate) dispatch_table: std::sync::Arc<Vec<BhOpcodeHandler>>,
    /// Callee interpreter kept for the next `BC_INLINE_CALL` made from this
    /// frame, so a recursive descent pays for its frames once instead of once
    /// per call.  Upstream has no counterpart because `bhimpl_inline_call_*`
    /// (`blackhole.py`) allocates no callee interpreter at all — it reaches
    /// the callee through `cpu.bh_call_*(jitcode.fnaddr, ...)`.  pyre
    /// byte-interprets sub-jitcodes, so it needs a frame; this makes that
    /// frame as close to free as upstream's absence of one.  Handed back by
    /// `handler_inline_call_nested_ext` through
    /// [`Self::reset_for_inline_reuse`].
    ///
    /// That reset does NOT restore `Default::default()`, and reading it as if
    /// it did is how `state_field_layout` came to be missing from
    /// [`Self::clone_context_from`].  `position` / `last_opcode_position` /
    /// `entry_position` / `jitcode` survive it and are overwritten by the
    /// [`Self::setposition`] the take path always runs; `dispatch_table` and
    /// the vable fields survive it and are overwritten by
    /// [`Self::clone_context_from`]; `inline_callee_scratch` survives it on
    /// purpose, which is what makes a grandchild frame free too.  A reused
    /// frame is indistinguishable from a fresh one because those THREE steps
    /// together cover every field, not because any one of them does.
    pub(crate) inline_callee_scratch: Option<Box<BlackholeInterpreter>>,
    /// Per-kind argument lists for the native arm of `BC_INLINE_CALL`.
    ///
    /// RPython `BlackholeInterpreter.bhimpl_inline_call_irf_*` receives the
    /// already-built `args_i` / `args_r` / `args_f` lists and immediately
    /// hands them to `cpu.bh_call_*`.  pyre's `inline_call_nested_ext`
    /// adapter has to regroup its `(kind, caller_src, callee_dst)` triples
    /// into those lists.  Keeping the backing stores on the interpreter
    /// preserves the upstream per-interpreter ownership and avoids three
    /// system-heap allocations on every blackhole inline call.  They are a
    /// disposable cache only: no identity or value survives a call.
    pub(crate) native_inline_args_i_scratch: Vec<i64>,
    pub(crate) native_inline_args_r_scratch: Vec<i64>,
    pub(crate) native_inline_args_f_scratch: Vec<i64>,
    /// Backing store for the six lists decoded by `bhimpl_jit_merge_point`.
    /// RPython creates those lists in the GC nursery, where each allocation is
    /// a bump-pointer advance. Rust `Vec` uses the process allocator instead;
    /// retain the box and its six capacities on the already-pooled blackhole
    /// frame and recover them once the `ContinueRunningNormally` handoff has
    /// been consumed. The list values and lifetime remain per handoff; only
    /// disposable backing capacity survives, like the native-inline argument
    /// scratch above. `None` until the frame's first merge point, and again
    /// while a handoff is outstanding.
    merge_point_args_scratch: Option<Box<MergePointArgs>>,
    /// State-field JIT register layout (`StateFieldLayout`).  Set on the
    /// root resume frame from `JitState::state_field_layout` so the
    /// `state_field` handlers can map a logical field/array index to the
    /// flat register slot the resume reader seeded.  Empty (no scalars /
    /// arrays) for pyre frames, which never dispatch `state_field` opcodes.
    /// An inline callee inherits the caller's, because a `split_dispatch`
    /// sub-jitcode dispatches the SAME machine's state fields — see
    /// [`Self::clone_context_from`], which used to leave it blanked.
    pub state_field_layout: StateFieldLayout,
    /// Address of this thread's [`BH_LAST_EXC_VALUE`] cell. Residual and
    /// inline-call handlers clear and read that cell once per host call;
    /// resolving the thread-local each time was `_tlv_get_addr` on every
    /// native `inline_call` (regex `shift`). Bound in [`Default`] and
    /// valid for the interpreter's whole life on this thread.
    last_exc_cell: *const std::cell::Cell<i64>,
    /// [`BlackholeInterpBuilder::shared_fields_gen`] at the last
    /// [`BlackholeInterpBuilder::bind_shared_fields`]. A pool hit whose
    /// generation still matches skips the six builder-shared copies
    /// (`blackhole.py` `BlackholeInterpBuilder.acquire_interp` returns the
    /// pooled object with the fields `__init__` already wrote).
    bound_gen: u32,
}

// blackhole.py: last exception value from a residual call.
// Set by pyre call helpers (bh_call_fn_impl etc.) on error.
// Read by handler dispatch to populate exception_last_value.
thread_local! {
    pub static BH_LAST_EXC_VALUE: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };

    /// llmodel.py `grab_exc_value(deadframe)`: the exception a failing
    /// guard carried, parked for the bridge / blackhole handoff.
    ///
    /// Grabbing the value reads `jf_guard_exc` off the deadframe and drops the
    /// jitframe, which was the collector's only handle on the exception
    /// (`jitframe_trace`). The handoff then reconstructs the resume state
    /// through the blackhole allocator before anything re-roots the value, so
    /// in that window the exception — and the young `args` / `__traceback__`
    /// reachable only through it — live behind a bare `i64`. RPython's
    /// `grab_exc_value` result is a shadowstack-rooted local across the same
    /// span; pyre has no GC transform, so the frontend registers a root walker
    /// over this cell instead, and carries the cell's address in the
    /// per-mutator root area next to `BH_LAST_EXC_VALUE` so a collection
    /// started by another thread still reaches a stopped mutator's parked
    /// exception.
    ///
    /// Thread-local is the faithful shape rather than a concession: upstream
    /// keeps one shadow stack per thread and switches between them on GIL
    /// hand-off (`shadowstack.py`, a `{tid: SHADOWSTACKREF}` map), and
    /// enumerates every thread's block at collection time instead of resolving
    /// the thread local on whichever thread started collecting
    /// (`rthread.py _trace_tlref`). This cell plus the per-mutator area
    /// is that pair, hand-materialized for one live value.
    pub static GUARD_EXC_VALUE: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// Park a grabbed guard exception in [`GUARD_EXC_VALUE`] for the duration of
/// one handoff.
///
/// Restores the previous value on drop rather than clearing, so a nested
/// handoff (a bridge trace that itself deopts) unwinds to the exception its
/// caller is still carrying. The displaced value stays a root while it is
/// set aside: a nursery exception can move during the nested handoff, and
/// restoring the copy taken at `park` would put the pre-move address back
/// into a walked cell.
pub struct GuardExcRoot {
    prev: Option<majit_gc::shadow_stack::OwnerRootGuard>,
}

impl GuardExcRoot {
    pub fn park(exc: i64) -> Self {
        let prev = GUARD_EXC_VALUE.with(|cell| cell.replace(exc));
        Self {
            prev: (prev != 0)
                .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(GcRef(prev as usize))),
        }
    }

    /// The parked exception's current address while this park is the
    /// innermost one. The collector rewrites the cell in place, so this is
    /// the forwarded value after a collection.
    pub fn get(&self) -> i64 {
        GUARD_EXC_VALUE.with(|cell| cell.get())
    }
}

impl Drop for GuardExcRoot {
    fn drop(&mut self) {
        let prev = self.prev.as_ref().map_or(0, |root| root.get().0 as i64);
        GUARD_EXC_VALUE.with(|cell| cell.set(prev));
    }
}

// rvmprof integration lives in the `rpython.rlib.rvmprof.cintf` analog.
// Blackhole calls through `rvmprof::cintf::jit_rvmprof_code` directly,
// matching `blackhole.py:416, 438, 1600` where the C intf function is
// invoked without any hook-registry indirection visible to dispatch code.
use majit_rlib::rvmprof::cintf::jit_rvmprof_code;

thread_local! {
    /// The placeholder `jitcode` every `BlackholeInterpreter::default()`
    /// starts with, shared instead of rebuilt.  It carries an empty
    /// `JitCodeBody` and is only ever read — `cleanup_registers` asks it for
    /// `num_regs_r()` before the first `setposition`, which then replaces it
    /// — so one instance per thread answers for every interpreter.  Building
    /// a fresh one per default cost a `JitCode` allocation, a `JitCodeBody`
    /// commit and, on drop, an `EffectInfo` teardown, once per
    /// `BC_INLINE_CALL`.
    static EMPTY_JITCODE: std::sync::Arc<JitCode> = std::sync::Arc::new(JitCode::default());
}

impl Drop for BlackholeInterpreter {
    fn drop(&mut self) {
        if self.rooted {
            majit_gc::shadow_stack::unregister_bh_interp(&self.registers_r);
        }
    }
}

impl Default for BlackholeInterpreter {
    /// Sentinel-value interpreter used by
    /// `BlackholeInterpBuilder::acquire_interp`'s `unwrap_or_default()`
    /// and by `handler_inline_call_nested_ext` as the receiver for
    /// `clone_context_from` when this frame has no
    /// `inline_callee_scratch` to reuse yet.  The 6 builder-shared fields stay at
    /// `u8::MAX` / empty until either `acquire_interp` populates them
    /// (RPython `blackhole.py` parity) or
    /// `clone_context_from(parent)` copies them from a parent
    /// interpreter.
    fn default() -> Self {
        Self {
            cpu: None,
            // blackhole.py `EMPTY_LIST_I = [] # shared`.
            descrs: EMPTY_DESCR_TABLE,
            // RPython blackhole.py — copied from builder in `acquire_interp`.
            // `BC_ABSENT` stands in for `insns.get('…', -1)`: a reserved byte
            // no opname is ever assigned, so it cannot match `code[position]`.
            op_catch_exception: majit_jitcode::insns::BC_ABSENT,
            op_rvmprof_code: majit_jitcode::insns::BC_ABSENT,
            op_live: majit_jitcode::insns::BC_ABSENT,
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
            tmpreg_i: 0,
            tmpreg_r: 0,
            tmpreg_f: 0,
            jitcode: EMPTY_JITCODE.with(std::sync::Arc::clone),
            position: 0,
            last_live_pc: usize::MAX,
            nextblackholeinterp: None,
            rooted: false,
            back: None,
            return_type: BhReturnType::Void,
            aborted: false,
            abort_permanent_bail: false,
            bail_residual: None,
            got_exception: false,
            last_opcode_position: 0,
            entry_position: 0,
            exception_last_value: 0,
            record_caught_exception: None,
            virtualizable_ptr: 0,
            virtualizable_info: std::ptr::null(),
            jitdrivers_sd: std::sync::Arc::from([] as [BhJitDriverSd; 0]),
            virtualizable_stack_base: 0,
            dispatch_table: std::sync::Arc::new(vec![
                unwired_handler_placeholder as BhOpcodeHandler;
                256
            ]),
            inline_callee_scratch: None,
            native_inline_args_i_scratch: Vec::new(),
            native_inline_args_r_scratch: Vec::new(),
            native_inline_args_f_scratch: Vec::new(),
            merge_point_args_scratch: None,
            state_field_layout: StateFieldLayout::default(),
            last_exc_cell: BH_LAST_EXC_VALUE.with(|c| c as *const std::cell::Cell<i64>),
            bound_gen: 0,
        }
    }
}

/// `history.py` `getint` / `getref_base` / `getfloatstorage` exist only on
/// the box class of that bank. `BlackholeInterpreter._copy_data_from_miframe`
/// calls the matching getter; an int box in `registers_r` is a tracer bug.
///
fn expect_box_bank(
    miframe: &crate::pyjitpl::MIFrame,
    index: usize,
    bank: majit_ir::Type,
    regs: &[Option<majit_ir::OpRef>],
) {
    let Some(box_ref) = regs.get(index).copied().flatten() else {
        return;
    };
    if box_ref.ty() == Some(bank) {
        return;
    }
    panic!(
        "copy_data_from_miframe: jitcode {} pc {} bank {bank:?} register {index} box type {:?}",
        miframe.jitcode.name,
        miframe.pc,
        box_ref.ty()
    );
}

/// `jitcode.py` `enumerate_vars` walks three banks. A live index at or past
/// `num_regs_*` of that bank is a register listed in the wrong bank.
pub(crate) fn expect_liveness_bank(
    jitcode_name: &str,
    pc: usize,
    bank: majit_ir::Type,
    index: u32,
    num_regs: usize,
) {
    if (index as usize) < num_regs {
        return;
    }
    panic!(
        "liveness: jitcode {jitcode_name} pc {pc} bank {bank:?} register {index} is outside that bank ({num_regs} registers)"
    );
}

impl BlackholeInterpreter {
    /// This thread's [`BH_LAST_EXC_VALUE`] cell, without a TLS resolve.
    #[inline]
    fn last_exc(&self) -> &std::cell::Cell<i64> {
        // Bound in `Default` to this thread's cell; pooled interpreters
        // never leave the thread that constructed them.
        unsafe { &*self.last_exc_cell }
    }
}

impl BlackholeInterpreter {
    /// Return a consumed `ContinueRunningNormally` payload to this pooled
    /// frame. Values are cleared immediately; only the six Vec capacities are
    /// retained for the next merge-point decode.
    pub(crate) fn recycle_merge_point_args(&mut self, mut args: Box<MergePointArgs>) {
        args.green_int.clear();
        args.green_ref.clear();
        args.green_float.clear();
        args.red_int.clear();
        args.red_ref.clear();
        args.red_float.clear();
        self.merge_point_args_scratch = Some(args);
    }

    /// The `self.position = position` that closes `blackhole.py setposition`,
    /// plus the two pyre-only diagnostic cursors derived from the same
    /// coordinate.
    ///
    /// Nothing else: upstream's `setposition` assigns `jitcode` and `position`
    /// and touches no other state.  `aborted` / `abort_permanent_bail` /
    /// `got_exception` are reset where an interpreter is handed back
    /// ([`BlackholeInterpBuilder::release_interp`]) and start `false` on a
    /// fresh one, and `exception_last_value` is cleared by
    /// [`Self::cleanup_registers`] — which is upstream's owner for it too.
    /// Clearing it here dropped the exception a caller had already been
    /// handed whenever a frame was re-seated before its handler ran.
    fn reset_position_state(&mut self, position: usize) {
        self.position = position;
        self.last_opcode_position = position;
        self.entry_position = position;
        self.last_live_pc = usize::MAX;
    }

    /// `blackhole.py setposition`'s per-bank half:
    ///
    /// ```python
    /// if num_regs_and_consts_i:
    ///     if self.registers_i is None or len(self.registers_i) < num_regs_and_consts_i:
    ///         self.registers_i = [default_i] * num_regs_and_consts_i
    ///     self.copy_constants(self.registers_i, jitcode.constants_i, jitcode.num_regs_i())
    /// ```
    ///
    /// Grow only when the bank is too small; `copy` then writes exactly
    /// `nconsts` entries at `num_regs` (`blackhole.py BlackholeInterpreter.copy_constants`).
    /// A jitcode that declares no registers of a kind leaves that bank alone.
    fn init_register_file(
        regs: &mut Vec<i64>,
        num_regs: usize,
        nconsts: usize,
        copy: impl FnOnce(&mut [i64]),
    ) {
        let need = num_regs + nconsts;
        if need == 0 {
            return;
        }
        if regs.len() < need {
            regs.clear();
            regs.resize(need, 0);
        }
        if nconsts != 0 {
            copy(&mut regs[num_regs..num_regs + nconsts]);
        }
    }

    fn init_register_file_from_i64s(regs: &mut Vec<i64>, num_regs: usize, constants: &[i64]) {
        Self::init_register_file(regs, num_regs, constants.len(), |dst| {
            dst.copy_from_slice(constants);
        });
    }

    fn init_register_file_from_const_slots_r(
        regs: &mut Vec<i64>,
        num_regs: usize,
        constants: &[majit_jitcode::jitcode::ConstSlotR],
    ) {
        Self::init_register_file(regs, num_regs, constants.len(), |dst| {
            for (dst, slot) in dst.iter_mut().zip(constants) {
                *dst = slot.get();
            }
        });
    }

    fn init_register_files_from_runtime_jitcode(&mut self, jitcode: &JitCode) {
        // One `body()` so a same-helper reseat does not pay nine
        // `OnceLock` walks (`num_regs_*` / `num_regs_and_consts_*` /
        // `constants_*` each Deref through the body).
        let body = jitcode.body();
        Self::init_register_file_from_i64s(
            &mut self.registers_i,
            body.c_num_regs_i as usize,
            &body.constants_i,
        );
        Self::init_register_file_from_const_slots_r(
            &mut self.registers_r,
            body.c_num_regs_r as usize,
            &body.constants_r,
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_f,
            body.c_num_regs_f as usize,
            &body.constants_f,
        );
    }

    /// RPython `blackhole.py:313-337` register-file setup against the
    /// canonical codewriter `JitCode`.
    ///
    /// This intentionally stops short of installing `self.jitcode` for
    /// dispatch: pyre's execution path still expects the runtime adapter
    /// `crate::jitcode::JitCode` for `exec.*` pools. The extracted helper
    /// keeps the upstream-common part of `setposition` usable without a
    /// build→runtime conversion layer.
    #[cfg(test)]
    pub(crate) fn prepare_registers_for_canonical_jitcode(
        &mut self,
        jitcode: &majit_jitcode::jitcode::JitCode,
        position: usize,
    ) {
        let body = jitcode.body();
        Self::init_register_file_from_i64s(
            &mut self.registers_i,
            body.c_num_regs_i as usize,
            &body.constants_i,
        );
        Self::init_register_file_from_const_slots_r(
            &mut self.registers_r,
            body.c_num_regs_r as usize,
            &body.constants_r,
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_f,
            body.c_num_regs_f as usize,
            &body.constants_f,
        );
        self.reset_position_state(position);
    }

    /// Restore an inline callee frame so [`Self::inline_callee_scratch`] can
    /// hand it to the next `BC_INLINE_CALL`.
    ///
    /// `blackhole.py BlackholeInterpreter.cleanup_registers` NULLs only the
    /// first `num_regs_r()` REF registers and `exception_last_value`. It
    /// never touches int/float registers and never clears constants. The
    /// earlier whole-bank `fill(0)` wiped the constant area and forced
    /// [`Self::setposition_ref`] to recopy it on every seating of the same
    /// helper. Working int/float leftovers are overwritten by the next
    /// `setposition` + argument copy the same way a pooled interp is.
    pub(crate) fn reset_for_inline_reuse(&mut self) {
        let n_r = self.jitcode.num_regs_r().min(self.registers_r.len());
        self.registers_r[..n_r].fill(0);
        self.tmpreg_i = 0;
        self.tmpreg_r = 0;
        self.tmpreg_f = 0;
        self.exception_last_value = 0;
        self.nextblackholeinterp = None;
        self.back = None;
        self.return_type = BhReturnType::Void;
        self.aborted = false;
        self.abort_permanent_bail = false;
        self.bail_residual = None;
        self.got_exception = false;
        self.state_field_layout = StateFieldLayout::default();
    }

    /// Copy the builder-shared context fields from `parent` onto `self`.
    /// Mirrors `BlackholeInterpBuilder::acquire_interp` in this file
    /// — the same 6 fields that builder→interp normally propagates.
    /// Used by `handler_inline_call_nested_ext` to give the callee
    /// `BlackholeInterpreter` the same `dispatch_table` (so a nested
    /// `BC_INLINE_CALL` byte routes through the same handler) plus the
    /// CPU/descrs/op_* slots the wired handlers would consult.  pyre-only:
    /// RPython `bhimpl_inline_call_*` (blackhole.py) does not
    /// allocate a callee interpreter; it calls `cpu.bh_call_*(jitcode.fnaddr,
    /// ...)` directly because RPython's inline_call jitcode carries a
    /// native entry point.  pyre's sub-jitcodes are byte-interpreted, so
    /// the callee path needs an interpreter that shares the parent's
    /// dispatch table to keep recursive inline_call working.
    pub fn clone_context_from(&mut self, parent: &Self) {
        // Six builder-shared fields per `BlackholeInterpBuilder::acquire_interp`
        // in this file.
        self.cpu = parent.cpu;
        self.descrs = parent.descrs;
        self.op_catch_exception = parent.op_catch_exception;
        self.op_rvmprof_code = parent.op_rvmprof_code;
        self.op_live = parent.op_live;
        self.dispatch_table = std::sync::Arc::clone(&parent.dispatch_table);
        // Virtualizable / jitdriver state: RPython `bhimpl_inline_call_*`
        // (`blackhole.py:1278-1320`) reaches the callee via
        // `cpu.bh_call_*(jitcode.fnaddr, ...)` so the callee never sees a
        // BlackholeInterpreter context at all — vable / recursive_call
        // bytecodes are unreachable through that path upstream.  pyre's
        // nested interpreter executes the sub-jitcode byte-by-byte, so a
        // sub-jitcode that contains a vable / recursive_call opcode would
        // otherwise read the callee's zero-defaults.  Mirror the parent's
        // state defensively — the closest equivalent to RPython's "callee
        // never has its own context" shape.
        self.virtualizable_ptr = parent.virtualizable_ptr;
        self.virtualizable_info = parent.virtualizable_info;
        self.jitdrivers_sd = std::sync::Arc::clone(&parent.jitdrivers_sd);
        self.virtualizable_stack_base = parent.virtualizable_stack_base;
        self.record_caught_exception = parent.record_caught_exception;
        // `state_field_layout` belongs to the same group and was missing from
        // it. A sub-jitcode's `load_state_field` reads
        // `state_field_layout.scalar_slot(field_idx)`, and the reused callee
        // frame arrives with the layout `reset_for_inline_reuse` blanked, so
        // without this line the slot resolves through an all-zero default:
        // `int_scalar_base + field_idx` = `0 + field_idx`, which addresses the
        // sub-jitcode's own ARGUMENT registers. The range check that would
        // have caught it is on the layout, not here, and it is an `assert!`
        // only since this line was written.
        //
        // The layout describes the machine, not the frame — a sub-jitcode
        // dispatches the same state fields the portal does — so the parent's
        // is the right one, exactly as for the vable handles above.
        self.state_field_layout = parent.state_field_layout.clone();
    }

    /// blackhole.py setposition
    ///
    /// Initialize register arrays for a jitcode and set the position.
    /// Allocates registers sized to hold both working regs and constants,
    /// then copies constants into the upper portion of each register array.
    pub fn setposition(&mut self, jitcode: std::sync::Arc<JitCode>, position: usize) {
        self.setposition_ref(&jitcode, position);
    }

    /// `setposition` without taking the `Arc`.
    ///
    /// Bank setup runs on every call. `blackhole.py
    /// BlackholeInterpreter.setposition` grows a bank only when it is too
    /// small (`len(registers_*) < num_regs_and_consts_*`), then
    /// `copy_constants` of `len(constants)` entries at `num_regs_*`.
    /// [`Self::init_register_file`] is that length gate, so an
    /// unconditional [`Self::init_register_files_from_runtime_jitcode`]
    /// re-copies constants and grows only when the bank is too small.
    /// The `Arc::ptr_eq` test skips only the `Arc` clone when the jitcode
    /// is unchanged.
    #[inline]
    pub fn setposition_ref(&mut self, jitcode: &std::sync::Arc<JitCode>, position: usize) {
        if !std::sync::Arc::ptr_eq(&self.jitcode, jitcode) {
            self.jitcode = std::sync::Arc::clone(jitcode);
        }
        self.init_register_files_from_runtime_jitcode(jitcode);
        self.reset_position_state(position);
        if crate::bh_debug_enabled() {
            eprintln!(
                "[bh-setpos] jitcode={:?} index={:?} position={position} code_len={}",
                self.jitcode.name,
                self.jitcode.try_index(),
                self.jitcode.code.len(),
            );
        }
    }

    /// blackhole.py `get_portal_runner(jdindex)`.
    ///
    /// ```python
    /// def get_portal_runner(self, jdindex):
    ///     jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    ///     fnptr = adr2int(jitdriver_sd.portal_runner_adr)
    ///     calldescr = jitdriver_sd.mainjitcode.calldescr
    ///     return fnptr, calldescr
    /// ```
    pub fn get_portal_runner(&self, jdindex: usize) -> (i64, BhCallDescr) {
        let jd = &self.jitdrivers_sd[jdindex];
        let fnptr = jd.portal_runner_ptr.map(|f| f as usize as i64).unwrap_or(0);
        (fnptr, jd.mainjitcode_calldescr.clone())
    }

    /// `blackhole.py bhimpl_recursive_call_i` — the four `bhimpl_recursive_call_{i,r,f,v}`
    /// share one prologue: resolve the driver's portal runner, then merge
    /// greens and reds per kind (`greens_i + reds_i`, ...) into the
    /// `bh_call_*` argument banks.
    ///
    /// A zero `fnptr` here would be an indirect call through a null address.
    /// `get_portal_runner` reports zero for a driver slot whose
    /// `portal_runner_adr` was never populated, and the recursive-call path is
    /// not optional, so reaching this point with zero means `jdindex` named the
    /// wrong slot of `jitdrivers_sd`.  Report the coordinates instead of
    /// faulting at PC 0.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn portal_runner_call_args(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> (i64, BhCallDescr, Vec<i64>, Vec<i64>, Vec<i64>) {
        let (fnptr, calldescr) = self.get_portal_runner(jdindex);
        assert!(
            fnptr != 0,
            "blackhole recursive_call: jitdrivers_sd[{jdindex}] carries no portal \
             runner ({} driver(s) registered, jitcode={}) — blackhole.py:1095-1099",
            self.jitdrivers_sd.len(),
            self.jitcode.name(),
        );
        let mut all_i = greens_i;
        all_i.extend(&reds_i);
        let mut all_r = greens_r;
        all_r.extend(&reds_r);
        let mut all_f = greens_f;
        all_f.extend(&reds_f);
        (fnptr, calldescr, all_i, all_r, all_f)
    }

    /// blackhole.py bhimpl_recursive_call_r:
    ///   fnptr, calldescr = self.get_portal_runner(jdindex)
    ///   return self.cpu.bh_call_r(fnptr, greens_i+reds_i, greens_r+reds_r,
    ///                             greens_f+reds_f, calldescr)
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn bhimpl_recursive_call_r(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> majit_ir::GcRef {
        // blackhole.py:1113-1116: greens + reds merged per kind.
        let (fnptr, calldescr, all_i, all_r, all_f) = self.portal_runner_call_args(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        );
        self.cpu()
            .bh_call_r(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// Set an integer register value.
    ///
    /// RPython: `BlackholeInterpreter.setarg_i(index, value)`
    ///
    /// The working registers occupy `0 .. num_regs_i`; everything above is the
    /// jitcode's constant table, which no write may reach.
    #[inline(always)]
    pub fn setarg_i(&mut self, index: usize, value: i64) {
        // `init_register_file_from_i64s` lays the jitcode's integer constants
        // out at `num_regs_i ..`, so a write at or above that bound replaces a
        // constant the instruction stream still reads as one.  The read side
        // cannot tell the difference, and the value surfaces far away — as a
        // bogus array index or callee address — so fail at the write.
        debug_assert_constant_slot_untouched(index, self.jitcode.num_regs_i(), "setarg_i");
        self.registers_i[index] = value;
    }

    /// Set a reference register value.
    ///
    /// Same constant-table bound as [`Self::setarg_i`]: a write at or past
    /// `num_regs_r` replaces a jitcode constant the instruction stream still
    /// reads as one.
    #[inline(always)]
    pub fn setarg_r(&mut self, index: usize, value: i64) {
        debug_assert_constant_slot_untouched(index, self.jitcode.num_regs_r(), "setarg_r");
        self.registers_r[index] = value;
    }

    /// Set a float register value.
    #[inline(always)]
    pub fn setarg_f(&mut self, index: usize, value: i64) {
        debug_assert_constant_slot_untouched(index, self.jitcode.num_regs_f(), "setarg_f");
        self.registers_f[index] = value;
    }

    /// Get the int return value from a completed frame.
    ///
    /// RPython: `BlackholeInterpreter.get_tmpreg_i()`
    pub fn get_tmpreg_i(&self) -> i64 {
        self.tmpreg_i
    }

    /// blackhole.py `get_tmpreg_r`.
    ///
    /// Reference scratch values are GC-visible while a blackhole is running,
    /// so consuming one must clear the slot immediately.  Leaving the last
    /// value here keeps an otherwise-dead object alive until the whole frame
    /// leaves blackhole mode.
    pub fn get_tmpreg_r(&mut self) -> i64 {
        let result = self.tmpreg_r;
        self.tmpreg_r = 0;
        result
    }

    pub fn get_tmpreg_f(&self) -> i64 {
        self.tmpreg_f
    }

    /// Copy register state from a tracing MIFrame into this blackhole frame.
    ///
    /// RPython: `BlackholeInterpreter._copy_data_from_miframe(miframe)`
    ///
    /// `registers_r[i].getref_base()` (`history.py`) is not defined on an
    /// int box. A box whose `AbstractValue.type` is not this bank's type
    /// is a tracer bug; stop here with the jitcode, pc, bank and index.
    pub fn copy_data_from_miframe(&mut self, miframe: &MIFrame) {
        self.setposition(miframe.jitcode.clone(), miframe.pc);
        for i in 0..self.jitcode.num_regs_i() {
            expect_box_bank(miframe, i, majit_ir::Type::Int, &miframe.int_regs);
            if let Some(val) = miframe.int_value_for_blackhole(i) {
                self.setarg_i(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_r() {
            expect_box_bank(miframe, i, majit_ir::Type::Ref, &miframe.ref_regs);
            if let Some(val) = miframe.ref_value_for_blackhole(i) {
                self.setarg_r(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_f() {
            expect_box_bank(miframe, i, majit_ir::Type::Float, &miframe.float_regs);
            if let Some(val) = miframe.float_value_for_blackhole(i) {
                self.setarg_f(i, val);
            }
        }
    }

    /// Whether this interpreter's traced fields are registered as GC roots
    /// for its whole life (`acquire_interp`).
    pub fn is_rooted(&self) -> bool {
        self.rooted
    }

    /// blackhole.py cleanup_registers
    ///
    /// Clear reference registers to avoid keeping objects alive.
    /// Does not clear constants (they are prebuilt).
    pub fn cleanup_registers(&mut self) {
        for i in 0..self.jitcode.num_regs_r() {
            if i < self.registers_r.len() {
                self.registers_r[i] = 0;
            }
        }
        // `tmpreg_r` is a reference slot on the same footing as `registers_r`:
        // `run()` hands both to `push_bh_regs`, and `walk_bh_regs` visits the
        // tmpreg word as a `GcRef` root.  Its live window is one opcode — the
        // gap between a call helper returning and the `registers_r[dst]` store
        // — but nothing resets it, so a released interp carries the last ref
        // return value into the pool.  Nothing roots it there, so the object
        // can be collected; the next `run()` on that pooled interp then pushes
        // the dead address as a root and the collector reads a header off it.
        // Upstream's `cleanup_registers` (`blackhole.py`) has no such slot
        // to clear: RPython's tmpreg is a traced field of a GC object.
        self.tmpreg_r = 0;
        self.exception_last_value = 0;
    }

    /// blackhole.py `get_current_position_info`.
    ///
    /// RPython returns an offset into `metainterp_sd.liveness_info`
    /// (via `jitcode.get_live_vars_info(self.position, self.builder.op_live)`),
    /// and so does this: `get_live_vars_info` reads the `-live-` marker out of
    /// `JitCode.code` at `position` and decodes the offset that follows it.
    /// There is no side table to disagree with the code stream, so the marker
    /// hook's register clear and a resume's register seeding read one source.
    pub fn get_current_position_info(&self) -> usize {
        self.jitcode.get_live_vars_info(self.position, self.op_live)
    }

    /// Jitcode pc of the last `-live-` marker, if this frame has executed one
    /// since it was seated. Collection uses it; `bhimpl_live` only stores it.
    pub fn last_live_marker_pc(&self) -> Option<usize> {
        if self.last_live_pc == usize::MAX {
            None
        } else {
            Some(self.last_live_pc)
        }
    }

    /// Result register of the call this frame is resuming after.
    ///
    /// `blackhole.py _setup_return_value_*` reads `code[position-1]`, the
    /// result operand of the `xxx_call_yyy` that the resume `-live-`
    /// follows.  Flatten emits that `-live-` immediately after every
    /// `inline_call` and every can-raise `residual_call`
    /// (`jtransform.py handle_regular_call` / `handle_residual_call`),
    /// and `inline_call_typed` closes on the destination byte.
    fn call_result_reg(&self, _kind: BhReturnType) -> usize {
        self.jitcode.code[self.position - 1] as usize
    }

    /// blackhole.py _setup_return_value_i
    ///
    /// Connect the return of values from the called frame to the
    /// 'xxx_call_yyy' instructions from the caller frame.
    /// blackhole.py _setup_return_value_i
    pub fn setup_return_value_i(&mut self, result: i64) {
        // blackhole.py:1655-1656
        let reg_idx = self.call_result_reg(BhReturnType::Int);
        self.registers_i[reg_idx] = result;
    }

    /// blackhole.py _setup_return_value_r
    pub fn setup_return_value_r(&mut self, result: i64) {
        // blackhole.py:1658-1659
        let reg_idx = self.call_result_reg(BhReturnType::Ref);
        self.registers_r[reg_idx] = result;
    }

    /// blackhole.py _setup_return_value_f
    pub fn setup_return_value_f(&mut self, result: i64) {
        // blackhole.py:1661-1662
        let reg_idx = self.call_result_reg(BhReturnType::Float);
        self.registers_f[reg_idx] = result;
    }

    /// blackhole.py _done_with_this_frame
    ///
    /// Rare case: the blackhole interps all returned normally
    /// (in general we get a ContinueRunningNormally exception).
    fn done_with_this_frame(&mut self) -> JitException {
        match self.return_type {
            BhReturnType::Void => JitException::DoneWithThisFrameVoid,
            BhReturnType::Int => JitException::DoneWithThisFrameInt(self.get_tmpreg_i()),
            BhReturnType::Ref => {
                JitException::DoneWithThisFrameRef(GcRef(self.get_tmpreg_r() as usize))
            }
            BhReturnType::Float => {
                JitException::DoneWithThisFrameFloat(f64::from_bits(self.get_tmpreg_f() as u64))
            }
        }
    }

    /// blackhole.py _exit_frame_with_exception
    fn exit_frame_with_exception(&self, exc: i64) -> JitException {
        JitException::ExitFrameWithExceptionRef(GcRef(exc as usize))
    }

    /// blackhole.py _prepare_resume_from_failure
    ///
    /// Extract exception from the CPU deadframe on guard failure.
    /// Returns the exception value (0 if none).
    pub fn prepare_resume_from_failure(deadframe_exc: i64) -> i64 {
        // RPython: lltype.cast_opaque_ptr(rclass.OBJECTPTR,
        //          self.cpu.grab_exc_value(deadframe))
        deadframe_exc
    }

    /// blackhole.py _resume_mainloop
    ///
    /// Execute one frame and handle its completion.
    /// Returns Ok(exc) where exc is the exception to propagate to caller (0 = none),
    /// or Err(JitException) for JIT-level control flow exits.
    pub fn resume_mainloop(&mut self, current_exc: i64) -> Result<i64, JitException> {
        // blackhole.py:1614-1618
        // If there is a current exception, raise it now
        // (it may be caught by a catch_operation in this frame)
        if current_exc != 0 && !self.handle_exception_in_frame(current_exc) {
            // No handler: propagate.  The exception leaves this frame here,
            // so record its node — see [`Self::record_frame_traceback`].
            // Publish into the walked slot first: the record allocates a
            // nursery traceback and the minor moves the exception. Re-read
            // the slot afterwards; `current_exc` is not rewritten.
            self.exception_last_value = current_exc;
            self.record_frame_traceback(self.exception_last_value);
            let current_exc = self.exception_last_value;
            if self.nextblackholeinterp.is_none() {
                return Err(self.exit_frame_with_exception(current_exc));
            }
            return Ok(current_exc);
        }

        // blackhole.py — run the bytecode.
        // blackhole.py `_resume_mainloop` does not catch
        // ContinueRunningNormally; it propagates out to `_run_forever`
        // and then to `handle_jitexception` (warmspot.py).
        match self.run() {
            BhRunOutcome::ContinueRunningNormally(args) => {
                return Err(JitException::ContinueRunningNormally(args));
            }
            // `blackhole.py:1633` reads `self._return_type` next, and it only
            // gets there because `dispatch_loop` raised `LeaveFrame` from a
            // `*_return` handler that had just written it.  Nothing writes it
            // on the way out of a frame that merely ran out of bytecode, and
            // nothing clears it either, so the read below would publish the
            // result of whatever ran on this pooled interpreter last.
            BhRunOutcome::EndOfCode => panic!(
                "resume_mainloop: jitcode {:?} index {:?} ended without a return \
                 opcode, so there is no result to pass to the caller",
                self.jitcode.name,
                self.jitcode.try_index(),
            ),
            BhRunOutcome::LeaveFrame | BhRunOutcome::Exception => {}
        }

        // Check for exception during execution
        if self.got_exception {
            let exc = self.exception_last_value;
            // `run` reached the end of the frame with an uncaught exception, so
            // the frame is left here — see [`Self::record_frame_traceback`].
            // The record moves the nursery exception. `exception_last_value`
            // is the walked root; the local is stale once it returns.
            self.record_frame_traceback(exc);
            let exc = self.exception_last_value;
            if self.nextblackholeinterp.is_none() {
                // blackhole.py:1629
                return Err(self.exit_frame_with_exception(exc));
            }
            return Ok(exc);
        }

        if self.aborted {
            if self.bail_residual.is_none() {
                self.bail_residual = Some(format!(
                    "blackhole abort in {} at {}",
                    self.jitcode.name(),
                    self.last_opcode_position
                ));
            }
            // No upstream counterpart — see `JitException::BailToInterpreter`.
            // The abort leaves this frame unfinished at every level, so it
            // propagates instead of being absorbed: `Ok(0)` would have popped
            // to the caller and resumed it right after the call whose callee
            // bailed, with the result register never written.
            //
            // One variant covers both aborts because only one of them is
            // resumable, and which one it is travels beside the exception in
            // `abort_permanent_bail` — on the frame here, and on the register
            // image the drivers hand out.  Every consumer checks that flag
            // before it treats the bailed frame as a resume point.
            return Err(JitException::BailToInterpreter);
        }

        // blackhole.py:1633 — pass the frame's return value to the caller
        if self.nextblackholeinterp.is_none() {
            // blackhole.py:1635 — bottommost frame
            return Err(self.done_with_this_frame());
        }

        // Copy return values to locals before borrowing caller mutably
        let ret_type = self.return_type;
        let tmp_i = self.tmpreg_i;
        let tmp_r = self.tmpreg_r;
        let tmp_f = self.tmpreg_f;

        let caller = self.nextblackholeinterp.as_mut().unwrap();
        match ret_type {
            BhReturnType::Int => caller.setup_return_value_i(tmp_i),
            BhReturnType::Ref => caller.setup_return_value_r(tmp_r),
            BhReturnType::Float => caller.setup_return_value_f(tmp_f),
            BhReturnType::Void => {}
        }

        // blackhole.py:1645 — return no exception
        Ok(0)
    }

    // -- Bytecode reading helpers (matching MIFrame.next_u8/next_u16) --

    fn next_u8(&mut self) -> u8 {
        jitcode::read_u8(&self.jitcode.code, &mut self.position)
    }

    /// Read one register operand ([`jitcode::JitcodeReg`]); register operands
    /// are always one byte. Use this at register reads instead of `next_u8`
    /// so the 1-byte register width stays enforced in one place.
    fn next_reg(&mut self) -> jitcode::JitcodeReg {
        self.next_u8()
    }

    #[allow(dead_code)]
    fn next_u16(&mut self) -> u16 {
        jitcode::read_u16(&self.jitcode.code, &mut self.position)
    }

    fn peek_u16_at(&self, pos: usize) -> Option<u16> {
        let code = &self.jitcode.code;
        if pos + 1 >= code.len() {
            return None;
        }
        Some((code[pos] as u16) | ((code[pos + 1] as u16) << 8))
    }

    fn runtime_bh_descr(&self, descr_idx: usize) -> Option<&BhDescr> {
        if let Some(entry) = self.jitcode.exec.descrs.get(descr_idx) {
            return entry.as_bh_descr();
        }
        self.descrs.get(descr_idx)
    }

    /// Read the static-field index from a canonical `VableField` descr
    /// pool entry at `pos`. Panics if the bytes do not point at a
    /// `BhDescr::VableField` — Stage 3c-3 collapses the dual-mode
    /// auto-detect, so callers must already be on the canonical layout
    /// (`assembler.py:165-167` + `:197-207`).
    #[allow(dead_code)]
    fn vable_field_index_at(&self, pos: usize) -> usize {
        let idx = self
            .peek_u16_at(pos)
            .expect("vable_field_index_at: descr operand out of bounds");
        match self.runtime_bh_descr(idx as usize) {
            Some(BhDescr::VableField { index }) => *index,
            other => {
                panic!("vable_field_index_at: expected VableField at descr {idx}, got {other:?}")
            }
        }
    }

    /// Read the array-field index from a canonical
    /// (`VableArray`, `Array`) descr pool pair at `field_pos` /
    /// `array_pos`. Panics if either entry is missing or the wrong
    /// variant — see [`Self::vable_field_index_at`].
    #[allow(dead_code)]
    fn vable_array_index_pair_at(&self, field_pos: usize, array_pos: usize) -> usize {
        let field_idx = self
            .peek_u16_at(field_pos)
            .expect("vable_array_index_pair_at: field descr out of bounds");
        let array_idx = self
            .peek_u16_at(array_pos)
            .expect("vable_array_index_pair_at: array descr out of bounds");
        let field_descr = self.runtime_bh_descr(field_idx as usize);
        let array_descr = self.runtime_bh_descr(array_idx as usize);
        match (field_descr, array_descr) {
            (Some(BhDescr::VableArray { index }), Some(BhDescr::Array { .. })) => *index,
            other => panic!(
                "vable_array_index_pair_at: expected (VableArray, Array) at descrs ({field_idx}, {array_idx}), got {other:?}"
            ),
        }
    }

    /// blackhole.py _get_list_of_values parity.
    ///
    /// Decodes a bytecode-encoded register list: [length:u8][indices:u8...].
    /// Returns a Vec of register values looked up from the appropriate
    /// register file (registers_i for 'I', registers_r for 'R').
    /// Every index in the list names a register the jitcode declared, so the
    /// bank is indexed directly: upstream's `self.registers_i[index]` raises
    /// on an out-of-range index, and its untranslated builds seed the bank
    /// with `MissingValue()` so that even an in-range-but-never-written slot
    /// is caught at the read.  Substituting `0` instead turned a codewriter
    /// bug — a list built against a different register allocation — into a
    /// call made with a zero argument.
    fn _get_list_of_values_i_into(&mut self, values: &mut Vec<i64>) {
        let length = self.next_u8() as usize;
        values.clear();
        values.reserve(length);
        for _ in 0..length {
            let index = self.next_reg() as usize;
            if crate::bh_debug_enabled() {
                eprintln!("[bh-getlist] i{index} -> {}", self.registers_i[index]);
            }
            values.push(self.registers_i[index]);
        }
    }

    /// blackhole.py _get_list_of_values(self, code, position, 'R')
    fn _get_list_of_values_r_into(&mut self, values: &mut Vec<i64>) {
        let length = self.next_u8() as usize;
        values.clear();
        values.reserve(length);
        for _ in 0..length {
            let index = self.next_reg() as usize;
            values.push(self.registers_r[index]);
        }
    }

    fn _get_list_of_values_f_into(&mut self, values: &mut Vec<i64>) {
        let length = self.next_u8() as usize;
        values.clear();
        values.reserve(length);
        for _ in 0..length {
            let index = self.next_reg() as usize;
            values.push(self.registers_f[index]);
        }
    }

    /// blackhole.py `bhimpl_jit_merge_point`. Decodes
    /// `@arguments("self", "i", "I", "R", "F", "I", "R", "F")` from
    /// `self.position` (advancing it past the jdindex byte and the six
    /// typed register lists), then either raises `ContinueRunningNormally`
    /// (bottommost level) or recursive_call's into the portal jitcode and
    /// raises `LeaveFrame`.
    ///
    /// `opcode` selects how the jdindex byte is decoded
    /// (blackhole.py): `BC_JIT_MERGE_POINT` reads it as a
    /// `registers_i` pool slot index (`'i'` argcode); `BC_JIT_MERGE_POINT_C`
    /// reads it as a raw signed byte (`'c'` argcode, assembler.py:312
    /// `USE_C_FORM`).
    pub(crate) fn bhimpl_jit_merge_point(&mut self, opcode: u8) -> Result<(), DispatchError> {
        let nbody_debug = crate::nbody_debug_enabled();
        let jdindex_byte = self.next_reg();
        let jdindex = if opcode == jitcode::insns::BC_JIT_MERGE_POINT_C {
            (jdindex_byte as i8) as usize
        } else {
            self.registers_i[jdindex_byte as usize] as usize
        };
        let mut args = self.merge_point_args_scratch.take().unwrap_or_default();
        self._get_list_of_values_i_into(&mut args.green_int);
        self._get_list_of_values_r_into(&mut args.green_ref);
        self._get_list_of_values_f_into(&mut args.green_float);
        self._get_list_of_values_i_into(&mut args.red_int);
        self._get_list_of_values_r_into(&mut args.red_ref);
        self._get_list_of_values_f_into(&mut args.red_float);
        if nbody_debug {
            eprintln!(
                "[nbody-debug][bh-jmp] pos={} jdindex={} gi={:?} gr={:?} ri={:?} rr={:#x?}",
                self.last_opcode_position,
                jdindex,
                args.green_int,
                args.green_ref,
                args.red_int,
                args.red_ref,
            );
        }

        if self.nextblackholeinterp.is_none() {
            // blackhole.py:1068-1069: bottommost level.
            //   raise ContinueRunningNormally(*args)
            return Err(DispatchError::ContinueRunningNormally(args));
        }
        // blackhole.py:1074-1093: recursive portal level.
        //   sd = self.builder.metainterp_sd
        //   result_type = sd.jitdrivers_sd[jdindex].result_type
        //   if result_type == 'v': bhimpl_recursive_call_v + void_return
        //   elif result_type == 'i': bhimpl_recursive_call_i + int_return
        //   elif result_type == 'r': bhimpl_recursive_call_r + ref_return
        //   elif result_type == 'f': bhimpl_recursive_call_f + float_return
        //   assert False
        // The jdindex operand names a jitdriver, and only this recursive-portal
        // level indexes the table with it — the bottommost level above raises
        // `ContinueRunningNormally` without ever reading it.  A value out of
        // range here is a decode fault, not a missing driver: either this frame
        // resumed at a position that is not a merge point, or a `setarg_i` wrote
        // past `num_regs_i` and overwrote the constant slot the operand reads.
        // Report both coordinates instead of an anonymous bounds panic.
        assert!(
            jdindex < self.jitdrivers_sd.len(),
            "blackhole jit_merge_point decoded jdindex={jdindex} but only {} jitdriver(s) exist \
             (jitcode={} c_form={} jdindex_byte={jdindex_byte} num_regs_i={} registers_i.len={} \
             position={} last_opcode_position={} entry_position={})",
            self.jitdrivers_sd.len(),
            self.jitcode.name(),
            opcode == jitcode::insns::BC_JIT_MERGE_POINT_C,
            self.jitcode.num_regs_i(),
            self.registers_i.len(),
            self.position,
            self.last_opcode_position,
            self.entry_position,
        );
        let result_type = self.jitdrivers_sd[jdindex].result_type;
        // The portal runner is published behind an `extern "C"` boundary and
        // cannot unwind, so a raise inside it arrives in `BH_LAST_EXC_VALUE`
        // instead of out of the call — where `blackhole.py run`'s blanket
        // `except Exception` catches upstream's real unwind and hands it to
        // `handle_exception_in_frame`.  Clear the cell before the call and test
        // it after, the protocol `check_residual_call_exception_after`
        // documents and every `residual_call_*` / `inline_call_*` /
        // `call_assembler_*` / `cond_call_*` family follows.  Without it the
        // call's value is whatever the runner returned on its error path —
        // `PY_NULL` — the frame leaves with that NULL installed as the result,
        // and the exception is left unread in the cell for an unrelated
        // opcode to clear or for `bhimpl_abort_permanent` to deliver at the
        // wrong bytecode.
        self.last_exc().set(0);
        // The post-call position, where the codewriter put the can-raise
        // opcode's `-live-` adjacency.  Read before the call: the arms below
        // do not advance `position`, but the handler search starts here.
        let post_call_position = self.position;
        match result_type {
            BhReturnType::Void => {
                self.bhimpl_recursive_call_v(
                    jdindex,
                    std::mem::take(&mut args.green_int),
                    std::mem::take(&mut args.green_ref),
                    std::mem::take(&mut args.green_float),
                    std::mem::take(&mut args.red_int),
                    std::mem::take(&mut args.red_ref),
                    std::mem::take(&mut args.red_float),
                );
                self.return_type = BhReturnType::Void;
            }
            BhReturnType::Int => {
                let x = self.bhimpl_recursive_call_i(
                    jdindex,
                    std::mem::take(&mut args.green_int),
                    std::mem::take(&mut args.green_ref),
                    std::mem::take(&mut args.green_float),
                    std::mem::take(&mut args.red_int),
                    std::mem::take(&mut args.red_ref),
                    std::mem::take(&mut args.red_float),
                );
                self.tmpreg_i = x;
                self.return_type = BhReturnType::Int;
            }
            BhReturnType::Ref => {
                let x = self.bhimpl_recursive_call_r(
                    jdindex,
                    std::mem::take(&mut args.green_int),
                    std::mem::take(&mut args.green_ref),
                    std::mem::take(&mut args.green_float),
                    std::mem::take(&mut args.red_int),
                    std::mem::take(&mut args.red_ref),
                    std::mem::take(&mut args.red_float),
                );
                self.tmpreg_r = x.0 as i64;
                self.return_type = BhReturnType::Ref;
            }
            BhReturnType::Float => {
                let x = self.bhimpl_recursive_call_f(
                    jdindex,
                    std::mem::take(&mut args.green_int),
                    std::mem::take(&mut args.green_ref),
                    std::mem::take(&mut args.green_float),
                    std::mem::take(&mut args.red_int),
                    std::mem::take(&mut args.red_ref),
                    std::mem::take(&mut args.red_float),
                );
                self.tmpreg_f = x.to_bits() as i64;
                self.return_type = BhReturnType::Float;
            }
        }
        // The recursive level owns the six lists now; the box they sat in is
        // this frame's, so put it back for the next merge point rather than
        // dropping it here.
        self.recycle_merge_point_args(args);
        // Upstream's raise unwinds out of `bhimpl_jit_merge_point`, so the
        // frame does not leave on this path: the dispatch loop searches THIS
        // frame's handlers at the post-call position first, and only a frame
        // without one propagates.
        check_residual_call_exception_after(self, post_call_position)?;
        Err(DispatchError::LeaveFrame)
    }

    /// pyre-only `BC_ABORT_PERMANENT` body. Routes a TLS-stashed exception
    /// through `RaiseException` (so `handle_exception_in_frame` can pick
    /// the except handler); falls back to `aborted = true` + `LeaveFrame`
    /// when no exception is set. RPython has no direct counterpart — the
    /// canonical `abort_permanent/` handler is emitted by pyre's codegen
    /// for fail-paths that should always terminate the blackhole frame.
    pub(crate) fn bhimpl_abort_permanent(&mut self) -> Result<(), DispatchError> {
        let exc = self.last_exc().get();
        if exc != 0 {
            // Hand the exception to the other walked root before dropping this
            // one.  A residual-call raise is reachable *only* through this cell
            // (`walk_bh_last_exc_value`) until something stores it, and the
            // `RaiseException` arm below routes it into
            // `handle_exception_in_frame` → `route_to_catch`, which records a
            // traceback node — an allocation — before it performs that store.
            // The exception is a nursery object (`w_exception_new_empty` →
            // `alloc_exception_nursery`, `malloc_fixedsize` →
            // `collect_and_reserve`), so the traceback allocation moves it.
            // An unrooted local in that window is left pointing at the
            // pre-move address.  `route_to_catch` clears the
            // cell only *after* the record for the same reason
            // (blackhole.py parity); `exception_last_value` is a
            // `walk_bh_regs` root, so writing it first keeps the value covered
            // across the handoff.
            self.exception_last_value = exc;
            self.last_exc().set(0);
            // `abort_permanent/` carries no operand bytes, so the position the
            // dispatch loop already advanced past the opcode byte is the end of
            // the instruction.
            return Err(DispatchError::RaiseException {
                exc,
                resume_position: self.position,
            });
        }
        self.aborted = true;
        self.abort_permanent_bail = true;
        Err(DispatchError::LeaveFrame)
    }

    /// blackhole.py bhimpl_rvmprof_code.
    pub fn bhimpl_rvmprof_code(&self, leaving: i64, unique_id: i64) {
        jit_rvmprof_code(leaving, unique_id);
    }

    /// blackhole.py handle_exception_in_frame: check if the current
    /// position has an immediately-following `catch_exception/L`.
    ///
    /// Upstream inspects exactly one instruction: the operation-owned resume
    /// `-live-` is skipped and whatever follows it either is this operation's
    /// `catch_exception` or is not.  Guard snapshots carry that exact trailing
    /// marker (`after_residual_guard_marker`, the
    /// `capture_resumedata(after_residual_call=True)` port), so searching other
    /// operation startpoints here would only permit a bad resume coordinate to
    /// steal a later operation's handler.
    pub fn handle_exception_in_frame(&mut self, exc_value: i64) -> bool {
        let code = &self.jitcode.code;
        let mut position = self.position;
        if position >= code.len() {
            return false;
        }
        if code[position] == self.op_live {
            position += majit_jitcode::liveness::OFFSET_SIZE + 1;
            if position >= code.len() {
                return false;
            }
        }
        let opcode = code[position];
        // `BlackholeInterpreter.handle_exception_in_frame` parity: the catch
        // is directly after this operation's resume `-live-`.
        if opcode == self.op_catch_exception {
            return self.route_to_catch(position, exc_value);
        }
        if opcode == self.op_rvmprof_code {
            // blackhole.py:412-420: on exception immediately before a
            // `rvmprof_code/ii`, run the leaving hook and continue
            // propagating.
            let leaving = self.registers_i[code[position + 1] as usize];
            let unique_id = self.registers_i[code[position + 2] as usize];
            assert_eq!(leaving, 1);
            self.bhimpl_rvmprof_code(leaving, unique_id);
        }
        false
    }

    /// Dispatch the in-frame `catch_exception/L` at `catch_pos`: stash the
    /// exception in `exception_last_value`, jump to the handler label, and
    /// clear the residual-call TLS slot (blackhole.py parity — once the
    /// handler runs, a later opcode reading `BH_LAST_EXC_VALUE` without
    /// issuing a new call must not pick up this already-caught exception).
    fn route_to_catch(&mut self, catch_pos: usize, exc_value: i64) -> bool {
        let code = &self.jitcode.code;
        if catch_pos + 2 >= code.len() {
            return false;
        }
        let target = (code[catch_pos + 1] as usize) | ((code[catch_pos + 2] as usize) << 8);
        // Take the exception into this frame's walked slot before anything
        // below can allocate.  `record` builds a traceback node, and on the
        // inline-callee path this slot is the exception's ONLY root at that
        // moment: `handler_inline_call_nested_ext` reads
        // `callee.exception_last_value` after `callee.run()` returned, which
        // popped the callee's `push_bh_regs` entry, and a callee-local `raise`
        // never went through `BH_LAST_EXC_VALUE`.  The exception is a nursery
        // object (`w_exception_new_empty` → `alloc_exception_nursery`), so
        // the record moves it; the walked slot is what the collector
        // rewrites.  A local copied before the record is stale afterwards.
        self.exception_last_value = exc_value;
        self.record_frame_traceback(exc_value);
        self.position = target;
        self.last_exc().set(0);
        // A residual `bh_call` that raised published the exception into BOTH
        // `BH_LAST_EXC_VALUE` and the backend `_store_exception` cells
        // (`publish_residual_call_exception`).  Clearing only the former leaves
        // the cell holding the now-caught exception; the next compiled re-entry
        // (a nested call from this handler) would read it at its first
        // `GUARD_NO_EXCEPTION` and re-deliver the already-handled exception.
        // Drain the backend cells here so the handler runs with a pristine cell.
        // Synthetic dispatcher tests can exercise a pure catch path without
        // any CPU operation.  A real builder always carries `codewriter.cpu`;
        // when present, keep the Rust residual-exception side channel in sync
        // with the exception slot that the upstream frame just consumed.
        if let Some(cpu) = self.cpu {
            cpu.get().clear_stored_exception();
        }
        true
    }

    /// `pyopcode.py pytraceback.record_application_traceback` for the
    /// frame this interpreter is leaving with `exc_value` live.
    ///
    /// Upstream takes the handler search and the traceback record from one
    /// `handle_operation_error`, so a frame that raises always gets its node no
    /// matter which way the search went.  Here the search half is the jitcode
    /// `catch_exception` scan and the record half has no jitcode counterpart,
    /// so every arm that ends an exception's stay in this frame has to make the
    /// call — the caught arm ([`Self::route_to_catch`]) and the two propagate
    /// arms in [`Self::resume_mainloop`] alike.
    ///
    /// `raise_varargs` records the raising instruction before the handler
    /// lookup; `RaiseWithExplicitTraceback` skips it for a bare reraise. The
    /// installed hook additionally drops a node the exception's chain head
    /// already carries, so a second call for one delivery is a no-op.
    fn record_frame_traceback(&self, exc_value: i64) {
        if exc_value == 0 {
            return;
        }
        if self.jitcode.code.get(self.last_opcode_position) == Some(&jitcode::insns::BC_RERAISE) {
            return;
        }
        let jitcode_index = self.jitcode.try_index().map_or(-1, |index| index as i32);
        let position = self.last_opcode_position as i32;
        match self.record_caught_exception {
            Some(record) => record(
                exc_value,
                self.virtualizable_ptr,
                i64::from(jitcode_index),
                i64::from(position),
            ),
            // The per-interpreter field is an override a chain builder or a
            // test installs on the frames it owns.  An interpreter leased
            // straight from the pool — every one the tracing-abort adoption
            // drives — has none, and dropping the record there is what left a
            // frame the blackhole finished out of its own traceback.  The
            // process-global hook is the same host callback
            // (`build_jit_driver_pair` installs one function for both) and
            // no-ops while unset.
            None => crate::pyjitpl::record_application_traceback_for_recording(
                exc_value,
                self.virtualizable_ptr,
                jitcode_index,
                position,
            ),
        }
    }

    /// blackhole.py handle_rvmprof_enter.
    pub fn handle_rvmprof_enter(&mut self) {
        let code = &self.jitcode.code;
        let mut position = self.position;
        let mut opcode = code[position];
        if opcode == self.op_live {
            position += majit_jitcode::liveness::OFFSET_SIZE + 1;
            opcode = code[position];
        }
        if opcode == self.op_rvmprof_code {
            let leaving = self.registers_i[code[position + 1] as usize];
            let unique_id = self.registers_i[code[position + 2] as usize];
            if leaving == 1 {
                self.bhimpl_rvmprof_code(0, unique_id);
            }
        }
    }

    // -- Call argument reading --

    fn read_call_arg(&self, kind: JitArgKind, reg: u16) -> i64 {
        match kind {
            JitArgKind::Int => self.registers_i[reg as usize],
            JitArgKind::Ref => self.registers_r[reg as usize],
            JitArgKind::Float => self.registers_f[reg as usize],
        }
    }

    /// Whether `run` must walk the caller chain to install register /
    /// virtualizable roots. False when every frame is already registered
    /// and no heap virtualizable (`has_vable_token`) is attached.
    fn chain_needs_run_rooting(&self) -> bool {
        let mut frame = Some(self);
        while let Some(f) = frame {
            if !f.rooted {
                return true;
            }
            if !f.virtualizable_info.is_null() {
                // SAFETY: `virtualizable_info` is the process-owned
                // `VirtualizableInfo` `seed_deopt_vinfo_ptr` / the
                // portal stamped; it outlives the drive.
                let vinfo = unsafe { &*f.virtualizable_info };
                if vinfo.has_vable_token() {
                    return true;
                }
            }
            frame = f.nextblackholeinterp.as_deref();
        }
        false
    }

    /// Execute the dispatch loop on the current jitcode.
    ///
    /// RPython: `BlackholeInterpreter.run()` catches `LeaveFrame` and breaks,
    /// catches exceptions and calls `handle_exception_in_frame`.
    /// Returns `Some(args)` for `ContinueRunningNormally` (RPython: raise
    /// jitexc.ContinueRunningNormally propagates through run→_run_forever).
    pub fn run(&mut self) -> BhRunOutcome {
        // rlib/jit.py we_are_jitted: true during tracing and blackholing.
        let _jitted = crate::JittedGuard::enter();
        let _bh_phase = majit_gc::BhProbePhase::enter("blackhole");
        // Pooled interpreters are registered for life (`acquire_interp`).
        // `seed_deopt_vinfo_ptr` still stamps a no-token state-field
        // `VirtualizableInfo` on every regex frame, and the vable register
        // scan below then walks the whole chain on every `run()` — O(frames²)
        // `gc_owns_object` probes per character, against a vtype that is
        // never a `NodeRec`. Skip that walk when every frame is already
        // rooted and no heap virtualizable is present.
        if !self.chain_needs_run_rooting() {
            return self.run_after_rooting();
        }
        // Root this frame's register bank AND every pending caller frame
        // reachable through `nextblackholeinterp`.  RPython keeps the whole
        // blackhole interpreter chain transitively GC-traced from the head for
        // the lifetime of `_run_forever`: each interp is a GC object and its
        // `registers_r` array plus the `nextblackholeinterp` link are traced
        // fields.  pyre's `push_bh_regs` roots one frame's `Vec` copy at a
        // time, so a caller frame that is not yet running is invisible to the
        // collector.  A minor collection triggered inside this frame's
        // `run_inner` (nursery allocation) would then fail to forward a young
        // object held only by a pending caller's register — most importantly
        // that caller's own materialized `PyFrame` self-pointer — and
        // `nursery.reset()` zeroes the reclaimed slot.  When control unwinds to
        // that caller it reads a stale/zeroed frame (its `locals_cells_stack_w`
        // slot NULL) and a vable opcode dereferences it.  Rooting the full
        // pending chain here mirrors the RPython invariant.
        // A pooled interpreter is registered for its whole life
        // (`acquire_interp`); only one built outside the pool is rooted here.
        let bh_depth = majit_gc::shadow_stack::bh_regs_depth();
        unsafe {
            let mut frame = Some(&mut *self);
            while let Some(f) = frame {
                if !f.rooted {
                    let ctx = f as *mut BlackholeInterpreter;
                    majit_gc::shadow_stack::push_bh_regs_with_live(
                        &mut (*ctx).registers_r,
                        &mut (*ctx).tmpreg_r,
                        &mut (*ctx).exception_last_value,
                        ctx as *const (),
                    );
                }
                frame = f.nextblackholeinterp.as_deref_mut();
            }
        }
        let vable_roots_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        unsafe {
            let mut current = Some(&mut *self);
            while let Some(frame) = current {
                // `virtualizable_ptr` is a bare copy of the frame red this
                // level was bound to, and the banks rooted just above do not
                // cover it: a collection forwards the register the copy came
                // from, never the copy.  A young virtualizable makes that
                // difference observable — a compiled trace allocates an
                // inlined callee's `PyFrame` with its own `NewWithVtable`,
                // which the GC rewriter lowers to a nursery allocation, so the
                // first minor collection `run_inner` triggers moves it and
                // leaves this field naming the vacated block.  Every later
                // reader takes that address: the vable opcodes below, and the
                // traceback recorded for each frame an exception propagates
                // through, which stores it into `PyTraceback.frame` — a slot
                // the collector does trace, so the stale pointer surfaces as
                // an invalid type id rather than as a wrong answer.  Root the
                // SLOT so the walker rewrites it, the same shape
                // `blackhole_from_resumedata` already applies to the resume
                // reader's own `virtualizable_ptr` for the chain-build window.
                if !frame.rooted {
                    majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(
                        &mut frame.virtualizable_ptr,
                    ));
                }
                if !frame.virtualizable_info.is_null() {
                    let vinfo = &*frame.virtualizable_info;
                    vinfo.push_resume_ref_roots_for_registers(&frame.registers_r);
                    crate::resume::VirtualizableInfo::push_resume_ref_roots(
                        vinfo,
                        frame.virtualizable_ptr,
                    );
                }
                current = frame.nextblackholeinterp.as_deref_mut();
            }
        }
        let result = self.run_after_rooting();
        majit_gc::shadow_stack::pop_resume_ref_roots_to(vable_roots_depth);
        majit_gc::shadow_stack::pop_bh_regs_to(bh_depth);
        result
    }

    /// blackhole.py `run`: dispatch from the resume position.
    fn run_after_rooting(&mut self) -> BhRunOutcome {
        self.run_inner()
    }

    fn run_inner(&mut self) -> BhRunOutcome {
        let trace = crate::majit_log_enabled() || crate::bh_debug_enabled();
        // blackhole.py:86-91:
        //
        // ```python
        // if (not we_are_translated()
        //     and self.jitcode._startpoints is not None):
        //     assert position in self.jitcode._startpoints, (
        //         "the current position %d is in the middle of "
        //         "an instruction!" % position)
        // ```
        //
        // The builder's `dispatch_loop` carries this check, but nothing
        // production dispatches through it — this loop is the one that runs.
        // Its subject is the resume coordinate somebody wrote into the frame,
        // and without the check a coordinate pointing into an operand payload
        // surfaces only as a bogus opcode byte, naming the dispatch table
        // instead of the writer.  `jit_strict_mode` is the untranslated-build
        // analog: debug builds plus `MAJIT_STRICT`.
        let check_startpoints = crate::jit_strict_mode();
        // `blackhole.py` `BlackholeInterpreter.run` binds the bytecode ONCE —
        // `self.dispatch_loop(self, self.jitcode.code, self.position)` — and
        // `dispatch_loop` carries it as a parameter for the whole loop.  Bind it
        // the same way here.  The `Arc` clone exists only to detach the slice
        // borrow from `&mut self` so handlers can still take `&mut self`; per
        // opcode it was an atomic increment and an atomic decrement, plus a
        // `OnceLock` acquire load in each of `finished`, `next_u8` and
        // `dispatch_step`, on a path this benchmark runs 8.7 million times per
        // 4096 input characters.
        //
        // Hoisting is sound because a frame's jitcode is fixed for the life of
        // its dispatch loop: the only production `setposition` callers seat a
        // DIFFERENT interpreter — `copy_data_from_miframe` and the resume
        // reader before `run`, and `handler_inline_call_nested_ext` on the
        // callee — so no handler re-seats the frame it is dispatching in.
        // Bind the code slice without an `Arc` clone: `self.jitcode` already
        // owns the bytes for the whole loop.
        let code: &[u8] = {
            let slice = self.jitcode.code.as_slice();
            // SAFETY: `self.jitcode` is not reseated while this loop runs
            // (see above). The slice is only read.
            unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) }
        };
        // `blackhole.py` `dispatch_loop` keeps `position` as a loop local
        // and increments it past the opcode before the inlined handler
        // (`opcode = ord(code[position]); position += 1`).
        let mut position = self.position;
        // SAFETY: `self.jitcode` / `registers_*` are not reseated by an
        // inlined handler. INLINE_CALL and the function-pointer fallback
        // refresh the register pointers below.
        let exec_descrs: &[crate::jitcode::RuntimeBhDescr] = unsafe {
            let slice = self.jitcode.exec.descrs.as_slice();
            std::slice::from_raw_parts(slice.as_ptr(), slice.len())
        };
        let mut regs_i = self.registers_i.as_mut_ptr();
        let mut regs_r = self.registers_r.as_mut_ptr();
        loop {
            if position >= code.len() {
                self.position = position;
                if trace {
                    eprintln!(
                        "[bh-trace] finished at pos={} reg0={}",
                        position,
                        self.registers_i.first().copied().unwrap_or(-1)
                    );
                }
                return BhRunOutcome::EndOfCode;
            }
            if check_startpoints && let Some(startpoints) = self.jitcode.startpoints.as_ref() {
                assert!(
                    startpoints.contains(&position),
                    "run_inner: position {position} is in the middle of an instruction \
                     (jitcode {:?} index {:?})",
                    self.jitcode.name,
                    self.jitcode.try_index(),
                );
            }
            // SAFETY: `position < code.len()` just held.
            let opcode = unsafe { *code.get_unchecked(position) };
            let pos_before = position;
            position += 1;
            // The remaining `shift` epilogue at pc 210 is `-live-`,
            // `goto/L`, `goto_if_not`, `int_copy`, getfield/setfield of
            // `left`/`right`/`marked`/`empty`, `int_eq`/`int_add`,
            // `int_return`, and one native INLINE_CALL.
            // RPython's translated `dispatch_loop` inlines those
            // `_get_method` bodies; the function-pointer table is the
            // untranslated form. Do not restart `fnaddr` from this
            // mid-node PC.
            if !trace {
                match opcode {
                    jitcode::insns::BC_LIVE => {
                        // `bhimpl_live`: remember the marker, skip `OFFSET_SIZE`.
                        self.last_live_pc = pos_before;
                        position += majit_jitcode::liveness::OFFSET_SIZE;
                        continue;
                    }
                    jitcode::insns::BC_JUMP => {
                        // SAFETY: well-formed `goto/L` carries two operand bytes.
                        position = unsafe { bh_code_u16(code, position) };
                        continue;
                    }
                    jitcode::insns::BC_GOTO_IF_NOT | jitcode::insns::BC_GOTO_IF_NOT_INT_IS_TRUE => {
                        // `if mark` / `if old_left` lower to
                        // `goto_if_not_int_is_true`; `bhimpl_goto_if_not_int_is_true`
                        // is `bhimpl_goto_if_not`.
                        // SAFETY: operand bytes and the named int register exist.
                        let a = unsafe { *regs_i.add(*code.get_unchecked(position) as usize) };
                        let target = unsafe { bh_code_u16(code, position + 1) };
                        position = bhimpl_goto_if_not(a, target, position + 3);
                        continue;
                    }
                    jitcode::insns::BC_INT_IS_TRUE => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            *regs_i.add(*code.get_unchecked(position + 1) as usize) =
                                bhimpl_int_is_true(a);
                        }
                        position += 2;
                        continue;
                    }
                    jitcode::insns::BC_MOVE_I => {
                        unsafe {
                            let src = *regs_i.add(*code.get_unchecked(position) as usize);
                            *regs_i.add(*code.get_unchecked(position + 1) as usize) = src;
                        }
                        position += 2;
                        continue;
                    }
                    jitcode::insns::BC_MOVE_I_C => {
                        unsafe {
                            let src = *code.get_unchecked(position) as i8 as i64;
                            *regs_i.add(*code.get_unchecked(position + 1) as usize) = src;
                        }
                        position += 2;
                        continue;
                    }
                    jitcode::insns::BC_GETFIELD_GC_I | jitcode::insns::BC_GETFIELD_GC_I_PURE => {
                        unsafe {
                            let struct_ptr = *regs_r.add(*code.get_unchecked(position) as usize);
                            let (descr, pos) =
                                read_descr_fast(exec_descrs, self, code, position + 1);
                            let dest = *code.get_unchecked(pos) as usize;
                            *regs_i.add(dest) = bh_load_int_field(struct_ptr, descr);
                            position = pos + 1;
                        }
                        continue;
                    }
                    jitcode::insns::BC_GETFIELD_GC_R | jitcode::insns::BC_GETFIELD_GC_R_PURE => {
                        unsafe {
                            let struct_ptr = *regs_r.add(*code.get_unchecked(position) as usize);
                            let (descr, pos) =
                                read_descr_fast(exec_descrs, self, code, position + 1);
                            let dest = *code.get_unchecked(pos) as usize;
                            *regs_r.add(dest) = bh_load_ref_field(struct_ptr, descr);
                            position = pos + 1;
                        }
                        continue;
                    }
                    jitcode::insns::BC_INT_EQ => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            let b = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            *regs_i.add(*code.get_unchecked(position + 2) as usize) =
                                bhimpl_int_eq(a, b);
                        }
                        position += 3;
                        continue;
                    }
                    jitcode::insns::BC_INT_NE => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            let b = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            *regs_i.add(*code.get_unchecked(position + 2) as usize) =
                                bhimpl_int_ne(a, b);
                        }
                        position += 3;
                        continue;
                    }
                    jitcode::insns::BC_INT_LT => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            let b = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            *regs_i.add(*code.get_unchecked(position + 2) as usize) =
                                bhimpl_int_lt(a, b);
                        }
                        position += 3;
                        continue;
                    }
                    jitcode::insns::BC_INT_ADD => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            let b = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            *regs_i.add(*code.get_unchecked(position + 2) as usize) =
                                bhimpl_int_add(a, b);
                        }
                        position += 3;
                        continue;
                    }
                    jitcode::insns::BC_GOTO_IF_NOT_INT_EQ => {
                        unsafe {
                            let a = *regs_i.add(*code.get_unchecked(position) as usize);
                            let b = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            let target = bh_code_u16(code, position + 2);
                            position = if a == b { position + 4 } else { target };
                        }
                        continue;
                    }
                    jitcode::insns::BC_SETFIELD_GC_I => {
                        unsafe {
                            let struct_ptr = *regs_r.add(*code.get_unchecked(position) as usize);
                            let value = *regs_i.add(*code.get_unchecked(position + 1) as usize);
                            let (descr, pos) =
                                read_descr_fast(exec_descrs, self, code, position + 2);
                            bh_store_int_field(struct_ptr, value, descr);
                            position = pos;
                        }
                        continue;
                    }
                    jitcode::insns::BC_SETFIELD_GC_I_C => {
                        unsafe {
                            let struct_ptr = *regs_r.add(*code.get_unchecked(position) as usize);
                            let value = *code.get_unchecked(position + 1) as i8 as i64;
                            let (descr, pos) =
                                read_descr_fast(exec_descrs, self, code, position + 2);
                            bh_store_int_field(struct_ptr, value, descr);
                            position = pos;
                        }
                        continue;
                    }
                    jitcode::insns::BC_INT_RETURN => {
                        self.tmpreg_i =
                            unsafe { *regs_i.add(*code.get_unchecked(position) as usize) };
                        self.return_type = BhReturnType::Int;
                        self.position = position + 1;
                        return BhRunOutcome::LeaveFrame;
                    }
                    jitcode::insns::BC_INT_RETURN_C => {
                        self.tmpreg_i = unsafe { *code.get_unchecked(position) as i8 as i64 };
                        self.return_type = BhReturnType::Int;
                        self.position = position + 1;
                        return BhRunOutcome::LeaveFrame;
                    }
                    jitcode::insns::BC_INLINE_CALL => {
                        self.position = pos_before;
                        self.last_opcode_position = pos_before;
                        match handler_inline_call_nested_ext(self, code, position) {
                            Ok(new_pos) => {
                                position = new_pos;
                                regs_i = self.registers_i.as_mut_ptr();
                                regs_r = self.registers_r.as_mut_ptr();
                                continue;
                            }
                            Err(DispatchError::LeaveFrame) => {
                                return BhRunOutcome::LeaveFrame;
                            }
                            Err(DispatchError::ContinueRunningNormally(args)) => {
                                return BhRunOutcome::ContinueRunningNormally(args);
                            }
                            Err(DispatchError::RaiseException {
                                exc,
                                resume_position,
                                ..
                            }) => {
                                // dispatch_step writes resume_position
                                // before re-raising so catch_exception
                                // sees the post-op cursor.
                                self.position = resume_position;
                                if self.handle_exception_in_frame(exc) {
                                    position = self.position;
                                    regs_i = self.registers_i.as_mut_ptr();
                                    regs_r = self.registers_r.as_mut_ptr();
                                    continue;
                                }
                                self.got_exception = true;
                                self.exception_last_value = exc;
                                return BhRunOutcome::Exception;
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Fallback: position is already past the opcode, matching
            // `dispatch_loop` / `dispatch_step`.
            self.position = position;
            self.last_opcode_position = pos_before;
            if trace {
                eprintln!(
                    "[bh-trace] pos={} op={} reg0={} reg1={}",
                    pos_before,
                    opcode,
                    self.registers_i.first().copied().unwrap_or(-1),
                    self.registers_i.get(1).copied().unwrap_or(-1),
                );
            }
            match self.dispatch_step(opcode, code) {
                Ok(()) => {
                    position = self.position;
                    regs_i = self.registers_i.as_mut_ptr();
                    regs_r = self.registers_r.as_mut_ptr();
                }
                Err(DispatchError::LeaveFrame) => {
                    if trace {
                        eprintln!(
                            "[bh-trace] leave-frame at pos={} ret_type={:?}",
                            pos_before, self.return_type,
                        );
                    }
                    return BhRunOutcome::LeaveFrame;
                }
                Err(DispatchError::ContinueRunningNormally(args)) => {
                    // blackhole.py:1068: raise ContinueRunningNormally(*args)
                    // Propagates out of run() like RPython's JitException.
                    if trace {
                        crate::debug::log_one(
                            "jit-blackhole",
                            &format!("ContinueRunningNormally at pos={pos_before}"),
                        );
                    }
                    return BhRunOutcome::ContinueRunningNormally(args);
                }
                Err(DispatchError::RaiseException { exc, .. }) => {
                    // blackhole.py: except Exception → handle_exception_in_frame
                    if trace {
                        crate::debug::log_one(
                            "jit-blackhole",
                            &format!("exception at pos={pos_before} exc=0x{exc:x}"),
                        );
                    }
                    if self.handle_exception_in_frame(exc) {
                        // Handler found, continue execution at handler target
                        position = self.position;
                        regs_i = self.registers_i.as_mut_ptr();
                        regs_r = self.registers_r.as_mut_ptr();
                        continue;
                    }
                    // No handler: propagate exception via got_exception flag
                    self.got_exception = true;
                    self.exception_last_value = exc;
                    return BhRunOutcome::Exception;
                }
            }
        }
    }

    /// Dispatch a single bytecode instruction through the per-instance
    /// `dispatch_table`.
    ///
    /// RPython parity: `blackhole.py` `dispatch_loop` calls
    /// `self.dispatch_table[opcode_byte](self, code, position)`
    /// unconditionally — every opcode is wired before the loop runs.
    /// Pyre matches that contract: every dispatching builder
    /// (`build_inline_call_only_bh_builder` for production, the unit
    /// `build_test_bh_builder` for `bh_interp_tests`) installs a full
    /// setup_insns surface before its first dispatch, so the placeholder
    /// branch panics on the post-C.5.2 codebase.
    ///
    /// `position` invariant: caller is `run_inner`, which has already
    /// advanced `self.position` past the opcode byte via `next_u8`.
    /// Wired handlers expect `position` to point at the first operand
    /// byte (handler signature `(bh, code, position) -> Result<usize>`)
    /// and return the post-operand position; we then store it back into
    /// `self.position`.
    fn dispatch_step(&mut self, opcode: u8, code: &[u8]) -> Result<(), DispatchError> {
        // `blackhole.py dispatch_loop`: `self.dispatch_table[opcode_byte](...)`.
        // The table is 256 slots (`setup_insns`), so a u8 index cannot miss.
        // Unwired bytes are the placeholder, which LeaveFrames on Grain and
        // panics when a MemoryError provider is registered.
        let handler = self.dispatch_table[opcode as usize];
        match handler(self, code, self.position) {
            Ok(new_pos) => {
                self.position = new_pos;
                Ok(())
            }
            // `blackhole.py` `_get_method`: the wrapper assigns
            // `self.position` and only then re-raises, so a raising
            // instruction leaves the frame pointing past itself.  Only
            // `RaiseException` carries a coordinate because only it is read
            // again — `handle_exception_in_frame` is the sole consumer of
            // `self.position` on an error path, and the two control-flow
            // errors end the frame without another dispatch.
            Err(err) => {
                if let DispatchError::RaiseException {
                    resume_position, ..
                } = err
                {
                    self.position = resume_position;
                }
                Err(err)
            }
        }
    }

    /// Read call arguments from bytecode (kind:u8, reg:u8 per arg).
    #[allow(dead_code)]
    fn read_call_args(&mut self, num_args: usize) -> Vec<i64> {
        let mut args = Vec::with_capacity(num_args);
        let mut metas: Vec<(u8, u8)> = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            let kind_byte = self.next_u8();
            let kind = JitArgKind::decode(kind_byte);
            let reg = self.next_reg();
            metas.push((kind_byte, reg));
            args.push(self.read_call_arg(kind, reg as u16));
        }
        if crate::majit_log_enabled() {
            // Heuristic: for Ref args, peek intval at +0x10 (W_IntObject layout).
            let mut intvals: Vec<Option<i64>> = Vec::with_capacity(args.len());
            for ((kind_byte, _reg), val) in metas.iter().zip(args.iter()) {
                if *kind_byte == 1 && *val != 0 {
                    intvals.push(Some(unsafe { *((*val as usize + 0x10) as *const i64) }));
                } else {
                    intvals.push(None);
                }
            }
            eprintln!(
                "[bh] read_call_args pos={} num={} kinds_regs={:?} vals={:?} maybe_intvals={:?}",
                self.last_opcode_position, num_args, metas, args, intvals
            );
        }
        args
    }
}

// bhimpl_*_call_* family (blackhole.py get_portal_runner)
//
// These methods mirror RPython's blackhole call dispatch table. Each
// variant unpacks one of the three calling-convention shapes
// (`r` / `ir` / `irf`) into a `Backend::bh_call_{i,r,f,v}` invocation
// and returns the typed result.  The `recursive_call` family adds a
// jitdriver-index lookup via `get_portal_runner`.
//
// **Wired into the codewriter-orthodox dispatch table.**  The
// `handler_residual_call_*` / `handler_inline_call_*` /
// `handler_recursive_call_*` functions registered by
// `wire_bhimpl_handlers` (`BlackholeInterpBuilder::wire_handler` →
// `dispatch_table`) decode the bytecode (func/jitcode + ListOfKind
// args + descr) and then call the corresponding `bhimpl_*_call_*`
// method on the BlackholeInterpreter, which in turn routes through
// `cpu().bh_call_{i,r,f,v}` exactly like RPython.  The legacy inline
// `jitcode::BC_RESIDUAL_CALL_*` / `jitcode::BC_CALL_MAY_FORCE_*` branches in
// `JitCodeMachine::dispatch_one` are pyre's IR-side fast path for
// JIT-compiled traces and operate on a different bytecode encoding
// (raw fn pointers, no calldescr); the codewriter-orthodox path runs
// through these handlers + bhimpl methods.

impl BlackholeInterpreter {
    fn cpu(&self) -> &dyn majit_backend::Backend {
        self.cpu
            .as_ref()
            .expect("blackhole cpu reference unset")
            .get()
    }

    // ── bhimpl_residual_call_* (blackhole.py:1224-1255) ──

    /// blackhole.py bhimpl_residual_call_r_i
    pub fn bhimpl_residual_call_r_i(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, None, Some(args_r), None, calldescr)
    }

    /// blackhole.py:1227-1229
    pub fn bhimpl_residual_call_r_r(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, None, Some(args_r), None, calldescr)
    }

    /// blackhole.py:1230-1232
    pub fn bhimpl_residual_call_r_v(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, None, Some(args_r), None, calldescr);
    }

    /// blackhole.py:1234-1236
    pub fn bhimpl_residual_call_ir_i(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, Some(args_i), Some(args_r), None, calldescr)
    }

    /// blackhole.py:1237-1239
    pub fn bhimpl_residual_call_ir_r(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, Some(args_i), Some(args_r), None, calldescr)
    }

    /// blackhole.py:1240-1242
    pub fn bhimpl_residual_call_ir_v(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, Some(args_i), Some(args_r), None, calldescr);
    }

    /// blackhole.py:1244-1246
    pub fn bhimpl_residual_call_irf_i(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1247-1249
    pub fn bhimpl_residual_call_irf_r(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1250-1252
    pub fn bhimpl_residual_call_irf_f(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> f64 {
        self.cpu()
            .bh_call_f(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1253-1255
    pub fn bhimpl_residual_call_irf_v(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, Some(args_i), Some(args_r), Some(args_f), calldescr);
    }

    // ── bhimpl_inline_call_* (`blackhole.py` `bhimpl_inline_call_*`) ──
    //
    // RPython unpacks `jitcode.fnaddr` and `jitcode.calldescr` from the
    // jitcode parameter; pyre passes them directly so the pyre tracer
    // can reuse the helpers without a JitCode object. The body is
    // `cpu.bh_call_*` (`blackhole.py` `bhimpl_inline_call_*`). A bridge
    // is recorded only from the guard's own `resumedescr`
    // (`compile.py` `ResumeGuardDescr.handle_fail` →
    // `_trace_and_compile_from_bridge`); `resume_in_blackhole` does not
    // start one.

    /// blackhole.py `bhimpl_inline_call_r_i`
    pub fn bhimpl_inline_call_r_i(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(fnaddr, None, Some(args_r), None, calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_r_r`
    pub fn bhimpl_inline_call_r_r(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(fnaddr, None, Some(args_r), None, calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_r_v`
    pub fn bhimpl_inline_call_r_v(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(fnaddr, None, Some(args_r), None, calldescr);
    }
    /// blackhole.py `bhimpl_inline_call_ir_i`
    pub fn bhimpl_inline_call_ir_i(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(fnaddr, Some(args_i), Some(args_r), None, calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_ir_r`
    pub fn bhimpl_inline_call_ir_r(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(fnaddr, Some(args_i), Some(args_r), None, calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_ir_v`
    pub fn bhimpl_inline_call_ir_v(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(fnaddr, Some(args_i), Some(args_r), None, calldescr);
    }
    /// blackhole.py `bhimpl_inline_call_irf_i`
    pub fn bhimpl_inline_call_irf_i(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(fnaddr, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_irf_r`
    pub fn bhimpl_inline_call_irf_r(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(fnaddr, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_irf_f`
    pub fn bhimpl_inline_call_irf_f(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> f64 {
        self.cpu()
            .bh_call_f(fnaddr, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }
    /// blackhole.py `bhimpl_inline_call_irf_v`
    pub fn bhimpl_inline_call_irf_v(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(fnaddr, Some(args_i), Some(args_r), Some(args_f), calldescr);
    }

    // ── bhimpl_recursive_call_{i,f,v} (blackhole.py bhimpl_recursive_call_i) ──
    //
    // The `_r` variant already lives further up in this file (line ~1165),
    // pre-existing pyre code that uses `jdindex` + `get_portal_runner`.
    // The remaining three follow the same pattern for parity.

    /// blackhole.py bhimpl_recursive_call_i
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn bhimpl_recursive_call_i(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> i64 {
        let (fnptr, calldescr, all_i, all_r, all_f) = self.portal_runner_call_args(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        );
        self.cpu()
            .bh_call_i(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// blackhole.py:1117-1124
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn bhimpl_recursive_call_f(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> f64 {
        let (fnptr, calldescr, all_i, all_r, all_f) = self.portal_runner_call_args(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        );
        self.cpu()
            .bh_call_f(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// blackhole.py:1125-1132
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn bhimpl_recursive_call_v(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) {
        let (fnptr, calldescr, all_i, all_r, all_f) = self.portal_runner_call_args(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        );
        self.cpu()
            .bh_call_v(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr);
    }
}

/// Pool manager + dispatch builder for blackhole interpreters.
///
/// RPython `blackhole.py` `class BlackholeInterpBuilder`.
///
/// Combines two responsibilities:
/// 1. Interpreter pool management (acquire/release/release_chain).
/// 2. Codewriter-orthodox dispatch setup (`setup_insns` → dispatch table
///    + `dispatch_loop`). Phase D incrementally wires this up as
///      `bhimpl_*` methods are ported from RPython.
pub struct BlackholeInterpBuilder {
    /// `BlackholeInterpBuilder.blackholeinterps`: intrusive free-list head.
    ///
    /// The head is the object itself, not a copy of it: `acquire_interp` hands
    /// one out and `release_interp` takes the same one back, so each of those
    /// steps — and the chain link `nextblackholeinterp` — moves one word
    /// rather than the register-bank struct.
    blackholeinterps: Option<Box<BlackholeInterpreter>>,
    /// RPython `blackhole.py` `self.cpu = codewriter.cpu`.
    /// Stored as raw pointer; the Backend outlives the builder.
    /// RPython `blackhole.py:286/56` `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    cpu: Option<BlackholeCpuRef>,
    /// RPython `blackhole.py:68` `self._insns`: opcode byte → "opname/argcodes".
    /// Populated by `setup_insns`; empty until called.
    pub _insns: Vec<String>,
    /// RPython `blackhole.py` `self.op_live = insns.get('live/', -1)`.
    ///
    /// A missing key stores [`majit_jitcode::insns::BC_ABSENT`] rather than
    /// upstream's `-1`.  The two carry the same guarantee for the
    /// `opcode == self.op_live` tests: `-1` cannot equal `ord(code[position])`
    /// because it is outside `0..=255`, and `BC_ABSENT` cannot because it is a
    /// reserved byte that no opname is ever assigned.
    pub op_live: u8,
    /// RPython `blackhole.py:73` `self.op_catch_exception`.
    /// `BC_ABSENT` when the key is missing — see [`Self::op_live`].
    pub op_catch_exception: u8,
    /// RPython `blackhole.py:74` `self.op_rvmprof_code`.
    /// `BC_ABSENT` when the key is missing — see [`Self::op_live`].
    pub op_rvmprof_code: u8,
    /// RPython `blackhole.py:103` `self.descrs`.
    /// Populated by `setup_descrs()` from the assembler's descriptor table.
    pub descrs: &'static dyn DescrTable,
    /// Dispatch table: opcode byte → handler fn pointer.
    /// RPython builds `dispatch_loop` closure via `unrolling_iterable`;
    /// Rust uses indirect call through this table.
    ///
    /// `Arc<Vec<...>>` so each `acquire_interp` cheaply shares the same
    /// table snapshot with the returned interpreter (RPython
    /// `blackhole.py` `self.dispatch_loop = builder.dispatch_loop`
    /// instance binding).  Wiring mutators (`setup_insns`,
    /// `wire_handler`) take `&mut self` and use `Arc::make_mut`; once
    /// wiring is done before the first `acquire_interp`, the inner Vec
    /// is uniquely owned and `make_mut` is cheap.
    pub(crate) dispatch_table: std::sync::Arc<Vec<BhOpcodeHandler>>,
    /// RPython `blackhole.py` `self.metainterp_sd = metainterp_sd`.
    /// Every interpreter resolves `jitdrivers_sd[jdindex]` through its
    /// builder (blackhole.py:1079 `sd = self.builder.metainterp_sd`,
    /// :1096 `self.builder.metainterp_sd.jitdrivers_sd[jdindex]`), so the
    /// table belongs here and `acquire_interp` hands each frame a copy.
    pub jitdrivers_sd: std::sync::Arc<[BhJitDriverSd]>,
    /// Bumped when a builder-shared field changes so a pool hit can skip
    /// recopying the six `__init__` slots (`blackhole.py`
    /// `BlackholeInterpBuilder.acquire_interp`).
    shared_fields_gen: u32,
}

impl Default for BlackholeInterpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// `blackhole.py _get_method`: the generated handler wrapper assigns
/// `self.position` and only then re-raises, so a raising instruction leaves the
/// frame pointing past itself.  Only `RaiseException` carries a coordinate
/// because only it is read again -- `handle_exception_in_frame` is the sole
/// consumer of `position` on an error path, and the two control-flow errors end
/// the frame without another dispatch.
///
/// `BlackholeInterpreter::dispatch_step` applies this for the trait path; the
/// two loops below reach the dispatch table directly, so they apply it here.
fn sync_raise_position(bh: &mut BlackholeInterpreter, err: &DispatchError) {
    if let DispatchError::RaiseException {
        resume_position, ..
    } = *err
    {
        bh.position = resume_position;
    }
}

impl BlackholeInterpBuilder {
    pub fn new() -> Self {
        Self {
            blackholeinterps: None,
            cpu: None,
            _insns: Vec::new(),
            op_live: majit_jitcode::insns::BC_ABSENT,
            op_catch_exception: majit_jitcode::insns::BC_ABSENT,
            op_rvmprof_code: majit_jitcode::insns::BC_ABSENT,
            // blackhole.py `EMPTY_LIST_I = [] # shared`.
            descrs: EMPTY_DESCR_TABLE,
            dispatch_table: std::sync::Arc::new(vec![
                unwired_handler_placeholder as BhOpcodeHandler;
                256
            ]),
            jitdrivers_sd: std::sync::Arc::from([] as [BhJitDriverSd; 0]),
            shared_fields_gen: 0,
        }
    }

    fn bump_shared_fields(&mut self) {
        self.shared_fields_gen = self.shared_fields_gen.wrapping_add(1);
    }

    /// Bind the exact CPU owned by the MetaInterp that produced the trace.
    /// The caller must keep that CPU alive and at a stable address for the
    /// builder's active lease; pooled interpreters are refreshed on acquire
    /// only when this pointer actually changes.
    pub fn set_cpu(&mut self, cpu: &dyn majit_backend::Backend) {
        let new = BlackholeCpuRef::new(cpu);
        let new_ptr = new.0 as *const dyn majit_backend::Backend as *const ();
        let old_ptr = self
            .cpu
            .map(|c| c.0 as *const dyn majit_backend::Backend as *const ());
        if old_ptr != Some(new_ptr) {
            self.cpu = Some(new);
            self.bump_shared_fields();
        }
    }

    /// TODO narrowed for parity:
    /// some Rust call sites construct a fresh builder away from the
    /// codewriter, but they already hold the cached opcode ids that
    /// RPython `setup_insns(insns)` would have stored on the builder.
    ///
    /// Copy those three cached control opcodes directly instead of
    /// consulting a synthetic global `wellknown_bh_insns()` table.
    /// Callers with a real assembler `insns` dict should still use
    /// `setup_insns`, which also fills `_insns` and `dispatch_table`
    /// exactly like `blackhole.py`.
    pub fn setup_cached_control_opcodes(
        &mut self,
        op_live: i32,
        op_catch_exception: i32,
        op_rvmprof_code: i32,
    ) {
        let to_u8 = |opcode: i32| -> u8 {
            if opcode < 0 {
                majit_jitcode::insns::BC_ABSENT
            } else {
                u8::try_from(opcode).expect("cached blackhole opcode does not fit in u8")
            }
        };
        let op_live = to_u8(op_live);
        let op_catch_exception = to_u8(op_catch_exception);
        let op_rvmprof_code = to_u8(op_rvmprof_code);
        if self.op_live != op_live
            || self.op_catch_exception != op_catch_exception
            || self.op_rvmprof_code != op_rvmprof_code
        {
            self.op_live = op_live;
            self.op_catch_exception = op_catch_exception;
            self.op_rvmprof_code = op_rvmprof_code;
            self.bump_shared_fields();
        }
    }

    /// RPython `blackhole.py` `setup_insns(insns)`.
    ///
    /// ```python
    /// def setup_insns(self, insns):
    ///     assert len(insns) <= 256, "too many instructions!"
    ///     self._insns = [None] * len(insns)
    ///     for key, value in insns.items():
    ///         assert self._insns[value] is None
    ///         self._insns[value] = key
    ///     self.op_live = insns.get('live/', -1)
    ///     self.op_catch_exception = insns.get('catch_exception/L', -1)
    ///     self.op_rvmprof_code = insns.get('rvmprof_code/ii', -1)
    /// ```
    ///
    /// Builds the reverse opcode table and a same-sized dispatch table of
    /// placeholders. [`wire_bhimpl_handlers`] then binds each key to its
    /// handler, the two-phase split of RPython `setup_insns` + `_get_method`.
    pub fn setup_insns(&mut self, insns: &indexmap::IndexMap<String, u8>) {
        assert!(insns.len() <= 256, "too many instructions!");
        // RPython blackhole.py:68-71: build reverse table.
        //
        // RPython sizes `_insns` by `len(insns)` because the assembler
        // numbers densely `0..n-1`. Pyre's `BC_*` bytes are sparse, so a
        // u8 opcode indexes a 256-slot table. Empty gaps stay `String::new()`
        // and the unwired placeholder (LeaveFrame on Grain).
        let table_len = 256;
        self._insns = vec![String::new(); table_len];
        for (key, &value) in insns {
            // `blackhole.py` `assert self._insns[value] is None`:
            // every byte slot is filled exactly once across the
            // forward map.  Pyre's `wellknown_bh_insns` +
            // `extension_insns` + `pipeline.insns` union must
            // not insert two distinct keys at the same byte.  Empty
            // gaps between sparse `BC_*` constants stay
            // `String::new()` and surface as the unwired-handler
            // placeholder if dispatched against.
            let slot = &mut self._insns[value as usize];
            assert!(
                slot.is_empty(),
                "setup_insns: byte {value} already bound to {slot:?}, refusing duplicate {key:?}",
            );
            *slot = key.clone();
        }
        // RPython blackhole.py:72-74: resolve well-known opcodes
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        self.op_rvmprof_code = insns.get("rvmprof_code/ii").copied().unwrap_or(u8::MAX);
        // RPython blackhole.py: build handler table.
        //
        // RPython immediately calls _get_method(name, argcodes) for every
        // insns key and panics if the corresponding bhimpl_* is missing.
        // We match that behavior: the default handler panics with the
        // opname so missing bhimpl_* methods surface at dispatch time
        // instead of being silently swallowed.
        self.dispatch_table = std::sync::Arc::new(vec![
            unwired_handler_placeholder
                as BhOpcodeHandler;
            self._insns.len()
        ]);
        self.bump_shared_fields();
    }

    /// List of opnames whose dispatch table entry is still the
    /// `setup_insns` placeholder — i.e. no `bhimpl_*` handler has been
    /// wired for them yet.
    ///
    /// RPython has no analogue: `setup_insns` (blackhole.py) resolves
    /// every entry via `_get_method` and would raise `AttributeError`
    /// immediately for a missing `bhimpl_*`. pyre splits that into two
    /// phases — `setup_insns` fills placeholders, `wire_bhimpl_handlers`
    /// resolves them — so the final fail-fast check lives at the caller
    /// (see `build_default_bh_builder` in `pyre-jit-trace`). This
    /// accessor is the diagnostic handle used by that assertion.
    pub fn unwired_opnames(&self) -> Vec<&str> {
        let placeholder = unwired_handler_placeholder as BhOpcodeHandler;
        self._insns
            .iter()
            .enumerate()
            .filter_map(|(i, key)| {
                if key.is_empty() {
                    return None;
                }
                if self.dispatch_table[i] as usize == placeholder as usize {
                    Some(key.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// RPython `blackhole.py` `setup_descrs(descrs)`.
    pub fn setup_descrs(&mut self, descrs: &'static dyn DescrTable) {
        if !std::ptr::addr_eq(self.descrs, descrs) {
            self.descrs = descrs;
            self.bump_shared_fields();
        }
    }

    /// Publish the jitdriver table every interpreter this builder hands out
    /// will resolve `jdindex` against — the builder-side half of
    /// `blackhole.py` `self.metainterp_sd = metainterp_sd` (read back at
    /// :1079 / :1096).  Call before `acquire_interp`: a chain built from a
    /// builder whose table is still empty gives every frame an empty table,
    /// and `bhimpl_jit_merge_point`'s recursive-portal branch indexes it
    /// directly.
    pub fn setup_jitdrivers_sd(
        &mut self,
        jitdrivers_sd: impl Into<std::sync::Arc<[BhJitDriverSd]>>,
    ) {
        let jitdrivers_sd = jitdrivers_sd.into();
        if !std::sync::Arc::ptr_eq(&self.jitdrivers_sd, &jitdrivers_sd) {
            self.jitdrivers_sd = jitdrivers_sd;
            self.bump_shared_fields();
        }
    }

    /// RPython `blackhole.py` `dispatch_loop(self, code, position)`.
    ///
    /// Runs the codewriter-orthodox bytecode dispatch loop. Each iteration
    /// reads one opcode byte, looks up the handler in `dispatch_table`,
    /// and calls it to advance position.
    pub fn dispatch_loop(
        &self,
        bh: &mut BlackholeInterpreter,
        code: &[u8],
        mut position: usize,
    ) -> Result<(), DispatchError> {
        loop {
            // RPython `blackhole.py:86-91`:
            //
            // ```python
            // if (not we_are_translated()
            //     and self.jitcode._startpoints is not None):
            //     assert position in self.jitcode._startpoints, (
            //         "the current position %d is in the middle of "
            //         "an instruction!" % position)
            // ```
            //
            // pyre is "non-translated" today; `_startpoints is None`
            // (jitcode.py:24 default) is the assembler's opt-out
            // signal. Assembled jitcodes always carry `Some(set)`
            // (assembler.rs `make_jitcode`); helper jitcodes built
            // without the assembler keep `None` and skip the check.
            if let Some(startpoints) = bh.jitcode.startpoints.as_ref() {
                debug_assert!(
                    startpoints.contains(&position),
                    "dispatch_loop: position {position} is in the middle of an instruction \
                     (jitcode {:?})",
                    bh.jitcode.name,
                );
            }
            let opcode = code[position] as usize;
            position += 1;
            if opcode >= self.dispatch_table.len() {
                panic!("bad opcode {opcode} at position {}", position - 1);
            }
            position = match self.dispatch_table[opcode](bh, code, position) {
                Ok(next) => next,
                Err(err) => {
                    sync_raise_position(bh, &err);
                    return Err(err);
                }
            };
        }
    }

    /// `dispatch_loop` with a per-opcode probe hook (Phase D-2 shadow
    /// execution scaffolding).
    ///
    /// TODO: RPython's `BlackholeInterpBuilder.dispatch_loop`
    /// has no probe parameter because upstream runs a single dispatch
    /// path (jitcode). Pyre is mid-migration: trait-based
    /// `MIFrame::execute_opcode_step` (trace_opcode.rs) and this
    /// codewriter-orthodox jitcode path coexist while Phase D
    /// validates the latter against the former. The probe lets a
    /// shadow-execute caller capture the jitcode op sequence so it
    /// can compare against the trace-side IR emitted by
    /// `execute_opcode_step`.
    ///
    /// Convergence path: removed in Phase E (eval-loop automation
    /// plan) when trait dispatch is deleted and `dispatch_loop`
    /// becomes the single production path. Until then this method
    /// stays alongside `dispatch_loop` without disturbing it — the
    /// non-probe loop is verbatim RPython parity, the probe variant
    /// is the opt-in transitional surface.
    ///
    /// `probe(bh_view, pc, opcode_byte, opname_key)` fires once per
    /// dispatched opcode, BEFORE the handler runs.
    ///
    /// * `bh_view` — shared reborrow of the interpreter. The probe
    ///   can read `registers_i/r/f`, `tmpreg_*`, `descrs`, etc. to
    ///   capture the input data flow consumed by the upcoming
    ///   handler. The reborrow expires at the end of the probe call,
    ///   so the handler runs against the same `&mut BlackholeInterpreter`
    ///   the loop holds.
    /// * `pc` — position of the opcode byte (not the position after
    ///   operand decode). `code[pc] == opcode_byte`.
    /// * `opname_key` — the `_insns[opcode]` entry (e.g.
    ///   `"int_add/ii>i"`).
    pub fn dispatch_loop_with_probe(
        &self,
        bh: &mut BlackholeInterpreter,
        code: &[u8],
        mut position: usize,
        mut probe: impl FnMut(&BlackholeInterpreter, usize, u8, &str),
    ) -> Result<(), DispatchError> {
        loop {
            let pc = position;
            let opcode = code[position];
            let opcode_idx = opcode as usize;
            if opcode_idx >= self.dispatch_table.len() {
                panic!("bad opcode {opcode_idx} at position {pc}");
            }
            let opname = self._insns[opcode_idx].as_str();
            probe(&*bh, pc, opcode, opname);
            position += 1;
            position = match self.dispatch_table[opcode_idx](bh, code, position) {
                Ok(next) => next,
                Err(err) => {
                    sync_raise_position(bh, &err);
                    return Err(err);
                }
            };
        }
    }

    /// Wire a handler for a specific opname/argcodes key into the dispatch table.
    ///
    /// RPython `blackhole.py`: iterates `_insns` and calls
    /// `_get_method(name, argcodes)` for each. In Rust we wire specific
    /// opnames one by one during Phase D migration. Returns false if
    /// the opname is not present in the insns table.
    /// Wire a handler for a specific opname/argcodes key.
    ///
    /// Returns true if the key was found in the insns table.
    /// Callers in wire_bhimpl_handlers use `try_wire_handler` for optional
    /// keys (aliases that may not exist in all assembler configurations).
    pub fn wire_handler(&mut self, opname_key: &str, handler: BhOpcodeHandler) -> bool {
        for (i, key) in self._insns.iter().enumerate() {
            if key == opname_key {
                std::sync::Arc::make_mut(&mut self.dispatch_table)[i] = handler;
                self.bump_shared_fields();
                return true;
            }
        }
        false
    }

    /// Acquire an interpreter from the pool or create a new one.
    ///
    /// `blackhole.py` `BlackholeInterpBuilder.acquire_interp` pops
    /// `self.blackholeinterps` and only constructs when the list is empty.
    /// Construction (`BlackholeInterpreter.__init__`) copies the six
    /// builder-shared fields once; a pool hit returns that same object.
    #[inline]
    pub fn acquire_interp(&mut self) -> Box<BlackholeInterpreter> {
        // `res` is the pooled object itself, not a copy of it: upstream
        // returns the same instance the free list held, so a pool hit costs
        // no allocation.  The Rust equivalent is to thread the `Box` through
        // unchanged — moving the interpreter out of its box and re-boxing it
        // in `release_interp` would malloc and free once per acquire/release
        // pair, which is exactly what the free list exists to avoid.
        if let Some(mut head) = self.blackholeinterps.take() {
            self.blackholeinterps = head.back.take();
            if head.bound_gen != self.shared_fields_gen {
                self.bind_shared_fields(&mut head);
            }
            head
        } else {
            self.acquire_interp_new()
        }
    }

    /// `blackhole.py` `BlackholeInterpreter.__init__(builder)`: allocate,
    /// register the GC-traced slots, and copy the six builder-shared fields.
    #[inline(never)]
    fn acquire_interp_new(&mut self) -> Box<BlackholeInterpreter> {
        let mut bh = Box::new(BlackholeInterpreter::default());
        // The interpreter is a GC object upstream, so its ref-holding
        // fields are traced for as long as it exists. The box gives them
        // a fixed address; `Drop` unregisters.
        let ctx = &mut *bh as *mut BlackholeInterpreter;
        unsafe {
            majit_gc::shadow_stack::register_bh_interp_with_live(
                &mut (*ctx).registers_r,
                &mut (*ctx).tmpreg_r,
                &mut (*ctx).exception_last_value,
                &mut (*ctx).virtualizable_ptr,
                ctx as *const (),
            );
        }
        bh.rooted = true;
        self.bind_shared_fields(&mut bh);
        bh
    }

    /// The six builder-shared fields `BlackholeInterpreter.__init__` copies
    /// (`blackhole.py`: cpu, dispatch_loop, descrs, op_catch_exception,
    /// op_rvmprof_code, op_live) plus the jitdrivers snapshot that stands
    /// in for `self.builder.metainterp_sd`.
    fn bind_shared_fields(&self, bh: &mut BlackholeInterpreter) {
        bh.cpu = self.cpu;
        bh.descrs = self.descrs;
        bh.op_catch_exception = self.op_catch_exception;
        bh.op_rvmprof_code = self.op_rvmprof_code;
        bh.op_live = self.op_live;
        if !std::sync::Arc::ptr_eq(&bh.dispatch_table, &self.dispatch_table) {
            bh.dispatch_table = std::sync::Arc::clone(&self.dispatch_table);
        }
        if !std::sync::Arc::ptr_eq(&bh.jitdrivers_sd, &self.jitdrivers_sd) {
            bh.jitdrivers_sd = std::sync::Arc::clone(&self.jitdrivers_sd);
        }
        bh.bound_gen = self.shared_fields_gen;
    }

    /// `blackhole.py` `BlackholeInterpBuilder.release_interp`
    pub fn release_interp(&mut self, mut interp: Box<BlackholeInterpreter>) {
        interp.cleanup_registers();
        // Live-chain and free-list links have separate upstream owners.
        interp.nextblackholeinterp = None;
        interp.aborted = false;
        interp.abort_permanent_bail = false;
        interp.bail_residual = None;
        interp.got_exception = false;
        // The virtualizable handle is frame identity, not builder state, and
        // `acquire_interp` refreshes only the six builder-shared fields.  A
        // pooled interp that kept `virtualizable_ptr` would hand the next user
        // a pointer to the frame this run just finished with:
        // `blackhole_from_resumedata` never assigns either field, so
        // `call_jit.rs`'s "a novable resume runs no vable opcodes, so leave
        // both the pointer and the vinfo handle unset (null)" only holds for
        // an interp that comes back from the pool clean.  `run()` dereferences
        // `virtualizable_ptr` for every chain frame whose `virtualizable_info`
        // is non-null (`push_resume_ref_roots`), so a stale pair reads a dead
        // frame and roots its slots.  Reset with `virtualizable_stack_base`,
        // which is the same frame-scoped triple.
        interp.virtualizable_ptr = 0;
        interp.virtualizable_info = std::ptr::null();
        interp.virtualizable_stack_base = 0;
        interp.last_live_pc = usize::MAX;
        // Same footing: the layout describes the register file of the machine
        // this run resumed, so it must not outlive it.  A stale one trips the
        // `assert!` range checks in `scalar_slot`/`array_identity_slots`, or,
        // when a field index happens to be in range, names the previous
        // machine's slots. Reset it so the next user does not inherit it.
        interp.state_field_layout = StateFieldLayout::default();
        // `record_caught_exception` is a callback the resume path installs on
        // every frame of a chain (`call_jit.rs`), and the chain goes back into
        // the same pool.  Handing the next user a hook it never asked for makes
        // it report exceptions to whoever owned the frame before it.
        // `acquire_interp` refreshes only the six builder-shared slots.
        interp.record_caught_exception = None;
        interp.back = self.blackholeinterps.take();
        self.blackholeinterps = Some(interp);
    }

    /// Release an entire blackhole chain (including all nextblackholeinterps).
    pub fn release_chain(&mut self, chain: Option<Box<BlackholeInterpreter>>) {
        let mut current = chain;
        while let Some(mut bh) = current {
            let next = bh.nextblackholeinterp.take();
            self.release_interp(bh);
            current = next;
        }
    }
}

/// warmspot.py handle_jitexception parity.
///
/// Dispatches on JitException type and returns (return_type, value).
/// For ContinueRunningNormally, calls portal_runner to re-enter the
/// portal function. The portal_runner may itself raise a JitException,
/// which is returned as Err for the caller to re-dispatch (while loop).
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn handle_jitexception_dispatch(
    exc: JitException,
    portal_runner: Option<
        &dyn Fn(&JitException) -> Result<(BhReturnType, i64), PortalRunnerFailure>,
    >,
    result_kind: Option<BhReturnType>,
) -> Result<(BhReturnType, i64), PortalRunnerFailure> {
    // `warmspot.py:984-996` guards each `DoneWithThisFrame*` arm with `if
    // result_kind == '<kind>':`, so a portal only ever absorbs the flavour it
    // returns; a mismatch falls off the end of the `while True` into
    // `:1007 raise AssertionError("all cases should have been handled")`.  The
    // kinds carry the value in different banks — `DoneWithThisFrameInt` at a
    // ref portal would install an integer through `setup_return_value_i` and
    // the caller would read it as a pointer — so the wrong arm is a wrong
    // answer, not a mislabelled one.  `None` is a jitcode no driver claims, or
    // a blackhole assembled without a driver table at all (the unit-test
    // builders leave `jitdrivers_sd` empty); there is no `result_kind` to test
    // against, so the variant alone decides as it did before the driver was
    // consulted.
    let kind_matches = |kind: BhReturnType| result_kind.is_none_or(|want| want == kind);
    let done_arm = |kind: BhReturnType, value: i64| {
        assert!(
            kind_matches(kind),
            "portal result_kind {:?} cannot absorb DoneWithThisFrame{:?}",
            result_kind,
            kind,
        );
        Ok((kind, value))
    };
    match exc {
        // warmspot.py:985-987
        JitException::DoneWithThisFrameVoid => done_arm(BhReturnType::Void, 0),
        // warmspot.py:988-990
        JitException::DoneWithThisFrameInt(result) => done_arm(BhReturnType::Int, result),
        // warmspot.py:991-993
        JitException::DoneWithThisFrameRef(result) => done_arm(BhReturnType::Ref, result.0 as i64),
        // warmspot.py:994-996
        JitException::DoneWithThisFrameFloat(result) => {
            done_arm(BhReturnType::Float, result.to_bits() as i64)
        }
        // warmspot.py:998-1005
        JitException::ExitFrameWithExceptionRef(_) => Err(PortalRunnerFailure::jit(exc)),
        // Not a result: a portal level has nothing to install for it.
        JitException::BailToInterpreter => unreachable!(
            "a recursive portal bail must cross the callback boundary with its terminal image"
        ),
        // warmspot.py:970-983
        JitException::ContinueRunningNormally(_) => {
            // warmspot.py:976-978: result = portal_ptr(*args)
            // May raise JitException → Err propagated for re-dispatch.
            let runner = portal_runner.expect("ContinueRunningNormally requires portal_runner");
            runner(&exc)
        }
    }
}

/// blackhole.py _handle_jitexception_in_portal +
/// warmspot.py handle_jitexception_from_blackhole
///
/// Handle a JitException at a recursive portal level.
/// warmspot.py: result = handle_jitexception(e)
/// warmspot.py: bhcaller._setup_return_value_{i,r,f}(result)
///
/// Returns Ok(()) on success (return value set in bhcaller), or the
/// `JitException` this level could not absorb: `ExitFrameWithExceptionRef`,
/// which the caller propagates as a regular exception, or the pyre-only
/// `BailToInterpreter`, which no portal level can absorb at all -- there is no
/// result to install and no exception to deliver, only a frame the interpreter
/// has to take back.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn handle_jitexception_in_portal(
    bhcaller: &mut BlackholeInterpreter,
    exc: JitException,
    portal_runner: Option<
        &dyn Fn(&JitException) -> Result<(BhReturnType, i64), PortalRunnerFailure>,
    >,
    result_kind: Option<BhReturnType>,
) -> Result<(), PortalRunnerFailure> {
    // warmspot.py handle_jitexception: while True loop.
    // ContinueRunningNormally → portal_runner → may raise JitException → loop.
    let mut current_exc = exc;
    loop {
        match handle_jitexception_dispatch(current_exc, portal_runner, result_kind) {
            Ok((ret_type, result)) => {
                // warmspot.py:1041-1050
                match ret_type {
                    BhReturnType::Void => {}
                    BhReturnType::Int => bhcaller.setup_return_value_i(result),
                    BhReturnType::Ref => bhcaller.setup_return_value_r(result),
                    BhReturnType::Float => bhcaller.setup_return_value_f(result),
                }
                return Ok(());
            }
            Err(
                failure @ PortalRunnerFailure {
                    exception: JitException::ExitFrameWithExceptionRef(_),
                    ..
                },
            ) => {
                // warmspot.py:998-1005: raise as regular exception
                return Err(failure);
            }
            // A `portal_runner` that re-entered the portal and hit an
            // `abort_permanent` marker or an unresolved callee reports the bail
            // here.  `handle_jitexception` refuses one before its walk for the
            // same reason this arm refuses to redispatch it: dispatching a bail
            // neither invokes the runner nor changes the exception, so looping
            // back would spin on the same value forever.
            Err(
                failure @ PortalRunnerFailure {
                    exception: JitException::BailToInterpreter,
                    ..
                },
            ) => {
                return Err(failure);
            }
            Err(PortalRunnerFailure {
                exception: next_exc,
                terminal,
            }) => {
                // warmspot.py:979-980: a JitException out of `portal_runner`
                // loops back in `handle_jitexception`.  Upstream reaches the
                // same `continue` from `:967-968` as well, where
                // `EnterJitAssembler.execute()` raised; pyre has no such
                // exception because `warmstate.py maybe_compile_and_run`'s
                // counterpart in `eval.rs` enters the assembler directly, so
                // that producer has no port and this arm serves the portal
                // re-entry alone.
                assert!(
                    terminal.is_none(),
                    "only BailToInterpreter may carry a terminal blackhole image"
                );
                current_exc = next_exc;
                continue;
            }
        }
    }
}

/// Register-bank image of the blackhole frame whose `JitException` propagated
/// out of `_run_forever`.
///
/// `_run_forever` releases that frame back to the pool before the exception
/// leaves, so a `ContinueRunningNormally` handoff that has to materialize the
/// resumed frame's live operand-stack slots cannot read the banks afterwards.
/// The three banks are moved out (not copied) before the release, matching
/// [`crate::jitdriver::SingleFrameBlackholeResult`], which keeps the terminal
/// interpreter of the single-frame path logically alive the same way.
pub struct BlackholeTerminalImage {
    pub jitcode_index: usize,
    pub registers_i: Vec<i64>,
    pub registers_r: Vec<i64>,
    pub registers_f: Vec<i64>,
    pub position: usize,
    pub last_opcode_position: usize,
    /// [`BlackholeInterpreter::abort_permanent_bail`] of the frame this image
    /// was taken from, so a consumer of `BailToInterpreter` can tell the abort
    /// that stamped a resume coordinate from the one that did not.  Without it
    /// the exception says only *that* the chain bailed, and every consumer has
    /// to assume the boundary it cannot check.
    pub abort_permanent_bail: bool,
    /// Residual named by the frame that bailed. `None` when the bail
    /// was not an unbound call.
    pub bail_residual: Option<String>,
}

/// Result that escaped a recursive portal runner.
///
/// PyPy propagates the live RPython blackhole object with the raised
/// `JitException`.  Pyre releases that object before crossing the Rust callback
/// boundary, so a pyre-only interpreter bail must carry the equivalent bank
/// image explicitly.  Real JitExceptions need no image because the enclosing
/// blackhole chain consumes them exactly as `_handle_jitexception_in_portal`
/// does upstream.
pub struct PortalRunnerFailure {
    exception: JitException,
    terminal: Option<BlackholeTerminalImage>,
}

impl PortalRunnerFailure {
    pub fn jit(exception: JitException) -> Self {
        assert!(!matches!(exception, JitException::BailToInterpreter));
        Self {
            exception,
            terminal: None,
        }
    }

    pub fn bail(terminal: BlackholeTerminalImage) -> Self {
        Self {
            exception: JitException::BailToInterpreter,
            terminal: Some(terminal),
        }
    }
}

impl BlackholeTerminalImage {
    fn take_from(bh: &mut BlackholeInterpreter) -> Self {
        Self {
            jitcode_index: bh.jitcode.index(),
            registers_i: std::mem::take(&mut bh.registers_i),
            registers_r: std::mem::take(&mut bh.registers_r),
            registers_f: std::mem::take(&mut bh.registers_f),
            position: bh.position,
            last_opcode_position: bh.last_opcode_position,
            abort_permanent_bail: bh.abort_permanent_bail,
            bail_residual: bh.bail_residual.take(),
        }
    }
}

/// blackhole.py _handle_jitexception
///
/// Route a JitException through the blackhole frame chain.
/// Walks up the chain until a portal frame is found. If the portal
/// is the bottommost frame, the exception propagates out. Otherwise
/// it's handled at the recursive portal level.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn handle_jitexception(
    builder: &mut BlackholeInterpBuilder,
    mut bh: Box<BlackholeInterpreter>,
    exc: JitException,
    portal_runner: Option<
        &dyn Fn(&JitException) -> Result<(BhReturnType, i64), PortalRunnerFailure>,
    >,
    on_leave_level: Option<&dyn Fn(i64)>,
    terminal_out: Option<&mut Option<BlackholeTerminalImage>>,
) -> Result<(Box<BlackholeInterpreter>, i64), JitException> {
    // A bail is not an outcome any portal level can absorb — there is no
    // result to install and no exception to deliver, only a frame the
    // interpreter has to take back — so it leaves `_run_forever` directly
    // instead of walking to a portal.  Upstream has no such exit; the walk
    // below is `blackhole.py _handle_jitexception`, which only ever sees the
    // five real jitexc classes.
    if matches!(exc, JitException::BailToInterpreter) {
        if let Some(terminal_out) = terminal_out {
            *terminal_out = Some(BlackholeTerminalImage::take_from(&mut bh));
        }
        builder.release_chain(Some(bh));
        return Err(exc);
    }
    // blackhole.py:1764: while blackholeinterp.jitcode.jitdriver_sd is None
    while bh.jitcode.jitdriver_sd().is_none() {
        let next = bh.nextblackholeinterp.take();
        // `pyopcode.py handle_operation_error` stores
        // `frame_finished_execution = True` on the route that propagates out
        // of a frame without a handler, and `executioncontext.py leave`
        // takes `got_exception`. `RETURN_VALUE`'s own store is the jitcode
        // `setfield_gc` the blackhole replays; this unwind has no such op.
        // These levels are the ones the walk abandons on its way to the
        // portal: their chain link is dropped here and never resumes, so
        // this is where each one leaves. Without the callback a traceback
        // that retains such a frame refuses `frame.clear()`.
        if let Some(on_leave_level) = on_leave_level {
            on_leave_level(bh.virtualizable_ptr);
        }
        builder.release_interp(bh);
        match next {
            Some(caller) => bh = caller,
            None => return Err(exc), // no portal found
        }
    }

    // blackhole.py:1767-1769
    if bh.nextblackholeinterp.is_none() {
        // Bottommost entry: exception goes through
        if let Some(terminal_out) = terminal_out {
            *terminal_out = Some(BlackholeTerminalImage::take_from(&mut bh));
        }
        builder.release_interp(bh);
        return Err(exc);
    }

    // blackhole.py:1770-1780: recursive portal level.
    // _handle_jitexception_in_portal(exc) calls jd.handle_jitexc_from_bh,
    // which is warmspot.py handle_jitexception_from_blackhole:
    //   result = handle_jitexception(e)
    //   bhcaller._setup_return_value_{i,r,f}(result)
    //
    // handle_jitexception (warmspot.py) extracts the result from
    // DoneWithThisFrame{Int,Ref,Float,Void} and returns it.
    //
    // In Rust we can do this directly since JitException carries the result.
    //
    // `bh` is upstream's `self` — the portal frame that raised — so its own
    // driver owns the runner, and `_run_forever`'s walk stopped here precisely
    // because this frame carries a driver stamp.  `portal_dispatch_for` hands
    // back that driver's runner and kind together, or — when it registered no
    // runner of its own — someone else's runner and no kind at all, so the two
    // never describe different drivers.
    let (own_runner, result_kind) = portal_dispatch_for(&bh);
    let own_runner = own_runner.as_ref().map(|hook| {
        hook as &dyn Fn(
            &crate::jitexc::JitException,
        ) -> Result<(BhReturnType, i64), PortalRunnerFailure>
    });
    let caller = bh.nextblackholeinterp.as_mut().unwrap();
    let portal_outcome =
        handle_jitexception_in_portal(caller, exc, own_runner.or(portal_runner), result_kind);
    let current_exc = match portal_outcome {
        Ok(()) => 0,
        Err(PortalRunnerFailure {
            exception: JitException::ExitFrameWithExceptionRef(exc_ref),
            ..
        }) => exc_ref.0 as i64,
        // A recursive runner's bail belongs to the re-entered portal, not to
        // `bh` (the outer portal level whose CALL is waiting for a result).
        // PyPy still owns that inner blackhole object while the exception
        // unwinds; pyre's callback therefore returns its moved bank image and
        // this level publishes it before releasing the outer chain.  Using
        // `bh` here would resume at the CALL and replay the inner effects.
        Err(PortalRunnerFailure {
            exception: exc @ JitException::BailToInterpreter,
            terminal: Some(terminal),
        }) => {
            if let Some(terminal_out) = terminal_out {
                *terminal_out = Some(terminal);
            }
            builder.release_chain(Some(bh));
            return Err(exc);
        }
        Err(PortalRunnerFailure {
            exception: JitException::BailToInterpreter,
            terminal: None,
        }) => unreachable!("recursive portal runner returned a bail without its terminal image"),
        // `handle_jitexception_in_portal` returns only those two.
        Err(PortalRunnerFailure {
            exception: other, ..
        }) => unreachable!("portal level reported {other:?}"),
    };
    // blackhole.py:1780: return blackholeinterp, lle
    Ok((bh, current_exc))
}

/// blackhole.py `resume_in_blackhole`.
///
/// `blackhole_from_resumedata` has already built `blackholeinterp` and
/// the caller has ended `deadframe`'s live range. `_prepare_resume_from_failure`
/// is `grab_exc_value`'s result, then `_run_forever`.
pub fn resume_in_blackhole(
    builder: &mut BlackholeInterpBuilder,
    blackholeinterp: Box<BlackholeInterpreter>,
    deadframe_exc: i64,
    on_enter_level: Option<&dyn Fn(i64)>,
    on_leave_level: Option<&dyn Fn(i64)>,
) -> JitException {
    let current_exc = BlackholeInterpreter::prepare_resume_from_failure(deadframe_exc);
    run_forever_with_portal(
        builder,
        blackholeinterp,
        current_exc,
        None,
        on_enter_level,
        on_leave_level,
        None,
    )
}

/// blackhole.py _run_forever
///
/// Execute a blackhole frame chain to completion.
/// Loops through frames: runs each one via `resume_mainloop`, releases it,
/// then moves to the caller frame. Terminates when the bottommost frame
/// raises a JitException (DoneWithThisFrame* or ExitFrameWithException*).
///
/// Returns the JitException that terminated execution.
pub fn run_forever(
    builder: &mut BlackholeInterpBuilder,
    bh: Box<BlackholeInterpreter>,
    current_exc: i64,
) -> JitException {
    run_forever_with_portal(builder, bh, current_exc, None, None, None, None)
}

/// blackhole.py _run_forever with optional portal runner callback.
///
/// `portal_runner` is warmspot.py handle_jitexception parity:
/// when ContinueRunningNormally is raised at a recursive portal level,
/// this callback re-enters the portal function with the exception's
/// green/red args and returns the result.
///
/// `terminal_out` receives the [`BlackholeTerminalImage`] of the bottommost
/// frame when its exception propagates out, so the caller can read the banks
/// the release would otherwise recycle.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn run_forever_with_portal(
    builder: &mut BlackholeInterpBuilder,
    mut bh: Box<BlackholeInterpreter>,
    mut current_exc: i64,
    portal_runner: Option<
        &dyn Fn(&JitException) -> Result<(BhReturnType, i64), PortalRunnerFailure>,
    >,
    on_enter_level: Option<&dyn Fn(i64)>,
    on_leave_level: Option<&dyn Fn(i64)>,
    mut terminal_out: Option<&mut Option<BlackholeTerminalImage>>,
) -> JitException {
    loop {
        // executioncontext.py enter: publish this level's frame as
        // `ec.topframeref` before its residuals run (pyjitpl multi-frame
        // resume only; None on the guard-failure path where the chain is
        // already correct via the compiled trace's real stores).
        if let Some(on_enter_level) = on_enter_level {
            on_enter_level(bh.virtualizable_ptr);
        }
        // blackhole.py:1754-1755
        match bh.resume_mainloop(current_exc) {
            Ok(exc) => {
                current_exc = exc;
            }
            Err(jit_exc) => {
                // blackhole.py:1756-1758
                match handle_jitexception(
                    builder,
                    bh,
                    jit_exc,
                    portal_runner,
                    on_leave_level,
                    terminal_out.as_deref_mut(),
                ) {
                    Ok((new_bh, exc)) => {
                        // Handled at a recursive portal level.  `new_bh` is
                        // that portal frame (blackhole.py:1780 returns
                        // `blackholeinterp`), and its caller already holds the
                        // re-entered portal's result
                        // (`_handle_jitexception_in_portal` →
                        // warmspot.py:1041-1050), so the frame is finished.
                        // Fall through to the shared release + step below —
                        // blackhole.py sit outside the `try`, so
                        // they run on this arm too.  Re-entering `new_bh`
                        // instead would execute its tail a second time and
                        // overwrite the installed result.
                        bh = new_bh;
                        current_exc = exc;
                    }
                    Err(propagated_exc) => {
                        // Bottommost or unhandled — propagate out
                        return propagated_exc;
                    }
                }
            }
        }

        // blackhole.py:1759
        let next = bh.nextblackholeinterp.take();
        // The level reached here has returned to its caller. `RETURN_VALUE`'s
        // `frame_finished_execution` store is the jitcode `setfield_gc` this
        // interpreter just replayed; the callback still runs because the
        // embedder hangs per-leave bookkeeping on it and because
        // `handle_operation_error` has no jitcode store of its own.
        // Threaded from the interpreter side for the same reason as
        // `on_enter_level` — the frame object is the embedder's, which
        // majit-metainterp cannot name. A level that unwinds instead of
        // returning never arrives here; `handle_jitexception` calls the same
        // callback for each level it abandons on its walk.
        if let Some(on_leave_level) = on_leave_level {
            on_leave_level(bh.virtualizable_ptr);
        }
        builder.release_interp(bh);
        // blackhole.py:1760
        // RPython: blackholeinterp = blackholeinterp.nextblackholeinterp
        // In RPython this can be None, but _resume_mainloop on the
        // bottommost frame always raises a JitException (via
        // _done_with_this_frame or _exit_frame_with_exception), so
        // this code is unreachable in normal operation.
        bh = next.expect("_run_forever: nextblackholeinterp is None (unreachable)");
    }
}

/// blackhole.py convert_and_run_from_pyjitpl
///
/// Get a chain of blackhole interpreters and fill them by copying
/// 'metainterp.framestack'.
pub struct PyjitplBlackholeFrameConfig<'a> {
    pub state_field_layout: StateFieldLayout,
    pub virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    pub virtualizable_ptr: i64,
    pub virtualizable_stack_base: usize,
    /// Per-frame concrete frame ptr + its own stack_base, aligned to
    /// `framestack.frames` (outermost-first).  When present, each blackhole
    /// level uses ITS OWN frame as the virtualizable instead of sharing the
    /// portal's, and `on_enter_level` publishes it as `ec.topframeref`.
    pub per_frame: Option<&'a [(i64, usize)]>,
    /// `executioncontext.py enter`: publish `ec.topframeref = <this level's
    /// frame>` before each frame runs so a re-executed `sys._getframe` observes
    /// the resumed frame chain.  Threaded from the interpreter side because
    /// majit-metainterp cannot reference `ExecutionContext`.
    pub on_enter_level: Option<&'a dyn Fn(i64)>,
    /// Leave hook for one blackhole level. `pyopcode.py RETURN_VALUE`
    /// stores `frame_finished_execution` in the jitcode this interpreter
    /// replays; `handle_operation_error` has no such op, so the embedder's
    /// callback finishes that propagation. Called once per level that
    /// returns to its caller, with that level's `virtualizable_ptr`.
    /// Threaded from the interpreter side for the same reason as
    /// [`Self::on_enter_level`].
    ///
    /// Set it only alongside [`Self::per_frame`], which is what makes that
    /// pointer name the level's OWN frame.  Without it every level shares the
    /// portal's virtualizable, and a nested level would hand back the frame
    /// ABOVE it — marking a caller that is still running as finished.
    pub on_leave_level: Option<&'a dyn Fn(i64)>,
}

pub fn convert_and_run_from_pyjitpl(
    builder: &mut BlackholeInterpBuilder,
    framestack: &MIFrameStack,
    last_exc_value: i64,
    raising_exception: bool,
    config: Option<PyjitplBlackholeFrameConfig<'_>>,
    terminal_out: Option<&mut Option<BlackholeTerminalImage>>,
) -> JitException {
    // blackhole.py:1803-1810
    let mut next_bh: Option<Box<BlackholeInterpreter>> = None;
    let roots_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    let on_enter_level = config.as_ref().and_then(|config| config.on_enter_level);
    let on_leave_level = config.as_ref().and_then(|config| config.on_leave_level);

    for (frame_index, frame) in framestack.frames.iter().enumerate() {
        let mut cur_bh = builder.acquire_interp();
        cur_bh.copy_data_from_miframe(frame);
        if let Some(config) = config.as_ref() {
            cur_bh.state_field_layout = config.state_field_layout.clone();
            cur_bh.virtualizable_info = config.virtualizable_info;
            cur_bh.virtualizable_ptr = config.virtualizable_ptr;
            cur_bh.virtualizable_stack_base = config.virtualizable_stack_base;
            if let Some(per_frame) = config.per_frame {
                let (frame_ptr, frame_stack_base) = per_frame[frame_index];
                cur_bh.virtualizable_ptr = frame_ptr;
                cur_bh.virtualizable_stack_base = frame_stack_base;
            }
        }
        // Keep every completed interpreter bank rooted while later frames are
        // acquired and throughout `run_forever`.  The chain is consumed and
        // released there, so these banks cannot be recovered by moving them
        // out after the run.
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(cur_bh.registers_r.as_mut_slice());
        }
        cur_bh.nextblackholeinterp = next_bh;
        next_bh = Some(cur_bh);
    }

    let Some(mut first_bh) = next_bh else {
        majit_gc::shadow_stack::pop_resume_ref_roots_to(roots_depth);
        // `blackhole.py convert_and_run_from_pyjitpl` unconditionally reads
        // `firstbh.exception_last_value`: its caller invariant is a non-empty
        // portal framestack.  A silent BailToInterpreter turns corruption of
        // that invariant into an apparently resumable interpreter exit.
        panic!("convert_and_run_from_pyjitpl: metainterp framestack is empty");
    };
    // blackhole.py:1812-1818
    let current_exc = if raising_exception {
        last_exc_value
    } else {
        first_bh.exception_last_value = last_exc_value;
        0
    };

    // `warmspot.py` hands `ll_portal_runner` to the blackhole so a
    // recursive portal level can re-enter the portal function.  Without it
    // `handle_jitexception_dispatch` has nothing to call and
    // `ContinueRunningNormally` reaches its `expect`.
    //
    // This is only the fallback for the entry frame, which has not passed
    // `_run_forever`'s walk and so need not carry a driver stamp: the recursive
    // level resolves its own runner from the raising portal frame
    // (`portal_dispatch_for`), because the driver that owns that jitcode is the
    // one whose portal has to be re-entered.
    let (hook, _) = portal_dispatch_for(&first_bh);
    let portal_runner = hook.as_ref().map(|hook| {
        hook as &dyn Fn(
            &crate::jitexc::JitException,
        ) -> Result<(BhReturnType, i64), PortalRunnerFailure>
    });
    let outcome = run_forever_with_portal(
        builder,
        first_bh,
        current_exc,
        portal_runner,
        on_enter_level,
        on_leave_level,
        terminal_out,
    );
    majit_gc::shadow_stack::pop_resume_ref_roots_to(roots_depth);
    outcome
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor: `rpython/jit/metainterp/test/test_blackhole.py`
    //! for interpreter pooling, exception-state handling, and blackhole control
    //! flow.
    //!
    //! The opcode-by-opcode coverage below drives the orthodox runtime
    //! helpers (`bhimpl_*`, `eval_*`) directly. PyPy keeps no dedicated
    //! `test_executor.py` in this tree, so these tests pin down the local
    //! helpers and arithmetic edge cases directly.

    use super::*;

    #[test]
    fn test_read_list_keeps_small_inline_call_args_off_heap() {
        // blackhole.py `_get_list_of_values` + `make_sure_not_resized`.
        // handler_inline_call_ir_i reads a 1-item I list and a 2-item R
        // list; a Vec<i64> each was the 8 B / 16 B class on the regex
        // and/or timed row.
        let regs = [10i64, 20, 30, 40];
        let code = [1u8, 0, 2, 1, 2];
        let (one, pos) = read_list_from_regs(&regs, &code, 0);
        assert_eq!(pos, 2);
        assert_eq!(&one[..], &[10]);
        assert!(
            !one.spilled(),
            "a 1-item I/R/F list must stay inline (8 B Vec was the timed-row class)"
        );
        let (two, pos) = read_list_from_regs(&regs, &code, 2);
        assert_eq!(pos, 5);
        assert_eq!(&two[..], &[20, 30]);
        assert!(
            !two.spilled(),
            "a 2-item I/R/F list must stay inline (16 B Vec was the timed-row class)"
        );
    }

    // ── StateFieldLayout field→slot mapping tests (#183) ──

    #[test]
    fn state_field_layout_scalar_slots_are_identity() {
        // No arrays, base 0: scalars fill `[0..num_scalars]`, slot == field_idx.
        let layout = StateFieldLayout::new(4, 0, 0, 0);
        assert_eq!(layout.total_slots(), 4);
        for i in 0..4 {
            assert_eq!(layout.scalar_slot(i), i);
        }
    }

    #[test]
    fn state_field_layout_int_scalar_base_offsets_slots() {
        // int_scalar_base shifts every int identity slot past the dispatch
        // JitCode's int arguments (pc at i0): scalars at i1..=i2, then ptr
        // then len of one plain `[int]`.
        let layout = StateFieldLayout::new(2, 1, 0, 1);
        assert_eq!(layout.total_slots(), 2 + 2);
        assert_eq!(layout.scalar_slot(0), 1);
        assert_eq!(layout.scalar_slot(1), 2);
        assert_eq!(layout.array_identity_slots(0), (3, 4));
    }

    #[test]
    fn state_field_layout_one_scalar_then_one_plain_int_array() {
        // One `int` scalar (slot 0) followed by one plain `[int]` as ptr
        // then len.
        let layout = StateFieldLayout::new(1, 1, 0, 0);
        assert_eq!(layout.total_slots(), 1 + 2);
        assert_eq!(layout.num_plain_int_arrays, 1);
        assert_eq!(layout.scalar_slot(0), 0);
        assert_eq!(layout.array_identity_slots(0), (1, 2));
    }

    #[test]
    fn state_field_layout_total_matches_live_slots_helper() {
        // The struct's total must equal the canonical liveness slot count.
        // `live_slots_for_state_field_jit` still takes ptr then len baked
        // into its `num_scalars` argument with empty `array_lens`.
        for &(s, n_arrays, v) in &[(1usize, 0usize, 1usize), (1, 1, 0), (2, 2, 2), (0, 0, 0)] {
            let layout = StateFieldLayout::new(s, n_arrays, v, 0);
            let int_identity = s + 2 * n_arrays;
            let (live_i, live_r, live_f) =
                crate::live_slots_for_state_field_jit(int_identity, &[], v, 0, 0, 0, 0, 0);
            assert_eq!(layout.total_slots(), live_i.len());
            assert!(live_r.is_empty() && live_f.is_empty());
        }
    }

    #[test]
    fn state_field_layout_total_live_values_includes_ref_scalars() {
        // Three int scalars plus one ref scalar, the latter past a ref arg
        // at r0 → base 1.
        // total_slots() counts only int-bank slots; the ref scalar lives in
        // the ref bank and is appended by extract_live, so
        // total_live_values() = total_slots() + num_ref_scalars. The
        // trace-start / run-compiled gate compares the flat extract_live count
        // against total_live_values(), not total_slots().
        let layout = StateFieldLayout::with_ref_scalars(3, 0, 0, 1, 1, 0);
        assert_eq!(layout.total_slots(), 3);
        assert_eq!(layout.total_live_values(), 4);
        assert_eq!(layout.ref_scalar_slot(0), 1);
        let (live_i, live_r, live_f) =
            crate::live_slots_for_state_field_jit(3, &[], 0, 1, 1, 0, 0, 0);
        assert_eq!(layout.total_slots(), live_i.len());
        assert_eq!(layout.num_ref_scalars, live_r.len());
        assert_eq!(live_r, vec![1]);
        assert_eq!(layout.total_live_values(), live_i.len() + live_r.len());
        assert!(live_f.is_empty());
    }

    // ── state_field handler register-slot tests (#183) ──
    #[test]
    fn handler_state_field_moves_between_register_slots() {
        // Two scalars: slot 0, slot 1. load_state_field copies a scalar slot
        // into a working register; store_state_field copies back.
        let mut bh = BlackholeInterpreter::default();
        bh.state_field_layout = StateFieldLayout::new(2, 0, 0, 0);
        bh.registers_i = vec![10, 20, 0, 0];

        // load_state_field/di: field_idx=1 (u16 LE) → dest reg 2.
        let next = handler_load_state_field_di(&mut bh, &[1, 0, 2], 0).unwrap();
        assert_eq!(next, 3);
        assert_eq!(bh.registers_i[2], 20);

        // store_state_field/di: src reg 2 (now holds 20+something) → field_idx=0.
        bh.registers_i[2] = 99;
        let next = handler_store_state_field_di(&mut bh, &[0, 0, 2], 0).unwrap();
        assert_eq!(next, 3);
        assert_eq!(bh.registers_i[0], 99);
    }

    #[test]
    fn handler_state_field_ref_moves_between_ref_register_slots() {
        // Two ref scalars at ref slots 1 and 2 (base 1: the `program` ref
        // arg keeps r0). load_state_field_ref copies a ref slot into a
        // working ref register; store_state_field_ref copies back. Raw
        // pointer bits round-trip.
        let mut bh = BlackholeInterpreter::default();
        bh.state_field_layout = StateFieldLayout::with_ref_scalars(0, 0, 0, 2, 1, 0);
        bh.registers_r = vec![0x9999, 0xAAAA, 0xBBBB, 0, 0];

        // load_state_field_ref/dr: field_idx=1 (u16 LE) → dest ref reg 3.
        let next = handler_load_state_field_ref_dr(&mut bh, &[1, 0, 3], 0).unwrap();
        assert_eq!(next, 3);
        assert_eq!(bh.registers_r[3], 0xBBBB);

        // store_state_field_ref/dr: src ref reg 3 → field_idx=0 (slot 1).
        bh.registers_r[3] = 0xCCCC;
        let next = handler_store_state_field_ref_dr(&mut bh, &[0, 0, 3], 0).unwrap();
        assert_eq!(next, 3);
        assert_eq!(bh.registers_r[1], 0xCCCC);
        // The argument register below the base is untouched.
        assert_eq!(bh.registers_r[0], 0x9999);

        // The int bank is untouched by the ref handlers.
        assert!(bh.registers_i.is_empty() || bh.registers_i.iter().all(|&v| v == 0));
    }

    // ── Executor opcode tests (bhimpl_* runtime coverage) ──
    //
    // Systematic correctness tests for each opcode category, driven directly
    // against the orthodox runtime helpers that the jitcode
    // `BlackholeInterpreter` dispatches to: `blackhole.py` `bhimpl_*` for the
    // ops that have one, `pyjitpl/dispatch.rs` `eval_*` for float
    // compares + float floordiv/mod, and `support.py` `_ll_2_int_{floordiv,mod}`
    // for integer division (`jtransform.py _do_builtin_call` residual).

    /// Dispatch a single binary opcode to its orthodox runtime helper and
    /// return the i64 result (float results are returned as `f64::to_bits`).
    fn exec_binop(opcode: OpCode, a: i64, b: i64) -> i64 {
        match opcode {
            // intmask(a OP b) — blackhole.py. INT_*_OVF run the same
            // wrapping body in the blackhole; overflow detection is the
            // separate `int_*_jump_if_ovf` bytecode (blackhole.py).
            OpCode::IntAdd | OpCode::IntAddOvf => bhimpl_int_add(a, b),
            OpCode::IntSub | OpCode::IntSubOvf => bhimpl_int_sub(a, b),
            OpCode::IntMul | OpCode::IntMulOvf => bhimpl_int_mul(a, b),
            // INT_FLOORDIV / INT_MOD are residual calls to the C-truncating
            // helpers (support.py:255-271). The zero-divisor and INT_MIN/-1
            // corners are guarded out of the trace before the call.
            OpCode::IntFloorDiv => _ll_2_int_floordiv(a, b),
            OpCode::IntMod => _ll_2_int_mod(a, b),
            OpCode::IntAnd => bhimpl_int_and(a, b),
            OpCode::IntOr => bhimpl_int_or(a, b),
            OpCode::IntXor => bhimpl_int_xor(a, b),
            OpCode::IntLshift => bhimpl_int_lshift(a, b),
            OpCode::IntRshift => bhimpl_int_rshift(a, b),
            OpCode::UintRshift => bhimpl_uint_rshift(a, b),
            OpCode::IntLt => bhimpl_int_lt(a, b),
            OpCode::IntLe => bhimpl_int_le(a, b),
            OpCode::IntGt => bhimpl_int_gt(a, b),
            OpCode::IntGe => bhimpl_int_ge(a, b),
            OpCode::IntEq => bhimpl_int_eq(a, b),
            OpCode::IntNe => bhimpl_int_ne(a, b),
            OpCode::UintLt => bhimpl_uint_lt(a, b),
            OpCode::UintLe => bhimpl_uint_le(a, b),
            OpCode::UintGt => bhimpl_uint_gt(a, b),
            OpCode::UintGe => bhimpl_uint_ge(a, b),
            OpCode::IntSignext => bhimpl_int_signext(a, b),
            OpCode::UintMulHigh => bhimpl_uint_mul_high(a, b),
            OpCode::PtrEq => bhimpl_ptr_eq(a, b),
            OpCode::PtrNe => bhimpl_ptr_ne(a, b),
            OpCode::FloatAdd
            | OpCode::FloatSub
            | OpCode::FloatMul
            | OpCode::FloatTrueDiv
            | OpCode::FloatFloorDiv
            | OpCode::FloatMod => crate::pyjitpl::dispatch::eval_binop_f(opcode, a, b),
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => crate::pyjitpl::dispatch::eval_float_cmp(opcode, a, b),
            other => panic!("exec_binop: unsupported opcode {other:?}"),
        }
    }

    /// Dispatch a single unary opcode to its orthodox runtime helper and
    /// return the i64 result. Float-typed operands/results are carried as
    /// `f64::to_bits` (the jitcode register convention).
    fn exec_unop(opcode: OpCode, a: i64) -> i64 {
        match opcode {
            OpCode::IntNeg => bhimpl_int_neg(a),
            OpCode::IntInvert => bhimpl_int_invert(a),
            OpCode::IntIsZero => bhimpl_int_is_zero(a),
            OpCode::IntIsTrue => bhimpl_int_is_true(a),
            OpCode::IntForceGeZero => bhimpl_int_force_ge_zero(a),
            OpCode::SameAsI | OpCode::SameAsR => bhimpl_int_same_as(a),
            OpCode::CastIntToFloat => bhimpl_cast_int_to_float(a).to_bits() as i64,
            OpCode::CastFloatToInt => bhimpl_cast_float_to_int(f64::from_bits(a as u64)),
            OpCode::ConvertFloatBytesToLonglong => {
                bhimpl_convert_float_bytes_to_longlong(f64::from_bits(a as u64))
            }
            OpCode::ConvertLonglongBytesToFloat => {
                bhimpl_convert_longlong_bytes_to_float(a).to_bits() as i64
            }
            OpCode::FloatNeg | OpCode::FloatAbs => {
                crate::pyjitpl::dispatch::eval_unary_f(opcode, a)
            }
            other => panic!("exec_unop: unsupported opcode {other:?}"),
        }
    }

    // ── Integer arithmetic & comparison (consolidated) ──

    #[test]
    fn test_executor_int_arithmetic() {
        // ADD
        assert_eq!(exec_binop(OpCode::IntAdd, 3, 4), 7);
        assert_eq!(exec_binop(OpCode::IntAdd, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, 0, 0), 0);
        // SUB
        assert_eq!(exec_binop(OpCode::IntSub, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntSub, 0, 5), -5);
        // MUL
        assert_eq!(exec_binop(OpCode::IntMul, 6, 7), 42);
        assert_eq!(exec_binop(OpCode::IntMul, -3, 4), -12);
        assert_eq!(exec_binop(OpCode::IntMul, 0, 999), 0);
        // FLOORDIV
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 17, 5), 3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -17, 5), -3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 100, 1), 100);
        // MOD
        assert_eq!(exec_binop(OpCode::IntMod, 17, 5), 2);
        assert_eq!(exec_binop(OpCode::IntMod, 10, 3), 1);
        assert_eq!(exec_binop(OpCode::IntMod, 6, 3), 0);
    }

    #[test]
    fn test_executor_int_comparisons() {
        // LT
        assert_eq!(exec_binop(OpCode::IntLt, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLt, 4, 4), 0);
        assert_eq!(exec_binop(OpCode::IntLt, 5, 4), 0);
        // GE
        assert_eq!(exec_binop(OpCode::IntGe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 3, 4), 0);
        // EQ / NE
        assert_eq!(exec_binop(OpCode::IntEq, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::IntEq, 5, 6), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 6), 1);
        // LE / GT
        assert_eq!(exec_binop(OpCode::IntLe, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 5, 4), 0);
        assert_eq!(exec_binop(OpCode::IntGt, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGt, 4, 4), 0);
    }

    #[test]
    fn test_executor_int_unary() {
        assert_eq!(exec_unop(OpCode::IntNeg, 42), -42);
        assert_eq!(exec_unop(OpCode::IntNeg, -1), 1);
        assert_eq!(exec_unop(OpCode::IntInvert, 0), -1);
        assert_eq!(exec_unop(OpCode::IntInvert, -1), 0);
    }

    #[test]
    fn test_executor_float_arithmetic() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_binop(OpCode::FloatAdd, fb(1.5), fb(2.5))), 4.0);
        assert_eq!(fr(exec_binop(OpCode::FloatMul, fb(3.0), fb(4.0))), 12.0);
        assert_eq!(fr(exec_binop(OpCode::FloatSub, fb(10.0), fb(3.5))), 6.5);
        assert_eq!(fr(exec_binop(OpCode::FloatTrueDiv, fb(10.0), fb(4.0))), 2.5);
    }

    #[test]
    fn test_executor_float_unary() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::FloatNeg, fb(3.25))), -3.25);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(-2.5))), 2.5);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(2.5))), 2.5);
    }

    #[test]
    fn test_executor_int_float_casts() {
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::CastIntToFloat, 42)), 42.0);
        assert_eq!(
            fr(exec_unop(OpCode::CastIntToFloat, i64::MAX)),
            i64::MAX as f64
        );
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(3.7) as i64),
            3
        );
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(-2.9) as i64),
            -2
        );
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(2.9)), 2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(-2.9)), -2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(0.999)), 0);
    }

    #[test]
    fn test_executor_cast_float_to_int_rejects_values_outside_signed() {
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            9_223_372_036_854_775_808.0,
        ] {
            assert!(
                std::panic::catch_unwind(|| { exec_unop(OpCode::CastFloatToInt, f64_bits(value)) })
                    .is_err(),
                "cast_float_to_int silently accepted {value:?}",
            );
        }
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64_bits(i64::MIN as f64)),
            i64::MIN,
        );
    }

    // ── Overflow arithmetic ──

    #[test]
    fn test_executor_overflow_arithmetic_uses_blackhole_wrapping_semantics() {
        assert_eq!(exec_binop(OpCode::IntAddOvf, 10, 20), 30);
        assert_eq!(exec_binop(OpCode::IntSubOvf, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntMulOvf, 6, 7), 42);

        // In blackhole mode, overflow ops use wrapping arithmetic.
        // GuardNoOverflow/GuardOverflow are separate ops.
        let result = exec_binop(OpCode::IntAddOvf, i64::MAX, 1);
        assert_eq!(result, i64::MIN);
        assert_eq!(exec_binop(OpCode::IntSubOvf, i64::MIN, 1), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntMulOvf, i64::MAX, 2), -2);
    }

    // ── Bitwise ops ──

    #[test]
    fn test_executor_bitwise_and_shift_ops() {
        assert_eq!(exec_binop(OpCode::IntAnd, 0xFF, 0x0F), 0x0F);
        assert_eq!(exec_binop(OpCode::IntAnd, 0xAB, 0x00), 0x00);
        assert_eq!(exec_binop(OpCode::IntAnd, -1, -1), -1);
        assert_eq!(exec_binop(OpCode::IntAnd, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntOr, 0xF0, 0x0F), 0xFF);
        assert_eq!(exec_binop(OpCode::IntOr, 0, 0), 0);
        assert_eq!(exec_binop(OpCode::IntOr, -1, 0), -1);
        assert_eq!(exec_binop(OpCode::IntOr, i64::MAX, i64::MIN), -1);
        assert_eq!(exec_binop(OpCode::IntXor, 0xFF, 0x0F), 0xF0);
        assert_eq!(exec_binop(OpCode::IntXor, 42, 42), 0);
        assert_eq!(exec_binop(OpCode::IntXor, i64::MAX, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntXor, i64::MIN, i64::MIN), 0);
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MAX), i64::MIN);
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MIN), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 4), 16);
        assert_eq!(exec_binop(OpCode::IntLshift, 0xFF, 8), 0xFF00);
        assert_eq!(exec_binop(OpCode::IntRshift, 16, 4), 1);
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 1), -1); // arithmetic shift
        // Logical (unsigned) right shift.
        let result = exec_binop(OpCode::UintRshift, -1, 1);
        assert_eq!(result, i64::MAX);
    }

    // ── Boolean predicates ──

    #[test]
    fn test_executor_int_predicates_and_force_ge_zero() {
        assert_eq!(exec_unop(OpCode::IntIsZero, 0), 1);
        assert_eq!(exec_unop(OpCode::IntIsZero, 1), 0);
        assert_eq!(exec_unop(OpCode::IntIsZero, -1), 0);
        assert_eq!(exec_unop(OpCode::IntIsZero, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 0), i64::MAX); // sanity
        assert_eq!(exec_unop(OpCode::IntIsTrue, 0), 0);
        assert_eq!(exec_unop(OpCode::IntIsTrue, 1), 1);
        assert_eq!(exec_unop(OpCode::IntIsTrue, -42), 1);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 5), 5);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 0), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, -10), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MIN), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MAX), i64::MAX);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 1), 1);
    }

    // ── Float comparisons ──

    #[test]
    fn test_executor_float_comparisons() {
        let f2 = f64::to_bits(2.0) as i64;
        let f3 = f64::to_bits(3.0) as i64;
        assert_eq!(exec_binop(OpCode::FloatLt, f2, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatLt, f3, f2), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGt, f3, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatEq, f3, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatNe, f2, f3), 1);
    }

    // ── FloatFloorDiv / FloatMod ──

    #[test]
    fn test_executor_float_floordiv_and_mod() {
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(2.0) as i64;
        let result = exec_binop(OpCode::FloatFloorDiv, a, b);
        assert_eq!(f64::from_bits(result as u64), 3.0);
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(3.0) as i64;
        let result = exec_binop(OpCode::FloatMod, a, b);
        assert_eq!(f64::from_bits(result as u64), 1.0);
        // -7.0 // 2.0 = -4.0
        let result = exec_binop(OpCode::FloatFloorDiv, f64_bits(-7.0), f64_bits(2.0));
        assert_eq!(bits_f64(result), -4.0);
        let result = exec_binop(OpCode::FloatMod, f64_bits(-7.0), f64_bits(3.0));
        let r = bits_f64(result);
        assert!(r.is_finite());
    }

    // ── Pointer comparisons ──

    #[test]
    fn test_executor_ptr_eq_ne() {
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 100), 1);
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 200), 0);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 200), 1);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 100), 0);
    }

    // ── SameAs ──

    #[test]
    fn test_executor_same_as() {
        assert_eq!(exec_unop(OpCode::SameAsI, 42), 42);
        assert_eq!(exec_unop(OpCode::SameAsR, 0xDEAD), 0xDEAD);
    }

    // ── UintMulHigh ──

    #[test]
    fn test_executor_uint_mul_high() {
        // Upper 64 bits of unsigned 128-bit multiply.
        // 0x8000_0000_0000_0000 * 2 = 0x1_0000_0000_0000_0000 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, i64::MIN, 2);
        assert_eq!(result, 1);

        // Small values: high 64 bits should be 0.
        let result = exec_binop(OpCode::UintMulHigh, 100, 200);
        assert_eq!(result, 0);

        // -1i64 = 0xFFFF_FFFF_FFFF_FFFF as u64, * 2 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, -1, 2);
        assert_eq!(result, 1);
        // u64::MAX * u64::MAX = (2^64-1)^2 = 2^128 - 2^65 + 1
        // High 64 bits = 0xFFFFFFFFFFFFFFFE
        let result = exec_binop(OpCode::UintMulHigh, -1, -1);
        assert_eq!(result, -2); // 0xFFFFFFFFFFFFFFFE as i64 = -2
        // Any value * 1 has high 64 bits = 0
        assert_eq!(exec_binop(OpCode::UintMulHigh, i64::MAX, 1), 0);
        assert_eq!(exec_binop(OpCode::UintMulHigh, -1, 1), 0);
    }

    // Executor edge-case parity tests
    // Ported from rpython/jit/metainterp/test/test_executor.py

    // ── Integer overflow boundaries ──

    #[test]
    fn test_executor_integer_overflow_boundaries() {
        // Wrapping: i64::MAX + 1 = i64::MIN
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 1), i64::MIN);
        // i64::MIN - 1 wraps to i64::MAX
        assert_eq!(exec_binop(OpCode::IntSub, i64::MIN, 1), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntMul, i64::MAX, 2), -2);
        assert_eq!(exec_binop(OpCode::IntMul, i64::MIN, -1), i64::MIN); // wrapping
        // -i64::MIN wraps to i64::MIN (two's complement)
        assert_eq!(exec_unop(OpCode::IntNeg, i64::MIN), i64::MIN);
        // i64::MIN / -1 would overflow; wrapping_div wraps to i64::MIN
        assert_eq!(exec_binop(OpCode::IntFloorDiv, i64::MIN, -1), i64::MIN);
        // i64::MIN % -1 would overflow; wrapping_rem wraps to 0
        assert_eq!(exec_binop(OpCode::IntMod, i64::MIN, -1), 0);
    }

    // ── Float special values ──

    fn f64_bits(v: f64) -> i64 {
        f64::to_bits(v) as i64
    }

    fn bits_f64(v: i64) -> f64 {
        f64::from_bits(v as u64)
    }

    #[test]
    fn test_executor_float_special_values() {
        let result = exec_binop(OpCode::FloatAdd, f64_bits(f64::NAN), f64_bits(1.0));
        assert!(bits_f64(result).is_nan());
        // Inf * 0 = NaN
        let result = exec_binop(OpCode::FloatMul, f64_bits(f64::INFINITY), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
        // 1.0 / 0.0 = Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::INFINITY);
        // -1.0 / 0.0 = -Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(-1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::NEG_INFINITY);
        // 0.0 / 0.0 = NaN
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(0.0), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
        // Inf - Inf = NaN
        let result = exec_binop(
            OpCode::FloatSub,
            f64_bits(f64::INFINITY),
            f64_bits(f64::INFINITY),
        );
        assert!(bits_f64(result).is_nan());
        // -NaN is still NaN
        let result = exec_unop(OpCode::FloatNeg, f64_bits(f64::NAN));
        assert!(bits_f64(result).is_nan());
        let result = exec_unop(OpCode::FloatAbs, f64_bits(f64::NEG_INFINITY));
        assert_eq!(bits_f64(result), f64::INFINITY);
        // All comparisons with NaN return false (0).
        let nan = f64_bits(f64::NAN);
        let one = f64_bits(1.0);
        assert_eq!(exec_binop(OpCode::FloatLt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatEq, nan, nan), 0);
        assert_eq!(exec_binop(OpCode::FloatNe, nan, nan), 1);
    }

    // ── Unsigned comparisons with negative values ──

    #[test]
    fn test_executor_uint_comparisons() {
        // -1i64 as u64 is u64::MAX, which is larger than 1
        assert_eq!(exec_binop(OpCode::UintLt, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, -1, 1), 1);
        // -1 as u64 = 0xFFFFFFFFFFFFFFFF (max unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::UintGe, -1, 0), 1);
        assert_eq!(exec_binop(OpCode::UintLe, 0, -1), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 0, -1), 0);

        // Equal values
        assert_eq!(exec_binop(OpCode::UintLt, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::UintLe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 5, 5), 0);

        // i64::MIN as u64 = 0x8000000000000000 (large positive unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, i64::MIN, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, i64::MIN, 1), 1);
    }

    // ── Shift edge cases ──

    #[test]
    fn test_executor_shift_edge_cases() {
        // 1 << 63 = i64::MIN (sign bit)
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 63), i64::MIN);
        // Arithmetic shift: -1 >> 63 = -1 (sign bit fills)
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 63), -1);
        // Logical shift: -1u >> 63 = 1 (top bit only)
        assert_eq!(exec_binop(OpCode::UintRshift, -1, 63), 1);
        assert_eq!(exec_binop(OpCode::IntLshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::IntRshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::UintRshift, 42, 0), 42);
        // -5 << 2 = -20
        assert_eq!(exec_binop(OpCode::IntLshift, -5, 2), -20);
    }

    #[test]
    fn test_executor_translated_shift_count_is_word_masked() {
        // `check_shift_count` is untranslated-test scaffolding in PyPy.  The
        // translated blackhole reaches the machine-word shift, whose count is
        // masked to the low six bits on pyre's 64-bit targets.
        assert_eq!(exec_binop(OpCode::IntLshift, 7, 64), 7);
        assert_eq!(exec_binop(OpCode::IntRshift, 7, 64), 7);
        assert_eq!(exec_binop(OpCode::UintRshift, 7, 64), 7);
        assert_eq!(exec_binop(OpCode::IntLshift, 7, -1), i64::MIN);
        assert_eq!(exec_binop(OpCode::IntRshift, 7, -1), 0);
        assert_eq!(exec_binop(OpCode::UintRshift, 7, -1), 0);
    }

    // ── IntSignext ──

    #[test]
    fn test_executor_int_signext() {
        // 0xFF sign-extended from 1 byte = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFF, 1), -1);
        // 0x7F sign-extended from 1 byte = 127
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7F, 1), 127);
        // 0x80 sign-extended from 1 byte = -128
        assert_eq!(exec_binop(OpCode::IntSignext, 0x80, 1), -128);
        // 0xFFFF sign-extended from 2 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFF, 2), -1);
        // 0x7FFF = 32767
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFF, 2), 32767);
        // 0x8000 = -32768
        assert_eq!(exec_binop(OpCode::IntSignext, 0x8000, 2), -32768);
        // 0xFFFFFFFF sign-extended from 4 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFFFFFF_i64, 4), -1);
        // 0x7FFFFFFF = 2147483647
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFFFFFF, 4), 2147483647);
        // 0x80000000 = -2147483648
        assert_eq!(
            exec_binop(OpCode::IntSignext, 0x80000000_i64, 4),
            -2147483648
        );
    }

    #[test]
    fn handler_int_signext_narrows_signed_widths() {
        // `blackhole.py` `bhimpl_int_signext` through `bhhandler_ii_i!`:
        // code bytes are register indices `[lhs, nbytes, dst]`.
        let cases = [
            (0x80_i64, 1_i64, -128_i64),
            (0xffff_ffff, 4, -1),
            (0x7fff_ffff, 4, 0x7fff_ffff),
        ];
        for (value, nbytes, expected) in cases {
            let mut bh = BlackholeInterpreter::default();
            bh.registers_i = vec![value, nbytes, 0];
            let next = handler_int_signext(&mut bh, &[0, 1, 2], 0).unwrap();
            assert_eq!(next, 3);
            assert_eq!(
                bh.registers_i[2], expected,
                "int_signext({value:#x}, {nbytes})"
            );
        }
    }

    #[test]
    fn test_executor_int_signext_rejects_nonpositive_width() {
        for numbytes in [0, -1] {
            assert!(
                std::panic::catch_unwind(|| { exec_binop(OpCode::IntSignext, 0x80, numbytes) })
                    .is_err(),
                "IntSignext silently accepted numbytes={numbytes}",
            );
        }
    }

    // ── ConvertFloatBytesToLonglong / ConvertLonglongBytesToFloat roundtrip ──

    #[test]
    fn test_executor_convert_float_bytes_roundtrip() {
        // ConvertFloatBytesToLonglong is identity (f64 bits as i64)
        // ConvertLonglongBytesToFloat is identity (i64 bits as f64)
        let val = f64::to_bits(3.25) as i64;
        let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, val);
        assert_eq!(ll, val);
        let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
        assert_eq!(back, val);
    }

    #[test]
    fn test_executor_convert_float_bytes_special_values() {
        for v in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let bits = f64::to_bits(v) as i64;
            let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, bits);
            assert_eq!(ll, bits);
            let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
            assert_eq!(back, bits);
        }
    }

    // Tests for jitcode-based BlackholeInterpreter.
    // Upstream parity anchor: `rpython/jit/metainterp/test/test_blackhole.py`
    // plus the dispatch-loop setup in `rpython/jit/metainterp/blackhole.py`.

    mod bh_interp_tests {
        use super::super::*;
        use crate::jitcode::JitCodeBuilder;

        /// C.5.1 — strict-dispatch builder for bare-new() unit fixtures.
        ///
        /// Mirrors `build_inline_call_only_bh_builder` for the BC_*
        /// universe these fixtures emit via `JitCodeBuilder` helpers
        /// (`load_const_i_value`, `record_binop_i`, `record_unary_i`,
        /// `move_i`, `jump`, `goto_if_not_int_is_true`,
        /// `load_const_r_value`, `ptr_nonzero`, `goto_if_not_ptr_nonzero`).
        /// Mirrors the RPython contract that every emitted bytecode key
        /// is wired before `dispatch_loop` runs (`blackhole.py
        /// setup_insns` resolving every key via `_get_method`).
        ///
        /// All emit helpers (`record_binop_i`, `goto_if_not_*`,
        /// `int/ref/float_guard_value`, etc.) now push 1-byte register
        /// operands matching the canonical `bhhandler_*` decoders, so
        /// every key here is the upstream-canonical opname/argcodes
        /// pair (no `_u16_ext` suffix).
        fn build_test_bh_builder() -> BlackholeInterpBuilder {
            use majit_jitcode::insns;
            let mut builder = BlackholeInterpBuilder::new();
            let mut entries: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            entries.insert("int_copy/i>i".to_string(), insns::BC_MOVE_I);
            // `load_const_i_value` picks the `USE_C_FORM` short encoding for
            // byte-sized constants, so both `int_copy` variants must be wired.
            entries.insert("int_copy/c>i".to_string(), insns::BC_MOVE_I_C);
            entries.insert("ref_copy/r>r".to_string(), insns::BC_MOVE_R);
            entries.insert("int_add/ii>i".to_string(), insns::BC_INT_ADD);
            entries.insert("int_mul/ii>i".to_string(), insns::BC_INT_MUL);
            entries.insert("int_neg/i>i".to_string(), insns::BC_INT_NEG);
            entries.insert("ptr_nonzero/r>i".to_string(), insns::BC_PTR_NONZERO);
            entries.insert("goto/L".to_string(), insns::BC_JUMP);
            entries.insert(
                "goto_if_not_int_is_true/iL".to_string(),
                insns::BC_GOTO_IF_NOT_INT_IS_TRUE,
            );
            // Canonical `goto_if_not/iL` alias — `bhimpl_goto_if_not_int_is_true =
            // bhimpl_goto_if_not` (`blackhole.py`) routes both bytes to
            // the same handler body.
            entries.insert("goto_if_not/iL".to_string(), insns::BC_GOTO_IF_NOT);
            entries.insert(
                "goto_if_not_ptr_nonzero/rL".to_string(),
                insns::BC_GOTO_IF_NOT_PTR_NONZERO,
            );
            builder.setup_insns(&entries);
            wire_bhimpl_handlers(&mut builder);
            builder
        }

        /// `test_blackhole.py test_convert_and_run_from_pyjitpl`.
        ///
        /// The fixture puts an illegal instruction at pc 0 and gives the
        /// MIFrame `pc = 1`, so the chain must start at exactly `frame.pc`
        /// and can never reach anything before it.  pyre's assembler emits
        /// only real opcodes, so the illegal instruction is an `int_mul`
        /// left out of the dispatch table: its slot holds
        /// `unwired_handler_placeholder`, which panics the same way
        /// `\xFF` does upstream.  It also writes `r0`, so a chain that
        /// started at 0 would return 1602 rather than panicking silently.
        #[test]
        fn test_convert_and_run_from_pyjitpl_starts_at_frame_pc() {
            use crate::pyjitpl::{MIFrame, MIFrameStack};
            use majit_jitcode::insns;

            let mut b = JitCodeBuilder::default();
            // pc 0 — never executed.
            b.record_binop_i(0, OpCode::IntMul, 0, 0);
            let second = b.current_pos();
            b.record_binop_i(2, OpCode::IntAdd, 0, 1);
            b.int_return(2);
            let jitcode = std::sync::Arc::new(b.finish());

            // test_blackhole.py `pc = 1`, `registers_i = [40, 2, None]`.
            let mut frame = MIFrame::new(jitcode, second);
            frame.int_regs[0] = Some(OpRef::const_int(40));
            frame.int_regs[1] = Some(OpRef::const_int(2));
            let framestack = MIFrameStack::new(frame);

            let mut builder = BlackholeInterpBuilder::new();
            let mut entries: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            entries.insert("int_add/ii>i".to_string(), insns::BC_INT_ADD);
            entries.insert("int_return/i".to_string(), insns::BC_INT_RETURN);
            builder.setup_insns(&entries);
            wire_bhimpl_handlers(&mut builder);

            // test_blackhole.py `DoneWithThisFrameInt`, `result == 42`.
            let outcome =
                convert_and_run_from_pyjitpl(&mut builder, &framestack, 0, false, None, None);
            assert_eq!(
                outcome,
                crate::jitexc::JitException::DoneWithThisFrameInt(42)
            );
        }

        /// `history.py` `ConstPtr.getref_base` is not defined on an int box.
        /// An int box in `registers_r` must stop in `_copy_data_from_miframe`
        /// before the blackhole loads a field through that register.
        #[test]
        #[should_panic(expected = "jitcode bank-mismatch pc 7 bank Ref register 0")]
        fn copy_data_from_miframe_panics_when_ref_bank_holds_an_int_box() {
            use crate::pyjitpl::MIFrame;
            use majit_ir::OpRef;

            let mut b = JitCodeBuilder::default();
            b.ref_return(0);
            let mut jitcode = b.finish();
            jitcode.name = "bank-mismatch".to_string();
            let jitcode = std::sync::Arc::new(jitcode);
            let mut frame = MIFrame::new(jitcode, 7);
            frame.ref_regs[0] = Some(OpRef::const_int(0x60));

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.copy_data_from_miframe(&frame);
        }

        /// `BlackholeInterpreter._copy_data_from_miframe` reads each
        /// `registers_i[i].getint()`. For a ConstInt that means the inline
        /// `value`, never pyre's execution mirror.
        #[test]
        fn copy_data_from_miframe_reads_constint_owner() {
            use crate::pyjitpl::MIFrame;
            use majit_ir::OpRef;

            let mut b = JitCodeBuilder::default();
            b.int_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            let mut frame = MIFrame::new(jitcode, 0);
            frame.int_regs[0] = Some(OpRef::const_int(42));

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.copy_data_from_miframe(&frame);

            assert_eq!(bh.registers_i[0], 42);
        }

        /// `IntOp.getint` returns `_resint`, which is 0 until `setint`.
        /// A present live box with no stamped mirror must still write that
        /// default: skipping the write leaves a pooled leftover index.
        #[test]
        fn copy_data_from_miframe_writes_resint_default_for_unstamped_live_int() {
            use crate::pyjitpl::MIFrame;
            use majit_ir::OpRef;

            let mut b = JitCodeBuilder::default();
            b.int_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            let mut frame = MIFrame::new(jitcode, 0);
            frame.int_regs[0] = Some(OpRef::int_op(5));

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            // `setposition` grows the bank and never re-zeros a larger one.
            bh.registers_i = vec![0x380000; 16];
            bh.copy_data_from_miframe(&frame);

            assert_eq!(bh.registers_i[0], 0);
        }

        /// `BlackholeInterpreter._copy_data_from_miframe` reads each
        /// `registers_r[i].getref_base()`. For a ConstPtr that means the
        /// GC-forwarded value stored on the box, never pyre's execution mirror.
        #[test]
        fn copy_data_from_miframe_reads_forwarded_constptr_owner() {
            use crate::pyjitpl::MIFrame;
            use majit_ir::{GcRef, OpRef};

            let mut b = JitCodeBuilder::default();
            b.ref_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            let mut frame = MIFrame::new(jitcode, 0);
            frame.ref_regs[0] = Some(OpRef::const_ptr(GcRef(0xBEEF)));

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.copy_data_from_miframe(&frame);

            assert_eq!(bh.registers_r[0], 0xBEEF);
        }

        #[test]
        #[should_panic(expected = "metainterp framestack is empty")]
        fn convert_and_run_from_pyjitpl_rejects_empty_framestack() {
            let mut builder = build_test_bh_builder();
            let framestack = crate::pyjitpl::MIFrameStack::empty();
            let _ = convert_and_run_from_pyjitpl(&mut builder, &framestack, 0, false, None, None);
        }

        #[test]
        fn rooted_state_field_chain_skips_run_rooting() {
            let mut builder = build_test_bh_builder();
            let bh = builder.acquire_interp();
            assert!(bh.rooted);
            assert!(bh.virtualizable_info.is_null());
            assert!(!bh.chain_needs_run_rooting());
        }

        #[test]
        fn acquire_interp_reuses_the_builder_dispatch_table() {
            let mut builder = build_test_bh_builder();
            let first = builder.acquire_interp();
            let table = std::sync::Arc::clone(&first.dispatch_table);
            builder.release_interp(first);
            let reused = builder.acquire_interp();
            assert!(
                std::sync::Arc::ptr_eq(&reused.dispatch_table, &table),
                "a pooled interp already holds this builder's dispatch table"
            );
        }

        #[test]
        fn unresolved_call_and_vtable_bail_set_abort_permanent_bail() {
            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            let err = super::super::reject_unresolved_call(&mut bh, 0);
            assert!(matches!(err, DispatchError::LeaveFrame));
            assert!(bh.aborted && bh.abort_permanent_bail);

            bh.aborted = false;
            bh.abort_permanent_bail = false;
            let err = super::super::handler_vtable_method_ptr_bail(&mut bh, &[], 0);
            assert!(matches!(err, Err(DispatchError::LeaveFrame)));
            assert!(bh.aborted && bh.abort_permanent_bail);
        }

        #[test]
        fn setposition_ref_does_not_clone_an_already_seated_jitcode() {
            let mut b = JitCodeBuilder::default();
            b.int_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition_ref(&jitcode, 0);
            let before = std::sync::Arc::strong_count(&jitcode);
            bh.setposition_ref(&jitcode, 0);
            assert_eq!(
                std::sync::Arc::strong_count(&jitcode),
                before,
                "reseating the same helper must not clone the Arc"
            );
        }

        /// `reset_for_inline_reuse` must not wipe the constant area.
        /// `blackhole.py BlackholeInterpreter.cleanup_registers` never
        /// clears constants, so a same-helper reseat keeps the previous
        /// `copy_constants` and does not walk the table again.
        #[test]
        fn setposition_ref_keeps_constants_after_inline_reuse() {
            let mut b = JitCodeBuilder::default();
            b.ensure_i_regs(1);
            b.load_const_i_value(0, 0x1234_5678);
            b.int_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            assert!(
                !jitcode.constants_i.is_empty(),
                "fixture: the wide load must occupy a constants_i slot"
            );
            let expected = jitcode.constants_i[0];
            let const_index = jitcode.num_regs_i();
            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition_ref(&jitcode, 0);
            assert_eq!(bh.registers_i[const_index], expected);
            bh.reset_for_inline_reuse();
            assert_eq!(
                bh.registers_i[const_index], expected,
                "reuse must not wipe the constant area"
            );
            bh.setposition_ref(&jitcode, 0);
            assert_eq!(
                bh.registers_i[const_index], expected,
                "same-helper reseat keeps the previous copy_constants"
            );
        }

        /// A different jitcode still gets `copy_constants` of its own table.
        #[test]
        fn setposition_ref_copies_constants_when_the_jitcode_changes() {
            let mut first = JitCodeBuilder::default();
            first.ensure_i_regs(1);
            first.load_const_i_value(0, 0x1111);
            first.int_return(0);
            let first = std::sync::Arc::new(first.finish());
            let mut second = JitCodeBuilder::default();
            second.ensure_i_regs(1);
            second.load_const_i_value(0, 0x2222);
            second.int_return(0);
            let second = std::sync::Arc::new(second.finish());
            let first_slot = first.num_regs_i();
            let second_slot = second.num_regs_i();
            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition_ref(&first, 0);
            assert_eq!(bh.registers_i[first_slot], 0x1111);
            bh.setposition_ref(&second, 0);
            assert_eq!(bh.registers_i[second_slot], 0x2222);
        }

        /// A drained interpreter whose `jitcode` is unchanged must still
        /// reseat its banks. `blackhole.py BlackholeInterpreter.setposition`
        /// grows a bank when `len(registers_*) < num_regs_and_consts_*` and
        /// copies constants on every call. Two production sites empty the
        /// banks with `std::mem::take` while leaving `jitcode` set
        /// (`BlackholeTerminalImage::take_from`,
        /// `SingleFrameBlackholeResult`).
        #[test]
        fn setposition_reseats_a_drained_register_file_for_the_same_jitcode() {
            let mut b = JitCodeBuilder::default();
            b.ensure_i_regs(1);
            b.int_return(0);
            let jitcode = std::sync::Arc::new(b.finish());
            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition_ref(&jitcode, 0);
            bh.setarg_i(0, 1);
            assert_eq!(bh.registers_i[0], 1);

            let _ = std::mem::take(&mut bh.registers_i);
            let _ = std::mem::take(&mut bh.registers_r);
            let _ = std::mem::take(&mut bh.registers_f);

            bh.setposition_ref(&jitcode, 0);
            assert!(
                bh.registers_i.len() >= jitcode.num_regs_and_consts_i(),
                "same-jitcode reseat after drain must grow the int bank"
            );
            bh.setarg_i(0, 7);
            assert_eq!(bh.registers_i[0], 7);
        }

        /// `_setup_return_value_i` reads `code[position-1]`, the single
        /// result register `inline_call_typed` now closes on.
        #[test]
        fn setup_return_value_i_reads_inline_calls_int_return_slot() {
            let mut b = JitCodeBuilder::default();
            b.inline_call_r_i(0, &[], Some(7));
            let jitcode = b.finish();
            // The call is the whole body, so the position the caller resumes at
            // is the end of the code.
            let resume_pos = jitcode.code.len();
            assert_eq!(
                jitcode.code[resume_pos - 1],
                7,
                "inline call must end on the int result register",
            );

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), resume_pos);
            bh.setup_return_value_i(42);
            assert_eq!(bh.registers_i[7], 42);
        }

        #[test]
        fn test_bh_interp_load_const_and_binop() {
            // Build jitcode: r0 = const(10), r1 = const(20), r2 = r0 + r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 10);
            b.load_const_i_value(1, 20);
            b.record_binop_i(2, OpCode::IntAdd, 0, 1);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 30);
        }

        #[test]
        fn test_bh_interp_goto_if_not_int_is_true_taken() {
            // Build jitcode: r0 = 0; if r0==0 goto end; r1 = 42; end: r2 = 99
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 0);
            let lbl = b.new_label();
            b.goto_if_not_int_is_true(0, lbl);
            b.load_const_i_value(1, 42); // should be skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 0); // skipped, still 0
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_goto_if_not_int_is_true_not_taken() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 1); // nonzero
            let lbl = b.new_label();
            b.goto_if_not_int_is_true(0, lbl);
            b.load_const_i_value(1, 42); // NOT skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_jump() {
            let mut b = JitCodeBuilder::default();
            let lbl = b.new_label();
            b.jump(lbl);
            b.load_const_i_value(0, 42); // skipped
            b.mark_label(lbl);
            b.load_const_i_value(1, 99);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[0], 0); // skipped
            assert_eq!(bh.registers_i[1], 99);
        }

        #[test]
        fn test_bh_interp_move() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.move_i(1, 0);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }

        #[test]
        fn dispatch_loop_int_eq_then_int_return() {
            // Translated `dispatch_loop` increments past the opcode first.
            // `int_eq` + `int_return` is the shift-epilogue exit.
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 3);
            b.load_const_i_value(1, 3);
            b.record_binop_i(2, OpCode::IntEq, 0, 1);
            b.int_return(2);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            match bh.run() {
                BhRunOutcome::LeaveFrame => {}
                other => panic!("expected LeaveFrame, got {other:?}"),
            }
            assert_eq!(bh.tmpreg_i, 1);
            assert_eq!(bh.return_type, BhReturnType::Int);
        }

        /// `p = malloc(T, flavor='raw'); setfield; getfield; free` returns
        /// the stored word. The helpers are the real `ll_raw_*` addresses.
        #[test]
        fn blackhole_raw_malloc_fixedsize_roundtrip_returns_x() {
            use majit_translate::assembler::AssemblerExt;
            use majit_translate::call::{CallControl, StructFieldLayout, StructLayout};
            use majit_translate::jtransform::{GraphTransformConfig, Transformer};
            use majit_translate::model::{
                FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType,
            };
            use majit_translate::{CallPath, ConcreteType, RegKind};
            use std::collections::HashMap;

            let owner = "Tuple<i64>";
            let mut cc = CallControl::new();
            let sid = majit_ir::descr::struct_id_for_name(owner).expect("Tuple<i64> shape id");
            cc.set_struct_layout(
                sid,
                StructLayout {
                    size: 8,
                    align: 8,
                    gckind: majit_translate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                    fields: vec![StructFieldLayout {
                        name: "f".into(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    }],
                    host: None,
                    ll_struct: std::cell::RefCell::new(None),
                    ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
                },
            );
            cc.register_function_fnaddr(
                CallPath::from_segments(["majit_rlib", "rffi", "ll_raw_malloc_fixedsize"]),
                majit_rlib::rffi::ll_raw_malloc_fixedsize as *const () as usize as i64,
            );
            cc.register_function_fnaddr(
                CallPath::from_segments(["majit_rlib", "rffi", "ll_raw_free"]),
                majit_rlib::rffi::ll_raw_free as *const () as usize as i64,
            );

            let mut graph = FunctionGraph::new("raw_fixedsize_bh");
            let x = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::Input {
                        name: "x".into(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                    true,
                )
                .expect("input");
            graph.push_inputarg_var(graph.startblock, x.clone());
            FunctionGraph::set_concretetype_of_inline(&x, ConcreteType::Signed);
            let field = FieldDescriptor::new("f", Some(owner.into()));
            let p = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::RawMalloc {
                        owner: owner.into(),
                        zero: false,
                    },
                    true,
                )
                .expect("raw malloc result");
            let _ = graph.push_op_var(
                graph.startblock,
                OpKind::FieldWrite {
                    base: p.clone(),
                    field: field.clone(),
                    value: LinkArg::Value(x.clone()),
                    ty: ValueType::Int,
                },
                false,
            );
            let y = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::FieldRead {
                        base: p.clone(),
                        field,
                        ty: ValueType::Int,
                        pure: false,
                    },
                    true,
                )
                .expect("field read result");
            let _ = graph.push_op_var(graph.startblock, OpKind::RawFree { ptr: p }, false);
            graph.set_return(graph.startblock, Some(y.clone()));
            FunctionGraph::set_concretetype_of_inline(&y, ConcreteType::Signed);

            let config = GraphTransformConfig::default();
            let mut rewritten = {
                let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
                transformer.transform(&graph).graph
            };
            let except_args = rewritten.block(rewritten.exceptblock).inputargs.clone();
            FunctionGraph::set_concretetype_of_inline(&except_args[0], ConcreteType::Signed);
            FunctionGraph::set_concretetype_of_inline(&except_args[1], ConcreteType::GcRef);
            let mut regallocs = HashMap::new();
            for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
                regallocs.insert(
                    kind,
                    majit_translate::regalloc::perform_register_allocation(&rewritten, kind),
                );
            }
            let mut flat = majit_translate::flatten_graph(&rewritten, &mut regallocs);
            let mut asm = majit_jitcode::codewriter::assembler::Assembler::new();
            let body = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
            let x_color = regallocs
                .get(&RegKind::Int)
                .and_then(|alloc| alloc.color_for_variable(&x))
                .expect("x has an int color");

            let core = majit_jitcode::jitcode::JitCode::new("raw_fixedsize_bh");
            core.set_body(body);
            let mut runtime = crate::jitcode::JitCode::from_canonical(core);
            runtime.exec.descrs = asm
                .snapshot_descrs()
                .into_iter()
                .map(|descr| crate::jitcode::RuntimeBhDescr::Descr(Box::new(descr)).into_resolved())
                .collect();

            let cpu = crate::pyjitpl::BackendImpl::new();
            let mut builder = BlackholeInterpBuilder::new();
            builder.set_cpu(&cpu);
            builder.setup_insns(&asm.insns);
            wire_bhimpl_handlers(&mut builder);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(runtime), 0);
            let x_value = 0x1357_9BDF;
            bh.setarg_i(x_color, x_value);
            match bh.run() {
                BhRunOutcome::LeaveFrame => {}
                other => panic!("expected LeaveFrame, got {other:?}"),
            }
            assert_eq!(bh.get_tmpreg_i(), x_value);
        }

        #[test]
        fn test_bh_interp_unary_neg() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.record_unary_i(1, OpCode::IntNeg, 0);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], -42);
        }

        #[test]
        fn test_bh_interp_ptr_nonzero() {
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0x1234);
            b.ptr_nonzero(1, 0);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 1);
        }

        #[test]
        fn test_bh_interp_goto_if_not_ptr_nonzero_taken() {
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0);
            let lbl = b.new_label();
            b.goto_if_not_ptr_nonzero(0, lbl);
            b.load_const_i_value(1, 42); // should be skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 0);
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_setarg() {
            let mut b = JitCodeBuilder::default();
            // Just record a binop to read r0 + r1
            b.record_binop_i(2, OpCode::IntMul, 0, 1);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            bh.setarg_i(0, 7);
            bh.setarg_i(1, 6);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 42);
        }

        // The range check is a debug_assert, so the panic exists only in
        // debug builds.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "setarg_r: register index")]
        fn setarg_r_refuses_constant_table_slot() {
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0);
            b.add_const_r(0x11);
            b.ref_return(0);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            assert_eq!(bh.jitcode.num_regs_r(), 1);
            bh.setarg_r(1, 0x22);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "setarg_f: register index")]
        fn setarg_f_refuses_constant_table_slot() {
            let mut b = JitCodeBuilder::default();
            b.load_const_f_value(0, 0);
            b.add_const_f(0);
            b.float_return(0);
            let jitcode = b.finish();

            let mut builder = build_test_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            assert_eq!(bh.jitcode.num_regs_f(), 1);
            bh.setarg_f(1, 0);
        }

        #[test]
        fn test_bh_interp_builder_pool() {
            let mut builder = BlackholeInterpBuilder::new();

            let mut bh1 = builder.acquire_interp();
            assert!(bh1.registers_i.is_empty());
            let allocation = std::ptr::from_ref(&*bh1);

            let mut merge_args = Box::new(MergePointArgs::default());
            merge_args.green_int.extend(0..7);
            merge_args.red_ref.extend(0..11);
            let merge_args_allocation = std::ptr::from_ref(&*merge_args);
            let green_int_capacity = merge_args.green_int.capacity();
            let red_ref_capacity = merge_args.red_ref.capacity();
            bh1.recycle_merge_point_args(merge_args);

            builder.release_interp(bh1);
            let bh2 = builder.acquire_interp();
            // `blackhole.py BlackholeInterpBuilder.acquire_interp` returns
            // the same object from its intrusive `back` list.  Keeping the
            // Box allocation is the Rust equivalent; unboxing here and
            // re-boxing in release made every nominal pool hit one free plus
            // one malloc.
            assert_eq!(std::ptr::from_ref(&*bh2), allocation);
            assert!(bh2.registers_i.is_empty());
            let scratch = bh2
                .merge_point_args_scratch
                .as_ref()
                .expect("a pooled frame keeps the merge-point scratch it was handed");
            // The payload the next `ContinueRunningNormally` fills is the one
            // the last handoff gave back: same box, same six capacities, and
            // no value carried over from the run that produced it.
            assert_eq!(std::ptr::from_ref(&**scratch), merge_args_allocation);
            assert!(scratch.green_int.is_empty());
            assert!(scratch.red_ref.is_empty());
            assert!(scratch.green_int.capacity() >= green_int_capacity);
            assert!(scratch.red_ref.capacity() >= red_ref_capacity);
        }

        /// `BlackholeInterpreter._get_method` stores the post-op position and
        /// re-raises every exception from a `bhimpl_*`.  The translated
        /// `do_malloc_*_clear` helpers use that edge for `MemoryError`; NULL
        /// must never be installed as an ordinary reference result.
        #[test]
        fn allocation_failure_raises_memory_error_at_the_post_op_position() {
            const MEMORY_ERROR: i64 = 0x5eed_0001;
            fn memory_error() -> i64 {
                MEMORY_ERROR
            }
            majit_backend::register_memory_error_provider(memory_error);

            let err = super::blackhole_allocation_error(37);
            assert!(matches!(
                err,
                super::DispatchError::RaiseException {
                    exc: MEMORY_ERROR,
                    resume_position: 37,
                }
            ));
        }

        #[test]
        fn reraise_of_a_null_exception_aborts_the_frame() {
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.exception_last_value = 0;
            let err = super::handler_reraise(&mut bh, &[], 0)
                .expect_err("a null exception must leave the frame");
            assert!(matches!(err, super::DispatchError::LeaveFrame));
            assert!(bh.aborted);
        }

        #[test]
        fn obsolete_vtable_method_opcode_bails_instead_of_panicking() {
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            let err = super::handler_vtable_method_ptr_bail(&mut bh, &[], 0)
                .expect_err("unsupported named vtable lookup must leave blackhole");
            assert!(matches!(err, super::DispatchError::LeaveFrame));
            assert!(bh.aborted);
        }

        /// A pooled interp must not carry the previous run's frame identity.
        ///
        /// `acquire_interp` refreshes only the six builder-shared fields
        /// (`blackhole.py`), and `blackhole_from_resumedata` assigns
        /// neither `virtualizable_ptr` nor `virtualizable_info` — the guard
        /// path sets them afterwards, and deliberately does not for a
        /// `novable` resume ("leave both the pointer and the vinfo handle
        /// unset (null)", `pyre-jit/src/call_jit.rs`).  That contract is only
        /// true if the pool hands back a cleared interp: `run()` dereferences
        /// `virtualizable_ptr` for every chain frame with a non-null
        /// `virtualizable_info` to push its GC roots, so a leaked pair reads
        /// the frame the previous blackhole finished with.
        ///
        /// The handle here is never dereferenced — the assertion is on the
        /// pointer word, so a dangling non-null stand-in is enough.
        #[test]
        fn released_interp_does_not_leak_its_virtualizable_handle_to_the_next_user() {
            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.virtualizable_ptr = 0x7f00_0000_1234;
            bh.virtualizable_info =
                std::ptr::NonNull::<crate::virtualizable::VirtualizableInfo>::dangling().as_ptr();
            bh.virtualizable_stack_base = 9;

            builder.release_interp(bh);
            let reused = builder.acquire_interp();

            assert_eq!(reused.virtualizable_ptr, 0);
            assert!(reused.virtualizable_info.is_null());
            assert_eq!(reused.virtualizable_stack_base, 0);
        }

        /// A pooled interp must expose no reference word as a GC root.
        ///
        /// `run()` roots `registers_r` AND `tmpreg_r` through
        /// `push_bh_regs`, and `walk_bh_regs` visits the tmpreg word as a
        /// `GcRef`.  `cleanup_registers` cleared the bank but not the tmpreg,
        /// so a released interp carried the last ref return value into the
        /// pool, where nothing roots it — and the next `run()` on that interp
        /// pushed the by-then-collectable address back as a root.
        #[test]
        fn released_interp_exposes_no_reference_word_to_the_root_walker() {
            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.tmpreg_r = 0x7f00_0000_5678;

            builder.release_interp(bh);
            let mut reused = builder.acquire_interp();
            assert_eq!(reused.tmpreg_r, 0);

            let depth = unsafe {
                majit_gc::shadow_stack::push_bh_regs(
                    &mut reused.registers_r,
                    &mut reused.tmpreg_r,
                    &mut reused.exception_last_value,
                )
            };
            let mut roots: Vec<usize> = Vec::new();
            majit_gc::shadow_stack::walk_bh_regs(|slot| roots.push(slot.0));
            majit_gc::shadow_stack::pop_bh_regs_to(depth);
            assert!(
                roots.iter().all(|&word| word == 0),
                "pooled interp handed the root walker stale references: {roots:x?}",
            );
        }

        /// `blackhole.py:get_tmpreg_r` consumes its GC-visible scratch slot.
        /// Keeping the value there until frame release retains a dead result
        /// throughout any remaining blackhole execution.
        #[test]
        fn get_tmpreg_r_clears_the_consumed_reference() {
            let mut bh = BlackholeInterpreter::default();
            bh.tmpreg_r = 0x7f00_0000_9abc;

            assert_eq!(bh.get_tmpreg_r(), 0x7f00_0000_9abc);
            assert_eq!(bh.tmpreg_r, 0);
        }

        /// The caught exception is reachable from the root walker while the
        /// frame's handler runs.
        ///
        /// `route_to_catch` stores it in `exception_last_value` and jumps to
        /// the handler (`blackhole.py`); the handler recovers it much
        /// later through `last_exc_value` / `last_exception`, and the latter
        /// dereferences it via `bh_classof`.  Every allocation the handler
        /// performs in between can trigger a minor collection, so the slot has
        /// to be a root — a moved object would leave the read on a from-space
        /// address, and an otherwise-unreferenced one would simply be
        /// reclaimed.  Upstream gets this for free: the field lives on a
        /// GC-managed interpreter object, exactly like `tmpreg_r`.
        #[test]
        fn the_caught_exception_slot_is_a_gc_root_for_the_handler_window() {
            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            const EXC: i64 = 0x7f00_0000_abc0;
            bh.exception_last_value = EXC;
            bh.tmpreg_r = 0;

            let depth = unsafe {
                majit_gc::shadow_stack::push_bh_regs(
                    &mut bh.registers_r,
                    &mut bh.tmpreg_r,
                    &mut bh.exception_last_value,
                )
            };
            let mut roots: Vec<usize> = Vec::new();
            majit_gc::shadow_stack::walk_bh_regs(|slot| roots.push(slot.0));
            majit_gc::shadow_stack::pop_bh_regs_to(depth);
            assert!(
                roots.contains(&(EXC as usize)),
                "the caught exception is invisible to the collector while the \
                 handler runs; roots were {roots:x?}",
            );
        }

        #[test]
        fn test_builder_new_leaves_control_opcodes_unset_until_explicit_setup() {
            let mut builder = BlackholeInterpBuilder::new();
            assert_eq!(builder.op_live, u8::MAX);
            assert_eq!(builder.op_catch_exception, u8::MAX);
            assert_eq!(builder.op_rvmprof_code, u8::MAX);

            builder.setup_cached_control_opcodes(88, 89, 91);
            assert_eq!(builder.op_live, 88);
            assert_eq!(builder.op_catch_exception, 89);
            assert_eq!(builder.op_rvmprof_code, 91);
        }

        /// Every canonical `inline_call_*` byte must reach its handler in the
        /// production builder.
        ///
        /// `wire_handler` resolves an opname through `_insns` and returns
        /// `false` for a key `setup_insns` never registered; both call sites
        /// discard the bool. So the ten `wire_handler("inline_call_*", …)` calls
        /// in `wire_bhimpl_handlers` were silent no-ops for as long as the
        /// curated `build_inline_call_only_bh_builder` map omitted the keys, and
        /// a build-time (LLBC-extracted) jitcode's `inline_call_*` reached
        /// `dispatch_step`'s unwired panic instead. `unwired_opnames()` cannot
        /// catch that — an absent key has no table slot to be a placeholder in.
        #[test]
        fn production_bh_builder_wires_every_canonical_inline_call_byte() {
            use majit_jitcode::insns;
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            for (opname, byte) in [
                ("inline_call_r_i/dR>i", insns::BC_INLINE_CALL_R_I),
                ("inline_call_r_r/dR>r", insns::BC_INLINE_CALL_R_R),
                ("inline_call_r_v/dR", insns::BC_INLINE_CALL_R_V),
                ("inline_call_ir_i/dIR>i", insns::BC_INLINE_CALL_IR_I),
                ("inline_call_ir_r/dIR>r", insns::BC_INLINE_CALL_IR_R),
                ("inline_call_ir_v/dIR", insns::BC_INLINE_CALL_IR_V),
                ("inline_call_irf_i/dIRF>i", insns::BC_INLINE_CALL_IRF_I),
                ("inline_call_irf_r/dIRF>r", insns::BC_INLINE_CALL_IRF_R),
                ("inline_call_irf_f/dIRF>f", insns::BC_INLINE_CALL_IRF_F),
                ("inline_call_irf_v/dIRF", insns::BC_INLINE_CALL_IRF_V),
            ] {
                let slot = builder.dispatch_table[byte as usize];
                assert_ne!(
                    slot as usize, placeholder as usize,
                    "`{opname}` (byte {byte}) is unwired in the production builder",
                );
            }
        }

        /// `_truediv` emits `cast_int_to_float/i>f` for `float(x)`.  A
        /// helper subwalk resume must dispatch it; `bhimpl_cast_int_to_float`
        /// is already wired, only the curated byte was missing.
        #[test]
        fn production_bh_builder_wires_cast_int_to_float() {
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            let byte = majit_jitcode::insns::BC_CAST_INT_TO_FLOAT;
            let slot = builder.dispatch_table[byte as usize];
            assert_ne!(
                slot as usize, placeholder as usize,
                "`cast_int_to_float/i>f` (byte {byte}) is unwired in the production builder",
            );
        }

        /// `random.gauss` contains a float-to-int conversion. A guard failure
        /// can resume forward through that byte, so RPython
        /// `blackhole.py setup_insns` must bind the already-ported
        /// `bhimpl_cast_float_to_int` handler in the production builder.
        #[test]
        fn production_bh_builder_wires_cast_float_to_int() {
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            let byte = majit_jitcode::insns::BC_CAST_FLOAT_TO_INT;
            let slot = builder.dispatch_table[byte as usize];
            assert_ne!(
                slot as usize, placeholder as usize,
                "`cast_float_to_int/f>i` (byte {byte}) is unwired in the production builder",
            );
        }

        /// `blackhole.py` has no `bhimpl_call_assembler`. Recursive portal
        /// calls emit `BC_RECURSIVE_CALL_*`; bytes 50-53 stay reserved for
        /// tracing `opimpl_call_assembler_*` and remain unwired here.
        #[test]
        fn production_bh_builder_leaves_call_assembler_unwired() {
            use majit_jitcode::insns;
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            for byte in [
                insns::BC_CALL_ASSEMBLER_INT,
                insns::BC_CALL_ASSEMBLER_REF,
                insns::BC_CALL_ASSEMBLER_FLOAT,
                insns::BC_CALL_ASSEMBLER_VOID,
            ] {
                let slot = builder.dispatch_table[byte as usize];
                assert_eq!(
                    slot as usize, placeholder as usize,
                    "byte {byte} is wired in the production builder",
                );
            }
        }

        /// Leftover `cond_call_*_ext` / `record_known_result_*_ext` stay
        /// unwired. Canonical `bhimpl_conditional_call_ir_v` /
        /// `bhimpl_conditional_call_value_ir_{i,r}` /
        /// pass-bodied `bhimpl_record_known_result_{i,r}_ir_v` bind the
        /// canonical bytes; leftover 79-83 stay reserved for tracing
        /// `opimpl_cond_call_void`.
        #[test]
        fn production_bh_builder_leaves_cond_call_ext_unwired() {
            use majit_jitcode::insns;
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            for byte in [
                insns::BC_COND_CALL_VOID,
                insns::BC_COND_CALL_VALUE_INT,
                insns::BC_COND_CALL_VALUE_REF,
                insns::BC_RECORD_KNOWN_RESULT_INT,
                insns::BC_RECORD_KNOWN_RESULT_REF,
            ] {
                let slot = builder.dispatch_table[byte as usize];
                assert_eq!(
                    slot as usize, placeholder as usize,
                    "byte {byte} is wired in the production builder",
                );
            }
            let canonical = builder.dispatch_table[insns::BC_CONDITIONAL_CALL_IR_V as usize];
            assert_ne!(
                canonical as usize,
                placeholder as usize,
                "`conditional_call_ir_v/iiIRd` (byte {}) is unwired in the production builder",
                insns::BC_CONDITIONAL_CALL_IR_V,
            );
        }

        /// `complex` arithmetic reaches the three interior-field loads in
        /// generated helper JitCodes.  They already have orthodox
        /// `bhimpl_getinteriorfield_gc_*` handlers; the production builder must
        /// register their emitted bytes so `setup_insns` can bind them.
        #[test]
        fn production_bh_builder_wires_every_interior_field_load() {
            use majit_jitcode::insns;
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as super::BhOpcodeHandler;
            for (opname, byte) in [
                (
                    "getinteriorfield_gc_i/rid>i",
                    insns::BC_GETINTERIORFIELD_GC_I,
                ),
                (
                    "getinteriorfield_gc_r/rid>r",
                    insns::BC_GETINTERIORFIELD_GC_R,
                ),
                (
                    "getinteriorfield_gc_f/rid>f",
                    insns::BC_GETINTERIORFIELD_GC_F,
                ),
            ] {
                let slot = builder.dispatch_table[byte as usize];
                assert_ne!(
                    slot as usize, placeholder as usize,
                    "`{opname}` (byte {byte}) is unwired in the production builder",
                );
            }
        }

        /// The two `fnaddr` classifiers agree with the walker's gate.
        ///
        /// A symbolic hash carries the `SYMBOLIC_FNADDR_BASE` high-16-bit tag;
        /// `0` is upstream's "no address" spelling
        /// (`jitcode.py::JitCode.__init__`) and is not callable either.
        #[test]
        fn fnaddr_classifiers_match_the_walker_symbolic_gate() {
            let real = fnaddr_classifiers_match_the_walker_symbolic_gate as *const () as i64;
            assert!(!super::is_symbolic_fnaddr(real));
            assert!(super::is_callable_fnaddr(real));

            let symbolic =
                majit_jitcode::codewriter::call::SYMBOLIC_FNADDR_BASE as i64 | 0x1234_5678_9abc_i64;
            assert!(super::is_symbolic_fnaddr(symbolic));
            assert!(!super::is_callable_fnaddr(symbolic));

            // aarch64 Linux 48-bit-VA funcptrs set bit 47 (PIE base 0xaaab…,
            // mmap 0xffff…); they are real addresses, not symbolic hashes.
            for real_high in [0x0000_ffff_7122_b9c0_i64, 0x0000_aaab_1234_5678_i64] {
                assert!(!super::is_symbolic_fnaddr(real_high));
                assert!(super::is_callable_fnaddr(real_high));
            }

            // The symbolic tag shares `BhDescr::JitCode.fnaddr` with the
            // synthetic tag, whose space is every negative `i64`
            // (`synthetic_cpu::is_synthetic_fnaddr`).  The two must not
            // overlap.
            assert!(
                symbolic > 0,
                "symbolic fnaddrs must stay out of the synthetic (negative) space"
            );
            assert!(!majit_backend::synthetic_cpu::is_synthetic_fnaddr(symbolic));

            assert!(!super::is_symbolic_fnaddr(0));
            assert!(!super::is_callable_fnaddr(0));
        }

        #[test]
        fn residual_call_handlers_reject_every_unresolved_target() {
            let handlers: &[super::BhOpcodeHandler] = &[
                super::handler_residual_call_irf_i,
                super::handler_residual_call_irf_r,
                super::handler_residual_call_irf_f,
                super::handler_residual_call_irf_v,
                super::handler_residual_call_ir_i,
                super::handler_residual_call_ir_r,
                super::handler_residual_call_ir_v,
                super::handler_residual_call_r_i,
                super::handler_residual_call_r_r,
                super::handler_residual_call_r_v,
            ];
            let symbolic =
                majit_jitcode::codewriter::call::SYMBOLIC_FNADDR_BASE as i64 | 0x1234_5678_9abc_i64;
            for target in [0, symbolic] {
                for handler in handlers {
                    let mut builder = BlackholeInterpBuilder::new();
                    let mut bh = builder.acquire_interp();
                    bh.registers_i = vec![target];
                    let result = handler(&mut bh, &[0], 0);
                    assert!(matches!(result, Err(super::DispatchError::LeaveFrame)));
                    assert!(bh.aborted);
                }
            }
        }

        #[test]
        fn test_bh_interp_inline_call() {
            // Build sub-jitcode: r0 = arg, result = r0 + r0, return r1.
            // pyjitpl.py:2247-2253 requires every callee to terminate
            // in a typed return opcode so the caller's BC_INLINE_CALL
            // can recover the callee's return register via
            // `trailing_return_info()`.
            let mut sub = JitCodeBuilder::default();
            sub.record_binop_i(1, OpCode::IntAdd, 0, 0);
            sub.int_return(1);
            let sub_jitcode = sub.finish();

            // Build main jitcode: r0 = 21, inline_call(sub, arg=r0) → r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 21);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_i(sub_idx, &[(0, 0)], &[], Some(1));
            let jitcode = b.finish();

            // route through `handler_inline_call_nested_ext`
            // (the production builder shape) so this test exercises the
            // same path as the production blackhole resume.
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            assert!(
                builder.unwired_opnames().is_empty(),
                "build_inline_call_only_bh_builder left opnames unwired: {:?}",
                builder.unwired_opnames(),
            );
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }

        /// `_run_forever` steps to `nextblackholeinterp` after the
        /// iteration whose `JitException` was resolved at a recursive
        /// portal level, not only after a plain return.
        ///
        /// `blackhole.py`:
        ///
        /// ```python
        /// while True:
        ///     try:
        ///         current_exc = blackholeinterp._resume_mainloop(current_exc)
        ///     except jitexc.JitException as e:
        ///         blackholeinterp, current_exc = _handle_jitexception(
        ///             blackholeinterp, e)
        ///     blackholeinterp.builder.release_interp(blackholeinterp)
        ///     blackholeinterp = blackholeinterp.nextblackholeinterp
        /// ```
        ///
        /// The release + step sit BELOW the `try`, so they run on both
        /// arms.  `_handle_jitexception` returns the portal frame
        /// (`blackhole.py`) after
        /// `_handle_jitexception_in_portal` has already installed the
        /// re-entered portal's result as the CALLER's return value
        /// (`blackhole.py` → `warmspot.py handle_jitexception_from_blackhole`), so that
        /// frame is finished and the loop must continue in its caller.
        ///
        /// The chain here is `inner -> caller`.  `inner` carries the
        /// portal `jitdriver_sd` and inline-calls a sub-jitcode whose
        /// `jit_merge_point` is bottommost (the callee interpreter
        /// `handler_inline_call_nested_ext` seats has no
        /// `nextblackholeinterp`), so
        /// the `ContinueRunningNormally` reaches `_run_forever` from a
        /// frame that DOES have a caller — the recursive-portal arm.
        /// Re-entering `inner` instead would run its tail a second time
        /// and overwrite the portal runner's result with `inner`'s own.
        /// `blackhole.py`: the walk up the chain stops at the frame
        /// whose jitcode names a jitdriver, and a bottommost such frame
        /// publishes its terminal image before the exception propagates.
        ///
        /// That only holds because `call.py:148` gives every portal jitcode its
        /// `jitdriver_sd` back-pointer.  Without it the walk runs off the end of
        /// the chain and returns through the "no portal found" arm, where the
        /// image is never taken — which is what a `#[jit_interp]` frontend saw
        /// before `JitDriver::register_dispatch_jitcode` set the slot.
        #[test]
        fn handle_jitexception_publishes_terminal_only_when_a_frame_names_its_jitdriver() {
            let build_chain = |portal: bool| {
                let mut b = JitCodeBuilder::default();
                b.load_const_i_value(0, 42);
                b.int_return(0);
                let jitcode = b.finish();
                jitcode.set_index(0);
                if portal {
                    jitcode.set_jitdriver_sd(0);
                }
                jitcode
            };

            for portal in [true, false] {
                let mut builder =
                    super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
                let mut bh = builder.acquire_interp();
                bh.setposition(std::sync::Arc::new(build_chain(portal)), 0);
                bh.registers_i[0] = 42;

                let mut terminal = None;
                let outcome = super::handle_jitexception(
                    &mut builder,
                    bh,
                    crate::jitexc::JitException::DoneWithThisFrameInt(42),
                    None,
                    None,
                    Some(&mut terminal),
                );
                assert!(
                    outcome.is_err(),
                    "a bottommost frame always propagates the exception",
                );
                assert_eq!(
                    terminal.is_some(),
                    portal,
                    "terminal image published for portal={portal}",
                );
                if let Some(terminal) = terminal {
                    assert_eq!(terminal.registers_i[0], 42);
                }
            }
        }

        /// `pyopcode.py handle_operation_error` stores
        /// `frame_finished_execution = True` on the no-handler propagation out
        /// of a frame, and the walk to the recursive portal
        /// (`blackhole.py`) discards exactly such frames.  The `Ok` arm's
        /// twin store lives in `run_forever_with_portal`, which never sees a
        /// level this walk released, so a frame left by an exception would read
        /// back as still executing and refuse `frame.clear()`.
        #[test]
        fn handle_jitexception_finishes_each_non_portal_level_it_unwinds() {
            let build = |portal: bool| {
                let mut b = JitCodeBuilder::default();
                b.load_const_i_value(0, 1);
                b.int_return(0);
                let jitcode = b.finish();
                jitcode.set_index(0);
                if portal {
                    jitcode.set_jitdriver_sd(0);
                }
                jitcode
            };

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut portal_bh = builder.acquire_interp();
            portal_bh.setposition(std::sync::Arc::new(build(true)), 0);
            portal_bh.virtualizable_ptr = 0x2000;
            let mut leaf = builder.acquire_interp();
            leaf.setposition(std::sync::Arc::new(build(false)), 0);
            leaf.virtualizable_ptr = 0x1000;
            leaf.nextblackholeinterp = Some(portal_bh);

            let left = std::cell::RefCell::new(Vec::new());
            let on_leave_level = |ptr: i64| left.borrow_mut().push(ptr);
            let outcome = super::handle_jitexception(
                &mut builder,
                leaf,
                crate::jitexc::JitException::DoneWithThisFrameInt(1),
                None,
                Some(&on_leave_level),
                None,
            );

            assert!(outcome.is_err(), "the portal frame here is bottommost");
            assert_eq!(
                left.into_inner(),
                vec![0x1000],
                "the unwound non-portal level is finished; the portal one leaves \
                 through the propagating arm, whose caller owns the store",
            );
        }

        #[test]
        fn run_forever_steps_past_the_portal_frame_after_handle_jitexception() {
            let mut sub = JitCodeBuilder::default();
            sub.jit_merge_point(0, &[], &[], &[], &[], &[], &[]);
            let sub_jitcode = sub.finish();

            let mut inner_b = JitCodeBuilder::default();
            let sub_idx = inner_b.add_sub_jitcode(sub_jitcode);
            inner_b.inline_call_ir_v(sub_idx, &[], &[], None);
            // The tail the bug re-executes: it would hand the caller `7`.
            inner_b.load_const_i_value(1, 7);
            inner_b.int_return(1);
            let inner_jitcode = inner_b.finish();
            inner_jitcode.set_jitdriver_sd(0);

            // `_setup_return_value_i` writes `registers_i[code[position-1]]`
            // (`blackhole.py`), so the caller resumes at an
            // `int_return` whose preceding byte is the `int_copy`
            // destination register 0.
            let mut caller_b = JitCodeBuilder::default();
            caller_b.load_const_i_value(0, 42);
            let caller_resume = caller_b.current_pos();
            caller_b.int_return(0);
            let caller_jitcode = caller_b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            // `jitdrivers_sd[jdindex]` for the stamped portal jitcode.
            builder.setup_jitdrivers_sd(vec![super::BhJitDriverSd {
                result_type: super::BhReturnType::Int,
                ..Default::default()
            }]);
            let mut caller = builder.acquire_interp();
            caller.setposition(std::sync::Arc::new(caller_jitcode), caller_resume);
            let mut inner = builder.acquire_interp();
            inner.setposition(std::sync::Arc::new(inner_jitcode), 0);
            inner.nextblackholeinterp = Some(caller);

            let portal_runner =
                |_exc: &crate::jitexc::JitException| Ok((super::BhReturnType::Int, 999));
            let outcome = super::run_forever_with_portal(
                &mut builder,
                inner,
                0,
                Some(&portal_runner),
                None,
                None,
                None,
            );
            assert!(
                matches!(
                    outcome,
                    crate::jitexc::JitException::DoneWithThisFrameInt(999)
                ),
                "the caller must return the portal runner's result, got {outcome:?}",
            );
        }

        /// PyPy keeps the re-entered blackhole object live while
        /// `_handle_jitexception_in_portal` propagates.  Pyre crosses a Rust
        /// callback after releasing it, so the callback must return the
        /// terminal banks with a pyre-only interpreter bail.  Publishing the
        /// recursive portal frame instead would resume at its CALL and replay
        /// the callee's already-performed effects.
        #[test]
        fn recursive_portal_bail_preserves_the_reentered_terminal_image() {
            let mut sub = JitCodeBuilder::default();
            sub.jit_merge_point(0, &[], &[], &[], &[], &[], &[]);
            let sub_jitcode = sub.finish();

            let mut inner_b = JitCodeBuilder::default();
            let sub_idx = inner_b.add_sub_jitcode(sub_jitcode);
            inner_b.inline_call_ir_v(sub_idx, &[], &[], None);
            let inner_jitcode = inner_b.finish();
            inner_jitcode.set_jitdriver_sd(0);

            let mut caller_b = JitCodeBuilder::default();
            caller_b.void_return();
            let caller_jitcode = caller_b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            // `jitdrivers_sd[jdindex]` for the stamped portal jitcode.
            builder.setup_jitdrivers_sd(vec![super::BhJitDriverSd {
                result_type: super::BhReturnType::Int,
                ..Default::default()
            }]);
            let mut caller = builder.acquire_interp();
            caller.setposition(std::sync::Arc::new(caller_jitcode), 0);
            let mut inner = builder.acquire_interp();
            inner.setposition(std::sync::Arc::new(inner_jitcode), 0);
            inner.nextblackholeinterp = Some(caller);

            const TERMINAL_JITCODE: usize = 73;
            const TERMINAL_POSITION: usize = 19;
            let portal_runner = |_exc: &crate::jitexc::JitException| {
                Err(super::PortalRunnerFailure::bail(
                    super::BlackholeTerminalImage {
                        jitcode_index: TERMINAL_JITCODE,
                        registers_i: vec![111],
                        registers_r: vec![222],
                        registers_f: vec![333],
                        position: TERMINAL_POSITION,
                        last_opcode_position: 17,
                        // The resumable bail: a consumer that checks the flag
                        // before adopting this image accepts it.
                        abort_permanent_bail: true,
                        bail_residual: None,
                    },
                ))
            };
            let mut terminal = None;
            let outcome = super::run_forever_with_portal(
                &mut builder,
                inner,
                0,
                Some(&portal_runner),
                None,
                None,
                Some(&mut terminal),
            );

            assert!(matches!(
                outcome,
                crate::jitexc::JitException::BailToInterpreter
            ));
            let terminal = terminal.expect("recursive portal bail lost its terminal frame");
            assert_eq!(terminal.jitcode_index, TERMINAL_JITCODE);
            assert_eq!(terminal.position, TERMINAL_POSITION);
            assert_eq!(terminal.registers_i, vec![111]);
            assert_eq!(terminal.registers_r, vec![222]);
            assert_eq!(terminal.registers_f, vec![333]);
        }

        /// `blackhole.py:1759` is the single position a returning level leaves
        /// from, so `on_leave_level` fires once there per level.  The store the
        /// callback carries (`frame_finished_execution`) is idempotent and
        /// hides a second fire, but the embedder hangs per-leave bookkeeping
        /// off the same callback — pyre returns one recursion unit per level —
        /// and a duplicate retires the caller's unit while the caller is still
        /// running.
        #[test]
        fn the_leave_callback_fires_once_for_the_level_that_returns() {
            const INNER_VABLE: i64 = 0x1111;
            const CALLER_VABLE: i64 = 0x2222;

            let mut sub = JitCodeBuilder::default();
            sub.jit_merge_point(0, &[], &[], &[], &[], &[], &[]);
            let sub_jitcode = sub.finish();

            let mut inner_b = JitCodeBuilder::default();
            let sub_idx = inner_b.add_sub_jitcode(sub_jitcode);
            inner_b.inline_call_ir_v(sub_idx, &[], &[], None);
            inner_b.load_const_i_value(1, 7);
            inner_b.int_return(1);
            let inner_jitcode = inner_b.finish();
            inner_jitcode.set_jitdriver_sd(0);

            let mut caller_b = JitCodeBuilder::default();
            caller_b.load_const_i_value(0, 42);
            let caller_resume = caller_b.current_pos();
            caller_b.int_return(0);
            let caller_jitcode = caller_b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            // `jitdrivers_sd[jdindex]` for the stamped portal jitcode.
            builder.setup_jitdrivers_sd(vec![super::BhJitDriverSd {
                result_type: super::BhReturnType::Int,
                ..Default::default()
            }]);
            let mut caller = builder.acquire_interp();
            caller.setposition(std::sync::Arc::new(caller_jitcode), caller_resume);
            caller.virtualizable_ptr = CALLER_VABLE;
            let mut inner = builder.acquire_interp();
            inner.setposition(std::sync::Arc::new(inner_jitcode), 0);
            inner.virtualizable_ptr = INNER_VABLE;
            inner.nextblackholeinterp = Some(caller);

            let left = std::cell::RefCell::new(Vec::new());
            let on_leave_level = |ptr: i64| left.borrow_mut().push(ptr);
            let portal_runner =
                |_exc: &crate::jitexc::JitException| Ok((super::BhReturnType::Int, 999));
            let outcome = super::run_forever_with_portal(
                &mut builder,
                inner,
                0,
                Some(&portal_runner),
                None,
                Some(&on_leave_level),
                None,
            );
            assert!(
                matches!(
                    outcome,
                    crate::jitexc::JitException::DoneWithThisFrameInt(999)
                ),
                "the caller must return the portal runner's result, got {outcome:?}",
            );
            // One entry per level, in the order each was left.  The inner
            // level leaves through `run_forever_with_portal`'s post-return
            // store; the caller carries no `jitdriver_sd`, so it is not the
            // portal level `handle_jitexception` stops at but one of the
            // levels its walk releases, and it leaves there.  Two entries for
            // two levels is the contract — a repeat of either would be the
            // double store that retires a still-running caller's recursion
            // unit.
            assert_eq!(
                left.into_inner(),
                vec![INNER_VABLE, CALLER_VABLE],
                "one leave per level, none twice",
            );
        }

        /// `bhimpl_jit_merge_point`'s recursive-portal branch
        /// (`blackhole.py`) and `get_portal_runner`
        /// (`blackhole.py`) both index `jitdrivers_sd[jdindex]`
        /// unchecked.  Upstream can, because the table hangs off
        /// `self.builder.metainterp_sd`: one table, reached by every frame of
        /// a chain through its builder back-reference.  A pool-owned Rust
        /// interpreter holds a snapshot instead, so the guarantee has to come
        /// from `acquire_interp` — `blackhole_from_resumedata` builds a chain
        /// by acquiring one interpreter per resumed frame and never touches
        /// the table, and every frame above the bottom one takes the
        /// recursive-portal branch.
        #[test]
        fn every_interp_a_builder_hands_out_carries_the_jitdriver_table() {
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            builder.setup_jitdrivers_sd(vec![super::BhJitDriverSd {
                result_type: super::BhReturnType::Int,
                ..Default::default()
            }]);

            // The shape `blackhole_from_resumedata` produces: one acquire per
            // resumed frame, linked caller-ward through `nextblackholeinterp`.
            let caller = builder.acquire_interp();
            let mut inner = builder.acquire_interp();
            inner.nextblackholeinterp = Some(caller);

            let mut frame = Some(&*inner);
            let mut depth = 0;
            while let Some(f) = frame {
                assert_eq!(
                    f.jitdrivers_sd.len(),
                    1,
                    "chain frame {depth} cannot resolve jdindex 0",
                );
                assert_eq!(f.get_portal_runner(0).0, 0);
                depth += 1;
                frame = f.nextblackholeinterp.as_deref();
            }
            assert_eq!(depth, 2, "expected a two-frame chain");

            // `release_interp` leaves the table on the pooled interpreter, so
            // the refresh must overwrite it.  Otherwise a builder whose table
            // is empty still hands out the previous run's entries, which is
            // what let an unseeded resume path look like it worked.
            builder.release_interp(inner);
            builder.setup_jitdrivers_sd(Vec::new());
            assert!(
                builder.acquire_interp().jitdrivers_sd.is_empty(),
                "a pooled interp must not carry the previous run's table",
            );
        }

        /// `state_field_layout` describes the register file of the machine one
        /// resume belongs to, so it must not survive into the pool.  The
        /// `assert!` range checks in `scalar_slot`/`ref_scalar_slot`/
        /// `array_identity_slots` fire on a stale or all-zero layout, or, when
        /// a field index happens to be in range, the slot named is the previous
        /// machine's — not the identity slots `ref_scalar_base` keeps clear of
        /// the dispatch JitCode's argument registers.
        #[test]
        fn released_interp_does_not_leak_its_state_field_layout_to_the_next_user() {
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.state_field_layout = super::StateFieldLayout::with_ref_scalars(1, 0, 0, 2, 8, 0);
            assert_eq!(bh.state_field_layout.ref_scalar_slot(0), 8);

            builder.release_interp(bh);
            let reused = builder.acquire_interp();
            assert_eq!(
                reused.state_field_layout,
                super::StateFieldLayout::default(),
                "a pooled interp must not carry the previous resume's layout",
            );
        }

        /// Tier 2.1: ref-typed return propagation through
        /// `handler_inline_call_nested_ext`.  Sub-jitcode is the ref
        /// identity (passes its ref arg through and `ref_return`s it);
        /// caller passes a non-trivial constant ref and verifies the
        /// caller-side ref dst slot received the same word.
        #[test]
        fn test_bh_interp_inline_call_ref_return() {
            let mut sub = JitCodeBuilder::default();
            sub.ref_return(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0xDEAD_BEEF);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_r(sub_idx, &[], &[(0, 0)], Some(1));
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_r[1], 0xDEAD_BEEF);
        }

        /// Tier 2.1: float-typed return propagation.
        /// Sub-jitcode is the float identity; caller passes a constant
        /// float bit-pattern and verifies the caller-side float dst.
        #[test]
        fn test_bh_interp_inline_call_float_return() {
            let mut sub = JitCodeBuilder::default();
            sub.float_return(0);
            let sub_jitcode = sub.finish();

            let bits = f64::to_bits(3.25_f64) as i64;
            let mut b = JitCodeBuilder::default();
            b.load_const_f_value(0, bits);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_irf_f(sub_idx, &[], &[], &[(0, 0)], Some(1));
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_f[1], bits);
        }

        /// Tier 2.1: void-return path skips the
        /// `trailing_return_info()` block in the handler (returns None
        /// for `BC_VOID_RETURN`-terminated jitcodes per
        /// `JitCodeRuntimeExt::trailing_return_info`).  No caller dst
        /// slot is consumed; the caller proceeds without aborted /
        /// got_exception flags being set.  Sub-jitcode keeps register
        /// files empty by skipping all `touch_reg` paths — `void_return`
        /// alone does not touch any register, so a no-arg call is the
        /// minimal void scenario.
        #[test]
        fn test_bh_interp_inline_call_void_return() {
            let mut sub = JitCodeBuilder::default();
            sub.void_return();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 99);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(!bh.aborted, "void inline_call should not abort");
            assert!(
                !bh.got_exception,
                "void inline_call should not set got_exception"
            );
        }

        /// Tier 2.2: exception raised inside the callee is
        /// caught by the caller's `catch_exception/L` immediately
        /// following the inline_call.  The handler should sync
        /// `bh.position` to operand-end before invoking
        /// `handle_exception_in_frame`, then return `Ok(bh.position)`
        /// (= catch-handler PC) so subsequent dispatch resumes there.
        #[test]
        fn test_bh_interp_inline_call_raises_caught_by_caller() {
            const EXC_VAL: i64 = 0xCAFE_F00D;
            const FALLTHROUGH_SENTINEL: i64 = 999;
            const CAUGHT_SENTINEL: i64 = 42;

            let mut sub = JitCodeBuilder::default();
            sub.load_const_r_value(0, EXC_VAL);
            sub.emit_raise(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            let handler_lbl = b.new_label();
            b.catch_exception(handler_lbl);
            // Fallthrough path — must not run if catch dispatch is wired.
            b.load_const_i_value(2, FALLTHROUGH_SENTINEL);
            b.int_return(2);
            // Catch handler — only reached if `handle_exception_in_frame`
            // honoured the immediately-following `catch_exception/L`.
            b.mark_label(handler_lbl);
            b.load_const_i_value(2, CAUGHT_SENTINEL);
            b.int_return(2);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                !bh.got_exception,
                "caller-side catch_exception should clear got_exception"
            );
            assert_eq!(
                bh.tmpreg_i, CAUGHT_SENTINEL,
                "caught path must run; fallthrough sentinel is {FALLTHROUGH_SENTINEL}"
            );
            assert_eq!(bh.return_type, BhReturnType::Int);
        }

        #[test]
        fn test_guard_exception_resume_uses_operation_owned_trailing_live() {
            let mut asm = majit_jitcode::codewriter::assembler::Assembler::new();
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 1);
            b.load_const_i_value(0, 2);
            b.live(&mut asm, &[], &[], &[]);
            // The normal-flow successor mirrors virtualizable state before
            // the can-raise block's trailing live/catch pair.
            b.vable_setfield_int_with_base(0, 0, 0);
            let resume_pc = b.current_pos();
            b.live(&mut asm, &[], &[], &[]);
            let handler_lbl = b.new_label();
            b.catch_exception(handler_lbl);
            b.load_const_i_value(2, 99);
            b.int_return(2);
            b.mark_label(handler_lbl);
            let handler_pc = b.current_pos();
            b.load_const_i_value(2, 42);
            b.int_return(2);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), resume_pc);

            assert!(bh.handle_exception_in_frame(0xCAFE_F00D));
            assert_eq!(bh.position, handler_pc);
        }

        #[test]
        fn test_guard_exception_resume_does_not_steal_later_operation_catch() {
            let mut asm = majit_jitcode::codewriter::assembler::Assembler::new();
            let mut b = JitCodeBuilder::default();
            let resume_pc = b.current_pos();
            b.live(&mut asm, &[], &[], &[]);
            b.load_const_i_value(0, 1);
            b.live(&mut asm, &[], &[], &[]);
            let handler_lbl = b.new_label();
            b.catch_exception(handler_lbl);
            b.mark_label(handler_lbl);
            let handler_pc = b.current_pos();
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), resume_pc);

            assert!(!bh.handle_exception_in_frame(0xCAFE_F00D));
            assert_eq!(bh.position, resume_pc);
            assert_ne!(bh.position, handler_pc);
        }

        thread_local! {
            /// Address of the interpreter's `exception_last_value` slot, read
            /// back by [`probe_exception_slot_at_record_time`].
            static PROBE_SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
            /// What that slot held when the recorder ran.
            static PROBE_SEEN: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
        }

        /// Stand-in for pyre's `record_caught_blackhole_traceback`.  The real
        /// one allocates a `PyTraceback`, i.e. it is a GC safepoint; this one
        /// just samples the slot that has to be covering the exception by then.
        extern "C" fn probe_exception_slot_at_record_time(
            _exc: i64,
            _frame: i64,
            _jitcode_index: i64,
            _opcode_position: i64,
        ) {
            let slot = PROBE_SLOT.with(|c| c.get()) as *const i64;
            let seen = if slot.is_null() { -1 } else { unsafe { *slot } };
            PROBE_SEEN.with(|c| c.set(seen));
        }

        /// `route_to_catch` must publish the exception into the walked
        /// `exception_last_value` slot BEFORE it invokes the traceback
        /// recorder.
        ///
        /// The recorder allocates, and on this path — a `raise` inside an
        /// inlined callee, caught by the caller — that slot is the exception's
        /// only root: `handler_inline_call_nested_ext` reads
        /// `callee.exception_last_value` after `callee.run()` popped the
        /// callee's `push_bh_regs` entry, and a callee-local raise never went
        /// through `BH_LAST_EXC_VALUE`.  The exception is a nursery object, so
        /// the record moves it; an unrooted local there is stale.
        #[test]
        fn test_bh_route_to_catch_roots_the_exception_before_recording() {
            const EXC_VAL: i64 = 0xCAFE_F00D;

            let mut sub = JitCodeBuilder::default();
            sub.load_const_r_value(0, EXC_VAL);
            sub.emit_raise(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            let handler_lbl = b.new_label();
            b.catch_exception(handler_lbl);
            b.load_const_i_value(2, 999);
            b.int_return(2);
            b.mark_label(handler_lbl);
            b.load_const_i_value(2, 42);
            b.int_return(2);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            bh.record_caught_exception = Some(probe_exception_slot_at_record_time);

            PROBE_SEEN.with(|c| c.set(-1));
            PROBE_SLOT.with(|c| c.set(std::ptr::addr_of!(bh.exception_last_value) as usize));
            let _ = bh.run();
            PROBE_SLOT.with(|c| c.set(0));

            assert_eq!(
                PROBE_SEEN.with(|c| c.get()),
                EXC_VAL,
                "the recorder is a GC safepoint, so `exception_last_value` must \
                 already carry the exception when it runs"
            );
        }

        /// Tier 2.2: exception raised inside the callee
        /// without a caller-side `catch_exception/L` propagates as
        /// `LeaveFrame` with `bh.got_exception` and
        /// `bh.exception_last_value` set.  The handler's
        /// `handle_exception_in_frame` returns false because no
        /// matching catch byte follows the inline_call operands.
        #[test]
        fn test_bh_interp_inline_call_raises_uncaught_propagates() {
            const EXC_VAL: i64 = 0x1234_5678;

            let mut sub = JitCodeBuilder::default();
            sub.load_const_r_value(0, EXC_VAL);
            sub.emit_raise(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            // No `catch_exception` — the next opcode is unreachable but
            // still needs to be a valid encoding so `JitCodeBuilder::finish`
            // produces a well-formed jitcode tail.
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.got_exception,
                "uncaught exception should set got_exception"
            );
            assert_eq!(
                bh.exception_last_value, EXC_VAL,
                "exception_last_value must propagate the callee's raised ref"
            );
        }

        /// Tier 2.3: callee `abort_permanent/` propagates
        /// `aborted = true` to the caller via the handler's
        /// `if callee.aborted { bh.aborted = true; LeaveFrame }` arm.
        /// `bhimpl_abort_permanent` sets aborted
        /// and returns LeaveFrame when no `BH_LAST_EXC_VALUE` is pending,
        /// which is the case for a clean callee seated by
        /// `handler_inline_call_nested_ext` (`clone_context_from` +
        /// `setposition`) (TLS reset between tests via thread isolation).
        #[test]
        fn test_bh_interp_inline_call_abort_permanent_in_sub_propagates() {
            let mut sub = JitCodeBuilder::default();
            sub.abort_permanent();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            // Trailing return — must not run on the abort path.
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.aborted,
                "callee `abort_permanent/` must propagate aborted=true to caller"
            );
            assert!(
                !bh.got_exception,
                "abort_permanent without pending TLS exception must not set got_exception"
            );
        }

        /// `bhimpl_abort_permanent` must re-home the exception into a walked
        /// root before it clears `BH_LAST_EXC_VALUE`.
        ///
        /// The cell is the only root over a residual-call raise
        /// (`walk_bh_last_exc_value`), and the `RaiseException` handoff runs
        /// `handle_exception_in_frame` → `route_to_catch`, which records a
        /// traceback node — an allocation — before it stores the value itself.
        /// The exception is a nursery object, so clearing the cell first
        /// leaves it unrooted across that allocation and the minor moves it.
        /// `exception_last_value` is a `walk_bh_regs` root, so it must already
        /// carry the exception at the moment the cell goes to zero.
        #[test]
        fn test_bh_abort_permanent_rehomes_exception_into_walked_root() {
            const EXC: i64 = 0x5eed_0001;
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            super::BH_LAST_EXC_VALUE.with(|c| c.set(EXC));

            let outcome = bh.bhimpl_abort_permanent();

            assert!(
                matches!(outcome, Err(super::DispatchError::RaiseException { exc, .. }) if exc == EXC),
                "a pending TLS exception must route through RaiseException"
            );
            assert_eq!(
                bh.exception_last_value, EXC,
                "the cell is cleared here, so the walked `exception_last_value` \
                 slot must already carry the exception"
            );
            assert_eq!(
                super::BH_LAST_EXC_VALUE.with(|c| c.get()),
                0,
                "the cell is drained once the value has another root"
            );
            super::BH_LAST_EXC_VALUE.with(|c| c.set(0));
        }

        /// Tier 2.3: callee `abort/` propagates aborted=true.
        /// callee shares the parent's dispatch_table, so
        /// byte `BC_ABORT` fires the wired handler
        /// (`handler_abort_marker`), which sets `aborted = true` +
        /// LeaveFrame.  The
        /// `if callee.aborted { bh.aborted = true; LeaveFrame }` arm
        /// inside `handler_inline_call_nested_ext` then propagates to
        /// the caller.
        #[test]
        fn test_bh_interp_inline_call_abort_in_sub_propagates() {
            let mut sub = JitCodeBuilder::default();
            sub.abort();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.aborted,
                "callee `abort/` must propagate aborted=true to caller"
            );
        }

        /// 2-level nested inline_call.  `outer_sub` itself
        /// emits `BC_INLINE_CALL` to invoke `inner_sub`.  Acceptance
        /// criterion for 's callee runtime context clone:
        /// the inner-most callee inherits the parent's dispatch_table
        /// via `clone_context_from`, so byte 17 routes to
        /// `handler_inline_call_nested_ext` recursively.  The
        /// caller-side ground truth is that the int value threaded
        /// through three frames lands in the outermost caller's
        /// destination register.
        #[test]
        fn test_bh_interp_inline_call_two_level_nested_int_propagates() {
            // inner_sub: receives r0, returns it.
            let mut inner = JitCodeBuilder::default();
            inner.int_return(0);
            let inner_jitcode = inner.finish();

            // outer_sub: receives r0, calls inner_sub(r0) → r1, returns r1.
            let mut outer = JitCodeBuilder::default();
            let inner_idx_in_outer = outer.add_sub_jitcode(inner_jitcode);
            outer.inline_call_ir_i(inner_idx_in_outer, &[(0, 0)], &[], Some(1));
            outer.int_return(1);
            let outer_jitcode = outer.finish();

            // main: r0 = 17, inline_call(outer_sub, arg=(0, 0)) → r1, return r1.
            let mut main = JitCodeBuilder::default();
            main.load_const_i_value(0, 17);
            let outer_idx_in_main = main.add_sub_jitcode(outer_jitcode);
            main.inline_call_ir_i(outer_idx_in_main, &[(0, 0)], &[], Some(1));
            main.int_return(1);
            let main_jitcode = main.finish();

            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(main_jitcode), 0);
            let _ = bh.run();

            assert_eq!(
                bh.registers_i[1], 17,
                "2-level nested inline_call must thread int value through both frames"
            );
        }

        /// `clone_context_from` mirrors
        /// `BlackholeInterpBuilder::acquire_interp`'s 6 builder-shared
        /// fields plus parent's virtualizable / jitdriver state (Fix 7
        /// extension).  Direct unit check that the callee receives the
        /// parent's `dispatch_table` Arc, the cached control opcode
        /// bytes, and the virtualizable / jitdriver_sd / stack-base
        /// state a sub-jitcode might consult if it contains a vable /
        /// recursive_call opcode.
        #[test]
        fn test_clone_context_from_mirrors_acquire_interp_fields() {
            let mut builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let mut parent = builder.acquire_interp();
            let table: &'static [BhDescr; 2] = Box::leak(Box::new([
                BhDescr::VableField { index: 1 },
                BhDescr::VableArray { index: 2 },
            ]));
            parent.descrs = table;
            // Make the parent's vable / jitdriver state non-default so
            // the assertion below distinguishes "copied" from
            // "callee-default".
            parent.virtualizable_ptr = 0xDEAD_BEEF;
            parent.virtualizable_stack_base = 7;
            let vinfo = Box::leak(Box::new(crate::virtualizable::VirtualizableInfo::new(8)));
            parent.virtualizable_info = vinfo;

            let mut callee = BlackholeInterpreter::default();
            callee.clone_context_from(&parent);

            assert!(
                std::sync::Arc::ptr_eq(&callee.dispatch_table, &parent.dispatch_table),
                "callee must share the parent's dispatch_table Arc"
            );
            assert_eq!(callee.op_catch_exception, parent.op_catch_exception);
            assert_eq!(callee.op_rvmprof_code, parent.op_rvmprof_code);
            assert_eq!(callee.op_live, parent.op_live);
            assert!(
                std::ptr::eq(
                    std::ptr::from_ref(callee.descrs).cast::<()>(),
                    std::ptr::from_ref(parent.descrs).cast::<()>(),
                ),
                "clone_context_from must alias the parent table"
            );
            assert_eq!(callee.virtualizable_ptr, parent.virtualizable_ptr);
            assert!(
                std::ptr::eq(callee.virtualizable_info, parent.virtualizable_info),
                "inline callee must inherit the parent's virtualizable_info \
                 (interpret_unresolved_inline_call clones through here)"
            );
            assert_eq!(
                callee.virtualizable_stack_base,
                parent.virtualizable_stack_base
            );
            assert_eq!(callee.jitdrivers_sd.len(), parent.jitdrivers_sd.len());
        }

        /// `handler_abort_marker` defines the pyre
        /// `BC_ABORT` marker semantics.  Reaching `OpKind::Abort` at
        /// runtime sets `aborted = true` and exits the frame —
        /// continuing dispatch past it would misread the next bytes
        /// as opcodes.
        #[test]
        fn test_all_abort_handlers_set_aborted_and_leave_frame() {
            let handlers: &[super::BhOpcodeHandler] = &[
                super::handler_abort_marker,
                super::handler_abort_result_marker_i,
                super::handler_abort_result_marker_r,
            ];
            for handler in handlers {
                let mut builder = BlackholeInterpBuilder::new();
                let mut bh = builder.acquire_interp();
                assert!(!bh.aborted);
                let result = handler(&mut bh, &[0], 0);
                match result {
                    Err(super::DispatchError::LeaveFrame) => {}
                    other => panic!("abort handler must return Err(LeaveFrame), got {other:?}"),
                }
                assert!(bh.aborted);
            }
        }

        /// Integration test: build bytecode manually with known opcode
        /// assignments and run it through the orthodox dispatch_loop.
        ///
        /// This validates the Phase D setup_insns → dispatch_loop → bhimpl
        /// pipeline end-to-end without depending on SSARepr assembly.
        ///
        /// RPython equivalent: the setup_insns + dispatch_loop closure + bhimpl
        /// flow described in blackhole.py and 452-460.
        #[test]
        fn test_orthodox_dispatch_loop_int_add() {
            // Build a minimal insns dict (as if the assembler had produced it).
            // Opcode 0 = "live/" (liveness marker, skip 2 bytes)
            // Opcode 1 = "int_add/ii>i" (3 register bytes: a, b, dst)
            // Opcode 2 = "int_return/i" (1 register byte)
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            insns.insert("live/".to_string(), 0u8);
            insns.insert("int_add/ii>i".to_string(), 1u8);
            insns.insert("int_return/i".to_string(), 2u8);

            // Setup builder
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            // Hand-assemble bytecode: live + int_add(r0, r1) → r2 + int_return(r2)
            let code: Vec<u8> = vec![
                0, 0, 0, // opcode 0 = live/, 2 bytes liveness offset (skipped)
                1, 0, 1, 2, // opcode 1 = int_add, a=r0, b=r1, dst=r2
                2, 2, // opcode 2 = int_return, src=r2
            ];

            // Acquire BlackholeInterpreter from the same builder so the
            // 6 builder-shared fields (op_live etc.) flow through, then
            // size the int register file for r0..=r2.
            let mut bh = builder.acquire_interp();
            bh.registers_i = vec![0i64; 3];
            bh.registers_i[0] = 10; // r0 = 10
            bh.registers_i[1] = 32; // r1 = 32

            // Run dispatch_loop
            let result = builder.dispatch_loop(&mut bh, &code, 0);

            // Should leave frame with LeaveFrame
            assert!(matches!(result, Err(DispatchError::LeaveFrame)));
            // tmpreg_i should hold 10 + 32 = 42
            assert_eq!(bh.tmpreg_i, 42, "int_add(10, 32) should produce 42");
            assert_eq!(bh.return_type, BhReturnType::Int);
        }

        /// Literal port of `test_blackhole.py test_simple_const` and
        /// `test_simple_loop`: these are the oracle fixtures that require
        /// independently decoded `c` operands in arithmetic and fused
        /// comparison branches.
        #[test]
        fn pypy_short_constant_arithmetic_and_branch_forms_execute() {
            let mut const_insns = indexmap::IndexMap::new();
            const_insns.insert("int_sub/ci>i".to_string(), 0);
            const_insns.insert("int_return/i".to_string(), 1);
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&const_insns);
            super::wire_bhimpl_handlers(&mut builder);
            let code = [0, 48, 1, 2, 1, 2];
            let mut bh = builder.acquire_interp();
            bh.registers_i = vec![0; 3];
            bh.registers_i[1] = 6;
            let result = builder.dispatch_loop(&mut bh, &code, 0);
            assert!(matches!(result, Err(DispatchError::LeaveFrame)));
            assert_eq!(bh.tmpreg_i, 42);

            let mut loop_insns = indexmap::IndexMap::new();
            loop_insns.insert("goto_if_not_int_gt/icL".to_string(), 0);
            loop_insns.insert("int_add/ii>i".to_string(), 1);
            loop_insns.insert("int_sub/ic>i".to_string(), 2);
            loop_insns.insert("goto/L".to_string(), 3);
            loop_insns.insert("int_return/i".to_string(), 4);
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&loop_insns);
            super::wire_bhimpl_handlers(&mut builder);
            let code = [
                0, 0x16, 2, 0x10, 0, 1, 0x17, 0x16, 0x17, 2, 0x16, 1, 0x16, 3, 0, 0, 4, 0x17,
            ];
            let mut bh = builder.acquire_interp();
            bh.registers_i = vec![0; 0x18];
            bh.registers_i[0x16] = 6;
            bh.registers_i[0x17] = 100;
            let result = builder.dispatch_loop(&mut bh, &code, 0);
            assert!(matches!(result, Err(DispatchError::LeaveFrame)));
            assert_eq!(bh.tmpreg_i, 118);
        }

        #[test]
        fn every_pypy_short_constant_shape_is_wired() {
            // `getarrayitem_gc_pure_{i,r}` in `USE_C_FORM` has no literal
            // `bhimpl_*` owner: blackhole.py names the aliases
            // `getarrayitem_gc_{i,r}_pure`, so those spellings never select
            // the short form in Assembler.write_insn.
            let mut keys = Vec::new();
            for opname in [
                "int_add", "int_sub", "int_and", "int_eq", "int_ne", "int_lt", "int_le", "int_gt",
                "int_ge",
            ] {
                for args in ["ci>i", "ic>i", "cc>i"] {
                    keys.push(format!("{opname}/{args}"));
                }
            }
            for opname in [
                "goto_if_not_int_eq",
                "goto_if_not_int_ne",
                "goto_if_not_int_lt",
                "goto_if_not_int_le",
                "goto_if_not_int_gt",
                "goto_if_not_int_ge",
            ] {
                for args in ["ciL", "icL", "ccL"] {
                    keys.push(format!("{opname}/{args}"));
                }
            }
            keys.extend(
                [
                    "int_copy/c>i",
                    "int_return/c",
                    "jit_merge_point/cIRFIRF",
                    "new_array/cd>r",
                    "new_array_clear/cd>r",
                    "newstr/c>r",
                    "strgetitem/rc>i",
                    "strsetitem/rci",
                    "strsetitem/ric",
                    "strsetitem/rcc",
                    "getarrayitem_gc_i/rcd>i",
                    "getarrayitem_gc_r/rcd>r",
                    "setarrayitem_gc_i/rcid",
                    "setarrayitem_gc_i/ricd",
                    "setarrayitem_gc_i/rccd",
                    "setarrayitem_gc_r/rcrd",
                    "setfield_gc_i/rcd",
                    "copystrcontent/rrcii",
                    "copystrcontent/rrici",
                    "copystrcontent/rriic",
                    "copystrcontent/rrcci",
                    "copystrcontent/rrcic",
                    "copystrcontent/rricc",
                    "copystrcontent/rrccc",
                ]
                .into_iter()
                .map(str::to_string),
            );
            let insns = keys
                .into_iter()
                .enumerate()
                .map(|(opcode, key)| (key, opcode as u8))
                .collect();
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);
            assert!(builder.unwired_opnames().is_empty());
        }

        #[test]
        fn wire_bhimpl_handlers_wires_raw_int_base_access() {
            // Int-bank field/array access is the raw family, not a
            // getfield_gc/setfield_gc alias.
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            insns.insert("getfield_raw_i/id>i".to_string(), 0u8);
            insns.insert("setfield_raw_i/iid".to_string(), 1u8);
            insns.insert("getarrayitem_raw_i/iid>i".to_string(), 2u8);

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for slot in 0usize..=2 {
                assert_ne!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} must be wired",
                );
            }
        }

        #[test]
        fn wire_bhimpl_handlers_rejects_vable_int_base_aliases() {
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            insns.insert("getfield_gc_i/id>i".to_string(), 0u8);
            insns.insert("getfield_gc_r/id>r".to_string(), 1u8);
            insns.insert("setfield_gc_i/iid".to_string(), 2u8);
            insns.insert("setfield_gc_r/ird".to_string(), 3u8);
            insns.insert("getarrayitem_gc_i/iid>i".to_string(), 4u8);
            insns.insert("getfield_vable_i/id>i".to_string(), 5u8);
            insns.insert("setfield_vable_i/iid".to_string(), 6u8);
            insns.insert("setfield_vable_r/ird".to_string(), 7u8);

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            assert_eq!(
                builder.unwired_opnames(),
                vec![
                    "getfield_gc_i/id>i",
                    "getfield_gc_r/id>r",
                    "setfield_gc_i/iid",
                    "setfield_gc_r/ird",
                    "getarrayitem_gc_i/iid>i",
                    "getfield_vable_i/id>i",
                    "setfield_vable_i/iid",
                    "setfield_vable_r/ird",
                ],
            );
        }

        #[test]
        fn wire_bhimpl_handlers_leaves_dead_v_access_aliases_unwired() {
            // Pyre-invented `_v` sentinel forms that no longer leave the
            // assembler (see majit-translate/src/codewriter/
            // assembler.rs negative asserts). Kept as
            // guard tests so any regression that reintroduces a `_v` key
            // surfaces at setup_insns time rather than at first dispatch.
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            insns.insert("setfield_gc_v/rid".to_string(), 0u8);
            insns.insert("setfield_gc_v/iid".to_string(), 1u8);
            insns.insert("setfield_gc_v/ird".to_string(), 2u8);
            insns.insert("setarrayitem_gc_v/riid".to_string(), 3u8);
            insns.insert("setarrayitem_gc_v/iiid".to_string(), 4u8);
            insns.insert("getfield_gc_v/id>i".to_string(), 5u8);
            insns.insert("getarrayitem_gc_v/rid>i".to_string(), 6u8);
            insns.insert("getarrayitem_gc_v/iid>i".to_string(), 7u8);
            insns.insert("getarrayitem_gc_v/ird>i".to_string(), 8u8);
            insns.insert("setfield_vable_v/id".to_string(), 9u8);
            insns.insert("setfield_vable_v/rd".to_string(), 10u8);
            insns.insert("setfield_vable_v/ird".to_string(), 11u8);
            insns.insert("getfield_vable_v/d>i".to_string(), 12u8);
            insns.insert("getfield_vable_v/id>i".to_string(), 13u8);
            insns.insert("input_v/>i".to_string(), 14u8);

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for slot in 0usize..=14 {
                assert_eq!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} must stay unwired — pyre `_v` keys are no \
                     longer produced, so re-wiring would normalise a fake \
                     surface back into the dispatch table",
                );
            }
        }

        #[test]
        fn wire_bhimpl_handlers_leaves_mixed_ref_int_int_ops_unwired() {
            // RPython blackhole integer arithmetic is `@arguments("i", "i",
            // returns="i")` (blackhole.py+). Ref/int-mixed `int_*`
            // opnames are kind-flow bugs, not alternate blackhole surfaces.
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            let fake_opnames = [
                "int_add/ri>i",
                "int_add/ir>i",
                "int_add/rr>i",
                "int_le/ri>i",
                "int_le/rr>i",
                "int_lshift/ri>i",
                "int_rshift/ir>i",
                "int_xor/rr>i",
                "int_same_as/r>i",
                "int_neg/r>i",
                "int_invert/r>i",
            ];
            for (opcode, opname) in fake_opnames.iter().enumerate() {
                insns.insert((*opname).to_string(), opcode as u8);
            }

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for (slot, opname) in fake_opnames.iter().enumerate() {
                assert_eq!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} ({}) must stay unwired; mixed ref/int int \
                     opnames should be fixed at the codewriter/rtyper source",
                    opname,
                );
            }
        }

        #[test]
        fn default_bh_builder_unwired_set_matches_task_85_snapshot() {
            let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
            for (opname, byte) in majit_jitcode::insns::wellknown_bh_insns()
                .into_iter()
                .chain(majit_jitcode::insns::extension_insns())
            {
                insns.insert(opname.to_string(), byte);
            }

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);
            let mut unwired: Vec<String> = builder
                .unwired_opnames()
                .into_iter()
                .map(str::to_string)
                .collect();
            unwired.sort();

            // blackhole.py has no `bhimpl_call_assembler` and no leftover
            // `cond_call_*_ext` / `record_known_result_*_ext`. Those
            // `extension_insns()` bytes stay reserved for tracing
            // `opimpl_call_assembler_*` / `opimpl_cond_call_void`.
            let expected = [
                "call_assembler_float_ext/P",
                "call_assembler_int_ext/P",
                "call_assembler_ref_ext/P",
                "call_assembler_void_ext/P",
                "cond_call_value_int_ext/P",
                "cond_call_value_ref_ext/P",
                "cond_call_void_ext/P",
                "record_known_result_int_ext/P",
                "record_known_result_ref_ext/P",
            ]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
            assert_eq!(
                unwired, expected,
                "the canonical and extension opname tables must not acquire a handler gap; \
                 fix the emitting kind shape instead of wiring a non-orthodox alias",
            );
        }

        #[test]
        fn inline_builder_wires_float_cmp_and_raw_load_f() {
            // A dispatch arm materializing a float comparison as an int
            // (`(a >= b) as i64`) emits `float_ge/ff>i` and siblings, and
            // `majit_raw_load_f` emits `raw_load_f/iid>f`.  Both must be in
            // the inline builder's insns map so `wire_handler` installs a
            // real handler; a forward blackhole resume through an unwired
            // byte panics at `dispatch_step`.
            let builder =
                super::build_inline_call_only_bh_builder(&[("recursive_call_v/iIRFIRF", 34)]);
            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for byte in [
                majit_jitcode::insns::BC_FLOAT_LT,
                majit_jitcode::insns::BC_FLOAT_LE,
                majit_jitcode::insns::BC_FLOAT_EQ,
                majit_jitcode::insns::BC_FLOAT_NE,
                majit_jitcode::insns::BC_FLOAT_GT,
                majit_jitcode::insns::BC_FLOAT_GE,
                majit_jitcode::insns::BC_RAW_LOAD_F,
            ] {
                assert_ne!(
                    builder.dispatch_table[byte as usize] as *const () as usize, placeholder,
                    "opcode byte {byte} must be wired in the inline blackhole builder",
                );
            }
        }
    }

    // ── `support.py _ll_2_int_floordiv` / `_ll_2_int_mod`
    //    C-truncating helper parity tests ─────────────────────────────

    #[test]
    fn _ll_2_int_floordiv_truncates_toward_zero_for_mixed_signs() {
        // `(-7).wrapping_div(3) == -2` (C truncation toward zero), whereas
        // Python-floor `ll_int_py_div(-7, 3) == -3`.
        assert_eq!(super::_ll_2_int_floordiv(-7, 3), -2);
        assert_eq!(super::_ll_2_int_floordiv(7, -3), -2);
        assert_eq!(super::_ll_2_int_floordiv(-7, -3), 2);
        assert_eq!(super::ll_int_py_div(-7, 3), -3);
    }

    #[test]
    fn _ll_2_int_floordiv_matches_division_for_aligned_signs() {
        assert_eq!(super::_ll_2_int_floordiv(7, 3), 2);
        assert_eq!(super::_ll_2_int_floordiv(9, 3), 3);
        assert_eq!(super::_ll_2_int_floordiv(-9, -3), 3);
    }

    #[test]
    fn _ll_2_int_mod_truncates_toward_zero_for_mixed_signs() {
        // `(-7).wrapping_rem(3) == -1` (C truncation toward zero),
        // whereas Python-floor `ll_int_py_mod(-7, 3) == 2`.
        assert_eq!(super::_ll_2_int_mod(-7, 3), -1);
        assert_eq!(super::_ll_2_int_mod(7, -3), 1);
        assert_eq!(super::_ll_2_int_mod(-7, -3), -1);
        assert_eq!(super::ll_int_py_mod(-7, 3), 2);
    }

    #[test]
    fn _ll_2_int_mod_matches_remainder_for_aligned_signs() {
        assert_eq!(super::_ll_2_int_mod(7, 3), 1);
        assert_eq!(super::_ll_2_int_mod(9, 3), 0);
        assert_eq!(super::_ll_2_int_mod(-9, -3), 0);
    }
}

// Blackhole operation handlers (`blackhole.py:452+`)
// RPython defines each bhimpl_* as a static method decorated with
// @arguments("i", "i", returns="i") etc. The handler closure generated
// by _get_method decodes args from the bytecode stream and calls the
// bhimpl method.
//
// In Rust we define each bhimpl as a standalone fn and generate
// BhOpcodeHandler wrappers. The handler decodes operands, calls the
// bhimpl fn, stores the result, and returns the updated position.

// Handler generators for common argument patterns
/// Decode pattern `@arguments("i", "i", returns="i")` — argcodes `"ii>i"`.
///
/// Read 2 int-register indices, call bhimpl fn, write result, advance by 3.
/// Each concrete handler is its own fn (not a closure) so it can be stored
/// as a bare BhOpcodeHandler fn pointer.
macro_rules! bhhandler_ii_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// The three short-constant variants assembler.py creates for an
/// `@arguments("i", "i", returns="i")` operation in `USE_C_FORM`.
/// A `c` operand is the signed byte itself (`blackhole.py` `_get_method`),
/// while an `i` operand indexes the int register/constant namespace.
macro_rules! bhhandler_short_binop_i {
    ($ci:ident, $ic:ident, $cc:ident, $bhimpl:ident) => {
        fn $ci(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = code[position] as i8 as i64;
            let b = bh.registers_i[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }

        fn $ic(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = code[position + 1] as i8 as i64;
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }

        fn $cc(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = code[position] as i8 as i64;
            let b = code[position + 1] as i8 as i64;
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("i", returns="i")` — argcodes `"i>i"`.
macro_rules! bhhandler_i_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("i", "i", "i", returns="i")` — argcodes `"iii>i"`.
macro_rules! bhhandler_iii_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            let c = bh.registers_i[code[position + 2] as usize];
            bh.registers_i[code[position + 3] as usize] = $bhimpl(a, b, c);
            Ok(position + 4)
        }
    };
}

// Blackhole operations ported from `blackhole.py`
/// blackhole.py `bhimpl_int_same_as`.
fn bhimpl_int_same_as(a: i64) -> i64 {
    a
}

/// blackhole.py `bhimpl_int_add(a, b): return intmask(a + b)`.
fn bhimpl_int_add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Port of `blackhole.bhimpl_int_sub`: `return intmask(a - b)`.
fn bhimpl_int_sub(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// Port of `blackhole.bhimpl_int_mul`: `return intmask(a * b)`.
fn bhimpl_int_mul(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// Port of `rint.ll_int_py_div` (oopspec `int.py_div`).
/// The OS_INT_PY_DIV residual call lands here at runtime. The JIT
/// trace contains two explicit guards upstream of this call,
/// produced by the inlined `rint.ll_int_py_div_ovf_zer` wrapper:
///   * `int_eq(rhs, 0) -> guard_false` (zero divisor),
///   * `int_and(int_eq(lhs, INT_MIN), int_eq(rhs, -1)) ->
///     guard_false` (overflow corner — `INT_MIN // -1` would
///     overflow to `INT_MIN` in two's-complement; PyPy
///     `intobject.py`'s `_floordiv`, `descr_floordiv`, and `descr_rfloordiv`
///     route this case through
///     `ovf2long` to long arithmetic).
///     Other negative operand combinations are valid: PyPy
///     `intobject.py`'s `_floordiv` only handles `ZeroDivisionError`,
///     and the no-branch floor correction
///     (`(a ^ b) < 0 && d * b != a -> d - 1`) yields Python-floor
///     semantics for every legal sign combination of `(a, b)`.
///
/// `wrapping_div(0)` panics (Rust integer division by zero is always
/// a panic, in both debug and release builds), and
/// `INT_MIN.wrapping_div(-1)` returns `INT_MIN` (wrong but
/// well-defined).  Both corners are unreachable from the JIT trace
/// path: the `_ovf_zer` wrapper's `int_eq(rhs, 0) -> guard_false` and
/// `(lhs == INT_MIN) & (rhs == -1) -> guard_false` runtime guards
/// (emitted at `specialize.rs::walker_emit_int_div_domain_guards`) bail out the
/// trace before this helper is invoked, matching
/// `rint.ll_int_py_div_ovf_zer`. Direct non-traced
/// callers must respect the same precondition.
///
/// `extern "C"`: the residual-call path
/// (`majit-backend/src/call_stub.rs` invokes the helper through an
/// `extern "C" fn(...)` pointer; matching ABI keeps the function
/// pointer correctly callable from both the Rust-side
/// `specialize.rs::walker_emit_int_py_div_or_mod` (which folds the
/// concrete result during trace recording) and the native call
/// stub.  RPython's `getfunctionptr(graph)` similarly hands the
/// codewriter a real C ABI pointer to the translated helper.
pub extern "C" fn ll_int_py_div(a: i64, b: i64) -> i64 {
    let d = a.wrapping_div(b);
    if (a ^ b) < 0 && d.wrapping_mul(b) != a {
        d.wrapping_sub(1)
    } else {
        d
    }
}

/// Port of `rint.ll_int_py_mod` (oopspec `int.py_mod`).
/// See [`ll_int_py_div`] for the JIT-side runtime guard
/// rationale (`int_eq(rhs, 0)` + `(lhs == INT_MIN) & (rhs == -1)`).
/// Uses `wrapping_rem` for the C-style remainder step, then applies
/// RPython `ll_int_py_mod`'s sign correction only when the remainder
/// and divisor have opposite signs.  As with [`ll_int_py_div`],
/// `wrapping_rem(0)` and `INT_MIN % -1` are unreachable from the
/// trace path; non-traced callers must respect the same precondition.
///
/// `extern "C"`: the residual-call path goes through
/// `majit-backend/src/call_stub.rs` which invokes the helper through
/// an `extern "C" fn(...)` pointer; see [`ll_int_py_div`] for the
/// ABI parity rationale.
pub extern "C" fn ll_int_py_mod(a: i64, b: i64) -> i64 {
    let r = a.wrapping_rem(b);
    if r != 0 && (r ^ b) < 0 {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// RPython `rint.py ll_uint_py_div` (oopspec `int.udiv`).  The
/// `OS_INT_UDIV` residual call lands here at runtime.  Unsigned division
/// has no sign correction to make — `llop.uint_floordiv` translates to
/// the C macro `OP_UINT_FLOORDIV(x, y, r) r = (x) / (y)` — so the whole
/// helper is the raw unsigned quotient of the two operands' bit patterns.
///
/// Unlike the signed [`ll_int_py_div`] there is no `INT_MIN / -1`
/// corner: every pair of `u64` operands with a nonzero divisor has a
/// representable quotient.  The zero divisor remains a precondition
/// (`wrapping_div(0)` panics in every Rust build), spelled in RPython as
/// the `rint.py ll_uint_py_div_zer` wrapper whose check is inlined
/// into the caller and becomes a runtime guard in the trace.
///
/// `extern "C"` for the same residual-call ABI reason as [`ll_int_py_div`].
pub extern "C" fn ll_uint_py_div(a: i64, b: i64) -> i64 {
    ((a as u64) / (b as u64)) as i64
}

/// RPython `rint.py ll_uint_py_mod` (oopspec `int.umod`).  The
/// unsigned remainder needs no sign correction either; see
/// [`ll_uint_py_div`] for the zero-divisor precondition.
pub extern "C" fn ll_uint_py_mod(a: i64, b: i64) -> i64 {
    ((a as u64) % (b as u64)) as i64
}

/// RPython `support.py _ll_2_int_floordiv`: C-truncating
/// floor-division helper.  The upstream comment calls it "the reverse
/// of `rpython.rtyper.rint.ll_int_py_div()`" — i.e. given an input
/// pair `(x, y)` it returns the C-style truncated quotient
/// (rounds toward zero), which is the no-branch reverse of
/// [`ll_int_py_div`]'s Python-floor output.  The RPython body achieves
/// this by starting from Python-floor `x // y` (the source-level `//`
/// is floor in Python 2 / RPython) and adding `((x ^ y) >> 63 &
/// (r * y != x))` to flip negative-with-remainder cases back to
/// truncation.  In Rust, `i64.wrapping_div` is natively C-truncating,
/// so the helper reduces to a direct call.
///
/// The single-segment oopspec name registered with this helper is
/// `int_floordiv` (no `int.py_div` oopspec stamping — that route goes
/// through [`ll_int_py_div`] which carries `@jit.oopspec("int.py_div")`
/// upstream).  Pyre's `jtransform.rs` BinOp{floordiv, Int} arm
/// rewrites to a `CallResidual` to this helper without an
/// `OS_INT_PY_DIV` markup, matching `jtransform.py rewrite_op_int_floordiv = _do_builtin_call` route (a).
///
/// `extern "C"`: residual-call ABI parity — see [`ll_int_py_div`].
pub extern "C" fn _ll_2_int_floordiv(x: i64, y: i64) -> i64 {
    x.wrapping_div(y)
}

/// RPython `support.py _ll_2_int_mod`: C-truncating remainder
/// helper.  See [`_ll_2_int_floordiv`] for the no-branch-reverse
/// rationale.  In Rust, `i64.wrapping_rem` is natively C-truncating.
pub extern "C" fn _ll_2_int_mod(x: i64, y: i64) -> i64 {
    x.wrapping_rem(y)
}

/// RPython `support.py _ll_1_cast_uint_to_float(x)` —
/// `r_uint(x)`-domain `float(x)` (matching `op_cast_uint_to_float`
/// at `opimpl.py`).  `_do_builtin_call` re-routes
/// `cast_uint_to_float` through this helper so blackhole sees a
/// `direct_call` instead of a bare `cast_uint_to_float` opname.
/// Mirror of `opimpl.rs::op_cast_uint_to_float`'s u64-domain
/// rounding so the runtime path agrees with const-fold.
pub fn cast_uint_to_float(x: i64) -> f64 {
    (x as u64) as f64
}

/// Residual-call ABI for [`cast_uint_to_float`]. `jtransform.py`
/// `rewrite_op_cast_uint_to_float = _do_builtin_call` residual-calls
/// `support.py _ll_1_cast_uint_to_float`.
pub extern "C" fn _ll_1_cast_uint_to_float(x: i64) -> f64 {
    cast_uint_to_float(x)
}

/// Residual-call ABI for [`cast_float_to_uint`]. `jtransform.py`
/// `rewrite_op_cast_float_to_uint = _do_builtin_call`.
pub extern "C" fn _ll_1_cast_float_to_uint(f: f64) -> i64 {
    cast_float_to_uint(f)
}

/// RPython `support.py:274 _ll_1_cast_float_to_uint(f)` —
/// `r_uint(long(f))` mod 2^64 wrap (matching
/// `op_cast_float_to_uint` at `opimpl.py`).  Plain
/// `f as u64` saturates outside `[0, 2^64)` rather than wrapping;
/// reuse the IEEE-754 mantissa+exponent decomposition that
/// `opimpl.rs::float_trunc_mod_2_pow_64` already implements so
/// runtime and const-fold agree.  Refuses to fold on NaN / inf
/// (RPython `OverflowError` / `ValueError`); callers must filter.
pub fn cast_float_to_uint(f: f64) -> i64 {
    // `opimpl.py op_cast_float_to_uint` routes through `long(f)` which raises
    // `OverflowError` / `ValueError` on NaN / inf — there is no
    // upstream guard outside this helper that filters non-finite
    // values, so reproduce the fail-loud here unconditionally (not
    // only in debug) instead of returning garbage at release.
    assert!(
        f.is_finite(),
        "cast_float_to_uint: NaN/inf is caller error (opimpl.py:432 raises ValueError)"
    );
    let bits = f.to_bits();
    let sign = (bits >> 63) & 1;
    let raw_exp = ((bits >> 52) & 0x7FF) as i64;
    if raw_exp == 0 {
        return 0;
    }
    let exp = raw_exp - 1023;
    if exp < 0 {
        return 0;
    }
    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let unsigned_trunc = if exp >= 52 {
        let shift = (exp - 52) as u32;
        if shift >= 64 {
            0
        } else {
            mantissa.wrapping_shl(shift)
        }
    } else {
        mantissa >> (52 - exp) as u32
    };
    let wrapped = if sign == 0 {
        unsigned_trunc
    } else {
        unsigned_trunc.wrapping_neg()
    };
    wrapped as i64
}

// `support.py _ll_2_str_eq_nonnull` and the other `_ll_*_str_eq_*`
// helpers `jtransform.py _handle_stroruni_call` registers beside
// `stroruni.equal` live in `pyre_object::lowlevel_string` over the rstr
// `STR` payload, and `jit_fnaddr.rs` publishes their addresses.
// `promote_string` / `promote_unicode` (`str_guard_value`) are not ported:
// the dual `PromoteOrString` hint takes the plain `<kind>_guard_value` arm.

/// blackhole.py `bhimpl_int_and(a, b): return a & b`.
fn bhimpl_int_and(a: i64, b: i64) -> i64 {
    a & b
}

/// blackhole.py `bhimpl_int_or(a, b): return a | b`.
fn bhimpl_int_or(a: i64, b: i64) -> i64 {
    a | b
}

/// blackhole.py `bhimpl_int_xor(a, b): return a ^ b`.
fn bhimpl_int_xor(a: i64, b: i64) -> i64 {
    a ^ b
}

/// blackhole.py `bhimpl_int_rshift(a, b): check_shift_count(b); return a >> b`.
fn bhimpl_int_rshift(a: i64, b: i64) -> i64 {
    // `check_shift_count` is guarded by `not we_are_translated()` upstream;
    // the translated blackhole reaches the machine operation directly.  Pyre
    // is always that translated runtime, so use the word-masked operation
    // rather than turning an out-of-range count into a Rust panic.
    a.wrapping_shr(b as u32)
}

/// blackhole.py `bhimpl_int_lshift(a, b): check_shift_count(b); return intmask(a << b)`.
fn bhimpl_int_lshift(a: i64, b: i64) -> i64 {
    a.wrapping_shl(b as u32)
}

/// blackhole.py `bhimpl_uint_rshift(a, b): check_shift_count(b); return
/// intmask(r_uint(a) >> r_uint(b))`.
fn bhimpl_uint_rshift(a: i64, b: i64) -> i64 {
    (a as u64).wrapping_shr(b as u32) as i64
}

/// blackhole.py `bhimpl_int_neg(a): return intmask(-a)`.
fn bhimpl_int_neg(a: i64) -> i64 {
    a.wrapping_neg()
}

/// blackhole.py `bhimpl_int_invert(a): return ~a`.
fn bhimpl_int_invert(a: i64) -> i64 {
    !a
}

/// blackhole.py `bhimpl_int_lt(a, b): return int(a < b)`.
fn bhimpl_int_lt(a: i64, b: i64) -> i64 {
    (a < b) as i64
}

/// blackhole.py `bhimpl_int_le(a, b): return int(a <= b)`.
fn bhimpl_int_le(a: i64, b: i64) -> i64 {
    (a <= b) as i64
}

/// blackhole.py `bhimpl_int_eq(a, b): return int(a == b)`.
fn bhimpl_int_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}

/// blackhole.py `bhimpl_int_ne(a, b): return int(a != b)`.
fn bhimpl_int_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}

/// blackhole.py `bhimpl_int_gt(a, b): return int(a > b)`.
fn bhimpl_int_gt(a: i64, b: i64) -> i64 {
    (a > b) as i64
}

/// blackhole.py `bhimpl_int_ge(a, b): return int(a >= b)`.
fn bhimpl_int_ge(a: i64, b: i64) -> i64 {
    (a >= b) as i64
}

/// blackhole.py `bhimpl_int_is_true(a): return int(bool(a))`.
fn bhimpl_int_is_true(a: i64) -> i64 {
    (a != 0) as i64
}

/// blackhole.py `bhimpl_int_is_zero(a): return int(not a)`.
fn bhimpl_int_is_zero(a: i64) -> i64 {
    (a == 0) as i64
}

/// blackhole.py `bhimpl_int_force_ge_zero(a): if a < 0: return 0; return a`.
fn bhimpl_int_force_ge_zero(a: i64) -> i64 {
    if a < 0 { 0 } else { a }
}

/// blackhole.py `bhimpl_int_between(a, b, c): return a <= b < c`.
fn bhimpl_int_between(a: i64, b: i64, c: i64) -> i64 {
    (a <= b && b < c) as i64
}

/// blackhole.py `bhimpl_int_signext(a, b): return int_signext(a, b)`.
fn bhimpl_int_signext(a: i64, numbytes: i64) -> i64 {
    crate::support::int_signext(a, numbytes)
}

/// blackhole.py `bhimpl_int_isconstant(x): return False`.
fn bhimpl_int_isconstant(_a: i64) -> i64 {
    0
}

/// blackhole.py `bhimpl_float_isconstant(x): return False`.
fn bhimpl_float_isconstant(_a: f64) -> i64 {
    0
}

/// blackhole.py `bhimpl_convert_float_bytes_to_longlong(a): return
/// float2longlong(a)`.
fn bhimpl_convert_float_bytes_to_longlong(a: f64) -> i64 {
    a.to_bits() as i64
}

/// blackhole.py `bhimpl_convert_longlong_bytes_to_float(a): return
/// longlong2float(a)`.
fn bhimpl_convert_longlong_bytes_to_float(a: i64) -> f64 {
    f64::from_bits(a as u64)
}

/// blackhole.py `bhimpl_cast_float_to_int(a): return int(int(a))`.
///
/// The outer `int()` deliberately makes the untranslated implementation fail
/// when the truncated result is not a machine Signed value.  Rust's `as i64`
/// instead saturates non-finite and out-of-range inputs, silently inventing a
/// result PyPy never returns from this source-level helper.  Keep the valid
/// domain explicit before applying the identical truncation-toward-zero cast.
fn bhimpl_cast_float_to_int(a: f64) -> i64 {
    const SIGNED_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    assert!(
        a.is_finite() && a >= (i64::MIN as f64) && a < SIGNED_UPPER_EXCLUSIVE,
        "bhimpl_cast_float_to_int: {a:?} is outside the Signed range",
    );
    a as i64
}

/// blackhole.py `bhimpl_cast_int_to_float(a): return float(a)`.
fn bhimpl_cast_int_to_float(a: i64) -> f64 {
    a as f64
}

/// blackhole.py `bhimpl_cast_float_to_singlefloat(a): return
/// singlefloat2int(r_singlefloat(a))`.
fn bhimpl_cast_float_to_singlefloat(a: f64) -> i64 {
    // `longlong.singlefloat2int`: `intmask` of the 32-bit pattern.
    let bits = (a as f32).to_bits();
    if cfg!(target_pointer_width = "32") {
        bits as i32 as i64
    } else {
        bits as i64
    }
}

/// blackhole.py `bhimpl_cast_singlefloat_to_float(a): return
/// getfloatstorage(float(int2singlefloat(a)))`.
fn bhimpl_cast_singlefloat_to_float(a: i64) -> f64 {
    f32::from_bits(a as u32) as f64
}

// Generate handler fns from bhimpl methods via macros.
// @arguments("i", returns="i") → argcodes "i>i" → 1 src reg + 1 dst reg = 2 bytes
bhhandler_i_i!(handler_int_same_as, bhimpl_int_same_as);
bhhandler_i_i!(handler_int_neg, bhimpl_int_neg);
bhhandler_i_i!(handler_int_invert, bhimpl_int_invert);
bhhandler_i_i!(handler_int_is_true, bhimpl_int_is_true);
bhhandler_i_i!(handler_int_is_zero, bhimpl_int_is_zero);
bhhandler_i_i!(handler_int_force_ge_zero, bhimpl_int_force_ge_zero);

// @arguments("i", "i", returns="i") → argcodes "ii>i" → 2 src regs + 1 dst reg = 3 bytes
bhhandler_ii_i!(handler_int_add, bhimpl_int_add);
bhhandler_ii_i!(handler_int_sub, bhimpl_int_sub);

// pyre-only: compound assignment `a += b` lowered to the BinOp name
// `"add_assign"`. Under SSA the compound assignment becomes two reads +
// one write, so the bytecode shape is identical to `int_add/ii>i` — same
// primitive. Has no RPython analog; canonical RPython lowers `+=` to plain
// BINARY_ADD before reaching jtransform.
bhhandler_ii_i!(handler_int_add_assign, bhimpl_int_add);
bhhandler_ii_i!(handler_int_sub_assign, bhimpl_int_sub);

// pyre-only: dereference `*x` lowered to the UnaryOp name `"deref"`.
// After rtyper lowering the &i64 input has already been resolved to a
// plain i64 value, so the operation degenerates to a copy at dispatch
// time. Same primitive as `int_same_as`.
bhhandler_i_i!(handler_int_deref, bhimpl_int_same_as);

// pyre-only: `OpKind::Abort` placeholder emitted by the front-end when a
// syntax node has no dedicated OpKind variant (assembler.rs
// `OpKind::Abort { .. } => "abort".into()`). Carries zero operand
// bytes (empty argcodes). Reaching here at runtime means a graph that
// made it through rtyper still has an untranslatable op — abort the
// frame rather than continue dispatching past it: subsequent bytes
// would be misread as opcodes. Pyre marker semantics are
// `aborted = true` + `LeaveFrame`. RPython has no
// direct analog: its codewriter raises before lowering, so a jitcode
// never carries an unrecognized op.
fn handler_abort_marker(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.aborted = true;
    Err(DispatchError::LeaveFrame)
}

/// Handler for pyre-only `abort/>i`, emitted when the unsupported operation
/// still owns an SSA result.  The trailing destination byte does not make the
/// abort recoverable: continuing would expose the destination's stale value.
/// Match [`handler_abort_marker`] and leave the frame immediately.
fn handler_abort_result_marker_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    handler_abort_marker(bh, code, position)
}

/// Register-slot layout for state-field JIT blackhole resume.
///
/// Mirrors `live_slots_for_state_field_jit` (`jitcode/assembler.rs`): the
/// blackhole int register file holds, in flat order,
/// `[scalars 0..num_scalars | ptr then len per plain [int] |
/// the virtualizable identity]`, seeded by the resume reader via `setarg_i`
/// (`resume.rs _prepare_next_section` → `blackhole.py setarg_i`).
///
/// majit's `state_field` opcodes carry a section-relative logical index
/// (scalar `field_idx`, array/varray `array_idx`) rather than a flat register
/// slot. RPython's codewriter assigns register indices directly, so this
/// mapping is the majit-port adaptation recovering the flat slot a handler
/// must read/write. Each plain `[int]` owns two int identity slots immediately
/// after the scalars (`jtransform.py` `rewrite_op_getarrayitem` raw arm: the
/// Vec / slice element buffer is a raw `{ptr, len}` view). The virtualizable,
/// when the state has one, contributes exactly ONE slot — its identity —
/// however many `[.. ; virt]` arrays it declares. `virtualizable.py`
/// `get_array_length` reads each array's length off the live object, so a
/// length is never a box, and `virtualizable.py` `load_list_of_boxes` names
/// the virtualizable once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateFieldLayout {
    pub num_scalars: usize,
    /// Count of plain `[int]` arrays. Each owns two int identity slots
    /// (items pointer then length) right after the scalar prefix.
    pub num_plain_int_arrays: usize,
    /// `1` when the state declares any `[.. ; virt]` array (or an explicit
    /// virtualizable), else `0`. `pyjitpl.py` `reached_loop_header` carries the
    /// virtualizable exactly once — it is one red, `warmspot.py`
    /// `jd.index_of_virtualizable = jitdriver.reds.index(vname)` — so this is
    /// a presence flag, not a per-array count.
    pub num_vable_identity_slots: usize,
    /// Ref-typed scalar state fields. They live in the SEPARATE ref
    /// register bank at `registers_r[ref_scalar_base ..
    /// ref_scalar_base + num_ref_scalars]`, seeded by the resume reader
    /// in ref-fail-arg order, so they do NOT consume any `total_slots()`
    /// int slot.
    pub num_ref_scalars: usize,
    /// First ref-bank register of the ref-scalar identity slots. The
    /// dispatch JitCode's ref-bank ARGUMENTS occupy the registers below
    /// it (`MIFrame::setup_call` packs args densely from r0: `program`
    /// at r0, the virtualizable identity at r1 when present), and the
    /// blackhole re-executes ops that read those argument registers, so
    /// the identity slots cannot alias them.
    pub ref_scalar_base: usize,
    /// Float-typed scalar state fields. They live in the SEPARATE float
    /// register bank at `registers_f[float_scalar_base ..
    /// float_scalar_base + num_float_scalars]`, carrying raw f64 bits.
    pub num_float_scalars: usize,
    /// First float-bank register of the float-scalar identity slots.
    pub float_scalar_base: usize,
    /// First int-bank register of the scalar/array identity slots —
    /// the int-bank mirror of `ref_scalar_base`. The dispatch JitCode's
    /// int argument (`pc` at i0) sits below it; an identity slot
    /// aliasing i0 would overwrite the pc register, so the resume
    /// stream would encode the state scalar where the re-executed
    /// jit_merge_point op expects the green pc.
    pub int_scalar_base: usize,
}

impl StateFieldLayout {
    pub fn new(
        num_scalars: usize,
        num_plain_int_arrays: usize,
        num_vable_identity_slots: usize,
        int_scalar_base: usize,
    ) -> Self {
        Self {
            num_scalars,
            num_plain_int_arrays,
            num_vable_identity_slots,
            num_ref_scalars: 0,
            ref_scalar_base: 0,
            num_float_scalars: 0,
            float_scalar_base: 0,
            int_scalar_base,
        }
    }

    /// Like [`Self::new`] but with ref-typed scalar fields in the ref bank,
    /// starting at `ref_scalar_base`.
    pub fn with_ref_scalars(
        num_scalars: usize,
        num_plain_int_arrays: usize,
        num_vable_identity_slots: usize,
        num_ref_scalars: usize,
        ref_scalar_base: usize,
        int_scalar_base: usize,
    ) -> Self {
        Self {
            num_scalars,
            num_plain_int_arrays,
            num_vable_identity_slots,
            num_ref_scalars,
            ref_scalar_base,
            num_float_scalars: 0,
            float_scalar_base: 0,
            int_scalar_base,
        }
    }

    /// Add float-typed scalar fields in the float bank, starting at
    /// `float_scalar_base`. Kept as a setter so existing int/ref constructor
    /// call sites do not churn.
    pub fn with_float_scalars(
        mut self,
        num_float_scalars: usize,
        float_scalar_base: usize,
    ) -> Self {
        self.num_float_scalars = num_float_scalars;
        self.float_scalar_base = float_scalar_base;
        self
    }

    /// Total int identity slots: the real scalar count plus two words
    /// (ptr then len) per plain `[int]`. The vable identity is a Ref, not
    /// an int slot.
    pub fn total_slots(&self) -> usize {
        self.num_scalars + 2 * self.num_plain_int_arrays
    }

    /// Total live values `extract_live_values` produces: the int register
    /// slots ([`Self::total_slots`]) plus the vable identity (Ref) plus
    /// the ref-bank and float-bank scalars.
    pub fn total_live_values(&self) -> usize {
        self.total_slots()
            + self.num_vable_identity_slots
            + self.num_ref_scalars
            + self.num_float_scalars
    }

    /// Flat slot of scalar `field_idx` (scalars occupy
    /// `[int_scalar_base..int_scalar_base + num_scalars]`).
    pub fn scalar_slot(&self, field_idx: usize) -> usize {
        // `assert!`, not `debug_assert!`. With the all-zero default layout
        // every index is out of range and `int_scalar_base + field_idx` is
        // `0 + field_idx`, which addresses the dispatch JitCode's own
        // ARGUMENT registers — a wrong VALUE, in release, with nothing said.
        // That is how a missing `state_field_layout` in
        // `clone_context_from` stayed invisible.
        assert!(
            field_idx < self.num_scalars,
            "scalar field_idx {field_idx} out of range (num_scalars={}); an \
             empty layout here means the frame was never given the machine's \
             `state_field_layout`",
            self.num_scalars
        );
        self.int_scalar_base + field_idx
    }

    /// Ref register slot of ref-typed scalar `field_idx`. Ref scalars are
    /// densely packed in the ref register bank starting at
    /// `ref_scalar_base` (past the dispatch JitCode's ref-bank argument
    /// registers) in the same order the resume reader seeds the ref
    /// fail-args.
    pub fn ref_scalar_slot(&self, field_idx: usize) -> usize {
        assert!(
            field_idx < self.num_ref_scalars,
            "ref scalar field_idx {field_idx} out of range \
             (num_ref_scalars={}); see `scalar_slot`",
            self.num_ref_scalars
        );
        self.ref_scalar_base + field_idx
    }

    /// Float register slot of float-typed scalar `field_idx`. Float scalars
    /// are densely packed in the float register bank starting at
    /// `float_scalar_base`, in the same order the resume reader seeds the
    /// float fail-args.
    pub fn float_scalar_slot(&self, field_idx: usize) -> usize {
        assert!(
            field_idx < self.num_float_scalars,
            "float scalar field_idx {field_idx} out of range \
             (num_float_scalars={}); see `scalar_slot`",
            self.num_float_scalars
        );
        self.float_scalar_base + field_idx
    }

    /// Int-bank slots of plain `[int]` array `k`: items pointer then length.
    pub fn array_identity_slots(&self, k: usize) -> (usize, usize) {
        assert!(
            k < self.num_plain_int_arrays,
            "plain [int] array {k} out of range (num_plain_int_arrays={}); \
             an empty layout here means the frame was never given the \
             machine's `state_field_layout`",
            self.num_plain_int_arrays
        );
        let ptr = self.int_scalar_base + self.num_scalars + 2 * k;
        (ptr, ptr + 1)
    }

    /// Ref-bank register of the virtualizable identity, the portal ref
    /// argument immediately before the ref-scalar identity prefix.
    /// `None` when the state has no virtualizable.
    pub fn vable_identity_ref_slot(&self) -> Option<usize> {
        if self.num_vable_identity_slots == 0 {
            return None;
        }
        // The identity is the portal ref argument immediately before the
        // ref-scalar prefix. `program` occupies r0, so the identity is r1
        // and `ref_scalar_base` is at least 2. `saturating_sub(1)` on an
        // unset base 0 would alias r0.
        assert!(
            self.ref_scalar_base >= 2,
            "vable identity requires ref_scalar_base >= 2 (program at r0); \
             got {}",
            self.ref_scalar_base
        );
        Some(self.ref_scalar_base - 1)
    }

    /// Former int-bank identity slot. Always `None`: the identity is a Ref.
    pub fn vable_identity_slot(&self) -> Option<usize> {
        None
    }
}

// pyre-only: `state_field` family. RPython has no opcode counterpart — the
// `#[jit_interp]` `jitcode_lower` macro emits these to read/write the
// Rust-port `state_fields`, which ARE the jitdriver reds.  PyPy reds are the
// blackhole frame's int registers (`blackhole.py:300-302`), seeded by the
// resume reader via `setarg_i` (`blackhole.py`) and read/written by the
// register-addressed dispatch opcodes (`blackhole.py/223`).  These
// handlers move between the canonical red register slots: `field_idx` /
// `array_idx` are section-relative logical indices that
// `StateFieldLayout` maps to the flat slot the seed populated.

/// `load_state_field/di` — `registers_i[dest] = registers_i[slot(field_idx)]`.
/// Encoding: 1× u16 `field_idx` + 1× u8 dest register = 3 bytes.
fn portal_registers_i(bh: &BlackholeInterpreter) -> &[i64] {
    let mut root = bh;
    while let Some(ref parent) = root.nextblackholeinterp {
        root = parent;
    }
    &root.registers_i
}

fn portal_registers_i_mut(bh: &mut BlackholeInterpreter) -> &mut [i64] {
    let mut root: *mut BlackholeInterpreter = bh;
    unsafe {
        while let Some(ref mut parent) = (*root).nextblackholeinterp {
            root = parent.as_mut();
        }
        (*root).registers_i.as_mut_slice()
    }
}

fn portal_registers_r(bh: &BlackholeInterpreter) -> &[i64] {
    let mut root = bh;
    while let Some(ref parent) = root.nextblackholeinterp {
        root = parent;
    }
    &root.registers_r
}

fn portal_registers_r_mut(bh: &mut BlackholeInterpreter) -> &mut [i64] {
    let mut root: *mut BlackholeInterpreter = bh;
    unsafe {
        while let Some(ref mut parent) = (*root).nextblackholeinterp {
            root = parent.as_mut();
        }
        (*root).registers_r.as_mut_slice()
    }
}

fn portal_registers_f(bh: &BlackholeInterpreter) -> &[i64] {
    let mut root = bh;
    while let Some(ref parent) = root.nextblackholeinterp {
        root = parent;
    }
    &root.registers_f
}

fn portal_registers_f_mut(bh: &mut BlackholeInterpreter) -> &mut [i64] {
    let mut root: *mut BlackholeInterpreter = bh;
    unsafe {
        while let Some(ref mut parent) = (*root).nextblackholeinterp {
            root = parent.as_mut();
        }
        (*root).registers_f.as_mut_slice()
    }
}

fn handler_load_state_field_di(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let dest = code[position + 2] as usize;
    let slot = bh.state_field_layout.scalar_slot(field_idx);
    let value = portal_registers_i(bh)[slot];
    bh.registers_i[dest] = value;
    Ok(position + 3)
}

/// `store_state_field/di` — `registers_i[slot(field_idx)] = registers_i[src]`.
/// Encoding: 1× u16 `field_idx` + 1× u8 src register = 3 bytes.
fn handler_store_state_field_di(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let src = code[position + 2] as usize;
    let slot = bh.state_field_layout.scalar_slot(field_idx);
    let value = bh.registers_i[src];
    portal_registers_i_mut(bh)[slot] = value;
    Ok(position + 3)
}

/// `load_state_field_ref/dr` — `registers_r[dest] = registers_r[ref_slot(field_idx)]`.
/// The ref-typed scalar state field lives in the ref register bank; the
/// resume reader seeded it from the ref fail-args. Mirror of
/// `handler_load_state_field_di` on the ref bank.
/// Encoding: 1× u16 `field_idx` + 1× u8 dest ref register = 3 bytes.
fn handler_load_state_field_ref_dr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let dest = code[position + 2] as usize;
    let slot = bh.state_field_layout.ref_scalar_slot(field_idx);
    let value = portal_registers_r(bh)[slot];
    bh.registers_r[dest] = value;
    Ok(position + 3)
}

/// `store_state_field_ref/dr` — `registers_r[ref_slot(field_idx)] = registers_r[src]`.
/// Encoding: 1× u16 `field_idx` + 1× u8 src ref register = 3 bytes.
fn handler_store_state_field_ref_dr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let src = code[position + 2] as usize;
    let slot = bh.state_field_layout.ref_scalar_slot(field_idx);
    let value = bh.registers_r[src];
    portal_registers_r_mut(bh)[slot] = value;
    Ok(position + 3)
}

/// `load_state_field_float/df` — `registers_f[dest] = registers_f[float_slot(field_idx)]`.
/// Encoding: 1× u16 `field_idx` + 1× u8 dest float register = 3 bytes.
fn handler_load_state_field_float_df(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let dest = code[position + 2] as usize;
    let slot = bh.state_field_layout.float_scalar_slot(field_idx);
    let value = portal_registers_f(bh)[slot];
    bh.registers_f[dest] = value;
    Ok(position + 3)
}

/// `store_state_field_float/df` — `registers_f[float_slot(field_idx)] = registers_f[src]`.
/// Encoding: 1× u16 `field_idx` + 1× u8 src float register = 3 bytes.
fn handler_store_state_field_float_df(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let field_idx = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let src = code[position + 2] as usize;
    let slot = bh.state_field_layout.float_scalar_slot(field_idx);
    let value = bh.registers_f[src];
    portal_registers_f_mut(bh)[slot] = value;
    Ok(position + 3)
}

/// Handler for `jit_merge_point/iIRFIRF` — `BC_JIT_MERGE_POINT`.
/// Forwards to `bhimpl_jit_merge_point` which uses `self.position`-mutating
/// helpers; `dispatch_step` already passes `self.position` as the
/// `position` argument so the local copy stays in sync.
fn handler_jit_merge_point_i(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_jit_merge_point(jitcode::insns::BC_JIT_MERGE_POINT)?;
    Ok(bh.position)
}

/// Handler for `jit_merge_point/cIRFIRF` — `BC_JIT_MERGE_POINT_C`
/// (assembler.py `USE_C_FORM`: jdindex inlined as a signed byte).
fn handler_jit_merge_point_c(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_jit_merge_point(jitcode::insns::BC_JIT_MERGE_POINT_C)?;
    Ok(bh.position)
}

/// Handler for `abort_permanent/` — pyre-only fail-path opcode that
/// forwards a TLS-stashed exception or aborts the frame. Carries no
/// operand bytes.
fn handler_abort_permanent(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_abort_permanent()?;
    Ok(bh.position)
}

/// Handler for pyre-only `abort/>r` — the Ref-result counterpart of
/// [`handler_abort_result_marker_i`].
fn handler_abort_result_marker_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    handler_abort_marker(bh, code, position)
}
bhhandler_ii_i!(handler_int_mul, bhimpl_int_mul);
// `int_floordiv` / `int_mod` are NOT registered as bytecode handlers:
// `jtransform.py` rewrites both via `_do_builtin_call` to
// `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)` before
// jitcode emission, so RPython's `blackhole.py` has no
// `bhimpl_int_floordiv` / `bhimpl_int_mod`.  pyre keeps the helper
// functions ([`ll_int_py_div`] / [`ll_int_py_mod`] above) for the
// translate-side residual call (`codegen.rs` emits them as
// `CallI` with `INT_PY_DIV_EFFECT_INFO` / `INT_PY_MOD_EFFECT_INFO`)
// but does not wire them into the bytecode dispatch table.
bhhandler_ii_i!(handler_int_and, bhimpl_int_and);
bhhandler_ii_i!(handler_int_or, bhimpl_int_or);
bhhandler_ii_i!(handler_int_xor, bhimpl_int_xor);
bhhandler_ii_i!(handler_int_rshift, bhimpl_int_rshift);
bhhandler_ii_i!(handler_int_lshift, bhimpl_int_lshift);
bhhandler_ii_i!(handler_uint_rshift, bhimpl_uint_rshift);
bhhandler_ii_i!(handler_int_lt, bhimpl_int_lt);
bhhandler_ii_i!(handler_int_le, bhimpl_int_le);
bhhandler_ii_i!(handler_int_eq, bhimpl_int_eq);
bhhandler_ii_i!(handler_int_ne, bhimpl_int_ne);
bhhandler_ii_i!(handler_int_gt, bhimpl_int_gt);
bhhandler_ii_i!(handler_int_ge, bhimpl_int_ge);

// assembler.py `USE_C_FORM`: every Constant operand of these canonical
// integer operations takes the inline signed-byte form independently.
bhhandler_short_binop_i!(
    handler_int_add_ci,
    handler_int_add_ic,
    handler_int_add_cc,
    bhimpl_int_add
);
bhhandler_short_binop_i!(
    handler_int_sub_ci,
    handler_int_sub_ic,
    handler_int_sub_cc,
    bhimpl_int_sub
);
bhhandler_short_binop_i!(
    handler_int_and_ci,
    handler_int_and_ic,
    handler_int_and_cc,
    bhimpl_int_and
);
bhhandler_short_binop_i!(
    handler_int_eq_ci,
    handler_int_eq_ic,
    handler_int_eq_cc,
    bhimpl_int_eq
);
bhhandler_short_binop_i!(
    handler_int_ne_ci,
    handler_int_ne_ic,
    handler_int_ne_cc,
    bhimpl_int_ne
);
bhhandler_short_binop_i!(
    handler_int_lt_ci,
    handler_int_lt_ic,
    handler_int_lt_cc,
    bhimpl_int_lt
);
bhhandler_short_binop_i!(
    handler_int_le_ci,
    handler_int_le_ic,
    handler_int_le_cc,
    bhimpl_int_le
);
bhhandler_short_binop_i!(
    handler_int_gt_ci,
    handler_int_gt_ic,
    handler_int_gt_cc,
    bhimpl_int_gt
);
bhhandler_short_binop_i!(
    handler_int_ge_ci,
    handler_int_ge_ic,
    handler_int_ge_cc,
    bhimpl_int_ge
);

// blackhole.py `bhimpl_int_copy(a): return a` — @arguments("i", returns="i").
// Decoded as `i>i` (same as int_same_as). Already have handler_int_same_as.
// Wire as alias.
// Control flow and copy handlers
bhhandler_i_i!(handler_int_copy, bhimpl_int_same_as);

// `int_copy/c>i` — `c`-argcode source: `int_copy` is in USE_C_FORM
// (`assembler.py:312`), so a small ConstInt is one inline signed byte
// (`assembler.py:99-107` short branch) read via `signedord`
// (`blackhole.py`) instead of a `registers_i` pool slot. `dst` is the
// next byte. `bhimpl_int_copy` is the identity, so the byte value is the
// stored result.
fn handler_int_copy_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = code[position] as i8 as i64;
    bh.registers_i[code[position + 1] as usize] = a;
    Ok(position + 2)
}

/// blackhole.py `bhimpl_ref_copy(a): return a` — @arguments("r", returns="r").
fn bhimpl_ref_copy(a: i64) -> i64 {
    a
}

/// blackhole.py `bhimpl_float_copy(a): return a` — @arguments("f", returns="f").
fn bhimpl_float_copy(a: f64) -> f64 {
    a
}

/// blackhole.py `bhimpl_ref_isconstant(x): return False`.
fn bhimpl_ref_isconstant(_a: i64) -> i64 {
    0
}

/// blackhole.py `bhimpl_ref_isvirtual(x): return False`.
fn bhimpl_ref_isvirtual(_a: i64) -> i64 {
    0
}

/// blackhole.py `bhimpl_assert_not_none(a): assert a`.
fn bhimpl_assert_not_none(a: i64) {
    assert!(a != 0, "bhimpl_assert_not_none: ref register is null");
}

/// blackhole.py `bhimpl_record_exact_class(a, b): pass`.
fn bhimpl_record_exact_class(_a: i64, _b: i64) {}

/// blackhole.py `bhimpl_record_exact_value_r(a, b): pass`.
fn bhimpl_record_exact_value_r(_a: i64, _b: i64) {}

/// blackhole.py `bhimpl_record_exact_value_i(a, b): pass`.
fn bhimpl_record_exact_value_i(_a: i64, _b: i64) {}

/// blackhole.py `bhimpl_loop_header(jdindex): pass`.
fn bhimpl_loop_header(_jdindex: i64) {}

/// blackhole.py `bhimpl_int_assert_green(x): pass`.
fn bhimpl_int_assert_green(_a: i64) {}

/// `bhimpl_ref_assert_green(x): pass`.
fn bhimpl_ref_assert_green(_a: i64) {}

/// `bhimpl_float_assert_green(x): pass`.
fn bhimpl_float_assert_green(_a: f64) {}

/// blackhole.py `bhimpl_int_guard_value(a): pass`.
fn bhimpl_int_guard_value(_a: i64) {}

/// blackhole.py `bhimpl_ref_guard_value(a): pass`.
fn bhimpl_ref_guard_value(_a: i64) {}

/// blackhole.py `bhimpl_float_guard_value(a): pass`.
fn bhimpl_float_guard_value(_a: f64) {}

/// blackhole.py `bhimpl_virtual_ref(a): return a`.
fn bhimpl_virtual_ref(a: i64) -> i64 {
    a
}

/// blackhole.py `bhimpl_virtual_ref_finish(a): pass`.
fn bhimpl_virtual_ref_finish(_a: i64) {}

/// blackhole.py `bhimpl_unreachable(): raise AssertionError("unreachable")`.
fn bhimpl_unreachable() -> ! {
    panic!("bhimpl_unreachable reached")
}

/// blackhole.py `bhimpl_int_push(self, a): self.tmpreg_i = a`.
fn bhimpl_int_push(bh: &mut BlackholeInterpreter, a: i64) {
    bh.tmpreg_i = a;
}

/// blackhole.py `bhimpl_ref_push(self, a): self.tmpreg_r = a`.
fn bhimpl_ref_push(bh: &mut BlackholeInterpreter, a: i64) {
    bh.tmpreg_r = a;
}

/// blackhole.py `bhimpl_float_push(self, a): self.tmpreg_f = a`.
/// `tmpreg_f` stores floatstorage bits; convert real f64 → bits.
fn bhimpl_float_push(bh: &mut BlackholeInterpreter, a: f64) {
    bh.tmpreg_f = a.to_bits() as i64;
}

/// blackhole.py `bhimpl_int_pop(self): return self.get_tmpreg_i()`.
fn bhimpl_int_pop(bh: &mut BlackholeInterpreter) -> i64 {
    bh.tmpreg_i
}

/// blackhole.py `bhimpl_ref_pop(self): return self.get_tmpreg_r()`.
fn bhimpl_ref_pop(bh: &mut BlackholeInterpreter) -> i64 {
    bh.get_tmpreg_r()
}

/// blackhole.py `bhimpl_float_pop(self): return self.get_tmpreg_f()`.
/// `tmpreg_f` stores floatstorage bits; convert bits → real f64.
fn bhimpl_float_pop(bh: &mut BlackholeInterpreter) -> f64 {
    f64::from_bits(bh.tmpreg_f as u64)
}

/// blackhole.py `bhimpl_jit_enter_portal_frame(x): pass`.
fn bhimpl_jit_enter_portal_frame(_x: i64) {}

/// blackhole.py `bhimpl_jit_leave_portal_frame(): pass`.
fn bhimpl_jit_leave_portal_frame() {}

/// blackhole.py `bhimpl_hint_force_virtualizable(r): pass`.
fn bhimpl_hint_force_virtualizable(_r: i64) {}

/// `warmspot.py` `jd.handle_jitexc_from_bh`.
///
/// Defined beside [`BhJitDriverSd`], which stores one hook per driver.
pub type PortalRunnerHook =
    fn(&crate::jitexc::JitException) -> Result<(BhReturnType, i64), PortalRunnerFailure>;

/// `blackhole.py` `_handle_jitexception_in_portal`:
///
/// ```python
/// for jd in self.builder.metainterp_sd.jitdrivers_sd:
///     if jd.mainjitcode is self.jitcode:
///         break
/// else:
///     assert 0, "portal jitcode not found??"
/// jd.handle_jitexc_from_bh(self.nextblackholeinterp, e)
/// ```
///
/// `bh` is the portal frame that raised, i.e. upstream's `self`.
///
/// The relation that loop walks is established from the other end by
/// `CallControl.find_all_graphs`'s `jd.mainjitcode.jitdriver_sd = jd`, so
/// `jd.mainjitcode is
/// self.jitcode` and `self.jitcode.jitdriver_sd is jd` answer the same
/// question.  Reading the stamp is also the form `_run_forever`'s walk already
/// uses to decide where to stop (`jitcode.jitdriver_sd().is_none()`), so asking
/// it here leaves one mechanism where two could disagree: `resolve_jitcode`
/// materializes a fresh `Arc` per resolution on the novable path, and an
/// `Arc::ptr_eq` against `mainjitcode` misses such a frame even after the walk
/// accepted it as a portal.
///
/// A miss returns `None`.  Upstream's `assert 0, "portal jitcode not found??"`
/// is unreachable for the walk's own caller, which stops only on a stamped
/// frame; the entry frame [`run_forever_with_portal`] is handed has not passed
/// that walk, and neither has the guard-failure loop in `call_jit.rs`.
fn portal_jd_for(bh: &BlackholeInterpreter) -> Option<usize> {
    bh.jitcode.jitdriver_sd()
}

/// The `jd.handle_jitexc_from_bh` and the `result_kind` a recursive portal
/// level uses, both read off `jitdrivers_sd[jdindex]`.
///
/// `blackhole.py` indexes `self.builder.metainterp_sd.jitdrivers_sd[jdindex]`
/// directly, so a stamped index outside the table is an `IndexError`, not a
/// driverless frame. A frame whose jitcode carries no driver stamp is not a
/// portal and gets neither.
fn portal_dispatch_for(
    bh: &BlackholeInterpreter,
) -> (Option<PortalRunnerHook>, Option<BhReturnType>) {
    let Some(jd_index) = portal_jd_for(bh) else {
        return (None, None);
    };
    // A codewriter `setup_jitdriver` for a plain function stamps a private
    // index that is not a runtime `jitdrivers_sd` slot. That frame is not a
    // portal (`call.py` stamps only `jd.mainjitcode`). An index the table
    // does not hold has no `handle_jitexc_from_bh`.
    let Some(jd) = bh.jitdrivers_sd.get(jd_index) else {
        return (None, None);
    };
    (jd.handle_jitexc_from_bh, Some(jd.result_type))
}

/// Handler for `live/` — `bhimpl_live` (`blackhole.py`): record the marker's
/// jitcode pc and skip `OFFSET_SIZE` bytes. Precise Ref liveness is applied
/// at the next collection (`walk_bh_regs`), not here.
fn handler_live(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // `position` is past the opcode byte; the marker op starts one earlier.
    bh.last_live_pc = position - 1;
    Ok(position + majit_jitcode::liveness::OFFSET_SIZE)
}

/// Handler for `goto/L` — unconditional jump. Argcodes: `L` (2-byte label).
/// RPython blackhole.py: `def bhimpl_goto(target): return target`.
fn handler_goto(
    _bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    Ok(bhimpl_goto(target))
}

// `handler_goto_if_not` extracted via `bhhandler_goto_if_not_i!` macro below
// after the macro definition site so the macro is in scope.

/// Handler for `int_return/i` — RPython blackhole.py:841-845.
/// @arguments("self", "i"): read one int register, store in tmpreg_i,
/// raise LeaveFrame.
fn handler_int_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    bh.tmpreg_i = a;
    bh.return_type = BhReturnType::Int;
    // RPython blackhole.py `_get_method` stores the decoded position
    // back into `self.position` before invoking the bhimpl_*; this is
    // visible after a LeaveFrame since the frame teardown reads the
    // post-operand position.
    bh.position = position + 1;
    Err(DispatchError::LeaveFrame)
}

/// Handler for `int_return/c` — USE_C_FORM short source (`assembler.py`):
/// `int_return` reads its value from one inline signed byte (`signedord`,
/// `blackhole.py`) rather than a `registers_i` slot.  Identical frame
/// teardown to `handler_int_return`.
fn handler_int_return_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = code[position] as i8 as i64;
    bh.tmpreg_i = a;
    bh.return_type = BhReturnType::Int;
    bh.position = position + 1; // blackhole.py:169 parity (see int_return).
    Err(DispatchError::LeaveFrame)
}

/// Handler for `ref_return/r` — RPython blackhole.py:847-851.
fn handler_ref_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    bh.tmpreg_r = a;
    bh.return_type = BhReturnType::Ref;
    bh.position = position + 1; // blackhole.py:169 parity (see int_return).
    Err(DispatchError::LeaveFrame)
}

/// Handler for `void_return/` — RPython blackhole.py:859-862.
fn handler_void_return(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.return_type = BhReturnType::Void;
    Err(DispatchError::LeaveFrame)
}

// RPython stores floats as longlong (i64 bits). pyre stores f64 in
// registers_f directly. The bhimpl methods work on f64 values.

// Float operations (`blackhole.py:676-808`)
fn bhimpl_float_neg(a: f64) -> f64 {
    -a
}
fn bhimpl_float_abs(a: f64) -> f64 {
    a.abs()
}
fn bhimpl_float_add(a: f64, b: f64) -> f64 {
    a + b
}
fn bhimpl_float_sub(a: f64, b: f64) -> f64 {
    a - b
}
fn bhimpl_float_mul(a: f64, b: f64) -> f64 {
    a * b
}
fn bhimpl_float_truediv(a: f64, b: f64) -> f64 {
    a / b
}

/// Decode pattern `@arguments("f", "f", returns="f")` — argcodes `"ff>f"`.
macro_rules! bhhandler_ff_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_f[code[position + 2] as usize] = $bhimpl(a, b).to_bits() as i64;
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("f", returns="f")` — argcodes `"f>f"`.
macro_rules! bhhandler_f_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            bh.registers_f[code[position + 1] as usize] = $bhimpl(a).to_bits() as i64;
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("f", "f", returns="i")` — argcodes `"ff>i"`.
macro_rules! bhhandler_ff_i {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_i[code[position + 2] as usize] = $cmp(a, b) as i64;
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("f", returns="i")` — argcodes `"f>i"`.
macro_rules! bhhandler_f_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("i", returns="f")` — argcodes `"i>f"`.
macro_rules! bhhandler_i_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            bh.registers_f[code[position + 1] as usize] = $bhimpl(a).to_bits() as i64;
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("r", returns="r")` — argcodes `"r>r"`.
macro_rules! bhhandler_r_r {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            bh.registers_r[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("r", "i")` (no return) — argcodes `"ri"`.
macro_rules! bhhandler_ri_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            $bhimpl(a, b);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("r", "r")` (no return) — argcodes `"rr"`.
macro_rules! bhhandler_rr_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_r[code[position + 1] as usize];
            $bhimpl(a, b);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("i", "i")` (no return) — argcodes `"ii"`.
macro_rules! bhhandler_ii_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            $bhimpl(a, b);
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("r")` (no return) — argcodes `"r"`.
macro_rules! bhhandler_r_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            $bhimpl(a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("i")` (no return) — argcodes `"i"`.
macro_rules! bhhandler_i_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            $bhimpl(a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("f")` (no return) — argcodes `"f"`.
macro_rules! bhhandler_f_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            $bhimpl(a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments()` (no operands, no return) — empty argcodes.
macro_rules! bhhandler_v_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            _bh: &mut BlackholeInterpreter,
            _code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            $bhimpl();
            Ok(position)
        }
    };
}

/// Decode pattern `@arguments("self", "i")` — 1 int read, no return, takes bh.
macro_rules! bhhandler_self_i_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            $bhimpl(bh, a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("self", "r")` — 1 ref read, no return, takes bh.
macro_rules! bhhandler_self_r_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            $bhimpl(bh, a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("self", "f")` — 1 float read, no return, takes bh.
macro_rules! bhhandler_self_f_v {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            $bhimpl(bh, a);
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("self", returns="i")` — no read, 1 int write, takes bh.
macro_rules! bhhandler_self_v_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let result = $bhimpl(bh);
            bh.registers_i[code[position] as usize] = result;
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("self", returns="r")` — no read, 1 ref write, takes bh.
macro_rules! bhhandler_self_v_r {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let result = $bhimpl(bh);
            bh.registers_r[code[position] as usize] = result;
            Ok(position + 1)
        }
    };
}

/// Decode pattern `@arguments("self", returns="f")` — no read, 1 float write, takes bh.
macro_rules! bhhandler_self_v_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let result: f64 = $bhimpl(bh);
            bh.registers_f[code[position] as usize] = result.to_bits() as i64;
            Ok(position + 1)
        }
    };
}

bhhandler_ff_f!(handler_float_add, bhimpl_float_add);
bhhandler_ff_f!(handler_float_sub, bhimpl_float_sub);
bhhandler_ff_f!(handler_float_mul, bhimpl_float_mul);
bhhandler_ff_f!(handler_float_truediv, bhimpl_float_truediv);
bhhandler_f_f!(handler_float_neg, bhimpl_float_neg);
bhhandler_f_f!(handler_float_abs, bhimpl_float_abs);
bhhandler_ff_i!(handler_float_lt, |a: f64, b: f64| a < b);
bhhandler_ff_i!(handler_float_le, |a: f64, b: f64| a <= b);
bhhandler_ff_i!(handler_float_eq, |a: f64, b: f64| a == b);
bhhandler_ff_i!(handler_float_ne, |a: f64, b: f64| a != b);
bhhandler_ff_i!(handler_float_gt, |a: f64, b: f64| a > b);
bhhandler_ff_i!(handler_float_ge, |a: f64, b: f64| a >= b);

// Unsigned comparisons (`blackhole.py:571-582`)
fn bhimpl_uint_lt(a: i64, b: i64) -> i64 {
    ((a as u64) < (b as u64)) as i64
}
fn bhimpl_uint_le(a: i64, b: i64) -> i64 {
    ((a as u64) <= (b as u64)) as i64
}
fn bhimpl_uint_gt(a: i64, b: i64) -> i64 {
    ((a as u64) > (b as u64)) as i64
}
fn bhimpl_uint_ge(a: i64, b: i64) -> i64 {
    ((a as u64) >= (b as u64)) as i64
}

/// `blackhole.py bhimpl_uint_mul_high` — high 64 bits of unsigned
/// 128-bit product.  Extracted from the inline body of
/// `handler_uint_mul_high` so the bhhandler macro can target a named
/// bhimpl uniformly with the rest of the binop family.
fn bhimpl_uint_mul_high(a: i64, b: i64) -> i64 {
    let a = a as u64 as u128;
    let b = b as u64 as u128;
    ((a * b) >> 64) as i64
}

bhhandler_ii_i!(handler_uint_lt, bhimpl_uint_lt);
bhhandler_ii_i!(handler_uint_le, bhimpl_uint_le);
bhhandler_ii_i!(handler_uint_gt, bhimpl_uint_gt);
bhhandler_ii_i!(handler_uint_ge, bhimpl_uint_ge);

// ── goto_if_not_int_* conditionals (RPython blackhole.py:871-920) ───

/// Decode pattern `@arguments("i", "i", "L", "pc", returns="L")`.
/// Read 2 int registers + 2-byte label; compare; return target or pc.
macro_rules! bhhandler_goto_if_not_ii {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}

macro_rules! bhhandler_goto_if_not_short {
    ($ci:ident, $ic:ident, $cc:ident, $cmp:expr) => {
        fn $ci(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = code[position] as i8 as i64;
            let b = bh.registers_i[code[position + 1] as usize];
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }

        fn $ic(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = code[position + 1] as i8 as i64;
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }

        fn $cc(
            _bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = code[position] as i8 as i64;
            let b = code[position + 1] as i8 as i64;
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}

bhhandler_goto_if_not_ii!(handler_goto_if_not_int_lt, |a: i64, b: i64| a < b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_le, |a: i64, b: i64| a <= b);
// Temporarily expanded from `bhhandler_goto_if_not_ii!` for MAJIT_BH_DEBUG
// cond inspection (#210).
fn handler_goto_if_not_int_eq(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let b = bh.registers_i[code[position + 1] as usize];
    let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
    let pc = position + 4;
    if crate::bh_debug_enabled() {
        eprintln!(
            "[bh-brcond] pos={} int_eq i{}={} i{}={} target={} fallthrough={}",
            position - 1,
            code[position],
            a,
            code[position + 1],
            b,
            target,
            pc
        );
    }
    if a == b { Ok(pc) } else { Ok(target) }
}
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ne, |a: i64, b: i64| a != b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_gt, |a: i64, b: i64| a > b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ge, |a: i64, b: i64| a >= b);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_lt_ci,
    handler_goto_if_not_int_lt_ic,
    handler_goto_if_not_int_lt_cc,
    |a: i64, b: i64| a < b
);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_le_ci,
    handler_goto_if_not_int_le_ic,
    handler_goto_if_not_int_le_cc,
    |a: i64, b: i64| a <= b
);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_eq_ci,
    handler_goto_if_not_int_eq_ic,
    handler_goto_if_not_int_eq_cc,
    |a: i64, b: i64| a == b
);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_ne_ci,
    handler_goto_if_not_int_ne_ic,
    handler_goto_if_not_int_ne_cc,
    |a: i64, b: i64| a != b
);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_gt_ci,
    handler_goto_if_not_int_gt_ic,
    handler_goto_if_not_int_gt_cc,
    |a: i64, b: i64| a > b
);
bhhandler_goto_if_not_short!(
    handler_goto_if_not_int_ge_ci,
    handler_goto_if_not_int_ge_ic,
    handler_goto_if_not_int_ge_cc,
    |a: i64, b: i64| a >= b
);

/// Decode pattern `@arguments("i", "L", "pc", returns="L")` — 1 int read +
/// 2-byte label target; bhimpl chooses target or fall-through pc.
macro_rules! bhhandler_goto_if_not_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
            let pc = position + 3;
            Ok($bhimpl(a, target, pc))
        }
    };
}

/// Decode pattern `@arguments("r", "L", "pc", returns="L")` — 1 ref read +
/// 2-byte label target; bhimpl chooses target or fall-through pc.
macro_rules! bhhandler_goto_if_not_r {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
            let pc = position + 3;
            Ok($bhimpl(a, target, pc))
        }
    };
}

/// Decode pattern `@arguments("r", "r", "L", "pc", returns="L")` — 2 ref reads +
/// 2-byte label target; bhimpl chooses target or fall-through pc.
macro_rules! bhhandler_goto_if_not_rr {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_r[code[position + 1] as usize];
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            Ok($bhimpl(a, b, target, pc))
        }
    };
}

/// blackhole.py `bhimpl_goto_if_not_int_is_zero(a, target, pc)`.
fn bhimpl_goto_if_not_int_is_zero(a: i64, target: usize, pc: usize) -> usize {
    if a == 0 { pc } else { target }
}

/// blackhole.py `bhimpl_goto_if_not_ptr_iszero(a, target, pc)`.
fn bhimpl_goto_if_not_ptr_iszero(a: i64, target: usize, pc: usize) -> usize {
    if a == 0 { pc } else { target }
}

/// blackhole.py `bhimpl_goto_if_not_ptr_nonzero(a, target, pc)`.
fn bhimpl_goto_if_not_ptr_nonzero(a: i64, target: usize, pc: usize) -> usize {
    if a != 0 { pc } else { target }
}

/// blackhole.py `bhimpl_goto_if_not_ptr_eq(a, b, target, pc)`.
fn bhimpl_goto_if_not_ptr_eq(a: i64, b: i64, target: usize, pc: usize) -> usize {
    if a == b { pc } else { target }
}

/// blackhole.py `bhimpl_goto_if_not_ptr_ne(a, b, target, pc)`.
fn bhimpl_goto_if_not_ptr_ne(a: i64, b: i64, target: usize, pc: usize) -> usize {
    if a != b { pc } else { target }
}

/// blackhole.py `bhimpl_goto_if_not(a, target, pc)`.
fn bhimpl_goto_if_not(a: i64, target: usize, pc: usize) -> usize {
    if a != 0 { pc } else { target }
}

/// blackhole.py `bhimpl_goto(target): return target`.
fn bhimpl_goto(target: usize) -> usize {
    target
}

// Reference operations (`blackhole.py:584-610`)
fn bhimpl_ptr_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}
fn bhimpl_ptr_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}
fn bhimpl_ptr_iszero(a: i64) -> i64 {
    (a == 0) as i64
}
fn bhimpl_ptr_nonzero(a: i64) -> i64 {
    (a != 0) as i64
}

/// `@arguments("r", "r", returns="i")` — rr>i: read 2 ref regs, result int.
macro_rules! bhhandler_rr_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_r[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// `@arguments("r", returns="i")` — r>i: read 1 ref reg, result int.
macro_rules! bhhandler_r_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

/// `@arguments("i", returns="r")` — i>r: read 1 int reg, result ref.
macro_rules! bhhandler_i_r {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            bh.registers_r[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

bhhandler_rr_i!(handler_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_ptr_ne, bhimpl_ptr_ne);
bhhandler_rr_i!(handler_instance_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_instance_ptr_ne, bhimpl_ptr_ne);
bhhandler_r_i!(handler_ptr_iszero, bhimpl_ptr_iszero);
bhhandler_r_i!(handler_ptr_nonzero, bhimpl_ptr_nonzero);

// ref/float copy (blackhole.py:641-645)
bhhandler_r_r!(handler_ref_copy, bhimpl_ref_copy);
bhhandler_f_f!(handler_float_copy, bhimpl_float_copy);

// float_return (blackhole.py:853-857)
fn handler_float_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_f = bh.registers_f[code[position] as usize];
    bh.return_type = BhReturnType::Float;
    bh.position = position + 1; // blackhole.py:169 parity (see int_return).
    Err(DispatchError::LeaveFrame)
}

// Guard-value no-ops (`blackhole.py:648-656`)
bhhandler_i_v!(handler_int_guard_value, bhimpl_int_guard_value);
bhhandler_r_v!(handler_ref_guard_value, bhimpl_ref_guard_value);
bhhandler_f_v!(handler_float_guard_value, bhimpl_float_guard_value);

// Push and pop (`blackhole.py:661-679`)
bhhandler_self_i_v!(handler_int_push, bhimpl_int_push);
bhhandler_self_r_v!(handler_ref_push, bhimpl_ref_push);
bhhandler_self_f_v!(handler_float_push, bhimpl_float_push);
bhhandler_self_v_i!(handler_int_pop, bhimpl_int_pop);
bhhandler_self_v_r!(handler_ref_pop, bhimpl_ref_pop);
bhhandler_self_v_f!(handler_float_pop, bhimpl_float_pop);

// Exact-class and exact-value recording no-ops (`blackhole.py:616-636`)
bhhandler_ri_v!(handler_record_exact_class, bhimpl_record_exact_class);
bhhandler_rr_v!(handler_record_exact_value_r, bhimpl_record_exact_value_r);
bhhandler_ii_v!(handler_record_exact_value_i, bhimpl_record_exact_value_i);

// Cast operations (`blackhole.py:800-831`)
bhhandler_f_i!(handler_cast_float_to_int, bhimpl_cast_float_to_int);
bhhandler_i_f!(handler_cast_int_to_float, bhimpl_cast_int_to_float);

// Integer sign extension (`blackhole.py:566-569`)
bhhandler_ii_i!(handler_int_signext, bhimpl_int_signext);

// Overflow operations (`blackhole.py:478-497`)
/// blackhole.py `bhimpl_int_add_jump_if_ovf(label, a, b)`.
/// On overflow: returns `(None, target)` so the handler jumps to label.
/// On success: returns `(Some(sum), pc)` so the handler stores sum at the
/// result register and falls through to pc.
fn bhimpl_int_add_jump_if_ovf(a: i64, b: i64, target: usize, pc: usize) -> (Option<i64>, usize) {
    match a.checked_add(b) {
        Some(r) => (Some(r), pc),
        None => (None, target),
    }
}

/// blackhole.py `bhimpl_int_sub_jump_if_ovf(label, a, b)`.
fn bhimpl_int_sub_jump_if_ovf(a: i64, b: i64, target: usize, pc: usize) -> (Option<i64>, usize) {
    match a.checked_sub(b) {
        Some(r) => (Some(r), pc),
        None => (None, target),
    }
}

/// blackhole.py `bhimpl_int_mul_jump_if_ovf(label, a, b)`.
fn bhimpl_int_mul_jump_if_ovf(a: i64, b: i64, target: usize, pc: usize) -> (Option<i64>, usize) {
    match a.checked_mul(b) {
        Some(r) => (Some(r), pc),
        None => (None, target),
    }
}

/// Decode pattern `@arguments("L", "i", "i", returns="iL")` — 2-byte label +
/// 2 int reads + 1 int write (only on no-overflow path). Total 5 bytes.
macro_rules! bhhandler_ovf_jump_ii {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
            let a = bh.registers_i[code[position + 2] as usize];
            let b = bh.registers_i[code[position + 3] as usize];
            let result_reg = code[position + 4] as usize;
            let pc = position + 5;
            let (maybe_result, new_pos) = $bhimpl(a, b, target, pc);
            if let Some(r) = maybe_result {
                bh.registers_i[result_reg] = r;
            }
            Ok(new_pos)
        }
    };
}

bhhandler_ovf_jump_ii!(handler_int_add_jump_if_ovf, bhimpl_int_add_jump_if_ovf);
bhhandler_ovf_jump_ii!(handler_int_sub_jump_if_ovf, bhimpl_int_sub_jump_if_ovf);
bhhandler_ovf_jump_ii!(handler_int_mul_jump_if_ovf, bhimpl_int_mul_jump_if_ovf);

// Miscellaneous simple operations
bhhandler_r_v!(handler_assert_not_none, bhimpl_assert_not_none);

bhhandler_r_r!(handler_virtual_ref, bhimpl_virtual_ref);
bhhandler_r_v!(handler_virtual_ref_finish, bhimpl_virtual_ref_finish);
bhhandler_i_v!(handler_loop_header, bhimpl_loop_header);
bhhandler_r_i!(handler_ref_isconstant, bhimpl_ref_isconstant);
bhhandler_r_i!(handler_ref_isvirtual, bhimpl_ref_isvirtual);
// Temporarily expanded from `bhhandler_goto_if_not_i!(handler_goto_if_not,
// bhimpl_goto_if_not)` for MAJIT_BH_DEBUG cond inspection (#210).
fn handler_goto_if_not(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if crate::bh_debug_enabled() {
        eprintln!(
            "[bh-brcond] pos={} cond_reg=i{} cond={} target={} fallthrough={}",
            position - 1,
            code[position],
            a,
            target,
            pc
        );
    }
    Ok(bhimpl_goto_if_not(a, target, pc))
}
bhhandler_goto_if_not_i!(
    handler_goto_if_not_int_is_zero,
    bhimpl_goto_if_not_int_is_zero
);
bhhandler_goto_if_not_r!(
    handler_goto_if_not_ptr_iszero,
    bhimpl_goto_if_not_ptr_iszero
);
bhhandler_goto_if_not_r!(
    handler_goto_if_not_ptr_nonzero,
    bhimpl_goto_if_not_ptr_nonzero
);
fn handler_unreachable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bhimpl_unreachable()
}

//
// RPython blackhole.py bhimpl_getfield_gc_i: bhimpl_getfield_gc_*/setfield_gc_*
// These call `cpu.bh_getfield_gc_i(struct_ptr, descr)` etc.
// The 'd' argcode is a 2-byte descriptor index into `bh.descrs`.
// In pyre, descrs[index] resolves to a field offset (usize).
//
// The default `Backend` impl is a raw load/store at that offset
// (`llmodel.py read_int_at_mem`). Going through `bh.cpu()` is two
// virtual calls per field on a path the regex leaf interprets for
// the rest of `shift`. Same bytes, no vtable.

#[inline(always)]
fn bh_load_int_field(struct_ptr: i64, descr: &BhDescr) -> i64 {
    let (offset, size, sign) = descr.unpack_fielddescr_size();
    let addr = (struct_ptr as usize).wrapping_add(offset);
    // SAFETY: `struct_ptr` is the GC object the resume reader seeded;
    // `offset`/`size` come from the field descriptor.
    unsafe {
        match (size, sign) {
            (1, true) => (addr as *const i8).read_unaligned() as i64,
            (1, false) => (addr as *const u8).read_unaligned() as i64,
            (2, true) => (addr as *const i16).read_unaligned() as i64,
            (2, false) => (addr as *const u16).read_unaligned() as i64,
            (4, true) => (addr as *const i32).read_unaligned() as i64,
            (4, false) => (addr as *const u32).read_unaligned() as i64,
            (8, _) => (addr as *const i64).read_unaligned(),
            other => panic!("bh_load_int_field: unsupported (size, signed) = {other:?}"),
        }
    }
}

#[inline(always)]
fn bh_store_int_field(struct_ptr: i64, value: i64, descr: &BhDescr) {
    let (offset, size, _sign) = descr.unpack_fielddescr_size();
    unsafe { majit_backend::llmodel::write_int_at_mem(struct_ptr as usize, offset, size, value) }
}

#[inline(always)]
fn bh_load_ref_field(struct_ptr: i64, descr: &BhDescr) -> i64 {
    let offset = descr.as_offset();
    let addr = (struct_ptr as usize).wrapping_add(offset);
    unsafe { (addr as *const usize).read_unaligned() as i64 }
}

#[inline(always)]
fn bh_store_ref_field(struct_ptr: i64, value: i64, descr: &BhDescr) {
    let offset = descr.as_offset();
    majit_gc::bh_probe_note_store(struct_ptr as usize, offset, 9);
    unsafe {
        majit_backend::llmodel::write_ref_at_mem(struct_ptr as usize, offset, value as usize);
    }
    // dynasm `bh_setfield_gc_r` / cranelift `write_barrier_if_managed`:
    // skip the header walk when the base is a reconstructed unmanaged
    // object. `gc_write_barrier` alone does not do that check.
    if majit_gc::gc_owns_object(struct_ptr as usize) {
        majit_gc::gc_write_barrier(majit_ir::GcRef(struct_ptr as usize));
    }
}

#[inline(always)]
fn bh_load_float_field(struct_ptr: i64, descr: &BhDescr) -> f64 {
    let offset = descr.as_offset();
    let addr = (struct_ptr as usize).wrapping_add(offset);
    unsafe { (addr as *const f64).read_unaligned() }
}

#[inline(always)]
fn bh_store_float_field(struct_ptr: i64, value: f64, descr: &BhDescr) {
    let offset = descr.as_offset();
    unsafe { majit_backend::llmodel::write_float_at_mem(struct_ptr as usize, offset, value) }
}

// CPU-dependent field and array operations
/// Two-byte little-endian operand. Translated `dispatch_loop` reads
/// `ord(code[position]) | (ord(code[position+1]) << 8)` with no check.
#[inline(always)]
unsafe fn bh_code_u16(code: &[u8], pos: usize) -> usize {
    unsafe { (*code.get_unchecked(pos) as usize) | ((*code.get_unchecked(pos + 1) as usize) << 8) }
}

/// `d` argcode against the jitcode's own descr pool, then the builder table.
/// RPython is `self.descrs[index]`; the runtime jitcode pool is that list
/// for a `JitCodeBuilder` helper.
#[inline(always)]
unsafe fn read_descr_fast<'a>(
    exec_descrs: &'a [crate::jitcode::RuntimeBhDescr],
    bh: &'a BlackholeInterpreter,
    code: &[u8],
    pos: usize,
) -> (&'a BhDescr, usize) {
    let descr_idx = unsafe { bh_code_u16(code, pos) };
    if descr_idx < exec_descrs.len() {
        let descr = unsafe { exec_descrs.get_unchecked(descr_idx) }
            .as_bh_descr()
            .unwrap_or_else(|| panic!("d-arg descrs[{descr_idx}] is not a BhDescr entry"));
        return (descr, pos + 2);
    }
    read_descr(bh, code, pos)
}

/// RPython `blackhole.py:150-157`: read a 2-byte descriptor index from
/// bytecode and return `(descr_object, new_position)`.
///
/// In RPython: `value = self.descrs[index]`. In pyre: returns the
/// `BhDescr` enum variant.
#[inline]
fn read_descr<'a>(bh: &'a BlackholeInterpreter, code: &[u8], pos: usize) -> (&'a BhDescr, usize) {
    let descr_idx = (code[pos] as usize) | ((code[pos + 1] as usize) << 8);
    // `descr_at` resolves the per-jitcode `exec.descrs` first, then the
    // process-global build-time descr pool — a build-time jitcode (jd1's drain
    // and its inlined callees) carries an empty `exec.descrs` and names its
    // `d`-arg descrs by their index in that shared pool, matching the trace
    // walker's own `descr_at` resolution.
    if let Some(entry) = bh.jitcode.descr_at(descr_idx) {
        let descr = entry.as_bh_descr().unwrap_or_else(|| {
            panic!("d-arg descrs[{descr_idx}] is not a BhDescr entry: {entry:?}")
        });
        return (descr, pos + 2);
    }
    let descr = bh.descrs.get(descr_idx).unwrap_or_else(|| {
        panic!(
            "d-arg descrs[{descr_idx}] is out of range for {} entries",
            bh.descrs.len()
        )
    }); // RPython: no fallback, index must be valid
    (descr, pos + 2)
}

/// A `d`-arg call descriptor detached from the frame that named it.
///
/// [`read_descr`] hands back a reference borrowed from the frame, and the
/// residual-call handlers need `&mut` it immediately afterwards. Copying the
/// descriptor to break that borrow is not free: `BhCallDescr` owns an
/// `arg_classes` string and an `EffectInfo` carrying six optional operand
/// vectors of `Arc` descriptors, so a copy per residual call is a handful of
/// allocations plus a refcount bump per operand. This retains the pool instead
/// and reads the descriptor back out of it.
enum CallDescrHandle {
    /// The descriptor is one the frame's jitcode resolves; the `Arc` keeps
    /// that pool reachable independently of the frame borrow, and of a later
    /// `setposition` on the frame.
    InJitCode(std::sync::Arc<JitCode>, usize),
    /// The descriptor came from the builder's table, which is `'static`.
    InBuilderTable(&'static BhDescr),
    /// The descriptor is a callee jitcode's own, named by a `j` operand.
    InCalleeJitCode(std::sync::Arc<JitCode>),
}

impl CallDescrHandle {
    /// The callee jitcode a canonical `inline_call_*` named, when the
    /// descr pool still holds that object. Needed to byte-interpret a
    /// helper whose `fnaddr` is a build-time symbolic hash.
    fn as_callee_jitcode(&self) -> Option<std::sync::Arc<JitCode>> {
        match self {
            Self::InCalleeJitCode(jitcode) => Some(jitcode.clone()),
            Self::InJitCode(jitcode, index) => jitcode
                .descr_at(*index)
                .and_then(|entry| entry.as_jitcode_owned()),
            Self::InBuilderTable(_) => None,
        }
    }

    fn get(&self) -> &BhCallDescr {
        let descr = match self {
            Self::InJitCode(jitcode, index) => jitcode
                .descr_at(*index)
                .and_then(|entry| entry.as_bh_descr())
                .unwrap_or_else(|| panic!("d-arg descrs[{index}] left the jitcode pool")),
            Self::InBuilderTable(descr) => descr,
            Self::InCalleeJitCode(jitcode) => return jitcode.calldescr(),
        };
        descr.as_calldescr()
    }
}

/// [`read_descr`] for the residual-call handlers, which need the descriptor to
/// outlive their `&mut` borrow of the frame.
fn read_calldescr(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (CallDescrHandle, usize) {
    // Resolve through `read_descr` first so a bad index or a non-`BhDescr`
    // entry fails there, with the message it already carries.
    let (_, next) = read_descr(bh, code, pos);
    let descr_idx = (code[pos] as usize) | ((code[pos + 1] as usize) << 8);
    let handle = if bh.jitcode.descr_at(descr_idx).is_some() {
        CallDescrHandle::InJitCode(bh.jitcode.clone(), descr_idx)
    } else {
        CallDescrHandle::InBuilderTable(
            bh.descrs
                .get(descr_idx)
                .expect("resolved by the read_descr above"),
        )
    };
    (handle, next)
}

/// Read a VableField descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved byte offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset directly; pyre VableField.index
/// needs vinfo.static_fields[index].offset resolution.
///
/// Copy `field_size` / `field_type` / signedness from
/// `vinfo.static_fields`.  A `float(f32)` vable scalar is four bytes;
/// synthesizing a word-sized `Ref` descr here would make
/// `bh_getfield_gc_f` consume the next field.
#[inline]
fn read_descr_vable_field(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhDescr, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let field_index = descr.as_vable_field_index();
    // Resolve field_index -> byte offset via VirtualizableInfo.  PyPy gets
    // this from fielddescr.get_vinfo(); pyre stores the equivalent vinfo on
    // the active blackhole frame, so absence is a contract bug, not a
    // byte-offset fallback.
    let vinfo = if bh.virtualizable_info.is_null() {
        panic!(
            "read_descr_vable_field: virtualizable_info must be set for VableField index {} \
             (RPython blackhole.py:1446 fielddescr.get_vinfo() parity)",
            field_index
        );
    } else {
        unsafe { &*bh.virtualizable_info }
    };
    let field = vinfo.static_fields.get(field_index).unwrap_or_else(|| {
        panic!(
            "read_descr_vable_field: VableField index {} out of bounds for {} static fields",
            field_index,
            vinfo.static_fields.len()
        )
    });
    let field_flag = match field.field_type {
        majit_ir::value::Type::Ref => majit_ir::descr::ArrayFlag::Pointer,
        majit_ir::value::Type::Float => majit_ir::descr::ArrayFlag::Float,
        majit_ir::value::Type::Int if field.field_signed => majit_ir::descr::ArrayFlag::Signed,
        majit_ir::value::Type::Int => majit_ir::descr::ArrayFlag::Unsigned,
        majit_ir::value::Type::Void => majit_ir::descr::ArrayFlag::Void,
    };
    (
        BhDescr::Field {
            offset: field.offset,
            field_size: field.field_size,
            field_type: field.field_type,
            field_flag,
            is_field_signed: field.field_signed,
            is_immutable: false,
            is_quasi_immutable: false,
            // No parent list, so there is no slot to claim — `None` rather than a
            // `0` that reads as a claim on the first field of a list that does
            // not exist here.
            index_in_parent: None,
            parent: None,
            name: String::new(),
            owner: String::new(),
        },
        pos,
    )
}

/// Synthesize the field descr `cpu.bh_getfield_gc_r` uses to load a
/// virtualizable array pointer. RPython's `fielddescr` already carries that
/// byte offset; pyre's `BhDescr::VableArray` is only an index into
/// `vinfo.array_fields`.
fn vable_array_field_descr(
    vinfo: &crate::virtualizable::VirtualizableInfo,
    array_index: usize,
) -> BhDescr {
    let offset = vinfo
        .array_fields
        .get(array_index)
        .unwrap_or_else(|| {
            panic!(
                "vable_array_field_descr: VableArray index {} out of bounds for {} array fields",
                array_index,
                vinfo.array_fields.len()
            )
        })
        .field_offset;
    BhDescr::Field {
        offset,
        // The array field is a pointer, so its width is the target word
        // like the scalars above.  Latent rather than live: the `_gc_r`
        // accessors this descr reaches take `as_offset()` and store at
        // pointer width without consulting the size.
        field_size: std::mem::size_of::<usize>(),
        field_type: majit_ir::value::Type::Ref,
        field_flag: majit_ir::descr::ArrayFlag::Pointer,
        is_field_signed: false,
        is_immutable: false,
        is_quasi_immutable: false,
        // No parent list, so there is no slot to claim — `None` rather than a
        // `0` that reads as a claim on the first field of a list that does
        // not exist here.
        index_in_parent: None,
        parent: None,
        name: String::new(),
        owner: String::new(),
    }
}

/// Read a VableArray descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved field_offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset for the array pointer field.
/// pyre: VableArray.index → vinfo.array_fields[index].field_offset.
///
/// Returns `(field_descr, array_index, next_pos)`.
#[inline]
fn read_descr_vable_array(
    bh: &BlackholeInterpreter,
    code: &[u8],
    pos: usize,
) -> (BhDescr, usize, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let array_index = descr.as_vable_array_index();
    let vinfo = if bh.virtualizable_info.is_null() {
        panic!(
            "read_descr_vable_array: virtualizable_info must be set for VableArray index {} \
             (RPython `bhimpl_getarrayitem_vable_*` `fielddescr.get_vinfo()` parity)",
            array_index
        );
    } else {
        unsafe { &*bh.virtualizable_info }
    };
    (
        vable_array_field_descr(vinfo, array_index),
        array_index,
        pos,
    )
}

// bhimpl_getfield_gc_i: @arguments("cpu", "r", "d", returns="i")
fn handler_getfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    bh.registers_i[code[pos] as usize] = bh_load_int_field(struct_ptr, descr);
    Ok(pos + 1)
}
fn handler_getfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    bh.registers_r[code[pos] as usize] = bh_load_ref_field(struct_ptr, descr);
    Ok(pos + 1)
}
fn handler_getfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    bh.registers_f[code[pos] as usize] = bh_load_float_field(struct_ptr, descr).to_bits() as i64;
    Ok(pos + 1)
}
// bhimpl_setfield_gc_i: @arguments("cpu", "r", "i", "d")
fn handler_setfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    bh_store_int_field(struct_ptr, value, descr);
    Ok(pos)
}
// `setfield_gc_i/rcd` — USE_C_FORM short value (`assembler.py`):
// the stored int is one inline signed byte read via `signedord`
// (`blackhole.py`) in place of a `registers_i` slot; the struct ref
// and descr are unchanged from the `rid` form.
fn handler_setfield_gc_i_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = code[position + 1] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 2);
    bh_store_int_field(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    bh_store_ref_field(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    bh_store_float_field(struct_ptr, value, descr);
    Ok(pos)
}

// bhimpl_arraylen_gc: @arguments("cpu", "r", "d", returns="i")
fn handler_arraylen_gc(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu();
    let result = cpu.bh_arraylen_gc(array_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}

// @arguments("cpu", "r", "i", "d", returns="X")

// GC array reads (`blackhole.py:1329-1341`)
fn handler_getarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu();
    bh.registers_i[code[pos] as usize] = cpu.bh_getarrayitem_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_i_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 2);
    let result = bh.cpu().bh_getarrayitem_gc_i(array, index, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu();
    bh.registers_r[code[pos] as usize] = cpu.bh_getarrayitem_gc_r(array, index, descr).0 as i64;
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_r_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 2);
    let result = bh.cpu().bh_getarrayitem_gc_r(array, index, descr).0 as i64;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}

// @arguments("cpu", "r", "i", "X", "d")

// GC array writes (`blackhole.py:1350-1358`)
fn handler_setarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}
fn handler_setarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_r[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}

// `setarrayitem_gc_r/rcrd` — `c`-argcode index (`assembler.py:99-107
// emit_const(allow_short=True)`, USE_C_FORM `assembler.py`): the
// index is one inline signed byte (`blackhole.py:127-129` decodes
// `'c'` as a signed char), not a `registers_i`/pool slot.
fn handler_setarrayitem_gc_r_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let value = bh.registers_r[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}

// `setarrayitem_gc_i/ricd` — `c`-argcode VALUE (`assembler.py:99-107
// emit_const(allow_short=True)`, USE_C_FORM `assembler.py`): the
// stored int is one inline signed byte (`blackhole.py:123` decodes `'c'`
// as a signed char), not a `registers_i`/pool slot; the array ref, index
// reg and descr keep the `riid` byte positions.  Mirror of
// `handler_setfield_gc_i_c`.
fn handler_setarrayitem_gc_i_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = code[position + 2] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}

fn handler_setarrayitem_gc_i_index_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    bh.cpu().bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}

fn handler_setarrayitem_gc_i_both_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let value = code[position + 2] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 3);
    bh.cpu().bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}

// Raw field reads (`blackhole.py:1464-1472`)
fn handler_getfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize]; // raw ptr is int
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu();
    bh.registers_i[code[pos] as usize] = cpu.bh_getfield_raw_i(struct_ptr, descr);
    Ok(pos + 1)
}
fn handler_getfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu();
    bh.registers_f[code[pos] as usize] = cpu.bh_getfield_raw_f(struct_ptr, descr).to_bits() as i64;
    Ok(pos + 1)
}

// Raw field writes (`blackhole.py:1497-1502`)
fn handler_setfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu();
    cpu.bh_setfield_raw_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu();
    cpu.bh_setfield_raw_f(struct_ptr, value, descr);
    Ok(pos)
}

// ── new / new_with_vtable / new_array (blackhole.py:1301-1327) ──────
// These need SizeDescr which pyre doesn't fully have yet.
// Stub handlers that read the descriptor and call cpu.bh_new etc.

fn handler_new(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "d", returns="r")
    let (descr, pos) = read_descr(bh, code, position);
    let result = bh.cpu().bh_new(descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_new_with_vtable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let (descr, pos) = read_descr(bh, code, position);
    let result = bh.cpu().bh_new_with_vtable(descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_new_array(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "d", returns="r")
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let result = bh.cpu().bh_new_array(length, descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}

fn handler_new_array_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = code[position] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 1);
    let result = bh.cpu().bh_new_array(length, descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_new_array_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let result = bh.cpu().bh_new_array_clear(length, descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}

// `new_array_clear/cd>r` — `c`-argcode length (see
// `handler_setarrayitem_gc_r_c`): one inline signed byte.
fn handler_new_array_clear_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = code[position] as i8 as i64;
    let (descr, pos) = read_descr(bh, code, position + 1);
    let result = bh.cpu().bh_new_array_clear(length, descr);
    check_blackhole_allocation_after(bh, result, pos + 1)?;
    bh.registers_r[code[pos] as usize] = result;
    Ok(pos + 1)
}

// String operations (`blackhole.py bhimpl_getlistitem_gc_r`)
fn handler_strlen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu();
    bh.registers_i[code[position + 1] as usize] = cpu.bh_strlen(s);
    Ok(position + 2)
}
fn handler_strgetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu();
    bh.registers_i[code[position + 2] as usize] = cpu.bh_strgetitem(s, index);
    Ok(position + 3)
}
fn handler_strgetitem_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = code[position + 1] as i8 as i64;
    let result = bh.cpu().bh_strgetitem(s, index);
    bh.registers_i[code[position + 2] as usize] = result;
    Ok(position + 3)
}
fn handler_strsetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu();
    cpu.bh_strsetitem(s, index, value);
    Ok(position + 3)
}
fn handler_strsetitem_short<const INDEX_SHORT: bool, const VALUE_SHORT: bool>(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = if INDEX_SHORT {
        code[position + 1] as i8 as i64
    } else {
        bh.registers_i[code[position + 1] as usize]
    };
    let value = if VALUE_SHORT {
        code[position + 2] as i8 as i64
    } else {
        bh.registers_i[code[position + 2] as usize]
    };
    bh.cpu().bh_strsetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newstr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let result = bh.cpu().bh_newstr(length);
    check_blackhole_allocation_after(bh, result, position + 2)?;
    bh.registers_r[code[position + 1] as usize] = result;
    Ok(position + 2)
}
fn handler_newstr_c(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = code[position] as i8 as i64;
    let result = bh.cpu().bh_newstr(length);
    check_blackhole_allocation_after(bh, result, position + 2)?;
    bh.registers_r[code[position + 1] as usize] = result;
    Ok(position + 2)
}
fn handler_unicodelen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu();
    bh.registers_i[code[position + 1] as usize] = cpu.bh_unicodelen(s);
    Ok(position + 2)
}
fn handler_unicodegetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu();
    bh.registers_i[code[position + 2] as usize] = cpu.bh_unicodegetitem(s, index);
    Ok(position + 3)
}
fn handler_unicodesetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu();
    cpu.bh_unicodesetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newunicode(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let result = bh.cpu().bh_newunicode(length);
    check_blackhole_allocation_after(bh, result, position + 2)?;
    bh.registers_r[code[position + 1] as usize] = result;
    Ok(position + 2)
}

// Exception handling (`blackhole.py:969-1009`)
fn handler_catch_exception(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("L") — no-op, skip 2-byte label
    Ok(position + 2)
}

// Miscellaneous no-ops (`blackhole.py:1017-1049`)
fn handler_jit_debug(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("r", "i", "i", "i", "i") = 1 ref + 4 int = 5 regs
    Ok(position + 5)
}
bhhandler_i_v!(
    handler_jit_enter_portal_frame,
    bhimpl_jit_enter_portal_frame
);
bhhandler_v_v!(
    handler_jit_leave_portal_frame,
    bhimpl_jit_leave_portal_frame
);

// @arguments("cpu", "r", "i", "d", returns="X")
// Interior GC fields (`blackhole.py`)
fn handler_getinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu();
    bh.registers_i[code[pos] as usize] = cpu.bh_getinteriorfield_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_setinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu();
    cpu.bh_setinteriorfield_gc_i(array, index, value, descr);
    Ok(pos)
}

/// blackhole.py `_get_list_of_values` + `make_sure_not_resized`.
/// I/R/F lists are the inline_call arity — a handful of register values.
/// A 1-item / 2-item `Vec<i64>` was the 8 B / 16 B class on the regex
/// and/or timed row (`handler_inline_call_ir_i`).
const BH_ARG_LIST_INLINE: usize = 8;
type BhArgList = smallvec::SmallVec<[i64; BH_ARG_LIST_INLINE]>;

#[inline]
fn read_list_from_regs(regs: &[i64], code: &[u8], pos: usize) -> (BhArgList, usize) {
    let count = code[pos] as usize;
    let mut values = BhArgList::new();
    for i in 0..count {
        values.push(regs[code[pos + 1 + i] as usize]);
    }
    (values, pos + 1 + count)
}

#[inline]
fn read_list_i(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhArgList, usize) {
    read_list_from_regs(&bh.registers_i, code, pos)
}
#[inline]
fn read_list_r(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhArgList, usize) {
    read_list_from_regs(&bh.registers_r, code, pos)
}
#[inline]
fn read_list_f(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhArgList, usize) {
    read_list_from_regs(&bh.registers_f, code, pos)
}

/// blackhole.py `_get_method`'s `except Exception` for every opcode that
/// reaches host code: turn a non-zero `BH_LAST_EXC_VALUE` into a
/// [`DispatchError::RaiseException`] carrying the post-call coordinate.
///
/// Upstream wraps *every* `bhimpl_*` this way, so `residual_call_*`,
/// `inline_call_*`, `call_assembler_*` and `cond_call_*` all owe the check —
/// a family that skips it writes the callee's uninitialised result into its
/// destination register and runs on with a live exception in the cell.
///
/// The exception travels through the cell rather than through Rust
/// unwinding, so this test is where the `except` clause lands; the handler
/// search and the `got_exception` propagation stay in the dispatch loop,
/// exactly as upstream leaves them to `run`.
///
/// `next_pos` is the position of the opcode immediately following the call
/// (the handler's normal return value), which is where the codewriter
/// emitted the can-raise opcode's `-live-`/`catch_exception` adjacency
/// (`pyre-jit`'s `codewriter.rs` `transform_graph_to_jitcode`).  Starting
/// the search at the operand start instead made `handle_exception_in_frame`
/// scan backward past the pre-call `-live-` and miss the catch, so a
/// value-guard deopt that re-executes a raising residual call (e.g.
/// `int_floordiv` by zero) escaped its try-block.
#[inline]
fn check_residual_call_exception_after(
    bh: &mut BlackholeInterpreter,
    next_pos: usize,
) -> Result<(), DispatchError> {
    let exc_val = bh.last_exc().get();
    if exc_val == 0 {
        return Ok(());
    }
    // A raise inside the callee reaches this side through the cell rather
    // than through Rust unwinding, so this is where `blackhole.py`
    // `_get_method`'s `except Exception` sits for pyre.  Everything after it
    // — advancing the position, the handler search, the `got_exception`
    // propagation — belongs to the dispatch loop, exactly as upstream leaves
    // it to `run`.  `next_pos` is the post-call position the handler is about
    // to return, where the codewriter emitted the can-raise opcode's
    // `-live-` / `catch_exception` adjacency (`pyre-jit`'s `codewriter.rs`
    // `transform_graph_to_jitcode`); starting the search at the pre-call
    // operand byte instead let a residual call escape its enclosing `try`.
    Err(DispatchError::RaiseException {
        exc: exc_val,
        resume_position: next_pos,
    })
}

/// The allocation-specialized edge of blackhole.py `_get_method`.
///
/// `GcLLDescr_framework._bh_malloc` / `_bh_malloc_array` call the translated
/// `do_malloc_*_clear` helpers, which raise `MemoryError` instead of returning
/// a null GC reference.  Rust backend helpers use NULL as that unwinding edge,
/// so convert it before a handler writes its result register.  In particular,
/// NULL is never an ordinary `r` result that the following opcode may consume.
#[inline]
fn check_blackhole_allocation_after(
    bh: &mut BlackholeInterpreter,
    result: i64,
    next_pos: usize,
) -> Result<(), DispatchError> {
    if result != 0 {
        return Ok(());
    }
    // String helpers also populate the compiled-code exception cell.  The
    // blackhole delivers the same singleton through DispatchError, so consume
    // the backend copy now; otherwise an unrelated later compiled call sees a
    // stale MemoryError.
    bh.cpu().clear_stored_exception();
    let err = blackhole_allocation_error(next_pos);
    // No MemoryError provider: `LeaveFrame` alone is a successful return
    // in `resume_mainloop`. Mark the frame aborted so the caller bails.
    if matches!(err, DispatchError::LeaveFrame) {
        bh.aborted = true;
    }
    Err(err)
}

#[inline]
fn blackhole_allocation_error(next_pos: usize) -> DispatchError {
    let exc = majit_backend::memory_error_singleton_ref();
    // Grain never installs MiniMark or a MemoryError singleton. A typed
    // `bh_new` then returns NULL; abort the frame rather than panic.
    // Hosts that registered a provider still raise it.
    if exc == 0 {
        return DispatchError::LeaveFrame;
    }
    DispatchError::RaiseException {
        exc,
        resume_position: next_pos,
    }
}

/// Refuse to jump to a `residual_call_*` funcptr that is either null or a
/// `symbolic_fnaddr_for_path` hash (see [`is_callable_fnaddr`]).
///
/// `bh_call_*` takes the funcptr straight to an indirect branch, so a hash
/// there is a wild call. The walker's residual path already declines this exact
/// case (`ResidualDecline::Symbolic`); the blackhole has no decline channel, so
/// it aborts the frame, which hands the continuation back to the interpreter.
///
#[inline]
/// Refuse a register write that would land in the constant table.
///
/// `init_register_file_from_i64s` places the jitcode's constants at
/// `num_regs_i ..` of the same bank the working registers live in, so the two
/// are only separated by that bound.  Nothing distinguishes them at the read,
/// which is why a clobbered constant is reported here rather than where its
/// value is finally used.
#[track_caller]
fn debug_assert_constant_slot_untouched(index: usize, num_regs: usize, who: &str) {
    if crate::jit_strict_mode() {
        assert!(
            index < num_regs,
            "{who}: register index {index} is at or past num_regs {num_regs}, so it \
             addresses the jitcode's constant table rather than a working register",
        );
    }
}

fn reject_unresolved_call(bh: &mut BlackholeInterpreter, func: i64) -> DispatchError {
    if crate::majit_log_enabled() {
        eprintln!(
            "[bh] residual_call declined: funcptr {func:#x} is not a callable code address; \
             register the callee's path in the host's fnaddr bindings (jitcode {:?} pos {} last \
             {})",
            bh.jitcode.name, bh.position, bh.last_opcode_position,
        );
    }
    bh.bail_residual = Some(
        crate::pyjitpl::resolve_symbolic_fnaddr_path(func)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("fnaddr {func:#x}")),
    );
    // `convert_and_run_from_pyjitpl` turns every abort into one resume.
    // Consumers accept that resume only when `abort_permanent_bail` is set
    // (`bhimpl_abort_permanent` is the other producer of this pair).
    bh.aborted = true;
    bh.abort_permanent_bail = true;
    DispatchError::LeaveFrame
}

// residual_call_irf_*

/// `MAJIT_BH_NULL_ARG`: report a null ref argument about to be handed to a
/// residual call, with the jitcode coordinate, before the callee can
/// dereference it.  Some ABIs pass a legitimate null sentinel (e.g. the
/// CallFn `null_or_self` slot), so this reports rather than aborts.
fn bh_null_arg_report(bh: &BlackholeInterpreter, ar: &[i64], position: usize) {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ARMED.get_or_init(|| std::env::var_os("MAJIT_BH_NULL_ARG").is_some()) {
        return;
    }
    if ar.contains(&0) {
        eprintln!(
            "[bh-null-arg] jitcode={} position={} last_opcode_position={} \
             entry_position={} args_r={:?}",
            bh.jitcode.name(),
            position,
            bh.last_opcode_position,
            bh.entry_position,
            ar,
        );
    }
}

// Call operations (`blackhole.py:1224-1276`)
fn handler_residual_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_irf_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_irf_i(func, &ai, &ar, &af, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_irf_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_irf_r(func, &ai, &ar, &af, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_irf_f.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_irf_f(func, &ai, &ar, &af, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_f[dst] = result.to_bits() as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py routes through bhimpl_residual_call_irf_v
    // which forwards to cpu.bh_call_v.
    bh.last_exc().set(0);
    bh.bhimpl_residual_call_irf_v(func, &ai, &ar, &af, calldescr);
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}
// residual_call_ir_*
fn handler_residual_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_ir_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_ir_i(func, &ai, &ar, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_ir_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_ir_r(func, &ai, &ar, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_ir_v.
    bh.last_exc().set(0);
    bh.bhimpl_residual_call_ir_v(func, &ai, &ar, calldescr);
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}
// residual_call_r_*
fn handler_residual_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_r_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_r_i(func, &ar, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    let dst = code[p] as usize;
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_r_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_residual_call_r_r(func, &ar, calldescr);
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    if !is_callable_fnaddr(func) {
        return Err(reject_unresolved_call(bh, func));
    }
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    bh_null_arg_report(bh, &ar, position);
    // blackhole.py → bhimpl_residual_call_r_v.
    bh.last_exc().set(0);
    bh.bhimpl_residual_call_r_v(func, &ar, calldescr);
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}

// A1-A8: every BC_* emitted by pyre now follows the canonical
// RPython argcode contract (1-byte register operands per
// `blackhole.py handler`).  All `*_u16_ext` width adapters have been
// retired; canonical `handler_*` decoders own every dispatch slot.

/// Build a strict `BlackholeInterpBuilder` for pyre's blackhole resume path.
///
/// TODO: pyre's `JitCodeBuilder` emits other
/// runtime opcodes in fixed BC_* bytes and several pyre-only payload
/// shapes.  This builder's `setup_insns` registers every opcode key
/// the production producers (`pyjitpl/dispatch.rs`, `majit-macros`
/// DSL lowerer, `pyre-jit/src/jit/assembler.rs`) can emit: byte-
/// identical canonical keys, single-byte register operands,
/// audited residual_call / vable / state-field families, and the pyre
/// nested inline-call handler.  The dispatch loop has no legacy
/// fallback; any emitted byte missing from this table reaches
/// `dispatch_step`'s unwired-opcode panic.
///
/// Initially registered:
///   - `inline_call_nested_ext/P` at `BC_INLINE_CALL` ()
///   - byte-identical canonical: `live/`, `loop_header/i`,
///     `goto/L`, `catch_exception/L`, `jit_merge_point/cIRFIRF`
///   - canonical-encoded: every `JitCodeBuilder`-emitted BC_*
///     now pushes 1-byte register operands matching the canonical
///     RPython argcode contract; no `_u16_ext` width adapters remain.
///     The pyre-jit production thread-locals (`BH_BUILDER3`,
///     `BH_BUILDER_RD`) and inline-call unit fixtures share this builder
///     shape.
///
/// See `pyre-jit-trace/src/jitcode_dispatch` for the
/// `pipeline.insns` ↔ `wellknown_bh_insns` table-unification epic
/// that this minimal install side-steps.
/// `dynamic_insns` carries the keys whose byte the assembler assigned
/// dynamically (`Assembler.get_opnum`), read off the producing build's
/// `asm.insns` by the caller; the canonical `BC_*` keys are registered here.
pub fn build_inline_call_only_bh_builder(dynamic_insns: &[(&str, u8)]) -> BlackholeInterpBuilder {
    let mut builder = BlackholeInterpBuilder::new();
    let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
    insns.insert(
        "inline_call_nested_ext/P".to_string(),
        majit_jitcode::insns::BC_INLINE_CALL,
    );
    // Leftover `_ext/P` adapters. `blackhole.py` has no
    // `bhimpl_call_assembler`; recursive portal calls emit
    // `BC_RECURSIVE_CALL_*` and bytes 50-53 stay unwired.
    // Canonical `conditional_call_*` / `record_known_result_*` keys
    // are registered below; leftover `cond_call_*_ext/P` /
    // `record_known_result_*_ext/P` bytes stay unwired.
    //
    // Five canonical keys whose `JitCodeBuilder` emit-side payload
    // matches the wired `bhimpl_*` handler byte-for-byte.  The rest of
    // the builder registers audited pyre families below; any byte absent
    // from setup_insns is a strict-dispatch error.
    //   * `live/` — operand-less; both paths skip 2-byte liveness offset.
    //   * `loop_header/i` — `i` is a 1-byte const-pool slot, RPython-aligned.
    //   * `goto/L` — operand-less for registers; 2-byte label.
    //   * `catch_exception/L` — same shape as `goto/L`.
    //   * `jit_merge_point/cIRFIRF` — both paths delegate to
    //     `bhimpl_jit_merge_point`, which decodes the post-opcode
    //     payload via `self.position`-mutating helpers; the IRFIRF
    //     register-list bytes are u8 in BOTH the emit-side and the
    //     decoder.
    insns.insert("live/".to_string(), majit_jitcode::insns::BC_LIVE);
    insns.insert(
        "loop_header/i".to_string(),
        majit_jitcode::insns::BC_LOOP_HEADER,
    );
    insns.insert("goto/L".to_string(), majit_jitcode::insns::BC_JUMP);
    insns.insert(
        "catch_exception/L".to_string(),
        majit_jitcode::insns::BC_CATCH_EXCEPTION,
    );
    insns.insert(
        "jit_merge_point/cIRFIRF".to_string(),
        majit_jitcode::insns::BC_JIT_MERGE_POINT_C,
    );
    // A1/A2 canonical-encoded family — `int_copy/i>i`, `ref_copy/r>r`,
    // `ref_return/r`, `raise/r`, `last_exc_value/>r` all emit 1-byte
    // register operands matching the canonical bhhandler decoders.
    insns.insert("int_copy/i>i".to_string(), majit_jitcode::insns::BC_MOVE_I);
    // `int_copy/c>i` — USE_C_FORM short source (`assembler.py`): a small
    // ConstInt materialised inline as one signed byte instead of a pool slot.
    // pyre emits this for every small int/bool constant, so the production
    // blackhole must dispatch it; `handler_int_copy_c` decodes via `signedord`.
    insns.insert(
        "int_copy/c>i".to_string(),
        majit_jitcode::insns::BC_MOVE_I_C,
    );
    insns.insert("ref_copy/r>r".to_string(), majit_jitcode::insns::BC_MOVE_R);
    insns.insert(
        "ref_return/r".to_string(),
        majit_jitcode::insns::BC_REF_RETURN,
    );
    insns.insert("raise/r".to_string(), majit_jitcode::insns::BC_RAISE);
    insns.insert(
        "last_exc_value/>r".to_string(),
        majit_jitcode::insns::BC_LAST_EXC_VALUE,
    );
    insns.insert(
        "goto_if_not_int_is_true/iL".to_string(),
        majit_jitcode::insns::BC_GOTO_IF_NOT_INT_IS_TRUE,
    );
    // `blackhole.py bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`
    // — both `(opname, argcodes)` keys route to the same handler body.
    // `Assembler.insns.setdefault` (`assembler.py`) allocates the
    // canonical key `goto_if_not/iL` its own byte (`BC_GOTO_IF_NOT`)
    // distinct from the alias byte (`BC_GOTO_IF_NOT_INT_IS_TRUE`).
    insns.insert(
        "goto_if_not/iL".to_string(),
        majit_jitcode::insns::BC_GOTO_IF_NOT,
    );
    // Cover the BC_* set the inline_call-only test
    // fixtures emit (`int_return`, `float_return`, `int_add`,
    // `float_copy`, `void_return`, `abort`, `abort_permanent`) so the
    // production builder is closed over every opname pyre's
    // `JitCodeBuilder` can produce.  The width migrations are complete:
    // every key here is canonical (1-byte register operands) and reuses
    // canonical `handler_*` decoders wired in `wire_bhimpl_handlers`.
    insns.insert(
        "int_return/i".to_string(),
        majit_jitcode::insns::BC_INT_RETURN,
    );
    // `int_return/c` — USE_C_FORM short source (`assembler.py`): a small
    // const-int return value emitted inline as one signed byte.
    insns.insert(
        "int_return/c".to_string(),
        majit_jitcode::insns::BC_INT_RETURN_C,
    );
    insns.insert(
        "float_return/f".to_string(),
        majit_jitcode::insns::BC_FLOAT_RETURN,
    );
    insns.insert(
        "void_return/".to_string(),
        majit_jitcode::insns::BC_VOID_RETURN,
    );
    insns.insert(
        "float_copy/f>f".to_string(),
        majit_jitcode::insns::BC_MOVE_F,
    );
    // A5 epic: full int binop+cmp+uint family (`record_binop_i` emit
    // shape), canonical 1-byte register encoding with `[lhs][rhs][dst]`
    // operand order matching `bhhandler_ii_i!`.
    for (key, byte) in [
        ("int_add/ii>i", majit_jitcode::insns::BC_INT_ADD),
        ("int_sub/ii>i", majit_jitcode::insns::BC_INT_SUB),
        ("int_mul/ii>i", majit_jitcode::insns::BC_INT_MUL),
        ("int_and/ii>i", majit_jitcode::insns::BC_INT_AND),
        ("int_signext/ii>i", majit_jitcode::insns::BC_INT_SIGNEXT),
        ("int_or/ii>i", majit_jitcode::insns::BC_INT_OR),
        ("int_xor/ii>i", majit_jitcode::insns::BC_INT_XOR),
        ("int_lshift/ii>i", majit_jitcode::insns::BC_INT_LSHIFT),
        ("int_rshift/ii>i", majit_jitcode::insns::BC_INT_RSHIFT),
        ("int_eq/ii>i", majit_jitcode::insns::BC_INT_EQ),
        ("int_ne/ii>i", majit_jitcode::insns::BC_INT_NE),
        ("int_lt/ii>i", majit_jitcode::insns::BC_INT_LT),
        ("int_le/ii>i", majit_jitcode::insns::BC_INT_LE),
        ("int_gt/ii>i", majit_jitcode::insns::BC_INT_GT),
        ("int_ge/ii>i", majit_jitcode::insns::BC_INT_GE),
        ("uint_lt/ii>i", majit_jitcode::insns::BC_UINT_LT),
        ("uint_le/ii>i", majit_jitcode::insns::BC_UINT_LE),
        ("uint_gt/ii>i", majit_jitcode::insns::BC_UINT_GT),
        ("uint_ge/ii>i", majit_jitcode::insns::BC_UINT_GE),
        ("uint_rshift/ii>i", majit_jitcode::insns::BC_UINT_RSHIFT),
        ("uint_mul_high/ii>i", majit_jitcode::insns::BC_UINT_MUL_HIGH),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // P5 — int unary + float arithmetic + ref/ptr family.
    // 14 keys covering record_unary_i/f, record_binop_f, record_binop_r,
    // and ptr_iszero/nonzero (`jitcode/assembler.rs`).
    for (key, byte) in [
        ("int_neg/i>i", majit_jitcode::insns::BC_INT_NEG),
        ("int_invert/i>i", majit_jitcode::insns::BC_INT_INVERT),
        // `int_is_true/i>i` — `@arguments("i", returns="i")` unary
        // (`blackhole.py bhimpl_int_is_true`). Emitted as the
        // loop-condition test of any int-typed `while`, including the
        // `_unpackiterable_unknown_length` drain's back-edge guard; its
        // handler is wired in `wire_bhimpl_handlers` but the byte was
        // absent from this curated set, so a blackhole-executed drain hit
        // the unwired-opcode panic at the first back-edge test.
        ("int_is_true/i>i", majit_jitcode::insns::BC_INT_IS_TRUE),
        // `int_is_zero/i>i` — the `not a` sibling of `int_is_true`
        // (`blackhole.py bhimpl_int_is_zero`).  Same wiring gap as the
        // entry above: `wire_bhimpl_handlers` binds `handler_int_is_zero`,
        // so without the byte here `wire_handler` no-ops and a
        // blackhole-executed `x == 0` test hits the unwired-opcode panic.
        ("int_is_zero/i>i", majit_jitcode::insns::BC_INT_IS_ZERO),
        // `int_force_ge_zero/i>i` — `blackhole.py bhimpl_int_force_ge_zero`.
        // `rlist.py ll_alloc_and_set` clamps `count` with this unary.
        // `wire_bhimpl_handlers` binds `handler_int_force_ge_zero`; without
        // the byte a forward resume through the clamp hits the unwired panic.
        (
            "int_force_ge_zero/i>i",
            majit_jitcode::insns::BC_INT_FORCE_GE_ZERO,
        ),
        ("float_add/ff>f", majit_jitcode::insns::BC_FLOAT_ADD),
        ("float_sub/ff>f", majit_jitcode::insns::BC_FLOAT_SUB),
        ("float_mul/ff>f", majit_jitcode::insns::BC_FLOAT_MUL),
        ("float_truediv/ff>f", majit_jitcode::insns::BC_FLOAT_TRUEDIV),
        // `cast_int_to_float/i>f` — `IntegerRepr.rtype_float` /
        // `_truediv`'s `float(x)`.  Handler is already wired
        // (`bhimpl_cast_int_to_float`); the byte was only in the
        // overlay-only gap until the codewriter started emitting it.
        (
            "cast_int_to_float/i>f",
            majit_jitcode::insns::BC_CAST_INT_TO_FLOAT,
        ),
        ("float_neg/f>f", majit_jitcode::insns::BC_FLOAT_NEG),
        ("float_abs/f>f", majit_jitcode::insns::BC_FLOAT_ABS),
        // Float value-comparisons materialized as an int result (e.g. a
        // dispatch arm writing `(a >= b) as i64`).  Their `bhimpl_*`
        // handlers are wired in `wire_bhimpl_handlers`; without the map
        // entry `wire_handler` no-ops and a forward resume through the
        // opcode hits the unwired-byte panic.
        ("float_lt/ff>i", majit_jitcode::insns::BC_FLOAT_LT),
        ("float_le/ff>i", majit_jitcode::insns::BC_FLOAT_LE),
        ("float_eq/ff>i", majit_jitcode::insns::BC_FLOAT_EQ),
        ("float_ne/ff>i", majit_jitcode::insns::BC_FLOAT_NE),
        ("float_gt/ff>i", majit_jitcode::insns::BC_FLOAT_GT),
        ("float_ge/ff>i", majit_jitcode::insns::BC_FLOAT_GE),
        ("ptr_eq/rr>i", majit_jitcode::insns::BC_PTR_EQ),
        ("ptr_ne/rr>i", majit_jitcode::insns::BC_PTR_NE),
        (
            "instance_ptr_eq/rr>i",
            majit_jitcode::insns::BC_INSTANCE_PTR_EQ,
        ),
        (
            "instance_ptr_ne/rr>i",
            majit_jitcode::insns::BC_INSTANCE_PTR_NE,
        ),
        ("ptr_iszero/r>i", majit_jitcode::insns::BC_PTR_ISZERO),
        ("ptr_nonzero/r>i", majit_jitcode::insns::BC_PTR_NONZERO),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // A7 — branch family canonical 1-byte register + 2-byte label
    // encoding.  Every `goto_if_not_*` opname JitCodeBuilder emits,
    // matching the canonical `bhhandler_goto_if_not_*` decoders.
    for (key, byte) in [
        (
            "goto_if_not_int_lt/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_LT,
        ),
        (
            "goto_if_not_int_le/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_LE,
        ),
        (
            "goto_if_not_int_eq/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_EQ,
        ),
        (
            "goto_if_not_int_ne/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_NE,
        ),
        (
            "goto_if_not_int_gt/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_GT,
        ),
        (
            "goto_if_not_int_ge/iiL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_GE,
        ),
        (
            "goto_if_not_float_lt/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LT,
        ),
        (
            "goto_if_not_float_le/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LE,
        ),
        (
            "goto_if_not_float_eq/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_EQ,
        ),
        (
            "goto_if_not_float_ne/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_NE,
        ),
        (
            "goto_if_not_float_gt/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GT,
        ),
        (
            "goto_if_not_float_ge/ffL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GE,
        ),
        (
            "goto_if_not_ptr_eq/rrL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_PTR_EQ,
        ),
        (
            "goto_if_not_ptr_ne/rrL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_PTR_NE,
        ),
        (
            "goto_if_not_int_is_zero/iL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_INT_IS_ZERO,
        ),
        (
            "goto_if_not_ptr_iszero/rL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_PTR_ISZERO,
        ),
        (
            "goto_if_exception_mismatch/iL",
            majit_jitcode::insns::BC_GOTO_IF_EXCEPTION_MISMATCH,
        ),
        (
            "goto_if_not_ptr_nonzero/rL",
            majit_jitcode::insns::BC_GOTO_IF_NOT_PTR_NONZERO,
        ),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // P7 — state_field family (u16 descr + u8 register),
    // push/pop/guard_value (canonical 1-byte register), and
    // misc operand-less (reraise, unreachable) + jit_merge_point/iIRFIRF
    // (already wired canonically).
    for (key, byte) in [
        // state_field/array/varray — canonical handlers wire directly.
        (
            "load_state_field_ref/dr",
            majit_jitcode::insns::BC_LOAD_STATE_FIELD_REF,
        ),
        (
            "store_state_field_ref/dr",
            majit_jitcode::insns::BC_STORE_STATE_FIELD_REF,
        ),
        (
            "load_state_field/di",
            majit_jitcode::insns::BC_LOAD_STATE_FIELD,
        ),
        (
            "store_state_field/di",
            majit_jitcode::insns::BC_STORE_STATE_FIELD,
        ),
        (
            "load_state_field_float/df",
            majit_jitcode::insns::BC_LOAD_STATE_FIELD_FLOAT,
        ),
        (
            "store_state_field_float/df",
            majit_jitcode::insns::BC_STORE_STATE_FIELD_FLOAT,
        ),
        // A3 epic: push/pop family migrated to canonical 1-byte register
        // encoding; handlers wired in `wire_bhimpl_handlers` decode via
        // `code[position]` (`bhhandler_push_*` / `bhhandler_pop_*`).
        ("int_push/i", majit_jitcode::insns::BC_INT_PUSH),
        ("int_pop/>i", majit_jitcode::insns::BC_INT_POP),
        ("ref_push/r", majit_jitcode::insns::BC_REF_PUSH),
        ("ref_pop/>r", majit_jitcode::insns::BC_REF_POP),
        ("float_push/f", majit_jitcode::insns::BC_FLOAT_PUSH),
        ("float_pop/>f", majit_jitcode::insns::BC_FLOAT_POP),
        // guard_value canonical — 1-byte register, blackhole no-op.
        (
            "int_guard_value/i",
            majit_jitcode::insns::BC_INT_GUARD_VALUE,
        ),
        (
            "ref_guard_value/r",
            majit_jitcode::insns::BC_REF_GUARD_VALUE,
        ),
        (
            "float_guard_value/f",
            majit_jitcode::insns::BC_FLOAT_GUARD_VALUE,
        ),
        // last_exception canonical — 1-byte dst register.
        ("last_exception/>i", majit_jitcode::insns::BC_LAST_EXCEPTION),
        // operand-less canonical (byte-identical between pyre and RPython).
        ("reraise/", majit_jitcode::insns::BC_RERAISE),
        ("unreachable/", majit_jitcode::insns::BC_UNREACHABLE),
        // jit_merge_point variant 2 (jdindex via const-pool).
        (
            "jit_merge_point/iIRFIRF",
            majit_jitcode::insns::BC_JIT_MERGE_POINT,
        ),
    ] {
        insns.insert(key.to_string(), byte);
    }
    insns.insert("abort/".to_string(), majit_jitcode::insns::BC_ABORT);
    // Result-carrying aborts. `wire_handler` only rebinds a key already in
    // this map, so the calls below are no-ops until the bytes are inserted
    // here.
    insns.insert(
        "abort/>r".to_string(),
        majit_jitcode::insns::BC_ABORT_RESULT_R,
    );
    insns.insert(
        "abort/>i".to_string(),
        majit_jitcode::insns::BC_ABORT_RESULT_I,
    );
    insns.insert(
        "abort_permanent/".to_string(),
        majit_jitcode::insns::BC_ABORT_PERMANENT,
    );
    // The canonical MIR dyn-call path is now a concrete vtable FieldRead plus
    // IndirectCall (`ClassRepr.getclsfield` / `FunctionReprBase.call`).  Keep
    // the old pyre-only byte registered for frozen/abstract-trait fallback
    // jitcodes; its handler safely bails to the source interpreter.
    insns.insert(
        "vtable_method_ptr/rd>i".to_string(),
        majit_jitcode::insns::BC_VTABLE_METHOD_PTR,
    );
    // blackhole.py bhimpl_switch. `handler_switch` is wired in
    // `wire_bhimpl_handlers` but the byte was absent from this builder's
    // insns map, so a deopt through a `switch/id` op landed on the
    // unwired-opcode placeholder. RPython's setup_insns binds every opname.
    insns.insert("switch/id".to_string(), majit_jitcode::insns::BC_SWITCH);
    // GC heap field load/store — `blackhole.py bhimpl_getfield_gc_i bhimpl_
    // {get,set}field_gc_{i,r,f}`.  `handler_{get,set}field_gc_*` are wired
    // in `wire_bhimpl_handlers` and the canonical bytes exist in
    // `wellknown_bh_insns`, but the whole family was absent from this
    // builder's insns map — so a deopt replaying a struct field access
    // (an int field store, `setfield_gc_i/rid` = BC_SETFIELD_GC_I 0xac)
    // landed on the unwired-opcode placeholder.
    // Canonical `rd`/`r{i,r,f}d` argcodes only (these access fields off a
    // ref base).  GC intbase (`id`/`iid`) is no longer a handler;
    // `promote_gc_field_bases` rehomes a Signed GC pointer so the
    // assembler emits `/rd>X` / `/rid`.  An Int-bank leftover is the
    // raw family.  Additive: every byte is currently unwired.
    for (key, byte) in [
        ("getfield_gc_i/rd>i", majit_jitcode::insns::BC_GETFIELD_GC_I),
        ("getfield_gc_r/rd>r", majit_jitcode::insns::BC_GETFIELD_GC_R),
        ("getfield_gc_f/rd>f", majit_jitcode::insns::BC_GETFIELD_GC_F),
        ("setfield_gc_i/rid", majit_jitcode::insns::BC_SETFIELD_GC_I),
        (
            "setfield_gc_i/rcd",
            majit_jitcode::insns::BC_SETFIELD_GC_I_C,
        ),
        ("setfield_gc_r/rrd", majit_jitcode::insns::BC_SETFIELD_GC_R),
        ("setfield_gc_f/rfd", majit_jitcode::insns::BC_SETFIELD_GC_F),
        (
            "getfield_gc_i_pure/rd>i",
            majit_jitcode::insns::BC_GETFIELD_GC_I_PURE,
        ),
        (
            "getfield_gc_r_pure/rd>r",
            majit_jitcode::insns::BC_GETFIELD_GC_R_PURE,
        ),
        (
            "getfield_gc_f_pure/rd>f",
            majit_jitcode::insns::BC_GETFIELD_GC_F_PURE,
        ),
        // `blackhole.py bhimpl_getfield_raw_i`. `wire_handler` is a no-op
        // until this map owns the byte (`setup_insns`).
        (
            "getfield_raw_i/id>i",
            majit_jitcode::insns::BC_GETFIELD_RAW_I,
        ),
        (
            "getfield_raw_f/id>f",
            majit_jitcode::insns::BC_GETFIELD_RAW_F,
        ),
        (
            "setfield_raw_i/iid",
            majit_jitcode::insns::BC_SETFIELD_RAW_I,
        ),
        (
            "setfield_raw_f/ifd",
            majit_jitcode::insns::BC_SETFIELD_RAW_F,
        ),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // residual_call family — `JitCodeBuilder::emit_canonical_call_void` /
    // `emit_canonical_call_typed` (jitcode/assembler.rs) emit each
    // register operand via `push_reg_u8` (asserts `reg <= u8::MAX`) and
    // each list count as `push_u8` (asserts `len <= u8::MAX`); only the
    // trailing `calldescr_idx` is `push_u16`.  This matches the
    // canonical RPython argcode contract (`blackhole.py:1224-1276`)
    // byte-for-byte, so the wired `handler_residual_call_*` decode
    // straight via `read_list_{i,r,f}` + `read_descr` without needing
    // width adapters.
    insns.insert(
        "residual_call_r_v/iRd".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_R_V,
    );
    insns.insert(
        "residual_call_r_i/iRd>i".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_R_I,
    );
    insns.insert(
        "residual_call_r_r/iRd>r".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_R_R,
    );
    insns.insert(
        "residual_call_ir_v/iIRd".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IR_V,
    );
    // jtransform `_rewrite_op_cond_call` now emits this key for
    // interpreter-facing `jit.conditional_call`. `wire_bhimpl_handlers`
    // already binds `handler_conditional_call_ir_v`.
    insns.insert(
        "conditional_call_ir_v/iiIRd".to_string(),
        majit_jitcode::insns::BC_CONDITIONAL_CALL_IR_V,
    );
    insns.insert(
        "conditional_call_value_ir_i/iiIRd>i".to_string(),
        majit_jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_I,
    );
    insns.insert(
        "conditional_call_value_ir_r/riIRd>r".to_string(),
        majit_jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_R,
    );
    insns.insert(
        "record_known_result_i_ir_v/iiIRd".to_string(),
        majit_jitcode::insns::BC_RECORD_KNOWN_RESULT_I_IR_V,
    );
    insns.insert(
        "record_known_result_r_ir_v/riIRd".to_string(),
        majit_jitcode::insns::BC_RECORD_KNOWN_RESULT_R_IR_V,
    );
    insns.insert(
        "residual_call_ir_i/iIRd>i".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IR_I,
    );
    insns.insert(
        "residual_call_ir_r/iIRd>r".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IR_R,
    );
    insns.insert(
        "residual_call_irf_v/iIRFd".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
    );
    insns.insert(
        "residual_call_irf_i/iIRFd>i".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
    );
    insns.insert(
        "residual_call_irf_r/iIRFd>r".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
    );
    insns.insert(
        "residual_call_irf_f/iIRFd>f".to_string(),
        majit_jitcode::insns::BC_RESIDUAL_CALL_IRF_F,
    );
    // Canonical `inline_call_*` family (blackhole.py:1278-1319). The ten
    // handlers and their `wire_bhimpl_handlers` calls already exist, but
    // `wire_handler` is a no-op for a key `setup_insns` never registered, so
    // without these entries the bytes reach `dispatch_step`'s unwired panic
    // instead of their handler. `JitCodeBuilder` now emits these keys.
    //
    // A target whose path the host never published has no runtime address;
    // `read_inline_call_jitcode` + `is_callable_fnaddr` decline it rather than
    // branch to a `symbolic_fnaddr_for_path` hash.
    for (key, byte) in [
        (
            "inline_call_r_i/dR>i",
            majit_jitcode::insns::BC_INLINE_CALL_R_I,
        ),
        (
            "inline_call_r_r/dR>r",
            majit_jitcode::insns::BC_INLINE_CALL_R_R,
        ),
        (
            "inline_call_r_v/dR",
            majit_jitcode::insns::BC_INLINE_CALL_R_V,
        ),
        (
            "inline_call_ir_i/dIR>i",
            majit_jitcode::insns::BC_INLINE_CALL_IR_I,
        ),
        (
            "inline_call_ir_r/dIR>r",
            majit_jitcode::insns::BC_INLINE_CALL_IR_R,
        ),
        (
            "inline_call_ir_v/dIR",
            majit_jitcode::insns::BC_INLINE_CALL_IR_V,
        ),
        (
            "inline_call_irf_i/dIRF>i",
            majit_jitcode::insns::BC_INLINE_CALL_IRF_I,
        ),
        (
            "inline_call_irf_r/dIRF>r",
            majit_jitcode::insns::BC_INLINE_CALL_IRF_R,
        ),
        (
            "inline_call_irf_f/dIRF>f",
            majit_jitcode::insns::BC_INLINE_CALL_IRF_F,
        ),
        (
            "inline_call_irf_v/dIRF",
            majit_jitcode::insns::BC_INLINE_CALL_IRF_V,
        ),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // vable family (full 14-key coverage).  Scalar
    // `getfield_vable_*` / `setfield_vable_*` / `hint_force_virtualizable`
    // and `getarrayitem_vable_*` / `setarrayitem_vable_*` /
    // `arraylen_vable` route through the canonical
    // `cpu.bh_getfield_gc_*` / `bh_getarrayitem_gc_*` /
    // `bh_setarrayitem_gc_*` / `bh_arraylen_gc` chain
    // (`bhimpl_getarrayitem_vable_*`).
    // Together this makes the vable family strict-dispatch ready.
    insns.insert(
        "getfield_vable_i/rd>i".to_string(),
        majit_jitcode::insns::BC_GETFIELD_VABLE_I,
    );
    insns.insert(
        "getfield_vable_r/rd>r".to_string(),
        majit_jitcode::insns::BC_GETFIELD_VABLE_R,
    );
    insns.insert(
        "getfield_vable_f/rd>f".to_string(),
        majit_jitcode::insns::BC_GETFIELD_VABLE_F,
    );
    insns.insert(
        "setfield_vable_i/rid".to_string(),
        majit_jitcode::insns::BC_SETFIELD_VABLE_I,
    );
    insns.insert(
        "setfield_vable_i_imm/rddd".to_string(),
        majit_jitcode::insns::BC_SETFIELD_VABLE_I_IMM,
    );
    insns.insert(
        "setfield_vable_r/rrd".to_string(),
        majit_jitcode::insns::BC_SETFIELD_VABLE_R,
    );
    insns.insert(
        "setfield_vable_f/rfd".to_string(),
        majit_jitcode::insns::BC_SETFIELD_VABLE_F,
    );
    insns.insert(
        "getarrayitem_vable_i/ridd>i".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_VABLE_I,
    );
    insns.insert(
        "getarrayitem_vable_r/ridd>r".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_VABLE_R,
    );
    insns.insert(
        "getarrayitem_vable_f/ridd>f".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_VABLE_F,
    );
    insns.insert(
        "setarrayitem_vable_i/riidd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_VABLE_I,
    );
    insns.insert(
        "setarrayitem_vable_r/rirdd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_VABLE_R,
    );
    insns.insert(
        "setarrayitem_vable_f/rifdd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_VABLE_F,
    );
    insns.insert(
        "arraylen_vable/rdd>i".to_string(),
        majit_jitcode::insns::BC_ARRAYLEN_VABLE,
    );
    // Registered for the same reason the array-build family below is: the
    // abort this op provokes resumes through the blackhole over the jitcode
    // that contains it, so leaving the byte unresolved would panic
    // `dispatch_step` on the op's own failure path.
    insns.insert(
        "arraybase_vable/rdd>i".to_string(),
        majit_jitcode::insns::BC_ARRAYBASE_VABLE,
    );
    // GC array-build family — `BuildTuple` / `BuildList` / `BuildMap` /
    // `BuildSet` / `BuildString` lower to `new_array_clear` (alloc) +
    // unrolled `setarrayitem_gc_r` (fill) + a `new*_from_array` residual
    // (already covered by the residual_call family above).  A
    // guard-failure resume into a jitcode whose forward path contains a
    // literal/return tuple (or list/map/set/str) walks the alloc + fill
    // ops, so the strict builder must wire them or `dispatch_step`
    // panics on the unwired byte.  Both the const-operand variants
    // (`cd>r` length, `rcrd` index — what the codewriter emits, since
    // the length and index are compile-time constants) and the
    // register-operand variants (`id>r`, `rird`) are registered so the
    // builder stays correct if the flattener ever colors them into
    // registers.  Handlers wired in `wire_bhimpl_handlers`.
    insns.insert(
        "new_array_clear/cd>r".to_string(),
        majit_jitcode::insns::BC_NEW_ARRAY_CLEAR_C,
    );
    insns.insert(
        "new_array_clear/id>r".to_string(),
        majit_jitcode::insns::BC_NEW_ARRAY_CLEAR,
    );
    // Plain struct allocation (`blackhole.py bhimpl_new` /
    // `bhimpl_new_with_vtable`) — a `#[jit_interp]` frontend's declared
    // `struct_allocs` literal lowers straight to it, so any resume that walks
    // forward over one needs the byte dispatchable.  The handlers were already
    // wired in `wire_bhimpl_handlers`; only the opname→byte entries were
    // missing, which turns into a `dispatch_step: unwired opcode` panic.
    insns.insert("new/d>r".to_string(), majit_jitcode::insns::BC_NEW);
    insns.insert(
        "new_with_vtable/d>r".to_string(),
        majit_jitcode::insns::BC_NEW_WITH_VTABLE,
    );
    insns.insert(
        "new_array/id>r".to_string(),
        majit_jitcode::insns::BC_NEW_ARRAY,
    );
    insns.insert(
        "setarrayitem_gc_r/rcrd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_R_C,
    );
    insns.insert(
        "setarrayitem_gc_r/rird".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_R,
    );
    insns.insert(
        "hint_force_virtualizable/r".to_string(),
        majit_jitcode::insns::BC_HINT_FORCE_VIRTUALIZABLE,
    );
    // GC heap array allocation + element access — emitted by BUILD_TUPLE /
    // BUILD_LIST (`new_array_clear` + `setarrayitem_gc_*`) and subscript
    // load/store (`getarrayitem_gc_*`). Unlike the vable family above
    // (which addresses the virtualizable frame), these touch ordinary heap
    // arrays. A guard failure that resumes forward through a tuple/list
    // build or an array element access must dispatch them; their `bhimpl_*`
    // handlers are wired in `wire_bhimpl_handlers` below. Canonical bytes
    // per `insns.rs` (blackhole.py:1311-1359). `getarrayitem_gc_*_pure` is
    // the immutable-array (e.g. tuple) read shape, aliased to the same
    // handler.
    insns.insert(
        "new_array_clear/id>r".to_string(),
        majit_jitcode::insns::BC_NEW_ARRAY_CLEAR,
    );
    insns.insert(
        "getarrayitem_gc_i/rid>i".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_I,
    );
    // `blackhole.py` `bhimpl_getarrayitem_raw_i`: int-bank sibling of
    // `getarrayitem_gc_i/rid>i`. Canonical `BC_GETARRAYITEM_RAW_I`;
    // `wire_handler` is a no-op until the key is in this map.
    insns.insert(
        "getarrayitem_raw_i/iid>i".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_RAW_I,
    );
    insns.insert(
        "getarrayitem_gc_r/rid>r".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_R_RID,
    );
    insns.insert(
        "getarrayitem_gc_f/rid>f".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_F,
    );
    insns.insert(
        "getarrayitem_gc_i_pure/rid>i".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_I_PURE,
    );
    insns.insert(
        "getarrayitem_gc_r_pure/rid>r".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_R_PURE,
    );
    insns.insert(
        "getarrayitem_gc_f_pure/rid>f".to_string(),
        majit_jitcode::insns::BC_GETARRAYITEM_GC_F_PURE,
    );
    // `raw_store_i/iiid` — emitted by the wasmi majit kernel's i64 memory
    // store arms (`majit_raw_store_i64` → `raw_store_i`, the analogue of
    // RPython `raw_storage_setitem`).  The kernel store is branchless and
    // never deopts on its own OOB access (it clamps out-of-bounds to an
    // in-bounds no-op); but a guard failure on another op can resume
    // forward through dispatch jitcode containing raw_store bytes, so the
    // blackhole must decode this op; without the map entry
    // `wire_handler("raw_store_i/iiid", ...)` silently no-ops and the byte
    // stays unwired, desyncing the blackhole decoder.
    insns.insert(
        "raw_store_i/iiid".to_string(),
        majit_jitcode::insns::BC_RAW_STORE_I,
    );
    // `raw_store_f/iifd` — the float-valued store (`rewrite_op_raw_store`
    // picks the value's kind).  `handler_raw_store_f` is wired below; the
    // byte needs the same resume-forward registration as `raw_store_i`.
    insns.insert(
        "raw_store_f/iifd".to_string(),
        majit_jitcode::insns::BC_RAW_STORE_F,
    );
    // `raw_load_i/iid>i` — the read-side companion the wasmi kernel emits for
    // the store's out-of-bounds memory-preserving no-op readback.  Same
    // resume-forward requirement as `raw_store_i` above.
    insns.insert(
        "raw_load_i/iid>i".to_string(),
        majit_jitcode::insns::BC_RAW_LOAD_I,
    );
    // `raw_load_f/iid>f` — the float read-side companion (`majit_raw_load_f`,
    // used by the celfloat example).  `handler_raw_load_f` is wired below;
    // register the opname so a forward resume through the opcode blackholes
    // instead of hitting the unwired-byte panic.
    insns.insert(
        "raw_load_f/iid>f".to_string(),
        majit_jitcode::insns::BC_RAW_LOAD_F,
    );
    // The two float<->int bitcasts, `majit_f64_to_bits` / `majit_bits_to_f64`.
    // A `#[jit_interp]` arm carries a float across a branch as its bits (the
    // lowerer expresses a float only as one float op's result), so a resume
    // through such an arm dispatches these; their handlers are wired below.
    insns.insert(
        "convert_float_bytes_to_longlong/f>i".to_string(),
        majit_jitcode::insns::BC_CONVERT_FLOAT_BYTES_TO_LONGLONG,
    );
    insns.insert(
        "convert_longlong_bytes_to_float/i>f".to_string(),
        majit_jitcode::insns::BC_CONVERT_LONGLONG_BYTES_TO_FLOAT,
    );
    insns.insert(
        "setarrayitem_gc_i/riid".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_I,
    );
    insns.insert(
        "setarrayitem_gc_r/rird".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_R,
    );
    insns.insert(
        "setarrayitem_gc_f/rifd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_F,
    );
    // Constant-operand "_C" forms, emitted when the array length, store
    // index, or store value is a compile-time constant — a fixed-size
    // tuple/list build (`new_array_clear/cd>r`) and its element stores
    // (`setarrayitem_gc_r/rcrd`, const index; `setarrayitem_gc_i/ricd`,
    // const int value).Whether the assembler emits the `_C` or
    // the register form depends on constant folding during
    // `make_jitcodes`, so a guard failure resuming forward through a
    // constant-size build must dispatch these too. Their `bhimpl_*`
    // handlers are already wired in `wire_bhimpl_handlers` below; without
    // the map entry `wire_handler` silently no-ops and the byte stays
    // unwired (`dispatch_step` panics on `0xd8`).
    insns.insert(
        "setarrayitem_gc_r/rcrd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_R_C,
    );
    insns.insert(
        "setarrayitem_gc_i/ricd".to_string(),
        majit_jitcode::insns::BC_SETARRAYITEM_GC_I_C,
    );
    insns.insert(
        "new_array_clear/cd>r".to_string(),
        majit_jitcode::insns::BC_NEW_ARRAY_CLEAR_C,
    );
    // Overflow-checked arithmetic (`int_{add,sub,mul}_jump_if_ovf`): a guard
    // failure on `GuardNoOverflow`/`GuardOverflow` resumes forward through the
    // fused op, so the blackhole must dispatch it. The `bhimpl_*` handlers are
    // wired in `wire_bhimpl_handlers` below; without the map entry `wire_handler`
    // silently no-ops and the byte stays unwired.
    insns.insert(
        "int_add_jump_if_ovf/Lii>i".to_string(),
        majit_jitcode::insns::BC_INT_ADD_JUMP_IF_OVF,
    );
    insns.insert(
        "int_sub_jump_if_ovf/Lii>i".to_string(),
        majit_jitcode::insns::BC_INT_SUB_JUMP_IF_OVF,
    );
    insns.insert(
        "int_mul_jump_if_ovf/Lii>i".to_string(),
        majit_jitcode::insns::BC_INT_MUL_JUMP_IF_OVF,
    );
    // The remaining opnames the build-time `pipeline.insns` records as
    // actually emitted (`build_emitted_insns()`) that this curated set did not
    // cover.  Upstream never has this gap: `setup_insns(asm.insns)`
    // (`blackhole.py`) registers exactly what the assembler emitted, so
    // its table spans the reachable bytecode universe by construction.  Here a
    // missing entry is a live `dispatch_step` unwired-opcode panic that fires
    // only once a forward resume happens to land on the byte.
    //
    // All of them emit canonical operands matching their wired decoders — a
    // 1-byte register per `r`/`i`, a 2-byte little-endian descr index per `d`
    // (`assembler.rs` `OpKind::{ArrayLen, New, NewWithVtable}` and the generic
    // `UnaryOp` path vs `handler_arraylen_gc`, `handler_new`,
    // `handler_new_with_vtable`, `bhhandler_r_i!`) — and `arraylen_gc` / `new`
    // / `new_with_vtable` read `bh.cpu`, which this builder sets above.
    for (key, byte) in [
        ("arraylen_gc/rd>i", majit_jitcode::insns::BC_ARRAYLEN_GC),
        (
            "getinteriorfield_gc_i/rid>i",
            majit_jitcode::insns::BC_GETINTERIORFIELD_GC_I,
        ),
        (
            "getinteriorfield_gc_r/rid>r",
            majit_jitcode::insns::BC_GETINTERIORFIELD_GC_R,
        ),
        (
            "getinteriorfield_gc_f/rid>f",
            majit_jitcode::insns::BC_GETINTERIORFIELD_GC_F,
        ),
        (
            "setinteriorfield_gc_i/riid",
            majit_jitcode::insns::BC_SETINTERIORFIELD_GC_I,
        ),
        (
            "setinteriorfield_gc_r/rird",
            majit_jitcode::insns::BC_SETINTERIORFIELD_GC_R,
        ),
        (
            "setinteriorfield_gc_f/rifd",
            majit_jitcode::insns::BC_SETINTERIORFIELD_GC_F,
        ),
        (
            "cast_float_to_int/f>i",
            majit_jitcode::insns::BC_CAST_FLOAT_TO_INT,
        ),
        (
            "cast_ptr_to_int/r>i",
            majit_jitcode::insns::BC_CAST_PTR_TO_INT,
        ),
        // `rbuiltin.py rtype_cast_int_to_ptr` → `cast_int_to_ptr/i>r`.
        // Sibling of `cast_ptr_to_int/r>i`; `wire_bhimpl_handlers` already
        // has `handler_cast_int_to_ptr` (`blackhole.py bhimpl_cast_int_to_ptr`).
        (
            "cast_int_to_ptr/i>r",
            majit_jitcode::insns::BC_CAST_INT_TO_PTR,
        ),
        (
            "int_isconstant/i>i",
            majit_jitcode::insns::BC_INT_ISCONSTANT,
        ),
        (
            "ref_isconstant/r>i",
            majit_jitcode::insns::BC_REF_ISCONSTANT,
        ),
        ("ref_isvirtual/r>i", majit_jitcode::insns::BC_REF_ISVIRTUAL),
        ("new/d>r", majit_jitcode::insns::BC_NEW),
        (
            "new_with_vtable/d>r",
            majit_jitcode::insns::BC_NEW_WITH_VTABLE,
        ),
        // The header read `jtransform.rs rewrite_op_getfield` replaces,
        // in the ref bank (`OpKind::GuardClass`: one ref register in, the
        // class out) — emitted once a descended helper reads `ob_type`
        // through a graph the codewriter looks inside.
        ("guard_class/r>r", majit_jitcode::insns::BC_GUARD_CLASS_R),
        ("strlen/r>i", majit_jitcode::insns::BC_STRLEN),
        ("strgetitem/ri>i", majit_jitcode::insns::BC_STRGETITEM),
        ("strgetitem/rc>i", majit_jitcode::insns::BC_STRGETITEM_C),
        // `(lo..=hi).contains` emits `int_between/iii>i`. The decoder is
        // `bhhandler_iii_i!` (`[a][b][c][dst]`), already wired below.
        ("int_between/iii>i", majit_jitcode::insns::BC_INT_BETWEEN),
        // `jtransform.py rewrite_op_getfield` emits this ahead of a
        // quasi-immutable getfield. The handler below is the no-op step;
        // `wire_handler` binds it once this map contains the key.
        (
            "record_quasiimmut_field/rdd",
            majit_jitcode::insns::BC_RECORD_QUASIIMMUT_FIELD,
        ),
    ] {
        insns.insert(key.to_string(), byte);
    }
    // `handle_recursive_call` emits `recursive_call_v/iIRFIRF`, a key the
    // assembler numbers dynamically; `handler_recursive_call_v` is wired
    // below and no-ops until the key is in this map (`blackhole.py
    // bhimpl_recursive_call_v`).
    for (key, byte) in dynamic_insns {
        insns.insert((*key).to_string(), *byte);
    }
    builder.setup_insns(&insns);
    // `setup_insns` already derives `op_live` and `op_catch_exception`
    // from the registered canonical subset above.  `rvmprof_code/ii` is
    // not registered in this minimal builder yet, but blackhole exception
    // and live handling expects these cached values to stay in the same
    // fixed-byte space as pyre's runtime opcodes.  Reapply the canonical
    // `BC_*` constants explicitly so this builder stays synchronized as
    // the registered subset changes.
    builder.setup_cached_control_opcodes(
        majit_jitcode::insns::BC_LIVE as i32,
        majit_jitcode::insns::BC_CATCH_EXCEPTION as i32,
        majit_jitcode::insns::BC_RVMPROF_CODE as i32,
    );
    wire_bhimpl_handlers(&mut builder);
    // Strict-coverage gate: enforce setup-time parity with RPython
    // `blackhole.py setup_insns` which raises `AttributeError` from
    // `_get_method(name, argcodes)` whenever an opname has no matching
    // `bhimpl_*`.  pyre's previous behaviour deferred the failure to
    // the `dispatch_step` placeholder panic, which surfaces only when
    // the unwired byte is actually executed; this gate matches the
    // upstream contract by failing the moment the builder is
    // assembled.
    let unwired = builder.unwired_opnames();
    if !unwired.is_empty() {
        panic!(
            "build_inline_call_only_bh_builder: {} insns opnames have no \
             bhimpl_* handler (RPython `blackhole.py:66` raises \
             AttributeError here): {:?}",
            unwired.len(),
            unwired,
        );
    }
    builder
}

/// Wire all currently-ported bhimpl methods into a `BlackholeInterpBuilder`'s
/// dispatch table. Called once after `setup_insns`.
///
/// RPython builds all handlers in `setup_insns` via `_get_method`; pyre
/// wires them incrementally as methods are ported from RPython.
pub fn wire_bhimpl_handlers(builder: &mut BlackholeInterpBuilder) {
    // @arguments("i", returns="i") pattern
    builder.wire_handler("int_same_as/i>i", handler_int_same_as);
    builder.wire_handler("int_neg/i>i", handler_int_neg);
    builder.wire_handler("int_invert/i>i", handler_int_invert);
    builder.wire_handler("int_is_true/i>i", handler_int_is_true);
    builder.wire_handler("int_is_zero/i>i", handler_int_is_zero);
    builder.wire_handler("int_force_ge_zero/i>i", handler_int_force_ge_zero);

    // @arguments("i", "i", returns="i") pattern
    builder.wire_handler("int_add/ii>i", handler_int_add);
    builder.wire_handler("int_sub/ii>i", handler_int_sub);
    // pyre-only primitives — see handler comments for rationale.
    builder.wire_handler("int_add_assign/ii>i", handler_int_add_assign);
    builder.wire_handler("int_sub_assign/ii>i", handler_int_sub_assign);
    builder.wire_handler("int_deref/i>i", handler_int_deref);
    builder.wire_handler("abort/", handler_abort_marker);

    // pyre-only state_field family (no RPython counterpart). The handlers do
    // real work: they map a logical field/array index to its flat register
    // slot via `state_field_layout` and move the value between that slot
    // (seeded by the resume reader from the guard resumedata) and a working
    // register, advancing position past the operands.  This re-fills the
    // state into the blackhole register file the way RPython's setarg/getarg
    // bytecodes feed its own resume.
    builder.wire_handler("load_state_field_ref/dr", handler_load_state_field_ref_dr);
    builder.wire_handler("store_state_field_ref/dr", handler_store_state_field_ref_dr);
    builder.wire_handler("load_state_field/di", handler_load_state_field_di);
    builder.wire_handler("store_state_field/di", handler_store_state_field_di);
    builder.wire_handler(
        "load_state_field_float/df",
        handler_load_state_field_float_df,
    );
    builder.wire_handler(
        "store_state_field_float/df",
        handler_store_state_field_float_df,
    );

    // jit_merge_point + abort_permanent — bodies live on
    // `BlackholeInterpreter::bhimpl_jit_merge_point` / `bhimpl_abort_permanent`
    // and are shared with direct test fixtures that bypass `acquire_interp`.
    builder.wire_handler("jit_merge_point/iIRFIRF", handler_jit_merge_point_i);
    builder.wire_handler("jit_merge_point/cIRFIRF", handler_jit_merge_point_c);
    builder.wire_handler("abort_permanent/", handler_abort_permanent);

    builder.wire_handler("int_mul/ii>i", handler_int_mul);
    // `int_div` / `int_floordiv` / `int_mod` are intentionally NOT
    // wired: `jtransform.py:576-577 rewrite_op_int_floordiv =
    // _do_builtin_call` rewrites these primitives to
    // `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)`
    // before jitcode emission, so RPython's blackhole never sees the
    // bare op.  Pyre's runtime path mirrors this via
    // `codegen.rs` emitting `CallI(ll_int_py_div, ...)`
    // directly.  Wiring `handler_int_floordiv` here would silently
    // re-introduce the upstream-absent bytecode dispatch path.
    builder.wire_handler("int_and/ii>i", handler_int_and);
    builder.wire_handler("int_or/ii>i", handler_int_or);
    builder.wire_handler("int_xor/ii>i", handler_int_xor);
    builder.wire_handler("int_rshift/ii>i", handler_int_rshift);
    builder.wire_handler("int_lshift/ii>i", handler_int_lshift);
    builder.wire_handler("uint_rshift/ii>i", handler_uint_rshift);
    builder.wire_handler("int_lt/ii>i", handler_int_lt);
    builder.wire_handler("int_le/ii>i", handler_int_le);
    builder.wire_handler("int_eq/ii>i", handler_int_eq);
    builder.wire_handler("int_ne/ii>i", handler_int_ne);
    builder.wire_handler("int_gt/ii>i", handler_int_gt);
    builder.wire_handler("int_ge/ii>i", handler_int_ge);
    for (key, handler) in [
        ("int_add/ci>i", handler_int_add_ci as BhOpcodeHandler),
        ("int_add/ic>i", handler_int_add_ic),
        ("int_add/cc>i", handler_int_add_cc),
        ("int_sub/ci>i", handler_int_sub_ci),
        ("int_sub/ic>i", handler_int_sub_ic),
        ("int_sub/cc>i", handler_int_sub_cc),
        ("int_and/ci>i", handler_int_and_ci),
        ("int_and/ic>i", handler_int_and_ic),
        ("int_and/cc>i", handler_int_and_cc),
        ("int_eq/ci>i", handler_int_eq_ci),
        ("int_eq/ic>i", handler_int_eq_ic),
        ("int_eq/cc>i", handler_int_eq_cc),
        ("int_ne/ci>i", handler_int_ne_ci),
        ("int_ne/ic>i", handler_int_ne_ic),
        ("int_ne/cc>i", handler_int_ne_cc),
        ("int_lt/ci>i", handler_int_lt_ci),
        ("int_lt/ic>i", handler_int_lt_ic),
        ("int_lt/cc>i", handler_int_lt_cc),
        ("int_le/ci>i", handler_int_le_ci),
        ("int_le/ic>i", handler_int_le_ic),
        ("int_le/cc>i", handler_int_le_cc),
        ("int_gt/ci>i", handler_int_gt_ci),
        ("int_gt/ic>i", handler_int_gt_ic),
        ("int_gt/cc>i", handler_int_gt_cc),
        ("int_ge/ci>i", handler_int_ge_ci),
        ("int_ge/ic>i", handler_int_ge_ic),
        ("int_ge/cc>i", handler_int_ge_cc),
    ] {
        builder.wire_handler(key, handler);
    }

    // Copy operations
    builder.wire_handler("int_copy/i>i", handler_int_copy);
    // `int_copy/c>i` — USE_C_FORM short-const source (`assembler.py`).
    builder.wire_handler("int_copy/c>i", handler_int_copy_c);
    // pyre-only abort placeholder emitted by `Assembler::encode_op`'s
    // default branch for `OpKind::Abort { .. }`.
    builder.wire_handler("abort/>i", handler_abort_result_marker_i);
    builder.wire_handler("abort/>r", handler_abort_result_marker_r);

    // Control flow
    builder.wire_handler("live/", handler_live);
    builder.wire_handler("goto/L", handler_goto);
    builder.wire_handler("goto_if_not/iL", handler_goto_if_not);

    // Unsigned comparisons (blackhole.py:571-582)
    builder.wire_handler("uint_lt/ii>i", handler_uint_lt);
    builder.wire_handler("uint_le/ii>i", handler_uint_le);
    builder.wire_handler("uint_gt/ii>i", handler_uint_gt);
    builder.wire_handler("uint_ge/ii>i", handler_uint_ge);

    // Float arithmetic (blackhole.py:676-718)
    builder.wire_handler("float_neg/f>f", handler_float_neg);
    builder.wire_handler("float_abs/f>f", handler_float_abs);
    builder.wire_handler("float_add/ff>f", handler_float_add);
    builder.wire_handler("float_sub/ff>f", handler_float_sub);
    builder.wire_handler("float_mul/ff>f", handler_float_mul);
    builder.wire_handler("float_truediv/ff>f", handler_float_truediv);

    // Float comparisons → int result (blackhole.py:720-749)
    builder.wire_handler("float_lt/ff>i", handler_float_lt);
    builder.wire_handler("float_le/ff>i", handler_float_le);
    builder.wire_handler("float_eq/ff>i", handler_float_eq);
    builder.wire_handler("float_ne/ff>i", handler_float_ne);
    builder.wire_handler("float_gt/ff>i", handler_float_gt);
    builder.wire_handler("float_ge/ff>i", handler_float_ge);

    // Ref operations (blackhole.py:584-610)
    builder.wire_handler("ptr_eq/rr>i", handler_ptr_eq);
    builder.wire_handler("ptr_ne/rr>i", handler_ptr_ne);
    builder.wire_handler("instance_ptr_eq/rr>i", handler_instance_ptr_eq);
    builder.wire_handler("instance_ptr_ne/rr>i", handler_instance_ptr_ne);
    builder.wire_handler("ptr_iszero/r>i", handler_ptr_iszero);
    builder.wire_handler("ptr_nonzero/r>i", handler_ptr_nonzero);

    // Copy operations (ref + float)
    builder.wire_handler("ref_copy/r>r", handler_ref_copy);
    builder.wire_handler("float_copy/f>f", handler_float_copy);

    // Conditional jumps (blackhole.py:871-920)
    builder.wire_handler("goto_if_not_int_lt/iiL", handler_goto_if_not_int_lt);
    builder.wire_handler("goto_if_not_int_le/iiL", handler_goto_if_not_int_le);
    builder.wire_handler("goto_if_not_int_eq/iiL", handler_goto_if_not_int_eq);
    builder.wire_handler("goto_if_not_int_ne/iiL", handler_goto_if_not_int_ne);
    builder.wire_handler("goto_if_not_int_gt/iiL", handler_goto_if_not_int_gt);
    builder.wire_handler("goto_if_not_int_ge/iiL", handler_goto_if_not_int_ge);
    for (key, handler) in [
        (
            "goto_if_not_int_lt/ciL",
            handler_goto_if_not_int_lt_ci as BhOpcodeHandler,
        ),
        ("goto_if_not_int_lt/icL", handler_goto_if_not_int_lt_ic),
        ("goto_if_not_int_lt/ccL", handler_goto_if_not_int_lt_cc),
        ("goto_if_not_int_le/ciL", handler_goto_if_not_int_le_ci),
        ("goto_if_not_int_le/icL", handler_goto_if_not_int_le_ic),
        ("goto_if_not_int_le/ccL", handler_goto_if_not_int_le_cc),
        ("goto_if_not_int_eq/ciL", handler_goto_if_not_int_eq_ci),
        ("goto_if_not_int_eq/icL", handler_goto_if_not_int_eq_ic),
        ("goto_if_not_int_eq/ccL", handler_goto_if_not_int_eq_cc),
        ("goto_if_not_int_ne/ciL", handler_goto_if_not_int_ne_ci),
        ("goto_if_not_int_ne/icL", handler_goto_if_not_int_ne_ic),
        ("goto_if_not_int_ne/ccL", handler_goto_if_not_int_ne_cc),
        ("goto_if_not_int_gt/ciL", handler_goto_if_not_int_gt_ci),
        ("goto_if_not_int_gt/icL", handler_goto_if_not_int_gt_ic),
        ("goto_if_not_int_gt/ccL", handler_goto_if_not_int_gt_cc),
        ("goto_if_not_int_ge/ciL", handler_goto_if_not_int_ge_ci),
        ("goto_if_not_int_ge/icL", handler_goto_if_not_int_ge_ic),
        ("goto_if_not_int_ge/ccL", handler_goto_if_not_int_ge_cc),
    ] {
        builder.wire_handler(key, handler);
    }
    // upstream `flatten.py:247` registers opname `goto_if_not`
    // (the `_int_is_true` suffix is a Python class-attribute alias of
    // the bhimpl function, not a separate opname — `blackhole.py:913`).
    builder.wire_handler("goto_if_not/iL", handler_goto_if_not);

    // Guard values — no-op in blackhole (blackhole.py:648-656)
    builder.wire_handler("int_guard_value/i", handler_int_guard_value);
    builder.wire_handler("ref_guard_value/r", handler_ref_guard_value);
    builder.wire_handler("float_guard_value/f", handler_float_guard_value);

    // Push/pop tmpreg (blackhole.py:661-679)
    builder.wire_handler("int_push/i", handler_int_push);
    builder.wire_handler("ref_push/r", handler_ref_push);
    builder.wire_handler("float_push/f", handler_float_push);
    builder.wire_handler("int_pop/>i", handler_int_pop);
    builder.wire_handler("ref_pop/>r", handler_ref_pop);
    builder.wire_handler("float_pop/>f", handler_float_pop);

    // Record — no-op (blackhole.py:616-636)
    builder.wire_handler("record_exact_class/ri", handler_record_exact_class);
    builder.wire_handler("record_exact_value_r/rr", handler_record_exact_value_r);
    builder.wire_handler("record_exact_value_i/ii", handler_record_exact_value_i);

    // Cast operations (blackhole.py:800-831)
    builder.wire_handler("cast_float_to_int/f>i", handler_cast_float_to_int);
    builder.wire_handler("cast_int_to_float/i>f", handler_cast_int_to_float);
    // No `cast_bool_to_int` / `cast_bool_to_float` / `float_is_true`
    // backend handlers — RPython jtransform canonicalises these
    // upstream of the assembler:
    //   * `cast_bool_to_int` is dropped entirely
    //     (`jtransform.py def rewrite_op_cast_bool_to_int(self, op): pass`,
    //     mirrored at `codewriter/jtransform.rs` `same_as`-family arm).
    //   * `cast_bool_to_float` → `cast_int_to_float`
    //     (`jtransform.py` rename pass).
    //   * `float_is_true` → `float_ne(x, 0.0)`
    //     (`jtransform.py`, mirrored in `jtransform.rs`'s `optimize_block`).
    // Adding backend opcodes for these would diverge from upstream's
    // "shrink the opcode table at jtransform" contract.

    // int_signext (blackhole.py:566-569)
    builder.wire_handler("int_signext/ii>i", handler_int_signext);

    // Overflow ops (blackhole.py:478-497)
    builder.wire_handler("int_add_jump_if_ovf/Lii>i", handler_int_add_jump_if_ovf);
    builder.wire_handler("int_sub_jump_if_ovf/Lii>i", handler_int_sub_jump_if_ovf);
    builder.wire_handler("int_mul_jump_if_ovf/Lii>i", handler_int_mul_jump_if_ovf);

    // Misc simple ops
    builder.wire_handler("assert_not_none/r", handler_assert_not_none);
    builder.wire_handler("virtual_ref/r>r", handler_virtual_ref);
    builder.wire_handler("virtual_ref_finish/r", handler_virtual_ref_finish);
    builder.wire_handler("loop_header/i", handler_loop_header);
    builder.wire_handler("ref_isconstant/r>i", handler_ref_isconstant);
    builder.wire_handler("ref_isvirtual/r>i", handler_ref_isvirtual);
    // `jtransform.py` emits `{kind}_isvirtual`; pyjitpl only names
    // `opimpl_ref_isvirtual`. An int-kind leftover uses the same always-
    // false blackhole body (`bhimpl_ref_isvirtual`) with the `i>i` decoder.
    builder.wire_handler("int_isvirtual/i>i", handler_int_isconstant);
    builder.wire_handler(
        "goto_if_not_int_is_zero/iL",
        handler_goto_if_not_int_is_zero,
    );
    builder.wire_handler("goto_if_not_ptr_iszero/rL", handler_goto_if_not_ptr_iszero);
    builder.wire_handler(
        "goto_if_not_ptr_nonzero/rL",
        handler_goto_if_not_ptr_nonzero,
    );
    // `bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not` alias
    // (`blackhole.py`) — both opcode bytes route to the same body.
    builder.wire_handler("goto_if_not_int_is_true/iL", handler_goto_if_not);
    builder.wire_handler(
        "goto_if_exception_mismatch/iL",
        handler_goto_if_exception_mismatch,
    );
    builder.wire_handler("unreachable/", handler_unreachable);

    // Field operations (blackhole.py bhimpl_getfield_gc_i).
    //
    // Canonical `/rd>X` and `/rXd` are the RPython-exact keys.
    // `bhimpl_getfield_gc_*` takes the struct in the Ref bank
    // (`@arguments("cpu", "r", "d", returns="X")`).  An Int-bank base
    // is `getfield_raw_*` (`@arguments("cpu", "i", "d", returns="X")`),
    // not a `getfield_gc` alias — `promote_gc_field_bases` rehomes a
    // GC pointer that arrived as Signed so the assembler never emits
    // `/id>X` for `getfield_gc`.
    builder.wire_handler("getfield_gc_i/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f/rd>f", handler_getfield_gc_f);
    builder.wire_handler("getfield_gc_i_pure/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r_pure/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_pure/rd>f", handler_getfield_gc_f);
    builder.wire_handler("setfield_gc_i/rid", handler_setfield_gc_i);
    // USE_C_FORM short value (`assembler.py`): small ConstInt
    // store value inline as one signed byte.
    builder.wire_handler("setfield_gc_i/rcd", handler_setfield_gc_i_c);
    builder.wire_handler("setfield_gc_r/rrd", handler_setfield_gc_r);
    builder.wire_handler("setfield_gc_f/rfd", handler_setfield_gc_f);
    builder.wire_handler("arraylen_gc/rd>i", handler_arraylen_gc);

    // Array item operations (`bhimpl_getarrayitem_gc_*`).
    // The GC form takes the array in the Ref bank.
    // An Int-bank base is `getarrayitem_raw_*`.
    builder.wire_handler("getarrayitem_gc_i/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r/rid>r", handler_getarrayitem_gc_r);
    builder.wire_handler("getarrayitem_gc_i/rcd>i", handler_getarrayitem_gc_i_c);
    builder.wire_handler("getarrayitem_gc_r/rcd>r", handler_getarrayitem_gc_r_c);
    builder.wire_handler("getarrayitem_gc_i_pure/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r_pure/rid>r", handler_getarrayitem_gc_r);
    builder.wire_handler("setarrayitem_gc_i/riid", handler_setarrayitem_gc_i);
    builder.wire_handler("setarrayitem_gc_i/ricd", handler_setarrayitem_gc_i_c);
    builder.wire_handler("setarrayitem_gc_i/rcid", handler_setarrayitem_gc_i_index_c);
    builder.wire_handler("setarrayitem_gc_i/rccd", handler_setarrayitem_gc_i_both_c);
    builder.wire_handler("setarrayitem_gc_r/rird", handler_setarrayitem_gc_r);
    builder.wire_handler("setarrayitem_gc_r/rcrd", handler_setarrayitem_gc_r_c);

    // Raw field operations (blackhole.py:1464-1502)
    builder.wire_handler("getfield_raw_i/id>i", handler_getfield_raw_i);
    builder.wire_handler("getfield_raw_f/id>f", handler_getfield_raw_f);
    builder.wire_handler("setfield_raw_i/iid", handler_setfield_raw_i);
    builder.wire_handler("setfield_raw_f/ifd", handler_setfield_raw_f);

    // Greenfield aliases
    builder.wire_handler("getfield_gc_i_greenfield/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r_greenfield/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_greenfield/rd>f", handler_getfield_gc_f);

    // New operations (blackhole.py:1301-1327)
    builder.wire_handler("new/d>r", handler_new);
    builder.wire_handler("new_with_vtable/d>r", handler_new_with_vtable);
    builder.wire_handler("new_array/id>r", handler_new_array);
    builder.wire_handler("new_array/cd>r", handler_new_array_c);
    builder.wire_handler("new_array_clear/id>r", handler_new_array_clear);
    builder.wire_handler("new_array_clear/cd>r", handler_new_array_clear_c);
    // String operations (blackhole.py bhimpl_getlistitem_gc_r)
    builder.wire_handler("strlen/r>i", handler_strlen);
    builder.wire_handler("strgetitem/ri>i", handler_strgetitem);
    builder.wire_handler("strgetitem/rc>i", handler_strgetitem_c);
    builder.wire_handler("strsetitem/rii", handler_strsetitem);
    builder.wire_handler("strsetitem/rci", handler_strsetitem_short::<true, false>);
    builder.wire_handler("strsetitem/ric", handler_strsetitem_short::<false, true>);
    builder.wire_handler("strsetitem/rcc", handler_strsetitem_short::<true, true>);
    builder.wire_handler("newstr/i>r", handler_newstr);
    builder.wire_handler("newstr/c>r", handler_newstr_c);
    builder.wire_handler("unicodelen/r>i", handler_unicodelen);
    builder.wire_handler("unicodegetitem/ri>i", handler_unicodegetitem);
    builder.wire_handler("unicodesetitem/rii", handler_unicodesetitem);
    builder.wire_handler("newunicode/i>r", handler_newunicode);

    // Exception handling (blackhole.py:969-975)
    builder.wire_handler("catch_exception/L", handler_catch_exception);

    // Interior field operations (blackhole.py:1411-1429)
    // _r/_f deferred until Backend gains those variants.
    builder.wire_handler("getinteriorfield_gc_i/rid>i", handler_getinteriorfield_gc_i);
    builder.wire_handler("setinteriorfield_gc_i/riid", handler_setinteriorfield_gc_i);

    // Residual call operations (blackhole.py:1224-1255)
    builder.wire_handler("residual_call_irf_i/iIRFd>i", handler_residual_call_irf_i);
    builder.wire_handler("residual_call_irf_r/iIRFd>r", handler_residual_call_irf_r);
    builder.wire_handler("residual_call_irf_f/iIRFd>f", handler_residual_call_irf_f);
    builder.wire_handler("residual_call_irf_v/iIRFd", handler_residual_call_irf_v);
    builder.wire_handler("residual_call_ir_i/iIRd>i", handler_residual_call_ir_i);
    builder.wire_handler("residual_call_ir_r/iIRd>r", handler_residual_call_ir_r);
    builder.wire_handler("residual_call_ir_v/iIRd", handler_residual_call_ir_v);
    builder.wire_handler("residual_call_r_i/iRd>i", handler_residual_call_r_i);
    builder.wire_handler("residual_call_r_r/iRd>r", handler_residual_call_r_r);
    builder.wire_handler("residual_call_r_v/iRd", handler_residual_call_r_v);
    // Misc no-ops (blackhole.py:1017-1049)
    builder.wire_handler("jit_debug/riiii", handler_jit_debug);
    builder.wire_handler("jit_enter_portal_frame/i", handler_jit_enter_portal_frame);
    builder.wire_handler("jit_leave_portal_frame/", handler_jit_leave_portal_frame);

    // Float conditional jumps (blackhole.py:751-798)
    builder.wire_handler("goto_if_not_float_lt/ffL", handler_goto_if_not_float_lt);
    builder.wire_handler("goto_if_not_float_le/ffL", handler_goto_if_not_float_le);
    builder.wire_handler("goto_if_not_float_eq/ffL", handler_goto_if_not_float_eq);
    builder.wire_handler("goto_if_not_float_ne/ffL", handler_goto_if_not_float_ne);
    builder.wire_handler("goto_if_not_float_gt/ffL", handler_goto_if_not_float_gt);
    builder.wire_handler("goto_if_not_float_ge/ffL", handler_goto_if_not_float_ge);
    builder.wire_handler("goto_if_not_ptr_eq/rrL", handler_goto_if_not_ptr_eq);
    builder.wire_handler("goto_if_not_ptr_ne/rrL", handler_goto_if_not_ptr_ne);

    // Assert/isconstant (no-ops in blackhole)
    builder.wire_handler("int_assert_green/i", handler_int_assert_green);
    builder.wire_handler("ref_assert_green/r", handler_ref_assert_green);
    builder.wire_handler("float_assert_green/f", handler_float_assert_green);
    builder.wire_handler("int_isconstant/i>i", handler_int_isconstant);
    builder.wire_handler("float_isconstant/f>i", handler_float_isconstant);

    // Misc integer ops
    builder.wire_handler("uint_mul_high/ii>i", handler_uint_mul_high);
    builder.wire_handler("int_between/iii>i", handler_int_between);

    // String hashing
    builder.wire_handler("strhash/r>i", handler_strhash);
    builder.wire_handler("unicodehash/r>i", handler_unicodehash);

    // Float <-> longlong / singlefloat conversions
    builder.wire_handler(
        "convert_float_bytes_to_longlong/f>i",
        handler_convert_float_bytes_to_longlong,
    );
    builder.wire_handler(
        "convert_longlong_bytes_to_float/i>f",
        handler_convert_longlong_bytes_to_float,
    );
    builder.wire_handler(
        "cast_float_to_singlefloat/f>i",
        handler_cast_float_to_singlefloat,
    );
    builder.wire_handler(
        "cast_singlefloat_to_float/i>f",
        handler_cast_singlefloat_to_float,
    );

    // Misc
    // RPython `rpython/jit/metainterp/blackhole.py:1546-1548`:
    //   @arguments("r")
    //   def bhimpl_hint_force_virtualizable(r): pass
    // Canonical key is `hint_force_virtualizable/r`, not the previous
    // pyre-invented `/rd` shape that inserted a phantom descr slot.
    builder.wire_handler(
        "hint_force_virtualizable/r",
        handler_hint_force_virtualizable,
    );
    // RPython `rpython/jit/metainterp/blackhole.py:1558-1560`:
    //   @arguments("cpu", "r", returns="i")
    //   def bhimpl_guard_class(cpu, struct): return cpu.bh_classof(struct)
    // Canonical key is `guard_class/r>i`; the previous `/ri` shape was
    // a pyre-invented bigram that omitted the `>i` return marker.
    builder.wire_handler("guard_class/r>i", handler_guard_class);
    // The same op with its result in the ref bank — the bank of the header
    // read `jtransform.rs rewrite_op_getfield` replaced (`BC_GUARD_CLASS_R`).
    builder.wire_handler("guard_class/r>r", handler_guard_class_r);
    // RPython `rpython/jit/metainterp/blackhole.py:1537-1539`:
    //   @arguments("r", "d", "d")
    //   def bhimpl_record_quasiimmut_field(struct, fielddescr, mutatefielddescr):
    builder.wire_handler(
        "record_quasiimmut_field/rdd",
        handler_record_quasiimmut_field,
    );
    // Compatibility fallback for the obsolete name-based vtable opcode.
    builder.wire_handler("vtable_method_ptr/rd>i", handler_vtable_method_ptr_bail);
    builder.wire_handler(
        "jit_force_quasi_immutable/rd",
        handler_jit_force_quasi_immutable,
    );
    builder.wire_handler(
        "record_known_result_i_ir_v/iiIRd",
        handler_record_known_result_i_ir_v,
    );
    builder.wire_handler(
        "record_known_result_r_ir_v/riIRd",
        handler_record_known_result_r_ir_v,
    );
    builder.wire_handler("str_guard_value/rid>r", handler_str_guard_value);
    builder.wire_handler("rvmprof_code/ii", handler_rvmprof_code);
    builder.wire_handler("copystrcontent/rriii", handler_copystrcontent);
    builder.wire_handler(
        "copystrcontent/rrcii",
        handler_copystrcontent_short::<true, false, false>,
    );
    builder.wire_handler(
        "copystrcontent/rrici",
        handler_copystrcontent_short::<false, true, false>,
    );
    builder.wire_handler(
        "copystrcontent/rriic",
        handler_copystrcontent_short::<false, false, true>,
    );
    builder.wire_handler(
        "copystrcontent/rrcci",
        handler_copystrcontent_short::<true, true, false>,
    );
    builder.wire_handler(
        "copystrcontent/rrcic",
        handler_copystrcontent_short::<true, false, true>,
    );
    builder.wire_handler(
        "copystrcontent/rricc",
        handler_copystrcontent_short::<false, true, true>,
    );
    builder.wire_handler(
        "copystrcontent/rrccc",
        handler_copystrcontent_short::<true, true, true>,
    );
    builder.wire_handler("copyunicodecontent/rriii", handler_copyunicodecontent);
    builder.wire_handler("current_trace_length/>i", handler_current_trace_length);

    // Exception ops (blackhole.py:976-1009)
    builder.wire_handler("raise/r", handler_raise);
    builder.wire_handler("reraise/", handler_reraise);
    builder.wire_handler("last_exception/>i", handler_last_exception);
    builder.wire_handler("last_exc_value/>r", handler_last_exc_value);
    builder.wire_handler(
        "goto_if_exception_mismatch/iL",
        handler_goto_if_exception_mismatch,
    );
    builder.wire_handler("debug_fatalerror/r", handler_debug_fatalerror);

    // Cast ptr<->int (blackhole.py bhimpl_cast_ptr_to_int)
    builder.wire_handler("cast_ptr_to_int/r>i", handler_cast_ptr_to_int);
    builder.wire_handler("cast_int_to_ptr/i>r", handler_cast_int_to_ptr);

    // Vable field operations — canonical `/rd>X` / `/rXd` RPython shape.
    // Every `BlackholeInterpreter.bhimpl_*field_vable_*` declares the base
    // operand as `r`; the assembler rejects an Int-bank base before it can
    // produce an unrooted `/id>X` or `/iXd` bytecode. The result/value suffix
    // still follows the concrete register kind, and the dead `_v` sentinel
    // form is not emitted.
    builder.wire_handler("getfield_vable_i/rd>i", handler_getfield_vable_i);
    builder.wire_handler("getfield_vable_r/rd>r", handler_getfield_vable_r);
    builder.wire_handler("getfield_vable_f/rd>f", handler_getfield_vable_f);
    builder.wire_handler("setfield_vable_i/rid", handler_setfield_vable_i);
    builder.wire_handler("setfield_vable_i_imm/rddd", handler_setfield_vable_i_imm);
    builder.wire_handler("setfield_vable_r/rrd", handler_setfield_vable_r);
    builder.wire_handler("setfield_vable_f/rfd", handler_setfield_vable_f);
    builder.wire_handler("getarrayitem_vable_i/ridd>i", handler_getarrayitem_vable_i);
    builder.wire_handler("getarrayitem_vable_r/ridd>r", handler_getarrayitem_vable_r);
    builder.wire_handler("setarrayitem_vable_i/riidd", handler_setarrayitem_vable_i);
    builder.wire_handler("setarrayitem_vable_r/rirdd", handler_setarrayitem_vable_r);
    builder.wire_handler("arraylen_vable/rdd>i", handler_arraylen_vable);
    builder.wire_handler("arraybase_vable/rdd>i", handler_arraybase_vable);
    builder.wire_handler("getarrayitem_raw_i/iid>i", handler_getarrayitem_raw_i);
    builder.wire_handler("setarrayitem_raw_i/iiid", handler_setarrayitem_raw_i);
    builder.wire_handler("conditional_call_ir_v/iiIRd", handler_conditional_call_ir_v);
    builder.wire_handler(
        "conditional_call_value_ir_i/iiIRd>i",
        handler_conditional_call_value_ir_i,
    );
    builder.wire_handler(
        "conditional_call_value_ir_r/riIRd>r",
        handler_conditional_call_value_ir_r,
    );
    builder.wire_handler("getlistitem_gc_i/ridd>i", handler_getlistitem_gc_i);
    builder.wire_handler("getlistitem_gc_r/ridd>r", handler_getlistitem_gc_r);
    builder.wire_handler("setlistitem_gc_i/riidd", handler_setlistitem_gc_i);
    builder.wire_handler("setlistitem_gc_r/rirdd", handler_setlistitem_gc_r);
    builder.wire_handler("switch/id", handler_switch);
    builder.wire_handler("getlistitem_gc_f/ridd>f", handler_getlistitem_gc_f);
    builder.wire_handler("setlistitem_gc_f/rifdd", handler_setlistitem_gc_f);
    builder.wire_handler("check_neg_index/rid>i", handler_check_neg_index);
    builder.wire_handler(
        "check_resizable_neg_index/rid>i",
        handler_check_resizable_neg_index,
    );

    // Float/raw array ops
    builder.wire_handler("getarrayitem_gc_f/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("getarrayitem_gc_f_pure/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("setarrayitem_gc_f/rifd", handler_setarrayitem_gc_f);
    builder.wire_handler("getarrayitem_raw_f/iid>f", handler_getarrayitem_raw_f);
    builder.wire_handler("setarrayitem_raw_f/iifd", handler_setarrayitem_raw_f);
    builder.wire_handler("getfield_raw_r/id>r", handler_getfield_raw_r);
    builder.wire_handler("getinteriorfield_gc_f/rid>f", handler_getinteriorfield_gc_f);
    builder.wire_handler("getinteriorfield_gc_r/rid>r", handler_getinteriorfield_gc_r);
    builder.wire_handler("setinteriorfield_gc_f/rifd", handler_setinteriorfield_gc_f);
    builder.wire_handler("setinteriorfield_gc_r/rird", handler_setinteriorfield_gc_r);
    // RPython `rpython/jit/metainterp/blackhole.py:1518-1534` canonical
    // signatures (`@arguments("cpu", "r", "i", "i", "i", "i", returns="X")`
    // and `@arguments("cpu", "r", "i", "X", "i", "i", "i", "d")`).
    // The previous wire keys carried one extra `i` on loads and one
    // missing `i` on stores relative to the handler body, so they
    // disagreed with both RPython and the handler's own byte math.
    builder.wire_handler("gc_load_indexed_i/riiii>i", handler_gc_load_indexed_i);
    builder.wire_handler("gc_load_indexed_f/riiii>f", handler_gc_load_indexed_f);
    builder.wire_handler("gc_store_indexed_i/riiiiid", handler_gc_store_indexed_i);
    builder.wire_handler("gc_store_indexed_f/rifiiid", handler_gc_store_indexed_f);
    builder.wire_handler("raw_store_i/iiid", handler_raw_store_i);
    builder.wire_handler("raw_store_f/iifd", handler_raw_store_f);
    builder.wire_handler("raw_load_i/iid>i", handler_raw_load_i);
    builder.wire_handler("raw_load_f/iid>f", handler_raw_load_f);
    builder.wire_handler("newlist/idddd>r", handler_newlist);
    builder.wire_handler("newlist_clear/idddd>r", handler_newlist_clear);
    builder.wire_handler("newlist_hint/idddd>r", handler_newlist_hint);
    builder.wire_handler("getarrayitem_vable_f/ridd>f", handler_getarrayitem_vable_f);
    builder.wire_handler("setarrayitem_vable_f/rifdd", handler_setarrayitem_vable_f);

    // Inline call
    builder.wire_handler("inline_call_irf_i/dIRF>i", handler_inline_call_irf_i);
    builder.wire_handler("inline_call_irf_r/dIRF>r", handler_inline_call_irf_r);
    builder.wire_handler("inline_call_irf_f/dIRF>f", handler_inline_call_irf_f);
    builder.wire_handler("inline_call_irf_v/dIRF", handler_inline_call_irf_v);
    builder.wire_handler("inline_call_ir_i/dIR>i", handler_inline_call_ir_i);
    builder.wire_handler("inline_call_ir_r/dIR>r", handler_inline_call_ir_r);
    builder.wire_handler("inline_call_ir_v/dIR", handler_inline_call_ir_v);
    builder.wire_handler("inline_call_r_i/dR>i", handler_inline_call_r_i);
    builder.wire_handler("inline_call_r_r/dR>r", handler_inline_call_r_r);
    builder.wire_handler("inline_call_r_v/dR", handler_inline_call_r_v);

    // TODO: pyre nested-bytecode `inline_call`.  See
    // the comment on `handler_inline_call_nested_ext` for rationale.
    // The canonical `inline_call_{r,ir,irf}_*` keys above are now pinned
    // in `wellknown_bh_insns()` (`BC_INLINE_CALL_*`,
    // `insns.rs`'s `wellknown_bh_insns`) so their handlers dispatch through
    // `setup_insns`-built `_insns` like every other canonical opcode.
    // Byte 17 (`BC_INLINE_CALL`) sits below the canonical range and is
    // exposed via the separate pyre-only `inline_call_nested_ext/P`
    // key in `extension_insns()` — the adapter shape that carries
    // pyre's nested-bytecode payload, distinct from the canonical
    // `dR`/`dIR`/`dIRF` arglists.
    builder.wire_handler("inline_call_nested_ext/P", handler_inline_call_nested_ext);

    // Recursive call
    // RPython `rpython/jit/metainterp/blackhole.py:1101-1132`:
    //   @arguments("self", "i", "I", "R", "F", "I", "R", "F", returns="X")
    // canonical keys `recursive_call_{i,r,f,v}/iIRFIRF{>X,}`. `recursive_call`
    // is not in `assembler.py USE_C_FORM`, so the `c` short-const
    // variant is not valid here.
    builder.wire_handler("recursive_call_i/iIRFIRF>i", handler_recursive_call_i);
    builder.wire_handler("recursive_call_r/iIRFIRF>r", handler_recursive_call_r);
    builder.wire_handler("recursive_call_f/iIRFIRF>f", handler_recursive_call_f);
    builder.wire_handler("recursive_call_v/iIRFIRF", handler_recursive_call_v);

    // Returns
    builder.wire_handler("int_return/i", handler_int_return);
    // `int_return/c` — USE_C_FORM short-const source (`assembler.py`).
    builder.wire_handler("int_return/c", handler_int_return_c);
    builder.wire_handler("ref_return/r", handler_ref_return);
    builder.wire_handler("float_return/f", handler_float_return);
    builder.wire_handler("void_return/", handler_void_return);

    // A3 epic: push/pop family wires the canonical 1-byte register
    // handlers (`handler_int_push`/`pop` and ref/float kin) defined
    // alongside the canonical bhimpl bodies.  Branch family (A7),
    // guard_value / last_exception family (A8), and every other
    // `JitCodeBuilder`-emitted BC_* are wired in the canonical block
    // above.
    for (key, handler) in [
        // A3 epic: push/pop family wires the canonical 1-byte register
        // handlers (`handler_int_push`/`pop` and ref/float kin) defined
        // alongside the canonical bhimpl bodies.
        ("int_push/i", handler_int_push as BhOpcodeHandler),
        ("ref_push/r", handler_ref_push),
        ("float_push/f", handler_float_push),
        ("int_pop/>i", handler_int_pop),
        ("ref_pop/>r", handler_ref_pop),
        ("float_pop/>f", handler_float_pop),
    ] {
        builder.wire_handler(key, handler);
    }
    // A8 — guard_value + last_exception canonical handlers
    // (`handler_int_guard_value` etc.) are wired in the canonical block
    // above (`wire_handler("int_guard_value/i", ...)`); nothing else to
    // wire here.
    // A5 epic: int binop+cmp+uint family migrated to canonical 1-byte
    // encoding (`[lhs][rhs][dst]` argcode order via `bhhandler_ii_i!`);
    // the per-opname wire_handler calls above
    // bind the canonical handlers directly.  `int_floordiv` /
    // `int_mod` have no `bhimpl_*` upstream: `jtransform.py`
    // rewrites both via `_do_builtin_call` to
    // `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)`
    // before jitcode emission, so neither key reaches `wire_handler`.
}

// Floating-point conditional branches (`blackhole.py:751-798`)
macro_rules! bhhandler_goto_if_not_ff {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_lt, |a: f64, b: f64| a < b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_le, |a: f64, b: f64| a <= b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_eq, |a: f64, b: f64| a == b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ne, |a: f64, b: f64| a != b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_gt, |a: f64, b: f64| a > b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ge, |a: f64, b: f64| a >= b);

bhhandler_goto_if_not_rr!(handler_goto_if_not_ptr_eq, bhimpl_goto_if_not_ptr_eq);
bhhandler_goto_if_not_rr!(handler_goto_if_not_ptr_ne, bhimpl_goto_if_not_ptr_ne);

// assert_green / isconstant — no-ops
bhhandler_i_v!(handler_int_assert_green, bhimpl_int_assert_green);
bhhandler_r_v!(handler_ref_assert_green, bhimpl_ref_assert_green);
bhhandler_f_v!(handler_float_assert_green, bhimpl_float_assert_green);
bhhandler_i_i!(handler_int_isconstant, bhimpl_int_isconstant);
bhhandler_f_i!(handler_float_isconstant, bhimpl_float_isconstant);

// misc
bhhandler_ii_i!(handler_uint_mul_high, bhimpl_uint_mul_high);
bhhandler_iii_i!(handler_int_between, bhimpl_int_between);
fn handler_strhash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let string = bh.registers_r[code[p] as usize];
    bh.registers_i[code[p + 1] as usize] = bh.cpu().bh_strhash(string);
    Ok(p + 2)
}
fn handler_unicodehash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let string = bh.registers_r[code[p] as usize];
    bh.registers_i[code[p + 1] as usize] = bh.cpu().bh_unicodehash(string);
    Ok(p + 2)
}
bhhandler_f_i!(
    handler_convert_float_bytes_to_longlong,
    bhimpl_convert_float_bytes_to_longlong
);
bhhandler_i_f!(
    handler_convert_longlong_bytes_to_float,
    bhimpl_convert_longlong_bytes_to_float
);
bhhandler_f_i!(
    handler_cast_float_to_singlefloat,
    bhimpl_cast_float_to_singlefloat
);
bhhandler_i_f!(
    handler_cast_singlefloat_to_float,
    bhimpl_cast_singlefloat_to_float
);
bhhandler_r_v!(
    handler_hint_force_virtualizable,
    bhimpl_hint_force_virtualizable
);
/// RPython `blackhole.py bhimpl_guard_class`:
/// ```python
/// @arguments("cpu", "r", returns="i")
/// def bhimpl_guard_class(cpu, struct):
///     return cpu.bh_classof(struct)
/// ```
/// The guard has already failed by the time the blackhole reaches this op,
/// so it produces the receiver's class pointer and does not branch.
fn handler_guard_class(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let cpu = bh.cpu();
    let typeptr = cpu.bh_classof(bh.registers_r[code[p] as usize]);
    bh.registers_i[code[p + 1] as usize] = typeptr;
    Ok(p + 2)
}
/// [`handler_guard_class`] with the class delivered to the ref bank —
/// `guard_class/r>r`, for a header read the graph typed as a GC ref.
fn handler_guard_class_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let cpu = bh.cpu();
    let typeptr = cpu.bh_classof(bh.registers_r[code[p] as usize]);
    bh.registers_r[code[p + 1] as usize] = typeptr;
    Ok(p + 2)
}
/// Fallback for the obsolete pyre-only named vtable lookup.
///
/// PyPy's `ClassRepr.getclsfield` emits an ordinary field read; it never tries
/// to resolve `(trait, method)` strings in the blackhole.  If an old frozen
/// graph or the still-conservative abstract-trait path reaches this opcode, no
/// faithful pointer can be manufactured from its descriptor.
/// `convert_and_run_from_pyjitpl` resumes from one abort, and the resume
/// consumers require `abort_permanent_bail` (`bhimpl_abort_permanent`).
fn handler_vtable_method_ptr_bail(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    bh.aborted = true;
    bh.abort_permanent_bail = true;
    Err(DispatchError::LeaveFrame)
}

fn handler_record_quasiimmut_field(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // RPython `bhimpl_record_quasiimmut_field` is a no-op during blackhole
    // execution (`blackhole.py pass`); the metainterp consumes the two
    // descriptors during tracing instead.  Skip past `r` (1 byte) +
    // `d` (2 bytes) + `d` (2 bytes) to reach the next opcode.
    let p = p + 1; // r
    let (_, p) = read_descr(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
/// blackhole.py:
/// ```python
/// @arguments("cpu", "r", "d")
/// def bhimpl_jit_force_quasi_immutable(cpu, struct, mutatefielddescr):
///     from rpython.jit.metainterp import quasiimmut
///     quasiimmut.do_force_quasi_immutable(cpu, struct, mutatefielddescr)
/// ```
/// Skipping past the operands without running the force left every loop that
/// folded a read of the field valid across the write that invalidates it.
fn handler_jit_force_quasi_immutable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let (mutatefielddescr, p) = read_descr(bh, code, p + 1);
    crate::quasiimmut::do_force_quasi_immutable(struct_ptr, mutatefielddescr);
    Ok(p)
}
fn handler_record_known_result_i_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_record_known_result_r_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_str_guard_value(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let r = bh.registers_r[code[position] as usize];
    let (_, p) = read_descr(bh, code, position + 2);
    bh.registers_r[code[p] as usize] = r;
    Ok(p + 1)
}
fn handler_rvmprof_code(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let leaving = bh.registers_i[code[p] as usize];
    let unique_id = bh.registers_i[code[p + 1] as usize];
    bh.bhimpl_rvmprof_code(leaving, unique_id);
    Ok(p + 2)
}
/// RPython `blackhole.py` `bhimpl_copystrcontent`.
fn handler_copystrcontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    bh.cpu()
        .bh_copystrcontent(src, dst, srcstart, dststart, length);
    Ok(p + 5)
}

fn handler_copystrcontent_short<
    const SRCSTART_SHORT: bool,
    const DSTSTART_SHORT: bool,
    const LENGTH_SHORT: bool,
>(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = if SRCSTART_SHORT {
        code[p + 2] as i8 as i64
    } else {
        bh.registers_i[code[p + 2] as usize]
    };
    let dststart = if DSTSTART_SHORT {
        code[p + 3] as i8 as i64
    } else {
        bh.registers_i[code[p + 3] as usize]
    };
    let length = if LENGTH_SHORT {
        code[p + 4] as i8 as i64
    } else {
        bh.registers_i[code[p + 4] as usize]
    };
    bh.cpu()
        .bh_copystrcontent(src, dst, srcstart, dststart, length);
    Ok(p + 5)
}
/// RPython `blackhole.py` `bhimpl_copyunicodecontent`.
fn handler_copyunicodecontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    bh.cpu()
        .bh_copyunicodecontent(src, dst, srcstart, dststart, length);
    Ok(p + 5)
}
fn handler_raise(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let exc = bh.registers_r[code[p] as usize];
    // RPython blackhole.py `bhimpl_raise(self, excvalue)`
    // `e = cast_opaque_ptr(...); assert e; reraise(e)`.
    // Grain never installs MiniMark, so a helper `NEW` of the exception
    // is NULL. Abort the frame rather than panic or publish a void return.
    if exc == 0 {
        bh.aborted = true;
        return Err(DispatchError::LeaveFrame);
    }
    Err(DispatchError::RaiseException {
        exc,
        resume_position: p + 1,
    })
}
fn handler_reraise(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // `reraise/` decodes no operands, so `p` is already the end of the
    // instruction.
    // Grain: no MiniMark means no exception object. Abort rather than
    // panic or publish a void return.
    if bh.exception_last_value == 0 {
        bh.aborted = true;
        return Err(DispatchError::LeaveFrame);
    }
    Err(DispatchError::RaiseException {
        exc: bh.exception_last_value,
        resume_position: p,
    })
}
/// RPython `blackhole.py:987-991`:
/// ```python
/// @arguments("self", returns="i")
/// def bhimpl_last_exception(self):
///     real_instance = self.exception_last_value
///     assert real_instance
///     return ptr2int(real_instance.typeptr)
/// ```
/// Returns the CLASS POINTER (typeptr) of the caught exception, not the
/// exception object itself. Uses `cpu.bh_classof(obj)` to get the typeptr.
fn handler_last_exception(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let exc_obj = bh.exception_last_value;
    // blackhole.py:990 `assert real_instance` — last_exception must
    // only fire while an active caught exception is in scope.
    assert!(
        exc_obj != 0,
        "blackhole.py:990 last_exception: exception_last_value must be non-null"
    );
    // RPython: ptr2int(real_instance.typeptr) — get class pointer.
    // `cpu` is a builder field every interpreter is born with
    // (`BlackholeInterpBuilder::acquire_interp`), so an unset one is a
    // construction bug; standing the exception object in for its own class
    // makes every later `bh_issubclass` compare an instance against a vtable
    // and answer "no match" for a handler that does match.
    bh.registers_i[code[p] as usize] = bh.cpu().bh_classof(exc_obj);
    Ok(p + 1)
}
/// RPython `blackhole.py:993-997`:
/// ```python
/// @arguments("self", returns="r")
/// def bhimpl_last_exc_value(self):
///     return cast_opaque_ptr(GCREF, self.exception_last_value)
/// ```
/// blackhole.py `bhimpl_last_exc_value(self): return self.exception_last_value`.
/// `assert real_instance` ensures last_exc_value fires only while an active
/// caught exception is in scope.
fn bhimpl_last_exc_value(bh: &mut BlackholeInterpreter) -> i64 {
    assert!(
        bh.exception_last_value != 0,
        "blackhole.py:996 last_exc_value: exception_last_value must be non-null"
    );
    bh.exception_last_value
}

bhhandler_self_v_r!(handler_last_exc_value, bhimpl_last_exc_value);
/// RPython `blackhole.py:976-985`:
/// ```python
/// @arguments("self", "i", "L", "pc", returns="L")
/// def bhimpl_goto_if_exception_mismatch(self, vtable, target, pc):
///     bounding_class = cast_adr_to_ptr(int2adr(vtable), CLASSTYPE)
///     real_instance = self.exception_last_value
///     if rclass.ll_issubclass(real_instance.typeptr, bounding_class):
///         return pc  # match → fall through
///     else:
///         return target  # mismatch → jump
/// ```
/// Uses `cpu.bh_classof` for the exception typeptr and
/// `cpu.bh_issubclass` for `rclass.ll_issubclass`.
fn handler_goto_if_exception_mismatch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let bounding_vtable = bh.registers_i[code[p] as usize];
    let target = (code[p + 1] as usize) | ((code[p + 2] as usize) << 8);
    let pc = p + 3;
    let exc_obj = bh.exception_last_value;
    // blackhole.py:981 `assert real_instance` —
    // goto_if_exception_mismatch must only fire while an active caught
    // exception is in scope.
    assert!(
        exc_obj != 0,
        "blackhole.py:981 goto_if_exception_mismatch: exception_last_value must be non-null"
    );
    // See `handler_last_exception` on why an unset `cpu` is not answered
    // with a substitute here.
    let exc_typeptr = bh.cpu().bh_classof(exc_obj);
    // RPython: rclass.ll_issubclass(real_instance.typeptr, bounding_class).
    // Uses Backend::bh_issubclass for the subclass check.
    let is_match = bh.cpu().bh_issubclass(exc_typeptr, bounding_vtable);
    if is_match {
        Ok(pc) // match → fall through
    } else {
        Ok(target) // mismatch → jump to target
    }
}
fn handler_debug_fatalerror(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let string = bh.registers_r[code[p] as usize];
    let length = bh.cpu().bh_strlen(string);
    let bytes: Vec<u8> = (0..length)
        .map(|index| bh.cpu().bh_strgetitem(string, index) as u8)
        .collect();
    panic!("{}", String::from_utf8_lossy(&bytes));
}
/// blackhole.py `bhimpl_cast_ptr_to_int(a)`. The cast is identity
/// (the tagging arithmetic lives at the erase site, never here); the
/// `ll_assert((i & 1) == 1)` checks the operand is a tagged immediate.
/// `ll_assert` remains a translated runtime assertion.
fn bhimpl_cast_ptr_to_int(a: i64) -> i64 {
    assert!((a & 1) == 1, "bhimpl_cast_ptr_to_int: not an odd int");
    a
}

/// blackhole.py `bhimpl_cast_int_to_ptr(i)`. Identity cast; the
/// `ll_assert((i & 1) == 1)` checks the operand is a tagged immediate
/// before it is reinterpreted as a `GCREF`.
fn bhimpl_cast_int_to_ptr(i: i64) -> i64 {
    assert!((i & 1) == 1, "bhimpl_cast_int_to_ptr: not an odd int");
    i
}

bhhandler_r_i!(handler_cast_ptr_to_int, bhimpl_cast_ptr_to_int);
bhhandler_i_r!(handler_cast_int_to_ptr, bhimpl_cast_int_to_ptr);
/// blackhole.py:
/// ```python
/// @arguments(returns="i")
/// def bhimpl_current_trace_length():
///     return -1
/// ```
/// The constant is what tells `rlib/jit.py current_trace_length()` that no
/// trace is being recorded; skipping the write left the result register
/// holding whatever the previous instruction put there.
fn handler_current_trace_length(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p] as usize] = -1;
    Ok(p + 1)
}

// RPython: fielddescr.get_vinfo().clear_vable_token(struct)
//          return cpu.bh_getfield_gc_*(struct, fielddescr)
// pyre: read_descr_vable_field resolves VableField.index → byte offset via VirtualizableInfo.

// Virtualizable field operations (`blackhole.py bhimpl_getfield_vable_i`)
fn handler_getfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu();
    bh.registers_i[code[p] as usize] = cpu.bh_getfield_gc_i(struct_ptr, &descr);
    Ok(p + 1)
}
fn handler_getfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu();
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_gc_r(struct_ptr, &descr).0 as i64;
    Ok(p + 1)
}
fn handler_getfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu();
    bh.registers_f[code[p] as usize] = cpu.bh_getfield_gc_f(struct_ptr, &descr).to_bits() as i64;
    Ok(p + 1)
}

fn handler_setfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_i[code[p + 1] as usize];
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu();
    cpu.bh_setfield_gc_i(struct_ptr, value, &descr);
    Ok(p)
}

/// `setfield_vable_i_imm/rddd`: same `_opimpl_setfield_vable` store as
/// `handler_setfield_vable_i`, with the int value inline as a u32
/// (`lo` + `hi`) instead of an `i` register. Layout: 1B base, 2B lo,
/// 2B hi, 2B `VableField` descr.
fn handler_setfield_vable_i_imm(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let lo = code[p + 1] as u32 | ((code[p + 2] as u32) << 8);
    let hi = code[p + 3] as u32 | ((code[p + 4] as u32) << 8);
    let value = (lo | (hi << 16)) as i64;
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 5);
    let cpu = bh.cpu();
    cpu.bh_setfield_gc_i(struct_ptr, value, &descr);
    Ok(p)
}
fn handler_setfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_r[code[p + 1] as usize];
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu();
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
    Ok(p)
}
fn handler_setfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 1] as usize] as u64);
    let (_, struct_ptr) = vable_clear_token_and_get_vinfo(bh, struct_ptr);
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu();
    cpu.bh_setfield_gc_f(struct_ptr, value, &descr);
    Ok(p)
}

// @arguments("cpu", "r", "i", "d", "d", returns="X")
// Two descriptors: fielddescr (VableArray) + arraydescr (Array).
// RPython `bhimpl_getarrayitem_vable_*`:
//   fielddescr.get_vinfo().clear_vable_token(vable)
//   array = cpu.bh_getfield_gc_r(vable, fielddescr)
//   return cpu.bh_getarrayitem_gc_*(array, index, arraydescr)
/// Resolve `bh.virtualizable_info` to a non-null reference and clear the
/// vable token on `vable`.  RPython parity: every
/// `BlackholeInterpreter.bhimpl_{get,set}field_vable_*` and
/// `bhimpl_{get,set}arrayitem_vable_*` obtains the vinfo from
/// `fielddescr.get_vinfo()` and clears the token before touching storage.
/// Pyre stores the same handle on `bh.virtualizable_info` because its
/// `BhDescr::VableField` and `BhDescr::VableArray` contain only indices into
/// the vinfo-owned field tables.  Vable opcodes therefore require
/// `bh.virtualizable_info` to be non-null; an unset pointer means a
/// builder constructed vable bytecode outside a virtualizable
/// context — a contract bug that must fail loud in both debug and
/// release builds, since the alternative is silent unsafe deref of a
/// null pointer.
fn vable_clear_token_and_get_vinfo(
    bh: &BlackholeInterpreter,
    vable: i64,
) -> (&'static crate::virtualizable::VirtualizableInfo, i64) {
    if bh.virtualizable_info.is_null() {
        panic!(
            "vable opcode requires `bh.virtualizable_info` to be set \
             (RPython `BlackholeInterpreter.bhimpl_*field_vable_*` parity); \
             a null pointer here is a contract bug, not a recoverable case"
        );
    }
    let bh_vinfo = unsafe { &*bh.virtualizable_info };
    // blackhole.py `fielddescr.get_vinfo().clear_vable_token(struct)`.
    // `BhDescr::VableField` is an index; recover the stamped FieldDescr
    // and prefer that backref when `finalize_arc` installed it.
    let upgraded = bh_vinfo
        .static_field_descrs()
        .first()
        .and_then(|descr| descr.as_field_descr()?.get_vinfo());
    let clear_info = upgraded
        .as_ref()
        .and_then(|marker| {
            marker
                .as_any()
                .downcast_ref::<crate::virtualizable::VirtualizableInfo>()
        })
        .unwrap_or(bh_vinfo);
    let vable =
        unsafe { crate::virtualizable::bh_clear_vable_token(clear_info, vable as *mut u8) } as i64;
    (bh_vinfo, vable)
}

/// Decode the vable-array descr pair after the register operands and
/// clear the token.
///
/// `bhimpl_getarrayitem_vable_*` / `bhimpl_setarrayitem_vable_*` /
/// `bhimpl_arraylen_vable` all start
/// `fielddescr.get_vinfo().clear_vable_token(vable)` then load the array
/// through `cpu.bh_getfield_gc_r`.
fn take_vable_array<'a>(
    bh: &'a BlackholeInterpreter,
    vable: i64,
    code: &[u8],
    descr_pos: usize,
) -> (&'a crate::virtualizable::VableArrayInfo, usize, i64) {
    let (vinfo, vable) = vable_clear_token_and_get_vinfo(bh, vable);
    let (_field_descr, array_idx, p) = read_descr_vable_array(bh, code, descr_pos);
    let (_array_descr, pos) = read_descr(bh, code, p);
    // Item access goes through `vable_read_array_item` /
    // `vable_write_array_item`, which follow `EmbeddedArray`'s container
    // pointer as well as a direct `Ptr(GcArray)`.
    let info = &vinfo.array_fields[array_idx];
    (info, pos, vable)
}

// Virtualizable array operations (`bhimpl_getarrayitem_vable_*`)
fn handler_getarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 2);
    bh.registers_i[code[p] as usize] =
        unsafe { crate::virtualizable::vable_read_array_item(vable as *const u8, ainfo, index) };
    Ok(p + 1)
}
fn handler_getarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let nbody_debug = crate::nbody_debug_enabled();
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 2);
    let value =
        unsafe { crate::virtualizable::vable_read_array_item(vable as *const u8, ainfo, index) };
    if nbody_debug && matches!(index, 5 | 6 | 8 | 9) {
        eprintln!(
            "[nbody-debug][bh-vable-get-r] position={} last_opcode_position={} index={} value={:#x}",
            bh.position, bh.last_opcode_position, index, value as usize
        );
    }
    bh.registers_r[code[p] as usize] = value;
    Ok(p + 1)
}
fn handler_setarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 3);
    unsafe {
        crate::virtualizable::vable_write_array_item(vable as *mut u8, ainfo, index, value);
    }
    Ok(p)
}
fn handler_setarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let nbody_debug = crate::nbody_debug_enabled();
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 3);
    if nbody_debug && matches!(index, 5 | 6 | 8 | 9) {
        eprintln!(
            "[nbody-debug][bh-vable-set-r] position={} last_opcode_position={} index={} value={:#x}",
            bh.position, bh.last_opcode_position, index, value as usize
        );
    }
    unsafe {
        crate::virtualizable::vable_write_array_item(vable as *mut u8, ainfo, index, value);
    }
    Ok(p)
}
fn handler_arraylen_vable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 1);
    bh.registers_i[code[p] as usize] =
        unsafe { crate::virtualizable::bhimpl_arraylen_vable(vable as *const u8, ainfo) as i64 };
    Ok(p + 1)
}

/// `arraybase_vable/rdd>i` — item-0 address of a virtualizable array.
///
/// Wiring this is not optional even though a trace carrying the op always
/// aborts: the abort resumes through the blackhole over this very jitcode, so
/// an unwired byte panics `dispatch_step` on exactly the path the op creates.
/// Operand layout is `arraylen_vable`'s, so the decode is line-for-line its
/// sibling above.
///
/// No escape signal is raised here. The signal exists to abort a *trace*, and
/// the blackhole is already running because that abort happened.
fn handler_arraybase_vable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let (vinfo, vable) = vable_clear_token_and_get_vinfo(bh, vable);
    let (field_descr, p) = read_descr(bh, code, p + 1);
    let array_idx = field_descr.as_vable_array_index();
    let (_, p) = read_descr(bh, code, p);
    let ainfo = &vinfo.array_fields[array_idx];
    let base = unsafe { crate::virtualizable::bhimpl_arraybase_vable(vable as *const u8, ainfo) };
    bh.registers_i[code[p] as usize] = base as usize as i64;
    Ok(p + 1)
}

// ── getarrayitem_raw / setarrayitem_raw (blackhole.py:1343-1365) ────
/// RPython `blackhole.py` `bhimpl_getarrayitem_raw_i`:
/// `return cpu.bh_getarrayitem_raw_i(array, index, arraydescr)`.
/// Raw memory access — NOT GC-managed. Uses unsafe direct read.
fn handler_getarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    bh.registers_i[code[p] as usize] = bh.cpu().bh_getarrayitem_raw_i(array, index, descr);
    Ok(p + 1)
}
/// RPython `blackhole.py` `bhimpl_setarrayitem_raw_i`.
fn handler_setarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    bh.cpu().bh_setarrayitem_raw_i(array, index, value, descr);
    Ok(p)
}

// Conditional calls (`blackhole.py:1257-1276`)
fn handler_conditional_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let condition = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    if condition != 0 {
        if !is_callable_fnaddr(func) {
            return Err(reject_unresolved_call(bh, func));
        }
        bh.last_exc().set(0);
        bh.cpu()
            .bh_call_v(func, Some(&ai), Some(&ar), None, calldescr);
        check_residual_call_exception_after(bh, p)?;
    }
    Ok(p)
}
fn handler_conditional_call_value_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    if value == 0 {
        if !is_callable_fnaddr(func) {
            return Err(reject_unresolved_call(bh, func));
        }
        bh.last_exc().set(0);
        value = bh
            .cpu()
            .bh_call_i(func, Some(&ai), Some(&ar), None, calldescr);
        check_residual_call_exception_after(bh, p + 1)?;
    }
    bh.registers_i[code[p] as usize] = value;
    Ok(p + 1)
}
fn handler_conditional_call_value_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_r[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr_handle, p) = read_calldescr(bh, code, p);
    let calldescr = calldescr_handle.get();
    if value == 0 {
        if !is_callable_fnaddr(func) {
            return Err(reject_unresolved_call(bh, func));
        }
        bh.last_exc().set(0);
        value = bh
            .cpu()
            .bh_call_r(func, Some(&ai), Some(&ar), None, calldescr)
            .0 as i64;
        check_residual_call_exception_after(bh, p + 1)?;
    }
    bh.registers_r[code[p] as usize] = value;
    Ok(p + 1)
}

// ── list ops (blackhole.py:1195-1219) — compound: getfield_gc_r + getarrayitem ──
fn handler_getlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_getarrayitem_gc_i(items, index, array_descr);
    Ok(p + 1)
}
fn handler_getlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_r[code[p] as usize] = cpu.bh_getarrayitem_gc_r(items, index, array_descr).0 as i64;
    Ok(p + 1)
}
fn handler_setlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_i(items, index, value, array_descr);
    Ok(p)
}
fn handler_setlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_r(items, index, majit_ir::GcRef(value as usize), array_descr);
    Ok(p)
}

// Switch dispatch (`blackhole.py:954-960`)
/// RPython `blackhole.py:954-960`:
/// ```python
/// @arguments("i", "d", "pc", returns="L")
/// def bhimpl_switch(switchvalue, switchdict, pc):
///     assert isinstance(switchdict, SwitchDictDescr)
///     try:
///         return switchdict.dict[switchvalue]
///     except KeyError:
///         return pc
/// ```
fn handler_switch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let switchvalue = bh.registers_i[code[p] as usize];
    let (descr, pos) = read_descr(bh, code, p + 1);
    // RPython blackhole.py:954-960:
    //   try: return switchdict.dict[switchvalue]
    //   except KeyError: return pc
    if let Some(target) = descr.switch_lookup(switchvalue) {
        Ok(target)
    } else {
        Ok(pos) // fallthrough (KeyError path)
    }
}

// ── check_neg_index / check_resizable_neg_index (blackhole.py:1148-1158) ─
fn handler_check_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu();
        index += cpu.bh_arraylen_gc(array, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}
fn handler_check_resizable_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu();
        index += cpu.bh_getfield_gc_i(lst, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}

// blackhole.py bhimpl_getarrayitem_gc_f
// Floating-point GC array operations
fn handler_getarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu();
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py bhimpl_setarrayitem_gc_f
fn handler_setarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_gc_f(array, index, value, descr);
    Ok(p)
}
// blackhole.py bhimpl_getarrayitem_raw_f
fn handler_getarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu();
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_raw_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py bhimpl_setarrayitem_raw_f
fn handler_setarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu();
    cpu.bh_setarrayitem_raw_f(array, index, value, descr);
    Ok(p)
}
// getfield_raw_r (pure only, blackhole.py:1467-1469)
fn handler_getfield_raw_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    let (descr, p) = read_descr(bh, code, p + 1);
    let cpu = bh.cpu();
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_raw_r(struct_ptr, descr).0 as i64;
    Ok(p + 1)
}
// getinteriorfield_gc_f / setinteriorfield_gc_f / setinteriorfield_gc_r
/// RPython `blackhole.py:1417-1419`:
/// `return cpu.bh_getinteriorfield_gc_f(array, index, descr)`
fn handler_getinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu();
    bh.registers_f[code[p] as usize] =
        cpu.bh_getinteriorfield_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
fn handler_getinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu();
    bh.registers_r[code[p] as usize] = cpu.bh_getinteriorfield_gc_r(array, index, descr).0 as i64;
    Ok(p + 1)
}
fn handler_setinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu();
    cpu.bh_setinteriorfield_gc_f(array, index, value, descr);
    Ok(p)
}
fn handler_setinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu();
    cpu.bh_setinteriorfield_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(p)
}
// gc_load_indexed_i/f, gc_store_indexed_i/f (blackhole.py:1518-1540)
fn handler_gc_load_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu();
    bh.registers_i[code[p + 5] as usize] =
        cpu.bh_gc_load_indexed_i(addr, index, scale, base_ofs, bytes);
    Ok(p + 6)
}
fn handler_gc_load_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu();
    bh.registers_f[code[p + 5] as usize] = cpu
        .bh_gc_load_indexed_f(addr, index, scale, base_ofs, bytes)
        .to_bits() as i64;
    Ok(p + 6)
}
// blackhole.py bhimpl_gc_store_indexed_i
fn handler_gc_store_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "i", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (descr, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu();
    cpu.bh_gc_store_indexed_i(addr, index, value, scale, base_ofs, bytes, descr);
    Ok(p)
}
// blackhole.py bhimpl_gc_store_indexed_f
fn handler_gc_store_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "f", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (descr, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu();
    cpu.bh_gc_store_indexed_f(addr, index, value, scale, base_ofs, bytes, descr);
    Ok(p)
}
// blackhole.py bhimpl_raw_store_i/f
fn handler_raw_store_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "i", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    // blackhole.py bhimpl_raw_store_i: cpu.bh_raw_store_i(addr, offset, newvalue, arraydescr)
    let cpu = bh.cpu();
    cpu.bh_raw_store_i(addr, offset, value, descr);
    Ok(p)
}
fn handler_raw_store_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "f", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    // blackhole.py:1510-1511: cpu.bh_raw_store_f(addr, offset, newvalue, arraydescr)
    let cpu = bh.cpu();
    cpu.bh_raw_store_f(addr, offset, value, descr);
    Ok(p)
}
fn handler_raw_load_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    // blackhole.py:1500-1501: cpu.bh_raw_load_i(addr, offset, arraydescr)
    let cpu = bh.cpu();
    bh.registers_i[code[p] as usize] = cpu.bh_raw_load_i(addr, offset, descr);
    Ok(p + 1)
}
fn handler_raw_load_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    // blackhole.py:1503-1504: cpu.bh_raw_load_f(addr, offset, arraydescr)
    let cpu = bh.cpu();
    bh.registers_f[code[p] as usize] = cpu.bh_raw_load_f(addr, offset, descr).to_bits() as i64;
    Ok(p + 1)
}
// newlist / newlist_clear / newlist_hint (blackhole.py:1160-1193)
// RPython: compound allocation: bh_new(structdescr) + setfield + bh_new_array + setfield.
// 4 descriptors: structdescr, lengthdescr, itemsdescr, arraydescr.
fn handler_newlist(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let structdescr = structdescr.clone();
    let lengthdescr = lengthdescr.clone();
    let itemsdescr = itemsdescr.clone();
    let arraydescr = arraydescr.clone();
    // blackhole.py:1163: result = cpu.bh_new(structdescr)
    let result = bh.cpu().bh_new(&structdescr);
    check_blackhole_allocation_after(bh, result, p + 1)?;
    // blackhole.py:1164: cpu.bh_setfield_gc_i(result, length, lengthdescr)
    bh.cpu().bh_setfield_gc_i(result, length, &lengthdescr);
    // blackhole.py:1165-1169: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        bh.cpu().bh_new_array_clear(length, &arraydescr)
    } else {
        bh.cpu().bh_new_array(length, &arraydescr)
    };
    check_blackhole_allocation_after(bh, items, p + 1)?;
    // blackhole.py:1170: cpu.bh_setfield_gc_r(result, items, itemsdescr)
    bh.cpu()
        .bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), &itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1173-1180: newlist_clear always uses bh_new_array_clear.
fn handler_newlist_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let structdescr = structdescr.clone();
    let lengthdescr = lengthdescr.clone();
    let itemsdescr = itemsdescr.clone();
    let arraydescr = arraydescr.clone();
    let result = bh.cpu().bh_new(&structdescr);
    check_blackhole_allocation_after(bh, result, p + 1)?;
    bh.cpu().bh_setfield_gc_i(result, length, &lengthdescr);
    // blackhole.py:1178: items = cpu.bh_new_array_clear(length, arraydescr)
    let items = bh.cpu().bh_new_array_clear(length, &arraydescr);
    check_blackhole_allocation_after(bh, items, p + 1)?;
    bh.cpu()
        .bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), &itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1182-1193: newlist_hint — length=0, allocate with hint capacity.
fn handler_newlist_hint(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lengthhint = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let structdescr = structdescr.clone();
    let lengthdescr = lengthdescr.clone();
    let itemsdescr = itemsdescr.clone();
    let arraydescr = arraydescr.clone();
    let result = bh.cpu().bh_new(&structdescr);
    check_blackhole_allocation_after(bh, result, p + 1)?;
    // blackhole.py:1186: cpu.bh_setfield_gc_i(result, 0, lengthdescr)
    bh.cpu().bh_setfield_gc_i(result, 0, &lengthdescr);
    // blackhole.py:1187-1191: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        bh.cpu().bh_new_array_clear(lengthhint, &arraydescr)
    } else {
        bh.cpu().bh_new_array(lengthhint, &arraydescr)
    };
    check_blackhole_allocation_after(bh, items, p + 1)?;
    bh.cpu()
        .bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), &itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py bhimpl_getarrayitem_vable_f
fn handler_getarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 2);
    bh.registers_f[code[p] as usize] =
        unsafe { crate::virtualizable::vable_read_array_item(vable as *const u8, ainfo, index) };
    Ok(p + 1)
}
// blackhole.py bhimpl_setarrayitem_vable_f
fn handler_setarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_f[code[p + 2] as usize];
    let (ainfo, p, vable) = take_vable_array(bh, vable, code, p + 3);
    unsafe {
        crate::virtualizable::vable_write_array_item(vable as *mut u8, ainfo, index, value);
    }
    Ok(p)
}
// blackhole.py bhimpl_getlistitem_gc_f
fn handler_getlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_f[code[p] as usize] = cpu
        .bh_getarrayitem_gc_f(items, index, array_descr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py bhimpl_setlistitem_gc_f
fn handler_setlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu();
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_f(items, index, value, array_descr);
    Ok(p)
}
// inline_call — RPython blackhole.py:1278-1319
// RPython: cpu.bh_call_*(adr2int(jitcode.fnaddr), args_i, args_r, args_f, jitcode.calldescr)
/// Read the `j` argcode of an `inline_call_*` op.
///
/// `blackhole.py:150-157` resolves `'d'` and `'j'` out of the *same* `descrs`
/// table and differs only in the trailing check — `if argtype == 'j': assert
/// isinstance(value, JitCode)` — because upstream's `JitCode` is itself an
/// `AbstractDescr` (`jitcode.py`). pyre's runtime pool models that with
/// `RuntimeBhDescr::JitCode(Arc<JitCode>)`, so this reads the entry as a
/// jitcode and takes `fnaddr` / `calldescr` off the object, exactly as
/// `bhimpl_inline_call_*` does (`blackhole.py:1280`).
///
/// Reading them off the object also matters for correctness: the flattened
/// `BhDescr::JitCode.fnaddr` reached through `read_descr` lives in
/// `ALL_DESCRS`, which no `runtime_fnaddr_patch` pass ever rewrites, so it is
/// always the build-script process's value. The `Arc<JitCode>` the pool wraps
/// comes from the patched jitcode table and carries the runtime address.
///
/// `descr_at` (not `exec.descrs`) is the resolver: an LLBC-extracted jitcode
/// carries an empty per-jitcode pool and names its descrs by index into the
/// process-global build-time pool.
fn read_inline_call_jitcode(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (usize, i64, CallDescrHandle, usize) {
    let idx = (code[p] as usize) | ((code[p + 1] as usize) << 8);
    let entry = bh
        .jitcode
        .descr_at(idx)
        .unwrap_or_else(|| panic!("inline_call: descrs[{idx}] is absent from both descr pools"));
    if let Some(jitcode) = entry.as_jitcode_owned() {
        return (
            jitcode.try_index().unwrap_or(0),
            jitcode.fnaddr,
            CallDescrHandle::InCalleeJitCode(jitcode),
            p + 2,
        );
    }
    // A runtime-built pool may still hold the flattened form.
    match entry.as_bh_descr() {
        Some(BhDescr::JitCode {
            jitcode_index,
            fnaddr,
            ..
        }) => (
            *jitcode_index,
            *fnaddr,
            CallDescrHandle::InJitCode(bh.jitcode.clone(), idx),
            p + 2,
        ),
        _ => panic!("inline_call: descrs[{idx}] is not a JitCode entry: {entry:?}"),
    }
}

/// Whether `fnaddr` is a `symbolic_fnaddr_for_path` hash rather than a real
/// code address.
///
/// Upstream never has to ask: `CallControl.get_jitcode` binds
/// `llmemory.cast_ptr_to_adr(getfunctionptr(graph))` (`call.py`) for
/// every jitcode it mints, so `jitcode.fnaddr` is always a linker-resolved
/// address in the same binary as the metainterp. pyre runs its codewriter in
/// `pyre-jit-trace/build.rs`, a different process, and substitutes a
/// `symbolic_fnaddr_for_path` hash for any callee whose real address the host
/// did not publish through `jit_trace_fnaddrs()`; `patch_constants_i_fnaddrs`
/// is keyed on the build-time address, so a hash is never rebound.
///
/// Symbolic hashes carry the `SYMBOLIC_FNADDR_BASE` high-16-bit tag stamped
/// at generation, which no real user-space funcptr can carry on any target
/// pyre builds for (aarch64 Linux maps code with bit 47 set, so a bit-range
/// heuristic would misclassify every real funcptr there).  Same discriminator
/// the walker's residual-call path uses to decline a symbolic funcptr →
/// `ResidualDecline::Symbolic`.
#[inline]
pub(crate) fn is_symbolic_fnaddr(fnaddr: i64) -> bool {
    majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr)
}

/// Whether a jitcode's `fnaddr` can be called as a function pointer.
///
/// Stricter than `!is_symbolic_fnaddr`: `0` is upstream's own "no address"
/// spelling (`jitcode.py JitCode.__init__ fnaddr=None`) and must not be called
/// either.  Both residual and inline calls decline it before reaching a
/// backend; otherwise a callee that never ran could fabricate a return value.
#[inline]
pub(crate) fn is_callable_fnaddr(fnaddr: i64) -> bool {
    fnaddr != 0 && !is_symbolic_fnaddr(fnaddr)
}

/// True when `position` is still a node-entry of this helper: no child
/// `INLINE_CALL` and no `SETFIELD` have run, and the param slots (`r0`,
/// `i0`, `i1`) have not been overwritten. Restarting `fnaddr` with those
/// slots is then the same as interpreting the rest of this frame.
///
/// A mid-`shift` Sequence/Alternative PC (left child already stored)
/// returns false — re-running the helper would reread `old_left` from
/// the current-character marks.
///
/// `run_after_rooting` only asks this at pc 0. Mid-tree resumes (regex
/// leaf pc 210) skip the probe; the first unsafe PC is cached for the
/// leftover-opcode / Char-19/28 tests that still call this directly.
pub fn native_entry_args_intact(jitcode: &JitCode, position: usize) -> bool {
    if position == 0 {
        return true;
    }
    if jitcode.startpoints.is_none() {
        return false;
    }
    position <= cached_first_unsafe_native_entry_pc(jitcode)
}

fn opcode_clobbers_native_entry_args(code: &[u8], pc: usize) -> bool {
    let Some(&op) = code.get(pc) else {
        return false;
    };
    match op {
        jitcode::insns::BC_INLINE_CALL
        | jitcode::insns::BC_SETFIELD_GC_I
        | jitcode::insns::BC_SETFIELD_GC_R
        | jitcode::insns::BC_SETFIELD_GC_F => true,
        jitcode::insns::BC_GETFIELD_GC_I | jitcode::insns::BC_GETFIELD_GC_R => {
            let dest = code.get(pc + 4).copied().unwrap_or(0xff);
            dest == 0 || (op == jitcode::insns::BC_GETFIELD_GC_I && dest == 1)
        }
        jitcode::insns::BC_MOVE_I | jitcode::insns::BC_MOVE_I_C => {
            let dest = code.get(pc + 2).copied().unwrap_or(0xff);
            dest == 0 || dest == 1
        }
        jitcode::insns::BC_INT_EQ => {
            let dest = code.get(pc + 3).copied().unwrap_or(0xff);
            dest == 0 || dest == 1
        }
        _ => false,
    }
}

/// Smallest startpoint whose opcode clobbers a native-entry param slot.
/// `native_entry_args_intact(p)` is then `p == 0 || p <= this`.
fn first_unsafe_native_entry_pc(jitcode: &JitCode) -> usize {
    let Some(starts) = jitcode.startpoints.as_ref() else {
        return 0;
    };
    let code = jitcode.code.as_slice();
    let mut first = usize::MAX;
    for &pc in starts {
        if opcode_clobbers_native_entry_args(code, pc) {
            first = first.min(pc);
        }
    }
    first
}

thread_local! {
    /// One-entry cache of [`first_unsafe_native_entry_pc`]. The regex
    /// deopt chain reseats the same `shift` helper on every frame.
    static NATIVE_ENTRY_UNSAFE_PC: std::cell::Cell<Option<(usize, usize, usize)>> =
        const { std::cell::Cell::new(None) };
}

fn cached_first_unsafe_native_entry_pc(jitcode: &JitCode) -> usize {
    let ptr = jitcode as *const JitCode as usize;
    let nstarts = jitcode.startpoints.as_ref().map_or(0, |s| s.len());
    NATIVE_ENTRY_UNSAFE_PC.with(|cell| {
        if let Some((cached_ptr, cached_n, limit)) = cell.get()
            && cached_ptr == ptr
            && cached_n == nstarts
        {
            return limit;
        }
        let limit = first_unsafe_native_entry_pc(jitcode);
        cell.set(Some((ptr, nstarts, limit)));
        limit
    })
}

/// `blackhole.py _setup_return_value_*` + `get_tmpreg_*`: copy the
/// value the callee's return opcode actually wrote, not the trailing
/// bytecode's source register (`trailing_return_info` misses
/// `int_return/c` and the non-last exit of a multi-return helper).
fn copy_inline_callee_tmpreg(
    bh: &mut BlackholeInterpreter,
    callee: &mut BlackholeInterpreter,
    dest: Option<(JitArgKind, usize)>,
) {
    match dest {
        None => {}
        Some((JitArgKind::Int, dst)) => {
            bh.registers_i[dst] = callee.get_tmpreg_i();
        }
        Some((JitArgKind::Ref, dst)) => {
            bh.registers_r[dst] = callee.get_tmpreg_r();
        }
        Some((JitArgKind::Float, dst)) => {
            bh.registers_f[dst] = callee.get_tmpreg_f();
        }
    }
}

fn return_kind_from_bh(return_type: BhReturnType) -> JitArgKind {
    match return_type {
        BhReturnType::Int => JitArgKind::Int,
        BhReturnType::Ref => JitArgKind::Ref,
        BhReturnType::Float => JitArgKind::Float,
        BhReturnType::Void => {
            panic!("inline_call: caller declared a dest but the callee returned void")
        }
    }
}

/// Refuse to call an unresolved `inline_call_*` target when its jitcode
/// body is not available to interpret either.
///
/// The walker declines the equivalent residual (`ResidualDecline::Symbolic`).
/// The blackhole must finish the half-opcode (`convert_and_run_from_pyjitpl`);
/// jumping to a symbolic hash is not a finish. Prefer
/// [`interpret_unresolved_inline_call`] when the descr still names a body.
#[inline]
fn reject_unresolved_inline_call(
    bh: &mut BlackholeInterpreter,
    jitcode_index: usize,
    fnaddr: i64,
) -> DispatchError {
    if crate::majit_log_enabled() {
        eprintln!(
            "[bh] inline_call declined: jitcodes[{jitcode_index}] has no runtime address \
             (fnaddr={fnaddr:#x}); register the callee's path in the host's fnaddr bindings"
        );
    }
    bh.bail_residual = Some(format!(
        "inline jitcode[{jitcode_index}] fnaddr={fnaddr:#x}"
    ));
    bh.aborted = true;
    DispatchError::LeaveFrame
}

/// Byte-interpret a canonical `inline_call_*` whose `fnaddr` is symbolic.
///
/// Upstream `bhimpl_inline_call_*` (`blackhole.py`) calls
/// `cpu.bh_call_*(jitcode.fnaddr)` because the codewriter and the runtime
/// share one process, so every helper jitcode carries a real entry.
/// pyre's lowering runs in `build.rs`; an unpublished path stays a
/// `symbolic_fnaddr_for_path` hash. `handler_inline_call_nested_ext`
/// already interprets that shape when `fnaddr` is 0. The canonical
/// handlers must do the same, or `convert_and_run_from_pyjitpl` aborts
/// mid-opcode and the portal sees `usize::MAX` instead of a merge point.
/// Copy the portal identity-slot registers into a callee blackhole frame.
///
/// `blackhole.py` `bhimpl_inline_call_*` does not build a callee interpreter.
/// `resume.py` `rebuild_from_resumedata` writes each live register through
/// `consume_boxes`. The slots `load_state_field` / `store_state_field` address
/// are those registers when the parent already holds them. A plain `[int]`
/// ptr/len is a declared `inline_call_*` argument (`MIFrame.setup_call`,
/// `bhimpl_inline_call_ir_v`), not an implicit copy.
fn copy_identity_slots_from_parent(
    parent: &BlackholeInterpreter,
    callee: &mut BlackholeInterpreter,
) {
    // The callee must already be seated (`setposition`). Only a body that
    // emitted `load_state_field` / `store_state_field` reads these indices
    // as identity slots.
    if !callee.jitcode.reads_identity_slots() {
        return;
    }
    let layout = &parent.state_field_layout;
    let i_end = layout.int_scalar_base.saturating_add(layout.total_slots());
    for slot in layout.int_scalar_base..i_end {
        if slot < parent.registers_i.len() && slot < callee.registers_i.len() {
            callee.registers_i[slot] = parent.registers_i[slot];
        }
    }
    let r_end = layout
        .ref_scalar_base
        .saturating_add(layout.num_ref_scalars);
    for slot in layout.ref_scalar_base..r_end {
        if slot < parent.registers_r.len() && slot < callee.registers_r.len() {
            callee.registers_r[slot] = parent.registers_r[slot];
        }
    }
    let f_end = layout
        .float_scalar_base
        .saturating_add(layout.num_float_scalars);
    for slot in layout.float_scalar_base..f_end {
        if slot < parent.registers_f.len() && slot < callee.registers_f.len() {
            callee.registers_f[slot] = parent.registers_f[slot];
        }
    }
}

/// Bytecode-only callee: `JitCodeBuilder` sub-jitcodes and translator
/// graphs absent from `jit_trace_fnaddrs` have no machine entry, so
/// `bhimpl_inline_call_*`'s `cpu.bh_call_*(jitcode.fnaddr)` cannot run
/// them. This interpreter is that class, not a second dispatch for a
/// callable `fnaddr`.
fn interpret_unresolved_inline_call(
    bh: &mut BlackholeInterpreter,
    handle: &CallDescrHandle,
    jitcode_index: usize,
    fnaddr: i64,
    args_i: &[i64],
    args_r: &[i64],
    args_f: &[i64],
    dest: Option<(JitArgKind, usize)>,
    post_p: usize,
) -> Result<usize, DispatchError> {
    let Some(sub_jitcode) = handle.as_callee_jitcode() else {
        return Err(reject_unresolved_inline_call(bh, jitcode_index, fnaddr));
    };
    if crate::majit_log_enabled() {
        eprintln!(
            "[bh] inline_call interpreting jitcodes[{jitcode_index}] name={:?} \
             (fnaddr={fnaddr:#x} is not callable)",
            sub_jitcode.name,
        );
    }
    let mut callee = bh
        .inline_callee_scratch
        .take()
        .unwrap_or_else(|| Box::new(BlackholeInterpreter::default()));
    callee.clone_context_from(bh);
    callee.setposition(sub_jitcode, 0);
    for (index, &value) in args_i.iter().enumerate() {
        if index < callee.registers_i.len() {
            callee.registers_i[index] = value;
        }
    }
    for (index, &value) in args_r.iter().enumerate() {
        if index < callee.registers_r.len() {
            callee.registers_r[index] = value;
        }
    }
    for (index, &value) in args_f.iter().enumerate() {
        if index < callee.registers_f.len() {
            callee.registers_f[index] = value;
        }
    }
    // `resume.py` `rebuild_from_resumedata` fills a callee's registers via
    // `consume_boxes`. A byte-interpreted arm also reads the portal identity
    // slots (`load_state_field`), which are not inline-call arguments, so
    // copy the slots the parent frame already holds.
    copy_identity_slots_from_parent(bh, &mut callee);

    let outcome = 'callee: {
        match callee.run() {
            BhRunOutcome::ContinueRunningNormally(args) => {
                bh.position = post_p;
                break 'callee Err(DispatchError::ContinueRunningNormally(args));
            }
            BhRunOutcome::EndOfCode => panic!(
                "inline_call: callee jitcode {:?} index {:?} ended without a \
                 return opcode, so there is no result for the caller",
                callee.jitcode.name,
                callee.jitcode.try_index(),
            ),
            BhRunOutcome::LeaveFrame | BhRunOutcome::Exception => {}
        }
        if callee.aborted {
            bh.position = post_p;
            bh.aborted = true;
            bh.abort_permanent_bail = callee.abort_permanent_bail;
            if let Some(name) = callee.bail_residual.take() {
                bh.bail_residual = Some(name);
            }
            break 'callee Err(DispatchError::LeaveFrame);
        }
        if callee.got_exception {
            let exc_val = callee.exception_last_value;
            if exc_val != 0 {
                break 'callee Err(DispatchError::RaiseException {
                    exc: exc_val,
                    resume_position: post_p,
                });
            }
            bh.position = post_p;
            bh.got_exception = true;
            break 'callee Err(DispatchError::LeaveFrame);
        }
        let dest = match dest {
            None => {
                assert!(
                    callee.return_type == BhReturnType::Void,
                    "inline_call interpret: callee jitcode {:?} returns {:?} \
                     but the caller declared no destination",
                    callee.jitcode.name,
                    callee.return_type,
                );
                None
            }
            Some((_, dst)) => Some((return_kind_from_bh(callee.return_type), dst)),
        };
        copy_inline_callee_tmpreg(bh, &mut callee, dest);
        Ok(post_p)
    };
    callee.reset_for_inline_reuse();
    bh.inline_callee_scratch = Some(callee);
    outcome
}
fn handler_inline_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &af,
            Some((JitArgKind::Int, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_irf_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_irf_i(fnaddr, &ai, &ar, &af, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_inline_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &af,
            Some((JitArgKind::Ref, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_irf_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_irf_r(fnaddr, &ai, &ar, &af, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &af,
            Some((JitArgKind::Float, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_irf_f.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_irf_f(fnaddr, &ai, &ar, &af, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_f[dst] = result.to_bits() as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &af,
            None,
            p,
        );
    }
    // blackhole.py → bhimpl_inline_call_irf_v.
    bh.last_exc().set(0);
    bh.bhimpl_inline_call_irf_v(fnaddr, &ai, &ar, &af, calldescr.get());
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}
fn handler_inline_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &[],
            Some((JitArgKind::Int, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_ir_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_ir_i(fnaddr, &ai, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_inline_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &[],
            Some((JitArgKind::Ref, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_ir_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_ir_r(fnaddr, &ai, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &ai,
            &ar,
            &[],
            None,
            p,
        );
    }
    // blackhole.py → bhimpl_inline_call_ir_v.
    bh.last_exc().set(0);
    bh.bhimpl_inline_call_ir_v(fnaddr, &ai, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}
fn handler_inline_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &[],
            &ar,
            &[],
            Some((JitArgKind::Int, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_r_i.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_r_i(fnaddr, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_inline_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let dst = code[p] as usize;
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &[],
            &ar,
            &[],
            Some((JitArgKind::Ref, dst)),
            p + 1,
        );
    }
    // blackhole.py → bhimpl_inline_call_r_r.
    bh.last_exc().set(0);
    let result = bh.bhimpl_inline_call_r_r(fnaddr, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    if !is_callable_fnaddr(fnaddr) {
        return interpret_unresolved_inline_call(
            bh,
            &calldescr,
            jitcode_index,
            fnaddr,
            &[],
            &ar,
            &[],
            None,
            p,
        );
    }
    // blackhole.py → bhimpl_inline_call_r_v.
    bh.last_exc().set(0);
    bh.bhimpl_inline_call_r_v(fnaddr, &ar, calldescr.get());
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}

/// TODO: pyre nested-bytecode `inline_call`.
///
/// Pyre does not compile inlined helpers into separate native functions —
/// trace IR merges them into a single compiled trace.  When a guard
/// fails and the blackhole interpreter resumes the original jitcode,
/// the inline call must be re-interpreted as nested bytecode because
/// no per-helper `fnaddr` exists.  RPython's canonical `inline_call_*`
/// handlers (`handler_inline_call_irf_i` etc.) instead expect a real
/// C-ABI fnaddr stored on `BhDescr::JitCode`.
///
/// This handler is the pyre nested-bytecode `inline_call` adapter under
/// the `(bh, code, position) -> Result<usize, _>` signature.  Operand
/// payload:
///   `sub_idx: u16`, `num_args: u16`,
///   `num_args × (kind: u8, caller_src: u8, callee_dst: u8)`,
///   `dest: u8`
/// The last byte is the single result register, matching
/// `blackhole.py _setup_return_value_*`'s `code[position-1]`.
/// `NO_RETURN_REG` encodes a void call.
///
/// Registered via the pyre-only opname `inline_call_nested_ext/P`
/// in `extension_insns()`.  Canonical `inline_call_{r,ir,irf}_*`
/// keys are pinned in `wellknown_bh_insns()` (`BC_INLINE_CALL_*`,
/// `insns.rs`'s `wellknown_bh_insns`) — this `_nested_ext/P` shape is a separate
/// pyre-only adapter that carries the nested-bytecode payload layout
/// described above, distinct from the canonical `dR`/`dIR`/`dIRF`
/// arglist shapes the upstream walker dispatches.
fn handler_inline_call_nested_ext(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut p = p;
    let sub_idx = jitcode::read_u16(code, &mut p) as usize;
    let num_args = jitcode::read_u16(code, &mut p) as usize;

    // blackhole.py:150-157 `j` argcode resolves via `descrs[idx]`
    // asserted to be a JitCode entry; pyre's helper-side
    // `RuntimeBhDescr::JitCode(Arc<JitCode>)` is the analogous slot.
    // Borrow the descrs slot only long enough to read `fnaddr` / `calldescr`
    // (or to clone the Arc for the interpret fallback). Holding `&entry`
    // across `inline_call_native` would alias `bh`.
    let native = {
        let entry = bh.jitcode.exec.descrs.get(sub_idx).unwrap_or_else(|| {
            panic!(
                "BC_INLINE_CALL: descrs[{sub_idx}] is absent \
                 (runtime pool has {} items)",
                bh.jitcode.exec.descrs.len()
            )
        });
        // `bhimpl_inline_call_*` does not interpret a
        // callee.  It runs `cpu.bh_call_X(adr2int(jitcode.fnaddr), args_i, args_r,
        // args_f, jitcode.calldescr)`, so a whole callee subtree — a recursive tree
        // walk included — executes as compiled code in ONE call.  Upstream's
        // blackhole never byte-interprets a callee, and the ten canonical
        // `inline_call_*` handlers in this file are the same thing for the bytes a
        // build-time jitcode emits.
        //
        // `JitCodeBuilder::set_native_entry` stages that pair for a
        // `#[jit_inline]` helper: the Rust function the macro re-emits IS the
        // program the body was lowered from, so calling it answers what running
        // the bytes answers.  Where it did not stage one — a `match`-arm fragment,
        // which is a shape upstream does not have, or a body carrying an opcode
        // whose operand comes from this interpreter rather than from the heap
        // (`JitCodeBuilder::native_entry_denied`) — `fnaddr` is 0 and the nested
        // interpreter below runs it, which is where every callee went before.
        if let Some(sub) = entry.as_jitcode_exec()
            && is_callable_fnaddr(sub.fnaddr)
            && let Some(body) = sub.try_body()
        {
            Some((
                sub.fnaddr,
                &body.calldescr as *const majit_jitcode::jitcode::BhCallDescr,
            ))
        } else {
            None
        }
    };
    if let Some((fnaddr, calldescr)) = native {
        // SAFETY: `calldescr` lives in the JitCode Arc stored in this
        // frame's descrs table. `inline_call_native` does not rebuild it.
        return inline_call_native(bh, code, p, num_args, fnaddr, unsafe { &*calldescr });
    }

    let sub_jitcode = bh
        .jitcode
        .exec
        .descrs
        .get(sub_idx)
        .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode_owned)
        .unwrap_or_else(|| {
            panic!(
                "BC_INLINE_CALL: descrs[{sub_idx}] is not a JitCode entry \
                 (runtime pool has {} items)",
                bh.jitcode.exec.descrs.len()
            )
        });

    // The callee frame is seated before the argument triples are decoded so
    // each one can be copied where it is read, with no list in between.  The
    // frame itself comes from `bh`'s scratch slot when this frame has already
    // made an inline call, so a recursive descent allocates its frames once.
    let mut callee = bh
        .inline_callee_scratch
        .take()
        .unwrap_or_else(|| Box::new(BlackholeInterpreter::default()));
    callee.clone_context_from(bh);
    callee.setposition(sub_jitcode, 0);

    for _ in 0..num_args {
        let kind = JitArgKind::decode(jitcode::read_u8(code, &mut p));
        let caller_src = jitcode::read_reg(code, &mut p) as usize;
        let callee_dst = jitcode::read_reg(code, &mut p) as usize;
        match kind {
            JitArgKind::Int => {
                callee.registers_i[callee_dst] = bh.registers_i[caller_src];
            }
            JitArgKind::Ref => {
                callee.registers_r[callee_dst] = bh.registers_r[caller_src];
            }
            JitArgKind::Float => {
                callee.registers_f[callee_dst] = bh.registers_f[caller_src];
            }
        }
    }
    copy_identity_slots_from_parent(bh, &mut callee);

    let dest = decode_return_slot_at(code, &mut p);

    // RPython `bhimpl_inline_call_*` calls `cpu.bh_call_*(jitcode.fnaddr,
    // ...)` directly, so any `JitException` (`ContinueRunningNormally`,
    // `DoneWithThisFrame*`, `ExitFrameWithExceptionRef`) raised by the
    // callee bubbles up to the portal runner via the C-level call
    // (`blackhole.py:1623`).  pyre's nested interpreter executes the
    // sub-jitcode in-process, so we have to re-raise
    // `ContinueRunningNormally` ourselves; otherwise a callee that hits a
    // recursive merge point silently completes "normally" and the parent
    // frame never sees the portal-restart signal.
    //
    // RPython's generic `dispatch_loop` wrapper (`blackhole.py`)
    // sets `self.position = position` AFTER decoding operands/result
    // and BEFORE re-raising, so the parent frame's post-exception
    // inspectors see the post-op cursor.  Mirror that invariant here:
    // sync `bh.position` to the operand-decoded post-op `p` before
    // propagating the JitException.
    //
    // Every exit below leaves through the single `break 'callee`, so the
    // frame always reaches the scratch slot on its way out.
    let outcome = 'callee: {
        match callee.run() {
            BhRunOutcome::ContinueRunningNormally(args) => {
                bh.position = p;
                break 'callee Err(DispatchError::ContinueRunningNormally(args));
            }
            // Same reason as `resume_mainloop`: the caller copies the callee's
            // `return_type` / `tmpreg_*` into its own result register below,
            // and a callee that ran out of bytecode never wrote either.
            BhRunOutcome::EndOfCode => panic!(
                "inline_call: callee jitcode {:?} index {:?} ended without a \
                 return opcode, so there is no result for the caller",
                callee.jitcode.name,
                callee.jitcode.try_index(),
            ),
            BhRunOutcome::LeaveFrame | BhRunOutcome::Exception => {}
        }

        // RPython `blackhole.py` invariant: after the handler has
        // decoded operands and consumed the result payload, every
        // exceptional exit path must update `self.position` to the
        // post-op cursor before re-raising.  Mirror that here for the
        // aborted / got_exception paths — `dispatch_step` only writes
        // `self.position = new_pos` when the handler returns `Ok`, so any
        // `Err(LeaveFrame)` that skips the assignment leaves the parent's
        // cursor pre-op.
        if callee.aborted {
            bh.position = p;
            bh.aborted = true;
            bh.abort_permanent_bail = callee.abort_permanent_bail;
            if let Some(name) = callee.bail_residual.take() {
                bh.bail_residual = Some(name);
            }
            break 'callee Err(DispatchError::LeaveFrame);
        }
        if callee.got_exception {
            let exc_val = callee.exception_last_value;
            if exc_val != 0 {
                break 'callee Err(DispatchError::RaiseException {
                    exc: exc_val,
                    resume_position: p,
                });
            }
            bh.position = p;
            bh.got_exception = true;
            break 'callee Err(DispatchError::LeaveFrame);
        }

        assert!(
            dest.is_none() == (callee.return_type == BhReturnType::Void),
            "inline_call: void callee <=> no dest (callee {:?} return_type={:?} dest={dest:?})",
            callee.jitcode.name,
            callee.return_type,
        );
        let dest = dest.map(|dst| (return_kind_from_bh(callee.return_type), dst));
        copy_inline_callee_tmpreg(bh, &mut callee, dest);
        Ok(p)
    };
    callee.reset_for_inline_reuse();
    bh.inline_callee_scratch = Some(callee);
    outcome
}

/// The native arm of `BC_INLINE_CALL`: `bhimpl_inline_call_irf_*`
/// (`blackhole.py:1306-1319`) over the operands `inline_call_nested_ext/P`
/// spells them with.
///
/// `p` is positioned at the first argument triple; the return is the post-op
/// cursor, the same one the canonical handlers return.
fn inline_call_native(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
    num_args: usize,
    fnaddr: i64,
    calldescr: &majit_jitcode::jitcode::BhCallDescr,
) -> Result<usize, DispatchError> {
    let mut p = p;
    // Translated `bhimpl_inline_call_*` becomes a direct C call of the
    // known `calldescr` signature. `shift` is `rii -> i`; skip the
    // `collect_call_args` / dispatch-table walk that the generic stub
    // pays on every child of every character.
    if num_args == 3 && calldescr.arg_classes == "rii" && calldescr.result_type == 'i' {
        return inline_call_native_rii(bh, code, p, fnaddr);
    }
    // `descr.py create_call_stub` places arguments by declaration position,
    // and `collect_call_args` recovers that position by walking `arg_classes`
    // and taking the next value from the matching per-kind list.  So the lists
    // have to be in declaration order WITHIN each kind, which is not the order
    // the triples are emitted in — a caller groups them by kind and
    // `check_inline_call_arg_classes` documents that the two disagree.
    //
    // The callee register each triple names IS that position:
    // `inline_helper_param_layout` hands register k of a bank to the k'th
    // parameter of that kind.  Place by it.
    // `num_args` for each bank, not the per-kind count, because the counts
    // are not known until the triples are read.  It is an upper bound on all
    // three at once: `callee_dst` is `inline_helper_param_layout`'s DENSE
    // per-kind index, so a bank's largest index is (that kind's parameter
    // count - 1) and every kind's count is <= `num_args`.  The `truncate`s
    // below cut each bank back to what was actually placed, which is what
    // `collect_call_args` walks.
    // `BlackholeInterpreter.bhimpl_inline_call_irf_*` owns no callee frame;
    // the argument lists live only for this direct `cpu.bh_call_*`.  Reuse
    // their storage just like `setposition` reuses the interpreter's register
    // lists.  Take them out while the call runs so the shared `&self` call
    // helpers and the later mutable result/exception handling do not require
    // a Rust-only interior-mutability owner.
    let mut args_i = std::mem::take(&mut bh.native_inline_args_i_scratch);
    let mut args_r = std::mem::take(&mut bh.native_inline_args_r_scratch);
    let mut args_f = std::mem::take(&mut bh.native_inline_args_f_scratch);
    args_i.resize(num_args, 0);
    args_r.resize(num_args, 0);
    args_f.resize(num_args, 0);
    let (mut len_i, mut len_r, mut len_f) = (0usize, 0usize, 0usize);
    for _ in 0..num_args {
        let kind = JitArgKind::decode(jitcode::read_u8(code, &mut p));
        let caller_src = jitcode::read_reg(code, &mut p) as usize;
        let callee_dst = jitcode::read_reg(code, &mut p) as usize;
        let (bank, len, value) = match kind {
            JitArgKind::Int => (&mut args_i, &mut len_i, bh.registers_i[caller_src]),
            JitArgKind::Ref => (&mut args_r, &mut len_r, bh.registers_r[caller_src]),
            JitArgKind::Float => (&mut args_f, &mut len_f, bh.registers_f[caller_src]),
        };
        bank[callee_dst] = value;
        *len = (*len).max(callee_dst + 1);
    }
    args_i.truncate(len_i);
    args_r.truncate(len_r);
    args_f.truncate(len_f);

    let dest = decode_return_slot_at(code, &mut p);

    // A raise inside the callee reaches this side through the cell, exactly as
    // it does for a residual call and for the canonical `inline_call_*`
    // handlers above; clearing it first is what makes a stale value from an
    // earlier call unable to be read as this one's.
    bh.last_exc().set(0);

    // The local ref list must remain a moving-GC root during the residual
    // call.  In RPython it is a GC-managed list and the shadow-stack transform
    // roots it automatically.  pyre's explicit transform already uses this
    // slice-root mechanism for resume construction; register this borrowed
    // list for the strictly nested call window and pop it before returning the
    // storage to the interpreter cache.
    let args_root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(args_r.as_mut_slice());
    }
    let outcome = match calldescr.result_type {
        'i' => {
            let result = bh.bhimpl_inline_call_irf_i(fnaddr, &args_i, &args_r, &args_f, calldescr);
            let outcome = check_residual_call_exception_after(bh, p);
            if outcome.is_ok() {
                if let Some(dst) = dest {
                    bh.registers_i[dst] = result;
                }
            }
            outcome
        }
        'r' => {
            let result = bh.bhimpl_inline_call_irf_r(fnaddr, &args_i, &args_r, &args_f, calldescr);
            let outcome = check_residual_call_exception_after(bh, p);
            if outcome.is_ok() {
                if let Some(dst) = dest {
                    bh.registers_r[dst] = result.0 as i64;
                }
            }
            outcome
        }
        'f' => {
            let result = bh.bhimpl_inline_call_irf_f(fnaddr, &args_i, &args_r, &args_f, calldescr);
            let outcome = check_residual_call_exception_after(bh, p);
            if outcome.is_ok() {
                if let Some(dst) = dest {
                    bh.registers_f[dst] = result.to_bits() as i64;
                }
            }
            outcome
        }
        _ => {
            bh.bhimpl_inline_call_irf_v(fnaddr, &args_i, &args_r, &args_f, calldescr);
            check_residual_call_exception_after(bh, p)
        }
    };
    majit_gc::shadow_stack::pop_resume_ref_roots_to(args_root_depth);

    // Clear the logical contents (especially dead refs) while retaining the
    // three capacities for the next native inline call.  Restore them before
    // propagating `outcome`, so exceptional calls reuse storage too.
    args_i.clear();
    args_r.clear();
    args_f.clear();
    bh.native_inline_args_i_scratch = args_i;
    bh.native_inline_args_r_scratch = args_r;
    bh.native_inline_args_f_scratch = args_f;
    outcome?;
    Ok(p)
}

/// Mirror of `BlackholeInterpreter::decode_return_slot` for handler
/// callsites that thread a local cursor instead of mutating
/// `self.position`.  `NO_RETURN_REG` encodes the "no caller destination"
/// sentinel.
/// `shift(n, c, mark)`: one Ref then two Ints, result Int.
/// Same placement as `inline_call_native` (`callee_dst` is the dense
/// per-kind index) and the same `extern "C" fn(i64, i64, i64)` ABI
/// `bh_call_i_dispatch` uses for three integer-class arguments.
fn inline_call_native_rii(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    mut p: usize,
    fnaddr: i64,
) -> Result<usize, DispatchError> {
    let mut r0 = 0i64;
    let mut i0 = 0i64;
    let mut i1 = 0i64;
    for _ in 0..3 {
        let kind = JitArgKind::decode(jitcode::read_u8(code, &mut p));
        let caller_src = jitcode::read_reg(code, &mut p) as usize;
        let callee_dst = jitcode::read_reg(code, &mut p) as usize;
        match kind {
            JitArgKind::Ref => r0 = bh.registers_r[caller_src],
            JitArgKind::Int if callee_dst == 0 => i0 = bh.registers_i[caller_src],
            JitArgKind::Int => i1 = bh.registers_i[caller_src],
            JitArgKind::Float => {
                unreachable!("rii calldescr has no float argument");
            }
        }
    }
    let dest = decode_return_slot_at(code, &mut p);
    bh.last_exc().set(0);
    // Same `cpu.bh_call_*` leave as `bhimpl_inline_call_*`; this rii
    // fast path does not go through those helpers.
    let args_root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut r0));
    }
    let result = unsafe {
        let f: unsafe extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(fnaddr as usize);
        f(r0, i0, i1)
    };
    let outcome = check_residual_call_exception_after(bh, p);
    majit_gc::shadow_stack::pop_resume_ref_roots_to(args_root_depth);
    outcome?;
    if let Some(dst) = dest {
        bh.registers_i[dst] = result;
    }
    Ok(p)
}

fn decode_return_slot_at(code: &[u8], cursor: &mut usize) -> Option<usize> {
    let dst = jitcode::read_reg(code, cursor) as usize;
    if dst == jitcode::NO_RETURN_REG as usize {
        None
    } else {
        Some(dst)
    }
}

// recursive_call
/// RPython `blackhole.py` `get_portal_runner(jdindex)`:
/// Returns (fnptr, calldescr) from jitdrivers_sd[jdindex].
/// pyre: uses portal_runner_ptr directly (single jitdriver).
///
/// RPython `blackhole.py`:
/// ```python
/// def bhimpl_recursive_call_i(self, jdindex, greens_i, greens_r, greens_f,
///                                            reds_i, reds_r, reds_f):
///     fnptr, calldescr = self.get_portal_runner(jdindex)
///     return self.cpu.bh_call_i(fnptr, greens_i+reds_i, greens_r+reds_r,
///                               greens_f+reds_f, calldescr)
/// ```
/// Read recursive_call args and merge greens+reds per kind.
/// Returns (jdindex, all_i, all_r, all_f, next_position).
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn read_recursive_call_args(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (
    usize,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    usize,
) {
    // RPython `blackhole.py:1101-1132` `bhimpl_recursive_call_*` declares
    // jd_index as `@arguments("self", "i", ...)` — `i` argcode means read
    // through the int-register file. The build-time assembler emits a
    // register-index (into the constant pool tail of `registers_i`), not
    // the jd_index literal byte, so dispatch must deref through the
    // register file to recover the jd_index value.
    let jdindex = bh.registers_i[code[p] as usize] as usize;
    let p = p + 1;
    let (greens_i, p) = read_list_i(bh, code, p);
    let (greens_r, p) = read_list_r(bh, code, p);
    let (greens_f, p) = read_list_f(bh, code, p);
    let (reds_i, p) = read_list_i(bh, code, p);
    let (reds_r, p) = read_list_r(bh, code, p);
    let (reds_f, p) = read_list_f(bh, code, p);
    // The greens+reds concatenation per kind happens inside
    // bhimpl_recursive_call_* (mirroring blackhole.py:1105-1108) — the
    // handlers hand over the raw greens / reds tuples so the bhimpl
    // method owns the merge.
    (
        jdindex,
        greens_i.into_vec(),
        greens_r.into_vec(),
        greens_f.into_vec(),
        reds_i.into_vec(),
        reds_r.into_vec(),
        reds_f.into_vec(),
        p,
    )
}
// blackhole.py bhimpl_recursive_call_i
fn handler_recursive_call_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.last_exc().set(0);
    let result = bh.bhimpl_recursive_call_i(
        jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
    );
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_i[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py bhimpl_recursive_call_r
fn handler_recursive_call_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.last_exc().set(0);
    let result = bh
        .bhimpl_recursive_call_r(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        )
        .0 as i64;
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py bhimpl_recursive_call_f
fn handler_recursive_call_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.last_exc().set(0);
    let result = bh
        .bhimpl_recursive_call_f(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        )
        .to_bits() as i64;
    check_residual_call_exception_after(bh, p + 1)?;
    bh.registers_f[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py bhimpl_recursive_call_v
fn handler_recursive_call_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.last_exc().set(0);
    bh.bhimpl_recursive_call_v(
        jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
    );
    check_residual_call_exception_after(bh, p)?;
    Ok(p)
}
