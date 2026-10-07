//! Tracing context: wraps the recorder Trace with a convenience API.
//!
//! `TraceCtx` owns the struct definition, constructors, accessors,
//! constant management, and virtualizable machinery.  The recording
//! and compile-bookkeeping roles are split across sibling modules:
//!
//! * **History role** → `history.rs` `impl TraceCtx`:
//!   `record_*`, `get_trace_position`, `cut_trace`, `replace_box`,
//!   `into_tree_loop`, all `call_*` / `guard_*` recording wrappers.
//!
//! * **Compile role** → `compile.rs` `impl TraceCtx`:
//!   `add/clear/get/has_merge_point*`,
//!   inline-trace tracking (`push/pop_inline_*`, `recursive_depth`).
//!
//! `MergePoint` is defined here alongside `current_merge_points`,
//! matching RPython where `MetaInterp` (pyjitpl.py) owns both.
//!
//! Remaining convergence: reshape `MetaInterp` fields from
//! `meta.trace_ctx` to `meta.history` + `meta.trace` (upstream
//! parity); that cascades into every call site of `TraceCtx::*`.

use crate::heapcache::{HeapCache, HeapCacheView, HeapCacheViewMut};
use crate::opencoder::Box as OcBox;
use crate::recorder::Trace;
use indexmap::IndexMap;
use majit_ir::{DescrRef, GcRef, GreenKey, GreenType, OpCode, OpRef, Type, Value};

use majit_backend::JitCellToken;

use crate::jitcode::JitArgKind;
// `make_resume_guard_descr*` is no longer needed at the tracer side —
// guards record `descr=None` and the optimizer's
// `store_final_boxes_in_guard` mints the descr (codex #3 / pyjitpl.py:2548
// generate_guard parity).
use crate::jitdriver::JitDriverStaticData;
use crate::virtualizable::VirtualizableInfo;

/// Project a tracer-side `DescrRef` to a backend `BhDescr::Field` for
/// `executor::do_getfield_gc_*` consumption (the cache-hit sanity check
/// load path).
///
/// RPython `executor.execute(cpu, mi, opnum, fielddescr, box)` takes
/// `fielddescr` directly as an `AbstractDescr` — backend cpu methods
/// (`cpu.bh_getfield_gc_*`) read offset/size off the same descr.
/// Pyre's two-tier descr model splits the runtime trace-level
/// `Arc<dyn FieldDescr>` from the build-time `BhDescr::Field` enum,
/// so the bridge has to fish out the offset/size/type/flag/name and
/// reassemble a `BhDescr::Field` on the stack at the call site.
/// `bh_getfield_gc_r` reads that name for `load_is_acquire`.
///
/// Returns `None` for non-field descrs (the sanity check is then
/// skipped at the caller — same behavior as `cpu == None`).
fn descr_to_bh_field_descr(descr: &DescrRef) -> Option<majit_jitcode::jitcode::BhDescr> {
    let f = descr.as_field_descr()?;
    Some(majit_jitcode::jitcode::BhDescr::Field {
        offset: f.offset(),
        field_size: f.field_size(),
        field_type: f.field_type(),
        // `bh_getfield_gc_r` reads `load_is_acquire`, which needs the
        // pointer flag, the quasi bit, and the field name. The signedness
        // the integer load uses is `is_field_signed`, not this flag.
        field_flag: bh_field_flag(f),
        is_field_signed: f.is_field_signed(),
        is_immutable: f.is_immutable(),
        is_quasi_immutable: descr.is_quasi_immutable(),
        // A live `FieldDescr` reached `get_field_descr` already, so its index
        // is the parent's arbitrated answer, not an unresolved claim.
        index_in_parent: Some(f.index_in_parent()),
        parent: None,
        name: f.field_name().to_string(),
        owner: String::new(),
        load_is_acquire: f.stored_load_is_acquire(),
    })
}

/// `BhFieldSpec::from_field_descr`'s flag: a pointer field stays
/// `FLAG_POINTER` so `load_is_acquire` can see it.
fn bh_field_flag(f: &dyn majit_ir::descr::FieldDescr) -> majit_ir::ArrayFlag {
    if f.is_pointer_field() {
        majit_ir::ArrayFlag::Pointer
    } else if f.is_float_field() {
        majit_ir::ArrayFlag::Float
    } else if f.field_type() == Type::Void {
        majit_ir::ArrayFlag::Void
    } else if f.is_field_signed() {
        majit_ir::ArrayFlag::Signed
    } else {
        majit_ir::ArrayFlag::Unsigned
    }
}

/// Project a tracer-side `DescrRef` to a backend `BhDescr::Array` for
/// `executor::do_getarrayitem_gc_*` consumption — the array-side
/// analogue of [`descr_to_bh_field_descr`].
///
/// `bh_getarrayitem_gc_*` consumes the descr through
/// `unpack_arraydescr_size` (`base_size`, `itemsize`, `is_item_signed`)
/// + the `array_base_size()` accessor for Ref/Float reads; the
///   remaining `BhDescr::Array` fields are placeholder defaults the load
///   path never reads.  Returns `None` for non-array descrs.
fn descr_to_bh_array_descr(descr: &DescrRef) -> Option<majit_jitcode::jitcode::BhDescr> {
    let a = descr.as_array_descr()?;
    Some(majit_jitcode::jitcode::BhDescr::Array {
        base_size: a.base_size(),
        itemsize: a.item_size(),
        len_offset: a.len_descr().map(|fd| fd.offset()),
        type_id: a.cache_key(),
        gc_type_id: a.type_id(),
        item_type: a.item_type(),
        is_array_of_pointers: a.is_array_of_pointers(),
        is_array_of_structs: false,
        is_item_signed: a.is_item_signed(),
        ei_index: u32::MAX,
        array_type_id: None,
        interior_fields: Vec::new(),
        is_gc_managed: a.is_gc_managed(),
        load_is_acquire: a.load_is_acquire(),
    })
}

fn descr_to_bh_size_descr(descr: &DescrRef) -> Option<majit_jitcode::jitcode::BhDescr> {
    let size = descr.as_size_descr()?;
    Some(majit_jitcode::jitcode::BhDescr::Size {
        size: size.size(),
        // `descr.py` cache identity via `SizeDescr.cache_key()`, matching
        // `eval.rs` `bh_new`. The collector tid lives on `type_id()` and
        // is recovered by `resolved_gc_tid_checked`.
        type_id: size.cache_key(),
        vtable: size.vtable() as u64,
        owner: String::new(),
        all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(size),
        is_gc_managed: size.is_gc_managed(),
    })
}

/// Inverse of `heap_value_for`: encode a typed `Value` into the raw i64
/// bit-pattern that `VirtualizableInfo::write_field`/`write_array_item`
/// interpret per field/item type.
pub(crate) fn value_to_raw_bits(value: Value) -> i64 {
    match value {
        Value::Int(v) => v,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.as_usize() as i64,
        Value::Void => 0,
    }
}

/// pyjitpl.py:1135-1138 `rop.PTR_EQ` runtime outcome.  Compare the
/// concrete ptrs carried by two Refs (virtualizable identity).  `None`
/// (no concrete known) and non-Ref values are never the standard box, so
/// falling into the catch-all `false` branch preserves the Step 4 "not
/// standard" path in `begin_nonstandard_virtualizable` — this is how an
/// `opref` with no resolvable concrete, or one backed by a non-Ref
/// constant (e.g. `ConstInt(0xCAFE)` in a test), still resolves to
/// `isstandard = 0` and proceeds to Step 5 / `emit_force_virtualizable`,
/// matching upstream's runtime behavior for the same bogus input.
fn concrete_ptrs_eq(a: Option<&Value>, b: Option<&Value>) -> bool {
    match (a, b) {
        (Some(Value::Ref(ra)), Some(Value::Ref(rb))) => ra == rb,
        _ => false,
    }
}

/// Reinterpret a virtualizable box's concrete as an optional value.
/// A box may carry `Value::Void` when no runtime concrete is recorded;
/// surface those as `None` so the vable-field read channel carries
/// the "unknown" marker explicitly rather than a synthesized `Void`.
fn concrete_shadow_value(value: Value) -> Option<Value> {
    match value {
        Value::Void => None,
        v => Some(v),
    }
}

/// The address a `GcRef` names, or `None` when it names no live object: the
/// unset marker, the all-ones tombstone, or NULL. A load through any of the
/// three reads memory the object model does not own.
fn live_gc_ptr(gcref: majit_ir::GcRef) -> Option<i64> {
    if gcref == majit_ir::GcRef::NO_CONCRETE {
        return None;
    }
    let ptr = gcref.0 as i64;
    (ptr != 0 && ptr != usize::MAX as i64).then_some(ptr)
}

/// pyjitpl.py:2989 box-with-type pair.
///
/// RPython's `Box` carries type implicitly via Python class identity
/// (`ConstInt`/`ConstFloat`/`InputArgRef` etc.).  Pyre's flat-OpRef
/// encoding stores type as a separate `Type` tag, so `GreenBox` bundles
/// the position + type tag into one struct, folding away the previous
/// parallel `Vec<OpRef>` + `Vec<Type>` adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreenBox {
    pub opref: OpRef,
    pub ty: Type,
}

impl GreenBox {
    pub fn new(opref: OpRef, ty: Type) -> Self {
        Self { opref, ty }
    }

    /// One GreenBox per recorder inputarg, in header order.
    ///
    /// `TraceCtx::new` / `with_green_key` used to collect `inputarg_types()`,
    /// clone that Vec, then collect a parallel `Vec<OpRef>` and zip them.
    /// GreenBox exists so those two lists are never materialized
    /// (`pyjitpl.py` live_arg_boxes already carry type on the box).
    pub(crate) fn from_recorder_inputargs(recorder: &Trace) -> Vec<Self> {
        recorder
            .inputargs()
            .iter()
            .map(|arg| {
                let tp = arg.tp.get();
                Self::new(arg.opref(), tp)
            })
            .collect()
    }
}

/// pyjitpl.py:2989 — a visited loop header with its trace position.
///
/// RPython stores `(original_boxes, start)` where `original_boxes` is the
/// full list of green+red args at the first visit, and `start` is a 5-tuple
/// trace position. majit stores the equivalent as a `Vec<GreenBox>` (each
/// pairing OpRef + Type tag) + TracePosition.
#[derive(Clone, Debug)]
pub struct MergePoint {
    /// Green key of the loop header.
    pub green_key: u64,
    /// The typed green key `green_key` is the `JitCell.get_uhash` of
    /// (warmstate.py `def get_uhash(*greenargs)`).
    ///
    /// `green_key` alone names a BUCKET, not a cell: `_get_index` truncates it
    /// (counter.py), so several cells share one chain and
    /// `JitCell.comparekey` is what picks this one out of it
    /// (warmstate.py `def comparekey(self, *greenargs2)`). A consumer
    /// that reaches a cell through the hash alone installs one carrying no
    /// comparekey, which no later typed lookup can match — so the same loop
    /// header ends up owning a second cell with its own token and flags.
    /// Upstream never creates a cell from a hash: `_ensure_jit_cell_at_key`
    /// (warmstate.py) is handed the greens themselves, and the one
    /// bare-hash entry point, `trace_next_iteration_hash`
    /// (warmstate.py), only moves a counter.
    ///
    /// The green boxes this merge point was registered with
    /// (`pyjitpl.py` `current_merge_points.append((live_arg_boxes, start))`).
    pub green_key_typed: majit_ir::GreenKey,
    /// Trace position when this loop header was first visited.
    pub position: crate::recorder::TracePosition,
    /// pyjitpl.py:2989: `original_boxes` — live variable boxes (OpRef +
    /// type tag) at the first visit to this loop header. Used by
    /// compile_loop/compile_retrace as the inputargs for trace cutting.
    pub green_boxes: Vec<GreenBox>,
    /// Bytecode PC of this loop header. Used by cut_trace_from to update
    /// meta when the trace closes at a different loop.
    pub header_pc: usize,
    /// The virtualizable `green_boxes` was snapshotted against, as a raw
    /// address (0 when none was live at registration).
    ///
    /// RPython's `original_boxes` are Boxes that carry their concrete value
    /// inline, so `compile.py:510
    /// orig_inpargs[index_of_virtualizable].getref_base()` reads the frame
    /// belonging to the very list that becomes `loop.inputargs`. A
    /// [`GreenBox`] is symbolic, so the address is paired with the snapshot
    /// here to keep that read exact — the trace-level
    /// `virtualizable_heap_ptr` tracks the walk and names a different frame
    /// once the trace is cut to a header the walk reached later
    /// (compile.py:269).
    pub vable_ptr: usize,
}

/// Tracing context: wraps the recorder Trace with a convenience API.
///
/// The interpreter uses this during trace recording to:
/// - Record IR operations
/// - Carry inline constant operands on the OpRef variants
/// - Record guards (with auto-generated FailDescr)
/// - Record function calls (with auto-generated CallDescr)
pub struct TraceCtx {
    pub(crate) recorder: Trace,
    /// opencoder.py:472 `self.metainterp_sd = metainterp_sd` — the trace
    /// recorder holds a shared reference to the JIT's static data so
    /// `_encode_descr` can route global descriptors through
    /// `metainterp_sd.all_descrs`. Pyre tracks it on TraceCtx instead
    /// of `recorder::Trace` because the swap to `TraceRecordBuffer`
    /// needs the Arc available at constructor time; wiring
    /// it at the TraceCtx layer lets the eventual swap reuse this
    /// plumbing without threading more parameters through
    /// `MetaInterp::setup_tracing` etc.
    pub(crate) metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    pub(crate) green_key: u64,
    root_green_key: u64,
    /// Structured `(code_ptr, pc)` counterpart to `green_key`. Keeps
    /// pyjitpl.py:1396-1401's element-wise `same_constant` parity when
    /// comparing against inline-frame green keys; the u64 `green_key`
    /// above is the hash derived from this pair and stays the identity
    /// key for HashMap lookups (warmstate / compiled_loops / pending
    /// token) while comparisons route through the raw pair.
    pub(crate) green_key_raw: (usize, usize),
    pub(crate) root_green_key_raw: (usize, usize),
    /// Stack of inlined function frames (callee green keys as raw
    /// `(code_ptr, pc)` pairs). rpython/jit/metainterp/pyjitpl.py:1390
    /// walks `self.metainterp.framestack` element-wise; pyre mirrors
    /// that by storing the structured greenkey per inline frame and
    /// doing tuple-equality comparisons in [`recursive_depth`] and
    /// [`is_tracing_key`].
    pub(crate) inline_frames: Vec<(usize, usize)>,
    /// Retired. `JitCodeMachine` writes the portal log through
    /// [`Self::portal_trace_push_fn`] into `MetaInterp.portal_trace_positions`.
    /// Structured green key values (if provided by the interpreter).
    green_key_values: Option<GreenKey>,
    /// Declarative driver layout metadata, if provided by the interpreter.
    pub(crate) driver_descriptor: Option<JitDriverStaticData>,
    /// Standard virtualizable boxes -- OpRefs for each static field + array element.
    /// When set, vable_getfield/setfield access these instead of emitting heap ops.
    /// Layout: [field_0, ..., field_N, arr_0[0], ..., arr_0[M], ..., vable_ref]
    ///
    /// The last element (`boxes[-1]`) is the standard virtualizable identity
    /// (RPython parity: `virtualizable_boxes[-1]`). Used by gen_store_back_in_vable
    /// to distinguish standard vs nonstandard virtualizable.
    pub(crate) virtualizable_boxes: Option<Vec<OpRef>>,
    /// Per-slot provenance for the virtualizable boxes: the last executed store
    /// into this flat slot wrote a live NULL Ref (the NULL companion an operand
    /// stack push leaves behind), as opposed to a slot no store has touched. Same
    /// layout as `virtualizable_boxes`; `None` when the boxes were seeded without
    /// concretes (bridge-entry rebuild / init-before-run).
    virtualizable_live_null_slots: Option<Vec<bool>>,
    /// VirtualizableInfo for the standard virtualizable (if any).
    virtualizable_info: Option<std::sync::Arc<VirtualizableInfo>>,
    /// Lengths of each virtualizable array field, needed for flat index computation.
    virtualizable_array_lengths: Option<Vec<usize>>,
    /// Live virtualizable heap pointer (pyjitpl.py:3446 write_boxes target).
    /// Seeded at trace entry from the virtualizable box, with
    /// `MetaInterp::pending_vable_ptr` only as the fallback for a vable that is
    /// not a red, and at bridge entry from the retrace's live vable.  It does
    /// not stay put for the session: a compiled entry moves it to the frame
    /// that entry runs and the exits put it back, and a residual call restores
    /// it on return.  Used by
    /// `synchronize_virtualizable` to write each box's concrete back to
    /// the live PyFrame after every standard vable setfield / setarrayitem
    /// (`virtualizable.py write_boxes`). `None` disables the
    /// write — unit-test or init-before-run path.
    virtualizable_heap_ptr: Option<*const u8>,
    /// Set when the current residual call receives a virtualizable array's raw
    /// base address. Drained by the post-call escape check.
    raw_vable_base_escape_pending: bool,
    /// Header PC at which this trace started (0 = function entry).
    pub header_pc: usize,
    /// When a cross-loop cut occurs (trace closes at inner loop header),
    /// the green key for the inner loop. Used to register an alias
    /// so can_enter_jit at the inner back-edge finds the outer key's entry.
    pub cut_inner_green_key: Option<u64>,
    /// Transient signal set when a back-edge is reached while an inline
    /// callee frame is active (opimpl_jit_merge_point portal_call_depth>0).
    /// Such a loop must not be unrolled as a root loop; the trace step
    /// reads and clears this to abort instead. See `request_inline_loop_abort`.
    pub(crate) inline_loop_abort_pending: bool,
    /// Transient signal set when an inline-frame back-edge can take the
    /// orthodox path instead of aborting: the callee loop's green key has
    /// compiled code, so the metainterp pops the inline frame and records
    /// a CALL_ASSEMBLER into the loop token from the parent
    /// (opimpl_jit_merge_point portal_call_depth>0 → finishframe +
    /// do_recursive_call(assembler_call=True), pyjitpl.py).
    /// `(green_key, target_pc)` of the callee loop header. See
    /// `request_recursive_call_assembler`.
    pub(crate) recursive_call_assembler_pending: Option<(u64, usize)>,
    /// pyjitpl.py:3030 current_merge_points — loop headers visited during
    /// tracing with their trace positions. First visit records the key +
    /// position; second visit closes the loop.
    pub(crate) current_merge_points: Vec<MergePoint>,
    /// pyjitpl.py `same_greenkey` reference — the trace-start loop
    /// header's concrete green constants, grouped by IR register slot
    /// (`(ints, refs, floats)`).  Captured on the first merge-point visit of a
    /// primary trace (the header), then compared element-wise against every
    /// later visit's greens to decline closing when a scalar green differs
    /// from the header.  `None` until captured;
    /// bridges never populate it (they close through the compiled-loop
    /// registry).  This is the merge-point green vocabulary — distinct from
    /// `green_key_values`, the back-edge/can_enter_jit key, which carries a
    /// different arity and cannot be compared against a merge point directly.
    pub(crate) header_greens: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>,
    /// pyjitpl.py:3005 `greenboxes` at the merge point the trace actually
    /// closed on, in the same `(ints, refs, floats)` slot grouping as
    /// `header_greens`.  Upstream reads `get_procedure_token(greenboxes)` off
    /// the greens of the merge point just reached, so a bridge that closes on
    /// an inner merge point must name THAT loop, not the parent it originated
    /// from.  Set by both close paths; `None` when the trace did not close on
    /// a merge point.
    pub(crate) close_greens: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>,
    /// Current portal green banks, re-read from the merge-point registers
    /// at Finish so a Halt after a green write still has the live values.
    pub(crate) live_portal_greens: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>,
    /// Merge-point green register bytes, declaration order per bank.
    /// Abort / SegmentedLoop re-read those slots off the live frame the
    /// way `snapshot_live_portal_greens` does at Finish.
    pub(crate) portal_green_regs_i: Vec<u8>,
    pub(crate) portal_green_regs_r: Vec<u8>,
    pub(crate) portal_green_regs_f: Vec<u8>,
    /// Merge-point red register bytes, `(I, R, F)` operand order.
    /// `prepare_list_of_boxes` copies these slots; header-revisit CloseLoop
    /// re-reads them off the live portal frame or the last-pop snapshot.
    pub(crate) portal_red_regs_i: Vec<u8>,
    pub(crate) portal_red_regs_r: Vec<u8>,
    pub(crate) portal_red_regs_f: Vec<u8>,
    /// Red `OpRef`s from [`Self::portal_red_regs_*`] at the last portal
    /// frame, for a Continue walk that has already dropped that frame.
    pub(crate) live_portal_reds: Option<Vec<(OpRef, Type)>>,
    /// The int pc green that belongs to [`Self::close_greens`].
    /// Walk resume (`resume_walk_after_close` / `close_header_pc`) reads it
    /// as the merge-point's guest pc; the close key itself is the declared
    /// greens (`merge_point_green_key`).
    pub(crate) close_green_pc: Option<i64>,
    /// Typed greens of the merge point that closed the trace, when the
    /// producer built them itself. `close_green_key` returns this before
    /// reconstructing from the int/ref/float banks.
    close_typed_key: Option<GreenKey>,
    /// pyjitpl.py `compile_trace(live_arg_boxes, ptoken)`: the procedure
    /// token key of the merge point the trace just reached, set when that merge point
    /// already has compiled targets.  A close carrying this JUMPs into the existing
    /// loop instead of compiling a new one, and is read-and-cleared by the driver.
    pub(crate) close_jump_into_key: Option<u64>,
    /// pyjitpl.py reached_loop_header parity: callback to check
    /// has_compiled_targets(ptoken) for a given green key. Bridge traces
    /// skip loop headers without compiled targets. Live lookup (not snapshot)
    /// matches RPython's get_procedure_token(greenboxes) + has_compiled_targets.
    pub has_compiled_targets_fn: Option<Box<dyn Fn(u64) -> bool>>,
    /// pyjitpl.py `ptoken = self.get_procedure_token(greenboxes)` for the
    /// greens of the merge point just reached, in the same `(ints, refs,
    /// floats)` slot grouping as `Self::close_greens`. `Some(key)` iff a
    /// compiled loop with jumpable targets already lives at those greens
    /// (`MetaInterp::compiled_key_for_greens`, which folds in
    /// `has_compiled_targets`).
    ///
    /// [`Self::has_compiled_targets_fn`] cannot answer this: it is keyed on the
    /// u64 green key, and the dispatch loop cannot derive a foreign merge
    /// point's key. `green_key_from_code_ptr(green_key_raw.0, pc)` is not it —
    /// `JitState::code_ptr()` defaults to 0, and the driver's key is
    /// `GreenKey::hash_u64` over the declared green tuple.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub compiled_key_for_greens_fn:
        Option<Box<dyn Fn(&(Vec<i64>, Vec<i64>, Vec<i64>)) -> Option<u64>>>,
    /// Explicit "this trace started from a guard failure" flag. RPython
    /// distinguishes via `self.resumekey` typing (`ResumeGuardDescr` vs
    /// `ResumeFromInterpDescr`); pyre sets this to `true` at
    /// `start_bridge_tracing` and leaves the default `false` for
    /// primary entries.
    ///
    /// This is NOT `MetaInterp::partial_trace`. That flag is set only by
    /// `retrace_needed` (pyjitpl.py) and means "this is a
    /// RETRACE"; a bridge from a guard failure has `partial_trace = None`
    /// and takes every `if not self.partial_trace:` branch. The walker
    /// cannot read `MetaInterp::partial_trace` while it holds this ctx, so
    /// the gate stays on the driver (`JitDriver::merge_point`). Gating it
    /// on `is_bridge_trace` would skip `compile_trace` for every guard
    /// bridge, which upstream does not.
    ///
    /// Consumers that need bridge-only behavior
    /// (e.g. `pyre-jit-trace::pyjitpl::run_to_end`'s close-loop
    /// skip when no compiled targets exist for the current
    /// greenkey) gate on this flag instead of fn presence.
    pub is_bridge_trace: bool,
    /// Set true during the walk when a LOAD_GLOBAL / LOAD_NAME resolves through
    /// the frame's module globals dict.  Read by
    /// `finish_trace_namespace_dependency` at walk end and by the entry-bridge
    /// fold mid-walk; drives `PyreMeta.namespace_dependent` (the re-entry
    /// namespace-length gate).  Per-trace: a fresh ctx starts `false`, so no
    /// manual reset is needed.
    pub reads_module_global: bool,
    /// For a bridge trace (`is_bridge_trace`), the loop-header bytecode pc of
    /// the parent loop the bridge will JUMP into. The bridge closes when it
    /// reaches this pc (a real compiled-loop header), NOT when it transiently
    /// revisits its own `resume_pc` (`header_pc`). `None` for primary traces
    /// and for bridges whose parent loop header pc is unknown.
    pub bridge_target_header_pc: Option<usize>,
    /// pyjitpl.py:1551 `if self.metainterp.portal_call_depth: return` parity
    /// — live read of `MetaInterp.portal_call_depth` at the
    /// `BC_JIT_MERGE_POINT` first-iteration auto loop-header gate.  When
    /// nested portal calls are active (`portal_call_depth != 0`), RPython
    /// skips the auto-stamp and waits for an explicit `loop_header` op.
    /// Pyre exposes this as a Fn pointer so the trace ctx (which owns
    /// the cross-component flow at dispatch time) can sample the
    /// metainterp's depth counter without holding a back-reference.
    pub portal_call_depth_fn: Option<Box<dyn Fn() -> i32>>,
    /// pyjitpl.py `MetaInterp.newframe` / `popframe` `portal_call_depth` ± 1
    /// for a `JitCodeMachine` that cannot borrow `MetaInterp`.
    pub portal_call_depth_adjust_fn: Option<Box<dyn Fn(i32)>>,
    /// pyjitpl.py `MetaInterp.unique_id_for_greenkey` for
    /// `ENTER_PORTAL_FRAME` recorded by the dispatch machine.
    pub unique_id_for_greenkey_fn:
        Option<Box<dyn Fn(usize, &crate::pyjitpl::PortalGreenKey) -> u64>>,
    /// pyjitpl.py `self.metainterp.call_ids[-1]` at `debug_merge_point`.
    pub current_call_id_fn: Option<Box<dyn Fn() -> u64>>,
    /// `newframe`/`popframe` log half for a `JitCodeMachine` that cannot
    /// borrow `MetaInterp`. Forwards to `MetaInterp.push_portal_trace_position`.
    pub portal_trace_push_fn: Option<
        Box<dyn Fn(usize, Option<crate::pyjitpl::PortalGreenKey>, crate::recorder::TracePosition)>,
    >,
    /// pyjitpl.py `MetaInterp.seen_loop_header_for_jdindex` parity for
    /// walkers that drive dispatch through `TraceCtx` (the pyre full-body
    /// walker has no dispatcher struct of its own, so the per-trace flag
    /// lives here; majit's own `pyjitpl::dispatch` keeps an equivalent
    /// field on the dispatcher).  Stamped by a `loop_header` op
    /// (pyjitpl.py, the lowered `can_enter_jit` at a backward
    /// jump), consumed and reset by the following `jit_merge_point`
    /// (pyjitpl.py).  `-1` = not seen.
    pub seen_loop_header_for_jdindex: i32,
    /// JitCode coordinate of the explicit `loop_header` that set
    /// [`Self::seen_loop_header_for_jdindex`].  The codewriter emits that op in
    /// the `JUMP_BACKWARD` block, so its containing Python coordinate is the
    /// back-edge instruction rather than the target `jit_merge_point`.
    /// Consumed together with the driver-index stamp.  `None` covers automatic
    /// loop headers and resume-at-position pre-arming, which have no explicit
    /// back-edge op at that crossing.
    pub seen_loop_header_jit_pc: Option<usize>,
    /// pyjitpl.py:2941-2942 `if isinstance(key, compile.ResumeAtPositionDescr):
    /// self.seen_loop_header_for_jdindex = self.jitdriver_sd.index` — a bridge
    /// grown from a guard `inline_short_preamble` replayed (unroll.py /
    /// :409) starts with the loop header already counted as seen, so its first
    /// merge point closes instead of running the auto-stamp ladder.
    ///
    /// Upstream stores the driver index directly; pyre arms a request here
    /// because the registered slot and the merge-point op's `jdindex` can
    /// disagree (the `ensure_default_driver_sd` placeholder shift), and the
    /// merge point asserts the flag equals its own `jdindex`. Consumed
    /// (`take`) by the first merge point, which then stamps that `jdindex`.
    pub bridge_resume_at_position: bool,
    /// pyjitpl.py/1574 `saved_pc = self.pc` / `self.pc = saved_pc`, the
    /// "do not re-consult the merge point" half. [`Self::walk_resume_pc`] is
    /// the position half.
    ///
    /// Upstream consults a merge point once per visit and then resumes the
    /// frame just after it, so the loop body runs before the next consult.
    /// Pyre fuses the merge point onto the guest instruction hosting it, so a
    /// walk re-entered at the merge point's own guest pc — the
    /// `current_merge_points.append` path, which keeps tracing instead of
    /// ending the trace — would re-consult it with nothing recorded in
    /// between. Set on that re-entry and consumed by the next
    /// `BC_JIT_MERGE_POINT`, which then falls straight through.
    pub merge_point_resumed: bool,
    /// The guest pc a re-entered walk must be seeded at — the position half of
    /// `self.pc = saved_pc`.
    ///
    /// Upstream consults exactly one merge point per `opimpl_jit_merge_point`
    /// call, so `saved_pc` is a function of that visit's own `orgpc` and the
    /// position that entered the call cannot differ from the position that
    /// declined. Pyre runs a whole walk segment — many merge-point visits —
    /// per `JitDriver::merge_point` call, and a walk re-entered from inside
    /// that call is re-seeded from the closure-captured `__pc`, which is where
    /// the walk STARTED and which the native loop has not advanced. Left
    /// unset that rewinds the position while the symbolic state stays at the
    /// merge point the walk reached, and the instructions between the two are
    /// re-executed against state they do not belong to.
    ///
    /// Set to the declining merge point's own guest pc; consumed (`take`) by
    /// the `jit_merge_point!` expansion before it hands the pc to the walk.
    /// `None` for a re-entry that publishes no resume pc, which keeps the
    /// closure-captured `__pc`.
    pub walk_resume_pc: Option<usize>,
    /// pyjitpl.py: `metainterp.staticdata.callinfocollection`. Needed by
    /// `ResumeDataBoxReader.concat_strings` / `slice_string` / `concat_unicodes`
    /// / `slice_unicode` (resume.py) which look up the
    /// `OS_STR_CONCAT` / `OS_STR_SLICE` / `OS_UNI_CONCAT` / `OS_UNI_SLICE`
    /// calldescr + func pointers while rematerializing virtual strings
    /// during bridge-virtual reconstruction.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// pyjitpl.py:2398: tracing-time heap cache.
    /// Tracks field/array values, allocations, escape status, and class/nullity
    /// knowledge during tracing to avoid recording redundant operations.
    pub(crate) heap_cache: HeapCache,
    /// pyjitpl.py:2411 force_finish_trace: when True, trace is segmented
    /// at 80% of limit via _create_segmented_trace_and_blackhole.
    pub(crate) force_finish: bool,
    /// pyjitpl.py:2594 frame.pc: last bytecode pc passed to trace_fn.
    /// Used by force_finish_trace segmenting to record the guard-point pc.
    pub last_traced_pc: usize,
    /// GC-safe constant value snapshot for each initial inputarg at trace
    /// start. Each entry is an inline-const `OpRef` mirroring
    /// history.py/268/314 (`Const*.value` lives on the box); the inline
    /// gcref of a `ConstPtr` entry is forwarded in place by
    /// `MetaInterp::walk_active_trace_refs`. Used by cut_trace_from to
    /// remap escaped original inputargs to their stable Const value.
    pub initial_inputarg_consts: Vec<OpRef>,
    /// Single-pass tracing: the resume-aligned bytecode pc the walk closed
    /// back to, captured at the CloseLoop decision point in the JitCode
    /// dispatch. `None` until the walk populates it. Surfaced through
    /// `MetaInterp::single_pass_outcome` (set before `compile_loop` drains the
    /// ctx) to the merge-point hook.
    pub walk_final_pc: Option<usize>,
    /// Interpreter pc named by the most recent root-frame merge point the walk
    /// passed through — the last position at which the interpreter was, by
    /// construction, re-enterable.
    ///
    /// Distinct from [`Self::walk_final_pc`], which the abort path derives from
    /// the root frame's i0 and which therefore names the position AFTER the
    /// opcode the walk stopped in. The two agree only when the walk stopped at
    /// an opcode boundary. `trace_jitcode_with_args_and_runtime` prefers this
    /// one for a degraded-stub abort, where the opcode provably applied
    /// nothing.
    ///
    /// `None` for a driver that declares no greens: its merge point yields no
    /// concrete pc, so there is nothing to record and the correction that reads
    /// this does not apply.
    pub last_mp_green_pc: Option<usize>,
    /// Set when the abort came out of a panic caught around `run_one_step`
    /// rather than a decision the walk took.  The unwind can leave the frame's
    /// `code_cursor` anywhere inside the panicking instruction, so the frames
    /// name no position a blackhole could resume at — the abort consumer must
    /// not convert them (`blackhole.py` assumes `frame.pc` is an
    /// instruction boundary).  RPython has no counterpart: it has no panic arm
    /// here.
    pub abort_after_panic: bool,
    /// Set when the walk refused a call before making it: a residual call
    /// whose target was still a symbolic path hash, or a recursive portal
    /// call with neither an inline portal frame nor an assembler token.  A
    /// dispatch-arm sub-JitCode may contain an earlier
    /// residual call that already executed concretely, so neither replaying
    /// the source opcode nor resuming after the refused call is sound —
    /// unless the host marked that earlier call via
    /// [`crate::note_residual_committed`], in which case replay applies the
    /// heap effect twice and the publisher keeps the live portal pc.
    /// The abort publisher consumes this flag; [`crate::symbolic_residual_trace_aborts`]
    /// is the embedder-facing signal.
    pub symbolic_residual_abort: bool,
    /// A bridge walk aborted on a walk-local residual that will fail the
    /// same way every time. `handle_fail` must not `must_compile` this
    /// guard again (`compile.py` blackhole `else` arm).
    pub deterministic_bridge_abort: bool,
    /// `pyjitpl.py run_blackhole_interp_to_cancel_tracing` needs
    /// `metainterp.framestack` to still exist when it calls
    /// `blackhole.py convert_and_run_from_pyjitpl(self, ...)`.  RPython
    /// keeps the stack on the MetaInterp for the whole trace; pyre's walk owns
    /// it locally (`trace_jitcode_with_args_and_runtime` allocates a
    /// `StandaloneFrameStack` and drops it on return), so an aborting walk
    /// moves it here — onto the tracing session the MetaInterp owns — for the
    /// jitdriver's Abort arm to convert.  `None` on every non-aborting walk and
    /// once the arm has taken it.
    pub aborted_framestack: Option<crate::pyjitpl::MIFrameStack>,
    /// Single-pass tracing: the walk-final concrete RED values captured from
    /// the closing merge point's red operands (their live register boxes),
    /// in operand order (slot 3 ints, slot 4 refs, slot 5 floats). The
    /// merge-point hook feeds these to
    /// `restore_values` to complete the `S_{k+1}` transfer that storage-only
    /// `recover` cannot reconstruct (loop-carried state held in a red bank but
    /// never written back to the shared heap). Empty unless single-pass.
    pub walk_final_reds: Vec<majit_ir::Value>,
    /// Loop-carried boxes collected from the portal frame at walk end,
    /// the `pyjitpl.py reached_loop_header` `live_arg_boxes` list.
    pub close_jump_boxes: Option<Vec<(OpRef, Type)>>,
    /// Walk-final int+float scalar identity values, in
    /// `collect_scalar_state_field_values` order.
    pub close_scalar_values: Option<Vec<i64>>,
    /// Walk-final ref scalar identity values.
    pub close_ref_scalar_values: Option<Vec<i64>>,
    /// Concrete payload of a root-frame FINISH reached by the tracing walk.
    /// RPython's return Box carries this value intrinsically; state-field
    /// synthetic register OpRefs do not all name recorder entries, so preserve
    /// the value read from the live MIFrame beside the symbolic FINISH arg.
    pub walk_finish_values: Vec<majit_ir::Value>,
    /// pyjitpl.py:1087 parity: quasi-immutable field read needs a
    /// GUARD_NOT_INVALIDATED with full snapshot at the field read's orgpc.
    /// Stores Some(orgpc) when pending.
    pending_guard_not_invalidated_pc: Option<usize>,
    /// pyjitpl.py `MetaInterp.forced_virtualizable` parity. Tracks the
    /// vbox handed to `gen_store_back_in_vable` so the second
    /// `opimpl_hint_force_virtualizable` of the same trace can be skipped.
    /// RPython resets this in `MetaInterp.__init__`; pyre keeps it on
    /// TraceCtx because TraceCtx is freshly created per trace and the
    /// MetaInterp is reused across traces.
    forced_virtualizable: Option<OpRef>,
    /// pyjitpl.py MetaInterp.__init__ creates one args_dict per attempt.
    /// TraceCtx is the native attempt owner; compilation and optimization
    /// share this dictionary and its rooted constants, not address snapshots.
    pub(crate) call_pure_results: crate::optimizeopt::util::ArgsDict,
    /// Cached `warmstate.trace_limit` snapshot for this tracing session.
    /// pyjitpl.py:2789 reads `self.jitdriver_sd.warmstate.trace_limit` each
    /// call; pyre snapshots it at `setup_tracing` time (warmstate owns the
    /// live value). Default mirrors rlib/jit.py:592 (trace_limit = 6000).
    pub(crate) trace_limit: usize,
    /// Pyre-only snapshot side table (opencoder.py stores snapshots inline
    /// in `_snapshot_data` / `_snapshot_array_data` byte streams).
    /// `capture_resumedata` pushes one entry per guard; the returned id
    /// is stored on the guard op's `rd_resume_position`.  Grows
    /// monotonically across `cut_trace` (matches the pre-
    /// behavior — see `cut_trace` for rationale).  Will migrate to the
    /// byte-stream form carried by `TraceRecordBuffer` alongside the
    /// eventual field swap (/ #70).
    pub(crate) snapshots: Vec<crate::recorder::Snapshot>,
    /// pyjitpl.py:2898 `self.resumekey_original_loop_token = ...`.
    /// The source loop token of the bridge trace, populated at
    /// `start_retrace_from_guard` from the failed guard descr's
    /// `rd_loop_token`.  `None` for loop-entry traces (RPython
    /// `isinstance(self.resumekey, compile.ResumeFromInterpDescr)` is
    /// True).  Used by `prepare_trace_segmenting` (pyjitpl.py-
    /// 2834) to set the `FORCE_BRIDGE_SEGMENTING` bit on the loop
    /// token when bridge tracing aborts without an inlinable function.
    pub(crate) resumekey_original_loop_token: Option<std::sync::Arc<JitCellToken>>,
    /// pyjitpl.py _opimpl_getfield_gc_any_pureornot `self.metainterp.cpu` analog.
    ///
    /// RPython's `_opimpl_getfield_gc_any_pureornot` runs
    /// `executor.execute(self.metainterp.cpu, self.metainterp, opnum,
    /// fielddescr, box)` on every cache hit and asserts the loaded
    /// value matches `upd.currfieldbox.getint()/getref_base()/
    /// constbox()` before bumping `HEAPCACHED_OPS`.
    ///
    /// Pyre's `MetaInterp.backend: BackendImpl` owns the cpu; TraceCtx
    /// lives alongside it on the same MetaInterp. The pointer captured
    /// here is to the metainterp-owned backend; it stays valid for the
    /// full duration of the trace because the metainterp does not
    /// move while tracing is active. `None` (the default for unit
    /// tests + standalone-trace entries) disables the sanity check —
    /// mirroring RPython's `translate_support_code=True` mode where
    /// the executor strips the load.
    ///
    /// Wired by `set_cpu` at trace setup; read by `field_sanity_load`
    /// which deref's the pointer to invoke `executor::do_getfield_gc_*`.
    /// The fat pointer is `*const dyn Backend` (16 bytes on 64-bit).
    pub(crate) cpu: Option<*const dyn majit_backend::Backend>,
    /// Called by [`Self::cut_trace`] / [`Self::cut_trace_with_snapshots`]
    /// with the position the recorder was restored to.  `Trace.cut_at`
    /// (`opencoder.py`) discards operations `execute_and_record` already
    /// executed; upstream only cuts where the executed prefix stands
    /// (`MetaInterp.cancel_count` paths re-enter from a fresh frame), while a
    /// tracer that cuts to hand the same region to another executor has to
    /// undo what the discarded operations wrote.  `None` for a tracer with no
    /// such undo log.
    pub(crate) cut_observer: Option<fn(&crate::recorder::TracePosition)>,
    /// `pyjitpl.py MetaInterp.cpu` analog for `execute_and_record` /
    /// `executor.execute` (`model.py AbstractCPU`). Production traces
    /// install this from `MetaInterp.cpu` at the same setup sites that
    /// call [`Self::set_cpu`]. Constructors default to `default_cpu()`.
    pub(crate) metainterp_cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    // `opref_concrete: HashMap<u32, Value>` retired — the concrete
    // value now lives intrinsically on each frontend object's
    // `value: Cell<Option<Value>>` field (`Op` / `InputArg`), matching
    // RPython `history.py` *FrontendOp(pos, value) where the
    // per-position concrete is an object field, not an external side
    // table.  `set_opref_concrete` / `lookup_opref_concrete` now route
    // through `recorder.set_concrete_at` / `recorder.concrete_at`, which
    // resolve `opref.raw()` to the canonical `InputArg` / `Op`.
    /// `pyjitpl.py:3389-3390` `raise SwitchToBlackhole(ABORT_ESCAPE,
    /// raising_exception=True)` — RPython surfaces the abort reason and
    /// the `raising_exception` flag as a real Python exception that
    /// propagates out of `interpret()` to `_compile_and_run_once`
    /// (`pyjitpl.py`), where the catch site invokes
    /// `run_blackhole_interp_to_cancel_tracing(stb)` (`pyjitpl.py`).
    /// That helper does TWO things: (1) `aborted_tracing(stb.reason)`
    /// accounting, (2) `convert_and_run_from_pyjitpl(self,
    /// stb.raising_exception)` — converting the framestack into
    /// blackhole interpreters and running them with the
    /// `raising_exception` flag so the eventual exception is preserved
    /// (`pyjitpl.py` comment).
    ///
    /// TODO: pyre's `TraceAction::Abort` carries no
    /// payload, so the dispatch site (`finalize_standard_virtualizable_may_force`)
    /// stashes the full `SwitchToBlackhole` here and the jitdriver-side
    /// consumer drains it.  Currently only `stb.reason` is consumed —
    /// the consumer mirrors only `pyjitpl.py` `aborted_tracing(reason)`
    /// accounting.  `stb.raising_exception` is preserved on this struct
    /// but the `convert_and_run_from_pyjitpl` invocation
    /// (in `blackhole.rs`, ported from `blackhole.py`) is NOT yet
    /// wired through this path; the helper-side exception raised during
    /// the residual call is therefore silently dropped at the abort
    /// boundary rather than re-raised via blackhole as RPython does.
    /// Full `pyjitpl.py / 2949` cancel-tracing semantics needs
    /// `BlackholeInterpBuilder` + `last_exc_value` plumbed to the
    /// `TraceAction::Abort` consumer and a JitException return surface
    /// on the back-edge runner — followup.
    ///
    /// `None` outside the brief window between the dispatch-site stash
    /// and the jitdriver-side drain.
    /// Framestack half of `pyjitpl.py MetaInterp.replace_box`.
    ///
    /// `_nonstandard_virtualizable` calls `self.metainterp.replace_box`
    /// immediately (`pyjitpl.py` `if isstandard: replace_box`). The
    /// framestack lives on the jitcode machine / walker, so that owner
    /// installs this hook for the duration of a vable op. `None` is the
    /// test path with no live frames.
    replace_frames: Option<(unsafe fn(*mut (), OpRef, OpRef), *mut ())>,

    /// `pyjitpl.py MetaInterp.virtualref_boxes`: pairs of `[virtualbox,
    /// vrefbox]` for every `opimpl_virtual_ref` ↔ `opimpl_virtual_ref_finish`
    /// LIFO scope.  Pyre stores `(OpRef, usize)` so the symbolic SSA value
    /// and the concrete `JitVirtualRef*` pointer both live in one slot:
    /// the OpRef feeds `replace_box` / `vrefs_after_residual_call`
    /// re-tagging; the ptr feeds `vrefinfo.tracing_after_residual_call`
    /// / `is_virtual_ref` runtime probes that decide whether a residual
    /// call forced the ref or whether `VIRTUAL_REF_FINISH` should fire.
    ///
    /// Lives on `TraceCtx` (not `MetaInterp`) because RPython's
    /// `MetaInterp` is per-`_compile_and_run_once` and pyre's
    /// per-trace counterpart is this `TraceCtx`; cross-trace
    /// MetaInterp would otherwise carry stale pairs.
    ///
    /// `replace_box` re-resolves the cached `.1` pointer from the new
    /// OpRef whenever the replacement is a `Const*` (the raw constant
    /// bits are the new `JitVirtualRef*` value).  For non-Const
    /// replacements the cache is preserved on the invariant that
    /// aliased OpRefs share `getref_base()` — the RPython shape, which
    /// reads `box.getref_base()` off the Box at every use.
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,

    /// The decoded inline-callee recipes for the bridge
    /// currently being set up, stashed by `setup_bridge_sym` and drained
    /// once by `trace_bytecode` right before `interpret()`. `None` for
    /// primary traces and single-frame bridges.
    ///
    /// Lives on `TraceCtx` (the per-trace MetaInterp-analog) rather than a
    /// thread-local: a fresh `TraceCtx` is built per bridge and dropped on
    /// every abort path (`abort_trace_live`), so the carrier is reborn
    /// `None` for each bridge exactly as RPython resets `self.framestack =
    /// []` before `rebuild_from_resumedata` (pyjitpl.py). This makes a
    /// stale carrier leaking across bridges structurally impossible.
    pub(crate) bridge_inline_carrier: Option<BridgeInlineCarrier>,
    /// The carrier `trace_bytecode` drained, kept here while the walk uses
    /// it. Its recipes are the deferred `MIFrame`s of
    /// `rebuild_from_resumedata` (resume.py): upstream those frames sit on
    /// `metainterp.framestack`, so the collector reaches their register
    /// boxes. `MetaInterp::walk_active_trace_refs` reads the boxes from here.
    pub(crate) active_bridge_inline_carrier: Option<std::rc::Rc<BridgeInlineCarrier>>,
    /// resume.py consume_boxes parity: per-bank live register indices
    /// of the bridge's root (section 0) guard resume frame, stashed by
    /// `start_bridge_tracing` (which has the dispatch JitCode) so a
    /// JitDriver `setup_bridge_sym` (a static trait method with no
    /// metainterp access) can map each decoded frame value to its
    /// sym slot via `reg_idx - identity_base`.
    pub(crate) bridge_reg_indices: Option<crate::resume::FrameLivenessRegIndices>,
    /// `resume.py` `VirtualCache`, shared by the split readers that
    /// upstream keeps on one `ResumeDataBoxReader`: `setup_bridge_sym`
    /// and `ResumeDataBoxReader.consume_boxes`. Indexed by virtual
    /// number, one slot per `rd_virtuals` entry. A later reader returns
    /// the box the first `getvirtual_ptr` allocated instead of emitting
    /// a second `NEW`.
    bridge_virtual_ops: Vec<Option<OpRef>>,
    /// `resume.py` `rebuild_from_resumedata` storage, parked so
    /// `consume_boxes` can `getvirtual_ptr` (`create_history` already
    /// made tracing live). `None` outside a bridge.
    bridge_resume_data: Option<crate::jit_state::ResumeDataResult>,
    /// Whether the source guard descr for this bridge is a
    /// ResumeGuardExcDescr analog. Set by `start_bridge_tracing` from
    /// `descr_arc.is_guard_exc()` and read by static bridge setup/walkers
    /// that only receive `TraceCtx`.
    pub(crate) bridge_source_is_exception_guard: bool,
    /// The carrier walk copied this failure's grabbed exception onto the
    /// sym. A later residual clears `BH_LAST_EXC_VALUE`, so the root
    /// frame's seed must not treat that empty cell as "no exception" and
    /// wipe the sym.
    pub(crate) bridge_grab_seeded: bool,
    /// `prepare_resume_from_failure` already recorded `RESTORE_EXCEPTION`
    /// and `handle_possible_exception`. The walker must not emit that
    /// sequence again. The walk of the framestack-top jitcode starts at
    /// `bridge_exception_resume_pc` and stamps that guard.
    /// `reached_loop_header` consumes this with
    /// [`Self::consume_bridge_exception_resume`].
    pub(crate) bridge_exception_resume_prepared: bool,
    /// Top `MIFrame.pc` before `handle_possible_exception`. The guard's
    /// resume snapshot is this coordinate; `finishframe_exception` may
    /// move the frame afterwards.
    pub(crate) bridge_exception_source_pc: Option<usize>,
    /// `JitCode` index of the frame that held `bridge_exception_source_pc`.
    pub(crate) bridge_exception_source_jitcode: Option<i32>,
    /// Top `MIFrame.pc` after `handle_possible_exception`.
    /// `finishframe_exception` has already moved it to the handler on
    /// `ChangeFrame`; otherwise it is the resume pc. Only the walk of
    /// `bridge_exception_resume_jitcode` starts there.
    pub(crate) bridge_exception_resume_pc: Option<usize>,
    /// `JitCode` index of the frame that holds `bridge_exception_resume_pc`.
    pub(crate) bridge_exception_resume_jitcode: Option<i32>,
    /// `SAVE_EXCEPTION` op from `_prepare_exception_resumption`. The
    /// walker reads it as the handler's exception box when the guard
    /// was already recorded. `reached_loop_header` consumes it with
    /// [`Self::consume_bridge_exception_resume`]: the op is a pre-cut
    /// producer, and only `live_arg_boxes` survive `cut_trace_from`.
    pub(crate) bridge_saved_exc_op: Option<OpRef>,
    /// Set when a bridge-entry resume replay ran as the applying reader and
    /// met a write it could not apply. The trace then holds recorded writes
    /// whose heap half did not happen, so the entry that asked for that reader
    /// must decline rather than resume against a heap it half-described.
    ///
    /// Only `resume.py`'s box reader can raise it: the recording-only walk has
    /// nothing to apply and leaves it false.
    pub(crate) bridge_replay_incomplete: bool,
    /// Set while this tracing session runs a residual or rebuilt callee that
    /// re-enters the portal. `JitDriver::jit_merge_point_keyed` must not start
    /// a nested trace on the same `TraceCtx` (`MIFrame.do_recursive_call`
    /// stays on this metainterp instead of a second tracer).
    pub trace_continuation_suspended: std::cell::Cell<bool>,
}

/// A decoded-but-not-yet-built description of one inlined
/// callee frame (`resume_data.frames[i]`, `i >= 1`) for a multi-frame
/// bridge. `setup_bridge_sym` decodes the resume stream into this recipe
/// while the resume data / rd_virtuals cache are in scope. On an
/// exception-guard bridge the trace ops that allocate this callee's
/// `PyFrame` are recorded into `rebuilt_frame` before
/// `prepare_resume_from_failure` (`pyjitpl.py` `rebuild_from_resumedata`
/// then `prepare_resume_from_failure`). The concrete `PyFrame` stays
/// deferred to the drain so it is not held unrooted across a collection.
///
/// The bank vectors are indexed by pyre's semantic register index: pyre
/// traces Python bytecode, so these are `locals_cells_stack_w` positions,
/// NOT RPython regalloc colors, and align with LOAD_FAST's `nlocals +
/// stack_idx`. `concrete_r` is parallel to `registers_r` and seeds the
/// assembled frame's `locals_cells_stack_w`.
#[derive(Clone)]
pub struct ReconstructRecipe {
    /// Raw `CodeObject*` identity of the callee (NOT the PyCode wrapper).
    /// The globals-stamped wrapper is recovered on demand from the
    /// `code_ptr -> live-wrapper` registry, so the recipe carries only the
    /// stable code identity rather than a live wrapper courier.
    pub code_ptr: *const (),
    pub jitcode_index: i32,
    /// Guard-carried JitCode offset from the decoded resume frame;
    /// `majit_ir::resumedata::NO_JITCODE_PC` when the frame carried none.
    pub jitcode_pc: i32,
    /// Semantic stack base: `co_nlocals + ncellvars + nfreevars`, matching
    /// `MIFrame.registers_r` / `PyFrame.locals_cells_stack_w`.  This was
    /// historically named `nlocals`; keep the field name for wire stability.
    pub nlocals: usize,
    pub valuestackdepth: usize,
    pub registers_i: Vec<OpRef>,
    pub registers_r: Vec<OpRef>,
    pub registers_f: Vec<OpRef>,
    pub concrete_r: Vec<majit_ir::Value>,
    /// The level's `frame` red, decoded from its resume section like every
    /// other live register (`resume.py consume_boxes`).  The walk resumes the
    /// callee on this box, so the frame the parent trace entered — the one its
    /// `virtual_ref` scope and the callee frames' `f_backref` name — stays the
    /// frame that runs.  `NONE` for a level that has no frame red.
    pub frame: OpRef,
    pub nargs: usize,
    /// Set only for a level that reconstructs NO frame: on the way out it
    /// discards its callee's result and yields this box to its own caller
    /// instead.  `typeobject.py descr_call` is the one such level — the
    /// discard of `__init__`'s result plus `return w_newobject` is its whole
    /// JIT-visible body, so there is no bytecode to walk and no
    /// `locals_cells_stack_w` to rebuild.  Every other field above is unread
    /// when this is `Some`.
    pub return_substitute: Option<OpRef>,
    /// `operation.py len` after `_len`. Not a Python frame: the drain runs
    /// `bh_len_tail` on the callee's return. `capture_resumedata`
    /// (`pyjitpl.py`) keeps that graph on the framestack.
    pub len_tail: bool,
    /// `NewWithVtable` of this frame on an exception-guard bridge, recorded
    /// while the framestack is rebuilt and before
    /// `prepare_resume_from_failure` records that guard. `None` for a level
    /// that builds no frame, and for a bridge whose source guard is not an
    /// exception guard: that bridge records no guard between the rebuild and
    /// the walk, and the walk emits the frame. The walk reuses this box; it
    /// does not emit another one after the guard.
    pub rebuilt_frame: Option<OpRef>,
    /// Execution-context box paired with [`Self::rebuilt_frame`].
    pub rebuilt_ec: Option<OpRef>,
}

/// The decoded inline-callee recipes for one multi-frame
/// bridge, plus the outermost (`frames[0]`) resume pc. `trace_bytecode`
/// builds the caller-visible root frame at `root_pc` and pushes each
/// recipe on top (innermost last), so the framestack matches the inline
/// depth the guard fired at (`rebuild_from_resumedata` resume.py).
pub struct BridgeInlineCarrier {
    /// `resume_data.frames[0].pc` — where the outermost (portal/root) frame
    /// resumes once the reconstructed callees return. The bridge's returned
    /// resume pc (`decode_and_restore_guard_failure`) is the INNERMOST frame's
    /// pc; the root must instead resume at its own `frames[0].pc`, so this is
    /// threaded separately rather than derived from the trace start pc.
    pub root_pc: usize,
    /// The JitCode body that owns `root_pc`.
    pub root_jitcode_index: i32,
    /// `resume_data.frames[1..]`, OUTERMOST-FIRST. The portal (`frames[0]`)
    /// is NOT here — it is the caller-visible root `sym`.
    pub recipes: Vec<ReconstructRecipe>,
}

impl ReconstructRecipe {
    /// Every box this level holds.
    pub fn for_each_box(&self, mut f: impl FnMut(OpRef)) {
        self.registers_r
            .iter()
            .chain(std::iter::once(&self.frame))
            .chain(self.return_substitute.iter())
            .chain(self.rebuilt_frame.iter())
            .chain(self.rebuilt_ec.iter())
            .for_each(|op| f(*op));
    }
}

impl BridgeInlineCarrier {
    /// Every box the recipes hold, for the live-box census of
    /// `MetaInterp::walk_active_trace_refs`.
    pub fn for_each_box(&self, mut f: impl FnMut(OpRef)) {
        for recipe in self.recipes.iter() {
            recipe.for_each_box(&mut f);
        }
    }

    /// history.py `ConstPtr.value` of the recipes' constant boxes: the
    /// carrier is their holder, so it traces their `const_ptr_table` slots.
    pub fn trace_const_ptr_boxes(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        self.for_each_box(|op| op.trace_const_ptr(visitor));
    }
}

/// The virtualizable shadow slot a `vable_set*` standard leg overwrote, and
/// the Box it held before.
///
/// `_opimpl_setfield_vable` / `_opimpl_setarrayitem_vable` (pyjitpl.py,
/// :1236) reach `virtualizable_boxes[index] = valuebox` only AFTER
/// `_nonstandard_virtualizable` / `_get_arrayitem_vable_index` have promoted,
/// and each promote captures its guard's resume data inside
/// `implement_guard_value` — that is, against the shadow as it stood before the
/// write.  Pyre fuses promote and store into one `TraceCtx` call and the
/// dispatcher builds the snapshot afterwards, so the caller puts this slot back
/// for the duration of the capture ([`TraceCtx::swap_virtualizable_entry`]).
/// Without it the guard's resume data carries the very write its resume pc will
/// re-execute: a failing index promote would restore the value into the
/// promoted slot and then write it again at the real index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VableEntryWrite {
    /// Flat index into `virtualizable_boxes`.
    pub index: usize,
    pub prev_box: OpRef,
    pub prev_value: Value,
}

impl VableEntryWrite {
    /// Read the slot `index` currently holds, or `None` when no shadow is
    /// active — the caller then has nothing to restore.
    fn of(ctx: &TraceCtx, index: usize) -> Option<Self> {
        let (prev_box, prev_value) = ctx.virtualizable_entry_at(index)?;
        Some(Self {
            index,
            prev_box,
            prev_value,
        })
    }
}

/// `_nonstandard_virtualizable` up through the `PTR_EQ` (`pyjitpl.py`).
///
/// Step 4 records `PTR_EQ` and then `implement_guard_value` on the
/// `MIFrame` that owns the framestack. `TraceCtx` is the recorder and
/// cannot capture that snapshot itself, so the identity/PTR_EQ portion
/// returns here and the caller runs `implement_guard_value` (dispatch
/// `record_state_guard`, walker `walker_implement_guard_value`) before
/// [`Self::commit_nonstandard_virtualizable`].
pub enum NonstandardVable {
    /// Steps 1, 3, or 5 already decided; no `GUARD_VALUE` is pending.
    Decided(bool),
    /// Step 4 recorded `PTR_EQ`. The caller promotes `eqbox` then commits.
    PendingEq {
        eqbox: OpRef,
        isstandard: i64,
        vable_opref: OpRef,
        standard_box: OpRef,
    },
}

/// Outcome of `vable_setarrayitem_indexed`.
pub enum VableArrayStore {
    /// The promoted index falls outside the standard virtualizable array, so
    /// the slot cannot be virtualized and the caller aborts the trace.
    OutOfVable,
    /// Stored.  `Some` names the shadow slot the standard leg overwrote;
    /// `None` is the nonstandard leg, which records a heap `SetarrayitemGc`.
    Stored(Option<VableEntryWrite>),
}

/// rlib/jit.py:592 default `trace_limit` — mirrored here so standalone
/// TraceCtx construction (unit tests, `setup_tracing` before a warmstate
/// override) matches the RPython baseline.
pub const DEFAULT_TRACE_LIMIT: usize = 6000;

impl TraceCtx {
    /// opencoder.py:472 `self.metainterp_sd` — shared static data the
    /// recorder was constructed with. Read-only handle for callers that
    /// need to reach the per-process descr pools and terminal descrs
    /// (`done_with_this_frame_descr_*`,
    /// `exit_frame_with_exception_descr_ref`) without owning a separate
    /// reference.
    pub fn metainterp_sd(&self) -> &std::sync::Arc<crate::MetaInterpStaticData> {
        &self.metainterp_sd
    }

    /// pyjitpl.py:2398: access the tracing-time heap cache.
    pub fn heap_cache(&self) -> HeapCacheView<'_> {
        HeapCacheView::new(&self.heap_cache, &self.recorder)
    }

    /// Mutable access to the tracing-time heap cache.
    pub fn heap_cache_mut(&mut self) -> HeapCacheViewMut<'_> {
        HeapCacheViewMut::new(&mut self.heap_cache, &mut self.recorder)
    }

    /// pyjitpl.py `newframe` / `popframe` log half for a JitCodeMachine
    /// that cannot borrow `MetaInterp`. Forwards to the MetaInterp log.
    pub fn push_portal_trace_event(
        &self,
        jd_no: usize,
        green_key: Option<crate::pyjitpl::PortalGreenKey>,
        pos: crate::recorder::TracePosition,
    ) {
        if let Some(ref push) = self.portal_trace_push_fn {
            push(jd_no, green_key, pos);
        }
    }

    /// pyjitpl.py `MetaInterp.portal_call_depth` sample.
    pub fn portal_call_depth(&self) -> i32 {
        self.portal_call_depth_fn.as_ref().map(|f| f()).unwrap_or(0)
    }

    /// pyjitpl.py `MetaInterp.newframe` / `popframe` `portal_call_depth` ± 1.
    pub fn adjust_portal_call_depth(&self, delta: i32) {
        if let Some(f) = &self.portal_call_depth_adjust_fn {
            f(delta);
        }
    }

    /// pyjitpl.py `MetaInterp.unique_id_for_greenkey`.
    pub fn unique_id_for_greenkey(
        &self,
        jd_no: usize,
        greenkey: &crate::pyjitpl::PortalGreenKey,
    ) -> u64 {
        self.unique_id_for_greenkey_fn
            .as_ref()
            .map(|f| f(jd_no, greenkey))
            .unwrap_or(0)
    }

    /// Install the observer both trace cuts report to (see
    /// [`Self::cut_observer`]).
    pub fn set_cut_observer(&mut self, observer: fn(&crate::recorder::TracePosition)) {
        self.cut_observer = Some(observer);
    }

    /// Whether a backend is wired, i.e. whether `executor.execute` has
    /// anything to run against.
    pub fn has_cpu(&self) -> bool {
        self.cpu.is_some()
    }

    /// Install the `self.metainterp.cpu` analog for the cache-hit
    /// sanity-check load.
    ///
    /// Captures a raw pointer to the metainterp-owned backend. SAFETY:
    /// the caller guarantees `backend` outlives this `TraceCtx` — true
    /// in production where both live on the same `MetaInterp` instance
    /// that does not move while tracing is active. Tests + standalone
    /// entries call `set_cpu(None)` (or never call this setter) to
    /// leave the sanity check disabled, mirroring RPython's
    /// `translate_support_code=True` mode where the executor strips
    /// the load.
    pub fn set_cpu(&mut self, cpu: Option<&dyn majit_backend::Backend>) {
        // Erase the borrow's lifetime: the caller owns the backend for
        // the lifetime of this TraceCtx (production: MetaInterp pins
        // both; tests: pass None or supply a sufficiently long-lived
        // backend reference). SAFETY: pyre's RPython parity contract —
        // `self.metainterp.cpu` is a stable identity for the duration
        // of any single trace.
        self.cpu = cpu.map(|b| {
            let raw: *const dyn majit_backend::Backend = b;
            // Lifetime erasure via raw pointer round-trip:
            // `*const dyn Trait` is a fat pointer; transmuting the
            // lifetime in the trait-object part is legal because the
            // pointee identity doesn't change.
            unsafe {
                std::mem::transmute::<
                    *const dyn majit_backend::Backend,
                    *const dyn majit_backend::Backend,
                >(raw)
            }
        });
    }

    /// Install the `model.py AbstractCPU` used by `execute_and_record`.
    /// Production: `MetaInterp.cpu` at the same setup sites as
    /// [`Self::set_cpu`]. Tests keep the constructor default.
    pub fn set_metainterp_cpu(&mut self, cpu: std::sync::Arc<dyn crate::cpu::Cpu>) {
        self.metainterp_cpu = cpu;
    }

    /// Borrow the exact MetaInterp CPU installed by [`Self::set_cpu`].
    pub fn blackhole_cpu(&self) -> Option<&dyn majit_backend::Backend> {
        self.cpu.map(|cpu| {
            // SAFETY: identical owner contract to `field_sanity_load`.
            unsafe { &*cpu }
        })
    }

    /// `executor.execute` for `GETFIELD_GC_{I,R,F}`.
    ///
    /// `do_getfield_gc_i` / `do_getfield_gc_r` / `do_getfield_gc_f` are
    /// selected by the opnum alone. `kind` is that opnum. Returns
    /// `Some(value)` when `self.cpu` is wired and the descr resolves to a
    /// `BhDescr::Field`; `None` when the cpu is unwired, the descr is not
    /// a field, or `kind` is `Type::Void`. The caller compares the loaded
    /// value with the heapcache box (`_do_getfield_gc_any`).
    pub fn field_sanity_load(
        &self,
        struct_ptr: i64,
        descr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let cpu_ptr = self.cpu?;
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let bh_descr = descr_to_bh_field_descr(descr)?;
        match kind {
            Type::Int => Some(Value::Int(crate::executor::do_getfield_gc_i(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Ref => Some(Value::Ref(crate::executor::do_getfield_gc_r(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Float => Some(Value::Float(crate::executor::do_getfield_gc_f(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Void => None,
        }
    }

    /// Address a box names when it carries a live GC pointer, else `None`.
    fn live_ptr_of(&self, opref: OpRef) -> Option<i64> {
        match self.concrete_of_opref(opref) {
            Some(Value::Ref(r)) => live_gc_ptr(r),
            _ => None,
        }
    }

    /// `executor.execute` for a GETFIELD: the explicit struct pointer when
    /// one was passed, otherwise the live pointer on `base`. Declines when
    /// neither is a real non-null concrete, matching
    /// `pyjitpl.py MetaInterp.execute_and_record` which only has a value
    /// after `cpu.bh_getfield_gc_*` runs on a real object.
    fn field_live_value(
        &self,
        struct_ptr: i64,
        base: OpRef,
        descr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let ptr = if struct_ptr != 0 {
            struct_ptr
        } else {
            self.live_ptr_of(base)?
        };
        self.field_sanity_load(ptr, descr, kind)
    }

    /// `executor.execute` for a GETARRAYITEM: the live pointer on the
    /// array box, then `array_sanity_load`. Declines when the box has no
    /// real non-null concrete.
    fn array_live_value(
        &self,
        array_opref: OpRef,
        item_index: i64,
        adescr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let ptr = self.live_ptr_of(array_opref)?;
        self.array_sanity_load(ptr, item_index, adescr, kind)
    }

    /// `executor.py do_setfield_gc(cpu, _, structbox, itembox,
    /// fielddescr)` analog — the store half of [`Self::field_sanity_load`].
    /// `_opimpl_setfield_gc_any` reaches it through `execute_and_record`
    /// (`pyjitpl.py`), so upstream really performs the field store while
    /// recording the `SETFIELD_GC`; a tracer that only records leaves the
    /// concrete object disagreeing with the value the heapcache now carries.
    ///
    /// Returns `false` when `self.cpu` is unwired, the descr does not
    /// resolve to a `BhDescr::Field`, or the value's bank disagrees with the
    /// field's type. `do_setfield_gc` is one opnum and branches on
    /// `is_pointer_field` / `is_float_field`; a load's opnum is `kind`
    /// (`do_getfield_gc_*`) and does not consult the descr bank.
    pub fn field_store(&self, struct_ptr: i64, descr: &DescrRef, value: Value) -> bool {
        let Some(cpu_ptr) = self.cpu else {
            return false;
        };
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let Some(bh_descr) = descr_to_bh_field_descr(descr) else {
            return false;
        };
        let Some(field_type) = descr.as_field_descr().map(|f| f.field_type()) else {
            return false;
        };
        crate::executor::do_setfield_gc(cpu, (), struct_ptr, value, &bh_descr, field_type)
    }

    /// `execute_setarrayitem_gc`'s store (`pyjitpl.py`): the array sibling
    /// of [`Self::field_store`], through `executor.py` `do_setarrayitem_gc`.
    ///
    /// Returns `false` when `self.cpu` is unwired, the descr is not an
    /// array descr, or the value's bank disagrees with the item type.
    pub fn array_store(&self, array_ptr: i64, index: i64, descr: &DescrRef, value: Value) -> bool {
        let Some(cpu_ptr) = self.cpu else {
            return false;
        };
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let Some(bh_descr) = descr_to_bh_array_descr(descr) else {
            return false;
        };
        let Some(item_type) = descr.as_array_descr().map(|a| a.item_type()) else {
            return false;
        };
        crate::executor::do_setarrayitem_gc(cpu, (), array_ptr, index, value, &bh_descr, item_type)
    }

    /// `blackhole.py bhimpl_arraylen_gc(cpu, array, arraydescr)`
    /// analog — read the GC array's length through
    /// `cpu.bh_arraylen_gc(array_ptr, &arraydescr)`.  RPython has no
    /// explicit `do_arraylen_gc` in `executor.py`; the dispatch path
    /// goes through the blackhole fallback wrapper which the bhimpl
    /// implements directly.  Returns `Some(Value::Int(len))` when
    /// `self.cpu` is wired and the descr resolves to a `BhDescr::
    /// Array` carrying a length word; `None` otherwise.  Used by
    /// `opimpl_arraylen_gc` to stamp the recorded `ArraylenGc` OpRef
    /// with its runtime concrete (RPython `BoxInt(length)` carrier).
    pub fn arraylen_sanity_load(&self, array_ptr: i64, descr: &DescrRef) -> Option<Value> {
        let cpu_ptr = self.cpu?;
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        // `bh_arraylen_gc` reads the length word at `lendescr.offset`, so a
        // nolength array descriptor has nothing for it to read and every
        // backend aborts on one. That is the precondition of this helper, not
        // of its callers: answer "no concrete" instead, which is already what
        // a caller does with an unwired cpu.
        if !descr
            .as_array_descr()
            .is_some_and(|a| a.len_descr().is_some())
        {
            return None;
        }
        let bh_descr = descr_to_bh_array_descr(descr)?;
        // A raw `rffi.CArray` descr (`raw_carray_descrof`) resolves to an
        // `Array` with no `lendescr`, because such an array carries no length
        // word — `bh_arraylen_gc` has nothing to read and asserts.  Decline it
        // here: `None` is the state the callers are written for, leaving the
        // recorded op unstamped and the bounds check unproven, rather than
        // aborting the process from a sanity read.
        bh_descr.array_len_offset()?;
        Some(Value::Int(crate::executor::do_arraylen_gc(
            cpu,
            (),
            array_ptr,
            &bh_descr,
        )))
    }

    /// Whether the allocation descriptor can produce a collector-readable
    /// header through the same `BhDescr` conversion used by `bh_new`.
    pub fn new_allocation_tid_is_sound(&self, descr: &DescrRef) -> bool {
        let Some(bh_descr) = descr_to_bh_size_descr(descr) else {
            return false;
        };
        if bh_descr.is_headerless() {
            return true;
        }
        let Some(type_id) = bh_descr.resolved_gc_tid_checked() else {
            return false;
        };
        // A typed allocation must carry an id present in the collector's type
        // table. Otherwise the recorded New and its concrete execution both
        // publish an object header that the collector can only trace as garbage.
        type_id == 0
            || !majit_gc::gc_allocator_installed()
            || majit_gc::is_registered_type_id(type_id)
    }

    /// `pyjitpl.py execute_new[_with_vtable]` concrete execution.
    /// RPython executes the allocation before recording the matching trace op,
    /// so later residual calls and field operations observe a real pointer
    /// while the optimizer remains free to virtualize the recorded allocation.
    ///
    /// Rooting contract: the result is returned unrooted.  The caller must
    /// keep it reachable until it is stamped onto the op it records for this
    /// allocation (`set_opref_concrete` / `execute_and_record`).  The result
    /// box is then written into `MIFrame.registers_r`; that live box is what
    /// makes the object a root — `MetaInterp::walk_active_trace_refs`
    /// forwards `RefFrontendOp._resref` of boxes still reachable from
    /// registers / heapcache, matching the `history.py` `*FrontendOp(pos,
    /// value)` slot upstream reaches through the object graph.  `record_op*`
    /// appends to `opencoder.py Trace._ops` and can minor-collect
    /// (`stress_trace_pool_alloc` / `alloc_fast_nursery_collecting`), so a
    /// bare Rust `Value::Ref` across that append is not enough —
    /// `execute_and_record` / `_record_helper` pin it the way the translated
    /// GCREF local would.
    ///
    /// A side list of executed allocations is NOT the way to widen that
    /// window: it duplicates a root the op graph already owns, and it hands
    /// the collector shapes the op graph never exposes it to.
    pub fn execute_new_allocation(&self, descr: &DescrRef, with_vtable: bool) -> Option<Value> {
        let cpu = unsafe { &*self.cpu? };
        let bh_descr = descr_to_bh_size_descr(descr)?;
        let ptr = if with_vtable {
            cpu.bh_new_with_vtable(&bh_descr)
        } else {
            cpu.bh_new(&bh_descr)
        };
        if ptr == 0 {
            return None;
        }
        Some(Value::Ref(majit_ir::GcRef(ptr as usize)))
    }

    /// `executor.py` `do_newstr` concrete execution for `pyjitpl.py
    /// opimpl_newstr`: `cpu.bh_newstr(length)`.  Same rooting contract as
    /// [`Self::execute_new_allocation`]: the caller stamps the result onto
    /// the recorded `NEWSTR` before any GC allocation.
    pub fn execute_newstr(&self, length: i64) -> Option<Value> {
        let cpu = unsafe { &*self.cpu? };
        if length < 0 {
            return None;
        }
        let ptr = cpu.bh_newstr(length);
        if ptr == 0 {
            return None;
        }
        Some(Value::Ref(majit_ir::GcRef(ptr as usize)))
    }

    /// `executor.py` `do_strsetitem` concrete execution for `pyjitpl.py
    /// opimpl_strsetitem`: `cpu.bh_strsetitem(string, index, newchar)`.
    pub fn execute_strsetitem(&self, string: i64, index: i64, newchar: i64) -> bool {
        let Some(cpu) = self.cpu.map(|cpu| unsafe { &*cpu }) else {
            return false;
        };
        cpu.bh_strsetitem(string, index, newchar);
        true
    }

    /// `executor.py` `do_copystrcontent` concrete execution for `pyjitpl.py
    /// opimpl_copystrcontent`: `cpu.bh_copystrcontent(src, dst, srcstart,
    /// dststart, length)`.
    pub fn execute_copystrcontent(
        &self,
        src: i64,
        dst: i64,
        srcstart: i64,
        dststart: i64,
        length: i64,
    ) -> bool {
        let Some(cpu) = self.cpu.map(|cpu| unsafe { &*cpu }) else {
            return false;
        };
        cpu.bh_copystrcontent(src, dst, srcstart, dststart, length);
        true
    }

    /// `executor.py:200 do_getfield_raw_{i,r,f}` analog — read a raw
    /// field at `struct_ptr + descr.offset` via `cpu.bh_getfield_raw_*`.
    /// Distinct from [`Self::field_sanity_load`] which dispatches the GC
    /// variant (`executor.py:188 do_getfield_gc_*`).  Used when the
    /// recorded opcode is `GetfieldRaw{I,R,F}` rather than
    /// `GetfieldGc{I,R,F}`.
    pub fn raw_field_sanity_load(
        &self,
        struct_ptr: i64,
        descr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let cpu_ptr = self.cpu?;
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let bh_descr = descr_to_bh_field_descr(descr)?;
        match kind {
            Type::Int => Some(Value::Int(crate::executor::do_getfield_raw_i(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Ref => Some(Value::Ref(crate::executor::do_getfield_raw_r(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Float => Some(Value::Float(crate::executor::do_getfield_raw_f(
                cpu,
                (),
                struct_ptr,
                &bh_descr,
            ))),
            Type::Void => None,
        }
    }

    /// `executor.py:132 do_getarrayitem_raw_{i,f}` analog — read a raw
    /// array element via `cpu.bh_getarrayitem_raw_*`.  Distinct from
    /// [`Self::array_sanity_load`] which dispatches the GC variant
    /// (`executor.py:117 do_getarrayitem_gc_*`).  Raw arrays carry the
    /// array pointer as an `int` (`arraybox.getint()` upstream), not a
    /// `getref_base()` projection — callers must pass the raw pointer
    /// as `i64` directly without the `Value::Ref` carrier indirection.
    pub fn raw_array_sanity_load(
        &self,
        array_ptr: i64,
        index: i64,
        descr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let cpu_ptr = self.cpu?;
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let bh_descr = descr_to_bh_array_descr(descr)?;
        match kind {
            Type::Int => Some(Value::Int(crate::executor::do_getarrayitem_raw_i(
                cpu,
                (),
                array_ptr,
                index,
                &bh_descr,
            ))),
            Type::Float => Some(Value::Float(crate::executor::do_getarrayitem_raw_f(
                cpu,
                (),
                array_ptr,
                index,
                &bh_descr,
            ))),
            Type::Ref | Type::Void => None,
        }
    }

    /// Array-side analogue of [`Self::field_sanity_load`].  `executor.execute`
    /// dispatches GETARRAYITEM_GC_{I,R,F} through `do_getarrayitem_gc_*`
    /// (executor.py:206-212); pyre's `kind` selects between the three
    /// variants.  Returns `Some(value)` when `self.cpu` is wired and the
    /// descr resolves to a `BhDescr::Array`; `None` otherwise.
    pub fn array_sanity_load(
        &self,
        array_ptr: i64,
        index: i64,
        descr: &DescrRef,
        kind: Type,
    ) -> Option<Value> {
        let cpu_ptr = self.cpu?;
        // SAFETY: cpu pointer was installed via `set_cpu` against a
        // backend that outlives this TraceCtx.
        let cpu = unsafe { &*cpu_ptr };
        let bh_descr = descr_to_bh_array_descr(descr)?;
        // `do_getarrayitem_gc_i` / `_r` / `_f` are selected by the opnum
        // (`kind`) alone, the same way `do_getfield_gc_*` is.
        match kind {
            Type::Int => Some(Value::Int(crate::executor::do_getarrayitem_gc_i(
                cpu,
                (),
                array_ptr,
                index,
                &bh_descr,
            ))),
            Type::Ref => Some(Value::Ref(crate::executor::do_getarrayitem_gc_r(
                cpu,
                (),
                array_ptr,
                index,
                &bh_descr,
            ))),
            Type::Float => Some(Value::Float(crate::executor::do_getarrayitem_gc_f(
                cpu,
                (),
                array_ptr,
                index,
                &bh_descr,
            ))),
            Type::Void => None,
        }
    }

    /// heapcache.py `getarrayitem(box, indexbox, descr)` parity.
    /// Extracts the index ConstInt's `getint()` value (returns `None`
    /// on non-ConstInt operands, matching the upstream early-out at
    /// `heapcache.py`) and routes the lookup through the indexcache
    /// (`heap_array_cache[descr][index_value]`).  Inside the indexcache,
    /// `array` is canonicalised by `_unique_const_heuristic` against
    /// the per-CacheEntry `last_const_box` (heapcache.py) so two
    /// distinct ConstPtr OpRefs for the same gcref share the same
    /// cache slot.
    pub fn heapcache_getarrayitem(
        &mut self,
        array: OpRef,
        index: OpRef,
        descr: u32,
    ) -> Option<OpRef> {
        let index_value = match index.inline_const_to_value()? {
            Value::Int(n) => n,
            _ => return None,
        };
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .getarrayitem_cache(array, index_value, descr, oracle)
    }

    /// heapcache.py `setarrayitem` parity.  `None` index_value
    /// (non-ConstInt operand) clears the entire `descr` submap;
    /// otherwise the write goes through the indexcache with `array`
    /// canonicalised by `_unique_const_heuristic`.
    pub fn heapcache_setarrayitem(&mut self, array: OpRef, index: OpRef, descr: u32, value: OpRef) {
        let index_value = match index.inline_const_to_value() {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .setarrayitem_cache(array, index_value, descr, value, oracle)
    }

    /// heapcache.py `getarrayitem_now_known` parity.
    pub fn heapcache_getarrayitem_now_known(
        &mut self,
        array: OpRef,
        index: OpRef,
        descr: u32,
        value: OpRef,
    ) {
        let index_value = match index.inline_const_to_value() {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .getarrayitem_now_known(array, index_value, descr, value, oracle)
    }

    /// heapcache.py `getfield` parity.  Routes `obj` through
    /// `_unique_const_heuristic` so two distinct ConstPtr OpRefs for
    /// the same gcref share the same `(obj, field_index)` cache slot.
    ///
    /// Returns the cached `OpRef` — RPython's `upd.currfieldbox` is a
    /// Box object carrying both identity and value; pyre returns the
    /// Box identity as an `OpRef` and sanity-check callers retrieve
    /// the intrinsic value via `box_value(cached)` (which composes
    /// the const pool, standard-virtualizable shadow, and the frontend
    /// object's `value: Cell<Option<Value>>` field — PyPy `history.py:680
    /// AbstractValue.getXXX()` / `history.py *FrontendOp(pos,
    /// value)` parity).
    pub fn heapcache_getfield_cached(&mut self, obj: OpRef, field_index: u32) -> Option<OpRef> {
        // PyPy keys by descriptor identity. An unnumbered Rust descriptor
        // has no such key: u32::MAX is shared by unrelated fallback fields.
        if field_index == u32::MAX {
            return None;
        }
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .getfield_cached(obj, field_index, oracle)
    }

    /// heapcache.py `setfield` parity.  Same canonicalisation
    /// as `heapcache_getfield_cached` plus alias-clearing semantics
    /// when `obj` is not known-unescaped.
    ///
    /// `value` is the cached Box identity (OpRef); its intrinsic
    /// runtime value travels with the frontend value slot so subsequent
    /// cache-hit sanity checks read it via `box_value(value)` —
    /// covering the const pool, standard-virtualizable shadow, and
    /// `Box::value: Cell<Option<Value>>` field in one call.
    pub fn heapcache_setfield_cached(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        if field_index == u32::MAX {
            return;
        }
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .setfield_cached(obj, field_index, value, oracle);
        self.recorder.hold_live_const_ptr(obj);
        self.recorder.hold_live_const_ptr(value);
    }

    /// heapcache.py `getfield_now_known` parity (no aliasing).
    /// `value` is the loaded Box identity (OpRef); the frontend value slot
    /// carries the intrinsic `executor.execute(...)`-produced value
    /// the cache-hit sanity check resolves later via
    /// `lookup_opref_concrete`.
    pub fn heapcache_getfield_now_known(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        if field_index == u32::MAX {
            return;
        }
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        self.heap_cache_mut()
            .getfield_now_known(obj, field_index, value, oracle);
        self.recorder.hold_live_const_ptr(obj);
        self.recorder.hold_live_const_ptr(value);
    }

    /// heapcache.py `invalidate_caches_varargs` parity.
    /// Routes through `clear_caches_varargs` → `_clear_caches_arraycopy` /
    /// `_clear_caches_arraymove` → `_clear_caches_arrayop_with_consts`
    /// where ConstPtr source/dest boxes are canonicalised by
    /// `_unique_const_heuristic` (heapcache.py) via the
    /// `SameConstantOracle` (`history::ConstOprefOracle`, value-compares
    /// inline Const OpRefs).  ConstPtr values are carried inline on the
    /// OpRef (history.py:314), and the active-trace GC walker
    /// (`walk_active_trace_refs`) forwards those inline GCREFs across minor
    /// collections, so reading one here yields the current address with no
    /// separate constant-pool re-read.
    ///
    /// The `const_value` closure resolves `srcstart` / `dststart` /
    /// `length` boxes to their `ConstInt.getint()` values
    /// (heapcache.py `isinstance(_, ConstInt) and ...getint()`).
    /// Without it the per-index copy branch at heapcache.py
    /// is unreachable and arraycopy/arraymove fall back to whole-descr
    /// clearing.
    pub fn heapcache_invalidate_caches_varargs(
        &mut self,
        opnum: majit_ir::OpCode,
        effectinfo: Option<&majit_ir::EffectInfo>,
        argboxes: &[OpRef],
    ) {
        if probe_subscr_enabled() {
            let ei_summary = effectinfo.map(|ei| {
                format!(
                    "extraeffect={:?} forces_vorv={} can_raise={} plain_call={} oopspec={:?}",
                    ei.extraeffect,
                    ei.check_forces_virtual_or_virtualizable(),
                    ei.check_can_raise(false),
                    opnum.is_plain_call(),
                    ei.oopspecindex,
                )
            });
            eprintln!(
                "[MAJIT_PROBE_SUBSCR] invalidate_caches_varargs opnum={:?} argboxes.len={} ei={:?}",
                opnum,
                argboxes.len(),
                ei_summary
            );
        }
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        let const_value = |opref: OpRef| match opref.inline_const_to_value() {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        self.heap_cache_mut().invalidate_caches_varargs(
            opnum,
            effectinfo,
            argboxes,
            oracle,
            const_value,
        )
    }

    /// `heapcache.py invalidate_caches(opnum, descr, *argboxes)`, the call
    /// `pyjitpl.py _record_helper` makes before recording a fixed-arity op.
    pub fn heapcache_invalidate_caches(&mut self, opnum: majit_ir::OpCode, argboxes: &[OpRef]) {
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        let const_value = |opref: OpRef| match opref.inline_const_to_value() {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        self.heap_cache_mut()
            .invalidate_caches(opnum, None, argboxes, oracle, const_value)
    }

    /// pyjitpl.py:1087 parity: check if a quasi-immut guard is pending.
    pub fn pending_guard_not_invalidated_pc(&self) -> Option<usize> {
        self.pending_guard_not_invalidated_pc
    }

    /// Set pending quasi-immut guard with the field read's orgpc.
    pub fn set_pending_guard_not_invalidated(&mut self, pc: Option<usize>) {
        self.pending_guard_not_invalidated_pc = pc;
    }

    /// Install the framestack half of `MetaInterp.replace_box` for one
    /// vable op. `walk` receives `(framestack, old, new)`.
    ///
    /// # Safety
    /// `data` must stay a valid `walk` receiver until
    /// [`Self::clear_replace_frames`] or the next `set_replace_frames`.
    pub unsafe fn set_replace_frames(
        &mut self,
        walk: Option<unsafe fn(*mut (), OpRef, OpRef)>,
        data: *mut (),
    ) {
        self.replace_frames = walk.map(|walk| (walk, data));
    }

    pub fn clear_replace_frames(&mut self) {
        self.replace_frames = None;
    }
}

/// Clears the replace-frames hook on drop, including unwind.
///
/// The hook is installed for the length of one walk, and the walk keeps using
/// the same `TraceCtx` while it is installed, so the guard cannot hold the
/// borrow it has to clear through. It holds a raw pointer instead, which is
/// why constructing it carries a contract rather than a lifetime.
pub struct ClearReplaceFrames(*mut TraceCtx);

impl ClearReplaceFrames {
    /// # Safety
    /// The guard must be dropped before `ctx` is: bind it to a local of a
    /// scope that `ctx` outlives (`let _clear = ...` in a body that borrows
    /// `ctx`), and never move it out of that scope. Dropping it afterwards
    /// writes the hook slot through a dangling pointer.
    pub unsafe fn new(ctx: &mut TraceCtx) -> Self {
        Self(ctx)
    }
}

impl Drop for ClearReplaceFrames {
    fn drop(&mut self) {
        // SAFETY: `new` stores a pointer to a live `TraceCtx`; drop only
        // writes the hook slot to `None`.
        unsafe { (*self.0).clear_replace_frames() };
    }
}

impl TraceCtx {
    /// `fielddescr.get_vinfo()`. Codewriter emits the vinfo's own
    /// FieldDescr, which already holds the Weak backref.
    fn vinfo_from_fielddescr(
        &self,
        fielddescr: &DescrRef,
    ) -> Option<std::sync::Arc<dyn majit_ir::descr::VinfoMarker>> {
        if let Some(v) = fielddescr.as_field_descr().and_then(|fd| fd.get_vinfo()) {
            return Some(v);
        }
        let vi = self.virtualizable_info.as_ref()?;
        if let Some(idx) = vi.static_field_by_descr(fielddescr) {
            return vi
                .static_field_descr(idx)
                .as_field_descr()
                .and_then(|fd| fd.get_vinfo());
        }
        if let Some(idx) = vi.array_field_by_descr(fielddescr) {
            return vi
                .array_pointer_field_descr(idx)
                .as_field_descr()
                .and_then(|fd| fd.get_vinfo());
        }
        None
    }

    /// `pyjitpl.py _nonstandard_virtualizable`:
    /// `self.metainterp.replace_box(box, standard_box)`.
    /// Framestack first, then vref / vable / heapcache.
    fn replace_standard_vable(&mut self, oldbox: OpRef, newbox: OpRef) {
        if let Some((walk, data)) = self.replace_frames {
            unsafe {
                walk(data, oldbox, newbox);
            }
        }
        self.replace_box(oldbox, newbox);
    }

    /// pyjitpl.py:1776-1780: jit.isvirtual(obj) — check if an object
    /// is likely virtual (allocated during this trace and not escaped).
    pub fn is_likely_virtual(&self, obj: OpRef) -> bool {
        self.heap_cache().is_likely_virtual(obj)
    }

    /// pyjitpl.py:1805-1806: record VIRTUAL_REF(box, cindex).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    /// The optimizer can later eliminate the vref if the object stays virtual.
    pub fn virtual_ref(&mut self, obj: OpRef, cindex: OpRef) -> OpRef {
        let result = Self::do_record_op(&mut self.recorder, OpCode::VirtualRefR, &[obj, cindex]);
        // pyjitpl.py:1807: heapcache.new(resbox)
        self.heap_cache_mut().new_object(result);
        result
    }

    /// Live `[virtualbox, vrefbox]` entry count — twice the number of open
    /// `virtual_ref` scopes.  `pyjitpl.py:2995` asserts this is zero when a
    /// loop header is reached ("missing virtual_ref_finish()?").
    pub fn virtualref_boxes_len(&self) -> usize {
        self.virtualref_boxes.len()
    }

    /// Snapshot the open virtual-ref scopes before a non-committal sub-walk.
    /// If that walk is cut from the recorder, its stack mutations must be
    /// rolled back with the operations.
    pub fn snapshot_virtualref_boxes(&self) -> Vec<(OpRef, usize)> {
        self.virtualref_boxes.clone()
    }

    /// The innermost still-open scope's `virtualbox` — `virtualref_boxes[-2]`,
    /// the operand `opimpl_virtual_ref_finish` pops next.  A bridge resumes
    /// into scopes its parent guard opened, so the frame box that closes one is
    /// the one the parent encoded, not a box this trace built.
    pub fn innermost_virtualref_virtual(&self) -> Option<(OpRef, usize)> {
        let len = self.virtualref_boxes.len();
        (len >= 2).then(|| self.virtualref_boxes[len - 2])
    }

    /// Drop the innermost `[virtualbox, vrefbox]` when it names `frame_ptr`.
    ///
    /// `blackhole_if_trace_too_long` raises `SwitchToBlackhole` out of
    /// `_interpret`, so `virtual_ref_finish` does not run and must not record.
    /// The pair is tracing-only state of the trace being discarded. A pair
    /// that names a different frame stays, so a still-live outer scope is not
    /// eaten.
    pub fn discard_innermost_virtualref_if_frame(&mut self, frame_ptr: usize) -> bool {
        let len = self.virtualref_boxes.len();
        if frame_ptr == 0 || len < 2 {
            return false;
        }
        let virtual_entry = self.virtualref_boxes[len - 2];
        let live = self.virtualref_entry_ptr(virtual_entry);
        if live != frame_ptr && virtual_entry.1 != frame_ptr {
            return false;
        }
        self.virtualref_boxes.pop();
        self.virtualref_boxes.pop();
        true
    }

    /// The scope enclosing the innermost one — `virtualref_boxes[-4]`, the
    /// caller frame's `virtualbox` when that caller is itself an open scope —
    /// with its current concrete address.
    pub fn enclosing_virtualref_virtual(&self) -> Option<(OpRef, usize)> {
        let len = self.virtualref_boxes.len();
        (len >= 4).then(|| {
            let entry = self.virtualref_boxes[len - 4];
            (entry.0, self.virtualref_entry_ptr(entry))
        })
    }

    /// The innermost still-open scope's `vrefbox` —
    /// `virtualref_boxes[-1]`.
    pub fn innermost_virtualref_vref(&self) -> Option<(OpRef, usize)> {
        self.virtualref_boxes.last().copied()
    }

    /// The current concrete address of one `virtualref_boxes` entry.
    ///
    /// The `usize` beside each box is the address the object had when the pair
    /// was pushed, and the object it names is movable: a minor collection
    /// relocates it and forwards the stamp, leaving the pushed copy naming the
    /// old address.  `opimpl_virtual_ref_finish` documents the same hazard on
    /// the same list.  Read the address back through `concrete_of_opref` —
    /// pyre's `getref_base()` — so a pair that has moved still matches, and
    /// keep the pushed copy only for an entry carrying no stamp at all.
    pub fn virtualref_entry_ptr(&self, entry: (OpRef, usize)) -> usize {
        match self.concrete_of_opref(entry.0) {
            Some(Value::Ref(r)) => r.as_usize(),
            _ => entry.1,
        }
    }

    /// Resolve a live tracing-time vref back to its `[virtualbox, vrefbox]`
    /// pair.  This is the paired walk `vrefs_after_residual_call` makes over
    /// `MetaInterp.virtualref_boxes` (`pyjitpl.py`): callers that execute
    /// `jit_force_virtual(vref)` need the paired virtual box that
    /// `stop_tracking_virtualref` publishes through `VIRTUAL_REF_FINISH`.
    ///
    /// Search from the innermost pair because frame-chain vrefs are nested in
    /// the same order as `virtualref_boxes`.  A stopped pair has had its vref
    /// entry replaced by `CONST_NULL`, exactly as upstream, so it cannot match.
    pub fn live_virtualref_pair_for_ptr(&self, vref_ptr: usize) -> Option<(OpRef, OpRef)> {
        if vref_ptr == 0 {
            return None;
        }
        self.virtualref_boxes
            .as_chunks::<2>()
            .0
            .iter()
            .rev()
            .find(|pair| self.virtualref_entry_ptr(pair[1]) == vref_ptr)
            .map(|pair| (pair[0].0, pair[1].0))
    }

    /// Find the virtual box for a concrete object named by either a live or an
    /// already-stopped vref pair. `stop_tracking_virtualref` replaces only
    /// `virtualref_boxes[i + 1]` with `CONST_NULL`; the adjacent virtual box
    /// remains in the upstream list until `virtual_ref_finish` pops the scope.
    pub fn virtualref_virtual_for_object_ptr(&self, object_ptr: usize) -> Option<OpRef> {
        if object_ptr == 0 {
            return None;
        }
        self.virtualref_boxes
            .as_chunks::<2>()
            .0
            .iter()
            .rev()
            .find(|pair| self.virtualref_entry_ptr(pair[0]) == object_ptr)
            .map(|pair| pair[0].0)
    }

    /// Concrete vref of a live `[virtualbox, vrefbox]` pair for `object_ptr`.
    ///
    /// `stop_tracking_virtualref` replaces the vref entry with `CONST_NULL`,
    /// so a stopped pair does not match.
    pub fn virtualref_ptr_for_object_ptr(&self, object_ptr: usize) -> Option<usize> {
        if object_ptr == 0 {
            return None;
        }
        self.virtualref_boxes
            .as_chunks::<2>()
            .0
            .iter()
            .rev()
            .find(|pair| self.virtualref_entry_ptr(pair[0]) == object_ptr)
            .map(|pair| self.virtualref_entry_ptr(pair[1]))
            .filter(|&vref_ptr| vref_ptr != 0)
    }

    /// `virtualref.py force_virtual` TOKEN_TRACING_RESCALL arm for an inlined
    /// object. `vrefs_after_residual_call` then `stop_tracking_virtualref`.
    ///
    /// The portal virtualizable is `virtualizable_boxes[-1]`; forcing a vref
    /// of an inlined frame is not `vable_after_residual_call`.
    pub fn force_virtualref_of_object_during_tracing(&self, object_ptr: usize) {
        let Some(vref_ptr) = self.virtualref_ptr_for_object_ptr(object_ptr) else {
            return;
        };
        let _ = unsafe {
            self.metainterp_sd
                .virtualref_info
                .force_virtual(vref_ptr as *mut u8, |_| {})
        };
    }

    /// `pyjitpl.py rebuild_state_after_failure`'s
    /// `self.virtualref_boxes = virtualref_boxes`.  A bridge resumes into its
    /// parent's still-open `virtual_ref` scopes, so the pairs the parent guard
    /// encoded are re-tracked before the bridge trace records anything —
    /// otherwise its own `virtual_ref_finish` would pop an empty stack.
    pub fn restore_virtualref_boxes(&mut self, boxes: Vec<(OpRef, usize)>) {
        self.virtualref_boxes = boxes;
    }

    /// `pyjitpl.py opimpl_virtual_ref` — `ExecutionContext.enter`'s
    /// `jit.virtual_ref(frame)` (`executioncontext.py`) as the tracer sees
    /// it.  Creates the concrete vref, records `VIRTUAL_REF(box, cindex)`, and
    /// pushes the `[virtualbox, vrefbox]` pair.
    ///
    /// Lives here rather than on `MetaInterp` because `virtualref_boxes` does:
    /// upstream has exactly one `MetaInterp.virtualref_boxes`, and both the
    /// MIFrame leg and the full-body walker record into this one trace.
    ///
    /// Returns the recorded box and the concrete vref.  Upstream returns only
    /// the box because the `jit.virtual_ref` call it lowers hands its runtime
    /// result straight back to the interpreter's `enter`; pyre's walker has to
    /// perform that store itself, so it needs both.
    pub fn opimpl_virtual_ref(
        &mut self,
        virtual_obj: OpRef,
        virtual_obj_ptr: usize,
    ) -> (OpRef, *mut u8) {
        // virtualref.py `virtual_ref_during_tracing(real_object)` starts with
        // `assert real_object`: the tracing-time vref exists to name a live
        // object, and one built over a null would carry a null `forced` that
        // every non-forcing reader resolves as "still virtual".
        assert_ne!(
            virtual_obj_ptr, 0,
            "opimpl_virtual_ref requires the tracing-time real object"
        );
        // pyjitpl.py:1804 `vref = vrefinfo.virtual_ref_during_tracing(box)`.
        let vref_ptr = self
            .metainterp_sd
            .virtualref_info
            .virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
        // pyjitpl.py:1805 `cindex = ConstInt(len(virtualref_boxes) // 2)`.
        let cindex = self.const_int((self.virtualref_boxes.len() / 2) as i64);
        // pyjitpl.py:1806-1807 `record2(VIRTUAL_REF, box, cindex)` +
        // `heapcache.new(resbox)`, bundled by `virtual_ref`.
        let vref = self.virtual_ref(virtual_obj, cindex);
        // pyjitpl.py:1814 `virtualref_boxes += [virtualbox, vrefbox]`.
        self.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
        self.virtualref_boxes.push((vref, vref_ptr as usize));
        (vref, vref_ptr)
    }

    /// `pyjitpl.py opimpl_virtual_ref_finish(box)` —
    /// `ExecutionContext.leave`'s `jit.virtual_ref_finish`
    /// (`executioncontext.py`).  The vrefbox is reconstituted by popping,
    /// not passed in, so the stack discipline is checked rather than assumed.
    ///
    /// Returns whether a pair was popped: the walker brackets a callee level
    /// whose `enter` may have been skipped, and an unbalanced pop would eat
    /// an enclosing level's pair.
    pub fn opimpl_virtual_ref_finish(&mut self, virtual_obj: OpRef) -> bool {
        // pyjitpl.py:1820-1822 `vrefbox = pop(); lastbox = pop()`.
        let Some((vrefbox, vref_ptr)) = self.virtualref_boxes.pop() else {
            return false;
        };
        let (lastbox, _lastbox_ptr) = self
            .virtualref_boxes
            .pop()
            .expect("opimpl_virtual_ref_finish: vrefbox without its virtualbox");
        // pyjitpl.py `assert box.getref_base() == lastbox.getref_base()`
        // — compare the concrete ref base, not the SSA OpRef.  PyPy permits
        // alias boxes that share `getref_base()` but differ in box identity;
        // an `OpRef`-identity assert would reject those.
        //
        // `concrete_of_opref` is pyre's `getref_base()`: it reads a ConstPtr's
        // value inline and otherwise resolves the `opref_concrete` stamp that
        // every recording site writes.  Sourcing it that way is what gives the
        // assert teeth — deriving it from `lastbox_ptr` on a non-const box, as
        // an earlier spelling did, compared the popped pointer against itself
        // and could never fire, which is exactly the mismatched-nesting bug
        // upstream is asserting against.  A box with no stamp at all is not a
        // mismatch, only an unknown, so it is skipped rather than failed.
        // Both sides are read now, through the stamp, exactly as upstream calls
        // `getref_base()` on both.  `lastbox_ptr` is the address the object had
        // when the pair was pushed, and the object it names is movable: a minor
        // collection between `opimpl_virtual_ref` and here relocates it and
        // forwards the stamp, leaving the pushed copy naming the old address.
        // Comparing a forwarded address against that copy reports a mismatch
        // that is only the move.  A box with no stamp is an unknown, not a
        // mismatch, so either side missing skips the check.
        if let Some(Value::Ref(r)) = self.concrete_of_opref(virtual_obj)
            && let Some(Value::Ref(last)) = self.concrete_of_opref(lastbox)
        {
            // RPython's plain `assert` fires in both untranslated and
            // translated builds, so this is an `assert_eq!`: a release build
            // must fail at the divergence rather than silently corrupt the
            // vref stack. `SwitchToBlackhole` never reaches this finish;
            // a mismatched pair is a walker that kept recording after the raise.
            assert_eq!(
                r.as_usize(),
                last.as_usize(),
                "opimpl_virtual_ref_finish: leaving frame ref != top virtualref ref \
                 (virtual_obj={virtual_obj:?}, lastbox={lastbox:?})"
            );
        }
        // pyjitpl.py:1826-1832 `if vrefinfo.is_virtual_ref(vref): record
        // VIRTUAL_REF_FINISH`.  False once `stop_tracking_virtualref` has
        // replaced the box with ConstPtr(NULL) — the finish already ran.
        let is_vref = vref_ptr != 0
            && unsafe {
                self.metainterp_sd
                    .virtualref_info
                    .is_virtual_ref(vref_ptr as *const u8)
            };
        if is_vref {
            // pyjitpl.py:1831-1832 `VIRTUAL_REF_FINISH(vrefbox, nullbox)`.
            let null = self.const_ref(0);
            let _ = Self::do_record_op(
                &mut self.recorder,
                OpCode::VirtualRefFinish,
                &[vrefbox, null],
            );
        }
        true
    }

    /// `opimpl_virtual_ref`'s recording half, for a `virtual_ref` over an
    /// object the trace builds and the tracer never holds: `VIRTUAL_REF(box,
    /// cindex)` with no tracing-time vref and no pair pushed.
    ///
    /// Nothing can force such a vref while the trace is recorded, so it stays
    /// out of `virtualref_boxes` across the residual-call protocol, whose
    /// `tracing_before_residual_call` / `tracing_after_residual_call` read a
    /// concrete vref.  [`Self::track_recorded_virtual_ref`] pushes the pair
    /// for the guards that resume inside its scope.
    pub fn record_virtual_ref(&mut self, virtual_obj: OpRef) -> OpRef {
        let cindex = self.const_int((self.virtualref_boxes.len() / 2) as i64);
        self.virtual_ref(virtual_obj, cindex)
    }

    /// `virtualref_boxes += [virtualbox, vrefbox]` for a
    /// [`Self::record_virtual_ref`] box.
    pub fn track_recorded_virtual_ref(&mut self, virtual_obj: OpRef, vref: OpRef) {
        self.virtualref_boxes.push((virtual_obj, 0));
        self.virtualref_boxes.push((vref, 0));
    }

    /// `opimpl_virtual_ref_finish` for a [`Self::track_recorded_virtual_ref`]
    /// pair: pops it and records `VIRTUAL_REF_FINISH(vrefbox, nullbox)`, or
    /// `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` when the scope's `leave`
    /// forces the vref.  With `record` false the pair is dropped and nothing
    /// is recorded, for a trace that is being abandoned.
    pub fn finish_recorded_virtual_ref(&mut self, forced: bool, record: bool) {
        let (vrefbox, _) = self
            .virtualref_boxes
            .pop()
            .expect("finish_recorded_virtual_ref: no open virtual_ref scope");
        let (virtualbox, _) = self
            .virtualref_boxes
            .pop()
            .expect("finish_recorded_virtual_ref: vrefbox without its virtualbox");
        if !record {
            return;
        }
        let second = if forced {
            virtualbox
        } else {
            self.const_ref(0)
        };
        let _ = Self::do_record_op(
            &mut self.recorder,
            OpCode::VirtualRefFinish,
            &[vrefbox, second],
        );
    }

    /// `pyjitpl.py MetaInterp.vable_and_vrefs_before_residual_call`
    /// — the vrefs half (the virtualizable-info half lives on
    /// `JitCodeMachine::prepare_standard_virtualizable_before_residual_call`).
    ///
    /// ```python
    /// vrefinfo = self.staticdata.virtualref_info
    /// for i in range(1, len(self.virtualref_boxes), 2):
    ///     vrefbox = self.virtualref_boxes[i]
    ///     vref = vrefbox.getref_base()
    ///     vrefinfo.tracing_before_residual_call(vref)
    /// ```
    ///
    /// Stamps `TOKEN_TRACING_RESCALL` on every live vref's FORCE_TOKEN
    /// field so `tracing_after_residual_call` can distinguish "forced
    /// during the call" (token differs) from "untouched" (still
    /// `TOKEN_TRACING_RESCALL`).  Without this pre-call stamp the
    /// post-call check sees `TOKEN_NONE` on every fresh vref and
    /// incorrectly flags it as forced.
    pub fn vrefs_before_residual_call(&mut self) {
        let mut i = 1;
        while i < self.virtualref_boxes.len() {
            let vref_ptr = self.virtualref_boxes[i].1;
            // SAFETY: `vref_ptr` was registered by `opimpl_virtual_ref`
            // with a valid `JitVirtualRef*`; `tracing_before_residual_call`
            // only writes the token field.
            unsafe {
                self.metainterp_sd
                    .virtualref_info
                    .tracing_before_residual_call(vref_ptr as *mut u8);
            }
            i += 2;
        }
    }

    /// `pyjitpl.py MetaInterp.vrefs_after_residual_call`.
    ///
    /// ```python
    /// def vrefs_after_residual_call(self):
    ///     vrefinfo = self.staticdata.virtualref_info
    ///     for i in range(0, len(self.virtualref_boxes), 2):
    ///         vrefbox = self.virtualref_boxes[i+1]
    ///         vref = vrefbox.getref_base()
    ///         if vrefinfo.tracing_after_residual_call(vref):
    ///             self.stop_tracking_virtualref(i)
    /// ```
    pub fn vrefs_after_residual_call(&mut self) {
        let mut forced_pairs: Vec<usize> = Vec::new();
        let mut i = 0;
        while i + 1 < self.virtualref_boxes.len() {
            let vref_ptr = self.virtualref_boxes[i + 1].1;
            // SAFETY: `vref_ptr` was registered by `opimpl_virtual_ref`
            // with a valid `JitVirtualRef*`; `tracing_after_residual_call`
            // only reads the token field.
            let forced = unsafe {
                self.metainterp_sd
                    .virtualref_info
                    .tracing_after_residual_call(vref_ptr as *mut u8)
            };
            if forced {
                forced_pairs.push(i);
            }
            i += 2;
        }
        for pair_index in forced_pairs {
            self.stop_tracking_virtualref(pair_index);
        }
    }

    /// `pyjitpl.py MetaInterp.stop_tracking_virtualref(i)`.
    ///
    /// ```python
    /// def stop_tracking_virtualref(self, i):
    ///     virtualbox = self.virtualref_boxes[i]
    ///     vrefbox = self.virtualref_boxes[i+1]
    ///     # record VIRTUAL_REF_FINISH here, which is before the actual
    ///     # CALL_xxx is recorded
    ///     self.history.record2(rop.VIRTUAL_REF_FINISH, vrefbox, virtualbox, None)
    ///     # mark this situation by replacing the vrefbox with ConstPtr(NULL)
    ///     self.virtualref_boxes[i+1] = CONST_NULL
    /// ```
    pub fn stop_tracking_virtualref(&mut self, i: usize) {
        let virtualbox = self.virtualref_boxes[i].0;
        let vrefbox = self.virtualref_boxes[i + 1].0;
        // `history.record2(VIRTUAL_REF_FINISH, vrefbox, virtualbox, None)`.
        Self::do_record_op(
            &mut self.recorder,
            OpCode::VirtualRefFinish,
            &[vrefbox, virtualbox],
        );
        let null_const = self.const_null();
        self.virtualref_boxes[i + 1] = (null_const, 0);
    }

    /// Create a standalone TraceCtx for testing or external use.
    ///
    /// Internally synthesizes a fresh `Arc<MetaInterpStaticData>` —
    /// test-only parity with `RPython test_opencoder.py metainterp_sd` `class
    /// metainterp_sd: all_descrs = []` which similarly stubs a
    /// MetaInterpStaticData fixture for unit tests. Production callers
    /// (`MetaInterp::force_start_tracing` / `setup_tracing` /
    /// `start_bridge_trace`) go through `TraceCtx::new` directly with
    /// `self.staticdata.clone()`.
    pub fn for_test(num_inputs: usize) -> Self {
        let mut recorder = Trace::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(majit_ir::Type::Int);
        }
        Self::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// Create a TraceCtx for tests whose input args have mixed types.
    /// Analog of RPython `MetaInterp.create_empty_loop()` +
    /// `inputargs = [Box(tp) for tp in types]`.
    pub fn for_test_types(types: &[majit_ir::Type]) -> Self {
        let mut recorder = Trace::new();
        for &tp in types {
            recorder.record_input_arg(tp);
        }
        Self::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// Like [`Self::for_test_types`] with a sparse failarg layout:
    /// reserved prefix `input_types.len()`, live boxes only where
    /// `live_inputs` is true (`Trace::with_input_layout`).
    pub fn for_test_input_layout(input_types: &[majit_ir::Type], live_inputs: &[bool]) -> Self {
        Self::new(
            Trace::with_input_layout(input_types, live_inputs),
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// Like [`Self::for_test_types`] but seeds the trace green key (and thus
    /// `root_green_key`).  A unit test that drives a loop-closing
    /// `jit_merge_point` uses this to model the trace as having STARTED at
    /// that loop header.
    pub fn for_test_types_with_green_key(types: &[majit_ir::Type], green_key: u64) -> Self {
        let mut recorder = Trace::new();
        for &tp in types {
            recorder.record_input_arg(tp);
        }
        Self::new(
            recorder,
            green_key,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// history.py `History.__init__` — bind `opencoder.Trace` once the
    /// live JIT has `metainterp_sd` and the inputarg cap. Tests that
    /// construct a `TraceCtx` without going through `setup_tracing` /
    /// `start_retrace` lazy-attach a dummy buffer on the first `record_*`.
    pub fn attach_live_byte_recorder(&mut self) {
        #[cfg(not(test))]
        self.recorder.attach_byte_buffer(self.metainterp_sd.clone());
        #[cfg(test)]
        let _ = self;
    }

    /// Take the recorder out of this context (consumes self).
    pub fn into_recorder(self) -> Trace {
        let mut recorder = self.recorder;
        recorder.materialize_into_ops();
        recorder
    }

    pub(crate) fn new(
        recorder: Trace,
        green_key: u64,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        // pyjitpl.py `self.current_merge_points = []` at MetaInterp
        // init and again in `handle_guard_failure`. The first header
        // visit appends; a seed entry here made every scan look like a
        // prior visit and forced a length filter in
        // `same_greenkey`.
        TraceCtx {
            recorder,
            metainterp_sd,
            green_key,
            root_green_key: green_key,
            green_key_raw: (0, 0),
            root_green_key_raw: (0, 0),
            inline_frames: Vec::new(),
            green_key_values: None,
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_live_null_slots: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            raw_vable_base_escape_pending: false,
            header_pc: 0,
            cut_inner_green_key: None,
            inline_loop_abort_pending: false,
            recursive_call_assembler_pending: None,
            current_merge_points: Vec::new(),
            header_greens: None,
            close_greens: None,
            live_portal_greens: None,
            portal_green_regs_i: Vec::new(),
            portal_green_regs_r: Vec::new(),
            portal_green_regs_f: Vec::new(),
            portal_red_regs_i: Vec::new(),
            portal_red_regs_r: Vec::new(),
            portal_red_regs_f: Vec::new(),
            live_portal_reds: None,
            close_green_pc: None,
            close_typed_key: None,
            close_jump_into_key: None,
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            walk_final_pc: None,
            last_mp_green_pc: None,
            abort_after_panic: false,
            symbolic_residual_abort: false,
            deterministic_bridge_abort: false,
            aborted_framestack: None,
            walk_final_reds: Vec::new(),
            close_jump_boxes: None,
            close_scalar_values: None,
            close_ref_scalar_values: None,
            walk_finish_values: Vec::new(),
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            compiled_key_for_greens_fn: None,
            is_bridge_trace: false,
            reads_module_global: false,
            bridge_target_header_pc: None,
            portal_call_depth_fn: None,
            portal_call_depth_adjust_fn: None,
            unique_id_for_greenkey_fn: None,
            current_call_id_fn: None,
            portal_trace_push_fn: None,
            seen_loop_header_for_jdindex: -1,
            seen_loop_header_jit_pc: None,
            bridge_resume_at_position: false,
            merge_point_resumed: false,
            walk_resume_pc: None,
            callinfocollection: None,
            call_pure_results: crate::optimizeopt::util::args_dict(),
            trace_limit: DEFAULT_TRACE_LIMIT,
            snapshots: Vec::new(),
            resumekey_original_loop_token: None,
            cpu: None,
            cut_observer: None,
            metainterp_cpu: crate::cpu::default_cpu(),
            bridge_exception_resume_prepared: false,
            bridge_exception_source_pc: None,
            bridge_exception_source_jitcode: None,
            bridge_exception_resume_pc: None,
            bridge_exception_resume_jitcode: None,
            bridge_saved_exc_op: None,
            replace_frames: None,
            virtualref_boxes: Vec::new(),
            bridge_inline_carrier: None,
            active_bridge_inline_carrier: None,
            bridge_reg_indices: None,
            bridge_virtual_ops: Vec::new(),
            bridge_resume_data: None,
            bridge_source_is_exception_guard: false,
            bridge_grab_seeded: false,
            bridge_replay_incomplete: false,
            trace_continuation_suspended: std::cell::Cell::new(false),
        }
    }

    /// Create a TraceCtx with a structured green key.
    pub(crate) fn with_green_key(
        recorder: Trace,
        green_key: u64,
        green_key_values: GreenKey,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        TraceCtx {
            recorder,
            metainterp_sd,
            green_key,
            root_green_key: green_key,
            green_key_raw: (0, 0),
            root_green_key_raw: (0, 0),
            inline_frames: Vec::new(),
            green_key_values: Some(green_key_values),
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_live_null_slots: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            raw_vable_base_escape_pending: false,
            header_pc: 0,
            cut_inner_green_key: None,
            inline_loop_abort_pending: false,
            recursive_call_assembler_pending: None,
            current_merge_points: Vec::new(),
            header_greens: None,
            close_greens: None,
            live_portal_greens: None,
            portal_green_regs_i: Vec::new(),
            portal_green_regs_r: Vec::new(),
            portal_green_regs_f: Vec::new(),
            portal_red_regs_i: Vec::new(),
            portal_red_regs_r: Vec::new(),
            portal_red_regs_f: Vec::new(),
            live_portal_reds: None,
            close_green_pc: None,
            close_typed_key: None,
            close_jump_into_key: None,
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            walk_final_pc: None,
            last_mp_green_pc: None,
            abort_after_panic: false,
            symbolic_residual_abort: false,
            deterministic_bridge_abort: false,
            aborted_framestack: None,
            walk_final_reds: Vec::new(),
            close_jump_boxes: None,
            close_scalar_values: None,
            close_ref_scalar_values: None,
            walk_finish_values: Vec::new(),
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            compiled_key_for_greens_fn: None,
            is_bridge_trace: false,
            reads_module_global: false,
            bridge_target_header_pc: None,
            portal_call_depth_fn: None,
            portal_call_depth_adjust_fn: None,
            unique_id_for_greenkey_fn: None,
            current_call_id_fn: None,
            portal_trace_push_fn: None,
            seen_loop_header_for_jdindex: -1,
            seen_loop_header_jit_pc: None,
            bridge_resume_at_position: false,
            merge_point_resumed: false,
            walk_resume_pc: None,
            callinfocollection: None,
            call_pure_results: crate::optimizeopt::util::args_dict(),
            trace_limit: DEFAULT_TRACE_LIMIT,
            snapshots: Vec::new(),
            resumekey_original_loop_token: None,
            cpu: None,
            cut_observer: None,
            metainterp_cpu: crate::cpu::default_cpu(),
            bridge_exception_resume_prepared: false,
            bridge_exception_source_pc: None,
            bridge_exception_source_jitcode: None,
            bridge_exception_resume_pc: None,
            bridge_exception_resume_jitcode: None,
            bridge_saved_exc_op: None,
            replace_frames: None,
            virtualref_boxes: Vec::new(),
            bridge_inline_carrier: None,
            active_bridge_inline_carrier: None,
            bridge_reg_indices: None,
            bridge_virtual_ops: Vec::new(),
            bridge_resume_data: None,
            bridge_source_is_exception_guard: false,
            bridge_grab_seeded: false,
            bridge_replay_incomplete: false,
            trace_continuation_suspended: std::cell::Cell::new(false),
        }
    }

    /// Stash the decoded inline-callee carrier for the bridge currently
    /// being set up. Overwrites any prior unconsumed value. `setup_bridge_sym`
    /// calls this; `trace_bytecode` drains it once via
    /// [`take_bridge_inline_carrier`](Self::take_bridge_inline_carrier).
    pub fn set_bridge_inline_carrier(&mut self, carrier: BridgeInlineCarrier) {
        self.bridge_inline_carrier = Some(carrier);
    }

    /// Take the decoded inline-callee carrier for the bridge about to be
    /// traced, leaving the field empty. Returns `None` for primary traces
    /// and single-frame bridges.
    pub fn take_bridge_inline_carrier(&mut self) -> Option<std::rc::Rc<BridgeInlineCarrier>> {
        let carrier = std::rc::Rc::new(self.bridge_inline_carrier.take()?);
        self.active_bridge_inline_carrier = Some(std::rc::Rc::clone(&carrier));
        Some(carrier)
    }

    /// The drained carrier the walk is using, see
    /// [`take_bridge_inline_carrier`](Self::take_bridge_inline_carrier).
    pub(crate) fn active_bridge_inline_carrier(&self) -> Option<&BridgeInlineCarrier> {
        self.active_bridge_inline_carrier.as_deref()
    }

    /// `box.getref_base()` of each Ref register of `recipe`: `concrete_r` was
    /// read off the guard's deadframe when the carrier was decoded, and a
    /// collection since then moved the objects it names. The register boxes
    /// are roots while the carrier is active, so their `_resref` is current.
    /// A slot whose box has no Ref concrete keeps the decoded value.
    pub fn reconstruct_recipe_with_current_refs(
        &self,
        recipe: &ReconstructRecipe,
    ) -> ReconstructRecipe {
        let mut current = recipe.clone();
        for (k, slot) in current.concrete_r.iter_mut().enumerate() {
            if !matches!(slot, Value::Ref(_)) {
                continue;
            }
            let Some(&opref) = recipe.registers_r.get(k) else {
                continue;
            };
            if opref.is_none() {
                continue;
            }
            if let Some(value @ Value::Ref(_)) = self.concrete_of_opref(opref) {
                *slot = value;
            }
        }
        current
    }

    /// Forward `ReconstructRecipe::concrete_r` ref words while the carrier
    /// still sits on this context. Those words are copies of resume values;
    /// once the carrier is taken off, the caller has to publish them onto a
    /// recorder cell or the shadow stack before the next minor.
    pub(crate) fn walk_bridge_carrier_concrete_refs(
        &mut self,
        visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    ) {
        let Some(carrier) = self.bridge_inline_carrier.as_mut() else {
            return;
        };
        for recipe in &mut carrier.recipes {
            for value in &mut recipe.concrete_r {
                if let majit_ir::Value::Ref(r) = value
                    && !r.is_null()
                    && *r != majit_ir::GcRef::NO_CONCRETE
                {
                    visitor(r);
                }
            }
        }
    }

    /// Stash the bridge guard frame's per-bank live register indices (set by
    /// `start_bridge_tracing` before `setup_bridge_sym`).
    pub fn set_bridge_reg_indices(&mut self, indices: crate::resume::FrameLivenessRegIndices) {
        self.bridge_reg_indices = Some(indices);
    }

    /// The bridge guard frame's per-bank live register indices, if stashed.
    /// A JitDriver `setup_bridge_sym` reads this to map each decoded frame
    /// value (laid out int-bank then ref-bank then float) to its sym slot.
    pub fn bridge_reg_indices(&self) -> Option<&crate::resume::FrameLivenessRegIndices> {
        self.bridge_reg_indices.as_ref()
    }

    /// Box already allocated for virtual `vidx` by an earlier reader of
    /// this bridge (`resume.py` `virtuals_cache.get_ptr`).
    pub fn bridge_virtual_op(&self, vidx: usize) -> Option<OpRef> {
        self.bridge_virtual_ops.get(vidx).copied().flatten()
    }

    /// Remember `getvirtual_ptr`'s box so the next reader stores that
    /// same `OpRef` (`resume.py` `virtuals_cache.set_ptr` / `set_int`).
    pub fn remember_bridge_virtual_op(&mut self, vidx: usize, op: OpRef) {
        if op.is_none() {
            return;
        }
        if self.bridge_virtual_ops.len() <= vidx {
            self.bridge_virtual_ops.resize(vidx + 1, None);
        }
        self.bridge_virtual_ops[vidx] = Some(op);
    }

    /// Guard resume storage for `consume_boxes`'s `getvirtual_ptr`.
    pub fn bridge_resume_data(&self) -> Option<&crate::jit_state::ResumeDataResult> {
        self.bridge_resume_data.as_ref()
    }

    /// Park the guard's `ResumeDataResult` before
    /// `rebuild_portal_framestack_from_resumedata` / `consume_boxes`.
    pub fn set_bridge_resume_data(&mut self, resume_data: crate::jit_state::ResumeDataResult) {
        self.bridge_resume_data = Some(resume_data);
    }

    /// Mark whether this bridge's source guard is an exception guard
    /// (`ResumeGuardExcDescr` / `ResumeGuardCopiedExcDescr` analog).
    pub fn set_bridge_source_is_exception_guard(&mut self, is_exception_guard: bool) {
        self.bridge_source_is_exception_guard = is_exception_guard;
        // A reused `TraceCtx` must not keep the previous bridge's grab.
        self.bridge_grab_seeded = false;
    }

    /// True only for bridge traces sourced from an exception guard descr.
    pub fn bridge_source_is_exception_guard(&self) -> bool {
        self.bridge_source_is_exception_guard
    }

    pub fn set_bridge_grab_seeded(&mut self, seeded: bool) {
        self.bridge_grab_seeded = seeded;
    }

    pub fn bridge_grab_seeded(&self) -> bool {
        self.bridge_grab_seeded
    }

    /// `prepare_resume_from_failure` already recorded the exception
    /// resume ops. The walker resumes at the handler it was given.
    pub fn bridge_exception_resume_prepared(&self) -> bool {
        self.bridge_exception_resume_prepared
    }

    /// `MIFrame.pc` after `prepare_resume_from_failure`.
    pub fn bridge_exception_resume_pc(&self) -> Option<usize> {
        self.bridge_exception_resume_pc
    }

    /// JitCode index of the frame `bridge_exception_resume_pc` belongs to.
    pub fn bridge_exception_resume_jitcode(&self) -> Option<i32> {
        self.bridge_exception_resume_jitcode
    }

    /// JitCode index of the frame whose pc the guard was recorded against.
    pub fn bridge_exception_source_jitcode(&self) -> Option<i32> {
        self.bridge_exception_source_jitcode
    }

    /// `MIFrame.pc` before `handle_possible_exception`. `generate_guard`
    /// captures the resume snapshot at this coordinate.
    pub fn bridge_exception_source_pc(&self) -> Option<usize> {
        self.bridge_exception_source_pc
    }

    /// Handler pc for a walk of `jitcode_index` entering at `position`.
    ///
    /// `None` when this walk is a different frame than the one
    /// `finishframe_exception` left on top: an inner jitcode pc must not
    /// be executed as an offset in the outer body.
    pub fn prepared_handler_pc_for(&self, jitcode_index: i32, position: usize) -> Option<usize> {
        if !self.bridge_exception_resume_prepared {
            return None;
        }
        if self.bridge_exception_resume_jitcode != Some(jitcode_index) {
            return None;
        }
        let prepared = self.bridge_exception_resume_pc?;
        if prepared == position {
            return None;
        }
        let same_body = self.bridge_exception_source_jitcode == Some(jitcode_index);
        let at_source = self.bridge_exception_source_pc == Some(position);
        if same_body {
            let moved = self.bridge_exception_source_pc != Some(prepared);
            if moved && !at_source {
                return None;
            }
            return Some(prepared);
        }
        if self.bridge_exception_source_jitcode.is_some() {
            return Some(prepared);
        }
        None
    }

    /// `SAVE_EXCEPTION` recorded by `_prepare_exception_resumption`.
    pub fn bridge_saved_exc_op(&self) -> Option<OpRef> {
        self.bridge_saved_exc_op
    }

    /// Drop the exception-guard resume `_prepare_exception_resumption`
    /// stored on this ctx.
    ///
    /// `pyjitpl.py MetaInterp.reached_loop_header` starts with
    /// `heapcache.reset()`. `prepare_resume_from_failure` has already
    /// consumed the exception at bridge start (`handle_possible_exception`),
    /// and only `live_arg_boxes` survive a later `compile.py compile_loop`
    /// `cut_trace_from`. A later walk must not name the pre-cut
    /// `SAVE_EXCEPTION` producer.
    pub fn consume_bridge_exception_resume(&mut self) {
        self.bridge_exception_resume_prepared = false;
        self.bridge_saved_exc_op = None;
        self.bridge_exception_resume_pc = None;
        self.bridge_exception_resume_jitcode = None;
        self.bridge_exception_source_pc = None;
        self.bridge_exception_source_jitcode = None;
    }

    /// Stand in for `prepare_resume_from_failure` in tests that then drive
    /// `reached_loop_header`.
    pub fn seed_bridge_exception_resume(&mut self, saved_exc: OpRef) {
        self.bridge_exception_resume_prepared = true;
        self.bridge_saved_exc_op = Some(saved_exc);
        self.bridge_exception_resume_pc = Some(0);
        self.bridge_exception_resume_jitcode = Some(0);
        self.bridge_exception_source_pc = Some(0);
        self.bridge_exception_source_jitcode = Some(0);
    }

    /// Get or create a constant OpRef for a given i64 value.
    ///
    /// history.py `ConstInt(value).value` is inline on the Box;
    /// pyre mirrors this with `OpRef::ConstInt` — no pool allocation.
    pub fn const_int(&mut self, value: i64) -> OpRef {
        OpRef::const_int(value)
    }

    /// executor.py constant_from_op(op) parity: get typed Value for OpRef.
    /// history.py/268/314 — inline-Const carries the value directly.
    pub fn constants_get_value(&self, opref: OpRef) -> Option<Value> {
        opref.inline_const_to_value()
    }

    /// `IntFrontendOp(pos, intval)` / `FloatFrontendOp(pos, floatval)`
    /// / `RefFrontendOp(pos, gcref)` parity — stamp the frontend object
    /// for this OpRef position with its runtime concrete value.  Routes
    /// the write to the canonical `InputArg` / `Op` `value` field
    /// (`recorder.set_concrete_at`) instead of a flat-OpRef side table;
    /// matches RPython where the value lives on the operation-result
    /// object itself.  Const OpRefs carry their value inline, so the
    /// call is a no-op for them.
    ///
    /// **Invariant** (`history.py *FrontendOp(pos, value)` parity):
    /// the recorded position for `opref.raw()` must already exist — Pyre
    /// allocates it at every `record_op*` / `record_input_arg` site,
    /// mirroring RPython where instantiating `IntFrontendOp(pos, value)`
    /// *is* the object and there is no "stamp before allocation" state.
    /// A missing position here means a synthetic / stale OpRef (test
    /// fixture or bridge auxiliary path) is trying to stamp an object
    /// that was never constructed — an invariant violation that would
    /// silently swallow the value under the previous `if let Some`
    /// shape and hide cache-hit sanity-check mismatches.  Panic instead.
    #[track_caller]
    pub fn set_opref_concrete(&mut self, opref: OpRef, concrete: Value) {
        if opref.is_constant() || matches!(opref, OpRef::VoidOp(_)) {
            return;
        }
        // Stamp the concrete value on the canonical `InputArg`/`Op` identity
        // (`history.py *FrontendOp(pos, value)` — the value lives on the
        // op object, not a side pool). A missing slot means a synthetic /
        // stale OpRef is trying to stamp a value before the op was recorded —
        // an invariant violation; panic rather than silently swallow it.
        if !self.recorder.set_concrete_at(opref.raw(), concrete) {
            panic!(
                "set_opref_concrete: no recorded op/inputarg for OpRef position \
                 {} ({opref:?}) — the *FrontendOp must be recorded by \
                 record_op*/record_input_arg before its value can be stamped \
                 (history.py:803 *FrontendOp invariant)",
                opref.raw(),
            );
        }
    }

    /// `resume.py ResumeDataBoxReader.load_box_from_cpu`: the bridge
    /// InputArg is born with `cpu.get_int_value(deadframe, num)`.
    pub fn load_box_from_cpu(&mut self, num: u32, kind: Type, value: Value) -> OpRef {
        self.recorder.load_box_from_cpu(num, kind, value)
    }

    /// Like [`Self::set_opref_concrete`] but returns `false` instead of
    /// panicking when no frontend op/inputarg is recorded at `opref`'s
    /// position.  The full-body walker's speculative residual-call
    /// execution (`try_execute_residual_call_via_walker`) can compute a
    /// concrete for an OpRef recorded in a context whose op was not
    /// allocated in the active recorder (a deeper inlined / recursive
    /// frame's result).  Leaving that result symbolic makes the downstream
    /// branch abort the trace cleanly rather than crash the tracer.
    #[track_caller]
    pub fn try_set_opref_concrete(&mut self, opref: OpRef, concrete: Value) -> bool {
        if opref.is_constant() || matches!(opref, OpRef::VoidOp(_)) {
            return true;
        }
        self.recorder.set_concrete_at(opref.raw(), concrete)
    }

    /// `get_value` reader — the concrete value stamped onto
    /// this OpRef's frontend value slot (`history.py *FrontendOp(pos,
    /// value)` analog).  Const variants delegate to
    /// `Forwarded::Const { value, .. }` directly.
    ///
    /// PyPy's normal record path attaches the value at FrontendOp
    /// construction time (execute() runs before record()), so for any
    /// op produced by the normal trace path the answer is always
    /// `Some(_)`.  The `None` arm is reserved for the residual-call /
    /// guard / unstamped result family where no trace-time concrete
    /// exists until blackhole runs the op — plus synthetic / test
    /// fixtures that materialise OpRefs without going through
    /// `record_op*`.  Callers MUST treat `None` as the exceptional
    /// branch (skip the sanity check, leave the cache entry alone);
    /// silently substituting `Value::Void` would conflate "unstamped"
    /// with "stamped Void", which the `set_value` type-check
    /// already forbids.
    ///
    /// A stamped `Value::Ref(GcRef::NO_CONCRETE)` comes back as a value, not
    /// as `None`: `heapcache_ops`' materialized-array walk writes that
    /// sentinel over a load it could not replay, and this table hands back
    /// what it holds.  [`Self::recover_ref_value`] rejects the sentinel; this
    /// method deliberately does not, because "never stamped" and "stamped
    /// unresolved" are different states and only the caller knows which one
    /// it can act on.
    ///
    /// Three sites pair the two in the order `lookup_opref_concrete(..)
    /// .or_else(|| recover_ref_value(..))` — `vable_value_concrete` and
    /// `current_inline_vable_target` in `jitcode_dispatch/vable_ops.rs`, and
    /// `fill_trace_too_long_register_banks` in
    /// `jitcode_dispatch/residual_call.rs`.  `Some(_).or_else(f)` never calls
    /// `f`, so a stamped sentinel short-circuits ahead of the guard in
    /// `recover_ref_value` and reaches the consumer.  Two more accept it
    /// through a bare `value.0 != 0` arm, which excludes NULL but not
    /// `usize::MAX - 1`: the snapshot live-root reads in the same file.
    ///
    /// Filtering the sentinel at those sites is NOT the one-line fix it looks
    /// like, and the reading that says so is worth keeping.  Each of them
    /// ends in a fallback tail — `.or(from_register)`, `.or(from_shadow)`,
    /// `_ => sym.live_vable_frame_addr()` — so rejecting the sentinel does
    /// not decline the image, it PROMOTES the next fallback, and that is a
    /// different concrete address landing in a blackhole frame's ref bank or,
    /// through `store_live_frame_array_slot`, in a live frame's locals array
    /// behind a write barrier.  `vable_ops.rs` records of the shadow it would
    /// promote that re-reading that color "can return a stale concrete value
    /// from a prior loop iteration".  That is wrong data, not a refusal.  A
    /// fix has to decide what each site should answer for an unresolved box,
    /// which is a per-site question; dropping the sentinel here or there is
    /// not it.
    pub fn lookup_opref_concrete(&self, opref: OpRef) -> Option<Value> {
        if opref.is_constant() {
            return opref.inline_const_to_value();
        }
        if matches!(opref, OpRef::VoidOp(_)) {
            return None;
        }
        self.recorder.concrete_at(opref.raw())
    }

    /// Canonical `Operand` (box object) for a value `OpRef`. The trace-record
    /// analogue of the box objects RPython holds in `MIFrame.registers_r`
    /// (`pyjitpl.py`): it surfaces the recorder's real per-op `Rc<Op>` /
    /// `Rc<InputArg>` identity so consumers can key by box identity
    /// (`Operand::eq` = `Rc::ptr_eq`) rather than flat `OpRef` position, matching
    /// the box-keyed dicts upstream (e.g. `heapcache.py` `cache_anything[ref_box]`).
    /// Deterministic: the same recorded position always yields an `Operand`
    /// wrapping the same producer `Rc`.
    pub fn operand_for(&mut self, opref: OpRef) -> majit_ir::operand::Operand {
        self.recorder.box_for_operand(opref)
    }

    /// Report that the applying half of the bridge-entry replay could not
    /// finish. See [`TraceCtx::bridge_replay_incomplete`].
    pub fn mark_bridge_replay_incomplete(&mut self) {
        self.bridge_replay_incomplete = true;
    }

    /// Whether [`TraceCtx::mark_bridge_replay_incomplete`] fired for this
    /// bridge entry.
    pub fn bridge_replay_incomplete(&self) -> bool {
        self.bridge_replay_incomplete
    }

    /// `AbstractValue.getint()` / `getref_base()` / `getfloatstorage()`:
    /// constants carry the payload inline; `*FrontendOp` / `InputArg*`
    /// read `_resint` / `_resref` / `_resfloat` off the recorded box.
    pub fn box_value(&self, opref: OpRef) -> Option<Value> {
        if let Some(v) = opref.inline_const_to_value() {
            Some(v)
        } else {
            self.lookup_opref_concrete(opref)
        }
    }

    /// `IntOp.getint` / `RefOp.getref_base` / `FloatOp.getfloat_storage`
    /// as machine bits. Constants are inline on the OpRef; value ops
    /// and inputargs read `_res*` off the FrontendOp table.
    pub fn box_bits(&self, opref: OpRef) -> Option<i64> {
        self.box_value(opref).map(|v| v.as_raw_i64())
    }

    /// `InputArgInt(value)` construction: stamp each recorded inputarg
    /// with the live value `create_empty_history` saw.
    pub fn stamp_live_inputargs(&mut self, live_values: &[Value]) {
        for (i, value) in live_values.iter().enumerate() {
            let opref = OpRef::input_arg_typed(i as u32, value.get_type());
            self.set_opref_concrete(opref, *value);
        }
    }

    /// Plant-time `box.getint()`: write the argbox's concrete bits onto
    /// the box (`try_set_opref_concrete`). Const OpRefs are a no-op.
    pub fn stamp_argboxes(&mut self, argboxes: &[(JitArgKind, OpRef, i64)]) {
        for (kind, opref, bits) in argboxes {
            let concrete = match kind {
                JitArgKind::Int => Value::Int(*bits),
                JitArgKind::Ref => Value::Ref(majit_ir::GcRef(*bits as usize)),
                JitArgKind::Float => Value::Float(f64::from_bits(*bits as u64)),
            };
            let _ = self.try_set_opref_concrete(*opref, concrete);
        }
    }

    /// RPython parity: Ref constants preserve their type so guard
    /// fail_args are correctly typed during guard failure recovery.
    /// history.py `ConstPtr.value` is a field on the Box; pyre
    /// stores a `const_ptr_table` index on `OpRef::ConstPtr`. The
    /// op-graph walker traces that index across minor collection.
    ///
    /// `intern` records the address; the slot is a root only while a
    /// holder traces it (`const_ptr_table::trace_index`). A Rust
    /// `OpRef` is a Copy index, not that holder. Register the box in
    /// `recorder.const_ptrs` immediately so the next Trace-pool append
    /// forwards `ConstPtr.value` the way the translated local would.
    pub fn const_ref(&mut self, value: i64) -> OpRef {
        let addr = majit_gc::gc_current_object_address(value as usize);
        let pin = (addr != 0 && majit_gc::gc_owns_object(addr))
            .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(addr)));
        let opref = OpRef::const_ptr(
            pin.as_ref()
                .map(|p| p.get())
                .unwrap_or(majit_ir::GcRef(addr)),
        );
        self.recorder.hold_live_const_ptr(opref);
        opref
    }

    /// history.py CONST_NULL = ConstPtr(ConstPtr.value).
    /// Ref-typed null pointer constant.
    pub fn const_null(&mut self) -> OpRef {
        self.const_ref(0)
    }

    /// Get or create a Float-typed constant OpRef.
    ///
    /// history.py `ConstFloat(valuestorage).value` is inline on the
    /// Box; pyre mirrors with `OpRef::ConstFloat`. The incoming
    /// `value: i64` is the longlong float-storage form (raw bits) per
    /// RPython `longlong.FLOATSTORAGE`; convert to `f64` for the inline
    /// payload so equality/hash use bitwise compare (history.py:283/292).
    pub fn const_float(&mut self, value: i64) -> OpRef {
        OpRef::const_float(f64::from_bits(value as u64))
    }

    /// Return the type of a constant OpRef, if recorded.
    /// history.py/268/314 — inline-Const carries type intrinsically.
    pub fn const_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        opref.ty()
    }

    /// Return the concrete value for a constant OpRef as raw i64 bits.
    /// history.py/268/314 — inline-Const carries the value directly.
    pub fn const_value(&self, opref: OpRef) -> Option<i64> {
        opref.inline_const_bits()
    }

    /// Typed counterpart to [`Self::const_value`] — returns the
    /// `Value` (`Int`/`Ref`/`Float`/`Void`) directly instead of the
    /// raw `i64` cast.  convergence path: optimizer /
    /// guard-recovery consumers that need to distinguish Ref vs Int
    /// constants should migrate to this reader so the raw-i64 API can
    /// retire once the backend `set_constants` signature flips.
    pub fn const_typed_value(&self, opref: OpRef) -> Option<majit_ir::Value> {
        opref.inline_const_to_value()
    }

    /// M1 bridge: translate a pyre `OpRef` into the `opencoder::Box` that
    /// `TraceRecordBuffer::record_op(&[Box], descr)` expects.
    ///
    /// Inline Const OpRefs carry their value directly
    /// (`OpRef::inline_const_to_value()`); no side pool is consulted.
    /// RPython's opencoder takes concrete `Const{Int,Float,Ptr}` /
    /// `AbstractResOp` boxes and encodes them inline through the
    /// `_bigints` / `_floats` / `_refs` pools in `_encode`.
    ///
    /// This helper is the conversion point between the two worlds.  It
    /// unblocks M2 (routing `TraceCtx::record_*` through TraceRecordBuffer's
    /// Box-taking API) without touching any call site yet.
    ///
    /// Panics when a constant OpRef is not inline-resolvable — that is a
    /// genuine invariant break.
    #[allow(dead_code)]
    pub(crate) fn opref_to_box(&self, opref: OpRef) -> OcBox {
        if opref.is_constant() {
            let value = opref.inline_const_to_value().unwrap_or_else(|| {
                panic!("opref_to_box: constant {:?} not inline-resolvable", opref)
            });
            match value {
                Value::Int(v) => OcBox::ConstInt(v),
                Value::Float(f) => OcBox::ConstFloat(f.to_bits()),
                Value::Ref(r) => OcBox::ConstPtr(r.as_usize() as u64),
                Value::Void => {
                    panic!("opref_to_box: constant {:?} has Void type", opref)
                }
            }
        } else {
            OcBox::of_op(opref)
        }
    }

    /// RPython `original_boxes[index]` lookup for the currently active trace.
    ///
    /// `MetaInterp.setup_tracing` snapshots each trace-entry concrete value in
    /// `initial_inputarg_consts`; the inputarg Box identity itself is still the
    /// ordinary `OpRef(index)`, matching RPython's `original_boxes` list.
    pub fn initial_inputarg_argbox(&self, index: usize) -> Option<(JitArgKind, OpRef, i64)> {
        let tp = self.recorder.inputarg_type_at(index as u32)?;
        let const_ref = self.initial_inputarg_consts.get(index)?;
        // history.py/268/314 — Const{Int,Float,Ptr}.value lives inline
        // on the Box; read it and resolve the raw bits (Int→value,
        // Float→bit pattern, Ref→gcref address).
        let bits = match const_ref.inline_const_to_value()? {
            Value::Int(v) => v,
            Value::Float(v) => v.to_bits() as i64,
            Value::Ref(r) => r.0 as i64,
            Value::Void => return None,
        };
        let kind = match tp {
            Type::Int => JitArgKind::Int,
            Type::Ref => JitArgKind::Ref,
            Type::Float => JitArgKind::Float,
            Type::Void => return None,
        };
        // resoperation.py InputArgInt/727/739 InputArg{Int,Float,Ref}: the
        // inputarg Box carries `box.type` directly. Mint the typed
        // variant here so callers see the same {Int,Float,Ref} discrimination
        // RPython's original_boxes[index] would produce.
        Some((kind, OpRef::input_arg_typed(index as u32, tp), bits))
    }

    /// JitCode setup argbox for the standard virtualizable.
    ///
    /// This is the walk's counterpart of
    /// `pyjitpl.py f.setup_call(original_boxes)`: prefer the exact
    /// trace-entry red inputarg named by `jd.index_of_virtualizable`, and
    /// fall back to `virtualizable_boxes[-1]` only for legacy pyre traces that
    /// initialized the standard virtualizable before descriptor metadata was
    /// threaded through.
    pub fn standard_virtualizable_jitcode_argbox(&self) -> Option<(JitArgKind, OpRef, i64)> {
        if let Some(argbox) = self
            .driver_descriptor()
            .and_then(|driver| driver.virtualizable_arg_index())
            .and_then(|index| self.initial_inputarg_argbox(index))
        {
            return Some(argbox);
        }

        let opref = self.standard_virtualizable_box()?;
        let concrete = match self.standard_virtualizable_concrete()? {
            Value::Ref(r) => r.as_usize() as i64,
            Value::Int(v) => v,
            Value::Float(v) => v.to_bits() as i64,
            Value::Void => return None,
        };
        Some((JitArgKind::Ref, opref, concrete))
    }

    /// to a reached loop header during tracing.
    pub fn root_green_key(&self) -> u64 {
        self.root_green_key
    }

    /// See `TraceCtx::close_jump_into_key`.  Read-and-clear: the answer is only
    /// meaningful to the close that just set it.
    pub fn take_close_jump_into_key(&mut self) -> Option<u64> {
        self.close_jump_into_key.take()
    }

    /// Mark that the current back-edge was reached inside an inline callee
    /// frame and must not be unrolled (opimpl_jit_merge_point
    /// portal_call_depth>0). The trace step drains this via
    /// [`Self::take_inline_loop_abort`] and aborts the trace.
    pub fn request_inline_loop_abort(&mut self) {
        self.inline_loop_abort_pending = true;
    }

    /// Read and clear the inline-loop abort signal.
    pub fn take_inline_loop_abort(&mut self) -> bool {
        std::mem::take(&mut self.inline_loop_abort_pending)
    }

    /// Mark that the current inline-frame back-edge targets a loop whose
    /// green key already has compiled code, so the metainterp should pop
    /// the inline frame and record a CALL_ASSEMBLER into the loop token
    /// from the parent frame (opimpl_jit_merge_point
    /// portal_call_depth>0, pyjitpl.py). Drained via
    /// [`Self::take_recursive_call_assembler`].
    pub fn request_recursive_call_assembler(&mut self, green_key: u64, target_pc: usize) {
        self.recursive_call_assembler_pending = Some((green_key, target_pc));
    }

    /// Read and clear the recursive-call-assembler signal.
    pub fn take_recursive_call_assembler(&mut self) -> Option<(u64, usize)> {
        self.recursive_call_assembler_pending.take()
    }

    /// Number of input arguments to the current trace.
    pub fn num_inputs(&self) -> usize {
        self.recorder.num_inputargs()
    }

    /// True when `r` names a recorded void opcode (e.g. `DebugMergePoint`)
    /// at that raw slot. A Ref-tagged copy of the same raw must not be
    /// used as a JUMP red — `box_for_operand` would otherwise bind it to
    /// the void producer.
    pub fn opref_is_void_producer(&self, r: OpRef) -> bool {
        if r.is_none() || r.ty() == Some(Type::Void) || matches!(r, OpRef::VoidOp(_)) {
            return true;
        }
        if r.is_constant() || r.is_input_arg() {
            return false;
        }
        // Byte-mode `ops` is empty until materialize. `opcode_of` reads
        // FrontendSlots by `_index` / `_count` (`history.py getopnum`).
        self.recorder
            .opcode_of(r)
            .is_some_and(|op| op.result_type() == Type::Void)
    }

    /// Types of the live inputargs, in `History.set_inputargs` order.
    /// Dead failarg holes are absent; do not index this list by
    /// `get_position()`. Use [`Self::inputarg_type_at`] or `opref.ty()`.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.recorder.inputarg_types()
    }

    /// Type of the live InputArg whose `get_position()` is `position`.
    /// `None` when that coordinate is a dead failarg hole.
    ///
    /// `resume.py ResumeDataBoxReader.load_box_from_cpu` yields a typed
    /// box; the type is on the box, not a compact-list index.
    pub fn inputarg_type_at(&self, position: u32) -> Option<Type> {
        self.recorder.inputarg_type_at(position)
    }

    /// One GreenBox per live recorder inputarg, each keeping its
    /// original `get_position()`.
    pub fn inputarg_greenboxes(&self) -> Vec<GreenBox> {
        GreenBox::from_recorder_inputargs(&self.recorder)
    }

    /// `History.length`: `trace._count - len(trace.inputargs)`.
    /// Trace-length limit; dead failarg holes count. The recorded-op
    /// index bound for [`Self::opcode_at`] is [`Self::opcode_at_len`].
    pub fn num_ops(&self) -> usize {
        self.recorder.num_ops()
    }

    /// Opcode of the recorded op named by `opref`
    /// (`history.py AbstractResOp.getopnum`).
    pub fn opcode_of(&self, opref: OpRef) -> Option<OpCode> {
        self.recorder.opcode_of(opref)
    }

    /// Diagnostic: dump every recorded op (result OpRef = pos, opcode, args)
    /// to stderr.  Used by the P2 carrier investigation to inspect the
    /// def-use of the fused (callee continuation + root) bridge trace —
    /// specifically whether the injected result OpRef has a reachable
    /// def-chain bottoming in trace input args.
    pub fn dump_trace_ops_diag(&self, label: &str) {
        use majit_ir::operand::Operand;
        eprintln!("[p2-ir] {label} num_ops={}", self.recorder.num_ops());
        for op in &self.recorder.materialize_ops() {
            let args: Vec<String> = op
                .args_slice()
                .iter()
                .map(|a| {
                    if a.is_none() {
                        "_".to_string()
                    } else if let Some(o) = a.bound_op() {
                        format!("{:?}", o.pos().get())
                    } else if let Some(ia) = a.bound_inputarg() {
                        format!("IA{}", ia.index)
                    } else if a.is_null_ref() {
                        "CRef(NULL)".to_string()
                    } else {
                        format!("C{:?}", a.const_value().unwrap())
                    }
                })
                .collect();
            eprintln!(
                "[p2-ir]   {:?} = {:?} [{}]",
                op.pos().get(),
                op.opcode,
                args.join(" ")
            );
        }
    }

    /// Number of guard operations recorded so far.  The walker compares
    /// this across a `vable_getfield_*` / `vable_setfield` call to detect
    /// the `_nonstandard_virtualizable` PTR_EQ promote guard those helpers
    /// emit internally, so it can attach a resume snapshot to it.
    pub fn num_guards(&self) -> usize {
        self.recorder.num_guards()
    }

    /// Opcode of the most recently recorded guard, if any
    /// (`pyjitpl.py:2599-2603` — snapshot capture keys
    /// `after_residual_call` on the guard opcode).
    pub fn last_guard_opcode(&self) -> Option<OpCode> {
        self.recorder.last_guard_opcode()
    }

    pub fn set_last_op_descr(&mut self, descr: DescrRef) {
        self.recorder.set_last_op_descr(descr);
    }

    pub fn set_guard_op_descr_from_end(&mut self, from_end: usize, descr: DescrRef) {
        self.recorder.set_guard_op_descr_from_end(from_end, descr);
    }

    pub fn last_op_opcode(&self) -> Option<OpCode> {
        self.recorder.last_op_opcode()
    }

    pub fn guard_op_opcode_from_end(&self, from_end: usize) -> Option<OpCode> {
        self.recorder.guard_op_opcode_from_end(from_end)
    }

    pub fn guard_op_resume_position_from_end(&self, from_end: usize) -> Option<i32> {
        self.recorder.guard_op_resume_position_from_end(from_end)
    }

    /// The structured green key values, if provided.
    pub fn green_key_values(&self) -> Option<&GreenKey> {
        self.green_key_values.as_ref()
    }

    /// pyjitpl.py: `compile_loop` keys the JitCell by
    /// `original_boxes[:num_green_args]`, i.e. the greens captured at the
    /// merge point that closed the trace.
    pub fn close_green_key_hash(&self) -> Option<u64> {
        Some(self.close_green_key()?.get_uhash())
    }

    /// The typed form of [`Self::close_green_key_hash`].
    ///
    /// A hash alone cannot say WHICH cell in a chained bucket the close
    /// belongs to, and the loop this key files under is the one a later entry
    /// has to find again. Callers with a `WarmEnterState` in hand resolve
    /// through this (`WarmEnterState::resolve_cell_key`) and file under the
    /// resolved cell key; the vectors it builds are the ones
    /// `merge_point_green_key_hash` already built to hash.
    pub fn set_close_typed_key(&mut self, key: GreenKey) {
        self.close_typed_key = Some(key);
    }

    pub fn close_green_key(&self) -> Option<GreenKey> {
        if let Some(key) = &self.close_typed_key {
            return Some(key.clone());
        }
        let greens = self.close_greens.as_ref()?;
        self.merge_point_green_key(&greens.0, &greens.1, &greens.2)
    }

    /// Green banks `warmspot.py handle_jitexception` assigns after a
    /// close. Prefer the merge-point snapshot (`close_greens`), then the
    /// live registers re-read after a later green write, then the
    /// trace-start header. A missing bank is a missing snapshot:
    /// `warmspot.py` `getattr(e, attrname)[count]` fails on a short list
    /// rather than inventing a pc-only `ContinueRunningNormally`.
    pub fn portal_resume_args(&self) -> crate::jitexc::ContinueRunningNormallyArgs {
        if let Some((ints, refs, floats)) = self.close_greens.clone() {
            return crate::jitexc::ContinueRunningNormallyArgs::from_green_banks(
                ints, refs, floats,
            );
        }
        if let Some((ints, refs, floats)) = self.live_portal_greens.clone() {
            return crate::jitexc::ContinueRunningNormallyArgs::from_green_banks(
                ints, refs, floats,
            );
        }
        if let Some((ints, refs, floats)) = self.header_greens.clone() {
            return crate::jitexc::ContinueRunningNormallyArgs::from_green_banks(
                ints, refs, floats,
            );
        }
        crate::jitexc::ContinueRunningNormallyArgs::from_green_banks(
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Re-read declaration-order greens off a live frame's registers.
    ///
    /// `warmspot.py handle_jitexception` takes those values from
    /// `jitexc.py ContinueRunningNormally`. A green write after the last
    /// merge point lives in the merge-point's register (the `join_merge`
    /// header slot for a loop-carried green).
    pub fn snapshot_portal_greens_from_frame(
        &mut self,
        ints: &[Option<OpRef>],
        refs: &[Option<OpRef>],
        floats: &[Option<OpRef>],
    ) {
        if self.portal_green_regs_i.is_empty()
            && self.portal_green_regs_r.is_empty()
            && self.portal_green_regs_f.is_empty()
        {
            return;
        }
        fn read(ctx: &TraceCtx, bank: &[Option<OpRef>], regs: &[u8], what: &str) -> Vec<i64> {
            regs.iter()
                .map(|&reg| {
                    bank.get(reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "merge-point green {what} register {reg} must be live \
                                 (blackhole.py bhimpl_jit_merge_point)"
                            )
                        })
                })
                .collect()
        }
        self.live_portal_greens = Some((
            read(self, ints, &self.portal_green_regs_i, "int"),
            read(self, refs, &self.portal_green_regs_r, "ref"),
            read(self, floats, &self.portal_green_regs_f, "float"),
        ));
    }

    /// Re-read merge-point red registers off a live frame (`prepare_list_of_boxes`).
    pub fn snapshot_portal_reds_from_frame(
        &mut self,
        ints: &[Option<OpRef>],
        refs: &[Option<OpRef>],
        floats: &[Option<OpRef>],
    ) {
        fn read(bank: &[Option<OpRef>], regs: &[u8], what: &str, ty: Type) -> Vec<(OpRef, Type)> {
            regs.iter()
                .map(|&reg| {
                    let opref = bank
                        .get(reg as usize)
                        .copied()
                        .flatten()
                        .unwrap_or_else(|| {
                            panic!(
                                "merge-point red {what} register {reg} must be live \
                                 (`prepare_list_of_boxes` / `reached_loop_header`)"
                            )
                        });
                    (opref, ty)
                })
                .collect()
        }
        let mut reds = read(ints, &self.portal_red_regs_i, "int", Type::Int);
        reds.extend(read(refs, &self.portal_red_regs_r, "ref", Type::Ref));
        reds.extend(read(floats, &self.portal_red_regs_f, "float", Type::Float));
        self.live_portal_reds = Some(reds);
    }

    /// Header-revisit close: copy the live portal greens into
    /// `close_greens` when the merge point did not write its own.
    ///
    /// Callers re-read `last_mp_green_*` first (`snapshot_live_portal_greens`
    /// on Finish / Abort / SegmentedLoop / too-long / last-portal-pop, or
    /// `MetaInterp::close_header_revisit` on the generated fast path).
    pub fn adopt_live_greens_as_close(&mut self) {
        if self.close_greens.is_none() {
            self.close_greens = self.live_portal_greens.clone();
        }
    }

    /// `pyjitpl.py MetaInterp.reached_loop_header`: one `duplicates` dict;
    /// reds first, then `virtualizable_boxes[:-1]`. Stashes `close_jump_boxes`
    /// as the JUMP/registration list (reds + vable elements).
    ///
    /// `dead_array_tail_from` is forwarded to
    /// [`Self::fill_virtualizable_boxes_to_declared_layout`]: the first array
    /// index the caller knows holds null. Generic `VirtualizableInfo` has no
    /// frame field names (`virtualizable.py`); pass `None` to read every hole
    /// from the heap (`read_boxes`).
    pub fn reached_loop_header_live_arg_boxes(
        &mut self,
        redboxes: &mut [(OpRef, Type)],
        dead_array_tail_from: Option<usize>,
    ) -> Vec<(OpRef, Type)> {
        self.heap_cache_mut().reset();
        self.consume_bridge_exception_resume();
        self.fill_virtualizable_boxes_to_declared_layout(dead_array_tail_from);
        let mut duplicates: indexmap::IndexSet<OpRef, rustc_hash::FxBuildHasher> =
            indexmap::IndexSet::with_hasher(rustc_hash::FxBuildHasher);
        self.remove_consts_and_duplicates_with(redboxes, &mut duplicates);
        if let Some(mut typed) = self.collect_virtualizable_typed_boxes()
            && let Some(end) = typed.len().checked_sub(1)
        {
            self.remove_consts_and_duplicates_with(&mut typed[..end], &mut duplicates);
            let elements: Vec<OpRef> = typed[..end].iter().map(|(opref, _)| *opref).collect();
            self.adopt_normalized_virtualizable_elements(&elements);
        }
        // One list per visit: the same concatenation `compile_loop` /
        // `current_merge_points.append` / the closing JUMP read
        // (`pyjitpl.py` `reached_loop_header` `live_arg_boxes`).
        let live = self.live_arg_boxes_from_reds(redboxes);
        self.close_jump_boxes = Some(live.clone());
        live
    }

    /// pyjitpl.py `get_procedure_token(greenboxes)` analog: the
    /// jitcell key the INTERPRETER would enter by for these greens.  The
    /// grouping is per JitCode register bank; rebuild the declared green order
    /// before hashing so this matches warmstate.py `JitCell.get_uhash`.
    pub fn merge_point_green_key_hash(
        &self,
        ints: &[i64],
        refs: &[i64],
        floats: &[i64],
    ) -> Option<u64> {
        Some(self.merge_point_green_key(ints, refs, floats)?.get_uhash())
    }

    /// The typed form of [`Self::merge_point_green_key_hash`], for callers
    /// that must resolve the key to a cell rather than only hash it.
    pub fn merge_point_green_key(
        &self,
        ints: &[i64],
        refs: &[i64],
        floats: &[i64],
    ) -> Option<GreenKey> {
        // pyjitpl.py `reached_loop_header` / compile.py `compile_loop`:
        // `greenkey = original_boxes[:num_green_args]` — the declared greens
        // in declared order. warmstate.py `JitCell.get_uhash` hashes that
        // tuple. A driver that needs the loop-header pc declares it
        // (`interp_jit.py` `next_instr`); it is not an extra prepended slot.
        let spec: smallvec::SmallVec<[GreenType; majit_ir::GREEN_INLINE]> =
            if let Some(key) = self.green_key_values.as_ref() {
                smallvec::SmallVec::from_slice(&key.types)
            } else {
                smallvec::SmallVec::from_iter(
                    self.driver_descriptor
                        .as_ref()
                        .map(|d| d.green_args_spec())?
                        .into_iter(),
                )
            };

        let mut values = smallvec::SmallVec::<[i64; majit_ir::GREEN_INLINE]>::new();
        let mut types = smallvec::SmallVec::<[GreenType; majit_ir::GREEN_INLINE]>::new();
        let mut int_i = 0;
        let mut ref_i = 0;
        let mut float_i = 0;
        for tp in &spec {
            let value = match tp {
                GreenType::Int | GreenType::Void => {
                    let value = *ints.get(int_i)?;
                    int_i += 1;
                    value
                }
                GreenType::Ref | GreenType::Str | GreenType::Unicode => {
                    let value = *refs.get(ref_i)?;
                    ref_i += 1;
                    value
                }
                GreenType::Float => {
                    let value = *floats.get(float_i)?;
                    float_i += 1;
                    value
                }
            };
            values.push(value);
            types.push(*tp);
        }
        Some(GreenKey::with_types(values, types))
    }

    /// Set the structured green key values.
    pub fn set_green_key_values(&mut self, values: GreenKey) {
        self.green_key_values = Some(values);
    }

    /// The declarative JitDriver descriptor, if provided.
    pub fn driver_descriptor(&self) -> Option<&JitDriverStaticData> {
        self.driver_descriptor.as_ref()
    }

    /// Attach declarative JitDriver metadata to the active trace.
    pub fn set_driver_descriptor(&mut self, descriptor: JitDriverStaticData) {
        self.driver_descriptor = Some(descriptor);
    }

    /// pyjitpl.py `initialize_withgreenfields`: the single red that owns
    /// the green fields is the whole virtualizable box list.
    pub fn set_greenfield_virtualizable_box(&mut self, box_ref: OpRef, value: Value) {
        self.virtualizable_boxes = Some(vec![box_ref]);
        self.virtualizable_live_null_slots = Some(vec![false]);
        let _ = self.try_set_opref_concrete(box_ref, value);
    }

    /// Stamp each box with the matching concrete (`*FrontendOp(pos, value)` /
    /// `Const*` inline payload). Empty `values` leaves boxes unstamped
    /// (bridge-entry rebuild / init-before-run).
    fn stamp_virtualizable_boxes(&mut self, boxes: &[OpRef], values: &[Value]) {
        if values.is_empty() {
            return;
        }
        assert_eq!(
            boxes.len(),
            values.len(),
            "stamp_virtualizable_boxes: OpRef and Value slices must match",
        );
        for (&opref, &value) in boxes.iter().zip(values) {
            let _ = self.try_set_opref_concrete(opref, value);
        }
    }

    /// Initialize standard virtualizable boxes from input args.
    /// Called at trace start when a virtualizable is registered.
    ///
    /// `input_oprefs` / `input_values` contain one (OpRef, Value) pair per
    /// static field + array element in the same flat layout as
    /// `VirtualizableInfo::get_index_in_array`. `vable_ref` / `vable_ref_value`
    /// are the OpRef and concrete of the virtualizable object (frame pointer).
    /// Boxes layout: `[field0, ..., fieldN, arr[0], ..., arr[M], vable_ref]`
    /// where `boxes[-1]` is the standard virtualizable identity.
    /// Each box carries its concrete (`InputArg*` `_res*` / `Const*` inline);
    /// empty `input_values` leaves the boxes unstamped.
    pub fn init_virtualizable_boxes(
        &mut self,
        info: &VirtualizableInfo,
        vable_ref: OpRef,
        vable_ref_value: Value,
        input_oprefs: &[OpRef],
        input_values: &[Value],
        array_lengths: &[usize],
    ) {
        let mut boxes = input_oprefs.to_vec();
        boxes.push(vable_ref); // virtualizable_boxes[-1] = vable identity
        if input_values.is_empty() {
            // Caller has no live concrete values (e.g. bridge-entry rebuild
            // helper in pyre-jit-trace::state::seed_virtualizable_boxes).
            // Leave boxes unstamped; `virtualizable_entry_at` reads the live
            // virtualizable for those slots (`virtualizable.py read_boxes`).
            self.virtualizable_live_null_slots = None;
        } else {
            assert_eq!(
                input_oprefs.len(),
                input_values.len(),
                "init_virtualizable_boxes: OpRef and Value slices must match",
            );
            let mut values = input_values.to_vec();
            values.push(vable_ref_value);
            self.virtualizable_live_null_slots = Some(vec![false; boxes.len()]);
            self.stamp_virtualizable_boxes(&boxes, &values);
        }
        self.virtualizable_boxes = Some(boxes);
        self.retain_or_store_vinfo(info);
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Keep `jitdriver_sd.virtualizable_info` as the trace's vinfo so
    /// `vinfo is fielddescr.get_vinfo()` can be object identity.
    pub fn install_virtualizable_info(
        &mut self,
        info: std::sync::Arc<crate::virtualizable::VirtualizableInfo>,
    ) {
        self.virtualizable_info = Some(info);
    }

    fn retain_or_store_vinfo(&mut self, info: &crate::virtualizable::VirtualizableInfo) {
        if self
            .virtualizable_info
            .as_ref()
            .is_some_and(|existing| std::ptr::eq(existing.as_ref(), info))
        {
            return;
        }
        self.virtualizable_info = Some(std::sync::Arc::new(info.clone()));
    }

    /// \[FR\] The current standard virtualizable's info (shape), if any.  A
    /// recursive-portal INLINE callee shares the caller's vable shape (same
    /// kernel), so it seeds its fresh vable with this same info.
    pub fn current_virtualizable_info(&self) -> Option<std::sync::Arc<VirtualizableInfo>> {
        self.virtualizable_info.clone()
    }

    /// Collect the current virtualizable boxes (for close_loop / finish).
    /// Returns `None` if no standard virtualizable is active.
    pub fn collect_virtualizable_boxes(&self) -> Option<Vec<OpRef>> {
        self.virtualizable_boxes.clone()
    }

    /// history.py `record_same_as`:
    ///
    /// ```python
    /// def record_same_as(self, box):
    ///     if box.type == 'i':
    ///         return self.record1(rop.SAME_AS_I, box, box.getint())
    ///     elif box.type == 'r':
    ///         return self.record1(rop.SAME_AS_R, box, box.getref_base())
    ///     else:
    ///         assert box.type == 'f'
    ///         return self.record1(rop.SAME_AS_F, box, box.getfloatstorage())
    /// ```
    /// `record1`'s third argument is the value, so the wrapper carries the
    /// source box's observed value; the closing JUMP's `runtime_boxes` deliver
    /// it to `_jump_to_existing_trace`'s runtime fallbacks.
    pub fn record_same_as(&mut self, opref: OpRef, tp: majit_ir::Type) -> OpRef {
        let value = self.concrete_of_opref(opref);
        self.record_op_with_value(majit_ir::OpCode::same_as_for_type(tp), &[opref], value)
    }

    /// pyjitpl.py `remove_consts_and_duplicates`:
    ///
    /// ```python
    /// for i in range(endindex):
    ///     box = boxes[i]
    ///     if isinstance(box, Const) or box in duplicates:
    ///         boxes[i] = self.history.record_same_as(box)
    ///     else:
    ///         duplicates[box] = None
    /// ```
    ///
    /// `reached_loop_header` runs this over the reds and, sharing ONE
    /// `duplicates` set, over `virtualizable_boxes[:-1]` (pyjitpl.py)
    /// — i.e. over exactly what the loop-carried list is about to become — so
    /// no entry of that list is a constant and none appears twice. Both
    /// properties are assumed downstream and neither is checked:
    /// `TreeLoop::cut_trace_from` keys its remap on the `OpRef`, so
    /// a repeat overwrites the earlier slot's mapping, and its `remap_ref`
    /// short-circuits on `!is_runtime_opref`, so a constant is never rewritten
    /// to the inputarg the LABEL declares for it.
    ///
    /// The `SameAs` variant follows the box's OWN type, never the slot's
    /// declared one: a cross-type `SameAs` absorbed by `make_equal_to` breaks
    /// the type invariant in the optimizer's replace path. An entry whose
    /// `OpRef` carries no type is left alone rather than guessed at.
    ///
    /// Upstream normalizes per `reached_loop_header` visit, so the registration
    /// and the closing JUMP are normalized by separate invocations; this is
    /// called at each of those sites rather than once for both.
    pub fn remove_consts_and_duplicates(&mut self, boxes: &mut [(OpRef, Type)]) {
        let mut duplicates: indexmap::IndexSet<OpRef, rustc_hash::FxBuildHasher> =
            indexmap::IndexSet::with_hasher(rustc_hash::FxBuildHasher);
        self.remove_consts_and_duplicates_with(boxes, &mut duplicates);
    }

    /// `pyjitpl.py MetaInterp.remove_consts_and_duplicates(boxes, endindex, duplicates)`.
    /// `reached_loop_header` shares one `duplicates` dict: reds first, then
    /// `virtualizable_boxes[:-1]`.
    pub fn remove_consts_and_duplicates_with(
        &mut self,
        boxes: &mut [(OpRef, Type)],
        duplicates: &mut indexmap::IndexSet<OpRef, rustc_hash::FxBuildHasher>,
    ) {
        for slot in boxes.iter_mut() {
            let (opref, declared) = *slot;
            if !opref.is_constant() && duplicates.insert(opref) {
                continue;
            }
            let Some(tp) = opref.ty() else {
                continue;
            };
            debug_assert!(
                tp == declared || opref.is_constant(),
                "loop-carried slot declared {declared:?} but its box is {tp:?}",
            );
            let same_as = self.record_same_as(opref, tp);
            *slot = (same_as, declared);
        }
    }

    /// [`Self::remove_consts_and_duplicates`] for the JUMP side, which carries
    /// no separate type tag — the LABEL it targets already declares the types,
    /// so each slot's own `OpRef` is the only type source and the only one
    /// upstream uses (`record_same_as` reads `box.type`).
    pub fn remove_consts_and_duplicates_untyped(&mut self, boxes: &mut [OpRef]) {
        // pyjitpl.py `remove_consts_and_duplicates` rewrites `boxes[i]`
        // in place. A side `Vec<(OpRef, Type)>` was a 128 B class on the
        // regex and/or bridge close (`start_bridge_tracing`).
        let mut duplicates: indexmap::IndexSet<OpRef, rustc_hash::FxBuildHasher> =
            indexmap::IndexSet::with_hasher(rustc_hash::FxBuildHasher);
        for slot in boxes.iter_mut() {
            let opref = *slot;
            if !opref.is_constant() && duplicates.insert(opref) {
                continue;
            }
            let Some(tp) = opref.ty() else {
                continue;
            };
            *slot = self.record_same_as(opref, tp);
        }
    }

    /// Write a normalized element block back over
    /// `virtualizable_boxes[..len-1]`.
    ///
    /// pyjitpl.py:2985-2988 normalizes the list IN PLACE and only then appends
    /// it:
    ///
    /// ```python
    /// self.remove_consts_and_duplicates(self.virtualizable_boxes,
    ///                                   len(self.virtualizable_boxes)-1,
    ///                                   duplicates)
    /// live_arg_boxes += self.virtualizable_boxes
    /// ```
    ///
    /// so the rewrite is visible to every later reader of
    /// `virtualizable_boxes`, not just to this merge point's JUMP.
    /// `collect_jump_args_with_boxes` splices a COPY of the element block into
    /// the loop-carried list, so the normalization has to be handed back
    /// explicitly for the two to stay the same list.
    ///
    /// They must: the LABEL a later cut mints declares one inputarg per
    /// element position, while the compiled entry supplies one value per
    /// element position. A repeat collapses the LABEL by one arg and every
    /// position after it is fed its predecessor's value.
    ///
    /// `elements` is `virtualizable_boxes[..len-1]` after normalization — the
    /// identity sits outside the `endindex = len - 1` window and is never
    /// rewritten.
    pub fn adopt_normalized_virtualizable_elements(&mut self, elements: &[OpRef]) {
        let Some(end) = self
            .virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.len().checked_sub(1))
        else {
            return;
        };
        assert_eq!(
            end,
            elements.len(),
            "adopt_normalized_virtualizable_elements: element block is \
             virtualizable_boxes[..len-1]",
        );
        let Some(boxes) = self.virtualizable_boxes.as_mut() else {
            return;
        };
        boxes[..end].copy_from_slice(elements);
    }

    /// [`Self::collect_virtualizable_boxes`] with each slot paired with its declared
    /// [`Type`] (`virtualizable_slot_type`); identity LAST, as always.
    ///
    /// pyjitpl.py:2981-2989 builds ONE `live_arg_boxes` and hands it to both
    /// the merge-point registration and the closing JUMP. The registration
    /// side becomes a cut trace's LABEL inputargs, and those are *typed*
    /// (`TreeLoop::cut_trace_from` → `OpRef::input_arg_typed`), so
    /// the type tag has to travel with the box for a registration to be able
    /// to reproduce the close's shape.
    pub fn collect_virtualizable_typed_boxes(&self) -> Option<Vec<(OpRef, Type)>> {
        let boxes = self.virtualizable_boxes.as_ref()?;
        Some(
            boxes
                .iter()
                .enumerate()
                .map(|(i, &opref)| {
                    // history.py Box.type on the box; virtualizable.py
                    // `static_extra_types` / `arrayitem_extra_types` on the
                    // field. wrap() mints a box of the field type, so the
                    // two agree. An empty pointer slot is ConstPtr(null)
                    // (`type == 'r'`), never an int.
                    let ty = opref.ty().or_else(|| self.virtualizable_slot_type(i));
                    let ty = match ty {
                        Some(ty) => ty,
                        None if opref.is_none() => Type::Ref,
                        None => panic!(
                            "virtualizable_boxes[{i}] has no box.type and no \
                             vinfo field type (virtualizable.py wrap / \
                             static_extra_types / arrayitem_extra_types)"
                        ),
                    };
                    (opref, ty)
                })
                .collect(),
        )
    }

    /// pyjitpl.py `reached_loop_header` `live_arg_boxes[num_green_args:]`.
    ///
    /// `live_arg_boxes = greenboxes + redboxes` then
    /// `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`.
    /// `compile_loop` slices greens off (`original_boxes[num_green_args:]`)
    /// so the LABEL/JUMP are the reds plus every virtualizable field except
    /// the trailing identity. First-visit registration and the closing JUMP
    /// both have to build this list the same way
    /// (`assert len(original_boxes) == len(live_arg_boxes)`).
    pub fn live_arg_boxes_from_reds(&self, redboxes: &[(OpRef, Type)]) -> Vec<(OpRef, Type)> {
        let mut boxes = redboxes.to_vec();
        if let Some(vable) = self.collect_virtualizable_typed_boxes() {
            let n = vable.len().saturating_sub(1);
            boxes.extend_from_slice(&vable[..n]);
        }
        boxes
    }

    /// The walk-final concrete values of the virtualizable's array elements, in
    /// the flat `[arr0_elem0.., arr1_elem0.., ..]` layout — the array portion of
    /// `virtualizable_boxes`, excluding the `num_static_extra_boxes` leading
    /// static-field slots and the trailing identity slot
    /// (`virtualizable_boxes[-1]`). Each slot's bits come from
    /// [`Self::virtualizable_entry_at`] (the box's own result, or the live
    /// virtualizable when the box carries no runtime concrete). `None` when no
    /// standard virtualizable is active. Used by the single-pass close
    /// to transfer walk-mutated loop-carried array state into native `state`
    /// before re-entering the compiled loop (`live_arg_boxes += virtualizable_boxes`).
    pub fn collect_virtualizable_element_values(&self) -> Option<Vec<i64>> {
        let boxes = self.virtualizable_boxes.as_ref()?;
        let static_count = self
            .virtualizable_info
            .as_ref()
            .map_or(0, |info| info.num_static_extra_boxes);
        let end = boxes.len().saturating_sub(1); // drop the trailing identity slot
        let start = static_count.min(end);
        Some(
            (start..end)
                .map(|i| {
                    self.virtualizable_entry_at(i)
                        .map(|(_, v)| value_to_raw_bits(v))
                        .unwrap_or(0)
                })
                .collect(),
        )
    }

    // (synchronize_virtualizable helper follows)

    /// Mirror of the host seed / `virtualizable_heap_ptr` used by `synchronize_virtualizable`.
    /// Callers set this at trace/bridge-entry so writes to
    /// `virtualizable_boxes` can propagate to the live PyFrame without
    /// routing back through MetaInterp (`virtualizable.py write_boxes` target).
    /// The object `refresh_virtualizable_shadow_from_heap` and
    /// `synchronize_virtualizable` read and write.  Diagnostic only: a caller
    /// that needs to know whether the frame it just mutated is the one the
    /// shadow tracks has no other way to ask.
    pub fn diag_virtualizable_heap_ptr(&self) -> usize {
        self.virtualizable_heap_ptr.map_or(0, |p| p as usize)
    }

    pub fn set_virtualizable_heap_ptr(&mut self, ptr: *const u8) {
        self.virtualizable_heap_ptr = if ptr.is_null() { None } else { Some(ptr) };
    }

    /// Inverse of `synchronize_virtualizable`: pull current heap virtualizable
    /// field values onto the JIT-tracked boxes.
    ///
    /// pyre-only sync hook.  RPython's metainterp IS the execution loop —
    /// every opcode flows through `_opimpl_*` which mutates
    /// `metainterp.virtualizable_boxes` in lockstep with the implicit heap
    /// write, so the boxes never drift.  Pyre's tracer dispatches some
    /// opcodes through the walker (which mirrors via
    /// `vable_setfield → synchronize_virtualizable`) and others through
    /// `execute_opcode_step` (which mutates the heap PyFrame directly
    /// via `PyFrame::push` / `PyFrame::pop` etc., bypassing the boxes).
    /// Between any pair of those dispatch paths the boxes can lag heap.
    ///
    /// Calling this at each walker step entry — *before* the walker
    /// arm body reads any vable box or `synchronize_virtualizable` writes
    /// a stale box back to heap — restores the invariant that
    /// each box's concrete equals the heap at every opcode boundary.  When
    /// dispatch unification retires `execute_opcode_step`, this hook becomes
    /// a no-op (every mutation already lands on the box) and can be deleted.
    ///
    /// A box that already carries `_res*` keeps it: that payload is the
    /// trace's observed value (`*FrontendOp(pos, value)` / `InputArg*`).
    /// Overwriting it with the live heap makes the next recorded use of the
    /// box (unbox for `i = i + 1`) see a store the walker is about to record,
    /// and apply it a second time. Unstamped boxes stay unstamped;
    /// [`Self::virtualizable_entry_at`] reads the live virtualizable for those
    /// slots. A residual that mutated the frame without going through
    /// `vable_setarrayitem` uses [`Self::reload_virtualizable_boxes_from_heap`].
    pub fn refresh_virtualizable_shadow_from_heap(&mut self) {
        self.refresh_virtualizable_from_heap(false);
    }

    /// After a residual call whose body wrote the frame without going through
    /// `vable_setarrayitem`, rebuild any stale stamped slot as a fresh
    /// `Const*` from the heap (`vable_after_residual_call` / `read_boxes`
    /// wrap). Existing boxes are never restamped with a value that is not
    /// their own result.
    pub fn reload_virtualizable_boxes_from_heap(&mut self) {
        self.refresh_virtualizable_from_heap(true);
    }

    fn refresh_virtualizable_from_heap(&mut self, overwrite_stamped: bool) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        let Some(boxes) = self.virtualizable_boxes.clone() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.clone() else {
            return;
        };
        let static_count = info.num_static_extra_boxes;
        // Last slot is the standard-vable identity (`virtualizable_boxes[-1]`).
        // `synchronize_virtualizable` already stops at `static_count + sum(lengths)`;
        // mirror that here so a short/misaligned list (only the identity slot
        // present, or fewer data slots than expected) can never overwrite it.
        let shadow_data_len = boxes.len().saturating_sub(1);
        let mut replacements = Vec::new();
        for (i, field) in info
            .static_fields
            .iter()
            .take(static_count.min(shadow_data_len))
            .enumerate()
        {
            let ty = field.field_type;
            let bits = unsafe { info.read_field(heap_ptr, i) };
            let concrete = crate::pyjitpl::heap_value_for_pub(ty, bits);
            if let Some(new_box) =
                self.refresh_vable_slot(boxes[i], ty, bits, concrete, overwrite_stamped)
            {
                replacements.push((i, new_box));
            }
        }
        let mut cursor = static_count;
        for (a_idx, &length) in lengths.iter().enumerate() {
            if a_idx >= info.array_fields.len() {
                break;
            }
            let ty = info.array_fields[a_idx].item_type;
            for item_idx in 0..length {
                if cursor >= shadow_data_len {
                    break;
                }
                let bits = unsafe { info.read_array_item(heap_ptr, a_idx, item_idx) };
                let concrete = crate::pyjitpl::heap_value_for_pub(ty, bits);
                if let Some(new_box) =
                    self.refresh_vable_slot(boxes[cursor], ty, bits, concrete, overwrite_stamped)
                {
                    replacements.push((cursor, new_box));
                }
                cursor += 1;
            }
        }
        if !replacements.is_empty()
            && let Some(boxes) = self.virtualizable_boxes.as_mut()
        {
            for (index, new_box) in replacements {
                if let Some(slot) = boxes.get_mut(index) {
                    *slot = new_box;
                }
            }
        }
    }

    /// Rebuild a slot from a heap read without stamping an existing box with
    /// a value that is not its own result (`history.py _make_op(pos, value)`).
    /// A `Const*` that disagrees is wrapped as a fresh `Const*`
    /// (`cpu.tsosvgcref_to_box` / `wrap(..., in_const_box)`). A non-const box
    /// stays as it is: unstamped slots are answered by
    /// [`Self::virtualizable_entry_at`]'s heap fallback, and a residual reload
    /// (`overwrite_stamped`) replaces a stale stamped box with a fresh
    /// `Const*` (`vable_after_residual_call` / `read_boxes`).
    fn refresh_vable_slot(
        &mut self,
        opref: OpRef,
        ty: Type,
        bits: i64,
        concrete: Value,
        overwrite_stamped: bool,
    ) -> Option<OpRef> {
        if opref.is_constant() {
            if opref.inline_const_to_value() == Some(concrete) {
                return None;
            }
            return Some(self.wrap_heap_const(ty, bits, opref));
        }
        if !overwrite_stamped {
            return None;
        }
        if self.box_carries_runtime_concrete(opref) {
            if self.box_value(opref) == Some(concrete) {
                return None;
            }
            return Some(self.wrap_heap_const(ty, bits, opref));
        }
        None
    }

    fn wrap_heap_const(&mut self, ty: Type, bits: i64, fallback: OpRef) -> OpRef {
        match ty {
            Type::Int => self.const_int(bits),
            Type::Ref => self.const_ref(bits),
            Type::Float => self.const_float(bits),
            Type::Void => fallback,
        }
    }

    /// True when `opref` already carries a runtime payload (`InputArg*` /
    /// `*FrontendOp._res*` / `Const*` inline) that is not the "no concrete"
    /// sentinel. Void and `GcRef::NO_CONCRETE` are the unstamped markers:
    /// [`Self::virtualizable_entry_at`] then reads the live virtualizable.
    pub fn box_carries_runtime_concrete(&self, opref: OpRef) -> bool {
        match self.box_value(opref) {
            None | Some(Value::Void) => false,
            Some(Value::Ref(r)) if r == majit_ir::GcRef::NO_CONCRETE => false,
            Some(_) => true,
        }
    }

    fn box_runtime_concrete(&self, opref: OpRef) -> Option<Value> {
        if !self.box_carries_runtime_concrete(opref) {
            return None;
        }
        self.box_value(opref)
    }

    /// `MetaInterp.synchronize_virtualizable()`.
    ///
    /// Writes each box's concrete (`box.getint` / `getref_base` /
    /// `getfloatstorage`) back to the live virtualizable via
    /// `write_field` / `write_array_item`. The trailing identity slot
    /// (`virtualizable_boxes[-1]`) is excluded — `write_boxes`
    /// stops at `self.num_arrays + self.static_fields.len()` and leaves the
    /// identity untouched. No-op when the heap pointer, `virtualizable_info`,
    /// or `virtualizable_boxes` is unavailable.
    pub fn synchronize_virtualizable(&self) {
        self.write_virtualizable_back(true);
    }

    /// `pyjitpl.py MetaInterp.synchronize_virtualizable_at`.
    ///
    /// Same body as [`Self::synchronize_virtualizable`], but
    /// `virtualizable.py write_box_at` writes only flat slot `index`.
    /// `pyjitpl.py MIFrame._opimpl_setfield_vable` and
    /// `MIFrame._opimpl_setarrayitem_vable` call this after storing
    /// `virtualizable_boxes[index]`.
    pub fn synchronize_virtualizable_at(&self, index: usize) {
        self.write_virtualizable_back_at(index, true);
    }

    /// The same write at a moment the carve-out does not apply:
    /// `pyjitpl.py rebuild_state_after_failure`'s closing
    /// `self.synchronize_virtualizable()`.
    ///
    /// A guard has just failed out of compiled code and nothing has run the
    /// outer executor since, so the resume stream is the only description of
    /// the virtualizable that exists and every field is this write's to make —
    /// including the arrays an executor would otherwise own. A frontend whose
    /// own guard-failure recovery already performed it does not reach here;
    /// `JitState::SYNCHRONIZES_VIRTUALIZABLE_AFTER_GUARD_FAILURE` is how it
    /// says so.
    pub fn synchronize_virtualizable_after_guard_failure(&self) {
        self.write_virtualizable_back(false);
    }

    /// The same full write ahead of a residual call that is handed the
    /// virtualizable.
    ///
    /// The outer executor is suspended for the duration of that call and the
    /// callee reads the live struct, so the shadow — which carries the values
    /// the walk has produced since the executor last wrote — is what the
    /// callee must see, arrays included.
    ///
    /// The write lands on `virtualizable_heap_ptr`, while the stores its one
    /// caller records name the identity box instead. Those two CAN name
    /// different objects — a frontend that traces against a private snapshot
    /// copy points them apart — but only a token-bearing machine seeds that
    /// way, and the caller returns above on one, so the pair is the same
    /// object on every path that reaches here.
    fn synchronize_virtualizable_before_residual_call(&self) {
        self.write_virtualizable_back(false);
    }

    /// Materialize a tokenless state-field virtualizable before a residual
    /// call that may mutate it.
    ///
    /// PyPy's translated virtualizable object carries `vable_token`; touching
    /// it from the residual helper forces the register image to the object.
    /// A `#[jit_interp(state_fields = ...)]` state is the interpreter's Rust
    /// struct itself and has no spare token field. Emit the field/array stores
    /// explicitly so the helper observes the live JIT values. The concrete
    /// tracing image is synchronized first; the recorded stores provide the
    /// same ordering in compiled code.
    pub fn materialize_tokenless_virtualizable_before_residual_call(&mut self) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        if info.has_vable_token() {
            return;
        }
        let Some(boxes) = self.virtualizable_boxes.clone() else {
            return;
        };
        let Some(vable) = boxes.last().copied() else {
            return;
        };
        let lengths = self.virtualizable_array_lengths.clone().unwrap_or_default();

        self.synchronize_virtualizable_before_residual_call();
        // Walk the fields, not the boxes: `load_fields_from_virtualizable` and
        // `reload_tokenless_virtualizable_after_residual_call` both `continue`
        // past a `Type::Void` static field, so the flat vector is shorter than
        // `static_fields` and its position is not the field index.  Enumerating
        // the boxes would hand field *i*'s descr to a later field's box, and
        // start the array section one slot per void field too late.
        let mut flat_index = 0usize;
        for (field_index, field) in info.static_fields.iter().enumerate() {
            if field.field_type == Type::Void {
                continue;
            }
            let Some(&value) = boxes.get(flat_index) else {
                return;
            };
            flat_index += 1;
            self.record_op_with_descr(
                OpCode::SetfieldGc,
                &[vable, value],
                info.static_field_descr(field_index),
            );
        }
        for (array_index, &length) in lengths.iter().enumerate() {
            let array_ref = self.record_op_with_descr(
                OpCode::GetfieldGcR,
                &[vable],
                info.array_pointer_field_descr(array_index),
            );
            // `EmbeddedArray` keeps length and items in different words of the
            // container (`virtualizable.py` `bhimpl_setarrayitem_vable_*`
            // loads the array, then indexes it). The field load is the
            // container; the items live at its data pointer.
            let array_ref = self.vable_embedded_items_base(array_ref, array_index);
            let array_descr = info.array_item_descr(array_index);
            for item_index in 0..length {
                let Some(&value) = boxes.get(flat_index) else {
                    return;
                };
                let index = self.const_int(item_index as i64);
                self.record_op_with_descr(
                    OpCode::SetarrayitemGc,
                    &[array_ref, index, value],
                    array_descr.clone(),
                );
                flat_index += 1;
            }
        }
    }

    /// Reload a tokenless state-field virtualizable after its residual helper.
    ///
    /// This is the non-aborting counterpart of
    /// `load_fields_from_virtualizable`: the heap reads are recorded after the
    /// call, so compiled code consumes the helper's runtime results rather
    /// than constants from the tracing run.
    pub fn reload_tokenless_virtualizable_after_residual_call(&mut self) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        if info.has_vable_token() {
            return;
        }
        let Some(vable) = self.standard_virtualizable_box() else {
            return;
        };
        let Some(vable_ptr) = self.standard_virtualizable_ptr() else {
            return;
        };
        // `record_op_with_descr_value` grows `Trace._ops`
        // (`opencoder.py Trace._double_ops`) and can minor-collect.
        // RPython traces the vable and each Ref field as GC locals; the
        // copied addresses here are not. Pin them for the whole rebuild
        // and re-read after the last append before publishing `values`.
        let vable_pin = majit_gc::shadow_stack::OwnerRootGuard::new(GcRef(vable_ptr));
        let lengths = self.virtualizable_array_lengths.clone().unwrap_or_default();
        let capacity = info.static_fields.len() + lengths.iter().sum::<usize>() + 1;
        let mut boxes = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity);
        let mut field_pins: Vec<Option<majit_gc::shadow_stack::OwnerRootGuard>> =
            Vec::with_capacity(capacity);

        for (field_index, field) in info.static_fields.iter().enumerate() {
            let opcode = match field.field_type {
                Type::Int => OpCode::GetfieldGcI,
                Type::Ref => OpCode::GetfieldGcR,
                Type::Float => OpCode::GetfieldGcF,
                Type::Void => continue,
            };
            let bits = unsafe { info.read_field(vable_pin.get().0 as *const u8, field_index) };
            let concrete = crate::pyjitpl::heap_value_for_pub(field.field_type, bits);
            let field_pin = match concrete {
                Value::Ref(r) if !r.is_null() => {
                    Some(majit_gc::shadow_stack::OwnerRootGuard::new(r))
                }
                _ => None,
            };
            let opref = self.record_op_with_descr_value(
                opcode,
                &[vable],
                info.static_field_descr(field_index),
                Some(concrete),
            );
            boxes.push(opref);
            values.push(concrete);
            field_pins.push(field_pin);
        }
        for (array_index, &length) in lengths.iter().enumerate() {
            let field_descr = info.array_pointer_field_descr(array_index);
            let array_ref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable], field_descr.clone());
            self.stamp_vable_array_base(
                array_ref,
                Some(Value::Ref(GcRef(vable_pin.get().0))),
                &field_descr,
            );
            let array_ref = self.vable_embedded_items_base(array_ref, array_index);
            let item_type = info.array_fields[array_index].item_type;
            let item_opcode = match item_type {
                Type::Int => OpCode::GetarrayitemGcI,
                Type::Ref => OpCode::GetarrayitemGcR,
                Type::Float => OpCode::GetarrayitemGcF,
                Type::Void => continue,
            };
            let array_descr = info.array_item_descr(array_index);
            for item_index in 0..length {
                let index = self.const_int(item_index as i64);
                let bits = unsafe {
                    info.read_array_item(vable_pin.get().0 as *const u8, array_index, item_index)
                };
                let concrete = crate::pyjitpl::heap_value_for_pub(item_type, bits);
                let item_pin = match concrete {
                    Value::Ref(r) if !r.is_null() => {
                        Some(majit_gc::shadow_stack::OwnerRootGuard::new(r))
                    }
                    _ => None,
                };
                let opref = self.record_op_with_descr_value(
                    item_opcode,
                    &[array_ref, index],
                    array_descr.clone(),
                    Some(concrete),
                );
                boxes.push(opref);
                values.push(concrete);
                field_pins.push(item_pin);
            }
        }
        debug_assert_eq!(field_pins.len(), values.len());
        for ((value, pin), opref) in values
            .iter_mut()
            .zip(field_pins.iter())
            .zip(boxes.iter().copied())
        {
            if let Some(pin) = pin {
                let live = Value::Ref(pin.get());
                *value = live;
                self.set_opref_concrete(opref, live);
            }
        }
        boxes.push(vable);
        values.push(Value::Ref(GcRef(vable_pin.get().0)));
        self.set_virtualizable_boxes_with_info(boxes, values, &info, &lengths);
    }

    /// `virtualizable.py write_boxes` over the whole box list.
    ///
    /// `skip_when_outer_owned` names the merge-point form whose write-back
    /// is not this function's to make; see the carve-out below.
    fn write_virtualizable_back(&self, skip_when_outer_owned: bool) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let Some(boxes) = self.virtualizable_boxes.as_ref() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.as_ref() else {
            return;
        };
        // When the merge point is the bare observer/replay form
        // (`jit_merge_point!()`), an outer executor (the macro-generated
        // mainloop) owns the live struct and writes it on every opcode. The
        // trace's boxes are seeded from that heap and tracked for IR
        // purposes only; flushing them back here would clobber the
        // outer executor's writes. The live struct is authoritative, so
        // skip the write-back during tracing — the resume path performs its
        // own field-aware flush on guard failure. The `; state`
        // single-executor close keeps the walk executing, so the flush is
        // required there: it is the only thing that keeps the live struct
        // and the boxes equal. `synchronize_virtualizable`
        // always writes because upstream's metainterp IS the interpreter.
        if skip_when_outer_owned && info.outer_executor_owns_state {
            return;
        }
        let static_count = info.num_static_extra_boxes;
        if boxes.len() < static_count {
            return;
        }
        let mut needed = static_count;
        for &len in lengths {
            needed = needed.saturating_add(len);
            if needed > boxes.len() {
                return;
            }
        }
        // virtualizable.py write_boxes: setattr each static field, then each
        // array item, with no intermediate collection. Each box's concrete is
        // `box.getint` / `getref_base` / `getfloatstorage`; a box that carries
        // no runtime concrete is skipped so synchronize leaves the existing
        // W_Root on the heap (lazy wrapint `NewWithVtable` before allocation).
        // Safety: `heap_ptr` comes from `virtualizable_heap_ptr`, which names
        // a frame kept alive for as long as the trace reads it.  The cell is
        // not pinned for the session — see its declaration for the writers that
        // move it — and a collection forwards the object it names
        // (`walk_virtualizable_value_refs`). Offsets come from the same
        // VirtualizableInfo used at the matching heap read.
        unsafe {
            let dst = heap_ptr as *mut u8;
            for (i, &opref) in boxes[..static_count].iter().enumerate() {
                let Some(v) = self.box_runtime_concrete(opref) else {
                    continue;
                };
                info.write_field(dst, i, value_to_raw_bits(v));
            }
            let mut cursor = static_count;
            for (array_index, &len) in lengths.iter().enumerate() {
                for item_index in 0..len {
                    if let Some(v) = self.box_runtime_concrete(boxes[cursor]) {
                        info.write_array_item(dst, array_index, item_index, value_to_raw_bits(v));
                    }
                    cursor += 1;
                }
            }
        }
    }

    /// `virtualizable.py write_box_at` — one flat slot of `write_boxes`.
    ///
    /// Static fields come first (`unroll_static_fields` order), then each
    /// array's items (`unroll_array_fields` order). `pyjitpl.py
    /// MIFrame._opimpl_setfield_vable` and
    /// `MIFrame._opimpl_setarrayitem_vable` synchronize only the slot they
    /// just stored; every other slot already equals the heap
    /// (`check_synchronized_virtualizable`).
    ///
    /// `skip_when_outer_owned` is the same carve-out as
    /// [`Self::write_virtualizable_back`].
    fn write_virtualizable_back_at(&self, index: usize, skip_when_outer_owned: bool) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let Some(boxes) = self.virtualizable_boxes.as_ref() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.as_ref() else {
            return;
        };
        if skip_when_outer_owned && info.outer_executor_owns_state {
            return;
        }
        let static_count = info.num_static_extra_boxes;
        if boxes.len() < static_count {
            return;
        }
        let mut needed = static_count;
        for &len in lengths {
            needed = needed.saturating_add(len);
            if needed > boxes.len() {
                return;
            }
        }
        assert!(
            index < needed,
            "write_virtualizable_back_at: index {index} is outside the {needed} flat slots"
        );
        let Some(v) = self.box_runtime_concrete(boxes[index]) else {
            return;
        };
        let bits = value_to_raw_bits(v);
        // virtualizable.py write_box_at: walk static fields with a running
        // index, then subtract each array's length until the index falls
        // inside one. The trailing identity slot is not a field.
        let mut i = index;
        // Safety: `heap_ptr` comes from `virtualizable_heap_ptr`, which names
        // a frame kept alive for as long as the trace reads it. Offsets come
        // from the same VirtualizableInfo used at the matching heap read.
        unsafe {
            let dst = heap_ptr as *mut u8;
            if i < static_count {
                info.write_field(dst, i, bits);
                return;
            }
            i -= static_count;
            for (array_index, &len) in lengths.iter().enumerate() {
                if i < len {
                    info.write_array_item(dst, array_index, i, bits);
                    return;
                }
                i -= len;
            }
        }
        unreachable!("write_virtualizable_back_at: index {index} fell out of the flat layout");
    }

    /// Write `value` into flat slot `index` of the live virtualizable without
    /// stamping a box (`virtualizable.py write_box_at` over a value that is
    /// not a box result). Used when SWAP rearranges unstamped slots: the
    /// boxes stay unstamped and the virtualizable holds the swapped
    /// contents, so a later `read_boxes` / `virtualizable_entry_at` heap
    /// fallback sees the new arrangement.
    pub fn write_virtualizable_heap_value_at(&self, index: usize, value: Value) {
        self.write_virtualizable_heap_bits_at(index, value_to_raw_bits(value), true);
    }

    fn write_virtualizable_heap_bits_at(
        &self,
        index: usize,
        bits: i64,
        skip_when_outer_owned: bool,
    ) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.as_ref() else {
            return;
        };
        if skip_when_outer_owned && info.outer_executor_owns_state {
            return;
        }
        let static_count = info.num_static_extra_boxes;
        let mut needed = static_count;
        for &len in lengths {
            needed = needed.saturating_add(len);
        }
        if index >= needed {
            return;
        }
        let mut i = index;
        unsafe {
            let dst = heap_ptr as *mut u8;
            if i < static_count {
                info.write_field(dst, i, bits);
                return;
            }
            i -= static_count;
            for (array_index, &len) in lengths.iter().enumerate() {
                if i < len {
                    info.write_array_item(dst, array_index, i, bits);
                    return;
                }
                i -= len;
            }
        }
    }

    /// pyjitpl.py `check_synchronized_virtualizable()`, whose body is
    /// `virtualizable.py check_boxes`.
    ///
    /// ```text
    /// def check_synchronized_virtualizable(self):
    ///     if not we_are_translated():
    ///         vinfo = self.jitdriver_sd.virtualizable_info
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.check_boxes(virtualizable, self.virtualizable_boxes)
    /// ```
    ///
    /// The shadow is only ever made coherent at the point of a write
    /// (`MIFrame._opimpl_setfield_vable` sets `virtualizable_boxes[index]` and
    /// then calls `synchronize_virtualizable_at`), so a divergence here is a
    /// missing write, not something a reader may repair. `not we_are_translated()`
    /// gates it to the untranslated metainterp; `debug_assertions` is the
    /// same gate for a Rust port, which keeps it on under `cargo test` and
    /// out of the released JIT.
    ///
    /// The trailing identity slot is excluded, matching `check_boxes`'
    /// closing `assert len(boxes) == i + 1`.
    pub fn check_synchronized_virtualizable(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let (Some(heap_ptr), Some(info), Some(boxes), Some(lengths)) = (
            self.virtualizable_heap_ptr,
            self.virtualizable_info.as_ref(),
            self.virtualizable_boxes.as_ref(),
            self.virtualizable_array_lengths.as_ref(),
        ) else {
            return;
        };
        // Observer/replay merge points leave an outer executor owning the
        // live struct, so the boxes are deliberately not kept equal to it
        // — the same carve-out `synchronize_virtualizable` makes before
        // writing back.
        if info.outer_executor_owns_state {
            return;
        }
        let static_count = info.num_static_extra_boxes;
        let shadow_data_len = boxes.len().saturating_sub(1);
        if shadow_data_len < static_count {
            return;
        }
        for (i, field) in info.static_fields.iter().take(static_count).enumerate() {
            let Some(value) = self.box_runtime_concrete(boxes[i]) else {
                continue;
            };
            let ty = field.field_type;
            let bits = unsafe { info.read_field(heap_ptr, i) };
            let heap = crate::pyjitpl::heap_value_for_pub(ty, bits);
            debug_assert_eq!(
                value, heap,
                "virtualizable static field {} ({:?}) diverged from the box: \
                 a vable write did not update virtualizable_boxes",
                i, field.name,
            );
        }
        let mut cursor = static_count;
        for (a_idx, &length) in lengths.iter().enumerate() {
            if a_idx >= info.array_fields.len() {
                break;
            }
            let ty = info.array_fields[a_idx].item_type;
            for item_idx in 0..length {
                if cursor >= shadow_data_len {
                    return;
                }
                let Some(value) = self.box_runtime_concrete(boxes[cursor]) else {
                    cursor += 1;
                    continue;
                };
                let bits = unsafe { info.read_array_item(heap_ptr, a_idx, item_idx) };
                let heap = crate::pyjitpl::heap_value_for_pub(ty, bits);
                debug_assert_eq!(
                    value, heap,
                    "virtualizable array {a_idx} item {item_idx} diverged from the \
                     box: a vable write did not update virtualizable_boxes",
                );
                cursor += 1;
            }
        }
    }

    /// Read a standard virtualizable box by flat index.
    ///
    /// The last slot is the standard virtualizable identity itself
    /// (`virtualizable_boxes[-1]` in RPython terms).
    pub fn virtualizable_box_at(&self, index: usize) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.get(index).copied())
    }

    /// The vable identity OpRef — `virtualizable_boxes[-1]`, seeded ONCE from
    /// the portal/owner frame in `init_virtualizable_boxes`. The snapshot's
    /// vable section identity (`_list_of_boxes_virtualizable`'s front pointer)
    /// must be this owner frame, never the current (possibly inlined-callee)
    /// frame: the decoder's `get_total_size(virtualizable)` reads the heap
    /// array length off this pointer and asserts it equals the owner-sourced
    /// field count. Returns `None` when no standard virtualizable is seeded
    /// (test fixtures), so callers fall back to the current frame.
    pub fn virtualizable_owner_identity(&self) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied())
    }

    /// Read a standard virtualizable slot as (OpRef, concrete Value) —
    /// `virtualizable_boxes[index]`: a Box carries both the traced
    /// reference and its concrete value. Callers that need to seed a register
    /// with both halves of the Box (e.g. `BC_GETARRAYITEM_VABLE_R` →
    /// `set_ref_reg`) MUST use this instead of `virtualizable_box_at`.
    ///
    /// A box's value is only ever its own result (`history.py _make_op`,
    /// `executor.py execute`). When the box carries no runtime concrete
    /// (lazy wrapint `NewWithVtable` before allocation), the current
    /// concrete is what the live virtualizable holds for that slot —
    /// `virtualizable.py read_boxes` via `read_field` / `read_array_item`.
    pub fn virtualizable_entry_at(&self, index: usize) -> Option<(OpRef, Value)> {
        let opref = self.virtualizable_box_at(index)?;
        if let Some(value) = self.box_runtime_concrete(opref) {
            return Some((opref, value));
        }
        let value = self.virtualizable_heap_value_at(index)?;
        Some((opref, value))
    }

    /// `_opimpl_getfield_vable` / `_opimpl_getarrayitem_vable`: the box in
    /// `virtualizable_boxes[index]`, and that box's own result.
    ///
    /// A recorded op that carries no runtime concrete (lazy wrapint
    /// `NewWithVtable` before allocation) must not claim the live
    /// virtualizable's W_Root as `_make_op`'s result. An `InputArg` the
    /// trace never replaced still reads the slot from the virtualizable
    /// (`read_boxes`).
    fn vable_box_result_at(&self, index: usize) -> Option<(OpRef, Option<Value>)> {
        let opref = self.virtualizable_box_at(index)?;
        if let Some(value) = self.box_runtime_concrete(opref) {
            return Some((opref, Some(value)));
        }
        if opref.is_input_arg() {
            return Some((
                opref,
                self.virtualizable_heap_value_at(index)
                    .and_then(concrete_shadow_value),
            ));
        }
        Some((opref, None))
    }

    /// Slot `index` of the live virtualizable, `virtualizable.py read_boxes`
    /// layout: static fields, then each array's items, then the identity.
    fn virtualizable_heap_value_at(&self, index: usize) -> Option<Value> {
        let heap_ptr = self.virtualizable_heap_ptr?;
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_deref().unwrap_or(&[]);
        let static_count = info.num_static_extra_boxes;
        if index < static_count {
            let ty = info.static_fields.get(index)?.field_type;
            let bits = unsafe { info.read_field(heap_ptr, index) };
            return Some(crate::pyjitpl::heap_value_for_pub(ty, bits));
        }
        let mut remaining = index - static_count;
        for (a_idx, &length) in lengths.iter().enumerate() {
            if a_idx >= info.array_fields.len() {
                break;
            }
            if remaining < length {
                let ty = info.array_fields[a_idx].item_type;
                let bits = unsafe { info.read_array_item(heap_ptr, a_idx, remaining) };
                return Some(crate::pyjitpl::heap_value_for_pub(ty, bits));
            }
            remaining -= length;
        }
        let total_array: usize = lengths.iter().sum();
        if index == static_count + total_array {
            return Some(Value::Ref(majit_ir::GcRef(heap_ptr as usize)));
        }
        None
    }

    /// Declared majit_ir::Type for a flat virtualizable slot.
    ///
    /// Mirrors the layout used by `initialize_virtualizable`: the first
    /// `num_static_extra_boxes` slots take their types from
    /// `VirtualizableInfo.static_fields[i].field_type`, subsequent array
    /// slots take `array_fields[a].item_type`, and the trailing identity
    /// slot (`virtualizable_boxes[-1]`) is always `Ref`.  Returns `None`
    /// when no VirtualizableInfo is registered or the index falls outside
    /// the active layout.
    pub fn virtualizable_slot_type(&self, flat_idx: usize) -> Option<Type> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_deref().unwrap_or(&[]);
        let total_array: usize = lengths.iter().sum();
        let static_count = info.num_static_extra_boxes;
        if flat_idx < static_count {
            return Some(info.static_fields[flat_idx].field_type);
        }
        let array_local_idx = flat_idx - static_count;
        if array_local_idx < total_array {
            let mut remaining = array_local_idx;
            for (a, &len) in lengths.iter().enumerate() {
                if remaining < len {
                    return Some(info.array_fields[a].item_type);
                }
                remaining -= len;
            }
        }
        if flat_idx == static_count + total_array {
            // virtualizable_boxes[-1] — the identity slot.
            return Some(Type::Ref);
        }
        None
    }

    /// Update a standard virtualizable box (OpRef) by flat index.
    ///
    /// Used by SameAs dedup / `replace_box` walks — SSA-rename operations that
    /// do NOT change the concrete value carried by the slot. For updates that
    /// also change concrete (vable set{field,arrayitem}), use
    /// `set_virtualizable_entry_at`.
    ///
    /// A box's value is only ever its own result (`history.py _make_op`,
    /// `executor.py execute`). The new box is stored as-is: if it carries no
    /// runtime concrete, [`Self::virtualizable_entry_at`] reads the live
    /// virtualizable for that slot, which `synchronize_virtualizable` left
    /// untouched.
    pub fn set_virtualizable_box_at(&mut self, index: usize, value: OpRef) -> bool {
        let Some(boxes) = self.virtualizable_boxes.as_mut() else {
            return false;
        };
        let Some(slot) = boxes.get_mut(index) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Update a standard virtualizable slot: store `valuebox` and stamp it
    /// with `value` (`*FrontendOp(pos, value)` / `Const*` inline).
    ///
    /// `_opimpl_setarrayitem_vable`:
    ///
    /// ```text
    ///     self.metainterp.virtualizable_boxes[index] = valuebox
    ///     self.metainterp.synchronize_virtualizable_at(index)
    /// ```
    ///
    /// Callers must ensure `value.get_type()` matches the slot's declared type
    /// (`virtualizable_slot_type(index)`); the source emits `NEW_W_INT` /
    /// `NEW_W_FLOAT` before any STORE into a Ref-typed `locals_cells_stack_w`
    /// slot (`list[W_Object]`). Until the codewriter mirrors that boxing at
    /// STORE_FAST → vable, a pyre-unboxed `Value::Int`/`Value::Float` in a Ref
    /// slot decodes 0 via `value_as_ref_bits`.
    pub fn set_virtualizable_entry_at(&mut self, index: usize, opref: OpRef, value: Value) {
        // The precondition above, checked rather than only stated.  A
        // `Value::Int` in a Ref slot is not a wrong number — it is a pointer
        // `value_as_ref_bits` decodes as 0, so a later
        // `BC_GETARRAYITEM_VABLE_R` reads NULL out of a slot that holds a live
        // object.  `Value::Void` is the absence of a live concrete and is
        // legal in every slot; a slot whose type is not declared (no
        // `virtualizable_info`, or an index past the layout) yields `None`
        // and is left to the range assert below.
        debug_assert!(
            matches!(value, Value::Void)
                || self
                    .virtualizable_slot_type(index)
                    .is_none_or(|declared| declared == value.get_type()),
            "set_virtualizable_entry_at: slot {index} is declared {:?} but the caller wrote a \
             {:?}; a mismatched Ref slot decodes to NULL through `value_as_ref_bits`",
            self.virtualizable_slot_type(index),
            value.get_type(),
        );
        let boxes = self
            .virtualizable_boxes
            .as_mut()
            .expect("set_virtualizable_entry_at: virtualizable_boxes missing");
        // `boxes.len() - 1` is the virtualizable identity
        // (`virtualizable_boxes[-1]`, appended once by
        // `init_virtualizable_boxes`), not a state slot: it is the box every
        // `_nonstandard_virtualizable` check compares against, so a store
        // landing on it does not corrupt one value, it renames the standard
        // virtualizable. Nothing may write it through this entry point.
        assert!(
            index + 1 < boxes.len(),
            "set_virtualizable_entry_at: index {index} is not a state slot; {} slots carry {} \
             vable entries plus the virtualizable identity at {}",
            boxes.len(),
            boxes.len() - 1,
            boxes.len() - 1,
        );
        boxes[index] = opref;
        if let Some(live_null_slots) = self.virtualizable_live_null_slots.as_mut() {
            live_null_slots[index] = false;
        }
        let _ = self.try_set_opref_concrete(opref, value);
    }

    /// Put `opref`/`value` into flat slot `index` and hand back what it held.
    ///
    /// This is the save/restore half of [`VableEntryWrite`], not a store: it
    /// deliberately leaves `virtualizable_live_null_slots` alone, where
    /// [`Self::set_virtualizable_entry_at`] clears it and
    /// `vable_setarrayitem_indexed`'s `live_null_push` arm sets it right after.
    /// Restoring through the store would drop that flag.
    ///
    /// `None` when no shadow is active or `index` is out of range — the same
    /// condition under which [`Self::virtualizable_entry_at`] reads `None`.
    pub fn swap_virtualizable_entry(
        &mut self,
        index: usize,
        opref: OpRef,
        value: Value,
    ) -> Option<(OpRef, Value)> {
        let prev = self.virtualizable_entry_at(index)?;
        let boxes = self.virtualizable_boxes.as_mut()?;
        *boxes.get_mut(index)? = opref;
        let _ = self.try_set_opref_concrete(opref, value);
        Some(prev)
    }

    /// Mark flat slot `index` as holding a live NULL Ref rather than an
    /// unwritten word. A bridge rebuilds its shadow from resume data, where
    /// `resume.py` `decode_box` turns a `NULLREF` tagged slot into
    /// `CONST_NULL`: a real box of the restored frame, the same as a NULL the
    /// trace itself stored.
    pub fn mark_virtualizable_slot_live_null(&mut self, index: usize) {
        if let Some(slot) = self
            .virtualizable_live_null_slots
            .as_mut()
            .and_then(|slots| slots.get_mut(index))
        {
            *slot = true;
        }
    }

    /// Whether the last executed store into flat slot `index` wrote a live NULL Ref.
    pub fn virtualizable_slot_stored_live_null(&self, index: usize) -> bool {
        self.virtualizable_live_null_slots
            .as_ref()
            .is_some_and(|slots| slots.get(index).copied().unwrap_or(false))
    }

    /// Return the standard virtualizable identity (`virtualizable_boxes[-1]`).
    pub fn standard_virtualizable_box(&self) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied())
    }

    /// `vinfo.unwrap_virtualizable_box(virtualizable_boxes[-1])` — the concrete
    /// object the identity box carries, as a raw address.
    ///
    /// This is the frame the shadow was expanded against, which is not the
    /// same as [`Self::virtualizable_heap_ptr`]: that one is the
    /// synchronization target, and a root portal seed deliberately points it
    /// at the `snapshot_for_tracing` copy while baking the identity against
    /// the live frame the compiled loop runs on.  Readers that need "which
    /// object do these boxes describe" — `compile.py:510` — want the identity.
    pub fn standard_virtualizable_ptr(&self) -> Option<usize> {
        match self.standard_virtualizable_concrete()? {
            Value::Ref(gcref) if gcref.as_usize() != 0 => Some(gcref.as_usize()),
            _ => None,
        }
    }

    /// Length of the symbolic virtualizable shadow, or `None` when no
    /// virtualizable is bound.
    ///
    /// NOT probe-only, whatever an older revision of this comment said. Three
    /// callers, all on correctness paths, none diagnostic:
    ///
    /// * `pyre-jit-trace/src/trace_opcode.rs` — bounds check whose failure calls
    ///   `request_trace_abort()`, so a trace resolves through the interpreter
    ///   instead of `set_virtualizable_entry_at` panicking.
    /// * `pyre-jit-trace/src/jitcode_dispatch/mod.rs` — `append_virtualizable_boxes`,
    ///   the shape fix that makes the merge-point `live_arg_boxes` match the
    ///   JUMP `close_loop_args_at` records.
    /// * the same file's register-bank candidate fill, which needs the length to
    ///   decide what liveness left unfilled.
    ///
    /// The `MAJIT_PROBE_BRIDGE` logging this once served is gone — no such gate
    /// is read anywhere. Deleting this accessor as dead probe scaffolding would
    /// take the abort guard with it.
    pub fn virtualizable_boxes_len(&self) -> Option<usize> {
        self.virtualizable_boxes.as_ref().map(|boxes| boxes.len())
    }

    /// `virtualizable.py VirtualizableInfo.read_boxes` layout: every static
    /// field and every item of every array is a box, identity last. A short
    /// shadow (init used a live-prefix fallback, or a vable array grew) is
    /// padded with `OpRef::NONE` holes before the identity so the hole scan
    /// below sees the missing tail. Live holes record `GETFIELD_GC` of the
    /// array pointer then `GETARRAYITEM_GC` per item
    /// (`fill_live_virtualizable_holes_from_heap`); dead-tail holes become
    /// typed null. `pyjitpl.py MetaInterp.reached_loop_header` can then
    /// `+= virtualizable_boxes; pop()`.
    ///
    /// `OpRef::NONE` is not always a heap `None`. `read_boxes` wraps the
    /// actual `lst[i]`; that value is null only where the interpreter stored
    /// null. `dead_array_tail_from` is the first array index the caller knows
    /// holds null; holes at that index and beyond become typed null. A NONE
    /// hole in a live slot is read from the heap with `GETARRAYITEM_GC` (the
    /// same recording `extend_vable_array_from_frame` uses for a missing array
    /// item). `None` means every hole is live: `read_boxes` wrapping `lst[i]`.
    pub fn fill_virtualizable_boxes_to_declared_layout(
        &mut self,
        dead_array_tail_from: Option<usize>,
    ) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths().map(|l| l.to_vec()) else {
            return;
        };
        let nstatic = info.num_static_extra_boxes;
        let narray: usize = lengths.iter().copied().sum();
        let declared = nstatic + narray;
        let new_len = {
            let Some(boxes) = self.virtualizable_boxes.as_mut() else {
                return;
            };
            if boxes.is_empty() {
                None
            } else {
                let data_len = boxes.len() - 1;
                if data_len < declared {
                    let identity = boxes.pop().unwrap();
                    boxes.resize(declared, OpRef::NONE);
                    boxes.push(identity);
                    Some(boxes.len())
                } else {
                    None
                }
            }
        };
        if let Some(n) = new_len
            && let Some(live_null_slots) = self.virtualizable_live_null_slots.as_mut()
        {
            live_null_slots.resize(n, false);
        }
        let hole_indices: Vec<usize> = {
            let Some(boxes) = self.virtualizable_boxes.as_ref() else {
                return;
            };
            let end = boxes.len().saturating_sub(1);
            boxes[..end]
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.is_none())
                .map(|(i, _)| i)
                .collect()
        };
        if hole_indices.is_empty() {
            return;
        }
        let mut dead_tail = Vec::new();
        let mut live_holes = Vec::new();
        for idx in hole_indices {
            if idx >= nstatic {
                let array_idx = idx - nstatic;
                if dead_array_tail_from.is_some_and(|from| array_idx >= from) {
                    dead_tail.push(idx);
                    continue;
                }
            }
            live_holes.push(idx);
        }
        for idx in dead_tail {
            let ty = self.virtualizable_slot_type(idx).unwrap_or(Type::Ref);
            let null = self.typed_null_box(ty);
            if let Some(boxes) = self.virtualizable_boxes.as_mut()
                && let Some(slot) = boxes.get_mut(idx)
            {
                *slot = null;
            }
        }
        if !live_holes.is_empty() {
            let dest_ptr = self
                .standard_virtualizable_ptr()
                .or_else(|| self.virtualizable_heap_ptr.map(|p| p as usize))
                .unwrap_or(0);
            self.fill_live_virtualizable_holes_from_heap(&info, dest_ptr, &lengths, &live_holes);
        }
    }

    fn typed_null_box(&mut self, ty: Type) -> OpRef {
        match ty {
            Type::Ref => self.const_null(),
            Type::Int => self.const_int(0),
            Type::Float => self.const_float(0),
            Type::Void => panic!("typed null for a Void virtualizable slot"),
        }
    }

    /// Record `GETFIELD_GC` / `GETARRAYITEM_GC` for each live `OpRef::NONE`
    /// hole, matching `virtualizable.py VirtualizableInfo.read_boxes`
    /// wrapping `getattr` / `lst[i]`.
    fn fill_live_virtualizable_holes_from_heap(
        &mut self,
        info: &VirtualizableInfo,
        dest_ptr: usize,
        lengths: &[usize],
        hole_indices: &[usize],
    ) {
        let nstatic = info.num_static_extra_boxes;
        let Some(vbox) = self.standard_virtualizable_box() else {
            panic!(
                "virtualizable_boxes has a live OpRef::NONE hole; \
                 virtualizable.py VirtualizableInfo.read_boxes wraps the heap \
                 value (getattr / lst[i]), which is null only where the \
                 interpreter stored null"
            );
        };
        let static_descrs = info.static_field_descrs().to_vec();
        let mut array_ops: Vec<Option<OpRef>> = vec![None; info.array_fields.len()];
        let mut replacements = Vec::with_capacity(hole_indices.len());
        for &idx in hole_indices {
            if idx < nstatic {
                let field_type = info.static_fields[idx].field_type;
                let opcode = match field_type {
                    Type::Int => OpCode::GetfieldGcI,
                    Type::Ref => OpCode::GetfieldGcR,
                    Type::Float => OpCode::GetfieldGcF,
                    Type::Void => panic!(
                        "virtualizable_boxes live hole at static field {idx} has Void type; \
                         virtualizable.py VirtualizableInfo.read_boxes"
                    ),
                };
                let op = self.record_getfield_stamped(
                    opcode,
                    vbox,
                    dest_ptr,
                    static_descrs[idx].clone(),
                    field_type,
                );
                replacements.push((idx, op));
                continue;
            }
            let mut remaining = idx - nstatic;
            let mut ai = 0;
            while ai < lengths.len() && remaining >= lengths[ai] {
                remaining -= lengths[ai];
                ai += 1;
            }
            if ai >= info.array_fields.len() {
                panic!(
                    "virtualizable_boxes live hole at flat index {idx} is outside \
                     the declared layout; virtualizable.py VirtualizableInfo.read_boxes \
                     fills every static field and every array item"
                );
            }
            if array_ops[ai].is_none() {
                let field_descr = info.array_pointer_field_descr(ai);
                let array_op = self.record_getfield_stamped(
                    OpCode::GetfieldGcR,
                    vbox,
                    dest_ptr,
                    field_descr,
                    Type::Ref,
                );
                array_ops[ai] = Some(array_op);
            }
            let array_op = array_ops[ai].unwrap();
            let item_type = info.array_fields[ai].item_type;
            let item_opcode = match item_type {
                Type::Int => OpCode::GetarrayitemGcI,
                Type::Ref => OpCode::GetarrayitemGcR,
                Type::Float => OpCode::GetarrayitemGcF,
                Type::Void => panic!(
                    "virtualizable_boxes live hole at array {ai}[{remaining}] has Void \
                     item_type; virtualizable.py VirtualizableInfo.read_boxes"
                ),
            };
            let array_descr = info.array_item_descr(ai);
            let op = self.record_getarrayitem_stamped(
                item_opcode,
                array_op,
                remaining as i64,
                array_descr,
                item_type,
            );
            replacements.push((idx, op));
        }
        if let Some(boxes) = self.virtualizable_boxes.as_mut() {
            for (idx, op) in replacements {
                boxes[idx] = op;
            }
        }
    }

    /// `virtualizable_boxes[:-1]` for `pyjitpl.py MetaInterp.reached_loop_header`.
    ///
    /// `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`.
    /// Every declared static field and array item is a box
    /// (`virtualizable.py VirtualizableInfo.read_boxes`); a missing slot is
    /// a producer bug.
    pub fn virtualizable_data_boxes(&self) -> Vec<OpRef> {
        let Some(boxes) = self.virtualizable_boxes.as_ref() else {
            return Vec::new();
        };
        let nstatic = self
            .virtualizable_info
            .as_ref()
            .map_or(0, |info| info.num_static_extra_boxes);
        let narray = self
            .virtualizable_array_lengths
            .as_ref()
            .map_or(0, |lengths| lengths.iter().copied().sum());
        let declared = nstatic + narray;
        let data_len = boxes.len().saturating_sub(1);
        // A short vec parks the identity at `boxes[data_len]`. Reading that
        // index as a data slot would hide the hole and emit a JUMP one box
        // shorter than the LABEL. Panic instead, as
        // `reached_loop_header` has no skip.
        if declared > data_len {
            panic!(
                "virtualizable_boxes is missing {} slot(s) at reached_loop_header \
                 (declared {declared} data boxes, shadow has {data_len}); \
                 pyjitpl.py MetaInterp.reached_loop_header does \
                 `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()` \
                 and virtualizable.py VirtualizableInfo.read_boxes fills every slot",
                declared - data_len,
            );
        }
        boxes[..data_len].to_vec()
    }

    /// `opencoder.py create_top_snapshot` parity for callers that
    /// need to feed `vable_boxes` / `vref_boxes` into
    /// `capture_snapshot_for_last_guard_with_vable_vref`.  Returns the
    /// pre-shaped `(vable_boxes, vref_boxes)` ready to attach to a top
    /// snapshot — identity-front reorder for vable, verbatim opref order
    /// for vref.  Empty vectors when neither a virtualizable nor any
    /// virtualref is live (matches RPython's `_list_of_boxes_virtualizable`
    /// / `_list_of_boxes` returning a 0-length array).
    /// Walker precondition for [`Self::build_snapshot_vable_vref_boxes`]:
    /// every virtualizable box (including the identity at `[-1]`) must carry
    /// `OpRef::ty()` — the invariant `crate::pyjitpl::build_vable_snapshot_boxes`
    /// enforces by panicking.  A deeper inlined / recursive frame can leave
    /// the identity box untyped, so the full-body walker calls this before
    /// recording a guard snapshot and aborts the trace into the trait
    /// fallback instead of tripping the panic.
    pub fn vable_snapshot_buildable(&self) -> bool {
        let vable_slice: &[OpRef] = self.virtualizable_boxes.as_deref().unwrap_or(&[]);
        vable_slice.iter().all(|op| op.ty().is_some())
    }

    pub fn build_snapshot_vable_vref_boxes(
        &self,
    ) -> (
        Vec<crate::recorder::SnapshotTagged>,
        Vec<crate::recorder::SnapshotTagged>,
    ) {
        // `pyjitpl.py capture_resumedata` passes `self.virtualizable_boxes`
        // when the jitdriver has a virtualizable (or greenfield). The list
        // is identity-appended by `initialize_virtualizable`;
        // `_list_of_boxes_virtualizable` encodes an empty array only when
        // that list is absent.
        let vable_slice: &[OpRef] = self.virtualizable_boxes.as_deref().unwrap_or(&[]);
        let vable_boxes = crate::pyjitpl::build_vable_snapshot_boxes(vable_slice);
        let vref_boxes = crate::pyjitpl::build_vref_snapshot_boxes(&self.virtualref_boxes);
        (vable_boxes, vref_boxes)
    }

    /// Concrete of the standard virtualizable — `virtualizable_boxes[-1].getref_base()`.
    /// Used by `begin_nonstandard_virtualizable` Step 4 to realize the runtime
    /// `isstandard = concrete_eq(box, standard_box)` compare that
    /// `_nonstandard_virtualizable` performs via `rop.PTR_EQ` +
    /// `implement_guard_value`.
    pub fn standard_virtualizable_concrete(&self) -> Option<Value> {
        self.standard_virtualizable_box()
            .and_then(|opref| self.box_value(opref))
    }

    /// Trace every concrete Ref carried by `virtualizable_boxes`.
    ///
    /// Each box is an `InputArg*` / `*FrontendOp` / `Const*` that carries its
    /// own value. `InputArg*` and `*FrontendOp` refs are forwarded by the
    /// recorder walk (`walk_active_trace_refs` visits `inputargs` and
    /// `value_slots`); this walk traces `ConstPtr` indexes the vable box
    /// list still holds, then forwards `virtualizable_heap_ptr`. The trailing
    /// identity is `virtualizable_boxes[-1]`; a bridge must keep that rebuilt
    /// frame identity live instead of falling back to an older cached
    /// portal-frame pointer.
    pub(crate) fn walk_virtualizable_value_refs(
        &mut self,
        mut visitor: impl FnMut(&mut majit_ir::GcRef),
    ) {
        let identity_before = match self.standard_virtualizable_concrete() {
            Some(Value::Ref(identity)) => Some(identity.as_usize()),
            _ => None,
        };
        if let Some(boxes) = self.virtualizable_boxes.as_ref() {
            for slot in boxes {
                slot.trace_const_ptr(&mut visitor);
            }
        }
        // The cell names either the identity or a different object: a frontend
        // that traces against a GC-owned snapshot copy seeds it with the
        // snapshot while the identity box names the live frame.  Forward the
        // object the cell names; re-deriving it from the identity would move
        // the synchronization target onto the live frame at whichever
        // collection happens to fire.
        let cell = self.virtualizable_heap_ptr;
        match cell {
            Some(ptr) if !ptr.is_null() && Some(ptr as usize) != identity_before => {
                let mut target = majit_ir::GcRef(ptr as usize);
                visitor(&mut target);
                self.virtualizable_heap_ptr = Some(target.as_usize() as *const u8);
            }
            _ => {
                if let Some(Value::Ref(identity)) = self.standard_virtualizable_concrete() {
                    self.virtualizable_heap_ptr = if identity.is_null() {
                        None
                    } else {
                        Some(identity.as_usize() as *const u8)
                    };
                }
            }
        }
    }

    /// Recover a concrete Ref value from trace-local state.
    ///
    /// [`TraceCtx::concrete_of_opref`] handles constants, the standard
    /// virtualizable, and operations whose result was stamped while recording.
    /// For an unstamped `GetfieldGcR`, this method recursively resolves the
    /// object and rereads the described field. That case occurs when an inlined
    /// sub-walk records the load before its outer-frame input acquires a concrete
    /// resume value. `depth` bounds recursive field chains; unsupported
    /// producers, null objects, and exhausted depth return `None`.
    ///
    /// `GcRef::NO_CONCRETE` is the stamp for "no runtime value is known for
    /// this box", not a value: `heapcache_ops` writes it over a load the walk
    /// could not replay. Returning it hands the sentinel address on as if it
    /// were an object — the caller-image ref fill in
    /// `jitcode_dispatch/resume_snapshot.rs` and the innermost-frame fill in
    /// `jitcode_dispatch/residual_call.rs` both write whatever `Value::Ref`
    /// arrives here into a blackhole frame's ref bank, and resuming through
    /// that frame dereferences it. Answer unresolved instead, so those sites
    /// decline the image the way they already do for a color with no box.
    /// The `GetfieldGcR` that produced `opref`, when that box is one.
    ///
    /// Byte-mode traces keep the producer in a frontend slot;
    /// `get_op_by_raw_pos` answers after the op has been materialized.
    pub fn ref_getfield_gc_r(&self, opref: OpRef) -> Option<(DescrRef, OpRef)> {
        if let Some(pair) = self.recorder.getfield_gc_r_at(opref.raw()) {
            return Some(pair);
        }
        let op = self.recorder.get_op_by_raw_pos(opref.raw())?;
        if op.opcode != OpCode::GetfieldGcR {
            return None;
        }
        let descr = op.descr.borrow().clone()?;
        let obj = op.args_slice().first()?.to_opref();
        Some((descr, obj))
    }

    pub fn recover_ref_value(&self, opref: OpRef, depth: u32) -> Option<Value> {
        if let Some(v) = self.concrete_of_opref(opref) {
            if matches!(v, Value::Ref(r) if r == majit_ir::GcRef::NO_CONCRETE) {
                return None;
            }
            return Some(v);
        }
        if depth == 0 {
            return None;
        }
        let (descr, obj) = if let Some(pair) = self.recorder.getfield_gc_r_at(opref.raw()) {
            pair
        } else {
            let op = self.recorder.get_op_by_raw_pos(opref.raw())?;
            if !matches!(op.opcode, OpCode::GetfieldGcR) {
                return None;
            }
            let descr = op.descr.borrow().clone()?;
            let obj = op.args_slice().first()?.to_opref();
            (descr, obj)
        };
        let Value::Ref(obj_ref) = self.recover_ref_value(obj, depth - 1)? else {
            return None;
        };
        // The receiver is about to be loaded through, so it is judged by
        // [`live_gc_ptr`] — the one place that names all three addresses no
        // object model owns — rather than by a local copy of two of them.
        let obj_ptr = live_gc_ptr(obj_ref)?;
        self.field_sanity_load(obj_ptr, &descr, Type::Ref)
    }

    pub fn concrete_of_opref(&self, opref: OpRef) -> Option<Value> {
        if opref.is_constant() {
            // history.py:220/261/307 box.type parity: the OpRef variant
            // carries the typed `Value` inline — the variant tag carries
            // the `Box.type` intrinsically, so no separate type lookup
            // is required.
            if let Some(value) = opref.inline_const_to_value() {
                return Some(value);
            }
        }
        self.lookup_opref_concrete(opref)
    }

    /// Whether standard virtualizable boxes are active.
    pub fn has_virtualizable_boxes(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Whether `init_virtualizable_boxes` / `set_virtualizable_boxes_with_info`
    /// stamped field concretes (`live_null_slots` is allocated in that arm).
    /// Empty `input_values` leaves InputArg/`*FrontendOp` boxes unstamped and
    /// this is false, even when a `Const*` identity already carries a payload.
    pub fn has_virtualizable_shadow(&self) -> bool {
        self.virtualizable_live_null_slots.is_some()
    }

    /// Drop the tracing-time virtualizable_boxes mirror.
    ///
    /// **Dormant — no caller.** The bridge-entry protocol below describes what
    /// this is *for*, not what currently happens; nothing invokes it, so no
    /// bridge entry clears the mirror today. The upstream counterpart it is
    /// modelled on is real, so the disposition is to wire it rather than delete
    /// it — see `rpython/jit/metainterp/pyjitpl.py:3400-3430`.
    ///
    /// Intended use is bridge entry: `init_symbolic` seeds the cache with OpRefs
    /// derived from the *parent* loop's `vable_array_base`, but the
    /// bridge owns a fresh inputarg stream (its own `OpRef::from_raw(0..N)` bound
    /// to parent-guard fail_args). Keeping the parent seed would make
    /// subsequent `vable_getarrayitem_*` / `vable_setarrayitem_*` reads
    /// return stale parent-loop OpRefs; clearing would force the vable path
    /// to fall through to the raw `GetarrayitemGc` / `SetarrayitemGc`
    /// (`ctx.has_virtualizable_boxes() == false` branch) until the
    /// bridge itself reseeds via resume data — matching the upstream site
    /// above, where the `virtualizable_boxes` are rebuilt from the guard's
    /// resume data before the bridge replays any vable op.
    pub fn clear_virtualizable_boxes(&mut self) {
        self.virtualizable_boxes = None;
        self.virtualizable_live_null_slots = None;
    }

    /// Set virtualizable_boxes with VirtualizableInfo and array lengths.
    /// Used by bridge tracing where the boxes are reconstructed from
    /// resume data (`rebuild_state_after_failure`).
    ///
    /// `values` is stamped onto `boxes` (`*FrontendOp(pos, value)` /
    /// `Const*` inline). An empty `values` slice leaves the boxes unstamped
    /// (only safe when the bridge does not execute any `BC_GET*_VABLE_*`
    /// opcodes that feed `set_*_reg`).
    pub fn set_virtualizable_boxes_with_info(
        &mut self,
        boxes: Vec<OpRef>,
        values: Vec<Value>,
        info: &VirtualizableInfo,
        array_lengths: &[usize],
    ) {
        if !values.is_empty() {
            self.stamp_virtualizable_boxes(&boxes, &values);
            self.virtualizable_live_null_slots = Some(vec![false; boxes.len()]);
        } else {
            self.virtualizable_live_null_slots = None;
        }
        self.virtualizable_boxes = Some(boxes);
        self.retain_or_store_vinfo(info);
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Reload the tracing-time `virtualizable_boxes` cache from the heap
    /// object — the `TraceCtx`-level body of
    /// `MetaInterp::load_fields_from_virtualizable` (pyjitpl.py).
    ///
    /// ```text
    /// def load_fields_from_virtualizable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         self.virtualizable_boxes = vinfo.read_boxes(self.cpu, virtualizable, 0)
    ///         self.virtualizable_boxes.append(virtualizable_box)
    /// ```
    ///
    /// It lives here rather than only on `MetaInterp` because the second
    /// upstream caller of this reload — the escape path of
    /// `vable_after_residual_call` (pyjitpl.py) — is reached from the
    /// state-field dispatcher, which holds no `MetaInterp` reference.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "The raw address is an internal JIT/GC handle validated by the descriptor and object-space boundary; making this orchestration API unsafe would incorrectly transfer collector invariants to every caller"
    )]
    pub fn load_fields_from_virtualizable(
        &mut self,
        info: &VirtualizableInfo,
        vable_ptr: *const u8,
    ) {
        if vable_ptr.is_null() {
            return;
        }
        let Some(vable_box) = self.standard_virtualizable_box() else {
            return;
        };
        let array_lengths = self
            .virtualizable_array_lengths()
            .map(|lengths| lengths.to_vec())
            .unwrap_or_default();
        let raw = unsafe { info.read_boxes(vable_ptr, &array_lengths) };
        let mut boxes = Vec::with_capacity(raw.len() + 1);
        let mut values = Vec::with_capacity(raw.len() + 1);
        for (value, ty) in raw.into_iter().zip(info.box_types(&array_lengths)) {
            let (opref, concrete) = match ty {
                majit_ir::Type::Int => (self.const_int(value), Value::Int(value)),
                majit_ir::Type::Ref => (
                    self.const_ref(value),
                    Value::Ref(majit_ir::GcRef(value as usize)),
                ),
                majit_ir::Type::Float => (
                    self.const_float(value),
                    Value::Float(f64::from_bits(value as u64)),
                ),
                majit_ir::Type::Void => continue,
            };
            boxes.push(opref);
            values.push(concrete);
        }
        boxes.push(vable_box);
        // The vable identity's concrete value is the heap pointer itself.
        values.push(Value::Ref(majit_ir::GcRef(vable_ptr as usize)));
        self.set_virtualizable_boxes_with_info(boxes, values, info, &array_lengths);
    }

    /// Canonical virtualizable metadata for the active standard virtualizable.
    pub fn virtualizable_info(&self) -> Option<&std::sync::Arc<VirtualizableInfo>> {
        self.virtualizable_info.as_ref()
    }

    /// Cached array lengths for the active standard virtualizable.
    pub fn virtualizable_array_lengths(&self) -> Option<&[usize]> {
        self.virtualizable_array_lengths.as_deref()
    }

    /// Live virtualizable heap pointer (`virtualizable_heap_ptr` / sync target).
    /// `vinfo.unwrap_virtualizable_box(virtualizable_box)` analogue for
    /// callers that need the concrete object behind
    /// `standard_virtualizable_box()` — e.g. the
    /// `tracing_before_residual_call` / `tracing_after_residual_call`
    /// token protocol around a concrete-executed residual call
    /// (pyjitpl.py:3329-3330, 3349-3353).
    pub fn virtualizable_heap_ptr(&self) -> Option<*const u8> {
        self.virtualizable_heap_ptr
    }

    /// pyjitpl.py:2394 `forced_virtualizable` accessor.
    pub fn forced_virtualizable(&self) -> Option<OpRef> {
        self.forced_virtualizable
    }

    /// pyjitpl.py:1126-1127 / 3478 `forced_virtualizable` mutator.
    pub fn set_forced_virtualizable(&mut self, value: Option<OpRef>) {
        self.forced_virtualizable = value;
    }

    // ── hint API consumption (RPython annotator/codewriter equivalent) ──

    /// Consume `hint(frame, access_directly=True)` during tracing.
    ///
    /// RPython's annotator generates JitCode that bypasses heap ops for
    /// virtualizable fields. In majit, this initializes the standard
    /// virtualizable boxes model so that subsequent vable_getfield/setfield
    /// calls access boxes directly instead of emitting heap ops.
    ///
    /// Must be called after `init_virtualizable_boxes`.
    /// Returns `true` if standard access is now active.
    pub fn hint_access_directly(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Consume `hint(frame, fresh_virtualizable=True)` during tracing.
    ///
    /// Marks that the virtualizable was freshly allocated, so its token is
    /// guaranteed to be TOKEN_NONE. The tracer skips token-check preamble.
    /// No IR is emitted; this is a tracing-time optimization.
    pub fn hint_fresh_virtualizable(&mut self, _vable_opref: OpRef) {
        // No IR needed — the token is already NONE for fresh objects.
        // This hint prevents the tracer from emitting unnecessary
        // GuardValue(token, 0) at loop entry for freshly created frames.
    }

    /// pyjitpl.py `MetaInterp.store_token_in_vable()`.
    ///
    /// ```text
    /// def store_token_in_vable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is None:
    ///         return
    ///     vbox = self.virtualizable_boxes[-1]
    ///     if vbox is self.forced_virtualizable:
    ///         return # we already forced it by hand
    ///     # in case the force_token has not been recorded, record it here
    ///     # to make sure we know the virtualizable can be broken. However,
    ///     # the contents of the virtualizable should be generally correct
    ///     force_token = self.history.record0(rop.FORCE_TOKEN,
    ///                                        lltype.nullptr(llmemory.GCREF.TO))
    ///     self.history.record2(rop.SETFIELD_GC, vbox, force_token,
    ///                          None, descr=vinfo.vable_token_descr)
    ///     self.generate_guard(rop.GUARD_NOT_FORCED_2)
    /// ```
    pub fn store_token_in_vable_setfield(&mut self) -> bool {
        let info = match self.virtualizable_info.clone() {
            Some(info) => info,
            None => return false,
        };
        let vbox = match self.standard_virtualizable_box() {
            Some(b) => b,
            None => return false,
        };
        if self.forced_virtualizable == Some(vbox) {
            return false;
        }
        let force_token = Self::do_record_op(&mut self.recorder, OpCode::ForceToken, &[]);
        let token_descr = info.token_field_descr();
        self.vable_setfield_descr(vbox, force_token, token_descr);
        // pyjitpl.py self.generate_guard(rop.GUARD_NOT_FORCED_2)
        // is recorded by the caller via the proper guard generation
        // path (`MIFrame::generate_guard` in the pyre frontend) so the
        // guard captures fresh resumedata at the current framestack
        // position, matching RPython's gen_store_back_in_vable.
        true
    }

    /// pyjitpl.py `MetaInterp.gen_store_back_in_vable(box)`.
    ///
    /// ```text
    /// def gen_store_back_in_vable(self, box):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         # xxx only write back the fields really modified
    ///         vbox = self.virtualizable_boxes[-1]
    ///         if vbox is not box:
    ///             # ignore the hint on non-standard virtualizable
    ///             # specifically, ignore it on a virtual
    ///             return
    ///         if self.forced_virtualizable is not None:
    ///             # this can happen only in strange cases, but we don't care
    ///             # it was already forced
    ///             return
    ///         self.forced_virtualizable = vbox
    ///         ...emit SETFIELD_GC for each static field...
    ///         ...emit SETARRAYITEM_GC for each array item...
    ///         ...emit final SETFIELD_GC(vbox, NULL, vable_token_descr)...
    /// ```
    pub fn gen_store_back_in_vable(&mut self, vable_opref: OpRef) {
        let (info, boxes, lengths) = match (
            self.virtualizable_info.clone(),
            self.virtualizable_boxes.clone(),
            self.virtualizable_array_lengths.clone(),
        ) {
            (Some(info), Some(boxes), Some(lengths)) => (info, boxes, lengths),
            _ => return,
        };

        // pyjitpl.py:3469 vbox = self.virtualizable_boxes[-1]
        // pyjitpl.py synchronize_virtualizable if vbox is not box: return  (ignore nonstandard)
        if boxes.last().copied() != Some(vable_opref) {
            return;
        }

        // pyjitpl.py:3474-3477 if forced_virtualizable is not None: return
        if self.forced_virtualizable.is_some() {
            return;
        }
        // pyjitpl.py:3478 self.forced_virtualizable = vbox
        self.forced_virtualizable = Some(vable_opref);

        // pyjitpl.py `gen_store_back_in_vable` writes every static field and
        // every array item. The `xxx only write back the fields really
        // modified` note is not a filter.
        for field_index in 0..info.static_fields.len() {
            if let Some(&value) = boxes.get(field_index) {
                // pyjitpl.py `gen_store_back_in_vable` records SETFIELD_GC
                // with `vinfo.static_field_descrs[i]` (`cpu.fielddescrof`).
                // OptHeap keys the lazy-set cache by descr identity, so
                // reuse `static_field_struct_descr` or last_instr is stored twice.
                let descr = info.static_field_struct_descr(field_index);
                // pyjitpl.py `gen_store_back_in_vable`. A store has no
                // `resvalue` and `SETFIELD_GC` is never pure, so no cpu.
                self.execute_and_record(
                    None,
                    OpCode::SetfieldGc,
                    Some(descr),
                    &[vable_opref, value],
                    None,
                    0,
                );
            }
        }

        let mut flat_box_index = info.static_fields.len();
        for array_index in 0..info.array_fields.len() {
            let len = lengths.get(array_index).copied().unwrap_or(0);
            let field_descr = info.array_pointer_struct_descr(array_index);
            let array_descr = info.array_item_descr(array_index);
            let array_ref = self.vable_getfield_ref_descr(vable_opref, field_descr);
            for item_index in 0..len {
                if let Some(&value) = boxes.get(flat_box_index) {
                    let index = self.const_int(item_index as i64);
                    self.execute_and_record(
                        None,
                        OpCode::SetarrayitemGc,
                        Some(array_descr.clone()),
                        &[array_ref, index, value],
                        None,
                        0,
                    );
                }
                flat_box_index += 1;
            }
        }

        // virtualizable.py `vable_token` is llmemory.GCREF.  Use the same
        // ConstPtr null here so the descriptor remains a pointer field for GC
        // rewriting; wasm in particular must barrier the preceding non-null
        // FORCE_TOKEN store into an old PyFrame.
        let null = self.const_null();
        self.record_op_with_descr(
            OpCode::SetfieldGc,
            &[vable_opref, null],
            info.token_field_descr(),
        );
    }

    /// Load dest-frame virtualizable fields as recorded IR boxes.
    ///
    /// When the merge-point JUMP is about a virtualizable other than the
    /// one `virtualizable_boxes` was seeded from, or whose array is
    /// longer, the cached boxes do not describe dest. Upstream fills that
    /// gap with `initialize_virtualizable` (`virtualizable.py read_boxes`
    /// / `get_array_length`) at trace start and with the nonstandard
    /// `getfield_gc_*` / `getarrayitem_gc_*` path at a later frame;
    /// `compile.py patch_new_loop_to_load_virtualizable_fields` emits the
    /// same loads at loop entry. This is that load sequence against
    /// `vbox`, installing the results as `virtualizable_boxes` ending in
    /// `vbox`.
    ///
    /// Same-object length growth keeps the live prefix and only records
    /// extra `GETARRAYITEM_GC` slots, so a crossed JUMP matches dest
    /// LABEL arity without replacing loop-carried boxes.
    pub fn gen_load_from_other_virtualizable(&mut self, vbox: OpRef) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        let dest_ptr = match self.concrete_of_opref(vbox) {
            Some(Value::Ref(gcref)) if gcref.as_usize() != 0 => gcref.as_usize(),
            _ => self
                .standard_virtualizable_ptr()
                .or_else(|| self.virtualizable_heap_ptr.map(|p| p as usize))
                .filter(|&ptr| ptr != 0)
                .unwrap_or(0),
        };
        let dest_lengths = if dest_ptr != 0 && info.can_read_all_array_lengths_from_heap() {
            unsafe { info.read_array_lengths_from_heap(dest_ptr as *const u8) }
        } else {
            return;
        };
        self.gen_load_from_other_virtualizable_with_lengths(vbox, dest_ptr, &dest_lengths);
    }

    fn gen_load_from_other_virtualizable_with_lengths(
        &mut self,
        vbox: OpRef,
        dest_ptr: usize,
        dest_lengths: &[usize],
    ) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        let dest_array: usize = dest_lengths.iter().copied().sum();
        let cached_array: usize = self
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum())
            .unwrap_or(0);
        let same_object = match (self.standard_virtualizable_ptr(), dest_ptr) {
            (Some(ptr), dest) if dest != 0 => ptr == dest,
            _ => self.standard_virtualizable_box() == Some(vbox),
        };
        if same_object && dest_array == cached_array {
            return;
        }
        if same_object && dest_array > cached_array && info.array_fields.len() == 1 {
            self.extend_vable_array_from_frame(
                vbox,
                dest_ptr,
                &info,
                cached_array,
                dest_array,
                dest_lengths,
            );
            return;
        }
        if same_object && dest_array < cached_array && info.array_fields.len() == 1 {
            self.truncate_vable_array_to(dest_array, dest_lengths);
            return;
        }
        let mut boxes = self.record_vable_field_reads(vbox, dest_ptr, &info, dest_lengths);
        boxes.push(vbox);
        self.replace_loaded_virtualizable_boxes(boxes, &info, dest_lengths);
        if dest_ptr != 0 {
            self.set_virtualizable_heap_ptr(dest_ptr as *const u8);
        }
    }

    /// Install `boxes` as `virtualizable_boxes`. Each box already carries
    /// its recorded result (`history.py _make_op` / `execute_and_record`);
    /// an empty `values` slice to `set_virtualizable_boxes_with_info` does
    /// not restamp. A previously-seeded shadow keeps
    /// `virtualizable_live_null_slots` allocated at the new length so
    /// `has_virtualizable_shadow` stays true for later vable stores.
    fn replace_loaded_virtualizable_boxes(
        &mut self,
        boxes: Vec<OpRef>,
        info: &VirtualizableInfo,
        dest_lengths: &[usize],
    ) {
        let keep_shadow = self.has_virtualizable_shadow();
        let n = boxes.len();
        let values = boxes
            .iter()
            .map(|&b| self.box_value(b))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
        self.set_virtualizable_boxes_with_info(boxes, values, info, dest_lengths);
        if keep_shadow && self.virtualizable_live_null_slots.is_none() {
            self.virtualizable_live_null_slots = Some(vec![false; n]);
        }
    }

    fn record_getfield_stamped(
        &mut self,
        opcode: OpCode,
        vbox: OpRef,
        dest_ptr: usize,
        descr: DescrRef,
        kind: Type,
    ) -> OpRef {
        // pyjitpl.py `MetaInterp.execute_and_record`: execute then
        // `_record_helper(opnum, resvalue, descr, *argboxes)`. The box
        // carries the result (`*FrontendOp(pos, value)`). `dest_ptr == 0`
        // still loads when `vbox` itself carries a live pointer.
        let live = self.field_live_value(dest_ptr as i64, vbox, &descr, kind);
        self.execute_and_record(None, opcode, Some(descr), &[vbox], live, 0)
    }

    fn record_getarrayitem_stamped(
        &mut self,
        opcode: OpCode,
        array_op: OpRef,
        index: i64,
        array_descr: DescrRef,
        item_type: Type,
    ) -> OpRef {
        let live = self.array_live_value(array_op, index, &array_descr, item_type);
        let const_idx = self.const_int(index);
        self.execute_and_record(
            None,
            opcode,
            Some(array_descr),
            &[array_op, const_idx],
            live,
            0,
        )
    }

    fn record_vable_field_reads(
        &mut self,
        vbox: OpRef,
        dest_ptr: usize,
        vinfo: &VirtualizableInfo,
        array_lengths: &[usize],
    ) -> Vec<OpRef> {
        let mut boxes = Vec::with_capacity(
            vinfo.static_fields.len() + array_lengths.iter().copied().sum::<usize>(),
        );
        let static_descrs = vinfo.static_field_descrs();
        for (fi, field) in vinfo.static_fields.iter().enumerate() {
            let opcode = match field.field_type {
                Type::Int => OpCode::GetfieldGcI,
                Type::Ref => OpCode::GetfieldGcR,
                Type::Float => OpCode::GetfieldGcF,
                Type::Void => continue,
            };
            let op = self.record_getfield_stamped(
                opcode,
                vbox,
                dest_ptr,
                static_descrs[fi].clone(),
                field.field_type,
            );
            boxes.push(op);
        }
        let array_field_descrs = vinfo.array_field_descrs();
        for (ai, array_field_descr) in array_field_descrs.iter().enumerate() {
            let array_len = array_lengths.get(ai).copied().unwrap_or(0);
            let array_op = self.record_getfield_stamped(
                OpCode::GetfieldGcR,
                vbox,
                dest_ptr,
                array_field_descr.clone(),
                Type::Ref,
            );
            let array_descr = vinfo.array_descrs[ai].clone();
            let item_type = vinfo.array_fields[ai].item_type;
            let item_opcode = match item_type {
                Type::Int => OpCode::GetarrayitemGcI,
                Type::Ref => OpCode::GetarrayitemGcR,
                Type::Float => OpCode::GetarrayitemGcF,
                Type::Void => {
                    panic!("gen_load_from_other_virtualizable: array {ai} has Void item_type")
                }
            };
            for index in 0..array_len {
                let op = self.record_getarrayitem_stamped(
                    item_opcode,
                    array_op,
                    index as i64,
                    array_descr.clone(),
                    item_type,
                );
                boxes.push(op);
            }
        }
        boxes
    }

    fn extend_vable_array_from_frame(
        &mut self,
        vbox: OpRef,
        dest_ptr: usize,
        info: &VirtualizableInfo,
        cached_array: usize,
        dest_array: usize,
        dest_lengths: &[usize],
    ) {
        let Some(mut boxes) = self.virtualizable_boxes.clone() else {
            return;
        };
        if boxes.is_empty() {
            return;
        }
        let field_descr = info.array_pointer_field_descr(0);
        let array_descr = info.array_item_descr(0);
        let item_type = info.array_fields[0].item_type;
        let item_opcode = match item_type {
            Type::Int => OpCode::GetarrayitemGcI,
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            Type::Void => return,
        };
        let array_op = self.record_getfield_stamped(
            OpCode::GetfieldGcR,
            vbox,
            dest_ptr,
            field_descr,
            Type::Ref,
        );
        let identity = boxes.pop().unwrap();
        for index in cached_array..dest_array {
            let op = self.record_getarrayitem_stamped(
                item_opcode,
                array_op,
                index as i64,
                array_descr.clone(),
                item_type,
            );
            boxes.push(op);
        }
        boxes.push(identity);
        self.replace_loaded_virtualizable_boxes(boxes, info, dest_lengths);
    }

    fn truncate_vable_array_to(&mut self, dest_array: usize, dest_lengths: &[usize]) {
        let Some(info) = self.virtualizable_info.clone() else {
            return;
        };
        let nstatic = info.num_static_extra_boxes;
        let Some(mut boxes) = self.virtualizable_boxes.clone() else {
            return;
        };
        if boxes.len() < nstatic + dest_array + 1 {
            return;
        }
        let identity = boxes[boxes.len() - 1];
        boxes.truncate(nstatic + dest_array);
        boxes.push(identity);
        self.replace_loaded_virtualizable_boxes(boxes, &info, dest_lengths);
    }

    /// `compile.py patch_new_loop_to_load_virtualizable_fields`
    /// mirrored at the call site instead of the callee preamble.
    ///
    /// Emits `GETFIELD_GC` for every static field and `GETFIELD_GC_R`
    /// + `GETARRAYITEM_GC` for every array item of the virtualizable
    ///   referenced by `vable`. Returns the freshly recorded OpRefs in
    ///   `[scalar_0, ..., scalar_{N-1}, array_0_item_0, ...,
    /// array_K_item_M]` order — the callee inputarg order minus the
    ///   leading frame reference.
    ///
    /// `array_lengths[i]` is the live element count of the i-th array
    /// field, mirroring `vinfo.get_array_length(vable, arrayindex)`
    /// at compile.py:443. The caller is expected to have read these
    /// off the concrete virtualizable before tracing the call.
    ///
    /// Dormant — the `call_assembler_red_only_*` call sites will plug
    /// this in once the callee JUMP-terminated paths run
    /// `patch_new_loop_to_load_virtualizable_fields`. Covered by
    /// `emit_vable_field_reads_emits_compile_py_shape` so the helper
    /// stays honest until the call-site flip lands.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn emit_vable_field_reads(
        &mut self,
        vable: OpRef,
        vinfo: &VirtualizableInfo,
        array_lengths: &[usize],
    ) -> Vec<OpRef> {
        let mut expanded = Vec::with_capacity(
            vinfo.static_fields.len() + array_lengths.iter().copied().sum::<usize>(),
        );

        // compile.py:434-440 — GETFIELD_GC per static field.
        let static_descrs = vinfo.static_field_descrs();
        for (fi, field) in vinfo.static_fields.iter().enumerate() {
            let opcode = match field.field_type {
                Type::Int => OpCode::GetfieldGcI,
                Type::Ref => OpCode::GetfieldGcR,
                Type::Float => OpCode::GetfieldGcF,
                Type::Void => panic!("emit_vable_field_reads: static field {fi} has Void type"),
            };
            let descr = static_descrs[fi].clone();
            let opref = self.record_op_with_descr(opcode, &[vable], descr);
            expanded.push(opref);
        }

        // compile.py:441-457 — GETFIELD_GC_R(array ptr) + GETARRAYITEM_GC.
        let array_field_descrs = vinfo.array_field_descrs();
        for (ai, array_field_descr) in array_field_descrs.iter().enumerate() {
            let array_len = array_lengths.get(ai).copied().unwrap_or(0);
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable], array_field_descr.clone());
            let array_descr = vinfo.array_descrs[ai].clone();
            let item_opcode = match vinfo.array_fields[ai].item_type {
                Type::Int => OpCode::GetarrayitemGcI,
                Type::Ref => OpCode::GetarrayitemGcR,
                Type::Float => OpCode::GetarrayitemGcF,
                Type::Void => panic!("emit_vable_field_reads: array {ai} has Void item_type"),
            };
            for index in 0..array_len {
                let const_idx = self.const_int(index as i64);
                let opref = self.record_op_with_descr(
                    item_opcode,
                    &[array_opref, const_idx],
                    array_descr.clone(),
                );
                expanded.push(opref);
            }
        }
        expanded
    }

    /// pyjitpl.py `MIFrame.emit_force_virtualizable(fielddescr, box)`.
    ///
    /// ```text
    /// def emit_force_virtualizable(self, fielddescr, box):
    ///     vinfo = fielddescr.get_vinfo()
    ///     assert vinfo is not None
    ///     token_descr = vinfo.vable_token_descr
    ///     mi = self.metainterp
    ///     tokenbox = mi.execute_and_record(rop.GETFIELD_GC_R, token_descr, box)
    ///     condbox = mi.execute_and_record(rop.PTR_NE, None, tokenbox, CONST_NULL)
    ///     funcbox = ConstInt(rffi.cast(lltype.Signed, vinfo.clear_vable_ptr))
    ///     calldescr = vinfo.clear_vable_descr
    ///     self.execute_varargs(rop.COND_CALL, [condbox, funcbox, box],
    ///                          calldescr, False, False)
    /// ```
    fn emit_force_virtualizable(&mut self, fielddescr: &DescrRef, vable_opref: OpRef) {
        //     vinfo = fielddescr.get_vinfo()
        //     assert vinfo is not None
        //
        // `finalize_arc` stamps every field descriptor with a
        // `Weak<dyn VinfoMarker>` backref; `get_vinfo()` upgrades it
        // and returns the owning `VirtualizableInfo`.  When the
        // descriptor was built via the legacy by-value
        // `set_parent_descr` path (no Arc available), `get_vinfo()`
        // returns `None` and pyre falls back to the active
        // `self.virtualizable_info` slot so the existing by-value
        // test harness keeps working.
        let marker = self.vinfo_from_fielddescr(fielddescr);
        let (token_descr, clear_ptr, clear_descr) = {
            let info_ref: &VirtualizableInfo = if let Some(ref m) = marker {
                m.as_any()
                    .downcast_ref::<VirtualizableInfo>()
                    .expect("emit_force_virtualizable: VinfoMarker is not a VirtualizableInfo")
            } else {
                self.virtualizable_info
                    .as_deref()
                    .expect("emit_force_virtualizable: vinfo is None")
            };
            //     token_descr = vinfo.vable_token_descr
            let token_descr = info_ref.token_field_descr();
            //     funcbox = ConstInt(rffi.cast(lltype.Signed, vinfo.clear_vable_ptr))
            let clear_ptr = info_ref
                .clear_vable_ptr
                .expect("emit_force_virtualizable: clear_vable_ptr not set");
            //     calldescr = vinfo.clear_vable_descr
            let clear_descr = info_ref
                .clear_vable_descr
                .clone()
                .expect("emit_force_virtualizable: clear_vable_descr not set");
            (token_descr, clear_ptr, clear_descr)
        };
        // pyjitpl.py `MetaInterp.execute_and_record`: `executor.execute`
        // then `history.record(..., resvalue)`. When `box` carries a live
        // pointer, `field_sanity_load` reads the token through
        // `vable_token_descr` (`history.py _make_op`).
        let token_live = self
            .live_ptr_of(vable_opref)
            .and_then(|ptr| self.field_sanity_load(ptr, &token_descr, Type::Ref));
        //     tokenbox = mi.execute_and_record(rop.GETFIELD_GC_R, token_descr, box)
        let tokenbox = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(token_descr),
            &[vable_opref],
            token_live,
            0,
        );
        //     condbox = mi.execute_and_record(rop.PTR_NE, None, tokenbox, CONST_NULL)
        // `MetaInterp.execute_and_record` runs `executor.execute` on the
        // metainterp CPU, then records. PTR_NE is `blackhole.py bhimpl_ptr_ne`
        // over `tokenbox.getref_base()` and CONST_NULL; the walker traces
        // the same compare through `execute_and_record` after that eval.
        let null_ref = self.const_null();
        let cond_live = self.box_value(tokenbox).and_then(|tv| {
            crate::executor::execute_ptr_compare_values(
                OpCode::PtrNe,
                tv,
                Value::Ref(majit_ir::GcRef::NULL),
            )
            .map(Value::Int)
        });
        // PTR_NE is always-pure; execute_and_record's fold gate needs a
        // model.py Cpu. Upstream `MetaInterp.execute_and_record` uses
        // `self.cpu`. The COND_CALL below still runs through
        // `blackhole_cpu` (`executor.py do_cond_call` → `cpu.bh_call_v`).
        let cpu = self.metainterp_cpu.clone();
        let condbox = self.execute_and_record(
            Some(cpu.as_ref()),
            OpCode::PtrNe,
            None,
            &[tokenbox, null_ref],
            cond_live,
            0,
        );
        let funcbox = self.const_int(clear_ptr as i64);
        //     self.execute_varargs(rop.COND_CALL, [condbox, funcbox, box],
        //                          calldescr, False, False)
        // `execute_and_record_varargs`: `executor.execute_varargs` then
        // record. `COND_CALL_N` is inside the can-raise range, which the
        // non-varargs funnel refuses. `executor.py do_cond_call`: when
        // `condbox.getint()` is true, call `clear_vable_ptr` so later
        // trace-time reads see the forced virtualizable.
        let cond_value = match self.box_value(condbox) {
            Some(Value::Int(n)) => n,
            _ => 0,
        };
        // `executor.py do_cond_call` → `do_call_n` reads
        // `argboxes[n].getref_base()`. The branch records ops with their
        // values (`*FrontendOp(pos, value)`), so the vable box always has
        // one.
        let vable_bits = self
            .box_bits(vable_opref)
            .expect("emit_force_virtualizable: vable box has no recorded value");
        let effectinfo = {
            let calldescr = clear_descr
                .as_call_descr()
                .expect("emit_force_virtualizable: clear_vable_descr is a CallDescr");
            crate::executor::do_cond_call(
                self.blackhole_cpu(),
                cond_value,
                clear_ptr as i64,
                &[vable_bits],
                calldescr,
            );
            calldescr.get_extra_info().clone()
        };
        self.profiler()
            .count_ops(OpCode::CondCallN, crate::counters::OPS);
        self.profiler()
            .count_ops(OpCode::CondCallN, crate::counters::RECORDED_OPS);
        self.heapcache_invalidate_caches_varargs(
            OpCode::CondCallN,
            Some(&effectinfo),
            &[condbox, funcbox, vable_opref],
        );
        Self::do_record_op_with_descr(
            &mut self.recorder,
            OpCode::CondCallN,
            &[condbox, funcbox, vable_opref],
            clear_descr,
        );
    }

    /// pyjitpl.py `_nonstandard_virtualizable(pc, box, fielddescr)`.
    ///
    /// ```text
    ///  def _nonstandard_virtualizable(self, pc, box, fielddescr):
    ///      # returns True if 'box' is actually not the "standard" virtualizable
    ///      # that is stored in metainterp.virtualizable_boxes[-1]
    ///      if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
    ///          self.metainterp.staticdata.profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
    ///          return True
    ///      if box is self.metainterp.forced_virtualizable:
    ///          self.metainterp.forced_virtualizable = None
    ///      if (self.metainterp.jitdriver_sd.virtualizable_info is not None or
    ///          self.metainterp.jitdriver_sd.greenfield_info is not None):
    ///          standard_box = self.metainterp.virtualizable_boxes[-1]
    ///          if standard_box is box:
    ///              return False
    ///          vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///          if vinfo is fielddescr.get_vinfo():
    ///              eqbox = self.metainterp.execute_and_record(rop.PTR_EQ, None,
    ///                                                         box, standard_box)
    ///              eqbox = self.implement_guard_value(eqbox, pc)
    ///              isstandard = eqbox.getint()
    ///              if isstandard:
    ///                  if box.type == 'r':
    ///                      self.metainterp.replace_box(box, standard_box)
    ///                  return False
    ///      if not self.metainterp.heapcache.is_unescaped(box):
    ///          self.emit_force_virtualizable(fielddescr, box)
    ///      self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
    ///      return True
    /// ```
    ///
    /// `_nonstandard_virtualizable` through the `PTR_EQ`.
    ///
    /// Step 4's `implement_guard_value` lives on the `MIFrame` (`pyjitpl.py`)
    /// and captures the live framestack at the promote. Callers that own
    /// that stack promote `eqbox` then [`Self::commit_nonstandard_virtualizable`].
    pub fn begin_nonstandard_virtualizable(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        fielddescr: &DescrRef,
    ) -> NonstandardVable {
        let concrete = self.concrete_of_opref(vable_opref);
        // Step 1: heapcache short-circuit.
        //     if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
        //         self.metainterp.staticdata.profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
        //         return True
        if self
            .heap_cache()
            .is_known_nonstandard_virtualizable(vable_opref)
        {
            // pyjitpl.py profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS).
            self.profiler()
                .count_ops(OpCode::PtrEq, crate::pyjitpl::counters::HEAPCACHED_OPS);
            return NonstandardVable::Decided(true);
        }
        // Step 2: forced_virtualizable reset on identity.
        //     if box is self.metainterp.forced_virtualizable:
        //         self.metainterp.forced_virtualizable = None
        if self.forced_virtualizable == Some(vable_opref) {
            self.forced_virtualizable = None;
        }
        // Step 3: standard_box identity check.
        //     if (self.metainterp.jitdriver_sd.virtualizable_info is not None or
        //         self.metainterp.jitdriver_sd.greenfield_info is not None):
        //         standard_box = self.metainterp.virtualizable_boxes[-1]
        //         if standard_box is box:
        //             return False
        //
        // Empty `virtualizable_boxes` is the vinfo-is-None / greenfield-is-None
        // arm: skip the standard identity check and fall through to Step 5
        // `emit_force_virtualizable`. An early `return True` here dropped the
        // force that `_nonstandard_virtualizable` still runs on that path.
        let standard_box = self
            .virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied());
        if let Some(standard_box) = standard_box {
            if standard_box == vable_opref {
                return NonstandardVable::Decided(false);
            }
            // Step 4: PTR_EQ. `implement_guard_value` is the caller's.
            //     vinfo = self.metainterp.jitdriver_sd.virtualizable_info
            //     if vinfo is fielddescr.get_vinfo():
            //         eqbox = self.metainterp.execute_and_record(
            //             rop.PTR_EQ, None, box, standard_box)
            //         eqbox = self.implement_guard_value(eqbox, pc)
            //
            // `fielddescr.get_vinfo()` upgrades the backref stamped by
            // `finalize_arc`.  When the descriptor carries a vinfo backref,
            // the upstream `vinfo is fielddescr.get_vinfo()` check holds iff
            // the active `virtualizable_info` is the same concrete type
            // (pyre single-driver: trivially true).  When the descriptor
            // lacks a backref (by-value legacy path), pyre skips the
            // PTR_EQ/replace_box short-circuit and falls through to Step 5 —
            // same behaviour as upstream when the fielddescr came from a
            // different jitdriver's vinfo.
            let descriptor_vinfo = self.vinfo_from_fielddescr(fielddescr);
            // pyjitpl.py `_nonstandard_virtualizable`:
            // `vinfo is fielddescr.get_vinfo()`. Object identity, not type.
            let descriptor_has_matching_vinfo =
                match (self.virtualizable_info.as_ref(), descriptor_vinfo.as_ref()) {
                    (Some(active), Some(marker)) => marker
                        .as_any()
                        .downcast_ref::<VirtualizableInfo>()
                        .is_some_and(|descr_info| std::ptr::eq(active.as_ref(), descr_info)),
                    _ => false,
                };
            if descriptor_has_matching_vinfo {
                let standard_concrete = self.standard_virtualizable_concrete();
                // pyjitpl.py `eqbox = self.metainterp.execute_and_record(
                //     rop.PTR_EQ, None, box, standard_box)`.
                //
                // pyre resolves `isstandard` by comparing the traced concrete
                // ptrs directly (see `concrete_of_opref` for how `concrete` is
                // reconstructed from tracer-local state). `pc` threads through
                // for signature parity with `_nonstandard_virtualizable`; the
                // caller stamps the promote's resumepc.
                let _ = pc;
                let isstandard: i64 =
                    if concrete_ptrs_eq(concrete.as_ref(), standard_concrete.as_ref()) {
                        1
                    } else {
                        0
                    };
                // pyjitpl.py `_nonstandard_virtualizable` execute leg. Step 3
                // already returned for `standard_box is box`, so two constants
                // arriving here are *different* constants and cannot be equal at
                // runtime either — the fold to `ConstInt(0)` is sound, and
                // `implement_guard_value` then short-circuits it into no
                // GUARD_VALUE. `PTR_EQ` reads no memory. The CPU is
                // `MetaInterp.cpu`, the same one `execute_and_record` uses
                // upstream.
                let cpu = self.metainterp_cpu.clone();
                let eqbox = self.execute_and_record(
                    Some(cpu.as_ref()),
                    OpCode::PtrEq,
                    None,
                    &[vable_opref, standard_box],
                    Some(Value::Int(isstandard)),
                    0,
                );
                return NonstandardVable::PendingEq {
                    eqbox,
                    isstandard,
                    vable_opref,
                    standard_box,
                };
            }
        }
        NonstandardVable::Decided(self.finish_known_nonstandard(vable_opref, fielddescr))
    }

    /// `_nonstandard_virtualizable` after `implement_guard_value(eqbox, pc)`.
    ///
    /// `promoted` is the box promote returned (`eqbox` after
    /// `implement_guard_value`). Upstream then does `isstandard = eqbox.getint()`.
    pub fn commit_nonstandard_virtualizable(
        &mut self,
        promoted: OpRef,
        vable_opref: OpRef,
        standard_box: OpRef,
        fielddescr: &DescrRef,
    ) -> bool {
        // pyjitpl.py: `isstandard = eqbox.getint()` after
        // `eqbox = self.implement_guard_value(eqbox, pc)`.
        let isstandard = match self.box_value(promoted) {
            Some(Value::Int(n)) => n,
            _ => 0,
        };
        if isstandard != 0 {
            // `_nonstandard_virtualizable`'s `if box.type == 'r':
            //     self.metainterp.replace_box(box, standard_box)`.
            // Virtualizables are always Refs here, so the
            // `box.type == 'r'` check is unconditional.
            self.replace_standard_vable(vable_opref, standard_box);
            return false;
        }
        self.finish_known_nonstandard(vable_opref, fielddescr)
    }

    /// Step 5 of `_nonstandard_virtualizable`: `emit_force_virtualizable`
    /// then `nonstandard_virtualizables_now_known`.
    fn finish_known_nonstandard(&mut self, vable_opref: OpRef, fielddescr: &DescrRef) -> bool {
        // Step 5a: emit_force_virtualizable.
        //     if not self.metainterp.heapcache.is_unescaped(box):
        //         self.emit_force_virtualizable(fielddescr, box)
        //
        //     def emit_force_virtualizable(self, fielddescr, box):
        //         vinfo = fielddescr.get_vinfo()
        //         token_descr = vinfo.vable_token_descr
        //         tokenbox = mi.execute_and_record(
        //             rop.GETFIELD_GC_R, token_descr, box)
        //         condbox = mi.execute_and_record(
        //             rop.PTR_NE, None, tokenbox, CONST_NULL)
        //         funcbox = ConstInt(rffi.cast(Signed, vinfo.clear_vable_ptr))
        //         self.execute_varargs(
        //             rop.COND_CALL, [condbox, funcbox, box],
        //             vinfo.clear_vable_descr, False, False)
        if !self.heap_cache().is_unescaped(vable_opref) {
            // `emit_force_virtualizable` starts `vinfo = fielddescr.get_vinfo();
            // assert vinfo is not None`. A plain heap FieldDescr (test harness)
            // has no backref and no active `virtualizable_info`; skip the
            // COND_CALL rather than invent a force helper.
            let can_emit = self.vinfo_from_fielddescr(fielddescr).is_some()
                || self.virtualizable_info.is_some();
            if can_emit {
                self.emit_force_virtualizable(fielddescr, vable_opref);
            }
        }
        // Step 5b: mark this box as a known nonstandard virtualizable so
        // future accesses short-circuit at Step 1.
        //     self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
        self.heap_cache_mut()
            .nonstandard_virtualizables_now_known(vable_opref);
        true
    }

    /// Resolve a `setfield_vable`/`getfield_vable` static field descr
    /// (the vinfo's `static_field_descrs[idx]`) to the parent-struct-layout
    /// `FieldDescr` for
    /// recording a real heap op on a NONSTANDARD virtualizable.
    ///
    /// A nonstandard virtualizable can be a force-materialized inline-callee
    /// VIRTUAL frame; its `NewWithVtable` construction stores the field via the
    /// parent struct descr, and the optimizer pairs reads/writes on a virtual by
    /// `index_in_parent`. The vinfo descr's `index_in_parent` follows the vinfo
    /// `[token, statics, arrays]` order, which diverges from struct declaration
    /// order for PyFrame — so recording the vinfo descr pairs the op against the
    /// WRONG slot. Resolve to the struct descr; unknown descrs pass through.
    fn vable_static_record_descr(&self, fielddescr: &DescrRef) -> DescrRef {
        self.virtualizable_info
            .as_ref()
            .and_then(|vi| {
                vi.static_field_by_descr(fielddescr)
                    .map(|idx| vi.static_field_struct_descr(idx))
            })
            .unwrap_or_else(|| fielddescr.clone())
    }

    /// Array-pointer counterpart of `vable_static_record_descr`: resolves a
    /// `*_vable_*`-indexed array-pointer field descr to the parent-struct-layout
    /// descr so the `GetfieldGcR` that fetches the array base off a virtual
    /// frame pairs with the frame's construction.
    fn vable_array_record_descr(&self, fdescr: &DescrRef) -> DescrRef {
        self.virtualizable_info
            .as_ref()
            .and_then(|vi| {
                vi.array_field_by_descr(fdescr)
                    .map(|idx| vi.array_pointer_struct_descr(idx))
            })
            .unwrap_or_else(|| fdescr.clone())
    }

    /// Resolve the array-base `OpRef` (`frame.locals_cells_stack`) for a
    /// NONSTANDARD virtualizable through the heapcache, mirroring the scalar
    /// `vable_getfield_ref` field-forward (`heapcache_getfield_cached` →
    /// `heapcache_getfield_now_known`).  A force-materialized inline-callee
    /// VIRTUAL frame stores its locals array once at construction
    /// (`emit_new_pyframe_inline_with_params`, recorded under the parent-struct
    /// `vable_array_record_descr` index); forwarding the cached array box keeps
    /// every subsequent `getarrayitem_vable`/`setarrayitem_vable` rooted at the
    /// SAME array `OpRef`, so the per-array heapcache can forward the stored
    /// element box (carrying its concrete shadow) instead of recording a fresh
    /// `GetfieldGcR` whose result has no concrete — the gap that made a pure
    /// in-callee comparison branch surface `GotoIfNotValueNotConcrete`.
    /// Recover the concrete shadow of `frame.locals_cells_stack[item_index]`
    /// for a NONSTANDARD virtualizable (a force-materialized inline-callee
    /// VIRTUAL frame) from the heapcache, WITHOUT recording any op.
    ///
    /// The frame's locals array is stored once at construction
    /// (`emit_new_pyframe_inline_with_params`): the array-pointer field is
    /// heapcached against the parent-struct `vable_array_record_descr` index,
    /// and each element box against the vinfo `array_item_descr` index.  Peek
    /// both caches (array base → element box) and read the element's intrinsic
    /// `box_value`.  The caller still RECORDS the `GetfieldGcR`/`Getarrayitem`
    /// ops exactly as before, so the recorded SSA — and every guard's resume
    /// snapshot — is byte-identical; only the read result's concrete shadow is
    /// recovered (`None` when the frame is not a heapcache-tracked virtual,
    /// matching the prior `Value::Void` behavior).  Recovering it lets a pure
    /// in-callee comparison branch fold instead of surfacing
    /// `GotoIfNotValueNotConcrete`.
    fn nonstandard_vable_element_concrete(
        &mut self,
        vable_opref: OpRef,
        fdescr: &DescrRef,
        index: OpRef,
        adescr_index: u32,
    ) -> Option<Value> {
        let record_descr = self.vable_array_record_descr(fdescr);
        let base = self.heapcache_getfield_cached(vable_opref, record_descr.index())?;
        let elem = self.heapcache_getarrayitem(base, index, adescr_index)?;
        self.box_value(elem)
    }

    /// Resolve the array-base `OpRef` (`frame.locals_cells_stack_w`) for a
    /// NONSTANDARD virtualizable the way `opimpl_getfield_gc_r` does
    /// (`_opimpl_getfield_gc_any_pureornot`, pyjitpl.py): forward the cached
    /// field box on a hit, otherwise record the `GetfieldGcR` once and publish
    /// it with `getfield_now_known`.
    ///
    /// Every `getarrayitem_vable` / `setarrayitem_vable` on such a frame has to
    /// come through here, because the per-array element cache is keyed by this
    /// base `OpRef`.  Recording a fresh base per access instead keys each store
    /// under an `OpRef` no read ever looks up, so a store neither updates nor
    /// invalidates the entry the reads share — a local written after the read
    /// that seeded the cache then keeps reading its pre-store value for as long
    /// as the base stays cached.
    fn nonstandard_vable_array_base(&mut self, vable_opref: OpRef, fdescr: &DescrRef) -> OpRef {
        let record_descr = self.vable_array_record_descr(fdescr);
        let field_index = record_descr.index();
        if let Some(cached) = self.heapcache_getfield_cached(vable_opref, field_index) {
            self.profiler().count_ops(
                OpCode::GetfieldGcR,
                crate::pyjitpl::counters::HEAPCACHED_OPS,
            );
            return cached;
        }
        // The array base is stamped after the fact by
        // `stamp_vable_array_base`, not handed in, so there is no `resvalue`
        // for the funnel to fold against and no reader for a cpu to be.
        let op = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(record_descr.clone()),
            &[vable_opref],
            None,
            0,
        );
        let vable_concrete = self.concrete_of_opref(vable_opref);
        self.stamp_vable_array_base(op, vable_concrete, &record_descr);
        self.heapcache_getfield_now_known(vable_opref, field_index, op);
        op
    }

    /// pyjitpl.py `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_i(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_i(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    /// `opimpl_getfield_vable_i` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    pub fn vable_getfield_int_checked(
        &mut self,
        nonstandard: bool,
        cpu: &dyn crate::cpu::Cpu,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if nonstandard {
            // self.opimpl_getfield_gc_i(box, fielddescr) →
            // _opimpl_getfield_gc_any_pureornot (pyjitpl.py).
            let record_descr = self.vable_static_record_descr(&fielddescr);
            let field_index = record_descr.index();
            if let Some(cached) = self.heapcache_getfield_cached(vable_opref, field_index) {
                // pyjitpl.py:934-945 sanity check: run the live field
                // load (`executor.execute`) and assert equality against
                // `upd.currfieldbox.getint()`.  The Box identity is
                // `cached` (an OpRef); its intrinsic runtime value is
                // surfaced via `box_value(cached)` — covering the
                // const pool, standard-virtualizable shadow, and
                // the frontend object's `value` field (RPython
                // `currfieldbox.getXXX()` dispatch parity).
                let cached_value = self.box_value(cached);
                let expected_int = match cached_value {
                    Some(Value::Int(n)) => Some(n),
                    _ => None,
                };
                if let Some(cached_int) = expected_int
                    && vable_struct_ptr != 0
                    && let Some(Value::Int(loaded)) =
                        self.field_sanity_load(vable_struct_ptr, &fielddescr, Type::Int)
                {
                    assert_eq!(
                        loaded, cached_int,
                        "_opimpl_getfield_gc_any_pureornot sanity \
                                 check: loaded {loaded} != cached {cached_int} \
                                 (field_index={field_index}, vable_struct_ptr=\
                                 {vable_struct_ptr:#x})"
                    );
                }
                // pyjitpl.py:946 profiler.count_ops(rop.GETFIELD_GC_I,
                // Counters.HEAPCACHED_OPS) on cache hit.
                self.profiler().count_ops(
                    OpCode::GetfieldGcI,
                    crate::pyjitpl::counters::HEAPCACHED_OPS,
                );
                return (cached, cached_value);
            }
            // pyjitpl.py:949 upd.getfield_now_known(resbox).  `resbox`
            // in RPython carries the loaded value via `BoxInt.value`;
            // pyre stamps the frontend value slot for `op` with the live
            // load so subsequent `box_value(op)` sees the
            // executor-returned payload (RPython `IntFrontendOp(pos,
            // intval)` construction-time field assignment).  It is the
            // funnel's `resvalue`, so the load runs before the record;
            // `None` — no live pointer on the box either, or an unwired
            // backend — records without folding.
            let live = self.field_live_value(vable_struct_ptr, vable_opref, &fielddescr, Type::Int);
            // pyjitpl.py:1173-1199 nonstandard vable miss delegates to
            // the standard heap operation.  GETFIELD_GC_I is not an OVF
            // opcode, so the funnel never reads `last_exc_value` and 0 names
            // no copy of it.
            let op = self.execute_and_record(
                Some(cpu),
                OpCode::GetfieldGcI,
                Some(record_descr),
                &[vable_opref],
                live,
                0,
            );
            self.heapcache_getfield_now_known(vable_opref, field_index, op);
            return (op, live);
        }
        // pyjitpl.py:1170,1177,1184,1228 — the reader asserts the shadow is
        // coherent; it never repairs it.
        self.check_synchronized_virtualizable();
        // index = self._get_virtualizable_field_index(fielddescr)
        // return self.metainterp.virtualizable_boxes[index]
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index
            && let Some((op, value)) = self.vable_box_result_at(idx)
        {
            return (op, value);
        }
        // Fallback for tests/missing layout.  No live load reached this leg,
        // so the funnel's `resvalue` is `None` and it records unconditionally.
        let op = self.execute_and_record(
            Some(cpu),
            OpCode::GetfieldGcI,
            Some(fielddescr),
            &[vable_opref],
            None,
            0,
        );
        (op, None)
    }

    /// Record a virtualizable field read with an explicit field descriptor.
    pub fn vable_getfield_int_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], descr)
    }

    /// `heapcache.py is_nullity_known(box)`, supplying the `getref_base()`
    /// reader its `Const` arm needs. `Some(true)` / `Some(false)` / `None` are
    /// known-nonnull / known-null / unknown; only `Some(true)` is upstream's
    /// truthy answer, for the reason spelled out at
    /// [`Self::trace_assert_not_none`].
    /// `if self.metainterp.heapcache.is_nullity_known(box):` — upstream's
    /// *truthiness* test, which is not the same question as
    /// [`Self::heapcache_nullity_known`] answering `Some`.
    ///
    /// `heapcache.py` returns `bool(box.getref_base())` for a `Const` and
    /// `_check_flag(box, HF_KNOWN_NULLITY)` otherwise, and
    /// `nullity_now_known` sets that flag for *either* nullity. So a
    /// non-`Const` box already known to be **null** answers true here and
    /// short-circuits, while a **null `Const`** answers false and falls
    /// through. Reading `Some(true)` alone gets the second of those right and
    /// the first wrong, and re-guards a box whose nullity is already settled.
    pub fn heapcache_nullity_answered(&self, opref: OpRef) -> bool {
        self.heap_cache().is_nullity_known(opref)
    }

    pub fn heapcache_nullity_known(&self, opref: OpRef) -> Option<bool> {
        if opref.is_constant() {
            return Some(self.heap_cache().is_nullity_known(opref));
        }
        if !self.heap_cache().is_nullity_known(opref) {
            return None;
        }
        match self.box_value(opref) {
            Some(Value::Ref(g)) => Some(g.0 != 0),
            Some(Value::Int(n)) => Some(n != 0),
            _ => Some(true),
        }
    }

    /// pyjitpl.py `opimpl_assert_not_none`:
    ///
    /// ```text
    ///  def opimpl_assert_not_none(self, box):
    ///      if self.metainterp.heapcache.is_nullity_known(box):
    ///          self.metainterp.staticdata.profiler.count_ops(
    ///              rop.ASSERT_NOT_NONE, Counters.HEAPCACHED_OPS)
    ///          return
    ///      self.execute(rop.ASSERT_NOT_NONE, box)
    ///      self.metainterp.heapcache.nullity_now_known(box)
    /// ```
    ///
    /// Mirrors RPython's `jit::assert_not_none` hint (rlib/jit.rs +
    /// rtyper/debug.py `ll_assert_not_none`). Cache hit short-circuits
    /// the record and bumps `HEAPCACHED_OPS`; cache miss records
    /// `AssertNotNone` and stamps `nullity_now_known(true)` so subsequent
    /// nullity-aware sites (`_establish_nullity`, KnownClass guards) can
    /// skip their own checks.
    pub fn trace_assert_not_none(&mut self, opref: OpRef, concrete: i64) {
        // pyjitpl.py `if self.metainterp.heapcache.is_nullity_known(box):`
        // — RPython's `is_nullity_known` (heapcache.py) returns
        // `bool(box.getref_base())` for `Const` and `_check_flag(...
        // HF_KNOWN_NULLITY)` otherwise.  `class_now_known` sets
        // `HF_KNOWN_NULLITY` alongside `HF_KNOWN_CLASS` (line 470-473),
        // so the flag semantically means "known to be non-null".
        // The `if`-test therefore short-circuits only on truthy values
        // — `Const` known-null returns `False` and falls through to
        // `executor.do_assert_not_none`, which `fatalerror`s on null
        // (executor.py:344-346).  Pyre's `is_nullity_known` returns
        // `Some(true)` for known non-null, `Some(false)` for known
        // null, `None` for unknown — match PyPy's semantics by
        // short-circuiting only on `Some(true)`.
        if self.heapcache_nullity_known(opref) == Some(true) {
            self.profiler().count_ops(
                OpCode::AssertNotNone,
                crate::pyjitpl::counters::HEAPCACHED_OPS,
            );
            return;
        }
        // pyjitpl.py `self.execute(rop.ASSERT_NOT_NONE, box)` →
        // executor.py `do_assert_not_none(cpu, _, box)`:
        //     if not box.getref_base():
        //         fatalerror("found during JITting: ll_assert_not_none() failed")
        assert!(
            concrete != 0,
            "do_assert_not_none: ref operand {opref:?} is null at trace time"
        );
        // pyjitpl.py `execute(ASSERT_NOT_NONE, ...)`. Void result, never
        // pure — the funnel records it and no cpu is consulted.
        self.execute_and_record(None, OpCode::AssertNotNone, None, &[opref], None, 0);
        // pyjitpl.py:391 `self.metainterp.heapcache.nullity_now_known(box)`.
        self.heap_cache_mut().nullity_now_known(opref);
    }

    /// pyjitpl.py `opimpl_record_exact_class`:
    ///
    /// ```text
    ///  def opimpl_record_exact_class(self, box, clsbox):
    ///      if self.metainterp.heapcache.is_class_known(box):
    ///          self.metainterp.staticdata.profiler.count_ops(
    ///              rop.RECORD_EXACT_CLASS, Counters.HEAPCACHED_OPS)
    ///          return
    ///      if isinstance(clsbox, Const):
    ///          self.execute(rop.RECORD_EXACT_CLASS, box, clsbox)
    ///          self.metainterp.heapcache.class_now_known(box)
    ///          self.metainterp.heapcache.nullity_now_known(box)
    /// ```
    ///
    /// Mirrors RPython's `jit::record_exact_class` hint (`majit-metainterp`'s
    /// `jit.rs` `record_exact_class`).
    /// `cls_const` is the class-vtable ConstInt OpRef, matching
    /// backend/model.py `cls_of_box()` and the `/ri` bytecode
    /// shape. Cache hit short-circuits and bumps
    /// `HEAPCACHED_OPS`; miss records `RecordExactClass` and stamps
    /// both `class_now_known` and `nullity_now_known(true)` per
    /// pyjitpl.py.  Panics if `cls_const` resolves to a non-Int
    /// constant — the dispatcher invariant guarantees int-kind here.
    pub fn trace_record_exact_class(&mut self, opref: OpRef, cls_const: OpRef) {
        if self.heap_cache().is_class_known(opref) {
            self.profiler().count_ops(
                OpCode::RecordExactClass,
                crate::pyjitpl::counters::HEAPCACHED_OPS,
            );
            return;
        }
        if !cls_const.is_constant() {
            // pyjitpl.py `if isinstance(clsbox, Const):` — non-Const
            // class argument silently skips the record in RPython.
            return;
        }
        // pyjitpl.py `execute(RECORD_EXACT_CLASS, ...)`. Void result, never
        // pure, so the funnel records it without consulting a cpu.
        self.execute_and_record(
            None,
            OpCode::RecordExactClass,
            None,
            &[opref, cls_const],
            None,
            0,
        );
        let cls_value = match self.constants_get_value(cls_const) {
            Some(Value::Int(vtable)) => vtable,
            other => panic!(
                "trace_record_exact_class: cls_const {:?} must resolve to a \
                 ConstInt vtable address; got {:?} — bytecode argcodes are /ri",
                cls_const, other
            ),
        };
        let _ = cls_value;
        self.heap_cache_mut().class_now_known(opref);
        self.heap_cache_mut().nullity_now_known(opref);
    }

    /// pyjitpl.py `_opimpl_setfield_vable(box, valuebox, fielddescr, pc)`.
    ///
    /// ```text
    ///  def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      self.metainterp.virtualizable_boxes[index] = valuebox
    ///      self.metainterp.synchronize_virtualizable_at(index)
    /// ```
    ///
    /// Returns the shadow slot the standard leg overwrote, so a caller that
    /// still has to build a resume snapshot for a guard the promote above
    /// emitted can put the old Box back first — see [`VableEntryWrite`].
    /// `None` on the nonstandard leg, which records a heap `SetfieldGc` and
    /// leaves the shadow untouched.
    /// `_opimpl_setfield_vable` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    pub fn vable_setfield_checked(
        &mut self,
        nonstandard: bool,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Option<Value>,
    ) -> Option<VableEntryWrite> {
        if nonstandard {
            // self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
            // (pyjitpl.py).
            //
            // The codewriter emits the vinfo's `static_field_descrs[idx]`
            // for `setfield_vable`. On the STANDARD virtualizable
            // path below this descr is only used as the virtualizable-boxes
            // index, never recorded. On the NONSTANDARD path the field write
            // is recorded as a real `SetfieldGc` — and a nonstandard frame can
            // be a force-materialized virtual (the multi-frame inline callee
            // frame), whose `optimize_setfield_gc` requires a real `FieldDescr`
            // (`as_field_descr()` / `get_parent_descr()`) AND pairs it with the
            // frame's construction by `index_in_parent`. Resolve to the
            // parent-struct-layout `FieldDescr`; a non-static-field descr passes
            // through unchanged.
            let record_descr = self.vable_static_record_descr(&fielddescr);
            let field_index = record_descr.index();
            if let Some(cached) = self.heapcache_getfield_cached(vable_opref, field_index)
                && cached == value
            {
                // pyjitpl.py:977 profiler.count_ops(rop.SETFIELD_GC,
                // Counters.HEAPCACHED_OPS) when the cache already
                // holds `valuebox` — `upd.currfieldbox is valuebox`
                // (Box identity, not value equality).
                self.profiler()
                    .count_ops(OpCode::SetfieldGc, crate::pyjitpl::counters::HEAPCACHED_OPS);
                return None;
            }
            // pyjitpl.py:1173-1199 nonstandard vable miss delegates to
            // the standard heap operation.
            self.execute_and_record(
                None,
                OpCode::SetfieldGc,
                Some(record_descr),
                &[vable_opref, value],
                None,
                0,
            );
            // pyjitpl.py:980 upd.setfield(valuebox).  Cache stores the
            // Box identity (`value` OpRef); the intrinsic concrete
            // travels with the frontend value slot — `value`'s slot was
            // stamped at the calling record-site with `concrete` via
            // `set_opref_concrete`, so cache-hit sanity readers retrieve
            // it through `box_value(cached)`.
            let _ = concrete;
            self.heapcache_setfield_cached(vable_opref, field_index, value);
            return None;
        }
        // index = self._get_virtualizable_field_index(fielddescr)
        // self.metainterp.virtualizable_boxes[index] = valuebox
        let index = self
            .virtualizable_info
            .as_ref()
            .expect("vable_setfield: virtualizable_info missing")
            .static_field_by_descr(&fielddescr)
            .expect("vable_setfield: standard virtualizable field descr missing");
        // An unknown concrete still needs a stamp so the box carries the
        // "no concrete" marker (`Value::Ref(GcRef::NO_CONCRETE)`).
        let stored = concrete.unwrap_or(Value::Ref(majit_ir::GcRef::NO_CONCRETE));
        let overwritten = VableEntryWrite::of(self, index);
        self.set_virtualizable_entry_at(index, value, stored);
        // virtualizable.py write_box_at via MetaInterp.synchronize_virtualizable_at.
        self.synchronize_virtualizable_at(index);
        overwritten
    }

    /// Record a virtualizable field write with an explicit field descriptor.
    pub fn vable_setfield_descr(&mut self, vable_opref: OpRef, value: OpRef, descr: DescrRef) {
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], descr);
    }

    /// pyjitpl.py `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_r(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_r(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    /// `opimpl_getfield_vable_r` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    pub fn vable_getfield_ref_checked(
        &mut self,
        nonstandard: bool,
        cpu: &dyn crate::cpu::Cpu,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if nonstandard {
            // self.opimpl_getfield_gc_r(box, fielddescr) →
            // _opimpl_getfield_gc_any_pureornot (pyjitpl.py).
            let record_descr = self.vable_static_record_descr(&fielddescr);
            let field_index = record_descr.index();
            if let Some(cached) = self.heapcache_getfield_cached(vable_opref, field_index) {
                // pyjitpl.py:934-945 + :938-939 sanity check (ref arm):
                //     resvalue = executor.execute(cpu, mi, opnum, fielddescr, box)
                //     assert resvalue == upd.currfieldbox.getref_base()
                // `box_value(cached)` resolves the upstream
                // `currfieldbox.getref_base()` payload through the
                // full chain (const pool, standard-virtualizable
                // shadow, the frontend object's `value` field).
                let cached_value = self.box_value(cached);
                let expected_ref = match cached_value {
                    Some(Value::Ref(r)) => Some(r),
                    _ => None,
                };
                if let Some(cached_ref) = expected_ref
                    && vable_struct_ptr != 0
                    && let Some(Value::Ref(loaded)) =
                        self.field_sanity_load(vable_struct_ptr, &fielddescr, Type::Ref)
                {
                    assert_eq!(
                        loaded, cached_ref,
                        "_opimpl_getfield_gc_any_pureornot sanity \
                                 check (ref): loaded {:#x} != cached {:#x} \
                                 (field_index={field_index}, vable_struct_ptr=\
                                 {vable_struct_ptr:#x})",
                        loaded.0, cached_ref.0,
                    );
                }
                self.profiler().count_ops(
                    OpCode::GetfieldGcI,
                    crate::pyjitpl::counters::HEAPCACHED_OPS,
                );
                return (cached, cached_value);
            }
            // pyjitpl.py:949 upd.getfield_now_known(resbox) — `resbox`
            // carries `.getref_base()` payload; pair it with the
            // recorded opref so subsequent `box_value(op)` matches
            // RPython's executor-returned Box.  It is the funnel's
            // `resvalue`, so the load runs before the record.
            let live = self.field_live_value(vable_struct_ptr, vable_opref, &fielddescr, Type::Ref);
            let op = self.execute_and_record(
                Some(cpu),
                OpCode::GetfieldGcR,
                Some(record_descr),
                &[vable_opref],
                live,
                0,
            );
            self.heapcache_getfield_now_known(vable_opref, field_index, op);
            return (op, live);
        }
        // pyjitpl.py:1170,1177,1184,1228 — the reader asserts the shadow is
        // coherent; it never repairs it.
        self.check_synchronized_virtualizable();
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index
            && let Some((op, value)) = self.vable_box_result_at(idx)
        {
            return (op, value);
        }
        let op = self.execute_and_record(
            Some(cpu),
            OpCode::GetfieldGcR,
            Some(fielddescr),
            &[vable_opref],
            None,
            0,
        );
        (op, None)
    }

    /// Record a virtualizable ref field read with an explicit field descriptor.
    pub fn vable_getfield_ref_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        // pyjitpl.py `gen_store_back_in_vable`. No live load reaches this
        // helper, so the funnel's `resvalue` is `None`; that closes the fold
        // on its own and no cpu is needed to read a field nobody folds.
        self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(descr),
            &[vable_opref],
            None,
            0,
        )
    }

    /// pyjitpl.py `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_f(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_f(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    /// `opimpl_getfield_vable_f` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    pub fn vable_getfield_float_checked(
        &mut self,
        nonstandard: bool,
        cpu: &dyn crate::cpu::Cpu,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fielddescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if nonstandard {
            // self.opimpl_getfield_gc_f(box, fielddescr) →
            // _opimpl_getfield_gc_any_pureornot (pyjitpl.py).
            let record_descr = self.vable_static_record_descr(&fielddescr);
            let field_index = record_descr.index();
            if let Some(cached) = self.heapcache_getfield_cached(vable_opref, field_index) {
                // pyjitpl.py:941-945 sanity check (float arm):
                //     resvalue = executor.execute(cpu, mi, opnum, fielddescr, box)
                //     assert ConstFloat(resvalue).same_constant(
                //         upd.currfieldbox.constbox())
                // ConstFloat.same_constant compares via
                // longlong.extract_bits (history.py:283-294); pyre's
                // Value::Eq for Float uses to_bits — bit-identical.
                // `box_value(cached)` resolves the upstream
                // `currfieldbox.constbox()` payload.
                let cached_value = self.box_value(cached);
                let expected_float = match cached_value {
                    Some(Value::Float(f)) => Some(f),
                    _ => None,
                };
                if let Some(cached_float) = expected_float
                    && vable_struct_ptr != 0
                    && let Some(Value::Float(loaded)) =
                        self.field_sanity_load(vable_struct_ptr, &fielddescr, Type::Float)
                {
                    assert_eq!(
                        loaded.to_bits(),
                        cached_float.to_bits(),
                        "_opimpl_getfield_gc_any_pureornot sanity \
                                 check (float): loaded {loaded} != cached \
                                 {cached_float} (field_index={field_index}, \
                                 vable_struct_ptr={vable_struct_ptr:#x})"
                    );
                }
                self.profiler().count_ops(
                    OpCode::GetfieldGcI,
                    crate::pyjitpl::counters::HEAPCACHED_OPS,
                );
                return (cached, cached_value);
            }
            // pyjitpl.py:949 upd.getfield_now_known(resbox) — pair the
            // float payload with the recorded opref so subsequent
            // `box_value(op)` matches RPython's executor-returned Box.  It is
            // the funnel's `resvalue`, so the load runs before the record.
            let live =
                self.field_live_value(vable_struct_ptr, vable_opref, &fielddescr, Type::Float);
            let op = self.execute_and_record(
                Some(cpu),
                OpCode::GetfieldGcF,
                Some(record_descr),
                &[vable_opref],
                live,
                0,
            );
            self.heapcache_getfield_now_known(vable_opref, field_index, op);
            return (op, live);
        }
        // pyjitpl.py:1170,1177,1184,1228 — the reader asserts the shadow is
        // coherent; it never repairs it.
        self.check_synchronized_virtualizable();
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index
            && let Some((op, value)) = self.vable_box_result_at(idx)
        {
            return (op, value);
        }
        let op = self.execute_and_record(
            Some(cpu),
            OpCode::GetfieldGcF,
            Some(fielddescr),
            &[vable_opref],
            None,
            0,
        );
        (op, None)
    }

    /// Standard virtualizable array item read (int).
    /// `array_field_offset` identifies which array field, `item_index` is the element index.
    /// If standard boxes are active, reads from the flat box array directly.
    pub fn vable_getarrayitem_int_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index)
            && let Some((op, value)) = self.vable_box_result_at(flat_idx)
        {
            return (op, value);
        }
        let index = self.const_int(item_index as i64);
        // pyjitpl.py execute_and_record: execute the load then record with
        // resvalue. `GETARRAYITEM_GC_*` is not descr-pure, so the funnel
        // does not consult a cpu for the fold.
        let live = self.array_live_value(array_opref, item_index as i64, &adescr, Type::Int);
        let op = self.execute_and_record(
            None,
            OpCode::GetarrayitemGcI,
            Some(adescr),
            &[array_opref, index],
            live,
            0,
        );
        (op, live)
    }

    /// pyjitpl.py `_get_arrayitem_vable_index(pc, arrayfielddescr, indexbox)`.
    ///
    /// ```text
    ///  def _get_arrayitem_vable_index(self, pc, arrayfielddescr, indexbox):
    ///      indexbox = self.implement_guard_value(indexbox, pc)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
    ///      index = indexbox.getint()
    ///      assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
    ///      return vinfo.get_index_in_array(virtualizable, arrayindex, index)
    /// ```
    fn get_arrayitem_vable_index(
        &mut self,
        pc: usize,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: &DescrRef,
    ) -> Option<usize> {
        // `MAJIT_VABLE_IDX_PROBE`: does a non-constant index ever arrive here?
        //
        // Prints on BOTH branches deliberately. A probe that prints only on
        // the branch it is hunting cannot distinguish "never taken" from
        // "never reached" — both read as silence.
        //
        // Reading the counts. This is a SHARED CALLEE, and its callers do
        // not agree about constness:
        //   - `pyjitpl/dispatch.rs` hoists `implement_guard_value` at all six
        //     of its sites, so every index arriving from there is CONST *by
        //     construction* and carries no information.
        //   - `jitcode_dispatch/specialize.rs` passes a `const_int` literal,
        //     likewise CONST by construction.
        //   - `jitcode_dispatch/vable_ops.rs` passes a raw int register,
        //     gated only on the index having a recorded *concrete value*
        //     (`concrete_of_opref`), which a non-constant OpRef can satisfy.
        //     That is the only family whose constness is an open question.
        // So `NONCONST > 0` is conclusive, but `NONCONST == 0` is NOT
        // evidence that the open family is constant — it is equally
        // consistent with that family never being reached, since the
        // constant-by-construction callers dilute the reading. To attribute,
        // pair this with a probe at the `vable_ops.rs` index read itself.
        //
        // Measured readings: dualtape CONST=7204 NONCONST=0 (all from the
        // hoisted dispatch.rs family, i.e. the hoist working as designed).
        if crate::vable_idx_probe_enabled() {
            let constness = if index.is_constant() {
                "CONST"
            } else {
                "NONCONST"
            };
            eprintln!("[vable-idx-probe] {constness} pc={pc} value={index_runtime_value}");
        }
        // `indexbox = self.implement_guard_value(indexbox, pc)` runs on the
        // `MIFrame` that owns the framestack (`pyjitpl.py
        // _get_arrayitem_vable_index`). Dispatch hoists it through
        // `implement_guard_value` / `record_state_guard`; the walker through
        // `walker_implement_guard_value`. This helper only flattens the
        // already-promoted index.
        let item_index = usize::try_from(index_runtime_value).ok()?;
        // arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
        // assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
        // return vinfo.get_index_in_array(virtualizable, arrayindex, index)
        self.vable_array_flat_index(fdescr, item_index)
    }

    /// pyjitpl.py `_opimpl_getarrayitem_vable(box, indexbox, fdescr, adescr, pc)`
    /// (int variant via `opimpl_getarrayitem_vable_i = _opimpl_getarrayitem_vable`).
    ///
    /// ```text
    ///  def _opimpl_getarrayitem_vable(self, box, indexbox, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          ...
    ///          return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    /// `_opimpl_getarrayitem_vable` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter list mirrors the corresponding RPython metainterpreter routine plus the hoisted branch decision; grouping arguments into a Rust-only context object would obscure line-by-line parity"
    )]
    pub fn vable_getarrayitem_int_checked(
        &mut self,
        nonstandard: bool,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        let concrete = self.concrete_of_opref(vable_opref);
        if nonstandard {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
            let array_opref = self.nonstandard_vable_array_base(vable_opref, &fdescr);
            let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
            return (
                self.vable_getarrayitem_int_descr(array_opref, index, adescr),
                None,
            );
        }
        // pyjitpl.py:1170,1177,1184,1228 — the reader asserts the shadow is
        // coherent; it never repairs it.
        self.check_synchronized_virtualizable();
        // index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
        // return self.metainterp.virtualizable_boxes[index]
        if let Some(flat_idx) =
            self.get_arrayitem_vable_index(pc, index, index_runtime_value, &fdescr)
            && let Some((op, value)) = self.vable_box_result_at(flat_idx)
        {
            return (op, value);
        }
        // Fallback: vable layout missing — go through getfield + arrayitem.
        // `stamp_vable_array_base` supplies the concrete after the record, so
        // the funnel sees no `resvalue` and cannot fold.
        let array_opref = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(fdescr.clone()),
            &[vable_opref],
            None,
            0,
        );
        self.stamp_vable_array_base(array_opref, concrete, &fdescr);
        let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_int_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_int_descr(array_opref, index, adescr),
                None,
            )
        }
    }

    /// Stamp the array-base box a `_opimpl_getarrayitem_vable` fallback reads
    /// through. pyjitpl.py reaches the array with `opimpl_getfield_gc_r`,
    /// which executes the load before recording it, so the box it hands on
    /// carries the loaded pointer. A base left symbolic propagates "no
    /// concrete" into every element read taken through it.
    /// Items pointer of an `EmbeddedArray` container.
    ///
    /// `bhimpl_getarrayitem_vable_*` loads the array field, then the data
    /// pointer inside that container. A direct `getarrayitem` on the field
    /// value indexes the container header (`Vec`'s length word).
    fn vable_embedded_items_base(&mut self, container: OpRef, array_index: usize) -> OpRef {
        let Some(info) = self.virtualizable_info.clone() else {
            return container;
        };
        let Some(array) = info.array_fields.get(array_index) else {
            return container;
        };
        let crate::virtualizable::VableArrayStorage::EmbeddedArray { ptr_offset } = array.storage
        else {
            return container;
        };
        let word = std::mem::size_of::<usize>();
        let field = std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
            0,
            ptr_offset,
            word,
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Pointer,
            "buf".to_string(),
            "buf",
        ));
        let mut parent = majit_ir::descr::SimpleSizeDescr::new(0, word * 3, 0);
        parent.set_gc_managed(false);
        parent.set_headerless(true);
        let parent = parent.with_all_fielddescrs(vec![field.clone()]);
        let parent: majit_ir::DescrRef = std::sync::Arc::new(parent);
        field.set_parent_descr(&parent);
        // `get_parent_descr` upgrades a `Weak`. The field descr stored on the
        // op is the only other owner, so the parent has to stay alive itself.
        Self::keep_embedded_parent(parent);
        self.record_op_with_descr(OpCode::GetfieldGcR, &[container], field)
    }

    fn vable_embedded_items_base_descr(&mut self, container: OpRef, fdescr: &DescrRef) -> OpRef {
        let Some(info) = self.virtualizable_info.clone() else {
            return container;
        };
        let Some(array_index) = info.array_field_by_descr(fdescr) else {
            return container;
        };
        self.vable_embedded_items_base(container, array_index)
    }

    fn keep_embedded_parent(parent: majit_ir::DescrRef) {
        use std::sync::Mutex;
        static KEPT: Mutex<Vec<majit_ir::DescrRef>> = Mutex::new(Vec::new());
        KEPT.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(parent);
    }

    fn stamp_vable_array_base(&mut self, op: OpRef, vable: Option<Value>, fdescr: &DescrRef) {
        let Some(Value::Ref(vable_ref)) = vable else {
            return;
        };
        let Some(vable_ptr) = live_gc_ptr(vable_ref) else {
            return;
        };
        if let Some(base) = self.field_sanity_load(vable_ptr, fdescr, Type::Ref) {
            self.set_opref_concrete(op, base);
        }
    }

    /// Standard virtualizable array item read (ref).
    pub fn vable_getarrayitem_ref_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index)
            && let Some((op, value)) = self.vable_box_result_at(flat_idx)
        {
            return (op, value);
        }
        let index = self.const_int(item_index as i64);
        // pyjitpl.py execute_and_record: execute the load then record with
        // resvalue. `GETARRAYITEM_GC_*` is not descr-pure, so the funnel
        // does not consult a cpu for the fold.
        let live = self.array_live_value(array_opref, item_index as i64, &adescr, Type::Ref);
        let op = self.execute_and_record(
            None,
            OpCode::GetarrayitemGcR,
            Some(adescr),
            &[array_opref, index],
            live,
            0,
        );
        (op, live)
    }

    /// pyjitpl.py `_opimpl_getarrayitem_vable` — ref variant.
    /// `_opimpl_getarrayitem_vable` ref body with the
    /// `_nonstandard_virtualizable` decision already taken by the caller
    /// (`begin` + framestack `implement_guard_value` + `commit`).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter list mirrors the corresponding RPython metainterpreter routine plus the hoisted branch decision; grouping arguments into a Rust-only context object would obscure line-by-line parity"
    )]
    pub fn vable_getarrayitem_ref_checked(
        &mut self,
        nonstandard: bool,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        let concrete = self.concrete_of_opref(vable_opref);
        if crate::vable_read_probe_enabled() {
            eprintln!(
                "[vable-read-probe] enter pc={pc} nonstandard={nonstandard} \
                 index_value={index_runtime_value} vable_concrete={} boxes={:?} lengths={:?} \
                 vable_opref={vable_opref:?} standard_box={:?} hc_nonstandard={} \
                 heap_ptr=0x{:x} shadow_concrete={:?}",
                concrete.is_some(),
                self.virtualizable_boxes.as_ref().map(Vec::len),
                self.virtualizable_array_lengths.as_deref(),
                self.standard_virtualizable_box(),
                self.heap_cache()
                    .is_known_nonstandard_virtualizable(vable_opref),
                self.diag_virtualizable_heap_ptr(),
                self.standard_virtualizable_concrete(),
            );
        }
        if nonstandard {
            // `_do_getarrayitem_gc_any` returns the heapcache box on a hit and
            // records nothing. A fresh `GetarrayitemGcR` is what
            // `optimize_getarrayitem_gc` folds onto the virtual array's entry
            // item, discarding the store the cache is holding.
            let record_descr = self.vable_array_record_descr(&fdescr);
            if let Some(base) = self.heapcache_getfield_cached(vable_opref, record_descr.index())
                && let Some(elem) = self.heapcache_getarrayitem(base, index, adescr.index())
            {
                self.profiler().count_ops(
                    OpCode::GetarrayitemGcR,
                    crate::pyjitpl::counters::HEAPCACHED_OPS,
                );
                let cached = self.box_value(elem);
                // `_do_getarrayitem_gc_any` sanity check: the current array
                // value must be what the cache thinks it is.
                if let Some(Value::Ref(base_ref)) = self.concrete_of_opref(base)
                    && let Some(base_ptr) = live_gc_ptr(base_ref)
                    && let Some(cached) = cached
                {
                    let resvalue =
                        self.array_sanity_load(base_ptr, index_runtime_value, &adescr, Type::Ref);
                    assert!(
                        resvalue.is_none_or(|v| v == cached),
                        "assertion in GETARRAYITEM_GC_R failed: {resvalue:?} != {cached:?}"
                    );
                }
                return (elem, cached);
            }
            let fwd = self.nonstandard_vable_element_concrete(
                vable_opref,
                &fdescr,
                index,
                adescr.index(),
            );
            let array_opref = self.nonstandard_vable_array_base(vable_opref, &fdescr);
            let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
            let item = self.vable_getarrayitem_ref_descr(array_opref, index, adescr);
            if crate::vable_read_probe_enabled() {
                eprintln!(
                    "[vable-read-probe] exit=nonstandard concrete={}",
                    fwd.is_some()
                );
            }
            if let Some(v) = fwd {
                self.set_opref_concrete(item, v);
                return (item, Some(v));
            }
            return (item, None);
        }
        let flat_idx = self.get_arrayitem_vable_index(pc, index, index_runtime_value, &fdescr);
        let entry = flat_idx.and_then(|flat_idx| self.vable_box_result_at(flat_idx));
        if crate::vable_read_probe_enabled() {
            eprintln!(
                "[vable-read-probe] exit=standard flat_idx={flat_idx:?} entry={} concrete={}",
                entry.is_some(),
                entry.is_some_and(|(_, value)| value.is_some()),
            );
        }
        if let Some((op, value)) = entry {
            return (op, value);
        }
        // `stamp_vable_array_base` supplies the concrete after the record, so
        // the funnel sees no `resvalue` and cannot fold.
        let array_opref = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(fdescr.clone()),
            &[vable_opref],
            None,
            0,
        );
        self.stamp_vable_array_base(array_opref, concrete, &fdescr);
        let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
        let result = if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_ref_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_ref_descr(array_opref, index, adescr),
                None,
            )
        };
        if crate::vable_read_probe_enabled() {
            eprintln!(
                "[vable-read-probe] exit=fallback base_concrete={} concrete={}",
                concrete.is_some(),
                result.1.is_some(),
            );
        }
        result
    }

    /// Standard virtualizable array item read (float).
    pub fn vable_getarrayitem_float_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index)
            && let Some((op, value)) = self.vable_box_result_at(flat_idx)
        {
            return (op, value);
        }
        let index = self.const_int(item_index as i64);
        // pyjitpl.py execute_and_record: execute the load then record with
        // resvalue. `GETARRAYITEM_GC_*` is not descr-pure, so the funnel
        // does not consult a cpu for the fold.
        let live = self.array_live_value(array_opref, item_index as i64, &adescr, Type::Float);
        let op = self.execute_and_record(
            None,
            OpCode::GetarrayitemGcF,
            Some(adescr),
            &[array_opref, index],
            live,
            0,
        );
        (op, live)
    }

    /// pyjitpl.py `_opimpl_getarrayitem_vable` — float variant.
    /// `_opimpl_getarrayitem_vable` float body with the
    /// `_nonstandard_virtualizable` decision already taken by the caller
    /// (`begin` + framestack `implement_guard_value` + `commit`).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter list mirrors the corresponding RPython metainterpreter routine plus the hoisted branch decision; grouping arguments into a Rust-only context object would obscure line-by-line parity"
    )]
    pub fn vable_getarrayitem_float_checked(
        &mut self,
        nonstandard: bool,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Option<Value>) {
        let concrete = self.concrete_of_opref(vable_opref);
        if nonstandard {
            let array_opref = self.nonstandard_vable_array_base(vable_opref, &fdescr);
            let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
            return (
                self.vable_getarrayitem_float_descr(array_opref, index, adescr),
                None,
            );
        }
        if let Some(flat_idx) =
            self.get_arrayitem_vable_index(pc, index, index_runtime_value, &fdescr)
            && let Some((op, value)) = self.vable_box_result_at(flat_idx)
        {
            return (op, value);
        }
        // `stamp_vable_array_base` supplies the concrete after the record, so
        // the funnel sees no `resvalue` and cannot fold.
        let array_opref = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(fdescr.clone()),
            &[vable_opref],
            None,
            0,
        );
        self.stamp_vable_array_base(array_opref, concrete, &fdescr);
        let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_float_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_float_descr(array_opref, index, adescr),
                None,
            )
        }
    }

    /// Standard virtualizable array item write at a known flat slot index.
    /// `item_index` is the element index within the array described by `fdescr`.
    pub fn vable_setarrayitem_vable(
        &mut self,
        fdescr: &DescrRef,
        item_index: usize,
        value: OpRef,
        concrete: Value,
    ) {
        let flat_idx = self
            .vable_array_flat_index(fdescr, item_index)
            .expect("vable_setarrayitem_vable: standard virtualizable array slot missing");
        self.set_virtualizable_entry_at(flat_idx, value, concrete);
        // pyjitpl.py MIFrame._opimpl_setarrayitem_vable →
        // virtualizable.py write_box_at.
        self.synchronize_virtualizable_at(flat_idx);
    }

    /// pyjitpl.py `_opimpl_setarrayitem_vable(box, indexbox, valuebox, fdescr, adescr, pc)`.
    ///
    /// `VableArrayStore::OutOfVable` means the promoted index did not resolve
    /// to a standard virtualizable slot: it was negative, or `fdescr` is not
    /// one of the virtualizable's array fields. Nothing was stored.
    /// `VableArrayStore::Stored(Some(write))` is a standard-leg store; `write`
    /// identifies the overwritten shadow slot and its prior box and value so a
    /// caller capturing a promote guard can roll it back during the capture.
    /// `VableArrayStore::Stored(None)` means the non-standard leg recorded a
    /// plain `SETARRAYITEM_GC`, with no virtualizable shadow slot to roll back.
    ///
    /// The three `MetaInterp::opimpl_setarrayitem_vable_*` wrappers deliberately
    /// assert `Stored`: `_get_arrayitem_vable_index` in
    /// `rpython/jit/metainterp/pyjitpl.py` asserts that the index is within
    /// the virtualizable array, making `OutOfVable` an invariant violation on
    /// that path. Graceful handling belongs to dispatcher/walker callers that
    /// have a trace to abort.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    /// The array half of `virtualizable.py write_boxes`, emitted into the trace
    /// for one array field of the STANDARD virtualizable.
    ///
    /// `pyjitpl.py synchronize_virtualizable` runs that write-back after every
    /// vable store, but only against the recording-time virtualizable: upstream
    /// readers of a virtualizable array are traced through and read the boxes,
    /// so the compiled trace never needs the array itself to be current.  A
    /// consumer that reads the array at run time instead needs the same writes
    /// emitted, which is what this records.
    ///
    /// `items` is `(element index, box)` pairs; only the listed slots are
    /// written, so a caller covering a sub-range of the array leaves the rest
    /// alone.  The emission shape is `gen_store_back_in_vable`'s array loop —
    /// one `getfield_gc_r` of `array_pointer_field_descr` followed by a
    /// `setarrayitem_gc` per item under `array_item_descr`.  Neither the token
    /// store nor `forced_virtualizable` is touched: this writes the image out,
    /// it does not force the virtualizable.
    ///
    /// The shadow is left alone — it already holds these values and stays
    /// authoritative for the rest of the trace.
    pub fn vable_array_region_write_back(
        &mut self,
        vable_opref: OpRef,
        array_index: usize,
        items: &[(i64, OpRef)],
    ) -> bool {
        let Some(info) = self.virtualizable_info.clone() else {
            return false;
        };
        if array_index >= info.array_fields.len() {
            return false;
        }
        let field_descr = info.array_pointer_field_descr(array_index);
        let array_descr = info.array_item_descr(array_index);
        let array_opref = self.vable_getfield_ref_descr(vable_opref, field_descr.clone());
        // `executor.execute` for the read: the array base has to carry its
        // concrete half, or the consumer below reaches the backend with an
        // operand no producer answers for.  Same step every other recorded
        // vable array-base read takes.
        let vable_concrete = self.concrete_of_opref(vable_opref);
        self.stamp_vable_array_base(array_opref, vable_concrete, &field_descr);
        self.heapcache_getfield_now_known(vable_opref, field_descr.index(), array_opref);
        let array_opref = self.vable_embedded_items_base(array_opref, array_index);
        for &(item_index, value) in items {
            let index = self.const_int(item_index);
            self.execute_and_record(
                None,
                OpCode::SetarrayitemGc,
                Some(array_descr.clone()),
                &[array_opref, index, value],
                None,
                0,
            );
        }
        true
    }

    /// `GETARRAYITEM_GC_R` of one locals slot. Does not consult the
    /// virtualizable shadow: the caller has already decided this slot is not
    /// a live vable box and must be read off the frame object.
    pub fn read_gc_array_item_ref(
        &mut self,
        array_opref: OpRef,
        item_index: i64,
        adescr: DescrRef,
    ) -> OpRef {
        let index = self.const_int(item_index);
        let live = self.array_live_value(array_opref, item_index, &adescr, Type::Ref);
        self.execute_and_record(
            None,
            OpCode::GetarrayitemGcR,
            Some(adescr),
            &[array_opref, index],
            live,
            0,
        )
    }

    /// `_opimpl_setarrayitem_vable` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn vable_setarrayitem_checked(
        &mut self,
        nonstandard: bool,
        pc: usize,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
        value: OpRef,
        concrete: Value,
        live_null_push: bool,
    ) -> VableArrayStore {
        if nonstandard {
            let array_opref = self.nonstandard_vable_array_base(vable_opref, &fdescr);
            let array_opref = self.vable_embedded_items_base_descr(array_opref, &fdescr);
            self.execute_setarrayitem_gc(array_opref, index, value, adescr);
            return VableArrayStore::Stored(None);
        }
        // index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
        // self.metainterp.virtualizable_boxes[index] = valuebox
        // self.metainterp.synchronize_virtualizable_at(index)
        let Some(flat_idx) =
            self.get_arrayitem_vable_index(pc, index, index_runtime_value, &fdescr)
        else {
            return VableArrayStore::OutOfVable;
        };
        let overwritten = VableEntryWrite::of(self, flat_idx);
        // `set_virtualizable_entry_at` → `try_set_opref_concrete` can intern
        // and collect. Pin the Ref before that call and store the live word
        // (`store_reconstructed_callee_array_image`).
        let pinned = match concrete {
            Value::Ref(gc) if !gc.is_null() && gc != GcRef::NO_CONCRETE => {
                let live = majit_gc::gc_current_object_address(gc.0);
                if live == 0 {
                    None
                } else {
                    Some(majit_gc::shadow_stack::OwnerRootGuard::new(GcRef(live)))
                }
            }
            _ => None,
        };
        // Stamp the pinned word, not the pre-pin copy: `box_value` is
        // the GETARRAYITEM_GC_R heapcache sanity's cache side.
        let stamp = pinned
            .as_ref()
            .map(|g| Value::Ref(g.get()))
            .unwrap_or(concrete);
        self.set_virtualizable_entry_at(flat_idx, value, stamp);
        if live_null_push
            && matches!(concrete, Value::Ref(r) if r.is_null())
            && let Some(live_null_slots) = self.virtualizable_live_null_slots.as_mut()
        {
            live_null_slots[flat_idx] = true;
        }
        // pyjitpl.py MIFrame._opimpl_setarrayitem_vable →
        // virtualizable.py write_box_at. A Const OpRef's
        // `inline_const_to_value` is not a root; write the live word the
        // caller pinned across `walker_promote_vable_array_index`.
        if matches!(concrete, Value::Ref(_)) {
            let live = pinned
                .as_ref()
                .map(|g| Value::Ref(g.get()))
                .unwrap_or(stamp);
            self.write_virtualizable_heap_value_at(flat_idx, live);
        } else {
            self.synchronize_virtualizable_at(flat_idx);
        }
        VableArrayStore::Stored(overwritten)
    }

    /// pyjitpl.py `opimpl_arraylen_gc(arraybox, arraydescr)`.
    ///
    /// ```text
    ///  def opimpl_arraylen_gc(self, arraybox, arraydescr):
    ///      lengthbox = self.metainterp.heapcache.arraylen(arraybox)
    ///      if lengthbox is None:
    ///          lengthbox = self.execute_with_descr(rop.ARRAYLEN_GC,
    ///                                              arraydescr, arraybox)
    ///          self.metainterp.heapcache.arraylen_now_known(arraybox, lengthbox)
    ///      else:
    ///          self.metainterp.staticdata.profiler.count_ops(
    ///              rop.ARRAYLEN_GC, Counters.HEAPCACHED_OPS)
    ///      return lengthbox
    /// ```
    ///
    /// `concrete` is the runtime length the `execute_with_descr` box carries
    /// (`arraylen_sanity_load`); `None` leaves the recorded OpRef unstamped
    /// (the cpu is unwired or the descr lacks a lendescr) and, by
    /// `execute_and_record`'s rule for a missing concrete, blocks the fold.
    ///
    /// `ARRAYLEN_GC` is always-pure, so a constant array base with a concrete
    /// length folds and records nothing. `last_exc_value` is not consulted for
    /// an always-pure opcode; only the overflow arm reads it.
    pub fn opimpl_arraylen_gc(
        &mut self,
        cpu: &dyn crate::cpu::Cpu,
        array_opref: OpRef,
        arraydescr: DescrRef,
        concrete: Option<Value>,
        last_exc_value: i64,
    ) -> OpRef {
        if let Some(cached_len) = self.heap_cache().arraylen(array_opref) {
            // pyjitpl.py:763 profiler.count_ops(rop.ARRAYLEN_GC, HEAPCACHED_OPS).
            self.profiler()
                .count_ops(OpCode::ArraylenGc, crate::pyjitpl::counters::HEAPCACHED_OPS);
            return cached_len;
        }
        // `execute_nonspec_const`'s ARRAYLEN_GC row reads the length through
        // `ArrayDescr::len_descr` and fails loud without one.  A caller that
        // supplies a concrete (the unit test, or a path that did not go
        // through `arraylen_sanity_load`) still needs the same withhold, so
        // the funnel records rather than entering a fold it cannot complete.
        let concrete = concrete.filter(|_| {
            arraydescr
                .as_array_descr()
                .is_some_and(|a| a.len_descr().is_some())
        });
        // pyjitpl.py `opimpl_arraylen_gc` miss. The funnel owns the OPS /
        // RECORDED_OPS pairing this leg used to count by hand.
        let len = self.execute_and_record(
            Some(cpu),
            OpCode::ArraylenGc,
            Some(arraydescr),
            &[array_opref],
            concrete,
            last_exc_value,
        );
        // pyjitpl.py:761 heapcache.arraylen_now_known(arraybox, lengthbox).
        // `HeapCache::arraylen_now_known` returns early for a constant array,
        // so the fold path self-cancels rather than depositing under a key the
        // cache does not track.
        self.heap_cache_mut().arraylen_now_known(array_opref, len);
        len
    }

    /// Live length for a nonstandard `opimpl_arraylen_vable` cache miss.
    ///
    /// `opimpl_arraylen_gc` reads `heapcache.arraylen` first and, on a miss,
    /// `execute_with_descr(ARRAYLEN_GC)`. The array pointer is the getfield
    /// box's value when that load was executed, otherwise the field load
    /// `opimpl_getfield_gc_r` would have performed on the live frame
    /// (`vable_struct_ptr`, or the frame box's own concrete when the
    /// register shadow was not passed in).
    fn executed_nonstandard_arraylen(
        &self,
        vable_struct_ptr: i64,
        vable_opref: OpRef,
        fdescr: &DescrRef,
        array_opref: OpRef,
        adescr: &DescrRef,
    ) -> Option<Value> {
        let array_ptr = match self.concrete_of_opref(array_opref) {
            Some(Value::Ref(r)) => live_gc_ptr(r),
            _ => None,
        };
        let array_ptr = array_ptr.or_else(|| {
            let frame_ptr = if vable_struct_ptr != 0 {
                Some(vable_struct_ptr)
            } else {
                match self.concrete_of_opref(vable_opref) {
                    Some(Value::Ref(r)) => live_gc_ptr(r),
                    _ => None,
                }
            }?;
            let record_descr = self.vable_array_record_descr(fdescr);
            match self.field_sanity_load(frame_ptr, &record_descr, Type::Ref) {
                Some(Value::Ref(r)) => live_gc_ptr(r),
                _ => None,
            }
        })?;
        self.arraylen_sanity_load(array_ptr, adescr)
    }

    /// pyjitpl.py `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_arraylen_vable(self, box, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          return self.opimpl_arraylen_gc(arraybox, adescr)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[fdescr]
    ///      result = vinfo.get_array_length(virtualizable, arrayindex)
    ///      return ConstInt(result)
    /// ```
    /// `opimpl_arraylen_vable` body with the `_nonstandard_virtualizable`
    /// decision already taken by the caller (`begin` + framestack
    /// `implement_guard_value` + `commit`).
    pub fn vable_arraylen_vable_checked(
        &mut self,
        nonstandard: bool,
        cpu: &dyn crate::cpu::Cpu,
        vable_opref: OpRef,
        vable_struct_ptr: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> OpRef {
        if nonstandard {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_arraylen_gc(arraybox, adescr)
            //
            // `nonstandard_vable_array_base` is that getfield: a hit
            // forwards the box `heapcache.new_array` stored the const
            // length on, and `opimpl_arraylen_gc` returns it. After the
            // frame escapes, `invalidate_caches_for_escaped` drops the
            // getfield, so the miss has to `execute_with_descr(ARRAYLEN_GC)`
            // and stamp the length it read. An unstamped `ArraylenGc`
            // leaves `goto_if_not` with no int (`GotoIfNotValueNotConcrete`).
            let array_opref = self.nonstandard_vable_array_base(vable_opref, &fdescr);
            let len_concrete = if self.heap_cache().arraylen(array_opref).is_some() {
                None
            } else {
                self.executed_nonstandard_arraylen(
                    vable_struct_ptr,
                    vable_opref,
                    &fdescr,
                    array_opref,
                    &adescr,
                )
            };
            return self.opimpl_arraylen_gc(cpu, array_opref, adescr, len_concrete, 0);
        }
        // arrayindex = vinfo.array_field_by_descrs[fdescr]
        // result = vinfo.get_array_length(virtualizable, arrayindex)
        // return ConstInt(result)
        if let (Some(info), Some(lengths)) =
            (&self.virtualizable_info, &self.virtualizable_array_lengths)
            && let Some(array_idx) = info.array_field_by_descr(&fdescr)
            && let Some(&length) = lengths.get(array_idx)
        {
            return self.const_int(length as i64);
        }
        // Fallback when the layout is unavailable. Neither leg has a
        // trace-time concrete, so both close the fold on `resvalue`.
        let array_opref = self.execute_and_record(
            None,
            OpCode::GetfieldGcR,
            Some(fdescr),
            &[vable_opref],
            None,
            0,
        );
        self.execute_and_record(
            None,
            OpCode::ArraylenGc,
            Some(adescr),
            &[array_opref],
            None,
            0,
        )
    }

    /// Address of item 0 of a virtualizable array field, as a constant int,
    /// and raise the raw-base escape signal.
    ///
    /// `None` when the array's layout cannot be resolved. That is not a
    /// recoverable miss to paper over with a zero or a recorded op: the walker
    /// really performs the residual call this address is an argument to, so an
    /// unresolved base would hand a live callee a wrong pointer and corrupt
    /// memory. The caller aborts the trace instead.
    ///
    /// The address is read from the live heap object rather than recorded as
    /// an operation because the trace does not survive: raising the escape
    /// signal here means the enclosing CALL_MAY_FORCE aborts with
    /// ABORT_ESCAPE before this constant can reach an optimizer or a backend.
    /// Returns the trace-side constant and the same address as a concrete, so
    /// the caller can stamp the destination register with both.
    pub fn vable_arraybase_vable(
        &mut self,
        vable_struct_ptr: i64,
        fdescr: DescrRef,
    ) -> Option<(OpRef, i64)> {
        if vable_struct_ptr == 0 {
            return None;
        }
        let info = self.virtualizable_info.as_ref()?;
        let array_idx = info.array_field_by_descr(&fdescr)?;
        let array = info.array_fields.get(array_idx)?;
        let base = unsafe {
            crate::virtualizable::bhimpl_arraybase_vable(vable_struct_ptr as *const u8, array)
        };
        if base.is_null() {
            return None;
        }
        // Raised only once the address is known to be real, so an aborted
        // resolution above cannot leave a signal behind for the next call.
        self.raw_vable_base_escape_pending = true;
        let addr = base as usize as i64;
        Some((self.const_int(addr), addr))
    }

    /// Consume the raw-base escape signal for the current residual call.
    pub(crate) fn take_raw_vable_base_escape(&mut self) -> bool {
        std::mem::take(&mut self.raw_vable_base_escape_pending)
    }

    /// Compute the flat index into virtualizable_boxes for an array element.
    /// Returns `None` if standard virtualizable is not active or the array field is unknown.
    ///
    /// `pyjitpl.py _get_arrayitem_vable_index` gates the same arithmetic on
    /// `assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)`,
    /// and that assert is load-bearing rather than documentary: the flat space
    /// runs `[static fields][array 0]..[array n-1][virtualizable identity]`, so
    /// an index one past the last array is not merely out of the array — it is
    /// the identity entry `virtualizable_boxes[-1]` that
    /// `_nonstandard_virtualizable` compares every later vable op against.
    /// Writing there silently retargets the whole shadow at whatever value the
    /// store carried. Answer `None` instead: the read paths fall back to a heap
    /// `GETARRAYITEM_GC_*` and the write path reports `VableArrayStore::
    /// OutOfVable`, which is the honest outcome for a slot the frame's
    /// `locals_cells_stack_w` does not have.
    fn vable_array_flat_index(&self, fdescr: &DescrRef, item_index: usize) -> Option<usize> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_ref()?;
        let array_idx = info.array_field_by_descr(fdescr)?;
        if item_index >= *lengths.get(array_idx)? {
            return None;
        }
        Some(info.get_index_in_array(array_idx, item_index, lengths))
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_int_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        // `GETARRAYITEM_GC_*` answers `is_pure_with_descr` false whatever its
        // descr says — an immutable GC array read is spelled with the
        // dedicated `_PURE` opcode — so the fold gate is shut and no cpu is
        // consulted.
        let live = match self.concrete_of_opref(index) {
            Some(Value::Int(item_index)) => {
                self.array_live_value(array_opref, item_index, &descr, Type::Int)
            }
            _ => None,
        };
        self.execute_and_record(
            None,
            OpCode::GetarrayitemGcI,
            Some(descr),
            &[array_opref, index],
            live,
            0,
        )
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_ref_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        // `GETARRAYITEM_GC_*` answers `is_pure_with_descr` false whatever its
        // descr says — an immutable GC array read is spelled with the
        // dedicated `_PURE` opcode — so the fold gate is shut and no cpu is
        // consulted.
        let live = match self.concrete_of_opref(index) {
            Some(Value::Int(item_index)) => {
                self.array_live_value(array_opref, item_index, &descr, Type::Ref)
            }
            _ => None,
        };
        self.execute_and_record(
            None,
            OpCode::GetarrayitemGcR,
            Some(descr),
            &[array_opref, index],
            live,
            0,
        )
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_float_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        // `GETARRAYITEM_GC_*` answers `is_pure_with_descr` false whatever its
        // descr says — an immutable GC array read is spelled with the
        // dedicated `_PURE` opcode — so the fold gate is shut and no cpu is
        // consulted.
        let live = match self.concrete_of_opref(index) {
            Some(Value::Int(item_index)) => {
                self.array_live_value(array_opref, item_index, &descr, Type::Float)
            }
            _ => None,
        };
        self.execute_and_record(
            None,
            OpCode::GetarrayitemGcF,
            Some(descr),
            &[array_opref, index],
            live,
            0,
        )
    }

    /// Record a virtualizable array item write with an explicit array descriptor.
    pub fn vable_setarrayitem_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        value: OpRef,
        descr: DescrRef,
    ) {
        self.record_op_with_descr(OpCode::SetarrayitemGc, &[array_opref, index, value], descr);
    }

    /// `execute_setarrayitem_gc(arraydescr, arraybox, indexbox, itembox)`
    /// (pyjitpl.py): record the `SETARRAYITEM_GC`, then publish the stored item
    /// to the heapcache.  `gen_store_back_in_vable` deliberately does NOT come
    /// through here — it records the op directly — so the raw recording form
    /// stays available as `vable_setarrayitem_descr`.
    ///
    /// The heapcache write is what keeps a later read of the same slot from
    /// forwarding the value the cache was seeded with: the element cache is the
    /// only thing a nonstandard virtualizable's reads consult for the stored
    /// box, so a store that skips it leaves every subsequent read answering the
    /// pre-store value.
    fn execute_setarrayitem_gc(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        value: OpRef,
        descr: DescrRef,
    ) {
        let descr_index = descr.index();
        // A store has no `resvalue` and `SETARRAYITEM_GC` is never pure, so
        // the funnel records it without consulting a cpu.
        self.execute_and_record(
            None,
            OpCode::SetarrayitemGc,
            Some(descr),
            &[array_opref, index, value],
            None,
            0,
        );
        self.heapcache_setarrayitem(array_opref, index, descr_index, value);
    }
}

#[cfg(test)]
#[allow(deprecated)] // test fixtures rebuild Op streams via OpRef::from_raw; production
// trace_ctx path has 0 OpRef::from_raw callers, so the deprecation gate
// narrows from crate-level to mod-level.
mod tests {
    use super::*;
    use crate::jit_state::JitState;
    use majit_backend::JitCellToken;
    use majit_ir::Type;

    /// LIVE at pc 0 so `generate_guard` can step back `SIZE_LIVE_OP`
    /// (`pyjitpl.py get_list_of_active_boxes`). Callers pass
    /// [`DUMMY_RESUME_PC`] as the promote's resumepc.
    const DUMMY_RESUME_PC: usize = majit_jitcode::liveness::OFFSET_SIZE + 1;

    fn dummy_framestack(ctx: &mut TraceCtx) -> crate::pyjitpl::MIFrameStack {
        let mut asm = crate::Assembler::new();
        let mut builder = crate::JitCodeBuilder::new();
        builder.live(&mut asm, &[], &[], &[]);
        let jitcode = std::sync::Arc::new(builder.finish());
        jitcode.set_index(0);
        let sd = std::sync::Arc::get_mut(&mut ctx.metainterp_sd)
            .expect("dummy_framestack: unique MetaInterpStaticData");
        sd.op_live = crate::jitcode::insns::BC_LIVE as i32;
        sd.liveness_info.set(asm.all_liveness().to_vec());
        crate::pyjitpl::MIFrameStack::new(crate::pyjitpl::MIFrame::new(jitcode, 0))
    }

    /// Test-side `MIFrame._nonstandard_virtualizable`: `begin` + framestack
    /// `implement_guard_value` + `commit`. PendingEq installs a LIVE dummy
    /// frame so the promote captures resume data.
    fn decide_nonstandard(ctx: &mut TraceCtx, vable: OpRef, fielddescr: &DescrRef) -> bool {
        match ctx.begin_nonstandard_virtualizable(DUMMY_RESUME_PC, vable, fielddescr) {
            NonstandardVable::Decided(n) => n,
            NonstandardVable::PendingEq {
                eqbox,
                isstandard,
                vable_opref,
                standard_box,
            } => {
                let mut frames = dummy_framestack(ctx);
                let promoted = crate::pyjitpl::implement_guard_value_on_frames(
                    ctx,
                    &mut frames,
                    eqbox,
                    isstandard,
                    DUMMY_RESUME_PC,
                );
                ctx.commit_nonstandard_virtualizable(
                    promoted,
                    vable_opref,
                    standard_box,
                    fielddescr,
                )
            }
        }
    }

    #[allow(dead_code)]
    extern "C" fn dummy_call_target() {}

    /// Test-side `self.metainterp.cpu` analog: implements the cache-hit
    /// load surface (`bh_getfield_gc_i/r/f`) plus the non-default
    /// Backend methods (compile_loop, compile_bridge, execute_token,
    /// invalidate_loop, get_latest_descr, get_latest_descr_arc,
    /// get_int/ref/float_value) as panics — these tests never exercise
    /// compilation/execution, only the sanity-check load.
    struct SanityTestCpu {
        int_value: i64,
        ref_value: majit_ir::GcRef,
        float_value: f64,
    }
    impl majit_backend::Backend for SanityTestCpu {
        fn compile_loop(
            &mut self,
            _inputargs: &[majit_ir::InputArgRc],
            _ops: &[majit_ir::OpRc],
            _token: &majit_backend::JitCellToken,
        ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
            unimplemented!("SanityTestCpu::compile_loop")
        }
        fn compile_bridge(
            &mut self,
            _fail_descr: &dyn majit_ir::FailDescr,
            _inputargs: &[majit_ir::InputArgRc],
            _ops: &[majit_ir::OpRc],
            _original_token: &majit_backend::JitCellToken,
            _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
        ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
            unimplemented!("SanityTestCpu::compile_bridge")
        }
        fn execute_token(
            &self,
            _token: &majit_backend::JitCellToken,
            _args: &[majit_ir::Value],
        ) -> majit_backend::DeadFrame {
            unimplemented!("SanityTestCpu::execute_token")
        }
        fn get_latest_descr<'a>(
            &self,
            _frame: &'a majit_backend::DeadFrame,
        ) -> &'a dyn majit_ir::FailDescr {
            unimplemented!("SanityTestCpu::get_latest_descr")
        }
        fn get_latest_descr_arc(
            &self,
            _frame: &majit_backend::DeadFrame,
        ) -> std::sync::Arc<dyn majit_ir::descr::Descr> {
            unimplemented!("SanityTestCpu::get_latest_descr_arc")
        }
        fn get_int_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> i64 {
            unimplemented!("SanityTestCpu::get_int_value")
        }
        fn get_value_direct(&self, frame: &majit_backend::DeadFrame, slot: usize) -> i64 {
            // SanityTestCpu's slot space is the dense fail-value vector.
            self.get_int_value(frame, slot)
        }
        fn get_float_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> f64 {
            unimplemented!("SanityTestCpu::get_float_value")
        }
        fn get_ref_value(
            &self,
            _frame: &majit_backend::DeadFrame,
            _index: usize,
        ) -> majit_ir::GcRef {
            unimplemented!("SanityTestCpu::get_ref_value")
        }
        fn invalidate_loop(&self, _token: &majit_backend::JitCellToken) {
            unimplemented!("SanityTestCpu::invalidate_loop")
        }
        fn bh_getfield_gc_i(
            &self,
            _struct_ptr: i64,
            _fielddescr: &majit_jitcode::jitcode::BhDescr,
        ) -> i64 {
            self.int_value
        }
        fn bh_getfield_gc_r(
            &self,
            _struct_ptr: i64,
            _fielddescr: &majit_jitcode::jitcode::BhDescr,
        ) -> majit_ir::GcRef {
            self.ref_value
        }
        fn bh_getfield_gc_f(
            &self,
            _struct_ptr: i64,
            _fielddescr: &majit_jitcode::jitcode::BhDescr,
        ) -> f64 {
            self.float_value
        }
    }

    /// M1: non-constant OpRefs (inputargs + recorded op results) map
    /// straight to Box::ResOp(opref.raw()).  No constant-pool lookup.
    #[test]
    fn test_opref_to_box_non_constant_m1() {
        let mut ctx = TraceCtx::for_test(2);
        let i0 = OpRef::input_arg_int(0); // first inputarg
        let i1 = OpRef::input_arg_int(1); // second inputarg
        let add = ctx.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(ctx.opref_to_box(i0), OcBox::ResOp(0));
        assert_eq!(ctx.opref_to_box(i1), OcBox::ResOp(1));
        assert_eq!(ctx.opref_to_box(add), OcBox::ResOp(add.raw()));
    }

    #[test]
    fn replace_box_does_not_install_a_framestack_hook() {
        // `pyjitpl.py replace_box` always walks frames, but
        // `TraceCtx::replace_box` is only the vref/vable/heapcache half.
        // The framestack hook is installed by the owner of the live frames.
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let old = OpRef::input_arg_ref(0);
        let new = OpRef::input_arg_ref(1);
        ctx.replace_box(old, new);
        assert!(
            ctx.replace_frames.is_none(),
            "a bare replace_box must not invent a framestack walk"
        );
    }

    #[test]
    fn nonstandard_standard_alias_walks_framestack_immediately() {
        // `_nonstandard_virtualizable` Step 4: two boxes, same pointer,
        // matching vinfo → `replace_box` at the promote itself.
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        let info = info.finalize_arc(majit_ir::descr::make_size_descr(16));
        let fd = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let standard = recorder.record_input_arg(Type::Ref);
        let alias = recorder.record_input_arg(Type::Ref);
        let field = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let pointer = Value::Ref(majit_ir::GcRef(0x1000));
        ctx.set_opref_concrete(standard, pointer);
        ctx.set_opref_concrete(alias, pointer);
        ctx.install_virtualizable_info(info.clone());
        ctx.set_virtualizable_boxes_with_info(
            vec![field, standard],
            vec![Value::Int(0), pointer],
            info.as_ref(),
            &[],
        );

        let mut walked = std::cell::Cell::new(None::<(OpRef, OpRef)>);
        unsafe fn walk(data: *mut (), oldbox: OpRef, newbox: OpRef) {
            let slot = unsafe { &*data.cast::<std::cell::Cell<Option<(OpRef, OpRef)>>>() };
            slot.set(Some((oldbox, newbox)));
        }
        unsafe { ctx.set_replace_frames(Some(walk), &raw mut walked as *mut ()) };
        let nonstandard = decide_nonstandard(&mut ctx, alias, &fd);
        ctx.clear_replace_frames();

        assert!(
            !nonstandard,
            "same-pointer alias must become the standard virtualizable"
        );
        assert_eq!(
            walked.get(),
            Some((alias, standard)),
            "the framestack hook must run inside replace_box, not after a drain"
        );
        assert_eq!(
            ctx.virtualizable_boxes
                .as_ref()
                .map(|boxes| boxes.as_slice()),
            Some([field, standard].as_slice()),
            "replace_box must already have walked virtualizable_boxes"
        );
    }

    /// `heapcache.py is_nullity_known` answers truthy for a non-`Const` box
    /// whatever its nullity — `nullity_now_known` sets one flag for both — and
    /// falsy for a null `Const`, whose answer is `bool(box.getref_base())`.
    ///
    /// Pinned here rather than through a second nullity branch: the walker's
    /// `replace_box` puts `CONST_NULL` in every register naming the box, so
    /// after the null arm no branch can reach it as a non-constant again.
    #[test]
    fn a_known_null_box_answers_the_nullity_question_unless_it_is_constant() {
        let mut ctx = TraceCtx::for_test(1);
        let box_ = OpRef::input_arg_ref(0);
        assert!(
            !ctx.heapcache_nullity_answered(box_),
            "an unknown box answers nothing",
        );
        ctx.heap_cache_mut().nullity_now_known(box_);
        assert!(
            ctx.heapcache_nullity_answered(box_),
            "a non-constant box known to be null short-circuits",
        );
        let null = ctx.const_null();
        assert!(
            !ctx.heapcache_nullity_answered(null),
            "a null constant falls through to the guard arm",
        );
    }

    /// M1: constant OpRefs resolve via `OpRef::inline_const_to_value` for
    /// type-preserving Box::Const* construction.
    #[test]
    fn test_opref_to_box_constant_int_m1() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx.const_int(42);
        assert!(c.is_constant());
        assert_eq!(ctx.opref_to_box(c), OcBox::ConstInt(42));
    }

    #[test]
    fn test_opref_to_box_constant_float_m1() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx.const_float((3.25_f64).to_bits() as i64);
        assert!(c.is_constant());
        match ctx.opref_to_box(c) {
            OcBox::ConstFloat(bits) => {
                assert_eq!(f64::from_bits(bits), 3.25);
            }
            other => panic!("expected ConstFloat, got {:?}", other),
        }
    }

    /// `recover_ref_value` answers unresolved for a box stamped with the
    /// `NO_CONCRETE` sentinel, and still answers a real address.
    ///
    /// The sentinel is what `heapcache_ops` writes over a load it could not
    /// replay, so handing it back as a `Value::Ref` puts `usize::MAX - 1`
    /// into every consumer that only matches on the variant — including the
    /// two blackhole-frame ref fills, which then seed a frame that resumption
    /// dereferences.
    #[test]
    fn recover_ref_value_rejects_the_no_concrete_sentinel() {
        let mut ctx = TraceCtx::for_test(0);
        let unknown = ctx.record_op(OpCode::NewWithVtable, &[]);
        assert!(ctx.try_set_opref_concrete(unknown, Value::Ref(majit_ir::GcRef::NO_CONCRETE)));
        assert_eq!(ctx.recover_ref_value(unknown, 8), None);

        // Positive control: an ordinary stamped address still comes back, so
        // the assertion above is about the sentinel and not about the stamp
        // failing to land.
        let known = ctx.record_op(OpCode::NewWithVtable, &[]);
        assert!(ctx.try_set_opref_concrete(known, Value::Ref(majit_ir::GcRef(0x1000))));
        assert_eq!(
            ctx.recover_ref_value(known, 8),
            Some(Value::Ref(majit_ir::GcRef(0x1000)))
        );
    }

    /// With no cpu wired, `field_sanity_load` returns `None` —
    /// `translate_support_code=True` analog (sanity check disabled).
    #[test]
    fn field_sanity_load_unwired_returns_none() {
        let ctx = TraceCtx::for_test(0);
        let descr = majit_ir::descr::make_vtable_field_descr();
        assert!(ctx.field_sanity_load(0x1000, &descr, Type::Int).is_none());
        assert!(ctx.field_sanity_load(0x1000, &descr, Type::Ref).is_none());
        assert!(ctx.field_sanity_load(0x1000, &descr, Type::Float).is_none());
    }

    /// With a wired SanityTestCpu, `field_sanity_load` dispatches
    /// through `executor::do_getfield_gc_*` and returns the cpu's
    /// configured value for each kind.
    #[test]
    fn field_sanity_load_wired_dispatches_to_executor() {
        let cpu = SanityTestCpu {
            int_value: 0x1234,
            ref_value: majit_ir::GcRef(0x5678),
            float_value: 2.5,
        };
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));
        let descr = majit_ir::make_field_descr_full(1, 0, 8, Type::Int, false);
        assert_eq!(
            ctx.field_sanity_load(0xCAFE_BABE, &descr, Type::Int),
            Some(Value::Int(0x1234))
        );
        // The opnum selects `do_getfield_gc_*`. The descr's bank is not a
        // second channel, so a different `kind` still reads that executor.
        assert_eq!(
            ctx.field_sanity_load(0xCAFE_BABE, &descr, Type::Ref),
            Some(Value::Ref(majit_ir::GcRef(0x5678)))
        );
        assert_eq!(
            ctx.field_sanity_load(0xCAFE_BABE, &descr, Type::Float),
            Some(Value::Float(2.5))
        );
        assert_eq!(ctx.field_sanity_load(0xCAFE_BABE, &descr, Type::Void), None);
        let ref_descr = majit_ir::make_field_descr_full(1, 0, 8, Type::Ref, false);
        assert_eq!(
            ctx.field_sanity_load(0xCAFE_BABE, &ref_descr, Type::Ref),
            Some(Value::Ref(majit_ir::GcRef(0x5678)))
        );
        assert_eq!(
            ctx.field_sanity_load(0xCAFE_BABE, &ref_descr, Type::Int),
            Some(Value::Int(0x1234))
        );
    }

    /// `do_getarrayitem_gc_i` / `_r` / `_f` are chosen by the opnum. The
    /// descr's `item_type` does not refuse the load.
    #[test]
    fn array_sanity_load_dispatches_by_opnum() {
        let cpu = SanityTestCpu {
            int_value: 0x1111,
            ref_value: majit_ir::GcRef(0x2222),
            float_value: 1.25,
        };
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));
        let mut words = [0x1111u64, 1.25f64.to_bits()];
        let base = words.as_mut_ptr() as i64;
        let int_items = majit_ir::descr::make_array_descr_full(1, 0, 8, 8, Type::Int);
        assert_eq!(
            int_items.as_array_descr().map(|a| a.item_type()),
            Some(Type::Int)
        );
        assert_eq!(
            ctx.array_sanity_load(base, 0, &int_items, Type::Int),
            Some(Value::Int(0x1111))
        );
        assert_eq!(
            ctx.array_sanity_load(base, 0, &int_items, Type::Ref),
            Some(Value::Ref(majit_ir::GcRef(0x1111)))
        );
        assert_eq!(
            ctx.array_sanity_load(base, 1, &int_items, Type::Float),
            Some(Value::Float(1.25))
        );
        assert_eq!(ctx.array_sanity_load(base, 0, &int_items, Type::Void), None);
        let ref_items = majit_ir::descr::make_array_descr_full(1, 0, 8, 8, Type::Ref);
        assert_eq!(
            ctx.array_sanity_load(base, 0, &ref_items, Type::Int),
            Some(Value::Int(0x1111))
        );
    }

    /// An array descr with no `lendescr` describes an array that
    /// carries no length word — `raw_carray_descrof`'s shape.  The
    /// sanity load must decline it, which is the state its three
    /// callers are written for: `dispatch.rs` leaves the recorded op
    /// unstamped, `execute_and_record.rs` takes the unsanitised path,
    /// and `index_in_array_bounds` returns `false`.  Reaching the
    /// executor instead aborts the process, because `bh_arraylen_gc`
    /// has no offset to read from.
    #[test]
    fn arraylen_sanity_load_declines_a_headerless_array_descr() {
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(0),
            float_value: 0.0,
        };
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));
        let headerless = majit_ir::descr::make_array_descr_full(1, 0, 8, 0, Type::Int);
        assert!(
            ctx.arraylen_sanity_load(0xCAFE_BABE, &headerless).is_none(),
            "a headerless array descr has no length word to sanity-read",
        );
    }

    /// The companion: an array descr that does carry a length word
    /// dispatches through `executor::do_arraylen_gc` and reads it, so
    /// the decline above is a property of the descr and not of the
    /// wiring.
    #[test]
    fn arraylen_sanity_load_wired_reads_the_length_word() {
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(0),
            float_value: 0.0,
        };
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));

        // The default `bh_arraylen_gc` reads the machine word at
        // `array_ptr + lendescr.offset()`, so the descr has to point at
        // memory this test owns.
        let header: [usize; 2] = [0xDEAD_BEEF, 5];
        let mut with_len = majit_ir::descr::SimpleArrayDescr::new(1, 16, 8, 0, Type::Int);
        with_len.lendescr = Some(majit_ir::make_field_descr_full(2, 8, 8, Type::Int, false));
        let with_len: DescrRef = std::sync::Arc::new(with_len);

        assert_eq!(
            ctx.arraylen_sanity_load(header.as_ptr() as i64, &with_len),
            Some(Value::Int(5))
        );
    }

    /// A nonstandard frame whose getfield cache missed still stamps
    /// `opimpl_arraylen_gc`'s recorded length from the live array.
    /// `SanityTestCpu::bh_getfield_gc_r` returns `ref_value`; the default
    /// `bh_arraylen_gc` then reads the length word this test owns.
    #[test]
    fn nonstandard_arraylen_vable_stamps_the_executed_length_on_a_cache_miss() {
        let header: [usize; 2] = [0xDEAD_BEEF, 4];
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(header.as_ptr() as usize),
            float_value: 0.0,
        };
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        ctx.set_cpu(Some(&cpu));
        let frame = OpRef::input_arg_ref(0);
        let fdescr = majit_ir::make_field_descr(
            8,
            std::mem::size_of::<usize>(),
            Type::Ref,
            majit_ir::ArrayFlag::Pointer,
        );
        let mut with_len = majit_ir::descr::SimpleArrayDescr::new(1, 16, 8, 0, Type::Int);
        with_len.lendescr = Some(majit_ir::make_field_descr_full(2, 8, 8, Type::Int, false));
        let adescr: DescrRef = std::sync::Arc::new(with_len);

        let nonstandard = decide_nonstandard(&mut ctx, frame, &fdescr);
        let result = ctx.vable_arraylen_vable_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            frame,
            0x1000,
            fdescr,
            adescr,
        );
        assert_eq!(ctx.concrete_of_opref(result), Some(Value::Int(4)));
    }

    /// `heapcache.new_array` answers before any live load, including when
    /// the frame pointer was not passed in.
    #[test]
    fn nonstandard_arraylen_vable_returns_the_cached_const_length() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let frame = OpRef::input_arg_ref(0);
        ctx.heap_cache_mut().new_object(frame);
        let fdescr = majit_ir::make_field_descr(
            8,
            std::mem::size_of::<usize>(),
            Type::Ref,
            majit_ir::ArrayFlag::Pointer,
        );
        let len = ctx.const_int(3);
        let array_descr = majit_ir::make_array_descr(16, 8, Type::Ref);
        let array = ctx.record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr);
        ctx.heap_cache_mut().new_array(array, len, true);
        ctx.heapcache_setfield_cached(frame, fdescr.index(), array);

        let nonstandard = decide_nonstandard(&mut ctx, frame, &fdescr);
        let result = ctx.vable_arraylen_vable_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            frame,
            0,
            fdescr,
            majit_ir::make_array_descr(16, 8, Type::Ref),
        );
        assert_eq!(result, len);
        assert_eq!(ctx.const_value(result), Some(3));
    }

    /// vable_getfield_int cache-hit with Const Int cached and wired cpu:
    /// pyjitpl.py `assert resvalue == upd.currfieldbox.getint()` panics
    /// on mismatch.
    #[test]
    #[should_panic(expected = "sanity")]
    fn vable_getfield_int_cache_hit_sanity_mismatch_panics() {
        let cpu = SanityTestCpu {
            int_value: 99,
            ref_value: majit_ir::GcRef(0),
            float_value: 0.0,
        };
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_cpu(Some(&cpu));
        let fd = majit_ir::make_field_descr_full(1, 0, 8, Type::Int, false);
        let cached = ctx.const_int(42);
        let field_index = fd.index();
        ctx.heapcache_getfield_now_known(vable, field_index, cached);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd);
        ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd,
        );
    }

    #[test]
    fn unnumbered_field_descriptors_do_not_share_a_heapcache_entry() {
        let mut recorder = Trace::new();
        let obj = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let a = majit_ir::make_field_descr_full(u32::MAX, 0, 8, Type::Int, false);
        let b = majit_ir::make_field_descr_full(u32::MAX, 8, 8, Type::Int, false);
        let value = ctx.const_int(42);
        assert_ne!(a.index(), u32::MAX);
        assert_ne!(b.index(), u32::MAX);
        assert_ne!(a.index(), b.index());
        ctx.heapcache_getfield_now_known(obj, a.index(), value);
        assert_eq!(ctx.heapcache_getfield_cached(obj, a.index()), Some(value));
        assert_eq!(ctx.heapcache_getfield_cached(obj, b.index()), None);
        ctx.heapcache_setfield_cached(obj, a.index(), value);
        assert_eq!(ctx.heapcache_getfield_cached(obj, b.index()), None);
        // A genuine descriptor identity still supports forwarding.
        ctx.heapcache_getfield_now_known(obj, 1, value);
        assert_eq!(ctx.heapcache_getfield_cached(obj, 1), Some(value));
    }

    /// `pyjitpl.py do_residual_call` step 5 invalidates on CALL_MAY_FORCE,
    /// the opcode executed in step 2, after CALL_ASSEMBLER is recorded.
    /// `clear_caches_not_necessary` does not list CALL_MAY_FORCE and
    /// `is_plain_call` excludes it, so `clear_caches_varargs` takes
    /// `reset_keep_likely_virtuals` and a seeded GETFIELD (int) cache is gone.
    /// `invalidate_caches_for_escaped` would keep an unescaped object's field,
    /// which is the leftover that failed
    /// `synth/frame_chain_survives_a_recursive_call_assembler`.
    #[test]
    fn call_assembler_invalidate_caches_varargs_drops_getfield_cache() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let obj = OpRef::input_arg_ref(0);
        let field = 1u32;
        let cached = ctx.const_int(0);
        ctx.heapcache_getfield_now_known(obj, field, cached);
        assert_eq!(ctx.heapcache_getfield_cached(obj, field), Some(cached));
        ctx.heapcache_invalidate_caches_varargs(OpCode::CallMayForceR, None, &[obj]);
        assert_eq!(ctx.heapcache_getfield_cached(obj, field), None);
    }

    /// `test_pyjitpl.py test_remove_consts_and_duplicates` — the upstream
    /// vector, verbatim: `[b1, b2, b1, c3]` leaves the first sighting of each
    /// box alone and replaces the repeat AND the constant with fresh `SAME_AS`
    /// results, recording one `SAME_AS` op per replacement.
    #[test]
    fn remove_consts_and_duplicates_matches_the_upstream_vector() {
        let mut recorder = Trace::new();
        let b1 = recorder.record_input_arg(Type::Int);
        let b2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let c3 = ctx.const_int(3);
        let ops_before = ctx.num_ops();

        let mut boxes = [
            (b1, Type::Int),
            (b2, Type::Int),
            (b1, Type::Int),
            (c3, Type::Int),
        ];
        ctx.remove_consts_and_duplicates(&mut boxes);

        assert_eq!(boxes[0].0, b1, "first sighting is left alone");
        assert_eq!(boxes[1].0, b2, "first sighting is left alone");
        assert_ne!(boxes[2].0, b1, "the repeat must become a fresh SAME_AS");
        assert_ne!(boxes[3].0, c3, "the constant must become a fresh SAME_AS");
        assert_ne!(boxes[2].0, boxes[3].0, "each replacement is its own op");
        assert!(
            !boxes[3].0.is_constant(),
            "a SAME_AS result is a runtime box, which is the point: \
             `TreeLoop::cut_trace_from`'s remap skips \
             non-runtime oprefs, so a constant would never be rewritten to \
             the inputarg the LABEL declares for it"
        );
        assert_eq!(
            ctx.num_ops(),
            ops_before + 2,
            "exactly one SAME_AS recorded per replaced slot"
        );

        // Idempotent: a normalized list has no constant and no repeat left.
        let ops_after = ctx.num_ops();
        let mut again = boxes;
        ctx.remove_consts_and_duplicates(&mut again);
        assert_eq!(again, boxes, "a normalized list is unchanged");
        assert_eq!(ctx.num_ops(), ops_after, "and records nothing");
    }

    #[test]
    fn live_arg_boxes_from_reds_appends_vable_fields_and_drops_identity() {
        // pyjitpl.py reached_loop_header:
        //   live_arg_boxes = greenboxes + redboxes
        //   live_arg_boxes += virtualizable_boxes; live_arg_boxes.pop()
        let mut recorder = Trace::new();
        let frame = recorder.record_input_arg(Type::Ref);
        let ec = recorder.record_input_arg(Type::Ref);
        let local = recorder.record_input_arg(Type::Ref);
        let identity = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.virtualizable_boxes = Some(vec![local, identity]);
        let reds = vec![(frame, Type::Ref), (ec, Type::Ref)];
        let live = ctx.live_arg_boxes_from_reds(&reds);
        let live_refs: Vec<OpRef> = live.iter().map(|(o, _)| *o).collect();
        assert_eq!(live_refs, vec![frame, ec, local]);
        assert!(
            !live_refs.contains(&identity),
            "virtualizable identity is popped"
        );
        assert_eq!(live[2].1, Type::Ref);
    }

    #[test]
    fn collect_virtualizable_typed_boxes_empty_pointer_slot_is_ref() {
        // virtualizable.py wrap() of a nullptr is ConstPtr, type 'r'.
        // An empty array item must not become InputArgInt (`q[i] =` then
        // sees an int).
        let recorder = Trace::new();
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.virtualizable_boxes = Some(vec![OpRef::NONE, OpRef::NONE]);
        let typed = ctx.collect_virtualizable_typed_boxes().unwrap();
        assert_eq!(typed[0].1, Type::Ref);
        assert_eq!(typed[1].1, Type::Ref);
    }

    #[test]
    fn new_seeds_green_boxes_from_inputargs_without_a_type_vec() {
        // `TraceCtx::new` does not seed (`pyjitpl.py` starts empty;
        // bridges stay empty). Primary traces seed via
        // `seed_compile_and_run_once_merge_point` (`pyjitpl.py` `_compile_and_run_once`).
        let mut recorder = Trace::new();
        let _i = recorder.record_input_arg(Type::Int);
        let _r = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        assert!(ctx.current_merge_points.is_empty());
        ctx.set_green_key_values(crate::test_greens::gk(0));
        ctx.seed_compile_and_run_once_merge_point();
        let boxes = &ctx.current_merge_points[0].green_boxes;
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].ty, Type::Int);
        assert_eq!(boxes[1].ty, Type::Ref);
        assert_eq!(boxes[0].opref, OpRef::input_arg_typed(0, Type::Int));
        assert_eq!(boxes[1].opref, OpRef::input_arg_typed(1, Type::Ref));
    }

    #[test]
    fn merge_point_green_key_does_not_double_prepend_pypyjit_portal_banks() {
        // pypyjit greens are `[next_instr, is_being_profiled, pycode]`
        // (`interp_jit.py`). `jit_merge_point` banks hold that full
        // tuple. `TraceCtx::with_green_key` stores the same tuple as
        // `green_key_values`. The close key is those declared greens
        // (`warmstate.py` `JitCell.get_uhash` / `get_jitcell`).
        let key = majit_ir::pypyjit_greenkey(5, false, 0x1000);
        let ctx = TraceCtx::with_green_key(
            Trace::new(),
            key.get_uhash(),
            key.clone(),
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let closed = ctx
            .merge_point_green_key(&[5, 0], &[0x1000], &[])
            .expect("portal banks rebuild the pypyjit green key");
        assert_eq!(closed, key);
        assert_eq!(closed.get_uhash(), key.get_uhash());
    }

    #[test]
    fn merge_point_green_key_equals_declared_greens_for_a_non_pypyjit_driver() {
        // `compile.py` `compile_loop` / `warmstate.py` `JitCell.get_uhash`
        // hash `original_boxes[:num_green_args]` — the declared greens in
        // declared order. A driver that needs the loop-header pc includes
        // it in that list (`interp_jit.py` `next_instr`); there is no
        // extra prepended element.
        let key = majit_ir::GreenKey::with_types(vec![7, 99], vec![Type::Int, Type::Int]);
        let ctx = TraceCtx::with_green_key(
            Trace::new(),
            key.get_uhash(),
            key.clone(),
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let closed = ctx
            .merge_point_green_key(&[7, 99], &[], &[])
            .expect("declared greens rebuild the close key");
        assert_eq!(closed, key);
        assert_eq!(closed.get_uhash(), key.get_uhash());
    }

    #[test]
    fn remove_consts_and_duplicates_untyped_rewrites_in_place() {
        let mut recorder = Trace::new();
        let b1 = recorder.record_input_arg(Type::Int);
        let b2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let c3 = ctx.const_int(3);
        let mut boxes = [b1, b2, b1, c3];
        ctx.remove_consts_and_duplicates_untyped(&mut boxes);
        assert_eq!(boxes[0], b1);
        assert_eq!(boxes[1], b2);
        assert_ne!(boxes[2], b1);
        assert_ne!(boxes[3], c3);
    }

    /// history.py `record_same_as` carries the source box's value on
    /// the freshly recorded `SAME_AS` result.
    #[test]
    fn remove_consts_and_duplicates_carries_the_source_box_value() {
        let mut recorder = Trace::new();
        let b1 = recorder.record_input_arg(Type::Int);
        let b2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_opref_concrete(b1, majit_ir::Value::Int(7));
        let c3 = ctx.const_int(3);
        let mut boxes = [
            (b1, Type::Int),
            (b2, Type::Int),
            (b1, Type::Int),
            (c3, Type::Int),
        ];

        ctx.remove_consts_and_duplicates(&mut boxes);

        assert_eq!(
            ctx.concrete_of_opref(boxes[2].0),
            Some(Value::Int(7)),
            "the duplicate's wrapper carries the source box's value"
        );
        assert_eq!(
            ctx.concrete_of_opref(boxes[3].0),
            Some(Value::Int(3)),
            "the constant's wrapper carries the constant's value"
        );
    }

    /// `record_same_as` carries a standard virtualizable payload box's
    /// value from the box itself (`InputArgInt.getint`).
    #[test]
    fn remove_consts_and_duplicates_carries_virtualizable_payload_value() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let payload = recorder.record_input_arg(Type::Int);
        let other = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[payload, other],
            &[Value::Int(37), ph(Type::Int)],
            &[],
        );

        assert_eq!(
            ctx.box_value(payload),
            Some(Value::Int(37)),
            "the payload box carries its concrete"
        );
        assert_eq!(
            ctx.virtualizable_entry_at(0),
            Some((payload, Value::Int(37))),
            "virtualizable_boxes[0] is the payload box with its value"
        );

        let mut boxes = [(payload, Type::Int), (payload, Type::Int)];
        ctx.remove_consts_and_duplicates(&mut boxes);

        assert_ne!(boxes[1].0, payload, "the duplicate becomes a SAME_AS");
        assert_eq!(
            ctx.concrete_of_opref(boxes[1].0),
            Some(Value::Int(37)),
            "the duplicate's wrapper carries the virtualizable payload value"
        );
    }

    /// `set_virtualizable_box_at` is SSA-rename: it never stamps a box with
    /// another box's value. An unstamped new box stays unstamped;
    /// `virtualizable_entry_at` answers the live virtualizable
    /// (`virtualizable.py read_boxes`).
    #[test]
    fn set_virtualizable_box_at_leaves_unstamped_new_box_heap_fallback() {
        let heap = TwoArrayVable::new(false);
        assert_eq!(heap.slots()[0], 1);
        let (mut ctx, _) = ctx_with_shadow(
            &heap,
            &[
                Value::Int(37),
                Value::Int(102),
                Value::Int(110),
                Value::Int(120),
                Value::Int(130),
            ],
        );
        let payload = ctx.virtualizable_box_at(0).expect("slot 0");
        assert_eq!(
            ctx.virtualizable_entry_at(0),
            Some((payload, Value::Int(37))),
            "a stamped box answers its own result even when the heap differs"
        );
        let renamed = ctx.record_op(majit_ir::OpCode::IntAdd, &[payload, payload]);
        assert!(ctx.box_value(renamed).is_none(), "new op starts unstamped");
        assert!(ctx.set_virtualizable_box_at(0, renamed));
        assert!(
            ctx.box_value(renamed).is_none(),
            "the renamed box stays unstamped; a box's value is only ever its own result"
        );
        assert!(!ctx.box_carries_runtime_concrete(renamed));
        assert_eq!(
            ctx.virtualizable_entry_at(0),
            Some((renamed, Value::Int(1))),
            "unstamped slot's current concrete is the live virtualizable"
        );
        ctx.synchronize_virtualizable_at(0);
        assert_eq!(
            heap.slots()[0],
            1,
            "synchronize skips a box that carries no runtime concrete"
        );
    }

    /// vable_getfield_ref cache-hit (pyjitpl.py:939
    /// `assert resvalue == upd.currfieldbox.getref_base()`).
    #[test]
    #[should_panic(expected = "sanity check (ref)")]
    fn vable_getfield_ref_cache_hit_sanity_mismatch_panics() {
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(0xCCCC_DDDD),
            float_value: 0.0,
        };
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_cpu(Some(&cpu));
        let fd = majit_ir::make_field_descr_full(1, 0, 8, Type::Ref, false);
        let cached = ctx.const_ref(0xAAAA_BBBB);
        let field_index = fd.index();
        ctx.heapcache_getfield_now_known(vable, field_index, cached);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd);
        ctx.vable_getfield_ref_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd,
        );
    }

    /// vable_getfield_float cache-hit (pyjitpl.py:944
    /// `assert ConstFloat(resvalue).same_constant(upd.currfieldbox.constbox())`).
    #[test]
    #[should_panic(expected = "sanity check (float)")]
    fn vable_getfield_float_cache_hit_sanity_mismatch_panics() {
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(0),
            float_value: 2.5,
        };
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_cpu(Some(&cpu));
        let fd = majit_ir::make_field_descr_full(1, 0, 8, Type::Float, false);
        let cached = ctx.const_float((1.5_f64).to_bits() as i64);
        let field_index = fd.index();
        ctx.heapcache_getfield_now_known(vable, field_index, cached);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd);
        ctx.vable_getfield_float_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd,
        );
    }

    /// Matched (loaded == cached) ref + float cache-hits — no panic;
    /// returns cached OpRefs.
    #[test]
    fn vable_getfield_ref_float_cache_hit_sanity_match_no_panic() {
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(0xAAAA_BBBB),
            float_value: 3.25,
        };
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_cpu(Some(&cpu));
        let fd_r = majit_ir::make_field_descr_full(1, 0, 8, Type::Ref, false);
        let cached_r = ctx.const_ref(0xAAAA_BBBB);
        let field_index_r = fd_r.index();
        ctx.heapcache_getfield_now_known(vable, field_index_r, cached_r);

        let fd_f = majit_ir::make_field_descr_full(2, 8, 8, Type::Float, false);
        let cached_f = ctx.const_float((3.25_f64).to_bits() as i64);
        let field_index_f = fd_f.index();
        ctx.heapcache_getfield_now_known(vable, field_index_f, cached_f);

        let nonstandard_r = decide_nonstandard(&mut ctx, vable, &fd_r);
        let (r_result, _) = ctx.vable_getfield_ref_checked(
            nonstandard_r,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd_r,
        );
        assert_eq!(r_result, cached_r);
        let nonstandard_f = decide_nonstandard(&mut ctx, vable, &fd_f);
        let (f_result, _) = ctx.vable_getfield_float_checked(
            nonstandard_f,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd_f,
        );
        assert_eq!(f_result, cached_f);
    }

    /// Matched (loaded == cached) int cache-hit — no panic.
    #[test]
    fn vable_getfield_int_cache_hit_sanity_match_no_panic() {
        let cpu = SanityTestCpu {
            int_value: 7,
            ref_value: majit_ir::GcRef(0),
            float_value: 0.0,
        };
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_cpu(Some(&cpu));
        let fd = majit_ir::make_field_descr_full(1, 0, 8, Type::Int, false);
        let cached = ctx.const_int(7);
        let field_index = fd.index();
        ctx.heapcache_getfield_now_known(vable, field_index, cached);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd);
        let (result, _) = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0xCAFE_BABE,
            fd,
        );
        assert_eq!(result, cached);
    }

    #[test]
    fn test_opref_to_box_constant_ref_m1() {
        let mut ctx = TraceCtx::for_test(0);
        let addr = 0xdead_beef_u64;
        let c = ctx.const_ref(addr as i64);
        assert!(c.is_constant());
        assert_eq!(ctx.opref_to_box(c), OcBox::ConstPtr(addr));
    }

    /// Other cargo test threads share this process-global counter, so the
    /// bound is loose: 100_000 calls must add fewer than 10_000 wide ids.
    #[test]
    fn const_ref_does_not_mint_wide_ids() {
        let mut ctx = TraceCtx::for_test(0);
        let addr = 0x1000_i64;
        let before = majit_ir::operand::wide_ids_minted();
        for _ in 0..100_000 {
            let _ = ctx.const_ref(addr);
        }
        let delta = majit_ir::operand::wide_ids_minted().wrapping_sub(before);
        assert!(
            delta < 10_000,
            "repeated const_ref of the same non-null address must not mint wide ids"
        );
    }

    #[test]
    fn initial_inputarg_argbox_reads_type_from_sparse_live_box() {
        // [Int, dead, Float]: compact inputarg_types() is [Int, Float].
        // Indexing it with position 2 is OOB; the box at get_position() 2
        // is the live Float (ResumeDataBoxReader.load_box_from_cpu).
        let rec =
            Trace::with_input_layout(&[Type::Int, Type::Ref, Type::Float], &[true, false, true]);
        let mut ctx = TraceCtx::new(
            rec,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.initial_inputarg_consts = vec![
            OpRef::const_int(1),
            OpRef::const_int(0),
            OpRef::const_float(1.5),
        ];
        assert_eq!(ctx.inputarg_types(), vec![Type::Int, Type::Float]);
        assert_eq!(ctx.inputarg_type_at(2), Some(Type::Float));
        let (kind, opref, bits) = ctx
            .initial_inputarg_argbox(2)
            .expect("live Float at position 2");
        assert_eq!(kind, crate::jitcode::JitArgKind::Float);
        assert_eq!(opref, OpRef::input_arg_float(2));
        assert_eq!(opref.ty(), Some(Type::Float));
        assert_eq!(bits, 1.5f64.to_bits() as i64);
        let boxes = ctx.inputarg_greenboxes();
        assert_eq!(
            boxes.iter().map(|b| (b.opref, b.ty)).collect::<Vec<_>>(),
            vec![
                (OpRef::input_arg_int(0), Type::Int),
                (OpRef::input_arg_float(2), Type::Float)
            ]
        );
    }

    fn take_all_ops(ctx: TraceCtx) -> Vec<majit_ir::Op> {
        let mut recorder = ctx.recorder;
        let jump_args: Vec<OpRef> = recorder.inputargs().iter().map(|arg| arg.opref()).collect();
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();
        // Return only non-JUMP ops
        trace
            .ops
            .iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .map(|rc| (**rc).clone())
            .collect()
    }

    // virtualizable_boxes tests

    fn make_test_vable_info() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_field("sp", Type::Int, 16);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    fn make_test_vable_info_with_array() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    /// Production `walk_active_trace_refs` forwards `InputArg*` `_res*` on
    /// the recorder, then `ConstPtr` in `virtualizable_boxes` and
    /// `virtualizable_heap_ptr` through `walk_virtualizable_value_refs`.
    fn walk_vable_refs_as_gc(ctx: &mut TraceCtx, mut visitor: impl FnMut(&mut majit_ir::GcRef)) {
        for ia in ctx.recorder.inputargs() {
            if let Some(Value::Ref(mut r)) = ia.get_value() {
                visitor(&mut r);
                ia.set_value(Value::Ref(r));
            }
        }
        ctx.walk_virtualizable_value_refs(visitor);
    }

    // Test helper: typed placeholder matching each slot's declared type so
    // the Box's (OpRef, concrete) pair stays internally consistent — the
    // `virtualizable_boxes[index] = valuebox` invariant.  Tests
    // only inspect OpRef plumbing; the concrete half is never read.
    fn ph(ty: Type) -> Value {
        match ty {
            Type::Int => Value::Int(0),
            Type::Float => Value::Float(0.0),
            Type::Ref => Value::Ref(majit_ir::GcRef::NULL),
            Type::Void => Value::Void,
        }
    }

    #[test]
    fn standard_vable_getfield_reads_from_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int); // pc
        let box1 = recorder.record_input_arg(Type::Int); // sp
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // getfield with offset=8 → static field 0 → box0
        let nonstandard8 = decide_nonstandard(&mut ctx, vable, &fd8);
        let (result, _) = ctx.vable_getfield_int_checked(
            nonstandard8,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );
        assert_eq!(result, box0);
        // getfield with offset=16 → static field 1 → box1
        let nonstandard16 = decide_nonstandard(&mut ctx, vable, &fd16);
        let (result, _) = ctx.vable_getfield_int_checked(
            nonstandard16,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd16,
        );
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getfield should not emit ops"
        );
    }

    #[test]
    fn standard_vable_setfield_writes_to_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // setfield offset=8 → updates box0
        let nonstandard_set = decide_nonstandard(&mut ctx, vable, &fd8);
        ctx.vable_setfield_checked(
            nonstandard_set,
            vable,
            fd8.clone(),
            new_val,
            Some(ph(Type::Int)),
        );

        // Box 0 should now be new_val
        let nonstandard8 = decide_nonstandard(&mut ctx, vable, &fd8);
        let (result, _) = ctx.vable_getfield_int_checked(
            nonstandard8,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );
        assert_eq!(result, new_val);
        // Box 1 unchanged
        let nonstandard16 = decide_nonstandard(&mut ctx, vable, &fd16);
        let (result, _) = ctx.vable_getfield_int_checked(
            nonstandard16,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd16,
        );
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable setfield should not emit ops"
        );
    }

    #[test]
    fn nonstandard_vable_getfield_emits_heap_op() {
        // Without init_virtualizable_boxes, falls back to GETFIELD_GC_I
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        let _result = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    /// `pyjitpl.py _nonstandard_virtualizable`: `vinfo is fielddescr.get_vinfo()`
    /// is false for a different VirtualizableInfo object.
    #[test]
    fn foreign_vinfo_skips_standard_ptr_eq() {
        extern "C" fn clear_vable_noop(_vable: *mut u8) {}
        let mut info_a = make_test_vable_info();
        info_a.set_clear_vable(
            clear_vable_noop as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info_a = info_a.finalize_arc(majit_ir::descr::make_size_descr(64));
        let mut info_b = make_test_vable_info();
        info_b.set_clear_vable(
            clear_vable_noop as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info_b = info_b.finalize_arc(majit_ir::descr::make_size_descr(64));
        let fd = info_a.static_field_descr(0);

        let mut recorder = Trace::new();
        let standard = recorder.record_input_arg(Type::Ref);
        let other = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.install_virtualizable_info(info_b);
        ctx.virtualizable_boxes = Some(vec![standard]);
        ctx.set_opref_concrete(standard, Value::Ref(majit_ir::GcRef(1)));
        ctx.set_opref_concrete(other, Value::Ref(majit_ir::GcRef(1)));
        let nonstandard = decide_nonstandard(&mut ctx, other, &fd);
        let _ = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            other,
            0,
            fd,
        );
        let ops = take_all_ops(ctx);
        assert!(
            ops.iter().all(|op| op.opcode != OpCode::PtrEq),
            "vinfo is fielddescr.get_vinfo() is false for a different info, got {ops:?}"
        );
    }

    #[test]
    fn matching_vinfo_takes_the_standard_ptr_eq_arm() {
        extern "C" fn clear_vable_noop(_vable: *mut u8) {}
        let mut info = make_test_vable_info();
        info.set_clear_vable(
            clear_vable_noop as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info = info.finalize_arc(majit_ir::descr::make_size_descr(64));
        let fd = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let standard = recorder.record_input_arg(Type::Ref);
        let other = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let field_box = ctx.const_int(0);
        ctx.install_virtualizable_info(info.clone());
        ctx.init_virtualizable_boxes(
            &info,
            standard,
            Value::Ref(majit_ir::GcRef(1)),
            &[field_box],
            &[Value::Int(0)],
            &[],
        );
        ctx.set_opref_concrete(standard, Value::Ref(majit_ir::GcRef(1)));
        ctx.set_opref_concrete(other, Value::Ref(majit_ir::GcRef(1)));
        let nonstandard = decide_nonstandard(&mut ctx, other, &fd);
        let _ = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            other,
            0,
            fd,
        );
        let ops = take_all_ops(ctx);
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::PtrEq),
            "the same vinfo must enter the PTR_EQ arm, got {ops:?}"
        );
    }

    /// `pyjitpl.py _nonstandard_virtualizable`: empty `virtualizable_boxes`
    /// is the `vinfo is None` arm of the standard-box gate. Step 5 still
    /// emits `emit_force_virtualizable` before returning True.
    /// `pyjitpl.py _nonstandard_virtualizable`: `if vinfo is fielddescr.get_vinfo()`
    /// is false when the descr has no vinfo. Do not PTR_EQ / replace_box.
    #[test]
    fn foreign_fielddescr_skips_standard_ptr_eq() {
        let mut recorder = Trace::new();
        let standard = recorder.record_input_arg(Type::Ref);
        let other = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.virtualizable_boxes = Some(vec![standard]);
        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let nonstandard = decide_nonstandard(&mut ctx, other, &fd8);
        let _ = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            other,
            0,
            fd8,
        );
        let ops = take_all_ops(ctx);
        assert!(
            ops.iter().all(|op| op.opcode != OpCode::PtrEq),
            "a descr with no vinfo must not enter the PTR_EQ arm, got {ops:?}"
        );
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::GetfieldGcI),
            "nonstandard getfield still records the heap load, got {ops:?}"
        );
    }

    #[test]
    fn empty_boxes_still_emits_force_virtualizable() {
        extern "C" fn clear_vable_noop(_vable: *mut u8) {}
        let mut info = make_test_vable_info();
        info.set_clear_vable(
            clear_vable_noop as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let fd8 = info.static_field_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.virtualizable_info = Some(std::sync::Arc::new(info));
        ctx.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(1)));

        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        let _result = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );

        let ops = take_all_ops(ctx);
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::CondCallN),
            "empty boxes must still emit emit_force_virtualizable, got {ops:?}"
        );
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::GetfieldGcI),
            "nonstandard getfield still records the heap load, got {ops:?}"
        );
    }

    /// `pyjitpl.py MIFrame.emit_force_virtualizable`:
    /// `execute_and_record(GETFIELD_GC_R, token_descr, box)` then
    /// `execute_and_record(PTR_NE, None, tokenbox, CONST_NULL)`. A wired
    /// `SanityTestCpu` makes `field_sanity_load` return the token, and
    /// the recorded boxes carry that value (`history.py _make_op`).
    #[test]
    fn emit_force_virtualizable_stamps_token_getfield_and_ptr_ne() {
        extern "C" fn clear_vable_noop(_vable: *mut u8) {}
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.set_clear_vable(
            clear_vable_noop as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info = info.finalize_arc(majit_ir::descr::make_size_descr(64));
        let fd = info.static_field_descr(0);

        let storage = [0usize; 8];
        let token = 0x1234_0000usize;
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(token),
            float_value: 0.0,
        };

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.install_virtualizable_info(info);
        ctx.set_cpu(Some(&cpu));
        ctx.set_opref_concrete(
            vable,
            Value::Ref(majit_ir::GcRef(storage.as_ptr() as usize)),
        );
        ctx.emit_force_virtualizable(&fd, vable);

        let tokenbox = OpRef::ref_op(1);
        assert_eq!(ctx.opcode_of(tokenbox), Some(OpCode::GetfieldGcR));
        assert_eq!(
            ctx.box_value(tokenbox),
            Some(Value::Ref(majit_ir::GcRef(token))),
        );
        let condbox = OpRef::int_op(2);
        assert_eq!(ctx.opcode_of(condbox), Some(OpCode::PtrNe));
        assert_eq!(ctx.box_value(condbox), Some(Value::Int(1)));
    }

    /// `executor.py do_cond_call`: when the token GETFIELD is non-null,
    /// the COND_CALL target (`clear_vable_ptr`) runs at trace time.
    #[test]
    fn emit_force_virtualizable_executes_cond_call_when_token_is_nonzero() {
        extern "C" fn clear_vable_probe(vable: i64) {
            // `make_clear_vable_descr` is `[Ref] -> Void`; mark the pc word.
            unsafe {
                *(vable as *mut usize).add(1) = 0xF0CED;
            }
        }
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.set_clear_vable(
            clear_vable_probe as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info = info.finalize_arc(majit_ir::descr::make_size_descr(64));
        let fd = info.static_field_descr(0);

        let mut storage = [0usize; 8];
        let ptr = storage.as_mut_ptr();
        let token = 0x1234_0000usize;
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef(token),
            float_value: 0.0,
        };

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.install_virtualizable_info(info);
        ctx.set_cpu(Some(&cpu));
        ctx.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(ptr as usize)));
        ctx.emit_force_virtualizable(&fd, vable);

        assert_eq!(
            unsafe { ptr.add(1).read() },
            0xF0CED,
            "do_cond_call must invoke clear_vable_ptr when the token is non-null",
        );
        let condbox = OpRef::int_op(2);
        assert_eq!(ctx.opcode_of(condbox), Some(OpCode::PtrNe));
        assert_eq!(ctx.box_value(condbox), Some(Value::Int(1)));
    }

    /// `executor.py do_cond_call`: a null token skips the helper.
    #[test]
    fn emit_force_virtualizable_skips_cond_call_when_token_is_null() {
        extern "C" fn clear_vable_probe(vable: i64) {
            unsafe {
                *(vable as *mut usize).add(1) = 0xF0CED;
            }
        }
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.set_clear_vable(
            clear_vable_probe as *const (),
            crate::virtualizable::VirtualizableInfo::make_clear_vable_descr(),
        );
        let info = info.finalize_arc(majit_ir::descr::make_size_descr(64));
        let fd = info.static_field_descr(0);

        let mut storage = [0usize; 8];
        let ptr = storage.as_mut_ptr();
        let cpu = SanityTestCpu {
            int_value: 0,
            ref_value: majit_ir::GcRef::NULL,
            float_value: 0.0,
        };

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.install_virtualizable_info(info);
        ctx.set_cpu(Some(&cpu));
        ctx.set_opref_concrete(vable, Value::Ref(majit_ir::GcRef(ptr as usize)));
        ctx.emit_force_virtualizable(&fd, vable);

        assert_eq!(
            unsafe { ptr.add(1).read() },
            0,
            "do_cond_call must not invoke clear_vable_ptr when the token is null",
        );
        let condbox = OpRef::int_op(2);
        assert_eq!(ctx.opcode_of(condbox), Some(OpCode::PtrNe));
        assert_eq!(ctx.box_value(condbox), Some(Value::Int(0)));
    }

    #[test]
    fn nonstandard_vable_setfield_emits_heap_op() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        ctx.vable_setfield_checked(nonstandard, vable, fd8, val, Some(ph(Type::Int)));

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
    }

    /// `add_merge_point_with_key` must stamp the header snapshot with the virtualizable
    /// IDENTITY, not the synchronization target.
    ///
    /// A root portal seed points `virtualizable_heap_ptr` at the
    /// `snapshot_for_tracing` copy while baking the identity against the live
    /// frame, so the two name different objects. `compile.py:510` wants the one the
    /// boxes describe; taking the sync target instead hands
    /// `patch_new_loop_to_load_virtualizable_fields` a frame whose array field
    /// is unrelated to the snapshot, and its `assert i == len(inputargs)`
    /// (compile.py:458) fires on the arity that comes back.
    #[test]
    fn merge_point_records_the_vable_identity_not_the_sync_target() {
        const LIVE_FRAME: usize = 0x5000;
        const SNAPSHOT_COPY: usize = 0x9000;

        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.set_virtualizable_heap_ptr(SNAPSHOT_COPY as *const u8);

        // Before the seed there is no snapshot to describe, so a header
        // registered here records nothing and the compile step keeps its
        // trace-start resolution.
        let key1 = crate::test_greens::gk(1);
        ctx.add_merge_point_with_key(
            key1.get_uhash(),
            key1,
            vec![GreenBox::new(box0, Type::Int)],
            7,
        );
        assert_eq!(ctx.current_merge_points.last().unwrap().vable_ptr, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            Value::Ref(majit_ir::GcRef(LIVE_FRAME)),
            &[box0],
            &[ph(Type::Int)],
            &[],
        );
        let key2 = crate::test_greens::gk(2);
        ctx.add_merge_point_with_key(
            key2.get_uhash(),
            key2,
            vec![GreenBox::new(box0, Type::Int)],
            9,
        );

        assert_eq!(ctx.standard_virtualizable_ptr(), Some(LIVE_FRAME));
        assert_eq!(
            ctx.current_merge_points.last().unwrap().vable_ptr,
            LIVE_FRAME,
            "merge point took the sync target instead of the identity",
        );
    }

    /// A collection forwards the synchronization target; it does not retarget
    /// it onto the identity.
    ///
    /// With the target on a snapshot copy and the identity on the live frame,
    /// re-deriving the target from the identity box moved every later
    /// `synchronize_virtualizable` onto the live frame at whichever collection
    /// happened to fire. The live frame runs an iteration behind the walk, so a
    /// read resumed from it saw the previous iteration's locals: a wrong answer
    /// that only a small nursery exposes.
    #[test]
    fn a_collection_forwards_a_sync_target_that_is_not_the_identity() {
        const LIVE_FRAME: usize = 0x5000;
        const SNAPSHOT_COPY: usize = 0x9000;
        const MOVED_BY: usize = 0x100;

        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            Value::Ref(majit_ir::GcRef(LIVE_FRAME)),
            &[box0],
            &[ph(Type::Int)],
            &[],
        );
        ctx.set_virtualizable_heap_ptr(SNAPSHOT_COPY as *const u8);

        walk_vable_refs_as_gc(&mut ctx, |gcref| gcref.0 += MOVED_BY);

        assert_eq!(
            ctx.virtualizable_heap_ptr(),
            Some((SNAPSHOT_COPY + MOVED_BY) as *const u8),
            "the collection moved the sync target onto the identity",
        );
        assert_eq!(
            ctx.standard_virtualizable_ptr(),
            Some(LIVE_FRAME + MOVED_BY)
        );
    }

    /// The same collection with the target ON the identity follows it: the
    /// two name one object, so forwarding one forwards the other.
    #[test]
    fn a_collection_forwards_a_sync_target_that_is_the_identity() {
        const LIVE_FRAME: usize = 0x5000;
        const MOVED_BY: usize = 0x100;

        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            Value::Ref(majit_ir::GcRef(LIVE_FRAME)),
            &[box0],
            &[ph(Type::Int)],
            &[],
        );
        ctx.set_virtualizable_heap_ptr(LIVE_FRAME as *const u8);

        walk_vable_refs_as_gc(&mut ctx, |gcref| gcref.0 += MOVED_BY);

        assert_eq!(
            ctx.virtualizable_heap_ptr(),
            Some((LIVE_FRAME + MOVED_BY) as *const u8)
        );
    }

    #[test]
    fn standard_vable_getfield_unknown_offset_emits_heap_op() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // Unknown offset (999) → fallback to heap op
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd999);
        let _result = ctx.vable_getfield_int_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd999,
        );

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn standard_vable_getfield_ref_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("obj", Type::Ref, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(&info, vable, ph(Type::Ref), &[box0], &[ph(Type::Ref)], &[]);

        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        let (result, _) = ctx.vable_getfield_ref_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn standard_vable_getfield_float_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("val", Type::Float, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0],
            &[ph(Type::Float)],
            &[],
        );

        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        let (result, _) = ctx.vable_getfield_float_checked(
            nonstandard,
            crate::cpu::default_cpu().as_ref(),
            vable,
            0,
            fd8,
        );
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_reads_from_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        // 1 static field (pc) + 3 array elements
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let box_arr2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1, box_arr2],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[3], // array has 3 elements
        );

        // Array field offset=24, item_index=0 → box_arr0
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0, adesc.clone());
        assert_eq!(r0, box_arr0);
        // item_index=1 → box_arr1
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1, adesc.clone());
        assert_eq!(r1, box_arr1);
        // item_index=2 → box_arr2
        let (r2, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 2, adesc);
        assert_eq!(r2, box_arr2);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getarrayitem should not emit ops"
        );
    }

    #[test]
    fn vable_setarrayitem_writes_to_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[2], // array has 2 elements
        );

        // Write to array[1]
        ctx.vable_setarrayitem_vable(&fd24, 1, new_val, ph(Type::Int));

        // Read back: array[0] unchanged, array[1] updated
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0, adesc.clone());
        assert_eq!(r0, box_arr0);
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1, adesc);
        assert_eq!(r1, new_val);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    /// Two static fields plus arrays `a` (len 2) and `b` (len 1).
    /// Flat slots are `pc, sp, a[0], a[1], b[0]`. The heap starts at
    /// `1, 2, 10, 20, 30`.
    struct TwoArrayVable {
        info: VirtualizableInfo,
        obj: Vec<u64>,
        /// Kept alive: `obj` stores this buffer's address.
        array_a: Vec<u64>,
        /// Kept alive: `obj` stores this buffer's address.
        array_b: Vec<u64>,
    }

    impl TwoArrayVable {
        fn new(outer_executor_owns_state: bool) -> Self {
            let mut info = VirtualizableInfo::new(0);
            info.add_field("pc", Type::Int, 8);
            info.add_field("sp", Type::Int, 16);
            info.add_array_field(
                "a",
                Type::Int,
                24,
                0,
                8,
                majit_ir::make_array_descr(8, 8, Type::Int),
            );
            info.add_array_field(
                "b",
                Type::Int,
                32,
                0,
                8,
                majit_ir::make_array_descr(8, 8, Type::Int),
            );
            info.set_parent_descr(majit_ir::descr::make_size_descr(40));
            info.outer_executor_owns_state = outer_executor_owns_state;

            // token, pc, sp, array-a pointer, array-b pointer.
            let mut obj = vec![0u64; 5];
            let mut array_a = vec![0u64; 3];
            let mut array_b = vec![0u64; 2];
            let obj_ptr = obj.as_mut_ptr() as *mut u8;
            let a_ptr = array_a.as_mut_ptr() as *mut u8;
            let b_ptr = array_b.as_mut_ptr() as *mut u8;
            unsafe {
                *(a_ptr as *mut usize) = 2;
                *(b_ptr as *mut usize) = 1;
                *(obj_ptr.add(24) as *mut usize) = a_ptr as usize;
                *(obj_ptr.add(32) as *mut usize) = b_ptr as usize;
                info.write_field(obj_ptr, 0, 1);
                info.write_field(obj_ptr, 1, 2);
                info.write_array_item(obj_ptr, 0, 0, 10);
                info.write_array_item(obj_ptr, 0, 1, 20);
                info.write_array_item(obj_ptr, 1, 0, 30);
            }
            Self {
                info,
                obj,
                array_a,
                array_b,
            }
        }

        fn ptr(&self) -> *const u8 {
            self.obj.as_ptr() as *const u8
        }

        fn slots(&self) -> [i64; 5] {
            let _keep_arrays = (&self.array_a, &self.array_b);
            let obj = self.ptr();
            unsafe {
                [
                    self.info.read_field(obj, 0),
                    self.info.read_field(obj, 1),
                    self.info.read_array_item(obj, 0, 0),
                    self.info.read_array_item(obj, 0, 1),
                    self.info.read_array_item(obj, 1, 0),
                ]
            }
        }
    }

    fn ctx_with_shadow(heap: &TwoArrayVable, shadow: &[Value]) -> (TraceCtx, OpRef) {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut boxes = Vec::with_capacity(shadow.len());
        for _ in shadow {
            boxes.push(recorder.record_input_arg(Type::Int));
        }
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(&heap.info, vable, ph(Type::Ref), &boxes, shadow, &[2, 1]);
        ctx.set_virtualizable_heap_ptr(heap.ptr());
        (ctx, vable)
    }

    /// `virtualizable.py write_box_at` / `pyjitpl.py MIFrame._opimpl_setfield_vable`
    /// and `MIFrame._opimpl_setarrayitem_vable`: the shadow differs from the
    /// heap in every slot, so a whole-shadow `write_boxes` would clobber the
    /// slots the store did not touch.
    #[test]
    fn synchronize_virtualizable_at_writes_only_that_slot() {
        let heap = TwoArrayVable::new(false);
        assert_eq!(heap.slots(), [1, 2, 10, 20, 30]);
        let (mut ctx, vable) = ctx_with_shadow(
            &heap,
            &[
                Value::Int(101),
                Value::Int(102),
                Value::Int(110),
                Value::Int(120),
                Value::Int(130),
            ],
        );
        let pc = heap.info.static_field_descr(0);
        let a_descr = heap.info.array_pointer_field_descr(0);
        let b_descr = heap.info.array_pointer_field_descr(1);
        let b_item = heap.info.array_item_descr(1);

        // Index 4 is b[0], past both statics and array a.
        ctx.synchronize_virtualizable_at(4);
        assert_eq!(heap.slots(), [1, 2, 10, 20, 130]);

        let new_pc = ctx.const_int(777);
        let nonstandard_pc = decide_nonstandard(&mut ctx, vable, &pc);
        ctx.vable_setfield_checked(nonstandard_pc, vable, pc, new_pc, Some(Value::Int(777)));
        assert_eq!(heap.slots(), [777, 2, 10, 20, 130]);

        let new_a1 = ctx.const_int(888);
        ctx.vable_setarrayitem_vable(&a_descr, 1, new_a1, Value::Int(888));
        assert_eq!(heap.slots(), [777, 2, 10, 888, 130]);

        let index = ctx.const_int(0);
        let new_b = ctx.const_int(999);
        let stored = ctx.vable_setarrayitem_checked(
            false,
            0,
            vable,
            index,
            0,
            b_descr,
            b_item,
            new_b,
            Value::Int(999),
            false,
        );
        assert!(matches!(stored, VableArrayStore::Stored(Some(_))));
        assert_eq!(heap.slots(), [777, 2, 10, 888, 999]);
    }

    /// The outer-executor carve-out on `write_virtualizable_back` also gates
    /// `write_virtualizable_back_at`. `skip_when_outer_owned == false` still
    /// writes the one slot.
    #[test]
    fn synchronize_virtualizable_at_skips_when_outer_executor_owns_state() {
        let heap = TwoArrayVable::new(true);
        let (ctx, _) = ctx_with_shadow(
            &heap,
            &[
                Value::Int(101),
                Value::Int(102),
                Value::Int(110),
                Value::Int(120),
                Value::Int(130),
            ],
        );
        ctx.synchronize_virtualizable_at(4);
        assert_eq!(heap.slots(), [1, 2, 10, 20, 30]);
        ctx.write_virtualizable_back_at(4, false);
        assert_eq!(heap.slots(), [1, 2, 10, 20, 130]);
    }

    #[test]
    fn vable_getarrayitem_unknown_array_emits_heap_op() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        // Unknown array field offset → fallback
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let adesc = info.array_item_descr(0);
        let _r = ctx.vable_getarrayitem_int_vable(vable, &fd999, 0, adesc);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetarrayitemGcI);
    }

    #[test]
    fn nonstandard_getarrayitem_after_store_does_not_reread() {
        let info = make_test_vable_info_with_array();
        let fd = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        let mut recorder = Trace::new();
        let portal = recorder.record_input_arg(Type::Ref);
        let callee = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let stored = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            portal,
            ph(Type::Ref),
            &[box_pc, stored],
            &[ph(Type::Int), ph(Type::Ref)],
            &[1],
        );
        let index = ctx.const_int(0);
        ctx.vable_setarrayitem_checked(
            true,
            0,
            callee,
            index,
            0,
            fd.clone(),
            adesc.clone(),
            stored,
            ph(Type::Ref),
            false,
        );
        let (got, _) = ctx.vable_getarrayitem_ref_checked(true, 0, callee, index, 0, fd, adesc);
        assert_eq!(got, stored);
        let ops = take_all_ops(ctx);
        assert!(
            ops.iter().all(|op| op.opcode != OpCode::GetarrayitemGcR),
            "heapcache hit must not record GETARRAYITEM_GC_R: {ops:?}"
        );
    }

    #[test]
    fn collect_virtualizable_boxes_returns_current_state() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        // Before init: None
        assert!(ctx.collect_virtualizable_boxes().is_none());

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // After init: has boxes (field0, field1, vable_ref sentinel)
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![box0, box1, vable]);

        // After mutation
        let nonstandard = decide_nonstandard(&mut ctx, vable, &fd8);
        ctx.vable_setfield_checked(nonstandard, vable, fd8, new_val, Some(ph(Type::Int)));
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![new_val, box1, vable]);
    }

    #[test]
    fn gen_store_back_in_vable_uses_field_and_array_descrs() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Ref,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Ref),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let box_arr1 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );

        ctx.gen_store_back_in_vable(vable);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[0].getdescr().map(|d| d.index()),
            Some(info.static_field_struct_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].getdescr().map(|d| d.index()),
            Some(info.array_pointer_struct_descr(0).index())
        );
        assert_eq!(ops[2].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[2].getdescr().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[3].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[3].getdescr().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[4].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[4].getdescr().map(|d| d.index()),
            Some(info.token_field_descr().index())
        );
    }

    #[test]
    fn gen_store_back_in_vable_ignores_nonstandard_virtualizable() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let other_vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_store_back_in_vable(other_vable);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "nonstandard virtualizable must not use standard store-back path"
        );
    }

    #[test]
    fn emit_vable_field_reads_emits_compile_py_shape() {
        // compile.py patch_new_loop_to_load_virtualizable_fields shape:
        //   [GETFIELD_GC_I(vable, pc_descr),
        //    GETFIELD_GC_R(vable, locals_array_descr),
        //    GETARRAYITEM_GC_I(arr, 0, item_descr),
        //    GETARRAYITEM_GC_I(arr, 1, item_descr),
        //    GETARRAYITEM_GC_I(arr, 2, item_descr)]
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let expanded = ctx.emit_vable_field_reads(vable, &info, &[3]);
        assert_eq!(
            expanded.len(),
            4,
            "1 scalar + 3 array items = 4 expanded slots"
        );

        let ops = take_all_ops(ctx);
        // 1 GETFIELD_GC (pc) + 1 GETFIELD_GC_R (locals ptr) + 3 GETARRAYITEM_GC = 5 ops.
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(
            ops[0].getdescr().map(|d| d.index()),
            Some(info.static_field_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].getdescr().map(|d| d.index()),
            Some(info.array_pointer_field_descr(0).index())
        );
        for k in 0..3 {
            assert_eq!(ops[2 + k].opcode, OpCode::GetarrayitemGcI);
            assert_eq!(
                ops[2 + k].getdescr().map(|d| d.index()),
                Some(info.array_item_descr(0).index())
            );
        }
    }

    #[test]
    fn gen_load_from_other_virtualizable_reloads_a_different_frame() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let origin = recorder.record_input_arg(Type::Ref);
        let dest = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            origin,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_load_from_other_virtualizable_with_lengths(dest, 0, &[2]);

        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes.len(), 4, "pc + 2 array items + dest identity");
        assert_eq!(*boxes.last().unwrap(), dest);
        assert_eq!(ctx.virtualizable_array_lengths(), Some(&[2][..]));
        // dest_ptr 0 records GETFIELD/GETARRAYITEM with no resvalue;
        // each box's value is its own recorded result, not a placeholder.
        assert!(
            ctx.box_value(boxes[0]).is_none()
                && ctx.box_value(boxes[1]).is_none()
                && ctx.box_value(boxes[2]).is_none(),
            "reloaded field boxes carry only their recorded result"
        );
        assert_eq!(ctx.box_value(dest), ctx.box_value(*boxes.last().unwrap()));

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[2].opcode, OpCode::GetarrayitemGcI);
        assert_eq!(ops[3].opcode, OpCode::GetarrayitemGcI);
    }

    #[test]
    fn gen_load_from_other_virtualizable_extends_same_object_array() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_load_from_other_virtualizable_with_lengths(vable, 0, &[2]);

        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes.len(), 4, "pc + 2 array items + identity");
        assert_eq!(boxes[0], box_pc, "live static prefix is kept");
        assert_eq!(boxes[1], box_arr0, "live array prefix is kept");
        assert_eq!(*boxes.last().unwrap(), vable);
        assert_ne!(boxes[2], box_arr0);
        assert_eq!(
            ctx.box_value(boxes[0]),
            Some(ph(Type::Int)),
            "live static prefix keeps its own value"
        );
        assert_eq!(
            ctx.box_value(boxes[1]),
            Some(ph(Type::Int)),
            "live array prefix keeps its own value"
        );
        assert!(
            ctx.box_value(boxes[2]).is_none(),
            "appended GETARRAYITEM carries only its recorded result"
        );
        assert_eq!(
            ctx.box_value(*boxes.last().unwrap()),
            Some(ph(Type::Ref)),
            "identity box keeps its own value last"
        );

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetarrayitemGcI);
    }

    fn test_vable_info_one_static_one_array() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    fn test_vable_info_one_static_two_arrays() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "a",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.add_array_field(
            "b",
            Type::Int,
            32,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    /// Two vable arrays, short shadow: `read_boxes` is statics then every
    /// item of array 0 then every item of array 1, identity last. A prefix
    /// that stops inside array 0 used to stay short because the extend
    /// path required `array_fields.len() == 1`.
    #[test]
    fn fill_virtualizable_boxes_extends_two_arrays_to_the_declared_length() {
        let info = test_vable_info_one_static_two_arrays();
        let nstatic = info.num_static_extra_boxes;
        let len0 = 2usize;
        let len1 = 1usize;
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_a0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_a0],
            &[ph(Type::Int), ph(Type::Int)],
            &[len0, len1],
        );
        ctx.fill_virtualizable_boxes_to_declared_layout(None);
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes.len(), nstatic + len0 + len1 + 1);
        assert!(
            boxes.iter().all(|b| !b.is_none()),
            "every declared slot is filled"
        );
        assert_eq!(boxes[0], box_pc);
        assert_eq!(boxes[1], box_a0);
        assert_eq!(*boxes.last().unwrap(), vable);
        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetarrayitemGcI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[3].opcode, OpCode::GetarrayitemGcI);
    }

    /// A declared array slot with no box used to be skipped by
    /// `append_virtualizable_boxes`, emitting a JUMP one arg shorter than
    /// the LABEL registered from `inputarg_types`.
    #[test]
    #[should_panic(expected = "reached_loop_header")]
    fn virtualizable_data_boxes_panics_when_a_declared_slot_is_missing() {
        let info = test_vable_info_one_static_one_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );
        ctx.virtualizable_array_lengths = Some(vec![2]);
        let _ = ctx.virtualizable_data_boxes();
    }

    #[test]
    fn fill_virtualizable_boxes_extends_a_short_array_to_the_declared_length() {
        let info = test_vable_info_one_static_one_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );
        ctx.virtualizable_array_lengths = Some(vec![2]);
        ctx.fill_virtualizable_boxes_to_declared_layout(None);
        let data = ctx.virtualizable_data_boxes();
        assert_eq!(data.len(), 3, "static + 2 array items");
        assert_eq!(data[0], box_pc);
        assert_eq!(data[1], box_arr0);
        assert_ne!(data[2], box_arr0);
        assert!(!data[2].is_none());
        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetarrayitemGcI);
    }

    /// A NONE hole in a live array slot is `read_boxes` wrapping `lst[i]`,
    /// not a heap None. Filling it with CONST_NULL would carry a wrong value
    /// into the next iteration.
    #[test]
    fn fill_virtualizable_boxes_reads_a_live_hole_from_the_heap() {
        let info = test_vable_info_one_static_one_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, OpRef::NONE],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );
        ctx.fill_virtualizable_boxes_to_declared_layout(None);
        let data = ctx.virtualizable_data_boxes();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], box_pc);
        assert!(!data[1].is_none(), "live hole is filled");
        assert!(
            !data[1].is_constant(),
            "live hole is GETARRAYITEM_GC, not CONST_NULL"
        );
        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetarrayitemGcI);
    }

    fn test_vable_info_with_valuestackdepth() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_field("valuestackdepth", Type::Int, 16);
        info.add_array_field(
            "locals",
            Type::Ref,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Ref),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    /// Dead stack-tail (`array index >= valuestackdepth`) is where
    /// `popvalue_maybe_none` stored null, so `read_boxes` wraps None.
    #[test]
    fn fill_virtualizable_boxes_nulls_dead_stack_tail_holes() {
        let info = test_vable_info_with_valuestackdepth();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_vsd = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_vsd, box_arr0, OpRef::NONE],
            &[ph(Type::Int), Value::Int(1), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );
        ctx.fill_virtualizable_boxes_to_declared_layout(Some(1));
        let data = ctx.virtualizable_data_boxes();
        assert_eq!(data.len(), 4, "2 static + 2 array items");
        assert_eq!(data[0], box_pc);
        assert_eq!(data[1], box_vsd);
        assert_eq!(data[2], box_arr0);
        assert!(!data[3].is_none(), "dead-tail hole is filled");
        assert!(
            data[3].is_constant(),
            "dead stack-tail hole becomes typed null"
        );
    }

    /// `None` is `read_boxes`: every hole is the heap value, even an array
    /// index the caller could have marked dead.
    #[test]
    fn fill_virtualizable_boxes_none_reads_every_hole_from_the_heap() {
        let info = test_vable_info_with_valuestackdepth();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_vsd = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_vsd, box_arr0, OpRef::NONE],
            &[ph(Type::Int), Value::Int(1), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );
        ctx.fill_virtualizable_boxes_to_declared_layout(None);
        let data = ctx.virtualizable_data_boxes();
        assert_eq!(data.len(), 4);
        assert!(!data[3].is_none(), "hole is filled");
        assert!(
            !data[3].is_constant(),
            "None boundary reads GETARRAYITEM_GC, not typed null"
        );
    }

    /// `vable_snapshot_buildable` is the precondition the walker checks
    /// before capturing a resume snapshot; a false answer is reported as
    /// `GuardSnapshotVableUntyped` and aborts to interpretation.  What it
    /// guards is `build_vable_snapshot_boxes`, whose two `.expect()` calls
    /// panic on an untyped entry — see
    /// `build_vable_snapshot_boxes_panics_on_an_untyped_entry`, which pins
    /// the other half of the pair.
    ///
    /// `OpRef::ty()` answers `None` for exactly `None` and `TempVar`
    /// (resoperation.rs), so an unseeded slot is what makes a box untyped.
    #[test]
    fn an_untyped_virtualizable_box_is_not_snapshot_buildable() {
        let mut ctx = TraceCtx::for_test(0);

        // No virtualizable at all: vacuously buildable, so a walk that never
        // seeded `virtualizable_boxes` must not take the abort.
        assert!(ctx.virtualizable_boxes.is_none());
        assert!(ctx.vable_snapshot_buildable());

        // Every slot typed, identity last: buildable.
        let identity = OpRef::ref_op(7);
        ctx.virtualizable_boxes = Some(vec![OpRef::int_op(3), identity]);
        assert!(ctx.vable_snapshot_buildable());

        // A non-identity slot left unseeded.
        ctx.virtualizable_boxes = Some(vec![OpRef::NONE, identity]);
        assert!(!ctx.vable_snapshot_buildable());

        // The identity slot itself left unseeded.
        ctx.virtualizable_boxes = Some(vec![OpRef::int_op(3), OpRef::NONE]);
        assert!(!ctx.vable_snapshot_buildable());
    }

    /// `[len][item0][item1][item2]`: the length word at offset 0, items from
    /// offset 8 — the shape `get_field_arraylen_descr` describes.
    fn boxed_int_array(items: &[i64]) -> (Box<Vec<i64>>, i64) {
        let mut words = vec![items.len() as i64];
        words.extend_from_slice(items);
        let storage = Box::new(words);
        let base = storage.as_ptr() as i64;
        (storage, base)
    }

    fn int_array_descr(with_lendescr: bool) -> DescrRef {
        let lendescr: Option<DescrRef> = with_lendescr.then(|| {
            std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new(
                0,
                0,
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
            )) as DescrRef
        });
        majit_ir::descr::make_array_descr_from_lltype_shape(
            1,
            std::mem::size_of::<i64>(),
            std::mem::size_of::<i64>(),
            None,
            Type::Int,
            false,
            false,
            true,
            true,
            lendescr,
            false,
            u32::MAX,
            Vec::new(),
        ) as DescrRef
    }

    /// `ARRAYLEN_GC` is always-pure, so a constant array with a trace-time
    /// length folds; every other combination records.
    #[test]
    fn arraylen_gc_folds_only_a_constant_array_with_a_concrete_length() {
        let cpu = crate::cpu::default_cpu();
        let (_storage, base) = boxed_int_array(&[10, 20, 30]);

        let mut ctx = TraceCtx::for_test(0);
        let len = ctx.opimpl_arraylen_gc(
            cpu.as_ref(),
            OpRef::const_ptr(majit_ir::GcRef(base as usize)),
            int_array_descr(true),
            Some(Value::Int(3)),
            0,
        );
        assert_eq!(len, OpRef::const_int(3));
        assert!(ctx.into_recorder().ops().is_empty());

        // No trace-time concrete: record, and leave the op unstamped.
        let mut ctx = TraceCtx::for_test(0);
        let len = ctx.opimpl_arraylen_gc(
            cpu.as_ref(),
            OpRef::const_ptr(majit_ir::GcRef(base as usize)),
            int_array_descr(true),
            None,
            0,
        );
        assert!(!len.is_constant());

        // A descr with no `lendescr` is one the executor row cannot read; the
        // concrete is withheld so the funnel records instead of failing loud.
        let mut ctx = TraceCtx::for_test(0);
        let len = ctx.opimpl_arraylen_gc(
            cpu.as_ref(),
            OpRef::const_ptr(majit_ir::GcRef(base as usize)),
            int_array_descr(false),
            Some(Value::Int(3)),
            0,
        );
        assert!(!len.is_constant());

        // Non-constant array base.
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let len = ctx.opimpl_arraylen_gc(
            cpu.as_ref(),
            OpRef::input_arg_ref(0),
            int_array_descr(true),
            Some(Value::Int(3)),
            0,
        );
        assert!(!len.is_constant());
    }

    /// `reached_loop_header` starts with `heapcache.reset()` and drops the
    /// prepared SAVE_EXCEPTION box. A later `cut_trace_from` must not see
    /// that pre-cut producer through `bridge_saved_exc_op`.
    #[test]
    fn reached_loop_header_consumes_prepared_bridge_exception_resume() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let saved = ctx.save_exception();
        ctx.seed_bridge_exception_resume(saved);
        assert!(ctx.bridge_exception_resume_prepared());
        assert_eq!(ctx.bridge_saved_exc_op(), Some(saved));
        assert_eq!(ctx.bridge_exception_resume_pc(), Some(0));
        let mut redboxes = [(OpRef::input_arg_ref(0), Type::Ref)];
        let _ = ctx.reached_loop_header_live_arg_boxes(&mut redboxes, None);
        assert!(!ctx.bridge_exception_resume_prepared());
        assert_eq!(ctx.bridge_saved_exc_op(), None);
        assert_eq!(ctx.bridge_exception_resume_pc(), None);
        assert_eq!(ctx.bridge_exception_resume_jitcode(), None);
        assert_eq!(ctx.bridge_exception_source_pc(), None);
        assert_eq!(ctx.bridge_exception_source_jitcode(), None);
    }
}

/// `MAJIT_PROBE_SUBSCR` startup-only diagnostic gate, read once: changes made
/// with `set_var` / `remove_var` after the first lookup do not take effect.
/// This sits on the path every recorded call takes, and `getenv` per operation
/// was 4% of tracing.
fn probe_subscr_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_PROBE_SUBSCR").is_some())
}
