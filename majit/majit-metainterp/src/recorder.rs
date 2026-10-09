/// Trace recorder — records IR operations during interpreter execution.
///
/// The recorder is the bridge between the interpreter and the JIT.
/// When tracing is active, every interpreter operation is fed to the
/// recorder, which builds a linear sequence of IR operations (a trace).
///
/// Upstream counterpart: `rpython/jit/metainterp/opencoder.py class Trace`
/// (raw op storage, `record_op`, `cut_point`, `cut_at`, `set_inputargs`).
/// The higher-level `History` wrapper (history.py:714) that creates
/// FrontendOps and provides `record_nospec` / `record_same_as` lives in
/// `history.rs` as `impl TraceCtx`.  Pyre callers reach the recorder via
/// `MetaInterp.history.record*` mirroring
/// `pyjitpl.py+ self.history.record2(...)`.
use crate::opencoder::{Box as OcBox, TraceRecordBuffer};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value};
use std::cell::Cell;

/// Compact stand-in for `history.py FrontendOp` while the live recorder
/// writes `TraceRecordBuffer` bytes. The 240-byte `Op` is materialized
/// once, at `into_parts` / `get_iter`, matching `opencoder.py TraceIterator.next`
/// (`cls()`).
///
/// The slot's index in `Trace.slots` names a specific void op
/// (`record` returns `VoidOp(seq)`). Opcode is the compact all-ops
/// table for `opcode_at` / last-guard walks. `descr` holds a foriter
/// marker stamped after record: a guard's stream descr slot is
/// `rd_resume_position`, so that marker cannot live there.
/// `descr_pos` is the byte offset of that stream slot, so a delayed
/// restamp patches it without walking `_ops`.
#[derive(Clone, Debug)]
struct FrontendSlot {
    opcode: OpCode,
    descr: Option<DescrRef>,
    descr_pos: Option<usize>,
}

/// Dense value-producing FrontendOp table, indexed by `_index - n`
/// (`history.py` `IntFrontendOp` / `RefFrontendOp` / `FloatFrontendOp`).
/// Concrete `_res*` lives here. The class of the FrontendOp is `ty`.
#[derive(Clone, Debug)]
struct ValueSlot {
    ty: Type,
    concrete: Cell<Option<Value>>,
    /// `history.py` `FrontendOp.getopnum` — the FrontendOp carries its opnum.
    opcode: OpCode,
    /// `history.py` `RefFrontendOp._heapc_flags` / `_heapc_deps` and
    /// `FrontendOp.position_and_flags & FO_REPLACED_WITH_CONST`.
    heapc: majit_trace::heapcache::HeapcRecord,
}

/// opencoder.py `cut_point()` — RPython 5-tuple
/// `(_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data))`.
///
/// The byte-stream recorder (`TraceRecordBuffer`) fills in every field
/// from its byte cursor / counter state.  The legacy `Vec<Op>` recorder
/// (`recorder::Trace`, being migrated away) maps `_pos` to
/// the ops-Vec cursor (number of ops currently stored). `_count` mirrors
/// the total number of recorded ops, while `_index` mirrors the number of
/// box-yielding positions (inputargs + non-void ops), matching
/// opencoder.py's split counters even though the legacy `Vec<Op>` recorder
/// still assigns `OpRef` positions in total-op order. Snapshot lens come
/// from the recorder-owned `snapshots` side table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePosition {
    /// opencoder.py:475 `self._pos` — byte cursor for TRB, ops-Vec cursor
    /// for `recorder::Trace`.
    pub _pos: usize,
    /// opencoder.py:497 `self._count` — total op count (including voids).
    pub _count: u32,
    /// opencoder.py:498 `self._index` — count of box-yielding (non-void)
    /// ops; equals `_count` in `recorder::Trace`.
    pub _index: u32,
    /// opencoder.py `Trace.cut_point` `len(self._snapshot_data)`.
    pub snapshot_data_len: usize,
    /// opencoder.py `Trace.cut_point` `len(self._snapshot_array_data)`.
    pub snapshot_array_data_len: usize,
}

impl TracePosition {
    /// Recorded-op index into a materialized `TreeLoop.ops`.
    ///
    /// `opencoder.py cut_point` `_count` is inputargs + recorded ops;
    /// `TreeLoop.ops` is recorded ops only. `_pos` is the byte cursor
    /// in byte mode and must not be used as this index.
    pub fn tree_loop_op_index(&self, num_inputargs: usize) -> usize {
        (self._count as usize).saturating_sub(num_inputargs)
    }

    /// True when this cut point sits after at least one recorded op.
    ///
    /// `compile.py compile_loop` compares `start != (0, 0, 0, 0, 0)`.
    /// The zero tuple is a sentinel, not "byte offset 0": after
    /// `Trace.__init__` the live `_pos` is already `max_num_inputargs`.
    pub fn has_prefix_ops(&self, num_inputargs: usize) -> bool {
        (self._count as usize) > num_inputargs
    }
}

/// opencoder.py Snapshot parity: per-guard snapshot of the interpreter
/// frame state, encoded as tagged references to boxes.
///
/// RPython stores snapshots inline in the trace byte stream
/// (`_snapshot_data` / `_snapshot_array_data`).  The live recorder writes
/// that stream; `Vec<Snapshot>` is rebuilt at compile (`into_tree_loop`)
/// so TreeLoop / resume look snapshots up by `rd_resume_position`.
/// Each snapshot captures the live variables of each frame in the call
/// stack at the guard point.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// `_snapshot_data` byte offset (`opencoder.py` `create_top_snapshot`).
    /// This is `GuardResOp.rd_resume_position` / `get_snapshot_iter(index)`.
    /// Sequential 0, 1, 2… when captured on the `Vec<Snapshot>` recorder.
    /// `-1` means unset (struct literals / the encoder input).
    pub resume_position: i32,
    /// Frames in the snapshot, outermost first.
    pub frames: Vec<SnapshotFrame>,
    /// Virtualizable box references (tagged).
    pub vable_boxes: Vec<SnapshotTagged>,
    /// VirtualRef box references (tagged).
    pub vref_boxes: Vec<SnapshotTagged>,
}

impl Snapshot {
    /// Forward every inline gcref this snapshot carries.
    ///
    /// `MIFrame.get_list_of_active_snapshot_boxes` copies a constant ref
    /// register's already-forwarded gcref out of its `OpRef::ConstPtr` into a
    /// raw [`SnapshotTagged::Const`] word, and `jitcode.constants_r` entries
    /// reach the same arm. The side table these land in accumulates for the
    /// whole trace and is only handed to the compiler at the end of it, so
    /// without this walk a collection between capture and compile leaves the
    /// resume data naming a pre-move address. RPython has no such copy: its
    /// snapshot holds the `ConstPtr` box itself, whose `value` the GC traces
    /// through the object graph.
    pub fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        let tagged = self
            .frames
            .iter_mut()
            .flat_map(|f| f.boxes.iter_mut())
            .chain(self.vable_boxes.iter_mut())
            .chain(self.vref_boxes.iter_mut());
        for t in tagged {
            match t {
                SnapshotTagged::Const(bits, Type::Ref) => {
                    let mut gcref = majit_ir::GcRef(*bits as usize);
                    visitor(&mut gcref);
                    *bits = gcref.0 as i64;
                }
                // `build_vable_snapshot_boxes` / `build_vref_snapshot_boxes`
                // tag their entries by `OpRef::ty()` without asking whether the
                // operand is constant, so this arm can hold an inline gcref too.
                SnapshotTagged::Box(OpRef::ConstPtr(gcref), _) => visitor(gcref),
                SnapshotTagged::Const(..) | SnapshotTagged::Box(..) => {}
            }
        }
    }

    /// Guard `rd_resume_position` is the `_snapshot_data` byte offset
    /// (`opencoder.py` `create_top_snapshot` / `get_snapshot_iter(index)`),
    /// not a dense `Vec` index. Decode pushes snapshots in increasing
    /// offset order (`decode_captured_snapshots`).
    pub fn by_resume_position(snapshots: &[Snapshot], resume_position: i32) -> Option<&Snapshot> {
        if resume_position < 0 {
            return None;
        }
        snapshots
            .binary_search_by_key(&resume_position, |s| s.resume_position)
            .ok()
            .map(|i| &snapshots[i])
    }
}

/// One frame in a snapshot — corresponds to one MIFrame/JitCode position.
#[derive(Clone, Debug)]
pub struct SnapshotFrame {
    /// Index of the jitcode (or 0 for the root portal).
    pub jitcode_index: u32,
    /// Program counter within the jitcode: the JitCode byte offset, as the
    /// MIFrame's `pc` field is upstream (`pyjitpl.py setposition`). Both
    /// writers stamp a JitCode offset — `build_state_field_snapshot` reads
    /// `MIFrame::pc` directly, and `capture_snapshot_for_last_guard_multi_frame`
    /// takes it from the walker's own `build_framestack_snapshot`. Resume
    /// recovers a Python pc with `resume_py_pc_for_jitcode_word`; the
    /// recorder does not store a copy.
    pub pc: u32,
    /// Tagged references to the live boxes in this frame.
    pub boxes: Vec<SnapshotTagged>,
}

/// opencoder.py _encode trace-snapshot encode parity: tagged reference to a
/// live value at a recorder snapshot site.  The recorder only sees Box
/// (live in deadframe fail_args) and Const (compile-time constant)
/// payloads; TAGVIRTUAL belongs to the resume-numbering layer
/// (resume.py:_number_boxes) and is synthesized later from
/// PtrInfo::is_virtual on the live Box's OpRef.  Keeping a `Virtual`
/// variant here would let a virtual index reach
/// `translate_trace_iter_opref`'s _cache as a Box position, silently
/// remapping it (recorder.rs has no Box at that integer).  The variant
/// is therefore intentionally absent — adding a virtual-tagged source
/// requires a dedicated resume enum, not this snapshot type.
///
/// TAGBOX(n)    → value lives in `fail_args[n]` (deadframe slot n)
/// TAGCONST(v)  → compile-time constant (i64 value)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotTagged {
    /// Value from deadframe fail_args slot.
    /// RPython: InputArgRef/InputArgInt carry type ('r'/'i'/'f') in
    /// the Box class itself. Pyre stores the typed `OpRef` so the
    /// variant tag (InputArg{Int,Float,Ref} / IntOp/FloatOp/RefOp)
    /// reaches `_number_boxes` (resume.py `box.type == 'r' vs
    /// 'i'`). The trailing `Type` is a redundant lockstep slot kept
    /// for callers that need the type without re-deriving it via
    /// `opref.ty()`.
    Box(majit_ir::OpRef, majit_ir::Type),
    /// Compile-time constant value with type.
    /// RPython resume.py getconst: Const boxes carry their type (INT/REF/FLOAT)
    /// for correct TAGINT/TAGCONST encoding in rd_numb.
    Const(i64, majit_ir::Type),
}

pub struct Trace {
    /// Recorded operations. Stored as `Rc<Op>` so a single `Op` identity
    /// flows from the recorder through the `TreeLoop` handoff into the
    /// optimizer (`AbstractValue` object identity).
    ops: Vec<OpRc>,
    /// Input arguments to the trace (live variables at the loop header).
    /// `History.set_inputargs` stores the hole-filtered list: dead fail args
    /// are absent, and each live `InputArg` keeps its original TAGBOX
    /// position (`get_position()`). Stored as `InputArgRc` so the recorder's
    /// `box_args` bridge can resolve a canonical, identity-stable `Operand`
    /// per input arg (`from_bound_inputarg` carries the `Rc<InputArg>`
    /// directly), and the same `Rc<InputArg>` flows through `into_parts` /
    /// `from_oprc` into `TreeLoop.inputargs` unchanged.
    inputargs: Vec<InputArgRc>,
    /// `opencoder.py Trace.__init__(max_num_inputargs)` reserved prefix.
    /// The byte stream's `_start` / `_count` / `_index` baseline; live
    /// `inputargs` may be shorter when a bridge drops dead fail args.
    max_num_inputargs: u32,
    /// Next OpRef index to assign.
    op_count: u32,
    /// opencoder.py parity: count of box-yielding positions
    /// (inputargs + non-void ops).
    box_count: u32,
    /// Live JIT path: `History.trace` is `opencoder.Trace`. `record_*`
    /// appends bytes and a [`FrontendSlot`]. `into_parts` / `get_iter`
    /// materializes through `ByteTraceIter` (`cls()`). Tests that
    /// construct `Trace::new()` without `metainterp_sd` lazy-attach a
    /// dummy buffer on the first `record_*`.
    trb: Option<Box<TraceRecordBuffer>>,
    slots: Vec<FrontendSlot>,
    /// Value-producing FrontendOps, dense in `_index` after the inputarg
    /// prefix. `OpRef.raw()` for a value op *is* that `_index`.
    value_slots: Vec<ValueSlot>,
    /// FrontendOp heapc records for inputargs at positions `0.._start`
    /// (`warmstate.py` `wrap` builds `RefFrontendOp(position, value)`).
    inputarg_heapc: Vec<majit_trace::heapcache::HeapcRecord>,
}

/// TAGBOX index to the recording `OpRef`, without borrowing the whole
/// recorder, so a `SnapshotIterator` can hold the snapshot bytes at the
/// same time. The tagged value *is* `_index` (`opencoder.py` `_encode`
/// `tag(TAGBOX, box.get_position())`).
fn box_index_to_opref_parts(
    inputargs: &[InputArgRc],
    value_slots: &[ValueSlot],
    box_index: u32,
    max_num_inputargs: u32,
) -> OpRef {
    if let Some(ia) = inputarg_at_position(inputargs, box_index) {
        return OpRef::input_arg_typed(box_index, ia.tp.get());
    }
    let vs = box_index
        .checked_sub(max_num_inputargs)
        .and_then(|i| value_slots.get(i as usize))
        .unwrap_or_else(|| panic!("decode snapshot: TAGBOX({box_index}) has no value FrontendOp"));
    OpRef::op_typed(box_index, vs.ty)
}

/// `opencoder.py` `AbstractResOpOrInputArg.get_position()` on the
/// hole-filtered `History.set_inputargs` list.
fn inputarg_at_position(inputargs: &[InputArgRc], position: u32) -> Option<&InputArgRc> {
    if let Some(ia) = inputargs.get(position as usize)
        && ia.index == position
    {
        return Some(ia);
    }
    inputargs
        .binary_search_by_key(&position, |ia| ia.index)
        .ok()
        .map(|i| &inputargs[i])
}

/// `Trace::untag_snapshot` over already-split const pools.
fn untag_snapshot_pools(
    inputargs: &[InputArgRc],
    value_slots: &[ValueSlot],
    trb: &TraceRecordBuffer,
    tagged: i64,
) -> SnapshotTagged {
    use crate::opencoder::{TAG_MASK, TAG_SHIFT, TAGBOX, TAGCONSTOTHER, TAGCONSTPTR, TAGINT};
    let tag = (tagged & TAG_MASK as i64) as u8;
    let v = tagged >> TAG_SHIFT;
    match tag {
        TAGBOX => {
            debug_assert!(v >= 0, "TAGBOX value must be non-negative, got {v}");
            let opref =
                box_index_to_opref_parts(inputargs, value_slots, v as u32, trb.max_num_inputargs);
            SnapshotTagged::Box(opref, opref.ty().unwrap_or(Type::Int))
        }
        TAGINT => SnapshotTagged::Const(v, Type::Int),
        TAGCONSTPTR => SnapshotTagged::Const(trb.current_ref(v as usize) as i64, Type::Ref),
        TAGCONSTOTHER => {
            let pool_idx = (v >> 1) as usize;
            if v & 1 != 0 {
                SnapshotTagged::Const(trb._floats[pool_idx] as i64, Type::Float)
            } else {
                SnapshotTagged::Const(trb._bigints[pool_idx], Type::Int)
            }
        }
        other => panic!("decode snapshot: unknown tag {other}"),
    }
}

/// Untag one snapshot word and rewrite it into the prepare cache's namespace.
fn snapshot_box_from_tagged<'c>(
    tagged: i64,
    inputargs: &[InputArgRc],
    value_slots: &[ValueSlot],
    trb: &TraceRecordBuffer,
    unique_cache: &'c [Option<Operand>],
) -> (crate::resume::SnapshotBox, Option<&'c Operand>) {
    let decoded = untag_snapshot_pools(inputargs, value_slots, trb, tagged);
    let snap_box = crate::pyjitpl::snapshot_tagged_to_box(&decoded, inputargs);
    // opencoder.py `SnapshotIterator._untag` returns `_cache[i]`, the box
    // object itself. `_cache` is `_index`-keyed.
    let cached = crate::pyjitpl::trace_iter_cached_box(snap_box.opref(), unique_cache);
    (
        snap_box.map_opref(|opref| crate::pyjitpl::translate_trace_iter_opref(opref, unique_cache)),
        cached,
    )
}

/// One byte-mode bridge's resume source.
///
/// `ResumeDataLoopMemo.number` walks `Trace.get_snapshot_iter` when a guard
/// is numbered. This holds that buffer plus the prepare-time cache that
/// rewrites recording `OpRef`s into the optimizer's `_fresh` namespace.
/// A guard the optimizer never emits never builds its box vectors.
///
/// `recorder` stays valid while `MetaInterp.tracing` owns the `Trace`.
/// `OptContext::reset_keep_capacity` drops this value without reading the
/// pointer, before a later compile can free the trace.
pub(crate) struct ByteBridgeResume {
    recorder: *const Trace,
    unique_cache: Vec<Option<Operand>>,
}

impl ByteBridgeResume {
    pub(crate) fn from_recorder(recorder: &Trace, unique_cache: Vec<Option<Operand>>) -> Self {
        Self {
            recorder: recorder as *const Trace,
            unique_cache,
        }
    }

    /// Highest position a numbered snapshot box can resolve to: every
    /// TAGBOX goes through `unique_cache`.
    pub(crate) fn max_box_position(&self) -> Option<u32> {
        self.unique_cache
            .iter()
            .flatten()
            .map(|operand| operand.to_opref())
            .filter(|opref| !opref.is_none() && !opref.is_constant())
            .map(|opref| opref.raw())
            .max()
    }

    pub(crate) fn contains(&self, resume_pos: i32) -> bool {
        if resume_pos < 0 {
            return false;
        }
        // The trace outlives optimize. Reset clears the pointer without
        // dereferencing it. `get_snapshot_iter(index)` reads `_snapshot_data`
        // at that byte offset; membership in the capture-order side list
        // is not the coordinate.
        let rec = unsafe { &*self.recorder };
        rec.trb
            .as_ref()
            .is_some_and(|trb| (resume_pos as usize) < trb._snapshot_data.len())
    }

    pub(crate) fn number_guard(
        &self,
        memo: &mut crate::resume::ResumeDataLoopMemo,
        env: &dyn crate::resume::BoxEnv,
        resume_pos: i32,
        minimum_virtualizable_size: i64,
    ) -> Result<crate::resume::NumberingState, crate::resume::TagOverflow> {
        let rec = unsafe { &*self.recorder };
        rec.number_byte_snapshot(
            resume_pos,
            &self.unique_cache,
            memo,
            env,
            minimum_virtualizable_size,
        )
    }
}

impl Trace {
    /// Create a new, empty trace recorder.
    ///
    /// opencoder.py Trace.__init__ — trace_limit is enforced at the
    /// MetaInterp / TraceCtx level by consulting warmstate.trace_limit,
    /// not stored on the recorder. The first `record_*` attaches a
    /// `TraceRecordBuffer` (`ensure_byte_buffer`); `record_input_arg`
    /// must run before that attach.
    pub fn new() -> Self {
        Trace {
            ops: Vec::with_capacity(256),
            inputargs: Vec::new(),
            max_num_inputargs: 0,
            op_count: 0,
            box_count: 0,
            trb: None,
            slots: Vec::new(),
            value_slots: Vec::new(),
            inputarg_heapc: Vec::new(),
        }
    }

    /// Create a trace recorder pre-configured for retracing from a guard.
    ///
    /// The recorder starts with `num_inputs` int-typed input args,
    /// matching the guard's fail_args.
    pub fn with_num_inputs(num_inputs: usize) -> Self {
        Self::with_input_types(&vec![Type::Int; num_inputs])
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with explicit input arg types.
    pub fn with_input_types(input_types: &[Type]) -> Self {
        let mut recorder = Self::new();
        for tp in input_types {
            recorder.record_input_arg(*tp);
        }
        recorder
    }

    /// Reserve the full guard failarg coordinate space while storing only
    /// live `InputArg`s, each keeping its original `get_position()`.
    /// `opencoder.py Trace.__init__(max_num_inputargs)` still reserves the
    /// dense prefix; dead fail args are simply absent from `inputargs`
    /// (`History.set_inputargs`).
    pub fn with_input_layout(input_types: &[Type], live_inputs: &[bool]) -> Self {
        assert_eq!(input_types.len(), live_inputs.len());
        let mut recorder = Self::new();
        for (&tp, &live) in input_types.iter().zip(live_inputs.iter()) {
            if live {
                recorder.record_input_arg(tp);
            } else {
                recorder
                    .inputarg_heapc
                    .push(majit_trace::heapcache::HeapcRecord::default());
                recorder.max_num_inputargs += 1;
                recorder.op_count += 1;
                recorder.box_count += 1;
            }
        }
        recorder
    }

    /// Attach `opencoder.Trace` as the live storage. `History.__init__`
    /// (`history.py`) builds that buffer with `metainterp_sd` once the
    /// inputarg cap is known — after `initialize_virtualizable` has
    /// appended every vable box (`pyjitpl.py create_empty_history`).
    /// Existing inputargs are replayed into the buffer and later
    /// `record_*` calls write bytes instead of `Rc<Op>`.
    pub fn attach_byte_buffer(
        &mut self,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) {
        if self.trb.is_some() || !self.ops.is_empty() || !self.slots.is_empty() {
            return;
        }
        let n = self.max_num_inputargs;
        let mut trb = TraceRecordBuffer::new(n, metainterp_sd);
        trb.set_inputargs(
            self.inputargs
                .iter()
                .map(|ia| InputArg::from_type(ia.tp.get(), ia.index))
                .collect(),
        );
        // Bridge traces are tens of ops; grow the tables once instead of
        // doubling through the 16/32/64-byte size classes on every record.
        self.slots.reserve(128);
        self.value_slots.reserve(128);
        self.trb = Some(Box::new(trb));
    }

    fn byte_mode(&self) -> bool {
        self.trb.is_some()
    }

    /// Attach a dummy `opencoder.Trace` so `record_*` always writes
    /// bytes. Production attaches with the live `metainterp_sd` after
    /// every inputarg exists (`create_empty_history`). Tests that
    /// built `Trace::new()` without that sd get this buffer on the
    /// first recorded op (`history.py` `History.__init__`).
    fn ensure_byte_buffer(&mut self) {
        if self.trb.is_none() {
            self.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        }
    }

    pub fn has_byte_buffer(&self) -> bool {
        self.trb.is_some()
    }

    /// opencoder.py `Trace.tracing_done` — `compile.py compile_trace`
    /// calls this before `optimize_trace`. A tag overflow becomes
    /// `SwitchToBlackhole(ABORT_TOO_LONG)`.
    pub fn tracing_done(&mut self) -> Result<(), crate::pyjitpl::AbortReason> {
        if let Some(trb) = self.trb.as_mut() {
            return trb.tracing_done();
        }
        Ok(())
    }

    /// `opencoder.py append_int` of a value outside `[MIN_VALUE, MAX_VALUE]`.
    /// Sets `tag_overflow`; the next `tracing_done` is `ABORT_TOO_LONG`.
    #[cfg(test)]
    pub(crate) fn append_out_of_range_int(&mut self) {
        let trb = self
            .trb
            .as_mut()
            .expect("append_out_of_range_int requires opencoder.Trace");
        trb.append_int(i64::MAX);
    }

    /// Count of live captured snapshots: guards in the current op
    /// stream whose descr slot is a `_snapshot_data` offset.
    pub fn snapshot_offset_count(&self) -> usize {
        self.captured_resume_positions().len()
    }

    fn captured_resume_positions(&self) -> Vec<usize> {
        let Some(trb) = self.trb.as_ref() else {
            return Vec::new();
        };
        let data_len = trb._snapshot_data.len();
        let mut out = Vec::new();
        for slot in &self.slots {
            let Some(p) = slot.descr_pos else {
                continue;
            };
            let (idx, _) = crate::opencoder::decode_varint_signed(&trb._ops[p..]);
            if idx >= 0 {
                let u = idx as usize;
                if u < data_len && out.last().copied() != Some(u) {
                    out.push(u);
                }
            }
        }
        out
    }

    pub(crate) fn captured_frame_count(&self) -> usize {
        let Some(trb) = self.trb.as_ref() else {
            return 0;
        };
        let mut n = 0;
        for offset in self.captured_resume_positions() {
            n += trb.get_snapshot_iter(offset).framestack.len();
        }
        n
    }

    fn encode_jitcode_index(idx: u32) -> i64 {
        i64::from(idx)
    }

    pub(crate) fn decode_jitcode_index(idx: i64) -> u32 {
        idx as u32
    }

    fn snapshot_tagged_to_box(&self, tagged: SnapshotTagged) -> OcBox {
        match tagged {
            SnapshotTagged::Const(v, Type::Int) => OcBox::ConstInt(v),
            SnapshotTagged::Const(v, Type::Float) => OcBox::ConstFloat(v as u64),
            SnapshotTagged::Const(v, Type::Ref) => OcBox::ConstPtr(v as u64),
            SnapshotTagged::Const(_, Type::Void) => {
                panic!("encode snapshot: Const Void is not a tagged value")
            }
            SnapshotTagged::Box(r, _) => self.arg_to_box(r),
        }
    }

    pub(crate) fn untag_snapshot(&self, tagged: i64) -> SnapshotTagged {
        let trb = self.trb.as_ref().expect("untag_snapshot requires TRB");
        untag_snapshot_pools(&self.inputargs, &self.value_slots, trb, tagged)
    }

    /// `ResumeDataLoopMemo.number` for one captured snapshot.
    ///
    /// `resume_pos` is the `_snapshot_data` byte offset
    /// (`opencoder.py` `get_snapshot_iter(index)`). TAGBOX is `_index`
    /// and resolves through the prepare `_cache`, matching
    /// `opencoder.py` `SnapshotIterator._untag`.
    fn number_byte_snapshot(
        &self,
        resume_pos: i32,
        unique_cache: &[Option<Operand>],
        memo: &mut crate::resume::ResumeDataLoopMemo,
        env: &dyn crate::resume::BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<crate::resume::NumberingState, crate::resume::TagOverflow> {
        use crate::opencoder::SnapshotIterator;
        use smallvec::SmallVec;

        let offset = resume_pos as usize;
        let inputargs = self.inputargs.as_slice();
        let value_slots = self.value_slots.as_slice();
        let trb = self
            .trb
            .as_ref()
            .expect("number_byte_snapshot requires TraceRecordBuffer");
        let it = SnapshotIterator::new(&trb._snapshot_data, &trb._snapshot_array_data, offset);

        let vable_len = it.iter_vable_array().total_length;
        let vref_len = it.iter_vref_array().total_length;
        // Copy the frame offsets out of `framestack` before calling back into
        // `it`: the iterator borrow and `iter_array` cannot overlap.
        let snaps: SmallVec<[usize; 16]> = it.framestack.iter().copied().collect();
        let mut headers: SmallVec<[(i32, i32, usize); 16]> = SmallVec::new();
        for &snap in snaps.iter() {
            let nboxes = it.iter_array(snap).len();
            let (jc, pc) = it.unpack_jitcode_pc(snap);
            headers.push((Self::decode_jitcode_index(jc) as i32, pc as i32, nboxes));
        }

        let mut vable = it.iter_vable_array();
        let mut vref = it.iter_vref_array();
        let mut frame_iters: SmallVec<[_; 16]> = SmallVec::new();
        for &snap in &snaps {
            frame_iters.push(it.iter_array(snap));
        }
        let mut frame_i = 0usize;
        memo.number_sections(
            vable_len,
            || {
                snapshot_box_from_tagged(
                    vable.next().expect("vable snapshot box"),
                    inputargs,
                    value_slots,
                    trb,
                    unique_cache,
                )
            },
            vref_len,
            || {
                snapshot_box_from_tagged(
                    vref.next().expect("vref snapshot box"),
                    inputargs,
                    value_slots,
                    trb,
                    unique_cache,
                )
            },
            &headers,
            || {
                while frame_iters[frame_i].len() == 0 {
                    frame_i += 1;
                }
                let tagged = frame_iters[frame_i].next().expect("frame snapshot box");
                snapshot_box_from_tagged(tagged, inputargs, value_slots, trb, unique_cache)
            },
            env,
            minimum_virtualizable_size,
        )
    }

    /// `_list_of_boxes` from tagged snapshot values, encoding each box
    /// into `_snapshot_array_data` the way `opencoder.py _add_box_to_storage`
    /// does — no intermediate `Vec<i64>`.
    fn write_tagged_array(&mut self, boxes: &[SnapshotTagged]) -> i64 {
        let res = self
            .trb
            .as_mut()
            .expect("write_tagged_array requires attach_byte_buffer")
            .new_array(boxes.len());
        for &tagged in boxes {
            let b = self.snapshot_tagged_to_box(tagged);
            self.trb
                .as_mut()
                .expect("write_tagged_array requires attach_byte_buffer")
                ._add_box_to_storage_box(b);
        }
        res
    }

    /// `opencoder.py create_top_snapshot` / `create_snapshot` from a
    /// structured `Snapshot`. Returns the `_snapshot_data` byte offset
    /// (`create_top_snapshot`); that offset is `rd_resume_position`.
    pub fn encode_captured_snapshot(&mut self, snapshot: &Snapshot) -> i32 {
        // `create_top_snapshot` patches the last op's descr slot when that
        // op is the guard just recorded (`opencoder.py` `_pos -= 2`).
        // A later non-guard must not be rewritten as if it were the
        // placeholder; delayed restamp walks the named guard instead.
        let patch_last = self.last_guard_has_descr_placeholder();
        let offset = if snapshot.frames.is_empty() {
            let empty_array = self.write_tagged_array(&[]);
            let vable_array = self.write_tagged_array(&snapshot.vable_boxes);
            let vref_array = self.write_tagged_array(&snapshot.vref_boxes);
            let trb = self
                .trb
                .as_mut()
                .expect("encode_captured_snapshot requires attach_byte_buffer");
            trb._total_snapshots += 1;
            let s = trb._snapshot_data.len() as i64;
            trb.append_snapshot_data_int(vable_array);
            trb.append_snapshot_data_int(vref_array);
            trb._encode_snapshot(-1, 0, empty_array, true);
            s
        } else {
            let last = snapshot.frames.len() - 1;
            let array = self.write_tagged_array(&snapshot.frames[last].boxes);
            let vable_array = self.write_tagged_array(&snapshot.vable_boxes);
            let vref_array = self.write_tagged_array(&snapshot.vref_boxes);
            let jitcode = Self::encode_jitcode_index(snapshot.frames[last].jitcode_index);
            let pc = i64::from(snapshot.frames[last].pc);
            let is_last = snapshot.frames.len() == 1;
            let s = {
                let trb = self
                    .trb
                    .as_mut()
                    .expect("encode_captured_snapshot requires attach_byte_buffer");
                trb._total_snapshots += 1;
                let s = trb._snapshot_data.len() as i64;
                trb.append_snapshot_data_int(vable_array);
                trb.append_snapshot_data_int(vref_array);
                trb._encode_snapshot(jitcode, pc, array, is_last);
                s
            };
            for i in (0..last).rev() {
                let array = self.write_tagged_array(&snapshot.frames[i].boxes);
                let jitcode = Self::encode_jitcode_index(snapshot.frames[i].jitcode_index);
                let pc = i64::from(snapshot.frames[i].pc);
                self.trb
                    .as_mut()
                    .expect("encode_captured_snapshot requires attach_byte_buffer")
                    .create_snapshot(jitcode, pc, array, i == 0);
            }
            s
        };
        if patch_last {
            self.trb
                .as_mut()
                .expect("encode_captured_snapshot requires attach_byte_buffer")
                .patch_last_guard_descr_slot(offset);
        }
        offset as i32
    }

    fn last_recorded_is_guard(&self) -> bool {
        // Byte mode keeps recording into `slots` after
        // `materialize_into_ops`; `ops` is then a stale copy and must
        // not decide whether the live last op is a guard
        // (`create_top_snapshot` / `patch_last_guard_descr_slot`).
        if self.byte_mode() {
            self.slots.last().is_some_and(|slot| slot.opcode.is_guard())
        } else {
            self.ops.last().is_some_and(|op| op.opcode.is_guard())
        }
    }

    /// `create_top_snapshot` rewinds the last op's descr slot only while
    /// that op is the guard just recorded with `descr=None`
    /// (`pyjitpl.py` `generate_guard` / `opencoder.py` `_op_end`).
    fn last_guard_has_descr_placeholder(&self) -> bool {
        self.last_recorded_is_guard()
            && self
                .trb
                .as_ref()
                .is_some_and(|trb| trb.last_descr_slot_is_placeholder())
    }

    /// opencoder.py `Trace.capture_resumedata(framestack, ...)`.
    /// Writes `_snapshot_data` from the live framestack — no
    /// `Vec<Snapshot>` / `SnapshotTagged` on the record path.
    pub fn capture_resumedata_from_framestack(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        virtualizable_boxes: &[OpRef],
        virtualref_boxes: &[(OpRef, usize)],
        after_residual_call: bool,
        op_live: u8,
        all_liveness: &[u8],
    ) -> i32 {
        // `create_top_snapshot` patches the guard's trailing descr slot
        // (`opencoder.py`) only while that guard is still last and the
        // slot is the 0-placeholder `record_op(..., descr=None)` wrote.
        let last_is_guard = self.last_guard_has_descr_placeholder();
        // opencoder.py Trace.create_top_snapshot encodes the existing box
        // lists directly. `OpRef.raw()` is already `_index`.
        let num_inputs = self.max_num_inputargs as usize;
        let vable = virtualizable_boxes
            .iter()
            .map(|r| Self::arg_to_box_at(*r, num_inputs));
        let vref = virtualref_boxes
            .iter()
            .map(|(r, _)| Self::arg_to_box_at(*r, num_inputs));
        let offset = {
            let trb = self
                .trb
                .as_mut()
                .expect("capture_resumedata_from_framestack requires attach_byte_buffer");
            trb.capture_resumedata_mapped(
                framestack,
                vable,
                vref,
                /* clear_result_register */ true,
                op_live,
                all_liveness,
                after_residual_call,
                last_is_guard,
            )
        };
        offset as i32
    }

    /// Visit opencoder.py SnapshotIterator views of the captured byte stream.
    /// Walks guards that carry a resume position (`get_snapshot_iter(index)`).
    pub(crate) fn for_each_captured_snapshot_arrays(
        &self,
        mut f: impl FnMut(usize, &crate::opencoder::SnapshotIterator<'_>),
    ) -> bool {
        let Some(trb) = self.trb.as_ref() else {
            return false;
        };
        for offset in self.captured_resume_positions() {
            let it = trb.get_snapshot_iter(offset);
            f(offset, &it);
        }
        true
    }

    /// Rebuild `Vec<Snapshot>` from `_snapshot_data` for each live guard
    /// that carries a resume position.
    pub fn decode_captured_snapshots(&self) -> Option<Vec<Snapshot>> {
        let trb = self.trb.as_ref()?;
        let mut offsets = self.captured_resume_positions();
        if offsets.is_empty() && !trb._snapshot_data.is_empty() {
            offsets.push(0);
        }
        let mut out = Vec::with_capacity(offsets.len());
        for offset in offsets {
            // opencoder.py SnapshotIterator keeps box arrays as iterators.
            // Decode directly into the consumer's snapshot, without copying
            // every tagged array to a temporary buffer first.
            let it = trb.get_snapshot_iter(offset);
            let frames = it
                .framestack
                .iter()
                .copied()
                .map(|snap_idx| {
                    let (jc, pc) = it.unpack_jitcode_pc(snap_idx);
                    SnapshotFrame {
                        jitcode_index: Self::decode_jitcode_index(jc),
                        pc: pc as u32,
                        boxes: it
                            .iter_array(snap_idx)
                            .map(|t| self.untag_snapshot(t))
                            .collect(),
                    }
                })
                .collect();
            out.push(Snapshot {
                resume_position: offset as i32,
                frames,
                vable_boxes: it
                    .iter_vable_array()
                    .map(|t| self.untag_snapshot(t))
                    .collect(),
                vref_boxes: it
                    .iter_vref_array()
                    .map(|t| self.untag_snapshot(t))
                    .collect(),
            });
        }
        Some(out)
    }

    fn arg_to_box(&self, r: OpRef) -> OcBox {
        Self::arg_to_box_at(r, self.max_num_inputargs as usize)
    }

    /// Encode a box for TAGBOX. `OpRef.raw()` of a value op *is* `_index`
    /// (`opencoder.py` `_encode` `tag(TAGBOX, box.get_position())`).
    fn arg_to_box_at(r: OpRef, _num_inputs: usize) -> OcBox {
        if r.is_constant() {
            let value = r
                .inline_const_to_value()
                .unwrap_or_else(|| panic!("arg_to_box: constant {r:?} not inline-resolvable"));
            return match value {
                Value::Int(v) => OcBox::ConstInt(v),
                Value::Float(f) => OcBox::ConstFloat(f.to_bits()),
                Value::Ref(g) => OcBox::ConstPtr(g.as_usize() as u64),
                Value::Void => panic!("arg_to_box: constant {r:?} has Void type"),
            };
        }
        assert!(
            r.ty() != Some(Type::Void) && !matches!(r, OpRef::VoidOp(_)),
            "arg_to_box: void op {r:?} used as a value box"
        );
        OcBox::ResOp(r.raw())
    }

    /// `history.py History._make_op`: append the op and attach `value` on
    /// the FrontendOp at construction. Void ops are named by their
    /// op-sequence position (`VoidOp(slots.len())`), not `_index`.
    fn record_bytes(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        value: Option<Value>,
    ) -> OpRef {
        // history.py record0/1/2/3 take the boxes inline. JUMP and
        // other N-ary ops exceed that 0–3 surface; eight OcBoxes stay
        // off the process allocator.
        let boxes: smallvec::SmallVec<[OcBox; 8]> =
            args.iter().copied().map(|a| self.arg_to_box(a)).collect();
        let trb = self
            .trb
            .as_mut()
            .expect("record_bytes requires attach_byte_buffer");
        let void_seq = trb._count;
        // `pyjitpl.py` `generate_guard` records `descr=None`. The stream
        // descr slot is `rd_resume_position` (`opencoder.py`
        // `create_top_snapshot`); a foriter marker stays on the
        // FrontendSlot and is overlaid at materialize.
        let stream_descr = if opcode.is_guard() {
            None
        } else {
            descr.as_ref()
        };
        let box_index = trb.record_op(opcode, &boxes, stream_descr);
        let ty = opcode.result_type();
        // Guard `record_op` writes a 2-byte 0 placeholder; later
        // `create_top_snapshot` / a delayed restamp overwrite it.
        let descr_pos = if opcode.is_guard() && opcode.has_descr() {
            Some(trb._pos.saturating_sub(2))
        } else {
            None
        };
        self.slots.push(FrontendSlot {
            opcode,
            // Guard markers (range foriter) stay on the slot; the stream
            // descr slot is `rd_resume_position`. Non-guard descrs live
            // only in the encoded stream (`_encode_descr`).
            descr: if opcode.is_guard() { descr } else { None },
            descr_pos,
        });
        let opref = if ty != Type::Void {
            self.value_slots.push(ValueSlot {
                ty,
                concrete: Cell::new(value),
                opcode,
                heapc: majit_trace::heapcache::HeapcRecord::default(),
            });
            self.box_count += 1;
            OpRef::op_typed(box_index, ty)
        } else {
            // `_count` is the all-ops sequence (inputargs + recorded ops),
            // the coordinate `TraceIterator._count` assigns to a void `cls()`.
            OpRef::void_op(void_seq)
        };
        self.op_count += 1;
        opref
    }

    /// `_index` / `_count` prefix: `Trace(max_num_inputargs)` / TRB `_start`.
    fn box_prefix(&self) -> u32 {
        self.trb
            .as_ref()
            .map(|t| t._start)
            .unwrap_or(self.max_num_inputargs)
    }

    fn value_slot(&self, box_index: u32) -> Option<&ValueSlot> {
        let n = self.box_prefix();
        box_index
            .checked_sub(n)
            .and_then(|i| self.value_slots.get(i as usize))
    }

    fn value_slot_mut(&mut self, box_index: u32) -> Option<&mut ValueSlot> {
        let n = self.box_prefix();
        box_index
            .checked_sub(n)
            .and_then(|i| self.value_slots.get_mut(i as usize))
    }

    fn slot_by_seq(&self, seq: u32) -> Option<&FrontendSlot> {
        self.slots.get(seq as usize)
    }

    fn last_slot_mut(&mut self) -> Option<&mut FrontendSlot> {
        self.slots.last_mut()
    }

    /// opencoder.py `Trace.get_iter()` materialize: one `ByteTraceIter`
    /// walk. Value ops keep their `_index` position (`FrontendOp.get_position()`);
    /// void ops keep the op-sequence `VoidOp` the iterator assigned.
    pub(crate) fn materialize_ops(&self) -> Vec<OpRc> {
        let Some(trb) = self.trb.as_ref() else {
            return self.ops.clone();
        };
        let mut ops = Vec::with_capacity(self.slots.len());
        ops.extend(crate::opencoder::ByteTraceIter::new(
            trb,
            trb._start as usize,
            trb._pos,
            0,
        ));
        for op in &ops {
            for i in 0..op.num_args() {
                let arg = op.arg(i);
                if arg.is_inputarg() {
                    if let Some(idx) = arg.position() {
                        // `ByteTraceIter` remints the live list densely from
                        // `start_fresh=0`, so the reminted index is the live
                        // list index. Rebind onto the recorder's `InputArgRc`.
                        if let Some(ours) = self.inputargs.get(idx as usize) {
                            op.setarg(i, Operand::from_bound_inputarg(ours));
                        }
                    }
                }
            }
        }
        // `ByteTraceIter` remints live inputargs densely from
        // `start_fresh=0`, so value-op `_fresh` would collide with a
        // surviving `get_position()` (InputArg(2) vs IntAdd at compact 2).
        // Restore `FrontendOp.get_position()` = `_index` (`Trace._start`
        // plus the value-op ordinal), matching `opencoder.py` `cls()`.
        let mut orig = trb._start;
        for (i, op) in ops.iter().enumerate() {
            if op.opcode.result_type() != Type::Void {
                let ty = op.opcode.result_type();
                op.pos().set(OpRef::op_typed(orig, ty));
                if let Some(vs) = self.value_slot(orig)
                    && let Some(v) = vs.concrete.get()
                {
                    op.set_value(v);
                }
                orig += 1;
            }
            if let Some(d) = self.slots.get(i).and_then(|s| s.descr.clone()) {
                op.setdescr(d);
            }
        }
        ops
    }

    /// opencoder.py `Trace.get_iter()` for `optimize_bridge`: one
    /// `ByteTraceIter` walk with the hole-filtered live inputargs.
    /// Inputargs keep their original `_index` (`TraceIterator.__init__`
    /// seeds `_cache` at `force_inputargs[i].get_position()`). Overlay
    /// FrontendOp concrete; resume comes from the stream descr slot
    /// (`TraceIterator.next` `rd_resume_position`). Fail_args come from
    /// `store_final_boxes_in_guard` after numbering. The `_index`-keyed
    /// cache *is* `TraceIterator._cache`.
    pub(crate) fn get_iter_for_optimizer<A: AsRef<InputArg>>(
        &self,
        live_inputargs: &[A],
        start_fresh: u32,
    ) -> Option<(Vec<OpRc>, Vec<InputArgRc>, Vec<Option<Operand>>)> {
        let trb = self.trb.as_ref()?;
        // ByteTraceIter still seeds from `InputArg` (opencoder Trace.inputargs).
        // Types and values come off the same InputArgRc boxes. `start_fresh`
        // is ignored for positions: boxes keep `get_position()`.
        let live_plain: Vec<InputArg> = live_inputargs
            .iter()
            .map(|ia| {
                let ia = ia.as_ref();
                let plain = InputArg::from_type(ia.tp.get(), ia.index);
                if let Some(value) = ia.get_value() {
                    plain.set_value(value);
                }
                plain
            })
            .collect();
        let mut iter = crate::opencoder::ByteTraceIter::new_with_inputargs(
            trb,
            trb._start as usize,
            trb._pos,
            &live_plain,
            start_fresh,
        );
        let mut ops = Vec::with_capacity(self.slots.len());
        while let Some(op) = iter.next() {
            ops.push(op);
        }
        debug_assert_eq!(ops.len(), self.slots.len());

        // Reuse the walk's reminted Rc. A second from_type_rc splits box
        // identity. Stamp FrontendOp value onto those same boxes
        // (`inputarg_from_tp` is type-only; the value lives on the source).
        // `opencoder.py` `inputarg_from_tp` runs once; compile_bridge must
        // see the same boxes as fail_args.
        for (src, ia) in live_inputargs.iter().zip(iter.inputargs.iter()) {
            if let Some(value) = src.as_ref().get_value() {
                ia.set_value(value);
            }
        }

        // Overlay FrontendOp concrete by original `_index`. Value slots
        // are `_index - prefix`; `op.pos()` is the iterator's `_fresh`
        // remint (`TraceIterator` `cls()`), so it is not the slot key.
        // `opencoder.py` `TraceIterator._cache` is keyed by `_index`.
        let mut orig = trb._start;
        for (op, slot) in ops.iter().zip(self.slots.iter()) {
            if op.opcode.result_type() != Type::Void {
                if let Some(vs) = self.value_slot(orig)
                    && let Some(v) = vs.concrete.get()
                {
                    op.set_value(v);
                }
                orig += 1;
            }
            if let Some(d) = slot.descr.clone() {
                op.setdescr(d);
            }
        }

        Some((ops, iter.inputargs, iter._cache))
    }

    /// Register an input argument of the given type.
    /// Returns an OpRef that can be used as an argument to subsequent operations.
    /// Input arguments are numbered starting from 0; the OpRef index matches
    /// the input argument index.
    ///
    /// resoperation.py/727/739 — InputArgInt/InputArgFloat/InputArgRef
    /// each pin `type = 'i'/'f'/'r'` at construction.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        assert!(
            self.ops.is_empty() && self.slots.is_empty(),
            "input args must be registered before any operations"
        );
        // opencoder.py `Trace.__init__(max_num_inputargs)` freezes the
        // reserved prefix. `create_empty_history` therefore runs after
        // `initialize_virtualizable` has appended every vable box
        // (`pyjitpl.py _compile_and_run_once`). Growing the cap after
        // `attach_byte_buffer` would shift `_start`/`_pos` under already-
        // written ops.
        assert!(
            self.trb.is_none(),
            "record_input_arg after attach_byte_buffer: History is created \
             after every inputarg exists (pyjitpl.py create_empty_history \
             after initialize_virtualizable)"
        );
        let index = self.max_num_inputargs;
        debug_assert!(self.inputargs.last().is_none_or(|prev| prev.index < index));
        self.inputargs.push(InputArg::from_type_rc(tp, index));
        self.inputarg_heapc
            .push(majit_trace::heapcache::HeapcRecord::default());
        self.max_num_inputargs += 1;
        let opref = match tp {
            Type::Int => OpRef::input_arg_int(index),
            Type::Float => OpRef::input_arg_float(index),
            Type::Ref => OpRef::input_arg_ref(index),
            Type::Void => panic!("input args cannot be Void"),
        };
        self.op_count += 1;
        self.box_count += 1;
        opref
    }

    /// Bridge the recorder's OpRef-operand API to the `Operand`-carrying
    /// `Op.args` / `Op.fail_args` storage. Each operand resolves to its
    /// *canonical* producer `Operand` so the stored args share one producer
    /// `Rc` (`AbstractValue` object identity): `from_bound_op` /
    /// `from_bound_inputarg` carry the producer `Rc<Op>` / `Rc<InputArg>`
    /// directly. Frame registers and the public `record_*` API stay OpRef; the
    /// optimizer bridges back with `Operand::to_opref`, which round-trips to the
    /// same `OpRef` the `from_opref` view produced.
    fn box_args(&mut self, args: &[OpRef]) -> smallvec::SmallVec<[Operand; 16]> {
        args.iter().map(|&a| self.box_for_operand(a)).collect()
    }

    /// Resolve one operand `OpRef` to its canonical `Operand`. Const operands
    /// carry their `Value` inline on the `OpRef` (history.py:227/268/314) and
    /// are minted via `from_opref`; InputArg / ResOp operands resolve to the
    /// producing `InputArg` / `Op` already held in `self.inputargs` / `self.ops`.
    /// A non-const operand that does not resolve to a recorded producer is a
    /// recorder invariant violation and panics (`opencoder.py` `_encode`
    /// asserts `get_position()` rather than mint); the S0/#121 probe
    /// measured 0 hits across the corpus. `#[cfg(test)]` fixtures that build
    /// position-only synthetic operands bind a synthetic producer via
    /// `bound_from_opref` (`to_opref`-identical) rather than panicking.
    /// Deterministic per recorded position: a ResOp / InputArg operand always
    /// binds to the SAME producer `Rc` (`self.ops` / `self.inputargs` are
    /// immutable once recorded), so two calls for the same `OpRef` return
    /// `Operand`s that compare equal under `Operand::eq` (`Rc::ptr_eq`). This is
    /// the real box object — `MIFrame.registers_r` holds these in `pyjitpl.py` —
    /// so consumers can key by box identity instead of flat `OpRef`.
    pub(crate) fn box_for_operand(&mut self, r: OpRef) -> Operand {
        if r.is_none() || r.is_constant() {
            return Operand::from_opref(r);
        }
        if let OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) = r {
            if let Some(ia) = self.inputarg_at(r.raw()) {
                return Operand::from_bound_inputarg(ia);
            }
            // Live `InputArg`s keep `get_position()`; a missing position is a
            // dead failarg hole or an unrecorded operand (`opencoder.py` `_encode`).
            #[cfg(not(test))]
            panic!(
                "box_for_operand: InputArg operand {r:?} not in live \
                 inputargs (max_num_inputargs={})",
                self.max_num_inputargs
            );
            #[cfg(test)]
            return Operand::bound_from_opref(r);
        }
        // ResOp operand: value-producing `OpRef.raw()` is `_index`
        // (`history.py` `FrontendOp.get_position`). The FrontendOp lives
        // at `value_slots[_index - _start]`. A materialized producer is
        // the op whose result position is that `_index`
        // (`opencoder.py` `TraceIterator._cache[_index]` / `get_op_by_pos`).
        if !r.is_none() && r.ty() != Some(Type::Void) && !matches!(r, OpRef::VoidOp(_)) {
            if self.value_slot(r.raw()).is_some() {
                if let Some(op) = self.ops.iter().find(|op| op.pos().get() == r) {
                    return Operand::from_bound_op(op);
                }
                if self.byte_mode() {
                    return Operand::bound_from_opref(r);
                }
            }
        }
        // S3/#124: a ResOp operand always has a dense producer in `ops`
        // (positions are `_index` order; opencoder.py `_encode` asserts).
        #[cfg(not(test))]
        panic!(
            "box_for_operand: ResOp operand {r:?} has no dense producer in \
             ops[0,{}) (opencoder.py:640)",
            self.ops.len()
        );
        #[cfg(test)]
        Operand::bound_from_opref(r)
    }

    /// Record a regular (non-guard) operation.
    /// Returns the OpRef for this operation's result.
    ///
    /// `AbstractResOp` + IntOp/FloatOp/RefOp mixins (resoperation.py)
    /// pin the result type at construction. Void-result ops keep the
    /// `AbstractResOp.type = 'v'` default (resoperation.py).
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        self.record_op_with_value(opcode, args, None)
    }

    /// `history.py History.record*` — `value` is `_make_op`'s runtime
    /// concrete (`IntFrontendOp(pos, value)`). `None` for void or an
    /// unstamped result.
    pub fn record_op_with_value(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        value: Option<Value>,
    ) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        self.ensure_byte_buffer();
        self.record_bytes(opcode, args, None, value)
    }

    /// Record an operation with a descriptor (e.g., field access, call).
    /// Returns the OpRef for this operation's result.
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr_value(opcode, args, descr, None)
    }

    pub fn record_op_with_descr_value(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
        value: Option<Value>,
    ) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        self.ensure_byte_buffer();
        self.record_bytes(opcode, args, Some(descr), value)
    }

    /// Record a guard operation.
    /// `pyjitpl.py generate_guard()` parity: tracer-stage guards
    /// carry `descr=None`. The optimizer creates the FailDescr later in
    /// `store_final_boxes_in_guard` / `invent_fail_descr_for_op`
    /// (compile.py:722-730 / 924-942). Tests that need a pre-stamped
    /// descr pass `Some(descr)`.
    /// Returns the OpRef for this guard.
    pub fn record_guard(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        self.ensure_byte_buffer();
        self.record_bytes(opcode, args, descr, None)
    }

    /// Set rd_resume_position on the last recorded op.
    /// Called after record_guard* to associate a snapshot.
    /// Byte mode patches the guard's descr slot in `_ops`
    /// (`create_top_snapshot`); the iterator reads it from the stream.
    pub fn set_last_op_resume_position(&mut self, snapshot_id: i32) {
        if self.trb.is_some() {
            self.set_guard_op_resume_position_from_end(0, snapshot_id);
            return;
        }
        if let Some(op) = self.ops.last() {
            op.set_rd_resume_position(snapshot_id);
        }
    }

    /// Set rd_resume_position on the most-recently recorded *guard* op,
    /// skipping any non-guard ops recorded after it.
    ///
    /// `set_last_op_resume_position` assumes the guard is the last op,
    /// which holds when a guard is captured immediately after recording.
    /// A guard emitted *inside* a helper (e.g. the
    /// `_nonstandard_virtualizable` PTR_EQ promote, after which
    /// `emit_force_virtualizable` records GETFIELD_GC / PTR_NE /
    /// COND_CALL) is not the last op when the caller captures, so the
    /// resume position must target the guard by its guard-ness.
    pub fn set_last_guard_op_resume_position(&mut self, snapshot_id: i32) {
        self.set_guard_op_resume_position_from_end(0, snapshot_id);
    }

    /// Set rd_resume_position on the guard op `from_end` guards back from the
    /// most recently recorded one (`0` == the most recent).
    ///
    /// One helper call can emit more than one guard: a vable array access
    /// promotes the `isstandard` PTR_EQ in `_nonstandard_virtualizable`
    /// (`pyjitpl.py`) and then the index in
    /// `_get_arrayitem_vable_index` (`:1201-1216`).  Upstream captures resume
    /// data inside each `implement_guard_value`, so both are stamped; a caller
    /// that only reaches the guards after the helper returns walks back over
    /// them with this.
    pub fn set_guard_op_resume_position_from_end(&mut self, from_end: usize, snapshot_id: i32) {
        if self.trb.is_some() {
            let descr_pos = self
                .slots
                .iter()
                .rev()
                .filter(|s| s.opcode.is_guard())
                .nth(from_end)
                .and_then(|s| s.descr_pos);
            if let Some(pos) = descr_pos {
                let trb = self.trb.as_mut().expect("byte buffer");
                let old_len = crate::opencoder::decode_varint_signed(&trb._ops[pos..]).1;
                trb.patch_descr_slot_at(pos, snapshot_id as i64);
                let new_len = crate::opencoder::decode_varint_signed(&trb._ops[pos..]).1;
                if new_len != old_len {
                    let delta = new_len as isize - old_len as isize;
                    let mut seen = 0;
                    let n_guards = self.slots.iter().filter(|s| s.opcode.is_guard()).count();
                    let target = n_guards.saturating_sub(from_end + 1);
                    for slot in &mut self.slots {
                        if slot.opcode.is_guard() {
                            if seen > target
                                && let Some(p) = slot.descr_pos.as_mut()
                            {
                                *p = (*p as isize + delta) as usize;
                            }
                            seen += 1;
                        }
                    }
                }
            }
        }
        if let Some(op) = self
            .ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
        {
            op.set_rd_resume_position(snapshot_id);
        }
    }

    /// Replace the descriptor on the last recorded operation.
    pub fn set_last_op_descr(&mut self, descr: DescrRef) {
        if let Some(slot) = self.last_slot_mut() {
            slot.descr = Some(descr);
            return;
        }
        if let Some(op) = self.ops.last() {
            op.setdescr(descr);
        }
    }

    /// Opcode of the op [`set_last_op_descr`](Self::set_last_op_descr) would
    /// stamp, for a caller that has to mint the descr subtype the opcode
    /// requires.
    pub fn last_op_opcode(&self) -> Option<OpCode> {
        if let Some(slot) = self.slots.last() {
            return Some(slot.opcode);
        }
        self.ops.last().map(|op| op.opcode)
    }

    /// Opcode of the op
    /// [`set_guard_op_descr_from_end`](Self::set_guard_op_descr_from_end)
    /// would stamp, selected by the same walk.
    pub fn guard_op_opcode_from_end(&self, from_end: usize) -> Option<OpCode> {
        if let Some(slot) = self
            .slots
            .iter()
            .rev()
            .filter(|s| s.opcode.is_guard())
            .nth(from_end)
        {
            return Some(slot.opcode);
        }
        self.ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
            .map(|op| op.opcode)
    }

    /// Resume position of the guard [`set_guard_op_resume_position_from_end`]
    /// would stamp, selected by the same walk. `None` when `from_end` does
    /// not name a recorded guard.
    pub fn guard_op_resume_position_from_end(&self, from_end: usize) -> Option<i32> {
        if let Some(trb) = self.trb.as_ref() {
            let slot = self
                .slots
                .iter()
                .rev()
                .filter(|s| s.opcode.is_guard())
                .nth(from_end)?;
            let p = slot.descr_pos?;
            let (idx, _) = crate::opencoder::decode_varint_signed(&trb._ops[p..]);
            return Some(idx as i32);
        }
        self.ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
            .map(|op| op.rd_resume_position())
    }

    /// Replace the descriptor on the guard `from_end` guards back from the
    /// most recently recorded one.
    pub fn set_guard_op_descr_from_end(&mut self, from_end: usize, descr: DescrRef) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .rev()
            .filter(|s| s.opcode.is_guard())
            .nth(from_end)
        {
            slot.descr = Some(descr);
            return;
        }
        if let Some(op) = self
            .ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
        {
            op.setdescr(descr);
        }
    }

    /// Opcode of the most recently recorded guard, if any.  Snapshot
    /// capture keys `after_residual_call` on the guard opcode itself
    /// (`pyjitpl.py generate_guard`).
    pub fn last_guard_opcode(&self) -> Option<OpCode> {
        if let Some(slot) = self.slots.iter().rev().find(|s| s.opcode.is_guard()) {
            return Some(slot.opcode);
        }
        self.ops
            .iter()
            .rev()
            .find(|op| op.opcode.is_guard())
            .map(|op| op.opcode)
    }

    /// Set fail_args on a recorded op identified by `opref`.
    ///
    /// `resoperation.py` `Op.setfailargs` on the materialized
    /// `GuardResOp`. Production recording stores none; the optimizer's
    /// `store_final_boxes_in_guard` writes them after numbering. Tests
    /// that construct a synthetic guard call this after the `Op` exists
    /// (`Vec<Op>` recorder or post-`materialize_into_ops`).
    pub fn set_op_fail_args(&mut self, opref: OpRef, fail_args: &[OpRef]) {
        self.materialize_into_ops();
        let boxed_fail_args = self.box_args(fail_args).iter().cloned().collect();
        let op = self
            .ops
            .iter()
            .rev()
            .find(|op| op.pos().get() == opref)
            .unwrap_or_else(|| panic!("set_op_fail_args: no op with pos {:?}", opref));
        op.setfailargs(boxed_fail_args);
    }

    /// Close the loop: add a JUMP operation back to the start.
    /// `jump_args` are the values of the input arguments at the end of the loop.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        self.close_loop_with_descr(jump_args, None);
    }

    /// Close the loop with an explicit JUMP descriptor.
    ///
    /// RPython pyjitpl.py:3188-3190 records the tentative JUMP with
    /// `descr=ptoken` before compile_trace(). Plain loop recording keeps
    /// `descr=None` until optimization rewrites it.
    pub fn close_loop_with_descr(&mut self, jump_args: &[OpRef], descr: Option<DescrRef>) {
        self.ensure_byte_buffer();
        self.record_bytes(OpCode::Jump, jump_args, descr, None);
    }

    /// Finish the trace (non-looping): add a FINISH operation.
    /// `finish_args` are the values returned from the trace.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        self.ensure_byte_buffer();
        self.record_bytes(OpCode::Finish, finish_args, Some(descr), None);
    }

    /// Consume the recorder and return its parts: (inputargs, ops).
    ///
    /// history.py parity: the recording phase ends and the trace is handed
    /// to the optimizer as a `TreeLoop`. See `TraceCtx::into_tree_loop` for
    /// the snapshot-bearing path.
    /// Clone the materialized inputargs and ops without draining the recorder.
    /// `compile_retrace`'s `InvalidLoop` arm cuts the tentative JUMP off the
    /// live history (`compile_retrace` `history.cut`) and keeps tracing, so the
    /// optimizer has to see a copy.
    pub fn clone_materialized_parts(&mut self) -> (Vec<InputArgRc>, Vec<OpRc>) {
        self.materialize_into_ops();
        // `History.set_inputargs` already stores the hole-filtered list
        // (`initialize_state_from_guard_failure` drops dead failargs).
        (self.inputargs.clone(), self.ops.clone())
    }

    pub fn into_parts(mut self) -> (Vec<InputArgRc>, Vec<OpRc>) {
        // `opencoder.py Trace.get_iter`: compile walks the live byte
        // stream, not a `Vec<Op>` that an earlier `materialize_into_ops`
        // may have filled before `close_loop` recorded JUMP.
        self.materialize_into_ops();
        (self.inputargs, self.ops)
    }

    /// Materialize the history's live input box list without consuming it.
    ///
    /// `pyjitpl.py initialize_state_from_guard_failure` installs the
    /// hole-filtered list with `History.set_inputargs`; each surviving box
    /// retains its original position in the recorder's reserved coordinate
    /// space.  This is the pre-`into_parts` view used while closing a bridge.
    pub fn live_inputargs_cloned(&self) -> Vec<InputArgRc> {
        self.inputargs.clone()
    }

    /// Convenience: consume the recorder and produce a `TreeLoop`.
    ///
    /// Snapshots are NOT included — callers that need them should use
    /// `TraceCtx::into_tree_loop` instead.
    pub fn get_trace(self) -> crate::history::TreeLoop {
        // `self.ops` is already `Vec<OpRc>`; `from_oprc` preserves that
        // shared identity.
        let (inputargs, ops) = self.into_parts();
        crate::history::TreeLoop::from_oprc(inputargs, ops, Vec::new())
    }

    /// `TreeLoop` view of the recorded trace that leaves the recorder whole.
    ///
    /// `compile.py compile_loop` hands `metainterp.history.trace` to the
    /// optimizer and keeps the opencoder buffer alive for
    /// `ResumeDataLoopMemo.number`, which walks `trace.get_snapshot_iter`
    /// per surviving guard. `unroll.py optimize_preamble` then does
    /// `trace.get_iter()` (`ByteTraceIter`); re-walk when `slots` has
    /// grown since the last fill so JUMP from `close_loop` is included.
    pub fn to_tree_loop(&mut self) -> crate::history::TreeLoop {
        self.materialize_into_ops();
        crate::history::TreeLoop::from_oprc(
            self.live_inputargs_cloned(),
            self.ops.clone(),
            Vec::new(),
        )
    }

    /// opencoder.py `cut_point()` — the recorder's local slice of
    /// the 5-tuple. Byte mode reports `len(_snapshot_data)` /
    /// `len(_snapshot_array_data)` (`opencoder.py cut_point`). The
    /// `Vec<Op>` recorder has no snapshot bytes; `TraceCtx::get_trace_position`
    /// fills `snapshot_data_len` from `Vec<Snapshot>` in that mode.
    pub fn get_position(&self) -> TracePosition {
        if let Some(trb) = self.trb.as_ref() {
            return TracePosition {
                _pos: trb._pos,
                _count: self.op_count,
                _index: self.box_count,
                snapshot_data_len: trb._snapshot_data.len(),
                snapshot_array_data_len: trb._snapshot_array_data.len(),
            };
        }
        TracePosition {
            _pos: self.ops.len(),
            _count: self.op_count,
            _index: self.box_count,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
        }
    }

    /// opencoder.py `cut_at(end)` — restore the recorder to a
    /// previously saved position.
    ///
    /// Discards all operations recorded after `pos`. Used to undo a
    /// tentative JUMP after compile_trace succeeds or fails
    /// (pyjitpl.py finally: `self.history.cut(cut_at)`).
    pub fn cut(&mut self, pos: TracePosition) {
        if let Some(trb) = self.trb.as_mut() {
            let n = self.max_num_inputargs;
            trb.cut_at(crate::recorder::TracePosition {
                _pos: pos._pos,
                _count: n + (pos._count.saturating_sub(n)),
                _index: pos._index,
                snapshot_data_len: 0,
                snapshot_array_data_len: 0,
            });
            let nops = pos._count.saturating_sub(n) as usize;
            self.slots.truncate(nops);
            let nvalues = pos._index.saturating_sub(n) as usize;
            self.value_slots.truncate(nvalues);
            self.op_count = pos._count;
            self.box_count = pos._index;
            self.ops.clear();
            return;
        }
        self.ops.truncate(pos._pos);
        let n = self.max_num_inputargs;
        self.value_slots
            .truncate(pos._index.saturating_sub(n) as usize);
        self.op_count = pos._count;
        self.box_count = pos._index;
    }

    /// history.py `length`: number of non-inputarg ops recorded so far.
    /// Compared against `warmstate.trace_limit` by
    /// `MetaInterp.blackhole_if_trace_too_long` (pyjitpl.py).
    pub fn num_ops(&self) -> usize {
        if self.byte_mode() {
            return self.slots.len();
        }
        self.ops.len()
    }

    /// `opencoder.py Trace.max_num_inputargs` — reserved TAGBOX prefix.
    /// Live `inputargs` may be shorter when a bridge drops dead fail args.
    pub fn num_inputargs(&self) -> usize {
        self.max_num_inputargs as usize
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|arg| arg.tp.get()).collect()
    }

    /// Number of guards recorded so far.
    ///
    /// `opencoder.py Trace` keeps no guard counter; scan recorded opcodes
    /// (`slots` in byte mode, `ops` otherwise) with `OpCode::is_guard`.
    pub fn num_guards(&self) -> usize {
        if self.byte_mode() {
            self.slots.iter().filter(|s| s.opcode.is_guard()).count()
        } else {
            self.ops.iter().filter(|op| op.opcode.is_guard()).count()
        }
    }

    /// Access the recorded operations.
    pub fn ops(&self) -> &[OpRc] {
        &self.ops
    }

    /// Opcode at recorded-op index `i` (`slots` in byte mode, `ops` otherwise).
    pub fn opcode_at(&self, i: usize) -> Option<OpCode> {
        if let Some(slot) = self.slots.get(i) {
            return Some(slot.opcode);
        }
        self.ops.get(i).map(|op| op.opcode)
    }

    /// Opcode of the recorded op named by `opref`.
    ///
    /// Reads the op already stored at that position
    /// (`history.py AbstractResOp.getopnum`). Constants, input args,
    /// and positions that do not hold a recorded op yield `None`.
    pub fn opcode_of(&self, opref: OpRef) -> Option<OpCode> {
        if opref.is_constant() || opref.is_input_arg() || opref.is_none() {
            return None;
        }
        let n = self.box_prefix();
        if let OpRef::VoidOp(count) = opref {
            if let Some(s) = self.slot_by_seq(count.saturating_sub(n)) {
                return Some(s.opcode);
            }
            return self
                .ops
                .get(count.saturating_sub(n) as usize)
                .map(|op| op.opcode);
        }
        // Value FrontendOp is `value_slots[_index - _start]`
        // (`history.py` FrontendOp carries its opnum).
        if let Some(vs) = self.value_slot(opref.raw()) {
            return Some(vs.opcode);
        }
        self.get_op_by_raw_pos(opref.raw()).map(|op| op.opcode)
    }

    /// Visit each `ConstPtr` once and re-key `_refs_dict` after a moving
    /// collection. TAGCONSTPTR interning is `TraceRecordBuffer._refs_dict`
    /// (`opencoder.py` `_cached_const_ptr`). Materialized `ops` are
    /// dropped so the next `get_iter` rebuilds them from the forwarded
    /// pool.
    pub(crate) fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        if let Some(trb) = self.trb.as_mut() {
            trb.refresh_from_gc();
            for r in trb._refs.iter_mut().skip(1) {
                let mut gcref = GcRef(*r as usize);
                visitor(&mut gcref);
                *r = gcref.0 as u64;
            }
            if !trb._refs_dict.is_empty() {
                trb._refs_dict.clear();
                for (index, &reference) in trb._refs.iter().enumerate().skip(1) {
                    trb._refs_dict.insert(reference, index as u32);
                }
            }
            trb.walk_descr_const_ptr_refs(visitor);
        }
        if self.trb.is_some() {
            self.ops.clear();
        }
        for slot in &mut self.slots {
            if let Some(qd) = slot.descr.as_ref().and_then(|d| d.as_quasi_immut_descr()) {
                qd.walk_const_ptr_refs(visitor);
            }
        }
        for vs in &self.value_slots {
            if let Some(Value::Ref(mut gcref)) = vs.concrete.get() {
                visitor(&mut gcref);
                vs.concrete.set(Some(Value::Ref(gcref)));
            }
        }
        for rec in &mut self.inputarg_heapc {
            rec.walk_const_ptr_refs(visitor);
        }
        for vs in &mut self.value_slots {
            vs.heapc.walk_const_ptr_refs(visitor);
        }
        if self.trb.is_none() {
            for op in &self.ops {
                for arg in op.args_slice().iter() {
                    arg.walk_const_ptr_refs(visitor);
                }
                if let Some(fail_args) = op.guard_fail_args() {
                    for arg in fail_args.iter() {
                        arg.walk_const_ptr_refs(visitor);
                    }
                }
                if let Some(Value::Ref(mut gcref)) = op.get_value() {
                    visitor(&mut gcref);
                    op.set_value(Value::Ref(gcref));
                }
                if let Some(descr) = op.getdescr() {
                    if let Some(qd) = descr.as_quasi_immut_descr() {
                        qd.walk_const_ptr_refs(visitor);
                    }
                }
            }
        }
    }

    /// Test-only direct append. Production callers go through
    /// `record_op` so `op_count` / `box_count` stay in sync; this helper
    /// is for GC walker unit tests that exercise the op-graph storage
    /// without driving the full record path.
    #[cfg(test)]
    pub fn push_op_for_test(&mut self, op: Op) {
        self.ops.push(OpRc::new(op));
    }

    /// Access the recorded live input arguments (`History.set_inputargs`).
    pub fn inputargs(&self) -> &[InputArgRc] {
        &self.inputargs
    }

    /// Lookup by `InputArg.get_position()`, not by compact list index.
    fn inputarg_at(&self, position: u32) -> Option<&InputArgRc> {
        inputarg_at_position(&self.inputargs, position)
    }

    /// Get an operation by its OpRef position.
    pub fn get_op_by_pos(&self, pos: OpRef) -> Option<&Op> {
        self.ops
            .iter()
            .find(|op| op.pos().get() == pos)
            .map(|op| &**op)
    }

    /// Byte-mode stand-in for `get_op_by_raw_pos` on a `GetfieldGcR`.
    /// Args and descr are decoded from the byte stream (`FrontendOp`
    /// does not store them). Used by `recover_ref_value` while the
    /// 240-byte `Op` does not yet exist.
    pub(crate) fn getfield_gc_r_at(&self, raw: u32) -> Option<(DescrRef, OpRef)> {
        let trb = self.trb.as_ref()?;
        let mut found = None;
        trb.for_each_encoded_op(|op| {
            if op.box_index > raw {
                return false;
            }
            if op.opcode == OpCode::GetfieldGcR
                && op.opcode.result_type() != Type::Void
                && op.box_index == raw
                && op.arity >= 1
            {
                found = Some((op.args_pos, op.descr_index));
                return false;
            }
            true
        });
        let (args_pos, descr_index) = found?;
        let descr = trb.resolve_descr_index(descr_index)?;
        let tagged = crate::opencoder::TraceRecordBuffer::encoded_arg_at(&trb._ops, args_pos, 0);
        let obj = match self.untag_snapshot(tagged) {
            SnapshotTagged::Box(r, _) => r,
            SnapshotTagged::Const(v, Type::Ref) => OpRef::const_ptr(GcRef(v as usize)),
            SnapshotTagged::Const(v, Type::Int) => OpRef::const_int(v),
            SnapshotTagged::Const(v, Type::Float) => OpRef::const_float(f64::from_bits(v as u64)),
            SnapshotTagged::Const(_, Type::Void) => return None,
        };
        Some((descr, obj))
    }

    /// Fill `self.ops` from the byte buffer so `&[Op]` readers
    /// (`get_op_by_raw_pos`, tests after `into_recorder`) see the
    /// materialized trace. Re-walks when `slots` has grown since the last
    /// fill (`opencoder.py Trace.get_iter`).
    pub fn materialize_into_ops(&mut self) {
        if self.trb.is_some() && self.ops.len() != self.slots.len() {
            self.ops = self.materialize_ops();
        }
    }

    /// Get an operation by its raw u32 position, ignoring the variant
    /// tag. Use this when iterating a numeric position range without
    /// knowing each op's RPython `box.type` upfront — the typed lookup
    /// `get_op_by_pos` requires the variant to match (variant-aware Eq).
    pub fn get_op_by_raw_pos(&self, raw: u32) -> Option<&Op> {
        self.ops
            .iter()
            .find(|op| op.pos().get().raw() == raw && op.opcode.result_type() != Type::Void)
            .map(|op| &**op)
    }

    /// Stamp the concrete runtime value on the canonical frontend object
    /// for `position` (`history.py *FrontendOp.setint` / `_make_op`).
    /// `position` is the `_index` TAGBOX coordinate: live inputargs keep
    /// their original `get_position()`, later indices are value-producing
    /// ops. Void ops have no value slot. Returns `false` if `position` is
    /// past the recorded range or a dead failarg hole.
    pub(crate) fn set_concrete_at(&self, position: u32, value: Value) -> bool {
        if let Some(ia) = self.inputarg_at(position) {
            ia.set_value(value);
            true
        } else if position < self.max_num_inputargs {
            false
        } else if let Some(vs) = self.value_slot(position) {
            vs.concrete.set(Some(value));
            if let Some(op) = self
                .ops
                .iter()
                .find(|op| op.pos().get().raw() == position && op.result_type() != Type::Void)
            {
                op.set_value(value);
            }
            true
        } else if let Some(op) = self
            .ops
            .iter()
            .find(|op| op.pos().get().raw() == position && op.result_type() != Type::Void)
        {
            op.set_value(value);
            true
        } else {
            false
        }
    }

    /// Read the concrete runtime value stamped on the canonical frontend
    /// object for `position` (`history.py *FrontendOp.getint()`).
    /// `position` is `_index`. `None` when never stamped, a dead failarg
    /// hole, or out of range.
    pub(crate) fn concrete_at(&self, position: u32) -> Option<Value> {
        if let Some(ia) = self.inputarg_at(position) {
            ia.get_value()
        } else if position < self.max_num_inputargs {
            None
        } else if let Some(vs) = self.value_slot(position) {
            vs.concrete.get()
        } else {
            self.ops
                .iter()
                .find(|op| op.pos().get().raw() == position && op.result_type() != Type::Void)
                .and_then(|op| op.get_value())
        }
    }
}

impl majit_trace::heapcache::HeapcBoxes for Trace {
    fn heapc(&self, opref: OpRef) -> Option<&majit_trace::heapcache::HeapcRecord> {
        if opref.is_constant() {
            return None;
        }
        let pos = opref.raw() as usize;
        if pos < self.inputarg_heapc.len() {
            return Some(&self.inputarg_heapc[pos]);
        }
        self.value_slot(opref.raw()).map(|vs| &vs.heapc)
    }

    fn heapc_mut(&mut self, opref: OpRef) -> Option<&mut majit_trace::heapcache::HeapcRecord> {
        if opref.is_constant() {
            return None;
        }
        let pos = opref.raw() as usize;
        if pos < self.inputarg_heapc.len() {
            return Some(&mut self.inputarg_heapc[pos]);
        }
        self.value_slot_mut(opref.raw()).map(|vs| &mut vs.heapc)
    }

    fn box_value(&self, opref: OpRef) -> Option<Value> {
        if opref.is_constant() {
            return opref.inline_const_to_value();
        }
        self.concrete_at(opref.raw())
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Descr, DescrRef, FailDescr, Type};
    use std::sync::Arc;

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn farg(pos: u32) -> OpRef {
        OpRef::input_arg_float(pos)
    }

    fn rarg(pos: u32) -> OpRef {
        OpRef::input_arg_ref(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    /// A minimal FailDescr implementation for testing.
    #[derive(Debug)]
    struct TestFailDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestFailDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
            Some(self)
        }
        fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
            Some(self)
        }
    }

    impl FailDescr for TestFailDescr {
        fn fail_index(&self) -> u32 {
            self.index
        }
        fn fail_arg_types(&self) -> &[Type] {
            &[]
        }
    }

    fn make_fail_descr(index: u32) -> DescrRef {
        Arc::new(TestFailDescr { index })
    }

    /// `cut(pos)` truncates the recorded ops back to the saved position.
    #[test]
    fn cut_truncates_recorded_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let saved = rec.get_position();
        let _add1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let _add2 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 2);
        rec.cut(saved);
        assert_eq!(rec.num_ops(), 0);
        assert_eq!(rec.num_inputargs(), 1);
    }

    /// A flag set on a box, then `cut` before it, then a new box at the
    /// same `_index`: the new FrontendOp record has no flags.
    #[test]
    fn cut_drops_heapc_flags_on_reused_index() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Ref);
        let saved = rec.get_position();
        let first = rec.record_op(OpCode::New, &[]);
        let mut cache = majit_trace::heapcache::HeapCache::new();
        {
            let mut view = majit_trace::heapcache::HeapCacheViewMut::new(&mut cache, &mut rec);
            view.new_object(first);
            assert!(view.is_unescaped(first));
        }
        rec.cut(saved);
        let second = rec.record_op(OpCode::New, &[]);
        assert_eq!(first.raw(), second.raw());
        let view = majit_trace::heapcache::HeapCacheView::new(&cache, &rec);
        assert!(!view.is_unescaped(second));
        assert!(!view.is_class_known(second));
        assert!(!view.saw_allocation(second));
    }

    /// An inputarg box gets and keeps a known-class flag (`warmstate.py`
    /// `wrap` builds `RefFrontendOp` for red arguments).
    #[test]
    fn inputarg_keeps_known_class_flag() {
        let mut rec = Trace::new();
        let ia = rec.record_input_arg(Type::Ref);
        let mut cache = majit_trace::heapcache::HeapCache::new();
        {
            let mut view = majit_trace::heapcache::HeapCacheViewMut::new(&mut cache, &mut rec);
            view.class_now_known(ia);
            assert!(view.is_class_known(ia));
        }
        let _ = rec.record_op(OpCode::New, &[]);
        let view = majit_trace::heapcache::HeapCacheView::new(&cache, &rec);
        assert!(view.is_class_known(ia));
    }

    #[test]
    fn box_for_operand_is_deterministic_per_position() {
        // The canonical Operand bridge must be stable per recorded position: the
        // same OpRef always binds to the same producer Rc, so a consumer can key
        // by box identity (Operand::eq = Rc::ptr_eq) instead of flat OpRef and
        // still get the "same value -> same key" behaviour, matching the
        // box-object-keyed dicts in heapcache.py / registers_r in pyjitpl.py.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.materialize_into_ops();

        // ResOp position: two calls resolve to the SAME producer box.
        let a = rec.box_for_operand(i1);
        let b = rec.box_for_operand(i1);
        assert_eq!(a, b, "same position must yield ptr_eq-equal Operands");
        assert_eq!(a.to_opref(), i1, "Operand round-trips to its OpRef");

        // InputArg position: same invariant.
        let ia_a = rec.box_for_operand(i0);
        let ia_b = rec.box_for_operand(i0);
        assert_eq!(ia_a, ia_b);
        assert_eq!(ia_a.to_opref(), i0);

        // Distinct positions are distinct boxes.
        assert_ne!(rec.box_for_operand(i0), rec.box_for_operand(i1));
    }

    /// `opencoder.py Trace._cached_const_ptr`: repeated non-null pointers
    /// intern once in `_refs_dict`, and a moving collection re-keys it.
    #[test]
    fn const_ptr_operands_share_the_trace_ref_pool_and_rekey_after_gc() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Ref);
        let old = OpRef::const_ptr(GcRef(0x1000));
        rec.record_op(OpCode::SameAsR, &[old]);
        rec.record_op(OpCode::SameAsR, &[old]);
        {
            let trb = rec.trb.as_ref().expect("byte buffer");
            assert_eq!(trb._refs_dict.len(), 1, "same address interned once");
            assert_eq!(trb._refs.iter().filter(|&&a| a == 0x1000).count(), 1);
        }

        rec.walk_const_ptr_refs(&mut |gcref| gcref.0 += 0x1000);
        let trb = rec.trb.as_ref().expect("byte buffer");
        assert!(trb._refs.iter().any(|&a| a == 0x2000));
        assert!(!trb._refs.iter().any(|&a| a == 0x1000));
        assert_eq!(trb._refs_dict.get(&0x2000).copied(), Some(1));
    }

    #[test]
    fn ref_pool_gc_visits_each_box_once_and_rekeys_overlapping_addresses() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Ref);
        let constants: Vec<_> = (1..=4)
            .map(|i| OpRef::const_ptr(GcRef(i * 0x1000)))
            .collect();
        for &constant in &constants {
            rec.record_op(OpCode::SameAsR, &[constant]);
        }
        rec.record_guard(OpCode::GuardTrue, &[OpRef::const_int(1)], None);
        let mut visited = Vec::new();
        rec.walk_const_ptr_refs(&mut |reference| {
            visited.push(reference.0);
            reference.0 += 0x1000;
        });
        visited.sort_unstable();
        assert_eq!(visited, vec![0x1000, 0x2000, 0x3000, 0x4000]);
        let trb = rec.trb.as_ref().expect("byte buffer");
        assert_eq!(trb._refs_dict.len(), 4);
        for i in 1..=4 {
            let address = ((i + 1) * 0x1000) as u64;
            assert!(
                trb._refs.iter().any(|&a| a == address),
                "forwarded {address:#x} missing from _refs"
            );
        }
    }

    #[test]
    fn test_record_simple_loop() {
        // Trace: i0 -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(i1, iop(1));

        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_inputargs(), 1);
        assert_eq!(trace.num_ops(), 2); // IntAdd + Jump
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::Jump);
        assert_eq!(trace.ops[1].arg(0).to_opref(), i1);
    }

    #[test]
    fn test_record_with_guard() {
        // i0 -> guard_true(i0) -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        let _g = rec.record_guard(OpCode::GuardTrue, &[i0], Some(descr));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_ops(), 3); // GuardTrue + IntAdd + Jump
        assert!(trace.ops[0].opcode.is_guard());
        assert!(trace.ops[0].has_descr());
    }

    #[test]
    fn test_record_finish() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(99);
        rec.finish(&[i1], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());
    }

    #[test]
    fn test_record_multiple_inputargs() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);
        assert_eq!(i0, iarg(0));
        assert_eq!(r0, rarg(1));
        assert_eq!(f0, farg(2));
        assert_eq!(rec.num_inputargs(), 3);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Ref);
        assert_eq!(trace.inputargs[2].tp.get(), Type::Float);
    }

    #[test]
    fn test_opref_assignment() {
        // Value-producing `OpRef.raw()` is opencoder `_index`. Void ops
        // are named by op-sequence position, not `_index`.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));
        assert_eq!(i1, iarg(1));

        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(i2, iop(2));

        let descr = make_fail_descr(0);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));
        assert_eq!(g0, vop(3));

        let i3 = rec.record_op(OpCode::IntSub, &[i2, i0]);
        assert_eq!(i3, iop(3));

        rec.close_loop(&[i3, i1]);
        let trace = rec.get_trace();

        // Value ops use `_index`; void ops use the all-ops `_count`.
        assert_eq!(trace.ops[0].pos().get(), iop(2)); // IntAdd
        assert_eq!(trace.ops[1].pos().get(), vop(3)); // GuardTrue
        assert_eq!(trace.ops[2].pos().get(), iop(3)); // IntSub
        assert_eq!(trace.ops[3].pos().get(), vop(5)); // Jump
    }

    #[test]
    fn byte_buffer_materialize_keeps_index_positions() {
        // history.py record + opencoder.py get_iter: bytes during
        // record, ResOp objects only at iterate. Value `OpRef.raw()` is
        // `_index` and survives materialize so snapshots still match.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], None);
        rec.set_last_op_resume_position(7);
        rec.close_loop(&[i2, i1]);
        assert_eq!(rec.num_ops(), 3);
        let (inputs, ops) = rec.into_parts();
        assert_eq!(inputs.len(), 2);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[0].pos().get(), i2);
        assert_eq!(ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(ops[1].pos().get(), g0);
        assert_eq!(ops[1].rd_resume_position(), 7);
        assert_eq!(ops[2].opcode, OpCode::Jump);
    }

    #[test]
    fn get_iter_for_optimizer_remints_live_inputargs_once() {
        // compile_bridge get_iter: hole-filtered live inputargs remint
        // densely. Resume comes from the slot overlay / descr varint;
        // fail_args are absent until `store_final_boxes_in_guard`.
        let mut rec =
            Trace::with_input_layout(&[Type::Int, Type::Ref, Type::Int], &[true, false, true]);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let add = rec.record_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(2)],
        );
        rec.record_guard(OpCode::GuardTrue, &[add], None);
        rec.set_last_op_resume_position(4);
        rec.close_loop(&[add]);
        let live = rec.live_inputargs_cloned();
        let (ops, reminted, cache) = rec
            .get_iter_for_optimizer(&live, 1000)
            .expect("byte buffer");
        assert_eq!(
            reminted
                .iter()
                .map(|arg| (arg.index, arg.tp.get()))
                .collect::<Vec<_>>(),
            vec![(1000, Type::Int), (1001, Type::Int)]
        );
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(ops[1].rd_resume_position(), 4);
        assert!(ops[1].guard_fail_args().is_none());
        assert!(
            cache[0]
                .as_ref()
                .is_some_and(|a| a.same_box(&ops[0].arg(0))),
            "unique_cache must reuse ByteTraceIter inputarg identity"
        );
        assert_eq!(
            cache[0].as_ref().map(|a| a.to_opref()),
            Some(OpRef::input_arg_int(1000))
        );
        assert_eq!(
            cache[2].as_ref().map(|a| a.to_opref()),
            Some(OpRef::input_arg_int(1001))
        );
    }

    #[test]
    fn byte_buffer_snapshot_roundtrip_keeps_boxes() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let _g = rec.record_guard(OpCode::GuardTrue, &[add], None);
        let snapshot = Snapshot {
            resume_position: -1,
            frames: vec![SnapshotFrame {
                jitcode_index: 3,
                pc: 11,
                boxes: vec![
                    SnapshotTagged::Box(i0, Type::Int),
                    SnapshotTagged::Box(add, Type::Int),
                    SnapshotTagged::Const(7, Type::Int),
                ],
            }],
            vable_boxes: vec![SnapshotTagged::Box(i1, Type::Int)],
            vref_boxes: Vec::new(),
        };
        let id = rec.encode_captured_snapshot(&snapshot);
        assert_eq!(id, 0);
        let decoded = rec.decode_captured_snapshots().expect("byte mode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].frames.len(), 1);
        assert_eq!(decoded[0].frames[0].jitcode_index, 3);
        assert_eq!(decoded[0].frames[0].pc, 11);
        assert_eq!(decoded[0].frames[0].boxes, snapshot.frames[0].boxes);
        assert_eq!(decoded[0].vable_boxes, snapshot.vable_boxes);
        assert!(decoded[0].vref_boxes.is_empty());
    }

    #[test]
    fn cut_then_record_guard_resumes_at_byte_offsets() {
        // Cross-step blocker 3: after resume positions are byte offsets,
        // a cut that rewinds ops must leave surviving guards pointing at
        // their `_snapshot_data` offsets, and a later guard must get a
        // new offset. `cut_at` does not rewind snapshot bytes.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_guard(OpCode::GuardTrue, &[i0], None);
        let frame = |pc: u32| SnapshotFrame {
            jitcode_index: 1,
            pc,
            boxes: vec![SnapshotTagged::Box(i0, Type::Int)],
        };
        let off0 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(11)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off0);
        let after_first = rec.get_position();
        let snap_bytes = after_first.snapshot_data_len;

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.record_guard(OpCode::GuardFalse, &[add], None);
        let off1 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(22)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off1);
        assert!(off1 > off0);
        let after_second_len = rec.get_position().snapshot_data_len;
        assert!(after_second_len > snap_bytes);

        rec.cut(after_first);
        // `cut_at` does not rewind snapshot bytes. Walking live guards
        // after the cut is what drops the discarded capture.
        assert_eq!(rec.get_position().snapshot_data_len, after_second_len);
        let after_cut_len = rec.get_position().snapshot_data_len;

        rec.record_guard(OpCode::GuardValue, &[i0, OpRef::const_int(0)], None);
        let off2 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(33)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off2);
        assert_ne!(off0, off2);
        assert!(off2 > off0);
        assert!(rec.get_position().snapshot_data_len > after_cut_len);

        let ops = rec.materialize_ops();
        let guards: Vec<_> = ops.iter().filter(|op| op.opcode.is_guard()).collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[0].rd_resume_position(), off0);
        assert_eq!(guards[1].opcode, OpCode::GuardValue);
        assert_eq!(guards[1].rd_resume_position(), off2);
    }

    #[test]
    fn guard_with_marker_descr_keeps_stream_placeholder() {
        // `generate_guard` records descr=None; a foriter marker lives on
        // the FrontendSlot. The stream descr slot stays the 0-placeholder
        // so `create_top_snapshot` can rewind it.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let marker = make_fail_descr(7);
        rec.record_guard(OpCode::GuardTrue, &[i0], Some(marker.clone()));
        let frame = SnapshotFrame {
            jitcode_index: 1,
            pc: 11,
            boxes: vec![SnapshotTagged::Box(i0, Type::Int)],
        };
        let off = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off);
        let ops = rec.materialize_ops();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].rd_resume_position(), off);
        assert!(ops[0].has_descr());
        assert!(std::sync::Arc::ptr_eq(&ops[0].getdescr().unwrap(), &marker));
    }

    #[test]
    fn recapture_of_last_guard_restamps_named_slot() {
        // First capture occupies `_snapshot_data`; a second capture while
        // the same guard is still last must not rewind a non-placeholder
        // descr slot (`create_top_snapshot` `_pos -= 2` only for `\x00\x00`).
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_guard(OpCode::GuardTrue, &[i0], None);
        let frame = |pc: u32| SnapshotFrame {
            jitcode_index: 1,
            pc,
            boxes: vec![SnapshotTagged::Box(i0, Type::Int)],
        };
        let off0 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(11)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off0);
        rec.record_guard(OpCode::GuardFalse, &[i0], None);
        let off1 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(22)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off1);
        assert!(off1 > off0);
        let off2 = rec.encode_captured_snapshot(&Snapshot {
            resume_position: -1,
            frames: vec![frame(33)],
            vable_boxes: vec![],
            vref_boxes: vec![],
        });
        rec.set_last_op_resume_position(off2);
        assert!(off2 > off1);
        let ops = rec.materialize_ops();
        let guards: Vec<_> = ops.iter().filter(|op| op.opcode.is_guard()).collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].rd_resume_position(), off0);
        assert_eq!(guards[1].rd_resume_position(), off2);

        let decoded = rec.decode_captured_snapshots().expect("byte mode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].frames[0].pc, 11);
        assert_eq!(decoded[1].frames[0].pc, 33);
        assert_eq!(
            rec.captured_resume_positions(),
            vec![off0 as usize, off2 as usize]
        );
    }

    #[test]
    fn byte_buffer_materialize_jump_over_extra_inputargs() {
        // pyjitpl.py `initialize_virtualizable` appends vable boxes onto
        // `original_boxes` before `create_empty_history`. The byte buffer
        // must be sized to that full cap so JUMP TAGBOX args past the
        // portal reds resolve.
        let mut rec = Trace::new();
        let portal = rec.record_input_arg(Type::Int);
        let extras: Vec<OpRef> = (0..8).map(|_| rec.record_input_arg(Type::Int)).collect();
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let mut jump_args = vec![portal];
        jump_args.extend(extras.iter().copied());
        rec.close_loop(&jump_args);
        let (inputs, ops) = rec.into_parts();
        assert_eq!(inputs.len(), 9);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::Jump);
        assert_eq!(ops[0].num_args(), 9);
    }

    #[test]
    #[should_panic(expected = "record_input_arg after attach_byte_buffer")]
    fn record_input_arg_after_attach_is_rejected() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_input_arg(Type::Int);
    }

    #[test]
    fn tree_loop_cut_uses_op_count_not_byte_cursor() {
        let pos = TracePosition {
            _pos: 80,
            _count: 5,
            _index: 5,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
        };
        assert!(pos.has_prefix_ops(2));
        assert_eq!(pos.tree_loop_op_index(2), 3);
        assert!(
            !TracePosition {
                _pos: 2,
                _count: 2,
                _index: 2,
                snapshot_data_len: 0,
                snapshot_array_data_len: 0,
            }
            .has_prefix_ops(2)
        );
    }

    #[test]
    fn into_parts_reuses_materialized_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.materialize_into_ops();
        let first = rec.ops()[0].clone();
        let (_, ops) = rec.into_parts();
        assert!(OpRc::ptr_eq(&first, &ops[0]));
    }

    #[test]
    fn to_tree_loop_rewalks_byte_stream_after_close_loop() {
        // `unroll.py optimize_preamble` `trace.get_iter()` walks the live
        // buffer. An earlier `materialize_into_ops` (`set_op_fail_args`)
        // must not hide JUMP recorded by `close_loop`.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_guard(OpCode::GuardTrue, &[i0], None);
        rec.materialize_into_ops();
        assert_eq!(rec.ops().len(), 1);
        rec.close_loop(&[i0]);
        let loop_ = rec.to_tree_loop();
        assert_eq!(loop_.ops.last().map(|op| op.opcode), Some(OpCode::Jump));
        assert_eq!(loop_.ops.len(), 2);
    }

    #[test]
    fn byte_mode_walk_forwards_trb_refs() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Ref);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let cptr = OpRef::const_ptr(GcRef(0x1000));
        rec.record_op(OpCode::GetfieldGcR, &[cptr]);
        rec.record_guard(OpCode::GuardTrue, &[i0], None);
        rec.walk_const_ptr_refs(&mut |gcref| gcref.0 += 0x1000);
        rec.materialize_into_ops();
        assert_eq!(
            rec.ops()[0].arg(0).to_opref(),
            OpRef::const_ptr(GcRef(0x2000))
        );
        assert!(rec.ops()[1].guard_fail_args().is_none());
    }

    #[test]
    fn walk_forwards_quasiimmut_constantfieldbox() {
        #[derive(Debug)]
        struct Handle;
        impl majit_ir::QuasiImmutHandle for Handle {
            fn is_current(&self) -> bool {
                true
            }
            fn register_loop_token(
                &self,
                _token: &std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken>,
            ) {
            }
            fn instance_identity(&self) -> usize {
                1
            }
        }
        let field = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(0, 8, 8, Type::Ref, false))
            as DescrRef;
        let descr = std::sync::Arc::new(majit_ir::QuasiImmutDescr::new(
            field,
            0,
            std::sync::Arc::new(Handle),
            Some(Value::Ref(GcRef(0x1000))),
        )) as DescrRef;
        let mut rec = Trace::new();
        let obj = rec.record_input_arg(Type::Ref);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
        rec.walk_const_ptr_refs(&mut |gcref| gcref.0 += 0x1000);
        assert_eq!(
            descr.as_quasi_immut_descr().unwrap().constantfieldbox(),
            Some(Value::Ref(GcRef(0x2000)))
        );
    }

    #[test]
    fn quasiimmut_field_materialize_keeps_the_wrapper() {
        // `opencoder.py _encode_descr` sees `AbstractDescr.get_descr_index
        // == -1` and appends the wrapper to `Trace._descrs`. Forwarding
        // the field's `setup_descrs` slot would restore the FieldDescr
        // and drop the qmut (`heap.py` `isinstance(qmutdescr,
        // QuasiImmutDescr)`).
        #[derive(Debug)]
        struct Handle;
        impl majit_ir::QuasiImmutHandle for Handle {
            fn is_current(&self) -> bool {
                true
            }
            fn register_loop_token(
                &self,
                _token: &std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken>,
            ) {
            }
            fn instance_identity(&self) -> usize {
                1
            }
        }
        let field = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(0, 8, 8, Type::Ref, false));
        field.set_descr_index(0);
        let qmut = std::sync::Arc::new(majit_ir::QuasiImmutDescr::new(
            field as DescrRef,
            0x10,
            std::sync::Arc::new(Handle),
            Some(Value::Ref(GcRef(0x1000))),
        )) as DescrRef;
        let mut rec = Trace::new();
        let obj = rec.record_input_arg(Type::Ref);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_op_with_descr(OpCode::QuasiimmutField, &[obj], qmut.clone());
        let ops = rec.materialize_ops();
        assert_eq!(ops.len(), 1);
        let restored = ops[0].getdescr().expect("stream descr");
        assert!(restored.as_quasi_immut_descr().is_some());
        assert!(std::sync::Arc::ptr_eq(&restored, &qmut));
    }

    #[test]
    fn byte_mode_opcode_of_and_getfield_skip_voids() {
        // Value OpRef.raw() is `_index`; voids are `_count`. A DebugMergePoint
        // between two value ops must not hide the GetfieldGcR from
        // `opcode_of` / `getfield_gc_r_at` (byte-mode `ops` is empty).
        let mut rec = Trace::new();
        let obj = rec.record_input_arg(Type::Ref);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let dmp = rec.record_op(OpCode::DebugMergePoint, &[]);
        assert!(matches!(dmp, OpRef::VoidOp(_)));
        let field_sd =
            std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(0, 8, 8, Type::Ref, false));
        let field = field_sd as DescrRef;
        let loaded = rec.record_op_with_descr(OpCode::GetfieldGcR, &[obj], field.clone());
        rec.set_concrete_at(loaded.raw(), Value::Ref(GcRef(0x20)));
        assert_eq!(rec.opcode_of(dmp), Some(OpCode::DebugMergePoint));
        assert_eq!(rec.opcode_of(loaded), Some(OpCode::GetfieldGcR));
        assert_eq!(rec.concrete_at(loaded.raw()), Some(Value::Ref(GcRef(0x20))));
        let (descr, base) = rec
            .getfield_gc_r_at(loaded.raw())
            .expect("GetfieldGcR args from the stream");
        assert!(std::sync::Arc::ptr_eq(&descr, &field));
        assert_eq!(base, obj);
        assert!(rec.ops().is_empty());
    }

    #[test]
    fn byte_mode_snapshot_after_materialize_does_not_patch_intadd_zero() {
        // `clone_materialized_parts` fills `ops` and recording continues
        // in `slots`. Deciding "last is guard" from the stale `ops` copy
        // plus `last_descr_slot_is_placeholder` (any trailing `00 00`,
        // including TAGINT 0) overwrote `IntAdd(x, ConstInt(0))`.
        let mut rec = Trace::new();
        let x = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        rec.record_guard(OpCode::GuardTrue, &[x], None);
        rec.materialize_into_ops();
        rec.record_op(OpCode::IntAdd, &[x, OpRef::const_int(0)]);
        let snap = Snapshot {
            resume_position: -1,
            frames: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 0,
                boxes: vec![],
            }],
            vable_boxes: Vec::new(),
            vref_boxes: Vec::new(),
        };
        rec.encode_captured_snapshot(&snap);
        let ops = rec.materialize_ops();
        let add = ops
            .iter()
            .find(|op| op.opcode == OpCode::IntAdd)
            .expect("IntAdd recorded after the guard");
        assert_eq!(
            add.arg(1).to_opref(),
            OpRef::const_int(0),
            "IntAdd ConstInt(0) argument must survive snapshot capture"
        );
    }

    #[test]
    #[should_panic(expected = "use record_guard")]
    fn test_record_op_with_guard_opcode() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        // Should panic: guard opcodes must use record_guard
        rec.record_op(OpCode::GuardTrue, &[iarg(0)]);
    }

    #[test]
    fn test_num_ops_counts_non_inputargs() {
        // history.py length() = trace._count - len(inputargs).
        // In pyre that's ops.len() since inputargs aren't stored in ops.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);
    }

    #[test]
    fn test_record_op_with_descr() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(42);
        let i1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr);
        assert_eq!(i1, iop(1));

        rec.materialize_into_ops();
        assert!(rec.ops()[0].has_descr());
    }

    #[test]
    fn test_complex_trace() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let i2 = rec.record_op(OpCode::IntLt, &[i0, i1]);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));

        let i3 = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        rec.close_loop(&[i3, i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_ops(), 4);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
    }

    // Opencoder parity tests
    // Ported from rpython/jit/metainterp/test/test_opencoder.py

    #[test]
    fn test_simple_iterator() {
        // Parity: test_simple_iterator
        // Record two INT_ADD ops and verify trace structure matches.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add1 = rec.record_op(OpCode::IntAdd, &[add0, i0]);

        rec.close_loop(&[add1, i1]);
        let trace = rec.get_trace();

        // Verify the trace has the correct number of ops (2 + Jump).
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);

        // First add uses input args i0, i1.
        assert_eq!(trace.ops[0].arg(0).to_opref(), i0);
        assert_eq!(trace.ops[0].arg(1).to_opref(), i1);

        // Second add references the result of first add and i0.
        assert_eq!(trace.ops[1].arg(0).to_opref(), add0);
        assert_eq!(trace.ops[1].arg(1).to_opref(), i0);
    }

    #[test]
    fn test_inputargs_preserved() {
        // Parity: Trace([i0, i1], ...) preserves input args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_inputargs(), 2);

        rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add = rec.record_op(OpCode::IntAdd, &[iop(2), i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Int);
    }

    #[test]
    fn test_op_references_chain() {
        // Parity: ops that reference previous ops form correct chains.
        // i0 -> add = int_add(i0, i0) -> sub = int_sub(add, i0) -> jump(sub)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[0].pos().get(), iop(1)); // after 1 inputarg
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[1].arg(0).to_opref(), add); // references the add result
        assert_eq!(trace.ops[1].arg(1).to_opref(), i0); // references the input arg
    }

    #[test]
    fn test_guard_with_fail_args() {
        // Parity: guards can carry fail_args describing live values at guard.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let descr = make_fail_descr(0);
        let guard = rec.record_guard(OpCode::GuardTrue, &[add], Some(descr));

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);
        rec.materialize_into_ops();
        rec.set_op_fail_args(guard, &[i0, i1, add]);

        let trace = rec.get_trace();
        // Find the guard op.
        let guard_op = &trace.ops[1]; // after IntAdd
        assert_eq!(guard_op.pos().get(), guard);
        assert!(guard_op.opcode.is_guard());
        let fail_args = guard_op.guard_fail_args().unwrap();
        assert_eq!(fail_args.len(), 3);
        assert_eq!(fail_args[0].to_opref(), i0);
        assert_eq!(fail_args[1].to_opref(), i1);
        assert_eq!(fail_args[2].to_opref(), add);
    }

    #[test]
    fn test_multiple_guards() {
        // Parity: multiple guards in one trace.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr0 = make_fail_descr(0);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i0], Some(descr0));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        let descr1 = make_fail_descr(1);
        let g1 = rec.record_guard(OpCode::GuardFalse, &[add], Some(descr1));

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);
        rec.materialize_into_ops();
        rec.set_op_fail_args(g0, &[i0, i1]);
        rec.set_op_fail_args(g1, &[i0, add]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);

        // First guard's fail_args
        let fa0 = guards[0].guard_fail_args().unwrap();
        assert_eq!(fa0.len(), 2);
        assert_eq!(fa0[0].to_opref(), i0);
        assert_eq!(fa0[1].to_opref(), i1);

        // Second guard's fail_args
        let fa1 = guards[1].guard_fail_args().unwrap();
        assert_eq!(fa1.len(), 2);
        assert_eq!(fa1[0].to_opref(), i0);
        assert_eq!(fa1[1].to_opref(), add);
    }

    #[test]
    fn test_close_loop_jump_targets_inputargs() {
        // Parity: close_loop produces a JUMP whose args correspond to inputargs.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let jump = trace.ops.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(jump.num_args(), trace.num_inputargs());
        assert_eq!(jump.arg(0).to_opref(), add);
        assert_eq!(jump.arg(1).to_opref(), i1);
    }

    #[test]
    fn test_finish_produces_finish_op() {
        // Parity: finish() produces a FINISH op with the given args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(42);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());

        let finish_op = trace.ops.last().unwrap();
        assert_eq!(finish_op.opcode, OpCode::Finish);
        assert_eq!(finish_op.num_args(), 1);
        assert_eq!(finish_op.arg(0).to_opref(), add);
        assert!(finish_op.has_descr());
    }

    #[test]
    fn test_trace_length_tracking() {
        // Parity: num_ops() tracks the count accurately.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);

        rec.record_op(OpCode::IntSub, &[iop(1), i0]);
        assert_eq!(rec.num_ops(), 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[iop(2)], Some(descr));
        assert_eq!(rec.num_ops(), 3);

        rec.close_loop(&[iop(2)]);
        // After close_loop, Jump is added.
        assert_eq!(rec.num_ops(), 4);
    }

    #[test]
    fn test_trace_position_splits_count_and_index_for_void_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let before_guard = rec.get_position();
        assert_eq!(before_guard._count, 2);
        assert_eq!(before_guard._index, 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i1], Some(descr));
        rec.close_loop(&[i1]);
        rec.finish(&[i1], make_fail_descr(1));

        let pos = rec.get_position();
        assert_eq!(pos._count, 5);
        assert_eq!(pos._index, 2);
    }

    #[test]
    fn test_with_num_inputs_creates_inputargs() {
        // Parity: Trace::with_num_inputs pre-creates input args.
        let rec = Trace::with_num_inputs(3);
        assert_eq!(rec.num_inputargs(), 3);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn guard_failure_history_drops_holes_but_keeps_frontend_positions() {
        let mut rec =
            Trace::with_input_layout(&[Type::Int, Type::Ref, Type::Int], &[true, false, true]);
        let result = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(2)]);
        assert_eq!(
            result.raw(),
            3,
            "the hole remains reserved in trace positions"
        );

        let live = rec.live_inputargs_cloned();
        assert_eq!(
            live.iter().map(|arg| arg.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_int(0), OpRef::input_arg_int(2)]
        );

        let (inputargs, ops) = rec.into_parts();
        assert_eq!(inputargs.len(), 2);
        assert_eq!(inputargs[0].opref(), iarg(0));
        assert_eq!(inputargs[1].opref(), iarg(2));
        assert_eq!(ops[0].arg(1).to_opref(), iarg(2));
    }

    #[test]
    fn clone_materialized_parts_drops_dead_failarg_holes() {
        // Guard-failure retrace: `History.set_inputargs` stores only live
        // InputArgs; the reserved hole stays in `max_num_inputargs`.
        // The snapshot clone is what `compile_retrace` optimizes, so it
        // has to hand over that same live list.
        let mut rec =
            Trace::with_input_layout(&[Type::Int, Type::Ref, Type::Int], &[true, false, true]);
        let result = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(2)]);
        rec.close_loop(&[result]);
        let (inputargs, ops) = rec.clone_materialized_parts();
        assert_eq!(
            inputargs.iter().map(|arg| arg.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_int(0), OpRef::input_arg_int(2)]
        );
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].arg(1).to_opref(), iarg(2));
        assert_eq!(
            rec.num_inputargs(),
            3,
            "the live recorder keeps the reserved hole"
        );
    }

    #[test]
    fn test_with_num_inputs_oprefs() {
        // Input args from with_num_inputs get InputArgInt(0), InputArgInt(1), ...
        let mut rec = Trace::with_num_inputs(3);

        // The input args consumed positions 0..2, so next op gets BoxInt(3).
        let add = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        assert_eq!(add, iop(3));

        rec.close_loop(&[iarg(0), iarg(1), iarg(2)]);
        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.ops[0].pos().get(), iop(3));
    }

    #[test]
    fn test_guard_descr_preserved() {
        // Parity: guard descriptors are preserved through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(77);
        rec.record_guard(OpCode::GuardNoException, &[], Some(descr));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        assert!(guard.has_descr());
        let d = guard.getdescr().unwrap();
        assert_eq!(d.index(), 77);
    }

    #[test]
    fn test_op_with_descr_preserved() {
        // Parity: op descriptors (e.g., for calls) are preserved.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let call_descr = make_fail_descr(55);
        let result = rec.record_op_with_descr(OpCode::CallI, &[i0], call_descr);

        rec.close_loop(&[result]);
        let trace = rec.get_trace();
        let call_op = &trace.ops[0];
        assert_eq!(call_op.opcode, OpCode::CallI);
        assert!(call_op.has_descr());
        assert_eq!(call_op.getdescr().unwrap().index(), 55);
    }

    #[test]
    fn test_empty_fail_args() {
        // Parity: guard with empty fail_args is valid.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i0], Some(descr));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        assert!(
            guard
                .guard_fail_args()
                .map(|a| a.is_empty())
                .unwrap_or(true)
        );
    }

    #[test]
    #[should_panic(expected = "opcode")]
    fn test_record_guard_with_non_guard_opcode() {
        // Parity: record_guard rejects non-guard opcodes.
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::IntAdd, &[iarg(0)], Some(descr));
    }

    #[test]
    #[should_panic(expected = "input args must be registered before any operations")]
    fn test_inputarg_after_ops() {
        // Parity: input args must come before any operations.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        // This should panic.
        rec.record_input_arg(Type::Int);
    }

    // Opencoder breadth tests — deeper parity with test_opencoder.py

    #[test]
    fn test_recorder_const_int_via_constant_oprefs() {
        // Constants live in a dedicated pool and keep stable OpRefs.
        // Recording ops that reference constants should preserve them.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        // Simulate a pooled constant reference.
        let const_ref = OpRef::const_int(0);
        let add = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);

        rec.close_loop(&[add]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].arg(1).to_opref(), const_ref);
        assert!(trace.ops[0].arg(1).is_constant());
    }

    #[test]
    fn test_recorder_const_deduplication_by_opref() {
        // Two ops referencing the same constant OpRef share the same value.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let const_ref = OpRef::const_int(1);
        let add1 = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);
        let add2 = rec.record_op(OpCode::IntAdd, &[add1, const_ref]);

        rec.close_loop(&[add2]);
        let trace = rec.get_trace();

        // Both ops reference the same constant
        assert_eq!(
            trace.ops[0].arg(1).to_opref(),
            trace.ops[1].arg(1).to_opref()
        );
        assert_eq!(trace.ops[0].arg(1).to_opref(), const_ref);
    }

    #[test]
    fn test_recorder_descriptors_preserved_on_ops() {
        // Descriptors on non-guard ops should survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr1 = make_fail_descr(10);
        let descr2 = make_fail_descr(20);

        let call1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr1);
        let call2 = rec.record_op_with_descr(OpCode::CallI, &[i1], descr2);

        rec.close_loop(&[call1, call2]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].getdescr().unwrap().index(), 10);
        assert_eq!(trace.ops[1].getdescr().unwrap().index(), 20);
    }

    #[test]
    fn test_recorder_100_plus_ops_stress() {
        // Recording 200 ops should work without issue.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut prev = i0;
        for _ in 0..200 {
            prev = rec.record_op(OpCode::IntAdd, &[prev, i0]);
        }
        assert_eq!(rec.num_ops(), 200);

        rec.close_loop(&[prev]);
        let trace = rec.get_trace();
        // 200 IntAdd + 1 Jump
        assert_eq!(trace.num_ops(), 201);
        assert!(trace.is_loop());

        // Verify chain: each op references the previous op's result.
        // i0 = input_arg_int(0), first IntAdd = int_op(1), second = int_op(2), ...
        for (i, op) in trace.ops[..200].iter().enumerate() {
            assert_eq!(op.opcode, OpCode::IntAdd);
            if i > 0 {
                // The previous IntAdd produced BoxInt(i) (offset by inputarg).
                assert_eq!(op.arg(0).to_opref(), iop(i as u32));
            }
        }
    }

    #[test]
    fn test_recorder_mixed_type_input_args() {
        // Mixed-type inputs (Int, Float, Ref) produce correct types.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let f0 = rec.record_input_arg(Type::Float);
        let r0 = rec.record_input_arg(Type::Ref);
        let i1 = rec.record_input_arg(Type::Int);

        assert_eq!(i0, iarg(0));
        assert_eq!(f0, farg(1));
        assert_eq!(r0, rarg(2));
        assert_eq!(i1, iarg(3));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, f0, r0, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Float);
        assert_eq!(trace.inputargs[2].tp.get(), Type::Ref);
        assert_eq!(trace.inputargs[3].tp.get(), Type::Int);
    }

    #[test]
    fn test_recorder_guard_with_many_fail_args() {
        // Guard with 10+ fail_args.
        let mut rec = Trace::new();
        let mut inputs = Vec::new();
        for _ in 0..12 {
            inputs.push(rec.record_input_arg(Type::Int));
        }

        // Record some ops
        let add = rec.record_op(OpCode::IntAdd, &[inputs[0], inputs[1]]);

        // Guard with all 12 inputs + 1 computed value as fail_args (13 total)
        let mut fail_args: Vec<OpRef> = inputs.clone();
        fail_args.push(add);

        let descr = make_fail_descr(0);
        let guard = rec.record_guard(OpCode::GuardTrue, &[add], Some(descr));

        rec.close_loop(&inputs);
        rec.materialize_into_ops();
        rec.set_op_fail_args(guard, &fail_args);
        let trace = rec.get_trace();

        // Find the guard
        let guard_op = trace.iter_guards().next().unwrap();
        assert_eq!(guard_op.pos().get(), guard);
        let fa = guard_op.guard_fail_args().unwrap();
        assert_eq!(fa.len(), 13);

        // Verify all fail_args match what we specified
        for (i, &expected) in fail_args.iter().enumerate() {
            assert_eq!(fa[i].to_opref(), expected);
        }
    }

    #[test]
    fn test_recorder_guard_descr_index_survives() {
        // Each guard's descriptor index must survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let d0 = make_fail_descr(100);
        let d1 = make_fail_descr(200);
        let d2 = make_fail_descr(300);

        rec.record_guard(OpCode::GuardTrue, &[i0], Some(d0));
        rec.record_guard(OpCode::GuardFalse, &[i0], Some(d1));
        rec.record_guard(OpCode::GuardNoException, &[], Some(d2));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].getdescr().unwrap().index(), 100);
        assert_eq!(guards[1].getdescr().unwrap().index(), 200);
        assert_eq!(guards[2].getdescr().unwrap().index(), 300);
    }

    #[test]
    fn test_recorder_ops_interleaved_with_guards() {
        // Interleaved ops and guards: verify ordering is correct.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let d0 = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[add], Some(d0));
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        let d1 = make_fail_descr(1);
        rec.record_guard(OpCode::GuardFalse, &[sub], Some(d1));
        let mul = rec.record_op(OpCode::IntMul, &[sub, i1]);

        rec.close_loop(&[mul, i1]);
        let trace = rec.get_trace();

        let expected = vec![
            OpCode::IntAdd,
            OpCode::GuardTrue,
            OpCode::IntSub,
            OpCode::GuardFalse,
            OpCode::IntMul,
            OpCode::Jump,
        ];
        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_recorder_finish_with_descr() {
        // finish() records a FINISH op with the given descriptor.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(999);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());

        let finish = trace.ops.last().unwrap();
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.getdescr().unwrap().index(), 999);
    }

    #[test]
    fn test_recorder_no_ops_just_inputargs_and_jump() {
        // Minimal trace: input args directly jump back.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        rec.close_loop(&[i0, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_ops(), 1); // Only Jump
        assert_eq!(trace.ops[0].opcode, OpCode::Jump);
        assert_eq!(trace.ops[0].arg(0).to_opref(), i0);
        assert_eq!(trace.ops[0].arg(1).to_opref(), i1);
    }

    #[test]
    fn test_get_position_and_cut() {
        let mut rec = Trace::with_num_inputs(2);
        let pos0 = rec.get_position();
        assert_eq!(pos0._pos, 0);
        assert_eq!(pos0._count, 2); // 2 inputargs
        assert_eq!(pos0._index, 2);
        assert_eq!(pos0.snapshot_data_len, 0);
        assert_eq!(pos0.snapshot_array_data_len, 0);

        let _a = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        let pos1 = rec.get_position();
        assert!(pos1._pos > pos0._pos, "byte cursor advances");
        assert_eq!(pos1._count, 3);
        assert_eq!(pos1._index, 3);

        let _b = rec.record_op(OpCode::IntSub, &[iarg(0), iarg(1)]);
        let _c = rec.record_op(OpCode::IntMul, &[iarg(0), iarg(1)]);
        assert_eq!(rec.num_ops(), 3);

        // Cut back to pos1 — should discard IntSub and IntMul
        rec.cut(pos1);
        assert_eq!(rec.num_ops(), 1);
        assert_eq!(rec.get_position(), pos1);

        // Can record more ops after cut
        let d = rec.record_op(OpCode::IntNeg, &[iarg(0)]);
        assert_eq!(d, iop(3)); // continues from pos1._count
        assert_eq!(rec.num_ops(), 2);

        // Cut back to pos0 — should discard everything
        rec.cut(pos0);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_get_position_tracks_count_and_index_separately() {
        let mut rec = Trace::with_num_inputs(1);
        let i0 = iarg(0);

        let _add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let pos_after_add = rec.get_position();
        assert_eq!(pos_after_add._count, 2);
        assert_eq!(pos_after_add._index, 2);

        let descr = make_fail_descr(1);
        rec.record_guard(OpCode::GuardTrue, &[iop(1)], Some(descr));
        let pos_after_guard = rec.get_position();
        assert!(pos_after_guard._pos > pos_after_add._pos);
        assert_eq!(pos_after_guard._count, 3);
        assert_eq!(pos_after_guard._index, 2);
    }
}
